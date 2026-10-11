"""CM1's per-process budgets and GM1's vorticity budget on the Cartesian lane.

Where each term is taken
------------------------
CM1 fills its ``udiag``/``vdiag``/``wdiag``/``tdiag``/``qdiag`` slots inside
the operators themselves, on the step that writes output and in the final RK
stage.  GM1's operators are fused CUDA kernels, so this stage takes every term
at CM1's own placement by *replaying the model's own kernel* on budget-owned
scratch with the other directions' inputs zeroed:

=====================  =================================  ==========================================
term                   CM1 placement                      GM1 replay
=====================  =================================  ==========================================
``*_rdamp``            ``solve1.F:489-529``               the tile's Rayleigh output, after the kernel
``ptb_diss``           ``solve1.F:920-996``               held-theta increment across idiss
``ptb_rad``            ``solve1.F:1007-1048``             held-theta increment across rterm
``ub/vb_lspg``         ``solve1.F:590-612`` (no slot)     the lspgrad=1 profile words; GM1 EXTENSION
``*_hturb``/``vturb``  ``turbtend.F`` turbu/v/w/s          SGS kernels with the vertical (resp.
                                                          horizontal) stresses and kmv/khv zeroed
``ub/vb_cor``          ``solve2.F:694-714``               rotation kernel into zero tendency
``*_hadv``/``vadv``    ``adv.F`` doingubud/sbud            advection kernel with rrw (resp. rru/rrv)
                                                          zeroed and the directional divergence
``*_hidiff``/``vidiff`` ``adv.F`` gethidiff/getvidiff      ``tot - ndiff``, sixth-order flux
``wb_buoy``            ``solve2.F:896-920``               buoyancy kernel into zero tendency
``*_pgrad``            ``solve2.F:925-956,1196-1245``     ``(x3d-xa)/dt`` minus the saved tendency
``ptb_div``            ``solve2.F:1331-1340``             ``-t11*thterm`` of the final stage
``ptb_mp``/``qvb_mp``  ``mp_driver.F:151-170,1670-1690``   ``(after-before)/dt`` around pdefq+physics
=====================  =================================  ==========================================

For every kernel the replay is a separate launch of the same compiled image,
so each term is the fused operator's own arithmetic with an exact zero in
place of the other direction.  Every launch goes through
:class:`cm1core.cm1_budget_operators.BudgetWriteGuard`, which refuses any
writable argument that is not an array this stage allocated: with budgets on,
no model carrier can receive a word it would not have received with budgets
off.

When
----
CM1 raises ``do*bud`` only on a step that writes output (``cm1.F:3268-3296``).
This stage is *due* on a history-frame step, and additionally on a parcel or
vorticity-frame step when the ``[lagrangian]`` GM1 extensions ask for the
vorticity budget there.  No state is carried from one due step to the next,
so a resumed run needs nothing from its restart.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .cm1_budget_operators import (
    BudgetWriteGuard,
    VorticityGrid,
    FlatMetricLevels,
    calcvort_x,
    calcvort_y,
    calcvort_z,
    curl_budget_term,
    horizontal_divergence,
    implicit_diffusion_split,
    nondiffusive_scalar,
    nondiffusive_u,
    nondiffusive_v,
    nondiffusive_w,
    pressure_gradient_residual,
    process_difference,
    velocity_gradient_decomposition,
    vertical_divergence,
)
from .cm1_budget_terms import (
    BUDGET_PREFIX,
    BudgetFacts,
    TERM_AUTHORITY,
    VORTICITY_AXES,
    VORTICITY_CURL_TERMS,
    VORTICITY_DECOMPOSITION_TERMS,
    budget_defects,
    resolve_terms,
    selected_budgets,
    vorticity_budget_names,
)

#: v2: ``closure`` ``qvb_*`` now measure the step's own qv tendency.  Under v1
#: the microphysics drain had already copied q3d into qa when
#: ``after_microphysics`` read it, so ``qvb_tendency_max_abs`` was always 0.0
#: and ``qvb_residual_max_abs`` was the term sum alone.  Terms are unchanged.
CARTESIAN_BUDGET_SCHEMA = "gpuwm-cm1.cartesian-budgets/v2"
NG = 3
THETA_SPECIES = 0xFFFFFFFF

UNVALIDATED_BUDGETS = (
    "UNVALIDATED against CM1: no CM1 r21.1 run with output_*budget=1 has been "
    "differenced against this lane. Each term is CM1's own expression at CM1's "
    "own placement evaluated by GM1's fused kernel with the other directions "
    "zeroed; closure is checked against the tendency the model took, by value, "
    "in every receipt."
)
UNVALIDATED_VORTICITY = (
    "GM1 EXTENSION, UNVALIDATED against CM1: CM1 r21.1 has no vorticity budget. "
    "Every curl term is misclibs.F calcvort applied to one CM1 momentum budget "
    "term, so the terms sum to the tendency of CM1's own xvort/yvort/zvort to "
    "rounding; the stretching/tilting decomposition is the continuous identity "
    "at scalar points and its difference from the curl of the model's advection "
    "is published as *_advres."
)


def _kernel_sources() -> dict[str, str]:
    from .cuda_advection5 import CUDA_ADVECTION5_SOURCE
    from .cuda_gm1_child_large_step import GM1_CHILD_THETA_PACK_SOURCE
    from .cuda_gm1_child_stretched_momentum import GM1_CHILD_STRETCHED_MOMENTUM_SOURCE
    from .cuda_gm1_child_stretched_scalar import (
        GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE,
        GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE,
    )
    from .cuda_gm1_child_stretched_scalar_turbulence import (
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE,
    )
    from .cuda_gm1_child_stretched_sgs_tke import CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE
    from .cuda_gm1_child_stretched_momentum import GM1_FLAT_STRETCHED_MOMENTUM_SOURCE
    from .cuda_gm1_child_stretched_scalar import (
        GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE,
        GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE,
    )
    from .cuda_gm1_flat_stretched_closure import GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE
    from .cuda_gm1_solve2_rotation import CUDA_GM1_SOLVE2_ROTATION_SOURCE
    from .cuda_gm1_two_part_sgs import two_part_column_additions
    from .cuda_moist_stage_bridge import CUDA_MOIST_STAGE_BRIDGE_SOURCE
    from .cuda_momentum_advection5 import CUDA_MOMENTUM_ADVECTION5_SOURCE
    from .cuda_scalar_moist_rk import CUDA_SCALAR_MOIST_RK_SOURCE

    return {
        "mom5": CUDA_MOMENTUM_ADVECTION5_SOURCE,
        "mom5_stretched": GM1_CHILD_STRETCHED_MOMENTUM_SOURCE,
        "sgs": CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE,
        # sgsmodel=3/4's turbu/turbv column kernels: the audited pair plus
        # turbuz/turbvz's dot2p block, with the same derivation compile_two_part
        # compiles (their signatures do not depend on max_nk).
        "sgs_t2p": two_part_column_additions(CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE),
        "turbs": CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE,
        "bridge": CUDA_MOIST_STAGE_BRIDGE_SOURCE,
        "rotation": CUDA_GM1_SOLVE2_ROTATION_SOURCE,
        "adv5": CUDA_ADVECTION5_SOURCE,
        "adv5_stretched": GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE,
        "rk": CUDA_SCALAR_MOIST_RK_SOURCE,
        "rk_stretched": GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE,
        "theta_pack": GM1_CHILD_THETA_PACK_SOURCE,
        # The stretched Cartesian column is terrain_flag=.false.: these are
        # the images its dynamics and closure launch.
        "mom5_flat": GM1_FLAT_STRETCHED_MOMENTUM_SOURCE,
        "sgs_flat": GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE,
        "adv5_flat": GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE,
        "rk_flat": GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE,
    }


def _blocks(words: int, threads: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((max(1, words) + threads - 1) // threads, 1, 1), (threads, 1, 1)


@dataclass
class BudgetColumn:
    """The vertical words every budget expression reads, as ``xp`` arrays."""

    rdsf_levels: Any   # (nk,)   rdsf(k), k=1..nk; FlatMetricLevels(rdz, mh) when flat
    rds_levels: Any    # (nk+1,) rds(k),  k=0..nk; FlatMetricLevels(rdz, mf) when flat
    mf: Any            # (nk+1,) mf(k),   k=1..nk+1
    zh: Any            # (nk,)
    zf: Any            # (nk+1,)


def budget_column(xp: Any, grid: Any) -> BudgetColumn:
    """The column this run's operators divide by, from its generated levels."""

    np = importlib.import_module("numpy")
    nk = int(grid.nz)
    f32 = np.float32
    if int(getattr(grid, "stretch_z", 0)) == 0:
        rdz = f32(1.0) / f32(grid.dz_m)
        rdsf = np.full((nk,), rdz, dtype=np.float32)
        rds = np.full((nk + 1,), rdz, dtype=np.float32)
        mf = np.ones((nk + 1,), dtype=np.float32)
        zf = (np.arange(nk + 1, dtype=np.float64) * float(grid.dz_m)).astype(np.float32)
        zh = ((np.arange(nk, dtype=np.float64) + 0.5) * float(grid.dz_m)).astype(np.float32)
    else:
        levels = grid.levels

        def cm1(values: Any, count: int) -> Any:
            array = np.asarray(values, dtype=np.float32).reshape(-1)
            # VerticalLevels is CM1-indexed with its k=0 ghost; accept a bare
            # column too so a caller-built profile cannot be off by one silently.
            if array.size == count:
                return array
            if array.size >= count + 1:
                return array[1:count + 1]
            raise ValueError(f"vertical profile has {array.size} words, need {count}")

        mf = cm1(levels.mf, nk + 1)
        zh = cm1(levels.zh, nk)
        zf = cm1(levels.zf, nk + 1)
        # The stretched Cartesian column is flat (terrain_flag=.false.): its
        # vertical differences are ((d*rdz)*mh(1,1,k)) and ((d*rdz)*mf(1,1,k)).
        rdz = f32(1.0) / f32(grid.dz_m)
        mh_full = np.asarray(levels.mh, dtype=np.float32).reshape(-1)
        mf_full = np.asarray(levels.mf, dtype=np.float32).reshape(-1)
        return BudgetColumn(
            FlatMetricLevels(xp.float32(rdz), xp.asarray(np.ascontiguousarray(mh_full[1:nk + 1]))),
            FlatMetricLevels(xp.float32(rdz), xp.asarray(np.ascontiguousarray(mf_full[0:nk + 1]))),
            *(xp.asarray(np.ascontiguousarray(a)) for a in (mf, zh, zf)),
        )
    return BudgetColumn(*(xp.asarray(np.ascontiguousarray(a)) for a in (rdsf, rds, mf, zh, zf)))


@dataclass
class CartesianBudgetStage:
    """One run's budget carriers, replay scratch, schedule and receipts."""

    xp: Any
    allocate: Callable[[str, tuple[int, ...]], Any]
    facts: BudgetFacts
    budgets: tuple[str, ...]
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    bbc: int
    tbc: int
    column: BudgetColumn
    stretched: bool = False
    vorticity_history: bool = False
    parcel_vorticity: bool = False
    vorticity_frames: bool = False
    guard: BudgetWriteGuard = field(init=False)
    fields: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    terms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    active_step: int | None = None
    completed_step: int | None = None
    closure: dict[str, float] = field(default_factory=dict)
    _decomposition: dict[str, Any] = field(default_factory=dict)
    _theta_mark: dict[str, Any] = field(default_factory=dict)
    _tile_spec: Any = None

    def __post_init__(self) -> None:
        self.guard = BudgetWriteGuard(self.xp)
        for family, source in _kernel_sources().items():
            self.guard.register_source(family, source)
        for switch in self.budgets:
            defects = budget_defects(switch, self.facts)
            if defects:
                raise RuntimeError(f"{switch}: {defects[0][1]}")
        self.momentum_needed = bool(
            {"output_ubudget", "output_vbudget", "output_wbudget"} & set(self.budgets)
        ) or self.vorticity_needed
        for switch in ("output_ubudget", "output_vbudget", "output_wbudget",
                       "output_thbudget", "output_qvbudget"):
            wanted = switch in self.budgets or (
                self.vorticity_needed and switch in ("output_ubudget", "output_vbudget", "output_wbudget")
            )
            if wanted:
                self.terms[switch] = resolve_terms(switch, self.facts)
        nk, nj, ni = self.nz, self.ny, self.nx
        shapes = {
            "output_ubudget": (nk, nj, ni + 1), "output_vbudget": (nk, nj + 1, ni),
            "output_wbudget": (nk + 1, nj, ni), "output_thbudget": (nk, nj, ni),
            "output_qvbudget": (nk, nj, ni),
        }
        for switch, names in self.terms.items():
            prefix = BUDGET_PREFIX[switch]
            for term in names:
                self.fields[f"{prefix}_{term}"] = self._own(f"{prefix}_{term}", shapes[switch])
        if self.vorticity_needed:
            for name in vorticity_budget_names(self.facts):
                self.fields[name] = self._own(name, (nk, nj, ni))

    # ------------------------------------------------------------ plumbing
    @property
    def vorticity_needed(self) -> bool:
        return self.vorticity_history or self.parcel_vorticity or self.vorticity_frames

    def _own(self, name: str, shape: tuple[int, ...]) -> Any:
        return self.guard.own(self.allocate(f"budget.{name}", tuple(shape)))

    def _scratch(self, name: str, shape: tuple[int, ...]) -> Any:
        value = self.scratch.get(name)
        if value is None or tuple(value.shape) != tuple(shape):
            value = self._own(f"scratch.{name}", tuple(shape))
            self.scratch[name] = value
        value.fill(self.xp.float32(0.0))
        return value

    @property
    def active(self) -> bool:
        return self.active_step is not None

    def has(self, switch: str, term: str) -> bool:
        return term in self.terms.get(switch, ())

    def _put(self, switch: str, term: str, value: Any) -> None:
        if self.has(switch, term):
            self.xp.copyto(self.fields[f"{BUDGET_PREFIX[switch]}_{term}"], value)

    def begin_step(self, step: int, *, due: bool) -> None:
        self.active_step = int(step) if due else None
        self._theta_mark.clear()
        self._decomposition.clear()

    # ============================================================ solve1
    def begin_tile(self, spec: Any) -> None:
        self._tile_spec = spec

    def _scatter(self, switch: str, term: str, tile_value: Any, stagger: str) -> None:
        if not self.has(switch, term):
            return
        spec = self._tile_spec
        target = self.fields[f"{BUDGET_PREFIX[switch]}_{term}"]
        y0, y1, x0, x1 = spec.owned_rect(stagger)
        self.xp.copyto(
            target[..., y0:y1, x0:x1],
            tile_value[..., y0 - spec.cj0:y1 - spec.cj0, x0 - spec.ci0:x1 - spec.ci0],
        )

    def after_rayleigh(self, tile: Mapping[str, Any]) -> None:
        """``solve1.F:489-529``: the held arrays hold exactly the damping here."""

        for switch, name, stagger in (("output_ubudget", "uten1", "u"),
                                      ("output_vbudget", "vten1", "v"),
                                      ("output_thbudget", "thten1", "mass")):
            self._scatter(switch, "rdamp", tile[name], stagger)
        if self.has("output_wbudget", "rdamp"):
            wt = tile["wten1"]
            full = self._scratch("tile_w_rdamp", (wt.shape[0] + 1, *wt.shape[1:]))
            full[:wt.shape[0]] = wt
            self._scatter("output_wbudget", "rdamp", full, "w")

    def large_scale_pressure_gradient(self, tile: Mapping[str, Any], ul: Any, vl: Any) -> None:
        """``solve1.F:590-612``: the GM1 EXTENSION terms ``ub_lspg``/``vb_lspg``.

        CM1 r21.1's 3-D budget has no slot for lspgrad (param.F:5573-5587),
        so its own u/v budget does not close under lspgrad=1; the ``ub_lspg``
        its domaindiag.F:7466-7480 writes is the ``ulspg`` profile, which
        solve1.F:592-593 leaves at zero for this branch.  The term
        is exactly the per-level word the kernel applied -- ``-ul`` on every u
        face, ``+vl`` on every v face -- so it is published from the profiles
        rather than differenced out of the held tendency.
        """

        xp = self.xp
        for switch, column, stagger, shape in (
            ("output_ubudget", xp.negative(ul), "u", tuple(tile["uten1"].shape)),
            ("output_vbudget", vl, "v", tuple(tile["vten1"].shape)),
        ):
            if not self.has(switch, "lspg"):
                continue
            value = self._scratch(f"tile_{BUDGET_PREFIX[switch]}_lspg", shape)
            xp.copyto(value, xp.broadcast_to(column[:, None, None], shape))
            self._scatter(switch, "lspg", value, stagger)

    def theta_mark(self, tile: Mapping[str, Any], label: str) -> None:
        """``solve1.F:920-926``/``:1007-1013``: save thten1 before the operator."""

        if self.has("output_thbudget", label):
            mark = self._scratch(f"theta_mark_{label}", tuple(tile["thten1"].shape))
            self.xp.copyto(mark, tile["thten1"])
            self._theta_mark[label] = mark

    def theta_delta(self, tile: Mapping[str, Any], label: str) -> None:
        """``solve1.F:990-996``/``:1042-1048``: the operator's increment."""

        if label in self._theta_mark:
            delta = self.xp.subtract(tile["thten1"], self._theta_mark.pop(label))
            self._scatter("output_thbudget", label, delta, "mass")

    def momentum_turbulence(self, closure: Any, stress: Mapping[str, Any], *,
                            ni: int, nj: int, nk: int, two_part: Any = None) -> None:
        """``turbtend.F`` turbu/turbv/turbw's two halves, by zeroed replay.

        ``stress`` is the step's own launch dictionary, so sgsmodel=3/4's
        gamk-scaled ``kmh``/``kmv`` are what the replay reads.  ``two_part`` is
        that step's :class:`TwoPartColumnBinding` (None for sgsmodel=1): u and v
        then replay the ``*_t2p_v1`` column kernels the step ran.  CM1 adds the
        dot2p flux to ``turbz`` (src/turbtend.F:1767-1802, :2424-2459), and
        ``turbz`` is ``ud_vturb``/``vd_vturb`` (:1326-1347, :2092-2114), so the
        flux is replayed inside ``vturb`` and ``hturb`` gets a zero flux
        profile.  Each replay writes its own ``t2pt`` scratch, never the step's
        ``u2pt``/``v2pt`` carriers.
        """

        xp = self.xp
        want = {
            s: self.has(s, "hturb") or self.has(s, "vturb")
            for s in ("output_ubudget", "output_vbudget", "output_wbudget")
        }
        if not any(want.values()):
            return
        if getattr(closure, "flat_compiled", None) is not None:
            self._flat_momentum_turbulence(closure, stress, want, ni=ni, nj=nj, nk=nk, two_part=two_part)
            return
        g, dt = closure.grid, closure.config.dt_s
        profiles = closure.vertical_view.arrays
        kernels = closure.sgs_compiled.sgs_kernels
        from .cuda_sgs_stress_tendency import CUDA_SGS_STRESS_THREADS

        suffix = {
            "ni": ni, "nj": nj, "nk": nk, "dx": g.dx_m, "dz": g.dz_m, "dt": dt,
            "sigma": profiles["sigma"], "sigmaf": profiles["sigmaf"],
            "rds": profiles["rds"], "rdsf": profiles["rdsf"], "zt": g.ztop_m,
        }
        zero = {name: self._scratch(f"sgs_zero_{name}", tuple(stress[name].shape))
                for name in ("t11", "t12", "t13", "t22", "t23", "t33", "kmh", "kmv")}
        plans = (
            ("output_ubudget", 4, "cm1_sgs_turbu_terrain_v1", "u", "uten", (ni + 1) * nj,
             {"t11": "t11", "t12": "t12", "t13": "t13", "kmv": "kmv", "rho": "rho", "rf": "rf",
              "mh": "mh", "mf": "mf", "gz": "gz", "rgz": "rgz", "gzu": "gzu", "gxu": "gxu",
              "u": "u3d", "w": "w3d"},
             ("t11", "t12"), ("t13", "kmv")),
            ("output_vbudget", 5, "cm1_sgs_turbv_terrain_v1", "v", "vten", ni * (nj + 1),
             {"t12": "t12", "t22": "t22", "t23": "t23", "kmv": "kmv", "rho": "rho", "rf": "rf",
              "mh": "mh", "mf": "mf", "gz": "gz", "rgz": "rgz", "gzv": "gzv", "gyv": "gyv",
              "v": "v3d", "w": "w3d"},
             ("t12", "t22"), ("t23", "kmv")),
            ("output_wbudget", 6, "cm1_sgs_turbw_terrain_v1", "w", "wten", ni * nj,
             {"t13": "t13", "t23": "t23", "t33": "t33", "kmh": "kmh", "rho": "rho", "rf": "rf",
              "mh": "mh", "mf": "mf", "gz": "gz", "rgzu": "rgzu", "rgzv": "rgzv", "w": "w3d"},
             ("t13", "t23"), ("t33", "kmh")),
        )
        for switch, index, name, stagger, out_name, words, inputs, horizontal, vertical in plans:
            if not want[switch]:
                continue
            grid, block = _blocks(words, CUDA_SGS_STRESS_THREADS)
            family, kernel = "sgs", kernels[index]
            flux_terms: dict[str, dict[str, Any]] = {}
            if two_part is not None and stagger in ("u", "v"):
                family = "sgs_t2p"
                name = name.replace("_terrain_v1", "_terrain_t2p_v1")
                kernel = two_part.kernels[0 if stagger == "u" else 1]
                flux = two_part.tfu if stagger == "u" else two_part.tfv
                carrier = two_part.u2pt if stagger == "u" else two_part.v2pt
                zero_flux = self._scratch(f"sgs_t2p_zero_{stagger}", tuple(flux.shape))
                t2pt = self._scratch(f"sgs_t2p_{stagger}", tuple(carrier.shape))
                for term, tfw in (("hturb", zero_flux), ("vturb", flux)):
                    flux_terms[term] = {"tfw": tfw, "t2pt": t2pt, "ntwk": int(two_part.ntwk)}
            for term, zeroed in (("hturb", vertical), ("vturb", horizontal)):
                if not self.has(switch, term):
                    continue
                out = self._scratch(f"sgs_{switch}_{term}", tuple(stress[out_name].shape))
                values = {param: (zero[source] if source in zeroed else stress[source])
                          for param, source in inputs.items()}
                values.update(suffix)
                values.update(flux_terms.get(term, {}))
                values[out_name] = out
                self.guard.call(family, name, kernel, grid, block, values,
                                stream=closure.stream)
                self._scatter(switch, term, out, stagger)
        del xp

    def _flat_momentum_turbulence(self, closure: Any, stress: Mapping[str, Any], want: Mapping[str, bool], *,
                                  ni: int, nj: int, nk: int, two_part: Any) -> None:
        """:meth:`momentum_turbulence` on a flat column's turbu/turbv/turbw.

        The flat u and v kernels carry turbuz/turbvz's dot2p block
        themselves, so sgsmodel=1 replays them with ``ntwk=0`` and a
        budget-owned ``t2pt`` the kernel never writes.
        """

        np = importlib.import_module("numpy")
        from .cuda_gm1_flat_stretched_closure import GM1_FLAT_STRETCHED_CLOSURE_THREADS

        g, v = closure.grid, closure.vertical
        f32 = np.float32
        kernels = closure.flat_compiled.kernels
        suffix = {
            "ni": ni, "nj": nj, "nk": nk, "rdx": float(f32(1.0) / f32(g.dx_m)),
            "rdy": float(f32(1.0) / f32(g.dy_m)), "rdz": float(v.rdz_f32), "dt": closure.config.dt_s,
        }
        zero = {name: self._scratch(f"sgs_zero_{name}", tuple(stress[name].shape))
                for name in ("t11", "t12", "t13", "t22", "t23", "t33", "kmh", "kmv")}
        common = {"rho": "rho", "rf": "rf", "mh": "mh", "mf": "mf", "w": "w3d"}
        plans = (
            ("output_ubudget", "gm1_flat_turbu_v1", "u", "uten", (ni + 1) * nj,
             {"t11": "t11", "t12": "t12", "t13": "t13", "kmv": "kmv", "u": "u3d", **common},
             ("t11", "t12"), ("t13", "kmv")),
            ("output_vbudget", "gm1_flat_turbv_v1", "v", "vten", ni * (nj + 1),
             {"t12": "t12", "t22": "t22", "t23": "t23", "kmv": "kmv", "v": "v3d", **common},
             ("t12", "t22"), ("t23", "kmv")),
            ("output_wbudget", "gm1_flat_turbw_v1", "w", "wten", ni * nj,
             {"t13": "t13", "t23": "t23", "t33": "t33", "kmh": "kmh", **common},
             ("t13", "t23"), ("t33", "kmh")),
        )
        for switch, name, stagger, out_name, words, inputs, horizontal, vertical in plans:
            if not want[switch]:
                continue
            grid, block = _blocks(words, GM1_FLAT_STRETCHED_CLOSURE_THREADS)
            flux_terms: dict[str, dict[str, Any]] = {}
            if stagger in ("u", "v"):
                carrier_shape = (nk, nj, ni + 1) if stagger == "u" else (nk, nj + 1, ni)
                if two_part is not None:
                    flux = two_part.tfu if stagger == "u" else two_part.tfv
                    carrier = two_part.u2pt if stagger == "u" else two_part.v2pt
                    zero_flux = self._scratch(f"sgs_t2p_zero_{stagger}", tuple(flux.shape))
                    t2pt = self._scratch(f"sgs_t2p_{stagger}", tuple(carrier.shape))
                    for term, tfw in (("hturb", zero_flux), ("vturb", flux)):
                        flux_terms[term] = {"tfw": tfw, "t2pt": t2pt, "ntwk": int(two_part.ntwk)}
                else:
                    unused = self._scratch(f"sgs_flat_t2pt_{stagger}", carrier_shape)
                    for term in ("hturb", "vturb"):
                        flux_terms[term] = {"tfw": v.mh, "t2pt": unused, "ntwk": 0}
            for term, zeroed in (("hturb", vertical), ("vturb", horizontal)):
                if not self.has(switch, term):
                    continue
                out = self._scratch(f"sgs_{switch}_{term}", tuple(stress[out_name].shape))
                values = {param: (zero[source] if source in zeroed else stress[source])
                          for param, source in inputs.items()}
                values.update(suffix)
                if stagger in ("u", "v"):
                    values["mhc"] = v.mh
                values.update(flux_terms.get(term, {}))
                values[out_name] = out
                self.guard.call("sgs_flat", name, kernels[name], grid, block, values,
                                stream=closure.stream)
                self._scatter(switch, term, out, stagger)

    def scalar_turbulence(self, closure: Any, *, species: int, kernel: Any,
                          arguments: Mapping[str, Any], flat: bool = False) -> None:
        """``turbtend.F:395-420`` for theta and qv, by zeroed replay.

        ``arguments`` is the fused ``gm1_turbs_implicit_apply_stretched_v1``
        launch, by parameter name -- or, on a flat column,
        ``gm1_flat_turbs_implicit_apply_v1``'s, which has no terrain fluxes.
        """

        switch = "output_thbudget" if species == THETA_SPECIES else (
            "output_qvbudget" if species == 0 else None)
        if switch is None or not (self.has(switch, "hturb") or self.has(switch, "vturb")):
            return
        from .cuda_gm1_scalar_turbulence import CUDA_GM1_SCALAR_TURBULENCE_THREADS

        ni, nj = int(arguments["ni"]), int(arguments["nj"])
        grid, block = _blocks(ni * nj, CUDA_GM1_SCALAR_TURBULENCE_THREADS)
        name = "gm1_flat_turbs_implicit_apply_v1" if flat else "gm1_turbs_implicit_apply_stretched_v1"
        family = "sgs_flat" if flat else "turbs"
        fluxes = ("fx", "fy") if flat else ("fx", "fy", "fxw", "fyw")
        for term in ("hturb", "vturb"):
            if not self.has(switch, term):
                continue
            values = dict(arguments)
            values["cp"] = self._scratch("turbs_cp", tuple(arguments["cp"].shape))
            values["dp"] = self._scratch("turbs_dp", tuple(arguments["dp"].shape))
            out = self._scratch(f"turbs_{switch}_{term}", tuple(arguments["target"].shape))
            values["target"] = out
            if term == "hturb":
                values["khv"] = self._scratch("turbs_zero_khv", tuple(arguments["khv"].shape))
                values["iflux"] = 0
            else:
                for flux in fluxes:
                    values[flux] = self._scratch(f"turbs_zero_{flux}", tuple(arguments[flux].shape))
            self.guard.call(family, name, kernel, grid, block, values, stream=closure.stream)
            self._scatter(switch, term, out, "mass")

    # ============================================================ solve2, nrk=3
    def pre_acoustic(self, dynamics: Any) -> None:
        """Everything CM1 reads before the final-stage acoustic steps.

        Called after the momentum tendency is assembled (advection, rotation,
        buoyancy, inflow nudging) and before the pre-acoustic scratch is
        released.
        """

        if not self.momentum_needed:
            return
        xp = self.xp
        g = dynamics.geometry
        ni, nj, nk = g.nx, g.ny, g.nz
        a, m = dynamics.arrays, dynamics.metrics
        stream = dynamics.stream
        # solve2.F:925-956: the explicit tendency, before the acoustic steps.
        for key, name, sl in (("u", "uten", (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1))),
                              ("v", "vten", (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni))),
                              ("w", "wten", (slice(1, nk + 2), slice(NG, NG + nj), slice(NG, NG + ni)))):
            saved = self._scratch(f"saved_{key}", tuple(a[name][sl].shape))
            xp.copyto(saved, a[name][sl])
        flat = bool(dynamics.stretched)
        family = "mom5_flat" if flat else "mom5"
        compiled = dynamics.stretched_compiled["momentum"] if flat else dynamics.momentum_advection_compiled
        _, _, _, advu, advv, advw = compiled.kernels
        col = self.column
        rdsf_levels, rds_levels = col.rdsf_levels, col.rds_levels
        profiles = None
        if flat:
            # adv.F's terrain_flag=.false. advu/advv/advw: ((d*rdz)*mh(1,1,k)),
            # ((d*rdz)*mf(1,1,k)), scaled by rr0(1,1,k) and rrf0(1,1,k).
            vertical = dynamics._stretched_vertical()
            rdz = xp.float32(vertical.rdz_f32)
            profiles = {"mh": vertical.mh, "mf": vertical.mf, "rdz": float(vertical.rdz_f32)}
            rdsf_levels = FlatMetricLevels(rdz, vertical.mh[1:nk + 1])
            rds_levels = FlatMetricLevels(rdz, vertical.mf[0:nk + 1])
        divh = horizontal_divergence(xp, a["rru"], a["rrv"], ni=ni, nj=nj, nk=nk, ng=NG,
                                     dx=g.dx_m, dy=g.dy_m)
        divv = vertical_divergence(xp, a["rrw"], rdsf_levels, ni=ni, nj=nj, nk=nk, ng=NG)
        divh = self.guard.own(divh)
        divv = self.guard.own(divv)
        zero = {name: self._scratch(f"zero_{name}", tuple(a[name].shape)) for name in ("rru", "rrv", "rrw")}
        from .cuda_boundary_families import bound_lateral_sides
        from .cuda_momentum_advection5 import CUDA_MOMENTUM_ADVECTION5_THREADS as T
        from .cuda_momentum_advection5 import momentum_advection5_counts, momentum_open_boundary_code

        # adv.F's i1..i2/j1..j2 and doitw..doitn: the replay runs the family the
        # model's own advection ran, read from the same bound fills.
        lateral_sides = bound_lateral_sides(dynamics.open_compiled)
        obc = momentum_open_boundary_code(lateral_sides)
        periodic_x, periodic_y = lateral_sides[0] == 1, lateral_sides[2] == 1
        u_words, v_words, w_words = momentum_advection5_counts(ni, nj, nk, lateral_sides)

        def replay(which: str, flux: Mapping[str, Any], div: Any, label: str) -> Any:
            carrier = {"u": "u3d", "v": "v3d", "w": "w3d"}[which]
            out = self._scratch(f"adv_{which}_{label}", tuple(a[carrier].shape))
            values: dict[str, Any] = {
                "rru": flux["rru"], "rrv": flux["rrv"], "rrw": flux["rrw"], "divx": div,
                "c2a": m["c2"], "ni": ni, "nj": nj, "nk": nk, "ng": NG, "dx": g.dx_m, "dy": g.dy_m,
                "obc": obc,
            }
            if profiles:
                values["rdz"] = profiles["rdz"]
            else:
                values["dz"] = g.dz_m
            if which == "u":
                if profiles:
                    values.update(u=a["u3d"], rr0=m["rr0"], vh=a["vh"], mh=profiles["mh"], uten=out)
                else:
                    values.update(u=a["u3d"], rho0=m["rho0_3d"], gzu=m["gzu"], vh=a["vh"], uten=out)
                words, kernel, name = u_words, advu, "cm1_mom5_advu_v1"
            elif which == "v":
                if profiles:
                    values.update(v=a["v3d"], rr0=m["rr0"], uh=a["uh"], mh=profiles["mh"], vten=out)
                else:
                    values.update(v=a["v3d"], rho0=m["rho0_3d"], gzv=m["gzv"], uh=a["uh"], vten=out)
                words, kernel, name = v_words, advv, "cm1_mom5_advv_v1"
            else:
                if profiles:
                    values.update(w=a["w3d"], rrf0=m["rrf0"], uh=a["uh"], vh=a["vh"], mf=profiles["mf"], wten=out)
                else:
                    values.update(w=a["w3d"], rrf0=m["rrf0"], gz=m["gz"], uh=a["uh"], vh=a["vh"], wten=out)
                words, kernel, name = w_words, advw, "cm1_mom5_advw_v1"
            grid, block = _blocks(words, T)
            self.guard.call(family, name, kernel, grid, block, values, stream=stream)
            return out

        horizontal_flux = {"rru": a["rru"], "rrv": a["rrv"], "rrw": zero["rrw"]}
        vertical_flux = {"rru": zero["rru"], "rrv": zero["rrv"], "rrw": a["rrw"]}
        half = xp.float32(0.5)
        # The kernel scales: gzu/(0.5*(rho0(i-1)+rho0(i))), gzv/(...), rrf0*gz.
        rho0 = m["rho0_3d"]
        scale_u = xp.zeros((nk, nj, ni + 1), dtype=xp.float32)
        scale_v = xp.zeros((nk, nj + 1, ni), dtype=xp.float32)
        # Faces i1..i2 and rows j1..j2 (adv.F:968-978, 1913-1923).
        f0, f1 = (0, ni) if periodic_x else (1, ni - 1)
        g0, g1 = (0, nj) if periodic_y else (1, nj - 1)
        if flat:
            # The flat advu/advv/advw scale by the column rr0(1,1,k)/rrf0(1,1,k).
            rr0_column = m["rr0"][:, :1, :1]
            scale_u[..., f0:f1 + 1] = rr0_column
            scale_v[:, g0:g1 + 1] = rr0_column
            scale_w = xp.ascontiguousarray(xp.broadcast_to(m["rrf0"][:, :1, :1], m["rrf0"].shape))
        else:
            # rho0 either side of a face, read across a periodic side as the kernel does.
            rho_x = xp.pad(rho0, ((0, 0), (0, 0), (1, 1)), mode="wrap" if periodic_x else "edge")
            rho_y = xp.pad(rho0, ((0, 0), (1, 1), (0, 0)), mode="wrap" if periodic_y else "edge")
            scale_u[..., f0:f1 + 1] = xp.divide(m["gzu"][:, f0:f1 + 1], xp.multiply(half, xp.add(rho_x[..., f0:f1 + 1], rho_x[..., f0 + 1:f1 + 2])))
            scale_v[:, g0:g1 + 1] = xp.divide(m["gzv"][g0:g1 + 1], xp.multiply(half, xp.add(rho_y[:, g0:g1 + 1], rho_y[:, g0 + 1:g1 + 2])))
            scale_w = xp.multiply(m["rrf0"], m["gz"])
        for which, switch, carrier, sl, nondiff in (
            ("u", "output_ubudget", "u3d", (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1)), "u"),
            ("v", "output_vbudget", "v3d", (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni)), "v"),
            ("w", "output_wbudget", "w3d", (slice(1, nk + 2), slice(NG, NG + nj), slice(NG, NG + ni)), "w"),
        ):
            if switch not in self.terms:
                continue
            htot = replay(which, horizontal_flux, divh, "h")[sl]
            vtot = replay(which, vertical_flux, divv, "v")[sl]
            if self.has(switch, "hidiff") or self.has(switch, "vidiff"):
                if nondiff == "u":
                    hnd, vnd = nondiffusive_u(xp, u=a["u3d"], rru=a["rru"], rrv=a["rrv"], rrw=a["rrw"],
                                              c2=m["c2"], divh=divh, divv=divv, scale=scale_u,
                                              rdsf_levels=rdsf_levels, ni=ni, nj=nj, nk=nk, ng=NG,
                                              dx=g.dx_m, dy=g.dy_m, periodic_x=periodic_x)
                elif nondiff == "v":
                    hnd, vnd = nondiffusive_v(xp, v=a["v3d"], rru=a["rru"], rrv=a["rrv"], rrw=a["rrw"],
                                              c2=m["c2"], divh=divh, divv=divv, scale=scale_v,
                                              rdsf_levels=rdsf_levels, ni=ni, nj=nj, nk=nk, ng=NG,
                                              dx=g.dx_m, dy=g.dy_m, periodic_y=periodic_y)
                else:
                    hnd, vnd = nondiffusive_w(xp, w=a["w3d"], rru=a["rru"], rrv=a["rrv"], rrw=a["rrw"],
                                              c1=m["c1"], c2=m["c2"], divh=divh, divv=divv,
                                              scale=scale_w, rds_levels=rds_levels,
                                              ni=ni, nj=nj, nk=nk, ng=NG, dx=g.dx_m, dy=g.dy_m)
                hadv, hidiff = implicit_diffusion_split(xp, htot, hnd)
                vadv, vidiff = implicit_diffusion_split(xp, vtot, vnd)
                self._put(switch, "hadv", hadv)
                self._put(switch, "hidiff", hidiff)
                self._put(switch, "vadv", vadv)
                self._put(switch, "vidiff", vidiff)
            else:
                self._put(switch, "hadv", htot)
                self._put(switch, "vadv", vtot)
        # solve2.F:694-714: the rotation kernel into a zero tendency.
        binding = getattr(dynamics, "rotation_bindings", {}).get(3)
        if binding is not None and (self.has("output_ubudget", "cor") or self.has("output_vbudget", "cor")):
            from .cuda_gm1_solve2_rotation import CUDA_GM1_SOLVE2_ROTATION_THREADS as RT

            ut = self._scratch("cor_u", tuple(a["uten"].shape))
            vt = self._scratch("cor_v", tuple(a["vten"].shape))
            selection = binding.receipt["selection"]
            scalars = binding.receipt["kernel_scalars"]
            grid, block = _blocks(nk * (nj + 1) * (ni + 1), RT)
            self.guard.call("rotation", "gm1_solve2_rotation_f32_v1", binding.compiled.kernel, grid, block, {
                "u": a["u3d"], "v": a["v3d"], "uten": ut, "vten": vt, "f2d": binding.f2d,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG,
                "beta_plane": selection["beta_plane"], "axisymmetric": selection["axisymmetric"],
                "fcor": scalars["fcor_f32"], "umove": scalars["u_move_f32"], "vmove": scalars["v_move_f32"],
            }, stream=stream)
            self._put("output_ubudget", "cor", ut[1:nk + 1, NG:NG + nj, NG:NG + ni + 1])
            self._put("output_vbudget", "cor", vt[1:nk + 1, NG:NG + nj + 1, NG:NG + ni])
        # solve2.F:896-920: buoyancy into a zero tendency.
        if self.has("output_wbudget", "buoy"):
            from .cuda_moist_stage_bridge import CUDA_MOIST_STAGE_BRIDGE_THREADS as BT

            bu = self._scratch("buoy_u", tuple(a["uten"].shape))
            bv = self._scratch("buoy_v", tuple(a["vten"].shape))
            bw = self._scratch("buoy_w", tuple(a["wten"].shape))
            total = (nk - 1) * nj * ni + nk * nj * (ni - 1) + nk * (nj - 1) * ni
            grid, block = _blocks(total, BT)
            self.guard.call("bridge", "cm1_msb_buoyancy_v1", dynamics.moist_compiled.kernels[2], grid, block, {
                "dum8": dynamics.moist_bridge.dum8, "c1": m["c1"], "c2": m["c2"], "gxu": m["gxu"],
                "gyv": m["gyv"], "uten": bu, "vten": bv, "wten": bw,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG,
            }, stream=stream)
            self._put("output_wbudget", "buoy", bw[1:nk + 2, NG:NG + nj, NG:NG + ni])
        if self.vorticity_needed:
            self._decompose(dynamics)

    def _grid(self) -> VorticityGrid:
        return VorticityGrid(ni=self.nx, nj=self.ny, nk=self.nz, dx=self.dx, dy=self.dy,
                             dz=self.dz, mf=self.column.mf, bbc=self.bbc, tbc=self.tbc)

    def _decompose(self, dynamics: Any) -> None:
        """Stretching/tilting/divergence/advection from the winds advection used."""

        xp, g = self.xp, dynamics.geometry
        ni, nj, nk = g.nx, g.ny, g.nz
        a = dynamics.arrays
        u = a["u3d"][1:nk + 1, NG:NG + nj, NG:NG + ni + 1]
        v = a["v3d"][1:nk + 1, NG:NG + nj + 1, NG:NG + ni]
        w = a["w3d"][1:nk + 2, NG:NG + nj, NG:NG + ni]
        grid = self._grid()
        cfg = dynamics.config
        xv = calcvort_x(xp, v, w, grid, vmove=float(cfg.v_move_m_s))
        yv = calcvort_y(xp, u, w, grid, umove=float(cfg.u_move_m_s))
        zv = calcvort_z(xp, u, v, grid)
        self._decomposition = velocity_gradient_decomposition(
            xp, u=u, v=v, w=w, xvort=xv, yvort=yv, zvort=zv,
            zh=self.column.zh, zf=self.column.zf, dx=self.dx, dy=self.dy,
        )

    def after_acoustic(self, dynamics: Any) -> None:
        """``solve2.F:1196-1245``: the pressure-gradient term as a residual."""

        if not self.momentum_needed:
            return
        xp, g = self.xp, dynamics.geometry
        ni, nj, nk = g.nx, g.ny, g.nz
        a, dt = dynamics.arrays, float(dynamics.config.dt_s)
        for key, switch, cur, base, sl in (
            ("u", "output_ubudget", "u3d", "ua", (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1))),
            ("v", "output_vbudget", "v3d", "va", (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni))),
            ("w", "output_wbudget", "w3d", "wa", (slice(1, nk + 2), slice(NG, NG + nj), slice(NG, NG + ni))),
        ):
            if switch not in self.terms:
                continue
            saved = self.scratch[f"saved_{key}"]
            pgrad = pressure_gradient_residual(xp, a[cur][sl], a[base][sl], saved, dt)
            if key == "w":
                # solve2.F:1240 loops k=2..nk; the two rigid faces keep CM1's zero.
                pgrad[0] = xp.float32(0.0)
                pgrad[nk] = xp.float32(0.0)
            self._put(switch, "pgrad", pgrad)
            explicit = xp.zeros_like(saved)
            for term in self.terms[switch]:
                if term != "pgrad":
                    explicit = xp.add(explicit, self.fields[f"{BUDGET_PREFIX[switch]}_{term}"])
            residual = xp.subtract(saved, explicit)
            if key == "w":
                residual = residual[1:nk]
            self.closure[f"{BUDGET_PREFIX[switch]}_explicit_residual_max_abs"] = float(xp.abs(residual).max())
            self.closure[f"{BUDGET_PREFIX[switch]}_explicit_tendency_max_abs"] = float(xp.abs(saved).max())

    # ============================================================ scalars, nrk=3
    def before_scalar(self, dynamics: Any) -> None:
        """The final-stage theta and qv advection, replayed on the stage's inputs."""

        if "output_thbudget" not in self.terms and "output_qvbudget" not in self.terms:
            return
        xp, g = self.xp, dynamics.geometry
        ni, nj, nk = g.nx, g.ny, g.nz
        a, m, f = dynamics.arrays, dynamics.metrics, dynamics.fields
        stream = dynamics.stream
        bridge = dynamics.moist_bridge
        if self.has("output_thbudget", "div"):
            # solve2.F:1337: tdiag(td_div) = -t11*thterm.
            self._put("output_thbudget", "div", xp.negative(bridge.theta_divergence_term))
        from .cuda_advection5 import (
            CUDA_ADVECTION5_THREADS,
            DEFAULT_ADVECTION5_SELECTION,
            scalar_advection_uses_weno,
        )
        from .cuda_scalar_moist_rk import (
            CUDA_SCALAR_MOIST_RK_THREADS,
            moisture_weps_f64,
            scalar_moist_rk_stage_timestep_f32,
            theta_weps_f64,
        )

        stretched = bool(dynamics.stretched)
        col = self.column
        rdsf_levels = col.rdsf_levels
        mh_profile = None
        rdz = None
        if stretched:
            # adv.F:562-572,681-695 and pdef.F's flat statements.
            vertical = dynamics._stretched_vertical()
            mh_profile = vertical.mh
            rdz = float(vertical.rdz_f32)
            rdsf_levels = FlatMetricLevels(xp.float32(rdz), vertical.mh[1:nk + 1])
        divh = self.guard.own(horizontal_divergence(xp, a["rru"], a["rrv"], ni=ni, nj=nj, nk=nk,
                                                     ng=NG, dx=g.dx_m, dy=g.dy_m))
        divv = self.guard.own(vertical_divergence(xp, a["rrw"], rdsf_levels, ni=ni, nj=nj, nk=nk, ng=NG))
        padded_shape = (nk + 2, nj + 2 * NG, ni + 2 * NG)
        words = nk * nj * ni
        grid_s, block_s = _blocks(words, CUDA_SCALAR_MOIST_RK_THREADS)
        compiled = dynamics.scalar_compiled
        advection = dynamics.scalar_advection_compiled
        open_compiled = dynamics.open_compiled
        # The model's own lateral family: adv.F:214-340 and pdef.F:49-166's
        # edge treatment, exactly as the scalar stage applied it.
        from .cuda_advection5 import scalar_open_boundary_code
        from .cuda_boundary_families import bound_lateral_sides

        obc = scalar_open_boundary_code(bound_lateral_sides(open_compiled))
        adv_family = "adv5_flat" if stretched else "adv5"
        rk_family = "rk_flat" if stretched else "rk"

        def fill_halos(value: Any) -> None:
            grid, block = _blocks(int(value.size), CUDA_SCALAR_MOIST_RK_THREADS)
            for kernel in open_compiled.kernels[5:7]:
                self.guard.call_positional(
                    kernel, grid, block,
                    (value, xp.uint32(ni), xp.uint32(nj), xp.uint32(nk), xp.uint32(NG), xp.uint32(0)),
                    writes=(0,), stream=stream,
                )

        def tendency(padded: Any, fx: Any, fy: Any, fz: Any, div: Any, label: str,
                     rru: Any, rrv: Any) -> Any:
            out = self._scratch(f"scalar_tend_{label}", (nk, nj, ni))
            grid, block = _blocks(words, CUDA_ADVECTION5_THREADS)
            values: dict[str, Any] = {
                "a": padded, "fx": fx, "fy": fy, "fz": fz, "divx": div, "rr0": m["rr0"],
                "tendency": out, "ni": ni, "nj": nj, "nk": nk, "ng": NG,
                "dx": g.dx_m, "dy": g.dy_m, "rru": rru, "rrv": rrv, "obc": obc,
            }
            if stretched:
                values.update(mh=mh_profile, rdz=rdz)
            else:
                values.update(gz=m["gz"], dz=g.dz_m)
            self.guard.call(adv_family, "cm1_adv5_scalar_tendency_v1", advection.kernels[6],
                            grid, block, values, stream=stream)
            return out

        if "output_thbudget" in self.terms:
            padded = self._scratch("theta_padded", padded_shape)
            self.guard.call("theta_pack", "gm1_child_rk_pack_theta_base3d_f32_v1", compiled.kernels[0],
                            grid_s, block_s, {"th0": f["th0"], "th": f["th3d"], "p": padded,
                                              "ni": ni, "nj": nj, "nk": nk, "ng": NG}, stream=stream)
            fill_halos(padded)
            use_weno = scalar_advection_uses_weno(3, wflag=True, selection=DEFAULT_ADVECTION5_SELECTION)
            fx = self._scratch("theta_fx", tuple(a["rru"].shape))
            fy = self._scratch("theta_fy", tuple(a["rrv"].shape))
            fz = self._scratch("theta_fz", tuple(a["rrw"].shape))
            weps = theta_weps_f64()
            grid, block = _blocks(nk * (nj + 1) * (ni + 1), CUDA_ADVECTION5_THREADS)
            self.guard.call(adv_family, "cm1_adv5_scalar_horizontal_v1", advection.kernels[4], grid, block, {
                "a": padded, "rru": a["rru"], "rrv": a["rrv"], "fx": fx, "fy": fy,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG, "use_weno": int(use_weno), "weps": weps,
            }, stream=stream)
            grid, block = _blocks((nk - 1) * nj * ni, CUDA_ADVECTION5_THREADS)
            self.guard.call(adv_family, "cm1_adv5_scalar_vertical_v1", advection.kernels[5], grid, block, {
                "a": padded, "rrw": a["rrw"], "c1": m["c1"], "c2": m["c2"], "fz": fz,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG, "use_weno": int(use_weno), "weps": weps,
            }, stream=stream)
            zfx = self._scratch("theta_zero_fx", tuple(fx.shape))
            zfy = self._scratch("theta_zero_fy", tuple(fy.shape))
            zfz = self._scratch("theta_zero_fz", tuple(fz.shape))
            # The inflow edits (adv.F:214-340) are horizontal terms: the
            # vertical split sees zero mass fluxes, so it takes none of them.
            htot = tendency(padded, fx, fy, zfz, divh, "th_h", a["rru"], a["rrv"])
            vtot = tendency(padded, zfx, zfy, fz, divv, "th_v", zfx, zfy)
            self._split_scalar("output_thbudget", padded, htot, vtot, divh, divv, a, m, rdsf_levels, g)
        if "output_qvbudget" in self.terms:
            padded = self._scratch("qv_padded", padded_shape)
            self.guard.call(rk_family, "cm1_rk_pack_scalar_v1", compiled.kernels[1], grid_s, block_s,
                            {"s": f["q3d"][0], "p": padded, "ni": ni, "nj": nj, "nk": nk, "ng": NG},
                            stream=stream)
            fill_halos(padded)
            qmag = getattr(getattr(dynamics.config, "profile", None), "qmag", None)
            weps = moisture_weps_f64(0) if (stretched or qmag is None) else moisture_weps_f64(0, qmag)
            dttmp = float(scalar_moist_rk_stage_timestep_f32(float(dynamics.config.dt_s), 3))
            fx = self._scratch("qv_fx", tuple(a["rru"].shape))
            fy = self._scratch("qv_fy", tuple(a["rrv"].shape))
            fz = self._scratch("qv_fz", tuple(a["rrw"].shape))
            ax = self._scratch("qv_ax", (nk, nj, ni))
            ay = self._scratch("qv_ay", (nk, nj, ni))
            az = self._scratch("qv_az", (nk, nj, ni))
            candidate = self._scratch("qv_candidate", (nk, nj, ni))
            k = compiled.kernels
            grid_h, block = _blocks(nk * (nj + 1) * (ni + 1), CUDA_SCALAR_MOIST_RK_THREADS)
            grid_v, _ = _blocks((nk - 1) * nj * ni, CUDA_SCALAR_MOIST_RK_THREADS)
            self.guard.call(rk_family, "cm1_rk_pdef_horizontal_flux_v1", k[4], grid_h, block, {
                "s": padded, "rru": a["rru"], "rrv": a["rrv"], "fx": fx, "fy": fy,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG, "weps": weps}, stream=stream)
            self.guard.call(rk_family, "cm1_rk_pdef_vertical_flux_v1", k[5], grid_v, block, {
                "s": padded, "rrw": a["rrw"], "c1": m["c1"], "c2": m["c2"], "fz": fz,
                "ni": ni, "nj": nj, "nk": nk, "ng": NG, "pdefweno": 0, "weps": weps}, stream=stream)
            directional = {"fx": fx, "fy": fy, "fz": fz, "ax": ax, "ay": ay, "az": az,
                           "ni": ni, "nj": nj, "nk": nk, "ng": NG, "dx": g.dx_m, "dy": g.dy_m,
                           "s": padded, "rru": a["rru"], "rrv": a["rrv"], "obc": obc}
            if stretched:
                directional.update(mh=mh_profile, rdz=rdz)
            else:
                directional["dz"] = g.dz_m
            self.guard.call(rk_family, "cm1_rk_pdef_directional_v1", k[6], grid_s, block_s, directional,
                            stream=stream)
            qa = f["qa"][0]
            for cand, corr, axis, flux, spacing, target in (
                (7, 8, "x", a["rru"], g.dx_m, ax), (9, 10, "y", a["rrv"], g.dy_m, ay),
                (11, 12, "z", a["rrw"], g.dz_m, az),
            ):
                mflux = {"x": "rru", "y": "rrv", "z": "rrw"}[axis]
                direction = {"x": "ax", "y": "ay", "z": "az"}[axis]
                values = {"a": qa, "s": padded, "rho0": m["rho0_3d"], mflux: flux,
                          direction: target, "dum": candidate, "ni": ni, "nj": nj, "nk": nk,
                          "ng": NG, "dt": dttmp}
                if not stretched:
                    values["gz"] = m["gz"]
                if stretched and axis == "z":
                    values.update(mh=mh_profile, rdz=rdz)
                else:
                    values[{"x": "dx", "y": "dy", "z": "dz"}[axis]] = spacing
                if axis != "z":
                    values["obc"] = obc
                self.guard.call(rk_family, f"cm1_rk_pdef_candidate_{axis}_v1", k[cand], grid_s, block_s,
                                values, stream=stream)
                values = dict(values)
                if not stretched:
                    values["rgz"] = m["rgz"]
                self.guard.call(rk_family, f"cm1_rk_pdef_correct_{axis}_v1", k[corr], grid_s, block_s,
                                values, stream=stream)
            zero = self._scratch("qv_zero_dir", (nk, nj, ni))

            def total(x: Any, y: Any, z: Any, div: Any, label: str) -> Any:
                out = self._scratch(f"qv_total_{label}", (nk, nj, ni))
                values = {"s": padded, "ax": x, "ay": y, "az": z, "divx": div, "rr0": m["rr0"],
                          "adv": out, "ni": ni, "nj": nj, "nk": nk, "ng": NG, "obc": obc}
                if not stretched:
                    values["gz"] = m["gz"]
                self.guard.call(rk_family, "cm1_rk_pdef_total_v1", k[13], grid_s, block_s, values, stream=stream)
                return out

            htot = total(ax, ay, zero, divh, "h")
            vtot = total(zero, zero, az, divv, "v")
            self._split_scalar("output_qvbudget", padded, htot, vtot, divh, divv, a, m, rdsf_levels, g)

    def _split_scalar(self, switch: str, padded: Any, htot: Any, vtot: Any, divh: Any, divv: Any,
                      a: Mapping[str, Any], m: Mapping[str, Any], rdsf_levels: Any, g: Any) -> None:
        xp = self.xp
        if self.has(switch, "hidiff") or self.has(switch, "vidiff"):
            hnd, vnd = nondiffusive_scalar(
                xp, s=padded, rru=a["rru"], rrv=a["rrv"],
                rrw=a["rrw"], c1=m["c1"], c2=m["c2"], divh=divh, divv=divv,
                rr0=m["rr0"], gz=m["gz"], rdsf_levels=rdsf_levels,
                ni=g.nx, nj=g.ny, nk=g.nz, ng=NG, dx=g.dx_m, dy=g.dy_m,
            )
            hadv, hidiff = implicit_diffusion_split(xp, htot, hnd)
            vadv, vidiff = implicit_diffusion_split(xp, vtot, vnd)
            self._put(switch, "hadv", hadv)
            self._put(switch, "hidiff", hidiff)
            self._put(switch, "vadv", vadv)
            self._put(switch, "vidiff", vidiff)
        else:
            self._put(switch, "hadv", htot)
            self._put(switch, "vadv", vtot)

    # ============================================================ mp_driver
    def before_microphysics(self, dynamics: Any) -> None:
        """``mp_driver.F:151-170``: save th3d and qv before pdefq and the scheme."""

        xp, f = self.xp, dynamics.fields
        if self.has("output_thbudget", "mp"):
            xp.copyto(self._scratch("mp_th", tuple(f["th3d"].shape)), f["th3d"])
        if self.has("output_qvbudget", "mp"):
            xp.copyto(self._scratch("mp_qv", tuple(f["q3d"][0].shape)), f["q3d"][0])

    def after_microphysics(self, dynamics: Any) -> None:
        """``mp_driver.F:1670-1690``: ``(after - before)*rdt``; then closure."""

        xp, f = self.xp, dynamics.fields
        dt = float(dynamics.config.dt_s)
        if self.has("output_thbudget", "mp"):
            self._put("output_thbudget", "mp", process_difference(xp, f["th3d"], self.scratch["mp_th"], dt))
        if self.has("output_qvbudget", "mp"):
            self._put("output_qvbudget", "mp", process_difference(xp, f["q3d"][0], self.scratch["mp_qv"], dt))
        # The step's scalar tendency, while tha/qa still hold its start:
        # CM1 runs mp_driver before solve3 (cm1.F:3416,3432) and only
        # solve3.F:612-643 moves th3d/q3d into tha/qa, which
        # CartesianFinish.solve3 does after this hook.
        for switch, current, start in (("output_thbudget", f["th3d"], f["tha"]),
                                       ("output_qvbudget", f["q3d"][0], f["qa"][0])):
            if switch not in self.terms:
                continue
            actual = process_difference(xp, current, start, dt)
            total = xp.zeros_like(actual)
            for term in self.terms[switch]:
                total = xp.add(total, self.fields[f"{BUDGET_PREFIX[switch]}_{term}"])
            interior = (slice(None), slice(3, max(3, self.ny - 3)), slice(3, max(3, self.nx - 3)))
            residual = xp.subtract(actual, total)[interior]
            prefix = BUDGET_PREFIX[switch]
            self.closure[f"{prefix}_residual_max_abs"] = float(xp.abs(residual).max())
            self.closure[f"{prefix}_tendency_max_abs"] = float(xp.abs(actual[interior]).max())

    # ============================================================ solve3
    def finish_step(self, dynamics: Any) -> None:
        """Curl every momentum term; publish the completed vorticity."""

        if self.active_step is None:
            return
        if self.vorticity_needed:
            self._vorticity(dynamics)
        self.completed_step = self.active_step
        self.active_step = None

    def _vorticity(self, dynamics: Any) -> None:
        xp, g = self.xp, dynamics.geometry
        ni, nj, nk = g.nx, g.ny, g.nz
        grid = self._grid()
        fields = self.fields
        tend = {axis: xp.zeros((nk, nj, ni), dtype=xp.float32) for axis in VORTICITY_AXES}
        advective = {axis: xp.zeros((nk, nj, ni), dtype=xp.float32) for axis in VORTICITY_AXES}
        present = set(self.terms.get("output_ubudget", ())) | set(self.terms.get("output_wbudget", ()))
        for term in VORTICITY_CURL_TERMS:
            if term not in present:
                continue
            tu = fields.get(f"ub_{term}")
            tv = fields.get(f"vb_{term}")
            tw = fields.get(f"wb_{term}")
            cx, cy, cz = curl_budget_term(xp, tu, tv, tw, grid)
            for axis, value in zip(VORTICITY_AXES, (cx, cy, cz)):
                name = f"{axis}vortb_{term}"
                if name in fields:
                    xp.copyto(fields[name], value)
                tend[axis] = xp.add(tend[axis], value)
                if term in ("hadv", "vadv", "hidiff", "vidiff"):
                    advective[axis] = xp.add(advective[axis], value)
        a = dynamics.arrays
        cfg = dynamics.config
        u = a["u3d"][1:nk + 1, NG:NG + nj, NG:NG + ni + 1]
        v = a["v3d"][1:nk + 1, NG:NG + nj + 1, NG:NG + ni]
        w = a["w3d"][1:nk + 2, NG:NG + nj, NG:NG + ni]
        omega = {
            "x": calcvort_x(xp, v, w, grid, vmove=float(cfg.v_move_m_s)),
            "y": calcvort_y(xp, u, w, grid, umove=float(cfg.u_move_m_s)),
            "z": calcvort_z(xp, u, v, grid),
        }
        for axis in VORTICITY_AXES:
            xp.copyto(fields[f"{axis}vortb"], omega[axis])
            xp.copyto(fields[f"{axis}vortb_tend"], tend[axis])
            decomposed = xp.zeros((nk, nj, ni), dtype=xp.float32)
            for term in VORTICITY_DECOMPOSITION_TERMS[:-1]:
                value = self._decomposition[f"{term}_{axis}"]
                xp.copyto(fields[f"{axis}vortb_{term}"], value)
                decomposed = xp.add(decomposed, value)
            xp.copyto(fields[f"{axis}vortb_advres"], xp.subtract(advective[axis], decomposed))

    # ============================================================ publication
    def history_fields(self, *, step: int, initial_condition: bool = False) -> dict[str, Any]:
        """Carriers for the history frame of ``step``, or a refusal."""

        names = [f"{BUDGET_PREFIX[s]}_{t}" for s in self.budgets for t in self.terms[s]]
        if self.vorticity_history:
            names.extend(vorticity_budget_names(self.facts))
        if not names:
            return {}
        if initial_condition or step == 0:
            # cm1.F allocates every diag array zero and cm1.F:3085-3088 writes
            # the nstep=0 frame before any solve, so CM1's own frame is zero.
            if self.completed_step is not None:
                raise RuntimeError("an initial-condition frame after a budget step")
            return {name: self.fields[name] for name in names}
        if self.completed_step != step:
            raise RuntimeError(
                f"history frame for step {step} asks for budget terms, but the budget "
                f"stage last completed step {self.completed_step}; cm1.F:3268-3296 "
                "computes the terms on the step whose state is written, and a frame "
                "carrying another step's terms would publish tendencies that are not "
                "the state's"
            )
        return {name: self.fields[name] for name in names}

    def parcel_fields(self, *, step: int) -> dict[str, Any]:
        if not self.parcel_vorticity:
            return {}
        if self.completed_step != step:
            raise RuntimeError(
                f"parcel frame for step {step} asks for vorticity-budget attributes "
                f"the stage did not compute (last completed step {self.completed_step})"
            )
        return {name: self.fields[name] for name in vorticity_budget_names(self.facts)}

    def parcel_hook(self) -> "ParcelVorticityBudgetHook":
        """The ``nest_parcels.ParcelAttributeHook`` that publishes :meth:`parcel_fields`."""

        from .lagrangian_parcels import parcel_attribute_units

        names = tuple(vorticity_budget_names(self.facts))
        return ParcelVorticityBudgetHook(self, names, tuple(parcel_attribute_units(n) for n in names))

    def record(self) -> dict[str, Any]:
        return {
            "schema": CARTESIAN_BUDGET_SCHEMA,
            "budgets": {
                switch: [f"{BUDGET_PREFIX[switch]}_{t}" for t in self.terms[switch]]
                for switch in self.budgets
            },
            "term_authority": {term: TERM_AUTHORITY[term][0] for term in TERM_AUTHORITY},
            "vorticity_budget": (
                list(vorticity_budget_names(self.facts)) if self.vorticity_needed else []
            ),
            "last_completed_step": self.completed_step,
            "closure": dict(self.closure),
            "closure_tendency": (
                "(current - start)/dt after mp_driver, start = tha/qa before "
                "solve3 publishes them (cm1.F:3416,3432; solve3.F:612-643)"
            ),
            "guarded_launches": int(self.guard.launches),
            "writes_model_carriers": False,
            "restart_state": (
                "none: every term is recomputed on its own due step "
                "(restart_write.F:2083-2092 writes diag arrays only under "
                "restart_file_diags, which run/namelist.input:296 ships .false.)"
            ),
            "unvalidated": {
                **({"budgets": UNVALIDATED_BUDGETS} if self.budgets else {}),
                **({"vorticity_budget": UNVALIDATED_VORTICITY} if self.vorticity_needed else {}),
            },
        }


@dataclass
class ParcelVorticityBudgetHook:
    """``[lagrangian] parcel_vorticity_budget``: the vorticity budget along parcels.

    GM1 EXTENSION.  Columns follow CM1's attributes; each is the scalar-point
    rectangle sampled by :func:`cm1core.nest_parcels.sample_mass_rectangles`
    on the owning grid's deck, the stencil CM1's ``prs``/``rho`` use.  Parcels
    no grid owns keep the sentinel.  The terms must be this frame's step's
    (:meth:`CartesianBudgetStage.parcel_fields` refuses any other).
    """

    stage: CartesianBudgetStage
    names: tuple[str, ...]
    units: tuple[str, ...]
    marker: str = UNVALIDATED_VORTICITY

    def sample(self, context: Any) -> dict[str, Any]:
        np = importlib.import_module("numpy")
        from .nest_parcels import SENTINEL_F32, sample_mass_rectangles

        n = int(context.positions.shape[1])
        out = {name: np.full((n,), np.float32(SENTINEL_F32), dtype=np.float32) for name in self.names}
        owned = [code for code, chosen in context.indices.items() if len(chosen)]
        if not owned:
            return out
        fields = self.stage.parcel_fields(step=int(context.step))
        for code in owned:
            chosen = context.indices[code]
            values = sample_mass_rectangles(context.xp, context.carriers[code],
                                            context.positions[:, chosen], fields)
            for name in self.names:
                out[name][chosen] = values[name]
        return out


def build_budget_stage(config: Any, *, xp: Any, allocate: Callable[[str, tuple[int, ...]], Any],
                       nest: bool = False) -> CartesianBudgetStage | None:
    """The stage a runtime configuration selects, or ``None`` for none."""

    from .cm1_budget_terms import facts_from_runtime

    lagrangian = getattr(config, "lagrangian", None)
    values = {} if lagrangian is None else dict(lagrangian.values)
    budgets = selected_budgets(getattr(config, "output", None))
    vorticity_history = bool(values.get("vorticity_budget_history", False))
    parcel_vorticity = bool(values.get("parcel_vorticity_budget", False))
    vorticity_frames = bool(values.get("vorticity_budget", False))
    if not budgets and not vorticity_history and not parcel_vorticity:
        return None
    facts = facts_from_runtime(config, nest=nest)
    g = config.grid
    return CartesianBudgetStage(
        xp=xp, allocate=allocate, facts=facts, budgets=budgets,
        nx=g.nx, ny=g.ny, nz=g.nz, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m,
        bbc=int(getattr(config, "bbc", 1)), tbc=int(getattr(config, "tbc", 1)),
        column=budget_column(xp, g), stretched=g.stretch_z != 0,
        vorticity_history=vorticity_history, parcel_vorticity=parcel_vorticity,
        vorticity_frames=vorticity_frames and (vorticity_history or parcel_vorticity),
    )


__all__ = [
    "CARTESIAN_BUDGET_SCHEMA", "CartesianBudgetStage", "BudgetColumn", "ParcelVorticityBudgetHook",
    "UNVALIDATED_BUDGETS", "UNVALIDATED_VORTICITY", "budget_column", "build_budget_stage",
]
