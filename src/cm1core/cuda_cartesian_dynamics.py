"""Fresh Cartesian dynamics assembled from verified CM1 operators.

This owns current/A carriers and phase-scoped storage.  It does not implement
closure, microphysics, mass correction, or the outer time-step scheduler.
No historical validation runner or fixed initialized-product export is used.

Two operator families run here, selected by the declared vertical coordinate
and by nothing else.  ``param.F:7789-7811``'s uniform grid keeps the operators
that reconstruct ``sigma``/``sigmaf``/``rds`` inline from the scalar ``dz``;
any grid ``param.F:7814-8139`` stretched runs the profile-bound family, which
reads the generated ``sigma``/``sigmaf``/``rds``/``rdsf`` the closure uploaded.
There is no third behaviour and no fallback: on a stretched grid the scalar-dz
operators would advect every layer whose depth is not ``dz`` at the wrong rate
(``adv.F:562-572,681-695`` multiply by ``rdz*mh``, not ``rdz``), so the
stretched selection raises rather than integrate that.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import importlib
import math
import os
from types import SimpleNamespace
from typing import Any

from . import cuda_graphs
from .cuda_memory_inventory import shape_bytes
from .cuda_psolver3_terrain_open import COLUMN_DIAGNOSTIC_NAMES, terrain_open_acoustic_workspace_shapes, terrain_open_sound_shapes
from .cuda_terrain_open_boundary import terrain_static_shapes
from .gm1_idealized_residency import DeviceBufferRegistry, IdealizedSource, SourceRequest, initial_prognostic_requests
from .gm1_runtime import RuntimeConfig


_MOMENTUM = ("ua", "u3d", "va", "v3d", "wa", "w3d")
_PRESSURE = ("ppi", "pp3d", "ppx")
_RATES = ("qpten", "qtten", "qvten", "qcten")
# CM1 param.F's uniform-horizontal branch sets these four to 1 and never
# changes them again; they are grid coefficients, not model state.
_UNIFORM_GRID_COEFFICIENTS = ("uh", "vh", "ruf", "rvf")
_PERSISTENT_PADDED = (*_MOMENTUM, *_PRESSURE, "ppten", "rru", "rrv", "rrw", "radbcw", "radbce", "radbcs", "radbcn", *_UNIFORM_GRID_COEFFICIENTS)
_PRE_PADDED = ("thv", "ppterm", "piadv", "uten", "vten", "wten")
# base.F builds these from the sounding and the grid and nothing writes them
# again.  README.namelist:1512-1516 says CM1's own restart files carry the
# base state only for specialized applications and that none of it is required
# for a bit-identical restart, so this port does not persist it either: the
# Rust initializer rebuilds it from the same configuration and sounding.
_IMMUTABLE_BASE_FIELDS = ("pi0", "th0", "qv0", "rho0")
#: base.F's base-state cloud water and ice (isnd = 4, 10, 11), rebuilt by the
#: initializer exactly as pi0/th0/qv0 are.
_BASE_CONDENSATE_FIELDS = ("qc0", "qi0")
_HELD = ("uten", "vten", "wten", "thten1", "qten")
_TKE = ("tkea", "tke3d", "tketen")

#: What a completed phase does with the CuPy pool blocks its scratch released.
#: ``keep`` (the default) leaves them cached for the next phase to reuse.
#: ``phase`` returns every unused block to the driver at the start of each
#: pre-acoustic stage and after each acoustic and scalar stage, which is what
#: every step did before.  What that cost is not yet measured: a census of
#: those frees backs ~150 ms per 512x512x75 step on an RTX 4090
#: with data and estimates 150-290 ms from a fitted model (~18.8 GiB returned
#: at 15-19 ms per GiB); the M1 run of ``tools/perf_probe/run_m0_m4.sh``
#: (``free_s``, ``sync_s``) replaces the estimate.
#: Neither choice is model arithmetic.  A buffer allocated with ``cupy.empty``
#: holds whatever the pool hands back either way, so every such buffer this
#: runtime reaches has to be written before it is read.  For GM1's own
#: allocations ``tools/oracle/cartesian_door_host_run.py --poison nan|big``
#: shows that on the host by filling every one of them and comparing the
#: published words; ArWen's column call is a host double there, so its
#: allocations are covered only by the poisoned device runs (NaN and
#: large-value poison, fresh and resumed), which gate ``keep`` as the default.  The environment
#: variable exists only so a device measurement can put the old behaviour
#: back in the same tree; it is not a configuration key and does not enter the
#: document's sealed hash.
DYNAMICS_POOL_RELEASE_ENVIRONMENT = "GM1_DYNAMICS_POOL_RELEASE"
DYNAMICS_POOL_RELEASE_MODES = ("keep", "phase")


def dynamics_pool_release_mode(environment: Mapping[str, str] | None = None) -> str:
    """The pool policy an environment selects; unknown values are refused."""

    value = (os.environ if environment is None else environment).get(
        DYNAMICS_POOL_RELEASE_ENVIRONMENT, "keep"
    )
    if value not in DYNAMICS_POOL_RELEASE_MODES:
        raise ValueError(
            f"{DYNAMICS_POOL_RELEASE_ENVIRONMENT}={value!r} is not one of "
            f"{DYNAMICS_POOL_RELEASE_MODES}; an unrecognised value would measure "
            "a pool policy nobody selected"
        )
    return value


def cartesian_dynamics_state_shapes(config: RuntimeConfig) -> dict[str, dict[str, tuple[int, ...]]]:
    """Every named float32 allocation this dynamics holds, by group.

    One table, read by the allocation plan that prices the run and by the
    restart profile that has to persist it.  A field added to the dynamics and
    not added here is priced by neither, which is the failure this exists to
    make impossible.
    """

    g = config.grid
    m, u, v, w = g.mass_shape, g.u_shape, g.v_shape, g.w_shape
    p = terrain_open_sound_shapes(g.nx, g.ny, g.nz)
    p.update({name: (1,) for name in COLUMN_DIAGNOSTIC_NAMES})
    # This is the flat, uniform Rust initializer's compressed layout.  The
    # upload independently proves every shared word/profile before use.
    metric_storage = {
        "rho0_3d": m, "rr0": m, "rf0": w, "rrf0": w,
        "constant_one": w, "constant_half": w,
        "constant_zero": (g.nz + 1, max(g.ny * (g.nx + 1), (g.ny + 1) * g.nx)),
        "zh_profile": (g.nz,), "zf_profile": (g.nz + 1,),
    }
    if g.stretch_z != 0:
        # The profile-bound horizontal mass flux reads rho0 on the halo
        # rectangle; _publish_padded_rho0 derives it once and keeps it.  It is
        # declared here rather than at the allocation site so that the restart
        # profile, which reads this same table, persists it too.
        metric_storage["rho0_padded"] = p["scalar"]
    return {
        "physical_fields": {
            "rf": w,
            **{name: m for name in (*_PRESSURE, "tha", "th3d", "prs", "rho", "rr", *_IMMUTABLE_BASE_FIELDS[:3], *_RATES)},
            # base.F's qc0/qi0 (solve2.F:404), only where the sounding has them.
            **({name: m for name in _BASE_CONDENSATE_FIELDS} if getattr(config, "base_condensate", False) else {}),
            # CM1's numq for the selected component (src/param.F), the same
            # extent the dynamics allocates.  A literal six here gave a
            # double-moment restart six of its ten rows: the writer streamed the
            # mass rows and restore left the number concentrations stale.
            **{name: (config.numq, *m) for name in ("qa", "q3d")},
        },
        "immutable_metrics": metric_storage,
        "held": {"uten": u, "vten": v, "wten": w, "thten1": m, "qten": (config.numq, *m)},
        "tke": {name: w for name in _TKE},
        "persistent_padded": {name: p[name] for name in _PERSISTENT_PADDED},
        "compact_base_profiles": {name: (g.nz,) for name in ("u0", "v0", "pi0", "th0", "qv0", *(_BASE_CONDENSATE_FIELDS if getattr(config, "base_condensate", False) else ()))},
    }


def cartesian_dynamics_restart_state(config: RuntimeConfig) -> dict[str, object]:
    """Split :func:`cartesian_dynamics_state_shapes` into carried and rebuilt.

    ``carried`` is what a restart must write, because nothing outside the
    integration can reproduce it.  ``rebuilt`` is what the Rust initializer
    produces again from the same configuration and sounding, each entry
    naming why persisting it is unnecessary rather than merely omitted.
    """

    shapes = cartesian_dynamics_state_shapes(config)
    padded = shapes["persistent_padded"]
    fields = shapes["physical_fields"]
    return {
        "carried": {
            # The padded owners, not the physical rectangles: the wind and
            # pressure fields are strided views into these, and the next large
            # step's fifth-order advection reads the halo words a physical
            # rectangle does not contain.
            "padded": {
                name: shape for name, shape in padded.items()
                if name not in _UNIFORM_GRID_COEFFICIENTS
            },
            "fields": {
                name: shape for name, shape in fields.items()
                if name not in _IMMUTABLE_BASE_FIELDS and name not in _BASE_CONDENSATE_FIELDS
            },
            "held": dict(shapes["held"]),
            "tke": dict(shapes["tke"]),
        },
        "rebuilt": {
            "immutable_metrics": (
                "grid metrics and base-state volumes the Rust initializer derives "
                "from the declared geometry; no operator writes them"
            ),
            "compact_base_profiles": (
                "base.F sounding and base-state wind profiles; rebuilt from the "
                "same sounding bytes the restart receipt pins by SHA-256"
            ),
            "physical_base_fields": (
                "pi0/th0/qv0/rho0 (and qc0/qi0 where base.F builds them); README.namelist:1512-1516 records that CM1's "
                "own base-state restart fields are for specialized applications "
                "and are not required for a bit-identical restart"
            ),
            "uniform_grid_coefficients": (
                "uh/vh/ruf/rvf; CM1 param.F's uniform-horizontal branch sets all "
                "four to exactly 1 and no operator writes them"
            ),
            "physical_momentum_views": (
                "ua/u3d/va/v3d/wa/w3d; strided views into the padded owners that "
                "are carried, so persisting them again would write the same words "
                "under a second name"
            ),
        },
    }


def _on_stream(method: Any) -> Any:
    @wraps(method)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self.stream:
            return method(self, *args, **kwargs)
    return run


def pack_cartesian_held_momentum(
    cupy: Any, held: Mapping[str, Any], padded: Mapping[str, Any],
    geometry: Any, *, stream: Any,
) -> None:
    """Copy physical held fields into their actual padded source rectangles.

    The old ``cm1_momentum_rk_seed_held_v1`` accepts padded inputs and cannot
    consume these physical allocations.  Fresh extra common-loop fringes and
    halos stay exact zero, as in the admitted live full-held adapter.
    """
    selections = []
    for name, stagger in (("uten", "u"), ("vten", "v"), ("wten", "w")):
        source, destination = held[name], padded[name]
        if (
            tuple(source.shape) != geometry.physical_shape(stagger)
            or tuple(destination.shape) != geometry.padded_shape(stagger, halo_xy=3, halo_z=1)
            or source.dtype != cupy.dtype("float32")
            or destination.dtype != cupy.dtype("float32")
            or not source.flags.c_contiguous or not destination.flags.c_contiguous
        ):
            raise ValueError(f"{name} held packing requires distinct physical-input and padded-output layouts")
        selections.append((source, destination, geometry.physical_slices(stagger, halo_xy=3, halo_z=1)))
    def byte_range(value: Any) -> tuple[int, int]:
        data = value.data
        pointer = int(data.ptr) if hasattr(data, "ptr") else int(value.ctypes.data)
        return pointer, pointer + int(value.nbytes)
    destinations = [byte_range(destination) for _, destination, _ in selections]
    sources = [byte_range(source) for source, _, _ in selections]
    def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return max(left[0], right[0]) < min(left[1], right[1])
    if any(overlap(left, right) for left in sources for right in destinations) or any(
        overlap(destinations[i], destinations[j]) for i in range(3) for j in range(i + 1, 3)
    ):
        raise ValueError("physical held packing inputs and padded outputs must not overlap")
    with stream:
        for source, destination, selection in selections:
            destination.fill(cupy.float32(0.0))
            cupy.copyto(destination[selection], source)


def cartesian_dynamics_allocation_plan(config: RuntimeConfig) -> dict[str, Any]:
    """Price exact named float32 shapes without initializing or touching CUDA."""

    g = config.grid
    m, u, v, w = g.mass_shape, g.u_shape, g.v_shape, g.w_shape
    p = terrain_open_sound_shapes(g.nx, g.ny, g.nz)
    p.update({name: (1,) for name in COLUMN_DIAGNOSTIC_NAMES})
    # One table, shared with the restart profile: it already declares the
    # stretched family's padded base density, which the dynamics uploads as a
    # metric on every stretch_z != 0 grid (extend_immutable_source_metrics), so
    # a second hand-written copy here would price a run without it.
    groups = cartesian_dynamics_state_shapes(config)
    numq = config.numq
    rows = {name: sum(shape_bytes(shape) for shape in shapes.values()) for name, shapes in groups.items()}
    persistent = sum(rows.values())
    pressure_scratch = sum(shape_bytes(shape) for name, shape in p.items() if name not in _PERSISTENT_PADDED)
    scalar_scratch = 2 * shape_bytes(p["scalar"]) + sum(shape_bytes(p[name]) for name in ("u3d", "v3d", "w3d")) + 5 * shape_bytes(m)
    # dum8 dies after buoyancy; only thterm and divx survive acoustics.
    boundary_scratch = sum(shape_bytes(shape) for shape in terrain_open_acoustic_workspace_shapes(g.nx, g.ny, g.nz).values())
    peak_acoustic = persistent - 3 * shape_bytes(m) + pressure_scratch + 2 * shape_bytes(m) + boundary_scratch
    peak_scalar = persistent + scalar_scratch + 2 * shape_bytes(m)
    # The Rust initializer publishes six mass rows.  A wider scalar axis holds
    # those uploaded rows and its own zeroed allocation at once for the length
    # of one copy, before the uploaded pair is released; the budget must cover
    # that transient or a run would be admitted and then fail mid-upload.
    packed_upload_transient = (
        0 if numq == 6 else 2 * 6 * shape_bytes(m)
    )
    peak_upload = persistent + packed_upload_transient
    return {
        "schema": "gm1.cartesian-dynamics-allocation-plan/v1",
        "geometry": g.record(),
        "groups_payload_bytes": rows,
        "persistent_payload_bytes": persistent,
        "acoustic_phase_payload_bytes": peak_acoustic,
        "acoustic_boundary_workspace_bytes": boundary_scratch,
        "expanded_base_volume_bytes_released_for_acoustics": 3 * shape_bytes(m),
        "scalar_phase_payload_bytes": peak_scalar,
        "packed_scalar_upload_transient_bytes": packed_upload_transient,
        "upload_phase_payload_bytes": peak_upload,
        "maximum_listed_phase_payload_bytes": max(peak_acoustic, peak_scalar, peak_upload),
        "aliases": {"fields.rho0": "metrics.rho0_3d", "scalar_A_current_held": "physical_fields+held", "physical_wind_fields": "strided physical views of each same-name padded owner; A/current owners remain distinct"},
        "claim_limit": "named dynamics payloads only; excludes allocator rounding, driver/context, closure/physics workspaces and cached pool reserve",
    }


class CartesianDynamics:
    """One source-initialized device state with explicit RK-phase methods."""

    # Set by CudaCartesianRuntime.enable_cuda_graphs; see _drop_expanded_base.
    keep_expanded_base = False

    #: ``None`` for every single-domain run, which is every run this lane made
    #: before a live nest existed: no nest kernel is compiled, enqueued or
    #: allocated.  A live nest's child carries a
    #: :class:`cm1core.nest_live.NestLateralForcing` here, which adds the Davies
    #: relaxation to the held tendencies and re-imposes the specified zone
    #: inside the acoustic small steps and after each RK scalar stage.
    nest_forcing: Any = None
    #: ``None`` unless an ``output_*budget`` or the vorticity budget is
    #: selected: a :class:`cm1core.cartesian_budgets.CartesianBudgetStage`
    #: that reads this stage's carriers through a write guard and writes none.
    budget_probe: Any = None

    def __init__(
        self, source: IdealizedSource, config: RuntimeConfig, *, cupy: Any, stream: Any,
        registry: DeviceBufferRegistry | None = None,
    ) -> None:
        if source.geometry != config.grid:
            raise ValueError("source geometry differs from runtime configuration")
        self.config, self.cupy, self.stream = config, cupy, stream
        self.source = source
        self.geometry = config.grid
        if registry is None:
            if config.memory_budget_bytes is None:
                raise ValueError("an explicit device memory budget is required before allocation")
            registry = DeviceBufferRegistry(cupy, stream, budget_bytes=config.memory_budget_bytes)
        if registry.cupy is not cupy or registry.stream is not stream:
            raise ValueError("registry must use this runtime's CuPy module and stream")
        self.registry = registry
        self.plan = cartesian_dynamics_allocation_plan(config)
        if self.plan["maximum_listed_phase_payload_bytes"] > registry.budget_bytes:
            raise MemoryError("declared dynamics phase payload exceeds the explicit registry budget")
        self._padded_shapes = terrain_open_sound_shapes(self.geometry.nx, self.geometry.ny, self.geometry.nz)
        self._padded_shapes.update({name: (1,) for name in COLUMN_DIAGNOSTIC_NAMES})
        self.fields: dict[str, Any] = {}
        self.base_profiles: dict[str, Any] = {}
        self.arrays: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.held: dict[str, Any] = {}
        self.tke: dict[str, Any] = {}
        self.last_receipts: dict[str, Mapping[str, Any]] = {}
        self.pool_release = dynamics_pool_release_mode()
        self.last_receipts["device_pool"] = {
            "schema": "gm1.cartesian-dynamics-device-pool/v1",
            "release": self.pool_release,
            "environment": DYNAMICS_POOL_RELEASE_ENVIRONMENT,
            "model_arithmetic": False,
        }
        self._expected_phase = ("held", 1)
        self._scalar_workspace = False
        self._bridge_workspace: set[str] = set()
        self._boundary_workspace: dict[str, Any] = {}
        self._acoustic_workspace: dict[str, Any] = {}
        # Bound by CartesianClosure._vertical_profiles before any phase runs;
        # it owns the one generator (cm1_vertical_grid) this port has.
        self.vertical: Any = None
        self.vertical_launch: Any = None

        self.numq = config.numq
        # The Rust idealized initializer publishes CM1's six mass mixing
        # ratios (src/param.F:3641-3678).  A double-moment component
        # transports those same six plus its number concentrations, which
        # CM1's own initialization leaves at exact zero (init3d.F zeroes the
        # whole q3d array and base.F fills only qv), so the extra rows are
        # allocated zeroed and the six uploaded rows are copied into place.
        # numq == 6 requests, reserves and keeps exactly the buffers it always
        # did, in exactly the order it always did.
        packed = ("qa", "q3d")
        packed_prefix = "field." if self.numq == 6 else "source."
        requests = [
            SourceRequest(
                (packed_prefix if r.name in packed else "field.") + r.name,
                r.source, r.layout, r.read_only,
            )
            for r in initial_prognostic_requests()
            if r.name not in _MOMENTUM
        ]
        requests += [SourceRequest(f"field.{name}", f"terrain_open.{name}", "physical", True) for name in ("pi0", "th0", "qv0")]
        if getattr(config, "base_condensate", False):
            # base.F's qc0/qi0 (solve2.F:404), from the initializer's own group.
            requests += [SourceRequest(f"field.{name}", f"base_condensate.{name}", "physical", True) for name in _BASE_CONDENSATE_FIELDS]
        uploaded = registry.upload(source, requests)
        self.fields.update({
            name.removeprefix("field."): value
            for name, value in uploaded.items()
            if name.startswith("field.")
        })
        if self.numq != 6:
            with stream:
                for name in packed:
                    source_rows = uploaded[f"source.{name}"]
                    value = registry.allocate(
                        f"field.{name}", (self.numq, *self.geometry.mass_shape), zero=True
                    )
                    cupy.copyto(value[: source_rows.shape[0]], source_rows)
                    self.fields[name] = value
            stream.synchronize()
            registry.release([f"source.{name}" for name in packed])
        from .cuda_immutable_metrics import upload_immutable_source_metrics
        self.metrics = upload_immutable_source_metrics(source, registry)
        self.fields["rho0"] = self.metrics["rho0_3d"]
        # The stretched operator family averages base density across the outer
        # faces of the open lateral boundary (sound.F:145-188,
        # misclibs.F:222-370), which the physical rectangle cannot source.  The
        # uniform family clamps at the face instead and never reads this, so it
        # is uploaded only where it is consumed.
        self.stretched_metrics: Mapping[str, Any] | None = None
        if self.geometry.stretch_z != 0:
            from .cuda_immutable_metrics import extend_immutable_source_metrics

            self.stretched_metrics = extend_immutable_source_metrics(
                self.metrics, source, registry, {"rho0_padded": "terrain_open.rho0"}
            )
            expected = self._padded_shapes["scalar"]
            if tuple(self.stretched_metrics["rho0_padded"].shape) != expected:
                raise ValueError(
                    f"Rust padded base density must be {expected}; the stretched "
                    "acoustic and mass-flux operators index it as the scalar halo"
                )
            # CM1 base.F:2140-2145 fills rho0's lateral halo with bcs and its
            # k=0 / nk+1 ghosts with extrapbcs (bc.F:1201-1202, three-point
            # extrapolation).  The Rust initializer reproduces both, so the
            # uploaded array is the one every stretched operator reads.
            self.metrics = self.stretched_metrics
        base_requests = [SourceRequest(f"base.{name}", f"base_wind.{name}", "profile", True) for name in ("u0", "v0")]
        base_requests += [SourceRequest(f"base.{name}", f"terrain_open.{name}", "profile", True) for name in ("pi0", "th0", "qv0")]
        if getattr(config, "base_condensate", False):
            base_requests += [SourceRequest(f"base.{name}", f"base_condensate.{name}", "profile", True) for name in _BASE_CONDENSATE_FIELDS]
        self.base_profiles.update({name.removeprefix("base."): value for name, value in registry.upload(source, base_requests).items()})
        terrain = source.groups["terrain_open"]
        for name in _PERSISTENT_PADDED:
            if name in terrain:
                value = registry.upload(source, [SourceRequest(f"padded.{name}", f"terrain_open.{name}")])[f"padded.{name}"]
                if tuple(value.shape) != self._padded_shapes[name]:
                    raise ValueError(f"Rust padded carrier {name} has unexpected source extent")
            else:
                value = registry.allocate(f"padded.{name}", self._padded_shapes[name], zero=True)
            self.arrays[name] = value
        # CM1 param.F's uniform-horizontal branch sets uh/vh/ruf/rvf=1.
        # These four grid coefficients are not atmospheric Rust carriers.
        with stream:
            for name in ("uh", "vh", "ruf", "rvf"):
                self.arrays[name].fill(cupy.float32(1.0))
        # A wind field has one owner.  Expose its physical rectangle as a
        # strided view; horizontal tile packing handles contiguous consumers.
        # A and current still have different underlying allocations.
        for name, stagger in (("ua", "u"), ("u3d", "u"), ("va", "v"), ("v3d", "v"), ("wa", "w"), ("w3d", "w")):
            self.fields[name] = self.arrays[name][self.geometry.physical_slices(stagger, halo_xy=3, halo_z=1)]
        for name in _RATES:
            self.fields[name] = registry.allocate(f"field.{name}", self.geometry.mass_shape, zero=True)
        for name, shape in {"uten": self.geometry.u_shape, "vten": self.geometry.v_shape, "wten": self.geometry.w_shape, "thten1": self.geometry.mass_shape, "qten": (self.numq, *self.geometry.mass_shape)}.items():
            self.held[name] = registry.allocate(f"held.{name}", shape, zero=True)
        for name in ("tkea", "tke3d", "tketen"):
            self.tke[name] = registry.allocate(f"tke.{name}", self.geometry.w_shape, zero=True)
        self._compile_operators()
        self._bind_rotation()
        # init3d.F:1303-1396: iinit = 9, 10 or 12 start the storm with a
        # forcing applied during the run; None for every other iinit.
        self.initiation: Any = None
        if getattr(config, "initiation_forcing", None) is not None:
            from .initiation_forcing import InitiationForcingStage

            self.initiation = InitiationForcingStage(config.initiation_forcing, self)
        self._bind_stage_states()
        # CM1 getset (init3d.F:2386-2467) rebuilds current prs/rho/rr/rf from the
        # initialized perturbation state before the model advances.  The warm
        # bubble is in th3d, so the ledger and the first closure must see it.
        self.startup_receipt = self._publish_startup_thermodynamics()
        self.mass1_f64 = self._initial_dry_mass()

    def _compile_operators(self) -> None:
        from .cuda_advection5 import compile_cuda_advection5
        from .cuda_apmasscon import compile_cuda_apmasscon
        from .cuda_cartesian_startup import compile_cuda_cartesian_startup
        from .cuda_gm1_child_large_step import compile_cuda_gm1_child_theta_pack
        from .cuda_gm1_child_step_composer import compile_cuda_gm1_child_core
        from .cuda_gm1_forecast import compile_cuda_gm1_forecast
        from .cuda_gm1_rk1_post_scalar_momentum_boundaries import compile_cuda_gm1_rk1_post_scalar_momentum_boundary
        from .cuda_lateral_closure import compile_cuda_lateral_closure
        from .cuda_moist_stage_bridge import compile_cuda_moist_stage_bridge
        from .cuda_momentum_advection5 import compile_cuda_momentum_advection5
        from .cuda_momentum_rk import compile_cuda_momentum_rk
        from .cuda_psolver3_terrain_open import compile_cuda_psolver3_terrain_open
        from .cuda_scalar_moist_rk import compile_cuda_scalar_moist_rk, CompiledScalarMoistRKKernels
        from .cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary
        from .cuda_terrain_open_boundary import compile_cuda_terrain_open_boundary

        from .cuda_boundary_families import (
            DEFAULT_LATERAL_BOUNDARY_SELECTION,
            bind_lateral_families,
            compile_cuda_boundary_families,
            lateral_boundary_workspace_shapes,
        )

        # CM1's lateral family travels with the compiled bundle: every consumer
        # in this port reaches the bcp and bcs/bcu/bcv/bcw fills through
        # open_compiled.kernels[3:7], so binding the selection here is what
        # makes a periodic, rigid or mixed family reach the momentum, scalar,
        # pressure and thermodynamic halos at once (src/bc.F:180-1143).
        self.lateral_selection = getattr(
            self.config, "lateral_boundary_selection", None
        ) or DEFAULT_LATERAL_BOUNDARY_SELECTION
        self.family_compiled = compile_cuda_boundary_families()
        self.open_compiled = bind_lateral_families(
            compile_cuda_sound3_open_boundary(), self.family_compiled, self.lateral_selection
        )
        g = self.geometry
        # Nothing is allocated for the family this lane has always run, so its
        # allocation plan, its receipts and its recorded bytes are unchanged.
        self.lateral_workspace: dict[str, Any] = {}
        if self.lateral_boundary_work_required:
            f64 = self.cupy.float64
            self.lateral_workspace = {
                name: self.registry.allocate(
                    f"lateral.{name}", shape, dtype=f64 if name in ("temout", "temin", "t3") else None, zero=True
                )
                for name, shape in lateral_boundary_workspace_shapes(g.nx, g.ny, g.nz).items()
            }
        self.terrain_compiled = compile_cuda_terrain_open_boundary()
        self.lateral_compiled = compile_cuda_lateral_closure()
        self.momentum_compiled = compile_cuda_momentum_rk()
        self.momentum_advection_compiled = compile_cuda_momentum_advection5()
        self.psolver_compiled = compile_cuda_psolver3_terrain_open(retain_column_diagnostics=False)
        self.moist_compiled = compile_cuda_moist_stage_bridge()
        self.scalar_advection_compiled = compile_cuda_advection5()
        legacy = compile_cuda_scalar_moist_rk()
        pack = compile_cuda_gm1_child_theta_pack()
        self.scalar_compiled = CompiledScalarMoistRKKernels(
            (legacy.module_owner, pack.module_owner), (pack.kernel, *legacy.kernels[1:]),
            legacy.ptx, legacy.cubin, legacy.target_arch, legacy.ptx_options, legacy.cubin_options, legacy.runtime_identity,
        )
        self.scalar_legacy_compiled, self.scalar_pack_compiled = legacy, pack
        if self.geometry.stretch_z != 0:
            # The Cartesian lane is terrain_flag=.false. (config.py admits
            # grid.terrain only on the real-product lane), so a stretched
            # column runs CM1's flat statements: rdz*mh(1,1,k) and
            # rdz*mf(1,1,k) (adv.F, pdef.F, misclibs.F, solve2.F), sound.F's
            # flat coefficient/pressure-gradient/pressure statements, and no
            # terrain-following term anywhere.  The terrain child keeps the
            # rds/rdsf family these images were derived from.
            from .cuda_gm1_child_psolver3_open import (
                compile_cuda_gm1_flat_stretched_psolver3_open,
            )
            from .cuda_gm1_child_stretched_moist import compile_cuda_gm1_flat_stretched_moist
            from .cuda_gm1_child_stretched_momentum import (
                compile_cuda_gm1_flat_stretched_momentum,
            )
            from .cuda_gm1_child_stretched_scalar import (
                compile_cuda_gm1_flat_stretched_scalar_advection,
                compile_cuda_gm1_flat_stretched_scalar_pdef,
            )

            self.momentum_advection_compiled = compile_cuda_gm1_flat_stretched_momentum()
            # This lane owns the six sound.F:659-730 column reports at one word
            # each, on both branches, because at 640x640x75 they are 762 MB of
            # pure diagnostic and nothing downstream reads them: post_acoustic
            # drops every non-persistent padded carrier before the bridge runs.
            # The uniform operator has always been compiled that way; the
            # stretched successor is compiled the same way, so the extents it
            # holds its caller to are the extents this lane actually allocated.
            self.psolver_compiled = compile_cuda_gm1_flat_stretched_psolver3_open(
                retain_column_diagnostics=False
            )
            self.stretched_moist_compiled = compile_cuda_gm1_flat_stretched_moist()
            self.scalar_advection_compiled = compile_cuda_gm1_flat_stretched_scalar_advection()
            stretched_pdef = compile_cuda_gm1_flat_stretched_scalar_pdef()
            self.scalar_compiled = CompiledScalarMoistRKKernels(
                (stretched_pdef.module_owner, pack.module_owner),
                (pack.kernel, *stretched_pdef.kernels[1:]),
                stretched_pdef.ptx, stretched_pdef.cubin, stretched_pdef.target_arch,
                stretched_pdef.ptx_options, stretched_pdef.cubin_options,
                stretched_pdef.runtime_identity,
            )
            self.scalar_legacy_compiled = stretched_pdef
        self.boundary_compiled = compile_cuda_gm1_rk1_post_scalar_momentum_boundary(open_compiled=self.open_compiled)
        self.forecast_compiled = compile_cuda_gm1_forecast()
        self.child_core_compiled = compile_cuda_gm1_child_core()
        self.apmasscon_compiled = compile_cuda_apmasscon()
        self.startup_compiled = compile_cuda_cartesian_startup()
        self.stretched_compiled: dict[str, Any] = {}
        if self.stretched:
            from .cuda_gm1_child_psolver3_open import compile_cuda_gm1_flat_stretched_psolver3_open
            from .cuda_gm1_child_stretched_moist import compile_cuda_gm1_flat_stretched_moist
            from .cuda_gm1_child_stretched_momentum import (
                compile_cuda_gm1_flat_stretched_momentum,
            )
            from .cuda_gm1_child_stretched_scalar import (
                compile_cuda_gm1_flat_stretched_scalar_advection,
                compile_cuda_gm1_flat_stretched_scalar_pdef,
            )

            self.stretched_compiled = {
                "momentum": compile_cuda_gm1_flat_stretched_momentum(),
                "psolver": compile_cuda_gm1_flat_stretched_psolver3_open(
                    retain_column_diagnostics=False
                ),
                "moist": compile_cuda_gm1_flat_stretched_moist(),
                "advection": compile_cuda_gm1_flat_stretched_scalar_advection(),
                "pdef": compile_cuda_gm1_flat_stretched_scalar_pdef(),
            }

    def _vertical(self) -> Any:
        """This run's generated column, or a refusal naming what it decides.

        ``CartesianClosure`` builds it from ``grid.levels`` before the first
        large step and binds it here.  A stretched stage that ran without it
        would have no ``rds``/``rdsf`` to divide by at all.
        """

        vertical = getattr(self, "vertical", None)
        if vertical is None:
            raise RuntimeError(
                f"grid.stretch_z={self.geometry.stretch_z} needs the generated "
                "sigma/sigmaf/rds/rdsf profiles bound before a stage runs; "
                "param.F:8162-8209 and init_terrain.F:292-298 are what every "
                "vertical difference in this column divides by"
            )
        return vertical

    def _bind_rotation(self) -> None:
        """CM1 solve2.F:631-720, the icor=1 Coriolis term, for this domain.

        ``icor=0`` binds nothing and enqueues nothing, which is the identity
        this lane has always integrated and what CM1's own supercell and
        squall-line decks select.  ``icor=1`` prepares one stage-bound
        binding per RK substep over this run's own ``yh`` column, so the
        f-plane and beta-plane branches are reachable on any grid rather than
        on the one sealed 300x300x75 child the binding used to demand.

        The frame interaction is the whole reason this is not a free switch.
        ``base.F:2667`` subtracted ``umove``/``vmove`` from every base-state
        wind, so every wind this runtime carries is frame-relative;
        ``solve2.F:645,647`` add the frame velocity back inside the Coriolis
        term so the acceleration is still a function of the ground-relative
        wind.  The binding carries this run's declared frame speeds for
        exactly that add-back.
        """

        from .cuda_gm1_solve2_rotation import (
            compile_cuda_gm1_solve2_rotation,
            prepare_cuda_gm1_solve2_rotation,
        )
        from .rotation_forcing import RotationForcingSelection

        self.rotation_bindings: dict[int, Any] = {}
        config, g = self.config, self.geometry
        if getattr(config, "icor", 0) != 1:
            return
        numpy = importlib.import_module("numpy")
        # param.F:6653 yfdp(j)=dy*(j-1) and param.F:7700 yh(j)=0.5*(yfdp(j)+
        # yfdp(j-1)), i.e. yh(j)=dy*(j-0.5); param.F:7743-7745 centery is the
        # midpoint of the physical domain.  Only (yh-centery) enters
        # param.F:8966, so the choice of iorigin cannot change one f2d word.
        yh = (numpy.arange(g.ny + 2, dtype=numpy.float64) - 0.5) * g.dy_m
        selection = RotationForcingSelection(
            icor=1,
            beta_plane=int(config.betaplane),
            axisymmetric=0,
            fcor_s_inv=float(config.fcor_s_inv),
            u_move_m_s=float(config.u_move_m_s),
            v_move_m_s=float(config.v_move_m_s),
            center_y_m=0.5 * g.ny * g.dy_m,
        )
        compiled = compile_cuda_gm1_solve2_rotation()
        for nrk in (1, 2, 3):
            self.rotation_bindings[nrk] = prepare_cuda_gm1_solve2_rotation(
                numpy.ascontiguousarray(yh, dtype=numpy.float32),
                ni=g.nx,
                nj=g.ny,
                nk=g.nz,
                selection=selection,
                nrk=nrk,
                stream=self.stream,
                compiled=compiled,
                adaptive_dt=bool(getattr(config, "adaptive_timestep", False)),
            )
        self.last_receipts["rotation_binding"] = self.rotation_bindings[1].receipt

    def _rotation_hook(self, nrk: int) -> Any:
        """The one seam CM1 puts between ``divx`` and ``advu``."""

        binding = self.rotation_bindings.get(nrk)
        if binding is None:
            return None
        from .cuda_gm1_solve2_rotation import (
            enqueue_cuda_gm1_solve2_rotation_after_divergence,
        )

        def hook(arrays: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
            receipt = enqueue_cuda_gm1_solve2_rotation_after_divergence(
                binding, arrays, **kwargs
            )
            self.last_receipts["rotation"] = receipt
            return receipt

        return hook

    @property
    def lateral_step(self) -> Any:
        """The lateral-family hook the acoustic launcher needs, or None.

        ``None`` is the sealed all-open family: no extra kernel is enqueued and
        this lane's recorded acoustic trajectory is exactly what it was.
        """

        if not self.lateral_boundary_work_required:
            return None
        from .cuda_boundary_families import LateralBoundaryStep

        return LateralBoundaryStep(
            self.lateral_selection, self.family_compiled, self.lateral_workspace
        )

    @property
    def lateral_boundary_work_required(self) -> bool:
        """True when this run's lateral family is not the sealed all-open one.

        ``roflux``, ``nudgeobc`` and ``irbc=1/2`` each add a call CM1 makes and
        this lane did not; a non-open side adds the face hold of
        ``src/sound.F:274-302`` and, for a periodic side, the boundary face of
        ``src/sound.F:388``.
        """

        s = self.lateral_selection
        return not (
            s.all_open and s.irbc == 4 and s.roflux == 0 and s.nudgeobc == 0
        )

    def _event(self) -> Any:
        event = self.cupy.cuda.Event(disable_timing=True)
        event.record(self.stream)
        return event

    @property
    def stretched(self) -> bool:
        """Is this run's vertical coordinate anything but param.F:7789-7811's?"""

        return self.geometry.stretch_z != 0

    def _stretched_vertical(self) -> Any:
        """The device sigma/sigmaf/rds/rdsf this run's coordinate generated."""

        from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics

        if not isinstance(self.vertical, Gm1ChildStretchedMetrics):
            raise RuntimeError(
                f"grid.stretch_z={self.geometry.stretch_z} selects the "
                "profile-bound operator family, but no vertical binding has "
                "been published; CartesianClosure must be constructed before "
                "the first large step"
            )
        return self.vertical

    def _bind_stage_states(self) -> None:
        from .cuda_moist_stage_bridge import CudaMoistStageBridgeState
        from .cuda_scalar_moist_rk import CudaScalarMoistRKState

        g, f, h = self.geometry, self.fields, self.held
        device = int(self.cupy.cuda.Device().id)
        self.moist_bridge = CudaMoistStageBridgeState(g.nx, g.ny, g.nz, True, None, None, None, None, None, self.stream, device, self.moist_compiled, self.open_compiled)
        self.scalar_rk = CudaScalarMoistRKState(
            g.nx, g.ny, g.nz, f["th0"], f["tha"], f["th3d"], f["qa"], f["q3d"], h["thten1"], h["qten"],
            None, None, {}, None, None, None, None, None, self._event(), self.stream, device, 1, False,
            # src/solve2.F:1472 scales the WENO positive-definite epsilon by
            # CM1's own per-scalar qmag, which differs between the mass and
            # number rows of a double-moment axis.
            self.config.profile.qmag,
            # The same rows' CM1 qname, so the stage receipt names each one.
            self.config.profile.names,
        )

    def mass_prefix_q3d(self) -> Any:
        """CM1's six mass rows of ``q3d``: the eqtset=2 stage terms' input.

        ``src/solve2.F:317-392`` forms the liquid total over ``nql1..nql2`` and
        the solid total over ``nqs1..nqs2``; buoyancy, theta-rho and the
        cpm/cvm energy terms (``:394-412``, ``:895-921``) read nothing else.
        A number concentration is therefore never a stage-term input.  Every
        admitted axis carries those six rows first
        (:func:`~cm1core.moisture_species.require_cm1_mass_prefix`), so the
        window is a C-contiguous leading-axis view of the same allocation and
        a six-row axis hands over its whole array, as it always has.
        """

        q3d = self.fields["q3d"]
        if self.numq == 6:
            return q3d
        from .moisture_species import require_cm1_mass_prefix

        return q3d[: require_cm1_mass_prefix(self.config.profile)]

    def _publish_startup_thermodynamics(self) -> Mapping[str, Any]:
        """CM1 init3d.F getset, psolver<=3 moist branch, run exactly once."""

        from .cuda_cartesian_startup import launch_cuda_cartesian_startup

        g = self.geometry
        return launch_cuda_cartesian_startup(
            self.fields,
            self.metrics,
            ni=g.nx,
            nj=g.ny,
            nk=g.nz,
            cupy=self.cupy,
            stream=self.stream,
            compiled=self.startup_compiled,
            density_kernel=self.forecast_compiled.kernels[1],
            numq=self.numq,
        )

    def _initial_dry_mass(self) -> float:
        g, cp = self.geometry, self.cupy
        mass = self.registry.allocate("initial.mass_layers", (g.nz,), dtype=cp.float64)
        pressure = self.registry.allocate("initial.pressure_layers", (g.nz,), dtype=cp.float64)
        result = self.registry.allocate("initial.mass_result", (1,), dtype=cp.float64)
        self.apmasscon_compiled.kernels[0](
            (g.nz, 1, 1), (256, 1, 1),
            (self.fields["rho"], self.arrays["uh"][3:3 + g.nx], self.arrays["vh"][3:3 + g.ny], self.metrics["rmh"], self.fields["pi0"], self.fields["pp3d"], mass, pressure, cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)),
            stream=self.stream,
        )
        self.forecast_compiled.kernels[2]((1, 1, 1), (1, 1, 1), (mass, result, cp.uint32(g.nz)), stream=self.stream)
        value = float(result.get(stream=self.stream)[0])
        self.registry.release(["initial.mass_layers", "initial.pressure_layers", "initial.mass_result"])
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("source-initialized dry-mass ledger is not finite and positive")
        return value

    def _ensure_arrays(self, names: Any) -> None:
        for name in names:
            if name not in self.arrays:
                self.arrays[name] = self.registry.allocate(f"padded.{name}", self._padded_shapes[name], zero=True)

    def _ensure_bridge(self, names: tuple[str, ...]) -> None:
        for name in names:
            if name not in self._bridge_workspace:
                shape = self._padded_shapes["scalar"] if name == "dum8" else self.geometry.mass_shape
                setattr(self.moist_bridge, name, self.registry.allocate(f"bridge.{name}", shape, zero=True))
                self._bridge_workspace.add(name)

    def _drop_bridge(self, names: tuple[str, ...]) -> None:
        active = [name for name in names if name in self._bridge_workspace]
        if active:
            cuda_graphs.synchronize(self.stream)
            for name in active:
                setattr(self.moist_bridge, name, None)
                self._bridge_workspace.remove(name)
            self.registry.release([f"bridge.{name}" for name in active])

    def _ensure_scalar_workspace(self) -> None:
        if self._scalar_workspace:
            return
        p, m = self._padded_shapes, self.geometry.mass_shape
        for name, shape in {"theta_padded": p["scalar"], "moisture_padded": p["scalar"], **{name: m for name in ("advection", "advx", "advy", "advz", "pdef_candidate")}}.items():
            setattr(self.scalar_rk, name, self.registry.allocate(f"scalar.{name}", shape, zero=True))
        self.scalar_rk.fluxes = {axis: self.registry.allocate(f"scalar.flux.{axis}", p[name], zero=True) for axis, name in (("x", "u3d"), ("y", "v3d"), ("z", "w3d"))}
        self._scalar_workspace = True

    def _drop_expanded_base(self) -> None:
        if self.keep_expanded_base:
            # --cuda-graphs: the three broadcast base fields stay resident, so
            # a step ends with them where it began them (cm1core.cuda_graphs:
            # a graph replays only on the inventory it ends on).  Their words
            # are the base profiles broadcast, which a drop and restore would
            # write again unchanged.
            return
        cuda_graphs.synchronize(self.stream)
        self.scalar_rk.th0 = None
        for name in ("pi0", "th0", "qv0"):
            del self.fields[name]
        self.registry.release([f"field.{name}" for name in ("pi0", "th0", "qv0")])

    def _restore_expanded_base(self) -> None:
        for name in ("pi0", "th0", "qv0"):
            if name not in self.fields:
                value = self.registry.allocate(f"field.{name}", self.geometry.mass_shape)
                self.cupy.copyto(value, self.cupy.broadcast_to(self.base_profiles[name][:, None, None], self.geometry.mass_shape))
                self.fields[name] = value
        self.scalar_rk.th0 = self.fields["th0"]

    def _require(self, phase: str, nrk: int) -> None:
        if self._expected_phase != (phase, nrk):
            raise ValueError(f"Cartesian dynamics expects {self._expected_phase}, got {(phase, nrk)}")

    def _rrw_open_boundary(self, rrw: Any, *, stream: Any, ni: int, nj: int, nk: int) -> Mapping[str, Any]:
        """CM1 solve2.F:575 bcw(rrw,0), between vertical flux and divx."""
        from .cuda_sound3_open_boundary import CM1_OPEN_NGXY, CUDA_SOUND3_OPEN_SOURCE_SHA256, CUDA_SOUND3_OPEN_THREADS

        g = self.geometry
        if (ni, nj, nk) != (g.nx, g.ny, g.nz) or stream is not self.stream or rrw is not self.arrays["rrw"]:
            raise ValueError("bcw(rrw,0) requires this exact runtime domain, stream and mass-flux object")
        if self.open_compiled.target_arch != self.momentum_advection_compiled.target_arch:
            raise ValueError("bcw and momentum images target different CUDA architectures")
        cp = self.cupy
        launch_grid = ((int(rrw.size) + CUDA_SOUND3_OPEN_THREADS - 1) // CUDA_SOUND3_OPEN_THREADS, 1, 1)
        block = (CUDA_SOUND3_OPEN_THREADS, 1, 1)
        # ngxy=3 for hadvordrv=5 (cm1.F:749); kind=3 selects the w family,
        # whose lateral loops run k=2..nk exactly as bcw does.
        arguments = (rrw, cp.uint32(ni), cp.uint32(nj), cp.uint32(nk), cp.uint32(CM1_OPEN_NGXY), cp.uint32(3))
        with stream:
            # Flag zero omits the rigid-face kernel.  Preserve EW then SN so
            # corner ghosts see the correct newly published lateral words.
            self.open_compiled.kernels[5](launch_grid, block, arguments, stream=stream)
            self.open_compiled.kernels[6](launch_grid, block, arguments, stream=stream)
        return {
            "schema": "gm1.cuda-bcw-rrw-open-flag0/v1", "performed": True, "flag": 0,
            "directions": ["west", "east", "south", "north"],
            "vertical_rigid_faces_written": False, "stream_pointer": int(stream.ptr),
            "source_sha256": CUDA_SOUND3_OPEN_SOURCE_SHA256,
            "lateral_boundaries": list(self.lateral_selection.sides),
            "halo_depth": int(CM1_OPEN_NGXY),
            # Honest scope: in this serial single-domain runtime every rrw reader
            # (divergence, advu, advv, advw) indexes the interior only, so the
            # filled lateral halo has no consumer.  The fill is maintained for
            # authority fidelity and for future MPI/nested consumers; it is not
            # claimed to change any model output here.
            "lateral_halo_has_consumer": False,
            "authority": "src/bc.F:942-1143; src/solve2.F:575",
        }

    @_on_stream
    def finalize_held_tendencies(self) -> None:
        self._require("held", 1)
        g, cp, f = self.geometry, self.cupy, self.fields
        self.child_core_compiled.kernels[0](
            ((g.nx * g.ny * g.nz + 255) // 256, 1, 1), (256, 1, 1),
            (f["th0"], f["pi0"], f["rho"], f["ppi"], f["tha"], f["qa"], self.held["thten1"], self.held["qten"], self.arrays["ppten"], *(f[name] for name in _RATES), cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz), cp.float32(self.config.dt_s)),
            stream=self.stream,
        )
        # src/solve1.F:402-409.  irbc=1 rebuilds the Klemp-Wilhelmson phase
        # speeds once per large step, from the A-level winds, exactly where
        # CM1's own pre-RK block does it.
        if self.lateral_selection.irbc == 1:
            from .cuda_boundary_families import enqueue_radbc
            from .cuda_sound3_open_boundary import CM1_OPEN_NGXY as _ng

            enqueue_radbc(
                self.arrays, ni=g.nx, nj=g.ny, nk=g.nz, ng=_ng,
                selection=self.lateral_selection, compiled=self.family_compiled,
                stream=self.stream, momentum="a",
            )
        # dyn_em/solve_em.F:933-957: WRF adds the relaxation-zone forcing once
        # per large step, at rk_step == 1, into the tendency every RK stage
        # re-reads.  CM1's held tendencies are that tendency.
        if self.nest_forcing is not None:
            self.nest_forcing.relax_held(self)
        # solve1.F:1245-1265: iinit=10's momentum forcing joins the held u
        # tendency once per large step while mtime < t2_uforce.
        if self.initiation is not None:
            self.initiation.held()
            self.last_receipts["initiation_forcing"] = self.initiation.record()
        self._expected_phase = ("pre", 1)

    @_on_stream
    def pre_acoustic(self, nrk: int) -> None:
        from .cuda_moist_stage_bridge import begin_cuda_moist_stage, launch_cuda_eqtset2_stage_terms, launch_cuda_moist_buoyancy
        from .cuda_momentum_advection5 import launch_cuda_terrain_open_momentum_advection5

        self._require("pre", nrk)
        self._release_pool()
        if nrk > 1:
            begin_cuda_moist_stage(self.moist_bridge)
        self._ensure_arrays(_PRE_PADDED)
        self._ensure_bridge(("dum8", "thterm", "divx"))
        g, cp, a, m = self.geometry, self.cupy, self.arrays, self.metrics
        launch_cuda_eqtset2_stage_terms(self.moist_bridge, a, m, th0=self.fields["th0"], qv0=self.fields["qv0"], pi0=self.fields["pi0"], th3d=self.fields["th3d"], q3d=self.mass_prefix_q3d(), stream=self.stream,
                                        qc0=self.fields.get("qc0"), qi0=self.fields.get("qi0"))
        _, bc2d = self.momentum_compiled.kernels
        pack_cartesian_held_momentum(cp, self.held, a, g, stream=self.stream)
        # adv.F's advu i1/i2, advv j1/j2 and doitw..doitn read the lateral family
        # the bcu/bcv/bcw fills apply; the sides come from those fills' binding.
        from .cuda_boundary_families import bound_lateral_sides

        lateral_sides = bound_lateral_sides(self.open_compiled)
        # solve2.F:484-487: bcwsfc and bc2d(w3d(k=1)) are terrain_flag
        # statements; a stretched (flat) column never runs them.
        if nrk >= 2 and not self.stretched:
            bc_words = max((g.ny + 6) * 3, (g.nx + 6) * 3, g.nx * g.ny)
            bc_grid = ((bc_words + 127) // 128, 1, 1)
            self.terrain_compiled.kernels[2](bc_grid, (128, 1, 1), (a["u3d"], a["v3d"], m["gz"], m["dzdx"], m["dzdy"], a["w3d"], cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz), cp.uint32(3)), stream=self.stream)
            for direction in range(4):
                bc2d(bc_grid, (128, 1, 1), (a["w3d"], cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(3), cp.uint32(direction)), stream=self.stream)
        if self.stretched:
            # adv.F:562-572 multiplies the vertical flux difference by
            # rdz*mh(k) -- ((d*rdz)*mh(1,1,k)) -- and this family binds the
            # column's own mh/mf, not a scalar dz.  solve2.F:575's
            # bcw(rrw,0) is a terrain_flag statement, so a flat column
            # passes no rrw hook.
            from .cuda_gm1_child_stretched_moist import (
                launch_cuda_gm1_flat_stretched_buoyancy,
            )
            from .cuda_gm1_child_stretched_momentum import (
                launch_cuda_gm1_child_stretched_momentum5,
            )

            momentum = launch_cuda_gm1_child_stretched_momentum5(
                a, m, self._stretched_vertical(), ni=g.nx, nj=g.ny, nk=g.nz,
                nrk=nrk, dx=g.dx_m, dy=g.dy_m, divx=self.moist_bridge.divx,
                stream=self.stream, compiled=self.stretched_compiled["momentum"],
                rrw_boundary_hook=None,
                post_divergence_hook=self._rotation_hook(nrk),
                lateral_sides=lateral_sides,
            )
            # solve2.F:897-906; the uten/vten term of :971-1008 is terrain-only.
            launch_cuda_gm1_flat_stretched_buoyancy(
                self.moist_bridge, a, m, stream=self.stream,
                compiled=self.stretched_compiled["moist"],
            )
        else:
            momentum = launch_cuda_terrain_open_momentum_advection5(a, m, ni=g.nx, nj=g.ny, nk=g.nz, nrk=nrk, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, divx=self.moist_bridge.divx, stream=self.stream, compiled=self.momentum_advection_compiled, rrw_boundary_hook=self._rrw_open_boundary, post_divergence_hook=self._rotation_hook(nrk), lateral_sides=lateral_sides)
            launch_cuda_moist_buoyancy(self.moist_bridge, a, m, stream=self.stream)
        # src/solve2.F:726-822.  The inflow nudging is one more additive term in
        # the held horizontal momentum tendency; CM1 places it inside the
        # U- and V-equation blocks, this port adds it once the stage's uten and
        # vten are assembled, which is the same sum.
        if self.lateral_selection.nudgeobc == 1:
            from .cuda_boundary_families import enqueue_nudge_open_boundary
            from .cuda_sound3_open_boundary import CM1_OPEN_NGXY as _ng

            enqueue_nudge_open_boundary(
                a, self.base_profiles, ni=g.nx, nj=g.ny, nk=g.nz, ng=_ng,
                selection=self.lateral_selection, compiled=self.family_compiled,
                stream=self.stream,
            )
        self.last_receipts["momentum"] = momentum.receipt
        # solve2.F:925-956, nrk == nrkmax on a budget step: the explicit
        # tendency and its per-process replay, while dum8 still exists.
        if nrk == 3 and self.budget_probe is not None and self.budget_probe.active:
            self.budget_probe.pre_acoustic(self)
        self._drop_bridge(("dum8",))
        self._drop_expanded_base()
        self._expected_phase = ("acoustic", nrk)

    @_on_stream
    def acoustic(self, nrk: int) -> None:
        from .cuda_boundary_families import bound_lateral_sides
        from .cuda_psolver3_terrain_open import TerrainOpenSoundAblations, launch_cuda_psolver3_terrain_open_stage

        self._require("acoustic", nrk)
        self._ensure_arrays(self._padded_shapes)
        g = self.geometry
        # The terrain/open edge snapshots and the radiation packing buffers the
        # acoustic launchers are handed.  Both launchers refuse an empty
        # inventory, and this allocation was lost from the integration merge,
        # so every large step stopped at the first acoustic stage.
        self._acoustic_workspace = {name: self.registry.allocate(f"acoustic.{name}", shape) for name, shape in terrain_open_acoustic_workspace_shapes(g.nx, g.ny, g.nz).items()}
        nest_substep = (
            None if self.nest_forcing is None else self.nest_forcing.acoustic_hook(self)
        )
        # sound.F:412-440 (iinit=9) and :564-590 (iinit=12) act inside every
        # small step, after the horizontal momentum update and before the
        # vertically implicit solve: the hook's after_momentum point.
        if self.initiation is not None:
            nest_substep = self.initiation.substep_hook(nest_substep)
        # solve2.F:1662-1706 owns these fills after the scalar RK update.
        # This uses the same deferred boundary contract as the admitted
        # RK1/RK23 live acoustic owners; the public boundary methods do them.
        if self.stretched:
            # sound.F:119-143,318-342,491-509,742-765 (terrain_flag=.false.)
            # on the generated coordinate's mh/mf.
            # There is deliberately no dz argument to this stage.  CM1's
            # off-centring weight and divergence damper (sound.F:81-82,
            # 756,782) are the document's dynamics.alph and dynamics.kdiv,
            # bound here exactly as the uniform stage below binds them; this
            # stage used to read the 0.60/0.10 constants whatever the
            # document declared.
            from dataclasses import replace as _replace

            from .cuda_gm1_child_psolver3_open import (
                DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION,
                launch_cuda_gm1_child_psolver3_open_stage,
            )

            launch = launch_cuda_gm1_child_psolver3_open_stage(
                self.arrays, self.metrics, ni=g.nx, nj=g.ny, nk=g.nz, nrk=nrk,
                dt=self.config.dt_s, nsound=self.config.acoustic_substeps,
                dx=g.dx_m, dy=g.dy_m, vertical=self._stretched_vertical(),
                # The operator reads the generated coordinate for every
                # param.F:7814-8139 branch; the selection is what its receipt
                # reports, so it must name this grid's branch, not the
                # default stretch_z=1 of the sealed child.
                selection=_replace(
                    DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION, stretch_z=g.stretch_z
                ),
                ablations=TerrainOpenSoundAblations(stage_family_fills=False),
                stream=self.stream, compiled=self.stretched_compiled["psolver"],
                open_compiled=self.open_compiled,
                lateral_compiled=self.lateral_compiled,
                workspace=self._acoustic_workspace,
                nest_substep=nest_substep,
                alph=self.config.alph,
                kdiv=self.config.kdiv,
                # sound.F:274-302 and param.F:9182-9219: ssopenbc on open
                # sides only, and the pressure gradient on a periodic side's
                # boundary face -- the family the bound fills apply.
                lateral_sides=bound_lateral_sides(self.open_compiled),
            )
        else:
            launch = launch_cuda_psolver3_terrain_open_stage(self.arrays, self.metrics, ni=g.nx, nj=g.ny, nk=g.nz, nrk=nrk, dt=self.config.dt_s, nsound=self.config.acoustic_substeps, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, ablations=TerrainOpenSoundAblations(stage_family_fills=False), stream=self.stream, compiled=self.psolver_compiled, terrain_compiled=self.terrain_compiled, open_compiled=self.open_compiled, lateral_compiled=self.lateral_compiled, lateral=self.lateral_step, workspace=self._acoustic_workspace, alph=self.config.alph, kdiv=self.config.kdiv, nest_substep=nest_substep)
        self.last_receipts["acoustic"] = launch.receipt
        cuda_graphs.synchronize(launch.ready_event)
        del launch
        names = list(self._acoustic_workspace)
        self._acoustic_workspace.clear()
        self.registry.release([f"acoustic.{name}" for name in names])
        self._expected_phase = ("post", nrk)

    @_on_stream
    def post_acoustic(self, nrk: int) -> None:
        from .cuda_moist_stage_bridge import launch_cuda_post_acoustic_flux_bridge

        self._require("post", nrk)
        # The post-acoustic bridge validates the pre-acoustic carrier family,
        # but no longer consumes implicit-column/acoustic diagnostic scratch.
        self._drop_padded([name for name in self.arrays if name not in (*_PERSISTENT_PADDED, *_PRE_PADDED)])
        self._ensure_bridge(("divergence", "theta_divergence_term", "divx"))
        g = self.geometry
        if self.stretched:
            # solve2.F:1278-1307: misclibs.F's flat getdiv/getdivx, with
            # ((d*rdz)*mh(1,1,k)) and rho0(1,1,k)/rf0(1,1,k).
            from .cuda_gm1_child_stretched_moist import (
                launch_cuda_gm1_child_stretched_post_acoustic,
            )

            launch = launch_cuda_gm1_child_stretched_post_acoustic(
                self.moist_bridge, self.arrays, self.metrics,
                self._stretched_vertical(), dx=g.dx_m, dy=g.dy_m,
                stream=self.stream, compiled=self.stretched_compiled["moist"],
            )
        else:
            launch = launch_cuda_post_acoustic_flux_bridge(self.moist_bridge, self.arrays, self.metrics, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, stream=self.stream)
        self.last_receipts["post_acoustic"] = launch.receipt
        self.sync_physical_from_padded()
        self._drop_bridge(("thterm", "divergence"))
        self.release_phase_workspace("acoustic")
        self._restore_expanded_base()
        self._expected_phase = ("scalar", nrk)

    @_on_stream
    def scalar_moist(self, nrk: int) -> None:
        from .cuda_scalar_moist_rk import launch_cuda_scalar_moist_rk_stage

        self._require("scalar", nrk)
        self._ensure_scalar_workspace()
        g = self.geometry
        mass_flux = {name: self.arrays[name] for name in ("rru", "rrv", "rrw")}
        if self.stretched:
            # adv.F:562-572,681-695's flat vertical flux divergence
            # ((d*rdz)*mh(1,1,k)), and pdef.F's flat final-stage chain.
            from .cuda_gm1_child_large_step import Gm1ChildScalarMoistRKBinding
            from .cuda_gm1_child_step_composer import (
                _launch_cuda_gm1_child_stretched_scalar_stage,
            )

            binding = Gm1ChildScalarMoistRKBinding(
                self.scalar_rk, self.scalar_compiled,
                self.scalar_pack_compiled, self.scalar_legacy_compiled,
            )
            launch = _launch_cuda_gm1_child_stretched_scalar_stage(
                binding, mass_flux, self.metrics, self.moist_bridge.divx,
                self.moist_bridge.theta_divergence_term,
                self._stretched_vertical(), nrk=nrk, dt=self.config.dt_s,
                dx=g.dx_m, dy=g.dy_m, stream=self.stream,
                advection_compiled=self.stretched_compiled["advection"],
                pdef_compiled=self.stretched_compiled["pdef"],
                open_compiled=self.open_compiled,
            )
        else:
            launch = launch_cuda_scalar_moist_rk_stage(self.scalar_rk, mass_flux, self.metrics, self.moist_bridge.divx, self.moist_bridge.theta_divergence_term, nrk=nrk, dt=self.config.dt_s, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m, stream=self.stream, compiled=self.scalar_compiled, advection_compiled=self.scalar_advection_compiled, open_compiled=self.open_compiled)
        self.last_receipts["scalar"] = launch.receipt
        # dyn_em/solve_em.F:2277,2305: the specified rows of every RK scalar
        # advance by the boundary tendency alone, from the stage's time-level-n
        # field.  psolver=3 advances theta here, not in the acoustic loop.
        if self.nest_forcing is not None:
            self.nest_forcing.specify_scalars(self, nrk)
        self.release_phase_workspace("scalar")
        self._drop_bridge(("theta_divergence_term",))
        self._expected_phase = ("momentum-boundary", nrk)

    @_on_stream
    def momentum_boundaries(self, nrk: int) -> None:
        self._require("momentum-boundary", nrk)
        g, cp, a, m = self.geometry, self.cupy, self.arrays, self.metrics
        kfwe, kfsn, rigid = self.open_compiled.kernels[5:8]
        words = max(int(a[name].size) for name in ("u3d", "v3d", "w3d"))
        grid, block = ((words + 255) // 256, 1, 1), (256, 1, 1)
        common = (cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz), cp.uint32(3))
        kfwe(grid, block, (a["u3d"], *common, cp.uint32(1)), stream=self.stream)
        kfsn(grid, block, (a["u3d"], *common, cp.uint32(1)), stream=self.stream)
        kfsn(grid, block, (a["v3d"], *common, cp.uint32(2)), stream=self.stream)
        kfwe(grid, block, (a["v3d"], *common, cp.uint32(2)), stream=self.stream)
        kfwe(grid, block, (a["w3d"], *common, cp.uint32(3)), stream=self.stream)
        kfsn(grid, block, (a["w3d"], *common, cp.uint32(3)), stream=self.stream)
        rigid(grid, block, (a["w3d"], *common), stream=self.stream)
        # solve2.F:1676: bcwsfc only with terrain_flag; a stretched (flat)
        # column keeps bcw's rigid w(1)=0.
        if not self.stretched:
            edge_words = (g.nx + 2) * (g.ny + 2)
            self.boundary_compiled.bcwsfc_kernel(((edge_words + 255) // 256, 1, 1), block, (a["u3d"], a["v3d"], a["w3d"], m["gz"], m["dzdx"], m["dzdy"], *common), stream=self.stream)
        self.sync_physical_from_padded(momentum_only=True)
        self._expected_phase = ("scalar-boundary", nrk) if nrk < 3 else ("finished-rk", 3)

    @_on_stream
    def scalar_boundaries(self, nrk: int) -> None:
        """solve2.F:1662-1706's pressure fill for the next RK stage.

        CM1 there also fills rho, theta and every q3d row.  This lane keeps
        those fields as physical rectangles; this phase packed them into three
        scratch volumes of its own, filled their halos and released all three
        without a reader, and the scalar RK stage packs theta and each row into
        its own workspace and fills those halos itself.  Only the ``pp3d`` bcp
        fills have a reader, and they remain, in the same order.
        """

        self._require("scalar-boundary", nrk)
        g, cp = self.geometry, self.cupy
        common = (cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz))
        value = self.arrays["pp3d"]
        grid = ((int(value.size) + 127) // 128, 1, 1)
        for kernel in self.open_compiled.kernels[3:5]:
            kernel(grid, (128, 1, 1), (value, *common), stream=self.stream)
        self._expected_phase = ("pre", nrk + 1)

    @_on_stream
    def sync_physical_from_padded(self, *, momentum_only: bool = False) -> None:
        g, cp = self.geometry, self.cupy
        # Physical winds are views of the canonical padded owners already.
        if not momentum_only:
            cp.copyto(self.fields["pp3d"], self.arrays["pp3d"][g.physical_slices("mass", halo_xy=1, halo_z=1)])

    @_on_stream
    def sync_padded_from_physical(self) -> None:
        """Publish completed solve3 current/A words before a new large step."""

        from .cuda_moist_stage_bridge import begin_cuda_moist_stage
        from .cuda_scalar_moist_rk import begin_cuda_scalar_moist_rk_step

        if self._expected_phase != ("finished-rk", 3):
            raise ValueError("physical-to-padded commit requires all three completed RK stages")
        g, cp = self.geometry, self.cupy
        for current, baseline, stagger in (("u3d", "ua", "u"), ("v3d", "va", "v"), ("w3d", "wa", "w")):
            cp.copyto(self.arrays[baseline], self.arrays[current])
        for name in _PRESSURE:
            cp.copyto(self.arrays[name][g.physical_slices("mass", halo_xy=1, halo_z=1)], self.fields[name])
            for kernel in self.open_compiled.kernels[3:5]:
                kernel(((int(self.arrays[name].size) + 127) // 128, 1, 1), (128, 1, 1), (self.arrays[name], cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)), stream=self.stream)
        begin_cuda_moist_stage(self.moist_bridge)
        begin_cuda_scalar_moist_rk_step(self.scalar_rk)
        self._expected_phase = ("held", 1)

    @_on_stream
    def release_phase_workspace(self, phase: str = "all") -> None:
        """Drop only completed phase scratch; current/A/held/history remain resident."""

        if phase not in ("all", "acoustic", "scalar"):
            raise ValueError("phase must be all, acoustic, or scalar")
        if phase in ("all", "acoustic"):
            names = [name for name in self.arrays if name not in _PERSISTENT_PADDED]
            self._drop_padded(names)
        if phase in ("all", "scalar") and self._scalar_workspace:
            names = ("theta_padded", "moisture_padded", "advection", "advx", "advy", "advz", "pdef_candidate")
            for name in names:
                setattr(self.scalar_rk, name, None)
            self.scalar_rk.fluxes.clear()
            # The registry still owns every one of these, so dropping this
            # module's references frees nothing and allocates nothing.
            # ``registry.release`` drains the stream before it drops them,
            # which is the barrier this used to take a second time.
            self.registry.release([*(f"scalar.{name}" for name in names), *(f"scalar.flux.{axis}" for axis in ("x", "y", "z"))])
            self._scalar_workspace = False
        self._release_pool()

    def _release_pool(self) -> None:
        """Return cached pool blocks to the driver only under ``phase``."""

        if getattr(self, "pool_release", "keep") == "phase":
            self.cupy.get_default_memory_pool().free_all_blocks()

    def _drop_padded(self, names: list[str]) -> None:
        if names:
            for name in names:
                del self.arrays[name]
            # Same as above: the registry is the owner, so nothing is freed
            # between here and the drain inside ``registry.release``.
            self.registry.release([f"padded.{name}" for name in names])


__all__ = (
    "CartesianDynamics",
    "DYNAMICS_POOL_RELEASE_ENVIRONMENT",
    "DYNAMICS_POOL_RELEASE_MODES",
    "cartesian_dynamics_allocation_plan",
    "dynamics_pool_release_mode",
    "pack_cartesian_held_momentum",
)
