"""A fresh, configurable Cartesian GM1 CUDA execution path.

The source initializer and all numerical kernels are explicitly bound.  The
outer scheduler follows CM1's stage order; validation/reference executions
are separate runs, not copies retained inside the production state graph.
"""

from __future__ import annotations

from dataclasses import replace
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import MappingProxyType
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from . import engine_pin
from .arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    ArwenWsm6Carriers,
    create_arwen_wsm6_component,
)
from .column_components import column_component_spec
from .cuda_gm1_tilestream_column import column_component_class
from .background_publication import (
    BackgroundPublisher,
    EventPublication,
    PublicationCancel,
    PublicationPart,
    TerminationRequested,
    attach_taken_parts,
    blocking_overlap,
    run_event,
    taken_parts,
    termination_as_exception,
)
from .cartesian_history import (
    COMPLETE_CARRIER_PROFILE,
    CartesianHistoryProfile,
    cartesian_history_profile,
    commit_cartesian_history,
    initial_frame_history_profile,
    publish_cartesian_history,
    require_forecast_publisher,
    require_history_carrier_profile,
    require_new_history_directory,
    snapshot_cartesian_history,
)
from .cartesian_restart import (
    RESTART_DIRECTORY_FORMAT,
    CartesianRestartProfile,
    RestoredCartesianRestart,
    cartesian_restart_profile,
    commit_cartesian_restart,
    read_cartesian_restart,
    restore_cartesian_restart,
    snapshot_cartesian_restart,
    write_cartesian_restart,
    write_restart_receipt,
)
from .forecast_arena import HostArena
from .cuda_graphs import active_capture
from .cuda_cartesian_closure import CartesianClosure, cartesian_closure_allocation_plan
from .cuda_cartesian_dynamics import CartesianDynamics, cartesian_dynamics_allocation_plan
from .cartesian_output_products import CartesianOutputProducts
from .cuda_cartesian_finish import CartesianFinish, cartesian_wsm6_scratch_reserve
from .gm1_idealized_residency import DeviceBufferRegistry, IdealizedSource
from .gm1_runtime import (
    CM1_COURANT_STOP,
    CM1_COURANT_STOP_AUTHORITY,
    CM1_COURANT_TARGET,
    FIRST_RESTART_NUMBER,
    RunSchedule,
    RuntimeClock,
    RuntimeConfig,
    ScheduledOperation,
    StepPhase,
    advective_courant_defect,
    history_cadence,
    run_schedule,
)
from .cm1_adaptive_timestep import (
    CM1_ADAPTIVE_TIMESTEP_AUTHORITY,
    RampDownTarget,
    cm1_diffusion_number,
    cm1_dtsmall,
    cm1_getnewdt,
    cm1_ks_limit,
)
from .dynamics_validation_register import (
    dynamics_validation_record as runtime_dynamics_validation_record,
)
from .lagrangian_stage import UNVALIDATED_MARKERS
from .storm_diagnostics import advective_courant_number


INTEGRATION_SCHEMA = "gm1.cartesian-runtime-integration/v1"
INTEGRATION_STATUS = "PASS_EXECUTED_CARTESIAN_INTEGRATION"
# misclibs.F:2714 reached mid-run.  A separate status word, because a run CM1
# would have abandoned must never share the word a completed one carries.
INTEGRATION_COURANT_STOP_STATUS = "STOPPED_AT_CM1_ADVECTIVE_COURANT_STOP"
# A declared [nest.tornado_criterion] stop rule ended the run on purpose.  Its
# own word, because the requested duration was not integrated.
INTEGRATION_CRITERION_STOP_STATUS = "STOPPED_AT_NEST_TORNADO_CRITERION_STOP_RULE"
# A run-control stop or pause file ended the run on request
# (cm1core.run_control).  Their own words, because the requested duration was
# not integrated; a paused run ends on a durable restart it can continue from.
INTEGRATION_CONTROL_STOP_STATUS = "STOPPED_AT_RUN_CONTROL_REQUEST"
INTEGRATION_CONTROL_PAUSE_STATUS = "PAUSED_AT_RUN_CONTROL_REQUEST"
# The model clock is a binary64 running sum under adapt_dt, so "has this frame
# time been reached" needs a tolerance rather than an equality.  One part in
# 1e9 of a second is far below any cadence and far above the sum's rounding.
_ADAPTIVE_CLOCK_EPSILON_S = 1.0e-9
# misclibs.F:654 caps the step at twice dtl and nothing caps it from below, so
# a diverging run can shrink dt without bound.  This turns that into a named
# failure instead of an integration that never reaches its next frame.
_ADAPTIVE_STEPS_PER_FRAME_LIMIT = 100_000


LAGRANGIAN_TRACK_DIRECTORY = "lagrangian-track"


class CartesianCourantStop(RuntimeError):
    """CM1's advective Courant stop, raised after the tripping frame is written.

    ``misclibs.F:2714`` sets ``stopit``; ``cm1.F:3517-3520`` then forces
    ``dostat`` and ``dowriteout`` so the statistics and the history output for
    the state that tripped it are written, and only ``cm1.F:4385-4392`` prints
    "Courant number has exceeded 1.5" and stops the model.  This exception is
    raised at that last point and never before: every frame up to and including
    the one that reached the stop is already published, and ``receipt`` is the
    complete integration receipt for them, so a caller keeps both the frame a
    user needs to see the diverging state and the record naming the step count
    and model time it happened at.
    """

    def __init__(self, message: str, *, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


# Frames are published under an ordinal name whose lexical order is their
# model order, so a watcher globbing "frame-*" reads them in time order.
FRAME_DIRECTORY_FORMAT = "frame-{index:06d}"
# Each frame is written inside this dot-prefixed staging subdirectory and then
# renamed into the root.  A watcher globbing "frame-*" therefore never opens a
# directory that is still being written.
FRAME_STAGING_DIRECTORY = ".incomplete"
# cartesian_history records the absolute paths its bytes were written to, which
# are staging paths.  This file is written inside the frame before the rename,
# so the published directory also names where its own artifacts finally live.
FRAME_LOCATION_FILENAME = "history-frame-location.json"
FRAME_LOCATION_SCHEMA = "gm1.cartesian-history-frame-location/v1"
# cm1.F:4341-4350.  A run that only builds the initial condition publishes the
# same two artifacts a restart cadence would, at nstep=0, and integrates
# nothing.
INITIAL_CONDITION_SCHEMA = "gm1.cartesian-initial-condition/v1"
INITIAL_CONDITION_STATUS = "PUBLISHED_CARTESIAN_INITIAL_CONDITION"
# The frame index an initial-condition history frame is published under, so it
# sorts before every integrated frame in the same directory.
INITIAL_CONDITION_FRAME_INDEX = 0
MEMORY_SAMPLE_KEYS = (
    "device_used_bytes",
    "registered_model_bytes",
    "pool_used_bytes",
    "pool_reserved_bytes",
)


@contextmanager
def _active_io(runtime: Any, event: Any) -> Iterator[Any]:
    """Make ``event`` the runtime's active I/O event for the enclosed snapshot."""

    if event is None:
        yield None
        return
    previous = getattr(runtime, "active_io_event", None)
    runtime.active_io_event = event
    event.profiler.range_push(f"io.{event.kind}")
    try:
        yield event
    finally:
        runtime.active_io_event = previous
        event.profiler.range_pop()


def _discard_staging(path: Path) -> None:
    """Remove a staging directory a publication did not finish.

    Only ever a path under a dot-prefixed staging root: the published name is
    never created until the last rename, so nothing here can touch a
    complete frame or restart.
    """

    import shutil

    if path.is_symlink() or not path.exists():
        return
    shutil.rmtree(path)


def _notice(stream: Any, line: str) -> None:
    """One operator-facing line; a closed or broken stream is not a failure."""

    import sys

    try:
        print(line, file=sys.stderr if stream is None else stream, flush=True)
    except (OSError, ValueError, AttributeError):
        pass


def _startup_record(runtime: Any, profiler: Any) -> dict[str, Any]:
    """What start-up cost, for the ``--events`` ``startup`` record.

    The profiler's start-up stages and NVRTC compile rows when a profiler
    runs (``--profile summary``, the default); otherwise only the device.
    ``seconds`` is the sum of the recorded stages.
    """

    stages: dict[str, float] = {}
    compile_s = programs = None
    if profiler is not None:
        for name, row in getattr(profiler, "startup", {}).items():
            try:
                stages[str(name)] = float(row["seconds"])
            except (KeyError, TypeError, ValueError):
                continue
        rows = getattr(profiler, "compile_rows", None)
        if rows:
            compile_s = float(sum(row[0] for row in rows.values()))
            programs = int(sum(row[1] for row in rows.values()))
    gpu = None
    try:
        properties = runtime.cp.cuda.runtime.getDeviceProperties(
            runtime.cp.cuda.runtime.getDevice()
        )
        name = properties["name"]
        uuid = properties.get("uuid")
        if isinstance(uuid, (bytes, bytearray)) and len(uuid) == 16:
            text = uuid.hex()
            # nvidia-smi's spelling, which --gpu-uuid takes.
            uuid = (
                f"GPU-{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"
            )
        gpu = {
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "uuid": None if uuid is None else str(uuid),
        }
    except Exception:  # noqa: BLE001 - the record names the device when it can
        gpu = None
    return {
        "stages": stages,
        "nvrtc_compile_s": compile_s,
        "nvrtc_programs": programs,
        "seconds": float(sum(stages.values())),
        "gpu": gpu,
    }


def settle_publication_io(event: EventPublication) -> None:
    """Close each part's profiler event on the stepping thread.

    ``seconds`` is the publication's own work -- the snapshot on the stepping
    thread plus the commit on the worker, not time spent queued behind an
    earlier part, nor waiting beside it for its turn to rename -- and
    ``blocking_seconds`` is what the stepping thread actually paid: the
    snapshot plus the part of any join that waited on this commit and on no
    earlier part.  A synchronous publication blocks for all of it.
    """

    for part in event.parts:
        io = part.io
        if io is None or io.finished:
            continue
        commit = 0.0
        if part.started is not None and part.finished is not None:
            commit = part.finished - part.started
        if part.turn_wait_seconds is not None:
            # Committed beside an earlier part of the event: the seconds it
            # then waited to rename after it are that part's, not this one's.
            commit = max(0.0, commit - part.turn_wait_seconds)
            io.add("background.turn_wait", part.turn_wait_seconds)
        if event.background:
            waited = blocking_overlap(part, event)
            io.add("background.join_wait", waited)
            io.finish(
                seconds=part.snapshot_seconds + commit,
                blocking_seconds=part.snapshot_seconds + waited,
                background=True,
            )
        else:
            io.finish(
                seconds=part.snapshot_seconds + commit,
                blocking_seconds=part.snapshot_seconds + commit,
            )


@contextmanager
def profiled_io(runtime: Any, kind: str, label: str) -> Iterator[Any]:
    """Record one publication's stages when ``runtime`` is profiled.

    Unprofiled runtimes -- and host doubles that carry no profiler attribute at
    all -- get ``None`` and run the publication exactly as before.
    """

    profiler = getattr(runtime, "profiler", None)
    event = None if profiler is None else profiler.io_event(kind, label)
    if event is None:
        yield None
        return
    previous = getattr(runtime, "active_io_event", None)
    runtime.active_io_event = event
    profiler.range_push(f"io.{kind}")
    try:
        yield event
    finally:
        runtime.active_io_event = previous
        profiler.range_pop()
        event.finish()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _warm_public_component(cp: Any, stream: Any, config: RuntimeConfig, capacity_profile: str | None) -> dict[str, Any]:
    """Materialize the driver's column-frame backing before sizing model RAM.

    A positive dry column compiles/loads the selected component.  Its output
    is discarded and is never used to initialize the model atmosphere.
    """
    nz = config.grid.nz
    # The component the run will execute, and its own carrier set: CM1's q3d
    # axis for that ptype (moisture_species) plus any interface carrier the
    # scheme's launcher reads (Thompson's w).  Warming a different scheme
    # would size the driver backing for kernels this run never loads, and
    # the WSM6 constructor refuses any other component id by name.
    spec = column_component_spec(config.microphysics)
    with stream:
        shape = (nz, 1, 1)
        thermodynamic = dict(
            thp=cp.full(shape, 300, dtype=cp.float32),
            thb=cp.zeros((nz,), dtype=cp.float32),
            p=cp.full(shape, 100000, dtype=cp.float32),
            alt=cp.ones(shape, dtype=cp.float32),
            phb=cp.arange(nz + 1, dtype=cp.float32) * cp.float32(config.grid.dz_m) * cp.float32(9.81),
            php=cp.zeros((nz + 1, 1, 1), dtype=cp.float32),
        )
        if spec.component_id == ARWEN_WSM6_COMPONENT_ID:
            # Exactly the construction every WSM6 run has always warmed.
            carriers = ArwenWsm6Carriers(
                **thermodynamic,
                **{name: cp.zeros(shape, dtype=cp.float32) for name in ("qv", "qc", "qr", "qi", "qs", "qg")},
            )
            component = create_arwen_wsm6_component(
                component_id=config.microphysics, nx=1, ny=1, nz=nz,
                dx_m=config.grid.dx_m, dy_m=config.grid.dy_m, dz_m=config.grid.dz_m,
                dt_s=config.dt_s, capacity_profile=capacity_profile,
            )
        else:
            carriers = spec.carriers_type(
                **thermodynamic,
                **{name: cp.zeros((nz + 1, 1, 1), dtype=cp.float32)
                   for name in spec.extra_interface_carriers},
                **{name: cp.zeros(shape, dtype=cp.float32) for name in spec.species},
            )
            component = column_component_class(spec.component_id)(
                component_id=config.microphysics, nx=1, ny=1, nz=nz,
                dx_m=config.grid.dx_m, dy_m=config.grid.dy_m, dz_m=config.grid.dz_m,
                dt_s=config.dt_s, capacity_profile=capacity_profile,
                ihail=config.ihail,
            )
        result = component.apply(carriers, refl_10cm_due=False)
        result.ready_event.synchronize()
        record = {"levels": nz, "component_receipt": dict(result.receipt),
                  "model_initialization": False, "model_steps": 0}
        del result, component, carriers
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    free, total = cp.cuda.runtime.memGetInfo()
    record.update(free_after_bytes=int(free), device_total_bytes=int(total),
                  driver_and_context_bytes=int(total - free))
    return record



def _graph_staging_allocator(runtime: Any) -> Any:
    """Under CUDA graphs, a frame's or restart's device staging uses the graphs' pool.

    A function of the runtime rather than a method, so every runtime that
    borrows the publication methods (the host doubles among them) has it.
    """

    graphs = getattr(runtime, "schedule_graphs", None)
    if graphs is None:
        driver = getattr(runtime, "nest_driver", None)
        child = None if driver is None else getattr(driver, "child", None)
        graphs = None if child is None else getattr(child, "schedule_graphs", None)
    if graphs is None:
        return contextlib.nullcontext()
    return graphs.memory.allocating()

class CudaCartesianRuntime:
    # Class defaults so a shell built with ``object.__new__`` -- and any
    # caller that never asked for bounded sampling -- reads the historical
    # unbounded behaviour instead of an AttributeError.
    memory_sample_budget: int | None = None
    memory_samples_dropped: int = 0
    _memory_peaks: Any = MappingProxyType({})
    # As above: a shell built with ``object.__new__`` reads an empty register
    # rather than an AttributeError, and a real runtime overwrites it in
    # __init__ with the rows its own configuration selects.
    dynamics_validation: Any = MappingProxyType({})
    # CM1's output_* diagnostics.  None until an entry point that knows the
    # run's step count builds them, and None for the whole life of a run that
    # declared no [diagnostics] table: the swaths are running extrema over a
    # known number of steps, so nothing can be folded before that is known.
    tornado: Any = None
    # A parent restart's swath words (cartesian_restart.read_restored_swaths),
    # held from the restore until open_tornado_diagnostics sizes the swaths
    # for the resumed run and continues them (restart_read.F:1064-1126).
    restored_swaths: Any = None
    # Class defaults for the same reason the three above have them: a shell
    # built with ``object.__new__`` -- which the lane tests use to drive
    # ``integrate`` without a device -- never runs ``__init__``, and an
    # AttributeError there would be a failure of this diagnostic stage rather
    # than of the thing under test.  The two frame lists default to an
    # immutable empty tuple, which is safe to share: nothing appends unless
    # ``_lagrangian_root`` is set, and only ``__init__`` sets it.
    lagrangian: Any = None
    # CM1's output_*budget terms and the GM1 vorticity budget, or None when
    # the document selects neither (cartesian_budgets.build_budget_stage).
    budget: Any = None
    parcel_frames: Any = ()
    lagrangian_frames: Any = ()
    _lagrangian_root: Any = None
    _budget_compiled: Any = None
    # cm1.F:371's own default.  A runtime that was never handed an adaptive
    # moving-domain controller integrates the fixed frame its configuration
    # declares, which is every run this port drove before the controller was
    # reachable from a document.
    adaptive_frame: Any = None
    # The completed-step numbers this run publishes a history frame at.
    # cm1.F:3642-3651 gates the storm-centre diagnosis on ``dowriteout``, so
    # the controller needs the same set the frame cadence resolved.
    history_frame_steps: frozenset[int] = frozenset()
    # A live nest's parent carries a cm1core.nest_live.NestDriver here, and
    # every completed parent step then forces, integrates and (with
    # feedback = 1) restricts its child before the step is reported complete
    # (frame/module_integrate.F:393-443).  None for every single-domain run.
    nest_driver: Any = None
    # cm1core.runtime_profiler.RuntimeProfiler, or None.  None -- the default
    # for every shell and for ``--profile off`` -- takes exactly the code path
    # this runtime always took; a profiler only records stream markers and
    # host clocks around the same operations, never a model word.
    profiler: Any = None
    # cm1core.cuda_graphs.ScheduleGraphs under --cuda-graphs on/verify, or
    # None: every large step's schedule then runs in Python exactly as before.
    schedule_graphs: Any = None
    # The I/O event a profiled publication is being recorded into; the history
    # and restart writers read it so their stages land in one event.
    active_io_event: Any = None
    startup_wall_seconds: Any = MappingProxyType({})

    def __init__(self, source: IdealizedSource, config: RuntimeConfig, *, cupy: Any,
                 stream: Any, capacity_profile: str | None = None) -> None:
        # A non-default component adds its own kernels to the audited surface
        # (engine_pin.component_extra_surface); WSM6 inspects exactly the
        # surface it always has.
        component_surface = (
            {} if config.microphysics == ARWEN_WSM6_COMPONENT_ID
            else {"component_id": config.microphysics}
        )
        inspection = engine_pin.inspect(capacity_profile=capacity_profile, **component_surface)
        if not inspection.admitted:
            raise RuntimeError(
                f"the selected {config.microphysics} dependency bytes do not match "
                "the requested runtime profile"
            )
        self.cp, self.stream = cupy, stream
        self.capacity_profile = capacity_profile
        self.trace_phases = False
        # Host wall clock of each construction stage (compile, allocation and
        # upload happen inside them).  Recorded for every runtime -- a handful
        # of clock reads at startup -- and published only by a profiled run.
        startup = self.startup_wall_seconds = {}
        tick = time.perf_counter()
        self.warm_component = _warm_public_component(cupy, stream, config, capacity_profile)
        startup["component_warmup"] = time.perf_counter() - tick
        # Price against the actual initialized CUDA driver, including its
        # per-thread local-memory backing.  The registry is only model-owned
        # storage; counting its allocations as total VRAM would be incorrect.
        measured_free = self.warm_component["free_after_bytes"]
        self.component_scratch_reserve = cartesian_wsm6_scratch_reserve(config)
        available = measured_free - 128 * 1024**2 - self.component_scratch_reserve["additional_reserve_bytes"]
        budget = available if config.memory_budget_bytes is None else min(config.memory_budget_bytes, available)
        if budget <= 0:
            raise MemoryError("CUDA context and component backing leave no model memory budget")
        self.config = replace(config, memory_budget_bytes=int(budget))
        self.registry = DeviceBufferRegistry(cupy, stream, budget_bytes=int(budget))
        self.source_identity = {
            "kind": "CM1-r21.1-Rust-sounding-warm-bubble",
            "sounding_sha256": source.sounding_sha256,
            "initializer_library_sha256": _sha_file(source.library),
            "geometry": config.grid.record(),
        }
        # Preserve the actual loaded source estate, including any forward
        # fixes made after a transport archive was created.
        package_root = Path(__file__).resolve().parent
        source_hashes = {path.name: _sha_file(path) for path in sorted(package_root.glob("*.py"))}
        self.runtime_source_identity = {
            "python_files": source_hashes,
            "aggregate_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True).encode()
            ).hexdigest(),
        }
        identity_sha = hashlib.sha256(json.dumps(self.source_identity, sort_keys=True).encode()).hexdigest()
        startup["source_identity_hashing"] = time.perf_counter() - tick - startup["component_warmup"]
        tick = time.perf_counter()
        self.dynamics = CartesianDynamics(source, self.config, cupy=cupy, stream=stream, registry=self.registry)
        startup["dynamics_construction"] = time.perf_counter() - tick
        # Retained only as an explicit source owner; source views cannot
        # outlive the enclosing IdealizedSource context.
        self.dynamics.source = source
        tick = time.perf_counter()
        self.closure = CartesianClosure(self.dynamics)
        startup["closure_construction"] = time.perf_counter() - tick
        tick = time.perf_counter()
        self.finish = CartesianFinish(self.dynamics, source_identity_sha256=identity_sha,
                                      capacity_profile=capacity_profile)
        startup["finish_construction"] = time.perf_counter() - tick
        tick = time.perf_counter()
        # CM1 &param9.  Allocate the run-long swaths and the reflectivity
        # volume before the first step, seed them with CM1's own init3d.F
        # sentinels, and bind the microphysics reflectivity sink.  A selection
        # this grid cannot carry must fail here, not at the first frame.
        self.output_fields: dict[str, Any] = {}
        self.output = CartesianOutputProducts(self)
        self.finish.reflectivity_sink = self.output.accept_reflectivity
        self._reflectivity_due = False
        self.clock = RuntimeClock(self.config)
        self.dynamics_validation = runtime_dynamics_validation_record(self.config)
        self.memory_samples: list[dict[str, Any]] = []
        self.memory_samples_dropped = 0
        self._memory_peaks = dict.fromkeys(MEMORY_SAMPLE_KEYS, 0)
        self.phase_wall_seconds: dict[str, float] = {}
        # cm1.F:3214-3237.  CM1 raises ``getdbz`` on the step whose completed
        # state is about to be written and leaves it down otherwise, so the
        # reflectivity diagnostic costs nothing on the steps nobody publishes.
        # An empty schedule is "never due", which is what a caller that steps
        # by hand and publishes nothing means; ``integrate`` and ``benchmark``
        # set it from their own frame cadence.
        self.reflectivity_frame_steps: frozenset[int] = frozenset()
        # CM1 calls parcel_driver from solve3.F:768-769, once per large step,
        # with the acoustic small-step-averaged rru/rrv/rrw it saved at
        # solve2.F:1290-1301.  This stage is that call site; it reads those
        # carriers and writes none of them, so no selector in [lagrangian] can
        # change the simulated storm.
        # CM1 &param9 output_*budget and the [lagrangian] vorticity-budget
        # extensions.  Built before the Lagrangian stage, whose parcel table
        # gains the vorticity-budget columns when they are sampled there.
        self.budget = self._build_budget_stage()
        self.lagrangian = self._build_lagrangian_stage()
        self.parcel_frames: list[dict[str, Any]] = []
        self.lagrangian_frames: list[dict[str, Any]] = []
        self._lagrangian_root: Path | None = None
        self._sample("initialized", step=0)
        startup["output_budget_lagrangian_construction"] = time.perf_counter() - tick

    def _build_lagrangian_stage(self) -> Any:
        selectors = getattr(self.config, "lagrangian", None)
        if selectors is None:
            return None
        from .lagrangian_parcels import parcel_switch_context
        from .lagrangian_stage import LagrangianStage
        from .nest_vertical import levels_of_grid

        c = self.config
        g = c.grid
        stage = LagrangianStage.build(
            selectors, nx=g.nx, ny=g.ny, nz=g.nz,
            dx_m=g.dx_m, dy_m=g.dy_m, dz_m=g.dz_m,
            dt_s=c.dt_s, xp=self.cp,
            # param.F:7212-7213's frame, the one the door checked extents in;
            # the grid's own CM1 deck, which parcel.F:251-260 searches.
            iorigin=int(getattr(c, "iorigin", 1)), levels=levels_of_grid(g),
            bbc=c.bbc, tbc=int(getattr(c, "tbc", 1)),
            u_move_m_s=c.u_move_m_s, v_move_m_s=c.v_move_m_s,
            imove=int(getattr(c, "imove", int(c.u_move_m_s != 0.0 or c.v_move_m_s != 0.0))),
            lateral=tuple(int(getattr(c, name, 2)) for name in ("wbc", "ebc", "sbc", "nbc")),
            # param.F:6257-6282 resolves the prcl_* switches against this run's
            # own words (moisture axis, closure, bbc) before :6385-6500 lays out
            # a column; nest_live hands the child tracker the same context.
            switch_context=parcel_switch_context(c),
        )
        budget = getattr(self, "budget", None)
        if budget is not None and budget.parcel_vorticity:
            # GM1 extension columns after CM1's attributes, sampled by
            # nest_parcels.sample_mass_rectangles on this grid's own deck --
            # the extension point NestParcelTracker hooks use.
            stage.register_hook(budget.parcel_hook())
        if selectors.parcels_enabled:
            stage.seed()
        return stage

    def _build_budget_stage(self) -> Any:
        from .cartesian_budgets import build_budget_stage

        registry = self.registry

        def allocate(name: str, shape: tuple[int, ...]) -> Any:
            return registry.allocate(name, shape, zero=True)

        stage = build_budget_stage(self.config, xp=self.cp, allocate=allocate)
        if stage is not None:
            self.dynamics.budget_probe = stage
            self.closure.budget_probe = stage
        return stage

    def _budget_due(self, step: int) -> bool:
        """``cm1.F:3268-3296``: budgets on the step whose state is written.

        The GM1 vorticity extensions add the parcel and vorticity-frame steps
        they sample, computed from the same cadences those hooks use.
        """

        budget = getattr(self, "budget", None)
        due = bool(budget.budgets or budget.vorticity_history) and step in self.history_frame_steps
        stage = getattr(self, "lagrangian", None)
        if stage is not None and self._lagrangian_root is not None:
            dt = self.config.dt_s
            if budget.parcel_vorticity and stage.config.parcels_enabled:
                cadence = max(1, int(round(stage.config.values["prclfrq_s"] / dt)))
                due = due or step % cadence == 0
            if budget.vorticity_frames and stage.config.budget_enabled:
                every = max(1, int(round(stage.config.values["vorticity_budget_frq_s"] / dt)))
                due = due or step % every == 0
        return due

    def _sample(self, phase: str, *, step: int) -> None:
        """Fold one device-memory observation; retain at most the budget.

        A long integration calls this once per scheduled phase, so an
        unconditional append would grow without bound -- roughly twenty rows
        per large step, hundreds of thousands over a two-hour run -- and would
        carry every one of them into the receipt.  The running peak is folded
        on every sample regardless, so a truncated log never understates the
        maximum it reports.
        """

        free, total = self.cp.cuda.runtime.memGetInfo()
        pool = self.cp.get_default_memory_pool()
        row = {
            "phase": phase, "step": step, "device_used_bytes": int(total - free),
            "registered_model_bytes": self.registry.registered_allocation_bytes,
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
        }
        peaks = self._memory_peaks
        if not isinstance(peaks, dict):
            # The class default is an immutable empty mapping, so an instance
            # that never ran __init__ gets its own writable peak table here
            # rather than raising on the first assignment.
            peaks = self._memory_peaks = dict.fromkeys(MEMORY_SAMPLE_KEYS, 0)
        for key in MEMORY_SAMPLE_KEYS:
            if row[key] > peaks.get(key, 0):
                peaks[key] = row[key]
        budget = self.memory_sample_budget
        if budget is None or len(self.memory_samples) < budget:
            self.memory_samples.append(row)
        else:
            self.memory_samples_dropped += 1

    def peak_bytes(self, key: str) -> int:
        """The largest observation of one memory key, retained or dropped."""

        retained = max((row[key] for row in self.memory_samples if key in row), default=0)
        return max(retained, int(self._memory_peaks.get(key, 0)))

    def execute(self, operation: ScheduledOperation, *, step: int) -> None:
        p, nrk = operation.phase, operation.nrk
        if self.trace_phases:
            print(f"STEP {step} RK {nrk} {p.value}", flush=True)
        # Under a CUDA graph capture the phase markers would be recorded into
        # the graph and could never be resolved; the capture is timed as one
        # phase by the graphs' caller instead.
        capturing = active_capture() is not None
        profiler = None if capturing else getattr(self, "profiler", None)
        token = (
            None if profiler is None
            else profiler.phase_begin(p.value if nrk is None else f"rk{nrk}.{p.value}")
        )
        started = time.perf_counter()
        budget = getattr(self, "budget", None)
        with self.stream:
            if budget is not None:
                if p == StepPhase.CLOSURE:
                    budget.begin_step(step, due=self._budget_due(step))
                elif budget.active and p == StepPhase.SCALAR_MOIST and nrk == 3:
                    budget.before_scalar(self.dynamics)
                elif budget.active and p == StepPhase.PDEFQ:
                    budget.before_microphysics(self.dynamics)
            if p == StepPhase.CLOSURE:
                self.closure.closure(step)
            elif p == StepPhase.HELD_TENDENCIES:
                # cm1.F:3724's rtime, the time at the start of the step, drives
                # the SGS start-up ramp (src/turb.F:240-262) and the two-part
                # ramp (src/turb.F:264-270).
                self.closure.model_time_s = float(self.clock.model_seconds)
                # cm1.F's mtime at the start of the step is the clock that
                # sound.F:569, sound.F:416 and solve1.F:1245 test.
                if getattr(self.dynamics, "initiation", None) is not None:
                    self.dynamics.initiation.begin_step(float(self.clock.model_seconds))
                self.closure.held_tendencies(step)
                self.dynamics.finalize_held_tendencies()
            elif p == StepPhase.PRE_ACOUSTIC:
                self.dynamics.pre_acoustic(nrk)
            elif p == StepPhase.ACOUSTIC:
                self.dynamics.acoustic(nrk)
                # solve2.F:1290-1301.  CM1 saves the acoustic small-step
                # time average for the parcel driver here, at nrk==nrkmax,
                # because the scalar stage immediately refills rru/rrv/rrw
                # with the mass flux rho0*u.
                if nrk == 3:
                    self._capture_parcel_velocity()
            elif p == StepPhase.POST_ACOUSTIC:
                self.dynamics.post_acoustic(nrk)
            elif p == StepPhase.SCALAR_MOIST:
                self.dynamics.scalar_moist(nrk)
            elif p == StepPhase.REMOVE_HELD_EOS:
                self.finish.remove_held_eos()
            elif p == StepPhase.MASS_BEGIN:
                self.finish.mass_begin()
            elif p == StepPhase.MOMENTUM_BOUNDARIES:
                self.dynamics.momentum_boundaries(nrk)
            elif p == StepPhase.SCALAR_BOUNDARIES:
                self.dynamics.scalar_boundaries(nrk)
            elif p == StepPhase.TKE:
                self.closure.tke_rk(nrk)
            elif p == StepPhase.MASS_FINISH:
                self.finish.mass_finish()
            elif p == StepPhase.PDEFQ:
                self.finish.pdefq()
            elif p == StepPhase.MICROPHYSICS:
                self.finish.microphysics(
                    step=step, reflectivity_due=self.reflectivity_due(step)
                )
            elif p == StepPhase.POST_PHYSICS:
                self.finish.post_physics()
            elif p == StepPhase.SOLVE3:
                two_part = getattr(self.closure, "two_part", None)
                if two_part is not None:
                    # src/solve3.F:340-352 reads u3d-ua before "Equate the
                    # two arrays" copies the new winds into ua.
                    two_part.solve3(dt_s=self.config.dt_s)
                self.finish.solve3(step=step)
                # solve3.F:829-1130 folds the surface swaths at the end of the
                # same routine, on every large step.
                self.output.per_step(step)
            else:
                raise RuntimeError(f"unimplemented runtime phase {p}")
            if budget is not None and budget.active:
                if p == StepPhase.ACOUSTIC and nrk == 3:
                    budget.after_acoustic(self.dynamics)
                elif p == StepPhase.MICROPHYSICS:
                    budget.after_microphysics(self.dynamics)
                elif p == StepPhase.SOLVE3:
                    budget.finish_step(self.dynamics)
        key = p.value if nrk is None else f"rk{nrk}.{p.value}"
        self.phase_wall_seconds[key] = self.phase_wall_seconds.get(key, 0.0) + time.perf_counter() - started
        if token is not None:
            profiler.phase_end(token)
        if not capturing:
            self._sample(key, step=step)

    # ------------------------------------------------------------ getdbz

    def reflectivity_due(self, step: int) -> bool:
        """CM1's ``getdbz`` for one large step (``cm1.F:3214-3237``).

        CM1 raises the flag when the step about to be taken lands on an output
        time, so the scheme produces the diagnostic from the same columns the
        frame will carry.  A step that is not an output step never asks for
        it: nothing in the forecast reads reflectivity, and computing it every
        step would spend the column kernel's time and its per-thread local
        frame on words no artifact contains.
        """

        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        # A backend shell that carries no physics stage carries no carrier to
        # write, so it is never due; only a run that allocated the carriers
        # can raise the flag.
        if not getattr(self.finish, "output_dbz", False):
            return False
        return step in self.reflectivity_frame_steps

    def schedule_reflectivity(self, steps: Any) -> frozenset[int]:
        """Declare the completed-step numbers a frame will be published at.

        Publication refuses a frame whose reflectivity belongs to an earlier
        model time, so the schedule and the frame cadence are the same fact
        and are set from one place.
        """

        selected = frozenset(int(value) for value in steps)
        if any(value < 1 for value in selected):
            raise ValueError("reflectivity output steps must be positive")
        self.reflectivity_frame_steps = selected
        # cm1.F:3642-3651's ``dowriteout`` is the same fact under another
        # name: the adaptive moving-domain controller diagnoses the storm
        # centre only on a step that is already writing output.
        self.history_frame_steps = selected
        return selected

    def schedule_adaptive_output_step(self, output_time_s: float) -> bool:
        """Under ``adapt_dt``, check whether the next step writes output.

        CM1 answers this after ``getnewdt`` (``cm1.F:3013-3048``) has chosen
        ``dbldt``.  One test raises both ``dowriteout`` (``cm1.F:3177``) and
        ``getdbz`` (``cm1.F:3215``):
        ``(mtime+dbldt) >= (taptim-0.1d0*dbldt)``.  Under a fixed step the
        output steps are known before the run starts, and ``integrate``
        schedules them in one call.  Under ``adapt_dt`` the step that reaches
        an output time exists only once the ramp-down (``misclibs.F:659-681``)
        has chosen it, so the step count of a fixed schedule does not apply.
        Without this check, a frame would carry the initial state's
        reflectivity beside the current winds, and publication refuses that.

        Call this after ``adaptive_large_step``, while ``config.dt_s`` holds
        the step about to be taken.  If the test holds, that step number is
        added to the schedule and True is returned.
        """

        dt = float(self.config.dt_s)
        if float(self.clock.model_seconds) + dt < float(output_time_s) - 0.1 * dt:
            return False
        step = int(self.clock.completed_steps) + 1
        self.schedule_reflectivity(self.reflectivity_frame_steps | {step})
        return True

    def _reflectivity_record(self) -> dict[str, Any]:
        """CM1's ``output_dbz`` block, for a run that asked for it.

        A run that did not writes exactly the receipt it always wrote, so an
        existing recorded run's receipt is unchanged by this key existing.
        """

        if not getattr(self.finish, "output_dbz", False):
            return {}
        from .cuda_cartesian_finish import cartesian_reflectivity_allocation_plan
        from .cartesian_history import reflectivity_envelope_cost
        from .gm1_runtime import REFLECTIVITY_AUTHORITY

        return {
            "reflectivity": {
                "output_dbz": True,
                "dbz_level_m": self.config.dbz_level_m,
                "dbz_level_index": self.config.reflectivity_level_index,
                "output_steps": sorted(self.reflectivity_frame_steps),
                "computed_at_step": int(self.finish.completed_reflectivity_step),
                "device_plan": cartesian_reflectivity_allocation_plan(self.config),
                "envelope_cost": reflectivity_envelope_cost(self.config),
                "authority": REFLECTIVITY_AUTHORITY,
            }
        }

    def attach_adaptive_frame(self, binding: Any, *, restarted: bool = False) -> Any:
        """Bind CM1's ``do_adapt_move`` controller to this runtime's clock.

        ``cm1.F:3512-3513`` and ``cm1.F:3669-3679`` are one sequence inside a
        completed large step, so the controller is driven from the clock that
        owns that step rather than from the integration loop above it -- a
        hand-stepped runtime and an ``integrate`` get the same cadence.
        """

        from .cartesian_adaptive_frame import CartesianAdaptiveFrame

        if binding is None:
            return None
        if self.adaptive_frame is not None:
            raise RuntimeError("this runtime already drives an adaptive moving domain")
        frame = CartesianAdaptiveFrame(self, binding, restarted=restarted)
        self.adaptive_frame = frame
        self.clock.adaptive_frame = frame
        return frame

    def synchronize(self) -> None:
        profiler = getattr(self, "profiler", None)
        if profiler is None:
            self.stream.synchronize()
            return
        started = time.perf_counter()
        self.stream.synchronize()
        profiler.synchronized(time.perf_counter() - started)

    def open_tornado_diagnostics(self, *, total_steps: int) -> Any:
        """Build CM1's selected output_* diagnostics for a run of this length.

        The surface swaths are whole-run running extrema and the ground-fixed
        copy is placed from the domain-location integral, so the number of
        large steps has to be known before the first one is taken; every
        entry point that steps therefore opens them here.  A selection that
        asks for nothing builds nothing.
        """

        selection = getattr(self.config, "diagnostics", None)
        # getattr: the lane's host doubles bind this method without the class.
        restored = getattr(self, "restored_swaths", None)
        if restored is not None:
            self.restored_swaths = None
        if selection is None or not selection.active:
            # The resume record already says a carried swath is not continued.
            return None
        if self.tornado is not None:
            raise RuntimeError(
                "this runtime already carries an open diagnostic set sized "
                "for another step count; folding a second run's steps into "
                "it would report one run's swath under the other's clock"
            )
        from .cartesian_tornado_diagnostics import CartesianTornadoDiagnostics

        self.tornado = CartesianTornadoDiagnostics(self, total_steps=total_steps)
        if restored is not None and self.tornado.swaths is not None:
            # restart_read.F:1064-1126, on the canvas and series this run's
            # own length sized.
            self.tornado.continue_swaths(restored)
        # A live nest child keeps the parent's swaths on its own grid and
        # clock (nest_live.child_swath_selection), opened for the same run.
        driver = getattr(self, "nest_driver", None)
        if driver is not None:
            driver.open_child_swaths(parent_total_steps=total_steps)
        return self.tornado

    # ------------------------------------------------------------------
    # CUDA graphs (cm1core.cuda_graphs).

    # Attributes a captured schedule neither reads nor writes on the device,
    # or that change every step on the host alone.  The parent's nest driver
    # is excluded because the child's own graphs account for the child.
    _GRAPH_INVENTORY_EXCLUDED = frozenset({
        "nest_driver", "profiler", "schedule_graphs", "memory_samples",
        "_memory_peaks", "phase_wall_seconds", "startup_wall_seconds",
        "active_io_event",
    })

    def graph_roots(self) -> list[tuple[str, Any]]:
        """What the CUDA-graph inventory walks for this runtime's schedule."""

        return [
            (f".{name}", value) for name, value in vars(self).items()
            if name not in self._GRAPH_INVENTORY_EXCLUDED
        ]

    def enable_cuda_graphs(self, mode: str, *, role: str, memory: Any = None) -> Any:
        """Run this runtime's large steps through ``cuda_graphs.ScheduleGraphs``.

        ``memory`` is the :class:`cuda_graphs.GraphMemory` of another runtime
        on the same stream (a nest's parent, for its child).
        """

        from .cuda_graphs import ScheduleGraphs, install_frozen_module_adapters

        if self.schedule_graphs is not None:
            raise RuntimeError("this runtime already runs its steps through CUDA graphs")
        self.dynamics.keep_expanded_base = True
        install_frozen_module_adapters()
        self.schedule_graphs = ScheduleGraphs(
            cupy=self.cp, stream=self.stream, mode=mode, role=role, roots=self.graph_roots,
            owner=self, memory=memory,
        )
        return self.schedule_graphs

    def graph_key(self, step: int) -> Any:
        """The host decisions a captured step freezes, or why ``step`` runs eagerly.

        A tuple is a key: two steps with equal keys (and equal device
        inventories) make the same launches.  A string names the reason this
        step's schedule reads step-varying host state no key here carries, or
        reads the device on the host mid-step, so it runs in Python.
        """

        if getattr(self, "budget", None) is not None:
            return "budget-stage"
        if self.adaptive_frame is not None:
            return "adaptive-moving-frame"
        if self.config.adaptive_timestep:
            return "adaptive-timestep"
        if getattr(self.closure, "two_part", None) is not None:
            # src/turb.F:264-270's ramp reads the model time.
            return "two-part-closure-ramp"
        closure = getattr(self, "closure", None)
        if closure is not None and getattr(closure, "ramp_sgs", 0) == 1 and closure.sgs_ramp_active(
            float(self.clock.model_seconds)
        ):
            # src/turb.F:240-262: cme is a launch word that changes every step
            # until the ramp ends; after it the sealed operator's words return.
            return "sgs-startup-ramp"
        if self.reflectivity_due(step):
            return "reflectivity-step"
        stage = getattr(self, "lagrangian", None)
        if stage is not None and stage.config.parcels_enabled:
            return "parcel-velocity-capture"
        driver = getattr(self, "nest_driver", None)
        if driver is not None and getattr(driver, "parcels", None) is not None:
            return "nest-parcel-velocity-capture"
        return ("large-step",)

    def step(self) -> None:
        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            profiler.step_begin()
        self.clock.step(self)
        token = None if profiler is None else profiler.phase_begin("after_step")
        if self.nest_driver is not None:
            self.nest_driver.after_parent_step()
        if self.tornado is not None:
            # solve3.F:855-1136 folds the swaths at the end of every large
            # step, not at history cadence: a swath sampled once a minute
            # would miss the minute a tornado exists in.
            self.tornado.update(int(self.clock.completed_steps))
        # parcel.F advances the parcels once per completed large step, on the
        # acoustic-averaged mass flux captured during it (solve2.F:1290-1301),
        # and the Eulerian budget is published on the same boundary.
        self._advance_lagrangian()
        self._publish_eulerian_diagnostics()
        if profiler is not None:
            profiler.phase_end(token)
            profiler.step_end(
                step=int(self.clock.completed_steps),
                model_time_s=float(self.clock.model_seconds),
            )

    # ------------------------------------------------------------------
    # CM1's adaptive large step, misclibs.F:593-810.

    def diffusion_number(self) -> float:
        """``calcksquick`` (misclibs.F:2822-2901) for the step in force.

        The eddy coefficients live only inside the closure phase's tile
        workspace, so their domain maximum is folded there, once per step,
        into one scalar.  This applies the ``dt`` and spacing factor CM1
        applies at misclibs.F:2841-2842.
        """

        g = self.config.grid
        return cm1_diffusion_number(
            dt_s=self.config.dt_s,
            dx_m=g.dx_m,
            dy_m=g.dy_m,
            max_eddy_coefficient_m2_s=float(
                getattr(self.closure, "max_eddy_coefficient_m2_s", 0.0)
            ),
        )

    def rebind_large_step(self, dt_s: float) -> dict[str, Any]:
        """Put a new ``dt`` in force, with the ``nsound`` it implies.

        ``misclibs.F:785`` calls ``dtsmall`` on every change of ``dt``,
        because the acoustic small-step count is a function of the large step
        and CM1 has never accepted it from a namelist.  Rebinding one without
        the other would leave the run taking a small step longer than the
        stability bound the same source line exists to enforce.

        Every holder of the configuration is rebound together: the runtime
        itself, the dynamics that reads ``dt_s`` in its held-tendency and
        acoustic phases, and the clock that integrates the model time and the
        domain location.  A partial rebind would step one phase at a
        different length from another.
        """

        g = self.config.grid
        dt_s, nsound = cm1_dtsmall(
            dt_s=float(dt_s), nx=g.nx, ny=g.ny,
            dx_m=g.dx_m, dy_m=g.dy_m, dz_m=g.dz_m,
            psolver=3, adapt_dt=True,
        )
        updated = replace(self.config, dt_s=dt_s, acoustic_substeps=nsound)
        self.config = updated
        self.dynamics.config = updated
        self.closure.config = updated
        self.clock.config = updated
        return {"dt_s": dt_s, "acoustic_substeps": nsound}

    def adaptive_large_step(
        self, *, ramp_down_targets: Sequence[RampDownTarget] = ()
    ) -> dict[str, Any]:
        """Choose the next large step exactly as ``getnewdt`` does."""

        g = self.config.grid
        cflmax = self.courant_number()
        ksmax = self.diffusion_number()
        decision = cm1_getnewdt(
            dt_s=self.config.dt_s,
            cflmax=cflmax,
            ksmax=ksmax,
            dtl_s=self.config.dtl_s,
            ks_limit=cm1_ks_limit(nx=g.nx, ny=g.ny),
            model_time_s=self.clock.model_seconds,
            ramp_down_targets=tuple(ramp_down_targets),
            # cm1.F:3039 chooses the first step with getnewdt(..., doinit=.true.),
            # which skips every ramp-down (misclibs.F:658); only the calls at
            # the end of a completed step (cm1.F:3815, doinit=.false.) ramp
            # onto an output time; before this, a first output time closer than
            # about three steps was ramped onto here and not in CM1.
            apply_ramp_down=self.clock.completed_steps > 0,
        )
        bound = self.rebind_large_step(decision.dt_s)
        record = decision.record()
        record.update(bound)
        record["step"] = self.clock.completed_steps
        record["model_time_s"] = self.clock.model_seconds
        # cm1.F:3028 calls calcksquick at startup, where CM1 already has the
        # eddy coefficients its initialization built.  This port forms them
        # inside the closure phase of a large step, so before the first step
        # there are none and ksmax is zero: the diffusion limb of
        # misclibs.F:647-650 cannot bind on the first step.  Said by value
        # rather than hidden, because a reader comparing the first chosen dt
        # against CM1's would otherwise have no way to know.
        record["eddy_coefficients_measured"] = self.clock.completed_steps > 0
        return record

    def _capture_parcel_velocity(self) -> None:
        """solve2.F:1290-1301, at ``nrk == nrkmax``."""

        # A nest's child parcels that were handed to this (parent) grid take
        # this step's own average (cm1core.nest_parcels).
        from .nest_parcels import parcel_roughness

        # parcel.F:470-515 on a semi-slip grid reads this step's zntmp.
        roughness = parcel_roughness(self)
        extra = {} if roughness is None else {"roughness": roughness}
        driver = getattr(self, "nest_driver", None)
        tracker = None if driver is None else getattr(driver, "parcels", None)
        if tracker is not None:
            tracker.capture_parent(self.dynamics.arrays, self.config, **extra)
        stage = getattr(self, "lagrangian", None)
        if stage is None or not stage.config.parcels_enabled:
            return
        stage.capture_advecting_velocity(self.dynamics.arrays, **extra)

    def _nest_parcel_track(self, root: Path | None = None) -> dict[str, Any]:
        """Open (``root`` given) or publish the nest child's parcel track, if any."""

        driver = getattr(self, "nest_driver", None)
        tracker = None if driver is None else getattr(driver, "parcels", None)
        if tracker is None:
            return {}
        if root is not None:
            from .nest_parcels import NEST_CHILD_TRACK_DIRECTORY

            tracker.open_track(root / NEST_CHILD_TRACK_DIRECTORY)
            return {}
        return {"nest_child_lagrangian": tracker.publish_track()}

    def _nest_child_window(self, root: Path | None = None) -> dict[str, Any]:
        """Open (``root`` given) or close the nest child's window stream, if any.

        The stream is opened beside the child's parcel track and closed with
        it, so a run that declares it either publishes a complete stream with
        its receipt or leaves the directory absent.
        """

        driver = getattr(self, "nest_driver", None)
        window = None if driver is None else getattr(driver, "window", None)
        if window is None:
            return {}
        if root is not None:
            window.open(root)
            return {}
        record = window.close()
        return {} if record is None else {"nest_child_window": record}

    def _nest_criterion(self) -> Any:
        """The live nest's ``[nest.tornado_criterion]`` (cm1core.nest_tornado_criterion), or None."""

        driver = getattr(self, "nest_driver", None)
        return None if driver is None else getattr(driver, "criterion", None)

    def _nest_tornado_criterion(self, root: Path | None = None) -> dict[str, Any]:
        """Open (``root`` given) or close and record the nest's tornado criterion, if any."""

        criterion = self._nest_criterion()
        if criterion is None:
            return {}
        if root is not None:
            criterion.open(root)
            return {}
        return {"nest_tornado_criterion": criterion.close()}

    def _nest_criterion_stop_request(self) -> dict[str, Any] | None:
        criterion = self._nest_criterion()
        return None if criterion is None else criterion.stop_request

    def _advance_lagrangian(self) -> None:
        """Advance the parcels with this step's own advecting velocity.

        The field is the snapshot taken at the end of the final RK stage's
        acoustics -- CM1's ``uten1``/``vten1``/``wten1`` -- and not the live
        ``rru``/``rrv``/``rrw``, which by ``solve3`` time hold the scalar
        stage's mass flux ``rho0*u`` (``cuda_advection5.py:330``, CM1's own
        ``src/adv.F`` contract).
        """

        stage = getattr(self, "lagrangian", None)
        if stage is None or not stage.config.parcels_enabled:
            return
        velocity = stage.advecting_velocity
        if velocity is None:
            raise RuntimeError(
                "the parcel stage has no captured advecting velocity; "
                "solve2.F:1290-1301's nrk==nrkmax snapshot did not run"
            )
        with self.stream:
            stage.advance(velocity)
        cadence = max(1, int(round(stage.config.values["prclfrq_s"] / self.config.dt_s)))
        # Frames are retained only for an integration.  ``benchmark`` is a
        # timing harness with no history root: it still steps the parcels, so
        # the cost of the stage is measured, but it retains nothing, because a
        # long timing run would otherwise grow one host copy of every parcel
        # per cadence and publish none of them.
        if self._lagrangian_root is not None and self.clock.completed_steps % cadence == 0:
            from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY
            from .nest_parcels import RuntimeCarriers

            # cm1.F:4010-4020: parcel_interp samples the completed state
            # (u3d/v3d/w3d and the scalars) on this grid's own deck.
            self.parcel_frames.append(stage.parcel_frame(
                step=self.clock.completed_steps,
                model_time_s=self.clock.model_seconds,
                carriers=RuntimeCarriers(self, stage.deck, self.cp, CM1_OPEN_NGXY),
            ))
            stage.circulation_frame(model_time_s=self.clock.model_seconds)

    def _publish_eulerian_diagnostics(self) -> None:
        """The budget and the high-frequency window, each on its own cadence.

        Both are diagnostics of the completed state, so they run after the
        clock, exactly where CM1 writes its own output (``cm1.F:4125-4166``).
        """

        stage = getattr(self, "lagrangian", None)
        root = self._lagrangian_root
        if stage is None or root is None:
            return
        step = self.clock.completed_steps
        when = self.clock.model_seconds
        if stage.config.budget_enabled:
            every = max(1, int(round(
                stage.config.values["vorticity_budget_frq_s"] / self.config.dt_s
            )))
            if step % every == 0:
                self.lagrangian_frames.append(stage.write_budget_frame(
                    root / f"budget-{step:08d}", self._vorticity_budget_fields(),
                    step=step, model_time_s=when,
                ))
        if stage.config.hifrq_enabled and stage.window is not None:
            every = max(1, int(round(
                stage.config.values["hifrqfrq_s"] / self.config.dt_s
            )))
            if step % every == 0:
                from .high_frequency_output import write_high_frequency_window

                fields = self._high_frequency_fields()
                self.lagrangian_frames.append(write_high_frequency_window(
                    root / f"hifrq-{step:08d}", fields, stage.window,
                    order=sorted(fields), nx=self.config.grid.nx,
                    ny=self.config.grid.ny, nz=self.config.grid.nz,
                    model_time_s=when, step=step,
                    extra={
                        "kind": "high-frequency-window",
                        "unvalidated": UNVALIDATED_MARKERS["high_frequency_window"],
                    },
                ))

    _pdcomp_plan: Any = None

    def _publish_pdcomp(self, index: int) -> None:
        """``cm1.F:3483-3503``: ``pidcomp`` at every history time.

        CM1 calls it when ``dowriteout``; this is the same cadence.  It reads
        the model carriers and writes only its own buffers and frames.
        """

        stage = getattr(self, "lagrangian", None)
        root = self._lagrangian_root
        if stage is None or root is None or not stage.config.pdcomp_enabled:
            return
        from .pressure_decomposition import publish_pdcomp_frames

        self.lagrangian_frames.extend(publish_pdcomp_frames(self, root, index))

    def _pdcomp_fields(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The eight pdcomp rectangles, inverted on the device, as host arrays."""

        from .cuda_pdcomp_open import (
            build_pdcomp_coefficients, cm1_thv0_profile, device_advect,
            launch_cuda_pdcomp, pdcomp_record, prepare_cuda_pdcomp,
        )
        from .pressure_decomposition import pdcomp_selection_for

        cp, dyn = self.cp, self.dynamics
        if self._pdcomp_plan is None:
            selection = pdcomp_selection_for(self.config)
            m, base = dyn.metrics, dyn.base_profiles
            host = lambda a: cp.asnumpy(cp.ascontiguousarray(a))  # noqa: E731
            coefficients = build_pdcomp_coefficients(
                selection,
                mh=host(m["mh"][:, 0, 0]), mf=host(m["mf"][:, 0, 0]),
                rho0=host(m["rho0_3d"][:, 0, 0]), rf0=host(m["rf0"][:, 0, 0]),
                thv0=cm1_thv0_profile(
                    host(base["th0"]), host(base["qv0"]),
                    host(base["qc0"]) if "qc0" in base else None,
                    host(base["qi0"]) if "qi0" in base else None,
                ),
            )
            f2d = dyn.rotation_bindings[1].f2d if selection.icor == 1 else None
            plan = prepare_cuda_pdcomp(
                selection, coefficients, metrics=m, base_profiles=base,
                f2d=f2d, stream=dyn.stream,
            )
            stretched = bool(dyn.stretched)
            advect = device_advect(
                dyn.stretched_compiled["momentum"] if stretched else dyn.momentum_advection_compiled,
                m, selection, vertical=dyn._stretched_vertical() if stretched else None,
                stream=dyn.stream,
            )
            self._pdcomp_plan = (plan, advect, pdcomp_record(selection, coefficients))
        plan, advect, record = self._pdcomp_plan
        a, f = dyn.arrays, dyn.fields
        out = launch_cuda_pdcomp(
            plan, u3d=a["u3d"], v3d=a["v3d"], w3d=a["w3d"], th3d=f["th3d"],
            pp3d=f["pp3d"], q3d=f["q3d"], uh=a["uh"], vh=a["vh"],
            advect=advect, stream=dyn.stream,
        )
        return {name: cp.asnumpy(value) for name, value in out.items()}, record

    def _vorticity_budget_fields(self) -> dict[str, Any]:
        """Components and every budget term, as host rectangles.

        The nine velocity-gradient terms come from the packaged kernel; the
        solenoid is formed here from the live ``rho``/``prs`` carriers; the
        friction slots are published as zero and named in
        ``BUDGET_UNFILLED_TERMS``.
        """

        from .vorticity_budget import (
            VORTICITY_BUDGET_FIELD_NAMES, VORTICITY_BUDGET_THREADS,
            VORTICITY_COMPONENT_NAMES, compile_cuda_vorticity_budget,
            launch_cuda_vorticity_budget,
        )

        cp, g = self.cp, self.config.grid
        if getattr(self, "_budget_compiled", None) is None:
            self._budget_compiled = compile_cuda_vorticity_budget()
        stage = self.lagrangian
        velocity = stage.advecting_velocity
        if velocity is None:
            # The budget is an Eulerian diagnostic of the end-of-step wind, so
            # without the parcel snapshot it reads the live carriers -- which
            # for u3d/v3d/w3d are the completed state, not a reused scratch.
            from .lagrangian_parcels import (
                ParcelVelocityField, fill_parcel_vertical_boundaries,
            )

            a = self.dynamics.arrays
            velocity = ParcelVelocityField(a["u3d"], a["v3d"], a["w3d"])
            fill_parcel_vertical_boundaries(
                velocity, nx=g.nx, ny=g.ny, nz=g.nz, bbc=self.config.bbc, tbc=1,
                u_move_m_s=self.config.u_move_m_s,
                v_move_m_s=self.config.v_move_m_s, xp=cp,
            )
        shape = (g.nz, g.ny, g.nx)
        out = {name: cp.empty(shape, dtype=cp.float32)
               for name in VORTICITY_COMPONENT_NAMES}
        terms = cp.zeros((len(VORTICITY_BUDGET_FIELD_NAMES), *shape), dtype=cp.float32)
        with self.stream:
            launch_cuda_vorticity_budget(
                u=velocity.u, v=velocity.v, w=velocity.w,
                xvort=out["xvort"], yvort=out["yvort"], zvort=out["zvort"],
                terms=terms, nx=g.nx, ny=g.ny, nz=g.nz,
                dx=g.dx_m, dy=g.dy_m, dz=g.dz_m,
                stream=self.stream, compiled=self._budget_compiled, cupy=cp,
            )
        self.stream.synchronize()
        del VORTICITY_BUDGET_THREADS
        payload = {name: cp.asnumpy(value) for name, value in out.items()}
        host_terms = cp.asnumpy(terms)
        for index, name in enumerate(VORTICITY_BUDGET_FIELD_NAMES):
            payload[name] = host_terms[index]
        self._fill_solenoid(payload)
        self._fill_friction(payload)
        return payload

    def _fill_friction(self, payload: dict[str, Any]) -> None:
        """The friction slots, from the budget stage's curl of the SGS terms.

        ``BUDGET_UNFILLED_TERMS`` refused to publish the curl of the complete
        held tendency under this name.  When the budget stage ran on this step
        it holds CM1's own ``*_hturb``/``*_vturb`` split, and the friction slot
        is the curl of exactly those two; otherwise the slot stays zero and
        the frame keeps naming it unfilled.
        """

        budget = getattr(self, "budget", None)
        if budget is None or budget.completed_step != int(self.clock.completed_steps):
            return
        cp = self.cp
        for axis in ("x", "y", "z"):
            names = [f"{axis}vortb_{term}" for term in ("hturb", "vturb")
                     if f"{axis}vortb_{term}" in budget.fields]
            if len(names) == 2:
                payload[f"friction_{axis}"] = cp.asnumpy(
                    cp.add(budget.fields[names[0]], budget.fields[names[1]])
                )

    def _fill_solenoid(self, payload: dict[str, Any]) -> None:
        """``(1/rho^2) grad(rho) x grad(p)``, from the live scalar carriers."""

        import numpy as np

        from .vorticity_budget import _one_sided_grad

        cp, g = self.cp, self.config.grid
        rho = cp.asnumpy(self.dynamics.fields["rho"])
        prs = cp.asnumpy(self.dynamics.fields["prs"])
        f = np.float32
        drdz = _one_sided_grad(np, rho, f(g.dz_m), 0)
        drdy = _one_sided_grad(np, rho, f(g.dy_m), 1)
        drdx = _one_sided_grad(np, rho, f(g.dx_m), 2)
        dpdz = _one_sided_grad(np, prs, f(g.dz_m), 0)
        dpdy = _one_sided_grad(np, prs, f(g.dy_m), 1)
        dpdx = _one_sided_grad(np, prs, f(g.dx_m), 2)
        inv = f(f(1.0) / f(rho * rho))
        payload["solenoid_x"] = f(inv * f(f(drdy * dpdz) - f(drdz * dpdy)))
        payload["solenoid_y"] = f(inv * f(f(drdz * dpdx) - f(drdx * dpdz)))
        payload["solenoid_z"] = f(inv * f(f(drdx * dpdy) - f(drdy * dpdx)))

    def _high_frequency_fields(self) -> dict[str, Any]:
        """The carriers ``hifrq.F`` writes that this lane holds, as rectangles."""

        cp, f = self.cp, self.dynamics.fields
        scalars = self._lagrangian_scalars()
        payload = {
            name: cp.asnumpy(scalars[name])
            for name in ("th", "prs", "rho", "qv", "qc", "qr", "qi", "qs", "qg", "t")
        }
        payload["pp3d"] = cp.asnumpy(f["pp3d"])
        g = self.config.grid
        payload["w"] = cp.asnumpy(
            cp.multiply(cp.float32(0.5),
                        cp.add(self.dynamics.fields["w3d"][0:g.nz],
                               self.dynamics.fields["w3d"][1:g.nz + 1]))
        )
        payload["u"] = cp.asnumpy(
            cp.multiply(cp.float32(0.5),
                        cp.add(self.dynamics.fields["u3d"][:, :, 0:g.nx],
                               self.dynamics.fields["u3d"][:, :, 1:g.nx + 1]))
        )
        payload["v"] = cp.asnumpy(
            cp.multiply(cp.float32(0.5),
                        cp.add(self.dynamics.fields["v3d"][:, 0:g.ny, :],
                               self.dynamics.fields["v3d"][:, 1:g.ny + 1, :]))
        )
        payload["zvort"] = cp.asnumpy(self._mass_zvort())
        return payload

    def _mass_zvort(self):
        """The corner vorticity averaged to scalar points: misclibs.F:2053."""

        cp = self.cp
        corner = self._corner_zvort()
        return cp.multiply(
            cp.float32(0.25),
            cp.add(cp.add(corner[:, :-1, :-1], corner[:, :-1, 1:]),
                   cp.add(corner[:, 1:, :-1], corner[:, 1:, 1:])),
        )

    def _publish_lagrangian_track(self, root: Path) -> Any:
        """Write the whole track beside the history frames, once, at the end.

        A trajectory is one object over the whole run, not a per-frame slice,
        so it is written once: ``parcel.F:1626`` appends to one
        ``cm1out_pdata.dat`` for the same reason.  The directory is created
        only when there is a track to put in it.
        """

        stage = getattr(self, "lagrangian", None)
        if stage is None:
            return None
        payload: dict[str, Any] = {
            **stage.record(),
            "parcel_frames_recorded": len(self.parcel_frames),
            "circulation_rows_recorded": len(stage.circulation_rows),
        }
        if self.parcel_frames:
            from .lagrangian_stage import write_parcel_track

            directory = root / LAGRANGIAN_TRACK_DIRECTORY
            payload["track"] = write_parcel_track(directory, stage, self.parcel_frames)
            payload["track_directory"] = os.fspath(directory)
        if self.lagrangian_frames:
            payload["eulerian_frames"] = len(self.lagrangian_frames)
            payload["eulerian_frame_kinds"] = sorted(
                {row.get("kind", "unknown") for row in self.lagrangian_frames}
            )
            payload["last_eulerian_frame"] = self.lagrangian_frames[-1]
        return payload

    def _lagrangian_scalars(self) -> dict[str, Any]:
        """The fields ``parcel.F:656-1690`` samples, by CM1's names.

        Six are live carriers; five are the derived fields ``parcel_interp``
        builds itself before it samples anything, transcribed here from the
        same lines.  An attribute with no carrier at all is left at the
        unplaced sentinel and named in ``PARCEL_UNFILLED_ATTRIBUTES``, so a
        column is never passed off as a measurement it is not.
        """

        cp, f, g = self.cp, self.dynamics.fields, self.config.grid
        from .lagrangian_parcels import CM1_CP, CM1_G, CM1_REPS, CM1_REPSM1

        q = f["q3d"]
        out: dict[str, Any] = {
            # th3d is the perturbation carrier; parcel.F:762 records the full
            # potential temperature, so the base state is added back here.
            "th": cp.add(f["th0"], f["th3d"]),
            "prs": f["prs"], "rho": f["rho"],
        }
        for index, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg")):
            out[name] = q[index]
        # parcel.F:763 -- t = th * (pi0 + pp3d)
        out["t"] = cp.multiply(out["th"], cp.add(f["pi0"], f["pp3d"]))
        # parcel.F:768-775 -- buoyancy at mass points, then :822-840 to w
        # points with c1/c2 and the two linear end extrapolations:
        # -(qt - qv - qc0 - qi0), left to right.  base.F's qc0/qi0 are held
        # only for a sounding that carries base-state cloud (isnd 4, 10, 11,
        # RuntimeConfig.base_condensate); every other base state's are CM1's
        # exact zero, and subtracting +0.0 changes no word, so they are
        # omitted there.
        total = q[0]
        for index in range(1, 6):
            total = cp.add(total, q[index])
        condensate = cp.subtract(total, q[0])
        if "qc0" in f and "qi0" in f:
            condensate = cp.subtract(cp.subtract(condensate, f["qc0"]), f["qi0"])
        buoy_mass = cp.multiply(
            cp.float32(CM1_G),
            cp.add(
                cp.add(cp.divide(f["th3d"], f["th0"]),
                       cp.multiply(cp.float32(CM1_REPSM1),
                                   cp.subtract(q[0], f["qv0"]))),
                cp.negative(condensate),
            ),
        )
        # parcel.F:777-783 -- dum8, the density potential temperature.
        dum8 = cp.divide(
            cp.multiply(out["th"],
                        cp.add(cp.float32(1.0),
                               cp.multiply(cp.float32(CM1_REPS), q[0]))),
            cp.add(cp.float32(1.0), total),
        )
        c1, c2 = self.dynamics.metrics["c1"], self.dynamics.metrics["c2"]
        face = (g.nz + 1, g.ny, g.nx)
        buoy = cp.zeros(face, dtype=cp.float32)
        buoy[1:g.nz] = cp.add(cp.multiply(c1[1:g.nz], buoy_mass[0:g.nz - 1]),
                              cp.multiply(c2[1:g.nz], buoy_mass[1:g.nz]))
        buoy[0] = cp.subtract(cp.multiply(cp.float32(2.0), buoy[1]), buoy[2])
        buoy[g.nz] = cp.subtract(cp.multiply(cp.float32(2.0), buoy[g.nz - 1]),
                                 buoy[g.nz - 2])
        out["b"] = buoy
        # parcel.F:844-852 -- vpg on the w grid; mf is 1 on this lane's
        # uniform coordinate (cartesian_door refuses stretch_z != 0).
        vpg = cp.zeros(face, dtype=cp.float32)
        pp = f["pp3d"]
        vpg[1:g.nz] = cp.multiply(
            cp.float32(-CM1_CP / g.dz_m),
            cp.multiply(
                cp.subtract(pp[1:g.nz], pp[0:g.nz - 1]),
                cp.add(cp.multiply(c2[1:g.nz], dum8[1:g.nz]),
                       cp.multiply(c1[1:g.nz], dum8[0:g.nz - 1])),
            ),
        )
        # parcel.F:860-870 -- at the top and bottom boundaries vpg + b = 0.
        vpg[0] = cp.negative(buoy[0])
        vpg[g.nz] = cp.negative(buoy[g.nz])
        out["vpg"] = vpg
        out["tke"] = self.dynamics.tke["tke3d"]
        # parcel.F:1061-1072 -- zvort at C-grid corners, sampled there.
        out["zvort"] = self._corner_zvort()
        return out

    def _corner_zvort(self) -> Any:
        """``parcel.F:1064-1066``: dv/dx - du/dy at the ``(nz, ny+1, nx+1)`` corners."""

        cp, g = self.cp, self.config.grid
        a = self.dynamics.arrays
        ng = 3
        u, v = a["u3d"], a["v3d"]
        # u(i, j, k) is stored at [k, j-1+ng, i-1+ng], so CM1 row j is y index
        # j-1+ng and CM1 face i is x index i-1+ng.  Corner (i, j) for
        # i=1..nx+1, j=1..ny+1 wants u(i,j) - u(i,j-1): y indices ng..ny+ng
        # minus ng-1..ny+ng-1, at x indices ng..nx+ng.
        du = cp.subtract(u[1:g.nz + 1, ng:ng + g.ny + 1, ng:ng + g.nx + 1],
                         u[1:g.nz + 1, ng - 1:ng + g.ny, ng:ng + g.nx + 1])
        # ...and v(i,j) - v(i-1,j): x indices ng..nx+ng minus ng-1..nx+ng-1,
        # at y indices ng..ny+ng.
        dv = cp.subtract(v[1:g.nz + 1, ng:ng + g.ny + 1, ng:ng + g.nx + 1],
                         v[1:g.nz + 1, ng:ng + g.ny + 1, ng - 1:ng + g.nx])
        return cp.subtract(cp.divide(dv, cp.float32(g.dx_m)),
                           cp.divide(du, cp.float32(g.dy_m)))

    def courant_number(self) -> float:
        """CM1's advective Courant number for the state as it stands now.

        ``misclibs.F:2726-2814`` with ``dt`` read from this run's
        configuration, not from a constant: a finer grid run at a smaller step
        is measured against its own step, exactly as CM1 measures a run whose
        ``dt`` its adaptive controller has just changed.
        """

        return advective_courant_number(self)

    def advective_courant_record(self) -> dict[str, Any]:
        """Measure CM1's advective Courant number and say whether it stops.

        The norm is ``calccflquick``/``calccfl`` (misclibs.F:2726-2814 and
        misclibs.F:2584-2718) and the stop value is misclibs.F:2714, which a
        serial ``adapt_dt=0`` CM1 reaches through ``statpack.F:469`` whenever
        ``stat_cfl=1`` -- CM1's own shipped default.  This only *reports* the
        stop: CM1 does not discard the state that tripped it.  cm1.F:3517-3520
        sets ``dostat`` and ``dowriteout`` precisely so the statistics and the
        history frame for the tripping step are written, and only then does
        cm1.F:4385-4392 print "Courant number has exceeded 1.5" and stop.  The
        integration loop therefore publishes the frame first and raises after.
        """

        courant = self.courant_number()
        g = self.config.grid
        defect = advective_courant_defect(
            courant=courant,
            dt_s=self.config.dt_s,
            # On param.F:7789-7811's uniform column dz_m is the layer
            # thickness and naming it describes the grid.  On any other branch
            # it is the metric reference of param.F:8170 and no layer is that
            # deep, so the note names the layers the run actually carries.
            grid_note=(
                f" on the declared {g.dx_m!r} x {g.dy_m!r} x {g.dz_m!r} m grid"
                if g.stretch_z == 0
                else (
                    f" on the declared {g.dx_m!r} x {g.dy_m!r} m grid with "
                    f"stretch_z={g.stretch_z} layers from "
                    f"{min(g.levels.spacings_m)!r} to "
                    f"{max(g.levels.spacings_m)!r} m under a {g.ztop_m!r} m top"
                )
            ),
        )
        return {
            "number": courant,
            "stop": CM1_COURANT_STOP,
            "target": CM1_COURANT_TARGET,
            "above_adaptive_target": (
                math.isfinite(courant) and courant > CM1_COURANT_TARGET
            ),
            "stop_reached": defect is not None,
            "stop_reason": defect,
            "authority": CM1_COURANT_STOP_AUTHORITY,
        }

    def require_advective_courant(self, *, when: str) -> float:
        """Measure the Courant number and refuse CM1's own stop condition.

        Used only where there is no state to publish yet -- before the first
        large step of an integration -- so raising loses nothing CM1 would
        have written.  Inside the loop the stop is applied after the frame is
        on disk instead; see ``advective_courant_record``.
        """

        record = self.advective_courant_record()
        if record["stop_reason"] is not None:
            raise RuntimeError(f"{when}: {record['stop_reason']}")
        return float(record["number"])

    def field_checks(self) -> dict[str, Any]:
        """Small scalar diagnostics; no rendering or field transformation."""
        out = {}
        cp = self.cp
        with self.stream:
            for name in ("u3d", "v3d", "w3d", "pp3d", "th3d", "q3d", "rho", "prs"):
                value = self.dynamics.fields[name]
                out[name] = {"finite": bool(cp.all(cp.isfinite(value)).get()),
                             "minimum": float(cp.min(value).get()), "maximum": float(cp.max(value).get())}
            tke = self.dynamics.tke["tkea"]
            out["tkea"] = {"finite": bool(cp.all(cp.isfinite(tke)).get()),
                           "minimum": float(cp.min(tke).get()), "maximum": float(cp.max(tke).get())}
        if not all(row["finite"] for row in out.values()):
            raise RuntimeError("completed model state contains nonfinite physical words")
        if out["q3d"]["minimum"] < 0 or out["tkea"]["minimum"] < 0:
            raise RuntimeError("completed model state contains negative moisture or TKE")
        if out["rho"]["minimum"] <= 0 or out["prs"]["minimum"] <= 0:
            raise RuntimeError("completed model density or pressure is nonpositive")
        # misclibs.F:2714 measured, recorded, and deliberately not raised
        # here: cm1.F:3517-3520 writes the tripping state's statistics and
        # history before cm1.F:4385-4392 stops the model, so the caller
        # publishes the frame and then acts on ``stop_reason``.
        out["advective_courant"] = self.advective_courant_record()
        return out

    def publish_history(self, directory: Any, *, publisher: Any = None,
                        profile: CartesianHistoryProfile | None = None,
                        carrier_profile: str = COMPLETE_CARRIER_PROFILE,
                        initial_condition: bool = False) -> dict[str, Any]:
        """Write one observational frame of the completed device state.

        This is never inside a timed section and never restores anything: a
        history frame is the observational half of the artifact pair and the
        restart envelope is the other, and only the restart is read back.
        """
        self._materialize_history_products(initial_condition=initial_condition)
        return publish_cartesian_history(
            self, directory, publisher=publisher, profile=profile,
            carrier_profile=carrier_profile, initial_condition=initial_condition,
        )

    def _materialize_history_products(self, *, initial_condition: bool) -> None:
        # CM1 &param9 products are computed here, once, from the completed
        # post-solve3 state -- the same state the carriers come from.  It has
        # to happen before the frame is staged, or the planes published beside
        # the carriers belong to an earlier model time than they do.
        io = getattr(self, "active_io_event", None)
        tick = None if io is None else io.begin("diagnostics.output_and_budgets")
        if getattr(self, "output", None) is not None:
            self.output.materialize()
        if getattr(self, "budget", None) is not None:
            fields = self.budget.history_fields(
                step=int(self.clock.completed_steps), initial_condition=initial_condition,
            )
            if fields:
                self.output_fields = {**getattr(self, "output_fields", {}), **fields}
        if io is not None:
            io.end("diagnostics.output_and_budgets", tick)

    def benchmark(self, *, warmup_steps: int, measured_steps: int,
                  history_directory: Any = None, history_publisher: Any = None,
                  history_carrier_profile: str = COMPLETE_CARRIER_PROFILE) -> dict[str, Any]:
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if type(measured_steps) is not int or measured_steps < 1:
            raise ValueError("measured_steps must be positive")
        if history_directory is None and history_publisher is not None:
            raise ValueError("a forecast publisher without a history directory publishes nothing")
        require_history_carrier_profile(history_carrier_profile)
        if self._nest_criterion() is not None:
            # Every step here would be sampled, warm-up included, with no
            # sidecar to publish the samples to: the receipt would carry a
            # verdict none of its rows can be replayed from.
            raise ValueError(
                "the timing harness does not sample [nest.tornado_criterion]: the "
                "criterion publishes every sample to <history>/nest-child/"
                "tornado-criterion.jsonl, which only an integration opens; run it "
                "with integrate, or drop the table to time the nest"
            )
        if history_directory is None and history_carrier_profile != COMPLETE_CARRIER_PROFILE:
            raise ValueError(
                "a carrier profile without a history directory publishes nothing; "
                "the benchmark record would name a carrier inventory that no frame "
                "of this run carries"
            )
        # Price the frame, admit the output directory and prove the publisher
        # before any stepping, so an unpublishable grid costs no GPU time.
        tornado = self.open_tornado_diagnostics(
            total_steps=int(self.clock.completed_steps) + warmup_steps + measured_steps
        )
        history_profile = None
        if history_directory is not None:
            history_profile = cartesian_history_profile(
                self.config,
                carrier_profile=history_carrier_profile,
                tornado_carriers=() if tornado is None else tornado.carrier_table(),
            )
            history_directory = require_new_history_directory(history_directory)
            if history_publisher is not None:
                history_publisher = require_forecast_publisher(history_publisher)
            # This harness publishes exactly one frame, after the last
            # measured step, so that is the only step cm1.F:3214-3237 would
            # raise getdbz on.
            self.schedule_reflectivity(
                (int(self.clock.completed_steps) + warmup_steps + measured_steps,)
            )
        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            profiler.set_target(
                total_steps=int(self.clock.completed_steps) + warmup_steps + measured_steps,
                model_start_s=float(self.clock.model_seconds),
            )
        for _ in range(warmup_steps):
            self.step()
        self.synchronize()
        if profiler is not None and warmup_steps:
            # The phase and step tables describe the measured steps; the
            # first (compiling) step stays recorded under startup.
            profiler.reset_stepping()
        start_model = self.clock.model_seconds
        step_seconds = []
        started = time.perf_counter()
        for measured in range(measured_steps):
            self._reflectivity_due = (
                history_profile is not None
                and self.output.reflectivity_wanted
                and measured == measured_steps - 1
            )
            tick = time.perf_counter()
            self.step()
            step_seconds.append(time.perf_counter() - tick)
        elapsed = time.perf_counter() - started
        simulated = self.clock.model_seconds - start_model
        checks = self.field_checks()
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise RuntimeError("invalid benchmark wall clock")
        history = None
        if history_profile is not None:
            with profiled_io(self, "history", os.fspath(history_directory)):
                history = self.publish_history(
                    history_directory, publisher=history_publisher, profile=history_profile,
                    carrier_profile=history_carrier_profile,
                )
        # misclibs.F:2714 applies to the timing harness too, and in CM1's own
        # order: cm1.F:3517-3520 writes the tripping state's output -- published
        # immediately above -- and cm1.F:4385-4392 only then stops the model.
        reason = checks["advective_courant"]["stop_reason"]
        courant_stop = None if reason is None else {
            "reason": reason,
            "frame_index": None if history is None else 1,
            "step": int(self.clock.completed_steps),
            "model_time_s": float(self.clock.model_seconds),
            "frame_directory": (
                None if history_directory is None
                else os.fspath(history_directory)
            ),
            "advective_courant": float(checks["advective_courant"]["number"]),
            "frames_requested": 0 if history_profile is None else 1,
            "frames_published": 0 if history is None else 1,
            "authority": CM1_COURANT_STOP_AUTHORITY,
            "note": (
                "CM1 writes the statistics and history output for the state "
                "that tripped the stop and only then abandons the run; the "
                "timing harness publishes its one frame for the same reason"
            ),
        }
        receipt = {
            "schema": "gm1.cartesian-runtime-benchmark/v1",
            "status": (
                "PASS_EXECUTED_CARTESIAN_STEPS" if courant_stop is None
                else INTEGRATION_COURANT_STOP_STATUS
            ),
            "courant_stop": courant_stop,
            "config": self.config.record(), "source": self.source_identity,
            "runtime_source": self.runtime_source_identity,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps,
            "completed_steps": self.clock.completed_steps,
            # cm1.F:3512-3513, at the end of the measured section.
            "domain_location": self.clock.domain_location_record(),
            "simulated_seconds_in_timed_section": simulated,
            "wall_seconds_in_timed_section": elapsed,
            "simulated_seconds_per_wall_second": simulated / elapsed,
            "steps_per_wall_second": measured_steps / elapsed,
            "mean_step_wall_seconds": statistics.mean(step_seconds),
            "median_step_wall_seconds": statistics.median(step_seconds),
            "individual_step_wall_seconds": step_seconds,
            "sampled_peak_device_used_bytes": self.peak_bytes("device_used_bytes"),
            "memory_samples": self.memory_samples,
            "memory_samples_dropped": self.memory_samples_dropped,
            "driver_component_warmup": self.warm_component,
            "component_scratch_reserve": self.component_scratch_reserve,
            **self._reflectivity_record(),
            "phase_host_wall_seconds_all_steps": self.phase_wall_seconds,
            "dynamics_allocation_plan": cartesian_dynamics_allocation_plan(self.config),
            "closure_allocation_plan": cartesian_closure_allocation_plan(self.config),
            "field_checks": checks,
            "output_products": self.output.record(),
            # CM1's output_* diagnostics for this run, when the document
            # selected any.  Absent -- not null -- when it selected none, so a
            # receipt written before they existed compares byte for byte.
            **({} if self.tornado is None else {"tornado_diagnostics": self.tornado.record()}),
            "history_publication": history,
            "history_publication_in_timed_section": False,
            "last_step_operators": self.finish.step_receipts,
            "last_step_dynamics": self.dynamics.last_receipts,
            "last_step_closure": self.closure.last_receipts,
            "timing_scope": "complete CUDA stepping and runtime checks; initialization, compilation warmup, final field scans, history staging/publication, rendering and persistence excluded",
            **({} if profiler is None else {"performance": profiler.record(
                wall_seconds=elapsed,
                runtime_startup=getattr(self, "startup_wall_seconds", None),
            )}),
            "claim_limit": (
                "fresh configurable-grid runtime under qualification; executed "
                "steps and measured resource use, not native-CM1 trajectory "
                "identity or forecast skill"
                if courant_stop is None else
                "the measured steps reached CM1's advective Courant stop; the "
                "throughput above was measured on a diverging state and no "
                "part of this record is a qualified run"
            ),
        }
        if courant_stop is not None:
            raise CartesianCourantStop(
                f"after {measured_steps} measured large steps: {reason}",
                receipt=receipt,
            )
        return receipt

    # ---------------------------------------------------------------- frames

    def _frame_location(self, index: int, published: Path, staged: Path,
                        receipt: dict[str, Any]) -> dict[str, Any]:
        """Name where this frame's own artifacts finally live.

        ``cartesian_history``'s publication receipt records the absolute paths
        it wrote to, and this runtime writes every frame into a staging
        directory that is renamed into place.  Those recorded paths are
        therefore the staging paths, and a consumer that followed them after
        the rename would miss the arena.  (The field table names the arena
        relative to the frame, so it stays right after the rename.)
        This record is written inside the frame before the rename, so the
        published directory says, in final paths, what it contains.
        """

        envelope = receipt["forecast_envelope"]
        return {
            "schema": FRAME_LOCATION_SCHEMA,
            "index": index,
            "step": int(receipt["step"]),
            "model_time_s": float(receipt["model_time_s"]),
            "published_directory": os.fspath(published),
            "staged_directory": os.fspath(staged),
            "arena_path": os.fspath(published / Path(receipt["arena"]["path"]).name),
            "arena_sha256": receipt["arena"]["sha256"],
            "arena_nbytes": int(receipt["arena"]["nbytes"]),
            "field_table_path": os.fspath(
                published / Path(receipt["field_table_path"]).name
            ),
            "scalar_table_path": os.fspath(
                published / Path(receipt["scalar_table_path"]).name
            ),
            # cm1.F:3512-3513.  A watcher that reads only these records still
            # knows what patch of ground each frame covers.
            "moving_frame": dict(receipt["moving_frame"]),
            "forecast_envelope_path": (
                None if not envelope["published"]
                else os.fspath(published / Path(envelope["path"]).name)
            ),
            "finite_checked_by_rust": bool(receipt["finite_checked_by_rust"]),
            "note": (
                "history-publication.json records the staging paths the bytes were "
                "written to, and history-field-table.tsv names the arena relative "
                "to this directory; this record names the published location of "
                "the same bytes"
            ),
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Make a directory entry durable where the platform allows it."""

        flags = getattr(os, "O_DIRECTORY", None)
        if flags is None:
            return
        try:
            handle = os.open(os.fspath(path), os.O_RDONLY | flags)
        except OSError:
            return
        try:
            os.fsync(handle)
        except OSError:
            pass
        finally:
            os.close(handle)

    @staticmethod
    def _drain_publication(publications: BackgroundPublisher | None,
                           failure: BaseException,
                           notices: Any = None) -> BaseException | None:
        """Leave no worker running and no unfinished publication looking finished.

        Called with the exception that is ending the integration.  A failure
        or a first interrupt waits for the event in flight: it is a completed
        model state, and the synchronous writer would already have put it on
        disk.  SIGTERM, or an interrupt during that wait, cancels it instead;
        the worker stops within one write window, the publisher is killed and
        the staging directory is removed.  Whatever the in-flight event did is
        added to the failure as a note; the failure itself is what propagates.

        A wait that could take a while is announced on ``notices`` (stderr by
        default) first, naming the artifacts and saying that another Ctrl-C
        cancels them: an operator who sees nothing after the first Ctrl-C
        presses it again, which is exactly what discards the restart.

        Returns a SIGTERM (``SystemExit``) that arrived during the wait, which
        the caller raises once its own cleanup is done -- it is not dropped.
        """

        if publications is None:
            return None
        event = None
        labels: list[str] = []
        termination: BaseException | None = None
        try:
            in_flight = publications.in_flight
            if in_flight is None:
                return None
            labels = in_flight.labels
            if not isinstance(failure, TerminationRequested):
                if not publications.settled:
                    again = " again" if isinstance(failure, KeyboardInterrupt) else ""
                    _notice(
                        notices,
                        f"gm1: waiting for {', '.join(labels)} to finish publishing "
                        f"(Ctrl-C{again} cancels it and discards its staging)",
                    )
                event = publications.join(raise_error=False)
            else:
                _notice(
                    notices,
                    f"gm1: SIGTERM: cancelling the publication of {', '.join(labels)}; "
                    "its staging is discarded",
                )
        except BaseException as interrupt:  # noqa: BLE001 - a further interrupt: cancel instead
            event = None
            if isinstance(interrupt, SystemExit) and not isinstance(failure, SystemExit):
                termination = interrupt
            _notice(
                notices,
                f"gm1: cancelling the publication of {', '.join(labels)}; its staging "
                "is discarded",
            )
        if event is None:
            event = publications.abort()
        if event is None:
            return termination
        if termination is None and event.termination is not None and not isinstance(
            failure, SystemExit
        ):
            termination = event.termination
        settle_publication_io(event)
        done = [part.label for part in event.completed]
        if event.error is not None:
            failure.add_note(
                f"the background publication of {labels} "
                + ("was cancelled" if event.aborted else "also failed")
                + f" ({event.error!r}); published: {done or 'nothing'}, and the "
                "unfinished staging was discarded"
            )
        elif done:
            failure.add_note(f"the background publication of {done} completed before the run ended")
        return termination

    @staticmethod
    def _discard_empty_history_root(root: Path) -> None:
        """Leave no directory behind that would block the corrected retry.

        ``cartesian_history.py:428-437`` does exactly this for a frame
        directory, and for exactly this reason: ``require_new_history_directory``
        admits only a path that does not exist, so an empty directory created
        before a refusal makes the operator hand-delete it before re-running
        with a corrected ``run.dt_s``.  A root that already holds published
        frames is never touched -- those are the frames CM1 guarantees.
        """

        # The staging directory first: while it exists the root is not empty,
        # so a root that holds nothing but an abandoned staging directory would
        # otherwise survive.  A missing or non-empty path is simply skipped --
        # a root with published frames in it is never touched.
        for path in (root / FRAME_STAGING_DIRECTORY, root):
            try:
                empty = next(path.iterdir(), None) is None
            except OSError:
                continue
            if not empty:
                continue
            try:
                path.rmdir()
            except OSError:
                continue

    def _snapshot_frame(self, root: Path, index: int, *, publisher: Any,
                        profile: CartesianHistoryProfile,
                        initial_condition: bool = False,
                        host: HostArena | None = None,
                        child_host: HostArena | None = None,
                        profiled: bool = True) -> PublicationPart:
        """Take one frame off the device; the returned part writes and renames it.

        With a ``host`` arena everything that reads the runtime happens here,
        on the calling thread, at the quiescent boundary -- a nest child's
        frame included.  ``commit`` touches only host memory and the
        filesystem, so it may run on a worker while the runtime steps on, and
        it makes the frame appear in the root in one rename only once every
        file in it is on disk.  If the child's snapshot fails, the parent
        part, whose words are already in host memory, travels on the error
        (:func:`~cm1core.background_publication.attach_taken_parts`) for the
        caller to publish before re-raising it.

        Without a ``host`` the part must be committed at once on this thread
        (``_publish_frame``): the words stay on the device until the commit
        streams them, and the nest child is taken and published after the
        parent frame is renamed into place, which is the order and the host
        memory of the one-call publication.

        ``profiled=False`` is for a frame published inside another frame's
        event (a nest child's), whose stages that event already times: an I/O
        event of its own would have to be closed on the worker thread.
        """

        name = FRAME_DIRECTORY_FORMAT.format(index=index)
        staging_root = root / FRAME_STAGING_DIRECTORY
        staged, published = staging_root / name, root / name
        if published.exists() or published.is_symlink():
            raise RuntimeError(
                f"history frame {name} already exists in {root}; publishing over "
                "it would replace one model time's frame with another"
            )
        profiler = getattr(self, "profiler", None) if profiled else None
        io = None if profiler is None else profiler.io_event("history", name)
        started = time.perf_counter()
        nested = self.nest_driver is not None and not initial_condition
        on_device = host is None
        try:
            with _active_io(self, io), _graph_staging_allocator(self):
                staging_root.mkdir(parents=True, exist_ok=True)
                self._materialize_history_products(initial_condition=initial_condition)
                snapshot = snapshot_cartesian_history(
                    self, staged, publisher=publisher, profile=profile,
                    carrier_profile=profile.carrier_profile,
                    initial_condition=initial_condition, host=host,
                )
        except BaseException:
            if io is not None:
                io.finish()
            raise
        child = None
        child_failure: BaseException | None = None
        try:
            if nested and not on_device:
                with _active_io(self, io), _graph_staging_allocator(self):
                    tick = None if io is None else io.begin("nest_child_snapshot")
                    child = self.nest_driver.snapshot_child_frame(
                        root, index, publisher=publisher,
                        carrier_profile=profile.carrier_profile,
                        # The child is committed with its parent, possibly on
                        # the worker, so its words must be off the device too.
                        host=child_host if child_host is not None else HostArena(
                            pinned=False, prefault=False,
                        ),
                    )
                    if io is not None:
                        io.end("nest_child_snapshot", tick)
        except BaseException as error:  # noqa: BLE001 - raised below, with the parent part
            child_failure = error

        def commit(cancel: PublicationCancel) -> tuple[dict[str, Any], dict[str, Any]]:
            if io is not None:
                io.profiler.range_push("io.history.commit")
            try:
                receipt = commit_cartesian_history(snapshot, io=io, cancel=cancel)
                tick = None if io is None else io.begin("location_record")
                location = self._frame_location(index, published, staged, receipt)
                document = (
                    json.dumps(location, allow_nan=False, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                with (staged / FRAME_LOCATION_FILENAME).open("xb", buffering=0) as stream:
                    stream.write(document)
                    os.fsync(stream.fileno())
                if io is not None:
                    io.end("location_record", tick, len(document))
                # A restart of the same step may be committing beside this
                # frame; whichever part is later in the event renames after
                # the earlier one, exactly as committing them in order does.
                cancel.await_turn()
                tick = None if io is None else io.begin("directory_fsync_rename")
                self._fsync_directory(staged)
                os.replace(staged, published)
                self._fsync_directory(root)
                if io is not None:
                    io.end("directory_fsync_rename", tick)
                if nested and on_device:
                    tick = None if io is None else io.begin("nest_child_frame")
                    receipt["nest_child_frame"] = self.nest_driver.publish_child_frame(
                        root, index, publisher=publisher,
                        carrier_profile=profile.carrier_profile,
                    )
                    if io is not None:
                        io.end("nest_child_frame", tick)
                elif child is not None:
                    tick = None if io is None else io.begin("nest_child_frame")
                    receipt["nest_child_frame"] = child.commit(cancel)
                    if io is not None:
                        io.end("nest_child_frame", tick)
                return receipt, location
            finally:
                if io is not None:
                    io.profiler.range_pop()

        def discard() -> None:
            try:
                _discard_staging(staged)
            finally:
                snapshot.arena.release()
                if child is not None:
                    child.discard()

        part = PublicationPart(
            label=name, commit=commit, discard=discard, io=io,
            snapshot_seconds=time.perf_counter() - started,
            detail=snapshot,
            # With a host arena the commit reads only its own snapshot, so it
            # may run beside a restart taken at the same step.
            awaits_turn=not on_device,
        )
        if child_failure is not None:
            # The parent frame is in host memory: a completed state the
            # one-call publication had already renamed into place before it
            # tried the child.  It goes with the error, to be published first.
            attach_taken_parts(child_failure, [part])
            raise child_failure
        return part

    def _commit_inline(self, parts: list[PublicationPart]) -> list[Any]:
        """Commit parts on this thread, exactly as the worker would."""

        event = EventPublication(parts, background=False)
        cancel = PublicationCancel()
        try:
            run_event(event, cancel)
        except BaseException:
            # Interrupted between parts or inside one: nothing unfinished may
            # stay staged under a name a later run would trip over.
            for part in parts:
                if part.finished is None:
                    part.discard()
            raise
        finally:
            settle_publication_io(event)
        if event.error is not None:
            raise event.error
        return [part.result for part in parts]

    def _publish_frame(self, root: Path, index: int, *, publisher: Any,
                       profile: CartesianHistoryProfile,
                       initial_condition: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write one frame, then make it appear in the root in one step."""

        part = self._snapshot_frame(
            root, index, publisher=publisher, profile=profile,
            initial_condition=initial_condition,
        )
        return self._commit_inline([part])[0]

    # -------------------------------------------------------------- restart

    def publish_restart(self, directory: Any, *, publisher: Any,
                        profile: CartesianRestartProfile | None = None,
                        restart_number: int = FIRST_RESTART_NUMBER,
                        next_history_index: int = 1,
                        published_directory: Any = None) -> dict[str, Any]:
        """Write one complete restart of the live device state.

        ``restart_write.F`` writes "all information that is carried from one
        timestep to the next" (README.restart), and so does this: the padded
        prognostic owners with their halos, the held tendencies, the TKE
        triple, the microphysics persistence, the rain accumulators, the
        dry-mass reference, the moisture budget and the moving-frame integral.
        """

        if getattr(self, "nest_driver", None) is not None:
            raise RuntimeError(
                "this runtime drives a live nest, and a restart written here would "
                "carry the parent without its child; a nested restart is written by "
                "integrate(restart_root=...), which puts the child's own restart "
                "beside the parent's (cm1core.nest_restart)"
            )
        return write_cartesian_restart(
            self, directory, publisher=publisher, profile=profile,
            restart_number=restart_number, next_history_index=next_history_index,
            published_directory=published_directory,
        )

    def _snapshot_restart(self, root: Path, number: int, *, publisher: Any,
                          profile: CartesianRestartProfile,
                          next_history_index: int,
                          host: HostArena | None = None,
                          child_host: HostArena | None = None) -> PublicationPart:
        """Take one restart off the device; the returned part writes and renames it.

        A half-written restart that a later run could find and resume from is
        worse than no restart at all, so the directory is assembled under the
        same dot-prefixed staging name the history frames use and renamed into
        place only once every file in it is on disk -- by ``commit``, which
        with a ``host`` arena touches only host memory and the filesystem.
        Without one it streams the live carriers and must run at once on this
        thread (``_publish_restart``).

        A live nest's child restart (:mod:`cm1core.nest_restart`) is taken at
        the same completed parent step and written inside the same staging
        directory, so the parent and child restarts appear together or not at
        all (WRF ``share/mediation_integrate.F:1157-1182``).  If the child
        cannot be taken, neither is published: a parent restart without its
        child would resume by rebuilding the child from the parent.
        """

        name = RESTART_DIRECTORY_FORMAT.format(number=number)
        staging_root = root / FRAME_STAGING_DIRECTORY
        staged, published = staging_root / name, root / name
        if published.exists() or published.is_symlink():
            raise RuntimeError(
                f"restart {name} already exists in {root}; publishing over it "
                "would replace one model time's carried state with another and "
                "leave a resume reading a mixture of the two"
            )
        staging_root.mkdir(parents=True, exist_ok=True)
        profiler = getattr(self, "profiler", None)
        io = None if profiler is None else profiler.io_event("restart", name)
        started = time.perf_counter()
        driver = getattr(self, "nest_driver", None)
        child = None
        try:
            with _active_io(self, io), _graph_staging_allocator(self):
                snapshot = snapshot_cartesian_restart(
                    self, staged, publisher=publisher, profile=profile,
                    restart_number=number, next_history_index=next_history_index,
                    published_directory=published, host=host,
                )
                if driver is not None:
                    from .nest_restart import NEST_CHILD_RESTART_DIRECTORY

                    tick = None if io is None else io.begin("nest_child_snapshot")
                    try:
                        child, declaration = driver.snapshot_child_restart(
                            staged / NEST_CHILD_RESTART_DIRECTORY, publisher=publisher,
                            published_directory=published / NEST_CHILD_RESTART_DIRECTORY,
                            restart_number=number, next_history_index=next_history_index,
                            # Committed with the parent, possibly on the worker,
                            # so the child's words leave the device here too.
                            host=None if host is None else (
                                child_host if child_host is not None
                                else HostArena(pinned=False, prefault=False)
                            ),
                        )
                    except BaseException:
                        snapshot.arena.release()
                        raise
                    snapshot = replace(
                        snapshot, receipt_extra={"nest_child": declaration},
                    )
                    if io is not None:
                        io.end("nest_child_snapshot", tick)
        except BaseException:
            if io is not None:
                io.finish()
            raise

        def commit(cancel: PublicationCancel) -> dict[str, Any]:
            if io is not None:
                io.profiler.range_push("io.restart.commit")
            try:
                # cm1.F:4246 writes the restart after the step's output: a
                # frame of the same step, committed beside this restart, is
                # renamed into place first (cancel.await_turn() below, once
                # every restart of this event is staged), and a frame that
                # failed stops it.
                # WRF's order (mediation_integrate.F:1157-1182): this domain's
                # restart, then the child's.  The parent's receipt is the last
                # file written, after the child's envelope is on disk, so it
                # declares the child by the digests of its committed envelope
                # header, contract and scalar table
                # (nest_restart.bind_nest_child_declaration): the child beside
                # a parent is the one written with it, not another run's at the
                # same step.  Neither is visible until both are renamed into
                # place below.
                if child is None:
                    receipt = commit_cartesian_restart(snapshot, io=io, cancel=cancel)
                else:
                    from .nest_restart import bind_nest_child_declaration

                    receipt = commit_cartesian_restart(
                        snapshot, io=io, cancel=cancel, defer_receipt=True,
                    )
                    cancel.check()
                    tick = None if io is None else io.begin("nest_child_restart")
                    child_receipt = commit_cartesian_restart(child, cancel=cancel)
                    self._fsync_directory(child.directory)
                    if io is not None:
                        io.end("nest_child_restart", tick)
                    receipt["nest_child"] = bind_nest_child_declaration(
                        receipt["nest_child"], child_receipt,
                    )
                    cancel.check()
                    write_restart_receipt(snapshot.directory, receipt, io=io)
                    receipt = {**receipt, "nest_child_restart": child_receipt}
                cancel.await_turn()
                tick = None if io is None else io.begin("directory_fsync_rename")
                self._fsync_directory(staged)
                os.replace(staged, published)
                self._fsync_directory(root)
                if io is not None:
                    io.end("directory_fsync_rename", tick)
                return receipt
            finally:
                if io is not None:
                    io.profiler.range_pop()

        def discard() -> None:
            try:
                _discard_staging(staged)
            finally:
                snapshot.arena.release()
                if child is not None:
                    child.arena.release()

        return PublicationPart(
            label=name, commit=commit, discard=discard, io=io,
            snapshot_seconds=time.perf_counter() - started, detail=snapshot,
            awaits_turn=host is not None,
        )

    def _publish_restart(self, root: Path, number: int, *, publisher: Any,
                         profile: CartesianRestartProfile,
                         next_history_index: int) -> dict[str, Any]:
        """Write one restart, then make it appear in the root in one step."""

        part = self._snapshot_restart(
            root, number, publisher=publisher, profile=profile,
            next_history_index=next_history_index,
        )
        return self._commit_inline([part])[0]

    def restore_restart(self, restored: RestoredCartesianRestart, *,
                        profile: CartesianRestartProfile | None = None) -> dict[str, Any]:
        """Continue from one validated restart: ``cm1.F:2666`` under irst=1."""

        # param.F:891-903 sets iinit = 0 on a restart, and init3d.F:175-176
        # then leaves convinit = wnudge = 0: a restarted CM1 run applies no
        # iinit=9/10/12 forcing, whatever the time it resumes at.
        if getattr(self.dynamics, "initiation", None) is not None:
            self.dynamics.initiation.disabled_by_restart = True
        return restore_cartesian_restart(self, restored, profile=profile)

    def initial_condition(self, *, restart_root: Any, restart_publisher: Any,
                          history_root: Any = None, history_publisher: Any = None,
                          history_carrier_profile: str = COMPLETE_CARRIER_PROFILE,
                          restart_number: int = FIRST_RESTART_NUMBER) -> dict[str, Any]:
        """CM1's ``iconly``: build the initial condition, write it, integrate nothing.

        ``cm1.F:4341-4350`` stops the model here with the message "User has
        requested initial conditions only".  README.namelist:233-235 is the
        selector.  The artifact is the complete carried state at ``nstep=0``,
        which is both what a user inspects to check a sounding, a bubble and a
        grid, and a restart the integration can be started from later without
        re-running the initializer.
        """

        if int(self.clock.completed_steps) != 0:
            raise RuntimeError(
                f"the runtime has completed {int(self.clock.completed_steps)} large "
                "steps; an initial-condition publication here would record an "
                "integrated state as the run's starting point"
            )
        # cm1.F:3213-3244 raises getdbz inside the same loop iteration that
        # performs the nstep=0 writeout (cm1.F:3943), so CM1's iconly frame
        # carries a real dBZ field.  GM1 fills the reflectivity carriers only
        # inside ``finish.microphysics(reflectivity_due=True)``, which iconly
        # never reaches, and ``_allocate_reflectivity`` leaves them zero.  The
        # history freshness check cannot catch it here -- the published step
        # and ``completed_reflectivity_step`` are both 0 -- so the frame would
        # be admitted carrying 0 dBZ at every gridpoint: a uniform light echo
        # over the whole domain, beside the initial sounding's real winds,
        # with nothing in the artifact saying the two disagree.
        if history_root is not None and getattr(self.finish, "output_dbz", False):
            raise RuntimeError(
                "this run declares output_dbz and iconly with a history root, but "
                "an initial condition takes no large step, so the reflectivity "
                "carriers are still the zeros they were allocated with; the frame "
                "would publish 0 dBZ at every gridpoint as though the initial state "
                "held a uniform light echo. Either drop run.output_dbz for the "
                "initial-condition publication, or drop the history root and keep "
                "the restart, which carries no reflectivity"
            )
        profile = cartesian_restart_profile(self.config)
        root = Path(require_new_history_directory(restart_root))
        frames = None
        history_profile = None
        if history_root is not None:
            history_profile = cartesian_history_profile(
                self.config,
                carrier_profile=require_history_carrier_profile(history_carrier_profile),
            )
            frames = Path(require_new_history_directory(history_root))
            if history_publisher is not None:
                history_publisher = require_forecast_publisher(history_publisher)
        started = time.perf_counter()
        checks = self.field_checks()
        root.mkdir(parents=True)
        restart = self._publish_restart(
            root, restart_number, publisher=restart_publisher, profile=profile,
            next_history_index=1,
        )
        frame = None
        if frames is not None:
            frames.mkdir(parents=True)
            frame, _ = self._publish_frame(
                frames, INITIAL_CONDITION_FRAME_INDEX, publisher=history_publisher,
                profile=history_profile, initial_condition=True,
            )
        staging_root = root / FRAME_STAGING_DIRECTORY
        try:
            staging_root.rmdir()
        except OSError:
            pass
        return {
            "schema": INITIAL_CONDITION_SCHEMA,
            "status": INITIAL_CONDITION_STATUS,
            "authority": "CM1 r21.1 cm1.F:4341-4350, README.namelist:233-235 (iconly)",
            "config": self.config.record(), "source": self.source_identity,
            "runtime_source": self.runtime_source_identity,
            "completed_steps": 0,
            "model_seconds": self.clock.model_seconds,
            "domain_location": self.clock.domain_location_record(),
            "restart_root": os.fspath(root),
            "restart_publication": restart,
            "history_root": None if frames is None else os.fspath(frames),
            "history_publication": frame,
            "wall_seconds": time.perf_counter() - started,
            "dynamics_allocation_plan": cartesian_dynamics_allocation_plan(self.config),
            "closure_allocation_plan": cartesian_closure_allocation_plan(self.config),
            "restart_profile": profile.record(),
            "field_checks": checks,
            "integrated": False,
            "claim_limit": "the initialized source state of one admitted grid, written as a complete restart and integrated no further; not a forecast",
            **({} if getattr(self, "profiler", None) is None else {
                "performance": self.profiler.record(
                    wall_seconds=time.perf_counter() - started,
                    runtime_startup=getattr(self, "startup_wall_seconds", None),
                ),
            }),
        }

    def _restart_schedule(
        self, *, cadence: Any, restart_interval_s: float | None, restart_root: Any,
        restart_executable: Any, restart_identity_sha256: str | None,
        restart_config_sha256: str | None,
        first_restart_number: int, reset_output_clock: bool,
    ) -> dict[str, Any] | None:
        """Price CM1's restart cadence on the host before a device word moves.

        ``src/param.F:6338`` starts ``rsttim`` at ``rstfrq``; on a restart
        ``src/cm1.F:2711-2714`` moves it to ``mtime + rstfrq`` when
        ``restart_reset_frqtim`` is true and otherwise leaves the persisted
        value, which ``src/cm1.F:4247-4252`` had already advanced past the
        model time.  Both are step counts here, because this lane's clock is
        an exact multiple of ``run.dt_s`` and ``config.py`` refuses a cadence
        that is not.
        """

        from .cartesian_restart import (
            CARTESIAN_RESTART_UNVALIDATED,
            CM1_FIRST_RESTART_NUMBER,
            MAX_RESTART_PAYLOAD_BYTES,
            CartesianRestartError,
            require_restart_cli,
            restart_carrier_inventory,
            restart_executable as resolve_restart_executable,
            restart_paths,
        )

        if restart_interval_s is None:
            if restart_root is not None:
                raise ValueError(
                    "a restart directory was supplied with no restart cadence; "
                    "the run would create the directory and never write a "
                    "restart into it"
                )
            return None
        if restart_root is None:
            raise ValueError(
                "a restart cadence was declared with no restart directory, so "
                "cm1.F:4246 would have nowhere to write and the run would "
                "advance believing it was resumable"
            )
        if restart_identity_sha256 is None:
            raise ValueError(
                "a restart binds the sealed configuration it was produced by; "
                "without that hash a later resume could not refuse a changed "
                "configuration"
            )
        if type(first_restart_number) is not int or first_restart_number < CM1_FIRST_RESTART_NUMBER:
            raise ValueError(
                f"the first restart file number is {CM1_FIRST_RESTART_NUMBER} "
                "or greater (cm1.F:301, cm1.F:4310)"
            )
        steps = restart_interval_s / self.config.dt_s
        if not math.isfinite(steps) or steps < 1.0 or not float(steps).is_integer():
            raise ValueError(
                f"restart_interval_s={restart_interval_s} is not a whole number "
                f"of {self.config.dt_s} s large steps, so cm1.F:3186 could never "
                "place a restart on a completed-step boundary"
            )
        steps = int(steps)
        tool = resolve_restart_executable(restart_executable)
        require_restart_cli(tool)
        # Price the artifact against the CM1FCP1 payload cap here, before the
        # first large step, so a grid whose mutable registry cannot fit stops
        # the run now rather than at the first restart hours in.
        carriers = restart_carrier_inventory(self)
        payload = sum(carrier.nbytes for carrier in carriers)
        if payload > MAX_RESTART_PAYLOAD_BYTES:
            raise CartesianRestartError(
                f"this runtime's mutable device registry is {payload} bytes over "
                f"{len(carriers)} carriers at nx={self.config.grid.nx}, "
                f"ny={self.config.grid.ny}, nz={self.config.grid.nz}; the CM1FCP1 "
                f"array contract caps one payload at {MAX_RESTART_PAYLOAD_BYTES} "
                "bytes (rust/cm1-state/src/forecast.rs:29), so no restart could "
                "be written and this run would be unresumable after spending "
                "the whole integration believing otherwise"
            )
        directory = Path(restart_root)
        directory.mkdir(parents=True, exist_ok=True)
        completed = self.clock.completed_steps
        if reset_output_clock or completed == 0:
            # cm1.F:2714 / param.F:6338
            following = completed + steps
        else:
            # cm1.F:4247-4252 left rsttim at the first multiple of rstfrq
            # strictly greater than the model time the restart was written at.
            following = ((completed // steps) + 1) * steps
        number = first_restart_number
        while True:
            artifact, sidecar = restart_paths(directory, number)
            if not artifact.exists() and not sidecar.exists():
                break
            number += 1
            if number > first_restart_number + 1_000_000:
                raise CartesianRestartError(
                    "no unused restart file number is available in this directory"
                )
        return {
            "authority": (
                "src/cm1.F:3186 decides, src/cm1.F:4246 writes, "
                "src/cm1.F:4247-4252 advances rsttim, src/cm1.F:301 and "
                "src/cm1.F:4310 number the files"
            ),
            "directory": os.fspath(directory),
            "executable": tool,
            "identity_sha256": restart_identity_sha256,
            "config_sha256": restart_config_sha256,
            "interval_s": float(restart_interval_s),
            "steps_per_restart": steps,
            "next_step": following,
            "next_number": number,
            "reset_output_clock": bool(reset_output_clock),
            "carrier_count": len(carriers),
            "declared_payload_nbytes": payload,
            "maximum_payload_nbytes": MAX_RESTART_PAYLOAD_BYTES,
            "unvalidated": [dict(entry) for entry in CARTESIAN_RESTART_UNVALIDATED],
            "written": [],
        }

    def _write_restart(self, restart: dict[str, Any]) -> dict[str, Any]:
        """Write one restart file for the large step that has just completed.

        ``src/cm1.F:4246`` reaches ``restart_write`` only after the step, and
        ``src/cm1.F:4247-4252`` then advances ``rsttim`` by ``rstfrq`` until it
        is past the new model time -- a loop, not one addition, so a cadence
        shorter than one large step cannot make the schedule fall behind.  The
        same loop is here, over step counts.
        """

        from .cartesian_restart import capture_cartesian_restart

        capture = capture_cartesian_restart(
            self, restart["directory"], number=restart["next_number"],
            identity_sha256=restart["identity_sha256"],
            config_sha256=restart["config_sha256"],
            executable=restart["executable"],
        )
        completed = self.clock.completed_steps
        following = restart["next_step"]
        while following <= completed:
            following += restart["steps_per_restart"]
        restart["next_step"] = following
        restart["next_number"] = capture.number + 1
        row = {
            "number": capture.number,
            "step": capture.step,
            "model_time_s": capture.model_time_s,
            "artifact": capture.artifact,
            "sidecar": capture.sidecar,
            "artifact_sha256": capture.receipt["artifact_sha256"],
            "contract_sha256": capture.receipt["contract_sha256"],
            "inventory_sha256": capture.inventory_sha256,
            "raw_arena_nbytes": capture.raw_arena_nbytes,
            "carrier_count": capture.carrier_count,
            "next_restart_step": following,
        }
        restart["written"].append(row)
        return row

    def integrate(self, *, duration_s: float, history_interval_s: float,
                  history_root: Any, history_publisher: Any = None,
                  history_carrier_profile: str = COMPLETE_CARRIER_PROFILE,
                  restart_root: Any = None, restart_interval_s: float | None = None,
                  restart_publisher: Any = None, restart_reset_frqtim: bool = False,
                  first_history_index: int = 1,
                  first_restart_number: int = FIRST_RESTART_NUMBER,
                  memory_sample_budget: int = 4096, frame_row_budget: int = 1024,
                  restart_row_budget: int = 1024, progress: Any = None,
                  background_publication: bool = True,
                  concurrent_event_parts: bool = True,
                  history_initial: bool = False,
                  control: Any = None) -> dict[str, Any]:
        """Step to a requested model duration, publishing as events fall due.

        This is the integration entry point, not the timing harness: nothing
        per step is retained.  The memory log is bounded by an explicit budget,
        the step wall clock is folded into running aggregates rather than a
        per-step list, and the per-frame and per-restart rows are bounded too,
        so a receipt for 14,400 large steps is the same size as one for
        fourteen.

        ``duration_s`` is absolute model time, as CM1's ``timax`` is, so a
        runtime that has already been restored from a restart integrates the
        remainder instead of the whole run again.

        History frames land at every multiple of ``history_interval_s``, each
        in its own directory under ``history_root``.  With a
        ``restart_interval_s`` and a ``restart_root``, a complete restart lands
        at every multiple of that interval as well -- ``cm1.F:3186`` and
        ``cm1.F:4247-4252`` -- each renamed into place only once written.

        With ``background_publication`` (the default) an event costs the
        stepping thread only its field scan and one copy of the published
        words into a reused host arena; the digests, the files, the publisher
        and the fsyncs of that event run on one worker thread while the next
        large steps are taken.  The previous event is joined before the next
        one is taken and before this returns, so frames and restarts are
        written and reported in model order and nothing is reported -- to
        ``progress`` or in this receipt -- before it is durable.  A worker
        error is raised at that join.  If the run fails or is interrupted
        (SIGINT) while an event is in flight, that publication of a completed
        state is finished first; a second interrupt, or SIGTERM, cancels it
        and discards its staging.  ``background_publication=False`` commits
        each event on the stepping thread instead; the bytes are the same.

        ``history_initial`` publishes the initialized state as frame 0
        (``frame-000000``) before the first large step, as CM1 writes its
        ``nstep = 0`` output (cm1.F:3943).  A run that declares ``output_dbz``
        publishes that one frame without dbz and cref: nothing has computed
        them yet (cartesian_history.INITIAL_FRAME_REFLECTIVITY_OMITTED).  Only
        an integration that starts at step 0 has an initialized state to
        publish.

        ``control`` is a :class:`cm1core.run_control.RunControl`.  It is told
        the start, every completed large step and the end; a stop request it
        reports ends the run at that step boundary once the publication in
        flight is durable (an event due at that step is taken first), and a
        pause request ends it right after the next scheduled restart is
        durable.  Either writes this receipt with its own status and a
        ``control_stop`` block.

        ``concurrent_event_parts`` (the default) lets the worker write a frame
        and the restart of the same step beside each other, renamed frame
        first.  ``concurrent_event_parts=False`` (``cm1core cartesian
        --event-parts in-turn``) writes them one after another, as a storage
        whose two writes contend, or a host short of disk or memory for both
        at once, may need: the bytes and the rename order are the same, the
        event takes the sum of the two writes, and a restart that runs out of
        disk or memory no longer takes its frame with it.  The receipt's
        ``publication.event_parts`` is ``beside`` or ``in-turn``.
        """

        adaptive = bool(self.config.adaptive_timestep)
        cadence = history_cadence(
            dt_s=self.config.dt_s, duration_s=duration_s, interval_s=history_interval_s,
            adaptive=adaptive,
        )
        timestep_rows: list[dict[str, Any]] = []
        for name, value in (
            ("memory_sample_budget", memory_sample_budget),
            ("frame_row_budget", frame_row_budget),
            ("restart_row_budget", restart_row_budget),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(background_publication, bool):
            raise TypeError("background_publication must be a bool")
        if not isinstance(concurrent_event_parts, bool):
            raise TypeError("concurrent_event_parts must be a bool")
        if not isinstance(history_initial, bool):
            raise TypeError("history_initial must be a bool")
        if history_initial and int(self.clock.completed_steps) != 0:
            raise ValueError(
                f"history_initial asks for the initialized state as frame 0, but this "
                f"runtime has completed {int(self.clock.completed_steps)} large steps "
                "(a resumed run); its frame 0 belongs to the run that started it"
            )
        if (restart_root is None) != (restart_interval_s is None):
            raise ValueError(
                "a restart cadence needs a restart directory and a restart "
                "directory needs a cadence; one without the other either writes "
                "nothing or writes it nowhere"
            )
        if restart_interval_s is not None and restart_publisher is None:
            raise ValueError(
                "a restart cadence needs an admitted forecast publisher; a "
                "checkpoint Rust never validated cannot be proved complete, and "
                "the first time anyone would find out is inside a resume"
            )
        schedule = run_schedule(
            dt_s=self.config.dt_s, duration_s=duration_s,
            history_interval_s=history_interval_s,
            restart_interval_s=restart_interval_s,
            start_step=int(self.clock.completed_steps),
            reset_frqtim=bool(restart_reset_frqtim),
            first_history_index=first_history_index,
            first_restart_number=first_restart_number,
        )
        cadence = schedule.cadence
        # Price the frame and the restart, admit both roots and prove both
        # publishers before the first step, so a run that cannot publish what
        # it was asked for costs no GPU time at all.
        # The swaths are opened before the frame is priced, because a selected
        # swath is a published carrier and the payload has to be priced with it.
        # cadence.total_steps is the run's final step, counted from the
        # initial condition as CM1's timax is (run_schedule), so a resumed run
        # sizes its swaths for the same run the restart belongs to rather
        # than for that run plus the steps it had already taken.
        tornado = self.open_tornado_diagnostics(total_steps=int(cadence.total_steps))
        profile = cartesian_history_profile(
            self.config,
            carrier_profile=require_history_carrier_profile(history_carrier_profile),
            tornado_carriers=() if tornado is None else tornado.carrier_table(),
        )
        root = Path(require_new_history_directory(history_root))
        publisher = (
            None if history_publisher is None
            else require_forecast_publisher(history_publisher)
        )
        restart_profile = None
        restart_home = None
        child_restart_profile = None
        if restart_interval_s is not None:
            # restart_write.F:696-758: the swaths this run keeps travel in its
            # restarts, laid out for this run's length.
            restart_profile = cartesian_restart_profile(
                self.config, swaths=None if tornado is None else tornado.restart_layout(),
            )
            nest_driver = getattr(self, "nest_driver", None)
            if nest_driver is not None:
                # The child's restart is priced with the parent's, before the
                # first step: a child whose carried state the envelope cannot
                # hold stops the run now, not at its first restart.
                from .nest_restart import (
                    driver_nparcels,
                    nest_child_restart_profile,
                    nest_restart_scalars,
                )

                child_restart_profile = nest_child_restart_profile(
                    nest_driver.child.config, nparcels=driver_nparcels(nest_driver),
                )
                # ...and its coupling scalars are taken once, dry, off the
                # device: a tracker, counter or parcel record the table cannot
                # name stops the run here, not at its first restart.
                nest_restart_scalars(nest_driver)
            restart_home = Path(require_new_history_directory(restart_root))
            restart_publisher = require_forecast_publisher(restart_publisher)
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable")
        # misclibs.F:2714 is measured on the state about to be integrated,
        # before any directory exists.  ``require_new_history_directory``
        # refuses a history root that is already present, so a refusal raised
        # after ``root.mkdir`` would leave an empty directory that blocks every
        # retry at the same path with a corrected run.dt_s.  This check joins
        # the admission block above the mkdir for that reason; the cost is one
        # device reduction, which is why it can be afforded here at all.
        initial_courant = self.require_advective_courant(
            when="before the first large step of this integration"
        )
        # cm1.F:3214-3237.  getdbz is raised on exactly the steps this run
        # publishes a frame at, which the schedule has already resolved: on a
        # resumed run the frame steps stay on the absolute grid the original
        # run declared (or move onto the resume time under
        # restart_reset_frqtim), so deriving them again from the clock and the
        # whole simulation's frame_count would raise the flag on steps this
        # run never reaches and miss the steps it publishes at -- and
        # cartesian_history refuses every frame whose reflectivity belongs to
        # an earlier model time.  Restart-only events get no getdbz: a restart
        # carries no dBZ carrier.
        # Under adapt_dt an event's step is its fixed-step position, which the
        # run does not follow: the stepping loop below schedules each output
        # step once getnewdt has chosen it (schedule_adaptive_output_step).
        self.schedule_reflectivity(
            ()
            if adaptive
            else (event.step for event in schedule.events if event.history)
        )
        root.mkdir(parents=True)
        if restart_home is not None:
            restart_home.mkdir(parents=True)
        # parcel.F's tracks, the vorticity budget, the circuits and the
        # high-frequency windows all land beside the history frames, under one
        # directory created before the first step so the per-step hooks have
        # somewhere to write.  It is created only when something will write.
        stage = getattr(self, "lagrangian", None)
        if stage is not None and (
            stage.config.parcels_enabled or stage.config.budget_enabled
            or stage.config.hifrq_enabled or stage.config.pdcomp_enabled
        ):
            self._lagrangian_root = root / LAGRANGIAN_TRACK_DIRECTORY
            self._lagrangian_root.mkdir(parents=True)
        self._nest_parcel_track(root)
        self._nest_child_window(root)
        self._nest_tornado_criterion(root)
        self.memory_sample_budget = memory_sample_budget
        publications = (
            BackgroundPublisher(concurrent_parts=concurrent_event_parts)
            if background_publication else None
        )
        # One reused host arena per artifact kind, sized once.  Page-locked
        # where the storage allows it, so an event's device copy is a direct
        # transfer and pinning cost is paid once per run, not once per event.
        arenas = {
            "history": HostArena(), "restart": HostArena(), "nest-child": HostArena(),
            "nest-child-restart": HostArena(),
        }
        terminating = termination_as_exception()
        terminating.__enter__()
        # cartesian_history.py:428-437 does this for a frame directory: an
        # empty directory left behind by a refusal blocks every retry at the
        # same path, because require_new_history_directory admits only a root
        # that does not exist.  The same is true of the history root itself.
        try:

            start_step = self.clock.completed_steps
            start_model = self.clock.model_seconds
            frames: list[dict[str, Any]] = []
            restarts: list[dict[str, Any]] = []
            frames_dropped, restarts_dropped = 0, 0
            frame_index = schedule.first_history_index
            restart_number = schedule.first_restart_number
            step_count, step_total, step_minimum, step_maximum = 0, 0.0, math.inf, 0.0
            last_receipt: dict[str, Any] | None = None
            last_restart: dict[str, Any] | None = None
            checks: dict[str, Any] = {}
            courant_stop: dict[str, Any] | None = None
            stop_event: dict[str, Any] | None = None
            criterion_stop: dict[str, Any] | None = None
            criterion_event: dict[str, Any] | None = None
            profiler = getattr(self, "profiler", None)
            arena_reserve_s = arenas["history"].reserve(self.cp, profile.payload_nbytes)
            if restart_profile is not None:
                arena_reserve_s += arenas["restart"].reserve(
                    self.cp, restart_profile.payload_nbytes
                )
            if child_restart_profile is not None:
                arena_reserve_s += arenas["nest-child-restart"].reserve(
                    self.cp, child_restart_profile.payload_nbytes
                )
            if profiler is not None:
                profiler.startup_stage(
                    "publication_host_arenas", arena_reserve_s,
                    bytes=sum(arena.capacity_bytes for arena in arenas.values()),
                )

            def finish_event(publication: EventPublication | None) -> None:
                """Report one settled event: rows, receipts, progress -- in that order."""

                nonlocal last_receipt, last_restart, frames_dropped, restarts_dropped
                if publication is None:
                    return
                settle_publication_io(publication)
                if publication.error is not None:
                    raise publication.error
                frame_base, restart_base = publication.context
                results = iter(part.result for part in publication.parts)
                row: dict[str, Any] | None = None
                restart_row: dict[str, Any] | None = None
                if frame_base is not None:
                    last_receipt, location = next(results)
                    row = {
                        "index": frame_base["index"],
                        "step": int(last_receipt["step"]),
                        "model_time_s": float(last_receipt["model_time_s"]),
                        "directory": location["published_directory"],
                        "arena_sha256": location["arena_sha256"],
                        "forecast_envelope_published": bool(
                            last_receipt["forecast_envelope"]["published"]
                        ),
                        "stepping_wall_seconds": frame_base["stepping_wall_seconds"],
                        "advective_courant": frame_base["advective_courant"],
                        "courant_stop_reason": frame_base["courant_stop_reason"],
                    }
                    if len(frames) < frame_row_budget:
                        frames.append(row)
                    else:
                        frames_dropped += 1
                if restart_base is not None:
                    last_restart = next(results)
                    restart_row = {
                        "number": restart_base["number"],
                        "step": int(last_restart["step"]),
                        "model_time_s": float(last_restart["model_time_s"]),
                        "directory": last_restart["directory"],
                        "envelope_path": last_restart["forecast_envelope"]["path"],
                        "arena_sha256": last_restart["arena"]["sha256"],
                        "next_history_index": int(last_restart["next_history_index"]),
                    }
                    if len(restarts) < restart_row_budget:
                        restarts.append(restart_row)
                    else:
                        restarts_dropped += 1
                    if row is not None:
                        row["restart_written"] = restart_row["number"]
                if progress is not None:
                    report = dict(row) if row is not None else dict(restart_row)
                    if restart_row is not None:
                        # The restart of this event, whole, for a reporter
                        # that records restarts (the --events log).
                        report["restart"] = dict(restart_row)
                    progress(report)

            if history_initial:
                # cm1.F:3943: CM1 writes the nstep=0 state before it takes a
                # step.  Taken like any other event, so it reaches disk in
                # model order and is reported before frame 1 is taken.
                initial_profile = initial_frame_history_profile(
                    self.config, carrier_profile=profile.carrier_profile,
                    tornado_carriers=() if tornado is None else tornado.carrier_table(),
                )
                initial_part = self._snapshot_frame(
                    root, INITIAL_CONDITION_FRAME_INDEX, publisher=publisher,
                    profile=initial_profile, initial_condition=True,
                    host=arenas["history"], child_host=arenas["nest-child"],
                )
                initial_publication = EventPublication(
                    [initial_part],
                    context=({
                        "index": INITIAL_CONDITION_FRAME_INDEX,
                        "stepping_wall_seconds": 0.0,
                        "advective_courant": float(initial_courant),
                        "courant_stop_reason": None,
                    }, None),
                    background=publications is not None,
                )
                if publications is not None:
                    publications.submit(initial_publication)
                else:
                    run_event(initial_publication, PublicationCancel())
                    finish_event(initial_publication)
            if profiler is not None:
                profiler.set_target(
                    duration_s=float(duration_s),
                    total_steps=None if adaptive else int(round(float(duration_s) / self.config.dt_s)),
                    model_start_s=float(start_model),
                )
            if control is not None:
                control.begin(
                    self,
                    total_steps=None if adaptive else int(schedule.final_step),
                    duration_s=float(duration_s), model_start_s=float(start_model),
                    startup=_startup_record(self, profiler),
                )
            control_event: dict[str, Any] | None = None
            started = time.perf_counter()
            final_event = schedule.events[-1] if schedule.events else None
            for event in schedule.events:
                event_started = time.perf_counter()
                if adaptive:
                    # misclibs.F:659-681.  getnewdt lands a step exactly on the
                    # next output time, so under adapt_dt the loop is driven by
                    # the model clock rather than by a step count nobody can
                    # know in advance.  The schedule still decides WHICH model
                    # times are events, so a resumed run keeps the absolute
                    # event grid the original run declared.
                    guard = 0
                    while (
                        self.clock.model_seconds
                        < event.model_time_s - _ADAPTIVE_CLOCK_EPSILON_S
                    ):
                        if control is not None and control.stop_requested:
                            break
                        chosen = self.adaptive_large_step(
                            ramp_down_targets=(
                                RampDownTarget(
                                    "taptim",
                                    event.model_time_s,
                                    cadence.interval_s,
                                    False,
                                ),
                            )
                        )
                        if len(timestep_rows) < frame_row_budget:
                            timestep_rows.append(chosen)
                        if event.history:
                            self.schedule_adaptive_output_step(event.model_time_s)
                        tick = time.perf_counter()
                        self.step()
                        elapsed = time.perf_counter() - tick
                        step_count += 1
                        step_total += elapsed
                        step_minimum = min(step_minimum, elapsed)
                        step_maximum = max(step_maximum, elapsed)
                        if publications is not None and publications.settled:
                            finish_event(publications.join(raise_error=False))
                        if control is not None:
                            control.after_step(elapsed)
                        guard += 1
                        if guard > _ADAPTIVE_STEPS_PER_FRAME_LIMIT:
                            raise RuntimeError(
                                "the adaptive controller took more than "
                                f"{_ADAPTIVE_STEPS_PER_FRAME_LIMIT} large steps "
                                f"inside one {cadence.interval_s} s history "
                                "interval; misclibs.F:654 caps the step at "
                                "twice dtl but nothing caps it from below, so "
                                "this is a collapsing timestep, not a cadence"
                            )
                else:
                    while self.clock.completed_steps < event.step:
                        # One stat() of the control directory per large step
                        # (after_step below); a stop seen there ends the run
                        # at this boundary, before another step is taken.
                        if control is not None and control.stop_requested:
                            break
                        tick = time.perf_counter()
                        self.step()
                        elapsed = time.perf_counter() - tick
                        step_count += 1
                        step_total += elapsed
                        step_minimum = min(step_minimum, elapsed)
                        step_maximum = max(step_maximum, elapsed)
                        # A publication that has finished is reported (or its
                        # error raised) at the first step boundary after it,
                        # not a whole output interval later.
                        if publications is not None and publications.settled:
                            finish_event(publications.join(raise_error=False))
                        if control is not None:
                            control.after_step(elapsed)
                if control is not None and control.stop_requested and not (
                    self.clock.model_seconds >= event.model_time_s - _ADAPTIVE_CLOCK_EPSILON_S
                    if adaptive else self.clock.completed_steps >= event.step
                ):
                    # Stopped between events: nothing is due at this step, so
                    # nothing is taken.  The publication in flight is joined
                    # after the loop, like the last event's.
                    control_event = {"between_events": True}
                    break
                self.synchronize()
                event_wall = time.perf_counter() - event_started
                # An event is the only place this run inspects the state, so it
                # is also the only place a nonfinite trajectory or a broken
                # advective Courant number can be caught before the remaining
                # hours of stepping are spent on it.
                # The previous event's publication is finished and reported
                # before this event is examined or taken: the host arenas are
                # reused, and cm1.F's frames and restarts reach disk -- and
                # reach ``progress`` -- in model order.
                if publications is not None:
                    finish_event(publications.join(raise_error=False))
                with profiled_io(self, "field_scan", f"step-{int(self.clock.completed_steps)}"):
                    checks = self.field_checks()
                courant = checks["advective_courant"]
                parts: list[PublicationPart] = []
                frame_base: dict[str, Any] | None = None
                restart_base: dict[str, Any] | None = None
                try:
                    if event.history:
                        parts.append(self._snapshot_frame(
                            root, frame_index, publisher=publisher, profile=profile,
                            host=arenas["history"], child_host=arenas["nest-child"],
                        ))
                        frame_base = {
                            "index": frame_index,
                            "stepping_wall_seconds": event_wall,
                            "advective_courant": float(courant["number"]),
                            "courant_stop_reason": courant["stop_reason"],
                        }
                        pdcomp = getattr(getattr(self, "lagrangian", None), "config", None)
                        if pdcomp is not None and pdcomp.pdcomp_enabled:
                            with profiled_io(self, "pdcomp", f"frame-{frame_index}"):
                                self._publish_pdcomp(frame_index)
                        else:
                            self._publish_pdcomp(frame_index)
                        frame_index += 1
                    if event.restart:
                        # cm1.F:4247-4252 writes the restart after that step's
                        # output, and the counter it persists is the next frame
                        # index, so a resumed run continues the frame numbering
                        # instead of republishing indices the interrupted run
                        # already wrote.
                        parts.append(self._snapshot_restart(
                            restart_home, restart_number, publisher=restart_publisher,
                            profile=restart_profile, next_history_index=frame_index,
                            host=arenas["restart"], child_host=arenas["nest-child-restart"],
                        ))
                        restart_base = {"number": restart_number}
                        restart_number += 1
                    publication = EventPublication(
                        parts, context=(frame_base, restart_base),
                        background=publications is not None,
                    )
                    if publications is not None:
                        publications.submit(publication)
                except BaseException as failure:
                    # A frame or restart already taken off the device is a
                    # completed state, and the synchronous writer would have
                    # put it on disk before this failure; so does this, then
                    # re-raises.  That includes a parent frame whose nest
                    # child's snapshot failed, and an event whose worker
                    # thread could not be started.  An event the worker did
                    # start is in flight instead, and is drained below.
                    parts.extend(taken_parts(failure))
                    in_flight = publications is not None and publications.in_flight is not None
                    if parts and not in_flight and isinstance(failure, TerminationRequested):
                        # SIGTERM cancels, here as for an event in flight.
                        for part in parts:
                            part.discard()
                            if part.io is not None and not part.io.finished:
                                part.io.finish()
                        failure.add_note(
                            f"{[part.label for part in parts]} taken before SIGTERM were "
                            "not published"
                        )
                    elif parts and not in_flight:
                        taken = EventPublication(parts, background=False)
                        again = " again" if isinstance(failure, KeyboardInterrupt) else ""
                        _notice(
                            None,
                            f"gm1: publishing {', '.join(taken.labels)}, taken before the "
                            f"run stopped (Ctrl-C{again} abandons it)",
                        )
                        try:
                            run_event(taken, PublicationCancel())
                            settle_publication_io(taken)
                        except BaseException as also:  # noqa: BLE001
                            failure.add_note(f"publishing {taken.labels} also failed: {also!r}")
                        else:
                            if taken.error is not None:
                                failure.add_note(
                                    f"publishing {taken.labels} also failed: {taken.error!r}"
                                )
                            elif taken.completed:
                                failure.add_note(
                                    f"{[part.label for part in taken.completed]} "
                                    "taken before this failure were published"
                                )
                    raise
                if publications is None:
                    run_event(publication, PublicationCancel())
                    finish_event(publication)
                # cm1.F:3517-3520 writes this event's output; cm1.F:4385-4392
                # then stops.  The order here is CM1's: this event's frame and
                # restart are joined below, before the run ends.
                if courant["stop_reason"] is not None:
                    stop_event = {
                        "courant": courant,
                        "frame_index": None if frame_base is None else frame_base["index"],
                        "restart_number": (
                            None if restart_base is None else restart_base["number"]
                        ),
                        "step": int(self.clock.completed_steps),
                        "model_time_s": float(self.clock.model_seconds),
                    }
                    break
                # [nest.tornado_criterion]'s stop rules end the run at the first
                # history event after one fires, once that event's frame (and
                # any restart due with it) is published -- the order the
                # Courant stop above follows.  The request is read here, now:
                # the ledger withdraws one when a qualifying run begins while
                # it is pending (the step's sample is taken inside the step,
                # before this check), so a vortex forming between the request
                # and this event keeps the run alive.  A restart-only event
                # does not end the run: the stop is recorded against a
                # published frame.  Nor does the final scheduled event: the run
                # ends there anyway, having integrated the requested duration,
                # and a "stopped" status would say it had not.  That request is
                # recorded unused.
                request = self._nest_criterion_stop_request()
                if request is not None and frame_base is not None and event is not final_event:
                    criterion_event = {
                        "request": dict(request),
                        "frame_index": frame_base["index"],
                        "restart_number": (
                            None if restart_base is None else restart_base["number"]
                        ),
                        "step": int(self.clock.completed_steps),
                        "model_time_s": float(self.clock.model_seconds),
                    }
                    break
                # The run-control request, after the event due at this step
                # was taken: a stop ends the run here, a pause ends it once
                # this event carried a restart.  Neither ends the final event,
                # which completes the requested duration.
                if control is not None and event is not final_event:
                    if control.stop_requested:
                        control_event = {"between_events": False}
                        break
                    if control.pause_requested:
                        if restart_base is not None:
                            control_event = {"between_events": False}
                            break
                        if not any(
                            later.restart for later in schedule.events
                            if later.step > event.step and later is not final_event
                        ):
                            control.refuse_pause(
                                "this run writes no restarts (no --restart-dir)"
                                if restart_home is None else
                                "its next restart is the one at the end of the run, "
                                "so it will complete rather than pause"
                                if final_event is not None and final_event.restart else
                                "it writes no restart before its last step"
                            )
            # The last event, and a run cannot end with a publication unjoined.
            if control is not None:
                control.finishing()
            if publications is not None:
                finish_event(publications.join(raise_error=False))
            control_stop = None
            if control_event is not None:
                # Every restart taken is durable by now: the number before the
                # next one is the last this run wrote, and for a pause the one
                # --restart-from continues from.
                control_stop = control.stop_record(
                    restart_number=(
                        restart_number - 1
                        if restart_number > schedule.first_restart_number else None
                    ),
                    # Frame 0 (history_initial) counts, as it does in the
                    # --events log's frame records and end record.
                    frames_published=(
                        frame_index - schedule.first_history_index + int(history_initial)
                    ),
                    frames_requested=schedule.history_count + int(history_initial),
                )
            if stop_event is not None:
                courant = stop_event["courant"]
                frame_row = next(
                    (row for row in reversed(frames) if row["index"] == stop_event["frame_index"]),
                    None,
                )
                courant_stop = {
                    "reason": courant["stop_reason"],
                    # Named only when this event published one.  A restart
                    # event that trips the stop writes no frame, and
                    # claiming an index it did not publish would send a
                    # reader to a directory that holds an earlier state.
                    "frame_index": stop_event["frame_index"],
                    "frame_directory": (
                        None if stop_event["frame_index"] is None
                        else (
                            frame_row["directory"] if frame_row is not None
                            else os.fspath(root / FRAME_DIRECTORY_FORMAT.format(
                                index=stop_event["frame_index"]
                            ))
                        )
                    ),
                    "restart_number": stop_event["restart_number"],
                    "step": stop_event["step"],
                    "model_time_s": stop_event["model_time_s"],
                    "advective_courant": float(courant["number"]),
                    # This run's own frames, not the whole simulation's:
                    # after a resume the full count would report frames an
                    # earlier run already published as missing.
                    "frames_requested": schedule.history_count,
                    "frames_published": frame_index - schedule.first_history_index,
                    "authority": CM1_COURANT_STOP_AUTHORITY,
                    "note": (
                        "CM1 writes the statistics and history output for the "
                        "state that tripped the stop and only then abandons "
                        "the run; this frame is published for the same reason"
                    ),
                }
            criterion = self._nest_criterion()
            if criterion_event is not None:
                from .nest_tornado_criterion import STOP_HONOURED, criterion_stop_record

                frame_row = next(
                    (row for row in reversed(frames)
                     if row["index"] == criterion_event["frame_index"]),
                    None,
                )
                criterion_stop = criterion_stop_record(
                    criterion_event,
                    # Past frame_row_budget the row is not kept; the directory
                    # is the formatted one, as courant_stop falls back to.
                    frame_directory=(
                        frame_row["directory"] if frame_row is not None
                        else os.fspath(root / FRAME_DIRECTORY_FORMAT.format(
                            index=criterion_event["frame_index"]
                        ))
                    ),
                    frames_requested=schedule.history_count,
                    frames_published=frame_index - schedule.first_history_index,
                )
                criterion.stop_request_disposition = STOP_HONOURED
            elif criterion is not None and criterion.stop_request is not None:
                from .nest_tornado_criterion import STOP_UNUSED_AT_COURANT, STOP_UNUSED_AT_DURATION

                # Pending when the run ended for another reason: CM1's Courant
                # stop, or the final scheduled event of the requested duration.
                criterion.stop_request_disposition = (
                    STOP_UNUSED_AT_COURANT if courant_stop is not None
                    else STOP_UNUSED_AT_DURATION
                )
            wall = time.perf_counter() - started
            simulated = self.clock.model_seconds - start_model
            if wall <= 0 or not math.isfinite(wall):
                raise RuntimeError("invalid integration wall clock")
            # A nest child's frames are staged under their own root
            # (NestDriver.snapshot_child_frame), so its staging directory is
            # emptied and removed here as well.
            nest_child_root = (
                None if getattr(self, "nest_driver", None) is None else root / "nest-child"
            )
            for home in (root, restart_home, nest_child_root):
                if home is None:
                    continue
                try:
                    (home / FRAME_STAGING_DIRECTORY).rmdir()
                except OSError:
                    pass
            receipt = {
                "schema": INTEGRATION_SCHEMA,
                "status": (
                    INTEGRATION_COURANT_STOP_STATUS if courant_stop is not None
                    else INTEGRATION_CRITERION_STOP_STATUS if criterion_stop is not None
                    else INTEGRATION_CONTROL_STOP_STATUS
                    if control_stop is not None and control_stop["request"] == "stop"
                    else INTEGRATION_CONTROL_PAUSE_STATUS if control_stop is not None
                    else INTEGRATION_STATUS
                ),
                "courant_stop": courant_stop,
                # Present only for a run given a control directory or an
                # event log, so every other receipt keeps exactly its keys.
                **({} if control is None else {
                    "control_stop": control_stop,
                    "run_control": control.record(),
                }),
                **({} if not history_initial else {
                    "history_initial": {
                        "frame_index": INITIAL_CONDITION_FRAME_INDEX,
                        "authority": "cm1.F:3943 (the nstep=0 writeout)",
                    },
                }),
                # Present only for a nest that declares a tornado criterion, so
                # every other receipt keeps exactly the keys it had.
                **({} if self._nest_criterion() is None
                   else {"criterion_stop": criterion_stop}),
                "config": self.config.record(), "source": self.source_identity,
                # Which of CM1's dynamical selectors this run actually used,
                # and what each one rests on.  Carried in the receipt as well
                # as in `gpuwm-cm1 doctor` because a reader who has only this
                # artifact must still be able to tell a measured number from
                # an unmeasured one.
                "dynamics_validation": self.dynamics_validation,
                "runtime_source": self.runtime_source_identity,
                "cadence": cadence.record(),
                "schedule": schedule.record(),
                "initial_advective_courant": initial_courant,
                # CM1 &param9 as this run resolved it, with a loud UNVALIDATED
                # row for every product selected but not checked in this pass.
                "output_products": self.output.record(),
                "history_root": os.fspath(root),
                "history_publisher": None if publisher is None else os.fspath(publisher),
                "history_profile": profile.record(),
                "restart_root": None if restart_home is None else os.fspath(restart_home),
                "restart_publisher": (
                    None if restart_profile is None else os.fspath(restart_publisher)
                ),
                "restart_profile": (
                    None if restart_profile is None else restart_profile.record()
                ),
                **({} if child_restart_profile is None else {
                    "nest_child_restart_profile": child_restart_profile.record(),
                }),
                "restart_interval_s": restart_interval_s,
                "restart_reset_frqtim": bool(restart_reset_frqtim),
                "steps_before_integration": start_step,
                "completed_steps": self.clock.completed_steps,
                "executed_steps": step_count,
                "model_seconds_at_start": start_model,
                "model_seconds_at_end": self.clock.model_seconds,
                # cm1.F:3512-3513, at the end of the integration.  A run that
                # ends at CM1's advective-Courant stop still records where the
                # domain had travelled to, because the receipt is written for
                # the stopping state exactly as cm1.F:3517-3520 writes it.
                "domain_location": self.clock.domain_location_record(),
                "simulated_seconds": simulated,
                "wall_seconds": wall,
                "simulated_seconds_per_wall_second": simulated / wall,
                "steps_per_wall_second": step_count / wall,
                "step_wall_seconds": {
                    "count": step_count,
                    "total": step_total,
                    # A run stopped before its first step took none.
                    "mean": step_total / step_count if step_count else None,
                    "minimum": None if step_minimum is math.inf else step_minimum,
                    "maximum": step_maximum,
                },
                "adaptive_timestep": (
                    None if not adaptive else {
                        "authority": CM1_ADAPTIVE_TIMESTEP_AUTHORITY,
                        "dtl_s": self.config.dtl_s,
                        "final_dt_s": self.config.dt_s,
                        "final_acoustic_substeps": self.config.acoustic_substeps,
                        "decisions_retained": len(timestep_rows),
                        "decisions": timestep_rows,
                    }
                ),
                "frames": frames,
                "frame_rows_retained": len(frames),
                "frame_rows_dropped": frames_dropped,
                "frame_row_budget": frame_row_budget,
                "restarts": restarts,
                "restart_rows_retained": len(restarts),
                "restart_rows_dropped": restarts_dropped,
                "restart_row_budget": restart_row_budget,
                "last_frame_publication": last_receipt,
                "last_restart_publication": last_restart,
                # How this run's events reached disk.  Every row above was
                # reported only after its event had been joined.
                "publication": {
                    "mode": "background" if publications is not None else "synchronous",
                    **({} if publications is None else publications.record()),
                    "host_arenas": {
                        kind: arena.record() for kind, arena in arenas.items()
                        if arena.capacity_bytes
                    },
                },
                "next_history_index": frame_index,
                "next_restart_number": restart_number,
                "sampled_peak_device_used_bytes": self.peak_bytes("device_used_bytes"),
                "sampled_peak_registered_model_bytes": self.peak_bytes("registered_model_bytes"),
                "memory_samples": self.memory_samples,
                "memory_samples_dropped": self.memory_samples_dropped,
                "memory_sample_budget": memory_sample_budget,
                "driver_component_warmup": self.warm_component,
                "component_scratch_reserve": self.component_scratch_reserve,
                **self._reflectivity_record(),
                "lagrangian": self._publish_lagrangian_track(root),
                **self._nest_parcel_track(),
                **self._nest_child_window(),
                **self._nest_tornado_criterion(),
                # CM1 output_*budget and the GM1 vorticity budget: the term
                # lists, their authority, and closure measured by value.
                **({} if getattr(self, "budget", None) is None else {"budgets": self.budget.record()}),
                # CM1's output_* diagnostics for this run, when the document
                # selected any: the swath extrema, and the tornado report if
                # the document asked for one.  Absent -- not null -- when the
                # document declared no [diagnostics] table.
                **({} if self.tornado is None
                   else {"tornado_diagnostics": self.tornado.record()}),
                "phase_host_wall_seconds_all_steps": self.phase_wall_seconds,
                "dynamics_allocation_plan": cartesian_dynamics_allocation_plan(self.config),
                "closure_allocation_plan": cartesian_closure_allocation_plan(self.config),
                "field_checks": checks,
                "last_step_operators": self.finish.step_receipts,
                "last_step_dynamics": self.dynamics.last_receipts,
                "last_step_closure": self.closure.last_receipts,
                "timing_scope": "complete CUDA stepping plus the per-event field scan, history publication and restart publication of this integration; initialization, compilation warmup and receipt writing excluded",
                **({} if profiler is None else {"performance": profiler.record(
                    wall_seconds=wall,
                    runtime_startup=getattr(self, "startup_wall_seconds", None),
                )}),
                "claim_limit": (
                    "abandoned at CM1's advective Courant stop after publishing "
                    "the output that reached it; the requested model duration was "
                    "not integrated and no part of this record is a completed run"
                    if courant_stop is not None else
                    "ended by the declared nest.tornado_criterion stop rule after "
                    "publishing the frame of the event that followed it; the "
                    "requested model duration was not integrated, by design, and "
                    "the verdicts cover the integrated interval only"
                    if criterion_stop is not None else
                    "ended at a run-control request after publishing every event "
                    "taken before it; the requested model duration was not "
                    "integrated"
                    + ("" if control_stop is None or control_stop["request"] != "pause"
                       else ", and the last restart continues it")
                    if control_stop is not None else
                    "executed large steps to the requested model duration on one "
                    "admitted grid and published the declared history frames and "
                    "restarts; not native-CM1 trajectory identity, restart "
                    "identity, or forecast skill"
                ),
            }
            if courant_stop is not None:
                raise CartesianCourantStop(
                    f"{courant_stop['reason']}; CM1 writes the tripping state's "
                    "output and then stops (cm1.F:3517-3520, cm1.F:4385-4392), so "
                    f"{courant_stop['frames_published']} of "
                    f"{courant_stop['frames_requested']} frames are published "
                    f"under {os.fspath(root)} and the run ends there",
                    receipt=receipt,
                )
            return receipt
        except BaseException as failure:
            termination = self._drain_publication(publications, failure)
            # The tornado criterion's per-sample rows are the only record a
            # run that died keeps of its samples: closed in place if it took
            # any, removed if it took none so the empty roots below go too.
            criterion = self._nest_criterion()
            if criterion is not None:
                try:
                    criterion.abandon()
                except Exception as also:  # noqa: BLE001
                    failure.add_note(f"closing the tornado-criterion time series also failed: {also!r}")
            # The child window's writer thread, stopped without a receipt: a
            # stream without one is refused by name when read, so nothing that
            # was not completely written can be mistaken for a whole stream.
            driver = getattr(self, "nest_driver", None)
            window = None if driver is None else getattr(driver, "window", None)
            if window is not None:
                try:
                    window.abandon()
                except Exception as also:  # noqa: BLE001
                    failure.add_note(f"stopping the child window stream also failed: {also!r}")
            if getattr(self, "nest_driver", None) is not None:
                # The child's staging root, emptied by its last rename.
                self._discard_empty_history_root(root / "nest-child")
            self._discard_empty_history_root(root)
            if restart_home is not None:
                self._discard_empty_history_root(restart_home)
            if termination is not None:
                # SIGTERM arrived while an interrupted run waited for its
                # publication: the process still ends as a SIGTERM (143).
                raise termination
            raise
        finally:
            # Belt and braces: an interrupt that landed inside the handler
            # above must still not leave a worker writing into the arenas
            # released below.
            if publications is not None and publications.in_flight is not None:
                publications.abort()
            terminating.__exit__(None, None, None)
            # The last events' snapshots are views of the arenas; drop them so
            # the page-locked pages go back to the driver, not just to the pool.
            publication = parts = taken = None
            for arena in arenas.values():
                arena.release(self.cp)
