"""CM1's restart pair for the Cartesian lane, proved through real envelope bytes.

Nothing here opens a CUDA device.  Storage is numpy and the stepping is a
deterministic host transformation, but every part of the persistence path is
the shipped one: the real restart profile, the real arena writer, the real
``cm1-forecast-publish`` binary built from this tree writing a real ``CM1FCP1``
envelope, the real ``restore`` reading it back, and the real restore that puts
the words into the live allocations.

The claim under test is CM1's own: ``restart_write.F`` saves "all information
that is carried from one timestep to the next" (README.restart) and
``restart_read.F`` reads it back, so a run that stops and resumes lands
exactly where an uninterrupted one lands.  A resume that is merely close is
worthless in a lane whose product is bit-exactness, so what is asserted is raw
words and raw bits.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_door import (
    build_runtime_config,
    require_run_schedule,
    resume_selection,
)
from cm1core.config import load_config
from cm1core.dynamics_validation_register import dynamics_validation_record
from cm1core.errors import CM1Refusal
from cm1core.arwen_wsm6_component import (
    WSM6_RESTART_REQUIRED,
    WSM6_SURFACE_SCRATCH,
)
from cm1core.cartesian_restart import (
    ARENA_FILENAME,
    CLOSURE_STEP_NAME,
    ASQ_LENGTH,
    ASQ_SCALAR_FORMAT,
    CARTESIAN_RESTART_PUBLICATION_SCHEMA,
    ENVELOPE_FILENAME,
    FORECAST_CHECKPOINT_MAGIC,
    HISTORY_COUNTER_NAME,
    MASS1_SCALAR_NAME,
    RECEIPT_FILENAME,
    RESTART_COUNTER_NAME,
    RESTART_DIRECTORY_FORMAT,
    cartesian_restart_profile,
    read_cartesian_restart,
    require_cartesian_restart_boundary,
    require_restart_receipt,
    require_resumable,
    restart_arena_disposition,
    restart_scalars,
    restore_cartesian_restart,
    write_cartesian_restart,
)
from cm1core.cartesian_output_products import CartesianOutputProducts
from cm1core.cuda_cartesian_dynamics import (
    cartesian_dynamics_restart_state,
    cartesian_dynamics_state_shapes,
)
from cm1core.cuda_cartesian_runtime import (
    CartesianCourantStop,
    FRAME_DIRECTORY_FORMAT,
    FRAME_STAGING_DIRECTORY,
    INITIAL_CONDITION_SCHEMA,
    MEMORY_SAMPLE_KEYS,
    CudaCartesianRuntime,
)
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.gm1_idealized_residency import DeviceBufferRegistry
from cm1core.gm1_runtime import (
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    MOVE_SPEED_X_NAME,
    MOVE_SPEED_Y_NAME,
    RuntimeClock,
    RuntimeConfig,
    StepPhase,
)
from cm1core.grid_geometry import CartesianGridGeometry


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def forecast_publisher(tmp_path_factory):
    """The actual cm1-forecast-publish binary, built from this tree."""

    target = tmp_path_factory.mktemp("gm1-restart-publisher")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "cargo", "build", "--release", "--bin", "cm1_forecast_publish",
            "--manifest-path", str(root / "rust/cm1-state/Cargo.toml"),
            "--target-dir", str(target),
        ],
        check=True, capture_output=True,
    )
    return (target / "release" / "cm1_forecast_publish").resolve(strict=True)


class Stream:
    def __init__(self) -> None:
        self.depth = 0
        self.syncs = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_):
        self.depth -= 1

    def synchronize(self) -> None:
        self.syncs += 1


class CpuStorage:
    """Only array storage, the host copy and element-wise arithmetic.

    No kernel is emulated.  The ufuncs are here because CM1's advective
    Courant norm (``storm_diagnostics._reduce_courant``, misclibs.F:2612-2631)
    is written as ``cp`` ufunc calls rather than a compiled kernel, so the
    shipped reducer runs here unmodified and this double measures the same
    number the device does.
    """

    float32 = np.float32
    float64 = np.float64
    ndarray = np.ndarray
    dtype = staticmethod(np.dtype)
    asnumpy = staticmethod(np.asarray)
    asarray = staticmethod(np.asarray)
    copyto = staticmethod(np.copyto)
    add = staticmethod(np.add)
    subtract = staticmethod(np.subtract)
    power = staticmethod(np.power)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    multiply = staticmethod(np.multiply)
    sqrt = staticmethod(np.sqrt)
    amax = staticmethod(np.amax)

    def __init__(self, stream) -> None:
        self.stream = stream

    def empty(self, shape, *, dtype):
        assert self.stream.depth > 0
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype):
        assert self.stream.depth > 0
        return np.zeros(shape, dtype=dtype)


def _config(nx: int = 9, ny: int = 11, nz: int = 8, **kwargs) -> RuntimeConfig:
    # A bare RuntimeConfig carries &param9's raw table defaults, but no
    # document ever reaches the runtime that way: cartesian_output_selection
    # rebinds output_dbz from physics.output_dbz and the swath/rotation words
    # from [diagnostics], all off when the table is absent.  Bind the same
    # selection here, so the double runs what a plain document runs (and
    # never asks for the swath fold, a CUDA kernel it does not emulate).
    from cm1core.cm1_output_selection import DEFAULT_OUTPUT_SELECTION, PER_STEP_PRODUCTS

    kwargs.setdefault("output", DEFAULT_OUTPUT_SELECTION.with_values({
        "output_dbz": int(bool(kwargs.get("output_dbz", False))),
        **{key: 0 for key in PER_STEP_PRODUCTS},
    }))
    return RuntimeConfig(
        CartesianGridGeometry(nx, ny, nz, 500.0, 500.0, 500.0), **kwargs
    )


# Exact binary32 constants: the stepping below must be reproducible word for
# word on any host, so every factor is a power of two or a sum of them.
_DECAY = np.float32(0.9990234375)
_COUPLE = np.float32(0.0009765625)
_STEP_FORCING = np.float32(0.001953125)

#: cuda_cartesian_dynamics._IMMUTABLE_BASE_FIELDS[:3]: mass-shaped base-state
#: fields the restart rebuilds (``physical_base_fields``) and never carries.
_REBUILT_BASE_FIELDS = ("pi0", "th0", "qv0")


class CpuCartesianRuntime:
    """A CudaCartesianRuntime-shaped host double with a deterministic step.

    Every structure the restart profile addresses is here, with the real
    runtime's aliasing: the wind fields are strided views into their padded
    owners, so a restart that persisted the physical rectangles instead of the
    owners would lose the halo words the next step's stencils read, and this
    double would show it.

    The stepping is not CM1.  It is a fixed, state-coupled float32 pass over
    every carrier in a fixed order, which is all that is needed for the claim
    under test: two runs that hold identical words take identical steps, and a
    single restored word that is wrong changes every later word.
    """

    def __init__(self, config: RuntimeConfig, *, seed: int = 1,
                 execution_identity: str = "11" * 32) -> None:
        self.config = config
        self.stream = Stream()
        self.cp = CpuStorage(self.stream)
        self.clock = RuntimeClock(config)
        self.registry = DeviceBufferRegistry(
            self.cp, self.stream, budget_bytes=256 * 1024**2
        )
        self.trace_phases = False
        self.memory_sample_budget = None
        self.memory_samples: list = []
        self.memory_samples_dropped = 0
        self._memory_peaks = dict.fromkeys(MEMORY_SAMPLE_KEYS, 0)
        self.warm_component = {"levels": config.grid.nz}
        self.component_scratch_reserve = {"levels": config.grid.nz}
        self.phase_wall_seconds: dict = {}
        self.source_identity = {"kind": "cpu-double", "seed": seed}
        # The real register, keyed off this run's own configuration, so the
        # receipt this double publishes names the same unvalidated dynamics
        # capabilities a device run's would.
        self.dynamics_validation = dynamics_validation_record(config)
        self.runtime_source_identity = {"aggregate_sha256": execution_identity}
        g = config.grid
        padded = terrain_open_sound_shapes(g.nx, g.ny, g.nz)
        state = cartesian_dynamics_restart_state(config)["carried"]
        rng = np.random.default_rng(seed)

        arrays = {
            name: rng.standard_normal(padded[name]).astype(np.float32)
            for name in state["padded"]
        }
        for name in ("uh", "vh", "ruf", "rvf"):
            arrays[name] = np.ones(padded[name], dtype=np.float32)
        fields = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in state["fields"].items()
        }
        # The real runtime exposes the wind fields as strided physical views
        # into the padded owners it allocated; so does this.
        for name, stagger in (("ua", "u"), ("u3d", "u"), ("va", "v"),
                              ("v3d", "v"), ("wa", "w"), ("w3d", "w")):
            fields[name] = arrays[name][g.physical_slices(stagger, halo_xy=3, halo_z=1)]
        held = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in state["held"].items()
        }
        tke = {
            name: np.abs(rng.standard_normal(shape)).astype(np.float32)
            for name, shape in state["tke"].items()
        }
        persistence = {
            name: rng.standard_normal(
                g.surface_shape if name in WSM6_SURFACE_SCRATCH else g.mass_shape
            ).astype(np.float32)
            for name in WSM6_RESTART_REQUIRED
        }
        # The two carriers a history frame needs that are not model state:
        # both are immutable, and both are rebuilt rather than carried.
        metrics = {
            "terrain_msl": np.zeros(g.surface_shape, dtype=np.float32),
            # param.F's uniform-vertical branch sets mh to exactly one;
            # misclibs.F:2612-2631 multiplies the vertical term by it.
            "mh": np.ones(g.mass_shape, dtype=np.float32),
            "zf": np.broadcast_to(
                (np.arange(g.nz + 1, dtype=np.float32) * np.float32(g.dz_m))[:, None, None],
                g.w_shape,
            ),
        }
        # base.F's compact base-state columns and the zh levels.  A restart
        # rebuilds both rather than carrying them, and the &param9 engine reads
        # both at publication time (output_basestate, output_zh, output_th), so
        # the double holds them exactly where the device runtime does: compact
        # profiles, plus read-only mass-shaped broadcasts in fields/metrics.
        levels = np.arange(g.nz, dtype=np.float32)
        base_profiles = {
            "pi0": (np.float32(1.0) - np.float32(0.015625) * levels).astype(np.float32),
            "th0": (np.float32(300.0) + levels).astype(np.float32),
            "qv0": np.full(g.nz, np.float32(0.0078125), dtype=np.float32),
            "u0": levels.copy(),
            "v0": (-levels).astype(np.float32),
        }
        metrics["zh"] = np.broadcast_to(
            ((levels + np.float32(0.5)) * np.float32(g.dz_m))[:, None, None], g.mass_shape
        )
        for name in _REBUILT_BASE_FIELDS:
            fields[name] = np.broadcast_to(base_profiles[name][:, None, None], g.mass_shape)
        self.dynamics = SimpleNamespace(
            arrays=arrays, fields=fields, held=held, tke=tke,
            metrics=metrics, base_profiles=base_profiles, last_receipts={},
            _expected_phase=("held", 1), mass1_f64=0.0,
        )
        self.finish = SimpleNamespace(
            persistence=persistence,
            asq=np.zeros((ASQ_LENGTH,), dtype=np.float64),
            completed_physics_step=0, completed_solve3_step=0, step_receipts={},
            initial_mass1_f64=0.0,
        )
        # CartesianClosure's own cadence tokens, with its own guard: it
        # refuses any step that is not `_held_step + 1`, so a resume that
        # restored every carrier and left this at zero would refuse its own
        # next large step after the device was opened and the arena uploaded.
        self.closure = SimpleNamespace(
            last_receipts={}, _held_step=0, _pending_step=None, _tke_stage=3,
        )
        # The dry-mass reference CM1 fixes once from the initial state
        # (restart_write.F:412) and corrects every later step back to.
        self.dynamics.mass1_f64 = self.finish.initial_mass1_f64 = self._dry_mass()
        # CudaCartesianRuntime.__init__ builds the &param9 engine the same way,
        # and publish_history materializes it before every frame is staged.  A
        # RuntimeConfig built without a selection publishes no &param9 rows
        # (cartesian_history._carrier_table), so it gets no engine either.
        self.output_fields: dict = {}
        self.output = None if config.output is None else CartesianOutputProducts(self)

    # ---- the ordered owners this double evolves and compares

    def owners(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, value in self.dynamics.arrays.items():
            out[f"padded.{name}"] = value
        for name, value in self.dynamics.fields.items():
            if name in self.dynamics.arrays:
                continue  # a strided view of a padded owner already listed
            if name in _REBUILT_BASE_FIELDS:
                continue  # read-only base state the restart rebuilds
            out[f"fields.{name}"] = value
        for name, value in self.dynamics.held.items():
            out[f"held.{name}"] = value
        for name, value in self.dynamics.tke.items():
            out[f"tke.{name}"] = value
        for name, value in self.finish.persistence.items():
            out[f"component.{name}"] = value
        return out

    def evolving(self) -> dict[str, np.ndarray]:
        """The owners a step changes.

        CM1 param.F's uniform-horizontal branch sets uh/vh/ruf/rvf to exactly
        one and no operator writes them again, which is why the restart
        rebuilds them rather than carrying them; this double holds them to the
        same rule so that "not carried" and "not changed" stay the same claim.
        """

        skip = {"padded.uh", "padded.vh", "padded.ruf", "padded.rvf"}
        return {name: value for name, value in self.owners().items() if name not in skip}

    def _dry_mass(self) -> float:
        total = np.float64(0.0)
        for name, value in sorted(self.evolving().items()):
            total += np.float64(np.sum(value.reshape(-1)[::7], dtype=np.float64))
        return float(total)

    # ---- CartesianBackend

    def execute(self, operation, *, step: int) -> None:
        if operation.phase is StepPhase.CLOSURE:
            if step != self.closure._held_step + 1:
                raise RuntimeError(
                    "closure requires the next complete large-step boundary"
                )
            self.closure._pending_step = step
        elif operation.phase is StepPhase.HELD_TENDENCIES:
            assert self.closure._pending_step == step
            self.closure._held_step = step
            self.closure._pending_step = None
            self.closure._tke_stage = 0
        elif operation.phase is StepPhase.TKE:
            assert operation.nrk == self.closure._tke_stage + 1
            self.closure._tke_stage = operation.nrk
        elif operation.phase is StepPhase.SOLVE3 and operation.nrk is None:
            self._evolve(step)
            # CudaCartesianRuntime's SOLVE3 arm: solve3.F:829-1130 folds the
            # swaths on every large step.  The fold is a CUDA kernel this
            # double does not emulate, so a selection with a per-step swath
            # cannot be stepped here (see _config); per_step returns at once
            # for every selection this file steps.
            if self.output is not None:
                self.output.per_step(step)

    def synchronize(self) -> None:
        self.stream.synchronize()

    def step(self) -> None:
        self.clock.step(self)
        self.finish.completed_physics_step = self.clock.completed_steps
        self.finish.completed_solve3_step = self.clock.completed_steps

    def _evolve(self, step: int) -> None:
        forcing = np.float32(step) * _STEP_FORCING
        carry = np.float32(self.finish.initial_mass1_f64 % 1.0)
        for name, value in sorted(self.evolving().items()):
            flat = value.reshape(-1)
            flat *= _DECAY
            flat += carry * _COUPLE + forcing
            carry = np.float32(flat[-1])
        for index in range(ASQ_LENGTH):
            self.finish.asq[index] += np.float64(carry) * np.float64(index + 1)

    # ---- the real integration entry point, bound to this host double
    #
    # These are CudaCartesianRuntime's own methods, unmodified.  The stepping
    # underneath them is a double; the schedule, the staging, the atomic
    # publication and the restart cadence are the shipped code.

    integrate = CudaCartesianRuntime.integrate
    initial_condition = CudaCartesianRuntime.initial_condition
    # misclibs.F:2714's stop and cm1.F:3214-3237's getdbz schedule: the
    # integration consults both on every event, so the double carries the
    # shipped implementations rather than shapes of them.
    courant_number = CudaCartesianRuntime.courant_number
    advective_courant_record = CudaCartesianRuntime.advective_courant_record
    require_advective_courant = CudaCartesianRuntime.require_advective_courant
    schedule_reflectivity = CudaCartesianRuntime.schedule_reflectivity
    _reflectivity_record = CudaCartesianRuntime._reflectivity_record
    _discard_empty_history_root = staticmethod(
        CudaCartesianRuntime._discard_empty_history_root
    )
    # The integration opens the swaths, advances the parcels and publishes the
    # &param9 products on every run, so the double carries the shipped methods
    # rather than shapes of them; each returns immediately when the document
    # declared no [diagnostics], [lagrangian] or [output] table.
    tornado = None
    lagrangian = None
    output = None
    # CudaCartesianRuntime's own class-level defaults (cuda_cartesian_runtime
    # .py:212-227), which the shipped integrate/_publish_frame and the door
    # read on every run: no Lagrangian track root, no adaptive frame
    # controller, no resolved frame steps yet, and no live nest -- every
    # document in this file is a fixed-frame single-domain run.
    _lagrangian_root = None
    adaptive_frame = None
    history_frame_steps: frozenset = frozenset()
    nest_driver = None
    open_tornado_diagnostics = CudaCartesianRuntime.open_tornado_diagnostics
    _advance_lagrangian = CudaCartesianRuntime._advance_lagrangian
    _publish_eulerian_diagnostics = CudaCartesianRuntime._publish_eulerian_diagnostics
    _publish_pdcomp = CudaCartesianRuntime._publish_pdcomp
    _publish_lagrangian_track = CudaCartesianRuntime._publish_lagrangian_track
    # a54d70f: integrate opens (and the receipt publishes) the live nest
    # child's parcel track; with nest_driver None it returns {} at once.
    _nest_parcel_track = CudaCartesianRuntime._nest_parcel_track
    # [lagrangian.nest_child_window]: integrate opens the child's window stream
    # beside that track and closes it into the receipt; with nest_driver None
    # (or no window on it) each returns at once.
    _nest_child_window = CudaCartesianRuntime._nest_child_window
    # [nest.tornado_criterion]: integrate opens its time series, consults its
    # stop request at every event and closes it into the receipt; with
    # nest_driver None (or no criterion on it) each returns at once.
    _nest_criterion = CudaCartesianRuntime._nest_criterion
    _nest_tornado_criterion = CudaCartesianRuntime._nest_tornado_criterion
    _nest_criterion_stop_request = CudaCartesianRuntime._nest_criterion_stop_request
    publish_history = CudaCartesianRuntime.publish_history
    publish_restart = CudaCartesianRuntime.publish_restart
    restore_restart = CudaCartesianRuntime.restore_restart
    _publish_frame = CudaCartesianRuntime._publish_frame
    _publish_restart = CudaCartesianRuntime._publish_restart
    _snapshot_frame = CudaCartesianRuntime._snapshot_frame
    _snapshot_restart = CudaCartesianRuntime._snapshot_restart
    _commit_inline = CudaCartesianRuntime._commit_inline
    _materialize_history_products = CudaCartesianRuntime._materialize_history_products
    _drain_publication = staticmethod(CudaCartesianRuntime._drain_publication)
    _frame_location = CudaCartesianRuntime._frame_location
    _fsync_directory = staticmethod(CudaCartesianRuntime._fsync_directory)

    def peak_bytes(self, key: str) -> int:
        return int(self._memory_peaks.get(key, 0))

    def field_checks(self) -> dict:
        out = {}
        for name in ("u3d", "v3d", "w3d", "pp3d", "th3d", "q3d", "rho", "prs"):
            value = self.dynamics.fields[name]
            out[name] = {
                "finite": bool(np.all(np.isfinite(value))),
                "minimum": float(np.min(value)), "maximum": float(np.max(value)),
            }
        if not all(row["finite"] for row in out.values()):
            raise RuntimeError("completed model state contains nonfinite physical words")
        # misclibs.F:2714 measured and recorded, never raised here: cm1.F
        # writes the tripping state's output first.  integrate() reads
        # "number" and "stop_reason" off this row at every event.
        out["advective_courant"] = self.advective_courant_record()
        return out

    # ---- comparison

    def raw_state(self) -> dict[str, bytes]:
        rows = {
            name: value.astype("<f4", copy=False).tobytes()
            for name, value in sorted(self.owners().items())
        }
        rows["#asq"] = self.finish.asq.astype("<f8", copy=False).tobytes()
        rows["#clock"] = struct.pack(
            "<Qdd", self.clock.completed_steps,
            self.clock.domainlocx_m, self.clock.domainlocy_m,
        )
        rows["#mass1"] = struct.pack("<d", self.finish.initial_mass1_f64)
        rows["#cadence"] = struct.pack(
            "<QQQ", self.closure._held_step,
            self.finish.completed_physics_step, self.finish.completed_solve3_step,
        )
        return rows


def _written(runtime, directory, publisher, **kwargs):
    return write_cartesian_restart(
        runtime, directory, publisher=publisher, **kwargs
    )


# ------------------------------------------------- profile and inventory


def test_the_profile_carries_every_mutable_allocation_and_names_the_rest() -> None:
    """A field added to the dynamics and not to the restart is silent data loss.

    The profile is built from ``cartesian_dynamics_state_shapes``, the same
    table the allocation plan prices the run from, so the two cannot drift:
    every group is either carried by name or excluded by name with a reason.
    """

    config = _config()
    profile = cartesian_restart_profile(config)
    shapes = cartesian_dynamics_state_shapes(config)
    carried = {carrier.name for carrier in profile.carriers}

    for name in shapes["persistent_padded"]:
        expected = f"padded.{name}"
        if name in ("uh", "vh", "ruf", "rvf"):
            assert expected not in carried, name
        else:
            assert expected in carried, name
    for name in shapes["physical_fields"]:
        expected = f"fields.{name}"
        if name in ("pi0", "th0", "qv0", "rho0"):
            assert expected not in carried, name
        else:
            assert expected in carried, name
    for group in ("held", "tke"):
        for name in shapes[group]:
            assert f"{group}.{name}" in carried, name
    for name in WSM6_RESTART_REQUIRED:
        assert f"component.{name}" in carried, name
    assert len(carried) == len(profile.carriers) == 51

    # Every exclusion states why the state is rebuilt rather than carried.
    rebuilt = profile.rebuilt
    assert set(rebuilt) == {
        "immutable_metrics", "compact_base_profiles", "physical_base_fields",
        "uniform_grid_coefficients", "physical_momentum_views",
    }
    assert "README.namelist:1512-1516" in rebuilt["physical_base_fields"]
    assert all(len(reason) > 40 for reason in rebuilt.values())


def test_the_wind_carriers_are_the_padded_owners_not_the_physical_views() -> None:
    """The next step's fifth-order stencils read halo words.

    A restart that persisted ``u3d``'s physical rectangle would restore an
    interior that agrees and a halo that does not, and the first advected
    point next to the boundary would differ on the very next step.
    """

    config = _config()
    profile = cartesian_restart_profile(config)
    shapes = profile.shapes()
    g = config.grid
    padded = terrain_open_sound_shapes(g.nx, g.ny, g.nz)
    for name in ("u3d", "v3d", "w3d", "ua", "va", "wa"):
        assert shapes[f"padded.{name}"] == padded[name]
        assert f"fields.{name}" not in shapes
    assert shapes["padded.u3d"] != g.u_shape


def test_an_oversized_grid_refuses_before_a_device_word_is_read() -> None:
    """Three GPU-hours into a run is not when to learn it cannot checkpoint."""

    with pytest.raises(ValueError) as caught:
        cartesian_restart_profile(_config(nx=2600, ny=2600, nz=180))
    assert "CM1FCP1" in str(caught.value)
    assert "could never be continued" in str(caught.value) or "never write" in str(caught.value)


# ------------------------------------------------------------ the boundary


def test_the_restart_boundary_admits_step_zero_and_refuses_a_split_cadence() -> None:
    """cm1.F:3110-3115 writes a restart at nstep=0; a mid-step one is refused."""

    runtime = CpuCartesianRuntime(_config())
    assert require_cartesian_restart_boundary(runtime) == 0
    runtime.step()
    assert require_cartesian_restart_boundary(runtime) == 1

    runtime.finish.completed_solve3_step = 0
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "pre- and post-physics" in str(caught.value)

    runtime.finish.completed_solve3_step = 1
    runtime.dynamics._expected_phase = ("acoustic", 2)
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "mid-RK-stage" in str(caught.value)

    runtime.dynamics._expected_phase = ("held", 1)
    runtime.clock.failed = True
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "undefined" in str(caught.value)


# --------------------------------------------------- the whole deliverable


def test_a_stopped_run_resumes_bit_for_bit_through_real_envelope_bytes(
    forecast_publisher, tmp_path
) -> None:
    """The single disqualifying gap, closed and proved.

    An uninterrupted forty-step run, and the same run stopped at twenty,
    written to a real CM1FCP1 envelope, read back through the Rust boundary
    into a freshly built runtime, and stepped the remaining twenty.  Every
    carrier word, the moisture budget, the model clock, the domain-location
    integral and the dry-mass reference are compared as raw bytes.
    """

    config = _config(u_move_m_s=12.5, v_move_m_s=3.0)

    uninterrupted = CpuCartesianRuntime(config, seed=7)
    for _ in range(40):
        uninterrupted.step()

    stopped = CpuCartesianRuntime(config, seed=7)
    for _ in range(20):
        stopped.step()
    halfway = stopped.raw_state()
    receipt = _written(
        stopped, tmp_path / "restart-000001", forecast_publisher,
        restart_number=1, next_history_index=3,
    )
    assert receipt["schema"] == CARTESIAN_RESTART_PUBLICATION_SCHEMA
    assert receipt["step"] == 20
    assert receipt["finite_checked_by_rust"] is True
    envelope = Path(receipt["forecast_envelope"]["path"])
    assert envelope.read_bytes().startswith(FORECAST_CHECKPOINT_MAGIC.encode("ascii"))

    resumed = CpuCartesianRuntime(config, seed=7)
    # A freshly built runtime is the initial condition, not the halfway state.
    assert resumed.raw_state() != halfway
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    record = restore_cartesian_restart(resumed, restored)
    assert record["step"] == 20
    assert record["next_history_index"] == 3
    assert record["restart_number"] == 1
    # Word for word the state the interrupted run stopped at.
    assert resumed.raw_state() == halfway

    for _ in range(20):
        resumed.step()

    assert resumed.clock.completed_steps == uninterrupted.clock.completed_steps == 40
    left, right = uninterrupted.raw_state(), resumed.raw_state()
    assert set(left) == set(right)
    differing = sorted(name for name in left if left[name] != right[name])
    assert differing == [], differing


def test_the_resumed_run_differs_when_one_persisted_word_is_wrong(
    forecast_publisher, tmp_path
) -> None:
    """The bit-for-bit claim is a claim, not an artefact of a weak comparison.

    One float32 word of one carrier is perturbed after the restart is written.
    If the comparison above could pass without the restore being exact, this
    would pass too.
    """

    config = _config(u_move_m_s=12.5, v_move_m_s=3.0)
    uninterrupted = CpuCartesianRuntime(config, seed=11)
    for _ in range(24):
        uninterrupted.step()

    stopped = CpuCartesianRuntime(config, seed=11)
    for _ in range(12):
        stopped.step()
    # A halo word of the padded owner, outside every physical rectangle: the
    # one place a restart that carried only the physical fields would differ.
    stopped.dynamics.arrays["u3d"][0, 0, 0] = np.float32(
        stopped.dynamics.arrays["u3d"][0, 0, 0] + np.float32(1.0)
    )
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)

    resumed = CpuCartesianRuntime(config, seed=11)
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    restore_cartesian_restart(resumed, restored)
    for _ in range(12):
        resumed.step()
    assert resumed.raw_state() != uninterrupted.raw_state()


def test_the_carried_scalars_are_read_back_and_not_recomputed(
    forecast_publisher, tmp_path
) -> None:
    """restart_write.F:412 and :657 by value, and the counters beside them.

    ``mass1`` is CM1's dry-air mass reference, fixed once from the initial
    state.  A resumed runtime computes its own from the state it was built
    with, and this asserts the persisted one replaces it: otherwise every
    later dry-mass correction would target a different total mass than the run
    being continued.
    """

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=3)
    for _ in range(5):
        stopped.step()
    stopped.finish.asq[:] = [1.5, -2.25, 1e-17, 4.0, 0.1, 6.0]
    receipt = _written(
        stopped, tmp_path / "restart-000009", forecast_publisher,
        restart_number=9, next_history_index=42,
    )
    scalars = receipt["scheduler_scalars"]
    assert scalars[MASS1_SCALAR_NAME] == (
        f"{struct.unpack('<Q', struct.pack('<d', stopped.finish.initial_mass1_f64))[0]:016x}"
    )
    assert scalars[HISTORY_COUNTER_NAME] == "42"
    assert scalars[RESTART_COUNTER_NAME] == "9"
    # 0.1 has no exact decimal-free binary form, so a decimal round trip on
    # the way through the command line would show here.
    assert scalars[ASQ_SCALAR_FORMAT.format(index=4)] == "3fb999999999999a"

    # Give the resumed runtime a different dry-mass reference before the
    # restore, so the assertion below can only pass if the persisted one won
    # rather than because both happened to compute the same number.
    resumed = CpuCartesianRuntime(config, seed=3)
    resumed.finish.initial_mass1_f64 = stopped.finish.initial_mass1_f64 + 1.0
    resumed.dynamics.mass1_f64 = resumed.finish.initial_mass1_f64
    assert resumed.finish.initial_mass1_f64 != stopped.finish.initial_mass1_f64
    restored = read_cartesian_restart(
        tmp_path / "restart-000009", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    record = restore_cartesian_restart(resumed, restored)
    assert resumed.finish.initial_mass1_f64 == stopped.finish.initial_mass1_f64
    assert resumed.dynamics.mass1_f64 == stopped.finish.initial_mass1_f64
    assert list(resumed.finish.asq) == [1.5, -2.25, 1e-17, 4.0, 0.1, 6.0]
    assert record["next_history_index"] == 42
    assert record["restart_number"] == 9
    assert resumed.finish.completed_physics_step == 5
    assert resumed.finish.completed_solve3_step == 5


def test_the_moving_frame_integral_is_continued_not_restarted(
    forecast_publisher, tmp_path
) -> None:
    """restart_read.F:419-420, through the envelope this time.

    The Rust boundary re-derives cm1.F:3512-3513 for the frame's own clock
    before writing, so the value in the envelope is the integral or there is
    no envelope.
    """

    config = _config(u_move_m_s=12.5, v_move_m_s=3.0)
    stopped = CpuCartesianRuntime(config, seed=5)
    for _ in range(120):
        stopped.step()
    assert stopped.clock.domainlocx_m == 750.0
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)

    payload = (tmp_path / "restart-000001" / ENVELOPE_FILENAME).read_bytes()
    for name, value in (
        (DOMAIN_LOCATION_X_NAME, 750.0), (DOMAIN_LOCATION_Y_NAME, 180.0),
        (MOVE_SPEED_X_NAME, 12.5), (MOVE_SPEED_Y_NAME, 3.0),
    ):
        bits = f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"
        assert name.encode("ascii") in payload, name
        assert bits.encode("ascii") in payload, name

    resumed = CpuCartesianRuntime(config, seed=5)
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    restore_cartesian_restart(resumed, restored)
    assert resumed.clock.completed_steps == 120
    assert resumed.clock.domainlocx_m == 750.0
    assert resumed.clock.domainlocy_m == 180.0
    resumed.step()
    assert resumed.clock.domainlocx_m == 756.25


# -------------------------------------------------------------- refusals


def test_a_restart_is_refused_by_name_when_its_words_would_mean_something_else(
    forecast_publisher, tmp_path
) -> None:
    """README.restart:20-23 drawn by value.

    Run length and publication cadence may change across a restart; the grid,
    the timestep, the frame velocity and the compiled operator set may not.
    """

    config = _config(u_move_m_s=12.5, v_move_m_s=3.0)
    stopped = CpuCartesianRuntime(config, seed=2)
    stopped.step()
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )

    other_grid = CpuCartesianRuntime(_config(nx=10, u_move_m_s=12.5, v_move_m_s=3.0), seed=2)
    with pytest.raises(ValueError) as caught:
        require_resumable(restored, other_grid, cartesian_restart_profile(other_grid.config))
    assert "another carrier's name" in str(caught.value)

    other_frame = CpuCartesianRuntime(_config(u_move_m_s=10.0, v_move_m_s=3.0), seed=2)
    with pytest.raises(ValueError) as caught:
        require_resumable(restored, other_frame, cartesian_restart_profile(other_frame.config))
    assert "frame this run is not in" in str(caught.value)

    other_code = CpuCartesianRuntime(config, seed=2, execution_identity="ab" * 32)
    with pytest.raises(ValueError) as caught:
        require_resumable(restored, other_code, cartesian_restart_profile(config))
    assert "changed kernel" in str(caught.value)

    # A different source identity is a different sounding or a different
    # initializer, and this restart does not carry the base state those build.
    other_source = CpuCartesianRuntime(config, seed=2)
    other_source.source_identity = {**other_source.source_identity, "seed": 404}
    with pytest.raises(ValueError) as caught:
        require_resumable(restored, other_source, cartesian_restart_profile(config))
    assert "README.namelist:1512-1516" in str(caught.value)
    assert "['seed']" in str(caught.value)

    # The same runtime, resumable, is admitted: a refusal that fired here
    # would be a defect.
    same = CpuCartesianRuntime(config, seed=2)
    require_resumable(restored, same, cartesian_restart_profile(config))


def test_restoring_over_a_stepped_runtime_is_refused(
    forecast_publisher, tmp_path
) -> None:
    """A resume replaces the state; doing it late would discard a trajectory."""

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=4)
    stopped.step()
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    late = CpuCartesianRuntime(config, seed=4)
    late.step()
    late.step()
    with pytest.raises(RuntimeError) as caught:
        restore_cartesian_restart(late, restored)
    assert "already completed 2 large steps" in str(caught.value)


def test_a_receipt_that_is_not_this_lanes_is_refused_by_schema(tmp_path) -> None:
    with pytest.raises(ValueError) as caught:
        require_restart_receipt({"schema": "gm1.cartesian-history-publication/v1"})
    assert "under the wrong contract" in str(caught.value)


def test_a_v1_restart_directory_with_its_raw_arena_still_resumes(
    forecast_publisher, tmp_path
) -> None:
    """Restarts written before the one-copy directory keep resuming.

    A v1 receipt names ``restart-carriers.f32`` beside the envelope; a v2
    receipt says the arena was not retained.  A resume reads only the
    envelope in both, so both schemas are admitted and give the same state.
    """

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=9)
    for _ in range(2):
        stopped.step()
    directory = tmp_path / "restart-000001"
    receipt = _written(stopped, directory, forecast_publisher)
    assert receipt["schema"] == "gm1.cartesian-restart-publication/v2"
    assert receipt["arena"]["retained"] is False
    # Written straight into its final directory, not staged and renamed.
    assert receipt["directory"] == receipt["staged_directory"]
    assert receipt["arena"]["disposition"] == restart_arena_disposition(staged=False)
    assert "staging" not in receipt["arena"]["disposition"]

    # The same directory as the release head published it.
    restored = read_cartesian_restart(directory, publisher=forecast_publisher, into=tmp_path / "words")
    back = restored.arena_path.read_bytes()
    order = sorted(receipt["arena"]["carrier_offset_bytes"].items(), key=lambda item: item[1])
    sizes = {row["name"]: row["nbytes"] for row in receipt["profile"]["carriers"]}
    (directory / ARENA_FILENAME).write_bytes(b"".join(
        back[restored.offsets[name]:restored.offsets[name] + sizes[name]] for name, _ in order
    ))
    v1 = json.loads((directory / RECEIPT_FILENAME).read_text())
    v1["schema"] = "gm1.cartesian-restart-publication/v1"
    v1["arena_filename"] = ARENA_FILENAME
    v1["arena"]["path"] = os.fspath(directory / ARENA_FILENAME)
    for key in ("retained", "disposition", "envelope_verification"):
        del v1["arena"][key]
    (directory / RECEIPT_FILENAME).write_text(json.dumps(v1, indent=2, sort_keys=True) + "\n")
    assert hashlib.sha256((directory / ARENA_FILENAME).read_bytes()).hexdigest() == v1["arena"]["sha256"]

    admitted, home = require_restart_receipt(directory)
    assert admitted["schema"].endswith("/v1") and home == directory
    resumed = CpuCartesianRuntime(config, seed=9)
    restore_cartesian_restart(resumed, read_cartesian_restart(
        directory, publisher=forecast_publisher, into=tmp_path / "restored",
    ))
    assert resumed.raw_state() == stopped.raw_state()


def test_a_refused_restart_written_in_place_says_it_keeps_no_receipt(
    forecast_publisher, tmp_path, monkeypatch
) -> None:
    """The refusal describes the directory it actually leaves.

    ``publish_restart`` writes into the final directory (the storm driver's
    path), so nothing is staged, discarded or renamed: the files stay, with no
    receipt, and no resume admits them.
    """

    from cm1core import cartesian_restart as restart_module

    def refuse(envelope, arena, *, contract_sha256):
        raise RuntimeError("injected verification failure")

    monkeypatch.setattr(restart_module, "verify_restart_envelope_carries_arena", refuse)
    runtime = CpuCartesianRuntime(_config(), seed=10)
    runtime.step()
    directory = tmp_path / "restart-000001"
    with pytest.raises(RuntimeError, match="injected verification failure") as caught:
        _written(runtime, directory, forecast_publisher)
    assert any(
        "written in place" in note and "so no resume admits it" in note
        for note in caught.value.__notes__
    ), caught.value.__notes__
    assert not any("renamed" in note for note in caught.value.__notes__)
    assert (directory / ARENA_FILENAME).exists() and not (directory / RECEIPT_FILENAME).exists()
    with pytest.raises(FileNotFoundError):
        require_restart_receipt(directory)


def test_a_restart_directory_without_its_envelope_is_refused(
    forecast_publisher, tmp_path
) -> None:
    """An envelope that is gone must refuse, not silently start from scratch."""

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=6)
    stopped.step()
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    (tmp_path / "restart-000001" / ENVELOPE_FILENAME).unlink()
    with pytest.raises(ValueError) as caught:
        read_cartesian_restart(
            tmp_path / "restart-000001", publisher=forecast_publisher,
            into=tmp_path / "restored",
        )
    assert "silently continue from the initial condition" in str(caught.value)


def test_a_restart_is_resumable_after_being_moved(
    forecast_publisher, tmp_path
) -> None:
    """The receipt names the envelope beside it, not an absolute path.

    An operator who copies a restart off the run host must be able to resume
    from it; a receipt that redirected the read to a path on another machine
    would either fail or, worse, find some other run's envelope there.
    """

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=8)
    for _ in range(3):
        stopped.step()
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    moved = tmp_path / "elsewhere" / RESTART_DIRECTORY_FORMAT.format(number=1)
    moved.parent.mkdir()
    (tmp_path / "restart-000001").rename(moved)

    resumed = CpuCartesianRuntime(config, seed=8)
    restored = read_cartesian_restart(
        moved, publisher=forecast_publisher, into=tmp_path / "restored"
    )
    restore_cartesian_restart(resumed, restored)
    assert resumed.raw_state() == stopped.raw_state()


def test_every_carrier_comes_back_out_of_the_envelope_word_for_word(
    forecast_publisher, tmp_path
) -> None:
    """The arena Rust gives back is the words it was handed, carrier by carrier.

    Not byte range by byte range: ``ForecastContract::new`` sorts the array
    specs by name, so the restored arena is the same words in the contract's
    canonical order rather than in the writer's declaration order.  That is
    why the restore emits a field table beside the arena and why the Python
    side addresses carriers by name and offset instead of assuming a layout.
    """

    import hashlib

    from cm1core.cartesian_restart import _restart_carriers

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=12)
    stopped.step()
    profile = cartesian_restart_profile(config)
    # The words handed to the writer, in its declaration order: the live
    # carriers at the boundary the restart was taken at.
    written = b"".join(
        np.ascontiguousarray(value).astype("<f4", copy=False).tobytes()
        for value in _restart_carriers(stopped, profile).values()
    )
    receipt = _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    # One copy on disk: the raw arena was the publisher's input and is gone;
    # the receipt says so and still carries the arena's digests.
    assert not (tmp_path / "restart-000001" / ARENA_FILENAME).exists()
    assert receipt["arena"]["retained"] is False and receipt["arena"]["path"] is None
    assert receipt["arena"]["sha256"] == hashlib.sha256(written).hexdigest()
    assert int(receipt["arena"]["nbytes"]) == len(written)
    assert receipt["profile"]["payload_nbytes"] == len(written)
    offsets = receipt["arena"]["carrier_offset_bytes"]
    sizes = {row["name"]: row["nbytes"] for row in receipt["profile"]["carriers"]}
    assert receipt["arena"]["envelope_verification"]["arrays_verified"] == len(offsets)

    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    back = restored.arena_path.read_bytes()
    assert len(back) == len(written)
    assert sorted(restored.shapes) == sorted(offsets)
    for name, offset in offsets.items():
        nbytes = sizes[name]
        start = restored.offsets[name]
        assert back[start:start + nbytes] == written[offset:offset + nbytes], name
        assert hashlib.sha256(back[start:start + nbytes]).hexdigest() == (
            receipt["arena"]["carrier_sha256"][name]
        ), name
    # Canonical order, and therefore not the writer's order for this profile.
    assert sorted(restored.offsets, key=restored.offsets.__getitem__) == sorted(offsets)
    assert list(offsets) != sorted(offsets)


def test_a_restart_never_replaces_an_existing_one(
    forecast_publisher, tmp_path
) -> None:
    config = _config()
    stopped = CpuCartesianRuntime(config, seed=13)
    stopped.step()
    _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    with pytest.raises(ValueError) as caught:
        _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    assert "already exists" in str(caught.value)


# ---------------------------------------- from TOML to a written restart
#
# The chain these exercise is the whole point of the gap: a key in a document,
# through the strict parser, through the door's schedule admission, into the
# shipped integration loop, out as restart directories on disk at exactly the
# model times the document asked for -- and back in again.

_TOML = """\
# CM1 r21.1 base.F isnd=5 with iwnd=2, on a small grid, with an intermediate
# restart cadence: README.namelist:103-104 rstfrq, cm1.F:3186/4247-4252.
[run]
lane = "cartesian"
duration_s = {duration}
dt_s = 0.5
history_interval_s = {history}
restart_interval_s = {restart}
{selectors}
[initialization]
isnd = 5
iwnd = 2

[grid]
nx = 9
ny = 11
nz = 8
dx_m = 500.0
dy_m = 500.0
dz_m = 500.0
terrain = false
stretch_x = false
stretch_y = false
stretch_z = false

[dynamics]
psolver = 3
adaptive_timestep = false
scalar_horizontal_order = 5
scalar_vertical_order = 5
velocity_horizontal_order = 5
velocity_vertical_order = 5
scalar_weno = 2
velocity_weno = 0
weno_order = 5
dry_mass_correction = true

[boundaries]
west = 2
east = 2
south = 2
north = 2
bottom = 1
top = 1
radiative_order = 4

[closures]
sgs_model = 1
vertical_implicit_diffusion = true
upper_damping = 1
upper_damping_depth_m = 2000.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = true
u_move_m_s = 12.5
v_move_m_s = 3.0

[runtime]
acoustic_substeps = 6
tile_nx = 64
tile_ny = 64
warmup_steps = 0
memory_budget_mib = 256
capacity_profile = "default"

[physics]
component = "wsm6-mp6"
equation_set = 2
"""


def _document(tmp_path: Path, *, duration=20.0, history=5.0, restart=10.0,
              selectors="", name="run.toml") -> Path:
    path = tmp_path / name
    path.write_text(
        _TOML.format(
            duration=duration, history=history, restart=restart,
            selectors=selectors + ("\n" if selectors else ""),
        ),
        encoding="utf-8",
    )
    return path


def test_a_declared_restart_cadence_reaches_the_schedule_the_runtime_follows(
    tmp_path,
) -> None:
    """The key parses, the door places it, and the events are CM1's.

    ``run.restart_interval_s = 10.0`` at ``dt_s = 0.5`` is twenty large steps,
    so a twenty-second run writes a restart at step 20 and at step 40, and a
    history frame at every ten.  Before this lane had a checkpoint path the
    same document was refused outright.
    """

    config = load_config(_document(tmp_path))
    assert config.run.restart_interval_s == 10.0
    schedule = require_run_schedule(config, write_restarts=True)
    assert schedule.steps_per_restart == 20
    assert [(event.step, event.model_time_s, event.history, event.restart)
            for event in schedule.events] == [
        (10, 5.0, True, False),
        (20, 10.0, True, True),
        (30, 15.0, True, False),
        (40, 20.0, True, True),
    ]
    assert schedule.restart_count == 2 and schedule.history_count == 4
    # Without a restart directory the same document schedules no restart at
    # all, rather than scheduling one that would be written nowhere.
    assert require_run_schedule(config, write_restarts=False).restart_count == 0


def test_the_integration_writes_the_declared_restarts_and_resumes_from_one(
    forecast_publisher, tmp_path
) -> None:
    """Config text to changed behaviour, and back to a bit-identical run.

    One document, two ways to satisfy it: integrate it straight through, or
    integrate half of it, take the restart the document asked for, and
    continue from that restart with ``run.irst = 1`` and ``run.rstnum``.  Both
    land on the same words, and the second one publishes the remaining history
    frames under the indices the first one would have used.
    """

    document = _document(tmp_path, duration=20.0, history=5.0, restart=10.0)
    config = load_config(document)
    settings = build_runtime_config(config)

    whole = CpuCartesianRuntime(settings, seed=21)
    whole_receipt = whole.integrate(
        duration_s=config.run.duration_s,
        history_interval_s=config.run.history_interval_s,
        history_root=tmp_path / "frames-whole",
        history_publisher=forecast_publisher,
        restart_root=tmp_path / "restarts-whole",
        restart_interval_s=config.run.restart_interval_s,
        restart_publisher=forecast_publisher,
        restart_reset_frqtim=config.run.reset_frqtim,
    )
    assert whole_receipt["executed_steps"] == 40
    assert whole_receipt["restart_interval_s"] == 10.0
    assert [row["number"] for row in whole_receipt["restarts"]] == [1, 2]
    assert [row["step"] for row in whole_receipt["restarts"]] == [20, 40]
    assert [row["model_time_s"] for row in whole_receipt["restarts"]] == [10.0, 20.0]
    published = sorted(
        path.name for path in (tmp_path / "restarts-whole").glob("restart-*")
    )
    assert published == ["restart-000001", "restart-000002"]
    assert not (tmp_path / "restarts-whole" / FRAME_STAGING_DIRECTORY).exists()

    # ---- the same run, stopped at ten seconds and continued
    stopped = CpuCartesianRuntime(settings, seed=21)
    half = _document(tmp_path, duration=10.0, history=5.0, restart=10.0, name="half.toml")
    half_config = load_config(half)
    stopped.integrate(
        duration_s=half_config.run.duration_s,
        history_interval_s=half_config.run.history_interval_s,
        history_root=tmp_path / "frames-part1",
        history_publisher=forecast_publisher,
        restart_root=tmp_path / "restarts-part1",
        restart_interval_s=half_config.run.restart_interval_s,
        restart_publisher=forecast_publisher,
    )

    resumed_document = _document(
        tmp_path, duration=20.0, history=5.0, restart=10.0,
        selectors="irst = 1\nrstnum = 1", name="resume.toml",
    )
    resumed_config = load_config(resumed_document)
    assert resumed_config.run.irst == 1 and resumed_config.run.rstnum == 1
    assert resumed_config.run.restarting
    selected, receipt = resume_selection(
        resumed_config, tmp_path / "restarts-part1"
    )
    assert selected == tmp_path / "restarts-part1" / "restart-000001"
    assert receipt["step"] == 20
    schedule = require_run_schedule(
        resumed_config, restart=receipt, write_restarts=True
    )
    # cm1.F:2711-2715 with restart_reset_frqtim false: the schedule the
    # restart carried, so the resumed run publishes at the same model times an
    # uninterrupted one does, under the next frame index.
    assert [(event.step, event.history, event.restart) for event in schedule.events] == [
        (30, True, False), (40, True, True)
    ]
    assert schedule.first_history_index == 3
    assert schedule.first_restart_number == 2

    resumed = CpuCartesianRuntime(settings, seed=21)
    restored = read_cartesian_restart(
        selected, publisher=forecast_publisher, into=tmp_path / "restored"
    )
    resumed.restore_restart(restored)
    resumed_receipt = resumed.integrate(
        duration_s=resumed_config.run.duration_s,
        history_interval_s=resumed_config.run.history_interval_s,
        history_root=tmp_path / "frames-part2",
        history_publisher=forecast_publisher,
        restart_root=tmp_path / "restarts-part2",
        restart_interval_s=resumed_config.run.restart_interval_s,
        restart_publisher=forecast_publisher,
        restart_reset_frqtim=resumed_config.run.reset_frqtim,
        first_history_index=schedule.first_history_index,
        first_restart_number=schedule.first_restart_number,
    )
    assert resumed_receipt["executed_steps"] == 20
    assert resumed_receipt["steps_before_integration"] == 20
    assert resumed_receipt["model_seconds_at_end"] == 20.0
    assert [row["index"] for row in resumed_receipt["frames"]] == [3, 4]
    assert [row["model_time_s"] for row in resumed_receipt["frames"]] == [15.0, 20.0]
    assert [row["number"] for row in resumed_receipt["restarts"]] == [2]

    # The whole point: the two runs end on the same words.
    assert resumed.raw_state() == whole.raw_state()
    # And the published frames are the same frames, byte for byte, under the
    # same indices -- the arena digest comes out of the Rust boundary.
    whole_frames = {row["index"]: row["arena_sha256"] for row in whole_receipt["frames"]}
    for row in resumed_receipt["frames"]:
        assert row["arena_sha256"] == whole_frames[row["index"]], row["index"]
    for index in (3, 4):
        name = FRAME_DIRECTORY_FORMAT.format(index=index)
        assert (tmp_path / "frames-part2" / name).is_dir()


def test_a_resumed_run_raises_getdbz_on_the_steps_it_actually_publishes_at(
    forecast_publisher, tmp_path
) -> None:
    """cm1.F:3213-3244 raises ``getdbz`` in the loop iteration that writes.

    ``cartesian_history`` refuses any frame whose reflectivity carriers were
    last written at a different step, so the getdbz schedule and the frame
    schedule are one fact.  A restart cadence need not be a multiple of the
    history cadence -- README.namelist declares ``rstfrq`` and ``tapfrq``
    independently -- so a resumed run starts at a step that is not on the
    frame grid.  Deriving the getdbz steps from the clock and the whole
    simulation's frame count instead of from the schedule would raise the flag
    at 12, 22, 32 and 42 for a run that publishes at 10, 20, 30 and 40: every
    frame of a resumed ``output_dbz`` run refused, and the run's GPU-hours
    with them.
    """

    # 20 s at dt=0.5 is 40 large steps; a 5 s frame cadence is 10 and a 1 s
    # restart cadence is 2, so restart 1 lands at step 2 -- off the frame grid.
    document = _document(tmp_path, duration=20.0, history=5.0, restart=1.0)
    config = load_config(document)
    settings = build_runtime_config(config)

    stopped = CpuCartesianRuntime(settings, seed=44)
    stopped.integrate(
        duration_s=2.0, history_interval_s=1.0,
        history_root=tmp_path / "dbz-part1",
        history_publisher=forecast_publisher,
        restart_root=tmp_path / "dbz-restarts",
        restart_interval_s=1.0, restart_publisher=forecast_publisher,
    )
    resumed = CpuCartesianRuntime(settings, seed=44)
    restored = read_cartesian_restart(
        tmp_path / "dbz-restarts" / RESTART_DIRECTORY_FORMAT.format(number=1),
        publisher=forecast_publisher, into=tmp_path / "dbz-restored",
    )
    resumed.restore_restart(restored)
    assert resumed.clock.completed_steps == 2

    receipt = resumed.integrate(
        duration_s=config.run.duration_s,
        history_interval_s=config.run.history_interval_s,
        history_root=tmp_path / "dbz-part2",
        history_publisher=forecast_publisher,
    )
    published = [row["step"] for row in receipt["frames"]]
    assert published == [10, 20, 30, 40]
    # The set getdbz is raised on is exactly the set of published steps, and
    # holds no step this run never reaches.
    assert sorted(resumed.reflectivity_frame_steps) == published


def test_the_courant_stop_counts_this_runs_own_frames_not_the_simulations(
    forecast_publisher, tmp_path
) -> None:
    """"N of M frames are published" is a claim about this run.

    misclibs.F:2714's stop is the one failure whose record is kept, and the
    number in it is what an operator reads to learn how much of the run
    survived.  A resumed run was asked for the remainder, so reporting the
    whole simulation's frame count would name frames the interrupted run
    already wrote as missing.
    """

    config = load_config(_document(tmp_path, duration=20.0, history=5.0, restart=10.0))
    settings = build_runtime_config(config)
    stopped = CpuCartesianRuntime(settings, seed=45)
    stopped.integrate(
        duration_s=10.0, history_interval_s=5.0,
        history_root=tmp_path / "count-part1",
        history_publisher=forecast_publisher,
        restart_root=tmp_path / "count-restarts",
        restart_interval_s=10.0, restart_publisher=forecast_publisher,
    )
    resumed = CpuCartesianRuntime(settings, seed=45)
    restored = read_cartesian_restart(
        tmp_path / "count-restarts" / RESTART_DIRECTORY_FORMAT.format(number=1),
        publisher=forecast_publisher, into=tmp_path / "count-restored",
    )
    resumed.restore_restart(restored)
    assert resumed.clock.completed_steps == 20

    # The remaining schedule is two frames, at steps 30 and 40, out of the
    # simulation's four.  The first is published normally; then the wind is
    # driven past CM1's stop so the second event trips it.
    def blow_up(row: dict) -> None:
        resumed.dynamics.fields["u3d"][...] = np.float32(4000.0)

    with pytest.raises(CartesianCourantStop) as caught:
        resumed.integrate(
            duration_s=config.run.duration_s,
            history_interval_s=config.run.history_interval_s,
            history_root=tmp_path / "count-part2",
            history_publisher=forecast_publisher,
            first_history_index=3,
            progress=blow_up,
        )
    stop = caught.value.receipt["courant_stop"]
    assert stop["frames_published"] == 2
    # Two, not four: frames 1 and 2 are the interrupted run's and were never
    # this run's to publish.
    assert stop["frames_requested"] == 2
    assert "2 of 2 frames are published" in str(caught.value)
    # CM1 writes the tripping state's output and only then stops, so the
    # second frame is on disk.
    assert stop["frame_index"] == 4
    for index in (3, 4):
        assert (tmp_path / "count-part2" / FRAME_DIRECTORY_FORMAT.format(index=index)).is_dir()


def test_iconly_refuses_a_history_frame_whose_reflectivity_was_never_computed(
    forecast_publisher, tmp_path
) -> None:
    """An initial condition takes no step, and dBZ only exists after one.

    CM1 has no such hole: ``taptim`` starts at zero, so the ``nstep=0``
    writeout at cm1.F:3943 happens inside a loop iteration that already raised
    ``getdbz`` at cm1.F:3213-3244.  GM1 fills the reflectivity carriers only
    inside ``finish.microphysics(reflectivity_due=True)``, which iconly never
    reaches, so they still hold the zeros ``_allocate_reflectivity`` gave them.
    The history freshness check cannot see it -- the published step and
    ``completed_reflectivity_step`` are both zero -- so the frame would be
    admitted carrying 0 dBZ at every gridpoint, a uniform light echo over the
    whole domain beside the initial sounding's real winds.
    """

    config = load_config(_document(tmp_path, selectors="iconly = 1", name="icdbz.toml"))
    settings = build_runtime_config(config)
    runtime = CpuCartesianRuntime(settings, seed=46)
    runtime.finish.output_dbz = True

    with pytest.raises(RuntimeError) as caught:
        runtime.initial_condition(
            restart_root=tmp_path / "icdbz-restart",
            restart_publisher=forecast_publisher,
            history_root=tmp_path / "icdbz-frames",
            history_publisher=forecast_publisher,
        )
    message = str(caught.value)
    assert "0 dBZ at every gridpoint" in message
    assert "output_dbz" in message and "drop the history root" in message
    # Refused before anything exists, so the corrected re-run is not blocked by
    # a directory require_new_history_directory would then refuse.
    assert not (tmp_path / "icdbz-restart").exists()
    assert not (tmp_path / "icdbz-frames").exists()

    # The restart alone carries no reflectivity, so it is still admitted.
    receipt = runtime.initial_condition(
        restart_root=tmp_path / "icdbz-restart",
        restart_publisher=forecast_publisher,
    )
    assert receipt["history_publication"] is None
    assert receipt["restart_publication"]["step"] == 0


def test_restart_reset_frqtim_is_reachable_and_changes_what_is_admitted(
    tmp_path,
) -> None:
    """README.namelist:1488-1510 and cm1.F:2711-2715, selected from a document.

    CM1's namelist default is ``.true.``: the next output, statistics and
    restart go at the resume time plus the namelist interval.  This lane
    defaults to ``.false.`` because README.namelist:1508-1510 names that as
    the branch that may be needed for bit-identical restarts, and because this
    lane additionally contracts that the run's own final state is published --
    a schedule anchored on an off-grid resume time walks past it.

    So the key is reachable and it decides: on a resume that lands on the
    declared grid the two branches agree, and on one that does not, ``.false.``
    continues the original schedule while ``.true.`` refuses by name rather
    than publishing frames that stop short of the run.
    """

    keep = load_config(_document(tmp_path, duration=30.0, history=10.0, restart=6.0))
    assert keep.run.restart_reset_frqtim is None and keep.run.reset_frqtim is False
    reset = load_config(_document(
        tmp_path, duration=30.0, history=10.0, restart=6.0,
        selectors="restart_reset_frqtim = true", name="reset.toml",
    ))
    assert reset.run.restart_reset_frqtim is True and reset.run.reset_frqtim is True

    # A restart at 12 s: on the six-second restart grid, off the ten-second
    # history grid.  cm1.F:4247-4252 writes exactly there.
    off_grid = {"step": 24, "next_history_index": 2, "restart_number": 2}
    kept = require_run_schedule(keep, restart=off_grid, write_restarts=True)
    assert [(event.step, event.history, event.restart) for event in kept.events] == [
        (36, False, True), (40, True, False), (48, False, True),
        (60, True, True),
    ]
    with pytest.raises(CM1Refusal) as caught:
        require_run_schedule(reset, restart=off_grid, write_restarts=True)
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-CADENCE"
    assert "restart_reset_frqtim" in caught.value.detail.reason
    assert "cm1.F:2711-2715" in caught.value.detail.reason

    # On the declared grid the two branches are the same schedule, which is
    # why README.namelist:1508-1510 can call one of them the bit-identical
    # option without the other being wrong.
    on_grid = {"step": 20, "next_history_index": 3, "restart_number": 4}
    assert [event.step for event in require_run_schedule(keep, restart=on_grid).events] \
        == [event.step for event in require_run_schedule(reset, restart=on_grid).events] \
        == [40, 60]


def test_the_restart_selection_surface_refuses_every_silent_no_op(
    tmp_path,
) -> None:
    """CM1's irst/rstnum/iconly, refused by key and value.

    Each of these combinations is one CM1 accepts and then ignores.  Ignoring
    it here would mean a receipt that records a restart the run never opened,
    or an initial-condition request that spends three GPU-hours instead.
    """

    with pytest.raises(CM1Refusal) as caught:
        load_config(_document(tmp_path, selectors="rstnum = 3", name="a.toml"))
    assert caught.value.detail.code == "CM1-CFG-RESTART-SELECTION"
    assert caught.value.detail.option == "run.rstnum"
    assert "recorded a restart number" in caught.value.detail.reason

    with pytest.raises(CM1Refusal) as caught:
        load_config(_document(tmp_path, selectors="irst = 1", name="b.toml"))
    assert caught.value.detail.option == "run.irst"
    assert "which model time to continue from" in caught.value.detail.reason

    with pytest.raises(CM1Refusal) as caught:
        load_config(_document(
            tmp_path, selectors="irst = 1\nrstnum = 2\niconly = 1", name="c.toml",
        ))
    assert caught.value.detail.option == "run.iconly"
    assert "cm1.F:4342" in caught.value.detail.reason

    with pytest.raises(CM1Refusal) as caught:
        load_config(_document(tmp_path, selectors="irst = 2\nrstnum = 1", name="d.toml"))
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert caught.value.detail.option == "run.irst"

    with pytest.raises(CM1Refusal) as caught:
        load_config(_document(tmp_path, selectors="irst = 1\nrstnum = 0", name="e.toml"))
    assert caught.value.detail.code == "CM1-CFG-INTEGER"
    assert caught.value.detail.option == "run.rstnum"

    # The combinations CM1 honours are admitted; a refusal here would be a
    # defect, not a guard.
    assert load_config(
        _document(tmp_path, selectors="irst = 1\nrstnum = 4", name="f.toml")
    ).run.rstnum == 4
    assert load_config(
        _document(tmp_path, selectors="iconly = 1", name="g.toml")
    ).run.initial_condition_only


def test_an_unreachable_restart_refuses_before_a_device_is_opened(
    forecast_publisher, tmp_path
) -> None:
    """A resume that would silently become a fresh start must stop instead."""

    resumable = load_config(
        _document(tmp_path, selectors="irst = 1\nrstnum = 2", name="r.toml")
    )
    empty = tmp_path / "restarts"
    empty.mkdir()
    with pytest.raises(CM1Refusal) as caught:
        resume_selection(resumable, empty)
    assert caught.value.detail.code == "CM1-CARTESIAN-RESTART-SELECTION"
    assert caught.value.detail.option == "run.rstnum"
    assert "no restarts at all" in caught.value.detail.reason

    with pytest.raises(CM1Refusal) as caught:
        resume_selection(resumable, tmp_path / "absent")
    assert "does not exist" in caught.value.detail.reason

    plain = load_config(_document(tmp_path, name="p.toml"))
    assert resume_selection(plain, None) is None
    with pytest.raises(CM1Refusal) as caught:
        resume_selection(plain, empty)
    assert caught.value.detail.option == "restart-from"

    with pytest.raises(CM1Refusal) as caught:
        resume_selection(resumable, None)
    assert "silently become a fresh start" in caught.value.detail.reason

    # A directory that holds a restart directory but no receipt in it.
    hollow = tmp_path / "hollow"
    (hollow / RESTART_DIRECTORY_FORMAT.format(number=2)).mkdir(parents=True)
    with pytest.raises(CM1Refusal) as caught:
        resume_selection(resumable, hollow)
    assert caught.value.detail.code == "CM1-CARTESIAN-RESTART-RECEIPT"


def test_iconly_writes_the_initial_condition_and_integrates_nothing(
    forecast_publisher, tmp_path
) -> None:
    """cm1.F:4341-4350 and README.namelist:233-235.

    A run here costs GPU-hours, so the check that a sounding, a bubble and a
    grid are what the user intended has to be available without paying for
    them.  What it writes is a complete restart at ``nstep=0``, which is both
    inspectable and resumable.
    """

    config = load_config(_document(tmp_path, selectors="iconly = 1", name="ic.toml"))
    assert config.run.initial_condition_only
    settings = build_runtime_config(config)
    runtime = CpuCartesianRuntime(settings, seed=31)
    initial = runtime.raw_state()

    receipt = runtime.initial_condition(
        restart_root=tmp_path / "ic-restart",
        restart_publisher=forecast_publisher,
        history_root=tmp_path / "ic-frames",
        history_publisher=forecast_publisher,
    )
    assert receipt["schema"] == INITIAL_CONDITION_SCHEMA
    assert receipt["integrated"] is False
    assert receipt["completed_steps"] == 0
    assert runtime.clock.completed_steps == 0
    assert runtime.raw_state() == initial

    restart = receipt["restart_publication"]
    assert restart["step"] == 0 and restart["restart_number"] == 1
    assert Path(restart["forecast_envelope"]["path"]).read_bytes().startswith(
        FORECAST_CHECKPOINT_MAGIC.encode("ascii")
    )
    frame = receipt["history_publication"]
    assert frame["initial_condition"] is True and frame["step"] == 0
    assert "iconly" in frame["quiescent_boundary"]
    assert (tmp_path / "ic-frames" / FRAME_DIRECTORY_FORMAT.format(index=0)).is_dir()

    # The initial condition is a restart like any other: a later run continues
    # from it without paying for the initializer again.
    integrated = CpuCartesianRuntime(settings, seed=31)
    restored = read_cartesian_restart(
        tmp_path / "ic-restart" / RESTART_DIRECTORY_FORMAT.format(number=1),
        publisher=forecast_publisher, into=tmp_path / "ic-restored",
    )
    integrated.restore_restart(restored)
    assert integrated.raw_state() == initial
    for _ in range(4):
        integrated.step()
    fresh = CpuCartesianRuntime(settings, seed=31)
    for _ in range(4):
        fresh.step()
    assert integrated.raw_state() == fresh.raw_state()


def test_an_integration_without_a_publisher_or_a_root_refuses_before_stepping(
    tmp_path,
) -> None:
    """A restart cadence with nowhere to write it costs GPU hours for nothing."""

    settings = build_runtime_config(load_config(_document(tmp_path)))
    runtime = CpuCartesianRuntime(settings, seed=41)
    for kwargs, expected in (
        ({"restart_interval_s": 10.0}, "needs a restart directory"),
        ({"restart_root": tmp_path / "nowhere"}, "needs a cadence"),
        (
            {"restart_interval_s": 10.0, "restart_root": tmp_path / "x"},
            "admitted forecast publisher",
        ),
    ):
        with pytest.raises(ValueError) as caught:
            runtime.integrate(
                duration_s=20.0, history_interval_s=5.0,
                history_root=tmp_path / "frames", **kwargs,
            )
        assert expected in str(caught.value)
    assert runtime.clock.completed_steps == 0
    assert not (tmp_path / "frames").exists()


# ------------------------------------------- the default stays the default


def test_every_shipped_document_keeps_the_identity_it_had(tmp_path) -> None:
    """The new selectors are optional, so nothing sealed moved.

    A document that does not declare irst, rstnum, iconly or
    restart_reset_frqtim hashes exactly as it did before those keys existed,
    because canonical_dict drops an undeclared optional key rather than
    projecting a null.  These are the identities recorded on this branch's
    parent commit.
    """

    root = Path(__file__).resolve().parents[1]
    expected = {
        "cartesian-supercell-attempt.toml":
            "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b",
        "cartesian-supercell-translating-frame.toml":
            "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca",
        "cartesian-uniform-grid.toml":
            "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48",
        "cartesian-wk-analytic-sounding-shear.toml":
            "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7",
        "dry-psolver2-bootstrap.toml":
            "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340",
        "official-supercell-native-context.toml":
            "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c",
        "official-supercell-product.toml":
            "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038",
        "real-atmosphere-100m-product.toml":
            "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00",
    }
    seen = {
        path.name: load_config(path).sha256
        for path in sorted((root / "examples").glob("*.toml"))
    }
    assert {name: seen[name] for name in expected} == expected
    # And a document that declares one is a different document, by design.
    plain = load_config(_document(tmp_path, name="plain.toml"))
    selected = load_config(_document(tmp_path, selectors="iconly = 1", name="sel.toml"))
    assert plain.sha256 != selected.sha256
    assert "iconly" not in plain.canonical_dict()["run"]
    assert selected.canonical_dict()["run"]["iconly"] == 1


def test_the_shipped_restart_documents_select_what_they_say_they_do() -> None:
    """The examples are the user-facing surface; they must reach the door.

    Each is loaded, admitted and taken through the same door helpers a real
    invocation goes through, so a document that reads well and selects nothing
    cannot ship.
    """

    root = Path(__file__).resolve().parents[1] / "examples"
    checkpointed = load_config(root / "cartesian-supercell-checkpointed.toml")
    assert checkpointed.run.restart_interval_s == 900.0
    assert not checkpointed.run.restarting
    assert not checkpointed.run.initial_condition_only
    schedule = require_run_schedule(checkpointed, write_restarts=True)
    # 7200 s at dt=0.5 is 14,400 large steps; 900 s is 1,800 of them.
    assert schedule.cadence.total_steps == 14_400
    assert schedule.steps_per_restart == 1_800
    assert schedule.restart_count == 8 and schedule.history_count == 120
    assert schedule.events[-1].step == 14_400
    assert schedule.events[-1].restart and schedule.events[-1].history

    resumed = load_config(root / "cartesian-supercell-resume.toml")
    assert resumed.run.irst == 1 and resumed.run.rstnum == 8
    assert resumed.run.restarting
    continued = require_run_schedule(
        resumed,
        restart={"step": 14_400 - 1_800, "next_history_index": 105, "restart_number": 8},
        write_restarts=True,
    )
    assert continued.first_restart_number == 9
    assert continued.first_history_index == 105
    assert continued.history_count == 15 and continued.restart_count == 1
    assert continued.events[-1].step == 14_400

    initial = load_config(root / "cartesian-initial-condition-only.toml")
    assert initial.run.iconly == 1 and initial.run.initial_condition_only
    assert not initial.run.restarting


# ------------------------------------------- the installed door, end to end
#
# The door is where a user's TOML meets the runtime.  Everything below runs
# the real `run_cartesian` with the real argument plumbing, the real Rust
# publisher and the real restart module; the two things replaced are the CUDA
# runtime factory, which cannot run without a device, and the ArWen estate
# gate, which has its own tests and is not what these are about.


class _FakeCuda:
    """Just enough of the CuPy surface for the door's own bookkeeping."""

    __version__ = "cpu-double"

    class cuda:
        class Stream:
            def __init__(self, non_blocking: bool = False) -> None:
                self.non_blocking = non_blocking

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def synchronize(self) -> None:
                pass

        class runtime:
            @staticmethod
            def getDeviceProperties(index):
                return {"name": b"cpu-double"}

            @staticmethod
            def driverGetVersion():
                return 0

            @staticmethod
            def runtimeGetVersion():
                return 0


@pytest.fixture
def door(monkeypatch, tmp_path, forecast_publisher):
    """Drive cartesian_door against the host double, with the real Rust."""

    from cm1core import cartesian_door as door_module
    from cm1core import cuda_cartesian_runtime as runtime_module
    from cm1core import gm1_idealized_residency as residency

    built: list[CpuCartesianRuntime] = []

    class _Source:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _factory(source, settings, *, cupy, stream, capacity_profile=None):
        runtime = CpuCartesianRuntime(settings, seed=77)
        built.append(runtime)
        return runtime

    # The ArWen estate gate is not installed in every checkout and is not what
    # this exercises; tests/test_storm_lane.py owns it.
    monkeypatch.setattr(door_module, "require_runnable", lambda config, **kwargs: None)
    monkeypatch.setattr(door_module, "require_cuda_runtime", lambda: _FakeCuda)
    monkeypatch.setattr(runtime_module, "CudaCartesianRuntime", _factory)
    monkeypatch.setattr(residency, "IdealizedSource", _Source)
    # base.F:3143-3189's t=0 report is computed by the real Rust initializer,
    # and every executed Cartesian run now states it.  This fixture replaces
    # IdealizedSource and hands the door a stub .so, so the report is taken
    # from the built library when there is one and stood down when there is
    # not; either way the restart door is what this file exercises, and
    # tests/test_getcape_environment.py owns the report itself.
    built_library = (
        Path(__file__).resolve().parents[1]
        / "rust" / "cm1-state" / "target" / "release" / "libcm1_state.so"
    )
    if built_library.is_file():
        real_report = door_module.require_sounding_report
        monkeypatch.setattr(
            door_module, "require_sounding_report",
            lambda config, *, library, sounding_bytes, selection: real_report(
                config, library=built_library, sounding_bytes=sounding_bytes,
                selection=selection,
            ),
        )
    else:  # pragma: no cover - a source checkout that has not built Rust yet
        # The door prints the report block too; the stood-down report has no
        # parcel rows for format_sounding_report to read.
        monkeypatch.setattr(
            door_module, "format_sounding_report",
            lambda report: "  (t=0 sounding report not computed: Rust not built)",
        )
        monkeypatch.setattr(
            door_module, "require_sounding_report",
            lambda config, *, library, sounding_bytes, selection: {
                "schema": "gpuwm-cm1.sounding-report/v1",
                "not_computed": "the Rust initializer library is not built here",
            },
        )

    library = tmp_path / "libcm1_state.so"
    library.write_bytes(b"not a real library; IdealizedSource is replaced\n")
    return SimpleNamespace(built=built, library=library, publisher=forecast_publisher)


def _run(document: Path, *, door, output: Path, **kwargs) -> dict:
    from cm1core.cartesian_door import run_cartesian

    assert run_cartesian(
        load_config(document), config_label=str(document), output=output,
        library=door.library, history_publisher_argument=door.publisher, **kwargs,
    ) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_the_door_writes_the_declared_restarts_and_resumes_from_one(
    door, tmp_path
) -> None:
    """A TOML key, through the installed door, to restarts on disk and back.

    This is the whole path a user takes: ``run.restart_interval_s`` writes
    them, ``run.irst``/``run.rstnum`` with ``--restart-from`` reads one, and
    the resumed run reaches the same model time as one that never stopped.
    """

    first = _document(tmp_path, duration=20.0, history=5.0, restart=10.0, name="a.toml")
    receipt = _run(
        first, door=door, output=tmp_path / "first.json", integrate=True,
        history_dir=tmp_path / "frames", restart_dir=tmp_path / "restarts",
    )
    assert receipt["schema"] == "gpuwm-cm1.cartesian-integration/v1"
    assert receipt["status"] == "executed-cartesian-integration-to-requested-duration"
    assert receipt["restart"]["restart_interval_s"] == 10.0
    assert receipt["restart"]["irst"] is None
    assert receipt["restart_directory"] == str(tmp_path / "restarts")
    assert "restart-NNNNNN" in receipt["restart_layout"]
    assert receipt["run_schedule"]["restart_count"] == 2
    assert sorted(
        path.name for path in (tmp_path / "restarts").glob("restart-*")
    ) == ["restart-000001", "restart-000002"]
    whole = door.built[-1].raw_state()

    # ---- the same document, run only to ten seconds, then continued
    half = _document(tmp_path, duration=10.0, history=5.0, restart=10.0, name="b.toml")
    _run(
        half, door=door, output=tmp_path / "half.json", integrate=True,
        history_dir=tmp_path / "frames-1", restart_dir=tmp_path / "restarts-1",
    )
    resume_document = _document(
        tmp_path, duration=20.0, history=5.0, restart=10.0,
        selectors="irst = 1\nrstnum = 1", name="c.toml",
    )
    resumed = _run(
        resume_document, door=door, output=tmp_path / "resumed.json", integrate=True,
        history_dir=tmp_path / "frames-2", restart_dir=tmp_path / "restarts-2",
        restart_from=tmp_path / "restarts-1",
    )
    assert resumed["status"] == "resumed-and-executed-cartesian-integration-to-requested-duration"
    assert resumed["restart"]["irst"] == 1 and resumed["restart"]["rstnum"] == 1
    assert resumed["restart"]["resumed_step"] == 20
    assert resumed["restart"]["resumed_model_time_s"] == 10.0
    assert resumed["restart"]["resumed_from"].endswith("restart-000001")
    assert resumed["runtime_benchmark"]["resume"]["step"] == 20
    assert resumed["runtime_benchmark"]["executed_steps"] == 20
    assert resumed["runtime_benchmark"]["model_seconds_at_end"] == 20.0
    assert [row["index"] for row in resumed["runtime_benchmark"]["frames"]] == [3, 4]
    assert door.built[-1].raw_state() == whole


def test_the_iconly_frame_carries_cm1s_nstep0_param9_products(
    forecast_publisher, tmp_path
) -> None:
    """cm1.F:3085-3088: a startup writes before any solve, then iconly stops.

    So the frame's accumulators and run-long extrema are exactly their
    initialized values (init3d.F:183-199, cm1.F:873-874), sourced from the
    live engine rather than refused or invented.  The seven swaths are
    selected too (a [diagnostics] choice): nothing is stepped, so the fold
    kernel never runs and the sentinels are what CM1 writes.
    """

    from dataclasses import replace

    from cm1core.cm1_output_selection import PER_STEP_PRODUCTS
    from cm1core.cuda_cm1_output_diagnostics import CM1_SWATH_INITIAL

    config = load_config(_document(tmp_path, selectors="iconly = 1", name="ic-defaults.toml"))
    settings = build_runtime_config(config)
    assert settings.output.on("output_rain")
    settings = replace(settings, output=settings.output.with_values(
        {key: 1 for key in PER_STEP_PRODUCTS}
    ))
    runtime = CpuCartesianRuntime(settings, seed=5)
    receipt = runtime.initial_condition(
        restart_root=tmp_path / "r", restart_publisher=forecast_publisher,
        history_root=tmp_path / "f", history_publisher=forecast_publisher,
    )
    assert receipt["history_publication"]["step"] == 0
    published = runtime.output_fields
    for name, value in CM1_SWATH_INITIAL.items():
        assert np.all(published[name] == np.float32(value)), name
    persistence = runtime.finish.persistence
    assert np.array_equal(
        published["rain"],
        (persistence["mp_rainnc"] * np.float32(0.1)).astype(np.float32),
    )
    assert np.array_equal(published["th"], runtime.dynamics.fields["th0"] + runtime.dynamics.fields["th3d"])
    table = (tmp_path / "f" / FRAME_DIRECTORY_FORMAT.format(index=0)
             / "history-field-table.tsv").read_text()
    for name in ("rain", "prate", *CM1_SWATH_INITIAL, "th", "zhval", "prs0"):
        assert f"\n{name}\t" in "\n" + table, name


def _host_initial_source(profile, rng):
    """IdealizedSource's field() surface, holding one word array per request."""

    from cm1core.cartesian_initial_condition import _CARRIER_SOURCES

    g_shape = {carrier.name: carrier.shape for carrier in profile.carriers}
    volumes = {
        qualified: rng.standard_normal(g_shape[name]).astype("<f4")
        for name, (qualified, _, _) in _CARRIER_SOURCES.items()
        if qualified is not None and name in g_shape
    }
    nz = g_shape["th3d"][0]
    profiles = {
        "terrain_open.th0": (np.float32(300.0) + np.arange(nz, dtype="<f4")),
        "terrain_open.pi0": (np.float32(1.0) - np.float32(0.015625) * np.arange(nz, dtype="<f4")),
        "terrain_open.qv0": np.full(nz, np.float32(0.01), dtype="<f4"),
        "base_wind.u0": np.arange(nz, dtype="<f4"),
        "base_wind.v0": -np.arange(nz, dtype="<f4"),
        "terrain_static.zh": (np.arange(nz, dtype="<f4") + np.float32(0.5)) * np.float32(500.0),
    }

    class _Source:
        def field(self, qualified, *, layout):
            return profiles[qualified] if layout == "profile" else volumes[qualified]

    return _Source(), volumes, profiles


def test_the_host_initial_condition_door_sources_every_default_product(tmp_path) -> None:
    """The host iconly door (no --restart-dir) publishes CM1's nstep=0 values.

    It refused the whole frame with "no source for history carrier 'rain'"
    for any document at CM1's default &param9 selection.
    """

    from cm1core.cartesian_history import cartesian_history_profile
    from cm1core.cartesian_initial_condition import initial_condition_carriers
    from cm1core.cartesian_output_products import CM1_CP, CM1_P00, CM1_RD
    from cm1core.cuda_cm1_output_diagnostics import CM1_SWATH_INITIAL

    from dataclasses import replace

    from cm1core.cm1_output_selection import PER_STEP_PRODUCTS

    settings = build_runtime_config(load_config(_document(
        tmp_path, selectors="iconly = 1", name="host.toml",
    )))
    # [diagnostics] swaths on as well, so every nstep=0 source is exercised.
    settings = replace(settings, output=settings.output.with_values(
        {key: 1 for key in PER_STEP_PRODUCTS}
    ))
    profile = cartesian_history_profile(settings)
    source, volumes, profiles = _host_initial_source(profile, np.random.default_rng(3))
    carriers, provenance = initial_condition_carriers(source, profile)
    assert list(carriers) == [carrier.name for carrier in profile.carriers]
    for name in ("rain", "prate", "snownc", "graupelnc", "sr"):
        assert carriers[name].dtype == np.dtype("<f4")
        assert np.all(carriers[name] == 0) and not np.any(np.signbit(carriers[name])), name
    assert "init3d.F:187" in provenance["rain"] and "cm1.F:3085-3088" in provenance["rain"]
    for name, value in CM1_SWATH_INITIAL.items():
        assert np.all(carriers[name] == np.float32(value)), name
        assert "init3d.F:183-199" in provenance[name]
    assert np.array_equal(
        carriers["th"], profiles["terrain_open.th0"][:, None, None] + volumes["terrain_open.th3d"]
    )
    assert np.array_equal(carriers["zhval"], profiles["terrain_static.zh"])
    assert np.array_equal(
        carriers["prs0"],
        (np.float32(CM1_P00) * np.power(profiles["terrain_open.pi0"], np.float32(CM1_CP / CM1_RD))).astype("<f4"),
    )
    for name, qualified in (("u0", "base_wind.u0"), ("qv0", "terrain_open.qv0")):
        assert np.array_equal(carriers[name], profiles[qualified])


def test_the_host_initial_condition_door_refuses_a_device_only_product(tmp_path) -> None:
    """calcvort is a CUDA kernel; the host door names the device door instead."""

    from dataclasses import replace

    from cm1core.cartesian_history import cartesian_history_profile
    from cm1core.cartesian_initial_condition import initial_condition_carriers

    settings = build_runtime_config(load_config(_document(
        tmp_path, selectors="iconly = 1", name="vort.toml",
    )))
    settings = replace(settings, output=settings.output.with_values({"output_vort": 1}))
    profile = cartesian_history_profile(settings)
    source, _, _ = _host_initial_source(profile, np.random.default_rng(4))
    with pytest.raises(ValueError, match="'xvort'.*--restart-dir"):
        initial_condition_carriers(source, profile)


def test_the_door_builds_the_initial_condition_and_stops(door, tmp_path) -> None:
    """run.iconly = 1, all the way to an envelope and no integration."""

    document = _document(tmp_path, selectors="iconly = 1", name="ic.toml")
    receipt = _run(
        document, door=door, output=tmp_path / "ic.json",
        restart_dir=tmp_path / "initial", history_dir=tmp_path / "initial-frame",
    )
    assert receipt["schema"] == "gpuwm-cm1.cartesian-initial-condition/v1"
    assert receipt["status"] == "published-cartesian-initial-condition-without-integrating"
    assert receipt["initial_condition_only"] is True
    assert receipt["runtime_benchmark"]["integrated"] is False
    assert door.built[-1].clock.completed_steps == 0
    restart = tmp_path / "initial" / RESTART_DIRECTORY_FORMAT.format(number=1)
    assert (restart / ENVELOPE_FILENAME).read_bytes().startswith(
        FORECAST_CHECKPOINT_MAGIC.encode("ascii")
    )
    assert json.loads((restart / RECEIPT_FILENAME).read_text())["step"] == 0
    assert (tmp_path / "initial-frame" / FRAME_DIRECTORY_FORMAT.format(index=0)).is_dir()


def test_the_door_refuses_every_restart_request_it_could_not_honour(
    door, tmp_path
) -> None:
    """Each of these would otherwise open a device and produce nothing."""

    plain = _document(tmp_path, name="p.toml")
    # A restart directory with the timing harness: created and left empty.
    with pytest.raises(CM1Refusal) as caught:
        _run(plain, door=door, output=tmp_path / "1.json",
             restart_dir=tmp_path / "r1")
    assert caught.value.detail.code == "CM1-CARTESIAN-RESTART-DIRECTORY"
    assert not (tmp_path / "r1").exists()
    assert not (tmp_path / "1.json").exists()

    # iconly with nowhere to put the initial condition.
    with pytest.raises(CM1Refusal) as caught:
        _run(_document(tmp_path, selectors="iconly = 1", name="ic2.toml"),
             door=door, output=tmp_path / "2.json")
    assert caught.value.detail.code == "CM1-CARTESIAN-INITIAL-CONDITION"
    assert caught.value.detail.option == "restart-dir"

    # iconly and --integrate ask for opposite things.
    with pytest.raises(CM1Refusal) as caught:
        _run(_document(tmp_path, selectors="iconly = 1", name="ic3.toml"),
             door=door, output=tmp_path / "3.json", integrate=True,
             history_dir=tmp_path / "f3", restart_dir=tmp_path / "r3")
    assert "spend the GPU hours the caller asked to avoid" in caught.value.detail.reason

    # A restart that would be measured as a benchmark and thrown away.
    with pytest.raises(CM1Refusal) as caught:
        _run(_document(tmp_path, selectors="irst = 1\nrstnum = 1", name="r.toml"),
             door=door, output=tmp_path / "4.json",
             restart_from=tmp_path / "restarts-absent")
    assert caught.value.detail.code == "CM1-CARTESIAN-RESTART-SELECTION"

    assert door.built == []


def test_one_restart_can_be_resumed_from_more_than_once(door, tmp_path) -> None:
    """An operator retries; an archive of restarts is read-only.

    The envelope is decoded into a directory beside the run's own receipt, not
    inside the restart being read, so neither of those is a failure.  Two
    resumes from the same restart also have to reach the same state, which is
    the same bit-exactness claim from a different direction.
    """

    half = _document(tmp_path, duration=10.0, history=5.0, restart=10.0, name="h.toml")
    _run(half, door=door, output=tmp_path / "half.json", integrate=True,
         history_dir=tmp_path / "frames-0", restart_dir=tmp_path / "restarts")
    resume_document = _document(
        tmp_path, duration=20.0, history=5.0, restart=10.0,
        selectors="irst = 1\nrstnum = 1", name="r.toml",
    )
    states = []
    for attempt in (1, 2):
        _run(
            resume_document, door=door, output=tmp_path / f"resume-{attempt}.json",
            integrate=True, history_dir=tmp_path / f"frames-{attempt}",
            restart_dir=tmp_path / f"restarts-{attempt}",
            restart_from=tmp_path / "restarts",
        )
        states.append(door.built[-1].raw_state())
    assert states[0] == states[1]
    # The restart that was read is untouched by either resume, and it holds
    # one copy of the state: the envelope, not a raw arena beside it.
    assert sorted(
        path.name for path in (tmp_path / "restarts" / "restart-000001").iterdir()
    ) == [
        "restart-field-table.tsv",
        "restart-publication.json", "restart-scheduler-scalars.tsv", "restart.cm1fcp",
    ]


def test_a_restart_directory_holding_another_run_is_refused_by_name(
    door, tmp_path
) -> None:
    """Two runs' restarts in one directory would misname what rstnum selects."""

    from cm1core.cartesian_door import run_cartesian

    document = _document(tmp_path, duration=10.0, history=5.0, restart=10.0)
    _run(document, door=door, output=tmp_path / "a.json", integrate=True,
         history_dir=tmp_path / "frames-a", restart_dir=tmp_path / "restarts")
    with pytest.raises(CM1Refusal) as caught:
        _run(document, door=door, output=tmp_path / "b.json", integrate=True,
             history_dir=tmp_path / "frames-b", restart_dir=tmp_path / "restarts")
    assert caught.value.detail.code == "CM1-CARTESIAN-RESTART-DIRECTORY"
    assert caught.value.detail.option == "restart-dir"
    assert "wrong trajectory" in caught.value.detail.reason


def test_the_closure_cadence_token_travels_with_the_state(
    forecast_publisher, tmp_path
) -> None:
    """The one piece of carried state that is neither a carrier nor a clock.

    ``CartesianClosure.closure`` refuses any step that is not ``_held_step + 1``
    (``cuda_cartesian_closure.py``).  A restart that carried every float32 word
    and left that counter at zero would upload the whole arena onto an opened
    device and then refuse its own next large step.  It is written and restored
    beside the physics and solve3 tokens.
    """

    config = _config()
    stopped = CpuCartesianRuntime(config, seed=17)
    for _ in range(6):
        stopped.step()
    assert stopped.closure._held_step == 6
    receipt = _written(stopped, tmp_path / "restart-000001", forecast_publisher)
    assert receipt["scheduler_scalars"][CLOSURE_STEP_NAME] == "6"

    resumed = CpuCartesianRuntime(config, seed=17)
    assert resumed.closure._held_step == 0
    restored = read_cartesian_restart(
        tmp_path / "restart-000001", publisher=forecast_publisher,
        into=tmp_path / "restored",
    )
    restore_cartesian_restart(resumed, restored)
    assert resumed.closure._held_step == 6
    assert resumed.closure._pending_step is None and resumed.closure._tke_stage == 3
    resumed.step()
    assert resumed.clock.completed_steps == 7


def test_a_mid_step_closure_state_refuses_to_be_written(
    forecast_publisher, tmp_path
) -> None:
    """cm1.F:3186 writes between completed large steps, never inside one."""

    config = _config()
    runtime = CpuCartesianRuntime(config, seed=19)
    runtime.step()
    runtime.closure._pending_step = 2
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "mid-step" in str(caught.value)

    runtime.closure._pending_step = None
    runtime.closure._tke_stage = 1
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "TKE stage" in str(caught.value)

    runtime.closure._tke_stage = 3
    runtime.closure._held_step = 0
    with pytest.raises(RuntimeError) as caught:
        require_cartesian_restart_boundary(runtime)
    assert "held step" in str(caught.value)


def _cpu_pdcomp_fields(self):
    """The CPU pidcomp over this double's own carriers, on a flat base state."""

    from cm1core.cuda_pdcomp_open import (
        build_pdcomp_coefficients, cm1_thv0_profile, cpu_pdcomp, cpu_uniform_advect,
        pdcomp_record,
    )
    from cm1core.pressure_decomposition import pdcomp_selection_for

    g = self.config.grid
    nx, ny, nz = g.nx, g.ny, g.nz
    selection = pdcomp_selection_for(self.config)
    base = self.dynamics.base_profiles
    rho0c = (np.float32(1.125) - np.float32(0.0625) * np.arange(nz, dtype=np.float32)).astype(np.float32)
    rf0c = np.concatenate([rho0c[:1], rho0c]).astype(np.float32)

    def full(column, levels):
        return np.ascontiguousarray(np.broadcast_to(column[:, None, None], (levels, ny, nx)))

    half = np.full((nz + 1, ny, nx), 0.5, dtype=np.float32)
    coefficients = build_pdcomp_coefficients(
        selection, mh=np.ones(nz, np.float32), mf=np.ones(nz + 1, np.float32),
        rho0=rho0c, rf0=rf0c, thv0=cm1_thv0_profile(base["th0"], base["qv0"]),
    )
    metrics = {
        "rho0_3d": full(rho0c, nz), "c2": half, "gz": np.ones((ny, nx), np.float32),
        "gzu": np.ones((ny, nx + 1), np.float32), "gzv": np.ones((ny + 1, nx), np.float32),
        "rrf0": full((np.float32(1.0) / rf0c).astype(np.float32), nz + 1),
    }
    a, f = self.dynamics.arrays, self.dynamics.fields
    out = cpu_pdcomp(
        selection, coefficients,
        u3d=a["u3d"], v3d=a["v3d"], w3d=a["w3d"], th3d=f["th3d"], pp3d=f["pp3d"], q3d=f["q3d"],
        th0=base["th0"], qv0=base["qv0"], u0=base["u0"], v0=base["v0"],
        rho0c=rho0c, rr0c=(np.float32(1.0) / rho0c).astype(np.float32), rf0c=rf0c,
        mhc=np.ones(nz, np.float32), mfc=np.ones(nz + 1, np.float32),
        rf0=full(rf0c, nz + 1), mh=np.ones((nz, ny, nx), np.float32),
        mf=np.ones((nz + 1, ny, nx), np.float32), c1=half, c2=half,
        uh=a["uh"], vh=a["vh"], f2d=None,
        advect=cpu_uniform_advect(metrics, selection),
    )
    return out, pdcomp_record(selection, coefficients)


def test_output_pdcomp_changes_no_simulated_word(forecast_publisher, tmp_path) -> None:
    """pdcomp is a diagnostic: the same run with it on and off ends on the same words.

    The shipped integrate/_publish_pdcomp/publish_pdcomp_frames run at every
    history time and read the double's live carriers; every owner the step
    evolves is then compared byte for byte against the run without it.
    """

    import types

    from cm1core.lagrangian_config import parse_lagrangian_table
    from cm1core.lagrangian_parcels import parcel_switch_context
    from cm1core.lagrangian_stage import LagrangianStage

    config = load_config(_document(tmp_path, duration=10.0, history=5.0, restart=10.0))
    settings = build_runtime_config(config)
    finals = {}
    for enabled in (False, True):
        runtime = CpuCartesianRuntime(settings, seed=33)
        if enabled:
            g = settings.grid
            runtime.lagrangian = LagrangianStage.build(
                parse_lagrangian_table({"output_pdcomp": 1}), nx=g.nx, ny=g.ny, nz=g.nz,
                dx_m=g.dx_m, dy_m=g.dy_m, dz_m=g.dz_m, dt_s=settings.dt_s, xp=np,
                # 5b31da7 made iorigin mandatory; the runtime passes the config's.
                iorigin=settings.iorigin,
                bbc=settings.bbc, switch_context=parcel_switch_context(settings),
            )
            runtime.lagrangian_frames = []
            runtime.parcel_frames = []
            runtime._pdcomp_fields = types.MethodType(_cpu_pdcomp_fields, runtime)
        runtime.integrate(
            duration_s=config.run.duration_s,
            history_interval_s=config.run.history_interval_s,
            history_root=tmp_path / f"frames-{enabled}",
            history_publisher=forecast_publisher,
            restart_root=tmp_path / f"restarts-{enabled}",
            restart_interval_s=config.run.restart_interval_s,
            restart_publisher=forecast_publisher,
            restart_reset_frqtim=config.run.reset_frqtim,
        )
        finals[enabled] = {name: value.tobytes() for name, value in runtime.owners().items()}
        if enabled:
            kinds = [frame["kind"] for frame in runtime.lagrangian_frames]
            assert kinds.count("pdcomp") == kinds.count("pdcomp-w") == 2
    assert finals[True].keys() == finals[False].keys()
    changed = [name for name in finals[True] if finals[True][name] != finals[False][name]]
    assert changed == []
    written = sorted(p.name for p in (tmp_path / "frames-True").rglob("pdcomp-*") if p.is_dir())
    assert len([n for n in written if n.startswith("pdcomp-w-")]) == 2 and len(written) == 4
    assert not list((tmp_path / "frames-False").rglob("pdcomp-*"))


# ----------------------------------------- publication bytes and profiling

#: SHA-256 of every file a two-frame, one-restart integration of this double
#: publishes, produced by the release head before profiling and the I/O
#: changes existed (5e83bad: its Python writer and its cm1-forecast-publish).
#: Two normalizations make the digests independent of where pytest puts the
#: run: the run root is replaced by ``<ROOT>``, and the two receipt keys that
#: hash a table containing that absolute path are replaced by a placeholder
#: (the tables themselves are digested, and the keys are rechecked below).
PRE_PROFILING_PUBLICATION_SHA256 = {
    "frames/frame-000001/history-carriers.f32": "5d76ce6c9326d20e4c4042d7e9fe3a5ca77b9d3b0172e13ca419f13114cc14ef",
    "frames/frame-000001/history-field-table.tsv": "ce38bcb92d9e9a1e565483a926fbc2b4f869c518b423cfc13ce4b4a3a4007419",
    "frames/frame-000001/history-frame-location.json": "2fd348adb7fc7fb960b727b84ed2a7be5fb7be850e6fdeed8bff64b01ee31c9d",
    "frames/frame-000001/history-publication.json": "db228663ef8a9b7bd68286e39ab1e27449672f1e494f30163bfd68885be8585e",
    "frames/frame-000001/history-scheduler-scalars.tsv": "dd2fd1ea14d48a4aeb445aa74bf6994623340713f20e718bc610790eef64c10b",
    "frames/frame-000001/history.cm1fhs": "ee67ee178771659fb53c7618c4542b52a066d1edae271a7a42cf2057c151d25d",
    "frames/frame-000002/history-carriers.f32": "fe3e3160688d700c47c28d9c6021986790a99d3e75ccf909ceb10b10a30e3b34",
    "frames/frame-000002/history-field-table.tsv": "a96e992520121cfd9aead41d9e0b0e7424e4951405ce6791c826053a1211f72f",
    "frames/frame-000002/history-frame-location.json": "113bf0f9e92c8ab64fdee73b00033dfbe8bf6e8b31e4e8ccd6f8b89864b9a6b6",
    "frames/frame-000002/history-publication.json": "78b57428888cdcb91e09e7fbcb7ac3486c819a121d2f7a7b0e996c3bb5002369",
    "frames/frame-000002/history-scheduler-scalars.tsv": "ee13f4e12568c3253de9f0559b15d63fc1a4fb917d51b6d3c4cd3b7957d749fc",
    "frames/frame-000002/history.cm1fhs": "0ddd384870a8934f1ae947456265b95d7ad81f8ae1a71f264e6720b38863b70d",
    "restarts/restart-000001/restart-carriers.f32": "8571758d5b53917817946c7cda201f2be940559dbfbc18141d4fca3764009c2a",
    "restarts/restart-000001/restart-field-table.tsv": "21aa0b752137fa82142622ddc7f4e95b1d137770437c64d96785da700da9224a",
    "restarts/restart-000001/restart-publication.json": "d38574d5ffa23d219446eefcbb82d766074994bcc79303a70ee325b766d30364",
    "restarts/restart-000001/restart-scheduler-scalars.tsv": "ed01813f9ad5a5b271c05db9bbdae260e6af1d356c428c8ef92580623e3e2ae9",
    "restarts/restart-000001/restart.cm1fcp": "3ac55286427f15bf79d3d8f89d1502461f75cb57ebe4b75e793ac8cd82f8ecda",
}
#: A restart directory keeps one copy of the carried state (receipt schema
#: gm1.cartesian-restart-publication/v2): the raw arena is the publisher's
#: input and is deleted once the envelope is verified to hold every one of
#: its words.  These are the only published differences from the
#: pre-profiling head -- restart-carriers.f32 is gone and
#: restart-publication.json carries the v2 schema, null ``arena_filename`` and
#: ``arena.path``, and three new arena fields -- and the test below undoes
#: exactly them, restoring the arena's words from the envelope and the
#: receipt's v1 fields, then requires the pinned digests above, byte for byte.
#: Every other file, the envelope included, is compared as published.
RESTART_ARENA_NOT_RETAINED = "restarts/restart-000001/restart-carriers.f32"
RESTART_RECEIPT_ARENA_FIELDS = ("retained", "disposition", "envelope_verification")
#: The runtime-events change made two more declared differences, and the
#: test below undoes both before it compares a digest with the pins above:
#: history-field-table.tsv names the arena relative to the frame instead of by
#: the absolute staging path (which no longer exists once the frame is
#: renamed), and history-frame-location.json's note says so.
FRAME_LOCATION_NOTE_BEFORE_EVENTS = (
    "history-publication.json and history-field-table.tsv record the "
    "staging paths the bytes were written to; this record names the "
    "published location of the same bytes"
)
FRAME_LOCATION_NOTE = (
    "history-publication.json records the staging paths the bytes were "
    "written to, and history-field-table.tsv names the arena relative "
    "to this directory; this record names the published location of "
    "the same bytes"
)
#: That change also stopped the &param9 record claiming that a word it never saw in
#: an [output] table was "declared in this run's [output] table"
#: (cm1_output_selection.OutputSelection.difference_reason).  That prose sits
#: in the configuration record every publication hashes, so the six files that
#: carry the record or its hash moved, and nothing else did: these are their
#: digests since then.  The words are still the pinned ones above -- both
#: frames' arenas, and the restart's words restored from its envelope -- and
#: with difference_reason patched back to the old sentence the same run
#: reproduces all seventeen pins above (checked when these were taken).
CONFIG_RECORD_PUBLICATION_SHA256_SINCE_EVENTS = {
    "frames/frame-000001/history-publication.json": "5586215f6f3397a17fe0678e98ad33c11c824d67ae9a93c392d4819a55c05e86",
    "frames/frame-000001/history.cm1fhs": "b885f6235c864d487d857ee59afee0a6c5cd195ea4a3100db80798b923b6f187",
    "frames/frame-000002/history-publication.json": "53bcc1a15c94f00ec3dcee9ebebfd0ad897c3bc4c8a5e1442a6641572491633e",
    "frames/frame-000002/history.cm1fhs": "18a769bb571dcddd63b250b19e7b050b1c8e8b30f0d94ac04788eb77a71ade56",
    "restarts/restart-000001/restart-publication.json": "354720b6ae9468005ca1b15a9d13de0cfe81aa7021075594151e1a99ad8a0e0c",
    "restarts/restart-000001/restart.cm1fcp": "0e26a613653e3e8639c28b8bcfa4f3d2f353d95fde227d898cb324a1673f381a",
}
#: Restart profile v2 carries the swath extrema and the ground canvas
#: in restarts, and names that in the restart profile: gm1.cartesian-restart/v2
#: with a ``swaths`` entry (null here: this run keeps no swath).  The profile
#: is hashed into the publication and the envelope's contract, so exactly these
#: two files moved; the restart still carries the same 51 arrays in 280,816
#: bytes, its words rebuilt from the envelope are the pinned
#: ``RESTART_ARENA_NOT_RETAINED`` digest, and every frame is unchanged.  Checked
#: at the re-pin (the release) by publishing the same run at 38968be5, the
#: last tree before profile v2, and at the release head: the two restart publications
#: differ only in profile_id, profile.swaths, profile_sha256, the contract hash
#: and the digests that follow from them.
RESTART_PROFILE_V2_SHA256 = {
    "restarts/restart-000001/restart-publication.json": "cc823cfe0f0db79382231d981b6af8aa96482e15794704605662ca0ab1740efa",
    "restarts/restart-000001/restart.cm1fcp": "9c837a40d0821c4ba4a847fc3a6e9981955eed86d6ae72f84f5ea2424c5941cc",
}


@pytest.mark.parametrize("background", (True, False), ids=("background", "synchronous"))
@pytest.mark.parametrize("level", ("off", "summary", "detailed"))
def test_published_frames_and_restarts_are_the_pre_profiling_bytes(
    forecast_publisher, tmp_path, level, background
) -> None:
    """Profiling, the faster writers and background publication change no word.

    The same integration the release head published is published again
    through today's arena snapshot and writer, today's publisher (SHA-NI,
    parallel and concurrent hashing, consuming encode), with every event
    committed on the background worker or on the stepping thread, and --
    unless ``off`` -- a profiler recording every I/O stage.  Every file is
    compared with the digest the release head produced.  The restart's raw
    arena is no longer kept, so its pinned bytes are rebuilt from the
    envelope that replaced it, and the receipt must be the pinned receipt
    once its declared arena-disposition fields are undone.
    """

    import hashlib
    import re

    from cm1core.runtime_profiler import RuntimeProfiler

    config = _config(u_move_m_s=12.5, v_move_m_s=3.0)
    runtime = CpuCartesianRuntime(config, seed=9)
    ranges: list[str] = []
    if level != "off":
        runtime.profiler = RuntimeProfiler(
            level, cupy=runtime.cp, stream=runtime.stream, progress_interval_s=0.0, nvml=False,
        )
        if level == "detailed":
            # This double's storage has no NVTX module; count the ranges the
            # detailed profiler would open around every I/O sub-step.
            runtime.profiler._nvtx = SimpleNamespace(
                RangePush=ranges.append, RangePop=lambda: ranges.append(None),
            )
    root = tmp_path / "published"
    receipt = runtime.integrate(
        duration_s=2.0, history_interval_s=1.0, history_root=root / "frames",
        history_publisher=forecast_publisher, restart_root=root / "restarts",
        restart_interval_s=2.0, restart_publisher=forecast_publisher,
        background_publication=background,
    )
    assert receipt["publication"]["mode"] == ("background" if background else "synchronous")
    root_bytes = os.fspath(root.resolve()).encode()

    def normalized_digest(raw: bytes) -> str:
        return hashlib.sha256(re.sub(
            rb'"(field_table_sha256|scalar_table_sha256)": "[0-9a-f]{64}"',
            rb'"\1": "<PATH-DEPENDENT>"', raw,
        ).replace(root_bytes, b"<ROOT>")).hexdigest()

    digests = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        raw = path.read_bytes()
        relative = os.fspath(path.relative_to(root))
        if path.name == "restart-publication.json":
            document = json.loads(raw)
            # The receipt's own formatting stays pinned: it is exactly the
            # writer's canonical serialization, so re-serializing the edited
            # document below normalizes nothing but the declared fields.
            assert raw == (
                json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            assert document["schema"] == CARTESIAN_RESTART_PUBLICATION_SCHEMA
            assert CARTESIAN_RESTART_PUBLICATION_SCHEMA.endswith("/v2")
            arena = document["arena"]
            assert arena["retained"] is False and arena["path"] is None
            assert document["arena_filename"] is None
            assert arena["envelope_verification"]["arrays_verified"] == len(arena["carrier_sha256"])
            # integrate stages every restart and renames it into place.
            assert arena["disposition"] == restart_arena_disposition(staged=True)
            for key in RESTART_RECEIPT_ARENA_FIELDS:
                del arena[key]
            arena["path"] = os.fspath(path.parent / "restart-carriers.f32")
            document["arena_filename"] = "restart-carriers.f32"
            document["schema"] = "gm1.cartesian-restart-publication/v1"
            raw = (json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        if path.name == "history-field-table.tsv":
            # The relative arena name, undone: the release head wrote the
            # absolute path of the arena in the frame's staging directory.
            rows = [row.split("\t") for row in raw.decode("ascii").splitlines()]
            assert {row[6] for row in rows} == {"history-carriers.f32"}
            staged = os.fspath(
                root.resolve() / "frames" / ".incomplete" / path.parent.name
                / "history-carriers.f32"
            )
            for row in rows:
                row[6] = staged
            raw = ("\n".join("\t".join(row) for row in rows) + "\n").encode("ascii")
        if path.name == "history-frame-location.json":
            note = json.dumps(FRAME_LOCATION_NOTE).encode()
            assert raw.count(note) == 1
            raw = raw.replace(note, json.dumps(FRAME_LOCATION_NOTE_BEFORE_EVENTS).encode())
        digests[relative] = normalized_digest(raw)
        if path.name.endswith("-publication.json"):
            document = json.loads(raw)
            for key, name in (("field_table_sha256", "field-table.tsv"),
                              ("scalar_table_sha256", "scheduler-scalars.tsv")):
                table = path.with_name(path.name.replace("publication.json", name))
                assert document[key] == hashlib.sha256(table.read_bytes()).hexdigest()
    # The raw arena is gone from the published restart; its words, in the
    # writer's order, come back out of the envelope exactly.
    assert not (root / RESTART_ARENA_NOT_RETAINED).exists()
    restart_home = root / "restarts" / "restart-000001"
    restored = read_cartesian_restart(
        restart_home, publisher=forecast_publisher, into=tmp_path / "restored",
    )
    published = json.loads((restart_home / "restart-publication.json").read_text())
    back = restored.arena_path.read_bytes()
    sizes = {row["name"]: row["nbytes"] for row in published["profile"]["carriers"]}
    writer_order = sorted(
        published["arena"]["carrier_offset_bytes"].items(), key=lambda item: item[1]
    )
    rebuilt = b"".join(
        back[restored.offsets[name]:restored.offsets[name] + sizes[name]]
        for name, _ in writer_order
    )
    digests[RESTART_ARENA_NOT_RETAINED] = hashlib.sha256(rebuilt).hexdigest()
    assert digests == {
        **PRE_PROFILING_PUBLICATION_SHA256, **CONFIG_RECORD_PUBLICATION_SHA256_SINCE_EVENTS,
        **RESTART_PROFILE_V2_SHA256,
    }
    if level == "off":
        assert "performance" not in receipt
        return
    io = receipt["performance"]["io"]
    assert io["history"]["events"] == 2 and io["restart"]["events"] == 1
    assert io["history"]["background_events"] == (2 if background else 0)
    assert io["restart"]["background_events"] == (1 if background else 0)
    for kind in ("history", "restart"):
        # The stepping thread never pays more than the whole publication, and
        # a synchronous one pays all of it.
        assert 0.0 < io[kind]["blocking_seconds"] <= io[kind]["seconds"] + 1e-9
        if not background:
            assert io[kind]["blocking_seconds"] == pytest.approx(io[kind]["seconds"])
    stages = {row["stage"] for row in io["restart"]["stages"]}
    assert {"arena.d2h", "arena.write", "arena.sha256_busy", "publisher",
            "publisher.read_arena_and_finite_check", "publisher.hash.carriers_and_payload",
            "publisher.write", "publisher.fsync", "envelope_verify", "arena.discard",
            "directory_fsync_rename"} <= stages
    # Nothing syncs a restart arena that is deleted once the envelope is verified.
    assert "arena.fsync" not in stages
    assert "background.join_wait" in stages if background else "background.join_wait" not in stages
    history_stages = {row["stage"] for row in io["history"]["stages"]}
    assert {"arena.d2h", "arena.write", "arena.fsync", "publisher", "receipt",
            "location_record", "directory_fsync_rename"} <= history_stages
    assert io["restart"]["bytes"] == published["arena"]["nbytes"] == len(rebuilt)
    assert ("carriers" in io["restart"]["last"]) is (level == "detailed")
    if level == "detailed":
        opened = [name for name in ranges if name is not None]
        assert len(opened) == ranges.count(None)
        assert {"io.restart", "io.restart.publisher", "io.restart.arena.write",
                "io.restart.envelope_verify", "io.history.arena.fsync",
                "io.history.device_staging_sync",
                "io.history.directory_fsync_rename"} <= set(opened)


def test_the_publisher_prints_stage_timings_only_on_request_and_never_on_stdout(
    forecast_publisher, tmp_path
) -> None:
    arena = tmp_path / "carriers.f32"
    words = np.arange(64, dtype="<f4")
    arena.write_bytes(words.tobytes())
    table = tmp_path / "table.tsv"
    table.write_text(f"th3d\tpotential temperature\tK\tmass\t4,16\t1,4,1,16\t{arena}\t0\n")
    scalars = tmp_path / "scalars.tsv"
    scalars.write_text("")
    outputs = {}
    for timed in (False, True):
        env = dict(os.environ)
        env.pop("GPUWM_CM1_PUBLISH_TIMING", None)
        if timed:
            env["GPUWM_CM1_PUBLISH_TIMING"] = "1"
        envelope = tmp_path / f"frame-{timed}.cm1fcp"
        completed = subprocess.run(
            [os.fspath(forecast_publisher), "checkpoint", os.fspath(envelope), "gm1.test/v1",
             "0" * 64, "1" * 64, "3", "1.5", "0.5", os.fspath(table), os.fspath(scalars)],
            capture_output=True, text=True, env=env, check=True,
        )
        outputs[timed] = (completed.stdout, completed.stderr, envelope.read_bytes())
    assert outputs[True][0] == outputs[False][0]
    assert outputs[True][2] == outputs[False][2]
    assert outputs[False][1] == ""
    stages = [line.split("\t")[1] for line in outputs[True][1].splitlines()]
    assert stages[0] == "read_tables" and stages[-1] == "total"
    assert {"allocate_payload", "read_arena_and_finite_check", "hash.carriers_and_payload",
            "hash.artifact", "write", "fsync", "link"} <= set(stages)


class _AdaptiveReflectivityRuntime(CpuCartesianRuntime):
    """The host double under ``adapt_dt=1``, with CM1's ``getdbz`` consulted.

    The controller (``getnewdt``, ``calcksquick``, the ``dtsmall`` rebind),
    the ``getdbz`` schedule and the frame freshness check are the shipped
    code.  The microphysics phase records only what
    ``finish.microphysics(reflectivity_due=True)`` records: the step whose
    reflectivity the carriers now hold.
    """

    adaptive_large_step = CudaCartesianRuntime.adaptive_large_step
    diffusion_number = CudaCartesianRuntime.diffusion_number
    rebind_large_step = CudaCartesianRuntime.rebind_large_step
    reflectivity_due = CudaCartesianRuntime.reflectivity_due
    # getattr, not attribute access: against a tree that predates the fix
    # there is no such method, and a class body that demanded one would raise
    # AttributeError at import and take this whole module with it -- the
    # regression below would then be a collection error instead of a run that
    # fails on the refusal it quotes.  The unfixed integrate() never calls it.
    schedule_adaptive_output_step = getattr(
        CudaCartesianRuntime, "schedule_adaptive_output_step", None
    )

    def __init__(self, config: RuntimeConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.reflectivity_frame_steps = frozenset()
        self.finish.output_dbz = True
        self.finish.completed_reflectivity_step = 0
        self.dbz_steps: list[int] = []

    def execute(self, operation, *, step: int) -> None:
        super().execute(operation, step=step)
        if operation.phase is StepPhase.MICROPHYSICS and self.reflectivity_due(step):
            self.finish.completed_reflectivity_step = step
            self.dbz_steps.append(step)


def test_an_adaptive_run_raises_getdbz_on_the_step_that_lands_on_each_frame(
    forecast_publisher, tmp_path
) -> None:
    """cm1.F:3177,3215 under adapt_dt: the output step is chosen, not counted.

    Before this, integrate() scheduled getdbz on the fixed-step positions of
    the frame times: steps 20 and 40 at dtl = 1 s with 20 s frames.  getnewdt
    grows the step by up to 10% each time (to at most 2*dtl) and then shortens
    it to land on the output time, so the first frame came at step 12, whose
    reflectivity had never been computed.  Publication refused it: "stands at
    completed step 12, but its reflectivity carriers were last written at
    step 0".  The 32x32x40 adapt_dt=1 differential cases failed on an RTX 4090
    with exactly that message, at the same cadence.
    """

    config = _config(adaptive_timestep=True, dt_s=1.0)
    runtime = _AdaptiveReflectivityRuntime(config, seed=11)
    receipt = runtime.integrate(
        duration_s=40.0,
        history_interval_s=20.0,
        history_root=tmp_path / "frames",
        history_publisher=forecast_publisher,
    )

    frames = receipt["frames"]
    assert [row["model_time_s"] for row in frames] == [20.0, 40.0]
    frame_steps = [row["step"] for row in frames]
    # The controller moved the frames off the fixed-step grid (20, 40), which
    # is the only case where the old schedule was wrong.
    assert frame_steps == [12, 22]
    assert receipt["executed_steps"] == frame_steps[-1]
    # getdbz was raised on exactly the steps a frame was published at, and on
    # no step the fixed schedule would have named instead.
    assert runtime.dbz_steps == frame_steps
    assert runtime.reflectivity_frame_steps == frozenset(frame_steps)
    assert runtime.history_frame_steps == frozenset(frame_steps)
    assert receipt["reflectivity"]["output_steps"] == frame_steps


def test_the_adaptive_output_test_is_cm1s_own_inequality() -> None:
    """(mtime+dbldt) >= (taptim-0.1d0*dbldt), evaluated with the chosen dt."""

    runtime = CudaCartesianRuntime.__new__(CudaCartesianRuntime)
    runtime.reflectivity_frame_steps = frozenset({3})
    runtime.history_frame_steps = frozenset({3})
    runtime.config = SimpleNamespace(dt_s=2.0)
    runtime.clock = SimpleNamespace(model_seconds=16.0, completed_steps=9)

    # 16 + 2 = 18 < 20 - 0.2: not an output step, nothing scheduled.
    assert runtime.schedule_adaptive_output_step(20.0) is False
    assert runtime.reflectivity_frame_steps == frozenset({3})
    # 16 + 2 = 18 >= 18.1 - 0.2: CM1 writes output after this step, although it
    # ends 0.1 s short of taptim.
    assert runtime.schedule_adaptive_output_step(18.1) is True
    assert runtime.reflectivity_frame_steps == frozenset({3, 10})
    assert runtime.history_frame_steps == frozenset({3, 10})
    # Exactly on the output time.
    runtime.clock = SimpleNamespace(model_seconds=18.0, completed_steps=10)
    assert runtime.schedule_adaptive_output_step(20.0) is True
    assert runtime.reflectivity_frame_steps == frozenset({3, 10, 11})
