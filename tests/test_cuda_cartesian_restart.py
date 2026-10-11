"""The Cartesian restart on the device it is for.  GPU-gated and opt-in.

Everything in this file opens a CUDA context, builds the real Rust-initialized
atmosphere and runs real kernels, so it is skipped wherever CuPy or a device is
absent, and it runs only when asked for (``GM1_CARTESIAN_RESTART_GPU=1``).
Its device run is not part of the recorded release evidence.  The host-side
proof of the same claim -- that a stopped run resumes bit for bit through real CM1FCP1 envelope bytes -- is
``tests/test_cartesian_restart.py``, which runs everywhere and exercises the
identical persistence path over a host-stepped state.

What this adds that the host proof cannot: the carriers are the ones the CUDA
kernels actually wrote, including padded halo words the device operators touch
and the ArWen WSM6 persistence the component owns, and the twenty steps after
the resume are real large steps through the real acoustic, scalar, TKE, mass
and microphysics schedule.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from cm1core.cartesian_restart import (
    RESTART_DIRECTORY_FORMAT,
    cartesian_restart_profile,
    read_cartesian_restart,
    restore_cartesian_restart,
    write_cartesian_restart,
)
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
# A device test that silently runs on a GPU another job is using is worse than
# one that does not run at all, so this needs an explicit opt-in as well as CuPy.
pytestmark = pytest.mark.skipif(
    os.environ.get("GM1_CARTESIAN_RESTART_GPU") != "1",
    reason=(
        "opens a CUDA context and runs real kernels; set "
        "GM1_CARTESIAN_RESTART_GPU=1 on a device no other job is using"
    ),
)

# A grid small enough to build twice in one process and still be a real
# three-dimensional integration through every bound operator.
GRID = CartesianGridGeometry(32, 32, 24, 500.0, 500.0, 400.0)
STEPS_BEFORE_STOP = 8
STEPS_AFTER_STOP = 8


@pytest.fixture(scope="session")
def forecast_publisher(tmp_path_factory):
    target = tmp_path_factory.mktemp("gm1-restart-publisher-gpu")
    subprocess.run(
        [
            "cargo", "build", "--release", "--bin", "cm1_forecast_publish",
            "--manifest-path", str(ROOT / "rust/cm1-state/Cargo.toml"),
            "--target-dir", str(target),
        ],
        check=True, capture_output=True,
    )
    return (target / "release" / "cm1_forecast_publish").resolve(strict=True)


@pytest.fixture(scope="session")
def initializer_library():
    from cm1core import rust_tools

    return rust_tools.native_library_path("cm1-state")


@pytest.fixture(scope="session")
def sounding_bytes():
    from cm1core.cartesian_door import resolve_sounding

    return resolve_sounding(None)[1]


def _settings() -> RuntimeConfig:
    return RuntimeConfig(GRID, memory_budget_bytes=6 * 1024**3)


def _raw_state(runtime, profile) -> dict[str, bytes]:
    """Every persisted carrier, copied to the host as raw little-endian words."""

    groups = {
        "padded": runtime.dynamics.arrays,
        "fields": runtime.dynamics.fields,
        "held": runtime.dynamics.held,
        "tke": runtime.dynamics.tke,
        "component": runtime.finish.persistence,
    }
    rows = {
        carrier.name: runtime.cp.asnumpy(
            groups[carrier.group][carrier.key]
        ).astype("<f4", copy=False).tobytes()
        for carrier in profile.carriers
    }
    rows["#asq"] = runtime.cp.asnumpy(runtime.finish.asq).astype("<f8", copy=False).tobytes()
    rows["#clock"] = repr((
        runtime.clock.completed_steps,
        runtime.clock.domainlocx_m,
        runtime.clock.domainlocy_m,
        runtime.finish.initial_mass1_f64,
    )).encode("ascii")
    return rows


def _build(cupy, stream, library, sounding_bytes, settings):
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.gm1_idealized_residency import IdealizedSource

    source = IdealizedSource(library, sounding_bytes, settings.grid)
    runtime = CudaCartesianRuntime(source, settings, cupy=cupy, stream=stream)
    return source, runtime


def test_a_stopped_cuda_run_resumes_bit_for_bit(
    forecast_publisher, initializer_library, sounding_bytes, tmp_path
) -> None:
    """The deliverable on the device: real kernels, real envelope, same words.

    An uninterrupted sixteen-step integration, and the same integration
    stopped at eight, checkpointed to a CM1FCP1 envelope, read back into a
    freshly initialized runtime and stepped the remaining eight.  Every
    persisted carrier is compared as raw device words.
    """

    cupy = pytest.importorskip("cupy")
    settings = _settings()
    profile = cartesian_restart_profile(settings)

    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, whole = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        for _ in range(STEPS_BEFORE_STOP + STEPS_AFTER_STOP):
            whole.step()
        whole.synchronize()
        uninterrupted = _raw_state(whole, profile)
    del whole
    cupy.get_default_memory_pool().free_all_blocks()

    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, stopped = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        for _ in range(STEPS_BEFORE_STOP):
            stopped.step()
        stopped.synchronize()
        halfway = _raw_state(stopped, profile)
        receipt = write_cartesian_restart(
            stopped, tmp_path / RESTART_DIRECTORY_FORMAT.format(number=1),
            publisher=forecast_publisher, profile=profile,
            restart_number=1, next_history_index=1,
        )
    assert receipt["step"] == STEPS_BEFORE_STOP
    assert receipt["finite_checked_by_rust"] is True
    del stopped
    cupy.get_default_memory_pool().free_all_blocks()

    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, resumed = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        restored = read_cartesian_restart(
            tmp_path / RESTART_DIRECTORY_FORMAT.format(number=1),
            publisher=forecast_publisher, into=tmp_path / "restored",
        )
        record = restore_cartesian_restart(resumed, restored, profile=profile)
        assert record["step"] == STEPS_BEFORE_STOP
        assert _raw_state(resumed, profile) == halfway
        for _ in range(STEPS_AFTER_STOP):
            resumed.step()
        resumed.synchronize()
        continued = _raw_state(resumed, profile)

    assert set(continued) == set(uninterrupted)
    differing = sorted(
        name for name in continued if continued[name] != uninterrupted[name]
    )
    assert differing == [], differing


def test_the_cuda_initial_condition_is_written_and_nothing_is_integrated(
    forecast_publisher, initializer_library, sounding_bytes, tmp_path
) -> None:
    """cm1.F:4341-4350 on the device: build it, write it, stop."""

    cupy = pytest.importorskip("cupy")
    settings = _settings()
    profile = cartesian_restart_profile(settings)

    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, runtime = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        initial = _raw_state(runtime, profile)
        receipt = runtime.initial_condition(
            restart_root=tmp_path / "initial",
            restart_publisher=forecast_publisher,
            history_root=tmp_path / "initial-frame",
            history_publisher=forecast_publisher,
        )
        assert receipt["integrated"] is False
        assert receipt["completed_steps"] == 0
        assert runtime.clock.completed_steps == 0
        assert _raw_state(runtime, profile) == initial
    del runtime
    cupy.get_default_memory_pool().free_all_blocks()

    # The initial condition resumes like any other restart, which is what
    # makes it worth writing: the initializer is paid for once.
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, later = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        restored = read_cartesian_restart(
            tmp_path / "initial" / RESTART_DIRECTORY_FORMAT.format(number=1),
            publisher=forecast_publisher, into=tmp_path / "restored",
        )
        restore_cartesian_restart(later, restored, profile=profile)
        assert _raw_state(later, profile) == initial


def test_a_cuda_integration_writes_the_declared_restarts(
    forecast_publisher, initializer_library, sounding_bytes, tmp_path
) -> None:
    """The shipped integration loop, with a restart cadence, on the device."""

    cupy = pytest.importorskip("cupy")
    settings = _settings()
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        source, runtime = _build(cupy, stream, initializer_library, sounding_bytes, settings)
    with source:
        receipt = runtime.integrate(
            duration_s=4.0, history_interval_s=1.0,
            history_root=tmp_path / "frames", history_publisher=forecast_publisher,
            restart_root=tmp_path / "restarts", restart_interval_s=2.0,
            restart_publisher=forecast_publisher,
        )
    assert receipt["executed_steps"] == 8
    assert [row["number"] for row in receipt["restarts"]] == [1, 2]
    assert [row["step"] for row in receipt["restarts"]] == [4, 8]
    assert sorted(
        path.name for path in (tmp_path / "restarts").glob("restart-*")
    ) == ["restart-000001", "restart-000002"]
