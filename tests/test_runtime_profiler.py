"""Performance profiling of the Cartesian lane: timing plumbing, counts, verdicts, doors.

No device is opened.  The device-timeline arithmetic is checked against a
simulated queue whose markers land where a real stream's would; the launch
and copy hooks are checked on a CuPy-shaped namespace; the words are
checked on the real closure through ``tools.oracle.host_cupy`` (g++-compiled
packaged CUDA text) with profiling off, summary and detailed.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cm1core import runtime_profiler
from cm1core.cli import main
from cm1core.errors import CM1Refusal
from cm1core.runtime_profiler import (
    CupyHooks,
    LaunchCounters,
    RuntimeProfiler,
    format_performance,
    parse_publish_timing,
    performance_verdict,
    require_profile_level,
    require_progress_interval,
)

ROOT = Path(__file__).resolve().parents[1]
needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="the host kernel emulator needs g++")
needs_monitoring = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason="call-site launch counting needs sys.monitoring (Python 3.12+)"
)
# Kernel launches are counted through sys.monitoring (Python 3.12+); copies,
# copyto launches and compile timing are not.  The hook tests run both ways:
# without it by hiding it from the profiler, as Python 3.11 does.
MONITORING = (
    pytest.param(True, marks=needs_monitoring, id="sys.monitoring"),
    pytest.param(False, id="without-sys.monitoring"),
)


# ------------------------------------------------------- a simulated device


class SimDevice:
    """One host clock and one device queue.

    ``launch`` enqueues device work behind whatever is already queued; a
    marker executes when the queue reaches it; ``synchronize`` blocks the host
    until the queue is empty.  This is the model CUDA streams follow.
    """

    def __init__(self) -> None:
        self.host = 0.0
        self.tail = 0.0

    def clock(self) -> float:
        return self.host

    def launch(self, device_seconds: float, host_seconds: float = 0.0) -> None:
        self.tail = max(self.tail, self.host) + device_seconds
        self.host += host_seconds


class SimEvent:
    def __init__(self, device: SimDevice) -> None:
        self.device = device
        self.at: float | None = None

    def record(self, stream=None) -> None:
        self.at = max(self.device.tail, self.device.host)

    @property
    def done(self) -> bool:
        return self.at is not None and self.device.host >= self.at


class SimStream:
    def __init__(self, device: SimDevice) -> None:
        self.device = device

    def synchronize(self) -> float:
        waited = max(0.0, self.device.tail - self.device.host)
        self.device.host += waited
        return waited


def _simulated(level: str = "summary"):
    device = SimDevice()
    cupy = SimpleNamespace(cuda=SimpleNamespace(
        Event=lambda **_: SimEvent(device),
        get_elapsed_time=lambda start, end: (end.at - start.at) * 1000.0,
    ))
    stream = SimStream(device)
    profiler = RuntimeProfiler(
        level, cupy=cupy, stream=stream, progress_interval_s=0.0,
        counters=LaunchCounters(), nvml=False, clock=device.clock,
    )
    return device, stream, profiler


def _step(device, stream, profiler, phases, *, step: int) -> None:
    profiler.step_begin()
    for name, device_s, host_s, launches in phases:
        token = profiler.phase_begin(name)
        device.launch(device_s, host_s)
        profiler.counters.launches += launches
        profiler.phase_end(token)
    profiler.synchronized(stream.synchronize())
    profiler.step_end(step=step, model_time_s=0.5 * step)


def test_device_time_queue_lag_and_sync_wait_come_from_the_markers() -> None:
    """A device-bound phase: 0.5 s of device work enqueued in 0.1 s of host time."""

    device, stream, profiler = _simulated()
    for step in (1, 2, 3):
        _step(device, stream, profiler, [("rk1.acoustic", 0.5, 0.1, 40)], step=step)
    record = profiler.record(wall_seconds=device.host)
    (row,) = record["phases"]
    assert row["phase"] == "rk1.acoustic" and row["calls"] == 3
    assert row["gpu_s"] == pytest.approx(1.5)
    assert row["host_s"] == pytest.approx(0.3)
    assert row["launches"] == 120 and row["launches_per_step"] == 40
    # The first step has no reference marker yet; the next two each end their
    # enqueue 0.4 s ahead of the device.
    assert row["queue_lag_end_mean_s"] == pytest.approx(0.4)
    assert row["queue_lag_end_max_s"] == pytest.approx(0.4)
    steps = record["steps"]
    assert steps["count"] == 3
    assert steps["mean_s"] == pytest.approx(0.5)
    assert steps["sync_wait_mean_s"] == pytest.approx(0.4)
    assert steps["host_enqueue_mean_s"] == pytest.approx(0.1)
    assert steps["gpu_phase_mean_s"] == pytest.approx(0.5)
    assert record["unresolved_markers"] == 0
    assert record["verdict"]["label"] == "GPU-bound"


def test_the_booking_problem_is_solved_a_wait_lands_on_the_phase_that_queued_it() -> None:
    """Host timing books microphysics' device time to the phase that syncs next."""

    device, stream, profiler = _simulated()
    phases = [
        ("microphysics", 0.9, 0.01, 1),     # heavy device work, fast enqueue
        ("solve3", 0.05, 0.01, 1),
    ]
    for step in (1, 2):
        _step(device, stream, profiler, phases, step=step)
    rows = {row["phase"]: row for row in profiler.record()["phases"]}
    assert rows["microphysics"]["gpu_s_per_step"] == pytest.approx(0.9)
    assert rows["solve3"]["gpu_s_per_step"] == pytest.approx(0.05)
    assert rows["microphysics"]["host_s_per_step"] == pytest.approx(0.01)


def test_a_host_bound_step_has_no_backlog_and_many_launches() -> None:
    device, stream, profiler = _simulated()
    phases = [(f"rk{n}.scalar-moist", 0.001, 0.02, 150) for n in (1, 2, 3)]
    for step in (1, 2, 3):
        _step(device, stream, profiler, phases, step=step)
    record = profiler.record(wall_seconds=device.host)
    assert record["steps"]["sync_wait_mean_s"] == pytest.approx(0.0)
    assert record["steps"]["launches_per_step"] == 450
    for row in record["phases"][:1]:
        assert row["queue_lag_end_mean_s"] == pytest.approx(0.0)
    assert record["verdict"]["label"] == "host-launch-bound"


def test_io_events_record_stages_bytes_and_the_io_share() -> None:
    device, stream, profiler = _simulated("detailed")
    _step(device, stream, profiler, [("solve3", 0.5, 0.5, 1)], step=1)
    event = profiler.io_event("restart", "restart-000001")
    event.set_bytes(8 * 2**30)
    device.host += 2.0
    event.add("arena.d2h", 2.0, 8 * 2**30)
    device.host += 3.0
    event.add("publisher", 3.0)
    event.carrier("padded.u3d", 1024, d2h_s=0.5)
    event.finish()
    record = profiler.record(wall_seconds=device.host)
    table = record["io"]["restart"]
    assert table["events"] == 1 and table["seconds"] == pytest.approx(5.0)
    assert table["bytes"] == 8 * 2**30
    stages = {row["stage"]: row for row in table["stages"]}
    assert stages["arena.d2h"]["mb_per_s"] == pytest.approx(8 * 2**30 / 1e6 / 2.0)
    assert table["last"]["carriers"][0]["carrier"] == "padded.u3d"
    # One 0.5 s step, then 5 s of I/O.
    assert record["verdict"]["evidence"]["io_share_of_wall"] == pytest.approx(5.0 / 5.5)
    assert "I/O-heavy" in record["verdict"]["label"]
    # The live line names the last I/O event.
    line = profiler.progress_line(step=1, model_time_s=0.5)
    assert "last io restart 5.0s 8.0 GiB" in line


def test_nothing_retained_grows_with_the_step_count() -> None:
    sizes = []
    for steps in (5, 500):
        device, stream, profiler = _simulated()
        for step in range(1, steps + 1):
            _step(device, stream, profiler, [("a", 0.01, 0.01, 3), ("b", 0.02, 0.0, 1)], step=step)
        assert len(profiler._pool) <= 5
        sizes.append(len(json.dumps(profiler.record(), sort_keys=True)))
    # Only the widths of the numbers differ between the two records.
    assert abs(sizes[1] - sizes[0]) < 1024


def test_off_records_nothing_and_needs_no_device() -> None:
    profiler = RuntimeProfiler("off", cupy=None, progress_interval_s=0.0)
    assert profiler.phase_begin("x") is None
    profiler.phase_end(None)
    profiler.synchronized(1.0)
    assert profiler.step_begin() is None
    profiler.step_end(step=1, model_time_s=0.5)
    assert profiler.io_event("history", "frame") is None
    assert profiler.record() == {"schema": runtime_profiler.PERFORMANCE_SCHEMA, "level": "off"}


def test_the_progress_line_carries_rate_eta_and_skips_absent_telemetry() -> None:
    device, stream, profiler = _simulated()
    profiler.set_target(duration_s=100.0, total_steps=200, model_start_s=0.0)
    for step in (1, 2, 3, 4):
        # 0.25 s of device work enqueued over 0.25 s of host time: the device
        # keeps pace and each step takes 0.25 s of wall clock.
        _step(device, stream, profiler, [("solve3", 0.25, 0.25, 1)], step=step)
    line = profiler.progress_line(step=4, model_time_s=2.0)
    assert line.startswith("gm1 progress: step 4/200 t=2s/100s")
    assert "0.250 s/step recent, 0.250 mean" in line
    assert "2.000 model-s/wall-s" in line
    assert "ETA 49s" in line
    assert "GPU" not in line  # nvml=False and no NVML: skipped silently
    sink = io.StringIO()
    profiler._progress_stream = sink
    profiler.progress_interval_s = 1.0
    device.host += 5.0
    assert profiler.maybe_progress(step=4, model_time_s=2.0) is not None
    assert sink.getvalue().count("gm1 progress:") == 1
    assert profiler.maybe_progress(step=4, model_time_s=2.0) is None


def test_nvml_absence_is_silent(monkeypatch) -> None:
    import importlib

    real = importlib.import_module

    def refuse_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no NVML here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(runtime_profiler.importlib, "import_module", refuse_pynvml)
    assert runtime_profiler._Nvml(SimpleNamespace()).reading() is None


# --------------------------------------------------------------- the hooks


def _cupy_shaped() -> ModuleType:
    calls: list[tuple] = []

    class ndarray(np.ndarray):  # noqa: N801
        pass

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __call__(self, grid, block, args, stream=None):
            calls.append(("launch", self.name, grid, block, args, stream))

    class RawModule:
        def __init__(self, *, code, options=()):
            self.code = code

        def get_function(self, name):
            return Kernel(name)

    class RawKernel:
        def __init__(self, code, name):
            self.name = name

        def __call__(self, grid, block, args):
            calls.append(("raw", self.name, grid, block, args))

    class Module:
        def load(self, image):
            self.image = image

        def get_function(self, name):
            return Kernel(name)

    class Program:
        def __init__(self, source, name="", method="ptx"):
            self.source, self.name = source, name

        def compile(self, options=()):
            return b"image", {}

    cp = ModuleType("cupy_shaped")
    cp.ndarray = ndarray
    cp.RawModule, cp.RawKernel = RawModule, RawKernel
    cp.cuda = SimpleNamespace(
        function=SimpleNamespace(Module=Module, Function=Kernel),
        compiler=SimpleNamespace(_NVRTCProgram=Program),
    )
    cp.copyto = lambda dst, src: np.copyto(dst, src)
    cp.asnumpy = lambda value: np.asarray(value).view(np.ndarray).copy()
    cp.asarray = lambda value, dtype=None: np.asarray(value, dtype=dtype).view(ndarray)
    cp.array = lambda value: np.array(value).view(ndarray)
    cp.calls = calls
    return cp


@pytest.mark.parametrize("monitoring", MONITORING)
def test_the_hooks_count_launches_and_copies_and_forward_every_argument(monkeypatch, monitoring) -> None:
    if not monitoring:
        monkeypatch.setattr(runtime_profiler, "_monitoring", lambda: None)
    cp = _cupy_shaped()
    originals = {name: getattr(cp, name) for name in ("RawModule", "RawKernel", "copyto", "asnumpy", "asarray", "array")}
    original_module = cp.cuda.function.Module
    original_program = cp.cuda.compiler._NVRTCProgram
    original_compile = original_program.compile
    counters = LaunchCounters()
    compiled = []
    hooks = CupyHooks(cp, counters, compile_sink=lambda label, seconds: compiled.append(label)).install()
    try:
        assert hooks.active == ["kernel-launches"] * monitoring + ["nvrtc-compile", "copyto", "asnumpy", "asarray", "array"]
        assert ("kernel-launches" in hooks.unavailable) is not monitoring

        kernel = cp.RawModule(code="x").get_function("k1")
        kernel((4,), (1,), ("a",), stream="s")
        raw = cp.RawKernel("code", "k2")
        raw((2,), (1,), ("b",))
        module = cp.cuda.function.Module()
        module.load(b"image")
        image_kernel = module.get_function("k3")
        image_kernel((1,), (1,), ())
        # Nothing CuPy constructs or returns is substituted.
        assert cp.RawModule is originals["RawModule"] and cp.RawKernel is originals["RawKernel"]
        assert cp.cuda.function.Module is original_module
        assert type(kernel) is type(image_kernel) is cp.cuda.function.Function
        program = cp.cuda.compiler._NVRTCProgram("src", "prog_cubin.cu", method="cubin")
        assert type(program) is original_program
        assert program.compile(("--std=c++14",)) == (b"image", {})
        assert compiled == ["nvrtc:prog_cubin.cu"]
        assert [call[:5] for call in cp.calls] == [
            ("launch", "k1", (4,), (1,), ("a",)),
            ("raw", "k2", (2,), (1,), ("b",)),
            ("launch", "k3", (1,), (1,), ()),
        ]
        assert cp.calls[0][5] == "s"

        device = cp.asarray(np.arange(8, dtype=np.float32))
        other = cp.asarray(np.zeros(8, dtype=np.float32))
        cp.copyto(other, device)
        cp.copyto(other, np.ones(8, dtype=np.float32))
        host = cp.asnumpy(device)
        assert np.array_equal(host, np.arange(8, dtype=np.float32))
        assert counters.snapshot() == (
            3 * monitoring + 2,  # three kernels (counted through sys.monitoring) and two copyto launches
            1, 32,      # device -> device
            3, 96,      # two uploads through asarray, one host copyto
            1, 32,      # device -> host
        )
    finally:
        hooks.uninstall()
    for name, value in originals.items():
        assert getattr(cp, name) is value
    assert cp.cuda.function.Module is original_module
    assert cp.cuda.compiler._NVRTCProgram is original_program
    assert original_program.compile is original_compile
    assert runtime_profiler._MONITOR.entries == ()
    # Removed means removed: nothing counts any more.
    image_kernel((1,), (1,), ())
    cp.copyto(other, device)
    assert counters.launches == 3 * monitoring + 2


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
@pytest.mark.parametrize("monitoring", MONITORING)
def test_the_hooks_on_the_real_cupy_module_replace_no_class_and_restore_it(monkeypatch, monitoring) -> None:
    """No device is needed to install on CuPy itself, and removal is exact."""

    import cupy
    import cupy.cuda.compiler
    import cupy.cuda.function

    if not monitoring:
        monkeypatch.setattr(runtime_profiler, "_monitoring", lambda: None)

    def attributes():
        return (cupy.RawModule, cupy.RawKernel, cupy.copyto, cupy.asnumpy, cupy.asarray,
                cupy.array, cupy.cuda.function.Module, cupy.cuda.function.Function,
                cupy.cuda.compiler._NVRTCProgram, cupy.cuda.compiler._NVRTCProgram.__dict__["compile"])

    before = attributes()
    counters = LaunchCounters()
    hooks = CupyHooks(cupy, counters).install()
    try:
        during = attributes()
        assert during[:2] == before[:2] and during[6:9] == before[6:9]
        assert during[9] is not before[9] and during[9].__wrapped__ is before[9]
        assert hooks.faults == {}
        if monitoring:
            assert hooks.kernel_types == (cupy.cuda.function.Function, cupy.RawKernel)
            assert hooks.unavailable == {}
        else:
            assert hooks.kernel_types == ()
            assert set(hooks.unavailable) == {"kernel-launches"}

        # CuPy's kernel types are counted where Python calls them.  A Python
        # subclass stands in for a launch: calling a real handle needs a device.
        class Handle(cupy.cuda.function.Function):
            def __call__(self, *args, **kwargs):
                return None

        handle = Handle.__new__(Handle)
        handle((1,), (1,), ())
        handle((1,), (1,), ())
        assert counters.launches == 2 * monitoring
    finally:
        hooks.uninstall()
    assert attributes() == before


# ---------------------------------------------------------------- the words


@needs_gxx
def test_profiling_off_summary_and_detailed_step_the_same_raw_words() -> None:
    """The real dispatcher and closure, profiled and not: every carrier word identical."""

    from tools.oracle import cartesian_profiling_identity as harness
    from tools.oracle.cartesian_tiling_identity import compare

    from tools.oracle import host_cupy

    off = harness.run(level="off", steps=2)
    assert off["performance"] is None
    for level in ("summary", "detailed"):
        host_cupy.COUNTS.clear()
        profiled = harness.run(level=level, steps=2)
        # Every NVTX range opened (step, RK stage, phase) was closed.
        assert host_cupy.COUNTS["nvtx_push"] == host_cupy.COUNTS["nvtx_pop"]
        assert (host_cupy.COUNTS["nvtx_push"] > 0) is (level == "detailed")
        result = compare(off, profiled)
        assert result["raw_word_identical"] is True, [
            name for name, row in result["carriers"].items() if not row["raw_word_identical"]
        ]
        performance = profiled["performance"]
        # Not vacuous: the kernel launches were counted where the closure made them and every
        # marker was resolved at the synchronization.
        assert performance["totals"]["launches"] > 100
        assert performance["gpu_event_timing"] is True
        assert performance["unresolved_markers"] == 0
        assert {row["phase"] for row in performance["phases"]} >= {
            "solve1-held-tendencies", "rk1.tke-rk", "rk3.tke-rk",
        }
        assert performance["nvtx_ranges"] is (level == "detailed")
    held = off["words"]["step2.held.held.uten"]
    assert np.isfinite(held).all() and np.count_nonzero(held) > held.size // 2


# ------------------------------------------------------------------ verdicts


@pytest.mark.parametrize(
    ("steps", "io_seconds", "wall", "label"),
    (
        ({"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.6, "launches_per_step": 900}, 0.0, 7.2, "GPU-bound"),
        ({"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.01, "launches_per_step": 2500}, 0.0, 7.2, "host-launch-bound"),
        ({"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.01, "launches_per_step": 20}, 0.0, 7.2, "host-bound"),
        ({"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.1, "launches_per_step": 900}, 0.0, 7.2, "mixed"),
        ({"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.6, "launches_per_step": 900}, 30.0, 37.2, "GPU-bound; I/O-heavy"),
        ({"count": 0}, 0.0, None, "insufficient-data"),
    ),
)
def test_the_verdict_names_where_the_wall_clock_went(steps, io_seconds, wall, label) -> None:
    verdict = performance_verdict({"steps": steps, "io_seconds": io_seconds, "wall_seconds": wall})
    assert verdict["label"] == label
    assert verdict["line"].startswith(label)


# ----------------------------------------------------------------- refusals


@pytest.mark.parametrize("value", ("fast", "", "Summary", None, 1))
def test_an_unknown_profiling_level_is_refused_by_value(value) -> None:
    from cm1core.cartesian_door import require_profile_level as door_level

    with pytest.raises(ValueError):
        require_profile_level(value)
    with pytest.raises(CM1Refusal) as caught:
        door_level(value)
    assert caught.value.detail.code == "CM1-CARTESIAN-PROFILE"
    assert caught.value.detail.option == "profile"
    assert "off, summary, detailed" in caught.value.detail.remedy


@pytest.mark.parametrize("value", ("-1", "nan", "inf", "soon", True))
def test_an_unschedulable_progress_interval_is_refused_by_value(value) -> None:
    from cm1core.cartesian_door import require_progress_interval as door_interval

    with pytest.raises(ValueError):
        require_progress_interval(value)
    with pytest.raises(CM1Refusal) as caught:
        door_interval(value)
    assert caught.value.detail.code == "CM1-CARTESIAN-PROGRESS-INTERVAL"
    assert require_progress_interval("0") == 0.0
    assert require_progress_interval("12.5") == 12.5


def test_the_cartesian_command_refuses_a_bad_profile_before_touching_anything(tmp_path, capsys) -> None:
    output = tmp_path / "run.json"
    status = main([
        "cartesian", str(ROOT / "examples" / "cartesian-uniform-grid.toml"),
        "--output", str(output), "--profile", "verbose",
    ])
    assert status == 2
    assert "profiling level" in capsys.readouterr().err
    assert not output.exists()


# ------------------------------------------------------------------ the CLI


def _receipt(tmp_path: Path) -> Path:
    device, stream, profiler = _simulated()
    for step in (1, 2, 3):
        _step(device, stream, profiler, [
            ("rk1.acoustic", 0.3, 0.05, 400), ("microphysics", 0.2, 0.02, 60),
        ], step=step)
    event = profiler.io_event("history", "frame-000001")
    event.set_bytes(3 * 2**30)
    device.host += 4.0
    event.add("arena.write", 4.0, 3 * 2**30)
    event.finish()
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "schema": "gpuwm-cm1.cartesian-integration-receipt/v1",
        "runtime_benchmark": {"performance": profiler.record(wall_seconds=device.host)},
    }))
    return path


def test_perf_prints_the_verdict_phases_io_and_startup(tmp_path, capsys) -> None:
    path = _receipt(tmp_path)
    assert main(["perf", str(path)]) == 0
    out = capsys.readouterr().out
    assert "verdict: GPU-bound" in out
    lines = out.splitlines()
    table = lines.index(next(line for line in lines if line.startswith("phase ")))
    assert lines[table + 1].startswith("rk1.acoustic")  # largest device time first
    assert "microphysics" in lines[table + 2]
    assert "history: 1 events" in out and "arena.write" in out
    assert "first_step" in out
    assert main(["perf", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["steps"]["count"] == 3


def test_perf_refuses_a_receipt_without_measurements(tmp_path, capsys) -> None:
    unprofiled = tmp_path / "off.json"
    unprofiled.write_text(json.dumps({"runtime_benchmark": {"completed_steps": 4}}))
    assert main(["perf", str(unprofiled)]) == 2
    assert "no performance section" in capsys.readouterr().err
    assert main(["perf", str(tmp_path / "absent.json")]) == 2
    assert "could not be read" in capsys.readouterr().err
    assert format_performance({"level": "off"}).startswith("profiling level: off")


def test_publisher_timing_rows_are_parsed_and_foreign_lines_ignored() -> None:
    rows = parse_publish_timing(
        "gm1-publish-timing\tread_tables\t0.001000\t0\n"
        "cm1-forecast-publish: something else\n"
        "gm1-publish-timing\twrite\t2.5\t4096\n"
        "gm1-publish-timing\tbroken\tx\t1\n"
    )
    assert rows == [("read_tables", 0.001, 0), ("write", 2.5, 4096)]


# ------------------------------------------------------- the GPU measurement tool


def _flatten(mapping, prefix=""):
    rows = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.update(_flatten(value, f"{name}."))
        else:
            rows[name] = value
    return rows


@pytest.mark.parametrize("example", (
    "cartesian-uniform-grid.toml", "cartesian-supercell-moore-20130520-18z-16gib.toml",
))
def test_the_profile_tool_changes_only_the_clock_and_the_horizontal_grid(tmp_path, example) -> None:
    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import load_config
    from tools.profile_cartesian_run import variants

    document = ROOT / "examples" / example
    texts = variants(document, steps=6, warmup=2, io_steps=1, io=True, half_mode="keep-domain")
    assert texts["off"] == texts["full"]
    original = load_config(document)
    base = _flatten(original.canonical_dict())
    clock = {"run.duration_s", "run.history_interval_s", "run.restart_interval_s", "runtime.warmup_steps"}
    allowed = {
        "full": clock,
        "off": clock,
        "io": clock,
        "half": clock | {"grid.nx", "grid.ny", "grid.dx_m", "grid.dy_m",
                         "runtime.tile_nx", "runtime.tile_ny",
                         "runtime.physics_block_nx", "runtime.physics_block_ny"},
    }
    for name, text in texts.items():
        path = tmp_path / f"{name}.toml"
        path.write_text(text)
        config = load_config(path)
        changed = {key for key, value in _flatten(config.canonical_dict()).items() if base.get(key) != value}
        assert changed <= allowed[name], (name, changed - allowed[name])
        steps = 1 if name == "io" else 6
        assert config.run.duration_s == steps * original.run.dt_s
        build_runtime_config(config)  # the door's grid/tile admission still passes
    half = load_config(tmp_path / "half.toml")
    assert (half.grid.nx, half.grid.ny) == (original.grid.nx // 2, original.grid.ny // 2)
    # keep-domain: the same patch of ground, so every metre placement still fits.
    assert half.grid.nx * half.grid.dx_m == original.grid.nx * original.grid.dx_m
    io_config = load_config(tmp_path / "io.toml")
    assert io_config.run.restart_interval_s == io_config.run.history_interval_s == io_config.run.dt_s


def test_the_profile_tool_runs_the_door_with_profiling_and_the_io_directories(tmp_path) -> None:
    from tools.profile_cartesian_run import door_command

    args = SimpleNamespace(python="python3", profile="detailed", progress_interval=30.0, io=True,
                           identity=True, sounding=Path("/s"), library=None,
                           history_publisher=None, gpu_uuid="GPU-1", lock=Path("/lock"))
    full = door_command("full", tmp_path / "doc.toml", tmp_path / "full", args)
    assert full[:4] == ["python3", "-m", "cm1core", "cartesian"]
    assert full[full.index("--profile") + 1] == "detailed"
    assert "--history-dir" in full and "--integrate" not in full
    io_run = door_command("io", tmp_path / "doc.toml", tmp_path / "io", args)
    assert {"--integrate", "--history-dir", "--restart-dir"} <= set(io_run)
    assert io_run[io_run.index("--gpu-uuid") + 1] == "GPU-1"
    half = door_command("half", tmp_path / "doc.toml", tmp_path / "half", args)
    assert "--history-dir" not in half
    # The identity run is the full run unprofiled, publishing its own frame.
    off = door_command("off", tmp_path / "doc.toml", tmp_path / "off", args)
    assert off[off.index("--profile") + 1] == "off" and "--history-dir" in off


def test_the_scaling_split_separates_fixed_and_per_point_cost() -> None:
    from tools.profile_cartesian_run import scaling

    def run(mean):
        return {"performance": {"steps": {"mean_s": mean}}}

    # 0.02 s fixed + 0.175 s per quarter-grid: device-bound.
    device = scaling(run(0.02 + 4 * 0.175), run(0.02 + 0.175), 4.0)
    assert device["fixed_per_step_s"] == pytest.approx(0.02)
    assert device["ratio"] == pytest.approx(0.72 / 0.195)
    assert device["reading"].startswith("device-bound")
    # 0.70 s fixed + 0.005 s per quarter-grid: host-bound.
    host = scaling(run(0.70 + 4 * 0.005), run(0.705), 4.0)
    assert host["fixed_per_step_s"] == pytest.approx(0.70)
    assert host["reading"].startswith("host-bound")
    assert scaling({}, run(0.1), 4.0) == {"available": False}


_FAKE_DOOR = r'''#!/usr/bin/env python3
"""Stands in for `python -m cm1core cartesian`: a receipt, a frame, no device."""
import json, sys, tomllib
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ["-m", "cm1core", "cartesian"], args
document = tomllib.loads(Path(args[3]).read_text())
option = lambda name: args[args.index(name) + 1] if name in args else None
points = document["grid"]["nx"] * document["grid"]["ny"]
mean = 0.02 + 0.7 * points / FULL_POINTS
steps = {"count": 4, "mean_s": mean, "sync_wait_mean_s": 0.6 * mean, "launches_per_step": 900}
performance = {"schema": "gm1.cartesian-performance/v1", "level": option("--profile"),
               "steps": steps, "phases": [], "io": {}, "startup": {}, "io_seconds": 0.0,
               "wall_seconds": 4 * mean}
benchmark = {"measured_steps": 4, "median_step_wall_seconds": mean}
if option("--profile") != "off":
    from cm1core.runtime_profiler import performance_verdict
    performance["verdict"] = performance_verdict(performance)
    benchmark["performance"] = performance
    # The harness's own step clock, which every level records: profiling costs 2%.
    benchmark["mean_step_wall_seconds"] = mean * 1.02
else:
    benchmark["mean_step_wall_seconds"] = mean
Path(option("--output")).write_text(json.dumps({"runtime_benchmark": benchmark}))
history = option("--history-dir")
if history:
    frame = Path(history) / "frame"
    frame.mkdir(parents=True)
    words = bytes(range(24))
    (frame / "history-carriers.f32").write_bytes(words)
    (frame / "history-field-table.tsv").write_text("u\tm\tu\tmass\t6\t1,6\tpath\t0\n")
    (frame / "history-scheduler-scalars.tsv").write_text("x\tu64\t1\n")
print("progress line", file=sys.stderr)
'''


def test_the_profile_tool_reports_scaling_identity_and_verdicts_end_to_end(tmp_path) -> None:
    """The tool's own orchestration, with a stand-in door in place of the GPU run."""

    import os

    from tools.profile_cartesian_run import main as profile_main

    document = ROOT / "examples" / "cartesian-supercell-moore-20130520-18z-16gib.toml"
    fake = tmp_path / "fake-python"
    grid = tomllib_loads(document.read_text())["grid"]
    fake.write_text(_FAKE_DOOR.replace("FULL_POINTS", str(grid["nx"] * grid["ny"])))
    fake.chmod(0o755)
    env_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + ([env_path] if env_path else []))
    try:
        status = profile_main([
            str(document), "--steps", "4", "--warmup", "1", "--identity",
            "--workdir", str(tmp_path / "work"), "--python", str(fake), "--installed",
        ])
    finally:
        if env_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = env_path
    assert status == 0
    summary = json.loads((tmp_path / "work" / "profile-summary.json").read_text())
    assert set(summary["runs"]) == {"full", "off", "half"}
    assert summary["identity"]["raw_word_identical"] is True
    scaling = summary["scaling"]
    assert scaling["points_ratio"] == 4.0
    assert scaling["fixed_per_step_s"] == pytest.approx(0.02)
    assert scaling["reading"].startswith("device-bound")
    report = (tmp_path / "work" / "profile-report.txt").read_text()
    assert "verdict: GPU-bound" in report and "raw_word_identical=True" in report
    assert (tmp_path / "work" / "full" / "stderr.txt").read_text() == "progress line\n"
    # The profiled step against the unprofiled one, from the harness's own
    # clock in each receipt: what --profile detailed cost the step.
    overhead = summary["profiling_overhead"]
    assert overhead["available"] is True and overhead["level"] == "detailed"
    assert overhead["ratio"] == pytest.approx(1.02) and overhead["median_ratio"] == pytest.approx(1.0)
    assert overhead["excess_s_per_step"] == pytest.approx(0.02 * 0.72)
    assert summary["runs"]["off"]["step_wall"] == {"mean_s": pytest.approx(0.72), "median_s": pytest.approx(0.72), "steps": 4}
    assert "step wall: --profile detailed" in report and "ratio 1.020" in report


def test_profiling_overhead_needs_both_step_clocks() -> None:
    from tools.profile_cartesian_run import profiling_overhead, step_wall

    assert step_wall({}) is None
    assert step_wall({"mean_step_wall_seconds": 0.5}) == {"mean_s": 0.5, "median_s": None, "steps": None}
    on = {"step_wall": {"mean_s": 0.51, "median_s": 0.5, "steps": 20}}
    off = {"step_wall": {"mean_s": 0.50, "median_s": 0.5, "steps": 20}}
    overhead = profiling_overhead(on, off, level="summary")
    assert overhead["available"] is True and overhead["ratio"] == pytest.approx(1.02)
    assert overhead["excess_s_per_step"] == pytest.approx(0.01) and overhead["median_ratio"] == pytest.approx(1.0)
    assert overhead["steps"] == {"profiled": 20, "off": 20}
    assert profiling_overhead(on, {}, level="summary") == {"available": False, "level": "summary"}
    assert profiling_overhead({}, off, level="detailed") == {"available": False, "level": "detailed"}


def tomllib_loads(text):
    import tomllib

    return tomllib.loads(text)
