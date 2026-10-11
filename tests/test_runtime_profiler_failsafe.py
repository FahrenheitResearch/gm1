"""The profiler under CuPy 14's type checks, and a profiler that never stops a run.

GPU verification on an RTX 4090 with CuPy 14.2.0 found ``--profile summary``
(the default) and ``--profile detailed`` crashing every run in ``cupy.full``::

    TypeError: Cannot convert _CountedKernel to cupy.cuda.function.Function

The hooks had replaced ``cupy.cuda.function.Module`` with a subclass whose
``get_function`` returned a counting wrapper.  CuPy's compiler builds its own
modules through that same attribute, and its Cython code assigns what
``get_function`` returns to a ``cdef function.Function``.  The stand-in below
enforces exactly those rules on the host; a calibration test checks them
against the CuPy 14.2 installed here, without a device.

No device is opened.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cm1core import runtime_profiler
from cm1core.runtime_profiler import (
    CupyHooks,
    LaunchCounters,
    RuntimeProfiler,
    format_performance,
    performance_verdict,
)

needs_monitoring = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason="call-site launch counting needs sys.monitoring (Python 3.12+)"
)


# ------------------------------------------------------ CuPy 14.2, on the host


class _Extension(type):
    """A Cython extension type: its own attributes cannot be reassigned.

    ``cupy.cuda.function.Function.__call__ = ...`` raises ``TypeError: cannot
    set '__call__' attribute of immutable type``.  Python subclasses of an
    extension type are ordinary heap types and stay mutable.
    """

    def __setattr__(cls, name, value):
        if cls.__dict__.get("_extension_type"):
            raise TypeError(f"cannot set {name!r} attribute of immutable type {cls.__qualname__!r}")
        super().__setattr__(name, value)


def _cupy14_shaped() -> ModuleType:
    """CuPy 14.2 as its Cython sources behave, with host arrays.

    * ``cupy/cuda/function.pyx:134,174``: ``Function`` and ``Module`` are
      extension types; ``Function.__init__(self, Module module, str funcname)``.
    * ``cupy/cuda/compiler.py:715,1055``: every compile makes its module with
      ``function.Module()``, looked up on ``cupy.cuda.function`` at call time.
    * ``cupy/_core/_kernel.pyx:110-113`` (every elementwise kernel, among them
      the copy ``cupy.copyto`` launches for a scalar) and
      ``cupy/_core/raw.pyx:114-129`` (``RawKernel``): the object
      ``module.get_function(name)`` returns is assigned to a
      ``cdef function.Function`` -- ``__Pyx_TypeTest`` against the type
      itself, not the module attribute.
    * ``cupy/_creation/basic.py:328-329``: ``cupy.full`` is Python; it
      allocates, then fills with ``cupy.copyto(a, fill_value,
      casting='unsafe')`` looked up on the ``cupy`` module at call time, and
      ``cupy/_manipulation/basic.py:36-42,102-104`` launches the elementwise
      copy for a scalar source.
    * ``cupy/_core/raw.pyx:470``: ``RawModule.get_function`` constructs the
      ``RawKernel`` class directly, not ``cupy.RawKernel``.
    * CuPy's own launches run in C and raise no Python CALL event; here they
      reach the kernel through the class (``Function.linear_launch(kern, ...)``,
      ``Function.__call__(kern, ...)``), never by calling a kernel object.
    """

    launches: list[tuple] = []
    copies: list[str] = []

    class ndarray(np.ndarray):  # noqa: N801 - CuPy's spelling
        pass

    class Module(metaclass=_Extension):
        _extension_type = True

        def __init__(self):
            self.ptr = 0

        def load(self, cubin):
            self.ptr = len(bytes(cubin))

        def get_function(self, name):
            return Function(self, name)

    class Function(metaclass=_Extension):
        _extension_type = True

        def __init__(self, module, funcname):
            if not isinstance(module, Module):
                raise TypeError(
                    "Argument 'module' has incorrect type (expected "
                    f"cupy.cuda.function.Module, got {type(module).__name__})"
                )
            self.module, self.name = module, funcname

        def __call__(self, grid, block, args, shared_mem=0, stream=None,
                     enable_cooperative_groups=False):
            launches.append(("call", self.name, grid, block, args, stream))

        def linear_launch(self, size, args, shared_mem=0, block_max_size=128,
                          stream=None, enable_cooperative_groups=False):
            launches.append(("linear", self.name, size))

    def as_function(value):
        # cdef function.Function kern = value
        if not isinstance(value, Function):
            raise TypeError(f"Cannot convert {type(value).__name__} to cupy.cuda.function.Function")
        return value

    class _NVRTCProgram:
        def __init__(self, src, name="default_program", headers=(), include_names=(),
                     name_expressions=None, method="ptx"):
            self.src, self.name, self.method = src, name, method

        def compile(self, options=(), log_stream=None):
            return b"\x7fELF" + self.src.encode("utf-8"), None

    function = ModuleType("cupy14_shaped.cuda.function")
    function.Module, function.Function = Module, Function
    compiler = ModuleType("cupy14_shaped.cuda.compiler")
    compiler._NVRTCProgram = _NVRTCProgram

    def _compile_module_with_cache(source, options=()):
        image, _mapping = compiler._NVRTCProgram(source, "kernel-cache.cu", method="cubin").compile(options)
        module = function.Module()
        module.load(image)
        return module

    compiler._compile_module_with_cache = _compile_module_with_cache

    class RawKernel(metaclass=_Extension):
        _extension_type = True

        def __init__(self, code, name, options=()):
            self.code, self.name, self.options = code, name, options
            self._kernel_cache = None

        @property
        def kernel(self):
            if self._kernel_cache is None:
                module = compiler._compile_module_with_cache(self.code, self.options)
                self._kernel_cache = as_function(module.get_function(self.name))
            return self._kernel_cache

        def __call__(self, grid, block, args, **kwargs):
            Function.__call__(self.kernel, grid, block, args, **kwargs)

    class RawModule(metaclass=_Extension):
        _extension_type = True

        def __init__(self, *, code, options=(), name_expressions=None):
            self.code, self.options = code, options

        def get_function(self, name):
            kernel = RawKernel(self.code, name, self.options)
            kernel.kernel  # noqa: B018 - raw.pyx registers the kernel in its cache
            return kernel

    elementwise: dict[str, Function] = {}

    def elementwise_copy(src, dst):
        kern = elementwise.get("copy")
        if kern is None:
            module = compiler._compile_module_with_cache("cupy_copy")
            kern = elementwise["copy"] = as_function(module.get_function("cupy_copy"))
        Function.linear_launch(kern, dst.size, (src, dst))
        dst[...] = src

    def full(shape, fill_value, dtype=None):
        if dtype is None:
            dtype = np.array(fill_value).dtype
        out = np.empty(shape, dtype=dtype).view(ndarray)
        cp.copyto(out, fill_value, casting="unsafe")  # the module attribute, as basic.py:329
        return out

    def copyto(dst, src, casting="same_kind", where=None):
        copies.append("copyto")
        if type(src) in (bool, int, float, complex):
            np.copyto(np.empty((), dtype=dst.dtype), src, casting=casting)
            elementwise_copy(src, dst)
            return
        np.copyto(dst, src, casting=casting)

    cp = ModuleType("cupy14_shaped")
    cp.__version__ = "14.2.0"
    cp.ndarray = ndarray
    cp.RawModule, cp.RawKernel = RawModule, RawKernel
    cp.full = full
    cp.copyto = copyto
    cp.asnumpy = lambda a, stream=None: np.asarray(a).view(np.ndarray).copy()
    cp.asarray = lambda a, dtype=None: np.asarray(a, dtype=dtype).view(ndarray)
    cp.array = lambda obj, dtype=None, copy=True: np.array(obj, dtype=dtype).view(ndarray)
    cp.cuda = SimpleNamespace(function=function, compiler=compiler)
    cp.launches, cp.copies = launches, copies
    return cp


class _CountedKernel:
    """The wrapper the hooks used to return from ``Module.get_function``."""

    def __init__(self, kernel):
        self._kernel = kernel

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)


def test_the_stand_in_refuses_what_cupy_14_refused_on_the_gpu() -> None:
    """The stand-in reproduces the RTX 4090 failure for the old hook's shape."""

    cp = _cupy14_shaped()
    original = cp.cuda.function.Module

    class CountingModule(original):
        def get_function(self, name):
            return _CountedKernel(original.get_function(self, name))

    cp.cuda.function.Module = CountingModule
    with pytest.raises(TypeError, match=r"^Cannot convert _CountedKernel to cupy\.cuda\.function\.Function$"):
        cp.full((4,), 1.0)
    with pytest.raises(TypeError, match=r"^Cannot convert _CountedKernel to cupy\.cuda\.function\.Function$"):
        cp.RawModule(code="gm1").get_function("gm1_kernel")
    with pytest.raises(TypeError, match="immutable type"):
        cp.cuda.function.Function.__call__ = lambda *args, **kwargs: None


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_the_installed_cupy_type_checks_as_the_stand_in_does() -> None:
    """Calibration against CuPy itself, with no device: ``__new__`` skips ``__init__``."""

    import cupy
    import cupy.cuda.function as function

    kernel = function.Function.__new__(function.Function)
    with pytest.raises(TypeError, match=r"^Cannot convert _CountedKernel to cupy\.cuda\.function\.Module$"):
        kernel.module = _CountedKernel(None)
    counting_module = type("CountingModule", (function.Module,), {})
    kernel.module = counting_module.__new__(counting_module)  # a subclass passes the same test
    for extension in (function.Function, function.Module, cupy.RawKernel, cupy.RawModule):
        before = extension.__dict__.get("__call__")
        try:
            with pytest.raises(TypeError, match="immutable type"):
                extension.__call__ = lambda *args, **kwargs: None
        finally:
            if extension.__dict__.get("__call__") is not before:  # pragma: no cover - never reached
                type.__setattr__(extension, "__call__", before)


@needs_monitoring
def test_cupy_14_builds_its_own_kernels_under_the_hooks_and_gm1_kernels_still_count() -> None:
    """Integration branch: TypeError: Cannot convert _CountedKernel to cupy.cuda.function.Function."""

    cp = _cupy14_shaped()
    counters = LaunchCounters()
    compiled: list[str] = []
    classes = (cp.RawModule, cp.RawKernel, cp.cuda.function.Module, cp.cuda.function.Function,
               cp.cuda.compiler._NVRTCProgram)
    hooks = CupyHooks(cp, counters, compile_sink=lambda label, seconds: compiled.append(label)).install()
    try:
        filled = cp.full((4,), 2.0)
        assert np.array_equal(filled, np.full(4, 2.0))
        # The two ways GM1 builds kernels: a loaded image, and RawModule.
        module = cp.cuda.function.Module()
        module.load(b"\x7fELF gm1 image")
        image_kernel = module.get_function("gm1_image_kernel")
        raw_kernel = cp.RawModule(code="gm1 source").get_function("gm1_raw_kernel")
        direct = cp.RawKernel("gm1 source", "gm1_direct_kernel")
        for _ in range(3):
            image_kernel((8,), (4,), ("a",))
        raw_kernel((2,), (1,), ("b",))
        direct((1,), (1,), ())
        cp.full((2,), 0.0)
        # Nothing CuPy constructs or returns is substituted.
        assert (cp.RawModule, cp.RawKernel, cp.cuda.function.Module, cp.cuda.function.Function,
                cp.cuda.compiler._NVRTCProgram) == classes
        assert type(image_kernel) is cp.cuda.function.Function
        assert type(raw_kernel) is type(direct) is cp.RawKernel
    finally:
        hooks.uninstall()
    assert [launch[:2] for launch in cp.launches] == [
        ("linear", "cupy_copy"),
        ("call", "gm1_image_kernel"), ("call", "gm1_image_kernel"), ("call", "gm1_image_kernel"),
        ("call", "gm1_raw_kernel"),
        ("call", "gm1_direct_kernel"),
        ("linear", "cupy_copy"),
    ]
    assert cp.copies == ["copyto", "copyto"]
    # GM1's five launches count, and so does each fill: cupy.full fills through
    # cupy.copyto, which the copy hook counts, as it did before.
    assert counters.launches == 5 + 2
    # The elementwise copy, the RawModule and the direct RawKernel each compiled
    # once; the image GM1 loaded needed no compile.
    assert compiled == ["nvrtc:kernel-cache.cu"] * 3
    assert hooks.faults == {} and hooks.unavailable == {}


@needs_monitoring
def test_a_call_site_that_launched_a_kernel_keeps_counting_after_calling_something_else() -> None:
    cp = _cupy14_shaped()
    counters = LaunchCounters()
    module = cp.cuda.function.Module()
    kernel = module.get_function("gm1_kernel")

    def launch(target):
        target((1,), (1,), ())

    def host_double(grid, block, args):
        return None

    hooks = CupyHooks(cp, counters).install()
    try:
        launch(kernel)
        launch(host_double)
        launch(kernel)
    finally:
        hooks.uninstall()
    assert counters.launches == 2
    # And a later profiler counts again at a site that launched under an earlier one.
    counters = LaunchCounters()
    hooks = CupyHooks(cp, counters).install()
    try:
        launch(kernel)
    finally:
        hooks.uninstall()
    assert counters.launches == 1


@needs_monitoring
@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="the de-instrumentation of a disabled site is asserted on CPython 3.12")
def test_a_disabled_call_site_is_de_instrumented_and_a_launch_site_stays_instrumented() -> None:
    """What the process-wide CALL instrumentation costs, as the bytecode shows it.

    Setting CALL events instruments every code object that runs; a site that
    answered DISABLE is de-instrumented at once and re-specializes, while a
    site that launched a kernel keeps ``INSTRUMENTED_CALL`` (its callable is
    an extension type's ``__call__``, which never specializes anyway).
    """

    import dis
    import io

    cp = _cupy14_shaped()
    kernel = cp.cuda.function.Module().get_function("gm1_kernel")

    def host_double(grid, block, args):
        return None

    namespace: dict = {}
    exec(compile(
        "def launch_site(target):\n    target((1,), (1,), ())\n"
        "def host_site(target):\n    target((1,), (1,), ())\n",
        "<call sites>", "exec"), namespace)
    launch_site, host_site = namespace["launch_site"], namespace["host_site"]

    def call_opcodes(function):
        text = io.StringIO()
        dis.dis(function, adaptive=True, file=text)
        return [line.split()[1] for line in text.getvalue().splitlines()
                if len(line.split()) > 1 and line.split()[1].split("_")[0] in ("CALL", "INSTRUMENTED")]

    for _ in range(200):
        host_site(host_double)
    assert call_opcodes(host_site) == ["CALL_PY_EXACT_ARGS"]
    counters = LaunchCounters()
    hooks = CupyHooks(cp, counters).install()
    try:
        launch_site(kernel)
        assert call_opcodes(launch_site) == ["INSTRUMENTED_CALL"]
        host_site(host_double)  # DISABLE: de-instrumented at once ...
        assert call_opcodes(host_site) == ["CALL"]
        for _ in range(200):
            host_site(host_double)
        assert call_opcodes(host_site) == ["CALL_PY_EXACT_ARGS"]  # ... and specialized again
        for _ in range(200):
            launch_site(kernel)
        assert call_opcodes(launch_site) == ["INSTRUMENTED_CALL"]
    finally:
        hooks.uninstall()
    assert counters.launches == 201


@needs_monitoring
def test_restart_events_is_called_only_to_rearm_sites_an_earlier_profiler_disabled(monkeypatch) -> None:
    """``sys.monitoring.restart_events`` re-arms every tool's disabled events, so
    a run's first profiler never calls it; a later one does, and must."""

    cp = _cupy14_shaped()
    kernel = cp.cuda.function.Module().get_function("gm1_kernel")
    restarts: list[int] = []

    class Watched:
        def __getattr__(self, name):
            return getattr(sys.monitoring, name)

        def restart_events(self):
            restarts.append(1)
            sys.monitoring.restart_events()

    monkeypatch.setattr(runtime_profiler, "_monitoring", lambda: Watched())
    # Earlier tests in this interpreter disabled sites of their own; this one
    # starts as a process whose profiler has disabled nothing yet.
    runtime_profiler._MONITOR.disabled = False
    namespace: dict = {}
    exec(compile("def site(target):\n    target((1,), (1,), ())\n", "<a mixed call site>", "exec"), namespace)
    site = namespace["site"]

    def host_double(grid, block, args):
        return None

    first = LaunchCounters()
    hooks = CupyHooks(cp, first).install()
    try:
        assert restarts == []
        site(host_double)  # not a kernel: the site answers DISABLE
        site(kernel)       # so this launch is not counted, as the receipt says
    finally:
        hooks.uninstall()
    assert first.launches == 0 and runtime_profiler._MONITOR.disabled is True
    second = LaunchCounters()
    hooks = CupyHooks(cp, second).install()
    try:
        assert restarts == [1]
        site(kernel)
    finally:
        hooks.uninstall()
    assert second.launches == 1
    assert any("undercounted" in row for row in runtime_profiler.UNCOUNTED_LAUNCH_SOURCES)


# --------------------------------------------------------- the hooks fail safe


@needs_monitoring
def test_a_failing_copy_hook_turns_itself_off_and_the_copy_still_happens_once() -> None:
    cp = _cupy14_shaped()
    notes: list[tuple] = []
    original_copyto = cp.copyto

    class Unmeasurable(np.ndarray):
        @property
        def nbytes(self):
            raise RuntimeError("nbytes unreadable")

    counters = LaunchCounters()
    hooks = CupyHooks(cp, counters, fault_sink=lambda hook, note, unavailable: notes.append(
        (hook, note, unavailable))).install()
    try:
        dst = np.zeros(4, np.float32).view(cp.ndarray)
        cp.copyto(dst, np.arange(4, dtype=np.float32).view(Unmeasurable))
        assert np.array_equal(dst, np.arange(4, dtype=np.float32))
        assert cp.copies == ["copyto"]
        assert notes == [("copyto", "RuntimeError: nbytes unreadable", False)]
        assert cp.copyto is original_copyto  # CuPy's own function is back
        assert "copyto" not in hooks.active and "asarray" in hooks.active
        # An error CuPy itself raises is CuPy's to raise, and no hook fault.
        with pytest.raises(TypeError):
            cp.asarray(np.zeros(3), dtype="not a dtype")
        assert "asarray" in hooks.active and set(hooks.faults) == {"copyto"}
        cp.asarray(np.zeros(3, np.float32))
        # The copy whose hook failed tallied nothing; the upload CuPy refused
        # was seen before CuPy refused it.
        assert counters.snapshot() == (0, 0, 0, 2, 24 + 12, 0, 0)
    finally:
        hooks.uninstall()
    assert cp.copyto is original_copyto


@needs_monitoring
def test_a_failing_launch_count_turns_kernel_counting_off_and_the_run_goes_on() -> None:
    cp = _cupy14_shaped()

    class Tally(LaunchCounters):
        broken = False

        @property
        def launches(self):
            return 0

        @launches.setter
        def launches(self, value):
            if type(self).broken:
                raise RuntimeError("the launch tally cannot be stored")

    sink = io.StringIO()
    profiler = RuntimeProfiler("summary", cupy=cp, progress_interval_s=0.0, progress_stream=sink,
                               counters=Tally(), nvml=False).install_hooks()
    try:
        kernel = cp.cuda.function.Module().get_function("gm1_kernel")
        Tally.broken = True
        kernel((1,), (1,), ())
        kernel((1,), (1,), ())
        assert [launch[1] for launch in cp.launches] == ["gm1_kernel", "gm1_kernel"]
        assert runtime_profiler._MONITOR.tool is None  # the event tool was released
        record = profiler.record()
    finally:
        profiler.uninstall_hooks()
    assert record["instrumentation"]["disabled"] == {
        "kernel-launches": "RuntimeError: the launch tally cannot be stored",
    }
    assert record["instrumentation"]["kernel_launches_counted"] is False
    assert not any("Function kernels" in row for row in record["counted_launch_sources"])
    assert any("Function kernels" in row for row in record["uncounted_launch_sources"])
    assert "gm1 profiler: kernel-launches turned off after RuntimeError" in sink.getvalue()
    assert "kernel-launches was turned off during the run" in format_performance(record)


def test_without_sys_monitoring_kernel_launches_are_named_uncounted_and_the_verdict_says_so(monkeypatch) -> None:
    cp = _cupy14_shaped()
    monkeypatch.setattr(runtime_profiler, "_monitoring", lambda: None)
    profiler = RuntimeProfiler("summary", cupy=cp, progress_interval_s=0.0, progress_stream=io.StringIO(),
                               counters=LaunchCounters(), nvml=False).install_hooks()
    try:
        cp.full((2,), 1.0)
        cp.cuda.function.Module().get_function("gm1_kernel")((1,), (1,), ())
        cp.copyto(np.zeros(2).view(cp.ndarray), np.ones(2).view(cp.ndarray))
        assert profiler.counters.launches == 2  # cupy.full's fill and the copy, both through copyto
        instrumentation = profiler.instrumentation()
    finally:
        profiler.uninstall_hooks()
    assert "Python 3.12" in instrumentation["unavailable"]["kernel-launches"]
    assert instrumentation["kernel_launches_counted"] is False
    assert "kernel-launches" not in instrumentation["hooks_installed"]
    steps = {"count": 10, "mean_s": 0.72, "sync_wait_mean_s": 0.01, "launches_per_step": 2500}
    verdict = performance_verdict({"steps": steps, "instrumentation": instrumentation})
    assert verdict["label"] == "host-bound"
    assert "kernel launches were not counted" in verdict["line"]
    assert performance_verdict({"steps": steps})["label"] == "host-launch-bound"


@needs_monitoring
def test_a_taken_monitoring_tool_id_leaves_kernel_launches_uncounted(monkeypatch) -> None:
    monitoring = sys.monitoring
    tool = 4
    if monitoring.get_tool(tool) is not None:
        pytest.skip("sys.monitoring tool id 4 is already in use in this interpreter")
    monkeypatch.setattr(runtime_profiler, "_MONITOR_TOOL_IDS", (tool,))
    monitoring.use_tool_id(tool, "another tool")
    try:
        hooks = CupyHooks(_cupy14_shaped(), LaunchCounters()).install()
        hooks.uninstall()
    finally:
        monitoring.free_tool_id(tool)
    assert "in use" in hooks.unavailable["kernel-launches"]


# ---------------------------------------------------- the profiler fails safe


class _Device:
    def __init__(self):
        self.host = 0.0

    def clock(self):
        return self.host


def test_device_marker_and_nvtx_failures_turn_those_off_and_every_step_is_still_timed() -> None:
    device = _Device()

    class Marker:
        def record(self, stream=None):
            raise RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")

    def refuse(*_args):
        raise OSError("libnvToolsExt is gone")

    cupy = SimpleNamespace(cuda=SimpleNamespace(
        Event=Marker, get_elapsed_time=lambda start, end: 0.0,
        nvtx=SimpleNamespace(RangePush=refuse, RangePop=refuse),
    ))
    sink = io.StringIO()
    profiler = RuntimeProfiler("detailed", cupy=cupy, stream=object(), progress_interval_s=0.0,
                               progress_stream=sink, counters=LaunchCounters(), nvml=False,
                               clock=device.clock)
    for step in (1, 2, 3):
        profiler.step_begin()
        token = profiler.phase_begin("rk1.acoustic")
        device.host += 0.25
        profiler.phase_end(token)
        profiler.synchronized(0.0)
        profiler.step_end(step=step, model_time_s=float(step))
    record = profiler.record(wall_seconds=device.host)
    assert record["steps"]["count"] == 3
    assert record["phases"][0]["host_s"] == pytest.approx(0.75)
    assert record["gpu_event_timing"] is False and record["nvtx_ranges"] is False
    assert record["instrumentation"]["disabled"] == {
        "device-events": "RuntimeError: CUDA_ERROR_ILLEGAL_ADDRESS",
        "nvtx": "OSError: libnvToolsExt is gone",
    }
    assert record["instrumentation"]["kernel_launches_counted"] is None  # no hooks were asked for
    lines = sink.getvalue().splitlines()
    assert len(lines) == 2 and all(line.startswith("gm1 profiler: ") for line in lines)


def test_progress_io_phase_and_record_failures_never_reach_the_run(monkeypatch) -> None:
    device = _Device()
    cupy = SimpleNamespace(cuda=SimpleNamespace())

    class BrokenStream:
        def write(self, _text):
            raise RuntimeError("stderr went away")

        def flush(self):
            pass

    profiler = RuntimeProfiler("summary", cupy=cupy, progress_interval_s=1.0,
                               progress_stream=BrokenStream(), counters=LaunchCounters(),
                               nvml=False, clock=device.clock)
    profiler.step_begin()
    device.host += 2.0
    profiler.step_end(step=1, model_time_s=1.0)
    assert profiler.step_count == 1
    assert "progress" in profiler.faults and profiler.progress_interval_s == 0.0

    event = profiler.io_event("history", "frame-000001")
    event.add("arena.write", "not seconds")
    assert "io" in profiler.faults
    event.finish()
    assert profiler.io_event("restart", "restart-000001") is None

    profiler.counters = None  # phase bookkeeping now raises
    token = profiler.phase_begin("solve3")
    assert token is None and "phases" in profiler.faults
    profiler.phase_end(token)

    def broken_record(**_kwargs):
        raise KeyError("steps")

    monkeypatch.setattr(profiler, "_record", broken_record)
    record = profiler.record(wall_seconds=2.0)
    assert record["record_error"] == "KeyError: 'steps'"
    assert set(record["instrumentation"]["disabled"]) == {"progress", "io", "phases"}
    text = format_performance(record)
    assert "the performance record could not be built: KeyError: 'steps'" in text


@pytest.mark.parametrize("error", (
    BrokenPipeError(32, "Broken pipe"), ValueError("I/O operation on closed file"),
), ids=("broken-pipe", "closed-file"))
def test_a_lost_progress_stream_turns_the_progress_line_off_with_a_receipt_note(error) -> None:
    """A terminal that went away or a closed stderr is a fault like any other:
    the line is not retried and swallowed every interval."""

    device = _Device()
    writes: list[str] = []

    class LostStream:
        def write(self, text):
            if text.startswith("gm1 progress:"):
                raise error
            writes.append(text)

        def flush(self):
            pass

    profiler = RuntimeProfiler("summary", cupy=SimpleNamespace(cuda=SimpleNamespace()),
                               progress_interval_s=1.0, progress_stream=LostStream(),
                               counters=LaunchCounters(), nvml=False, clock=device.clock)
    for number in range(1, 4):
        profiler.step_begin()
        device.host += 2.0
        profiler.step_end(step=number, model_time_s=float(number))
    assert profiler.step_count == 3 and profiler.progress_lines == 0
    assert profiler.faults["progress"] == f"{type(error).__name__}: {error}"
    assert profiler.progress_interval_s == 0.0
    assert profiler.maybe_progress(step=4, model_time_s=4.0, force=True) is None
    # The one fail-safe note went to the same stream (it takes ordinary lines).
    assert [text for text in writes if text.startswith("gm1 profiler:")] == [
        f"gm1 profiler: progress turned off after {type(error).__name__}: {error}; the run continues"
    ]
    assert profiler.record()["instrumentation"]["disabled"] == {"progress": f"{type(error).__name__}: {error}"}


class _Marker:
    """``cupy.cuda.Event``: every marker ever made is kept in ``made``."""

    made: list = []
    done = True

    def __init__(self):
        type(self).made.append(self)

    def record(self, stream=None):
        return None


def _marker_profiler(device):
    marker = type("Marker", (_Marker,), {"made": []})
    cupy = SimpleNamespace(cuda=SimpleNamespace(Event=marker, get_elapsed_time=lambda start, end: 1.0))
    sink = io.StringIO()
    profiler = RuntimeProfiler("summary", cupy=cupy, stream=object(), progress_interval_s=0.0,
                               progress_stream=sink, counters=LaunchCounters(), nvml=False,
                               clock=device.clock)
    return profiler, marker, sink


class _RefusingPhases(dict):
    def get(self, *_args):
        raise RuntimeError("the phase table cannot be read")


@pytest.mark.parametrize(("instrument", "brake"), (
    ("steps", lambda profiler: setattr(profiler, "recent", None)),
    ("phases", lambda profiler: setattr(profiler, "phases", _RefusingPhases(profiler.phases))),
), ids=("steps", "phases"))
def test_a_steps_or_phases_fault_stops_the_markers_so_none_pile_up(instrument, brake) -> None:
    """Markers are resolved and pooled only at the end-of-step synchronization."""

    device = _Device()
    profiler, marker, sink = _marker_profiler(device)
    keys = ("rk1.pre-acoustic", "rk1.acoustic", "rk3.acoustic", "after_step")

    def step(number):
        profiler.step_begin()
        for key in keys:
            token = profiler.phase_begin(key)
            device.host += 0.01
            profiler.phase_end(token)
        profiler.synchronized(0.0)
        profiler.step_end(step=number, model_time_s=float(number))

    step(1)
    healthy = len(marker.made)
    assert healthy == 2 * len(keys) and profiler.gpu_events
    brake(profiler)
    for number in range(2, 1002):
        step(number)
        assert len(profiler._pending) <= len(keys)
    assert set(profiler.faults) == {instrument, "device-events"}
    assert "device-events turned off after " + instrument + " was turned off" in sink.getvalue()
    assert profiler._pending == [] and profiler._pool == [] and profiler._reference is None
    assert len(marker.made) <= healthy + 2 * len(keys)
    record = profiler.record(wall_seconds=device.host)
    assert record["gpu_event_timing"] is False and record["unresolved_markers"] == 0
    if instrument == "steps":
        assert [row["calls"] for row in record["phases"]] == [1001] * len(keys)


def test_markers_past_the_pending_bound_are_dropped_when_no_synchronization_resolves_them(monkeypatch) -> None:
    monkeypatch.setattr(runtime_profiler, "_MAX_PENDING_MARKERS", 8)
    device = _Device()
    profiler, marker, _sink = _marker_profiler(device)
    for _ in range(100):
        profiler.phase_end(profiler.phase_begin("solve3"))
    assert len(profiler._pending) == 8 and profiler.unresolved_markers == 92
    assert len(marker.made) == 2 * 8 + 2  # a dropped pair goes back to the pool
    record = profiler.record()
    assert record["unresolved_markers"] == 92 and record["phases"][0]["calls"] == 100
    assert record["instrumentation"]["disabled"] == {}


def test_hooks_that_cannot_install_are_named_and_the_rest_still_count() -> None:
    cp = _cupy14_shaped()
    del cp.asnumpy
    cp.cuda.compiler = None
    counters = LaunchCounters()
    hooks = CupyHooks(cp, counters).install()
    try:
        cp.copyto(np.zeros(2).view(cp.ndarray), np.ones(2))
    finally:
        hooks.uninstall()
    assert set(hooks.unavailable) >= {"asnumpy", "nvrtc-compile"}
    assert counters.h2d_count == 1 and counters.launches == 1
    assert not hasattr(cp, "asnumpy")


@needs_monitoring
def test_launch_counts_on_the_host_closure_lose_nothing_to_disabled_call_sites(monkeypatch) -> None:
    """Every call site that ever launches a kernel is counted: DISABLE drops none of them."""

    import shutil

    if shutil.which("g++") is None:
        pytest.skip("the host kernel emulator needs g++")
    from tools.oracle import cartesian_profiling_identity as harness

    def launches(run):
        return {row["phase"]: row["launches"] for row in run["performance"]["phases"]}

    disabling = harness.run(level="summary", steps=2)

    class NeverDisable:
        def __getattr__(self, name):
            return getattr(sys.monitoring, name)

        DISABLE = None

    monkeypatch.setattr(runtime_profiler, "_monitoring", lambda: NeverDisable())
    exhaustive = harness.run(level="summary", steps=2)
    assert launches(disabling) == launches(exhaustive)
    assert disabling["performance"]["totals"]["launches"] > 100
    assert disabling["performance"]["instrumentation"]["kernel_launches_counted"] is True


# ----------------------------------------------- the real door: nothing is lost
#
# Two small decks that reach the call sites the closure harness above never
# does: a live, storm-following nest with child parcels, parent parcels and
# tornado diagnostics; and a flat run with Eulerian budgets, parcel vorticity
# budgets, circuits and high-frequency output.  (pdcomp is not here: its
# spectral solve takes complex pointers the host emulator does not compile.)
# The child's 0.25 s step keeps the moving frame's domain-location sum exact
# (cm1-forecast-publish refuses a child step of 1/6 s).

_DOOR_DECK_HEAD = """\
[run]
lane = "cartesian"
duration_s = 3.0
dt_s = 0.75
history_interval_s = 3.0
restart_interval_s = 3.0

[initialization]
isnd = 5
iwnd = 2

[grid]
nx = 24
ny = 24
nz = 20
dx_m = 600.0
dy_m = 600.0
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
upper_damping_depth_m = 3000.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = true
u_move_m_s = 12.5
v_move_m_s = 3.0

"""

_DOOR_DECK_TAIL = """\
[runtime]
acoustic_substeps = 6
tile_nx = 16
tile_ny = 16
warmup_steps = 0
memory_budget_mib = 2048
capacity_profile = "default"

[physics]
component = "wsm6-mp6"
equation_set = 2
"""

_NESTED_TABLES = """\
[nest]
refinement_ratio = 3
time_refinement_ratio = 3
parent_i0 = 8
parent_j0 = 8
parent_span_x = 8
parent_span_y = 8
spec_zone = 1
relax_zone = 4
feedback = 1
acoustic_substeps = 6
follow_storm = true
follow_interval_s = 1.5
follow_max_speed_m_s = 1000.0

[diagnostics]
output_uh = 1
output_vort = 1
output_pv = 1
output_sws = 1
output_svs = 1
output_sps = 1
output_srs = 1
output_sgs = 1
output_sus = 1
output_shs = 1
tornado_report = true

[lagrangian]
iprcl = 1
nparcels = 27
prclfrq_s = 0.75
seeding = "box"
seed_nx = 3
seed_ny = 3
seed_nz = 3
seed_dx_m = 600.0
seed_dy_m = 600.0
seed_dz_m = 500.0
seed_x0_m = 6000.0
seed_y0_m = 6000.0
seed_z0_m = 500.0
prcl_ptra = 0
prcl_nc = 0
prcl_dbz = 0
prcl_sfc = 0

[lagrangian.nest_child]
seed_frame = "parent-grid-origin"
nparcels = 18
prclfrq_s = 0.75
seed_nx = 3
seed_ny = 3
seed_nz = 2
seed_dx_m = 400.0
seed_dy_m = 400.0
seed_dz_m = 500.0
seed_x0_m = 6400.0
seed_y0_m = 6400.0
seed_z0_m = 500.0

"""

_BUDGET_TABLES = """\
[diagnostics]
output_uh = 1
output_vort = 1
output_pv = 1
output_sws = 1
output_svs = 1
output_sps = 1
output_srs = 1
output_sgs = 1
output_sus = 1
output_shs = 1
tornado_report = true

[output]
output_ubudget = 1
output_vbudget = 1
output_wbudget = 1
output_thbudget = 1
output_qvbudget = 1

[lagrangian]
iprcl = 1
nparcels = 43
prclfrq_s = 0.75
prcl_th = 1
prcl_t = 1
prcl_prs = 1
prcl_ptra = 0
prcl_q = 1
prcl_nc = 0
prcl_km = 1
prcl_kh = 1
prcl_tke = 1
prcl_dbz = 0
prcl_b = 1
prcl_vpg = 1
prcl_vort = 1
prcl_rho = 1
prcl_qsat = 1
prcl_sfc = 0
output_pdcomp = 0
dohifrq = true
hifrqfrq_s = 1.5
hifrq_i0 = 4
hifrq_i1 = 12
hifrq_j0 = 4
hifrq_j1 = 12
hifrq_k0 = 0
hifrq_k1 = 6
direction = "forward"
seeding = "box"
seed_nx = 3
seed_ny = 3
seed_nz = 3
seed_dx_m = 600.0
seed_dy_m = 600.0
seed_dz_m = 500.0
seed_x0_m = 6000.0
seed_y0_m = 6000.0
seed_z0_m = 500.0
circuits = 2
circuit_parcels = 8
circuit_radius_m = 1200.0
circuit_center_x_m = 7200.0
circuit_center_y_m = 7200.0
circuit_center_z_m = 500.0
circuit_spacing_z_m = 500.0
vorticity_budget = true
vorticity_budget_frq_s = 1.5
vorticity_budget_history = true
parcel_vorticity_budget = true

"""


@needs_monitoring
@pytest.mark.parametrize("tables", (_NESTED_TABLES, _BUDGET_TABLES), ids=("nest-parcels-diagnostics", "budgets-parcels-hifrq"))
def test_door_launch_counts_lose_nothing_to_disabled_call_sites(tmp_path, tables) -> None:
    """The shipped counts against never answering DISABLE, phase by phase, on the real door.

    The shipped run goes first, as a run's profiler does; the exhaustive run
    after it counts only if arming re-armed every site the first one disabled.
    Both share the CALL-event counter, so each run is also held to the host
    stand-in's own tally of every kernel ``__call__`` and every ``copyto`` it
    executed (``host_cupy.COUNTS``), which raises no event and sees a launch
    the hooks could not.
    """

    import json
    import os
    import shutil
    import subprocess

    if shutil.which("g++") is None:
        pytest.skip("the host kernel emulator needs g++")
    from test_step_speed_identity import _cargo_artifact

    library = _cargo_artifact("libcm1_state.so", "GM1_CM1_STATE_LIBRARY", ["--lib"])
    publisher = _cargo_artifact("cm1_forecast_publish", "GM1_FORECAST_PUBLISHER",
                                ["--bin", "cm1_forecast_publish"])
    deck = tmp_path / "deck.toml"
    deck.write_text(_DOOR_DECK_HEAD + tables + _DOOR_DECK_TAIL)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools/oracle/profiler_call_site_identity.py"), str(deck),
         "--library", str(library), "--publisher", str(publisher), "--workdir", str(tmp_path / "work")],
        capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert (tmp_path / "work" / "summary.json").is_file(), completed.stderr[-3000:]
    summary = json.loads((tmp_path / "work" / "summary.json").read_text())
    assert summary["status"] == {"disable": 0, "never_disable": 0}, completed.stderr[-3000:]
    assert summary["kernel_launches_counted"] is True
    assert summary["differing_phases"] == {}
    assert summary["identical"] is True and completed.returncode == 0
    assert summary["launches"]["disable"] > 4 * 1000
    # The independent ground truth: the hooks counted exactly the kernel
    # launches and copyto calls the stand-in executed, in both runs.
    assert summary["every_host_launch_counted"] is True
    for row in summary["host_ground_truth"].values():
        assert row["host_kernel_launches"] > 2000 and row["host_copyto_calls"] > 0
        assert row["hook_launches"] == row["host_kernel_launches"] + row["host_copyto_calls"]
        # Phases hold the launches made inside them; the few outside (startup,
        # publication) are the difference, and it is small.
        assert 0 <= row["hook_launches"] - summary["launches"]["disable"] < 100
    if tables is _NESTED_TABLES:
        # The nest child, its parcels and the diagnostics all run after the parent's step.
        assert summary["launches_by_phase"]["after_step"] > summary["launches"]["disable"] / 2


# -------------------------------------------------------------- on the device
#
# On a GPU host, holding its GPU mutex (the preamble -- $GPU
# from nvidia-smi, the host's $LOCK, a Python >= 3.12 and a CuPy 14 assertion
# -- and the door and overhead checks that follow this test are
# docs/GM1-CARTESIAN-RUNTIME.md, "Verifying the profiler on a device").  With
# an interpreter without sys.monitoring the test is skipped, not run:
#
#   $PY -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
#   CM1_CUDA_TESTS=1 PYTHONPATH=$TREE/src $PY tools/perf_probe/with_gpu_mutex.py $GPU $LOCK -- \
#     $PY -m pytest -p no:cacheprovider -rfEs -v \
#     "tests/test_runtime_profiler_failsafe.py::test_device_cupy_builds_its_own_kernels_under_the_hooks_and_gm1_kernels_count"

_PROBE_SOURCE = r'''
extern "C" __global__ void gm1_profiler_probe(float* x, const int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { x[i] += 1.0f; }
}
'''


def _device_cupy():
    import os

    if os.environ.get("CM1_CUDA_TESTS") != "1":
        pytest.skip("requires coordinated GPU mutex (CM1_CUDA_TESTS=1)")
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as error:  # pragma: no cover - only without a driver
        pytest.skip(f"CUDA unavailable: {error}")
    return cupy


@needs_monitoring
def test_device_cupy_builds_its_own_kernels_under_the_hooks_and_gm1_kernels_count() -> None:
    """The RTX 4090 failure, on the device: CuPy's own kernels, then GM1's two kinds."""

    import time

    cupy = _device_cupy()
    import cupy.cuda.compiler as compiler
    import cupy.cuda.function as function

    stream = cupy.cuda.Stream(non_blocking=True)
    sink = io.StringIO()
    profiler = RuntimeProfiler("detailed", cupy=cupy, stream=stream, progress_interval_s=0.0,
                               progress_stream=sink, counters=LaunchCounters(), nvml=False).install_hooks()
    try:
        with stream:
            profiler.step_begin()
            token = profiler.phase_begin("probe")
            counters = profiler.counters
            before_fill = counters.launches
            x = cupy.full((4096,), 2.0, dtype=cupy.float32)  # integration: the TypeError
            # CuPy 14.2's full is Python and fills through cupy.copyto
            # (cupy/_creation/basic.py:329), which the copy hook counts; any
            # further Python-level copy CuPy makes on first touch counts too,
            # so this is a floor, not an exact count.
            assert counters.launches - before_fill >= 1
            x *= cupy.float32(1.5)
            total = cupy.sum(x)
            count = cupy.int32(x.size)
            raw = cupy.RawModule(code=_PROBE_SOURCE, options=("--std=c++14",)).get_function(
                "gm1_profiler_probe")
            nvrtc_major = int(tuple(compiler._get_nvrtc_version())[0])
            options = ("--std=c++14", "--fmad=false") + (
                ("--device-as-default-execution-space",) if nvrtc_major >= 12 else ()
            ) + (f"--gpu-architecture=sm_{compiler._get_arch()}",)
            cubin, _ = compiler._NVRTCProgram(
                _PROBE_SOURCE, "gm1_profiler_probe_cubin.cu", method="cubin").compile(options)
            module = function.Module()
            module.load(cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin))
            image_kernel = module.get_function("gm1_profiler_probe")
            # GM1's launches, and nothing else, between these two snapshots:
            # exactly three, whatever CuPy did to build its own kernels above.
            before_gm1 = counters.launches
            raw((8,), (512,), (x, count))
            raw.kernel((8,), (512,), (x, count))  # the Function behind the RawKernel
            image_kernel((8,), (512,), (x, count))
            assert counters.launches - before_gm1 == 3
            profiler.phase_end(token)
            started = time.perf_counter()
            stream.synchronize()
            profiler.synchronized(time.perf_counter() - started)
            profiler.step_end(step=1, model_time_s=1.0)
        record = profiler.record(wall_seconds=1.0)
        assert type(image_kernel) is function.Function and type(raw) is cupy.RawKernel
    finally:
        profiler.uninstall_hooks()
    assert float(total.get()) == pytest.approx(4096 * 3.0)
    assert np.array_equal(cupy.asnumpy(x), np.full(4096, 6.0, np.float32))
    assert record["instrumentation"] == {
        "hooks_installed": ["kernel-launches", "nvrtc-compile", "copyto", "asnumpy", "asarray", "array"],
        "kernel_launches_counted": True, "unavailable": {}, "disabled": {},
    }
    (row,) = record["phases"]
    # The three GM1-shaped launches and at least cupy.full's fill (asserted
    # exactly above, around the calls themselves).  The scale and the sum
    # launch inside Cython and are not counted.
    assert row["launches"] >= 3 + 1
    assert record["gpu_event_timing"] is True and row["gpu_s"] is not None
    assert record["unresolved_markers"] == 0 and record["nvtx_ranges"] is True
    assert "nvrtc:gm1_profiler_probe_cubin.cu" in {row["program"] for row in record["compile"]["rows"]}
    assert sink.getvalue() == ""
