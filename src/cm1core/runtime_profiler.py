"""Low-overhead performance profiling for the Cartesian storm lane.

The per-phase host wall clock the runtime has always recorded measures only
how long Python took to *enqueue* a phase.  CuPy launches are asynchronous, so
device work is booked to whichever phase next synchronizes: a microphysics
wait shows up as "rk1.pre-acoustic" host time on the following step.  That
record cannot say whether a run is limited by the device (compute or memory
bandwidth), by the host launching kernels, or by I/O.

This module measures the three separately and never changes a model word:

* **device time per phase** from a pair of ``cupy.cuda.Event`` markers
  recorded on the run's own stream around each phase.  Recording a marker
  enqueues nothing that reads or writes a carrier and forces no
  synchronization; the pairs are resolved with ``get_elapsed_time`` only after
  the step's existing end-of-step ``stream.synchronize()`` has returned, when
  every marker before it has already executed.
* **queue lag** at the end of each phase: how far the device timeline was
  behind the host when the host finished enqueuing that phase.  A reference
  marker is recorded immediately after every synchronization (the queue is
  empty then), so ``elapsed(reference, phase_end) - host_offset`` is the lag.
  Together with the end-of-step synchronization wait it separates "the device
  could not keep up" (large lag, long wait) from "the device was waiting for
  the host" (lag near zero).
* **host enqueue time, kernel launches and copies per phase**.  Kernel
  launches are counted where Python code calls a ``cupy.cuda.function.Function``
  or ``cupy.RawKernel`` object, through ``sys.monitoring`` CALL events (Python
  3.12+); no CuPy class or kernel handle is replaced, because CuPy's Cython
  code refuses substitutes for the objects its factories return.  Copies are
  counted by forwarding functions in place of ``copyto``, ``asnumpy`` and
  ``asarray``/``array`` that return exactly CuPy's result; CuPy's own Python
  code reaches ``copyto`` the same way, so ``cupy.full`` and
  ``cupy.full_like`` count one launch each.  CuPy's built-in ufunc and
  elementwise launches and ``ndarray.get``/``set`` are not counted; the
  receipt names the counted sources.
* **I/O stages** of every history frame and restart: device staging,
  device-to-host copy, hashing, writing, fsync, the Rust publisher (with its
  own read/encode/hash/write/fsync breakdown), receipts and renames.
* **startup**: the Rust initializer, runtime construction, every NVRTC
  compilation by program name, and the first step against the steady mean.

Profiling never stops a run: an instrument that raises is turned off, named
on stderr and listed under ``instrumentation.disabled`` in the receipt.

Levels: ``off`` installs nothing (the runtime keeps ``profiler = None`` and
takes exactly the code path it always took), ``summary`` (the default)
records all of the above in aggregate, and ``detailed`` adds per-carrier I/O
rows and NVTX ranges around steps, phases and I/O stages for ``nsys``.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from functools import wraps
import importlib
import math
import os
import sys
import time
from typing import Any, Iterator


PERFORMANCE_SCHEMA = "gm1.cartesian-performance/v1"
PROFILE_LEVELS = ("off", "summary", "detailed")
DEFAULT_PROFILE_LEVEL = "summary"
DEFAULT_PROGRESS_INTERVAL_S = 30.0
# The environment variable cm1-forecast-publish reads to print its own stage
# timings on stderr.  Its stdout -- which history and restart receipts embed
# as ``rust_output`` -- is unchanged by it.
PUBLISH_TIMING_ENV = "GPUWM_CM1_PUBLISH_TIMING"
PUBLISH_TIMING_PREFIX = "gm1-publish-timing"

KERNEL_LAUNCH_HOOK = "kernel-launches"
COUNTED_LAUNCH_SOURCES = (
    "calls Python code makes to cupy.cuda.function.Function kernels (Module.get_function)",
    "calls Python code makes to cupy.RawKernel kernels (RawModule.get_function, RawKernel)",
    # CuPy's own Python code calls copyto through the module attribute:
    # cupy/_creation/basic.py:329,374 (full, full_like),
    # cupy/_indexing/indexing.py:218,223 (select), cupy/_math/trigonometric.py:156,159 (unwrap).
    "cupy.copyto, including the fills cupy.full and cupy.full_like make through it "
    "(and cupy.select, cupy.unwrap)",
)
UNCOUNTED_LAUNCH_SOURCES = (
    "CuPy ufunc/elementwise/reduction launches (a + b, cupy.where, ndarray.fill, cupy.sum, ...)",
    "ndarray.get/set/item and implicit host reads",
    "kernel launches made from C or Cython rather than Python bytecode",
    "a Python call site whose callable was not a kernel the first time it ran and is a kernel "
    "later (that site stops raising events, so its launches are undercounted)",
)

# Verdict thresholds.  The synchronization wait at the end of a step is the
# device work still queued when the host finished enqueuing it: a quarter of
# the step or more means the device, not the host, sets the pace.
GPU_BOUND_BACKLOG_FRACTION = 0.25
HOST_BOUND_BACKLOG_FRACTION = 0.10
HIGH_LAUNCHES_PER_STEP = 100
IO_HEAVY_SHARE = 0.20

# Receipt-size bounds: nothing retained grows with the step count.
_MAX_COMPILE_ROWS = 256
# Stream-marker pairs awaiting the end-of-step synchronization.  A step records
# a few dozen; a pair past this bound is dropped and counted as unresolved, so
# markers cannot accumulate if synchronizations stop arriving.
_MAX_PENDING_MARKERS = 1 << 14
_MAX_CARRIER_ROWS = 512
_RECENT_STEPS = 32
_MAX_FAULT_TEXT = 400

COUNTER_FIELDS = (
    "launches", "d2d_count", "d2d_bytes", "h2d_count", "h2d_bytes",
    "d2h_count", "d2h_bytes",
)


def require_profile_level(value: object) -> str:
    """The admitted profiling level, or a ValueError naming the admitted set."""

    if not isinstance(value, str) or value not in PROFILE_LEVELS:
        raise ValueError(
            f"profiling level {value!r} is not one of {', '.join(PROFILE_LEVELS)}"
        )
    return value


def require_progress_interval(value: object) -> float:
    """Seconds between live progress lines; zero disables them."""

    if isinstance(value, bool):
        raise ValueError("the progress interval must be a number of seconds")
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"the progress interval {value!r} is not a number of seconds"
        ) from None
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            f"the progress interval {value!r} must be a finite, nonnegative number of seconds"
        )
    return seconds


def _fault_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    if len(text) > _MAX_FAULT_TEXT:
        text = text[: _MAX_FAULT_TEXT - 3] + "..."
    return text


# --------------------------------------------------------------- counters


class LaunchCounters:
    """Process-wide launch and copy tallies the counting hooks increment."""

    __slots__ = COUNTER_FIELDS

    def __init__(self) -> None:
        for name in COUNTER_FIELDS:
            setattr(self, name, 0)

    def snapshot(self) -> tuple[int, ...]:
        return (
            self.launches, self.d2d_count, self.d2d_bytes, self.h2d_count,
            self.h2d_bytes, self.d2h_count, self.d2h_bytes,
        )


COUNTERS = LaunchCounters()


def _monitoring() -> Any:
    """``sys.monitoring`` (Python 3.12 and newer), or None where it is absent."""

    return getattr(sys, "monitoring", None)


# The sys.monitoring tool ids this profiler may take, in order: PROFILER_ID,
# then the two ids CPython leaves unnamed.  The debugger (0), coverage (1) and
# optimizer (5) ids are left to the tools they are named for.
_MONITOR_TOOL_IDS = (2, 3, 4)
_MONITOR_TOOL_NAME = "gm1-runtime-profiler"
# A call site is (code object, bytecode offset); offsets stay below 2**24.
_SITE_SHIFT = 24


class _KernelCallMonitor:
    """Count kernel launches where Python code makes them, substituting nothing.

    CuPy's kernel handles are Cython extension types.  Their ``__call__``
    cannot be patched, and CuPy's own Cython code refuses any stand-in for the
    objects its factories return: ``cdef function.Function kern =
    module.get_function(name)`` (``cupy/_core/_kernel.pyx:113``,
    ``cupy/_core/raw.pyx:129``) raises ``TypeError: Cannot convert ... to
    cupy.cuda.function.Function``, and with CuPy 14.2 that is reached from
    ``cupy.full`` the moment ``cupy.cuda.function.Module`` returns anything
    else.  So nothing is wrapped: a ``sys.monitoring`` CALL event fires before
    every call Python bytecode makes and names the callable, and a call to a
    kernel object is one launch.  CuPy's own launches run inside Cython,
    raise no event and stay uncounted, as they always were.

    A call site whose callable is not a kernel answers ``DISABLE`` and never
    raises the event again, so the callback keeps running only at sites that
    launch kernels.  The instrumentation itself is process-wide: CALL events
    are set for the whole interpreter, so every code object that runs while
    they are set has its CALL instructions instrumented on its first
    execution, and every call site pays one callback before it disables
    itself.  On CPython 3.12 a disabled site is de-instrumented at once and
    re-specializes (``dis.dis(adaptive=True)`` shows ``CALL`` return to
    ``CALL_PY_EXACT_ARGS`` after the usual warmup), so the steady cost is one
    callback per launch plus a one-time cost per site; on a call-dense
    pure-Python loop here it is within timing noise, and the whole-run number
    on a device is what ``tools/profile_cartesian_run.py --identity`` reports
    as ``profiling_overhead`` (the profiled step wall against ``--profile
    off``).  A site that has launched a kernel is never disabled, even if it
    later calls something else; a site whose first callable was not a kernel
    and that launches one later is not counted (the receipt names that source
    as uncounted).

    ``DISABLE`` can outlive the tool id (on CPython 3.12 a site that does not
    run while the tool is off stays disabled when the id is taken again), and
    ``sys.monitoring.restart_events`` is the only way to re-arm it.  That call
    re-arms every tool's disabled events, not just this one's, so arming makes
    it only when an earlier arming in this process disabled a site: a run's
    first profiler never touches another tool's state, and a later one counts
    again at sites an earlier one disabled.
    """

    def __init__(self) -> None:
        self.tool: int | None = None
        self.entries: tuple[tuple[frozenset[type], tuple[type, ...], LaunchCounters, Any], ...] = ()
        self.sites: set[int] = set()
        # True once a callback has answered DISABLE since events were last restarted.
        self.disabled = False

    def add(self, owner: Any, kinds: tuple[type, ...], counters: LaunchCounters) -> str | None:
        """Count launches of ``kinds`` into ``counters``; the reason it cannot, or None."""

        monitoring = _monitoring()
        if monitoring is None:
            return "sys.monitoring needs Python 3.12 or newer; kernel launches are not counted"
        acquired = False
        if self.tool is None:
            for tool in _MONITOR_TOOL_IDS:
                try:
                    monitoring.use_tool_id(tool, _MONITOR_TOOL_NAME)
                except ValueError:
                    continue
                self.tool, acquired = tool, True
                break
            else:
                return (
                    "every sys.monitoring tool id this profiler may take "
                    f"({', '.join(map(str, _MONITOR_TOOL_IDS))}) is in use; "
                    "kernel launches are not counted"
                )
        previous = self.entries
        self.entries = tuple(entry for entry in previous if entry[3] is not owner) + (
            (frozenset(kinds), tuple(kinds), counters, owner),
        )
        try:
            if acquired:
                self.sites = set()
                if self.disabled:
                    monitoring.restart_events()
                    self.disabled = False
            monitoring.register_callback(self.tool, monitoring.events.CALL, self._callback(monitoring))
            if acquired:
                monitoring.set_events(self.tool, monitoring.events.CALL)
        except Exception:
            self.entries = previous
            if not previous:
                self._release(monitoring)
            raise
        return None

    def remove(self, owner: Any) -> None:
        entries = tuple(entry for entry in self.entries if entry[3] is not owner)
        if len(entries) == len(self.entries):
            return
        self.entries = entries
        monitoring = _monitoring()
        if self.tool is None or monitoring is None:
            return
        if entries:
            monitoring.register_callback(self.tool, monitoring.events.CALL, self._callback(monitoring))
        else:
            self._release(monitoring)

    def _release(self, monitoring: Any) -> None:
        tool, self.tool = self.tool, None
        self.sites = set()
        if tool is None or monitoring is None:
            return
        for release in (
            lambda: monitoring.set_events(tool, 0),
            lambda: monitoring.register_callback(tool, monitoring.events.CALL, None),
            lambda: monitoring.free_tool_id(tool),
        ):
            try:
                release()
            except Exception:  # noqa: BLE001 - releasing is best effort
                pass

    def fault(self, error: BaseException) -> None:
        """A failure inside the event callback: stop counting and tell every owner."""

        owners = [entry[3] for entry in self.entries]
        self.entries = ()
        try:
            self._release(_monitoring())
        except Exception:  # noqa: BLE001
            pass
        for owner in owners:
            try:
                owner._fault(KERNEL_LAUNCH_HOOK, error)
            except Exception:  # noqa: BLE001
                pass

    def _callback(self, monitoring: Any) -> Any:
        entries = self.entries
        sites = self.sites
        disable = monitoring.DISABLE
        shift = _SITE_SHIFT
        fault = self.fault
        monitor = self

        def on_call(code: Any, offset: int, callable_: Any, arg0: Any) -> Any:
            try:
                cls = type(callable_)
                hit = False
                for exact, kinds, counters, _owner in entries:
                    if cls in exact or issubclass(cls, kinds):
                        counters.launches += 1
                        hit = True
                site = (id(code) << shift) + offset
                if hit:
                    sites.add(site)
                    return None
                if site in sites or disable is None:
                    return None
                monitor.disabled = True
                return disable
            except Exception as error:  # noqa: BLE001 - never into the launching code
                fault(error)
                return None

        return on_call


_MONITOR = _KernelCallMonitor()
_MISSING = object()


class _Switch:
    __slots__ = ("on",)

    def __init__(self) -> None:
        self.on = True


class CupyHooks:
    """Count launches and copies and time NVRTC compiles on one CuPy module.

    Nothing CuPy constructs or returns is replaced, and no CuPy class is
    substituted, because CuPy's Cython code type-checks what its own
    factories hand back (see :class:`_KernelCallMonitor`):

    * kernel launches are counted at the Python call sites that make them;
    * ``copyto``, ``asnumpy``, ``asarray`` and ``array`` are replaced by
      forwarding functions that return exactly what CuPy's function returned;
    * NVRTC compile time is measured by wrapping ``_NVRTCProgram.compile`` in
      place on its pure-Python class: the class, its instances and the
      compiled image are CuPy's own.

    Every hook fails safe.  An error in a hook's own bookkeeping turns that
    hook off, puts CuPy's attribute back, and is reported to ``fault_sink``
    as ``(hook, note, unavailable=False)``; the call it was observing goes
    ahead untouched.  A hook that cannot be installed is reported the same
    way with ``unavailable=True``.  An error CuPy itself raises propagates
    exactly as it would unprofiled.  ``uninstall`` restores every attribute
    that was replaced and stops all counting.
    """

    def __init__(self, cupy: Any, counters: LaunchCounters = COUNTERS,
                 compile_sink: Any = None, fault_sink: Any = None) -> None:
        self.cp = cupy
        self.counters = counters
        self.compile_sink = compile_sink
        self.fault_sink = fault_sink
        self._saved: list[tuple[str, Any, str, Any, Any]] = []
        self._switches: dict[str, _Switch] = {}
        self._installed = False
        self.unavailable: dict[str, str] = {}
        self.faults: dict[str, str] = {}
        self.kernel_types: tuple[type, ...] = ()

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def active(self) -> list[str]:
        """The hooks installed and still counting."""

        return [name for name, switch in self._switches.items() if switch.on]

    # ---- bookkeeping

    def _replace(self, hook: str, owner: Any, name: str, value: Any) -> None:
        namespace = getattr(owner, "__dict__", None)
        if namespace is not None and hasattr(namespace, "get"):
            original = namespace.get(name, _MISSING)
        else:
            original = getattr(owner, name)
        setattr(owner, name, value)
        self._saved.append((hook, owner, name, original, value))

    @staticmethod
    def _put_back(owner: Any, name: str, original: Any) -> None:
        if original is _MISSING:
            if name in getattr(owner, "__dict__", {}):
                delattr(owner, name)
        else:
            setattr(owner, name, original)

    def _restore(self, hook: str) -> None:
        kept = []
        for entry in self._saved:
            if entry[0] != hook:
                kept.append(entry)
                continue
            _, owner, name, original, value = entry
            try:
                # Only while the attribute is still this hook's replacement.
                if getattr(owner, name, _MISSING) is value:
                    self._put_back(owner, name, original)
            except Exception:  # noqa: BLE001 - restoring is best effort
                pass
        self._saved = kept

    def _compiled(self, label: str, seconds: float) -> None:
        sink = self.compile_sink
        if sink is not None:
            sink(label, seconds)

    def _report(self, hook: str, note: str, *, unavailable: bool) -> None:
        sink = self.fault_sink
        if sink is None:
            return
        try:
            sink(hook, note, unavailable=unavailable)
        except Exception:  # noqa: BLE001 - reporting never stops a run
            pass

    def _fault(self, hook: str, error: BaseException) -> None:
        """Turn ``hook`` off after an error in its own bookkeeping.  Never raises."""

        try:
            switch = self._switches.get(hook)
            if switch is not None:
                switch.on = False
            if hook in self.faults:
                return
            self.faults[hook] = note = _fault_text(error)
            if hook == KERNEL_LAUNCH_HOOK:
                _MONITOR.remove(self)
            self._restore(hook)
            self._report(hook, note, unavailable=False)
        except Exception:  # noqa: BLE001
            pass

    # ---- installation

    def install(self) -> "CupyHooks":
        if self._installed:
            return self
        self._installed = True
        for hook, installer in (
            (KERNEL_LAUNCH_HOOK, self._install_kernel_launches),
            ("nvrtc-compile", self._install_compile_timer),
            ("copyto", self._install_copyto),
            ("asnumpy", self._install_asnumpy),
            ("asarray", lambda: self._install_upload("asarray")),
            ("array", lambda: self._install_upload("array")),
        ):
            try:
                reason = installer()
            except Exception as error:  # noqa: BLE001 - a hook that cannot install is skipped
                self._restore(hook)
                self._switches.pop(hook, None)
                reason = f"could not be installed: {_fault_text(error)}"
            if reason is not None:
                self.unavailable[hook] = reason
                self._report(hook, reason, unavailable=True)
        return self

    def _install_kernel_launches(self) -> str | None:
        cp = self.cp
        function = _submodule(cp, getattr(cp, "cuda", None), "function")
        kinds: list[type] = []
        for candidate in (getattr(function, "Function", None), getattr(cp, "RawKernel", None)):
            if isinstance(candidate, type) and candidate not in kinds:
                kinds.append(candidate)
        if not kinds:
            return (
                "neither cupy.cuda.function.Function nor cupy.RawKernel is a type here; "
                "kernel launches are not counted"
            )
        switch = _Switch()
        reason = _MONITOR.add(self, tuple(kinds), self.counters)
        if reason is None:
            self.kernel_types = tuple(kinds)
            self._switches[KERNEL_LAUNCH_HOOK] = switch
        return reason

    def _install_compile_timer(self) -> str | None:
        cp = self.cp
        compiler = _submodule(cp, getattr(cp, "cuda", None), "compiler")
        program = getattr(compiler, "_NVRTCProgram", None) if compiler is not None else None
        compile_ = getattr(program, "compile", None) if isinstance(program, type) else None
        if not callable(compile_):
            return "cupy.cuda.compiler._NVRTCProgram.compile is absent; NVRTC compiles are not timed"
        switch = _Switch()
        hooks = self
        clock = time.perf_counter

        @wraps(compile_)
        def compile(program_self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: A001
            if not switch.on:
                return compile_(program_self, *args, **kwargs)
            started = clock()
            try:
                return compile_(program_self, *args, **kwargs)
            finally:
                try:
                    hooks._compiled(
                        f"nvrtc:{getattr(program_self, 'name', None) or 'program'}",
                        clock() - started,
                    )
                except Exception as error:  # noqa: BLE001
                    hooks._fault("nvrtc-compile", error)

        self._replace("nvrtc-compile", program, "compile", compile)
        self._switches["nvrtc-compile"] = switch
        return None

    def _classifiers(self) -> tuple[Any, Any]:
        ndarray = getattr(self.cp, "ndarray", None)
        if not isinstance(ndarray, type):
            ndarray = None
        host_type = importlib.import_module("numpy").ndarray

        def is_device(value: Any) -> bool:
            return ndarray is not None and isinstance(value, ndarray)

        def is_host(value: Any) -> bool:
            return isinstance(value, host_type) and not is_device(value)

        return is_device, is_host

    def _install_copyto(self) -> str | None:
        copyto = getattr(self.cp, "copyto", None)
        if not callable(copyto):
            return "cupy.copyto is absent; its launches and copies are not counted"
        is_device, is_host = self._classifiers()
        counters, hooks, switch = self.counters, self, _Switch()

        def counting_copyto(*args: Any, **kwargs: Any) -> Any:
            if switch.on:
                try:
                    # Everything is read before anything is tallied, so a
                    # failure leaves the counters as they were.
                    dst = args[0] if args else kwargs.get("dst")
                    src = args[1] if len(args) > 1 else kwargs.get("src")
                    if is_device(src):
                        nbytes = int(getattr(dst, "nbytes", 0))
                        counters.d2d_count += 1
                        counters.d2d_bytes += nbytes
                    elif is_host(src):
                        nbytes = int(getattr(src, "nbytes", 0))
                        counters.h2d_count += 1
                        counters.h2d_bytes += nbytes
                    counters.launches += 1
                except Exception as error:  # noqa: BLE001
                    hooks._fault("copyto", error)
            return copyto(*args, **kwargs)

        self._replace("copyto", self.cp, "copyto", counting_copyto)
        self._switches["copyto"] = switch
        return None

    def _install_asnumpy(self) -> str | None:
        asnumpy = getattr(self.cp, "asnumpy", None)
        if not callable(asnumpy):
            return "cupy.asnumpy is absent; its device-to-host copies are not counted"
        is_device, _ = self._classifiers()
        counters, hooks, switch = self.counters, self, _Switch()

        def counting_asnumpy(*args: Any, **kwargs: Any) -> Any:
            if switch.on:
                try:
                    value = args[0] if args else kwargs.get("a")
                    if is_device(value):
                        nbytes = int(value.nbytes)
                        counters.d2h_count += 1
                        counters.d2h_bytes += nbytes
                except Exception as error:  # noqa: BLE001
                    hooks._fault("asnumpy", error)
            return asnumpy(*args, **kwargs)

        self._replace("asnumpy", self.cp, "asnumpy", counting_asnumpy)
        self._switches["asnumpy"] = switch
        return None

    def _install_upload(self, name: str) -> str | None:
        original = getattr(self.cp, name, None)
        if not callable(original):
            return f"cupy.{name} is absent; its host-to-device copies are not counted"
        _, is_host = self._classifiers()
        counters, hooks, switch = self.counters, self, _Switch()

        def counting_upload(*args: Any, **kwargs: Any) -> Any:
            if switch.on:
                try:
                    value = args[0] if args else kwargs.get("a", kwargs.get("obj"))
                    if is_host(value):
                        nbytes = int(value.nbytes)
                        counters.h2d_count += 1
                        counters.h2d_bytes += nbytes
                except Exception as error:  # noqa: BLE001
                    hooks._fault(name, error)
            return original(*args, **kwargs)

        self._replace(name, self.cp, name, counting_upload)
        self._switches[name] = switch
        return None

    def uninstall(self) -> None:
        """Stop every count and put back every replaced attribute.  Never raises."""

        for switch in self._switches.values():
            switch.on = False
        try:
            _MONITOR.remove(self)
        except Exception:  # noqa: BLE001
            pass
        while self._saved:
            _, owner, name, original, _value = self._saved.pop()
            try:
                self._put_back(owner, name, original)
            except Exception:  # noqa: BLE001
                pass
        self._switches = {}
        self._installed = False


def _submodule(cp: Any, cuda: Any, name: str) -> Any:
    value = getattr(cuda, name, None) if cuda is not None else None
    if value is not None:
        return value
    if getattr(cp, "__name__", None) != "cupy":
        return None
    try:
        return importlib.import_module(f"cupy.cuda.{name}")
    except ImportError:
        return None


# ------------------------------------------------------------ fail-safety


def _guarded(instrument: str, *, event: bool = False) -> Any:
    """Run a profiler entry point so that its failure can never stop the run.

    An exception inside the method turns ``instrument`` off (every later call
    returns None at once) and leaves a note in the receipt; the runtime code
    that called it carries on as if it were unprofiled.  ``event`` marks an
    :class:`IoEvent` method, whose profiler is ``self.profiler``.
    """

    def decorate(method: Any) -> Any:
        if event:
            @wraps(method)
            def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
                profiler = self.profiler
                if instrument in profiler._off:
                    return None
                try:
                    return method(self, *args, **kwargs)
                except Exception as error:  # noqa: BLE001 - profiling never stops a run
                    profiler._fault(instrument, error)
                    return None
        else:
            @wraps(method)
            def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
                if instrument in self._off:
                    return None
                try:
                    return method(self, *args, **kwargs)
                except Exception as error:  # noqa: BLE001 - profiling never stops a run
                    self._fault(instrument, error)
                    return None

        return guarded

    return decorate


# ------------------------------------------------------------------ stats


class PhaseStats:
    __slots__ = (
        "calls", "host_s", "gpu_s", "gpu_calls", "lag_sum_s", "lag_max_s",
        "lag_calls", "counters",
    )

    def __init__(self) -> None:
        self.calls = 0
        self.host_s = 0.0
        self.gpu_s = 0.0
        self.gpu_calls = 0
        self.lag_sum_s = 0.0
        self.lag_max_s = -math.inf
        self.lag_calls = 0
        self.counters = [0] * len(COUNTER_FIELDS)


def _mb_per_s(nbytes: int, seconds: float) -> float | None:
    if nbytes <= 0 or seconds <= 0.0:
        return None
    return nbytes / 1.0e6 / seconds


class IoEvent:
    """One history frame, restart or other publication, stage by stage."""

    def __init__(self, profiler: "RuntimeProfiler", kind: str, label: str) -> None:
        self.profiler = profiler
        self.kind = kind
        self.label = label
        self.stages: dict[str, list[float]] = {}
        self.carriers: list[dict[str, Any]] | None = [] if profiler.detailed else None
        self.nbytes = 0
        self.started = profiler.clock()
        self.seconds = 0.0
        # What the stepping thread paid.  Equal to ``seconds`` for a
        # publication that ran on it; for one committed on the background
        # worker, the snapshot plus whatever part of a join waited on it.
        self.blocking_seconds = 0.0
        self.background = False
        self.finished = False

    @property
    def detailed(self) -> bool:
        return self.carriers is not None

    @_guarded("io", event=True)
    def add(self, stage: str, seconds: float, nbytes: int = 0) -> None:
        row = self.stages.get(stage)
        if row is None:
            row = self.stages[stage] = [0.0, 0, 0]
        row[0] += float(seconds)
        row[1] += int(nbytes)
        row[2] += 1

    @_guarded("io", event=True)
    def carrier(self, name: str, nbytes: int, **stage_seconds: float) -> None:
        if self.carriers is None or len(self.carriers) >= _MAX_CARRIER_ROWS:
            return
        self.carriers.append({"carrier": name, "bytes": int(nbytes),
                              **{key: float(value) for key, value in stage_seconds.items()}})

    @contextmanager
    def stage(self, name: str, nbytes: int = 0) -> Iterator[None]:
        profiler = self.profiler
        profiler.range_push(f"io.{self.kind}.{name}")
        started = profiler.clock()
        try:
            yield
        finally:
            self.add(name, profiler.clock() - started, nbytes)
            profiler.range_pop()

    def begin(self, stage: str) -> float:
        """Open one timed sub-step (and its NVTX range when detailed)."""

        self.range_push(stage)
        return self.profiler.clock()

    @_guarded("io", event=True)
    def end(self, stage: str, started: float, nbytes: int = 0) -> None:
        """Close the sub-step :meth:`begin` opened."""

        self.add(stage, self.profiler.clock() - started, nbytes)
        self.range_pop()

    def range_push(self, name: str) -> None:
        """An NVTX range for one I/O sub-step (detailed profiling only)."""

        self.profiler.range_push(f"io.{self.kind}.{name}")

    def range_pop(self) -> None:
        self.profiler.range_pop()

    @_guarded("io", event=True)
    def set_bytes(self, nbytes: int) -> None:
        self.nbytes = max(self.nbytes, int(nbytes))

    @_guarded("io", event=True)
    def finish(self, *, seconds: float | None = None, blocking_seconds: float | None = None,
               background: bool = False) -> None:
        """Close the event; a background publication passes its own accounting."""

        if self.finished:
            return
        self.finished = True
        self.seconds = (
            self.profiler.clock() - self.started if seconds is None else float(seconds)
        )
        self.blocking_seconds = (
            self.seconds if blocking_seconds is None else float(blocking_seconds)
        )
        self.background = bool(background)
        self.profiler._io_finished(self)

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "seconds": self.seconds,
            "blocking_seconds": self.blocking_seconds,
            "background": self.background,
            "bytes": self.nbytes,
            "mb_per_s": _mb_per_s(self.nbytes, self.seconds),
            "stages": _stage_rows(self.stages),
            **({} if self.carriers is None else {"carriers": list(self.carriers)}),
        }


def _stage_rows(stages: dict[str, list[float]]) -> list[dict[str, Any]]:
    return [
        {"stage": name, "seconds": row[0], "bytes": int(row[1]), "calls": int(row[2]),
         "mb_per_s": _mb_per_s(int(row[1]), row[0])}
        for name, row in stages.items()
    ]


# --------------------------------------------------------------- progress


class _Nvml:
    """Optional NVML readings; any failure disables them silently."""

    def __init__(self, cupy: Any) -> None:
        self.handle = None
        self.module = None
        try:
            module = importlib.import_module("pynvml")
            module.nvmlInit()
            handle = None
            bus = None
            try:
                bus = cupy.cuda.Device().pci_bus_id
            except Exception:  # noqa: BLE001 - optional telemetry only
                bus = None
            if bus:
                handle = module.nvmlDeviceGetHandleByPciBusId(bus)
            if handle is None:
                return
            self.module, self.handle = module, handle
        except Exception:  # noqa: BLE001 - NVML is optional and never required
            self.module = self.handle = None

    def values(self) -> dict[str, Any] | None:
        if self.handle is None:
            return None
        try:
            m, h = self.module, self.handle
            util = m.nvmlDeviceGetUtilizationRates(h).gpu
            power = m.nvmlDeviceGetPowerUsage(h) / 1000.0
            memory = m.nvmlDeviceGetMemoryInfo(h)
            return {
                "utilization_percent": int(util),
                "power_w": float(power),
                "memory_used_bytes": int(memory.used),
                "memory_total_bytes": int(memory.total),
            }
        except Exception:  # noqa: BLE001
            self.handle = None
            return None

    def reading(self) -> str | None:
        values = self.values()
        return None if values is None else gpu_text(values)


def gpu_text(values: dict[str, Any]) -> str:
    return (
        f"GPU {values['utilization_percent']}% {values['power_w']:.0f}W "
        f"{values['memory_used_bytes'] / 2**30:.1f}/"
        f"{values['memory_total_bytes'] / 2**30:.1f}GiB"
    )


def progress_fields(
    *, step: int, total_steps: int | None, model_time_s: float,
    duration_s: float | None, model_start_s: float | None, wall_s: float,
    mean_s: float | None, recent_s: float | None,
) -> dict[str, Any]:
    """Step, rate and ETA arithmetic shared by every progress reporter.

    ``wall_s`` is the wall time since stepping started and ``model_start_s``
    the model time it started from, so a resumed run's rate is the rate of
    the part this process integrated.  The ETA extrapolates that rate to
    ``duration_s``; with no duration it uses the recent step time and the
    remaining step count.  Absent inputs give ``None``, never a guess.
    """

    simulated = None if model_start_s is None else model_time_s - model_start_s
    rate = None
    if simulated is not None and wall_s > 0.0:
        rate = simulated / wall_s
    eta = None
    if duration_s is not None and simulated is not None and simulated > 0.0 and wall_s > 0.0:
        eta = max(0.0, duration_s - model_time_s) / (simulated / wall_s)
    elif total_steps is not None and recent_s is not None:
        eta = max(0, total_steps - step) * recent_s
    return {
        "step": int(step),
        "total_steps": None if total_steps is None else int(total_steps),
        "model_time_s": float(model_time_s),
        "duration_s": None if duration_s is None else float(duration_s),
        "s_per_step_recent": recent_s,
        "s_per_step_mean": mean_s,
        "model_s_per_wall_s": rate,
        "eta_s": eta,
    }


def io_summary(event: "IoEvent") -> dict[str, Any]:
    """The last finished I/O event, as the progress record names it."""

    return {
        "kind": event.kind,
        "label": event.label,
        "seconds": float(event.seconds),
        "bytes": int(event.nbytes),
        "background": bool(event.background),
        "blocking_seconds": float(event.blocking_seconds),
    }


def format_progress_line(fields: dict[str, Any]) -> str:
    """The ``gm1 progress:`` stderr line for one :func:`progress_fields` record."""

    parts = []
    total_steps, duration_s = fields["total_steps"], fields["duration_s"]
    target_steps = "" if total_steps is None else f"/{total_steps}"
    target_time = "" if duration_s is None else f"/{duration_s:g}s"
    parts.append(
        f"step {fields['step']}{target_steps} t={fields['model_time_s']:g}s{target_time}"
    )
    if fields["s_per_step_mean"] is not None:
        parts.append(
            f"{fields['s_per_step_recent']:.3f} s/step recent, "
            f"{fields['s_per_step_mean']:.3f} mean"
        )
    if fields["model_s_per_wall_s"] is not None:
        parts.append(f"{fields['model_s_per_wall_s']:.3f} model-s/wall-s")
    if fields["eta_s"] is not None:
        parts.append(f"ETA {_duration_text(fields['eta_s'])}")
    io = fields.get("last_io")
    if io is not None:
        parts.append(
            f"last io {io['kind']} {io['seconds']:.1f}s {_bytes_text(io['bytes'])}"
            + (f" ({io['blocking_seconds']:.1f}s blocking)" if io["background"] else "")
        )
    gpu = fields.get("gpu")
    if gpu is not None:
        parts.append(gpu_text(gpu))
    return "gm1 progress: " + " | ".join(parts)


def _duration_text(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _bytes_text(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{nbytes} B"


# --------------------------------------------------------------- profiler


class RuntimeProfiler:
    """Per-phase device/host timing, launch counts, I/O stages and startup.

    Profiling is fail-safe.  Each instrument -- the CuPy counting hooks
    (``kernel-launches``, ``nvrtc-compile``, ``copyto``, ``asnumpy``,
    ``asarray``, ``array``), ``device-events``, ``nvtx``, ``progress``,
    ``phases``, ``steps``, ``io`` and ``startup`` -- that raises is turned off
    for the rest of the run, named once on stderr, and recorded under
    ``instrumentation.disabled`` in the receipt.  The run is never stopped by
    its profiler.
    """

    def __init__(self, level: str = DEFAULT_PROFILE_LEVEL, *, cupy: Any = None,
                 stream: Any = None,
                 progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
                 progress_stream: Any = None, counters: LaunchCounters = COUNTERS,
                 nvml: bool = True, clock: Any = None) -> None:
        self.level = require_profile_level(level)
        # Instruments turned off after a failure, with the note that says why,
        # and hooks that could not be installed at all.
        self._off: set[str] = set()
        self.faults: dict[str, str] = {}
        self.unavailable: dict[str, str] = {}
        self.hooks_installed: list[str] | None = None
        # The host clock; injectable so the timing arithmetic can be tested
        # against a simulated device queue.
        self.clock = time.perf_counter if clock is None else clock
        self.enabled = self.level != "off"
        self.detailed = self.level == "detailed"
        self.cp = cupy
        self.stream = stream
        self.counters = counters
        self.progress_interval_s = require_progress_interval(progress_interval_s)
        self._progress_stream = progress_stream
        self._nvml_wanted = bool(nvml)
        self._nvml: _Nvml | None = None
        self.hooks: CupyHooks | None = None
        cuda = getattr(cupy, "cuda", None)
        self._event_type = getattr(cuda, "Event", None) if self.enabled else None
        self._elapsed = getattr(cuda, "get_elapsed_time", None) if self.enabled else None
        self.gpu_events = self._event_type is not None and self._elapsed is not None
        self._nvtx = _submodule(cupy, cuda, "nvtx") if self.detailed else None
        if self._nvtx is not None and not hasattr(self._nvtx, "RangePush"):
            self._nvtx = None
        self._nvtx_stage: str | None = None
        self._pool: list[Any] = []
        self._pending: list[tuple[PhaseStats, Any, Any, float]] = []
        self._reference: tuple[Any, float] | None = None
        self.unresolved_markers = 0
        self.phases: dict[str, PhaseStats] = {}
        self.phase_order: list[str] = []
        self.startup: dict[str, dict[str, Any]] = {}
        self.compile_rows: dict[str, list[float]] = {}
        self.compile_dropped = 0
        self.io: dict[str, dict[str, Any]] = {}
        self.last_io: IoEvent | None = None
        self.profiler_calls = 0
        self._reset_steps()
        self.first_step_s: float | None = None
        self.duration_s: float | None = None
        self.total_steps: int | None = None
        self._progress_last = self.clock()
        self._run_started = self.clock()
        self._model_start_s: float | None = None
        self.progress_lines = 0

    # ---- lifecycle

    def install_hooks(self) -> "RuntimeProfiler":
        if not (self.enabled and self.cp is not None and self.hooks is None):
            return self
        hooks = CupyHooks(self.cp, self.counters, compile_sink=self.compiled,
                          fault_sink=self._hook_note)
        self.hooks = hooks
        try:
            hooks.install()
            self.hooks_installed = hooks.active
        except Exception as error:  # noqa: BLE001 - a run without counts is still a run
            hooks.uninstall()
            self.hooks = None
            self.hooks_installed = []
            self._fault("cupy-hooks", error)
        return self

    def uninstall_hooks(self) -> None:
        hooks, self.hooks = self.hooks, None
        if hooks is not None:
            hooks.uninstall()

    def bind_stream(self, stream: Any) -> None:
        self.stream = stream

    def _hook_note(self, hook: str, note: str, *, unavailable: bool) -> None:
        if unavailable:
            self.unavailable.setdefault(hook, note)
        else:
            self._disable(hook, note)

    def _fault(self, instrument: str, error: BaseException) -> None:
        """Turn ``instrument`` off after ``error``.  Never raises."""

        try:
            self._disable(instrument, _fault_text(error))
        except Exception:  # noqa: BLE001
            self._off.add(instrument)

    def _disable(self, instrument: str, note: str) -> None:
        self._off.add(instrument)
        if instrument in self.faults:
            return
        self.faults[instrument] = note
        if instrument == "device-events":
            self.gpu_events = False
            self._pending.clear()
            self._reference = None
            self._pool.clear()
        elif instrument == "nvtx":
            self._nvtx = None
            self._nvtx_stage = None
        elif instrument == "progress":
            self.progress_interval_s = 0.0
        stream = self._progress_stream if self._progress_stream is not None else sys.stderr
        try:
            print(
                f"gm1 profiler: {instrument} turned off after {note}; the run continues",
                file=stream, flush=True,
            )
        except Exception:  # noqa: BLE001
            pass
        if instrument in ("steps", "phases") and self.gpu_events:
            # Markers are recorded around phases and resolved, and returned to
            # the pool, only at the end-of-step synchronization ("steps").
            # Without either, the pairs would pile up unresolved for the rest
            # of the run, so the device timing stops with them.
            self._disable(
                "device-events",
                f"{instrument} was turned off (stream markers are recorded around "
                "phases and resolved at the end-of-step synchronization)",
            )

    def _device_fault(self, error: BaseException) -> None:
        self._fault("device-events", error)

    def _reset_steps(self) -> None:
        self.step_count = 0
        self.step_wall_s = 0.0
        self.step_min_s = math.inf
        self.step_max_s = 0.0
        self.sync_wait_s = 0.0
        self.step_sync_wait_s = 0.0
        self.recent = deque(maxlen=_RECENT_STEPS)
        self._step_started: float | None = None
        self._step_wait_at_begin = 0.0
        self._step_counters_at_begin: tuple[int, ...] | None = None
        self.step_counters = [0] * len(COUNTER_FIELDS)

    @_guarded("steps")
    def reset_stepping(self) -> None:
        """Forget stepping statistics (after a warmup), keeping startup and I/O."""

        self.phases = {}
        self.phase_order = []
        self._reset_steps()

    # ---- events

    def _take_event(self) -> Any:
        if self._pool:
            return self._pool.pop()
        return self._event_type()

    def range_push(self, name: str) -> None:
        nvtx = self._nvtx
        if nvtx is not None:
            try:
                nvtx.RangePush(name)
            except Exception as error:  # noqa: BLE001
                self._fault("nvtx", error)

    def range_pop(self) -> None:
        nvtx = self._nvtx
        if nvtx is not None:
            try:
                nvtx.RangePop()
            except Exception as error:  # noqa: BLE001
                self._fault("nvtx", error)

    @_guarded("phases")
    def phase_begin(self, key: str) -> Any:
        if not self.enabled:
            return None
        self.profiler_calls += 1
        start = None
        if self.gpu_events:
            try:
                start = self._take_event()
                start.record(self.stream)
            except Exception as error:  # noqa: BLE001
                start = None
                self._device_fault(error)
        nvtx = self._nvtx
        if nvtx is not None:
            try:
                # The RK stage is a range of its own around its contiguous
                # phases, so an nsys timeline reads step > rk stage > phase.
                stage = key.split(".", 1)[0] if key.startswith("rk") and "." in key else None
                if stage != self._nvtx_stage:
                    if self._nvtx_stage is not None:
                        nvtx.RangePop()
                    if stage is not None:
                        nvtx.RangePush(f"stage.{stage}")
                    self._nvtx_stage = stage
                nvtx.RangePush(key)
            except Exception as error:  # noqa: BLE001
                self._fault("nvtx", error)
        return (key, self.clock(), self.counters.snapshot(), start)

    @_guarded("phases")
    def phase_end(self, token: Any) -> None:
        if token is None:
            return
        host_end = self.clock()
        key, host_start, before, start = token
        self.profiler_calls += 1
        end = None
        if start is not None and self.gpu_events:
            try:
                end = self._take_event()
                end.record(self.stream)
            except Exception as error:  # noqa: BLE001
                end = None
                self._device_fault(error)
        self.range_pop()
        stats = self.phases.get(key)
        if stats is None:
            stats = self.phases[key] = PhaseStats()
            self.phase_order.append(key)
        stats.calls += 1
        stats.host_s += host_end - host_start
        after = self.counters.snapshot()
        tally = stats.counters
        for index in range(len(tally)):
            tally[index] += after[index] - before[index]
        if end is not None and self.gpu_events:
            if len(self._pending) < _MAX_PENDING_MARKERS:
                self._pending.append((stats, start, end, host_end))
            else:
                # No synchronization has resolved the queue: drop the pair
                # rather than hold it, and say so in the receipt.
                self.unresolved_markers += 1
                self._pool.append(start)
                self._pool.append(end)

    @_guarded("steps")
    def synchronized(self, wait_s: float) -> None:
        """Called right after the run's stream synchronization returned."""

        if not self.enabled:
            return
        self.profiler_calls += 1
        self.sync_wait_s += wait_s
        if not self.gpu_events:
            return
        try:
            self._resolve_pending()
            if self._reference is not None:
                self._pool.append(self._reference[0])
                self._reference = None
            # The queue is empty now, so the device reaches this marker at
            # once: it is the origin every later phase end's lag is measured
            # from.
            marker = self._take_event()
            marker.record(self.stream)
            self._reference = (marker, self.clock())
        except Exception as error:  # noqa: BLE001
            self._device_fault(error)

    def _resolve_pending(self) -> None:
        elapsed = self._elapsed
        reference = self._reference
        for stats, start, end, host_end in self._pending:
            try:
                stats.gpu_s += float(elapsed(start, end)) / 1000.0
                stats.gpu_calls += 1
                if reference is not None:
                    device_offset = float(elapsed(reference[0], end)) / 1000.0
                    lag = device_offset - (host_end - reference[1])
                    stats.lag_sum_s += lag
                    stats.lag_calls += 1
                    if lag > stats.lag_max_s:
                        stats.lag_max_s = lag
            except Exception:  # noqa: BLE001 - an unreadable marker is counted, never fatal
                self.unresolved_markers += 1
            self._pool.append(start)
            self._pool.append(end)
        self._pending.clear()

    # ---- steps

    @_guarded("steps")
    def set_target(self, *, duration_s: float | None = None,
                   total_steps: int | None = None, model_start_s: float | None = None) -> None:
        self.duration_s = duration_s
        self.total_steps = total_steps
        self._model_start_s = model_start_s
        self._run_started = self.clock()
        self._progress_last = self._run_started

    @_guarded("steps")
    def step_begin(self) -> Any:
        if not self.enabled:
            return None
        self.range_push("step")
        self._step_started = self.clock()
        self._step_wait_at_begin = self.sync_wait_s
        self._step_counters_at_begin = self.counters.snapshot()
        return self._step_started

    @_guarded("steps")
    def step_end(self, *, step: int, model_time_s: float) -> None:
        started = self._step_started
        if started is None:
            return
        wall = self.clock() - started
        self._step_started = None
        if self._nvtx is not None and self._nvtx_stage is not None:
            self.range_pop()
            self._nvtx_stage = None
        self.range_pop()
        self.step_count += 1
        self.step_wall_s += wall
        self.step_min_s = min(self.step_min_s, wall)
        self.step_max_s = max(self.step_max_s, wall)
        self.recent.append(wall)
        self.step_sync_wait_s += self.sync_wait_s - self._step_wait_at_begin
        if self.first_step_s is None:
            self.first_step_s = wall
        before = self._step_counters_at_begin
        if before is not None:
            after = self.counters.snapshot()
            for index in range(len(self.step_counters)):
                self.step_counters[index] += after[index] - before[index]
        self.maybe_progress(step=step, model_time_s=model_time_s)

    # ---- progress

    @_guarded("progress")
    def maybe_progress(self, *, step: int, model_time_s: float, force: bool = False) -> str | None:
        interval = self.progress_interval_s
        if interval <= 0.0 and not force:
            return None
        now = self.clock()
        if not force and now - self._progress_last < interval:
            return None
        self._progress_last = now
        line = self.progress_line(step=step, model_time_s=model_time_s, now=now)
        stream = self._progress_stream if self._progress_stream is not None else sys.stderr
        try:
            print(line, file=stream, flush=True)
        except (OSError, ValueError) as error:
            # A closed or broken stream (a lost terminal, a full pipe) is a
            # fault like any other: the progress line is turned off and the
            # receipt says so, rather than retried and swallowed every interval.
            self._fault("progress", error)
            return line
        self.progress_lines += 1
        return line

    def progress_fields(self, *, step: int, model_time_s: float,
                        now: float | None = None, telemetry: bool = True) -> dict[str, Any]:
        """The live progress state as one structured record.

        The stderr line (:meth:`progress_line`) and the ``--events`` progress
        record (:mod:`cm1core.run_events`) are both this record, so the two
        can never disagree about a rate or an ETA.
        """

        now = self.clock() if now is None else now
        fields = progress_fields(
            step=step, total_steps=self.total_steps, model_time_s=model_time_s,
            duration_s=self.duration_s, model_start_s=self._model_start_s,
            wall_s=now - self._run_started,
            mean_s=self.step_wall_s / self.step_count if self.step_count else None,
            recent_s=sum(self.recent) / len(self.recent) if self.recent else None,
        )
        fields["last_io"] = None if self.last_io is None else io_summary(self.last_io)
        fields["gpu"] = self.gpu_reading() if telemetry else None
        return fields

    def gpu_reading(self) -> dict[str, Any] | None:
        """NVML utilization, power and memory, when NVML is importable."""

        if not (self._nvml_wanted and self.cp is not None):
            return None
        if self._nvml is None:
            self._nvml = _Nvml(self.cp)
        return self._nvml.values()

    def progress_line(self, *, step: int, model_time_s: float, now: float | None = None) -> str:
        return format_progress_line(
            self.progress_fields(step=step, model_time_s=model_time_s, now=now)
        )

    # ---- I/O and startup

    @_guarded("io")
    def io_event(self, kind: str, label: str) -> IoEvent | None:
        if not self.enabled:
            return None
        return IoEvent(self, kind, label)

    def _io_finished(self, event: IoEvent) -> None:
        table = self.io.get(event.kind)
        if table is None:
            table = self.io[event.kind] = {
                "events": 0, "seconds": 0.0, "bytes": 0, "maximum_seconds": 0.0,
                "blocking_seconds": 0.0, "maximum_blocking_seconds": 0.0,
                "background_events": 0, "stages": {},
            }
        table["events"] += 1
        table["seconds"] += event.seconds
        table["bytes"] += event.nbytes
        table["maximum_seconds"] = max(table["maximum_seconds"], event.seconds)
        table["blocking_seconds"] += event.blocking_seconds
        table["maximum_blocking_seconds"] = max(
            table["maximum_blocking_seconds"], event.blocking_seconds
        )
        table["background_events"] += int(event.background)
        stages = table["stages"]
        for name, row in event.stages.items():
            total = stages.get(name)
            if total is None:
                total = stages[name] = [0.0, 0, 0]
            total[0] += row[0]
            total[1] += row[1]
            total[2] += row[2]
        table["last"] = event.record()
        self.last_io = event

    @_guarded("startup")
    def startup_stage(self, name: str, seconds: float, **extra: Any) -> None:
        row = self.startup.get(name)
        if row is None:
            self.startup[name] = {"seconds": float(seconds), "calls": 1, **extra}
        else:
            row["seconds"] += float(seconds)
            row["calls"] += 1
            row.update(extra)

    @contextmanager
    def startup_timer(self, name: str) -> Iterator[None]:
        self.range_push(f"startup.{name}")
        started = self.clock()
        try:
            yield
        finally:
            self.startup_stage(name, self.clock() - started)
            self.range_pop()

    @_guarded("startup")
    def compiled(self, label: str, seconds: float) -> None:
        row = self.compile_rows.get(label)
        if row is None:
            if len(self.compile_rows) >= _MAX_COMPILE_ROWS:
                self.compile_dropped += 1
                label = "(other programs)"
                row = self.compile_rows.get(label)
                if row is None:
                    row = self.compile_rows[label] = [0.0, 0]
            else:
                row = self.compile_rows[label] = [0.0, 0]
        row[0] += float(seconds)
        row[1] += 1

    # ---- receipt

    def kernel_launches_counted(self) -> bool | None:
        """True while kernel launches are counted, False when they could not be or
        stopped being, None when no counting hooks were ever installed."""

        if KERNEL_LAUNCH_HOOK in self.faults or KERNEL_LAUNCH_HOOK in self.unavailable:
            return False
        if "cupy-hooks" in self.faults:
            return False
        if self.hooks_installed is None:
            return None
        return KERNEL_LAUNCH_HOOK in self.hooks_installed

    def instrumentation(self) -> dict[str, Any]:
        """Which counting hooks ran, which could not, and which were turned off."""

        return {
            "hooks_installed": None if self.hooks_installed is None else list(self.hooks_installed),
            "kernel_launches_counted": self.kernel_launches_counted(),
            "unavailable": dict(self.unavailable),
            "disabled": dict(self.faults),
        }

    def record(self, *, wall_seconds: float | None = None,
               runtime_startup: dict[str, float] | None = None) -> dict[str, Any]:
        """The receipt's performance section.  Never raises."""

        if not self.enabled:
            return {"schema": PERFORMANCE_SCHEMA, "level": self.level}
        try:
            return self._record(wall_seconds=wall_seconds, runtime_startup=runtime_startup)
        except Exception as error:  # noqa: BLE001 - a receipt without numbers is still a receipt
            # Not turned off: the next receipt tries again.
            note = _fault_text(error)
            try:
                instrumentation = self.instrumentation()
            except Exception:  # noqa: BLE001
                instrumentation = None
            return {
                "schema": PERFORMANCE_SCHEMA,
                "level": self.level,
                "record_error": note,
                "instrumentation": instrumentation,
            }

    def _record(self, *, wall_seconds: float | None,
                runtime_startup: dict[str, float] | None) -> dict[str, Any]:
        if self.gpu_events and self._pending:
            try:
                self._resolve_done()
            except Exception as error:  # noqa: BLE001
                self._device_fault(error)
        steps = self.step_count
        per_step = (lambda value: value / steps) if steps else (lambda value: None)
        phases = []
        totals = [0] * len(COUNTER_FIELDS)
        gpu_total = 0.0
        host_total = 0.0
        for key in self.phase_order:
            stats = self.phases[key]
            for index, value in enumerate(stats.counters):
                totals[index] += value
            gpu_total += stats.gpu_s
            host_total += stats.host_s
            row = {
                "phase": key,
                "calls": stats.calls,
                "host_s": stats.host_s,
                "gpu_s": stats.gpu_s if stats.gpu_calls else None,
                "host_s_per_step": per_step(stats.host_s),
                "gpu_s_per_step": per_step(stats.gpu_s) if stats.gpu_calls else None,
                "queue_lag_end_mean_s": (
                    stats.lag_sum_s / stats.lag_calls if stats.lag_calls else None
                ),
                "queue_lag_end_max_s": stats.lag_max_s if stats.lag_calls else None,
            }
            row.update({name: stats.counters[index] for index, name in enumerate(COUNTER_FIELDS)})
            row["launches_per_step"] = per_step(stats.counters[0])
            phases.append(row)
        steady = None
        if steps > 1 and self.first_step_s is not None and self.step_wall_s >= self.first_step_s:
            steady = (self.step_wall_s - self.first_step_s) / (steps - 1)
        step_record = {
            "count": steps,
            "wall_total_s": self.step_wall_s,
            "mean_s": per_step(self.step_wall_s),
            "recent_mean_s": (sum(self.recent) / len(self.recent)) if self.recent else None,
            "minimum_s": None if self.step_min_s is math.inf else self.step_min_s,
            "maximum_s": self.step_max_s if steps else None,
            "first_step_s": self.first_step_s,
            "steady_mean_s_excluding_first": steady,
            "sync_wait_total_s": self.step_sync_wait_s,
            "sync_wait_mean_s": per_step(self.step_sync_wait_s),
            "host_enqueue_mean_s": per_step(self.step_wall_s - self.step_sync_wait_s),
            "gpu_phase_total_s": gpu_total if self.gpu_events else None,
            "gpu_phase_mean_s": per_step(gpu_total) if self.gpu_events else None,
            "host_phase_mean_s": per_step(host_total),
        }
        step_record.update({
            f"{name}_per_step": per_step(self.step_counters[index])
            for index, name in enumerate(COUNTER_FIELDS)
        })
        io_rows = {}
        io_seconds = 0.0
        io_blocking_seconds = 0.0
        for kind, table in self.io.items():
            io_seconds += table["seconds"]
            io_blocking_seconds += table["blocking_seconds"]
            io_rows[kind] = {
                "events": table["events"],
                "seconds": table["seconds"],
                "mean_seconds": table["seconds"] / table["events"],
                "maximum_seconds": table["maximum_seconds"],
                "blocking_seconds": table["blocking_seconds"],
                "mean_blocking_seconds": table["blocking_seconds"] / table["events"],
                "maximum_blocking_seconds": table["maximum_blocking_seconds"],
                "background_events": table["background_events"],
                "bytes": table["bytes"],
                "mb_per_s": _mb_per_s(table["bytes"], table["seconds"]),
                "stages": _stage_rows(table["stages"]),
                "last": table.get("last"),
            }
        startup = dict(self.startup)
        for name, seconds in (runtime_startup or {}).items():
            startup.setdefault(f"runtime.{name}", {"seconds": float(seconds), "calls": 1})
        if self.first_step_s is not None:
            startup["first_step"] = {
                "seconds": self.first_step_s,
                "steady_mean_s": steady,
                "warmup_excess_s": None if steady is None else self.first_step_s - steady,
            }
        compile_rows = sorted(
            ({"program": label, "seconds": row[0], "calls": int(row[1])}
             for label, row in self.compile_rows.items()),
            key=lambda row: -row["seconds"],
        )
        instrumentation = self.instrumentation()
        counted_sources = list(COUNTED_LAUNCH_SOURCES)
        uncounted_sources = list(UNCOUNTED_LAUNCH_SOURCES)
        if instrumentation["kernel_launches_counted"] is False:
            # The first two counted sources are the kernel launches.
            uncounted_sources = counted_sources[:2] + uncounted_sources
            counted_sources = counted_sources[2:]
        record = {
            "schema": PERFORMANCE_SCHEMA,
            "level": self.level,
            "gpu_event_timing": self.gpu_events,
            "nvtx_ranges": self._nvtx is not None,
            "counted_launch_sources": counted_sources,
            "uncounted_launch_sources": uncounted_sources,
            "instrumentation": instrumentation,
            "timing_semantics": (
                "gpu_s is the device-timeline span between two stream markers "
                "around the phase (it includes device idle time while the host "
                "is still enqueuing); host_s is the host enqueue time; "
                "queue_lag_end is how far the device was behind the host when "
                "the phase finished enqueuing; sync_wait is the end-of-step "
                "stream synchronization, i.e. device work still queued when "
                "the host finished the step"
            ),
            "wall_seconds": wall_seconds,
            "steps": step_record,
            "phases": phases,
            "totals": dict(zip(COUNTER_FIELDS, totals)),
            "io": io_rows,
            # All publication work, wherever it ran; and the part of it the
            # stepping thread waited for, which is what the wall clock paid.
            "io_seconds": io_seconds,
            "io_blocking_seconds": io_blocking_seconds,
            "startup": startup,
            "compile": {
                "programs": len(compile_rows) + self.compile_dropped,
                "seconds": sum(row["seconds"] for row in compile_rows),
                "rows": compile_rows,
                "rows_dropped_into_other": self.compile_dropped,
            },
            "unresolved_markers": self.unresolved_markers + len(self._pending),
            "profiler_calls": self.profiler_calls,
            "profiler_calls_per_step": per_step(self.profiler_calls),
            "progress_lines": self.progress_lines,
        }
        record["verdict"] = performance_verdict(record)
        return record

    def _resolve_done(self) -> None:
        """Resolve markers the device has already passed, without waiting."""

        if all(bool(getattr(end, "done", False)) for _, _, end, _ in self._pending):
            self._resolve_pending()


def performance_verdict(record: dict[str, Any]) -> dict[str, Any]:
    """A one-line reading of where a run's wall time went."""

    steps = record.get("steps") or {}
    count = int(steps.get("count") or 0)
    wall = steps.get("mean_s")
    wait = steps.get("sync_wait_mean_s")
    launches = steps.get("launches_per_step")
    gpu = steps.get("gpu_phase_mean_s")
    # False when the kernel-launch hook could not run or was turned off: the
    # launch count then undercounts and cannot name a launch-bound host.
    counted = (record.get("instrumentation") or {}).get("kernel_launches_counted")
    labels: list[str] = []
    evidence: dict[str, Any] = {
        "step_mean_s": wall, "sync_wait_mean_s": wait, "gpu_phase_mean_s": gpu,
        "launches_per_step": launches, "kernel_launches_counted": counted,
    }
    reason = []
    if count and wall:
        backlog = (wait or 0.0) / wall
        evidence["backlog_fraction"] = backlog
        if backlog >= GPU_BOUND_BACKLOG_FRACTION:
            labels.append("GPU-bound")
            reason.append(
                f"{100 * backlog:.0f}% of each step is spent waiting for queued "
                "device work, so GPU time is close to the wall step time"
            )
        elif (backlog < HOST_BOUND_BACKLOG_FRACTION and counted is not False
              and (launches or 0) >= HIGH_LAUNCHES_PER_STEP):
            labels.append("host-launch-bound")
            reason.append(
                f"the device finishes within {100 * backlog:.0f}% of the step of the "
                f"host enqueuing it, with {launches:.0f} counted launches per step"
            )
        elif backlog < HOST_BOUND_BACKLOG_FRACTION:
            labels.append("host-bound")
            reason.append(
                f"the device finishes within {100 * backlog:.0f}% of the step of the "
                "host enqueuing it"
                + ("" if counted is not False else
                   " (kernel launches were not counted, so a launch-bound host is not ruled out)")
            )
        else:
            labels.append("mixed")
            reason.append(f"{100 * backlog:.0f}% of each step waits for the device")
    wall_total = record.get("wall_seconds")
    # Publication work on the background worker overlaps the steps; only the
    # part the stepping thread blocked on is wall time the run lost to I/O.
    io_seconds = float(
        record.get("io_blocking_seconds", record.get("io_seconds")) or 0.0
    )
    if wall_total:
        share = io_seconds / wall_total
        evidence["io_share_of_wall"] = share
        if share > IO_HEAVY_SHARE:
            labels.append("I/O-heavy")
            reason.append(f"I/O took {100 * share:.0f}% of the wall clock")
    if not labels:
        labels.append("insufficient-data")
    return {
        "label": "; ".join(labels),
        "line": f"{'; '.join(labels)}: " + ("; ".join(reason) if reason else "no measured steps"),
        "evidence": evidence,
        "thresholds": {
            "gpu_bound_backlog_fraction": GPU_BOUND_BACKLOG_FRACTION,
            "host_bound_backlog_fraction": HOST_BOUND_BACKLOG_FRACTION,
            "high_launches_per_step": HIGH_LAUNCHES_PER_STEP,
            "io_heavy_share": IO_HEAVY_SHARE,
        },
    }


def publish_timing_environment() -> dict[str, str]:
    """The publisher environment that asks it for stage timings on stderr."""

    environment = dict(os.environ)
    environment[PUBLISH_TIMING_ENV] = "1"
    return environment


def parse_publish_timing(stderr: str) -> list[tuple[str, float, int]]:
    """Rows ``gm1-publish-timing<TAB>stage<TAB>seconds<TAB>bytes`` from stderr."""

    rows = []
    for line in (stderr or "").splitlines():
        columns = line.split("\t")
        if len(columns) != 4 or columns[0] != PUBLISH_TIMING_PREFIX:
            continue
        try:
            rows.append((columns[1], float(columns[2]), int(columns[3])))
        except ValueError:
            continue
    return rows


# ------------------------------------------------------------------- text


def format_performance(performance: dict[str, Any], *, top: int = 12) -> str:
    """The human summary ``gpuwm-cm1 perf`` prints."""

    lines = []
    level = performance.get("level")
    lines.append(f"profiling level: {level}")
    instrumentation = performance.get("instrumentation") or {}
    for hook, note in (instrumentation.get("disabled") or {}).items():
        lines.append(f"profiler: {hook} was turned off during the run: {note}")
    for hook, note in (instrumentation.get("unavailable") or {}).items():
        lines.append(f"profiler: {hook} was not installed: {note}")
    if performance.get("record_error"):
        lines.append(f"the performance record could not be built: {performance['record_error']}")
        return "\n".join(lines) + "\n"
    if level == "off" or "steps" not in performance:
        lines.append("no performance measurements were recorded (profiling was off)")
        return "\n".join(lines) + "\n"
    verdict = performance.get("verdict") or {}
    lines.append(f"verdict: {verdict.get('line', 'n/a')}")
    steps = performance["steps"]

    def num(value: Any, fmt: str = "{:.3f}") -> str:
        return "-" if value is None else fmt.format(value)

    lines.append("")
    lines.append(
        f"steps: {steps.get('count')}  mean {num(steps.get('mean_s'))} s  "
        f"steady {num(steps.get('steady_mean_s_excluding_first'))} s  "
        f"first {num(steps.get('first_step_s'))} s  "
        f"min {num(steps.get('minimum_s'))}  max {num(steps.get('maximum_s'))}"
    )
    lines.append(
        f"per step: gpu phases {num(steps.get('gpu_phase_mean_s'))} s  "
        f"host enqueue {num(steps.get('host_enqueue_mean_s'))} s  "
        f"sync wait {num(steps.get('sync_wait_mean_s'))} s  "
        f"launches {num(steps.get('launches_per_step'), '{:.0f}')}  "
        f"d2d {num(steps.get('d2d_count_per_step'), '{:.0f}')} "
        f"({_bytes_text(int(steps.get('d2d_bytes_per_step') or 0))})  "
        f"h2d {num(steps.get('h2d_count_per_step'), '{:.1f}')}  "
        f"d2h {num(steps.get('d2h_count_per_step'), '{:.1f}')}"
    )
    phases = list(performance.get("phases") or [])
    if phases:
        lines.append("")
        lines.append(
            f"{'phase':32s} {'gpu s/step':>10s} {'host s/step':>11s} "
            f"{'lag end':>8s} {'launch/step':>11s} {'d2d MiB/step':>12s}"
        )
        count = max(1, int(steps.get("count") or 1))
        key = (lambda row: -(row.get("gpu_s") or 0.0)) if any(
            row.get("gpu_s") is not None for row in phases
        ) else (lambda row: -row.get("host_s", 0.0))
        for row in sorted(phases, key=key)[:top]:
            lines.append(
                f"{row['phase'][:32]:32s} {num(row.get('gpu_s_per_step'), '{:10.4f}'):>10s} "
                f"{num(row.get('host_s_per_step'), '{:11.4f}'):>11s} "
                f"{num(row.get('queue_lag_end_mean_s'), '{:8.3f}'):>8s} "
                f"{num(row.get('launches_per_step'), '{:11.1f}'):>11s} "
                f"{(row.get('d2d_bytes', 0) / count / 2**20):12.1f}"
            )
    io = performance.get("io") or {}
    if io:
        lines.append("")
        lines.append("I/O:")
        for kind, table in io.items():
            lines.append(
                f"  {kind}: {table['events']} events, {table['seconds']:.2f} s total, "
                f"max {table['maximum_seconds']:.2f} s, {_bytes_text(int(table['bytes']))}"
                + ("" if table.get("mb_per_s") is None else f", {table['mb_per_s']:.0f} MB/s")
                + (
                    "" if not table.get("background_events") else
                    f"; stepping blocked {table['blocking_seconds']:.2f} s "
                    f"(max {table['maximum_blocking_seconds']:.2f} s), "
                    f"{table['background_events']} committed in the background"
                )
            )
            for stage in sorted(table["stages"], key=lambda row: -row["seconds"]):
                rate = "" if stage.get("mb_per_s") is None else f"  {stage['mb_per_s']:.0f} MB/s"
                lines.append(
                    f"    {stage['stage']:34s} {stage['seconds']:9.3f} s  "
                    f"{_bytes_text(int(stage['bytes'])):>10s}{rate}"
                )
        lines.append(f"  io share of wall: {num((verdict.get('evidence') or {}).get('io_share_of_wall'), '{:.1%}')}")
    startup = performance.get("startup") or {}
    if startup:
        lines.append("")
        lines.append("startup:")
        for name, row in startup.items():
            lines.append(f"  {name:36s} {num(row.get('seconds'), '{:9.3f}')} s")
    compiled = performance.get("compile") or {}
    if compiled.get("rows"):
        lines.append(
            f"  compile: {compiled['programs']} programs, {compiled['seconds']:.2f} s"
        )
        for row in compiled["rows"][:5]:
            lines.append(f"    {row['program'][:48]:48s} {row['seconds']:8.3f} s")
    return "\n".join(lines) + "\n"


__all__ = [
    "COUNTED_LAUNCH_SOURCES",
    "COUNTERS",
    "CupyHooks",
    "DEFAULT_PROFILE_LEVEL",
    "DEFAULT_PROGRESS_INTERVAL_S",
    "IoEvent",
    "KERNEL_LAUNCH_HOOK",
    "LaunchCounters",
    "PERFORMANCE_SCHEMA",
    "PROFILE_LEVELS",
    "PUBLISH_TIMING_ENV",
    "RuntimeProfiler",
    "UNCOUNTED_LAUNCH_SOURCES",
    "format_performance",
    "format_progress_line",
    "io_summary",
    "parse_publish_timing",
    "performance_verdict",
    "progress_fields",
    "publish_timing_environment",
    "require_profile_level",
    "require_progress_interval",
]
