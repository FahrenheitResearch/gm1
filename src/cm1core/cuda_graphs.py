"""CUDA graph capture and replay of one runtime's large-step schedule.

A GM1 large step is the source-ordered schedule of ``gm1_runtime.large_step_schedule``
executed by Python: tens to thousands of small kernel launches, copies and
fills per step, each one built, validated and receipted on the host.  On a
nested tornado run the device finishes each of them before Python has built
the next (the profiler's ``host-launch-bound`` verdict), so the step costs the
host's time, not the device's.

This module records one step's device work once, as a CUDA graph, by stream
capture of the unchanged Python schedule, and launches the recorded graph for
later steps instead of running the Python again.  Nothing numerical changes:
a graph launch executes the same kernel nodes with the same argument bytes in
the same stream order the Python would have enqueued.

What a replay must still do on the host, and how each is carried
----------------------------------------------------------------
* **Host reads of device words** (a validation count that stops the run, the
  closure's eddy-coefficient maximum).  :func:`read_device` is the one door:
  eager, it reads at once; under capture it records a device-to-device copy of
  the words into a slot the graph owns, and after every launch and its
  synchronization the slot is read and the same callback runs.
* **Host bookkeeping** (the completed-step counters the phase order checks and
  the restart writer read).  :func:`bookkeeping` runs the update at once and,
  under capture, records it; every replay runs the recorded updates with the
  replayed step number.
* **Host waits** (``stream.synchronize()`` before a registry release, a
  launch's ready event).  On one stream the graph's own node order is the
  wait; :func:`synchronize` is a no-op while capturing and a wait otherwise.

When a recorded graph may replace a step, and when it may not
--------------------------------------------------------------
A graph's kernel arguments are device addresses and host scalars frozen at
capture.  A replay is the step the Python would have run only if

1. the host decisions the Python makes from step-varying host state are the
   same (``graph_key``: the runtime names every one it knows -- a reflectivity
   step, a nest substep's boundary time; a step it cannot key it runs eagerly);
2. every device array reachable from the runtime sits at the address and view
   it had when the graph was captured (the *inventory*, compared before every
   replay; a mismatch captures a new graph);
3. every block the graph writes that is not such an array is a private block
   no one else can be handed: all allocations during capture come from a pool
   owned by these graphs and used for nothing else, and after capture the pool
   may hold no live block the inventory cannot name (else the key is refused).

``verify`` mode captures every step, replays nothing, and compares each new
graph node by node -- kernel, launch shape and every argument byte, copies and
fills by their full parameter records -- with the graph cached for the same
key and inventory.  A mismatch names the node and kernel: it is the proof, per
configuration, that ``on`` mode replays exactly the Python's launches.
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable

CUDA_GRAPHS_SCHEMA = "gm1.cuda-graphs/v1"
CUDA_GRAPH_MODES = ("off", "on", "verify")
DEFAULT_CUDA_GRAPH_MODE = "off"
# A key whose inventory changes on every capture never reaches a replay; after
# this many captures without one the key runs eagerly for the rest of the run,
# so a configuration the design does not cover costs one capture per step at
# most this many times, never for the whole run.
MAX_CAPTURES_WITHOUT_REPLAY = 4
MAX_ENTRIES_PER_KEY = 4
# Consecutive disagreeing captures of one key before it is never replayed.
MAX_CONFIRMATION_FAILURES = 3
# ``on`` mode re-captures a replayed graph after this many replays and holds
# the new capture to it node by node; a difference stops the run
# (CudaGraphReplayMismatch).  Graphs are first trusted only after two
# consecutive captures agree.
AUDIT_INTERVAL = 256

_LOCAL = threading.local()


class CudaGraphCaptureError(RuntimeError):
    """Stream capture of a step failed; the step's host state is half-applied."""


class CudaGraphReplayMismatch(RuntimeError):
    """An audit capture differs from the graph the run has been replaying."""


# ------------------------------------------------------------ capture doors


def active_capture() -> "_Capture | None":
    """The schedule capture recording on this thread, or None."""

    return getattr(_LOCAL, "capture", None)


def synchronize(waitable: Any) -> None:
    """``waitable.synchronize()``, unless a schedule capture is recording.

    Every wait this is used for orders host work after device work on the
    runtime's one stream (a buffer drop, a ready event).  Inside a graph the
    same order is the graph's own node order, and a host wait on a capturing
    stream is illegal (``cudaErrorStreamCaptureUnsupported``).
    """

    if active_capture() is None:
        waitable.synchronize()


_STREAM_CLASSES: dict[int, type] = {}


def capture_stream_class(cupy: Any) -> type:
    """The runtime's stream class under ``--cuda-graphs``: a ``wait_event`` a capture skips.

    The runtime runs on the one stream the door creates, and every ready event
    its launchers wait for was recorded on that stream, so the wait orders
    nothing stream order has not already ordered.  Under capture it would join
    the graph to work recorded before the capture began, which CUDA refuses
    (``cudaErrorStreamCaptureIsolation``).  Everything else is CuPy's stream:
    a host ``synchronize`` during a capture still fails loudly.
    """

    known = _STREAM_CLASSES.get(id(cupy))
    if known is not None:
        return known

    class CaptureAwareStream(cupy.cuda.Stream):  # type: ignore[misc,valid-type]
        def wait_event(self, event: Any) -> None:
            if active_capture() is None:
                super().wait_event(event)

    _STREAM_CLASSES[id(cupy)] = CaptureAwareStream
    return CaptureAwareStream


_VERTICAL_WORDS = ("sigma", "sigmaf", "rds", "rdsf")


def install_frozen_module_adapters() -> None:
    """Keep a frozen operator module's per-stage host read out of the captured step.

    ``cuda_gm1_child_psolver3_open._require_vertical`` validates the stretched
    coordinate on every acoustic stage, and for a generated coordinate that
    includes reading ``sigmaf(nk+1)`` back from the device (``param.F:8193``):
    a host wait, which a stream capture refuses.  The module's bytes are pinned
    by the child operator manifest (``gm1_child_operator_manifest``), so the
    read is not moved inside it.  Under ``--cuda-graphs`` the function is
    wrapped instead: the first call for a coordinate runs it whole; a later
    call for the same coordinate object, holding the same four profile arrays
    and the same top and reference spacing, returns what that call returned.
    The coordinate's words are immutable once generated
    (``cuda_gm1_child_stretched_metrics.require_generated_vertical`` already
    reads the same word once per coordinate for the same reason).  Installed
    only by :meth:`CudaCartesianRuntime.enable_cuda_graphs`; without graphs
    the module runs exactly as pinned.
    """

    import weakref

    from . import cuda_gm1_child_psolver3_open as module

    original = module._require_vertical
    if getattr(original, "cuda_graphs_adapter", False):
        return
    verified: dict[tuple[int, int, bool], tuple[Any, ...]] = {}

    def _require_vertical(cupy: Any, vertical: Any, *, nk: int,
                          allow_uniform_idealized_validation: bool = False) -> Any:
        key = (id(vertical), nk, allow_uniform_idealized_validation)
        hit = verified.get(key)
        if hit is not None:
            reference, arrays, words, result = hit
            if (reference() is vertical
                    and all(getattr(vertical, name) is array
                            for name, array in zip(_VERTICAL_WORDS, arrays))
                    and (vertical.ztop_f32, vertical.dz_reference_f32) == words):
                return result
        result = original(cupy, vertical, nk=nk,
                          allow_uniform_idealized_validation=allow_uniform_idealized_validation)
        verified[key] = (
            weakref.ref(vertical),
            tuple(getattr(vertical, name) for name in _VERTICAL_WORDS),
            (vertical.ztop_f32, vertical.dz_reference_f32),
            result,
        )
        return result

    _require_vertical.cuda_graphs_adapter = True  # type: ignore[attr-defined]
    _require_vertical.__wrapped__ = original  # type: ignore[attr-defined]
    module._require_vertical = _require_vertical


def read_device(value: Any, apply: Callable[[Any], None], *, stream: Any) -> None:
    """Give ``apply`` the host copy of the small device array ``value``.

    Eager: ``apply(value.get(stream=stream))`` now.  Under capture: the graph
    copies ``value`` into a slot it owns at this point of the step, and after
    every launch of the graph (and its synchronization) ``apply`` receives the
    slot's words -- the same words, read at the same point of the same step.
    ``apply`` may raise; the step then fails as it would have eagerly.
    """

    capture = active_capture()
    if capture is None:
        apply(value.get(stream=stream))
        return
    capture.defer_read(value, apply)


def bookkeeping(update: Callable[[int], None], step: int) -> None:
    """Run the host-only ``update(step)`` now, and on every replay of this capture."""

    update(step)
    capture = active_capture()
    if capture is not None:
        capture.epilogues.append(update)


# ------------------------------------------------------------- inventory


# type -> the __slots__ names an inventory walk reads, or False for a type it
# never enters (a library object that cannot hold a runtime's array).
_TYPE_SLOTS: dict[type, Any] = {}
_OPAQUE_MODULES = ("cupy", "numpy", "threading", "_thread", "concurrent", "queue",
                   "weakref", "pathlib", "io", "_io", "logging", "enum")


def _walkable_slots(kind: type) -> Any:
    if (kind.__module__ or "").startswith(_OPAQUE_MODULES):
        return False
    names = []
    for cls in kind.__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if isinstance(slot, str) and slot not in ("__dict__", "__weakref__") and slot not in names:
                names.append(slot)
    return tuple(names)


def device_inventory(roots: Iterable[tuple[str, Any]], *, cupy: Any,
                     private_pool: Any = None, rows: list[str] | None = None,
                     stop: Iterable[Any] = (),
                     private_seen: set[int] | None = None) -> tuple[str, int, int]:
    """Every device array reachable from ``roots``: its path, address and view.

    Returns ``(sha256, arrays, private_bytes)``: ``private_bytes`` is the size
    of the distinct allocations those arrays hold from ``private_pool`` (a
    ``SingleDeviceMemoryPool``), which :class:`ScheduleGraphs` compares with
    what the pool says is in use.  The walk follows instance attributes, slots, dicts, lists, tuples and sets; it
    stops at modules, types, functions, strings and numbers.  A path is the
    first one the deterministic walk reaches an object by, so two arrays that
    swap owners change the digest even though their address set does not.
    ``stop`` objects are never entered (the runtime itself, reached back
    through an attribute that holds it); neither is a :class:`ScheduleGraphs`.
    """

    ndarray = cupy.ndarray
    digest = hashlib.sha256()
    seen: set[int] = {id(obj) for obj in stop}
    count = 0
    private: dict[int, int] = {}
    private_seen = set() if private_seen is None else private_seen
    leaf = (str, bytes, int, float, bool, complex, type(None))
    skip = (type, type(len), type(device_inventory), type(threading))
    # A path is built only for an array: every other entry carries its parent
    # link, (parent, label), and the string is joined when an array is found.
    stack: list[tuple[Any, Any]] = [((None, name), root) for name, root in reversed(list(roots))]
    while stack:
        link, obj = stack.pop()
        if isinstance(obj, leaf) or isinstance(obj, skip):
            continue
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(obj, ndarray):
            count += 1
            data = obj.data
            parts = []
            while link is not None:
                link, label = link
                parts.append(label)
            row = f"{''.join(reversed(parts))}|{data.ptr:x}|{obj.shape}|{obj.strides}|{obj.dtype.str}\n"
            digest.update(row.encode())
            if rows is not None:
                rows.append(row)
            if private_pool is not None:
                mem = data.mem
                owner = getattr(mem, "pool", None)
                if owner is not None and owner() is private_pool and int(mem.ptr) not in private_seen:
                    private_seen.add(int(mem.ptr))
                    private[int(mem.ptr)] = int(mem.size)
            continue
        kind = type(obj)
        if kind is dict or isinstance(obj, (dict, MappingProxyType)):
            for name, value in reversed(list(obj.items())):
                if not isinstance(value, leaf):
                    stack.append(((link, f"[{name}]" if isinstance(name, str) else f"[{name!r}]"), value))
            continue
        if kind is list or kind is tuple or isinstance(obj, (list, tuple)):
            for index in range(len(obj) - 1, -1, -1):
                value = obj[index]
                if not isinstance(value, leaf):
                    stack.append(((link, f"[{index}]"), value))
            continue
        if isinstance(obj, (set, frozenset)):
            for index, value in enumerate(sorted(obj, key=repr)):
                stack.append(((link, f"{{{index}}}"), value))
            continue
        slots = _TYPE_SLOTS.get(kind)
        if slots is None:
            slots = _TYPE_SLOTS[kind] = _walkable_slots(kind)
        if slots is False or isinstance(obj, ScheduleGraphs):
            continue
        attrs: dict[str, Any] = {}
        try:
            attrs.update(vars(obj))
        except TypeError:
            pass
        for slot in slots:
            if slot not in attrs:
                try:
                    attrs[slot] = getattr(obj, slot)
                except AttributeError:
                    pass
        for name in reversed(list(attrs)):
            value = attrs[name]
            if not isinstance(value, leaf):
                stack.append(((link, f".{name}"), value))
    return digest.hexdigest(), count, sum(private.values())


# ------------------------------------------------------------ fingerprint


class _KernelNodeParams(ctypes.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v2 (cuda.h, CUDA 12.0 and newer).
    _fields_ = [
        ("func", ctypes.c_void_p),
        ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("kern", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
    ]


# sizeof(CUDA_MEMCPY3D) and sizeof(CUDA_MEMSET_NODE_PARAMS) on LP64.
_MEMCPY3D_BYTES = 200
_MEMSET_BYTES = 40
_NODE_TYPES = {
    0: "kernel", 1: "memcpy", 2: "memset", 3: "host", 4: "graph", 5: "empty",
    6: "wait-event", 7: "event-record", 8: "ext-semas-signal", 9: "ext-semas-wait",
    10: "mem-alloc", 11: "mem-free", 12: "batch-mem-op", 13: "conditional",
}
_DRIVER: Any = None


def _driver() -> Any:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = ctypes.CDLL("libcuda.so.1")
    return _DRIVER


def _check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"{what} returned CUresult {result}")


class _Canonical:
    """Name the words of a node record that point into a capture's own allocations.

    Two captures of the same step make the same allocations in the same order,
    but the pool may hand them different addresses (the first capture of a run
    finds it empty).  A word inside the ``k``-th allocation of a capture is
    written ``@k+offset``; every other word -- persistent state, sizes,
    scalars -- stays its bytes.  So two fingerprints are equal exactly when the
    captures launched the same kernels on the same persistent arrays, the same
    allocations and the same scalars.
    """

    # How far left of the bisection point to look for an older allocation
    # that also contains a word (a block split, freed and taken again).
    _SCAN = 256

    def __init__(self, allocations: list[tuple[int, int, int]]) -> None:
        import bisect

        self._bisect = bisect.bisect_right
        order = sorted(
            (ptr, ptr + size, nodes, index)
            for index, (ptr, size, nodes) in enumerate(allocations)
        )
        self.starts = [row[0] for row in order]
        self.rows = order
        self.low = order[0][0] if order else 0
        self.high = max((row[1] for row in order), default=0)
        self.node = 0

    def label(self, word: int) -> str | None:
        """The allocation live at the current node that contains ``word``, or None.

        Memory a capture frees is taken again later in the same capture, so an
        address can lie in several of its allocations; the one that names it at
        node ``n`` is the latest made before node ``n`` was.
        """

        if not self.low <= word < self.high:
            return None
        position = self._bisect(self.starts, word) - 1
        best = None
        for slot in range(position, max(-1, position - self._SCAN), -1):
            start, end, nodes, index = self.rows[slot]
            if start <= word < end and nodes <= self.node and (best is None or index > best[1]):
                best = (start, index)
        if best is None:
            return None
        return f"@{best[1]}+{word - best[0]}"

    def words(self, raw: bytes) -> str:
        if not self.rows or len(raw) % 8:
            return raw.hex()
        out = []
        for at in range(0, len(raw), 8):
            word = int.from_bytes(raw[at:at + 8], "little")
            named = self.label(word)
            out.append(raw[at:at + 8].hex() if named is None else named)
        return ".".join(out)


def _cupy_indexer_prefix(name: str, raw: bytes) -> bytes:
    """A CuPy ``CIndexer`` argument without its uninitialised ``index`` scratch.

    CuPy's generated kernels (``cupy_*``) take ``CArray`` (pointer, size,
    shape[n], strides[n]: an even number of 8-byte words) and ``CIndexer``
    (size, shape[n], index[n]: an odd number).  ``index`` is per-thread
    scratch the host never writes, so its bytes are whatever the host buffer
    held; ``size`` and ``shape`` are the argument.
    """

    words = len(raw) // 8
    if name.startswith("cupy_") and len(raw) % 8 == 0 and words >= 3 and words % 2 == 1:
        return raw[: 8 * (1 + (words - 1) // 2)]
    return raw


def graph_fingerprint(graph_handle: int,
                      allocations: list[tuple[int, int, int]] | None = None) -> list[tuple[Any, ...]]:
    """Every node of a captured graph, in topological order, by its full parameters.

    Kernel nodes: the function's name, grid, block, shared memory and every
    argument's bytes (``cuFuncGetParamInfo`` gives each argument's size), with
    words that point into the capture's own ``allocations`` named by
    allocation (:class:`_Canonical`).
    Copy and fill nodes: their whole parameter records.  Any other node type
    is recorded by type only -- a capture of this runtime's single stream
    makes none, and a graph that has one differs from one that has not.
    Edges are not listed: a single-stream capture is a chain, and the order
    below is ``cuGraphGetNodes``' creation order, which is capture order.
    """

    cuda = _driver()
    canonical = _Canonical(list(allocations or ()))
    graph = ctypes.c_void_p(graph_handle)
    count = ctypes.c_size_t(0)
    _check(cuda.cuGraphGetNodes(graph, None, ctypes.byref(count)), "cuGraphGetNodes")
    nodes = (ctypes.c_void_p * count.value)()
    _check(cuda.cuGraphGetNodes(graph, nodes, ctypes.byref(count)), "cuGraphGetNodes")
    edges = ctypes.c_size_t(0)
    rows: list[tuple[Any, ...]] = []
    names: dict[int, str] = {}
    for position, node in enumerate(nodes):
        canonical.node = position
        handle = ctypes.c_void_p(node)
        kind = ctypes.c_int()
        _check(cuda.cuGraphNodeGetType(handle, ctypes.byref(kind)), "cuGraphNodeGetType")
        if kind.value == 0:
            params = _KernelNodeParams()
            _check(cuda.cuGraphKernelNodeGetParams_v2(handle, ctypes.byref(params)),
                   "cuGraphKernelNodeGetParams")
            func = int(params.func or 0)
            name = names.get(func)
            if name is None:
                text = ctypes.c_char_p()
                _check(cuda.cuFuncGetName(ctypes.byref(text), ctypes.c_void_p(func)), "cuFuncGetName")
                name = names[func] = text.value.decode()
            arguments = []
            index = 0
            while True:
                offset, size = ctypes.c_size_t(), ctypes.c_size_t()
                if cuda.cuFuncGetParamInfo(ctypes.c_void_p(func), ctypes.c_size_t(index),
                                           ctypes.byref(offset), ctypes.byref(size)) != 0:
                    break
                raw = _cupy_indexer_prefix(
                    name, ctypes.string_at(params.kernelParams[index], size.value)
                )
                arguments.append(canonical.words(raw))
                index += 1
            rows.append((
                "kernel", name,
                (params.gridDimX, params.gridDimY, params.gridDimZ,
                 params.blockDimX, params.blockDimY, params.blockDimZ, params.sharedMemBytes),
                tuple(arguments),
            ))
        elif kind.value == 1:
            record = ctypes.create_string_buffer(512)
            _check(cuda.cuGraphMemcpyNodeGetParams(handle, record), "cuGraphMemcpyNodeGetParams")
            rows.append(("memcpy", canonical.words(record.raw[:_MEMCPY3D_BYTES])))
        elif kind.value == 2:
            record = ctypes.create_string_buffer(128)
            _check(cuda.cuGraphMemsetNodeGetParams(handle, record), "cuGraphMemsetNodeGetParams")
            rows.append(("memset", canonical.words(record.raw[:_MEMSET_BYTES])))
        else:
            rows.append((_NODE_TYPES.get(kind.value, f"type-{kind.value}"),))
    _check(cuda.cuGraphGetEdges_v2(graph, None, None, None, ctypes.byref(edges))
           if hasattr(cuda, "cuGraphGetEdges_v2") else
           cuda.cuGraphGetEdges(graph, None, None, ctypes.byref(edges)), "cuGraphGetEdges")
    rows.append(("edges", int(edges.value)))
    return rows


def graph_node_count(graph_handle: int) -> int:
    count = ctypes.c_size_t(0)
    _check(_driver().cuGraphGetNodes(ctypes.c_void_p(graph_handle), None, ctypes.byref(count)),
           "cuGraphGetNodes")
    return int(count.value)


def fingerprint_sha256(rows: list[tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def first_difference(old: list[tuple[Any, ...]], new: list[tuple[Any, ...]]) -> dict[str, Any] | None:
    """The first node at which two fingerprints differ, named for a reader."""

    if old == new:
        return None
    for index, (a, b) in enumerate(zip(old, new)):
        if a != b:
            row: dict[str, Any] = {"node": index, "kind": b[0]}
            if a[0] == b[0] == "kernel":
                row["kernel"] = b[1]
                if a[2] != b[2]:
                    row["launch_shape"] = [list(a[2]), list(b[2])]
                row["arguments"] = [
                    {"index": i, "captured": x, "now": y}
                    for i, (x, y) in enumerate(zip(a[3], b[3])) if x != y
                ][:8]
            else:
                row["captured"] = str(a)[:300]
                row["now"] = str(b)[:300]
            return row
    return {"node_count": [len(old), len(new)]}


# --------------------------------------------------------------- capture


class _Capture:
    """One stream capture of a schedule: its readback slots and host updates."""

    def __init__(self, owner: "ScheduleGraphs") -> None:
        self.owner = owner
        self.reads: list[tuple[Any, Callable[[Any], None]]] = []
        self.epilogues: list[Callable[[int], None]] = []
        self.events: list[Any] = []
        self.thread = threading.get_ident()
        # (phase, event) at the start of every phase and (None, event) at the
        # end: external event-record nodes, recorded by every launch of the
        # graph, whose spans are each phase's device time.
        self.marks: list[tuple[str | None, Any]] = []
        # (address, extent, graph nodes before it) of every allocation the
        # capture made, in order; kept only where a fingerprint will read it.
        self.allocations: list[tuple[int, int, int]] = []
        self.track = True

    def nodes_so_far(self) -> int:
        """How many nodes the capture's graph holds now (``cuStreamGetCaptureInfo``)."""

        cuda = _driver()
        status, ident, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
        _check(cuda.cuStreamGetCaptureInfo_v2(
            ctypes.c_void_p(self.owner.stream.ptr), ctypes.byref(status), ctypes.byref(ident),
            ctypes.byref(graph), None, None,
        ), "cuStreamGetCaptureInfo")
        count = ctypes.c_size_t(0)
        _check(cuda.cuGraphGetNodes(graph, None, ctypes.byref(count)), "cuGraphGetNodes")
        return int(count.value)

    def mark(self, phase: str | None, event_class: type) -> None:
        """Record a timing event into the graph as an external event-record node.

        A plain ``Event.record`` during a capture only orders the capture;
        ``CU_EVENT_RECORD_EXTERNAL`` makes it a node every launch records, so
        the replayed step keeps a per-phase device time.
        """

        event = event_class()                     # timing enabled, not tracked
        _check(_driver().cuEventRecordWithFlags(
            ctypes.c_void_p(event.ptr), ctypes.c_void_p(self.owner.stream.ptr), ctypes.c_uint(1),
        ), "cuEventRecordWithFlags")
        self.marks.append((phase, event))

    def malloc(self, size: int) -> Any:
        pointer = self.owner.pool.malloc(size)
        if self.track:
            self.allocations.append(
                (int(pointer.ptr), int(pointer.mem.size), self.nodes_so_far())
            )
        return pointer

    def tracked_event_class(self) -> type:
        """``cupy.cuda.Event`` for the capture: every event made on this thread is kept.

        An event recorded while a stream is capturing is a node dependency of
        that capture, not a recorded event: after the capture CUDA refuses a
        ``wait_event`` on it (``cudaErrorInvalidValue``), and a launcher that
        hands one on as a ready event would break the next eager step that
        waits for it.  :meth:`ScheduleGraphs._capture` records every such event
        again on the stream once the capture has ended.
        """

        base = self.owner.cp.cuda.Event
        capture = self

        class _CaptureEvent(base):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                if threading.get_ident() == capture.thread:
                    capture.events.append(self)

        return _CaptureEvent

    def defer_read(self, value: Any, apply: Callable[[Any], None]) -> None:
        cp = self.owner.cp
        # From a pool of their own: a slot outlives the capture (its graph
        # writes it on every launch), and taken from the graphs' pool it would
        # leave that pool in a different state for the next capture, which
        # would then place its scratch at other addresses.
        with cp.cuda.using_allocator(self.owner.slot_pool.malloc):
            slot = cp.empty(value.shape, dtype=value.dtype)
        if self.track:
            self.allocations.append(
                (int(slot.data.ptr), int(slot.data.mem.size), self.nodes_so_far())
            )
        cp.copyto(slot, value)
        self.reads.append((slot, apply))


def _phase_name(operation: Any) -> str:
    phase = getattr(operation, "phase", operation)
    name = str(getattr(phase, "value", phase))
    nrk = getattr(operation, "nrk", None)
    return name if nrk is None else f"rk{nrk}.{name}"


class _Entry:
    __slots__ = ("graph", "inventory_in", "inventory_out", "reads", "epilogues", "marks",
                 "fingerprint", "fingerprint_sha256", "nodes", "captured_step", "replays",
                 "confirmed", "since_audit", "audits")

    def __init__(self, **values: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, values.get(name))
        self.replays = 0
        self.since_audit = 0
        self.audits = 0
        self.confirmed = bool(values.get("confirmed", False))


class GraphMemory:
    """The device memory every graph of one stream allocates from.

    One pool for all the runtimes that share the stream (a nest's parent and
    child): their graphs run one after another on it, so the scratch one graph
    frees at its end is scratch the next may use, exactly as the one default
    pool serves both runtimes eagerly -- two pools would hold both peaks.
    ``owners`` are the inventories that must name every live block of it.
    """

    def __init__(self, cupy: Any) -> None:
        self.cp = cupy
        self.pool = cupy.cuda.MemoryPool()
        self.slot_pool = cupy.cuda.MemoryPool()
        with cupy.cuda.using_allocator(self.pool.malloc):
            self.probe = cupy.empty(1, dtype=cupy.float32)
        self.single_pool = self.probe.data.mem.pool()
        self.owners: list["ScheduleGraphs"] = []
        # The pool's in-use bytes when no graph is running: every live block
        # is a named array (checked after each capture).  Anything more at a
        # step boundary is a block held outside the graphs -- a frame's staging
        # not yet released -- that a replay's baked scratch could overwrite.
        self.quiescent_bytes: int | None = None

    def allocating(self) -> Any:
        """Allocate from this pool, between steps (a frame's or a restart's staging).

        The staging is released before the next step (it is off the device
        once the snapshot returns), and while it is not, ``run_step`` sees the
        pool above its quiescent bytes and runs the step in Python.  So the
        graphs' reserved scratch serves the staging too, as the one default
        pool served both eagerly, instead of the staging needing its own.
        """

        return self.cp.cuda.using_allocator(self.pool.malloc)

    def may_capture(self, graphs: "ScheduleGraphs") -> bool:
        """Whether ``graphs`` may capture yet: every owner registered before it has.

        The parent registers first and its step's scratch is the larger; a
        child capturing first would leave the pool holding blocks of the
        child's sizes, and the parent's capture would then add its own beside
        them.  Captured in order, the child's scratch is carved from the
        parent's freed blocks and the pool holds about one step peak, as the
        default pool does eagerly.
        """

        for owner in self.owners:
            if owner is graphs:
                return True
            if owner.counts["captures"] == 0:
                return False
        return True

    def unnamed_bytes(self) -> int:
        """Live bytes of the pool no owner's inventory (or the probe) holds."""

        named = 0
        seen: set[int] = set()
        for graphs in self.owners:
            _, _, owned = device_inventory(
                graphs.roots(), cupy=self.cp, private_pool=self.single_pool,
                stop=graphs.stop, private_seen=seen,
            )
            named += owned
        _, _, probe = device_inventory(
            [("<probe>", self.probe)], cupy=self.cp, private_pool=self.single_pool,
            private_seen=seen,
        )
        return int(self.pool.used_bytes()) - named - probe


class ScheduleGraphs:
    """Capture, cache and replay one runtime's large-step schedule.

    ``roots`` names what the inventory walks (the runtime's attributes);
    ``owner`` is the runtime itself, never re-entered through a back-reference.
    ``memory`` is shared by every runtime on the same stream.
    """

    def __init__(self, *, cupy: Any, stream: Any, mode: str, role: str,
                 roots: Callable[[], Iterable[tuple[str, Any]]], owner: Any = None,
                 memory: GraphMemory | None = None) -> None:
        if mode not in ("on", "verify"):
            raise ValueError(f"cuda graph mode {mode!r} is not one of on, verify")
        self.cp, self.stream, self.mode, self.role = cupy, stream, mode, role
        self.roots = roots
        self.stop = () if owner is None else (owner,)
        self.memory = GraphMemory(cupy) if memory is None else memory
        self.memory.owners.append(self)
        self.pool = self.memory.pool
        self.slot_pool = self.memory.slot_pool
        # The first large step this runtime takes in this process runs
        # eagerly: it builds every lazily created workspace and cache from the
        # default pool, so no capture has to create one the inventory cannot see.
        self.warm_steps = 1
        self.steps_seen = 0
        self.entries: dict[Any, dict[str, _Entry]] = {}
        self.captures_without_replay: dict[Any, int] = {}
        self.confirmation_failures: dict[Any, int] = {}
        self.unstable: dict[Any, str] = {}
        self.eager_reasons: dict[str, int] = {}
        self.counts = {"eager": 0, "captures": 0, "replays": 0, "verified": 0,
                       "verify_mismatches": 0, "inventory_misses": 0,
                       "confirmations": 0, "audits": 0}
        self.mismatches: list[dict[str, Any]] = []
        self.seconds = {"inventory": 0.0, "capture": 0.0,
                        "fingerprint": 0.0, "launch": 0.0, "readback": 0.0}
        self.max_nodes = 0
        # Device seconds of every launched graph, by phase (external event
        # nodes), and in all: the per-phase timing a replay would otherwise lose.
        self.phase_device_seconds: dict[str, float] = {}
        self.graph_device_seconds = 0.0
        self.graph_launches_timed = 0

    # -- the step --------------------------------------------------------
    def run_step(self, backend: Any, step: int, schedule: Iterable[Any]) -> str:
        """Execute one step's schedule and synchronize; return how it ran."""

        schedule = tuple(schedule)
        self.steps_seen += 1
        if self.steps_seen <= self.warm_steps:
            return self._eager(backend, step, schedule, "warm-step")
        key = backend.graph_key(step)
        if isinstance(key, str):
            return self._eager(backend, step, schedule, key)
        if not self.memory.may_capture(self):
            return self._eager(backend, step, schedule, "waiting-for-an-earlier-runtime-to-capture")
        if key in self.unstable:
            return self._eager(backend, step, schedule, "unstable-inventory")
        quiescent = self.memory.quiescent_bytes
        if quiescent is not None and int(self.pool.used_bytes()) != quiescent:
            # Something allocated from the graphs' pool between steps.  An
            # array a runtime now holds (a frame's product planes) is named by
            # the inventory, which has changed, so the step is captured anew
            # around it; a block nothing names (staging not yet released) could
            # be overwritten by a replay's baked scratch, so the step runs in
            # Python until it is gone.
            if self.memory.unnamed_bytes() != 0:
                return self._eager(backend, step, schedule, "graph-pool-held-between-steps")
            self.memory.quiescent_bytes = int(self.pool.used_bytes())
        tick = time.perf_counter()
        inventory = self._inventory()
        self.seconds["inventory"] += time.perf_counter() - tick
        cached = self.entries.get(key, {}).get(inventory)
        if cached is None and key in self.entries:
            self.counts["inventory_misses"] += 1
        # A replay runs no Python, so every host object is left exactly as the
        # capture left it: only a graph that ends on the inventory it began on
        # leaves the host naming the arrays the device state is in.
        replay = (cached is not None and self.mode == "on" and cached.confirmed
                  and cached.inventory_in == cached.inventory_out
                  and cached.since_audit < AUDIT_INTERVAL)
        profiler = getattr(backend, "profiler", None)
        token = None if profiler is None else profiler.phase_begin(
            "cuda-graph.replay" if replay else "cuda-graph.capture"
        )
        try:
            if replay:
                return self._replay(backend, step, key, cached)
            return self._capture(backend, step, schedule, key, inventory, cached)
        finally:
            if token is not None:
                profiler.phase_end(token)

    def _eager(self, backend: Any, step: int, schedule: tuple[Any, ...], reason: str) -> str:
        for operation in schedule:
            backend.execute(operation, step=step)
        backend.synchronize()
        if any(self.entries.values()):
            # The step's scratch came from the default pool, which would keep
            # it cached beside the graphs' own copy of the same scratch: two
            # step peaks resident on a card sized for one.  An eager step
            # among replays is rare (a frame, a window record), so its blocks
            # go back to the driver.
            self.cp.get_default_memory_pool().free_all_blocks()
        self.counts["eager"] += 1
        self.eager_reasons[reason] = self.eager_reasons.get(reason, 0) + 1
        return "eager"

    def _replay(self, backend: Any, step: int, key: Any, entry: _Entry) -> str:
        tick = time.perf_counter()
        entry.graph.launch(stream=self.stream)
        self.seconds["launch"] += time.perf_counter() - tick
        for update in entry.epilogues:
            update(step)
        backend.synchronize()
        self._phase_times(entry.marks)
        self._readback(entry.reads)
        entry.replays += 1
        entry.since_audit += 1
        self.counts["replays"] += 1
        self.captures_without_replay[key] = 0
        return "replay"

    def _phase_times(self, marks: list[tuple[str | None, Any]]) -> None:
        """Fold one launch's per-phase device spans (after its synchronization)."""

        if len(marks) < 2:
            return
        elapsed = self.cp.cuda.get_elapsed_time
        for (phase, start), (_, end) in zip(marks, marks[1:]):
            seconds = elapsed(start, end) / 1000.0
            self.phase_device_seconds[phase] = self.phase_device_seconds.get(phase, 0.0) + seconds
        self.graph_device_seconds += elapsed(marks[0][1], marks[-1][1]) / 1000.0
        self.graph_launches_timed += 1

    def _readback(self, reads: list[tuple[Any, Callable[[Any], None]]]) -> None:
        tick = time.perf_counter()
        for slot, apply in reads:
            apply(slot.get(stream=self.stream))
        self.seconds["readback"] += time.perf_counter() - tick

    def _capture(self, backend: Any, step: int, schedule: tuple[Any, ...], key: Any,
                 inventory: str, previous: _Entry | None) -> str:
        cp, stream = self.cp, self.stream
        if active_capture() is not None:
            raise CudaGraphCaptureError("a schedule capture is already recording on this thread")
        # The default pool may still cache the scratch of the eager steps
        # before this one; the capture takes its own copy from the graphs'
        # pool, and both at once is two step peaks on a card sized for one.
        cp.get_default_memory_pool().free_all_blocks()
        capture = _Capture(self)
        tick = time.perf_counter()
        _LOCAL.capture = capture
        graph = None
        failure: BaseException | None = None
        phase = "begin"
        event_class = cp.cuda.Event
        try:
            with stream:
                cp.cuda.Event = capture.tracked_event_class()
                stream.begin_capture()
                try:
                    with cp.cuda.using_allocator(capture.malloc):
                        for operation in schedule:
                            phase = _phase_name(operation)
                            capture.mark(phase, event_class)
                            backend.execute(operation, step=step)
                        capture.mark(None, event_class)
                except BaseException as error:  # noqa: BLE001 - re-raised below
                    failure = error
                # end_capture also when a phase raised, so the stream is usable
                # for the error path; its graph is then discarded.
                try:
                    graph = stream.end_capture()
                except BaseException as error:  # noqa: BLE001 - re-raised below
                    if failure is None:
                        failure = error
                cp.cuda.Event = event_class
                if failure is None:
                    for event in capture.events:
                        event.record(stream)
        finally:
            cp.cuda.Event = event_class
            _LOCAL.capture = None
        if failure is not None:
            raise CudaGraphCaptureError(
                f"{self.role}: CUDA graph capture of large step {step} failed in "
                f"phase {phase} ({type(failure).__name__}: {failure}); the step's "
                "host state is partly advanced and the run cannot continue from "
                "it. Resume from the last restart with --cuda-graphs off, and "
                "report the phase: every host wait or read in a captured phase "
                "must go through cm1core.cuda_graphs"
            ) from failure
        self.seconds["capture"] += time.perf_counter() - tick
        tick = time.perf_counter()
        # Every private block must be named by an array a runtime on this
        # stream reaches (or the pool's probe).
        unnamed = self.memory.unnamed_bytes()
        inventory_out = self._inventory()
        self.seconds["inventory"] += time.perf_counter() - tick
        rows = None
        digest = None
        nodes = None
        tick = time.perf_counter()
        rows = graph_fingerprint(graph.graph, capture.allocations)
        digest = fingerprint_sha256(rows)
        nodes = len(rows) - 1
        self.seconds["fingerprint"] += time.perf_counter() - tick
        self.max_nodes = max(self.max_nodes, nodes)
        difference = None
        if previous is not None:
            difference = first_difference(previous.fingerprint, rows)
            if difference is not None and len(self.mismatches) < 16:
                self.mismatches.append({
                    "step": step, "key": repr(key), "captured_step": previous.captured_step,
                    "check": ("verify" if self.mode == "verify" else
                              "audit" if previous.confirmed else "confirmation"),
                    **difference,
                })
        if previous is not None and self.mode == "verify":
            self.counts["verified"] += 1
            if difference is not None:
                self.counts["verify_mismatches"] += 1
        tick = time.perf_counter()
        graph.launch(stream=stream)
        self.seconds["launch"] += time.perf_counter() - tick
        backend.synchronize()
        self._phase_times(capture.marks)
        self._readback(capture.reads)
        self.counts["captures"] += 1
        entry = _Entry(
            graph=graph, inventory_in=inventory, inventory_out=inventory_out,
            reads=capture.reads, epilogues=capture.epilogues, marks=capture.marks,
            fingerprint=rows,
            fingerprint_sha256=digest, nodes=nodes, captured_step=step,
        )
        slots = self.entries.setdefault(key, {})
        if unnamed != 0:
            # A block the graph writes is live after the capture and no array
            # the inventory walks holds it: something outside the runtime kept
            # it, and could not be told if a later replay rewrote it.
            self.unstable[key] = (
                f"{unnamed} bytes of the graphs' private pool are live after the "
                "capture of step {step} but held by no array the inventory reaches"
            ).replace("{step}", str(step))
            self.entries.pop(key, None)
            return "capture"
        self.memory.quiescent_bytes = int(self.pool.used_bytes())
        if self.mode == "on" and previous is not None:
            if previous.confirmed:
                # An audit: the graph replayed since the last one against what
                # the Python launches now.
                self.counts["audits"] += 1
                previous.audits += 1
                if difference is not None:
                    raise CudaGraphReplayMismatch(
                        f"{self.role}: the audit capture of large step {step} differs "
                        f"from the graph captured at step {previous.captured_step} and "
                        f"replayed {previous.replays} times ({difference}); up to "
                        f"{previous.since_audit} replayed steps before this one may not "
                        "be the steps the Python schedule would have run. Resume from "
                        "a restart taken before them with --cuda-graphs off"
                    )
                previous.since_audit = 0
                return "capture"
            if difference is not None:
                # Two consecutive captures of the same key disagree.  The first
                # capture after an eager step can meet workspace that step left
                # cached and allocate in another order; the newer capture takes
                # its place and must be matched by the next.  A key that never
                # settles reads host state its key does not carry and is never
                # replayed.
                failures = self.confirmation_failures.get(key, 0) + 1
                self.confirmation_failures[key] = failures
                if failures >= MAX_CONFIRMATION_FAILURES:
                    self.unstable[key] = (
                        f"{failures} consecutive captures differ, the last at node "
                        f"{difference.get('node')} "
                        f"({difference.get('kernel', difference.get('kind'))})"
                    )
                    self.entries.pop(key, None)
                    return "capture"
                slots[inventory] = entry
                return "capture"
            self.counts["confirmations"] += 1
            entry.confirmed = True
        # verify keeps the first capture as the reference every later one is
        # compared with; on keeps the newest (confirmed once it has matched
        # the capture before it).
        if previous is None or self.mode == "on":
            if inventory not in slots and len(slots) >= MAX_ENTRIES_PER_KEY:
                slots.pop(next(iter(slots)))
            slots[inventory] = entry
        misses = self.captures_without_replay.get(key, 0) + 1
        self.captures_without_replay[key] = misses
        if self.mode == "on" and misses > MAX_CAPTURES_WITHOUT_REPLAY:
            self.unstable[key] = (
                f"{misses} captures without a replay: the device inventory changed "
                "at every step boundary"
            )
            self.entries.pop(key, None)
        return "capture"

    def _inventory(self) -> str:
        digest, _, _ = device_inventory(self.roots(), cupy=self.cp, stop=self.stop)
        return digest

    # -- receipt ---------------------------------------------------------
    def record(self) -> dict[str, Any]:
        keys = []
        for key, slots in self.entries.items():
            for entry in slots.values():
                keys.append({
                    "key": repr(key), "captured_step": entry.captured_step,
                    "nodes": entry.nodes, "replays": entry.replays,
                    "confirmed": entry.confirmed, "audits": entry.audits,
                    "fingerprint_sha256": entry.fingerprint_sha256,
                    "deferred_reads": len(entry.reads), "host_updates": len(entry.epilogues),
                })
        return {
            "schema": CUDA_GRAPHS_SCHEMA,
            "role": self.role,
            "mode": self.mode,
            "counts": dict(self.counts),
            "eager_reasons": dict(self.eager_reasons),
            "unstable_keys": {repr(k): v for k, v in self.unstable.items()},
            "verify_mismatches": list(self.mismatches),
            "graphs": keys,
            "max_nodes": self.max_nodes,
            "shared_pool_bytes": int(self.pool.total_bytes()),
            "shared_pool_owners": [graphs.role for graphs in self.memory.owners],
            "warm_steps": self.warm_steps,
            "audit_interval_replays": AUDIT_INTERVAL,
            "host_seconds": {k: round(v, 6) for k, v in self.seconds.items()},
            "graph_launches_timed": self.graph_launches_timed,
            "graph_device_seconds": round(self.graph_device_seconds, 6),
            "phase_device_seconds": {
                k: round(v, 6) for k, v in sorted(
                    self.phase_device_seconds.items(), key=lambda row: -row[1]
                )
            },
        }
