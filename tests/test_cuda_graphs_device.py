"""CUDA graphs on a device: real capture, real fingerprints, real replay.

``CM1_CUDA_TESTS=1`` under the GPU mutex (tools/perf_probe/with_gpu_mutex.py).
A small schedule of real kernels -- a RawKernel with scalar arguments, CuPy
elementwise kernels (their ``CIndexer`` scratch), a fill, a scratch
allocation freed and taken again inside the step, a deferred device read and
host bookkeeping -- runs through :class:`cm1core.cuda_graphs.ScheduleGraphs`
beside the same schedule run eagerly, and every word must agree.  The
whole-runtime evidence (a nested run's history and restart words with and
without graphs) is the door run in docs/GM1-CARTESIAN-RUNTIME.md.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1", reason="requires coordinated GPU mutex"
)


def _rig():
    import cupy

    from cm1core import cuda_graphs
    from cm1core.cuda_graphs import ScheduleGraphs

    kernel = cupy.RawKernel(
        r'''extern "C" __global__ void axpb(float* x, const float a, const float b, const int n)
        { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = a * x[i] + b; }''',
        "axpb",
    )

    class Backend:
        def __init__(self, stream) -> None:
            self.stream = stream
            self.cp = cupy
            with stream:
                self.x = cupy.arange(1000, dtype=cupy.float32)
                self.counts = cupy.zeros(1, dtype=cupy.int32)
            self.done_steps = 0
            self.maximum = None
            self.profiler = None
            self.b = 1.0

        def graph_key(self, step: int):
            return ("large-step", self.b)

        def execute(self, operation, *, step: int) -> None:
            cp, s = cupy, self.stream
            with s:
                scratch = cp.empty(self.x.shape, dtype=cp.float32)
                kernel((4,), (256,), (self.x, cp.float32(0.5), cp.float32(self.b),
                                      cp.int32(self.x.size)), stream=s)
                cp.multiply(self.x, cp.float32(2.0), out=scratch)
                cp.copyto(self.x, scratch[::-1])
                del scratch
                again = cp.empty((500,), dtype=cp.float32)
                again.fill(cp.float32(operation))
                cp.add(self.x[:500], again, out=self.x[:500])
                peak = cp.max(self.x).reshape(1)
                cuda_graphs.synchronize(s)
                cuda_graphs.read_device(peak, self._set_maximum, stream=s)
                cuda_graphs.bookkeeping(self._set_step, step)

        def _set_maximum(self, value) -> None:
            self.maximum = float(value[0])

        def _set_step(self, step: int) -> None:
            self.done_steps = step

        def synchronize(self) -> None:
            self.stream.synchronize()

        def roots(self):
            return [(".x", self.x), (".counts", self.counts)]

    return cupy, Backend, ScheduleGraphs


SCHEDULE = (1, 2, 3)


def test_replayed_steps_are_the_eager_steps_word_for_word() -> None:
    cupy, Backend, ScheduleGraphs = _rig()
    eager = Backend(cupy.cuda.Stream(non_blocking=True))
    graphed = Backend(cupy.cuda.Stream(non_blocking=True))
    graphs = ScheduleGraphs(cupy=cupy, stream=graphed.stream, mode="on", role="test",
                            roots=graphed.roots, owner=graphed)
    maxima = []
    for step in range(1, 13):
        for operation in SCHEDULE:
            eager.execute(operation, step=step)
        eager.synchronize()
        outcome = graphs.run_step(graphed, step, SCHEDULE)
        maxima.append((outcome, eager.maximum, graphed.maximum))
        assert graphed.done_steps == eager.done_steps == step
        assert graphed.maximum == eager.maximum
        assert bool(cupy.all(graphed.x == eager.x))
    outcomes = [row[0] for row in maxima]
    assert outcomes[:3] == ["eager", "capture", "capture"]
    assert set(outcomes[3:]) == {"replay"}
    record = graphs.record()
    assert record["counts"]["confirmations"] == 1 and not record["unstable_keys"]


def test_verify_mode_finds_consecutive_captures_identical_and_a_scalar_change() -> None:
    cupy, Backend, ScheduleGraphs = _rig()
    backend = Backend(cupy.cuda.Stream(non_blocking=True))
    graphs = ScheduleGraphs(cupy=cupy, stream=backend.stream, mode="verify", role="test",
                            roots=backend.roots, owner=backend)
    for step in range(1, 6):
        graphs.run_step(backend, step, SCHEDULE)
    counts = graphs.record()["counts"]
    assert counts["verified"] == 3 and counts["verify_mismatches"] == 0
    # A scalar the key does not carry: the fingerprint sees the argument bytes.
    backend.graph_key = lambda step: ("large-step", 1.0)
    backend.b = 3.0
    graphs.run_step(backend, 6, SCHEDULE)
    record = graphs.record()
    assert record["counts"]["verify_mismatches"] == 1
    mismatch = record["verify_mismatches"][0]
    assert mismatch["kernel"] == "axpb"
    assert [row["index"] for row in mismatch["arguments"]] == [2]


def test_audit_stops_a_run_whose_replays_would_differ(monkeypatch) -> None:
    cupy, Backend, ScheduleGraphs = _rig()
    from cm1core import cuda_graphs

    monkeypatch.setattr(cuda_graphs, "AUDIT_INTERVAL", 2)
    backend = Backend(cupy.cuda.Stream(non_blocking=True))
    graphs = ScheduleGraphs(cupy=cupy, stream=backend.stream, mode="on", role="test",
                            roots=backend.roots, owner=backend)
    backend.graph_key = lambda step: ("large-step",)
    outcomes = [graphs.run_step(backend, step, SCHEDULE) for step in range(1, 9)]
    assert outcomes == ["eager", "capture", "capture", "replay", "replay", "capture",
                        "replay", "replay"]
    backend.b = 2.0
    with pytest.raises(cuda_graphs.CudaGraphReplayMismatch):
        graphs.run_step(backend, 9, SCHEDULE)


def test_capture_refuses_a_host_wait_outside_the_door() -> None:
    cupy, Backend, ScheduleGraphs = _rig()
    from cm1core import cuda_graphs

    backend = Backend(cupy.cuda.Stream(non_blocking=True))
    graphs = ScheduleGraphs(cupy=cupy, stream=backend.stream, mode="on", role="test",
                            roots=backend.roots, owner=backend)
    graphs.run_step(backend, 1, SCHEDULE)
    original = backend.execute

    def waiting(operation, *, step):
        original(operation, step=step)
        backend.stream.synchronize()

    backend.execute = waiting
    with pytest.raises(cuda_graphs.CudaGraphCaptureError, match="phase 1"):
        graphs.run_step(backend, 2, SCHEDULE)
    # The stream is usable again: a plain eager step runs on it.
    backend.execute = original
    for operation in SCHEDULE:
        backend.execute(operation, step=3)
    backend.synchronize()
