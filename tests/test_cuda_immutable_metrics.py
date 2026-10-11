from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cuda_immutable_metrics import (
    ImmutableMetricMap, _PROVEN, allows_compact_coordinate,
    immutable_metric_subset, require_cuda_argument_aliases, upload_immutable_source_metrics,
)
from cm1core.gm1_idealized_residency import DeviceBufferRegistry, IdealizedSource
from cm1core.grid_geometry import CartesianGridGeometry


class MetricOwner:
    def __init__(self, _library, _sounding, *, iinit, geometry):
        nz, ny, nx = geometry.shape_zyx
        mass, face = (nz, ny, nx), (nz + 1, ny, nx)
        self.groups = {"terrain_static": {
            **{name: np.ones(mass, dtype=np.float32) for name in ("mh", "rmh")},
            **{name: np.ones(face, dtype=np.float32) for name in ("mf", "rmf")},
            **{name: np.full(face, 0.5, dtype=np.float32) for name in ("c1", "c2")},
            **{name: np.zeros(face, dtype=np.float32) for name in ("gx", "gy")},
            "gxu": np.zeros((nz + 1, ny, nx + 1), dtype=np.float32),
            "gyv": np.zeros((nz + 1, ny + 1, nx), dtype=np.float32),
            "dzdx": np.zeros((ny, nx), dtype=np.float32),
            "dzdy": np.full((ny, nx), -0.0, dtype=np.float32),
            "zh": np.broadcast_to(np.arange(nz, dtype=np.float32)[:, None, None], mass).copy(),
            "zf": np.broadcast_to(np.arange(nz + 1, dtype=np.float32)[:, None, None], face).copy(),
        }}

    def close(self):
        pass


class Stream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def synchronize(self):
        pass


class Storage:
    float32 = np.float32
    broadcast_to = staticmethod(np.broadcast_to)

    @staticmethod
    def asarray(value, *, dtype, order):
        return np.array(value, dtype=dtype, order=order, copy=True)


def proof_map():
    registry = DeviceBufferRegistry(Storage(), Stream(), budget_bytes=10_000_000)
    with IdealizedSource("unused", b"words", CartesianGridGeometry(9, 11, 8, 75, 75, 75), owner_factory=MetricOwner) as source:
        host = source.groups["terrain_static"]
        metrics = upload_immutable_source_metrics(source, registry)
        for name in host:
            assert np.array_equal(host[name].view(np.uint32), metrics[name].view(np.uint32))
    return metrics, registry


def test_source_word_proof_shares_constants_preserves_signed_zero_and_profiles():
    metrics, registry = proof_map()
    assert np.shares_memory(metrics["mh"], metrics["mf"])
    assert np.shares_memory(metrics["c1"], metrics["c2"])
    assert np.shares_memory(metrics["gx"], metrics["gxu"])
    assert not np.shares_memory(metrics["dzdx"], metrics["dzdy"])
    assert not metrics["zh"].flags.c_contiguous
    assert allows_compact_coordinate(metrics, "zh", metrics["zh"])
    assert not allows_compact_coordinate(dict(metrics), "zh", metrics["zh"])
    assert registry.registered_allocation_bytes < sum(value.nbytes for value in metrics.values())


def test_alias_gate_requires_exact_source_objects_and_never_allows_output_aliases():
    host_metrics, _ = proof_map()
    # Pointer-only CUDA doubles preserve the real host sharing relationships.
    values = {name: SimpleNamespace(data=SimpleNamespace(ptr=int(value.ctypes.data))) for name, value in host_metrics.items()}
    metrics = ImmutableMetricMap(values, host_metrics.proofs, _PROVEN)
    args = {"metric.c1": metrics["c1"], "metric.c2": metrics["c2"]}
    require_cuda_argument_aliases(args, immutable_metrics=metrics)
    subset = immutable_metric_subset(metrics, {name: metrics[name] for name in ("c1", "c2")})
    require_cuda_argument_aliases(args, immutable_metrics=subset)
    with pytest.raises(ValueError, match="must not alias"):
        require_cuda_argument_aliases(args, immutable_metrics=dict(metrics))
    with pytest.raises(ValueError, match="must not alias"):
        require_cuda_argument_aliases({**args, "state.output": metrics["c1"]}, immutable_metrics=metrics)
    foreign = SimpleNamespace(data=SimpleNamespace(ptr=metrics["c2"].data.ptr))
    with pytest.raises(ValueError, match="must not alias"):
        require_cuda_argument_aliases({"metric.c1": metrics["c1"], "metric.c2": foreign}, immutable_metrics=metrics)
