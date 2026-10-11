from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_cartesian_finish as finish
from cm1core.gm1_idealized_residency import DeviceBufferRegistry
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


def test_component_reserve_scales_with_actual_block_and_includes_temporaries():
    grid = CartesianGridGeometry(400, 400, 237, 75, 75, 75)
    small = finish.cartesian_wsm6_scratch_reserve(RuntimeConfig(grid))
    assert small["component_scratch_reserve_bytes"] == 23_314_432
    assert small["expression_temporary_reserve_bytes"] == 7_798_784
    large = finish.cartesian_wsm6_scratch_reserve(RuntimeConfig(grid, tile_nx=400, tile_ny=400))
    assert large["component_scratch_reserve_bytes"] == 910_720_000
    assert large["additional_reserve_bytes"] > 128 * 1024**2
    oversized = finish.cartesian_wsm6_scratch_reserve(RuntimeConfig(grid, tile_nx=800, tile_ny=800))
    assert oversized == large
    with pytest.raises(ValueError, match="alignment"):
        finish.cartesian_wsm6_scratch_reserve(RuntimeConfig(grid), alignment=0)


def test_asq_is_budgeted_and_initialized_on_the_runtime_stream(monkeypatch):
    class Stream:
        active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            return self

        def __exit__(self, *_args):
            self.active = False

    stream = Stream()

    class Backend:
        float32, float64 = np.float32, np.float64

        @staticmethod
        def zeros(shape, dtype):
            assert stream.active, "initialization escaped the nonblocking runtime stream"
            return np.zeros(shape, dtype=dtype)

    cp = Backend()
    config = RuntimeConfig(CartesianGridGeometry(7, 7, 8, 75, 75, 75))
    registry = DeviceBufferRegistry(cp, stream, budget_bytes=1024**2)
    dynamics = SimpleNamespace(config=config, cupy=cp, stream=stream, registry=registry)
    for name in (
        "compile_cuda_apmasscon", "compile_cuda_gm1_forecast",
        "compile_cuda_gm1_rk3_eqtset2_remove_eos", "compile_cuda_wsm6_pdefq",
        "compile_cuda_gm1_tiled_post_wsm6", "compile_cuda_cartesian_moist_heating",
    ):
        monkeypatch.setattr(finish, name, lambda: object())
    # The pdefq workspace is a device allocation made through CuPy directly
    # (de35847); this test runs on a NumPy backend, so it gets a stand-in.
    monkeypatch.setattr(finish, "create_cuda_wsm6_pdefq_workspace", lambda **_kwargs: object())
    monkeypatch.setattr(finish.CartesianFinish, "_initial_mass", lambda _self: 1.0)
    owner = finish.CartesianFinish(dynamics, source_identity_sha256="1" * 64)
    assert registry.buffers["physics.asq"] is owner.asq
    assert owner.asq.shape == (6,) and owner.asq.dtype == np.dtype("float64")
    assert np.all(owner.asq == 0.0)
    assert registry.registered_allocation_bytes >= 512 + sum(value.nbytes for value in owner.persistence.values())
