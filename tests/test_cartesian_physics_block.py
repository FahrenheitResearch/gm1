"""The WSM6 block is sized by WSM6, not by the closure tile.

CM1 calls WSM6 once per physical column (``src/mp_driver.F:1469-1479``, inside
``src/cm1.F:3405-3432``'s completed-step microphysics), and no column reads
another.  So the rectangle this port stages per call has no stencil and no
halo, and its size decides only how many times the per-call staging, launch
and drain are paid and how much scratch one call needs.

It was taking the closure tile's extents, which are sized by a fifth-order
stencil for an entirely different operator.  These tests pin the two apart,
and pin that a document which declares neither key runs exactly the
decomposition and the scratch reserve it ran before.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cm1core.cartesian_door import build_runtime_config
from cm1core.config import load_config, parse_mapping
from cm1core.cuda_cartesian_finish import (
    cartesian_physics_blocks,
    cartesian_wsm6_scratch_reserve,
)
from cm1core.errors import CM1Refusal
from cm1core.gm1_runtime import MAX_PHYSICS_BLOCK_CELLS, RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"


def _mapping() -> dict[str, object]:
    payload = load_config(EXAMPLE).canonical_dict()
    return {
        section: copy.deepcopy(payload[section])
        for section in (
            "run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime"
        )
    }


def _grid() -> CartesianGridGeometry:
    return CartesianGridGeometry(448, 448, 160, 125.0, 125.0, 125.0)


def test_an_undeclared_block_follows_the_closure_tile_exactly() -> None:
    config = RuntimeConfig(_grid())
    assert (config.physics_block_nx, config.physics_block_ny) == (
        config.tile_nx,
        config.tile_ny,
    )
    assert config.record()["physics_block_follows_closure_tile"] is True
    assert len(cartesian_physics_blocks(config)) == 7 * 7 == 49


def test_declaring_a_block_changes_the_decomposition_and_nothing_else() -> None:
    grid = _grid()
    tiled = RuntimeConfig(grid)
    wide = RuntimeConfig(grid, physics_block_nx=224, physics_block_ny=224)
    assert (wide.tile_nx, wide.tile_ny) == (tiled.tile_nx, tiled.tile_ny)
    assert wide.record()["physics_block_follows_closure_tile"] is False
    assert len(cartesian_physics_blocks(wide)) == 4
    assert wide.record()["physics_block_count"] == 4

    # Whatever the decomposition, it covers the domain exactly once.
    for config in (tiled, wide):
        covered = set()
        for j0, j1, i0, i1 in cartesian_physics_blocks(config):
            for j in range(j0, j1):
                for i in range(i0, i1):
                    assert (j, i) not in covered
                    covered.add((j, i))
        assert len(covered) == grid.ny * grid.nx


def test_a_block_wider_than_the_grid_is_one_block() -> None:
    grid = _grid()
    config = RuntimeConfig(grid, physics_block_nx=4096, physics_block_ny=4096)
    assert cartesian_physics_blocks(config) == ((0, grid.ny, 0, grid.nx),)
    assert config.record()["physics_block_shape_yx"] == [grid.ny, grid.nx]


def test_the_scratch_reserve_is_priced_against_the_block_not_the_tile() -> None:
    grid = _grid()
    tiled = cartesian_wsm6_scratch_reserve(RuntimeConfig(grid))
    wide = cartesian_wsm6_scratch_reserve(
        RuntimeConfig(grid, physics_block_nx=224, physics_block_ny=224)
    )
    assert tiled["block_nx"] == 64 and wide["block_nx"] == 224
    assert wide["component_scratch_reserve_bytes"] > tiled["component_scratch_reserve_bytes"]
    # A declared block that happens to equal the tile prices identically, so
    # the pre-existing reserve is unchanged word for word.
    same = cartesian_wsm6_scratch_reserve(
        RuntimeConfig(grid, physics_block_nx=64, physics_block_ny=64)
    )
    assert same == tiled


def test_the_runtime_contract_refuses_an_unpriceable_block() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="pre-WSM6 seam"):
        RuntimeConfig(grid, physics_block_nx=MAX_PHYSICS_BLOCK_CELLS + 1)
    for bad in (0, -8, 8.0, True):
        with pytest.raises(ValueError):
            RuntimeConfig(grid, physics_block_ny=bad)


def test_the_document_carries_the_block_to_the_runtime() -> None:
    raw = _mapping()
    raw["runtime"]["physics_block_nx"] = 96  # type: ignore[index]
    raw["runtime"]["physics_block_ny"] = 48  # type: ignore[index]
    config = parse_mapping(raw)
    assert config.runtime is not None
    assert (config.runtime.physics_block_nx, config.runtime.physics_block_ny) == (96, 48)
    runtime_config = build_runtime_config(config)
    assert (runtime_config.physics_block_nx, runtime_config.physics_block_ny) == (96, 48)
    grid = runtime_config.grid
    assert len(cartesian_physics_blocks(runtime_config)) == (
        -(-grid.ny // 48) * -(-grid.nx // 96)
    )
    assert runtime_config.record()["physics_block_shape_yx"] == [48, 96]


def test_an_undeclared_block_leaves_the_sealed_document_identity_alone() -> None:
    config = load_config(EXAMPLE)
    assert config.runtime is not None
    assert config.runtime.physics_block_nx is None
    assert config.runtime.physics_block_ny is None
    projected = config.canonical_dict()["runtime"]
    assert "physics_block_nx" not in projected  # type: ignore[operator]
    assert "physics_block_ny" not in projected  # type: ignore[operator]
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )
    runtime_config = build_runtime_config(config)
    assert runtime_config.physics_block_nx == runtime_config.tile_nx
    assert runtime_config.physics_block_ny == runtime_config.tile_ny


@pytest.mark.parametrize(
    ("value", "code"),
    (
        (0, "CM1-CFG-INTEGER"),
        (-1, "CM1-CFG-INTEGER"),
        (64.0, "CM1-CFG-INTEGER"),
        (True, "CM1-CFG-INTEGER"),
        (MAX_PHYSICS_BLOCK_CELLS + 1, "CM1-CFG-UNSUPPORTED-VALUE"),
    ),
)
def test_the_document_refuses_a_block_by_key_and_value(value: object, code: str) -> None:
    raw = _mapping()
    raw["runtime"]["physics_block_nx"] = value  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == code
    assert caught.value.detail.option == "runtime.physics_block_nx"


def test_the_registry_drains_the_stream_before_it_drops_a_buffer() -> None:
    """The invariant the block loop's second barrier was duplicating.

    ``microphysics`` and ``_drop_padded`` used to synchronize the stream
    themselves and then call ``registry.release``, which synchronizes again --
    once per WSM6 block, so 49 extra device-wide barriers a step on a 448x448
    grid with the default block.  Dropping the caller's barrier is only safe
    because ``release`` takes one first and nothing allocates in between; this
    pins that ordering so the barrier cannot quietly move.
    """

    import numpy as np

    from cm1core.gm1_idealized_residency import DeviceBufferRegistry

    events: list[str] = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def synchronize(self) -> None:
            events.append("synchronize")

    class Tracked(np.ndarray):
        def __del__(self):
            events.append("freed")

    class FakeCupy:
        float32 = np.float32

        @staticmethod
        def empty(shape, dtype=None):
            return np.empty(shape, dtype=dtype).view(Tracked)

    registry = DeviceBufferRegistry(FakeCupy(), Stream(), budget_bytes=1 << 20)
    registry.allocate("block.value", (4, 4))
    events.clear()
    registry.release(["block.value"])
    assert events[0] == "synchronize"
    assert "freed" not in events[:1]
