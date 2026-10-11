from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cm1core.gm1_child_stretched_closure_v3 import (
    CALCNM_KERNEL_SHA256,
    CARRIER_ORDER,
    GENERIC_TURBSMAG_KERNEL_SHA256,
    GRID_5M,
    GRID_50M,
    STATIC_METRIC_KERNEL_SHA256,
    STRETCHED_CALCDEF_KERNEL_SHA256,
    ClosureStreamWriter,
    capacity_record,
    closure_layout,
    closure_payload_nbytes,
    closure_tile_plan,
    inspect_closure_artifact,
)


def _zero_sha256(nbytes: int) -> str:
    digest = hashlib.sha256()
    block = bytes(1 << 20)
    for _ in range(nbytes // len(block)):
        digest.update(block)
    digest.update(block[: nbytes % len(block)])
    return digest.hexdigest()


def test_frozen_closure_layouts_are_exact() -> None:
    assert closure_payload_nbytes(GRID_50M) == 355_320_000
    assert closure_payload_nbytes(GRID_5M) == 35_532_000_000
    for grid in (GRID_50M, GRID_5M):
        layout = closure_layout(grid)
        assert tuple(item.name for item in layout) == CARRIER_ORDER
        assert layout[0].offset_nbytes == 0
        assert all(
            left.offset_nbytes + left.nbytes == right.offset_nbytes
            for left, right in zip(layout, layout[1:])
        )
        assert layout[-1].offset_nbytes + layout[-1].nbytes == (
            closure_payload_nbytes(grid)
        )
        assert layout[3].shape == (grid.nz, grid.ny, grid.nx)
        assert all(
            item.shape == (grid.nz + 1, grid.ny, grid.nx)
            for index, item in enumerate(layout)
            if index != 3
        )


def test_operator_source_pins_match_live_producers() -> None:
    from cm1core.cuda_gm1_child_nm_def import (
        CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
    )
    from cm1core.cuda_gm1_child_stretched_nm_def import (
        CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
    )
    from cm1core.cuda_gm1_child_turbsmag_init_generic import (
        CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
    )
    from cm1core.cuda_stretch_z1_terrain import (
        CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
    )

    assert CALCNM_KERNEL_SHA256 == CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256
    assert (
        STRETCHED_CALCDEF_KERNEL_SHA256
        == CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256
    )
    assert (
        GENERIC_TURBSMAG_KERNEL_SHA256
        == CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256
    )
    assert STATIC_METRIC_KERNEL_SHA256 == CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256


def test_capacity_is_single_copy_and_fail_closed() -> None:
    required = closure_payload_nbytes(GRID_5M)
    receipt = capacity_record(
        GRID_5M, free_bytes=required + 9_000_000_000, reserve_bytes=8_000_000_000
    )
    assert receipt["post_publish_above_reserve_bytes"] == 1_000_000_000
    with pytest.raises(ValueError, match="exactly one"):
        capacity_record(
            GRID_5M,
            free_bytes=required * 3,
            reserve_bytes=0,
            planned_payload_copies=2,
        )
    with pytest.raises(OSError, match="only"):
        capacity_record(
            GRID_5M, free_bytes=required, reserve_bytes=1
        )


def test_tile_plan_owns_every_mass_cell_once() -> None:
    for grid in (GRID_50M, GRID_5M):
        plan = closure_tile_plan(grid)
        counts = np.zeros((grid.ny, grid.nx), np.uint8)
        for spec in plan.specs:
            y0, y1, x0, x1 = spec.owned_rect("mass")
            counts[y0:y1, x0:x1] += 1
        assert np.all(counts == 1)
        assert plan.halo == 3
    assert closure_tile_plan(GRID_5M).ntiles == 256


def test_atomic_writer_and_independent_inspector(tmp_path: Path) -> None:
    # Exercise the real 50 m schema with highly compressible all-zero pages.
    # The writer still visits every owned word and the inspector hashes all
    # 355,320,000 bytes; no synthetic meteorological claim is made.
    destination = tmp_path / "closure"
    plan = closure_tile_plan(GRID_50M)
    free = int(__import__("shutil").disk_usage(tmp_path).free)
    writer = ClosureStreamWriter(
        destination,
        grid=GRID_50M,
        plan=plan,
        input_pins={"fixture": "0" * 64},
        producer={"source_sha256": "1" * 64},
        free_bytes=free,
        reserve_bytes=0,
    )
    for spec in plan.specs:
        y0, y1, x0, x1 = spec.owned_rect("mass")
        height, width = y1 - y0, x1 - x0
        arrays = {
            name: np.zeros(
                ((GRID_50M.nz if name == "grdscl" else GRID_50M.nz + 1), height, width),
                np.float32,
            )
            for name in CARRIER_ORDER
        }
        writer.write_tile(spec, arrays)
    pins = writer.finish()
    assert destination.is_dir()
    assert sorted(path.name for path in destination.iterdir()) == [
        "closure.f32le",
        "manifest.json",
        "receipt.json",
    ]
    assert pins["payload_sha256"] == _zero_sha256(
        closure_payload_nbytes(GRID_50M)
    )
    assert inspect_closure_artifact(
        destination,
        grid=GRID_50M,
        expected_manifest_sha256=pins["manifest_sha256"],
        expected_payload_sha256=pins["payload_sha256"],
        expected_receipt_sha256=pins["receipt_sha256"],
    )["action"] == "inspected"
    manifest = json.loads((destination / "manifest.json").read_bytes())
    assert manifest["historical_uniform_closure_disposition"].startswith(
        "immutable historical evidence"
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        ClosureStreamWriter(
            destination,
            grid=GRID_50M,
            plan=plan,
            input_pins={"fixture": "0" * 64},
            producer={"source_sha256": "1" * 64},
            free_bytes=int(__import__("shutil").disk_usage(tmp_path).free),
            reserve_bytes=0,
        )


def test_free_space_is_the_smaller_of_the_declaration_and_the_observation(
    tmp_path: Path,
) -> None:
    """A live filesystem does not hold still between two samples.

    The writer used to require the caller's ``free_bytes`` to equal its own
    observation exactly, which refused a correct caller whenever anything else
    on the host wrote a byte in between -- and named no wrong result, only a
    disagreement. The result worth preventing is sizing the payload against
    room the destination volume does not have, so the decision is the smaller
    of the two figures and the receipt records both.
    """

    plan = closure_tile_plan(GRID_50M)
    observed = int(__import__("shutil").disk_usage(tmp_path).free)
    payload = closure_payload_nbytes(GRID_50M)

    def writer(destination: Path, free_bytes: int) -> ClosureStreamWriter:
        return ClosureStreamWriter(
            destination,
            grid=GRID_50M,
            plan=plan,
            input_pins={"fixture": "0" * 64},
            producer={"source_sha256": "1" * 64},
            free_bytes=free_bytes,
            reserve_bytes=0,
        )

    # Skew in either direction is admitted: a sample taken before something
    # else freed space, and one taken before something else consumed it.
    for offset, name in ((-(1 << 20), "stale-low"), (1 << 20, "stale-high")):
        opened = writer(tmp_path / name, observed + offset)
        assert opened.capacity["declared_free_bytes"] == observed + offset
        assert opened.capacity["observed_free_bytes"] == pytest.approx(
            observed, abs=1 << 30
        )
        assert opened.capacity["free_bytes"] <= opened.capacity["observed_free_bytes"]
        opened.staging.joinpath("closure.f32le").unlink()
        opened.staging.rmdir()

    # A declaration the destination volume cannot honour is still refused, and
    # the refusal names the byte counts.
    with pytest.raises(OSError, match=f"needs {payload} bytes"):
        writer(tmp_path / "too-small", payload - 1)
