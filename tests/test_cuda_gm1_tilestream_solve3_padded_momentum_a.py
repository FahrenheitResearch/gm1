from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_cm1_tarball, requires_native_reference
from cm1core import cuda_gm1_tilestream_solve3_padded_momentum_a as padded
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_solve3_momentum_a import (
    Gm1TiledSolve3PhysicalMomentumAToken,
)


ROOT = Path(__file__).resolve().parents[1]


def _physical_token(source: str = "1" * 64, stream: int = 1234):
    receipt = {
        "schema": "gm1.tilestream-solve3-physical-momentum-a-token/v1",
        "status": "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY",
        "source_identity_sha256": source,
        "held_operation_token_sha256": "2" * 64,
        "held_receipt": {"file_sha256": "3" * 64, "semantic_sha256": "4" * 64},
        "tile_count": 1,
        "tile_plan_sha256": "5" * 64,
        "tile_operation_tokens": ["6" * 64],
        "stream_pointer": stream,
        "outputs": {
            name: character * 64
            for name, character in zip(
                ("u3d", "v3d", "w3d", "ua", "va", "wa"),
                "789abc",
                strict=True,
            )
        },
        "source_order": ["physical momentum A"],
        "ready_for": "horizontal bc2d and padded momentum-A completion only",
        "full_momentum_current_to_a_available": False,
        "tke_commit_available": False,
        "claim_limit": "compact physical token",
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        padded._canonical(receipt)
    ).hexdigest()
    return Gm1TiledSolve3PhysicalMomentumAToken(MappingProxyType(receipt))


def _binding(*, ni: int = 17, nj: int = 15, nk: int = 4):
    shapes = padded.padded_momentum_a_shapes(ni, nj, nk)
    arrays = {}
    for offset, (name, shape) in enumerate(shapes.items()):
        k, j, i = np.indices(shape, dtype=np.int32)
        arrays[name] = np.ascontiguousarray(
            np.add(
                np.float32(10.0 + offset),
                np.multiply(3 * k + 2 * j - i, np.float32(1.0e-4)),
                dtype=np.float32,
            )
        )
    return padded.Gm1TiledSolve3PaddedMomentumABinding.create(
        **arrays,
        physical_token=_physical_token(),
        ni=ni,
        nj=nj,
        nk=nk,
        source_identity_sha256="1" * 64,
        physical_receipt_file_sha256="d" * 64,
        physical_receipt_semantic_sha256="e" * 64,
        allow_compact_validation=True,
    )


def _clone(binding):
    return padded.Gm1TiledSolve3PaddedMomentumABinding.create(
        **{
            name: getattr(binding, name).copy(order="C")
            for name in ("u3d", "v3d", "w3d", "ua", "va", "wa")
        },
        physical_token=binding.physical_token,
        ni=binding.ni,
        nj=binding.nj,
        nk=binding.nk,
        source_identity_sha256=binding.source_identity_sha256,
        physical_receipt_file_sha256=binding.physical_receipt_file_sha256,
        physical_receipt_semantic_sha256=binding.physical_receipt_semantic_sha256,
        allow_compact_validation=True,
    )


def _tile_arrays(spec, nk):
    return {
        name: np.empty(shape, dtype=np.float32)
        for name, shape in padded.tiled_solve3_padded_momentum_a_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
        ).items()
    }


def _fake_launch(binding, spec):
    y0, y1, x0, x1 = spec.owned_rect("mass")
    receipt = {
        "schema": padded.CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SCHEMA,
        "status": "ENQUEUED_FOUR_OPEN_BC2D_AND_PADDED_MOMENTUM_A_TILE",
        "tile": spec.index,
        "source_identity_sha256": binding.source_identity_sha256,
        "physical_operation_token_sha256": binding.physical_token.receipt[
            "operation_token_sha256"
        ],
        "stream_pointer": binding.physical_token.receipt["stream_pointer"],
        "padded_owned_yx": [y0, y1, x0, x1],
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        padded._canonical(receipt)
    ).hexdigest()
    return padded.CudaGm1TiledSolve3PaddedMomentumALaunch(
        object(), object(), MappingProxyType(receipt)
    )


@requires_cm1_tarball
@requires_native_reference
def test_authority_and_literal_source_order_are_sealed() -> None:
    assert (
        hashlib.sha256(
            (ROOT / "upstream/cm1-21.1/cm1r21.1.tar.gz").read_bytes()
        ).hexdigest()
        == padded.AUTHORITY_TARBALL_SHA256
    )
    solve3 = (ROOT / "work/native-reference/cm1r21.1/src/solve3.F").read_text(
        encoding="ascii"
    )
    # The w lower-face fill (bc2d) comes before the joint copy into ua.
    first = cm1_marks.find(solve3, "0a9712d770c9019f/22")
    assert 0 <= first < cm1_marks.find(solve3, "0795fa3b930134c9/20")
    bc = ROOT / "work/native-reference/cm1r21.1/src/bc.F"
    assert hashlib.sha256(bc.read_bytes()).hexdigest() == padded.AUTHORITY_BC_SHA256
    text = bc.read_text(encoding="ascii")
    # The west, east, south and north boundary blocks, in that order.
    sides = [cm1_marks.find(text, mark) for mark in (
        "44722228c7b12b22/22",
        "fde575ba63e11696/22",
        "0d37c373af05b397/23",
        "cbe7cf6d312f849b/23",
    )]
    assert 0 <= sides[0] < sides[1] < sides[2] < sides[3]


def test_cpu_tiled_matches_literal_resident_and_owns_every_padded_word_once() -> None:
    seed = _binding()
    resident, tiled = _clone(seed), _clone(seed)
    initial_wa = tiled.wa.copy(order="C")
    padded.cpu_gm1_solve3_padded_momentum_a_resident_inplace(resident)
    plan = build_gm1_open_tile_plan(
        nx=tiled.ni + 6,
        ny=tiled.nj + 6,
        tile_nx=8,
        tile_ny=7,
    )
    coverage = np.zeros((tiled.nj + 6, tiled.ni + 6), dtype=np.uint8)
    copy_coverage = np.zeros_like(coverage)
    launches = []
    for spec in plan.specs:
        arrays = _tile_arrays(spec, tiled.nk)
        info = padded.fill_gm1_tiled_solve3_padded_momentum_a_source(
            tiled, spec, arrays
        )
        _, active_ny, active_nx = info["active_shape"]
        y0, y1, x0, x1 = info["owned_yx"]
        padded.cpu_gm1_tiled_solve3_padded_momentum_a_inplace(
            arrays,
            active_nx=active_nx,
            active_ny=active_ny,
            global_x0=x0,
            global_y0=y0,
            global_nx=tiled.ni + 6,
            global_ny=tiled.nj + 6,
        )
        padded.scatter_gm1_tiled_solve3_padded_momentum_a_output(tiled, spec, arrays)
        coverage[y0:y1, x0:x1] += 1
        copy_coverage[
            max(y0, 2) : min(y1, tiled.nj + 5), max(x0, 2) : min(x1, tiled.ni + 5)
        ] += 1
        launches.append(_fake_launch(tiled, spec))
    assert (int(coverage.min()), int(coverage.max())) == (1, 1)
    assert np.all(copy_coverage[2 : tiled.nj + 5, 2 : tiled.ni + 5] == 1)
    for name in ("u3d", "v3d", "w3d", "ua", "va", "wa"):
        assert np.array_equal(
            getattr(tiled, name).view(np.uint32),
            getattr(resident, name).view(np.uint32),
        )
    plane = tiled.w3d[1]
    assert all(
        np.array_equal(plane[row].view(np.uint32), plane[3].view(np.uint32))
        for row in range(3)
    )
    assert all(
        np.array_equal(plane[:, column].view(np.uint32), plane[:, 3].view(np.uint32))
        for column in range(3)
    )
    assert np.array_equal(
        tiled.wa[1 : tiled.nk + 1, 2 : tiled.nj + 5, 2 : tiled.ni + 5].view(np.uint32),
        tiled.w3d[1 : tiled.nk + 1, 2 : tiled.nj + 5, 2 : tiled.ni + 5].view(np.uint32),
    )
    assert np.array_equal(tiled.wa[:, 0, 0], initial_wa[:, 0, 0])
    token = padded.finish_gm1_tiled_solve3_padded_momentum_a(tiled, plan, launches)
    assert token.receipt["status"] == "ADMITTED_PADDED_MOMENTUM_A_ONLY"
    assert token.receipt["tke_commit_available"] is False


def test_bc2d_plan_refuses_split_boundary_support_and_realigns_generically() -> None:
    seed = _binding(ni=20, nj=18, nk=4)
    bad = build_gm1_open_tile_plan(nx=26, ny=24, tile_nx=8, tile_ny=7)
    split = bad.specs[2]
    with pytest.raises(
        ValueError, match="east bc2d boundary-support ownership is split"
    ):
        padded.fill_gm1_tiled_solve3_padded_momentum_a_source(
            seed, split, _tile_arrays(split, seed.nk)
        )

    resident, tiled = _clone(seed), _clone(seed)
    padded.cpu_gm1_solve3_padded_momentum_a_resident_inplace(resident)
    plan = padded.build_gm1_solve3_padded_momentum_a_plan(
        ni=tiled.ni,
        nj=tiled.nj,
        tile_nx=8,
        tile_ny=7,
    )
    assert (plan.tile_nx, plan.tile_ny) == (9, 8)
    for spec in plan.specs:
        arrays = _tile_arrays(spec, tiled.nk)
        info = padded.fill_gm1_tiled_solve3_padded_momentum_a_source(
            tiled, spec, arrays
        )
        _, active_ny, active_nx = info["active_shape"]
        y0, _y1, x0, _x1 = info["owned_yx"]
        padded.cpu_gm1_tiled_solve3_padded_momentum_a_inplace(
            arrays,
            active_nx=active_nx,
            active_ny=active_ny,
            global_x0=x0,
            global_y0=y0,
            global_nx=tiled.ni + 6,
            global_ny=tiled.nj + 6,
        )
        padded.scatter_gm1_tiled_solve3_padded_momentum_a_output(tiled, spec, arrays)
    for name in ("u3d", "v3d", "w3d", "ua", "va", "wa"):
        assert np.array_equal(
            getattr(tiled, name).view(np.uint32),
            getattr(resident, name).view(np.uint32),
        )


def test_alias_nonfinite_and_rotated_physical_token_refuse() -> None:
    binding = _binding()
    arrays = {
        name: getattr(binding, name).copy(order="C")
        for name in ("u3d", "v3d", "w3d", "ua", "va", "wa")
    }
    common = {
        "physical_token": _physical_token(),
        "ni": binding.ni,
        "nj": binding.nj,
        "nk": binding.nk,
        "source_identity_sha256": "1" * 64,
        "physical_receipt_file_sha256": "d" * 64,
        "physical_receipt_semantic_sha256": "e" * 64,
        "allow_compact_validation": True,
    }
    alias = dict(arrays)
    alias["ua"] = alias["u3d"]
    with pytest.raises(ValueError, match="must not alias"):
        padded.Gm1TiledSolve3PaddedMomentumABinding.create(**alias, **common)
    bad = {name: value.copy(order="C") for name, value in arrays.items()}
    bad["w3d"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        padded.Gm1TiledSolve3PaddedMomentumABinding.create(**bad, **common)
    receipt = dict(common["physical_token"].receipt)
    receipt["operation_token_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="token identity changed"):
        padded.Gm1TiledSolve3PaddedMomentumABinding.create(
            **arrays,
            **{
                **common,
                "physical_token": Gm1TiledSolve3PhysicalMomentumAToken(
                    MappingProxyType(receipt)
                ),
            },
        )


def test_cuda_has_ordered_bc2d_phases_and_no_later_solve3_state() -> None:
    source = padded.CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE
    assert (
        source.index("gm1_tiled_solve3_bc2d_we")
        < source.index("gm1_tiled_solve3_bc2d_sn")
        < source.index("gm1_tiled_solve3_padded_momentum_a")
    )
    assert "gx>=2u&&gx<global_nx-1u" in source
    assert "gy>=2u&&gy<global_ny-1u" in source
    assert "atomic" not in source
    assert "tkea" not in source
    assert "float*rr" not in source
    assert "radbc" not in source
    assert hashlib.sha256(source.encode("ascii")).hexdigest() == (
        padded.CUDA_GM1_TILESTREAM_SOLVE3_PADDED_MOMENTUM_A_SOURCE_SHA256
    )


def test_five_meter_inventory_remains_only_two_tile_working_sets() -> None:
    inventory = padded.five_meter_tiled_solve3_padded_momentum_a_inventory()
    shapes = padded.tiled_solve3_padded_momentum_a_buffer_shapes(
        tile_nx=192, tile_ny=192, nk=75
    )
    expected = sum(np.prod(shape, dtype=np.int64) * 4 for shape in shapes.values())
    assert inventory["host_per_buffer_bytes"] == expected
    assert inventory["device_per_buffer_bytes"] == expected
    assert inventory["double_host_bytes"] == 2 * expected
    assert inventory["double_device_bytes"] == 2 * expected
    assert inventory["full_domain_duplicate_bytes"] == 0
    assert inventory["global_pressure_workspace_executed"] is False
    assert inventory["tke_commit_executed"] is False
    assert inventory["open_history_mutation"] is False
    assert inventory["five_meter_meteorological_product"] is False
