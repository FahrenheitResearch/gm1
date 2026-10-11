from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_cm1_tarball, requires_native_reference
from cm1core import cuda_gm1_tilestream_solve3_momentum_a as momentum
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_solve3_held import Gm1TiledSolve3HeldToken


ROOT = Path(__file__).resolve().parents[1]


def _held(source: str = "1" * 64, stream: int = 1234, *, real: bool = False):
    receipt = {
        "schema": "gm1.tilestream-solve3-held-token/v1",
        "status": "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY",
        "source_identity_sha256": source,
        "preheld_operation_token_sha256": "2" * 64,
        "boundary_receipt": {
            "file_sha256": "3" * 64,
            "semantic_sha256": "4" * 64,
        },
        "initial_held_positive_zero_sha256": "5" * 64,
        "tile_count": 1,
        "tile_plan_sha256": "6" * 64,
        "tile_operation_tokens": ["7" * 64],
        "stream_pointer": stream,
        "source_order": ["held recapture"],
        "outputs": (
            dict(momentum.GM1_ACCEPTED_HELD_OUTPUT_SHA256)
            if real
            else {
                name: character * 64
                for name, character in zip(
                    ("qpten", "qtten", "qvten", "qcten"), "89ab", strict=True
                )
            }
        ),
        "ready_for": "momentum current-to-A admission only",
        "full_operation_b_available": False,
        "tke_commit_performed": False,
        "claim_limit": "compact held token",
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        momentum._canonical(receipt)
    ).hexdigest()
    return Gm1TiledSolve3HeldToken(MappingProxyType(receipt))


def _binding(*, nx: int = 17, ny: int = 15, nk: int = 4):
    k, j, i = np.indices((nk, ny, nx + 1), dtype=np.int32)
    u = np.ascontiguousarray(
        np.add(
            np.float32(3.0),
            np.multiply(k + 2 * j - i, np.float32(1.0e-3)),
            dtype=np.float32,
        )
    )
    k, j, i = np.indices((nk, ny + 1, nx), dtype=np.int32)
    v = np.ascontiguousarray(
        np.add(
            np.float32(-2.0),
            np.multiply(2 * k - j + i, np.float32(2.0e-3)),
            dtype=np.float32,
        )
    )
    k, j, i = np.indices((nk + 1, ny, nx), dtype=np.int32)
    w = np.ascontiguousarray(
        np.add(
            np.float32(0.25),
            np.multiply(k + j - i, np.float32(3.0e-4)),
            dtype=np.float32,
        )
    )
    jj, ii = np.indices((ny, nx), dtype=np.int32)
    gz = np.ascontiguousarray(
        np.add(
            np.float32(1.01),
            np.multiply(ii + jj, np.float32(1.0e-5)),
            dtype=np.float32,
        )
    )
    dzdx = np.ascontiguousarray(
        np.multiply(ii - jj, np.float32(3.0e-4), dtype=np.float32)
    )
    dzdy = np.ascontiguousarray(
        np.multiply(ii + 2 * jj, np.float32(-2.0e-4), dtype=np.float32)
    )
    return momentum.Gm1TiledSolve3MomentumABinding.create(
        u3d=u,
        v3d=v,
        w3d=w,
        ua=np.full_like(u, np.float32(-91.0)),
        va=np.full_like(v, np.float32(-92.0)),
        wa=np.full_like(w, np.float32(-93.0)),
        gz=gz,
        dzdx=dzdx,
        dzdy=dzdy,
        held_token=_held(),
        source_identity_sha256="1" * 64,
        held_receipt_file_sha256="c" * 64,
        held_receipt_semantic_sha256="d" * 64,
        allow_compact_validation=True,
    )


def _tile_arrays(spec, nk):
    return {
        name: np.empty(shape, dtype=np.float32)
        for name, shape in momentum.tiled_solve3_momentum_a_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
        ).items()
    }


def _fake_launch(spec, binding):
    y0, y1, x0, x1 = spec.owned_rect("mass")
    receipt = {
        "schema": momentum.CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SCHEMA,
        "status": "ENQUEUED_TERRAIN_FINALIZE_AND_PHYSICAL_MOMENTUM_A_TILE",
        "tile": spec.index,
        "source_identity_sha256": binding.source_identity_sha256,
        "held_operation_token_sha256": binding.held_token.receipt[
            "operation_token_sha256"
        ],
        "stream_pointer": binding.held_token.receipt["stream_pointer"],
        "mass_owned_yx": [y0, y1, x0, x1],
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        momentum._canonical(receipt)
    ).hexdigest()
    return momentum.CudaGm1TiledSolve3MomentumALaunch(
        object(), object(), MappingProxyType(receipt)
    )


@requires_cm1_tarball
@requires_native_reference
def test_authority_and_source_order_cover_terrain_then_joint_copy() -> None:
    assert (
        hashlib.sha256(
            (ROOT / "upstream/cm1-21.1/cm1r21.1.tar.gz").read_bytes()
        ).hexdigest()
        == momentum.AUTHORITY_TARBALL_SHA256
    )
    solve3 = (ROOT / "work/native-reference/cm1r21.1/src/solve3.F").read_text(
        encoding="ascii"
    )
    # The terrain lower boundary (bcwsfc) comes before the joint u, v, w copy into ua, va, wa.
    order = [cm1_marks.find(solve3, mark) for mark in (
        "cad570f249234630/36",
        "0795fa3b930134c9/20",
        "ec6383785e7d9051/20",
        "ecd850100ed8e301/20",
    )]
    assert 0 <= order[0] < order[1] < order[2] < order[3]
    bc = ROOT / "work/native-reference/cm1r21.1/src/bc.F"
    assert hashlib.sha256(bc.read_bytes()).hexdigest() == momentum.AUTHORITY_BC_SHA256


def test_cpu_tiled_matches_resident_with_exact_staggered_ownership() -> None:
    binding = _binding()
    initial = {
        name: getattr(binding, name).copy(order="C")
        for name in ("u3d", "v3d", "w3d", "ua", "va", "wa")
    }
    resident = _binding(nx=binding.nx, ny=binding.ny, nk=binding.nk)
    full = {
        name: getattr(resident, name)
        for name in momentum.tiled_solve3_momentum_a_buffer_shapes(
            tile_nx=binding.nx,
            tile_ny=binding.ny,
            nk=binding.nk,
        )
    }
    momentum.cpu_gm1_tiled_solve3_momentum_a_inplace(
        full,
        active_nx=binding.nx,
        active_ny=binding.ny,
        active_u_owned_nx=binding.nx + 1,
        active_v_owned_ny=binding.ny + 1,
    )
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    count_u = np.zeros((binding.ny, binding.nx + 1), dtype=np.uint8)
    count_v = np.zeros((binding.ny + 1, binding.nx), dtype=np.uint8)
    count_w = np.zeros((binding.ny, binding.nx), dtype=np.uint8)
    launches = []
    for spec in plan.specs:
        arrays = _tile_arrays(spec, binding.nk)
        info = momentum.fill_gm1_tiled_solve3_momentum_a_source(binding, spec, arrays)
        momentum.cpu_gm1_tiled_solve3_momentum_a_inplace(
            arrays,
            active_nx=info["active_mass_shape"][2],
            active_ny=info["active_mass_shape"][1],
            active_u_owned_nx=info["active_u_owned_nx"],
            active_v_owned_ny=info["active_v_owned_ny"],
        )
        scatter = momentum.scatter_gm1_tiled_solve3_momentum_a_output(
            binding, spec, arrays
        )
        y0, y1, x0, x1 = scatter["mass_owned_yx"]
        count_w[y0:y1, x0:x1] += 1
        uy0, uy1, ux0, ux1 = scatter["u_owned_yx"]
        count_u[uy0:uy1, ux0:ux1] += 1
        vy0, vy1, vx0, vx1 = scatter["v_owned_yx"]
        count_v[vy0:vy1, vx0:vx1] += 1
        launches.append(_fake_launch(spec, binding))
    assert all(
        (int(c.min()), int(c.max())) == (1, 1) for c in (count_u, count_v, count_w)
    )
    for name in ("u3d", "v3d", "w3d", "ua", "va", "wa"):
        assert np.array_equal(
            getattr(binding, name).view(np.uint32),
            getattr(resident, name).view(np.uint32),
        )
    assert np.array_equal(binding.ua.view(np.uint32), binding.u3d.view(np.uint32))
    assert np.array_equal(binding.va.view(np.uint32), binding.v3d.view(np.uint32))
    assert np.array_equal(
        binding.wa[:-1].view(np.uint32), binding.w3d[:-1].view(np.uint32)
    )
    assert np.array_equal(
        binding.wa[-1].view(np.uint32), initial["wa"][-1].view(np.uint32)
    )
    assert not np.array_equal(
        binding.w3d[0].view(np.uint32), initial["w3d"][0].view(np.uint32)
    )
    token = momentum.finish_gm1_tiled_solve3_momentum_a(binding, plan, launches)
    assert token.receipt["status"] == "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY"
    assert token.receipt["full_momentum_current_to_a_available"] is False
    assert token.receipt["tke_commit_available"] is False


def test_alias_bad_held_token_and_nonfinite_refuse() -> None:
    binding = _binding()
    kwargs = {
        name: getattr(binding, name).copy(order="C")
        for name in ("u3d", "v3d", "w3d", "ua", "va", "wa", "gz", "dzdx", "dzdy")
    }
    common = {
        "held_token": _held(),
        "source_identity_sha256": "1" * 64,
        "held_receipt_file_sha256": "c" * 64,
        "held_receipt_semantic_sha256": "d" * 64,
        "allow_compact_validation": True,
    }
    alias = dict(kwargs)
    alias["ua"] = alias["u3d"]
    with pytest.raises(ValueError, match="must not alias"):
        momentum.Gm1TiledSolve3MomentumABinding.create(**alias, **common)
    bad = {name: value.copy(order="C") for name, value in kwargs.items()}
    bad["gz"][0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        momentum.Gm1TiledSolve3MomentumABinding.create(**bad, **common)
    token = _held()
    changed = dict(token.receipt)
    changed["operation_token_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="operation identity changed"):
        momentum.Gm1TiledSolve3MomentumABinding.create(
            **kwargs,
            **{
                **common,
                "held_token": Gm1TiledSolve3HeldToken(MappingProxyType(changed)),
            },
        )


def test_cuda_source_avoids_bcwsfc_to_wa_race_and_refuses_later_state() -> None:
    source = momentum.CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE
    assert source.index("w3d[zw]=bottom") < source.index("wa[zw]=bottom")
    assert "(nk-1)*active_ny*active_nx" in source
    assert "k=1u+" in source
    assert source.count("__fadd_rn") == 3
    assert source.count("__fmul_rn") == 4
    assert "atomic" not in source
    assert "tkea" not in source
    assert "float*rr" not in source
    assert "rr[" not in source
    assert hashlib.sha256(source.encode("ascii")).hexdigest() == (
        momentum.CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_SHA256
    )


def test_five_meter_inventory_remains_two_tile_buffers_only() -> None:
    inventory = momentum.five_meter_tiled_solve3_momentum_a_inventory()
    shapes = momentum.tiled_solve3_momentum_a_buffer_shapes(
        tile_nx=192, tile_ny=192, nk=75
    )
    expected = sum(np.prod(shape, dtype=np.int64) * 4 for shape in shapes.values())
    assert inventory["host_per_buffer_bytes"] == expected
    assert inventory["device_per_buffer_bytes"] == expected
    assert inventory["double_host_bytes"] == 2 * expected
    assert inventory["double_device_bytes"] == 2 * expected
    assert inventory["full_domain_duplicate_bytes"] == 0
    assert inventory["padded_halo_completion_executed"] is False
    assert inventory["tke_commit_executed"] is False
    assert inventory["global_pressure_workspace_executed"] is False
    assert inventory["five_meter_meteorological_product"] is False
