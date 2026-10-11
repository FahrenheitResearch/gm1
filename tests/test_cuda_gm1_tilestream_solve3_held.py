from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import cm1_marks

from authority_trees import requires_cm1_tarball, requires_native_reference
from cm1core import cuda_gm1_tilestream_solve3_held as held
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_solve3_boundaries import (
    Gm1TiledSolve3PreHeldBoundaryToken,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preheld(source: str = "1" * 64, stream: int = 1234, *, real: bool = False):
    receipt = {
        "schema": "gm1.tilestream-solve3-preheld-boundary-token/v1",
        "status": "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY",
        "source_identity_sha256": source,
        "operation_a": {
            "receipt_file_sha256": "2" * 64,
            "receipt_semantic_sha256": "3" * 64,
        },
        "tile_count": 1,
        "tile_plan_sha256": "4" * 64,
        "tile_operation_tokens": ["5" * 64],
        "stream_pointer": stream,
        "source_order": ["operation A", "bcp(ppx)", "bcp(rho)"],
        "outputs": {
            "ppx_padded_sha256": (
                held.GM1_ACCEPTED_PREHELD_PPX_PADDED_SHA256 if real else "6" * 64
            ),
            "rho_padded_sha256": (
                held.GM1_ACCEPTED_PREHELD_RHO_PADDED_SHA256 if real else "7" * 64
            ),
        },
        "ready_for": "held-rate recapture admission only",
        "operation_b_current_to_a_available": False,
        "mpi_performed": False,
        "claim_limit": "compact test token",
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        held._canonical(receipt)
    ).hexdigest()
    return Gm1TiledSolve3PreHeldBoundaryToken(MappingProxyType(receipt))


def _binding(*, nx: int = 17, ny: int = 15, nk: int = 3):
    k, j, i = np.indices((nk, ny, nx), dtype=np.int32)
    pp3d = np.ascontiguousarray(
        np.float32(-0.002)
        + np.multiply(k + 2 * j - i, np.float32(1.0e-6), dtype=np.float32)
    )
    th3d = np.ascontiguousarray(
        np.float32(0.1)
        + np.multiply(2 * k - j + i, np.float32(1.0e-4), dtype=np.float32)
    )
    q3d = np.empty((6, nk, ny, nx), dtype=np.float32)
    for species in range(6):
        q3d[species] = np.float32((species + 1) * 1.0e-5) + np.multiply(
            k + j + 2 * i, np.float32(1.0e-8), dtype=np.float32
        )
    snapshots = {
        "qpten": np.ascontiguousarray(pp3d - np.float32(2.0e-5)),
        "qtten": np.ascontiguousarray(th3d + np.float32(0.03)),
        "qvten": np.ascontiguousarray(q3d[0] - np.float32(3.0e-7)),
        "qcten": np.ascontiguousarray(q3d[1] + np.float32(4.0e-7)),
    }
    return held.Gm1TiledSolve3HeldBinding.create(
        pp3d=pp3d,
        th3d=th3d,
        q3d=np.ascontiguousarray(q3d),
        **snapshots,
        preheld_token=_preheld(),
        dt=0.5,
        source_identity_sha256="1" * 64,
        pdef_apply_receipt_file_sha256="8" * 64,
        pdef_apply_receipt_semantic_sha256="9" * 64,
        post_wsm6_receipt_file_sha256="a" * 64,
        post_wsm6_receipt_semantic_sha256="b" * 64,
        boundary_receipt_file_sha256="c" * 64,
        boundary_receipt_semantic_sha256="d" * 64,
        initial_held_positive_zero_sha256="e" * 64,
        allow_compact_validation=True,
    )


def _fake_launch(spec, binding):
    receipt = {
        "schema": held.CUDA_GM1_TILESTREAM_SOLVE3_HELD_SCHEMA,
        "status": "ENQUEUED_SOURCE_ORDERED_HELD_RECAPTURE_TILE",
        "tile": spec.index,
        "source_identity_sha256": binding.source_identity_sha256,
        "preheld_operation_token_sha256": binding.preheld_token.receipt[
            "operation_token_sha256"
        ],
        "boundary_receipt_file_sha256": binding.boundary_receipt_file_sha256,
        "initial_held_positive_zero_sha256": (
            binding.initial_held_positive_zero_sha256
        ),
        "stream_pointer": binding.preheld_token.receipt["stream_pointer"],
    }
    receipt["operation_token_sha256"] = hashlib.sha256(
        held._canonical(receipt)
    ).hexdigest()
    return held.CudaGm1TiledSolve3HeldLaunch(
        object(), object(), MappingProxyType(receipt)
    )


@requires_cm1_tarball
@requires_native_reference
def test_authority_seals_literal_held_statement_order() -> None:
    assert _sha(ROOT / "upstream/cm1-21.1/cm1r21.1.tar.gz") == (
        held.AUTHORITY_TARBALL_SHA256
    )
    solve3 = ROOT / "work/native-reference/cm1r21.1/src/solve3.F"
    assert _sha(solve3) == held.AUTHORITY_SOLVE3_SHA256
    lines = solve3.read_text(encoding="ascii").splitlines()
    selected = "\n".join(lines[318:337])
    # The four held rates (pressure, theta, qv, qc), each (new - held)*rdt, in this order.
    statements = (
        "8d0c3179a8523858/43",
        "4d2936e86ce6af24/43",
        "6cca61a019a1d807/46",
        "51a0db304bffed91/46",
    )
    positions = [cm1_marks.find(selected, statement) for statement in statements]
    assert min(positions) >= 0
    assert positions == sorted(positions)
    assert cm1_marks.has(selected, "f8f838da0c1017b1/10")


def test_cpu_tiled_held_matches_resident_and_emits_narrow_token() -> None:
    binding = _binding()
    initial = {
        name: getattr(binding, name).copy(order="C")
        for name in ("pp3d", "th3d", "q3d", *held._HELD_ORDER)
    }
    resident = {
        "pp3d": initial["pp3d"].copy(),
        "th3d": initial["th3d"].copy(),
        "qv": initial["q3d"][0].copy(),
        "qc": initial["q3d"][1].copy(),
        **{name: initial[name].copy() for name in held._HELD_ORDER},
    }
    held.cpu_gm1_tiled_solve3_held_inplace(
        resident,
        active_nx=binding.nx,
        active_ny=binding.ny,
        dt=0.5,
    )
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    launches = []
    counts = np.zeros((binding.ny, binding.nx), dtype=np.uint8)
    for spec in plan.specs:
        shapes = held.tiled_solve3_held_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
        )
        tile = {
            name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()
        }
        fill = held.fill_gm1_tiled_solve3_held_source(binding, spec, tile)
        _, active_ny, active_nx = fill["active_shape"]
        held.cpu_gm1_tiled_solve3_held_inplace(
            tile, active_nx=active_nx, active_ny=active_ny, dt=0.5
        )
        scatter = held.scatter_gm1_tiled_solve3_held_output(binding, spec, tile)
        assert scatter["halo_or_history_mutation"] is False
        y0, y1, x0, x1 = spec.owned_rect("mass")
        counts[y0:y1, x0:x1] += 1
        launches.append(_fake_launch(spec, binding))
    assert (int(counts.min()), int(counts.max())) == (1, 1)
    for name in held._HELD_ORDER:
        assert np.array_equal(
            getattr(binding, name).view(np.uint32), resident[name].view(np.uint32)
        )
    token = held.finish_gm1_tiled_solve3_held(binding, plan, launches)
    assert token.receipt["status"] == (
        "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY"
    )
    assert token.receipt["ready_for"] == "momentum current-to-A admission only"
    assert token.receipt["full_operation_b_available"] is False
    assert token.receipt["tke_commit_performed"] is False


def test_alias_nonfinite_bad_dt_and_stale_real_geometry_refuse() -> None:
    binding = _binding()
    kwargs = {
        name: getattr(binding, name).copy(order="C")
        for name in ("pp3d", "th3d", "q3d", *held._HELD_ORDER)
    }
    common = {
        "preheld_token": _preheld(),
        "dt": 0.5,
        "source_identity_sha256": "1" * 64,
        "pdef_apply_receipt_file_sha256": "8" * 64,
        "pdef_apply_receipt_semantic_sha256": "9" * 64,
        "post_wsm6_receipt_file_sha256": "a" * 64,
        "post_wsm6_receipt_semantic_sha256": "b" * 64,
        "boundary_receipt_file_sha256": "c" * 64,
        "boundary_receipt_semantic_sha256": "d" * 64,
        "initial_held_positive_zero_sha256": "e" * 64,
        "allow_compact_validation": True,
    }
    alias = dict(kwargs)
    alias["qpten"] = alias["pp3d"]
    with pytest.raises(ValueError, match="must not alias"):
        held.Gm1TiledSolve3HeldBinding.create(**alias, **common)
    bad = dict(kwargs)
    bad["qcten"] = bad["qcten"].copy()
    bad["qcten"][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        held.Gm1TiledSolve3HeldBinding.create(**bad, **common)
    with pytest.raises(ValueError, match="dt=0.5"):
        held.Gm1TiledSolve3HeldBinding.create(**kwargs, **{**common, "dt": 1.0})
    with pytest.raises(ValueError, match=r"\[75,300,300\]"):
        held.Gm1TiledSolve3HeldBinding.create(
            **kwargs,
            **{
                **common,
                "preheld_token": _preheld(held.GM1_SOURCE_IDENTITY_SHA256, real=True),
                "source_identity_sha256": held.GM1_SOURCE_IDENTITY_SHA256,
                "allow_compact_validation": False,
            },
        )


def test_rotated_preheld_token_and_missing_tile_refuse() -> None:
    token = _preheld()
    changed = dict(token.receipt)
    changed["operation_token_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="operation token changed"):
        held._require_preheld_token(
            Gm1TiledSolve3PreHeldBoundaryToken(MappingProxyType(changed)),
            source_identity_sha256="1" * 64,
            real=False,
        )
    binding = _binding()
    plan = build_gm1_open_tile_plan(nx=binding.nx, ny=binding.ny, tile_nx=8, tile_ny=7)
    with pytest.raises(ValueError, match="every exact tile"):
        held.finish_gm1_tiled_solve3_held(binding, plan, [])


def test_cuda_source_preserves_default_real_statement_order() -> None:
    cuda = held.CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE
    assert cuda.index("pp3d[z],qpten[z]") < cuda.index("th3d[z],qtten[z]")
    assert cuda.index("th3d[z],qtten[z]") < cuda.index("qv[z],qvten[z]")
    assert cuda.index("qv[z],qvten[z]") < cuda.index("qc[z],qcten[z]")
    assert "k*tile_ny+j)*tile_nx+i" in cuda
    assert cuda.count("__fsub_rn") == 4
    assert cuda.count("__fmul_rn") == 4
    assert "atomic" not in cuda
    assert "ppx" not in cuda
    assert hashlib.sha256(cuda.encode("ascii")).hexdigest() == (
        held.CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_SHA256
    )
    source = Path(held.__file__).read_text(encoding="utf-8")
    assert 'method="ptx"' in source and 'method="cubin"' in source


def test_five_meter_inventory_keeps_held_work_tile_local() -> None:
    inventory = held.five_meter_tiled_solve3_held_inventory()
    each = 8 * 75 * 192 * 192 * 4
    assert inventory["host_per_buffer_bytes"] == each
    assert inventory["device_per_buffer_bytes"] == each
    assert inventory["double_host_bytes"] == 2 * each
    assert inventory["double_device_bytes"] == 2 * each
    assert inventory["full_domain_held_snapshot_duplicate_bytes"] == 0
    assert inventory["canonical_store_mutations"] == list(held._HELD_ORDER)
    assert inventory["global_pressure_workspace_executed"] is False
    assert inventory["current_to_a_performed"] is False
    assert inventory["five_meter_meteorological_product"] is False
