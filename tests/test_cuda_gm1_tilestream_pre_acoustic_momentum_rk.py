from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from cm1core import cuda_gm1_tilestream_pre_acoustic_momentum_rk as pre
from cm1core import cuda_gm1_tilestream_rk1_acoustic as acoustic
from cm1core.cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
)
from cm1core.cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    Gm1TiledSolve1FullHeldToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: dict[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _solve1_tke_receipt(*, stream_pointer: int = 31) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
        "completed_nrk": 0,
        "stream_pointer": stream_pointer,
        "output_tketen_shape": [76, 300, 300],
        "output_tketen_words": 76 * 300 * 300,
        "output_tketen_sha256": "1" * 64,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return receipt


def _full_held_receipt(
    *, stream_pointer: int = 31, solve1_tke_sha256: str | None = None
) -> dict[str, object]:
    shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "thten1": [75, 300, 300],
        "qten": [6, 75, 300, 300],
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "stream_pointer": stream_pointer,
        "solve1_tke_operation_token_sha256": (
            _solve1_tke_receipt(stream_pointer=stream_pointer)[
                "operation_token_sha256"
            ]
            if solve1_tke_sha256 is None
            else solve1_tke_sha256
        ),
        "output_shapes": shapes,
        "output_sha256": {
            name: f"{index:x}" * 64 for index, name in enumerate(shapes, 2)
        },
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return receipt


def _pre_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": 31,
        "stage_identity_sha256": "a" * 64,
        "solve1_full_held_token_sha256": "b" * 64,
        "solve1_tke_rhs_token_sha256": "c" * 64,
        "output_sha256": {
            name: f"{index:x}" * 64
            for index, name in enumerate(pre.GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS, 2)
        },
        "output_shapes": {
            name: list(pre._OUTPUT_SHAPES[name])
            for name in pre.GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
        },
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return receipt


def test_reserved_token_types_are_now_exact_and_acoustic_discoverable() -> None:
    assert acoustic._token_types() == (
        pre.Gm1TiledPreAcousticMomentumRkToken,
        pre.Gm1TiledRk1AcousticContinuationToken,
    )
    token = pre.Gm1TiledPreAcousticMomentumRkToken(
        MappingProxyType(_pre_receipt())
    )
    assert acoustic._require_pre_acoustic_token(token) == token.receipt


def test_full_held_lineage_is_exact_typed_self_hashed_and_same_stream() -> None:
    receipt = _full_held_receipt()
    token = Gm1TiledSolve1FullHeldToken(MappingProxyType(receipt))
    assert pre._require_full_held_token(token, stream_pointer=31) == receipt
    with pytest.raises(TypeError, match="exact Gm1TiledSolve1FullHeldToken"):
        pre._require_full_held_token(receipt)
    with pytest.raises(ValueError, match="stream differs"):
        pre._require_full_held_token(token, stream_pointer=32)
    changed = dict(receipt)
    changed["completed_nrk"] = 1
    with pytest.raises(ValueError, match="identity or cadence"):
        pre._require_full_held_token(
            Gm1TiledSolve1FullHeldToken(MappingProxyType(changed))
        )


def test_solve1_tke_lineage_is_exact_object_self_hashed_and_same_stream() -> None:
    tke_receipt = _solve1_tke_receipt()
    tke_token = Gm1TiledSolve1TkeRhsToken(MappingProxyType(tke_receipt))
    held_receipt = _full_held_receipt(
        solve1_tke_sha256=str(tke_receipt["operation_token_sha256"])
    )
    assert (
        pre._require_solve1_tke_token(
            tke_token, full_held_receipt=held_receipt, stream_pointer=31
        )
        == tke_receipt
    )
    with pytest.raises(TypeError, match="exact Gm1TiledSolve1TkeRhsToken"):
        pre._require_solve1_tke_token(
            tke_receipt, full_held_receipt=held_receipt, stream_pointer=31
        )
    with pytest.raises(ValueError, match="lineage or stream"):
        pre._require_solve1_tke_token(
            tke_token, full_held_receipt=held_receipt, stream_pointer=32
        )


def test_live_result_retains_exact_binding_and_full_held_token() -> None:
    tke_receipt = _solve1_tke_receipt()
    tke_token = Gm1TiledSolve1TkeRhsToken(MappingProxyType(tke_receipt))
    full_token = Gm1TiledSolve1FullHeldToken(
        MappingProxyType(
            _full_held_receipt(
                solve1_tke_sha256=str(tke_receipt["operation_token_sha256"])
            )
        )
    )
    arrays, metrics, terrain, vertical, stream, bridge = (
        MappingProxyType({"array": object()}),
        MappingProxyType({"metric": object()}),
        object(),
        object(),
        SimpleNamespace(ptr=31),
        object(),
    )
    binding = pre.Gm1TiledPreAcousticMomentumRkBinding(
        full_token,
        tke_token,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        object(),
        "a" * 64,
        MappingProxyType({}),
    )
    token = pre.Gm1TiledPreAcousticMomentumRkToken(
        MappingProxyType(_pre_receipt())
    )
    result = pre.FinishedGm1TiledPreAcousticMomentumRk(
        token,
        binding,
        full_token,
        tke_token,
        arrays,
        metrics,
        terrain,  # type: ignore[arg-type]
        vertical,  # type: ignore[arg-type]
        stream,
        bridge,  # type: ignore[arg-type]
        MappingProxyType({"uten": "d" * 64}),
        object(),  # type: ignore[arg-type]
    )
    assert result.binding is binding
    assert result.full_held_token is binding.full_held_token
    assert result.solve1_tke_token is binding.solve1_tke_token
    assert result.arrays is binding.arrays
    assert result.metrics is binding.metrics
    assert result.terrain_metrics is binding.terrain_metrics
    assert result.vertical is binding.vertical
    assert result.stream is binding.stream
    assert result.bridge is binding.bridge


def test_source_order_is_literal_and_bcw_is_between_mass_flux_and_divx() -> None:
    source = (
        ROOT / "src/cm1core/cuda_gm1_tilestream_pre_acoustic_momentum_rk.py"
    ).read_text(encoding="utf-8")
    prepare = source.index(
        "def prepare_cuda_gm1_tilestream_pre_acoustic_momentum_rk"
    )
    terms = source.index("launch_cuda_eqtset2_stage_terms(", prepare)
    seed = source.index("cupy.copyto(", terms)
    launch = source.index(
        "def launch_cuda_gm1_tilestream_pre_acoustic_momentum_rk", seed
    )
    momentum = source.index("launch_cuda_gm1_child_stretched_momentum5(", launch)
    buoyancy = source.index("launch_cuda_moist_buoyancy(", momentum)
    assert terms < seed < momentum < buoyancy

    stretched = (
        ROOT / "src/cm1core/cuda_gm1_child_stretched_momentum.py"
    ).read_text(encoding="utf-8")
    stretched_launch = stretched.index("def launch_cuda_gm1_child_stretched_momentum5")
    mass_flux = stretched.index("vertical_mass(", stretched_launch)
    bcw = stretched.index("rrw_boundary_hook(", mass_flux)
    divx = stretched.index("divergence(", bcw)
    assert mass_flux < bcw < divx
    assert '"flag": 0' in source
    assert '"vertical_rigid_faces_written": False' in source


def test_claim_and_status_stop_before_acoustic_and_completed_stage() -> None:
    claim = pre.GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT
    assert "global acoustic solve" in claim
    assert "completed stage" in claim
    assert "five-metre product" in claim
    assert pre.CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_STATUS == (
        "ENQUEUED_LIVE_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK1"
    )


def test_nested_immutable_receipts_are_canonical_but_foreign_objects_refuse() -> None:
    receipt = MappingProxyType({"b": MappingProxyType({"value": 1}), "a": 2})
    assert json.loads(pre._canonical(receipt)) == {"a": 2, "b": {"value": 1}}
    with pytest.raises(TypeError, match="is not JSON serializable"):
        pre._canonical(object())
