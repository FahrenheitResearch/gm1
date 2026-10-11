from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from cm1core.cuda_gm1_tilestream_stretched_closure import (
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
    Gm1TiledStretchedClosureToken,
)
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)
from cm1core.cuda_gm1_tilestream_tke_rk1 import (
    CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
    Gm1TiledTkeRk1Token,
)
from cm1core.cuda_gm1_tilestream_tke_stage_state import (
    AUTHORITY_ANCHORS,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeBaselineToken,
    bind_gm1_tiled_fresh_tke_baseline,
    bind_gm1_tiled_tke_held_state_after_rk1,
    require_gm1_tiled_tke_rk2_live_prerequisites,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tokenize(receipt: dict[str, object]) -> MappingProxyType:
    receipt["operation_token_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fixture() -> tuple[
    Gm1TiledSolve1TkeRhsToken,
    Gm1TiledTkeRk1Token,
    Gm1TiledTkeBaselineToken,
    dict[str, np.ndarray],
]:
    shape = (76, 300, 300)
    tkea = np.full(shape, np.float32(0.02), dtype="<f4")
    tkea[-1].fill(np.float32(0.0))
    tketen = np.zeros(shape, dtype="<f4")
    current = np.zeros(shape, dtype="<f4")
    current[1:-1].fill(np.float32(0.021))
    closure = Gm1TiledStretchedClosureToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
                "status": ("ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"),
                "invalid_interior_cell_count": 0,
                "stream_pointer": 17,
            }
        )
    )
    baseline = bind_gm1_tiled_fresh_tke_baseline(closure, tkea)
    solve1 = Gm1TiledSolve1TkeRhsToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
                "completed_nrk": 0,
                "stream_pointer": 17,
                "closure_operation_token_sha256": closure.receipt[
                    "operation_token_sha256"
                ],
                "output_tketen_sha256": _sha(tketen),
            }
        )
    )
    rk1 = Gm1TiledTkeRk1Token(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1",
                "completed_nrk": 1,
                "stream_pointer": 17,
                "solve1_tke_rhs_token_sha256": solve1.receipt["operation_token_sha256"],
                "stage_identity_sha256": "a" * 64,
                "output_tke3d_sha256": _sha(current),
            }
        )
    )
    return (
        solve1,
        rk1,
        baseline,
        {"tkea": tkea, "tketen": tketen, "tke3d": current},
    )


def test_fresh_baseline_is_bound_before_solve1() -> None:
    solve1, _, baseline, arrays = _fixture()
    receipt = baseline.receipt
    assert receipt["schema"] == CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA
    assert receipt["captured_before_solve1"] is True
    assert receipt["output_tkea_sha256"] == _sha(arrays["tkea"])
    assert (
        receipt["closure_operation_token_sha256"]
        == solve1.receipt["closure_operation_token_sha256"]
    )


def test_evolved_post_solve3_baseline_requires_exact_rigid_checkpoint() -> None:
    shape = (76, 300, 300)
    tkea = np.full(shape, np.float32(0.02), dtype="<f4")
    tkea[0].fill(np.float32(0.0))
    tkea[-1].fill(np.float32(0.0))
    source_sha256 = _sha(tkea)
    fresh_closure = Gm1TiledStretchedClosureToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
                "status": ("ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"),
                "invalid_interior_cell_count": 0,
                "stream_pointer": 17,
                "closure_lifecycle": "fresh-initial-tkea",
            }
        )
    )
    with pytest.raises(ValueError, match="raw-copy"):
        bind_gm1_tiled_fresh_tke_baseline(fresh_closure, tkea)
    with pytest.raises(ValueError, match="checkpoint hash"):
        bind_gm1_tiled_fresh_tke_baseline(
            fresh_closure,
            tkea,
            restored_post_solve3_tkea_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="retained-tkea closure lineage"):
        bind_gm1_tiled_fresh_tke_baseline(
            fresh_closure,
            tkea,
            restored_post_solve3_tkea_sha256=source_sha256,
        )
    closure = Gm1TiledStretchedClosureToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
                "status": ("ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"),
                "invalid_interior_cell_count": 0,
                "stream_pointer": 17,
                "closure_lifecycle": "restored-post-solve3-retained-tkea",
                "retained_tkea_source_sha256": source_sha256,
            }
        )
    )
    token = bind_gm1_tiled_fresh_tke_baseline(
        closure,
        tkea,
        restored_post_solve3_tkea_sha256=source_sha256,
    )
    assert token.receipt["boundary_disposition"] == (
        "restored-post-solve3-rigid-bottom-top-positive-zero"
    )
    assert token.receipt["restored_post_solve3_tkea_sha256"] == source_sha256


def test_held_state_binds_fixed_rhs_and_current_raw_bits() -> None:
    solve1, rk1, baseline, arrays = _fixture()
    token = bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)
    receipt = token.receipt
    assert receipt["schema"] == CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA
    assert receipt["completed_nrk"] == 1
    assert receipt["next_nrk"] == 2
    assert receipt["fixed_tkea_sha256"] == _sha(arrays["tkea"])
    assert receipt["held_tketen_sha256"] == solve1.receipt["output_tketen_sha256"]
    assert receipt["current_tke3d_sha256"] == rk1.receipt["output_tke3d_sha256"]
    assert (
        receipt["closure_tkea_baseline_token_sha256"]
        == baseline.receipt["operation_token_sha256"]
    )
    assert receipt["ready_for_rk2"] is False
    assert (
        CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA
        in receipt["missing_live_prerequisites"][0]
    )


def test_held_state_refuses_tampered_rhs_or_current() -> None:
    solve1, rk1, baseline, arrays = _fixture()
    arrays["tketen"][3, 4, 5] = np.float32(1.0)
    with pytest.raises(ValueError, match="solve1"):
        bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)
    solve1, rk1, baseline, arrays = _fixture()
    arrays["tke3d"][3, 4, 5] = np.float32(1.0)
    with pytest.raises(ValueError, match="RK1"):
        bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)


def test_held_state_refuses_alias_and_negative_tke() -> None:
    solve1, rk1, baseline, arrays = _fixture()
    arrays["tkea"] = arrays["tke3d"]
    with pytest.raises(ValueError, match="alias"):
        bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)
    solve1, rk1, baseline, arrays = _fixture()
    arrays["tkea"][2, 2, 2] = np.float32(-1.0)
    with pytest.raises(ValueError, match="baseline"):
        bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)


def test_rk2_remains_structurally_refused_without_broader_stage_and_flux() -> None:
    solve1, rk1, baseline, arrays = _fixture()
    token = bind_gm1_tiled_tke_held_state_after_rk1(solve1, rk1, baseline, arrays)
    with pytest.raises(RuntimeError, match="completed broader RK1"):
        require_gm1_tiled_tke_rk2_live_prerequisites(token, None, None)
    with pytest.raises(RuntimeError, match="unavailable"):
        require_gm1_tiled_tke_rk2_live_prerequisites(token, object(), object())


def test_authority_captures_full_stage_boundary_cadence_before_tke() -> None:
    assert "src/solve2.F:1662-1706" in AUTHORITY_ANCHORS
    assert "src/solve2.F:1708-1803" in AUTHORITY_ANCHORS
