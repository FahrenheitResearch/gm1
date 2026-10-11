"""Typed held-state bridge between tile-streamed SGS-TKE RK stages.

CM1 r21.1 computes ``tketen`` once in ``solve1`` and holds both it and the
full-step ``tkea`` baseline fixed while ``solve2`` advances ``tke3d`` through
three RK stages.  This module first binds fresh ``tkea`` immediately after the
closure, then proves that baseline survived unchanged through the sealed RK1
pass.  It deliberately does not manufacture the broader completed RK-stage
token: velocity, pressure/density, theta, moisture, and their boundary cadence
must all complete before a fresh nrk=2 acoustic/flux token is legal.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .cuda_gm1_tilestream_stretched_closure import (
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
    Gm1TiledStretchedClosureToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)
from .cuda_gm1_tilestream_tke_rk1 import (
    AUTHORITY_ADV_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
    Gm1TiledTkeRk1Token,
)


CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA = (
    "gm1.tilestream-tke-held-stage-state-token/v1"
)
CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA = (
    "gm1.tilestream-fresh-tke-baseline-token/v1"
)
CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA = (
    "gm1.tilestream-completed-rk-stage-token/v1"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:294-299",
    "src/solve2.F:1662-1706",
    "src/solve2.F:1708-1803",
)
GM1_TILESTREAM_TKE_HELD_STAGE_CLAIM_LIMIT = (
    "raw-bit binding of the live 300x300x75 CM1 r21.1 fixed tkea baseline, "
    "held solve1 tketen, and completed tile-streamed RK1 tke3d on one CUDA "
    "stream. It is state provenance only: it does not complete the broader "
    "RK1 state, produce nrk=2 sound or mass flux, execute TKE RK2/RK3, commit "
    "solve3 tkea, tile-stream the global pressure solver, provide MPI/open "
    "histories, or establish a full forecast or five-metre product"
)
GM1_TILESTREAM_TKE_BASELINE_CLAIM_LIMIT = (
    "raw-bit binding of the physical fresh-closure tkea baseline immediately "
    "after the live tiled closure and before solve1/RK mutation. It does not "
    "produce a TKE tendency, execute an RK stage, complete solve3, tile-stream "
    "the pressure solver, or establish a full forecast or five-metre product"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _sha(array: object) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()  # type: ignore[attr-defined]


def _require_live_rk1_tokens(
    solve1_token: object, rk1_token: object
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not isinstance(solve1_token, Gm1TiledSolve1TkeRhsToken):
        raise TypeError("solve1_token must be the exact live TKE RHS token")
    if not isinstance(rk1_token, Gm1TiledTkeRk1Token):
        raise TypeError("rk1_token must be the exact completed TKE RK1 token")
    solve1, rk1 = solve1_token.receipt, rk1_token.receipt
    if (
        solve1.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or solve1.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or solve1.get("completed_nrk") != 0
        or solve1.get("operation_token_sha256") != _operation_token(solve1)
    ):
        raise ValueError("solve1 TKE RHS token identity changed")
    if (
        rk1.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA
        or rk1.get("status") != "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1"
        or rk1.get("completed_nrk") != 1
        or rk1.get("solve1_tke_rhs_token_sha256")
        != solve1.get("operation_token_sha256")
        or rk1.get("stream_pointer") != solve1.get("stream_pointer")
        or rk1.get("operation_token_sha256") != _operation_token(rk1)
    ):
        raise ValueError("TKE RK1 token identity, lineage, or stream changed")
    return solve1, rk1


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeBaselineToken:
    receipt: Mapping[str, object]


def bind_gm1_tiled_fresh_tke_baseline(
    closure_token: Gm1TiledStretchedClosureToken,
    tkea: object,
    *,
    restored_post_solve3_tkea_sha256: str | None = None,
) -> Gm1TiledTkeBaselineToken:
    """Bind fresh ``tkea`` before solve1 so later stages cannot substitute it."""

    if not isinstance(closure_token, Gm1TiledStretchedClosureToken):
        raise TypeError("closure_token must be the exact completed closure token")
    closure = closure_token.receipt
    if (
        closure.get("schema") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA
        or closure.get("status")
        != "ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"
        or closure.get("invalid_interior_cell_count") != 0
        or closure.get("operation_token_sha256") != _operation_token(closure)
    ):
        raise ValueError("fresh closure token identity changed")
    np = importlib.import_module("numpy")
    shape = (76, 300, 300)
    if (
        not isinstance(tkea, np.ndarray)
        or tkea.dtype != np.dtype("<f4")
        or tkea.shape != shape
        or not tkea.flags.c_contiguous
        or not bool(np.all(np.isfinite(tkea)))
    ):
        raise ValueError(f"fresh tkea must be finite host C-<f4 {shape}")
    if bool(np.any(tkea[1:-1] < np.float32(0.0))):
        raise ValueError("fresh tkea contains negative interior TKE")
    tkea_sha256 = _sha(tkea)
    if restored_post_solve3_tkea_sha256 is None:
        if closure.get("closure_lifecycle") not in {None, "fresh-initial-tkea"}:
            raise ValueError("fresh baseline requires the initial closure lifecycle")
        if not np.array_equal(tkea[0].view(np.uint32), tkea[1].view(np.uint32)):
            raise ValueError("fresh tkea bottom does not raw-copy its k=1 interior")
        boundary_disposition = "fresh-initial-bottom-raw-copies-k1"
    else:
        if (
            not isinstance(restored_post_solve3_tkea_sha256, str)
            or len(restored_post_solve3_tkea_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in restored_post_solve3_tkea_sha256
            )
            or restored_post_solve3_tkea_sha256 != tkea_sha256
        ):
            raise ValueError(
                "restored post-solve3 tkea differs from its exact checkpoint hash"
            )
        if (
            closure.get("closure_lifecycle") != "restored-post-solve3-retained-tkea"
            or closure.get("retained_tkea_source_sha256")
            != restored_post_solve3_tkea_sha256
        ):
            raise ValueError(
                "restored baseline requires the exact retained-tkea closure lineage"
            )
        if np.count_nonzero(tkea[0].view(np.uint32)):
            raise ValueError("restored post-solve3 tkea bottom is not positive zero")
        boundary_disposition = "restored-post-solve3-rigid-bottom-top-positive-zero"
    if np.count_nonzero(tkea[-1].view(np.uint32)):
        raise ValueError("fresh tkea model top is not positive zero")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_FRESH_TKE_BASELINE_BEFORE_SOLVE1",
        "closure_operation_token_sha256": closure["operation_token_sha256"],
        "stream_pointer": closure["stream_pointer"],
        "output_tkea_shape": list(shape),
        "output_tkea_words": 76 * 300 * 300,
        "output_tkea_sha256": tkea_sha256,
        "captured_before_solve1": True,
        "fixed_until_solve3": True,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_BASELINE_CLAIM_LIMIT,
    }
    if restored_post_solve3_tkea_sha256 is not None:
        receipt.update(
            {
                "boundary_disposition": boundary_disposition,
                "restored_post_solve3_tkea_sha256": (restored_post_solve3_tkea_sha256),
            }
        )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeBaselineToken(MappingProxyType(receipt))


def _require_baseline_token(
    token: object,
    solve1: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledTkeBaselineToken):
        raise TypeError("baseline_token must be the exact fresh TKE baseline token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_FRESH_TKE_BASELINE_BEFORE_SOLVE1"
        or receipt.get("closure_operation_token_sha256")
        != solve1.get("closure_operation_token_sha256")
        or receipt.get("stream_pointer") != solve1.get("stream_pointer")
        or receipt.get("captured_before_solve1") is not True
        or receipt.get("fixed_until_solve3") is not True
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("fresh TKE baseline identity, lineage, or stream changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeHeldStageToken:
    receipt: Mapping[str, object]


def bind_gm1_tiled_tke_held_state_after_rk1(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    rk1_token: Gm1TiledTkeRk1Token,
    baseline_token: Gm1TiledTkeBaselineToken,
    arrays: Mapping[str, object],
) -> Gm1TiledTkeHeldStageToken:
    """Raw-hash the three TKE carriers that must survive into RK2/RK3."""

    solve1, rk1 = _require_live_rk1_tokens(solve1_token, rk1_token)
    baseline = _require_baseline_token(baseline_token, solve1)
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "tkea",
        "tketen",
        "tke3d",
    }:
        raise ValueError("held TKE inventory must be exactly tkea/tketen/tke3d")
    np = importlib.import_module("numpy")
    shape = (76, 300, 300)
    ranges: list[tuple[int, int, str]] = []
    for name in ("tkea", "tketen", "tke3d"):
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"held {name} must be finite host C-<f4 {shape}")
        start = int(value.ctypes.data)
        ranges.append((start, start + int(value.nbytes), name))
    for index, (start, stop, name) in enumerate(ranges):
        for other_start, other_stop, other_name in ranges[index + 1 :]:
            if start < other_stop and other_start < stop:
                raise ValueError(f"held TKE arrays alias: {name}/{other_name}")

    tkea = arrays["tkea"]
    tketen = arrays["tketen"]
    current = arrays["tke3d"]
    tkea_sha = _sha(tkea)
    tketen_sha = _sha(tketen)
    current_sha = _sha(current)
    if tkea_sha != baseline.get("output_tkea_sha256"):
        raise ValueError("fixed tkea differs from pre-solve1 fresh baseline")
    if tketen_sha != solve1.get("output_tketen_sha256"):
        raise ValueError("held tketen differs raw-bit from completed solve1 output")
    if current_sha != rk1.get("output_tke3d_sha256"):
        raise ValueError("current tke3d differs raw-bit from completed RK1 output")
    if np.count_nonzero(tkea[-1].view(np.uint32)):
        raise ValueError("fixed tkea model top is not positive zero")
    if np.count_nonzero(tketen[[0, -1]].view(np.uint32)):
        raise ValueError("held tketen rigid faces are not positive zero")
    if np.count_nonzero(current[[0, -1]].view(np.uint32)):
        raise ValueError("current tke3d rigid faces are not positive zero")
    if bool(np.any(tkea[1:-1] < np.float32(0.0))):
        raise ValueError("fixed tkea contains negative interior TKE")
    if bool(np.any(current[1:-1] < np.float32(0.0))):
        raise ValueError("current tke3d contains negative interior TKE")

    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1",
        "completed_nrk": 1,
        "next_nrk": 2,
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "tke_rk1_token_sha256": rk1["operation_token_sha256"],
        "closure_tkea_baseline_token_sha256": baseline["operation_token_sha256"],
        "rk1_stage_identity_sha256": rk1["stage_identity_sha256"],
        "stream_pointer": rk1["stream_pointer"],
        "shape_zyx": list(shape),
        "fixed_tkea_sha256": tkea_sha,
        "held_tketen_sha256": tketen_sha,
        "current_tke3d_sha256": current_sha,
        "tkea_fixed_until_solve3": True,
        "tketen_held_from_solve1": True,
        "current_completed_nrk": 1,
        "ready_for_rk2": False,
        "missing_live_prerequisites": [
            CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA
            + " with completed_nrk=1",
            "gm1.tilestream-post-acoustic-mass-flux-token/v1 with nrk=2",
        ],
        "structural_refusals": [
            "TKE-only assertion of completed broader RK1",
            "reuse of nrk=1 post-acoustic mass flux at nrk=2",
            "RK2/RK3 before the matching broader stage and fresh flux tokens",
            "solve3 tkea commit before completed broader RK3",
            "global psolver tilestream claim",
            "MPI exchange or open-history production",
            "full forecast or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_HELD_STAGE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeHeldStageToken(MappingProxyType(receipt))


def require_gm1_tiled_tke_rk2_live_prerequisites(
    held_token: object,
    completed_rk1_token: object,
    nrk2_flux_token: object,
) -> None:
    """Fail closed until the two non-TKE live prerequisites are frozen."""

    if not isinstance(held_token, Gm1TiledTkeHeldStageToken):
        raise TypeError("held_token must be the exact held TKE stage token")
    held = held_token.receipt
    if (
        held.get("schema") != CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA
        or held.get("status") != "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1"
        or held.get("completed_nrk") != 1
        or held.get("next_nrk") != 2
        or held.get("operation_token_sha256") != _operation_token(held)
    ):
        raise ValueError("held TKE stage token identity changed")
    # The broader token class is owned by the source-order RK aggregator, but
    # no live completed-RK1 emitter exists yet.  Do not accept duck-typed maps.
    if completed_rk1_token is None or nrk2_flux_token is None:
        raise RuntimeError(
            "GM1-TILESTREAM-TKE-RK2-PREREQUISITE: exact completed broader "
            "RK1 token and fresh nrk=2 sound -> post-sound mass-flux token "
            "are not both available"
        )
    raise RuntimeError(
        "GM1-TILESTREAM-TKE-RK2-PREREQUISITE: live completed-RK1 aggregator "
        "output is unavailable; foreign or duck-typed tokens are refused"
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_BASELINE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA",
    "GM1_TILESTREAM_TKE_BASELINE_CLAIM_LIMIT",
    "GM1_TILESTREAM_TKE_HELD_STAGE_CLAIM_LIMIT",
    "Gm1TiledTkeBaselineToken",
    "Gm1TiledTkeHeldStageToken",
    "bind_gm1_tiled_fresh_tke_baseline",
    "bind_gm1_tiled_tke_held_state_after_rk1",
    "require_gm1_tiled_tke_rk2_live_prerequisites",
)
