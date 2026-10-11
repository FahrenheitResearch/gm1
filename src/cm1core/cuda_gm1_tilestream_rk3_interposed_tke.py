"""Bind exact tile-streamed RK3 TKE after the interposed RK3 prefix.

The exact prefix coordinator ends after ``solve2.F:1662-1680`` final momentum
boundaries.  This module makes the next source edge executable: every RK3 TKE
tile explicitly waits on that prefix's ready event on the identical CUDA
stream before calling the frozen RK3 TKE launcher.  The aggregate token then
binds the prefix operation hash as the TKE predecessor.

This still stops before ``apmasscon`` finish at ``solve2.F:1972-2030``.  It
does not publish corrected pressure/density hashes, complete broader RK3, or
enter solve3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from .cuda_gm1_child_stretched_sgs_tke import CompiledChildStretchedSgsTke
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_completed_rk_stage import (
    Gm1TiledCompletedRkStageToken,
)
from .cuda_gm1_tilestream_rk3_interposed_prefix import (
    CudaGm1Rk3InterposedPrefixLaunch,
    Gm1Rk3InterposedPrefixToken,
    require_gm1_rk3_interposed_prefix_token,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_gm1_tilestream_tke_rk23 import (
    CudaGm1TiledTkeRk23Launch,
    Gm1TiledTkeRk23TileSourceToken,
    Gm1TiledTkeRk23Token,
    finish_cuda_gm1_tiled_tke_rk23,
    launch_cuda_gm1_tiled_tke_rk23,
    require_gm1_tiled_tke_rk23_token,
)
from .cuda_gm1_tilestream_tke_stage_state import Gm1TiledTkeHeldStageToken


CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-interposed-tke-tile-token/v1"
)
CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_STATUS = (
    "ENQUEUED_EXACT_RK3_TKE_TILE_AFTER_PREFIX_EVENT"
)
CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-interposed-prefix-tke-token/v1"
)
CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS = (
    "ADMITTED_EXACT_RK3_PREFIX_THROUGH_TKE_COMPONENT"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1561-1595",
    "src/solve2.F:1616-1656",
    "src/solve2.F:1662-1680",
    "src/solve2.F:1708-1803",
)
SOURCE_ORDER = (
    "exact eqtset2/EOS -> apmasscon reduction -> final momentum-boundary prefix",
    "explicit prefix-ready-event wait on every identical-stream TKE tile",
    "solve2.F:1708-1803 exact TKE advw/RK3/clip/bcw and exact-once scatter",
)
GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT = (
    "one exact CM1 r21.1 local final-RK chain through tile-streamed SGS-TKE "
    "RK3, binding the eqtset/EOS-reduction-final-momentum prefix as an explicit "
    "same-stream event predecessor of every TKE tile and the admitted TKE RK3 "
    "output. Fresh scalar/RK2 science remains limited by the prefix token; "
    "apmasscon finish, corrected output hashes, completed broader RK3, solve3, "
    "histories, checkpoint, MPI, forecast, and five-metre product claims are "
    "unavailable"
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


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class Gm1Rk3InterposedTkeTileToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3InterposedTkeTileLaunch:
    ready_event: Any
    prefix: CudaGm1Rk3InterposedPrefixLaunch
    tke: CudaGm1TiledTkeRk23Launch
    token: Gm1Rk3InterposedTkeTileToken


@dataclass(frozen=True, slots=True)
class Gm1Rk3InterposedPrefixTkeToken:
    receipt: Mapping[str, object]


def _require_live_prefix_launch(
    prefix: object,
) -> tuple[
    CudaGm1Rk3InterposedPrefixLaunch,
    Mapping[str, object],
]:
    if type(prefix) is not CudaGm1Rk3InterposedPrefixLaunch:
        raise TypeError(
            "prefix must be exact CudaGm1Rk3InterposedPrefixLaunch type"
        )
    if type(prefix.token) is not Gm1Rk3InterposedPrefixToken:
        raise TypeError("prefix token exact type changed")
    receipt = require_gm1_rk3_interposed_prefix_token(prefix.token)
    if receipt.get("compact_validation_only") is not False:
        raise ValueError("RK3 TKE requires the live 300x300x75 prefix")
    if receipt.get("domain") != {"nx": 300, "ny": 300, "nk": 75}:
        raise ValueError("RK3 TKE prefix domain changed")
    stream_pointer = receipt.get("stream_pointer")
    if (
        prefix.stream is not prefix.eqtset.stream
        or prefix.stream is not prefix.reduction.stream
        or prefix.stream is not prefix.momentum_boundary.stream
        or int(prefix.stream.ptr) != stream_pointer
        or int(prefix.ready_event.ptr)
        != int(prefix.momentum_boundary.ready_event.ptr)
    ):
        raise ValueError("RK3 TKE prefix live stream/event identity changed")
    return prefix, receipt


def launch_cuda_gm1_rk3_tke_after_interposed_prefix(
    prefix: CudaGm1Rk3InterposedPrefixLaunch,
    held_token: Gm1TiledTkeHeldStageToken,
    completed_rk2_token: Gm1TiledCompletedRkStageToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    prior_tke_rk2_token: Gm1TiledTkeRk23Token,
    source_token: Gm1TiledTkeRk23TileSourceToken,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    metrics: Mapping[str, Any],
    compiled: CompiledChildStretchedSgsTke,
    workspace: Any | None = None,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1Rk3InterposedTkeTileLaunch:
    """Wait on the final-boundary event, then enqueue one exact RK3 TKE tile."""

    prefix, prefix_receipt = _require_live_prefix_launch(prefix)
    stream = prefix.stream
    stream.wait_event(prefix.ready_event)
    tke = launch_cuda_gm1_tiled_tke_rk23(
        held_token,
        completed_rk2_token,
        flux_token,
        prior_tke_rk2_token,
        source_token,
        arrays,
        nrk=3,
        spec=spec,
        metrics=metrics,
        stream=stream,
        compiled=compiled,
        workspace=workspace,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    if type(tke) is not CudaGm1TiledTkeRk23Launch:
        raise TypeError("underlying TKE tile launcher exact type changed")
    tke_receipt = tke.receipt
    if (
        not isinstance(tke_receipt, Mapping)
        or tke_receipt.get("nrk") != 3
        or tke_receipt.get("tile") != spec.record()
        or tke_receipt.get("stream_pointer") != prefix_receipt["stream_pointer"]
        or tke_receipt.get("operation_token_sha256")
        != _operation_token(tke_receipt)
    ):
        raise ValueError("underlying RK3 TKE tile identity or stream changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_STATUS,
        "nrk": 3,
        "tile": spec.record(),
        "stream_pointer": prefix_receipt["stream_pointer"],
        "prefix_operation_token_sha256": prefix_receipt[
            "operation_token_sha256"
        ],
        "prefix_ready_event_pointer": int(prefix.ready_event.ptr),
        "tke_tile_operation_token_sha256": tke_receipt[
            "operation_token_sha256"
        ],
        "tke_tile_ready_event_pointer": int(tke.ready_event.ptr),
        "predecessor_operation_token_sha256": prefix_receipt[
            "operation_token_sha256"
        ],
        "explicit_prefix_event_wait_enqueued": True,
        "exact_stream_object_identity_bound": True,
        "source_order_edge_bound": True,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "completed_rk3_token_emitted": False,
        "claim_limit": GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Rk3InterposedTkeTileLaunch(
        tke.ready_event,
        prefix,
        tke,
        Gm1Rk3InterposedTkeTileToken(MappingProxyType(receipt)),
    )


def _require_tile_wrapper(
    launch: object,
    *,
    prefix: CudaGm1Rk3InterposedPrefixLaunch,
    prefix_receipt: Mapping[str, object],
    spec: Gm1TileSpec,
) -> Mapping[str, object]:
    if type(launch) is not CudaGm1Rk3InterposedTkeTileLaunch:
        raise TypeError("RK3 interposed TKE finish requires typed tile launches")
    if launch.prefix is not prefix:
        raise ValueError("RK3 TKE tile belongs to another live prefix object")
    if type(launch.tke) is not CudaGm1TiledTkeRk23Launch:
        raise TypeError("wrapped TKE launch exact type changed")
    if type(launch.token) is not Gm1Rk3InterposedTkeTileToken:
        raise TypeError("wrapped TKE tile token exact type changed")
    receipt = launch.token.receipt
    tke_receipt = launch.tke.receipt
    if not isinstance(receipt, Mapping) or not isinstance(tke_receipt, Mapping):
        raise TypeError("wrapped TKE tile receipts must be mappings")
    if (
        receipt.get("schema")
        != CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_STATUS
        or receipt.get("nrk") != 3
        or receipt.get("tile") != spec.record()
        or receipt.get("stream_pointer") != prefix_receipt["stream_pointer"]
        or receipt.get("prefix_operation_token_sha256")
        != prefix_receipt["operation_token_sha256"]
        or receipt.get("predecessor_operation_token_sha256")
        != prefix_receipt["operation_token_sha256"]
        or receipt.get("prefix_ready_event_pointer")
        != int(prefix.ready_event.ptr)
        or receipt.get("tke_tile_operation_token_sha256")
        != tke_receipt.get("operation_token_sha256")
        or receipt.get("tke_tile_ready_event_pointer")
        != int(launch.tke.ready_event.ptr)
        or receipt.get("explicit_prefix_event_wait_enqueued") is not True
        or receipt.get("exact_stream_object_identity_bound") is not True
        or receipt.get("source_order_edge_bound") is not True
        or receipt.get("apmasscon_finish_bound") is not False
        or receipt.get("post_output_sha256_bound") is not False
        or receipt.get("completed_rk3_token_emitted") is not False
        or receipt.get("claim_limit") != GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 interposed TKE tile identity or lineage changed")
    return receipt


def finish_cuda_gm1_rk3_tke_after_interposed_prefix(
    prefix: CudaGm1Rk3InterposedPrefixLaunch,
    held_token: Gm1TiledTkeHeldStageToken,
    completed_rk2_token: Gm1TiledCompletedRkStageToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    prior_tke_rk2_token: Gm1TiledTkeRk23Token,
    launches: Sequence[CudaGm1Rk3InterposedTkeTileLaunch],
    *,
    plan: Gm1TilePlan,
    output_tke3d: Any,
) -> Gm1Rk3InterposedPrefixTkeToken:
    """Admit exact RK3 TKE and bind it as the prefix successor."""

    prefix, prefix_receipt = _require_live_prefix_launch(prefix)
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("RK3 interposed TKE finish requires the live 300 grid")
    if len(launches) != plan.ntiles:
        raise ValueError("RK3 interposed TKE finish requires every exact tile")
    wrapper_operations: list[str] = []
    underlying: list[CudaGm1TiledTkeRk23Launch] = []
    for spec, launch in zip(plan.specs, launches, strict=True):
        receipt = _require_tile_wrapper(
            launch,
            prefix=prefix,
            prefix_receipt=prefix_receipt,
            spec=spec,
        )
        wrapper_operations.append(str(receipt["operation_token_sha256"]))
        underlying.append(launch.tke)
    tke_token = finish_cuda_gm1_tiled_tke_rk23(
        held_token,
        completed_rk2_token,
        flux_token,
        prior_tke_rk2_token,
        underlying,
        nrk=3,
        plan=plan,
        output_tke3d=output_tke3d,
    )
    tke = require_gm1_tiled_tke_rk23_token(
        tke_token,
        completed_nrk=3,
        stream_pointer=int(prefix_receipt["stream_pointer"]),
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS,
        "completed_nrk": 3,
        "stream_pointer": prefix_receipt["stream_pointer"],
        "stage_identity_sha256": tke["stage_identity_sha256"],
        "prefix_operation_token_sha256": prefix_receipt[
            "operation_token_sha256"
        ],
        "tke_rk3_operation_token_sha256": tke["operation_token_sha256"],
        "tke_rk3_output_sha256": tke["output_tke3d_sha256"],
        "predecessor_operation_token_sha256": {
            "tke_rk3": prefix_receipt["operation_token_sha256"]
        },
        "tile_wrapper_operations_sha256": hashlib.sha256(
            _canonical(wrapper_operations)
        ).hexdigest(),
        "tile_count": plan.ntiles,
        "source_order": list(SOURCE_ORDER),
        "source_order_enqueue_chain_bound": True,
        "exact_stream_object_identity_bound": True,
        "prefix_event_wait_bound_on_every_tile": True,
        "rk3_tke_bound": True,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "history_checkpoint_emitted": False,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Rk3InterposedPrefixTkeToken(MappingProxyType(receipt))


def require_gm1_rk3_interposed_prefix_tke_token(
    token: object,
    *,
    prefix_token: Gm1Rk3InterposedPrefixToken | None = None,
    tke_rk3_token: Gm1TiledTkeRk23Token | None = None,
) -> Mapping[str, object]:
    if type(token) is not Gm1Rk3InterposedPrefixTkeToken:
        raise TypeError(
            "token must be exact Gm1Rk3InterposedPrefixTkeToken type"
        )
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("RK3 interposed TKE receipt must be a mapping")
    predecessor = receipt.get("predecessor_operation_token_sha256")
    if (
        receipt.get("schema") != CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("source_order") != list(SOURCE_ORDER)
        or not isinstance(predecessor, Mapping)
        or dict(predecessor)
        != {"tke_rk3": receipt.get("prefix_operation_token_sha256")}
        or receipt.get("source_order_enqueue_chain_bound") is not True
        or receipt.get("exact_stream_object_identity_bound") is not True
        or receipt.get("prefix_event_wait_bound_on_every_tile") is not True
        or receipt.get("rk3_tke_bound") is not True
        or receipt.get("apmasscon_finish_bound") is not False
        or receipt.get("post_output_sha256_bound") is not False
        or receipt.get("mass_finish_source_order_complete") is not False
        or receipt.get("completed_rk3_token_emitted") is not False
        or receipt.get("solve3_entered") is not False
        or receipt.get("history_checkpoint_emitted") is not False
        or receipt.get("claim_limit") != GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 interposed TKE identity, lineage, or claim changed")
    stream_pointer = receipt.get("stream_pointer")
    tile_count = receipt.get("tile_count")
    if (
        isinstance(stream_pointer, bool)
        or not isinstance(stream_pointer, int)
        or stream_pointer <= 0
        or isinstance(tile_count, bool)
        or not isinstance(tile_count, int)
        or tile_count <= 0
    ):
        raise ValueError("RK3 interposed TKE stream/tile identity changed")
    for name in (
        "stage_identity_sha256",
        "prefix_operation_token_sha256",
        "tke_rk3_operation_token_sha256",
        "tke_rk3_output_sha256",
        "tile_wrapper_operations_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(name, receipt.get(name))
    if prefix_token is not None:
        prefix = require_gm1_rk3_interposed_prefix_token(
            prefix_token,
            stream_pointer=stream_pointer,
        )
        if receipt.get("prefix_operation_token_sha256") != prefix.get(
            "operation_token_sha256"
        ):
            raise ValueError("RK3 interposed TKE belongs to another prefix")
    if tke_rk3_token is not None:
        tke = require_gm1_tiled_tke_rk23_token(
            tke_rk3_token,
            completed_nrk=3,
            stream_pointer=stream_pointer,
        )
        if (
            receipt.get("tke_rk3_operation_token_sha256")
            != tke.get("operation_token_sha256")
            or receipt.get("stage_identity_sha256")
            != tke.get("stage_identity_sha256")
            or receipt.get("tke_rk3_output_sha256")
            != tke.get("output_tke3d_sha256")
        ):
            raise ValueError("RK3 interposed TKE output lineage changed")
    return MappingProxyType(dict(receipt))


def refuse_gm1_rk3_interposed_tke_as_mass_finish_or_completed_stage(
    token: object,
) -> NoReturn:
    receipt = require_gm1_rk3_interposed_prefix_tke_token(token)
    raise RuntimeError(
        "GM1-RK3-INTERPOSED-TKE-NOT-MASS-FINISH: exact prefix-through-TKE "
        + str(receipt["operation_token_sha256"])
        + " still lacks apmasscon finish and corrected pp3d/prs/rho hashes; "
        "completed RK3 remains unavailable"
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_STATUS",
    "CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS",
    "CudaGm1Rk3InterposedTkeTileLaunch",
    "GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT",
    "Gm1Rk3InterposedPrefixTkeToken",
    "Gm1Rk3InterposedTkeTileToken",
    "SOURCE_ORDER",
    "finish_cuda_gm1_rk3_tke_after_interposed_prefix",
    "launch_cuda_gm1_rk3_tke_after_interposed_prefix",
    "refuse_gm1_rk3_interposed_tke_as_mass_finish_or_completed_stage",
    "require_gm1_rk3_interposed_prefix_tke_token",
)
