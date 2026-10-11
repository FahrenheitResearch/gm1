"""Local final-RK momentum boundary replay in exact CM1 source order.

For the selected branch, ``solve2.F:1662-1680`` executes the same numerical
``bcu``, ``bcv``, ``bcw(flag=1)``, and terrain ``bcwsfc`` operators in every
RK stage.  The frozen RK1 implementation already owns and raw-bit validates
those stage-independent kernels.  This module reuses only that numerical ABI
while publishing a distinct RK3-local token; it never relabels the RK1 token.

The local receipt deliberately does not bind the fresh RK3 post-acoustic
momentum carrier, the RK3 scalar update, the preceding eqtset2/removal and
split mass-reduction tokens, or the following RK3 TKE token.  It therefore
cannot complete RK3 or enter solve3.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from .cuda_gm1_rk1_post_scalar_momentum_boundaries import (
    AUTHORITY_BC_SHA256,
    AUTHORITY_INIT_TERRAIN_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CM1_OPEN_NGXY,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256,
    GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT,
    CompiledGm1Rk1PostScalarMomentumBoundary,
    compile_cuda_gm1_rk1_post_scalar_momentum_boundary,
    cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace,
    rk1_post_scalar_momentum_boundary_shapes,
)
from .cuda_sound3_open_boundary import CUDA_SOUND3_OPEN_SOURCE_SHA256
from .errors import refuse


CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-final-momentum-boundary-local-token/v1"
)
CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_STATUS = (
    "ENQUEUED_LOCAL_RK3_FINAL_MOMENTUM_BOUNDARIES_ONLY"
)
CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_SCHEMA = (
    "gm1.cuda-rk3-final-momentum-boundary-local/v1"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1662-1680",
    "src/bc.F:528-1143",
    "src/bc.F:1151-1178",
    "src/init_terrain.F:160-230",
)
SOURCE_ORDER = (
    "bcu(u3d): west,east,south,north",
    "bcv(v3d): south,north,west,east",
    "bcw(w3d,1): west,east,south,north then rigid bottom/top",
    "terrain bcwsfc replaces the rigid bottom on physical plus one-halo cells",
)
GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_CLAIM_LIMIT = (
    "one local CM1 r21.1 selected serial final-RK u3d/v3d/w3d open-family "
    "replay, rigid w faces, and terrain bcwsfc enqueue using the exact frozen "
    "stage-independent numerical kernels. Fresh RK3 acoustic/momentum and "
    "scalar lineage, eqtset2 removal, split apmasscon reduction, RK3 TKE, "
    "corrected output hashes, completed RK3, solve3, MPI, histories, forecast, "
    "tornado, and five-metre product claims remain unavailable"
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


def _stream_pointer(stream: object) -> int:
    pointer = getattr(stream, "ptr", None)
    if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0:
        raise ValueError("CUDA stream pointer must be a positive integer")
    return pointer


def rk3_final_momentum_boundary_shapes(
    *, nx: int, ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    return rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)


def rk3_final_momentum_boundary_buffer_nbytes(
    *, nx: int, ny: int, nk: int
) -> Mapping[str, int]:
    shapes = rk3_final_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    field_bytes = {
        name: int(4 * math.prod(shape)) for name, shape in shapes.items()
    }
    return MappingProxyType(
        {
            **field_bytes,
            "total_device_binding_bytes": sum(field_bytes.values()),
            "incremental_device_workspace_bytes": 0,
        }
    )


def cpu_gm1_rk3_final_momentum_boundaries_inplace(
    arrays: Mapping[str, Any], *, nx: int, ny: int, nk: int
) -> None:
    """Use the frozen independent CPU transcription for the RK3 call site."""

    cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace(
        arrays, nx=nx, ny=ny, nk=nk
    )


@dataclass(frozen=True, slots=True)
class Gm1Rk3FinalMomentumBoundaryLocalToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3FinalMomentumBoundaryLocalLaunch:
    ready_event: Any
    compiled: CompiledGm1Rk1PostScalarMomentumBoundary
    stream: Any
    arrays: Mapping[str, Any]
    token: Gm1Rk3FinalMomentumBoundaryLocalToken


def _require_device_arrays(
    arrays: Mapping[str, Any], *, nx: int, ny: int, nk: int
) -> Mapping[str, Any]:
    shapes = rk3_final_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("RK3 final momentum boundary inventory/order changed")
    cupy = importlib.import_module("cupy")
    checked: dict[str, Any] = {}
    pointers: list[int] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {shape}")
        checked[name] = value
        pointers.append(int(value.data.ptr))
    if len(set(pointers)) != len(pointers):
        raise ValueError("RK3 final momentum arrays/metrics must not alias")
    return MappingProxyType(checked)


def _pointer_inventory_sha256(arrays: Mapping[str, Any]) -> str:
    pointers = {name: int(value.data.ptr) for name, value in arrays.items()}
    return hashlib.sha256(_canonical(pointers)).hexdigest()


def launch_cuda_gm1_rk3_final_momentum_boundaries_local(
    arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    stream: Any,
    compiled: CompiledGm1Rk1PostScalarMomentumBoundary | None = None,
    allow_compact_validation: bool = False,
) -> CudaGm1Rk3FinalMomentumBoundaryLocalLaunch:
    """Enqueue only the stage-independent numerical boundary replay."""

    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be bool")
    if allow_compact_validation:
        if min(nx, ny, nk) < 2:
            raise ValueError("compact RK3 final momentum domain is too small")
    elif (nx, ny, nk) != (300, 300, 75):
        raise ValueError("live-sized RK3 final momentum domain must be 300x300x75")
    stream_pointer = _stream_pointer(stream)
    checked = _require_device_arrays(arrays, nx=nx, ny=ny, nk=nk)
    cupy = importlib.import_module("cupy")
    compiled = compiled or compile_cuda_gm1_rk1_post_scalar_momentum_boundary()
    if type(compiled) is not CompiledGm1Rk1PostScalarMomentumBoundary:
        raise TypeError(
            "compiled must be the exact frozen momentum-boundary image type"
        )
    _, _, _, _, _, kfwe, kfsn, kwrigid, _, _ = compiled.open_compiled.kernels
    threads = (256, 1, 1)
    words = max(value.size for value in checked.values())
    grid = ((words + threads[0] - 1) // threads[0], 1, 1)
    edge_words = (ny + 2) * (nx + 2)
    edge_grid = ((edge_words + threads[0] - 1) // threads[0], 1, 1)
    u32 = cupy.uint32
    ng = u32(CM1_OPEN_NGXY)
    common = (u32(nx), u32(ny), u32(nk), ng)
    with stream:
        kfwe(grid, threads, (checked["u3d"], *common, u32(1)), stream=stream)
        kfsn(grid, threads, (checked["u3d"], *common, u32(1)), stream=stream)
        kfsn(grid, threads, (checked["v3d"], *common, u32(2)), stream=stream)
        kfwe(grid, threads, (checked["v3d"], *common, u32(2)), stream=stream)
        kfwe(grid, threads, (checked["w3d"], *common, u32(3)), stream=stream)
        kfsn(grid, threads, (checked["w3d"], *common, u32(3)), stream=stream)
        kwrigid(grid, threads, (checked["w3d"], *common), stream=stream)
        compiled.bcwsfc_kernel(
            edge_grid,
            threads,
            (
                checked["u3d"],
                checked["v3d"],
                checked["w3d"],
                checked["gz"],
                checked["dzdx"],
                checked["dzdy"],
                *common,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_SCHEMA,
        "operator_schema": CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_SCHEMA,
        "status": CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_STATUS,
        "nrk": 3,
        "nrkmax": 3,
        "source_order_phase": "solve2.F:1662-1680",
        "source_order": list(SOURCE_ORDER),
        "next_source_order_phase": "solve2.F:1708-1803 SGS-TKE RK3",
        "stream_pointer": stream_pointer,
        "event_pointer": int(event.ptr),
        "domain": {
            "nx": nx,
            "ny": ny,
            "nk": nk,
            "ngxy": CM1_OPEN_NGXY,
            "terrain": True,
            "mpi": False,
        },
        "pointer_inventory_sha256": _pointer_inventory_sha256(checked),
        "kernel": {
            "bcwsfc_source_sha256": (
                CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256
            ),
            "open_family_source_sha256": CUDA_SOUND3_OPEN_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "runtime": dict(compiled.runtime_identity),
            "stage_independent_numeric_abi_reused": True,
            "rk1_token_relabelled": False,
        },
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "init_terrain_sha256": AUTHORITY_INIT_TERRAIN_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "fresh_rk3_post_acoustic_momentum_lineage_bound": False,
        "fresh_rk3_scalar_moist_lineage_bound": False,
        "eqtset2_remove_eos_token_bound": False,
        "apmasscon_reduction_token_bound": False,
        "scalar_pressure_boundaries_performed": False,
        "rk3_tke_lineage_bound": False,
        "apmasscon_finish_bound": False,
        "post_output_hashes_bound": False,
        "broader_completed_rk3": False,
        "claim_limit": GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Rk3FinalMomentumBoundaryLocalLaunch(
        event,
        compiled,
        stream,
        checked,
        Gm1Rk3FinalMomentumBoundaryLocalToken(MappingProxyType(receipt)),
    )


def require_gm1_rk3_final_momentum_boundary_local_token(
    token: object, *, stream_pointer: int | None = None
) -> Mapping[str, object]:
    if type(token) is not Gm1Rk3FinalMomentumBoundaryLocalToken:
        raise TypeError(
            "token must be exact Gm1Rk3FinalMomentumBoundaryLocalToken type"
        )
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("RK3 final momentum receipt must be a mapping")
    negative_claims = (
        "fresh_rk3_post_acoustic_momentum_lineage_bound",
        "fresh_rk3_scalar_moist_lineage_bound",
        "eqtset2_remove_eos_token_bound",
        "apmasscon_reduction_token_bound",
        "scalar_pressure_boundaries_performed",
        "rk3_tke_lineage_bound",
        "apmasscon_finish_bound",
        "post_output_hashes_bound",
        "broader_completed_rk3",
    )
    if (
        receipt.get("schema")
        != CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_SCHEMA
        or receipt.get("operator_schema")
        != CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_STATUS
        or receipt.get("nrk") != 3
        or receipt.get("nrkmax") != 3
        or receipt.get("source_order_phase") != "solve2.F:1662-1680"
        or receipt.get("source_order") != list(SOURCE_ORDER)
        or receipt.get("next_source_order_phase")
        != "solve2.F:1708-1803 SGS-TKE RK3"
        or any(receipt.get(name) is not False for name in negative_claims)
        or receipt.get("claim_limit")
        != GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 final momentum local token identity or claim changed")
    actual_stream = receipt.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("RK3 final momentum stream pointer is invalid")
    if stream_pointer is not None:
        if (
            isinstance(stream_pointer, bool)
            or not isinstance(stream_pointer, int)
            or stream_pointer <= 0
        ):
            raise ValueError("stream_pointer must be a positive integer")
        if actual_stream != stream_pointer:
            raise ValueError("RK3 final momentum token belongs to another stream")
    for name in ("pointer_inventory_sha256", "operation_token_sha256"):
        value = receipt.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return MappingProxyType(dict(receipt))


def refuse_gm1_rk3_final_momentum_boundary_live_promotion(
    local_token: object,
    *,
    post_acoustic_momentum_token: object | None = None,
    scalar_moist_token: object | None = None,
) -> NoReturn:
    receipt = require_gm1_rk3_final_momentum_boundary_local_token(local_token)
    refuse(
        code="GM1-RK3-FINAL-MOMENTUM-LIVE-LINEAGE-UNAVAILABLE",
        gate="gm1-rk3-final-momentum-boundaries",
        option="live_prerequisite_types",
        value={
            "post_acoustic_momentum": (
                None
                if post_acoustic_momentum_token is None
                else type(post_acoustic_momentum_token).__name__
            ),
            "scalar_moist": (
                None if scalar_moist_token is None else type(scalar_moist_token).__name__
            ),
        },
        reason=(
            "the local boundary token "
            + str(receipt["operation_token_sha256"])
            + " leaves every RK3 stage prerequisite and successor unbound"
        ),
        remedy=(
            "bind exact fresh RK3 acoustic/momentum, scalar/moist, eqtset2, and "
            "split-reduction tokens on the same stream before emitting a new live token"
        ),
    )


def require_gm1_rk3_final_momentum_boundary_source_audit() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "schema": "gm1.rk3-final-momentum-boundary-source-audit/v1",
            "status": "SOURCE_AUDITED_LOCAL_NUMERIC_OPERATOR_LIVE_FAIL_CLOSED",
            "source_order": list(SOURCE_ORDER),
            "predecessors": [
                "fresh RK3 scalar/moist update",
                "eqtset2 remove/snapshot and moist EOS",
                "split apmasscon layer reductions enqueued",
            ],
            "successor": "SGS-TKE advw/RK3/clip/bcw",
            "rk3_skips_scalar_pressure_boundary_family": True,
            "numerical_kernel_stage_independent": True,
            "frozen_numeric_kernel_source_sha256": {
                "bcwsfc": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256,
                "open_family": CUDA_SOUND3_OPEN_SOURCE_SHA256,
            },
            "prior_rk1_operator_claim_limit": (
                GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT
            ),
            "local_operator_available": True,
            "live_rk3_token_available": False,
            "completed_rk3_emitter_available": False,
            "authority": {
                "release": "CM1 r21.1",
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
                "bc_sha256": AUTHORITY_BC_SHA256,
                "init_terrain_sha256": AUTHORITY_INIT_TERRAIN_SHA256,
                "anchors": list(AUTHORITY_ANCHORS),
            },
            "claim_limit": GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_CLAIM_LIMIT,
        }
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_SCHEMA",
    "CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_TOKEN_STATUS",
    "CudaGm1Rk3FinalMomentumBoundaryLocalLaunch",
    "GM1_RK3_FINAL_MOMENTUM_BOUNDARY_LOCAL_CLAIM_LIMIT",
    "Gm1Rk3FinalMomentumBoundaryLocalToken",
    "SOURCE_ORDER",
    "cpu_gm1_rk3_final_momentum_boundaries_inplace",
    "launch_cuda_gm1_rk3_final_momentum_boundaries_local",
    "refuse_gm1_rk3_final_momentum_boundary_live_promotion",
    "require_gm1_rk3_final_momentum_boundary_local_token",
    "require_gm1_rk3_final_momentum_boundary_source_audit",
    "rk3_final_momentum_boundary_buffer_nbytes",
    "rk3_final_momentum_boundary_shapes",
)
