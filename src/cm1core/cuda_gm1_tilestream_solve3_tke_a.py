"""Tile-local replay of the CM1 solve3 TKE current-to-A copy.

The live solve3 seam is source-ordered after padded momentum current-to-A and
requires ``tke3d`` after all three TKE RK stages and the final ``bcw``.  That
live producer is not yet present in the tile stream.  This module therefore
exposes two deliberately distinct token classes:

* a completed-RK3 live token, accepted only on the same CUDA stream; and
* a detached replay token admitted from the exact real post-step checkpoint.

Only the latter currently has a producer.  It proves the literal padded copy
operator against real evolved TKE bytes, but can never advance the live token
chain or be relabelled as completed RK3.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec, build_gm1_open_tile_plan
from .cuda_gm1_tilestream_solve3_padded_momentum_a import (
    GM1_ACCEPTED_PADDED_OUTPUT_SHA256,
    GM1_SOURCE_IDENTITY_SHA256,
    Gm1TiledSolve3PaddedMomentumAToken,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SCHEMA = "gm1.cuda-tilestream-solve3-tke-a/v1"
CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_ABI = (
    "gm1.cuda-tilestream-solve3-padded-tke-current-to-a-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME = (
    "gm1_tiled_solve3_tke_a_f32_v1"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_ANCHORS = (
    "src/solve2.F:1708-1803",
    "src/bc.F:942-1143",
    "src/solve3.F:563-578",
)

GM1_ACCEPTED_ONE_STEP_CHECKPOINT_ARTIFACT_SHA256 = (
    "910996c635b775e961e6b5d15d157e8febb793e514bc42ba88ea06ffc71e5d7c"
)
GM1_ACCEPTED_ONE_STEP_CHECKPOINT_CONTRACT_SHA256 = (
    "1ba302ea7450a21ed356201a58b3c2e9a863721ae2e8ad74915e1aed8bb7ec01"
)
GM1_ACCEPTED_ONE_STEP_CHECKPOINT_PAYLOAD_SHA256 = (
    "24e0bfed8f745edfb176c2f8893a780418781ef9c3afbfa745378d3c57d0b861"
)
GM1_ACCEPTED_EVOLVED_TKEA_SHA256 = (
    "427b90a1120e9bc8fd2bf93eb21d9165d785cd60db8e510402b8a033e3893a27"
)
GM1_ACCEPTED_FRESH_TKEA_SHA256 = (
    "3bb187c73f153935d4312d56d6225b1bc81665f5419dbddeba69c16e678ff19b"
)
GM1_ACCEPTED_PADDED_MOMENTUM_EVIDENCE_FILE_SHA256 = (
    "ebd885e810b51a090d4bd81434bdc7c9b32340a8549f96789478d6ed433ae030"
)
GM1_TILESTREAM_SOLVE3_TKE_A_CLAIM_LIMIT = (
    "literal CM1 r21.1 solve3.F:563-578 serial four-open padded tke3d-to-tkea "
    "copy over k=1..nk, j=0..nj+1, i=0..ni+1; detached real evolved-TKE "
    "checkpoint replay proves only this operator. Live RK2/RK3 TKE production, "
    "MPI exchange, rr/rf, scalar A, global psolver, radiation/open histories, "
    "full solve3/RK, forecast, tornado, and five-metre product remain refused"
)

_BUFFER_ORDER = ("tke3d", "tkea")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    detached = dict(receipt)
    detached.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(detached)).hexdigest()


def _sha_array(value: Any) -> str:
    return hashlib.sha256(
        value.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_array(name: str, value: Any, shape: tuple[int, ...], *, writable: bool) -> None:
    np = importlib.import_module("numpy")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f4")
        or value.shape != shape
        or not value.flags.c_contiguous
        or (writable and not value.flags.writeable)
    ):
        qualifier = "writable " if writable else ""
        raise ValueError(f"{name} must be {qualifier}host C-<f4 {shape}")


def reconstruct_gm1_four_open_padded_tke(physical: Any) -> Any:
    """Rebuild the one-cell ``bcw`` subset retained by solve3's copy range."""

    np = importlib.import_module("numpy")
    if (
        not isinstance(physical, np.ndarray)
        or physical.dtype != np.dtype("<f4")
        or physical.ndim != 3
        or min(physical.shape) < 2
        or not physical.flags.c_contiguous
        or not bool(np.all(np.isfinite(physical)))
    ):
        raise ValueError("physical TKE must be finite C-order little-endian float32")
    nk1, nj, ni = physical.shape
    result = np.zeros((nk1, nj + 2, ni + 2), dtype="<f4")
    result[:, 1:-1, 1:-1] = physical
    # bcw executes west/east before south/north for k=2..nk.  The second
    # phase consequently owns the final corner words.
    result[1:-1, 1:-1, 0] = result[1:-1, 1:-1, 1]
    result[1:-1, 1:-1, -1] = result[1:-1, 1:-1, -2]
    result[1:-1, 0, :] = result[1:-1, 1, :]
    result[1:-1, -1, :] = result[1:-1, -2, :]
    result[0].fill(np.float32(0.0))
    result[-1].fill(np.float32(0.0))
    return result


def reconstruct_gm1_padded_tkea_baseline(physical: Any) -> Any:
    """Pad real baseline ``tkea`` without falsely applying ``bcw`` to it.

    ``bcw`` acts on evolving ``tke3d``.  The persistent baseline retains its
    diagnosed bottom face until solve3 overwrites k=1..nk; only its model-top
    plane survives the copy and therefore must already be rigid zero.
    """

    np = importlib.import_module("numpy")
    if (
        not isinstance(physical, np.ndarray)
        or physical.dtype != np.dtype("<f4")
        or physical.ndim != 3
        or min(physical.shape) < 2
        or not physical.flags.c_contiguous
        or not bool(np.all(np.isfinite(physical)))
    ):
        raise ValueError("physical baseline TKE must be finite C-order float32")
    if np.count_nonzero(physical[-1].view(np.uint32)):
        raise ValueError("physical baseline TKE model top must be positive zero")
    nk1, nj, ni = physical.shape
    result = np.zeros((nk1, nj + 2, ni + 2), dtype="<f4")
    result[:, 1:-1, 1:-1] = physical
    result[:-1, 1:-1, 0] = result[:-1, 1:-1, 1]
    result[:-1, 1:-1, -1] = result[:-1, 1:-1, -2]
    result[:-1, 0, :] = result[:-1, 1, :]
    result[:-1, -1, :] = result[:-1, -2, :]
    return result


def tiled_solve3_tke_a_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    return MappingProxyType(
        {name: (nk, tile_ny, tile_nx) for name in _BUFFER_ORDER}
    )


def tiled_solve3_tke_a_buffer_nbytes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, int]:
    shapes = tiled_solve3_tke_a_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    return MappingProxyType(
        {
            "host_per_buffer_bytes": per_buffer,
            "device_per_buffer_bytes": per_buffer,
            "double_host_bytes": 2 * per_buffer,
            "double_device_bytes": 2 * per_buffer,
        }
    )


def build_gm1_solve3_tke_a_plan(
    *, ni: int, nj: int, tile_nx: int, tile_ny: int
) -> Gm1TilePlan:
    return build_gm1_open_tile_plan(
        nx=_positive_int("ni", ni) + 2,
        ny=_positive_int("nj", nj) + 2,
        tile_nx=_positive_int("tile_nx", tile_nx),
        tile_ny=_positive_int("tile_ny", tile_ny),
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3CompletedRk3TkeToken:
    """Future live token; this module intentionally provides no producer."""

    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3EvolvedTkeReplayToken:
    receipt: Mapping[str, Any]


def admit_gm1_evolved_tke_checkpoint_replay(
    physical_tkea: Any,
    extraction_receipt: Mapping[str, Any],
) -> Gm1TiledSolve3EvolvedTkeReplayToken:
    """Admit the exact persisted real one-step TKE as detached replay bytes."""

    _validate_array("physical_tkea", physical_tkea, (76, 300, 300), writable=False)
    np = importlib.import_module("numpy")
    if not bool(np.all(np.isfinite(physical_tkea))):
        raise ValueError("evolved checkpoint TKE contains non-finite words")
    digest = _sha_array(physical_tkea)
    checkpoint = extraction_receipt.get("checkpoint", {})
    array = extraction_receipt.get("array", {})
    if (
        extraction_receipt.get("schema")
        != "gm1.cm1fcp-single-array-extraction/v1"
        or extraction_receipt.get("status")
        != "PASS_HASH_BOUND_RAW_ARRAY_EXTRACTION"
        or checkpoint.get("artifact_sha256")
        != GM1_ACCEPTED_ONE_STEP_CHECKPOINT_ARTIFACT_SHA256
        or checkpoint.get("contract_sha256")
        != GM1_ACCEPTED_ONE_STEP_CHECKPOINT_CONTRACT_SHA256
        or checkpoint.get("payload_sha256")
        != GM1_ACCEPTED_ONE_STEP_CHECKPOINT_PAYLOAD_SHA256
        or array.get("name") != "closure.tkea"
        or array.get("shape") != [76, 300, 300]
        or array.get("nbytes") != 27_360_000
        or array.get("offset") != 0
        or array.get("sha256") != digest
        or digest != GM1_ACCEPTED_EVOLVED_TKEA_SHA256
    ):
        raise ValueError("evolved TKE extraction identity differs from sealed real step")
    if np.count_nonzero(physical_tkea[0].view(np.uint32)) or np.count_nonzero(
        physical_tkea[-1].view(np.uint32)
    ):
        raise ValueError("evolved TKE rigid bottom/top words changed")
    record = {
        "schema": "gm1.tilestream-evolved-tke-checkpoint-replay-token/v1",
        "status": "ADMITTED_DETACHED_REAL_EVOLVED_TKE_REPLAY_ONLY",
        "source_identity_sha256": GM1_SOURCE_IDENTITY_SHA256,
        "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "checkpoint_payload_sha256": checkpoint["payload_sha256"],
        "physical_tkea_sha256": digest,
        "physical_shape": [76, 300, 300],
        "completed_large_steps": 1,
        "persisted_after_tke_commit": True,
        "precommit_tke3d_snapshot_available": False,
        "completed_rk3_live_token": False,
        "stream_pointer": None,
        "ready_for": "detached solve3 TKE-copy replay operator only",
        "claim_limit": GM1_TILESTREAM_SOLVE3_TKE_A_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = _operation_token(record)
    return Gm1TiledSolve3EvolvedTkeReplayToken(MappingProxyType(record))


def _require_token(receipt: Mapping[str, Any], *, schema: str, status: str) -> None:
    if receipt.get("schema") != schema or receipt.get("status") != status:
        raise ValueError("TKE token schema or status changed")
    if receipt.get("source_identity_sha256") != GM1_SOURCE_IDENTITY_SHA256:
        raise ValueError("TKE token source identity changed")
    if receipt.get("operation_token_sha256") != _operation_token(receipt):
        raise ValueError("TKE token operation identity changed")


def _require_padded_momentum_token(
    token: object,
) -> Mapping[str, Any]:
    if not isinstance(token, Gm1TiledSolve3PaddedMomentumAToken):
        raise TypeError("padded_momentum_token must be its exact typed token")
    receipt = token.receipt
    _require_token(
        receipt,
        schema="gm1.tilestream-solve3-padded-momentum-a-token/v1",
        status="ADMITTED_PADDED_MOMENTUM_A_ONLY",
    )
    if (
        receipt.get("ready_for") != "independent TKE-commit admission audit only"
        or receipt.get("tke_commit_available") is not False
        or receipt.get("outputs") != dict(GM1_ACCEPTED_PADDED_OUTPUT_SHA256)
    ):
        raise ValueError("padded momentum token scope or real outputs changed")
    return receipt


def _require_replay_token(token: object) -> Mapping[str, Any]:
    if not isinstance(token, Gm1TiledSolve3EvolvedTkeReplayToken):
        raise TypeError("evolved_tke_token must be the detached replay token")
    receipt = token.receipt
    _require_token(
        receipt,
        schema="gm1.tilestream-evolved-tke-checkpoint-replay-token/v1",
        status="ADMITTED_DETACHED_REAL_EVOLVED_TKE_REPLAY_ONLY",
    )
    if (
        receipt.get("physical_tkea_sha256") != GM1_ACCEPTED_EVOLVED_TKEA_SHA256
        or receipt.get("completed_rk3_live_token") is not False
        or receipt.get("stream_pointer") is not None
    ):
        raise ValueError("detached replay token was relabelled as live")
    return receipt


def require_live_gm1_tiled_solve3_tke_a(
    token: object,
    *,
    stream_pointer: int,
) -> Mapping[str, Any]:
    """Fail closed until a real tile-streamed RK3/bcw producer exists."""

    if isinstance(token, Gm1TiledSolve3EvolvedTkeReplayToken):
        _require_replay_token(token)
        raise RuntimeError(
            "GM1-TILESTREAM-SOLVE3-TKE-A-LIVE-PREREQUISITE: persisted tkea is "
            "detached replay evidence, not same-stream post-RK3 tke3d"
        )
    if not isinstance(token, Gm1TiledSolve3CompletedRk3TkeToken):
        raise TypeError("live TKE commit requires the exact completed-RK3 token type")
    receipt = token.receipt
    _require_token(
        receipt,
        schema="gm1.tilestream-completed-rk3-tke-token/v1",
        status="ADMITTED_LIVE_COMPLETED_RK3_TKE_AND_BCW",
    )
    if (
        receipt.get("completed_nrk") != 3
        or receipt.get("bcw_complete") is not True
        or receipt.get("same_stream_no_host_sync") is not True
        or receipt.get("stream_pointer") != stream_pointer
    ):
        raise ValueError("live completed-RK3 TKE token cadence or stream changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3TkeAReplayBinding:
    tke3d: Any
    tkea: Any
    padded_momentum_token: Gm1TiledSolve3PaddedMomentumAToken
    evolved_tke_token: Gm1TiledSolve3EvolvedTkeReplayToken
    ni: int
    nj: int
    nk: int
    padded_momentum_evidence_file_sha256: str

    @classmethod
    def create(
        cls,
        *,
        tke3d: Any,
        tkea: Any,
        padded_momentum_token: Gm1TiledSolve3PaddedMomentumAToken,
        evolved_tke_token: Gm1TiledSolve3EvolvedTkeReplayToken,
        ni: int,
        nj: int,
        nk: int,
        padded_momentum_evidence_file_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3TkeAReplayBinding":
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        ni, nj, nk = (
            _positive_int(name, value)
            for name, value in (("ni", ni), ("nj", nj), ("nk", nk))
        )
        shape = (nk + 1, nj + 2, ni + 2)
        _validate_array("tke3d", tke3d, shape, writable=False)
        _validate_array("tkea", tkea, shape, writable=True)
        np = importlib.import_module("numpy")
        if np.shares_memory(tke3d, tkea):
            raise ValueError("padded tke3d and tkea must not alias")
        if not all(bool(np.all(np.isfinite(value))) for value in (tke3d, tkea)):
            raise ValueError("padded TKE replay arrays contain non-finite words")
        _require_padded_momentum_token(padded_momentum_token)
        replay = _require_replay_token(evolved_tke_token)
        if allow_compact_validation:
            if min(ni, nj, nk) < 2:
                raise ValueError("compact TKE replay geometry is too small")
        else:
            if (ni, nj, nk) != (300, 300, 75):
                raise ValueError("real TKE replay geometry changed")
            if (
                padded_momentum_evidence_file_sha256
                != GM1_ACCEPTED_PADDED_MOMENTUM_EVIDENCE_FILE_SHA256
            ):
                raise ValueError("padded momentum evidence file identity changed")
            if _sha_array(tke3d[:, 1:-1, 1:-1]) != replay["physical_tkea_sha256"]:
                raise ValueError("padded tke3d interior differs from evolved replay token")
            if _sha_array(tkea[:, 1:-1, 1:-1]) != GM1_ACCEPTED_FRESH_TKEA_SHA256:
                raise ValueError("padded tkea baseline differs from real fresh closure")
            if not np.array_equal(
                tke3d.view(np.uint32),
                reconstruct_gm1_four_open_padded_tke(
                    np.ascontiguousarray(tke3d[:, 1:-1, 1:-1])
                ).view(np.uint32),
            ):
                raise ValueError("padded tke3d does not contain exact four-open bcw words")
            if np.count_nonzero(tkea[-1].view(np.uint32)):
                raise ValueError("padded fresh tkea model top is not positive zero")
        return cls(
            tke3d,
            tkea,
            padded_momentum_token,
            evolved_tke_token,
            ni,
            nj,
            nk,
            padded_momentum_evidence_file_sha256,
        )


def fill_gm1_tiled_solve3_tke_a_source(
    binding: Gm1TiledSolve3TkeAReplayBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3TkeAReplayBinding):
        raise TypeError("binding must be Gm1TiledSolve3TkeAReplayBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.ni + 2,
        binding.nj + 2,
    ):
        raise ValueError("tile spec does not cover padded TKE plane")
    shapes = tiled_solve3_tke_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("TKE replay tile inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name, shape in shapes.items():
        _validate_array(name, target[name], shape, writable=True)
        target[name].fill(0)
        target[name][:, :active_ny, :active_nx] = getattr(binding, name)[
            : binding.nk, y0:y1, x0:x1
        ]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_shape": [binding.nk, active_ny, active_nx],
            "copy_fortran_range": {
                "k": [1, binding.nk],
                "j": [0, binding.nj + 1],
                "i": [0, binding.ni + 1],
            },
        }
    )


def cpu_gm1_solve3_tke_a_resident_inplace(
    binding: Gm1TiledSolve3TkeAReplayBinding,
) -> None:
    if not isinstance(binding, Gm1TiledSolve3TkeAReplayBinding):
        raise TypeError("binding must be Gm1TiledSolve3TkeAReplayBinding")
    binding.tkea[: binding.nk] = binding.tke3d[: binding.nk]


def cpu_gm1_tiled_solve3_tke_a_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    nk: int,
) -> None:
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError("CPU TKE replay tile inventory/order changed")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    nk = _positive_int("nk", nk)
    arrays["tkea"][:nk, :active_ny, :active_nx] = arrays["tke3d"][
        :nk, :active_ny, :active_nx
    ]


def scatter_gm1_tiled_solve3_tke_a_output(
    binding: Gm1TiledSolve3TkeAReplayBinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    shapes = tiled_solve3_tke_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("TKE replay scatter inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    binding.tkea[: binding.nk, y0:y1, x0:x1] = source["tkea"][
        :, :active_ny, :active_nx
    ]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "copied_words": binding.nk * active_ny * active_nx,
            "top_tkea_mutated": False,
            "live_chain_advanced": False,
        }
    )


CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE = r'''// gm1.cuda-tilestream-solve3-padded-tke-current-to-a-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
extern "C" __global__ void gm1_tiled_solve3_tke_a_f32_v1(
 const float*tke3d,float*tkea,unsigned active_nx,unsigned active_ny,
 unsigned nk,unsigned tile_nx,unsigned tile_ny){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long words=(unsigned long long)nk*active_ny*active_nx;
 if(q<words){unsigned i=(unsigned)(q%active_nx);unsigned long long t=q/active_nx;
  unsigned j=(unsigned)(t%active_ny),k=(unsigned)(t/active_ny);
  unsigned long long z=((unsigned long long)k*tile_ny+j)*tile_nx+i;
  tkea[z]=tke3d[z];}
}
'''
CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3TkeA:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_tke_a() -> CompiledGm1TiledSolve3TkeA:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE,
        "gm1_tiled_solve3_tke_a_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE,
        "gm1_tiled_solve3_tke_a_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks the TKE current-to-A entry")
    module = function.Module()
    module.load(cubin)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledGm1TiledSolve3TkeA(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3TkeALaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3TkeA
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_tke_a_replay(
    arrays: Mapping[str, Any],
    binding: Gm1TiledSolve3TkeAReplayBinding,
    spec: Gm1TileSpec,
    *,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3TkeA | None = None,
) -> CudaGm1TiledSolve3TkeALaunch:
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_tke_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA TKE replay tile inventory/order changed")
    ranges = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        start = int(value.data.ptr)
        ranges.append((start, start + int(value.nbytes)))
    if ranges[0][0] < ranges[1][1] and ranges[1][0] < ranges[0][1]:
        raise ValueError("CUDA TKE replay arrays must not alias")
    momentum = _require_padded_momentum_token(binding.padded_momentum_token)
    replay = _require_replay_token(binding.evolved_tke_token)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compile_cuda_gm1_tiled_solve3_tke_a() if compiled is None else compiled
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    words = binding.nk * active_ny * active_nx
    threads = 256
    with stream:
        compiled.kernel(
            ((words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                arrays["tke3d"],
                arrays["tkea"],
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(binding.nk),
                cupy.uint32(spec.tile_nx),
                cupy.uint32(spec.tile_ny),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_ABI,
        "status": "ENQUEUED_DETACHED_REAL_TKE_A_REPLAY_TILE",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "copied_words": words,
        "source_identity_sha256": GM1_SOURCE_IDENTITY_SHA256,
        "padded_momentum_operation_token_sha256": momentum[
            "operation_token_sha256"
        ],
        "evolved_tke_replay_operation_token_sha256": replay[
            "operation_token_sha256"
        ],
        "stream_pointer": int(stream.ptr),
        "same_stream_as_archived_tokens": False,
        "live_chain_advanced": False,
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "refusals": {
            "live_completed_rk3_token": True,
            "rk2_rk3_tke_evolution": True,
            "mpi_exchange": True,
            "scalar_current_to_a": True,
            "reciprocal_density": True,
            "global_psolver": True,
            "open_history_mutation": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
    }
    record["operation_token_sha256"] = _operation_token(record)
    return CudaGm1TiledSolve3TkeALaunch(
        event, compiled, MappingProxyType(record)
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3TkeAReplayToken:
    receipt: Mapping[str, Any]


def finish_gm1_tiled_solve3_tke_a_replay(
    binding: Gm1TiledSolve3TkeAReplayBinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledSolve3TkeALaunch],
) -> Gm1TiledSolve3TkeAReplayToken:
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.ni + 2,
        binding.nj + 2,
    ):
        raise ValueError("TKE replay completion plan differs from padded domain")
    if len(launches) != plan.ntiles:
        raise ValueError("TKE replay completion requires every exact tile")
    momentum = _require_padded_momentum_token(binding.padded_momentum_token)
    replay = _require_replay_token(binding.evolved_tke_token)
    tokens = []
    streams = set()
    copied_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve3TkeALaunch):
            raise TypeError("TKE replay completion requires exact typed launches")
        receipt = launch.receipt
        if (
            launch.ready_event is None
            or receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SCHEMA
            or receipt.get("status") != "ENQUEUED_DETACHED_REAL_TKE_A_REPLAY_TILE"
            or receipt.get("tile") != spec.index
            or receipt.get("padded_momentum_operation_token_sha256")
            != momentum["operation_token_sha256"]
            or receipt.get("evolved_tke_replay_operation_token_sha256")
            != replay["operation_token_sha256"]
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("TKE replay tile receipt identity/order changed")
        tokens.append(receipt["operation_token_sha256"])
        streams.add(int(receipt["stream_pointer"]))
        copied_words += int(receipt["copied_words"])
    if len(streams) != 1:
        raise ValueError("TKE replay tiles changed CUDA streams")
    expected_words = binding.nk * (binding.ni + 2) * (binding.nj + 2)
    if copied_words != expected_words:
        raise ValueError("TKE replay tile coverage is not exact once")
    np = importlib.import_module("numpy")
    if not bool(np.all(np.isfinite(binding.tkea))):
        raise ValueError("completed TKE replay output is non-finite")
    record = {
        "schema": "gm1.tilestream-solve3-tke-a-replay-token/v1",
        "status": "ADMITTED_DETACHED_REAL_TKE_A_REPLAY_OPERATOR_ONLY",
        "source_identity_sha256": GM1_SOURCE_IDENTITY_SHA256,
        "padded_momentum_operation_token_sha256": momentum[
            "operation_token_sha256"
        ],
        "evolved_tke_replay_operation_token_sha256": replay[
            "operation_token_sha256"
        ],
        "tile_count": plan.ntiles,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_operation_tokens": tokens,
        "copied_words": copied_words,
        "stream_pointer": next(iter(streams)),
        "output_tkea_sha256": _sha_array(binding.tkea),
        "output_physical_tkea_sha256": _sha_array(binding.tkea[:, 1:-1, 1:-1]),
        "top_level_sha256": _sha_array(binding.tkea[-1]),
        "top_level_mutated": False,
        "live_chain_advanced": False,
        "ready_for": "operator evidence only; no downstream live source-order seam",
        "claim_limit": GM1_TILESTREAM_SOLVE3_TKE_A_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = _operation_token(record)
    return Gm1TiledSolve3TkeAReplayToken(MappingProxyType(record))


def five_meter_tiled_solve3_tke_a_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    memory = dict(
        tiled_solve3_tke_a_buffer_nbytes(
            tile_nx=tile_nx, tile_ny=tile_ny, nk=75
        )
    )
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-tke-a-memory/v1",
        "domain": [75, 3000, 3000],
        "padded_plane": [3002, 3002],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3002 / tile_nx) * math.ceil(3002 / tile_ny),
        "carriers": list(_BUFFER_ORDER),
        **memory,
        "full_domain_duplicate_bytes": 0,
        "canonical_store_mutations": ["tkea(k=1..nk,i=0..ni+1,j=0..nj+1)"],
        "live_completed_rk3_token_available": False,
        "global_pressure_workspace_executed": False,
        "open_history_mutation": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


def audit_gm1_tiled_solve3_tke_a_admission() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "schema": "gm1.tilestream-solve3-tke-a-admission-audit/v1",
            "status": "BLOCKED_LIVE_ADMITTED_DETACHED_REPLAY_ONLY",
            "newly_unlocked": "real evolved-TKE detached copy replay",
            "live_unmet_prerequisite": (
                "same-stream typed post-RK3 tke3d token after per-stage live "
                "rru/rrv/rrw/divx rebuilds and final bcw"
            ),
            "fresh_closure_is_completed_rk3": False,
            "persisted_post_commit_tkea_is_precommit_tke3d_snapshot": False,
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
                "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
                "bc_sha256": AUTHORITY_BC_SHA256,
                "anchors": list(AUTHORITY_ANCHORS),
            },
            "structural_refusals": [
                "rr/rf",
                "scalar current-to-A",
                "global psolver",
                "MPI exchange",
                "radiation/open histories",
                "full RK or solve3",
                "five-metre meteorological product",
            ],
            "claim_limit": GM1_TILESTREAM_SOLVE3_TKE_A_CLAIM_LIMIT,
        }
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_KERNEL_NAME",
    "CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_TKE_A_SOURCE_SHA256",
    "CompiledGm1TiledSolve3TkeA",
    "CudaGm1TiledSolve3TkeALaunch",
    "GM1_ACCEPTED_EVOLVED_TKEA_SHA256",
    "GM1_ACCEPTED_FRESH_TKEA_SHA256",
    "GM1_TILESTREAM_SOLVE3_TKE_A_CLAIM_LIMIT",
    "Gm1TiledSolve3CompletedRk3TkeToken",
    "Gm1TiledSolve3EvolvedTkeReplayToken",
    "Gm1TiledSolve3TkeAReplayBinding",
    "Gm1TiledSolve3TkeAReplayToken",
    "admit_gm1_evolved_tke_checkpoint_replay",
    "audit_gm1_tiled_solve3_tke_a_admission",
    "build_gm1_solve3_tke_a_plan",
    "compile_cuda_gm1_tiled_solve3_tke_a",
    "cpu_gm1_solve3_tke_a_resident_inplace",
    "cpu_gm1_tiled_solve3_tke_a_inplace",
    "fill_gm1_tiled_solve3_tke_a_source",
    "finish_gm1_tiled_solve3_tke_a_replay",
    "five_meter_tiled_solve3_tke_a_inventory",
    "launch_cuda_gm1_tiled_solve3_tke_a_replay",
    "reconstruct_gm1_four_open_padded_tke",
    "reconstruct_gm1_padded_tkea_baseline",
    "require_live_gm1_tiled_solve3_tke_a",
    "scatter_gm1_tiled_solve3_tke_a_output",
    "tiled_solve3_tke_a_buffer_nbytes",
    "tiled_solve3_tke_a_buffer_shapes",
)
