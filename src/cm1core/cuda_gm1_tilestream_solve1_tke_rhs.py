"""Live tile-streamed CM1 solve1 SGS-TKE right-hand side.

The admitted source order is deliberately narrow: a typed stretched fresh-
closure tile is followed, on the same CUDA stream, by the selected ``tconfig=1``
local TKE source and one terrain ``turbt`` diffusion call.  The completed token
binds the exactly-once host-scattered ``tketen`` carrier.  It is not an RK token:
solve2 still requires a new post-acoustic mass-flux/divergence producer for
every RK stage.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_stretched_sgs_tke import (
    CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
    CompiledChildStretchedSgsTke,
    compile_cuda_gm1_child_stretched_sgs_tke,
)
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_stretched_closure import (
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA,
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
    CudaGm1TiledStretchedClosureLaunch,
    Gm1TiledStretchedClosureToken,
)
from .nvrtc_module import compile_nvrtc_module


CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA = (
    "gm1.cuda-tilestream-solve1-tke-rhs/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_ABI = (
    "gm1.cuda-tilestream-solve1-local-source-turbt-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME = (
    "gm1_tiled_solve1_tke_local_source_f32_v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA = (
    "gm1.tilestream-solve1-tke-rhs-token/v1"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE1_SHA256 = (
    "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
)
AUTHORITY_TURBTEND_SHA256 = (
    "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
)
AUTHORITY_ANCHORS = (
    "src/solve1.F:314-345",
    "src/solve1.F:362-369",
    "src/turbtend.F:700-1044",
)
GM1_TILESTREAM_SOLVE1_TKE_RHS_CLAIM_LIMIT = (
    "live tile-local CM1 r21.1 tconfig=1 SGS-TKE local source followed once "
    "by terrain implicit turbt at solve1 cadence, for the serial four-open "
    "positive-normal stretched child; exact owned-rectangle host scatter is "
    "required. solve2 acoustic state, per-stage rru/rrv/rrw/divx, TKE RK, "
    "bcw, solve3, global psolver, MPI, histories, forecast, tornado, and a "
    "five-metre meteorological product remain refused"
)

_REQUIRED_ARRAY_SHAPES = MappingProxyType(
    {
        "tke3d": "face",
        "tketen": "face",
        "kmh": "face",
        "kmv": "face",
        "khv": "face",
        "nm": "face",
        "dissten": "face",
        "defv": "face",
        "defh": "face",
        "rho": "mass",
        "rf": "face",
        "mh": "mass",
        "mf": "face",
        "gz": "surface",
        "rgz": "surface",
        "rgzu": "u_surface",
        "rgzv": "v_surface",
        "sigma": "profile",
        "rds": "profile",
    }
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


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _vertical_disposition(allow_uniform_idealized_validation: bool) -> str:
    """Match the exact upstream closure disposition consumed by full-held."""

    return (
        "source-audited-flat-idealized-uniform-200m"
        if allow_uniform_idealized_validation
        else "sealed-stretched-child"
    )


def _geometry(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    dt: float,
    *,
    allow_compact_validation: bool,
) -> tuple[int, int, int, float, float, float, float, float]:
    ni = _positive_int("ni", ni, 7)
    nj = _positive_int("nj", nj, 7)
    nk = _positive_int("nk", nk, 8)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    values = []
    for name, value in (
        ("dx", dx),
        ("dy", dy),
        ("dz", dz),
        ("ztop", ztop),
        ("dt", dt),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
        values.append(result)
    dx, dy, dz, ztop, dt = values
    if not allow_compact_validation and (
        (ni, nj, nk) not in ((300, 300, 75), (3000, 3000, 75))
        or (dx, dy) != ((50.0, 50.0) if ni == 300 else (5.0, 5.0))
        or dz != 100.0
        or ztop != 15000.0
    ):
        raise ValueError("solve1 TKE RHS geometry differs from an admitted grid")
    if nk > 75:
        raise ValueError("solve1 TKE RHS exceeds the sealed column workspace")
    return ni, nj, nk, dx, dy, dz, ztop, dt


def _shape(kind: str, ni: int, nj: int, nk: int) -> tuple[int, ...]:
    return {
        "face": (nk + 1, nj, ni),
        "mass": (nk, nj, ni),
        "surface": (nj, ni),
        "u_surface": (nj, ni + 1),
        "v_surface": (nj + 1, ni),
        "profile": (nk + 2,),
    }[kind]


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    if not isinstance(arrays, Mapping) or set(arrays) != set(_REQUIRED_ARRAY_SHAPES):
        raise ValueError("solve1 TKE RHS device inventory changed")
    result: dict[str, Any] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, kind in _REQUIRED_ARRAY_SHAPES.items():
        value = arrays[name]
        expected = _shape(kind, ni, nj, nk)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != expected
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {expected}")
        result[name] = value
        start = int(value.data.ptr)
        ranges.append((start, start + int(value.nbytes), name))
    # The local source and diffusion are in-place only on tketen.  An alias
    # would silently turn source order into a different numerical method.
    output = next(item for item in ranges if item[2] == "tketen")
    for start, stop, name in ranges:
        if name != "tketen" and output[0] < stop and start < output[1]:
            raise ValueError(f"tketen aliases live input {name}")
    return result


CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE = r'''// gm1.cuda-tilestream-solve1-tke-rhs/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires binary32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);}
__device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);}
__device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);}
extern "C" __global__ void gm1_tiled_solve1_tke_local_source_f32_v1(
 const float*khv,const float*nm,const float*dissten,const float*kmv,
 const float*defv,const float*defh,float*tketen,
 unsigned ni,unsigned nj,unsigned nk){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long nxy=(unsigned long long)ni*nj,words=nxy*(nk-1u);
 if(q>=words)return;
 unsigned long long p=q+nxy;
 float buoyancy=M(-khv[p],nm[p]);
 float source=S(buoyancy,dissten[p]);
 float deformation=fmaxf(0.0f,A(defv[p],defh[p]));
 tketen[p]=A(source,M(kmv[p],deformation));
}
'''
CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE.encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve1TkeRhs:
    local_module: Any
    local_kernel: Any
    stretched: CompiledChildStretchedSgsTke
    target_arch: str
    runtime_identity: Mapping[str, object]


def compile_cuda_gm1_tiled_solve1_tke_rhs() -> CompiledGm1TiledSolve1TkeRhs:
    cupy = importlib.import_module("cupy")
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    local = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE,
        options=options,
        name_expressions=(CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME,),
        filename="gm1_tilestream_solve1_tke.cu",
    )
    stretched = compile_cuda_gm1_child_stretched_sgs_tke()
    return CompiledGm1TiledSolve1TkeRhs(
        local,
        local.get_function(CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME),
        stretched,
        stretched.target_arch,
        stretched.runtime_identity,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve1TkeRhsLaunch:
    ready_event: Any
    tketen: Any
    compiled: CompiledGm1TiledSolve1TkeRhs
    receipt: Mapping[str, object]


def _launch_rhs(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    dt: float,
    stream: Any,
    compiled: CompiledGm1TiledSolve1TkeRhs,
    wait_event: Any | None,
    allow_compact_validation: bool,
) -> tuple[Any, dict[str, Any]]:
    ni, nj, nk, dx, dy, dz, ztop, dt = _geometry(
        ni,
        nj,
        nk,
        dx,
        dy,
        dz,
        ztop,
        dt,
        allow_compact_validation=allow_compact_validation,
    )
    cupy = importlib.import_module("cupy")
    values = _require_device_arrays(cupy, arrays, ni, nj, nk)
    threads = 128
    source_words = (nk - 1) * nj * ni
    columns = nj * ni
    with stream:
        if wait_event is not None:
            stream.wait_event(wait_event)
        compiled.local_kernel(
            ((source_words + threads - 1) // threads,),
            (threads,),
            (
                values["khv"],
                values["nm"],
                values["dissten"],
                values["kmv"],
                values["defv"],
                values["defh"],
                values["tketen"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
            ),
            stream=stream,
        )
        compiled.stretched.tke_kernels[0](
            ((columns + threads - 1) // threads,),
            (threads,),
            (
                values["tke3d"],
                values["kmh"],
                values["kmv"],
                values["rho"],
                values["rf"],
                values["mh"],
                values["mf"],
                values["gz"],
                values["rgz"],
                values["rgzu"],
                values["rgzv"],
                values["tketen"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
                cupy.float32(dt),
                values["sigma"],
                values["rds"],
                cupy.float32(ztop),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return event, {
        "compute_shape_zyx": [nk, nj, ni],
        "source_words": source_words,
        "diffusion_columns": columns,
        "spacing_m": {
            "dx": dx,
            "dy": dy,
            "reference_dz": dz,
            "ztop": ztop,
        },
        "dt_seconds": dt,
    }


def launch_cuda_gm1_resident_solve1_tke_rhs(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    dt: float,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve1TkeRhs | None = None,
    wait_event: Any | None = None,
    allow_compact_validation: bool = False,
) -> CudaGm1TiledSolve1TkeRhsLaunch:
    """Launch the same kernels residently for the raw-bit oracle."""

    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compile_cuda_gm1_tiled_solve1_tke_rhs() if compiled is None else compiled
    event, detail = _launch_rhs(
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        ztop=ztop,
        dt=dt,
        stream=stream,
        compiled=compiled,
        wait_event=wait_event,
        allow_compact_validation=allow_compact_validation,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_ABI,
        "status": "ENQUEUED_RESIDENT_SOLVE1_TKE_RHS_ORACLE_ONLY",
        **detail,
        "stream_pointer": int(stream.ptr),
        "local_source_sha256": CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256,
        "turbt_source_sha256": CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_TKE_RHS_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledSolve1TkeRhsLaunch(
        event, arrays["tketen"], compiled, MappingProxyType(receipt)
    )


def _require_closure_launch(
    launch: object, spec: Gm1TileSpec, stream_pointer: int
) -> Mapping[str, object]:
    if not isinstance(launch, CudaGm1TiledStretchedClosureLaunch):
        raise TypeError("closure_launch must be the exact typed closure tile")
    receipt = launch.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA
        or receipt.get("status")
        != "ENQUEUED_TILE_LOCAL_CALCNM_CALCDEF_FRESH_TKE_PENDING_GLOBAL_REDUCTION"
        or receipt.get("tile") != spec.record()
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_no_host_sync") is not True
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("fresh closure tile identity, order, or stream changed")
    return receipt


def launch_cuda_gm1_tiled_solve1_tke_rhs(
    closure_launch: CudaGm1TiledStretchedClosureLaunch,
    closure_inputs: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    sigma: Any,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    dt: float,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve1TkeRhs | None = None,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledSolve1TkeRhsLaunch:
    """Enqueue local source and ``turbt`` after one live closure tile."""

    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if not isinstance(allow_uniform_idealized_validation, bool):
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if allow_uniform_idealized_validation:
        if (
            allow_compact_validation
            or (spec.nx, spec.ny, nk) != (300, 300, 75)
            or (float(dx), float(dy), float(dz), float(ztop))
            != (50.0, 50.0, 200.0, 15000.0)
        ):
            raise ValueError(
                "uniform idealized solve1 TKE requires exact 300x300x75, "
                "50/50/200 m, 15 km geometry"
            )
    elif not allow_compact_validation:
        admitted = {
            (300, 300, 75): (50.0, 50.0),
            (3000, 3000, 75): (5.0, 5.0),
        }
        expected_spacing = admitted.get((spec.nx, spec.ny, nk))
        if (
            expected_spacing is None
            or (float(dx), float(dy)) != expected_spacing
            or float(dz) != 100.0
            or float(ztop) != 15000.0
        ):
            raise ValueError("solve1 TKE RHS tile belongs to an unadmitted grid")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    closure_receipt = _require_closure_launch(closure_launch, spec, int(stream.ptr))
    if not isinstance(closure_inputs, Mapping) or not {"rho", "rf"}.issubset(
        closure_inputs
    ):
        raise ValueError("closure inputs must retain rho and rf")
    work = closure_launch.workspace
    # ``sigma`` is owned by the shared live CudaStretchZ1Launch rather than by
    # the per-tile workspace, so it is an explicit sequencing operand.
    arrays = {
        name: (
            closure_inputs[name]
            if name in {"rho", "rf"}
            else sigma
            if name == "sigma"
            else work[name]
        )
        for name in _REQUIRED_ARRAY_SHAPES
    }
    compiled = compile_cuda_gm1_tiled_solve1_tke_rhs() if compiled is None else compiled
    event, detail = _launch_rhs(
        arrays,
        ni=spec.cnx,
        nj=spec.cny,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        ztop=ztop,
        dt=dt,
        stream=stream,
        compiled=compiled,
        wait_event=closure_launch.ready_event,
        # The global production geometry was admitted above.  The kernel sees
        # the fixed compute-window extents, which are intentionally not a
        # standalone production domain.
        allow_compact_validation=True,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_ABI,
        "status": "ENQUEUED_LIVE_TILE_LOCAL_SOLVE1_TKE_RHS",
        "tile": spec.record(),
        "owned_local_yx": closure_receipt["owned_local_yx"],
        "closure_operation_token_sha256": closure_receipt[
            "operation_token_sha256"
        ],
        **detail,
        "stream_pointer": int(stream.ptr),
        "same_stream_after_fresh_closure": True,
        "vertical_disposition": _vertical_disposition(
            allow_uniform_idealized_validation
        ),
        "closure_global_reduction_pending": True,
        "local_source_sha256": CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256,
        "turbt_source_sha256": CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "completed_nrk": 0,
        "post_acoustic_flux_available": False,
        "claim_limit": GM1_TILESTREAM_SOLVE1_TKE_RHS_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledSolve1TkeRhsLaunch(
        event, arrays["tketen"], compiled, MappingProxyType(receipt)
    )


def _require_closure_token(
    token: object, plan: Gm1TilePlan
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledStretchedClosureToken):
        raise TypeError("closure_token must be the exact completed closure token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"
        or receipt.get("tile_plan_sha256") != plan.ownership_sha256
        or receipt.get("tile_count") != plan.ntiles
        or receipt.get("invalid_interior_cell_count") != 0
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("completed fresh closure token identity changed")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve1TkeRhsToken:
    receipt: Mapping[str, object]


def finish_cuda_gm1_tiled_solve1_tke_rhs(
    closure_token: Gm1TiledStretchedClosureToken,
    launches: Sequence[CudaGm1TiledSolve1TkeRhsLaunch],
    *,
    plan: Gm1TilePlan,
    output_tketen: Any,
    nk: int,
) -> Gm1TiledSolve1TkeRhsToken:
    """Admit exact coverage and bind the complete live host TKE tendency."""

    if not isinstance(plan, Gm1TilePlan):
        raise TypeError("plan must be Gm1TilePlan")
    closure = _require_closure_token(closure_token, plan)
    if len(launches) != plan.ntiles:
        raise ValueError("solve1 TKE RHS requires every exact tile launch")
    closure_tokens: list[str] = []
    rhs_tokens: list[str] = []
    streams: set[int] = set()
    owned_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve1TkeRhsLaunch):
            raise TypeError("solve1 TKE RHS completion requires typed launches")
        receipt = launch.receipt
        owned = spec.record()["owned_mass_yx"]
        tile_words = (nk + 1) * (owned[1] - owned[0]) * (owned[3] - owned[2])
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA
            or receipt.get("status") != "ENQUEUED_LIVE_TILE_LOCAL_SOLVE1_TKE_RHS"
            or receipt.get("tile") != spec.record()
            or receipt.get("same_stream_after_fresh_closure") is not True
            or receipt.get("completed_nrk") != 0
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("solve1 TKE RHS tile identity or source order changed")
        closure_tokens.append(str(receipt["closure_operation_token_sha256"]))
        rhs_tokens.append(str(receipt["operation_token_sha256"]))
        streams.add(int(receipt["stream_pointer"]))
        owned_words += tile_words
    if len(streams) != 1 or next(iter(streams)) != closure.get("stream_pointer"):
        raise ValueError("solve1 TKE RHS tiles differ from the closure CUDA stream")
    closure_digest = hashlib.sha256(_canonical(sorted(closure_tokens))).hexdigest()
    if closure_digest != closure.get("tile_operation_tokens_sha256"):
        raise ValueError("solve1 TKE RHS did not consume every admitted closure tile")
    np = importlib.import_module("numpy")
    shape = (nk + 1, plan.ny, plan.nx)
    if (
        not isinstance(output_tketen, np.ndarray)
        or output_tketen.dtype != np.dtype("<f4")
        or output_tketen.shape != shape
        or not output_tketen.flags.c_contiguous
        or not bool(np.all(np.isfinite(output_tketen)))
    ):
        raise ValueError(f"output_tketen must be finite host C-<f4 {shape}")
    if np.count_nonzero(output_tketen[[0, -1]].view(np.uint32)):
        raise ValueError("solve1 TKE RHS changed rigid bottom/top tendency faces")
    expected_words = math.prod(shape)
    if owned_words != expected_words:
        raise ValueError("solve1 TKE RHS host scatter coverage is not exact once")
    output_sha = hashlib.sha256(output_tketen.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
        "closure_operation_token_sha256": closure["operation_token_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(rhs_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "same_stream_per_tile_after_fresh_closure": True,
        "host_scatter_exact_once": True,
        "output_tketen_shape": list(shape),
        "output_tketen_words": expected_words,
        "output_tketen_sha256": output_sha,
        "completed_nrk": 0,
        "rk_stage_available": False,
        "post_acoustic_flux_available": False,
        "ready_for": (
            "solve2 TKE RK1 only after an exact typed live nrk=1 post-acoustic "
            "rru/rrv/rrw/divx tile-stream token"
        ),
        "structural_refusals": [
            "frozen or detached mass flux",
            "RK2/RK3 without per-stage post-acoustic rebuild",
            "solve3 TKE commit",
            "global psolver substitution",
            "MPI exchange",
            "radiation/open histories",
            "full RK or forecast",
            "five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_TKE_RHS_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve1TkeRhsToken(MappingProxyType(receipt))


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve2PostAcousticFluxToken:
    """Future live token; this module intentionally provides no producer."""

    receipt: Mapping[str, object]


def require_gm1_tiled_solve2_tke_rk1(
    solve1_token: object, post_acoustic_flux_token: object
) -> None:
    """Fail closed until a live tile-streamed nrk=1 flux producer exists."""

    if not isinstance(solve1_token, Gm1TiledSolve1TkeRhsToken):
        raise TypeError("solve1_token must be the exact live TKE RHS token")
    receipt = solve1_token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve1 TKE RHS token identity changed")
    if not isinstance(post_acoustic_flux_token, Gm1TiledSolve2PostAcousticFluxToken):
        raise RuntimeError(
            "GM1-TILESTREAM-TKE-RK1-PREREQUISITE: exact live nrk=1 "
            "post-acoustic rru/rrv/rrw/divx token is not available"
        )
    flux = post_acoustic_flux_token.receipt
    if (
        flux.get("schema") != "gm1.tilestream-post-acoustic-mass-flux-token/v1"
        or flux.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or flux.get("nrk") != 1
        or flux.get("stream_pointer") != receipt.get("stream_pointer")
        or flux.get("operation_token_sha256") != _operation_token(flux)
    ):
        raise ValueError("post-acoustic mass-flux token cadence or stream changed")


def five_meter_tiled_solve1_tke_rhs_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk, 8)
    record = {
        "schema": "gm1.hybrid-tilestream-solve1-tke-rhs-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "reused_fresh_closure_carriers": list(_REQUIRED_ARRAY_SHAPES),
        "incremental_global_device_buffer_bytes": 0,
        "column_workspace_storage": "CUDA thread-local fixed nk<=75 arrays in turbt",
        "canonical_store_mutations": ["tketen(k=2..nk,i=1..ni,j=1..nj)"],
        "post_acoustic_flux_token_available": False,
        "completed_rk_stage_available": False,
        "global_pressure_workspace_executed": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE1_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "AUTHORITY_TURBTEND_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_KERNEL_NAME",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_SOURCE_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA",
    "CompiledGm1TiledSolve1TkeRhs",
    "CudaGm1TiledSolve1TkeRhsLaunch",
    "GM1_TILESTREAM_SOLVE1_TKE_RHS_CLAIM_LIMIT",
    "Gm1TiledSolve1TkeRhsToken",
    "Gm1TiledSolve2PostAcousticFluxToken",
    "compile_cuda_gm1_tiled_solve1_tke_rhs",
    "finish_cuda_gm1_tiled_solve1_tke_rhs",
    "five_meter_tiled_solve1_tke_rhs_inventory",
    "launch_cuda_gm1_resident_solve1_tke_rhs",
    "launch_cuda_gm1_tiled_solve1_tke_rhs",
    "require_gm1_tiled_solve2_tke_rk1",
)
