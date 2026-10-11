"""Tile-local CM1 post-WSM6 EOS rebuild and physical ``qa`` publication.

This boundary starts with the exact device carriers returned by
``cuda_gm1_tilestream_wsm6``.  Before that WSM6 call, callers use the snapshot
entry point here to preserve CM1's old temperature, water-vapour, and Exner
perturbation words.  After WSM6, the finish entry point applies the literal
ptype-5/equation-set-2 threshold and EOS order from ``mp_driver.F:603-617``.
It then publishes the six physical current moisture carriers into ``qa``.

The ``qa`` copy is the bounded physical-view portion of ``solve3.F:624-643``.
It is not solve3 admission: pressure completion, halos, boundary histories,
MPI exchange, all other current-to-A copies, and a complete RK step remain
fail-closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .cuda_gm1_tilestream_wsm6 import (
    CUDA_GM1_TILESTREAM_WSM6_ABI,
    CUDA_GM1_TILESTREAM_WSM6_SCHEMA,
    GM1_ACCEPTED_REAL_PDEFQ_RECEIPT_SHA256,
    CudaGm1TiledWsm6Launch,
    Gm1TiledWsm6SourceBinding,
    five_meter_tiled_wsm6_inventory,
    scatter_gm1_tiled_wsm6_output,
    tiled_wsm6_device_buffer_shapes,
    tiled_wsm6_host_buffer_shapes,
)
from .cuda_glibc_pow13 import (
    CUDA_GLIBC_POW13_DEVICE_SOURCE,
    CUDA_GLIBC_POW13_DEVICE_SOURCE_BYTES,
    CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options
from .eqtset2_wsm6_bridge import cpu_ptype5_rebuild_f32


CUDA_GM1_TILESTREAM_POST_WSM6_SCHEMA = "gm1.cuda-tilestream-post-wsm6/v1"
CUDA_GM1_TILESTREAM_POST_WSM6_ABI = (
    "gm1.cuda-tilestream-post-wsm6-eos-qa-f32/v1"
)
CUDA_GM1_TILESTREAM_POST_WSM6_SNAPSHOT_KERNEL = (
    "gm1_tiled_post_wsm6_snapshot_f32_v1"
)
CUDA_GM1_TILESTREAM_POST_WSM6_FINISH_KERNEL = (
    "gm1_tiled_post_wsm6_eos_qa_f32_v1"
)
GM1_ACCEPTED_REAL_WSM6_RECEIPT_SHA256 = (
    "25bc91089d0421dd760b60c10bea5ce6735b92f12aa50f02488602c4995d97e3"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_MP_DRIVER_SHA256 = (
    "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
CM1_TSMALL_F32 = 1.0e-4
CM1_QSMALL_F32 = 1.0e-14
GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT = (
    "tile-local CM1 r21.1 ptype-5 eqtset-2 conditional thermodynamic EOS "
    "rebuild after the accepted public WSM6 boundary and raw physical "
    "q3d-to-qa publication; no solve3 admission, psolver/global pressure, "
    "halo or open-history mutation, MPI exchange, five-metre meteorological "
    "product, or coupled-RK-step claim"
)

_AUX_ORDER = ("temperature_before", "qv_before", "pp3d_before", "qa")
_DRAIN_ORDER = ("p", "pp3d", "th3d", "qa")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def tiled_post_wsm6_aux_device_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact extra device inventory beyond the admitted WSM6 tile arrays."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    mass = (nk, tile_ny, tile_nx)
    return MappingProxyType(
        {
            "temperature_before": mass,
            "qv_before": mass,
            "pp3d_before": mass,
            "qa": (6, *mass),
        }
    )


def tiled_post_wsm6_host_drain_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Durable outputs; the first three reuse existing WSM6 host buffers."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    mass = (nk, tile_ny, tile_nx)
    return MappingProxyType(
        {"p": mass, "pp3d": mass, "th3d": mass, "qa": (6, *mass)}
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledPostWsm6PublicationBinding:
    """Physical canonical-store views mutated by this bounded publication."""

    wsm6: Gm1TiledWsm6SourceBinding
    qa: Any
    wsm6_receipt_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        wsm6: Gm1TiledWsm6SourceBinding,
        qa: Any,
        wsm6_receipt_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledPostWsm6PublicationBinding":
        if not isinstance(wsm6, Gm1TiledWsm6SourceBinding):
            raise TypeError("wsm6 must be Gm1TiledWsm6SourceBinding")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if bool(wsm6.compact_validation) != allow_compact_validation:
            raise ValueError("post-WSM6 and WSM6 compact-validation scopes differ")
        np = importlib.import_module("numpy")
        expected = (6, wsm6.nk, wsm6.ny, wsm6.nx)
        if (
            not isinstance(qa, np.ndarray)
            or qa.dtype != np.dtype("<f4")
            or qa.shape != expected
            or not qa.flags.c_contiguous
            or not qa.flags.writeable
        ):
            raise ValueError(f"qa must be writable host C-<f4 {expected}")
        for name in ("p", "pp3d", "th3d"):
            value = getattr(wsm6, name)
            if not value.flags.writeable:
                raise ValueError(f"post-WSM6 output {name} must be writable")
        mutable = [
            wsm6.q3d,
            wsm6.p,
            wsm6.pp3d,
            wsm6.th3d,
            qa,
            *wsm6.persistence.values(),
        ]
        for index, left in enumerate(mutable):
            if any(np.shares_memory(left, right) for right in mutable[index + 1 :]):
                raise ValueError("post-WSM6 publications must not alias")
        if not bool(np.all(np.isfinite(qa))):
            raise ValueError("initial qa contains non-finite words")
        wsm6_receipt_sha256 = _require_sha256(
            "wsm6_receipt_sha256", wsm6_receipt_sha256
        )
        if not allow_compact_validation and (
            wsm6_receipt_sha256 != GM1_ACCEPTED_REAL_WSM6_RECEIPT_SHA256
        ):
            raise ValueError("WSM6 receipt differs from the accepted real-domain boundary")
        return cls(wsm6, qa, wsm6_receipt_sha256, allow_compact_validation)


def scatter_gm1_tiled_post_wsm6_output(
    binding: Gm1TiledPostWsm6PublicationBinding,
    spec: Gm1TileSpec,
    wsm6_host: Mapping[str, Any],
    qa_host: Any,
) -> Mapping[str, object]:
    """Publish only physical EOS and qa words; execute no halo policy."""

    if not isinstance(binding, Gm1TiledPostWsm6PublicationBinding):
        raise TypeError("binding must be Gm1TiledPostWsm6PublicationBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.wsm6.nx,
        binding.wsm6.ny,
    ):
        raise ValueError("tile spec does not cover the post-WSM6 source domain")
    np = importlib.import_module("numpy")
    shapes = tiled_wsm6_host_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.wsm6.nk
    )
    if not isinstance(wsm6_host, Mapping) or tuple(wsm6_host) != tuple(shapes):
        raise ValueError("post-WSM6 source is not the exact WSM6 host inventory")
    expected_qa = (6, binding.wsm6.nk, spec.tile_ny, spec.tile_nx)
    if (
        not isinstance(qa_host, np.ndarray)
        or qa_host.dtype != np.dtype("<f4")
        or qa_host.shape != expected_qa
        or not qa_host.flags.c_contiguous
    ):
        raise ValueError(f"qa host drain must be C-<f4 {expected_qa}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name in ("p", "pp3d", "th3d"):
        value = wsm6_host[name]
        if value.dtype != np.dtype("<f4") or value.shape != shapes[name]:
            raise ValueError(f"WSM6 drain {name} layout changed")
        np.copyto(
            getattr(binding.wsm6, name)[:, y0:y1, x0:x1],
            value[:, :active_ny, :active_nx],
        )
    np.copyto(
        binding.qa[:, :, y0:y1, x0:x1],
        qa_host[:, :, :active_ny, :active_nx],
    )
    prior = scatter_gm1_tiled_wsm6_output(binding.wsm6, spec, wsm6_host)
    return MappingProxyType(
        {
            "schema": "gm1.hybrid-tilestream-post-wsm6-scatter/v1",
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "published_eos_words": 3
            * binding.wsm6.nk
            * active_ny
            * active_nx,
            "published_qa_words": 6
            * binding.wsm6.nk
            * active_ny
            * active_nx,
            "qa_raw_copy_scope": "physical mass cells only",
            "padding_published": False,
            "halo_or_open_history_mutated": False,
            "prior_wsm6_scatter": dict(prior),
            "claim_limit": GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT,
        }
    )


CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE = CUDA_GLIBC_POW13_DEVICE_SOURCE + r'''// gm1.cuda-tilestream-post-wsm6-eos-qa-f32/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ float cm1_glibc239_pow_positive_normal(float x,float y){
 double logx=cm1_powf_log2_positive_normal(__float_as_uint(x));
 double ylogx=DMU((double)y,logx);
 return __double2float_rn(cm1_powf_exp2(ylogx));
}
extern "C" __global__ void gm1_tiled_post_wsm6_snapshot_f32_v1(
 unsigned long long n,const float*th0,const float*th3d,const float*pi0,
 const float*pp3d,const float*q3d,float*temperature_before,float*qv_before,
 float*pp3d_before){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 temperature_before[p]=M(A(th0[p],th3d[p]),A(pi0[p],pp3d[p]));
 qv_before[p]=q3d[p];pp3d_before[p]=pp3d[p];
}
extern "C" __global__ void gm1_tiled_post_wsm6_eos_qa_f32_v1(
 unsigned long long n,const float*rho,const float*pi0,const float*th0,
 float*p3d,float*pp3d,float*th3d,const float*q3d,const float*thp,float*qa,
 const float*temperature_before,const float*qv_before,const float*pp3d_before){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 float temperature=M(A(th0[p],thp[p]),pp3d[p]);
 float td=fabsf(S(temperature,temperature_before[p]));
 float qd=fabsf(S(q3d[p],qv_before[p]));
 if(td>=1.0e-4f||qd>=1.0e-14f){
  float gas=A(287.04f,M(461.5f,q3d[p]));
  float prs=M(M(rho[p],gas),temperature);p3d[p]=prs;
  float base=M(prs,1.0e-5f);unsigned base_bits=__float_as_uint(base);
  if(base_bits>=0x00800000u&&base_bits<0x7f800000u){
   float perturb=S(cm1_glibc239_pow_positive_normal(base,__uint_as_float(0x3e9221acu)),pi0[p]);pp3d[p]=perturb;
   th3d[p]=S(D(temperature,A(pi0[p],perturb)),th0[p]);
  }else{pp3d[p]=__uint_as_float(0x7fc00000u);th3d[p]=__uint_as_float(0x7fc00000u);}
 }else{pp3d[p]=pp3d_before[p];}
 #pragma unroll
 for(unsigned s=0;s<6u;++s)qa[(unsigned long long)s*n+p]=q3d[(unsigned long long)s*n+p];
}
'''
CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE.encode("utf-8")
)
CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256 = (
    "fad44335c347c559b8f8e5b0b41b54a06115661e1a4303ce8e286d57d09fe1f2"
)


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledPostWsm6:
    module_owner: Any
    snapshot_kernel: Any
    finish_kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_post_wsm6() -> CompiledGm1TiledPostWsm6:
    glibc_source = hashlib.sha256(
        CUDA_GLIBC_POW13_DEVICE_SOURCE_BYTES
    ).hexdigest()
    if glibc_source != CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256:
        raise RuntimeError(
            f"embedded glibc powf CUDA source differs from pin: {glibc_source}"
        )
    actual = hashlib.sha256(CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_BYTES).hexdigest()
    if actual != CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256:
        raise RuntimeError(
            f"packaged tiled post-WSM6 CUDA source differs from pin: {actual}"
        )
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx_options = tuple(
        "--std=c++17" if option.startswith("--std=") else option
        for option in ptx_options
    )
    cubin_options = tuple(
        "--std=c++17" if option.startswith("--std=") else option
        for option in cubin_options
    )
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE,
        "gm1_tiled_post_wsm6_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE,
        "gm1_tiled_post_wsm6_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    for name in (
        CUDA_GM1_TILESTREAM_POST_WSM6_SNAPSHOT_KERNEL,
        CUDA_GM1_TILESTREAM_POST_WSM6_FINISH_KERNEL,
    ):
        if not re.search(rf"\.entry\s+{name}(?:\s|\()", decoded):
            raise ValueError(f"compiled PTX lacks {name}")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1TiledPostWsm6(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_POST_WSM6_SNAPSHOT_KERNEL),
        module.get_function(CUDA_GM1_TILESTREAM_POST_WSM6_FINISH_KERNEL),
        ptx,
        cubin,
        arch,
        runtime,
    )


def _require_device_mapping(
    cupy: Any,
    arrays: Mapping[str, Any],
    aux: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_arrays = tiled_wsm6_device_buffer_shapes(
        tile_nx=nx, tile_ny=ny, nk=nk
    )
    expected_aux = tiled_post_wsm6_aux_device_shapes(
        tile_nx=nx, tile_ny=ny, nk=nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(expected_arrays):
        raise ValueError("post-WSM6 arrays differ from the exact WSM6 inventory")
    if not isinstance(aux, Mapping) or tuple(aux) != tuple(expected_aux):
        raise ValueError("post-WSM6 auxiliary device inventory/order changed")
    checked: dict[str, Any] = {}
    checked_aux: dict[str, Any] = {}
    for output, values, expected, family in (
        (checked, arrays, expected_arrays, "WSM6"),
        (checked_aux, aux, expected_aux, "post-WSM6 auxiliary"),
    ):
        for name, shape in expected.items():
            value = values[name]
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.float32
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"{family} {name} must be C-order CUDA float32 {shape}"
                )
            output[name] = value
    ranges: list[tuple[int, int, str]] = []
    for name, value in (*checked.items(), *checked_aux.items()):
        begin = int(value.data.ptr)
        ranges.append((begin, begin + int(value.nbytes), name))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:], strict=False):
        if left[1] > right[0]:
            raise ValueError(
                f"post-WSM6 device arrays overlap: {left[2]} and {right[2]}"
            )
    return checked, checked_aux


def _pointer_inventory(values: Mapping[str, Any]) -> dict[str, object]:
    return {
        name: {
            "pointer": int(value.data.ptr),
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
        }
        for name, value in values.items()
    }


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPostWsm6Snapshot:
    ready_event: Any
    compiled: CompiledGm1TiledPostWsm6
    arrays: Mapping[str, Any]
    aux: Mapping[str, Any]
    receipt: Mapping[str, Any]


def prepare_cuda_gm1_tiled_post_wsm6_snapshot(
    arrays: Mapping[str, Any],
    aux: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    source_identity_sha256: str,
    wsm6_receipt_sha256: str,
    block_index: int,
    active_nx: int,
    active_ny: int,
    stream: Any,
    compiled: CompiledGm1TiledPostWsm6 | None = None,
) -> CudaGm1TiledPostWsm6Snapshot:
    """Enqueue exact old-state snapshots immediately before tiled WSM6."""

    nx = _positive_int("nx", nx)
    ny = _positive_int("ny", ny)
    nk = _positive_int("nk", nk)
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if active_nx > nx or active_ny > ny:
        raise ValueError("active post-WSM6 rectangle exceeds the fixed block")
    block_index = _nonnegative_int("block_index", block_index)
    source_identity_sha256 = _require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    wsm6_receipt_sha256 = _require_sha256(
        "wsm6_receipt_sha256", wsm6_receipt_sha256
    )
    cupy = importlib.import_module("cupy")
    values, scratch = _require_device_mapping(
        cupy, arrays, aux, nx=nx, ny=ny, nk=nk
    )
    compiled = compile_cuda_gm1_tiled_post_wsm6() if compiled is None else compiled
    n = nx * ny * nk
    threads = 256
    with stream:
        compiled.snapshot_kernel(
            ((n + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                cupy.uint64(n),
                values["th0"],
                values["th3d"],
                values["pi0"],
                values["pp3d"],
                values["q3d"],
                scratch["temperature_before"],
                scratch["qv_before"],
                scratch["pp3d_before"],
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record: dict[str, Any] = {
        "schema": "gm1.cuda-tilestream-post-wsm6-snapshot/v1",
        "abi": CUDA_GM1_TILESTREAM_POST_WSM6_ABI,
        "status": "ENQUEUED_PRE_WSM6_SOURCE_SNAPSHOTS",
        "block_index": block_index,
        "fixed_shape_zyx": [nk, ny, nx],
        "active_shape_yx": [active_ny, active_nx],
        "source_identity_sha256": source_identity_sha256,
        "wsm6_receipt_sha256": wsm6_receipt_sha256,
        "stream_pointer": int(stream.ptr),
        "same_stream_no_host_sync": True,
        "bindings": {
            "source": _pointer_inventory(values),
            "snapshot": _pointer_inventory(scratch),
        },
        "source_order": (
            "immediately before tile-local WSM6; preserve old temperature, qv, pp3d"
        ),
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256,
        "glibc_powf_device_source_sha256": CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
        "nvrtc_language_standard": "c++17",
        "target_arch": compiled.target_arch,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledPostWsm6Snapshot(
        event,
        compiled,
        MappingProxyType(values),
        MappingProxyType(scratch),
        MappingProxyType(record),
    )


def _require_token(record: Mapping[str, Any], *, label: str) -> str:
    detached = dict(record)
    token = _require_sha256(
        f"{label} operation_token_sha256",
        detached.pop("operation_token_sha256", None),
    )
    if hashlib.sha256(_canonical(detached)).hexdigest() != token:
        raise ValueError(f"{label} operation token differs")
    return token


def _require_snapshot(
    snapshot: CudaGm1TiledPostWsm6Snapshot,
    values: Mapping[str, Any],
    scratch: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    active_nx: int,
    active_ny: int,
    block_index: int,
    source_identity_sha256: str,
    wsm6_receipt_sha256: str,
    stream_pointer: int,
) -> str:
    if not isinstance(snapshot, CudaGm1TiledPostWsm6Snapshot):
        raise TypeError("snapshot must be CudaGm1TiledPostWsm6Snapshot")
    receipt = snapshot.receipt
    if (
        receipt.get("schema") != "gm1.cuda-tilestream-post-wsm6-snapshot/v1"
        or receipt.get("abi") != CUDA_GM1_TILESTREAM_POST_WSM6_ABI
        or receipt.get("status") != "ENQUEUED_PRE_WSM6_SOURCE_SNAPSHOTS"
        or receipt.get("fixed_shape_zyx") != [nk, ny, nx]
        or receipt.get("active_shape_yx") != [active_ny, active_nx]
        or receipt.get("block_index") != block_index
        or receipt.get("source_identity_sha256") != source_identity_sha256
        or receipt.get("wsm6_receipt_sha256") != wsm6_receipt_sha256
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_no_host_sync") is not True
    ):
        raise ValueError("pre-WSM6 snapshot identity/order differs")
    if receipt.get("bindings") != {
        "source": _pointer_inventory(values),
        "snapshot": _pointer_inventory(scratch),
    }:
        raise ValueError("pre-WSM6 snapshot device bindings differ")
    if snapshot.ready_event is None:
        raise ValueError("pre-WSM6 snapshot has no CUDA event")
    return _require_token(receipt, label="pre-WSM6 snapshot")


def _require_wsm6_launch(
    launch: CudaGm1TiledWsm6Launch,
    values: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    active_nx: int,
    active_ny: int,
    block_index: int,
    source_identity_sha256: str,
    wsm6_receipt_sha256: str,
    stream_pointer: int,
) -> str:
    if not isinstance(launch, CudaGm1TiledWsm6Launch):
        raise TypeError("wsm6_launch must be CudaGm1TiledWsm6Launch")
    receipt = launch.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_WSM6_SCHEMA
        or receipt.get("abi") != CUDA_GM1_TILESTREAM_WSM6_ABI
        or receipt.get("status") != "ENQUEUED_TILE_LOCAL_PUBLIC_WSM6"
        or receipt.get("fixed_shape_zyx") != [nk, ny, nx]
        or receipt.get("active_shape_yx") != [active_ny, active_nx]
        or receipt.get("block_index") != block_index
        or receipt.get("source_identity_sha256") != source_identity_sha256
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_no_host_sync") is not True
        or (
            wsm6_receipt_sha256 == GM1_ACCEPTED_REAL_WSM6_RECEIPT_SHA256
            and receipt.get("pdefq_receipt_sha256")
            != GM1_ACCEPTED_REAL_PDEFQ_RECEIPT_SHA256
        )
    ):
        raise ValueError("WSM6 launch identity/order differs")
    if any(
        int(launch.arrays[name].data.ptr) != int(values[name].data.ptr)
        for name in values
    ):
        raise ValueError("WSM6 launch did not retain the exact device arrays")
    if launch.ready_event is None:
        raise ValueError("WSM6 launch has no CUDA event")
    return _require_token(receipt, label="WSM6 launch")


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPostWsm6Launch:
    ready_event: Any
    compiled: CompiledGm1TiledPostWsm6
    arrays: Mapping[str, Any]
    aux: Mapping[str, Any]
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_post_wsm6_eos_qa(
    arrays: Mapping[str, Any],
    aux: Mapping[str, Any],
    *,
    snapshot: CudaGm1TiledPostWsm6Snapshot,
    wsm6_launch: CudaGm1TiledWsm6Launch,
    nx: int,
    ny: int,
    nk: int,
    source_identity_sha256: str,
    wsm6_receipt_sha256: str,
    block_index: int,
    active_nx: int,
    active_ny: int,
    stream: Any,
    compiled: CompiledGm1TiledPostWsm6 | None = None,
) -> CudaGm1TiledPostWsm6Launch:
    """Enqueue conditional EOS and physical qa after the exact WSM6 event."""

    nx = _positive_int("nx", nx)
    ny = _positive_int("ny", ny)
    nk = _positive_int("nk", nk)
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if active_nx > nx or active_ny > ny:
        raise ValueError("active post-WSM6 rectangle exceeds the fixed block")
    block_index = _nonnegative_int("block_index", block_index)
    source_identity_sha256 = _require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    wsm6_receipt_sha256 = _require_sha256(
        "wsm6_receipt_sha256", wsm6_receipt_sha256
    )
    cupy = importlib.import_module("cupy")
    values, scratch = _require_device_mapping(
        cupy, arrays, aux, nx=nx, ny=ny, nk=nk
    )
    compiled = snapshot.compiled if compiled is None else compiled
    if compiled.target_arch != snapshot.compiled.target_arch:
        raise ValueError("snapshot and finish target architectures differ")
    stream_pointer = int(stream.ptr)
    snapshot_token = _require_snapshot(
        snapshot,
        values,
        scratch,
        nx=nx,
        ny=ny,
        nk=nk,
        active_nx=active_nx,
        active_ny=active_ny,
        block_index=block_index,
        source_identity_sha256=source_identity_sha256,
        wsm6_receipt_sha256=wsm6_receipt_sha256,
        stream_pointer=stream_pointer,
    )
    wsm6_token = _require_wsm6_launch(
        wsm6_launch,
        values,
        nx=nx,
        ny=ny,
        nk=nk,
        active_nx=active_nx,
        active_ny=active_ny,
        block_index=block_index,
        source_identity_sha256=source_identity_sha256,
        wsm6_receipt_sha256=wsm6_receipt_sha256,
        stream_pointer=stream_pointer,
    )
    n = nx * ny * nk
    threads = 256
    with stream:
        compiled.finish_kernel(
            ((n + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                cupy.uint64(n),
                values["rho"],
                values["pi0"],
                values["th0"],
                values["p"],
                values["pp3d"],
                values["th3d"],
                values["q3d"],
                values["thp"],
                scratch["qa"],
                scratch["temperature_before"],
                scratch["qv_before"],
                scratch["pp3d_before"],
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record: dict[str, Any] = {
        "schema": CUDA_GM1_TILESTREAM_POST_WSM6_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_POST_WSM6_ABI,
        "status": "ENQUEUED_POST_WSM6_EOS_AND_PHYSICAL_QA",
        "block_index": block_index,
        "fixed_shape_zyx": [nk, ny, nx],
        "active_shape_yx": [active_ny, active_nx],
        "source_identity_sha256": source_identity_sha256,
        "accepted_wsm6_receipt_sha256": wsm6_receipt_sha256,
        "snapshot_operation_token_sha256": snapshot_token,
        "wsm6_operation_token_sha256": wsm6_token,
        "stream_pointer": stream_pointer,
        "same_stream_no_host_sync": True,
        "thresholds_f32": {"tsmall": CM1_TSMALL_F32, "qsmall": CM1_QSMALL_F32},
        "source_order": [
            "old temperature/qv/pp3d snapshot before WSM6",
            "accepted tile-local public WSM6 on the same stream",
            "conditional ptype-5 eqtset-2 prs/pp3d/th3d rebuild",
            "raw q3d[0:6] to physical qa[0:6] publication",
        ],
        "mutated": ["p", "pp3d", "th3d", "qa[0:6]"],
        "rho_fixed_across_component_and_eos": True,
        "qa_raw_copy_of_q3d": True,
        "bindings": {
            "wsm6": _pointer_inventory(values),
            "auxiliary": _pointer_inventory(scratch),
        },
        "kernel": {
            "source_sha256": CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256,
            "glibc_powf_device_source_sha256": (
                CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256
            ),
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "nvrtc_language_standard": "c++17",
        },
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "mp_driver_source_sha256": AUTHORITY_MP_DRIVER_SHA256,
            "solve3_source_sha256": AUTHORITY_SOLVE3_SHA256,
            "anchors": [
                "src/mp_driver.F:603-617",
                "src/solve3.F:624-643 physical copy only",
                "src/cm1.F:312-314 tsmall/qsmall selected-real constants",
            ],
        },
        "refusals": {
            "solve3_admission": True,
            "ppi_tha_or_momentum_a_publication": True,
            "psolver_or_global_pressure": True,
            "halo_or_open_history_mutation": True,
            "mpi_exchange": True,
            "five_meter_meteorological_product": True,
            "full_rk_stage": True,
        },
        "claim_limit": GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledPostWsm6Launch(
        event,
        compiled,
        MappingProxyType(values),
        MappingProxyType(scratch),
        MappingProxyType(record),
    )


def cpu_gm1_post_wsm6_snapshot_f32(
    *, th0: Any, th3d: Any, pi0: Any, pp3d: Any, q3d: Any
) -> Mapping[str, Any]:
    """Independent default-real transcription of the pre-WSM6 snapshots."""

    np = importlib.import_module("numpy")
    mass = tuple(th0.shape)
    for name, value, shape in (
        ("th0", th0, mass),
        ("th3d", th3d, mass),
        ("pi0", pi0, mass),
        ("pp3d", pp3d, mass),
        ("q3d", q3d, (6, *mass)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"CPU snapshot {name} must be C-<f4 {shape}")
    theta = np.add(th0, th3d, dtype=np.float32)
    exner = np.add(pi0, pp3d, dtype=np.float32)
    temperature = np.multiply(theta, exner, dtype=np.float32)
    return MappingProxyType(
        {
            "temperature_before": np.ascontiguousarray(temperature),
            "qv_before": np.array(q3d[0], copy=True, order="C"),
            "pp3d_before": np.array(pp3d, copy=True, order="C"),
        }
    )


def cpu_gm1_post_wsm6_eos_qa_f32(
    *,
    rho: Any,
    pi0: Any,
    th0: Any,
    p_before: Any,
    th3d_before: Any,
    q3d_after: Any,
    thp_after: Any,
    arwen_pii: Any,
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Compact native oracle for the conditional EOS and physical qa copy."""

    np = importlib.import_module("numpy")
    mass = tuple(rho.shape)
    values = {
        "rho": rho,
        "pi0": pi0,
        "th0": th0,
        "p_before": p_before,
        "th3d_before": th3d_before,
        "thp_after": thp_after,
        "arwen_pii": arwen_pii,
    }
    for name, value in values.items():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != mass
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"CPU EOS {name} must be C-<f4 {mass}")
    if (
        not isinstance(q3d_after, np.ndarray)
        or q3d_after.dtype != np.dtype("<f4")
        or q3d_after.shape != (6, *mass)
        or not q3d_after.flags.c_contiguous
    ):
        raise ValueError(f"CPU EOS q3d_after must be C-<f4 {(6, *mass)}")
    if not isinstance(snapshot, Mapping) or tuple(snapshot) != (
        "temperature_before",
        "qv_before",
        "pp3d_before",
    ):
        raise ValueError("CPU EOS snapshot inventory/order changed")
    temperature = np.multiply(
        np.add(th0, thp_after, dtype=np.float32),
        arwen_pii,
        dtype=np.float32,
    )
    changed = np.logical_or(
        np.abs(
            np.subtract(temperature, snapshot["temperature_before"], dtype=np.float32)
        )
        >= np.float32(CM1_TSMALL_F32),
        np.abs(
            np.subtract(q3d_after[0], snapshot["qv_before"], dtype=np.float32)
        )
        >= np.float32(CM1_QSMALL_F32),
    )
    pressure_all, pp_all, theta_all = cpu_ptype5_rebuild_f32(
        rho=rho.ravel(order="C"),
        temperature=temperature.ravel(order="C"),
        qv=q3d_after[0].ravel(order="C"),
        pi0=pi0.ravel(order="C"),
        th0=th0.ravel(order="C"),
    )
    pressure_calc = np.asarray(pressure_all, dtype=np.float32).reshape(mass)
    pp_calc = np.asarray(pp_all, dtype=np.float32).reshape(mass)
    theta_calc = np.asarray(theta_all, dtype=np.float32).reshape(mass)
    pressure = np.array(p_before, copy=True, order="C")
    pp3d = np.array(snapshot["pp3d_before"], copy=True, order="C")
    th3d = np.array(th3d_before, copy=True, order="C")
    pressure[changed] = pressure_calc[changed]
    pp3d[changed] = pp_calc[changed]
    th3d[changed] = theta_calc[changed]
    return MappingProxyType(
        {
            "p": pressure,
            "pp3d": pp3d,
            "th3d": th3d,
            "qa": np.array(q3d_after, copy=True, order="C"),
            "temperature_after": np.ascontiguousarray(temperature),
            "changed_mask": np.ascontiguousarray(changed),
            "changed_words": int(np.count_nonzero(changed)),
        }
    )


def five_meter_tiled_post_wsm6_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, Any]:
    """Exact incremental footprint layered on the accepted WSM6 tile seam."""

    aux = tiled_post_wsm6_aux_device_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    mass_bytes = nk * tile_ny * tile_nx * 4
    aux_bytes = sum(math.prod(shape) * 4 for shape in aux.values())
    wsm6 = dict(
        five_meter_tiled_wsm6_inventory(
            tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
        )
    )
    record = {
        "schema": "gm1.hybrid-tilestream-post-wsm6-five-meter-inventory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [nk, tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "incremental_host_qa_bytes_each": 6 * mass_bytes,
        "incremental_device_snapshot_and_qa_bytes_each": aux_bytes,
        "combined_host_buffer_bytes_each": wsm6["host_buffer_bytes_each"]
        + 6 * mass_bytes,
        "combined_double_host_buffers_bytes": wsm6["double_host_buffers_bytes"]
        + 12 * mass_bytes,
        "combined_device_working_set_bytes_each": wsm6[
            "device_working_set_bytes_each"
        ]
        + aux_bytes,
        "combined_double_device_working_sets_bytes": wsm6[
            "double_device_working_sets_bytes"
        ]
        + 2 * aux_bytes,
        "reused_host_drain_carriers": ["p", "pp3d", "th3d"],
        "new_device_carriers": list(aux),
        "full_domain_eos_or_qa_duplicate_bytes": 0,
        "canonical_store_updates": ["p", "pp3d", "th3d", "qa"],
        "pressure_workspace_executed": False,
        "pressure_workspace_incremental_bytes": 0,
        "claim_limit": GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_MP_DRIVER_SHA256",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_QSMALL_F32",
    "CM1_TSMALL_F32",
    "CUDA_GM1_TILESTREAM_POST_WSM6_ABI",
    "CUDA_GM1_TILESTREAM_POST_WSM6_FINISH_KERNEL",
    "CUDA_GM1_TILESTREAM_POST_WSM6_SCHEMA",
    "CUDA_GM1_TILESTREAM_POST_WSM6_SNAPSHOT_KERNEL",
    "CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE",
    "CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256",
    "CompiledGm1TiledPostWsm6",
    "CudaGm1TiledPostWsm6Launch",
    "CudaGm1TiledPostWsm6Snapshot",
    "GM1_ACCEPTED_REAL_WSM6_RECEIPT_SHA256",
    "GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT",
    "Gm1TiledPostWsm6PublicationBinding",
    "compile_cuda_gm1_tiled_post_wsm6",
    "cpu_gm1_post_wsm6_eos_qa_f32",
    "cpu_gm1_post_wsm6_snapshot_f32",
    "five_meter_tiled_post_wsm6_inventory",
    "launch_cuda_gm1_tiled_post_wsm6_eos_qa",
    "prepare_cuda_gm1_tiled_post_wsm6_snapshot",
    "scatter_gm1_tiled_post_wsm6_output",
    "tiled_post_wsm6_aux_device_shapes",
    "tiled_post_wsm6_host_drain_shapes",
)
