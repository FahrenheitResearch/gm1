"""Typed live CM1 solve2 rotation forcing at the divergence/advection seam.

The independently evidenced CPU family owns ``param.F:8945-8969`` and
``solve2.F:631-720``.  This module retains its exact selection and ``f2d``
identity, then enqueues the active default-REAL arithmetic after ``divx`` and
before ``advu/advv/advw`` on the source-owning stream.  The omitted binding is
the existing ``icor=0`` identity and changes no receipt fields.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cuda_momentum_advection5 import _canonical_arch
from .rotation_forcing import (
    CM1_ROTATION_FORCING_STATUS,
    RotationForcingSelection,
    cpu_rotation_forcing_inplace,
)


CUDA_GM1_SOLVE2_ROTATION_BINDING_SCHEMA = "gm1.cuda-solve2-rotation-binding/v1"
CUDA_GM1_SOLVE2_ROTATION_BINDING_STATUS = (
    "PREPARED_LIVE_SOLVE2_ROTATION_AFTER_DIVX"
)
CUDA_GM1_SOLVE2_ROTATION_TILE_SCHEMA = "gm1.cuda-solve2-rotation-enqueue/v1"
CUDA_GM1_SOLVE2_ROTATION_TILE_STATUS = (
    "ENQUEUED_LIVE_SOLVE2_ROTATION_AFTER_DIVX_BEFORE_ADVECTION"
)
CUDA_GM1_SOLVE2_ROTATION_CLAIM_LIMIT = (
    "typed CUDA enqueue boundary for source-exact icor=1 Cartesian "
    "f-plane/beta-plane forcing and the bounded axisymmetric u inner "
    "term after divx and before momentum advection, on any padded "
    "(nk+2, nj+2ng, ni+1+2ng)/(nk+2, nj+1+2ng, ni+2ng) momentum rectangle; "
    "no active CUDA raw-bit "
    "oracle, complete axisymmetric coordinate system, diagnostics, adaptive "
    "combination, complete RK trajectory, production profile, forecast, or product"
)
CUDA_GM1_SOLVE2_ROTATION_SOURCE_ORDER = (
    "terrain rru/rrv/rrw then bcw(rrw,0) then divx",
    "active icor=1 rotation forcing updates the common u/v tendency rectangle",
    "advu then advv then advw accumulate after rotation forcing",
)
ROTATION_CPU_OPERATOR_SOURCE_SHA256 = (
    "a0340efd165a7bc97f565d1f8f76fd4ffe2df537f63777f29e9472f4d96a07c6"
)
CUDA_GM1_SOLVE2_ROTATION_THREADS = 128
CUDA_GM1_SOLVE2_ROTATION_KERNEL_NAME = "gm1_solve2_rotation_f32_v1"
CUDA_GM1_SOLVE2_ROTATION_SOURCE = r"""
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_solve2_rotation_f32_v1(
 const float* __restrict__ u,const float* __restrict__ v,
 float* __restrict__ uten,float* __restrict__ vten,
 const float* __restrict__ f2d,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
 unsigned int beta_plane,unsigned int axisymmetric,
 float fcor,float umove,float vmove)
{
 const unsigned long long total=(unsigned long long)nk*(nj+1u)*(ni+1u);
 const unsigned int unx=ni+1u+2u*ng,uny=nj+2u*ng;
 const unsigned int vnx=ni+2u*ng,vny=nj+1u+2u*ng;
 const unsigned int fnx=ni+2u;
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 for(;q<total;q+=step){
  const unsigned int ci=(unsigned int)(q%(ni+1u));
  const unsigned int cj=(unsigned int)((q/(ni+1u))%(nj+1u));
  const unsigned int k=(unsigned int)(q/((unsigned long long)(ni+1u)*(nj+1u)));
  const unsigned int z=k+1u,x=ci+ng,y=cj+ng;
  if(axisymmetric){
   if(cj<nj){
    const float pair=a32(v[i3(z,y,x,vny,vnx)],v[i3(z,y,x-1u,vny,vnx)]);
    const float coefficient=m32(fcor,0.5f);
    const unsigned long long t=i3(z,y,x,uny,unx);
    uten[t]=a32(uten[t],m32(pair,coefficient));
   }
   continue;
  }
  float first=a32(v[i3(z,y,x,vny,vnx)],v[i3(z,y+1u,x,vny,vnx)]);
  float second=a32(v[i3(z,y,x-1u,vny,vnx)],v[i3(z,y+1u,x-1u,vny,vnx)]);
  first=a32(first,second);
  if(beta_plane==0u){
   first=m32(first,0.25f);first=a32(first,vmove);first=m32(first,fcor);
  }else{
   const float fs=a32(f2d[(unsigned long long)(cj+1u)*fnx+ci+1u],f2d[(unsigned long long)(cj+1u)*fnx+ci]);
   first=m32(first,m32(fs,0.125f));
  }
  const unsigned long long tu=i3(z,y,x,uny,unx);
  uten[tu]=a32(uten[tu],first);
  first=a32(u[i3(z,y,x,uny,unx)],u[i3(z,y,x+1u,uny,unx)]);
  second=a32(u[i3(z,y-1u,x,uny,unx)],u[i3(z,y-1u,x+1u,uny,unx)]);
  first=a32(first,second);
  if(beta_plane==0u){
   first=m32(first,0.25f);first=a32(first,umove);first=m32(first,fcor);
  }else{
   const float fs=a32(f2d[(unsigned long long)(cj+1u)*fnx+ci+1u],f2d[(unsigned long long)cj*fnx+ci+1u]);
   first=m32(first,m32(fs,0.125f));
  }
  const unsigned long long tv=i3(z,y,x,vny,vnx);
  vten[tv]=s32(vten[tv],first);
 }
}
"""
CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_SOLVE2_ROTATION_SOURCE.encode("utf-8")
).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_operator_source() -> None:
    actual = hashlib.sha256(Path(__file__).with_name("rotation_forcing.py").read_bytes()).hexdigest()
    if actual != ROTATION_CPU_OPERATOR_SOURCE_SHA256:
        raise RuntimeError("installed rotation CPU operator differs from its source pin")


def _stage(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    return value


@dataclass(frozen=True, slots=True)
class CompiledGm1Solve2Rotation:
    module_owner: Any
    kernel: Any
    source_sha256: str
    target_arch: str


@dataclass(frozen=True, slots=True)
class CudaGm1Solve2RotationBinding:
    f2d: Any
    ready_event: Any
    stream: Any
    compiled: CompiledGm1Solve2Rotation
    receipt: Mapping[str, object]
    ni: int = 300
    nj: int = 300
    nk: int = 75


def compile_cuda_gm1_solve2_rotation() -> CompiledGm1Solve2Rotation:
    """Compile the pinned rotation kernel for the current CUDA device."""

    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_SOLVE2_ROTATION_SOURCE,
        filename="gm1_solve2_rotation.cu",
        options=(
            "--std=c++14",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ),
        name_expressions=(CUDA_GM1_SOLVE2_ROTATION_KERNEL_NAME,),
    )
    kernel = module.get_function(CUDA_GM1_SOLVE2_ROTATION_KERNEL_NAME)
    arch = _canonical_arch(cupy.cuda.Device().compute_capability)
    return CompiledGm1Solve2Rotation(
        module, kernel, CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256, arch
    )


def _build_f2d_cpu(
    yh_window: Any, selection: RotationForcingSelection, *, ni: int, nj: int
) -> tuple[Any, Mapping[str, object]]:
    """``param.F:8945-8969`` f2d for one domain, through the CPU operator.

    ``f2d`` is a function of ``yh(j)`` alone in both branches CM1 builds --
    a constant on the f plane, ``fcor + beta0*(yh(j)-centery)`` on the beta
    plane -- so the operator is driven once on a one-column window and the
    column is broadcast across ``i``.  That keeps the raw words this kernel
    reads identical to the ones the independently evidenced CPU family
    produces, at any ``ni``.
    """

    _require_operator_source()
    np = importlib.import_module("numpy")
    if (
        not isinstance(yh_window, np.ndarray)
        or yh_window.dtype != np.dtype(np.float32)
        or yh_window.shape != (nj + 2,)
        or not yh_window.flags.c_contiguous
    ):
        raise ValueError(
            f"yh_window must be C-order native float32 shape ({nj + 2},)"
        )
    arrays = {
        "u3d_window": np.zeros((1, nj + 2, 3), dtype=np.float32),
        "v3d_window": np.zeros((1, nj + 2, 3), dtype=np.float32),
        "uten": np.zeros((1, nj + 1, 2), dtype=np.float32),
        "vten": np.zeros((1, nj + 1, 2), dtype=np.float32),
        "yh_window": yh_window,
        "f2d": np.empty((nj + 2, 3), dtype=np.float32),
    }
    result = cpu_rotation_forcing_inplace(
        arrays, ni=1, nj=nj, nk=1, selection=selection
    )
    if result.receipt.get("status") != CM1_ROTATION_FORCING_STATUS:
        raise ValueError("source-exact rotation operator receipt status changed")
    f2d = np.ascontiguousarray(
        np.broadcast_to(result.f2d[:, :1], (nj + 2, ni + 2)), dtype=np.float32
    )
    return f2d, result.receipt


def prepare_cuda_gm1_solve2_rotation(
    yh_window: Any,
    *,
    ni: int = 300,
    nj: int = 300,
    nk: int = 75,
    selection: RotationForcingSelection,
    nrk: int,
    stream: Any,
    compiled: CompiledGm1Solve2Rotation | None = None,
    allow_bounded_axisymmetric_u_validation: bool = False,
    adaptive_dt: bool = False,
    adaptive_move: bool = False,
    budget_diagnostics: bool = False,
) -> CudaGm1Solve2RotationBinding:
    """Prepare exact selection/f2d identity on one live RK stream."""

    nrk = _stage(nrk)
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(selection) is not RotationForcingSelection:
        raise TypeError("selection must be the exact rotation selection type")
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) <= 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    for name, value in (
        ("allow_bounded_axisymmetric_u_validation", allow_bounded_axisymmetric_u_validation),
        ("adaptive_dt", adaptive_dt),
        ("adaptive_move", adaptive_move),
        ("budget_diagnostics", budget_diagnostics),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be boolean")
    if adaptive_move:
        # Concrete defect: solve2.F:645,647 read umove/vmove, and this binding
        # bakes them into kernel_scalars once at prepare time.  A controller
        # that moved the frame mid-run would leave the Coriolis add-back on
        # the frame velocity the run started with, so the term would be wrong
        # by fcor times the frame's change for the rest of the integration.
        raise ValueError(
            "active rotation binding does not admit an adaptive moving frame: "
            "umove/vmove are sealed into this binding, so a frame the "
            "controller changed would keep the solve2.F:645,647 add-back of "
            "the frame velocity the run started with"
        )
    # adaptive_dt used to be refused here beside adaptive_move.  It names no
    # defect: solve2.F:633-673 reads no timestep at all -- the term is a
    # tendency increment, and the stage that integrates it applies whatever dt
    # it was handed.  Retired with the fix, as the guard rule requires.
    if budget_diagnostics:
        raise ValueError("rotation budget diagnostics are outside the live binding")
    if selection.axisymmetric == 1 and not allow_bounded_axisymmetric_u_validation:
        raise ValueError(
            "axisymmetric rotation requires explicit bounded u-inner-term validation"
        )
    f2d, cpu_receipt = _build_f2d_cpu(yh_window, selection, ni=ni, nj=nj)
    np = importlib.import_module("numpy")
    kernel_scalars = {
        name.removesuffix("_bits"): float(
            np.asarray(bits, dtype=np.uint32).view(np.float32)
        )
        for name, bits in cpu_receipt["selection"].items()
        if name in {"fcor_f32_bits", "u_move_f32_bits", "v_move_f32_bits"}
    }
    compiled = compiled or compile_cuda_gm1_solve2_rotation()
    if type(compiled) is not CompiledGm1Solve2Rotation:
        raise TypeError("compiled must be the exact rotation CUDA image type")
    if compiled.source_sha256 != CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256:
        raise ValueError("compiled rotation CUDA source identity changed")
    cupy = importlib.import_module("cupy")
    with stream:
        device_f2d = cupy.asarray(f2d)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE2_ROTATION_BINDING_SCHEMA,
        "status": CUDA_GM1_SOLVE2_ROTATION_BINDING_STATUS,
        "nrk": nrk,
        "stream_pointer": int(stream.ptr),
        "domain": {"nx": ni, "ny": nj, "nk": nk},
        "selection": dict(cpu_receipt["selection"]),
        "kernel_scalars": kernel_scalars,
        "bounded_axisymmetric_u_inner_term": selection.axisymmetric == 1,
        "outer_axisymmetric_coordinate_admitted": False,
        "adaptive_dt": bool(adaptive_dt),
        "adaptive_move": False,
        "budget_diagnostics": False,
        "yh_window_sha256": _array_sha256(yh_window),
        "f2d_sha256": _array_sha256(f2d),
        "cpu_operator_operation_token_sha256": cpu_receipt["operation_token_sha256"],
        "cpu_operator_source_sha256": ROTATION_CPU_OPERATOR_SOURCE_SHA256,
        "cuda_source_sha256": CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256,
        "source_order": list(CUDA_GM1_SOLVE2_ROTATION_SOURCE_ORDER),
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE2_ROTATION_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Solve2RotationBinding(
        device_f2d, event, stream, compiled, MappingProxyType(receipt), ni, nj, nk
    )


def require_cuda_gm1_solve2_rotation_binding(
    binding: CudaGm1Solve2RotationBinding,
    *,
    nrk: int,
    stream: Any,
) -> CudaGm1Solve2RotationBinding:
    nrk = _stage(nrk)
    if type(binding) is not CudaGm1Solve2RotationBinding:
        raise TypeError("binding must be the exact live solve2 rotation binding")
    receipt = binding.receipt
    if (
        stream is not binding.stream
        or int(stream.ptr) != receipt.get("stream_pointer")
        or receipt.get("schema") != CUDA_GM1_SOLVE2_ROTATION_BINDING_SCHEMA
        or receipt.get("status") != CUDA_GM1_SOLVE2_ROTATION_BINDING_STATUS
        or receipt.get("nrk") != nrk
        or receipt.get("outer_axisymmetric_coordinate_admitted") is not False
        or not isinstance(receipt.get("adaptive_dt"), bool)
        or receipt.get("adaptive_move") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("rotation binding identity, stage, or stream changed")
    if (
        type(binding.compiled) is not CompiledGm1Solve2Rotation
        or binding.compiled.source_sha256
        != CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
    ):
        raise ValueError("rotation binding lost its exact compiled source identity")
    cupy = importlib.import_module("cupy")
    expected_f2d = (binding.nj + 2, binding.ni + 2)
    if (
        not isinstance(binding.f2d, cupy.ndarray)
        or binding.f2d.dtype != cupy.float32
        or tuple(binding.f2d.shape) != expected_f2d
        or not binding.f2d.flags.c_contiguous
    ):
        raise ValueError(
            f"rotation f2d must remain device C-f32 shape {expected_f2d}"
        )
    return binding


def enqueue_cuda_gm1_solve2_rotation_after_divergence(
    binding: CudaGm1Solve2RotationBinding,
    arrays: Mapping[str, Any],
    *,
    stream: Any,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    ng: int,
) -> Mapping[str, object]:
    """Enqueue exact active rotation between divergence and momentum advection."""

    binding = require_cuda_gm1_solve2_rotation_binding(
        binding, nrk=nrk, stream=stream
    )
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk), ("ng", ng)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (ni, nj, nk) != (binding.ni, binding.nj, binding.nk):
        raise ValueError(
            "rotation enqueue geometry "
            f"{(ni, nj, nk)} is not the ({binding.ni}, {binding.nj}, "
            f"{binding.nk}) this binding built f2d for; the beta-plane "
            "coefficient column would be read for the wrong rows"
        )
    if ng < 1:
        raise ValueError("rotation enqueue requires a halo of at least one cell")
    cupy = importlib.import_module("cupy")
    expected = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
    }
    for name, shape in expected.items():
        value = arrays.get(name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"rotation {name} must be device C-f32 {shape}")
    selection = binding.receipt["selection"]
    scalars = binding.receipt["kernel_scalars"]
    count = nk * (nj + 1) * (ni + 1)
    block = (CUDA_GM1_SOLVE2_ROTATION_THREADS, 1, 1)
    grid = ((count + block[0] - 1) // block[0], 1, 1)
    with stream:
        stream.wait_event(binding.ready_event)
        binding.compiled.kernel(
            grid,
            block,
            (
                arrays["u3d"],
                arrays["v3d"],
                arrays["uten"],
                arrays["vten"],
                binding.f2d,
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.uint32(ng),
                cupy.uint32(selection["beta_plane"]),
                cupy.uint32(selection["axisymmetric"]),
                cupy.float32(scalars["fcor_f32"]),
                cupy.float32(scalars["u_move_f32"]),
                cupy.float32(scalars["v_move_f32"]),
            ),
            shared_mem=0,
            stream=stream,
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE2_ROTATION_TILE_SCHEMA,
        "status": CUDA_GM1_SOLVE2_ROTATION_TILE_STATUS,
        "nrk": nrk,
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": binding.receipt["operation_token_sha256"],
        "selection": dict(selection),
        "source_order": list(CUDA_GM1_SOLVE2_ROTATION_SOURCE_ORDER),
        "same_stream_after_divx_before_advection": True,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE2_ROTATION_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


def make_cuda_gm1_solve2_rotation_post_divergence_hook(
    binding: CudaGm1Solve2RotationBinding,
) -> Callable[..., Mapping[str, object]]:
    """Return the exact hook accepted by stretched momentum."""

    def hook(arrays: Mapping[str, Any], **kwargs: Any) -> Mapping[str, object]:
        return enqueue_cuda_gm1_solve2_rotation_after_divergence(
            binding, arrays, **kwargs
        )

    return hook


__all__ = [
    "CUDA_GM1_SOLVE2_ROTATION_BINDING_SCHEMA",
    "CUDA_GM1_SOLVE2_ROTATION_BINDING_STATUS",
    "CUDA_GM1_SOLVE2_ROTATION_CLAIM_LIMIT",
    "CUDA_GM1_SOLVE2_ROTATION_KERNEL_NAME",
    "CUDA_GM1_SOLVE2_ROTATION_SOURCE",
    "CUDA_GM1_SOLVE2_ROTATION_SOURCE_ORDER",
    "CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256",
    "CUDA_GM1_SOLVE2_ROTATION_TILE_SCHEMA",
    "CUDA_GM1_SOLVE2_ROTATION_TILE_STATUS",
    "CompiledGm1Solve2Rotation",
    "CudaGm1Solve2RotationBinding",
    "ROTATION_CPU_OPERATOR_SOURCE_SHA256",
    "compile_cuda_gm1_solve2_rotation",
    "enqueue_cuda_gm1_solve2_rotation_after_divergence",
    "make_cuda_gm1_solve2_rotation_post_divergence_hook",
    "prepare_cuda_gm1_solve2_rotation",
    "require_cuda_gm1_solve2_rotation_binding",
]
