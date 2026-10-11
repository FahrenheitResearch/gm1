"""Typed live binding for CM1's ``ramp_sgs=1`` coefficient boundary.

CM1 applies the startup ramp at the beginning of ``sfc_and_turb``, before
the SGS closure consumes ``cme``/``csm``.  The independently admitted CPU
operator owns the default-REAL coefficient arithmetic.  This module narrows
that family to the selected GM1 ``cm1setup=1, sgsmodel=1`` profile, retains
the exact CPU token and coefficient bits, and exposes one same-stream CUDA
refresh of the fresh closure outputs before ``tke3d``/``tketen`` are copied.

Omitting the binding is the existing ``ramp_sgs=0`` identity and changes no
receipt fields.  Restored/retained TKE, simultaneous optional solve forcing,
and active CUDA raw-bit or coupled-trajectory claims remain fail closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .cuda_momentum_advection5 import _canonical_arch
from .nvrtc_module import compile_nvrtc_module
from .sgs_startup_ramp import (
    CM1_SGS_STARTUP_RAMP_STATUS,
    SgsStartupRampSelection,
    cpu_sgs_startup_ramp_inplace,
)


CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_SCHEMA = (
    "gm1.cuda-tilestream-sgs-startup-ramp-binding/v1"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_STATUS = (
    "PREPARED_LIVE_FRESH_CLOSURE_SGS_STARTUP_RAMP"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA = (
    "gm1.cuda-tilestream-sgs-startup-ramp-tile/v1"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS = (
    "ENQUEUED_LIVE_FRESH_CLOSURE_SGS_STARTUP_RAMP"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_CLAIM_LIMIT = (
    "typed source-order delivery of independently source-exact CPU "
    "ramp_sgs=1 coefficients into the selected fresh cm1setup=1/sgsmodel=1 "
    "300x300x75 GM1 closure on one explicit CUDA stream; no retained TKE, "
    "simultaneous optional solve forcing, active CUDA raw-bit oracle, complete "
    "RK trajectory, production configuration, forecast, or product"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_ORDER = (
    "source-normalized rtime/ramp_time and cm1setup=1 cm0=c_m selection",
    "ramp_sgs=1 updates cme then ce1/ce2/csm before turbulence dispatch",
    "fresh closure km/kh/dissten/tkea is refreshed with the ramped coefficients",
    "rigid faces and exact tke3d/tketen copy follow the refresh",
)
SGS_STARTUP_RAMP_CPU_OPERATOR_SOURCE_SHA256 = (
    "0a4ad122def32b6cef427d5c3455a6deecca42d5e4d17e420d70fbe8eb1b9f72"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_THREADS = 128
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_KERNEL_NAME = (
    "gm1_fresh_closure_sgs_startup_ramp_f32_v1"
)
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE = r"""
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void gm1_fresh_closure_sgs_startup_ramp_f32_v1(
 const float* __restrict__ nm,const float* __restrict__ defv,
 const float* __restrict__ defh,const float* __restrict__ grdscl,
 float* __restrict__ dissten,float* __restrict__ kmh,float* __restrict__ kmv,
 float* __restrict__ khh,float* __restrict__ khv,float* __restrict__ tkea,
 unsigned int ni,unsigned int nj,unsigned int nk,float cme,float csm)
{
 const unsigned long long nxy=(unsigned long long)ni*nj;
 const unsigned long long words=nxy*(nk-1u);
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 for(;q<words;q+=step){
  const unsigned int k=(unsigned int)(q/nxy)+1u;
  const unsigned long long p=(unsigned long long)k*nxy+(q%nxy);
  const float grid=grdscl[p];
  float drive=a32(a32(defv[p],defh[p]),m32(nm[p],3.0f));
  drive=fmaxf(drive,1.0e-10f);
  const float scale=m32(csm,grid);
  const float scale2=m32(scale,scale);
  const float km=m32(scale2,sqrtf(drive));
  const float kh=m32(km,3.0f);
  const float numerator=m32(m32(km,km),km);
  const float denominator=m32(scale2,scale2);
  const float diss=d32(numerator,denominator);
  const float ratio=d32(km,m32(cme,grid));
  dissten[p]=diss;kmh[p]=km;kmv[p]=km;khh[p]=kh;khv[p]=kh;
  tkea[p]=m32(ratio,ratio);
 }
}
"""
CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE.encode("utf-8")
).hexdigest()
_COEFFICIENT_NAMES = ("cme", "ce1", "ce2", "csm")
_WORKSPACE_NAMES = (
    "nm",
    "defv",
    "defh",
    "grdscl",
    "dissten",
    "kmh",
    "kmv",
    "khh",
    "khv",
    "tkea",
)


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


def _require_operator_source() -> None:
    actual = hashlib.sha256(
        Path(__file__).with_name("sgs_startup_ramp.py").read_bytes()
    ).hexdigest()
    if actual != SGS_STARTUP_RAMP_CPU_OPERATOR_SOURCE_SHA256:
        raise RuntimeError("installed SGS startup-ramp operator differs from its pin")


def _f32_bits(np: Any, value: object) -> int:
    return int(np.asarray(np.float32(value), dtype=np.float32).view(np.uint32))


def _validate_geometry(
    *,
    nx: int,
    ny: int,
    nk: int,
    dx: float,
    dy: float,
    completed_nrk: int,
    allow_compact_validation: bool,
) -> None:
    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    if any(type(value) is not int or value < 1 for value in (nx, ny, nk)):
        raise ValueError("nx, ny, and nk must be positive integers")
    if completed_nrk != 0:
        raise ValueError("SGS startup ramp is admitted only before RK stage 1")
    for name, value in (("dx", dx), ("dy", dy)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not allow_compact_validation and (
        (nx, ny, nk) != (300, 300, 75) or (float(dx), float(dy)) != (50.0, 50.0)
    ):
        raise ValueError(
            "live SGS startup-ramp binding requires exact 300x300x75/50 m geometry"
        )


def _validate_live_selection(selection: SgsStartupRampSelection) -> None:
    if type(selection) is not SgsStartupRampSelection:
        raise TypeError("selection must be the exact SGS startup-ramp selection type")
    if (
        selection.ramp_sgs != 1
        or selection.idoles is not True
        or selection.sgsmodel != 1
    ):
        raise ValueError(
            "live startup-ramp binding requires ramp_sgs=1/idoles/sgsmodel=1"
        )


@dataclass(frozen=True, slots=True)
class Gm1SgsStartupRampCpuCoefficients:
    """Exact scalar coefficients for the selected uniform ``cm0=c_m`` profile."""

    cme: float
    ce1: float
    ce2: float
    csm: float
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CompiledCudaGm1TiledSgsStartupRamp:
    module_owner: Any
    kernel: Any
    source_sha256: str
    target_arch: str


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSgsStartupRampBinding:
    coefficients: Mapping[str, float]
    ready_event: Any
    stream: Any
    compiled: CompiledCudaGm1TiledSgsStartupRamp
    receipt: Mapping[str, object]


def build_gm1_sgs_startup_ramp_cpu_coefficients(
    *,
    selection: SgsStartupRampSelection,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    dx: float = 50.0,
    dy: float = 50.0,
    completed_nrk: int = 0,
    cm1setup: int = 1,
    tconfig: int = 1,
    cm0: float = 0.10,
    mpi: bool = False,
    adaptive_dt: bool = False,
    simultaneous_lspgrad: int = 0,
    simultaneous_rterm: int = 0,
    simultaneous_rayleigh_variant: bool = False,
    allow_compact_validation: bool = False,
) -> Gm1SgsStartupRampCpuCoefficients:
    """Run the exact CPU operator on the uniform selected coefficient column."""

    _validate_geometry(
        nx=nx,
        ny=ny,
        nk=nk,
        dx=dx,
        dy=dy,
        completed_nrk=completed_nrk,
        allow_compact_validation=allow_compact_validation,
    )
    _validate_live_selection(selection)
    if (cm1setup, tconfig) != (1, 1):
        raise ValueError("live startup ramp requires cm1setup=1/tconfig=1")
    if type(mpi) is not bool or type(adaptive_dt) is not bool:
        raise TypeError("mpi and adaptive_dt must be boolean")
    if mpi or adaptive_dt:
        raise ValueError("live startup ramp requires serial fixed-dt execution")
    if (
        simultaneous_lspgrad != 0
        or simultaneous_rterm != 0
        or simultaneous_rayleigh_variant is not False
    ):
        raise ValueError(
            "live startup ramp has no admitted simultaneous optional solve forcing"
        )
    if isinstance(cm0, bool) or not isinstance(cm0, (int, float)):
        raise TypeError("cm0 must be numeric")
    if not math.isfinite(float(cm0)) or float(cm0) != 0.10:
        raise ValueError("cm1setup=1 startup ramp requires exact cm0=c_m=0.10")

    _require_operator_source()
    np = importlib.import_module("numpy")
    arrays: dict[str, Any] = {
        "cm0": np.full((1, 1), np.float32(cm0), dtype=np.float32),
        **{
            name: np.zeros((2, 1, 1), dtype=np.float32)
            for name in _COEFFICIENT_NAMES
        },
    }
    result = cpu_sgs_startup_ramp_inplace(
        arrays, ni=1, nj=1, nk=1, selection=selection
    )
    if result.receipt.get("status") != CM1_SGS_STARTUP_RAMP_STATUS:
        raise ValueError("source-exact startup-ramp operator status changed")
    coefficient_bits: dict[str, int] = {}
    coefficient_values: dict[str, float] = {}
    for name in _COEFFICIENT_NAMES:
        words = arrays[name].reshape(-1).view(np.uint32)
        if not np.all(words == words[0]):
            raise ValueError("uniform cm0 produced nonuniform startup-ramp coefficients")
        coefficient_bits[name] = int(words[0])
        coefficient_values[name] = float(arrays[name].reshape(-1)[0])

    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_GM1_SGS_STARTUP_RAMP_COEFFICIENTS",
        "completed_nrk": 0,
        "domain": {
            "nx": nx,
            "ny": ny,
            "nk": nk,
            "dx_m": float(dx),
            "dy_m": float(dy),
        },
        "selection": dict(result.receipt["selection"]),
        "selected_profile": {
            "cm1setup": 1,
            "sgsmodel": 1,
            "tconfig": 1,
            "cm0_raw_bits": _f32_bits(np, cm0),
            "uniform_cm0": True,
        },
        "companion_selection": {
            "mpi": False,
            "adaptive_dt": False,
            "lspgrad": 0,
            "rterm": 0,
            "rayleigh_variant": False,
        },
        "coefficient_raw_bits": coefficient_bits,
        "cpu_operator_operation_token_sha256": result.receipt[
            "operation_token_sha256"
        ],
        "cpu_operator_source_sha256": SGS_STARTUP_RAMP_CPU_OPERATOR_SOURCE_SHA256,
        "source_order": list(CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_ORDER),
        "retained_tke_admitted": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "allow_compact_validation": allow_compact_validation,
        "claim_limit": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1SgsStartupRampCpuCoefficients(
        **coefficient_values, receipt=MappingProxyType(receipt)
    )


def compile_cuda_gm1_tilestream_sgs_startup_ramp(
) -> CompiledCudaGm1TiledSgsStartupRamp:
    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE,
        filename="gm1_tilestream_sgs_startup_ramp.cu",
        options=(
            "--std=c++14",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ),
        name_expressions=(CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_KERNEL_NAME,),
    )
    kernel = module.get_function(CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_KERNEL_NAME)
    return CompiledCudaGm1TiledSgsStartupRamp(
        module,
        kernel,
        CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256,
        _canonical_arch(cupy.cuda.Device().compute_capability),
    )


def prepare_cuda_gm1_tilestream_sgs_startup_ramp(
    *,
    selection: SgsStartupRampSelection,
    stream: Any,
    compiled: CompiledCudaGm1TiledSgsStartupRamp | None = None,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    dx: float = 50.0,
    dy: float = 50.0,
    completed_nrk: int = 0,
    allow_compact_validation: bool = False,
    **selection_guards: object,
) -> CudaGm1TiledSgsStartupRampBinding:
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) <= 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    cpu = build_gm1_sgs_startup_ramp_cpu_coefficients(
        selection=selection,
        nx=nx,
        ny=ny,
        nk=nk,
        dx=dx,
        dy=dy,
        completed_nrk=completed_nrk,
        allow_compact_validation=allow_compact_validation,
        **selection_guards,
    )
    compiled = compiled or compile_cuda_gm1_tilestream_sgs_startup_ramp()
    if type(compiled) is not CompiledCudaGm1TiledSgsStartupRamp:
        raise TypeError("compiled must be the exact startup-ramp CUDA image type")
    if compiled.source_sha256 != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256:
        raise ValueError("compiled startup-ramp CUDA source identity changed")
    cupy = importlib.import_module("cupy")
    with stream:
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    coefficients = MappingProxyType(
        {name: float(getattr(cpu, name)) for name in _COEFFICIENT_NAMES}
    )
    receipt = dict(cpu.receipt)
    receipt.update(
        {
            "status": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_STATUS,
            "stream_pointer": int(stream.ptr),
            "cuda_source_sha256": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256,
            "cuda_target_arch": compiled.target_arch,
            "same_stream_prepared": True,
        }
    )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledSgsStartupRampBinding(
        coefficients, event, stream, compiled, MappingProxyType(receipt)
    )


def require_cuda_gm1_tilestream_sgs_startup_ramp_binding(
    binding: CudaGm1TiledSgsStartupRampBinding,
    *,
    spec: Gm1TileSpec,
    nk: int,
    dx: float,
    dy: float,
    stream: Any,
) -> CudaGm1TiledSgsStartupRampBinding:
    if type(binding) is not CudaGm1TiledSgsStartupRampBinding:
        raise TypeError("binding must be the exact live SGS startup-ramp binding")
    receipt = binding.receipt
    domain = receipt.get("domain")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if domain != {
        "nx": spec.nx,
        "ny": spec.ny,
        "nk": nk,
        "dx_m": float(dx),
        "dy_m": float(dy),
    }:
        raise ValueError("startup-ramp binding geometry differs from the closure tile")
    if (
        stream is not binding.stream
        or int(stream.ptr) != receipt.get("stream_pointer")
        or receipt.get("schema")
        != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_STATUS
        or receipt.get("completed_nrk") != 0
        or receipt.get("selected_profile")
        != {
            "cm1setup": 1,
            "sgsmodel": 1,
            "tconfig": 1,
            "cm0_raw_bits": 1036831949,
            "uniform_cm0": True,
        }
        or receipt.get("retained_tke_admitted") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("startup-ramp binding identity, cadence, or stream changed")
    if (
        type(binding.compiled) is not CompiledCudaGm1TiledSgsStartupRamp
        or binding.compiled.source_sha256
        != CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256
    ):
        raise ValueError("startup-ramp binding lost its compiled source identity")
    np = importlib.import_module("numpy")
    coefficient_raw_bits = receipt.get("coefficient_raw_bits")
    if (
        tuple(binding.coefficients) != _COEFFICIENT_NAMES
        or not isinstance(coefficient_raw_bits, Mapping)
        or tuple(coefficient_raw_bits) != _COEFFICIENT_NAMES
        or {
            name: _f32_bits(np, binding.coefficients[name])
            for name in _COEFFICIENT_NAMES
        }
        != dict(coefficient_raw_bits)
    ):
        raise ValueError("startup-ramp coefficient inventory or raw bits changed")
    return binding


def enqueue_cuda_gm1_tilestream_sgs_startup_ramp(
    binding: CudaGm1TiledSgsStartupRampBinding,
    workspace: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    nk: int,
    dx: float,
    dy: float,
    stream: Any,
) -> Mapping[str, object]:
    """Refresh one fresh closure tile with exact ramped scalar coefficients."""

    binding = require_cuda_gm1_tilestream_sgs_startup_ramp_binding(
        binding, spec=spec, nk=nk, dx=dx, dy=dy, stream=stream
    )
    if not isinstance(workspace, Mapping):
        raise TypeError("workspace must be a mapping")
    cupy = importlib.import_module("cupy")
    mass = (nk, spec.cny, spec.cnx)
    face = (nk + 1, spec.cny, spec.cnx)
    for name in _WORKSPACE_NAMES:
        value = workspace.get(name)
        shape = mass if name == "grdscl" else face
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"startup-ramp {name} must be device C-f32 {shape}")
    count = spec.cnx * spec.cny * (nk - 1)
    block = (CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_THREADS, 1, 1)
    grid = ((count + block[0] - 1) // block[0], 1, 1)
    with stream:
        stream.wait_event(binding.ready_event)
        binding.compiled.kernel(
            grid,
            block,
            (
                *(workspace[name] for name in _WORKSPACE_NAMES),
                cupy.uint32(spec.cnx),
                cupy.uint32(spec.cny),
                cupy.uint32(nk),
                cupy.float32(binding.coefficients["cme"]),
                cupy.float32(binding.coefficients["csm"]),
            ),
            shared_mem=0,
            stream=stream,
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": binding.receipt[
            "operation_token_sha256"
        ],
        "cpu_operator_operation_token_sha256": binding.receipt[
            "cpu_operator_operation_token_sha256"
        ],
        "selection": dict(binding.receipt["selection"]),
        "selected_profile": dict(binding.receipt["selected_profile"]),
        "coefficient_raw_bits": dict(binding.receipt["coefficient_raw_bits"]),
        "source_order": list(CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_ORDER),
        "same_stream_before_fresh_tke_copy": True,
        "retained_tke_admitted": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_MODULE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = [
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_SCHEMA",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_BINDING_STATUS",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_CLAIM_LIMIT",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_KERNEL_NAME",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_MODULE_SHA256",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_ORDER",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_SOURCE_SHA256",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_SCHEMA",
    "CUDA_GM1_TILESTREAM_SGS_STARTUP_RAMP_TILE_STATUS",
    "CompiledCudaGm1TiledSgsStartupRamp",
    "CudaGm1TiledSgsStartupRampBinding",
    "Gm1SgsStartupRampCpuCoefficients",
    "SGS_STARTUP_RAMP_CPU_OPERATOR_SOURCE_SHA256",
    "build_gm1_sgs_startup_ramp_cpu_coefficients",
    "compile_cuda_gm1_tilestream_sgs_startup_ramp",
    "enqueue_cuda_gm1_tilestream_sgs_startup_ramp",
    "prepare_cuda_gm1_tilestream_sgs_startup_ramp",
    "require_cuda_gm1_tilestream_sgs_startup_ramp_binding",
]
