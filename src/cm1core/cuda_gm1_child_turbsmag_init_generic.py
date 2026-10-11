"""Arbitrary-positive-normal CM1 child Smagorinsky/TKE initialization.

This is the general-grid successor to the selected 77-word compatibility
table in :mod:`cm1core.cuda_gm1_child_turbsmag_init`.  It preserves the exact
CM1 r21.1 source ordering and fresh-child TKE disposition, but evaluates the
grid-volume power with the separately sealed glibc 2.39 CUDA transcription.
Before any output is produced, a device-side gate checks every consumed metric
word and every resulting grid-volume operand; unsupported arithmetic domains
fail closed.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_glibc_pow13 import (
    CUDA_GLIBC_POW13_DEVICE_SOURCE,
    CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
    DEFAULT_GLIBC_POW13_SELECTION,
    GlibcPow13Selection,
    require_glibc_pow13_selection,
)
from cm1core.cuda_gm1_child_turbsmag_init import (
    AUTHORITY_SOURCE_SHA256,
    AUTHORITY_TARBALL_SHA256,
    DEFAULT_CHILD_TURBSMAG_INIT_SELECTION,
    INPUT_ARRAY_NAMES,
    OUTPUT_ARRAY_NAMES,
    SOURCE_ANCHORS,
    ChildTurbsmagInitSelection,
    CudaChildTurbsmagInitLaunch,
    child_turbsmag_init_shapes,
    require_child_turbsmag_init_selection,
)
from cm1core.errors import refuse


CUDA_GM1_CHILD_TURBSMAG_GENERIC_SCHEMA = "gm1.cuda-child-turbsmag-init/v2"
CUDA_GM1_CHILD_TURBSMAG_GENERIC_ABI = (
    "gm1.cuda-child-turbsmag-init-arbitrary-positive-normal-f32/v2"
)
CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS = 256
CUDA_GM1_CHILD_TURBSMAG_GENERIC_KERNEL_NAMES = (
    "cm1_child_turbsmag_metric_validate_v2",
    "cm1_child_turbsmag_interior_v2",
    "cm1_child_turbsmag_faces_copy_v2",
)
CUDA_GM1_CHILD_TURBSMAG_GENERIC_CLAIM_LIMIT = (
    "CM1 r21.1 serial cm1setup=1/sgsmodel=1/tconfig=1 interpolated-restart "
    "local turbsmag coefficients, diagnosed TKE, rigid bottom/top disposition, "
    "and tke3d copy for arbitrary positive-normal CM1 grid-volume operands; "
    "glibc powf 2.39 and exponent 0x3eaaaaab are fixed, calcnm/calcdef remain "
    "required admitted inputs, and no other libm/exponent, subnormal/invalid "
    "metric, halos/MPI, atmospheric remap, time evolution, forecast, storm, or "
    "tornado claim is made; raw-bit CUDA evidence is architecture-scoped and "
    "must be replayed before claiming a new GPU architecture"
)


_GENERIC_KERNEL_SOURCE = r'''// gm1.cuda-child-turbsmag-init/v2
static_assert(sizeof(float)==4,"CM1 default REAL requires binary32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ bool PN(float x){unsigned b=__float_as_uint(x);return b>=0x00800000u&&b<0x7f800000u;}
__device__ __forceinline__ bool FI(float x){return (__float_as_uint(x)&0x7f800000u)!=0x7f800000u;}

extern "C" __global__ void cm1_child_turbsmag_metric_validate_v2(
 const float*ruh,const float*rvh,const float*rmf,const float*nm,const float*defv,const float*defh,
 unsigned*invalid,unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nxy=(unsigned long long)ni*nj,words=nxy*(nk-1u);unsigned local=0u;
 for(unsigned long long linear=q;linear<words;linear+=step){unsigned i=(unsigned)(linear%ni);unsigned long long jk=linear/ni;unsigned j=(unsigned)(jk%nj),k=(unsigned)(jk/nj)+1u;unsigned long long p=(unsigned long long)k*nxy+(unsigned long long)j*ni+i;float rx=ruh[i],ry=rvh[j],rz=rmf[p],grid=M(M(M(dx,rx),M(dy,ry)),M(dz,rz));local+=(unsigned)(!PN(rx)||!PN(ry)||!PN(rz)||!PN(grid)||!FI(nm[p])||!FI(defv[p])||!FI(defh[p]));}
 if(local)atomicAdd(invalid,local);
}

extern "C" __global__ void cm1_child_turbsmag_interior_v2(
 const float*ruh,const float*rvh,const float*rmf,const float*nm,const float*defv,const float*defh,
 float*grdscl,float*lenscl,float*dissten,float*kmh,float*kmv,float*khh,float*khv,float*tkea,
 unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nxy=(unsigned long long)ni*nj,words=nxy*(nk-1u);const float csm=0.17841240763664246f,cme=0.10f,prinv=3.0f,dmin=1.0e-10f;
 for(unsigned long long linear=q;linear<words;linear+=step){unsigned i=(unsigned)(linear%ni);unsigned long long jk=linear/ni;unsigned j=(unsigned)(jk%nj),k=(unsigned)(jk/nj)+1u;unsigned long long p=(unsigned long long)k*nxy+(unsigned long long)j*ni+i;float grid=M(dx,ruh[i]);grid=M(grid,M(dy,rvh[j]));grid=M(grid,M(dz,rmf[p]));grid=cm1_glibc239_pow13_positive_normal(grid);float drive=S(A(defv[p],defh[p]),M(nm[p],prinv));drive=fmaxf(drive,dmin);float scale=M(csm,grid);float km=M(M(scale,scale),sqrtf(drive));float kh=M(km,prinv);float num=M(M(km,km),km);float scale2=M(scale,scale);float den=M(scale2,scale2);float diss=D(num,den);float ratio=D(km,M(cme,grid));float tke=M(ratio,ratio);grdscl[p]=grid;lenscl[p]=grid;kmh[p]=km;kmv[p]=km;khh[p]=kh;khv[p]=kh;dissten[p]=diss;tkea[p]=tke;}
}

extern "C" __global__ void cm1_child_turbsmag_faces_copy_v2(
 float*tkea,float*tke3d,float*tketen,unsigned ni,unsigned nj,unsigned nk){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nxy=(unsigned long long)ni*nj,total=nxy*(nk+1u);
 for(unsigned long long col=q;col<nxy;col+=step){tkea[col]=tkea[nxy+col];tkea[(unsigned long long)nk*nxy+col]=0.0f;}
 for(unsigned long long p=q;p<total;p+=step){tke3d[p]=tkea[p];tketen[p]=0.0f;}
}
'''
CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE = (
    CUDA_GLIBC_POW13_DEVICE_SOURCE + _GENERIC_KERNEL_SOURCE
)
CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_BYTES = (
    CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE.encode("ascii")
)
CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256 = (
    "5264deec03cc61fe3e6e96874cd215b6e62e44fa7adc071b4808b9d2b26470ea"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    text = str(value).lower().replace("compute_", "").replace("sm_", "")
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        raise ValueError(f"cannot decode CUDA compute capability {value!r}")
    return digits


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        "--std=c++17",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return common + (f"--gpu-architecture=compute_{arch}",), common + (
        f"--gpu-architecture=sm_{arch}",
    )


def _positive_normal_spacing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive normal binary32 number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    try:
        raw = struct.pack("<f", result)
    except OverflowError as exc:
        raise ValueError(f"{name} must fit binary32") from exc
    rounded = struct.unpack("<f", raw)[0]
    bits = struct.unpack("<I", raw)[0]
    if bits < 0x00800000 or bits >= 0x7F800000:
        refuse(
            code="GM1-CHILD-TURBSMAG-GENERIC-SPACING-DOMAIN",
            gate="gm1-child-turbsmag-generic",
            option=name,
            value=value,
            reason="spacing must round to a positive finite normal binary32 word",
            remedy="repair the admitted CM1 grid geometry",
        )
    return rounded


def _require_device_inputs(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    if set(arrays) != set(INPUT_ARRAY_NAMES):
        raise ValueError("generic child turbsmag CUDA input inventory changed")
    result: dict[str, Any] = {}
    for name in INPUT_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-contiguous CUDA float32 {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("generic child turbsmag CUDA inputs must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CompiledGenericChildTurbsmagInit:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_child_turbsmag_init_generic() -> CompiledGenericChildTurbsmagInit:
    if hashlib.sha256(CUDA_GLIBC_POW13_DEVICE_SOURCE.encode("ascii")).hexdigest() != (
        CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256
    ):
        raise RuntimeError("embedded glibc pow13 device source pin differs")
    actual = hashlib.sha256(CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_BYTES).hexdigest()
    if actual != CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256:
        raise RuntimeError(f"generic child turbsmag CUDA source pin differs: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE,
        "gm1_child_turbsmag_generic_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE,
        "gm1_child_turbsmag_generic_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_GM1_CHILD_TURBSMAG_GENERIC_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks a generic child turbsmag entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled generic child turbsmag image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_GM1_CHILD_TURBSMAG_GENERIC_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledGenericChildTurbsmagInit(
        module,
        kernels,
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        identity,
    )


def launch_cuda_child_turbsmag_init_generic(
    inputs: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: ChildTurbsmagInitSelection = DEFAULT_CHILD_TURBSMAG_INIT_SELECTION,
    pow13_selection: GlibcPow13Selection = DEFAULT_GLIBC_POW13_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGenericChildTurbsmagInit | None = None,
) -> CudaChildTurbsmagInitLaunch:
    """Validate and launch the arbitrary-positive-normal fresh-TKE slice."""

    require_child_turbsmag_init_selection(selection)
    require_glibc_pow13_selection(pow13_selection)
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    dx = _positive_normal_spacing("dx", dx)
    dy = _positive_normal_spacing("dy", dy)
    dz = _positive_normal_spacing("dz", dz)
    cupy = importlib.import_module("cupy")
    a = _require_device_inputs(cupy, inputs, ni, nj, nk)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compiled or compile_cuda_child_turbsmag_init_generic()
    interior_words = ni * nj * (nk - 1)
    total_words = ni * nj * (nk + 1)
    blocks = min(
        65535,
        (max(interior_words, total_words) + CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS - 1)
        // CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS,
    )
    grid = (blocks, 1, 1)
    block = (CUDA_GM1_CHILD_TURBSMAG_GENERIC_THREADS, 1, 1)
    with stream:
        invalid = cupy.zeros(1, dtype=cupy.uint32)
        compiled.kernels[0](
            grid,
            block,
            tuple(a[name] for name in INPUT_ARRAY_NAMES)
            + (
                invalid,
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
            ),
            stream=stream,
        )
        stream.synchronize()
        invalid_count = int(invalid.get()[0])
        if invalid_count:
            refuse(
                code="GM1-CHILD-TURBSMAG-GENERIC-OPERAND-DOMAIN",
                gate="gm1-child-turbsmag-generic",
                option="metrics_or_closure_inputs",
                value={"invalid_interior_cell_count": invalid_count},
                reason=(
                    "one or more consumed metric/grid-volume words are not "
                    "positive normal, or nm/defv/defh contains a non-finite word"
                ),
                remedy="repair the admitted child metric, calcnm, or calcdef producer",
            )
        outputs = {
            name: cupy.zeros(shapes[name], dtype=cupy.float32)
            for name in OUTPUT_ARRAY_NAMES
        }
        compiled.kernels[1](
            grid,
            block,
            tuple(a[name] for name in INPUT_ARRAY_NAMES)
            + tuple(
                outputs[name]
                for name in (
                    "grdscl",
                    "lenscl",
                    "dissten",
                    "kmh",
                    "kmv",
                    "khh",
                    "khv",
                    "tkea",
                )
            )
            + (
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
            ),
            stream=stream,
        )
        compiled.kernels[2](
            grid,
            block,
            (
                outputs["tkea"],
                outputs["tke3d"],
                outputs["tketen"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_CHILD_TURBSMAG_GENERIC_SCHEMA,
        "abi": CUDA_GM1_CHILD_TURBSMAG_GENERIC_ABI,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "shape_zyx": [nk, nj, ni],
        "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
        "metric_domain_gate": {
            "consumed_interior_cells": interior_words,
            "invalid_interior_cell_count": 0,
            "grid_volume_domain": "positive-finite-normal-binary32",
            "nm_def_domain": "finite-binary32",
        },
        "pow13": {
            "schema": "gm1.cuda-glibc239-pow13/v1",
            "device_source_sha256": CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
            "glibc_release": pow13_selection.glibc_release,
            "exponent_bits": f"{pow13_selection.exponent_bits:08x}",
        },
        "kernel": {
            "source_sha256": CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "names": list(CUDA_GM1_CHILD_TURBSMAG_GENERIC_KERNEL_NAMES),
        },
        "runtime": dict(compiled.runtime_identity),
        "tkea_parent_disposition": "not interpolated; diagnosed on new grid",
        "claim_limit": CUDA_GM1_CHILD_TURBSMAG_GENERIC_CLAIM_LIMIT,
    }
    return CudaChildTurbsmagInitLaunch(event, outputs, compiled, receipt)


__all__ = [
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_ABI",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_CLAIM_LIMIT",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_KERNEL_NAMES",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_SCHEMA",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_BYTES",
    "CUDA_GM1_CHILD_TURBSMAG_GENERIC_SOURCE_SHA256",
    "CompiledGenericChildTurbsmagInit",
    "compile_cuda_child_turbsmag_init_generic",
    "launch_cuda_child_turbsmag_init_generic",
]
