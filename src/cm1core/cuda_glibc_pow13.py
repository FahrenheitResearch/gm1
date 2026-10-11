"""Deterministic CUDA transcription of CM1's native ``REAL**0.33333333``.

CM1 r21.1 ``turbsmag`` evaluates the grid-volume power with a default-REAL
exponent.  The pinned gfortran 13.3 build lowers that expression to glibc
2.39 ``powf``.  CUDA ``powf`` is a different function.  This module carries
the positive-normal-input fast path of glibc 2.39 ``e_powf.c`` statement for
statement, using explicit round-to-nearest operations so NVRTC cannot contract
the binary64 polynomials.

The admitted domain is every positive, finite, normal binary32 base with the
single exponent word 0x3eaaaaab.  That covers arbitrary valid CM1 grid-volume
operands; zero, subnormal, negative, infinity, NaN, and other exponents remain
fail-closed rather than inheriting unproved libm edge semantics.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_GLIBC_POW13_SCHEMA = "gm1.cuda-glibc239-pow13/v1"
CUDA_GLIBC_POW13_ABI = "gm1.cuda-glibc239-pow13-positive-normal-f32/v1"
CUDA_GLIBC_POW13_THREADS = 256
CUDA_GLIBC_POW13_KERNEL_NAME = "cm1_glibc239_pow13_probe_v1"
CUDA_GLIBC_POW13_VALIDATE_KERNEL_NAME = "cm1_glibc239_pow13_validate_v1"
CM1_POW13_EXPONENT_BITS = 0x3EAAAAAB
CM1_POW13_EXPONENT = 0.3333333432674408

GLIBC_RELEASE = "2.39"
GLIBC_SOURCE_ANCHORS = {
    "algorithm": "sysdeps/ieee754/flt-32/e_powf.c",
    "log2_table": "sysdeps/ieee754/flt-32/e_powf_log2_data.c",
    "exp2_table": "sysdeps/ieee754/flt-32/e_exp2f_data.c",
    "configuration": "sysdeps/ieee754/flt-32/math_config.h",
}
CM1_SOURCE_ANCHOR = "src/turb.F:3434-3436"
CM1_TURB_SOURCE_SHA256 = (
    "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d"
)


@dataclass(frozen=True, slots=True)
class GlibcPow13Selection:
    glibc_release: str = GLIBC_RELEASE
    exponent_bits: int = CM1_POW13_EXPONENT_BITS
    positive_finite_normal_only: bool = True
    native_rounding: str = "round-to-nearest-ties-to-even"


DEFAULT_GLIBC_POW13_SELECTION = GlibcPow13Selection()


def require_glibc_pow13_selection(selection: GlibcPow13Selection) -> None:
    """Refuse any exponent, libm, rounding, or operand family not transcribed."""

    if not isinstance(selection, GlibcPow13Selection):
        raise TypeError("selection must be GlibcPow13Selection")
    expected = GlibcPow13Selection()
    for option in (
        "glibc_release",
        "exponent_bits",
        "positive_finite_normal_only",
        "native_rounding",
    ):
        value = getattr(selection, option)
        wanted = getattr(expected, option)
        if value != wanted or type(value) is not type(wanted):
            refuse(
                code="GM1-GLIBC-POW13-UNSUPPORTED",
                gate="gm1-glibc239-pow13",
                option=option,
                value=value,
                reason=(
                    f"the deterministic transcription covers only {option}="
                    f"{wanted!r}"
                ),
                remedy="add a separately pinned native libm oracle and CUDA path",
            )


# Direct positive-normal fast-path transcription of glibc 2.39:
#   sysdeps/ieee754/flt-32/e_powf.c
#   sysdeps/ieee754/flt-32/e_powf_log2_data.c
#   sysdeps/ieee754/flt-32/e_exp2f_data.c
# POWF_SCALE is 1 because TOINT_INTRINSICS is false on the pinned x86-64 build.
CUDA_GLIBC_POW13_SOURCE = r'''// gm1.cuda-glibc239-pow13/v1
#define DAD(a,b) __dadd_rn((a),(b))
#define DSU(a,b) __dsub_rn((a),(b))
#define DMU(a,b) __dmul_rn((a),(b))

__device__ const double CM1_POWF_INVC[16]={
0x1.661ec79f8f3bep+0,0x1.571ed4aaf883dp+0,0x1.49539f0f010bp+0,0x1.3c995b0b80385p+0,
0x1.30d190c8864a5p+0,0x1.25e227b0b8eap+0,0x1.1bb4a4a1a343fp+0,0x1.12358f08ae5bap+0,
0x1.0953f419900a7p+0,0x1p+0,0x1.e608cfd9a47acp-1,0x1.ca4b31f026aap-1,
0x1.b2036576afce6p-1,0x1.9c2d163a1aa2dp-1,0x1.886e6037841edp-1,0x1.767dcf5534862p-1};
__device__ const double CM1_POWF_LOGC[16]={
-0x1.efec65b963019p-2,-0x1.b0b6832d4fca4p-2,-0x1.7418b0a1fb77bp-2,-0x1.39de91a6dcf7bp-2,
-0x1.01d9bf3f2b631p-2,-0x1.97c1d1b3b7afp-3,-0x1.2f9e393af3c9fp-3,-0x1.960cbbf788d5cp-4,
-0x1.a6f9db6475fcep-5,0x0p+0,0x1.338ca9f24f53dp-4,0x1.476a9543891bap-3,
0x1.e840b4ac4e4d2p-3,0x1.40645f0c6651cp-2,0x1.88e9c2c1b9ff8p-2,0x1.ce0a44eb17bccp-2};
__device__ const double CM1_POWF_A[5]={
0x1.27616c9496e0bp-2,-0x1.71969a075c67ap-2,0x1.ec70a6ca7baddp-2,
-0x1.7154748bef6c8p-1,0x1.71547652ab82bp+0};

__device__ const unsigned long long CM1_EXP2F_TAB[32]={
0x3ff0000000000000ULL,0x3fefd9b0d3158574ULL,0x3fefb5586cf9890fULL,0x3fef9301d0125b51ULL,
0x3fef72b83c7d517bULL,0x3fef54873168b9aaULL,0x3fef387a6e756238ULL,0x3fef1e9df51fdee1ULL,
0x3fef06fe0a31b715ULL,0x3feef1a7373aa9cbULL,0x3feedea64c123422ULL,0x3feece086061892dULL,
0x3feebfdad5362a27ULL,0x3feeb42b569d4f82ULL,0x3feeab07dd485429ULL,0x3feea47eb03a5585ULL,
0x3feea09e667f3bcdULL,0x3fee9f75e8ec5f74ULL,0x3feea11473eb0187ULL,0x3feea589994cce13ULL,
0x3feeace5422aa0dbULL,0x3feeb737b0cdc5e5ULL,0x3feec49182a3f090ULL,0x3feed503b23e255dULL,
0x3feee89f995ad3adULL,0x3feeff76f2fb5e47ULL,0x3fef199bdd85529cULL,0x3fef3720dcef9069ULL,
0x3fef5818dcfba487ULL,0x3fef7c97337b9b5fULL,0x3fefa4afa2a490daULL,0x3fefd0765b6e4540ULL};

__device__ __forceinline__ double cm1_powf_log2_positive_normal(unsigned ix){
  unsigned tmp=ix-0x3f330000u;
  int i=(int)((tmp>>19)&15u);
  unsigned top=tmp&0xff800000u;
  unsigned iz=ix-top;
  int k=(int)top>>23;
  double z=(double)__uint_as_float(iz);
  double r=DSU(DMU(z,CM1_POWF_INVC[i]),1.0);
  double y0=DAD(CM1_POWF_LOGC[i],(double)k);
  double r2=DMU(r,r);
  double y=DAD(DMU(CM1_POWF_A[0],r),CM1_POWF_A[1]);
  double p=DAD(DMU(CM1_POWF_A[2],r),CM1_POWF_A[3]);
  double r4=DMU(r2,r2);
  double q=DAD(DMU(CM1_POWF_A[4],r),y0);
  q=DAD(DMU(p,r2),q);
  return DAD(DMU(y,r4),q);
}

__device__ __forceinline__ double cm1_powf_exp2(double xd){
  const double shift=0x1.8p+52/32.0;
  const double p0=0x1.c6af84b912394p-5;
  const double p1=0x1.ebfce50fac4f3p-3;
  const double p2=0x1.62e42ff0c52d6p-1;
  double kd=DAD(xd,shift);
  unsigned long long ki=(unsigned long long)__double_as_longlong(kd);
  kd=DSU(kd,shift);
  double r=DSU(xd,kd);
  unsigned long long t=CM1_EXP2F_TAB[ki&31ULL];
  t+=ki<<(52-5);
  double s=__longlong_as_double((long long)t);
  double z=DAD(DMU(p0,r),p1);
  double r2=DMU(r,r);
  double out=DAD(DMU(p2,r),1.0);
  out=DAD(DMU(z,r2),out);
  return DMU(out,s);
}

__device__ __forceinline__ float cm1_glibc239_pow13_positive_normal(float x){
  double logx=cm1_powf_log2_positive_normal(__float_as_uint(x));
  double ylogx=DMU((double)__uint_as_float(0x3eaaaaabu),logx);
  return __double2float_rn(cm1_powf_exp2(ylogx));
}

extern "C" __global__ void cm1_glibc239_pow13_probe_v1(
    const float*operand,float*result,unsigned long long words){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
  for(unsigned long long p=q;p<words;p+=step)
    result[p]=cm1_glibc239_pow13_positive_normal(operand[p]);
}

extern "C" __global__ void cm1_glibc239_pow13_validate_v1(
    const float*operand,unsigned*invalid,unsigned long long words){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
  unsigned local=0u;
  for(unsigned long long p=q;p<words;p+=step){
    unsigned bits=__float_as_uint(operand[p]);
    local+=(bits<0x00800000u||bits>=0x7f800000u);
  }
  if(local)atomicAdd(invalid,local);
}
'''
CUDA_GLIBC_POW13_SOURCE_BYTES = CUDA_GLIBC_POW13_SOURCE.encode("ascii")
_CUDA_GLIBC_POW13_PROBE_MARKER = (
    '\nextern "C" __global__ void cm1_glibc239_pow13_probe_v1('
)
if CUDA_GLIBC_POW13_SOURCE.count(_CUDA_GLIBC_POW13_PROBE_MARKER) != 1:
    raise RuntimeError("glibc pow13 device/probe source boundary changed")
CUDA_GLIBC_POW13_DEVICE_SOURCE = (
    CUDA_GLIBC_POW13_SOURCE.split(_CUDA_GLIBC_POW13_PROBE_MARKER, 1)[0] + "\n"
)
CUDA_GLIBC_POW13_DEVICE_SOURCE_BYTES = CUDA_GLIBC_POW13_DEVICE_SOURCE.encode("ascii")
CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256 = (
    "cf31b641462b6f2f6f4c2f1fa6b741153b08090b6f208780b2f95e23b39c57dd"
)
# Literal pin is filled from the byte-exact source above and checked before NVRTC.
CUDA_GLIBC_POW13_SOURCE_SHA256 = (
    "bc128c5bdec183a42820a43dcfa6f26380346ecf62fe3e71cdfd7e648447025a"
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


@dataclass(frozen=True, slots=True)
class CompiledGlibcPow13:
    module_owner: Any
    kernel: Any
    validator: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_glibc_pow13() -> CompiledGlibcPow13:
    actual = hashlib.sha256(CUDA_GLIBC_POW13_SOURCE_BYTES).hexdigest()
    if actual != CUDA_GLIBC_POW13_SOURCE_SHA256:
        raise RuntimeError(f"glibc pow13 CUDA source pin differs: {actual}")
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
        CUDA_GLIBC_POW13_SOURCE,
        "gm1_glibc239_pow13_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GLIBC_POW13_SOURCE,
        "gm1_glibc239_pow13_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in (
            CUDA_GLIBC_POW13_KERNEL_NAME,
            CUDA_GLIBC_POW13_VALIDATE_KERNEL_NAME,
        )
    ):
        raise ValueError("compiled PTX lacks a glibc pow13 entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled glibc pow13 image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
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
    return CompiledGlibcPow13(
        module,
        module.get_function(CUDA_GLIBC_POW13_KERNEL_NAME),
        module.get_function(CUDA_GLIBC_POW13_VALIDATE_KERNEL_NAME),
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        identity,
    )


@dataclass(frozen=True, slots=True)
class CudaGlibcPow13Launch:
    ready_event: Any
    result: Any
    compiled: CompiledGlibcPow13
    receipt: Mapping[str, Any]


def launch_cuda_glibc_pow13(
    operand: Any,
    *,
    selection: GlibcPow13Selection = DEFAULT_GLIBC_POW13_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGlibcPow13 | None = None,
) -> CudaGlibcPow13Launch:
    """Evaluate glibc 2.39 ``powf(x, 0.33333333f)`` on a CUDA f32 vector.

    The caller must supply positive finite normal operands from an admitted
    CM1 geometry producer.  Production integrations must preserve that
    producer's independent positivity/finite/range gate.
    """

    require_glibc_pow13_selection(selection)
    cupy = importlib.import_module("cupy")
    if (
        not isinstance(operand, cupy.ndarray)
        or operand.dtype != cupy.dtype("float32")
        or operand.ndim != 1
        or not operand.flags.c_contiguous
        or int(operand.size) < 1
    ):
        raise ValueError(
            "pow13 operand must be a nonempty one-dimensional contiguous CUDA float32"
        )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compiled or compile_cuda_glibc_pow13()
    with stream:
        invalid = cupy.zeros(1, dtype=cupy.uint32)
        result = cupy.empty_like(operand)
        blocks = min(
            65535,
            (int(operand.size) + CUDA_GLIBC_POW13_THREADS - 1)
            // CUDA_GLIBC_POW13_THREADS,
        )
        compiled.validator(
            (blocks, 1, 1),
            (CUDA_GLIBC_POW13_THREADS, 1, 1),
            (operand, invalid, cupy.uint64(int(operand.size))),
            stream=stream,
        )
        stream.synchronize()
        invalid_count = int(invalid.get()[0])
        if invalid_count:
            refuse(
                code="GM1-GLIBC-POW13-OPERAND-DOMAIN",
                gate="gm1-glibc239-pow13",
                option="operand",
                value={"invalid_word_count": invalid_count},
                reason=(
                    "the deterministic fast path accepts only positive finite "
                    "normal binary32 bases"
                ),
                remedy="repair the admitted CM1 geometry producer",
            )
        compiled.kernel(
            (blocks, 1, 1),
            (CUDA_GLIBC_POW13_THREADS, 1, 1),
            (operand, result, cupy.uint64(int(operand.size))),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GLIBC_POW13_SCHEMA,
        "abi": CUDA_GLIBC_POW13_ABI,
        "operand_domain": "positive-finite-normal-binary32",
        "word_count": int(operand.size),
        "exponent_bits": f"{CM1_POW13_EXPONENT_BITS:08x}",
        "authority": {
            "cm1": CM1_SOURCE_ANCHOR,
            "glibc_release": GLIBC_RELEASE,
            "glibc_sources": dict(GLIBC_SOURCE_ANCHORS),
        },
        "kernel": {
            "name": CUDA_GLIBC_POW13_KERNEL_NAME,
            "source_sha256": CUDA_GLIBC_POW13_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
    }
    return CudaGlibcPow13Launch(event, result, compiled, receipt)


__all__ = [
    "CM1_POW13_EXPONENT",
    "CM1_POW13_EXPONENT_BITS",
    "CM1_SOURCE_ANCHOR",
    "CM1_TURB_SOURCE_SHA256",
    "CUDA_GLIBC_POW13_ABI",
    "CUDA_GLIBC_POW13_DEVICE_SOURCE",
    "CUDA_GLIBC_POW13_DEVICE_SOURCE_BYTES",
    "CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256",
    "CUDA_GLIBC_POW13_KERNEL_NAME",
    "CUDA_GLIBC_POW13_SCHEMA",
    "CUDA_GLIBC_POW13_SOURCE",
    "CUDA_GLIBC_POW13_SOURCE_BYTES",
    "CUDA_GLIBC_POW13_SOURCE_SHA256",
    "CUDA_GLIBC_POW13_VALIDATE_KERNEL_NAME",
    "CompiledGlibcPow13",
    "CudaGlibcPow13Launch",
    "DEFAULT_GLIBC_POW13_SELECTION",
    "GLIBC_RELEASE",
    "GLIBC_SOURCE_ANCHORS",
    "GlibcPow13Selection",
    "compile_cuda_glibc_pow13",
    "launch_cuda_glibc_pow13",
    "require_glibc_pow13_selection",
]
