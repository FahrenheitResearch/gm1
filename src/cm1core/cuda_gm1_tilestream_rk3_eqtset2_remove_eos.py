"""Tile-local final-RK eqtset2 removal, snapshot, and moist EOS.

CM1 r21.1 performs this cell-local prefix at ``solve2.F:1561-1595`` after
the final scalar/moist RK update and before the ``apmasscon`` layer
reductions.  This module exposes that exact narrow CUDA operator for fixed
tile buffers, including the four in-place held-carrier snapshots and the
pre-reduction pressure/density rebuild.

The emitted token is intentionally local.  No producer currently binds the
fresh broader RK3 scalar/moist lineage to these buffers, so this module does
not emit a live tile-stream stage token and cannot complete RK3.  Momentum
boundaries, RK3 TKE, the split mass finish, output hashes, solve3, MPI, and
history/checkpoint publication remain separate source-ordered gates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from .cuda_glibc_pow13 import (
    CUDA_GLIBC_POW13_DEVICE_SOURCE,
    CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
    _canonical_arch,
    _image_options,
)
from .errors import refuse
from .gm1_runtime import (
    CM1_MIN_NORMAL_DT_S,
    positive_normal_f32_bits,
)


CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-eqtset2-remove-eos-local-token/v1"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_STATUS = (
    "ENQUEUED_LOCAL_RK3_EQTSET2_REMOVE_SNAPSHOT_EOS_ONLY"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_ABI = (
    "gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v2"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_KERNEL_NAME = (
    "gm1_rk3_eqtset2_remove_snapshot_eos_f32_v2"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_THREADS = 256
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_CONSTANTS_SHA256 = (
    "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1561-1579",
    "src/solve2.F:1582-1595",
    "src/constants.F:17-18,104-137",
)
GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_CLAIM_LIMIT = (
    "one local selected-profile CM1 r21.1 final-RK eqtset2 held-rate "
    "remove/snapshot plus moist pressure-density EOS enqueue over active "
    "physical tile words on one CUDA stream. Fresh broader RK3 scalar/moist "
    "lineage, apmasscon, momentum boundaries, RK3 TKE, corrected output "
    "hashes, completed RK3, solve3, WSM6, MPI, radiation/open histories, "
    "forecast, tornado, and five-metre product claims remain unavailable"
)

_BUFFER_ORDER = (
    "pi0",
    "th0",
    "pp3d",
    "th3d",
    "qv",
    "qc",
    "qpten",
    "qtten",
    "qvten",
    "qcten",
    "prs",
    "rho",
)
_MUTATED_ORDER = (
    "pp3d",
    "qpten",
    "th3d",
    "qtten",
    "qv",
    "qvten",
    "qc",
    "qcten",
    "prs",
    "rho",
)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# CM1 r21.1 carries the large step in ``dt``, a default-REAL (binary32)
# variable, not a parameter: ``solve2.F:145`` declares
# ``real, intent(inout) :: dt`` and ``solve2.F:1569-1576`` multiplies the four
# held rates by it.  ``CM1_DEFAULT_DT_S`` is only the value
# this port's configuration defaults ``run.dt_s`` to; the kernel takes the step
# as an argument, exactly as the Fortran does, and ``CM1_DT_F32_BITS`` is that
# default's word -- the identity the welded predecessor applied, kept so the
# default can be proven unchanged.
#
# It is deliberately not a default on any function here.  A launcher or oracle
# that defaulted its step would apply the half second to a caller that simply
# omitted the argument: a wrong answer that raises nothing and looks like a
# right one.  ``dt_seconds`` is keyword-only and required throughout this
# module, so an omission is a TypeError at the call site, never a silent 0.5.
CM1_DEFAULT_DT_S = 0.5
CM1_DT_F32_BITS = 0x3F000000
CM1_P00_F32_BITS = 0x47C35000
CM1_RD_F32_BITS = 0x438F851F
CM1_RV_F32_BITS = 0x43E6C000
CM1_CP_F32_BITS = 0x447B6CCD
CM1_CPDRD_F32_BITS = 0x40603C81

if _f32_bits(0.5) != CM1_DT_F32_BITS:
    raise RuntimeError("host binary32 dt encoding changed")


CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE = CUDA_GLIBC_POW13_DEVICE_SOURCE + r'''
// gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v2
__device__ __forceinline__ int gm1_finite_f32(float x) {
  return (__float_as_uint(x) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ int gm1_positive_normal_f32(float x) {
  unsigned bits = __float_as_uint(x);
  return bits >= 0x00800000u && bits < 0x7f800000u;
}

__device__ __forceinline__ float gm1_glibc239_pow_cpdrd_positive_normal(float x) {
  double logx = cm1_powf_log2_positive_normal(__float_as_uint(x));
  double ylogx = DMU((double)__uint_as_float(0x40603c81u), logx);
  return __double2float_rn(cm1_powf_exp2(ylogx));
}

extern "C" __global__ void gm1_rk3_eqtset2_remove_snapshot_eos_f32_v2(
    const float* __restrict__ pi0,
    const float* __restrict__ th0,
    float* __restrict__ pp3d,
    float* __restrict__ th3d,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ qpten,
    float* __restrict__ qtten,
    float* __restrict__ qvten,
    float* __restrict__ qcten,
    float* __restrict__ prs,
    float* __restrict__ rho,
    unsigned* __restrict__ invalid_words,
    unsigned tile_nx,
    unsigned tile_ny,
    unsigned active_nx,
    unsigned active_ny,
    unsigned nk,
    float dt)
{
  unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long active_xy = (unsigned long long)active_nx * active_ny;
  unsigned long long active_words = active_xy * nk;
  if (q >= active_words) return;
  unsigned i = (unsigned)(q % active_nx);
  unsigned long long row = q / active_nx;
  unsigned j = (unsigned)(row % active_ny);
  unsigned k = (unsigned)(row / active_ny);
  unsigned long long p = ((unsigned long long)k * tile_ny + j) * tile_nx + i;

  float pp = __fsub_rn(pp3d[p], __fmul_rn(dt, qpten[p]));
  pp3d[p] = pp;
  qpten[p] = pp;
  float th = __fsub_rn(th3d[p], __fmul_rn(dt, qtten[p]));
  th3d[p] = th;
  qtten[p] = th;
  float vapor = __fsub_rn(qv[p], __fmul_rn(dt, qvten[p]));
  qv[p] = vapor;
  qvten[p] = vapor;
  float cloud = __fsub_rn(qc[p], __fmul_rn(dt, qcten[p]));
  qc[p] = cloud;
  qcten[p] = cloud;

  float pi = __fadd_rn(pi0[p], pp);
  float theta = __fadd_rn(th0[p], th);
  if (!gm1_finite_f32(pp) || !gm1_finite_f32(th) ||
      !gm1_finite_f32(vapor) || !gm1_finite_f32(cloud) ||
      !gm1_positive_normal_f32(pi) || !gm1_positive_normal_f32(theta)) {
    prs[p] = __uint_as_float(0x7fc00000u);
    rho[p] = __uint_as_float(0x7fc00000u);
    atomicAdd(invalid_words, 1u);
    return;
  }

  float pressure = __fmul_rn(
      __uint_as_float(0x47c35000u),
      gm1_glibc239_pow_cpdrd_positive_normal(pi));
  float qv_nonnegative = fmaxf(__uint_as_float(0x00000000u), vapor);
  float gas = __fadd_rn(
      __uint_as_float(0x438f851fu),
      __fmul_rn(qv_nonnegative, __uint_as_float(0x43e6c000u)));
  float denominator = __fmul_rn(theta, pi);
  denominator = __fmul_rn(denominator, gas);
  float density = __fdiv_rn(pressure, denominator);
  if (!gm1_positive_normal_f32(pressure) ||
      !gm1_positive_normal_f32(denominator) ||
      !gm1_positive_normal_f32(density)) {
    prs[p] = __uint_as_float(0x7fc00000u);
    rho[p] = __uint_as_float(0x7fc00000u);
    atomicAdd(invalid_words, 1u);
    return;
  }
  prs[p] = pressure;
  rho[p] = density;
}
'''
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_BYTES = (
    CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE.encode("ascii")
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_SHA256 = (
    "2fbf6a23fc578cee90f9ece38f5e34841bd2d5d069a8159fc6cf39f6d22cb9c1"
)

# The immediate predecessor of this kernel welded the large step into the PTX
# as ``const float dt = __uint_as_float(0x3f000000u);``.  Its exact source
# bytes are pinned here by digest, and ``welded_predecessor_source`` rebuilds
# them from the live source by the four mechanical edits below.  That rebuild
# succeeding is the proof that threading ``dt`` changed nothing else: every
# statement, every intrinsic, every operand order and every whitespace byte of
# the operator is identical, so the only remaining question is what word the
# ``dt`` register holds -- and at the default step the launcher supplies
# ``CM1_DT_F32_BITS``, the same 0x3f000000 the weld held.  The digest is also
# the ``source_sha256`` recorded in this operator's sealed device receipt
# (the ``gm1-rk3-eqtset2-remove-eos`` receipt under ``evidence/validation/``),
# so the pin is bound to real measured hardware output, not to a local copy.
#
# What that sealed receipt is, and is not, stated rather than left implied: it
# was produced by the welded v1 image on the validation node, and it still
# records ``local_tokens.resident.kernel.abi`` as
# ``gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v1`` and
# ``constants_f32_bits.dt`` as ``3f000000``, a key the v2 receipt no longer
# emits because the step is no longer a constant.  The v2 image has never been
# executed on that node.  The receipt is therefore evidence about the
# predecessor -- which is exactly the role it plays here, as the measured
# provenance of the digest the bit-identity rebuild reproduces -- and it is not
# evidence that the threaded image ran on validation hardware.  That claim is
# carried only by the GPU-gated welded-vs-threaded differential test, which
# compiles both images and compares device output word for word wherever a CUDA
# device is present.
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELDED_PREDECESSOR_SHA256 = (
    "de47fca9ea02a5cc330491bc2ee6499f9ba4331b64fd9a2119715fe834f48673"
)
# The ABI the sealed device receipt was produced under, kept beside the live one
# so a test can assert the relationship instead of a reader having to notice it.
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_ABI = (
    "gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v1"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_CLAIM_LIMIT = (
    "the sealed device receipt measures the welded-dt v1 image on the validation "
    "node and is the provenance of the welded-predecessor source digest; the "
    "threaded v2 image has never been executed on that node, and its raw-bit "
    "equality with the welded image at the default step is carried by the "
    "GPU-gated welded-vs-threaded differential test alone"
)
CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELD_EDITS = (
    (
        "// gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v2",
        "// gm1.cuda-rk3-eqtset2-remove-snapshot-moist-eos-f32/v1",
    ),
    (
        'extern "C" __global__ void gm1_rk3_eqtset2_remove_snapshot_eos_f32_v2(',
        'extern "C" __global__ void gm1_rk3_eqtset2_remove_snapshot_eos_f32_v1(',
    ),
    ("    unsigned nk,\n    float dt)\n", "    unsigned nk)\n"),
    (
        "  float pp = __fsub_rn(pp3d[p]",
        "  const float dt = __uint_as_float(0x3f000000u);\n"
        "  float pp = __fsub_rn(pp3d[p]",
    ),
)


def welded_predecessor_source() -> str:
    """Rebuild the welded-``dt`` predecessor from the live kernel source.

    Each substitution must match exactly once; a fifth change anywhere in the
    operator would either break one of those counts or change the digest the
    caller compares against, so this cannot pass while the arithmetic differs.
    """

    text = CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE
    for new, old in CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELD_EDITS:
        if text.count(new) != 1:
            raise RuntimeError(
                "the live kernel source no longer carries exactly one "
                f"{new!r}; the welded-predecessor rebuild is not mechanical"
            )
        text = text.replace(new, old)
    return text


def require_default_dt_bit_identity() -> Mapping[str, object]:
    """Prove the threaded default step is the welded constant, bit for bit.

    Returns the audit record; raises rather than returning a false one.  The
    claim is narrow and exact: the operator source differs from its welded
    predecessor only in where ``dt`` comes from, and at
    ``CM1_DEFAULT_DT_S`` the launcher hands the kernel the identical binary32
    word the weld held, so both kernels evaluate the same ``__fmul_rn`` and
    ``__fsub_rn`` on the same operands and must produce the same bits.
    """

    rebuilt = welded_predecessor_source()
    digest = hashlib.sha256(rebuilt.encode("ascii")).hexdigest()
    if digest != CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELDED_PREDECESSOR_SHA256:
        raise RuntimeError(
            "the rebuilt welded predecessor does not match its pinned digest; "
            "threading dt changed something other than the dt binding"
        )
    default_bits = _f32_bits(CM1_DEFAULT_DT_S)
    if default_bits != CM1_DT_F32_BITS:
        raise RuntimeError("the default large step is no longer 0x3f000000")
    if positive_normal_f32_bits(CM1_DEFAULT_DT_S) != CM1_DT_F32_BITS:
        raise RuntimeError("the default large step is not an admitted step")
    return MappingProxyType(
        {
            "schema": "gm1.rk3-eqtset2-remove-eos-default-dt-identity/v1",
            "authority": "src/solve2.F:145,1569-1576",
            "welded_predecessor_sha256": digest,
            "threaded_source_sha256": (
                CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_SHA256
            ),
            "weld_edit_count": len(CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELD_EDITS),
            "default_dt_seconds": CM1_DEFAULT_DT_S,
            "default_dt_f32_bits": f"{default_bits:08x}",
            "welded_dt_f32_bits": f"{CM1_DT_F32_BITS:08x}",
            "identical_operands": True,
            "live_abi": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_ABI,
            "sealed_evidence_abi": (
                CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_ABI
            ),
            "sealed_evidence_claim_limit": (
                CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_CLAIM_LIMIT
            ),
            "claim_limit": (
                "the threaded-dt operator source is the welded-dt operator "
                "source with the dt binding replaced and nothing else, and the "
                "default step supplies the welded word; raw-bit equality of "
                "two compiled kernels on real device memory is the separate "
                "GPU-gated differential test"
            ),
        }
    )



def admitted_dt_f32_bits(dt_seconds: object, *, gate: str) -> int:
    """Return the binary32 word this operator will apply, or refuse by value.

    ``solve2.F:1569-1576`` removes the four held rates with ``state - dt*rate``
    and stores the result back over the rate, so the sign and magnitude of
    ``dt`` decide what the final RK stage hands to microphysics.  Three values
    are refused here rather than in the kernel, where they would be silent:

    * a non-positive step re-adds the estimated diabatic terms instead of
      removing them, and ``solve3``'s ``(state - snapshot)/dt`` inverse
      (``cuda_cartesian_finish.post_physics``) divides by zero or flips the
      sign of every microphysics tendency;
    * a non-finite step makes every ``pp3d``/``th3d``/``qv``/``qc`` word NaN
      in one large step;
    * a step that is not a positive *normal* binary32 number -- anything below
      ``CM1_MIN_NORMAL_DT_S`` -- flushes to zero or to a subnormal inside the
      kernel, so the held-rate removal would quietly not happen at all while
      every other stage advanced by the requested step.

    This is a refusal by value, not a pin: every positive normal step is
    admitted, and the kernel takes ``dt`` as an argument.
    """

    if isinstance(dt_seconds, bool) or not isinstance(dt_seconds, (int, float)):
        raise TypeError("dt_seconds must be a number")
    bits = positive_normal_f32_bits(dt_seconds)
    if bits is None:
        refuse(
            code="GM1-RK3-EQTSET2-DT-NOT-POSITIVE-NORMAL-F32",
            gate=gate,
            option="dt_seconds",
            value=dt_seconds,
            reason=(
                "solve2.F:1569-1576 removes the held pressure, potential "
                "temperature, vapour and cloud rates as state-dt*rate and then "
                "overwrites the rate with the state; a step that is not a "
                "positive normal binary32 number either re-adds the estimated "
                "diabatic terms, makes every mutated word NaN, or flushes the "
                "removal to zero while every other stage advances by the "
                "requested step, and solve3's (state-snapshot)/dt inverse then "
                "divides by that same unusable step"
            ),
            remedy=(
                "pass a finite dt_seconds of at least "
                f"{CM1_MIN_NORMAL_DT_S!r} s whose binary32 rounding is a "
                "positive normal number"
            ),
        )
    return bits


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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _stream_pointer(stream: object) -> int:
    value = getattr(stream, "ptr", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("CUDA stream pointer must be a positive integer")
    return value


def _pointer_inventory(arrays: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(arrays[name].data.ptr) for name in _BUFFER_ORDER}


def _pointer_inventory_sha256(pointers: Mapping[str, int]) -> str:
    return hashlib.sha256(_canonical(dict(pointers))).hexdigest()


def tiled_rk3_eqtset2_remove_eos_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    shape = (
        _positive_int("nk", nk),
        _positive_int("tile_ny", tile_ny),
        _positive_int("tile_nx", tile_nx),
    )
    return MappingProxyType({name: shape for name in _BUFFER_ORDER})


def tiled_rk3_eqtset2_remove_eos_buffer_nbytes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, int]:
    shapes = tiled_rk3_eqtset2_remove_eos_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values()) + 4
    return MappingProxyType(
        {
            "device_per_buffer_bytes": per_buffer,
            "double_device_bytes": 2 * per_buffer,
            "field_count": len(_BUFFER_ORDER),
            "validation_counter_bytes": 4,
            "full_300x300x75_device_bytes": 12 * 75 * 300 * 300 * 4 + 4,
        }
    )


@dataclass(frozen=True, slots=True)
class CompiledGm1Rk3Eqset2RemoveEos:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def _compiled_receipt(compiled: CompiledGm1Rk3Eqset2RemoveEos) -> dict[str, object]:
    return {
        "abi": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_ABI,
        "name": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_KERNEL_NAME,
        "source_sha256": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_SHA256,
        "embedded_glibc_pow_source_sha256": (
            CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256
        ),
        "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
        "ptx_bytes": len(compiled.ptx),
        "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
        "cubin_bytes": len(compiled.cubin),
        "target_arch": compiled.target_arch,
        "ptx_options": list(compiled.ptx_options),
        "cubin_options": list(compiled.cubin_options),
    }


def compile_cuda_gm1_rk3_eqtset2_remove_eos() -> CompiledGm1Rk3Eqset2RemoveEos:
    actual = hashlib.sha256(
        CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_BYTES
    ).hexdigest()
    if actual != CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_SHA256:
        raise RuntimeError(f"RK3 eqtset2 remove/EOS CUDA source pin differs: {actual}")
    if hashlib.sha256(CUDA_GLIBC_POW13_DEVICE_SOURCE.encode("ascii")).hexdigest() != (
        CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256
    ):
        raise RuntimeError("embedded glibc pow device source pin differs")
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
        CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE,
        "gm1_rk3_eqtset2_remove_eos_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE,
        "gm1_rk3_eqtset2_remove_eos_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_KERNEL_NAME}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks RK3 eqtset2 remove/EOS entry")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled RK3 eqtset2 image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledGm1Rk3Eqset2RemoveEos(
        module,
        module.get_function(CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_KERNEL_NAME),
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        runtime_identity,
    )


@dataclass(frozen=True, slots=True)
class Gm1Rk3Eqset2RemoveEosLocalToken:
    """Exact identity of a local enqueue, never a broader RK3 admission."""

    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3Eqset2RemoveEosLaunch:
    ready_event: Any
    compiled: CompiledGm1Rk3Eqset2RemoveEos
    stream: Any
    arrays: Mapping[str, Any]
    invalid_words: Any
    token: Gm1Rk3Eqset2RemoveEosLocalToken


def _validate_device_arrays(arrays: Mapping[str, Any]) -> tuple[int, int, int]:
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError(
            "RK3 eqtset2 tile array inventory/order must be "
            + ",".join(_BUFFER_ORDER)
        )
    cupy = importlib.import_module("cupy")
    first = arrays[_BUFFER_ORDER[0]]
    if (
        not isinstance(first, cupy.ndarray)
        or first.dtype != cupy.dtype("float32")
        or first.ndim != 3
        or not first.flags.c_contiguous
    ):
        raise ValueError("RK3 eqtset2 tile arrays must be CUDA C-f32 rank 3")
    nk, tile_ny, tile_nx = map(int, first.shape)
    if min(nk, tile_ny, tile_nx) < 1:
        raise ValueError("RK3 eqtset2 tile shape must be positive")
    pointers: list[int] = []
    for name in _BUFFER_ORDER:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != first.shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be CUDA C-f32 with shape {tuple(first.shape)}"
            )
        pointers.append(int(value.data.ptr))
    if len(set(pointers)) != len(pointers):
        raise ValueError("RK3 eqtset2 tile arrays must not alias")
    return tile_nx, tile_ny, nk


def launch_cuda_gm1_rk3_eqtset2_remove_eos_local(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    dt_seconds: float,
    stream: Any | None = None,
    compiled: CompiledGm1Rk3Eqset2RemoveEos | None = None,
) -> CudaGm1Rk3Eqset2RemoveEosLaunch:
    """Enqueue the exact local prefix over one tile's active rectangle."""

    tile_nx, tile_ny, nk = _validate_device_arrays(arrays)
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if active_nx > tile_nx or active_ny > tile_ny:
        raise ValueError("active tile rectangle exceeds its fixed buffer")
    dt_f32_bits = admitted_dt_f32_bits(
        dt_seconds, gate="gm1-rk3-eqtset2-remove-eos"
    )
    cupy = importlib.import_module("cupy")
    stream = stream or cupy.cuda.get_current_stream()
    stream_pointer = _stream_pointer(stream)
    compiled = compiled or compile_cuda_gm1_rk3_eqtset2_remove_eos()
    if type(compiled) is not CompiledGm1Rk3Eqset2RemoveEos:
        raise TypeError("compiled must be CompiledGm1Rk3Eqset2RemoveEos")
    invalid_words = cupy.zeros((1,), dtype=cupy.uint32)
    active_words = nk * active_ny * active_nx
    blocks = (active_words + CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_THREADS - 1) // (
        CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_THREADS
    )
    with stream:
        compiled.kernel(
            (blocks, 1, 1),
            (CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_THREADS, 1, 1),
            (
                *(arrays[name] for name in _BUFFER_ORDER),
                invalid_words,
                cupy.uint32(tile_nx),
                cupy.uint32(tile_ny),
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(nk),
                cupy.float32(dt_seconds),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    pointers = _pointer_inventory(arrays)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_STATUS,
        "completed_nrk": 3,
        "source_order_phase": "solve2.F:1561-1595",
        "next_source_order_phase": "solve2.F:1616-1656 apmasscon reductions",
        "selection": {
            "runtime.nrk": 3,
            "runtime.nrkmax": 3,
            "runtime.dt_seconds_f32_bits": f"{dt_f32_bits:08x}",
            "dynamics.equation_set": 2,
            "physics.moist": 1,
            "physics.ptype": 5,
            "dynamics.dry_mass_correction": 1,
        },
        "domain": {
            "tile_nx": tile_nx,
            "tile_ny": tile_ny,
            "active_nx": active_nx,
            "active_ny": active_ny,
            "nk": nk,
            "active_words": active_words,
        },
        # solve2.F:145 makes dt a variable, not a constant, so the applied
        # step is reported beside the constants rather than among them.
        "dt_f32_bits": f"{dt_f32_bits:08x}",
        "constants_f32_bits": {
            "p00": f"{CM1_P00_F32_BITS:08x}",
            "rd": f"{CM1_RD_F32_BITS:08x}",
            "rv": f"{CM1_RV_F32_BITS:08x}",
            "cp": f"{CM1_CP_F32_BITS:08x}",
            "cpdrd": f"{CM1_CPDRD_F32_BITS:08x}",
        },
        "statement_order": [
            "pp3d=pp3d-dt*qpten; qpten=pp3d",
            "th3d=th3d-dt*qtten; qtten=th3d",
            "qv=qv-dt*qvten; qvten=qv",
            "qc=qc-dt*qcten; qcten=qc",
            "prs=p00*(pi0+pp3d)**cpdrd",
            "rho=prs/((th0+th3d)*(pi0+pp3d)*(rd+max(0,qv)*rv))",
        ],
        "mutated_fields": list(_MUTATED_ORDER),
        "stream_pointer": stream_pointer,
        "event_pointer": int(event.ptr),
        "invalid_words_pointer": int(invalid_words.data.ptr),
        "pointer_inventory_sha256": _pointer_inventory_sha256(pointers),
        "kernel": _compiled_receipt(compiled),
        "runtime": dict(compiled.runtime_identity),
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "constants_sha256": AUTHORITY_CONSTANTS_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "fresh_rk3_scalar_moist_lineage_bound": False,
        "prior_completed_rk2_lineage_bound": False,
        "native_glibc_output_oracle_bound": False,
        "invalid_counter_host_checked": False,
        "apmasscon_reduction_enqueued": False,
        "momentum_boundary_lineage_bound": False,
        "rk3_tke_lineage_bound": False,
        "post_output_hashes_bound": False,
        "broader_completed_rk3": False,
        "claim_limit": GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Rk3Eqset2RemoveEosLaunch(
        event,
        compiled,
        stream,
        MappingProxyType(dict(arrays)),
        invalid_words,
        Gm1Rk3Eqset2RemoveEosLocalToken(MappingProxyType(receipt)),
    )


def require_gm1_rk3_eqtset2_remove_eos_local_token(
    token: object,
    *,
    stream_pointer: int | None = None,
) -> Mapping[str, object]:
    """Validate a local token while preserving every negative claim."""

    if type(token) is not Gm1Rk3Eqset2RemoveEosLocalToken:
        raise TypeError(
            "token must be the exact Gm1Rk3Eqset2RemoveEosLocalToken type"
        )
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("local RK3 eqtset2 receipt must be a mapping")
    if (
        receipt.get("schema")
        != CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("source_order_phase") != "solve2.F:1561-1595"
        or receipt.get("next_source_order_phase")
        != "solve2.F:1616-1656 apmasscon reductions"
        or receipt.get("mutated_fields") != list(_MUTATED_ORDER)
        or receipt.get("fresh_rk3_scalar_moist_lineage_bound") is not False
        or receipt.get("prior_completed_rk2_lineage_bound") is not False
        or receipt.get("native_glibc_output_oracle_bound") is not False
        or receipt.get("invalid_counter_host_checked") is not False
        or receipt.get("apmasscon_reduction_enqueued") is not False
        or receipt.get("momentum_boundary_lineage_bound") is not False
        or receipt.get("rk3_tke_lineage_bound") is not False
        or receipt.get("post_output_hashes_bound") is not False
        or receipt.get("broader_completed_rk3") is not False
        or receipt.get("claim_limit")
        != GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("local RK3 eqtset2 token identity or claim changed")
    actual_stream = receipt.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("local RK3 eqtset2 stream pointer is invalid")
    if stream_pointer is not None and actual_stream != _positive_int(
        "stream_pointer", stream_pointer
    ):
        raise ValueError("local RK3 eqtset2 token belongs to another stream")
    operation = receipt.get("operation_token_sha256")
    pointer_sha = receipt.get("pointer_inventory_sha256")
    for name, value in (
        ("operation_token_sha256", operation),
        ("pointer_inventory_sha256", pointer_sha),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return MappingProxyType(dict(receipt))


def refuse_gm1_rk3_eqtset2_remove_eos_live_promotion(
    local_token: object,
    *,
    rk3_scalar_moist_token: object | None = None,
) -> NoReturn:
    """Structurally refuse promotion until the exact upstream producer lands."""

    receipt = require_gm1_rk3_eqtset2_remove_eos_local_token(local_token)
    refuse(
        code="GM1-RK3-EQTSET2-LIVE-LINEAGE-UNAVAILABLE",
        gate="gm1-rk3-eqtset2-remove-eos",
        option="rk3_scalar_moist_token",
        value=(
            None
            if rk3_scalar_moist_token is None
            else type(rk3_scalar_moist_token).__name__
        ),
        reason=(
            "the local token explicitly leaves fresh RK3 scalar/moist and prior "
            "completed-RK2 lineage unbound; local operation token "
            + str(receipt["operation_token_sha256"])
            + " cannot be relabelled as a live source-ordered RK3 prefix"
        ),
        remedy=(
            "bind a separately frozen exact RK3 scalar/moist producer and emit a "
            "new live wrapper before beginning apmasscon reductions"
        ),
    )


def cpu_gm1_rk3_eqtset2_remove_eos_inplace(
    arrays: Mapping[str, Any], *, dt_seconds: float
) -> Mapping[str, Any]:
    """Independent binary32 scheduling oracle for the narrow cell operator.

    NumPy's platform power implementation is used only for CPU test
    diagnostics.  It is not the native glibc raw-bit oracle and the returned
    metadata says so explicitly.
    """

    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError("CPU RK3 eqtset2 array inventory/order changed")
    first = arrays["pi0"]
    if (
        not isinstance(first, np.ndarray)
        or first.dtype != np.dtype("float32")
        or first.ndim != 3
        or not first.flags.c_contiguous
    ):
        raise ValueError("CPU RK3 eqtset2 arrays must be C-f32 rank 3")
    pointers: list[int] = []
    for name in _BUFFER_ORDER:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != first.shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"CPU {name} must be writable C-f32 {first.shape}")
        pointers.append(int(value.__array_interface__["data"][0]))
    if len(set(pointers)) != len(pointers):
        raise ValueError("CPU RK3 eqtset2 arrays must not alias")
    admitted_dt_f32_bits(dt_seconds, gate="gm1-rk3-eqtset2-remove-eos-cpu-oracle")
    dt = np.float32(dt_seconds)
    work = np.empty_like(first)
    for state_name, held_name in (
        ("pp3d", "qpten"),
        ("th3d", "qtten"),
        ("qv", "qvten"),
        ("qc", "qcten"),
    ):
        np.multiply(arrays[held_name], dt, out=work)
        np.subtract(arrays[state_name], work, out=arrays[state_name])
        np.copyto(arrays[held_name], arrays[state_name])
    np.add(arrays["pi0"], arrays["pp3d"], out=work)
    np.power(
        work,
        np.asarray(CM1_CPDRD_F32_BITS, dtype=np.uint32).view(np.float32),
        out=arrays["prs"],
    )
    np.multiply(
        arrays["prs"],
        np.asarray(CM1_P00_F32_BITS, dtype=np.uint32).view(np.float32),
        out=arrays["prs"],
    )
    pi = work.copy(order="C")
    np.add(arrays["th0"], arrays["th3d"], out=arrays["rho"])
    np.multiply(arrays["rho"], pi, out=arrays["rho"])
    np.maximum(arrays["qv"], np.float32(0.0), out=work)
    np.multiply(
        work,
        np.asarray(CM1_RV_F32_BITS, dtype=np.uint32).view(np.float32),
        out=work,
    )
    np.add(
        work,
        np.asarray(CM1_RD_F32_BITS, dtype=np.uint32).view(np.float32),
        out=work,
    )
    np.multiply(arrays["rho"], work, out=arrays["rho"])
    np.divide(arrays["prs"], arrays["rho"], out=arrays["rho"])
    return MappingProxyType(
        {
            "statement_order": list(
                require_gm1_rk3_eqtset2_remove_eos_source_audit()[
                    "statement_order"
                ]
            ),
            "native_glibc_raw_bit_oracle": False,
            "finite": all(bool(np.all(np.isfinite(arrays[name]))) for name in _MUTATED_ORDER),
        }
    )


def require_gm1_rk3_eqtset2_remove_eos_source_audit() -> Mapping[str, object]:
    """Return the immutable selected-branch source-order audit."""

    return MappingProxyType(
        {
            "schema": "gm1.rk3-eqtset2-remove-snapshot-eos-source-audit/v1",
            "status": "SOURCE_AUDITED_LOCAL_OPERATOR_LIVE_LINEAGE_FAIL_CLOSED",
            "authority": {
                "release": "CM1 r21.1",
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
                "constants_sha256": AUTHORITY_CONSTANTS_SHA256,
                "anchors": list(AUTHORITY_ANCHORS),
            },
            "selection": {
                "nrk": 3,
                "nrkmax": 3,
                "dt_seconds": CM1_DEFAULT_DT_S,
                "eqtset": 2,
                "imoist": 1,
                "ptype": 5,
                "apmasscon": 1,
                "domain": [75, 300, 300],
                "dx_m": 50.0,
            },
            "statement_order": [
                "pp3d=pp3d-dt*qpten; qpten=pp3d",
                "th3d=th3d-dt*qtten; qtten=th3d",
                "qv=qv-dt*qvten; qvten=qv",
                "qc=qc-dt*qcten; qcten=qc",
                "prs=p00*(pi0+pp3d)**cpdrd",
                "rho=prs/((th0+th3d)*(pi0+pp3d)*(rd+max(0,qv)*rv))",
            ],
            # solve2.F:145 declares dt a default-REAL variable; the kernel
            # takes it as an argument and no timestep is welded into the PTX.
            "dt_seconds_is_kernel_argument": True,
            "dt_seconds_authority": "src/solve2.F:145,1569-1576",
            "default_dt_bit_identity": dict(require_default_dt_bit_identity()),
            "predecessor": "fresh broader RK3 scalar/moist update",
            "successor": "split apmasscon layer reductions",
            "tile_streamable": True,
            "fresh_rk3_scalar_moist_producer_available": False,
            "local_operator_available": True,
            "completed_rk3_emitter_available": False,
            "claim_limit": GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_CLAIM_LIMIT,
        }
    )


__all__ = [
    "AUTHORITY_ANCHORS",
    "AUTHORITY_CONSTANTS_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_CPDRD_F32_BITS",
    "CM1_DEFAULT_DT_S",
    "CM1_CP_F32_BITS",
    "CM1_DT_F32_BITS",
    "CM1_P00_F32_BITS",
    "CM1_RD_F32_BITS",
    "CM1_RV_F32_BITS",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_ABI",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_KERNEL_NAME",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_TOKEN_STATUS",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_ABI",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SEALED_EVIDENCE_CLAIM_LIMIT",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_SOURCE_SHA256",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELDED_PREDECESSOR_SHA256",
    "CUDA_GM1_RK3_EQTSET2_REMOVE_EOS_WELD_EDITS",
    "CompiledGm1Rk3Eqset2RemoveEos",
    "CudaGm1Rk3Eqset2RemoveEosLaunch",
    "Gm1Rk3Eqset2RemoveEosLocalToken",
    "GM1_RK3_EQTSET2_REMOVE_EOS_LOCAL_CLAIM_LIMIT",
    "admitted_dt_f32_bits",
    "require_default_dt_bit_identity",
    "welded_predecessor_source",
    "compile_cuda_gm1_rk3_eqtset2_remove_eos",
    "cpu_gm1_rk3_eqtset2_remove_eos_inplace",
    "launch_cuda_gm1_rk3_eqtset2_remove_eos_local",
    "refuse_gm1_rk3_eqtset2_remove_eos_live_promotion",
    "require_gm1_rk3_eqtset2_remove_eos_local_token",
    "require_gm1_rk3_eqtset2_remove_eos_source_audit",
    "tiled_rk3_eqtset2_remove_eos_buffer_nbytes",
    "tiled_rk3_eqtset2_remove_eos_buffer_shapes",
]
