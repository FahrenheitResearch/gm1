"""CM1 r21.1 psolver=4/5 singleton periodic Poisson solve.

This closes the global pressure seam between the local RHS and projection
operations in :mod:`cm1core.cuda_anelp_periodic` for the admitted flat,
serial, uniform-or-proof-bound ``stretch_z=1``, four-periodic geometry.  CM1's unitary Singleton FFT
normalization, mode indexing, vertical Thomas solve, and singular zero-mode
reconstruction are preserved explicitly.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_anelp_periodic import (
    ALL_ARRAY_NAMES,
    CUDA_ANELP_PERIODIC_SOURCE_SHA256,
    CUDA_ANELP_PERIODIC_THREADS,
    AnelpPeriodicDeviceProof,
    AnelpPeriodicSelection,
    _require_device_arrays,
    _require_host_arrays,
    _validate_vertical_metric_admission,
    compile_cuda_anelp_periodic,
    require_anelp_periodic_selection,
    validate_anelp_periodic_contract,
)
from .cuda_stretch_z1 import StretchZ1MetricProof
from .cuda_sound3_periodic import (
    _canonical_arch,
    _image_options,
    fill_periodic_uv_inplace,
    scalar_shape_zyx,
)


CUDA_ANELP_POISSON_SCHEMA = "gpuwm-cm1.cuda-anelp-poisson-periodic/v1"
CUDA_ANELP_POISSON_ABI = "gpuwm-cm1.cuda-anelp-poisson-periodic-kernels/v1"
CUDA_ANELP_POISSON_THREADS = 128
CM1_ANELP_POISSON_CLAIM_LIMIT = (
    "CM1 r21.1 poiss.F plus singleton.F global periodic pressure solve for "
    "the already admitted flat Cartesian, single-process, uniform or "
    "proof-bound Wilhelmson-Chen stretch_z=1, "
    "four-periodic psolver=4/5 frame, with imirror=jmirror=0, ipe=ni, "
    "jpe=nj, horizontally homogeneous positive mh/mf/rho0/rf0 profiles, "
    "source-built psolver>3 cfa/cfb/cfc/d2 coefficients, unitary 2-D FFTs, "
    "vertical spectral Thomas solve, and exact zero-mode gauge recurrence; "
    "no mirrored open/rigid transform, terrain, MPI/distributed FFT, "
    "axisymmetry, stretch_z=2/3/4, immersed boundary, forcing, complete "
    "solve2, coupled trajectory, forecast, product, tornado, or performance claim"
)
SOURCE_ANCHORS = {
    "periodic_transform_shape": "src/param.F:5178-5205",
    "coefficient_build": "src/init3d.F:1693-1720",
    "rhs_and_forward_fft": "src/poiss.F:51-117",
    "forward_thomas": "src/poiss.F:118-159",
    "backward_thomas": "src/poiss.F:161-174",
    "zero_mode_reconstruction": "src/poiss.F:176-185",
    "inverse_fft_and_d2": "src/poiss.F:187-213",
    "singleton_scaling": "src/singleton.F:74-104,330-373",
    "anelp_composition": "src/anelp.F:237-280",
}


@dataclass(frozen=True, slots=True)
class AnelpPeriodicPoissonCoefficients:
    cfa: Any
    cfc: Any
    d2: Any
    cfb: Any
    dx_f32: float
    dz_f32: float
    stretch_z: int
    vertical_grid_proof_sha256: str | None
    sha256: str


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")


def build_anelp_periodic_poisson_coefficients(
    *,
    mh: Any,
    mf: Any,
    rho0: Any,
    rf0: Any,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dz: float,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    vertical_grid_proof: StretchZ1MetricProof | None = None,
) -> AnelpPeriodicPoissonCoefficients:
    """Build exact psolver>3 coefficient arrays from 1-D base profiles."""

    require_anelp_periodic_selection(selection)
    _validate_dimensions(ni, nj, nk)
    np = importlib.import_module("numpy")
    profiles = {"mh": mh, "mf": mf, "rho0": rho0, "rf0": rf0}
    for name, value in profiles.items():
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} profile must be a float32 numpy.ndarray")
        if value.shape != (nk + 2,) or not value.flags.c_contiguous:
            raise ValueError(f"{name} profile must have shape (nk+2,) and C layout")
        if not bool(np.isfinite(value).all()) or bool(np.any(value <= np.float32(0.0))):
            raise ValueError(f"{name} profile must be finite and positive")
    _validate_vertical_metric_admission(
        selection,
        vertical_grid_proof,
        mh=mh,
        mf=mf,
        nk=nk,
    )
    if not math.isfinite(dx) or dx <= 0.0 or not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("dx and dz must be finite and positive")
    dx32, dz32 = np.float32(dx), np.float32(dz)
    dz2 = _mul(np, dz32, dz32)
    cfa = np.zeros(nk, dtype=np.float32)
    cfc = np.zeros(nk, dtype=np.float32)
    d2 = np.empty(nk, dtype=np.float32)
    upper = np.empty(nk, dtype=np.float32)
    lower = np.empty(nk, dtype=np.float32)
    for k0 in range(nk):
        k = k0 + 1
        numerator = _mul(np, _mul(np, mh[k], mf[k]), rf0[k])
        cfa[k0] = _div(np, numerator, _mul(np, dz2, rho0[k - 1]))
        numerator = _mul(np, _mul(np, mh[k], mf[k + 1]), rf0[k + 1])
        cfc[k0] = _div(np, numerator, _mul(np, dz2, rho0[k + 1]))
        d2[k0] = _div(np, np.float32(1.0), rho0[k])
        upper[k0] = _div(np, numerator, _mul(np, dz2, rho0[k]))
        numerator = _mul(np, _mul(np, mh[k], mf[k]), rf0[k])
        lower[k0] = _div(np, numerator, _mul(np, dz2, rho0[k]))
    cfa[0] = np.float32(0.0)
    cfc[-1] = np.float32(0.0)
    hmode = np.empty((nj, ni), dtype=np.float64)
    dx64 = float(dx32)
    for j0 in range(nj):
        for i0 in range(ni):
            hmode[j0, i0] = 2.0 * (
                math.cos(2.0 * math.pi * float(i0) / float(ni))
                + math.cos(2.0 * math.pi * float(j0) / float(nj))
                - 2.0
            ) / (dx64 * dx64)
    cfb = np.empty((nk, nj, ni), dtype=np.float32)
    for k0 in range(nk):
        if k0 == 0:
            value = hmode - float(upper[k0])
        elif k0 == nk - 1:
            value = hmode - float(lower[k0])
        else:
            value = hmode - float(upper[k0]) - float(lower[k0])
        cfb[k0] = value.astype(np.float32)
    digest = hashlib.sha256()
    for value in (cfa, cfc, d2, cfb):
        digest.update(value.astype("<f4", copy=False).tobytes(order="C"))
    digest.update(dx32.astype("<f4").tobytes())
    digest.update(dz32.astype("<f4").tobytes())
    if vertical_grid_proof is not None:
        digest.update(b"stretch-z1-proof\0")
        digest.update(vertical_grid_proof.proof_sha256.encode("ascii"))
    return AnelpPeriodicPoissonCoefficients(
        cfa,
        cfc,
        d2,
        cfb,
        float(dx32),
        float(dz32),
        selection.stretch_z,
        None if vertical_grid_proof is None else vertical_grid_proof.proof_sha256,
        digest.hexdigest(),
    )


def _validate_host_coefficients(
    coefficients: AnelpPeriodicPoissonCoefficients, ni: int, nj: int, nk: int
) -> None:
    np = importlib.import_module("numpy")
    expected = {"cfa": (nk,), "cfc": (nk,), "d2": (nk,), "cfb": (nk, nj, ni)}
    for name, shape in expected.items():
        value = getattr(coefficients, name)
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be float32 numpy")
        if value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid shape or layout")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if coefficients.cfa.view(np.uint32)[0] != np.uint32(0):
        raise ValueError("cfa(1) must be positive zero")
    if coefficients.cfc.view(np.uint32)[-1] != np.uint32(0):
        raise ValueError("cfc(nk) must be positive zero")
    if coefficients.stretch_z not in (0, 1):
        raise ValueError("coefficient stretch_z identity is invalid")
    if (coefficients.stretch_z == 1) != (
        coefficients.vertical_grid_proof_sha256 is not None
    ):
        raise ValueError("coefficient vertical-grid proof identity is inconsistent")


def _validate_coefficient_selection(
    coefficients: Any,
    selection: AnelpPeriodicSelection,
) -> None:
    if coefficients.stretch_z != selection.stretch_z:
        raise ValueError("coefficient vertical-grid selector differs from live selection")


def _validate_global_spacing(
    coefficients: Any,
    *,
    rdx: float,
    rdy: float,
    rdz: float,
) -> None:
    np = importlib.import_module("numpy")
    supplied = tuple(np.float32(value) for value in (rdx, rdy, rdz))
    if not all(bool(np.isfinite(value)) and value > np.float32(0.0) for value in supplied):
        raise ValueError("rdx, rdy, and rdz must be finite and positive")
    expected_x = _div(np, np.float32(1.0), np.float32(coefficients.dx_f32))
    expected_z = _div(np, np.float32(1.0), np.float32(coefficients.dz_f32))
    words = tuple(int(np.asarray(value, dtype=np.float32).view(np.uint32)) for value in supplied)
    expected = (
        int(np.asarray(expected_x, dtype=np.float32).view(np.uint32)),
        int(np.asarray(expected_x, dtype=np.float32).view(np.uint32)),
        int(np.asarray(expected_z, dtype=np.float32).view(np.uint32)),
    )
    if words != expected:
        raise ValueError(
            "the admitted Singleton coefficient field requires raw-f32 "
            "rdx=rdy=1/dx and rdz=1/dz"
        )


@dataclass(frozen=True, slots=True)
class SpectralSolveResult:
    pdt: Any
    lgbth: Any
    lgbph: Any


def cpu_anelp_spectral_vertical_solve(
    trans: Any,
    coefficients: AnelpPeriodicPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> SpectralSolveResult:
    """Exact float32 source-order vertical solve on supplied Fourier modes."""

    np = importlib.import_module("numpy")
    _validate_host_coefficients(coefficients, ni, nj, nk)
    if not isinstance(trans, np.ndarray) or trans.dtype != np.dtype("complex64"):
        raise TypeError("trans must be a complex64 numpy.ndarray")
    if trans.shape != (nk, nj, ni) or not trans.flags.c_contiguous:
        raise ValueError("trans has invalid shape or layout")
    if not bool(np.isfinite(trans.real).all() and np.isfinite(trans.imag).all()):
        raise ValueError("trans contains non-finite values")
    cfa, cfc, cfb = coefficients.cfa, coefficients.cfc, coefficients.cfb
    th = np.zeros((nk, nj, ni), dtype=np.float32)
    ph = np.zeros((nk, nj, ni), dtype=np.complex64)
    pdt = np.zeros((nk, nj, ni), dtype=np.complex64)
    for j0 in range(nj):
        for i0 in range(ni):
            if i0 == 0 and j0 == 0:
                continue
            tem = _div(np, np.float32(1.0), cfb[0, j0, i0])
            th[0, j0, i0] = np.float32(-_mul(np, cfc[0], tem))
            ph[0, j0, i0] = complex(
                _mul(np, trans[0, j0, i0].real, tem),
                _mul(np, trans[0, j0, i0].imag, tem),
            )
            for k0 in range(1, nk):
                denominator = _add(np, _mul(np, cfa[k0], th[k0 - 1, j0, i0]), cfb[k0, j0, i0])
                tem = _div(np, np.float32(1.0), denominator)
                th[k0, j0, i0] = np.float32(-_mul(np, cfc[k0], tem))
                ph[k0, j0, i0] = complex(
                    _mul(np, _sub(np, trans[k0, j0, i0].real, _mul(np, cfa[k0], ph[k0 - 1, j0, i0].real)), tem),
                    _mul(np, _sub(np, trans[k0, j0, i0].imag, _mul(np, cfa[k0], ph[k0 - 1, j0, i0].imag)), tem),
                )
            pdt[-1, j0, i0] = ph[-1, j0, i0]
            for k0 in range(nk - 2, -1, -1):
                pdt[k0, j0, i0] = complex(
                    _add(np, _mul(np, th[k0, j0, i0], pdt[k0 + 1, j0, i0].real), ph[k0, j0, i0].real),
                    _add(np, _mul(np, th[k0, j0, i0], pdt[k0 + 1, j0, i0].imag), ph[k0, j0, i0].imag),
                )
    # poiss.F:176-185 replaces the singular (1,1) Fourier mode with a fixed
    # bottom gauge followed by the exact upward recurrence.
    pdt[0, 0, 0] = np.complex64(0.0 + 0.0j)
    rkm1 = np.float32(0.0)
    rk = _div(np, _sub(np, trans[0, 0, 0].real, _mul(np, cfb[0, 0, 0], rkm1)), cfc[0])
    pdt[1, 0, 0] = np.complex64(complex(rk, 0.0))
    for k0 in range(1, nk - 1):
        numerator = _sub(np, trans[k0, 0, 0].real, _mul(np, cfa[k0], rkm1))
        numerator = _sub(np, numerator, _mul(np, cfb[k0, 0, 0], rk))
        rnext = _div(np, numerator, cfc[k0])
        pdt[k0 + 1, 0, 0] = np.complex64(complex(rnext, 0.0))
        rkm1, rk = rk, rnext
    return SpectralSolveResult(pdt, th.astype(np.complex64), ph)


def _singleton_scale_f32(np: Any, ni: int, nj: int) -> Any:
    return np.float32(np.sqrt(np.float32(np.float32(1.0) / np.float32(ni * nj)), dtype=np.float32))


@dataclass(frozen=True, slots=True)
class CpuAnelpPoissonDiagnostics:
    fft_scale_f32: float
    coefficient_sha256: str
    spectral_relative_linf_residual: float
    zero_mode_bottom_gauge_bits: str


def spectral_relative_linf_residual(
    trans: Any,
    pdt: Any,
    coefficients: AnelpPeriodicPoissonCoefficients,
) -> float:
    np = importlib.import_module("numpy")
    lhs = coefficients.cfb.astype(np.complex64) * pdt
    lhs[1:] += coefficients.cfa[1:, None, None].astype(np.complex64) * pdt[:-1]
    lhs[:-1] += coefficients.cfc[:-1, None, None].astype(np.complex64) * pdt[1:]
    residual = np.abs(lhs - trans)
    scale = max(float(np.max(np.abs(trans))), float(np.finfo(np.float32).tiny))
    return float(np.max(residual)) / scale


def cpu_anelp_periodic_poisson_inplace(
    rhs_real: Any,
    phi2: Any,
    coefficients: AnelpPeriodicPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
) -> CpuAnelpPoissonDiagnostics:
    require_anelp_periodic_selection(selection)
    np = importlib.import_module("numpy")
    _validate_host_coefficients(coefficients, ni, nj, nk)
    _validate_coefficient_selection(coefficients, selection)
    scalar = scalar_shape_zyx(ni, nj, nk)
    for name, value in (("rhs_real", rhs_real), ("phi2", phi2)):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be float32 numpy")
        if value.shape != scalar or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid shape or layout")
    scale = _singleton_scale_f32(np, ni, nj)
    live = np.ascontiguousarray(rhs_real[1 : nk + 1, 1 : nj + 1, 1 : ni + 1])
    scaled = np.empty(live.shape, dtype=np.complex64)
    scaled.real = np.multiply(live, scale, dtype=np.float32)
    scaled.imag.fill(np.float32(0.0))
    trans = np.fft.fft2(scaled, axes=(-2, -1), norm="backward").astype(np.complex64)
    solved = cpu_anelp_spectral_vertical_solve(trans, coefficients, ni=ni, nj=nj, nk=nk)
    inverse_input = np.multiply(solved.pdt, np.complex64(scale), dtype=np.complex64)
    physical = np.fft.ifft2(inverse_input, axes=(-2, -1), norm="forward").astype(np.complex64)
    for k0 in range(nk):
        phi2[k0 + 1, 1 : nj + 1, 1 : ni + 1] = np.multiply(
            physical[k0].real, coefficients.d2[k0], dtype=np.float32
        )
    from .cuda_sound3_periodic import fill_periodic_scalar_inplace

    fill_periodic_scalar_inplace(phi2, ni=ni, nj=nj, nk=nk)
    return CpuAnelpPoissonDiagnostics(
        float(scale),
        coefficients.sha256,
        spectral_relative_linf_residual(trans, solved.pdt, coefficients),
        # The fixed gauge is imposed on pdt(1,1,1), not on the physical
        # pressure at grid cell (1,1,1) after the inverse transform.
        "00000000",
    )


CUDA_ANELP_POISSON_SOURCE = r"""// gpuwm-cm1.cuda-anelp-poisson-periodic-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long si(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}

extern "C" __global__ void cm1_anelp_spectral_solve_v1(const float2* trans,const float* cfa,const float* cfc,const float* cfb,float2* pdt,float2* lgbth,float2* lgbph,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj;if(q>=n)return;if(q==0u){for(unsigned k=0u;k<nk;++k){unsigned long long x=(unsigned long long)k*n;lgbth[x]=make_float2(0.0f,0.0f);lgbph[x]=make_float2(0.0f,0.0f);pdt[x]=make_float2(0.0f,0.0f);}float rkm1=0.0f;float rk=d32(s32(trans[0].x,m32(cfb[0],rkm1)),cfc[0]);pdt[n]=make_float2(rk,0.0f);for(unsigned k=1u;k+1u<nk;++k){unsigned long long x=(unsigned long long)k*n;float num=s32(trans[x].x,m32(cfa[k],rkm1));num=s32(num,m32(cfb[x],rk));float rn=d32(num,cfc[k]);pdt[x+n]=make_float2(rn,0.0f);rkm1=rk;rk=rn;}return;}float tem=d32(1.0f,cfb[q]);lgbth[q]=make_float2(-m32(cfc[0],tem),0.0f);lgbph[q]=make_float2(m32(trans[q].x,tem),m32(trans[q].y,tem));for(unsigned k=1u;k<nk;++k){unsigned long long x=(unsigned long long)k*n+q,p=x-n;float den=a32(m32(cfa[k],lgbth[p].x),cfb[x]);tem=d32(1.0f,den);lgbth[x]=make_float2(-m32(cfc[k],tem),0.0f);lgbph[x]=make_float2(m32(s32(trans[x].x,m32(cfa[k],lgbph[p].x)),tem),m32(s32(trans[x].y,m32(cfa[k],lgbph[p].y)),tem));}unsigned long long x=(unsigned long long)(nk-1u)*n+q;pdt[x]=lgbph[x];for(int k=(int)nk-2;k>=0;--k){x=(unsigned long long)k*n+q;unsigned long long u=x+n;pdt[x]=make_float2(a32(m32(lgbth[x].x,pdt[u].x),lgbph[x].x),a32(m32(lgbth[x].x,pdt[u].y),lgbph[x].y));}}

extern "C" __global__ void cm1_anelp_poisson_store_v1(const float2* physical,const float* d2,float* phi2,unsigned ni,unsigned nj,unsigned nk){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;for(unsigned long long q=first;q<n;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned j=z%nj+1u,k=z/nj+1u;phi2[si(k,j,i,ni,nj)]=m32(physical[q].x,d2[k-1u]);}}

extern "C" __global__ void cm1_anelp_poisson_bc_v1(float* phi2,unsigned ni,unsigned nj,unsigned nk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk;++k){for(unsigned j=0u;j<=nj+1u;++j){phi2[si(k,j,0u,ni,nj)]=phi2[si(k,j,ni,ni,nj)];phi2[si(k,j,ni+1u,ni,nj)]=phi2[si(k,j,1u,ni,nj)];}for(unsigned i=0u;i<=ni+1u;++i){phi2[si(k,0u,i,ni,nj)]=phi2[si(k,nj,i,ni,nj)];phi2[si(k,nj+1u,i,ni,nj)]=phi2[si(k,1u,i,ni,nj)];}}}
"""
CUDA_ANELP_POISSON_SOURCE_BYTES = CUDA_ANELP_POISSON_SOURCE.encode("utf-8")
CUDA_ANELP_POISSON_SOURCE_SHA256 = (
    "2696f5431d153ba3191666d7c08e6ace294719695ac61e5af85208d0ca8296d4"
)
CUDA_ANELP_POISSON_KERNEL_NAMES = (
    "cm1_anelp_spectral_solve_v1",
    "cm1_anelp_poisson_store_v1",
    "cm1_anelp_poisson_bc_v1",
)


@dataclass(frozen=True, slots=True)
class CompiledAnelpPoissonKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_anelp_poisson() -> CompiledAnelpPoissonKernels:
    actual = hashlib.sha256(CUDA_ANELP_POISSON_SOURCE_BYTES).hexdigest()
    if actual != CUDA_ANELP_POISSON_SOURCE_SHA256:
        raise RuntimeError(f"packaged anelp Poisson CUDA source differs from its pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_ANELP_POISSON_SOURCE, "cm1_anelp_poisson_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_ANELP_POISSON_SOURCE, "cm1_anelp_poisson_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_ANELP_POISSON_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks anelp Poisson entry points")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_ANELP_POISSON_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledAnelpPoissonKernels(module, kernels, ptx, cubin, arch, identity)


@dataclass(frozen=True, slots=True)
class DeviceAnelpPoissonCoefficients:
    cfa: Any
    cfc: Any
    d2: Any
    cfb: Any
    host_sha256: str
    dx_f32: float
    dz_f32: float
    stretch_z: int
    vertical_grid_proof_sha256: str | None


def upload_anelp_periodic_poisson_coefficients(
    coefficients: AnelpPeriodicPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> DeviceAnelpPoissonCoefficients:
    _validate_host_coefficients(coefficients, ni, nj, nk)
    cupy = importlib.import_module("cupy")
    return DeviceAnelpPoissonCoefficients(
        cupy.asarray(coefficients.cfa),
        cupy.asarray(coefficients.cfc),
        cupy.asarray(coefficients.d2),
        cupy.asarray(coefficients.cfb),
        coefficients.sha256,
        coefficients.dx_f32,
        coefficients.dz_f32,
        coefficients.stretch_z,
        coefficients.vertical_grid_proof_sha256,
    )


def _require_device_coefficients(cupy: Any, coefficients: DeviceAnelpPoissonCoefficients, ni: int, nj: int, nk: int) -> None:
    for name, shape in (("cfa", (nk,)), ("cfc", (nk,)), ("d2", (nk,)), ("cfb", (nk, nj, ni))):
        value = getattr(coefficients, name)
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32"):
            raise TypeError(f"device {name} must be float32 cupy")
        if value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"device {name} has invalid shape/layout")
    if coefficients.stretch_z not in (0, 1):
        raise ValueError("device coefficient stretch_z identity is invalid")
    if (coefficients.stretch_z == 1) != (
        coefficients.vertical_grid_proof_sha256 is not None
    ):
        raise ValueError("device coefficient vertical-grid proof is inconsistent")


@dataclass(frozen=True, slots=True)
class CudaAnelpSpectralLaunch:
    ready_event: Any
    compiled: CompiledAnelpPoissonKernels
    pdt: Any
    lgbth: Any
    lgbph: Any


def launch_cuda_anelp_spectral_vertical_solve(
    trans: Any,
    coefficients: DeviceAnelpPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any | None = None,
    compiled: CompiledAnelpPoissonKernels | None = None,
) -> CudaAnelpSpectralLaunch:
    cupy = importlib.import_module("cupy")
    _require_device_coefficients(cupy, coefficients, ni, nj, nk)
    if not isinstance(trans, cupy.ndarray) or trans.dtype != cupy.dtype("complex64"):
        raise TypeError("trans must be complex64 cupy")
    if trans.shape != (nk, nj, ni) or not trans.flags.c_contiguous:
        raise ValueError("trans has invalid shape/layout")
    compiled = compile_cuda_anelp_poisson() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        pdt = cupy.empty_like(trans)
        lgbth = cupy.empty_like(trans)
        lgbph = cupy.empty_like(trans)
        solve = compiled.kernels[0]
        modes = ni * nj
        block = (CUDA_ANELP_POISSON_THREADS, 1, 1)
        grid = ((modes + CUDA_ANELP_POISSON_THREADS - 1) // CUDA_ANELP_POISSON_THREADS, 1, 1)
        solve(grid, block, (trans, coefficients.cfa, coefficients.cfc, coefficients.cfb, pdt, lgbth, lgbph, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaAnelpSpectralLaunch(event, compiled, pdt, lgbth, lgbph)


@dataclass(frozen=True, slots=True)
class CudaAnelpPoissonLaunch:
    ready_event: Any
    compiled: CompiledAnelpPoissonKernels
    trans: Any
    pdt: Any
    receipt: dict[str, Any]


def launch_cuda_anelp_periodic_poisson(
    rhs_real: Any,
    phi2: Any,
    coefficients: DeviceAnelpPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledAnelpPoissonKernels | None = None,
) -> CudaAnelpPoissonLaunch:
    require_anelp_periodic_selection(selection)
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    _require_device_coefficients(cupy, coefficients, ni, nj, nk)
    _validate_coefficient_selection(coefficients, selection)
    scalar = scalar_shape_zyx(ni, nj, nk)
    for name, value in (("rhs_real", rhs_real), ("phi2", phi2)):
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32"):
            raise TypeError(f"{name} must be float32 cupy")
        if value.shape != scalar or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid shape/layout")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    scale = _singleton_scale_f32(np, ni, nj)
    with stream:
        live = cupy.ascontiguousarray(rhs_real[1 : nk + 1, 1 : nj + 1, 1 : ni + 1])
        scaled = live.astype(cupy.complex64) * cupy.complex64(scale)
        trans = cupy.fft.fft2(scaled, axes=(-2, -1), norm="backward").astype(cupy.complex64)
        spectral = launch_cuda_anelp_spectral_vertical_solve(
            trans,
            coefficients,
            ni=ni,
            nj=nj,
            nk=nk,
            stream=stream,
            compiled=compiled,
        )
        inverse_input = spectral.pdt * cupy.complex64(scale)
        physical = cupy.fft.ifft2(inverse_input, axes=(-2, -1), norm="forward").astype(cupy.complex64)
        store, bc = spectral.compiled.kernels[1:]
        words = ni * nj * nk
        block = (CUDA_ANELP_POISSON_THREADS, 1, 1)
        grid = ((words + CUDA_ANELP_POISSON_THREADS - 1) // CUDA_ANELP_POISSON_THREADS, 1, 1)
        store(grid, block, (physical, coefficients.d2, phi2, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)), stream=stream)
        # Separate launch is a required grid-wide ordering point before halo
        # reads; a block-local barrier cannot order arbitrary store blocks.
        bc((1, 1, 1), (1, 1, 1), (phi2, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_ANELP_POISSON_SCHEMA,
        "abi": CUDA_ANELP_POISSON_ABI,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "dimensions": {"ni": ni, "nj": nj, "nk": nk, "ipe": ni, "jpe": nj, "imirror": 0, "jmirror": 0},
        "fft": {"implementation": "cuFFT via cupy.fft.fft2/ifft2", "forward_sign": "negative", "inverse_sign": "positive", "scale_each_direction_f32": float(scale), "normalization": "CM1 singleton unitary pre-scale sqrt(1/(ni*nj))"},
        "coefficient_host_sha256": coefficients.host_sha256,
        "vertical_grid_proof_sha256": coefficients.vertical_grid_proof_sha256,
        "spacing": {
            "dx_f32": coefficients.dx_f32,
            "dz_f32": coefficients.dz_f32,
        },
        "kernel": {"source_sha256": CUDA_ANELP_POISSON_SOURCE_SHA256, "names": list(CUDA_ANELP_POISSON_KERNEL_NAMES), "target_arch": spectral.compiled.target_arch},
        "runtime": dict(spectral.compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_ANELP_POISSON_CLAIM_LIMIT,
    }
    return CudaAnelpPoissonLaunch(event, spectral.compiled, trans, spectral.pdt, receipt)


@dataclass(frozen=True, slots=True)
class CpuAnelpGlobalFrameDiagnostics:
    poisson: CpuAnelpPoissonDiagnostics
    rhs_cells: int
    projected_u_faces: int
    projected_v_faces: int
    projected_w_faces: int
    phi1_written: bool
    spectral_solve_status: str


def cpu_anelp_periodic_global_frame_inplace(
    arrays: MutableMapping[str, Any],
    coefficients: AnelpPeriodicPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    vertical_grid_proof: StretchZ1MetricProof | None = None,
) -> CpuAnelpGlobalFrameDiagnostics:
    """Execute the admitted local RHS, global solve, and projection frame."""

    _validate_global_spacing(coefficients, rdx=rdx, rdy=rdy, rdz=rdz)
    _validate_coefficient_selection(coefficients, selection)
    expected_proof_sha256 = (
        None if vertical_grid_proof is None else vertical_grid_proof.proof_sha256
    )
    if coefficients.vertical_grid_proof_sha256 != expected_proof_sha256:
        raise ValueError("coefficient proof differs from the live vertical-grid proof")
    validate_anelp_periodic_contract(
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=selection,
        vertical_grid_proof=vertical_grid_proof,
    )
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if not math.isfinite(dttmp) or dttmp <= 0.0:
        raise ValueError("dttmp must be finite and positive")
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    dttmp32, rdx32, rdy32, rdz32 = map(np.float32, (dttmp, rdx, rdy, rdz))
    rdt = _div(np, np.float32(1.0), dttmp32)
    a["def"].fill(np.float32(0.0))
    a["rhs_real"].fill(np.float32(0.0))
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                du = _mul(
                    np,
                    _mul(np, _sub(np, a["uten"][k, j, i], a["uten"][k, j, i - 1]), rdx32),
                    a["uh"][i],
                )
                dv = _mul(
                    np,
                    _mul(np, _sub(np, a["vten"][k, j, i], a["vten"][k, j - 1, i]), rdy32),
                    a["vh"][j],
                )
                horizontal = _mul(np, a["rho0"][k, 1, 1], _add(np, du, dv))
                vertical = _sub(
                    np,
                    _mul(np, a["rf0"][k + 1, j, i], a["wten"][k + 1, j0, i0]),
                    _mul(np, a["rf0"][k, j, i], a["wten"][k, j0, i0]),
                )
                vertical = _mul(np, _mul(np, vertical, rdz32), a["mh"][k, j, i])
                value = _add(
                    np,
                    _add(np, horizontal, vertical),
                    _mul(np, a["divx"][k, j, i], rdt),
                )
                a["def"][k, j, i] = value
                a["rhs_real"][k, j, i] = _mul(np, value, a["d1"][k])
    poisson = cpu_anelp_periodic_poisson_inplace(
        arrays["rhs_real"],
        arrays["phi2"],
        coefficients,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=selection,
    )
    temx, temy, temz = (
        _mul(np, dttmp32, rdx32),
        _mul(np, dttmp32, rdy32),
        _mul(np, dttmp32, rdz32),
    )
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for f in range(ni + 1):
                provisional = _add(
                    np, a["ua"][k, j, f], _mul(np, dttmp32, a["uten"][k, j, f])
                )
                pressure = _mul(
                    np,
                    _mul(np, temx, _sub(np, a["phi2"][k, j, f + 1], a["phi2"][k, j, f])),
                    a["uf"][f + 1],
                )
                a["u3d"][k, j, f] = _sub(np, provisional, pressure)
        for f in range(nj + 1):
            for i in range(1, ni + 1):
                provisional = _add(
                    np, a["va"][k, f, i], _mul(np, dttmp32, a["vten"][k, f, i])
                )
                pressure = _mul(
                    np,
                    _mul(np, temy, _sub(np, a["phi2"][k, f + 1, i], a["phi2"][k, f, i])),
                    a["vf"][f + 1],
                )
                a["v3d"][k, f, i] = _sub(np, provisional, pressure)
    for k in range(2, nk + 1):
        for j0 in range(nj):
            for i0 in range(ni):
                provisional = _add(
                    np, a["wa"][k, j0, i0], _mul(np, dttmp32, a["wten"][k, j0, i0])
                )
                pressure = _mul(
                    np,
                    _mul(
                        np,
                        temz,
                        _sub(np, a["phi2"][k, j0 + 1, i0 + 1], a["phi2"][k - 1, j0 + 1, i0 + 1]),
                    ),
                    a["mf"][k, j0 + 1, i0 + 1],
                )
                a["w3d"][k, j0, i0] = _sub(np, provisional, pressure)
    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    a["w3d"][[1, nk + 1]] = np.float32(0.0)
    if nrk == 3:
        a["phi1"][1 : nk + 1] = a["phi2"][1 : nk + 1]
    return CpuAnelpGlobalFrameDiagnostics(
        poisson,
        ni * nj * nk,
        (ni + 1) * nj * nk,
        ni * (nj + 1) * nk,
        ni * nj * (nk - 1),
        nrk == 3,
        "completed-singleton-periodic-global-solve-and-projection",
    )


@dataclass(frozen=True, slots=True)
class CudaAnelpGlobalFrameLaunch:
    ready_event: Any
    compiled_local: Any
    poisson: CudaAnelpPoissonLaunch
    receipt: dict[str, Any]


def launch_cuda_anelp_periodic_global_frame(
    arrays: Mapping[str, Any],
    coefficients: DeviceAnelpPoissonCoefficients,
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    device_proof: AnelpPeriodicDeviceProof,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    stream: Any | None = None,
    compiled_local: Any | None = None,
    compiled_poisson: CompiledAnelpPoissonKernels | None = None,
) -> CudaAnelpGlobalFrameLaunch:
    """Launch RHS, global periodic Poisson solve, then live projection."""

    _validate_global_spacing(coefficients, rdx=rdx, rdy=rdy, rdz=rdz)
    require_anelp_periodic_selection(selection)
    _validate_coefficient_selection(coefficients, selection)
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if not math.isfinite(dttmp) or dttmp <= 0.0:
        raise ValueError("dttmp must be finite and positive")
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    expected = tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES)
    if (
        (device_proof.ni, device_proof.nj, device_proof.nk, device_proof.device_id)
        != (ni, nj, nk, int(cupy.cuda.Device().id))
        or device_proof.selection != selection
        or device_proof.vertical_grid_proof_sha256
        != coefficients.vertical_grid_proof_sha256
        or device_proof.pointers != expected
    ):
        raise ValueError("anelp device proof does not match these allocations")
    compiled_local = (
        compile_cuda_anelp_periodic() if compiled_local is None else compiled_local
    )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    clear, rhs, project, bc = compiled_local.kernels
    f32, u32, u64 = cupy.float32, cupy.uint32, cupy.uint64
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    maximum = max(
        scalar_words,
        ni * nj * nk,
        nk * nj * (ni + 1),
        nk * (nj + 1) * ni,
        (nk - 1) * nj * ni,
    )
    block = (CUDA_ANELP_PERIODIC_THREADS, 1, 1)
    grid = (
        (maximum + CUDA_ANELP_PERIODIC_THREADS - 1)
        // CUDA_ANELP_PERIODIC_THREADS,
        1,
        1,
    )
    dttmp32, rdx32, rdy32, rdz32 = map(np.float32, (dttmp, rdx, rdy, rdz))
    rdt32 = _div(np, np.float32(1.0), dttmp32)
    with stream:
        clear(grid, block, (a["def"], a["rhs_real"], u64(scalar_words)), stream=stream)
        rhs(
            grid,
            block,
            (
                a["mh"],
                a["rho0"],
                a["rf0"],
                a["divx"],
                a["d1"],
                a["uh"],
                a["vh"],
                a["uten"],
                a["vten"],
                a["wten"],
                a["def"],
                a["rhs_real"],
                u32(ni),
                u32(nj),
                u32(nk),
                f32(rdx32),
                f32(rdy32),
                f32(rdz32),
                f32(rdt32),
            ),
            stream=stream,
        )
        poisson = launch_cuda_anelp_periodic_poisson(
            a["rhs_real"],
            a["phi2"],
            coefficients,
            ni=ni,
            nj=nj,
            nk=nk,
            selection=selection,
            stream=stream,
            compiled=compiled_poisson,
        )
        project(
            grid,
            block,
            (
                a["phi2"],
                a["mf"],
                a["uf"],
                a["vf"],
                a["ua"],
                a["uten"],
                a["va"],
                a["vten"],
                a["wa"],
                a["wten"],
                a["u3d"],
                a["v3d"],
                a["w3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                f32(dttmp32),
                f32(_mul(np, dttmp32, rdx32)),
                f32(_mul(np, dttmp32, rdy32)),
                f32(_mul(np, dttmp32, rdz32)),
            ),
            stream=stream,
        )
        bc(
            (1, 1, 1),
            (1, 1, 1),
            (
                a["phi2"],
                a["phi1"],
                a["u3d"],
                a["v3d"],
                a["w3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(nrk),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": "gpuwm-cm1.cuda-anelp-periodic-global-frame/v1",
        "cadence": {
            "sequence": ["local-rhs", "global-poisson", "projection/final-closure"],
            "nrk": nrk,
            "phi1_written": nrk == 3,
            "spectral_solve_status": "completed-singleton-periodic-global-solve-and-projection",
        },
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "device_contract_proof_sha256": device_proof.contract_sha256,
        "vertical_grid_proof_sha256": device_proof.vertical_grid_proof_sha256,
        "local_kernels": {
            "source_sha256": CUDA_ANELP_PERIODIC_SOURCE_SHA256,
            "target_arch": compiled_local.target_arch,
            "runtime": dict(compiled_local.runtime_identity),
        },
        "global_poisson": poisson.receipt,
        "claim_limit": CM1_ANELP_POISSON_CLAIM_LIMIT,
    }
    return CudaAnelpGlobalFrameLaunch(event, compiled_local, poisson, receipt)


__all__ = (
    "AnelpPeriodicPoissonCoefficients",
    "CM1_ANELP_POISSON_CLAIM_LIMIT",
    "CUDA_ANELP_POISSON_ABI",
    "CUDA_ANELP_POISSON_SCHEMA",
    "CUDA_ANELP_POISSON_SOURCE_BYTES",
    "CUDA_ANELP_POISSON_SOURCE_SHA256",
    "CompiledAnelpPoissonKernels",
    "CpuAnelpGlobalFrameDiagnostics",
    "CpuAnelpPoissonDiagnostics",
    "CudaAnelpPoissonLaunch",
    "CudaAnelpGlobalFrameLaunch",
    "CudaAnelpSpectralLaunch",
    "DeviceAnelpPoissonCoefficients",
    "SOURCE_ANCHORS",
    "SpectralSolveResult",
    "build_anelp_periodic_poisson_coefficients",
    "compile_cuda_anelp_poisson",
    "cpu_anelp_periodic_poisson_inplace",
    "cpu_anelp_periodic_global_frame_inplace",
    "cpu_anelp_spectral_vertical_solve",
    "launch_cuda_anelp_periodic_poisson",
    "launch_cuda_anelp_periodic_global_frame",
    "launch_cuda_anelp_spectral_vertical_solve",
    "spectral_relative_linf_residual",
    "upload_anelp_periodic_poisson_coefficients",
)
