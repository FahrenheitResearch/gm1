"""Device-resident CM1 storm-volume diagnostics for Rust history/rendering.

The vertical-vorticity interior follows the terrain branch of CM1 21.1
``calcvort``.  The Rust initialized product does not yet publish metric halos,
so the outermost scalar row/column is explicitly copied from the nearest
source-valid interior diagnostic.  Composite reduction remains Rust-owned.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY


CUDA_STORM_DIAGNOSTICS_SCHEMA = "gpuwm-cm1.cuda-storm-diagnostics/v1"
CUDA_STORM_DIAGNOSTICS_ABI = "gpuwm-cm1.cuda-storm-diagnostics-kernels/v1"
CUDA_STORM_DIAGNOSTICS_THREADS = 256
CUDA_STORM_DIAGNOSTICS_KERNEL_NAMES = ("cm1_storm_diagnostics_v1",)
CUDA_STORM_DIAGNOSTICS_CLAIM_LIMIT = (
    "CM1 r21.1 terrain-coordinate interior vertical vorticity at scalar points, "
    "mass-centered vertical velocity, and pass-through public ArWen WSM6 10-cm "
    "reflectivity on one Cartesian uniform-horizontal-grid domain; missing "
    "metric halos are not invented, so only the outermost scalar row/column is "
    "nearest-interior display closure; no reflectivity derivation, composite "
    "reduction, forecast evolution, tornado detection, or model-validation claim"
)

SOURCE_ANCHORS = {
    "terrain_vertical_vorticity": "src/misclibs.F:2035-2107",
    "mass_centered_w": "src/misclibs.F:1579-1587",
    "composite_reduction_rust_consumer": "src/misclibs.F:2259-2286",
    "fornberg_face_extrapolation": "src/param.F:8988-9052",
}

INPUT_NAMES = ("u3d", "v3d", "w3d", "terrain_msl", "rgzu", "rgzv", "reflectivity_dbz")
OUTPUT_NAMES = ("reflectivity_dbz", "updraft_ms", "vertical_vorticity_s")

CM1_CGS1_F32 = 1.875
CM1_CGS2_F32 = -1.25
CM1_CGS3_F32 = 0.375


def storm_diagnostic_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    for name, value, minimum in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    sound = terrain_open_sound_shapes(ni, nj, nk)
    volume = (nk, nj, ni)
    return {
        "u3d": sound["u3d"],
        "v3d": sound["v3d"],
        "w3d": sound["w3d"],
        "terrain_msl": (nj, ni),
        "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni),
        "reflectivity_dbz": volume,
        "updraft_ms": volume,
        "vertical_vorticity_s": volume,
    }


def _validated_spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _require_host_inputs(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = storm_diagnostic_shapes(ni, nj, nk)
    missing = [name for name in INPUT_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing storm diagnostic arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in INPUT_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"storm diagnostic {name} must have C-order float32 shape {shapes[name]}"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"storm diagnostic {name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("storm diagnostic input arrays must not alias")
    return result


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _interp_u(np: Any, u: Any, face: int, j: int, zface: int, nk: int) -> Any:
    ng = CM1_OPEN_NGXY

    def mass(k: int) -> Any:
        return u[k + 1, j + ng, face + ng]

    if zface == 0:
        value = _mul(np, CM1_CGS1_F32, mass(0))
        value = _add(np, value, _mul(np, CM1_CGS2_F32, mass(1)))
        return _add(np, value, _mul(np, CM1_CGS3_F32, mass(2)))
    if zface == nk:
        value = _mul(np, CM1_CGS1_F32, mass(nk - 1))
        value = _add(np, value, _mul(np, CM1_CGS2_F32, mass(nk - 2)))
        return _add(np, value, _mul(np, CM1_CGS3_F32, mass(nk - 3)))
    return _mul(np, 0.5, _add(np, mass(zface - 1), mass(zface)))


def _interp_v(np: Any, v: Any, i: int, face: int, zface: int, nk: int) -> Any:
    ng = CM1_OPEN_NGXY

    def mass(k: int) -> Any:
        return v[k + 1, face + ng, i + ng]

    if zface == 0:
        value = _mul(np, CM1_CGS1_F32, mass(0))
        value = _add(np, value, _mul(np, CM1_CGS2_F32, mass(1)))
        return _add(np, value, _mul(np, CM1_CGS3_F32, mass(2)))
    if zface == nk:
        value = _mul(np, CM1_CGS1_F32, mass(nk - 1))
        value = _add(np, value, _mul(np, CM1_CGS2_F32, mass(nk - 2)))
        return _add(np, value, _mul(np, CM1_CGS3_F32, mass(nk - 3)))
    return _mul(np, 0.5, _add(np, mass(zface - 1), mass(zface)))


def _corner_vorticity(
    np: Any,
    a: Mapping[str, Any],
    *,
    corner_i: int,
    corner_j: int,
    k: int,
    ni: int,
    nj: int,
    nk: int,
    dx: Any,
    dy: Any,
    dz: Any,
) -> Any:
    del ni, nj
    u, v = a["u3d"], a["v3d"]
    zs, rgzu, rgzv = a["terrain_msl"], a["rgzu"], a["rgzv"]
    ng = CM1_OPEN_NGXY
    zt = _mul(np, np.float32(nk), dz)
    terrain_sum = _add(np, zs[corner_j - 1, corner_i - 1], zs[corner_j, corner_i])
    terrain_sum = _add(np, terrain_sum, zs[corner_j, corner_i - 1])
    terrain_sum = _add(np, terrain_sum, zs[corner_j - 1, corner_i])
    denominator = _sub(np, zt, _mul(np, 0.25, terrain_sum))
    transform = _div(np, zt, denominator)
    rdx, rdy, rdz = _div(np, 1.0, dx), _div(np, 1.0, dy), _div(np, 1.0, dz)
    sigma_lower = _mul(np, np.float32(k), dz)
    sigma_upper = _mul(np, np.float32(k + 1), dz)
    lower_height = _sub(np, zt, sigma_lower)
    upper_height = _sub(np, zt, sigma_upper)

    v_right = v[k + 1, corner_j + ng, corner_i + ng]
    v_left = v[k + 1, corner_j + ng, corner_i - 1 + ng]
    v_gradient = _sub(
        np,
        _mul(np, v_right, rgzv[corner_j, corner_i]),
        _mul(np, v_left, rgzv[corner_j, corner_i - 1]),
    )
    v_gradient = _mul(np, v_gradient, rdx)
    v_upper_sum = _add(
        np,
        _interp_v(np, v, corner_i - 1, corner_j, k + 1, nk),
        _interp_v(np, v, corner_i, corner_j, k + 1, nk),
    )
    v_lower_sum = _add(
        np,
        _interp_v(np, v, corner_i - 1, corner_j, k, nk),
        _interp_v(np, v, corner_i, corner_j, k, nk),
    )
    v_vertical = _sub(
        np, _mul(np, upper_height, v_upper_sum), _mul(np, lower_height, v_lower_sum)
    )
    v_vertical = _mul(np, 0.5, v_vertical)
    v_vertical = _mul(np, v_vertical, rdz)
    v_vertical = _mul(
        np,
        v_vertical,
        _sub(np, rgzv[corner_j, corner_i], rgzv[corner_j, corner_i - 1]),
    )
    v_vertical = _mul(np, v_vertical, rdx)
    v_term = _mul(np, transform, _add(np, v_gradient, v_vertical))

    u_north = u[k + 1, corner_j + ng, corner_i + ng]
    u_south = u[k + 1, corner_j - 1 + ng, corner_i + ng]
    u_gradient = _sub(
        np,
        _mul(np, u_north, rgzu[corner_j, corner_i]),
        _mul(np, u_south, rgzu[corner_j - 1, corner_i]),
    )
    u_gradient = _mul(np, u_gradient, rdy)
    u_upper_sum = _add(
        np,
        _interp_u(np, u, corner_i, corner_j - 1, k + 1, nk),
        _interp_u(np, u, corner_i, corner_j, k + 1, nk),
    )
    u_lower_sum = _add(
        np,
        _interp_u(np, u, corner_i, corner_j - 1, k, nk),
        _interp_u(np, u, corner_i, corner_j, k, nk),
    )
    u_vertical = _sub(
        np, _mul(np, upper_height, u_upper_sum), _mul(np, lower_height, u_lower_sum)
    )
    u_vertical = _mul(np, 0.5, u_vertical)
    u_vertical = _mul(np, u_vertical, rdz)
    u_vertical = _mul(
        np,
        u_vertical,
        _sub(np, rgzu[corner_j, corner_i], rgzu[corner_j - 1, corner_i]),
    )
    u_vertical = _mul(np, u_vertical, rdy)
    u_term = _mul(np, transform, _add(np, u_gradient, u_vertical))
    return _sub(np, v_term, u_term)


@dataclass(frozen=True, slots=True)
class CpuStormDiagnostics:
    reflectivity_dbz: Any
    updraft_ms: Any
    vertical_vorticity_s: Any
    exact_interior_words: int
    display_closed_boundary_words: int


def cpu_storm_diagnostics(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
) -> CpuStormDiagnostics:
    """Evaluate the independent source-ordered validation reference."""

    np = importlib.import_module("numpy")
    a = _require_host_inputs(arrays, ni=ni, nj=nj, nk=nk)
    dx, dy, dz = (
        np.float32(_validated_spacing("dx", dx)),
        np.float32(_validated_spacing("dy", dy)),
        np.float32(_validated_spacing("dz", dz)),
    )
    reflectivity = a["reflectivity_dbz"].copy(order="C")
    updraft = np.empty_like(reflectivity)
    vorticity = np.empty_like(reflectivity)
    ng = CM1_OPEN_NGXY
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                updraft[k, j, i] = _mul(
                    np,
                    0.5,
                    _add(
                        np,
                        a["w3d"][k + 1, j + ng, i + ng],
                        a["w3d"][k + 2, j + ng, i + ng],
                    ),
                )
                source_i = min(max(i, 1), ni - 2)
                source_j = min(max(j, 1), nj - 2)
                value = _corner_vorticity(
                    np,
                    a,
                    corner_i=source_i,
                    corner_j=source_j,
                    k=k,
                    ni=ni,
                    nj=nj,
                    nk=nk,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                )
                value = _add(
                    np,
                    value,
                    _corner_vorticity(
                        np,
                        a,
                        corner_i=source_i + 1,
                        corner_j=source_j,
                        k=k,
                        ni=ni,
                        nj=nj,
                        nk=nk,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    ),
                )
                value = _add(
                    np,
                    value,
                    _corner_vorticity(
                        np,
                        a,
                        corner_i=source_i,
                        corner_j=source_j + 1,
                        k=k,
                        ni=ni,
                        nj=nj,
                        nk=nk,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    ),
                )
                value = _add(
                    np,
                    value,
                    _corner_vorticity(
                        np,
                        a,
                        corner_i=source_i + 1,
                        corner_j=source_j + 1,
                        k=k,
                        ni=ni,
                        nj=nj,
                        nk=nk,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    ),
                )
                vorticity[k, j, i] = _mul(np, 0.25, value)
    interior = nk * (ni - 2) * (nj - 2)
    return CpuStormDiagnostics(
        reflectivity,
        updraft,
        vorticity,
        interior,
        nk * ni * nj - interior,
    )


CUDA_STORM_DIAGNOSTICS_SOURCE = r'''// gpuwm-cm1.cuda-storm-diagnostics-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default real must be binary32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}

__device__ __forceinline__ float umat(const float* u,unsigned int face,unsigned int j,unsigned int k,unsigned int nj,unsigned int ni,unsigned int ng){return u[i3(k+1u,j+ng,face+ng,nj+2u*ng,ni+1u+2u*ng)];}
__device__ __forceinline__ float vmat(const float* v,unsigned int i,unsigned int face,unsigned int k,unsigned int nj,unsigned int ni,unsigned int ng){return v[i3(k+1u,face+ng,i+ng,nj+1u+2u*ng,ni+2u*ng)];}
__device__ __forceinline__ float uw(const float* u,unsigned int face,unsigned int j,unsigned int f,unsigned int nk,unsigned int nj,unsigned int ni,unsigned int ng){
  if(f==0u)return a32(a32(m32(1.875f,umat(u,face,j,0u,nj,ni,ng)),m32(-1.25f,umat(u,face,j,1u,nj,ni,ng))),m32(0.375f,umat(u,face,j,2u,nj,ni,ng)));
  if(f==nk)return a32(a32(m32(1.875f,umat(u,face,j,nk-1u,nj,ni,ng)),m32(-1.25f,umat(u,face,j,nk-2u,nj,ni,ng))),m32(0.375f,umat(u,face,j,nk-3u,nj,ni,ng)));
  return m32(0.5f,a32(umat(u,face,j,f-1u,nj,ni,ng),umat(u,face,j,f,nj,ni,ng)));
}
__device__ __forceinline__ float vw(const float* v,unsigned int i,unsigned int face,unsigned int f,unsigned int nk,unsigned int nj,unsigned int ni,unsigned int ng){
  if(f==0u)return a32(a32(m32(1.875f,vmat(v,i,face,0u,nj,ni,ng)),m32(-1.25f,vmat(v,i,face,1u,nj,ni,ng))),m32(0.375f,vmat(v,i,face,2u,nj,ni,ng)));
  if(f==nk)return a32(a32(m32(1.875f,vmat(v,i,face,nk-1u,nj,ni,ng)),m32(-1.25f,vmat(v,i,face,nk-2u,nj,ni,ng))),m32(0.375f,vmat(v,i,face,nk-3u,nj,ni,ng)));
  return m32(0.5f,a32(vmat(v,i,face,f-1u,nj,ni,ng),vmat(v,i,face,f,nj,ni,ng)));
}
__device__ __forceinline__ float corner(const float* u,const float* v,const float* zs,const float* rgzu,const float* rgzv,unsigned int ci,unsigned int cj,unsigned int k,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz){
  const float zt=m32((float)nk,dz),rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);
  float tsum=a32(zs[(unsigned long long)(cj-1u)*ni+ci-1u],zs[(unsigned long long)cj*ni+ci]);
  tsum=a32(tsum,zs[(unsigned long long)cj*ni+ci-1u]);tsum=a32(tsum,zs[(unsigned long long)(cj-1u)*ni+ci]);
  const float tr=d32(zt,s32(zt,m32(0.25f,tsum))),zl=s32(zt,m32((float)k,dz)),zu=s32(zt,m32((float)(k+1u),dz));
  float vg=s32(m32(vmat(v,ci,cj,k,nj,ni,ng),rgzv[(unsigned long long)cj*ni+ci]),m32(vmat(v,ci-1u,cj,k,nj,ni,ng),rgzv[(unsigned long long)cj*ni+ci-1u]));vg=m32(vg,rdx);
  float vup=a32(vw(v,ci-1u,cj,k+1u,nk,nj,ni,ng),vw(v,ci,cj,k+1u,nk,nj,ni,ng));float vlo=a32(vw(v,ci-1u,cj,k,nk,nj,ni,ng),vw(v,ci,cj,k,nk,nj,ni,ng));
  float vv=m32(0.5f,s32(m32(zu,vup),m32(zl,vlo)));vv=m32(vv,rdz);vv=m32(vv,s32(rgzv[(unsigned long long)cj*ni+ci],rgzv[(unsigned long long)cj*ni+ci-1u]));vv=m32(vv,rdx);float vt=m32(tr,a32(vg,vv));
  float ug=s32(m32(umat(u,ci,cj,k,nj,ni,ng),rgzu[(unsigned long long)cj*(ni+1u)+ci]),m32(umat(u,ci,cj-1u,k,nj,ni,ng),rgzu[(unsigned long long)(cj-1u)*(ni+1u)+ci]));ug=m32(ug,rdy);
  float uup=a32(uw(u,ci,cj-1u,k+1u,nk,nj,ni,ng),uw(u,ci,cj,k+1u,nk,nj,ni,ng));float ulo=a32(uw(u,ci,cj-1u,k,nk,nj,ni,ng),uw(u,ci,cj,k,nk,nj,ni,ng));
  float uv=m32(0.5f,s32(m32(zu,uup),m32(zl,ulo)));uv=m32(uv,rdz);uv=m32(uv,s32(rgzu[(unsigned long long)cj*(ni+1u)+ci],rgzu[(unsigned long long)(cj-1u)*(ni+1u)+ci]));uv=m32(uv,rdy);float ut=m32(tr,a32(ug,uv));
  return s32(vt,ut);
}
extern "C" __global__ void cm1_storm_diagnostics_v1(const float* u,const float* v,const float* w,const float* zs,const float* rgzu,const float* rgzv,const float* refl,float* out_refl,float* out_w,float* out_zeta,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz){
  const unsigned long long words=(unsigned long long)ni*nj*nk,first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,stride=(unsigned long long)blockDim.x*gridDim.x;
  for(unsigned long long q=first;q<words;q+=stride){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj));unsigned int si=i<1u?1u:(i>ni-2u?ni-2u:i),sj=j<1u?1u:(j>nj-2u?nj-2u:j);out_refl[q]=refl[q];out_w[q]=m32(0.5f,a32(w[i3(k+1u,j+ng,i+ng,nj+2u*ng,ni+2u*ng)],w[i3(k+2u,j+ng,i+ng,nj+2u*ng,ni+2u*ng)]));float z=corner(u,v,zs,rgzu,rgzv,si,sj,k,ni,nj,nk,ng,dx,dy,dz);z=a32(z,corner(u,v,zs,rgzu,rgzv,si+1u,sj,k,ni,nj,nk,ng,dx,dy,dz));z=a32(z,corner(u,v,zs,rgzu,rgzv,si,sj+1u,k,ni,nj,nk,ng,dx,dy,dz));z=a32(z,corner(u,v,zs,rgzu,rgzv,si+1u,sj+1u,k,ni,nj,nk,ng,dx,dy,dz));out_zeta[q]=m32(0.25f,z);}
}
'''
CUDA_STORM_DIAGNOSTICS_SOURCE_BYTES = CUDA_STORM_DIAGNOSTICS_SOURCE.encode("utf-8")
CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256 = (
    "e3c43ec58038b84b5e3f69f3878a16b4f82d4a851c6bc379af6231ac8eecc9fa"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledStormDiagnosticsKernel:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_storm_diagnostics() -> CompiledStormDiagnosticsKernel:
    digest = hashlib.sha256(CUDA_STORM_DIAGNOSTICS_SOURCE_BYTES).hexdigest()
    if digest != CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256:
        raise RuntimeError("packaged storm diagnostic CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_STORM_DIAGNOSTICS_SOURCE,
        "cm1_storm_diagnostics_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_STORM_DIAGNOSTICS_SOURCE,
        "cm1_storm_diagnostics_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if re.search(
        rf"\.entry\s+{CUDA_STORM_DIAGNOSTICS_KERNEL_NAMES[0]}(?:\s|\()",
        ptx.decode("utf-8"),
    ) is None:
        raise ValueError("compiled PTX lacks storm diagnostic entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled storm diagnostic image is not ELF64 CUDA")
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
    return CompiledStormDiagnosticsKernel(
        module,
        module.get_function(CUDA_STORM_DIAGNOSTICS_KERNEL_NAMES[0]),
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        identity,
    )


def _require_device_inputs(
    cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = storm_diagnostic_shapes(ni, nj, nk)
    missing = [name for name in INPUT_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing storm diagnostic arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in INPUT_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"CUDA storm diagnostic {name} must have C-order float32 shape {shapes[name]}"
            )
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("CUDA storm diagnostic input arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaStormDiagnosticsLaunch:
    fields: dict[str, Any]
    ready_event: Any
    compiled: CompiledStormDiagnosticsKernel
    receipt: dict[str, Any]


def launch_cuda_storm_diagnostics(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    stream: Any | None = None,
    compiled: CompiledStormDiagnosticsKernel | None = None,
) -> CudaStormDiagnosticsLaunch:
    """Enqueue all three history volumes without synchronizing the host."""

    dx, dy, dz = (
        _validated_spacing("dx", dx),
        _validated_spacing("dy", dy),
        _validated_spacing("dz", dz),
    )
    cupy = importlib.import_module("cupy")
    a = _require_device_inputs(cupy, arrays, ni=ni, nj=nj, nk=nk)
    compiled = compiled or compile_cuda_storm_diagnostics()
    stream = stream or cupy.cuda.get_current_stream()
    shape = (nk, nj, ni)
    fields = {
        "reflectivity_dbz": cupy.empty(shape, dtype=cupy.float32),
        "updraft_ms": cupy.empty(shape, dtype=cupy.float32),
        "vertical_vorticity_s": cupy.empty(shape, dtype=cupy.float32),
    }
    words = ni * nj * nk
    grid = (words + CUDA_STORM_DIAGNOSTICS_THREADS - 1) // CUDA_STORM_DIAGNOSTICS_THREADS
    compiled.kernel(
        (grid, 1, 1),
        (CUDA_STORM_DIAGNOSTICS_THREADS, 1, 1),
        (
            a["u3d"],
            a["v3d"],
            a["w3d"],
            a["terrain_msl"],
            a["rgzu"],
            a["rgzv"],
            a["reflectivity_dbz"],
            fields["reflectivity_dbz"],
            fields["updraft_ms"],
            fields["vertical_vorticity_s"],
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
            cupy.uint32(CM1_OPEN_NGXY),
            cupy.float32(dx),
            cupy.float32(dy),
            cupy.float32(dz),
        ),
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    interior = nk * (ni - 2) * (nj - 2)
    receipt = {
        "schema": CUDA_STORM_DIAGNOSTICS_SCHEMA,
        "claim_limit": CUDA_STORM_DIAGNOSTICS_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "exact_interior_words": interior,
            "display_closed_boundary_words": words - interior,
        },
        "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
        "kernel": {
            "abi": CUDA_STORM_DIAGNOSTICS_ABI,
            "source_sha256": CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "launch": {
            "block_x": CUDA_STORM_DIAGNOSTICS_THREADS,
            "grid_x": grid,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
    }
    return CudaStormDiagnosticsLaunch(fields, event, compiled, receipt)


__all__ = [
    "CUDA_STORM_DIAGNOSTICS_ABI",
    "CUDA_STORM_DIAGNOSTICS_CLAIM_LIMIT",
    "CUDA_STORM_DIAGNOSTICS_SCHEMA",
    "CUDA_STORM_DIAGNOSTICS_SOURCE",
    "CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256",
    "CompiledStormDiagnosticsKernel",
    "CpuStormDiagnostics",
    "CudaStormDiagnosticsLaunch",
    "SOURCE_ANCHORS",
    "compile_cuda_storm_diagnostics",
    "cpu_storm_diagnostics",
    "launch_cuda_storm_diagnostics",
    "storm_diagnostic_shapes",
]
