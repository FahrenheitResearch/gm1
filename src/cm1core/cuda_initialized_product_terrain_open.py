"""Strict corrected-real adapter for the terrain/all-open acoustic gate.

The Rust initialized-product arena owns every physical-domain carrier.  This
module adds only the source-required padded working arrays and the ``eqtset=2``
``solve2`` preparation words needed by :mod:`cuda_psolver3_terrain_open`.
Scientific preparation remains on-device in the production path.  The CPU
packer is an independent validation reference for caller-supplied arrays; it
never opens or decodes the Rust payload.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_initialized_product import CudaInitializedProduct
from cm1core.cuda_psolver3_terrain_open import (
    CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT,
    TerrainOpenSoundAblations,
    launch_cuda_psolver3_terrain_open_three_stage,
    terrain_open_sound_shapes,
)
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    cpu_bcp_open_inplace,
    cpu_bcs_open_inplace,
    cpu_bcu_open_inplace,
    cpu_bcv_open_inplace,
    cpu_bcw_open_inplace,
)
from cm1core.cuda_terrain_open_boundary import (
    TERRAIN_STATIC_REQUIRED,
    cpu_bcwsfc_terrain_physical_inplace,
    terrain_static_shapes,
)


CUDA_INITIALIZED_TERRAIN_OPEN_SCHEMA = (
    "gpuwm-cm1.cuda-initialized-product-terrain-open/v1"
)
CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT = (
    "corrected-real Rust physical carriers adapted to the CM1 r21.1 serial "
    "terrain psolver=3 all-open acoustic validation gate; exact eqtset=2 "
    "theta-rho/pressure-term preparation, zero initialized radiation histories "
    "and held zero large-step tendencies, source-family horizontal fills, and "
    "physical terrain bcwsfc are included; source-valid metric/base halos, "
    "terrain-varying hydrostatic base admission, large-step tendencies, full "
    "dynamics/physics, forecast, product, tornado, and performance claims are "
    "excluded"
)
SOURCE_ANCHORS = {
    "moist_accumulators": "src/solve2.F:300-392",
    "eqtset2_preparation": "src/solve2.F:394-412",
    "theta_rho_fill": "src/solve2.F:453-455;src/bc.F:12-172",
    "terrain_piadv": "src/solve2.F:1027-1050",
    "acoustic_composition": "src/sound.F:81-853",
    "stage_family_fills": "src/solve2.F:1665-1706",
    "initial_radiation_history": "src/cm1.F:1559-1566",
    "terrain_bottom_w": "src/bc.F:1151-1178",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    "src/sound.F": "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54",
}

CM1_RD_F32 = 287.04
CM1_RV_F32 = 461.5
CM1_CP_F32 = 1005.7
CM1_CPV_F32 = 1870.0
CM1_CPL_F32 = 4190.0
CM1_CPI_F32 = 2106.0

PROFILE_NAMES = ("pi0", "th0")
STATE_NAMES = (
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "ppx",
    "th3d",
    "q3d",
)
REQUIRED_PRODUCT_NAMES = PROFILE_NAMES + STATE_NAMES + TERRAIN_STATIC_REQUIRED
PREPARATION_COMPARE_NAMES = (
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "ppx",
    "piadv",
    "ppterm",
    "thv",
    "scalar",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
)


def _expected_product_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    return {
        "pi0": (nk,),
        "th0": (nk,),
        "ua": (nk, nj, ni + 1),
        "u3d": (nk, nj, ni + 1),
        "va": (nk, nj + 1, ni),
        "v3d": (nk, nj + 1, ni),
        "wa": zface,
        "w3d": zface,
        "ppi": mass,
        "pp3d": mass,
        "ppx": mass,
        "th3d": mass,
        "q3d": (6, nk, nj, ni),
        **terrain_static_shapes(ni=ni, nj=nj, nk=nk),
    }


def _positive_extent(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 3:
        raise ValueError(f"{name} must be an integer >= 3")
    return value


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _require_host_product_carriers(
    carriers: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    expected = _expected_product_shapes(ni, nj, nk)
    missing = [name for name in REQUIRED_PRODUCT_NAMES if name not in carriers]
    if missing:
        raise ValueError(f"missing initialized-product carriers: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in REQUIRED_PRODUCT_NAMES:
        value = carriers[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != expected[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"initialized-product carrier {name} must have C-order float32 "
                f"shape {expected[name]}"
            )
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"initialized-product carrier {name} is non-finite")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("initialized-product validation carriers must not alias")
    return result


def _fill_host_initial_boundaries(
    arrays: dict[str, Any], metrics: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    for name in ("ua", "u3d"):
        cpu_bcu_open_inplace(arrays[name], ni=ni, nj=nj, nk=nk)
    for name in ("va", "v3d"):
        cpu_bcv_open_inplace(arrays[name], ni=ni, nj=nj, nk=nk)
    for name in ("wa", "w3d"):
        cpu_bcw_open_inplace(arrays[name], ni=ni, nj=nj, nk=nk, flag=1)
    for u_name, v_name, w_name in (
        ("ua", "va", "wa"),
        ("u3d", "v3d", "w3d"),
    ):
        view = dict(arrays)
        view["u3d"] = arrays[u_name]
        view["v3d"] = arrays[v_name]
        view["w3d"] = arrays[w_name]
        cpu_bcwsfc_terrain_physical_inplace(
            view, metrics, ni=ni, nj=nj, nk=nk
        )
    for name in ("ppi", "pp3d", "ppx", "piadv", "thv"):
        cpu_bcp_open_inplace(arrays[name], ni=ni, nj=nj, nk=nk)
    cpu_bcs_open_inplace(arrays["scalar"], ni=ni, nj=nj, nk=nk)


@dataclass(frozen=True, slots=True)
class CpuInitializedTerrainOpenState:
    arrays: dict[str, Any]
    metrics: dict[str, Any]
    source_anchors: dict[str, str]
    claim_limit: str


def cpu_pack_initialized_product_terrain_open(
    carriers: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> CpuInitializedTerrainOpenState:
    """Build the independent CPU validation state from supplied carriers."""

    ni = _positive_extent("ni", ni)
    nj = _positive_extent("nj", nj)
    nk = _positive_extent("nk", nk)
    np = importlib.import_module("numpy")
    source = _require_host_product_carriers(carriers, ni=ni, nj=nj, nk=nk)
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in terrain_open_sound_shapes(ni, nj, nk).items()
    }
    for name in ("uh", "vh", "ruf", "rvf"):
        arrays[name].fill(np.float32(1.0))
    metrics = {
        name: source[name].copy(order="C") for name in TERRAIN_STATIC_REQUIRED
    }
    ng = CM1_OPEN_NGXY
    mass = (slice(1, nk + 1), slice(1, nj + 1), slice(1, ni + 1))
    arrays["ua"][1 : nk + 1, ng : ng + nj, ng : ng + ni + 1] = source["ua"]
    arrays["u3d"][1 : nk + 1, ng : ng + nj, ng : ng + ni + 1] = source["u3d"]
    arrays["va"][1 : nk + 1, ng : ng + nj + 1, ng : ng + ni] = source["va"]
    arrays["v3d"][1 : nk + 1, ng : ng + nj + 1, ng : ng + ni] = source["v3d"]
    arrays["wa"][1 : nk + 2, ng : ng + nj, ng : ng + ni] = source["wa"]
    arrays["w3d"][1 : nk + 2, ng : ng + nj, ng : ng + ni] = source["w3d"]
    for name in ("ppi", "pp3d", "ppx"):
        arrays[name][mass] = source[name]
    arrays["scalar"][1 : nk + 1, ng : ng + nj, ng : ng + ni] = source["th3d"]

    rd = np.float32(CM1_RD_F32)
    rv = np.float32(CM1_RV_F32)
    cp = np.float32(CM1_CP_F32)
    cpv = np.float32(CM1_CPV_F32)
    cv = _sub(np, cp, rd)
    cvv = _sub(np, cpv, rv)
    rovcp = _div(np, rd, cp)
    reps = _div(np, rv, rd)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    cpl = np.float32(CM1_CPL_F32)
    cpi = np.float32(CM1_CPI_F32)
    q = source["q3d"]
    for k0 in range(nk):
        k = k0 + 1
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                qv = np.maximum(q[0, k0, j0, i0], zero)
                liquid = _add(np, q[1, k0, j0, i0], q[2, k0, j0, i0])
                solid = _add(
                    np,
                    _add(np, q[3, k0, j0, i0], q[4, k0, j0, i0]),
                    q[5, k0, j0, i0],
                )
                qli = np.maximum(zero, _add(np, liquid, solid))
                cpli = _add(
                    np,
                    _mul(np, cpl, np.maximum(zero, liquid)),
                    _mul(np, cpi, np.maximum(zero, solid)),
                )
                theta = _add(np, source["th0"][k0], source["th3d"][k0, j0, i0])
                numerator = _mul(np, theta, _add(np, one, _mul(np, reps, qv)))
                arrays["thv"][k, j, i] = _div(
                    np, numerator, _add(np, _add(np, one, qv), qli)
                )
                cpm = _add(np, _add(np, cp, _mul(np, cpv, qv)), cpli)
                cvm = _div(np, one, _add(np, _add(np, cv, _mul(np, cvv, qv)), cpli))
                piadv = _add(np, source["pi0"][k0], source["pp3d"][k0, j0, i0])
                arrays["piadv"][k, j, i] = piadv
                arrays["ppterm"][k, j, i] = _mul(
                    np, _mul(np, _mul(np, piadv, rovcp), cpm), cvm
                )
    _fill_host_initial_boundaries(arrays, metrics, ni=ni, nj=nj, nk=nk)
    return CpuInitializedTerrainOpenState(
        arrays,
        metrics,
        dict(SOURCE_ANCHORS),
        CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT,
    )


def compare_initialized_terrain_open_preparation_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every prepared source-live family with raw-bit semantics."""

    np = importlib.import_module("numpy")
    fields: dict[str, Any] = {}
    for name in PREPARATION_COMPARE_NAMES:
        if name not in expected or name not in candidate:
            raise ValueError(f"missing initialized terrain/open field {name}")
        left, right = expected[name], candidate[name]
        if (
            not isinstance(left, np.ndarray)
            or not isinstance(right, np.ndarray)
            or left.dtype != np.dtype("float32")
            or right.dtype != np.dtype("float32")
            or left.shape != right.shape
        ):
            raise ValueError(f"initialized terrain/open comparison layout differs for {name}")
        exact = bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
            "candidate_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
        }
    return {
        "fields": fields,
        "max_ulp": 0,
        "mode": "raw-bit",
        "null_or_missing_is_pass": False,
        "passed": all(item["exact"] for item in fields.values()),
    }


CUDA_INITIALIZED_TERRAIN_OPEN_KERNEL_NAMES = (
    "cm1_ipto_pack_mass_v1",
    "cm1_ipto_pack_u_v1",
    "cm1_ipto_pack_v_v1",
    "cm1_ipto_pack_w_v1",
)

CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE = r'''// gpuwm-cm1.cuda-initialized-product-terrain-open/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void cm1_ipto_pack_mass_v1(
 const float* pi0,const float* th0,const float* ppi0,const float* pp0,const float* ppx0,const float* th3d,const float* q,
 float* ppi,float* pp,float* ppx,float* sadv,float* ppterm,float* thv,float* scalar,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
 const float rd=287.04f,rv=461.5f,cp=1005.7f,cpv=1870.0f,cpl=4190.0f,cpi=2106.0f,cv=s32(cp,rd),cvv=s32(cpv,rv),rovcp=d32(rd,cp),reps=d32(rv,rd);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/(ni*nj));unsigned long long p=i3(k+1u,j+1u,i+1u,nj+2u,ni+2u),s=i3(k+1u,j+ng,i+ng,nj+2u*ng,ni+2u*ng);
  ppi[p]=ppi0[z];pp[p]=pp0[z];ppx[p]=ppx0[z];scalar[s]=th3d[z];
  float qv=fmaxf(q[z],0.0f),liq=a32(q[n+z],q[2u*n+z]),sol=a32(a32(q[3u*n+z],q[4u*n+z]),q[5u*n+z]),qli=fmaxf(a32(liq,sol),0.0f),cpli=a32(m32(cpl,fmaxf(liq,0.0f)),m32(cpi,fmaxf(sol,0.0f)));
  float theta=a32(th0[k],th3d[z]),num=m32(theta,a32(1.0f,m32(reps,qv)));thv[p]=d32(num,a32(a32(1.0f,qv),qli));
  float cpm=a32(a32(cp,m32(cpv,qv)),cpli),cvm=d32(1.0f,a32(a32(cv,m32(cvv,qv)),cpli)),pi=a32(pi0[k],pp0[z]);sadv[p]=pi;ppterm[p]=m32(m32(m32(pi,rovcp),cpm),cvm);
 }}
extern "C" __global__ void cm1_ipto_pack_u_v1(const float* aa,const float* cc,float* a,float* c,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*(ni+1u);for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%(ni+1u)),j=(unsigned int)((z/(ni+1u))%nj),k=(unsigned int)(z/(nj*(ni+1u)));unsigned long long d=i3(k+1u,j+ng,i+ng,nj+2u*ng,ni+1u+2u*ng);a[d]=aa[z];c[d]=cc[z];}}
extern "C" __global__ void cm1_ipto_pack_v_v1(const float* aa,const float* cc,float* a,float* c,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*(nj+1u)*ni;for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%(nj+1u)),k=(unsigned int)(z/((nj+1u)*ni));unsigned long long d=i3(k+1u,j+ng,i+ng,nj+1u+2u*ng,ni+2u*ng);a[d]=aa[z];c[d]=cc[z];}}
extern "C" __global__ void cm1_ipto_pack_w_v1(const float* aa,const float* cc,float* a,float* c,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/(nj*ni));unsigned long long d=i3(k+1u,j+ng,i+ng,nj+2u*ng,ni+2u*ng);a[d]=aa[z];c[d]=cc[z];}}
'''
CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_BYTES = (
    CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE.encode("utf-8")
)
CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256 = (
    "b4e645681eb07e026e547e242191fedc48f8d083d015a39e30063fb05241b054"
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
class CompiledInitializedTerrainOpenKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]


def compile_cuda_initialized_terrain_open() -> CompiledInitializedTerrainOpenKernels:
    digest = hashlib.sha256(CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_BYTES).hexdigest()
    if digest != CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256:
        raise RuntimeError("packaged initialized terrain/open CUDA source differs from pin")
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
        CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE,
        "cm1_initialized_terrain_open_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE,
        "cm1_initialized_terrain_open_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx.decode("utf-8"))
        for name in CUDA_INITIALIZED_TERRAIN_OPEN_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks initialized terrain/open entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled initialized terrain/open image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_INITIALIZED_TERRAIN_OPEN_KERNEL_NAMES
    )
    return CompiledInitializedTerrainOpenKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options
    )


def _require_cuda_product_geometry(
    product: CudaInitializedProduct, cupy: Any
) -> tuple[int, int, int, dict[str, Any]]:
    if not isinstance(product, CudaInitializedProduct):
        raise TypeError("product must be a CudaInitializedProduct")
    nj, ni = product.export.crop_count_yx
    nk = product.export.carrier("pi0").shape[0]
    ni = _positive_extent("ni", ni)
    nj = _positive_extent("nj", nj)
    nk = _positive_extent("nk", nk)
    expected = _expected_product_shapes(ni, nj, nk)
    fields: dict[str, Any] = {}
    for name in REQUIRED_PRODUCT_NAMES:
        value = product.field(name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != expected[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"CUDA initialized-product carrier {name} layout differs")
        fields[name] = value
    if len({int(value.data.ptr) for value in fields.values()}) != len(fields):
        raise ValueError("CUDA initialized-product carrier views alias")
    return ni, nj, nk, fields


@dataclass(frozen=True, slots=True)
class CudaInitializedTerrainOpenState:
    product: CudaInitializedProduct
    arrays: dict[str, Any]
    metrics: dict[str, Any]
    ready_event: Any
    compiled: CompiledInitializedTerrainOpenKernels
    open_compiled: Any
    terrain_compiled: Any
    receipt: dict[str, Any]


def prepare_cuda_initialized_product_terrain_open(
    product: CudaInitializedProduct,
    *,
    allow_cropped_validation: bool = False,
    stream: Any | None = None,
    compiled: CompiledInitializedTerrainOpenKernels | None = None,
    open_compiled: Any | None = None,
    terrain_compiled: Any | None = None,
) -> CudaInitializedTerrainOpenState:
    """Pack one hash-bound initialized product into padded acoustic arrays."""

    if not isinstance(allow_cropped_validation, bool):
        raise TypeError("allow_cropped_validation must be boolean")
    if not product.export.is_full_domain and not allow_cropped_validation:
        raise ValueError(
            "cropped initialized product is validation-only; set "
            "allow_cropped_validation=True"
        )
    cupy = importlib.import_module("cupy")
    from cm1core.cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary
    from cm1core.cuda_terrain_open_boundary import compile_cuda_terrain_open_boundary

    ni, nj, nk, source = _require_cuda_product_geometry(product, cupy)
    arrays = {
        name: cupy.zeros(shape, dtype=cupy.float32)
        for name, shape in terrain_open_sound_shapes(ni, nj, nk).items()
    }
    metrics = {name: source[name] for name in TERRAIN_STATIC_REQUIRED}
    compiled = compiled or compile_cuda_initialized_terrain_open()
    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    terrain_compiled = terrain_compiled or compile_cuda_terrain_open_boundary()
    stream = stream or product.stream
    kmass, ku, kv, kw = compiled.kernels
    _, _, _, kpwe, kpsn, kfwe, kfsn, kwrigid, _, _ = open_compiled.kernels
    _, _, kbcwsfc = terrain_compiled.kernels
    threads = (128, 1, 1)
    largest = max(max(value.size for value in arrays.values()), nk * nj * (ni + 1))
    grid = ((largest + 127) // 128, 1, 1)
    mass_grid = (((nk * nj * ni) + 127) // 128, 1, 1)
    u32 = cupy.uint32
    with stream:
        stream.wait_event(product.ready_event)
        for name in ("uh", "vh", "ruf", "rvf"):
            arrays[name].fill(cupy.float32(1.0))
        kmass(
            mass_grid,
            threads,
            (
                source["pi0"],
                source["th0"],
                source["ppi"],
                source["pp3d"],
                source["ppx"],
                source["th3d"],
                source["q3d"],
                arrays["ppi"],
                arrays["pp3d"],
                arrays["ppx"],
                arrays["piadv"],
                arrays["ppterm"],
                arrays["thv"],
                arrays["scalar"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        ku(
            grid,
            threads,
            (
                source["ua"],
                source["u3d"],
                arrays["ua"],
                arrays["u3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        kv(
            grid,
            threads,
            (
                source["va"],
                source["v3d"],
                arrays["va"],
                arrays["v3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        kw(
            grid,
            threads,
            (
                source["wa"],
                source["w3d"],
                arrays["wa"],
                arrays["w3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        for name in ("ua", "u3d"):
            kfwe(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)), stream=stream)
            kfsn(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)), stream=stream)
        for name in ("va", "v3d"):
            kfsn(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)), stream=stream)
            kfwe(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)), stream=stream)
        for name in ("wa", "w3d"):
            kfwe(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)), stream=stream)
            kfsn(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)), stream=stream)
            kwrigid(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
        for u_name, v_name, w_name in (("ua", "va", "wa"), ("u3d", "v3d", "w3d")):
            kbcwsfc(grid, threads, (arrays[u_name], arrays[v_name], metrics["gz"], metrics["dzdx"], metrics["dzdy"], arrays[w_name], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
        for name in ("ppi", "pp3d", "ppx", "piadv", "thv"):
            kpwe(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk)), stream=stream)
            kpsn(grid, threads, (arrays[name], u32(ni), u32(nj), u32(nk)), stream=stream)
        kfwe(grid, threads, (arrays["scalar"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
        kfsn(grid, threads, (arrays["scalar"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "base_policy": (
            "Rust corrected-v3 horizontal-broadcast validation base; "
            "terrain-varying hydrostatic admission remains refused"
        ),
        "claim_limit": CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT,
        "comparison_contract": {
            "max_ulp": 0,
            "mode": "raw-bit",
            "null_or_missing_is_pass": False,
        },
        "crop": {
            "count_yx": list(product.export.crop_count_yx),
            "full_domain": product.export.is_full_domain,
            "origin_yx": list(product.export.crop_origin_yx),
        },
        "kernel": {
            "source_sha256": CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "policies": {
            "large_step_tendencies": "held-exact-zero validation seam",
            "radiation_history": "initialized exact zero; produced after each RK stage",
            "source_halos": "not present; open-family horizontal validation fills only",
        },
        "schema": CUDA_INITIALIZED_TERRAIN_OPEN_SCHEMA,
        "authority": {
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
        },
        "source": {
            "artifact_sha256": product.export.source_artifact_sha256,
            "contract_sha256": product.export.source_contract_sha256,
            "manifest_sha256": product.export.manifest_sha256,
            "payload_sha256": product.export.payload_sha256,
        },
        "source_anchors": dict(SOURCE_ANCHORS),
    }
    return CudaInitializedTerrainOpenState(
        product,
        arrays,
        metrics,
        event,
        compiled,
        open_compiled,
        terrain_compiled,
        receipt,
    )


@dataclass(frozen=True, slots=True)
class CudaInitializedTerrainOpenLaunch:
    state: CudaInitializedTerrainOpenState
    acoustic: Any
    receipt: dict[str, Any]


def launch_cuda_initialized_product_terrain_open_three_stage(
    product: CudaInitializedProduct,
    *,
    dt: float,
    nsound: int,
    allow_cropped_validation: bool = False,
    ablations: TerrainOpenSoundAblations | None = None,
    stream: Any | None = None,
) -> CudaInitializedTerrainOpenLaunch:
    """Prepare and execute the corrected-real terrain/all-open acoustic gate."""

    state = prepare_cuda_initialized_product_terrain_open(
        product,
        allow_cropped_validation=allow_cropped_validation,
        stream=stream,
    )
    ni, nj = product.export.crop_count_yx[1], product.export.crop_count_yx[0]
    nk = product.export.carrier("pi0").shape[0]
    acoustic = launch_cuda_psolver3_terrain_open_three_stage(
        state.arrays,
        state.metrics,
        ni=ni,
        nj=nj,
        nk=nk,
        dt=dt,
        nsound=nsound,
        dx=100.0,
        dy=100.0,
        dz=200.0,
        ablations=ablations,
        stream=stream or product.stream,
        terrain_compiled=state.terrain_compiled,
        open_compiled=state.open_compiled,
    )
    receipt = {
        **state.receipt,
        "acoustic": acoustic.receipt,
        "acoustic_claim_limit": CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT,
    }
    return CudaInitializedTerrainOpenLaunch(state, acoustic, receipt)


__all__ = [
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT",
    "CUDA_INITIALIZED_TERRAIN_OPEN_KERNEL_NAMES",
    "CUDA_INITIALIZED_TERRAIN_OPEN_SCHEMA",
    "CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE",
    "CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256",
    "CudaInitializedTerrainOpenLaunch",
    "CudaInitializedTerrainOpenState",
    "CpuInitializedTerrainOpenState",
    "PREPARATION_COMPARE_NAMES",
    "SOURCE_ANCHORS",
    "compare_initialized_terrain_open_preparation_raw",
    "compile_cuda_initialized_terrain_open",
    "cpu_pack_initialized_product_terrain_open",
    "launch_cuda_initialized_product_terrain_open_three_stage",
    "prepare_cuda_initialized_product_terrain_open",
]
