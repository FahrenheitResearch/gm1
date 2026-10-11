"""Selected solve1 upper-Rayleigh held tendencies for the real product.

CM1 r21.1 constructs the static ``tauh/taus/tauf`` coefficients in
``param.F:8631-8673`` and assigns the four held tendencies once per large
step in ``solve1.F:412-454``.  This module applies that assignment on compact
physical-domain carriers.  The horizontal one-cell coefficient halo is an
input because the source averages mass-point coefficients onto the two
staggered wind families.

The bound real-product selection has no Coriolis or large-scale momentum
forcing: ``icor=0`` and ``lspgrad=0`` in the authority namelist, and all
large-scale nudging switches are false.  Coriolis is in solve2, not solve1,
so this module deliberately contains no rotation/pressure-gradient kernel.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_RAYLEIGH_HELD_SCHEMA = "gpuwm-cm1.cuda-rayleigh-held/v1"
CUDA_RAYLEIGH_HELD_ABI = "gpuwm-cm1.cuda-rayleigh-held-kernels/v1"
CUDA_RAYLEIGH_HELD_THREADS = 256
CUDA_RAYLEIGH_HELD_KERNEL_NAME = "cm1_rayleigh_held_physical_f32_v1"
CUDA_RAYLEIGH_HELD_CLAIM_LIMIT = (
    "CM1 r21.1 selected irdamp=1/hrdamp=0 solve1 assignment on compact "
    "physical u/v/w/theta carriers using caller-supplied source-valid "
    "tauh/taus/tauf coefficients with one horizontal mass halo; no tau "
    "libm identity claim, lateral halo tendency publication, Coriolis "
    "(selected icor=0), large-scale pressure gradient (selected lspgrad=0), "
    "large-scale nudging, other solve1 tendencies, RK integration, evolved "
    "trajectory, forecast, tornado, or model-validation claim"
)

SOURCE_ANCHORS = {
    "rayleigh_coefficients": "src/param.F:8631-8673",
    "held_assignment": "src/solve1.F:412-454",
    "large_scale_pressure_gradient_disabled": "src/solve1.F:536-681",
    "large_scale_nudging_disabled": "src/solve1.F:1608-1693",
    "coriolis_stage_local_disabled": "src/solve2.F:631-720",
    "authority_namelist": "run/config_files/supercell/namelist.input",
}

AUTHORITY_SELECTION = {
    "axisymm": 0,
    "betaplane": 0,
    "do_lsnudge": False,
    "do_lsnudge_u": False,
    "do_lsnudge_v": False,
    "hrdamp": 0,
    "icor": 0,
    "iinit": 1,
    "irdamp": 1,
    "lspgrad": 0,
}

REAL_PRODUCT_SELECTION = {
    "grid": [300, 300, 75],
    "spacing_m": [100.0, 100.0, 200.0],
    "model_top_m": 15000.0,
    "upper_damping_depth_m": 3000.0,
    "upper_damping_start_m": 12000.0,
    "upper_damping_rate_s_inv": 0.0033333333333,
    "moving_frame": False,
    "u_move_m_s": 0.0,
    "v_move_m_s": 0.0,
}


@dataclass(frozen=True, slots=True)
class RayleighHeldSelection:
    """The only upper-damping selection admitted by this production slice."""

    upper_damping: int = 1
    horizontal_rayleigh: int = 0
    model_top_m: float = 15000.0
    damping_depth_m: float = 3000.0
    damping_rate_s_inv: float = 0.0033333333333

    @property
    def damping_start_m(self) -> float:
        return self.model_top_m - self.damping_depth_m


DEFAULT_RAYLEIGH_HELD_SELECTION = RayleighHeldSelection()


def _validate_selection(selection: RayleighHeldSelection) -> None:
    if not isinstance(selection, RayleighHeldSelection):
        raise TypeError("selection must be a RayleighHeldSelection")
    expected = DEFAULT_RAYLEIGH_HELD_SELECTION
    for name in (
        "upper_damping",
        "horizontal_rayleigh",
        "model_top_m",
        "damping_depth_m",
        "damping_rate_s_inv",
    ):
        value = getattr(selection, name)
        wanted = getattr(expected, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="CM1-RAYLEIGH-HELD-SELECTION",
                gate="boundaries-closures",
                option=name,
                value=value,
                reason="this bridge is sealed to the real-product upper-Rayleigh selection",
                remedy=f"use {name}={wanted!r} or create a separately evidenced profile",
            )
    if not math.isfinite(selection.damping_start_m) or selection.damping_start_m <= 0.0:
        raise ValueError("damping layer must begin above the model bottom")


def validate_rotation_large_scale_disabled(selection: Mapping[str, object]) -> None:
    """Refuse any attempt to silently activate the audited-off forcing branches."""

    if not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping")
    missing = [name for name in AUTHORITY_SELECTION if name not in selection]
    if missing:
        raise ValueError(f"forcing selection is missing: {', '.join(missing)}")
    for name, expected in AUTHORITY_SELECTION.items():
        value = selection[name]
        if type(value) is not type(expected) or value != expected:  # noqa: E721
            refuse(
                code="CM1-ROTATION-LARGE-SCALE-SELECTION",
                gate="boundaries-closures",
                option=name,
                value=value,
                reason="the authority-bound selection disables Coriolis and large-scale momentum forcing",
                remedy="add and evidence a separately named configuration before enabling this branch",
            )


def rayleigh_held_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    mass = (nk, nj, ni)
    return {
        "tauh": (nk, nj + 2, ni + 2),
        "taus": (nk, nj + 2, ni + 2),
        "tauf": (nk + 1, nj + 2, ni + 2),
        "u0": (nk,),
        "v0": (nk,),
        "ua": (nk, nj, ni + 1),
        "va": (nk, nj + 1, ni),
        "wa": mass,
        "tha": mass,
        "uten1": (nk, nj, ni + 1),
        "vten1": (nk, nj + 1, ni),
        "wten1": mass,
        "thten1": mass,
    }


INPUT_NAMES = ("tauh", "taus", "tauf", "u0", "v0", "ua", "va", "wa", "tha")
OUTPUT_NAMES = ("uten1", "vten1", "wten1", "thten1")
ALL_ARRAY_NAMES = INPUT_NAMES + OUTPUT_NAMES


def _require_host_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = rayleigh_held_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing Rayleigh-held arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be a C-order float32 array with shape {shapes[name]}"
            )
        if name in OUTPUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if name in INPUT_NAMES and not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("Rayleigh-held arrays must not alias")
    return result


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.add(left, right, dtype=np.float32)


def _f32_sub(np: Any, left: Any, right: Any) -> Any:
    return np.subtract(left, right, dtype=np.float32)


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.multiply(left, right, dtype=np.float32)


def cpu_rayleigh_held_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RayleighHeldSelection = DEFAULT_RAYLEIGH_HELD_SELECTION,
) -> None:
    """Apply the source assignment over the live compact physical domains."""

    _validate_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    rate = np.float32(selection.damping_rate_s_inv)
    negative_rate = np.float32(-rate)
    negative_rate_half = _f32_mul(np, negative_rate, np.float32(0.5))

    u_tau = _f32_add(
        np,
        a["tauh"][:, 1 : nj + 1, 0 : ni + 1],
        a["tauh"][:, 1 : nj + 1, 1 : ni + 2],
    )
    u_coefficient = _f32_mul(np, negative_rate_half, u_tau)
    u_departure = _f32_sub(np, a["ua"], a["u0"][:, None, None])
    np.copyto(a["uten1"], _f32_mul(np, u_coefficient, u_departure))

    v_tau = _f32_add(
        np,
        a["tauh"][:, 0 : nj + 1, 1 : ni + 1],
        a["tauh"][:, 1 : nj + 2, 1 : ni + 1],
    )
    v_coefficient = _f32_mul(np, negative_rate_half, v_tau)
    v_departure = _f32_sub(np, a["va"], a["v0"][:, None, None])
    np.copyto(a["vten1"], _f32_mul(np, v_coefficient, v_departure))

    w_coefficient = _f32_mul(np, negative_rate, a["tauf"][:nk, 1 : nj + 1, 1 : ni + 1])
    np.copyto(a["wten1"], _f32_mul(np, w_coefficient, a["wa"]))
    theta_coefficient = _f32_mul(
        np, negative_rate, a["taus"][:, 1 : nj + 1, 1 : ni + 1]
    )
    np.copyto(a["thten1"], _f32_mul(np, theta_coefficient, a["tha"]))


@dataclass(frozen=True, slots=True)
class RayleighCoefficients:
    tauh: Any
    taus: Any
    tauf: Any
    receipt: dict[str, Any]


def cpu_build_rayleigh_coefficients(
    zh_halo: Any,
    zf_halo: Any,
    *,
    selection: RayleighHeldSelection = DEFAULT_RAYLEIGH_HELD_SELECTION,
) -> RayleighCoefficients:
    """Transcribe ``param.F`` using the host float32 cosine implementation.

    The result is suitable for the exact held-tendency bridge.  Since CM1's
    source does not pin a cross-platform ``cos`` implementation, the receipt
    explicitly limits coefficient generation to source-formula parity.
    """

    _validate_selection(selection)
    np = importlib.import_module("numpy")
    if (
        not isinstance(zh_halo, np.ndarray)
        or zh_halo.dtype != np.dtype("float32")
        or zh_halo.ndim != 3
        or not zh_halo.flags.c_contiguous
    ):
        raise ValueError("zh_halo must be a C-order rank-3 float32 array")
    nk, halo_nj, halo_ni = zh_halo.shape
    if (
        not isinstance(zf_halo, np.ndarray)
        or zf_halo.dtype != np.dtype("float32")
        or zf_halo.shape != (nk + 1, halo_nj, halo_ni)
        or not zf_halo.flags.c_contiguous
    ):
        raise ValueError("zf_halo must be C-order float32 with one extra z face")
    if halo_nj < 3 or halo_ni < 3:
        raise ValueError("coefficient geometry requires one horizontal halo")
    if not bool(np.all(np.isfinite(zh_halo))) or not bool(np.all(np.isfinite(zf_halo))):
        raise ValueError("Rayleigh height arrays contain non-finite values")

    start = np.float32(selection.damping_start_m)
    pi = np.float32(math.pi)
    half = np.float32(0.5)
    one = np.float32(1.0)
    top = zf_halo[-1]

    def coefficient(height: Any) -> Any:
        result = np.zeros_like(height, dtype=np.float32, order="C")
        mask = height > start
        numerator = _f32_sub(np, height, start)
        denominator = _f32_sub(np, top[None, :, :], start)
        ratio = np.divide(numerator, denominator, dtype=np.float32)
        angle = _f32_mul(np, pi, ratio)
        cosine = np.cos(angle, dtype=np.float32)
        computed = _f32_mul(np, half, _f32_sub(np, one, cosine))
        result[mask] = computed[mask]
        return np.ascontiguousarray(result)

    tauh = coefficient(zh_halo)
    tauf = coefficient(zf_halo)
    taus = tauh.copy(order="C")
    receipt = {
        "schema": "gpuwm-cm1.rayleigh-coefficients/v1",
        "source_anchor": SOURCE_ANCHORS["rayleigh_coefficients"],
        "selection": {
            "irdamp": selection.upper_damping,
            "hrdamp": selection.horizontal_rayleigh,
            "model_top_m": selection.model_top_m,
            "damping_start_m": selection.damping_start_m,
            "damping_rate_s_inv": selection.damping_rate_s_inv,
        },
        "libm_claim": "host float32 source-formula transcription; cross-libm raw-bit identity is not claimed",
    }
    return RayleighCoefficients(tauh=tauh, taus=taus, tauf=tauf, receipt=receipt)


CUDA_RAYLEIGH_HELD_SOURCE = r"""
__device__ __forceinline__ float cm1_add(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float cm1_sub(const float a,const float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float cm1_mul(const float a,const float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}

extern "C" __global__ void cm1_rayleigh_held_physical_f32_v1(
 const float* __restrict__ tauh,const float* __restrict__ taus,const float* __restrict__ tauf,
 const float* __restrict__ u0,const float* __restrict__ v0,
 const float* __restrict__ ua,const float* __restrict__ va,
 const float* __restrict__ wa,const float* __restrict__ tha,
 float* __restrict__ uten1,float* __restrict__ vten1,
 float* __restrict__ wten1,float* __restrict__ thten1,
 unsigned int ni,unsigned int nj,unsigned int nk,float rdalpha)
{
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned long long nu=(unsigned long long)nk*nj*(ni+1u);
 const unsigned long long nv=(unsigned long long)nk*(nj+1u)*ni;
 const unsigned long long ns=(unsigned long long)nk*nj*ni;
 const float neg=-rdalpha;
 const float negh=cm1_mul(neg,0.5f);
 for(unsigned long long z=first;z<nu;z+=step){
  unsigned int f=(unsigned int)(z%(ni+1u));unsigned long long q=z/(ni+1u);
  unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);
  float pair=cm1_add(tauh[i3(k,j+1u,f,nj+2u,ni+2u)],tauh[i3(k,j+1u,f+1u,nj+2u,ni+2u)]);
  uten1[z]=cm1_mul(cm1_mul(negh,pair),cm1_sub(ua[z],u0[k]));
 }
 for(unsigned long long z=first;z<nv;z+=step){
  unsigned int i=(unsigned int)(z%ni);unsigned long long q=z/ni;
  unsigned int f=(unsigned int)(q%(nj+1u)),k=(unsigned int)(q/(nj+1u));
  float pair=cm1_add(tauh[i3(k,f,i+1u,nj+2u,ni+2u)],tauh[i3(k,f+1u,i+1u,nj+2u,ni+2u)]);
  vten1[z]=cm1_mul(cm1_mul(negh,pair),cm1_sub(va[z],v0[k]));
 }
 for(unsigned long long z=first;z<ns;z+=step){
  unsigned int i=(unsigned int)(z%ni);unsigned long long q=z/ni;
  unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);
  unsigned long long p=i3(k,j+1u,i+1u,nj+2u,ni+2u);
  wten1[z]=cm1_mul(cm1_mul(neg,tauf[p]),wa[z]);
  thten1[z]=cm1_mul(cm1_mul(neg,taus[p]),tha[z]);
 }
}
"""

CUDA_RAYLEIGH_HELD_SOURCE_BYTES = CUDA_RAYLEIGH_HELD_SOURCE.encode("utf-8")
# Updated only when the reviewed source above changes.
CUDA_RAYLEIGH_HELD_SOURCE_SHA256 = (
    "c9d18eb70a56ee9158ac3cafc82a6f3e439626d934e6a04cbdbd92ed0ee2cb43"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_RAYLEIGH_HELD_SOURCE_BYTES).hexdigest() == (
        CUDA_RAYLEIGH_HELD_SOURCE_SHA256
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
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledRayleighHeldKernel:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_rayleigh_held() -> CompiledRayleighHeldKernel:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged Rayleigh-held CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if arch != compiler_arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
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
    ptx_program = compiler._NVRTCProgram(
        CUDA_RAYLEIGH_HELD_SOURCE, "cm1_rayleigh_held_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_RAYLEIGH_HELD_SOURCE, "cm1_rayleigh_held_v1_cubin.cu", method="cubin"
    )
    cubin, _ = cubin_program.compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_RAYLEIGH_HELD_KERNEL_NAME}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks the Rayleigh-held entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled Rayleigh-held image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function(CUDA_RAYLEIGH_HELD_KERNEL_NAME)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledRayleighHeldKernel(
        module_owner=module,
        kernel=kernel,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = rayleigh_held_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA Rayleigh-held arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be a C-order CUDA float32 array with shape {shapes[name]}"
            )
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA Rayleigh-held arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaRayleighHeldLaunch:
    ready_event: Any
    compiled: CompiledRayleighHeldKernel
    receipt: dict[str, Any]


def launch_cuda_rayleigh_held(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RayleighHeldSelection = DEFAULT_RAYLEIGH_HELD_SELECTION,
    stream: Any | None = None,
    compiled: CompiledRayleighHeldKernel | None = None,
) -> CudaRayleighHeldLaunch:
    """Enqueue the exact held assignment without host synchronization."""

    _validate_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    if compiled is None:
        compiled = compile_cuda_rayleigh_held()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    count = max(nk * nj * (ni + 1), nk * (nj + 1) * ni, nk * nj * ni)
    grid = (count + CUDA_RAYLEIGH_HELD_THREADS - 1) // CUDA_RAYLEIGH_HELD_THREADS
    compiled.kernel(
        (grid, 1, 1),
        (CUDA_RAYLEIGH_HELD_THREADS, 1, 1),
        (
            a["tauh"],
            a["taus"],
            a["tauf"],
            a["u0"],
            a["v0"],
            a["ua"],
            a["va"],
            a["wa"],
            a["tha"],
            a["uten1"],
            a["vten1"],
            a["wten1"],
            a["thten1"],
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
            cupy.float32(selection.damping_rate_s_inv),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_RAYLEIGH_HELD_SCHEMA,
        "claim_limit": CUDA_RAYLEIGH_HELD_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": "once per large step in solve1, before all three solve2 RK passes",
        "selection": {
            **AUTHORITY_SELECTION,
            **REAL_PRODUCT_SELECTION,
        },
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "kernel": {
            "abi": CUDA_RAYLEIGH_HELD_ABI,
            "name": CUDA_RAYLEIGH_HELD_KERNEL_NAME,
            "source_sha256": CUDA_RAYLEIGH_HELD_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": CUDA_RAYLEIGH_HELD_THREADS,
            "grid_x": grid,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaRayleighHeldLaunch(ready_event=event, compiled=compiled, receipt=receipt)


__all__ = [
    "ALL_ARRAY_NAMES",
    "AUTHORITY_SELECTION",
    "CUDA_RAYLEIGH_HELD_ABI",
    "CUDA_RAYLEIGH_HELD_CLAIM_LIMIT",
    "CUDA_RAYLEIGH_HELD_KERNEL_NAME",
    "CUDA_RAYLEIGH_HELD_SCHEMA",
    "CUDA_RAYLEIGH_HELD_SOURCE",
    "CUDA_RAYLEIGH_HELD_SOURCE_SHA256",
    "CompiledRayleighHeldKernel",
    "CudaRayleighHeldLaunch",
    "DEFAULT_RAYLEIGH_HELD_SELECTION",
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "REAL_PRODUCT_SELECTION",
    "RayleighCoefficients",
    "RayleighHeldSelection",
    "SOURCE_ANCHORS",
    "compile_cuda_rayleigh_held",
    "cpu_build_rayleigh_coefficients",
    "cpu_rayleigh_held_inplace",
    "launch_cuda_rayleigh_held",
    "rayleigh_held_shapes",
    "validate_rotation_large_scale_disabled",
]
