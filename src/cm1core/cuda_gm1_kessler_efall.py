"""Device-resident GM1 Kessler precipitation-energy-fallout operator.

GM1 is the product name. CM1 r21.1 ``mp_driver.F:197-225``,
``misclibs.F:975-1029``, and ``kessler.F:485-522`` are the pinned
mathematical authority. This module implements the optional ``efall=1``
branch at its exact location after rain fall speed/energy accounting and
before mass fallout. It deliberately does not infer terminal velocity or run
the surrounding Kessler phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import math
import re
from typing import Any

from cm1core.errors import refuse


CUDA_GM1_KESSLER_EFALL_SCHEMA = "gm1.cuda-kessler-efall/v1"
CUDA_GM1_KESSLER_EFALL_ABI = "gm1.cuda-kessler-efall-kernels/v1"
CUDA_GM1_KESSLER_EFALL_THREADS = 256
CUDA_GM1_KESSLER_EFALL_KERNEL_NAMES = (
    "gm1_kessler_getcvm_v1",
    "gm1_kessler_getefall_v1",
    "gm1_kessler_apply_efall_v1",
)
CUDA_GM1_KESSLER_EFALL_CLAIM_LIMIT = (
    "Exact optional precipitation-energy-fallout branch for GM1 ptype=1, "
    "efall=1, Cartesian terrain-following geometry and equation_set 1 or 2, "
    "from caller-supplied pre-fallout temperature and Kessler rain fall "
    "velocity. Surrounding pdefq, geterain, mass fallout, local conversion, "
    "satadj, scheduler coupling, trajectory, forecast, and tornado are not "
    "claimed by this isolated operator."
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/mp_driver.F": "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134",
    "src/misclibs.F": "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1",
    "src/kessler.F": "f232f840e425d8c9abd28841e0a46a06e0dbf641460224fa61b6a82e5a650472",
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
}
SOURCE_ANCHORS = {
    "driver_cadence": "src/mp_driver.F:96-108,190-227",
    "mixture_heat_capacity": "src/misclibs.F:975-1029",
    "energy_fallout": "src/kessler.F:485-522",
    "thermodynamic_apply": "src/mp_driver.F:200-225",
    "constants": "src/constants.F:5-31,69-153; src/cm1.F:309-318",
}


@dataclass(frozen=True, slots=True)
class Gm1KesslerEfallSelection:
    ptype: int = 1
    equation_set: int = 1
    testcase: int = 0
    efall: int = 1
    axisymmetric: int = 0
    input_phase: str = "post-geterain"
    output_phase: str = "pre-fallout"


DEFAULT_GM1_KESSLER_EFALL_SELECTION = Gm1KesslerEfallSelection()


def _validate_selection(selection: Gm1KesslerEfallSelection) -> None:
    if not isinstance(selection, Gm1KesslerEfallSelection):
        raise TypeError("selection must be a Gm1KesslerEfallSelection")
    fixed = {
        "ptype": 1,
        "testcase": 0,
        "efall": 1,
        "axisymmetric": 0,
        "input_phase": "post-geterain",
        "output_phase": "pre-fallout",
    }
    for name, wanted in fixed.items():
        value = getattr(selection, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="GM1-KESSLER-EFALL-SELECTION",
                gate="physics-seam",
                option=name,
                value=value,
                reason="the energy-fallout operator is sealed to one exact native phase",
                remedy=f"use {name}={wanted!r} or select the identity efall=0 path outside this operator",
            )
    if type(selection.equation_set) is not int or selection.equation_set not in (1, 2):
        refuse(
            code="GM1-KESSLER-EFALL-EQTSET",
            gate="physics-seam",
            option="equation_set",
            value=selection.equation_set,
            reason="getcvm has exactly the traditional and energy-conserving branches",
            remedy="use equation_set=1 or equation_set=2",
        )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def gm1_kessler_efall_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    mass = (nk, nj, ni)
    return {
        "mf": (nk + 1, nj, ni),
        "pi0": mass,
        "th0": mass,
        "rho": mass,
        "qv": mass,
        "qc": mass,
        "qr": mass,
        "fall_velocity": mass,
        "temperature": mass,
        "pp3d": mass,
        "th3d": mass,
        "prs": mass,
    }


INPUT_NAMES = ("mf", "pi0", "th0", "rho", "qv", "qc", "qr", "fall_velocity")
INOUT_NAMES = ("temperature", "pp3d", "th3d", "prs")
ALL_ARRAY_NAMES = INPUT_NAMES + INOUT_NAMES


def _require_host_arrays(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = gm1_kessler_efall_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing GM1 Kessler efall arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order float32 with shape {shapes[name]}")
        if name in INOUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("GM1 Kessler efall arrays must not alias")
    for name in ("rho", "prs"):
        if bool(np.any(result[name] <= np.float32(0.0))):
            raise ValueError(f"{name} must be > 0")
    if bool(np.any(result["pi0"] + result["pp3d"] <= np.float32(0.0))):
        raise ValueError("pi0 + pp3d must be > 0")
    return result


def _f(np: Any, value: Any) -> Any:
    return np.float32(value)


def _a(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) + _f(np, right))


def _s(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) - _f(np, right))


def _m(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) * _f(np, right))


def _d(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) / _f(np, right))


def _pow(np: Any, base: Any, exponent: Any) -> Any:
    return _f(np, np.power(_f(np, base), _f(np, exponent)))


def cpu_gm1_kessler_efall_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    dt: float,
    selection: Gm1KesslerEfallSelection = DEFAULT_GM1_KESSLER_EFALL_SELECTION,
) -> dict[str, Any]:
    """Apply source-ordered getcvm, getefall, and thermodynamic update."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dz, dt = (_positive(name, value) for name, value in (("dz", dz), ("dt", dt)))
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    rd, cp, cpv, rv = (_f(np, value) for value in (287.04, 1005.7, 1870.0, 461.5))
    cv, cvv = _s(np, cp, rd), _s(np, cpv, rv)
    reps, rovcp = _d(np, rv, rd), _d(np, rd, cp)
    cpl, gravity = _f(np, 4190.0), _f(np, 9.81)
    rp00, rdz = _d(np, 1.0, 1.0e5), _d(np, 1.0, dz)
    dt32 = _f(np, dt)
    cvm = np.empty((nk, nj, ni), dtype=np.float32)
    tendency = np.zeros((nk, nj, ni), dtype=np.float32)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                if selection.equation_set <= 1:
                    value = cv
                else:
                    value = _a(np, cv, _m(np, cvv, a["qv"][p]))
                    value = _a(np, value, _m(np, cpl, a["qc"][p]))
                    value = _a(np, value, _m(np, cpl, a["qr"][p]))
                cvm[p] = value
    for k in range(nk - 1):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                vertical = _m(np, cpl, _s(np, a["temperature"][k + 1, j, i], a["temperature"][p]))
                vertical = _m(np, vertical, rdz)
                vertical = _m(np, vertical, a["mf"][k + 1, j, i])
                tendency[p] = _d(np, _m(np, _m(np, a["qr"][p], a["fall_velocity"][p]), _a(np, vertical, gravity)), cvm[p])
    for k in range(nk - 1):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                increment = _m(np, dt32, tendency[p])
                if abs(float(increment)) >= float(_f(np, 0.0001)):
                    a["temperature"][p] = _a(np, a["temperature"][p], increment)
                    virtual = _a(np, 1.0, _m(np, a["qv"][p], reps))
                    a["prs"][p] = _m(np, _m(np, _m(np, a["rho"][p], rd), a["temperature"][p]), virtual)
                    a["pp3d"][p] = _s(np, _pow(np, _m(np, a["prs"][p], rp00), rovcp), a["pi0"][p])
                    a["th3d"][p] = _s(np, _d(np, a["temperature"][p], _a(np, a["pi0"][p], a["pp3d"][p])), a["th0"][p])
    return {"mixture_cv": cvm, "energy_tendency": tendency}


CUDA_GM1_KESSLER_EFALL_SOURCE = r'''// gm1.cuda-kessler-efall-kernels/v1
#define A(x,y) __fadd_rn((x),(y))
#define S(x,y) __fsub_rn((x),(y))
#define M(x,y) __fmul_rn((x),(y))
#define D(x,y) __fdiv_rn((x),(y))
#define I3(k,j,i,nj,ni) ((((unsigned long long)(k))*(nj)+(j))*(ni)+(i))
extern "C" __global__ void gm1_kessler_getcvm_v1(unsigned eqtset,unsigned ni,unsigned nj,unsigned nk,const float*qv,const float*qc,const float*qr,float*cvm){unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj*nk;if(p>=n)return;float cv=S(1005.7f,287.04f),value=cv;if(eqtset>1u){value=A(value,M(S(1870.0f,461.5f),qv[p]));value=A(value,M(4190.0f,qc[p]));value=A(value,M(4190.0f,qr[p]));}cvm[p]=value;}
extern "C" __global__ void gm1_kessler_getefall_v1(float dz,unsigned ni,unsigned nj,unsigned nk,const float*mf,const float*temperature,const float*cvm,const float*qr,const float*vq,float*tendency){unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj*nk;if(p>=n)return;unsigned i=(unsigned)(p%ni),j=(unsigned)((p/ni)%nj),k=(unsigned)(p/((unsigned long long)ni*nj));tendency[p]=0.0f;if(k+1u<nk){float vertical=M(4190.0f,S(temperature[I3(k+1u,j,i,nj,ni)],temperature[p]));vertical=M(vertical,D(1.0f,dz));vertical=M(vertical,mf[I3(k+1u,j,i,nj,ni)]);tendency[p]=D(M(M(qr[p],vq[p]),A(vertical,9.81f)),cvm[p]);}}
extern "C" __global__ void gm1_kessler_apply_efall_v1(float dt,unsigned ni,unsigned nj,unsigned nk,const float*pi0,const float*th0,const float*rho,const float*qv,const float*tendency,float*temperature,float*pp3d,float*th3d,float*prs){unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj*nk;if(p>=n)return;unsigned k=(unsigned)(p/((unsigned long long)ni*nj));if(k+1u>=nk)return;float inc=M(dt,tendency[p]);if(fabsf(inc)>=0.0001f){float rd=287.04f,reps=D(461.5f,rd),rovcp=D(rd,1005.7f),rp00=D(1.0f,1.0e5f);temperature[p]=A(temperature[p],inc);prs[p]=M(M(M(rho[p],rd),temperature[p]),A(1.0f,M(qv[p],reps)));pp3d[p]=S(powf(M(prs[p],rp00),rovcp),pi0[p]);th3d[p]=S(D(temperature[p],A(pi0[p],pp3d[p])),th0[p]);}}
'''

CUDA_GM1_KESSLER_EFALL_SOURCE_BYTES = CUDA_GM1_KESSLER_EFALL_SOURCE.encode("utf-8")
CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256 = (
    "2b2281a1ee0c96975623608fbe478fa5ea963053c71d364ad555ec73442d9c6e"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledGm1KesslerEfallKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_kessler_efall() -> CompiledGm1KesslerEfallKernels:
    if hashlib.sha256(CUDA_GM1_KESSLER_EFALL_SOURCE_BYTES).hexdigest() != CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256:
        raise RuntimeError("packaged GM1 Kessler efall CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(CUDA_GM1_KESSLER_EFALL_SOURCE, "gm1_kessler_efall_v1_ptx.cu", method="ptx").compile(common + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(CUDA_GM1_KESSLER_EFALL_SOURCE, "gm1_kessler_efall_v1_cubin.cu", method="cubin").compile(common + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in CUDA_GM1_KESSLER_EFALL_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks GM1 Kessler efall entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled GM1 Kessler efall image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_GM1_KESSLER_EFALL_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledGm1KesslerEfallKernels(module, kernels, ptx, cubin, arch, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = gm1_kessler_efall_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA GM1 Kessler efall arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order CUDA float32 with shape {shapes[name]}")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("CUDA GM1 Kessler efall arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Gm1KesslerEfallWorkspace:
    ni: int
    nj: int
    nk: int
    mixture_cv: Any
    energy_tendency: Any


def create_cuda_gm1_kessler_efall_workspace(*, ni: int, nj: int, nk: int) -> Gm1KesslerEfallWorkspace:
    _validate_grid(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    shape = (nk, nj, ni)
    return Gm1KesslerEfallWorkspace(ni, nj, nk, cupy.empty(shape, dtype=cupy.float32), cupy.empty(shape, dtype=cupy.float32))


@dataclass(frozen=True, slots=True)
class CudaGm1KesslerEfallLaunch:
    ready_event: Any
    compiled: CompiledGm1KesslerEfallKernels
    workspace: Gm1KesslerEfallWorkspace
    receipt: dict[str, Any]


def launch_cuda_gm1_kessler_efall(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    dt: float,
    selection: Gm1KesslerEfallSelection = DEFAULT_GM1_KESSLER_EFALL_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGm1KesslerEfallKernels | None = None,
    workspace: Gm1KesslerEfallWorkspace | None = None,
) -> CudaGm1KesslerEfallLaunch:
    """Enqueue getcvm, getefall, and the guarded thermodynamic update."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dz, dt = (_positive(name, value) for name, value in (("dz", dz), ("dt", dt)))
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    compiled = compiled or compile_cuda_gm1_kessler_efall()
    workspace = workspace or create_cuda_gm1_kessler_efall_workspace(ni=ni, nj=nj, nk=nk)
    if not isinstance(workspace, Gm1KesslerEfallWorkspace) or (workspace.ni, workspace.nj, workspace.nk) != (ni, nj, nk):
        raise ValueError("GM1 Kessler efall workspace geometry differs")
    for name in ("mixture_cv", "energy_tendency"):
        value = getattr(workspace, name)
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != (nk, nj, ni) or not value.flags.c_contiguous:
            raise ValueError(f"workspace {name} must be C-order CUDA float32 mass shape")
    pointers = {int(value.data.ptr) for value in a.values()}
    scratch = {int(workspace.mixture_cv.data.ptr), int(workspace.energy_tendency.data.ptr)}
    if len(scratch) != 2 or pointers.intersection(scratch):
        raise ValueError("GM1 Kessler efall workspace arrays alias")
    stream = stream or cupy.cuda.get_current_stream()
    words = ni * nj * nk
    grid = ((words + CUDA_GM1_KESSLER_EFALL_THREADS - 1) // CUDA_GM1_KESSLER_EFALL_THREADS, 1, 1)
    threads = (CUDA_GM1_KESSLER_EFALL_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    getcvm, getefall, apply = compiled.kernels
    with stream:
        getcvm(grid, threads, (u32(selection.equation_set), u32(ni), u32(nj), u32(nk), a["qv"], a["qc"], a["qr"], workspace.mixture_cv), stream=stream)
        getefall(grid, threads, (f32(dz), u32(ni), u32(nj), u32(nk), a["mf"], a["temperature"], workspace.mixture_cv, a["qr"], a["fall_velocity"], workspace.energy_tendency), stream=stream)
        apply(grid, threads, (f32(dt), u32(ni), u32(nj), u32(nk), a["pi0"], a["th0"], a["rho"], a["qv"], workspace.energy_tendency, a["temperature"], a["pp3d"], a["th3d"], a["prs"]), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_KESSLER_EFALL_SCHEMA,
        "claim_limit": CUDA_GM1_KESSLER_EFALL_CLAIM_LIMIT,
        "authority": {"product_name": "GM1", "math_release": "CM1 r21.1", "tarball_sha256": AUTHORITY_TARBALL_SHA256, "source_sha256": dict(AUTHORITY_SOURCE_SHA256)},
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "dz_m": dz, "dt_s": dt},
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "source_order": ["getcvm", "getefall(setzero=1)", "guarded temperature/pressure/Exner/theta update"],
        "residency": {"inputs": "CUDA device", "outputs": "CUDA device in-place", "scratch": "CUDA device reusable workspace", "host_scientific_copy_or_sync": False},
        "kernel": {"abi": CUDA_GM1_KESSLER_EFALL_ABI, "names": list(CUDA_GM1_KESSLER_EFALL_KERNEL_NAMES), "source_sha256": CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256, "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(), "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(), "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {"float32": "explicit tolerance against independent native default-REAL oracle for pow", "null_or_missing_is_pass": False},
    }
    return CudaGm1KesslerEfallLaunch(event, compiled, workspace, receipt)


__all__ = [
    "ALL_ARRAY_NAMES", "AUTHORITY_SOURCE_SHA256", "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_KESSLER_EFALL_ABI", "CUDA_GM1_KESSLER_EFALL_CLAIM_LIMIT",
    "CUDA_GM1_KESSLER_EFALL_KERNEL_NAMES", "CUDA_GM1_KESSLER_EFALL_SCHEMA",
    "CUDA_GM1_KESSLER_EFALL_SOURCE", "CUDA_GM1_KESSLER_EFALL_SOURCE_BYTES",
    "CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256", "CompiledGm1KesslerEfallKernels",
    "CudaGm1KesslerEfallLaunch", "DEFAULT_GM1_KESSLER_EFALL_SELECTION",
    "Gm1KesslerEfallSelection", "Gm1KesslerEfallWorkspace", "SOURCE_ANCHORS",
    "compile_cuda_gm1_kessler_efall", "cpu_gm1_kessler_efall_inplace",
    "create_cuda_gm1_kessler_efall_workspace", "gm1_kessler_efall_shapes",
    "launch_cuda_gm1_kessler_efall",
]
