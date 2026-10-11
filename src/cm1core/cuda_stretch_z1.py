"""CM1 r21.1 Wilhelmson-Chen ``stretch_z=1`` vertical grid metrics."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .cuda_sound3_periodic import _canonical_arch, _image_options
from .errors import refuse


CUDA_STRETCH_Z1_SCHEMA = "gpuwm-cm1.cuda-stretch-z1/v1"
CUDA_STRETCH_Z1_ABI = "gpuwm-cm1.cuda-stretch-z1-kernel/v1"
STRETCH_Z1_METRIC_PROOF_SCHEMA = "gpuwm-cm1.stretch-z1-metric-proof/v1"
CM1_STRETCH_Z1_CLAIM_LIMIT = (
    "CM1 r21.1 param.F:7814-8184 Wilhelmson-Chen stretch_z=1 vertical "
    "w/scalar levels and mh/mf/rmh/rmf metrics for a flat Cartesian grid; "
    "no stretch_z=2/3/4, terrain composition, horizontal stretch, moving "
    "domain, coupled forecast, storm, tornado, product, or performance claim"
)
SOURCE_ANCHORS = {
    "selector_and_feasibility": "src/param.F:7814-7914",
    "wilhelmson_chen_levels": "src/param.F:7916-7953",
    "halo_scalar_and_w_levels": "src/param.F:8162-8173",
    "mh_mf_metrics": "src/param.F:8168-8179",
    "inverse_metrics": "src/param.F:8189-8209",
    "positive_spacing_check": "src/param.F:8211-8225",
}


@dataclass(frozen=True, slots=True)
class StretchZ1Selection:
    stretch_z: int = 1
    terrain_flag: bool = False
    stretch_x: int = 0
    stretch_y: int = 0
    axisymm: int = 0


def require_stretch_z1_selection(selection: StretchZ1Selection) -> None:
    checks = (
        ("stretch_z", selection.stretch_z, 1, "ARWEN-STRETCH-Z1-SELECTOR"),
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-STRETCH-Z1-TERRAIN"),
        ("stretch_x", selection.stretch_x, 0, "ARWEN-STRETCH-Z1-X"),
        ("stretch_y", selection.stretch_y, 0, "ARWEN-STRETCH-Z1-Y"),
        ("axisymm", selection.axisymm, 0, "ARWEN-STRETCH-Z1-AXISYMM"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="stretch-z1-selection",
                reason=f"the bounded grid metric path requires {option}={expected!r}",
                remedy="select the exact flat Cartesian stretch_z=1 family",
                option=option,
                value=value,
            )


@dataclass(frozen=True, slots=True)
class StretchZ1Parameters:
    nk: int
    dz: float
    ztop: float
    str_bot: float
    str_top: float
    dz_bot: float
    dz_top: float


@dataclass(frozen=True, slots=True)
class StretchZ1Derived:
    nk1: int
    nk2: int
    nk3: int
    nominal_dz_f32: float
    c1_f32: float
    c2_f32: float


@dataclass(frozen=True, slots=True)
class StretchZ1Grid:
    zf: Any
    zh: Any
    sigmaf: Any
    sigma: Any
    mh: Any
    mf: Any
    rmh: Any
    rmf: Any
    derived: StretchZ1Derived
    sha256: str


@dataclass(frozen=True, slots=True)
class StretchZ1MetricProof:
    schema: str
    parameters: StretchZ1Parameters
    grid_sha256: str
    mh_profile_sha256: str
    mf_profile_sha256: str
    proof_sha256: str


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _nint_positive(np: Any, value: Any) -> int:
    return int(math.floor(float(np.float32(value)) + 0.5))


def derive_stretch_z1(
    parameters: StretchZ1Parameters,
    selection: StretchZ1Selection = StretchZ1Selection(),
) -> StretchZ1Derived:
    require_stretch_z1_selection(selection)
    if isinstance(parameters.nk, bool) or not isinstance(parameters.nk, int) or parameters.nk < 4:
        raise ValueError("nk must be an integer >= 4")
    np = importlib.import_module("numpy")
    dz, ztop, str_bot, str_top, dz_bot, dz_top = map(
        np.float32,
        (
            parameters.dz,
            parameters.ztop,
            parameters.str_bot,
            parameters.str_top,
            parameters.dz_bot,
            parameters.dz_top,
        ),
    )
    if not all(bool(np.isfinite(value)) for value in (dz, ztop, str_bot, str_top, dz_bot, dz_top)):
        raise ValueError("stretch_z=1 parameters must be finite")
    if dz <= 0 or ztop <= 0 or dz_bot <= 0 or dz_top <= 0:
        raise ValueError("dz, ztop, dz_bot, and dz_top must be positive")
    if str_bot < 0 or str_top <= str_bot or str_top > ztop:
        raise ValueError("require 0 <= str_bot < str_top <= ztop")
    nominal = _mul(np, np.float32(0.5), _add(np, dz_bot, dz_top))
    nk1 = _nint_positive(np, _div(np, str_bot, dz_bot))
    nk3 = _nint_positive(np, _div(np, _sub(np, ztop, str_top), dz_top))
    nk2 = parameters.nk - (nk1 + nk3)
    if nk1 < 0 or nk2 < 2 or nk3 < 0:
        raise ValueError("stretch_z=1 requires nk1>=0, nk2>=2, and nk3>=0")
    bottom_error = abs(float(_sub(np, _mul(np, np.float32(nk1), dz_bot), str_bot)))
    middle_error = abs(
        float(
            _sub(
                np,
                _mul(np, np.float32(nk2), nominal),
                _sub(np, str_top, str_bot),
            )
        )
    )
    top_error = abs(
        float(
            _sub(
                np,
                _mul(np, np.float32(nk3), dz_top),
                _sub(np, ztop, str_top),
            )
        )
    )
    if bottom_error > 0.01 or middle_error >= 0.01 or top_error > 0.01:
        raise ValueError(
            "stretch_z=1 segment depths must satisfy the official 0.01 m feasibility checks"
        )
    c2 = _div(
        np,
        _sub(np, nominal, dz_bot),
        _mul(np, _mul(np, nominal, nominal), np.float32(nk2 - 1)),
    )
    c1 = _sub(np, _div(np, dz_bot, nominal), _mul(np, c2, nominal))
    test_len = _add(
        np,
        _add(
            np,
            _mul(np, dz_bot, np.float32(nk1)),
            _mul(np, nominal, np.float32(nk2)),
        ),
        _mul(np, dz_top, np.float32(nk3)),
    )
    if abs(float(_sub(np, ztop, test_len))) > float(_mul(np, np.float32(1.0e-6), ztop)):
        raise ValueError("stretch_z=1 total length fails the official relative feasibility check")
    return StretchZ1Derived(nk1, nk2, nk3, float(nominal), float(c1), float(c2))


def build_stretch_z1_grid(
    parameters: StretchZ1Parameters,
    selection: StretchZ1Selection = StretchZ1Selection(),
) -> StretchZ1Grid:
    np = importlib.import_module("numpy")
    derived = derive_stretch_z1(parameters, selection)
    nk = parameters.nk
    dz, dz_bot, dz_top = map(np.float32, (parameters.dz, parameters.dz_bot, parameters.dz_top))
    nominal, c1, c2 = map(
        np.float32,
        (derived.nominal_dz_f32, derived.c1_f32, derived.c2_f32),
    )
    zf = np.zeros(nk + 3, dtype=np.float32)
    sigmaf = np.zeros(nk + 3, dtype=np.float32)
    for k in range(1, derived.nk1 + 2):
        zf[k] = _mul(np, np.float32(k - 1), dz_bot)
        sigmaf[k] = zf[k]
    for k in range(derived.nk1 + 1, derived.nk1 + derived.nk2 + 2):
        delta = np.float32(k - 1 - derived.nk1)
        factor = _add(np, c1, _mul(np, _mul(np, c2, delta), nominal))
        distance = _mul(np, _mul(np, factor, delta), nominal)
        zf[k] = _add(np, zf[derived.nk1 + 1], distance)
        sigmaf[k] = zf[k]
    for k in range(derived.nk1 + derived.nk2 + 2, nk + 2):
        zf[k] = _add(np, zf[k - 1], dz_top)
        sigmaf[k] = zf[k]
    zf[0] = np.float32(-zf[2])
    zf[nk + 2] = _add(np, zf[nk + 1], _sub(np, zf[nk + 1], zf[nk]))
    sigmaf[0] = np.float32(-sigmaf[2])
    sigmaf[nk + 2] = _add(
        np, sigmaf[nk + 1], _sub(np, sigmaf[nk + 1], sigmaf[nk])
    )
    zh = np.empty(nk + 2, dtype=np.float32)
    mh = np.empty(nk + 2, dtype=np.float32)
    for k in range(nk + 2):
        zh[k] = _mul(np, np.float32(0.5), _add(np, zf[k + 1], zf[k]))
        mh[k] = _div(np, dz, _sub(np, zf[k + 1], zf[k]))
    zh[0] = np.float32(-zh[1])
    zh[nk + 1] = _add(
        np, zh[nk], _mul(np, np.float32(2.0), _sub(np, zf[nk + 1], zh[nk]))
    )
    mf = np.empty(nk + 3, dtype=np.float32)
    for k in range(1, nk + 2):
        mf[k] = _div(np, dz, _sub(np, zh[k], zh[k - 1]))
    mf[0], mf[nk + 2] = mf[1], mf[nk + 1]
    sigma = np.empty(nk + 2, dtype=np.float32)
    for k in range(nk + 2):
        sigma[k] = _mul(np, np.float32(0.5), _add(np, sigmaf[k], sigmaf[k + 1]))
    rmh = np.empty_like(mh)
    rmf = np.empty_like(mf)
    for k in range(nk + 2):
        rmh[k] = _div(np, np.float32(1.0), mh[k])
    for k in range(nk + 3):
        rmf[k] = _div(np, np.float32(1.0), mf[k])
    if not bool(np.all(np.diff(zf[1 : nk + 2]) > np.float32(0.0))):
        raise ValueError("stretch_z=1 generated nonpositive physical spacing")
    digest = hashlib.sha256()
    for value in (zf, zh, sigmaf, sigma, mh, mf, rmh, rmf):
        digest.update(value.astype("<f4", copy=False).tobytes())
    return StretchZ1Grid(zf, zh, sigmaf, sigma, mh, mf, rmh, rmf, derived, digest.hexdigest())


def _profile_sha256(value: Any) -> str:
    np = importlib.import_module("numpy")
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32)
        .astype("<f4", copy=False)
        .tobytes(order="C")
    ).hexdigest()


def build_stretch_z1_metric_proof(
    parameters: StretchZ1Parameters,
) -> StretchZ1MetricProof:
    """Seal the exact official grid identity consumed by pressure operators."""

    grid = build_stretch_z1_grid(parameters)
    mh_sha256 = _profile_sha256(grid.mh)
    mf_sha256 = _profile_sha256(grid.mf[: parameters.nk + 2])
    payload = {
        "schema": STRETCH_Z1_METRIC_PROOF_SCHEMA,
        "parameters": {
            name: getattr(parameters, name) for name in parameters.__dataclass_fields__
        },
        "grid_sha256": grid.sha256,
        "mh_profile_sha256": mh_sha256,
        "mf_profile_sha256": mf_sha256,
    }
    proof_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return StretchZ1MetricProof(
        STRETCH_Z1_METRIC_PROOF_SCHEMA,
        parameters,
        grid.sha256,
        mh_sha256,
        mf_sha256,
        proof_sha256,
    )


def validate_stretch_z1_metric_profiles(
    proof: StretchZ1MetricProof,
    *,
    mh: Any,
    mf: Any,
    nk: int,
) -> None:
    """Fail closed unless supplied pressure profiles are the sealed grid."""

    if not isinstance(proof, StretchZ1MetricProof):
        raise TypeError("stretch_z=1 metric proof has the wrong type")
    if proof.schema != STRETCH_Z1_METRIC_PROOF_SCHEMA:
        raise ValueError("stretch_z=1 metric proof schema mismatch")
    if proof.parameters.nk != nk:
        raise ValueError("stretch_z=1 metric proof level count mismatch")
    expected = build_stretch_z1_metric_proof(proof.parameters)
    if proof != expected:
        raise ValueError("stretch_z=1 metric proof identity mismatch")
    np = importlib.import_module("numpy")
    for name, value, expected_shape, expected_sha256 in (
        ("mh", mh, (nk + 2,), proof.mh_profile_sha256),
        ("mf", mf, (nk + 2,), proof.mf_profile_sha256),
    ):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} stretch_z=1 profile must be float32 numpy")
        if value.shape != expected_shape or not value.flags.c_contiguous:
            raise ValueError(
                f"{name} stretch_z=1 profile must have shape {expected_shape} and C layout"
            )
        if _profile_sha256(value) != expected_sha256:
            raise ValueError(f"{name} differs from the sealed stretch_z=1 grid")


CUDA_STRETCH_Z1_SOURCE = r"""// gpuwm-cm1.cuda-stretch-z1-kernel/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void cm1_stretch_z1_v1(float* zf,float* zh,float* sigmaf,float* sigma,float* mh,float* mf,float* rmh,float* rmf,unsigned nk,unsigned nk1,unsigned nk2,unsigned nk3,float dz,float dzbot,float dztop,float nominal,float c1,float c2){if(blockIdx.x||threadIdx.x)return;for(unsigned k=0;k<nk+3u;++k){zf[k]=0.0f;sigmaf[k]=0.0f;}for(unsigned k=1u;k<=nk1+1u;++k){zf[k]=m32((float)(k-1u),dzbot);sigmaf[k]=zf[k];}for(unsigned k=nk1+1u;k<=nk1+nk2+1u;++k){float delta=(float)(k-1u-nk1);float factor=a32(c1,m32(m32(c2,delta),nominal));float distance=m32(m32(factor,delta),nominal);zf[k]=a32(zf[nk1+1u],distance);sigmaf[k]=zf[k];}for(unsigned k=nk1+nk2+2u;k<=nk+1u;++k){zf[k]=a32(zf[k-1u],dztop);sigmaf[k]=zf[k];}zf[0]=-zf[2];zf[nk+2u]=a32(zf[nk+1u],s32(zf[nk+1u],zf[nk]));sigmaf[0]=-sigmaf[2];sigmaf[nk+2u]=a32(sigmaf[nk+1u],s32(sigmaf[nk+1u],sigmaf[nk]));for(unsigned k=0u;k<=nk+1u;++k){zh[k]=m32(0.5f,a32(zf[k+1u],zf[k]));mh[k]=d32(dz,s32(zf[k+1u],zf[k]));}zh[0]=-zh[1];zh[nk+1u]=a32(zh[nk],m32(2.0f,s32(zf[nk+1u],zh[nk])));for(unsigned k=1u;k<=nk+1u;++k)mf[k]=d32(dz,s32(zh[k],zh[k-1u]));mf[0]=mf[1];mf[nk+2u]=mf[nk+1u];for(unsigned k=0u;k<=nk+1u;++k){sigma[k]=m32(0.5f,a32(sigmaf[k],sigmaf[k+1u]));rmh[k]=d32(1.0f,mh[k]);}for(unsigned k=0u;k<=nk+2u;++k)rmf[k]=d32(1.0f,mf[k]);}
"""
CUDA_STRETCH_Z1_SOURCE_BYTES = CUDA_STRETCH_Z1_SOURCE.encode("utf-8")
CUDA_STRETCH_Z1_SOURCE_SHA256 = (
    "fb015416dc07405654ea264b50b13ce7eb986b5406c4894ab4e5d082bb6a7441"
)


@dataclass(frozen=True, slots=True)
class CompiledStretchZ1:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_stretch_z1() -> CompiledStretchZ1:
    actual = hashlib.sha256(CUDA_STRETCH_Z1_SOURCE_BYTES).hexdigest()
    if actual != CUDA_STRETCH_Z1_SOURCE_SHA256:
        raise RuntimeError(f"packaged stretch_z=1 CUDA source differs from pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_STRETCH_Z1_SOURCE, "cm1_stretch_z1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_STRETCH_Z1_SOURCE, "cm1_stretch_z1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(r"\.entry\s+cm1_stretch_z1_v1(?:\s|\()", ptx.decode("utf-8")):
        raise ValueError("compiled PTX lacks stretch_z=1 entry point")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function("cm1_stretch_z1_v1")
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
    return CompiledStretchZ1(module, kernel, ptx, cubin, arch, identity)


@dataclass(frozen=True, slots=True)
class CudaStretchZ1Launch:
    ready_event: Any
    compiled: CompiledStretchZ1
    grid: DeviceStretchZ1Grid
    receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DeviceStretchZ1Grid:
    zf: Any
    zh: Any
    sigmaf: Any
    sigma: Any
    mh: Any
    mf: Any
    rmh: Any
    rmf: Any
    derived: StretchZ1Derived


def launch_cuda_stretch_z1(
    parameters: StretchZ1Parameters,
    selection: StretchZ1Selection = StretchZ1Selection(),
    *,
    stream: Any | None = None,
    compiled: CompiledStretchZ1 | None = None,
) -> CudaStretchZ1Launch:
    cupy = importlib.import_module("cupy")
    derived = derive_stretch_z1(parameters, selection)
    compiled = compile_cuda_stretch_z1() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        arrays = {
            "zf": cupy.empty(parameters.nk + 3, dtype=cupy.float32),
            "zh": cupy.empty(parameters.nk + 2, dtype=cupy.float32),
            "sigmaf": cupy.empty(parameters.nk + 3, dtype=cupy.float32),
            "sigma": cupy.empty(parameters.nk + 2, dtype=cupy.float32),
            "mh": cupy.empty(parameters.nk + 2, dtype=cupy.float32),
            "mf": cupy.empty(parameters.nk + 3, dtype=cupy.float32),
            "rmh": cupy.empty(parameters.nk + 2, dtype=cupy.float32),
            "rmf": cupy.empty(parameters.nk + 3, dtype=cupy.float32),
        }
        compiled.kernel(
            (1, 1, 1),
            (1, 1, 1),
            (
                *(arrays[name] for name in ("zf", "zh", "sigmaf", "sigma", "mh", "mf", "rmh", "rmf")),
                cupy.uint32(parameters.nk),
                cupy.uint32(derived.nk1),
                cupy.uint32(derived.nk2),
                cupy.uint32(derived.nk3),
                cupy.float32(parameters.dz),
                cupy.float32(parameters.dz_bot),
                cupy.float32(parameters.dz_top),
                cupy.float32(derived.nominal_dz_f32),
                cupy.float32(derived.c1_f32),
                cupy.float32(derived.c2_f32),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    grid = DeviceStretchZ1Grid(
        arrays["zf"], arrays["zh"], arrays["sigmaf"], arrays["sigma"],
        arrays["mh"], arrays["mf"], arrays["rmh"], arrays["rmf"], derived
    )
    receipt = {
        "schema": CUDA_STRETCH_Z1_SCHEMA,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "parameters": {name: getattr(parameters, name) for name in parameters.__dataclass_fields__},
        "derived": {name: getattr(derived, name) for name in derived.__dataclass_fields__},
        "kernel": {"source_sha256": CUDA_STRETCH_Z1_SOURCE_SHA256, "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_STRETCH_Z1_CLAIM_LIMIT,
    }
    return CudaStretchZ1Launch(event, compiled, grid, receipt)


__all__ = (
    "CM1_STRETCH_Z1_CLAIM_LIMIT",
    "CUDA_STRETCH_Z1_SOURCE_BYTES",
    "CUDA_STRETCH_Z1_SOURCE_SHA256",
    "CudaStretchZ1Launch",
    "DeviceStretchZ1Grid",
    "SOURCE_ANCHORS",
    "STRETCH_Z1_METRIC_PROOF_SCHEMA",
    "StretchZ1Derived",
    "StretchZ1Grid",
    "StretchZ1MetricProof",
    "StretchZ1Parameters",
    "StretchZ1Selection",
    "build_stretch_z1_grid",
    "build_stretch_z1_metric_proof",
    "compile_cuda_stretch_z1",
    "derive_stretch_z1",
    "launch_cuda_stretch_z1",
    "require_stretch_z1_selection",
    "validate_stretch_z1_metric_profiles",
)
