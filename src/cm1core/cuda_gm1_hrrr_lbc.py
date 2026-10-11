"""Real-HRRR time-varying lateral forcing for the GM1 terrain backend.

This is a deliberately named GM1 real-data extension, not CM1 r21.1's
``lsnudge`` branch.  CM1's stock large-scale nudging operates on domain-mean
profiles and refuses terrain.  The present operator instead supplies the
standard limited-area-model boundary role needed by the real-HRRR terrain
forecast: Rust-validated source carriers are interpolated in time and imposed
at the outer lateral edge.  Legacy callers retain the original direct
quadratic blend.  Corrected future-boundary callers select an exact-edge,
time-consistent Davies tendency with a C2 interior taper.

Python owns validation and launch orchestration only.  Every evolving-field
operation executes in the CUDA kernels below.  The model interior is never
replaced by the driving analysis.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import stat
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping

from cm1core.cuda_momentum_rk import momentum_rk_stage_timestep_f32
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.errors import refuse


GM1_HRRR_LBC_SCHEMA = "gm1.cuda-hrrr-lateral-boundary/v1"
GM1_HRRR_LBC_ABI = "gm1.cuda-hrrr-lateral-boundary-kernels/v1"
GM1_HRRR_LBC_THREADS = 128
GM1_HRRR_LBC_REQUIRED_ARRAYS = (
    "th0",
    "u3d",
    "v3d",
    "w3d",
    "th3d",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
)
GM1_HRRR_LBC_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
GM1_HRRR_HISTORY_ARRAYS = (
    "pi0",
    "pp3d",
    "prs0",
    "qc",
    "qg",
    "qi",
    "qr",
    "qs",
    "qv",
    "qv0",
    "refc_dbz",
    "refd_1000m_agl_dbz",
    "refd_4000m_agl_dbz",
    "rho0",
    "terrain_msl",
    "th0",
    "th3d",
    "u0",
    "u3d",
    "v0",
    "v3d",
    "w3d",
)
GM1_FORCING_HISTORY_EXPORT_SCHEMA = "gm1.forcing-history-export-manifest/v1"
GM1_FORCING_HISTORY_RAW_FILE = "arrays.f32le"
GM1_FORCING_HISTORY_TABLE_FILE = "arrays.tsv"
GM1_FORCING_HISTORY_MANIFEST_FILE = "manifest.json"


GM1_HRRR_LBC_KERNEL_NAMES = (
    "gm1_hrrr_lbc_u_v1",
    "gm1_hrrr_lbc_v_v1",
    "gm1_hrrr_lbc_w_v1",
    "gm1_hrrr_lbc_mass_v1",
    "gm1_hrrr_lbc_moisture_v1",
)


GM1_HRRR_LBC_CUDA_SOURCE = r"""// gm1.cuda-hrrr-lateral-boundary-kernels/v1
static_assert(sizeof(float)==4,"GM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float lbc_weight(unsigned int d,unsigned int width,float stage_factor,unsigned int c2_tendency){if(d>=width)return 0.0f;float x=__fdiv_rn((float)(width-d),(float)width);if(c2_tendency==0u)return __fmul_rn(x,x);if(d==0u)return 1.0f;float x2=m32(x,x),x3=m32(x2,x);float polynomial=a32(m32(x,a32(m32(6.0f,x),-15.0f)),10.0f);return m32(stage_factor,m32(x3,polynomial));}
__device__ __forceinline__ unsigned int min4(unsigned int a,unsigned int b,unsigned int c,unsigned int d){return min(min(a,b),min(c,d));}
extern "C" __global__ void gm1_hrrr_lbc_u_v1(
 float* current,const float* f00,const float* f01,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int width,float alpha,float stage_factor,unsigned int c2_tendency){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*(ni+1u);
 const unsigned long long cy=(unsigned long long)(nj+2u*ng),cx=(unsigned long long)(ni+1u+2u*ng);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%(ni+1u)),j=(unsigned int)((z/(ni+1u))%nj),k=(unsigned int)(z/((unsigned long long)(ni+1u)*nj));
  unsigned int d=min4(i,ni-i,j,nj-1u-j);float w=lbc_weight(d,width,stage_factor,c2_tendency);if(w>0.0f){unsigned long long c=((unsigned long long)(k+1u)*cy+(j+ng))*cx+(i+ng);float target=a32(f00[z],m32(alpha,s32(f01[z],f00[z])));current[c]=a32(current[c],m32(w,s32(target,current[c])));}}
}
extern "C" __global__ void gm1_hrrr_lbc_v_v1(
 float* current,const float* f00,const float* f01,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int width,float alpha,float stage_factor,unsigned int c2_tendency){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*(nj+1u)*ni;
 const unsigned long long cy=(unsigned long long)(nj+1u+2u*ng),cx=(unsigned long long)(ni+2u*ng);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%(nj+1u)),k=(unsigned int)(z/((unsigned long long)ni*(nj+1u)));
  unsigned int d=min4(i,ni-1u-i,j,nj-j);float w=lbc_weight(d,width,stage_factor,c2_tendency);if(w>0.0f){unsigned long long c=((unsigned long long)(k+1u)*cy+(j+ng))*cx+(i+ng);float target=a32(f00[z],m32(alpha,s32(f01[z],f00[z])));current[c]=a32(current[c],m32(w,s32(target,current[c])));}}
}
extern "C" __global__ void gm1_hrrr_lbc_w_v1(
 float* current,const float* f00,const float* f01,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int width,float alpha,float stage_factor,unsigned int c2_tendency){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;
 const unsigned long long cy=(unsigned long long)(nj+2u*ng),cx=(unsigned long long)(ni+2u*ng);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj));if(k==0u||k==nk)continue;
  unsigned int d=min4(i,ni-1u-i,j,nj-1u-j);float w=lbc_weight(d,width,stage_factor,c2_tendency);if(w>0.0f){unsigned long long c=((unsigned long long)(k+1u)*cy+(j+ng))*cx+(i+ng);float target=a32(f00[z],m32(alpha,s32(f01[z],f00[z])));current[c]=a32(current[c],m32(w,s32(target,current[c])));}}
}
extern "C" __global__ void gm1_hrrr_lbc_mass_v1(
 float* current,const float* f00,const float* f01,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int width,float alpha,float stage_factor,unsigned int c2_tendency){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj);unsigned int d=min4(i,ni-1u-i,j,nj-1u-j);float w=lbc_weight(d,width,stage_factor,c2_tendency);if(w>0.0f){float target=a32(f00[z],m32(alpha,s32(f01[z],f00[z])));current[z]=a32(current[z],m32(w,s32(target,current[z])));}}
}
extern "C" __global__ void gm1_hrrr_lbc_moisture_v1(
 float* current,const float* f00,const float* f01,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int nq,unsigned int width,float alpha,float stage_factor,unsigned int c2_tendency){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,plane=(unsigned long long)nk*nj*ni,n=(unsigned long long)nq*plane;
 for(unsigned long long z=first;z<n;z+=step){unsigned long long p=z%plane;unsigned int i=(unsigned int)(p%ni),j=(unsigned int)((p/ni)%nj);unsigned int d=min4(i,ni-1u-i,j,nj-1u-j);float w=lbc_weight(d,width,stage_factor,c2_tendency);if(w>0.0f){float target=a32(f00[z],m32(alpha,s32(f01[z],f00[z])));current[z]=fmaxf(0.0f,a32(current[z],m32(w,s32(target,current[z]))));}}
}
"""
GM1_HRRR_LBC_CUDA_SOURCE_BYTES = GM1_HRRR_LBC_CUDA_SOURCE.encode("utf-8")
GM1_HRRR_LBC_CUDA_SOURCE_SHA256 = (
    "cd992447257f086388e3f984febb628426e25dc05bbe92b28ed8c112d38df934"
)


def hrrr_lbc_edge_weight(distance: int, width: int) -> float:
    """Return the CUDA operator's exact mathematical edge-zone weight."""

    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("distance must be a nonnegative integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if distance >= width:
        return 0.0
    fraction = (width - distance) / width
    return fraction * fraction


def hrrr_lbc_davies_c2_spatial_weight(distance: int, width: int) -> float:
    """C2 spatial taper: exact one at the edge and exact zero inside."""

    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 0:
        raise ValueError("distance must be a nonnegative integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 2:
        raise ValueError("Davies width must be an integer of at least two points")
    if distance >= width:
        return 0.0
    x = (width - distance) / width
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def hrrr_lbc_davies_c2_weight(
    distance: int,
    width: int,
    *,
    stage_timestep_s: float,
    relaxation_timescale_s: float,
) -> float:
    """Exact-edge/C2-interior Davies factor for one CM1 RK stage."""

    spatial = hrrr_lbc_davies_c2_spatial_weight(distance, width)
    for value, label in (
        (stage_timestep_s, "stage timestep"),
        (relaxation_timescale_s, "relaxation timescale"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} must be finite and positive")
    if float(stage_timestep_s) > float(relaxation_timescale_s):
        raise ValueError("stage timestep may not exceed the relaxation timescale")
    if distance == 0:
        return 1.0
    return spatial * float(stage_timestep_s) / float(relaxation_timescale_s)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
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
class CompiledGm1HrrrLbcKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_hrrr_lbc() -> CompiledGm1HrrrLbcKernels:
    digest = hashlib.sha256(GM1_HRRR_LBC_CUDA_SOURCE_BYTES).hexdigest()
    if digest != GM1_HRRR_LBC_CUDA_SOURCE_SHA256:
        raise RuntimeError("packaged GM1 HRRR LBC CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        options += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        GM1_HRRR_LBC_CUDA_SOURCE, "gm1_hrrr_lbc_v1_ptx.cu", method="ptx"
    ).compile(options + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        GM1_HRRR_LBC_CUDA_SOURCE, "gm1_hrrr_lbc_v1_cubin.cu", method="cubin"
    ).compile(options + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in GM1_HRRR_LBC_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks GM1 HRRR LBC entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled GM1 HRRR LBC image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in GM1_HRRR_LBC_KERNEL_NAMES)
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
    return CompiledGm1HrrrLbcKernels(module, kernels, ptx, cubin, arch, identity)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"forcing-history manifest duplicates key {key!r}")
        result[key] = value
    return result


def _regular_nonsymlink(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _decode_f64_bits(value: object, label: str) -> float:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise ValueError(f"{label} must be 16 lower-case hexadecimal digits")
    decoded = struct.unpack(">d", int(value, 16).to_bytes(8, "big"))[0]
    if not math.isfinite(decoded):
        raise ValueError(f"{label} must encode a finite float64")
    return decoded


def load_hrrr_lbc_frame_export(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_raw_sha256: str,
    expected_artifact_sha256: str,
    expected_contract_sha256: str,
    expected_model_time_s: float = 3600.0,
    expected_step: int = 7200,
) -> HrrrLbcFrame:
    """Validate and mmap a Rust-owned forcing-history export.

    This function is intentionally an opaque transport boundary.  Rust has
    already decoded and validated the CM1FHS1 envelope; Python independently
    pins the immutable export, checks its exact carrier layout, and exposes
    read-only float32 views without recomputing any atmospheric quantity.
    """

    manifest_pin = _require_sha256(
        expected_manifest_sha256, "expected forcing manifest hash"
    )
    raw_pin = _require_sha256(expected_raw_sha256, "expected forcing raw hash")
    artifact_pin = _require_sha256(
        expected_artifact_sha256, "expected forcing artifact hash"
    )
    contract_pin = _require_sha256(
        expected_contract_sha256, "expected forcing contract hash"
    )
    root = Path(directory).absolute()
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect forcing-history export: {error}") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("forcing-history export must be a real directory")
    expected_files = {
        GM1_FORCING_HISTORY_MANIFEST_FILE,
        GM1_FORCING_HISTORY_RAW_FILE,
        GM1_FORCING_HISTORY_TABLE_FILE,
    }
    actual_files = {entry.name for entry in root.iterdir()}
    if actual_files != expected_files:
        raise ValueError("forcing-history export file inventory differs from v1")
    manifest_path = root / GM1_FORCING_HISTORY_MANIFEST_FILE
    raw_path = root / GM1_FORCING_HISTORY_RAW_FILE
    table_path = root / GM1_FORCING_HISTORY_TABLE_FILE
    for path, label in (
        (manifest_path, "forcing manifest"),
        (raw_path, "forcing arena"),
        (table_path, "forcing array table"),
    ):
        _regular_nonsymlink(path, label)
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_pin:
        raise ValueError("forcing-history manifest differs from its independent pin")
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("forcing-history manifest is not strict UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest_raw != _canonical_json_bytes(
        manifest
    ):
        raise ValueError("forcing-history manifest is not canonical JSON")
    required_top = {
        "arrays",
        "contract",
        "contract_sha256",
        "dtype",
        "model",
        "order",
        "publication_contract",
        "raw_file",
        "raw_nbytes",
        "raw_sha256",
        "schema",
        "source_history",
        "table_file",
        "table_sha256",
    }
    if set(manifest) != required_top:
        raise ValueError("forcing-history manifest key inventory differs from v1")
    if (
        manifest["schema"] != GM1_FORCING_HISTORY_EXPORT_SCHEMA
        or manifest["model"] != "GM1"
        or manifest["dtype"] != "<f4"
        or manifest["order"] != "C"
        or manifest["raw_file"] != GM1_FORCING_HISTORY_RAW_FILE
        or manifest["table_file"] != GM1_FORCING_HISTORY_TABLE_FILE
    ):
        raise ValueError("forcing-history manifest identity differs from v1")
    if manifest["contract_sha256"] != contract_pin:
        raise ValueError("forcing-history contract differs from its independent pin")
    if manifest["raw_sha256"] != raw_pin or _sha256_file(raw_path) != raw_pin:
        raise ValueError("forcing-history arena differs from its independent pin")
    source = manifest["source_history"]
    if not isinstance(source, dict):
        raise ValueError("forcing-history source receipt must be an object")
    if (
        isinstance(expected_model_time_s, bool)
        or not isinstance(expected_model_time_s, (int, float))
        or not math.isfinite(float(expected_model_time_s))
        or float(expected_model_time_s) <= 0.0
    ):
        raise ValueError("expected forcing model time must be finite and positive")
    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step < 1:
        raise ValueError("expected forcing step must be a positive integer")
    if (
        source.get("artifact_sha256") != artifact_pin
        or source.get("contract_sha256") != contract_pin
        or source.get("payload_sha256") != raw_pin
        or source.get("kind") != "forecast-history"
        or source.get("step") != expected_step
    ):
        raise ValueError("forcing-history source receipt differs from the trusted frame")
    model_time_s = _decode_f64_bits(
        source.get("model_time_f64_bits"), "forcing source model time"
    )
    if model_time_s != float(expected_model_time_s):
        raise ValueError("forcing-history source time differs from the expected frame")
    entries = manifest["arrays"]
    if not isinstance(entries, list) or len(entries) != len(GM1_HRRR_HISTORY_ARRAYS):
        raise ValueError("forcing-history carrier count differs from v1")
    numpy = importlib.import_module("numpy")
    arena = numpy.memmap(raw_path, dtype="u1", mode="r")
    raw_nbytes = int(manifest["raw_nbytes"])
    if raw_nbytes != raw_path.stat().st_size or raw_nbytes != int(arena.size):
        raise ValueError("forcing-history arena byte count differs from manifest")
    offset = 0
    all_arrays: dict[str, Any] = {}
    table_rows: list[str] = []
    for expected_name, entry in zip(GM1_HRRR_HISTORY_ARRAYS, entries, strict=True):
        if not isinstance(entry, dict) or set(entry) != {
            "dtype",
            "name",
            "nbytes",
            "offset_nbytes",
            "order",
            "sha256",
            "shape",
        }:
            raise ValueError("forcing-history carrier metadata differs from v1")
        name = entry["name"]
        shape_value = entry["shape"]
        if (
            name != expected_name
            or entry["dtype"] != "<f4"
            or entry["order"] != "C"
            or not isinstance(shape_value, list)
            or not shape_value
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape_value
            )
        ):
            raise ValueError("forcing-history carrier identity/shape differs from v1")
        shape = tuple(shape_value)
        words = math.prod(shape)
        nbytes = int(entry["nbytes"])
        entry_offset = int(entry["offset_nbytes"])
        field_sha256 = _require_sha256(entry["sha256"], f"forcing carrier {name} hash")
        if (
            nbytes != 4 * words
            or entry_offset != offset
            or offset + nbytes > raw_nbytes
        ):
            raise ValueError("forcing-history carrier byte layout is not contiguous")
        field_bytes = memoryview(arena)[offset : offset + nbytes]
        if hashlib.sha256(field_bytes).hexdigest() != field_sha256:
            raise ValueError(f"forcing-history carrier {name} differs from manifest")
        all_arrays[name] = numpy.ndarray(
            shape=shape,
            dtype="<f4",
            buffer=arena,
            offset=offset,
            order="C",
        )
        table_rows.append(
            f"{name}\t<f4\tC\t{','.join(map(str, shape))}\t{offset}\t{nbytes}\t{field_sha256}\n"
        )
        offset += nbytes
    if offset != raw_nbytes:
        raise ValueError("forcing-history carrier inventory does not consume the arena")
    table_raw = table_path.read_bytes()
    table_sha256 = _require_sha256(manifest["table_sha256"], "forcing array table hash")
    if hashlib.sha256(table_raw).hexdigest() != table_sha256 or table_raw != "".join(
        table_rows
    ).encode("ascii"):
        raise ValueError("forcing-history array table differs from manifest")
    nz = all_arrays["th0"].shape[0]
    ny, nx = all_arrays["terrain_msl"].shape
    expected_shapes = {
        "pi0": (nz,),
        "pp3d": (nz, ny, nx),
        "prs0": (nz,),
        "qv0": (nz,),
        "rho0": (nz,),
        "th0": (nz,),
        "u0": (nz,),
        "v0": (nz,),
        "terrain_msl": (ny, nx),
        "refc_dbz": (ny, nx),
        "refd_1000m_agl_dbz": (ny, nx),
        "refd_4000m_agl_dbz": (ny, nx),
        "u3d": (nz, ny, nx + 1),
        "v3d": (nz, ny + 1, nx),
        "w3d": (nz + 1, ny, nx),
        "th3d": (nz, ny, nx),
        **{name: (nz, ny, nx) for name in GM1_HRRR_LBC_SPECIES},
    }
    if any(all_arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("forcing-history staggered shape relationships differ from v1")
    selected = {name: all_arrays[name] for name in GM1_HRRR_LBC_REQUIRED_ARRAYS}
    result = HrrrLbcFrame(
        selected,
        artifact_pin,
        contract_pin,
        raw_pin,
        model_time_s,
    )
    result.validate(nx=nx, ny=ny, nz=nz)
    return result


def _require_array_shape(value: Any, shape: tuple[int, ...], label: str) -> None:
    if tuple(getattr(value, "shape", ())) != shape:
        raise ValueError(f"{label} shape must be {shape!r}")
    dtype = getattr(value, "dtype", None)
    if dtype is None or str(dtype) not in ("float32", "<f4"):
        raise ValueError(f"{label} must contain little-endian float32 values")


@dataclass(frozen=True, slots=True)
class HrrrLbcFrame:
    """Rust-validated host frame transported to the CUDA boundary operator."""

    arrays: Mapping[str, Any]
    artifact_sha256: str
    contract_sha256: str
    payload_sha256: str
    model_time_s: float

    def validate(self, *, nx: int, ny: int, nz: int) -> None:
        if set(self.arrays) != set(GM1_HRRR_LBC_REQUIRED_ARRAYS):
            raise ValueError(
                "HRRR LBC frame array inventory differs from the exact required set"
            )
        _require_sha256(self.artifact_sha256, "HRRR LBC artifact hash")
        _require_sha256(self.contract_sha256, "HRRR LBC contract hash")
        _require_sha256(self.payload_sha256, "HRRR LBC payload hash")
        if not math.isfinite(self.model_time_s) or self.model_time_s <= 0.0:
            raise ValueError("HRRR LBC model time must be finite and positive")
        shapes = {
            "th0": (nz,),
            "u3d": (nz, ny, nx + 1),
            "v3d": (nz, ny + 1, nx),
            "w3d": (nz + 1, ny, nx),
            "th3d": (nz, ny, nx),
            **{name: (nz, ny, nx) for name in GM1_HRRR_LBC_SPECIES},
        }
        for name, shape in shapes.items():
            _require_array_shape(self.arrays[name], shape, name)


@dataclass(slots=True)
class CudaGm1HrrrLbcState:
    nx: int
    ny: int
    nz: int
    ng: int
    width: int
    valid_offset_s: float
    initialization_source_offset_s: float
    source_start_offset_s: float
    source_end_offset_s: float
    f00_u: Any
    f01_u: Any
    f00_v: Any
    f01_v: Any
    f00_w: Any
    f01_w: Any
    f00_th: Any
    f01_th: Any
    f00_q: Any
    f01_q: Any
    artifact_sha256: str
    contract_sha256: str
    payload_sha256: str
    compiled: CompiledGm1HrrrLbcKernels
    stream: Any
    ready_event: Any
    relaxation_timescale_s: float | None = None
    retained: list[Any] = field(default_factory=list)

    @property
    def target_arch(self) -> str:
        return self.compiled.target_arch


def create_cuda_gm1_hrrr_lbc_state(
    forecast_state: Any,
    frame: HrrrLbcFrame,
    *,
    width: int = 20,
    initialization_source_offset_s: float | None = None,
    source_start_offset_s: float = 0.0,
    source_end_offset_s: float | None = None,
    relaxation_timescale_s: float | None = None,
) -> CudaGm1HrrrLbcState:
    """Copy one independently validated future driver frame to the forecast GPU.

    ``relaxation_timescale_s=None`` preserves the legacy direct quadratic
    blend.  A positive timescale selects the corrected exact-edge/C2-interior
    Davies tendency whose per-stage factor is CM1's ``dttmp / tau``.
    """

    selection = forecast_state.selection
    nx, ny, nz = selection.nx, selection.ny, selection.nz
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("HRRR LBC relaxation width must be a positive integer")
    if 2 * width > min(nx, ny):
        raise ValueError("HRRR LBC relaxation zones may not overlap the interior")
    if relaxation_timescale_s is not None:
        if (
            isinstance(relaxation_timescale_s, bool)
            or not isinstance(relaxation_timescale_s, (int, float))
            or not math.isfinite(float(relaxation_timescale_s))
            or float(relaxation_timescale_s) < float(selection.dt_s)
        ):
            raise ValueError(
                "Davies relaxation timescale must be finite and at least one large timestep"
            )
        if width < 2:
            raise ValueError("Davies C2 relaxation requires at least two points")
    frame.validate(nx=nx, ny=ny, nz=nz)
    if (
        isinstance(source_start_offset_s, bool)
        or not isinstance(source_start_offset_s, (int, float))
        or not math.isfinite(float(source_start_offset_s))
        or float(source_start_offset_s) < 0.0
    ):
        raise ValueError("HRRR LBC source start must be finite nonnegative seconds")
    initialization_source = (
        float(source_start_offset_s)
        if initialization_source_offset_s is None
        else initialization_source_offset_s
    )
    if (
        isinstance(initialization_source, bool)
        or not isinstance(initialization_source, (int, float))
        or not math.isfinite(float(initialization_source))
        or float(initialization_source) < 0.0
        or float(initialization_source) > float(source_start_offset_s)
    ):
        raise ValueError(
            "HRRR initialization source offset must be finite and no later than interval start"
        )
    source_end = frame.model_time_s if source_end_offset_s is None else source_end_offset_s
    if (
        isinstance(source_end, bool)
        or not isinstance(source_end, (int, float))
        or not math.isfinite(float(source_end))
        or float(source_end) <= float(source_start_offset_s)
        or float(source_end) != frame.model_time_s
    ):
        raise ValueError(
            "HRRR LBC source end must equal the trusted future frame time and exceed start"
        )
    cupy = importlib.import_module("cupy")
    stream = forecast_state.stream
    with stream:
        f00_u = forecast_state.product.field("u3d").copy()
        f00_v = forecast_state.product.field("v3d").copy()
        f00_w = forecast_state.product.field("w3d").copy()
        f00_th = forecast_state.product.field("th3d").copy()
        f00_q = forecast_state.product.field("q3d").copy()
        f01_u = cupy.asarray(frame.arrays["u3d"])
        f01_v = cupy.asarray(frame.arrays["v3d"])
        f01_w = cupy.asarray(frame.arrays["w3d"])
        f01_th = cupy.asarray(frame.arrays["th3d"])
        f01_th = f01_th + cupy.asarray(frame.arrays["th0"])[:, None, None]
        f01_th = f01_th - forecast_state.product.field("th0")[:, None, None]
        f01_q = cupy.stack(
            [cupy.asarray(frame.arrays[name]) for name in GM1_HRRR_LBC_SPECIES]
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    compiled = compile_cuda_gm1_hrrr_lbc()
    if compiled.target_arch not in (
        {forecast_state.compiled.target_arch} | forecast_state.closure.target_arches
    ):
        raise ValueError(
            "HRRR LBC and forecast CUDA images target different architectures"
        )
    return CudaGm1HrrrLbcState(
        nx,
        ny,
        nz,
        CM1_OPEN_NGXY,
        width,
        frame.model_time_s,
        float(initialization_source),
        float(source_start_offset_s),
        float(source_end),
        f00_u,
        f01_u,
        f00_v,
        f01_v,
        f00_w,
        f01_w,
        f00_th,
        f01_th,
        f00_q,
        f01_q,
        frame.artifact_sha256,
        frame.contract_sha256,
        frame.payload_sha256,
        compiled,
        stream,
        event,
        None if relaxation_timescale_s is None else float(relaxation_timescale_s),
    )


def resolve_hrrr_lbc_stage_clock(
    lbc: CudaGm1HrrrLbcState,
    forecast_state: Any,
    *,
    nrk: int,
    clock_point: Any | None = None,
) -> dict[str, float | str]:
    """Resolve and independently audit the interpolation clock without CUDA."""

    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    interval = (
        lbc.initialization_source_offset_s,
        lbc.source_start_offset_s,
        lbc.source_end_offset_s,
        lbc.valid_offset_s,
    )
    if (
        any(not math.isfinite(float(value)) for value in interval)
        or lbc.initialization_source_offset_s < 0.0
        or lbc.initialization_source_offset_s > lbc.source_start_offset_s
        or lbc.source_start_offset_s >= lbc.source_end_offset_s
        or lbc.valid_offset_s != lbc.source_end_offset_s
    ):
        refuse(
            code="GM1-HRRR-LBC-SOURCE-INTERVAL",
            gate="boundaries-closures",
            option="source_interval_s",
            value=interval,
            reason="the bound LBC state has an invalid or inconsistent source interval",
            remedy="bind initialization <= interval start < interval end and pin the future frame time",
        )
    if clock_point is None:
        if lbc.source_start_offset_s != 0.0:
            refuse(
                code="GM1-HRRR-LBC-EXPLICIT-CLOCK-REQUIRED",
                gate="boundaries-closures",
                option="source_start_offset_s",
                value=lbc.source_start_offset_s,
                reason="a rebased HRRR interval cannot use the legacy time-zero interpolation",
                remedy="pass the coordinator's explicit CM1 RK-stage clock point",
            )
        relative = forecast_state.time_seconds + forecast_state.selection.dt_s
        if relative > lbc.valid_offset_s + 1.0e-9:
            raise ValueError(
                "real HRRR LBC forcing cannot run past its final independently supplied frame"
            )
        return {
            "relative_model_time_s": relative,
            "absolute_source_time_s": relative,
            "temporal_interpolation_fraction": min(
                1.0, max(0.0, relative / lbc.valid_offset_s)
            ),
            "clock_mode": "legacy-f00-f01-end-step",
        }

    def clock_value(name: str) -> float:
        value = (
            clock_point.get(name)
            if isinstance(clock_point, Mapping)
            else getattr(clock_point, name, None)
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            refuse(
                code="GM1-HRRR-LBC-EXPLICIT-CLOCK",
                gate="boundaries-closures",
                option=name,
                value=value,
                reason="the explicit stage clock is incomplete or nonfinite",
                remedy="pass relative_model_time_s, absolute_source_time_s, and alpha",
            )
        return float(value)

    relative = clock_value("relative_model_time_s")
    absolute = clock_value("absolute_source_time_s")
    alpha = clock_value("temporal_interpolation_fraction")
    expected_relative = forecast_state.time_seconds + float(
        momentum_rk_stage_timestep_f32(forecast_state.selection.dt_s, nrk)
    )
    expected_absolute = lbc.initialization_source_offset_s + expected_relative
    expected_alpha = (
        expected_absolute - lbc.source_start_offset_s
    ) / (lbc.source_end_offset_s - lbc.source_start_offset_s)
    if (
        abs(relative - expected_relative) > 1.0e-9
        or abs(absolute - expected_absolute) > 1.0e-9
        or abs(alpha - expected_alpha) > 2.0e-7
    ):
        refuse(
            code="GM1-HRRR-LBC-EXPLICIT-CLOCK-MISMATCH",
            gate="boundaries-closures",
            option="clock_point",
            value={
                "relative_model_time_s": relative,
                "absolute_source_time_s": absolute,
                "alpha": alpha,
            },
            reason="the boundary clock does not match CM1's dt/(4-nrk) stage time",
            remedy="use the exact clock point supplied by the GM1 forecast coordinator",
        )
    if (
        absolute < lbc.source_start_offset_s
        or absolute > lbc.source_end_offset_s
        or alpha < 0.0
        or alpha > 1.0
    ):
        refuse(
            code="GM1-HRRR-LBC-EXPLICIT-CLOCK-RANGE",
            gate="boundaries-closures",
            option="clock_point",
            value={
                "absolute_source_time_s": absolute,
                "source_start_offset_s": lbc.source_start_offset_s,
                "source_end_offset_s": lbc.source_end_offset_s,
            },
            reason="the CM1 stage lies outside the supplied HRRR source interval",
            remedy="supply the next contiguous boundary frame before advancing",
        )
    return {
        "relative_model_time_s": relative,
        "absolute_source_time_s": absolute,
        "temporal_interpolation_fraction": alpha,
        "clock_mode": "explicit-cm1-rk-stage",
    }


def launch_cuda_gm1_hrrr_lbc_stage(
    lbc: CudaGm1HrrrLbcState,
    forecast_state: Any,
    *,
    nrk: int,
    clock_point: Any | None = None,
) -> dict[str, Any]:
    """Apply one source interval after a GM1 RK stage.

    Omitting ``clock_point`` preserves the original f00-to-f01 end-step clock.
    Nonzero source starts require the coordinator's explicit CM1 stage point.
    """

    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if forecast_state.stream is not lbc.stream:
        raise ValueError("HRRR LBC and forecast state must share one CUDA stream")
    if (lbc.nx, lbc.ny, lbc.nz) != (
        forecast_state.selection.nx,
        forecast_state.selection.ny,
        forecast_state.selection.nz,
    ):
        raise ValueError("HRRR LBC and forecast geometry differ")
    clock = resolve_hrrr_lbc_stage_clock(
        lbc, forecast_state, nrk=nrk, clock_point=clock_point
    )
    model_time_s = float(clock["relative_model_time_s"])
    absolute_source_time_s = float(clock["absolute_source_time_s"])
    alpha = float(clock["temporal_interpolation_fraction"])
    clock_mode = str(clock["clock_mode"])
    cupy = importlib.import_module("cupy")
    numpy = importlib.import_module("numpy")
    stage_timestep_s = float(
        momentum_rk_stage_timestep_f32(forecast_state.selection.dt_s, nrk)
    )
    if lbc.relaxation_timescale_s is None:
        relaxation_mode = "legacy-direct-quadratic"
        stage_factor = 1.0
        c2_tendency = 0
        inner_spatial = hrrr_lbc_edge_weight(lbc.width - 1, lbc.width)
    else:
        relaxation_mode = "davies-c2-stage-tendency"
        stage_factor = float(
            numpy.float32(
                numpy.float32(stage_timestep_s)
                / numpy.float32(lbc.relaxation_timescale_s)
            )
        )
        if not 0.0 < stage_factor <= 1.0:
            raise ValueError("Davies CM1 RK-stage relaxation factor must lie in (0,1]")
        c2_tendency = 1
        inner_spatial = hrrr_lbc_davies_c2_spatial_weight(
            lbc.width - 1, lbc.width
        )
    inner_effective = (
        inner_spatial if c2_tendency == 0 else inner_spatial * stage_factor
    )
    nx, ny, nz = lbc.nx, lbc.ny, lbc.nz
    threads = (GM1_HRRR_LBC_THREADS, 1, 1)

    def grid(words: int) -> tuple[int, int, int]:
        return (
            (words + GM1_HRRR_LBC_THREADS - 1) // GM1_HRRR_LBC_THREADS,
            1,
            1,
        )

    arrays = forecast_state.prepared.arrays
    with lbc.stream:
        common = (
            cupy.uint32(nx),
            cupy.uint32(ny),
            cupy.uint32(nz),
            cupy.uint32(lbc.ng),
            cupy.uint32(lbc.width),
            cupy.float32(alpha),
            cupy.float32(stage_factor),
            cupy.uint32(c2_tendency),
        )
        lbc.compiled.kernels[0](
            grid(nz * ny * (nx + 1)),
            threads,
            (arrays["u3d"], lbc.f00_u, lbc.f01_u, *common),
            stream=lbc.stream,
        )
        lbc.compiled.kernels[1](
            grid(nz * (ny + 1) * nx),
            threads,
            (arrays["v3d"], lbc.f00_v, lbc.f01_v, *common),
            stream=lbc.stream,
        )
        lbc.compiled.kernels[2](
            grid((nz + 1) * ny * nx),
            threads,
            (arrays["w3d"], lbc.f00_w, lbc.f01_w, *common),
            stream=lbc.stream,
        )
        mass_common = (
            cupy.uint32(nx),
            cupy.uint32(ny),
            cupy.uint32(nz),
            cupy.uint32(lbc.width),
            cupy.float32(alpha),
            cupy.float32(stage_factor),
            cupy.uint32(c2_tendency),
        )
        lbc.compiled.kernels[3](
            grid(nz * ny * nx),
            threads,
            (
                forecast_state.product.field("th3d"),
                lbc.f00_th,
                lbc.f01_th,
                *mass_common,
            ),
            stream=lbc.stream,
        )
        lbc.compiled.kernels[4](
            grid(6 * nz * ny * nx),
            threads,
            (
                forecast_state.product.field("q3d"),
                lbc.f00_q,
                lbc.f01_q,
                cupy.uint32(nx),
                cupy.uint32(ny),
                cupy.uint32(nz),
                cupy.uint32(6),
                cupy.uint32(lbc.width),
                cupy.float32(alpha),
                cupy.float32(stage_factor),
                cupy.uint32(c2_tendency),
            ),
            stream=lbc.stream,
        )
        # Rebuild the source-ordered CM1 family halos immediately.  Momentum
        # advection for the next RK stage precedes the next acoustic open-BC
        # call, so leaving stale halos here would delay the external driver by
        # one stage and contaminate the edge-zone stencil.
        family_we, family_sn, rigid_w = forecast_state.open_compiled.kernels[5:8]
        for name, kind in (("u3d", 1), ("v3d", 2), ("w3d", 3)):
            family_grid = grid(int(arrays[name].size))
            family_we(
                family_grid,
                threads,
                (
                    arrays[name],
                    cupy.uint32(nx),
                    cupy.uint32(ny),
                    cupy.uint32(nz),
                    cupy.uint32(lbc.ng),
                    cupy.uint32(kind),
                ),
                stream=lbc.stream,
            )
            family_sn(
                family_grid,
                threads,
                (
                    arrays[name],
                    cupy.uint32(nx),
                    cupy.uint32(ny),
                    cupy.uint32(nz),
                    cupy.uint32(lbc.ng),
                    cupy.uint32(kind),
                ),
                stream=lbc.stream,
            )
        rigid_w(
            grid((ny + 2) * (nx + 2)),
            threads,
            (
                arrays["w3d"],
                cupy.uint32(nx),
                cupy.uint32(ny),
                cupy.uint32(nz),
                cupy.uint32(lbc.ng),
            ),
            stream=lbc.stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(lbc.stream)
    lbc.ready_event = event
    receipt = {
        "schema": GM1_HRRR_LBC_SCHEMA,
        "model": "GM1",
        "role": "real-HRRR time-varying nested lateral driver; not CM1 lsnudge",
        "nrk": nrk,
        "target_model_time_s": model_time_s,
        "target_absolute_source_time_s": absolute_source_time_s,
        "temporal_interpolation_fraction": alpha,
        "clock_mode": clock_mode,
        "relaxation_width_points": lbc.width,
        "relaxation_mode": relaxation_mode,
        "relaxation_timescale_s": lbc.relaxation_timescale_s,
        "rk_stage_timestep_s": stage_timestep_s,
        "stage_relaxation_factor": stage_factor,
        "edge_weight": 1.0,
        "inner_zone_spatial_weight": inner_spatial,
        "inner_zone_effective_weight": inner_effective,
        # Compatibility alias.  In Davies mode this is the effective
        # per-stage tendency factor, never the spatial taper alone.
        "inner_zone_weight": inner_effective,
        "forced_fields": ["u", "v", "w-interior-faces", "theta", *GM1_HRRR_LBC_SPECIES],
        "unforced_fields": [
            "pressure",
            "interior-outside-edge-zone",
            "rigid-bottom/top-w",
        ],
        "halo_cadence": "CM1 family west/east then south/north fills, followed by rigid w faces",
        "source": {
            "artifact_sha256": lbc.artifact_sha256,
            "contract_sha256": lbc.contract_sha256,
            "payload_sha256": lbc.payload_sha256,
            "valid_offset_s": lbc.valid_offset_s,
            "interval_start_offset_s": lbc.source_start_offset_s,
            "interval_end_offset_s": lbc.source_end_offset_s,
            "initialization_source_offset_s": lbc.initialization_source_offset_s,
            "synthetic_atmospheric_fields": False,
        },
        "kernel": {
            "abi": GM1_HRRR_LBC_ABI,
            "source_sha256": GM1_HRRR_LBC_CUDA_SOURCE_SHA256,
            "target_arch": lbc.compiled.target_arch,
        },
        "ready_event_recorded": True,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return receipt


__all__ = [
    "CompiledGm1HrrrLbcKernels",
    "CudaGm1HrrrLbcState",
    "GM1_HRRR_LBC_ABI",
    "GM1_HRRR_LBC_CUDA_SOURCE",
    "GM1_HRRR_LBC_CUDA_SOURCE_SHA256",
    "GM1_HRRR_LBC_KERNEL_NAMES",
    "GM1_HRRR_LBC_REQUIRED_ARRAYS",
    "GM1_HRRR_LBC_SCHEMA",
    "GM1_HRRR_HISTORY_ARRAYS",
    "HrrrLbcFrame",
    "compile_cuda_gm1_hrrr_lbc",
    "create_cuda_gm1_hrrr_lbc_state",
    "hrrr_lbc_davies_c2_spatial_weight",
    "hrrr_lbc_davies_c2_weight",
    "hrrr_lbc_edge_weight",
    "load_hrrr_lbc_frame_export",
    "launch_cuda_gm1_hrrr_lbc_stage",
    "resolve_hrrr_lbc_stage_clock",
]
