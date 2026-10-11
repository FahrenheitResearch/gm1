"""Source-ordered tiled CM1 pdefq repair for the stretched GM1 child.

The selected WSM6 path calls ``pdefq`` once for each of six moisture species
after the completed final-RK update and ``apmasscon``, immediately before
WSM6.  Each vertical column is independent, but the diagnostic correction
ledger is not an arbitrary reduction: CM1 carries ``budj`` west-to-east with
the exact left-associated expression ``(budj + a2) - a1`` and then accumulates
``asq`` south-to-north.  This module tiles only the column work.  A small
domain-wide float64 ``[6,ny]`` budget workspace preserves the source order.

Terrain-following ``rmh`` is reconstructed inside the tile kernel from the
sealed terrain carrier and stretch-z1 ``sigmaf`` profile with the same f32
statement order as CM1's admitted static-metric operator.  No full-domain rmh
duplicate is introduced.  WSM6, qa publication, solve3, pressure, halos, and
MPI reductions remain outside this boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_sound3_periodic import _canonical_arch, _image_options
from .cuda_wsm6_pdefq import (
    AUTHORITY_SOURCE_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_WSM6_PDEFQ_SOURCE_SHA256,
    CompiledWsm6Pdefq,
    DEFAULT_WSM6_PDEFQ_SELECTION,
    SOURCE_ANCHORS,
    WSM6_SPECIES_ORDER,
    Wsm6PdefqSelection,
    compile_cuda_wsm6_pdefq,
    require_wsm6_pdefq_selection,
)
from .stretch_z1_child_15km import (
    STRETCH_Z1_CHILD_15KM_GRID_SHA256,
    STRETCH_Z1_CHILD_15KM_PARAMETERS,
    STRETCH_Z1_CHILD_15KM_PROOF_SHA256,
    build_stretch_z1_child_15km_grid,
)


CUDA_GM1_TILESTREAM_PDEFQ_SCHEMA = "gm1.cuda-tilestream-wsm6-pdefq/v1"
CUDA_GM1_TILESTREAM_PDEFQ_ABI = "gm1.cuda-tilestream-wsm6-pdefq-f32-f64/v1"
CUDA_GM1_TILESTREAM_PDEFQ_KERNEL = "gm1_tiled_wsm6_pdefq_chunk_f32_v1"
GM1_UNIFORM_HORIZONTAL_METRIC_SHA256 = (
    "9daebe2966c6a8065d691f81625f3cc8ba345f05f44b143fc2e31e5c59d72b5c"
)
GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT = (
    "CM1 r21.1 selected WSM6 pdscheme=1 pdefq column repair on exact physical "
    "q3d/rho/terrain with serial west-to-east budj carry and one source-order "
    "signed-f64 asq finish; no qa mutation, WSM6, solve3, pressure, halo/open "
    "history mutation, MPI reduction, five-meter meteorological state, or "
    "forecast claim"
)


def _canonical(value: object) -> bytes:
    import json

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def tiled_pdefq_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact per-buffer inventory; global budget/sigma profiles are separate."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    mass = (nk, tile_ny, tile_nx)
    return MappingProxyType(
        {
            "q3d": mass,
            "rho": mass,
            "terrain_msl": (tile_ny, tile_nx),
            "ruh": (tile_nx,),
            "rvh": (tile_ny,),
        }
    )


@dataclass(frozen=True, slots=True)
class Gm1TiledPdefqSourceBinding:
    """Read/write physical state and sealed metric sources for tile fills."""

    q3d: Any
    rho: Any
    terrain_msl: Any
    ruh: Any
    rvh: Any
    sigmaf: Any
    nx: int
    ny: int
    nk: int
    source_identity_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        q3d: Any,
        rho: Any,
        terrain_msl: Any,
        ruh: Any,
        rvh: Any,
        sigmaf: Any,
        source_identity_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledPdefqSourceBinding":
        np = importlib.import_module("numpy")
        if (
            not isinstance(q3d, np.ndarray)
            or q3d.dtype != np.dtype("<f4")
            or q3d.ndim != 4
            or q3d.shape[0] != 6
            or not q3d.flags.c_contiguous
            or not q3d.flags.writeable
        ):
            raise ValueError("q3d must be writable host C-<f4 [6,nk,ny,nx]")
        _, nk, ny, nx = map(int, q3d.shape)
        expected = {
            "rho": (nk, ny, nx),
            "terrain_msl": (ny, nx),
            "ruh": (nx,),
            "rvh": (ny,),
            "sigmaf": (nk + 3,),
        }
        values = {
            "rho": rho,
            "terrain_msl": terrain_msl,
            "ruh": ruh,
            "rvh": rvh,
            "sigmaf": sigmaf,
        }
        for name, shape in expected.items():
            value = values[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be host C-<f4 {shape}")
        if any(np.shares_memory(q3d, value) for value in values.values()):
            raise ValueError("pdefq q3d must not alias metric or density inputs")
        if not all(bool(np.all(np.isfinite(value))) for value in (q3d, *values.values())):
            raise ValueError("pdefq source contains non-finite words")
        if not bool(np.all(rho > np.float32(0.0))):
            raise ValueError("rho must be strictly positive")
        if not bool(np.all(ruh > np.float32(0.0))) or not bool(
            np.all(rvh > np.float32(0.0))
        ):
            raise ValueError("ruh and rvh must be strictly positive")
        if nk != STRETCH_Z1_CHILD_15KM_PARAMETERS.nk:
            raise ValueError("tiled pdefq is sealed to the exact 75-level child")
        grid = build_stretch_z1_child_15km_grid()
        if not np.array_equal(sigmaf.view(np.uint32), grid.sigmaf.view(np.uint32)):
            raise ValueError("sigmaf differs from the sealed stretch-z1 child profile")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if allow_compact_validation:
            if not bool(np.all(ruh == np.float32(1.0))) or not bool(
                np.all(rvh == np.float32(1.0))
            ):
                raise ValueError("compact child ruh/rvh must remain exact uniform metrics")
        elif (nx, ny) != (300, 300):
            raise ValueError("real tiled pdefq source must be the accepted 300-square child")
        elif hashlib.sha256(ruh.tobytes(order="C")).hexdigest() != (
            GM1_UNIFORM_HORIZONTAL_METRIC_SHA256
        ) or hashlib.sha256(rvh.tobytes(order="C")).hexdigest() != (
            GM1_UNIFORM_HORIZONTAL_METRIC_SHA256
        ):
            raise ValueError("selected child ruh/rvh must be exact uniform metrics")
        top = np.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop)
        if not bool(np.all(terrain_msl < top)):
            raise ValueError("terrain reaches or exceeds the sealed model top")
        return cls(
            q3d,
            rho,
            terrain_msl,
            ruh,
            rvh,
            sigmaf,
            nx,
            ny,
            nk,
            _require_sha256("source_identity_sha256", source_identity_sha256),
            allow_compact_validation,
        )


def fill_gm1_tiled_pdefq_source(
    binding: Gm1TiledPdefqSourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
    *,
    species: int,
) -> dict[str, object]:
    """Fill one fixed tile buffer from its unique physical owned rectangle."""

    if not isinstance(binding, Gm1TiledPdefqSourceBinding):
        raise TypeError("binding must be Gm1TiledPdefqSourceBinding")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if (spec.nx, spec.ny) != (binding.nx, binding.ny):
        raise ValueError("tile and pdefq source domains differ")
    if isinstance(species, bool) or not isinstance(species, int) or species not in range(6):
        raise ValueError("species must index qv,qc,qr,qi,qs,qg")
    np = importlib.import_module("numpy")
    shapes = tiled_pdefq_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if tuple(target) != tuple(shapes):
        raise ValueError("tiled pdefq target inventory or order changed")
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"tiled pdefq target {name} must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))
    y0, y1, x0, x1 = spec.owned_rect("mass")
    height, width = y1 - y0, x1 - x0
    np.copyto(target["q3d"][:, :height, :width], binding.q3d[species, :, y0:y1, x0:x1])
    np.copyto(target["rho"][:, :height, :width], binding.rho[:, y0:y1, x0:x1])
    np.copyto(target["terrain_msl"][:height, :width], binding.terrain_msl[y0:y1, x0:x1])
    np.copyto(target["ruh"][:width], binding.ruh[x0:x1])
    np.copyto(target["rvh"][:height], binding.rvh[y0:y1])
    return {
        "schema": "gm1.hybrid-tilestream-pdefq-source/v1",
        "source_identity_sha256": binding.source_identity_sha256,
        "tile": spec.index,
        "species": species,
        "species_name": WSM6_SPECIES_ORDER[species],
        "owned_yx": [y0, y1, x0, x1],
        "active_nx": width,
        "active_ny": height,
        "rmh_disposition": (
            "reconstruct per column from terrain_msl and sealed sigmaf; no "
            "full-domain rmh duplicate"
        ),
        "qa_mutated": False,
        "wsm6_performed": False,
    }


def scatter_gm1_tiled_pdefq_output(
    binding: Gm1TiledPdefqSourceBinding,
    spec: Gm1TileSpec,
    tile_q3d: Any,
    *,
    species: int,
) -> dict[str, object]:
    """Publish exactly the physical rectangle processed by one pdefq chunk."""

    if not isinstance(binding, Gm1TiledPdefqSourceBinding):
        raise TypeError("binding must be Gm1TiledPdefqSourceBinding")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if isinstance(species, bool) or not isinstance(species, int) or species not in range(6):
        raise ValueError("species must index qv,qc,qr,qi,qs,qg")
    np = importlib.import_module("numpy")
    expected = (binding.nk, spec.tile_ny, spec.tile_nx)
    if (
        not isinstance(tile_q3d, np.ndarray)
        or tile_q3d.dtype != np.dtype("<f4")
        or tile_q3d.shape != expected
        or not tile_q3d.flags.c_contiguous
    ):
        raise ValueError(f"tile_q3d must be host C-<f4 {expected}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    height, width = y1 - y0, x1 - x0
    destination = binding.q3d[species, :, y0:y1, x0:x1]
    np.copyto(destination, tile_q3d[:, :height, :width])
    return {
        "schema": "gm1.hybrid-tilestream-pdefq-scatter/v1",
        "tile": spec.index,
        "species": species,
        "owned_yx": [y0, y1, x0, x1],
        "published_words": int(destination.size),
        "qa_mutated": False,
        "halo_or_open_history_mutated": False,
    }


def _m(np: Any, left: Any, right: Any) -> Any:
    return np.multiply(left, right, dtype=np.float32)


def _a(np: Any, left: Any, right: Any) -> Any:
    return np.add(left, right, dtype=np.float32)


def _s(np: Any, left: Any, right: Any) -> Any:
    return np.subtract(left, right, dtype=np.float32)


def _d(np: Any, left: Any, right: Any) -> Any:
    return np.divide(left, right, dtype=np.float32)


def _rmh_at(
    np: Any, surface: Any, sigma_lo: Any, sigma_hi: Any, top: Any, dz: Any
) -> Any:
    remaining = _s(np, top, surface)
    zlo = _a(np, surface, _d(np, _m(np, sigma_lo, remaining), top))
    zhi = _a(np, surface, _d(np, _m(np, sigma_hi, remaining), top))
    mh = _d(np, dz, _s(np, zhi, zlo))
    return _d(np, np.float32(1.0), mh)


def cpu_gm1_tiled_wsm6_pdefq_pdscheme1_inplace(
    binding: Gm1TiledPdefqSourceBinding,
    plan: Gm1TilePlan,
    asq: Any,
    *,
    dx: float,
    dy: float,
    dz: float,
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
) -> dict[str, Any]:
    """Source-order CPU oracle for the tiled carry and terrain reconstruction."""

    if not isinstance(binding, Gm1TiledPdefqSourceBinding):
        raise TypeError("binding must be Gm1TiledPdefqSourceBinding")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("plan must cover the bound pdefq source domain")
    require_wsm6_pdefq_selection(selection)
    np = importlib.import_module("numpy")
    if (
        not isinstance(asq, np.ndarray)
        or asq.dtype != np.dtype("<f8")
        or asq.shape != (6,)
        or not asq.flags.c_contiguous
        or not asq.flags.writeable
        or not bool(np.all(np.isfinite(asq)))
    ):
        raise ValueError("asq must be finite writable host C-<f8 [6]")
    spacing = tuple(np.float32(value) for value in (dx, dy, dz))
    if not all(math.isfinite(float(value)) and value > np.float32(0.0) for value in spacing):
        raise ValueError("dx, dy, and dz must be positive finite float32")
    volume = _m(np, _m(np, spacing[0], spacing[1]), spacing[2])
    top = np.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop)
    budgets = np.empty((6, binding.ny), dtype=np.float64)
    corrected = 0
    for species in range(6):
        threshold = np.float32(0.0 if species == 0 else selection.qsmall)
        q = binding.q3d[species]
        for spec in plan.specs:
            y0, y1, x0, x1 = spec.owned_rect("mass")
            for j in range(y0, y1):
                budj = 0.0 if x0 == 0 else float(budgets[species, j])
                for i in range(x0, x1):
                    t1 = t2 = a1 = a2 = 0.0
                    surface = binding.terrain_msl[j, i]
                    for k in range(binding.nk):
                        p = (k, j, i)
                        word = q[p]
                        t1 += float(_m(np, binding.rho[p], word))
                        rmh = _rmh_at(
                            np,
                            surface,
                            binding.sigmaf[k + 1],
                            binding.sigmaf[k + 2],
                            top,
                            spacing[2],
                        )
                        cell = _m(
                            np,
                            _m(
                                np,
                                _m(
                                    np,
                                    _m(np, binding.rho[p], word),
                                    binding.ruh[i],
                                ),
                                binding.rvh[j],
                            ),
                            rmh,
                        )
                        a1 += float(cell)
                        if word < threshold:
                            word = np.float32(0.0)
                            q[p] = word
                            corrected += 1
                        t2 += float(_m(np, binding.rho[p], word))
                    t3 = (t1 + 1.0e-20) / (t2 + 1.0e-20)
                    if t3 < 0.0:
                        t3 = 1.0
                    for k in range(binding.nk):
                        p = (k, j, i)
                        q[p] = np.float32(t3 * float(q[p]))
                        rmh = _rmh_at(
                            np,
                            surface,
                            binding.sigmaf[k + 1],
                            binding.sigmaf[k + 2],
                            top,
                            spacing[2],
                        )
                        cell = _m(
                            np,
                            _m(
                                np,
                                _m(
                                    np,
                                    _m(np, binding.rho[p], q[p]),
                                    binding.ruh[i],
                                ),
                                binding.rvh[j],
                            ),
                            rmh,
                        )
                        a2 += float(cell)
                    budj = (budj + a2) - a1
                budgets[species, j] = budj
    for species in range(6):
        for j in range(binding.ny):
            asq[species] += budgets[species, j] * float(volume)
    return {
        "schema": CUDA_GM1_TILESTREAM_PDEFQ_SCHEMA,
        "source_order": [
            "species qv,qc,qr,qi,qs,qg",
            "tile rows south-to-north and tiles west-to-east",
            "budj=(budj+a2)-a1 carried without regrouping",
            "asq species then physical j",
        ],
        "corrected_input_words": corrected,
        "budget_columns": budgets,
        "claim_limit": GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT,
    }


CUDA_GM1_TILESTREAM_PDEFQ_SOURCE = r'''// gm1.cuda-tilestream-wsm6-pdefq-f32-f64/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(double)==8,"float64 required");
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ float TH(float s,float sig,float top){return A(s,D(M(sig,S(top,s)),top));}
extern "C" __global__ void gm1_tiled_wsm6_pdefq_chunk_f32_v1(
 unsigned species,unsigned active_nx,unsigned active_ny,unsigned nk,
 unsigned pitch_nx,unsigned pitch_ny,unsigned global_y0,unsigned global_ny,
 unsigned reset_west,float top,float dz,const float*ruh,const float*rvh,
 const float*terrain,const float*sigmaf,const float*rho,float*q3d,double*budgets){
 unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=active_ny)return;
 unsigned gj=global_y0+j;double budj=reset_west?0.0:budgets[(unsigned long long)species*global_ny+gj];
 float rmax=species?1.0e-14f:0.0f;
 for(unsigned i=0;i<active_nx;++i){double t1=0.0,t2=0.0,a1=0.0,a2=0.0;float s=terrain[(unsigned long long)j*pitch_nx+i];
  for(unsigned k=0;k<nk;++k){unsigned long long p=((unsigned long long)k*pitch_ny+j)*pitch_nx+i;float q=q3d[p];t1=__dadd_rn(t1,double(M(rho[p],q)));float z0=TH(s,sigmaf[k+1u],top),z1=TH(s,sigmaf[k+2u],top);float mh=D(dz,S(z1,z0)),rmh=D(1.0f,mh);float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh);a1=__dadd_rn(a1,double(cell));if(q<rmax){q=0.0f;q3d[p]=q;}t2=__dadd_rn(t2,double(M(rho[p],q)));}
  double t3=__ddiv_rn(__dadd_rn(t1,1.0e-20),__dadd_rn(t2,1.0e-20));if(t3<0.0)t3=1.0;
  for(unsigned k=0;k<nk;++k){unsigned long long p=((unsigned long long)k*pitch_ny+j)*pitch_nx+i;float q=__double2float_rn(__dmul_rn(t3,double(q3d[p])));q3d[p]=q;float z0=TH(s,sigmaf[k+1u],top),z1=TH(s,sigmaf[k+2u],top);float mh=D(dz,S(z1,z0)),rmh=D(1.0f,mh);float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh);a2=__dadd_rn(a2,double(cell));}
  budj=__dsub_rn(__dadd_rn(budj,a2),a1);
 }
 budgets[(unsigned long long)species*global_ny+gj]=budj;
}
'''
CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES = CUDA_GM1_TILESTREAM_PDEFQ_SOURCE.encode(
    "utf-8"
)
CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256 = (
    "cd26f4ee08719a6459fe112097d13ac336a143ed9868f6cdbb8007dad1a059b5"
)


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledPdefq:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_wsm6_pdefq() -> CompiledGm1TiledPdefq:
    actual = hashlib.sha256(CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES).hexdigest()
    if actual != CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256:
        raise RuntimeError(f"packaged tiled pdefq CUDA source differs from pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_PDEFQ_SOURCE,
        "gm1_tiled_wsm6_pdefq_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_PDEFQ_SOURCE,
        "gm1_tiled_wsm6_pdefq_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(rf"\.entry\s+{CUDA_GM1_TILESTREAM_PDEFQ_KERNEL}(?:\s|\()", ptx.decode("utf-8")):
        raise ValueError("compiled PTX lacks the tiled pdefq entry point")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1TiledPdefq(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_PDEFQ_KERNEL),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(slots=True)
class CudaGm1TiledPdefqWorkspace:
    nx: int
    ny: int
    nk: int
    tile_count: int
    plan_sha256: str
    budget_columns: Any
    next_sequence: int = 0
    stream_pointer: int | None = None
    finished: bool = False


def create_cuda_gm1_tiled_wsm6_pdefq_workspace(
    plan: Gm1TilePlan, *, nk: int
) -> CudaGm1TiledPdefqWorkspace:
    if not isinstance(plan, Gm1TilePlan):
        raise TypeError("plan must be a Gm1TilePlan")
    nk = _positive_int("nk", nk)
    if nk != STRETCH_Z1_CHILD_15KM_PARAMETERS.nk:
        raise ValueError("tiled pdefq is sealed to the exact 75-level child")
    cupy = importlib.import_module("cupy")
    return CudaGm1TiledPdefqWorkspace(
        plan.nx,
        plan.ny,
        nk,
        plan.ntiles,
        plan.ownership_sha256,
        cupy.empty((6, plan.ny), dtype=cupy.float64),
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPdefqChunk:
    ready_event: Any
    compiled: CompiledGm1TiledPdefq
    workspace: CudaGm1TiledPdefqWorkspace
    receipt: Mapping[str, Any]


def _require_device_tile_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, spec: Gm1TileSpec, nk: int
) -> dict[str, Any]:
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    shapes = tiled_pdefq_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
    )
    if tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA tiled pdefq inventory or order changed")
    result: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA tiled pdefq arrays must not alias")
    return result


def launch_cuda_gm1_tiled_wsm6_pdefq_chunk(
    arrays: Mapping[str, Any],
    sigmaf: Any,
    plan: Gm1TilePlan,
    spec: Gm1TileSpec,
    workspace: CudaGm1TiledPdefqWorkspace,
    *,
    species: int,
    dz: float,
    stream: Any | None = None,
    compiled: CompiledGm1TiledPdefq | None = None,
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
) -> CudaGm1TiledPdefqChunk:
    """Enqueue one exact schedule chunk and carry its budj into the next tile."""

    require_wsm6_pdefq_selection(selection)
    if not isinstance(plan, Gm1TilePlan) or not isinstance(spec, Gm1TileSpec):
        raise TypeError("plan/spec must be Gm1TilePlan/Gm1TileSpec")
    if spec not in plan.specs:
        raise ValueError("tile spec is not a member of the supplied plan")
    if not isinstance(workspace, CudaGm1TiledPdefqWorkspace):
        raise TypeError("workspace must be CudaGm1TiledPdefqWorkspace")
    if (
        workspace.nx,
        workspace.ny,
        workspace.nk,
        workspace.tile_count,
        workspace.plan_sha256,
    ) != (plan.nx, plan.ny, STRETCH_Z1_CHILD_15KM_PARAMETERS.nk, plan.ntiles, plan.ownership_sha256):
        raise ValueError("tiled pdefq workspace geometry/plan identity differs")
    if workspace.finished:
        raise ValueError("tiled pdefq workspace has already been finished")
    if isinstance(species, bool) or not isinstance(species, int) or species not in range(6):
        raise ValueError("species must index qv,qc,qr,qi,qs,qg")
    expected_sequence = species * plan.ntiles + spec.index
    if workspace.next_sequence != expected_sequence:
        raise ValueError(
            "tiled pdefq must run species-major and west-to-east/south-to-north "
            f"without substitution; expected sequence {workspace.next_sequence}, got {expected_sequence}"
        )
    cupy = importlib.import_module("cupy")
    a = _require_device_tile_arrays(cupy, arrays, spec=spec, nk=workspace.nk)
    if (
        not isinstance(sigmaf, cupy.ndarray)
        or sigmaf.dtype != cupy.float32
        or sigmaf.shape != (workspace.nk + 3,)
        or not sigmaf.flags.c_contiguous
    ):
        raise ValueError("sigmaf must be C-order CUDA float32 [nk+3]")
    pointers = {int(value.data.ptr) for value in a.values()}
    pointers.add(int(sigmaf.data.ptr))
    if (
        len(pointers) != len(a) + 1
        or int(workspace.budget_columns.data.ptr) in pointers
    ):
        raise ValueError("tiled pdefq profiles/workspace must not alias tile arrays")
    dz32 = cupy.float32(dz)
    if not math.isfinite(float(dz32)) or dz32 <= cupy.float32(0.0):
        raise ValueError("dz must be positive finite float32")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    pointer = int(stream.ptr)
    if workspace.stream_pointer is None:
        workspace.stream_pointer = pointer
    elif workspace.stream_pointer != pointer:
        raise ValueError("all tiled pdefq chunks must use one exact CUDA stream")
    compiled = compile_cuda_gm1_tiled_wsm6_pdefq() if compiled is None else compiled
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    if x0 == 0 and spec.tx != 0:
        raise ValueError("only the west-most tile may reset a pdefq budget row")
    threads = 128
    blocks = ((active_ny + threads - 1) // threads, 1, 1)
    with stream:
        compiled.kernel(
            blocks,
            (threads, 1, 1),
            (
                cupy.uint32(species),
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(workspace.nk),
                cupy.uint32(spec.tile_nx),
                cupy.uint32(spec.tile_ny),
                cupy.uint32(y0),
                cupy.uint32(plan.ny),
                cupy.uint32(1 if x0 == 0 else 0),
                cupy.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop),
                dz32,
                a["ruh"],
                a["rvh"],
                a["terrain_msl"],
                sigmaf,
                a["rho"],
                a["q3d"],
                workspace.budget_columns,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    workspace.next_sequence += 1
    receipt = MappingProxyType(
        {
            "schema": CUDA_GM1_TILESTREAM_PDEFQ_SCHEMA,
            "abi": CUDA_GM1_TILESTREAM_PDEFQ_ABI,
            "status": "ENQUEUED_SOURCE_ORDERED_TILE_CHUNK",
            "species": species,
            "species_name": WSM6_SPECIES_ORDER[species],
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "sequence_index": expected_sequence,
            "next_sequence": workspace.next_sequence,
            "budj_input": "exact prior west tile or +0 at physical west",
            "same_stream_no_host_sync": True,
            "stream_pointer": pointer,
            "kernel_source_sha256": CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "qa_mutated": False,
            "wsm6_or_solve3_performed": False,
        }
    )
    return CudaGm1TiledPdefqChunk(event, compiled, workspace, receipt)


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPdefqFinish:
    ready_event: Any
    tiled_compiled: CompiledGm1TiledPdefq
    pdefq_compiled: CompiledWsm6Pdefq
    workspace: CudaGm1TiledPdefqWorkspace
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_wsm6_pdefq_finish(
    workspace: CudaGm1TiledPdefqWorkspace,
    asq: Any,
    *,
    dx: float,
    dy: float,
    dz: float,
    stream: Any,
    tiled_compiled: CompiledGm1TiledPdefq,
    pdefq_compiled: CompiledWsm6Pdefq | None = None,
) -> CudaGm1TiledPdefqFinish:
    """Run the existing exact species/j asq finish after every tile chunk."""

    if not isinstance(workspace, CudaGm1TiledPdefqWorkspace):
        raise TypeError("workspace must be CudaGm1TiledPdefqWorkspace")
    if workspace.finished:
        raise ValueError("tiled pdefq workspace has already been finished")
    if workspace.next_sequence != 6 * workspace.tile_count:
        raise ValueError("cannot finish tiled pdefq before every exact tile/species chunk")
    cupy = importlib.import_module("cupy")
    if (
        not isinstance(asq, cupy.ndarray)
        or asq.dtype != cupy.float64
        or asq.shape != (6,)
        or not asq.flags.c_contiguous
    ):
        raise ValueError("asq must be C-order CUDA float64 [6]")
    if int(asq.data.ptr) == int(workspace.budget_columns.data.ptr):
        raise ValueError("asq must not alias the global budget workspace")
    if int(stream.ptr) != workspace.stream_pointer:
        raise ValueError("tiled pdefq finish must use the exact chunk stream")
    values = tuple(cupy.float32(value) for value in (dx, dy, dz))
    if not all(math.isfinite(float(value)) and value > cupy.float32(0.0) for value in values):
        raise ValueError("dx, dy, and dz must be positive finite float32")
    volume = cupy.float32(cupy.float32(values[0] * values[1]) * values[2])
    pdefq_compiled = compile_cuda_wsm6_pdefq() if pdefq_compiled is None else pdefq_compiled
    if pdefq_compiled.target_arch != tiled_compiled.target_arch:
        raise ValueError("tiled chunk and canonical pdefq finish target architectures differ")
    finish = pdefq_compiled.kernels[1]
    with stream:
        finish(
            (1, 1, 1),
            (1, 1, 1),
            (
                volume,
                cupy.uint32(workspace.ny),
                workspace.budget_columns,
                asq,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    workspace.finished = True
    record = {
        "schema": CUDA_GM1_TILESTREAM_PDEFQ_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_PDEFQ_ABI,
        "status": "ENQUEUED_SOURCE_EXACT_TILED_PDEFQ_FINISH",
        "geometry": {
            "nx": workspace.nx,
            "ny": workspace.ny,
            "nk": workspace.nk,
            "dx": float(values[0]),
            "dy": float(values[1]),
            "dz": float(values[2]),
        },
        "tile_count": workspace.tile_count,
        "chunk_count": workspace.next_sequence,
        "plan_sha256": workspace.plan_sha256,
        "source_order": [
            "qv,qc,qr,qi,qs,qg",
            "south-to-north tile rows; west-to-east carry within each row",
            "one canonical pdefq finish over species then physical j",
        ],
        "global_workspace": {
            "carrier": "budgets",
            "dtype": "<f8",
            "shape": [6, workspace.ny],
            "nbytes": int(workspace.budget_columns.nbytes),
            "tile_local": False,
            "deterministic_reduction": "serial source order; no atomics/regrouping",
        },
        "mutated": ["q3d[0:6]", "asq[0:6]"],
        "signed_asq": True,
        "same_stream_no_host_sync": True,
        "stream_pointer": workspace.stream_pointer,
        "kernels": {
            "tiled_chunk_source_sha256": CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256,
            "canonical_finish_source_sha256": CUDA_WSM6_PDEFQ_SOURCE_SHA256,
            "target_arch": tiled_compiled.target_arch,
        },
        "vertical": {
            "grid_sha256": STRETCH_Z1_CHILD_15KM_GRID_SHA256,
            "proof_sha256": STRETCH_Z1_CHILD_15KM_PROOF_SHA256,
            "rmh": "reconstructed per tile from terrain_msl/sigmaf",
        },
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": AUTHORITY_SOURCE_SHA256,
            "anchors": SOURCE_ANCHORS,
        },
        "refusals": {
            "qa_mutation": True,
            "wsm6": True,
            "solve3": True,
            "pressure": True,
            "halo_or_open_history_mutation": True,
            "mpi_reduction": True,
            "branch_substitution": True,
        },
        "claim_limit": GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledPdefqFinish(
        event,
        tiled_compiled,
        pdefq_compiled,
        workspace,
        MappingProxyType(record),
    )


def five_meter_tiled_pdefq_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, Any]:
    """Exact incremental memory/carry schedule for the 3000-square child."""

    shapes = tiled_pdefq_buffer_shapes(
        tile_nx=tile_nx,
        tile_ny=tile_ny,
        nk=STRETCH_Z1_CHILD_15KM_PARAMETERS.nk,
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    budgets = 6 * 3000 * 8
    sigmaf = (STRETCH_Z1_CHILD_15KM_PARAMETERS.nk + 3) * 4
    record = {
        "schema": "gm1.hybrid-tilestream-pdefq-memory/v1",
        "domain": [75, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "per_host_or_device_buffer_bytes": per_buffer,
        "double_host_buffers_bytes": 2 * per_buffer,
        "double_device_buffers_bytes": 2 * per_buffer,
        "global_budget_f64_bytes": budgets,
        "global_sigmaf_f32_bytes": sigmaf,
        "full_domain_rmh_duplicate_bytes": 0,
        "full_domain_q3d_duplicate_bytes": 0,
        "schedule": "species-major; tile-row south-to-north; west-to-east budj carry",
        "pressure_workspace_executed": False,
        "wsm6_executed": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "CUDA_GM1_TILESTREAM_PDEFQ_ABI",
    "CUDA_GM1_TILESTREAM_PDEFQ_KERNEL",
    "CUDA_GM1_TILESTREAM_PDEFQ_SCHEMA",
    "CUDA_GM1_TILESTREAM_PDEFQ_SOURCE",
    "CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_PDEFQ_SOURCE_SHA256",
    "CompiledGm1TiledPdefq",
    "CudaGm1TiledPdefqChunk",
    "CudaGm1TiledPdefqFinish",
    "CudaGm1TiledPdefqWorkspace",
    "GM1_TILESTREAM_PDEFQ_CLAIM_LIMIT",
    "Gm1TiledPdefqSourceBinding",
    "compile_cuda_gm1_tiled_wsm6_pdefq",
    "cpu_gm1_tiled_wsm6_pdefq_pdscheme1_inplace",
    "create_cuda_gm1_tiled_wsm6_pdefq_workspace",
    "fill_gm1_tiled_pdefq_source",
    "five_meter_tiled_pdefq_inventory",
    "launch_cuda_gm1_tiled_wsm6_pdefq_chunk",
    "launch_cuda_gm1_tiled_wsm6_pdefq_finish",
    "scatter_gm1_tiled_pdefq_output",
    "tiled_pdefq_buffer_shapes",
)
