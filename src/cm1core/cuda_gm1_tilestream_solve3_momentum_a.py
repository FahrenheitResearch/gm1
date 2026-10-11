"""Physical tiled solve3 terrain finalize and momentum current-to-A subset.

The selected CM1 r21.1 serial terrain path reaches ``solve3.F:548-558`` only
after held-rate recapture and the terrain lower-face update at lines 430-433.
This operator consumes the exact held token, applies the physical portion of
``bcwsfc`` in default REAL statement order, then copies the physical u/v/w
current state into ua/va/wa.  Horizontal ``bc2d`` halos and the wider padded
copy range are deliberately excluded, so the completion token does not yet
authorize TKE commit or any later solve3 operation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_solve3_held import Gm1TiledSolve3HeldToken
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SCHEMA = (
    "gm1.cuda-tilestream-solve3-physical-momentum-a/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_ABI = (
    "gm1.cuda-tilestream-solve3-terrain-momentum-a-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_KERNEL = (
    "gm1_tiled_solve3_terrain_momentum_a_f32_v1"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_ANCHORS = (
    "src/solve3.F:339-559",
    "src/bc.F:12-172",
    "src/bc.F:1151-1178",
)
GM1_SOURCE_IDENTITY_SHA256 = (
    "d6d7588b9a2a2ecd979d4d5e609a602f52d488fd918c590bc3998a5319aebfe3"
)
GM1_ACCEPTED_HELD_RECEIPT_FILE_SHA256 = (
    "f094d097c3dae2f026359595767331da49f1d95734b2a3ab8ae9ffc982aba93b"
)
GM1_ACCEPTED_HELD_RECEIPT_SEMANTIC_SHA256 = (
    "d52c56975f45243c26713a6c92783e570e661c6b7005c7aa4f085bd55e53c7eb"
)
GM1_ACCEPTED_HELD_OUTPUT_SHA256 = MappingProxyType(
    {
        "qpten": "de3f846f3f26435c66efa195e572043395258fe42ec52df551b4d7830f9cd8c4",
        "qtten": "b7a07314e9aeb5d56e29ee5cbc7b280db2c0d83461b2ebf45659b2bbdd2fe477",
        "qvten": "f171745a7fff7bb96dbeb689676ad7aaa4eefcea3f7c7d343a9fecdc698bccd9",
        "qcten": "3bda097d9fb712f49013ae1c1a493e08c54651516e59a1bb042563b05470e48a",
    }
)
GM1_ACCEPTED_CURRENT_MOMENTUM_SHA256 = MappingProxyType(
    {
        "u3d": "b55dcf1792d1fdd8dafb5eabda2b70f1a95c8aa4c7b67070d3421e6a05147402",
        "v3d": "4c2b5fecd445fcae0564f936bd12e90caa52f643c78d471ff5f3fe6b1c23ee89",
        "w3d": "02407d5aa4a26a2c1deda435c357a36094fe647504076b6281a41016de71c21b",
    }
)
GM1_ACCEPTED_INITIAL_A_SHA256 = MappingProxyType(
    {
        "ua": "b55dcf1792d1fdd8dafb5eabda2b70f1a95c8aa4c7b67070d3421e6a05147402",
        "va": "4c2b5fecd445fcae0564f936bd12e90caa52f643c78d471ff5f3fe6b1c23ee89",
        "wa": "02407d5aa4a26a2c1deda435c357a36094fe647504076b6281a41016de71c21b",
    }
)
GM1_ACCEPTED_TERRAIN_METRIC_SHA256 = MappingProxyType(
    {
        "gz": "d3c340f7d6cdc3f2b33e7014455b28fbe8e49085525fa832d347a92b9450b4cf",
        "dzdx": "3ec5b45aed3e79eb5cf94cc2f53ca3d5bf8ad0c40c715f7721582b0e7e8c50cf",
        "dzdy": "e8b9f310d854ff8a1805cf1dcb4249ccf1da7708873fb7e55a26584db2ae4423",
    }
)
GM1_ACCEPTED_FINALIZED_W3D_SHA256 = (
    "67177ebbc40e2a02f56967c0371e5497277ba9c2c536d82ef061f4e1eee75824"
)
GM1_ACCEPTED_FINALIZED_W_BOTTOM_SHA256 = (
    "09b9008a4ad637121672369fab9a2e80a35198cfb4d87d8e6972903662a8ca64"
)
GM1_SELECTED_INTERVENING_BRANCHES = MappingProxyType(
    {
        "dot2p": False,
        "large_scale_nudging_u_v": False,
        "mpi": False,
        "parcels": False,
        "infrequent_momentum_diagnostics": False,
        "terrain": True,
        "passive_tracers_before_momentum_a": False,
    }
)
GM1_TILESTREAM_SOLVE3_MOMENTUM_A_CLAIM_LIMIT = (
    "CM1 r21.1 selected serial fixed-terrain physical bcwsfc and physical "
    "u3d/v3d/w3d to ua/va/wa subset through solve3.F:548-558 after the exact "
    "held token; horizontal bc2d/padded current-to-A, TKE commit, scalar A, "
    "rr/rf, global psolver, MPI, open histories, full solve3/RK, and five-metre "
    "meteorological product remain refused"
)

_BUFFER_ORDER = (
    "u3d",
    "v3d",
    "w3d",
    "ua",
    "va",
    "wa",
    "gz",
    "dzdx",
    "dzdy",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha_array(value: Any) -> str:
    return hashlib.sha256(
        value.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def tiled_solve3_momentum_a_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    return MappingProxyType(
        {
            "u3d": (nk, tile_ny, tile_nx + 1),
            "v3d": (nk, tile_ny + 1, tile_nx),
            "w3d": (nk + 1, tile_ny, tile_nx),
            "ua": (nk, tile_ny, tile_nx + 1),
            "va": (nk, tile_ny + 1, tile_nx),
            "wa": (nk + 1, tile_ny, tile_nx),
            "gz": (tile_ny, tile_nx),
            "dzdx": (tile_ny, tile_nx),
            "dzdy": (tile_ny, tile_nx),
        }
    )


def _require_held_token(
    token: object, *, source_identity_sha256: str, real: bool
) -> Mapping[str, Any]:
    if not isinstance(token, Gm1TiledSolve3HeldToken):
        raise TypeError("held token must be the exact held-recapture token type")
    receipt = token.receipt
    if (
        receipt.get("schema") != "gm1.tilestream-solve3-held-token/v1"
        or receipt.get("status")
        != "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY"
        or receipt.get("ready_for") != "momentum current-to-A admission only"
        or receipt.get("full_operation_b_available") is not False
        or receipt.get("tke_commit_performed") is not False
        or receipt.get("source_identity_sha256") != source_identity_sha256
    ):
        raise ValueError("held token scope or source identity changed")
    detached = dict(receipt)
    operation_token = detached.pop("operation_token_sha256", None)
    if operation_token != hashlib.sha256(_canonical(detached)).hexdigest():
        raise ValueError("held token operation identity changed")
    if real and receipt.get("outputs") != dict(GM1_ACCEPTED_HELD_OUTPUT_SHA256):
        raise ValueError("held token output hashes differ from accepted evidence")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3MomentumABinding:
    u3d: Any
    v3d: Any
    w3d: Any
    ua: Any
    va: Any
    wa: Any
    gz: Any
    dzdx: Any
    dzdy: Any
    held_token: Gm1TiledSolve3HeldToken
    nx: int
    ny: int
    nk: int
    source_identity_sha256: str
    held_receipt_file_sha256: str
    held_receipt_semantic_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        u3d: Any,
        v3d: Any,
        w3d: Any,
        ua: Any,
        va: Any,
        wa: Any,
        gz: Any,
        dzdx: Any,
        dzdy: Any,
        held_token: Gm1TiledSolve3HeldToken,
        source_identity_sha256: str,
        held_receipt_file_sha256: str,
        held_receipt_semantic_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3MomentumABinding":
        np = importlib.import_module("numpy")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if (
            not isinstance(u3d, np.ndarray)
            or u3d.dtype != np.dtype("<f4")
            or u3d.ndim != 3
            or not u3d.flags.c_contiguous
        ):
            raise ValueError("u3d must be host C-<f4 [nk,ny,nx+1]")
        nk, ny, nx1 = map(int, u3d.shape)
        nx = nx1 - 1
        shapes = {
            "v3d": (nk, ny + 1, nx),
            "w3d": (nk + 1, ny, nx),
            "ua": (nk, ny, nx + 1),
            "va": (nk, ny + 1, nx),
            "wa": (nk + 1, ny, nx),
            "gz": (ny, nx),
            "dzdx": (ny, nx),
            "dzdy": (ny, nx),
        }
        values = {
            "v3d": v3d,
            "w3d": w3d,
            "ua": ua,
            "va": va,
            "wa": wa,
            "gz": gz,
            "dzdx": dzdx,
            "dzdy": dzdy,
        }
        for name, shape in shapes.items():
            value = values[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be host C-<f4 {shape}")
        mutable = {"w3d": w3d, "ua": ua, "va": va, "wa": wa}
        if any(not value.flags.writeable for value in mutable.values()):
            raise ValueError("w3d and momentum A destinations must be writable")
        all_values = (u3d, *values.values())
        for index, left in enumerate(all_values):
            if any(np.shares_memory(left, right) for right in all_values[index + 1 :]):
                raise ValueError("momentum current/A/metric arrays must not alias")
        if not all(bool(np.all(np.isfinite(value))) for value in all_values):
            raise ValueError("momentum-A input contains non-finite words")
        source_identity_sha256 = _sha256(
            "source_identity_sha256", source_identity_sha256
        )
        held_receipt_file_sha256 = _sha256(
            "held_receipt_file_sha256", held_receipt_file_sha256
        )
        held_receipt_semantic_sha256 = _sha256(
            "held_receipt_semantic_sha256", held_receipt_semantic_sha256
        )
        _require_held_token(
            held_token,
            source_identity_sha256=source_identity_sha256,
            real=not allow_compact_validation,
        )
        if allow_compact_validation:
            if min(nx, ny, nk) < 2:
                raise ValueError("compact momentum-A domain is too small")
        else:
            if (nx, ny, nk) != (300, 300, 75):
                raise ValueError("real momentum-A source geometry changed")
            if source_identity_sha256 != GM1_SOURCE_IDENTITY_SHA256:
                raise ValueError("real momentum-A source identity changed")
            if (
                held_receipt_file_sha256 != GM1_ACCEPTED_HELD_RECEIPT_FILE_SHA256
                or held_receipt_semantic_sha256
                != GM1_ACCEPTED_HELD_RECEIPT_SEMANTIC_SHA256
            ):
                raise ValueError("held evidence receipt pins changed")
            current = {
                name: _sha_array(value)
                for name, value in {"u3d": u3d, "v3d": v3d, "w3d": w3d}.items()
            }
            initial_a = {
                name: _sha_array(value)
                for name, value in {"ua": ua, "va": va, "wa": wa}.items()
            }
            metrics = {
                name: _sha_array(value)
                for name, value in {"gz": gz, "dzdx": dzdx, "dzdy": dzdy}.items()
            }
            if current != dict(GM1_ACCEPTED_CURRENT_MOMENTUM_SHA256):
                raise ValueError("accepted current momentum bytes changed")
            if initial_a != dict(GM1_ACCEPTED_INITIAL_A_SHA256):
                raise ValueError("accepted initial momentum-A bytes changed")
            if metrics != dict(GM1_ACCEPTED_TERRAIN_METRIC_SHA256):
                raise ValueError("accepted physical terrain metric bytes changed")
        return cls(
            u3d,
            v3d,
            w3d,
            ua,
            va,
            wa,
            gz,
            dzdx,
            dzdy,
            held_token,
            nx,
            ny,
            nk,
            source_identity_sha256,
            held_receipt_file_sha256,
            held_receipt_semantic_sha256,
            allow_compact_validation,
        )


def fill_gm1_tiled_solve3_momentum_a_source(
    binding: Gm1TiledSolve3MomentumABinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3MomentumABinding):
        raise TypeError("binding must be Gm1TiledSolve3MomentumABinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover momentum-A source")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("momentum-A tile inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"{name} tile must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))
    target["u3d"][:, :active_ny, : active_nx + 1] = binding.u3d[:, y0:y1, x0 : x1 + 1]
    target["ua"][:, :active_ny, : active_nx + 1] = binding.ua[:, y0:y1, x0 : x1 + 1]
    target["v3d"][:, : active_ny + 1, :active_nx] = binding.v3d[:, y0 : y1 + 1, x0:x1]
    target["va"][:, : active_ny + 1, :active_nx] = binding.va[:, y0 : y1 + 1, x0:x1]
    for name in ("w3d", "wa"):
        target[name][:, :active_ny, :active_nx] = getattr(binding, name)[
            :, y0:y1, x0:x1
        ]
    for name in ("gz", "dzdx", "dzdy"):
        target[name][:active_ny, :active_nx] = getattr(binding, name)[y0:y1, x0:x1]
    values = tuple(target.values())
    for index, left in enumerate(values):
        if any(np.shares_memory(left, right) for right in values[index + 1 :]):
            raise ValueError("momentum-A tile arrays must not alias")
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_mass_shape": [binding.nk, active_ny, active_nx],
            "active_u_input_shape": [binding.nk, active_ny, active_nx + 1],
            "active_v_input_shape": [binding.nk, active_ny + 1, active_nx],
            "active_u_owned_nx": active_nx + int(x1 == binding.nx),
            "active_v_owned_ny": active_ny + int(y1 == binding.ny),
            "artificial_padding_initialized": "+0",
        }
    )


def cpu_gm1_tiled_solve3_momentum_a_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    active_u_owned_nx: int,
    active_v_owned_ny: int,
) -> None:
    """Apply physical bcwsfc then physical momentum current-to-A copies."""

    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError("CPU momentum-A inventory/order changed")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    active_u_owned_nx = _positive_int("active_u_owned_nx", active_u_owned_nx)
    active_v_owned_ny = _positive_int("active_v_owned_ny", active_v_owned_ny)
    uterm = np.add(
        arrays["u3d"][0, :active_ny, :active_nx],
        arrays["u3d"][0, :active_ny, 1 : active_nx + 1],
        dtype=np.float32,
    )
    np.multiply(uterm, arrays["dzdx"][:active_ny, :active_nx], out=uterm)
    vterm = np.add(
        arrays["v3d"][0, :active_ny, :active_nx],
        arrays["v3d"][0, 1 : active_ny + 1, :active_nx],
        dtype=np.float32,
    )
    np.multiply(vterm, arrays["dzdy"][:active_ny, :active_nx], out=vterm)
    bottom = arrays["w3d"][0, :active_ny, :active_nx]
    np.add(uterm, vterm, out=bottom)
    np.multiply(bottom, np.float32(0.5), out=bottom)
    np.multiply(bottom, arrays["gz"][:active_ny, :active_nx], out=bottom)
    arrays["ua"][:, :active_ny, :active_u_owned_nx] = arrays["u3d"][
        :, :active_ny, :active_u_owned_nx
    ]
    arrays["va"][:, :active_v_owned_ny, :active_nx] = arrays["v3d"][
        :, :active_v_owned_ny, :active_nx
    ]
    arrays["wa"][:-1, :active_ny, :active_nx] = arrays["w3d"][
        :-1, :active_ny, :active_nx
    ]


def scatter_gm1_tiled_solve3_momentum_a_output(
    binding: Gm1TiledSolve3MomentumABinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    shapes = tiled_solve3_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("momentum-A scatter inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    u1 = x1 + int(x1 == binding.nx)
    v1 = y1 + int(y1 == binding.ny)
    binding.w3d[0, y0:y1, x0:x1] = source["w3d"][0, :active_ny, :active_nx]
    binding.ua[:, y0:y1, x0:u1] = source["ua"][:, :active_ny, : u1 - x0]
    binding.va[:, y0:v1, x0:x1] = source["va"][:, : v1 - y0, :active_nx]
    binding.wa[:-1, y0:y1, x0:x1] = source["wa"][:-1, :active_ny, :active_nx]
    return MappingProxyType(
        {
            "tile": spec.index,
            "mass_owned_yx": [y0, y1, x0, x1],
            "u_owned_yx": [y0, y1, x0, u1],
            "v_owned_yx": [y0, v1, x0, x1],
            "terrain_bottom_words": active_ny * active_nx,
            "padded_halo_words": 0,
            "open_history_mutation": False,
        }
    )


CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE = r"""// gm1.cuda-tilestream-solve3-terrain-momentum-a-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
extern "C" __global__ void gm1_tiled_solve3_terrain_momentum_a_f32_v1(
 const float*u3d,const float*v3d,float*w3d,float*ua,float*va,float*wa,
 const float*gz,const float*dzdx,const float*dzdy,
 unsigned int active_nx,unsigned int active_ny,unsigned int active_u_owned_nx,
 unsigned int active_v_owned_ny,unsigned int nk,unsigned int tile_nx,unsigned int tile_ny){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long nm=(unsigned long long)active_ny*active_nx;
 const unsigned long long nu=(unsigned long long)nk*active_ny*active_u_owned_nx;
 const unsigned long long nv=(unsigned long long)nk*active_v_owned_ny*active_nx;
 const unsigned long long nw=(unsigned long long)(nk-1)*active_ny*active_nx;
 if(q<nm){
  const unsigned int i=(unsigned int)(q%active_nx),j=(unsigned int)(q/active_nx);
  const unsigned long long zu=(unsigned long long)j*(tile_nx+1)+i;
  const unsigned long long zv=(unsigned long long)j*tile_nx+i;
  const unsigned long long zw=(unsigned long long)j*tile_nx+i;
  float us=__fadd_rn(u3d[zu],u3d[zu+1]);
  float ut=__fmul_rn(us,dzdx[zw]);
  float vs=__fadd_rn(v3d[zv],v3d[zv+tile_nx]);
  float vt=__fmul_rn(vs,dzdy[zw]);
  float s=__fadd_rn(ut,vt);
  float h=__fmul_rn(0.5f,s);
  const float bottom=__fmul_rn(h,gz[zw]);
  w3d[zw]=bottom;
  wa[zw]=bottom;
 }
 if(q<nu){
  const unsigned int i=(unsigned int)(q%active_u_owned_nx);
  const unsigned long long t=q/active_u_owned_nx;
  const unsigned int j=(unsigned int)(t%active_ny),k=(unsigned int)(t/active_ny);
  const unsigned long long z=((unsigned long long)k*tile_ny+j)*(tile_nx+1)+i;
  ua[z]=u3d[z];
 }
 if(q<nv){
  const unsigned int i=(unsigned int)(q%active_nx);
  const unsigned long long t=q/active_nx;
  const unsigned int j=(unsigned int)(t%active_v_owned_ny),k=(unsigned int)(t/active_v_owned_ny);
  const unsigned long long z=((unsigned long long)k*(tile_ny+1)+j)*tile_nx+i;
  va[z]=v3d[z];
 }
 if(q<nw){
  const unsigned int i=(unsigned int)(q%active_nx);
  const unsigned long long t=q/active_nx;
  const unsigned int j=(unsigned int)(t%active_ny),k=1u+(unsigned int)(t/active_ny);
  const unsigned long long z=((unsigned long long)k*tile_ny+j)*tile_nx+i;
  wa[z]=w3d[z];
 }
}
"""
CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3MomentumA:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_momentum_a() -> CompiledGm1TiledSolve3MomentumA:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE,
        "gm1_tiled_solve3_momentum_a_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE,
        "gm1_tiled_solve3_momentum_a_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_KERNEL}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks terrain momentum-A entry")
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
    return CompiledGm1TiledSolve3MomentumA(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_KERNEL),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3MomentumALaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3MomentumA
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_momentum_a(
    arrays: Mapping[str, Any],
    binding: Gm1TiledSolve3MomentumABinding,
    spec: Gm1TileSpec,
    *,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3MomentumA | None = None,
) -> CudaGm1TiledSolve3MomentumALaunch:
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_momentum_a_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA momentum-A inventory/order changed")
    ranges = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        start = int(value.data.ptr)
        ranges.append((start, start + int(value.nbytes)))
    for index, (left_start, left_end) in enumerate(ranges):
        if any(
            left_start < right_end and right_start < left_end
            for right_start, right_end in ranges[index + 1 :]
        ):
            raise ValueError("CUDA momentum-A arrays must not alias")
    held = _require_held_token(
        binding.held_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    if int(held["stream_pointer"]) != int(stream.ptr):
        raise ValueError("momentum-A must remain on the held-token stream")
    compiled = (
        compile_cuda_gm1_tiled_solve3_momentum_a() if compiled is None else compiled
    )
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    active_u_owned_nx = active_nx + int(x1 == binding.nx)
    active_v_owned_ny = active_ny + int(y1 == binding.ny)
    words = max(
        active_ny * active_nx,
        binding.nk * active_ny * active_u_owned_nx,
        binding.nk * active_v_owned_ny * active_nx,
        binding.nk * active_ny * active_nx,
    )
    threads = 256
    with stream:
        compiled.kernel(
            ((words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                *(arrays[name] for name in _BUFFER_ORDER),
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(active_u_owned_nx),
                cupy.uint32(active_v_owned_ny),
                cupy.uint32(binding.nk),
                cupy.uint32(spec.tile_nx),
                cupy.uint32(spec.tile_ny),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_ABI,
        "status": "ENQUEUED_TERRAIN_FINALIZE_AND_PHYSICAL_MOMENTUM_A_TILE",
        "tile": spec.index,
        "mass_owned_yx": [y0, y1, x0, x1],
        "u_owned_yx": [y0, y1, x0, x1 + int(x1 == binding.nx)],
        "v_owned_yx": [y0, y1 + int(y1 == binding.ny), x0, x1],
        "source_identity_sha256": binding.source_identity_sha256,
        "held_operation_token_sha256": held["operation_token_sha256"],
        "held_receipt_file_sha256": binding.held_receipt_file_sha256,
        "intervening_branch_disposition": dict(GM1_SELECTED_INTERVENING_BRANCHES),
        "source_order": [
            "physical terrain bcwsfc",
            "physical u3d/v3d/w3d to ua/va/wa",
        ],
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "refusals": {
            "horizontal_bc2d_and_padded_momentum_a": True,
            "tke_commit": True,
            "scalar_current_to_a": True,
            "reciprocal_density": True,
            "global_psolver": True,
            "mpi": True,
            "open_history_mutation": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledSolve3MomentumALaunch(event, compiled, MappingProxyType(record))


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3PhysicalMomentumAToken:
    receipt: Mapping[str, Any]


def finish_gm1_tiled_solve3_momentum_a(
    binding: Gm1TiledSolve3MomentumABinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledSolve3MomentumALaunch],
) -> Gm1TiledSolve3PhysicalMomentumAToken:
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("momentum-A completion plan differs from source domain")
    if len(launches) != plan.ntiles:
        raise ValueError("momentum-A completion requires every exact tile")
    held = _require_held_token(
        binding.held_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    tokens = []
    streams = set()
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve3MomentumALaunch):
            raise TypeError("momentum-A completion requires exact launch objects")
        receipt = launch.receipt
        if (
            launch.ready_event is None
            or receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SCHEMA
            or receipt.get("status")
            != "ENQUEUED_TERRAIN_FINALIZE_AND_PHYSICAL_MOMENTUM_A_TILE"
            or receipt.get("tile") != spec.index
            or receipt.get("source_identity_sha256") != binding.source_identity_sha256
            or receipt.get("held_operation_token_sha256")
            != held["operation_token_sha256"]
        ):
            raise ValueError("momentum-A tile receipt identity/order differs")
        detached = dict(receipt)
        token = detached.pop("operation_token_sha256", None)
        if token != hashlib.sha256(_canonical(detached)).hexdigest():
            raise ValueError("momentum-A tile token changed")
        tokens.append(token)
        streams.add(int(receipt["stream_pointer"]))
    if streams != {int(held["stream_pointer"])}:
        raise ValueError("momentum-A tiles changed streams")
    np = importlib.import_module("numpy")
    outputs = {
        "u3d": binding.u3d,
        "v3d": binding.v3d,
        "w3d": binding.w3d,
        "ua": binding.ua,
        "va": binding.va,
        "wa": binding.wa,
    }
    if not all(bool(np.all(np.isfinite(value))) for value in outputs.values()):
        raise ValueError("completed physical momentum-A output is non-finite")
    hashes = {name: _sha_array(value) for name, value in outputs.items()}
    record = {
        "schema": "gm1.tilestream-solve3-physical-momentum-a-token/v1",
        "status": "ADMITTED_PHYSICAL_MOMENTUM_A_ONLY",
        "source_identity_sha256": binding.source_identity_sha256,
        "held_operation_token_sha256": held["operation_token_sha256"],
        "held_receipt": {
            "file_sha256": binding.held_receipt_file_sha256,
            "semantic_sha256": binding.held_receipt_semantic_sha256,
        },
        "tile_count": plan.ntiles,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_operation_tokens": tokens,
        "stream_pointer": next(iter(streams)),
        "outputs": hashes,
        "source_order": [
            "accepted held-recapture token",
            "selected inactive intervening branches",
            "physical terrain bcwsfc",
            "physical momentum current-to-A",
        ],
        "ready_for": "horizontal bc2d and padded momentum-A completion only",
        "full_momentum_current_to_a_available": False,
        "tke_commit_available": False,
        "claim_limit": GM1_TILESTREAM_SOLVE3_MOMENTUM_A_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return Gm1TiledSolve3PhysicalMomentumAToken(MappingProxyType(record))


def five_meter_tiled_solve3_momentum_a_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    shapes = tiled_solve3_momentum_a_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=75
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-physical-momentum-a-memory/v1",
        "domain": [75, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "carriers": list(shapes),
        "host_per_buffer_bytes": per_buffer,
        "device_per_buffer_bytes": per_buffer,
        "double_host_bytes": 2 * per_buffer,
        "double_device_bytes": 2 * per_buffer,
        "full_domain_duplicate_bytes": 0,
        "canonical_store_mutations": ["w3d(bottom)", "ua", "va", "wa(k=1..nk)"],
        "global_pressure_workspace_executed": False,
        "padded_halo_completion_executed": False,
        "tke_commit_executed": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_KERNEL",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_SOLVE3_MOMENTUM_A_SOURCE_SHA256",
    "CompiledGm1TiledSolve3MomentumA",
    "CudaGm1TiledSolve3MomentumALaunch",
    "GM1_ACCEPTED_CURRENT_MOMENTUM_SHA256",
    "GM1_ACCEPTED_FINALIZED_W3D_SHA256",
    "GM1_ACCEPTED_FINALIZED_W_BOTTOM_SHA256",
    "GM1_ACCEPTED_HELD_OUTPUT_SHA256",
    "GM1_ACCEPTED_HELD_RECEIPT_FILE_SHA256",
    "GM1_ACCEPTED_HELD_RECEIPT_SEMANTIC_SHA256",
    "GM1_ACCEPTED_INITIAL_A_SHA256",
    "GM1_ACCEPTED_TERRAIN_METRIC_SHA256",
    "GM1_SELECTED_INTERVENING_BRANCHES",
    "GM1_SOURCE_IDENTITY_SHA256",
    "GM1_TILESTREAM_SOLVE3_MOMENTUM_A_CLAIM_LIMIT",
    "Gm1TiledSolve3MomentumABinding",
    "Gm1TiledSolve3PhysicalMomentumAToken",
    "compile_cuda_gm1_tiled_solve3_momentum_a",
    "cpu_gm1_tiled_solve3_momentum_a_inplace",
    "fill_gm1_tiled_solve3_momentum_a_source",
    "finish_gm1_tiled_solve3_momentum_a",
    "five_meter_tiled_solve3_momentum_a_inventory",
    "launch_cuda_gm1_tiled_solve3_momentum_a",
    "scatter_gm1_tiled_solve3_momentum_a_output",
    "tiled_solve3_momentum_a_buffer_shapes",
)
