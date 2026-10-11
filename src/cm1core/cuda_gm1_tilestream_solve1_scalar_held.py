"""Tile-streamed live solve1 held theta/moisture tendencies for GM1.

CM1 builds the held scalar tendencies once before its three ``solve2`` RK
stages.  The selected child order is exact: positive-zero the targets,
Rayleigh establishes ``thten1``, eqtset-2 dissipative heating adds to it, and
the stretched scalar SGS operator adds theta and six moisture contributions.
This module consumes one already-enqueued corrected-closure tile so its live
``dissten/khh/khv`` words remain on the same CUDA stream.

No scalar advection, RK update, pressure solve, boundary history, checkpoint,
forecast, or five-metre meteorological product is produced here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_large_step import (
    CHILD_TERRAIN_RAYLEIGH_ARRAYS,
    CudaGm1ChildTerrainRayleighLaunch,
    CompiledGm1ChildTerrainRayleigh,
    child_terrain_rayleigh_shapes,
    launch_cuda_gm1_child_terrain_rayleigh,
)
from .cuda_gm1_child_stretched_scalar_turbulence import (
    CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
    CompiledGm1ScalarTurbulenceKernels,
    child_stretched_scalar_turbulence_shapes,
    launch_cuda_gm1_child_stretched_scalar_turbulence,
)
from .cuda_gm1_scalar_turbulence import (
    Gm1ScalarTurbulenceWorkspace,
    create_cuda_gm1_scalar_turbulence_workspace,
)
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_stretched_closure import (
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA,
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS,
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
    CudaGm1TiledStretchedClosureLaunch,
    Gm1TiledStretchedClosureToken,
)
from .eqtset2_energy_terms import (
    EQTSET2_IDISS_RECEIPT_SCHEMA,
    Eqtset2IdissCarriers,
    Eqtset2IdissConsumer,
    Eqtset2IdissResult,
)


CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_SCHEMA = (
    "gm1.cuda-tilestream-solve1-scalar-held/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA = (
    "gm1.tilestream-solve1-scalar-held-token/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_STATUS = (
    "ENQUEUED_LIVE_TILESTREAM_SOLVE1_SCALAR_HELD"
)
CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SOLVE1_SCALAR_HELD"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE1_SHA256 = (
    "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
)
AUTHORITY_TURBTEND_SHA256 = (
    "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
)
AUTHORITY_ANCHORS = (
    "src/solve1.F:314-345",
    "src/solve1.F:744-848",
    "src/solve1.F:917-975",
    "src/solve1.F:1169-1227",
    "src/turbtend.F:157-308",
    "src/turbtend.F:442-692",
)
SOURCE_ORDER = (
    "positive-zero held thten1 and six-species qten",
    "terrain Rayleigh establishes thten1",
    "eqtset=2 idiss adds live closure dissten heating to thten1",
    "stretched scalar SGS adds theta then qv/qc/qr/qi/qs/qg",
)
GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT = (
    "one source-ordered live tile-streamed CM1 r21.1 solve1 held theta and "
    "six-species moisture tendency pass using corrected closure dissten/khh/khv; "
    "no scalar advection, scalar/moist RK, pressure solve, boundary or MPI "
    "history, TKE RK, checkpoint, full forecast, or five-metre meteorological "
    "product"
)


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


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def solve1_scalar_held_tile_shapes(
    *, ni: int, nj: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Return the exact caller-owned device inventory for one compute tile."""

    rayleigh = child_terrain_rayleigh_shapes(ni, nj, nk)
    scalar = child_stretched_scalar_turbulence_shapes(ni, nj, nk)
    shapes = dict(rayleigh)
    for name, shape in scalar.items():
        if name not in {"khh", "khv"}:
            prior = shapes.get(name)
            if prior is not None and prior != shape:
                raise RuntimeError(f"solve1 scalar-held shape collision: {name}")
            shapes[name] = shape
    mass = (nk, nj, ni)
    shapes.update({"pi0": mass, "ppi": mass})
    return MappingProxyType(shapes)


_ARRAY_NAMES = tuple(
    solve1_scalar_held_tile_shapes(ni=8, nj=7, nk=5)
)


def _require_arrays(
    cupy: Any,
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, Any]:
    shapes = solve1_scalar_held_tile_shapes(ni=ni, nj=nj, nk=nk)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("solve1 scalar-held caller array inventory changed")
    checked: dict[str, Any] = {}
    for name in _ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-order CUDA float32 shape {shapes[name]}"
            )
        checked[name] = value
    pointers = [int(value.data.ptr) for value in checked.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("solve1 scalar-held caller arrays must not alias")
    return checked


def _require_closure_launch(
    launch: object, *, spec: Gm1TileSpec, stream_pointer: int
) -> CudaGm1TiledStretchedClosureLaunch:
    if not isinstance(launch, CudaGm1TiledStretchedClosureLaunch):
        raise TypeError("closure_launch must be the exact corrected closure result")
    receipt = launch.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS
        or receipt.get("tile") != spec.record()
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_no_host_sync") is not True
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("corrected closure tile identity or stream changed")
    return launch


def _require_closure_token(
    token: object, *, plan: Gm1TilePlan
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledStretchedClosureToken):
        raise TypeError("closure_token must be the exact corrected closure token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_TILE_LOCAL_STRETCHED_FRESH_CLOSURE_OPERATOR_ONLY"
        or receipt.get("tile_plan_sha256") != plan.ownership_sha256
        or receipt.get("tile_count") != plan.ntiles
        or receipt.get("invalid_interior_cell_count") != 0
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("corrected closure token identity or domain changed")
    return receipt


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve1ScalarHeldLaunch:
    ready_event: Any
    rayleigh: CudaGm1ChildTerrainRayleighLaunch
    idiss: Eqtset2IdissResult
    scalar_turbulence: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve1ScalarHeldToken:
    receipt: Mapping[str, object]


def launch_cuda_gm1_tiled_solve1_scalar_held(
    closure_launch: CudaGm1TiledStretchedClosureLaunch,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    vertical: Any,
    dx: float,
    dy: float,
    dt: float,
    stream: Any,
    rayleigh_compiled: CompiledGm1ChildTerrainRayleigh | None = None,
    scalar_compiled: CompiledGm1ScalarTurbulenceKernels | None = None,
    scalar_workspace: Gm1ScalarTurbulenceWorkspace | None = None,
    idiss_consumer: Eqtset2IdissConsumer | None = None,
) -> CudaGm1TiledSolve1ScalarHeldLaunch:
    """Enqueue the exact held scalar source order for one closure tile."""

    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    for name, value in (("dx", dx), ("dy", dy), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    stream_pointer = int(stream.ptr)
    closure_launch = _require_closure_launch(
        closure_launch, spec=spec, stream_pointer=stream_pointer
    )
    ni, nj, nk = spec.cnx, spec.cny, 75
    cupy = importlib.import_module("cupy")
    checked = _require_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    closure_arrays = {
        name: closure_launch.arrays[name] for name in ("dissten", "khh", "khv")
    }
    expected_closure_shapes = {
        "dissten": (nk + 1, nj, ni),
        "khh": (nk + 1, nj, ni),
        "khv": (nk + 1, nj, ni),
    }
    for name, shape in expected_closure_shapes.items():
        value = closure_arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"closure {name} must be CUDA float32 {shape}")
    all_pointers = [int(value.data.ptr) for value in checked.values()] + [
        int(value.data.ptr) for value in closure_arrays.values()
    ]
    if len(set(all_pointers)) != len(all_pointers):
        raise ValueError("solve1 scalar-held arrays alias live closure outputs")
    if scalar_workspace is None:
        scalar_workspace = create_cuda_gm1_scalar_turbulence_workspace(
            ni=ni, nj=nj, nk=nk
        )
    if idiss_consumer is None:
        idiss_consumer = Eqtset2IdissConsumer(nx=ni, ny=nj, nz=nk)
    if not isinstance(idiss_consumer, Eqtset2IdissConsumer):
        raise TypeError("idiss_consumer must be Eqtset2IdissConsumer")

    with stream:
        checked["thten1"].fill(cupy.float32(0.0))
        checked["qten"].fill(cupy.float32(0.0))
        rayleigh = launch_cuda_gm1_child_terrain_rayleigh(
            {name: checked[name] for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS},
            ni=ni,
            nj=nj,
            nk=nk,
            stream=stream,
            compiled=rayleigh_compiled,
        )
        idiss = idiss_consumer.apply(
            Eqtset2IdissCarriers(
                thten1=checked["thten1"],
                dissten=closure_arrays["dissten"],
                pi0=checked["pi0"],
                ppi=checked["ppi"],
                qv=checked["qa"][0],
                qc=checked["qa"][1],
                qr=checked["qa"][2],
                qi=checked["qa"][3],
                qs=checked["qa"][4],
                qg=checked["qa"][5],
            )
        )
        scalar_arrays = {
            name: (
                closure_arrays[name]
                if name in {"khh", "khv"}
                else checked[name]
            )
            for name in child_stretched_scalar_turbulence_shapes(ni, nj, nk)
        }
        scalar = launch_cuda_gm1_child_stretched_scalar_turbulence(
            scalar_arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            dx=float(dx),
            dy=float(dy),
            dt=float(dt),
            vertical=vertical,
            stream=stream,
            compiled=scalar_compiled,
            workspace=scalar_workspace,
            allow_compact_validation=True,
        )
    if idiss.receipt.get("schema") != EQTSET2_IDISS_RECEIPT_SCHEMA:
        raise ValueError("idiss receipt schema changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": stream_pointer,
        "closure_tile_operation_token_sha256": closure_launch.receipt[
            "operation_token_sha256"
        ],
        "source_order": list(SOURCE_ORDER),
        "suboperator_receipt_sha256": {
            "rayleigh": hashlib.sha256(_canonical(rayleigh.receipt)).hexdigest(),
            "idiss": hashlib.sha256(_canonical(dict(idiss.receipt))).hexdigest(),
            "scalar_turbulence": hashlib.sha256(
                _canonical(scalar.receipt)
            ).hexdigest(),
        },
        "operator_source_sha256": {
            "stretched_scalar_turbulence": (
                CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
            ),
        },
        "positive_zero_initialized": ["thten1", "qten"],
        "same_stream_no_host_sync": True,
        "scalar_advection_or_rk_performed": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledSolve1ScalarHeldLaunch(
        scalar.ready_event,
        rayleigh,
        idiss,
        scalar,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_tiled_solve1_scalar_held(
    closure_token: Gm1TiledStretchedClosureToken,
    launches: Sequence[CudaGm1TiledSolve1ScalarHeldLaunch],
    *,
    plan: Gm1TilePlan,
    outputs: Mapping[str, Any],
) -> Gm1TiledSolve1ScalarHeldToken:
    """Bind exact-once host-scattered held theta/moisture tendencies."""

    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("solve1 scalar-held admission requires the 300 grid oracle")
    closure = _require_closure_token(closure_token, plan=plan)
    if len(launches) != plan.ntiles:
        raise ValueError("solve1 scalar-held launch count differs from tile plan")
    tile_tokens: list[str] = []
    closure_tile_tokens: list[str] = []
    streams: set[int] = set()
    owned_columns = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve1ScalarHeldLaunch):
            raise TypeError("solve1 scalar-held launches contain a foreign result")
        receipt = launch.receipt
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_SCHEMA
            or receipt.get("status") != CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_STATUS
            or receipt.get("completed_nrk") != 0
            or receipt.get("tile") != spec.record()
            or receipt.get("same_stream_no_host_sync") is not True
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("solve1 scalar-held tile identity or cadence changed")
        tile_tokens.append(str(receipt["operation_token_sha256"]))
        closure_tile_tokens.append(
            str(receipt["closure_tile_operation_token_sha256"])
        )
        streams.add(int(receipt["stream_pointer"]))
        y0, y1, x0, x1 = spec.owned_rect("mass")
        owned_columns += (y1 - y0) * (x1 - x0)
    if owned_columns != 300 * 300:
        raise ValueError("solve1 scalar-held host scatter is not exact once")
    if len(streams) != 1 or next(iter(streams)) != closure.get("stream_pointer"):
        raise ValueError("closure and solve1 scalar-held streams differ")
    if hashlib.sha256(_canonical(sorted(closure_tile_tokens))).hexdigest() != closure.get(
        "tile_operation_tokens_sha256"
    ):
        raise ValueError("solve1 scalar-held launches do not bind every closure tile")

    np = importlib.import_module("numpy")
    expected = {
        "thten1": (75, 300, 300),
        "qten": (6, 75, 300, 300),
    }
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected):
        raise ValueError("solve1 scalar-held output inventory changed")
    hashes: dict[str, str] = {}
    for name, shape in expected.items():
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{name} must be finite host C-<f4 {shape}")
        hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "closure_operation_token_sha256": closure["operation_token_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(tile_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in expected.items()},
        "output_sha256": hashes,
        "output_words": 7 * 75 * 300 * 300,
        "source_order": list(SOURCE_ORDER),
        "ready_for": (
            "typed scalar/moist RK1 only after matching post-acoustic mass-flux "
            "and scalar theta-divergence tokens"
        ),
        "scalar_advection_or_rk_performed": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "auxiliary-initialized or frozen thten1/qten substitution",
            "scalar RK without matching live post-acoustic flux and theta divergence",
            "RK2/RK3 reuse without fresh same-stage prerequisites",
            "global psolver tilestream claim",
            "MPI exchange or open-history production",
            "full forecast or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve1ScalarHeldToken(MappingProxyType(receipt))


def five_meter_solve1_scalar_held_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    """Report exact tile-local memory without claiming a live 5 m token."""

    for name, value in (("tile_nx", tile_nx), ("tile_ny", tile_ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    ni, nj = tile_nx + 6, tile_ny + 6
    caller_shapes = solve1_scalar_held_tile_shapes(ni=ni, nj=nj, nk=nk)
    closure_shapes = {
        name: (nk + 1, nj, ni) for name in ("dissten", "khh", "khv")
    }
    workspace_shapes = {
        "rho_halo": (nk, nj + 2, ni + 2),
        "khh_halo": (nk + 1, nj + 2, ni + 2),
        "gz_halo": (nj + 2, ni + 2),
        "rgz_halo": (nj + 2, ni + 2),
        "scalar_halo": (nk, nj + 2, ni + 2),
        "scalar_w": (nk + 1, nj + 2, ni + 2),
        "flux_x": (nk, nj, ni + 1),
        "flux_y": (nk, nj + 1, ni),
        "flux_x_w": (nk + 1, nj, ni + 1),
        "flux_y_w": (nk + 1, nj + 1, ni),
        "thomas_c": (nk, nj, ni),
        "thomas_d": (nk, nj, ni),
    }
    idiss_shapes = {name: (nk, nj, ni) for name in ("ql", "qice", "cvm", "work", "epsd")}
    def byte_count(values: Mapping[str, tuple[int, ...]]) -> int:
        return 4 * sum(math.prod(shape) for shape in values.values())

    record: dict[str, object] = {
        "schema": "gm1.hybrid-solve1-scalar-held-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "compute_tile": [nj, ni],
        "caller_array_bytes_each": byte_count(caller_shapes),
        "closure_output_bytes_reused_each": byte_count(closure_shapes),
        "scalar_workspace_bytes_each": byte_count(workspace_shapes),
        "idiss_workspace_bytes_each": byte_count(idiss_shapes),
        "tile_streamable": True,
        "live_five_meter_scalar_held_token_available": False,
        "scalar_rk1_available_at_five_meters": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "claim_limit": GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT,
    }
    record["one_tile_incremental_device_bytes"] = (
        int(record["caller_array_bytes_each"])
        + int(record["scalar_workspace_bytes_each"])
        + int(record["idiss_workspace_bytes_each"])
    )
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE1_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "AUTHORITY_TURBTEND_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_STATUS",
    "CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS",
    "CudaGm1TiledSolve1ScalarHeldLaunch",
    "GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT",
    "Gm1TiledSolve1ScalarHeldToken",
    "SOURCE_ORDER",
    "finish_cuda_gm1_tiled_solve1_scalar_held",
    "five_meter_solve1_scalar_held_inventory",
    "launch_cuda_gm1_tiled_solve1_scalar_held",
    "solve1_scalar_held_tile_shapes",
)
