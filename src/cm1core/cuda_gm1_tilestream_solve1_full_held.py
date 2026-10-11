"""Source-ordered tiled solve1 held momentum, theta, and moisture terms.

The historical tiled scalar-held operator intentionally excludes momentum
stress.  Momentum stress cannot be appended to that result: CM1 calls the
stress operator before scalar turbulence.  This module therefore provides a
separate, strictly typed chain for one tile:

``fresh closure -> TKE diffusion -> Rayleigh -> [active lspgrad] -> idiss ->
[active idealized rterm] -> stress -> scalars -> [active large-scale nudging]``.

The closure tile owns the live strain, diffusivity, and terrain-metric
carriers.  The typed solve1-TKE result proves that the TKE source and ``turbt``
were enqueued on the same stream immediately after that closure tile.  This
module then builds the five held output families without a host sync.  It
does not perform momentum/scalar advection, an RK update, pressure or acoustic
work, boundary-history production, a checkpoint, or a forecast.
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
    GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
    CudaGm1ChildTerrainRayleighLaunch,
    CompiledGm1ChildTerrainRayleigh,
    child_terrain_rayleigh_shapes,
    launch_cuda_gm1_child_terrain_rayleigh,
)
from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_child_stretched_scalar_turbulence import (
    CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
    CompiledGm1ScalarTurbulenceKernels,
    child_stretched_scalar_turbulence_shapes,
    launch_cuda_gm1_child_stretched_scalar_turbulence,
)
from .cuda_gm1_child_stretched_sgs_tke import (
    CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256,
    CompiledChildStretchedSgsTke,
    CudaChildStretchedLaunch,
    launch_cuda_gm1_child_stretched_sgs_stress,
)
from .cuda_gm1_solve1_lspgrad import (
    CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_SHA256,
    CudaGm1Solve1LspgradBinding,
    enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh,
    require_cuda_gm1_solve1_lspgrad_binding,
)
from .cuda_gm1_tilestream_large_scale_nudging import (
    CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_SHA256,
    CudaGm1LargeScaleNudgingBinding,
    enqueue_cuda_gm1_large_scale_nudging_after_held_composition,
    require_cuda_gm1_large_scale_nudging_binding,
)
from .cuda_gm1_solve1_rayleigh_variants import (
    CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_SHA256,
    CudaGm1Solve1RayleighVariantBinding,
    enqueue_cuda_gm1_solve1_rayleigh_variant_override,
    require_cuda_gm1_solve1_rayleigh_variant_binding,
)
from .cuda_gm1_solve1_idealized_radiation import (
    CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_SHA256,
    CudaGm1Solve1IdealizedRadiationBinding,
    enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss,
    require_cuda_gm1_solve1_idealized_radiation_binding,
)
from .cuda_gm1_scalar_turbulence import (
    Gm1ScalarTurbulenceWorkspace,
    create_cuda_gm1_scalar_turbulence_workspace,
)
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    CudaGm1TiledSolve1TkeRhsLaunch,
    Gm1TiledSolve1TkeRhsToken,
)
from .cuda_gm1_tilestream_solve1_scalar_held import (
    AUTHORITY_ANCHORS as SCALAR_AUTHORITY_ANCHORS,
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
    GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT,
    Gm1TiledSolve1ScalarHeldToken,
    SOURCE_ORDER as SCALAR_SOURCE_ORDER,
)
from .cuda_gm1_tilestream_stretched_closure import (
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_SCHEMA,
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_STATUS,
    CUDA_GM1_TILESTREAM_STRETCHED_CLOSURE_TOKEN_SCHEMA,
    CudaGm1TiledStretchedClosureLaunch,
    Gm1TiledStretchedClosureToken,
)
from .cuda_sgs_stress_tendency import sgs_stress_shapes
from .eqtset2_energy_terms import (
    EQTSET2_IDISS_RECEIPT_SCHEMA,
    Eqtset2IdissCarriers,
    Eqtset2IdissConsumer,
    Eqtset2IdissResult,
)


CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_SCHEMA = (
    "gm1.cuda-tilestream-solve1-full-held/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA = (
    "gm1.tilestream-solve1-full-held-token/v1"
)
CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_STATUS = (
    "ENQUEUED_LIVE_TILESTREAM_SOLVE1_FULL_HELD"
)
CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SOLVE1_FULL_HELD"
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
    "src/solve1.F:314-383",
    "src/solve1.F:414-454",
    "src/solve1.F:744-848",
    "src/solve1.F:917-1001",
    "src/solve1.F:1115-1240",
    "src/turbtend.F:157-692",
    "src/turbtend.F:1052-2885",
)
SOURCE_ORDER = (
    "live calcnm then calcdef then corrected fresh TKE closure",
    "same-tile TKE source plus stretched turbt diffusion",
    "positive-zero held scalar and SGS stress accumulators",
    "terrain Rayleigh establishes momentum and theta held terms",
    "eqtset=2 idiss adds live closure dissten heating to theta",
    "stretched SGS stress adds momentum stress divergence",
    "stretched scalar SGS adds theta then qv/qc/qr/qi/qs/qg",
    "Rayleigh plus SGS stress are composed into final u/v/w held terms",
)
SOURCE_ORDER_WITH_ACTIVE_LSPGRAD = (
    *SOURCE_ORDER[:4],
    "active lspgrad=1..4 adds exact large-scale pressure-gradient momentum forcing",
    *SOURCE_ORDER[4:],
)
SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION = (
    *SOURCE_ORDER[:5],
    "active rterm=1 adds exact idealized-radiation theta forcing",
    *SOURCE_ORDER[5:],
)
SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT = (
    *SOURCE_ORDER[:3],
    "active flat irdamp/hrdamp variant establishes exact Rayleigh held values",
    *SOURCE_ORDER[4:],
)
SOURCE_ORDER_WITH_ACTIVE_LARGE_SCALE_NUDGING = (
    *SOURCE_ORDER,
    "active large-scale nudging adds exact u/v/theta/qv held terms",
)
GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT = (
    "one source-ordered live tile-streamed CM1 r21.1 solve1 held pass for "
    "momentum, theta, six-species moisture, and TKE-source lineage using the "
    "corrected stretched closure; no advection, RK update, pressure/acoustic "
    "solve, boundary or MPI history, checkpoint, complete child, forecast, "
    "or five-metre meteorological product"
)


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _aggregate_lspgrad_lineage(
    modes: set[bool],
    binding_tokens: Sequence[str],
    tile_tokens: Sequence[str],
    selections: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return active-only token fields while preserving the disabled ABI."""

    if modes == {False}:
        if binding_tokens or tile_tokens or selections:
            raise ValueError("disabled lspgrad tiles carry active lineage")
        return None
    if modes != {True}:
        raise ValueError("solve1 full-held tiles mix active and disabled lspgrad")
    if not binding_tokens or len(set(binding_tokens)) != 1:
        raise ValueError("solve1 full-held tiles bind different lspgrad configs")
    if len(binding_tokens) != len(tile_tokens) or len(tile_tokens) != len(selections):
        raise ValueError("active lspgrad tile lineage inventory is incomplete")
    canonical_selections = {_canonical(value) for value in selections}
    if len(canonical_selections) != 1:
        raise ValueError("solve1 full-held tiles bind different lspgrad selections")
    return MappingProxyType(
        {
            "lspgrad_binding_operation_token_sha256": binding_tokens[0],
            "lspgrad_tile_operation_tokens_sha256": hashlib.sha256(
                _canonical(tile_tokens)
            ).hexdigest(),
            "lspgrad_selection": dict(selections[0]),
            "lspgrad_applied_after_rayleigh_before_idiss": True,
            "lspgrad_coupled_trajectory_admitted": False,
        }
    )


def _aggregate_large_scale_nudging_lineage(
    modes: set[bool],
    binding_tokens: Sequence[str],
    tile_tokens: Sequence[str],
    cpu_operator_tokens: Sequence[str],
    selections: Sequence[Mapping[str, object]],
    schedules: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return active-only nudging lineage while preserving the disabled ABI."""

    if modes == {False}:
        if binding_tokens or tile_tokens or cpu_operator_tokens or selections or schedules:
            raise ValueError("disabled large-scale-nudging tiles carry active lineage")
        return None
    if modes != {True}:
        raise ValueError(
            "solve1 full-held tiles mix active and disabled large-scale nudging"
        )
    if not binding_tokens or len(set(binding_tokens)) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different large-scale-nudging configs"
        )
    if not cpu_operator_tokens or len(set(cpu_operator_tokens)) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different large-scale-nudging CPU operators"
        )
    if not (
        len(binding_tokens)
        == len(tile_tokens)
        == len(cpu_operator_tokens)
        == len(selections)
        == len(schedules)
    ):
        raise ValueError("active large-scale-nudging tile lineage is incomplete")
    if len({_canonical(value) for value in selections}) != 1 or len(
        {_canonical(value) for value in schedules}
    ) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different large-scale-nudging schedules"
        )
    return MappingProxyType(
        {
            "large_scale_nudging_binding_operation_token_sha256": binding_tokens[0],
            "large_scale_nudging_tile_operation_tokens_sha256": hashlib.sha256(
                _canonical(tile_tokens)
            ).hexdigest(),
            "large_scale_nudging_cpu_operator_operation_token_sha256": (
                cpu_operator_tokens[0]
            ),
            "large_scale_nudging_selection": dict(selections[0]),
            "large_scale_nudging_schedule": dict(schedules[0]),
            "large_scale_nudging_applied_after_final_held_composition": True,
            "large_scale_nudging_solve3_base_update_pending": True,
            "large_scale_nudging_coupled_trajectory_admitted": False,
        }
    )


def _aggregate_idealized_radiation_lineage(
    modes: set[bool],
    binding_tokens: Sequence[str],
    tile_tokens: Sequence[str],
    cpu_operator_tokens: Sequence[str],
    selections: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return active-only rterm lineage while preserving the default ABI."""

    if modes == {False}:
        if binding_tokens or tile_tokens or cpu_operator_tokens or selections:
            raise ValueError("disabled idealized-radiation tiles carry active lineage")
        return None
    if modes != {True}:
        raise ValueError(
            "solve1 full-held tiles mix active and disabled idealized radiation"
        )
    if not binding_tokens or len(set(binding_tokens)) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different idealized-radiation configs"
        )
    if not cpu_operator_tokens or len(set(cpu_operator_tokens)) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different idealized-radiation CPU operators"
        )
    if not (
        len(binding_tokens)
        == len(tile_tokens)
        == len(cpu_operator_tokens)
        == len(selections)
    ):
        raise ValueError("active idealized-radiation tile lineage is incomplete")
    canonical_selections = {_canonical(value) for value in selections}
    if len(canonical_selections) != 1:
        raise ValueError(
            "solve1 full-held tiles bind different idealized-radiation selections"
        )
    return MappingProxyType(
        {
            "idealized_radiation_binding_operation_token_sha256": binding_tokens[0],
            "idealized_radiation_tile_operation_tokens_sha256": hashlib.sha256(
                _canonical(tile_tokens)
            ).hexdigest(),
            "idealized_radiation_cpu_operator_operation_token_sha256": (
                cpu_operator_tokens[0]
            ),
            "idealized_radiation_selection": dict(selections[0]),
            "idealized_radiation_applied_after_idiss_before_sgs": True,
            "idealized_radiation_pressure_tendency_coupled": False,
            "idealized_radiation_energy_budget_coupled": False,
            "idealized_radiation_active_cuda_raw_bit_oracle": False,
            "idealized_radiation_coupled_trajectory_admitted": False,
        }
    )


def _aggregate_rayleigh_variant_lineage(
    modes: set[bool],
    binding_tokens: Sequence[str],
    tile_tokens: Sequence[str],
    cpu_operator_tokens: Sequence[str],
    selections: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return active-only Rayleigh-variant lineage; omit default fields."""

    if modes == {False}:
        if binding_tokens or tile_tokens or cpu_operator_tokens or selections:
            raise ValueError("disabled Rayleigh-variant tiles carry active lineage")
        return None
    if modes != {True}:
        raise ValueError("solve1 full-held tiles mix active/default Rayleigh variants")
    if not binding_tokens or len(set(binding_tokens)) != 1:
        raise ValueError("solve1 full-held tiles bind different Rayleigh variants")
    if not cpu_operator_tokens or len(set(cpu_operator_tokens)) != 1:
        raise ValueError("solve1 full-held tiles bind different Rayleigh CPU operators")
    if not (
        len(binding_tokens)
        == len(tile_tokens)
        == len(cpu_operator_tokens)
        == len(selections)
    ):
        raise ValueError("active Rayleigh-variant tile lineage is incomplete")
    if len({_canonical(value) for value in selections}) != 1:
        raise ValueError("solve1 full-held tiles bind different Rayleigh selections")
    return MappingProxyType(
        {
            "rayleigh_variant_binding_operation_token_sha256": binding_tokens[0],
            "rayleigh_variant_tile_operation_tokens_sha256": hashlib.sha256(
                _canonical(tile_tokens)
            ).hexdigest(),
            "rayleigh_variant_cpu_operator_operation_token_sha256": (
                cpu_operator_tokens[0]
            ),
            "rayleigh_variant_selection": dict(selections[0]),
            "rayleigh_variant_overwrote_default_before_other_held_terms": True,
            "rayleigh_variant_active_cuda_raw_bit_oracle": False,
            "rayleigh_variant_coupled_trajectory_admitted": False,
        }
    )


_CLOSURE_STRESS_NAMES = frozenset(
    {
        "s11",
        "s22",
        "s33",
        "s12",
        "s13",
        "s23",
        "kmh",
        "kmv",
        "mh",
        "mf",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "gxu",
        "gyv",
    }
)
_CLOSURE_SCALAR_NAMES = frozenset(
    {
        "mh",
        "mf",
        "khh",
        "khv",
        "gz",
        "rgz",
        "rgzu",
        "rgzv",
        "gx",
        "gxu",
        "gy",
        "gyv",
    }
)
_STRESS_OUTPUT_NAMES = (
    "t11",
    "t22",
    "t33",
    "t12",
    "t13",
    "t23",
    "uten",
    "vten",
    "wten",
)
_STRESS_TENDENCY_NAMES = ("uten", "vten", "wten")


def solve1_full_held_tile_shapes(
    *, ni: int, nj: int, nk: int, numq: int = 6
) -> Mapping[str, tuple[int, ...]]:
    """Return exact caller-owned arrays; closure-owned carriers are omitted."""

    rayleigh = child_terrain_rayleigh_shapes(ni, nj, nk)
    scalar = child_stretched_scalar_turbulence_shapes(ni, nj, nk, numq)
    stress = sgs_stress_shapes(ni, nj, nk)
    result = dict(rayleigh)
    result.update({"pi0": (nk, nj, ni), "ppi": (nk, nj, ni)})
    for names, shapes, excluded in (
        (tuple(scalar), scalar, _CLOSURE_SCALAR_NAMES),
        (tuple(stress), stress, _CLOSURE_STRESS_NAMES),
    ):
        for name in names:
            if name in excluded:
                continue
            shape = shapes[name]
            prior = result.get(name)
            if prior is not None and prior != shape:
                raise RuntimeError(f"solve1 full-held shape collision: {name}")
            result[name] = shape
    return MappingProxyType(result)


_ARRAY_NAMES = tuple(solve1_full_held_tile_shapes(ni=8, nj=7, nk=5))


def _require_arrays(
    cupy: Any,
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, Any]:
    shapes = solve1_full_held_tile_shapes(ni=ni, nj=nj, nk=nk)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("solve1 full-held caller array inventory changed")
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
        raise ValueError("solve1 full-held caller arrays must not alias")
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


def _require_tke_launch(
    launch: object,
    *,
    closure: CudaGm1TiledStretchedClosureLaunch,
    spec: Gm1TileSpec,
    stream_pointer: int,
) -> CudaGm1TiledSolve1TkeRhsLaunch:
    if not isinstance(launch, CudaGm1TiledSolve1TkeRhsLaunch):
        raise TypeError("tke_launch must be the exact same-tile solve1 TKE result")
    receipt = launch.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_RHS_SCHEMA
        or receipt.get("status") != "ENQUEUED_LIVE_TILE_LOCAL_SOLVE1_TKE_RHS"
        or receipt.get("tile") != spec.record()
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_after_fresh_closure") is not True
        or receipt.get("vertical_disposition")
        != closure.receipt.get("vertical_disposition")
        or receipt.get("completed_nrk") != 0
        or receipt.get("closure_operation_token_sha256")
        != closure.receipt.get("operation_token_sha256")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve1 TKE tile identity, lineage, or stream changed")
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


def _require_tke_token(
    token: object,
    *,
    closure: Mapping[str, object],
    plan: Gm1TilePlan,
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledSolve1TkeRhsToken):
        raise TypeError("tke_token must be the exact completed solve1 TKE token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or receipt.get("completed_nrk") != 0
        or receipt.get("closure_operation_token_sha256")
        != closure.get("operation_token_sha256")
        or receipt.get("tile_plan_sha256") != plan.ownership_sha256
        or receipt.get("tile_count") != plan.ntiles
        or receipt.get("stream_pointer") != closure.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve1 TKE token identity, lineage, or domain changed")
    return receipt


def _closure_stress_arrays(
    closure: CudaGm1TiledStretchedClosureLaunch,
    checked: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, Any]:
    work = closure.workspace
    values = {
        name: (
            checked[name]
            if name in checked
            else closure.arrays[name]
            if name in {"kmh", "kmv"}
            else work[name]
        )
        for name in sgs_stress_shapes(ni, nj, nk)
    }
    return values


def _closure_scalar_arrays(
    closure: CudaGm1TiledStretchedClosureLaunch,
    checked: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, Any]:
    work = closure.workspace
    values = {
        name: (
            checked[name]
            if name in checked
            else closure.arrays[name]
            if name in {"khh", "khv"}
            else work[name]
        )
        for name in child_stretched_scalar_turbulence_shapes(ni, nj, nk)
    }
    return values


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve1FullHeldLaunch:
    ready_event: Any
    rayleigh: CudaGm1ChildTerrainRayleighLaunch
    idiss: Eqtset2IdissResult
    stress: CudaChildStretchedLaunch
    scalar_turbulence: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve1FullHeldToken:
    receipt: Mapping[str, object]


def launch_cuda_gm1_tiled_solve1_full_held(
    closure_launch: CudaGm1TiledStretchedClosureLaunch,
    tke_launch: CudaGm1TiledSolve1TkeRhsLaunch,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    vertical: Gm1ChildStretchedMetrics,
    dx: float,
    dy: float,
    dt: float,
    stream: Any,
    rayleigh_compiled: CompiledGm1ChildTerrainRayleigh | None = None,
    stress_compiled: CompiledChildStretchedSgsTke | None = None,
    scalar_compiled: CompiledGm1ScalarTurbulenceKernels | None = None,
    scalar_workspace: Gm1ScalarTurbulenceWorkspace | None = None,
    idiss_consumer: Eqtset2IdissConsumer | None = None,
    rayleigh_variant_binding: CudaGm1Solve1RayleighVariantBinding | None = None,
    lspgrad_binding: CudaGm1Solve1LspgradBinding | None = None,
    idealized_radiation_binding: CudaGm1Solve1IdealizedRadiationBinding
    | None = None,
    large_scale_nudging_binding: CudaGm1LargeScaleNudgingBinding | None = None,
    reference_dz: float = 100.0,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledSolve1FullHeldLaunch:
    """Enqueue the exact held solve1 source order for one corrected tile."""

    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    for name, value in (("dx", dx), ("dy", dy), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(dx) != float(dy):
        raise ValueError("solve1 full-held requires square horizontal spacing")
    if not isinstance(allow_uniform_idealized_validation, bool):
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if isinstance(reference_dz, bool) or not isinstance(reference_dz, (int, float)):
        raise TypeError("reference_dz must be numeric")
    reference_dz = float(reference_dz)
    expected_reference_dz = 200.0 if allow_uniform_idealized_validation else 100.0
    if reference_dz != expected_reference_dz:
        raise ValueError(
            "solve1 full-held reference dz differs from the selected vertical grid"
        )
    admitted = {(300, 300): 50.0, (3000, 3000): 5.0}
    if admitted.get((spec.nx, spec.ny)) != float(dx):
        raise ValueError("solve1 full-held tile belongs to an unadmitted grid")
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be the exact stretched child binding")

    stream_pointer = int(stream.ptr)
    closure_launch = _require_closure_launch(
        closure_launch, spec=spec, stream_pointer=stream_pointer
    )
    tke_launch = _require_tke_launch(
        tke_launch,
        closure=closure_launch,
        spec=spec,
        stream_pointer=stream_pointer,
    )
    active_optional_count = sum(
        value is not None
        for value in (
            rayleigh_variant_binding,
            lspgrad_binding,
            idealized_radiation_binding,
            large_scale_nudging_binding,
        )
    )
    if active_optional_count > 1:
        raise ValueError(
            "solve1 full-held admits only one active Rayleigh/lspgrad/rterm/nudging variant"
        )
    if (
        closure_launch.receipt.get("sgs_startup_ramp_active") is True
        and active_optional_count
    ):
        raise ValueError(
            "active SGS startup ramp has no admitted simultaneous optional solve forcing"
        )
    if rayleigh_variant_binding is not None:
        if not allow_uniform_idealized_validation:
            raise ValueError("active Rayleigh variant requires flat idealized geometry")
        rayleigh_variant_binding = require_cuda_gm1_solve1_rayleigh_variant_binding(
            rayleigh_variant_binding, spec=spec, stream=stream
        )
    if lspgrad_binding is not None:
        lspgrad_binding = require_cuda_gm1_solve1_lspgrad_binding(
            lspgrad_binding,
            spec=spec,
            stream=stream,
        )
    if idealized_radiation_binding is not None:
        if not allow_uniform_idealized_validation:
            raise ValueError(
                "active idealized radiation requires the flat uniform idealized path"
            )
        idealized_radiation_binding = (
            require_cuda_gm1_solve1_idealized_radiation_binding(
                idealized_radiation_binding,
                spec=spec,
                stream=stream,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=reference_dz,
            )
        )
    if large_scale_nudging_binding is not None:
        large_scale_nudging_binding = require_cuda_gm1_large_scale_nudging_binding(
            large_scale_nudging_binding,
            stream=stream,
            nx=spec.nx,
            ny=spec.ny,
            nk=75,
        )
    ni, nj, nk = spec.cnx, spec.cny, 75
    cupy = importlib.import_module("cupy")
    checked = _require_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)

    caller_pointers = {int(value.data.ptr) for value in checked.values()}
    closure_pointers = {
        int(value.data.ptr) for value in closure_launch.workspace.values()
    }
    if caller_pointers.intersection(closure_pointers):
        raise ValueError("solve1 full-held arrays alias live closure carriers")
    if int(tke_launch.tketen.data.ptr) != int(closure_launch.workspace["tketen"].data.ptr):
        raise ValueError("solve1 TKE launch no longer owns this closure tketen")

    scalar_workspace = (
        create_cuda_gm1_scalar_turbulence_workspace(ni=ni, nj=nj, nk=nk)
        if scalar_workspace is None
        else scalar_workspace
    )
    if idiss_consumer is None:
        idiss_consumer = Eqtset2IdissConsumer(nx=ni, ny=nj, nz=nk)
    if not isinstance(idiss_consumer, Eqtset2IdissConsumer):
        raise TypeError("idiss_consumer must be Eqtset2IdissConsumer")

    stress_arrays = _closure_stress_arrays(
        closure_launch, checked, ni=ni, nj=nj, nk=nk
    )
    scalar_arrays = _closure_scalar_arrays(
        closure_launch, checked, ni=ni, nj=nj, nk=nk
    )
    stress_metrics = {
        "sigma": vertical.sigma,
        "sigmaf": vertical.sigmaf,
        "rds": closure_launch.workspace["rds"],
        "rdsf": closure_launch.workspace["rdsf"],
        "ready_event": tke_launch.ready_event,
    }

    with stream:
        stream.wait_event(tke_launch.ready_event)
        checked["thten1"].fill(cupy.float32(0.0))
        checked["qten"].fill(cupy.float32(0.0))
        for name in _STRESS_TENDENCY_NAMES:
            checked[name].fill(cupy.float32(0.0))
        rayleigh = launch_cuda_gm1_child_terrain_rayleigh(
            {name: checked[name] for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS},
            ni=ni,
            nj=nj,
            nk=nk,
            stream=stream,
            compiled=rayleigh_compiled,
        )
        rayleigh_variant = (
            enqueue_cuda_gm1_solve1_rayleigh_variant_override(
                rayleigh_variant_binding,
                checked,
                spec=spec,
                stream=stream,
            )
            if rayleigh_variant_binding is not None
            else None
        )
        lspgrad = (
            enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(
                lspgrad_binding,
                checked,
                spec=spec,
                stream=stream,
            )
            if lspgrad_binding is not None
            else None
        )
        idiss = idiss_consumer.apply(
            Eqtset2IdissCarriers(
                thten1=checked["thten1"],
                dissten=closure_launch.arrays["dissten"],
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
        idealized_radiation = (
            enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(
                idealized_radiation_binding,
                checked,
                spec=spec,
                stream=stream,
                dt=dt,
                dx=dx,
                dy=dy,
                dz=reference_dz,
            )
            if idealized_radiation_binding is not None
            else None
        )
        stress = launch_cuda_gm1_child_stretched_sgs_stress(
            stress_arrays,
            metrics=stress_metrics,
            ni=ni,
            nj=nj,
            nk=nk,
            dx=float(dx),
            dy=float(dy),
            dz=reference_dz,
            ztop=15000.0,
            dt=float(dt),
            stream=stream,
            compiled=stress_compiled,
            allow_compact_validation=True,
            allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        )
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
            allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        )
        cupy.add(checked["uten1"], checked["uten"], out=checked["uten"])
        cupy.add(checked["vten1"], checked["vten"], out=checked["vten"])
        cupy.add(
            checked["wten1"], checked["wten"][:nk], out=checked["wten"][:nk]
        )
        large_scale_nudging = (
            enqueue_cuda_gm1_large_scale_nudging_after_held_composition(
                large_scale_nudging_binding,
                checked,
                spec=spec,
                stream=stream,
            )
            if large_scale_nudging_binding is not None
            else None
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)

    if idiss.receipt.get("schema") != EQTSET2_IDISS_RECEIPT_SCHEMA:
        raise ValueError("idiss receipt schema changed")
    active_lspgrad = lspgrad is not None
    active_idealized_radiation = idealized_radiation is not None
    active_rayleigh_variant = rayleigh_variant is not None
    active_large_scale_nudging = large_scale_nudging is not None
    source_order = (
        SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT
        if active_rayleigh_variant
        else SOURCE_ORDER_WITH_ACTIVE_LSPGRAD
        if active_lspgrad
        else SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION
        if active_idealized_radiation
        else SOURCE_ORDER_WITH_ACTIVE_LARGE_SCALE_NUDGING
        if active_large_scale_nudging
        else SOURCE_ORDER
    )
    suboperator_receipt_sha256 = {
        "rayleigh": hashlib.sha256(_canonical(rayleigh.receipt)).hexdigest(),
        "idiss": hashlib.sha256(_canonical(dict(idiss.receipt))).hexdigest(),
        "sgs_stress": hashlib.sha256(_canonical(stress.receipt)).hexdigest(),
        "scalar_turbulence": hashlib.sha256(_canonical(scalar.receipt)).hexdigest(),
    }
    operator_source_sha256 = {
        "rayleigh": GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
        "stretched_sgs_stress": CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256,
        "stretched_scalar_turbulence": (
            CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
        ),
    }
    if active_lspgrad:
        suboperator_receipt_sha256["lspgrad"] = hashlib.sha256(
            _canonical(lspgrad)
        ).hexdigest()
        operator_source_sha256["lspgrad_live_binding"] = (
            CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_SHA256
        )
    if active_idealized_radiation:
        suboperator_receipt_sha256["idealized_radiation"] = hashlib.sha256(
            _canonical(idealized_radiation)
        ).hexdigest()
        operator_source_sha256["idealized_radiation_live_binding"] = (
            CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_SHA256
        )
    if active_rayleigh_variant:
        suboperator_receipt_sha256["rayleigh_variant"] = hashlib.sha256(
            _canonical(rayleigh_variant)
        ).hexdigest()
        operator_source_sha256["rayleigh_variant_live_binding"] = (
            CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_SHA256
        )
    if active_large_scale_nudging:
        suboperator_receipt_sha256["large_scale_nudging"] = hashlib.sha256(
            _canonical(large_scale_nudging)
        ).hexdigest()
        operator_source_sha256["large_scale_nudging_live_binding"] = (
            CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_SHA256
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": stream_pointer,
        "closure_tile_operation_token_sha256": closure_launch.receipt[
            "operation_token_sha256"
        ],
        "solve1_tke_tile_operation_token_sha256": tke_launch.receipt[
            "operation_token_sha256"
        ],
        "source_order": list(source_order),
        "suboperator_receipt_sha256": suboperator_receipt_sha256,
        "operator_source_sha256": operator_source_sha256,
        "positive_zero_initialized": ["thten1", "qten", "uten", "vten", "wten"],
        "momentum_composition": {
            "uten": "Rayleigh uten1 + SGS-stress uten",
            "vten": "Rayleigh vten1 + SGS-stress vten",
            "wten_k_0_to_nk_minus_1": "Rayleigh wten1 + SGS-stress wten",
            "wten_top_face": "SGS-stress wten only",
        },
        "same_stream_no_host_sync": True,
        "vertical_disposition": (
            "source-audited-flat-idealized-uniform-200m"
            if allow_uniform_idealized_validation
            else "sealed-stretched-child"
        ),
        "momentum_or_scalar_advection_performed": False,
        "rk_stage_performed": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": [
                *AUTHORITY_ANCHORS,
                *(
                    [
                        "src/param.F:8631-8778",
                        "src/testcase_simple_phys.F:245-299",
                        "src/solve1.F:390-454",
                    ]
                    if active_rayleigh_variant
                    else []
                ),
                *(["src/solve1.F:536-681"] if active_lspgrad else []),
                *(
                    ["src/solve1.F:744-850,1002-1056"]
                    if active_idealized_radiation
                    else []
                ),
                *(
                    ["src/solve1.F:252-288,390-398,1608-1693"]
                    if active_large_scale_nudging
                    else []
                ),
            ],
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT,
    }
    if active_lspgrad:
        receipt["momentum_composition"] = {
            "uten": "Rayleigh uten1 + lspgrad uten1 + SGS-stress uten",
            "vten": "Rayleigh vten1 + lspgrad vten1 + SGS-stress vten",
            "wten_k_0_to_nk_minus_1": "Rayleigh wten1 + SGS-stress wten",
            "wten_top_face": "SGS-stress wten only",
        }
        receipt.update(
            {
                "lspgrad_binding_operation_token_sha256": lspgrad[
                    "binding_operation_token_sha256"
                ],
                "lspgrad_tile_operation_token_sha256": lspgrad[
                    "operation_token_sha256"
                ],
                "lspgrad_selection": {
                    "active": dict(lspgrad["selection"]),
                    "companion": dict(lspgrad["companion_selection"]),
                },
                "lspgrad_applied_after_rayleigh_before_idiss": True,
                "lspgrad_coupled_trajectory_admitted": False,
            }
        )
    if active_idealized_radiation:
        receipt.update(
            {
                "idealized_radiation_binding_operation_token_sha256": (
                    idealized_radiation["binding_operation_token_sha256"]
                ),
                "idealized_radiation_tile_operation_token_sha256": (
                    idealized_radiation["operation_token_sha256"]
                ),
                "idealized_radiation_cpu_operator_operation_token_sha256": (
                    idealized_radiation["cpu_operator_operation_token_sha256"]
                ),
                "idealized_radiation_selection": {
                    "active": dict(idealized_radiation["selection"]),
                    "companion": dict(
                        idealized_radiation["companion_selection"]
                    ),
                },
                "idealized_radiation_applied_after_idiss_before_sgs": True,
                "idealized_radiation_pressure_tendency_coupled": False,
                "idealized_radiation_energy_budget_coupled": False,
                "idealized_radiation_active_cuda_raw_bit_oracle": False,
                "idealized_radiation_coupled_trajectory_admitted": False,
            }
        )
    if active_rayleigh_variant:
        receipt.update(
            {
                "rayleigh_variant_binding_operation_token_sha256": (
                    rayleigh_variant["binding_operation_token_sha256"]
                ),
                "rayleigh_variant_tile_operation_token_sha256": (
                    rayleigh_variant["operation_token_sha256"]
                ),
                "rayleigh_variant_cpu_operator_operation_token_sha256": (
                    rayleigh_variant["cpu_operator_operation_token_sha256"]
                ),
                "rayleigh_variant_selection": {
                    "active": dict(rayleigh_variant["selection"]),
                    "companion": dict(rayleigh_variant["companion_selection"]),
                },
                "rayleigh_variant_overwrote_default_before_other_held_terms": True,
                "rayleigh_variant_active_cuda_raw_bit_oracle": False,
                "rayleigh_variant_coupled_trajectory_admitted": False,
            }
        )
    if active_large_scale_nudging:
        receipt.update(
            {
                "large_scale_nudging_binding_operation_token_sha256": (
                    large_scale_nudging["binding_operation_token_sha256"]
                ),
                "large_scale_nudging_tile_operation_token_sha256": (
                    large_scale_nudging["operation_token_sha256"]
                ),
                "large_scale_nudging_cpu_operator_operation_token_sha256": (
                    large_scale_nudging_binding.receipt[
                        "cpu_operator_operation_token_sha256"
                    ]
                ),
                "large_scale_nudging_selection": dict(
                    large_scale_nudging["selection"]
                ),
                "large_scale_nudging_schedule": dict(
                    large_scale_nudging["schedule"]
                ),
                "large_scale_nudging_applied": dict(
                    large_scale_nudging["applied"]
                ),
                "large_scale_nudging_applied_after_final_held_composition": True,
                "large_scale_nudging_solve3_base_update_pending": True,
                "large_scale_nudging_active_cuda_raw_bit_oracle": False,
                "large_scale_nudging_coupled_trajectory_admitted": False,
            }
        )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledSolve1FullHeldLaunch(
        event,
        rayleigh,
        idiss,
        stress,
        scalar,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_tiled_solve1_full_held(
    closure_token: Gm1TiledStretchedClosureToken,
    tke_token: Gm1TiledSolve1TkeRhsToken,
    launches: Sequence[CudaGm1TiledSolve1FullHeldLaunch],
    *,
    plan: Gm1TilePlan,
    outputs: Mapping[str, Any],
) -> Gm1TiledSolve1FullHeldToken:
    """Admit the 300/50 exact-once host scatter; five metres stays refused."""

    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("solve1 full-held admission requires the 300 grid oracle")
    closure = _require_closure_token(closure_token, plan=plan)
    tke = _require_tke_token(tke_token, closure=closure, plan=plan)
    if len(launches) != plan.ntiles:
        raise ValueError("solve1 full-held launch count differs from tile plan")

    tile_tokens: list[str] = []
    closure_tile_tokens: list[str] = []
    tke_tile_tokens: list[str] = []
    lspgrad_binding_tokens: list[str] = []
    lspgrad_tile_tokens: list[str] = []
    lspgrad_selections: list[Mapping[str, object]] = []
    lspgrad_modes: set[bool] = set()
    radiation_binding_tokens: list[str] = []
    radiation_tile_tokens: list[str] = []
    radiation_cpu_operator_tokens: list[str] = []
    radiation_selections: list[Mapping[str, object]] = []
    radiation_modes: set[bool] = set()
    rayleigh_binding_tokens: list[str] = []
    rayleigh_tile_tokens: list[str] = []
    rayleigh_cpu_operator_tokens: list[str] = []
    rayleigh_selections: list[Mapping[str, object]] = []
    rayleigh_modes: set[bool] = set()
    nudging_binding_tokens: list[str] = []
    nudging_tile_tokens: list[str] = []
    nudging_cpu_operator_tokens: list[str] = []
    nudging_selections: list[Mapping[str, object]] = []
    nudging_schedules: list[Mapping[str, object]] = []
    nudging_modes: set[bool] = set()
    streams: set[int] = set()
    owned_words = {name: 0 for name in ("uten", "vten", "wten", "thten1", "qten")}
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve1FullHeldLaunch):
            raise TypeError("solve1 full-held launches contain a foreign result")
        receipt = launch.receipt
        active_lspgrad = "lspgrad_binding_operation_token_sha256" in receipt
        active_idealized_radiation = (
            "idealized_radiation_binding_operation_token_sha256" in receipt
        )
        active_rayleigh_variant = (
            "rayleigh_variant_binding_operation_token_sha256" in receipt
        )
        active_large_scale_nudging = (
            "large_scale_nudging_binding_operation_token_sha256" in receipt
        )
        if sum(
            (
                active_lspgrad,
                active_idealized_radiation,
                active_rayleigh_variant,
                active_large_scale_nudging,
            )
        ) > 1:
            raise ValueError("solve1 full-held tile binds simultaneous optional variants")
        lspgrad_modes.add(active_lspgrad)
        radiation_modes.add(active_idealized_radiation)
        rayleigh_modes.add(active_rayleigh_variant)
        nudging_modes.add(active_large_scale_nudging)
        expected_source_order = (
            SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT
            if active_rayleigh_variant
            else SOURCE_ORDER_WITH_ACTIVE_LSPGRAD
            if active_lspgrad
            else SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION
            if active_idealized_radiation
            else SOURCE_ORDER_WITH_ACTIVE_LARGE_SCALE_NUDGING
            if active_large_scale_nudging
            else SOURCE_ORDER
        )
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_SCHEMA
            or receipt.get("status") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_STATUS
            or receipt.get("completed_nrk") != 0
            or receipt.get("tile") != spec.record()
            or receipt.get("same_stream_no_host_sync") is not True
            or receipt.get("source_order") != list(expected_source_order)
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("solve1 full-held tile identity or cadence changed")
        if active_lspgrad:
            selection = receipt.get("lspgrad_selection")
            if (
                not isinstance(selection, Mapping)
                or not isinstance(selection.get("active"), Mapping)
                or selection["active"].get("lspgrad") not in (1, 2, 3, 4)
                or selection.get("companion")
                != {
                    "icor": 0,
                    "axisymmetric": 0,
                    "adaptive_lspgrad": False,
                }
                or receipt.get("lspgrad_applied_after_rayleigh_before_idiss")
                is not True
                or receipt.get("lspgrad_coupled_trajectory_admitted") is not False
                or not isinstance(
                    receipt.get("lspgrad_binding_operation_token_sha256"), str
                )
                or not isinstance(
                    receipt.get("lspgrad_tile_operation_token_sha256"), str
                )
            ):
                raise ValueError("active lspgrad tile identity changed")
            lspgrad_binding_tokens.append(
                str(receipt["lspgrad_binding_operation_token_sha256"])
            )
            lspgrad_tile_tokens.append(
                str(receipt["lspgrad_tile_operation_token_sha256"])
            )
            lspgrad_selections.append(selection)
        if active_idealized_radiation:
            selection = receipt.get("idealized_radiation_selection")
            if (
                not isinstance(selection, Mapping)
                or not isinstance(selection.get("active"), Mapping)
                or selection["active"].get("rterm") != 1
                or selection["active"].get("radopt") != 0
                or selection["active"].get("eqtset") != 2
                or selection["active"].get("imoist") != 1
                or selection["active"].get("iice") != 1
                or selection["active"].get("categories")
                != {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6}
                or selection["active"].get("axisymmetric") != 0
                or selection["active"].get("diagnostic") is not False
                or selection["active"].get("mpi") is not False
                or selection.get("companion")
                != {
                    "lspgrad": 0,
                    "icor": 0,
                    "adaptive_dt": False,
                    "axisymmetric_outer_coordinate": False,
                }
                or receipt.get(
                    "idealized_radiation_applied_after_idiss_before_sgs"
                )
                is not True
                or receipt.get("idealized_radiation_pressure_tendency_coupled")
                is not False
                or receipt.get("idealized_radiation_energy_budget_coupled")
                is not False
                or receipt.get("idealized_radiation_active_cuda_raw_bit_oracle")
                is not False
                or receipt.get(
                    "idealized_radiation_coupled_trajectory_admitted"
                )
                is not False
                or not isinstance(
                    receipt.get(
                        "idealized_radiation_binding_operation_token_sha256"
                    ),
                    str,
                )
                or not isinstance(
                    receipt.get("idealized_radiation_tile_operation_token_sha256"),
                    str,
                )
                or not isinstance(
                    receipt.get(
                        "idealized_radiation_cpu_operator_operation_token_sha256"
                    ),
                    str,
                )
            ):
                raise ValueError("active idealized-radiation tile identity changed")
            radiation_binding_tokens.append(
                str(
                    receipt[
                        "idealized_radiation_binding_operation_token_sha256"
                    ]
                )
            )
            radiation_tile_tokens.append(
                str(receipt["idealized_radiation_tile_operation_token_sha256"])
            )
            radiation_cpu_operator_tokens.append(
                str(
                    receipt[
                        "idealized_radiation_cpu_operator_operation_token_sha256"
                    ]
                )
            )
            radiation_selections.append(selection)
        if active_rayleigh_variant:
            selection = receipt.get("rayleigh_variant_selection")
            active = selection.get("active") if isinstance(selection, Mapping) else None
            if (
                not isinstance(active, Mapping)
                or active.get("irdamp") not in (0, 1, 2)
                or active.get("hrdamp_input") not in (0, 1)
                or (active.get("irdamp"), active.get("hrdamp_input")) == (0, 0)
                or active.get("hrdamp_effective") not in (0, 1, 2)
                or active.get("terrain_flag") is not False
                or active.get("axisymmetric") != 0
                or not isinstance(active.get("parameter_f32_bits"), Mapping)
                or tuple(active["parameter_f32_bits"])
                != (
                    "rdalpha_s_inv",
                    "zd_m",
                    "maxz_m",
                    "xhd_m",
                    "minx_m",
                    "maxx_m",
                    "miny_m",
                    "maxy_m",
                    "dx_m",
                    "dy_m",
                )
                or selection.get("companion")
                != {
                    "lspgrad": 0,
                    "rterm": 0,
                    "mpi": False,
                    "diagnostics": False,
                    "adaptive_dt": False,
                }
                or receipt.get(
                    "rayleigh_variant_overwrote_default_before_other_held_terms"
                )
                is not True
                or receipt.get("rayleigh_variant_active_cuda_raw_bit_oracle")
                is not False
                or receipt.get("rayleigh_variant_coupled_trajectory_admitted")
                is not False
                or not isinstance(
                    receipt.get("rayleigh_variant_binding_operation_token_sha256"),
                    str,
                )
                or not isinstance(
                    receipt.get("rayleigh_variant_tile_operation_token_sha256"),
                    str,
                )
                or not isinstance(
                    receipt.get(
                        "rayleigh_variant_cpu_operator_operation_token_sha256"
                    ),
                    str,
                )
            ):
                raise ValueError("active Rayleigh-variant tile identity changed")
            rayleigh_binding_tokens.append(
                str(receipt["rayleigh_variant_binding_operation_token_sha256"])
            )
            rayleigh_tile_tokens.append(
                str(receipt["rayleigh_variant_tile_operation_token_sha256"])
            )
            rayleigh_cpu_operator_tokens.append(
                str(
                    receipt[
                        "rayleigh_variant_cpu_operator_operation_token_sha256"
                    ]
                )
            )
            rayleigh_selections.append(selection)
        if active_large_scale_nudging:
            selection = receipt.get("large_scale_nudging_selection")
            schedule = receipt.get("large_scale_nudging_schedule")
            applied = receipt.get("large_scale_nudging_applied")
            if (
                not isinstance(selection, Mapping)
                or selection.get("do_lsnudge") is not True
                or not isinstance(schedule, Mapping)
                or not isinstance(schedule.get("donudge"), bool)
                or not isinstance(applied, Mapping)
                or tuple(applied) != ("u", "v", "th", "qv")
                or receipt.get(
                    "large_scale_nudging_applied_after_final_held_composition"
                )
                is not True
                or receipt.get("large_scale_nudging_solve3_base_update_pending")
                is not True
                or receipt.get("large_scale_nudging_active_cuda_raw_bit_oracle")
                is not False
                or receipt.get("large_scale_nudging_coupled_trajectory_admitted")
                is not False
                or not isinstance(
                    receipt.get(
                        "large_scale_nudging_binding_operation_token_sha256"
                    ),
                    str,
                )
                or not isinstance(
                    receipt.get("large_scale_nudging_tile_operation_token_sha256"),
                    str,
                )
                or not isinstance(
                    receipt.get(
                        "large_scale_nudging_cpu_operator_operation_token_sha256"
                    ),
                    str,
                )
            ):
                raise ValueError("active large-scale-nudging tile identity changed")
            nudging_binding_tokens.append(
                str(receipt["large_scale_nudging_binding_operation_token_sha256"])
            )
            nudging_tile_tokens.append(
                str(receipt["large_scale_nudging_tile_operation_token_sha256"])
            )
            nudging_cpu_operator_tokens.append(
                str(
                    receipt[
                        "large_scale_nudging_cpu_operator_operation_token_sha256"
                    ]
                )
            )
            nudging_selections.append(selection)
            nudging_schedules.append(schedule)
        tile_tokens.append(str(receipt["operation_token_sha256"]))
        closure_tile_tokens.append(
            str(receipt["closure_tile_operation_token_sha256"])
        )
        tke_tile_tokens.append(str(receipt["solve1_tke_tile_operation_token_sha256"]))
        streams.add(int(receipt["stream_pointer"]))
        for name, variant, levels, species in (
            ("uten", "u", 75, 1),
            ("vten", "v", 75, 1),
            ("wten", "mass", 76, 1),
            ("thten1", "mass", 75, 1),
            ("qten", "mass", 75, 6),
        ):
            y0, y1, x0, x1 = spec.owned_rect(variant)
            owned_words[name] += levels * species * (y1 - y0) * (x1 - x0)
    if len(streams) != 1 or next(iter(streams)) != closure.get("stream_pointer"):
        raise ValueError("closure, TKE, and solve1 full-held streams differ")
    if next(iter(streams)) != tke.get("stream_pointer"):
        raise ValueError("solve1 full-held stream differs from admitted TKE")
    if hashlib.sha256(_canonical(sorted(closure_tile_tokens))).hexdigest() != closure.get(
        "tile_operation_tokens_sha256"
    ):
        raise ValueError("solve1 full-held launches do not bind every closure tile")
    if hashlib.sha256(_canonical(tke_tile_tokens)).hexdigest() != tke.get(
        "tile_operation_tokens_sha256"
    ):
        raise ValueError("solve1 full-held launches do not bind every TKE tile")
    lspgrad_lineage = _aggregate_lspgrad_lineage(
        lspgrad_modes,
        lspgrad_binding_tokens,
        lspgrad_tile_tokens,
        lspgrad_selections,
    )
    active_lspgrad = lspgrad_lineage is not None
    radiation_lineage = _aggregate_idealized_radiation_lineage(
        radiation_modes,
        radiation_binding_tokens,
        radiation_tile_tokens,
        radiation_cpu_operator_tokens,
        radiation_selections,
    )
    active_idealized_radiation = radiation_lineage is not None
    rayleigh_lineage = _aggregate_rayleigh_variant_lineage(
        rayleigh_modes,
        rayleigh_binding_tokens,
        rayleigh_tile_tokens,
        rayleigh_cpu_operator_tokens,
        rayleigh_selections,
    )
    active_rayleigh_variant = rayleigh_lineage is not None
    nudging_lineage = _aggregate_large_scale_nudging_lineage(
        nudging_modes,
        nudging_binding_tokens,
        nudging_tile_tokens,
        nudging_cpu_operator_tokens,
        nudging_selections,
        nudging_schedules,
    )
    active_large_scale_nudging = nudging_lineage is not None

    np = importlib.import_module("numpy")
    expected = {
        "uten": (75, 300, 301),
        "vten": (75, 301, 300),
        "wten": (76, 300, 300),
        "thten1": (75, 300, 300),
        "qten": (6, 75, 300, 300),
    }
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected):
        raise ValueError("solve1 full-held output inventory changed")
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
        if owned_words[name] != math.prod(shape):
            raise ValueError(f"{name} host scatter coverage is not exact once")
        hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()

    source_order = (
        SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT
        if active_rayleigh_variant
        else SOURCE_ORDER_WITH_ACTIVE_LSPGRAD
        if active_lspgrad
        else SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION
        if active_idealized_radiation
        else SOURCE_ORDER_WITH_ACTIVE_LARGE_SCALE_NUDGING
        if active_large_scale_nudging
        else SOURCE_ORDER
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "closure_operation_token_sha256": closure["operation_token_sha256"],
        "solve1_tke_operation_token_sha256": tke["operation_token_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(tile_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "same_stream_per_tile_after_fresh_closure_and_tke": True,
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in expected.items()},
        "output_sha256": hashes,
        "output_words": sum(math.prod(shape) for shape in expected.values()),
        "source_order": list(source_order),
        "ready_for": (
            "typed nrk=1 momentum/scalar advection only after matching live "
            "pre-acoustic momentum and post-acoustic flux/divergence tokens"
        ),
        "advection_or_rk_performed": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "historical scalar-held token substituted for full held physics",
            "SGS stress appended after scalar turbulence",
            "solve1 held launch without same-tile TKE-source lineage",
            "momentum or scalar RK without fresh stage prerequisites",
            "global psolver tilestream claim",
            "MPI exchange or open-history production",
            "complete child, forecast, or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "anchors": [
                *AUTHORITY_ANCHORS,
                *(
                    [
                        "src/param.F:8631-8778",
                        "src/testcase_simple_phys.F:245-299",
                        "src/solve1.F:390-454",
                    ]
                    if active_rayleigh_variant
                    else []
                ),
                *(["src/solve1.F:536-681"] if active_lspgrad else []),
                *(
                    ["src/solve1.F:744-850,1002-1056"]
                    if active_idealized_radiation
                    else []
                ),
                *(
                    ["src/solve1.F:252-288,390-398,1608-1693"]
                    if active_large_scale_nudging
                    else []
                ),
            ],
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT,
    }
    if lspgrad_lineage is not None:
        receipt.update(lspgrad_lineage)
    if radiation_lineage is not None:
        receipt.update(radiation_lineage)
    if rayleigh_lineage is not None:
        receipt.update(rayleigh_lineage)
    if nudging_lineage is not None:
        receipt.update(nudging_lineage)
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve1FullHeldToken(MappingProxyType(receipt))


def project_gm1_tiled_solve1_scalar_held_from_full(
    full_held_token: Gm1TiledSolve1FullHeldToken,
    outputs: Mapping[str, Any],
    *,
    plan: Gm1TilePlan,
    stream: Any,
) -> Gm1TiledSolve1ScalarHeldToken:
    """Project the exact scalar subset from a stronger live full-held token."""

    if type(full_held_token) is not Gm1TiledSolve1FullHeldToken:
        raise TypeError("full_held_token must be the exact full-held token")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("scalar-held projection requires the 300 grid oracle")
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be the exact explicit live CUDA stream")
    full = full_held_token.receipt
    full_shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "thten1": [75, 300, 300],
        "qten": [6, 75, 300, 300],
    }
    full_hashes = full.get("output_sha256")
    if (
        full.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA
        or full.get("status") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS
        or full.get("completed_nrk") != 0
        or full.get("tile_plan_sha256") != plan.ownership_sha256
        or full.get("tile_count") != plan.ntiles
        or full.get("stream_pointer") != int(stream.ptr)
        or full.get("host_scatter_exact_once") is not True
        or full.get("output_shapes") != full_shapes
        or not isinstance(full_hashes, Mapping)
        or tuple(full_hashes) != tuple(full_shapes)
        or full.get("operation_token_sha256") != _operation_token(full)
    ):
        raise ValueError("full-held token identity, stream, plan, or output map changed")

    np = importlib.import_module("numpy")
    expected = {
        "thten1": (75, 300, 300),
        "qten": (6, 75, 300, 300),
    }
    if not isinstance(outputs, Mapping) or tuple(outputs) != tuple(expected):
        raise ValueError("scalar-held projection output inventory or order changed")
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
        raw_sha = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if full_hashes.get(name) != raw_sha:
            raise ValueError(f"{name} bytes differ from the exact full-held token")
        hashes[name] = raw_sha

    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
        "completed_nrk": 0,
        "projected_from_solve1_full_held_token_sha256": full[
            "operation_token_sha256"
        ],
        "closure_operation_token_sha256": full[
            "closure_operation_token_sha256"
        ],
        "solve1_tke_operation_token_sha256": full[
            "solve1_tke_operation_token_sha256"
        ],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": full[
            "tile_operation_tokens_sha256"
        ],
        "tile_operation_token_kind": "solve1_full_held",
        "stream_pointer": int(stream.ptr),
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in expected.items()},
        "output_sha256": hashes,
        "output_words": 7 * 75 * 300 * 300,
        "source_order": [
            *SCALAR_SOURCE_ORDER,
            "project the exact scalar subset only after full-held admission",
        ],
        "full_held_source_order_bound": True,
        "ready_for": (
            "typed scalar/moist RK1 only after matching post-acoustic mass-flux "
            "and scalar theta-divergence tokens"
        ),
        "scalar_advection_or_rk_performed": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "historical scalar-held token substitution without a live full-held token",
            "foreign thten1/qten bytes or stream",
            "scalar RK without matching live post-acoustic flux and theta divergence",
            "RK2/RK3 reuse without fresh same-stage prerequisites",
            "full forecast or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve1_sha256": AUTHORITY_SOLVE1_SHA256,
            "turbtend_sha256": AUTHORITY_TURBTEND_SHA256,
            "scalar_anchors": list(SCALAR_AUTHORITY_ANCHORS),
            "full_held_anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SOLVE1_SCALAR_HELD_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve1ScalarHeldToken(MappingProxyType(receipt))


def five_meter_solve1_full_held_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    """Report incremental tile memory while refusing a live 5 m token."""

    for name, value in (("tile_nx", tile_nx), ("tile_ny", tile_ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    ni, nj = tile_nx + 6, tile_ny + 6
    caller_shapes = solve1_full_held_tile_shapes(ni=ni, nj=nj, nk=nk)
    scalar_workspace_shapes = {
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
    idiss_shapes = {
        name: (nk, nj, ni) for name in ("ql", "qice", "cvm", "work", "epsd")
    }

    def byte_count(values: Mapping[str, tuple[int, ...]]) -> int:
        return 4 * sum(math.prod(shape) for shape in values.values())

    record: dict[str, object] = {
        "schema": "gm1.hybrid-solve1-full-held-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "compute_tile": [nj, ni],
        "caller_array_bytes_each": byte_count(caller_shapes),
        "closure_strain_km_static_reused": sorted(
            _CLOSURE_STRESS_NAMES | _CLOSURE_SCALAR_NAMES
        ),
        "scalar_workspace_bytes_each": byte_count(scalar_workspace_shapes),
        "idiss_workspace_bytes_each": byte_count(idiss_shapes),
        "tile_streamable": True,
        "live_five_meter_full_held_token_available": False,
        "completed_rk_stage_available": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "claim_limit": GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT,
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
    "CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_STATUS",
    "CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS",
    "CudaGm1TiledSolve1FullHeldLaunch",
    "GM1_TILESTREAM_SOLVE1_FULL_HELD_CLAIM_LIMIT",
    "Gm1TiledSolve1FullHeldToken",
    "SOURCE_ORDER",
    "SOURCE_ORDER_WITH_ACTIVE_IDEALIZED_RADIATION",
    "SOURCE_ORDER_WITH_ACTIVE_LARGE_SCALE_NUDGING",
    "SOURCE_ORDER_WITH_ACTIVE_LSPGRAD",
    "SOURCE_ORDER_WITH_ACTIVE_RAYLEIGH_VARIANT",
    "finish_cuda_gm1_tiled_solve1_full_held",
    "five_meter_solve1_full_held_inventory",
    "launch_cuda_gm1_tiled_solve1_full_held",
    "project_gm1_tiled_solve1_scalar_held_from_full",
    "solve1_full_held_tile_shapes",
)
