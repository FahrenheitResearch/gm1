"""Tile-streamed post-sound ``getdiv`` prerequisite for scalar RK1.

CM1 uses the small-step-averaged velocities twice after ``sound``.  The
already sealed post-acoustic tilestream lane converts them to mass flux and
``divx`` for TKE.  Before that destructive conversion, ``solve2`` also calls
``getdiv`` and retains ``t11 * thterm`` for the theta equation.  This module
types and executes only that missing scalar-RK prerequisite.

The upstream averaged-velocity token remains unchanged and its narrow
``ready_for`` claim is not broadened.  This module emits a distinct token for
the theta-divergence result.  It does not perform scalar advection, an RK
update, boundaries, TKE, pressure, or a complete model step.  In particular,
the five-metre path remains refused because no admitted 3000x3000 acoustic
velocity token exists yet.
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

from .cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_PROOF_SHA256,
    Gm1ChildStretchedMetrics,
)
from .cuda_gm1_child_stretched_moist import (
    GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256,
    CompiledGm1ChildStretchedMoist,
    compile_cuda_gm1_child_stretched_moist,
)
from .cuda_gm1_tilestream import (
    CM1_OPEN_NGXY,
    Gm1TileLocalSourceBinding,
    Gm1TilePlan,
    Gm1TileSpec,
    fill_gm1_tile_local_source,
    tile_host_buffer_shapes,
)
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    AUTHORITY_MISCLIBS_SHA256,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_moist_stage_bridge import (
    AUTHORITY_SOURCE_SHA256 as MOIST_AUTHORITY_SOURCE_SHA256,
    CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
    CudaMoistStageBridgeState,
)


CUDA_GM1_SCALAR_POST_SOUND_THERMODYNAMIC_TOKEN_SCHEMA = (
    "gm1.scalar-post-sound-thermodynamic-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA = (
    "gm1.cuda-tilestream-scalar-post-sound-getdiv/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-theta-divergence-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_ABI = (
    "gm1.cuda-tilestream-scalar-post-sound-getdiv-f32/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_THREADS = 128
AUTHORITY_ANCHORS = (
    "src/solve2.F:394-412",
    "src/sound.F:806-832",
    "src/solve2.F:1277-1307",
    "src/misclibs.F:108-214",
    "src/solve2.F:1313-1343",
)
GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_CLAIM_LIMIT = (
    "one real 300x300x75 RK1 post-sound thermodynamic checkpoint followed "
    "by exact tile-local terrain/stretch_z=1 getdiv and exact-once host "
    "scatter of divergence and t11*thterm. getdivx, scalar advection/update, "
    "open boundaries, TKE, RK2/RK3, pressure tiling, MPI, forecast, and the "
    "five-metre meteorological product remain refused"
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


def _require_velocity_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1PostSoundVelocityToken):
        raise TypeError("velocity_token must be the exact post-sound token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-sound averaged-velocity token changed")
    _lower_sha("stage_identity_sha256", receipt.get("stage_identity_sha256"))
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1ScalarPostSoundThermodynamicToken:
    receipt: Mapping[str, object]
    thterm: Any


@dataclass(frozen=True, slots=True)
class CudaGm1TiledScalarPostSoundGetdivLaunch:
    ready_event: Any
    compiled: CompiledGm1ChildStretchedMoist
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarThetaDivergenceToken:
    receipt: Mapping[str, object]


def checkpoint_cuda_gm1_scalar_post_sound_thermodynamics(
    velocity_token: Gm1PostSoundVelocityToken,
    state: CudaMoistStageBridgeState,
) -> Gm1ScalarPostSoundThermodynamicToken:
    """Synchronize the admitted RK1 acoustic checkpoint and retain ``thterm``.

    The caller must have produced eqtset=2 terms and applied buoyancy before
    launching ``sound`` on the same stream.  This checkpoint is deliberately
    explicit because tile streaming crosses a device/host scheduling boundary.
    """

    velocity = _require_velocity_token(velocity_token)
    if not isinstance(state, CudaMoistStageBridgeState):
        raise TypeError("state must be CudaMoistStageBridgeState")
    if not state.terms_ready or not state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError(
            "thermodynamic checkpoint requires fresh terms/buoyancy before getdiv"
        )
    if int(state.stream.ptr) != velocity.get("stream_pointer"):
        raise ValueError("thermodynamic state left the post-sound CUDA stream")
    if (state.ni, state.nj, state.nk) != (300, 300, 75) or not state.base3d:
        raise ValueError("admitted thermodynamic checkpoint requires real 300x300x75 base3d")
    if state.compiled.target_arch is None:
        raise ValueError("thermodynamic CUDA image has no target architecture")
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    value = np.ascontiguousarray(cupy.asnumpy(state.thterm), dtype="<f4")
    if value.shape != (75, 300, 300) or not bool(np.all(np.isfinite(value))):
        raise ValueError("post-sound thterm checkpoint is invalid")
    value.flags.writeable = False
    raw_sha = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SCALAR_POST_SOUND_THERMODYNAMIC_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_THERMODYNAMIC_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "stream_pointer": int(state.stream.ptr),
        "thterm_shape": [75, 300, 300],
        "thterm_sha256": raw_sha,
        "host_checkpoint_transport": True,
        "eqtset2_terms_ready": True,
        "buoyancy_applied_before_sound": True,
        "post_sound_getdiv_performed": False,
        "source": {
            "thermodynamics_sha256": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
            "solve2_sha256": MOIST_AUTHORITY_SOURCE_SHA256["src/solve2.F"],
            "anchors": list(AUTHORITY_ANCHORS[:3]),
        },
        "ready_for": "typed tile-local scalar post-sound getdiv only",
        "claim_limit": GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1ScalarPostSoundThermodynamicToken(
        MappingProxyType(receipt), value
    )


def _require_thermodynamic_token(
    token: object,
) -> tuple[Mapping[str, object], Any]:
    if not isinstance(token, Gm1ScalarPostSoundThermodynamicToken):
        raise TypeError("thermodynamic_token must be the exact typed snapshot")
    receipt = token.receipt
    np = importlib.import_module("numpy")
    value = token.thterm
    if (
        receipt.get("schema")
        != CUDA_GM1_SCALAR_POST_SOUND_THERMODYNAMIC_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_THERMODYNAMIC_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
        or not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f4")
        or value.shape != (75, 300, 300)
        or not value.flags.c_contiguous
        or value.flags.writeable
        or hashlib.sha256(value.tobytes(order="C")).hexdigest()
        != receipt.get("thterm_sha256")
    ):
        raise ValueError("post-sound thermodynamic token or payload changed")
    return receipt, value


def tile_scalar_post_sound_getdiv_host_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Return the exact host inventory for one scalar ``getdiv`` tile."""

    shapes = dict(
        tile_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz)
    )
    cnx, cny = tile_nx + 6, tile_ny + 6
    mass = (nz, cny, cnx)
    shapes.update(
        {
            "thterm": mass,
            "divergence": mass,
            "theta_divergence_term": mass,
        }
    )
    return MappingProxyType(shapes)


def fill_gm1_tiled_scalar_post_sound_getdiv_source(
    thermodynamic_token: Gm1ScalarPostSoundThermodynamicToken,
    binding: Gm1TileLocalSourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    """Stage one exact averaged-velocity/thterm tile without global scratch."""

    thermodynamics, thterm = _require_thermodynamic_token(thermodynamic_token)
    if not isinstance(binding, Gm1TileLocalSourceBinding):
        raise TypeError("binding must be Gm1TileLocalSourceBinding")
    if binding.scalar != "theta":
        raise ValueError("scalar post-sound getdiv requires the theta source binding")
    if (
        binding.source_identity_sha256
        != thermodynamics.get("post_sound_velocity_token_sha256")
    ):
        raise ValueError("tile velocities belong to another post-sound snapshot")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    expected = tile_scalar_post_sound_getdiv_host_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=binding.canonical.nz
    )
    if set(target) != set(expected):
        raise ValueError("scalar post-sound getdiv host inventory changed")
    np = importlib.import_module("numpy")
    for name, shape in expected.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"tile {name} must be writable host C-<f4 {shape}")
    base_names = tuple(
        tile_host_buffer_shapes(
            tile_nx=spec.tile_nx,
            tile_ny=spec.tile_ny,
            nz=binding.canonical.nz,
        )
    )
    base = fill_gm1_tile_local_source(
        binding, spec, {name: target[name] for name in base_names}
    )
    y0, x0, cny, cnx = spec.cj0, spec.ci0, spec.cny, spec.cnx
    np.copyto(target["thterm"], thterm[:, y0 : y0 + cny, x0 : x0 + cnx])
    target["divergence"].fill(np.float32(0.0))
    target["theta_divergence_term"].fill(np.float32(0.0))
    tile_sha = hashlib.sha256(target["thterm"].tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        **base,
        "schema": CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_ABI,
        "thermodynamic_token_sha256": thermodynamics["operation_token_sha256"],
        "thterm_tile_sha256": tile_sha,
        "thterm_compute_yx": [y0, y0 + cny, x0, x0 + cnx],
        "outputs_initialized_positive_zero": True,
        "source_order": [
            "eqtset=2 builds thterm before sound",
            "sound returns small-step-averaged u/v/w",
            "tile stages the immutable averaged velocities and thterm",
        ],
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


def _cuda_f32(
    cupy: Any,
    value: Any,
    name: str,
    shape: tuple[int, ...],
    *,
    writable: bool = False,
) -> Any:
    if not isinstance(value, cupy.ndarray) or value.dtype != cupy.float32:
        raise TypeError(f"{name} must be a CUDA float32 array")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


def launch_cuda_gm1_tiled_scalar_post_sound_getdiv(
    thermodynamic_token: Gm1ScalarPostSoundThermodynamicToken,
    arrays: Mapping[str, Any],
    vertical: Gm1ChildStretchedMetrics,
    *,
    spec: Gm1TileSpec,
    fill_receipt: Mapping[str, object],
    dx: float,
    dy: float,
    stream: Any,
    compiled: CompiledGm1ChildStretchedMoist | None = None,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledScalarPostSoundGetdivLaunch:
    """Enqueue only the exact stretched post-sound ``getdiv`` kernel."""

    thermodynamics, _ = _require_thermodynamic_token(thermodynamic_token)
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if not isinstance(fill_receipt, Mapping):
        raise TypeError("fill_receipt must be the exact tile-source receipt")
    if (
        fill_receipt.get("schema")
        != CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA
        or fill_receipt.get("tile") != spec.index
        or fill_receipt.get("thermodynamic_token_sha256")
        != thermodynamics.get("operation_token_sha256")
        or fill_receipt.get("source_identity_sha256")
        != thermodynamics.get("post_sound_velocity_token_sha256")
        or fill_receipt.get("operation_token_sha256")
        != _operation_token(fill_receipt)
    ):
        raise ValueError("scalar post-sound tile source identity changed")
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if allow_uniform_idealized_validation:
        if (
            vertical.identity.get("fixture_class")
            != "source-audited-flat-idealized-uniform-200m"
            or vertical.ztop_f32 != 15000.0
            or vertical.dz_reference_f32 != 200.0
        ):
            raise ValueError(
                "scalar post-sound getdiv requires the exact uniform idealized metrics"
            )
    elif (
        vertical.identity.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256
        or vertical.identity.get("proof_sha256")
        != GM1_CHILD_STRETCHED_PROOF_SHA256
        or vertical.ztop_f32 != 15000.0
    ):
        raise ValueError("scalar post-sound getdiv requires sealed stretched metrics")
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be an explicit CUDA stream")
    if int(stream.ptr) != thermodynamics.get("stream_pointer"):
        raise ValueError("scalar post-sound getdiv left the acoustic CUDA stream")
    for name, spacing in (("dx", dx), ("dy", dy)):
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(spacing)) or float(spacing) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    cupy = importlib.import_module("cupy")
    ni, nj, nk, ng = spec.cnx, spec.cny, 75, CM1_OPEN_NGXY
    shapes = {
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni),
        "dzdx": (nj, ni),
        "dzdy": (nj, ni),
        "gz": (nj, ni),
        "thterm": (nk, nj, ni),
        "divergence": (nk, nj, ni),
        "theta_divergence_term": (nk, nj, ni),
    }
    checked = {
        name: _cuda_f32(
            cupy,
            arrays[name],
            name,
            shape,
            writable=name in {"divergence", "theta_divergence_term"},
        )
        for name, shape in shapes.items()
    }
    pointers = [int(value.data.ptr) for value in checked.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("scalar post-sound getdiv inputs and outputs must not alias")
    compiled = compiled or compile_cuda_gm1_child_stretched_moist()
    if compiled.target_arch != vertical.target_arch:
        raise ValueError("getdiv and stretched metrics target different architectures")
    count = nk * nj * ni
    block = (CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_THREADS, 1, 1)
    grid = (
        (
            count
            + CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_THREADS
            - 1
        )
        // CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_THREADS,
        1,
        1,
    )
    u32, f32 = cupy.uint32, cupy.float32
    with stream:
        compiled.kernels[0](
            grid,
            block,
            (
                checked["rru"],
                checked["rrv"],
                checked["rrw"],
                checked["rgzu"],
                checked["rgzv"],
                checked["dzdx"],
                checked["dzdy"],
                checked["gz"],
                vertical.sigma,
                vertical.sigmaf,
                vertical.rds,
                vertical.rdsf,
                checked["thterm"],
                checked["divergence"],
                checked["theta_divergence_term"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(ng),
                f32(dx),
                f32(dy),
                f32(vertical.ztop_f32),
                f32(1.0e-20),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA,
        "status": "ENQUEUED_LIVE_RK1_TILE_SCALAR_POST_SOUND_GETDIV",
        "abi": CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_ABI,
        "nrk": 1,
        "tile": spec.record(),
        "thermodynamic_token_sha256": thermodynamics["operation_token_sha256"],
        "post_sound_velocity_token_sha256": thermodynamics[
            "post_sound_velocity_token_sha256"
        ],
        "stage_identity_sha256": thermodynamics["stage_identity_sha256"],
        "fill_receipt_sha256": fill_receipt["operation_token_sha256"],
        "stream_pointer": int(stream.ptr),
        "horizontal_spacing_f32": [float(f32(dx)), float(f32(dy))],
        "kernel_source_sha256": GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256,
        "source_order": [
            "eqtset=2 thterm retained across sound",
            "sound returns small-step-averaged velocities",
            "misclibs.F getdiv rebuilds divergence and t11*thterm",
        ],
        "same_stream_after_global_acoustic_checkpoint": True,
        "host_sync_performed": False,
        "getdivx_or_scalar_rk_performed": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "misclibs_sha256": AUTHORITY_MISCLIBS_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledScalarPostSoundGetdivLaunch(
        event, compiled, MappingProxyType(receipt)
    )


def finish_cuda_gm1_tiled_scalar_post_sound_getdiv(
    thermodynamic_token: Gm1ScalarPostSoundThermodynamicToken,
    launches: Sequence[CudaGm1TiledScalarPostSoundGetdivLaunch],
    *,
    plan: Gm1TilePlan,
    outputs: Mapping[str, Any],
) -> Gm1TiledScalarThetaDivergenceToken:
    """Bind exact-once host-scattered divergence outputs for scalar RK1."""

    thermodynamics, _ = _require_thermodynamic_token(thermodynamic_token)
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("scalar post-sound getdiv requires the admitted 300 grid")
    if len(launches) != plan.ntiles:
        raise ValueError("scalar post-sound getdiv requires every exact tile")
    operation_tokens: list[str] = []
    streams: set[int] = set()
    uniform_idealized_dispositions: set[bool] = set()
    owned_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledScalarPostSoundGetdivLaunch):
            raise TypeError("completion requires typed scalar getdiv tile launches")
        receipt = launch.receipt
        y0, y1, x0, x1 = spec.owned_rect("mass")
        owned_words += 75 * (y1 - y0) * (x1 - x0)
        if (
            receipt.get("schema")
            != CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA
            or receipt.get("status")
            != "ENQUEUED_LIVE_RK1_TILE_SCALAR_POST_SOUND_GETDIV"
            or receipt.get("nrk") != 1
            or receipt.get("tile") != spec.record()
            or receipt.get("thermodynamic_token_sha256")
            != thermodynamics.get("operation_token_sha256")
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("scalar getdiv tile identity or source order changed")
        operation_tokens.append(str(receipt["operation_token_sha256"]))
        streams.add(int(receipt["stream_pointer"]))
        uniform_idealized_dispositions.add(
            receipt.get("uniform_idealized_validation") is True
        )
    if owned_words != 75 * 300 * 300:
        raise ValueError("scalar getdiv scatter ownership is not exact once")
    if (
        len(streams) != 1
        or next(iter(streams)) != thermodynamics.get("stream_pointer")
    ):
        raise ValueError("thermodynamic checkpoint and scalar getdiv streams differ")
    if len(uniform_idealized_dispositions) != 1:
        raise ValueError("scalar getdiv tiles disagree on idealized disposition")
    np = importlib.import_module("numpy")
    expected_names = ("divergence", "theta_divergence_term")
    if tuple(outputs) != expected_names:
        raise ValueError("scalar getdiv output inventory or order changed")
    hashes: dict[str, str] = {}
    for name in expected_names:
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != (75, 300, 300)
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{name} must be finite host C-<f4 (75,300,300)")
        hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_RK1_SCALAR_THETA_DIVERGENCE",
        "nrk": 1,
        "stage_identity_sha256": thermodynamics["stage_identity_sha256"],
        "post_sound_velocity_token_sha256": thermodynamics[
            "post_sound_velocity_token_sha256"
        ],
        "thermodynamic_token_sha256": thermodynamics["operation_token_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(operation_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "host_scatter_exact_once": True,
        "uniform_idealized_validation": next(
            iter(uniform_idealized_dispositions)
        ),
        "output_shapes": {
            name: [75, 300, 300] for name in expected_names
        },
        "output_sha256": hashes,
        "output_words": 2 * 75 * 300 * 300,
        "source_order": [
            "eqtset=2 builds thterm before sound",
            "global sound returns averaged velocities",
            "tile-local getdiv emits divergence and t11*thterm",
        ],
        "ready_for": (
            "typed source-independent-interior scalar/moist RK1 only after a "
            "distinct matching post-acoustic mass-flux token"
        ),
        "post_acoustic_mass_flux_token_bound": False,
        "scalar_rk1_performed": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "frozen or pre-acoustic velocity substitution",
            "scalar RK without a matching post-acoustic mass-flux token",
            "RK2/RK3 without new same-stage sound and getdiv tokens",
            "open-boundary scalar publication",
            "global psolver tilestream claim",
            "full RK or forecast",
            "five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "misclibs_sha256": AUTHORITY_MISCLIBS_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarThetaDivergenceToken(MappingProxyType(receipt))


def five_meter_scalar_post_sound_getdiv_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    """Return the exact incremental memory and unresolved 5 m dependency."""

    for name, value in (("tile_nx", tile_nx), ("tile_ny", tile_ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cnx, cny, ng = tile_nx + 6, tile_ny + 6, CM1_OPEN_NGXY
    shapes = {
        "rru": (nk + 2, cny + 2 * ng, cnx + 1 + 2 * ng),
        "rrv": (nk + 2, cny + 1 + 2 * ng, cnx + 2 * ng),
        "rrw": (nk + 3, cny + 2 * ng, cnx + 2 * ng),
        "rgzu": (cny, cnx + 1),
        "rgzv": (cny + 1, cnx),
        "dzdx": (cny, cnx),
        "dzdy": (cny, cnx),
        "gz": (cny, cnx),
        "thterm": (nk, cny, cnx),
        "divergence": (nk, cny, cnx),
        "theta_divergence_term": (nk, cny, cnx),
    }
    one_tile_bytes = 4 * sum(math.prod(shape) for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-scalar-post-sound-getdiv-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "one_device_tile_bytes": one_tile_bytes,
        "double_device_tile_bytes": 2 * one_tile_bytes,
        "getdiv_tile_streamable": True,
        "eqtset2_thterm_tile_streamable": True,
        "global_psolver_tile_streamable": False,
        "post_sound_velocity_token_available_at_5m": False,
        "scalar_theta_divergence_token_available_at_5m": False,
        "scalar_rk1_available_at_5m": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "CUDA_GM1_SCALAR_POST_SOUND_THERMODYNAMIC_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_ABI",
    "CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA",
    "CudaGm1TiledScalarPostSoundGetdivLaunch",
    "GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_CLAIM_LIMIT",
    "Gm1ScalarPostSoundThermodynamicToken",
    "Gm1TiledScalarThetaDivergenceToken",
    "checkpoint_cuda_gm1_scalar_post_sound_thermodynamics",
    "fill_gm1_tiled_scalar_post_sound_getdiv_source",
    "finish_cuda_gm1_tiled_scalar_post_sound_getdiv",
    "five_meter_scalar_post_sound_getdiv_inventory",
    "launch_cuda_gm1_tiled_scalar_post_sound_getdiv",
    "tile_scalar_post_sound_getdiv_host_shapes",
)
