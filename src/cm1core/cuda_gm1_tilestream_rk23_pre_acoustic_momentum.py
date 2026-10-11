"""Runnable fresh RK2/RK3 pre-acoustic momentum continuation.

This module starts only from an exact completed prior RK stage and reuses the
same resident 300x300x75 child carriers.  It replays the source-required
current-velocity terrain boundary for RK2/RK3, restores the immutable solve1
held momentum tendencies, rebuilds terrain mass flux/divergence, and adds
fifth-order momentum advection plus moist buoyancy.  It stops before the
global acoustic solve and emits the exact pre-acoustic result consumed by
``cuda_gm1_tilestream_rk23_acoustic``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_child_stretched_momentum import (
    GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA,
    GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
    CompiledMomentumAdvection5Kernels,
    CudaMomentumAdvection5Launch,
    launch_cuda_gm1_child_stretched_momentum5,
)
from .cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch
from .cuda_gm1_solve2_rotation import (
    CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256,
    CudaGm1Solve2RotationBinding,
    make_cuda_gm1_solve2_rotation_post_divergence_hook,
    require_cuda_gm1_solve2_rotation_binding,
)
from .cuda_gm1_tilestream_completed_rk_stage import (
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_gm1_tilestream_pre_acoustic_momentum_rk import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    Gm1TiledPreAcousticMomentumRkToken,
    _HELD_SHAPES,
    _OUTPUT_SHAPES,
    _PHYSICAL_SLICES,
    _host_f32,
    _physical_host,
    _pointer_sha256,
    _raw_sha_host,
    _require_full_held_token,
)
from .cuda_gm1_tilestream_rk23_acoustic import (
    CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA,
    CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS,
    CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS,
    FinishedGm1TiledRk23Acoustic,
    FinishedGm1TiledRk23PreAcousticMomentum,
    GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS,
    Gm1TiledRk23PreAcousticMomentumBinding,
    Gm1TiledRk23PreAcousticMomentumToken,
    audit_gm1_rk23_live_acoustic_source_contract,
    finish_cuda_gm1_tilestream_rk23_acoustic,
    launch_cuda_gm1_tilestream_rk23_acoustic,
)
from .cuda_moist_stage_bridge import (
    CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
    CudaMoistStageBridgeState,
    begin_cuda_moist_stage,
    launch_cuda_eqtset2_stage_terms,
    launch_cuda_moist_buoyancy,
)
from .cuda_momentum_rk import (
    CUDA_MOMENTUM_RK_SOURCE_SHA256,
    CUDA_MOMENTUM_RK_THREADS,
    CompiledMomentumRKKernels,
    compile_cuda_momentum_rk,
)
from .cuda_psolver3_terrain_open import _require_device_arrays, _require_device_metrics
from .cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    CUDA_SOUND3_OPEN_SOURCE_SHA256,
    CUDA_SOUND3_OPEN_THREADS,
)
from .cuda_terrain_open_boundary import (
    CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
    CompiledTerrainOpenBoundaryKernels,
    compile_cuda_terrain_open_boundary,
)


CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_EXECUTION_SCHEMA = (
    "gm1.cuda-tilestream-rk23-pre-acoustic-momentum/v1"
)
CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_PENDING_STATUS = (
    "ENQUEUED_LIVE_300X300X75_RK23_PRE_ACOUSTIC_MOMENTUM"
)
RRW_OPEN_BOUNDARY_SCHEMA = "gm1.cuda-bcw-rrw-open-flag0/v1"
GM1_RK23_PRE_ACOUSTIC_CLAIM_LIMIT = (
    "one live source-ordered real 300x300x75, 50 m RK2 or RK3 pre-acoustic "
    "momentum continuation after an exact completed prior stage, through "
    "current-velocity terrain boundary replay, immutable solve1 held seed, "
    "terrain mass flux, bcw(rrw,0), divx, fifth-order momentum advection, "
    "and moist buoyancy. Acoustic, post-sound/scalar/TKE continuation, a "
    "completed current stage, solve3, forecast, and five-metre claims remain outside"
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


def _requested_nrk(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (2, 3):
        raise ValueError("requested_nrk must be exactly 2 or 3")
    return value


def _require_prior_pre_component(
    token: object,
    *,
    requested_nrk: int,
    completed: Mapping[str, object],
    full_held: Mapping[str, object],
) -> Mapping[str, object]:
    components = completed["component_operation_token_sha256"]
    expected_operation = components.get("pre_acoustic_momentum_rk")
    _lower_sha("prior pre-acoustic component", expected_operation)
    if requested_nrk == 2:
        if type(token) is not Gm1TiledPreAcousticMomentumRkToken:
            raise TypeError("RK2 requires exact RK1 pre-acoustic component token")
        receipt = token.receipt
        expected_schema = CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA
        expected_status = CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS
        actual_nrk = receipt.get("nrk")
    else:
        if type(token) is not Gm1TiledRk23PreAcousticMomentumToken:
            raise TypeError("RK3 requires exact RK2 pre-acoustic component token")
        receipt = token.receipt
        expected_schema = CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA
        expected_status = CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS
        actual_nrk = receipt.get("requested_nrk")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != expected_schema
        or receipt.get("status") != expected_status
        or actual_nrk != requested_nrk - 1
        or receipt.get("solve1_full_held_token_sha256")
        != full_held.get("operation_token_sha256")
        or receipt.get("operation_token_sha256") != expected_operation
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
        or receipt.get("completed_rk_stage") is not False
    ):
        raise ValueError("prior pre-acoustic component or held lineage changed")
    return MappingProxyType(dict(receipt))


def _rrw_open_boundary_hook(*, stream: Any, bridge: Any, vertical: Any):
    def fill(rrw: Any, *, stream: Any, ni: int, nj: int, nk: int):
        if (ni, nj, nk) != (300, 300, 75) or int(stream.ptr) <= 0:
            raise ValueError("bcw(rrw,0) domain or stream changed")
        if int(stream.ptr) != int(bridge.stream.ptr):
            raise ValueError("bcw(rrw,0) left the live RK stream")
        compiled = bridge.open_compiled
        if compiled.target_arch != vertical.target_arch:
            raise ValueError("bcw and stretched momentum CUDA images differ")
        cupy = importlib.import_module("cupy")
        words = int(rrw.size)
        grid = (
            (words + CUDA_SOUND3_OPEN_THREADS - 1) // CUDA_SOUND3_OPEN_THREADS,
            1,
            1,
        )
        block = (CUDA_SOUND3_OPEN_THREADS, 1, 1)
        u32 = cupy.uint32
        with stream:
            for kernel in compiled.kernels[5:7]:
                kernel(
                    grid,
                    block,
                    (rrw, u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)),
                    stream=stream,
                )
        return MappingProxyType(
            {
                "schema": RRW_OPEN_BOUNDARY_SCHEMA,
                "performed": True,
                "flag": 0,
                "directions": ["west", "east", "south", "north"],
                "vertical_rigid_faces_written": False,
                "stream_pointer": int(stream.ptr),
                "source_sha256": CUDA_SOUND3_OPEN_SOURCE_SHA256,
                "authority": "src/bc.F:942-1143; src/solve2.F:575",
            }
        )

    return fill


@dataclass(frozen=True, slots=True)
class CudaGm1TiledRk23PreAcousticMomentumLaunch:
    ready_event: Any
    binding: Gm1TiledRk23PreAcousticMomentumBinding
    requested_nrk: int
    prior_pre_acoustic_token: object
    full_held_token: object
    upstream: CudaMomentumAdvection5Launch
    receipt: Mapping[str, object]
    rotation_binding: CudaGm1Solve2RotationBinding | None = None


def launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    prior_pre_acoustic_token: object,
    full_held_token: object,
    held_outputs: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch,
    vertical: Gm1ChildStretchedMetrics,
    bridge: CudaMoistStageBridgeState,
    *,
    requested_nrk: int,
    th0: Any,
    qv0: Any,
    pi0: Any,
    th3d: Any,
    q3d: Any,
    stream: Any,
    momentum_rk_compiled: CompiledMomentumRKKernels | None = None,
    terrain_boundary_compiled: CompiledTerrainOpenBoundaryKernels | None = None,
    momentum_compiled: CompiledMomentumAdvection5Kernels | None = None,
    allow_uniform_idealized_validation: bool = False,
    rotation_binding: CudaGm1Solve2RotationBinding | None = None,
) -> CudaGm1TiledRk23PreAcousticMomentumLaunch:
    """Enqueue one exact fresh RK2/RK3 pre-sound continuation."""

    requested_nrk = _requested_nrk(requested_nrk)
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if type(prior_completed_stage_token) is not Gm1TiledCompletedRkStageToken:
        raise TypeError("prior token must be exact Gm1TiledCompletedRkStageToken")
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) <= 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    stream_pointer = int(stream.ptr)
    if rotation_binding is not None:
        rotation_binding = require_cuda_gm1_solve2_rotation_binding(
            rotation_binding, nrk=requested_nrk, stream=stream
        )
    completed = require_gm1_tiled_completed_rk_stage_token(
        prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
        stream_pointer=stream_pointer,
    )
    full_held = _require_full_held_token(full_held_token, stream_pointer=stream_pointer)
    if rotation_binding is not None and "lspgrad_binding_operation_token_sha256" in full_held:
        raise ValueError(
            "active solve2 rotation does not admit simultaneous live lspgrad forcing"
        )
    prior_pre = _require_prior_pre_component(
        prior_pre_acoustic_token,
        requested_nrk=requested_nrk,
        completed=completed,
        full_held=full_held,
    )
    if (
        prior_pre.get("uniform_idealized_validation") is True
    ) != allow_uniform_idealized_validation:
        raise ValueError(
            "RK23 uniform idealized admission differs from prior pre-acoustic lineage"
        )
    if type(terrain_metrics) is not CudaGm1ChildTerrainMetricsLaunch:
        raise TypeError("terrain_metrics must be exact child terrain launch")
    if type(vertical) is not Gm1ChildStretchedMetrics:
        raise TypeError("vertical must be exact stretched child profiles")
    if type(bridge) is not CudaMoistStageBridgeState:
        raise TypeError("bridge must be exact moist-stage bridge state")
    if (
        (bridge.ni, bridge.nj, bridge.nk, bridge.base3d, int(bridge.stream.ptr))
        != (300, 300, 75, True, stream_pointer)
        or not bridge.terms_ready
        or not bridge.buoyancy_applied
        or not bridge.post_acoustic_ready
    ):
        raise ValueError("prior stage moist bridge is not complete on this stream")

    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    checked_arrays = _require_device_arrays(cupy, arrays, ni=300, nj=300, nk=75)
    _require_device_metrics(cupy, metrics, ni=300, nj=300, nk=75)
    checked_metrics = MappingProxyType(dict(metrics))
    if "rho0_padded" not in checked_metrics:
        raise ValueError("RK23 pre-acoustic metrics require padded rho0")
    base_current = {
        "th0": th0,
        "qv0": qv0,
        "pi0": pi0,
        "th3d": th3d,
        "q3d": q3d,
    }
    expected_shapes = {
        "th0": (75, 300, 300),
        "qv0": (75, 300, 300),
        "pi0": (75, 300, 300),
        "th3d": (75, 300, 300),
        "q3d": (6, 75, 300, 300),
    }
    for name, value in base_current.items():
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or tuple(value.shape) != expected_shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {expected_shapes[name]}")
    checked_held: dict[str, Any] = {}
    held_hashes = full_held["output_sha256"]
    for name, shape in _HELD_SHAPES.items():
        value = _host_f32(np, held_outputs.get(name), name=name, shape=shape)
        if _raw_sha_host(value) != held_hashes[name]:
            raise ValueError(f"live held seed differs from admitted {name} bytes")
        checked_held[name] = value

    momentum_rk_compiled = momentum_rk_compiled or compile_cuda_momentum_rk()
    terrain_boundary_compiled = (
        terrain_boundary_compiled or compile_cuda_terrain_open_boundary()
    )
    arches = {
        momentum_rk_compiled.target_arch,
        terrain_boundary_compiled.target_arch,
        bridge.compiled.target_arch,
        bridge.open_compiled.target_arch,
        vertical.target_arch,
    }
    if len(arches) != 1:
        raise ValueError("RK23 pre-acoustic CUDA images target different architectures")
    source = audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)
    stage_identity = hashlib.sha256(
        _canonical(
            {
                "schema": "gm1.rk23-stage-input-identity/v1",
                "requested_nrk": requested_nrk,
                "prior_completed_rk_stage_token_sha256": completed[
                    "operation_token_sha256"
                ],
                "prior_pre_acoustic_component_token_sha256": prior_pre[
                    "operation_token_sha256"
                ],
                "solve1_full_held_token_sha256": full_held["operation_token_sha256"],
                "source_contract_sha256": source["source_contract_sha256"],
                "array_pointer_sha256": _pointer_sha256(checked_arrays),
                "metric_pointer_sha256": _pointer_sha256(checked_metrics),
                "terrain_metrics_receipt_sha256": hashlib.sha256(
                    _canonical(dict(terrain_metrics.receipt))
                ).hexdigest(),
                "vertical_identity": dict(vertical.identity),
            }
        )
    ).hexdigest()

    begin_cuda_moist_stage(bridge)
    terms_event = launch_cuda_eqtset2_stage_terms(
        bridge,
        checked_arrays,
        checked_metrics,
        th0=th0,
        qv0=qv0,
        pi0=pi0,
        th3d=th3d,
        q3d=q3d,
        stream=stream,
    )
    _, bc2d = momentum_rk_compiled.kernels
    bc_words = max((300 + 2 * CM1_OPEN_NGXY) * CM1_OPEN_NGXY, 300 * 300)
    block = (CUDA_MOMENTUM_RK_THREADS, 1, 1)
    bc_grid = ((bc_words + block[0] - 1) // block[0], 1, 1)
    u32 = cupy.uint32
    # The admitted solve1 token exposes the exact source-active physical held
    # carriers, not a padded duplicate.  Restore precisely those admitted
    # regions; do not imply that unavailable common-rectangle halo bytes were
    # independently carried by the token.
    with stream:
        for name in ("uten", "vten", "wten"):
            cupy.copyto(
                checked_arrays[name][_PHYSICAL_SLICES[name]],
                cupy.asarray(checked_held[name]),
            )
        terrain_boundary_compiled.kernels[2](
            bc_grid,
            block,
            (
                checked_arrays["u3d"],
                checked_arrays["v3d"],
                checked_metrics["gz"],
                checked_metrics["dzdx"],
                checked_metrics["dzdy"],
                checked_arrays["w3d"],
                u32(300),
                u32(300),
                u32(75),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        for direction in range(4):
            bc2d(
                bc_grid,
                block,
                (
                    checked_arrays["w3d"],
                    u32(300),
                    u32(300),
                    u32(CM1_OPEN_NGXY),
                    u32(direction),
                ),
                stream=stream,
            )
        momentum = launch_cuda_gm1_child_stretched_momentum5(
            checked_arrays,
            checked_metrics,
            vertical,
            ni=300,
            nj=300,
            nk=75,
            nrk=requested_nrk,
            dx=50.0,
            dy=50.0,
            divx=bridge.divx,
            stream=stream,
            compiled=momentum_compiled,
            rrw_boundary_hook=_rrw_open_boundary_hook(
                stream=stream, bridge=bridge, vertical=vertical
            ),
            post_divergence_hook=(
                make_cuda_gm1_solve2_rotation_post_divergence_hook(
                    rotation_binding
                )
                if rotation_binding is not None
                else None
            ),
            allow_compact_validation=allow_uniform_idealized_validation,
            allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        )
        rrw_boundary = momentum.receipt.get("rrw_open_boundary")
        if (
            momentum.receipt.get("schema") != GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA
            or not isinstance(rrw_boundary, Mapping)
            or rrw_boundary.get("schema") != RRW_OPEN_BOUNDARY_SCHEMA
            or rrw_boundary.get("performed") is not True
            or rrw_boundary.get("flag") != 0
        ):
            raise ValueError("RK23 momentum did not perform exact bcw(rrw,0)")
        rotation_hook = momentum.receipt.get("post_divergence_hook")
        if rotation_binding is not None:
            if (
                rotation_binding.compiled.target_arch
                != momentum.compiled.target_arch
                or
                not isinstance(rotation_hook, Mapping)
                or rotation_hook.get("binding_operation_token_sha256")
                != rotation_binding.receipt.get("operation_token_sha256")
                or rotation_hook.get("nrk") != requested_nrk
                or rotation_hook.get("same_stream_after_divx_before_advection")
                is not True
            ):
                raise ValueError("RK23 rotation hook identity or source order changed")
        elif rotation_hook is not None:
            raise ValueError("disabled RK23 rotation emitted a hook receipt")
        buoyancy_event = launch_cuda_moist_buoyancy(
            bridge, checked_arrays, checked_metrics, stream=stream
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)

    binding_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": completed["operation_token_sha256"],
        "prior_pre_acoustic_component_token_sha256": prior_pre[
            "operation_token_sha256"
        ],
        "solve1_full_held_token_sha256": full_held["operation_token_sha256"],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "array_pointer_sha256": _pointer_sha256(checked_arrays),
        "metric_pointer_sha256": _pointer_sha256(checked_metrics),
        "moist_bridge_advanced_from_completed_prior_stage": True,
        "eqtset2_terms_before_held_seed": True,
        "current_velocity_boundary_replay": {
            "bcwsfc": True,
            "bc2d_w_bottom_directions": [0, 1, 2, 3],
            "mpi": False,
        },
        "terms_ready_event_recorded": terms_event is not None,
        "buoyancy_ready_event_recorded": buoyancy_event is not None,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK23_PRE_ACOUSTIC_CLAIM_LIMIT,
    }
    if rotation_binding is not None:
        binding_receipt.update(
            {
                "rotation_binding_operation_token_sha256": rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_hook_operation_token_sha256": rotation_hook[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(rotation_binding.receipt["selection"]),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    binding_receipt["operation_token_sha256"] = _operation_token(binding_receipt)
    binding = Gm1TiledRk23PreAcousticMomentumBinding(
        prior_completed_stage_token,
        MappingProxyType(checked_arrays),
        checked_metrics,
        terrain_metrics,
        vertical,
        stream,
        bridge,
        stage_identity,
        MappingProxyType(binding_receipt),
    )
    pending: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_EXECUTION_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_PENDING_STATUS,
        "requested_nrk": requested_nrk,
        "prior_completed_rk_stage_token_sha256": completed["operation_token_sha256"],
        "prior_pre_acoustic_component_token_sha256": prior_pre[
            "operation_token_sha256"
        ],
        "solve1_full_held_token_sha256": full_held["operation_token_sha256"],
        "binding_operation_token_sha256": binding_receipt["operation_token_sha256"],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "source_contract_sha256": source["source_contract_sha256"],
        "current_velocity_boundary_replay": True,
        "fresh_current_stage_mass_flux": True,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "host_sync_performed": False,
        "completed_rk_stage": False,
        "claim_limit": GM1_RK23_PRE_ACOUSTIC_CLAIM_LIMIT,
    }
    if rotation_binding is not None:
        pending.update(
            {
                "rotation_binding_operation_token_sha256": rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_hook_operation_token_sha256": rotation_hook[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(rotation_binding.receipt["selection"]),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    pending["operation_token_sha256"] = _operation_token(pending)
    return CudaGm1TiledRk23PreAcousticMomentumLaunch(
        event,
        binding,
        requested_nrk,
        prior_pre_acoustic_token,
        full_held_token,
        momentum,
        MappingProxyType(pending),
        rotation_binding,
    )


def finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
    launch: CudaGm1TiledRk23PreAcousticMomentumLaunch,
) -> FinishedGm1TiledRk23PreAcousticMomentum:
    """Synchronize once, hash physical outputs, and retain live carriers."""

    if type(launch) is not CudaGm1TiledRk23PreAcousticMomentumLaunch:
        raise TypeError("launch must be exact RK23 pre-acoustic launch")
    requested_nrk = _requested_nrk(launch.requested_nrk)
    binding = launch.binding
    if type(binding) is not Gm1TiledRk23PreAcousticMomentumBinding:
        raise TypeError("launch lost exact RK23 pre-acoustic binding")
    allow_uniform_idealized_validation = (
        binding.receipt.get("uniform_idealized_validation") is True
    )
    pending = launch.receipt
    if (
        pending.get("schema") != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_EXECUTION_SCHEMA
        or pending.get("status") != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_PENDING_STATUS
        or pending.get("requested_nrk") != requested_nrk
        or pending.get("binding_operation_token_sha256")
        != binding.receipt.get("operation_token_sha256")
        or pending.get("stage_identity_sha256") != binding.stage_identity_sha256
        or pending.get("stream_pointer") != int(binding.stream.ptr)
        or pending.get("uniform_idealized_validation")
        is not allow_uniform_idealized_validation
        or pending.get("operation_token_sha256") != _operation_token(pending)
    ):
        raise ValueError("pending RK23 pre-acoustic identity changed")
    rotation_active = launch.rotation_binding is not None
    if rotation_active:
        rotation = require_cuda_gm1_solve2_rotation_binding(
            launch.rotation_binding, nrk=requested_nrk, stream=binding.stream
        )
        if (
            pending.get("rotation_binding_operation_token_sha256")
            != rotation.receipt.get("operation_token_sha256")
            or pending.get("rotation_selection") != rotation.receipt.get("selection")
            or pending.get("rotation_after_divx_before_momentum_advection")
            is not True
        ):
            raise ValueError("pending RK23 rotation lineage changed")
    elif "rotation_binding_operation_token_sha256" in pending:
        raise ValueError("disabled RK23 rotation pending receipt changed")
    completed = require_gm1_tiled_completed_rk_stage_token(
        binding.prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
        stream_pointer=int(binding.stream.ptr),
    )
    full_held = _require_full_held_token(
        launch.full_held_token, stream_pointer=int(binding.stream.ptr)
    )
    prior_pre = _require_prior_pre_component(
        launch.prior_pre_acoustic_token,
        requested_nrk=requested_nrk,
        completed=completed,
        full_held=full_held,
    )
    launch.ready_event.synchronize()
    cupy = importlib.import_module("cupy")
    output_hashes: dict[str, str] = {}
    for name in GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS:
        value = (
            _physical_host(cupy, binding.arrays[name], _PHYSICAL_SLICES[name])
            if name != "divx"
            else _physical_host(
                cupy, launch.upstream.divx, (slice(None), slice(None), slice(None))
            )
        )
        if tuple(value.shape) != _OUTPUT_SHAPES[name]:
            raise ValueError(f"RK23 pre-acoustic {name} physical shape changed")
        output_hashes[name] = _raw_sha_host(value)
    source = audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)
    operator_source_sha256 = {
        "momentum_rk_bc2d": CUDA_MOMENTUM_RK_SOURCE_SHA256,
        "terrain_bcwsfc": CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
        "stretched_momentum": GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
        "moist_stage_bridge": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
        "rrw_open_boundary": CUDA_SOUND3_OPEN_SOURCE_SHA256,
    }
    if rotation_active:
        operator_source_sha256["rotation_forcing"] = (
            CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": completed["operation_token_sha256"],
        "prior_pre_acoustic_component_token_sha256": prior_pre[
            "operation_token_sha256"
        ],
        "solve1_full_held_token_sha256": full_held["operation_token_sha256"],
        "stage_identity_sha256": binding.stage_identity_sha256,
        "stream_pointer": int(binding.stream.ptr),
        "source_contract_sha256": source["source_contract_sha256"],
        "binding_operation_token_sha256": binding.receipt["operation_token_sha256"],
        "pending_operation_token_sha256": pending["operation_token_sha256"],
        "current_velocity_boundary_replay": True,
        "fresh_current_stage_mass_flux": True,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "historical_rk1_post_sound_token_reused": False,
        "output_sha256": output_hashes,
        "output_shapes": {
            name: list(_OUTPUT_SHAPES[name])
            for name in GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS
        },
        "operator_source_sha256": operator_source_sha256,
        "same_live_arrays_metrics_vertical_stream_retained": True,
        "host_sync_performed": True,
        "completed_rk_stage": False,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    if rotation_active:
        rotation_hook = launch.upstream.receipt["post_divergence_hook"]
        receipt.update(
            {
                "rotation_binding_operation_token_sha256": launch.rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_hook_operation_token_sha256": rotation_hook[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(
                    launch.rotation_binding.receipt["selection"]
                ),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = Gm1TiledRk23PreAcousticMomentumToken(MappingProxyType(receipt))
    return FinishedGm1TiledRk23PreAcousticMomentum(
        token,
        binding,
        binding.prior_completed_stage_token,
        requested_nrk,
        binding.arrays,
        binding.metrics,
        binding.terrain_metrics,
        binding.vertical,
        binding.stream,
        binding.bridge,
        MappingProxyType(output_hashes),
    )


def execute_cuda_gm1_tilestream_rk23_acoustic_stage(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    prior_pre_acoustic_token: object,
    full_held_token: object,
    held_outputs: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch,
    vertical: Gm1ChildStretchedMetrics,
    bridge: CudaMoistStageBridgeState,
    *,
    requested_nrk: int,
    th0: Any,
    qv0: Any,
    pi0: Any,
    th3d: Any,
    q3d: Any,
    stream: Any,
    momentum_rk_compiled: CompiledMomentumRKKernels | None = None,
    terrain_boundary_compiled: CompiledTerrainOpenBoundaryKernels | None = None,
    momentum_compiled: CompiledMomentumAdvection5Kernels | None = None,
    acoustic_compiled: Any | None = None,
    acoustic_open_compiled: Any | None = None,
    acoustic_lateral_compiled: Any | None = None,
    allow_uniform_idealized_validation: bool = False,
    rotation_binding: CudaGm1Solve2RotationBinding | None = None,
    continuation: Callable[[Any], Any] | None = None,
) -> FinishedGm1TiledRk23Acoustic:
    """Execute fresh RK2/RK3 pre-sound and sound on the resident stage.

    The explicit synchronizations in both finish calls are the admitted first
    full-stage proof boundary.  No array reload, completed-stage token, or new
    token schema is introduced here; downstream work receives the existing
    exact acoustic live handoff before the caller tears down residency.
    """

    pre_launch = launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(
        prior_completed_stage_token,
        prior_pre_acoustic_token,
        full_held_token,
        held_outputs,
        arrays,
        metrics,
        terrain_metrics,
        vertical,
        bridge,
        requested_nrk=requested_nrk,
        th0=th0,
        qv0=qv0,
        pi0=pi0,
        th3d=th3d,
        q3d=q3d,
        stream=stream,
        momentum_rk_compiled=momentum_rk_compiled,
        terrain_boundary_compiled=terrain_boundary_compiled,
        momentum_compiled=momentum_compiled,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
        rotation_binding=rotation_binding,
    )
    pre_result = finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum(pre_launch)
    acoustic_launch = launch_cuda_gm1_tilestream_rk23_acoustic(
        pre_result,
        compiled=acoustic_compiled,
        open_compiled=acoustic_open_compiled,
        lateral_compiled=acoustic_lateral_compiled,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    return finish_cuda_gm1_tilestream_rk23_acoustic(
        acoustic_launch,
        continuation=continuation,
    )


__all__ = (
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_EXECUTION_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_PENDING_STATUS",
    "CudaGm1TiledRk23PreAcousticMomentumLaunch",
    "GM1_RK23_PRE_ACOUSTIC_CLAIM_LIMIT",
    "execute_cuda_gm1_tilestream_rk23_acoustic_stage",
    "finish_cuda_gm1_tilestream_rk23_pre_acoustic_momentum",
    "launch_cuda_gm1_tilestream_rk23_pre_acoustic_momentum",
)
