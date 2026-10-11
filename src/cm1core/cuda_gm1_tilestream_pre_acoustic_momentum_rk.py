"""Exact live RK1 momentum chronology immediately before the global sound solve.

This module owns the missing CM1 ``solve2`` seam between the admitted full
``solve1`` held tendencies and the independently owned global ``psolver=3``
acoustic continuation.  It preserves the selected source order:

``eqtset2 -> held seed -> terrain mass flux -> bcw(rrw,0) -> divx ->``
``advu/advv/advw -> vertical/terrain buoyancy``.

The finished result deliberately retains the exact live arrays, metrics,
vertical profiles, CUDA stream, moist bridge, and full-held token by object
identity.  A downstream acoustic launcher can therefore consume the typed
result directly instead of accepting a valid token beside substituted arrays.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_child_stretched_momentum import (
    GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA,
    GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
    CompiledMomentumAdvection5Kernels,
    CudaMomentumAdvection5Launch,
    launch_cuda_gm1_child_stretched_momentum5,
)
from .cuda_gm1_solve2_rotation import (
    CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256,
    CudaGm1Solve2RotationBinding,
    make_cuda_gm1_solve2_rotation_post_divergence_hook,
    require_cuda_gm1_solve2_rotation_binding,
)
from .cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch
from .cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS,
    GM1_RK1_PRE_ACOUSTIC_SELECTED_PROFILE,
    GM1_RK1_PRE_ACOUSTIC_SOURCE_ANCHORS,
    audit_gm1_rk1_pre_acoustic_source_contract,
)
from .cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    Gm1TiledSolve1FullHeldToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)
from .cuda_moist_stage_bridge import (
    CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
    CudaMoistStageBridgeState,
    launch_cuda_eqtset2_stage_terms,
    launch_cuda_moist_buoyancy,
)
from .cuda_psolver3_terrain_open import _require_device_arrays, _require_device_metrics
from .cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    CUDA_SOUND3_OPEN_SOURCE_SHA256,
    CUDA_SOUND3_OPEN_THREADS,
)


CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_SCHEMA = (
    "gm1.cuda-tilestream-pre-acoustic-momentum-rk/v1"
)
CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_STATUS = (
    "ENQUEUED_LIVE_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK1"
)
CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_SCHEMA = (
    "gm1.cuda-tilestream-pre-acoustic-momentum-rk-binding/v1"
)
CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_STATUS = (
    "BOUND_LIVE_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK1"
)
RRW_OPEN_BOUNDARY_SCHEMA = "gm1.cuda-bcw-rrw-open-flag0/v1"
GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT = (
    "one live source-ordered real 300x300x75, 50 m, dt=0.5 s RK1 pre-acoustic "
    "momentum pass through terrain mass flux, literal bcw(rrw,0), divx, "
    "fifth-order momentum advection, and moist vertical/terrain buoyancy; "
    "the global acoustic solve, post-sound getdiv/getdivx, scalar/moisture/TKE "
    "RK, completed stage, RK2/RK3, solve3, forecast, and five-metre product "
    "remain outside this token"
)

_OUTPUT_SHAPES = MappingProxyType(
    {
        "uten": (75, 300, 301),
        "vten": (75, 301, 300),
        "wten": (76, 300, 300),
        "rru": (75, 300, 301),
        "rrv": (75, 301, 300),
        "rrw": (76, 300, 300),
        "divx": (75, 300, 300),
    }
)
_HELD_SHAPES = MappingProxyType(
    {
        "uten": (75, 300, 301),
        "vten": (75, 301, 300),
        "wten": (76, 300, 300),
        "thten1": (75, 300, 300),
        "qten": (6, 75, 300, 300),
    }
)
_PHYSICAL_SLICES = MappingProxyType(
    {
        "u3d": (slice(1, 76), slice(3, 303), slice(3, 304)),
        "uten": (slice(1, 76), slice(3, 303), slice(3, 304)),
        "rru": (slice(1, 76), slice(3, 303), slice(3, 304)),
        "v3d": (slice(1, 76), slice(3, 304), slice(3, 303)),
        "vten": (slice(1, 76), slice(3, 304), slice(3, 303)),
        "rrv": (slice(1, 76), slice(3, 304), slice(3, 303)),
        "w3d": (slice(1, 77), slice(3, 303), slice(3, 303)),
        "wten": (slice(1, 77), slice(3, 303), slice(3, 303)),
        "rrw": (slice(1, 77), slice(3, 303), slice(3, 303)),
        "pp3d": (slice(1, 76), slice(1, 301), slice(1, 301)),
    }
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


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def _require_full_held_token(
    token: object, *, stream_pointer: int | None = None
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledSolve1FullHeldToken:
        raise TypeError("full_held_token must be the exact Gm1TiledSolve1FullHeldToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS
        or receipt.get("completed_nrk") != 0
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("full-held token identity or cadence changed")
    _lower_sha("full-held operation token", receipt.get("operation_token_sha256"))
    actual_stream = receipt.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("full-held stream pointer is invalid")
    if stream_pointer is not None and actual_stream != stream_pointer:
        raise ValueError("pre-acoustic stream differs from full-held stream")
    hashes, shapes = receipt.get("output_sha256"), receipt.get("output_shapes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_HELD_SHAPES):
        raise ValueError("full-held output hash inventory changed")
    if not isinstance(shapes, Mapping) or {
        name: tuple(shape) for name, shape in shapes.items()
    } != dict(_HELD_SHAPES):
        raise ValueError("full-held output shape inventory changed")
    for name in _HELD_SHAPES:
        _lower_sha(f"full-held {name}", hashes.get(name))
    return receipt


def _require_solve1_tke_token(
    token: object,
    *,
    full_held_receipt: Mapping[str, object],
    stream_pointer: int,
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledSolve1TkeRhsToken:
        raise TypeError("solve1_tke_token must be the exact Gm1TiledSolve1TkeRhsToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or receipt.get("completed_nrk") != 0
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve1 TKE RHS token identity or cadence changed")
    operation = _lower_sha(
        "solve1 TKE RHS operation token", receipt.get("operation_token_sha256")
    )
    if (
        receipt.get("stream_pointer") != stream_pointer
        or full_held_receipt.get("solve1_tke_operation_token_sha256") != operation
    ):
        raise ValueError("solve1 TKE RHS token differs from full-held lineage or stream")
    if (
        receipt.get("output_tketen_shape") != [76, 300, 300]
        or not isinstance(receipt.get("output_tketen_words"), int)
        or receipt.get("output_tketen_words") != 76 * 300 * 300
    ):
        raise ValueError("solve1 TKE RHS output inventory changed")
    _lower_sha("solve1 TKE RHS tketen", receipt.get("output_tketen_sha256"))
    return receipt


def _host_f32(
    np: Any, value: Any, *, name: str, shape: tuple[int, ...]
) -> Any:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("float32")
        or tuple(value.shape) != shape
        or not value.flags.c_contiguous
        or not bool(np.all(np.isfinite(value)))
    ):
        raise ValueError(f"{name} must be finite host C-f32 {shape}")
    return value


def _device_f32(
    cupy: Any, value: Any, *, name: str, shape: tuple[int, ...]
) -> Any:
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != cupy.dtype("float32")
        or tuple(value.shape) != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be CUDA C-f32 {shape}")
    return value


def _raw_sha_host(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _physical_host(cupy: Any, value: Any, slices: tuple[slice, ...]) -> Any:
    np = importlib.import_module("numpy")
    host = np.ascontiguousarray(cupy.asnumpy(value[slices]), dtype=np.float32)
    if not bool(np.all(np.isfinite(host))):
        raise ValueError("live pre-acoustic carrier contains non-finite words")
    return host


def _pointer_sha256(values: Mapping[str, Any]) -> str:
    pointers = {name: int(value.data.ptr) for name, value in values.items()}
    return hashlib.sha256(_canonical(pointers)).hexdigest()


@dataclass(frozen=True, slots=True)
class Gm1TiledPreAcousticMomentumRkToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk1AcousticContinuationToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledPreAcousticMomentumRkBinding:
    full_held_token: Gm1TiledSolve1FullHeldToken
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    terms_ready_event: Any
    stage_identity_sha256: str
    receipt: Mapping[str, object]
    allow_uniform_idealized_validation: bool = False
    rotation_binding: CudaGm1Solve2RotationBinding | None = None


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPreAcousticMomentumRkLaunch:
    ready_event: Any
    binding: Gm1TiledPreAcousticMomentumRkBinding
    upstream: CudaMomentumAdvection5Launch
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FinishedGm1TiledPreAcousticMomentumRk:
    token: Gm1TiledPreAcousticMomentumRkToken
    binding: Gm1TiledPreAcousticMomentumRkBinding
    full_held_token: Gm1TiledSolve1FullHeldToken
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    output_sha256: Mapping[str, str]
    upstream: CudaMomentumAdvection5Launch


def prepare_cuda_gm1_tilestream_pre_acoustic_momentum_rk(
    full_held_token: Gm1TiledSolve1FullHeldToken,
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken,
    held_outputs: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch,
    vertical: Gm1ChildStretchedMetrics,
    bridge: CudaMoistStageBridgeState,
    *,
    th0: Any,
    qv0: Any,
    pi0: Any,
    th3d: Any,
    q3d: Any,
    stream: Any,
    allow_uniform_idealized_validation: bool = False,
    rotation_binding: CudaGm1Solve2RotationBinding | None = None,
) -> Gm1TiledPreAcousticMomentumRkBinding:
    """Validate and seed one exact live RK1 pre-acoustic resident stage."""

    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) <= 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    stream_pointer = int(stream.ptr)
    if rotation_binding is not None:
        rotation_binding = require_cuda_gm1_solve2_rotation_binding(
            rotation_binding, nrk=1, stream=stream
        )
    held = _require_full_held_token(full_held_token, stream_pointer=stream_pointer)
    if rotation_binding is not None and "lspgrad_binding_operation_token_sha256" in held:
        raise ValueError(
            "active solve2 rotation does not admit simultaneous live lspgrad forcing"
        )
    solve1_tke = _require_solve1_tke_token(
        solve1_tke_token,
        full_held_receipt=held,
        stream_pointer=stream_pointer,
    )
    if not isinstance(terrain_metrics, CudaGm1ChildTerrainMetricsLaunch):
        raise TypeError("terrain_metrics must be the sealed child terrain launch")
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be the sealed stretched child profiles")
    if not isinstance(bridge, CudaMoistStageBridgeState):
        raise TypeError("bridge must be the exact moist-stage bridge state")
    if (
        bridge.ni,
        bridge.nj,
        bridge.nk,
        bridge.base3d,
        int(bridge.stream.ptr),
    ) != (300, 300, 75, True, stream_pointer):
        raise ValueError("moist bridge domain, base-state mode, or stream changed")
    if bridge.terms_ready or bridge.buoyancy_applied or bridge.post_acoustic_ready:
        raise ValueError("moist bridge is not at a fresh RK1 boundary")

    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    checked_arrays = _require_device_arrays(cupy, arrays, ni=300, nj=300, nk=75)
    _require_device_metrics(cupy, metrics, ni=300, nj=300, nk=75)
    checked_metrics = MappingProxyType(dict(metrics))
    if "rho0_padded" not in checked_metrics:
        raise ValueError("pre-acoustic metrics require source-valid padded rho0")
    _device_f32(
        cupy,
        checked_metrics["rho0_padded"],
        name="rho0_padded",
        shape=(77, 306, 306),
    )

    held_hashes = held["output_sha256"]
    checked_held: dict[str, Any] = {}
    for name, shape in _HELD_SHAPES.items():
        value = _host_f32(np, held_outputs.get(name), name=name, shape=shape)
        if _raw_sha_host(value) != held_hashes[name]:
            raise ValueError(f"live held seed differs from admitted {name} bytes")
        checked_held[name] = value

    base_current = {
        "th0": _device_f32(cupy, th0, name="th0", shape=(75, 300, 300)),
        "qv0": _device_f32(cupy, qv0, name="qv0", shape=(75, 300, 300)),
        "pi0": _device_f32(cupy, pi0, name="pi0", shape=(75, 300, 300)),
        "th3d": _device_f32(cupy, th3d, name="th3d", shape=(75, 300, 300)),
        "q3d": _device_f32(cupy, q3d, name="q3d", shape=(6, 75, 300, 300)),
    }
    input_hashes: dict[str, str] = {}
    for name in ("u3d", "v3d", "w3d", "pp3d"):
        input_hashes[name] = _raw_sha_host(
            _physical_host(cupy, checked_arrays[name], _PHYSICAL_SLICES[name])
        )
    for name, value in base_current.items():
        input_hashes[name] = _raw_sha_host(
            np.ascontiguousarray(cupy.asnumpy(value), dtype=np.float32)
        )
    stage_identity_record = {
        "schema": "gm1.rk-stage-input-identity/v1",
        "nrk": 1,
        "selected_profile": dict(GM1_RK1_PRE_ACOUSTIC_SELECTED_PROFILE),
        "solve1_full_held_token_sha256": held["operation_token_sha256"],
        "input_sha256": input_hashes,
        "terrain_metrics_receipt_sha256": hashlib.sha256(
            _canonical(terrain_metrics.receipt)
        ).hexdigest(),
        "vertical_identity": dict(vertical.identity),
    }
    stage_identity_sha256 = hashlib.sha256(
        _canonical(stage_identity_record)
    ).hexdigest()

    terms_event = launch_cuda_eqtset2_stage_terms(
        bridge,
        checked_arrays,
        checked_metrics,
        th0=base_current["th0"],
        qv0=base_current["qv0"],
        pi0=base_current["pi0"],
        th3d=base_current["th3d"],
        q3d=base_current["q3d"],
        stream=stream,
    )
    with stream:
        for name in ("uten", "vten", "wten"):
            cupy.copyto(
                checked_arrays[name][_PHYSICAL_SLICES[name]],
                cupy.asarray(checked_held[name]),
            )
        seeded_event = cupy.cuda.Event(disable_timing=True)
        seeded_event.record(stream)

    binding_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_STATUS,
        "nrk": 1,
        "stream_pointer": stream_pointer,
        "stage_identity_sha256": stage_identity_sha256,
        "solve1_full_held_token_sha256": held["operation_token_sha256"],
        "solve1_tke_rhs_token_sha256": solve1_tke["operation_token_sha256"],
        "input_sha256": input_hashes,
        "held_seed_sha256": {
            name: held_hashes[name] for name in ("uten", "vten", "wten")
        },
        "array_pointer_sha256": _pointer_sha256(checked_arrays),
        "metric_pointer_sha256": _pointer_sha256(checked_metrics),
        "eqtset2_terms_enqueued_before_held_seed": True,
        "terms_ready_event_recorded": terms_event is not None,
        "seed_ready_event_recorded": seeded_event is not None,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT,
    }
    if rotation_binding is not None:
        binding_receipt.update(
            {
                "rotation_binding_operation_token_sha256": rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(rotation_binding.receipt["selection"]),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    binding_receipt["binding_sha256"] = hashlib.sha256(
        _canonical(binding_receipt)
    ).hexdigest()
    return Gm1TiledPreAcousticMomentumRkBinding(
        full_held_token,
        solve1_tke_token,
        MappingProxyType(checked_arrays),
        checked_metrics,
        terrain_metrics,
        vertical,
        stream,
        bridge,
        seeded_event,
        stage_identity_sha256,
        MappingProxyType(binding_receipt),
        allow_uniform_idealized_validation,
        rotation_binding,
    )


def _require_binding(
    binding: object,
) -> tuple[Gm1TiledPreAcousticMomentumRkBinding, Mapping[str, object]]:
    if type(binding) is not Gm1TiledPreAcousticMomentumRkBinding:
        raise TypeError("binding must be the exact live pre-acoustic binding")
    receipt = binding.receipt
    record = dict(receipt)
    binding_sha = record.pop("binding_sha256", None)
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_STATUS
        or receipt.get("nrk") != 1
        or binding_sha != hashlib.sha256(_canonical(record)).hexdigest()
    ):
        raise ValueError("pre-acoustic binding identity changed")
    stream_pointer = int(binding.stream.ptr)
    held = _require_full_held_token(
        binding.full_held_token, stream_pointer=stream_pointer
    )
    solve1_tke = _require_solve1_tke_token(
        binding.solve1_tke_token,
        full_held_receipt=held,
        stream_pointer=stream_pointer,
    )
    rotation_active = "rotation_binding_operation_token_sha256" in receipt
    if rotation_active != (binding.rotation_binding is not None):
        raise ValueError("RK1 rotation binding object and receipt differ")
    if binding.rotation_binding is not None:
        rotation = require_cuda_gm1_solve2_rotation_binding(
            binding.rotation_binding, nrk=1, stream=binding.stream
        )
        if (
            receipt.get("rotation_binding_operation_token_sha256")
            != rotation.receipt.get("operation_token_sha256")
            or receipt.get("rotation_selection")
            != rotation.receipt.get("selection")
            or receipt.get("rotation_after_divx_before_momentum_advection")
            is not True
            or receipt.get("active_rotation_cuda_raw_bit_oracle") is not False
        ):
            raise ValueError("RK1 rotation config or token lineage changed")
    if (
        receipt.get("stream_pointer") != stream_pointer
        or receipt.get("stage_identity_sha256") != binding.stage_identity_sha256
        or receipt.get("solve1_full_held_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("solve1_tke_rhs_token_sha256")
        != solve1_tke.get("operation_token_sha256")
        or receipt.get("array_pointer_sha256") != _pointer_sha256(binding.arrays)
        or receipt.get("metric_pointer_sha256") != _pointer_sha256(binding.metrics)
        or int(binding.bridge.stream.ptr) != stream_pointer
        or binding.receipt.get("uniform_idealized_validation")
        is not binding.allow_uniform_idealized_validation
    ):
        raise ValueError("live pre-acoustic arrays, metrics, stage, or stream changed")
    return binding, held


def _rrw_open_boundary_hook(binding: Gm1TiledPreAcousticMomentumRkBinding):
    def fill(
        rrw: Any, *, stream: Any, ni: int, nj: int, nk: int
    ) -> Mapping[str, object]:
        if (ni, nj, nk) != (300, 300, 75) or int(stream.ptr) != int(
            binding.stream.ptr
        ):
            raise ValueError("bcw(rrw,0) domain or stream changed")
        compiled = binding.bridge.open_compiled
        if compiled.target_arch != binding.vertical.target_arch:
            raise ValueError("bcw and stretched momentum CUDA images differ")
        words = int(rrw.size)
        cupy = importlib.import_module("cupy")
        u32 = cupy.uint32
        grid = ((words + CUDA_SOUND3_OPEN_THREADS - 1) // CUDA_SOUND3_OPEN_THREADS, 1, 1)
        block = (CUDA_SOUND3_OPEN_THREADS, 1, 1)
        with stream:
            compiled.kernels[5](
                grid,
                block,
                (
                    rrw,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(CM1_OPEN_NGXY),
                    u32(3),
                ),
                stream=stream,
            )
            compiled.kernels[6](
                grid,
                block,
                (
                    rrw,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(CM1_OPEN_NGXY),
                    u32(3),
                ),
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


def launch_cuda_gm1_tilestream_pre_acoustic_momentum_rk(
    binding: Gm1TiledPreAcousticMomentumRkBinding,
    *,
    compiled: CompiledMomentumAdvection5Kernels | None = None,
) -> CudaGm1TiledPreAcousticMomentumRkLaunch:
    """Enqueue terrain momentum and moist buoyancy on the bound live stream."""

    binding, held = _require_binding(binding)
    stream = binding.stream
    with stream:
        stream.wait_event(binding.terms_ready_event)
        momentum = launch_cuda_gm1_child_stretched_momentum5(
            binding.arrays,
            binding.metrics,
            binding.vertical,
            ni=300,
            nj=300,
            nk=75,
            nrk=1,
            dx=50.0,
            dy=50.0,
            divx=binding.bridge.divx,
            stream=stream,
            compiled=compiled,
            rrw_boundary_hook=_rrw_open_boundary_hook(binding),
            post_divergence_hook=(
                make_cuda_gm1_solve2_rotation_post_divergence_hook(
                    binding.rotation_binding
                )
                if binding.rotation_binding is not None
                else None
            ),
            allow_compact_validation=binding.allow_uniform_idealized_validation,
            allow_uniform_idealized_validation=(
                binding.allow_uniform_idealized_validation
            ),
        )
        boundary = momentum.receipt.get("rrw_open_boundary")
        if (
            momentum.receipt.get("schema") != GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA
            or not isinstance(boundary, Mapping)
            or boundary.get("schema") != RRW_OPEN_BOUNDARY_SCHEMA
            or boundary.get("performed") is not True
            or boundary.get("flag") != 0
        ):
            raise ValueError("stretched momentum did not perform exact bcw(rrw,0)")
        buoyancy_event = launch_cuda_moist_buoyancy(
            binding.bridge, binding.arrays, binding.metrics, stream=stream
        )
        event = importlib.import_module("cupy").cuda.Event(disable_timing=True)
        event.record(stream)
    rotation_receipt = momentum.receipt.get("post_divergence_hook")
    if binding.rotation_binding is not None:
        if (
            binding.rotation_binding.compiled.target_arch
            != momentum.compiled.target_arch
            or
            not isinstance(rotation_receipt, Mapping)
            or rotation_receipt.get("binding_operation_token_sha256")
            != binding.rotation_binding.receipt.get("operation_token_sha256")
            or rotation_receipt.get("nrk") != 1
            or rotation_receipt.get("same_stream_after_divx_before_advection")
            is not True
        ):
            raise ValueError("RK1 rotation hook identity or source order changed")
    elif rotation_receipt is not None:
        raise ValueError("disabled RK1 rotation emitted a hook receipt")
    source_audit = audit_gm1_rk1_pre_acoustic_source_contract(
        binding.rotation_binding.receipt["selection"]
        if binding.rotation_binding is not None
        else None
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_STATUS,
        "nrk": 1,
        "stream_pointer": int(stream.ptr),
        "stage_identity_sha256": binding.stage_identity_sha256,
        "solve1_full_held_token_sha256": held["operation_token_sha256"],
        "binding_sha256": binding.receipt["binding_sha256"],
        "source_contract_sha256": source_audit["source_contract_sha256"],
        "operator_receipt_sha256": {
            "stretched_momentum": hashlib.sha256(
                _canonical(momentum.receipt)
            ).hexdigest(),
            "moist_buoyancy_source": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
        },
        "rrw_open_boundary": dict(boundary),
        "eqtset2_terms_before_held_seed": True,
        "momentum_after_held_seed": True,
        "buoyancy_after_momentum": True,
        "host_sync_performed": False,
        "uniform_idealized_validation": (
            binding.allow_uniform_idealized_validation
        ),
        "ready_event_recorded": event is not None and buoyancy_event is not None,
        "claim_limit": GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT,
    }
    if binding.rotation_binding is not None:
        receipt.update(
            {
                "rotation_binding_operation_token_sha256": binding.rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_hook_operation_token_sha256": rotation_receipt[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(
                    binding.rotation_binding.receipt["selection"]
                ),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledPreAcousticMomentumRkLaunch(
        event, binding, momentum, MappingProxyType(receipt)
    )


def finish_cuda_gm1_tilestream_pre_acoustic_momentum_rk(
    launch: CudaGm1TiledPreAcousticMomentumRkLaunch,
) -> FinishedGm1TiledPreAcousticMomentumRk:
    """Synchronize once, hash exact physical outputs, and retain live lineage."""

    if type(launch) is not CudaGm1TiledPreAcousticMomentumRkLaunch:
        raise TypeError("launch must be the exact pre-acoustic momentum launch")
    binding, held = _require_binding(launch.binding)
    pending = launch.receipt
    if (
        pending.get("schema") != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_SCHEMA
        or pending.get("status")
        != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_STATUS
        or pending.get("nrk") != 1
        or pending.get("operation_token_sha256") != _operation_token(pending)
        or pending.get("stage_identity_sha256") != binding.stage_identity_sha256
        or pending.get("stream_pointer") != int(binding.stream.ptr)
    ):
        raise ValueError("pending pre-acoustic momentum identity changed")
    rotation_active = binding.rotation_binding is not None
    if rotation_active:
        if (
            pending.get("rotation_binding_operation_token_sha256")
            != binding.rotation_binding.receipt.get("operation_token_sha256")
            or pending.get("rotation_selection")
            != binding.rotation_binding.receipt.get("selection")
            or pending.get("rotation_after_divx_before_momentum_advection")
            is not True
        ):
            raise ValueError("pending RK1 rotation lineage changed")
    elif "rotation_binding_operation_token_sha256" in pending:
        raise ValueError("disabled RK1 rotation pending receipt changed")
    launch.ready_event.synchronize()
    cupy = importlib.import_module("cupy")
    output_hashes: dict[str, str] = {}
    for name in GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS:
        value = (
            _physical_host(cupy, binding.arrays[name], _PHYSICAL_SLICES[name])
            if name != "divx"
            else _physical_host(
                cupy, launch.upstream.divx, (slice(None), slice(None), slice(None))
            )
        )
        if tuple(value.shape) != _OUTPUT_SHAPES[name]:
            raise ValueError(f"pre-acoustic {name} physical shape changed")
        output_hashes[name] = _raw_sha_host(value)

    source_audit = audit_gm1_rk1_pre_acoustic_source_contract(
        binding.rotation_binding.receipt["selection"]
        if rotation_active
        else None
    )
    operator_source_sha256 = {
        "stretched_momentum": GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
        "moist_stage_bridge": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
        "rrw_open_boundary": CUDA_SOUND3_OPEN_SOURCE_SHA256,
    }
    if rotation_active:
        operator_source_sha256["rotation_forcing"] = (
            CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
        "nrk": 1,
        "stream_pointer": int(binding.stream.ptr),
        "stage_identity_sha256": binding.stage_identity_sha256,
        "solve1_full_held_token_sha256": held["operation_token_sha256"],
        "solve1_tke_rhs_token_sha256": held["solve1_tke_operation_token_sha256"],
        "binding_sha256": binding.receipt["binding_sha256"],
        "pending_operation_token_sha256": pending["operation_token_sha256"],
        "output_sha256": output_hashes,
        "output_shapes": {
            name: list(_OUTPUT_SHAPES[name])
            for name in GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
        },
        "source_order": list(source_audit["source_order"]),
        "source_anchors": list(GM1_RK1_PRE_ACOUSTIC_SOURCE_ANCHORS),
        "operator_source_sha256": operator_source_sha256,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
        },
        "same_live_arrays_metrics_vertical_stream_retained": True,
        "host_sync_performed": True,
        "completed_rk_stage": False,
        "uniform_idealized_validation": (
            binding.allow_uniform_idealized_validation
        ),
        "claim_limit": GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT,
    }
    if rotation_active:
        rotation_hook = launch.upstream.receipt["post_divergence_hook"]
        receipt.update(
            {
                "rotation_binding_operation_token_sha256": binding.rotation_binding.receipt[
                    "operation_token_sha256"
                ],
                "rotation_hook_operation_token_sha256": rotation_hook[
                    "operation_token_sha256"
                ],
                "rotation_selection": dict(
                    binding.rotation_binding.receipt["selection"]
                ),
                "rotation_after_divx_before_momentum_advection": True,
                "active_rotation_cuda_raw_bit_oracle": False,
            }
        )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = Gm1TiledPreAcousticMomentumRkToken(MappingProxyType(receipt))
    return FinishedGm1TiledPreAcousticMomentumRk(
        token,
        binding,
        binding.full_held_token,
        binding.solve1_tke_token,
        binding.arrays,
        binding.metrics,
        binding.terrain_metrics,
        binding.vertical,
        binding.stream,
        binding.bridge,
        MappingProxyType(output_hashes),
        launch.upstream,
    )


__all__ = (
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_SCHEMA",
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_BINDING_STATUS",
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_SCHEMA",
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_STATUS",
    "CudaGm1TiledPreAcousticMomentumRkLaunch",
    "FinishedGm1TiledPreAcousticMomentumRk",
    "GM1_RK1_PRE_ACOUSTIC_CLAIM_LIMIT",
    "Gm1TiledPreAcousticMomentumRkBinding",
    "Gm1TiledPreAcousticMomentumRkToken",
    "Gm1TiledRk1AcousticContinuationToken",
    "RRW_OPEN_BOUNDARY_SCHEMA",
    "finish_cuda_gm1_tilestream_pre_acoustic_momentum_rk",
    "launch_cuda_gm1_tilestream_pre_acoustic_momentum_rk",
    "prepare_cuda_gm1_tilestream_pre_acoustic_momentum_rk",
)
