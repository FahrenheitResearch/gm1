"""Typed live boundary for source-exact CM1 large-scale nudging.

The independently sealed CPU operator owns the default-REAL schedule, the
late-solve1 u/v/theta/qv increments, and the solve3 u/v base-profile update.
This module authenticates caller-supplied target profiles and averages, stages
the exact increments on one explicit stream, and exposes only the two source
order insertion points.  Omitting the binding remains the historical
``do_lsnudge = .false.`` identity.

There is deliberately no profile-file reader, interpolation, MPI reduction,
diagnostic budget, adaptive coupling, complete trajectory, or CUDA raw-bit
claim here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .large_scale_nudging import (
    CM1_LARGE_SCALE_NUDGING_STATUS,
    LargeScaleNudgingSelection,
    cpu_large_scale_nudging_inplace,
    large_scale_nudging_shapes,
)


CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_SCHEMA = (
    "gm1.cuda-tilestream-large-scale-nudging-binding/v1"
)
CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_STATUS = (
    "PREPARED_LIVE_TILESTREAM_LARGE_SCALE_NUDGING_SOLVE1_SOLVE3"
)
CUDA_GM1_LARGE_SCALE_NUDGING_TILE_SCHEMA = (
    "gm1.cuda-tilestream-large-scale-nudging-solve1-tile/v1"
)
CUDA_GM1_LARGE_SCALE_NUDGING_TILE_STATUS = (
    "ENQUEUED_LIVE_LARGE_SCALE_NUDGING_AFTER_HELD_COMPOSITION"
)
CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_SCHEMA = (
    "gm1.cuda-large-scale-nudging-solve3-base-update/v1"
)
CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_STATUS = (
    "ENQUEUED_LIVE_LARGE_SCALE_NUDGING_BASE_UPDATE_BEFORE_CURRENT_TO_A"
)
CUDA_GM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT = (
    "typed delivery of the independently source-exact CPU large-scale-nudging "
    "schedule into the late solve1 held u/v/theta/qv boundary and the solve3 "
    "u/v base-profile boundary on one explicit stream; supplied authenticated "
    "target profiles/averages/base state only; no external profile I/O or "
    "interpolation, CUDA raw-bit oracle, MPI average, adaptive combination, "
    "complete coupled trajectory, checkpoint persistence of base profiles, "
    "forecast, validation, or product claim"
)
CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_ORDER = (
    "solve1 held Rayleigh/idiss/stress/scalar composition completes",
    "active supplied-profile large-scale nudging adds u/v/theta/qv held terms",
    "solve2 RK/acoustic stages consume the admitted nudged held token",
    "solve3 held-rate recapture completes",
    "active u/v target profiles update ug/vg and horizontally broadcast u0/v0",
    "solve3 momentum current-to-A follows the base-profile update",
)
LARGE_SCALE_NUDGING_CPU_OPERATOR_SOURCE_SHA256 = (
    "f3f3642c830a2fb0fd6115727684a8d6b9917b32563f8bc5d258d93b5ed01e5c"
)

_PROFILE_KEYS = (
    "lsnudge_time1",
    "lsnudge_time2",
    "lsnudge_u",
    "lsnudge_v",
    "lsnudge_th",
    "lsnudge_qv",
)
_AVERAGE_KEYS = ("uavg", "vavg", "thavg", "qavg")
_BASE_KEYS = ("ug", "vg", "u0", "v0")
_SOLVE1_KEYS = ("uten", "vten", "thten1", "qvten")
_SOLVE3_PROFILE_KEYS = ("ug", "vg", "u0_profile", "v0_profile")
_FULL_HELD_TOKEN_SCHEMA = "gm1.tilestream-solve1-full-held-token/v1"
_FULL_HELD_TOKEN_STATUS = "ADMITTED_LIVE_TILESTREAM_SOLVE1_FULL_HELD"


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


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_operator_source() -> None:
    installed = Path(__file__).with_name("large_scale_nudging.py")
    if hashlib.sha256(installed.read_bytes()).hexdigest() != (
        LARGE_SCALE_NUDGING_CPU_OPERATOR_SOURCE_SHA256
    ):
        raise RuntimeError(
            "installed large-scale-nudging CPU operator differs from its source pin"
        )


def _require_exact_mapping(name: str, value: object, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != keys:
        raise ValueError(f"{name} inventory/order changed; expected " + ",".join(keys))
    return value


def _require_companions(
    *,
    lspgrad: int,
    icor: int,
    rayleigh_variant: bool,
    idealized_radiation: bool,
    ramp_sgs: int,
    adaptive_dt: bool,
    adaptive_move: bool,
    mpi_average: bool,
    budget_diagnostics: bool,
) -> Mapping[str, object]:
    values = {
        "lspgrad": lspgrad,
        "icor": icor,
        "rayleigh_variant": rayleigh_variant,
        "idealized_radiation": idealized_radiation,
        "ramp_sgs": ramp_sgs,
        "adaptive_dt": adaptive_dt,
        "adaptive_move": adaptive_move,
        "mpi_average": mpi_average,
        "budget_diagnostics": budget_diagnostics,
    }
    if type(lspgrad) is not int or type(icor) is not int or type(ramp_sgs) is not int:
        raise TypeError("lspgrad, icor, and ramp_sgs must be exact integers")
    for key in (
        "rayleigh_variant",
        "idealized_radiation",
        "adaptive_dt",
        "adaptive_move",
        "mpi_average",
        "budget_diagnostics",
    ):
        if type(values[key]) is not bool:
            raise TypeError(f"{key} must be boolean")
    if any(
        (
            lspgrad != 0,
            icor != 0,
            rayleigh_variant,
            idealized_radiation,
            ramp_sgs != 0,
            adaptive_dt,
            adaptive_move,
            mpi_average,
            budget_diagnostics,
        )
    ):
        raise ValueError(
            "active large-scale nudging has no admitted simultaneous optional, "
            "adaptive, MPI-average, or diagnostic branch"
        )
    return MappingProxyType(values)


def _validate_host_source(
    np: Any, name: str, value: object, shape: tuple[int, ...]
) -> Any:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have native float32 dtype")
    if tuple(value.shape) != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-order float32 shape {shape}")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values")
    return value


@dataclass(frozen=True, slots=True)
class Gm1LargeScaleNudgingCpuProjection:
    """Exact host solve1 increments and solve3 base-profile results."""

    solve1_increments: Mapping[str, Any]
    solve3_base_profiles: Mapping[str, Any]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1LargeScaleNudgingBinding:
    """One exact source-stream binding shared by solve1 tiles and solve3."""

    solve1_increments: Mapping[str, Any]
    solve3_base_profiles: Mapping[str, Any]
    ready_event: Any
    stream: Any
    receipt: Mapping[str, object]


def build_gm1_large_scale_nudging_cpu_projection(
    profiles: Mapping[str, Any],
    averages: Mapping[str, Any],
    base_state: Mapping[str, Any],
    *,
    selection: LargeScaleNudgingSelection,
    external_target_authority_sha256: str,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    lspgrad: int = 0,
    icor: int = 0,
    rayleigh_variant: bool = False,
    idealized_radiation: bool = False,
    ramp_sgs: int = 0,
    adaptive_dt: bool = False,
    adaptive_move: bool = False,
    mpi_average: bool = False,
    budget_diagnostics: bool = False,
) -> Gm1LargeScaleNudgingCpuProjection:
    """Produce source-exact active increments without importing CUDA."""

    if type(selection) is not LargeScaleNudgingSelection:
        raise TypeError("selection must be the exact large-scale-nudging selection")
    if selection.do_lsnudge is not True:
        raise ValueError(
            "disabled large-scale nudging is represented only by omitting the binding"
        )
    if not any(
        (
            selection.do_lsnudge_u,
            selection.do_lsnudge_v,
            selection.do_lsnudge_th,
            selection.do_lsnudge_qv,
        )
    ):
        raise ValueError("active large-scale nudging requires a selected component")
    authority_sha = _require_sha256(
        "external_target_authority_sha256", external_target_authority_sha256
    )
    companions = _require_companions(
        lspgrad=lspgrad,
        icor=icor,
        rayleigh_variant=rayleigh_variant,
        idealized_radiation=idealized_radiation,
        ramp_sgs=ramp_sgs,
        adaptive_dt=adaptive_dt,
        adaptive_move=adaptive_move,
        mpi_average=mpi_average,
        budget_diagnostics=budget_diagnostics,
    )
    profiles = _require_exact_mapping("profiles", profiles, _PROFILE_KEYS)
    averages = _require_exact_mapping("averages", averages, _AVERAGE_KEYS)
    base_state = _require_exact_mapping("base_state", base_state, _BASE_KEYS)
    first = profiles["lsnudge_time1"]
    if not hasattr(first, "shape") or len(first.shape) != 1:
        raise ValueError("lsnudge_time1 must expose the target-profile count")
    num_profiles = int(first.shape[0])
    _require_operator_source()
    np = importlib.import_module("numpy")
    shapes = large_scale_nudging_shapes(nx, ny, nk, num_profiles)
    workspace: dict[str, Any] = {}
    for name in _PROFILE_KEYS:
        source = _validate_host_source(np, name, profiles[name], shapes[name])
        workspace[name] = np.array(source, dtype=np.float32, order="C", copy=True)
    for name in _AVERAGE_KEYS:
        source = _validate_host_source(np, name, averages[name], shapes[name])
        workspace[name] = np.array(source, dtype=np.float32, order="C", copy=True)
    for name in _BASE_KEYS:
        source = _validate_host_source(np, name, base_state[name], shapes[name])
        workspace[name] = np.array(source, dtype=np.float32, order="C", copy=True)
    for name in ("uten", "vten", "thten", "qvten"):
        workspace[name] = np.zeros(shapes[name], dtype=np.float32)

    result = cpu_large_scale_nudging_inplace(
        workspace,
        ni=nx,
        nj=ny,
        nk=nk,
        num_profiles=num_profiles,
        selection=selection,
    )
    if result.receipt.get("status") != CM1_LARGE_SCALE_NUDGING_STATUS:
        raise ValueError("source-exact large-scale-nudging operator status changed")
    solve1 = MappingProxyType(
        {
            "uten": np.ascontiguousarray(workspace["uten"], dtype=np.float32),
            "vten": np.ascontiguousarray(workspace["vten"], dtype=np.float32),
            "thten1": np.ascontiguousarray(workspace["thten"], dtype=np.float32),
            "qvten": np.ascontiguousarray(workspace["qvten"], dtype=np.float32),
        }
    )
    applied = result.receipt["applied"]
    for component, name in (("u", "u0"), ("v", "v0")):
        if applied[component]:
            value = workspace[name]
            if not bool(np.all(value == value[:, :1, :1])):
                raise ValueError(
                    f"source-exact active {name} base update did not broadcast horizontally"
                )
    solve3 = MappingProxyType(
        {
            "ug": np.ascontiguousarray(workspace["ug"], dtype=np.float32),
            "vg": np.ascontiguousarray(workspace["vg"], dtype=np.float32),
            "u0_profile": np.ascontiguousarray(workspace["u0"][:, 0, 0]),
            "v0_profile": np.ascontiguousarray(workspace["v0"][:, 0, 0]),
        }
    )
    source_sha = {
        "profiles": {name: _array_sha256(profiles[name]) for name in _PROFILE_KEYS},
        "averages": {name: _array_sha256(averages[name]) for name in _AVERAGE_KEYS},
        "base_state": {name: _array_sha256(base_state[name]) for name in _BASE_KEYS},
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_LARGE_SCALE_NUDGING_SOLVE1_SOLVE3",
        "domain": {
            "nx": nx,
            "ny": ny,
            "nk": nk,
            "num_profiles": num_profiles,
        },
        "external_target_authority_sha256": authority_sha,
        "source_sha256": source_sha,
        "selection": dict(result.receipt["selection"]),
        "schedule": dict(result.receipt["schedule"]),
        "applied": dict(applied),
        "companion_selection": dict(companions),
        "solve1_output_shapes": {
            name: list(value.shape) for name, value in solve1.items()
        },
        "solve1_output_sha256": {
            name: _array_sha256(value) for name, value in solve1.items()
        },
        "solve3_base_output_sha256": {
            name: result.receipt["output_sha256"][name]
            for name in _BASE_KEYS
        },
        "solve3_base_profile_sha256": {
            name: _array_sha256(value) for name, value in solve3.items()
        },
        "cpu_operator_operation_token_sha256": result.receipt[
            "operation_token_sha256"
        ],
        "cpu_operator_source_sha256": (
            LARGE_SCALE_NUDGING_CPU_OPERATOR_SOURCE_SHA256
        ),
        "source_order": list(CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_ORDER),
        "external_profile_io_or_interpolation_performed": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "base_profiles_in_checkpoint_37": False,
        "claim_limit": CUDA_GM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1LargeScaleNudgingCpuProjection(
        solve1, solve3, MappingProxyType(receipt)
    )


def prepare_cuda_gm1_large_scale_nudging(
    profiles: Mapping[str, Any],
    averages: Mapping[str, Any],
    base_state: Mapping[str, Any],
    *,
    selection: LargeScaleNudgingSelection,
    external_target_authority_sha256: str,
    stream: Any,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    allow_compact_validation: bool = False,
    **companion_selection: object,
) -> CudaGm1LargeScaleNudgingBinding:
    """Stage exact increments and base profiles on the source-owning stream."""

    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) <= 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    if (nx, ny, nk) != (300, 300, 75) and not allow_compact_validation:
        raise ValueError("live large-scale nudging requires 300x300x75 geometry")
    projection = build_gm1_large_scale_nudging_cpu_projection(
        profiles,
        averages,
        base_state,
        selection=selection,
        external_target_authority_sha256=external_target_authority_sha256,
        nx=nx,
        ny=ny,
        nk=nk,
        **companion_selection,
    )
    cupy = importlib.import_module("cupy")
    with stream:
        solve1 = MappingProxyType(
            {name: cupy.asarray(value) for name, value in projection.solve1_increments.items()}
        )
        solve3 = MappingProxyType(
            {name: cupy.asarray(value) for name, value in projection.solve3_base_profiles.items()}
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = dict(projection.receipt)
    receipt.update(
        {
            "status": CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_STATUS,
            "stream_pointer": int(stream.ptr),
            "host_to_device_on_source_stream": True,
            "compact_validation_only": (nx, ny, nk) != (300, 300, 75),
            "live_geometry_admitted": (nx, ny, nk) == (300, 300, 75),
        }
    )
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1LargeScaleNudgingBinding(
        solve1, solve3, event, stream, MappingProxyType(receipt)
    )


def require_cuda_gm1_large_scale_nudging_binding(
    binding: object,
    *,
    stream: Any,
    nx: int,
    ny: int,
    nk: int,
) -> CudaGm1LargeScaleNudgingBinding:
    """Validate the exact selection, source hashes, carriers, and stream."""

    if type(binding) is not CudaGm1LargeScaleNudgingBinding:
        raise TypeError("binding must be the exact large-scale-nudging binding")
    receipt = binding.receipt
    if (
        stream is not binding.stream
        or int(stream.ptr) != receipt.get("stream_pointer")
        or receipt.get("schema") != CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_SCHEMA
        or receipt.get("status") != CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_STATUS
        or receipt.get("domain")
        != {
            "nx": nx,
            "ny": ny,
            "nk": nk,
            "num_profiles": receipt.get("domain", {}).get("num_profiles")
            if isinstance(receipt.get("domain"), Mapping)
            else None,
        }
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("large-scale-nudging binding identity, domain, or stream changed")
    if receipt.get("companion_selection") != {
        "lspgrad": 0,
        "icor": 0,
        "rayleigh_variant": False,
        "idealized_radiation": False,
        "ramp_sgs": 0,
        "adaptive_dt": False,
        "adaptive_move": False,
        "mpi_average": False,
        "budget_diagnostics": False,
    }:
        raise ValueError("large-scale-nudging companion selection changed")
    cupy = importlib.import_module("cupy")
    expected_solve1 = {
        "uten": (nk, ny, nx + 1),
        "vten": (nk, ny + 1, nx),
        "thten1": (nk, ny, nx),
        "qvten": (nk, ny, nx),
    }
    expected_solve3 = {
        "ug": (nk,),
        "vg": (nk,),
        "u0_profile": (nk + 2,),
        "v0_profile": (nk + 2,),
    }
    for values, expected in (
        (binding.solve1_increments, expected_solve1),
        (binding.solve3_base_profiles, expected_solve3),
    ):
        if tuple(values) != tuple(expected):
            raise ValueError("large-scale-nudging device inventory/order changed")
        for name, shape in expected.items():
            value = values[name]
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.float32
                or tuple(value.shape) != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"large-scale-nudging {name} must be device C-f32 {shape}"
                )
    return binding


def enqueue_cuda_gm1_large_scale_nudging_after_held_composition(
    binding: CudaGm1LargeScaleNudgingBinding,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    stream: Any,
) -> Mapping[str, object]:
    """Add the four exact late-solve1 component increments to one tile."""

    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    domain = binding.receipt.get("domain")
    if not isinstance(domain, Mapping) or not isinstance(domain.get("nk"), int):
        raise ValueError("large-scale-nudging binding lost its vertical domain")
    nk = int(domain["nk"])
    binding = require_cuda_gm1_large_scale_nudging_binding(
        binding, stream=stream, nx=spec.nx, ny=spec.ny, nk=nk
    )
    cupy = importlib.import_module("cupy")
    expected = {
        "uten": (nk, spec.cny, spec.cnx + 1),
        "vten": (nk, spec.cny + 1, spec.cnx),
        "thten1": (nk, spec.cny, spec.cnx),
        "qten": (6, nk, spec.cny, spec.cnx),
    }
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    for name, shape in expected.items():
        value = arrays.get(name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {shape}")
    source = binding.solve1_increments
    with stream:
        stream.wait_event(binding.ready_event)
        cupy.add(
            arrays["uten"],
            source["uten"][:, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx + 1],
            out=arrays["uten"],
        )
        cupy.add(
            arrays["vten"],
            source["vten"][:, spec.cj0 : spec.cj0 + spec.cny + 1, spec.ci0 : spec.ci0 + spec.cnx],
            out=arrays["vten"],
        )
        cupy.add(
            arrays["thten1"],
            source["thten1"][:, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx],
            out=arrays["thten1"],
        )
        cupy.add(
            arrays["qten"][0],
            source["qvten"][:, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx],
            out=arrays["qten"][0],
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_LARGE_SCALE_NUDGING_TILE_SCHEMA,
        "status": CUDA_GM1_LARGE_SCALE_NUDGING_TILE_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": binding.receipt[
            "operation_token_sha256"
        ],
        "selection": dict(binding.receipt["selection"]),
        "schedule": dict(binding.receipt["schedule"]),
        "applied": dict(binding.receipt["applied"]),
        "source_order": list(CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_ORDER),
        "same_stream_after_final_held_composition": True,
        "qv_increment_species_index": 0,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


def enqueue_cuda_gm1_large_scale_nudging_solve3_base_update(
    binding: CudaGm1LargeScaleNudgingBinding,
    base_state: Mapping[str, Any],
    *,
    full_held_receipt: Mapping[str, object],
    completed_rk3_operation_token_sha256: str,
    stream: Any,
    allow_compact_validation: bool = False,
) -> Mapping[str, object]:
    """Apply the exact solve3 base update before momentum current-to-A."""

    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    domain = binding.receipt.get("domain") if type(binding) is CudaGm1LargeScaleNudgingBinding else None
    if not isinstance(domain, Mapping):
        raise ValueError("large-scale-nudging binding lost its domain")
    nx, ny, nk = (domain.get(name) for name in ("nx", "ny", "nk"))
    if not all(type(value) is int for value in (nx, ny, nk)):
        raise ValueError("large-scale-nudging binding domain changed")
    if (nx, ny, nk) != (300, 300, 75) and not allow_compact_validation:
        raise ValueError("live solve3 large-scale nudging requires 300x300x75 geometry")
    binding = require_cuda_gm1_large_scale_nudging_binding(
        binding, stream=stream, nx=nx, ny=ny, nk=nk
    )
    completed_rk3_sha = _require_sha256(
        "completed_rk3_operation_token_sha256",
        completed_rk3_operation_token_sha256,
    )
    if (
        not isinstance(full_held_receipt, Mapping)
        or full_held_receipt.get("schema") != _FULL_HELD_TOKEN_SCHEMA
        or full_held_receipt.get("status") != _FULL_HELD_TOKEN_STATUS
        or full_held_receipt.get("operation_token_sha256")
        != _operation_token(full_held_receipt)
        or full_held_receipt.get(
            "large_scale_nudging_binding_operation_token_sha256"
        )
        != binding.receipt.get("operation_token_sha256")
    ):
        raise ValueError(
            "solve3 large-scale nudging requires the exact active solve1 full-held lineage"
        )
    if not isinstance(base_state, Mapping) or tuple(base_state) != _BASE_KEYS:
        raise ValueError("solve3 base_state inventory/order changed")
    cupy = importlib.import_module("cupy")
    expected = {
        "ug": (nk,),
        "vg": (nk,),
        "u0": (nk + 2, ny, nx + 1),
        "v0": (nk + 2, ny + 1, nx),
    }
    for name, shape in expected.items():
        value = base_state[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"solve3 {name} must be device C-f32 {shape}")
    applied = binding.receipt["applied"]
    with stream:
        stream.wait_event(binding.ready_event)
        if applied["u"]:
            cupy.copyto(base_state["ug"], binding.solve3_base_profiles["ug"])
            base_state["u0"][...] = binding.solve3_base_profiles["u0_profile"][:, None, None]
        if applied["v"]:
            cupy.copyto(base_state["vg"], binding.solve3_base_profiles["vg"])
            base_state["v0"][...] = binding.solve3_base_profiles["v0_profile"][:, None, None]
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_SCHEMA,
        "status": CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_STATUS,
        "completed_nrk": 3,
        "stream_pointer": int(stream.ptr),
        "completed_rk3_operation_token_sha256": completed_rk3_sha,
        "solve1_full_held_operation_token_sha256": full_held_receipt[
            "operation_token_sha256"
        ],
        "binding_operation_token_sha256": binding.receipt[
            "operation_token_sha256"
        ],
        "selection": dict(binding.receipt["selection"]),
        "schedule": dict(binding.receipt["schedule"]),
        "applied": dict(applied),
        "source_order": list(CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_ORDER),
        "same_stream_after_held_recapture_before_momentum_current_to_a": True,
        "ready_event_recorded": event is not None,
        "base_profiles_in_checkpoint_37": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = (
    "CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_SCHEMA",
    "CUDA_GM1_LARGE_SCALE_NUDGING_BINDING_STATUS",
    "CUDA_GM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT",
    "CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_SCHEMA",
    "CUDA_GM1_LARGE_SCALE_NUDGING_SOLVE3_STATUS",
    "CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_ORDER",
    "CUDA_GM1_LARGE_SCALE_NUDGING_SOURCE_SHA256",
    "CUDA_GM1_LARGE_SCALE_NUDGING_TILE_SCHEMA",
    "CUDA_GM1_LARGE_SCALE_NUDGING_TILE_STATUS",
    "CudaGm1LargeScaleNudgingBinding",
    "Gm1LargeScaleNudgingCpuProjection",
    "LARGE_SCALE_NUDGING_CPU_OPERATOR_SOURCE_SHA256",
    "build_gm1_large_scale_nudging_cpu_projection",
    "enqueue_cuda_gm1_large_scale_nudging_after_held_composition",
    "enqueue_cuda_gm1_large_scale_nudging_solve3_base_update",
    "prepare_cuda_gm1_large_scale_nudging",
    "require_cuda_gm1_large_scale_nudging_binding",
)
