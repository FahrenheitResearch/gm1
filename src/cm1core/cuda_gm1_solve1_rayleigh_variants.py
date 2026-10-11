"""Typed flat-grid solve1 delivery for source-exact Rayleigh variants.

The default GM1 child Rayleigh kernel remains the unchanged terrain-aware
``irdamp=1/hrdamp=0`` path.  This module prepares the independently evidenced
flat Cartesian active ``irdamp=0,1,2`` / ``hrdamp=0,1`` family on the CPU and exposes
one exact tiled overwrite point immediately after that default assignment.
The overwrite produces the source-selected Rayleigh held values before any
optional large-scale forcing or ``idiss``.  Omitting the binding preserves the
existing receipt shape and bytes.

No active CUDA-vs-native raw-bit or coupled-trajectory claim is made.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .rayleigh_damping import (
    CM1_RAYLEIGH_DAMPING_STATUS,
    RayleighDampingSelection,
    cpu_rayleigh_damping_inplace,
    rayleigh_damping_shapes,
)


CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_SCHEMA = (
    "gm1.cuda-solve1-rayleigh-variant-binding/v1"
)
CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_STATUS = (
    "PREPARED_LIVE_FLAT_SOLVE1_RAYLEIGH_VARIANT"
)
CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_SCHEMA = (
    "gm1.cuda-solve1-rayleigh-variant-tile/v1"
)
CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_STATUS = (
    "ENQUEUED_LIVE_FLAT_SOLVE1_RAYLEIGH_VARIANT"
)
CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_CLAIM_LIMIT = (
    "typed 300x300x75/50 m flat Cartesian CUDA delivery of independently "
    "source-exact CPU active irdamp=0,1,2/hrdamp=0,1 held values at solve1.F "
    "Rayleigh assignment boundary; no terrain, MPI reduction, axisymmetric "
    "lateral coordinate, diagnostics, simultaneous lspgrad/rterm, active "
    "CUDA-vs-native raw-bit oracle, coupled trajectory, production "
    "configuration, forecast, or product"
)
CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_ORDER = (
    "default terrain-Rayleigh assignment is replaced on the identical held arrays",
    "active flat irdamp/hrdamp source-exact held values are established",
    "optional large-scale forcing then idiss follow",
)
RAYLEIGH_CPU_OPERATOR_SOURCE_SHA256 = (
    "028fe4a414fec7ee206afd1c58ae6876aee26b439c42e7ba4e93d7253b9859bb"
)
_SOURCE_NAMES = (
    "zh",
    "zf",
    "xh",
    "yh",
    "ua",
    "va",
    "wa",
    "tha",
    "u0",
    "v0",
    "th0",
    "ruh",
    "ruf",
    "rvh",
    "rvf",
)
_OUTPUT_NAMES = ("uten1", "vten1", "wten1", "thten1")


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


def _f32_bits(np: Any, value: float) -> int:
    return int(np.asarray(np.float32(value), dtype=np.float32).view(np.uint32))


def _require_operator_source() -> None:
    installed = Path(__file__).with_name("rayleigh_damping.py")
    if hashlib.sha256(installed.read_bytes()).hexdigest() != RAYLEIGH_CPU_OPERATOR_SOURCE_SHA256:
        raise RuntimeError("installed Rayleigh CPU operator differs from its source pin")


def _nonnegative_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_geometry(
    *, nx: int, ny: int, nk: int, completed_nrk: int, allow_compact_validation: bool
) -> None:
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    for name, value in (("nx", nx), ("ny", ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not allow_compact_validation and (nx, ny, nk) != (300, 300, 75):
        raise ValueError("live Rayleigh variant requires 300x300x75 geometry")
    if completed_nrk != 0:
        raise ValueError("solve1 Rayleigh variant is admitted only before RK stage 1")


def _validate_selection(selection: RayleighDampingSelection) -> None:
    if type(selection) is not RayleighDampingSelection:
        raise TypeError("selection must be the exact Rayleigh-damping selection type")
    if selection.terrain_flag or selection.axisymmetric != 0:
        raise ValueError("live Rayleigh variant requires flat Cartesian geometry")
    if (
        selection.upper_damping not in (0, 1, 2)
        or selection.horizontal_damping not in (0, 1)
        or (selection.upper_damping, selection.horizontal_damping) == (0, 0)
    ):
        raise ValueError(
            "live Rayleigh variant requires an active irdamp=0,1,2/hrdamp=0,1 pair"
        )
    if float(selection.dx_m) != 50.0 or float(selection.dy_m) != 50.0:
        raise ValueError("live Rayleigh variant requires exact 50 m horizontal spacing")


def _validate_companions(
    *, lspgrad: int, rterm: int, mpi: bool, diagnostics: bool, adaptive_dt: bool
) -> None:
    if lspgrad != 0 or rterm != 0:
        raise ValueError("active Rayleigh variant does not admit simultaneous lspgrad/rterm")
    if mpi is not False or diagnostics is not False or adaptive_dt is not False:
        raise ValueError("active Rayleigh variant requires serial fixed-dt diagnostics-off mode")


@dataclass(frozen=True, slots=True)
class Gm1Solve1RayleighVariantCpuHeld:
    uten1: Any
    vten1: Any
    wten1: Any
    thten1: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Solve1RayleighVariantBinding:
    uten1: Any
    vten1: Any
    wten1: Any
    thten1: Any
    ready_event: Any
    stream: Any
    receipt: Mapping[str, object]


def build_gm1_solve1_rayleigh_variant_cpu_held(
    sources: Mapping[str, Any],
    *,
    selection: RayleighDampingSelection,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    completed_nrk: int = 0,
    lspgrad: int = 0,
    rterm: int = 0,
    mpi: bool = False,
    diagnostics: bool = False,
    adaptive_dt: bool = False,
    allow_compact_validation: bool = False,
) -> Gm1Solve1RayleighVariantCpuHeld:
    """Run the exact CPU operator and project physical staggered held arrays."""

    _validate_selection(selection)
    _validate_geometry(
        nx=nx,
        ny=ny,
        nk=nk,
        completed_nrk=completed_nrk,
        allow_compact_validation=allow_compact_validation,
    )
    _nonnegative_scalar("rdalpha_s_inv", selection.rdalpha_s_inv)
    _validate_companions(
        lspgrad=lspgrad,
        rterm=rterm,
        mpi=mpi,
        diagnostics=diagnostics,
        adaptive_dt=adaptive_dt,
    )
    if not isinstance(sources, Mapping) or tuple(sources) != _SOURCE_NAMES:
        raise ValueError(
            "Rayleigh source inventory/order changed; expected " + ",".join(_SOURCE_NAMES)
        )
    _require_operator_source()
    np = importlib.import_module("numpy")
    shapes = rayleigh_damping_shapes(nx, ny, nk)
    work: dict[str, Any] = {name: sources[name] for name in _SOURCE_NAMES}
    for name in ("tauh", "taus", "tauf", *_OUTPUT_NAMES, "uavg", "vavg", "thavg"):
        work[name] = np.zeros(shapes[name], dtype=np.float32)
    work["cavg"] = np.zeros(shapes["cavg"], dtype=np.float64)
    result = cpu_rayleigh_damping_inplace(
        work, ni=nx, nj=ny, nk=nk, selection=selection
    )
    if result.receipt.get("status") != CM1_RAYLEIGH_DAMPING_STATUS:
        raise ValueError("source-exact Rayleigh receipt status changed")
    projected = {
        "uten1": np.ascontiguousarray(work["uten1"][:, :ny, :], dtype=np.float32),
        "vten1": np.ascontiguousarray(work["vten1"][:, :, :nx], dtype=np.float32),
        "wten1": np.ascontiguousarray(work["wten1"][:, :ny, :nx], dtype=np.float32),
        "thten1": np.ascontiguousarray(work["thten1"][:, :ny, :nx], dtype=np.float32),
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_FLAT_SOLVE1_RAYLEIGH_VARIANT",
        "completed_nrk": 0,
        "domain": {"nx": nx, "ny": ny, "nk": nk, "dx_m": 50.0, "dy_m": 50.0},
        "selection": {
            **dict(result.receipt["selection"]),
            "parameter_f32_bits": {
                name: _f32_bits(np, getattr(selection, name))
                for name in (
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
            },
        },
        "companion_selection": {
            "lspgrad": 0,
            "rterm": 0,
            "mpi": False,
            "diagnostics": False,
            "adaptive_dt": False,
        },
        "required_source_names": list(_SOURCE_NAMES),
        "source_sha256": {name: _array_sha256(sources[name]) for name in _SOURCE_NAMES},
        "output_shapes": {name: list(value.shape) for name, value in projected.items()},
        "output_sha256": {name: _array_sha256(value) for name, value in projected.items()},
        "cpu_operator_operation_token_sha256": result.receipt["operation_token_sha256"],
        "cpu_operator_source_sha256": RAYLEIGH_CPU_OPERATOR_SOURCE_SHA256,
        "source_order": list(CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_ORDER),
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Solve1RayleighVariantCpuHeld(
        **projected, receipt=MappingProxyType(receipt)
    )


def prepare_cuda_gm1_solve1_rayleigh_variant(
    sources: Mapping[str, Any],
    *,
    selection: RayleighDampingSelection,
    stream: Any,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    completed_nrk: int = 0,
    lspgrad: int = 0,
    rterm: int = 0,
    mpi: bool = False,
    diagnostics: bool = False,
    adaptive_dt: bool = False,
) -> CudaGm1Solve1RayleighVariantBinding:
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    cpu = build_gm1_solve1_rayleigh_variant_cpu_held(
        sources,
        selection=selection,
        nx=nx,
        ny=ny,
        nk=nk,
        completed_nrk=completed_nrk,
        lspgrad=lspgrad,
        rterm=rterm,
        mpi=mpi,
        diagnostics=diagnostics,
        adaptive_dt=adaptive_dt,
    )
    cupy = importlib.import_module("cupy")
    with stream:
        device = {name: cupy.asarray(getattr(cpu, name)) for name in _OUTPUT_NAMES}
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = dict(cpu.receipt)
    receipt["status"] = CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_STATUS
    receipt["stream_pointer"] = int(stream.ptr)
    receipt["host_to_device_on_source_stream"] = True
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Solve1RayleighVariantBinding(
        **device, ready_event=event, stream=stream, receipt=MappingProxyType(receipt)
    )


def require_cuda_gm1_solve1_rayleigh_variant_binding(
    binding: CudaGm1Solve1RayleighVariantBinding,
    *,
    spec: Gm1TileSpec,
    stream: Any,
) -> CudaGm1Solve1RayleighVariantBinding:
    if type(binding) is not CudaGm1Solve1RayleighVariantBinding:
        raise TypeError("binding must be the exact live Rayleigh-variant binding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (300, 300):
        raise ValueError("Rayleigh-variant tile must belong to the 300 grid")
    receipt = binding.receipt
    if stream is not binding.stream or int(stream.ptr) != receipt.get("stream_pointer"):
        raise ValueError("Rayleigh-variant tile stream differs from source-owning stream")
    if (
        receipt.get("schema") != CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_SCHEMA
        or receipt.get("status") != CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_STATUS
        or receipt.get("completed_nrk") != 0
        or receipt.get("domain")
        != {"nx": 300, "ny": 300, "nk": 75, "dx_m": 50.0, "dy_m": 50.0}
        or receipt.get("companion_selection")
        != {
            "lspgrad": 0,
            "rterm": 0,
            "mpi": False,
            "diagnostics": False,
            "adaptive_dt": False,
        }
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("Rayleigh-variant binding identity or cadence changed")
    cupy = importlib.import_module("cupy")
    shapes = {
        "uten1": (75, 300, 301),
        "vten1": (75, 301, 300),
        "wten1": (75, 300, 300),
        "thten1": (75, 300, 300),
    }
    for name, shape in shapes.items():
        value = getattr(binding, name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"binding {name} must be C-order CUDA float32 {shape}")
    return binding


def enqueue_cuda_gm1_solve1_rayleigh_variant_override(
    binding: CudaGm1Solve1RayleighVariantBinding,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    stream: Any,
) -> Mapping[str, object]:
    binding = require_cuda_gm1_solve1_rayleigh_variant_binding(
        binding, spec=spec, stream=stream
    )
    cupy = importlib.import_module("cupy")
    expected = {
        "uten1": (75, spec.cny, spec.cnx + 1),
        "vten1": (75, spec.cny + 1, spec.cnx),
        "wten1": (75, spec.cny, spec.cnx),
        "thten1": (75, spec.cny, spec.cnx),
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
            raise ValueError(f"{name} must be C-order CUDA float32 shape {shape}")
    slices = {
        "uten1": (slice(None), slice(spec.cj0, spec.cj0 + spec.cny), slice(spec.ci0, spec.ci0 + spec.cnx + 1)),
        "vten1": (slice(None), slice(spec.cj0, spec.cj0 + spec.cny + 1), slice(spec.ci0, spec.ci0 + spec.cnx)),
        "wten1": (slice(None), slice(spec.cj0, spec.cj0 + spec.cny), slice(spec.ci0, spec.ci0 + spec.cnx)),
        "thten1": (slice(None), slice(spec.cj0, spec.cj0 + spec.cny), slice(spec.ci0, spec.ci0 + spec.cnx)),
    }
    with stream:
        stream.wait_event(binding.ready_event)
        for name in _OUTPUT_NAMES:
            cupy.copyto(arrays[name], getattr(binding, name)[slices[name]])
    receipt = binding.receipt
    tile_receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_SCHEMA,
        "status": CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": receipt["operation_token_sha256"],
        "cpu_operator_operation_token_sha256": receipt["cpu_operator_operation_token_sha256"],
        "selection": dict(receipt["selection"]),
        "companion_selection": dict(receipt["companion_selection"]),
        "source_order": list(CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_ORDER),
        "default_rayleigh_values_overwritten_before_other_held_terms": True,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_CLAIM_LIMIT,
    }
    tile_receipt["operation_token_sha256"] = _operation_token(tile_receipt)
    return MappingProxyType(tile_receipt)


CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = [
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_SCHEMA",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_BINDING_STATUS",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_CLAIM_LIMIT",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_ORDER",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_SOURCE_SHA256",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_SCHEMA",
    "CUDA_GM1_SOLVE1_RAYLEIGH_VARIANT_TILE_STATUS",
    "CudaGm1Solve1RayleighVariantBinding",
    "Gm1Solve1RayleighVariantCpuHeld",
    "RAYLEIGH_CPU_OPERATOR_SOURCE_SHA256",
    "build_gm1_solve1_rayleigh_variant_cpu_held",
    "enqueue_cuda_gm1_solve1_rayleigh_variant_override",
    "prepare_cuda_gm1_solve1_rayleigh_variant",
    "require_cuda_gm1_solve1_rayleigh_variant_binding",
]
