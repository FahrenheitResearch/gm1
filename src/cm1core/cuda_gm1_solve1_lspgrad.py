"""Typed live solve1 binding for the source-exact CM1 ``lspgrad`` operator.

The independently evidenced CPU operator owns ``solve1.F:536-681``.  This
module prepares its exact full-domain default-REAL increments once and exposes
the one legal tiled enqueue point: after Rayleigh and before ``idiss``.  It
does not admit a coupled trajectory.  ``lspgrad=0`` remains the existing
production identity path and is represented by the caller omitting a binding.
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
from .large_scale_pressure_gradient import (
    CM1_LSPGRAD_STATUS,
    LargeScalePressureGradientSelection,
    cpu_large_scale_pressure_gradient_inplace,
)


CUDA_GM1_SOLVE1_LSPGRAD_BINDING_SCHEMA = (
    "gm1.cuda-solve1-lspgrad-binding/v1"
)
CUDA_GM1_SOLVE1_LSPGRAD_BINDING_STATUS = (
    "PREPARED_LIVE_SOLVE1_LSPGRAD_AFTER_RAYLEIGH"
)
CUDA_GM1_SOLVE1_LSPGRAD_TILE_SCHEMA = "gm1.cuda-solve1-lspgrad-tile/v1"
CUDA_GM1_SOLVE1_LSPGRAD_TILE_STATUS = (
    "ENQUEUED_LIVE_SOLVE1_LSPGRAD_AFTER_RAYLEIGH"
)
CUDA_GM1_SOLVE1_LSPGRAD_CLAIM_LIMIT = (
    "typed 300x300x75/50 m CUDA delivery of independently source-exact CPU "
    "lspgrad=1..4 increments at the solve1.F post-Rayleigh/pre-idiss boundary; "
    "no CUDA-vs-native coupled oracle, adaptive lspgrad, Coriolis, complete "
    "solve1/RK trajectory, production configuration, forecast, or product"
)
CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_ORDER = (
    "terrain Rayleigh establishes momentum and theta held terms",
    "active lspgrad=1..4 adds exact large-scale pressure-gradient momentum forcing",
    "eqtset=2 idiss follows the pressure-gradient forcing boundary",
)
LSPGRAD_CPU_OPERATOR_SOURCE_SHA256 = (
    "a36ae80064c24a538ec8a34af6ea7e5744ed49b6b5b13d8fc53602bef0378f1c"
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


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _required_source_names(lspgrad: int) -> tuple[str, ...]:
    if lspgrad == 1:
        return ("u0_window", "v0_window")
    if lspgrad == 2:
        return ("ug", "vg")
    if lspgrad == 3:
        return ("vg",)
    if lspgrad == 4:
        return ("ulspg", "vlspg")
    raise ValueError("live solve1 lspgrad binding supports only active values 1..4")


def _validate_geometry(
    *, nx: int, ny: int, nk: int, dx: float, dy: float, completed_nrk: int
) -> None:
    if (nx, ny, nk) != (300, 300, 75):
        raise ValueError(
            "live solve1 lspgrad binding requires the admitted 300x300x75 geometry"
        )
    for name, value in (("dx", dx), ("dy", dy)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) != 50.0:
            raise ValueError("live solve1 lspgrad binding requires exact 50 m spacing")
    if completed_nrk != 0:
        raise ValueError("solve1 lspgrad forcing is admitted only before RK stage 1")


def _validate_companion_selection(
    *, icor: int, axisymmetric: int, adaptive_lspgrad: bool
) -> None:
    if icor != 0:
        raise ValueError(
            "active lspgrad binding does not admit simultaneous Coriolis forcing"
        )
    if axisymmetric != 0:
        raise ValueError("active lspgrad binding requires Cartesian axisymmetric=0")
    if adaptive_lspgrad is not False:
        raise ValueError("adaptive lspgrad is hard-disabled by CM1 r21.1")


def _require_operator_source() -> None:
    installed = Path(__file__).with_name("large_scale_pressure_gradient.py")
    actual = hashlib.sha256(installed.read_bytes()).hexdigest()
    if actual != LSPGRAD_CPU_OPERATOR_SOURCE_SHA256:
        raise RuntimeError("installed lspgrad CPU operator differs from its source pin")


@dataclass(frozen=True, slots=True)
class Gm1Solve1LspgradCpuIncrements:
    """Exact host increments projected onto CM1 physical u/v staggering."""

    uten1: Any
    vten1: Any
    ulspg: Any
    vlspg: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Solve1LspgradBinding:
    """One full-domain live device binding shared by every solve1 tile."""

    uten1: Any
    vten1: Any
    ready_event: Any
    stream: Any
    receipt: Mapping[str, object]


def build_gm1_solve1_lspgrad_cpu_increments(
    sources: Mapping[str, Any],
    *,
    selection: LargeScalePressureGradientSelection,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    dx: float = 50.0,
    dy: float = 50.0,
    completed_nrk: int = 0,
    icor: int = 0,
    axisymmetric: int = 0,
    adaptive_lspgrad: bool = False,
) -> Gm1Solve1LspgradCpuIncrements:
    """Build exact active-branch increments without importing CUDA."""

    _validate_geometry(
        nx=nx,
        ny=ny,
        nk=nk,
        dx=dx,
        dy=dy,
        completed_nrk=completed_nrk,
    )
    _validate_companion_selection(
        icor=icor,
        axisymmetric=axisymmetric,
        adaptive_lspgrad=adaptive_lspgrad,
    )
    if type(selection) is not LargeScalePressureGradientSelection:
        raise TypeError("selection must be the exact lspgrad selection type")
    required = _required_source_names(selection.lspgrad)
    if not isinstance(sources, Mapping) or tuple(sources) != required:
        raise ValueError(
            "lspgrad source inventory/order changed; expected " + ",".join(required)
        )
    _require_operator_source()
    np = importlib.import_module("numpy")
    workspace: dict[str, Any] = {
        "uten1": np.zeros((nk, ny + 1, nx + 1), dtype=np.float32),
        "vten1": np.zeros((nk, ny + 1, nx + 1), dtype=np.float32),
        "ulspg": np.zeros((nk,), dtype=np.float32),
        "vlspg": np.zeros((nk,), dtype=np.float32),
    }
    for name in required:
        value = sources[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        workspace[name] = value
    result = cpu_large_scale_pressure_gradient_inplace(
        workspace,
        ni=nx,
        nj=ny,
        nk=nk,
        selection=selection,
    )
    if result.receipt.get("status") != CM1_LSPGRAD_STATUS:
        raise ValueError("source-exact lspgrad operator receipt status changed")
    uten1 = np.ascontiguousarray(workspace["uten1"][:, :ny, :], dtype=np.float32)
    vten1 = np.ascontiguousarray(workspace["vten1"][:, :, :nx], dtype=np.float32)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_LSPGRAD_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_SOLVE1_LSPGRAD_INCREMENTS",
        "completed_nrk": 0,
        "domain": {"nx": nx, "ny": ny, "nk": nk, "dx": 50.0, "dy": 50.0},
        "selection": dict(result.receipt["selection"]),
        "companion_selection": {
            "icor": 0,
            "axisymmetric": 0,
            "adaptive_lspgrad": False,
        },
        "required_source_names": list(required),
        "source_sha256": {name: _array_sha256(sources[name]) for name in required},
        "output_shapes": {
            "uten1": list(uten1.shape),
            "vten1": list(vten1.shape),
            "ulspg": [nk],
            "vlspg": [nk],
        },
        "output_sha256": {
            "uten1": _array_sha256(uten1),
            "vten1": _array_sha256(vten1),
            "ulspg": _array_sha256(result.ulspg),
            "vlspg": _array_sha256(result.vlspg),
        },
        "cpu_operator_operation_token_sha256": result.receipt[
            "operation_token_sha256"
        ],
        "cpu_operator_source_sha256": LSPGRAD_CPU_OPERATOR_SOURCE_SHA256,
        "source_order": list(CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_ORDER),
        "adaptive_lspgrad": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_LSPGRAD_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Solve1LspgradCpuIncrements(
        uten1=uten1,
        vten1=vten1,
        ulspg=result.ulspg,
        vlspg=result.vlspg,
        receipt=MappingProxyType(receipt),
    )


def prepare_cuda_gm1_solve1_lspgrad(
    sources: Mapping[str, Any],
    *,
    selection: LargeScalePressureGradientSelection,
    stream: Any,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    dx: float = 50.0,
    dy: float = 50.0,
    completed_nrk: int = 0,
    icor: int = 0,
    axisymmetric: int = 0,
    adaptive_lspgrad: bool = False,
) -> CudaGm1Solve1LspgradBinding:
    """Upload exact full-domain increments onto the explicit source stream."""

    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    cpu = build_gm1_solve1_lspgrad_cpu_increments(
        sources,
        selection=selection,
        nx=nx,
        ny=ny,
        nk=nk,
        dx=dx,
        dy=dy,
        completed_nrk=completed_nrk,
        icor=icor,
        axisymmetric=axisymmetric,
        adaptive_lspgrad=adaptive_lspgrad,
    )
    cupy = importlib.import_module("cupy")
    with stream:
        uten1 = cupy.asarray(cpu.uten1)
        vten1 = cupy.asarray(cpu.vten1)
        ready_event = cupy.cuda.Event(disable_timing=True)
        ready_event.record(stream)
    receipt = dict(cpu.receipt)
    receipt["status"] = CUDA_GM1_SOLVE1_LSPGRAD_BINDING_STATUS
    receipt["stream_pointer"] = int(stream.ptr)
    receipt["host_to_device_on_source_stream"] = True
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Solve1LspgradBinding(
        uten1=uten1,
        vten1=vten1,
        ready_event=ready_event,
        stream=stream,
        receipt=MappingProxyType(receipt),
    )


def require_cuda_gm1_solve1_lspgrad_binding(
    binding: CudaGm1Solve1LspgradBinding,
    *,
    spec: Gm1TileSpec,
    stream: Any,
) -> CudaGm1Solve1LspgradBinding:
    """Validate exact config, cadence, stream, and device carriers pre-enqueue."""

    if type(binding) is not CudaGm1Solve1LspgradBinding:
        raise TypeError("binding must be the exact live solve1 lspgrad binding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (300, 300):
        raise ValueError("lspgrad tile must belong to the admitted 300 grid")
    if stream is not binding.stream or int(stream.ptr) != binding.receipt.get(
        "stream_pointer"
    ):
        raise ValueError("lspgrad tile stream differs from the source-owning stream")
    receipt = binding.receipt
    if (
        receipt.get("schema") != CUDA_GM1_SOLVE1_LSPGRAD_BINDING_SCHEMA
        or receipt.get("status") != CUDA_GM1_SOLVE1_LSPGRAD_BINDING_STATUS
        or receipt.get("completed_nrk") != 0
        or receipt.get("companion_selection")
        != {"icor": 0, "axisymmetric": 0, "adaptive_lspgrad": False}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("lspgrad binding identity or cadence changed")
    cupy = importlib.import_module("cupy")
    for name, value, shape in (
        ("uten1", binding.uten1, (75, 300, 301)),
        ("vten1", binding.vten1, (75, 301, 300)),
    ):
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"binding {name} must be C-order CUDA float32 {shape}")
    return binding


def enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh(
    binding: CudaGm1Solve1LspgradBinding,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    stream: Any,
) -> Mapping[str, object]:
    """Add one exact global forcing slice to a tile after Rayleigh."""

    binding = require_cuda_gm1_solve1_lspgrad_binding(
        binding, spec=spec, stream=stream
    )
    receipt = binding.receipt
    cupy = importlib.import_module("cupy")
    expected = {
        "uten1": (75, spec.cny, spec.cnx + 1),
        "vten1": (75, spec.cny + 1, spec.cnx),
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
    with stream:
        stream.wait_event(binding.ready_event)
        cupy.add(
            arrays["uten1"],
            binding.uten1[
                :, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx + 1
            ],
            out=arrays["uten1"],
        )
        cupy.add(
            arrays["vten1"],
            binding.vten1[
                :, spec.cj0 : spec.cj0 + spec.cny + 1, spec.ci0 : spec.ci0 + spec.cnx
            ],
            out=arrays["vten1"],
        )
    tile_receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_LSPGRAD_TILE_SCHEMA,
        "status": CUDA_GM1_SOLVE1_LSPGRAD_TILE_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": receipt["operation_token_sha256"],
        "selection": dict(receipt["selection"]),
        "companion_selection": dict(receipt["companion_selection"]),
        "source_order": list(CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_ORDER),
        "same_stream_after_rayleigh_before_idiss": True,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_LSPGRAD_CLAIM_LIMIT,
    }
    tile_receipt["operation_token_sha256"] = _operation_token(tile_receipt)
    return MappingProxyType(tile_receipt)


CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = [
    "CUDA_GM1_SOLVE1_LSPGRAD_BINDING_SCHEMA",
    "CUDA_GM1_SOLVE1_LSPGRAD_BINDING_STATUS",
    "CUDA_GM1_SOLVE1_LSPGRAD_CLAIM_LIMIT",
    "CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_ORDER",
    "CUDA_GM1_SOLVE1_LSPGRAD_SOURCE_SHA256",
    "CUDA_GM1_SOLVE1_LSPGRAD_TILE_SCHEMA",
    "CUDA_GM1_SOLVE1_LSPGRAD_TILE_STATUS",
    "CudaGm1Solve1LspgradBinding",
    "Gm1Solve1LspgradCpuIncrements",
    "LSPGRAD_CPU_OPERATOR_SOURCE_SHA256",
    "build_gm1_solve1_lspgrad_cpu_increments",
    "enqueue_cuda_gm1_solve1_lspgrad_after_rayleigh",
    "prepare_cuda_gm1_solve1_lspgrad",
    "require_cuda_gm1_solve1_lspgrad_binding",
]
