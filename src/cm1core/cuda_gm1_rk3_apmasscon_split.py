"""Split CUDA lifecycle for CM1's source-ordered final-RK mass correction.

``solve2.F`` starts the final-RK dry-mass and Exner layer reductions before
momentum boundaries and SGS-TKE, then consumes those ledgers and applies the
uniform Exner correction after those operators.  The older bounded CUDA
launcher enqueues all three kernels contiguously.  This module exposes the
same pinned kernels as a typed begin/finish pair on one CUDA stream so a
future live RK3 coordinator can place other exact operators between them.

The local tokens prove only enqueue identity and device bindings.  They do
not prove which operations were interposed, publish post-output hashes, or
admit a broader RK3 mass-finish/completed-stage token.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_apmasscon import (
    CUDA_APMASSCON_ABI,
    CUDA_APMASSCON_KERNEL_NAMES,
    CUDA_APMASSCON_SOURCE_SHA256,
    CUDA_APMASSCON_THREADS,
    DEFAULT_APMASSCON_SELECTION,
    SOURCE_ANCHORS,
    ApMassConSelection,
    CompiledApMassConKernels,
    _default_constants_f32,
    _require_device_arrays,
    _validate_grid,
    _validate_selection,
    _validated_mass1,
    compile_cuda_apmasscon,
)


CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-apmasscon-split-reduction-token/v1"
)
CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_STATUS = (
    "ENQUEUED_LOCAL_RK3_APMASSCON_REDUCTION_ONLY"
)
CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-apmasscon-split-finish-token/v1"
)
CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_STATUS = (
    "ENQUEUED_LOCAL_RK3_APMASSCON_FINISH_ONLY"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT = (
    "one local serial Cartesian moist final-RK apmasscon begin or finish "
    "enqueue using the pinned CM1 r21.1 CUDA kernels and the same physical "
    "arrays on one CUDA stream. The pair does not prove the intervening "
    "momentum-boundary/TKE lineage, bind eqtset2 remove/snapshot/EOS, publish "
    "post-output hashes, emit a GM1 RK3 mass-finish or completed-stage token, "
    "rebuild rr/rf, perform MPI, or establish solve3, WSM6, forecast, or a "
    "five-metre meteorological product"
)

_SELECTION_RECEIPT = MappingProxyType(
    {
        "dynamics.dry_mass_correction": 1,
        "runtime.nrk": 3,
        "runtime.nrkmax": 3,
        "dynamics.pressure_solver": 3,
        "physics.moist": 1,
        "grid.axisymmetric": 0,
        "runtime.testcase": 0,
    }
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


def _pointer_inventory(arrays: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(arrays[name].data.ptr) for name in arrays}


def _pointer_inventory_sha256(pointers: Mapping[str, int]) -> str:
    return hashlib.sha256(_canonical(dict(pointers))).hexdigest()


def _compiled_receipt(compiled: CompiledApMassConKernels) -> dict[str, object]:
    return {
        "abi": CUDA_APMASSCON_ABI,
        "names": list(CUDA_APMASSCON_KERNEL_NAMES),
        "source_sha256": CUDA_APMASSCON_SOURCE_SHA256,
        "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
        "ptx_bytes": len(compiled.ptx),
        "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
        "cubin_bytes": len(compiled.cubin),
        "target_arch": compiled.target_arch,
        "ptx_options": list(compiled.ptx_options),
        "cubin_options": list(compiled.cubin_options),
    }


def _stream_pointer(stream: object) -> int:
    value = getattr(stream, "ptr", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("CUDA stream pointer must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Gm1Rk3ApMassConSplitReductionToken:
    """Exact local identity for the enqueued layer-reduction phase only."""

    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3ApMassConSplitReductionLaunch:
    ready_event: Any
    compiled: CompiledApMassConKernels
    stream: Any
    arrays: Mapping[str, Any]
    ni: int
    nj: int
    nk: int
    mass1_f64: float
    layer_dry_mass: Any
    layer_pressure_sum: Any
    diagnostics: Any
    correction: Any
    token: Gm1Rk3ApMassConSplitReductionToken


@dataclass(frozen=True, slots=True)
class Gm1Rk3ApMassConSplitFinishToken:
    """Exact local identity for finish/apply enqueue after one reduction."""

    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3ApMassConSplitFinishLaunch:
    ready_event: Any
    reduction: CudaGm1Rk3ApMassConSplitReductionLaunch
    diagnostics: Any
    correction: Any
    token: Gm1Rk3ApMassConSplitFinishToken


def _require_reduction_token(
    launch: object,
) -> tuple[CudaGm1Rk3ApMassConSplitReductionLaunch, Mapping[str, object]]:
    if type(launch) is not CudaGm1Rk3ApMassConSplitReductionLaunch:
        raise TypeError(
            "reduction must be the exact "
            "CudaGm1Rk3ApMassConSplitReductionLaunch type"
        )
    if type(launch.token) is not Gm1Rk3ApMassConSplitReductionToken:
        raise TypeError(
            "reduction token must be the exact "
            "Gm1Rk3ApMassConSplitReductionToken type"
        )
    receipt = launch.token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("split reduction receipt must be a mapping")
    if (
        receipt.get("schema")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("source_order_phase") != "solve2.F:1616-1656"
        or receipt.get("interposed_runtime_lineage_bound") is not False
        or receipt.get("output_checkpoint_bound") is not False
        or receipt.get("broader_rk3_mass_finish") is not False
        or receipt.get("claim_limit") != GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("split reduction token identity or claim changed")
    _lower_sha(
        "split reduction pointer inventory",
        receipt.get("pointer_inventory_sha256"),
    )
    _lower_sha(
        "split reduction operation token",
        receipt.get("operation_token_sha256"),
    )
    if receipt.get("selection") != dict(_SELECTION_RECEIPT):
        raise ValueError("split reduction selected branch changed")
    if type(launch.compiled) is not CompiledApMassConKernels:
        raise TypeError("split reduction compiled image type changed")
    domain = receipt.get("domain")
    expected_domain = {
        "ni": launch.ni,
        "nj": launch.nj,
        "nk": launch.nk,
        "words": launch.ni * launch.nj * launch.nk,
    }
    if not isinstance(domain, Mapping) or dict(domain) != expected_domain:
        raise ValueError("split reduction domain binding changed")
    if receipt.get("mass1_f64") != launch.mass1_f64:
        raise ValueError("split reduction mass1 binding changed")
    _validated_mass1(launch.mass1_f64)
    stream_pointer = _stream_pointer(launch.stream)
    if receipt.get("stream_pointer") != stream_pointer:
        raise ValueError("split reduction stream binding changed")
    if receipt.get("kernel") != _compiled_receipt(launch.compiled):
        raise ValueError("split reduction compiled kernel binding changed")
    if receipt.get("runtime") != dict(launch.compiled.runtime_identity):
        raise ValueError("split reduction CUDA runtime binding changed")
    return launch, receipt


def require_gm1_rk3_apmasscon_split_reduction_token(
    reduction_token: object,
    *,
    stream_pointer: int | None = None,
) -> Mapping[str, object]:
    """Validate the exact local begin token without implying a finish.

    The reduction is deliberately exposed as its own source-order checkpoint:
    CM1 interposes final momentum boundaries and SGS-TKE between this token and
    the finish token.  A successful validation therefore authenticates only
    the local reduction enqueue and retains every negative scientific claim.
    """

    if type(reduction_token) is not Gm1Rk3ApMassConSplitReductionToken:
        raise TypeError(
            "reduction_token must be the exact "
            "Gm1Rk3ApMassConSplitReductionToken type"
        )
    reduction = reduction_token.receipt
    if not isinstance(reduction, Mapping):
        raise TypeError("split reduction token receipt must be a mapping")
    if (
        reduction.get("schema")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_SCHEMA
        or reduction.get("status")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_STATUS
        or reduction.get("completed_nrk") != 3
        or reduction.get("source_order_phase") != "solve2.F:1616-1656"
        or reduction.get("next_source_order_phase")
        != "momentum boundaries then SGS-TKE RK3"
        or reduction.get("interposed_runtime_lineage_bound") is not False
        or reduction.get("output_checkpoint_bound") is not False
        or reduction.get("broader_rk3_mass_finish") is not False
        or reduction.get("mpi_performed") is not False
        or reduction.get("claim_limit") != GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT
        or reduction.get("operation_token_sha256")
        != _operation_token(reduction)
        or reduction.get("selection") != dict(_SELECTION_RECEIPT)
    ):
        raise ValueError("split reduction token identity or claim changed")
    operation = _lower_sha(
        "split reduction operation token",
        reduction.get("operation_token_sha256"),
    )
    pointer_inventory = _lower_sha(
        "split reduction pointer inventory",
        reduction.get("pointer_inventory_sha256"),
    )
    actual_stream = reduction.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("split reduction stream pointer is invalid")
    if stream_pointer is not None:
        if (
            isinstance(stream_pointer, bool)
            or not isinstance(stream_pointer, int)
            or stream_pointer <= 0
        ):
            raise ValueError("stream_pointer must be a positive integer")
        if actual_stream != stream_pointer:
            raise ValueError("split reduction token belongs to another CUDA stream")
    domain = reduction.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError("split reduction domain binding is missing")
    ni = domain.get("ni")
    nj = domain.get("nj")
    nk = domain.get("nk")
    words = domain.get("words")
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in (ni, nj, nk, words)
        )
        or words != ni * nj * nk
    ):
        raise ValueError("split reduction domain binding changed")
    mass1 = reduction.get("mass1_f64")
    _validated_mass1(mass1)
    kernel = reduction.get("kernel")
    runtime = reduction.get("runtime")
    if not isinstance(kernel, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("split reduction compiled/runtime identity is missing")
    return MappingProxyType(
        {
            "operation_token_sha256": operation,
            "stream_pointer": actual_stream,
            "domain": dict(domain),
            "mass1_f64": mass1,
            "pointer_inventory_sha256": pointer_inventory,
            "kernel": dict(kernel),
            "runtime": dict(runtime),
            "interposed_runtime_lineage_bound": False,
            "output_checkpoint_bound": False,
            "broader_rk3_mass_finish": False,
        }
    )


def require_gm1_rk3_apmasscon_split_token_pair(
    reduction_token: object,
    finish_token: object,
    *,
    stream_pointer: int | None = None,
) -> Mapping[str, object]:
    """Validate one exact local begin/finish token pair.

    This check deliberately proves only the immutable relationship between the
    two local enqueue receipts.  In particular, the receipts themselves say
    that no intervening momentum-boundary/TKE lineage or post-output
    checkpoint is bound, so a successful return is not a broader RK3
    mass-finish admission.
    """

    if type(finish_token) is not Gm1Rk3ApMassConSplitFinishToken:
        raise TypeError(
            "finish_token must be the exact Gm1Rk3ApMassConSplitFinishToken type"
        )
    admitted_reduction = require_gm1_rk3_apmasscon_split_reduction_token(
        reduction_token,
        stream_pointer=stream_pointer,
    )
    finish = finish_token.receipt
    if not isinstance(finish, Mapping):
        raise TypeError("split finish token receipt must be a mapping")
    reduction_operation = admitted_reduction["operation_token_sha256"]
    pointer_inventory = admitted_reduction["pointer_inventory_sha256"]
    actual_stream = admitted_reduction["stream_pointer"]
    domain = admitted_reduction["domain"]
    mass1 = admitted_reduction["mass1_f64"]
    kernel = admitted_reduction["kernel"]
    runtime = admitted_reduction["runtime"]
    if (
        finish.get("schema")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_SCHEMA
        or finish.get("status")
        != CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_STATUS
        or finish.get("completed_nrk") != 3
        or finish.get("source_order_phase") != "solve2.F:1972-2030"
        or finish.get("required_interposed_source_order")
        != [
            "solve2.F:1662-1680 final momentum boundaries",
            "solve2.F:1708-1803 SGS-TKE RK3",
        ]
        or finish.get("output_fields_mutated") != ["pp3d", "prs", "rho"]
        or finish.get("interposed_runtime_lineage_bound") is not False
        or finish.get("output_checkpoint_bound") is not False
        or finish.get("broader_rk3_mass_finish") is not False
        or finish.get("rr_rf_rebuilt") is not False
        or finish.get("mpi_performed") is not False
        or finish.get("claim_limit") != GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT
        or finish.get("operation_token_sha256") != _operation_token(finish)
        or finish.get("reduction_token_sha256") != reduction_operation
        or finish.get("stream_pointer") != actual_stream
        or finish.get("pointer_inventory_sha256") != pointer_inventory
        or finish.get("domain") != dict(domain)
        or finish.get("mass1_f64") != mass1
        or finish.get("selection") != dict(_SELECTION_RECEIPT)
        or finish.get("kernel") != dict(kernel)
        or finish.get("runtime") != dict(runtime)
    ):
        raise ValueError("split finish token identity, lineage, or claim changed")
    finish_operation = _lower_sha(
        "split finish operation token", finish.get("operation_token_sha256")
    )
    return MappingProxyType(
        {
            "reduction_operation_token_sha256": reduction_operation,
            "finish_operation_token_sha256": finish_operation,
            "stream_pointer": actual_stream,
            "domain": dict(domain),
            "mass1_f64": mass1,
            "pointer_inventory_sha256": pointer_inventory,
            "interposed_runtime_lineage_bound": False,
            "output_checkpoint_bound": False,
            "broader_rk3_mass_finish": False,
        }
    )


def begin_cuda_gm1_rk3_apmasscon_split(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    mass1: float,
    selection: ApMassConSelection = DEFAULT_APMASSCON_SELECTION,
    stream: Any | None = None,
    compiled: CompiledApMassConKernels | None = None,
) -> CudaGm1Rk3ApMassConSplitReductionLaunch:
    """Enqueue only ``solve2.F:1616-1656`` layer reductions."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    mass1_value = _validated_mass1(mass1)
    cupy = importlib.import_module("cupy")
    checked = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_apmasscon()
    elif type(compiled) is not CompiledApMassConKernels:
        raise TypeError("compiled must be the exact CompiledApMassConKernels type")
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    stream_pointer = _stream_pointer(stream)
    np = importlib.import_module("numpy")
    constants = _default_constants_f32(np)
    layer_dry_mass = cupy.empty((nk,), dtype=cupy.float64)
    layer_pressure_sum = cupy.empty((nk,), dtype=cupy.float64)
    diagnostics = cupy.empty((3,), dtype=cupy.float64)
    correction = cupy.empty((1,), dtype=cupy.float32)
    compiled.kernels[0](
        (nk, 1, 1),
        (CUDA_APMASSCON_THREADS, 1, 1),
        (
            checked["rho"],
            checked["ruh"],
            checked["rvh"],
            checked["rmh"],
            checked["pi0"],
            checked["pp3d"],
            layer_dry_mass,
            layer_pressure_sum,
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    pointers = _pointer_inventory(checked)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_STATUS,
        "completed_nrk": 3,
        "stream_pointer": stream_pointer,
        "event_pointer": int(event.ptr),
        "source_order_phase": "solve2.F:1616-1656",
        "next_source_order_phase": "momentum boundaries then SGS-TKE RK3",
        "domain": {"ni": ni, "nj": nj, "nk": nk, "words": ni * nj * nk},
        "mass1_f64": mass1_value,
        "selection": dict(_SELECTION_RECEIPT),
        "pointer_inventory_sha256": _pointer_inventory_sha256(pointers),
        "ledger_shapes": {
            "layer_dry_mass": [nk],
            "layer_pressure_sum": [nk],
        },
        "kernel": _compiled_receipt(compiled),
        "runtime": dict(compiled.runtime_identity),
        "constants_f32": {
            "rddcv": float(constants["rddcv"]),
            "p00": float(constants["p00"]),
            "rd": float(constants["rd"]),
            "rv": float(constants["rv"]),
            "cpdrd": float(constants["cpdrd"]),
        },
        "interposed_runtime_lineage_bound": False,
        "output_checkpoint_bound": False,
        "broader_rk3_mass_finish": False,
        "mpi_performed": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchor": SOURCE_ANCHORS["layer_reductions"],
        },
        "claim_limit": GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = Gm1Rk3ApMassConSplitReductionToken(MappingProxyType(receipt))
    return CudaGm1Rk3ApMassConSplitReductionLaunch(
        ready_event=event,
        compiled=compiled,
        stream=stream,
        arrays=MappingProxyType(checked),
        ni=ni,
        nj=nj,
        nk=nk,
        mass1_f64=mass1_value,
        layer_dry_mass=layer_dry_mass,
        layer_pressure_sum=layer_pressure_sum,
        diagnostics=diagnostics,
        correction=correction,
        token=token,
    )


def finish_cuda_gm1_rk3_apmasscon_split(
    reduction: CudaGm1Rk3ApMassConSplitReductionLaunch,
    *,
    stream: Any | None = None,
) -> CudaGm1Rk3ApMassConSplitFinishLaunch:
    """Enqueue only ``solve2.F:1972-2030`` on the original live stream."""

    reduction, prior = _require_reduction_token(reduction)
    if stream is None:
        stream = reduction.stream
    stream_pointer = _stream_pointer(stream)
    if stream_pointer != prior.get("stream_pointer"):
        raise ValueError("split finish belongs to another CUDA stream")
    pointers = _pointer_inventory(reduction.arrays)
    if _pointer_inventory_sha256(pointers) != prior.get(
        "pointer_inventory_sha256"
    ):
        raise ValueError("split finish device array bindings changed")
    domain = prior.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError("split reduction domain receipt is missing")
    ni = reduction.ni
    nj = reduction.nj
    nk = reduction.nk
    words = ni * nj * nk
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    constants = _default_constants_f32(np)
    compiled = reduction.compiled
    arrays = reduction.arrays
    compiled.kernels[1](
        (1, 1, 1),
        (1, 1, 1),
        (
            reduction.layer_dry_mass,
            reduction.layer_pressure_sum,
            reduction.diagnostics,
            reduction.correction,
            cupy.float64(reduction.mass1_f64),
            cupy.float32(constants["rddcv"]),
            cupy.uint64(words),
            cupy.uint32(nk),
        ),
        shared_mem=0,
        stream=stream,
    )
    apply_grid = (words + CUDA_APMASSCON_THREADS - 1) // CUDA_APMASSCON_THREADS
    compiled.kernels[2](
        (apply_grid, 1, 1),
        (CUDA_APMASSCON_THREADS, 1, 1),
        (
            arrays["pi0"],
            arrays["th0"],
            arrays["th3d"],
            arrays["qv"],
            arrays["pp3d"],
            arrays["prs"],
            arrays["rho"],
            reduction.correction,
            cupy.float32(constants["p00"]),
            cupy.float32(constants["rd"]),
            cupy.float32(constants["rv"]),
            cupy.float32(constants["cpdrd"]),
            cupy.uint64(words),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_STATUS,
        "completed_nrk": 3,
        "stream_pointer": stream_pointer,
        "event_pointer": int(event.ptr),
        "reduction_token_sha256": prior["operation_token_sha256"],
        "pointer_inventory_sha256": prior["pointer_inventory_sha256"],
        "source_order_phase": "solve2.F:1972-2030",
        "required_interposed_source_order": [
            "solve2.F:1662-1680 final momentum boundaries",
            "solve2.F:1708-1803 SGS-TKE RK3",
        ],
        "domain": dict(domain),
        "mass1_f64": prior["mass1_f64"],
        "selection": dict(_SELECTION_RECEIPT),
        "kernel": prior["kernel"],
        "runtime": prior["runtime"],
        "output_fields_mutated": ["pp3d", "prs", "rho"],
        "interposed_runtime_lineage_bound": False,
        "output_checkpoint_bound": False,
        "broader_rk3_mass_finish": False,
        "rr_rf_rebuilt": False,
        "mpi_performed": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchor": SOURCE_ANCHORS["finish_correction"],
        },
        "claim_limit": GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = Gm1Rk3ApMassConSplitFinishToken(MappingProxyType(receipt))
    return CudaGm1Rk3ApMassConSplitFinishLaunch(
        ready_event=event,
        reduction=reduction,
        diagnostics=reduction.diagnostics,
        correction=reduction.correction,
        token=token,
    )


__all__ = (
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_APMASSCON_SPLIT_FINISH_TOKEN_STATUS",
    "CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_APMASSCON_SPLIT_REDUCTION_TOKEN_STATUS",
    "GM1_RK3_APMASSCON_SPLIT_CLAIM_LIMIT",
    "CudaGm1Rk3ApMassConSplitFinishLaunch",
    "CudaGm1Rk3ApMassConSplitReductionLaunch",
    "Gm1Rk3ApMassConSplitFinishToken",
    "Gm1Rk3ApMassConSplitReductionToken",
    "begin_cuda_gm1_rk3_apmasscon_split",
    "finish_cuda_gm1_rk3_apmasscon_split",
    "require_gm1_rk3_apmasscon_split_reduction_token",
    "require_gm1_rk3_apmasscon_split_token_pair",
)
