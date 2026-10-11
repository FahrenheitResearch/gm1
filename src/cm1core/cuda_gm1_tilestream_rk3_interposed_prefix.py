"""Exact same-stream RK3 eqtset -> reduction -> momentum-boundary chain.

CM1 r21.1 orders the selected final-RK thermodynamic removal/EOS block at
``solve2.F:1561-1595``, begins ``apmasscon`` reductions at 1616-1656, then
applies final momentum boundaries at 1662-1680.  The individual local
operators intentionally cannot prove this chronology.  This coordinator
prevalidates every input and compiled image, then enqueues all three itself on
one CUDA stream.  Its token binds exact predecessor operation hashes, stream
object identity, and the shared thermodynamic carrier objects/pointers.

The prefix remains deliberately narrow.  It does not bind fresh RK3
scalar/moist or completed-RK2 scientific lineage, run SGS-TKE RK3, finish the
mass correction, checkpoint post-output bytes, complete RK3, or enter solve3.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from .cuda_apmasscon import (
    ALL_ARRAY_NAMES,
    CompiledApMassConKernels,
    compile_cuda_apmasscon,
)
from .cuda_gm1_rk1_post_scalar_momentum_boundaries import (
    CompiledGm1Rk1PostScalarMomentumBoundary,
    compile_cuda_gm1_rk1_post_scalar_momentum_boundary,
)
from .cuda_gm1_rk3_apmasscon_split import (
    CudaGm1Rk3ApMassConSplitReductionLaunch,
    begin_cuda_gm1_rk3_apmasscon_split,
    require_gm1_rk3_apmasscon_split_reduction_token,
)
from .cuda_gm1_tilestream_rk3_eqtset2_remove_eos import (
    CompiledGm1Rk3Eqset2RemoveEos,
    CudaGm1Rk3Eqset2RemoveEosLaunch,
    compile_cuda_gm1_rk3_eqtset2_remove_eos,
    launch_cuda_gm1_rk3_eqtset2_remove_eos_local,
    require_gm1_rk3_eqtset2_remove_eos_local_token,
)
from .cuda_gm1_tilestream_rk3_final_momentum_boundaries import (
    CudaGm1Rk3FinalMomentumBoundaryLocalLaunch,
    launch_cuda_gm1_rk3_final_momentum_boundaries_local,
    require_gm1_rk3_final_momentum_boundary_local_token,
    rk3_final_momentum_boundary_shapes,
)


CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA = (
    "gm1.cuda-rk3-interposed-prefix-token/v1"
)
CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS = (
    "ENQUEUED_EXACT_LOCAL_RK3_EQTSET_REDUCTION_BOUNDARY_CHAIN"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1561-1595",
    "src/solve2.F:1616-1656",
    "src/solve2.F:1662-1680",
)
EQTSET_ARRAY_ORDER = (
    "pi0",
    "th0",
    "pp3d",
    "th3d",
    "qv",
    "qc",
    "qpten",
    "qtten",
    "qvten",
    "qcten",
    "prs",
    "rho",
)
MASS_METRIC_ORDER = ("ruh", "rvh", "rmh")
SHARED_THERMODYNAMIC_ORDER = (
    "pi0",
    "th0",
    "th3d",
    "qv",
    "pp3d",
    "prs",
    "rho",
)
SOURCE_ORDER = (
    "solve2.F:1561-1595 eqtset2 remove/snapshot and moist EOS",
    "solve2.F:1616-1656 apmasscon layer reductions begin",
    "solve2.F:1662-1680 bcu/bcv/bcw and terrain bcwsfc",
)
GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT = (
    "one exact local CM1 r21.1 final-RK enqueue chain containing full-domain "
    "eqtset2 removal/moist EOS, apmasscon reduction begin, and final momentum "
    "boundaries on one CUDA stream with identical shared thermodynamic object "
    "pointers and explicit predecessor hashes. Fresh RK3 scalar/moist and "
    "completed-RK2 scientific lineage, SGS-TKE RK3, apmasscon finish, output "
    "hashes, completed RK3, solve3, histories, checkpoint, MPI, forecast, and "
    "five-metre product claims remain unavailable"
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
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_dimensions(
    *, nx: int, ny: int, nk: int, allow_compact_validation: bool
) -> None:
    for name, value in (("nx", nx), ("ny", ny), ("nk", nk)):
        _positive_int(name, value)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be bool")
    if allow_compact_validation:
        if min(nx, ny, nk) < 2:
            raise ValueError("compact RK3 prefix domain is too small")
    elif (nx, ny, nk) != (300, 300, 75):
        raise ValueError("live RK3 prefix domain must be exactly 300x300x75")


def _require_exact_device_inputs(
    eqtset_arrays: Mapping[str, Any],
    mass_metrics: Mapping[str, Any],
    momentum_boundary_arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    cupy = importlib.import_module("cupy")
    f32 = cupy.dtype("float32")
    physical_shape = (nk, ny, nx)
    if not isinstance(eqtset_arrays, Mapping) or tuple(eqtset_arrays) != (
        EQTSET_ARRAY_ORDER
    ):
        raise ValueError("RK3 prefix eqtset inventory/order changed")
    checked_eqtset: dict[str, Any] = {}
    for name in EQTSET_ARRAY_ORDER:
        value = eqtset_arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != f32
            or value.shape != physical_shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {physical_shape}")
        checked_eqtset[name] = value
    if not isinstance(mass_metrics, Mapping) or tuple(mass_metrics) != (
        MASS_METRIC_ORDER
    ):
        raise ValueError("RK3 prefix mass-metric inventory/order changed")
    expected_metric_shapes = {
        "ruh": (nx,),
        "rvh": (ny,),
        "rmh": physical_shape,
    }
    checked_metrics: dict[str, Any] = {}
    for name in MASS_METRIC_ORDER:
        value = mass_metrics[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != f32
            or value.shape != expected_metric_shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be device C-f32 {expected_metric_shapes[name]}"
            )
        checked_metrics[name] = value
    boundary_shapes = rk3_final_momentum_boundary_shapes(
        nx=nx,
        ny=ny,
        nk=nk,
    )
    if (
        not isinstance(momentum_boundary_arrays, Mapping)
        or tuple(momentum_boundary_arrays) != tuple(boundary_shapes)
    ):
        raise ValueError("RK3 prefix momentum-boundary inventory/order changed")
    checked_boundary: dict[str, Any] = {}
    for name, shape in boundary_shapes.items():
        value = momentum_boundary_arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != f32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {shape}")
        checked_boundary[name] = value
    all_values = (
        list(checked_eqtset.values())
        + list(checked_metrics.values())
        + list(checked_boundary.values())
    )
    pointers = [int(value.data.ptr) for value in all_values]
    if len(set(pointers)) != len(pointers):
        raise ValueError("RK3 prefix input arrays must not alias")
    return (
        MappingProxyType(checked_eqtset),
        MappingProxyType(checked_metrics),
        MappingProxyType(checked_boundary),
    )


def _pointer_sha(values: Mapping[str, Any]) -> str:
    pointers = {name: int(value.data.ptr) for name, value in values.items()}
    return hashlib.sha256(_canonical(pointers)).hexdigest()


@dataclass(frozen=True, slots=True)
class Gm1Rk3InterposedPrefixToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Rk3InterposedPrefixLaunch:
    ready_event: Any
    stream: Any
    eqtset: CudaGm1Rk3Eqset2RemoveEosLaunch
    reduction: CudaGm1Rk3ApMassConSplitReductionLaunch
    momentum_boundary: CudaGm1Rk3FinalMomentumBoundaryLocalLaunch
    token: Gm1Rk3InterposedPrefixToken


def launch_cuda_gm1_rk3_interposed_prefix(
    eqtset_arrays: Mapping[str, Any],
    mass_metrics: Mapping[str, Any],
    momentum_boundary_arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    mass1: float,
    dt_seconds: float,
    stream: Any | None = None,
    eqtset_compiled: CompiledGm1Rk3Eqset2RemoveEos | None = None,
    apmasscon_compiled: CompiledApMassConKernels | None = None,
    boundary_compiled: CompiledGm1Rk1PostScalarMomentumBoundary | None = None,
    allow_compact_validation: bool = False,
) -> CudaGm1Rk3InterposedPrefixLaunch:
    """Prevalidate and enqueue the exact three-operator prefix in one call.

    ``dt_seconds`` is required and is handed straight to the eqtset2 held-rate
    removal, which multiplies the four held diabatic rates by it
    (``solve2.F:1569-1576``).  It has no default on purpose: this coordinator
    used to omit the argument entirely and inherit the launcher's half second,
    so a caller integrating at any other step would have had this one stage
    remove a half second of held tendency out of a step that never happened.
    """

    _validate_dimensions(
        nx=nx,
        ny=ny,
        nk=nk,
        allow_compact_validation=allow_compact_validation,
    )
    if isinstance(mass1, bool) or not isinstance(mass1, (int, float)):
        raise TypeError("mass1 must be a finite positive number")
    if not math.isfinite(float(mass1)) or float(mass1) <= 0.0:
        raise ValueError("mass1 must be finite and > 0")
    eqtset_arrays, mass_metrics, momentum_boundary_arrays = (
        _require_exact_device_inputs(
            eqtset_arrays,
            mass_metrics,
            momentum_boundary_arrays,
            nx=nx,
            ny=ny,
            nk=nk,
        )
    )
    cupy = importlib.import_module("cupy")
    stream = stream or cupy.cuda.get_current_stream()
    stream_pointer = _positive_int("stream pointer", getattr(stream, "ptr", None))

    eqtset_compiled = (
        eqtset_compiled or compile_cuda_gm1_rk3_eqtset2_remove_eos()
    )
    apmasscon_compiled = apmasscon_compiled or compile_cuda_apmasscon()
    boundary_compiled = (
        boundary_compiled
        or compile_cuda_gm1_rk1_post_scalar_momentum_boundary()
    )
    if type(eqtset_compiled) is not CompiledGm1Rk3Eqset2RemoveEos:
        raise TypeError("eqtset_compiled exact type changed")
    if type(apmasscon_compiled) is not CompiledApMassConKernels:
        raise TypeError("apmasscon_compiled exact type changed")
    if type(boundary_compiled) is not CompiledGm1Rk1PostScalarMomentumBoundary:
        raise TypeError("boundary_compiled exact type changed")
    reduction_arrays = {
        name: (
            mass_metrics[name]
            if name in mass_metrics
            else eqtset_arrays[name]
        )
        for name in ALL_ARRAY_NAMES
    }
    eqtset = launch_cuda_gm1_rk3_eqtset2_remove_eos_local(
        eqtset_arrays,
        active_nx=nx,
        active_ny=ny,
        dt_seconds=dt_seconds,
        stream=stream,
        compiled=eqtset_compiled,
    )
    reduction = begin_cuda_gm1_rk3_apmasscon_split(
        reduction_arrays,
        ni=nx,
        nj=ny,
        nk=nk,
        mass1=float(mass1),
        stream=stream,
        compiled=apmasscon_compiled,
    )
    boundary = launch_cuda_gm1_rk3_final_momentum_boundaries_local(
        momentum_boundary_arrays,
        nx=nx,
        ny=ny,
        nk=nk,
        stream=stream,
        compiled=boundary_compiled,
        allow_compact_validation=allow_compact_validation,
    )
    if not (eqtset.stream is reduction.stream is boundary.stream is stream):
        raise RuntimeError("RK3 prefix lost exact CUDA stream object identity")
    for name in SHARED_THERMODYNAMIC_ORDER:
        if reduction.arrays[name] is not eqtset.arrays[name]:
            raise RuntimeError(f"RK3 prefix lost exact {name} object identity")

    eqtset_receipt = require_gm1_rk3_eqtset2_remove_eos_local_token(
        eqtset.token,
        stream_pointer=stream_pointer,
    )
    reduction_receipt = require_gm1_rk3_apmasscon_split_reduction_token(
        reduction.token,
        stream_pointer=stream_pointer,
    )
    boundary_receipt = require_gm1_rk3_final_momentum_boundary_local_token(
        boundary.token,
        stream_pointer=stream_pointer,
    )
    shared = {
        name: eqtset.arrays[name] for name in SHARED_THERMODYNAMIC_ORDER
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS,
        "nrk": 3,
        "nrkmax": 3,
        "dt_f32_bits": eqtset_receipt["dt_f32_bits"],
        "stream_pointer": stream_pointer,
        "domain": {"nx": nx, "ny": ny, "nk": nk},
        "source_order": list(SOURCE_ORDER),
        "component_operation_token_sha256": {
            "eqtset2_remove_eos": eqtset_receipt["operation_token_sha256"],
            "apmasscon_reduction": reduction_receipt[
                "operation_token_sha256"
            ],
            "momentum_boundary": boundary_receipt["operation_token_sha256"],
        },
        "predecessor_operation_token_sha256": {
            "apmasscon_reduction": eqtset_receipt["operation_token_sha256"],
            "momentum_boundary": reduction_receipt[
                "operation_token_sha256"
            ],
        },
        "shared_thermodynamic_carrier_order": list(
            SHARED_THERMODYNAMIC_ORDER
        ),
        "shared_thermodynamic_pointer_inventory_sha256": _pointer_sha(shared),
        "momentum_boundary_pointer_inventory_sha256": boundary_receipt[
            "pointer_inventory_sha256"
        ],
        "event_pointer_order": [
            int(eqtset.ready_event.ptr),
            int(reduction.ready_event.ptr),
            int(boundary.ready_event.ptr),
        ],
        "exact_stream_object_identity_bound": True,
        "shared_thermodynamic_carrier_object_identity_bound": True,
        "source_order_enqueue_chain_bound": True,
        "component_negative_claims_preserved": True,
        "fresh_rk3_scalar_moist_lineage_bound": False,
        "prior_completed_rk2_scientific_lineage_bound": False,
        "invalid_counter_host_checked": False,
        "rk3_tke_bound": False,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "history_checkpoint_emitted": False,
        "compact_validation_only": allow_compact_validation,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Rk3InterposedPrefixLaunch(
        boundary.ready_event,
        stream,
        eqtset,
        reduction,
        boundary,
        Gm1Rk3InterposedPrefixToken(MappingProxyType(receipt)),
    )


def require_gm1_rk3_interposed_prefix_token(
    token: object,
    *,
    stream_pointer: int | None = None,
) -> Mapping[str, object]:
    if type(token) is not Gm1Rk3InterposedPrefixToken:
        raise TypeError("token must be exact Gm1Rk3InterposedPrefixToken type")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("RK3 interposed prefix receipt must be a mapping")
    true_claims = (
        "exact_stream_object_identity_bound",
        "shared_thermodynamic_carrier_object_identity_bound",
        "source_order_enqueue_chain_bound",
        "component_negative_claims_preserved",
    )
    false_claims = (
        "fresh_rk3_scalar_moist_lineage_bound",
        "prior_completed_rk2_scientific_lineage_bound",
        "invalid_counter_host_checked",
        "rk3_tke_bound",
        "apmasscon_finish_bound",
        "post_output_sha256_bound",
        "mass_finish_source_order_complete",
        "completed_rk3_token_emitted",
        "solve3_entered",
        "history_checkpoint_emitted",
    )
    components = receipt.get("component_operation_token_sha256")
    predecessors = receipt.get("predecessor_operation_token_sha256")
    if (
        receipt.get("schema") != CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS
        or receipt.get("nrk") != 3
        or receipt.get("nrkmax") != 3
        or receipt.get("source_order") != list(SOURCE_ORDER)
        or receipt.get("shared_thermodynamic_carrier_order")
        != list(SHARED_THERMODYNAMIC_ORDER)
        or not isinstance(components, Mapping)
        or tuple(components)
        != ("eqtset2_remove_eos", "apmasscon_reduction", "momentum_boundary")
        or not isinstance(predecessors, Mapping)
        or tuple(predecessors) != ("apmasscon_reduction", "momentum_boundary")
        or predecessors.get("apmasscon_reduction")
        != components.get("eqtset2_remove_eos")
        or predecessors.get("momentum_boundary")
        != components.get("apmasscon_reduction")
        or any(receipt.get(name) is not True for name in true_claims)
        or any(receipt.get(name) is not False for name in false_claims)
        or not isinstance(receipt.get("compact_validation_only"), bool)
        or receipt.get("claim_limit") != GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 interposed prefix identity, lineage, or claim changed")
    actual_stream = _positive_int(
        "RK3 prefix stream_pointer",
        receipt.get("stream_pointer"),
    )
    if stream_pointer is not None and actual_stream != _positive_int(
        "stream_pointer", stream_pointer
    ):
        raise ValueError("RK3 interposed prefix belongs to another stream")
    for name, value in components.items():
        _lower_sha(f"component {name}", value)
    for name in (
        "shared_thermodynamic_pointer_inventory_sha256",
        "momentum_boundary_pointer_inventory_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(name, receipt.get(name))
    events = receipt.get("event_pointer_order")
    if (
        not isinstance(events, list)
        or len(events) != 3
        or any(_positive_int("event pointer", value) <= 0 for value in events)
    ):
        raise ValueError("RK3 interposed prefix event order changed")
    return MappingProxyType(dict(receipt))


def refuse_gm1_rk3_interposed_prefix_as_tke_or_completed_stage(
    token: object,
) -> NoReturn:
    receipt = require_gm1_rk3_interposed_prefix_token(token)
    raise RuntimeError(
        "GM1-RK3-INTERPOSED-PREFIX-NOT-TKE-OR-COMPLETION: exact prefix "
        + str(receipt["operation_token_sha256"])
        + " ends after final momentum boundaries; fresh RK3 TKE, apmasscon "
        "finish, output hashes, and completed RK3 remain unavailable"
    )


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA",
    "CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS",
    "CudaGm1Rk3InterposedPrefixLaunch",
    "EQTSET_ARRAY_ORDER",
    "GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT",
    "Gm1Rk3InterposedPrefixToken",
    "MASS_METRIC_ORDER",
    "SHARED_THERMODYNAMIC_ORDER",
    "SOURCE_ORDER",
    "launch_cuda_gm1_rk3_interposed_prefix",
    "refuse_gm1_rk3_interposed_prefix_as_tke_or_completed_stage",
    "require_gm1_rk3_interposed_prefix_token",
)
