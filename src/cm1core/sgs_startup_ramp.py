"""Source-exact CM1 r21.1 SGS startup-ramp operator.

This bounded CPU operator owns the complete localized ``ramp_sgs=1`` block
from ``turb.F``: source normalization of ``ramp_time``, the pre/in/post-ramp
clamps, the ``cm0 > cmemin`` mask, and the source-ordered updates of ``cme``,
``ce1``, ``ce2``, and ``csm``.  Coupled turbulence execution remains excluded.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_SGS_STARTUP_RAMP_SCHEMA = "gpuwm-cm1.sgs-startup-ramp/v1"
CM1_SGS_STARTUP_RAMP_STATUS = "ADMITTED_CM1_R21_1_SGS_STARTUP_RAMP_OPERATOR_ONLY"
CM1_SGS_STARTUP_RAMP_CLAIM_LIMIT = (
    "CM1 r21.1 ramp_sgs=1 source-normalized time factor and cme/ce1/ce2/csm "
    "update on supplied compact LES carriers only; no SGS-model dispatch, "
    "turbulence tendencies, surface fluxes, coupled large step, production "
    "configuration, trajectory, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_SGS_STARTUP_RAMP_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/constants.F": (
            "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/turb.F": (
            "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d"
        ),
    }
)
CM1_SGS_STARTUP_RAMP_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1081-1087",
        "constants": "src/constants.F:14,22,49-52",
        "ramp_time_normalization": "src/param.F:750",
        "operator": "src/turb.F:233-262",
    }
)


@dataclass(frozen=True, slots=True)
class SgsStartupRampSelection:
    """Explicit outer-dispatch and time selection for the bounded operator."""

    ramp_sgs: int
    idoles: bool
    sgsmodel: int
    rtime_seconds: float
    ramp_time_seconds: float


@dataclass(frozen=True, slots=True)
class SgsStartupRampResult:
    """Completed source-order update and immutable receipt."""

    turbfac: float
    receipt: Mapping[str, object]


def sgs_startup_ramp_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "cm0": (nj, ni),
        "cme": (nk + 1, nj, ni),
        "ce1": (nk + 1, nj, ni),
        "ce2": (nk + 1, nj, ni),
        "csm": (nk + 1, nj, ni),
    }


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(np: Any, selection: SgsStartupRampSelection) -> tuple[Any, Any]:
    if not isinstance(selection, SgsStartupRampSelection):
        raise TypeError("selection must be an SgsStartupRampSelection")
    for name in ("ramp_sgs", "sgsmodel"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.ramp_sgs != 1:
        refuse(
            code="CM1-SGS-STARTUP-RAMP-SELECTION",
            gate="GM1-PHYSICS-SUPPORT",
            option="ramp_sgs",
            value=selection.ramp_sgs,
            reason="the active operator owns exactly ramp_sgs=1",
            remedy="use the ramp_sgs=0 identity path or ramp_sgs=1",
        )
    if not isinstance(selection.idoles, bool):
        raise TypeError("idoles must be bool")
    if not selection.idoles or selection.sgsmodel not in range(1, 7):
        refuse(
            code="CM1-SGS-STARTUP-RAMP-OUTER-DISPATCH",
            gate="GM1-PHYSICS-SUPPORT",
            option="idoles/sgsmodel",
            value=(selection.idoles, selection.sgsmodel),
            reason="turb.F enters the ramp only for idoles and sgsmodel>=1",
            remedy="use idoles=true with a source-defined sgsmodel in 1..6",
        )
    rtime = _f32(np, "rtime_seconds", selection.rtime_seconds)
    requested_ramp_time = _f32(
        np, "ramp_time_seconds", selection.ramp_time_seconds
    )
    effective_ramp_time = np.maximum(requested_ramp_time, np.float32(1.0e-10))
    return rtime, effective_ramp_time


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = sgs_startup_ramp_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing SGS startup-ramp arrays: {', '.join(missing)}")
    values: list[tuple[str, Any]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must have native float32 dtype")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        values.append((name, value))
    for name in ("cme", "ce1", "ce2", "csm"):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    f"SGS startup-ramp arrays must not alias: {left_name}, {right_name}"
                )


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_sgs_startup_ramp_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: SgsStartupRampSelection,
) -> SgsStartupRampResult:
    """Apply the bounded ``ramp_sgs=1`` coefficient update in source order."""

    np = importlib.import_module("numpy")
    rtime, ramp_time = _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)

    one = np.float32(1.0)
    zero = np.float32(0.0)
    turbfac = np.float32(np.float32(rtime - zero) / ramp_time)
    turbfac = np.maximum(zero, turbfac)
    turbfac = np.minimum(one, turbfac)

    cmemin = np.float32(1.0e-4)
    cme_floor = np.float32(0.001)
    c_l = np.float32(0.8165)
    ri_c = np.float32(0.25)
    pi = np.float32("3.1415926535897932384626433")
    reciprocal_ric_minus_one = np.float32(np.float32(one / ri_c) - one)
    active_columns = 0
    for k in range(nk + 1):
        for j in range(nj):
            for i in range(ni):
                cm0 = arrays["cm0"][j, i]
                if cm0 > cmemin:
                    if k == 0:
                        active_columns += 1
                    cme = np.float32(turbfac * cm0)
                    cme = np.maximum(cme_floor, cme)
                    arrays["cme"][k, j, i] = cme

                    ce1 = np.float32(cme * c_l)
                    ce1 = np.float32(ce1 * c_l)
                    ce1 = np.float32(ce1 * reciprocal_ric_minus_one)
                    arrays["ce1"][k, j, i] = ce1

                    ce2 = np.float32(cme * pi)
                    ce2 = np.float32(ce2 * pi)
                    ce2 = np.float32(ce2 - ce1)
                    ce2 = np.maximum(zero, ce2)
                    arrays["ce2"][k, j, i] = ce2

                    numerator = np.float32(cme * cme)
                    numerator = np.float32(numerator * cme)
                    denominator = np.float32(ce1 + ce2)
                    quotient = np.float32(numerator / denominator)
                    arrays["csm"][k, j, i] = np.float32(
                        quotient ** np.float32(0.25)
                    )

    output_names = ("cme", "ce1", "ce2", "csm")
    receipt: dict[str, object] = {
        "schema": CM1_SGS_STARTUP_RAMP_SCHEMA,
        "status": CM1_SGS_STARTUP_RAMP_STATUS,
        "claim_limit": CM1_SGS_STARTUP_RAMP_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_SGS_STARTUP_RAMP_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_SGS_STARTUP_RAMP_SOURCE_ANCHORS),
        },
        "selection": {
            "ramp_sgs": selection.ramp_sgs,
            "idoles": selection.idoles,
            "sgsmodel": selection.sgsmodel,
            "rtime_seconds_raw_bits": int(np.asarray(rtime).view(np.uint32)),
            "requested_ramp_time_seconds_raw_bits": int(
                np.asarray(np.float32(selection.ramp_time_seconds)).view(np.uint32)
            ),
            "effective_ramp_time_seconds_raw_bits": int(
                np.asarray(ramp_time).view(np.uint32)
            ),
            "turbfac_raw_bits": int(np.asarray(turbfac).view(np.uint32)),
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "active_cm0_columns": active_columns,
        "output_sha256": {name: _sha256_array(arrays[name]) for name in output_names},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return SgsStartupRampResult(
        turbfac=float(turbfac), receipt=MappingProxyType(receipt)
    )


__all__ = [
    "CM1_SGS_STARTUP_RAMP_AUTHORITY_SHA256",
    "CM1_SGS_STARTUP_RAMP_CLAIM_LIMIT",
    "CM1_SGS_STARTUP_RAMP_SCHEMA",
    "CM1_SGS_STARTUP_RAMP_SOURCE_ANCHORS",
    "CM1_SGS_STARTUP_RAMP_STATUS",
    "SgsStartupRampResult",
    "SgsStartupRampSelection",
    "cpu_sgs_startup_ramp_inplace",
    "sgs_startup_ramp_shapes",
]
