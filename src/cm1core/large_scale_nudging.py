"""Source-exact bounded CM1 r21.1 large-scale nudging operator.

The operator owns the solve1 schedule/ramp and u/v/theta/qv tendency blocks
plus the solve3 u/v reference-profile update on supplied, already-interpolated
target profiles.  Reading/interpolating external ``lsnudge_XXXX.dat`` files,
terrain, MPI, diagnostics, and coupled solve1/solve3 execution remain outside
this bounded CPU-only claim.
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


CM1_LARGE_SCALE_NUDGING_SCHEMA = "gpuwm-cm1.large-scale-nudging/v1"
CM1_LARGE_SCALE_NUDGING_STATUS = (
    "ADMITTED_CM1_R21_1_LARGE_SCALE_NUDGING_OPERATOR_ONLY"
)
CM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT = (
    "CM1 r21.1 solve1.F:252-288,1608-1693 and solve3.F:357-408 compact "
    "source-order schedule, u/v/theta/qv tendencies, and u/v base-profile "
    "update on supplied interpolated targets only; no lsnudge_XXXX.dat I/O "
    "or interpolation, terrain, MPI, diagnostics, complete solve1/solve3, "
    "production configuration, trajectory, forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_LARGE_SCALE_NUDGING_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/lsnudge.F": (
            "be1766baf4df3865bc6ea52bb029c5b4027fd838bdd2cf3e81dcc93700cba0cd"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
        "src/solve3.F": (
            "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
        ),
    }
)
CM1_LARGE_SCALE_NUDGING_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1863-1904",
        "external_profile_reader": "src/lsnudge.F:41-432 (excluded; supplied profiles)",
        "schedule_and_ramp": "src/solve1.F:252-288",
        "tendency_application": "src/solve1.F:1608-1693",
        "base_profile_update": "src/solve3.F:357-408",
        "native_compatibility": "src/param.F:1136-1148",
    }
)


@dataclass(frozen=True, slots=True)
class LargeScaleNudgingSelection:
    """Exact selectors, cadence, and supplied active-profile index."""

    do_lsnudge: bool
    do_lsnudge_u: bool
    do_lsnudge_v: bool
    do_lsnudge_th: bool
    do_lsnudge_qv: bool
    imoist: int
    profile_index: int
    mtime_s: float
    dt_s: float
    lsnudge_start_s: float
    lsnudge_end_s: float
    lsnudge_ramp_time_s: float
    lsnudge_tau_s: float
    umove_m_s: float
    vmove_m_s: float
    cgs1: float
    cgs2: float
    cgs3: float
    cgt1: float
    cgt2: float
    cgt3: float


@dataclass(frozen=True, slots=True)
class LargeScaleNudgingResult:
    """Completed in-place nudging update, schedule state, and receipt."""

    arrays: Mapping[str, Any]
    profile_index: int
    timenudge_s: float
    lsnudgefac: float
    donudge: bool
    receipt: Mapping[str, object]


_PROFILE_ARRAYS = (
    "lsnudge_time1",
    "lsnudge_time2",
    "lsnudge_u",
    "lsnudge_v",
    "lsnudge_th",
    "lsnudge_qv",
)
_AVERAGE_ARRAYS = ("uavg", "vavg", "thavg", "qavg")
_OUTPUT_ARRAYS = (
    "uten",
    "vten",
    "thten",
    "qvten",
    "ug",
    "vg",
    "u0",
    "v0",
)


def large_scale_nudging_shapes(
    ni: int, nj: int, nk: int, num_profiles: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (
        ("ni", ni),
        ("nj", nj),
        ("nk", nk),
        ("num_profiles", num_profiles),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if nk < 3:
        raise ValueError("nk must be at least 3 for source boundary extrapolation")
    shapes: dict[str, tuple[int, ...]] = {
        "lsnudge_time1": (num_profiles,),
        "lsnudge_time2": (num_profiles,),
        "uten": (nk, nj, ni + 1),
        "vten": (nk, nj + 1, ni),
        "thten": (nk, nj, ni),
        "qvten": (nk, nj, ni),
        "ug": (nk,),
        "vg": (nk,),
        "u0": (nk + 2, nj, ni + 1),
        "v0": (nk + 2, nj + 1, ni),
    }
    shapes.update({name: (num_profiles, nk) for name in _PROFILE_ARRAYS[2:]})
    shapes.update({name: (nk,) for name in _AVERAGE_ARRAYS})
    return shapes


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(
    np: Any,
    selection: LargeScaleNudgingSelection,
    *,
    num_profiles: int,
) -> dict[str, Any]:
    if not isinstance(selection, LargeScaleNudgingSelection):
        raise TypeError("selection must be a LargeScaleNudgingSelection")
    for name in (
        "do_lsnudge",
        "do_lsnudge_u",
        "do_lsnudge_v",
        "do_lsnudge_th",
        "do_lsnudge_qv",
    ):
        if not isinstance(getattr(selection, name), bool):
            raise TypeError(f"{name} must be a bool")
    if isinstance(selection.imoist, bool) or not isinstance(selection.imoist, int):
        raise TypeError("imoist must be an integer")
    if selection.imoist not in (0, 1):
        refuse(
            code="CM1-LARGE-SCALE-NUDGING-SELECTION",
            gate="GM1-PHYSICS-SUPPORT",
            option="imoist",
            value=selection.imoist,
            reason="the bounded qv branch admits the native dry/moist 0/1 selector",
            remedy="select imoist=0 or 1",
        )
    if isinstance(selection.profile_index, bool) or not isinstance(
        selection.profile_index, int
    ):
        raise TypeError("profile_index must be an integer")
    if not 0 <= selection.profile_index < num_profiles:
        raise ValueError("profile_index is outside supplied target profiles")
    scalar = {
        name: _f32(np, name, getattr(selection, name))
        for name in (
            "mtime_s",
            "dt_s",
            "lsnudge_start_s",
            "lsnudge_end_s",
            "lsnudge_ramp_time_s",
            "lsnudge_tau_s",
            "umove_m_s",
            "vmove_m_s",
            "cgs1",
            "cgs2",
            "cgs3",
            "cgt1",
            "cgt2",
            "cgt3",
        )
    }
    if scalar["dt_s"] <= np.float32(0.0):
        raise ValueError("dt_s must be strictly positive")
    if scalar["lsnudge_tau_s"] <= np.float32(0.0):
        raise ValueError("lsnudge_tau_s must be strictly positive")
    if scalar["lsnudge_end_s"] <= scalar["lsnudge_start_s"]:
        raise ValueError("lsnudge_end_s must exceed lsnudge_start_s")
    if scalar["lsnudge_ramp_time_s"] < np.float32(0.0):
        raise ValueError("lsnudge_ramp_time_s must be nonnegative")
    if scalar["lsnudge_ramp_time_s"] > _sub(
        np, scalar["lsnudge_end_s"], scalar["lsnudge_start_s"]
    ):
        refuse(
            code="CM1-LARGE-SCALE-NUDGING-RAMP",
            gate="GM1-PHYSICS-SUPPORT",
            option="lsnudge_ramp_time",
            value=float(scalar["lsnudge_ramp_time_s"]),
            reason="lsnudge.F refuses a ramp longer than the active window",
            remedy="use lsnudge_ramp_time <= lsnudge_end-lsnudge_start",
        )
    return scalar


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    num_profiles: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = large_scale_nudging_shapes(ni, nj, nk, num_profiles)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing large-scale-nudging arrays: " + ", ".join(missing))
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
    for name in _OUTPUT_ARRAYS:
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    if np.any(arrays["lsnudge_time2"] < arrays["lsnudge_time1"]):
        raise ValueError("every lsnudge_time2 must be >= lsnudge_time1")
    if num_profiles > 1:
        if np.any(arrays["lsnudge_time1"][1:] < arrays["lsnudge_time1"][:-1]):
            raise ValueError("lsnudge_time1 must be nondecreasing")
        if np.any(arrays["lsnudge_time2"][1:] < arrays["lsnudge_time2"][:-1]):
            raise ValueError("lsnudge_time2 must be nondecreasing")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "large-scale-nudging arrays must not alias: "
                    f"{left_name}, {right_name}"
                )


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _nudging_tendency(
    np: Any, factor: Any, current: Any, target: Any, tau: Any
) -> Any:
    return _div(np, _mul(np, np.float32(-1.0), _mul(np, factor, _sub(np, current, target))), tau)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_large_scale_nudging_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    num_profiles: int,
    selection: LargeScaleNudgingSelection,
) -> LargeScaleNudgingResult:
    """Execute the bounded combined solve1/solve3 nudging blocks in-place."""

    np = importlib.import_module("numpy")
    scalar = _validate_selection(np, selection, num_profiles=num_profiles)
    _validate_arrays(
        np,
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        num_profiles=num_profiles,
    )

    timenudge = _add(np, scalar["mtime_s"], scalar["dt_s"])
    profile_index = selection.profile_index
    donudge = False
    factor = np.float32(0.0)
    if selection.do_lsnudge:
        if (
            timenudge > scalar["lsnudge_start_s"]
            and timenudge < scalar["lsnudge_end_s"]
        ):
            donudge = True
            factor = np.float32(1.0)
            if timenudge > arrays["lsnudge_time2"][profile_index]:
                if profile_index + 1 >= num_profiles:
                    refuse(
                        code="CM1-LARGE-SCALE-NUDGING-PROFILE-EXHAUSTED",
                        gate="GM1-PHYSICS-SUPPORT",
                        option="profile_index",
                        value=profile_index,
                        reason="source cadence requests a later supplied target profile",
                        remedy="supply the next authenticated nudging profile",
                    )
                profile_index += 1
            if scalar["lsnudge_ramp_time_s"] < np.float32(1.0e-30):
                factor = np.float32(1.0)
            else:
                factor = _div(
                    np,
                    _sub(np, timenudge, arrays["lsnudge_time1"][profile_index]),
                    scalar["lsnudge_ramp_time_s"],
                )
                factor = max(factor, np.float32(0.0))
                factor = min(factor, np.float32(1.0))

    apply_u = donudge and selection.do_lsnudge_u
    apply_v = donudge and selection.do_lsnudge_v
    apply_th = donudge and selection.do_lsnudge_th
    apply_qv = donudge and selection.do_lsnudge_qv and selection.imoist == 1
    tau = scalar["lsnudge_tau_s"]
    if apply_u:
        for k in range(nk):
            current = _add(np, scalar["umove_m_s"], arrays["uavg"][k])
            tendency = _nudging_tendency(
                np, factor, current, arrays["lsnudge_u"][profile_index, k], tau
            )
            arrays["uten"][k] += tendency
    if apply_v:
        for k in range(nk):
            current = _add(np, scalar["vmove_m_s"], arrays["vavg"][k])
            tendency = _nudging_tendency(
                np, factor, current, arrays["lsnudge_v"][profile_index, k], tau
            )
            arrays["vten"][k] += tendency
    if apply_th:
        for k in range(nk):
            tendency = _nudging_tendency(
                np,
                factor,
                arrays["thavg"][k],
                arrays["lsnudge_th"][profile_index, k],
                tau,
            )
            arrays["thten"][k] += tendency
    if apply_qv:
        for k in range(nk):
            tendency = _nudging_tendency(
                np,
                factor,
                arrays["qavg"][k],
                arrays["lsnudge_qv"][profile_index, k],
                tau,
            )
            arrays["qvten"][k] += tendency

    base_fraction = _div(np, _mul(np, factor, scalar["dt_s"]), tau)
    if apply_u:
        for k in range(nk):
            target = arrays["lsnudge_u"][profile_index, k]
            arrays["ug"][k] = _sub(
                np,
                arrays["ug"][k],
                _mul(
                    np,
                    base_fraction,
                    _sub(
                        np,
                        _add(np, scalar["umove_m_s"], arrays["ug"][k]),
                        target,
                    ),
                ),
            )
            reference = arrays["u0"][k + 1, 0, 0]
            reference = _sub(
                np,
                reference,
                _mul(
                    np,
                    base_fraction,
                    _sub(np, _add(np, scalar["umove_m_s"], reference), target),
                ),
            )
            arrays["u0"][k + 1].fill(reference)
        bottom = _add(
            np,
            _add(
                np,
                _mul(np, scalar["cgs1"], arrays["u0"][1, 0, 0]),
                _mul(np, scalar["cgs2"], arrays["u0"][2, 0, 0]),
            ),
            _mul(np, scalar["cgs3"], arrays["u0"][3, 0, 0]),
        )
        top = _add(
            np,
            _add(
                np,
                _mul(np, scalar["cgt1"], arrays["u0"][nk, 0, 0]),
                _mul(np, scalar["cgt2"], arrays["u0"][nk - 1, 0, 0]),
            ),
            _mul(np, scalar["cgt3"], arrays["u0"][nk - 2, 0, 0]),
        )
        arrays["u0"][0].fill(bottom)
        arrays["u0"][nk + 1].fill(top)
    if apply_v:
        for k in range(nk):
            target = arrays["lsnudge_v"][profile_index, k]
            arrays["vg"][k] = _sub(
                np,
                arrays["vg"][k],
                _mul(
                    np,
                    base_fraction,
                    _sub(
                        np,
                        _add(np, scalar["vmove_m_s"], arrays["vg"][k]),
                        target,
                    ),
                ),
            )
            reference = arrays["v0"][k + 1, 0, 0]
            reference = _sub(
                np,
                reference,
                _mul(
                    np,
                    base_fraction,
                    _sub(np, _add(np, scalar["vmove_m_s"], reference), target),
                ),
            )
            arrays["v0"][k + 1].fill(reference)
        bottom = _add(
            np,
            _add(
                np,
                _mul(np, scalar["cgs1"], arrays["v0"][1, 0, 0]),
                _mul(np, scalar["cgs2"], arrays["v0"][2, 0, 0]),
            ),
            _mul(np, scalar["cgs3"], arrays["v0"][3, 0, 0]),
        )
        top = _add(
            np,
            _add(
                np,
                _mul(np, scalar["cgt1"], arrays["v0"][nk, 0, 0]),
                _mul(np, scalar["cgt2"], arrays["v0"][nk - 1, 0, 0]),
            ),
            _mul(np, scalar["cgt3"], arrays["v0"][nk - 2, 0, 0]),
        )
        arrays["v0"][0].fill(bottom)
        arrays["v0"][nk + 1].fill(top)

    for name in _OUTPUT_ARRAYS:
        if not np.all(np.isfinite(arrays[name])):
            refuse(
                code="CM1-LARGE-SCALE-NUDGING-NONFINITE",
                gate="GM1-PHYSICS-SUPPORT",
                option="large_scale_nudging_output",
                value=name,
                reason="source-order nudging produced a non-finite carrier",
                remedy="inspect supplied averages, targets, cadence, and base state",
            )

    receipt: dict[str, object] = {
        "schema": CM1_LARGE_SCALE_NUDGING_SCHEMA,
        "status": CM1_LARGE_SCALE_NUDGING_STATUS,
        "claim_limit": CM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_LARGE_SCALE_NUDGING_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_LARGE_SCALE_NUDGING_SOURCE_ANCHORS),
        },
        "selection": {
            "do_lsnudge": selection.do_lsnudge,
            "do_lsnudge_u": selection.do_lsnudge_u,
            "do_lsnudge_v": selection.do_lsnudge_v,
            "do_lsnudge_th": selection.do_lsnudge_th,
            "do_lsnudge_qv": selection.do_lsnudge_qv,
            "imoist": selection.imoist,
            "initial_profile_index": selection.profile_index,
            "final_profile_index": profile_index,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
            },
        },
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "num_profiles": num_profiles,
        },
        "schedule": {
            "timenudge_raw_bits": int(np.asarray(timenudge).view(np.uint32)),
            "lsnudgefac_raw_bits": int(np.asarray(factor).view(np.uint32)),
            "donudge": donudge,
        },
        "applied": {"u": apply_u, "v": apply_v, "th": apply_th, "qv": apply_qv},
        "output_sha256": {name: _sha256_array(arrays[name]) for name in _OUTPUT_ARRAYS},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return LargeScaleNudgingResult(
        arrays=MappingProxyType({name: arrays[name] for name in _OUTPUT_ARRAYS}),
        profile_index=profile_index,
        timenudge_s=float(timenudge),
        lsnudgefac=float(factor),
        donudge=donudge,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_LARGE_SCALE_NUDGING_AUTHORITY_SHA256",
    "CM1_LARGE_SCALE_NUDGING_CLAIM_LIMIT",
    "CM1_LARGE_SCALE_NUDGING_SCHEMA",
    "CM1_LARGE_SCALE_NUDGING_SOURCE_ANCHORS",
    "CM1_LARGE_SCALE_NUDGING_STATUS",
    "LargeScaleNudgingResult",
    "LargeScaleNudgingSelection",
    "cpu_large_scale_nudging_inplace",
    "large_scale_nudging_shapes",
]
