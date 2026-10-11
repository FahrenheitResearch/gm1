"""Source-exact bounded CM1 r21.1 neutral surface-exchange operator.

This CPU operator owns the complete branch structure of ``getcecd`` in
``sfcphys.F:18-267`` on supplied compact carriers: prescribed roughness,
prescribed friction velocity, constant/Deacon/Fairall-Donelan drag, land and
water paths, optional heat/moisture coefficients, and the source's averaged
surface disposition.  It deliberately does not couple those coefficients to
the surrounding turbulence or surface-model drivers.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_SURFACE_EXCHANGE_SCHEMA = "gpuwm-cm1.surface-exchange-coefficients/v1"
CM1_SURFACE_EXCHANGE_STATUS = (
    "ADMITTED_CM1_R21_1_SURFACE_EXCHANGE_COEFFICIENT_OPERATOR_ONLY"
)
CM1_SURFACE_EXCHANGE_CLAIM_LIMIT = (
    "CM1 r21.1 sfcphys.F:18-267 getcecd compact source-order operator for "
    "set_znt=0/1, set_ust=0/1, cecd=1/2/3, land/water, isfcflx=0/1, "
    "and use_avg_sfc false/true only; no turbulence tendencies, surface-flux "
    "application, external surface model, MPI, coupled step, production "
    "configuration, trajectory, forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_SURFACE_EXCHANGE_AUTHORITY_SHA256 = MappingProxyType(
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
        "src/sfcphys.F": (
            "7cb54de579b8bb038285430890e075ab19ed2e6d93158bd50d656209d64696a7"
        ),
    }
)
CM1_SURFACE_EXCHANGE_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1019-1031,1072-1080",
        "selection_validation": "src/param.F:1324-1360",
        "averaged_surface_derivation": "src/param.F:3079-3094",
        "prescribed_roughness": "src/sfcphys.F:83-99",
        "prescribed_friction_velocity": "src/sfcphys.F:102-118",
        "constant_drag": "src/sfcphys.F:120-136",
        "land_water_drag": "src/sfcphys.F:138-195",
        "enthalpy_exchange": "src/sfcphys.F:197-236",
        "averaged_surface_broadcast": "src/sfcphys.F:238-259",
    }
)


@dataclass(frozen=True, slots=True)
class SurfaceExchangeSelection:
    """Exact outer selectors and scalar inputs for one ``getcecd`` call."""

    set_znt: int
    set_ust: int
    cecd: int
    isfcflx: int
    use_avg_sfc: bool
    cnst_ust_m_s: float
    cnstcd: float
    cnstce: float
    avgsfcu_m_s: float
    avgsfcv_m_s: float
    avgsfcs_m_s: float


@dataclass(frozen=True, slots=True)
class SurfaceExchangeResult:
    """Completed in-place coefficient update and immutable receipt."""

    arrays: Mapping[str, Any]
    max_water_iterations: int
    receipt: Mapping[str, object]


_ARRAY_NAMES = (
    "u1",
    "v1",
    "s1",
    "za",
    "u10",
    "v10",
    "s10",
    "xland",
    "znt",
    "ust",
    "cd",
    "ch",
    "cq",
)
_OUTPUT_NAMES = ("u10", "v10", "s10", "znt", "ust", "cd", "ch", "cq")


def surface_exchange_shapes(ni: int, nj: int) -> dict[str, tuple[int, int]]:
    for name, value in (("ni", ni), ("nj", nj)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {name: (nj, ni) for name in _ARRAY_NAMES}


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(np: Any, selection: SurfaceExchangeSelection) -> dict[str, Any]:
    if not isinstance(selection, SurfaceExchangeSelection):
        raise TypeError("selection must be a SurfaceExchangeSelection")
    for name in ("set_znt", "set_ust", "cecd", "isfcflx"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    for name in ("set_znt", "set_ust", "isfcflx"):
        if getattr(selection, name) not in (0, 1):
            refuse(
                code="CM1-SURFACE-EXCHANGE-SELECTION",
                gate="GM1-PHYSICS-SUPPORT",
                option=name,
                value=getattr(selection, name),
                reason="the source dispatch admits only integer 0/1",
                remedy=f"select {name}=0 or 1",
            )
    if selection.cecd not in (1, 2, 3):
        refuse(
            code="CM1-SURFACE-EXCHANGE-SELECTION",
            gate="GM1-PHYSICS-SUPPORT",
            option="cecd",
            value=selection.cecd,
            reason="the documented source dispatch contains cecd=1,2,3 only",
            remedy="select cecd=1, 2, or 3",
        )
    if not isinstance(selection.use_avg_sfc, bool):
        raise TypeError("use_avg_sfc must be a bool")
    scalar = {
        name: _f32(np, name, getattr(selection, name))
        for name in (
            "cnst_ust_m_s",
            "cnstcd",
            "cnstce",
            "avgsfcu_m_s",
            "avgsfcv_m_s",
            "avgsfcs_m_s",
        )
    }
    if selection.set_ust == 1 and scalar["cnst_ust_m_s"] <= np.float32(0.0):
        raise ValueError("cnst_ust_m_s must be positive when set_ust=1")
    return scalar


def _validate_arrays(
    np: Any, arrays: MutableMapping[str, Any], *, ni: int, nj: int
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = surface_exchange_shapes(ni, nj)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing surface-exchange arrays: " + ", ".join(missing))
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
    for name in _OUTPUT_NAMES:
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    if np.any(arrays["za"] <= np.float32(0.0)):
        raise ValueError("za must be strictly positive")
    if np.any(arrays["znt"] <= np.float32(0.0)):
        raise ValueError("znt must be strictly positive")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "surface-exchange arrays must not alias: "
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


@lru_cache(maxsize=1)
def _libm_f32() -> tuple[object, object, object, str]:
    import ctypes
    import ctypes.util


    if os.name == "nt":
        library_name: str | None = "ucrtbase"
        backend = "ucrtbase"
    else:
        library_name = ctypes.util.find_library("m")
        backend = library_name or "process-libc"
    library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
    functions = []
    for name in ("logf", "expf", "sqrtf"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
        functions.append(function)
    return functions[0], functions[1], functions[2], backend


def _call_libm_f32(np: Any, index: int, value: Any) -> Any:
    import ctypes

    function = _libm_f32()[index]
    return np.float32(function(ctypes.c_float(float(np.float32(value)))))


def _log(np: Any, value: Any) -> Any:
    return _call_libm_f32(np, 0, value)


def _exp(np: Any, value: Any) -> Any:
    return _call_libm_f32(np, 1, value)


def _sqrt(np: Any, value: Any) -> Any:
    return _call_libm_f32(np, 2, value)


def _surface_ratio(np: Any, znt: Any, za: Any) -> tuple[Any, Any, Any]:
    rznt = _div(np, np.float32(1.0), znt)
    numerator = _log(np, _mul(np, _add(np, np.float32(10.0), znt), rznt))
    denominator = _log(np, _mul(np, _add(np, za, znt), rznt))
    return rznt, _div(np, numerator, denominator), denominator


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_surface_exchange_coefficients_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    selection: SurfaceExchangeSelection,
) -> SurfaceExchangeResult:
    """Execute the bounded ``getcecd`` source block in-place."""

    np = importlib.import_module("numpy")
    scalar = _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj)

    karman = np.float32(0.40)
    dcd1 = np.float32(1.0e-3)
    dcd2 = np.float32(2.4e-3)
    dwsp1 = np.float32(5.0)
    dwsp2 = np.float32(25.0)
    dfac = _div(np, _sub(np, dcd2, dcd1), _sub(np, dwsp2, dwsp1))

    i2 = ni
    j2 = nj
    saved: tuple[Any, Any, Any] | None = None
    if selection.use_avg_sfc:
        i2 = 1
        j2 = 1
        saved = (arrays["u1"][0, 0], arrays["v1"][0, 0], arrays["s1"][0, 0])
        arrays["u1"][0, 0] = scalar["avgsfcu_m_s"]
        arrays["v1"][0, 0] = scalar["avgsfcv_m_s"]
        arrays["s1"][0, 0] = scalar["avgsfcs_m_s"]

    max_iterations = 0
    for j in range(j2):
        for i in range(i2):
            za = arrays["za"][j, i]
            s1 = arrays["s1"][j, i]
            znt = arrays["znt"][j, i]
            if selection.set_znt == 1:
                rznt = _div(np, np.float32(1.0), znt)
                log10 = _log(np, _mul(np, np.float32(10.0), rznt))
                cdroot = _div(np, karman, log10)
                arrays["cd"][j, i] = _mul(np, cdroot, cdroot)
                logza = _log(np, _mul(np, za, rznt))
                var = _div(np, log10, logza)
                arrays["ust"][j, i] = max(
                    _div(np, _mul(np, s1, karman), logza), np.float32(1.0e-6)
                )
            elif selection.set_ust == 1:
                arrays["ust"][j, i] = scalar["cnst_ust_m_s"]
                speed = max(np.float32(1.0e-10), s1)
                friction = max(np.float32(1.0e-10), arrays["ust"][j, i])
                exponent = _div(np, _mul(np, speed, karman), friction)
                denominator = _sub(np, _exp(np, exponent), np.float32(1.0))
                arrays["znt"][j, i] = _div(np, za, denominator)
                znt = arrays["znt"][j, i]
                _, var, _ = _surface_ratio(np, znt, za)
                rznt = _div(np, np.float32(1.0), znt)
                log10 = _log(
                    np, _mul(np, _add(np, np.float32(10.0), znt), rznt)
                )
                cdroot = _div(np, karman, log10)
                arrays["cd"][j, i] = _mul(np, cdroot, cdroot)
            elif selection.cecd == 1:
                arrays["cd"][j, i] = max(
                    np.float32(1.0e-10), scalar["cnstcd"]
                )
                exponent = _div(np, karman, _sqrt(np, arrays["cd"][j, i]))
                denominator = _sub(np, _exp(np, exponent), np.float32(1.0))
                arrays["znt"][j, i] = _div(np, np.float32(10.0), denominator)
                znt = arrays["znt"][j, i]
                _, var, logza = _surface_ratio(np, znt, za)
                arrays["ust"][j, i] = max(
                    _div(np, _mul(np, s1, karman), logza), np.float32(1.0e-6)
                )
            else:
                rznt, var, _ = _surface_ratio(np, znt, za)
                arrays["s10"][j, i] = _mul(np, s1, var)
                if arrays["xland"][j, i] > np.float32(1.5):
                    wlast = np.float32(-1000.0)
                    iterations = 0
                    while abs(_sub(np, arrays["s10"][j, i], wlast)) > np.float32(
                        0.001
                    ):
                        iterations += 1
                        wlast = arrays["s10"][j, i]
                        if selection.cecd == 2:
                            arrays["cd"][j, i] = _add(
                                np,
                                np.float32(1.1e-3),
                                _mul(np, np.float32(4.0e-5), arrays["s10"][j, i]),
                            )
                        else:
                            cd = _add(
                                np,
                                dcd1,
                                _mul(
                                    np,
                                    _sub(np, arrays["s10"][j, i], dwsp1),
                                    dfac,
                                ),
                            )
                            cd = min(cd, dcd2)
                            arrays["cd"][j, i] = max(cd, dcd1)
                        exponent = _div(
                            np, karman, _sqrt(np, arrays["cd"][j, i])
                        )
                        denominator = _sub(
                            np, _exp(np, exponent), np.float32(1.0)
                        )
                        arrays["znt"][j, i] = max(
                            np.float32(1.0e-20),
                            _div(np, np.float32(10.0), denominator),
                        )
                        znt = arrays["znt"][j, i]
                        rznt, var, _ = _surface_ratio(np, znt, za)
                        arrays["s10"][j, i] = _mul(np, s1, var)
                        if iterations > 20:
                            refuse(
                                code="CM1-SURFACE-EXCHANGE-NONCONVERGENCE",
                                gate="GM1-PHYSICS-SUPPORT",
                                option="cecd",
                                value=selection.cecd,
                                reason="water roughness iteration exceeded source stop limit",
                                remedy="inspect supplied wind/height/roughness carriers",
                            )
                    max_iterations = max(max_iterations, iterations)
                else:
                    cdroot = _div(
                        np,
                        karman,
                        _log(
                            np,
                            _mul(
                                np,
                                _add(np, np.float32(10.0), znt),
                                rznt,
                            ),
                        ),
                    )
                    arrays["cd"][j, i] = _mul(np, cdroot, cdroot)
                    _, var, _ = _surface_ratio(np, znt, za)
                    arrays["s10"][j, i] = _mul(np, s1, var)
                _, _, logza = _surface_ratio(np, arrays["znt"][j, i], za)
                arrays["ust"][j, i] = max(
                    _div(np, _mul(np, s1, karman), logza), np.float32(1.0e-6)
                )

            arrays["u10"][j, i] = _mul(np, arrays["u1"][j, i], var)
            arrays["v10"][j, i] = _mul(np, arrays["v1"][j, i], var)
            arrays["s10"][j, i] = _mul(np, arrays["s1"][j, i], var)

    if selection.isfcflx == 1:
        for j in range(j2):
            for i in range(i2):
                if arrays["xland"][j, i] > np.float32(1.5):
                    if selection.cecd == 1:
                        arrays["ch"][j, i] = scalar["cnstce"]
                        arrays["cq"][j, i] = scalar["cnstce"]
                    elif selection.cecd == 2:
                        coefficient = _add(
                            np,
                            np.float32(1.1e-3),
                            _mul(np, np.float32(4.0e-5), arrays["s10"][j, i]),
                        )
                        arrays["ch"][j, i] = coefficient
                        arrays["cq"][j, i] = coefficient
                    else:
                        arrays["ch"][j, i] = np.float32(1.20e-3)
                        arrays["cq"][j, i] = np.float32(1.20e-3)
                else:
                    arrays["ch"][j, i] = arrays["cd"][j, i]
                    arrays["cq"][j, i] = arrays["cd"][j, i]

    if selection.use_avg_sfc:
        for name in _OUTPUT_NAMES:
            arrays[name].fill(arrays[name][0, 0])
        if saved is None:
            raise AssertionError("missing averaged-surface saved state")
        arrays["u1"][0, 0], arrays["v1"][0, 0], arrays["s1"][0, 0] = saved

    for name in _OUTPUT_NAMES:
        if not np.all(np.isfinite(arrays[name])):
            refuse(
                code="CM1-SURFACE-EXCHANGE-NONFINITE",
                gate="GM1-PHYSICS-SUPPORT",
                option="surface_exchange_output",
                value=name,
                reason="source-order coefficient math produced a non-finite carrier",
                remedy="inspect supplied height, wind, and roughness inputs",
            )

    receipt: dict[str, object] = {
        "schema": CM1_SURFACE_EXCHANGE_SCHEMA,
        "status": CM1_SURFACE_EXCHANGE_STATUS,
        "claim_limit": CM1_SURFACE_EXCHANGE_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_SURFACE_EXCHANGE_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_SURFACE_EXCHANGE_SOURCE_ANCHORS),
        },
        "selection": {
            "set_znt": selection.set_znt,
            "set_ust": selection.set_ust,
            "cecd": selection.cecd,
            "isfcflx": selection.isfcflx,
            "use_avg_sfc": selection.use_avg_sfc,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
            },
        },
        "domain": {"ni": ni, "nj": nj},
        "max_water_iterations": max_iterations,
        "libm_f32_backend": _libm_f32()[3],
        "output_sha256": {name: _sha256_array(arrays[name]) for name in _OUTPUT_NAMES},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return SurfaceExchangeResult(
        arrays=MappingProxyType({name: arrays[name] for name in _OUTPUT_NAMES}),
        max_water_iterations=max_iterations,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_SURFACE_EXCHANGE_AUTHORITY_SHA256",
    "CM1_SURFACE_EXCHANGE_CLAIM_LIMIT",
    "CM1_SURFACE_EXCHANGE_SCHEMA",
    "CM1_SURFACE_EXCHANGE_SOURCE_ANCHORS",
    "CM1_SURFACE_EXCHANGE_STATUS",
    "SurfaceExchangeResult",
    "SurfaceExchangeSelection",
    "cpu_surface_exchange_coefficients_inplace",
    "surface_exchange_shapes",
]
