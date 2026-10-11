"""Source-exact CM1 r21.1 prescribed surface-flux operator.

This bounded CPU operator owns the ``set_flx=1`` branch in ``sfcflux`` and
the subsequent source-defined testcase 2/6 and 11 overrides, followed by the
serial moisture-budget accumulation.  Testcase 7 is deliberately excluded
because it replaces the prescribed fields through saturation thermodynamics.
Production surface-physics coupling remains fail closed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_PRESCRIBED_SURFACE_FLUX_SCHEMA = "gpuwm-cm1.prescribed-surface-flux/v1"
CM1_PRESCRIBED_SURFACE_FLUX_STATUS = (
    "ADMITTED_CM1_R21_1_PRESCRIBED_SURFACE_FLUX_OPERATOR_ONLY"
)
CM1_PRESCRIBED_SURFACE_FLUX_CLAIM_LIMIT = (
    "CM1 r21.1 sfcflux set_flx=1 source-ordered dry/moist assignment, "
    "testcase 2/6 post-3000-s heat-flux shutdown, testcase 11 three-region "
    "time forcing, and serial moisture-budget accumulation on supplied "
    "compact carriers only; no testcase 7 saturation override, normal "
    "computed-flux branch, surface-layer model, MPI, coupled large step, "
    "production configuration, trajectory, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_PRESCRIBED_SURFACE_FLUX_AUTHORITY_SHA256 = MappingProxyType(
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
CM1_PRESCRIBED_SURFACE_FLUX_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1064-1070",
        "namelist_and_broadcast": "src/input.F:322-329; src/param.F:585-591",
        "prescribed_assignment": "src/sfcphys.F:312-335",
        "testcase_overrides": "src/sfcphys.F:368-429",
        "serial_moisture_budget": "src/sfcphys.F:431-447",
        "pi_constant": "src/constants.F:14",
    }
)
CM1_PRESCRIBED_SURFACE_FLUX_TESTCASES = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15}
)


@dataclass(frozen=True, slots=True)
class PrescribedSurfaceFluxSelection:
    """Exact outer selectors and scalars for the bounded source block."""

    set_flx: int
    isfcflx: int
    imoist: int
    testcase: int
    rtime_seconds: float
    dt_seconds: float
    dx_m: float
    dy_m: float
    cnst_shflx_k_m_s: float
    cnst_lhflx_kg_kg_m_s: float
    initial_qbsfc: float


@dataclass(frozen=True, slots=True)
class PrescribedSurfaceFluxResult:
    """Completed source-order update and immutable receipt."""

    thflux: Any
    qvflux: Any
    qbsfc: float
    receipt: Mapping[str, object]


def prescribed_surface_flux_shapes(ni: int, nj: int) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "ruh": (ni,),
        "rvh": (nj,),
        "rf_bottom": (nj, ni),
        "thflux": (nj, ni),
        "qvflux": (nj, ni),
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


def _validate_selection(
    np: Any, selection: PrescribedSurfaceFluxSelection
) -> dict[str, Any]:
    if not isinstance(selection, PrescribedSurfaceFluxSelection):
        raise TypeError("selection must be a PrescribedSurfaceFluxSelection")
    for name in ("set_flx", "isfcflx", "imoist", "testcase"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.set_flx != 1 or selection.isfcflx != 1:
        refuse(
            code="CM1-PRESCRIBED-SURFACE-FLUX-DISPATCH",
            gate="GM1-PHYSICS-SUPPORT",
            option="set_flx/isfcflx",
            value=(selection.set_flx, selection.isfcflx),
            reason="this operator owns only the active prescribed-flux call path",
            remedy="select set_flx=1 and isfcflx=1 or use a separate surface operator",
        )
    if selection.imoist not in (0, 1):
        refuse(
            code="CM1-PRESCRIBED-SURFACE-FLUX-MOISTURE",
            gate="GM1-PHYSICS-SUPPORT",
            option="imoist",
            value=selection.imoist,
            reason="the bounded source branch admits dry 0 and moist 1 only",
            remedy="select imoist=0/1 or add separately evidenced moisture semantics",
        )
    if selection.testcase == 7:
        refuse(
            code="CM1-PRESCRIBED-SURFACE-FLUX-SATURATION",
            gate="GM1-PHYSICS-SUPPORT",
            option="testcase",
            value=selection.testcase,
            reason="testcase=7 replaces prescribed fluxes using rslf saturation physics",
            remedy="use the separately evidenced saturation operator when available",
        )
    if selection.testcase not in CM1_PRESCRIBED_SURFACE_FLUX_TESTCASES:
        refuse(
            code="CM1-PRESCRIBED-SURFACE-FLUX-TESTCASE",
            gate="GM1-PHYSICS-SUPPORT",
            option="testcase",
            value=selection.testcase,
            reason="the testcase is outside the documented bounded fallthrough set",
            remedy="select a documented non-saturation testcase or add exact evidence",
        )

    scalar = {
        "rtime": _f32(np, "rtime_seconds", selection.rtime_seconds),
        "dt": _f32(np, "dt_seconds", selection.dt_seconds),
        "dx": _f32(np, "dx_m", selection.dx_m),
        "dy": _f32(np, "dy_m", selection.dy_m),
        "cnst_shflx": _f32(
            np, "cnst_shflx_k_m_s", selection.cnst_shflx_k_m_s
        ),
        "cnst_lhflx": _f32(
            np, "cnst_lhflx_kg_kg_m_s", selection.cnst_lhflx_kg_kg_m_s
        ),
    }
    for name in ("dt", "dx", "dy"):
        if scalar[name] <= np.float32(0.0):
            raise ValueError(f"{name} must be positive")
    if isinstance(selection.initial_qbsfc, bool) or not isinstance(
        selection.initial_qbsfc, (int, float)
    ):
        raise TypeError("initial_qbsfc must be a finite binary64 scalar")
    if not math.isfinite(float(selection.initial_qbsfc)):
        raise ValueError("initial_qbsfc must be finite")
    scalar["initial_qbsfc"] = np.float64(selection.initial_qbsfc)
    return scalar


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = prescribed_surface_flux_shapes(ni, nj)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(
            "missing prescribed surface-flux arrays: " + ", ".join(missing)
        )
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
    for name in ("thflux", "qvflux"):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "prescribed surface-flux arrays must not alias: "
                    f"{left_name}, {right_name}"
                )


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_prescribed_surface_flux_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    selection: PrescribedSurfaceFluxSelection,
) -> PrescribedSurfaceFluxResult:
    """Apply the bounded prescribed-flux and serial budget source blocks."""

    np = importlib.import_module("numpy")
    scalar = _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj)

    arrays["thflux"].fill(scalar["cnst_shflx"])
    if selection.imoist == 1:
        arrays["qvflux"].fill(scalar["cnst_lhflx"])

    if selection.testcase in (2, 6):
        if scalar["rtime"] > np.float32(3000.0):
            arrays["thflux"].fill(np.float32(0.0))
    elif selection.testcase == 11:
        if scalar["rtime"] <= np.float32(3600.0):
            thmag = np.float32(0.0)
        else:
            numerator = np.float32(scalar["rtime"] - np.float32(3600.0))
            denominator = _f32_mul(np, np.float32(3600.0), np.float32(13.5))
            phase = _f32_mul(
                np, np.float32("3.1415926535897932384626433"), numerator
            )
            phase = np.float32(phase / denominator)
            thmag = _f32_mul(np, np.float32(0.1), np.sin(phase, dtype=np.float32))
        if scalar["rtime"] <= np.float32(7200.0):
            qvmag = np.float32(0.0)
        else:
            numerator = np.float32(scalar["rtime"] - np.float32(7200.0))
            denominator = _f32_mul(np, np.float32(3600.0), np.float32(12.5))
            phase = _f32_mul(
                np, np.float32("3.1415926535897932384626433"), numerator
            )
            phase = np.float32(phase / denominator)
            qvmag = _f32_mul(np, np.float32(0.15e-3), np.sin(phase, dtype=np.float32))
        arrays["thflux"].fill(thmag)
        arrays["qvflux"].fill(qvmag)

    qbsfc = np.float64(scalar["initial_qbsfc"])
    if selection.imoist == 1:
        bud1 = np.zeros(nj, dtype=np.float64)
        for j in range(nj):
            for i in range(ni):
                contribution = _f32_mul(
                    np, arrays["qvflux"][j, i], arrays["ruh"][i]
                )
                contribution = _f32_mul(np, contribution, arrays["rvh"][j])
                contribution = _f32_mul(
                    np, contribution, arrays["rf_bottom"][j, i]
                )
                bud1[j] = np.float64(bud1[j] + np.float64(contribution))
        tem = _f32_mul(np, scalar["dt"], scalar["dx"])
        tem = _f32_mul(np, tem, scalar["dy"])
        for j in range(nj):
            qbsfc = np.float64(qbsfc + bud1[j] * np.float64(tem))

    qbsfc_bytes = struct.pack("<d", float(qbsfc))
    receipt: dict[str, object] = {
        "schema": CM1_PRESCRIBED_SURFACE_FLUX_SCHEMA,
        "status": CM1_PRESCRIBED_SURFACE_FLUX_STATUS,
        "claim_limit": CM1_PRESCRIBED_SURFACE_FLUX_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_PRESCRIBED_SURFACE_FLUX_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_PRESCRIBED_SURFACE_FLUX_SOURCE_ANCHORS),
        },
        "selection": {
            "set_flx": selection.set_flx,
            "isfcflx": selection.isfcflx,
            "imoist": selection.imoist,
            "testcase": selection.testcase,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
                if name != "initial_qbsfc"
            },
            "initial_qbsfc_raw_bits": int(
                np.asarray(scalar["initial_qbsfc"]).view(np.uint64)
            ),
        },
        "domain": {"ni": ni, "nj": nj},
        "output_sha256": {
            "thflux": _sha256_array(arrays["thflux"]),
            "qvflux": _sha256_array(arrays["qvflux"]),
            "qbsfc_f64le": hashlib.sha256(qbsfc_bytes).hexdigest(),
        },
        "qbsfc_raw_bits": int(np.asarray(qbsfc).view(np.uint64)),
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return PrescribedSurfaceFluxResult(
        thflux=arrays["thflux"],
        qvflux=arrays["qvflux"],
        qbsfc=float(qbsfc),
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_PRESCRIBED_SURFACE_FLUX_AUTHORITY_SHA256",
    "CM1_PRESCRIBED_SURFACE_FLUX_CLAIM_LIMIT",
    "CM1_PRESCRIBED_SURFACE_FLUX_SCHEMA",
    "CM1_PRESCRIBED_SURFACE_FLUX_SOURCE_ANCHORS",
    "CM1_PRESCRIBED_SURFACE_FLUX_STATUS",
    "CM1_PRESCRIBED_SURFACE_FLUX_TESTCASES",
    "PrescribedSurfaceFluxResult",
    "PrescribedSurfaceFluxSelection",
    "cpu_prescribed_surface_flux_inplace",
    "prescribed_surface_flux_shapes",
]
