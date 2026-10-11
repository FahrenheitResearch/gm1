"""Source-exact CM1 r21.1 idealized Rotunno--Emanuel radiation operator.

This bounded CPU operator covers the complete localized ``rterm=1`` family in
``solve1.F``: dry/moist heat-capacity preparation, optional liquid and ice
category sums, Cartesian/axisymmetric budget mass weights, all three capped
radiative-tendency regions, optional tendency diagnostics, associated pressure
tendency, and the binary64 energy-budget accumulation.  Production coupling,
MPI reduction, and simultaneous full radiation schemes remain excluded.
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


CM1_IDEALIZED_RADIATION_SCHEMA = "gpuwm-cm1.idealized-radiation/v1"
CM1_IDEALIZED_RADIATION_STATUS = (
    "ADMITTED_CM1_R21_1_IDEALIZED_RADIATION_OPERATOR_ONLY"
)
CM1_IDEALIZED_RADIATION_CLAIM_LIMIT = (
    "CM1 r21.1 default-constant cvm preparation, Cartesian/axisymmetric inner "
    "budget weight, capped rterm=1 theta/pressure tendency, optional dedicated "
    "diagnostic, and serial binary64 energy budget only; no arbitrary testcase "
    "constants, radopt composition, MPI reduction, complete axisymmetric "
    "coordinate system, coupled solve1 trajectory, production configuration, "
    "forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_IDEALIZED_RADIATION_AUTHORITY_SHA256 = MappingProxyType(
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
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
    }
)
CM1_IDEALIZED_RADIATION_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:555-562",
        "default_constants": "src/constants.F:86-152",
        "full_radiation_conflict": "src/param.F:2279-2289",
        "cvm_and_budget_weight": "src/solve1.F:744-850",
        "radiative_tendency_and_budget": "src/solve1.F:1002-1056",
    }
)


@dataclass(frozen=True, slots=True)
class IdealizedRadiationSelection:
    """Explicit bounded source selectors and category disposition."""

    rterm: int
    radopt: int
    eqtset: int
    imoist: int
    iice: int
    nqv: int
    nql1: int
    nql2: int
    nqs1: int
    nqs2: int
    axisymmetric: int
    diagnostic: bool
    mpi: bool
    dt_seconds: float
    dx_m: float
    dy_m: float
    dz_m: float


@dataclass(frozen=True, slots=True)
class IdealizedRadiationResult:
    """Completed in-place bounded operator receipt."""

    receipt: Mapping[str, object]


def idealized_radiation_shapes(
    ni: int, nj: int, nk: int, numq: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk), ("numq", numq)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    state = (nk, nj, ni)
    return {
        "tha": state,
        "thten1": state,
        "pi0": state,
        "ppi": state,
        "th0": state,
        "rho": state,
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": state,
        "xf": (ni + 2,),
        "qa": (numq, nk, nj, ni),
        "cvm": state,
        "mass_weight": state,
        "thrad": state,
        "prad": state,
        "bud": (nk,),
        "qbudget": (10,),
        "tdiag": state,
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


def _validate_category_range(
    *, name: str, first: int, last: int, numq: int, required: bool
) -> set[int]:
    if first == 0 and last == 0 and not required:
        return set()
    if first < 1 or last < first or last > numq:
        raise ValueError(f"{name} category range must be 1..numq in source order")
    return set(range(first, last + 1))


def _validate_selection(
    np: Any, selection: IdealizedRadiationSelection, *, numq: int
) -> None:
    if not isinstance(selection, IdealizedRadiationSelection):
        raise TypeError("selection must be an IdealizedRadiationSelection")
    for name in (
        "rterm",
        "radopt",
        "eqtset",
        "imoist",
        "iice",
        "nqv",
        "nql1",
        "nql2",
        "nqs1",
        "nqs2",
        "axisymmetric",
    ):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.rterm != 1:
        refuse(
            code="CM1-IDEALIZED-RADIATION-SELECTION",
            gate="GM1-PHYSICS-SUPPORT",
            option="rterm",
            value=selection.rterm,
            reason="the active operator owns exactly CM1 rterm=1",
            remedy="use the rterm=0 identity path or rterm=1",
        )
    if selection.radopt != 0:
        refuse(
            code="CM1-IDEALIZED-RADIATION-CONFLICT",
            gate="GM1-PHYSICS-SUPPORT",
            option="radopt",
            value=selection.radopt,
            reason="param.F refuses simultaneous radopt and rterm",
            remedy="set radopt=0 for the idealized radiation operator",
        )
    if selection.eqtset not in (1, 2):
        raise ValueError("eqtset must be 1 or 2")
    if selection.imoist not in (0, 1) or selection.iice not in (0, 1):
        raise ValueError("imoist and iice must be 0 or 1")
    if selection.axisymmetric not in (0, 1):
        raise ValueError("axisymmetric must be 0 or 1")
    if not isinstance(selection.diagnostic, bool) or not isinstance(selection.mpi, bool):
        raise TypeError("diagnostic and mpi must be bool")
    if selection.mpi:
        refuse(
            code="CM1-IDEALIZED-RADIATION-SELECTION",
            gate="GM1-PHYSICS-SUPPORT",
            option="mpi",
            value=True,
            reason="the bounded energy-budget accumulator is serial",
            remedy="use mpi=false",
        )
    for name in ("dt_seconds", "dx_m", "dy_m", "dz_m"):
        if _f32(np, name, getattr(selection, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if selection.imoist == 0:
        if any(
            value != 0
            for value in (
                selection.nqv,
                selection.nql1,
                selection.nql2,
                selection.nqs1,
                selection.nqs2,
                selection.iice,
            )
        ):
            raise ValueError("dry selection requires zero moisture category indices")
        return
    if not 1 <= selection.nqv <= numq:
        raise ValueError("nqv must be a valid 1-based category")
    liquid = _validate_category_range(
        name="liquid", first=selection.nql1, last=selection.nql2, numq=numq, required=False
    )
    ice = _validate_category_range(
        name="ice",
        first=selection.nqs1,
        last=selection.nqs2,
        numq=numq,
        required=selection.iice == 1,
    )
    if selection.iice == 0 and ice:
        raise ValueError("iice=0 requires zero ice category indices")
    if {selection.nqv} & (liquid | ice) or liquid & ice:
        raise ValueError("vapor, liquid, and ice category ranges must be disjoint")


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    numq: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = idealized_radiation_shapes(ni, nj, nk, numq)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing idealized-radiation arrays: {', '.join(missing)}")
    values: list[tuple[str, Any]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        dtype = np.float64 if name in ("bud", "qbudget") else np.float32
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(dtype):
            raise TypeError(f"{name} must have native {np.dtype(dtype)} dtype")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        values.append((name, value))
    for name in (
        "thten1",
        "cvm",
        "mass_weight",
        "thrad",
        "prad",
        "bud",
        "qbudget",
        "tdiag",
    ):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "idealized-radiation arrays must not alias: "
                    f"{left_name}, {right_name}"
                )
    denominator = np.add(arrays["th0"], arrays["tha"], dtype=np.float32)
    if np.any(denominator == np.float32(0.0)):
        raise ValueError("th0+tha pressure-tendency denominator must be nonzero")


def _default_constants(np: Any) -> dict[str, Any]:
    rd = np.float32(287.04)
    cp = np.float32(1005.7)
    cpv = np.float32(1870.0)
    rv = np.float32(461.5)
    cv = np.float32(cp - rd)
    return {
        "pi": np.float32("3.1415926535897932384626433"),
        "cv": cv,
        "cvv": np.float32(cpv - rv),
        "rddcv": np.float32(rd / cv),
        "cpl": np.float32(4190.0),
        "cpi": np.float32(2106.0),
    }


def _build_cvm(
    np: Any,
    arrays: MutableMapping[str, Any],
    selection: IdealizedRadiationSelection,
    constants: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    cvm = arrays["cvm"]
    if selection.eqtset != 2 or selection.imoist != 1:
        cvm.fill(constants["cv"])
        return
    qa = arrays["qa"]
    liquid = np.empty((nk, nj, ni), dtype=np.float32)
    ice = np.empty((nk, nj, ni), dtype=np.float32)
    if selection.nql1 >= 1:
        liquid[...] = qa[selection.nql1 - 1]
        for category in range(selection.nql1, selection.nql2):
            np.add(liquid, qa[category], out=liquid)
    else:
        liquid.fill(np.float32(0.0))
    if selection.iice == 1:
        ice[...] = qa[selection.nqs1 - 1]
        for category in range(selection.nqs1, selection.nqs2):
            np.add(ice, qa[category], out=ice)
    else:
        ice.fill(np.float32(0.0))
    qv = qa[selection.nqv - 1]
    np.multiply(constants["cvv"], qv, out=cvm)
    np.add(constants["cv"], cvm, out=cvm)
    work = np.empty_like(cvm)
    np.multiply(constants["cpl"], liquid, out=work)
    np.add(cvm, work, out=cvm)
    np.multiply(constants["cpi"], ice, out=work)
    np.add(cvm, work, out=cvm)


def _build_mass_weight(
    np: Any,
    arrays: MutableMapping[str, Any],
    selection: IdealizedRadiationSelection,
    constants: Mapping[str, Any],
    *,
    ni: int,
) -> None:
    mass = arrays["mass_weight"]
    if selection.axisymmetric == 0:
        mass[...] = arrays["rho"]
        return
    dx = _f32(np, "dx_m", selection.dx_m)
    dy = _f32(np, "dy_m", selection.dy_m)
    area = np.float32(dx * dy)
    for i in range(ni):
        outer = np.float32(arrays["xf"][i + 2] * arrays["xf"][i + 2])
        inner = np.float32(arrays["xf"][i + 1] * arrays["xf"][i + 1])
        annulus = np.float32(outer - inner)
        # Preserve the default-REAL source association exactly:
        # rho*pi*(xf(i+1)**2-xf(i)**2)/(dx*dy).  Collapsing the last three
        # factors into one scalar changes legitimate binary32 roundings.
        np.multiply(
            arrays["rho"][:, :, i], constants["pi"], out=mass[:, :, i]
        )
        np.multiply(mass[:, :, i], annulus, out=mass[:, :, i])
        np.divide(mass[:, :, i], area, out=mass[:, :, i])


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_idealized_radiation_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    numq: int,
    selection: IdealizedRadiationSelection,
) -> IdealizedRadiationResult:
    """Apply the bounded source-order ``rterm=1`` operator in place."""

    np = importlib.import_module("numpy")
    _validate_selection(np, selection, numq=numq)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk, numq=numq)
    constants = _default_constants(np)
    _build_cvm(np, arrays, selection, constants, ni=ni, nj=nj, nk=nk)
    _build_mass_weight(np, arrays, selection, constants, ni=ni)

    if selection.diagnostic:
        arrays["tdiag"][...] = arrays["thten1"]

    dt = _f32(np, "dt_seconds", selection.dt_seconds)
    dx = _f32(np, "dx_m", selection.dx_m)
    dy = _f32(np, "dy_m", selection.dy_m)
    dz = _f32(np, "dz_m", selection.dz_m)
    tem0 = np.float32(dt * dx)
    tem0 = np.float32(tem0 * dy)
    tem0 = np.float32(tem0 * dz)
    cap_denominator = np.float32(np.float32(12.0) * np.float32(3600.0))
    negative_cap = np.float32(np.float32(-1.0) / cap_denominator)
    positive_cap = np.float32(np.float32(1.0) / cap_denominator)
    arrays["bud"].fill(np.float64(0.0))

    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                theta_perturbation = arrays["tha"][k, j, i]
                thrad = np.float32(np.float32(-theta_perturbation) / cap_denominator)
                if theta_perturbation > np.float32(1.0):
                    thrad = negative_cap
                if theta_perturbation < np.float32(-1.0):
                    thrad = positive_cap
                arrays["thrad"][k, j, i] = thrad
                arrays["thten1"][k, j, i] = np.float32(
                    arrays["thten1"][k, j, i] + thrad
                )
                pressure_sum = np.float32(
                    arrays["pi0"][k, j, i] + arrays["ppi"][k, j, i]
                )
                prad = np.float32(pressure_sum * constants["rddcv"])
                prad = np.float32(prad * thrad)
                theta_sum = np.float32(
                    arrays["th0"][k, j, i] + theta_perturbation
                )
                prad = np.float32(prad / theta_sum)
                arrays["prad"][k, j, i] = prad
                first = np.float32(thrad * pressure_sum)
                second = np.float32(prad * theta_sum)
                thermodynamic = np.float32(first + second)
                term = np.float32(
                    arrays["cvm"][k, j, i] * arrays["mass_weight"][k, j, i]
                )
                term = np.float32(term * arrays["ruh"][i])
                term = np.float32(term * arrays["rvh"][j])
                term = np.float32(term * arrays["rmh"][k, j, i])
                term = np.float32(term * thermodynamic)
                arrays["bud"][k] = np.float64(
                    arrays["bud"][k] + np.float64(term)
                )
        arrays["qbudget"][9] = np.float64(
            arrays["qbudget"][9] + np.float64(tem0) * arrays["bud"][k]
        )

    if selection.diagnostic:
        np.subtract(arrays["thten1"], arrays["tdiag"], out=arrays["tdiag"])

    output_names = (
        "cvm",
        "mass_weight",
        "thrad",
        "prad",
        "bud",
        "qbudget",
        "thten1",
        "tdiag",
    )
    receipt: dict[str, object] = {
        "schema": CM1_IDEALIZED_RADIATION_SCHEMA,
        "status": CM1_IDEALIZED_RADIATION_STATUS,
        "claim_limit": CM1_IDEALIZED_RADIATION_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_IDEALIZED_RADIATION_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_IDEALIZED_RADIATION_SOURCE_ANCHORS),
        },
        "selection": {
            "rterm": selection.rterm,
            "radopt": selection.radopt,
            "eqtset": selection.eqtset,
            "imoist": selection.imoist,
            "iice": selection.iice,
            "axisymmetric_inner_budget_weight": selection.axisymmetric,
            "diagnostic": selection.diagnostic,
            "mpi": selection.mpi,
            "categories": {
                "nqv": selection.nqv,
                "nql1": selection.nql1,
                "nql2": selection.nql2,
                "nqs1": selection.nqs1,
                "nqs2": selection.nqs2,
            },
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk, "numq": numq},
        "constants_raw_bits": {
            name: int(np.asarray(value, dtype=np.float32).view(np.uint32))
            for name, value in constants.items()
        },
        "output_sha256": {name: _sha256_array(arrays[name]) for name in output_names},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return IdealizedRadiationResult(receipt=MappingProxyType(receipt))


__all__ = [
    "CM1_IDEALIZED_RADIATION_AUTHORITY_SHA256",
    "CM1_IDEALIZED_RADIATION_CLAIM_LIMIT",
    "CM1_IDEALIZED_RADIATION_SCHEMA",
    "CM1_IDEALIZED_RADIATION_SOURCE_ANCHORS",
    "CM1_IDEALIZED_RADIATION_STATUS",
    "IdealizedRadiationResult",
    "IdealizedRadiationSelection",
    "cpu_idealized_radiation_inplace",
    "idealized_radiation_shapes",
]
