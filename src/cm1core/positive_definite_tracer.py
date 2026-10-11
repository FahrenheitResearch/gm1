"""Bounded source-exact CM1 r21.1 positive-definite tracer update.

This CPU operator owns the serial-periodic, flat Cartesian final-RK slice of
``pdtra=1``: the complete ``pdefx1/x2``, ``pdefy1/y2``, and ``pdefz`` donor
repair family, the source-ordered scalar tendency/update, and the final
``pdefq(0)`` column redistribution.  Directional advection tendencies and
mass fluxes are supplied carriers.  Advection reconstruction, terrain,
axisymmetry, lateral-open/MPI exchange, diagnostics, and the surrounding RK
trajectory remain outside the claim.
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


CM1_POSITIVE_DEFINITE_TRACER_SCHEMA = "gpuwm-cm1.positive-definite-tracer/v1"
CM1_POSITIVE_DEFINITE_TRACER_STATUS = (
    "ADMITTED_CM1_R21_1_POSITIVE_DEFINITE_TRACER_OPERATOR_ONLY"
)
CM1_POSITIVE_DEFINITE_TRACER_CLAIM_LIMIT = (
    "CM1 r21.1 flat Cartesian serial-periodic final-RK pdtra=1 correction: "
    "pdef.F:15-486, adv.F:428-445,616-653, solve2.F:1906-1946,2061-2071, "
    "and misclibs.F:1037-1117 on supplied directional advection, mass-flux, "
    "density, divergence, and metric carriers; no advection reconstruction, "
    "terrain, axisymmetry, open boundary, MPI exchange, diagnostics, coupled "
    "RK trajectory, production configuration, or forecast claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_POSITIVE_DEFINITE_TRACER_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/adv.F": (
            "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
        ),
        "src/input.F": (
            "e8b3bb25e0b624d79da7d361a2027431b55f98434abfe19b4385d7c7e1692663"
        ),
        "src/misclibs.F": (
            "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/pdef.F": (
            "815cb7729c441b07c8d171596543b15ec630ab0c87eda5ab59a95d4eab2ae670"
        ),
        "src/solve2.F": (
            "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
        ),
    }
)
CM1_POSITIVE_DEFINITE_TRACER_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:726-732",
        "hardwired_pdscheme": "src/param.F:731-732",
        "final_rk_dispatch": "src/solve2.F:1906-1946",
        "directional_dispatch": "src/adv.F:428-445,616-627",
        "tendency_finish": "src/adv.F:631-653",
        "donor_repair_family": "src/pdef.F:15-486",
        "tracer_update": "src/solve2.F:1937-1955",
        "final_pdefq_dispatch": "src/solve2.F:2061-2071",
        "column_redistribution": "src/misclibs.F:1037-1117",
    }
)


@dataclass(frozen=True, slots=True)
class PositiveDefiniteTracerSelection:
    """Selector/cadence and bounded-topology inputs."""

    pdtra: int
    nrk: int
    nrkmax: int
    terrain_flag: bool
    axisymmetric: bool
    mpi_enabled: bool
    lateral_boundary: str
    dt_s: float
    dx_m: float
    dy_m: float
    dz_m: float


@dataclass(frozen=True, slots=True)
class PositiveDefiniteTracerResult:
    """In-place result and immutable operator receipt."""

    arrays: Mapping[str, Any]
    applied: bool
    final_rk: bool
    receipt: Mapping[str, object]


_PHYSICAL = (
    "scalar_a",
    "scalar",
    "rho_final",
    "rmh",
    "advx",
    "advy",
    "advz",
    "sten",
    "divx",
    "scalar_current",
)
_VERTICAL = ("rho0", "rr0", "mh")
_X_METRICS = ("uh", "ruh")
_Y_METRICS = ("vh", "rvh")
_OUTPUTS = ("advx", "advy", "advz", "sten", "scalar_current", "asq")


def positive_definite_tracer_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    physical = (nk, nj, ni)
    shapes = {name: physical for name in _PHYSICAL}
    shapes.update({name: (nk,) for name in _VERTICAL})
    shapes.update({name: (ni,) for name in _X_METRICS})
    shapes.update({name: (nj,) for name in _Y_METRICS})
    shapes.update(
        {
            "rru": (nk, nj, ni + 1),
            "rrv": (nk, nj + 1, ni),
            "rrw": (nk + 1, nj, ni),
            "asq": (1,),
        }
    )
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


def _selection(
    np: Any, selection: PositiveDefiniteTracerSelection
) -> tuple[dict[str, Any], bool]:
    if not isinstance(selection, PositiveDefiniteTracerSelection):
        raise TypeError("selection must be PositiveDefiniteTracerSelection")
    if isinstance(selection.pdtra, bool) or not isinstance(selection.pdtra, int):
        raise TypeError("pdtra must be an integer")
    if selection.pdtra not in (0, 1):
        refuse(
            code="CM1-POSITIVE-DEFINITE-TRACER-SELECTION",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="pdtra",
            value=selection.pdtra,
            reason="CM1 r21.1 documents only pdtra=0 or 1",
            remedy="select pdtra=0 or pdtra=1",
        )
    for name in ("nrk", "nrkmax"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if selection.nrk > selection.nrkmax:
        raise ValueError("nrk must not exceed nrkmax")
    for name in ("terrain_flag", "axisymmetric", "mpi_enabled"):
        if not isinstance(getattr(selection, name), bool):
            raise TypeError(f"{name} must be a bool")
    if not isinstance(selection.lateral_boundary, str):
        raise TypeError("lateral_boundary must be a string")
    scalar = {
        name: _f32(np, name, getattr(selection, name))
        for name in ("dt_s", "dx_m", "dy_m", "dz_m")
    }
    if any(value <= np.float32(0.0) for value in scalar.values()):
        raise ValueError("dt_s, dx_m, dy_m, and dz_m must be strictly positive")
    final_rk = selection.nrk == selection.nrkmax
    active = selection.pdtra == 1 and final_rk
    if active and selection.terrain_flag:
        refuse(
            code="CM1-POSITIVE-DEFINITE-TRACER-TERRAIN",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="terrain_flag",
            value=True,
            reason="the bounded source-exact operator owns only pdef.F flat branches",
            remedy="use a separately audited terrain positive-definite operator",
        )
    if active and selection.axisymmetric:
        refuse(
            code="CM1-POSITIVE-DEFINITE-TRACER-AXISYMMETRY",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="axisymmetric",
            value=True,
            reason="the axisymmetric radial donor weighting is outside this slice",
            remedy="use a separately audited axisymmetric positive-definite operator",
        )
    if active and selection.mpi_enabled:
        refuse(
            code="CM1-POSITIVE-DEFINITE-TRACER-MPI",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="mpi_enabled",
            value=True,
            reason="pdef x/y candidate exchange has not been admitted across MPI ranks",
            remedy="use serial execution or a separately audited MPI exchange",
        )
    if active and selection.lateral_boundary != "periodic":
        refuse(
            code="CM1-POSITIVE-DEFINITE-TRACER-BOUNDARY",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="lateral_boundary",
            value=selection.lateral_boundary,
            reason="only exact serial periodic bcs(dum) halos are owned",
            remedy="select periodic or use a separately audited boundary family",
        )
    return scalar, active


def _validate_arrays(
    np: Any, arrays: MutableMapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = positive_definite_tracer_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing positive-definite tracer arrays: " + ", ".join(missing))
    values: list[tuple[str, Any]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        dtype = np.dtype(np.float64) if name == "asq" else np.dtype(np.float32)
        if value.dtype != dtype:
            raise TypeError(f"{name} must have native {dtype.name} dtype")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        values.append((name, value))
    for name in _OUTPUTS:
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for name in ("rho0", "rr0", "rho_final", "uh", "ruh", "vh", "rvh", "mh", "rmh"):
        if np.any(arrays[name] <= np.float32(0.0)):
            raise ValueError(f"{name} must be strictly positive")
    if not np.array_equal(arrays["rru"][:, :, 0].view(np.uint32), arrays["rru"][:, :, -1].view(np.uint32)):
        raise ValueError("periodic rru first/last faces must be raw-bit identical")
    if not np.array_equal(arrays["rrv"][:, 0, :].view(np.uint32), arrays["rrv"][:, -1, :].view(np.uint32)):
        raise ValueError("periodic rrv first/last faces must be raw-bit identical")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "positive-definite tracer arrays must not alias: "
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


def _sha_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _periodic_line(values: Any) -> Any:
    """CM1 periodic ``bcs`` view for indices -1..n+2."""

    np = importlib.import_module("numpy")
    length = int(values.size)
    result = np.empty((length + 4,), dtype=np.float32)
    for source_index in range(-1, length + 3):
        result[source_index + 1] = values[(source_index - 1) % length]
    return result


def _repair_horizontal_line(np: Any, physical: Any) -> tuple[Any, Any]:
    extended = _periodic_line(physical)
    length = int(physical.size)
    mass = np.zeros((length + 4,), dtype=np.float32)
    flag = np.zeros((length + 4,), dtype=np.bool_)
    # CM1 centers i=0..ni+1, with extended storage indexed by i+1.
    for center in range(0, length + 2):
        here = center + 1
        if extended[here] >= np.float32(0.0):
            continue
        left = np.maximum(np.float32(0.0), extended[here - 1])
        right = np.maximum(np.float32(0.0), extended[here + 1])
        donor_sum = _add(np, left, right)
        if donor_sum <= np.float32(1.0e-20):
            continue
        ratio = _div(
            np,
            np.maximum(extended[here], _sub(np, np.float32(0.0), donor_sum)),
            donor_sum,
        )
        mass[here - 1] = _add(np, mass[here - 1], _mul(np, left, ratio))
        mass[here] = _sub(np, mass[here], _mul(np, donor_sum, ratio))
        mass[here + 1] = _add(np, mass[here + 1], _mul(np, right, ratio))
        if extended[here - 1] > np.float32(1.0e-20):
            flag[here - 1] = True
        flag[here] = True
        if extended[here + 1] > np.float32(1.0e-20):
            flag[here + 1] = True
    return mass[2 : length + 2], flag[2 : length + 2]


def _repair_vertical_line(np: Any, physical: Any) -> tuple[Any, Any]:
    length = int(physical.size)
    extended = np.zeros((length + 2,), dtype=np.float32)
    extended[1 : length + 1] = physical
    mass = np.zeros_like(extended)
    flag = np.zeros((length + 2,), dtype=np.bool_)
    for center in range(1, length + 1):
        if extended[center] >= np.float32(0.0):
            continue
        left = np.maximum(np.float32(0.0), extended[center - 1])
        right = np.maximum(np.float32(0.0), extended[center + 1])
        donor_sum = _add(np, left, right)
        if donor_sum <= np.float32(1.0e-20):
            continue
        ratio = _div(
            np,
            np.maximum(extended[center], _sub(np, np.float32(0.0), donor_sum)),
            donor_sum,
        )
        mass[center - 1] = _add(np, mass[center - 1], _mul(np, left, ratio))
        mass[center] = _sub(np, mass[center], _mul(np, donor_sum, ratio))
        mass[center + 1] = _add(np, mass[center + 1], _mul(np, right, ratio))
        if extended[center - 1] > np.float32(1.0e-20):
            flag[center - 1] = True
        flag[center] = True
        if extended[center + 1] > np.float32(1.0e-20):
            flag[center + 1] = True
    return mass[1 : length + 1], flag[1 : length + 1]


def _candidate(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    axis: str,
    reciprocal_spacing: Any,
    dt: Any,
) -> Any:
    scalar_a = arrays["scalar_a"]
    scalar = arrays["scalar"]
    nk, nj, ni = scalar.shape
    directional = arrays[f"adv{axis}"]
    flux = arrays[{"x": "rru", "y": "rrv", "z": "rrw"}[axis]]
    metric = arrays[{"x": "uh", "y": "vh", "z": "mh"}[axis]]
    result = np.empty_like(scalar)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                if axis == "x":
                    difference = _sub(np, flux[k, j, i + 1], flux[k, j, i])
                    factor = metric[i]
                elif axis == "y":
                    difference = _sub(np, flux[k, j + 1, i], flux[k, j, i])
                    factor = metric[j]
                else:
                    difference = _sub(np, flux[k + 1, j, i], flux[k, j, i])
                    factor = metric[k]
                scalar_divergence = _mul(
                    np,
                    _mul(
                        np,
                        _mul(np, scalar[k, j, i], difference),
                        reciprocal_spacing,
                    ),
                    factor,
                )
                total = _add(np, directional[k, j, i], scalar_divergence)
                result[k, j, i] = _add(
                    np,
                    _mul(np, arrays["rho0"][k], scalar_a[k, j, i]),
                    _mul(np, dt, total),
                )
    return result


def _correct_direction(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    axis: str,
    candidate: Any,
    reciprocal_spacing: Any,
    dt: Any,
) -> None:
    scalar_a = arrays["scalar_a"]
    scalar = arrays["scalar"]
    nk, nj, ni = scalar.shape
    directional = arrays[f"adv{axis}"]
    flux = arrays[{"x": "rru", "y": "rrv", "z": "rrw"}[axis]]
    metric = arrays[{"x": "uh", "y": "vh", "z": "mh"}[axis]]
    reciprocal_dt = _div(np, np.float32(1.0), dt)
    if axis == "x":
        lines = ((k, j, candidate[k, j, :]) for k in range(nk) for j in range(nj))
    elif axis == "y":
        lines = ((k, i, candidate[k, :, i]) for k in range(nk) for i in range(ni))
    else:
        lines = ((j, i, candidate[:, j, i]) for j in range(nj) for i in range(ni))
    for first, second, values in lines:
        if axis == "z":
            mass, flags = _repair_vertical_line(np, values)
        else:
            mass, flags = _repair_horizontal_line(np, values)
        for index in range(int(values.size)):
            if not flags[index]:
                continue
            if axis == "x":
                k, j, i = first, second, index
                difference = _sub(np, flux[k, j, i + 1], flux[k, j, i])
                factor = metric[i]
            elif axis == "y":
                k, j, i = first, index, second
                difference = _sub(np, flux[k, j + 1, i], flux[k, j, i])
                factor = metric[j]
            else:
                k, j, i = index, first, second
                difference = _sub(np, flux[k + 1, j, i], flux[k, j, i])
                factor = metric[k]
            scalar_divergence = _mul(
                np,
                _mul(
                    np,
                    _mul(np, scalar[k, j, i], difference),
                    reciprocal_spacing,
                ),
                factor,
            )
            adjusted = _sub(
                np,
                _add(np, candidate[k, j, i], mass[index]),
                _mul(np, arrays["rho0"][k], scalar_a[k, j, i]),
            )
            adjusted = _mul(np, adjusted, reciprocal_dt)
            directional[k, j, i] = _sub(np, adjusted, scalar_divergence)


def _pdefq_inplace(
    np: Any, arrays: MutableMapping[str, Any], *, dx: Any, dy: Any, dz: Any
) -> None:
    q = arrays["scalar_current"]
    rho = arrays["rho_final"]
    ruh = arrays["ruh"]
    rvh = arrays["rvh"]
    rmh = arrays["rmh"]
    nk, nj, ni = q.shape
    budj = np.zeros((nj,), dtype=np.float64)
    for j in range(nj):
        for i in range(ni):
            t1 = np.float64(0.0)
            t2 = np.float64(0.0)
            a1 = np.float64(0.0)
            a2 = np.float64(0.0)
            for k in range(nk):
                rho_q = _mul(np, rho[k, j, i], q[k, j, i])
                t1 = np.float64(t1 + np.float64(rho_q))
                weight = _mul(
                    np,
                    _mul(np, _mul(np, rho_q, ruh[i]), rvh[j]),
                    rmh[k, j, i],
                )
                a1 = np.float64(a1 + np.float64(weight))
                if q[k, j, i] < np.float32(0.0):
                    q[k, j, i] = np.float32(0.0)
                t2 = np.float64(t2 + np.float64(_mul(np, rho[k, j, i], q[k, j, i])))
            ratio = np.float64(
                (t1 + np.float64(1.0e-20)) / (t2 + np.float64(1.0e-20))
            )
            if ratio < np.float64(0.0):
                ratio = np.float64(1.0)
            for k in range(nk):
                q[k, j, i] = np.float32(ratio * np.float64(q[k, j, i]))
                rho_q = _mul(np, rho[k, j, i], q[k, j, i])
                weight = _mul(
                    np,
                    _mul(np, _mul(np, rho_q, ruh[i]), rvh[j]),
                    rmh[k, j, i],
                )
                a2 = np.float64(a2 + np.float64(weight))
            budj[j] = np.float64(np.float64(budj[j] + a2) - a1)
    volume = np.float64(_mul(np, _mul(np, dx, dy), dz))
    for j in range(nj):
        arrays["asq"][0] = np.float64(
            arrays["asq"][0] + np.float64(budj[j] * volume)
        )


def cpu_positive_definite_tracer_update_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: PositiveDefiniteTracerSelection,
) -> PositiveDefiniteTracerResult:
    """Apply the bounded CM1 ``pdtra`` correction/update in source order."""

    np = importlib.import_module("numpy")
    scalar, active = _selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)
    before = {name: _sha_array(arrays[name]) for name in _OUTPUTS}
    if active:
        reciprocal = {
            "x": _div(np, np.float32(1.0), scalar["dx_m"]),
            "y": _div(np, np.float32(1.0), scalar["dy_m"]),
            "z": _div(np, np.float32(1.0), scalar["dz_m"]),
        }
        candidates = {
            axis: _candidate(
                np,
                arrays,
                axis=axis,
                reciprocal_spacing=reciprocal[axis],
                dt=scalar["dt_s"],
            )
            for axis in ("x", "y", "z")
        }
        # Native source order is pdefz, then pdefx2 and pdefy2; candidates for
        # x/y were prepared earlier by pdefx1/pdefy1.
        for axis in ("z", "x", "y"):
            _correct_direction(
                np,
                arrays,
                axis=axis,
                candidate=candidates[axis],
                reciprocal_spacing=reciprocal[axis],
                dt=scalar["dt_s"],
            )
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    directional = _add(
                        np,
                        _add(np, arrays["advx"][k, j, i], arrays["advy"][k, j, i]),
                        arrays["advz"][k, j, i],
                    )
                    total = _add(
                        np,
                        directional,
                        _mul(np, arrays["scalar"][k, j, i], arrays["divx"][k, j, i]),
                    )
                    arrays["sten"][k, j, i] = _add(
                        np,
                        arrays["sten"][k, j, i],
                        _mul(np, total, arrays["rr0"][k]),
                    )
                    arrays["scalar_current"][k, j, i] = _add(
                        np,
                        arrays["scalar_a"][k, j, i],
                        _mul(np, scalar["dt_s"], arrays["sten"][k, j, i]),
                    )
        _pdefq_inplace(
            np,
            arrays,
            dx=scalar["dx_m"],
            dy=scalar["dy_m"],
            dz=scalar["dz_m"],
        )
    after = {name: _sha_array(arrays[name]) for name in _OUTPUTS}
    receipt = {
        "schema": CM1_POSITIVE_DEFINITE_TRACER_SCHEMA,
        "status": CM1_POSITIVE_DEFINITE_TRACER_STATUS,
        "claim_limit": CM1_POSITIVE_DEFINITE_TRACER_CLAIM_LIMIT,
        "selector": {
            "pdtra": selection.pdtra,
            "nrk": selection.nrk,
            "nrkmax": selection.nrkmax,
            "hardwired_pdscheme": 1,
            "lateral_boundary": selection.lateral_boundary,
        },
        "applied": active,
        "final_rk": selection.nrk == selection.nrkmax,
        "topology": {
            "terrain_flag": selection.terrain_flag,
            "axisymmetric": selection.axisymmetric,
            "mpi_enabled": selection.mpi_enabled,
        },
        "output_sha256_before": before,
        "output_sha256": after,
        "authority_tarball_sha256": CM1_R21_1_TARBALL_SHA256,
        "authority_source_sha256": dict(CM1_POSITIVE_DEFINITE_TRACER_AUTHORITY_SHA256),
        "source_anchors": dict(CM1_POSITIVE_DEFINITE_TRACER_SOURCE_ANCHORS),
    }
    receipt["operation_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return PositiveDefiniteTracerResult(
        MappingProxyType(dict(arrays)),
        active,
        selection.nrk == selection.nrkmax,
        MappingProxyType(receipt),
    )


__all__ = [
    "CM1_POSITIVE_DEFINITE_TRACER_AUTHORITY_SHA256",
    "CM1_POSITIVE_DEFINITE_TRACER_CLAIM_LIMIT",
    "CM1_POSITIVE_DEFINITE_TRACER_SCHEMA",
    "CM1_POSITIVE_DEFINITE_TRACER_SOURCE_ANCHORS",
    "CM1_POSITIVE_DEFINITE_TRACER_STATUS",
    "PositiveDefiniteTracerResult",
    "PositiveDefiniteTracerSelection",
    "cpu_positive_definite_tracer_update_inplace",
    "positive_definite_tracer_shapes",
]
