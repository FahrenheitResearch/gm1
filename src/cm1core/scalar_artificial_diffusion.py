"""Bounded source-exact CM1 r21.1 scalar artificial diffusion.

The operator owns the flat Cartesian scalar branches for ``difforder=2`` in
``diff2.F`` and ``difforder=6`` in ``adv.F`` on supplied halo/metric carriers.
Velocity diffusion, terrain, axisymmetry, boundary exchange, diagnostics, and
the surrounding advection/turbulence trajectory remain outside this CPU-only
claim.
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


CM1_SCALAR_ARTIFICIAL_DIFFUSION_SCHEMA = (
    "gpuwm-cm1.scalar-artificial-diffusion/v1"
)
CM1_SCALAR_ARTIFICIAL_DIFFUSION_STATUS = (
    "ADMITTED_CM1_R21_1_SCALAR_ARTIFICIAL_DIFFUSION_OPERATOR_ONLY"
)
CM1_SCALAR_ARTIFICIAL_DIFFUSION_CLAIM_LIMIT = (
    "CM1 r21.1 flat Cartesian scalar difforder=2 diff2.F:540-735 and "
    "difforder=6 adv.F:342-403 on supplied halo/metric carriers, including "
    "mdiff=0/1 and scalar idiff=0/1/2 disposition; no velocity diffusion, "
    "terrain, axisymmetry, boundary exchange, diagnostics, complete advection "
    "or turbulence dispatch, production configuration, trajectory, forecast, "
    "or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_SCALAR_ARTIFICIAL_DIFFUSION_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/adv.F": (
            "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
        ),
        "src/diff2.F": (
            "00ebe1dc8fc3060758e99eb80f8167108fc81e5bd534781ddcfff0a1c46c1696"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
        "src/solve2.F": (
            "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
        ),
    }
)
CM1_SCALAR_ARTIFICIAL_DIFFUSION_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:278-306,743-747",
        "second_order_dispatch": "src/solve1.F:686-728",
        "second_order_scalar": "src/diff2.F:540-735",
        "sixth_order_scalar": "src/adv.F:342-403",
        "sixth_order_rk_cadence": "src/solve2.F:1393,1475,1582-1599,1847,1940,2036",
        "second_order_compatibility": "src/param.F:1087-1102",
        "axisymmetric_sixth_order_compatibility": "src/param.F:2068-2071",
    }
)


@dataclass(frozen=True, slots=True)
class ScalarArtificialDiffusionSelection:
    """Exact selector and coefficient inputs for the bounded scalar slice."""

    idiff: int
    difforder: int
    mdiff: int
    cm1setup: int
    axisymmetric: bool
    terrain_flag: bool
    kdiff2_m2_s: float
    kdiff6: float
    dt_s: float
    rdx_m_inv: float
    rdy_m_inv: float
    rdz_m_inv: float


@dataclass(frozen=True, slots=True)
class ScalarArtificialDiffusionResult:
    """Completed in-place scalar diffusion update and immutable receipt."""

    arrays: Mapping[str, Any]
    effective_idiff: int
    applied: bool
    receipt: Mapping[str, object]


_HALO_ARRAYS = ("scalar", "rho")
_INTERIOR_ARRAYS = (
    "rho0",
    "ri",
    "rr",
    "advx",
    "advy",
    "diffx",
    "diffy",
    "diffz",
    "sten",
)
_METRIC_ARRAYS = ("uf", "uh", "vf", "vh", "mf", "mh", "rf")
_OUTPUT_ARRAYS = ("advx", "advy", "diffx", "diffy", "diffz", "sten")


def scalar_artificial_diffusion_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    shapes = {name: (nk, nj + 6, ni + 6) for name in _HALO_ARRAYS}
    shapes.update({name: (nk, nj, ni) for name in _INTERIOR_ARRAYS})
    shapes.update(
        {
            "uf": (ni + 1,),
            "uh": (ni,),
            "vf": (nj + 1,),
            "vh": (nj,),
            "mf": (nk + 1,),
            "mh": (nk,),
            "rf": (nk + 1,),
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


def _validate_selection(
    np: Any, selection: ScalarArtificialDiffusionSelection
) -> tuple[dict[str, Any], int]:
    if not isinstance(selection, ScalarArtificialDiffusionSelection):
        raise TypeError("selection must be a ScalarArtificialDiffusionSelection")
    for name, admitted in (
        ("idiff", (0, 1, 2)),
        ("difforder", (2, 6)),
        ("mdiff", (0, 1)),
    ):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value not in admitted:
            refuse(
                code="CM1-SCALAR-ARTIFICIAL-DIFFUSION-SELECTION",
                gate="GM1-DYNAMICS-BRANCH-MATRIX",
                option=name,
                value=value,
                reason=f"the bounded source branch admits {name} in {admitted}",
                remedy=f"select one of {admitted}",
            )
    if isinstance(selection.cm1setup, bool) or not isinstance(selection.cm1setup, int):
        raise TypeError("cm1setup must be an integer")
    if selection.cm1setup < 0:
        raise ValueError("cm1setup must be nonnegative")
    for name in ("axisymmetric", "terrain_flag"):
        if not isinstance(getattr(selection, name), bool):
            raise TypeError(f"{name} must be a bool")
    if selection.axisymmetric:
        refuse(
            code="CM1-SCALAR-ARTIFICIAL-DIFFUSION-AXISYMMETRIC",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="axisymmetric",
            value=True,
            reason="this bounded operator admits only the flat Cartesian scalar branch",
            remedy="use the separately audited axisymmetric operator when available",
        )
    if selection.terrain_flag:
        refuse(
            code="CM1-SCALAR-ARTIFICIAL-DIFFUSION-TERRAIN",
            gate="GM1-DYNAMICS-BRANCH-MATRIX",
            option="terrain_flag",
            value=True,
            reason="diff2s refuses terrain and the bounded sixth-order slice is flat",
            remedy="use a separately audited terrain diffusion operator",
        )
    scalar = {
        name: _f32(np, name, getattr(selection, name))
        for name in (
            "kdiff2_m2_s",
            "kdiff6",
            "dt_s",
            "rdx_m_inv",
            "rdy_m_inv",
            "rdz_m_inv",
        )
    }
    for name in ("dt_s", "rdx_m_inv", "rdy_m_inv", "rdz_m_inv"):
        if scalar[name] <= np.float32(0.0):
            raise ValueError(f"{name} must be strictly positive")
    effective_idiff = selection.idiff
    if selection.cm1setup >= 1 and selection.idiff >= 1 and selection.difforder == 2:
        effective_idiff = 0
    return scalar, effective_idiff


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
    shapes = scalar_artificial_diffusion_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing scalar-artificial-diffusion arrays: " + ", ".join(missing))
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
    for name in ("rho", "rho0", "ri", "rr", "uf", "uh", "vf", "vh", "mf", "mh", "rf"):
        if np.any(arrays[name] <= np.float32(0.0)):
            raise ValueError(f"{name} must be strictly positive")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "scalar-artificial-diffusion arrays must not alias: "
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


def _sixth_difference(np: Any, plus0: Any, minus1: Any, plus1: Any, minus2: Any, plus2: Any, minus3: Any) -> Any:
    return _add(
        np,
        _sub(
            np,
            _mul(np, np.float32(10.0), _sub(np, plus0, minus1)),
            _mul(np, np.float32(5.0), _sub(np, plus1, minus2)),
        ),
        _sub(np, plus2, minus3),
    )


def _apply_second_order(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    coefficient: Any,
    rdx: Any,
    rdy: Any,
    rdz: Any,
) -> None:
    halo = 3
    scalar = arrays["scalar"]
    rho = arrays["rho"]
    dumx = np.empty((ni + 1,), dtype=np.float32)
    dumy = np.empty((nj + 1,), dtype=np.float32)
    for k in range(nk):
        for j in range(nj):
            jj = halo + j
            for i in range(ni + 1):
                ii = halo + i
                density = _mul(
                    np,
                    np.float32(0.5),
                    _add(np, rho[k, jj, ii], rho[k, jj, ii - 1]),
                )
                dumx[i] = _mul(
                    np,
                    _mul(np, _mul(np, density, _sub(np, scalar[k, jj, ii], scalar[k, jj, ii - 1])), rdx),
                    arrays["uf"][i],
                )
            for i in range(ni):
                arrays["diffx"][k, j, i] = _mul(
                    np,
                    _mul(np, _mul(np, coefficient, _sub(np, dumx[i + 1], dumx[i])), rdx),
                    arrays["uh"][i],
                )
        for i in range(ni):
            ii = halo + i
            for j in range(nj + 1):
                jj = halo + j
                density = _mul(
                    np,
                    np.float32(0.5),
                    _add(np, rho[k, jj, ii], rho[k, jj - 1, ii]),
                )
                dumy[j] = _mul(
                    np,
                    _mul(np, _mul(np, density, _sub(np, scalar[k, jj, ii], scalar[k, jj - 1, ii])), rdy),
                    arrays["vf"][j],
                )
            for j in range(nj):
                arrays["diffy"][k, j, i] = _mul(
                    np,
                    _mul(np, _mul(np, coefficient, _sub(np, dumy[j + 1], dumy[j])), rdy),
                    arrays["vh"][j],
                )
    dumz = np.zeros((nk + 1,), dtype=np.float32)
    for j in range(nj):
        jj = halo + j
        for i in range(ni):
            ii = halo + i
            dumz.fill(np.float32(0.0))
            for k in range(1, nk):
                dumz[k] = _mul(
                    np,
                    _mul(np, _sub(np, scalar[k, jj, ii], scalar[k - 1, jj, ii]), rdz),
                    arrays["mf"][k],
                )
            for k in range(nk):
                flux = _sub(
                    np,
                    _mul(np, arrays["rf"][k + 1], dumz[k + 1]),
                    _mul(np, arrays["rf"][k], dumz[k]),
                )
                arrays["diffz"][k, j, i] = _mul(
                    np,
                    _mul(np, _mul(np, coefficient, flux), rdz),
                    arrays["mh"][k],
                )
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                total = _add(
                    np,
                    _add(np, arrays["diffx"][k, j, i], arrays["diffy"][k, j, i]),
                    arrays["diffz"][k, j, i],
                )
                arrays["sten"][k, j, i] = _add(
                    np,
                    arrays["sten"][k, j, i],
                    _mul(np, total, arrays["rr"][k, j, i]),
                )


def _apply_sixth_order(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    coefficient: Any,
    mdiff: int,
) -> None:
    halo = 3
    scalar = arrays["scalar"]
    rho = arrays["rho"]
    dumx = np.empty((ni + 1,), dtype=np.float32)
    dumy = np.empty((nj + 1,), dtype=np.float32)
    for k in range(nk):
        for j in range(nj):
            jj = halo + j
            for i in range(ni + 1):
                ii = halo + i
                difference = _sixth_difference(
                    np,
                    scalar[k, jj, ii],
                    scalar[k, jj, ii - 1],
                    scalar[k, jj, ii + 1],
                    scalar[k, jj, ii - 2],
                    scalar[k, jj, ii + 2],
                    scalar[k, jj, ii - 3],
                )
                density = _mul(
                    np,
                    np.float32(0.5),
                    _add(np, rho[k, jj, ii - 1], rho[k, jj, ii]),
                )
                dumx[i] = _mul(np, difference, density)
                if mdiff == 1 and _mul(
                    np,
                    dumx[i],
                    _sub(np, scalar[k, jj, ii], scalar[k, jj, ii - 1]),
                ) <= np.float32(0.0):
                    dumx[i] = np.float32(0.0)
            for i in range(ni):
                increment = _mul(
                    np,
                    _mul(
                        np,
                        _mul(np, coefficient, _sub(np, dumx[i + 1], dumx[i])),
                        arrays["ri"][k, j, i],
                    ),
                    arrays["rho0"][k, j, i],
                )
                arrays["advx"][k, j, i] = _add(
                    np, arrays["advx"][k, j, i], increment
                )
        for i in range(ni):
            ii = halo + i
            for j in range(nj + 1):
                jj = halo + j
                difference = _sixth_difference(
                    np,
                    scalar[k, jj, ii],
                    scalar[k, jj - 1, ii],
                    scalar[k, jj + 1, ii],
                    scalar[k, jj - 2, ii],
                    scalar[k, jj + 2, ii],
                    scalar[k, jj - 3, ii],
                )
                density = _mul(
                    np,
                    np.float32(0.5),
                    _add(np, rho[k, jj - 1, ii], rho[k, jj, ii]),
                )
                dumy[j] = _mul(np, difference, density)
                if mdiff == 1 and _mul(
                    np,
                    dumy[j],
                    _sub(np, scalar[k, jj, ii], scalar[k, jj - 1, ii]),
                ) <= np.float32(0.0):
                    dumy[j] = np.float32(0.0)
            for j in range(nj):
                increment = _mul(
                    np,
                    _mul(
                        np,
                        _mul(np, coefficient, _sub(np, dumy[j + 1], dumy[j])),
                        arrays["ri"][k, j, i],
                    ),
                    arrays["rho0"][k, j, i],
                )
                arrays["advy"][k, j, i] = _add(
                    np, arrays["advy"][k, j, i], increment
                )


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_scalar_artificial_diffusion_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: ScalarArtificialDiffusionSelection,
) -> ScalarArtificialDiffusionResult:
    """Execute the bounded CM1 scalar artificial-diffusion branch in-place."""

    np = importlib.import_module("numpy")
    scalar, effective_idiff = _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)
    applied = effective_idiff == 1
    if applied and selection.difforder == 2:
        _apply_second_order(
            np,
            arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            coefficient=scalar["kdiff2_m2_s"],
            rdx=scalar["rdx_m_inv"],
            rdy=scalar["rdy_m_inv"],
            rdz=scalar["rdz_m_inv"],
        )
    elif applied:
        coefficient = _div(
            np,
            _div(np, scalar["kdiff6"], np.float32(64.0)),
            scalar["dt_s"],
        )
        _apply_sixth_order(
            np,
            arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            coefficient=coefficient,
            mdiff=selection.mdiff,
        )
    for name in _OUTPUT_ARRAYS:
        if not np.all(np.isfinite(arrays[name])):
            refuse(
                code="CM1-SCALAR-ARTIFICIAL-DIFFUSION-NONFINITE",
                gate="GM1-DYNAMICS-BRANCH-MATRIX",
                option="scalar_artificial_diffusion_output",
                value=name,
                reason="source-order diffusion produced a non-finite carrier",
                remedy="inspect supplied scalar, density, metrics, and coefficients",
            )
    receipt: dict[str, object] = {
        "schema": CM1_SCALAR_ARTIFICIAL_DIFFUSION_SCHEMA,
        "status": CM1_SCALAR_ARTIFICIAL_DIFFUSION_STATUS,
        "claim_limit": CM1_SCALAR_ARTIFICIAL_DIFFUSION_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_SCALAR_ARTIFICIAL_DIFFUSION_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_SCALAR_ARTIFICIAL_DIFFUSION_SOURCE_ANCHORS),
        },
        "selection": {
            "idiff": selection.idiff,
            "effective_idiff": effective_idiff,
            "difforder": selection.difforder,
            "mdiff": selection.mdiff,
            "cm1setup": selection.cm1setup,
            "axisymmetric": selection.axisymmetric,
            "terrain_flag": selection.terrain_flag,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
            },
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk, "supplied_halo_width": 3},
        "applied": applied,
        "output_sha256": {name: _sha256_array(arrays[name]) for name in _OUTPUT_ARRAYS},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return ScalarArtificialDiffusionResult(
        arrays=MappingProxyType({name: arrays[name] for name in _OUTPUT_ARRAYS}),
        effective_idiff=effective_idiff,
        applied=applied,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_SCALAR_ARTIFICIAL_DIFFUSION_AUTHORITY_SHA256",
    "CM1_SCALAR_ARTIFICIAL_DIFFUSION_CLAIM_LIMIT",
    "CM1_SCALAR_ARTIFICIAL_DIFFUSION_SCHEMA",
    "CM1_SCALAR_ARTIFICIAL_DIFFUSION_SOURCE_ANCHORS",
    "CM1_SCALAR_ARTIFICIAL_DIFFUSION_STATUS",
    "ScalarArtificialDiffusionResult",
    "ScalarArtificialDiffusionSelection",
    "cpu_scalar_artificial_diffusion_inplace",
    "scalar_artificial_diffusion_shapes",
]
