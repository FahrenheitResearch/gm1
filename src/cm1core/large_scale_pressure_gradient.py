"""CM1 r21.1 ``lspgrad=1..4`` large-scale pressure forcing.

This is a bounded CPU operator for the four active branches in
``solve1.F:536-681``.  It updates the exact common ``i=1..ni+1``,
``j=1..nj+1``, ``k=1..nk`` tendency window and the source ``ulspg`` /
``vlspg`` profiles.  The base-wind branch accepts compact staggered halo
windows whose origins are stated by :func:`large_scale_pressure_gradient_shapes`.

The operator is deliberately not wired into an existing production profile.
Its independently validated branches are operator-only: callers must select
one explicitly and provide every branch carrier.  The disabled ``lspgrad=0``
identity remains outside this active operator.
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


CM1_LSPGRAD_SCHEMA = "gpuwm-cm1.large-scale-pressure-gradient/v1"
CM1_LSPGRAD_STATUS = "ADMITTED_CM1_R21_1_LSPGRAD_OPERATOR_ONLY"
CM1_LSPGRAD_CLAIM_LIMIT = (
    "CM1 r21.1 solve1.F:536-681 lspgrad=1..4 default-REAL source-order "
    "large-scale pressure-gradient tendency window and ulspg/vlspg profiles "
    "only; no adaptive lspgrad branch (hard-disabled by the source), Coriolis, "
    "other solve1 terms, halo publication outside i=1..ni+1/j=1..nj+1, "
    "coupled trajectory, production configuration, forecast, or validation claim"
)

CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_LSPGRAD_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
    }
)
CM1_LSPGRAD_SOURCE_ANCHORS = MappingProxyType(
    {
        "selection_documentation": "README.namelist:531-538",
        "adaptive_branch_hard_disabled": "src/solve1.F:538-588",
        "base_wind_geostrophic": "src/solve1.F:590-612",
        "ug_vg_geostrophic": "src/solve1.F:614-631",
        "gradient_wind": "src/solve1.F:633-659",
        "user_acceleration": "src/solve1.F:661-678",
        "hurricane_angle": "src/param.F:8971-8974",
    }
)


@dataclass(frozen=True, slots=True)
class LargeScalePressureGradientSelection:
    """Explicit scalar selection for one active CM1 ``lspgrad`` branch."""

    lspgrad: int
    fcor: float
    umove: float = 0.0
    vmove: float = 0.0
    hurr_rad_m: float | None = None
    hurr_rotate_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class LargeScalePressureGradientResult:
    """Completed in-place operator and its canonical audit receipt."""

    ulspg: Any
    vlspg: Any
    receipt: Mapping[str, object]


def large_scale_pressure_gradient_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    """Return the exact compact C-order shapes for the source loop window.

    ``u0_window[k, j, i]`` maps ``u0(i+1,j,k)`` for Python ``i=0..ni+1``
    and source ``j=0..nj+1``.  ``v0_window[k, j, i]`` maps ``v0(i,j+1,k)``
    for source ``i=0..ni+1`` and Python ``j=0..nj+1``.  These distinct
    origins make both required one-cell staggered stencil windows compact.
    """

    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "uten1": (nk, nj + 1, ni + 1),
        "vten1": (nk, nj + 1, ni + 1),
        "ulspg": (nk,),
        "vlspg": (nk,),
        "u0_window": (nk, nj + 2, ni + 2),
        "v0_window": (nk, nj + 2, ni + 2),
        "ug": (nk,),
        "vg": (nk,),
    }


def _f32_scalar(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not representable as finite CM1 default REAL")
    return result


def _validate_selection(np: Any, selection: LargeScalePressureGradientSelection) -> None:
    if not isinstance(selection, LargeScalePressureGradientSelection):
        raise TypeError("selection must be a LargeScalePressureGradientSelection")
    if isinstance(selection.lspgrad, bool) or not isinstance(selection.lspgrad, int):
        raise TypeError("lspgrad must be an integer")
    if selection.lspgrad not in (1, 2, 3, 4):
        refuse(
            code="CM1-LSPGRAD-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="lspgrad",
            value=selection.lspgrad,
            reason="the active operator is source-bound only to lspgrad=1,2,3,4",
            remedy="use the separate lspgrad=0 identity path or add a separately evidenced branch",
        )
    for name in ("fcor", "umove", "vmove"):
        _f32_scalar(np, name, getattr(selection, name))
    if selection.lspgrad == 3:
        if selection.hurr_rad_m is None or selection.hurr_rotate_degrees is None:
            raise ValueError(
                "lspgrad=3 requires hurr_rad_m and hurr_rotate_degrees"
            )
        radius = _f32_scalar(np, "hurr_rad_m", selection.hurr_rad_m)
        if radius <= np.float32(0.0):
            raise ValueError("hurr_rad_m must be positive")
        _f32_scalar(
            np, "hurr_rotate_degrees", selection.hurr_rotate_degrees
        )


def _required_names(lspgrad: int) -> tuple[str, ...]:
    common = ("uten1", "vten1", "ulspg", "vlspg")
    if lspgrad == 1:
        return (*common, "u0_window", "v0_window")
    if lspgrad == 2:
        return (*common, "ug", "vg")
    if lspgrad == 3:
        return (*common, "vg")
    return common


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    lspgrad: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = large_scale_pressure_gradient_shapes(ni, nj, nk)
    required = _required_names(lspgrad)
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing lspgrad arrays: {', '.join(missing)}")
    values = []
    for name in required:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must have native float32 dtype")
        if tuple(value.shape) != shapes[name]:
            raise ValueError(f"{name} shape {value.shape} != {shapes[name]}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        values.append((name, value))
    for name in ("uten1", "vten1", "ulspg", "vlspg"):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    f"lspgrad arrays must not alias: {left_name}, {right_name}"
                )


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cpu_large_scale_pressure_gradient_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: LargeScalePressureGradientSelection,
) -> LargeScalePressureGradientResult:
    """Apply one exact active ``lspgrad`` branch to compact CPU carriers."""

    np = importlib.import_module("numpy")
    _validate_selection(np, selection)
    _validate_arrays(
        np,
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        lspgrad=selection.lspgrad,
    )

    uten1 = arrays["uten1"]
    vten1 = arrays["vten1"]
    ulspg = arrays["ulspg"]
    vlspg = arrays["vlspg"]
    fcor = _f32_scalar(np, "fcor", selection.fcor)
    umove = _f32_scalar(np, "umove", selection.umove)
    vmove = _f32_scalar(np, "vmove", selection.vmove)

    if selection.lspgrad == 1:
        # solve1.F:592-609.  Keep each parenthesized default-REAL operation
        # separate so NumPy cannot reassociate the staggered four-point sum.
        ulspg.fill(np.float32(0.0))
        vlspg.fill(np.float32(0.0))
        u0 = arrays["u0_window"]
        v0 = arrays["v0_window"]
        first = np.empty_like(uten1)
        second = np.empty_like(uten1)
        np.add(v0[:, :-1, 1:], v0[:, 1:, 1:], out=first)
        np.add(v0[:, :-1, :-1], v0[:, 1:, :-1], out=second)
        np.add(first, second, out=first)
        np.multiply(first, np.float32(0.25), out=first)
        np.add(first, vmove, out=first)
        np.multiply(first, fcor, out=first)
        np.subtract(uten1, first, out=uten1)

        np.add(u0[:, 1:, :-1], u0[:, 1:, 1:], out=first)
        np.add(u0[:, :-1, :-1], u0[:, :-1, 1:], out=second)
        np.add(first, second, out=first)
        np.multiply(first, np.float32(0.25), out=first)
        np.add(first, umove, out=first)
        np.multiply(first, fcor, out=first)
        np.add(vten1, first, out=vten1)
    elif selection.lspgrad == 2:
        # solve1.F:623-628.
        np.add(arrays["vg"], vmove, out=ulspg)
        np.multiply(ulspg, np.negative(fcor), out=ulspg)
        np.add(arrays["ug"], umove, out=vlspg)
        np.multiply(vlspg, fcor, out=vlspg)
        np.add(uten1, ulspg[:, None, None], out=uten1)
        np.add(vten1, vlspg[:, None, None], out=vten1)
    elif selection.lspgrad == 3:
        # solve1.F:646-655 and param.F:8971-8974.  ``fr`` is default REAL,
        # while hurr_angle and the trig result are double precision.
        vg = arrays["vg"]
        radius = _f32_scalar(np, "hurr_rad_m", selection.hurr_rad_m)
        np.multiply(vg, np.negative(fcor), out=ulspg)
        work = np.empty_like(vg)
        np.multiply(vg, vg, out=work)
        np.divide(work, radius, out=work)
        np.subtract(ulspg, work, out=ulspg)
        fr = ulspg.copy(order="C")
        rotate_real = _f32_scalar(
            np, "hurr_rotate_degrees", selection.hurr_rotate_degrees
        )
        angle = float(rotate_real) * (math.pi / 180.0)
        ulspg[...] = (
            fr.astype(np.float64) * math.cos(-angle)
        ).astype(np.float32)
        vlspg[...] = (
            fr.astype(np.float64) * math.sin(-angle)
        ).astype(np.float32)
        np.add(uten1, ulspg[:, None, None], out=uten1)
        np.add(vten1, vlspg[:, None, None], out=vten1)
    else:
        # solve1.F:661-673: user-supplied profiles are inputs and unchanged.
        np.add(uten1, ulspg[:, None, None], out=uten1)
        np.add(vten1, vlspg[:, None, None], out=vten1)

    receipt: dict[str, object] = {
        "schema": CM1_LSPGRAD_SCHEMA,
        "status": CM1_LSPGRAD_STATUS,
        "claim_limit": CM1_LSPGRAD_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_LSPGRAD_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_LSPGRAD_SOURCE_ANCHORS),
        },
        "selection": {
            "lspgrad": selection.lspgrad,
            "fcor_f32_bits": int(fcor.view(np.uint32)),
            "umove_f32_bits": int(umove.view(np.uint32)),
            "vmove_f32_bits": int(vmove.view(np.uint32)),
            "adaptive_lspgrad": False,
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "output_sha256": {
            name: _sha256_array(arrays[name])
            for name in ("uten1", "vten1", "ulspg", "vlspg")
        },
    }
    if selection.lspgrad == 3:
        radius = _f32_scalar(np, "hurr_rad_m", selection.hurr_rad_m)
        rotate = _f32_scalar(
            np, "hurr_rotate_degrees", selection.hurr_rotate_degrees
        )
        receipt["selection"].update(  # type: ignore[union-attr]
            {
                "hurr_rad_m_f32_bits": int(radius.view(np.uint32)),
                "hurr_rotate_degrees_f32_bits": int(rotate.view(np.uint32)),
            }
        )
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return LargeScalePressureGradientResult(
        ulspg=ulspg,
        vlspg=vlspg,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_LSPGRAD_AUTHORITY_SHA256",
    "CM1_LSPGRAD_CLAIM_LIMIT",
    "CM1_LSPGRAD_SCHEMA",
    "CM1_LSPGRAD_SOURCE_ANCHORS",
    "CM1_LSPGRAD_STATUS",
    "LargeScalePressureGradientResult",
    "LargeScalePressureGradientSelection",
    "cpu_large_scale_pressure_gradient_inplace",
    "large_scale_pressure_gradient_shapes",
]
