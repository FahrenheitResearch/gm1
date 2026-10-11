"""Source-exact CM1 r21.1 Coriolis and beta-plane CPU operator.

The module covers every inner branch under ``icor=1`` in
``solve2.F:631-720``: Cartesian f-plane, Cartesian beta-plane, and the
axisymmetric u-tendency term.  It also builds the source ``f2d`` window from
``param.F:8945-8969``.  Only the compact tendency window is admitted; this
operator does not promote the separately unported axisymmetric coordinate
system or any coupled trajectory.
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


CM1_ROTATION_FORCING_SCHEMA = "gpuwm-cm1.rotation-forcing/v1"
CM1_ROTATION_FORCING_STATUS = "ADMITTED_CM1_R21_1_ROTATION_OPERATOR_ONLY"
CM1_ROTATION_FORCING_CLAIM_LIMIT = (
    "CM1 r21.1 param.F:8945-8969 f2d construction and solve2.F:631-720 "
    "icor=1 compact tendency-window operator for Cartesian f-plane, Cartesian "
    "beta-plane, and the axisymmetric u term only; no budget diagnostics, "
    "complete axisymmetric coordinate system, other solve2 terms, coupled "
    "trajectory, production configuration, forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_ROTATION_AUTHORITY_SHA256 = MappingProxyType(
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
        "src/solve2.F": (
            "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
        ),
    }
)
CM1_ROTATION_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:518-529",
        "constants": "src/constants.F:14-25,87",
        "f2d_builder": "src/param.F:8945-8969",
        "cartesian_f_plane": "src/solve2.F:633-653",
        "cartesian_beta_plane": "src/solve2.F:655-673",
        "axisymmetric": "src/solve2.F:675-690",
        "optional_budget_diagnostics": "src/solve2.F:693-716",
    }
)


@dataclass(frozen=True, slots=True)
class RotationForcingSelection:
    """Explicit selection for one active CM1 Coriolis inner branch."""

    icor: int
    beta_plane: int
    axisymmetric: int
    fcor_s_inv: float
    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0
    center_y_m: float = 0.0


@dataclass(frozen=True, slots=True)
class RotationForcingResult:
    """Completed in-place forcing and its source-bound receipt."""

    f2d: Any
    receipt: Mapping[str, object]


def rotation_forcing_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    """Return compact C-order shapes for the full source stencil window."""

    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "u3d_window": (nk, nj + 2, ni + 2),
        "v3d_window": (nk, nj + 2, ni + 2),
        "uten": (nk, nj + 1, ni + 1),
        "vten": (nk, nj + 1, ni + 1),
        "yh_window": (nj + 2,),
        "f2d": (nj + 2, ni + 2),
    }


def _f32_scalar(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite CM1 default REAL")
    return result


def _validate_selection(np: Any, selection: RotationForcingSelection) -> None:
    if not isinstance(selection, RotationForcingSelection):
        raise TypeError("selection must be a RotationForcingSelection")
    for name in ("icor", "beta_plane", "axisymmetric"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.icor != 1:
        refuse(
            code="CM1-ROTATION-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="icor",
            value=selection.icor,
            reason="this active operator is source-bound only to icor=1",
            remedy="use the separate icor=0 identity path",
        )
    if selection.axisymmetric not in (0, 1):
        refuse(
            code="CM1-ROTATION-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="axisymmetric",
            value=selection.axisymmetric,
            reason="only the two source dispatch values are represented",
            remedy="use axisymmetric=0 or 1",
        )
    if selection.beta_plane not in (0, 1):
        refuse(
            code="CM1-ROTATION-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="beta_plane",
            value=selection.beta_plane,
            reason="only the two source beta-plane values are represented",
            remedy="use beta_plane=0 or 1",
        )
    if selection.axisymmetric == 1 and selection.beta_plane != 0:
        refuse(
            code="CM1-ROTATION-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="beta_plane",
            value=selection.beta_plane,
            reason="the axisymmetric solve2 branch bypasses f/beta-plane dispatch",
            remedy="use beta_plane=0 for the bounded axisymmetric u-term operator",
        )
    for name in (
        "fcor_s_inv",
        "u_move_m_s",
        "v_move_m_s",
        "center_y_m",
    ):
        _f32_scalar(np, name, getattr(selection, name))


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
    shapes = rotation_forcing_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing rotation arrays: {', '.join(missing)}")
    values = []
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
        values.append((name, value))
    for name in ("uten", "vten", "f2d"):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    f"rotation arrays must not alias: {left_name}, {right_name}"
                )


def _cm1_constants(np: Any) -> tuple[Any, Any, Any]:
    pi = np.float32("3.1415926535897932384626433")
    omega_numerator = np.float32(np.float32(2.0) * pi)
    omega = np.float32(omega_numerator / np.float32(86400.0))
    earth_radius = np.float32(6370000.0)
    return pi, omega, earth_radius


def _build_f2d_inplace(
    np: Any,
    f2d: Any,
    yh_window: Any,
    selection: RotationForcingSelection,
) -> dict[str, int | str]:
    fcor = _f32_scalar(np, "fcor_s_inv", selection.fcor_s_inv)
    f2d.fill(fcor)
    result: dict[str, int | str] = {
        "fcor_f32_bits": int(fcor.view(np.uint32)),
        "libm_claim": "not used for f-plane",
    }
    if selection.beta_plane == 1 and selection.axisymmetric == 0:
        _, omega, earth_radius = _cm1_constants(np)
        denominator = np.float32(np.float32(2.0) * omega)
        ratio = np.float32(fcor / denominator)
        if abs(float(ratio)) > 1.0:
            raise ValueError("fcor/(2*omega) must be within asin domain")
        phi = np.float32(np.arcsin(ratio))
        first = np.float32(np.float32(2.0) * omega)
        beta0 = np.float32(first * np.float32(np.cos(phi)))
        beta0 = np.float32(beta0 / earth_radius)
        center_y = _f32_scalar(np, "center_y_m", selection.center_y_m)
        delta = np.empty_like(yh_window)
        np.subtract(yh_window, center_y, out=delta)
        np.multiply(delta, beta0, out=delta)
        np.add(delta, fcor, out=delta)
        f2d[...] = delta[:, None]
        result.update(
            {
                "phi_f32_bits": int(phi.view(np.uint32)),
                "beta0_f32_bits": int(beta0.view(np.uint32)),
                "libm_claim": (
                    "native GNU Fortran raw-word identity is separately sealed by "
                    "the all-branch oracle"
                ),
            }
        )
    return result


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cpu_rotation_forcing_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RotationForcingSelection,
) -> RotationForcingResult:
    """Apply one active CM1 rotation branch to compact float32 carriers."""

    np = importlib.import_module("numpy")
    _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)
    fcor = _f32_scalar(np, "fcor_s_inv", selection.fcor_s_inv)
    u_move = _f32_scalar(np, "u_move_m_s", selection.u_move_m_s)
    v_move = _f32_scalar(np, "v_move_m_s", selection.v_move_m_s)
    f2d_audit = _build_f2d_inplace(
        np, arrays["f2d"], arrays["yh_window"], selection
    )
    u3d = arrays["u3d_window"]
    v3d = arrays["v3d_window"]
    uten = arrays["uten"]
    vten = arrays["vten"]

    if selection.axisymmetric == 1:
        # solve2.F:680-688. Only j=1..nj is written; vten is untouched.
        velocity = np.empty((nk, nj, ni + 1), dtype=np.float32)
        np.add(v3d[:, :nj, 1:], v3d[:, :nj, :-1], out=velocity)
        coefficient = np.float32(fcor * np.float32(0.5))
        np.multiply(velocity, coefficient, out=velocity)
        np.add(uten[:, :nj, :], velocity, out=uten[:, :nj, :])
        branch = "axisymmetric-u"
    elif selection.beta_plane == 0:
        # solve2.F:639-653. Preserve the two parenthesized pair sums.
        first = np.empty_like(uten)
        second = np.empty_like(uten)
        np.add(v3d[:, :-1, 1:], v3d[:, 1:, 1:], out=first)
        np.add(v3d[:, :-1, :-1], v3d[:, 1:, :-1], out=second)
        np.add(first, second, out=first)
        np.multiply(first, np.float32(0.25), out=first)
        np.add(first, v_move, out=first)
        np.multiply(first, fcor, out=first)
        np.add(uten, first, out=uten)

        np.add(u3d[:, 1:, :-1], u3d[:, 1:, 1:], out=first)
        np.add(u3d[:, :-1, :-1], u3d[:, :-1, 1:], out=second)
        np.add(first, second, out=first)
        np.multiply(first, np.float32(0.25), out=first)
        np.add(first, u_move, out=first)
        np.multiply(first, fcor, out=first)
        np.subtract(vten, first, out=vten)
        branch = "cartesian-f-plane"
    else:
        # solve2.F:655-673. The beta branch intentionally has no moving-frame
        # offsets in CM1 r21.1; preserve that source behavior.
        first = np.empty_like(uten)
        second = np.empty_like(uten)
        np.add(v3d[:, :-1, 1:], v3d[:, 1:, 1:], out=first)
        np.add(v3d[:, :-1, :-1], v3d[:, 1:, :-1], out=second)
        np.add(first, second, out=first)
        f_sum = np.empty((nj + 1, ni + 1), dtype=np.float32)
        np.add(arrays["f2d"][1:, 1:], arrays["f2d"][1:, :-1], out=f_sum)
        coefficient = np.empty_like(f_sum)
        np.multiply(f_sum, np.float32(0.125), out=coefficient)
        np.multiply(first, coefficient[None, :, :], out=first)
        np.add(uten, first, out=uten)

        np.add(u3d[:, 1:, :-1], u3d[:, 1:, 1:], out=first)
        np.add(u3d[:, :-1, :-1], u3d[:, :-1, 1:], out=second)
        np.add(first, second, out=first)
        np.add(arrays["f2d"][1:, 1:], arrays["f2d"][:-1, 1:], out=f_sum)
        np.multiply(f_sum, np.float32(0.125), out=coefficient)
        np.multiply(first, coefficient[None, :, :], out=first)
        np.subtract(vten, first, out=vten)
        branch = "cartesian-beta-plane"

    receipt: dict[str, object] = {
        "schema": CM1_ROTATION_FORCING_SCHEMA,
        "status": CM1_ROTATION_FORCING_STATUS,
        "claim_limit": CM1_ROTATION_FORCING_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_ROTATION_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_ROTATION_SOURCE_ANCHORS),
        },
        "selection": {
            "icor": selection.icor,
            "beta_plane": selection.beta_plane,
            "axisymmetric": selection.axisymmetric,
            "branch": branch,
            "u_move_f32_bits": int(u_move.view(np.uint32)),
            "v_move_f32_bits": int(v_move.view(np.uint32)),
            **f2d_audit,
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "output_sha256": {
            name: _sha256_array(arrays[name]) for name in ("f2d", "uten", "vten")
        },
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return RotationForcingResult(
        f2d=arrays["f2d"], receipt=MappingProxyType(receipt)
    )


__all__ = [
    "CM1_ROTATION_AUTHORITY_SHA256",
    "CM1_ROTATION_FORCING_CLAIM_LIMIT",
    "CM1_ROTATION_FORCING_SCHEMA",
    "CM1_ROTATION_FORCING_STATUS",
    "CM1_ROTATION_SOURCE_ANCHORS",
    "RotationForcingResult",
    "RotationForcingSelection",
    "cpu_rotation_forcing_inplace",
    "rotation_forcing_shapes",
]
