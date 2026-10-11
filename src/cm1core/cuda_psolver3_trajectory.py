"""Flat-periodic CM1 r21.1 dry trajectory with the psolver=3 acoustic path.

This is a bounded executable composition gate.  It preserves CM1's separate
large-step (``*a``) and current-stage carriers, evaluates source-ordered
fifth-order advection on the staggered grid, applies dry buoyancy, executes
all 2 + 3 + 6 psolver=3 acoustic small steps, and performs the solve3-style
``ppx`` completion and carrier promotion.  The specialization is Cartesian,
uniform, serial-periodic in x/y, rigid in z, dry, and closure-free.  It is not
a native-CM1 oracle, a terrain/open-boundary implementation, or a forecast.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from .cuda_sound3_periodic import (
    ALL_ARRAY_NAMES as SOUND_ARRAY_NAMES,
    CM1_SOUND3_PERIODIC_ALPH_F32,
    CM1_SOUND3_PERIODIC_CP_F32,
    CM1_SOUND3_PERIODIC_KDIV_F32,
    CM1_SOUND3_PERIODIC_SMEPS_F32,
    CompiledSound3PeriodicKernels,
    array_shapes as sound_array_shapes,
    compile_cuda_sound3_periodic,
    default_periodic_face_weights,
    fill_periodic_scalar_inplace,
    scalar_shape_zyx,
)


CUDA_PSOLVER3_TRAJECTORY_SCHEMA = "gpuwm-cm1.cuda-psolver3-trajectory/v1"
CUDA_PSOLVER3_TRAJECTORY_ABI = "gpuwm-cm1.cuda-psolver3-trajectory-kernels/v1"
CUDA_PSOLVER3_TRAJECTORY_THREADS = 128
CM1_GRAVITY_F32 = 9.81
CM1_RD_F32 = 287.04
CM1_CV_F32 = 1005.7 - 287.04
CM1_RDDCV_F32 = CM1_RD_F32 / CM1_CV_F32
CM1_TH0R_F32 = 300.0

CUDA_PSOLVER3_TRAJECTORY_CLAIM_LIMIT = (
    "CM1 r21.1 dry, flat, uniform, serial-periodic x/y and rigid-z "
    "composition gate with current/A carriers, source-ordered three-stage "
    "RK, fifth-order sadv=(th0-300)+th3d scalar and staggered-velocity "
    "advection including "
    "vertical lower-order boundary fallbacks, dry buoyancy, and all 2+3+6 "
    "psolver=3 acoustic small steps; held external tendencies, diffusion, "
    "Coriolis, damping, SGS, moisture/eqtset2, mass correction, terrain, "
    "open/radiative boundaries, native parity, product, forecast, and "
    "performance claims are excluded"
)

SOURCE_ANCHORS = {
    "rk_schedule": "src/solve2.F:294-299",
    "dry_thermodynamics": "src/solve2.F:434-449",
    "stage_mass_flux_and_divergence": "src/solve2.F:490-628",
    "momentum_advection_calls": "src/solve2.F:883-892",
    "buoyancy_application": "src/solve2.F:895-921",
    "pressure_dispatch": "src/solve2.F:1119-1138",
    "theta_advection_update": "src/solve2.F:1303-1307;1310-1426",
    "theta_advection_carrier": "src/solve2.F:1354-1399; src/constants.F:15",
    "stage_boundaries": "src/solve2.F:1665-1706",
    "hadv5_staggered_fluxes": "src/adv_routines.F:1335-1476",
    "vadv5_and_fallbacks": (
        "src/adv_routines.F:3269-3388;4817-4995;5001-5169"
    ),
    "scalar_tendency": "src/adv.F:270-283;450-659",
    "u_tendency": "src/adv.F:1128-1139;1254-1479",
    "v_tendency": "src/adv.F:2065-2077;2192-2414",
    "w_tendency": "src/adv.F:3017-3029;3192-3359",
    "sound3": "src/sound.F:81-113;191-248;255-853",
    "post_step_ppx": "src/solve3.F:270-305",
    "post_step_promotion": "src/solve3.F:548-620",
    "fresh_initial_ppx": "src/init3d.F:2535-2575",
    "base_face_density": "src/base.F:2171-2190",
}

EXTRA_ARRAY_NAMES = (
    "pi0",
    "rho0",
    "rrf0",
    "th0",
    "rth0",
    "tha",
    "th3d",
    "sadv",
    "thten",
    "divx",
)
ALL_ARRAY_NAMES = SOUND_ARRAY_NAMES + EXTRA_ARRAY_NAMES
STATE_OUTPUT_NAMES = (
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "ppx",
    "tha",
    "th3d",
    "sadv",
    "rru",
    "rrv",
    "rrw",
    "uten",
    "vten",
    "wten",
    "thten",
    "divx",
    "tri_pivot",
    "tri_residual",
)


def trajectory_array_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    shapes = sound_array_shapes(ni, nj, nk)
    scalar = scalar_shape_zyx(ni, nj, nk)
    shapes.update({name: scalar for name in EXTRA_ARRAY_NAMES})
    return shapes


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = trajectory_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver3 trajectory arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, got {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite float32 values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("psolver3 trajectory arrays must not alias")
    return result


def _bits_equal(left: Any, right: Any) -> bool:
    np = importlib.import_module("numpy")
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def validate_trajectory_contract(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    """Validate the bounded layout, periodicity, metrics, and carrier identity."""

    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    expected_u, expected_v = default_periodic_face_weights(ni, nj)
    if not _bits_equal(a["dtu0"], expected_u) or not _bits_equal(
        a["dtv0"], expected_v
    ):
        raise ValueError("dtu0/dtv0 must be serial-periodic default weights")
    for name in ("uf", "vf", "uh", "vh", "mh", "mf"):
        if not _bits_equal(a[name], np.ones_like(a[name])):
            raise ValueError(f"{name} must be exactly one in the uniform-flat gate")
    profile_names = (
        "pi0",
        "rho0",
        "rr0",
        "rf0",
        "rrf0",
        "th0",
        "rth0",
        "c1",
        "c2",
    )
    for name in profile_names:
        profile = a[name][:, 1:2, 1:2]
        if not _bits_equal(a[name], np.broadcast_to(profile, a[name].shape)):
            raise ValueError(
                f"{name} must be an exact horizontally uniform source profile"
            )
    for name in ("pi0", "rho0", "rr0", "rf0", "rrf0", "th0", "rth0"):
        if not bool(np.all(a[name][:, 1, 1] > np.float32(0.0))):
            raise ValueError(f"{name} source profile must be strictly positive")
    for name in ("c1", "c2"):
        if not bool(
            np.all(a[name][:, 1, 1] >= np.float32(0.0))
            and np.all(a[name][:, 1, 1] <= np.float32(1.0))
        ):
            raise ValueError(f"{name} source weights must lie in [0, 1]")
    weight_sum = np.add(a["c1"], a["c2"], dtype=np.float32)
    if not _bits_equal(weight_sum, np.ones_like(weight_sum)):
        raise ValueError("c1/c2 source weights must sum exactly to float32 one")
    for name in ("ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "tha", "th3d"):
        peer = {
            "ua": "u3d",
            "u3d": "ua",
            "va": "v3d",
            "v3d": "va",
            "wa": "w3d",
            "w3d": "wa",
            "ppi": "pp3d",
            "pp3d": "ppi",
            "tha": "th3d",
            "th3d": "tha",
        }[name]
        if not _bits_equal(a[name], a[peer]):
            raise ValueError(f"entry current/A carriers differ: {name} versus {peer}")
    for name in ("pi0", "rho0", "rr0", "rf0", "rrf0", "th0", "rth0", "ppi", "ppx", "tha"):
        value = a[name]
        if not _bits_equal(value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]):
            raise ValueError(f"{name} west periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]):
            raise ValueError(f"{name} east periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]):
            raise ValueError(f"{name} south periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]):
            raise ValueError(f"{name} north periodic halo is invalid")
    if not _bits_equal(
        a["u3d"][1 : nk + 1, 1 : nj + 1, 0],
        a["u3d"][1 : nk + 1, 1 : nj + 1, ni],
    ):
        raise ValueError("u duplicate periodic x faces differ")
    if not _bits_equal(
        a["v3d"][1 : nk + 1, 0, 1 : ni + 1],
        a["v3d"][1 : nk + 1, nj, 1 : ni + 1],
    ):
        raise ValueError("v duplicate periodic y faces differ")
    for name in ("w3d", "wa"):
        if np.any(a[name][1].view(np.uint32)) or np.any(a[name][nk + 1].view(np.uint32)):
            raise ValueError(f"{name} violates rigid bottom/top zero faces")
    if not _bits_equal(a["ppx"], a["pp3d"]):
        raise ValueError(
            "fresh entry ppx must be an exact non-aliasing copy of pp3d"
        )


def _validate_run_parameters(
    *, ni: int, nj: int, nk: int, dt: float, nsound: int, dx: float, dy: float, dz: float
) -> None:
    """Reject dimensions or timesteps outside this fifth-order bounded gate."""

    np = importlib.import_module("numpy")
    if ni < 5 or nj < 5 or nk < 6:
        raise ValueError("fifth-order trajectory requires ni >= 5, nj >= 5, nk >= 6")
    for name, value in (("dt", dt), ("dx", dx), ("dy", dy), ("dz", dz)):
        value32 = np.float32(value)
        if not bool(np.isfinite(value32)) or not bool(value32 > np.float32(0.0)):
            raise ValueError(f"{name} must be finite and positive in float32")
    for nrk in (1, 2, 3):
        sound3_stage_schedule_f32(dt, nsound, nrk)


def _add(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _sub(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _mul(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


def _div(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) / np.float32(y))


def _flx3(np: Any, s1: Any, s2: Any, s3: Any) -> Any:
    result = _mul(np, _div(np, np.float32(-1.0), np.float32(6.0)), s1)
    result = _add(np, result, _mul(np, _div(np, np.float32(5.0), np.float32(6.0)), s2))
    return _add(np, result, _mul(np, _div(np, np.float32(2.0), np.float32(6.0)), s3))


def _flx5(np: Any, s1: Any, s2: Any, s3: Any, s4: Any, s5: Any) -> Any:
    coefficients = (2.0, -13.0, 47.0, 27.0, -3.0)
    values = (s1, s2, s3, s4, s5)
    result = _mul(
        np,
        _div(np, np.float32(coefficients[0]), np.float32(60.0)),
        values[0],
    )
    for coefficient, value in zip(coefficients[1:], values[1:], strict=True):
        result = _add(
            np,
            result,
            _mul(np, _div(np, np.float32(coefficient), np.float32(60.0)), value),
        )
    return result


def _upwind5(np: Any, velocity: Any, values: tuple[Any, Any, Any, Any, Any]) -> Any:
    reconstructed = _flx5(np, *values)
    return _mul(np, velocity, reconstructed)


def _scalar_x_flux(np: Any, state: Any, velocity: Any, face: int, j: int, k: int) -> Any:
    ni = state.shape[2]
    if bool(velocity >= np.float32(0.0)):
        q = (face - 3, face - 2, face - 1, face, face + 1)
    else:
        q = (face + 2, face + 1, face, face - 1, face - 2)
    return _upwind5(np, velocity, tuple(state[k, j, i % ni] for i in q))


def _scalar_y_flux(np: Any, state: Any, velocity: Any, i: int, face: int, k: int) -> Any:
    nj = state.shape[1]
    if bool(velocity >= np.float32(0.0)):
        q = (face - 3, face - 2, face - 1, face, face + 1)
    else:
        q = (face + 2, face + 1, face, face - 1, face - 2)
    return _upwind5(np, velocity, tuple(state[k, j % nj, i] for j in q))


def _centered_x_flux(np: Any, state: Any, velocity: Any, between: int, j: int, k: int) -> Any:
    ni = state.shape[2]
    if bool(velocity >= np.float32(0.0)):
        q = (between - 2, between - 1, between, between + 1, between + 2)
    else:
        q = (between + 3, between + 2, between + 1, between, between - 1)
    return _upwind5(np, velocity, tuple(state[k, j, i % ni] for i in q))


def _centered_y_flux(np: Any, state: Any, velocity: Any, i: int, between: int, k: int) -> Any:
    nj = state.shape[1]
    if bool(velocity >= np.float32(0.0)):
        q = (between - 2, between - 1, between, between + 1, between + 2)
    else:
        q = (between + 3, between + 2, between + 1, between, between - 1)
    return _upwind5(np, velocity, tuple(state[k, j % nj, i] for j in q))


def _vertical_mass_flux(
    np: Any,
    state: Any,
    velocity: Any,
    face: int,
    j: int,
    i: int,
    c1: Any,
    c2: Any,
) -> Any:
    """CM1 stag=1/2/3 fifth-order vertical flux at Fortran face ``face``."""

    nk = state.shape[0]
    v = np.float32(velocity)
    if face == 2:
        if bool(v >= np.float32(0.0)):
            recon = _add(np, _mul(np, c1, state[0, j, i]), _mul(np, c2, state[1, j, i]))
        else:
            recon = _flx3(np, state[2, j, i], state[1, j, i], state[0, j, i])
    elif face == 3:
        if bool(v >= np.float32(0.0)):
            recon = _flx3(np, state[0, j, i], state[1, j, i], state[2, j, i])
        else:
            recon = _flx5(np, state[4, j, i], state[3, j, i], state[2, j, i], state[1, j, i], state[0, j, i])
    elif face == nk - 1:
        if bool(v >= np.float32(0.0)):
            recon = _flx5(np, state[nk - 5, j, i], state[nk - 4, j, i], state[nk - 3, j, i], state[nk - 2, j, i], state[nk - 1, j, i])
        else:
            recon = _flx3(np, state[nk - 1, j, i], state[nk - 2, j, i], state[nk - 3, j, i])
    elif face == nk:
        if bool(v >= np.float32(0.0)):
            recon = _flx3(np, state[nk - 3, j, i], state[nk - 2, j, i], state[nk - 1, j, i])
        else:
            recon = _add(np, _mul(np, c1, state[nk - 2, j, i]), _mul(np, c2, state[nk - 1, j, i]))
    else:
        if bool(v >= np.float32(0.0)):
            q = (face - 4, face - 3, face - 2, face - 1, face)
        else:
            q = (face + 1, face, face - 1, face - 2, face - 3)
        recon = _flx5(np, *(state[z, j, i] for z in q))
    return _mul(np, v, recon)


def _vertical_w_flux(np: Any, state: Any, velocity: Any, level: int, j: int, i: int) -> Any:
    """CM1 stag=4 fifth-order vertical flux at mass level ``level`` (1..nk)."""

    nk = state.shape[0] - 1
    v = np.float32(velocity)
    if level == 1:
        recon = (
            _mul(np, np.float32(0.5), _add(np, state[0, j, i], state[1, j, i]))
            if bool(v >= np.float32(0.0))
            else _flx3(np, state[2, j, i], state[1, j, i], state[0, j, i])
        )
    elif level == 2:
        recon = (
            _flx3(np, state[0, j, i], state[1, j, i], state[2, j, i])
            if bool(v >= np.float32(0.0))
            else _flx5(np, state[5, j, i], state[4, j, i], state[3, j, i], state[2, j, i], state[1, j, i])
        )
    elif level == nk - 1:
        recon = (
            _flx5(np, state[nk - 4, j, i], state[nk - 3, j, i], state[nk - 2, j, i], state[nk - 1, j, i], state[nk, j, i])
            if bool(v >= np.float32(0.0))
            else _flx3(np, state[nk, j, i], state[nk - 1, j, i], state[nk - 2, j, i])
        )
    elif level == nk:
        recon = (
            _flx3(np, state[nk - 2, j, i], state[nk - 1, j, i], state[nk, j, i])
            if bool(v >= np.float32(0.0))
            else _mul(np, np.float32(0.5), _add(np, state[nk - 1, j, i], state[nk, j, i]))
        )
    else:
        if bool(v >= np.float32(0.0)):
            q = (level - 3, level - 2, level - 1, level, level + 1)
        else:
            q = (level + 2, level + 1, level, level - 1, level - 2)
        recon = _flx5(np, *(state[z, j, i] for z in q))
    return _mul(np, v, recon)


@dataclass(frozen=True, slots=True)
class TrajectoryAblations:
    momentum_advection: bool = True
    scalar_advection: bool = True
    base_theta_advection: bool = True
    buoyancy: bool = True
    post_step_ppx: bool = True
    carrier_promotion: bool = True


@dataclass(frozen=True, slots=True)
class TrajectoryStageDiagnostics:
    large_step: int
    nrk: int
    dttmp_f32: float
    nloop: int
    dts_f32: float
    max_tridiagonal_residual: float
    horizontal_mass_flux_sum: float
    total_mass_flux_divergence_sum: float
    rigid_w_boundary_exact: bool


@dataclass(frozen=True, slots=True)
class CpuPsolver3TrajectoryResult:
    cadence: tuple[TrajectoryStageDiagnostics, ...]
    ledgers: tuple[dict[str, Any], ...]


def _prepare_stage_cpu(a: MutableMapping[str, Any], *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    rddcv = _div(np, np.float32(CM1_RD_F32), np.float32(CM1_CV_F32))
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                a["thv"][k, j, i] = _add(np, a["th0"][k, j, i], a["th3d"][k, j, i])
                a["piadv"][k, j, i] = _add(np, a["pi0"][k, j, i], a["pp3d"][k, j, i])
                a["ppterm"][k, j, i] = _mul(np, a["piadv"][k, j, i], rddcv)
    fill_periodic_scalar_inplace(a["thv"], ni=ni, nj=nj, nk=nk)
    fill_periodic_scalar_inplace(a["piadv"], ni=ni, nj=nj, nk=nk)
    a["ppten"].fill(np.float32(0.0))
    a["thten"].fill(np.float32(0.0))
    a["uten"].fill(np.float32(0.0))
    a["vten"].fill(np.float32(0.0))
    a["wten"].fill(np.float32(0.0))


def _instant_mass_flux_cpu(a: MutableMapping[str, Any], *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    a["rru"].fill(np.float32(0.0))
    a["rrv"].fill(np.float32(0.0))
    a["rrw"].fill(np.float32(0.0))
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for f in range(ni + 1):
                rho = a["rho0"][k, j, 1]
                a["rru"][k, j, f] = _mul(np, rho, a["u3d"][k, j, f])
        for f in range(nj + 1):
            for i in range(1, ni + 1):
                rho = a["rho0"][k, 1, i]
                a["rrv"][k, f, i] = _mul(np, rho, a["v3d"][k, f, i])
    for q in range(2, nk + 1):
        for j0 in range(nj):
            for i0 in range(ni):
                a["rrw"][q, j0, i0] = _mul(
                    np, a["rf0"][q, j0 + 1, i0 + 1], a["w3d"][q, j0, i0]
                )


def _scale_averages_cpu(a: MutableMapping[str, Any], *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for f in range(ni + 1):
                a["rru"][k, j, f] = _mul(np, a["rru"][k, j, f], a["rho0"][k, j, 1])
        for f in range(nj + 1):
            for i in range(1, ni + 1):
                a["rrv"][k, f, i] = _mul(np, a["rrv"][k, f, i], a["rho0"][k, 1, i])
    for q in range(2, nk + 1):
        for j0 in range(nj):
            for i0 in range(ni):
                a["rrw"][q, j0, i0] = _mul(
                    np, a["rrw"][q, j0, i0], a["rf0"][q, j0 + 1, i0 + 1]
                )
    a["rrw"][1].fill(np.float32(0.0))
    a["rrw"][nk + 1].fill(np.float32(0.0))


def _divergence_cpu(
    a: MutableMapping[str, Any], *, ni: int, nj: int, nk: int, rdx: Any, rdy: Any, rdz: Any
) -> None:
    np = importlib.import_module("numpy")
    a["divx"].fill(np.float32(0.0))
    smeps = np.float32(CM1_SOUND3_PERIODIC_SMEPS_F32)
    for k0 in range(nk):
        k = k0 + 1
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                dx = _mul(np, _sub(np, a["rru"][k, j, i0 + 1], a["rru"][k, j, i0]), rdx)
                dy = _mul(np, _sub(np, a["rrv"][k, j0 + 1, i], a["rrv"][k, j0, i]), rdy)
                dz = _mul(np, _sub(np, a["rrw"][k + 1, j0, i0], a["rrw"][k, j0, i0]), rdz)
                value = _add(np, _add(np, dx, dy), dz)
                if bool(np.abs(value) < smeps):
                    value = np.float32(0.0)
                a["divx"][k, j, i] = value


def _momentum_tendencies_cpu(
    a: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    rdx: Any,
    rdy: Any,
    rdz: Any,
    ablations: TrajectoryAblations,
) -> None:
    np = importlib.import_module("numpy")
    u = a["u3d"][1 : nk + 1, 1 : nj + 1, :ni]
    v = a["v3d"][1 : nk + 1, :nj, 1 : ni + 1]
    w = a["w3d"][1 : nk + 2]
    rru = a["rru"][1 : nk + 1, 1 : nj + 1, :ni]
    rrv = a["rrv"][1 : nk + 1, :nj, 1 : ni + 1]
    rrw = a["rrw"][1 : nk + 2]
    div = a["divx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    if ablations.momentum_advection:
        for k in range(nk):
            ks = k + 1
            for j in range(nj):
                js = j + 1
                for f in range(ni + 1):
                    fu = f % ni
                    east_v = _mul(np, np.float32(0.5), _add(np, rru[k, j, fu], rru[k, j, (fu + 1) % ni]))
                    west_v = _mul(np, np.float32(0.5), _add(np, rru[k, j, (fu - 1) % ni], rru[k, j, fu]))
                    fx_e = _centered_x_flux(np, u, east_v, fu, j, k)
                    fx_w = _centered_x_flux(np, u, west_v, fu - 1, j, k)
                    south_v = _mul(np, np.float32(0.5), _add(np, rrv[k, j, fu % ni], rrv[k, j, (fu - 1) % ni]))
                    north_v = _mul(np, np.float32(0.5), _add(np, rrv[k, (j + 1) % nj, fu % ni], rrv[k, (j + 1) % nj, (fu - 1) % ni]))
                    fy_s = _scalar_y_flux(np, u, south_v, fu, j, k)
                    fy_n = _scalar_y_flux(np, u, north_v, fu, j + 1, k)
                    advx = _mul(np, -_sub(np, fx_e, fx_w), rdx)
                    advy = _mul(np, -_sub(np, fy_n, fy_s), rdy)
                    advz = np.float32(0.0)
                    if nk >= 6:
                        if k == 0:
                            top = _vertical_mass_flux(np, u, _mul(np, np.float32(0.5), _add(np, rrw[1, j, fu % ni], rrw[1, j, (fu - 1) % ni])), 2, j, fu, a["c1"][2, js, (fu % ni) + 1], a["c2"][2, js, (fu % ni) + 1])
                            advz = _mul(np, -top, rdz)
                        elif k == nk - 1:
                            bottom = _vertical_mass_flux(np, u, _mul(np, np.float32(0.5), _add(np, rrw[nk - 1, j, fu % ni], rrw[nk - 1, j, (fu - 1) % ni])), nk, j, fu, a["c1"][nk, js, (fu % ni) + 1], a["c2"][nk, js, (fu % ni) + 1])
                            advz = _mul(np, bottom, rdz)
                        else:
                            face_lo = k + 1
                            face_hi = k + 2
                            vlo = _mul(np, np.float32(0.5), _add(np, rrw[face_lo - 1, j, fu % ni], rrw[face_lo - 1, j, (fu - 1) % ni]))
                            vhi = _mul(np, np.float32(0.5), _add(np, rrw[face_hi - 1, j, fu % ni], rrw[face_hi - 1, j, (fu - 1) % ni]))
                            flo = _vertical_mass_flux(np, u, vlo, face_lo, j, fu, a["c1"][face_lo, js, (fu % ni) + 1], a["c2"][face_lo, js, (fu % ni) + 1])
                            fhi = _vertical_mass_flux(np, u, vhi, face_hi, j, fu, a["c1"][face_hi, js, (fu % ni) + 1], a["c2"][face_hi, js, (fu % ni) + 1])
                            advz = _mul(np, -_sub(np, fhi, flo), rdz)
                    correction = _mul(np, u[k, j, fu], _mul(np, np.float32(0.5), _add(np, div[k, j, fu], div[k, j, (fu - 1) % ni])))
                    total = _mul(np, _add(np, _add(np, _add(np, advx, advy), advz), correction), a["rr0"][ks, js, (fu % ni) + 1])
                    a["uten"][ks, js, f] = total
        for k in range(nk):
            ks = k + 1
            for g in range(nj + 1):
                gv = g % nj
                for i in range(ni):
                    is_ = i + 1
                    west_v = _mul(np, np.float32(0.5), _add(np, rru[k, gv, i], rru[k, (gv - 1) % nj, i]))
                    east_v = _mul(np, np.float32(0.5), _add(np, rru[k, gv, (i + 1) % ni], rru[k, (gv - 1) % nj, (i + 1) % ni]))
                    fx_w = _scalar_x_flux(np, v, west_v, i, gv, k)
                    fx_e = _scalar_x_flux(np, v, east_v, i + 1, gv, k)
                    south_v = _mul(np, np.float32(0.5), _add(np, rrv[k, (gv - 1) % nj, i], rrv[k, gv, i]))
                    north_v = _mul(np, np.float32(0.5), _add(np, rrv[k, gv, i], rrv[k, (gv + 1) % nj, i]))
                    fy_s = _centered_y_flux(np, v, south_v, i, gv - 1, k)
                    fy_n = _centered_y_flux(np, v, north_v, i, gv, k)
                    advx = _mul(np, -_sub(np, fx_e, fx_w), rdx)
                    advy = _mul(np, -_sub(np, fy_n, fy_s), rdy)
                    advz = np.float32(0.0)
                    if nk >= 6:
                        if k == 0:
                            vv = _mul(np, np.float32(0.5), _add(np, rrw[1, gv, i], rrw[1, (gv - 1) % nj, i]))
                            advz = _mul(np, -_vertical_mass_flux(np, v, vv, 2, gv, i, a["c1"][2, (gv % nj) + 1, is_], a["c2"][2, (gv % nj) + 1, is_]), rdz)
                        elif k == nk - 1:
                            vv = _mul(np, np.float32(0.5), _add(np, rrw[nk - 1, gv, i], rrw[nk - 1, (gv - 1) % nj, i]))
                            advz = _mul(np, _vertical_mass_flux(np, v, vv, nk, gv, i, a["c1"][nk, (gv % nj) + 1, is_], a["c2"][nk, (gv % nj) + 1, is_]), rdz)
                        else:
                            lo, hi = k + 1, k + 2
                            vlo = _mul(np, np.float32(0.5), _add(np, rrw[lo - 1, gv, i], rrw[lo - 1, (gv - 1) % nj, i]))
                            vhi = _mul(np, np.float32(0.5), _add(np, rrw[hi - 1, gv, i], rrw[hi - 1, (gv - 1) % nj, i]))
                            flo = _vertical_mass_flux(np, v, vlo, lo, gv, i, a["c1"][lo, (gv % nj) + 1, is_], a["c2"][lo, (gv % nj) + 1, is_])
                            fhi = _vertical_mass_flux(np, v, vhi, hi, gv, i, a["c1"][hi, (gv % nj) + 1, is_], a["c2"][hi, (gv % nj) + 1, is_])
                            advz = _mul(np, -_sub(np, fhi, flo), rdz)
                    correction = _mul(np, v[k, gv, i], _mul(np, np.float32(0.5), _add(np, div[k, gv, i], div[k, (gv - 1) % nj, i])))
                    a["vten"][ks, g, is_] = _mul(np, _add(np, _add(np, _add(np, advx, advy), advz), correction), a["rr0"][ks, (gv % nj) + 1, is_])
        for q in range(1, nk):
            qs = q + 1
            for j in range(nj):
                js = j + 1
                for i in range(ni):
                    is_ = i + 1
                    c2 = a["c2"][qs, js, is_]
                    c1 = _sub(np, np.float32(1.0), c2)
                    def ux(face: int) -> Any:
                        return _add(np, _mul(np, c2, rru[q, j, face % ni]), _mul(np, c1, rru[q - 1, j, face % ni]))
                    def vy(face: int) -> Any:
                        return _add(np, _mul(np, c2, rrv[q, face % nj, i]), _mul(np, c1, rrv[q - 1, face % nj, i]))
                    fx_w = _scalar_x_flux(np, w, ux(i), i, j, q)
                    fx_e = _scalar_x_flux(np, w, ux(i + 1), i + 1, j, q)
                    fy_s = _scalar_y_flux(np, w, vy(j), i, j, q)
                    fy_n = _scalar_y_flux(np, w, vy(j + 1), i, j + 1, q)
                    advx = _mul(np, -_sub(np, fx_e, fx_w), rdx)
                    advy = _mul(np, -_sub(np, fy_n, fy_s), rdy)
                    vlo = _mul(np, np.float32(0.5), _add(np, rrw[q - 1, j, i], rrw[q, j, i]))
                    vhi = _mul(np, np.float32(0.5), _add(np, rrw[q, j, i], rrw[q + 1, j, i]))
                    flo = _vertical_w_flux(np, w, vlo, q, j, i)
                    fhi = _vertical_w_flux(np, w, vhi, q + 1, j, i)
                    advz = _mul(np, -_sub(np, fhi, flo), rdz)
                    correction = _mul(np, w[q, j, i], _add(np, _mul(np, c2, div[q, j, i]), _mul(np, c1, div[q - 1, j, i])))
                    a["wten"][qs, j, i] = _mul(np, _add(np, _add(np, _add(np, advx, advy), advz), correction), a["rrf0"][qs, js, is_])
    if ablations.buoyancy:
        gravity = np.float32(CM1_GRAVITY_F32)
        for q in range(1, nk):
            qs = q + 1
            for j in range(nj):
                js = j + 1
                for i in range(ni):
                    is_ = i + 1
                    lower = _mul(np, gravity, _mul(np, a["th3d"][q, js, is_], a["rth0"][q, js, is_]))
                    upper = _mul(np, gravity, _mul(np, a["th3d"][q + 1, js, is_], a["rth0"][q + 1, js, is_]))
                    buoyancy = _add(np, _mul(np, a["c1"][qs, js, is_], lower), _mul(np, a["c2"][qs, js, is_], upper))
                    a["wten"][qs, j, i] = _add(np, a["wten"][qs, j, i], buoyancy)


def _theta_tendency_update_cpu(
    a: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    rdx: Any,
    rdy: Any,
    rdz: Any,
    dttmp: Any,
    enabled: bool,
    include_base_theta: bool,
) -> None:
    np = importlib.import_module("numpy")
    a["sadv"].fill(np.float32(0.0))
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                base_anomaly = (
                    _sub(np, a["th0"][k, j, i], np.float32(CM1_TH0R_F32))
                    if include_base_theta
                    else np.float32(0.0)
                )
                a["sadv"][k, j, i] = _add(
                    np, base_anomaly, a["th3d"][k, j, i]
                )
    fill_periodic_scalar_inplace(a["sadv"], ni=ni, nj=nj, nk=nk)
    theta = a["sadv"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    tha = a["tha"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    rru = a["rru"][1 : nk + 1, 1 : nj + 1, :ni]
    rrv = a["rrv"][1 : nk + 1, :nj, 1 : ni + 1]
    rrw = a["rrw"][1 : nk + 2]
    div = a["divx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    smeps = np.float32(CM1_SOUND3_PERIODIC_SMEPS_F32)
    for k in range(nk):
        ks = k + 1
        for j in range(nj):
            js = j + 1
            for i in range(ni):
                is_ = i + 1
                tendency = np.float32(0.0)
                if enabled:
                    fx_w = _scalar_x_flux(np, theta, rru[k, j, i], i, j, k)
                    fx_e = _scalar_x_flux(np, theta, rru[k, j, (i + 1) % ni], i + 1, j, k)
                    fy_s = _scalar_y_flux(np, theta, rrv[k, j, i], i, j, k)
                    fy_n = _scalar_y_flux(np, theta, rrv[k, (j + 1) % nj, i], i, j + 1, k)
                    advx = _mul(np, -_sub(np, fx_e, fx_w), rdx)
                    advy = _mul(np, -_sub(np, fy_n, fy_s), rdy)
                    if k == 0:
                        advz = _mul(np, -_vertical_mass_flux(np, theta, rrw[1, j, i], 2, j, i, a["c1"][2, js, is_], a["c2"][2, js, is_]), rdz)
                    elif k == nk - 1:
                        advz = _mul(np, _vertical_mass_flux(np, theta, rrw[nk - 1, j, i], nk, j, i, a["c1"][nk, js, is_], a["c2"][nk, js, is_]), rdz)
                    else:
                        lo, hi = k + 1, k + 2
                        flo = _vertical_mass_flux(np, theta, rrw[lo - 1, j, i], lo, j, i, a["c1"][lo, js, is_], a["c2"][lo, js, is_])
                        fhi = _vertical_mass_flux(np, theta, rrw[hi - 1, j, i], hi, j, i, a["c1"][hi, js, is_], a["c2"][hi, js, is_])
                        advz = _mul(np, -_sub(np, fhi, flo), rdz)
                    correction = _mul(np, theta[k, j, i], div[k, j, i])
                    tendency = _mul(np, _add(np, _add(np, _add(np, advx, advy), advz), correction), a["rr0"][ks, js, is_])
                a["thten"][ks, js, is_] = tendency
                updated = _add(np, tha[k, j, i], _mul(np, dttmp, tendency))
                if bool(np.abs(updated) < smeps):
                    updated = np.float32(0.0)
                a["th3d"][ks, js, is_] = updated


def _max_residual(a: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> float:
    np = importlib.import_module("numpy")
    live = a["tri_residual"][2 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    return float(np.max(np.abs(live), initial=np.float32(0.0)))


def _post_step_cpu(
    a: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ablations: TrajectoryAblations,
) -> None:
    np = importlib.import_module("numpy")
    fill_periodic_scalar_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
    fill_periodic_scalar_inplace(a["th3d"], ni=ni, nj=nj, nk=nk)
    if ablations.post_step_ppx:
        live = (slice(1, nk + 1), slice(1, nj + 1), slice(1, ni + 1))
        a["ppx"][live] = np.add(a["pp3d"][live], a["ppx"][live], dtype=np.float32)
        fill_periodic_scalar_inplace(a["ppx"], ni=ni, nj=nj, nk=nk)
    if ablations.carrier_promotion:
        for target, source in (
            ("ua", "u3d"),
            ("va", "v3d"),
            ("wa", "w3d"),
            ("ppi", "pp3d"),
            ("tha", "th3d"),
        ):
            np.copyto(a[target], a[source])


def cpu_psolver3_trajectory_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    steps: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    ablations: TrajectoryAblations | None = None,
) -> CpuPsolver3TrajectoryResult:
    """Execute the independent CPU transition in native source order."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an integer >= 1")
    np = importlib.import_module("numpy")
    _validate_run_parameters(
        ni=ni, nj=nj, nk=nk, dt=dt, nsound=nsound, dx=dx, dy=dy, dz=dz
    )
    a = _require_host_arrays(arrays, ni, nj, nk)
    validate_trajectory_contract(a, ni=ni, nj=nj, nk=nk)
    ablations = TrajectoryAblations() if ablations is None else ablations
    rdx = _div(np, np.float32(1.0), np.float32(dx))
    rdy = _div(np, np.float32(1.0), np.float32(dy))
    rdz = _div(np, np.float32(1.0), np.float32(dz))
    cadence: list[TrajectoryStageDiagnostics] = []
    ledgers: list[dict[str, Any]] = []
    # Local import prevents an import cycle at module initialization.
    from .cuda_sound3_periodic import cpu_sound3_periodic_stage_inplace

    for large_step in range(1, steps + 1):
        theta_sum_before = float(
            np.sum(a["th3d"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1], dtype=np.float64)
        )
        for nrk in (1, 2, 3):
            _prepare_stage_cpu(a, ni=ni, nj=nj, nk=nk)
            _instant_mass_flux_cpu(a, ni=ni, nj=nj, nk=nk)
            _divergence_cpu(a, ni=ni, nj=nj, nk=nk, rdx=rdx, rdy=rdy, rdz=rdz)
            _momentum_tendencies_cpu(
                a,
                ni=ni,
                nj=nj,
                nk=nk,
                rdx=rdx,
                rdy=rdy,
                rdz=rdz,
                ablations=ablations,
            )
            sound = cpu_sound3_periodic_stage_inplace(
                a,
                ni=ni,
                nj=nj,
                nk=nk,
                dt=dt,
                nsound=nsound,
                nrk=nrk,
                rdx=float(rdx),
                rdy=float(rdy),
                rdz=float(rdz),
            )
            _scale_averages_cpu(a, ni=ni, nj=nj, nk=nk)
            _divergence_cpu(a, ni=ni, nj=nj, nk=nk, rdx=rdx, rdy=rdy, rdz=rdz)
            dttmp = np.float32(rk_stage_dt_f32(dt, nrk))
            _theta_tendency_update_cpu(
                a,
                ni=ni,
                nj=nj,
                nk=nk,
                rdx=rdx,
                rdy=rdy,
                rdz=rdz,
                dttmp=dttmp,
                enabled=ablations.scalar_advection,
                include_base_theta=ablations.base_theta_advection,
            )
            if nrk < 3:
                fill_periodic_scalar_inplace(a["th3d"], ni=ni, nj=nj, nk=nk)
            rigid = not bool(
                np.any(a["w3d"][1].view(np.uint32))
                or np.any(a["w3d"][nk + 1].view(np.uint32))
            )
            horizontal_sum = float(
                np.sum(
                    a["rru"][1 : nk + 1, 1 : nj + 1, 1:]
                    - a["rru"][1 : nk + 1, 1 : nj + 1, :-1],
                    dtype=np.float64,
                )
                + np.sum(
                    a["rrv"][1 : nk + 1, 1:, 1 : ni + 1]
                    - a["rrv"][1 : nk + 1, :-1, 1 : ni + 1],
                    dtype=np.float64,
                )
            )
            divergence_sum = float(
                np.sum(
                    a["divx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1],
                    dtype=np.float64,
                )
            )
            cadence.append(
                TrajectoryStageDiagnostics(
                    large_step,
                    nrk,
                    float(dttmp),
                    sound.nloop,
                    sound.dts_f32,
                    _max_residual(a, ni=ni, nj=nj, nk=nk),
                    horizontal_sum,
                    divergence_sum,
                    rigid,
                )
            )
        _post_step_cpu(a, ni=ni, nj=nj, nk=nk, ablations=ablations)
        theta_sum_after = float(
            np.sum(a["th3d"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1], dtype=np.float64)
        )
        ledgers.append(
            {
                "large_step": large_step,
                "theta_perturbation_sum_before_f64": theta_sum_before,
                "theta_perturbation_sum_after_f64": theta_sum_after,
                "theta_perturbation_sum_delta_f64": theta_sum_after - theta_sum_before,
                "rigid_w_boundary_exact": bool(
                    not np.any(a["w3d"][1].view(np.uint32))
                    and not np.any(a["w3d"][nk + 1].view(np.uint32))
                ),
                "max_tridiagonal_residual_f32": _max_residual(a, ni=ni, nj=nj, nk=nk),
                "horizontal_mass_flux_telescoping_max_abs_f64": max(
                    abs(item.horizontal_mass_flux_sum)
                    for item in cadence[-3:]
                ),
                "total_mass_flux_divergence_sum_max_abs_f64": max(
                    abs(item.total_mass_flux_divergence_sum)
                    for item in cadence[-3:]
                ),
            }
        )
    return CpuPsolver3TrajectoryResult(tuple(cadence), tuple(ledgers))


def compare_psolver3_trajectory_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare all declared trajectory outputs in strict raw-f32 mode."""

    np = importlib.import_module("numpy")
    missing = [
        name for name in STATE_OUTPUT_NAMES if name not in expected or name not in candidate
    ]
    if missing:
        raise ValueError(f"missing trajectory comparison outputs: {', '.join(missing)}")
    fields: dict[str, Any] = {}
    for name in STATE_OUTPUT_NAMES:
        left, right = expected[name], candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison values must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32"):
            raise TypeError(f"{name} comparison values must be float32")
        if left.shape != right.shape:
            raise ValueError(f"{name} comparison shapes differ")
        if not bool(np.all(np.isfinite(left))) or not bool(np.all(np.isfinite(right))):
            raise ValueError(f"{name} comparison values must be finite")
        exact = _bits_equal(left, right)
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
            "candidate_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
        }
    return {
        "mode": "raw-bit",
        "max_ulp": 0,
        "null_or_missing_is_pass": False,
        "passed": all(value["exact"] for value in fields.values()),
        "fields": fields,
    }


def compact_carrier_contract(ni: int, nj: int, nk: int) -> dict[str, Any]:
    """Public compact carrier schema for a strict Rust-owned decoded export."""

    return {
        "dtype": "float32",
        "order": "C, z/y/x",
        "state": {
            "u3d": [nk, nj, ni + 1],
            "v3d": [nk, nj + 1, ni],
            "w3d": [nk + 1, nj, ni],
            "pp3d": [nk, nj, ni],
            "th3d": [nk, nj, ni],
        },
        "base_profiles": {
            "pi0": [nk],
            "rho0": [nk],
            "th0": [nk],
        },
        "fresh_initialization": {
            "ppx": "initialized exactly from pp3d per init3d.F:2535-2575",
            "current_A": "bit-copied for u/v/w/pp/theta",
        },
        "required_flat_auxiliary": {
            "defaults_allowed": False,
            "external_authority_required": True,
            "source_profile_shapes": {
                "c1": [nk + 2],
                "c2": [nk + 2],
                "rf0": [nk + 2],
                "mh": [nk + 2],
                "mf": [nk + 2],
            },
            "expanded_storage_shape": [nk + 2, nj + 2, ni + 2],
            "expansion": "exact horizontal broadcast; no Python scientific derivation",
            "dependent_reciprocals": {
                "rr0": "source-defined reciprocal of rho0",
                "rrf0": "source-defined reciprocal of rf0",
                "rth0": "source-defined reciprocal of th0",
            },
        },
        "claim_limit": CUDA_PSOLVER3_TRAJECTORY_CLAIM_LIMIT,
    }


# CUDA implementation is appended below.  Keeping CPU and device arithmetic in
# one module makes the executable seam and raw-word comparator independently
# reviewable without changing either previously ratified operator module.

CUDA_PSOLVER3_TRAJECTORY_KERNEL_NAMES = (
    "cm1_traj_prepare_v1",
    "cm1_traj_zero_v1",
    "cm1_traj_mass_flux_v1",
    "cm1_traj_divergence_v1",
    "cm1_traj_momentum_v1",
    "cm1_traj_scale_average_v1",
    "cm1_traj_prepare_sadv_v1",
    "cm1_traj_theta_tendency_v1",
    "cm1_traj_theta_update_v1",
    "cm1_traj_ppx_complete_v1",
    "cm1_traj_copy_v1",
)

CUDA_PSOLVER3_TRAJECTORY_SOURCE = r"""// gpuwm-cm1.cuda-psolver3-trajectory-kernels/v1
static_assert(sizeof(float)==4,"CM1 default-real trajectory requires float32");
static_assert(sizeof(unsigned long long)==8,"CM1 trajectory offsets require uint64");

__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ int wrapi(int x,unsigned int n){int r=x%(int)n;return r<0?r+(int)n:r;}
__device__ __forceinline__ unsigned long long si(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long ui(unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+1u)+f;}
__device__ __forceinline__ unsigned long long vi(unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+1u)+f)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long wi(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*nj+j)*ni+i;}

__device__ __forceinline__ float fg(const float* a,int stag,int k,int j,int i,unsigned int ni,unsigned int nj){
  unsigned int jj=(unsigned int)wrapi(j,nj), ii=(unsigned int)wrapi(i,ni);
  if(stag==1) return a[si((unsigned int)k+1u,jj+1u,ii+1u,ni,nj)];
  if(stag==2) return a[ui((unsigned int)k+1u,jj+1u,ii,ni,nj)];
  if(stag==3) return a[vi((unsigned int)k+1u,jj,ii+1u,ni,nj)];
  return a[wi((unsigned int)k+1u,jj,ii,ni,nj)];
}
__device__ __forceinline__ float rug(const float* a,int k,int j,int f,unsigned int ni,unsigned int nj){return a[ui((unsigned int)k+1u,(unsigned int)wrapi(j,nj)+1u,(unsigned int)wrapi(f,ni),ni,nj)];}
__device__ __forceinline__ float rvg(const float* a,int k,int f,int i,unsigned int ni,unsigned int nj){return a[vi((unsigned int)k+1u,(unsigned int)wrapi(f,nj),(unsigned int)wrapi(i,ni)+1u,ni,nj)];}
__device__ __forceinline__ float rwg(const float* a,int q,int j,int i,unsigned int ni,unsigned int nj){return a[wi((unsigned int)q+1u,(unsigned int)wrapi(j,nj),(unsigned int)wrapi(i,ni),ni,nj)];}
__device__ __forceinline__ float dg(const float* a,int k,int j,int i,unsigned int ni,unsigned int nj){return a[si((unsigned int)k+1u,(unsigned int)wrapi(j,nj)+1u,(unsigned int)wrapi(i,ni)+1u,ni,nj)];}

__device__ __forceinline__ float flx3(float s1,float s2,float s3){
  float r=m32(d32(-1.0f,6.0f),s1); r=a32(r,m32(d32(5.0f,6.0f),s2)); return a32(r,m32(d32(2.0f,6.0f),s3));
}
__device__ __forceinline__ float flx5(float s1,float s2,float s3,float s4,float s5){
  float r=m32(d32(2.0f,60.0f),s1); r=a32(r,m32(d32(-13.0f,60.0f),s2)); r=a32(r,m32(d32(47.0f,60.0f),s3)); r=a32(r,m32(d32(27.0f,60.0f),s4)); return a32(r,m32(d32(-3.0f,60.0f),s5));
}
__device__ __forceinline__ float xflux(const float* a,int stag,float vel,int face,int j,int k,unsigned int ni,unsigned int nj){
  int q1,q2,q3,q4,q5;
  if(stag==2){if(vel>=0.0f){q1=face-2;q2=face-1;q3=face;q4=face+1;q5=face+2;}else{q1=face+3;q2=face+2;q3=face+1;q4=face;q5=face-1;}}
  else{if(vel>=0.0f){q1=face-3;q2=face-2;q3=face-1;q4=face;q5=face+1;}else{q1=face+2;q2=face+1;q3=face;q4=face-1;q5=face-2;}}
  return m32(vel,flx5(fg(a,stag,k,j,q1,ni,nj),fg(a,stag,k,j,q2,ni,nj),fg(a,stag,k,j,q3,ni,nj),fg(a,stag,k,j,q4,ni,nj),fg(a,stag,k,j,q5,ni,nj)));
}
__device__ __forceinline__ float yflux(const float* a,int stag,float vel,int i,int face,int k,unsigned int ni,unsigned int nj){
  int q1,q2,q3,q4,q5;
  if(stag==3){if(vel>=0.0f){q1=face-2;q2=face-1;q3=face;q4=face+1;q5=face+2;}else{q1=face+3;q2=face+2;q3=face+1;q4=face;q5=face-1;}}
  else{if(vel>=0.0f){q1=face-3;q2=face-2;q3=face-1;q4=face;q5=face+1;}else{q1=face+2;q2=face+1;q3=face;q4=face-1;q5=face-2;}}
  return m32(vel,flx5(fg(a,stag,k,q1,i,ni,nj),fg(a,stag,k,q2,i,ni,nj),fg(a,stag,k,q3,i,ni,nj),fg(a,stag,k,q4,i,ni,nj),fg(a,stag,k,q5,i,ni,nj)));
}
__device__ __forceinline__ float vflux(const float* a,int stag,float vel,int face,int j,int i,float c1,float c2,unsigned int ni,unsigned int nj,unsigned int nk){
  float r;
  if(face==2){r=(vel>=0.0f)?a32(m32(c1,fg(a,stag,0,j,i,ni,nj)),m32(c2,fg(a,stag,1,j,i,ni,nj))):flx3(fg(a,stag,2,j,i,ni,nj),fg(a,stag,1,j,i,ni,nj),fg(a,stag,0,j,i,ni,nj));}
  else if(face==3){r=(vel>=0.0f)?flx3(fg(a,stag,0,j,i,ni,nj),fg(a,stag,1,j,i,ni,nj),fg(a,stag,2,j,i,ni,nj)):flx5(fg(a,stag,4,j,i,ni,nj),fg(a,stag,3,j,i,ni,nj),fg(a,stag,2,j,i,ni,nj),fg(a,stag,1,j,i,ni,nj),fg(a,stag,0,j,i,ni,nj));}
  else if(face==(int)nk-1){r=(vel>=0.0f)?flx5(fg(a,stag,(int)nk-5,j,i,ni,nj),fg(a,stag,(int)nk-4,j,i,ni,nj),fg(a,stag,(int)nk-3,j,i,ni,nj),fg(a,stag,(int)nk-2,j,i,ni,nj),fg(a,stag,(int)nk-1,j,i,ni,nj)):flx3(fg(a,stag,(int)nk-1,j,i,ni,nj),fg(a,stag,(int)nk-2,j,i,ni,nj),fg(a,stag,(int)nk-3,j,i,ni,nj));}
  else if(face==(int)nk){r=(vel>=0.0f)?flx3(fg(a,stag,(int)nk-3,j,i,ni,nj),fg(a,stag,(int)nk-2,j,i,ni,nj),fg(a,stag,(int)nk-1,j,i,ni,nj)):a32(m32(c1,fg(a,stag,(int)nk-2,j,i,ni,nj)),m32(c2,fg(a,stag,(int)nk-1,j,i,ni,nj)));}
  else if(vel>=0.0f){r=flx5(fg(a,stag,face-4,j,i,ni,nj),fg(a,stag,face-3,j,i,ni,nj),fg(a,stag,face-2,j,i,ni,nj),fg(a,stag,face-1,j,i,ni,nj),fg(a,stag,face,j,i,ni,nj));}
  else{r=flx5(fg(a,stag,face+1,j,i,ni,nj),fg(a,stag,face,j,i,ni,nj),fg(a,stag,face-1,j,i,ni,nj),fg(a,stag,face-2,j,i,ni,nj),fg(a,stag,face-3,j,i,ni,nj));}
  return m32(vel,r);
}
__device__ __forceinline__ float vwflux(const float* a,float vel,int level,int j,int i,unsigned int ni,unsigned int nj,unsigned int nk){
  float r;
  if(level==1){r=(vel>=0.0f)?m32(0.5f,a32(fg(a,4,0,j,i,ni,nj),fg(a,4,1,j,i,ni,nj))):flx3(fg(a,4,2,j,i,ni,nj),fg(a,4,1,j,i,ni,nj),fg(a,4,0,j,i,ni,nj));}
  else if(level==2){r=(vel>=0.0f)?flx3(fg(a,4,0,j,i,ni,nj),fg(a,4,1,j,i,ni,nj),fg(a,4,2,j,i,ni,nj)):flx5(fg(a,4,5,j,i,ni,nj),fg(a,4,4,j,i,ni,nj),fg(a,4,3,j,i,ni,nj),fg(a,4,2,j,i,ni,nj),fg(a,4,1,j,i,ni,nj));}
  else if(level==(int)nk-1){r=(vel>=0.0f)?flx5(fg(a,4,(int)nk-4,j,i,ni,nj),fg(a,4,(int)nk-3,j,i,ni,nj),fg(a,4,(int)nk-2,j,i,ni,nj),fg(a,4,(int)nk-1,j,i,ni,nj),fg(a,4,(int)nk,j,i,ni,nj)):flx3(fg(a,4,(int)nk,j,i,ni,nj),fg(a,4,(int)nk-1,j,i,ni,nj),fg(a,4,(int)nk-2,j,i,ni,nj));}
  else if(level==(int)nk){r=(vel>=0.0f)?flx3(fg(a,4,(int)nk-2,j,i,ni,nj),fg(a,4,(int)nk-1,j,i,ni,nj),fg(a,4,(int)nk,j,i,ni,nj)):m32(0.5f,a32(fg(a,4,(int)nk-1,j,i,ni,nj),fg(a,4,(int)nk,j,i,ni,nj)));}
  else if(vel>=0.0f){r=flx5(fg(a,4,level-3,j,i,ni,nj),fg(a,4,level-2,j,i,ni,nj),fg(a,4,level-1,j,i,ni,nj),fg(a,4,level,j,i,ni,nj),fg(a,4,level+1,j,i,ni,nj));}
  else{r=flx5(fg(a,4,level+2,j,i,ni,nj),fg(a,4,level+1,j,i,ni,nj),fg(a,4,level,j,i,ni,nj),fg(a,4,level-1,j,i,ni,nj),fg(a,4,level-2,j,i,ni,nj));}
  return m32(vel,r);
}

extern "C" __global__ void cm1_traj_prepare_v1(const float* pi0,const float* pp3d,const float* th0,const float* th3d,float* piadv,float* ppterm,float* thv,unsigned int ni,unsigned int nj,unsigned int nk,float rddcv){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
  for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni)+1u;unsigned long long q=z/ni;unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;unsigned long long x=si(k,j,i,ni,nj);thv[x]=a32(th0[x],th3d[x]);piadv[x]=a32(pi0[x],pp3d[x]);ppterm[x]=m32(piadv[x],rddcv);}
}
extern "C" __global__ void cm1_traj_zero_v1(float* a,unsigned long long n){unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x;for(;x<n;x+=s)a[x]=0.0f;}
extern "C" __global__ void cm1_traj_mass_flux_v1(const float* rho0,const float* rf0,const float* u,const float* v,const float* w,float* rru,float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,nw=(unsigned long long)(nk+1u)*nj*ni,total=nu+nv+nw;
  for(unsigned long long z=first;z<total;z+=step){if(z<nu){unsigned int f=(unsigned int)(z%(ni+1u));unsigned long long q=z/(ni+1u);unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);rru[ui(k+1u,j+1u,f,ni,nj)]=m32(rho0[si(k+1u,j+1u,1u,ni,nj)],u[ui(k+1u,j+1u,f,ni,nj)]);}else if(z<nu+nv){unsigned long long zz=z-nu;unsigned int i=(unsigned int)(zz%ni);unsigned long long q=zz/ni;unsigned int f=(unsigned int)(q%(nj+1u)),k=(unsigned int)(q/(nj+1u));rrv[vi(k+1u,f,i+1u,ni,nj)]=m32(rho0[si(k+1u,1u,i+1u,ni,nj)],v[vi(k+1u,f,i+1u,ni,nj)]);}else{unsigned long long zz=z-nu-nv;unsigned int i=(unsigned int)(zz%ni);unsigned long long q=zz/ni;unsigned int j=(unsigned int)(q%nj),face=(unsigned int)(q/nj);rrw[wi(face+1u,j,i,ni,nj)]=(face==0u||face==nk)?0.0f:m32(rf0[si(face+1u,j+1u,i+1u,ni,nj)],w[wi(face+1u,j,i,ni,nj)]);}}
}
extern "C" __global__ void cm1_traj_divergence_v1(const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rdz,float smeps){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
  for(unsigned long long z=first;z<n;z+=step){int i=(int)(z%ni);unsigned long long q=z/ni;int j=(int)(q%nj),k=(int)(q/nj);float dx=m32(s32(rug(rru,k,j,i+1,ni,nj),rug(rru,k,j,i,ni,nj)),rdx),dy=m32(s32(rvg(rrv,k,j+1,i,ni,nj),rvg(rrv,k,j,i,ni,nj)),rdy),dz=m32(s32(rwg(rrw,k+1,j,i,ni,nj),rwg(rrw,k,j,i,ni,nj)),rdz);float d=a32(a32(dx,dy),dz);if(fabsf(d)<smeps)d=0.0f;divx[si((unsigned int)k+1u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)]=d;}
}
extern "C" __global__ void cm1_traj_momentum_v1(const float* u,const float* v,const float* w,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rr0,const float* rrf0,const float* c1a,const float* c2a,const float* th3d,const float* rth0,float* uten,float* vten,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rdz,float gravity,unsigned int do_adv,unsigned int do_buoy){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,nw=(unsigned long long)(nk-1u)*nj*ni,total=nu+nv+nw;
  for(unsigned long long z=first;z<total;z+=step){
    if(z<nu){int f=(int)(z%(ni+1u)),fu=f%(int)ni;unsigned long long q=z/(ni+1u);int j=(int)(q%nj),k=(int)(q/nj);float totalv=0.0f;if(do_adv){float ev=m32(0.5f,a32(rug(rru,k,j,fu,ni,nj),rug(rru,k,j,fu+1,ni,nj))),wv=m32(0.5f,a32(rug(rru,k,j,fu-1,ni,nj),rug(rru,k,j,fu,ni,nj)));float fx=xflux(u,2,ev,fu,j,k,ni,nj),fw=xflux(u,2,wv,fu-1,j,k,ni,nj);float sv=m32(0.5f,a32(rvg(rrv,k,j,fu,ni,nj),rvg(rrv,k,j,fu-1,ni,nj))),nv0=m32(0.5f,a32(rvg(rrv,k,j+1,fu,ni,nj),rvg(rrv,k,j+1,fu-1,ni,nj)));float fs=yflux(u,2,sv,fu,j,k,ni,nj),fn=yflux(u,2,nv0,fu,j+1,k,ni,nj);float az;if(k==0){float vv=m32(0.5f,a32(rwg(rrw,1,j,fu,ni,nj),rwg(rrw,1,j,fu-1,ni,nj)));az=m32(-vflux(u,2,vv,2,j,fu,c1a[si(2u,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],c2a[si(2u,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],ni,nj,nk),rdz);}else if(k==(int)nk-1){float vv=m32(0.5f,a32(rwg(rrw,(int)nk-1,j,fu,ni,nj),rwg(rrw,(int)nk-1,j,fu-1,ni,nj)));az=m32(vflux(u,2,vv,(int)nk,j,fu,c1a[si(nk,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],c2a[si(nk,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],ni,nj,nk),rdz);}else{int lo=k+1,hi=k+2;float vl=m32(0.5f,a32(rwg(rrw,lo-1,j,fu,ni,nj),rwg(rrw,lo-1,j,fu-1,ni,nj))),vh=m32(0.5f,a32(rwg(rrw,hi-1,j,fu,ni,nj),rwg(rrw,hi-1,j,fu-1,ni,nj)));az=m32(-s32(vflux(u,2,vh,hi,j,fu,c1a[si((unsigned int)hi,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],c2a[si((unsigned int)hi,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],ni,nj,nk),vflux(u,2,vl,lo,j,fu,c1a[si((unsigned int)lo,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],c2a[si((unsigned int)lo,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)],ni,nj,nk)),rdz);}float cor=m32(fg(u,2,k,j,fu,ni,nj),m32(0.5f,a32(dg(divx,k,j,fu,ni,nj),dg(divx,k,j,fu-1,ni,nj))));totalv=m32(a32(a32(a32(m32(-s32(fx,fw),rdx),m32(-s32(fn,fs),rdy)),az),cor),rr0[si((unsigned int)k+1u,(unsigned int)j+1u,(unsigned int)fu+1u,ni,nj)]);}uten[ui((unsigned int)k+1u,(unsigned int)j+1u,(unsigned int)f,ni,nj)]=totalv;}
    else if(z<nu+nv){unsigned long long zz=z-nu;int i=(int)(zz%ni);unsigned long long q=zz/ni;int g=(int)(q%(nj+1u)),gv=g%(int)nj,k=(int)(q/(nj+1u));float totalv=0.0f;if(do_adv){float wv=m32(0.5f,a32(rug(rru,k,gv,i,ni,nj),rug(rru,k,gv-1,i,ni,nj))),ev=m32(0.5f,a32(rug(rru,k,gv,i+1,ni,nj),rug(rru,k,gv-1,i+1,ni,nj)));float fw=xflux(v,3,wv,i,gv,k,ni,nj),fe=xflux(v,3,ev,i+1,gv,k,ni,nj);float sv=m32(0.5f,a32(rvg(rrv,k,gv-1,i,ni,nj),rvg(rrv,k,gv,i,ni,nj))),nv0=m32(0.5f,a32(rvg(rrv,k,gv,i,ni,nj),rvg(rrv,k,gv+1,i,ni,nj)));float fs=yflux(v,3,sv,i,gv-1,k,ni,nj),fn=yflux(v,3,nv0,i,gv,k,ni,nj);float az;if(k==0){float vv=m32(0.5f,a32(rwg(rrw,1,gv,i,ni,nj),rwg(rrw,1,gv-1,i,ni,nj)));az=m32(-vflux(v,3,vv,2,gv,i,c1a[si(2u,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],c2a[si(2u,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),rdz);}else if(k==(int)nk-1){float vv=m32(0.5f,a32(rwg(rrw,(int)nk-1,gv,i,ni,nj),rwg(rrw,(int)nk-1,gv-1,i,ni,nj)));az=m32(vflux(v,3,vv,(int)nk,gv,i,c1a[si(nk,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],c2a[si(nk,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),rdz);}else{int lo=k+1,hi=k+2;float vl=m32(0.5f,a32(rwg(rrw,lo-1,gv,i,ni,nj),rwg(rrw,lo-1,gv-1,i,ni,nj))),vh=m32(0.5f,a32(rwg(rrw,hi-1,gv,i,ni,nj),rwg(rrw,hi-1,gv-1,i,ni,nj)));az=m32(-s32(vflux(v,3,vh,hi,gv,i,c1a[si((unsigned int)hi,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],c2a[si((unsigned int)hi,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),vflux(v,3,vl,lo,gv,i,c1a[si((unsigned int)lo,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],c2a[si((unsigned int)lo,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk)),rdz);}float cor=m32(fg(v,3,k,gv,i,ni,nj),m32(0.5f,a32(dg(divx,k,gv,i,ni,nj),dg(divx,k,gv-1,i,ni,nj))));totalv=m32(a32(a32(a32(m32(-s32(fe,fw),rdx),m32(-s32(fn,fs),rdy)),az),cor),rr0[si((unsigned int)k+1u,(unsigned int)gv+1u,(unsigned int)i+1u,ni,nj)]);}vten[vi((unsigned int)k+1u,(unsigned int)g,(unsigned int)i+1u,ni,nj)]=totalv;}
    else{unsigned long long zz=z-nu-nv;int i=(int)(zz%ni);unsigned long long q0=zz/ni;int j=(int)(q0%nj),q=(int)(q0/nj)+1,ks=q+1;float totalv=0.0f;if(do_adv){float c2=c2a[si((unsigned int)ks,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],c1=s32(1.0f,c2);float uw=a32(m32(c2,rug(rru,q,j,i,ni,nj)),m32(c1,rug(rru,q-1,j,i,ni,nj))),ue=a32(m32(c2,rug(rru,q,j,i+1,ni,nj)),m32(c1,rug(rru,q-1,j,i+1,ni,nj))),vs=a32(m32(c2,rvg(rrv,q,j,i,ni,nj)),m32(c1,rvg(rrv,q-1,j,i,ni,nj))),vn=a32(m32(c2,rvg(rrv,q,j+1,i,ni,nj)),m32(c1,rvg(rrv,q-1,j+1,i,ni,nj)));float fw=xflux(w,4,uw,i,j,q,ni,nj),fe=xflux(w,4,ue,i+1,j,q,ni,nj),fs=yflux(w,4,vs,i,j,q,ni,nj),fn=yflux(w,4,vn,i,j+1,q,ni,nj);float vl=m32(0.5f,a32(rwg(rrw,q-1,j,i,ni,nj),rwg(rrw,q,j,i,ni,nj))),vh=m32(0.5f,a32(rwg(rrw,q,j,i,ni,nj),rwg(rrw,q+1,j,i,ni,nj)));float az=m32(-s32(vwflux(w,vh,q+1,j,i,ni,nj,nk),vwflux(w,vl,q,j,i,ni,nj,nk)),rdz),cor=m32(fg(w,4,q,j,i,ni,nj),a32(m32(c2,dg(divx,q,j,i,ni,nj)),m32(c1,dg(divx,q-1,j,i,ni,nj))));totalv=m32(a32(a32(a32(m32(-s32(fe,fw),rdx),m32(-s32(fn,fs),rdy)),az),cor),rrf0[si((unsigned int)ks,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)]);}if(do_buoy){float lo=m32(gravity,m32(th3d[si((unsigned int)q,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],rth0[si((unsigned int)q,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)])),hi=m32(gravity,m32(th3d[si((unsigned int)q+1u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],rth0[si((unsigned int)q+1u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)]));totalv=a32(totalv,a32(m32(c1a[si((unsigned int)ks,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],lo),m32(c2a[si((unsigned int)ks,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],hi)));}wten[wi((unsigned int)ks,(unsigned int)j,(unsigned int)i,ni,nj)]=totalv;}
  }
}
extern "C" __global__ void cm1_traj_scale_average_v1(const float* rho0,const float* rf0,float* rru,float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,nw=(unsigned long long)(nk+1u)*nj*ni,total=nu+nv+nw;
  for(unsigned long long z=first;z<total;z+=step){if(z<nu){unsigned int f=(unsigned int)(z%(ni+1u));unsigned long long q=z/(ni+1u);unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);unsigned long long x=ui(k+1u,j+1u,f,ni,nj);rru[x]=m32(rru[x],rho0[si(k+1u,j+1u,1u,ni,nj)]);}else if(z<nu+nv){unsigned long long zz=z-nu;unsigned int i=(unsigned int)(zz%ni);unsigned long long q=zz/ni;unsigned int f=(unsigned int)(q%(nj+1u)),k=(unsigned int)(q/(nj+1u));unsigned long long x=vi(k+1u,f,i+1u,ni,nj);rrv[x]=m32(rrv[x],rho0[si(k+1u,1u,i+1u,ni,nj)]);}else{unsigned long long zz=z-nu-nv;unsigned int i=(unsigned int)(zz%ni);unsigned long long q=zz/ni;unsigned int j=(unsigned int)(q%nj),face=(unsigned int)(q/nj);unsigned long long x=wi(face+1u,j,i,ni,nj);rrw[x]=(face==0u||face==nk)?0.0f:m32(rrw[x],rf0[si(face+1u,j+1u,i+1u,ni,nj)]);}}
}
extern "C" __global__ void cm1_traj_prepare_sadv_v1(const float* th0,const float* th3d,float* sadv,unsigned int ni,unsigned int nj,unsigned int nk,float th0r,unsigned int include_base){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
  for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni)+1u;unsigned long long q=z/ni;unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;unsigned long long x=si(k,j,i,ni,nj);float base=include_base?s32(th0[x],th0r):0.0f;sadv[x]=a32(base,th3d[x]);}
}
extern "C" __global__ void cm1_traj_theta_tendency_v1(const float* sadv,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rr0,const float* c1a,const float* c2a,float* thten,unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rdz,unsigned int enabled){
  unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
  for(unsigned long long z=first;z<n;z+=step){int i=(int)(z%ni);unsigned long long q=z/ni;int j=(int)(q%nj),k=(int)(q/nj);float t=0.0f;if(enabled){float fw=xflux(sadv,1,rug(rru,k,j,i,ni,nj),i,j,k,ni,nj),fe=xflux(sadv,1,rug(rru,k,j,i+1,ni,nj),i+1,j,k,ni,nj),fs=yflux(sadv,1,rvg(rrv,k,j,i,ni,nj),i,j,k,ni,nj),fn=yflux(sadv,1,rvg(rrv,k,j+1,i,ni,nj),i,j+1,k,ni,nj);float az;if(k==0)az=m32(-vflux(sadv,1,rwg(rrw,1,j,i,ni,nj),2,j,i,c1a[si(2u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],c2a[si(2u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),rdz);else if(k==(int)nk-1)az=m32(vflux(sadv,1,rwg(rrw,(int)nk-1,j,i,ni,nj),(int)nk,j,i,c1a[si(nk,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],c2a[si(nk,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),rdz);else{int lo=k+1,hi=k+2;float fl=vflux(sadv,1,rwg(rrw,lo-1,j,i,ni,nj),lo,j,i,c1a[si((unsigned int)lo,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],c2a[si((unsigned int)lo,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk),fh=vflux(sadv,1,rwg(rrw,hi-1,j,i,ni,nj),hi,j,i,c1a[si((unsigned int)hi,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],c2a[si((unsigned int)hi,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)],ni,nj,nk);az=m32(-s32(fh,fl),rdz);}t=m32(a32(a32(a32(m32(-s32(fe,fw),rdx),m32(-s32(fn,fs),rdy)),az),m32(fg(sadv,1,k,j,i,ni,nj),dg(divx,k,j,i,ni,nj))),rr0[si((unsigned int)k+1u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)]);}thten[si((unsigned int)k+1u,(unsigned int)j+1u,(unsigned int)i+1u,ni,nj)]=t;}
}
extern "C" __global__ void cm1_traj_theta_update_v1(const float* tha,const float* thten,float* th3d,unsigned int ni,unsigned int nj,unsigned int nk,float dttmp,float smeps){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni)+1u;unsigned long long q=z/ni;unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;unsigned long long x=si(k,j,i,ni,nj);float v=a32(tha[x],m32(dttmp,thten[x]));if(fabsf(v)<smeps)v=0.0f;th3d[x]=v;}}
extern "C" __global__ void cm1_traj_ppx_complete_v1(const float* pp3d,float* ppx,unsigned int ni,unsigned int nj,unsigned int nk){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni)+1u;unsigned long long q=z/ni;unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;unsigned long long x=si(k,j,i,ni,nj);ppx[x]=a32(pp3d[x],ppx[x]);}}
extern "C" __global__ void cm1_traj_copy_v1(const float* src,float* dst,unsigned long long n){unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x;for(;x<n;x+=s)dst[x]=src[x];}
"""

CUDA_PSOLVER3_TRAJECTORY_SOURCE_BYTES = CUDA_PSOLVER3_TRAJECTORY_SOURCE.encode("utf-8")
CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256 = (
    "bfa0fe9fd6c5381ec03b0cac61a5a55d8233e3253a87f16bb51d2827f5a98d9a"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return common + (f"--gpu-architecture=compute_{arch}",), common + (
        f"--gpu-architecture=sm_{arch}",
    )


@dataclass(frozen=True, slots=True)
class CompiledPsolver3TrajectoryKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_psolver3_trajectory() -> CompiledPsolver3TrajectoryKernels:
    digest = hashlib.sha256(CUDA_PSOLVER3_TRAJECTORY_SOURCE_BYTES).hexdigest()
    if digest != CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256:
        raise RuntimeError("packaged psolver3 trajectory CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_PSOLVER3_TRAJECTORY_SOURCE,
        "cm1_psolver3_trajectory_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_PSOLVER3_TRAJECTORY_SOURCE,
        "cm1_psolver3_trajectory_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", text)
        for name in CUDA_PSOLVER3_TRAJECTORY_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks psolver3 trajectory entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_PSOLVER3_TRAJECTORY_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", name),
    )
    return CompiledPsolver3TrajectoryKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = trajectory_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver3 trajectory arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if (
            value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("psolver3 trajectory device arrays must not alias")
    return result


def _grid(count: int) -> tuple[int, int, int]:
    return (
        (count + CUDA_PSOLVER3_TRAJECTORY_THREADS - 1)
        // CUDA_PSOLVER3_TRAJECTORY_THREADS,
        1,
        1,
    )


def _enqueue_sound_stage(
    cupy: Any,
    a: Mapping[str, Any],
    compiled: CompiledSound3PeriodicKernels,
    *,
    stream: Any,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    rdx: Any,
    rdy: Any,
    rdz: Any,
) -> tuple[int, float]:
    """Enqueue one already-ratified periodic sound stage without synchronizing."""

    kprep, kprepuv, kupdate, kcolumn, kavg, kbcp, kbcuv = compiled.kernels
    f32, u32 = cupy.float32, cupy.uint32
    alph = f32(CM1_SOUND3_PERIODIC_ALPH_F32)
    beta = f32(f32(1.0) - alph)
    asq = f32(alph * alph)
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    face_words = max(
        (nk + 2) * (nj + 2) * (ni + 1),
        (nk + 2) * (nj + 1) * (ni + 2),
    )
    grid_prep = _grid(max(scalar_words, ni * nj))
    grid_faces = _grid(face_words)
    grid_cols = _grid(ni * nj)
    block = (CUDA_PSOLVER3_TRAJECTORY_THREADS, 1, 1)
    kprep(
        grid_prep,
        block,
        (
            a["mh"],
            a["rr0"],
            a["rf0"],
            a["c1"],
            a["c2"],
            a["mf"],
            a["piadv"],
            a["ppterm"],
            a["thv"],
            a["ppi"],
            a["ppx"],
            a["wa"],
            a["w3d"],
            a["pp3d"],
            a["ppd"],
            a["rrw"],
            a["qk"],
            a["pk1"],
            a["pk2"],
            a["mm"],
            a["fpk"],
            a["divh"],
            a["hadvp"],
            a["elim_c"],
            a["elim_d"],
            a["tri_aa"],
            a["tri_bb"],
            a["tri_cc"],
            a["tri_dd"],
            a["tri_pivot"],
            a["tri_residual"],
            u32(ni),
            u32(nj),
            u32(nk),
            u32(nrk),
            f32(dts),
            f32(rdz),
            f32(CM1_SOUND3_PERIODIC_CP_F32),
            beta,
        ),
        stream=stream,
    )
    kprepuv(
        grid_faces,
        block,
        (
            a["ua"],
            a["va"],
            a["u3d"],
            a["v3d"],
            a["rru"],
            a["rrv"],
            u32(ni),
            u32(nj),
            u32(nk),
            u32(nrk),
        ),
        stream=stream,
    )
    for n in range(1, nloop + 1):
        kupdate(
            grid_faces,
            block,
            (
                a["ppd"],
                a["thv"],
                a["uten"],
                a["vten"],
                a["dtu0"],
                a["dtv0"],
                a["uf"],
                a["vf"],
                a["u3d"],
                a["v3d"],
                u32(ni),
                u32(nj),
                u32(nk),
                f32(dts),
                f32(rdx),
                f32(rdy),
                f32(CM1_SOUND3_PERIODIC_CP_F32),
            ),
            stream=stream,
        )
        kcolumn(
            grid_cols,
            block,
            (
                a["piadv"],
                a["ppterm"],
                a["ppten"],
                a["wten"],
                a["u3d"],
                a["v3d"],
                a["uh"],
                a["vh"],
                a["w3d"],
                a["pp3d"],
                a["ppx"],
                a["ppd"],
                a["rrw"],
                a["qk"],
                a["pk1"],
                a["pk2"],
                a["mm"],
                a["fpk"],
                a["divh"],
                a["hadvp"],
                a["elim_c"],
                a["elim_d"],
                a["tri_aa"],
                a["tri_bb"],
                a["tri_cc"],
                a["tri_dd"],
                a["tri_pivot"],
                a["tri_residual"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(nrk),
                u32(n),
                u32(nloop),
                f32(dts),
                f32(rdx),
                f32(rdy),
                alph,
                beta,
                asq,
                f32(CM1_SOUND3_PERIODIC_KDIV_F32),
                f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
            ),
            stream=stream,
        )
        if n < nloop:
            kbcp(
                (1, 1, 1),
                (1, 1, 1),
                (a["ppd"], u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
        kavg(
            grid_faces,
            block,
            (
                a["u3d"],
                a["v3d"],
                a["rru"],
                a["rrv"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(n),
                u32(nloop),
            ),
            stream=stream,
        )
    kbcuv(
        (1, 1, 1),
        (1, 1, 1),
        (a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk)),
        stream=stream,
    )
    if nrk < 3:
        kbcp(
            (1, 1, 1),
            (1, 1, 1),
            (a["pp3d"], u32(ni), u32(nj), u32(nk)),
            stream=stream,
        )
    return nloop, float(f32(dts))


@dataclass(frozen=True, slots=True)
class CudaPsolver3TrajectoryLaunch:
    ready_event: Any
    compiled: CompiledPsolver3TrajectoryKernels
    sound_compiled: CompiledSound3PeriodicKernels
    receipt: dict[str, Any]


def launch_cuda_psolver3_trajectory(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    steps: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    ablations: TrajectoryAblations | None = None,
    stream: Any | None = None,
    compiled: CompiledPsolver3TrajectoryKernels | None = None,
    sound_compiled: CompiledSound3PeriodicKernels | None = None,
) -> CudaPsolver3TrajectoryLaunch:
    """Enqueue a device-resident one/multi-large-step trajectory."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an integer >= 1")
    _validate_run_parameters(
        ni=ni, nj=nj, nk=nk, dt=dt, nsound=nsound, dx=dx, dy=dy, dz=dz
    )
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    compiled = compile_cuda_psolver3_trajectory() if compiled is None else compiled
    sound_compiled = (
        compile_cuda_sound3_periodic() if sound_compiled is None else sound_compiled
    )
    if compiled.target_arch != sound_compiled.target_arch:
        raise ValueError("trajectory and sound kernels target different CUDA architectures")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    ablations = TrajectoryAblations() if ablations is None else ablations
    (
        kprepare,
        kzero,
        kmass,
        kdiv,
        kmomentum,
        kscale,
        ksadv,
        ktheta,
        ktheta_update,
        kppx,
        kcopy,
    ) = compiled.kernels
    *_, kbcp, _ = sound_compiled.kernels
    f32, u32, u64 = cupy.float32, cupy.uint32, cupy.uint64
    rdx = f32(f32(1.0) / f32(dx))
    rdy = f32(f32(1.0) / f32(dy))
    rdz = f32(f32(1.0) / f32(dz))
    rddcv = f32(f32(CM1_RD_F32) / f32(CM1_CV_F32))
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    u_words = (nk + 2) * (nj + 2) * (ni + 1)
    v_words = (nk + 2) * (nj + 1) * (ni + 2)
    w_words = (nk + 2) * nj * ni
    physical = ni * nj * nk
    face_total = nk * nj * (ni + 1) + nk * (nj + 1) * ni + (nk + 1) * nj * ni
    momentum_total = nk * nj * (ni + 1) + nk * (nj + 1) * ni + (nk - 1) * nj * ni
    block = (CUDA_PSOLVER3_TRAJECTORY_THREADS, 1, 1)
    cadence: list[dict[str, Any]] = []
    for large_step in range(1, steps + 1):
        for nrk in (1, 2, 3):
            for value, count in (
                (a["ppten"], scalar_words),
                (a["thten"], scalar_words),
                (a["uten"], u_words),
                (a["vten"], v_words),
                (a["wten"], w_words),
            ):
                kzero(_grid(count), block, (value, u64(count)), stream=stream)
            kprepare(
                _grid(physical),
                block,
                (
                    a["pi0"],
                    a["pp3d"],
                    a["th0"],
                    a["th3d"],
                    a["piadv"],
                    a["ppterm"],
                    a["thv"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    rddcv,
                ),
                stream=stream,
            )
            for value in (a["piadv"], a["thv"]):
                kbcp(
                    (1, 1, 1),
                    (1, 1, 1),
                    (value, u32(ni), u32(nj), u32(nk)),
                    stream=stream,
                )
            kmass(
                _grid(face_total),
                block,
                (
                    a["rho0"],
                    a["rf0"],
                    a["u3d"],
                    a["v3d"],
                    a["w3d"],
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                ),
                stream=stream,
            )
            kdiv(
                _grid(physical),
                block,
                (
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    a["divx"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    rdx,
                    rdy,
                    rdz,
                    f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
                ),
                stream=stream,
            )
            kmomentum(
                _grid(momentum_total),
                block,
                (
                    a["u3d"],
                    a["v3d"],
                    a["w3d"],
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    a["divx"],
                    a["rr0"],
                    a["rrf0"],
                    a["c1"],
                    a["c2"],
                    a["th3d"],
                    a["rth0"],
                    a["uten"],
                    a["vten"],
                    a["wten"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    rdx,
                    rdy,
                    rdz,
                    f32(CM1_GRAVITY_F32),
                    u32(int(ablations.momentum_advection)),
                    u32(int(ablations.buoyancy)),
                ),
                stream=stream,
            )
            nloop, dts = _enqueue_sound_stage(
                cupy,
                a,
                sound_compiled,
                stream=stream,
                ni=ni,
                nj=nj,
                nk=nk,
                dt=dt,
                nsound=nsound,
                nrk=nrk,
                rdx=rdx,
                rdy=rdy,
                rdz=rdz,
            )
            kscale(
                _grid(face_total),
                block,
                (
                    a["rho0"],
                    a["rf0"],
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                ),
                stream=stream,
            )
            kdiv(
                _grid(physical),
                block,
                (
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    a["divx"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    rdx,
                    rdy,
                    rdz,
                    f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
                ),
                stream=stream,
            )
            ksadv(
                _grid(physical),
                block,
                (
                    a["th0"],
                    a["th3d"],
                    a["sadv"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    f32(CM1_TH0R_F32),
                    u32(int(ablations.base_theta_advection)),
                ),
                stream=stream,
            )
            kbcp(
                (1, 1, 1),
                (1, 1, 1),
                (a["sadv"], u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
            ktheta(
                _grid(physical),
                block,
                (
                    a["sadv"],
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    a["divx"],
                    a["rr0"],
                    a["c1"],
                    a["c2"],
                    a["thten"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    rdx,
                    rdy,
                    rdz,
                    u32(int(ablations.scalar_advection)),
                ),
                stream=stream,
            )
            dttmp = f32(rk_stage_dt_f32(dt, nrk))
            ktheta_update(
                _grid(physical),
                block,
                (
                    a["tha"],
                    a["thten"],
                    a["th3d"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    dttmp,
                    f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
                ),
                stream=stream,
            )
            if nrk < 3:
                kbcp(
                    (1, 1, 1),
                    (1, 1, 1),
                    (a["th3d"], u32(ni), u32(nj), u32(nk)),
                    stream=stream,
                )
            cadence.append(
                {
                    "large_step": large_step,
                    "nrk": nrk,
                    "dttmp_f32": float(dttmp),
                    "nloop": nloop,
                    "dts_f32": dts,
                }
            )
        for value in (a["pp3d"], a["th3d"]):
            kbcp(
                (1, 1, 1),
                (1, 1, 1),
                (value, u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
        if ablations.post_step_ppx:
            kppx(
                _grid(physical),
                block,
                (a["pp3d"], a["ppx"], u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
            kbcp(
                (1, 1, 1),
                (1, 1, 1),
                (a["ppx"], u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
        if ablations.carrier_promotion:
            for source, target, count in (
                (a["u3d"], a["ua"], u_words),
                (a["v3d"], a["va"], v_words),
                (a["w3d"], a["wa"], w_words),
                (a["pp3d"], a["ppi"], scalar_words),
                (a["th3d"], a["tha"], scalar_words),
            ):
                kcopy(
                    _grid(count),
                    block,
                    (source, target, u64(count)),
                    stream=stream,
                )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_PSOLVER3_TRAJECTORY_SCHEMA,
        "claim_limit": CUDA_PSOLVER3_TRAJECTORY_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "steps": steps,
            "dx_m": float(f32(dx)),
            "dy_m": float(f32(dy)),
            "dz_m": float(f32(dz)),
            "dt_s": float(f32(dt)),
            "boundary": "serial-periodic x/y; rigid zero w at bottom/top",
        },
        "cadence": cadence,
        "ablations": {
            "momentum_advection": ablations.momentum_advection,
            "scalar_advection": ablations.scalar_advection,
            "base_theta_advection": ablations.base_theta_advection,
            "buoyancy": ablations.buoyancy,
            "post_step_ppx": ablations.post_step_ppx,
            "carrier_promotion": ablations.carrier_promotion,
        },
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "trajectory_kernel": {
            "abi": CUDA_PSOLVER3_TRAJECTORY_ABI,
            "names": list(CUDA_PSOLVER3_TRAJECTORY_KERNEL_NAMES),
            "source_sha256": CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "sound_kernel": {
            "source_sha256": hashlib.sha256(
                importlib.import_module(
                    "cm1core.cuda_sound3_periodic"
                ).CUDA_SOUND3_PERIODIC_SOURCE_BYTES
            ).hexdigest(),
            "target_arch": sound_compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "carrier_interface": compact_carrier_contract(ni, nj, nk),
    }
    return CudaPsolver3TrajectoryLaunch(event, compiled, sound_compiled, receipt)


__all__ = (
    "ALL_ARRAY_NAMES",
    "CUDA_PSOLVER3_TRAJECTORY_CLAIM_LIMIT",
    "CUDA_PSOLVER3_TRAJECTORY_SCHEMA",
    "CUDA_PSOLVER3_TRAJECTORY_SOURCE_BYTES",
    "CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256",
    "STATE_OUTPUT_NAMES",
    "CpuPsolver3TrajectoryResult",
    "CudaPsolver3TrajectoryLaunch",
    "TrajectoryAblations",
    "TrajectoryStageDiagnostics",
    "compact_carrier_contract",
    "compare_psolver3_trajectory_raw",
    "compile_cuda_psolver3_trajectory",
    "cpu_psolver3_trajectory_inplace",
    "launch_cuda_psolver3_trajectory",
    "trajectory_array_shapes",
    "validate_trajectory_contract",
)
