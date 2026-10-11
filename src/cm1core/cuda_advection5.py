"""CM1 r21.1 fifth-order scalar advection on the terrain C grid.

This module is intentionally narrow.  It implements the scalar ``stag=1``
branch selected by the product profiles: fifth-order upwind-biased fluxes on
RK stages one and two and CM1's five-point WENO-Z reconstruction on the final
stage.  It also supplies the terrain mass-flux preparation used immediately
before ``advs``.  Momentum tendency assembly and positive-definite moisture
repair remain separate source slices.

CUDA is imported lazily.  The production entry points consume and produce
device arrays; Python only validates the ABI and launches kernels.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes
from cm1core.errors import refuse


CUDA_ADVECTION5_SCHEMA = "gpuwm-cm1.cuda-advection5/v1"
CUDA_ADVECTION5_ABI = "gpuwm-cm1.cuda-advection5-kernels/v1"
CUDA_ADVECTION5_THREADS = 128
CM1_WENO_SIFORM = 2
CM1_NRKMAX = 3

SOURCE_ANCHORS = {
    "selector": "src/adv.F:75-78;src/adv.F:958-960;src/adv.F:1894-1896;src/adv.F:2828-2830",
    "scalar_horizontal_centered": "src/adv_routines.F:1335-1476",
    "scalar_horizontal_weno": "src/adv_routines.F:204-430",
    "scalar_vertical_centered": "src/adv_routines.F:3269-3388;src/adv_routines.F:4817-5169",
    "scalar_vertical_weno": "src/adv_routines.F:2359-2557;src/adv_routines.F:4038-4447",
    "flx3": "src/adv_routines.F:5971-5982",
    "flx5": "src/adv_routines.F:6006-6019",
    "weno5": "src/adv_routines.F:6166-6199",
    "upstrpd": "src/adv_routines.F:6331-6348",
    "terrain_mass_flux": "src/solve2.F:490-571",
    "terrain_divergence": "src/solve2.F:613-627",
    "scalar_tendency": "src/adv.F:535-577;src/adv.F:631-659",
}

CUDA_ADVECTION5_CLAIM_LIMIT = (
    "CM1 r21.1 scalar stag=1 fifth-order centered/upwind-biased flux and "
    "final-stage WENO-Z selection, terrain mass flux, divergence, and scalar "
    "tendency on one serial uniform-horizontal grid; positive-definite pdef "
    "repair, diffusion, open-boundary inflow overrides, momentum tendency "
    "assembly, source-external base halos, coupled RK state updates, forecast, "
    "supercell, and tornado claims remain excluded"
)


@dataclass(frozen=True, slots=True)
class Advection5Selection:
    scalar_horizontal_order: int = 5
    scalar_vertical_order: int = 5
    scalar_weno: int = 2
    velocity_horizontal_order: int = 5
    velocity_vertical_order: int = 5
    velocity_weno: int = 0
    weno_order: int = 5
    weno_siform: int = CM1_WENO_SIFORM
    terrain: bool = True
    axisymmetric: int = 0
    mpi: bool = False
    positive_definite: bool = False


DEFAULT_ADVECTION5_SELECTION = Advection5Selection()
_EXPECTED_SELECTION = {
    "dynamics.scalar_horizontal_order": ("scalar_horizontal_order", 5),
    "dynamics.scalar_vertical_order": ("scalar_vertical_order", 5),
    "dynamics.scalar_weno": ("scalar_weno", 2),
    "dynamics.velocity_horizontal_order": ("velocity_horizontal_order", 5),
    "dynamics.velocity_vertical_order": ("velocity_vertical_order", 5),
    "dynamics.velocity_weno": ("velocity_weno", 0),
    "dynamics.weno_order": ("weno_order", 5),
    "dynamics.weno_siform": ("weno_siform", 2),
    "grid.terrain": ("terrain", True),
    "grid.axisymmetric": ("axisymmetric", 0),
    "runtime.mpi": ("mpi", False),
    "advection.positive_definite": ("positive_definite", False),
}


def validate_advection5_selection(selection: Advection5Selection) -> None:
    if not isinstance(selection, Advection5Selection):
        raise TypeError("selection must be an Advection5Selection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if type(value) is not type(expected) or value != expected:
            if attribute == "positive_definite":
                reason = (
                    "pdef=1 changes vertical WENO boundary behavior and invokes "
                    "separate mass/flag repair; using the non-pdef operator would "
                    "produce the wrong condensate transport"
                )
            elif attribute == "velocity_weno":
                reason = (
                    "the fixed product profile uses advwenov=0; enabling velocity "
                    "WENO would change all three momentum flux families"
                )
            else:
                reason = (
                    "this CUDA image contains only the source-mapped fifth-order "
                    "terrain scalar branch; another value selects an unmeasured "
                    "stencil, cadence, or geometry"
                )
            refuse(
                code="CM1-ADVECTION5-UNSUPPORTED",
                gate="dry-dynamics",
                option=option,
                value=value,
                reason=reason,
                remedy=(
                    "use the sealed fifth-order terrain scalar profile or add a "
                    "separate native oracle and CUDA image for the requested branch"
                ),
            )


def scalar_advection_uses_weno(
    nrk: int,
    *,
    nrkmax: int = CM1_NRKMAX,
    wflag: bool = True,
    selection: Advection5Selection = DEFAULT_ADVECTION5_SELECTION,
) -> bool:
    """Return the exact ``advs`` entry selector without importing CUDA."""

    validate_advection5_selection(selection)
    if isinstance(nrk, bool) or not isinstance(nrk, int):
        raise TypeError("nrk must be an integer")
    if isinstance(nrkmax, bool) or not isinstance(nrkmax, int):
        raise TypeError("nrkmax must be an integer")
    if nrkmax != CM1_NRKMAX or nrk < 1 or nrk > nrkmax:
        raise ValueError("the sealed low-storage schedule requires nrkmax=3 and nrk=1..3")
    if not isinstance(wflag, bool):
        raise TypeError("wflag must be boolean")
    return bool(wflag and selection.scalar_weno == 2 and nrk == nrkmax)


def _f32(np: Any, value: Any) -> Any:
    return np.float32(value)


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _flx3_f32(np: Any, s1: Any, s2: Any, s3: Any) -> Any:
    result = _mul(np, _f32(np, -1.0 / 6.0), s1)
    result = _add(np, result, _mul(np, _f32(np, 5.0 / 6.0), s2))
    return _add(np, result, _mul(np, _f32(np, 2.0 / 6.0), s3))


def _flx5_f32(np: Any, values: Any) -> Any:
    s1, s2, s3, s4, s5 = (np.float32(value) for value in values)
    result = _mul(np, _f32(np, 2.0 / 60.0), s1)
    result = _add(np, result, _mul(np, _f32(np, -13.0 / 60.0), s2))
    result = _add(np, result, _mul(np, _f32(np, 47.0 / 60.0), s3))
    result = _add(np, result, _mul(np, _f32(np, 27.0 / 60.0), s4))
    return _add(np, result, _mul(np, _f32(np, -3.0 / 60.0), s5))


def _weno5_f32(np: Any, values: Any, weps: float) -> Any:
    s1, s2, s3, s4, s5 = (np.float32(value) for value in values)

    d1 = _add(np, _sub(np, s1, _mul(np, np.float32(2.0), s2)), s3)
    e1 = _add(np, _sub(np, s1, _mul(np, np.float32(4.0), s2)), _mul(np, np.float32(3.0), s3))
    b1r = _add(np, _mul(np, np.float32(13.0 / 12.0), _mul(np, d1, d1)), _mul(np, np.float32(0.25), _mul(np, e1, e1)))
    d2 = _add(np, _sub(np, s2, _mul(np, np.float32(2.0), s3)), s4)
    e2 = _sub(np, s2, s4)
    b2r = _add(np, _mul(np, np.float32(13.0 / 12.0), _mul(np, d2, d2)), _mul(np, np.float32(0.25), _mul(np, e2, e2)))
    d3 = _add(np, _sub(np, s3, _mul(np, np.float32(2.0), s4)), s5)
    e3 = _add(np, _sub(np, _mul(np, np.float32(3.0), s3), _mul(np, np.float32(4.0), s4)), s5)
    b3r = _add(np, _mul(np, np.float32(13.0 / 12.0), _mul(np, d3, d3)), _mul(np, np.float32(0.25), _mul(np, e3, e3)))
    b1, b2, b3 = float(b1r), float(b2r), float(b3r)
    eps = float(weps)
    tau = abs(b1 - b3)
    # adv_routines.F:6189-6191 spells the linear weights as default-REAL
    # literals: binary32 0.1/0.6/0.3, promoted to double in the product.  Its
    # integer ``**2`` is q*q under gfortran, as in the kernels; Python's ``**``
    # calls pow(), which glibc only rounds to within 0.52 ULP.
    q1 = min(1.0e30, tau / (b1 + eps))
    q2 = min(1.0e30, tau / (b2 + eps))
    q3 = min(1.0e30, tau / (b3 + eps))
    w1 = float(np.float32(0.1)) * (1.0 + q1 * q1)
    w2 = float(np.float32(0.6)) * (1.0 + q2 * q2)
    w3 = float(np.float32(0.3)) * (1.0 + q3 * q3)
    p1 = _add(np, _add(np, _mul(np, np.float32(2.0 / 6.0), s1), _mul(np, np.float32(-7.0 / 6.0), s2)), _mul(np, np.float32(11.0 / 6.0), s3))
    p2 = _add(np, _add(np, _mul(np, np.float32(-1.0 / 6.0), s2), _mul(np, np.float32(5.0 / 6.0), s3)), _mul(np, np.float32(2.0 / 6.0), s4))
    p3 = _add(np, _add(np, _mul(np, np.float32(2.0 / 6.0), s3), _mul(np, np.float32(5.0 / 6.0), s4)), _mul(np, np.float32(-1.0 / 6.0), s5))
    return np.float32((w1 * float(p1) + w2 * float(p2) + w3 * float(p3)) / (w1 + w2 + w3))


def reconstruct_face5_f32(
    values: Any,
    mass_flux: float,
    *,
    use_weno: bool,
    weps: float,
) -> Any:
    """CPU transcription of one signed scalar face flux."""

    np = importlib.import_module("numpy")
    stencil = np.asarray(values)
    if stencil.dtype != np.dtype("float32") or stencil.shape != (5,):
        raise TypeError("values must be a float32 numpy array with shape (5,)")
    if not isinstance(use_weno, bool):
        raise TypeError("use_weno must be boolean")
    if isinstance(mass_flux, bool) or not isinstance(mass_flux, (int, float)):
        raise TypeError("mass_flux must be numeric")
    flux = float(mass_flux)
    epsilon = float(weps)
    if not math.isfinite(flux):
        raise ValueError("mass_flux must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("weps must be finite and > 0")
    oriented = stencil if flux >= 0.0 else stencil[::-1]
    value = _weno5_f32(np, oriented, epsilon) if use_weno else _flx5_f32(np, oriented)
    return _mul(np, np.float32(flux), value)


def _upstrpd_f32(np: Any, s1: Any, s2: Any, s3: Any, weps: float) -> Any:
    dd = _sub(np, s2, s1)
    signed_epsilon = np.copysign(np.float32(abs(np.float32(weps))), dd)
    rr = _div(np, _sub(np, s3, s2), _add(np, signed_epsilon, dd))
    candidate = _add(np, np.float32(1.0 / 6.0), _mul(np, np.float32(2.0 / 6.0), rr))
    phi = np.maximum(
        np.float32(0.0),
        np.minimum(rr, np.minimum(candidate, np.float32(1.0))),
    )
    return _add(np, s2, _mul(np, phi, _sub(np, s2, s1)))


def _require_numpy_array(
    np: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool = True
) -> Any:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


def cpu_terrain_mass_flux5_inplace(
    arrays: MutableMapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: Advection5Selection = DEFAULT_ADVECTION5_SELECTION,
) -> None:
    """Independent source-ordered CPU transcription for validation."""

    validate_advection5_selection(selection)
    ni, nj, nk = _validated_grid(ni, nj, nk)
    dx, dy, dz = (
        _validated_spacing(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz))
    )
    np = importlib.import_module("numpy")
    ng = CM1_OPEN_NGXY
    expected = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w3d": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    a = {
        name: _require_numpy_array(np, arrays[name], name, shape)
        for name, shape in expected.items()
    }
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    required_metrics = (
        "rho0_3d",
        "rf0",
        "rgzu",
        "rgzv",
        "dzdx",
        "dzdy",
        "gz",
    )
    m = {
        name: _require_numpy_array(
            np, metrics[name], name, shapes[name], writable=False
        )
        for name in required_metrics
    }
    divx = _require_numpy_array(np, divx, "divx", (nk, nj, ni))
    pointers = [
        int(value.ctypes.data) for value in (*a.values(), *m.values(), divx)
    ]
    if len(pointers) != len(set(pointers)):
        raise ValueError("terrain mass-flux inputs and outputs must not alias")
    half = np.float32(0.5)
    for k0 in range(nk):
        k = k0 + 1
        for j in range(nj):
            y = j + ng
            for face in range(1, ni):
                x = face + ng
                density = _add(
                    np, m["rho0_3d"][k0, j, face - 1], m["rho0_3d"][k0, j, face]
                )
                a["rru"][k, y, x] = _mul(
                    np,
                    _mul(np, half, density),
                    _mul(np, a["u3d"][k, y, x], m["rgzu"][j, face]),
                )
        for face in range(1, nj):
            y = face + ng
            for i in range(ni):
                x = i + ng
                density = _add(
                    np,
                    m["rho0_3d"][k0, face - 1, i],
                    m["rho0_3d"][k0, face, i],
                )
                a["rrv"][k, y, x] = _mul(
                    np,
                    _mul(np, half, density),
                    _mul(np, a["v3d"][k, y, x], m["rgzv"][face, i]),
                )
    a["rrw"][1, ng : ng + nj, ng : ng + ni] = np.float32(0.0)
    a["rrw"][nk + 1, ng : ng + nj, ng : ng + ni] = np.float32(0.0)
    dz32 = np.float32(dz)
    zt = _mul(np, np.float32(nk), dz32)
    rzt = _div(np, np.float32(1.0), zt)
    for k in range(2, nk + 1):
        sigmaf = _mul(np, np.float32(k - 1), dz32)
        sigma = _mul(np, np.float32(k - 1.5), dz32)
        r2 = _mul(np, _sub(np, sigmaf, sigma), _div(np, np.float32(1.0), dz32))
        r1 = _mul(np, half, _sub(np, np.float32(1.0), r2))
        r2 = _mul(np, half, r2)
        for j in range(1, nj - 1):
            y = j + ng
            for i in range(1, ni - 1):
                x = i + ng
                ux = _add(
                    np,
                    _mul(
                        np,
                        r2,
                        _add(np, a["rru"][k, y, x], a["rru"][k, y, x + 1]),
                    ),
                    _mul(
                        np,
                        r1,
                        _add(
                            np,
                            a["rru"][k - 1, y, x],
                            a["rru"][k - 1, y, x + 1],
                        ),
                    ),
                )
                vy = _add(
                    np,
                    _mul(
                        np,
                        r2,
                        _add(np, a["rrv"][k, y, x], a["rrv"][k, y + 1, x]),
                    ),
                    _mul(
                        np,
                        r1,
                        _add(
                            np,
                            a["rrv"][k - 1, y, x],
                            a["rrv"][k - 1, y + 1, x],
                        ),
                    ),
                )
                terrain = _add(
                    np,
                    _mul(np, ux, m["dzdx"][j, i]),
                    _mul(np, vy, m["dzdy"][j, i]),
                )
                terrain = _mul(np, terrain, _sub(np, sigmaf, zt))
                terrain = _mul(np, terrain, m["gz"][j, i])
                terrain = _mul(np, terrain, rzt)
                a["rrw"][k, y, x] = _add(
                    np,
                    _mul(np, m["rf0"][k - 1, j, i], a["w3d"][k, y, x]),
                    terrain,
                )
    rdx = _div(np, np.float32(1.0), np.float32(dx))
    rdy = _div(np, np.float32(1.0), np.float32(dy))
    rdsf = _div(np, np.float32(1.0), dz32)
    for k0 in range(nk):
        k = k0 + 1
        for j in range(1, nj - 1):
            y = j + ng
            for i in range(1, ni - 1):
                x = i + ng
                value = _add(
                    np,
                    _mul(np, _sub(np, a["rru"][k, y, x + 1], a["rru"][k, y, x]), rdx),
                    _mul(np, _sub(np, a["rrv"][k, y + 1, x], a["rrv"][k, y, x]), rdy),
                )
                value = _add(
                    np,
                    value,
                    _mul(np, _sub(np, a["rrw"][k + 1, y, x], a["rrw"][k, y, x]), rdsf),
                )
                divx[k0, j, i] = np.float32(0.0) if abs(float(value)) < 1.0e-20 else value


def cpu_scalar_advection5_inplace(
    scalar: Any,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    tendency: Any,
    fluxes: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    weps: float,
    wflag: bool = True,
    selection: Advection5Selection = DEFAULT_ADVECTION5_SELECTION,
) -> None:
    """CPU reference for the admitted scalar flux/tendency slice."""

    use_weno = scalar_advection_uses_weno(nrk, wflag=wflag, selection=selection)
    ni, nj, nk = _validated_grid(ni, nj, nk)
    dx, dy, dz = (
        _validated_spacing(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz))
    )
    if isinstance(weps, bool) or not isinstance(weps, (int, float)):
        raise TypeError("weps must be numeric")
    weps = float(weps)
    if not math.isfinite(weps) or weps <= 0.0:
        raise ValueError("weps must be finite and > 0")
    np = importlib.import_module("numpy")
    ng = CM1_OPEN_NGXY
    scalar = _require_numpy_array(
        np, scalar, "scalar", (nk + 2, nj + 2 * ng, ni + 2 * ng), writable=False
    )
    rru = _require_numpy_array(np, arrays["rru"], "rru", (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng), writable=False)
    rrv = _require_numpy_array(np, arrays["rrv"], "rrv", (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng), writable=False)
    rrw = _require_numpy_array(np, arrays["rrw"], "rrw", (nk + 3, nj + 2 * ng, ni + 2 * ng), writable=False)
    fx = _require_numpy_array(np, fluxes["x"], "flux_x", rru.shape)
    fy = _require_numpy_array(np, fluxes["y"], "flux_y", rrv.shape)
    fz = _require_numpy_array(np, fluxes["z"], "flux_z", rrw.shape)
    divx = _require_numpy_array(np, divx, "divx", (nk, nj, ni), writable=False)
    tendency = _require_numpy_array(np, tendency, "tendency", (nk, nj, ni))
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    c1 = _require_numpy_array(np, metrics["c1"], "c1", shapes["c1"], writable=False)
    c2 = _require_numpy_array(np, metrics["c2"], "c2", shapes["c2"], writable=False)
    rr0 = _require_numpy_array(np, metrics["rr0"], "rr0", shapes["rr0"], writable=False)
    gz = _require_numpy_array(np, metrics["gz"], "gz", shapes["gz"], writable=False)
    pointers = [int(value.ctypes.data) for value in (scalar, rru, rrv, rrw, fx, fy, fz, divx, tendency, c1, c2, rr0, gz)]
    if len(pointers) != len(set(pointers)):
        raise ValueError("scalar advection inputs, flux scratch, and tendency must not alias")
    for k in range(1, nk + 1):
        for j in range(nj):
            y = j + ng
            for face in range(ni + 1):
                x = face + ng
                mass = rru[k, y, x]
                if mass >= np.float32(0.0):
                    values = scalar[k, y, x - 3 : x + 2]
                else:
                    values = scalar[k, y, x - 2 : x + 3][::-1]
                reconstruction = _weno5_f32(np, values, weps) if use_weno else _flx5_f32(np, values)
                fx[k, y, x] = _mul(np, mass, reconstruction)
        for face in range(nj + 1):
            y = face + ng
            for i in range(ni):
                x = i + ng
                mass = rrv[k, y, x]
                if mass >= np.float32(0.0):
                    values = scalar[k, y - 3 : y + 2, x]
                else:
                    values = scalar[k, y - 2 : y + 3, x][::-1]
                reconstruction = _weno5_f32(np, values, weps) if use_weno else _flx5_f32(np, values)
                fy[k, y, x] = _mul(np, mass, reconstruction)
    fz[1, ng : ng + nj, ng : ng + ni] = np.float32(0.0)
    fz[nk + 1, ng : ng + nj, ng : ng + ni] = np.float32(0.0)
    for k in range(2, nk + 1):
        for j in range(nj):
            y = j + ng
            for i in range(ni):
                x = i + ng
                mass = rrw[k, y, x]
                if k == 2:
                    if mass >= np.float32(0.0):
                        reconstruction = _add(np, _mul(np, c1[k - 1, j, i], scalar[k - 1, y, x]), _mul(np, c2[k - 1, j, i], scalar[k, y, x]))
                    elif use_weno:
                        reconstruction = _upstrpd_f32(np, scalar[k + 1, y, x], scalar[k, y, x], scalar[k - 1, y, x], weps)
                    else:
                        reconstruction = _flx3_f32(np, scalar[k + 1, y, x], scalar[k, y, x], scalar[k - 1, y, x])
                elif k == nk:
                    # vadv_lwr_weno_23 takes the upstream face for rrw > 0 at
                    # k = nk; vadv_lwr_flx_23 for rrw >= 0 (adv_routines.F:
                    # 4078, 4853).  At rrw = +-0 only the sign of the zero moves.
                    if (mass > np.float32(0.0)) if use_weno else (mass >= np.float32(0.0)):
                        reconstruction = _upstrpd_f32(np, scalar[k - 2, y, x], scalar[k - 1, y, x], scalar[k, y, x], weps) if use_weno else _flx3_f32(np, scalar[k - 2, y, x], scalar[k - 1, y, x], scalar[k, y, x])
                    else:
                        reconstruction = _add(np, _mul(np, c1[k - 1, j, i], scalar[k - 1, y, x]), _mul(np, c2[k - 1, j, i], scalar[k, y, x]))
                elif k == 3:
                    if mass >= np.float32(0.0):
                        reconstruction = _upstrpd_f32(np, scalar[k - 2, y, x], scalar[k - 1, y, x], scalar[k, y, x], weps) if use_weno else _flx3_f32(np, scalar[k - 2, y, x], scalar[k - 1, y, x], scalar[k, y, x])
                    else:
                        values = scalar[k - 2 : k + 3, y, x][::-1]
                        reconstruction = _weno5_f32(np, values, weps) if use_weno else _flx5_f32(np, values)
                elif k == nk - 1:
                    if mass >= np.float32(0.0):
                        values = scalar[k - 3 : k + 2, y, x]
                        reconstruction = _weno5_f32(np, values, weps) if use_weno else _flx5_f32(np, values)
                    elif use_weno:
                        reconstruction = _upstrpd_f32(np, scalar[k + 1, y, x], scalar[k, y, x], scalar[k - 1, y, x], weps)
                    else:
                        reconstruction = _flx3_f32(np, scalar[k + 1, y, x], scalar[k, y, x], scalar[k - 1, y, x])
                else:
                    if mass >= np.float32(0.0):
                        values = scalar[k - 3 : k + 2, y, x]
                    else:
                        values = scalar[k - 2 : k + 3, y, x][::-1]
                    reconstruction = _weno5_f32(np, values, weps) if use_weno else _flx5_f32(np, values)
                fz[k, y, x] = _mul(np, mass, reconstruction)
    rdx = _div(np, np.float32(1.0), np.float32(dx))
    rdy = _div(np, np.float32(1.0), np.float32(dy))
    rdsf = _div(np, np.float32(1.0), np.float32(dz))
    for k0 in range(nk):
        k = k0 + 1
        for j in range(3, nj - 3):
            y = j + ng
            for i in range(3, ni - 3):
                x = i + ng
                advx = _mul(np, _sub(np, fx[k, y, x], fx[k, y, x + 1]), rdx)
                advy = _mul(np, _sub(np, fy[k, y, x], fy[k, y + 1, x]), rdy)
                if k == 1:
                    advz = _mul(np, -fz[2, y, x], rdsf)
                elif k == nk:
                    advz = _mul(np, fz[nk, y, x], rdsf)
                else:
                    advz = _mul(np, _sub(np, fz[k, y, x], fz[k + 1, y, x]), rdsf)
                total = _add(np, _add(np, advx, advy), advz)
                total = _add(np, total, _mul(np, scalar[k, y, x], divx[k0, j, i]))
                tendency[k0, j, i] = _mul(np, _mul(np, total, rr0[k0, j, i]), gz[j, i])


def advection5_full_domain_interior_bounds(ni: int, nj: int) -> dict[str, tuple[int, int]]:
    """Return one-based cells independent of unresolved external base halos."""

    if isinstance(ni, bool) or not isinstance(ni, int) or ni < 8:
        raise ValueError("ni must be an integer >= 8")
    if isinstance(nj, bool) or not isinstance(nj, int) or nj < 8:
        raise ValueError("nj must be an integer >= 8")
    return {"i": (4, ni - 3), "j": (4, nj - 3)}


CUDA_ADVECTION5_KERNEL_NAMES = (
    "cm1_adv5_oracle_batch_v1",
    "cm1_adv5_terrain_horizontal_mass_flux_v1",
    "cm1_adv5_terrain_vertical_mass_flux_v1",
    "cm1_adv5_terrain_divergence_v1",
    "cm1_adv5_scalar_horizontal_v1",
    "cm1_adv5_scalar_vertical_v1",
    "cm1_adv5_scalar_tendency_v1",
)


CUDA_ADVECTION5_SOURCE = r'''// gpuwm-cm1.cuda-advection5-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float flx3(float s1,float s2,float s3){float r=m32(-1.0f/6.0f,s1);r=a32(r,m32(5.0f/6.0f,s2));return a32(r,m32(2.0f/6.0f,s3));}
__device__ __forceinline__ float flx5(float s1,float s2,float s3,float s4,float s5){float r=m32(2.0f/60.0f,s1);r=a32(r,m32(-13.0f/60.0f,s2));r=a32(r,m32(47.0f/60.0f,s3));r=a32(r,m32(27.0f/60.0f,s4));return a32(r,m32(-3.0f/60.0f,s5));}
__device__ __forceinline__ float weno5(float s1,float s2,float s3,float s4,float s5,double weps){
 float d1=a32(s32(s1,m32(2.0f,s2)),s3),e1=a32(s32(s1,m32(4.0f,s2)),m32(3.0f,s3));
 float b1r=a32(m32(13.0f/12.0f,m32(d1,d1)),m32(0.25f,m32(e1,e1)));
 float d2=a32(s32(s2,m32(2.0f,s3)),s4),e2=s32(s2,s4);
 float b2r=a32(m32(13.0f/12.0f,m32(d2,d2)),m32(0.25f,m32(e2,e2)));
 float d3=a32(s32(s3,m32(2.0f,s4)),s5),e3=a32(s32(m32(3.0f,s3),m32(4.0f,s4)),s5);
 float b3r=a32(m32(13.0f/12.0f,m32(d3,d3)),m32(0.25f,m32(e3,e3)));
 double b1=(double)b1r,b2=(double)b2r,b3=(double)b3r,tau=fabs(b1-b3);
 double q1=fmin(1.0e30,tau/(b1+weps)),q2=fmin(1.0e30,tau/(b2+weps)),q3=fmin(1.0e30,tau/(b3+weps));
 double w1=(double)0.1f*(1.0+q1*q1),w2=(double)0.6f*(1.0+q2*q2),w3=(double)0.3f*(1.0+q3*q3);
 float p1=a32(a32(m32(2.0f/6.0f,s1),m32(-7.0f/6.0f,s2)),m32(11.0f/6.0f,s3));
 float p2=a32(a32(m32(-1.0f/6.0f,s2),m32(5.0f/6.0f,s3)),m32(2.0f/6.0f,s4));
 float p3=a32(a32(m32(2.0f/6.0f,s3),m32(5.0f/6.0f,s4)),m32(-1.0f/6.0f,s5));
 return (float)((w1*(double)p1+w2*(double)p2+w3*(double)p3)/(w1+w2+w3));}
__device__ __forceinline__ float upstrpd(float s1,float s2,float s3,float weps){float dd=s32(s2,s1);float signed_eps=copysignf(fabsf(weps),dd);float rr=d32(s32(s3,s2),a32(signed_eps,dd));float phi=fmaxf(0.0f,fminf(rr,fminf(a32(1.0f/6.0f,m32(2.0f/6.0f,rr)),1.0f)));return a32(s2,m32(phi,s32(s2,s1)));}
__device__ __forceinline__ float recon(float s1,float s2,float s3,float s4,float s5,unsigned int use_weno,double weps){return use_weno?weno5(s1,s2,s3,s4,s5,weps):flx5(s1,s2,s3,s4,s5);}
// obc packs CM1 wbc|ebc<<4|sbc<<8|nbc<<12 (bc.F lateral values 1..4).  obc=0 is a
// tile-local launch: the outer three cells belong to a neighbour tile and are skipped.
__device__ __forceinline__ unsigned int obc_side(unsigned int obc,unsigned int side){return (obc>>(4u*side))&15u;}
__device__ __forceinline__ bool obc_owned(unsigned int i,unsigned int j,unsigned int ni,unsigned int nj,unsigned int obc){return obc!=0u||(i>=3u&&i+3u<ni&&j>=3u&&j+3u<nj);}
// adv.F:86-98,214-340: an open side (value 2) with inflow through its boundary face.
__device__ __forceinline__ bool obc_inflow_lo(unsigned int i,unsigned int obc,unsigned int side,float m){return i==0u&&obc_side(obc,side)==2u&&m>=0.0f;}
__device__ __forceinline__ bool obc_inflow_hi(unsigned int i,unsigned int n,unsigned int obc,unsigned int side,float m){return i+1u==n&&obc_side(obc,side)==2u&&m<=0.0f;}

extern "C" __global__ void cm1_adv5_oracle_batch_v1(const float* s,const float* mf,const unsigned int* use_weno,float* out,unsigned int n,double weps){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=step){const float* p=s+5ull*q;float v=(mf[q]>=0.0f)?recon(p[0],p[1],p[2],p[3],p[4],use_weno[q],weps):recon(p[4],p[3],p[2],p[1],p[0],use_weno[q],weps);out[q]=m32(mf[q],v);}}

extern "C" __global__ void cm1_adv5_terrain_horizontal_mass_flux_v1(
 const float* u,const float* v,const float* rho0,const float* rgzu,const float* rgzv,float* rru,float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;
 unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng,nxw=nxs,nyw=nys;
 for(unsigned long long z=q;z<total;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;
  if(i>=1u){unsigned int f=i;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j)*ni+i-1u],rho0[((unsigned long long)(k-1u)*nj+j)*ni+i]);rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(m32(0.5f,rh),m32(u[i3(k,j+ng,f+ng,nyu,nxu)],rgzu[(unsigned long long)j*(ni+1u)+f]));}
  if(j>=1u){unsigned int f=j;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j-1u)*ni+i],rho0[((unsigned long long)(k-1u)*nj+j)*ni+i]);rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(m32(0.5f,rh),m32(v[i3(k,f+ng,i+ng,nyv,nxv)],rgzv[(unsigned long long)f*ni+i]));}
  if(k==1u){rrw[i3(1u,j+ng,i+ng,nyw,nxw)]=0.0f;rrw[i3(nk+1u,j+ng,i+ng,nyw,nxw)]=0.0f;}
 }}

extern "C" __global__ void cm1_adv5_terrain_vertical_mass_flux_v1(
 const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng,nxw=nxs,nyw=nys;float zt=m32((float)nk,dz),rzt=d32(1.0f,zt);
 for(unsigned long long z=q;z<total;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;if(k>=2u&&i>=1u&&i+1u<ni&&j>=1u&&j+1u<nj){float sigmaf=m32((float)(k-1u),dz),sigma=m32((float)k-1.5f,dz),r2=m32(s32(sigmaf,sigma),d32(1.0f,dz)),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);
  unsigned int x=i+ng,y=j+ng;float ux=a32(m32(r2,a32(rru[i3(k,y,x,nyu,nxu)],rru[i3(k,y,x+1u,nyu,nxu)])),m32(r1,a32(rru[i3(k-1u,y,x,nyu,nxu)],rru[i3(k-1u,y,x+1u,nyu,nxu)])));float vy=a32(m32(r2,a32(rrv[i3(k,y,x,nyv,nxv)],rrv[i3(k,y+1u,x,nyv,nxv)])),m32(r1,a32(rrv[i3(k-1u,y,x,nyv,nxv)],rrv[i3(k-1u,y+1u,x,nyv,nxv)])));float terrain=a32(m32(ux,dzdx[(unsigned long long)j*ni+i]),m32(vy,dzdy[(unsigned long long)j*ni+i]));terrain=m32(m32(m32(terrain,s32(sigmaf,zt)),gz[(unsigned long long)j*ni+i]),rzt);rrw[i3(k,y,x,nyw,nxw)]=a32(m32(rf0[((unsigned long long)(k-1u)*nj+j)*ni+i],w[i3(k,y,x,nyw,nxw)]),terrain);}}
}

extern "C" __global__ void cm1_adv5_terrain_divergence_v1(
 const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng,nxw=nxs,nyw=nys;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);
 for(unsigned long long z=q;z<total;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;if(i>=1u&&i+1u<ni&&j>=1u&&j+1u<nj){unsigned int x=i+ng,y=j+ng;float d=a32(m32(s32(rru[i3(k,y,x+1u,nyu,nxu)],rru[i3(k,y,x,nyu,nxu)]),rdx),m32(s32(rrv[i3(k,y+1u,x,nyv,nxv)],rrv[i3(k,y,x,nyv,nxv)]),rdy));d=a32(d,m32(s32(rrw[i3(k+1u,y,x,nyw,nxw)],rrw[i3(k,y,x,nyw,nxw)]),rdsf));divx[z]=(fabsf(d)<1.0e-20f)?0.0f:d;}}
}

extern "C" __global__ void cm1_adv5_scalar_horizontal_v1(const float* a,const float* rru,const float* rrv,float* fx,float* fy,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int use_weno,double weps){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*(nj+1u)*(ni+1u);unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng;for(;q<total;q+=step){unsigned int i=(unsigned int)(q%(ni+1u)),j=(unsigned int)((q/(ni+1u))%(nj+1u)),k=(unsigned int)(q/((unsigned long long)(ni+1u)*(nj+1u)))+1u;if(j<nj){unsigned int x=i+ng,y=j+ng;float m=rru[i3(k,y,x,nyu,nxu)],r=(m>=0.0f)?recon(a[i3(k,y,x-3u,nys,nxs)],a[i3(k,y,x-2u,nys,nxs)],a[i3(k,y,x-1u,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k,y,x+1u,nys,nxs)],use_weno,weps):recon(a[i3(k,y,x+2u,nys,nxs)],a[i3(k,y,x+1u,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k,y,x-1u,nys,nxs)],a[i3(k,y,x-2u,nys,nxs)],use_weno,weps);fx[i3(k,y,x,nyu,nxu)]=m32(m,r);}if(i<ni){unsigned int x=i+ng,y=j+ng;float m=rrv[i3(k,y,x,nyv,nxv)],r=(m>=0.0f)?recon(a[i3(k,y-3u,x,nys,nxs)],a[i3(k,y-2u,x,nys,nxs)],a[i3(k,y-1u,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k,y+1u,x,nys,nxs)],use_weno,weps):recon(a[i3(k,y+2u,x,nys,nxs)],a[i3(k,y+1u,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k,y-1u,x,nys,nxs)],a[i3(k,y-2u,x,nys,nxs)],use_weno,weps);fy[i3(k,y,x,nyv,nxv)]=m32(m,r);}}}

extern "C" __global__ void cm1_adv5_scalar_vertical_v1(const float* a,const float* rrw,const float* c1,const float* c2,float* fz,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int use_weno,double weps){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)(nk-1u)*nj*ni;unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxw=nxs,nyw=nys;for(;q<total;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+2u,x=i+ng,y=j+ng;float m=rrw[i3(k,y,x,nyw,nxw)],r;if(k==2u){r=(m>=0.0f)?a32(m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],a[i3(k-1u,y,x,nys,nxs)]),m32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],a[i3(k,y,x,nys,nxs)])):(use_weno?upstrpd(a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],(float)weps):flx3(a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)]));}
 else if(k==nk){r=(use_weno?m>0.0f:m>=0.0f)?(use_weno?upstrpd(a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],(float)weps):flx3(a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)])):a32(m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],a[i3(k-1u,y,x,nys,nxs)]),m32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],a[i3(k,y,x,nys,nxs)]));}
 else if(k==3u){r=(m>=0.0f)?(use_weno?upstrpd(a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],(float)weps):flx3(a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)])):recon(a[i3(k+2u,y,x,nys,nxs)],a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k-2u,y,x,nys,nxs)],use_weno,weps);}
 else if(k==nk-1u){r=(m>=0.0f)?recon(a[i3(k-3u,y,x,nys,nxs)],a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k+1u,y,x,nys,nxs)],use_weno,weps):(use_weno?upstrpd(a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],(float)weps):flx3(a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)]));}
 else {r=(m>=0.0f)?recon(a[i3(k-3u,y,x,nys,nxs)],a[i3(k-2u,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k+1u,y,x,nys,nxs)],use_weno,weps):recon(a[i3(k+2u,y,x,nys,nxs)],a[i3(k+1u,y,x,nys,nxs)],a[i3(k,y,x,nys,nxs)],a[i3(k-1u,y,x,nys,nxs)],a[i3(k-2u,y,x,nys,nxs)],use_weno,weps);}fz[i3(k,y,x,nyw,nxw)]=m32(m,r);}}

extern "C" __global__ void cm1_adv5_scalar_tendency_v1(const float* a,const float* fx,const float* fy,const float* fz,const float* divx,const float* rr0,const float* gz,float* tendency,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,const float* rru,const float* rrv,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng,nxw=nxs,nyw=nys;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);for(;q<total;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+1u;if(obc_owned(i,j,ni,nj,obc)){unsigned int x=i+ng,y=j+ng;float fw=fx[i3(k,y,x,nyu,nxu)],fe=fx[i3(k,y,x+1u,nyu,nxu)],fs=fy[i3(k,y,x,nyv,nxv)],fn=fy[i3(k,y+1u,x,nyv,nxv)],sv=a[i3(k,y,x,nys,nxs)];float uw=rru[i3(k,y,x,nyu,nxu)],ue=rru[i3(k,y,x+1u,nyu,nxu)],vs=rrv[i3(k,y,x,nyv,nxv)],vn=rrv[i3(k,y+1u,x,nyv,nxv)];bool iw=obc_inflow_lo(i,obc,0u,uw),ie=obc_inflow_hi(i,ni,obc,1u,ue),js=obc_inflow_lo(j,obc,2u,vs),jn=obc_inflow_hi(j,nj,obc,3u,vn);if(iw)fw=fe;if(ie)fe=fw;if(js)fs=fn;if(jn)fn=fs;float ax=m32(s32(fw,fe),rdx),ay=m32(s32(fs,fn),rdy),az;if(iw)ax=s32(ax,m32(m32(sv,s32(ue,uw)),rdx));if(ie)ax=s32(ax,m32(m32(sv,s32(ue,uw)),rdx));if(js)ay=s32(ay,m32(m32(sv,s32(vn,vs)),rdy));if(jn)ay=s32(ay,m32(m32(sv,s32(vn,vs)),rdy));if(k==1u)az=m32(-fz[i3(2u,y,x,nyw,nxw)],rdsf);else if(k==nk)az=m32(fz[i3(nk,y,x,nyw,nxw)],rdsf);else az=m32(s32(fz[i3(k,y,x,nyw,nxw)],fz[i3(k+1u,y,x,nyw,nxw)]),rdsf);float total_adv=a32(a32(ax,ay),az);total_adv=a32(total_adv,m32(sv,divx[q]));tendency[q]=m32(m32(total_adv,rr0[q]),gz[(unsigned long long)j*ni+i]);}}}
'''

CUDA_ADVECTION5_SOURCE_BYTES = CUDA_ADVECTION5_SOURCE.encode("utf-8")
CUDA_ADVECTION5_SOURCE_SHA256 = (
    "b7325c2e0b507b45a8667b92cd0e03e8f31c32310fc20473b59080f7d76d214f"
)
#: The source the sealed ``cuda-advection5-v1`` / ``real-advection5-v1`` GPU
#: receipts were measured with.  It differs from the live source in two
#: spellings only: ``weno5`` multiplied its weights by binary64 ``0.1``/``0.6``/
#: ``0.3`` where CM1's ``adv_routines.F:6189-6191`` promotes binary32 literals,
#: and ``cm1_adv5_scalar_vertical_v1`` took the ``k=nk`` WENO face upstream for
#: ``rrw >= 0`` where ``vadv_lwr_weno_23`` tests ``rrw > 0`` (a zero-sign
#: difference only).  The receipts' words are those of that source, not CM1's;
#: ``tools/oracle/cm1_native_weno5_differential.pre_fix_spelling`` rebuilds it.
CUDA_ADVECTION5_EVIDENCE_SOURCE_SHA256 = (
    "48ac2ddd291841060a73ff4ed11711565c8f15547ac72a1c4daf7d733859581b"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_ADVECTION5_SOURCE_BYTES).hexdigest() == CUDA_ADVECTION5_SOURCE_SHA256


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
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledAdvection5Kernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_advection5() -> CompiledAdvection5Kernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged advection5 CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_ADVECTION5_SOURCE, "cm1_advection5_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_ADVECTION5_SOURCE, "cm1_advection5_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_ADVECTION5_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more advection5 entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled advection5 image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_ADVECTION5_KERNEL_NAMES)
    properties = importlib.import_module("cupy").cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledAdvection5Kernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, runtime_identity
    )


def _validated_grid(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    for name, value, minimum in (("ni", ni, 8), ("nj", nj, 8), ("nk", nk, 8)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    return ni, nj, nk


def _validated_spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _require_cuda_array(
    cupy: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool = True
) -> Any:
    if not isinstance(value, cupy.ndarray):
        raise TypeError(f"{name} must be a cupy.ndarray")
    if value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


@dataclass(frozen=True, slots=True)
class CudaTerrainMassFluxLaunch:
    ready_event: Any
    compiled: CompiledAdvection5Kernels
    receipt: dict[str, Any]


#: ``obc`` value of a tile-local launch: the outer three cells of the launch
#: rectangle belong to a neighbouring tile (or are restored from the old-value
#: ring) and receive no tendency here.
SCALAR_TILE_LOCAL_OBC = 0


def scalar_open_boundary_code(lateral_sides: Any) -> int:
    """Pack CM1's ``(wbc, ebc, sbc, nbc)`` into the scalar kernels' ``obc`` word.

    ``None`` is a tile-local launch (:data:`SCALAR_TILE_LOCAL_OBC`).  Any other
    value is a whole domain: every cell i=1..ni, j=1..nj receives CM1's
    tendency (adv.F:273-281, 646-656), and on a side whose value is 2 the
    inflow boundary-face flux copy and divx-piece subtraction of
    adv.F:214-340 (and pdef.F:49-100's inflow candidate) apply.  The pdef
    corrections read their lateral halo as ``bcs(dum)`` fills it
    (bc.F:180-364): 1 periodic, 2 zero-gradient, 3/4 mirror.
    """

    if lateral_sides is None:
        return SCALAR_TILE_LOCAL_OBC
    sides = tuple(lateral_sides)
    if len(sides) != 4 or any(
        isinstance(v, bool) or not isinstance(v, int) or v not in (1, 2, 3, 4) for v in sides
    ):
        raise ValueError(
            "lateral_sides must be CM1's (wbc, ebc, sbc, nbc), each 1, 2, 3 or 4"
        )
    return sides[0] | (sides[1] << 4) | (sides[2] << 8) | (sides[3] << 12)


def launch_cuda_terrain_mass_flux5(
    arrays: MutableMapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: Advection5Selection = DEFAULT_ADVECTION5_SELECTION,
    stream: Any | None = None,
    compiled: CompiledAdvection5Kernels | None = None,
) -> CudaTerrainMassFluxLaunch:
    """Enqueue source-ordered terrain mass flux and interior divergence."""

    validate_advection5_selection(selection)
    ni, nj, nk = _validated_grid(ni, nj, nk)
    dx, dy, dz = (_validated_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz)))
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    expected = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w3d": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    a = {name: _require_cuda_array(cupy, arrays[name], name, shape) for name, shape in expected.items()}
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    required_metrics = ("rho0_3d", "rf0", "rgzu", "rgzv", "dzdx", "dzdy", "gz")
    m = {name: _require_cuda_array(cupy, metrics[name], name, shapes[name], writable=False) for name in required_metrics}
    divx = _require_cuda_array(cupy, divx, "divx", (nk, nj, ni))
    pointers = [int(value.data.ptr) for value in (*a.values(), *m.values(), divx)]
    if len(pointers) != len(set(pointers)):
        raise ValueError("terrain mass-flux inputs and outputs must not alias")
    compiled = compiled or compile_cuda_advection5()
    stream = stream or cupy.cuda.get_current_stream()
    words = ni * nj * nk
    grid = ((words + CUDA_ADVECTION5_THREADS - 1) // CUDA_ADVECTION5_THREADS, 1, 1)
    horizontal, vertical, divergence = compiled.kernels[1:4]
    horizontal(
        grid,
        (CUDA_ADVECTION5_THREADS, 1, 1),
        (
            a["u3d"], a["v3d"], m["rho0_3d"], m["rgzu"], m["rgzv"],
            a["rru"], a["rrv"], a["rrw"],
            cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng),
        ),
        stream=stream,
    )
    vertical(
        grid,
        (CUDA_ADVECTION5_THREADS, 1, 1),
        (
            a["w3d"], m["rf0"], m["dzdx"], m["dzdy"], m["gz"],
            a["rru"], a["rrv"], a["rrw"],
            cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng),
            cupy.float32(dz),
        ),
        stream=stream,
    )
    divergence(
        grid,
        (CUDA_ADVECTION5_THREADS, 1, 1),
        (
            a["rru"], a["rrv"], a["rrw"], divx,
            cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng),
            cupy.float32(dx), cupy.float32(dy), cupy.float32(dz),
        ),
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return CudaTerrainMassFluxLaunch(
        event,
        compiled,
        {
            "schema": CUDA_ADVECTION5_SCHEMA,
            "operation": "terrain-mass-flux-and-divergence",
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
            "source_anchors": {key: SOURCE_ANCHORS[key] for key in ("terrain_mass_flux", "terrain_divergence")},
            "interior_bounds_one_based": advection5_full_domain_interior_bounds(ni, nj),
            "kernel_source_sha256": CUDA_ADVECTION5_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
    )


@dataclass(frozen=True, slots=True)
class CudaScalarAdvection5Launch:
    ready_event: Any
    compiled: CompiledAdvection5Kernels
    receipt: dict[str, Any]


def launch_cuda_scalar_advection5(
    scalar: Any,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    tendency: Any,
    fluxes: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    weps: float,
    wflag: bool = True,
    selection: Advection5Selection = DEFAULT_ADVECTION5_SELECTION,
    stream: Any | None = None,
    compiled: CompiledAdvection5Kernels | None = None,
    lateral_sides: tuple[int, int, int, int] | None = None,
) -> CudaScalarAdvection5Launch:
    """Enqueue scalar fluxes and tendency for one low-storage RK stage.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)`` of a whole-domain
    launch; ``None`` is a tile-local launch whose outer three cells are not
    written (see :func:`scalar_open_boundary_code`).
    """

    use_weno = scalar_advection_uses_weno(nrk, wflag=wflag, selection=selection)
    ni, nj, nk = _validated_grid(ni, nj, nk)
    dx, dy, dz = (_validated_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz)))
    if isinstance(weps, bool) or not isinstance(weps, (int, float)):
        raise TypeError("weps must be numeric")
    weps = float(weps)
    if not math.isfinite(weps) or weps <= 0.0:
        raise ValueError("weps must be finite and > 0")
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    scalar = _require_cuda_array(cupy, scalar, "scalar", (nk + 2, nj + 2 * ng, ni + 2 * ng), writable=False)
    rru = _require_cuda_array(cupy, arrays["rru"], "rru", (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng), writable=False)
    rrv = _require_cuda_array(cupy, arrays["rrv"], "rrv", (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng), writable=False)
    rrw = _require_cuda_array(cupy, arrays["rrw"], "rrw", (nk + 3, nj + 2 * ng, ni + 2 * ng), writable=False)
    fx = _require_cuda_array(cupy, fluxes["x"], "flux_x", rru.shape)
    fy = _require_cuda_array(cupy, fluxes["y"], "flux_y", rrv.shape)
    fz = _require_cuda_array(cupy, fluxes["z"], "flux_z", rrw.shape)
    divx = _require_cuda_array(cupy, divx, "divx", (nk, nj, ni), writable=False)
    tendency = _require_cuda_array(cupy, tendency, "tendency", (nk, nj, ni))
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    c1 = _require_cuda_array(cupy, metrics["c1"], "c1", shapes["c1"], writable=False)
    c2 = _require_cuda_array(cupy, metrics["c2"], "c2", shapes["c2"], writable=False)
    rr0 = _require_cuda_array(cupy, metrics["rr0"], "rr0", shapes["rr0"], writable=False)
    gz = _require_cuda_array(cupy, metrics["gz"], "gz", shapes["gz"], writable=False)
    from .cuda_immutable_metrics import require_cuda_argument_aliases
    require_cuda_argument_aliases(
        {**{f"state.{index}": value for index, value in enumerate((scalar, rru, rrv, rrw, fx, fy, fz, divx, tendency))},
         "metric.c1": c1, "metric.c2": c2, "metric.rr0": rr0, "metric.gz": gz},
        immutable_metrics=metrics,
        message="scalar advection inputs, flux scratch, tendency and unproved metrics must not alias",
    )
    compiled = compiled or compile_cuda_advection5()
    stream = stream or cupy.cuda.get_current_stream()
    horizontal_words = nk * (nj + 1) * (ni + 1)
    vertical_words = (nk - 1) * nj * ni
    tendency_words = nk * nj * ni
    threads = (CUDA_ADVECTION5_THREADS, 1, 1)
    grid_h = ((horizontal_words + CUDA_ADVECTION5_THREADS - 1) // CUDA_ADVECTION5_THREADS, 1, 1)
    grid_v = ((vertical_words + CUDA_ADVECTION5_THREADS - 1) // CUDA_ADVECTION5_THREADS, 1, 1)
    grid_t = ((tendency_words + CUDA_ADVECTION5_THREADS - 1) // CUDA_ADVECTION5_THREADS, 1, 1)
    flag = cupy.uint32(1 if use_weno else 0)
    compiled.kernels[4](grid_h, threads, (scalar, rru, rrv, fx, fy, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng), flag, cupy.float64(weps)), stream=stream)
    compiled.kernels[5](grid_v, threads, (scalar, rrw, c1, c2, fz, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng), flag, cupy.float64(weps)), stream=stream)
    obc = scalar_open_boundary_code(lateral_sides)
    compiled.kernels[6](grid_t, threads, (scalar, fx, fy, fz, divx, rr0, gz, tendency, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.uint32(ng), cupy.float32(dx), cupy.float32(dy), cupy.float32(dz), rru, rrv, cupy.uint32(obc)), stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return CudaScalarAdvection5Launch(
        event,
        compiled,
        {
            "schema": CUDA_ADVECTION5_SCHEMA,
            "claim_limit": CUDA_ADVECTION5_CLAIM_LIMIT,
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
            "stage": {"nrk": nrk, "nrkmax": CM1_NRKMAX, "wflag": wflag, "uses_weno": use_weno},
            "weno": {"order": 5, "siform": CM1_WENO_SIFORM, "weps_f64": weps},
            "interior_bounds_one_based": advection5_full_domain_interior_bounds(ni, nj),
            "lateral_sides": None if lateral_sides is None else list(lateral_sides),
            "tendency_cells": "tile-local interior 3<=i<ni-3, 3<=j<nj-3" if obc == 0 else "every cell (adv.F:273-281); open-side inflow edits adv.F:214-340",
            "source_anchors": dict(SOURCE_ANCHORS),
            "kernel": {
                "abi": CUDA_ADVECTION5_ABI,
                "source_sha256": CUDA_ADVECTION5_SOURCE_SHA256,
                "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
                "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
                "target_arch": compiled.target_arch,
                "ptx_options": list(compiled.ptx_options),
                "cubin_options": list(compiled.cubin_options),
            },
            "runtime": dict(compiled.runtime_identity),
        },
    )


__all__ = [
    "Advection5Selection",
    "CM1_NRKMAX",
    "CM1_WENO_SIFORM",
    "CUDA_ADVECTION5_ABI",
    "CUDA_ADVECTION5_CLAIM_LIMIT",
    "CUDA_ADVECTION5_KERNEL_NAMES",
    "CUDA_ADVECTION5_SCHEMA",
    "CUDA_ADVECTION5_SOURCE",
    "CUDA_ADVECTION5_EVIDENCE_SOURCE_SHA256",
    "CUDA_ADVECTION5_SOURCE_SHA256",
    "CompiledAdvection5Kernels",
    "CudaScalarAdvection5Launch",
    "CudaTerrainMassFluxLaunch",
    "DEFAULT_ADVECTION5_SELECTION",
    "SCALAR_TILE_LOCAL_OBC",
    "SOURCE_ANCHORS",
    "advection5_full_domain_interior_bounds",
    "scalar_open_boundary_code",
    "compile_cuda_advection5",
    "cpu_scalar_advection5_inplace",
    "cpu_terrain_mass_flux5_inplace",
    "launch_cuda_scalar_advection5",
    "launch_cuda_terrain_mass_flux5",
    "reconstruct_face5_f32",
    "scalar_advection_uses_weno",
    "validate_advection5_selection",
]
