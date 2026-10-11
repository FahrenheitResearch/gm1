"""CM1 r21.1 terrain SGS stresses and momentum stress-divergence tendencies.

This is the selected serial, Cartesian, four-open-boundary LES slice.  It
consumes the live ``rho*Sij`` carriers produced by :mod:`cuda_terrain_deformation`
and the live ``kmh/kmv`` carriers produced by :mod:`cuda_sgs_tke`, applies
``gettau``, and then applies the terrain ``turbu/turbv/turbw`` branches with
CM1's selected implicit vertical solves.  CUDA is imported only by compile or
launch entry points.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from cm1core.errors import refuse


CUDA_SGS_STRESS_SCHEMA = "gpuwm-cm1.cuda-sgs-stress-tendency/v1"
CUDA_SGS_STRESS_ABI = "gpuwm-cm1.cuda-sgs-stress-tendency-kernels/v1"
CUDA_SGS_STRESS_THREADS = 128
CUDA_SGS_STRESS_MAX_NK = 75
CUDA_SGS_STRESS_KERNEL_NAMES = (
    "cm1_sgs_build_normal_v1",
    "cm1_sgs_build_t12_v1",
    "cm1_sgs_build_t13_v1",
    "cm1_sgs_build_t23_v1",
    "cm1_sgs_turbu_terrain_v1",
    "cm1_sgs_turbv_terrain_v1",
    "cm1_sgs_turbw_terrain_v1",
)
CUDA_SGS_STRESS_CLAIM_LIMIT = (
    "CM1 r21.1 gettau plus terrain turbu/turbv/turbw stress-divergence for "
    "serial Cartesian uniform square-horizontal grids (dx=dy), four open lateral boundaries, "
    "cm1setup=1, dohturb/dovturb true, doimpl=1, vialpha=1, vibeta=0, and "
    "free-slip top and a bbc=1/2/3 bottom whose non-free-slip lower face is "
    "supplied by cm1core.cuda_surface_drag; supplied live "
    "rho*Sij/km/rho/metric/state carriers only; non-open lateral boundaries, MPI halos, axisymmetry, "
    "two-part flow, immersed boundaries, other SGS models, RK integration, "
    "forecast, storm, and tornado claims are excluded"
)

SOURCE_ANCHORS = {
    "gettau_dispatch": "src/turb.F:1601-1638",
    "gettau": "src/turb.F:4065-4506",
    "surface_free_slip": "src/turb.F:4536-4545",
    "sfcstress": "src/turb.F:4515-4702",
    "sfcstress_dispatch": "src/turb.F:4414-4418",
    "turbu": "src/turbtend.F:1052-1361",
    "turbuz": "src/turbtend.F:1369-1808",
    "turbv": "src/turbtend.F:1816-2128",
    "turbvz": "src/turbtend.F:2136-2465",
    "turbw": "src/turbtend.F:2473-2885",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "src/turb.F": "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d",
    "src/turbtend.F": "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de",
}


@dataclass(frozen=True, slots=True)
class SgsStressSelection:
    terrain: bool = True
    axisymmetric: int = 0
    lateral_boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)
    serial_full_domain: bool = True
    cm1setup: int = 1
    dohturb: bool = True
    dovturb: bool = True
    doimpl: int = 1
    bbc: int = 1
    tbc: int = 1
    dot2p: bool = False


DEFAULT_SGS_STRESS_SELECTION = SgsStressSelection()
# src/turb.F:4536-4605.  bbc selects the lower face of t13/t23 only; every
# other selector still pins one gettau/turbu branch.  bbc=2 and bbc=3 replace
# the free-slip copy this image writes at k=1 with the values
# cm1core.cuda_surface_drag computes, so the momentum stress divergence reads
# a surface stress instead of a zero one.  The three admitted lower
# boundaries are therefore the three CM1 has (src/param.F:1518-1532).
ADMITTED_BBC = (1, 2, 3)


def _validate_selection(selection: SgsStressSelection) -> None:
    if not isinstance(selection, SgsStressSelection):
        raise TypeError("selection must be an SgsStressSelection")
    if selection.bbc not in ADMITTED_BBC:
        refuse(
            code="CM1-SGS-STRESS-BBC",
            gate="physics-seam",
            option="sgs_stress_selection.bbc",
            value=selection.bbc,
            reason=(
                "CM1 has exactly three wind lower boundary conditions; any "
                "other value would leave t13 and t23 at k=1 holding the "
                "free-slip copy of the level above while the run claimed a "
                "surface it never applied"
            ),
            remedy="select bbc 1 (free slip), 2 (no slip) or 3 (semi slip)",
        )
    if replace(selection, bbc=1) != DEFAULT_SGS_STRESS_SELECTION:
        refuse(
            code="CM1-SGS-STRESS-UNSUPPORTED",
            gate="physics-seam",
            option="sgs_stress_selection",
            value={name: getattr(selection, name) for name in selection.__dataclass_fields__},
            reason="another selection changes the gettau or momentum-diffusion source branch",
            remedy="use the sealed real-product selection or add an independent native oracle",
        )


def _require_surface_binding(selection: SgsStressSelection, surface: Any | None) -> None:
    """Bind ``bbc`` to the operator that actually writes the lower face.

    ``src/turb.F:4536-4545`` is the free-slip branch this image already
    writes at k=1; ``bbc=2`` and ``bbc=3`` are written by
    :mod:`cm1core.cuda_surface_drag`.  A selection and a binding that
    disagree would integrate one lower boundary and publish another, so the
    two are checked against each other by value.
    """

    from cm1core.cuda_surface_drag import SurfaceDragBinding

    if surface is not None and not isinstance(surface, SurfaceDragBinding):
        raise TypeError("surface must be a SurfaceDragBinding")
    declared = 1 if surface is None else surface.selection.bbc
    if declared != selection.bbc:
        refuse(
            code="CM1-SGS-STRESS-SURFACE-BINDING",
            gate="physics-seam",
            option="sgs_stress_selection.bbc",
            value={"selection_bbc": selection.bbc, "bound_bbc": declared},
            reason=(
                "the selected lower boundary condition and the bound surface "
                "operator name different values of bbc, so the momentum "
                "stress divergence would consume one lower face while the "
                "receipt published the other"
            ),
            remedy=(
                "pass the SurfaceDragBinding whose selection.bbc equals "
                "selection.bbc, or leave both at bbc=1"
            ),
        )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max(ni, nj, nk) > (1 << 31) - 8:
        raise ValueError("SGS dimensions plus halos exceed signed CUDA indexing")
    if (nk + 1) * (nj + 1) * (ni + 1) > ((1 << 63) - 1) // 4:
        raise ValueError("SGS float32 allocation exceeds signed size_t")


def _spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _validate_square_spacing(dx: float, dy: float) -> None:
    if dx != dy:
        refuse(
            code="CM1-SGS-STRESS-SPACING",
            gate="physics-seam",
            option="grid.dx_dy",
            value={"dx": dx, "dy": dy},
            reason="the sealed stress-divergence CUDA image shares one horizontal reciprocal between transposed terms",
            remedy="use the admitted 100 m square product grid or admit a distinct rectangular-grid kernel with an oracle",
        )


def sgs_stress_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    u = (nk, nj, ni + 1)
    v = (nk, nj + 1, ni)
    s12 = (nk, nj + 1, ni + 1)
    s13 = (nk + 1, nj, ni + 1)
    s23 = (nk + 1, nj + 1, ni)
    return {
        "s11": mass, "s22": mass, "s33": mass, "s12": s12,
        "s13": s13, "s23": s23, "kmh": zface, "kmv": zface,
        "rho": mass, "rf": zface, "mh": mass, "mf": zface,
        "gz": (nj, ni), "rgz": (nj, ni), "gzu": (nj, ni + 1),
        "rgzu": (nj, ni + 1), "gzv": (nj + 1, ni),
        "rgzv": (nj + 1, ni), "gxu": (nk + 1, nj, ni + 1),
        "gyv": (nk + 1, nj + 1, ni), "u3d": u, "v3d": v,
        "w3d": zface, "t11": mass, "t22": mass, "t33": mass,
        "t12": s12, "t13": s13, "t23": s23, "uten": u,
        "vten": v, "wten": zface,
    }


_INPUTS = (
    "s11", "s22", "s33", "s12", "s13", "s23", "kmh", "kmv",
    "rho", "rf", "mh", "mf", "gz", "rgz", "gzu", "rgzu", "gzv",
    "rgzv", "gxu", "gyv", "u3d", "v3d", "w3d",
)
_OUTPUTS = ("t11", "t22", "t33", "t12", "t13", "t23", "uten", "vten", "wten")


def _require_host(arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> tuple[Any, dict[str, Any]]:
    np = importlib.import_module("numpy")
    shapes = sgs_stress_shapes(ni, nj, nk)
    missing = [name for name in _INPUTS + _OUTPUTS if name not in arrays]
    if missing:
        raise ValueError(f"missing SGS stress arrays: {', '.join(missing)}")
    out: dict[str, Any] = {}
    for name in _INPUTS + _OUTPUTS:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must have C-order shape {shapes[name]}")
        if name in _OUTPUTS and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        out[name] = value
    if len({int(v.ctypes.data) for v in out.values()}) != len(out):
        raise ValueError("SGS stress arrays must not alias")
    return np, out


def _f(np: Any, value: Any) -> Any:
    return np.float32(value)


def _a(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _s(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _m(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


def _d(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) / np.float32(y))


def _clamp(value: int, size: int) -> int:
    return min(max(value, 0), size - 1)


def _cell(value: Any, k: int, j: int, i: int, nj: int, ni: int) -> Any:
    if value.ndim == 2:
        return value[_clamp(j, nj), _clamp(i, ni)]
    return value[k, _clamp(j, nj), _clamp(i, ni)]


def _face(value: Any, k: int, j: int, i: int, nj: int, ni: int) -> Any:
    return value[k, _clamp(j, nj), _clamp(i, ni)]


def _atw(np: Any, value: Any, f: int, j: int, i: int, nk: int, nj: int, ni: int) -> Any:
    if f == 0:
        return _a(np, _a(np, _m(np, 1.875, _cell(value, 0, j, i, nj, ni)), _m(np, -1.25, _cell(value, 1, j, i, nj, ni))), _m(np, 0.375, _cell(value, 2, j, i, nj, ni)))
    if f == nk:
        return _a(np, _a(np, _m(np, 1.875, _cell(value, nk - 1, j, i, nj, ni)), _m(np, -1.25, _cell(value, nk - 2, j, i, nj, ni))), _m(np, 0.375, _cell(value, nk - 3, j, i, nj, ni)))
    return _m(np, 0.5, _a(np, _cell(value, f - 1, j, i, nj, ni), _cell(value, f, j, i, nj, ni)))


def _atw_xy(np: Any, value: Any, f: int, jf: int, iface: int, nk: int) -> Any:
    if f == 0:
        return _a(np, _a(np, _m(np, 1.875, value[0, jf, iface]), _m(np, -1.25, value[1, jf, iface])), _m(np, 0.375, value[2, jf, iface]))
    if f == nk:
        return _a(np, _a(np, _m(np, 1.875, value[nk - 1, jf, iface]), _m(np, -1.25, value[nk - 2, jf, iface])), _m(np, 0.375, value[nk - 3, jf, iface]))
    return _m(np, 0.5, _a(np, value[f - 1, jf, iface], value[f, jf, iface]))


def cpu_sgs_stress_tendency_inplace(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int,
    dx: float, dy: float, dz: float, dt: float,
    selection: SgsStressSelection = DEFAULT_SGS_STRESS_SELECTION,
) -> dict[str, int]:
    """Default-REAL scalar transcription of selected gettau/turbu/v/w."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz, dt = (_spacing(n, v) for n, v in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    _validate_square_spacing(dx, dy)
    np, a = _require_host(arrays, ni, nj, nk)
    q = np.float32(0.25); half = np.float32(0.5); zero = np.float32(0.0)
    rdx = _d(np, 1.0, dx); rdy = _d(np, 1.0, dy); rdz = _d(np, 1.0, dz)
    dt32 = np.float32(dt); zt = _m(np, nk, dz)

    # turb.F:gettau.  Open-cell coefficient halos are constant extensions.
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                tem = _a(np, a["kmh"][k, j, i], a["kmh"][k + 1, j, i])
                a["t11"][k, j, i] = _m(np, a["s11"][k, j, i], tem)
                a["t22"][k, j, i] = _m(np, a["s22"][k, j, i], tem)
                a["t33"][k, j, i] = _m(np, a["s33"][k, j, i], tem)
        for jf in range(nj + 1):
            for iface in range(ni + 1):
                total = zero
                for kf in (k, k + 1):
                    for jj in (jf - 1, jf):
                        for ii in (iface - 1, iface):
                            total = _a(np, total, _face(a["kmh"], kf, jj, ii, nj, ni))
                a["t12"][k, jf, iface] = _m(np, a["s12"][k, jf, iface], _m(np, q, total))
    for f in range(1, nk):
        for j in range(nj):
            for iface in range(ni + 1):
                a["t13"][f, j, iface] = _m(np, a["s13"][f, j, iface], _a(np, _face(a["kmv"], f, j, iface - 1, nj, ni), _face(a["kmv"], f, j, iface, nj, ni)))
        for jf in range(nj + 1):
            for i in range(ni):
                a["t23"][f, jf, i] = _m(np, a["s23"][f, jf, i], _a(np, _face(a["kmv"], f, jf - 1, i, nj, ni), _face(a["kmv"], f, jf, i, nj, ni)))
    a["t13"][0] = a["t13"][1]; a["t13"][nk] = a["t13"][nk - 1]
    a["t23"][0] = a["t23"][1]; a["t23"][nk] = a["t23"][nk - 1]

    tema_u = _m(np, -0.0625, _m(np, dt32, _m(np, rdz, rdz)))
    temc_u = _m(np, 0.5, _m(np, dt32, rdz))
    rdt_u = _d(np, 0.5, dt32)

    # turbu/turbuz: one independent implicit column per u point.
    for j in range(nj):
        for iface in range(ni + 1):
            left, right = iface - 1, iface
            hx = np.empty(nk, np.float32); hy = np.empty(nk, np.float32)
            cp = np.empty(nk, np.float32); dp = np.empty(nk, np.float32)
            sol = np.empty(nk, np.float32); cross = np.zeros(nk + 1, np.float32)
            for k in range(nk):
                rg_l = _cell(a["rgz"], 0, j, left, nj, ni); rg_r = _cell(a["rgz"], 0, j, right, nj, ni)
                x = _m(np, a["gzu"][j, iface], _m(np, _s(np, _m(np, _cell(a["t11"], k, j, right, nj, ni), rg_r), _m(np, _cell(a["t11"], k, j, left, nj, ni), rg_l)), rdx))
                up = _m(np, a["gxu"][k + 1, j, iface], _a(np, _atw(np, a["t11"], k + 1, j, left, nk, nj, ni), _atw(np, a["t11"], k + 1, j, right, nk, nj, ni)))
                lo = _m(np, a["gxu"][k, j, iface], _a(np, _atw(np, a["t11"], k, j, left, nk, nj, ni), _atw(np, a["t11"], k, j, right, nk, nj, ni)))
                hx[k] = _a(np, x, _m(np, 0.5, _m(np, _s(np, up, lo), rdz)))
                r1 = _m(np, q, _a(np, _a(np, _cell(a["rgz"], 0, j - 1, left, nj, ni), _cell(a["rgz"], 0, j, right, nj, ni)), _a(np, _cell(a["rgz"], 0, j, left, nj, ni), _cell(a["rgz"], 0, j - 1, right, nj, ni))))
                r2 = _m(np, q, _a(np, _a(np, _cell(a["rgz"], 0, j + 1, left, nj, ni), _cell(a["rgz"], 0, j, right, nj, ni)), _a(np, _cell(a["rgz"], 0, j, left, nj, ni), _cell(a["rgz"], 0, j + 1, right, nj, ni))))
                y = _m(np, a["gzu"][j, iface], _m(np, _s(np, _m(np, a["t12"][k, j + 1, iface], r2), _m(np, a["t12"][k, j, iface], r1)), rdy))
                su = _m(np, _s(np, zt, _m(np, k + 1, dz)), _a(np, _atw_xy(np, a["t12"], k + 1, j, iface, nk), _atw_xy(np, a["t12"], k + 1, j + 1, iface, nk)))
                sl = _m(np, _s(np, zt, _m(np, k, dz)), _a(np, _atw_xy(np, a["t12"], k, j, iface, nk), _atw_xy(np, a["t12"], k, j + 1, iface, nk)))
                y2 = _m(np, 0.5, _m(np, _s(np, su, sl), _m(np, a["gzu"][j, iface], _m(np, _s(np, r2, r1), _m(np, rdy, rdz)))))
                hy[k] = _a(np, y, y2)
                if iface in (0, ni): hx[k] = zero
                if j in (0, nj - 1): hy[k] = zero

            rho_lr0 = _a(np, _cell(a["rho"], 0, j, left, nj, ni), _cell(a["rho"], 0, j, right, nj, ni))
            rru0 = _d(np, 1.0, _m(np, half, rho_lr0)); mh_lr = _a(np, _cell(a["mh"], 0, j, left, nj, ni), _cell(a["mh"], 0, j, right, nj, ni))
            r2v = _m(np, _a(np, _face(a["kmv"], 1, j, left, nj, ni), _face(a["kmv"], 1, j, right, nj, ni)), _m(np, _a(np, _face(a["mf"], 1, j, left, nj, ni), _face(a["mf"], 1, j, right, nj, ni)), _m(np, _a(np, _face(a["rf"], 1, j, left, nj, ni), _face(a["rf"], 1, j, right, nj, ni)), _m(np, mh_lr, rru0))))
            cfc = _m(np, tema_u, r2v); cfb = _s(np, 1.0, cfc)
            cfd = _s(np, a["u3d"][0, j, iface], _m(np, temc_u, _m(np, a["t13"][0, j, iface], _m(np, mh_lr, rru0))))
            inv = _d(np, 1.0, cfb); cp[0] = _m(np, -1.0, _m(np, cfc, inv)); dp[0] = _m(np, cfd, inv)
            for k in range(1, nk - 1):
                rho_lr = _a(np, _cell(a["rho"], k, j, left, nj, ni), _cell(a["rho"], k, j, right, nj, ni)); rru0 = _d(np, 1.0, _m(np, half, rho_lr)); mh_lr = _a(np, _cell(a["mh"], k, j, left, nj, ni), _cell(a["mh"], k, j, right, nj, ni))
                common = _m(np, mh_lr, rru0)
                r1v = _m(np, _a(np, _face(a["kmv"], k, j, left, nj, ni), _face(a["kmv"], k, j, right, nj, ni)), _m(np, _a(np, _face(a["mf"], k, j, left, nj, ni), _face(a["mf"], k, j, right, nj, ni)), _m(np, _a(np, _face(a["rf"], k, j, left, nj, ni), _face(a["rf"], k, j, right, nj, ni)), common)))
                r2v = _m(np, _a(np, _face(a["kmv"], k + 1, j, left, nj, ni), _face(a["kmv"], k + 1, j, right, nj, ni)), _m(np, _a(np, _face(a["mf"], k + 1, j, left, nj, ni), _face(a["mf"], k + 1, j, right, nj, ni)), _m(np, _a(np, _face(a["rf"], k + 1, j, left, nj, ni), _face(a["rf"], k + 1, j, right, nj, ni)), common)))
                cfa = _m(np, tema_u, r1v); cfc = _m(np, tema_u, r2v); cfb = _s(np, _s(np, 1.0, cfa), cfc)
                inv = _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb)); cp[k] = _m(np, -1.0, _m(np, cfc, inv)); dp[k] = _m(np, _s(np, a["u3d"][k, j, iface], _m(np, cfa, dp[k - 1])), inv)
                wm = _m(np, half, _a(np, a["w3d"][k, j, _clamp(right, ni)], a["w3d"][k + 1, j, _clamp(right, ni)])); wl = _m(np, half, _a(np, a["w3d"][k, j, _clamp(left, ni)], a["w3d"][k + 1, j, _clamp(left, ni)]))
                grad = _m(np, _s(np, _m(np, a["w3d"][k, j, _clamp(right, ni)], rg_r), _m(np, a["w3d"][k, j, _clamp(left, ni)], rg_l)), rdx)
                terrain = _m(np, 0.5, _m(np, _s(np, _m(np, _s(np, zt, _m(np, k + 0.5, dz)), _a(np, wm, wl)), _m(np, _s(np, zt, _m(np, k - 0.5, dz)), _a(np, _m(np, half, _a(np, a["w3d"][k - 1, j, _clamp(right, ni)], a["w3d"][k, j, _clamp(right, ni)])), _m(np, half, _a(np, a["w3d"][k - 1, j, _clamp(left, ni)], a["w3d"][k, j, _clamp(left, ni)]))))), _m(np, rdz, _m(np, _s(np, rg_r, rg_l), rdx))))
                grad = _a(np, grad, terrain)
                cross[k] = _m(np, grad, _m(np, q, _m(np, _a(np, _face(a["kmv"], k, j, left, nj, ni), _face(a["kmv"], k, j, right, nj, ni)), _a(np, _m(np, _cell(a["gz"], 0, j, left, nj, ni), _face(a["rf"], k, j, left, nj, ni)), _m(np, _cell(a["gz"], 0, j, right, nj, ni), _face(a["rf"], k, j, right, nj, ni))))))
            k = nk - 1
            rho_lr = _a(np, _cell(a["rho"], k, j, left, nj, ni), _cell(a["rho"], k, j, right, nj, ni)); rru0 = _d(np, 1.0, _m(np, half, rho_lr)); mh_lr = _a(np, _cell(a["mh"], k, j, left, nj, ni), _cell(a["mh"], k, j, right, nj, ni))
            r1v = _m(np, _a(np, _face(a["kmv"], k, j, left, nj, ni), _face(a["kmv"], k, j, right, nj, ni)), _m(np, _a(np, _face(a["mf"], k, j, left, nj, ni), _face(a["mf"], k, j, right, nj, ni)), _m(np, _a(np, _face(a["rf"], k, j, left, nj, ni), _face(a["rf"], k, j, right, nj, ni)), _m(np, mh_lr, rru0))))
            cfa = _m(np, tema_u, r1v); cfb = _s(np, 1.0, cfa); cfd = _a(np, a["u3d"][k, j, iface], _m(np, temc_u, _m(np, a["t13"][nk, j, iface], _m(np, mh_lr, rru0))))
            sol[k] = _m(np, _s(np, cfd, _m(np, cfa, dp[k - 1])), _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb)))
            # top cross term and back substitution
            rg_l = _cell(a["rgz"], 0, j, left, nj, ni); rg_r = _cell(a["rgz"], 0, j, right, nj, ni)
            wm = _m(np, half, _a(np, a["w3d"][k, j, _clamp(right, ni)], a["w3d"][k + 1, j, _clamp(right, ni)])); wl = _m(np, half, _a(np, a["w3d"][k, j, _clamp(left, ni)], a["w3d"][k + 1, j, _clamp(left, ni)]))
            grad = _m(np, _s(np, _m(np, a["w3d"][k, j, _clamp(right, ni)], rg_r), _m(np, a["w3d"][k, j, _clamp(left, ni)], rg_l)), rdx)
            terrain = _m(np, 0.5, _m(np, _s(np, _m(np, _s(np, zt, _m(np, k + 0.5, dz)), _a(np, wm, wl)), _m(np, _s(np, zt, _m(np, k - 0.5, dz)), _a(np, _m(np, half, _a(np, a["w3d"][k - 1, j, _clamp(right, ni)], a["w3d"][k, j, _clamp(right, ni)])), _m(np, half, _a(np, a["w3d"][k - 1, j, _clamp(left, ni)], a["w3d"][k, j, _clamp(left, ni)]))))), _m(np, rdz, _m(np, _s(np, rg_r, rg_l), rdx))))
            cross[k] = _m(np, _a(np, grad, terrain), _m(np, q, _m(np, _a(np, _face(a["kmv"], k, j, left, nj, ni), _face(a["kmv"], k, j, right, nj, ni)), _a(np, _m(np, _cell(a["gz"], 0, j, left, nj, ni), _face(a["rf"], k, j, left, nj, ni)), _m(np, _cell(a["gz"], 0, j, right, nj, ni), _face(a["rf"], k, j, right, nj, ni))))))
            for kk in range(nk - 2, -1, -1): sol[kk] = _a(np, _m(np, cp[kk], sol[kk + 1]), dp[kk])
            for k in range(nk):
                rho_lr = _a(np, _cell(a["rho"], k, j, left, nj, ni), _cell(a["rho"], k, j, right, nj, ni)); mh_lr = _a(np, _cell(a["mh"], k, j, left, nj, ni), _cell(a["mh"], k, j, right, nj, ni))
                vert = _m(np, rho_lr, _m(np, _s(np, sol[k], a["u3d"][k, j, iface]), rdt_u)); vert = _a(np, vert, _m(np, _s(np, cross[k + 1] if k + 1 < nk else zero, cross[k]), _m(np, rdz, _m(np, half, mh_lr))))
                tendency = _d(np, _a(np, _a(np, hx[k], hy[k]), vert), _m(np, half, rho_lr)); a["uten"][k, j, iface] = _a(np, a["uten"][k, j, iface], tendency)

    # v is the exact x/y transpose of u with t22/t23/gyv.
    for jface in range(nj + 1):
        for i in range(ni):
            south, north = jface - 1, jface
            hx = np.empty(nk, np.float32); hy = np.empty(nk, np.float32); cp = np.empty(nk, np.float32); dp = np.empty(nk, np.float32); sol = np.empty(nk, np.float32); cross = np.zeros(nk + 1, np.float32)
            for k in range(nk):
                r1 = _m(np, q, _a(np, _a(np, _cell(a["rgz"], 0, south, i - 1, nj, ni), _cell(a["rgz"], 0, north, i, nj, ni)), _a(np, _cell(a["rgz"], 0, north, i - 1, nj, ni), _cell(a["rgz"], 0, south, i, nj, ni))))
                r2 = _m(np, q, _a(np, _a(np, _cell(a["rgz"], 0, south, i + 1, nj, ni), _cell(a["rgz"], 0, north, i, nj, ni)), _a(np, _cell(a["rgz"], 0, north, i + 1, nj, ni), _cell(a["rgz"], 0, south, i, nj, ni))))
                x = _m(np, a["gzv"][jface, i], _m(np, _s(np, _m(np, a["t12"][k, jface, i + 1], r2), _m(np, a["t12"][k, jface, i], r1)), rdx))
                su = _m(np, _s(np, zt, _m(np, k + 1, dz)), _a(np, _atw_xy(np, a["t12"], k + 1, jface, i, nk), _atw_xy(np, a["t12"], k + 1, jface, i + 1, nk))); sl = _m(np, _s(np, zt, _m(np, k, dz)), _a(np, _atw_xy(np, a["t12"], k, jface, i, nk), _atw_xy(np, a["t12"], k, jface, i + 1, nk)))
                hx[k] = _a(np, x, _m(np, 0.5, _m(np, _s(np, su, sl), _m(np, a["gzv"][jface, i], _m(np, _s(np, r2, r1), _m(np, rdx, rdz))))))
                rg_s = _cell(a["rgz"], 0, south, i, nj, ni); rg_n = _cell(a["rgz"], 0, north, i, nj, ni)
                y = _m(np, a["gzv"][jface, i], _m(np, _s(np, _m(np, _cell(a["t22"], k, north, i, nj, ni), rg_n), _m(np, _cell(a["t22"], k, south, i, nj, ni), rg_s)), rdy))
                up = _m(np, a["gyv"][k + 1, jface, i], _a(np, _atw(np, a["t22"], k + 1, south, i, nk, nj, ni), _atw(np, a["t22"], k + 1, north, i, nk, nj, ni))); lo = _m(np, a["gyv"][k, jface, i], _a(np, _atw(np, a["t22"], k, south, i, nk, nj, ni), _atw(np, a["t22"], k, north, i, nk, nj, ni)))
                hy[k] = _a(np, y, _m(np, 0.5, _m(np, _s(np, up, lo), rdz)))
                if i in (0, ni - 1): hx[k] = zero
                if jface in (0, nj): hy[k] = zero
            rho_sn = _a(np, _cell(a["rho"], 0, south, i, nj, ni), _cell(a["rho"], 0, north, i, nj, ni)); rrv0 = _d(np, 1.0, _m(np, half, rho_sn)); mh_sn = _a(np, _cell(a["mh"], 0, south, i, nj, ni), _cell(a["mh"], 0, north, i, nj, ni))
            r2v = _m(np, _a(np, _face(a["kmv"], 1, south, i, nj, ni), _face(a["kmv"], 1, north, i, nj, ni)), _m(np, _a(np, _face(a["mf"], 1, south, i, nj, ni), _face(a["mf"], 1, north, i, nj, ni)), _m(np, _a(np, _face(a["rf"], 1, south, i, nj, ni), _face(a["rf"], 1, north, i, nj, ni)), _m(np, mh_sn, rrv0))))
            cfc = _m(np, tema_u, r2v); cfb = _s(np, 1.0, cfc); cfd = _s(np, a["v3d"][0, jface, i], _m(np, temc_u, _m(np, a["t23"][0, jface, i], _m(np, mh_sn, rrv0)))); inv = _d(np, 1.0, cfb); cp[0] = _m(np, -1.0, _m(np, cfc, inv)); dp[0] = _m(np, cfd, inv)
            for k in range(1, nk - 1):
                rho_sn = _a(np, _cell(a["rho"], k, south, i, nj, ni), _cell(a["rho"], k, north, i, nj, ni)); rrv0 = _d(np, 1.0, _m(np, half, rho_sn)); mh_sn = _a(np, _cell(a["mh"], k, south, i, nj, ni), _cell(a["mh"], k, north, i, nj, ni)); common = _m(np, mh_sn, rrv0)
                r1v = _m(np, _a(np, _face(a["kmv"], k, south, i, nj, ni), _face(a["kmv"], k, north, i, nj, ni)), _m(np, _a(np, _face(a["mf"], k, south, i, nj, ni), _face(a["mf"], k, north, i, nj, ni)), _m(np, _a(np, _face(a["rf"], k, south, i, nj, ni), _face(a["rf"], k, north, i, nj, ni)), common)))
                r2v = _m(np, _a(np, _face(a["kmv"], k + 1, south, i, nj, ni), _face(a["kmv"], k + 1, north, i, nj, ni)), _m(np, _a(np, _face(a["mf"], k + 1, south, i, nj, ni), _face(a["mf"], k + 1, north, i, nj, ni)), _m(np, _a(np, _face(a["rf"], k + 1, south, i, nj, ni), _face(a["rf"], k + 1, north, i, nj, ni)), common)))
                cfa = _m(np, tema_u, r1v); cfc = _m(np, tema_u, r2v); cfb = _s(np, _s(np, 1.0, cfa), cfc); inv = _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb)); cp[k] = _m(np, -1.0, _m(np, cfc, inv)); dp[k] = _m(np, _s(np, a["v3d"][k, jface, i], _m(np, cfa, dp[k - 1])), inv)
                rg_s = _cell(a["rgz"], 0, south, i, nj, ni); rg_n = _cell(a["rgz"], 0, north, i, nj, ni); wn = _m(np, half, _a(np, a["w3d"][k, _clamp(north, nj), i], a["w3d"][k + 1, _clamp(north, nj), i])); ws = _m(np, half, _a(np, a["w3d"][k, _clamp(south, nj), i], a["w3d"][k + 1, _clamp(south, nj), i])); prevn = _m(np, half, _a(np, a["w3d"][k - 1, _clamp(north, nj), i], a["w3d"][k, _clamp(north, nj), i])); prevs = _m(np, half, _a(np, a["w3d"][k - 1, _clamp(south, nj), i], a["w3d"][k, _clamp(south, nj), i]))
                grad = _m(np, _s(np, _m(np, a["w3d"][k, _clamp(north, nj), i], rg_n), _m(np, a["w3d"][k, _clamp(south, nj), i], rg_s)), rdy); terrain = _m(np, 0.5, _m(np, _s(np, _m(np, _s(np, zt, _m(np, k + 0.5, dz)), _a(np, wn, ws)), _m(np, _s(np, zt, _m(np, k - 0.5, dz)), _a(np, prevn, prevs))), _m(np, rdz, _m(np, _s(np, rg_n, rg_s), rdy))))
                cross[k] = _m(np, _a(np, grad, terrain), _m(np, q, _m(np, _a(np, _face(a["kmv"], k, south, i, nj, ni), _face(a["kmv"], k, north, i, nj, ni)), _a(np, _m(np, _cell(a["gz"], 0, south, i, nj, ni), _face(a["rf"], k, south, i, nj, ni)), _m(np, _cell(a["gz"], 0, north, i, nj, ni), _face(a["rf"], k, north, i, nj, ni))))))
            k = nk - 1; rho_sn = _a(np, _cell(a["rho"], k, south, i, nj, ni), _cell(a["rho"], k, north, i, nj, ni)); rrv0 = _d(np, 1.0, _m(np, half, rho_sn)); mh_sn = _a(np, _cell(a["mh"], k, south, i, nj, ni), _cell(a["mh"], k, north, i, nj, ni))
            r1v = _m(np, _a(np, _face(a["kmv"], k, south, i, nj, ni), _face(a["kmv"], k, north, i, nj, ni)), _m(np, _a(np, _face(a["mf"], k, south, i, nj, ni), _face(a["mf"], k, north, i, nj, ni)), _m(np, _a(np, _face(a["rf"], k, south, i, nj, ni), _face(a["rf"], k, north, i, nj, ni)), _m(np, mh_sn, rrv0)))); cfa = _m(np, tema_u, r1v); cfb = _s(np, 1.0, cfa); cfd = _a(np, a["v3d"][k, jface, i], _m(np, temc_u, _m(np, a["t23"][nk, jface, i], _m(np, mh_sn, rrv0)))); sol[k] = _m(np, _s(np, cfd, _m(np, cfa, dp[k - 1])), _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb)))
            rg_s = _cell(a["rgz"], 0, south, i, nj, ni); rg_n = _cell(a["rgz"], 0, north, i, nj, ni); wn = _m(np, half, _a(np, a["w3d"][k, _clamp(north, nj), i], a["w3d"][k + 1, _clamp(north, nj), i])); ws = _m(np, half, _a(np, a["w3d"][k, _clamp(south, nj), i], a["w3d"][k + 1, _clamp(south, nj), i])); prevn = _m(np, half, _a(np, a["w3d"][k - 1, _clamp(north, nj), i], a["w3d"][k, _clamp(north, nj), i])); prevs = _m(np, half, _a(np, a["w3d"][k - 1, _clamp(south, nj), i], a["w3d"][k, _clamp(south, nj), i])); grad = _m(np, _s(np, _m(np, a["w3d"][k, _clamp(north, nj), i], rg_n), _m(np, a["w3d"][k, _clamp(south, nj), i], rg_s)), rdy); terrain = _m(np, 0.5, _m(np, _s(np, _m(np, _s(np, zt, _m(np, k + 0.5, dz)), _a(np, wn, ws)), _m(np, _s(np, zt, _m(np, k - 0.5, dz)), _a(np, prevn, prevs))), _m(np, rdz, _m(np, _s(np, rg_n, rg_s), rdy)))); cross[k] = _m(np, _a(np, grad, terrain), _m(np, q, _m(np, _a(np, _face(a["kmv"], k, south, i, nj, ni), _face(a["kmv"], k, north, i, nj, ni)), _a(np, _m(np, _cell(a["gz"], 0, south, i, nj, ni), _face(a["rf"], k, south, i, nj, ni)), _m(np, _cell(a["gz"], 0, north, i, nj, ni), _face(a["rf"], k, north, i, nj, ni))))))
            for kk in range(nk - 2, -1, -1): sol[kk] = _a(np, _m(np, cp[kk], sol[kk + 1]), dp[kk])
            for k in range(nk):
                rho_sn = _a(np, _cell(a["rho"], k, south, i, nj, ni), _cell(a["rho"], k, north, i, nj, ni)); mh_sn = _a(np, _cell(a["mh"], k, south, i, nj, ni), _cell(a["mh"], k, north, i, nj, ni)); vert = _m(np, rho_sn, _m(np, _s(np, sol[k], a["v3d"][k, jface, i]), rdt_u)); vert = _a(np, vert, _m(np, _s(np, cross[k + 1] if k + 1 < nk else zero, cross[k]), _m(np, rdz, _m(np, half, mh_sn)))); a["vten"][k, jface, i] = _a(np, a["vten"][k, jface, i], _d(np, _a(np, _a(np, hx[k], hy[k]), vert), _m(np, half, rho_sn)))

    # turbw, including its implicit vertical solve on internal w faces.
    tema_w = _m(np, -1.0, _m(np, dt32, _m(np, rdz, rdz))); temc_w = _m(np, dt32, rdz); rdt_w = _d(np, 1.0, dt32)
    for j in range(nj):
        for i in range(ni):
            n = nk - 1; hx = np.empty(n, np.float32); hy = np.empty(n, np.float32); cp = np.empty(n, np.float32); dp = np.empty(n, np.float32); sol = np.empty(n, np.float32)
            for p, f in enumerate(range(1, nk)):
                d1 = _m(np, q, _a(np, _a(np, a["t13"][f + 1, j, i], a["t13"][f + 1, j, i + 1]), _a(np, a["t13"][f, j, i], a["t13"][f, j, i + 1]))); d0 = _m(np, q, _a(np, _a(np, a["t13"][f, j, i], a["t13"][f, j, i + 1]), _a(np, a["t13"][f - 1, j, i], a["t13"][f - 1, j, i + 1])))
                x = _m(np, a["gz"][j, i], _m(np, _s(np, _m(np, a["t13"][f, j, i + 1], a["rgzu"][j, i + 1]), _m(np, a["t13"][f, j, i], a["rgzu"][j, i])), rdx)); x = _a(np, x, _m(np, _s(np, _m(np, _s(np, zt, _m(np, f + 0.5, dz)), d1), _m(np, _s(np, zt, _m(np, f - 0.5, dz)), d0)), _m(np, a["gz"][j, i], _m(np, _s(np, a["rgzu"][j, i + 1], a["rgzu"][j, i]), _m(np, rdx, rdz)))))
                e1 = _m(np, q, _a(np, _a(np, a["t23"][f + 1, j, i], a["t23"][f + 1, j + 1, i]), _a(np, a["t23"][f, j, i], a["t23"][f, j + 1, i]))); e0 = _m(np, q, _a(np, _a(np, a["t23"][f, j, i], a["t23"][f, j + 1, i]), _a(np, a["t23"][f - 1, j, i], a["t23"][f - 1, j + 1, i])))
                y = _m(np, a["gz"][j, i], _m(np, _s(np, _m(np, a["t23"][f, j + 1, i], a["rgzv"][j + 1, i]), _m(np, a["t23"][f, j, i], a["rgzv"][j, i])), rdy)); y = _a(np, y, _m(np, _s(np, _m(np, _s(np, zt, _m(np, f + 0.5, dz)), e1), _m(np, _s(np, zt, _m(np, f - 0.5, dz)), e0)), _m(np, a["gz"][j, i], _m(np, _s(np, a["rgzv"][j + 1, i], a["rgzv"][j, i]), _m(np, rdy, rdz)))))
                hx[p] = zero if i in (0, ni - 1) else x; hy[p] = zero if j in (0, nj - 1) else y
            f = 1; rrf = _d(np, a["mf"][f, j, i], a["rf"][f, j, i]); r2v = _m(np, _a(np, a["kmh"][f, j, i], a["kmh"][f + 1, j, i]), _m(np, a["mh"][f, j, i], _m(np, a["rho"][f, j, i], rrf))); cfc = _m(np, tema_w, r2v); cfb = _s(np, 1.0, cfc); cfd = _s(np, a["w3d"][f, j, i], _m(np, temc_w, _m(np, a["t33"][0, j, i], rrf))); inv = _d(np, 1.0, cfb); cp[0] = _m(np, -1.0, _m(np, cfc, inv)); dp[0] = _m(np, cfd, inv)
            for p, f in enumerate(range(2, nk - 1), start=1):
                rrf = _d(np, a["mf"][f, j, i], a["rf"][f, j, i]); r1v = _m(np, _a(np, a["kmh"][f - 1, j, i], a["kmh"][f, j, i]), _m(np, a["mh"][f - 1, j, i], _m(np, a["rho"][f - 1, j, i], rrf))); r2v = _m(np, _a(np, a["kmh"][f, j, i], a["kmh"][f + 1, j, i]), _m(np, a["mh"][f, j, i], _m(np, a["rho"][f, j, i], rrf))); cfa = _m(np, tema_w, r1v); cfc = _m(np, tema_w, r2v); cfb = _s(np, _s(np, 1.0, cfa), cfc); inv = _d(np, 1.0, _a(np, _m(np, cfa, cp[p - 1]), cfb)); cp[p] = _m(np, -1.0, _m(np, cfc, inv)); dp[p] = _m(np, _s(np, a["w3d"][f, j, i], _m(np, cfa, dp[p - 1])), inv)
            f = nk - 1; p = n - 1; rrf = _d(np, a["mf"][f, j, i], a["rf"][f, j, i]); r1v = _m(np, _a(np, a["kmh"][f - 1, j, i], a["kmh"][f, j, i]), _m(np, a["mh"][f - 1, j, i], _m(np, a["rho"][f - 1, j, i], rrf))); cfa = _m(np, tema_w, r1v); cfb = _s(np, 1.0, cfa); cfd = _a(np, a["w3d"][f, j, i], _m(np, temc_w, _m(np, a["t33"][nk - 1, j, i], rrf))); sol[p] = _m(np, _s(np, cfd, _m(np, cfa, dp[p - 1])), _d(np, 1.0, _a(np, _m(np, cfa, cp[p - 1]), cfb)))
            for p in range(n - 2, -1, -1): sol[p] = _a(np, _m(np, cp[p], sol[p + 1]), dp[p])
            for p, f in enumerate(range(1, nk)):
                vert = _m(np, a["rf"][f, j, i], _m(np, _s(np, sol[p], a["w3d"][f, j, i]), rdt_w)); a["wten"][f, j, i] = _a(np, a["wten"][f, j, i], _d(np, _a(np, _a(np, hx[p], hy[p]), vert), a["rf"][f, j, i]))
    return {"stress_words": nk * (3 * ni * nj + (ni + 1) * (nj + 1)) + (nk + 1) * ((ni + 1) * nj + ni * (nj + 1)), "u_tendency_words": nk * nj * (ni + 1), "v_tendency_words": nk * (nj + 1) * ni, "w_tendency_words": (nk - 1) * nj * ni}


# CUDA transcription.  Each implicit solve is one independent horizontal
# column.  Explicit round-to-nearest intrinsics and --fmad=false preserve the
# source statement boundaries used by the bounded oracle.
CUDA_SGS_STRESS_SOURCE = r'''// gpuwm-cm1.cuda-sgs-stress-tendency-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);} __device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);} __device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);} __device__ __forceinline__ float D(float x,float y){return __fdiv_rn(x,y);}
__device__ __forceinline__ int C(int x,int n){return x<0?0:(x>=n?n-1:x);} __device__ __forceinline__ unsigned long long I3(int k,int j,int i,int ny,int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float CELL(const float* a,int k,int j,int i,int nj,int ni){return a[I3(k,C(j,nj),C(i,ni),nj,ni)];}
__device__ __forceinline__ float ATW(const float* a,int f,int j,int i,int nk,int nj,int ni){if(f==0)return A(A(M(1.875f,CELL(a,0,j,i,nj,ni)),M(-1.25f,CELL(a,1,j,i,nj,ni))),M(0.375f,CELL(a,2,j,i,nj,ni)));if(f==nk)return A(A(M(1.875f,CELL(a,nk-1,j,i,nj,ni)),M(-1.25f,CELL(a,nk-2,j,i,nj,ni))),M(0.375f,CELL(a,nk-3,j,i,nj,ni)));return M(0.5f,A(CELL(a,f-1,j,i,nj,ni),CELL(a,f,j,i,nj,ni)));}
__device__ __forceinline__ float ATW12(const float* a,int f,int j,int i,int nk,int nj,int ni){unsigned long long p=I3(0,C(j,nj+1),C(i,ni+1),nj+1,ni+1);unsigned long long z=(unsigned long long)(nj+1)*(ni+1);if(f==0)return A(A(M(1.875f,a[p]),M(-1.25f,a[p+z])),M(0.375f,a[p+2*z]));if(f==nk)return A(A(M(1.875f,a[p+(nk-1)*z]),M(-1.25f,a[p+(nk-2)*z])),M(0.375f,a[p+(nk-3)*z]));return M(0.5f,A(a[p+(f-1)*z],a[p+f*z]));}
extern "C" __global__ void cm1_sgs_build_normal_v1(const float*s11,const float*s22,const float*s33,const float*kmh,float*t11,float*t22,float*t33,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj*nk,st=(unsigned long long)blockDim.x*gridDim.x,z=(unsigned long long)ni*nj;for(;q<n;q+=st){unsigned k=(unsigned)(q/z);float m=A(kmh[q],kmh[q+z]);t11[q]=M(s11[q],m);t22[q]=M(s22[q],m);t33[q]=M(s33[q],m);}}
extern "C" __global__ void cm1_sgs_build_t12_v1(const float*s,const float*km,float*t,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*(nj+1u)*(ni+1u),st=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=st){unsigned i=(unsigned)(q%(ni+1u)),j=(unsigned)((q/(ni+1u))%(nj+1u)),k=(unsigned)(q/((unsigned long long)(ni+1u)*(nj+1u)));float z=0.0f;for(int dk=0;dk<2;dk++)for(int dj=-1;dj<1;dj++)for(int di=-1;di<1;di++)z=A(z,CELL(km,k+dk,(int)j+dj,(int)i+di,nj,ni));t[q]=M(s[q],M(0.25f,z));}}
extern "C" __global__ void cm1_sgs_build_t13_v1(const float*s,const float*km,float*t,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)(nk+1u)*nj*(ni+1u),st=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=st){unsigned i=(unsigned)(q%(ni+1u)),j=(unsigned)((q/(ni+1u))%nj),f=(unsigned)(q/((unsigned long long)(ni+1u)*nj));unsigned ff=f==0?1:(f==nk?nk-1:f);t[q]=M(s[I3(ff,j,i,nj,ni+1)],A(CELL(km,ff,j,(int)i-1,nj,ni),CELL(km,ff,j,i,nj,ni)));}}
extern "C" __global__ void cm1_sgs_build_t23_v1(const float*s,const float*km,float*t,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)(nk+1u)*(nj+1u)*ni,st=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=st){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%(nj+1u)),f=(unsigned)(q/((unsigned long long)ni*(nj+1u)));unsigned ff=f==0?1:(f==nk?nk-1:f);t[q]=M(s[I3(ff,j,i,nj+1,ni)],A(CELL(km,ff,(int)j-1,i,nj,ni),CELL(km,ff,j,i,nj,ni)));}}
// The three column kernels are generated from the same source-ordered scalar
// loops as the CPU function.  Compact helper bodies live below this marker.
'''

# The large CUDA column bodies are appended below to keep the reviewable source
# anchors and Python contract near the top of the module.


def _column_cuda_source() -> str:
    """Return the source-ordered u/v/w column kernels."""
    # Kept as a separate literal so source hashing still seals the exact image.
    return r'''
// COLUMN_KERNELS_V1
__device__ __forceinline__ float G(const float*a,int j,int i,int nj,int ni){return a[(unsigned long long)C(j,nj)*ni+C(i,ni)];}
extern "C" __global__ void cm1_sgs_turbu_terrain_v1(const float*t11,const float*t12,const float*t13,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,const float*gz,const float*rgz,const float*gzu,const float*gxu,const float*u,const float*w,float*uten,unsigned ni,unsigned nj,unsigned nk,float dx,float dz,float dt){unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nj*(ni+1u);if(col>=n)return;int f=(int)(col%(ni+1u)),j=(int)(col/(ni+1u)),l=f-1,r=f;float hx[75],hy[75],cp[75],dp[75],sol[75],cr[76];float rdx=D(1.0f,dx),rdz=D(1.0f,dz),zt=M((float)nk,dz),tema=M(-0.0625f,M(dt,M(rdz,rdz))),temc=M(0.5f,M(dt,rdz)),rdt=D(0.5f,dt);for(unsigned k=0;k<=nk;k++)cr[k]=0.0f;for(unsigned k=0;k<nk;k++){float rl=G(rgz,j,l,nj,ni),rr=G(rgz,j,r,nj,ni);float x=M(gzu[(unsigned long long)j*(ni+1u)+f],M(S(M(CELL(t11,k,j,r,nj,ni),rr),M(CELL(t11,k,j,l,nj,ni),rl)),rdx));float up=M(gxu[I3(k+1,j,f,nj,ni+1)],A(ATW(t11,k+1,j,l,nk,nj,ni),ATW(t11,k+1,j,r,nk,nj,ni))),lo=M(gxu[I3(k,j,f,nj,ni+1)],A(ATW(t11,k,j,l,nk,nj,ni),ATW(t11,k,j,r,nk,nj,ni)));hx[k]=A(x,M(0.5f,M(S(up,lo),rdz)));float r1=M(0.25f,A(A(G(rgz,j-1,l,nj,ni),G(rgz,j,r,nj,ni)),A(G(rgz,j,l,nj,ni),G(rgz,j-1,r,nj,ni)))),r2=M(0.25f,A(A(G(rgz,j+1,l,nj,ni),G(rgz,j,r,nj,ni)),A(G(rgz,j,l,nj,ni),G(rgz,j+1,r,nj,ni))));float y=M(gzu[(unsigned long long)j*(ni+1u)+f],M(S(M(t12[I3(k,j+1,f,nj+1,ni+1)],r2),M(t12[I3(k,j,f,nj+1,ni+1)],r1)),D(1.0f,dx)));float su=M(S(zt,M((float)k+1.0f,dz)),A(ATW12(t12,k+1,j,f,nk,nj,ni),ATW12(t12,k+1,j+1,f,nk,nj,ni))),sl=M(S(zt,M((float)k,dz)),A(ATW12(t12,k,j,f,nk,nj,ni),ATW12(t12,k,j+1,f,nk,nj,ni)));hy[k]=A(y,M(0.5f,M(S(su,sl),M(gzu[(unsigned long long)j*(ni+1u)+f],M(S(r2,r1),M(D(1.0f,dx),rdz))))));if(f==0||f==(int)ni)hx[k]=0.0f;if(j==0||j==(int)nj-1)hy[k]=0.0f;}
float rh=A(CELL(rho,0,j,l,nj,ni),CELL(rho,0,j,r,nj,ni)),ri=D(1.0f,M(0.5f,rh)),hm=A(CELL(mh,0,j,l,nj,ni),CELL(mh,0,j,r,nj,ni));float r2=A(CELL(kmv,1,j,l,nj,ni),CELL(kmv,1,j,r,nj,ni));r2=M(r2,M(A(CELL(mf,1,j,l,nj,ni),CELL(mf,1,j,r,nj,ni)),M(A(CELL(rf,1,j,l,nj,ni),CELL(rf,1,j,r,nj,ni)),M(hm,ri))));float cfc=M(tema,r2),cfb=S(1.0f,cfc),cfd=S(u[I3(0,j,f,nj,ni+1)],M(temc,M(t13[I3(0,j,f,nj,ni+1)],M(hm,ri)))),inv=D(1.0f,cfb);cp[0]=M(-1.0f,M(cfc,inv));dp[0]=M(cfd,inv);
for(unsigned k=1;k<nk-1;k++){rh=A(CELL(rho,k,j,l,nj,ni),CELL(rho,k,j,r,nj,ni));ri=D(1.0f,M(0.5f,rh));hm=A(CELL(mh,k,j,l,nj,ni),CELL(mh,k,j,r,nj,ni));float co=M(hm,ri),r1=A(CELL(kmv,k,j,l,nj,ni),CELL(kmv,k,j,r,nj,ni));r1=M(r1,M(A(CELL(mf,k,j,l,nj,ni),CELL(mf,k,j,r,nj,ni)),M(A(CELL(rf,k,j,l,nj,ni),CELL(rf,k,j,r,nj,ni)),co)));r2=A(CELL(kmv,k+1,j,l,nj,ni),CELL(kmv,k+1,j,r,nj,ni));r2=M(r2,M(A(CELL(mf,k+1,j,l,nj,ni),CELL(mf,k+1,j,r,nj,ni)),M(A(CELL(rf,k+1,j,l,nj,ni),CELL(rf,k+1,j,r,nj,ni)),co)));float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1.0f,cfa),cfc);inv=D(1.0f,A(M(cfa,cp[k-1]),cfb));cp[k]=M(-1.0f,M(cfc,inv));dp[k]=M(S(u[I3(k,j,f,nj,ni+1)],M(cfa,dp[k-1])),inv);float rl=G(rgz,j,l,nj,ni),rr=G(rgz,j,r,nj,ni),wr=M(0.5f,A(CELL(w,k,j,r,nj,ni),CELL(w,k+1,j,r,nj,ni))),wl=M(0.5f,A(CELL(w,k,j,l,nj,ni),CELL(w,k+1,j,l,nj,ni))),pr=M(0.5f,A(CELL(w,k-1,j,r,nj,ni),CELL(w,k,j,r,nj,ni))),pl=M(0.5f,A(CELL(w,k-1,j,l,nj,ni),CELL(w,k,j,l,nj,ni)));float gr=M(S(M(CELL(w,k,j,r,nj,ni),rr),M(CELL(w,k,j,l,nj,ni),rl)),rdx);float tr=M(0.5f,M(S(M(S(zt,M((float)k+0.5f,dz)),A(wr,wl)),M(S(zt,M((float)k-0.5f,dz)),A(pr,pl))),M(rdz,M(S(rr,rl),rdx))));cr[k]=M(A(gr,tr),M(0.25f,M(A(CELL(kmv,k,j,l,nj,ni),CELL(kmv,k,j,r,nj,ni)),A(M(G(gz,j,l,nj,ni),CELL(rf,k,j,l,nj,ni)),M(G(gz,j,r,nj,ni),CELL(rf,k,j,r,nj,ni))))));}
unsigned k=nk-1;rh=A(CELL(rho,k,j,l,nj,ni),CELL(rho,k,j,r,nj,ni));ri=D(1.0f,M(0.5f,rh));hm=A(CELL(mh,k,j,l,nj,ni),CELL(mh,k,j,r,nj,ni));float r1=A(CELL(kmv,k,j,l,nj,ni),CELL(kmv,k,j,r,nj,ni));r1=M(r1,M(A(CELL(mf,k,j,l,nj,ni),CELL(mf,k,j,r,nj,ni)),M(A(CELL(rf,k,j,l,nj,ni),CELL(rf,k,j,r,nj,ni)),M(hm,ri))));float cfa=M(tema,r1);cfb=S(1.0f,cfa);cfd=A(u[I3(k,j,f,nj,ni+1)],M(temc,M(t13[I3(nk,j,f,nj,ni+1)],M(hm,ri))));sol[k]=M(S(cfd,M(cfa,dp[k-1])),D(1.0f,A(M(cfa,cp[k-1]),cfb)));float rl=G(rgz,j,l,nj,ni),rr=G(rgz,j,r,nj,ni),wr=M(0.5f,A(CELL(w,k,j,r,nj,ni),CELL(w,k+1,j,r,nj,ni))),wl=M(0.5f,A(CELL(w,k,j,l,nj,ni),CELL(w,k+1,j,l,nj,ni))),pr=M(0.5f,A(CELL(w,k-1,j,r,nj,ni),CELL(w,k,j,r,nj,ni))),pl=M(0.5f,A(CELL(w,k-1,j,l,nj,ni),CELL(w,k,j,l,nj,ni)));float gr=M(S(M(CELL(w,k,j,r,nj,ni),rr),M(CELL(w,k,j,l,nj,ni),rl)),rdx),tr=M(0.5f,M(S(M(S(zt,M((float)k+0.5f,dz)),A(wr,wl)),M(S(zt,M((float)k-0.5f,dz)),A(pr,pl))),M(rdz,M(S(rr,rl),rdx))));cr[k]=M(A(gr,tr),M(0.25f,M(A(CELL(kmv,k,j,l,nj,ni),CELL(kmv,k,j,r,nj,ni)),A(M(G(gz,j,l,nj,ni),CELL(rf,k,j,l,nj,ni)),M(G(gz,j,r,nj,ni),CELL(rf,k,j,r,nj,ni))))));for(int z=(int)nk-2;z>=0;z--)sol[z]=A(M(cp[z],sol[z+1]),dp[z]);for(k=0;k<nk;k++){rh=A(CELL(rho,k,j,l,nj,ni),CELL(rho,k,j,r,nj,ni));hm=A(CELL(mh,k,j,l,nj,ni),CELL(mh,k,j,r,nj,ni));float vz=M(rh,M(S(sol[k],u[I3(k,j,f,nj,ni+1)]),rdt));vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rdz,M(0.5f,hm))));unsigned long long p=I3(k,j,f,nj,ni+1);uten[p]=A(uten[p],D(A(A(hx[k],hy[k]),vz),M(0.5f,rh)));}}
// v kernel uses the exact transpose; its compact implementation is deliberately
// separate so asymmetric terrain metrics gzu/gzv and gxu/gyv stay explicit.
extern "C" __global__ void cm1_sgs_turbv_terrain_v1(const float*t12,const float*t22,const float*t23,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,const float*gz,const float*rgz,const float*gzv,const float*gyv,const float*v,const float*w,float*vten,unsigned ni,unsigned nj,unsigned nk,float dx,float dz,float dt){unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)(nj+1u)*ni;if(col>=n)return;int i=(int)(col%ni),f=(int)(col/ni),s=f-1,r=f;float hx[75],hy[75],cp[75],dp[75],sol[75],cr[76];float rd=D(1.0f,dx),rz=D(1.0f,dz),zt=M((float)nk,dz),tema=M(-0.0625f,M(dt,M(rz,rz))),temc=M(0.5f,M(dt,rz)),rdt=D(0.5f,dt);for(unsigned k=0;k<=nk;k++)cr[k]=0.0f;for(unsigned k=0;k<nk;k++){float r1=M(0.25f,A(A(G(rgz,s,i-1,nj,ni),G(rgz,r,i,nj,ni)),A(G(rgz,r,i-1,nj,ni),G(rgz,s,i,nj,ni)))),r2=M(0.25f,A(A(G(rgz,s,i+1,nj,ni),G(rgz,r,i,nj,ni)),A(G(rgz,r,i+1,nj,ni),G(rgz,s,i,nj,ni))));float x=M(gzv[(unsigned long long)f*ni+i],M(S(M(t12[I3(k,f,i+1,nj+1,ni+1)],r2),M(t12[I3(k,f,i,nj+1,ni+1)],r1)),rd));float up=M(S(zt,M((float)k+1.0f,dz)),A(ATW12(t12,k+1,f,i,nk,nj,ni),ATW12(t12,k+1,f,i+1,nk,nj,ni))),lo=M(S(zt,M((float)k,dz)),A(ATW12(t12,k,f,i,nk,nj,ni),ATW12(t12,k,f,i+1,nk,nj,ni)));hx[k]=A(x,M(0.5f,M(S(up,lo),M(gzv[(unsigned long long)f*ni+i],M(S(r2,r1),M(rd,rz))))));float rs=G(rgz,s,i,nj,ni),rn=G(rgz,r,i,nj,ni);float y=M(gzv[(unsigned long long)f*ni+i],M(S(M(CELL(t22,k,r,i,nj,ni),rn),M(CELL(t22,k,s,i,nj,ni),rs)),rd));float yu=M(gyv[I3(k+1,f,i,nj+1,ni)],A(ATW(t22,k+1,s,i,nk,nj,ni),ATW(t22,k+1,r,i,nk,nj,ni))),yl=M(gyv[I3(k,f,i,nj+1,ni)],A(ATW(t22,k,s,i,nk,nj,ni),ATW(t22,k,r,i,nk,nj,ni)));hy[k]=A(y,M(0.5f,M(S(yu,yl),rz)));if(i==0||i==(int)ni-1)hx[k]=0.0f;if(f==0||f==(int)nj)hy[k]=0.0f;}
float rh=A(CELL(rho,0,s,i,nj,ni),CELL(rho,0,r,i,nj,ni)),ri=D(1.0f,M(0.5f,rh)),hm=A(CELL(mh,0,s,i,nj,ni),CELL(mh,0,r,i,nj,ni));float r2=A(CELL(kmv,1,s,i,nj,ni),CELL(kmv,1,r,i,nj,ni));r2=M(r2,M(A(CELL(mf,1,s,i,nj,ni),CELL(mf,1,r,i,nj,ni)),M(A(CELL(rf,1,s,i,nj,ni),CELL(rf,1,r,i,nj,ni)),M(hm,ri))));float cfc=M(tema,r2),cfb=S(1.0f,cfc),cfd=S(v[I3(0,f,i,nj+1,ni)],M(temc,M(t23[I3(0,f,i,nj+1,ni)],M(hm,ri)))),inv=D(1.0f,cfb);cp[0]=M(-1.0f,M(cfc,inv));dp[0]=M(cfd,inv);for(unsigned k=1;k<nk-1;k++){rh=A(CELL(rho,k,s,i,nj,ni),CELL(rho,k,r,i,nj,ni));ri=D(1.0f,M(0.5f,rh));hm=A(CELL(mh,k,s,i,nj,ni),CELL(mh,k,r,i,nj,ni));float co=M(hm,ri),r1=A(CELL(kmv,k,s,i,nj,ni),CELL(kmv,k,r,i,nj,ni));r1=M(r1,M(A(CELL(mf,k,s,i,nj,ni),CELL(mf,k,r,i,nj,ni)),M(A(CELL(rf,k,s,i,nj,ni),CELL(rf,k,r,i,nj,ni)),co)));r2=A(CELL(kmv,k+1,s,i,nj,ni),CELL(kmv,k+1,r,i,nj,ni));r2=M(r2,M(A(CELL(mf,k+1,s,i,nj,ni),CELL(mf,k+1,r,i,nj,ni)),M(A(CELL(rf,k+1,s,i,nj,ni),CELL(rf,k+1,r,i,nj,ni)),co)));float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1.0f,cfa),cfc);inv=D(1.0f,A(M(cfa,cp[k-1]),cfb));cp[k]=M(-1.0f,M(cfc,inv));dp[k]=M(S(v[I3(k,f,i,nj+1,ni)],M(cfa,dp[k-1])),inv);float rs=G(rgz,s,i,nj,ni),rn=G(rgz,r,i,nj,ni),wn=M(0.5f,A(CELL(w,k,r,i,nj,ni),CELL(w,k+1,r,i,nj,ni))),ws=M(0.5f,A(CELL(w,k,s,i,nj,ni),CELL(w,k+1,s,i,nj,ni))),pn=M(0.5f,A(CELL(w,k-1,r,i,nj,ni),CELL(w,k,r,i,nj,ni))),ps=M(0.5f,A(CELL(w,k-1,s,i,nj,ni),CELL(w,k,s,i,nj,ni)));float gr=M(S(M(CELL(w,k,r,i,nj,ni),rn),M(CELL(w,k,s,i,nj,ni),rs)),rd),tr=M(0.5f,M(S(M(S(zt,M((float)k+0.5f,dz)),A(wn,ws)),M(S(zt,M((float)k-0.5f,dz)),A(pn,ps))),M(rz,M(S(rn,rs),rd))));cr[k]=M(A(gr,tr),M(0.25f,M(A(CELL(kmv,k,s,i,nj,ni),CELL(kmv,k,r,i,nj,ni)),A(M(G(gz,s,i,nj,ni),CELL(rf,k,s,i,nj,ni)),M(G(gz,r,i,nj,ni),CELL(rf,k,r,i,nj,ni))))));}
unsigned k=nk-1;rh=A(CELL(rho,k,s,i,nj,ni),CELL(rho,k,r,i,nj,ni));ri=D(1.0f,M(0.5f,rh));hm=A(CELL(mh,k,s,i,nj,ni),CELL(mh,k,r,i,nj,ni));float r1=A(CELL(kmv,k,s,i,nj,ni),CELL(kmv,k,r,i,nj,ni));r1=M(r1,M(A(CELL(mf,k,s,i,nj,ni),CELL(mf,k,r,i,nj,ni)),M(A(CELL(rf,k,s,i,nj,ni),CELL(rf,k,r,i,nj,ni)),M(hm,ri))));float cfa=M(tema,r1);cfb=S(1.0f,cfa);cfd=A(v[I3(k,f,i,nj+1,ni)],M(temc,M(t23[I3(nk,f,i,nj+1,ni)],M(hm,ri))));sol[k]=M(S(cfd,M(cfa,dp[k-1])),D(1.0f,A(M(cfa,cp[k-1]),cfb)));float rs=G(rgz,s,i,nj,ni),rn=G(rgz,r,i,nj,ni),wn=M(0.5f,A(CELL(w,k,r,i,nj,ni),CELL(w,k+1,r,i,nj,ni))),ws=M(0.5f,A(CELL(w,k,s,i,nj,ni),CELL(w,k+1,s,i,nj,ni))),pn=M(0.5f,A(CELL(w,k-1,r,i,nj,ni),CELL(w,k,r,i,nj,ni))),ps=M(0.5f,A(CELL(w,k-1,s,i,nj,ni),CELL(w,k,s,i,nj,ni)));float gr=M(S(M(CELL(w,k,r,i,nj,ni),rn),M(CELL(w,k,s,i,nj,ni),rs)),rd),tr=M(0.5f,M(S(M(S(zt,M((float)k+0.5f,dz)),A(wn,ws)),M(S(zt,M((float)k-0.5f,dz)),A(pn,ps))),M(rz,M(S(rn,rs),rd))));cr[k]=M(A(gr,tr),M(0.25f,M(A(CELL(kmv,k,s,i,nj,ni),CELL(kmv,k,r,i,nj,ni)),A(M(G(gz,s,i,nj,ni),CELL(rf,k,s,i,nj,ni)),M(G(gz,r,i,nj,ni),CELL(rf,k,r,i,nj,ni))))));for(int z=(int)nk-2;z>=0;z--)sol[z]=A(M(cp[z],sol[z+1]),dp[z]);for(k=0;k<nk;k++){rh=A(CELL(rho,k,s,i,nj,ni),CELL(rho,k,r,i,nj,ni));hm=A(CELL(mh,k,s,i,nj,ni),CELL(mh,k,r,i,nj,ni));float vz=M(rh,M(S(sol[k],v[I3(k,f,i,nj+1,ni)]),rdt));vz=A(vz,M(S(k+1<nk?cr[k+1]:0.0f,cr[k]),M(rz,M(0.5f,hm))));unsigned long long p=I3(k,f,i,nj+1,ni);vten[p]=A(vten[p],D(A(A(hx[k],hy[k]),vz),M(0.5f,rh)));}}
extern "C" __global__ void cm1_sgs_turbw_terrain_v1(const float*t13,const float*t23,const float*t33,const float*kmh,const float*rho,const float*rf,const float*mh,const float*mf,const float*gz,const float*rgzu,const float*rgzv,const float*w,float*wten,unsigned ni,unsigned nj,unsigned nk,float dx,float dz,float dt){unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nj*ni;if(col>=n)return;int i=(int)(col%ni),j=(int)(col/ni),nn=(int)nk-1;float hx[74],hy[74],cp[74],dp[74],sol[74];float rd=D(1.0f,dx),rz=D(1.0f,dz),zt=M((float)nk,dz),tema=M(-1.0f,M(dt,M(rz,rz))),temc=M(dt,rz),rdt=D(1.0f,dt);for(int p=0;p<nn;p++){int f=p+1;float d1=M(0.25f,A(A(t13[I3(f+1,j,i,nj,ni+1)],t13[I3(f+1,j,i+1,nj,ni+1)]),A(t13[I3(f,j,i,nj,ni+1)],t13[I3(f,j,i+1,nj,ni+1)]))),d0=M(0.25f,A(A(t13[I3(f,j,i,nj,ni+1)],t13[I3(f,j,i+1,nj,ni+1)]),A(t13[I3(f-1,j,i,nj,ni+1)],t13[I3(f-1,j,i+1,nj,ni+1)])));float x=M(gz[(unsigned long long)j*ni+i],M(S(M(t13[I3(f,j,i+1,nj,ni+1)],rgzu[(unsigned long long)j*(ni+1u)+i+1]),M(t13[I3(f,j,i,nj,ni+1)],rgzu[(unsigned long long)j*(ni+1u)+i])),rd));x=A(x,M(S(M(S(zt,M((float)f+0.5f,dz)),d1),M(S(zt,M((float)f-0.5f,dz)),d0)),M(gz[(unsigned long long)j*ni+i],M(S(rgzu[(unsigned long long)j*(ni+1u)+i+1],rgzu[(unsigned long long)j*(ni+1u)+i]),M(rd,rz)))));float e1=M(0.25f,A(A(t23[I3(f+1,j,i,nj+1,ni)],t23[I3(f+1,j+1,i,nj+1,ni)]),A(t23[I3(f,j,i,nj+1,ni)],t23[I3(f,j+1,i,nj+1,ni)]))),e0=M(0.25f,A(A(t23[I3(f,j,i,nj+1,ni)],t23[I3(f,j+1,i,nj+1,ni)]),A(t23[I3(f-1,j,i,nj+1,ni)],t23[I3(f-1,j+1,i,nj+1,ni)])));float y=M(gz[(unsigned long long)j*ni+i],M(S(M(t23[I3(f,j+1,i,nj+1,ni)],rgzv[(unsigned long long)(j+1)*ni+i]),M(t23[I3(f,j,i,nj+1,ni)],rgzv[(unsigned long long)j*ni+i])),rd));y=A(y,M(S(M(S(zt,M((float)f+0.5f,dz)),e1),M(S(zt,M((float)f-0.5f,dz)),e0)),M(gz[(unsigned long long)j*ni+i],M(S(rgzv[(unsigned long long)(j+1)*ni+i],rgzv[(unsigned long long)j*ni+i]),M(rd,rz)))));hx[p]=(i==0||i==(int)ni-1)?0.0f:x;hy[p]=(j==0||j==(int)nj-1)?0.0f:y;}
int f=1;float rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]),r2=M(A(kmh[I3(f,j,i,nj,ni)],kmh[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr))),cfc=M(tema,r2),cfb=S(1.0f,cfc),cfd=S(w[I3(f,j,i,nj,ni)],M(temc,M(t33[I3(0,j,i,nj,ni)],rr))),inv=D(1.0f,cfb);cp[0]=M(-1.0f,M(cfc,inv));dp[0]=M(cfd,inv);for(int p=1;p<nn-1;p++){f=p+1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmh[I3(f-1,j,i,nj,ni)],kmh[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr)));r2=M(A(kmh[I3(f,j,i,nj,ni)],kmh[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr)));float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1.0f,cfa),cfc);inv=D(1.0f,A(M(cfa,cp[p-1]),cfb));cp[p]=M(-1.0f,M(cfc,inv));dp[p]=M(S(w[I3(f,j,i,nj,ni)],M(cfa,dp[p-1])),inv);}int p=nn-1;f=nk-1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmh[I3(f-1,j,i,nj,ni)],kmh[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr))),cfa=M(tema,r1);cfb=S(1.0f,cfa);cfd=A(w[I3(f,j,i,nj,ni)],M(temc,M(t33[I3(nk-1,j,i,nj,ni)],rr)));sol[p]=M(S(cfd,M(cfa,dp[p-1])),D(1.0f,A(M(cfa,cp[p-1]),cfb)));for(p=nn-2;p>=0;p--)sol[p]=A(M(cp[p],sol[p+1]),dp[p]);for(p=0;p<nn;p++){f=p+1;float vz=M(rf[I3(f,j,i,nj,ni)],M(S(sol[p],w[I3(f,j,i,nj,ni)]),rdt));unsigned long long z=I3(f,j,i,nj,ni);wten[z]=A(wten[z],D(A(A(hx[p],hy[p]),vz),rf[z]));}}
'''


CUDA_SGS_STRESS_SOURCE += _column_cuda_source()
CUDA_SGS_STRESS_SOURCE_SHA256 = hashlib.sha256(CUDA_SGS_STRESS_SOURCE.encode()).hexdigest()


def _validate_column_capacity(max_nk: int) -> int:
    if isinstance(max_nk, bool) or not isinstance(max_nk, int) or not 3 <= max_nk <= (1 << 31) - 8:
        raise ValueError("max_nk must be an integer >= 3 within signed CUDA indexing")
    return max_nk


def _resize_sgs_column_workspace(source: str, *, max_nk: int) -> str:
    """Resize only the audited turbu/v/w local-array declarations."""

    max_nk = _validate_column_capacity(max_nk)
    replacements: dict[str, str] = {}
    for old, new, count in (
        (
            "float hx[75],hy[75],cp[75],dp[75],sol[75],cr[76];",
            f"float hx[{max_nk}],hy[{max_nk}],cp[{max_nk}],dp[{max_nk}],sol[{max_nk}],cr[{max_nk + 1}];",
            2,
        ),
        (
            "float hx[74],hy[74],cp[74],dp[74],sol[74];",
            f"float hx[{max_nk - 1}],hy[{max_nk - 1}],cp[{max_nk - 1}],dp[{max_nk - 1}],sol[{max_nk - 1}];",
            1,
        ),
    ):
        if source.count(old) != count:
            raise RuntimeError("SGS column workspace declaration inventory changed")
        replacements[old] = new
    return re.sub(
        "|".join(re.escape(old) for old in replacements),
        lambda match: replacements[match.group()], source,
    )


def build_cuda_sgs_stress_source(*, max_nk: int = CUDA_SGS_STRESS_MAX_NK) -> str:
    """Return identical arithmetic with an explicit local-column capacity."""

    if not _source_pin_is_valid():
        raise RuntimeError("base SGS stress source differs from its pin")
    return _resize_sgs_column_workspace(CUDA_SGS_STRESS_SOURCE, max_nk=max_nk)


@dataclass(frozen=True, slots=True)
class CompiledSgsStressKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: Mapping[str, object]
    max_nk: int = CUDA_SGS_STRESS_MAX_NK
    source_sha256: str = CUDA_SGS_STRESS_SOURCE_SHA256


def compile_cuda_sgs_stress_tendency(
    *, max_nk: int = CUDA_SGS_STRESS_MAX_NK
) -> CompiledSgsStressKernels:
    source = build_cuda_sgs_stress_source(max_nk=max_nk)
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    device = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(device.id)
    major = int(props["major"]); minor = int(props["minor"])
    # One architecture rule for the package: the clamped target every GM1 image
    # and every CuPy compile uses, not getDeviceProperties' unclamped pair.
    target = importlib.import_module("cm1core.nvrtc_module").nvrtc_target_arch(cupy)
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    nvrtc = tuple(compiler._get_nvrtc_version())
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(source, "cm1_sgs_stress_v1_ptx.cu", method="ptx").compile(common + (f"--gpu-architecture=compute_{target}",))
    cubin, _ = compiler._NVRTCProgram(source, "cm1_sgs_stress_v1_cubin.cu", method="cubin").compile(common + (f"--gpu-architecture=sm_{target}",))
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    # The image that runs is the cubin compiled above (``--ftz=false``), not a
    # RawModule, which CuPy compiles with ``-ftz=true`` appended.
    module = importlib.import_module("cupy.cuda.function").Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_SGS_STRESS_KERNEL_NAMES)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    return CompiledSgsStressKernels(module, kernels, ptx, cubin, target, {"device_id": device.id, "device_name": name, "compute_capability": f"{major}.{minor}", "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()), "driver_version": int(cupy.cuda.runtime.driverGetVersion())}, max_nk, source_sha256)


def _require_compiled_capacity(compiled: CompiledSgsStressKernels | None, nk: int) -> None:
    capacity = CUDA_SGS_STRESS_MAX_NK if compiled is None else compiled.max_nk
    _validate_column_capacity(capacity)
    if nk > capacity:
        raise ValueError(f"nk={nk} exceeds compiled SGS column workspace {capacity}; compile with max_nk >= nk")
    if compiled is not None and compiled.source_sha256 != hashlib.sha256(
        build_cuda_sgs_stress_source(max_nk=capacity).encode()
    ).hexdigest():
        raise ValueError("compiled SGS source hash differs from its column capacity")


def _capacity_receipt(compiled: CompiledSgsStressKernels) -> dict[str, object]:
    return {
        "column_workspace_max_nk": compiled.max_nk,
        "base_source_sha256": CUDA_SGS_STRESS_SOURCE_SHA256,
    } if compiled.max_nk != CUDA_SGS_STRESS_MAX_NK else {}


@dataclass(frozen=True, slots=True)
class CudaSgsStressLaunch:
    ready_event: Any
    receipt: Mapping[str, object]
    compiled: CompiledSgsStressKernels


def launch_cuda_sgs_stress_tendency(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int,
    dx: float, dy: float, dz: float, dt: float,
    stream: Any | None = None,
    compiled: CompiledSgsStressKernels | None = None,
    selection: SgsStressSelection = DEFAULT_SGS_STRESS_SELECTION,
    surface: Any | None = None,
) -> CudaSgsStressLaunch:
    _validate_selection(selection)
    _require_surface_binding(selection, surface)
    _validate_grid(ni, nj, nk)
    _require_compiled_capacity(compiled, nk)
    dx, dy, dz, dt = (_spacing(n, v) for n, v in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    _validate_square_spacing(dx, dy)
    cupy = importlib.import_module("cupy"); shapes = sgs_stress_shapes(ni, nj, nk)
    values: dict[str, Any] = {}
    for name in _INPUTS + _OUTPUTS:
        if name not in arrays: raise ValueError(f"missing SGS stress array: {name}")
        value = arrays[name]
        if not isinstance(value, cupy.ndarray): raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.float32 or value.shape != shapes[name] or not value.flags.c_contiguous: raise ValueError(f"{name} must be C-order float32 with shape {shapes[name]}")
        values[name] = value
    if len({int(v.data.ptr) for v in values.values()}) != len(values): raise ValueError("SGS stress arrays must not alias")
    compiled = compiled or compile_cuda_sgs_stress_tendency(); stream = stream or cupy.cuda.get_current_stream()
    u32, f32 = cupy.uint32, cupy.float32; block = (CUDA_SGS_STRESS_THREADS,); kernels = compiled.kernels
    totals = (ni*nj*nk, (ni+1)*(nj+1)*nk, (ni+1)*nj*(nk+1), ni*(nj+1)*(nk+1), (ni+1)*nj, ni*(nj+1), ni*nj)
    grids = tuple(((n + CUDA_SGS_STRESS_THREADS - 1)//CUDA_SGS_STRESS_THREADS,) for n in totals)
    with stream:
        kernels[0](grids[0], block, (values["s11"], values["s22"], values["s33"], values["kmh"], values["t11"], values["t22"], values["t33"], u32(ni), u32(nj), u32(nk)), stream=stream)
        kernels[1](grids[1], block, (values["s12"], values["kmh"], values["t12"], u32(ni), u32(nj), u32(nk)), stream=stream)
        kernels[2](grids[2], block, (values["s13"], values["kmv"], values["t13"], u32(ni), u32(nj), u32(nk)), stream=stream)
        kernels[3](grids[3], block, (values["s23"], values["kmv"], values["t23"], u32(ni), u32(nj), u32(nk)), stream=stream)
        # src/turb.F:4417.  gettau calls sfcstress between the interior
        # stress build and the momentum stress divergence.
        surface_kernels: tuple[str, ...] = ()
        if surface is not None:
            from cm1core.cuda_surface_drag import enqueue_surface_stress

            surface_kernels = enqueue_surface_stress(
                surface,
                {name: values[name] for name in ("t13", "t23", "rf", "mf", "kmv", "u3d", "v3d")},
                ni=ni, nj=nj, nk=nk, stream=stream,
            )
        kernels[4](grids[4], block, (values["t11"], values["t12"], values["t13"], values["kmv"], values["rho"], values["rf"], values["mh"], values["mf"], values["gz"], values["rgz"], values["gzu"], values["gxu"], values["u3d"], values["w3d"], values["uten"], u32(ni), u32(nj), u32(nk), f32(dx), f32(dz), f32(dt)), stream=stream)
        kernels[5](grids[5], block, (values["t12"], values["t22"], values["t23"], values["kmv"], values["rho"], values["rf"], values["mh"], values["mf"], values["gz"], values["rgz"], values["gzv"], values["gyv"], values["v3d"], values["w3d"], values["vten"], u32(ni), u32(nj), u32(nk), f32(dy), f32(dz), f32(dt)), stream=stream)
        kernels[6](grids[6], block, (values["t13"], values["t23"], values["t33"], values["kmh"], values["rho"], values["rf"], values["mh"], values["mf"], values["gz"], values["rgzu"], values["rgzv"], values["w3d"], values["wten"], u32(ni), u32(nj), u32(nk), f32(dx), f32(dz), f32(dt)), stream=stream)
        event = cupy.cuda.Event(); event.record(stream)
    return CudaSgsStressLaunch(event, {"schema": CUDA_SGS_STRESS_SCHEMA, "abi": CUDA_SGS_STRESS_ABI, "claim_limit": CUDA_SGS_STRESS_CLAIM_LIMIT, "authority": AUTHORITY_SOURCE_SHA256, "source_sha256": compiled.source_sha256, **_capacity_receipt(compiled), "selection": {"terrain": True, "axisymmetric": 0, "lateral_boundaries": [2,2,2,2], "doimpl": 1, "vialpha": 1, "vibeta": 0, "bbc": selection.bbc, "tbc": selection.tbc}, "domain": {"ni": ni, "nj": nj, "nk": nk}, "spacing_m": {"dx": dx, "dy": dy, "dz": dz}, "dt_seconds": dt, "lower_boundary": ({"bbc": 1, "disposition": "free-slip copy of k=2 (src/turb.F:4536-4545)"} if surface is None else {**surface.record(), "kernel_names": list(surface_kernels)}), "kernel": {"names": list(CUDA_SGS_STRESS_KERNEL_NAMES), "target_arch": compiled.target_arch}}, compiled)


def _source_pin_is_valid() -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", CUDA_SGS_STRESS_SOURCE_SHA256)) and hashlib.sha256(CUDA_SGS_STRESS_SOURCE.encode()).hexdigest() == CUDA_SGS_STRESS_SOURCE_SHA256
