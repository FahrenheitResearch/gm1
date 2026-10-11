"""GM1 solve1 scalar turbulent-diffusion tendencies for the real product.

The public name is GM1.  CM1 r21.1 is cited only as the mathematical
authority for this selected implementation.  ``solve1.F:1169-1227`` builds
the implicit coefficients once and calls ``turbs`` in this exact order:
theta, qv, qc, qr, qi, qs, qg.  ``turbtend.F:18-434,442-692`` supplies the
terrain horizontal flux divergence and the implicit vertical Thomas solve.

This module is deliberately sealed to the real 300 x 300 x 75 product
selection: Cartesian terrain, four physical open sides, uniform 100/100/200 m
spacing, SGS model 1, implicit scalar diffusion, zero surface flux, and the
CM1-setup-1 zero-flux vertical boundary.  Compact physical fields are padded
on device with the source open-boundary edge-copy rule before the operator is
applied.  Production launch performs no scientific host copy or synchronize.

``idiss`` is not fused here.  The authority places it in a separate solve1
held-tendency block; a coupled scheduler must compose that independently and
must not count it again in this operator.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_GM1_SCALAR_TURBULENCE_SCHEMA = "gm1.cuda-scalar-turbulence/v1"
CUDA_GM1_SCALAR_TURBULENCE_ABI = "gm1.cuda-scalar-turbulence-kernels/v1"
CUDA_GM1_SCALAR_TURBULENCE_THREADS = 256
GM1_SCALAR_TURBULENCE_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
GM1_SCALAR_TURBULENCE_FIELD_ORDER = ("theta",) + GM1_SCALAR_TURBULENCE_SPECIES
CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES = (
    "gm1_turbs_pad_static_open_v1",
    "gm1_turbs_pack_scalar_open_v1",
    "gm1_turbs_reconstruct_w_v1",
    "gm1_turbs_xy_flux_v1",
    "gm1_turbs_interpolate_flux_w_v1",
    "gm1_turbs_implicit_apply_v1",
)
CUDA_GM1_SCALAR_TURBULENCE_CLAIM_LIMIT = (
    "GM1 device-resident solve1 held scalar turbulent-diffusion contribution "
    "for the source-pinned real terrain/all-open/SGS=1/doimpl=1/no-surface-flux "
    "selection: theta then qv,qc,qr,qi,qs,qg; compact physical-domain open "
    "halos, uniform-grid coefficients, terrain horizontal diffusion, implicit "
    "vertical diffusion, and in-place addition to caller-owned thten1/qten. "
    "No momentum stress, TKE evolution, idiss, surface flux, nonuniform grid, "
    "MPI subdomain, RK integration, microphysics, trajectory, forecast, tornado, "
    "or general model-validation claim"
)

SOURCE_ANCHORS = {
    "solve1_coefficients_and_call_order": "src/solve1.F:1169-1227",
    "turbs_terrain_horizontal": "src/turbtend.F:157-308",
    "turbs_final_composition": "src/turbtend.F:349-382",
    "turbsz_implicit": "src/turbtend.F:442-692",
    "open_scalar_halos": "src/bc.F:180-364",
    "open_pressure_halos": "src/bc.F:372-520",
    "theta_reference": "src/constants.F:69-140",
    "idiss_separate_seam": "src/solve1.F:744-848,917-975",
    "authority_namelist": "run/config_files/supercell/namelist.input",
}
AUTHORITY_SOURCE_SHA256 = {
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "src/solve1.F": "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076",
    "src/turbtend.F": "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de",
    "run/config_files/supercell/namelist.input": "3854f731efe6a6a9d56d6aa3db198434ccaba85bc5584b5e8ce1c5edfd3b56a4",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)


@dataclass(frozen=True, slots=True)
class Gm1ScalarTurbulenceSelection:
    """Only the source selection admitted by this implementation."""

    terrain: bool = True
    axisymmetric: int = 0
    west_boundary: int = 2
    east_boundary: int = 2
    south_boundary: int = 2
    north_boundary: int = 2
    sgs_model: int = 1
    cm1_setup: int = 1
    horizontal_turbulence: bool = True
    vertical_turbulence: bool = True
    implicit_vertical: int = 1
    scalar_vertical_bc: int = 1
    # src/turbtend.F:517-521,587-596.  ``iflux=1 .and. dosfcflx`` replaces the
    # zero lower flux with ``sflux*rf`` for theta and qv; every other scalar
    # keeps the zero.  The selection carries it because the same compiled
    # kernel runs both ways and only the launch argument differs.
    surface_flux: bool = False
    uniform_grid: bool = True
    moisture: int = 1
    moisture_species: tuple[str, ...] = GM1_SCALAR_TURBULENCE_SPECIES


DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION = Gm1ScalarTurbulenceSelection()


def _validate_selection(selection: Gm1ScalarTurbulenceSelection) -> None:
    if not isinstance(selection, Gm1ScalarTurbulenceSelection):
        raise TypeError("selection must be a Gm1ScalarTurbulenceSelection")
    expected = DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION
    for name in selection.__dataclass_fields__:
        value = getattr(selection, name)
        wanted = getattr(expected, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="GM1-SCALAR-TURBULENCE-SELECTION",
                gate="boundaries-closures",
                option=name,
                value=value,
                reason="the scalar-turbulence bridge is sealed to the corrected-real product selection",
                remedy=f"use {name}={wanted!r} or add a separately source-pinned implementation",
            )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 2), ("nj", nj, 2), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * (nk + 1) > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("GM1 scalar-turbulence word count exceeds signed 64-bit indexing")


def _spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def gm1_scalar_turbulence_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    return {
        "th0": (nk,),
        "tha": mass,
        "qa": (6, nk, nj, ni),
        "rho": mass,
        "rr": mass,
        "rf": zface,
        "mh": mass,
        "mf": zface,
        "khh": zface,
        "khv": zface,
        "gz": (nj, ni),
        "rgz": (nj, ni),
        "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni),
        "gx": zface,
        "gxu": (nk + 1, nj, ni + 1),
        "gy": zface,
        "gyv": (nk + 1, nj + 1, ni),
        "thten1": mass,
        "qten": (6, nk, nj, ni),
    }


INPUT_NAMES = (
    "th0", "tha", "qa", "rho", "rr", "rf", "mh", "mf", "khh", "khv",
    "gz", "rgz", "rgzu", "rgzv", "gx", "gxu", "gy", "gyv",
)
OUTPUT_NAMES = ("thten1", "qten")
ALL_ARRAY_NAMES = INPUT_NAMES + OUTPUT_NAMES


def _require_host_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = gm1_scalar_turbulence_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing GM1 scalar-turbulence arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order float32 with shape {shapes[name]}")
        if name in OUTPUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("GM1 scalar-turbulence arrays must not alias")
    return result


def _a(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _s(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _m(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _d(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _n(np: Any, value: Any) -> Any:
    return np.float32(-np.float32(value))


def _pad_xy_edge(np: Any, value: Any) -> Any:
    return np.ascontiguousarray(np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="edge"))


def _at_w(np: Any, scalar_halo: Any, face: int, j: int, i: int, nk: int) -> Any:
    if face == 0:
        return _a(
            np,
            _a(np, _m(np, 1.875, scalar_halo[0, j, i]), _m(np, -1.25, scalar_halo[1, j, i])),
            _m(np, 0.375, scalar_halo[2, j, i]),
        )
    if face == nk:
        return _a(
            np,
            _a(np, _m(np, 1.875, scalar_halo[nk - 1, j, i]), _m(np, -1.25, scalar_halo[nk - 2, j, i])),
            _m(np, 0.375, scalar_halo[nk - 3, j, i]),
        )
    return _a(
        np,
        _m(np, 0.5, scalar_halo[face - 1, j, i]),
        _m(np, 0.5, scalar_halo[face, j, i]),
    )


def _interpolate_flux(np: Any, value: Any, face: int, nk: int) -> Any:
    if face == 0:
        return _a(np, _a(np, _m(np, 1.875, value[0]), _m(np, -1.25, value[1])), _m(np, 0.375, value[2]))
    if face == nk:
        return _a(np, _a(np, _m(np, 1.875, value[nk - 1]), _m(np, -1.25, value[nk - 2])), _m(np, 0.375, value[nk - 3]))
    return _a(np, _m(np, 0.5, value[face - 1]), _m(np, 0.5, value[face]))


def _cpu_one_scalar(
    np: Any,
    a: Mapping[str, Any],
    scalar: Any,
    target: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
) -> None:
    scalar_halo = _pad_xy_edge(np, scalar)
    rho_halo = _pad_xy_edge(np, a["rho"])
    khh_halo = _pad_xy_edge(np, a["khh"])
    gz_halo = np.pad(a["gz"], ((1, 1), (1, 1)), mode="edge")
    rgz_halo = np.pad(a["rgz"], ((1, 1), (1, 1)), mode="edge")
    scalar_w = np.empty((nk + 1, nj + 2, ni + 2), dtype=np.float32)
    for face in range(nk + 1):
        for j in range(nj + 2):
            for i in range(ni + 2):
                scalar_w[face, j, i] = _at_w(np, scalar_halo, face, j, i, nk)

    rdx = _d(np, 1.0, dx)
    rdy = _d(np, 1.0, dy)
    rdz = _d(np, 1.0, dz)
    flux_x = np.empty((nk, nj, ni + 1), dtype=np.float32)
    flux_y = np.empty((nk, nj + 1, ni), dtype=np.float32)
    for k in range(nk):
        for j in range(nj):
            jj = j + 1
            for face in range(ni + 1):
                il, ir = face, face + 1
                density = _a(np, _m(np, gz_halo[jj, ir], rho_halo[k, jj, ir]), _m(np, gz_halo[jj, il], rho_halo[k, jj, il]))
                diffusivity = _a(
                    np,
                    _a(np, khh_halo[k, jj, ir], khh_halo[k, jj, il]),
                    _a(np, khh_halo[k + 1, jj, ir], khh_halo[k + 1, jj, il]),
                )
                horizontal = _m(
                    np,
                    _s(np, _m(np, scalar_halo[k, jj, ir], rgz_halo[jj, ir]), _m(np, scalar_halo[k, jj, il], rgz_halo[jj, il])),
                    rdx,
                )
                upper = _m(np, a["gxu"][k + 1, j, face], _a(np, scalar_w[k + 1, jj, ir], scalar_w[k + 1, jj, il]))
                lower = _m(np, a["gxu"][k, j, face], _a(np, scalar_w[k, jj, ir], scalar_w[k, jj, il]))
                terrain = _m(np, _m(np, _m(np, 0.5, _s(np, upper, lower)), rdz), a["rgzu"][j, face])
                flux_x[k, j, face] = _m(np, _m(np, _m(np, -0.125, density), diffusivity), _a(np, horizontal, terrain))
        for face in range(nj + 1):
            js, jn = face, face + 1
            for i in range(ni):
                ii = i + 1
                density = _a(np, _m(np, gz_halo[jn, ii], rho_halo[k, jn, ii]), _m(np, gz_halo[js, ii], rho_halo[k, js, ii]))
                diffusivity = _a(
                    np,
                    _a(np, khh_halo[k, jn, ii], khh_halo[k, js, ii]),
                    _a(np, khh_halo[k + 1, jn, ii], khh_halo[k + 1, js, ii]),
                )
                horizontal = _m(
                    np,
                    _s(np, _m(np, scalar_halo[k, jn, ii], rgz_halo[jn, ii]), _m(np, scalar_halo[k, js, ii], rgz_halo[js, ii])),
                    rdy,
                )
                upper = _m(np, a["gyv"][k + 1, face, i], _a(np, scalar_w[k + 1, jn, ii], scalar_w[k + 1, js, ii]))
                lower = _m(np, a["gyv"][k, face, i], _a(np, scalar_w[k, jn, ii], scalar_w[k, js, ii]))
                terrain = _m(np, _m(np, _m(np, 0.5, _s(np, upper, lower)), rdz), a["rgzv"][face, i])
                flux_y[k, face, i] = _m(np, _m(np, _m(np, -0.125, density), diffusivity), _a(np, horizontal, terrain))

    flux_x_w = np.empty((nk + 1, nj, ni + 1), dtype=np.float32)
    flux_y_w = np.empty((nk + 1, nj + 1, ni), dtype=np.float32)
    for face in range(nk + 1):
        for j in range(nj):
            for i in range(ni + 1):
                flux_x_w[face, j, i] = _interpolate_flux(np, flux_x[:, j, i], face, nk)
        for j in range(nj + 1):
            for i in range(ni):
                flux_y_w[face, j, i] = _interpolate_flux(np, flux_y[:, j, i], face, nk)

    rdt = _d(np, 1.0, dt)
    tema = _m(np, _m(np, _m(np, _m(np, -1.0, dt), 1.0), rdz), rdz)
    temb = _m(np, _m(np, _m(np, dt, 0.0), rdz), rdz)
    temc = _m(np, dt, rdz)
    cp = np.empty((nk,), dtype=np.float32)
    dp = np.empty((nk,), dtype=np.float32)

    def coefficient(face: int, k: int, j: int, i: int) -> Any:
        value = _m(np, a["khv"][face, j, i], a["mf"][face, j, i])
        value = _m(np, value, a["rf"][face, j, i])
        value = _m(np, value, a["mh"][k, j, i])
        return _m(np, value, a["rr"][k, j, i])

    for j in range(nj):
        for i in range(ni):
            r2 = coefficient(1, 0, j, i)
            cfc = _m(np, tema, r2)
            cfb = _s(np, 1.0, cfc)
            explicit = _a(np, _m(np, _n(np, r2), scalar[0, j, i]), _m(np, r2, scalar[1, j, i]))
            boundary = _m(np, _m(np, _m(np, temc, 0.0), a["mh"][0, j, i]), a["rr"][0, j, i])
            cfd = _a(np, _a(np, scalar[0, j, i], _m(np, temb, explicit)), boundary)
            tem = _d(np, 1.0, cfb)
            cp[0] = _m(np, _n(np, cfc), tem)
            dp[0] = _m(np, cfd, tem)
            for k in range(1, nk - 1):
                r1 = coefficient(k, k, j, i)
                r2 = coefficient(k + 1, k, j, i)
                cfa = _m(np, tema, r1)
                cfc = _m(np, tema, r2)
                cfb = _s(np, _s(np, 1.0, cfa), cfc)
                explicit = _a(
                    np,
                    _s(np, _m(np, r1, scalar[k - 1, j, i]), _m(np, _a(np, r1, r2), scalar[k, j, i])),
                    _m(np, r2, scalar[k + 1, j, i]),
                )
                cfd = _a(np, scalar[k, j, i], _m(np, temb, explicit))
                tem = _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb))
                cp[k] = _m(np, _n(np, cfc), tem)
                dp[k] = _m(np, _s(np, cfd, _m(np, cfa, dp[k - 1])), tem)
            k = nk - 1
            r1 = coefficient(k, k, j, i)
            cfa = _m(np, tema, r1)
            cfb = _s(np, 1.0, cfa)
            explicit = _s(np, _m(np, r1, scalar[k - 1, j, i]), _m(np, r1, scalar[k, j, i]))
            boundary = _m(np, _m(np, _m(np, temc, 0.0), a["mh"][k, j, i]), a["rr"][k, j, i])
            cfd = _s(np, _a(np, scalar[k, j, i], _m(np, temb, explicit)), boundary)
            tem = _d(np, 1.0, _a(np, _m(np, cfa, cp[k - 1]), cfb))
            solution = _m(np, _s(np, cfd, _m(np, cfa, dp[k - 1])), tem)

            for kk in range(nk - 1, -1, -1):
                if kk != nk - 1:
                    solution = _a(np, _m(np, cp[kk], solution), dp[kk])
                turbz = _m(np, _m(np, a["rho"][kk, j, i], _s(np, solution, scalar[kk, j, i])), rdt)
                right = _m(np, flux_x[kk, j, i + 1], a["rgzu"][j, i + 1])
                left = _m(np, flux_x[kk, j, i], a["rgzu"][j, i])
                tx0 = _m(np, _m(np, _n(np, a["gz"][j, i]), _s(np, right, left)), rdx)
                upper = _m(np, a["gx"][kk + 1, j, i], _a(np, flux_x_w[kk + 1, j, i], flux_x_w[kk + 1, j, i + 1]))
                lower = _m(np, a["gx"][kk, j, i], _a(np, flux_x_w[kk, j, i], flux_x_w[kk, j, i + 1]))
                tx = _a(np, tx0, _m(np, _m(np, _n(np, _s(np, upper, lower)), 0.5), rdz))
                if i == 0 or i == ni - 1:
                    tx = np.float32(0.0)
                north = _m(np, flux_y[kk, j + 1, i], a["rgzv"][j + 1, i])
                south = _m(np, flux_y[kk, j, i], a["rgzv"][j, i])
                ty0 = _m(np, _m(np, _n(np, a["gz"][j, i]), _s(np, north, south)), rdy)
                upper = _m(np, a["gy"][kk + 1, j, i], _a(np, flux_y_w[kk + 1, j, i], flux_y_w[kk + 1, j + 1, i]))
                lower = _m(np, a["gy"][kk, j, i], _a(np, flux_y_w[kk, j, i], flux_y_w[kk, j + 1, i]))
                ty = _a(np, ty0, _m(np, _m(np, _n(np, _s(np, upper, lower)), 0.5), rdz))
                if j == 0 or j == nj - 1:
                    ty = np.float32(0.0)
                target[kk, j, i] = _a(np, target[kk, j, i], _m(np, _a(np, _a(np, tx, ty), turbz), a["rr"][kk, j, i]))


def cpu_gm1_scalar_turbulence_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1ScalarTurbulenceSelection = DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION,
) -> None:
    """Independent binary32 source-order CPU validation transcription."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz, dt = (_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    theta = np.empty_like(a["tha"])
    for k in range(nk):
        base = _s(np, a["th0"][k], 300.0)
        for j in range(nj):
            for i in range(ni):
                theta[k, j, i] = _a(np, base, a["tha"][k, j, i])
    _cpu_one_scalar(np, a, theta, a["thten1"], ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, dt=dt)
    for species in range(6):
        _cpu_one_scalar(np, a, a["qa"][species], a["qten"][species], ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, dt=dt)


CUDA_GM1_SCALAR_TURBULENCE_SOURCE = r'''// gm1.cuda-scalar-turbulence-kernels/v1
static_assert(sizeof(float)==4,"GM1 authority default REAL requires float32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);}
__device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);}
__device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);}
__device__ __forceinline__ float D(float x,float y){return __fdiv_rn(x,y);}
__device__ __forceinline__ unsigned long long I3(unsigned k,unsigned j,unsigned i,unsigned ny,unsigned nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ int C(int x,int n){return x<0?0:(x>=n?n-1:x);}

extern "C" __global__ void gm1_turbs_pad_static_open_v1(
 const float*rho,const float*khh,const float*gz,const float*rgz,
 float*rhoh,float*khhh,float*gzh,float*rgzh,unsigned ni,unsigned nj,unsigned nk){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,hx=ni+2u,hy=nj+2u,hxy=hx*hy;
 for(unsigned long long z=q;z<(unsigned long long)nk*hxy;z+=step){unsigned i=(unsigned)(z%hx),j=(unsigned)((z/hx)%hy),k=(unsigned)(z/hxy);int ii=C((int)i-1,(int)ni),jj=C((int)j-1,(int)nj);rhoh[z]=rho[I3(k,(unsigned)jj,(unsigned)ii,nj,ni)];}
 for(unsigned long long z=q;z<(unsigned long long)(nk+1u)*hxy;z+=step){unsigned i=(unsigned)(z%hx),j=(unsigned)((z/hx)%hy),k=(unsigned)(z/hxy);int ii=C((int)i-1,(int)ni),jj=C((int)j-1,(int)nj);khhh[z]=khh[I3(k,(unsigned)jj,(unsigned)ii,nj,ni)];}
 for(unsigned long long z=q;z<hxy;z+=step){unsigned i=(unsigned)(z%hx),j=(unsigned)(z/hx);int ii=C((int)i-1,(int)ni),jj=C((int)j-1,(int)nj);unsigned long long p=(unsigned long long)jj*ni+(unsigned)ii;gzh[z]=gz[p];rgzh[z]=rgz[p];}
}

extern "C" __global__ void gm1_turbs_pack_scalar_open_v1(
 const float*th0,const float*tha,const float*qa,float*sh,unsigned ni,unsigned nj,unsigned nk,unsigned species){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,hx=ni+2u,hy=nj+2u,hxy=hx*hy,n=(unsigned long long)ni*nj*nk;
 for(unsigned long long z=q;z<(unsigned long long)nk*hxy;z+=step){unsigned i=(unsigned)(z%hx),j=(unsigned)((z/hx)%hy),k=(unsigned)(z/hxy);int ii=C((int)i-1,(int)ni),jj=C((int)j-1,(int)nj);unsigned long long p=I3(k,(unsigned)jj,(unsigned)ii,nj,ni);sh[z]=(species==0xffffffffu)?A(S(th0[k],300.0f),tha[p]):qa[(unsigned long long)species*n+p];}
}

extern "C" __global__ void gm1_turbs_reconstruct_w_v1(const float*sh,float*sw,unsigned ni,unsigned nj,unsigned nk){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,hx=ni+2u,hy=nj+2u,hxy=hx*hy;
 for(unsigned long long z=q;z<(unsigned long long)(nk+1u)*hxy;z+=step){unsigned f=(unsigned)(z/hxy),p=(unsigned)(z%hxy);if(f==0u)sw[z]=A(A(M(1.875f,sh[p]),M(-1.25f,sh[hxy+p])),M(0.375f,sh[2ull*hxy+p]));else if(f==nk)sw[z]=A(A(M(1.875f,sh[(unsigned long long)(nk-1u)*hxy+p]),M(-1.25f,sh[(unsigned long long)(nk-2u)*hxy+p])),M(0.375f,sh[(unsigned long long)(nk-3u)*hxy+p]));else sw[z]=A(M(0.5f,sh[(unsigned long long)(f-1u)*hxy+p]),M(0.5f,sh[(unsigned long long)f*hxy+p]));}
}

extern "C" __global__ void gm1_turbs_xy_flux_v1(
 const float*sh,const float*sw,const float*rhoh,const float*khhh,const float*gzh,const float*rgzh,
 const float*rgzu,const float*rgzv,const float*gxu,const float*gyv,float*fx,float*fy,
 unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,hx=ni+2u,hy=nj+2u,hxy=hx*hy,nx=(unsigned long long)nk*nj*(ni+1u),ny=(unsigned long long)nk*(nj+1u)*ni;float rx=D(1.0f,dx),ry=D(1.0f,dy),rz=D(1.0f,dz);
 for(unsigned long long z=q;z<nx;z+=step){unsigned f=(unsigned)(z%(ni+1u)),j=(unsigned)((z/(ni+1u))%nj),k=(unsigned)(z/((unsigned long long)(ni+1u)*nj)),il=f,ir=f+1u,jj=j+1u;unsigned long long l=I3(k,jj,il,hy,hx),r=I3(k,jj,ir,hy,hx),lu=I3(k+1u,jj,il,hy,hx),ru=I3(k+1u,jj,ir,hy,hx);float den=A(M(gzh[(unsigned long long)jj*hx+ir],rhoh[r]),M(gzh[(unsigned long long)jj*hx+il],rhoh[l])),kh=A(A(khhh[r],khhh[l]),A(khhh[ru],khhh[lu])),hor=M(S(M(sh[r],rgzh[(unsigned long long)jj*hx+ir]),M(sh[l],rgzh[(unsigned long long)jj*hx+il])),rx),up=M(gxu[I3(k+1u,j,f,nj,ni+1u)],A(sw[ru],sw[lu])),lo=M(gxu[I3(k,j,f,nj,ni+1u)],A(sw[lu-hxy],sw[ru-hxy])),ter=M(M(M(0.5f,S(up,lo)),rz),rgzu[(unsigned long long)j*(ni+1u)+f]);fx[z]=M(M(M(-0.125f,den),kh),A(hor,ter));}
 for(unsigned long long z=q;z<ny;z+=step){unsigned i=(unsigned)(z%ni),f=(unsigned)((z/ni)%(nj+1u)),k=(unsigned)(z/((unsigned long long)ni*(nj+1u))),js=f,jn=f+1u,ii=i+1u;unsigned long long s=I3(k,js,ii,hy,hx),n=I3(k,jn,ii,hy,hx),su=I3(k+1u,js,ii,hy,hx),nu=I3(k+1u,jn,ii,hy,hx);float den=A(M(gzh[(unsigned long long)jn*hx+ii],rhoh[n]),M(gzh[(unsigned long long)js*hx+ii],rhoh[s])),kh=A(A(khhh[n],khhh[s]),A(khhh[nu],khhh[su])),hor=M(S(M(sh[n],rgzh[(unsigned long long)jn*hx+ii]),M(sh[s],rgzh[(unsigned long long)js*hx+ii])),ry),up=M(gyv[I3(k+1u,f,i,nj+1u,ni)],A(sw[nu],sw[su])),lo=M(gyv[I3(k,f,i,nj+1u,ni)],A(sw[su-hxy],sw[nu-hxy])),ter=M(M(M(0.5f,S(up,lo)),rz),rgzv[(unsigned long long)f*ni+i]);fy[z]=M(M(M(-0.125f,den),kh),A(hor,ter));}
}

extern "C" __global__ void gm1_turbs_interpolate_flux_w_v1(const float*fx,const float*fy,float*fxw,float*fyw,unsigned ni,unsigned nj,unsigned nk){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nx=(unsigned long long)nj*(ni+1u),ny=(unsigned long long)(nj+1u)*ni;
 for(unsigned long long z=q;z<(unsigned long long)(nk+1u)*nx;z+=step){unsigned f=(unsigned)(z/nx),p=(unsigned)(z%nx);if(f==0u)fxw[z]=A(A(M(1.875f,fx[p]),M(-1.25f,fx[nx+p])),M(0.375f,fx[2ull*nx+p]));else if(f==nk)fxw[z]=A(A(M(1.875f,fx[(unsigned long long)(nk-1u)*nx+p]),M(-1.25f,fx[(unsigned long long)(nk-2u)*nx+p])),M(0.375f,fx[(unsigned long long)(nk-3u)*nx+p]));else fxw[z]=A(M(0.5f,fx[(unsigned long long)(f-1u)*nx+p]),M(0.5f,fx[(unsigned long long)f*nx+p]));}
 for(unsigned long long z=q;z<(unsigned long long)(nk+1u)*ny;z+=step){unsigned f=(unsigned)(z/ny),p=(unsigned)(z%ny);if(f==0u)fyw[z]=A(A(M(1.875f,fy[p]),M(-1.25f,fy[ny+p])),M(0.375f,fy[2ull*ny+p]));else if(f==nk)fyw[z]=A(A(M(1.875f,fy[(unsigned long long)(nk-1u)*ny+p]),M(-1.25f,fy[(unsigned long long)(nk-2u)*ny+p])),M(0.375f,fy[(unsigned long long)(nk-3u)*ny+p]));else fyw[z]=A(M(0.5f,fy[(unsigned long long)(f-1u)*ny+p]),M(0.5f,fy[(unsigned long long)f*ny+p]));}
}

extern "C" __global__ void gm1_turbs_implicit_apply_v1(
 const float*sh,const float*rho,const float*rr,const float*rf,const float*mh,const float*mf,const float*khv,
 const float*gz,const float*rgzu,const float*rgzv,const float*gx,const float*gy,
 const float*fx,const float*fy,const float*fxw,const float*fyw,const float*sflux,float*cp,float*dp,float*target,
 unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz,float dt,unsigned iflux){
 unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,ncol=(unsigned long long)ni*nj;if(col>=ncol)return;unsigned i=(unsigned)(col%ni),j=(unsigned)(col/ni),hx=ni+2u,hy=nj+2u;float rx=D(1.0f,dx),ry=D(1.0f,dy),rz=D(1.0f,dz),rdt=D(1.0f,dt),tema=M(M(M(M(-1.0f,dt),1.0f),rz),rz),temb=M(M(M(dt,0.0f),rz),rz),temc=M(dt,rz);unsigned long long nxy=ncol;
 #define SC(K) sh[I3((K),j+1u,i+1u,hy,hx)]
 #define COEF(F,K) M(M(M(M(khv[I3((F),j,i,nj,ni)],mf[I3((F),j,i,nj,ni)]),rf[I3((F),j,i,nj,ni)]),mh[I3((K),j,i,nj,ni)]),rr[I3((K),j,i,nj,ni)])
 float r2=COEF(1u,0u),cfc=M(tema,r2),cfb=S(1.0f,cfc),ex=A(M(-r2,SC(0u)),M(r2,SC(1u))),bc=M(M(M(temc,(iflux==1u)?M(sflux[col],rf[col]):0.0f),mh[col]),rr[col]),cfd=A(A(SC(0u),M(temb,ex)),bc),tem=D(1.0f,cfb);cp[col]=M(-cfc,tem);dp[col]=M(cfd,tem);
 for(unsigned k=1u;k+1u<nk;++k){float r1=COEF(k,k);r2=COEF(k+1u,k);float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1.0f,cfa),cfc);ex=A(S(M(r1,SC(k-1u)),M(A(r1,r2),SC(k))),M(r2,SC(k+1u)));cfd=A(SC(k),M(temb,ex));unsigned long long p=(unsigned long long)k*nxy+col,pm=p-nxy;tem=D(1.0f,A(M(cfa,cp[pm]),cfb));cp[p]=M(-cfc,tem);dp[p]=M(S(cfd,M(cfa,dp[pm])),tem);}
 unsigned k=nk-1u;float r1=COEF(k,k),cfa=M(tema,r1);cfb=S(1.0f,cfa);ex=S(M(r1,SC(k-1u)),M(r1,SC(k)));bc=M(M(M(temc,0.0f),mh[(unsigned long long)k*nxy+col]),rr[(unsigned long long)k*nxy+col]);cfd=S(A(SC(k),M(temb,ex)),bc);unsigned long long p=(unsigned long long)k*nxy+col,pm=p-nxy;tem=D(1.0f,A(M(cfa,cp[pm]),cfb));float sol=M(S(cfd,M(cfa,dp[pm])),tem);
 for(int kk=(int)nk-1;kk>=0;--kk){unsigned ku=(unsigned)kk;unsigned long long mass=(unsigned long long)ku*nxy+col;if(ku!=nk-1u)sol=A(M(cp[mass],sol),dp[mass]);float tz=M(M(rho[mass],S(sol,SC(ku))),rdt);unsigned long long xp=((unsigned long long)ku*nj+j)*(ni+1u)+i,yp=((unsigned long long)ku*(nj+1u)+j)*ni+i;float right=M(fx[xp+1u],rgzu[(unsigned long long)j*(ni+1u)+i+1u]),left=M(fx[xp],rgzu[(unsigned long long)j*(ni+1u)+i]),tx=A(M(M(-gz[col],S(right,left)),rx),M(M(-S(M(gx[I3(ku+1u,j,i,nj,ni)],A(fxw[((unsigned long long)(ku+1u)*nj+j)*(ni+1u)+i],fxw[((unsigned long long)(ku+1u)*nj+j)*(ni+1u)+i+1u])),M(gx[I3(ku,j,i,nj,ni)],A(fxw[((unsigned long long)ku*nj+j)*(ni+1u)+i],fxw[((unsigned long long)ku*nj+j)*(ni+1u)+i+1u]))),0.5f),rz));if(i==0u||i+1u==ni)tx=0.0f;float north=M(fy[yp+ni],rgzv[(unsigned long long)(j+1u)*ni+i]),south=M(fy[yp],rgzv[(unsigned long long)j*ni+i]),ty=A(M(M(-gz[col],S(north,south)),ry),M(M(-S(M(gy[I3(ku+1u,j,i,nj,ni)],A(fyw[((unsigned long long)(ku+1u)*(nj+1u)+j)*ni+i],fyw[((unsigned long long)(ku+1u)*(nj+1u)+j+1u)*ni+i])),M(gy[I3(ku,j,i,nj,ni)],A(fyw[((unsigned long long)ku*(nj+1u)+j)*ni+i],fyw[((unsigned long long)ku*(nj+1u)+j+1u)*ni+i]))),0.5f),rz));if(j==0u||j+1u==nj)ty=0.0f;target[mass]=A(target[mass],M(A(A(tx,ty),tz),rr[mass]));}
 #undef SC
 #undef COEF
}
'''

CUDA_GM1_SCALAR_TURBULENCE_SOURCE_BYTES = CUDA_GM1_SCALAR_TURBULENCE_SOURCE.encode("utf-8")
# Updated only after the reviewed source string above changes.
CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256 = (
    "d8f138c63a0858e8ea4e1e5ee51906a55b46344f005d6b6296d24dcbb7ff4b62"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_GM1_SCALAR_TURBULENCE_SOURCE_BYTES).hexdigest() == CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledGm1ScalarTurbulenceKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_scalar_turbulence() -> CompiledGm1ScalarTurbulenceKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged GM1 scalar-turbulence CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(CUDA_GM1_SCALAR_TURBULENCE_SOURCE, "gm1_scalar_turbs_v1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_GM1_SCALAR_TURBULENCE_SOURCE, "gm1_scalar_turbs_v1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks GM1 scalar-turbulence entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled GM1 scalar-turbulence image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ScalarTurbulenceKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = gm1_scalar_turbulence_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA GM1 scalar-turbulence arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order CUDA float32 with shape {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA GM1 scalar-turbulence arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Gm1ScalarTurbulenceWorkspace:
    ni: int
    nj: int
    nk: int
    rho_halo: Any
    khh_halo: Any
    gz_halo: Any
    rgz_halo: Any
    scalar_halo: Any
    scalar_w: Any
    flux_x: Any
    flux_y: Any
    flux_x_w: Any
    flux_y_w: Any
    thomas_c: Any
    thomas_d: Any


def create_cuda_gm1_scalar_turbulence_workspace(*, ni: int, nj: int, nk: int) -> Gm1ScalarTurbulenceWorkspace:
    _validate_grid(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    return Gm1ScalarTurbulenceWorkspace(
        ni, nj, nk,
        cupy.empty((nk, nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nk + 1, nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nk, nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nk + 1, nj + 2, ni + 2), dtype=cupy.float32),
        cupy.empty((nk, nj, ni + 1), dtype=cupy.float32),
        cupy.empty((nk, nj + 1, ni), dtype=cupy.float32),
        cupy.empty((nk + 1, nj, ni + 1), dtype=cupy.float32),
        cupy.empty((nk + 1, nj + 1, ni), dtype=cupy.float32),
        cupy.empty((nk, nj, ni), dtype=cupy.float32),
        cupy.empty((nk, nj, ni), dtype=cupy.float32),
    )


def _validate_workspace(cupy: Any, workspace: Gm1ScalarTurbulenceWorkspace, *, ni: int, nj: int, nk: int, input_pointers: set[int]) -> None:
    if not isinstance(workspace, Gm1ScalarTurbulenceWorkspace) or (workspace.ni, workspace.nj, workspace.nk) != (ni, nj, nk):
        raise ValueError("GM1 scalar-turbulence workspace geometry differs")
    expected = {
        "rho_halo": (nk, nj + 2, ni + 2), "khh_halo": (nk + 1, nj + 2, ni + 2),
        "gz_halo": (nj + 2, ni + 2), "rgz_halo": (nj + 2, ni + 2),
        "scalar_halo": (nk, nj + 2, ni + 2), "scalar_w": (nk + 1, nj + 2, ni + 2),
        "flux_x": (nk, nj, ni + 1), "flux_y": (nk, nj + 1, ni),
        "flux_x_w": (nk + 1, nj, ni + 1), "flux_y_w": (nk + 1, nj + 1, ni),
        "thomas_c": (nk, nj, ni), "thomas_d": (nk, nj, ni),
    }
    pointers: list[int] = []
    for name, shape in expected.items():
        value = getattr(workspace, name)
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"workspace {name} must be C-order CUDA float32 with shape {shape}")
        pointers.append(int(value.data.ptr))
    if len(set(pointers)) != len(pointers) or input_pointers.intersection(pointers):
        raise ValueError("GM1 scalar-turbulence workspace arrays alias")


@dataclass(frozen=True, slots=True)
class CudaGm1ScalarTurbulenceLaunch:
    ready_event: Any
    compiled: CompiledGm1ScalarTurbulenceKernels
    workspace: Gm1ScalarTurbulenceWorkspace
    receipt: dict[str, Any]


def launch_cuda_gm1_scalar_turbulence(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dx: float, dy: float, dz: float, dt: float,
    selection: Gm1ScalarTurbulenceSelection = DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION,
    stream: Any | None = None, compiled: CompiledGm1ScalarTurbulenceKernels | None = None,
    workspace: Gm1ScalarTurbulenceWorkspace | None = None,
) -> CudaGm1ScalarTurbulenceLaunch:
    """Enqueue theta plus six moisture contributions without host synchronization."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz, dt = (_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    compiled = compiled or compile_cuda_gm1_scalar_turbulence()
    stream = stream or cupy.cuda.get_current_stream()
    workspace = workspace or create_cuda_gm1_scalar_turbulence_workspace(ni=ni, nj=nj, nk=nk)
    _validate_workspace(cupy, workspace, ni=ni, nj=nj, nk=nk, input_pointers={int(value.data.ptr) for value in a.values()})
    kpad, kpack, krecon, kflux, kinterp, kapply = compiled.kernels
    threads = (CUDA_GM1_SCALAR_TURBULENCE_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    hxy = (nj + 2) * (ni + 2)
    largest = max((nk + 1) * hxy, nk * nj * (ni + 1), nk * (nj + 1) * ni)
    grid = ((largest + CUDA_GM1_SCALAR_TURBULENCE_THREADS - 1) // CUDA_GM1_SCALAR_TURBULENCE_THREADS, 1, 1)
    columns = ni * nj
    column_grid = ((columns + CUDA_GM1_SCALAR_TURBULENCE_THREADS - 1) // CUDA_GM1_SCALAR_TURBULENCE_THREADS, 1, 1)
    with stream:
        kpad(grid, threads, (a["rho"], a["khh"], a["gz"], a["rgz"], workspace.rho_halo, workspace.khh_halo, workspace.gz_halo, workspace.rgz_halo, u32(ni), u32(nj), u32(nk)), stream=stream)
        order = ((0xFFFF_FFFF, a["thten1"]),) + tuple((index, a["qten"][index]) for index in range(6))
        for species, target in order:
            kpack(grid, threads, (a["th0"], a["tha"], a["qa"], workspace.scalar_halo, u32(ni), u32(nj), u32(nk), u32(species)), stream=stream)
            krecon(grid, threads, (workspace.scalar_halo, workspace.scalar_w, u32(ni), u32(nj), u32(nk)), stream=stream)
            kflux(grid, threads, (workspace.scalar_halo, workspace.scalar_w, workspace.rho_halo, workspace.khh_halo, workspace.gz_halo, workspace.rgz_halo, a["rgzu"], a["rgzv"], a["gxu"], a["gyv"], workspace.flux_x, workspace.flux_y, u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(dz)), stream=stream)
            kinterp(grid, threads, (workspace.flux_x, workspace.flux_y, workspace.flux_x_w, workspace.flux_y_w, u32(ni), u32(nj), u32(nk)), stream=stream)
            # src/turbtend.F:588-596: the zero lower flux, since this
            # sealed launcher's selection has surface_flux false.  The
            # pointer argument is never dereferenced with iflux=0, so gz --
            # which every caller already supplies at (nj,ni) -- stands in for
            # an array this selection does not allocate.
            kapply(column_grid, threads, (workspace.scalar_halo, a["rho"], a["rr"], a["rf"], a["mh"], a["mf"], a["khv"], a["gz"], a["rgzu"], a["rgzv"], a["gx"], a["gy"], workspace.flux_x, workspace.flux_y, workspace.flux_x_w, workspace.flux_y_w, a["gz"], workspace.thomas_c, workspace.thomas_d, target, u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(dz), f32(dt), u32(0)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_SCALAR_TURBULENCE_SCHEMA,
        "claim_limit": CUDA_GM1_SCALAR_TURBULENCE_CLAIM_LIMIT,
        "authority": {"product_name": "GM1", "math_release": "CM1 r21.1", "tarball_sha256": AUTHORITY_TARBALL_SHA256, "source_sha256": dict(AUTHORITY_SOURCE_SHA256)},
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "dx_m": dx, "dy_m": dy, "dz_m": dz, "dt_s": dt},
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "source_call_order": list(GM1_SCALAR_TURBULENCE_FIELD_ORDER),
        "cadence": "once per large step in solve1; held across all three RK stages",
        "idiss_seam": "separate solve1 held theta tendency; intentionally not fused or counted here",
        "residency": {"inputs": "CUDA device", "outputs": "CUDA device in-place addition", "scratch": "CUDA device reusable workspace", "host_scientific_copy_or_sync": False},
        "kernel": {"abi": CUDA_GM1_SCALAR_TURBULENCE_ABI, "names": list(CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES), "source_sha256": CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256, "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(), "ptx_bytes": len(compiled.ptx), "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(), "cubin_bytes": len(compiled.cubin), "target_arch": compiled.target_arch, "ptx_options": list(compiled.ptx_options), "cubin_options": list(compiled.cubin_options)},
        "launch": {"block_x": CUDA_GM1_SCALAR_TURBULENCE_THREADS, "grid_x": grid[0], "column_grid_x": column_grid[0], "kernel_launch_count": 1 + 7 * 5, "stream_pointer": int(stream.ptr), "event_pointer": int(event.ptr)},
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {"mode": "raw-bit", "max_ulp": 0, "null_or_missing_is_pass": False},
    }
    return CudaGm1ScalarTurbulenceLaunch(event, compiled, workspace, receipt)


__all__ = [
    "ALL_ARRAY_NAMES", "AUTHORITY_SOURCE_SHA256", "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_SCALAR_TURBULENCE_ABI", "CUDA_GM1_SCALAR_TURBULENCE_CLAIM_LIMIT",
    "CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES", "CUDA_GM1_SCALAR_TURBULENCE_SCHEMA",
    "CUDA_GM1_SCALAR_TURBULENCE_SOURCE", "CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256",
    "CompiledGm1ScalarTurbulenceKernels", "CudaGm1ScalarTurbulenceLaunch",
    "DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION", "GM1_SCALAR_TURBULENCE_FIELD_ORDER",
    "GM1_SCALAR_TURBULENCE_SPECIES", "Gm1ScalarTurbulenceSelection",
    "Gm1ScalarTurbulenceWorkspace", "INPUT_NAMES", "OUTPUT_NAMES", "SOURCE_ANCHORS",
    "compile_cuda_gm1_scalar_turbulence", "cpu_gm1_scalar_turbulence_inplace",
    "create_cuda_gm1_scalar_turbulence_workspace", "gm1_scalar_turbulence_shapes",
    "launch_cuda_gm1_scalar_turbulence",
]
