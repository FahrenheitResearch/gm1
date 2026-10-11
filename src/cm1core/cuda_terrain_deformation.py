"""CM1 r21.1 terrain ``calcdef`` CUDA operator for the all-open product grid.

This module is deliberately narrow.  It implements the Cartesian-with-terrain
branch selected by the corrected-real 300 x 300 x 75 configuration, including
the serial all-open boundary values that are consumed by deformation.  The
input ABI is the physical-domain initialized-product ABI; no atmospheric field
is synthesized and no host scientific transform is used by the CUDA path.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_TERRAIN_DEFORMATION_SCHEMA = "gpuwm-cm1.cuda-terrain-deformation/v1"
CUDA_TERRAIN_DEFORMATION_ABI = "gpuwm-cm1.cuda-terrain-deformation-kernels/v1"
CUDA_TERRAIN_DEFORMATION_THREADS = 256
CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES = (
    "cm1_calcdef_terrain_mass_v1",
    "cm1_calcdef_terrain_face_v1",
    "cm1_calcdef_terrain_reduce_v1",
)
CUDA_TERRAIN_DEFORMATION_CLAIM_LIMIT = (
    "CM1 r21.1 serial Cartesian terrain calcdef for the uniform 100 m x 100 m "
    "x 200 m, all-open, axisymmetric=0 physical-domain product ABI; source "
    "open-boundary constant extension and s12 boundary copies are included; "
    "MPI partition halos, periodic/free-slip/no-slip lateral branches, "
    "nonterrain/axisymmetric branches, momentum stress divergence, RK "
    "evolution, forecast, storm, and tornado claims are excluded"
)

SOURCE_ANCHORS = {
    "dispatch": "src/turb.F:1326-1389",
    "calcdef": "src/turb.F:4711-5423",
    "terrain_strain": "src/turb.F:5060-5167",
    "open_s12": "src/turb.F:5171-5290",
    "deformation": "src/turb.F:5340-5382",
    "open_boundary_fill": "src/bc.F:12-172",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/turb.F": "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d",
}

INPUT_NAMES = (
    "u3d",
    "v3d",
    "w3d",
    "rho",
    "rf",
    "c1",
    "c2",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "gx",
    "gy",
)
OUTPUT_NAMES = ("defv", "defh")
ALL_ARRAY_NAMES = INPUT_NAMES + OUTPUT_NAMES


@dataclass(frozen=True, slots=True)
class TerrainDeformationSelection:
    terrain: bool = True
    axisymmetric: int = 0
    lateral_boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)
    serial_full_domain: bool = True


DEFAULT_TERRAIN_DEFORMATION_SELECTION = TerrainDeformationSelection()


def _validate_selection(selection: TerrainDeformationSelection) -> None:
    if not isinstance(selection, TerrainDeformationSelection):
        raise TypeError("selection must be a TerrainDeformationSelection")
    expected = DEFAULT_TERRAIN_DEFORMATION_SELECTION
    if selection != expected:
        refuse(
            code="CM1-CALCDEF-UNSUPPORTED",
            gate="physics-seam",
            option="terrain_calcdef_selection",
            value={
                "terrain": selection.terrain,
                "axisymmetric": selection.axisymmetric,
                "lateral_boundaries": list(selection.lateral_boundaries),
                "serial_full_domain": selection.serial_full_domain,
            },
            reason=(
                "the sealed operator covers only the corrected-real serial "
                "Cartesian terrain branch with four open lateral boundaries"
            ),
            remedy="use the product selection or add an independent native oracle",
        )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * (nk + 1) > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError(
            "terrain-deformation word count exceeds signed 64-bit indexing"
        )


def _spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def terrain_deformation_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    cell = (nk, nj, ni)
    face = (nk + 1, nj, ni)
    return {
        "u3d": (nk, nj, ni + 1),
        "v3d": (nk, nj + 1, ni),
        "w3d": face,
        "rho": cell,
        "rf": face,
        "c1": face,
        "c2": face,
        "gz": (nj, ni),
        "rgz": (nj, ni),
        "gzu": (nj, ni + 1),
        "rgzu": (nj, ni + 1),
        "gzv": (nj + 1, ni),
        "rgzv": (nj + 1, ni),
        "gx": face,
        "gy": face,
        "defv": face,
        "defh": face,
    }


def terrain_deformation_scratch_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    return {
        "s11": (nk, nj, ni),
        "s22": (nk, nj, ni),
        "s33": (nk, nj, ni),
        "s12": (nk, nj + 1, ni + 1),
        "s13": (nk + 1, nj, ni + 1),
        "s23": (nk + 1, nj + 1, ni),
    }


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = terrain_deformation_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing terrain-deformation arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must have C-order shape {shapes[name]}")
        if name in OUTPUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("terrain-deformation arrays must not alias")
    return result


def _a(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _s(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _m(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _d(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _atw(np: Any, value: Any, kf: int, j: int, i: int, nk: int) -> Any:
    if kf == 0:
        return _a(
            np,
            _a(
                np,
                _m(np, np.float32(1.875), value[0, j, i]),
                _m(np, np.float32(-1.25), value[1, j, i]),
            ),
            _m(np, np.float32(0.375), value[2, j, i]),
        )
    if kf == nk:
        return _a(
            np,
            _a(
                np,
                _m(np, np.float32(1.875), value[nk - 1, j, i]),
                _m(np, np.float32(-1.25), value[nk - 2, j, i]),
            ),
            _m(np, np.float32(0.375), value[nk - 3, j, i]),
        )
    return _m(np, np.float32(0.5), _a(np, value[kf - 1, j, i], value[kf, j, i]))


def cpu_terrain_deformation_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainDeformationSelection = DEFAULT_TERRAIN_DEFORMATION_SELECTION,
) -> dict[str, int | float]:
    """Independent scalar reference for the selected terrain ``calcdef`` branch."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz = _spacing("dx", dx), _spacing("dy", dy), _spacing("dz", dz)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    scratch = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in terrain_deformation_scratch_shapes(ni, nj, nk).items()
    }
    rdx, rdy, rds = (_d(np, np.float32(1.0), np.float32(v)) for v in (dx, dy, dz))
    half, quarter = np.float32(0.5), np.float32(0.25)
    zt = _m(np, np.float32(nk), np.float32(dz))

    for k in range(nk):
        sigma_lo = _m(np, np.float32(k), np.float32(dz))
        sigma_hi = _m(np, np.float32(k + 1), np.float32(dz))
        for j in range(nj):
            for i in range(ni):
                hor = _m(
                    np,
                    a["gz"][j, i],
                    _s(
                        np,
                        _m(np, a["u3d"][k, j, i + 1], a["rgzu"][j, i + 1]),
                        _m(np, a["u3d"][k, j, i], a["rgzu"][j, i]),
                    ),
                )
                hor = _m(np, hor, rdx)
                upper = _m(
                    np,
                    a["gx"][k + 1, j, i],
                    _a(
                        np,
                        _atw(np, a["u3d"], k + 1, j, i, nk),
                        _atw(np, a["u3d"], k + 1, j, i + 1, nk),
                    ),
                )
                lower = _m(
                    np,
                    a["gx"][k, j, i],
                    _a(
                        np,
                        _atw(np, a["u3d"], k, j, i, nk),
                        _atw(np, a["u3d"], k, j, i + 1, nk),
                    ),
                )
                vert = _m(np, _m(np, _s(np, upper, lower), half), rds)
                scratch["s11"][k, j, i] = _m(np, _a(np, hor, vert), a["rho"][k, j, i])

                hor = _m(
                    np,
                    a["gz"][j, i],
                    _s(
                        np,
                        _m(np, a["v3d"][k, j + 1, i], a["rgzv"][j + 1, i]),
                        _m(np, a["v3d"][k, j, i], a["rgzv"][j, i]),
                    ),
                )
                hor = _m(np, hor, rdy)
                upper = _m(
                    np,
                    a["gy"][k + 1, j, i],
                    _a(
                        np,
                        _atw(np, a["v3d"], k + 1, j, i, nk),
                        _atw(np, a["v3d"], k + 1, j + 1, i, nk),
                    ),
                )
                lower = _m(
                    np,
                    a["gy"][k, j, i],
                    _a(
                        np,
                        _atw(np, a["v3d"], k, j, i, nk),
                        _atw(np, a["v3d"], k, j + 1, i, nk),
                    ),
                )
                vert = _m(np, _m(np, _s(np, upper, lower), half), rds)
                scratch["s22"][k, j, i] = _m(np, _a(np, hor, vert), a["rho"][k, j, i])
                scratch["s33"][k, j, i] = _m(
                    np,
                    _m(
                        np,
                        _m(np, _s(np, a["w3d"][k + 1, j, i], a["w3d"][k, j, i]), rds),
                        a["gz"][j, i],
                    ),
                    a["rho"][k, j, i],
                )

        for jc in range(nj + 1):
            jj = min(max(jc, 1), nj - 1)
            for ic in range(ni + 1):
                ii = min(max(ic, 1), ni - 1)
                r1 = _m(
                    np,
                    quarter,
                    _a(
                        np,
                        _a(
                            np,
                            _m(
                                np, a["rho"][k, jj - 1, ii - 1], a["gz"][jj - 1, ii - 1]
                            ),
                            _m(np, a["rho"][k, jj, ii], a["gz"][jj, ii]),
                        ),
                        _a(
                            np,
                            _m(np, a["rho"][k, jj, ii - 1], a["gz"][jj, ii - 1]),
                            _m(np, a["rho"][k, jj - 1, ii], a["gz"][jj - 1, ii]),
                        ),
                    ),
                )
                x1 = _m(
                    np,
                    _m(
                        np,
                        r1,
                        _s(
                            np,
                            _m(np, a["u3d"][k, jj, ii], a["rgzu"][jj, ii]),
                            _m(np, a["u3d"][k, jj - 1, ii], a["rgzu"][jj - 1, ii]),
                        ),
                    ),
                    rdy,
                )
                xu = _m(
                    np,
                    _s(np, zt, sigma_hi),
                    _a(
                        np,
                        _atw(np, a["u3d"], k + 1, jj - 1, ii, nk),
                        _atw(np, a["u3d"], k + 1, jj, ii, nk),
                    ),
                )
                xl = _m(
                    np,
                    _s(np, zt, sigma_lo),
                    _a(
                        np,
                        _atw(np, a["u3d"], k, jj - 1, ii, nk),
                        _atw(np, a["u3d"], k, jj, ii, nk),
                    ),
                )
                x2 = _m(
                    np,
                    _m(
                        np,
                        _m(np, _m(np, _m(np, half, _s(np, xu, xl)), rds), r1),
                        _s(np, a["rgzu"][jj, ii], a["rgzu"][jj - 1, ii]),
                    ),
                    rdy,
                )
                y1 = _m(
                    np,
                    _m(
                        np,
                        r1,
                        _s(
                            np,
                            _m(np, a["v3d"][k, jj, ii], a["rgzv"][jj, ii]),
                            _m(np, a["v3d"][k, jj, ii - 1], a["rgzv"][jj, ii - 1]),
                        ),
                    ),
                    rdx,
                )
                yu = _m(
                    np,
                    _s(np, zt, sigma_hi),
                    _a(
                        np,
                        _atw(np, a["v3d"], k + 1, jj, ii - 1, nk),
                        _atw(np, a["v3d"], k + 1, jj, ii, nk),
                    ),
                )
                yl = _m(
                    np,
                    _s(np, zt, sigma_lo),
                    _a(
                        np,
                        _atw(np, a["v3d"], k, jj, ii - 1, nk),
                        _atw(np, a["v3d"], k, jj, ii, nk),
                    ),
                )
                y2 = _m(
                    np,
                    _m(
                        np,
                        _m(np, _m(np, _m(np, half, _s(np, yu, yl)), rds), r1),
                        _s(np, a["rgzv"][jj, ii], a["rgzv"][jj, ii - 1]),
                    ),
                    rdx,
                )
                scratch["s12"][k, jc, ic] = _m(
                    np, half, _a(np, _a(np, x1, x2), _a(np, y1, y2))
                )

    for kf in range(1, nk):
        sigma_lower = _m(np, _s(np, np.float32(kf), half), np.float32(dz))
        sigma_upper = _m(np, _a(np, np.float32(kf), half), np.float32(dz))
        for j in range(nj):
            for ic in range(ni + 1):
                il, ir = max(ic - 1, 0), min(ic, ni - 1)
                du = _m(np, _s(np, a["u3d"][kf, j, ic], a["u3d"][kf - 1, j, ic]), rds)
                hw = _m(
                    np,
                    _s(
                        np,
                        _m(np, a["w3d"][kf, j, ir], a["rgz"][j, ir]),
                        _m(np, a["w3d"][kf, j, il], a["rgz"][j, il]),
                    ),
                    rdx,
                )
                wu = _m(
                    np,
                    _s(np, zt, sigma_upper),
                    _a(
                        np,
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf, j, ir], a["w3d"][kf + 1, j, ir]),
                        ),
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf, j, il], a["w3d"][kf + 1, j, il]),
                        ),
                    ),
                )
                wl = _m(
                    np,
                    _s(np, zt, sigma_lower),
                    _a(
                        np,
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf - 1, j, ir], a["w3d"][kf, j, ir]),
                        ),
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf - 1, j, il], a["w3d"][kf, j, il]),
                        ),
                    ),
                )
                terrain = _m(
                    np,
                    _m(
                        np,
                        _m(np, _m(np, half, rds), _s(np, wu, wl)),
                        _s(np, a["rgz"][j, ir], a["rgz"][j, il]),
                    ),
                    rdx,
                )
                strain = _m(np, half, _a(np, _a(np, du, hw), terrain))
                density = _m(
                    np,
                    half,
                    _a(
                        np,
                        _m(np, a["gz"][j, il], a["rf"][kf, j, il]),
                        _m(np, a["gz"][j, ir], a["rf"][kf, j, ir]),
                    ),
                )
                scratch["s13"][kf, j, ic] = _m(np, strain, density)
        for jc in range(nj + 1):
            jb, jt = max(jc - 1, 0), min(jc, nj - 1)
            for i in range(ni):
                dv = _m(np, _s(np, a["v3d"][kf, jc, i], a["v3d"][kf - 1, jc, i]), rds)
                hw = _m(
                    np,
                    _s(
                        np,
                        _m(np, a["w3d"][kf, jt, i], a["rgz"][jt, i]),
                        _m(np, a["w3d"][kf, jb, i], a["rgz"][jb, i]),
                    ),
                    rdy,
                )
                wu = _m(
                    np,
                    _s(np, zt, sigma_upper),
                    _a(
                        np,
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf, jt, i], a["w3d"][kf + 1, jt, i]),
                        ),
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf, jb, i], a["w3d"][kf + 1, jb, i]),
                        ),
                    ),
                )
                wl = _m(
                    np,
                    _s(np, zt, sigma_lower),
                    _a(
                        np,
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf - 1, jt, i], a["w3d"][kf, jt, i]),
                        ),
                        _m(
                            np,
                            half,
                            _a(np, a["w3d"][kf - 1, jb, i], a["w3d"][kf, jb, i]),
                        ),
                    ),
                )
                terrain = _m(
                    np,
                    _m(
                        np,
                        _m(np, _m(np, half, rds), _s(np, wu, wl)),
                        _s(np, a["rgz"][jt, i], a["rgz"][jb, i]),
                    ),
                    rdy,
                )
                strain = _m(np, half, _a(np, _a(np, dv, hw), terrain))
                density = _m(
                    np,
                    half,
                    _a(
                        np,
                        _m(np, a["gz"][jb, i], a["rf"][kf, jb, i]),
                        _m(np, a["gz"][jt, i], a["rf"][kf, jt, i]),
                    ),
                )
                scratch["s23"][kf, jc, i] = _m(np, strain, density)

    a["defv"].fill(np.float32(0.0))
    a["defh"].fill(np.float32(0.0))
    for kf in range(1, nk):
        for j in range(nj):
            for i in range(ni):
                c1, c2 = a["c1"][kf, j, i], a["c2"][kf, j, i]
                tmp11 = _a(
                    np,
                    _m(
                        np,
                        c1,
                        _m(
                            np,
                            scratch["s11"][kf - 1, j, i],
                            scratch["s11"][kf - 1, j, i],
                        ),
                    ),
                    _m(
                        np,
                        c2,
                        _m(np, scratch["s11"][kf, j, i], scratch["s11"][kf, j, i]),
                    ),
                )
                tmp22 = _a(
                    np,
                    _m(
                        np,
                        c1,
                        _m(
                            np,
                            scratch["s22"][kf - 1, j, i],
                            scratch["s22"][kf - 1, j, i],
                        ),
                    ),
                    _m(
                        np,
                        c2,
                        _m(np, scratch["s22"][kf, j, i], scratch["s22"][kf, j, i]),
                    ),
                )
                tmp33 = _a(
                    np,
                    _m(
                        np,
                        c1,
                        _m(
                            np,
                            scratch["s33"][kf - 1, j, i],
                            scratch["s33"][kf - 1, j, i],
                        ),
                    ),
                    _m(
                        np,
                        c2,
                        _m(np, scratch["s33"][kf, j, i], scratch["s33"][kf, j, i]),
                    ),
                )
                lo12 = _a(
                    np,
                    _a(
                        np,
                        _m(
                            np,
                            scratch["s12"][kf - 1, j, i],
                            scratch["s12"][kf - 1, j, i],
                        ),
                        _m(
                            np,
                            scratch["s12"][kf - 1, j + 1, i + 1],
                            scratch["s12"][kf - 1, j + 1, i + 1],
                        ),
                    ),
                    _a(
                        np,
                        _m(
                            np,
                            scratch["s12"][kf - 1, j + 1, i],
                            scratch["s12"][kf - 1, j + 1, i],
                        ),
                        _m(
                            np,
                            scratch["s12"][kf - 1, j, i + 1],
                            scratch["s12"][kf - 1, j, i + 1],
                        ),
                    ),
                )
                hi12 = _a(
                    np,
                    _a(
                        np,
                        _m(np, scratch["s12"][kf, j, i], scratch["s12"][kf, j, i]),
                        _m(
                            np,
                            scratch["s12"][kf, j + 1, i + 1],
                            scratch["s12"][kf, j + 1, i + 1],
                        ),
                    ),
                    _a(
                        np,
                        _m(
                            np,
                            scratch["s12"][kf, j + 1, i],
                            scratch["s12"][kf, j + 1, i],
                        ),
                        _m(
                            np,
                            scratch["s12"][kf, j, i + 1],
                            scratch["s12"][kf, j, i + 1],
                        ),
                    ),
                )
                tmp12 = _m(np, quarter, _a(np, _m(np, c1, lo12), _m(np, c2, hi12)))
                tmp13 = _m(
                    np,
                    half,
                    _a(
                        np,
                        _m(np, scratch["s13"][kf, j, i], scratch["s13"][kf, j, i]),
                        _m(
                            np,
                            scratch["s13"][kf, j, i + 1],
                            scratch["s13"][kf, j, i + 1],
                        ),
                    ),
                )
                tmp23 = _m(
                    np,
                    half,
                    _a(
                        np,
                        _m(np, scratch["s23"][kf, j, i], scratch["s23"][kf, j, i]),
                        _m(
                            np,
                            scratch["s23"][kf, j + 1, i],
                            scratch["s23"][kf, j + 1, i],
                        ),
                    ),
                )
                rrf = _d(
                    np, np.float32(1.0), _m(np, a["rf"][kf, j, i], a["rf"][kf, j, i])
                )
                a["defv"][kf, j, i] = _m(
                    np, _m(np, np.float32(4.0), _a(np, tmp13, tmp23)), rrf
                )
                diag = _m(np, np.float32(2.0), _a(np, _a(np, tmp11, tmp22), tmp33))
                a["defh"][kf, j, i] = _m(
                    np, _a(np, diag, _m(np, np.float32(4.0), tmp12)), rrf
                )
    return {
        "live_words": ni * nj * (nk - 1),
        "boundary_face_zero_words": 2 * ni * nj,
        "minimum_defv": float(a["defv"][1:nk].min()),
        "maximum_defv": float(a["defv"][1:nk].max()),
        "minimum_defh": float(a["defh"][1:nk].min()),
        "maximum_defh": float(a["defh"][1:nk].max()),
    }


CUDA_TERRAIN_DEFORMATION_SOURCE = r"""// gpuwm-cm1.cuda-terrain-deformation-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i2(unsigned int j,unsigned int i,unsigned int nx){return (unsigned long long)j*nx+i;}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ unsigned int clampc(int x,unsigned int n){return x<0?0u:(x>=(int)n?n-1u:(unsigned int)x);}
__device__ __forceinline__ float atwu(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni){
 if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(1u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(2u,j,i,nj,ni+1u)]));
 if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(nk-2u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(nk-3u,j,i,nj,ni+1u)]));
 return m32(0.5f,a32(q[i3(kf-1u,j,i,nj,ni+1u)],q[i3(kf,j,i,nj,ni+1u)]));}
__device__ __forceinline__ float atwv(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni){
 if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(1u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(2u,j,i,nj+1u,ni)]));
 if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(nk-2u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(nk-3u,j,i,nj+1u,ni)]));
 return m32(0.5f,a32(q[i3(kf-1u,j,i,nj+1u,ni)],q[i3(kf,j,i,nj+1u,ni)]));}
extern "C" __global__ void cm1_calcdef_terrain_mass_v1(
 const float* u,const float* v,const float* w,const float* rho,const float* gz,const float* rgzu,const float* rgzv,const float* gx,const float* gy,
 float* s11,float* s22,float* s33,float* s12,unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rds,float dz){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*(nj+1u)*(ni+1u);const float zt=m32((float)nk,dz);
 for(unsigned long long q=first;q<total;q+=step){unsigned int ic=(unsigned int)(q%(ni+1u)),jc=(unsigned int)((q/(ni+1u))%(nj+1u)),k=(unsigned int)(q/((unsigned long long)(ni+1u)*(nj+1u)));
  if(ic<ni&&jc<nj){unsigned long long p=i3(k,jc,ic,nj,ni);float h=m32(gz[i2(jc,ic,ni)],s32(m32(u[i3(k,jc,ic+1u,nj,ni+1u)],rgzu[i2(jc,ic+1u,ni+1u)]),m32(u[i3(k,jc,ic,nj,ni+1u)],rgzu[i2(jc,ic,ni+1u)])));h=m32(h,rdx);float up=m32(gx[i3(k+1u,jc,ic,nj,ni)],a32(atwu(u,k+1u,jc,ic,nk,nj,ni),atwu(u,k+1u,jc,ic+1u,nk,nj,ni))),lo=m32(gx[i3(k,jc,ic,nj,ni)],a32(atwu(u,k,jc,ic,nk,nj,ni),atwu(u,k,jc,ic+1u,nk,nj,ni)));s11[p]=m32(a32(h,m32(m32(s32(up,lo),0.5f),rds)),rho[p]);
   h=m32(gz[i2(jc,ic,ni)],s32(m32(v[i3(k,jc+1u,ic,nj+1u,ni)],rgzv[i2(jc+1u,ic,ni)]),m32(v[i3(k,jc,ic,nj+1u,ni)],rgzv[i2(jc,ic,ni)])));h=m32(h,rdy);up=m32(gy[i3(k+1u,jc,ic,nj,ni)],a32(atwv(v,k+1u,jc,ic,nk,nj,ni),atwv(v,k+1u,jc+1u,ic,nk,nj,ni)));lo=m32(gy[i3(k,jc,ic,nj,ni)],a32(atwv(v,k,jc,ic,nk,nj,ni),atwv(v,k,jc+1u,ic,nk,nj,ni)));s22[p]=m32(a32(h,m32(m32(s32(up,lo),0.5f),rds)),rho[p]);s33[p]=m32(m32(m32(s32(w[i3(k+1u,jc,ic,nj,ni)],w[i3(k,jc,ic,nj,ni)]),rds),gz[i2(jc,ic,ni)]),rho[p]);}
  unsigned int ii=ic<1u?1u:(ic>ni-1u?ni-1u:ic),jj=jc<1u?1u:(jc>nj-1u?nj-1u:jc);float r1=m32(0.25f,a32(a32(m32(rho[i3(k,jj-1u,ii-1u,nj,ni)],gz[i2(jj-1u,ii-1u,ni)]),m32(rho[i3(k,jj,ii,nj,ni)],gz[i2(jj,ii,ni)])),a32(m32(rho[i3(k,jj,ii-1u,nj,ni)],gz[i2(jj,ii-1u,ni)]),m32(rho[i3(k,jj-1u,ii,nj,ni)],gz[i2(jj-1u,ii,ni)]))));float sl=m32((float)k,dz),su=m32((float)(k+1u),dz);
  float x1=m32(m32(r1,s32(m32(u[i3(k,jj,ii,nj,ni+1u)],rgzu[i2(jj,ii,ni+1u)]),m32(u[i3(k,jj-1u,ii,nj,ni+1u)],rgzu[i2(jj-1u,ii,ni+1u)]))),rdy);float xu=m32(s32(zt,su),a32(atwu(u,k+1u,jj-1u,ii,nk,nj,ni),atwu(u,k+1u,jj,ii,nk,nj,ni))),xl=m32(s32(zt,sl),a32(atwu(u,k,jj-1u,ii,nk,nj,ni),atwu(u,k,jj,ii,nk,nj,ni)));float x2=m32(m32(m32(m32(m32(0.5f,s32(xu,xl)),rds),r1),s32(rgzu[i2(jj,ii,ni+1u)],rgzu[i2(jj-1u,ii,ni+1u)])),rdy);
  float y1=m32(m32(r1,s32(m32(v[i3(k,jj,ii,nj+1u,ni)],rgzv[i2(jj,ii,ni)]),m32(v[i3(k,jj,ii-1u,nj+1u,ni)],rgzv[i2(jj,ii-1u,ni)]))),rdx);float yu=m32(s32(zt,su),a32(atwv(v,k+1u,jj,ii-1u,nk,nj,ni),atwv(v,k+1u,jj,ii,nk,nj,ni))),yl=m32(s32(zt,sl),a32(atwv(v,k,jj,ii-1u,nk,nj,ni),atwv(v,k,jj,ii,nk,nj,ni)));float y2=m32(m32(m32(m32(m32(0.5f,s32(yu,yl)),rds),r1),s32(rgzv[i2(jj,ii,ni)],rgzv[i2(jj,ii-1u,ni)])),rdx);s12[i3(k,jc,ic,nj+1u,ni+1u)]=m32(0.5f,a32(a32(x1,x2),a32(y1,y2)));}}
extern "C" __global__ void cm1_calcdef_terrain_face_v1(
 const float* u,const float* v,const float* w,const float* rf,const float* gz,const float* rgz,float* s13,float* s23,
 unsigned int ni,unsigned int nj,unsigned int nk,float rdx,float rdy,float rds,float dz){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)(nk-1u)*(nj+1u)*(ni+1u);const float zt=m32((float)nk,dz);
 for(unsigned long long q=first;q<total;q+=step){unsigned int ic=(unsigned int)(q%(ni+1u)),jc=(unsigned int)((q/(ni+1u))%(nj+1u)),kf=(unsigned int)(q/((unsigned long long)(ni+1u)*(nj+1u)))+1u;float slo=m32(s32((float)kf,0.5f),dz),sup=m32(a32((float)kf,0.5f),dz);
  if(jc<nj){unsigned int il=ic?ic-1u:0u,ir=ic<ni?ic:ni-1u;float du=m32(s32(u[i3(kf,jc,ic,nj,ni+1u)],u[i3(kf-1u,jc,ic,nj,ni+1u)]),rds),hw=m32(s32(m32(w[i3(kf,jc,ir,nj,ni)],rgz[i2(jc,ir,ni)]),m32(w[i3(kf,jc,il,nj,ni)],rgz[i2(jc,il,ni)])),rdx);float wur=m32(0.5f,a32(w[i3(kf,jc,ir,nj,ni)],w[i3(kf+1u,jc,ir,nj,ni)])),wul=m32(0.5f,a32(w[i3(kf,jc,il,nj,ni)],w[i3(kf+1u,jc,il,nj,ni)])),wlr=m32(0.5f,a32(w[i3(kf-1u,jc,ir,nj,ni)],w[i3(kf,jc,ir,nj,ni)])),wll=m32(0.5f,a32(w[i3(kf-1u,jc,il,nj,ni)],w[i3(kf,jc,il,nj,ni)]));float tr=m32(m32(m32(m32(0.5f,rds),s32(m32(s32(zt,sup),a32(wur,wul)),m32(s32(zt,slo),a32(wlr,wll)))),s32(rgz[i2(jc,ir,ni)],rgz[i2(jc,il,ni)])),rdx);float den=m32(0.5f,a32(m32(gz[i2(jc,il,ni)],rf[i3(kf,jc,il,nj,ni)]),m32(gz[i2(jc,ir,ni)],rf[i3(kf,jc,ir,nj,ni)])));s13[i3(kf,jc,ic,nj,ni+1u)]=m32(m32(0.5f,a32(a32(du,hw),tr)),den);}
  if(ic<ni){unsigned int jb=jc?jc-1u:0u,jt=jc<nj?jc:nj-1u;float dv=m32(s32(v[i3(kf,jc,ic,nj+1u,ni)],v[i3(kf-1u,jc,ic,nj+1u,ni)]),rds),hw=m32(s32(m32(w[i3(kf,jt,ic,nj,ni)],rgz[i2(jt,ic,ni)]),m32(w[i3(kf,jb,ic,nj,ni)],rgz[i2(jb,ic,ni)])),rdy);float wut=m32(0.5f,a32(w[i3(kf,jt,ic,nj,ni)],w[i3(kf+1u,jt,ic,nj,ni)])),wub=m32(0.5f,a32(w[i3(kf,jb,ic,nj,ni)],w[i3(kf+1u,jb,ic,nj,ni)])),wlt=m32(0.5f,a32(w[i3(kf-1u,jt,ic,nj,ni)],w[i3(kf,jt,ic,nj,ni)])),wlb=m32(0.5f,a32(w[i3(kf-1u,jb,ic,nj,ni)],w[i3(kf,jb,ic,nj,ni)]));float tr=m32(m32(m32(m32(0.5f,rds),s32(m32(s32(zt,sup),a32(wut,wub)),m32(s32(zt,slo),a32(wlt,wlb)))),s32(rgz[i2(jt,ic,ni)],rgz[i2(jb,ic,ni)])),rdy);float den=m32(0.5f,a32(m32(gz[i2(jb,ic,ni)],rf[i3(kf,jb,ic,nj,ni)]),m32(gz[i2(jt,ic,ni)],rf[i3(kf,jt,ic,nj,ni)])));s23[i3(kf,jc,ic,nj+1u,ni)]=m32(m32(0.5f,a32(a32(dv,hw),tr)),den);}}}
extern "C" __global__ void cm1_calcdef_terrain_reduce_v1(
 const float* rf,const float* c1,const float* c2,const float* s11,const float* s22,const float* s33,const float* s12,const float* s13,const float* s23,float* defv,float* defh,
 unsigned int ni,unsigned int nj,unsigned int nk){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)ni*nj*(nk-1u),nxy=(unsigned long long)ni*nj;
 for(unsigned long long q=first;q<total;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),kf=(unsigned int)(q/nxy)+1u;unsigned long long p=i3(kf,j,i,nj,ni),lo=i3(kf-1u,j,i,nj,ni),hi=i3(kf,j,i,nj,ni);float a=c1[p],b=c2[p];float t11=a32(m32(a,m32(s11[lo],s11[lo])),m32(b,m32(s11[hi],s11[hi]))),t22=a32(m32(a,m32(s22[lo],s22[lo])),m32(b,m32(s22[hi],s22[hi]))),t33=a32(m32(a,m32(s33[lo],s33[lo])),m32(b,m32(s33[hi],s33[hi])));
  unsigned long long l00=i3(kf-1u,j,i,nj+1u,ni+1u),l11=i3(kf-1u,j+1u,i+1u,nj+1u,ni+1u),l10=i3(kf-1u,j+1u,i,nj+1u,ni+1u),l01=i3(kf-1u,j,i+1u,nj+1u,ni+1u),h00=i3(kf,j,i,nj+1u,ni+1u),h11=i3(kf,j+1u,i+1u,nj+1u,ni+1u),h10=i3(kf,j+1u,i,nj+1u,ni+1u),h01=i3(kf,j,i+1u,nj+1u,ni+1u);float ls=a32(a32(m32(s12[l00],s12[l00]),m32(s12[l11],s12[l11])),a32(m32(s12[l10],s12[l10]),m32(s12[l01],s12[l01]))),hs=a32(a32(m32(s12[h00],s12[h00]),m32(s12[h11],s12[h11])),a32(m32(s12[h10],s12[h10]),m32(s12[h01],s12[h01]))),t12=m32(0.25f,a32(m32(a,ls),m32(b,hs)));unsigned long long x0=i3(kf,j,i,nj,ni+1u),x1=i3(kf,j,i+1u,nj,ni+1u),y0=i3(kf,j,i,nj+1u,ni),y1=i3(kf,j+1u,i,nj+1u,ni);float t13=m32(0.5f,a32(m32(s13[x0],s13[x0]),m32(s13[x1],s13[x1]))),t23=m32(0.5f,a32(m32(s23[y0],s23[y0]),m32(s23[y1],s23[y1]))),rrf=d32(1.0f,m32(rf[p],rf[p]));defv[p]=m32(m32(4.0f,a32(t13,t23)),rrf);defh[p]=m32(a32(m32(2.0f,a32(a32(t11,t22),t33)),m32(4.0f,t12)),rrf);}}
"""
CUDA_TERRAIN_DEFORMATION_SOURCE_BYTES = CUDA_TERRAIN_DEFORMATION_SOURCE.encode("utf-8")
CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256 = (
    "e642efc44273461ddfc1934012aaac87bca222a0b5ce5f713438f263d3265c62"
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
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledTerrainDeformationKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_terrain_deformation() -> CompiledTerrainDeformationKernels:
    digest = hashlib.sha256(CUDA_TERRAIN_DEFORMATION_SOURCE_BYTES).hexdigest()
    if digest != CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256:
        raise RuntimeError("packaged terrain-deformation CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from active CUDA device")
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
        CUDA_TERRAIN_DEFORMATION_SOURCE,
        "cm1_terrain_deformation_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_TERRAIN_DEFORMATION_SOURCE,
        "cm1_terrain_deformation_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", text)
        for name in CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks terrain-deformation entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled terrain-deformation image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledTerrainDeformationKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, runtime
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = terrain_deformation_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing terrain-deformation arrays: {', '.join(missing)}")
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
            raise ValueError(f"{name} must have C-order float32 shape {shapes[name]}")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("terrain-deformation arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaTerrainDeformationLaunch:
    ready_event: Any
    compiled: CompiledTerrainDeformationKernels
    scratch: Mapping[str, Any]
    receipt: dict[str, Any]


def launch_cuda_terrain_deformation(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainDeformationSelection = DEFAULT_TERRAIN_DEFORMATION_SELECTION,
    stream: Any | None = None,
    compiled: CompiledTerrainDeformationKernels | None = None,
) -> CudaTerrainDeformationLaunch:
    """Enqueue the selected source-ordered terrain ``calcdef`` operator."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz = _spacing("dx", dx), _spacing("dy", dy), _spacing("dz", dz)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_terrain_deformation()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    scratch = {
        name: cupy.empty(shape, dtype=cupy.float32)
        for name, shape in terrain_deformation_scratch_shapes(ni, nj, nk).items()
    }
    a["defv"].fill(cupy.float32(0.0))
    a["defh"].fill(cupy.float32(0.0))
    mass_words = nk * (nj + 1) * (ni + 1)
    face_words = (nk - 1) * (nj + 1) * (ni + 1)
    live_words = (nk - 1) * nj * ni
    grids = tuple(
        (words + CUDA_TERRAIN_DEFORMATION_THREADS - 1)
        // CUDA_TERRAIN_DEFORMATION_THREADS
        for words in (mass_words, face_words, live_words)
    )
    u32, f32 = cupy.uint32, cupy.float32
    rdx, rdy, rds = (f32(1.0 / value) for value in (dx, dy, dz))
    compiled.kernels[0](
        (grids[0], 1, 1),
        (CUDA_TERRAIN_DEFORMATION_THREADS, 1, 1),
        (
            a["u3d"],
            a["v3d"],
            a["w3d"],
            a["rho"],
            a["gz"],
            a["rgzu"],
            a["rgzv"],
            a["gx"],
            a["gy"],
            scratch["s11"],
            scratch["s22"],
            scratch["s33"],
            scratch["s12"],
            u32(ni),
            u32(nj),
            u32(nk),
            rdx,
            rdy,
            rds,
            f32(dz),
        ),
        stream=stream,
    )
    compiled.kernels[1](
        (grids[1], 1, 1),
        (CUDA_TERRAIN_DEFORMATION_THREADS, 1, 1),
        (
            a["u3d"],
            a["v3d"],
            a["w3d"],
            a["rf"],
            a["gz"],
            a["rgz"],
            scratch["s13"],
            scratch["s23"],
            u32(ni),
            u32(nj),
            u32(nk),
            rdx,
            rdy,
            rds,
            f32(dz),
        ),
        stream=stream,
    )
    compiled.kernels[2](
        (grids[2], 1, 1),
        (CUDA_TERRAIN_DEFORMATION_THREADS, 1, 1),
        (
            a["rf"],
            a["c1"],
            a["c2"],
            scratch["s11"],
            scratch["s22"],
            scratch["s33"],
            scratch["s12"],
            scratch["s13"],
            scratch["s23"],
            a["defv"],
            a["defh"],
            u32(ni),
            u32(nj),
            u32(nk),
        ),
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_TERRAIN_DEFORMATION_SCHEMA,
        "claim_limit": CUDA_TERRAIN_DEFORMATION_CLAIM_LIMIT,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {
            "terrain": True,
            "axisymmetric": 0,
            "lateral_boundaries": [2, 2, 2, 2],
            "serial_full_domain": True,
        },
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "live_words": live_words,
            "boundary_face_zero_words": 2 * ni * nj,
        },
        "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
        "kernel": {
            "abi": CUDA_TERRAIN_DEFORMATION_ABI,
            "names": list(CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES),
            "source_sha256": CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "grid_x": list(grids),
            "block_x": CUDA_TERRAIN_DEFORMATION_THREADS,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaTerrainDeformationLaunch(event, compiled, scratch, receipt)


__all__ = [
    "ALL_ARRAY_NAMES",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_TERRAIN_DEFORMATION_ABI",
    "CUDA_TERRAIN_DEFORMATION_CLAIM_LIMIT",
    "CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES",
    "CUDA_TERRAIN_DEFORMATION_SCHEMA",
    "CUDA_TERRAIN_DEFORMATION_SOURCE",
    "CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256",
    "CompiledTerrainDeformationKernels",
    "CudaTerrainDeformationLaunch",
    "DEFAULT_TERRAIN_DEFORMATION_SELECTION",
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "SOURCE_ANCHORS",
    "TerrainDeformationSelection",
    "compile_cuda_terrain_deformation",
    "cpu_terrain_deformation_inplace",
    "launch_cuda_terrain_deformation",
    "terrain_deformation_scratch_shapes",
    "terrain_deformation_shapes",
]
