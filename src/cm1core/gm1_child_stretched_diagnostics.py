"""Source-ordered diagnostics for the admitted stretched-terrain GM1 child.

The legacy storm-diagnostic helper reconstructs a uniform vertical grid from a
scalar ``dz``.  That is invalid for the 75-level child.  This module evaluates
the terrain branch of CM1 r21.1 ``calcvort`` with the sealed sigma/sigmaf and
rds/rdsf profiles, and mass-centres the accepted w-face carrier.  Reflectivity
remains an independently supplied public WSM6 diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from typing import Any, Mapping

from cm1core.cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_PROFILE_SHA256,
)
from cm1core.stretch_z1_child_15km import build_stretch_z1_child_15km_grid


GM1_CHILD_STRETCHED_DIAGNOSTICS_SCHEMA = (
    "gm1.real-50m-child-stretched-diagnostics/v1"
)
GM1_CHILD_STRETCHED_DIAGNOSTICS_CLAIM_LIMIT = (
    "CM1 r21.1 terrain-coordinate interior vertical vorticity at scalar "
    "points and mass-centred vertical velocity for the exact admitted "
    "stretch_z=1 child; the outermost scalar row/column is nearest-interior "
    "display closure because the render history intentionally carries no "
    "diagnostic metric halos. Reflectivity is pass-through only. This is not "
    "a tornado detector, classification, forecast-skill, or model-validation claim"
)
SOURCE_ANCHORS = {
    "terrain_vertical_vorticity": "src/misclibs.F:2035-2107",
    "mass_centered_w": "src/misclibs.F:1579-1587",
    "vertical_grid": "src/param.F:7814-7953,8162-8225",
}

CM1_CGS1_F32 = 1.875
CM1_CGS2_F32 = -1.25
CM1_CGS3_F32 = 0.375


def _a(np: Any, left: Any, right: Any) -> Any:
    return np.add(left, right, dtype=np.float32)


def _s(np: Any, left: Any, right: Any) -> Any:
    return np.subtract(left, right, dtype=np.float32)


def _m(np: Any, left: Any, right: Any) -> Any:
    return np.multiply(left, right, dtype=np.float32)


def _d(np: Any, left: Any, right: Any) -> Any:
    return np.divide(left, right, dtype=np.float32)


def _profile_sha256(np: Any, value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32)
        .astype("<f4", copy=False)
        .tobytes(order="C")
    ).hexdigest()


def _sealed_profiles(np: Any, nk: int) -> tuple[Any, Any, Any, Any]:
    if nk != 75:
        raise ValueError("the admitted stretched child diagnostic requires nk=75")
    grid = build_stretch_z1_child_15km_grid()
    if grid.sha256 != GM1_CHILD_STRETCHED_GRID_SHA256:
        raise RuntimeError("stretched diagnostic grid differs from its static pin")
    sigma = np.ascontiguousarray(grid.sigma, dtype=np.float32)
    sigmaf = np.ascontiguousarray(grid.sigmaf, dtype=np.float32)
    rds = np.ones(nk + 2, dtype=np.float32)
    rdsf = np.ones(nk + 3, dtype=np.float32)
    for k in range(2, nk + 1):
        rds[k] = _d(np, np.float32(1.0), _s(np, sigma[k], sigma[k - 1]))
    for k in range(1, nk + 1):
        rdsf[k] = _d(
            np, np.float32(1.0), _s(np, sigmaf[k + 1], sigmaf[k])
        )
    for name, value in (
        ("sigma", sigma),
        ("sigmaf", sigmaf),
        ("rds", rds),
        ("rdsf", rdsf),
    ):
        if _profile_sha256(np, value) != GM1_CHILD_STRETCHED_PROFILE_SHA256[name]:
            raise RuntimeError(f"stretched diagnostic {name} differs from its pin")
    return sigma, sigmaf, rds, rdsf


def _require_array(
    np: Any, arrays: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> Any:
    if name not in arrays:
        raise ValueError(f"missing stretched diagnostic carrier {name}")
    value = arrays[name]
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("float32")
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"stretched diagnostic {name} must be C-f32 with shape {shape}"
        )
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"stretched diagnostic {name} contains non-finite words")
    return value


def _face_interpolate(np: Any, values: Any, rds: Any) -> Any:
    nk = values.shape[0]
    result = np.empty((nk + 1, *values.shape[1:]), dtype=np.float32)
    value = _m(np, np.float32(CM1_CGS1_F32), values[0])
    value = _a(np, value, _m(np, np.float32(CM1_CGS2_F32), values[1]))
    result[0] = _a(np, value, _m(np, np.float32(CM1_CGS3_F32), values[2]))
    value = _m(np, np.float32(CM1_CGS1_F32), values[-1])
    value = _a(np, value, _m(np, np.float32(CM1_CGS2_F32), values[-2]))
    result[-1] = _a(np, value, _m(np, np.float32(CM1_CGS3_F32), values[-3]))
    grid = build_stretch_z1_child_15km_grid()
    sigma = grid.sigma
    sigmaf = grid.sigmaf
    for face in range(1, nk):
        direct_k = face + 1
        r2 = _m(
            np,
            _s(np, sigmaf[direct_k], sigma[direct_k - 1]),
            rds[direct_k],
        )
        r1 = _s(np, np.float32(1.0), r2)
        result[face] = _a(
            np,
            _m(np, r1, values[face - 1]),
            _m(np, r2, values[face]),
        )
    return result


@dataclass(frozen=True, slots=True)
class Gm1ChildStretchedDiagnostics:
    reflectivity_dbz: Any
    updraft_ms: Any
    vertical_vorticity_s: Any
    exact_interior_words: int
    display_closed_boundary_words: int
    receipt: Mapping[str, Any]


def evaluate_gm1_child_stretched_diagnostics(
    arrays: Mapping[str, Any],
    *,
    ni: int = 300,
    nj: int = 300,
    nk: int = 75,
    dx: float = 50.0,
    dy: float = 50.0,
) -> Gm1ChildStretchedDiagnostics:
    """Evaluate the exact interior diagnostic and explicit display closure."""

    if (ni, nj, nk) != (300, 300, 75) or (float(dx), float(dy)) != (50.0, 50.0):
        raise ValueError(
            "this evidence path is sealed to the admitted 300x300x75 50 m child"
        )
    np = importlib.import_module("numpy")
    u = _require_array(np, arrays, "u3d", (nk, nj, ni + 1))
    v = _require_array(np, arrays, "v3d", (nk, nj + 1, ni))
    w = _require_array(np, arrays, "w3d", (nk + 1, nj, ni))
    terrain = _require_array(np, arrays, "terrain_msl", (nj, ni))
    rgzu = _require_array(np, arrays, "rgzu", (nj, ni + 1))
    rgzv = _require_array(np, arrays, "rgzv", (nj + 1, ni))
    reflectivity = _require_array(
        np, arrays, "reflectivity_dbz", (nk, nj, ni)
    ).copy(order="C")
    if len({int(x.ctypes.data) for x in (u, v, w, terrain, rgzu, rgzv, reflectivity)}) != 7:
        raise ValueError("stretched diagnostic carriers must not alias")

    sigma, sigmaf, rds, rdsf = _sealed_profiles(np, nk)
    del sigma
    uw = _face_interpolate(np, u, rds)
    vw = _face_interpolate(np, v, rds)
    updraft = _m(np, np.float32(0.5), _a(np, w[:-1], w[1:]))
    vorticity = np.empty((nk, nj, ni), dtype=np.float32)

    zt = np.float32(15000.0)
    rdx = _d(np, np.float32(1.0), np.float32(dx))
    rdy = _d(np, np.float32(1.0), np.float32(dy))
    terrain_sum = _a(np, terrain[:-1, :-1], terrain[1:, 1:])
    terrain_sum = _a(np, terrain_sum, terrain[1:, :-1])
    terrain_sum = _a(np, terrain_sum, terrain[:-1, 1:])
    transform = _d(
        np, zt, _s(np, zt, _m(np, np.float32(0.25), terrain_sum))
    )
    rgzv_delta = _s(np, rgzv[1:ni, 1:ni], rgzv[1:ni, : ni - 1])
    rgzu_delta = _s(np, rgzu[1:nj, 1:ni], rgzu[: nj - 1, 1:ni])

    for k in range(nk):
        lower_height = _s(np, zt, sigmaf[k + 1])
        upper_height = _s(np, zt, sigmaf[k + 2])
        metric = rdsf[k + 1]

        v_gradient = _s(
            np,
            _m(np, v[k, 1:nj, 1:ni], rgzv[1:nj, 1:ni]),
            _m(np, v[k, 1:nj, : ni - 1], rgzv[1:nj, : ni - 1]),
        )
        v_gradient = _m(np, v_gradient, rdx)
        v_upper = _a(
            np, vw[k + 1, 1:nj, : ni - 1], vw[k + 1, 1:nj, 1:ni]
        )
        v_lower = _a(np, vw[k, 1:nj, : ni - 1], vw[k, 1:nj, 1:ni])
        v_vertical = _m(
            np,
            np.float32(0.5),
            _s(
                np,
                _m(np, upper_height, v_upper),
                _m(np, lower_height, v_lower),
            ),
        )
        v_vertical = _m(np, v_vertical, metric)
        v_vertical = _m(np, v_vertical, rgzv_delta)
        v_vertical = _m(np, v_vertical, rdx)
        v_term = _m(np, transform, _a(np, v_gradient, v_vertical))

        u_gradient = _s(
            np,
            _m(np, u[k, 1:nj, 1:ni], rgzu[1:nj, 1:ni]),
            _m(np, u[k, : nj - 1, 1:ni], rgzu[: nj - 1, 1:ni]),
        )
        u_gradient = _m(np, u_gradient, rdy)
        u_upper = _a(
            np, uw[k + 1, : nj - 1, 1:ni], uw[k + 1, 1:nj, 1:ni]
        )
        u_lower = _a(np, uw[k, : nj - 1, 1:ni], uw[k, 1:nj, 1:ni])
        u_vertical = _m(
            np,
            np.float32(0.5),
            _s(
                np,
                _m(np, upper_height, u_upper),
                _m(np, lower_height, u_lower),
            ),
        )
        u_vertical = _m(np, u_vertical, metric)
        u_vertical = _m(np, u_vertical, rgzu_delta)
        u_vertical = _m(np, u_vertical, rdy)
        u_term = _m(np, transform, _a(np, u_gradient, u_vertical))
        corner = _s(np, v_term, u_term)
        value = _a(np, corner[: nj - 2, : ni - 2], corner[: nj - 2, 1:])
        value = _a(np, value, corner[1:, : ni - 2])
        value = _a(np, value, corner[1:, 1:])
        vorticity[k, 1 : nj - 1, 1 : ni - 1] = _m(
            np, np.float32(0.25), value
        )
        vorticity[k, 0] = vorticity[k, 1]
        vorticity[k, -1] = vorticity[k, -2]
        vorticity[k, :, 0] = vorticity[k, :, 1]
        vorticity[k, :, -1] = vorticity[k, :, -2]

    exact = nk * (ni - 2) * (nj - 2)
    words = nk * ni * nj
    receipt = {
        "schema": GM1_CHILD_STRETCHED_DIAGNOSTICS_SCHEMA,
        "claim_limit": GM1_CHILD_STRETCHED_DIAGNOSTICS_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "vertical": {
            "grid_sha256": GM1_CHILD_STRETCHED_GRID_SHA256,
            "profile_sha256": dict(GM1_CHILD_STRETCHED_PROFILE_SHA256),
            "profile_indexing": "direct CM1 indices",
        },
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "dx_m": float(dx),
            "dy_m": float(dy),
            "ztop_m": 15000.0,
            "exact_interior_words": exact,
            "display_closed_boundary_words": words - exact,
        },
        "source_order": [
            "mass u/v to w faces with sealed sigma/sigmaf/rds",
            "terrain-coordinate corner z-vorticity with sealed rdsf",
            "four-corner scalar average",
            "nearest-interior outer display closure",
            "mass-centred w=0.5*(w(k)+w(k+1))",
            "pass-through public WSM6 reflectivity",
        ],
    }
    return Gm1ChildStretchedDiagnostics(
        reflectivity, updraft, vorticity, exact, words - exact, receipt
    )


__all__ = (
    "GM1_CHILD_STRETCHED_DIAGNOSTICS_CLAIM_LIMIT",
    "GM1_CHILD_STRETCHED_DIAGNOSTICS_SCHEMA",
    "Gm1ChildStretchedDiagnostics",
    "SOURCE_ANCHORS",
    "evaluate_gm1_child_stretched_diagnostics",
)
