"""CM1 r21.1 fifth-order terrain/open momentum advection on CUDA.

This module is the selected-profile ``solve2`` slice immediately before the
``psolver=3`` acoustic call.  It forms terrain-coordinate mass fluxes and
``divx`` and then adds the exact ``advu``, ``advv``, and ``advw`` transport
terms to caller-owned held tendency arrays.  Velocity WENO is disabled by the
admitted profile, so every RK stage uses the fifth-order centered/upwind-
biased flux with CM1's lower-order vertical closures.

The production path operates only on the existing padded device state.  The
CPU path is an independent statement-ordered reference for bounded fixtures.

The three kernels take CM1's lateral family as an ``obc`` word
(:func:`momentum_open_boundary_code`).  ``param.F:6678-6681`` zeroes ``ibw``
on a periodic side, so advu advances faces ``i1..i2`` (adv.F:968-978) and advv
rows ``j1..j2`` (adv.F:1913-1923) including that side's boundary face, and
reads the mass fluxes, ``divx`` and ``rho0`` across it at the wrapped index,
which are the words solve2.F forms in its halo from the periodic wind copies.
The inflow edits -- the boundary-face flux copy and the divx-piece
subtraction -- run on open sides only (``doitw``..``doitn``).  With the
all-open word ``0x2222`` every kernel is the pre-family text word for word, the
schedule every earlier receipt was measured with.  Those edits keep that
text's grouping, ``(q*0.5)*(c*(rdy*vh))``, where adv.F:1145-1172 writes
``(((q*0.5)*c)*rdy)*vh``: the flat stretched image regroups them, this text
and the terrain image still differ from CM1's compiled advu by rounding in the
open-edge inflow cells.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


CUDA_MOMENTUM_ADVECTION5_SCHEMA = "gpuwm-cm1.cuda-momentum-advection5/v1"
CUDA_MOMENTUM_ADVECTION5_ABI = "gpuwm-cm1.cuda-momentum-advection5-kernels/v1"
CUDA_MOMENTUM_ADVECTION5_THREADS = 128
CM1_SMEPS_F32 = 1.0e-20
CM1_ADV_F_SHA256 = "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
CM1_ADV_ROUTINES_F_SHA256 = (
    "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00"
)
CM1_SOLVE2_F_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
SOURCE_ANCHORS = {
    "mass_flux_and_divergence": "src/solve2.F:490-627",
    "momentum_calls": "src/solve2.F:883-892",
    "advu": "src/adv.F:911-1839",
    "advv": "src/adv.F:1847-2770",
    "advw": "src/adv.F:2778-3610",
    "hadv_flx5": "src/adv_routines.F:1335-1476",
    "vadv_flx5": "src/adv_routines.F:3269-3388",
    "vertical_lower_23": "src/adv_routines.F:4817-4995",
    "vertical_lower_35": "src/adv_routines.F:5001-5169",
    "flx3_flx5": "src/adv_routines.F:5989-6019",
}
CUDA_MOMENTUM_ADVECTION5_CLAIM_LIMIT = (
    "CM1 r21.1 serial uniform-horizontal and uniform-sigma Cartesian "
    "terrain-coordinate mass flux/divx for any serial lateral family "
    "(open, periodic, rigid: advu's i1..i2, advv's j1..j2 and the doitw..doitn "
    "inflow edits) plus selected hadvordrv=vadvordrv=5 and advwenov=0 "
    "advu/advv/advw tendency increments on padded physical-domain state, and "
    "CM1's own solve2.F:631-720 seam between divx and advu for a caller's "
    "post_divergence_hook; "
    "no diffusion, WENO velocity branch, immersed boundary, axisymmetry, "
    "subsidence, buoyancy, Coriolis arithmetic of its own, pressure gradient, "
    "acoustic evolution, "
    "physics, coupled forecast, tornado, or validation claim"
)


@dataclass(frozen=True, slots=True)
class MomentumAdvection5Selection:
    hadvordrv: int = 5
    vadvordrv: int = 5
    advwenov: int = 0
    axisymmetric: int = 0
    terrain: bool = True
    idiff: int = 0
    immersed_boundary: bool = False
    uniform_horizontal_grid: bool = True
    uniform_vertical_sigma: bool = True
    lateral_boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)


DEFAULT_MOMENTUM_ADVECTION5_SELECTION = MomentumAdvection5Selection()

#: CM1's ``(wbc, ebc, sbc, nbc)`` for four open-radiative sides: the family
#: every receipt before the lateral-family fix was measured with, and the one
#: the sealed tilestream and nest-child lanes run.
MOMENTUM_ALL_OPEN_SIDES = (2, 2, 2, 2)


def momentum_lateral_sides(lateral_sides: Any) -> tuple[int, int, int, int]:
    """CM1's ``(wbc, ebc, sbc, nbc)``, checked as ``param.F`` checks them.

    ``param.F:2562-2588`` stops CM1 when only one side of a direction is
    periodic ("Can not have periodic b.c.'s on one side only!"): the domain
    would wrap one way and not the other, and advu's face range and the
    wrapped halo reads below have no meaning there.
    """

    sides = tuple(lateral_sides)
    if len(sides) != 4 or any(
        isinstance(v, bool) or not isinstance(v, int) or v not in (1, 2, 3, 4) for v in sides
    ):
        raise ValueError("lateral_sides must be CM1's (wbc, ebc, sbc, nbc), each 1, 2, 3 or 4")
    if (sides[0] == 1) != (sides[1] == 1) or (sides[2] == 1) != (sides[3] == 1):
        raise ValueError(
            f"lateral_sides {sides}: param.F:2562-2588 stops CM1 when only one "
            "side of a direction is periodic"
        )
    return sides  # type: ignore[return-value]


def momentum_open_boundary_code(lateral_sides: Any) -> int:
    """Pack CM1's ``(wbc, ebc, sbc, nbc)`` into advu/advv/advw's ``obc`` word.

    The kernels read it exactly as adv.F reads the family: ``param.F:6678-6681``
    zeroes ``ibw``..``ibn`` on a periodic side, so advu advances faces
    ``i1..i2`` with ``i1=1``/``i2=ni+1`` there (advv likewise ``j1..j2``) and
    reads the mass fluxes, ``divx`` and ``rho0`` across that side at the
    wrapped index; ``doitw``..``doitn`` (adv.F:983-998, 1905-1932, 2838-2850)
    run the inflow edits on an open side (value 2) only.
    """

    w, e, s, n = momentum_lateral_sides(lateral_sides)
    return w | (e << 4) | (s << 8) | (n << 12)


def momentum_advection5_counts(
    ni: int, nj: int, nk: int, lateral_sides: Any = MOMENTUM_ALL_OPEN_SIDES
) -> tuple[int, int, int]:
    """Words advu, advv and advw visit: faces ``i1..i2``, rows ``j1..j2``, every w cell.

    The kernels map a thread to face ``1-pw .. ni-1+pe`` (advu) and row
    ``1-ps .. nj-1+pn`` (advv), so the launch must cover exactly these counts.
    """

    w, e, s, n = momentum_lateral_sides(lateral_sides)
    faces = ni - 1 + int(w == 1) + int(e == 1)
    rows = nj - 1 + int(s == 1) + int(n == 1)
    return nk * nj * faces, nk * rows * ni, (nk - 1) * nj * ni


def momentum_boundary_faces(lateral_sides: Any) -> dict[str, Any]:
    """The receipt words that say which faces advu/advv advanced and why."""

    w, e, s, n = momentum_lateral_sides(lateral_sides)
    return {
        "advu_faces_one_based": [1 if w == 1 else 2, "ni+1" if e == 1 else "ni"],
        "advv_rows_one_based": [1 if s == 1 else 2, "nj+1" if n == 1 else "nj"],
        "inflow_edits_on": [name for name, side in zip(("west", "east", "south", "north"), (w, e, s, n)) if side == 2],
        "authority": "adv.F advu i1/i2 :968-978 and doitw..doitn :983-998, advv j1/j2 :1913-1923, advw doitw..doitn :2838-2850; param.F:6678-6681",
    }


@dataclass(frozen=True, slots=True)
class _LateralFamily:
    """The CPU reference's view of the family: wrapped reads and open sides."""

    ni: int
    nj: int
    sides: tuple[int, int, int, int]

    def periodic(self, side: int) -> bool:
        return self.sides[side] == 1

    def open(self, side: int) -> bool:
        return self.sides[side] == 2

    def cx(self, i: int) -> int:
        if self.periodic(0) and i < 0:
            return i + self.ni
        if self.periodic(1) and i >= self.ni:
            return i - self.ni
        return i

    def fx(self, i: int) -> int:
        if self.periodic(0) and i < 0:
            return i + self.ni
        if self.periodic(1) and i > self.ni:
            return i - self.ni
        return i

    def cy(self, j: int) -> int:
        if self.periodic(2) and j < 0:
            return j + self.nj
        if self.periodic(3) and j >= self.nj:
            return j - self.nj
        return j

    def fy(self, j: int) -> int:
        if self.periodic(2) and j < 0:
            return j + self.nj
        if self.periodic(3) and j > self.nj:
            return j - self.nj
        return j


def validate_momentum_advection5_selection(
    selection: MomentumAdvection5Selection,
) -> None:
    if not isinstance(selection, MomentumAdvection5Selection):
        raise TypeError("selection must be MomentumAdvection5Selection")
    expected = DEFAULT_MOMENTUM_ADVECTION5_SELECTION
    if selection != expected:
        for name in (
            "hadvordrv",
            "vadvordrv",
            "advwenov",
            "axisymmetric",
            "terrain",
            "idiff",
            "immersed_boundary",
            "uniform_horizontal_grid",
            "uniform_vertical_sigma",
            "lateral_boundaries",
        ):
            if getattr(selection, name) != getattr(expected, name):
                raise ValueError(
                    f"unsupported momentum-advection selection {name}="
                    f"{getattr(selection, name)!r}; required {getattr(expected, name)!r}"
                )


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _flx3(np: Any, s1: Any, s2: Any, s3: Any) -> Any:
    value = _mul(np, np.float32(-1.0 / 6.0), s1)
    value = _add(np, value, _mul(np, np.float32(5.0 / 6.0), s2))
    return _add(np, value, _mul(np, np.float32(2.0 / 6.0), s3))


def _flx5(np: Any, values: tuple[Any, Any, Any, Any, Any]) -> Any:
    value = _mul(np, np.float32(2.0 / 60.0), values[0])
    value = _add(np, value, _mul(np, np.float32(-13.0 / 60.0), values[1]))
    value = _add(np, value, _mul(np, np.float32(47.0 / 60.0), values[2]))
    value = _add(np, value, _mul(np, np.float32(27.0 / 60.0), values[3]))
    return _add(np, value, _mul(np, np.float32(-3.0 / 60.0), values[4]))


def _positive_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _spacing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _require_host(
    np: Any,
    value: Any,
    name: str,
    shape: tuple[int, ...],
    *,
    writable: bool,
) -> Any:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must be a float32 numpy.ndarray")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


def _validate_host_inputs(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Any]:
    np = importlib.import_module("numpy")
    ng = CM1_OPEN_NGXY
    shapes = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w3d": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "wten": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "uh": (ni + 2 * ng,),
        "vh": (nj + 2 * ng,),
    }
    writable = {"uten", "vten", "wten", "rru", "rrv", "rrw"}
    a = {
        name: _require_host(
            np, arrays[name], name, shape, writable=name in writable
        )
        for name, shape in shapes.items()
    }
    metric_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    names = (
        "rho0_3d",
        "rf0",
        "rr0",
        "rrf0",
        "c1",
        "c2",
        "gz",
        "gzu",
        "gzv",
        "rgzu",
        "rgzv",
        "dzdx",
        "dzdy",
    )
    m = {
        name: _require_host(
            np, metrics[name], name, metric_shapes[name], writable=False
        )
        for name in names
    }
    divx = _require_host(np, divx, "divx", (nk, nj, ni), writable=True)
    pointers = [int(item.ctypes.data) for item in (*a.values(), *m.values(), divx)]
    if len(pointers) != len(set(pointers)):
        raise ValueError("momentum-advection arrays and metrics must not alias")
    return np, a, m, divx


def _metric_at(value: Any, kface: int, j: int, i: int, *, ni: int, nj: int) -> Any:
    return value[kface - 1, min(max(j, 0), nj - 1), min(max(i, 0), ni - 1)]


def _u_fx(
    np: Any, a: Mapping[str, Any], k: int, j: int, face: int, ng: int, lat: _LateralFamily
) -> Any:
    y = j + ng
    x = face + ng
    mass = _mul(
        np,
        np.float32(0.5),
        _add(np, a["rru"][k, y, lat.fx(face) + ng], a["rru"][k, y, lat.fx(face + 1) + ng]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["u3d"][k, y, x + offset] for offset in (-2, -1, 0, 1, 2))
    else:
        values = tuple(a["u3d"][k, y, x + offset] for offset in (3, 2, 1, 0, -1))
    return _mul(np, mass, _flx5(np, values))


def _u_fy_base(
    np: Any, a: Mapping[str, Any], k: int, face: int, i_face: int, ng: int, lat: _LateralFamily
) -> Any:
    y = face + ng
    x = i_face + ng
    mass = _mul(
        np,
        np.float32(0.5),
        _add(np, a["rrv"][k, y, lat.cx(i_face) + ng], a["rrv"][k, y, lat.cx(i_face - 1) + ng]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["u3d"][k, y + offset, x] for offset in (-3, -2, -1, 0, 1))
    else:
        values = tuple(a["u3d"][k, y + offset, x] for offset in (2, 1, 0, -1, -2))
    return _mul(np, mass, _flx5(np, values))


def _u_fy(
    np: Any,
    a: Mapping[str, Any],
    k: int,
    face: int,
    i_face: int,
    ng: int,
    nj: int,
    lat: _LateralFamily,
) -> Any:
    y = face + ng
    mass_twice = _add(np, a["rrv"][k, y, lat.cx(i_face) + ng], a["rrv"][k, y, lat.cx(i_face - 1) + ng])
    if face == 0 and lat.open(2) and mass_twice >= np.float32(0.0):
        return _u_fy_base(np, a, k, 1, i_face, ng, lat)
    if face == nj and lat.open(3) and mass_twice <= np.float32(0.0):
        return _u_fy_base(np, a, k, nj - 1, i_face, ng, lat)
    return _u_fy_base(np, a, k, face, i_face, ng, lat)


def _v_fx_base(
    np: Any, a: Mapping[str, Any], k: int, j_face: int, face: int, ng: int, lat: _LateralFamily
) -> Any:
    y = j_face + ng
    x = face + ng
    mass = _mul(
        np,
        np.float32(0.5),
        _add(np, a["rru"][k, lat.cy(j_face) + ng, x], a["rru"][k, lat.cy(j_face - 1) + ng, x]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["v3d"][k, y, x + offset] for offset in (-3, -2, -1, 0, 1))
    else:
        values = tuple(a["v3d"][k, y, x + offset] for offset in (2, 1, 0, -1, -2))
    return _mul(np, mass, _flx5(np, values))


def _v_fx(
    np: Any,
    a: Mapping[str, Any],
    k: int,
    j_face: int,
    face: int,
    ng: int,
    ni: int,
    lat: _LateralFamily,
) -> Any:
    x = face + ng
    mass_twice = _add(np, a["rru"][k, lat.cy(j_face) + ng, x], a["rru"][k, lat.cy(j_face - 1) + ng, x])
    if face == 0 and lat.open(0) and mass_twice >= np.float32(0.0):
        return _v_fx_base(np, a, k, j_face, 1, ng, lat)
    if face == ni and lat.open(1) and mass_twice <= np.float32(0.0):
        return _v_fx_base(np, a, k, j_face, ni - 1, ng, lat)
    return _v_fx_base(np, a, k, j_face, face, ng, lat)


def _v_fy(
    np: Any, a: Mapping[str, Any], k: int, face: int, i: int, ng: int, lat: _LateralFamily
) -> Any:
    y = face + ng
    x = i + ng
    mass = _mul(
        np,
        np.float32(0.5),
        _add(np, a["rrv"][k, lat.fy(face) + ng, x], a["rrv"][k, lat.fy(face + 1) + ng, x]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["v3d"][k, y + offset, x] for offset in (-2, -1, 0, 1, 2))
    else:
        values = tuple(a["v3d"][k, y + offset, x] for offset in (3, 2, 1, 0, -1))
    return _mul(np, mass, _flx5(np, values))


def _uv_vertical_flux(
    np: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    *,
    field: str,
    kface: int,
    j: int,
    i: int,
    ng: int,
    ni: int,
    nj: int,
    nk: int,
    lat: _LateralFamily,
) -> Any:
    if field == "u3d":
        y, x = j + ng, i + ng
        mass = _mul(
            np,
            np.float32(0.5),
            _add(np, a["rrw"][kface, y, lat.cx(i) + ng], a["rrw"][kface, y, lat.cx(i - 1) + ng]),
        )
        c2 = _mul(
            np,
            np.float32(0.5),
            _add(
                np,
                _metric_at(m["c2"], kface, j, i - 1, ni=ni, nj=nj),
                _metric_at(m["c2"], kface, j, i, ni=ni, nj=nj),
            ),
        )
    else:
        y, x = j + ng, i + ng
        mass = _mul(
            np,
            np.float32(0.5),
            _add(np, a["rrw"][kface, lat.cy(j) + ng, x], a["rrw"][kface, lat.cy(j - 1) + ng, x]),
        )
        c2 = _mul(
            np,
            np.float32(0.5),
            _add(
                np,
                _metric_at(m["c2"], kface, j - 1, i, ni=ni, nj=nj),
                _metric_at(m["c2"], kface, j, i, ni=ni, nj=nj),
            ),
        )
    c1 = _sub(np, np.float32(1.0), c2)
    q = a[field]
    if kface == 2:
        reconstruction = (
            _add(np, _mul(np, c1, q[1, y, x]), _mul(np, c2, q[2, y, x]))
            if mass >= np.float32(0.0)
            else _flx3(np, q[3, y, x], q[2, y, x], q[1, y, x])
        )
    elif kface == nk:
        reconstruction = (
            _flx3(np, q[nk - 2, y, x], q[nk - 1, y, x], q[nk, y, x])
            if mass >= np.float32(0.0)
            else _add(
                np, _mul(np, c1, q[nk - 1, y, x]), _mul(np, c2, q[nk, y, x])
            )
        )
    elif kface == 3:
        reconstruction = (
            _flx3(np, q[1, y, x], q[2, y, x], q[3, y, x])
            if mass >= np.float32(0.0)
            else _flx5(
                np,
                (
                    q[5, y, x],
                    q[4, y, x],
                    q[3, y, x],
                    q[2, y, x],
                    q[1, y, x],
                ),
            )
        )
    elif kface == nk - 1:
        reconstruction = (
            _flx5(
                np,
                tuple(q[level, y, x] for level in range(kface - 3, kface + 2)),
            )
            if mass >= np.float32(0.0)
            else _flx3(
                np, q[kface + 1, y, x], q[kface, y, x], q[kface - 1, y, x]
            )
        )
    elif mass >= np.float32(0.0):
        reconstruction = _flx5(
            np,
            tuple(q[level, y, x] for level in range(kface - 3, kface + 2)),
        )
    else:
        reconstruction = _flx5(
            np,
            tuple(q[level, y, x] for level in range(kface + 2, kface - 3, -1)),
        )
    return _mul(np, mass, reconstruction)


def _w_fx_base(
    np: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    kface: int,
    j: int,
    face: int,
    ng: int,
    ni: int,
    nj: int,
) -> Any:
    y, x = j + ng, face + ng
    c2 = _mul(
        np,
        np.float32(0.5),
        _add(
            np,
            _metric_at(m["c2"], kface, j, face - 1, ni=ni, nj=nj),
            _metric_at(m["c2"], kface, j, face, ni=ni, nj=nj),
        ),
    )
    c1 = _sub(np, np.float32(1.0), c2)
    mass = _add(
        np,
        _mul(np, c2, a["rru"][kface, y, x]),
        _mul(np, c1, a["rru"][kface - 1, y, x]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["w3d"][kface, y, x + offset] for offset in (-3, -2, -1, 0, 1))
    else:
        values = tuple(a["w3d"][kface, y, x + offset] for offset in (2, 1, 0, -1, -2))
    return _mul(np, mass, _flx5(np, values))


def _w_fx(
    np: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    kface: int,
    j: int,
    face: int,
    ng: int,
    ni: int,
    nj: int,
    lat: _LateralFamily,
) -> Any:
    y, x = j + ng, face + ng
    c2 = _mul(
        np,
        np.float32(0.5),
        _add(
            np,
            _metric_at(m["c2"], kface, j, face - 1, ni=ni, nj=nj),
            _metric_at(m["c2"], kface, j, face, ni=ni, nj=nj),
        ),
    )
    c1 = _sub(np, np.float32(1.0), c2)
    mass = _add(
        np,
        _mul(np, c2, a["rru"][kface, y, x]),
        _mul(np, c1, a["rru"][kface - 1, y, x]),
    )
    if face == 0 and lat.open(0) and mass >= np.float32(0.0):
        return _w_fx_base(np, a, m, kface, j, 1, ng, ni, nj)
    if face == ni and lat.open(1) and mass <= np.float32(0.0):
        return _w_fx_base(np, a, m, kface, j, ni - 1, ng, ni, nj)
    return _w_fx_base(np, a, m, kface, j, face, ng, ni, nj)


def _w_fy_base(
    np: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    kface: int,
    face: int,
    i: int,
    ng: int,
    ni: int,
    nj: int,
) -> Any:
    y, x = face + ng, i + ng
    c2 = _mul(
        np,
        np.float32(0.5),
        _add(
            np,
            _metric_at(m["c2"], kface, face - 1, i, ni=ni, nj=nj),
            _metric_at(m["c2"], kface, face, i, ni=ni, nj=nj),
        ),
    )
    c1 = _sub(np, np.float32(1.0), c2)
    mass = _add(
        np,
        _mul(np, c2, a["rrv"][kface, y, x]),
        _mul(np, c1, a["rrv"][kface - 1, y, x]),
    )
    if mass >= np.float32(0.0):
        values = tuple(a["w3d"][kface, y + offset, x] for offset in (-3, -2, -1, 0, 1))
    else:
        values = tuple(a["w3d"][kface, y + offset, x] for offset in (2, 1, 0, -1, -2))
    return _mul(np, mass, _flx5(np, values))


def _w_fy(
    np: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    kface: int,
    face: int,
    i: int,
    ng: int,
    ni: int,
    nj: int,
    lat: _LateralFamily,
) -> Any:
    y, x = face + ng, i + ng
    c2 = _mul(
        np,
        np.float32(0.5),
        _add(
            np,
            _metric_at(m["c2"], kface, face - 1, i, ni=ni, nj=nj),
            _metric_at(m["c2"], kface, face, i, ni=ni, nj=nj),
        ),
    )
    c1 = _sub(np, np.float32(1.0), c2)
    mass = _add(
        np,
        _mul(np, c2, a["rrv"][kface, y, x]),
        _mul(np, c1, a["rrv"][kface - 1, y, x]),
    )
    if face == 0 and lat.open(2) and mass >= np.float32(0.0):
        return _w_fy_base(np, a, m, kface, 1, i, ng, ni, nj)
    if face == nj and lat.open(3) and mass <= np.float32(0.0):
        return _w_fy_base(np, a, m, kface, nj - 1, i, ng, ni, nj)
    return _w_fy_base(np, a, m, kface, face, i, ng, ni, nj)


def _w_vertical_flux(
    np: Any, a: Mapping[str, Any], k: int, j: int, i: int, ng: int, nk: int
) -> Any:
    y, x = j + ng, i + ng
    mass = _mul(
        np,
        np.float32(0.5),
        _add(np, a["rrw"][k, y, x], a["rrw"][k + 1, y, x]),
    )
    q = a["w3d"]
    if k == 1:
        reconstruction = (
            _mul(np, np.float32(0.5), _add(np, q[1, y, x], q[2, y, x]))
            if mass >= np.float32(0.0)
            else _flx3(np, q[3, y, x], q[2, y, x], q[1, y, x])
        )
    elif k == 2:
        reconstruction = (
            _flx3(np, q[1, y, x], q[2, y, x], q[3, y, x])
            if mass >= np.float32(0.0)
            else _flx5(np, (q[5, y, x], q[4, y, x], q[3, y, x], q[2, y, x], q[1, y, x]))
        )
    elif k == nk - 1:
        reconstruction = (
            _flx5(np, tuple(q[level, y, x] for level in range(k - 2, k + 3)))
            if mass >= np.float32(0.0)
            else _flx3(np, q[k + 2, y, x], q[k + 1, y, x], q[k, y, x])
        )
    elif k == nk:
        reconstruction = (
            _flx3(np, q[k - 1, y, x], q[k, y, x], q[k + 1, y, x])
            if mass >= np.float32(0.0)
            else _mul(np, np.float32(0.5), _add(np, q[k, y, x], q[k + 1, y, x]))
        )
    elif mass >= np.float32(0.0):
        reconstruction = _flx5(
            np, tuple(q[level, y, x] for level in range(k - 2, k + 3))
        )
    else:
        reconstruction = _flx5(
            np, tuple(q[level, y, x] for level in range(k + 3, k - 2, -1))
        )
    return _mul(np, mass, reconstruction)


def cpu_terrain_open_momentum_advection5_inplace(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: MomentumAdvection5Selection = DEFAULT_MOMENTUM_ADVECTION5_SELECTION,
    lateral_sides: tuple[int, int, int, int] = MOMENTUM_ALL_OPEN_SIDES,
) -> None:
    """Apply the selected terrain/open mass-flux and momentum slice on CPU.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)``; see
    :func:`momentum_open_boundary_code` for what each side changes.
    """

    validate_momentum_advection5_selection(selection)
    lateral_sides = momentum_lateral_sides(lateral_sides)
    ni = _positive_int("ni", ni, 7)
    nj = _positive_int("nj", nj, 7)
    nk = _positive_int("nk", nk, 8)
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    dx, dy, dz = (_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz)))
    np, a, m, divx = _validate_host_inputs(arrays, metrics, divx, ni=ni, nj=nj, nk=nk)
    ng = CM1_OPEN_NGXY
    zero = np.float32(0.0)
    half = np.float32(0.5)

    for k0 in range(nk):
        k = k0 + 1
        for j in range(nj):
            y = j + ng
            for face in range(ni + 1):
                left = max(face - 1, 0)
                right = min(face, ni - 1)
                density = _add(np, m["rho0_3d"][k0, j, left], m["rho0_3d"][k0, j, right])
                a["rru"][k, y, face + ng] = _mul(
                    np,
                    _mul(np, half, density),
                    _mul(np, a["u3d"][k, y, face + ng], m["rgzu"][j, face]),
                )
        for face in range(nj + 1):
            south = max(face - 1, 0)
            north = min(face, nj - 1)
            y = face + ng
            for i in range(ni):
                density = _add(np, m["rho0_3d"][k0, south, i], m["rho0_3d"][k0, north, i])
                a["rrv"][k, y, i + ng] = _mul(
                    np,
                    _mul(np, half, density),
                    _mul(np, a["v3d"][k, y, i + ng], m["rgzv"][face, i]),
                )

    a["rrw"][1, ng : ng + nj, ng : ng + ni] = zero
    a["rrw"][nk + 1, ng : ng + nj, ng : ng + ni] = zero
    dz32 = np.float32(dz)
    zt = _mul(np, np.float32(nk), dz32)
    rzt = _div(np, np.float32(1.0), zt)
    for k in range(2, nk + 1):
        sigmaf = _mul(np, np.float32(k - 1), dz32)
        sigma = _mul(np, np.float32(k - 1.5), dz32)
        r2 = _mul(np, _sub(np, sigmaf, sigma), _div(np, np.float32(1.0), dz32))
        r1 = _mul(np, half, _sub(np, np.float32(1.0), r2))
        r2 = _mul(np, half, r2)
        for j in range(nj):
            y = j + ng
            for i in range(ni):
                x = i + ng
                ux = _add(
                    np,
                    _mul(np, r2, _add(np, a["rru"][k, y, x], a["rru"][k, y, x + 1])),
                    _mul(np, r1, _add(np, a["rru"][k - 1, y, x], a["rru"][k - 1, y, x + 1])),
                )
                vy = _add(
                    np,
                    _mul(np, r2, _add(np, a["rrv"][k, y, x], a["rrv"][k, y + 1, x])),
                    _mul(np, r1, _add(np, a["rrv"][k - 1, y, x], a["rrv"][k - 1, y + 1, x])),
                )
                terrain = _add(np, _mul(np, ux, m["dzdx"][j, i]), _mul(np, vy, m["dzdy"][j, i]))
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
    rdz = _div(np, np.float32(1.0), dz32)
    smeps = np.float32(CM1_SMEPS_F32)
    for k0 in range(nk):
        k = k0 + 1
        for j in range(nj):
            y = j + ng
            for i in range(ni):
                x = i + ng
                value = _add(
                    np,
                    _mul(np, _sub(np, a["rru"][k, y, x + 1], a["rru"][k, y, x]), rdx),
                    _mul(np, _sub(np, a["rrv"][k, y + 1, x], a["rrv"][k, y, x]), rdy),
                )
                value = _add(np, value, _mul(np, _sub(np, a["rrw"][k + 1, y, x], a["rrw"][k, y, x]), rdz))
                divx[k0, j, i] = zero if abs(float(value)) < float(smeps) else value

    _cpu_momentum_transport5(
        np, a, m, divx, ni=ni, nj=nj, nk=nk, rdx=rdx, rdy=rdy, rdz=rdz,
        lateral_sides=lateral_sides,
    )


def _cpu_momentum_transport5(
    np: Any, a: Mapping[str, Any], m: Mapping[str, Any], divx: Any, *,
    ni: int, nj: int, nk: int, rdx: Any, rdy: Any, rdz: Any,
    lateral_sides: tuple[int, int, int, int] = MOMENTUM_ALL_OPEN_SIDES,
) -> None:
    """``advu``/``advv``/``advw`` increments for fluxes and ``divx`` already formed.

    The statements are exactly the tail of
    :func:`cpu_terrain_open_momentum_advection5_inplace`; they read the
    advected field from ``a["u3d"]``/``a["v3d"]``/``a["w3d"]`` and the mass
    fluxes from ``a["rru"]``/``a["rrv"]``/``a["rrw"]``, which ``pdcomp.F:317-326``
    supplies separately (perturbation winds advected by ``rho0``-weighted
    total winds).

    ``lateral_sides`` selects adv.F's face ranges and inflow edits: faces
    ``i1..i2`` of advu and rows ``j1..j2`` of advv include a periodic side's
    boundary face, whose mass fluxes, ``divx`` and ``rho0`` are read across
    the side at the wrapped index, and ``doitw..doitn`` run on open sides only.
    """

    ng = CM1_OPEN_NGXY
    zero = np.float32(0.0)
    half = np.float32(0.5)
    lat = _LateralFamily(ni, nj, momentum_lateral_sides(lateral_sides))
    first_face, last_face = (0 if lat.periodic(0) else 1), (ni if lat.periodic(1) else ni - 1)
    first_row, last_row = (0 if lat.periodic(2) else 1), (nj if lat.periodic(3) else nj - 1)

    for k0 in range(nk):
        k = k0 + 1
        for j in range(nj):
            y = j + ng
            for i_face in range(first_face, last_face + 1):
                x = i_face + ng
                west, east = lat.cx(i_face - 1), lat.cx(i_face)
                advx = _mul(np, -_sub(np, _u_fx(np, a, k, j, i_face, ng, lat), _u_fx(np, a, k, j, i_face - 1, ng, lat)), rdx)
                advy = _mul(np, -_sub(np, _u_fy(np, a, k, j + 1, i_face, ng, nj, lat), _u_fy(np, a, k, j, i_face, ng, nj, lat)), rdy)
                if j == 0 and lat.open(2):
                    mass = _add(np, a["rrv"][k, ng, east + ng], a["rrv"][k, ng, west + ng])
                    if mass >= zero:
                        correction = _add(
                            np,
                            _sub(np, a["rrv"][k, ng + 1, west + ng], a["rrv"][k, ng, west + ng]),
                            _sub(np, a["rrv"][k, ng + 1, east + ng], a["rrv"][k, ng, east + ng]),
                        )
                        advy = _sub(np, advy, _mul(np, _mul(np, a["u3d"][k, y, x], half), _mul(np, correction, _mul(np, rdy, a["vh"][y]))))
                elif j == nj - 1 and lat.open(3):
                    mass = _add(np, a["rrv"][k, ng + nj, east + ng], a["rrv"][k, ng + nj, west + ng])
                    if mass <= zero:
                        correction = _add(
                            np,
                            _sub(np, a["rrv"][k, y + 1, west + ng], a["rrv"][k, y, west + ng]),
                            _sub(np, a["rrv"][k, y + 1, east + ng], a["rrv"][k, y, east + ng]),
                        )
                        advy = _sub(np, advy, _mul(np, _mul(np, a["u3d"][k, y, x], half), _mul(np, correction, _mul(np, rdy, a["vh"][y]))))
                if k == 1:
                    advz = _mul(np, -_uv_vertical_flux(np, a, m, field="u3d", kface=2, j=j, i=i_face, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), rdz)
                elif k == nk:
                    advz = _mul(np, _uv_vertical_flux(np, a, m, field="u3d", kface=nk, j=j, i=i_face, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), rdz)
                else:
                    advz = _mul(np, -_sub(np, _uv_vertical_flux(np, a, m, field="u3d", kface=k + 1, j=j, i=i_face, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), _uv_vertical_flux(np, a, m, field="u3d", kface=k, j=j, i=i_face, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat)), rdz)
                total = _add(np, _add(np, advx, advy), advz)
                total = _add(np, total, _mul(np, a["u3d"][k, y, x], _mul(np, half, _add(np, divx[k0, j, east], divx[k0, j, west]))))
                scale = _div(np, m["gzu"][j, i_face], _mul(np, half, _add(np, m["rho0_3d"][k0, j, west], m["rho0_3d"][k0, j, east])))
                a["uten"][k, y, x] = _add(np, a["uten"][k, y, x], _mul(np, total, scale))

    for k0 in range(nk):
        k = k0 + 1
        for j_face in range(first_row, last_row + 1):
            y = j_face + ng
            south, north = lat.cy(j_face - 1), lat.cy(j_face)
            for i in range(ni):
                x = i + ng
                advx = _mul(np, -_sub(np, _v_fx(np, a, k, j_face, i + 1, ng, ni, lat), _v_fx(np, a, k, j_face, i, ng, ni, lat)), rdx)
                advy = _mul(np, -_sub(np, _v_fy(np, a, k, j_face, i, ng, lat), _v_fy(np, a, k, j_face - 1, i, ng, lat)), rdy)
                if i == 0 and lat.open(0):
                    mass = _add(np, a["rru"][k, north + ng, ng], a["rru"][k, south + ng, ng])
                    if mass >= zero:
                        correction = _add(
                            np,
                            _sub(np, a["rru"][k, south + ng, ng + 1], a["rru"][k, south + ng, ng]),
                            _sub(np, a["rru"][k, north + ng, ng + 1], a["rru"][k, north + ng, ng]),
                        )
                        advx = _sub(np, advx, _mul(np, _mul(np, a["v3d"][k, y, x], half), _mul(np, correction, _mul(np, rdx, a["uh"][ng]))))
                elif i == ni - 1 and lat.open(1):
                    mass = _add(np, a["rru"][k, north + ng, ng + ni], a["rru"][k, south + ng, ng + ni])
                    if mass <= zero:
                        correction = _add(
                            np,
                            _sub(np, a["rru"][k, south + ng, x + 1], a["rru"][k, south + ng, x]),
                            _sub(np, a["rru"][k, north + ng, x + 1], a["rru"][k, north + ng, x]),
                        )
                        advx = _sub(np, advx, _mul(np, _mul(np, a["v3d"][k, y, x], half), _mul(np, correction, _mul(np, rdx, a["uh"][x]))))
                if k == 1:
                    advz = _mul(np, -_uv_vertical_flux(np, a, m, field="v3d", kface=2, j=j_face, i=i, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), rdz)
                elif k == nk:
                    advz = _mul(np, _uv_vertical_flux(np, a, m, field="v3d", kface=nk, j=j_face, i=i, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), rdz)
                else:
                    advz = _mul(np, -_sub(np, _uv_vertical_flux(np, a, m, field="v3d", kface=k + 1, j=j_face, i=i, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat), _uv_vertical_flux(np, a, m, field="v3d", kface=k, j=j_face, i=i, ng=ng, ni=ni, nj=nj, nk=nk, lat=lat)), rdz)
                total = _add(np, _add(np, advx, advy), advz)
                total = _add(np, total, _mul(np, a["v3d"][k, y, x], _mul(np, half, _add(np, divx[k0, north, i], divx[k0, south, i]))))
                scale = _div(np, m["gzv"][j_face, i], _mul(np, half, _add(np, m["rho0_3d"][k0, south, i], m["rho0_3d"][k0, north, i])))
                a["vten"][k, y, x] = _add(np, a["vten"][k, y, x], _mul(np, total, scale))

    for kface in range(2, nk + 1):
        for j in range(nj):
            y = j + ng
            for i in range(ni):
                x = i + ng
                fx0 = _w_fx(np, a, m, kface, j, i, ng, ni, nj, lat)
                fx1 = _w_fx(np, a, m, kface, j, i + 1, ng, ni, nj, lat)
                advx = _mul(np, -_sub(np, fx1, fx0), _mul(np, rdx, a["uh"][x]))
                fy0 = _w_fy(np, a, m, kface, j, i, ng, ni, nj, lat)
                fy1 = _w_fy(np, a, m, kface, j + 1, i, ng, ni, nj, lat)
                advy = _mul(np, -_sub(np, fy1, fy0), _mul(np, rdy, a["vh"][y]))
                c2 = m["c2"][kface - 1, j, i]
                c1 = _sub(np, np.float32(1.0), c2)
                if i == 0 and lat.open(0):
                    mass = _add(np, _mul(np, c2, a["rru"][kface, y, ng]), _mul(np, c1, a["rru"][kface - 1, y, ng]))
                    if mass >= zero:
                        correction = _add(np, _mul(np, c1, _sub(np, a["rru"][kface - 1, y, ng + 1], a["rru"][kface - 1, y, ng])), _mul(np, c2, _sub(np, a["rru"][kface, y, ng + 1], a["rru"][kface, y, ng])))
                        advx = _sub(np, advx, _mul(np, a["w3d"][kface, y, x], _mul(np, correction, _mul(np, rdx, a["uh"][x]))))
                elif i == ni - 1 and lat.open(1):
                    mass = _add(np, _mul(np, c2, a["rru"][kface, y, ng + ni]), _mul(np, c1, a["rru"][kface - 1, y, ng + ni]))
                    if mass <= zero:
                        correction = _add(np, _mul(np, c1, _sub(np, a["rru"][kface - 1, y, x + 1], a["rru"][kface - 1, y, x])), _mul(np, c2, _sub(np, a["rru"][kface, y, x + 1], a["rru"][kface, y, x])))
                        advx = _sub(np, advx, _mul(np, a["w3d"][kface, y, x], _mul(np, correction, _mul(np, rdx, a["uh"][x]))))
                if j == 0 and lat.open(2):
                    mass = _add(np, _mul(np, c2, a["rrv"][kface, ng, x]), _mul(np, c1, a["rrv"][kface - 1, ng, x]))
                    if mass >= zero:
                        correction = _add(np, _mul(np, c1, _sub(np, a["rrv"][kface - 1, ng + 1, x], a["rrv"][kface - 1, ng, x])), _mul(np, c2, _sub(np, a["rrv"][kface, ng + 1, x], a["rrv"][kface, ng, x])))
                        advy = _sub(np, advy, _mul(np, a["w3d"][kface, y, x], _mul(np, correction, _mul(np, rdy, a["vh"][y]))))
                elif j == nj - 1 and lat.open(3):
                    mass = _add(np, _mul(np, c2, a["rrv"][kface, ng + nj, x]), _mul(np, c1, a["rrv"][kface - 1, ng + nj, x]))
                    if mass <= zero:
                        correction = _add(np, _mul(np, c1, _sub(np, a["rrv"][kface - 1, y + 1, x], a["rrv"][kface - 1, y, x])), _mul(np, c2, _sub(np, a["rrv"][kface, y + 1, x], a["rrv"][kface, y, x])))
                        advy = _sub(np, advy, _mul(np, a["w3d"][kface, y, x], _mul(np, correction, _mul(np, rdy, a["vh"][y]))))
                advz = _mul(np, -_sub(np, _w_vertical_flux(np, a, kface, j, i, ng, nk), _w_vertical_flux(np, a, kface - 1, j, i, ng, nk)), rdz)
                total = _add(np, _add(np, advx, advy), advz)
                total = _add(np, total, _mul(np, a["w3d"][kface, y, x], _add(np, _mul(np, c2, divx[kface - 1, j, i]), _mul(np, c1, divx[kface - 2, j, i]))))
                scale = _mul(np, m["rrf0"][kface - 1, j, i], m["gz"][j, i])
                a["wten"][kface, y, x] = _add(np, a["wten"][kface, y, x], _mul(np, total, scale))


CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES = (
    "cm1_mom5_horizontal_mass_v1",
    "cm1_mom5_vertical_mass_v1",
    "cm1_mom5_divergence_v1",
    "cm1_mom5_advu_v1",
    "cm1_mom5_advv_v1",
    "cm1_mom5_advw_v1",
)


CUDA_MOMENTUM_ADVECTION5_SOURCE = r'''// gpuwm-cm1.cuda-momentum-advection5-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ unsigned int clampi(int a,unsigned int n){return (a<0)?0u:((unsigned int)a>=n?n-1u:(unsigned int)a);}
__device__ __forceinline__ float flx3(float s1,float s2,float s3){float r=m32(-1.0f/6.0f,s1);r=a32(r,m32(5.0f/6.0f,s2));return a32(r,m32(2.0f/6.0f,s3));}
__device__ __forceinline__ float flx5(float s1,float s2,float s3,float s4,float s5){float r=m32(2.0f/60.0f,s1);r=a32(r,m32(-13.0f/60.0f,s2));r=a32(r,m32(47.0f/60.0f,s3));r=a32(r,m32(27.0f/60.0f,s4));return a32(r,m32(-3.0f/60.0f,s5));}
__device__ __forceinline__ float ma(const float* q,unsigned int k,int j,int i,unsigned int ng,unsigned int ny,unsigned int nx){return q[i3(k,(unsigned int)(j+(int)ng),(unsigned int)(i+(int)ng),ny,nx)];}
__device__ __forceinline__ float met3(const float* q,unsigned int kf,int j,int i,unsigned int ni,unsigned int nj){return q[((unsigned long long)(kf-1u)*nj+clampi(j,nj))*ni+clampi(i,ni)];}
// obc packs CM1 wbc|ebc<<4|sbc<<8|nbc<<12 (bc.F lateral values 1..4; side 0..3 is w,e,s,n).
__device__ __forceinline__ unsigned int obc_side(unsigned int obc,unsigned int side){return (obc>>(4u*side))&15u;}
// adv.F advu/advv/advw doitw..doitn: the inflow edits run on an open side (value 2) only.
__device__ __forceinline__ bool obc_open(unsigned int obc,unsigned int side){return obc_side(obc,side)==2u;}
// param.F:6678-6681 zeroes ibw..ibn exactly on a periodic side, so advu's i1..i2 and advv's j1..j2 reach its boundary face.
__device__ __forceinline__ unsigned int obc_per(unsigned int obc,unsigned int side){return (obc_side(obc,side)==1u)?1u:0u;}
// Across a periodic side solve2.F:493-617 reads rru/rrv/rrw and divx in the halo it formed from bc.F's periodic
// u3d/v3d/w3d copies, which are the words at the wrapped index: wc wraps a cell, wf a face (face n is the boundary face).
__device__ __forceinline__ int wc(int i,int n,unsigned int lo,unsigned int hi){return (lo&&i<0)?i+n:((hi&&i>=n)?i-n:i);}
__device__ __forceinline__ int wf(int i,int n,unsigned int lo,unsigned int hi){return (lo&&i<0)?i+n:((hi&&i>n)?i-n:i);}
__device__ __forceinline__ float ufx(const float* u,const float* rru,unsigned int k,int j,int f,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+1u+2u*ng,ny=nj+2u*ng,pw=obc_per(obc,0u),pe=obc_per(obc,1u);float m=m32(0.5f,a32(ma(rru,k,j,wf(f,(int)ni,pw,pe),ng,ny,nx),ma(rru,k,j,wf(f+1,(int)ni,pw,pe),ng,ny,nx)));float r=(m>=0.0f)?flx5(ma(u,k,j,f-2,ng,ny,nx),ma(u,k,j,f-1,ng,ny,nx),ma(u,k,j,f,ng,ny,nx),ma(u,k,j,f+1,ng,ny,nx),ma(u,k,j,f+2,ng,ny,nx)):flx5(ma(u,k,j,f+3,ng,ny,nx),ma(u,k,j,f+2,ng,ny,nx),ma(u,k,j,f+1,ng,ny,nx),ma(u,k,j,f,ng,ny,nx),ma(u,k,j,f-1,ng,ny,nx));return m32(m,r);}
__device__ __forceinline__ float ufy0(const float* u,const float* rrv,unsigned int k,int p,int f,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,pw=obc_per(obc,0u),pe=obc_per(obc,1u);float m=m32(0.5f,a32(ma(rrv,k,p,wc(f,(int)ni,pw,pe),ng,nyv,nxv),ma(rrv,k,p,wc(f-1,(int)ni,pw,pe),ng,nyv,nxv)));float r=(m>=0.0f)?flx5(ma(u,k,p-3,f,ng,nyu,nxu),ma(u,k,p-2,f,ng,nyu,nxu),ma(u,k,p-1,f,ng,nyu,nxu),ma(u,k,p,f,ng,nyu,nxu),ma(u,k,p+1,f,ng,nyu,nxu)):flx5(ma(u,k,p+2,f,ng,nyu,nxu),ma(u,k,p+1,f,ng,nyu,nxu),ma(u,k,p,f,ng,nyu,nxu),ma(u,k,p-1,f,ng,nyu,nxu),ma(u,k,p-2,f,ng,nyu,nxu));return m32(m,r);}
__device__ __forceinline__ float ufy(const float* u,const float* rrv,unsigned int k,int p,int f,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+2u*ng,ny=nj+1u+2u*ng,pw=obc_per(obc,0u),pe=obc_per(obc,1u);float mt=a32(ma(rrv,k,p,wc(f,(int)ni,pw,pe),ng,ny,nx),ma(rrv,k,p,wc(f-1,(int)ni,pw,pe),ng,ny,nx));if(p==0&&obc_open(obc,2u)&&mt>=0.0f)return ufy0(u,rrv,k,1,f,ni,nj,ng,obc);if(p==(int)nj&&obc_open(obc,3u)&&mt<=0.0f)return ufy0(u,rrv,k,(int)nj-1,f,ni,nj,ng,obc);return ufy0(u,rrv,k,p,f,ni,nj,ng,obc);}
__device__ __forceinline__ float vfx0(const float* v,const float* rru,unsigned int k,int g,int p,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,ps=obc_per(obc,2u),pn=obc_per(obc,3u);float m=m32(0.5f,a32(ma(rru,k,wc(g,(int)nj,ps,pn),p,ng,nyu,nxu),ma(rru,k,wc(g-1,(int)nj,ps,pn),p,ng,nyu,nxu)));float r=(m>=0.0f)?flx5(ma(v,k,g,p-3,ng,nyv,nxv),ma(v,k,g,p-2,ng,nyv,nxv),ma(v,k,g,p-1,ng,nyv,nxv),ma(v,k,g,p,ng,nyv,nxv),ma(v,k,g,p+1,ng,nyv,nxv)):flx5(ma(v,k,g,p+2,ng,nyv,nxv),ma(v,k,g,p+1,ng,nyv,nxv),ma(v,k,g,p,ng,nyv,nxv),ma(v,k,g,p-1,ng,nyv,nxv),ma(v,k,g,p-2,ng,nyv,nxv));return m32(m,r);}
__device__ __forceinline__ float vfx(const float* v,const float* rru,unsigned int k,int g,int p,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+1u+2u*ng,ny=nj+2u*ng,ps=obc_per(obc,2u),pn=obc_per(obc,3u);float mt=a32(ma(rru,k,wc(g,(int)nj,ps,pn),p,ng,ny,nx),ma(rru,k,wc(g-1,(int)nj,ps,pn),p,ng,ny,nx));if(p==0&&obc_open(obc,0u)&&mt>=0.0f)return vfx0(v,rru,k,g,1,ni,nj,ng,obc);if(p==(int)ni&&obc_open(obc,1u)&&mt<=0.0f)return vfx0(v,rru,k,g,(int)ni-1,ni,nj,ng,obc);return vfx0(v,rru,k,g,p,ni,nj,ng,obc);}
__device__ __forceinline__ float vfy(const float* v,const float* rrv,unsigned int k,int p,int i,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+2u*ng,ny=nj+1u+2u*ng,ps=obc_per(obc,2u),pn=obc_per(obc,3u);float m=m32(0.5f,a32(ma(rrv,k,wf(p,(int)nj,ps,pn),i,ng,ny,nx),ma(rrv,k,wf(p+1,(int)nj,ps,pn),i,ng,ny,nx)));float r=(m>=0.0f)?flx5(ma(v,k,p-2,i,ng,ny,nx),ma(v,k,p-1,i,ng,ny,nx),ma(v,k,p,i,ng,ny,nx),ma(v,k,p+1,i,ng,ny,nx),ma(v,k,p+2,i,ng,ny,nx)):flx5(ma(v,k,p+3,i,ng,ny,nx),ma(v,k,p+2,i,ng,ny,nx),ma(v,k,p+1,i,ng,ny,nx),ma(v,k,p,i,ng,ny,nx),ma(v,k,p-1,i,ng,ny,nx));return m32(m,r);}
__device__ __forceinline__ float uvz(const float* q,const float* rrw,const float* c2a,unsigned int stag,unsigned int kf,int j,int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int obc){unsigned int pw=obc_per(obc,0u),pe=obc_per(obc,1u),ps=obc_per(obc,2u),pn=obc_per(obc,3u),nx=ni+2u*ng,ny=nj+2u*ng,nxq=(stag==2u)?ni+1u+2u*ng:nx,nyq=(stag==2u)?ny:nj+1u+2u*ng;float m,c2;if(stag==2u){m=m32(0.5f,a32(ma(rrw,kf,j,wc(i,(int)ni,pw,pe),ng,ny,nx),ma(rrw,kf,j,wc(i-1,(int)ni,pw,pe),ng,ny,nx)));c2=m32(0.5f,a32(met3(c2a,kf,j,i-1,ni,nj),met3(c2a,kf,j,i,ni,nj)));}else{m=m32(0.5f,a32(ma(rrw,kf,wc(j,(int)nj,ps,pn),i,ng,ny,nx),ma(rrw,kf,wc(j-1,(int)nj,ps,pn),i,ng,ny,nx)));c2=m32(0.5f,a32(met3(c2a,kf,j-1,i,ni,nj),met3(c2a,kf,j,i,ni,nj)));}float c1=s32(1.0f,c2),r;if(kf==2u)r=(m>=0.0f)?a32(m32(c1,ma(q,1u,j,i,ng,nyq,nxq)),m32(c2,ma(q,2u,j,i,ng,nyq,nxq))):flx3(ma(q,3u,j,i,ng,nyq,nxq),ma(q,2u,j,i,ng,nyq,nxq),ma(q,1u,j,i,ng,nyq,nxq));else if(kf==nk)r=(m>=0.0f)?flx3(ma(q,nk-2u,j,i,ng,nyq,nxq),ma(q,nk-1u,j,i,ng,nyq,nxq),ma(q,nk,j,i,ng,nyq,nxq)):a32(m32(c1,ma(q,nk-1u,j,i,ng,nyq,nxq)),m32(c2,ma(q,nk,j,i,ng,nyq,nxq)));else if(kf==3u)r=(m>=0.0f)?flx3(ma(q,1u,j,i,ng,nyq,nxq),ma(q,2u,j,i,ng,nyq,nxq),ma(q,3u,j,i,ng,nyq,nxq)):flx5(ma(q,5u,j,i,ng,nyq,nxq),ma(q,4u,j,i,ng,nyq,nxq),ma(q,3u,j,i,ng,nyq,nxq),ma(q,2u,j,i,ng,nyq,nxq),ma(q,1u,j,i,ng,nyq,nxq));else if(kf==nk-1u)r=(m>=0.0f)?flx5(ma(q,kf-3u,j,i,ng,nyq,nxq),ma(q,kf-2u,j,i,ng,nyq,nxq),ma(q,kf-1u,j,i,ng,nyq,nxq),ma(q,kf,j,i,ng,nyq,nxq),ma(q,kf+1u,j,i,ng,nyq,nxq)):flx3(ma(q,kf+1u,j,i,ng,nyq,nxq),ma(q,kf,j,i,ng,nyq,nxq),ma(q,kf-1u,j,i,ng,nyq,nxq));else r=(m>=0.0f)?flx5(ma(q,kf-3u,j,i,ng,nyq,nxq),ma(q,kf-2u,j,i,ng,nyq,nxq),ma(q,kf-1u,j,i,ng,nyq,nxq),ma(q,kf,j,i,ng,nyq,nxq),ma(q,kf+1u,j,i,ng,nyq,nxq)):flx5(ma(q,kf+2u,j,i,ng,nyq,nxq),ma(q,kf+1u,j,i,ng,nyq,nxq),ma(q,kf,j,i,ng,nyq,nxq),ma(q,kf-1u,j,i,ng,nyq,nxq),ma(q,kf-2u,j,i,ng,nyq,nxq));return m32(m,r);}
__device__ __forceinline__ float wfx0(const float* w,const float* rru,const float* c2a,unsigned int kf,int j,int p,unsigned int ni,unsigned int nj,unsigned int ng){unsigned int nxu=ni+1u+2u*ng,ny=nj+2u*ng,nxw=ni+2u*ng;float c2=m32(0.5f,a32(met3(c2a,kf,j,p-1,ni,nj),met3(c2a,kf,j,p,ni,nj))),c1=s32(1.0f,c2),m=a32(m32(c2,ma(rru,kf,j,p,ng,ny,nxu)),m32(c1,ma(rru,kf-1u,j,p,ng,ny,nxu)));float r=(m>=0.0f)?flx5(ma(w,kf,j,p-3,ng,ny,nxw),ma(w,kf,j,p-2,ng,ny,nxw),ma(w,kf,j,p-1,ng,ny,nxw),ma(w,kf,j,p,ng,ny,nxw),ma(w,kf,j,p+1,ng,ny,nxw)):flx5(ma(w,kf,j,p+2,ng,ny,nxw),ma(w,kf,j,p+1,ng,ny,nxw),ma(w,kf,j,p,ng,ny,nxw),ma(w,kf,j,p-1,ng,ny,nxw),ma(w,kf,j,p-2,ng,ny,nxw));return m32(m,r);}
__device__ __forceinline__ float wfx(const float* w,const float* rru,const float* c2a,unsigned int kf,int j,int p,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+1u+2u*ng,ny=nj+2u*ng;float c2=m32(0.5f,a32(met3(c2a,kf,j,p-1,ni,nj),met3(c2a,kf,j,p,ni,nj))),c1=s32(1.0f,c2),m=a32(m32(c2,ma(rru,kf,j,p,ng,ny,nx)),m32(c1,ma(rru,kf-1u,j,p,ng,ny,nx)));if(p==0&&obc_open(obc,0u)&&m>=0.0f)return wfx0(w,rru,c2a,kf,j,1,ni,nj,ng);if(p==(int)ni&&obc_open(obc,1u)&&m<=0.0f)return wfx0(w,rru,c2a,kf,j,(int)ni-1,ni,nj,ng);return wfx0(w,rru,c2a,kf,j,p,ni,nj,ng);}
__device__ __forceinline__ float wfy0(const float* w,const float* rrv,const float* c2a,unsigned int kf,int p,int i,unsigned int ni,unsigned int nj,unsigned int ng){unsigned int nx=ni+2u*ng,nyv=nj+1u+2u*ng,nyw=nj+2u*ng;float c2=m32(0.5f,a32(met3(c2a,kf,p-1,i,ni,nj),met3(c2a,kf,p,i,ni,nj))),c1=s32(1.0f,c2),m=a32(m32(c2,ma(rrv,kf,p,i,ng,nyv,nx)),m32(c1,ma(rrv,kf-1u,p,i,ng,nyv,nx)));float r=(m>=0.0f)?flx5(ma(w,kf,p-3,i,ng,nyw,nx),ma(w,kf,p-2,i,ng,nyw,nx),ma(w,kf,p-1,i,ng,nyw,nx),ma(w,kf,p,i,ng,nyw,nx),ma(w,kf,p+1,i,ng,nyw,nx)):flx5(ma(w,kf,p+2,i,ng,nyw,nx),ma(w,kf,p+1,i,ng,nyw,nx),ma(w,kf,p,i,ng,nyw,nx),ma(w,kf,p-1,i,ng,nyw,nx),ma(w,kf,p-2,i,ng,nyw,nx));return m32(m,r);}
__device__ __forceinline__ float wfy(const float* w,const float* rrv,const float* c2a,unsigned int kf,int p,int i,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int obc){unsigned int nx=ni+2u*ng,ny=nj+1u+2u*ng;float c2=m32(0.5f,a32(met3(c2a,kf,p-1,i,ni,nj),met3(c2a,kf,p,i,ni,nj))),c1=s32(1.0f,c2),m=a32(m32(c2,ma(rrv,kf,p,i,ng,ny,nx)),m32(c1,ma(rrv,kf-1u,p,i,ng,ny,nx)));if(p==0&&obc_open(obc,2u)&&m>=0.0f)return wfy0(w,rrv,c2a,kf,1,i,ni,nj,ng);if(p==(int)nj&&obc_open(obc,3u)&&m<=0.0f)return wfy0(w,rrv,c2a,kf,(int)nj-1,i,ni,nj,ng);return wfy0(w,rrv,c2a,kf,p,i,ni,nj,ng);}
__device__ __forceinline__ float wz(const float* w,const float* rrw,unsigned int k,int j,int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float m=m32(0.5f,a32(ma(rrw,k,j,i,ng,ny,nx),ma(rrw,k+1u,j,i,ng,ny,nx))),r;if(k==1u)r=(m>=0.0f)?m32(0.5f,a32(ma(w,1u,j,i,ng,ny,nx),ma(w,2u,j,i,ng,ny,nx))):flx3(ma(w,3u,j,i,ng,ny,nx),ma(w,2u,j,i,ng,ny,nx),ma(w,1u,j,i,ng,ny,nx));else if(k==2u)r=(m>=0.0f)?flx3(ma(w,1u,j,i,ng,ny,nx),ma(w,2u,j,i,ng,ny,nx),ma(w,3u,j,i,ng,ny,nx)):flx5(ma(w,5u,j,i,ng,ny,nx),ma(w,4u,j,i,ng,ny,nx),ma(w,3u,j,i,ng,ny,nx),ma(w,2u,j,i,ng,ny,nx),ma(w,1u,j,i,ng,ny,nx));else if(k==nk-1u)r=(m>=0.0f)?flx5(ma(w,k-2u,j,i,ng,ny,nx),ma(w,k-1u,j,i,ng,ny,nx),ma(w,k,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx),ma(w,k+2u,j,i,ng,ny,nx)):flx3(ma(w,k+2u,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx),ma(w,k,j,i,ng,ny,nx));else if(k==nk)r=(m>=0.0f)?flx3(ma(w,k-1u,j,i,ng,ny,nx),ma(w,k,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx)):m32(0.5f,a32(ma(w,k,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx)));else r=(m>=0.0f)?flx5(ma(w,k-2u,j,i,ng,ny,nx),ma(w,k-1u,j,i,ng,ny,nx),ma(w,k,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx),ma(w,k+2u,j,i,ng,ny,nx)):flx5(ma(w,k+3u,j,i,ng,ny,nx),ma(w,k+2u,j,i,ng,ny,nx),ma(w,k+1u,j,i,ng,ny,nx),ma(w,k,j,i,ng,ny,nx),ma(w,k-1u,j,i,ng,ny,nx));return m32(m,r);}

extern "C" __global__ void cm1_mom5_horizontal_mass_v1(const float* u,const float* v,const float* rho0,const float* rgzu,const float* rgzv,float* rru,float* rrv,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni;unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;for(;q<nu+nv;q+=s){if(q<nu){unsigned int f=(unsigned int)(q%(ni+1u));unsigned long long z=q/(ni+1u);unsigned int j=(unsigned int)(z%nj),k=(unsigned int)(z/nj)+1u,l=(f==0u)?0u:f-1u,r=(f==ni)?ni-1u:f;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j)*ni+l],rho0[((unsigned long long)(k-1u)*nj+j)*ni+r]);rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(m32(0.5f,rh),m32(u[i3(k,j+ng,f+ng,nyu,nxu)],rgzu[(unsigned long long)j*(ni+1u)+f]));}else{unsigned long long z=q-nu;unsigned int i=(unsigned int)(z%ni);z/=ni;unsigned int f=(unsigned int)(z%(nj+1u)),k=(unsigned int)(z/(nj+1u))+1u,b=(f==0u)?0u:f-1u,t=(f==nj)?nj-1u:f;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+b)*ni+i],rho0[((unsigned long long)(k-1u)*nj+t)*ni+i]);rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(m32(0.5f,rh),m32(v[i3(k,f+ng,i+ng,nyv,nxv)],rgzv[(unsigned long long)f*ni+i]));}}}
extern "C" __global__ void cm1_mom5_vertical_mass_v1(const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nxu=ni+1u+2u*ng,nyv=nj+1u+2u*ng;float zt=m32((float)nk,dz),rzt=d32(1.0f,zt),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),kf=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;if(kf==1u||kf==nk+1u){rrw[i3(kf,y,x,ny,nx)]=0.0f;}else{float sf=m32((float)(kf-1u),dz),sm=m32((float)kf-1.5f,dz),r2=m32(s32(sf,sm),rdz),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);float ux=a32(m32(r2,a32(rru[i3(kf,y,x,ny,nxu)],rru[i3(kf,y,x+1u,ny,nxu)])),m32(r1,a32(rru[i3(kf-1u,y,x,ny,nxu)],rru[i3(kf-1u,y,x+1u,ny,nxu)]))),vy=a32(m32(r2,a32(rrv[i3(kf,y,x,nyv,nx)],rrv[i3(kf,y+1u,x,nyv,nx)])),m32(r1,a32(rrv[i3(kf-1u,y,x,nyv,nx)],rrv[i3(kf-1u,y+1u,x,nyv,nx)])));float tr=a32(m32(ux,dzdx[(unsigned long long)j*ni+i]),m32(vy,dzdy[(unsigned long long)j*ni+i]));tr=m32(tr,s32(sf,zt));tr=m32(tr,gz[(unsigned long long)j*ni+i]);tr=m32(tr,rzt);rrw[i3(kf,y,x,ny,nx)]=a32(m32(rf0[((unsigned long long)(kf-1u)*nj+j)*ni+i],w[i3(kf,y,x,ny,nx)]),tr);}}}
extern "C" __global__ void cm1_mom5_divergence_v1(const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,float smeps){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nxu=ni+1u+2u*ng,nyv=nj+1u+2u*ng;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;float d=a32(m32(s32(rru[i3(k,y,x+1u,ny,nxu)],rru[i3(k,y,x,ny,nxu)]),rdx),m32(s32(rrv[i3(k,y+1u,x,nyv,nx)],rrv[i3(k,y,x,nyv,nx)]),rdy));d=a32(d,m32(s32(rrw[i3(k+1u,y,x,ny,nx)],rrw[i3(k,y,x,ny,nx)]),rdz));divx[q]=(fabsf(d)<smeps)?0.0f:d;}}
extern "C" __global__ void cm1_mom5_advu_v1(const float* u,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzu,const float* vh,float* uten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){unsigned int pw=obc_per(obc,0u),pe=obc_per(obc,1u),nf=ni-1u+pw+pe;unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*nf;unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int f=(unsigned int)(q%nf)+1u-pw;unsigned long long z=q/nf;unsigned int j=(unsigned int)(z%nj),k0=(unsigned int)(z/nj),k=k0+1u,x=f+ng,y=j+ng;int fw=wc((int)f-1,(int)ni,pw,pe),fe=wc((int)f,(int)ni,pw,pe);float ax=m32(-s32(ufx(u,rru,k,j,(int)f,ni,nj,ng,obc),ufx(u,rru,k,j,(int)f-1,ni,nj,ng,obc)),rdx),ay=m32(-s32(ufy(u,rrv,k,(int)j+1,(int)f,ni,nj,ng,obc),ufy(u,rrv,k,(int)j,(int)f,ni,nj,ng,obc)),rdy);if(j==0u&&obc_open(obc,2u)&&a32(ma(rrv,k,0,fe,ng,nyv,nxv),ma(rrv,k,0,fw,ng,nyv,nxv))>=0.0f){float c=a32(s32(ma(rrv,k,1,fw,ng,nyv,nxv),ma(rrv,k,0,fw,ng,nyv,nxv)),s32(ma(rrv,k,1,fe,ng,nyv,nxv),ma(rrv,k,0,fe,ng,nyv,nxv)));ay=s32(ay,m32(m32(ma(u,k,0,(int)f,ng,nyu,nxu),0.5f),m32(c,m32(rdy,vh[ng]))));}else if(j==nj-1u&&obc_open(obc,3u)&&a32(ma(rrv,k,(int)nj,fe,ng,nyv,nxv),ma(rrv,k,(int)nj,fw,ng,nyv,nxv))<=0.0f){float c=a32(s32(ma(rrv,k,(int)j+1,fw,ng,nyv,nxv),ma(rrv,k,(int)j,fw,ng,nyv,nxv)),s32(ma(rrv,k,(int)j+1,fe,ng,nyv,nxv),ma(rrv,k,(int)j,fe,ng,nyv,nxv)));ay=s32(ay,m32(m32(ma(u,k,(int)j,(int)f,ng,nyu,nxu),0.5f),m32(c,m32(rdy,vh[y]))));}float az=(k==1u)?m32(-uvz(u,rrw,c2a,2u,2u,(int)j,(int)f,ni,nj,nk,ng,obc),rdz):((k==nk)?m32(uvz(u,rrw,c2a,2u,nk,(int)j,(int)f,ni,nj,nk,ng,obc),rdz):m32(-s32(uvz(u,rrw,c2a,2u,k+1u,(int)j,(int)f,ni,nj,nk,ng,obc),uvz(u,rrw,c2a,2u,k,(int)j,(int)f,ni,nj,nk,ng,obc)),rdz));float t=a32(a32(ax,ay),az);t=a32(t,m32(ma(u,k,(int)j,(int)f,ng,nyu,nxu),m32(0.5f,a32(divx[((unsigned long long)k0*nj+j)*ni+(unsigned int)fe],divx[((unsigned long long)k0*nj+j)*ni+(unsigned int)fw]))));float sc=d32(gzu[(unsigned long long)j*(ni+1u)+f],m32(0.5f,a32(rho0[((unsigned long long)k0*nj+j)*ni+(unsigned int)fw],rho0[((unsigned long long)k0*nj+j)*ni+(unsigned int)fe])));unsigned long long p=i3(k,y,x,nyu,nxu);uten[p]=a32(uten[p],m32(t,sc));}}
extern "C" __global__ void cm1_mom5_advv_v1(const float* v,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzv,const float* uh,float* vten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){unsigned int ps=obc_per(obc,2u),pn=obc_per(obc,3u),nr=nj-1u+ps+pn;unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nr*ni;unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxu=ni+1u+2u*ng,nyu=nj+2u*ng;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni);unsigned long long z=q/ni;unsigned int g=(unsigned int)(z%nr)+1u-ps,k0=(unsigned int)(z/nr),k=k0+1u,x=i+ng,y=g+ng;int gs=wc((int)g-1,(int)nj,ps,pn),gn=wc((int)g,(int)nj,ps,pn);float ax=m32(-s32(vfx(v,rru,k,(int)g,(int)i+1,ni,nj,ng,obc),vfx(v,rru,k,(int)g,(int)i,ni,nj,ng,obc)),rdx),ay=m32(-s32(vfy(v,rrv,k,(int)g,(int)i,ni,nj,ng,obc),vfy(v,rrv,k,(int)g-1,(int)i,ni,nj,ng,obc)),rdy);if(i==0u&&obc_open(obc,0u)&&a32(ma(rru,k,gn,0,ng,nyu,nxu),ma(rru,k,gs,0,ng,nyu,nxu))>=0.0f){float c=a32(s32(ma(rru,k,gs,1,ng,nyu,nxu),ma(rru,k,gs,0,ng,nyu,nxu)),s32(ma(rru,k,gn,1,ng,nyu,nxu),ma(rru,k,gn,0,ng,nyu,nxu)));ax=s32(ax,m32(m32(ma(v,k,(int)g,0,ng,nyv,nxv),0.5f),m32(c,m32(rdx,uh[ng]))));}else if(i==ni-1u&&obc_open(obc,1u)&&a32(ma(rru,k,gn,(int)ni,ng,nyu,nxu),ma(rru,k,gs,(int)ni,ng,nyu,nxu))<=0.0f){float c=a32(s32(ma(rru,k,gs,(int)i+1,ng,nyu,nxu),ma(rru,k,gs,(int)i,ng,nyu,nxu)),s32(ma(rru,k,gn,(int)i+1,ng,nyu,nxu),ma(rru,k,gn,(int)i,ng,nyu,nxu)));ax=s32(ax,m32(m32(ma(v,k,(int)g,(int)i,ng,nyv,nxv),0.5f),m32(c,m32(rdx,uh[x]))));}float az=(k==1u)?m32(-uvz(v,rrw,c2a,3u,2u,(int)g,(int)i,ni,nj,nk,ng,obc),rdz):((k==nk)?m32(uvz(v,rrw,c2a,3u,nk,(int)g,(int)i,ni,nj,nk,ng,obc),rdz):m32(-s32(uvz(v,rrw,c2a,3u,k+1u,(int)g,(int)i,ni,nj,nk,ng,obc),uvz(v,rrw,c2a,3u,k,(int)g,(int)i,ni,nj,nk,ng,obc)),rdz));float t=a32(a32(ax,ay),az);t=a32(t,m32(ma(v,k,(int)g,(int)i,ng,nyv,nxv),m32(0.5f,a32(divx[((unsigned long long)k0*nj+(unsigned int)gn)*ni+i],divx[((unsigned long long)k0*nj+(unsigned int)gs)*ni+i]))));float sc=d32(gzv[(unsigned long long)g*ni+i],m32(0.5f,a32(rho0[((unsigned long long)k0*nj+(unsigned int)gs)*ni+i],rho0[((unsigned long long)k0*nj+(unsigned int)gn)*ni+i])));unsigned long long p=i3(k,y,x,nyv,nxv);vten[p]=a32(vten[p],m32(t,sc));}}
extern "C" __global__ void cm1_mom5_advw_v1(const float* w,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* c2a,const float* rrf0,const float* gz,const float* uh,const float* vh,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk-1u)*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nxu=ni+1u+2u*ng,nyv=nj+1u+2u*ng;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),kf=(unsigned int)(q/((unsigned long long)ni*nj))+2u,x=i+ng,y=j+ng;float ax=m32(-s32(wfx(w,rru,c2a,kf,(int)j,(int)i+1,ni,nj,ng,obc),wfx(w,rru,c2a,kf,(int)j,(int)i,ni,nj,ng,obc)),m32(rdx,uh[x])),ay=m32(-s32(wfy(w,rrv,c2a,kf,(int)j+1,(int)i,ni,nj,ng,obc),wfy(w,rrv,c2a,kf,(int)j,(int)i,ni,nj,ng,obc)),m32(rdy,vh[y]));float c2=c2a[((unsigned long long)(kf-1u)*nj+j)*ni+i],c1=s32(1.0f,c2);if(i==0u&&obc_open(obc,0u)){float m=a32(m32(c2,ma(rru,kf,(int)j,0,ng,ny,nxu)),m32(c1,ma(rru,kf-1u,(int)j,0,ng,ny,nxu)));if(m>=0.0f){float c=a32(m32(c1,s32(ma(rru,kf-1u,(int)j,1,ng,ny,nxu),ma(rru,kf-1u,(int)j,0,ng,ny,nxu))),m32(c2,s32(ma(rru,kf,(int)j,1,ng,ny,nxu),ma(rru,kf,(int)j,0,ng,ny,nxu))));ax=s32(ax,m32(ma(w,kf,(int)j,0,ng,ny,nx),m32(c,m32(rdx,uh[x]))));}}else if(i==ni-1u&&obc_open(obc,1u)){float m=a32(m32(c2,ma(rru,kf,(int)j,(int)ni,ng,ny,nxu)),m32(c1,ma(rru,kf-1u,(int)j,(int)ni,ng,ny,nxu)));if(m<=0.0f){float c=a32(m32(c1,s32(ma(rru,kf-1u,(int)j,(int)i+1,ng,ny,nxu),ma(rru,kf-1u,(int)j,(int)i,ng,ny,nxu))),m32(c2,s32(ma(rru,kf,(int)j,(int)i+1,ng,ny,nxu),ma(rru,kf,(int)j,(int)i,ng,ny,nxu))));ax=s32(ax,m32(ma(w,kf,(int)j,(int)i,ng,ny,nx),m32(c,m32(rdx,uh[x]))));}}if(j==0u&&obc_open(obc,2u)){float m=a32(m32(c2,ma(rrv,kf,0,(int)i,ng,nyv,nx)),m32(c1,ma(rrv,kf-1u,0,(int)i,ng,nyv,nx)));if(m>=0.0f){float c=a32(m32(c1,s32(ma(rrv,kf-1u,1,(int)i,ng,nyv,nx),ma(rrv,kf-1u,0,(int)i,ng,nyv,nx))),m32(c2,s32(ma(rrv,kf,1,(int)i,ng,nyv,nx),ma(rrv,kf,0,(int)i,ng,nyv,nx))));ay=s32(ay,m32(ma(w,kf,0,(int)i,ng,ny,nx),m32(c,m32(rdy,vh[y]))));}}else if(j==nj-1u&&obc_open(obc,3u)){float m=a32(m32(c2,ma(rrv,kf,(int)nj,(int)i,ng,nyv,nx)),m32(c1,ma(rrv,kf-1u,(int)nj,(int)i,ng,nyv,nx)));if(m<=0.0f){float c=a32(m32(c1,s32(ma(rrv,kf-1u,(int)j+1,(int)i,ng,nyv,nx),ma(rrv,kf-1u,(int)j,(int)i,ng,nyv,nx))),m32(c2,s32(ma(rrv,kf,(int)j+1,(int)i,ng,nyv,nx),ma(rrv,kf,(int)j,(int)i,ng,nyv,nx))));ay=s32(ay,m32(ma(w,kf,(int)j,(int)i,ng,ny,nx),m32(c,m32(rdy,vh[y]))));}}float az=m32(-s32(wz(w,rrw,kf,(int)j,(int)i,ni,nj,nk,ng),wz(w,rrw,kf-1u,(int)j,(int)i,ni,nj,nk,ng)),rdz),t=a32(a32(ax,ay),az);t=a32(t,m32(ma(w,kf,(int)j,(int)i,ng,ny,nx),a32(m32(c2,divx[((unsigned long long)(kf-1u)*nj+j)*ni+i]),m32(c1,divx[((unsigned long long)(kf-2u)*nj+j)*ni+i]))));float sc=m32(rrf0[((unsigned long long)(kf-1u)*nj+j)*ni+i],gz[(unsigned long long)j*ni+i]);unsigned long long p=i3(kf,y,x,ny,nx);wten[p]=a32(wten[p],m32(t,sc));}}
'''

CUDA_MOMENTUM_ADVECTION5_SOURCE_BYTES = CUDA_MOMENTUM_ADVECTION5_SOURCE.encode("utf-8")
CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256 = "fa1d6182d52652fc74b4c6511efd219a9341b9b8df5193d32f6cbe10e1d7877d"


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
class CompiledMomentumAdvection5Kernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_momentum_advection5() -> CompiledMomentumAdvection5Kernels:
    if hashlib.sha256(CUDA_MOMENTUM_ADVECTION5_SOURCE_BYTES).hexdigest() != CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256:
        raise RuntimeError("packaged momentum-advection CUDA source differs from its pin")
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
        CUDA_MOMENTUM_ADVECTION5_SOURCE,
        "cm1_momentum_advection5_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_MOMENTUM_ADVECTION5_SOURCE,
        "cm1_momentum_advection5_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) for name in CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks momentum-advection entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled momentum-advection image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledMomentumAdvection5Kernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_cuda(cupy: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool) -> Any:
    if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must be a float32 cupy.ndarray")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


@dataclass(frozen=True, slots=True)
class CudaMomentumAdvection5Launch:
    divx: Any
    ready_event: Any
    compiled: CompiledMomentumAdvection5Kernels
    receipt: dict[str, Any]


def launch_cuda_terrain_open_momentum_advection5(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    divx: Any | None = None,
    selection: MomentumAdvection5Selection = DEFAULT_MOMENTUM_ADVECTION5_SELECTION,
    stream: Any | None = None,
    compiled: CompiledMomentumAdvection5Kernels | None = None,
    rrw_boundary_hook: Any | None = None,
    post_divergence_hook: Callable[..., Mapping[str, object]] | None = None,
    lateral_sides: tuple[int, int, int, int] = MOMENTUM_ALL_OPEN_SIDES,
) -> CudaMomentumAdvection5Launch:
    """Enqueue the selected full physical-domain momentum-advection slice.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)`` of the run: advu
    advances faces ``i1..i2`` and advv rows ``j1..j2`` of that family, and the
    inflow edits run on its open sides only (:func:`momentum_open_boundary_code`).
    A run's caller passes the sides its bound ``bcu``/``bcv``/``bcw`` fills
    apply, so the advection and the halo it reads cannot disagree.

    ``post_divergence_hook`` is the one seam CM1 opens between ``divx`` and
    ``advu``: ``solve2.F:631-720`` puts the ``icor=1`` Coriolis term exactly
    there, after the divergence at ``solve2.F:490-627`` and before the three
    advection calls at ``solve2.F:883-892``.  An operator enqueued here is on
    this stream, in that order, or it is not enqueued at all.
    """

    validate_momentum_advection5_selection(selection)
    lateral_sides = momentum_lateral_sides(lateral_sides)
    obc = momentum_open_boundary_code(lateral_sides)
    if rrw_boundary_hook is not None and not callable(rrw_boundary_hook):
        raise TypeError("rrw_boundary_hook must be callable or None")
    if post_divergence_hook is not None and not callable(post_divergence_hook):
        raise TypeError("post_divergence_hook must be callable or None")
    ni = _positive_int("ni", ni, 7)
    nj = _positive_int("nj", nj, 7)
    nk = _positive_int("nk", nk, 8)
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    dx, dy, dz = (_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz)))
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    shapes = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w3d": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "wten": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "uh": (ni + 2 * ng,),
        "vh": (nj + 2 * ng,),
    }
    writable = {"uten", "vten", "wten", "rru", "rrv", "rrw"}
    a = {name: _require_cuda(cupy, arrays[name], name, shape, writable=name in writable) for name, shape in shapes.items()}
    metric_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    metric_names = ("rho0_3d", "rf0", "rr0", "rrf0", "c1", "c2", "gz", "gzu", "gzv", "rgzu", "rgzv", "dzdx", "dzdy")
    m = {name: _require_cuda(cupy, metrics[name], name, metric_shapes[name], writable=False) for name in metric_names}
    if divx is None:
        divx = cupy.empty((nk, nj, ni), dtype=cupy.float32)
    else:
        divx = _require_cuda(cupy, divx, "divx", (nk, nj, ni), writable=True)
    from .cuda_immutable_metrics import require_cuda_argument_aliases
    require_cuda_argument_aliases(
        {**{f"state.{name}": value for name, value in a.items()},
         **{f"metric.{name}": value for name, value in m.items()}, "divx": divx},
        immutable_metrics=metrics,
        message="momentum-advection mutable arrays and unproved metrics must not alias",
    )
    compiled = compiled or compile_cuda_momentum_advection5()
    stream = stream or cupy.cuda.Stream(non_blocking=True)
    horizontal, vertical, divergence, advu, advv, advw = compiled.kernels
    counts = (
        nk * nj * (ni + 1) + nk * (nj + 1) * ni,
        (nk + 1) * nj * ni,
        nk * nj * ni,
        *momentum_advection5_counts(ni, nj, nk, lateral_sides),
    )
    block = (CUDA_MOMENTUM_ADVECTION5_THREADS, 1, 1)
    grids = tuple(((count + CUDA_MOMENTUM_ADVECTION5_THREADS - 1) // CUDA_MOMENTUM_ADVECTION5_THREADS, 1, 1) for count in counts)
    u32, f32 = cupy.uint32, cupy.float32
    with stream:
        horizontal(grids[0], block, (a["u3d"], a["v3d"], m["rho0_3d"], m["rgzu"], m["rgzv"], a["rru"], a["rrv"], u32(ni), u32(nj), u32(nk), u32(ng)), stream=stream)
        vertical(grids[1], block, (a["w3d"], m["rf0"], m["dzdx"], m["dzdy"], m["gz"], a["rru"], a["rrv"], a["rrw"], u32(ni), u32(nj), u32(nk), u32(ng), f32(dz)), stream=stream)
        rrw_boundary_receipt = None
        if rrw_boundary_hook is not None:
            rrw_boundary_receipt = rrw_boundary_hook(a["rrw"], stream=stream, ni=ni, nj=nj, nk=nk)
            from .cuda_sound3_open_boundary import CUDA_SOUND3_OPEN_SOURCE_SHA256
            if (
                not isinstance(rrw_boundary_receipt, Mapping)
                or rrw_boundary_receipt.get("performed") is not True
                or rrw_boundary_receipt.get("flag") != 0
                or rrw_boundary_receipt.get("vertical_rigid_faces_written") is not False
                or rrw_boundary_receipt.get("stream_pointer") != int(stream.ptr)
                # Without these a stub returning the four scalars above would
                # pass while launching the wrong kernels, or nothing at all.
                or rrw_boundary_receipt.get("schema") != "gm1.cuda-bcw-rrw-open-flag0/v1"
                or list(rrw_boundary_receipt.get("directions") or ())
                != ["west", "east", "south", "north"]
                or rrw_boundary_receipt.get("source_sha256") != CUDA_SOUND3_OPEN_SOURCE_SHA256
            ):
                raise ValueError("bcw(rrw,0) hook did not confirm source-ordered same-stream lateral-only filling")
        divergence(grids[2], block, (a["rru"], a["rrv"], a["rrw"], divx, u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(dz), f32(CM1_SMEPS_F32)), stream=stream)
        post_divergence_receipt = None
        if post_divergence_hook is not None:
            post_divergence_receipt = post_divergence_hook(a, stream=stream, ni=ni, nj=nj, nk=nk, nrk=nrk, ng=ng)
            if not isinstance(post_divergence_receipt, Mapping):
                raise TypeError("post_divergence_hook must return a receipt mapping")
        advu(grids[3], block, (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzu"], a["vh"], a["uten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(dz), u32(obc)), stream=stream)
        advv(grids[4], block, (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"], m["c2"], m["gzv"], a["uh"], a["vten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(dz), u32(obc)), stream=stream)
        advw(grids[5], block, (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"], m["rrf0"], m["gz"], a["uh"], a["vh"], a["wten"], u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(dz), u32(obc)), stream=stream)
        event = cupy.cuda.Event()
        event.record(stream)
    return CudaMomentumAdvection5Launch(
        divx,
        event,
        compiled,
        {
            "schema": CUDA_MOMENTUM_ADVECTION5_SCHEMA,
            "abi": CUDA_MOMENTUM_ADVECTION5_ABI,
            "source_sha256": CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256,
            "authority": {
                "src/adv.F": CM1_ADV_F_SHA256,
                "src/adv_routines.F": CM1_ADV_ROUTINES_F_SHA256,
                "src/solve2.F": CM1_SOLVE2_F_SHA256,
                "anchors": dict(SOURCE_ANCHORS),
            },
            "selection": {
                "hadvordrv": 5,
                "vadvordrv": 5,
                "advwenov": 0,
                "axisymmetric": 0,
                "terrain": True,
                "idiff": 0,
                "immersed_boundary": False,
                "uniform_horizontal_grid": True,
                "uniform_vertical_sigma": True,
                "lateral_boundaries": list(lateral_sides),
                "nrk": nrk,
            },
            "boundary_faces": momentum_boundary_faces(lateral_sides),
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "target_arch": compiled.target_arch,
            "runtime": dict(compiled.runtime_identity),
            **({"rrw_open_boundary": dict(rrw_boundary_receipt)} if rrw_boundary_receipt is not None else {}),
            **({"post_divergence": dict(post_divergence_receipt)} if post_divergence_receipt is not None else {}),
            "claim_limit": CUDA_MOMENTUM_ADVECTION5_CLAIM_LIMIT,
        },
    )


__all__ = [
    "CM1_ADV_F_SHA256",
    "CM1_ADV_ROUTINES_F_SHA256",
    "CM1_SOLVE2_F_SHA256",
    "CUDA_MOMENTUM_ADVECTION5_ABI",
    "CUDA_MOMENTUM_ADVECTION5_CLAIM_LIMIT",
    "CUDA_MOMENTUM_ADVECTION5_SCHEMA",
    "CUDA_MOMENTUM_ADVECTION5_SOURCE_BYTES",
    "CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256",
    "CompiledMomentumAdvection5Kernels",
    "CudaMomentumAdvection5Launch",
    "DEFAULT_MOMENTUM_ADVECTION5_SELECTION",
    "MOMENTUM_ALL_OPEN_SIDES",
    "MomentumAdvection5Selection",
    "SOURCE_ANCHORS",
    "compile_cuda_momentum_advection5",
    "cpu_terrain_open_momentum_advection5_inplace",
    "launch_cuda_terrain_open_momentum_advection5",
    "momentum_advection5_counts",
    "momentum_boundary_faces",
    "momentum_lateral_sides",
    "momentum_open_boundary_code",
    "validate_momentum_advection5_selection",
]
