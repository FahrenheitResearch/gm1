"""CM1 r21.1 flat serial-periodic ``psolver=7`` acoustic operator.

The implementation is intentionally a narrow source-ordered family: the
modified-compressible acoustic small steps in ``sounde.F`` for a flat,
Cartesian, single-process domain with periodic lateral boundaries and rigid
vertical ``w`` faces.  It exposes one-stage and held-input three-stage APIs.
It does not derive ``nsound`` or compose the surrounding large-step physics.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from .errors import refuse


CUDA_PSOLVER7_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-psolver7-periodic/v1"
CUDA_PSOLVER7_PERIODIC_ABI = "gpuwm-cm1.cuda-psolver7-periodic-kernels/v1"
CUDA_PSOLVER7_PERIODIC_THREADS = 128
CM1_PSOLVER7_CP_F32 = 1005.7
CM1_PSOLVER7_KDIV_F32 = 0.10
CM1_PSOLVER7_SMEPS_F32 = 1.0e-30
CM1_PSOLVER7_CLAIM_LIMIT = (
    "CM1 r21.1 sounde.F flat Cartesian serial-periodic psolver=7 acoustic "
    "small-step operator with supplied nsound, held tendencies/base arrays, "
    "source-ordered pressure filter, acoustic velocity averages, periodic "
    "pressure/velocity cadence, and rigid vertical w faces; no adaptive "
    "nsound derivation, terrain, MPI, axisymmetry, open or rigid lateral "
    "boundaries, immersed boundaries, convergence/updraft forcing, solve3 "
    "completion, coupled forecast, meteorological-product, or performance claim"
)

SOURCE_ANCHORS = {
    "dispatch": "src/solve2.F:1100-1117",
    "schedule": "src/sounde.F:78-108",
    "stage_prepare": "src/sounde.F:171-229",
    "flat_cartesian_uv": "src/sounde.F:399-425",
    "flat_w": "src/sounde.F:589-606",
    "flat_pressure": "src/sounde.F:863-895",
    "small_step_pressure_fill": "src/sounde.F:958-961",
    "velocity_average": "src/sounde.F:973-1000",
    "final_filter_commit": "src/sounde.F:1009-1020",
    "stage_boundaries": "src/solve2.F:1662-1689",
    "periodic_pressure": "src/bc.F:372-520",
    "rigid_vertical_w": "src/bc.F:1123-1137",
}


@dataclass(frozen=True, slots=True)
class Psolver7PeriodicSelection:
    """Exact admitted namelist/build envelope for this operator family."""

    psolver: int = 7
    terrain_flag: bool = False
    axisymm: int = 0
    mpi: bool = False
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    eqtset: int = 1
    apmasscon: int = 0
    do_ib: bool = False
    convinit: int = 0
    wnudge: int = 0


def require_psolver7_periodic_selection(
    selection: Psolver7PeriodicSelection,
) -> None:
    """Fail closed unless *selection* is exactly inside the admitted family."""

    checks = (
        ("psolver", selection.psolver, 7, "ARWEN-PS7-PSOLVER"),
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-PS7-TERRAIN"),
        ("axisymm", selection.axisymm, 0, "ARWEN-PS7-AXISYMM"),
        ("mpi", selection.mpi, False, "ARWEN-PS7-MPI"),
        ("wbc", selection.wbc, 1, "ARWEN-PS7-WBC"),
        ("ebc", selection.ebc, 1, "ARWEN-PS7-EBC"),
        ("sbc", selection.sbc, 1, "ARWEN-PS7-SBC"),
        ("nbc", selection.nbc, 1, "ARWEN-PS7-NBC"),
        ("irbc", selection.irbc, 4, "ARWEN-PS7-IRBC"),
        ("roflux", selection.roflux, 0, "ARWEN-PS7-ROFLUX"),
        ("nudgeobc", selection.nudgeobc, 0, "ARWEN-PS7-NUDGE-OBC"),
        ("eqtset", selection.eqtset, 1, "ARWEN-PS7-EQTSET"),
        ("apmasscon", selection.apmasscon, 0, "ARWEN-PS7-APMASSCON"),
        ("do_ib", selection.do_ib, False, "ARWEN-PS7-IMMERSED"),
        ("convinit", selection.convinit, 0, "ARWEN-PS7-CONVINIT"),
        ("wnudge", selection.wnudge, 0, "ARWEN-PS7-WNUDGE"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="psolver7-periodic-selection",
                reason=(
                    f"the bounded psolver=7 family requires {option}={expected!r}; "
                    f"the official alternate branch is not implemented here"
                ),
                remedy="select the exact flat serial-periodic family or use a separately ratified operator",
                option=option,
                value=value,
            )


SCALAR_NAMES = (
    "mh",
    "mf",
    "rho0",
    "rf0",
    "c1",
    "c2",
    "thv0",
    "ppi",
    "ppx",
    "pp3d",
    "ppd",
    "pp_filter",
    "div",
)
U_NAMES = ("ua", "u3d", "uten", "rru")
V_NAMES = ("va", "v3d", "vten", "rrv")
W_NAMES = ("wa", "w3d", "wten", "rrw")
METRIC_NAMES = ("dtu0", "dtv0", "uf", "vf", "uh", "vh")
ALL_ARRAY_NAMES = SCALAR_NAMES + U_NAMES + V_NAMES + W_NAMES + METRIC_NAMES
WRITABLE_NAMES = ("ppx", "pp3d", "ppd", "pp_filter", "div", "u3d", "rru", "v3d", "rrv", "w3d", "rrw")
COMPARE_OUTPUT_NAMES = WRITABLE_NAMES


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")


def scalar_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 2, ni + 2


def u_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 2, ni + 1


def v_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 1, ni + 2


def w_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj, ni


def psolver7_periodic_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    scalar = scalar_shape_zyx(ni, nj, nk)
    result = {name: scalar for name in SCALAR_NAMES}
    result.update({name: u_shape_zyx(ni, nj, nk) for name in U_NAMES})
    result.update({name: v_shape_zyx(ni, nj, nk) for name in V_NAMES})
    result.update({name: w_shape_zyx(ni, nj, nk) for name in W_NAMES})
    result.update(
        {
            "dtu0": (nj + 2, ni + 1),
            "dtv0": (nj + 1, ni + 2),
            "uf": (ni + 2,),
            "vf": (nj + 2,),
            "uh": (ni + 2,),
            "vh": (nj + 2,),
        }
    )
    return result


def default_periodic_face_weights(ni: int, nj: int) -> tuple[Any, Any]:
    np = importlib.import_module("numpy")
    _validate_dimensions(ni, nj, 3)
    dtu0 = np.zeros((nj + 2, ni + 1), dtype=np.float32)
    dtv0 = np.zeros((nj + 1, ni + 2), dtype=np.float32)
    dtu0[1 : nj + 1, :] = np.float32(1.0)
    dtv0[:, 1 : ni + 1] = np.float32(1.0)
    return dtu0, dtv0


def fill_periodic_scalar_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
        raise TypeError("periodic scalar must be a float32 numpy.ndarray")
    if value.shape != scalar_shape_zyx(ni, nj, nk) or not value.flags.c_contiguous:
        raise ValueError("periodic scalar has an invalid shape or layout")
    value[1 : nk + 1, :, 0] = value[1 : nk + 1, :, ni]
    value[1 : nk + 1, :, ni + 1] = value[1 : nk + 1, :, 1]
    value[1 : nk + 1, 0, :] = value[1 : nk + 1, nj, :]
    value[1 : nk + 1, nj + 1, :] = value[1 : nk + 1, 1, :]


def fill_periodic_uv_inplace(u: Any, v: Any, *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    for name, value, shape in (
        ("u", u, u_shape_zyx(ni, nj, nk)),
        ("v", v, v_shape_zyx(ni, nj, nk)),
    ):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be a float32 numpy.ndarray")
        if value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} has an invalid shape or layout")
    u[1 : nk + 1, 0, :] = u[1 : nk + 1, nj, :]
    u[1 : nk + 1, nj + 1, :] = u[1 : nk + 1, 1, :]
    v[1 : nk + 1, :, 0] = v[1 : nk + 1, :, ni]
    v[1 : nk + 1, :, ni + 1] = v[1 : nk + 1, :, 1]


def _require_host_arrays(arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = psolver7_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver7 arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must have C-contiguous shape {shapes[name]}")
        if name in WRITABLE_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} contains a non-finite value")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("psolver7 arrays must not alias")
    return result


def _bits_equal(left: Any, right: Any) -> bool:
    np = importlib.import_module("numpy")
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def validate_psolver7_periodic_contract(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    expected_u, expected_v = default_periodic_face_weights(ni, nj)
    if not _bits_equal(a["dtu0"], expected_u) or not _bits_equal(a["dtv0"], expected_v):
        raise ValueError("dtu0/dtv0 differ from CM1 serial-periodic default weights")
    for name in ("uf", "vf", "uh", "vh"):
        if not _bits_equal(a[name], np.ones_like(a[name])):
            raise ValueError(f"{name} must be exactly one for the uniform-grid family")
    for name in ("mh", "mf", "rho0", "rf0", "thv0"):
        if bool((a[name][1 : nk + 1, 1 : nj + 1, 1 : ni + 1] <= 0).any()):
            raise ValueError(f"{name} must be positive on physical scalar cells")
    for name in ("mh", "mf", "rho0", "rf0", "c1", "c2", "thv0"):
        value = a[name][:, 1 : nj + 1, 1 : ni + 1]
        reference = np.broadcast_to(value[:, :1, :1], value.shape)
        if not _bits_equal(value, reference):
            raise ValueError(f"{name} must be horizontally homogeneous for the flat base-state family")
    for name in ("ppi", "ppx", "pp3d", "ppd", "thv0", "rho0", "rf0", "c1", "c2", "mh"):
        value = a[name]
        if not _bits_equal(value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]):
            raise ValueError(f"{name} west periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]):
            raise ValueError(f"{name} east periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]):
            raise ValueError(f"{name} south periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]):
            raise ValueError(f"{name} north periodic halo is invalid")
    for name in ("ua", "u3d", "uten"):
        if not _bits_equal(a[name][1 : nk + 1, 1 : nj + 1, 0], a[name][1 : nk + 1, 1 : nj + 1, ni]):
            raise ValueError(f"{name} duplicate periodic x faces differ")
    for name in ("va", "v3d", "vten"):
        if not _bits_equal(a[name][1 : nk + 1, 0, 1 : ni + 1], a[name][1 : nk + 1, nj, 1 : ni + 1]):
            raise ValueError(f"{name} duplicate periodic y faces differ")


def compare_psolver7_periodic_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    fields: dict[str, Any] = {}
    for name in COMPARE_OUTPUT_NAMES:
        if name not in expected or name not in candidate:
            raise ValueError(f"missing comparison output {name}")
        left, right = expected[name], candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison values must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32") or left.shape != right.shape:
            raise ValueError(f"{name} comparison dtype or shape differs")
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
        "passed": all(item["exact"] for item in fields.values()),
        "fields": fields,
    }


def _a(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _s(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _m(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


def _d(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) / np.float32(y))


@dataclass(frozen=True, slots=True)
class CpuPsolver7StageDiagnostics:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    pressure_boundary_fills: int
    velocity_boundary_fills: int
    stage_pressure_boundary_fills: int
    vertical_w_boundary_fills: int
    ppx_written: bool
    boundary_trace_sha256: str


def cpu_psolver7_periodic_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    csound: float,
    cp: float = CM1_PSOLVER7_CP_F32,
    kdiv: float = CM1_PSOLVER7_KDIV_F32,
    smeps: float = CM1_PSOLVER7_SMEPS_F32,
    selection: Psolver7PeriodicSelection = Psolver7PeriodicSelection(),
) -> CpuPsolver7StageDiagnostics:
    """Execute one source-ordered ``sounde`` psolver=7 RK/acoustic stage."""

    require_psolver7_periodic_selection(selection)
    if not math.isfinite(csound) or csound <= 0:
        raise ValueError("csound must be finite and positive")
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts, rdx, rdy, rdz = map(np.float32, (dts_value, rdx, rdy, rdz))
    csound, cp, kdiv, smeps = map(np.float32, (csound, cp, kdiv, smeps))
    a["ppd"][1 : nk + 1] = a["ppx"][1 : nk + 1]
    a["rru"].fill(np.float32(0.0))
    a["rrv"].fill(np.float32(0.0))
    a["rrw"].fill(np.float32(0.0))
    a["pp_filter"].fill(np.float32(0.0))
    a["div"].fill(np.float32(0.0))
    if nrk != 1:
        np.copyto(a["u3d"], a["ua"])
        np.copyto(a["v3d"], a["va"])
        np.copyto(a["w3d"], a["wa"])
        a["pp3d"][1 : nk + 1] = a["ppi"][1 : nk + 1]
    boundary_trace = hashlib.sha256()
    for n in range(1, nloop + 1):
        for k in range(1, nk + 1):
            tem1 = _m(np, _m(np, rdx, cp), a["thv0"][k, 1, 1])
            tem2 = _m(np, _m(np, rdy, cp), a["thv0"][k, 1, 1])
            for j in range(1, nj + 2):
                for f in range(ni + 1):
                    pg = _m(np, tem1, _s(np, a["ppd"][k, j, f + 1], a["ppd"][k, j, f]))
                    pg = _m(np, pg, a["uf"][f + 1])
                    tendency = _s(np, a["uten"][k, j, f], pg)
                    weight = _m(np, dts, a["dtu0"][j, f])
                    a["u3d"][k, j, f] = _a(np, a["u3d"][k, j, f], _m(np, weight, tendency))
            for f in range(nj + 1):
                for i in range(1, ni + 2):
                    pg = _m(np, tem2, _s(np, a["ppd"][k, f + 1, i], a["ppd"][k, f, i]))
                    pg = _m(np, pg, a["vf"][f + 1])
                    tendency = _s(np, a["vten"][k, f, i], pg)
                    weight = _m(np, dts, a["dtv0"][f, i])
                    a["v3d"][k, f, i] = _a(np, a["v3d"][k, f, i], _m(np, weight, tendency))
        for k in range(2, nk + 1):
            vertical_thv = _a(
                np,
                _m(np, a["c2"][k, 1, 1], a["thv0"][k, 1, 1]),
                _m(np, a["c1"][k, 1, 1], a["thv0"][k - 1, 1, 1]),
            )
            tem1 = _m(np, _m(np, _m(np, rdz, a["mf"][k, 1, 1]), cp), vertical_thv)
            for j0 in range(nj):
                for i0 in range(ni):
                    pg = _m(np, tem1, _s(np, a["ppd"][k, j0 + 1, i0 + 1], a["ppd"][k - 1, j0 + 1, i0 + 1]))
                    tendency = _s(np, a["wten"][k, j0, i0], pg)
                    a["w3d"][k, j0, i0] = _a(np, a["w3d"][k, j0, i0], _m(np, dts, tendency))
        for k in range(1, nk + 1):
            upper_thv = _a(
                np,
                _m(np, a["c2"][k + 1, 1, 1], a["thv0"][k + 1, 1, 1]),
                _m(np, a["c1"][k + 1, 1, 1], a["thv0"][k, 1, 1]),
            )
            lower_thv = _a(
                np,
                _m(np, a["c2"][k, 1, 1], a["thv0"][k, 1, 1]),
                _m(np, a["c1"][k, 1, 1], a["thv0"][k - 1, 1, 1]),
            )
            denom = _m(np, a["rho0"][k, 1, 1], a["thv0"][k, 1, 1])
            r2 = _m(np, a["rf0"][k + 1, 1, 1], upper_thv)
            r2 = _m(np, _m(np, r2, rdz), a["mh"][k, 1, 1])
            r2 = _d(np, r2, denom)
            r1 = _m(np, a["rf0"][k, 1, 1], lower_thv)
            r1 = _m(np, _m(np, r1, rdz), a["mh"][k, 1, 1])
            r1 = _d(np, r1, denom)
            tem = _m(np, _m(np, dts, csound), csound)
            tem = _d(np, tem, _m(np, cp, a["thv0"][k, 1, 1]))
            for j0 in range(nj):
                j = j0 + 1
                for i0 in range(ni):
                    i = i0 + 1
                    du = _m(np, _m(np, _s(np, a["u3d"][k, j, i], a["u3d"][k, j, i - 1]), rdx), a["uh"][i])
                    dv = _m(np, _m(np, _s(np, a["v3d"][k, j, i], a["v3d"][k, j - 1, i]), rdy), a["vh"][j])
                    vertical = _s(np, _m(np, r2, a["w3d"][k + 1, j0, i0]), _m(np, r1, a["w3d"][k, j0, i0]))
                    div = _a(np, _a(np, du, dv), vertical)
                    if bool(np.abs(div) < smeps):
                        div = np.float32(0.0)
                    a["div"][k, j, i] = div
                    old = a["pp3d"][k, j, i]
                    updated = _s(np, old, _m(np, tem, div))
                    if bool(np.abs(updated) < smeps):
                        updated = np.float32(0.0)
                    filt = _m(np, kdiv, _s(np, updated, old))
                    a["pp3d"][k, j, i] = updated
                    a["pp_filter"][k, j, i] = filt
                    a["ppd"][k, j, i] = _a(np, updated, filt)
        if n < nloop:
            fill_periodic_scalar_inplace(a["ppd"], ni=ni, nj=nj, nk=nk)
            boundary_trace.update(b"ppd\0" + n.to_bytes(4, "little"))
            boundary_trace.update(a["ppd"].tobytes(order="C"))
        tavg = _d(np, np.float32(1.0), np.float32(nloop))
        for k in range(1, nk + 1):
            for j in range(1, nj + 2):
                for f in range(ni + 1):
                    value = _a(np, a["rru"][k, j, f], a["u3d"][k, j, f])
                    a["rru"][k, j, f] = value if n < nloop else _m(np, value, tavg)
            for f in range(nj + 1):
                for i in range(1, ni + 2):
                    value = _a(np, a["rrv"][k, f, i], a["v3d"][k, f, i])
                    a["rrv"][k, f, i] = value if n < nloop else _m(np, value, tavg)
            for j0 in range(nj):
                for i0 in range(ni):
                    value = _a(np, a["rrw"][k, j0, i0], a["w3d"][k, j0, i0])
                    a["rrw"][k, j0, i0] = value if n < nloop else _m(np, value, tavg)
    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    a["w3d"][1, :, :] = np.float32(0.0)
    a["w3d"][nk + 1, :, :] = np.float32(0.0)
    boundary_trace.update(b"uvw-stage\0")
    boundary_trace.update(a["u3d"].tobytes(order="C"))
    boundary_trace.update(a["v3d"].tobytes(order="C"))
    boundary_trace.update(a["w3d"].tobytes(order="C"))
    if nrk < 3:
        fill_periodic_scalar_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
        boundary_trace.update(b"pp3d-stage\0")
        boundary_trace.update(a["pp3d"].tobytes(order="C"))
    if nrk == 3:
        a["ppx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1] = a["pp_filter"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    return CpuPsolver7StageDiagnostics(
        nrk=nrk,
        nloop=nloop,
        dts_f32=float(dts),
        dttmp_f32=rk_stage_dt_f32(dt, nrk),
        pressure_boundary_fills=nloop - 1,
        velocity_boundary_fills=1,
        stage_pressure_boundary_fills=1 if nrk < 3 else 0,
        vertical_w_boundary_fills=1,
        ppx_written=nrk == 3,
        boundary_trace_sha256=boundary_trace.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CpuPsolver7Result:
    cadence: tuple[CpuPsolver7StageDiagnostics, ...]


def cpu_psolver7_periodic_three_stage_inplace(
    arrays: MutableMapping[str, Any], **kwargs: Any
) -> CpuPsolver7Result:
    return CpuPsolver7Result(
        tuple(
            cpu_psolver7_periodic_stage_inplace(arrays, nrk=nrk, **kwargs)
            for nrk in (1, 2, 3)
        )
    )


CUDA_PSOLVER7_PERIODIC_SOURCE = r"""// gpuwm-cm1.cuda-psolver7-periodic-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(unsigned long long)==8,"uint64 offsets required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long si(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long ui(unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+1u)+f;}
__device__ __forceinline__ unsigned long long vi(unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+1u)+f)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long wi(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*nj+j)*ni+i;}

extern "C" __global__ void cm1_ps7_prepare_v1(
 const float* ua,const float* va,const float* wa,const float* ppi,const float* ppx,
 float* u,float* v,float* w,float* pp,float* ppd,float* rru,float* rrv,float* rrw,float* pf,float* div,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int nrk){
 unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;
 unsigned long long sn=(unsigned long long)(nk+2u)*(nj+2u)*(ni+2u),un=(unsigned long long)(nk+2u)*(nj+2u)*(ni+1u),vn=(unsigned long long)(nk+2u)*(nj+1u)*(ni+2u),wn=(unsigned long long)(nk+2u)*nj*ni;
 for(unsigned long long q=x;q<sn;q+=step){unsigned int i=q%(ni+2u),t=q/(ni+2u),j=t%(nj+2u),k=t/(nj+2u);pf[q]=0.0f;div[q]=0.0f;if(k>=1u&&k<=nk){ppd[q]=ppx[q];if(nrk!=1u)pp[q]=ppi[q];}}
 for(unsigned long long q=x;q<un;q+=step){rru[q]=0.0f;if(nrk!=1u)u[q]=ua[q];}
 for(unsigned long long q=x;q<vn;q+=step){rrv[q]=0.0f;if(nrk!=1u)v[q]=va[q];}
 for(unsigned long long q=x;q<wn;q+=step){rrw[q]=0.0f;if(nrk!=1u)w[q]=wa[q];}
}

extern "C" __global__ void cm1_ps7_uv_v1(
 const float* thv0,const float* ppd,const float* uten,const float* vten,const float* dtu0,const float* dtv0,const float* uf,const float* vf,
 float* u,float* v,unsigned int ni,unsigned int nj,unsigned int nk,float dts,float rdx,float rdy,float cp){
 unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;
 unsigned long long un=(unsigned long long)nk*(nj+1u)*(ni+1u),vn=(unsigned long long)nk*(nj+1u)*(ni+1u);
 for(unsigned long long q=x;q<un;q+=step){unsigned int f=q%(ni+1u),t=q/(ni+1u),j=t%(nj+1u)+1u,k=t/(nj+1u)+1u;float tem=m32(m32(rdx,cp),thv0[si(k,1u,1u,ni,nj)]);float pg=m32(tem,s32(ppd[si(k,j,f+1u,ni,nj)],ppd[si(k,j,f,ni,nj)]));pg=m32(pg,uf[f+1u]);float ten=s32(uten[ui(k,j,f,ni,nj)],pg);float wt=m32(dts,dtu0[(unsigned long long)j*(ni+1u)+f]);u[ui(k,j,f,ni,nj)]=a32(u[ui(k,j,f,ni,nj)],m32(wt,ten));}
 for(unsigned long long q=x;q<vn;q+=step){unsigned int ii=q%(ni+1u),t=q/(ni+1u),f=t%(nj+1u),k=t/(nj+1u)+1u,i=ii+1u;float tem=m32(m32(rdy,cp),thv0[si(k,1u,1u,ni,nj)]);float pg=m32(tem,s32(ppd[si(k,f+1u,i,ni,nj)],ppd[si(k,f,i,ni,nj)]));pg=m32(pg,vf[f+1u]);float ten=s32(vten[vi(k,f,i,ni,nj)],pg);float wt=m32(dts,dtv0[(unsigned long long)f*(ni+2u)+i]);v[vi(k,f,i,ni,nj)]=a32(v[vi(k,f,i,ni,nj)],m32(wt,ten));}
}

extern "C" __global__ void cm1_ps7_w_v1(
 const float* mf,const float* c1,const float* c2,const float* thv0,const float* ppd,const float* wten,float* w,
 unsigned int ni,unsigned int nj,unsigned int nk,float dts,float rdz,float cp){
 unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)(nk-1u)*nj*ni;
 for(unsigned long long q=x;q<total;q+=step){unsigned int i=q%ni,t=q/ni,j=t%nj,k=t/nj+2u;float vt=a32(m32(c2[si(k,1u,1u,ni,nj)],thv0[si(k,1u,1u,ni,nj)]),m32(c1[si(k,1u,1u,ni,nj)],thv0[si(k-1u,1u,1u,ni,nj)]));float tem=m32(m32(m32(rdz,mf[si(k,1u,1u,ni,nj)]),cp),vt);float pg=m32(tem,s32(ppd[si(k,j+1u,i+1u,ni,nj)],ppd[si(k-1u,j+1u,i+1u,ni,nj)]));float ten=s32(wten[wi(k,j,i,ni,nj)],pg);w[wi(k,j,i,ni,nj)]=a32(w[wi(k,j,i,ni,nj)],m32(dts,ten));}
}

extern "C" __global__ void cm1_ps7_pressure_v1(
 const float* mh,const float* rho0,const float* rf0,const float* c1,const float* c2,const float* thv0,const float* uh,const float* vh,const float* u,const float* v,const float* w,
 float* pp,float* ppd,float* pf,float* div,unsigned int ni,unsigned int nj,unsigned int nk,float dts,float rdx,float rdy,float rdz,float csound,float cp,float kdiv,float smeps){
 unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;
 for(unsigned long long q=x;q<total;q+=step){unsigned int i0=q%ni,t=q/ni,j0=t%nj,k=t/nj+1u,i=i0+1u,j=j0+1u;float up=a32(m32(c2[si(k+1u,1u,1u,ni,nj)],thv0[si(k+1u,1u,1u,ni,nj)]),m32(c1[si(k+1u,1u,1u,ni,nj)],thv0[si(k,1u,1u,ni,nj)]));float lo=a32(m32(c2[si(k,1u,1u,ni,nj)],thv0[si(k,1u,1u,ni,nj)]),m32(c1[si(k,1u,1u,ni,nj)],thv0[si(k-1u,1u,1u,ni,nj)]));float den=m32(rho0[si(k,1u,1u,ni,nj)],thv0[si(k,1u,1u,ni,nj)]);float r2=m32(rf0[si(k+1u,1u,1u,ni,nj)],up);r2=m32(m32(r2,rdz),mh[si(k,1u,1u,ni,nj)]);r2=d32(r2,den);float r1=m32(rf0[si(k,1u,1u,ni,nj)],lo);r1=m32(m32(r1,rdz),mh[si(k,1u,1u,ni,nj)]);r1=d32(r1,den);float tem=m32(m32(dts,csound),csound);tem=d32(tem,m32(cp,thv0[si(k,1u,1u,ni,nj)]));float du=m32(m32(s32(u[ui(k,j,i,ni,nj)],u[ui(k,j,i-1u,ni,nj)]),rdx),uh[i]);float dv=m32(m32(s32(v[vi(k,j,i,ni,nj)],v[vi(k,j-1u,i,ni,nj)]),rdy),vh[j]);float vert=s32(m32(r2,w[wi(k+1u,j0,i0,ni,nj)]),m32(r1,w[wi(k,j0,i0,ni,nj)]));float dd=a32(a32(du,dv),vert);if(fabsf(dd)<smeps)dd=0.0f;unsigned long long z=si(k,j,i,ni,nj);div[z]=dd;float old=pp[z],updated=s32(old,m32(tem,dd));if(fabsf(updated)<smeps)updated=0.0f;float filt=m32(kdiv,s32(updated,old));pp[z]=updated;pf[z]=filt;ppd[z]=a32(updated,filt);}
}

extern "C" __global__ void cm1_ps7_average_v1(float* rru,float* rrv,float* rrw,const float* u,const float* v,const float* w,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int last,float tavg){
 unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,un=(unsigned long long)nk*(nj+1u)*(ni+1u),vn=un,wn=(unsigned long long)nk*nj*ni;
 for(unsigned long long q=x;q<un;q+=step){unsigned int f=q%(ni+1u),t=q/(ni+1u),j=t%(nj+1u)+1u,k=t/(nj+1u)+1u;unsigned long long z=ui(k,j,f,ni,nj);float y=a32(rru[z],u[z]);rru[z]=last?m32(y,tavg):y;}
 for(unsigned long long q=x;q<vn;q+=step){unsigned int ii=q%(ni+1u),t=q/(ni+1u),f=t%(nj+1u),k=t/(nj+1u)+1u,i=ii+1u;unsigned long long z=vi(k,f,i,ni,nj);float y=a32(rrv[z],v[z]);rrv[z]=last?m32(y,tavg):y;}
 for(unsigned long long q=x;q<wn;q+=step){unsigned int i=q%ni,t=q/ni,j=t%nj,k=t/nj+1u;unsigned long long z=wi(k,j,i,ni,nj);float y=a32(rrw[z],w[z]);rrw[z]=last?m32(y,tavg):y;}
}

extern "C" __global__ void cm1_ps7_bcp_v1(float* p,unsigned int ni,unsigned int nj,unsigned int nk){if(blockIdx.x||threadIdx.x)return;for(unsigned int k=1u;k<=nk;++k){for(unsigned int j=0u;j<=nj+1u;++j){p[si(k,j,0u,ni,nj)]=p[si(k,j,ni,ni,nj)];p[si(k,j,ni+1u,ni,nj)]=p[si(k,j,1u,ni,nj)];}for(unsigned int i=0u;i<=ni+1u;++i){p[si(k,0u,i,ni,nj)]=p[si(k,nj,i,ni,nj)];p[si(k,nj+1u,i,ni,nj)]=p[si(k,1u,i,ni,nj)];}}}

extern "C" __global__ void cm1_ps7_stage_bc_v1(float* u,float* v,float* w,float* pp,float* ppx,const float* pf,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int nrk){unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*((ni+1u)+(nj+1u)+ni*nj);for(unsigned long long q=x;q<total;q+=step){unsigned int k=q/((ni+1u)+(nj+1u)+ni*nj)+1u,r=q%((ni+1u)+(nj+1u)+ni*nj);if(r<ni+1u){unsigned int f=r;u[ui(k,0u,f,ni,nj)]=u[ui(k,nj,f,ni,nj)];u[ui(k,nj+1u,f,ni,nj)]=u[ui(k,1u,f,ni,nj)];}else if(r<(ni+1u)+(nj+1u)){unsigned int f=r-(ni+1u);v[vi(k,f,0u,ni,nj)]=v[vi(k,f,ni,ni,nj)];v[vi(k,f,ni+1u,ni,nj)]=v[vi(k,f,1u,ni,nj)];}else if(nrk==3u){unsigned int z=r-(ni+1u)-(nj+1u),j=z/ni,i=z%ni;ppx[si(k,j+1u,i+1u,ni,nj)]=pf[si(k,j+1u,i+1u,ni,nj)];}}
 unsigned long long horiz=(unsigned long long)nj*ni;for(unsigned long long q=x;q<horiz;q+=step){unsigned int j=q/ni,i=q%ni;w[wi(1u,j,i,ni,nj)]=0.0f;w[wi(nk+1u,j,i,ni,nj)]=0.0f;}}
"""

CUDA_PSOLVER7_PERIODIC_SOURCE_BYTES = CUDA_PSOLVER7_PERIODIC_SOURCE.encode("utf-8")
CUDA_PSOLVER7_PERIODIC_SOURCE_SHA256 = (
    "fb1ba68f2567ba73a40f35e0870aa4b1aab9c7a0c77906de3e554c0c998cee93"
)
CUDA_PSOLVER7_KERNEL_NAMES = (
    "cm1_ps7_prepare_v1",
    "cm1_ps7_uv_v1",
    "cm1_ps7_w_v1",
    "cm1_ps7_pressure_v1",
    "cm1_ps7_average_v1",
    "cm1_ps7_bcp_v1",
    "cm1_ps7_stage_bc_v1",
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
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return common + (f"--gpu-architecture=compute_{arch}",), common + (f"--gpu-architecture=sm_{arch}",)


@dataclass(frozen=True, slots=True)
class CompiledPsolver7PeriodicKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_psolver7_periodic() -> CompiledPsolver7PeriodicKernels:
    if hashlib.sha256(CUDA_PSOLVER7_PERIODIC_SOURCE_BYTES).hexdigest() != CUDA_PSOLVER7_PERIODIC_SOURCE_SHA256:
        raise RuntimeError("packaged psolver7 CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_PSOLVER7_PERIODIC_SOURCE, "cm1_psolver7_periodic_v1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_PSOLVER7_PERIODIC_SOURCE, "cm1_psolver7_periodic_v1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_PSOLVER7_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks psolver7 entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_PSOLVER7_KERNEL_NAMES)
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
    return CompiledPsolver7PeriodicKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = psolver7_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver7 device arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("psolver7 device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaPsolver7PeriodicLaunch:
    ready_event: Any
    compiled: CompiledPsolver7PeriodicKernels
    receipt: dict[str, Any]


def launch_cuda_psolver7_periodic_stage(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    csound: float,
    cp: float = CM1_PSOLVER7_CP_F32,
    kdiv: float = CM1_PSOLVER7_KDIV_F32,
    smeps: float = CM1_PSOLVER7_SMEPS_F32,
    selection: Psolver7PeriodicSelection = Psolver7PeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledPsolver7PeriodicKernels | None = None,
) -> CudaPsolver7PeriodicLaunch:
    """Enqueue one stage; the caller may interleave the surrounding RK work."""

    require_psolver7_periodic_selection(selection)
    if not math.isfinite(csound) or csound <= 0:
        raise ValueError("csound must be finite and positive")
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    compiled = compile_cuda_psolver7_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    prep, kuv, kw, kp, kavg, kbcp, kbc = compiled.kernels
    f32, u32 = cupy.float32, cupy.uint32
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
    maximum = max(
        (nk + 2) * (nj + 2) * (ni + 2),
        (nk + 2) * (nj + 2) * (ni + 1),
        (nk + 2) * (nj + 1) * (ni + 2),
        (nk + 2) * nj * ni,
    )
    grid = ((maximum + CUDA_PSOLVER7_PERIODIC_THREADS - 1) // CUDA_PSOLVER7_PERIODIC_THREADS, 1, 1)
    block = (CUDA_PSOLVER7_PERIODIC_THREADS, 1, 1)
    prep(grid, block, (a["ua"], a["va"], a["wa"], a["ppi"], a["ppx"], a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppd"], a["rru"], a["rrv"], a["rrw"], a["pp_filter"], a["div"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    for n in range(1, nloop + 1):
        kuv(grid, block, (a["thv0"], a["ppd"], a["uten"], a["vten"], a["dtu0"], a["dtv0"], a["uf"], a["vf"], a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk), f32(dts), f32(rdx), f32(rdy), f32(cp)), stream=stream)
        kw(grid, block, (a["mf"], a["c1"], a["c2"], a["thv0"], a["ppd"], a["wten"], a["w3d"], u32(ni), u32(nj), u32(nk), f32(dts), f32(rdz), f32(cp)), stream=stream)
        kp(grid, block, (a["mh"], a["rho0"], a["rf0"], a["c1"], a["c2"], a["thv0"], a["uh"], a["vh"], a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppd"], a["pp_filter"], a["div"], u32(ni), u32(nj), u32(nk), f32(dts), f32(rdx), f32(rdy), f32(rdz), f32(csound), f32(cp), f32(kdiv), f32(smeps)), stream=stream)
        if n < nloop:
            kbcp(grid, block, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
        kavg(grid, block, (a["rru"], a["rrv"], a["rrw"], a["u3d"], a["v3d"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(n == nloop), f32(1.0 / nloop)), stream=stream)
    kbc(grid, block, (a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppx"], a["pp_filter"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    if nrk < 3:
        kbcp(grid, block, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream)
    event = cupy.cuda.Event()
    event.record(stream)
    receipt = {
        "schema": CUDA_PSOLVER7_PERIODIC_SCHEMA,
        "abi": CUDA_PSOLVER7_PERIODIC_ABI,
        "source_sha256": CUDA_PSOLVER7_PERIODIC_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "runtime_identity": dict(compiled.runtime_identity),
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": {"nrk": nrk, "nloop": nloop, "dts_f32": float(f32(dts)), "supplied_nsound": nsound},
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "claim_limit": CM1_PSOLVER7_CLAIM_LIMIT,
    }
    return CudaPsolver7PeriodicLaunch(event, compiled, receipt)


def launch_cuda_psolver7_periodic_three_stage(
    arrays: Mapping[str, Any], **kwargs: Any
) -> CudaPsolver7PeriodicLaunch:
    compiled = kwargs.pop("compiled", None)
    stream = kwargs.pop("stream", None)
    last: CudaPsolver7PeriodicLaunch | None = None
    stages: list[dict[str, Any]] = []
    for nrk in (1, 2, 3):
        last = launch_cuda_psolver7_periodic_stage(
            arrays,
            nrk=nrk,
            compiled=compiled,
            stream=stream,
            **kwargs,
        )
        compiled = last.compiled
        stages.append(last.receipt["cadence"])
    assert last is not None
    receipt = dict(last.receipt)
    receipt["cadence"] = stages
    receipt["transition"] = "held-input three-stage wrapper; callers needing live interleaving must use launch_cuda_psolver7_periodic_stage"
    return CudaPsolver7PeriodicLaunch(last.ready_event, last.compiled, receipt)


__all__ = [
    "ALL_ARRAY_NAMES",
    "CM1_PSOLVER7_CLAIM_LIMIT",
    "COMPARE_OUTPUT_NAMES",
    "CUDA_PSOLVER7_PERIODIC_ABI",
    "CUDA_PSOLVER7_PERIODIC_SCHEMA",
    "CUDA_PSOLVER7_PERIODIC_SOURCE_SHA256",
    "CpuPsolver7Result",
    "CpuPsolver7StageDiagnostics",
    "CudaPsolver7PeriodicLaunch",
    "Psolver7PeriodicSelection",
    "compare_psolver7_periodic_raw",
    "compile_cuda_psolver7_periodic",
    "cpu_psolver7_periodic_stage_inplace",
    "cpu_psolver7_periodic_three_stage_inplace",
    "default_periodic_face_weights",
    "fill_periodic_scalar_inplace",
    "fill_periodic_uv_inplace",
    "launch_cuda_psolver7_periodic_stage",
    "launch_cuda_psolver7_periodic_three_stage",
    "psolver7_periodic_shapes",
    "require_psolver7_periodic_selection",
    "validate_psolver7_periodic_contract",
]
