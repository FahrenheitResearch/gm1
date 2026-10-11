"""Bounded CUDA lateral-closure operators for the CM1 r21.1 product lane.

The module covers held upper-Rayleigh tendencies, the Cartesian f-plane
moving-frame source, the ``irbc=4`` phase-speed history update, and a fixed
binary64 dry/total-water mass-neutrality ledger.  Stage wind snapshots are
inputs: this module does not claim to integrate advection, pressure acoustics,
or family-specific boundary halos/corners.

The Coriolis kernel here -- ``cm1_coriolis_moving_f32_v1`` and its CPU
reference ``cpu_coriolis_moving_inplace`` -- is SUPERSEDED and is reached only
by this module's own bounded-fixture harness and its tests.  It covers the
Cartesian f-plane branch of ``solve2.F:639-653`` and nothing else: no beta
plane, no axisymmetric term.  The Cartesian lane's live ``icor=1`` operator is
:mod:`cm1core.cuda_gm1_solve2_rotation`, which covers all three inner
branches, is raw-word tied to :mod:`cm1core.rotation_forcing`, and is enqueued
between ``divx`` and ``advu`` by ``cuda_momentum_advection5``'s
``post_divergence_hook``.  Nothing selects the kernel below, and nothing
should: it is retained only because this module's mass-neutrality and
radiation-history fixtures are sealed against its exact byte identity.
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


CUDA_LATERAL_CLOSURE_SCHEMA = "gpuwm-cm1.cuda-lateral-closure/v1"
CUDA_LATERAL_CLOSURE_ABI = "gpuwm-cm1.cuda-lateral-closure-kernels/v1"
CUDA_LATERAL_CLOSURE_THREADS = 256
CUDA_LATERAL_CLOSURE_CLAIM_LIMIT = (
    "CM1 r21.1 flat Cartesian all-open-edge specialization: selected "
    "irdamp=1 held tendencies, a SUPERSEDED fixture-only f-plane Coriolis "
    "with fixed moving-frame offsets (the live operator is "
    "cuda_gm1_solve2_rotation), irbc=4 phase-speed histories over prescribed "
    "three-stage wind "
    "snapshots, and a deterministic binary64 operator-level mass-neutrality "
    "ledger; no acoustic boundary application, halo/corner fill, nonlinear "
    "apmasscon correction, SGS/TKE, integrated trajectory, or forecast claim"
)

SOURCE_ANCHORS = {
    "rayleigh_profile_semantics": "src/param.F:8652-8671",
    "rayleigh_held": "src/solve1.F:414-426",
    "stage_tendency_reset": "src/solve2.F:460-471",
    "coriolis_moving_frame": "src/solve2.F:639-653",
    "radiation4_call": "src/solve2.F:1261-1269",
    "radiation4_edges": "src/bc.F:2054-2172",
    "mass_reduction": "src/init3d.F:2492-2524;src/solve2.F:1618-1656",
}

FLOAT32_ARRAY_NAMES = (
    "tauh",
    "tauf",
    "ruf",
    "rvf",
    "u0",
    "ua",
    "u_stages",
    "v0",
    "va",
    "v_stages",
    "wa",
    "tha",
    "uten1",
    "vten1",
    "wten1",
    "thten1",
    "uten_stages",
    "vten_stages",
    "radbcw_stages",
    "radbce_stages",
    "radbcs_stages",
    "radbcn_stages",
    "ruh",
    "rvh",
    "rmh",
    "rho",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
)

FLOAT64_ARRAY_NAMES = (
    "dry_layer_before",
    "water_layer_before",
    "mass_totals_before",
    "dry_layer_after",
    "water_layer_after",
    "mass_totals_after",
)

ALL_ARRAY_NAMES = FLOAT32_ARRAY_NAMES + FLOAT64_ARRAY_NAMES


@dataclass(frozen=True, slots=True)
class LateralClosureSelection:
    """Exact selected CM1 branch values admitted by this bounded module."""

    upper_damping: int = 1
    horizontal_rayleigh: int = 0
    coriolis: int = 1
    beta_plane: int = 0
    axisymmetric: int = 0
    moving_frame: int = 1
    west_boundary: int = 2
    east_boundary: int = 2
    south_boundary: int = 2
    north_boundary: int = 2
    radiative_order: int = 4
    rdalpha_s_inv: float = 3.3333333333e-3
    fcor_s_inv: float = 5.0e-5
    u_move_m_s: float = 12.5
    v_move_m_s: float = 3.0
    radiation_speed_limit_m_s: float = 350.0


DEFAULT_LATERAL_CLOSURE_SELECTION = LateralClosureSelection()

_EXPECTED_SELECTION = {
    "closures.upper_damping": ("upper_damping", 1),
    "closures.horizontal_rayleigh": ("horizontal_rayleigh", 0),
    "dynamics.coriolis": ("coriolis", 1),
    "dynamics.beta_plane": ("beta_plane", 0),
    "grid.axisymmetric": ("axisymmetric", 0),
    "closures.moving_frame": ("moving_frame", 1),
    "boundaries.west": ("west_boundary", 2),
    "boundaries.east": ("east_boundary", 2),
    "boundaries.south": ("south_boundary", 2),
    "boundaries.north": ("north_boundary", 2),
    "boundaries.radiative_order": ("radiative_order", 4),
    "closures.upper_damping_rate_s_inv": ("rdalpha_s_inv", 3.3333333333e-3),
    "dynamics.fcor_s_inv": ("fcor_s_inv", 5.0e-5),
    "closures.u_move_m_s": ("u_move_m_s", 12.5),
    "closures.v_move_m_s": ("v_move_m_s", 3.0),
    "boundaries.radiation_speed_limit_m_s": (
        "radiation_speed_limit_m_s",
        350.0,
    ),
}


def _validate_selection(selection: LateralClosureSelection) -> None:
    if not isinstance(selection, LateralClosureSelection):
        raise TypeError("selection must be a LateralClosureSelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if isinstance(expected, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, float) and math.isfinite(value)
        if not valid_type or value != expected:
            refuse(
                code="CM1-LATERAL-CLOSURE-UNSUPPORTED",
                gate="boundaries-closures",
                option=option,
                value=value,
                reason=(
                    f"this operator is measured only for {option}={expected!r}; "
                    "another value would select an unmeasured damping, moving-frame, "
                    "or boundary equation and produce the wrong retained state"
                ),
                remedy="use the sealed official-supercell value or add a separate source oracle",
            )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 4), ("nj", nj, 3), ("nk", nk, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def _validated_positive_f32(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def closure_array_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    """Return the exact compact storage shapes used by this specialization."""

    _validate_grid(ni, nj, nk)
    u = (nk, nj + 2, ni + 3)
    v = (nk, nj + 3, ni + 2)
    scalar = (nk, nj + 2, ni + 2)
    w = (nk + 1, nj + 2, ni + 2)
    physical = (nk, nj, ni)
    return {
        "tauh": (nk,),
        "tauf": (nk + 1,),
        "ruf": (ni + 3,),
        "rvf": (nj + 3,),
        "u0": u,
        "ua": u,
        "u_stages": (3, *u),
        "v0": v,
        "va": v,
        "v_stages": (3, *v),
        "wa": w,
        "tha": scalar,
        "uten1": u,
        "vten1": v,
        "wten1": w,
        "thten1": scalar,
        "uten_stages": (3, *u),
        "vten_stages": (3, *v),
        "radbcw_stages": (3, nk, nj),
        "radbce_stages": (3, nk, nj),
        "radbcs_stages": (3, nk, ni),
        "radbcn_stages": (3, nk, ni),
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": physical,
        "rho": physical,
        "qv": physical,
        "qc": physical,
        "qr": physical,
        "qi": physical,
        "qs": physical,
        "qg": physical,
        "dry_layer_before": (nk,),
        "water_layer_before": (nk,),
        "mass_totals_before": (2,),
        "dry_layer_after": (nk,),
        "water_layer_after": (nk,),
        "mass_totals_after": (2,),
    }


def rk_stage_dt_f32(dt: float, nrk: int) -> float:
    """Return CM1's ``dt/(nrkmax+1-nrk)`` in default-real order."""

    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be exactly 1, 2, or 3")
    np = importlib.import_module("numpy")
    dt32 = np.float32(_validated_positive_f32("dt", dt))
    return float(np.float32(dt32 / np.float32(4 - nrk)))


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = closure_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing lateral-closure arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        expected_dtype = np.dtype("float64" if name in FLOAT64_ARRAY_NAMES else "float32")
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype.name}")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name in {
            "uten1",
            "vten1",
            "wten1",
            "thten1",
            "uten_stages",
            "vten_stages",
            "radbcw_stages",
            "radbce_stages",
            "radbcs_stages",
            "radbcn_stages",
            *FLOAT64_ARRAY_NAMES,
        } and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    return result


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _f32_sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _f32_div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def cpu_rayleigh_held_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: LateralClosureSelection = DEFAULT_LATERAL_CLOSURE_SELECTION,
) -> None:
    """Apply ``solve1.F:414-426`` to the compact flat-grid storage."""

    _validate_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    rdalpha = np.float32(selection.rdalpha_s_inv)
    neg_rate_half = _f32_mul(np, np.float32(-rdalpha), np.float32(0.5))
    neg_rate = np.float32(-rdalpha)
    for k in range(nk):
        tau_pair = _f32_add(np, a["tauh"][k], a["tauh"][k])
        u_coefficient = _f32_mul(np, neg_rate_half, tau_pair)
        v_coefficient = u_coefficient
        w_coefficient = _f32_mul(np, neg_rate, a["tauf"][k])
        th_coefficient = _f32_mul(np, neg_rate, a["tauh"][k])
        for j in range(1, nj + 2):
            for i in range(1, ni + 2):
                a["uten1"][k, j, i] = _f32_mul(
                    np,
                    u_coefficient,
                    _f32_sub(np, a["ua"][k, j, i], a["u0"][k, j, i]),
                )
                a["vten1"][k, j, i] = _f32_mul(
                    np,
                    v_coefficient,
                    _f32_sub(np, a["va"][k, j, i], a["v0"][k, j, i]),
                )
                a["wten1"][k, j, i] = _f32_mul(
                    np, w_coefficient, a["wa"][k, j, i]
                )
                a["thten1"][k, j, i] = _f32_mul(
                    np, th_coefficient, a["tha"][k, j, i]
                )


def cpu_coriolis_moving_inplace(
    u3d: Any,
    v3d: Any,
    uten: Any,
    vten: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: LateralClosureSelection = DEFAULT_LATERAL_CLOSURE_SELECTION,
) -> None:
    """Apply the selected f-plane branch at ``solve2.F:639-653``."""

    _validate_selection(selection)
    np = importlib.import_module("numpy")
    fcor = np.float32(selection.fcor_s_inv)
    umove = np.float32(selection.u_move_m_s)
    vmove = np.float32(selection.v_move_m_s)
    quarter = np.float32(0.25)
    for k in range(nk):
        for j in range(1, nj + 2):
            for i in range(1, ni + 2):
                v_first = _f32_add(np, v3d[k, j, i], v3d[k, j + 1, i])
                v_second = _f32_add(
                    np, v3d[k, j, i - 1], v3d[k, j + 1, i - 1]
                )
                v_interp = _f32_mul(np, quarter, _f32_add(np, v_first, v_second))
                u_increment = _f32_mul(np, fcor, _f32_add(np, v_interp, vmove))
                uten[k, j, i] = _f32_add(np, uten[k, j, i], u_increment)

                u_first = _f32_add(np, u3d[k, j, i], u3d[k, j, i + 1])
                u_second = _f32_add(
                    np, u3d[k, j - 1, i], u3d[k, j - 1, i + 1]
                )
                u_interp = _f32_mul(np, quarter, _f32_add(np, u_first, u_second))
                v_increment = _f32_mul(np, fcor, _f32_add(np, u_interp, umove))
                vten[k, j, i] = _f32_sub(np, vten[k, j, i], v_increment)


def cpu_radiation4_inplace(
    ruf: Any,
    rvf: Any,
    ua: Any,
    u3d: Any,
    va: Any,
    v3d: Any,
    radbcw: Any,
    radbce: Any,
    radbcs: Any,
    radbcn: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dttmp: float,
    selection: LateralClosureSelection = DEFAULT_LATERAL_CLOSURE_SELECTION,
) -> None:
    """Apply ``radbcew4/radbcns4`` in source statement order."""

    _validate_selection(selection)
    np = importlib.import_module("numpy")
    dx32 = np.float32(_validated_positive_f32("dx", dx))
    dy32 = np.float32(_validated_positive_f32("dy", dy))
    dt32 = np.float32(_validated_positive_f32("dttmp", dttmp))
    epsilon = np.float32(1.0e-10)
    zero = np.float32(0.0)
    limit = np.float32(selection.radiation_speed_limit_m_s)
    for j0 in range(nj):
        j = j0 + 1
        avgw = np.float32(0.0)
        avge = np.float32(0.0)
        for k in range(nk):
            foo = _f32_sub(np, ua[k, j, 3], ua[k, j, 2])
            numerator = _f32_mul(np, dx32, ruf[2])
            numerator = _f32_mul(
                np, numerator, _f32_sub(np, ua[k, j, 2], u3d[k, j, 2])
            )
            signed = np.copysign(epsilon, foo).astype(np.float32)
            denominator = _f32_mul(np, dt32, _f32_add(np, signed, foo))
            cbcw = _f32_div(np, numerator, denominator)
            cbcw = np.maximum(np.minimum(cbcw, zero), np.float32(-limit))
            avgw = _f32_add(np, avgw, cbcw)

            foo = _f32_sub(np, ua[k, j, ni], ua[k, j, ni - 1])
            numerator = _f32_mul(np, dx32, ruf[ni])
            numerator = _f32_mul(
                np, numerator, _f32_sub(np, ua[k, j, ni], u3d[k, j, ni])
            )
            signed = np.copysign(epsilon, foo).astype(np.float32)
            denominator = _f32_mul(np, dt32, _f32_add(np, signed, foo))
            cbce = _f32_div(np, numerator, denominator)
            cbce = np.minimum(np.maximum(cbce, zero), limit)
            avge = _f32_add(np, avge, cbce)
        avgw = _f32_div(np, avgw, np.float32(nk))
        avge = _f32_div(np, avge, np.float32(nk))
        radbcw[:, j0] = avgw
        radbce[:, j0] = avge

    for i0 in range(ni):
        i = i0 + 1
        avgs = np.float32(0.0)
        avgn = np.float32(0.0)
        for k in range(nk):
            foo = _f32_sub(np, va[k, 3, i], va[k, 2, i])
            numerator = _f32_mul(np, dy32, rvf[2])
            numerator = _f32_mul(
                np, numerator, _f32_sub(np, va[k, 2, i], v3d[k, 2, i])
            )
            signed = np.copysign(epsilon, foo).astype(np.float32)
            denominator = _f32_mul(np, dt32, _f32_add(np, signed, foo))
            cbcs = _f32_div(np, numerator, denominator)
            cbcs = np.maximum(np.minimum(cbcs, zero), np.float32(-limit))
            avgs = _f32_add(np, avgs, cbcs)

            foo = _f32_sub(np, va[k, nj, i], va[k, nj - 1, i])
            numerator = _f32_mul(np, dy32, rvf[nj])
            numerator = _f32_mul(
                np, numerator, _f32_sub(np, va[k, nj, i], v3d[k, nj, i])
            )
            signed = np.copysign(epsilon, foo).astype(np.float32)
            denominator = _f32_mul(np, dt32, _f32_add(np, signed, foo))
            cbcn = _f32_div(np, numerator, denominator)
            cbcn = np.minimum(np.maximum(cbcn, zero), limit)
            avgn = _f32_add(np, avgn, cbcn)
        avgs = _f32_div(np, avgs, np.float32(nk))
        avgn = _f32_div(np, avgn, np.float32(nk))
        radbcs[:, i0] = avgs
        radbcn[:, i0] = avgn


def cpu_mass_ledger_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    suffix: str,
) -> tuple[float, float]:
    """Evaluate the fixed per-level/global binary64 reduction profile."""

    if suffix not in {"before", "after"}:
        raise ValueError("mass ledger suffix must be 'before' or 'after'")
    np = importlib.import_module("numpy")
    dry_layer = arrays[f"dry_layer_{suffix}"]
    water_layer = arrays[f"water_layer_{suffix}"]
    totals = arrays[f"mass_totals_{suffix}"]
    dry_layer.fill(np.float64(0.0))
    water_layer.fill(np.float64(0.0))
    for k in range(nk):
        dry_sum = np.float64(0.0)
        water_sum = np.float64(0.0)
        for j in range(nj):
            for i in range(ni):
                dry_term = _f32_mul(np, arrays["rho"][k, j, i], arrays["ruh"][i])
                dry_term = _f32_mul(np, dry_term, arrays["rvh"][j])
                dry_term = _f32_mul(np, dry_term, arrays["rmh"][k, j, i])
                qtot = _f32_add(np, arrays["qv"][k, j, i], arrays["qc"][k, j, i])
                for name in ("qr", "qi", "qs", "qg"):
                    qtot = _f32_add(np, qtot, arrays[name][k, j, i])
                water_term = _f32_mul(np, dry_term, qtot)
                dry_sum = np.float64(dry_sum + np.float64(dry_term))
                water_sum = np.float64(water_sum + np.float64(water_term))
        dry_layer[k] = dry_sum
        water_layer[k] = water_sum
    dry_total = np.float64(0.0)
    water_total = np.float64(0.0)
    for k in range(nk):
        dry_total = np.float64(dry_total + dry_layer[k])
        water_total = np.float64(water_total + water_layer[k])
    totals[0] = dry_total
    totals[1] = water_total
    return float(dry_total), float(water_total)


@dataclass(frozen=True, slots=True)
class CpuLateralClosureResult:
    cadence: tuple[tuple[int, float], ...]
    dry_mass_before: float
    dry_mass_after: float
    water_mass_before: float
    water_mass_after: float


def cpu_lateral_closure_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    dx: float,
    dy: float,
    selection: LateralClosureSelection = DEFAULT_LATERAL_CLOSURE_SELECTION,
) -> CpuLateralClosureResult:
    """Execute the bounded source-ordered CPU closure transition."""

    _validate_selection(selection)
    dt = _validated_positive_f32("dt", dt)
    dx = _validated_positive_f32("dx", dx)
    dy = _validated_positive_f32("dy", dy)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    dry_before, water_before = cpu_mass_ledger_inplace(
        a, ni=ni, nj=nj, nk=nk, suffix="before"
    )
    cpu_rayleigh_held_inplace(a, ni=ni, nj=nj, nk=nk, selection=selection)
    cadence = []
    for stage in range(3):
        nrk = stage + 1
        np.copyto(a["uten_stages"][stage], a["uten1"])
        np.copyto(a["vten_stages"][stage], a["vten1"])
        cpu_coriolis_moving_inplace(
            a["u_stages"][stage],
            a["v_stages"][stage],
            a["uten_stages"][stage],
            a["vten_stages"][stage],
            ni=ni,
            nj=nj,
            nk=nk,
            selection=selection,
        )
        dttmp = rk_stage_dt_f32(dt, nrk)
        cpu_radiation4_inplace(
            a["ruf"],
            a["rvf"],
            a["ua"],
            a["u_stages"][stage],
            a["va"],
            a["v_stages"][stage],
            a["radbcw_stages"][stage],
            a["radbce_stages"][stage],
            a["radbcs_stages"][stage],
            a["radbcn_stages"][stage],
            ni=ni,
            nj=nj,
            nk=nk,
            dx=dx,
            dy=dy,
            dttmp=dttmp,
            selection=selection,
        )
        cadence.append((nrk, dttmp))
    dry_after, water_after = cpu_mass_ledger_inplace(
        a, ni=ni, nj=nj, nk=nk, suffix="after"
    )
    return CpuLateralClosureResult(
        cadence=tuple(cadence),
        dry_mass_before=dry_before,
        dry_mass_after=dry_after,
        water_mass_before=water_before,
        water_mass_after=water_after,
    )


CUDA_LATERAL_CLOSURE_KERNEL_NAMES = (
    "cm1_copy_f32_v1",
    "cm1_rayleigh_held_f32_v1",
    "cm1_coriolis_moving_f32_v1",
    "cm1_radiation4_f32_v1",
    "cm1_mass_layers_f64_v1",
    "cm1_mass_totals_f64_v1",
)

CUDA_LATERAL_CLOSURE_SOURCE = r"""// gpuwm-cm1.cuda-lateral-closure-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default REAL requires float32");
static_assert(sizeof(double) == 8, "CM1 mass ledger requires float64");

__device__ __forceinline__ float cm1_add(const float a, const float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float cm1_sub(const float a, const float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float cm1_mul(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float cm1_div(const float a, const float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ unsigned long long cm1_u_index(
    const unsigned int i, const unsigned int j, const unsigned int k,
    const unsigned int ni, const unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj + 2u) + j)
        * (unsigned long long)(ni + 3u) + i;
}

__device__ __forceinline__ unsigned long long cm1_v_index(
    const unsigned int i, const unsigned int j, const unsigned int k,
    const unsigned int ni, const unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj + 3u) + j)
        * (unsigned long long)(ni + 2u) + i;
}

__device__ __forceinline__ unsigned long long cm1_s_index(
    const unsigned int i, const unsigned int j, const unsigned int k,
    const unsigned int ni, const unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj + 2u) + j)
        * (unsigned long long)(ni + 2u) + i;
}

extern "C" __global__ void cm1_copy_f32_v1(
    const float* __restrict__ source, float* __restrict__ destination,
    const unsigned long long count)
{
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long index = first; index < count; index += stride) {
        destination[index] = source[index];
    }
}

extern "C" __global__ void cm1_rayleigh_held_f32_v1(
    const float* __restrict__ tauh, const float* __restrict__ tauf,
    const float* __restrict__ u0, const float* __restrict__ ua,
    const float* __restrict__ v0, const float* __restrict__ va,
    const float* __restrict__ wa, const float* __restrict__ tha,
    float* __restrict__ uten1, float* __restrict__ vten1,
    float* __restrict__ wten1, float* __restrict__ thten1,
    const unsigned int ni, const unsigned int nj, const unsigned int nk,
    const float rdalpha)
{
    const unsigned long long count = (unsigned long long)(ni + 1u) * (nj + 1u) * nk;
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i0 = (unsigned int)(linear % (ni + 1u));
        const unsigned long long jk = linear / (ni + 1u);
        const unsigned int j0 = (unsigned int)(jk % (nj + 1u));
        const unsigned int k = (unsigned int)(jk / (nj + 1u));
        const unsigned int i = i0 + 1u;
        const unsigned int j = j0 + 1u;
        const unsigned long long ui = cm1_u_index(i, j, k, ni, nj);
        const unsigned long long vi = cm1_v_index(i, j, k, ni, nj);
        const unsigned long long si = cm1_s_index(i, j, k, ni, nj);
        float coefficient = cm1_mul(-rdalpha, 0.5f);
        coefficient = cm1_mul(coefficient, cm1_add(tauh[k], tauh[k]));
        uten1[ui] = cm1_mul(coefficient, cm1_sub(ua[ui], u0[ui]));
        vten1[vi] = cm1_mul(coefficient, cm1_sub(va[vi], v0[vi]));
        wten1[si] = cm1_mul(cm1_mul(-rdalpha, tauf[k]), wa[si]);
        thten1[si] = cm1_mul(cm1_mul(-rdalpha, tauh[k]), tha[si]);
    }
}

extern "C" __global__ void cm1_coriolis_moving_f32_v1(
    const float* __restrict__ u3d, const float* __restrict__ v3d,
    float* __restrict__ uten, float* __restrict__ vten,
    const unsigned int ni, const unsigned int nj, const unsigned int nk,
    const float fcor, const float umove, const float vmove)
{
    const unsigned long long count = (unsigned long long)(ni + 1u) * (nj + 1u) * nk;
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i0 = (unsigned int)(linear % (ni + 1u));
        const unsigned long long jk = linear / (ni + 1u);
        const unsigned int j0 = (unsigned int)(jk % (nj + 1u));
        const unsigned int k = (unsigned int)(jk / (nj + 1u));
        const unsigned int i = i0 + 1u;
        const unsigned int j = j0 + 1u;
        const unsigned long long ui = cm1_u_index(i, j, k, ni, nj);
        const unsigned long long vi = cm1_v_index(i, j, k, ni, nj);
        float first_pair = cm1_add(
            v3d[cm1_v_index(i, j, k, ni, nj)],
            v3d[cm1_v_index(i, j + 1u, k, ni, nj)]);
        float second_pair = cm1_add(
            v3d[cm1_v_index(i - 1u, j, k, ni, nj)],
            v3d[cm1_v_index(i - 1u, j + 1u, k, ni, nj)]);
        float interpolated = cm1_mul(0.25f, cm1_add(first_pair, second_pair));
        uten[ui] = cm1_add(uten[ui], cm1_mul(fcor, cm1_add(interpolated, vmove)));

        first_pair = cm1_add(
            u3d[cm1_u_index(i, j, k, ni, nj)],
            u3d[cm1_u_index(i + 1u, j, k, ni, nj)]);
        second_pair = cm1_add(
            u3d[cm1_u_index(i, j - 1u, k, ni, nj)],
            u3d[cm1_u_index(i + 1u, j - 1u, k, ni, nj)]);
        interpolated = cm1_mul(0.25f, cm1_add(first_pair, second_pair));
        vten[vi] = cm1_sub(vten[vi], cm1_mul(fcor, cm1_add(interpolated, umove)));
    }
}

extern "C" __global__ void cm1_radiation4_f32_v1(
    const float* __restrict__ ruf, const float* __restrict__ rvf,
    const float* __restrict__ ua, const float* __restrict__ u3d,
    const float* __restrict__ va, const float* __restrict__ v3d,
    float* __restrict__ radbcw, float* __restrict__ radbce,
    float* __restrict__ radbcs, float* __restrict__ radbcn,
    const unsigned int ni, const unsigned int nj, const unsigned int nk,
    const float dx, const float dy, const float stage_dt, const float speed_limit)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1.0e-10f;
    if (index < nj) {
        const unsigned int j = index + 1u;
        float avgw = 0.0f;
        float avge = 0.0f;
        for (unsigned int k = 0u; k < nk; ++k) {
            float foo = cm1_sub(
                ua[cm1_u_index(3u, j, k, ni, nj)],
                ua[cm1_u_index(2u, j, k, ni, nj)]);
            float numerator = cm1_mul(dx, ruf[2u]);
            numerator = cm1_mul(numerator, cm1_sub(
                ua[cm1_u_index(2u, j, k, ni, nj)],
                u3d[cm1_u_index(2u, j, k, ni, nj)]));
            const float signed_epsilon = copysignf(epsilon, foo);
            float denominator = cm1_mul(stage_dt, cm1_add(signed_epsilon, foo));
            float cbcw = cm1_div(numerator, denominator);
            cbcw = fmaxf(fminf(cbcw, 0.0f), -speed_limit);
            avgw = cm1_add(avgw, cbcw);

            foo = cm1_sub(
                ua[cm1_u_index(ni, j, k, ni, nj)],
                ua[cm1_u_index(ni - 1u, j, k, ni, nj)]);
            numerator = cm1_mul(dx, ruf[ni]);
            numerator = cm1_mul(numerator, cm1_sub(
                ua[cm1_u_index(ni, j, k, ni, nj)],
                u3d[cm1_u_index(ni, j, k, ni, nj)]));
            denominator = cm1_mul(stage_dt, cm1_add(copysignf(epsilon, foo), foo));
            float cbce = cm1_div(numerator, denominator);
            cbce = fminf(fmaxf(cbce, 0.0f), speed_limit);
            avge = cm1_add(avge, cbce);
        }
        avgw = cm1_div(avgw, (float)nk);
        avge = cm1_div(avge, (float)nk);
        for (unsigned int k = 0u; k < nk; ++k) {
            radbcw[(unsigned long long)k * nj + index] = avgw;
            radbce[(unsigned long long)k * nj + index] = avge;
        }
    }
    if (index < ni) {
        const unsigned int i = index + 1u;
        float avgs = 0.0f;
        float avgn = 0.0f;
        for (unsigned int k = 0u; k < nk; ++k) {
            float foo = cm1_sub(
                va[cm1_v_index(i, 3u, k, ni, nj)],
                va[cm1_v_index(i, 2u, k, ni, nj)]);
            float numerator = cm1_mul(dy, rvf[2u]);
            numerator = cm1_mul(numerator, cm1_sub(
                va[cm1_v_index(i, 2u, k, ni, nj)],
                v3d[cm1_v_index(i, 2u, k, ni, nj)]));
            float denominator = cm1_mul(stage_dt, cm1_add(copysignf(epsilon, foo), foo));
            float cbcs = cm1_div(numerator, denominator);
            cbcs = fmaxf(fminf(cbcs, 0.0f), -speed_limit);
            avgs = cm1_add(avgs, cbcs);

            foo = cm1_sub(
                va[cm1_v_index(i, nj, k, ni, nj)],
                va[cm1_v_index(i, nj - 1u, k, ni, nj)]);
            numerator = cm1_mul(dy, rvf[nj]);
            numerator = cm1_mul(numerator, cm1_sub(
                va[cm1_v_index(i, nj, k, ni, nj)],
                v3d[cm1_v_index(i, nj, k, ni, nj)]));
            denominator = cm1_mul(stage_dt, cm1_add(copysignf(epsilon, foo), foo));
            float cbcn = cm1_div(numerator, denominator);
            cbcn = fminf(fmaxf(cbcn, 0.0f), speed_limit);
            avgn = cm1_add(avgn, cbcn);
        }
        avgs = cm1_div(avgs, (float)nk);
        avgn = cm1_div(avgn, (float)nk);
        for (unsigned int k = 0u; k < nk; ++k) {
            radbcs[(unsigned long long)k * ni + index] = avgs;
            radbcn[(unsigned long long)k * ni + index] = avgn;
        }
    }
}

extern "C" __global__ void cm1_mass_layers_f64_v1(
    const float* __restrict__ rho, const float* __restrict__ qv,
    const float* __restrict__ qc, const float* __restrict__ qr,
    const float* __restrict__ qi, const float* __restrict__ qs,
    const float* __restrict__ qg, const float* __restrict__ ruh,
    const float* __restrict__ rvh, const float* __restrict__ rmh,
    double* __restrict__ dry_layer, double* __restrict__ water_layer,
    const unsigned int ni, const unsigned int nj, const unsigned int nk)
{
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nk) return;
    double dry_sum = 0.0;
    double water_sum = 0.0;
    for (unsigned int j = 0u; j < nj; ++j) {
        for (unsigned int i = 0u; i < ni; ++i) {
            const unsigned long long index = ((unsigned long long)k * nj + j) * ni + i;
            float dry_term = cm1_mul(rho[index], ruh[i]);
            dry_term = cm1_mul(dry_term, rvh[j]);
            dry_term = cm1_mul(dry_term, rmh[index]);
            float qtot = cm1_add(qv[index], qc[index]);
            qtot = cm1_add(qtot, qr[index]);
            qtot = cm1_add(qtot, qi[index]);
            qtot = cm1_add(qtot, qs[index]);
            qtot = cm1_add(qtot, qg[index]);
            const float water_term = cm1_mul(dry_term, qtot);
            dry_sum = __dadd_rn(dry_sum, (double)dry_term);
            water_sum = __dadd_rn(water_sum, (double)water_term);
        }
    }
    dry_layer[k] = dry_sum;
    water_layer[k] = water_sum;
}

extern "C" __global__ void cm1_mass_totals_f64_v1(
    const double* __restrict__ dry_layer, const double* __restrict__ water_layer,
    double* __restrict__ totals, const unsigned int nk)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    double dry_total = 0.0;
    double water_total = 0.0;
    for (unsigned int k = 0u; k < nk; ++k) {
        dry_total = __dadd_rn(dry_total, dry_layer[k]);
        water_total = __dadd_rn(water_total, water_layer[k]);
    }
    totals[0] = dry_total;
    totals[1] = water_total;
}
"""

CUDA_LATERAL_CLOSURE_SOURCE_BYTES = CUDA_LATERAL_CLOSURE_SOURCE.encode("utf-8")
CUDA_LATERAL_CLOSURE_SOURCE_SHA256 = (
    "5d26281a10aa9fd070f248a115de5f25dca5b431377bfefb9377757de763bee7"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_LATERAL_CLOSURE_SOURCE_BYTES).hexdigest() == (
        CUDA_LATERAL_CLOSURE_SOURCE_SHA256
    )


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        major, minor = text.split(".", 1)
        text = major + minor
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(
    arch: str, nvrtc_major: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return (
        common + (f"--gpu-architecture=compute_{arch}",),
        common + (f"--gpu-architecture=sm_{arch}",),
    )


@dataclass(frozen=True, slots=True)
class CompiledLateralClosureKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_lateral_closure() -> CompiledLateralClosureKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged lateral-closure CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if compiler_arch != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx_program = compiler._NVRTCProgram(
        CUDA_LATERAL_CLOSURE_SOURCE,
        "cm1_lateral_closure_v1_ptx.cu",
        method="ptx",
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_LATERAL_CLOSURE_SOURCE,
        "cm1_lateral_closure_v1_cubin.cu",
        method="cubin",
    )
    cubin, _ = cubin_program.compile(cubin_options)
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")
    elif not isinstance(ptx, bytes):
        ptx = bytes(ptx)
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    elif not isinstance(cubin, bytes):
        cubin = bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_LATERAL_CLOSURE_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more lateral-closure entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled lateral-closure image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_LATERAL_CLOSURE_KERNEL_NAMES)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc_parts[0])}.{int(nvrtc_parts[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledLateralClosureKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = closure_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing lateral-closure arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        expected_dtype = cupy.dtype("float64" if name in FLOAT64_ARRAY_NAMES else "float32")
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype.name}")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("lateral-closure arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaLateralClosureLaunch:
    ready_event: Any
    compiled: CompiledLateralClosureKernels
    retained_events: tuple[Any, ...]
    receipt: dict[str, Any]


def launch_cuda_lateral_closure(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    dx: float,
    dy: float,
    selection: LateralClosureSelection = DEFAULT_LATERAL_CLOSURE_SELECTION,
    stream: Any | None = None,
    compiled: CompiledLateralClosureKernels | None = None,
) -> CudaLateralClosureLaunch:
    """Enqueue the bounded three-stage closure without host synchronization."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dt = _validated_positive_f32("dt", dt)
    dx = _validated_positive_f32("dx", dx)
    dy = _validated_positive_f32("dy", dy)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_lateral_closure()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    u32 = cupy.uint32
    u64 = cupy.uint64
    f32 = cupy.float32
    live_count = (ni + 1) * (nj + 1) * nk
    live_grid = (live_count + CUDA_LATERAL_CLOSURE_THREADS - 1) // CUDA_LATERAL_CLOSURE_THREADS
    edge_grid = (max(ni, nj) + CUDA_LATERAL_CLOSURE_THREADS - 1) // CUDA_LATERAL_CLOSURE_THREADS
    layer_grid = (nk + CUDA_LATERAL_CLOSURE_THREADS - 1) // CUDA_LATERAL_CLOSURE_THREADS
    u_words = nk * (nj + 2) * (ni + 3)
    v_words = nk * (nj + 3) * (ni + 2)
    u_copy_grid = (u_words + CUDA_LATERAL_CLOSURE_THREADS - 1) // CUDA_LATERAL_CLOSURE_THREADS
    v_copy_grid = (v_words + CUDA_LATERAL_CLOSURE_THREADS - 1) // CUDA_LATERAL_CLOSURE_THREADS
    threads = (CUDA_LATERAL_CLOSURE_THREADS, 1, 1)

    def launch_mass(suffix: str) -> None:
        compiled.kernels[4](
            (layer_grid, 1, 1),
            threads,
            (
                a["rho"],
                a["qv"],
                a["qc"],
                a["qr"],
                a["qi"],
                a["qs"],
                a["qg"],
                a["ruh"],
                a["rvh"],
                a["rmh"],
                a[f"dry_layer_{suffix}"],
                a[f"water_layer_{suffix}"],
                u32(ni),
                u32(nj),
                u32(nk),
            ),
            shared_mem=0,
            stream=stream,
        )
        compiled.kernels[5](
            (1, 1, 1),
            threads,
            (
                a[f"dry_layer_{suffix}"],
                a[f"water_layer_{suffix}"],
                a[f"mass_totals_{suffix}"],
                u32(nk),
            ),
            shared_mem=0,
            stream=stream,
        )

    launch_mass("before")
    compiled.kernels[1](
        (live_grid, 1, 1),
        threads,
        (
            a["tauh"],
            a["tauf"],
            a["u0"],
            a["ua"],
            a["v0"],
            a["va"],
            a["wa"],
            a["tha"],
            a["uten1"],
            a["vten1"],
            a["wten1"],
            a["thten1"],
            u32(ni),
            u32(nj),
            u32(nk),
            f32(selection.rdalpha_s_inv),
        ),
        shared_mem=0,
        stream=stream,
    )
    cadence = []
    retained_events = []
    for stage in range(3):
        nrk = stage + 1
        compiled.kernels[0](
            (u_copy_grid, 1, 1),
            threads,
            (a["uten1"], a["uten_stages"][stage], u64(u_words)),
            shared_mem=0,
            stream=stream,
        )
        compiled.kernels[0](
            (v_copy_grid, 1, 1),
            threads,
            (a["vten1"], a["vten_stages"][stage], u64(v_words)),
            shared_mem=0,
            stream=stream,
        )
        compiled.kernels[2](
            (live_grid, 1, 1),
            threads,
            (
                a["u_stages"][stage],
                a["v_stages"][stage],
                a["uten_stages"][stage],
                a["vten_stages"][stage],
                u32(ni),
                u32(nj),
                u32(nk),
                f32(selection.fcor_s_inv),
                f32(selection.u_move_m_s),
                f32(selection.v_move_m_s),
            ),
            shared_mem=0,
            stream=stream,
        )
        dttmp = rk_stage_dt_f32(dt, nrk)
        compiled.kernels[3](
            (edge_grid, 1, 1),
            threads,
            (
                a["ruf"],
                a["rvf"],
                a["ua"],
                a["u_stages"][stage],
                a["va"],
                a["v_stages"][stage],
                a["radbcw_stages"][stage],
                a["radbce_stages"][stage],
                a["radbcs_stages"][stage],
                a["radbcn_stages"][stage],
                u32(ni),
                u32(nj),
                u32(nk),
                f32(dx),
                f32(dy),
                f32(dttmp),
                f32(selection.radiation_speed_limit_m_s),
            ),
            shared_mem=0,
            stream=stream,
        )
        stage_event = cupy.cuda.Event(disable_timing=True)
        stage_event.record(stream)
        retained_events.append(stage_event)
        cadence.append({"nrk": nrk, "dttmp_f32": dttmp})
    launch_mass("after")
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_LATERAL_CLOSURE_SCHEMA,
        "claim_limit": CUDA_LATERAL_CLOSURE_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": cadence,
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "reduction_profile": {
            "id": "cm1-mass-f64-level-serial-global-serial-v1",
            "cell_terms": "source-ordered binary32",
            "per_level": "one CUDA thread per k; serial j then i-inner binary64",
            "global": "one CUDA thread; serial increasing-k binary64",
        },
        "selection": {
            option: getattr(selection, attribute)
            for option, (attribute, _) in _EXPECTED_SELECTION.items()
        },
        "kernel": {
            "abi": CUDA_LATERAL_CLOSURE_ABI,
            "names": list(CUDA_LATERAL_CLOSURE_KERNEL_NAMES),
            "source_sha256": CUDA_LATERAL_CLOSURE_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": CUDA_LATERAL_CLOSURE_THREADS,
            "grid_live_x": live_grid,
            "grid_edge_x": edge_grid,
            "grid_layer_x": layer_grid,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaLateralClosureLaunch(
        ready_event=event,
        compiled=compiled,
        retained_events=tuple(retained_events),
        receipt=receipt,
    )


__all__ = [
    "ALL_ARRAY_NAMES",
    "CUDA_LATERAL_CLOSURE_ABI",
    "CUDA_LATERAL_CLOSURE_CLAIM_LIMIT",
    "CUDA_LATERAL_CLOSURE_KERNEL_NAMES",
    "CUDA_LATERAL_CLOSURE_SCHEMA",
    "CUDA_LATERAL_CLOSURE_SOURCE",
    "CUDA_LATERAL_CLOSURE_SOURCE_SHA256",
    "CompiledLateralClosureKernels",
    "CpuLateralClosureResult",
    "CudaLateralClosureLaunch",
    "DEFAULT_LATERAL_CLOSURE_SELECTION",
    "FLOAT32_ARRAY_NAMES",
    "FLOAT64_ARRAY_NAMES",
    "LateralClosureSelection",
    "SOURCE_ANCHORS",
    "closure_array_shapes",
    "compile_cuda_lateral_closure",
    "cpu_coriolis_moving_inplace",
    "cpu_lateral_closure_inplace",
    "cpu_mass_ledger_inplace",
    "cpu_radiation4_inplace",
    "cpu_rayleigh_held_inplace",
    "launch_cuda_lateral_closure",
    "rk_stage_dt_f32",
]
