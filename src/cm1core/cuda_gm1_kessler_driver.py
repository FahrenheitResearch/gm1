"""Complete device-resident GM1 Kessler path for admitted ``efall=0/1`` selectors.

GM1 is the product name.  CM1 r21.1 is used only as the pinned mathematical
authority.  The native order reproduced here is::

    pdefq(qv) -> pdefq(qc) -> pdefq(qr) -> k_fallout -> geterain
    -> [getcvm -> getefall -> thermodynamic update] -> fallout -> kessler
    -> satadj

The bracketed phase is selected only for ``efall=1``.  The local-conversion
budget kernels remain delegated to :mod:`cuda_gm1_kessler_local`; for the
energy-fallout branch the driver supplies the source-native held temperature
to its otherwise identical cell calculation.

The array key ``rr`` preserves the native Kessler dummy-argument spelling,
but its source binding is ``mp_driver``'s ``dum3`` budget-density snapshot
(equal to ``rho`` for this Cartesian selector), never global reciprocal
density.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import math
import re
from typing import Any

from cm1core.cuda_gm1_kessler_local import (
    CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256,
    CompiledGm1KesslerLocalKernels,
    Gm1KesslerLocalSelection,
    Gm1KesslerLocalWorkspace,
    compile_cuda_gm1_kessler_local,
    cpu_gm1_kessler_local_inplace,
    create_cuda_gm1_kessler_local_workspace,
    launch_cuda_gm1_kessler_local,
)
from cm1core.cuda_gm1_kessler_efall import (
    CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256,
    CompiledGm1KesslerEfallKernels,
    Gm1KesslerEfallSelection,
    Gm1KesslerEfallWorkspace,
    compile_cuda_gm1_kessler_efall,
    cpu_gm1_kessler_efall_inplace,
    create_cuda_gm1_kessler_efall_workspace,
    launch_cuda_gm1_kessler_efall,
)
from cm1core.errors import refuse


CUDA_GM1_KESSLER_DRIVER_SCHEMA = "gm1.cuda-kessler-driver/v1"
CUDA_GM1_KESSLER_DRIVER_ABI = "gm1.cuda-kessler-driver-kernels/v1"
CUDA_GM1_KESSLER_DRIVER_THREADS = 256
CUDA_GM1_KESSLER_DRIVER_KERNEL_NAMES = (
    "gm1_kessler_driver_pre_v1",
    "gm1_kessler_driver_finish_pre_v1",
    "gm1_kessler_driver_fallout_v1",
    "gm1_kessler_driver_finish_fallout_v1",
    "gm1_kessler_driver_local_temperature_v1",
    "gm1_kessler_satadj_cell_v1",
    "gm1_kessler_satadj_level_v1",
    "gm1_kessler_satadj_finish_v1",
)
CUDA_GM1_KESSLER_DRIVER_CLAIM_LIMIT = (
    "Complete source-ordered GM1 ptype=1 warm-rain operator family for the "
    "explicitly admitted Cartesian, single-rain-bin, pdscheme=1, efall=0/1, "
    "nrk=4 selector: positivity repair, rain fall speed/energy budget, fallout, "
    "optional precipitation-energy fallout, local Kessler conversion, and "
    "saturation adjustment. Axisymmetric/moving-domain rain bins, scheduler "
    "coupling, forecasts, and "
    "tornado skill are not claimed."
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/mp_driver.F": "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134",
    "src/misclibs.F": "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1",
    "src/kessler.F": "f232f840e425d8c9abd28841e0a46a06e0dbf641460224fa61b6a82e5a650472",
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
}
SOURCE_ANCHORS = {
    "driver_order": "src/mp_driver.F:99-149,185-232",
    "pdefq": "src/misclibs.F:1037-1117",
    "fall_velocity": "src/kessler.F:368-396",
    "rain_energy": "src/kessler.F:530-562",
    "energy_fallout": "src/mp_driver.F:197-225; src/misclibs.F:975-1029; src/kessler.F:485-522",
    "fallout": "src/kessler.F:404-477",
    "local_conversion": "src/kessler.F:8-138",
    "saturation_adjustment": "src/kessler.F:146-360",
    "constants": "src/constants.F:5-31,61,69-153; src/cm1.F:309-318",
}


@dataclass(frozen=True, slots=True)
class Gm1KesslerDriverSelection:
    ptype: int = 1
    equation_set: int = 1
    testcase: int = 0
    pdscheme: int = 1
    efall: int = 0
    axisymmetric: int = 0
    moving_domain: int = 0
    rain_bins: int = 1
    nrk: int = 4
    input_phase: str = "pre-pdefq"
    output_phase: str = "post-satadj"


DEFAULT_GM1_KESSLER_DRIVER_SELECTION = Gm1KesslerDriverSelection()


def _validate_selection(selection: Gm1KesslerDriverSelection) -> None:
    if not isinstance(selection, Gm1KesslerDriverSelection):
        raise TypeError("selection must be a Gm1KesslerDriverSelection")
    fixed = {
        "ptype": 1,
        "testcase": 0,
        "pdscheme": 1,
        "axisymmetric": 0,
        "moving_domain": 0,
        "rain_bins": 1,
        "nrk": 4,
        "input_phase": "pre-pdefq",
        "output_phase": "post-satadj",
    }
    for name, wanted in fixed.items():
        value = getattr(selection, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="GM1-KESSLER-DRIVER-SELECTION",
                gate="physics-seam",
                option=name,
                value=value,
                reason="the complete Kessler operator is sealed to one exact native branch",
                remedy=f"use {name}={wanted!r} or port and independently validate that native branch",
            )
    if type(selection.efall) is not int or selection.efall not in (0, 1):
        refuse(
            code="GM1-KESSLER-DRIVER-EFALL",
            gate="physics-seam",
            option="efall",
            value=selection.efall,
            reason="the complete driver admits only the native disabled/enabled energy-fallout branches",
            remedy="use efall=0 or efall=1",
        )
    if type(selection.equation_set) is not int or selection.equation_set not in (1, 2):
        refuse(
            code="GM1-KESSLER-DRIVER-EQTSET",
            gate="physics-seam",
            option="equation_set",
            value=selection.equation_set,
            reason="the authority has exactly traditional and energy-conserving branches",
            remedy="use equation_set=1 or equation_set=2",
        )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if ni * nj * nk > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("GM1 Kessler driver word count exceeds signed 64-bit indexing")


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def gm1_kessler_driver_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    mass = (nk, nj, ni)
    return {
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": mass,
        "mh": mass,
        "pi0": mass,
        "th0": mass,
        "rho": mass,
        "rr": mass,
        "pp3d": mass,
        "th3d": mass,
        "prs": mass,
        "qv": mass,
        "qc": mass,
        "qr": mass,
        "fall_velocity": mass,
        "rain": (1, nj, ni),
        "prate": (nj, ni),
        "asq": (3,),
        "qbudget": (10,),
    }


INPUT_NAMES = ("ruh", "rvh", "rmh", "mh", "pi0", "th0", "rr")
INOUT_NAMES = (
    "rho", "pp3d", "th3d", "prs", "qv", "qc", "qr", "fall_velocity",
    "rain", "prate", "asq", "qbudget",
)
ALL_ARRAY_NAMES = INPUT_NAMES + INOUT_NAMES
DOUBLE_NAMES = ("asq", "qbudget")


def _require_host_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int,
    selection: Gm1KesslerDriverSelection,
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = gm1_kessler_driver_shapes(ni, nj, nk)
    required = ALL_ARRAY_NAMES + (("mf",) if selection.efall == 1 else ())
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing GM1 Kessler driver arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in required:
        value = arrays[name]
        dtype = np.dtype("float64") if name in DOUBLE_NAMES else np.dtype("float32")
        shape = (nk + 1, nj, ni) if name == "mf" else shapes[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order {dtype} with shape {shape}")
        if name in INOUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("GM1 Kessler driver arrays must not alias")
    for name in ("rho", "prs", "mh"):
        if bool(np.any(result[name] <= np.float32(0.0))):
            raise ValueError(f"{name} must be > 0")
    if bool(np.any(result["pi0"] + result["pp3d"] <= np.float32(0.0))):
        raise ValueError("pi0 + pp3d must be > 0")
    return result


def _f(np: Any, value: Any) -> Any:
    return np.float32(value)


def _a(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) + _f(np, right))


def _s(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) - _f(np, right))


def _m(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) * _f(np, right))


def _d(np: Any, left: Any, right: Any) -> Any:
    return _f(np, _f(np, left) / _f(np, right))


def _pow(np: Any, base: Any, exponent: Any) -> Any:
    return _f(np, np.power(_f(np, base), _f(np, exponent)))


def _exp(np: Any, value: Any) -> Any:
    return _f(np, np.exp(_f(np, value)))


def _constants(np: Any) -> dict[str, Any]:
    rd = _f(np, 287.04)
    cp = _f(np, 1005.7)
    cpv = _f(np, 1870.0)
    rv = _f(np, 461.5)
    xlv = _f(np, 2501000.0)
    cv = _s(np, cp, rd)
    cvv = _s(np, cpv, rv)
    rovcp = _d(np, rd, cp)
    eps = _d(np, rd, rv)
    reps = _d(np, rv, rd)
    cpl = _f(np, 4190.0)
    cpi = _f(np, 2106.0)
    lv2 = _s(np, cpl, cpv)
    lv1 = _a(np, xlv, _m(np, lv2, _f(np, 273.15)))
    rp00 = _d(np, 1.0, _f(np, 1.0e5))
    return locals()


def _cpu_pdefq(
    np: Any,
    q: Any,
    *,
    rmax: Any,
    a: dict[str, Any],
    ni: int,
    nj: int,
    nk: int,
    volume: float,
    asq_index: int,
) -> None:
    for j in range(nj):
        budj = 0.0
        for i in range(ni):
            t1 = t2 = a1 = a2 = 0.0
            for k in range(nk):
                p = (k, j, i)
                rq = _m(np, a["rho"][p], q[p])
                t1 += float(rq)
                cell = _m(np, _m(np, _m(np, _m(np, a["rho"][p], q[p]), a["ruh"][i]), a["rvh"][j]), a["rmh"][p])
                a1 += float(cell)
                if q[p] < rmax:
                    q[p] = _f(np, 0.0)
                t2 += float(_m(np, a["rho"][p], q[p]))
            t3 = (t1 + 1.0e-20) / (t2 + 1.0e-20)
            if t3 < 0.0:
                t3 = 1.0
            for k in range(nk):
                p = (k, j, i)
                q[p] = _f(np, t3 * float(q[p]))
                cell = _m(np, _m(np, _m(np, _m(np, a["rho"][p], q[p]), a["ruh"][i]), a["rvh"][j]), a["rmh"][p])
                a2 += float(cell)
            budj += a2 - a1
        a["asq"][asq_index] += budj * volume


def _cpu_satadj(
    np: Any,
    a: dict[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: Any,
    dy: Any,
    dz: Any,
    dt: Any,
    equation_set: int,
) -> dict[str, Any]:
    c = _constants(np)
    cond = np.zeros((nk, nj, ni), dtype=np.float32)
    evac = np.zeros((nk, nj, ni), dtype=np.float32)
    iterations = np.zeros((nk, nj, ni), dtype=np.uint32)
    bud1 = np.zeros(nk, dtype=np.float64)
    bud2 = np.zeros(nk, dtype=np.float64)
    converge = _m(np, 2.0, 0.0001)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                old_qv = _f(np, a["qv"][p])
                old_qc = _f(np, a["qc"][p])
                qvnew, qcnew = old_qv, old_qc
                tnew = _m(np, _a(np, a["th0"][p], a["th3d"][p]), _a(np, a["pi0"][p], a["pp3d"][p]))
                esl = _m(np, 611.2, _exp(np, _d(np, _m(np, 17.67, _s(np, tnew, 273.15)), _s(np, tnew, 29.65))))
                esl = min(esl, _m(np, a["prs"][p], 0.5))
                qvs = _d(np, _m(np, c["eps"], esl), _s(np, a["prs"][p], esl))
                if equation_set == 2:
                    if old_qc > _f(np, 1.0e-12) or old_qv > qvs:
                        ql = max(_a(np, old_qc, a["qr"][p]), _f(np, 0.0))
                        cvml = _a(np, _a(np, c["cv"], _m(np, c["cvv"], qvnew)), _m(np, c["cpl"], ql))
                        lhv = _s(np, c["lv1"], _m(np, c["lv2"], tnew))
                        t1 = _d(np, _s(np, lhv, _m(np, c["rv"], tnew)), cvml)
                        d1 = _m(np, _m(np, t1, 17.67), 243.5)
                        tlast = tnew
                        n = 0
                        while True:
                            n += 1
                            denom = _a(np, 1.0, _d(np, _m(np, d1, qvs), _m(np, _s(np, tnew, 29.65), _s(np, tnew, 29.65))))
                            dqv = _d(np, _s(np, qvs, qvnew), denom)
                            dqv = min(dqv, qcnew)
                            if _a(np, qvnew, dqv) < _f(np, 1.0e-20):
                                dqv = _s(np, 1.0e-20, qvnew)
                            qvnew = _a(np, qvnew, dqv)
                            qcnew = _s(np, qcnew, dqv)
                            tnew = _s(np, tnew, _m(np, dqv, t1))
                            pnew = _m(np, _m(np, a["rho"][p], _a(np, c["rd"], _m(np, c["rv"], qvnew))), tnew)
                            converged = abs(float(_s(np, tnew, tlast))) <= float(converge)
                            if not converged:
                                tlast = tnew
                                esl = _m(np, 611.2, _exp(np, _d(np, _m(np, 17.67, _s(np, tnew, 273.15)), _s(np, tnew, 29.65))))
                                esl = min(esl, _m(np, a["prs"][p], 0.5))
                                qvs = _d(np, _m(np, c["eps"], esl), _s(np, pnew, esl))
                            if n == 100:
                                raise RuntimeError("GM1 Kessler satadj failed to converge in 100 iterations")
                            if converged:
                                break
                        iterations[p] = n
                        a["prs"][p] = pnew
                        a["pp3d"][p] = _s(np, _pow(np, _m(np, pnew, c["rp00"]), c["rovcp"]), a["pi0"][p])
                        a["th3d"][p] = _s(np, _d(np, tnew, _a(np, a["pi0"][p], a["pp3d"][p])), a["th0"][p])
                        a["qv"][p], a["qc"][p] = qvnew, qcnew
                else:
                    lhv = _s(np, c["lv1"], _m(np, c["lv2"], tnew))
                    square = _m(np, _s(np, tnew, 29.65), _s(np, tnew, 29.65))
                    numerator = _m(np, _m(np, _m(np, lhv, qvs), 17.67), 243.5)
                    dqv = _d(np, _s(np, qvs, qvnew), _a(np, 1.0, _d(np, numerator, _m(np, c["cp"], square))))
                    if _a(np, qvnew, dqv) < _f(np, 1.0e-20):
                        dqv = _s(np, 1.0e-20, qvnew)
                    dqv = min(dqv, max(_f(np, 0.0), qcnew))
                    qvnew = _a(np, qvnew, dqv)
                    qcnew = _s(np, qcnew, dqv)
                    iterations[p] = 1
                    a["th3d"][p] = _s(np, a["th3d"][p], _m(np, dqv, _d(np, lhv, _m(np, c["cp"], _a(np, a["pi0"][p], a["pp3d"][p])))))
                    a["qv"][p], a["qc"][p] = qvnew, qcnew
                    denominator = _m(np, _m(np, _m(np, c["rd"], _a(np, a["th0"][p], a["th3d"][p])), _a(np, a["pi0"][p], a["pp3d"][p])), _a(np, 1.0, _m(np, qvnew, c["reps"])))
                    a["rho"][p] = _d(np, a["prs"][p], denominator)
                cond[p] = max(_s(np, qcnew, old_qc), _f(np, 0.0))
                evac[p] = max(_s(np, qvnew, old_qv), _f(np, 0.0))
                tem = _m(np, _m(np, a["ruh"][i], a["rvh"][j]), a["rmh"][p])
                bud1[k] += float(_m(np, _m(np, a["rr"][p], cond[p]), tem))
                bud2[k] += float(_m(np, _m(np, a["rr"][p], evac[p]), tem))
    volume = float(_m(np, _m(np, dx, dy), dz))
    for k in range(nk):
        a["qbudget"][0] += bud1[k] * volume
    for k in range(nk):
        a["qbudget"][1] += bud2[k] * volume
    return {"condensation": cond, "cloud_evaporation": evac, "iterations": iterations}


def cpu_gm1_kessler_driver_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerDriverSelection = DEFAULT_GM1_KESSLER_DRIVER_SELECTION,
) -> dict[str, Any]:
    """Apply every native ptype=1 phase admitted by ``efall=0/1`` in place."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    if selection.efall == 1 and nk < 2:
        raise ValueError("efall=1 requires nk >= 2")
    dx, dy, dz, dt = (_positive(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    np = importlib.import_module("numpy")
    a = _require_host_arrays(
        arrays, ni=ni, nj=nj, nk=nk, selection=selection
    )
    c = _constants(np)
    dx32, dy32, dz32, dt32 = (_f(np, value) for value in (dx, dy, dz, dt))
    temperature = _m(np, _a(np, a["th0"], a["th3d"]), _a(np, a["pi0"], a["pp3d"]))
    temperature_initial = temperature.copy(order="C")
    pdef_volume = float(_m(np, _m(np, dx32, dy32), dz32))
    _cpu_pdefq(np, a["qv"], rmax=_f(np, 0.0), a=a, ni=ni, nj=nj, nk=nk, volume=pdef_volume, asq_index=0)
    _cpu_pdefq(np, a["qc"], rmax=_f(np, 1.0e-14), a=a, ni=ni, nj=nj, nk=nk, volume=pdef_volume, asq_index=1)
    _cpu_pdefq(np, a["qr"], rmax=_f(np, 1.0e-14), a=a, ni=ni, nj=nj, nk=nk, volume=pdef_volume, asq_index=2)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                velocity = _f(np, 0.0)
                if a["qr"][p] > _f(np, 1.0e-12):
                    velocity = _m(np, _m(np, 14.34, _pow(np, _m(np, a["rho"][p], a["qr"][p]), 0.1346)), _f(np, np.sqrt(_d(np, 1.15, a["rho"][p]))))
                a["fall_velocity"][p] = velocity
    for j in range(nj):
        bud = 0.0
        for i in range(ni):
            p = (0, j, i)
            term = _m(np, dt32, a["fall_velocity"][p])
            term = _m(np, term, a["rr"][p])
            term = _m(np, term, a["qr"][p])
            term = _m(np, term, _s(np, _m(np, c["cpl"], temperature[p]), c["lv1"]))
            term = _m(np, term, a["ruh"][i])
            term = _m(np, term, a["rvh"][j])
            bud += float(term)
        a["qbudget"][6] += bud * float(dx32) * float(dy32)
    efall = None
    if selection.efall == 1:
        efall = cpu_gm1_kessler_efall_inplace(
            {
                "mf": a["mf"],
                "pi0": a["pi0"],
                "th0": a["th0"],
                "rho": a["rho"],
                "qv": a["qv"],
                "qc": a["qc"],
                "qr": a["qr"],
                "fall_velocity": a["fall_velocity"],
                "temperature": temperature,
                "pp3d": a["pp3d"],
                "th3d": a["th3d"],
                "prs": a["prs"],
            },
            ni=ni,
            nj=nj,
            nk=nk,
            dz=dz,
            dt=dt,
            selection=Gm1KesslerEfallSelection(
                equation_set=selection.equation_set
            ),
        )
    rdz = _d(np, 1.0, dz32)
    max_nfall = 1
    fallout_area = float(_m(np, dx32, dy32))
    for j in range(nj):
        bud = 0.0
        for i in range(ni):
            crmax = _f(np, 0.0)
            for k in range(nk):
                p = (k, j, i)
                crmax = max(crmax, _m(np, _m(np, _m(np, a["fall_velocity"][p], dt32), rdz), a["mh"][p]))
            nfall = max(1, int(_a(np, crmax, 1.0)))
            max_nfall = max(max_nfall, nfall)
            for k in range(nk - 2, -1, -1):
                if a["qr"][k, j, i] < _f(np, 1.0e-8):
                    a["fall_velocity"][k, j, i] = a["fall_velocity"][k + 1, j, i]
            dtfall = _d(np, dt32, _f(np, nfall))
            rq = np.empty(nk + 1, dtype=np.float32)
            for _ in range(nfall):
                for k in range(nk):
                    p = (k, j, i)
                    rq[k] = _m(np, _m(np, a["rho"][p], a["fall_velocity"][p]), max(_f(np, 0.0), a["qr"][p]))
                rq[nk] = _f(np, 0.0)
                for k in range(nk):
                    p = (k, j, i)
                    delta = _d(np, _m(np, _m(np, _m(np, dtfall, _s(np, rq[k + 1], rq[k])), rdz), a["mh"][p]), a["rho"][p])
                    a["qr"][p] = _a(np, a["qr"][p], delta)
                a["rain"][0, j, i] = _a(np, a["rain"][0, j, i], _m(np, _m(np, 0.1, dtfall), rq[0]))
                a["prate"][j, i] = rq[0]
                p0 = (0, j, i)
                term = _m(np, dtfall, a["rr"][p0])
                term = _m(np, term, a["fall_velocity"][p0])
                term = _m(np, term, max(_f(np, 0.0), a["qr"][p0]))
                term = _m(np, term, a["ruh"][i])
                term = _m(np, term, a["rvh"][j])
                bud += float(term)
        a["qbudget"][5] += bud * fallout_area
    local_arrays = {name: a[name] for name in ("ruh", "rvh", "rmh", "pi0", "th0", "rho", "rr", "pp3d", "th3d", "prs", "qv", "qc", "qr")}
    local_arrays["budget"] = a["qbudget"][2:5]
    local = cpu_gm1_kessler_local_inplace(
        local_arrays, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, dt=dt,
        selection=Gm1KesslerLocalSelection(equation_set=selection.equation_set),
        temperature_override=temperature if selection.efall == 1 else None,
    )
    satadj = _cpu_satadj(np, a, ni=ni, nj=nj, nk=nk, dx=dx32, dy=dy32, dz=dz32, dt=dt32, equation_set=selection.equation_set)
    return {
        "temperature_initial": temperature_initial,
        "temperature_held": temperature,
        "efall": efall,
        "local": local,
        "satadj": satadj,
        "max_fall_substeps": max_nfall,
    }


CUDA_GM1_KESSLER_DRIVER_SOURCE = r'''// gm1.cuda-kessler-driver-kernels/v1
#define A(x,y) __fadd_rn((x),(y))
#define S(x,y) __fsub_rn((x),(y))
#define M(x,y) __fmul_rn((x),(y))
#define D(x,y) __fdiv_rn((x),(y))
#define I3(k,j,i,nj,ni) ((((unsigned long long)(k))*(nj)+(j))*(ni)+(i))

__device__ double pdef_one(float rmax,unsigned j,unsigned ni,unsigned nj,unsigned nk,
 const float*ruh,const float*rvh,const float*rmh,const float*rho,float*q){
 double bj=0.0;
 for(unsigned i=0;i<ni;++i){double t1=0.0,t2=0.0,a1=0.0,a2=0.0;
  for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);t1+=(double)M(rho[p],q[p]);a1+=(double)M(M(M(M(rho[p],q[p]),ruh[i]),rvh[j]),rmh[p]);if(q[p]<rmax)q[p]=0.0f;t2+=(double)M(rho[p],q[p]);}
  double t3=(t1+1.0e-20)/(t2+1.0e-20);if(t3<0.0)t3=1.0;
  for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);q[p]=(float)(t3*(double)q[p]);a2+=(double)M(M(M(M(rho[p],q[p]),ruh[i]),rvh[j]),rmh[p]);}
  bj+=a2-a1;
 }return bj;
}

extern "C" __global__ void gm1_kessler_driver_pre_v1(
 float dt,float dz,unsigned efall,unsigned ni,unsigned nj,unsigned nk,
 const float*ruh,const float*rvh,const float*rmh,const float*mh,const float*pi0,const float*th0,
 const float*rho,const float*rr,const float*pp3d,const float*th3d,const float*prs,
 float*qv,float*qc,float*qr,float*vq,float*rain,float*prate,float*temperature,
 double*pdefbud,double*fallbud,float*rq,unsigned*maxnfall){
 unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nj)return;
 float cpl=4190.0f,cpv=1870.0f,xlv=2501000.0f,lv2=S(cpl,cpv),lv1=A(xlv,M(lv2,273.15f)),rdz=D(1.0f,dz);
 for(unsigned k=0;k<nk;++k)for(unsigned i=0;i<ni;++i){unsigned long long p=I3(k,j,i,nj,ni);temperature[p]=M(A(th0[p],th3d[p]),A(pi0[p],pp3d[p]));}
 pdefbud[j]=pdef_one(0.0f,j,ni,nj,nk,ruh,rvh,rmh,rho,qv);
 pdefbud[(unsigned long long)nj+j]=pdef_one(1.0e-14f,j,ni,nj,nk,ruh,rvh,rmh,rho,qc);
 pdefbud[2ull*nj+j]=pdef_one(1.0e-14f,j,ni,nj,nk,ruh,rvh,rmh,rho,qr);
 for(unsigned k=0;k<nk;++k)for(unsigned i=0;i<ni;++i){unsigned long long p=I3(k,j,i,nj,ni);float v=0.0f;if(qr[p]>1.0e-12f)v=M(M(14.34f,powf(M(rho[p],qr[p]),0.1346f)),sqrtf(D(1.15f,rho[p])));vq[p]=v;}
 double eb=0.0;
 for(unsigned i=0;i<ni;++i){unsigned long long p=I3(0,j,i,nj,ni);eb+=(double)M(M(M(M(M(M(dt,vq[p]),rr[p]),qr[p]),S(M(cpl,temperature[p]),lv1)),ruh[i]),rvh[j]);}
 fallbud[j]=eb;if(efall!=0u){fallbud[(unsigned long long)nj+j]=0.0;return;}double tb=0.0;
 for(unsigned i=0;i<ni;++i){float crmax=0.0f;for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);crmax=fmaxf(crmax,M(M(M(vq[p],dt),rdz),mh[p]));}
  unsigned nfall=(unsigned)fmaxf(1.0f,(float)((int)A(crmax,1.0f)));atomicMax(maxnfall,nfall);
  if(nk>1u)for(int k=(int)nk-2;k>=0;--k){unsigned long long p=I3((unsigned)k,j,i,nj,ni);if(qr[p]<1.0e-8f)vq[p]=vq[I3((unsigned)k+1u,j,i,nj,ni)];}
  float dtfall=D(dt,(float)nfall);float*colrq=rq+(unsigned long long)j*(nk+1u);
  for(unsigned n=0;n<nfall;++n){for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);colrq[k]=M(M(rho[p],vq[p]),fmaxf(0.0f,qr[p]));}colrq[nk]=0.0f;
   for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);qr[p]=A(qr[p],D(M(M(M(dtfall,S(colrq[k+1u],colrq[k])),rdz),mh[p]),rho[p]));}
   unsigned long long r=(unsigned long long)j*ni+i;rain[r]=A(rain[r],M(M(0.1f,dtfall),colrq[0]));prate[r]=colrq[0];unsigned long long p=I3(0,j,i,nj,ni);float bt=M(dtfall,rr[p]);bt=M(bt,vq[p]);bt=M(bt,fmaxf(0.0f,qr[p]));bt=M(bt,ruh[i]);bt=M(bt,rvh[j]);tb+=(double)bt;
  }
 }fallbud[(unsigned long long)nj+j]=tb;
}

extern "C" __global__ void gm1_kessler_driver_finish_pre_v1(
 float dx,float dy,float dz,unsigned nj,const double*pdefbud,const double*fallbud,double*asq,double*qbudget){
 if(blockIdx.x||threadIdx.x)return;double volume=(double)M(M(dx,dy),dz);
 for(unsigned s=0;s<3u;++s)for(unsigned j=0;j<nj;++j)asq[s]=asq[s]+pdefbud[(unsigned long long)s*nj+j]*volume;
 for(unsigned j=0;j<nj;++j)qbudget[6]=qbudget[6]+fallbud[j]*(double)dx*(double)dy;
 double area=(double)M(dx,dy);for(unsigned j=0;j<nj;++j)qbudget[5]=qbudget[5]+fallbud[(unsigned long long)nj+j]*area;
}

extern "C" __global__ void gm1_kessler_driver_fallout_v1(
 float dt,float dz,unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,
 const float*mh,const float*rho,const float*rr,float*qr,float*vq,float*rain,float*prate,
 double*fallbud,float*rq,unsigned*maxnfall){
 unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nj)return;float rdz=D(1.0f,dz);double tb=0.0;
 for(unsigned i=0;i<ni;++i){float crmax=0.0f;for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);crmax=fmaxf(crmax,M(M(M(vq[p],dt),rdz),mh[p]));}
  unsigned nfall=(unsigned)fmaxf(1.0f,(float)((int)A(crmax,1.0f)));atomicMax(maxnfall,nfall);
  if(nk>1u)for(int k=(int)nk-2;k>=0;--k){unsigned long long p=I3((unsigned)k,j,i,nj,ni);if(qr[p]<1.0e-8f)vq[p]=vq[I3((unsigned)k+1u,j,i,nj,ni)];}
  float dtfall=D(dt,(float)nfall);float*colrq=rq+(unsigned long long)j*(nk+1u);
  for(unsigned n=0;n<nfall;++n){for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);colrq[k]=M(M(rho[p],vq[p]),fmaxf(0.0f,qr[p]));}colrq[nk]=0.0f;
   for(unsigned k=0;k<nk;++k){unsigned long long p=I3(k,j,i,nj,ni);qr[p]=A(qr[p],D(M(M(M(dtfall,S(colrq[k+1u],colrq[k])),rdz),mh[p]),rho[p]));}
   unsigned long long r=(unsigned long long)j*ni+i;rain[r]=A(rain[r],M(M(0.1f,dtfall),colrq[0]));prate[r]=colrq[0];unsigned long long p=I3(0,j,i,nj,ni);float bt=M(dtfall,rr[p]);bt=M(bt,vq[p]);bt=M(bt,fmaxf(0.0f,qr[p]));bt=M(bt,ruh[i]);bt=M(bt,rvh[j]);tb+=(double)bt;
  }
 }fallbud[(unsigned long long)nj+j]=tb;
}

extern "C" __global__ void gm1_kessler_driver_finish_fallout_v1(
 float dx,float dy,unsigned nj,const double*fallbud,double*qbudget){
 if(blockIdx.x||threadIdx.x)return;double area=(double)M(dx,dy);for(unsigned j=0;j<nj;++j)qbudget[5]=qbudget[5]+fallbud[(unsigned long long)nj+j]*area;
}

extern "C" __global__ void gm1_kessler_driver_local_temperature_v1(
 float dt,unsigned eqtset,unsigned ni,unsigned nj,unsigned nk,
 const float*ruh,const float*rvh,const float*rmh,const float*pi0,const float*th0,
 float*rho,const float*rr,float*pp3d,float*th3d,float*prs,float*qv,float*qc,float*qr,
 const float*temperature,float*tmpout,float*arout,float*crout,float*erout){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long n=(unsigned long long)ni*nj*nk;if(p>=n)return;
 float rd=287.04f,cp=1005.7f,cpv=1870.0f,xlv=2501000.0f,rv=461.5f;
 float cv=S(cp,rd),cvv=S(cpv,rv),cpdcv=D(cp,cv),rovcp=D(rd,cp),rddcv=D(rd,cv),cpdrd=D(cp,rd),eps=D(rd,rv),reps=D(rv,rd),cpl=4190.0f,lv2=S(cpl,cpv),lv1=A(xlv,M(lv2,273.15f)),p00=1.0e5f,rp00=D(1.0f,p00);
 float pisum=A(pi0[p],pp3d[p]),tmp=temperature[p],qvn=qv[p],qcn=qc[p],qrn=qr[p],ar=0.0f,cr=0.0f,er=0.0f;
 if(qcn>0.0f){ar=fmaxf(M(0.001f,S(qcn,0.001f)),0.0f);ar=M(ar,dt);}
 if(qcn>0.0f&&qrn>0.0f)cr=M(M(M(2.2f,qcn),powf(qrn,0.875f)),dt);
 if(qrn>0.0f){float exponent=D(M(17.67f,S(tmp,273.15f)),S(tmp,29.65f));float esl=M(611.2f,expf(exponent));esl=fminf(esl,M(prs[p],0.5f));float qvs=D(M(eps,esl),S(prs[p],esl));if(qvn<qvs){float rhoqr=M(rho[p],qrn);float ventilation=A(1.6f,M(30.3922f,powf(rhoqr,0.2046f)));float deficit=S(1.0f,D(qvn,qvs));float numerator=M(M(ventilation,deficit),powf(rhoqr,0.525f));float denominator=M(A(2.03e4f,D(9.584e6f,M(qvs,prs[p]))),rho[p]);er=fminf(M(D(numerator,denominator),dt),qrn);if(A(qvn,er)>qvs)er=S(qvs,qvn);}}
 if(A(ar,cr)>qcn){float term1=A(ar,cr);ar=D(M(qcn,ar),term1);cr=D(M(qcn,cr),term1);}
 qvn=A(qvn,er);qcn=S(qcn,A(ar,cr));qrn=A(qrn,S(A(ar,cr),er));
 if(er>1.0e-7f){if(eqtset==2u){float cpml=A(A(cp,M(cpv,qvn)),M(cpl,A(qcn,qrn))),cvml=A(A(cv,M(cvv,qvn)),M(cpl,A(qcn,qrn))),rm=A(rd,M(rv,qvn));float latent=S(lv1,M(lv2,tmp));float first=D(latent,M(M(cpdcv,cvml),pisum));float second=M(M(A(th0[p],th3d[p]),D(rv,cvml)),S(1.0f,D(M(rovcp,cpml),rm)));th3d[p]=S(th3d[p],M(er,S(first,second)));float base=M(M(M(rho[p],rm),A(th0[p],th3d[p])),rp00);pp3d[p]=S(powf(base,rddcv),pi0[p]);pisum=A(pi0[p],pp3d[p]);prs[p]=M(p00,powf(pisum,cpdrd));}else{float latent=S(lv1,M(lv2,tmp));th3d[p]=S(th3d[p],M(er,D(latent,M(cp,pisum))));float denominator=M(M(M(rd,A(th0[p],th3d[p])),pisum),A(1.0f,M(qvn,reps)));rho[p]=D(prs[p],denominator);}}
 qv[p]=qvn;qc[p]=qcn;qr[p]=qrn;tmpout[p]=tmp;arout[p]=ar;crout[p]=cr;erout[p]=er;
}

extern "C" __global__ void gm1_kessler_satadj_cell_v1(
 unsigned eqtset,unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,const float*rmh,
 const float*pi0,const float*th0,float*rho,const float*rr,float*pp3d,float*prs,float*th3d,
 float*qv,float*qc,const float*qr,float*cond,float*evac,unsigned*iterations,unsigned*failure){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj*nk;if(p>=n)return;
 float rd=287.04f,cp=1005.7f,cpv=1870.0f,rv=461.5f,xlv=2501000.0f,cv=S(cp,rd),cvv=S(cpv,rv),rovcp=D(rd,cp),eps=D(rd,rv),reps=D(rv,rd),cpl=4190.0f,lv2=S(cpl,cpv),lv1=A(xlv,M(lv2,273.15f)),rp00=D(1.0f,1.0e5f);
 float oqv=qv[p],oqc=qc[p],qvn=oqv,qcn=oqc,t=M(A(th0[p],th3d[p]),A(pi0[p],pp3d[p]));float esl=M(611.2f,expf(D(M(17.67f,S(t,273.15f)),S(t,29.65f))));esl=fminf(esl,M(prs[p],0.5f));float qvs=D(M(eps,esl),S(prs[p],esl));unsigned it=0;
 if(eqtset==2u){if(oqc>1.0e-12f||oqv>qvs){float ql=fmaxf(0.0f,A(oqc,qr[p])),cvml=A(A(cv,M(cvv,qvn)),M(cpl,ql)),lhv=S(lv1,M(lv2,t)),t1=D(S(lhv,M(rv,t)),cvml),d1=M(M(t1,17.67f),243.5f),tlast=t,pnew=prs[p];
  while(1){++it;float tm=S(t,29.65f),dq=D(S(qvs,qvn),A(1.0f,D(M(d1,qvs),M(tm,tm))));dq=fminf(dq,qcn);if(A(qvn,dq)<1.0e-20f)dq=S(1.0e-20f,qvn);qvn=A(qvn,dq);qcn=S(qcn,dq);t=S(t,M(dq,t1));pnew=M(M(rho[p],A(rd,M(rv,qvn))),t);bool done=fabsf(S(t,tlast))<=M(2.0f,0.0001f);if(!done){tlast=t;esl=M(611.2f,expf(D(M(17.67f,S(t,273.15f)),S(t,29.65f))));esl=fminf(esl,M(prs[p],0.5f));qvs=D(M(eps,esl),S(pnew,esl));}if(it==100u){atomicExch(failure,1u);break;}if(done)break;}
  prs[p]=pnew;pp3d[p]=S(powf(M(pnew,rp00),rovcp),pi0[p]);th3d[p]=S(D(t,A(pi0[p],pp3d[p])),th0[p]);qv[p]=qvn;qc[p]=qcn;
 }}else{float lhv=S(lv1,M(lv2,t)),tm=S(t,29.65f),num=M(M(M(lhv,qvs),17.67f),243.5f),dq=D(S(qvs,qvn),A(1.0f,D(num,M(cp,M(tm,tm)))));if(A(qvn,dq)<1.0e-20f)dq=S(1.0e-20f,qvn);dq=fminf(dq,fmaxf(0.0f,qcn));qvn=A(qvn,dq);qcn=S(qcn,dq);it=1u;th3d[p]=S(th3d[p],M(dq,D(lhv,M(cp,A(pi0[p],pp3d[p])))));qv[p]=qvn;qc[p]=qcn;rho[p]=D(prs[p],M(M(M(rd,A(th0[p],th3d[p])),A(pi0[p],pp3d[p])),A(1.0f,M(qvn,reps))));}
 cond[p]=fmaxf(S(qcn,oqc),0.0f);evac[p]=fmaxf(S(qvn,oqv),0.0f);iterations[p]=it;atomicMax(failure+2,it);
}

extern "C" __global__ void gm1_kessler_satadj_level_v1(
 unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,const float*rmh,const float*rr,const float*cond,const float*evac,double*levels){
 unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nk)return;double b1=0.0,b2=0.0;
 for(unsigned j=0;j<nj;++j)for(unsigned i=0;i<ni;++i){unsigned long long p=I3(k,j,i,nj,ni);float tem=M(M(ruh[i],rvh[j]),rmh[p]);b1+=(double)M(M(rr[p],cond[p]),tem);b2+=(double)M(M(rr[p],evac[p]),tem);}levels[k]=b1;levels[(unsigned long long)nk+k]=b2;
}

extern "C" __global__ void gm1_kessler_satadj_finish_v1(
 float dx,float dy,float dz,unsigned nk,const double*levels,double*qbudget){if(blockIdx.x||threadIdx.x)return;double volume=(double)M(M(dx,dy),dz);for(unsigned k=0;k<nk;++k)qbudget[0]=qbudget[0]+levels[k]*volume;for(unsigned k=0;k<nk;++k)qbudget[1]=qbudget[1]+levels[(unsigned long long)nk+k]*volume;}
'''

CUDA_GM1_KESSLER_DRIVER_SOURCE_BYTES = CUDA_GM1_KESSLER_DRIVER_SOURCE.encode("utf-8")
# Updated only after the reviewed CUDA translation changes.
CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256 = (
    "f880dc2ddffd9625bd5c75aad91d9249ba1011f73c1bb250d3d2a0c8d08d43f4"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_GM1_KESSLER_DRIVER_SOURCE_BYTES).hexdigest() == CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256


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
class CompiledGm1KesslerDriverKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_kessler_driver() -> CompiledGm1KesslerDriverKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged GM1 Kessler driver CUDA source differs from pin")
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
    ptx, _ = compiler._NVRTCProgram(CUDA_GM1_KESSLER_DRIVER_SOURCE, "gm1_kessler_driver_v1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_GM1_KESSLER_DRIVER_SOURCE, "gm1_kessler_driver_v1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in CUDA_GM1_KESSLER_DRIVER_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks GM1 Kessler driver entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled GM1 Kessler driver image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_GM1_KESSLER_DRIVER_KERNEL_NAMES)
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
    return CompiledGm1KesslerDriverKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(
    cupy: Any,
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Gm1KesslerDriverSelection,
) -> dict[str, Any]:
    shapes = gm1_kessler_driver_shapes(ni, nj, nk)
    required = ALL_ARRAY_NAMES + (("mf",) if selection.efall == 1 else ())
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA GM1 Kessler driver arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in required:
        value = arrays[name]
        dtype = cupy.dtype("float64") if name in DOUBLE_NAMES else cupy.dtype("float32")
        shape = (nk + 1, nj, ni) if name == "mf" else shapes[name]
        if not isinstance(value, cupy.ndarray) or value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order CUDA {dtype} with shape {shape}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA GM1 Kessler driver arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Gm1KesslerDriverWorkspace:
    ni: int
    nj: int
    nk: int
    temperature_initial: Any
    pdef_budget: Any
    fallout_budget: Any
    rain_flux_column: Any
    condensation: Any
    cloud_evaporation: Any
    iterations: Any
    failure: Any
    satadj_level_budget: Any
    local: Gm1KesslerLocalWorkspace
    efall: Gm1KesslerEfallWorkspace | None


def create_cuda_gm1_kessler_driver_workspace(*, ni: int, nj: int, nk: int) -> Gm1KesslerDriverWorkspace:
    _validate_grid(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    mass = (nk, nj, ni)
    return Gm1KesslerDriverWorkspace(
        ni, nj, nk,
        cupy.empty(mass, dtype=cupy.float32),
        cupy.empty((3, nj), dtype=cupy.float64),
        cupy.empty((2, nj), dtype=cupy.float64),
        cupy.empty((nj, nk + 1), dtype=cupy.float32),
        cupy.empty(mass, dtype=cupy.float32),
        cupy.empty(mass, dtype=cupy.float32),
        cupy.empty(mass, dtype=cupy.uint32),
        cupy.empty((3,), dtype=cupy.uint32),
        cupy.empty((2, nk), dtype=cupy.float64),
        create_cuda_gm1_kessler_local_workspace(ni=ni, nj=nj, nk=nk),
        (
            create_cuda_gm1_kessler_efall_workspace(ni=ni, nj=nj, nk=nk)
            if nk >= 2
            else None
        ),
    )


def _validate_workspace(
    cupy: Any,
    workspace: Gm1KesslerDriverWorkspace,
    *,
    ni: int,
    nj: int,
    nk: int,
    input_pointers: set[int],
) -> None:
    if not isinstance(workspace, Gm1KesslerDriverWorkspace) or (
        workspace.ni,
        workspace.nj,
        workspace.nk,
    ) != (ni, nj, nk):
        raise ValueError("GM1 Kessler driver workspace geometry differs")
    mass = (nk, nj, ni)
    expected = {
        "temperature_initial": (mass, cupy.dtype("float32")),
        "pdef_budget": ((3, nj), cupy.dtype("float64")),
        "fallout_budget": ((2, nj), cupy.dtype("float64")),
        "rain_flux_column": ((nj, nk + 1), cupy.dtype("float32")),
        "condensation": (mass, cupy.dtype("float32")),
        "cloud_evaporation": (mass, cupy.dtype("float32")),
        "iterations": (mass, cupy.dtype("uint32")),
        "failure": ((3,), cupy.dtype("uint32")),
        "satadj_level_budget": ((2, nk), cupy.dtype("float64")),
    }
    pointers: list[int] = []
    for name, (shape, dtype) in expected.items():
        value = getattr(workspace, name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"workspace {name} must be C-order CUDA {dtype} with shape {shape}")
        pointers.append(int(value.data.ptr))
    local_pointers = [
        int(getattr(workspace.local, name).data.ptr)
        for name in ("tmp", "autoconversion", "accretion", "evaporation", "level_budget")
    ]
    efall_pointers: list[int] = []
    if workspace.efall is not None:
        if (workspace.efall.ni, workspace.efall.nj, workspace.efall.nk) != (
            ni,
            nj,
            nk,
        ):
            raise ValueError("GM1 Kessler efall workspace geometry differs")
        for name in ("mixture_cv", "energy_tendency"):
            value = getattr(workspace.efall, name)
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.dtype("float32")
                or value.shape != mass
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"workspace efall.{name} must be C-order CUDA float32 mass shape"
                )
            efall_pointers.append(int(value.data.ptr))
    all_pointers = pointers + local_pointers + efall_pointers
    if len(set(all_pointers)) != len(all_pointers) or input_pointers.intersection(all_pointers):
        raise ValueError("GM1 Kessler driver workspace arrays alias")


@dataclass(frozen=True, slots=True)
class CudaGm1KesslerDriverLaunch:
    ready_event: Any
    compiled: CompiledGm1KesslerDriverKernels
    local_compiled: CompiledGm1KesslerLocalKernels
    efall_compiled: CompiledGm1KesslerEfallKernels | None
    workspace: Gm1KesslerDriverWorkspace
    receipt: dict[str, Any]


def launch_cuda_gm1_kessler_driver(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerDriverSelection = DEFAULT_GM1_KESSLER_DRIVER_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGm1KesslerDriverKernels | None = None,
    local_compiled: CompiledGm1KesslerLocalKernels | None = None,
    efall_compiled: CompiledGm1KesslerEfallKernels | None = None,
    workspace: Gm1KesslerDriverWorkspace | None = None,
) -> CudaGm1KesslerDriverLaunch:
    """Enqueue the complete admitted ptype=1 family without a scientific host copy."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    if selection.efall == 1 and nk < 2:
        raise ValueError("efall=1 requires nk >= 2")
    dx, dy, dz, dt = (_positive(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)))
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(
        cupy, arrays, ni=ni, nj=nj, nk=nk, selection=selection
    )
    compiled = compiled or compile_cuda_gm1_kessler_driver()
    local_compiled = local_compiled or compile_cuda_gm1_kessler_local()
    workspace = workspace or create_cuda_gm1_kessler_driver_workspace(ni=ni, nj=nj, nk=nk)
    _validate_workspace(
        cupy,
        workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        input_pointers={int(value.data.ptr) for value in a.values()},
    )
    stream = stream or cupy.cuda.get_current_stream()
    (
        pre,
        finish_pre,
        fallout,
        finish_fallout,
        local_temperature,
        sat,
        sat_level,
        sat_finish,
    ) = compiled.kernels
    threads = (CUDA_GM1_KESSLER_DRIVER_THREADS, 1, 1)
    jgrid = ((nj + CUDA_GM1_KESSLER_DRIVER_THREADS - 1) // CUDA_GM1_KESSLER_DRIVER_THREADS, 1, 1)
    cells = ni * nj * nk
    cellgrid = ((cells + CUDA_GM1_KESSLER_DRIVER_THREADS - 1) // CUDA_GM1_KESSLER_DRIVER_THREADS, 1, 1)
    kgrid = ((nk + CUDA_GM1_KESSLER_DRIVER_THREADS - 1) // CUDA_GM1_KESSLER_DRIVER_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    with stream:
        workspace.failure.fill(u32(0))
        pre(jgrid, threads, (
            f32(dt), f32(dz), u32(selection.efall), u32(ni), u32(nj), u32(nk), a["ruh"], a["rvh"], a["rmh"], a["mh"], a["pi0"], a["th0"], a["rho"], a["rr"], a["pp3d"], a["th3d"], a["prs"], a["qv"], a["qc"], a["qr"], a["fall_velocity"], a["rain"], a["prate"], workspace.temperature_initial, workspace.pdef_budget, workspace.fallout_budget, workspace.rain_flux_column, workspace.failure[1:2],
        ), stream=stream)
        finish_pre((1, 1, 1), (1, 1, 1), (f32(dx), f32(dy), f32(dz), u32(nj), workspace.pdef_budget, workspace.fallout_budget, a["asq"], a["qbudget"]), stream=stream)
        if selection.efall == 1:
            if workspace.efall is None:  # guarded above; keeps the type explicit
                raise RuntimeError("efall=1 workspace is unavailable")
            efall_launch = launch_cuda_gm1_kessler_efall(
                {
                    "mf": a["mf"], "pi0": a["pi0"], "th0": a["th0"],
                    "rho": a["rho"], "qv": a["qv"], "qc": a["qc"],
                    "qr": a["qr"], "fall_velocity": a["fall_velocity"],
                    "temperature": workspace.temperature_initial,
                    "pp3d": a["pp3d"], "th3d": a["th3d"], "prs": a["prs"],
                },
                ni=ni, nj=nj, nk=nk, dz=dz, dt=dt,
                selection=Gm1KesslerEfallSelection(
                    equation_set=selection.equation_set
                ),
                stream=stream, compiled=efall_compiled, workspace=workspace.efall,
            )
            efall_compiled = efall_launch.compiled
            fallout(jgrid, threads, (
                f32(dt), f32(dz), u32(ni), u32(nj), u32(nk), a["ruh"],
                a["rvh"], a["mh"], a["rho"], a["rr"], a["qr"],
                a["fall_velocity"], a["rain"], a["prate"],
                workspace.fallout_budget, workspace.rain_flux_column,
                workspace.failure[1:2],
            ), stream=stream)
            finish_fallout((1, 1, 1), (1, 1, 1), (
                f32(dx), f32(dy), u32(nj), workspace.fallout_budget,
                a["qbudget"],
            ), stream=stream)
            local_temperature(cellgrid, threads, (
                f32(dt), u32(selection.equation_set), u32(ni), u32(nj),
                u32(nk), a["ruh"], a["rvh"], a["rmh"], a["pi0"],
                a["th0"], a["rho"], a["rr"], a["pp3d"], a["th3d"],
                a["prs"], a["qv"], a["qc"], a["qr"],
                workspace.temperature_initial, workspace.local.tmp,
                workspace.local.autoconversion, workspace.local.accretion,
                workspace.local.evaporation,
            ), stream=stream)
            local_level, local_finish = local_compiled.kernels[1:]
            local_level(kgrid, threads, (
                u32(ni), u32(nj), u32(nk), a["ruh"], a["rvh"], a["rmh"],
                a["rr"], workspace.local.autoconversion,
                workspace.local.accretion, workspace.local.evaporation,
                workspace.local.level_budget,
            ), stream=stream)
            local_finish((1, 1, 1), (1, 1, 1), (
                u32(nk), f32(dx), f32(dy), f32(dz),
                workspace.local.level_budget, a["qbudget"][2:5],
            ), stream=stream)
        else:
            local_launch = launch_cuda_gm1_kessler_local(
                {"ruh": a["ruh"], "rvh": a["rvh"], "rmh": a["rmh"], "pi0": a["pi0"], "th0": a["th0"], "rho": a["rho"], "rr": a["rr"], "pp3d": a["pp3d"], "th3d": a["th3d"], "prs": a["prs"], "qv": a["qv"], "qc": a["qc"], "qr": a["qr"], "budget": a["qbudget"][2:5]},
                ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, dt=dt,
                selection=Gm1KesslerLocalSelection(equation_set=selection.equation_set), stream=stream, compiled=local_compiled, workspace=workspace.local,
            )
            local_compiled = local_launch.compiled
        sat(cellgrid, threads, (u32(selection.equation_set), u32(ni), u32(nj), u32(nk), a["ruh"], a["rvh"], a["rmh"], a["pi0"], a["th0"], a["rho"], a["rr"], a["pp3d"], a["prs"], a["th3d"], a["qv"], a["qc"], a["qr"], workspace.condensation, workspace.cloud_evaporation, workspace.iterations, workspace.failure), stream=stream)
        sat_level(kgrid, threads, (u32(ni), u32(nj), u32(nk), a["ruh"], a["rvh"], a["rmh"], a["rr"], workspace.condensation, workspace.cloud_evaporation, workspace.satadj_level_budget), stream=stream)
        sat_finish((1, 1, 1), (1, 1, 1), (f32(dx), f32(dy), f32(dz), u32(nk), workspace.satadj_level_budget, a["qbudget"]), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_KESSLER_DRIVER_SCHEMA,
        "claim_limit": CUDA_GM1_KESSLER_DRIVER_CLAIM_LIMIT,
        "authority": {"product_name": "GM1", "math_release": "CM1 r21.1", "tarball_sha256": AUTHORITY_TARBALL_SHA256, "source_sha256": dict(AUTHORITY_SOURCE_SHA256)},
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "dx_m": dx, "dy_m": dy, "dz_m": dz, "dt_s": dt},
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "source_order": [
            "pdefq(qv)", "pdefq(qc)", "pdefq(qr)", "k_fallout",
            "geterain",
            *(
                ["getcvm", "getefall", "thermodynamic update"]
                if selection.efall == 1
                else []
            ),
            "fallout", "kessler", "satadj",
        ],
        "family_completeness": {
            "admitted_efall_0_operator": True,
            "admitted_efall_1_operator": True,
            "selected_efall": selection.efall,
            "scheduler_coupling": False,
        },
        "device_status_words": {"satadj_nonconvergence": "failure[0], zero is pass", "max_fall_substeps": "failure[1]", "max_satadj_iterations": "failure[2]", "host_sync_inside_launch": False},
        "residency": {"inputs": "CUDA device", "outputs": "CUDA device in-place", "scratch": "CUDA device reusable workspace", "host_scientific_copy_or_sync": False},
        "carrier_semantics": {
            "rr": "CM1 mp_driver dum3 budget-density snapshot; not reciprocal density",
        },
        "kernel": {
            "abi": CUDA_GM1_KESSLER_DRIVER_ABI,
            "names": list(CUDA_GM1_KESSLER_DRIVER_KERNEL_NAMES),
            "source_sha256": CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256,
            "local_source_sha256": CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256,
            "efall_source_sha256": (
                CUDA_GM1_KESSLER_EFALL_SOURCE_SHA256
                if selection.efall == 1
                else None
            ),
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": CUDA_GM1_KESSLER_DRIVER_THREADS,
            "driver_kernel_launch_count": 8 if selection.efall == 1 else 5,
            "local_kernel_launch_count": 2 if selection.efall == 1 else 3,
            "efall_kernel_launch_count": 3 if selection.efall == 1 else 0,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {"float_state": "explicit tolerance against independent native default-REAL oracle for exp/pow/sqrt", "budgets": "source-order double accumulation of default-REAL terms", "null_or_missing_is_pass": False},
    }
    return CudaGm1KesslerDriverLaunch(
        event, compiled, local_compiled, efall_compiled, workspace, receipt
    )


__all__ = [
    "ALL_ARRAY_NAMES", "AUTHORITY_SOURCE_SHA256", "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_KESSLER_DRIVER_ABI", "CUDA_GM1_KESSLER_DRIVER_CLAIM_LIMIT",
    "CUDA_GM1_KESSLER_DRIVER_KERNEL_NAMES", "CUDA_GM1_KESSLER_DRIVER_SCHEMA",
    "CUDA_GM1_KESSLER_DRIVER_SOURCE", "CUDA_GM1_KESSLER_DRIVER_SOURCE_BYTES",
    "CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256", "CompiledGm1KesslerDriverKernels",
    "CudaGm1KesslerDriverLaunch", "DEFAULT_GM1_KESSLER_DRIVER_SELECTION",
    "Gm1KesslerDriverSelection", "Gm1KesslerDriverWorkspace", "INOUT_NAMES",
    "INPUT_NAMES", "SOURCE_ANCHORS", "compile_cuda_gm1_kessler_driver",
    "cpu_gm1_kessler_driver_inplace", "create_cuda_gm1_kessler_driver_workspace",
    "gm1_kessler_driver_shapes", "launch_cuda_gm1_kessler_driver",
]
