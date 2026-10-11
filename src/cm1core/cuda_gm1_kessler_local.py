"""Device-resident GM1 Kessler warm-rain local conversion.

The public model name is GM1. CM1 r21.1 ``kessler.F:8-138`` is the
mathematical authority for the local, column-independent conversion phase.
The native ``ptype=1`` driver order is broader::

    pdefq(qv, qc, qr) -> k_fallout/geterain -> optional efall -> fallout
    -> kessler local conversion -> satadj

This module implements only the named local conversion, in source statement
order: cloud autoconversion, cloud/rain accretion, rain evaporation, the cloud
water limiter, water-species updates, conditional thermodynamic update, and
the three process-budget reductions. Callers must supply the post-fallout
state and must compose saturation adjustment afterward. Neither seam is
silently approximated.

The public array key ``rr`` follows the dummy-argument name in
``kessler.F``. At this call site CM1 ``mp_driver.F`` passes ``dum3``, the
pre-physics budget-density snapshot (equal to ``rho`` for the admitted
Cartesian selector); it is not the model's global reciprocal-density array.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import math
import re
from typing import Any

from cm1core.errors import refuse


CUDA_GM1_KESSLER_LOCAL_SCHEMA = "gm1.cuda-kessler-local/v1"
CUDA_GM1_KESSLER_LOCAL_ABI = "gm1.cuda-kessler-local-kernels/v1"
CUDA_GM1_KESSLER_LOCAL_THREADS = 256
CUDA_GM1_KESSLER_LOCAL_KERNEL_NAMES = (
    "gm1_kessler_local_cell_v1",
    "gm1_kessler_local_level_budget_v1",
    "gm1_kessler_local_finish_budget_v1",
)
CUDA_GM1_KESSLER_LOCAL_CLAIM_LIMIT = (
    "GM1 device-resident ptype=1 Kessler local conversion from a caller-supplied "
    "post-fallout state through autoconversion, accretion, evaporation, cloud "
    "limiting, qv/qc/qr and equation-set-specific thermodynamic updates, plus "
    "source-ordered process budgets. It is not the complete ptype=1 driver: "
    "pdefq, k_fallout, geterain, optional efall, fallout, and following satadj "
    "are explicit unimplemented seams. No precipitation trajectory, forecast, "
    "tornado, or arbitrary-physics-family claim."
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/kessler.F": "f232f840e425d8c9abd28841e0a46a06e0dbf641460224fa61b6a82e5a650472",
    "src/mp_driver.F": "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134",
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
}
SOURCE_ANCHORS = {
    "native_driver_order": "src/mp_driver.F:190-232",
    "local_conversion": "src/kessler.F:8-138",
    "autoconversion": "src/kessler.F:45-50",
    "accretion": "src/kessler.F:52-57",
    "evaporation": "src/kessler.F:59-76",
    "limiter_and_water_update": "src/kessler.F:78-89",
    "thermodynamic_update": "src/kessler.F:91-109",
    "budgets": "src/kessler.F:111-135",
    "constants": "src/constants.F:5-31,69-140",
}


@dataclass(frozen=True, slots=True)
class Gm1KesslerLocalSelection:
    """Source choices supported by the local conversion kernel."""

    ptype: int = 1
    equation_set: int = 1
    testcase: int = 0
    input_phase: str = "post-fallout"
    output_phase: str = "pre-satadj"


def _validate_selection(selection: Gm1KesslerLocalSelection) -> None:
    if not isinstance(selection, Gm1KesslerLocalSelection):
        raise TypeError("selection must be a Gm1KesslerLocalSelection")
    fixed = {
        "ptype": 1,
        "testcase": 0,
        "input_phase": "post-fallout",
        "output_phase": "pre-satadj",
    }
    for name, wanted in fixed.items():
        value = getattr(selection, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="GM1-KESSLER-LOCAL-SELECTION",
                gate="physics-seam",
                option=name,
                value=value,
                reason="the local conversion is sealed to the source ptype=1 post-fallout/pre-satadj phase",
                remedy=f"use {name}={wanted!r} or port the missing native driver phase",
            )
    if type(selection.equation_set) is not int or selection.equation_set not in (1, 2):
        refuse(
            code="GM1-KESSLER-LOCAL-EQTSET",
            gate="physics-seam",
            option="equation_set",
            value=selection.equation_set,
            reason="kessler.F has exactly the traditional and energy-conserving branches",
            remedy="use equation_set=1 or equation_set=2",
        )


DEFAULT_GM1_KESSLER_LOCAL_SELECTION = Gm1KesslerLocalSelection()


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if ni * nj * nk > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("GM1 Kessler local word count exceeds signed 64-bit indexing")


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def gm1_kessler_local_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    mass = (nk, nj, ni)
    return {
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": mass,
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
        "budget": (3,),
    }


INPUT_NAMES = ("ruh", "rvh", "rmh", "pi0", "th0", "rr")
INOUT_NAMES = ("rho", "pp3d", "th3d", "prs", "qv", "qc", "qr", "budget")
ALL_ARRAY_NAMES = INPUT_NAMES + INOUT_NAMES


def _require_host_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = gm1_kessler_local_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing GM1 Kessler local arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        dtype = np.dtype("float64") if name == "budget" else np.dtype("float32")
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order {dtype} with shape {shapes[name]}")
        if name in INOUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("GM1 Kessler local arrays must not alias")
    if bool(np.any(result["rho"] <= np.float32(0.0))):
        raise ValueError("rho must be > 0")
    if bool(np.any(result["prs"] <= np.float32(0.0))):
        raise ValueError("prs must be > 0")
    if any(bool(np.any(result[name] < np.float32(0.0))) for name in ("qv", "qc", "qr")):
        raise ValueError("qv, qc, and qr must be >= 0")
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


def _default_constants(np: Any) -> dict[str, Any]:
    rd = _f(np, 287.04)
    cp = _f(np, 1005.7)
    cpv = _f(np, 1870.0)
    xlv = _f(np, 2501000.0)
    rv = _f(np, 461.5)
    cv = _s(np, cp, rd)
    cvv = _s(np, cpv, rv)
    cpdcv = _d(np, cp, cv)
    rovcp = _d(np, rd, cp)
    rddcv = _d(np, rd, cv)
    cpdrd = _d(np, cp, rd)
    eps = _d(np, rd, rv)
    reps = _d(np, rv, rd)
    cpl = _f(np, 4190.0)
    lv2 = _s(np, cpl, cpv)
    lv1 = _a(np, xlv, _m(np, lv2, _f(np, 273.15)))
    p00 = _f(np, 1.0e5)
    rp00 = _d(np, 1.0, p00)
    return locals()


def cpu_gm1_kessler_local_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerLocalSelection = DEFAULT_GM1_KESSLER_LOCAL_SELECTION,
    temperature_override: Any | None = None,
) -> dict[str, Any]:
    """Apply the source-ordered local conversion to compact host arrays."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz, dt = (
        _positive(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt))
    )
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    if temperature_override is not None:
        if (
            not isinstance(temperature_override, np.ndarray)
            or temperature_override.dtype != np.dtype("float32")
            or temperature_override.shape != (nk, nj, ni)
            or not temperature_override.flags.c_contiguous
            or not bool(np.all(np.isfinite(temperature_override)))
        ):
            raise ValueError("temperature_override must be finite C-order float32 mass shape")
        if int(temperature_override.ctypes.data) in {
            int(value.ctypes.data) for value in a.values()
        }:
            raise ValueError("temperature_override must not alias a Kessler local array")
    c = _default_constants(np)
    diagnostics = {
        name: np.empty((nk, nj, ni), dtype=np.float32)
        for name in ("tmp", "autoconversion", "accretion", "evaporation")
    }
    bud = np.zeros((3, nk), dtype=np.float64)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                p = (k, j, i)
                pisum = _a(np, a["pi0"][p], a["pp3d"][p])
                tmp = (
                    _f(np, temperature_override[p])
                    if temperature_override is not None
                    else _m(np, _a(np, a["th0"][p], a["th3d"][p]), pisum)
                )
                qvnew = _f(np, a["qv"][p])
                qcnew = _f(np, a["qc"][p])
                qrnew = _f(np, a["qr"][p])
                ar = _f(np, 0.0)
                cr = _f(np, 0.0)
                er = _f(np, 0.0)
                if qcnew > _f(np, 0.0):
                    ar = max(_m(np, 0.001, _s(np, qcnew, 0.001)), _f(np, 0.0))
                    ar = _m(np, ar, dt)
                if qcnew > _f(np, 0.0) and qrnew > _f(np, 0.0):
                    cr = _m(np, _m(np, _m(np, 2.2, qcnew), _pow(np, qrnew, 0.875)), dt)
                if qrnew > _f(np, 0.0):
                    exponent = _d(
                        np,
                        _m(np, 17.67, _s(np, tmp, 273.15)),
                        _s(np, tmp, 29.65),
                    )
                    esl = _m(np, 611.2, _exp(np, exponent))
                    esl = min(esl, _m(np, a["prs"][p], 0.5))
                    qvs = _d(np, _m(np, c["eps"], esl), _s(np, a["prs"][p], esl))
                    if qvnew < qvs:
                        rhoqr = _m(np, a["rho"][p], qrnew)
                        ventilation = _a(
                            np, 1.6, _m(np, 30.3922, _pow(np, rhoqr, 0.2046))
                        )
                        deficit = _s(np, 1.0, _d(np, qvnew, qvs))
                        numerator = _m(
                            np,
                            _m(np, ventilation, deficit),
                            _pow(np, rhoqr, 0.525),
                        )
                        denominator = _m(
                            np,
                            _a(
                                np,
                                2.03e4,
                                _d(np, 9.584e6, _m(np, qvs, a["prs"][p])),
                            ),
                            a["rho"][p],
                        )
                        er = min(_m(np, _d(np, numerator, denominator), dt), qrnew)
                        if _a(np, qvnew, er) > qvs:
                            er = _s(np, qvs, qvnew)
                if _a(np, ar, cr) > qcnew:
                    term1 = _a(np, ar, cr)
                    ar = _d(np, _m(np, qcnew, ar), term1)
                    cr = _d(np, _m(np, qcnew, cr), term1)
                qvnew = _a(np, qvnew, er)
                qcnew = _s(np, qcnew, _a(np, ar, cr))
                qrnew = _a(np, qrnew, _s(np, _a(np, ar, cr), er))
                if er > _f(np, 1.0e-7):
                    if selection.equation_set == 2:
                        cpml = _a(
                            np,
                            _a(np, c["cp"], _m(np, c["cpv"], qvnew)),
                            _m(np, c["cpl"], _a(np, qcnew, qrnew)),
                        )
                        cvml = _a(
                            np,
                            _a(np, c["cv"], _m(np, c["cvv"], qvnew)),
                            _m(np, c["cpl"], _a(np, qcnew, qrnew)),
                        )
                        rm = _a(np, c["rd"], _m(np, c["rv"], qvnew))
                        latent = _s(np, c["lv1"], _m(np, c["lv2"], tmp))
                        first = _d(
                            np,
                            latent,
                            _m(np, _m(np, c["cpdcv"], cvml), pisum),
                        )
                        second = _m(
                            np,
                            _m(
                                np,
                                _a(np, a["th0"][p], a["th3d"][p]),
                                _d(np, c["rv"], cvml),
                            ),
                            _s(np, 1.0, _d(np, _m(np, c["rovcp"], cpml), rm)),
                        )
                        a["th3d"][p] = _s(
                            np, a["th3d"][p], _m(np, er, _s(np, first, second))
                        )
                        base = _m(
                            np,
                            _m(np, _m(np, a["rho"][p], rm), _a(np, a["th0"][p], a["th3d"][p])),
                            c["rp00"],
                        )
                        a["pp3d"][p] = _s(
                            np, _pow(np, base, c["rddcv"]), a["pi0"][p]
                        )
                        pisum = _a(np, a["pi0"][p], a["pp3d"][p])
                        a["prs"][p] = _m(np, c["p00"], _pow(np, pisum, c["cpdrd"]))
                    else:
                        latent = _s(np, c["lv1"], _m(np, c["lv2"], tmp))
                        a["th3d"][p] = _s(
                            np,
                            a["th3d"][p],
                            _m(np, er, _d(np, latent, _m(np, c["cp"], pisum))),
                        )
                        denominator = _m(
                            np,
                            _m(
                                np,
                                _m(np, c["rd"], _a(np, a["th0"][p], a["th3d"][p])),
                                pisum,
                            ),
                            _a(np, 1.0, _m(np, qvnew, c["reps"])),
                        )
                        a["rho"][p] = _d(np, a["prs"][p], denominator)
                tem = _m(np, _m(np, a["ruh"][i], a["rvh"][j]), a["rmh"][p])
                bud[0, k] += float(_m(np, _m(np, a["rr"][p], ar), tem))
                bud[1, k] += float(_m(np, _m(np, a["rr"][p], cr), tem))
                bud[2, k] += float(_m(np, _m(np, a["rr"][p], er), tem))
                a["qv"][p] = qvnew
                a["qc"][p] = qcnew
                a["qr"][p] = qrnew
                diagnostics["tmp"][p] = tmp
                diagnostics["autoconversion"][p] = ar
                diagnostics["accretion"][p] = cr
                diagnostics["evaporation"][p] = er
    volume = float(_m(np, _m(np, dx, dy), dz))
    for process in range(3):
        for k in range(nk):
            a["budget"][process] = a["budget"][process] + bud[process, k] * volume
    return diagnostics


CUDA_GM1_KESSLER_LOCAL_SOURCE = r'''// gm1.cuda-kessler-local-kernels/v1
#define A(x,y) __fadd_rn((x),(y))
#define S(x,y) __fsub_rn((x),(y))
#define M(x,y) __fmul_rn((x),(y))
#define D(x,y) __fdiv_rn((x),(y))
#define I3(k,j,i,nj,ni) ((((unsigned long long)(k))*(nj)+(j))*(ni)+(i))

extern "C" __global__ void gm1_kessler_local_cell_v1(
 float dt,unsigned eqtset,unsigned ni,unsigned nj,unsigned nk,
 const float*ruh,const float*rvh,const float*rmh,const float*pi0,const float*th0,
 float*rho,const float*rr,float*pp3d,float*th3d,float*prs,float*qv,float*qc,float*qr,
 float*tmpout,float*arout,float*crout,float*erout){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 unsigned long long n=(unsigned long long)ni*nj*nk;if(p>=n)return;
 unsigned i=(unsigned)(p%ni),j=(unsigned)((p/ni)%nj);
 float rd=287.04f,cp=1005.7f,cpv=1870.0f,xlv=2501000.0f,rv=461.5f;
 float cv=S(cp,rd),cvv=S(cpv,rv),cpdcv=D(cp,cv),rovcp=D(rd,cp),rddcv=D(rd,cv),cpdrd=D(cp,rd),eps=D(rd,rv),reps=D(rv,rd),cpl=4190.0f,lv2=S(cpl,cpv),lv1=A(xlv,M(lv2,273.15f)),p00=1.0e5f,rp00=D(1.0f,p00);
 float pisum=A(pi0[p],pp3d[p]),tmp=M(A(th0[p],th3d[p]),pisum),qvn=qv[p],qcn=qc[p],qrn=qr[p],ar=0.0f,cr=0.0f,er=0.0f;
 if(qcn>0.0f){ar=fmaxf(M(0.001f,S(qcn,0.001f)),0.0f);ar=M(ar,dt);}
 if(qcn>0.0f&&qrn>0.0f)cr=M(M(M(2.2f,qcn),powf(qrn,0.875f)),dt);
 if(qrn>0.0f){float exponent=D(M(17.67f,S(tmp,273.15f)),S(tmp,29.65f));float esl=M(611.2f,expf(exponent));esl=fminf(esl,M(prs[p],0.5f));float qvs=D(M(eps,esl),S(prs[p],esl));if(qvn<qvs){float rhoqr=M(rho[p],qrn);float ventilation=A(1.6f,M(30.3922f,powf(rhoqr,0.2046f)));float deficit=S(1.0f,D(qvn,qvs));float numerator=M(M(ventilation,deficit),powf(rhoqr,0.525f));float denominator=M(A(2.03e4f,D(9.584e6f,M(qvs,prs[p]))),rho[p]);er=fminf(M(D(numerator,denominator),dt),qrn);if(A(qvn,er)>qvs)er=S(qvs,qvn);}}
 if(A(ar,cr)>qcn){float term1=A(ar,cr);ar=D(M(qcn,ar),term1);cr=D(M(qcn,cr),term1);}
 qvn=A(qvn,er);qcn=S(qcn,A(ar,cr));qrn=A(qrn,S(A(ar,cr),er));
 if(er>1.0e-7f){if(eqtset==2u){float cpml=A(A(cp,M(cpv,qvn)),M(cpl,A(qcn,qrn))),cvml=A(A(cv,M(cvv,qvn)),M(cpl,A(qcn,qrn))),rm=A(rd,M(rv,qvn));float latent=S(lv1,M(lv2,tmp));float first=D(latent,M(M(cpdcv,cvml),pisum));float second=M(M(A(th0[p],th3d[p]),D(rv,cvml)),S(1.0f,D(M(rovcp,cpml),rm)));th3d[p]=S(th3d[p],M(er,S(first,second)));float base=M(M(M(rho[p],rm),A(th0[p],th3d[p])),rp00);pp3d[p]=S(powf(base,rddcv),pi0[p]);pisum=A(pi0[p],pp3d[p]);prs[p]=M(p00,powf(pisum,cpdrd));}else{float latent=S(lv1,M(lv2,tmp));th3d[p]=S(th3d[p],M(er,D(latent,M(cp,pisum))));float denominator=M(M(M(rd,A(th0[p],th3d[p])),pisum),A(1.0f,M(qvn,reps)));rho[p]=D(prs[p],denominator);}}
 qv[p]=qvn;qc[p]=qcn;qr[p]=qrn;tmpout[p]=tmp;arout[p]=ar;crout[p]=cr;erout[p]=er;
}

extern "C" __global__ void gm1_kessler_local_level_budget_v1(
 unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,const float*rmh,const float*rr,
 const float*ar,const float*cr,const float*er,double*levels){
 unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nk)return;double b1=0.0,b2=0.0,b3=0.0;
 for(unsigned j=0;j<nj;++j)for(unsigned i=0;i<ni;++i){unsigned long long p=I3(k,j,i,nj,ni);float tem=M(M(ruh[i],rvh[j]),rmh[p]);b1+=(double)M(M(rr[p],ar[p]),tem);b2+=(double)M(M(rr[p],cr[p]),tem);b3+=(double)M(M(rr[p],er[p]),tem);}
 levels[k]=b1;levels[(unsigned long long)nk+k]=b2;levels[2ull*nk+k]=b3;
}

extern "C" __global__ void gm1_kessler_local_finish_budget_v1(
 unsigned nk,float dx,float dy,float dz,const double*levels,double*budget){
 if(blockIdx.x||threadIdx.x)return;double volume=(double)M(M(dx,dy),dz);
 for(unsigned process=0;process<3u;++process)for(unsigned k=0;k<nk;++k)budget[process]=budget[process]+levels[(unsigned long long)process*nk+k]*volume;
}
'''

CUDA_GM1_KESSLER_LOCAL_SOURCE_BYTES = CUDA_GM1_KESSLER_LOCAL_SOURCE.encode("utf-8")
# This literal is updated only after the reviewed CUDA translation changes.
CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256 = (
    "320eeb15809a66fc446770081d61b404e9ebade0bf86695392cc6407ca427278"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_GM1_KESSLER_LOCAL_SOURCE_BYTES).hexdigest() == CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256


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
class CompiledGm1KesslerLocalKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_kessler_local() -> CompiledGm1KesslerLocalKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged GM1 Kessler local CUDA source differs from pin")
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
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_KESSLER_LOCAL_SOURCE, "gm1_kessler_local_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_KESSLER_LOCAL_SOURCE, "gm1_kessler_local_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in CUDA_GM1_KESSLER_LOCAL_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks GM1 Kessler local entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled GM1 Kessler local image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_GM1_KESSLER_LOCAL_KERNEL_NAMES)
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
    return CompiledGm1KesslerLocalKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = gm1_kessler_local_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA GM1 Kessler local arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        dtype = cupy.dtype("float64") if name == "budget" else cupy.dtype("float32")
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != dtype
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA {dtype} with shape {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA GM1 Kessler local arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Gm1KesslerLocalWorkspace:
    ni: int
    nj: int
    nk: int
    tmp: Any
    autoconversion: Any
    accretion: Any
    evaporation: Any
    level_budget: Any


def create_cuda_gm1_kessler_local_workspace(
    *, ni: int, nj: int, nk: int
) -> Gm1KesslerLocalWorkspace:
    _validate_grid(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    shape = (nk, nj, ni)
    return Gm1KesslerLocalWorkspace(
        ni,
        nj,
        nk,
        cupy.empty(shape, dtype=cupy.float32),
        cupy.empty(shape, dtype=cupy.float32),
        cupy.empty(shape, dtype=cupy.float32),
        cupy.empty(shape, dtype=cupy.float32),
        cupy.empty((3, nk), dtype=cupy.float64),
    )


def _validate_workspace(
    cupy: Any,
    workspace: Gm1KesslerLocalWorkspace,
    *,
    ni: int,
    nj: int,
    nk: int,
    input_pointers: set[int],
) -> None:
    if not isinstance(workspace, Gm1KesslerLocalWorkspace) or (
        workspace.ni,
        workspace.nj,
        workspace.nk,
    ) != (ni, nj, nk):
        raise ValueError("GM1 Kessler local workspace geometry differs")
    expected = {
        "tmp": ((nk, nj, ni), cupy.dtype("float32")),
        "autoconversion": ((nk, nj, ni), cupy.dtype("float32")),
        "accretion": ((nk, nj, ni), cupy.dtype("float32")),
        "evaporation": ((nk, nj, ni), cupy.dtype("float32")),
        "level_budget": ((3, nk), cupy.dtype("float64")),
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
    if len(set(pointers)) != len(pointers) or input_pointers.intersection(pointers):
        raise ValueError("GM1 Kessler local workspace arrays alias")


@dataclass(frozen=True, slots=True)
class CudaGm1KesslerLocalLaunch:
    ready_event: Any
    compiled: CompiledGm1KesslerLocalKernels
    workspace: Gm1KesslerLocalWorkspace
    receipt: dict[str, Any]


def launch_cuda_gm1_kessler_local(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerLocalSelection = DEFAULT_GM1_KESSLER_LOCAL_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGm1KesslerLocalKernels | None = None,
    workspace: Gm1KesslerLocalWorkspace | None = None,
) -> CudaGm1KesslerLocalLaunch:
    """Enqueue local conversion and deterministic source-order budgets."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz, dt = (
        _positive(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt))
    )
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    compiled = compiled or compile_cuda_gm1_kessler_local()
    stream = stream or cupy.cuda.get_current_stream()
    workspace = workspace or create_cuda_gm1_kessler_local_workspace(ni=ni, nj=nj, nk=nk)
    _validate_workspace(
        cupy,
        workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        input_pointers={int(value.data.ptr) for value in a.values()},
    )
    cell, level, finish = compiled.kernels
    threads = (CUDA_GM1_KESSLER_LOCAL_THREADS, 1, 1)
    cells = ni * nj * nk
    cell_grid = ((cells + CUDA_GM1_KESSLER_LOCAL_THREADS - 1) // CUDA_GM1_KESSLER_LOCAL_THREADS, 1, 1)
    level_grid = ((nk + CUDA_GM1_KESSLER_LOCAL_THREADS - 1) // CUDA_GM1_KESSLER_LOCAL_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    with stream:
        cell(
            cell_grid,
            threads,
            (
                f32(dt), u32(selection.equation_set), u32(ni), u32(nj), u32(nk),
                a["ruh"], a["rvh"], a["rmh"], a["pi0"], a["th0"], a["rho"],
                a["rr"], a["pp3d"], a["th3d"], a["prs"], a["qv"], a["qc"],
                a["qr"], workspace.tmp, workspace.autoconversion, workspace.accretion,
                workspace.evaporation,
            ),
            stream=stream,
        )
        level(
            level_grid,
            threads,
            (
                u32(ni), u32(nj), u32(nk), a["ruh"], a["rvh"], a["rmh"], a["rr"],
                workspace.autoconversion, workspace.accretion, workspace.evaporation,
                workspace.level_budget,
            ),
            stream=stream,
        )
        finish(
            (1, 1, 1),
            (1, 1, 1),
            (u32(nk), f32(dx), f32(dy), f32(dz), workspace.level_budget, a["budget"]),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_KESSLER_LOCAL_SCHEMA,
        "claim_limit": CUDA_GM1_KESSLER_LOCAL_CLAIM_LIMIT,
        "authority": {
            "product_name": "GM1",
            "math_release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "dx_m": dx, "dy_m": dy, "dz_m": dz, "dt_s": dt},
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "source_order": ["autoconversion", "accretion", "evaporation", "cloud limiter", "water update", "thermodynamic update", "budgets"],
        "driver_seams": {
            "required_before": ["pdefq(qv,qc,qr)", "k_fallout/geterain", "optional efall", "fallout"],
            "required_after": ["satadj"],
            "included_here": ["kessler local conversion"],
        },
        "carrier_semantics": {
            "rr": "CM1 mp_driver dum3 budget-density snapshot; not reciprocal density",
        },
        "residency": {
            "inputs": "CUDA device",
            "outputs": "CUDA device in-place",
            "scratch": "CUDA device reusable workspace",
            "host_scientific_copy_or_sync": False,
        },
        "kernel": {
            "abi": CUDA_GM1_KESSLER_LOCAL_ABI,
            "names": list(CUDA_GM1_KESSLER_LOCAL_KERNEL_NAMES),
            "source_sha256": CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": CUDA_GM1_KESSLER_LOCAL_THREADS,
            "cell_grid_x": cell_grid[0],
            "level_grid_x": level_grid[0],
            "kernel_launch_count": 3,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {
            "water_and_state": "float32 tolerance against independent native default-REAL oracle because exp/pow cross-libm raw-bit identity is not claimed",
            "process_budget": "source-order double accumulation of float32 cell terms",
            "null_or_missing_is_pass": False,
        },
    }
    return CudaGm1KesslerLocalLaunch(event, compiled, workspace, receipt)


__all__ = [
    "ALL_ARRAY_NAMES",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_KESSLER_LOCAL_ABI",
    "CUDA_GM1_KESSLER_LOCAL_CLAIM_LIMIT",
    "CUDA_GM1_KESSLER_LOCAL_KERNEL_NAMES",
    "CUDA_GM1_KESSLER_LOCAL_SCHEMA",
    "CUDA_GM1_KESSLER_LOCAL_SOURCE",
    "CUDA_GM1_KESSLER_LOCAL_SOURCE_SHA256",
    "CompiledGm1KesslerLocalKernels",
    "CudaGm1KesslerLocalLaunch",
    "DEFAULT_GM1_KESSLER_LOCAL_SELECTION",
    "Gm1KesslerLocalSelection",
    "Gm1KesslerLocalWorkspace",
    "INOUT_NAMES",
    "INPUT_NAMES",
    "SOURCE_ANCHORS",
    "compile_cuda_gm1_kessler_local",
    "cpu_gm1_kessler_local_inplace",
    "create_cuda_gm1_kessler_local_workspace",
    "gm1_kessler_local_shapes",
    "launch_cuda_gm1_kessler_local",
]
