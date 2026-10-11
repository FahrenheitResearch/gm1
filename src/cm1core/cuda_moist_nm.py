"""CM1 r21.1 moist squared-buoyancy-frequency operator for CUDA.

This is the exact six-species, ice-aware ``calcnm`` branch selected by the
ArWen WSM6 product profile.  One CUDA thread owns an entire vertical column,
which preserves CM1's bottom-to-top dependency and avoids host transforms.
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


CUDA_MOIST_NM_SCHEMA = "gpuwm-cm1.cuda-moist-nm/v1"
CUDA_MOIST_NM_ABI = "gpuwm-cm1.cuda-moist-nm-kernels/v1"
CUDA_MOIST_NM_THREADS = 128
CUDA_MOIST_NM_KERNEL_NAMES = ("cm1_moist_nm_columns_v1",)
CUDA_MOIST_NM_CLAIM_LIMIT = (
    "CM1 r21.1 ptype=5, imoist=1, iice=1, six-species WSM6 calcnm "
    "branch on physical columns; no horizontal halo exchange, calcdef, SGS "
    "coefficient, turbulent flux, microphysics advance, or forecast claim"
)
SOURCE_ANCHORS = {
    "calcnm": "src/turb.F:5432-5685",
    "liquid_saturation": "src/cm1libs.F:16-62",
    "ice_saturation": "src/cm1libs.F:69-115",
    "constants": "src/constants.F:14-22;100-153",
}


@dataclass(frozen=True, slots=True)
class MoistNmSelection:
    ptype: int = 5
    imoist: int = 1
    iice: int = 1
    numq: int = 6
    nqv: int = 1
    nqc: int = 2
    nqi: int = 4
    nql1: int = 2
    nql2: int = 3
    nqs1: int = 4
    nqs2: int = 6


DEFAULT_MOIST_NM_SELECTION = MoistNmSelection()
_EXPECTED_SELECTION = {
    "physics.native_ptype_context": ("ptype", 5),
    "physics.moist": ("imoist", 1),
    "physics.ice": ("iice", 1),
    "physics.moist_species": ("numq", 6),
    "physics.nqv": ("nqv", 1),
    "physics.nqc": ("nqc", 2),
    "physics.nqi": ("nqi", 4),
    "physics.nql1": ("nql1", 2),
    "physics.nql2": ("nql2", 3),
    "physics.nqs1": ("nqs1", 4),
    "physics.nqs2": ("nqs2", 6),
}

INPUT_ARRAY_NAMES = ("c1", "c2", "mf", "pi0", "th0", "prs", "pp3d", "th3d", "q3d")
FLOAT_OUTPUT_NAMES = ("nm", "temperature", "qt", "thv", "qvci")
MASK_NAME = "saturation_mask"
ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + FLOAT_OUTPUT_NAMES + (MASK_NAME,)


def _validate_selection(selection: MoistNmSelection) -> None:
    if not isinstance(selection, MoistNmSelection):
        raise TypeError("selection must be a MoistNmSelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            refuse(
                code="CM1-MOIST-NM-UNSUPPORTED",
                gate="physics-seam",
                option=option,
                value=value,
                reason=(
                    f"the sealed moist-Nm operator covers only {option}={expected}; "
                    "another value selects different CM1 species or thermodynamics"
                ),
                remedy="use the ArWen WSM6 product branch or add a separate native oracle",
            )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * nk > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("moist-Nm word count exceeds signed 64-bit indexing")


def _validated_dz(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dz must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("dz must be finite and > 0")
    return result


def moist_nm_array_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    cell = (nk, nj, ni)
    face = (nk + 1, nj, ni)
    return {
        "c1": face,
        "c2": face,
        "mf": face,
        "pi0": (nk,),
        "th0": (nk,),
        "prs": cell,
        "pp3d": cell,
        "th3d": cell,
        "q3d": (6, nk, nj, ni),
        "nm": face,
        "temperature": cell,
        "qt": cell,
        "thv": cell,
        "qvci": cell,
        MASK_NAME: cell,
    }


def _require_host_arrays(arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = moist_nm_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing moist-Nm arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        expected_dtype = np.dtype("uint8") if name == MASK_NAME else np.dtype("float32")
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype.name}")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name in FLOAT_OUTPUT_NAMES + (MASK_NAME,) and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("moist-Nm arrays must not alias")
    return result


def _f32_add(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) + np.float32(b))


def _f32_sub(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) - np.float32(b))


def _f32_mul(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) * np.float32(b))


def _f32_div(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) / np.float32(b))


def _f32_log(np: Any, value: Any) -> Any:
    return np.float32(np.log(np.float32(value)))


def _f32_exp(np: Any, value: Any) -> Any:
    return np.float32(np.exp(np.float32(value)))


def _constants(np: Any) -> dict[str, Any]:
    g = np.float32(9.81)
    rd = np.float32(287.04)
    rv = np.float32(461.5)
    cp = np.float32(1005.7)
    cpv = np.float32(1870.0)
    cpl = np.float32(4190.0)
    cpi = np.float32(2106.0)
    xlv = np.float32(2501000.0)
    xls = np.float32(2834000.0)
    to = np.float32(273.15)
    return {
        "g": g,
        "rd": rd,
        "rv": rv,
        "cp": cp,
        "cpv": cpv,
        "cpl": cpl,
        "cpi": cpi,
        "eps": _f32_div(np, rd, rv),
        "reps": _f32_div(np, rv, rd),
        "lv1": _f32_add(np, xlv, _f32_mul(np, _f32_sub(np, cpl, cpv), to)),
        "lv2": _f32_sub(np, cpl, cpv),
        "ls1": _f32_add(np, xls, _f32_mul(np, _f32_sub(np, cpi, cpv), to)),
        "ls2": _f32_sub(np, cpi, cpv),
    }


def _rslf(np: Any, pressure: Any, temperature: Any, eps: Any) -> Any:
    numerator = _f32_mul(np, np.float32(17.67), _f32_sub(np, temperature, np.float32(273.15)))
    exponent = _f32_div(np, numerator, _f32_sub(np, temperature, np.float32(29.65)))
    es = _f32_mul(np, np.float32(611.2), _f32_exp(np, exponent))
    es = np.minimum(es, _f32_mul(np, pressure, np.float32(0.5))).astype(np.float32)
    return _f32_div(np, _f32_mul(np, eps, es), _f32_sub(np, pressure, es))


def _rsif(np: Any, pressure: Any, temperature: Any, eps: Any) -> Any:
    numerator = _f32_mul(np, np.float32(21.8745584), _f32_sub(np, temperature, np.float32(273.15)))
    exponent = _f32_div(np, numerator, _f32_sub(np, temperature, np.float32(7.66)))
    es = _f32_mul(np, np.float32(611.2), _f32_exp(np, exponent))
    es = np.minimum(es, _f32_mul(np, pressure, np.float32(0.5))).astype(np.float32)
    return _f32_div(np, _f32_mul(np, eps, es), _f32_sub(np, pressure, es))


def _saturated_nm(
    np: Any,
    *,
    upper_t: Any,
    lower_t: Any,
    upper_qt: Any,
    lower_qt: Any,
    tavg: Any,
    qtavg: Any,
    qvs: Any,
    condensate_liquid: Any,
    condensate_ice: Any,
    qv: Any,
    metric_rdz: Any,
    constants: Mapping[str, Any],
    ice: bool,
) -> Any:
    cpml = _f32_add(np, constants["cp"], _f32_mul(np, constants["cpv"], qv))
    cpml = _f32_add(np, cpml, _f32_mul(np, constants["cpl"], condensate_liquid))
    if condensate_ice != np.float32(0.0):
        cpml = _f32_add(np, cpml, _f32_mul(np, constants["cpi"], condensate_ice))
    if ice:
        lh = _f32_sub(np, constants["ls1"], _f32_mul(np, constants["ls2"], tavg))
        drdt = _f32_mul(np, np.float32(21.8745584), _f32_sub(np, np.float32(273.15), np.float32(7.66)))
        denominator = _f32_mul(np, _f32_sub(np, tavg, np.float32(7.66)), _f32_sub(np, tavg, np.float32(7.66)))
    else:
        lh = _f32_sub(np, constants["lv1"], _f32_mul(np, constants["lv2"], tavg))
        drdt = _f32_mul(np, np.float32(17.67), _f32_sub(np, np.float32(273.15), np.float32(29.65)))
        denominator = _f32_mul(np, _f32_sub(np, tavg, np.float32(29.65)), _f32_sub(np, tavg, np.float32(29.65)))
    drdt = _f32_div(np, _f32_mul(np, drdt, qvs), denominator)
    gamma = _f32_mul(np, constants["g"], _f32_add(np, np.float32(1.0), qtavg))
    lh_qvs = _f32_mul(np, lh, qvs)
    rd_tavg = _f32_mul(np, constants["rd"], tavg)
    gamma = _f32_mul(np, gamma, _f32_add(np, np.float32(1.0), _f32_div(np, lh_qvs, rd_tavg)))
    gamma = _f32_div(np, gamma, _f32_add(np, cpml, _f32_mul(np, lh, drdt)))
    first = _f32_mul(np, _f32_log(np, _f32_div(np, upper_t, lower_t)), metric_rdz)
    first = _f32_add(np, first, _f32_div(np, gamma, tavg))
    second = _f32_mul(np, tavg, drdt)
    second = _f32_div(np, second, _f32_add(np, constants["eps"], qvs))
    first = _f32_mul(np, first, _f32_add(np, np.float32(1.0), second))
    moisture = _f32_log(
        np,
        _f32_div(
            np,
            _f32_add(np, np.float32(1.0), upper_qt),
            _f32_add(np, np.float32(1.0), lower_qt),
        ),
    )
    moisture = _f32_mul(np, moisture, metric_rdz)
    return _f32_mul(np, constants["g"], _f32_sub(np, first, moisture))


@dataclass(frozen=True, slots=True)
class CpuMoistNmResult:
    columns: int
    interior_words: int
    saturated_words: int
    liquid_words: int
    mixed_words: int
    ice_words: int
    minimum_nm: float
    maximum_nm: float


def cpu_moist_nm_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    selection: MoistNmSelection = DEFAULT_MOIST_NM_SELECTION,
) -> CpuMoistNmResult:
    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dz_value = _validated_dz(dz)
    a = _require_host_arrays(arrays, ni, nj, nk)
    np = importlib.import_module("numpy")
    c = _constants(np)
    rdz = _f32_div(np, np.float32(1.0), np.float32(dz_value))
    saturated = liquid = mixed = ice = 0
    minimum = math.inf
    maximum = -math.inf
    for j in range(nj):
        for i in range(ni):
            for k in range(nk):
                qv = a["q3d"][0, k, j, i]
                qc = a["q3d"][1, k, j, i]
                qi = a["q3d"][3, k, j, i]
                qvci = _f32_add(np, _f32_add(np, qv, qc), qi)
                qt = np.float32(0.0)
                for species in range(6):
                    qt = _f32_add(np, qt, a["q3d"][species, k, j, i])
                theta = _f32_add(np, a["th0"][k], a["th3d"][k, j, i])
                exner = _f32_add(np, a["pi0"][k], a["pp3d"][k, j, i])
                temperature = _f32_mul(np, theta, exner)
                thv = _f32_mul(np, theta, _f32_add(np, np.float32(1.0), _f32_mul(np, c["reps"], qv)))
                thv = _f32_div(np, thv, _f32_add(np, np.float32(1.0), qt))
                a["qvci"][k, j, i] = qvci
                a["qt"][k, j, i] = qt
                a["temperature"][k, j, i] = temperature
                a["thv"][k, j, i] = thv
                a[MASK_NAME][k, j, i] = np.uint8(0)
            a["nm"][0, j, i] = np.float32(0.0)
            a["nm"][nk, j, i] = np.float32(0.0)
            for k in range(1, nk):
                metric_rdz = _f32_mul(np, rdz, a["mf"][k, j, i])
                value = _f32_mul(
                    np,
                    c["g"],
                    _f32_mul(
                        np,
                        _f32_log(np, _f32_div(np, a["thv"][k, j, i], a["thv"][k - 1, j, i])),
                        metric_rdz,
                    ),
                )
                pavg = _f32_add(
                    np,
                    _f32_mul(np, a["c1"][k, j, i], a["prs"][k - 1, j, i]),
                    _f32_mul(np, a["c2"][k, j, i], a["prs"][k, j, i]),
                )
                tavg = _f32_add(
                    np,
                    _f32_mul(np, a["c1"][k, j, i], a["temperature"][k - 1, j, i]),
                    _f32_mul(np, a["c2"][k, j, i], a["temperature"][k, j, i]),
                )
                qtavg = _f32_add(
                    np,
                    _f32_mul(np, a["c1"][k, j, i], a["qt"][k - 1, j, i]),
                    _f32_mul(np, a["c2"][k, j, i], a["qt"][k, j, i]),
                )
                qvciavg = _f32_add(
                    np,
                    _f32_mul(np, a["c1"][k, j, i], a["qvci"][k - 1, j, i]),
                    _f32_mul(np, a["c2"][k, j, i], a["qvci"][k, j, i]),
                )
                if tavg >= np.float32(273.15):
                    qvs = _rslf(np, pavg, tavg, c["eps"])
                    ql = np.maximum(_f32_sub(np, qvciavg, qvs), np.float32(0.0)).astype(np.float32)
                    if ql > np.float32(1.0e-10):
                        qv = np.maximum(_f32_sub(np, qvciavg, ql), np.float32(0.0)).astype(np.float32)
                        value = _saturated_nm(np, upper_t=a["temperature"][k, j, i], lower_t=a["temperature"][k - 1, j, i], upper_qt=a["qt"][k, j, i], lower_qt=a["qt"][k - 1, j, i], tavg=tavg, qtavg=qtavg, qvs=qvs, condensate_liquid=ql, condensate_ice=np.float32(0.0), qv=qv, metric_rdz=metric_rdz, constants=c, ice=False)
                        a[MASK_NAME][k, j, i] = np.uint8(1)
                        saturated += 1
                        liquid += 1
                elif tavg <= np.float32(233.15):
                    qvs = _rsif(np, pavg, tavg, c["eps"])
                    qi = np.maximum(_f32_sub(np, qvciavg, qvs), np.float32(0.0)).astype(np.float32)
                    if qi > np.float32(1.0e-10):
                        qv = np.maximum(_f32_sub(np, qvciavg, qi), np.float32(0.0)).astype(np.float32)
                        value = _saturated_nm(np, upper_t=a["temperature"][k, j, i], lower_t=a["temperature"][k - 1, j, i], upper_qt=a["qt"][k, j, i], lower_qt=a["qt"][k - 1, j, i], tavg=tavg, qtavg=qtavg, qvs=qvs, condensate_liquid=np.float32(0.0), condensate_ice=qi, qv=qv, metric_rdz=metric_rdz, constants=c, ice=True)
                        a[MASK_NAME][k, j, i] = np.uint8(1)
                        saturated += 1
                        ice += 1
                else:
                    ff = _f32_div(np, _f32_sub(np, tavg, np.float32(233.15)), _f32_sub(np, np.float32(273.15), np.float32(233.15)))
                    qvsl = _rslf(np, pavg, tavg, c["eps"])
                    qvsi = _rsif(np, pavg, tavg, c["eps"])
                    one_minus = _f32_sub(np, np.float32(1.0), ff)
                    qvs = _f32_add(np, _f32_mul(np, ff, qvsl), _f32_mul(np, one_minus, qvsi))
                    excess = _f32_sub(np, qvciavg, qvs)
                    qi = np.maximum(np.float32(0.0), _f32_mul(np, one_minus, excess)).astype(np.float32)
                    ql = np.maximum(np.float32(0.0), _f32_mul(np, ff, excess)).astype(np.float32)
                    if _f32_add(np, ql, qi) > np.float32(1.0e-10):
                        qv = np.maximum(np.float32(0.0), _f32_sub(np, _f32_sub(np, qvciavg, ql), qi)).astype(np.float32)
                        nml = _saturated_nm(np, upper_t=a["temperature"][k, j, i], lower_t=a["temperature"][k - 1, j, i], upper_qt=a["qt"][k, j, i], lower_qt=a["qt"][k - 1, j, i], tavg=tavg, qtavg=qtavg, qvs=qvsl, condensate_liquid=ql, condensate_ice=qi, qv=qv, metric_rdz=metric_rdz, constants=c, ice=False)
                        nmi = _saturated_nm(np, upper_t=a["temperature"][k, j, i], lower_t=a["temperature"][k - 1, j, i], upper_qt=a["qt"][k, j, i], lower_qt=a["qt"][k - 1, j, i], tavg=tavg, qtavg=qtavg, qvs=qvsi, condensate_liquid=ql, condensate_ice=qi, qv=qv, metric_rdz=metric_rdz, constants=c, ice=True)
                        value = _f32_add(np, _f32_mul(np, ff, nml), _f32_mul(np, one_minus, nmi))
                        a[MASK_NAME][k, j, i] = np.uint8(1)
                        saturated += 1
                        mixed += 1
                a["nm"][k, j, i] = value
                minimum = min(minimum, float(value))
                maximum = max(maximum, float(value))
    return CpuMoistNmResult(ni * nj, ni * nj * (nk - 1), saturated, liquid, mixed, ice, minimum, maximum)


CUDA_MOIST_NM_SOURCE = r"""// gpuwm-cm1.cuda-moist-nm-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned char) == 1, "saturation mask requires uint8");
__device__ __forceinline__ float addf(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float subf(const float a,const float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float mulf(const float a,const float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float divf(const float a,const float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ float rslf(const float p,const float t,const float eps){
 float e=mul f(0.0f,0.0f); return e;
}
"""

# The complete kernel is appended below to keep the public source pin singular.
CUDA_MOIST_NM_SOURCE = CUDA_MOIST_NM_SOURCE.replace(
    " float e=mul f(0.0f,0.0f); return e;",
    " float x=divf(mulf(17.67f,subf(t,273.15f)),subf(t,29.65f)); float e=mulf(611.2f,expf(x)); e=fminf(e,mulf(p,0.5f)); return divf(mulf(eps,e),subf(p,e));",
) + r"""
__device__ __forceinline__ float rsif(const float p,const float t,const float eps){
 float x=divf(mulf(21.8745584f,subf(t,273.15f)),subf(t,7.66f)); float e=mulf(611.2f,expf(x)); e=fminf(e,mulf(p,0.5f)); return divf(mulf(eps,e),subf(p,e));
}
__device__ __forceinline__ float satnm(float tu,float tl,float qu,float ql0,float ta,float qta,float qvs,float cl,float ci,float qv,float mr,float g,float rd,float cp,float cpv,float cpl,float cpi,float eps,float l1,float l2,int ice){
 float cpml=addf(cp,mulf(cpv,qv)); cpml=addf(cpml,mulf(cpl,cl)); if(ci!=0.0f) cpml=addf(cpml,mulf(cpi,ci));
 float lh,dr,den; if(ice){lh=subf(l1,mulf(l2,ta));dr=mulf(21.8745584f,subf(273.15f,7.66f));den=mulf(subf(ta,7.66f),subf(ta,7.66f));}else{lh=subf(l1,mulf(l2,ta));dr=mulf(17.67f,subf(273.15f,29.65f));den=mulf(subf(ta,29.65f),subf(ta,29.65f));}
 dr=divf(mulf(dr,qvs),den); float ga=mulf(g,addf(1.0f,qta)); ga=mulf(ga,addf(1.0f,divf(mulf(lh,qvs),mulf(rd,ta)))); ga=divf(ga,addf(cpml,mulf(lh,dr)));
 float a=addf(mulf(logf(divf(tu,tl)),mr),divf(ga,ta)); a=mulf(a,addf(1.0f,divf(mulf(ta,dr),addf(eps,qvs)))); float m=mulf(logf(divf(addf(1.0f,qu),addf(1.0f,ql0))),mr); return mulf(g,subf(a,m));
}
extern "C" __global__ void cm1_moist_nm_columns_v1(const float* c1,const float* c2,const float* mf,const float* pi0,const float* th0,const float* prs,const float* pp,const float* th,const float* q,float* nm,float* temp,float* qt,float* thv,float* qvci,unsigned char* sat,unsigned int ni,unsigned int nj,unsigned int nk,float dz){
 unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,ncol=(unsigned long long)ni*nj;if(col>=ncol)return;unsigned long long nxy=ncol;
 const float g=9.81f,rd=287.04f,rv=461.5f,cp=1005.7f,cpv=1870.0f,cpl=4190.0f,cpi=2106.0f,eps=divf(rd,rv),reps=divf(rv,rd),to=273.15f;
 const float lv1=addf(2501000.0f,mulf(subf(cpl,cpv),to)),lv2=subf(cpl,cpv),ls1=addf(2834000.0f,mulf(subf(cpi,cpv),to)),ls2=subf(cpi,cpv),rdz=divf(1.0f,dz);
 for(unsigned int k=0;k<nk;k++){unsigned long long z=(unsigned long long)k*nxy+col;float qv=q[z],qc=q[(unsigned long long)nk*nxy+z],qi=q[(unsigned long long)3*nk*nxy+z];qvci[z]=addf(addf(qv,qc),qi);float sum=0.0f;for(unsigned int s=0;s<6;s++)sum=addf(sum,q[(unsigned long long)s*nk*nxy+z]);qt[z]=sum;float theta=addf(th0[k],th[z]);temp[z]=mulf(theta,addf(pi0[k],pp[z]));thv[z]=divf(mulf(theta,addf(1.0f,mulf(reps,qv))),addf(1.0f,sum));sat[z]=0;}
 nm[col]=0.0f;nm[(unsigned long long)nk*nxy+col]=0.0f;
 for(unsigned int k=1;k<nk;k++){unsigned long long z=(unsigned long long)k*nxy+col,lo=z-nxy,face=z;float mr=mulf(rdz,mf[face]);float value=mulf(g,mulf(logf(divf(thv[z],thv[lo])),mr));float pa=addf(mulf(c1[face],prs[lo]),mulf(c2[face],prs[z]));float ta=addf(mulf(c1[face],temp[lo]),mulf(c2[face],temp[z]));float qta=addf(mulf(c1[face],qt[lo]),mulf(c2[face],qt[z]));float qca=addf(mulf(c1[face],qvci[lo]),mulf(c2[face],qvci[z]));
 if(ta>=273.15f){float qs=rslf(pa,ta,eps),cl=fmaxf(subf(qca,qs),0.0f);if(cl>1.0e-10f){float qv=fmaxf(subf(qca,cl),0.0f);value=satnm(temp[z],temp[lo],qt[z],qt[lo],ta,qta,qs,cl,0.0f,qv,mr,g,rd,cp,cpv,cpl,cpi,eps,lv1,lv2,0);sat[z]=1;}}
 else if(ta<=233.15f){float qs=rsif(pa,ta,eps),ci=fmaxf(subf(qca,qs),0.0f);if(ci>1.0e-10f){float qv=fmaxf(subf(qca,ci),0.0f);value=satnm(temp[z],temp[lo],qt[z],qt[lo],ta,qta,qs,0.0f,ci,qv,mr,g,rd,cp,cpv,cpl,cpi,eps,ls1,ls2,1);sat[z]=1;}}
 else{float ff=divf(subf(ta,233.15f),subf(273.15f,233.15f)),om=subf(1.0f,ff),qsl=rslf(pa,ta,eps),qsi=rsif(pa,ta,eps),qs=addf(mulf(ff,qsl),mulf(om,qsi)),ex=subf(qca,qs),ci=fmaxf(0.0f,mulf(om,ex)),cl=fmaxf(0.0f,mulf(ff,ex));if(addf(cl,ci)>1.0e-10f){float qv=fmaxf(0.0f,subf(subf(qca,cl),ci));float nl=satnm(temp[z],temp[lo],qt[z],qt[lo],ta,qta,qsl,cl,ci,qv,mr,g,rd,cp,cpv,cpl,cpi,eps,lv1,lv2,0);float nii=satnm(temp[z],temp[lo],qt[z],qt[lo],ta,qta,qsi,cl,ci,qv,mr,g,rd,cp,cpv,cpl,cpi,eps,ls1,ls2,1);value=addf(mulf(ff,nl),mulf(om,nii));sat[z]=1;}}
 nm[face]=value;}
}
"""

CUDA_MOIST_NM_SOURCE_BYTES = CUDA_MOIST_NM_SOURCE.encode("utf-8")
CUDA_MOIST_NM_SOURCE_SHA256 = (
    "5c43c51c796185f3baaa46024a84ffe096b71b4da3506f2236902b3037240dd0"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_MOIST_NM_SOURCE_BYTES).hexdigest() == CUDA_MOIST_NM_SOURCE_SHA256


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip().lower().removeprefix("compute_").removeprefix("sm_")
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledMoistNmKernel:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_moist_nm() -> CompiledMoistNmKernel:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged moist-Nm CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(CUDA_MOIST_NM_SOURCE, "cm1_moist_nm_v1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_MOIST_NM_SOURCE, "cm1_moist_nm_v1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(r"\.entry\s+cm1_moist_nm_columns_v1(?:\s|\()", ptx.decode()):
        raise ValueError("compiled PTX lacks moist-Nm entry point")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled moist-Nm image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    identity = (("cupy_version", str(cupy.__version__)), ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())), ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())), ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"), ("device_id", str(int(device.id))), ("device_name", str(name)))
    return CompiledMoistNmKernel(module, module.get_function(CUDA_MOIST_NM_KERNEL_NAMES[0]), ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = moist_nm_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing moist-Nm arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        expected = cupy.dtype("uint8") if name == MASK_NAME else cupy.dtype("float32")
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != expected:
            raise TypeError(f"{name} must have dtype {expected.name}")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous with shape {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("moist-Nm arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaMoistNmLaunch:
    ready_event: Any
    compiled: CompiledMoistNmKernel
    receipt: dict[str, Any]


def launch_cuda_moist_nm(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dz: float, selection: MoistNmSelection = DEFAULT_MOIST_NM_SELECTION, stream: Any | None = None, compiled: CompiledMoistNmKernel | None = None) -> CudaMoistNmLaunch:
    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dz_value = _validated_dz(dz)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_moist_nm()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    columns = ni * nj
    grid = (columns + CUDA_MOIST_NM_THREADS - 1) // CUDA_MOIST_NM_THREADS
    compiled.kernel((grid, 1, 1), (CUDA_MOIST_NM_THREADS, 1, 1), tuple(a[name] for name in ALL_ARRAY_NAMES) + (cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk), cupy.float32(dz_value)), shared_mem=0, stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {"schema": CUDA_MOIST_NM_SCHEMA, "claim_limit": CUDA_MOIST_NM_CLAIM_LIMIT, "source_anchors": SOURCE_ANCHORS, "selection": {option: getattr(selection, attribute) for option, (attribute, _) in _EXPECTED_SELECTION.items()}, "domain": {"ni": ni, "nj": nj, "nk": nk, "columns": columns}, "dz_m": dz_value, "kernel": {"abi": CUDA_MOIST_NM_ABI, "name": CUDA_MOIST_NM_KERNEL_NAMES[0], "source_sha256": CUDA_MOIST_NM_SOURCE_SHA256, "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(), "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(), "target_arch": compiled.target_arch, "ptx_options": list(compiled.ptx_options), "cubin_options": list(compiled.cubin_options)}, "runtime": dict(compiled.runtime_identity), "launch": {"grid_x": grid, "block_x": CUDA_MOIST_NM_THREADS, "stream_pointer": int(stream.ptr), "event_pointer": int(event.ptr)}}
    return CudaMoistNmLaunch(event, compiled, receipt)


__all__ = ["ALL_ARRAY_NAMES", "CUDA_MOIST_NM_ABI", "CUDA_MOIST_NM_CLAIM_LIMIT", "CUDA_MOIST_NM_KERNEL_NAMES", "CUDA_MOIST_NM_SCHEMA", "CUDA_MOIST_NM_SOURCE", "CUDA_MOIST_NM_SOURCE_SHA256", "CompiledMoistNmKernel", "CpuMoistNmResult", "CudaMoistNmLaunch", "DEFAULT_MOIST_NM_SELECTION", "FLOAT_OUTPUT_NAMES", "INPUT_ARRAY_NAMES", "MASK_NAME", "MoistNmSelection", "SOURCE_ANCHORS", "compile_cuda_moist_nm", "cpu_moist_nm_inplace", "launch_cuda_moist_nm", "moist_nm_array_shapes"]
