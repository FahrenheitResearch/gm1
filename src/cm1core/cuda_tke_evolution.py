"""CM1 r21.1 terrain TKE diffusion, advection, RK update, and commit.

The admitted path is the source-ordered ``turbt`` operator followed by the
three ``advw(stag=4)`` calls selected for SGS TKE.  Stages one and two use the
fifth-order flux reconstruction; stage three uses WENO-Z because
``advwenos=2``.  Four-open inflow corrections and ``bcw(tke3d,1)`` rigid
bottom/top values are included.
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


CUDA_TKE_EVOLUTION_SCHEMA = "gpuwm-cm1.cuda-tke-evolution/v1"
CUDA_TKE_EVOLUTION_ABI = "gpuwm-cm1.cuda-tke-evolution-kernels/v1"
CUDA_TKE_EVOLUTION_THREADS = 128
CUDA_TKE_EVOLUTION_MAX_NK = 75
CUDA_TKE_EVOLUTION_NGXY = 3
CUDA_TKE_EVOLUTION_KERNEL_NAMES = (
    "cm1_turbt_terrain_v1",
    "cm1_tke_advw_rk_v1",
    "cm1_tke_commit_v1",
    "cm1_tke_copy_stage_v1",
)
CUDA_TKE_EVOLUTION_CLAIM_LIMIT = (
    "CM1 r21.1 terrain turbt plus three-stage SGS-TKE advw/RK/clip/bcw/commit "
    "for the serial Cartesian uniform-horizontal four-open product grid, "
    "hadvordrs=vadvordrs=5, advwenos=2, weno_order=5, doimpl=1, vialpha=1, "
    "vibeta=0, dt/(3,2,1), supplied live mass flux/divergence, and no pdef; "
    "mass-flux production, local TKE source, MPI, other stencils/boundaries, "
    "other SGS/PBL branches, coupled state recomputation between large steps, "
    "forecast, storm, and tornado claims are excluded"
)
SOURCE_ANCHORS = {
    "turbt_dispatch": "src/solve1.F:314-369",
    "turbt": "src/turbtend.F:700-1044",
    "rk_schedule": "src/solve2.F:294-299",
    "tke_advection_update": "src/solve2.F:1708-1803",
    "advw": "src/adv.F:2778-3610",
    "weno_horizontal": "src/adv_routines.F:204-430",
    "weno_vertical": "src/adv_routines.F:2359-2557;src/adv_routines.F:4038-4394",
    "commit": "src/solve3.F:560-590",
    "bcw": "src/bc.F:942-1143",
}
AUTHORITY_TARBALL_SHA256 = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
AUTHORITY_SOURCE_SHA256 = {
    "src/adv.F": "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f",
    "src/adv_routines.F": "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    "src/solve3.F": "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393",
    "src/turbtend.F": "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de",
}


@dataclass(frozen=True, slots=True)
class TkeEvolutionSelection:
    terrain: bool = True
    axisymmetric: int = 0
    lateral_boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)
    serial_full_domain: bool = True
    doimpl: int = 1
    scalar_horizontal_order: int = 5
    scalar_vertical_order: int = 5
    scalar_weno: int = 2
    weno_order: int = 5
    nrkmax: int = 3
    positive_definite: bool = False


DEFAULT_TKE_EVOLUTION_SELECTION = TkeEvolutionSelection()

def _validate_selection(selection: TkeEvolutionSelection) -> None:
    if not isinstance(selection, TkeEvolutionSelection):
        raise TypeError("selection must be a TkeEvolutionSelection")
    if selection != DEFAULT_TKE_EVOLUTION_SELECTION:
        refuse(code="CM1-TKE-EVOLUTION-UNSUPPORTED", gate="physics-seam", option="tke_evolution_selection", value={n: getattr(selection, n) for n in selection.__dataclass_fields__}, reason="another value selects a different turbt, advw, boundary, or cadence branch", remedy="use the sealed real-product selection or add an independent native oracle")


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 7), ("nj", nj, 7), ("nk", nk, 8)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max(ni, nj, nk) > (1 << 31) - 8:
        raise ValueError("TKE dimensions plus halos exceed signed CUDA indexing")
    if (nk + 3) * (nj + 7) * (ni + 7) > ((1 << 63) - 1) // 4:
        raise ValueError("TKE float32 allocation exceeds signed size_t")


def _spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def tke_evolution_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk); ng = CUDA_TKE_EVOLUTION_NGXY
    mass = (nk, nj, ni); face = (nk + 1, nj, ni)
    return {
        "tkea": face, "tke3d": face, "tketen": face, "kmh": face,
        "kmv": face, "rho": mass, "rf": face, "mh": mass, "mf": face,
        "gz": (nj, ni), "rgz": (nj, ni), "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni), "c1": face, "c2": face, "rrf0": face,
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng), "divx": mass,
    }


_NAMES = tuple(tke_evolution_shapes(7, 7, 8))


def _require_host(arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> tuple[Any, dict[str, Any]]:
    np = importlib.import_module("numpy"); shapes = tke_evolution_shapes(ni, nj, nk)
    missing = [n for n in _NAMES if n not in arrays]
    if missing: raise ValueError(f"missing TKE evolution arrays: {', '.join(missing)}")
    out: dict[str, Any] = {}
    for name in _NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray): raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"): raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name] or not value.flags.c_contiguous: raise ValueError(f"{name} must have C-order shape {shapes[name]}")
        out[name] = value
    if len({int(v.ctypes.data) for v in out.values()}) != len(out): raise ValueError("TKE evolution arrays must not alias")
    return np, out


def _a(np: Any, x: Any, y: Any) -> Any: return np.float32(np.float32(x) + np.float32(y))
def _s(np: Any, x: Any, y: Any) -> Any: return np.float32(np.float32(x) - np.float32(y))
def _m(np: Any, x: Any, y: Any) -> Any: return np.float32(np.float32(x) * np.float32(y))
def _d(np: Any, x: Any, y: Any) -> Any: return np.float32(np.float32(x) / np.float32(y))
def _c(x: int, n: int) -> int: return min(max(x, 0), n - 1)
def _cell(a: Any, k: int, j: int, i: int, nj: int, ni: int) -> Any: return a[k, _c(j, nj), _c(i, ni)]
def _g(a: Any, j: int, i: int, nj: int, ni: int) -> Any: return a[_c(j, nj), _c(i, ni)]


def _flx3(np: Any, x: Any, y: Any, z: Any) -> Any:
    return _a(np, _a(np, _m(np, -1.0/6.0, x), _m(np, 5.0/6.0, y)), _m(np, 2.0/6.0, z))


def _flx5(np: Any, v: tuple[Any, Any, Any, Any, Any]) -> Any:
    r = _m(np, 2.0/60.0, v[0]); r = _a(np, r, _m(np, -13.0/60.0, v[1])); r = _a(np, r, _m(np, 47.0/60.0, v[2])); r = _a(np, r, _m(np, 27.0/60.0, v[3])); return _a(np, r, _m(np, -3.0/60.0, v[4]))


def _weno5(np: Any, v: tuple[Any, Any, Any, Any, Any], weps: float) -> Any:
    s1,s2,s3,s4,s5=(np.float32(x) for x in v)
    d1=_a(np,_s(np,s1,_m(np,2,s2)),s3); e1=_a(np,_s(np,s1,_m(np,4,s2)),_m(np,3,s3)); b1=_a(np,_m(np,13/12,_m(np,d1,d1)),_m(np,.25,_m(np,e1,e1)))
    d2=_a(np,_s(np,s2,_m(np,2,s3)),s4); e2=_s(np,s2,s4); b2=_a(np,_m(np,13/12,_m(np,d2,d2)),_m(np,.25,_m(np,e2,e2)))
    d3=_a(np,_s(np,s3,_m(np,2,s4)),s5); e3=_a(np,_s(np,_m(np,3,s3),_m(np,4,s4)),s5); b3=_a(np,_m(np,13/12,_m(np,d3,d3)),_m(np,.25,_m(np,e3,e3)))
    tau=abs(float(b1)-float(b3)); q1=min(1e30,tau/(float(b1)+weps)); q2=min(1e30,tau/(float(b2)+weps)); q3=min(1e30,tau/(float(b3)+weps))
    w1=float(np.float32(.1))*(1+q1*q1); w2=float(np.float32(.6))*(1+q2*q2); w3=float(np.float32(.3))*(1+q3*q3)  # q*q as the WENO5 device function and gfortran's **2, not pow()
    p1=_a(np,_a(np,_m(np,2/6,s1),_m(np,-7/6,s2)),_m(np,11/6,s3)); p2=_a(np,_a(np,_m(np,-1/6,s2),_m(np,5/6,s3)),_m(np,2/6,s4)); p3=_a(np,_a(np,_m(np,2/6,s3),_m(np,5/6,s4)),_m(np,-1/6,s5))
    return np.float32((w1*float(p1)+w2*float(p2)+w3*float(p3))/(w1+w2+w3))


def _upstrpd(np: Any, s1: Any, s2: Any, s3: Any, weps: float) -> Any:
    dd=_s(np,s2,s1); se=np.copysign(np.float32(abs(np.float32(weps))),dd); rr=_d(np,_s(np,s3,s2),_a(np,se,dd)); phi=np.maximum(np.float32(0),np.minimum(rr,np.minimum(_a(np,np.float32(1/6),_m(np,2/6,rr)),np.float32(1)))); return _a(np,s2,_m(np,phi,_s(np,s2,s1)))


def _hflux(np: Any, q: Any, a: Mapping[str, Any], *, f: int, j: int, face: int, axis: int, ni: int, nj: int, ng: int, weno: bool, weps: float, override: bool = True) -> tuple[Any, Any]:
    if axis == 0:
        l,r=face-1,face; c2=_m(np,.5,_a(np,_cell(a["c2"],f,j,l,nj,ni),_cell(a["c2"],f,j,r,nj,ni))); mass=_a(np,_m(np,c2,a["rru"][f+1,j+ng,face+ng]),_m(np,_s(np,1,c2),a["rru"][f,j+ng,face+ng]))
        if override and face==0 and mass>=0:
            flux,_ = _hflux(np,q,a,f=f,j=j,face=1,axis=axis,ni=ni,nj=nj,ng=ng,weno=weno,weps=weps,override=False)
            return flux,mass
        if override and face==ni and mass<=0:
            flux,_ = _hflux(np,q,a,f=f,j=j,face=ni-1,axis=axis,ni=ni,nj=nj,ng=ng,weno=weno,weps=weps,override=False)
            return flux,mass
        ids=(-3,-2,-1,0,1) if mass>=0 else (2,1,0,-1,-2); vals=tuple(q[f,j,_c(face+x,ni)] for x in ids)
    else:
        s,n=face-1,face; c2=_m(np,.5,_a(np,_cell(a["c2"],f,s,j,nj,ni),_cell(a["c2"],f,n,j,nj,ni))); mass=_a(np,_m(np,c2,a["rrv"][f+1,face+ng,j+ng]),_m(np,_s(np,1,c2),a["rrv"][f,face+ng,j+ng]))
        if override and face==0 and mass>=0:
            flux,_ = _hflux(np,q,a,f=f,j=j,face=1,axis=axis,ni=ni,nj=nj,ng=ng,weno=weno,weps=weps,override=False)
            return flux,mass
        if override and face==nj and mass<=0:
            flux,_ = _hflux(np,q,a,f=f,j=j,face=nj-1,axis=axis,ni=ni,nj=nj,ng=ng,weno=weno,weps=weps,override=False)
            return flux,mass
        ids=(-3,-2,-1,0,1) if mass>=0 else (2,1,0,-1,-2); vals=tuple(q[f,_c(face+x,nj),j] for x in ids)
    return _m(np,mass,_weno5(np,vals,weps) if weno else _flx5(np,vals)), mass


def _vflux(np: Any, q: Any, rrw: Any, ks: int, j: int, i: int, nk: int, ng: int, weno: bool, weps: float) -> Any:
    # k=nk-1, nk: vadv_lwr_weno_35/_23 test wbar>0, vadv_lwr_flx_35/_23 test wbar>=0.
    mass=_m(np,.5,_a(np,rrw[ks,j+ng,i+ng],rrw[ks+1,j+ng,i+ng]))
    if ks==1: rec=_m(np,.5,_a(np,q[0,j,i],q[1,j,i])) if mass>=0 else (_upstrpd(np,q[2,j,i],q[1,j,i],q[0,j,i],weps) if weno else _flx3(np,q[2,j,i],q[1,j,i],q[0,j,i]))
    elif ks==2: rec=(_upstrpd(np,q[0,j,i],q[1,j,i],q[2,j,i],weps) if weno else _flx3(np,q[0,j,i],q[1,j,i],q[2,j,i])) if mass>=0 else (_weno5(np,tuple(q[x,j,i] for x in (4,3,2,1,0)),weps) if weno else _flx5(np,tuple(q[x,j,i] for x in (4,3,2,1,0))))
    elif ks==nk-1: rec=(_weno5(np,tuple(q[x,j,i] for x in range(ks-3,ks+2)),weps) if weno else _flx5(np,tuple(q[x,j,i] for x in range(ks-3,ks+2)))) if (mass>0 if weno else mass>=0) else (_upstrpd(np,q[ks+1,j,i],q[ks,j,i],q[ks-1,j,i],weps) if weno else _flx3(np,q[ks+1,j,i],q[ks,j,i],q[ks-1,j,i]))
    elif ks==nk: rec=(_upstrpd(np,q[nk-2,j,i],q[nk-1,j,i],q[nk,j,i],weps) if weno else _flx3(np,q[nk-2,j,i],q[nk-1,j,i],q[nk,j,i])) if (mass>0 if weno else mass>=0) else _m(np,.5,_a(np,q[nk-1,j,i],q[nk,j,i]))
    elif mass>=0: rec=_weno5(np,tuple(q[x,j,i] for x in range(ks-3,ks+2)),weps) if weno else _flx5(np,tuple(q[x,j,i] for x in range(ks-3,ks+2)))
    else: rec=_weno5(np,tuple(q[x,j,i] for x in range(ks+2,ks-3,-1)),weps) if weno else _flx5(np,tuple(q[x,j,i] for x in range(ks+2,ks-3,-1)))
    return _m(np,mass,rec)


def cpu_tke_evolution_inplace(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dx: float, dy: float, dz: float, dt: float, selection: TkeEvolutionSelection = DEFAULT_TKE_EVOLUTION_SELECTION) -> dict[str, object]:
    _validate_selection(selection); _validate_grid(ni,nj,nk); dx,dy,dz,dt=(_spacing(n,v) for n,v in (("dx",dx),("dy",dy),("dz",dz),("dt",dt))); np,a=_require_host(arrays,ni,nj,nk); ng=CUDA_TKE_EVOLUTION_NGXY
    rdx=_d(np,1,dx);rdy=_d(np,1,dy);rdz=_d(np,1,dz);dt32=np.float32(dt);zt=_m(np,nk,dz);zero=np.float32(0);half=np.float32(.5)
    t=a["tke3d"]; hx=np.zeros_like(t);hy=np.zeros_like(t)
    # terrain horizontal turbt, source flux order.
    tx=np.zeros((nk+1,nj,ni+1),np.float32);ty=np.zeros((nk+1,nj+1,ni),np.float32)
    sm=_m(np,.5,_a(np,t[:-1],t[1:]))
    for f in range(1,nk):
        for j in range(nj):
            for face in range(ni+1):
                l,r=face-1,face; den=_a(np,_m(np,_g(a["gz"],j,r,nj,ni),_cell(a["rf"],f,j,r,nj,ni)),_m(np,_g(a["gz"],j,l,nj,ni),_cell(a["rf"],f,j,l,nj,ni))); kval=_a(np,_cell(a["kmh"],f,j,r,nj,ni),_cell(a["kmh"],f,j,l,nj,ni)); term=_m(np,_s(np,_m(np,_cell(t,f,j,r,nj,ni),_g(a["rgz"],j,r,nj,ni)),_m(np,_cell(t,f,j,l,nj,ni),_g(a["rgz"],j,l,nj,ni))),rdx); up=_m(np,_s(np,zt,_m(np,f+.5,dz)),_a(np,_cell(sm,f,j,l,nj,ni),_cell(sm,f,j,r,nj,ni)));lo=_m(np,_s(np,zt,_m(np,f-.5,dz)),_a(np,_cell(sm,f-1,j,l,nj,ni),_cell(sm,f-1,j,r,nj,ni)));term=_a(np,term,_m(np,.5,_m(np,_s(np,up,lo),_m(np,rdz,_m(np,_s(np,_g(a["rgz"],j,r,nj,ni),_g(a["rgz"],j,l,nj,ni)),rdx)))));tx[f,j,face]=_m(np,-.25,_m(np,den,_m(np,2,_m(np,kval,term))))
        for face in range(nj+1):
            s,n=face-1,face
            for i in range(ni):
                den=_a(np,_m(np,_g(a["gz"],n,i,nj,ni),_cell(a["rf"],f,n,i,nj,ni)),_m(np,_g(a["gz"],s,i,nj,ni),_cell(a["rf"],f,s,i,nj,ni)));kval=_a(np,_cell(a["kmh"],f,n,i,nj,ni),_cell(a["kmh"],f,s,i,nj,ni));term=_m(np,_s(np,_m(np,_cell(t,f,n,i,nj,ni),_g(a["rgz"],n,i,nj,ni)),_m(np,_cell(t,f,s,i,nj,ni),_g(a["rgz"],s,i,nj,ni))),rdy);up=_m(np,_s(np,zt,_m(np,f+.5,dz)),_a(np,_cell(sm,f,s,i,nj,ni),_cell(sm,f,n,i,nj,ni)));lo=_m(np,_s(np,zt,_m(np,f-.5,dz)),_a(np,_cell(sm,f-1,s,i,nj,ni),_cell(sm,f-1,n,i,nj,ni)));term=_a(np,term,_m(np,.5,_m(np,_s(np,up,lo),_m(np,rdz,_m(np,_s(np,_g(a["rgz"],n,i,nj,ni),_g(a["rgz"],s,i,nj,ni)),rdy)))));ty[f,face,i]=_m(np,-.25,_m(np,den,_m(np,2,_m(np,kval,term))))
    sx=_m(np,.5,_a(np,tx[:-1],tx[1:]));sy=_m(np,.5,_a(np,ty[:-1],ty[1:]))
    for f in range(1,nk):
        for j in range(nj):
            for i in range(ni):
                x=_m(np,-1,_m(np,_s(np,_m(np,tx[f,j,i+1],a["rgzu"][j,i+1]),_m(np,tx[f,j,i],a["rgzu"][j,i])),_m(np,a["gz"][j,i],rdx)));up=_m(np,_s(np,zt,_m(np,f+.5,dz)),_a(np,sx[f,j,i],sx[f,j,i+1]));lo=_m(np,_s(np,zt,_m(np,f-.5,dz)),_a(np,sx[f-1,j,i],sx[f-1,j,i+1]));x=_s(np,x,_m(np,.5,_m(np,_s(np,up,lo),_m(np,rdz,_m(np,_s(np,a["rgzu"][j,i+1],a["rgzu"][j,i]),_m(np,a["gz"][j,i],rdx))))));y=_m(np,-1,_m(np,_s(np,_m(np,ty[f,j+1,i],a["rgzv"][j+1,i]),_m(np,ty[f,j,i],a["rgzv"][j,i])),_m(np,a["gz"][j,i],rdy)));up=_m(np,_s(np,zt,_m(np,f+.5,dz)),_a(np,sy[f,j,i],sy[f,j+1,i]));lo=_m(np,_s(np,zt,_m(np,f-.5,dz)),_a(np,sy[f-1,j,i],sy[f-1,j+1,i]));y=_s(np,y,_m(np,.5,_m(np,_s(np,up,lo),_m(np,rdz,_m(np,_s(np,a["rgzv"][j+1,i],a["rgzv"][j,i]),_m(np,a["gz"][j,i],rdy))))));hx[f,j,i]=zero if i in (0,ni-1) else x;hy[f,j,i]=zero if j in (0,nj-1) else y
    tema=_m(np,-1,_m(np,dt32,_m(np,rdz,rdz)));rdt=_d(np,1,dt32)
    for j in range(nj):
        for i in range(ni):
            n=nk-1;cp=np.empty(n,np.float32);dp=np.empty(n,np.float32);sol=np.empty(n,np.float32);f=1;rrf=_d(np,a["mf"][f,j,i],a["rf"][f,j,i]);r2=_m(np,_a(np,a["kmv"][f,j,i],a["kmv"][f+1,j,i]),_m(np,a["mh"][f,j,i],_m(np,a["rho"][f,j,i],rrf)));cfc=_m(np,tema,r2);cfb=_s(np,1,cfc);inv=_d(np,1,cfb);cp[0]=_m(np,-1,_m(np,cfc,inv));dp[0]=_m(np,t[f,j,i],inv)
            for p,f in enumerate(range(2,nk-1),start=1):
                rrf=_d(np,a["mf"][f,j,i],a["rf"][f,j,i]);r1=_m(np,_a(np,a["kmv"][f-1,j,i],a["kmv"][f,j,i]),_m(np,a["mh"][f-1,j,i],_m(np,a["rho"][f-1,j,i],rrf)));r2=_m(np,_a(np,a["kmv"][f,j,i],a["kmv"][f+1,j,i]),_m(np,a["mh"][f,j,i],_m(np,a["rho"][f,j,i],rrf)));cfa=_m(np,tema,r1);cfc=_m(np,tema,r2);cfb=_s(np,_s(np,1,cfa),cfc);inv=_d(np,1,_a(np,_m(np,cfa,cp[p-1]),cfb));cp[p]=_m(np,-1,_m(np,cfc,inv));dp[p]=_m(np,_s(np,t[f,j,i],_m(np,cfa,dp[p-1])),inv)
            p=n-1;f=nk-1;rrf=_d(np,a["mf"][f,j,i],a["rf"][f,j,i]);r1=_m(np,_a(np,a["kmv"][f-1,j,i],a["kmv"][f,j,i]),_m(np,a["mh"][f-1,j,i],_m(np,a["rho"][f-1,j,i],rrf)));cfa=_m(np,tema,r1);cfb=_s(np,1,cfa);sol[p]=_m(np,_s(np,t[f,j,i],_m(np,cfa,dp[p-1])),_d(np,1,_a(np,_m(np,cfa,cp[p-1]),cfb)))
            for p in range(n-2,-1,-1):sol[p]=_a(np,_m(np,cp[p],sol[p+1]),dp[p])
            for p,f in enumerate(range(1,nk)):
                vz=_m(np,a["rf"][f,j,i],_m(np,_s(np,sol[p],t[f,j,i]),rdt));a["tketen"][f,j,i]=_a(np,a["tketen"][f,j,i],_d(np,_a(np,_a(np,hx[f,j,i],hy[f,j,i]),vz),a["rf"][f,j,i]))
    baseline=a["tkea"].copy();cur=t.copy();weps=float(np.float32(np.float32(100)*np.float32(1e-18)))
    stage_hashes=[]
    for nrk in (1,2,3):
        nxt=np.zeros_like(cur);dttmp=_d(np,dt32,np.float32(4-nrk));use_weno=nrk==3
        for f in range(1,nk):
            for j in range(nj):
                for i in range(ni):
                    fx0,mw=_hflux(np,cur,a,f=f,j=j,face=i,axis=0,ni=ni,nj=nj,ng=ng,weno=use_weno,weps=weps);fx1,me=_hflux(np,cur,a,f=f,j=j,face=i+1,axis=0,ni=ni,nj=nj,ng=ng,weno=use_weno,weps=weps);fy0,ms=_hflux(np,cur,a,f=f,j=i,face=j,axis=1,ni=ni,nj=nj,ng=ng,weno=use_weno,weps=weps);fy1,mn=_hflux(np,cur,a,f=f,j=i,face=j+1,axis=1,ni=ni,nj=nj,ng=ng,weno=use_weno,weps=weps);ax=_m(np,-1,_m(np,_s(np,fx1,fx0),rdx));ay=_m(np,-1,_m(np,_s(np,fy1,fy0),rdy));c2=a["c2"][f,j,i];c1=_s(np,1,c2)
                    if i==0 and mw>=0:ax=_s(np,ax,_m(np,cur[f,j,i],_m(np,_a(np,_m(np,c1,_s(np,a["rru"][f,j+ng,i+1+ng],a["rru"][f,j+ng,i+ng])),_m(np,c2,_s(np,a["rru"][f+1,j+ng,i+1+ng],a["rru"][f+1,j+ng,i+ng]))),rdx)))
                    elif i==ni-1 and me<=0:ax=_s(np,ax,_m(np,cur[f,j,i],_m(np,_a(np,_m(np,c1,_s(np,a["rru"][f,j+ng,i+1+ng],a["rru"][f,j+ng,i+ng])),_m(np,c2,_s(np,a["rru"][f+1,j+ng,i+1+ng],a["rru"][f+1,j+ng,i+ng]))),rdx)))
                    if j==0 and ms>=0:ay=_s(np,ay,_m(np,cur[f,j,i],_m(np,_a(np,_m(np,c1,_s(np,a["rrv"][f,j+1+ng,i+ng],a["rrv"][f,j+ng,i+ng])),_m(np,c2,_s(np,a["rrv"][f+1,j+1+ng,i+ng],a["rrv"][f+1,j+ng,i+ng]))),rdy)))
                    elif j==nj-1 and mn<=0:ay=_s(np,ay,_m(np,cur[f,j,i],_m(np,_a(np,_m(np,c1,_s(np,a["rrv"][f,j+1+ng,i+ng],a["rrv"][f,j+ng,i+ng])),_m(np,c2,_s(np,a["rrv"][f+1,j+1+ng,i+ng],a["rrv"][f+1,j+ng,i+ng]))),rdy)))
                    az=_m(np,-1,_m(np,_s(np,_vflux(np,cur,a["rrw"],f+1,j,i,nk,ng,use_weno,weps),_vflux(np,cur,a["rrw"],f,j,i,nk,ng,use_weno,weps)),rdz));total=_a(np,_a(np,ax,ay),az);total=_a(np,total,_m(np,cur[f,j,i],_a(np,_m(np,c2,a["divx"][f,j,i]),_m(np,c1,a["divx"][f-1,j,i]))));wten=_a(np,a["tketen"][f,j,i],_m(np,total,_m(np,a["rrf0"][f,j,i],a["gz"][j,i])));value=_a(np,baseline[f,j,i],_m(np,dttmp,wten));nxt[f,j,i]=zero if value<np.float32(1e-10) else value
        cur=nxt;stage_hashes.append(hashlib.sha256(cur.astype('<f4',copy=False).tobytes()).hexdigest())
    a["tke3d"][:]=cur;a["tkea"][:]=cur
    return {"stage_sha256":stage_hashes,"weno_stages":[3],"committed_words":nk*nj*ni,"boundary_zero":bool((cur[[0,-1]]==0).all())}


CUDA_TKE_EVOLUTION_SOURCE = r'''// gpuwm-cm1.cuda-tke-evolution-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);}__device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);}__device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);}__device__ __forceinline__ float D(float x,float y){return __fdiv_rn(x,y);}__device__ __forceinline__ int C(int x,int n){return x<0?0:(x>=n?n-1:x);}__device__ __forceinline__ unsigned long long I3(int k,int j,int i,int ny,int nx){return((unsigned long long)k*ny+j)*nx+i;}__device__ __forceinline__ float CELL(const float*a,int k,int j,int i,int nj,int ni){return a[I3(k,C(j,nj),C(i,ni),nj,ni)];}__device__ __forceinline__ float G(const float*a,int j,int i,int nj,int ni){return a[(unsigned long long)C(j,nj)*ni+C(i,ni)];}
__device__ __forceinline__ float F3(float a,float b,float c){return A(A(M(-1.0f/6.0f,a),M(5.0f/6.0f,b)),M(2.0f/6.0f,c));}__device__ __forceinline__ float F5(float a,float b,float c,float d,float e){float r=M(2.0f/60.0f,a);r=A(r,M(-13.0f/60.0f,b));r=A(r,M(47.0f/60.0f,c));r=A(r,M(27.0f/60.0f,d));return A(r,M(-3.0f/60.0f,e));}
__device__ __forceinline__ float W5(float s1,float s2,float s3,float s4,float s5,double ep){float d1=A(S(s1,M(2,s2)),s3),e1=A(S(s1,M(4,s2)),M(3,s3)),b1=A(M(13.0f/12.0f,M(d1,d1)),M(.25f,M(e1,e1)));float d2=A(S(s2,M(2,s3)),s4),e2=S(s2,s4),b2=A(M(13.0f/12.0f,M(d2,d2)),M(.25f,M(e2,e2)));float d3=A(S(s3,M(2,s4)),s5),e3=A(S(M(3,s3),M(4,s4)),s5),b3=A(M(13.0f/12.0f,M(d3,d3)),M(.25f,M(e3,e3)));double tau=fabs((double)b1-(double)b3),q1=fmin(1e30,tau/((double)b1+ep)),q2=fmin(1e30,tau/((double)b2+ep)),q3=fmin(1e30,tau/((double)b3+ep)),w1=(double).1f*(1+q1*q1),w2=(double).6f*(1+q2*q2),w3=(double).3f*(1+q3*q3);float p1=A(A(M(2.0f/6.0f,s1),M(-7.0f/6.0f,s2)),M(11.0f/6.0f,s3)),p2=A(A(M(-1.0f/6.0f,s2),M(5.0f/6.0f,s3)),M(2.0f/6.0f,s4)),p3=A(A(M(2.0f/6.0f,s3),M(5.0f/6.0f,s4)),M(-1.0f/6.0f,s5));return(float)((w1*p1+w2*p2+w3*p3)/(w1+w2+w3));}
__device__ __forceinline__ float UP(float a,float b,float c,float ep){float dd=S(b,a),se=copysignf(fabsf(ep),dd),rr=D(S(c,b),A(se,dd)),phi=fmaxf(0.0f,fminf(rr,fminf(A(1.0f/6.0f,M(2.0f/6.0f,rr)),1.0f)));return A(b,M(phi,S(b,a)));}
__device__ __forceinline__ float REC(float a,float b,float c,float d,float e,int w,double ep){return w?W5(a,b,c,d,e,ep):F5(a,b,c,d,e);}
// turbt is one independent cell column; horizontal fluxes are evaluated on demand.
__device__ __forceinline__ float SM(const float*t,int k,int j,int i,int nj,int ni){return M(.5f,A(CELL(t,k,j,i,nj,ni),CELL(t,k+1,j,i,nj,ni)));}
__device__ __forceinline__ float TX(const float*t,const float*km,const float*rf,const float*gz,const float*rgz,int f,int j,int face,int nk,int nj,int ni,float rd,float rz,float dz){if(f<=0||f>=nk)return 0;int l=face-1,r=face;float den=A(M(G(gz,j,r,nj,ni),CELL(rf,f,j,r,nj,ni)),M(G(gz,j,l,nj,ni),CELL(rf,f,j,l,nj,ni))),kv=A(CELL(km,f,j,r,nj,ni),CELL(km,f,j,l,nj,ni)),zt=M((float)nk,dz);float q=M(S(M(CELL(t,f,j,r,nj,ni),G(rgz,j,r,nj,ni)),M(CELL(t,f,j,l,nj,ni),G(rgz,j,l,nj,ni))),rd);float up=M(S(zt,M((float)f+.5f,dz)),A(SM(t,f,j,l,nj,ni),SM(t,f,j,r,nj,ni))),lo=M(S(zt,M((float)f-.5f,dz)),A(SM(t,f-1,j,l,nj,ni),SM(t,f-1,j,r,nj,ni)));q=A(q,M(.5f,M(S(up,lo),M(rz,M(S(G(rgz,j,r,nj,ni),G(rgz,j,l,nj,ni)),rd)))));return M(-.25f,M(den,M(2, M(kv,q))));}
__device__ __forceinline__ float TY(const float*t,const float*km,const float*rf,const float*gz,const float*rgz,int f,int face,int i,int nk,int nj,int ni,float rd,float rz,float dz){if(f<=0||f>=nk)return 0;int s=face-1,n=face;float den=A(M(G(gz,n,i,nj,ni),CELL(rf,f,n,i,nj,ni)),M(G(gz,s,i,nj,ni),CELL(rf,f,s,i,nj,ni))),kv=A(CELL(km,f,n,i,nj,ni),CELL(km,f,s,i,nj,ni)),zt=M((float)nk,dz);float q=M(S(M(CELL(t,f,n,i,nj,ni),G(rgz,n,i,nj,ni)),M(CELL(t,f,s,i,nj,ni),G(rgz,s,i,nj,ni))),rd);float up=M(S(zt,M((float)f+.5f,dz)),A(SM(t,f,s,i,nj,ni),SM(t,f,n,i,nj,ni))),lo=M(S(zt,M((float)f-.5f,dz)),A(SM(t,f-1,s,i,nj,ni),SM(t,f-1,n,i,nj,ni)));q=A(q,M(.5f,M(S(up,lo),M(rz,M(S(G(rgz,n,i,nj,ni),G(rgz,s,i,nj,ni)),rd)))));return M(-.25f,M(den,M(2,M(kv,q))));}
#if 0
extern "C" __global__ void cm1_turbt_terrain_v1(const float*t,const float*kmh,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,const float*gz,const float*rgz,const float*rgzu,const float*rgzv,float*ten,unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz,float dt){unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nj*ni;if(col>=n)return;int i=(int)(col%ni),j=(int)(col/ni),nn=(int)nk-1;float hx[74],hy[74],cp[74],dp[74],sol[74],rx=D(1,dx),ry=D(1,dy),rz=D(1,dz),zt=M((float)nk,dz);for(int p=0;p<nn;p++){int f=p+1;float x=M(-1,M(S(M(TX(t,kmh,rf,gz,rgz,f,j,i+1,nk,nj,ni,rx,rz,dz),rgzu[(unsigned long long)j*(ni+1)+i+1]),M(TX(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,rx,rz,dz),rgzu[(unsigned long long)j*(ni+1)+i])),M(gz[(unsigned long long)j*ni+i],rx)));float up=M(S(zt,M((float)f+.5f,dz)),A(M(.5f,A(TX(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,rx,rz,dz),TX(t,kmh,rf,gz,rgz,f+1,j,i,nk,nj,ni,rx,rz,dz))),M(.5f,A(TX(t,kmh,rf,gz,rgz,f,j,i+1,nk,nj,ni,rx,rz,dz),TX(t,kmh,rf,gz,rgz,f+1,j,i+1,nk,nj,ni,rx,rz,dz))))),lo=M(S(zt,M((float)f-.5f,dz)),A(M(.5f,A(TX(t,kmh,rf,gz,rgz,f-1,j,i,nk,nj,ni,rx,rz,dz),TX(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,rx,rz,dz))),M(.5f,A(TX(t,kmh,rf,gz,rgz,f-1,j,i+1,nk,nj,ni,rx,rz,dz),TX(t,kmh,rf,gz,rgz,f,j,i+1,nk,nj,ni,rx,rz,dz)))));x=S(x,M(.5f,M(S(up,lo),M(rz,M(S(rgzu[(unsigned long long)j*(ni+1)+i+1],rgzu[(unsigned long long)j*(ni+1)+i]),M(gz[(unsigned long long)j*ni+i],rx)))));float y=M(-1,M(S(M(TY(t,kmh,rf,gz,rgz,f,j+1,i,nk,nj,ni,ry,rz,dz),rgzv[(unsigned long long)(j+1)*ni+i]),M(TY(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,ry,rz,dz),rgzv[(unsigned long long)j*ni+i])),M(gz[(unsigned long long)j*ni+i],ry)));up=M(S(zt,M((float)f+.5f,dz)),A(M(.5f,A(TY(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,ry,rz,dz),TY(t,kmh,rf,gz,rgz,f+1,j,i,nk,nj,ni,ry,rz,dz))),M(.5f,A(TY(t,kmh,rf,gz,rgz,f,j+1,i,nk,nj,ni,ry,rz,dz),TY(t,kmh,rf,gz,rgz,f+1,j+1,i,nk,nj,ni,ry,rz,dz)))));lo=M(S(zt,M((float)f-.5f,dz)),A(M(.5f,A(TY(t,kmh,rf,gz,rgz,f-1,j,i,nk,nj,ni,ry,rz,dz),TY(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,ry,rz,dz))),M(.5f,A(TY(t,kmh,rf,gz,rgz,f-1,j+1,i,nk,nj,ni,ry,rz,dz),TY(t,kmh,rf,gz,rgz,f,j+1,i,nk,nj,ni,ry,rz,dz)))));y=S(y,M(.5f,M(S(up,lo),M(rz,M(S(rgzv[(unsigned long long)(j+1)*ni+i],rgzv[(unsigned long long)j*ni+i]),M(gz[(unsigned long long)j*ni+i],ry)))));hx[p]=(i==0||i==(int)ni-1)?0:x;hy[p]=(j==0||j==(int)nj-1)?0:y;}
float tema=M(-1,M(dt,M(rz,rz)));int f=1;float rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]),r2=M(A(kmv[I3(f,j,i,nj,ni)],kmv[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr))),cfc=M(tema,r2),cfb=S(1,cfc),inv=D(1,cfb);cp[0]=M(-1,M(cfc,inv));dp[0]=M(t[I3(f,j,i,nj,ni)],inv);for(int p=1;p<nn-1;p++){f=p+1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmv[I3(f-1,j,i,nj,ni)],kmv[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr)));r2=M(A(kmv[I3(f,j,i,nj,ni)],kmv[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr)));float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1,cfa),cfc);inv=D(1,A(M(cfa,cp[p-1]),cfb));cp[p]=M(-1,M(cfc,inv));dp[p]=M(S(t[I3(f,j,i,nj,ni)],M(cfa,dp[p-1])),inv);}int p=nn-1;f=nk-1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmv[I3(f-1,j,i,nj,ni)],kmv[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr))),cfa=M(tema,r1);cfb=S(1,cfa);sol[p]=M(S(t[I3(f,j,i,nj,ni)],M(cfa,dp[p-1])),D(1,A(M(cfa,cp[p-1]),cfb)));for(p=nn-2;p>=0;p--)sol[p]=A(M(cp[p],sol[p+1]),dp[p]);float rdt=D(1,dt);for(p=0;p<nn;p++){f=p+1;float vz=M(rf[I3(f,j,i,nj,ni)],M(S(sol[p],t[I3(f,j,i,nj,ni)]),rdt));unsigned long long z=I3(f,j,i,nj,ni);ten[z]=A(ten[z],D(A(A(hx[p],hy[p]),vz),rf[z]));}}
#endif

__device__ __forceinline__ float SFX(const float*t,const float*km,const float*rf,const float*gz,const float*rgz,int m,int j,int face,int nk,int nj,int ni,float rx,float rz,float dz){return M(.5f,A(TX(t,km,rf,gz,rgz,m,j,face,nk,nj,ni,rx,rz,dz),TX(t,km,rf,gz,rgz,m+1,j,face,nk,nj,ni,rx,rz,dz)));}
__device__ __forceinline__ float SFY(const float*t,const float*km,const float*rf,const float*gz,const float*rgz,int m,int face,int i,int nk,int nj,int ni,float ry,float rz,float dz){return M(.5f,A(TY(t,km,rf,gz,rgz,m,face,i,nk,nj,ni,ry,rz,dz),TY(t,km,rf,gz,rgz,m+1,face,i,nk,nj,ni,ry,rz,dz)));}
extern "C" __global__ void cm1_turbt_terrain_v1(
 const float*t,const float*kmh,const float*kmv,const float*rho,const float*rf,
 const float*mh,const float*mf,const float*gz,const float*rgz,const float*rgzu,
 const float*rgzv,float*ten,unsigned ni,unsigned nj,unsigned nk,float dx,float dy,
 float dz,float dt){
 unsigned long long col=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(col>=(unsigned long long)nj*ni)return;
 int i=(int)(col%ni),j=(int)(col/ni),nn=(int)nk-1;
 float hx[74],hy[74],cp[74],dp[74],sol[74];
 float rx=D(1,dx),ry=D(1,dy),rz=D(1,dz),zt=M((float)nk,dz);
 for(int p=0;p<nn;p++){
  int f=p+1;
  float x0=TX(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,rx,rz,dz);
  float x1=TX(t,kmh,rf,gz,rgz,f,j,i+1,nk,nj,ni,rx,rz,dz);
  float x=M(-1,M(S(M(x1,rgzu[(unsigned long long)j*(ni+1)+i+1]),M(x0,rgzu[(unsigned long long)j*(ni+1)+i])),M(gz[(unsigned long long)j*ni+i],rx)));
  float up=M(S(zt,M((float)f+.5f,dz)),A(SFX(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,rx,rz,dz),SFX(t,kmh,rf,gz,rgz,f,j,i+1,nk,nj,ni,rx,rz,dz)));
  float lo=M(S(zt,M((float)f-.5f,dz)),A(SFX(t,kmh,rf,gz,rgz,f-1,j,i,nk,nj,ni,rx,rz,dz),SFX(t,kmh,rf,gz,rgz,f-1,j,i+1,nk,nj,ni,rx,rz,dz)));
  float mx=M(rz,M(S(rgzu[(unsigned long long)j*(ni+1)+i+1],rgzu[(unsigned long long)j*(ni+1)+i]),M(gz[(unsigned long long)j*ni+i],rx)));
  x=S(x,M(.5f,M(S(up,lo),mx)));
  float y0=TY(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,ry,rz,dz);
  float y1=TY(t,kmh,rf,gz,rgz,f,j+1,i,nk,nj,ni,ry,rz,dz);
  float y=M(-1,M(S(M(y1,rgzv[(unsigned long long)(j+1)*ni+i]),M(y0,rgzv[(unsigned long long)j*ni+i])),M(gz[(unsigned long long)j*ni+i],ry)));
  up=M(S(zt,M((float)f+.5f,dz)),A(SFY(t,kmh,rf,gz,rgz,f,j,i,nk,nj,ni,ry,rz,dz),SFY(t,kmh,rf,gz,rgz,f,j+1,i,nk,nj,ni,ry,rz,dz)));
  lo=M(S(zt,M((float)f-.5f,dz)),A(SFY(t,kmh,rf,gz,rgz,f-1,j,i,nk,nj,ni,ry,rz,dz),SFY(t,kmh,rf,gz,rgz,f-1,j+1,i,nk,nj,ni,ry,rz,dz)));
  float my=M(rz,M(S(rgzv[(unsigned long long)(j+1)*ni+i],rgzv[(unsigned long long)j*ni+i]),M(gz[(unsigned long long)j*ni+i],ry)));
  y=S(y,M(.5f,M(S(up,lo),my)));
  hx[p]=(i==0||i==(int)ni-1)?0:x;hy[p]=(j==0||j==(int)nj-1)?0:y;
 }
 float tema=M(-1,M(dt,M(rz,rz)));int f=1;
 float rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);
 float r2=M(A(kmv[I3(f,j,i,nj,ni)],kmv[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr)));
 float cfc=M(tema,r2),cfb=S(1,cfc),inv=D(1,cfb);cp[0]=M(-1,M(cfc,inv));dp[0]=M(t[I3(f,j,i,nj,ni)],inv);
 for(int p=1;p<nn-1;p++){f=p+1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmv[I3(f-1,j,i,nj,ni)],kmv[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr)));r2=M(A(kmv[I3(f,j,i,nj,ni)],kmv[I3(f+1,j,i,nj,ni)]),M(mh[I3(f,j,i,nj,ni)],M(rho[I3(f,j,i,nj,ni)],rr)));float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1,cfa),cfc);inv=D(1,A(M(cfa,cp[p-1]),cfb));cp[p]=M(-1,M(cfc,inv));dp[p]=M(S(t[I3(f,j,i,nj,ni)],M(cfa,dp[p-1])),inv);}
 int p=nn-1;f=nk-1;rr=D(mf[I3(f,j,i,nj,ni)],rf[I3(f,j,i,nj,ni)]);float r1=M(A(kmv[I3(f-1,j,i,nj,ni)],kmv[I3(f,j,i,nj,ni)]),M(mh[I3(f-1,j,i,nj,ni)],M(rho[I3(f-1,j,i,nj,ni)],rr))),cfa=M(tema,r1);cfb=S(1,cfa);sol[p]=M(S(t[I3(f,j,i,nj,ni)],M(cfa,dp[p-1])),D(1,A(M(cfa,cp[p-1]),cfb)));
 for(p=nn-2;p>=0;p--)sol[p]=A(M(cp[p],sol[p+1]),dp[p]);float rdt=D(1,dt);
 for(p=0;p<nn;p++){f=p+1;float vz=M(rf[I3(f,j,i,nj,ni)],M(S(sol[p],t[I3(f,j,i,nj,ni)]),rdt));unsigned long long z=I3(f,j,i,nj,ni);ten[z]=A(ten[z],D(A(A(hx[p],hy[p]),vz),rf[z]));}
}
'''


CUDA_TKE_EVOLUTION_SOURCE += r'''
__device__ __forceinline__ float RRU(const float*a,int k,int j,int f,int ni,int nj,int ng){return a[I3(k+1,j+ng,f+ng,nj+2*ng,ni+1+2*ng)];}__device__ __forceinline__ float RRV(const float*a,int k,int f,int i,int ni,int nj,int ng){return a[I3(k+1,f+ng,i+ng,nj+1+2*ng,ni+2*ng)];}__device__ __forceinline__ float RRW(const float*a,int f,int j,int i,int ni,int nj,int ng){return a[I3(f+1,j+ng,i+ng,nj+2*ng,ni+2*ng)];}
__device__ __forceinline__ float HFM(const float*c2,const float*rru,const float*rrv,int f,int j,int face,int axis,int ni,int nj,int ng){float cc;if(axis==0){cc=M(.5f,A(CELL(c2,f,j,face-1,nj,ni),CELL(c2,f,j,face,nj,ni)));return A(M(cc,RRU(rru,f,j,face,ni,nj,ng)),M(S(1,cc),RRU(rru,f-1,j,face,ni,nj,ng)));}cc=M(.5f,A(CELL(c2,f,face-1,j,nj,ni),CELL(c2,f,face,j,nj,ni)));return A(M(cc,RRV(rrv,f,face,j,ni,nj,ng)),M(S(1,cc),RRV(rrv,f-1,face,j,ni,nj,ng)));}
__device__ __forceinline__ float HF(const float*q,const float*c2,const float*rru,const float*rrv,int f,int j,int face,int axis,int ni,int nj,int ng,int w,double ep,int ov,float*mass){float cc,m;if(axis==0){cc=M(.5f,A(CELL(c2,f,j,face-1,nj,ni),CELL(c2,f,j,face,nj,ni)));m=A(M(cc,RRU(rru,f,j,face,ni,nj,ng)),M(S(1,cc),RRU(rru,f-1,j,face,ni,nj,ng)));if(ov&&face==0&&m>=0)return HF(q,c2,rru,rrv,f,j,1,axis,ni,nj,ng,w,ep,0,mass);if(ov&&face==ni&&m<=0)return HF(q,c2,rru,rrv,f,j,ni-1,axis,ni,nj,ng,w,ep,0,mass);int b=face;if(m>=0)*mass=m;else *mass=m;float v1,v2,v3,v4,v5;if(m>=0){v1=CELL(q,f,j,b-3,nj,ni);v2=CELL(q,f,j,b-2,nj,ni);v3=CELL(q,f,j,b-1,nj,ni);v4=CELL(q,f,j,b,nj,ni);v5=CELL(q,f,j,b+1,nj,ni);}else{v1=CELL(q,f,j,b+2,nj,ni);v2=CELL(q,f,j,b+1,nj,ni);v3=CELL(q,f,j,b,nj,ni);v4=CELL(q,f,j,b-1,nj,ni);v5=CELL(q,f,j,b-2,nj,ni);}return M(m,REC(v1,v2,v3,v4,v5,w,ep));}cc=M(.5f,A(CELL(c2,f,face-1,j,nj,ni),CELL(c2,f,face,j,nj,ni)));m=A(M(cc,RRV(rrv,f,face,j,ni,nj,ng)),M(S(1,cc),RRV(rrv,f-1,face,j,ni,nj,ng)));if(ov&&face==0&&m>=0)return HF(q,c2,rru,rrv,f,j,1,axis,ni,nj,ng,w,ep,0,mass);if(ov&&face==nj&&m<=0)return HF(q,c2,rru,rrv,f,j,nj-1,axis,ni,nj,ng,w,ep,0,mass);*mass=m;int b=face;float v1,v2,v3,v4,v5;if(m>=0){v1=CELL(q,f,b-3,j,nj,ni);v2=CELL(q,f,b-2,j,nj,ni);v3=CELL(q,f,b-1,j,nj,ni);v4=CELL(q,f,b,j,nj,ni);v5=CELL(q,f,b+1,j,nj,ni);}else{v1=CELL(q,f,b+2,j,nj,ni);v2=CELL(q,f,b+1,j,nj,ni);v3=CELL(q,f,b,j,nj,ni);v4=CELL(q,f,b-1,j,nj,ni);v5=CELL(q,f,b-2,j,nj,ni);}return M(m,REC(v1,v2,v3,v4,v5,w,ep));}
__device__ __forceinline__ float HFO(const float*q,const float*c2,const float*rru,const float*rrv,int f,int j,int face,int axis,int ni,int nj,int ng,int w,double ep,int ov,float*mass){float boundary_mass=HFM(c2,rru,rrv,f,j,face,axis,ni,nj,ng);float flux=HF(q,c2,rru,rrv,f,j,face,axis,ni,nj,ng,w,ep,ov,mass);*mass=boundary_mass;return flux;}
#define HF HFO
// k=nk-1 and k=nk take the lower stencil for wbar>0 under WENO (vadv_lwr_weno_35/_23,
// adv_routines.F:4038-4396) but for wbar>=0 under flx5 (vadv_lwr_flx_35/_23, :4817-5169).
__device__ __forceinline__ float VF(const float*q,const float*rrw,int ks,int j,int i,int nk,int nj,int ni,int ng,int w,double ep){float m=M(.5f,A(RRW(rrw,ks-1,j,i,ni,nj,ng),RRW(rrw,ks,j,i,ni,nj,ng))),r;if(ks==1)r=m>=0?M(.5f,A(CELL(q,0,j,i,nj,ni),CELL(q,1,j,i,nj,ni))):(w?UP(CELL(q,2,j,i,nj,ni),CELL(q,1,j,i,nj,ni),CELL(q,0,j,i,nj,ni),(float)ep):F3(CELL(q,2,j,i,nj,ni),CELL(q,1,j,i,nj,ni),CELL(q,0,j,i,nj,ni)));else if(ks==2)r=m>=0?(w?UP(CELL(q,0,j,i,nj,ni),CELL(q,1,j,i,nj,ni),CELL(q,2,j,i,nj,ni),(float)ep):F3(CELL(q,0,j,i,nj,ni),CELL(q,1,j,i,nj,ni),CELL(q,2,j,i,nj,ni))):REC(CELL(q,4,j,i,nj,ni),CELL(q,3,j,i,nj,ni),CELL(q,2,j,i,nj,ni),CELL(q,1,j,i,nj,ni),CELL(q,0,j,i,nj,ni),w,ep);else if(ks==nk-1)r=(w?m>0:m>=0)?REC(CELL(q,ks-3,j,i,nj,ni),CELL(q,ks-2,j,i,nj,ni),CELL(q,ks-1,j,i,nj,ni),CELL(q,ks,j,i,nj,ni),CELL(q,ks+1,j,i,nj,ni),w,ep):(w?UP(CELL(q,ks+1,j,i,nj,ni),CELL(q,ks,j,i,nj,ni),CELL(q,ks-1,j,i,nj,ni),(float)ep):F3(CELL(q,ks+1,j,i,nj,ni),CELL(q,ks,j,i,nj,ni),CELL(q,ks-1,j,i,nj,ni)));else if(ks==nk)r=(w?m>0:m>=0)?(w?UP(CELL(q,nk-2,j,i,nj,ni),CELL(q,nk-1,j,i,nj,ni),CELL(q,nk,j,i,nj,ni),(float)ep):F3(CELL(q,nk-2,j,i,nj,ni),CELL(q,nk-1,j,i,nj,ni),CELL(q,nk,j,i,nj,ni))):M(.5f,A(CELL(q,nk-1,j,i,nj,ni),CELL(q,nk,j,i,nj,ni)));else if(m>=0)r=REC(CELL(q,ks-3,j,i,nj,ni),CELL(q,ks-2,j,i,nj,ni),CELL(q,ks-1,j,i,nj,ni),CELL(q,ks,j,i,nj,ni),CELL(q,ks+1,j,i,nj,ni),w,ep);else r=REC(CELL(q,ks+2,j,i,nj,ni),CELL(q,ks+1,j,i,nj,ni),CELL(q,ks,j,i,nj,ni),CELL(q,ks-1,j,i,nj,ni),CELL(q,ks-2,j,i,nj,ni),w,ep);return M(m,r);}
extern "C" __global__ void cm1_tke_advw_rk_v1(const float*q,const float*base,const float*ten,const float*c1,const float*c2,const float*rrf0,const float*gz,const float*rru,const float*rrv,const float*rrw,const float*divx,float*out,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float dx,float dy,float dz,float dttmp,unsigned w,double ep){unsigned long long z=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)(nk+1)*nj*ni;if(z>=n)return;int i=(int)(z%ni),j=(int)((z/ni)%nj),f=(int)(z/((unsigned long long)ni*nj));if(f==0||f==(int)nk){out[z]=0;return;}float rx=D(1,dx),ry=D(1,dy),rz=D(1,dz),mw,me,ms,mn,fx0=HF(q,c2,rru,rrv,f,j,i,0,ni,nj,ng,w,ep,1,&mw),fx1=HF(q,c2,rru,rrv,f,j,i+1,0,ni,nj,ng,w,ep,1,&me),fy0=HF(q,c2,rru,rrv,f,i,j,1,ni,nj,ng,w,ep,1,&ms),fy1=HF(q,c2,rru,rrv,f,i,j+1,1,ni,nj,ng,w,ep,1,&mn),ax=M(-1,M(S(fx1,fx0),rx)),ay=M(-1,M(S(fy1,fy0),ry)),cc2=c2[z],cc1=S(1,cc2);if((i==0&&mw>=0)||(i==(int)ni-1&&me<=0)){float d=A(M(cc1,S(RRU(rru,f-1,j,i+1,ni,nj,ng),RRU(rru,f-1,j,i,ni,nj,ng))),M(cc2,S(RRU(rru,f,j,i+1,ni,nj,ng),RRU(rru,f,j,i,ni,nj,ng))));ax=S(ax,M(q[z],M(d,rx)));}if((j==0&&ms>=0)||(j==(int)nj-1&&mn<=0)){float d=A(M(cc1,S(RRV(rrv,f-1,j+1,i,ni,nj,ng),RRV(rrv,f-1,j,i,ni,nj,ng))),M(cc2,S(RRV(rrv,f,j+1,i,ni,nj,ng),RRV(rrv,f,j,i,ni,nj,ng))));ay=S(ay,M(q[z],M(d,ry)));}float az=M(-1,M(S(VF(q,rrw,f+1,j,i,nk,nj,ni,ng,w,ep),VF(q,rrw,f,j,i,nk,nj,ni,ng,w,ep)),rz)),tot=A(A(ax,ay),az);tot=A(tot,M(q[z],A(M(cc2,divx[I3(f,j,i,nj,ni)]),M(cc1,divx[I3(f-1,j,i,nj,ni)]))));float wt=A(ten[z],M(tot,M(rrf0[z],gz[(unsigned long long)j*ni+i]))),v=A(base[z],M(dttmp,wt));out[z]=v<1e-10f?0:v;}
extern "C" __global__ void cm1_tke_commit_v1(const float*src,float*tke,float*base,unsigned long long n){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(q<n){tke[q]=src[q];base[q]=src[q];}}
extern "C" __global__ void cm1_tke_copy_stage_v1(const float*src,float*dst,unsigned long long n){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(q<n)dst[q]=src[q];}
'''


CUDA_TKE_EVOLUTION_SOURCE_SHA256 = hashlib.sha256(CUDA_TKE_EVOLUTION_SOURCE.encode()).hexdigest()


def _validate_column_capacity(max_nk: int) -> int:
    if isinstance(max_nk, bool) or not isinstance(max_nk, int) or not 8 <= max_nk <= (1 << 31) - 8:
        raise ValueError("max_nk must be an integer >= 8 within signed CUDA indexing")
    return max_nk


def _resize_tke_column_workspace(source: str, *, max_nk: int, declaration_count: int) -> str:
    """Resize the audited turbt local arrays, including retained dead source."""

    max_nk = _validate_column_capacity(max_nk)
    old = "float hx[74],hy[74],cp[74],dp[74],sol[74]"
    if source.count(old) != declaration_count:
        raise RuntimeError("TKE column workspace declaration inventory changed")
    size = max_nk - 1
    return source.replace(old, f"float hx[{size}],hy[{size}],cp[{size}],dp[{size}],sol[{size}]")


def build_cuda_tke_evolution_source(*, max_nk: int = CUDA_TKE_EVOLUTION_MAX_NK) -> str:
    """Return identical arithmetic with an explicit local-column capacity."""

    if not _source_pin_is_valid():
        raise RuntimeError("base TKE evolution source differs from its pin")
    return _resize_tke_column_workspace(CUDA_TKE_EVOLUTION_SOURCE, max_nk=max_nk, declaration_count=2)


@dataclass(frozen=True, slots=True)
class CompiledTkeEvolutionKernels:
    module_owner: Any; kernels: tuple[Any, ...]; ptx: bytes; cubin: bytes; target_arch: str; runtime_identity: Mapping[str, object]
    max_nk: int = CUDA_TKE_EVOLUTION_MAX_NK
    source_sha256: str = CUDA_TKE_EVOLUTION_SOURCE_SHA256


def compile_cuda_tke_evolution(*, max_nk: int = CUDA_TKE_EVOLUTION_MAX_NK) -> CompiledTkeEvolutionKernels:
    source = build_cuda_tke_evolution_source(max_nk=max_nk)
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    # The target is nvrtc_module.nvrtc_target_arch's: one architecture rule for
    # the package, clamped to what this NVRTC accepts, not getDeviceProperties'.
    cupy=importlib.import_module("cupy");compiler=importlib.import_module("cupy.cuda.compiler");device=cupy.cuda.Device();props=cupy.cuda.runtime.getDeviceProperties(device.id);major=int(props["major"]);minor=int(props["minor"]);arch=importlib.import_module("cm1core.nvrtc_module").nvrtc_target_arch(cupy);common=("--std=c++14","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true");nv=tuple(compiler._get_nvrtc_version());
    if int(nv[0])>=12:common+=("--device-as-default-execution-space",)
    ptx,_=compiler._NVRTCProgram(source,"cm1_tke_evolution_v1_ptx.cu",method="ptx").compile(common+(f"--gpu-architecture=compute_{arch}",));cubin,_=compiler._NVRTCProgram(source,"cm1_tke_evolution_v1_cubin.cu",method="cubin").compile(common+(f"--gpu-architecture=sm_{arch}",));ptx=ptx.encode() if isinstance(ptx,str) else bytes(ptx);cubin=cubin.encode("latin-1") if isinstance(cubin,str) else bytes(cubin);module=importlib.import_module("cupy.cuda.function").Module();module.load(cubin);kernels=tuple(module.get_function(n) for n in CUDA_TKE_EVOLUTION_KERNEL_NAMES);name=props["name"].decode() if isinstance(props["name"],bytes) else str(props["name"]);return CompiledTkeEvolutionKernels(module,kernels,ptx,cubin,arch,{"device_id":device.id,"device_name":name,"compute_capability":f"{major}.{minor}","runtime_version":int(cupy.cuda.runtime.runtimeGetVersion()),"driver_version":int(cupy.cuda.runtime.driverGetVersion())},max_nk,source_sha256)


def _require_compiled_capacity(compiled: CompiledTkeEvolutionKernels | None, nk: int) -> None:
    capacity = CUDA_TKE_EVOLUTION_MAX_NK if compiled is None else compiled.max_nk
    _validate_column_capacity(capacity)
    if nk > capacity:
        raise ValueError(f"nk={nk} exceeds compiled TKE column workspace {capacity}; compile with max_nk >= nk")
    if compiled is not None and compiled.source_sha256 != hashlib.sha256(
        build_cuda_tke_evolution_source(max_nk=capacity).encode()
    ).hexdigest():
        raise ValueError("compiled TKE source hash differs from its column capacity")


def _capacity_receipt(compiled: CompiledTkeEvolutionKernels) -> dict[str, object]:
    return {
        "column_workspace_max_nk": compiled.max_nk,
        "base_source_sha256": CUDA_TKE_EVOLUTION_SOURCE_SHA256,
    } if compiled.max_nk != CUDA_TKE_EVOLUTION_MAX_NK else {}


@dataclass(frozen=True, slots=True)
class CudaTkeEvolutionLaunch:
    ready_event: Any; receipt: Mapping[str, object]; scratch: Any; compiled: CompiledTkeEvolutionKernels


@dataclass(frozen=True, slots=True)
class CudaTkeDiffusionLaunch:
    ready_event: Any; receipt: Mapping[str, object]; compiled: CompiledTkeEvolutionKernels


@dataclass(frozen=True, slots=True)
class CudaTkeStageLaunch:
    ready_event: Any; receipt: Mapping[str, object]; workspace: Any; compiled: CompiledTkeEvolutionKernels


@dataclass(frozen=True, slots=True)
class CudaTkeCommitLaunch:
    ready_event: Any; receipt: Mapping[str, object]; compiled: CompiledTkeEvolutionKernels


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes=tke_evolution_shapes(ni,nj,nk);out={}
    for name in _NAMES:
        if name not in arrays:raise ValueError(f"missing TKE evolution array: {name}")
        value=arrays[name]
        if not isinstance(value,cupy.ndarray) or value.dtype!=cupy.float32 or value.shape!=shapes[name] or not value.flags.c_contiguous:raise ValueError(f"{name} must be a C-order cupy.float32 array with shape {shapes[name]}")
        out[name]=value
    if len({int(v.data.ptr) for v in out.values()})!=len(out):raise ValueError("TKE evolution arrays must not alias")
    return out


def launch_cuda_tke_diffusion(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dx: float, dy: float, dz: float, dt: float, stream: Any|None=None, compiled: CompiledTkeEvolutionKernels|None=None, selection: TkeEvolutionSelection=DEFAULT_TKE_EVOLUTION_SELECTION) -> CudaTkeDiffusionLaunch:
    """Apply ``turbt`` once at solve1 cadence; do not call inside RK."""
    _require_compiled_capacity(compiled, nk)
    _validate_selection(selection);_validate_grid(ni,nj,nk);dx,dy,dz,dt=(_spacing(n,v) for n,v in (("dx",dx),("dy",dy),("dz",dz),("dt",dt)));cupy=importlib.import_module("cupy");a=_require_device_arrays(cupy,arrays,ni,nj,nk);compiled=compiled or compile_cuda_tke_evolution();stream=stream or cupy.cuda.get_current_stream();block=(CUDA_TKE_EVOLUTION_THREADS,);n=ni*nj;grid=((n+CUDA_TKE_EVOLUTION_THREADS-1)//CUDA_TKE_EVOLUTION_THREADS,)
    with stream:
        compiled.kernels[0](grid,block,(a["tke3d"],a["kmh"],a["kmv"],a["rho"],a["rf"],a["mh"],a["mf"],a["gz"],a["rgz"],a["rgzu"],a["rgzv"],a["tketen"],cupy.uint32(ni),cupy.uint32(nj),cupy.uint32(nk),cupy.float32(dx),cupy.float32(dy),cupy.float32(dz),cupy.float32(dt)),stream=stream);event=cupy.cuda.Event();event.record(stream)
    return CudaTkeDiffusionLaunch(event,{"schema":CUDA_TKE_EVOLUTION_SCHEMA,"phase":"solve1-turbt-once","source":"src/solve1.F:314-369;src/turbtend.F:700-1044","source_sha256":compiled.source_sha256,**_capacity_receipt(compiled)},compiled)


def launch_cuda_tke_rk_stage(arrays: Mapping[str, Any], *, workspace: Any, ni: int, nj: int, nk: int, nrk: int, dx: float, dy: float, dz: float, dt: float, stream: Any|None=None, compiled: CompiledTkeEvolutionKernels|None=None, selection: TkeEvolutionSelection=DEFAULT_TKE_EVOLUTION_SELECTION) -> CudaTkeStageLaunch:
    """Apply one TKE ``advw``/RK/bcw stage using that stage's live mass flux."""
    _validate_selection(selection);_validate_grid(ni,nj,nk)
    _require_compiled_capacity(compiled, nk)
    if isinstance(nrk,bool) or not isinstance(nrk,int) or nrk not in (1,2,3):raise ValueError("nrk must be an integer in 1..3")
    dx,dy,dz,dt=(_spacing(n,v) for n,v in (("dx",dx),("dy",dy),("dz",dz),("dt",dt)));cupy=importlib.import_module("cupy");a=_require_device_arrays(cupy,arrays,ni,nj,nk);shape=(nk+1,nj,ni)
    if not isinstance(workspace,cupy.ndarray) or workspace.dtype!=cupy.float32 or workspace.shape!=shape or not workspace.flags.c_contiguous:raise ValueError(f"workspace must be a C-order cupy.float32 array with shape {shape}")
    if int(workspace.data.ptr) in {int(v.data.ptr) for v in a.values()}:raise ValueError("TKE stage workspace must not alias persistent arrays")
    compiled=compiled or compile_cuda_tke_evolution();stream=stream or cupy.cuda.get_current_stream();words=(nk+1)*nj*ni;block=(CUDA_TKE_EVOLUTION_THREADS,);grid=((words+CUDA_TKE_EVOLUTION_THREADS-1)//CUDA_TKE_EVOLUTION_THREADS,);dttmp=cupy.float32(cupy.float32(dt)/cupy.float32(4-nrk));weps=float(cupy.float32(cupy.float32(100)*cupy.float32(1e-18)).item())
    with stream:
        compiled.kernels[1](grid,block,(a["tke3d"],a["tkea"],a["tketen"],a["c1"],a["c2"],a["rrf0"],a["gz"],a["rru"],a["rrv"],a["rrw"],a["divx"],workspace,cupy.uint32(ni),cupy.uint32(nj),cupy.uint32(nk),cupy.uint32(CUDA_TKE_EVOLUTION_NGXY),cupy.float32(dx),cupy.float32(dy),cupy.float32(dz),dttmp,cupy.uint32(1 if nrk==3 else 0),cupy.float64(weps)),stream=stream)
        compiled.kernels[3](grid,block,(workspace,a["tke3d"],cupy.uint64(words)),stream=stream);event=cupy.cuda.Event();event.record(stream)
    return CudaTkeStageLaunch(event,{"schema":CUDA_TKE_EVOLUTION_SCHEMA,"phase":"solve2-tke-advw-rk-stage","nrk":nrk,"dttmp_seconds":float(dttmp),"uses_weno":nrk==3,"mass_flux_contract":"rru/rrv/rrw/divx must be rebuilt for this nrk immediately before this call","tkea_contract":"fixed full-step baseline; not modified by this call","source_sha256":compiled.source_sha256,**_capacity_receipt(compiled)},workspace,compiled)


def launch_cuda_tke_commit(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, completed_nrk: int, stream: Any|None=None, compiled: CompiledTkeEvolutionKernels|None=None, selection: TkeEvolutionSelection=DEFAULT_TKE_EVOLUTION_SELECTION) -> CudaTkeCommitLaunch:
    """Commit final ``tke3d`` to ``tkea`` only after RK stage three."""
    _validate_selection(selection);_validate_grid(ni,nj,nk)
    _require_compiled_capacity(compiled, nk)
    if completed_nrk!=3:raise ValueError("TKE commit requires completed_nrk=3")
    cupy=importlib.import_module("cupy");a=_require_device_arrays(cupy,arrays,ni,nj,nk);compiled=compiled or compile_cuda_tke_evolution();stream=stream or cupy.cuda.get_current_stream();words=(nk+1)*nj*ni;block=(CUDA_TKE_EVOLUTION_THREADS,);grid=((words+CUDA_TKE_EVOLUTION_THREADS-1)//CUDA_TKE_EVOLUTION_THREADS,)
    with stream:
        compiled.kernels[2](grid,block,(a["tke3d"],a["tke3d"],a["tkea"],cupy.uint64(words)),stream=stream);event=cupy.cuda.Event();event.record(stream)
    return CudaTkeCommitLaunch(event,{"schema":CUDA_TKE_EVOLUTION_SCHEMA,"phase":"solve3-tkea-commit","completed_nrk":3,"source":"src/solve3.F:560-590","source_sha256":compiled.source_sha256,**_capacity_receipt(compiled)},compiled)


def launch_cuda_tke_evolution_frozen_flux(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dx: float, dy: float, dz: float, dt: float, stream: Any|None=None, compiled: CompiledTkeEvolutionKernels|None=None, selection: TkeEvolutionSelection=DEFAULT_TKE_EVOLUTION_SELECTION) -> CudaTkeEvolutionLaunch:
    """Run the three stages with one frozen flux snapshot for operator tests only.

    Forecast integration must call :func:`launch_cuda_tke_rk_stage` after each
    stage's post-acoustic ``getdivx`` rebuild.  This helper deliberately makes
    no coupled-trajectory claim.
    """
    _require_compiled_capacity(compiled, nk)
    _validate_selection(selection);_validate_grid(ni,nj,nk);dx,dy,dz,dt=(_spacing(n,v) for n,v in (("dx",dx),("dy",dy),("dz",dz),("dt",dt)));cupy=importlib.import_module("cupy");sh=tke_evolution_shapes(ni,nj,nk);a={}
    for name in _NAMES:
        if name not in arrays:raise ValueError(f"missing TKE evolution array: {name}")
        v=arrays[name]
        if not isinstance(v,cupy.ndarray) or v.dtype!=cupy.float32 or v.shape!=sh[name] or not v.flags.c_contiguous:raise ValueError(f"{name} must be a C-order cupy.float32 array with shape {sh[name]}")
        a[name]=v
    if len({int(v.data.ptr) for v in a.values()})!=len(a):raise ValueError("TKE evolution arrays must not alias")
    compiled=compiled or compile_cuda_tke_evolution();stream=stream or cupy.cuda.get_current_stream();u32=cupy.uint32;f32=cupy.float32;f64=cupy.float64;block=(CUDA_TKE_EVOLUTION_THREADS,);col=ni*nj;words=(nk+1)*col;gcol=((col+CUDA_TKE_EVOLUTION_THREADS-1)//CUDA_TKE_EVOLUTION_THREADS,);gword=((words+CUDA_TKE_EVOLUTION_THREADS-1)//CUDA_TKE_EVOLUTION_THREADS,);tmp=cupy.empty_like(a["tke3d"]);weps=float(cupy.float32(cupy.float32(100)*cupy.float32(1e-18)).item())
    with stream:
        compiled.kernels[0](gcol,block,(a["tke3d"],a["kmh"],a["kmv"],a["rho"],a["rf"],a["mh"],a["mf"],a["gz"],a["rgz"],a["rgzu"],a["rgzv"],a["tketen"],u32(ni),u32(nj),u32(nk),f32(dx),f32(dy),f32(dz),f32(dt)),stream=stream)
        cur=a["tke3d"]
        for nrk,out in ((1,tmp),(2,a["tke3d"]),(3,tmp)):
            dttmp=f32(f32(dt)/f32(4-nrk));compiled.kernels[1](gword,block,(cur,a["tkea"],a["tketen"],a["c1"],a["c2"],a["rrf0"],a["gz"],a["rru"],a["rrv"],a["rrw"],a["divx"],out,u32(ni),u32(nj),u32(nk),u32(CUDA_TKE_EVOLUTION_NGXY),f32(dx),f32(dy),f32(dz),dttmp,u32(1 if nrk==3 else 0),f64(weps)),stream=stream);cur=out
        compiled.kernels[2](gword,block,(tmp,a["tke3d"],a["tkea"],cupy.uint64(words)),stream=stream);event=cupy.cuda.Event();event.record(stream)
    return CudaTkeEvolutionLaunch(event,{"schema":CUDA_TKE_EVOLUTION_SCHEMA,"abi":CUDA_TKE_EVOLUTION_ABI,"claim_limit":CUDA_TKE_EVOLUTION_CLAIM_LIMIT,"authority":AUTHORITY_SOURCE_SHA256,"source_sha256":compiled.source_sha256,**_capacity_receipt(compiled),"domain":{"ni":ni,"nj":nj,"nk":nk},"spacing_m":{"dx":dx,"dy":dy,"dz":dz},"dt_seconds":dt,"mass_flux_cadence":"one frozen supplied snapshot; analytic operator-validation only; not a forecast API","rk":{"nrkmax":3,"dttmp_seconds":[float(f32(f32(dt)/f32(x))) for x in (3,2,1)],"weno_stages":[3]},"kernel":{"names":list(CUDA_TKE_EVOLUTION_KERNEL_NAMES),"target_arch":compiled.target_arch}},tmp,compiled)


def launch_cuda_tke_evolution(*args: Any, **kwargs: Any) -> CudaTkeEvolutionLaunch:
    """Refuse the cadence-ambiguous legacy all-stage entry point."""
    del args, kwargs
    refuse(
        code="CM1-TKE-FLUX-CADENCE",
        gate="physics-seam",
        option="launch_cuda_tke_evolution",
        value="all-three-stages",
        reason="CM1 rebuilds rru/rrv/rrw/divx after each acoustic stage; one snapshot cannot represent an evolved RK trajectory",
        remedy="call launch_cuda_tke_diffusion once, launch_cuda_tke_rk_stage after each nrk getdivx, then launch_cuda_tke_commit",
    )


def _source_pin_is_valid() -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}",CUDA_TKE_EVOLUTION_SOURCE_SHA256)) and hashlib.sha256(CUDA_TKE_EVOLUTION_SOURCE.encode()).hexdigest()==CUDA_TKE_EVOLUTION_SOURCE_SHA256
