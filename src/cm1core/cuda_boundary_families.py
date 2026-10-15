"""CM1 r21.1 lateral boundary families and the open-boundary companions.

This module owns the four things ``src/bc.F`` does at a lateral boundary that
this port's sealed all-open operator does not express:

* the *family* fills of ``bcs``/``bcp``/``bcu``/``bcv``/``bcw``
  (``src/bc.F:180-1143``) for every CM1 side value -- ``1`` periodic, ``2``
  open-radiative, ``3`` rigid free-slip, ``4`` rigid no-slip -- over this
  port's ``ngxy`` padded layout, with each side selected independently so a
  mixed family (open in x, periodic in y, which is CM1's own squall-line
  deck) is one call;
* ``roflux=1``, ``restrict_openbc_we``/``restrict_openbc_sn``
  (``src/bc.F:2180-2391``), which rescales the outward normal mass flux so it
  cannot exceed the inward flux;
* ``nudgeobc=1``/``alphobc``, the inflow nudging of ``src/solve2.F:726-822``;
* ``irbc=1``/``irbc=2``, the Klemp-Wilhelmson phase-speed histories of
  ``radbcew``/``radbcns`` (``src/bc.F:1948-2053``) with the ``cstar``
  prescribed speed of ``src/param.F:722``.

It also carries the *boundary-face* half of ``src/sound.F:382-402``.  CM1
integrates the horizontal pressure gradient over ``i=1+ibw .. ni+1-ibe`` and
``j=1+ibs .. nj+1-ibn``, and ``src/param.F:6678-6681`` sets ``ibw=0``
precisely when ``wbc=1``.  A periodic side therefore has its *boundary* face
updated by the pressure gradient, where an open or rigid side does not: the
open side is written by ``ssopenbc*`` and the rigid side is held at zero by
``bcu``/``bcv``.  The sealed operator in :mod:`cm1core.cuda_psolver3_terrain_open`
covers the interior faces only, because it was measured for the all-open
family where ``ibw=ibe=ibs=ibn=1``; :func:`enqueue_periodic_face_pressure_gradient`
adds exactly the faces a periodic side turns on, and nothing else.

Every routine here has a NumPy oracle beside the CUDA kernel so the
arithmetic is checkable without a device.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .errors import refuse


CUDA_BOUNDARY_FAMILIES_SCHEMA = "gpuwm-cm1.cuda-boundary-families/v1"
CUDA_BOUNDARY_FAMILIES_ABI = "gpuwm-cm1.cuda-boundary-families-kernels/v1"
CUDA_BOUNDARY_FAMILIES_THREADS = 128

#: CM1's own lateral selector values (``README.namelist:561-568``).
PERIODIC = 1
OPEN_RADIATIVE = 2
RIGID_FREE_SLIP = 3
RIGID_NO_SLIP = 4
LATERAL_VALUES = (PERIODIC, OPEN_RADIATIVE, RIGID_FREE_SLIP, RIGID_NO_SLIP)

#: ``kind`` selects which of CM1's five boundary subroutines is being applied.
KIND_SCALAR = 0  # bcs  -- src/bc.F:180-364
KIND_U = 1  # bcu  -- src/bc.F:528-714
KIND_V = 2  # bcv  -- src/bc.F:722-934
KIND_W = 3  # bcw  -- src/bc.F:942-1143

#: ``src/param.F:722`` -- a declared cstar below 0.1 m/s is CM1's "unset"
#: sentinel and becomes 30 m/s.
CM1_CSTAR_FLOOR = 0.1
CM1_CSTAR_DEFAULT = 30.0
#: ``src/cm1.F:346``.
CM1_ALPHOBC_DEFAULT = 60.0

SOURCE_ANCHORS = {
    "family_scalar": "src/bc.F:180-364",
    "family_pressure": "src/bc.F:372-520",
    "family_u": "src/bc.F:528-714",
    "family_v": "src/bc.F:722-934",
    "family_w": "src/bc.F:942-1143",
    "radbc_large_step": "src/bc.F:1948-2053",
    "radbc_irbc1_call": "src/solve1.F:402-409",
    "radbc_irbc2_call": "src/sound.F:260-268",
    "restrict_openbc": "src/bc.F:2180-2391",
    "restrict_openbc_call": "src/sound.F:283-302",
    "nudgeobc": "src/solve2.F:726-822",
    "boundary_index_flags": "src/param.F:6678-6681",
    "face_pressure_gradient": "src/sound.F:382-402",
    "cstar_default": "src/param.F:722",
    "alphobc_default": "src/cm1.F:346",
}


@dataclass(frozen=True, slots=True)
class LateralBoundarySelection:
    """CM1's lateral-boundary namelist, by CM1's own names.

    ``west``/``east``/``south``/``north`` are ``wbc``/``ebc``/``sbc``/``nbc``.
    ``irbc`` only has meaning on a side whose value is ``2``.
    """

    west: int = OPEN_RADIATIVE
    east: int = OPEN_RADIATIVE
    south: int = OPEN_RADIATIVE
    north: int = OPEN_RADIATIVE
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    alphobc: float = CM1_ALPHOBC_DEFAULT
    cstar: float = CM1_CSTAR_DEFAULT

    def __post_init__(self) -> None:
        for name in ("west", "east", "south", "north"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in LATERAL_VALUES:
                raise ValueError(f"{name} must be one of CM1's lateral values {LATERAL_VALUES}")
        if isinstance(self.irbc, bool) or self.irbc not in (1, 2, 4):
            raise ValueError("irbc must be CM1's 1, 2 or 4")
        for name in ("roflux", "nudgeobc"):
            value = getattr(self, name)
            if isinstance(value, bool) or value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        for name in ("alphobc", "cstar"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "alphobc", float(self.alphobc))
        # src/param.F:722.  CM1 rewrites an unset cstar rather than refusing,
        # and a port that kept 0.0 would make radbcew a zero-phase-speed
        # scheme -- an outflow boundary that never radiates.
        object.__setattr__(
            self,
            "cstar",
            CM1_CSTAR_DEFAULT if float(self.cstar) < CM1_CSTAR_FLOOR else float(self.cstar),
        )
        if self.alphobc <= 0.0 and self.nudgeobc == 1:
            raise ValueError("nudgeobc=1 needs a positive alphobc: solve2.F:728 forms 1/alphobc")

    @property
    def sides(self) -> tuple[int, int, int, int]:
        return (self.west, self.east, self.south, self.north)

    @property
    def all_open(self) -> bool:
        """The sealed family this port's evidenced operator was measured for."""

        return self.sides == (OPEN_RADIATIVE,) * 4

    @property
    def any_open(self) -> bool:
        return OPEN_RADIATIVE in self.sides

    def no_effect(self) -> dict[str, str]:
        """The open-side settings this selection turns on with no open side.

        CM1 accepts them and they change nothing: ``restrict_openbc`` rescales
        the outward flow through a side whose value is 2 only
        (``src/bc.F:2180-2391``), and the phase speeds ``irbc = 1`` and ``2``
        build feed ``ssopenbc``, which runs on such a side only
        (``src/sound.F:274-302``).  The run is CM1's either way, and the
        record says so rather than name an effect the run never had.
        """

        if self.any_open:
            return {}
        notes: dict[str, str] = {}
        if self.roflux == 1:
            notes["roflux"] = (
                "no effect: no side is open, and CM1 rescales the outward flow through open sides only "
                "(src/bc.F:2180-2391)"
            )
        if self.irbc in (1, 2):
            notes["irbc"] = (
                f"no effect: no side is open, and the phase speeds irbc = {self.irbc} builds are used by the "
                "open-side update only (src/sound.F:274-302)"
            )
        return notes

    # src/param.F:6678-6681.  ib* is 1 on a side this rank owns whose value is
    # not periodic.  Serial GM1 owns every side, so ib* is purely wbc/=1.
    @property
    def ibw(self) -> int:
        return 0 if self.west == PERIODIC else 1

    @property
    def ibe(self) -> int:
        return 0 if self.east == PERIODIC else 1

    @property
    def ibs(self) -> int:
        return 0 if self.south == PERIODIC else 1

    @property
    def ibn(self) -> int:
        return 0 if self.north == PERIODIC else 1

    def record(self) -> dict[str, Any]:
        return {
            "wbc": self.west,
            "ebc": self.east,
            "sbc": self.south,
            "nbc": self.north,
            "irbc": self.irbc,
            "roflux": self.roflux,
            "nudgeobc": self.nudgeobc,
            "alphobc": self.alphobc,
            "cstar": self.cstar,
            "ibw": self.ibw,
            "ibe": self.ibe,
            "ibs": self.ibs,
            "ibn": self.ibn,
            "source_anchors": dict(SOURCE_ANCHORS),
            # Only when a setting is on with no open side, so every other
            # selection records exactly what it recorded before.
            **({"no_effect": no_effect} if (no_effect := self.no_effect()) else {}),
        }


DEFAULT_LATERAL_BOUNDARY_SELECTION = LateralBoundarySelection()


CUDA_BOUNDARY_FAMILIES_KERNEL_NAMES = (
    "cm1_bc_family_we_f32_v1",
    "cm1_bc_family_sn_f32_v1",
    "cm1_bc_pressure_we_f32_v1",
    "cm1_bc_pressure_sn_f32_v1",
    "cm1_bc_radbcew_f32_v1",
    "cm1_bc_radbcns_f32_v1",
    "cm1_bc_roflux_reduce_we_f64_v1",
    "cm1_bc_roflux_reduce_sn_f64_v1",
    "cm1_bc_roflux_ratio_f64_v1",
    "cm1_bc_roflux_apply_we_f32_v1",
    "cm1_bc_roflux_apply_sn_f32_v1",
    "cm1_bc_nudgeobc_we_f32_v1",
    "cm1_bc_nudgeobc_sn_f32_v1",
    "cm1_bc_face_pgrad_f32_v1",
    "cm1_bc_face_save_f32_v1",
    "cm1_bc_face_restore_f32_v1",
    "cm1_bc_hrdamp_tauf_f32_v1",
    "cm1_bc_irdamp2_operands_f32_v1",
)


CUDA_BOUNDARY_FAMILIES_SOURCE = r"""// gpuwm-cm1.cuda-boundary-families-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default REAL requires float32");
static_assert(sizeof(double) == 8, "CM1 double precision requires float64");

__device__ __forceinline__ float a32(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(const float a,const float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(const float a,const float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(const float a,const float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(
    const unsigned int k,const unsigned int j,const unsigned int i,
    const unsigned int ny,const unsigned int nx)
{ return ((unsigned long long)k*ny+j)*nx+i; }
__device__ __forceinline__ unsigned long long pidx(
    const unsigned int k,const unsigned int j,const unsigned int i,
    const unsigned int ni,const unsigned int nj)
{ return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i; }

// ---------------------------------------------------------------------------
// src/bc.F:180-1143.  One side value per side; layer runs over the ngxy ghost
// planes exactly as CM1's inner `do i=1,ngxy` does.
// ---------------------------------------------------------------------------
extern "C" __global__
void cm1_bc_family_we_f32_v1(float* a,unsigned int ni,unsigned int nj,unsigned int nk,
                             unsigned int ng,unsigned int kind,
                             unsigned int wbc,unsigned int ebc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nx=(kind==1u)?ni+1u+2u*ng:ni+2u*ng;
    const unsigned int ny=(kind==2u)?nj+1u+2u*ng:nj+2u*ng;
    const unsigned int kfirst=(kind==3u)?2u:1u,kcount=(kind==3u)?nk-1u:nk;
    const int jhi=(kind==2u)?(int)nj+2:(int)nj+1;
    const unsigned long long count=(unsigned long long)kcount*ny*ng*2u;
    for(unsigned long long q=first;q<count;q+=step){
        unsigned long long r=q;
        const unsigned int layer=(unsigned int)(r%ng); r/=ng;
        const unsigned int side=(unsigned int)(r%2u); r/=2u;
        const unsigned int jp=(unsigned int)(r%ny),k=(unsigned int)(r/ny)+kfirst;
        const unsigned int mode=(side==0u)?wbc:ebc;
        if(mode==0u) continue;
        const int jc=(int)jp-((int)ng-1);
        if(mode!=1u&&(jc<0||jc>jhi)) continue;
        unsigned int dst,src,wall=0u; float sign=1.0f; unsigned int zero=0u;
        if(side==0u){
            if(mode==1u){ dst=ng-1u-layer; src=ni+ng-1u-layer; }
            else if(mode==2u){ dst=layer; src=ng; }
            else if(kind==1u){ dst=ng-1u-layer; src=ng+1u+layer; sign=-1.0f; zero=1u; wall=ng; }
            else { dst=ng-1u-layer; src=ng+layer; }
        }else{
            if(mode==1u){
                if(kind==1u){ dst=ni+ng+1u+layer; src=ng+1u+layer; }
                else { dst=ni+ng+layer; src=ng+layer; }
            }else if(mode==2u){
                if(kind==1u){ dst=ni+ng+1u+layer; src=ni+ng; }
                else { dst=ni+ng+layer; src=ni+ng-1u; }
            }else if(kind==1u){ dst=ni+ng+1u+layer; src=ni+ng-1u-layer; sign=-1.0f; zero=1u; wall=ni+ng; }
            else { dst=ni+ng+layer; src=ni+ng-1u-layer; }
        }
        const float value=a[i3(k,jp,src,ny,nx)];
        a[i3(k,jp,dst,ny,nx)]=(sign<0.0f)?-value:value;
        if(zero&&layer==0u) a[i3(k,jp,wall,ny,nx)]=0.0f;
    }
}

extern "C" __global__
void cm1_bc_family_sn_f32_v1(float* a,unsigned int ni,unsigned int nj,unsigned int nk,
                             unsigned int ng,unsigned int kind,
                             unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nx=(kind==1u)?ni+1u+2u*ng:ni+2u*ng;
    const unsigned int ny=(kind==2u)?nj+1u+2u*ng:nj+2u*ng;
    const unsigned int kfirst=(kind==3u)?2u:1u,kcount=(kind==3u)?nk-1u:nk;
    const int ihi=(kind==1u)?(int)ni+2:(int)ni+1;
    const unsigned long long count=(unsigned long long)kcount*nx*ng*2u;
    for(unsigned long long q=first;q<count;q+=step){
        unsigned long long r=q;
        const unsigned int layer=(unsigned int)(r%ng); r/=ng;
        const unsigned int side=(unsigned int)(r%2u); r/=2u;
        const unsigned int ip=(unsigned int)(r%nx),k=(unsigned int)(r/nx)+kfirst;
        const unsigned int mode=(side==0u)?sbc:nbc;
        if(mode==0u) continue;
        const int ic=(int)ip-((int)ng-1);
        if(mode!=1u&&(ic<0||ic>ihi)) continue;
        unsigned int dst,src,wall=0u; float sign=1.0f; unsigned int zero=0u;
        if(side==0u){
            if(mode==1u){ dst=ng-1u-layer; src=nj+ng-1u-layer; }
            else if(mode==2u){ dst=layer; src=ng; }
            else if(kind==2u){ dst=ng-1u-layer; src=ng+1u+layer; sign=-1.0f; zero=1u; wall=ng; }
            else { dst=ng-1u-layer; src=ng+layer; }
        }else{
            if(mode==1u){
                if(kind==2u){ dst=nj+ng+1u+layer; src=ng+1u+layer; }
                else { dst=nj+ng+layer; src=ng+layer; }
            }else if(mode==2u){
                if(kind==2u){ dst=nj+ng+1u+layer; src=nj+ng; }
                else { dst=nj+ng+layer; src=nj+ng-1u; }
            }else if(kind==2u){ dst=nj+ng+1u+layer; src=nj+ng-1u-layer; sign=-1.0f; zero=1u; wall=nj+ng; }
            else { dst=nj+ng+layer; src=nj+ng-1u-layer; }
        }
        const float value=a[i3(k,src,ip,ny,nx)];
        a[i3(k,dst,ip,ny,nx)]=(sign<0.0f)?-value:value;
        if(zero&&layer==0u) a[i3(k,wall,ip,ny,nx)]=0.0f;
    }
}

// src/bc.F:372-520.  Only the periodic branch differs from a copy of the
// first interior column; rigid and open share one branch in CM1 too.
extern "C" __global__
void cm1_bc_pressure_we_f32_v1(float* p,unsigned int ni,unsigned int nj,unsigned int nk,
                               unsigned int wbc,unsigned int ebc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long count=(unsigned long long)nk*(nj+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int j=(unsigned int)(q%(nj+2u)),k=(unsigned int)(q/(nj+2u))+1u;
        if(wbc!=0u) p[pidx(k,j,0u,ni,nj)]=p[pidx(k,j,(wbc==1u)?ni:1u,ni,nj)];
        if(ebc!=0u) p[pidx(k,j,ni+1u,ni,nj)]=p[pidx(k,j,(ebc==1u)?1u:ni,ni,nj)];
    }
}

extern "C" __global__
void cm1_bc_pressure_sn_f32_v1(float* p,unsigned int ni,unsigned int nj,unsigned int nk,
                               unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long count=(unsigned long long)nk*(ni+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+2u)),k=(unsigned int)(q/(ni+2u))+1u;
        if(sbc!=0u) p[pidx(k,0u,i,ni,nj)]=p[pidx(k,(sbc==1u)?nj:1u,i,ni,nj)];
        if(nbc!=0u) p[pidx(k,nj+1u,i,ni,nj)]=p[pidx(k,(nbc==1u)?1u:nj,i,ni,nj)];
    }
}

// src/bc.F:1948-2053.  radbcw is <=0 and radbce is >=0 by construction: an
// inflow point radiates nothing.
extern "C" __global__
void cm1_bc_radbcew_f32_v1(const float* ua,float* radbcw,float* radbce,
                           unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                           unsigned int wbc,unsigned int ebc,float cstar)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,rw=nj+2u*ng;
    const unsigned long long count=(unsigned long long)nk*nj;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;
        const unsigned int y=j+ng-1u;
        if(wbc==2u){
            const float c=s32(ua[i3(k,y,ng,nyu,nxu)],cstar);
            radbcw[(unsigned long long)k*rw+y]=(c<0.0f)?c:0.0f;
        }
        if(ebc==2u){
            const float c=a32(ua[i3(k,y,ni+ng,nyu,nxu)],cstar);
            radbce[(unsigned long long)k*rw+y]=(c>0.0f)?c:0.0f;
        }
    }
}

extern "C" __global__
void cm1_bc_radbcns_f32_v1(const float* va,float* radbcs,float* radbcn,
                           unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                           unsigned int sbc,unsigned int nbc,float cstar)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng,rw=ni+2u*ng;
    const unsigned long long count=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%ni)+1u,k=(unsigned int)(q/ni)+1u;
        const unsigned int x=i+ng-1u;
        if(sbc==2u){
            const float c=s32(va[i3(k,ng,x,nyv,nxv)],cstar);
            radbcs[(unsigned long long)k*rw+x]=(c<0.0f)?c:0.0f;
        }
        if(nbc==2u){
            const float c=a32(va[i3(k,nj+ng,x,nyv,nxv)],cstar);
            radbcn[(unsigned long long)k*rw+x]=(c>0.0f)?c:0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// src/bc.F:2180-2391.  The per-level partial sums are accumulated in the same
// j order CM1's serial loop uses, so one thread owns one k and the double
// accumulation is reproducible; the k sum and the ratio are then formed by a
// single thread, as CM1's serial `do k=1,nk` does.
// ---------------------------------------------------------------------------
extern "C" __global__
void cm1_bc_roflux_reduce_we_f64_v1(const float* rho0,const float* u3d,const float* rvh,
                                    const float* rmh,double* temout,double* temin,
                                    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                                    unsigned int wbc,unsigned int ebc)
{
    const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=nk) return;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
    double out=0.0,in=0.0;
    if(wbc==2u){
        for(unsigned int j=1u;j<=nj;++j){
            const unsigned long long m=((unsigned long long)k*nj+(j-1u))*ni;
            const float f=m32(m32(m32(rho0[m],u3d[i3(k+1u,j+ng-1u,ng,nyu,nxu)]),rvh[j+ng-1u]),rmh[m]);
            out-=(double)fminf(0.0f,f);
            in +=(double)fmaxf(0.0f,f);
        }
    }
    if(ebc==2u){
        for(unsigned int j=1u;j<=nj;++j){
            const unsigned long long m=((unsigned long long)k*nj+(j-1u))*ni+(ni-1u);
            const float f=m32(m32(m32(rho0[m],u3d[i3(k+1u,j+ng-1u,ni+ng,nyu,nxu)]),rvh[j+ng-1u]),rmh[m]);
            out+=(double)fmaxf(0.0f,f);
            in -=(double)fminf(0.0f,f);
        }
    }
    temout[k]=out; temin[k]=in;
}

extern "C" __global__
void cm1_bc_roflux_reduce_sn_f64_v1(const float* rho0,const float* v3d,const float* ruh,
                                    const float* rmh,double* temout,double* temin,
                                    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                                    unsigned int sbc,unsigned int nbc)
{
    const unsigned int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=nk) return;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    double out=0.0,in=0.0;
    if(sbc==2u){
        for(unsigned int i=1u;i<=ni;++i){
            const unsigned long long m=(unsigned long long)k*nj*ni+(i-1u);
            const float f=m32(m32(m32(rho0[m],v3d[i3(k+1u,ng,i+ng-1u,nyv,nxv)]),ruh[i+ng-1u]),rmh[m]);
            out-=(double)fminf(0.0f,f);
            in +=(double)fmaxf(0.0f,f);
        }
    }
    if(nbc==2u){
        for(unsigned int i=1u;i<=ni;++i){
            const unsigned long long m=((unsigned long long)k*nj+(nj-1u))*ni+(i-1u);
            const float f=m32(m32(m32(rho0[m],v3d[i3(k+1u,nj+ng,i+ng-1u,nyv,nxv)]),ruh[i+ng-1u]),rmh[m]);
            out+=(double)fmaxf(0.0f,f);
            in -=(double)fminf(0.0f,f);
        }
    }
    temout[k]=out; temin[k]=in;
}

extern "C" __global__
void cm1_bc_roflux_ratio_f64_v1(const double* temout,const double* temin,double* t3,unsigned int nk)
{
    if(blockIdx.x!=0u||threadIdx.x!=0u) return;
    double fluxout=0.0,fluxin=0.0;
    for(unsigned int k=0u;k<nk;++k){ fluxout+=temout[k]; fluxin+=temin[k]; }
    t3[0]=(fluxin+1.0e-20)/(fluxout+1.0e-20);
}

extern "C" __global__
void cm1_bc_roflux_apply_we_f32_v1(const float* rho0,float* u3d,const double* t3,
                                   unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                                   unsigned int wbc,unsigned int ebc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
    const double ratio=t3[0];
    const unsigned long long count=(unsigned long long)nk*nj;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;
        const unsigned int y=j+ng-1u;
        if(wbc==2u){
            const unsigned long long m=((unsigned long long)(k-1u)*nj+(j-1u))*ni;
            const unsigned long long x=i3(k,y,ng,nyu,nxu);
            const double u1=(double)rho0[m]*(double)u3d[x];
            if(u1<0.0) u3d[x]=(float)(u1*ratio/(double)rho0[m]);
        }
        if(ebc==2u){
            const unsigned long long m=((unsigned long long)(k-1u)*nj+(j-1u))*ni+(ni-1u);
            const unsigned long long x=i3(k,y,ni+ng,nyu,nxu);
            const double u1=(double)rho0[m]*(double)u3d[x];
            if(u1>0.0) u3d[x]=(float)(u1*ratio/(double)rho0[m]);
        }
    }
}

extern "C" __global__
void cm1_bc_roflux_apply_sn_f32_v1(const float* rho0,float* v3d,const double* t3,
                                   unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                                   unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const double ratio=t3[0];
    const unsigned long long count=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%ni)+1u,k=(unsigned int)(q/ni)+1u;
        const unsigned int x=i+ng-1u;
        if(sbc==2u){
            const unsigned long long m=(unsigned long long)(k-1u)*nj*ni+(i-1u);
            const unsigned long long y=i3(k,ng,x,nyv,nxv);
            const double v1=(double)rho0[m]*(double)v3d[y];
            if(v1<0.0) v3d[y]=(float)(v1*ratio/(double)rho0[m]);
        }
        if(nbc==2u){
            const unsigned long long m=((unsigned long long)(k-1u)*nj+(nj-1u))*ni+(i-1u);
            const unsigned long long y=i3(k,nj+ng,x,nyv,nxv);
            const double v1=(double)rho0[m]*(double)v3d[y];
            if(v1>0.0) v3d[y]=(float)(v1*ratio/(double)rho0[m]);
        }
    }
}

// src/solve2.F:726-822.  u0/v0 are this lane's horizontally uniform base
// profiles, indexed by level exactly as CM1 indexes u0(1,j,k).
extern "C" __global__
void cm1_bc_nudgeobc_we_f32_v1(const float* u3d,const float* u0,float* uten,
                               unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                               unsigned int wbc,unsigned int ebc,float rtem)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
    const unsigned long long count=(unsigned long long)nk*nj;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;
        const unsigned int y=j+ng-1u;
        if(wbc==2u){
            const unsigned long long x=i3(k,y,ng,nyu,nxu);
            if(u3d[x]>0.0f) uten[x]=s32(uten[x],m32(s32(u3d[x],u0[k-1u]),rtem));
        }
        if(ebc==2u){
            const unsigned long long x=i3(k,y,ni+ng,nyu,nxu);
            if(u3d[x]<0.0f) uten[x]=s32(uten[x],m32(s32(u3d[x],u0[k-1u]),rtem));
        }
    }
}

extern "C" __global__
void cm1_bc_nudgeobc_sn_f32_v1(const float* v3d,const float* v0,float* vten,
                               unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                               unsigned int sbc,unsigned int nbc,float rtem)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const unsigned long long count=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%ni)+1u,k=(unsigned int)(q/ni)+1u;
        const unsigned int x=i+ng-1u;
        if(sbc==2u){
            const unsigned long long y=i3(k,ng,x,nyv,nxv);
            if(v3d[y]>0.0f) vten[y]=s32(vten[y],m32(s32(v3d[y],v0[k-1u]),rtem));
        }
        if(nbc==2u){
            const unsigned long long y=i3(k,nj+ng,x,nyv,nxv);
            if(v3d[y]<0.0f) vten[y]=s32(vten[y],m32(s32(v3d[y],v0[k-1u]),rtem));
        }
    }
}

// src/param.F:8736-8776.  Since cm1r19.6 the lateral Rayleigh sponge acts on w
// only, so it enters as a four-side maximum over the vertical tauf field and
// nothing else changes.  minx cancels: CM1's x1 is (xhd+minx)-xh(i) and
// xh(i)-minx is dx*(i-0.5) for either iorigin.  pi is src/constants.F:14, a
// default REAL, so the whole taper is single precision as CM1 builds it.
extern "C" __global__
void cm1_bc_hrdamp_tauf_f32_v1(float* tauf,unsigned int ni,unsigned int nj,unsigned int nk,
                               unsigned int i0,unsigned int j0,unsigned int nx,unsigned int ny,
                               float dx,float dy,float xhd)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const float pi=3.1415926535897932384626433f;
    const unsigned long long count=(unsigned long long)(nk+1u)*(nj+2u)*(ni+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+2u));
        const unsigned long long r=q/(ni+2u);
        const unsigned int j=(unsigned int)(r%(nj+2u));
        float value=tauf[q];
        const float xh=m32(dx,s32((float)(i0+i),0.5f));
        const float yh=m32(dy,s32((float)(j0+j),0.5f));
        if(nx>1u){
            const float x1=s32(xhd,xh);
            if(x1>0.0f) value=fmaxf(value,m32(0.5f,s32(1.0f,cosf(d32(m32(pi,x1),xhd)))));
            const float x2=s32(xh,s32(m32((float)nx,dx),xhd));
            if(x2>0.0f) value=fmaxf(value,m32(0.5f,s32(1.0f,cosf(d32(m32(pi,x2),xhd)))));
        }
        if(ny>1u){
            const float y1=s32(xhd,yh);
            if(y1>0.0f) value=fmaxf(value,m32(0.5f,s32(1.0f,cosf(d32(m32(pi,y1),xhd)))));
            const float y2=s32(yh,s32(m32((float)ny,dy),xhd));
            if(y2>0.0f) value=fmaxf(value,m32(0.5f,s32(1.0f,cosf(d32(m32(pi,y2),xhd)))));
        }
        tauf[q]=value;
    }
}

// src/solve1.F:427-438.  irdamp=2 is the irdamp=1 expression with the base
// state replaced by the level's horizontal average, so the same held-assignment
// kernel produces it once these three operands are built: u0 and v0 become the
// average broadcast, and the theta departure becomes (th0-thavg)+tha, which is
// one sum inside the one product CM1 forms.
extern "C" __global__
void cm1_bc_irdamp2_operands_f32_v1(const float* uavg,const float* vavg,const float* thavg,
                                    const float* th0,const float* tha,
                                    float* u0e,float* v0e,float* thae,
                                    unsigned int ni,unsigned int nj,unsigned int nk)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long nu=(unsigned long long)nk*nj*(ni+1u);
    const unsigned long long nv=(unsigned long long)nk*(nj+1u)*ni;
    const unsigned long long ns=(unsigned long long)nk*nj*ni;
    for(unsigned long long z=first;z<nu;z+=step){
        const unsigned int k=(unsigned int)(z/((unsigned long long)nj*(ni+1u)));
        u0e[z]=uavg[k];
    }
    for(unsigned long long z=first;z<nv;z+=step){
        const unsigned int k=(unsigned int)(z/((unsigned long long)(nj+1u)*ni));
        v0e[z]=vavg[k];
    }
    for(unsigned long long z=first;z<ns;z+=step){
        const unsigned int k=(unsigned int)(z/((unsigned long long)nj*ni));
        thae[z]=a32(s32(th0[z],thavg[k]),tha[z]);
    }
}

// The sealed ssopenbc operator in cuda_terrain_open_boundary updates all four
// normal-velocity faces unconditionally, because it was measured for the
// all-open family.  src/sound.F:274-302 calls ssopenbcw only when wbc=2, and a
// periodic or rigid side's boundary face must not receive that tendency: a
// rigid face is held at zero by bcu/bcv and a periodic face is the pressure
// gradient's own.  These two kernels save the faces a non-open side owns and
// put them back, so the sealed operator stays the one live transcription of
// the open-boundary arithmetic and no side receives a branch CM1 would skip.
extern "C" __global__
void cm1_bc_face_save_f32_v1(const float* u,const float* v,float* su,float* sv,
                             unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                             unsigned int wbc,unsigned int ebc,unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const unsigned long long we=(unsigned long long)nk*nj,sn=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<we;q+=step){
        const unsigned int j0=(unsigned int)(q%nj),k=(unsigned int)(q/nj)+1u,y=j0+ng;
        if(wbc!=2u) su[q]=u[i3(k,y,ng,nyu,nxu)];
        if(ebc!=2u) su[we+q]=u[i3(k,y,ni+ng,nyu,nxu)];
    }
    for(unsigned long long q=first;q<sn;q+=step){
        const unsigned int i0=(unsigned int)(q%ni),k=(unsigned int)(q/ni)+1u,x=i0+ng;
        if(sbc!=2u) sv[q]=v[i3(k,ng,x,nyv,nxv)];
        if(nbc!=2u) sv[sn+q]=v[i3(k,nj+ng,x,nyv,nxv)];
    }
}

extern "C" __global__
void cm1_bc_face_restore_f32_v1(float* u,float* v,const float* su,const float* sv,
                                unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                                unsigned int wbc,unsigned int ebc,unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const unsigned long long we=(unsigned long long)nk*nj,sn=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<we;q+=step){
        const unsigned int j0=(unsigned int)(q%nj),k=(unsigned int)(q/nj)+1u,y=j0+ng;
        if(wbc!=2u) u[i3(k,y,ng,nyu,nxu)]=su[q];
        if(ebc!=2u) u[i3(k,y,ni+ng,nyu,nxu)]=su[we+q];
    }
    for(unsigned long long q=first;q<sn;q+=step){
        const unsigned int i0=(unsigned int)(q%ni),k=(unsigned int)(q/ni)+1u,x=i0+ng;
        if(sbc!=2u) v[i3(k,ng,x,nyv,nxv)]=sv[q];
        if(nbc!=2u) v[i3(k,nj+ng,x,nyv,nxv)]=sv[sn+q];
    }
}

// src/sound.F:382-402 for the boundary faces only.  side: 0 west u(i=1),
// 1 east u(i=ni+1), 2 south v(j=1), 3 north v(j=nj+1).  The expression is
// CM1's terrain-branch horizontal pressure gradient; on a flat grid gxu and
// gyv are zero and only the horizontal difference survives.
extern "C" __global__
void cm1_bc_face_pgrad_f32_v1(const float* ppd,const float* thv,const float* uten,const float* vten,
                              const float* gxu,const float* gyv,float* u,float* v,
                              unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
                              float dts,float rdx,float rdy,float dz,float cp,
                              unsigned int wbc,unsigned int ebc,unsigned int sbc,unsigned int nbc)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const float halfcp=m32(cp,0.5f),eighth=0.125f,rdsv=d32(1.0f,dz);
    const unsigned long long nu=(unsigned long long)nk*nj*2u;
    const unsigned long long nv=(unsigned long long)nk*ni*2u;
    for(unsigned long long z=first;z<nu+nv;z+=step){
        if(z<nu){
            const unsigned int half=(unsigned int)(z%2u);
            const unsigned long long q=z/2u;
            const unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u;
            if(half==0u){ if(wbc!=1u) continue; } else if(ebc!=1u) continue;
            const unsigned int ff=(half==0u)?1u:(ni+1u);
            const float h=m32(s32(ppd[pidx(k,j,ff,ni,nj)],ppd[pidx(k,j,ff-1u,ni,nj)]),rdx);
            const float dph=(k<nk)?m32(s32(ppd[pidx(k+1u,j,ff,ni,nj)],ppd[pidx(k,j,ff,ni,nj)]),rdsv):0.0f;
            const float dphl=(k<nk)?m32(s32(ppd[pidx(k+1u,j,ff-1u,ni,nj)],ppd[pidx(k,j,ff-1u,ni,nj)]),rdsv):0.0f;
            const float dpl=(k>1u)?m32(s32(ppd[pidx(k,j,ff,ni,nj)],ppd[pidx(k-1u,j,ff,ni,nj)]),rdsv):0.0f;
            const float dpll=(k>1u)?m32(s32(ppd[pidx(k,j,ff-1u,ni,nj)],ppd[pidx(k-1u,j,ff-1u,ni,nj)]),rdsv):0.0f;
            const float vs=a32(a32(dph,dphl),a32(dpl,dpll));
            const unsigned int j0=j-1u,i0=ff-1u;
            const float tr=m32(m32(eighth,vs),a32(gxu[((unsigned long long)(k-1u)*nj+j0)*(ni+1u)+i0],
                                                 gxu[((unsigned long long)k*nj+j0)*(ni+1u)+i0]));
            const float pg=m32(m32(halfcp,a32(thv[pidx(k,j,ff,ni,nj)],thv[pidx(k,j,ff-1u,ni,nj)])),a32(h,tr));
            const unsigned long long x=i3(k,j0+ng,i0+ng,nyu,nxu);
            u[x]=a32(u[x],m32(dts,s32(uten[x],pg)));
        }else{
            const unsigned long long zz=z-nu;
            const unsigned int half=(unsigned int)(zz%2u);
            const unsigned long long q=zz/2u;
            const unsigned int i=(unsigned int)(q%ni)+1u,k=(unsigned int)(q/ni)+1u;
            if(half==0u){ if(sbc!=1u) continue; } else if(nbc!=1u) continue;
            const unsigned int ff=(half==0u)?1u:(nj+1u);
            const float h=m32(s32(ppd[pidx(k,ff,i,ni,nj)],ppd[pidx(k,ff-1u,i,ni,nj)]),rdy);
            const float dph=(k<nk)?m32(s32(ppd[pidx(k+1u,ff,i,ni,nj)],ppd[pidx(k,ff,i,ni,nj)]),rdsv):0.0f;
            const float dphl=(k<nk)?m32(s32(ppd[pidx(k+1u,ff-1u,i,ni,nj)],ppd[pidx(k,ff-1u,i,ni,nj)]),rdsv):0.0f;
            const float dpl=(k>1u)?m32(s32(ppd[pidx(k,ff,i,ni,nj)],ppd[pidx(k-1u,ff,i,ni,nj)]),rdsv):0.0f;
            const float dpll=(k>1u)?m32(s32(ppd[pidx(k,ff-1u,i,ni,nj)],ppd[pidx(k-1u,ff-1u,i,ni,nj)]),rdsv):0.0f;
            const float vs=a32(a32(dph,dphl),a32(dpl,dpll));
            const unsigned int j0=ff-1u,i0=i-1u;
            const float tr=m32(m32(eighth,vs),a32(gyv[((unsigned long long)(k-1u)*(nj+1u)+j0)*ni+i0],
                                                 gyv[((unsigned long long)k*(nj+1u)+j0)*ni+i0]));
            const float pg=m32(m32(halfcp,a32(thv[pidx(k,ff,i,ni,nj)],thv[pidx(k,ff-1u,i,ni,nj)])),a32(h,tr));
            const unsigned long long y=i3(k,j0+ng,i0+ng,nyv,nxv);
            v[y]=a32(v[y],m32(dts,s32(vten[y],pg)));
        }
    }
}
"""

CUDA_BOUNDARY_FAMILIES_SOURCE_BYTES = CUDA_BOUNDARY_FAMILIES_SOURCE.encode("utf-8")
CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256 = (
    "4698c5baf5e4e2c4d17ae832ac730347e82c01fd45191c21a5a3eed8f3afda46"
)


#: bc.F:2254-2256 and :2266-2268 (and :2360-2362, :2372-2374 for v):
#:     u1=rho0(1,j,k)*u3d(i,j,k)
#:     if(u1.lt.0.0) u3d(i,j,k)=u1*t3/rho0(1,j,k)
#: ``u1`` is double precision but ``rho0*u3d`` is a product of two default
#: REALs, so CM1 rounds it to binary32 before it widens it; the rescale above
#: widens both factors first and keeps the exact product.  The two rescale
#: kernels in CM1's order, derived from the sealed text (which stays what the
#: uniform column's acoustic stage launches); the flat stage launches these.
CUDA_BOUNDARY_ROFLUX_CM1_ORDER_KERNEL_NAMES = (
    "cm1_bc_roflux_apply_we_cm1order_f32_v1",
    "cm1_bc_roflux_apply_sn_cm1order_f32_v1",
)


def _roflux_cm1_order_source() -> str:
    text = CUDA_BOUNDARY_FAMILIES_SOURCE
    head = text[: text.index("// ---------------------------------------------------------------------------\n// src/bc.F:180-1143.")]
    start = text.index('extern "C" __global__\nvoid cm1_bc_roflux_apply_we_f32_v1(')
    stop = text.index("// src/solve2.F:726-822.")
    body = text[start:stop]
    for old, new, count in (
        ("// gpuwm-cm1.cuda-boundary-families-kernels/v1\n", "// gpuwm-cm1.cuda-boundary-roflux-cm1-order-kernels/v1\n", 1),
    ):
        if head.count(old) != count:
            raise RuntimeError("the boundary-family header changed; derive the roflux rescale again")
        head = head.replace(old, new)
    for old, new, count in (
        ("void cm1_bc_roflux_apply_we_f32_v1(", "void cm1_bc_roflux_apply_we_cm1order_f32_v1(", 1),
        ("void cm1_bc_roflux_apply_sn_f32_v1(", "void cm1_bc_roflux_apply_sn_cm1order_f32_v1(", 1),
        ("const double u1=(double)rho0[m]*(double)u3d[x];", "const double u1=(double)m32(rho0[m],u3d[x]);", 2),
        ("const double v1=(double)rho0[m]*(double)v3d[y];", "const double v1=(double)m32(rho0[m],v3d[y]);", 2),
    ):
        if body.count(old) != count:
            raise RuntimeError(f"roflux rescale derivation: {old!r} found {body.count(old)} times")
        body = body.replace(old, new)
    return head + body


CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE = _roflux_cm1_order_source()
CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE_SHA256 = hashlib.sha256(
    CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE.encode("utf-8")
).hexdigest()


def _validate_dimensions(ni: int, nj: int, nk: int, ng: int) -> None:
    for name, value, lower in (("ni", ni, 4), ("nj", nj, 4), ("nk", nk, 3), ("ng", ng, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if ni <= ng or nj <= ng:
        raise ValueError(
            "a periodic or rigid fill reads ngxy interior planes; with ni or nj "
            "no larger than ngxy the ghost planes of the two sides would read "
            "each other and the wrap would be built from ghost words"
        )


def family_array_shape(kind: int, *, ni: int, nj: int, nk: int, ng: int) -> tuple[int, int, int]:
    """The padded extent of the carrier ``kind`` names."""

    _validate_dimensions(ni, nj, nk, ng)
    if kind not in (KIND_SCALAR, KIND_U, KIND_V, KIND_W):
        raise ValueError("kind must be 0 scalar, 1 u, 2 v or 3 w")
    nx = ni + 1 + 2 * ng if kind == KIND_U else ni + 2 * ng
    ny = nj + 1 + 2 * ng if kind == KIND_V else nj + 2 * ng
    nz = nk + 3 if kind == KIND_W else nk + 2
    return (nz, ny, nx)


def _side_plan(
    mode: int, *, side: int, kind: int, staggered_kind: int, layer: int, ng: int, n: int
) -> tuple[int, int, float, int | None]:
    """Return ``(destination, source, sign, wall)`` padded indices for one ghost plane.

    ``staggered_kind`` is the ``kind`` that is staggered along this axis --
    ``KIND_U`` for west/east and ``KIND_V`` for south/north.
    """

    stag = kind == staggered_kind
    if side == 0:
        if mode == PERIODIC:
            return ng - 1 - layer, n + ng - 1 - layer, 1.0, None
        if mode == OPEN_RADIATIVE:
            return layer, ng, 1.0, None
        if stag:
            return ng - 1 - layer, ng + 1 + layer, -1.0, ng
        return ng - 1 - layer, ng + layer, 1.0, None
    if mode == PERIODIC:
        if stag:
            return n + ng + 1 + layer, ng + 1 + layer, 1.0, None
        return n + ng + layer, ng + layer, 1.0, None
    if mode == OPEN_RADIATIVE:
        if stag:
            return n + ng + 1 + layer, n + ng, 1.0, None
        return n + ng + layer, n + ng - 1, 1.0, None
    if stag:
        return n + ng + 1 + layer, n + ng - 1 - layer, -1.0, n + ng
    return n + ng + layer, n + ng - 1 - layer, 1.0, None


def cpu_family_we_inplace(
    array: Any, *, ni: int, nj: int, nk: int, ng: int, kind: int, wbc: int, ebc: int
) -> None:
    """``src/bc.F`` west and east ghost planes for one carrier, in place."""

    _validate_dimensions(ni, nj, nk, ng)
    shape = family_array_shape(kind, ni=ni, nj=nj, nk=nk, ng=ng)
    if tuple(array.shape) != shape:
        raise ValueError(f"kind {kind} carrier must have padded shape {shape}")
    ny = shape[1]
    kfirst, kcount = (2, nk - 1) if kind == KIND_W else (1, nk)
    jhi = nj + 2 if kind == KIND_V else nj + 1
    ks = slice(kfirst, kfirst + kcount)
    for side, mode in ((0, wbc), (1, ebc)):
        if mode == 0:
            continue
        rows = slice(0, ny) if mode == PERIODIC else slice(ng - 1, ng - 1 + jhi + 1)
        for layer in range(ng):
            dst, src, sign, wall = _side_plan(
                mode, side=side, kind=kind, staggered_kind=KIND_U, layer=layer, ng=ng, n=ni
            )
            value = array[ks, rows, src]
            array[ks, rows, dst] = -value if sign < 0 else value
            if wall is not None and layer == 0:
                array[ks, rows, wall] = 0.0


def cpu_family_sn_inplace(
    array: Any, *, ni: int, nj: int, nk: int, ng: int, kind: int, sbc: int, nbc: int
) -> None:
    """``src/bc.F`` south and north ghost planes for one carrier, in place."""

    _validate_dimensions(ni, nj, nk, ng)
    shape = family_array_shape(kind, ni=ni, nj=nj, nk=nk, ng=ng)
    if tuple(array.shape) != shape:
        raise ValueError(f"kind {kind} carrier must have padded shape {shape}")
    nx = shape[2]
    kfirst, kcount = (2, nk - 1) if kind == KIND_W else (1, nk)
    ihi = ni + 2 if kind == KIND_U else ni + 1
    ks = slice(kfirst, kfirst + kcount)
    for side, mode in ((0, sbc), (1, nbc)):
        if mode == 0:
            continue
        cols = slice(0, nx) if mode == PERIODIC else slice(ng - 1, ng - 1 + ihi + 1)
        for layer in range(ng):
            dst, src, sign, wall = _side_plan(
                mode, side=side, kind=kind, staggered_kind=KIND_V, layer=layer, ng=ng, n=nj
            )
            value = array[ks, src, cols]
            array[ks, dst, cols] = -value if sign < 0 else value
            if wall is not None and layer == 0:
                array[ks, wall, cols] = 0.0


def cpu_pressure_we_inplace(pressure: Any, *, ni: int, nj: int, nk: int, wbc: int, ebc: int) -> None:
    """``src/bc.F:372-520`` west and east pressure ghost columns."""

    if tuple(pressure.shape) != (nk + 2, nj + 2, ni + 2):
        raise ValueError("pressure carrier must have shape (nk+2, nj+2, ni+2)")
    ks = slice(1, nk + 1)
    if wbc:
        pressure[ks, :, 0] = pressure[ks, :, ni if wbc == PERIODIC else 1]
    if ebc:
        pressure[ks, :, ni + 1] = pressure[ks, :, 1 if ebc == PERIODIC else ni]


def cpu_pressure_sn_inplace(pressure: Any, *, ni: int, nj: int, nk: int, sbc: int, nbc: int) -> None:
    """``src/bc.F:372-520`` south and north pressure ghost rows."""

    if tuple(pressure.shape) != (nk + 2, nj + 2, ni + 2):
        raise ValueError("pressure carrier must have shape (nk+2, nj+2, ni+2)")
    ks = slice(1, nk + 1)
    if sbc:
        pressure[ks, 0, :] = pressure[ks, nj if sbc == PERIODIC else 1, :]
    if nbc:
        pressure[ks, nj + 1, :] = pressure[ks, 1 if nbc == PERIODIC else nj, :]


def cpu_radbcew_inplace(
    ua: Any, radbcw: Any, radbce: Any, *, ni: int, nj: int, nk: int, ng: int, wbc: int, ebc: int, cstar: float
) -> None:
    """``src/bc.F:1948-1993`` -- the irbc=1/2 west and east phase speeds."""

    np = importlib.import_module("numpy")
    c = np.float32(cstar)
    rows = slice(ng, ng + nj)
    ks = slice(1, nk + 1)
    if wbc == OPEN_RADIATIVE:
        value = np.subtract(ua[ks, rows, ng], c, dtype=np.float32)
        radbcw[ks, rows] = np.where(value < np.float32(0.0), value, np.float32(0.0))
    if ebc == OPEN_RADIATIVE:
        value = np.add(ua[ks, rows, ni + ng], c, dtype=np.float32)
        radbce[ks, rows] = np.where(value > np.float32(0.0), value, np.float32(0.0))


def cpu_radbcns_inplace(
    va: Any, radbcs: Any, radbcn: Any, *, ni: int, nj: int, nk: int, ng: int, sbc: int, nbc: int, cstar: float
) -> None:
    """``src/bc.F:2001-2046`` -- the irbc=1/2 south and north phase speeds."""

    np = importlib.import_module("numpy")
    c = np.float32(cstar)
    cols = slice(ng, ng + ni)
    ks = slice(1, nk + 1)
    if sbc == OPEN_RADIATIVE:
        value = np.subtract(va[ks, ng, cols], c, dtype=np.float32)
        radbcs[ks, cols] = np.where(value < np.float32(0.0), value, np.float32(0.0))
    if nbc == OPEN_RADIATIVE:
        value = np.add(va[ks, nj + ng, cols], c, dtype=np.float32)
        radbcn[ks, cols] = np.where(value > np.float32(0.0), value, np.float32(0.0))


def cpu_restrict_openbc_we_inplace(
    rho0: Any, u3d: Any, rvh: Any, rmh: Any, *, ni: int, nj: int, nk: int, ng: int, wbc: int, ebc: int
) -> float:
    """``src/bc.F:2180-2278``.  Returns CM1's ``t3`` flux ratio."""

    np = importlib.import_module("numpy")
    temout = np.zeros(nk, dtype=np.float64)
    temin = np.zeros(nk, dtype=np.float64)
    zero32 = np.float32(0.0)
    for k in range(nk):
        if wbc == OPEN_RADIATIVE:
            for j in range(1, nj + 1):
                f = np.float32(
                    np.float32(np.float32(rho0[k, j - 1, 0] * u3d[k + 1, j + ng - 1, ng]) * rvh[j + ng - 1])
                    * rmh[k, j - 1, 0]
                )
                temout[k] -= float(min(zero32, f))
                temin[k] += float(max(zero32, f))
        if ebc == OPEN_RADIATIVE:
            for j in range(1, nj + 1):
                f = np.float32(
                    np.float32(np.float32(rho0[k, j - 1, ni - 1] * u3d[k + 1, j + ng - 1, ni + ng]) * rvh[j + ng - 1])
                    * rmh[k, j - 1, ni - 1]
                )
                temout[k] += float(max(zero32, f))
                temin[k] -= float(min(zero32, f))
    fluxout = 0.0
    fluxin = 0.0
    for k in range(nk):
        fluxout += float(temout[k])
        fluxin += float(temin[k])
    t3 = (fluxin + 1.0e-20) / (fluxout + 1.0e-20)
    for k in range(1, nk + 1):
        if wbc == OPEN_RADIATIVE:
            for j in range(1, nj + 1):
                rho = float(rho0[k - 1, j - 1, 0])
                u1 = rho * float(u3d[k, j + ng - 1, ng])
                if u1 < 0.0:
                    u3d[k, j + ng - 1, ng] = np.float32(u1 * t3 / rho)
        if ebc == OPEN_RADIATIVE:
            for j in range(1, nj + 1):
                rho = float(rho0[k - 1, j - 1, ni - 1])
                u1 = rho * float(u3d[k, j + ng - 1, ni + ng])
                if u1 > 0.0:
                    u3d[k, j + ng - 1, ni + ng] = np.float32(u1 * t3 / rho)
    return t3


def cpu_restrict_openbc_sn_inplace(
    rho0: Any, v3d: Any, ruh: Any, rmh: Any, *, ni: int, nj: int, nk: int, ng: int, sbc: int, nbc: int
) -> float:
    """``src/bc.F:2286-2384``.  Returns CM1's ``t3`` flux ratio."""

    np = importlib.import_module("numpy")
    temout = np.zeros(nk, dtype=np.float64)
    temin = np.zeros(nk, dtype=np.float64)
    zero32 = np.float32(0.0)
    for k in range(nk):
        if sbc == OPEN_RADIATIVE:
            for i in range(1, ni + 1):
                f = np.float32(
                    np.float32(np.float32(rho0[k, 0, i - 1] * v3d[k + 1, ng, i + ng - 1]) * ruh[i + ng - 1])
                    * rmh[k, 0, i - 1]
                )
                temout[k] -= float(min(zero32, f))
                temin[k] += float(max(zero32, f))
        if nbc == OPEN_RADIATIVE:
            for i in range(1, ni + 1):
                f = np.float32(
                    np.float32(np.float32(rho0[k, nj - 1, i - 1] * v3d[k + 1, nj + ng, i + ng - 1]) * ruh[i + ng - 1])
                    * rmh[k, nj - 1, i - 1]
                )
                temout[k] += float(max(zero32, f))
                temin[k] -= float(min(zero32, f))
    fluxout = 0.0
    fluxin = 0.0
    for k in range(nk):
        fluxout += float(temout[k])
        fluxin += float(temin[k])
    t3 = (fluxin + 1.0e-20) / (fluxout + 1.0e-20)
    for k in range(1, nk + 1):
        if sbc == OPEN_RADIATIVE:
            for i in range(1, ni + 1):
                rho = float(rho0[k - 1, 0, i - 1])
                v1 = rho * float(v3d[k, ng, i + ng - 1])
                if v1 < 0.0:
                    v3d[k, ng, i + ng - 1] = np.float32(v1 * t3 / rho)
        if nbc == OPEN_RADIATIVE:
            for i in range(1, ni + 1):
                rho = float(rho0[k - 1, nj - 1, i - 1])
                v1 = rho * float(v3d[k, nj + ng, i + ng - 1])
                if v1 > 0.0:
                    v3d[k, nj + ng, i + ng - 1] = np.float32(v1 * t3 / rho)
    return t3


def cpu_nudgeobc_we_inplace(
    u3d: Any, u0: Any, uten: Any, *, ni: int, nj: int, nk: int, ng: int, wbc: int, ebc: int, alphobc: float
) -> None:
    """``src/solve2.F:726-748`` -- nudge the inflow u faces toward the base state."""

    np = importlib.import_module("numpy")
    tem = np.divide(np.float32(1.0), np.float32(alphobc), dtype=np.float32)
    rows = slice(ng, ng + nj)
    ks = slice(1, nk + 1)
    profile = np.asarray(u0, dtype=np.float32)[:nk, None]
    if wbc == OPEN_RADIATIVE:
        face = u3d[ks, rows, ng]
        delta = np.multiply(np.subtract(face, profile, dtype=np.float32), tem, dtype=np.float32)
        uten[ks, rows, ng] = np.where(
            face > np.float32(0.0),
            np.subtract(uten[ks, rows, ng], delta, dtype=np.float32),
            uten[ks, rows, ng],
        )
    if ebc == OPEN_RADIATIVE:
        face = u3d[ks, rows, ni + ng]
        delta = np.multiply(np.subtract(face, profile, dtype=np.float32), tem, dtype=np.float32)
        uten[ks, rows, ni + ng] = np.where(
            face < np.float32(0.0),
            np.subtract(uten[ks, rows, ni + ng], delta, dtype=np.float32),
            uten[ks, rows, ni + ng],
        )


def cpu_nudgeobc_sn_inplace(
    v3d: Any, v0: Any, vten: Any, *, ni: int, nj: int, nk: int, ng: int, sbc: int, nbc: int, alphobc: float
) -> None:
    """``src/solve2.F:801-822`` -- nudge the inflow v faces toward the base state."""

    np = importlib.import_module("numpy")
    tem = np.divide(np.float32(1.0), np.float32(alphobc), dtype=np.float32)
    cols = slice(ng, ng + ni)
    ks = slice(1, nk + 1)
    profile = np.asarray(v0, dtype=np.float32)[:nk, None]
    if sbc == OPEN_RADIATIVE:
        face = v3d[ks, ng, cols]
        delta = np.multiply(np.subtract(face, profile, dtype=np.float32), tem, dtype=np.float32)
        vten[ks, ng, cols] = np.where(
            face > np.float32(0.0),
            np.subtract(vten[ks, ng, cols], delta, dtype=np.float32),
            vten[ks, ng, cols],
        )
    if nbc == OPEN_RADIATIVE:
        face = v3d[ks, nj + ng, cols]
        delta = np.multiply(np.subtract(face, profile, dtype=np.float32), tem, dtype=np.float32)
        vten[ks, nj + ng, cols] = np.where(
            face < np.float32(0.0),
            np.subtract(vten[ks, nj + ng, cols], delta, dtype=np.float32),
            vten[ks, nj + ng, cols],
        )


def cpu_hrdamp_tauf_inplace(
    tauf: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    i0: int = 0,
    j0: int = 0,
    nx: int | None = None,
    ny: int | None = None,
    dx: float,
    dy: float,
    xhd: float,
) -> None:
    """``src/param.F:8736-8776`` -- the four-side lateral taper, in place on tauf."""

    np = importlib.import_module("numpy")
    nx = ni if nx is None else nx
    ny = nj if ny is None else ny
    if tuple(tauf.shape) != (nk + 1, nj + 2, ni + 2):
        raise ValueError("tauf must have shape (nk+1, nj+2, ni+2)")
    pi = np.float32(3.1415926535897932384626433)
    f32 = np.float32
    xhd32 = f32(xhd)

    def taper(distance: Any) -> Any:
        scaled = np.divide(np.multiply(pi, distance, dtype=f32), xhd32, dtype=f32)
        return np.multiply(
            f32(0.5), np.subtract(f32(1.0), np.cos(scaled, dtype=f32), dtype=f32), dtype=f32
        )

    xh = np.multiply(
        f32(dx), np.subtract(np.arange(i0, i0 + ni + 2, dtype=f32), f32(0.5), dtype=f32), dtype=f32
    )
    yh = np.multiply(
        f32(dy), np.subtract(np.arange(j0, j0 + nj + 2, dtype=f32), f32(0.5), dtype=f32), dtype=f32
    )
    if nx > 1:
        for distance in (
            np.subtract(xhd32, xh, dtype=f32),
            np.subtract(xh, np.subtract(np.multiply(f32(nx), f32(dx), dtype=f32), xhd32, dtype=f32), dtype=f32),
        ):
            active = distance > f32(0.0)
            if not bool(np.any(active)):
                continue
            value = np.where(active, taper(distance), f32(0.0))
            np.maximum(tauf, value[None, None, :], out=tauf)
    if ny > 1:
        for distance in (
            np.subtract(xhd32, yh, dtype=f32),
            np.subtract(yh, np.subtract(np.multiply(f32(ny), f32(dy), dtype=f32), xhd32, dtype=f32), dtype=f32),
        ):
            active = distance > f32(0.0)
            if not bool(np.any(active)):
                continue
            value = np.where(active, taper(distance), f32(0.0))
            np.maximum(tauf, value[None, :, None], out=tauf)


def cpu_irdamp2_operands(
    uavg: Any, vavg: Any, thavg: Any, th0: Any, tha: Any, *, ni: int, nj: int, nk: int
) -> tuple[Any, Any, Any]:
    """``src/solve1.F:427-438`` -- the three operands irdamp=2 replaces."""

    np = importlib.import_module("numpy")
    f32 = np.float32
    u0e = np.broadcast_to(np.asarray(uavg, dtype=f32)[:nk, None, None], (nk, nj, ni + 1)).copy()
    v0e = np.broadcast_to(np.asarray(vavg, dtype=f32)[:nk, None, None], (nk, nj + 1, ni)).copy()
    thae = np.add(
        np.subtract(th0, np.asarray(thavg, dtype=f32)[:nk, None, None], dtype=f32), tha, dtype=f32
    )
    return u0e, v0e, thae


def cpu_domain_average_uvt(
    ua: Any, va: Any, th0: Any, tha: Any, *, ni: int, nj: int, nk: int, dx: float, dy: float
) -> tuple[Any, Any, Any]:
    """``src/testcase_simple_phys.F:307-375`` -- the area-weighted level means.

    ``ruh``/``rvh``/``ruf``/``rvf`` are exactly one on this port's uniform
    horizontal grid, so they are absent from the product rather than carried as
    a multiply by one.
    """

    np = importlib.import_module("numpy")
    tem = np.float32(np.float32(dx) * np.float32(dy))
    cavg = np.zeros((3, nk), dtype=np.float64)
    for k in range(nk):
        cavg[0, k] = np.sum(np.multiply(ua[k, :nj, :ni], tem, dtype=np.float32).astype(np.float64))
        cavg[1, k] = np.sum(np.multiply(va[k, :nj, :ni], tem, dtype=np.float32).astype(np.float64))
        total = np.add(th0[k, :nj, :ni], tha[k, :nj, :ni], dtype=np.float32)
        cavg[2, k] = np.sum(np.multiply(total, tem, dtype=np.float32).astype(np.float64))
    temd = 1.0 / (float(ni) * float(dx) * float(nj) * float(dy))
    return tuple(np.asarray(cavg[n] * temd, dtype=np.float32) for n in range(3))


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    ):
        return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "", 1)
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledBoundaryFamilyKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    source_sha256: str


def compile_cuda_boundary_families() -> CompiledBoundaryFamilyKernels:
    if hashlib.sha256(CUDA_BOUNDARY_FAMILIES_SOURCE_BYTES).hexdigest() != CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256:
        raise RuntimeError("packaged boundary-family CUDA source differs from its pin")
    return _compile_family_image(CUDA_BOUNDARY_FAMILIES_SOURCE, CUDA_BOUNDARY_FAMILIES_KERNEL_NAMES,
                                 "cm1_boundary_families", CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256)


def compile_cuda_boundary_roflux_cm1_order() -> CompiledBoundaryFamilyKernels:
    """The roflux rescale in CM1's order (bc.F:2254, :2360), for the flat acoustic stage."""

    return _compile_family_image(CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE, CUDA_BOUNDARY_ROFLUX_CM1_ORDER_KERNEL_NAMES,
                                 "cm1_boundary_roflux_cm1_order", CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE_SHA256)


def _compile_family_image(source: str, names: tuple[str, ...], stem: str, digest: str) -> CompiledBoundaryFamilyKernels:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        source, f"{stem}_ptx.cu", method="ptx"
    ).compile(common + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        source, f"{stem}_cubin.cu", method="cubin"
    ).compile(common + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in names
    ):
        raise ValueError("compiled PTX lacks a boundary-family entry point")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in names)
    return CompiledBoundaryFamilyKernels(
        module, kernels, ptx, cubin, arch, digest
    )


class _BoundFamilyKernel:
    """A family kernel that carries its two side selectors.

    The sealed all-open fills take ``(array, ni, nj, nk, ng, kind)``; this
    appends the two CM1 side values so the general operator can be bound at
    exactly the call sites that already exist, with no call-site churn and no
    second live fill path.
    """

    __slots__ = ("_kernel", "_first", "_second", "_cupy")

    def __init__(self, kernel: Any, first: int, second: int) -> None:
        self._kernel = kernel
        self._first = first
        self._second = second
        self._cupy = importlib.import_module("cupy")

    def __call__(self, grid: Any, block: Any, arguments: Any, **kwargs: Any) -> Any:
        u32 = self._cupy.uint32
        return self._kernel(
            grid, block, (*tuple(arguments), u32(self._first), u32(self._second)), **kwargs
        )


def bound_lateral_sides(open_compiled: Any) -> tuple[int, int, int, int]:
    """CM1's ``(wbc, ebc, sbc, nbc)`` that ``open_compiled``'s bcs fills apply.

    An image bound by :func:`bind_lateral_families` carries its selection on
    the two ``bcs``/``bcu``/``bcv``/``bcw`` fill kernels (indices 5 and 6);
    the sealed unbound image is the all-open family its fills implement.
    Consumers that must agree with the halo those fills write -- the scalar
    advection and pdef edge treatment of adv.F:214-340 and pdef.F:49-166 --
    read the sides from here, so they cannot disagree with the halo.
    """

    kernels = tuple(getattr(open_compiled, "kernels", ()))
    if len(kernels) > 6 and isinstance(kernels[5], _BoundFamilyKernel) and isinstance(kernels[6], _BoundFamilyKernel):
        return (
            int(kernels[5]._first), int(kernels[5]._second),
            int(kernels[6]._first), int(kernels[6]._second),
        )
    return (OPEN_RADIATIVE, OPEN_RADIATIVE, OPEN_RADIATIVE, OPEN_RADIATIVE)


def bind_lateral_families(
    open_compiled: Any,
    family_compiled: CompiledBoundaryFamilyKernels,
    selection: LateralBoundarySelection,
) -> Any:
    """Return ``open_compiled`` with its four fill kernels bound to ``selection``.

    Indices 3 and 4 are the ``bcp`` pressure fills and 5 and 6 the
    ``bcs``/``bcu``/``bcv``/``bcw`` family fills.  Every consumer in this port
    reaches those four through the compiled tuple, so binding here is what
    makes a periodic, rigid or mixed family reach the momentum, scalar,
    pressure and thermodynamic halos at once.
    """

    if not isinstance(selection, LateralBoundarySelection):
        raise TypeError("selection must be a LateralBoundarySelection")
    if open_compiled.target_arch != family_compiled.target_arch:
        raise ValueError("the boundary-family image targets another architecture")
    kernels = list(open_compiled.kernels)
    kernels[3] = _BoundFamilyKernel(family_compiled.kernels[2], selection.west, selection.east)
    kernels[4] = _BoundFamilyKernel(family_compiled.kernels[3], selection.south, selection.north)
    kernels[5] = _BoundFamilyKernel(family_compiled.kernels[0], selection.west, selection.east)
    kernels[6] = _BoundFamilyKernel(family_compiled.kernels[1], selection.south, selection.north)
    return replace(open_compiled, kernels=tuple(kernels))


@dataclass(frozen=True, slots=True)
class LateralBoundaryStep:
    """One run's lateral family, its compiled kernels and its workspace.

    Handed to the acoustic launcher so the small-step loop can make the calls
    ``src/sound.F:260-302`` makes and this port's sealed all-open operator does
    not: the face hold for a non-open side, ``roflux``, the ``irbc=2``
    histories, and the periodic boundary face of ``src/sound.F:388``.
    """

    selection: LateralBoundarySelection
    compiled: CompiledBoundaryFamilyKernels
    workspace: Mapping[str, Any]
    #: The roflux rescale in CM1's order (compile_cuda_boundary_roflux_cm1_order),
    #: which the flat acoustic stage launches; ``None`` keeps the sealed one.
    roflux_cm1_order: CompiledBoundaryFamilyKernels | None = None

    def record(self) -> dict[str, Any]:
        return {
            "selection": self.selection.record(),
            "kernel_source_sha256": self.compiled.source_sha256,
            "workspace": sorted(self.workspace),
            **(
                {"roflux_rescale_source_sha256": self.roflux_cm1_order.source_sha256}
                if self.roflux_cm1_order is not None
                else {}
            ),
        }


def lateral_boundary_workspace_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    """The device scalars and face planes a non-default lateral family needs."""

    return {
        "temout": (nk,),
        "temin": (nk,),
        "t3": (1,),
        "face_u": (2, nk, nj),
        "face_v": (2, nk, ni),
    }


def enqueue_save_non_open_faces(
    arrays: Mapping[str, Any],
    workspace: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """Hold the normal-velocity faces ``src/sound.F:274-302`` would not touch."""

    _enqueue_face_transfer(
        arrays, workspace, index=14, ni=ni, nj=nj, nk=nk, ng=ng,
        selection=selection, compiled=compiled, stream=stream,
    )


def enqueue_restore_non_open_faces(
    arrays: Mapping[str, Any],
    workspace: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """Put them back, so a periodic or rigid face never takes an ssopenbc step."""

    _enqueue_face_transfer(
        arrays, workspace, index=15, ni=ni, nj=nj, nk=nk, ng=ng,
        selection=selection, compiled=compiled, stream=stream,
    )


def _enqueue_face_transfer(
    arrays: Mapping[str, Any],
    workspace: Mapping[str, Any],
    *,
    index: int,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    if selection.all_open:
        return
    cupy = importlib.import_module("cupy")
    u32 = cupy.uint32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    grid = (
        (max(ni, nj) * nk + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1
    )
    # Save and restore take the same four pointers in the same order; only the
    # direction of the copy differs, and that is which kernel is called.
    compiled.kernels[index](
        grid, threads,
        (arrays["u3d"], arrays["v3d"], workspace["face_u"], workspace["face_v"],
         u32(ni), u32(nj), u32(nk), u32(ng),
         u32(selection.west), u32(selection.east), u32(selection.south), u32(selection.north)),
        stream=stream,
    )


def enqueue_radbc(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
    momentum: str,
) -> None:
    """``src/bc.F:1948-2053`` over ``momentum`` -- ``"a"`` for irbc=1, ``"3d"`` for irbc=2."""

    cupy = importlib.import_module("cupy")
    u32, f32 = cupy.uint32, cupy.float32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    grid = ((max(ni, nj) * nk + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    u = arrays["ua"] if momentum == "a" else arrays["u3d"]
    v = arrays["va"] if momentum == "a" else arrays["v3d"]
    common = (u32(ni), u32(nj), u32(nk), u32(ng))
    compiled.kernels[4](
        grid, threads,
        (u, arrays["radbcw"], arrays["radbce"], *common,
         u32(selection.west), u32(selection.east), f32(selection.cstar)),
        stream=stream,
    )
    compiled.kernels[5](
        grid, threads,
        (v, arrays["radbcs"], arrays["radbcn"], *common,
         u32(selection.south), u32(selection.north), f32(selection.cstar)),
        stream=stream,
    )


def enqueue_restrict_outward_flux(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    workspace: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
    apply_compiled: CompiledBoundaryFamilyKernels | None = None,
) -> None:
    """``src/bc.F:2180-2391`` in CM1's own ``src/sound.F:283-302`` position.

    West/east first and south/north second, each with its own reduction, is
    CM1's order.  The two are independent: ``restrict_openbc_we`` reduces and
    rescales ``u3d`` alone and ``restrict_openbc_sn`` ``v3d`` alone
    (``restrict_openbc_sn(ruh,rmh,rho0,v3d)``), so neither reads what the other
    wrote.  With no open side nothing is enqueued.  ``apply_compiled`` is the
    rescale in CM1's order (:func:`compile_cuda_boundary_roflux_cm1_order`);
    ``None`` launches the sealed rescale, which widens ``rho0`` and ``u3d``
    before multiplying.
    """

    cupy = importlib.import_module("cupy")
    u32 = cupy.uint32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    reduce_grid = ((nk + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    apply_grid = (
        (max(ni, nj) * nk + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1
    )
    one = (1, 1, 1)
    common = (u32(ni), u32(nj), u32(nk), u32(ng))
    rho0, rmh = metrics["rho0_3d"], metrics["rmh"]
    if selection.west == OPEN_RADIATIVE or selection.east == OPEN_RADIATIVE:
        compiled.kernels[6](
            reduce_grid, threads,
            (rho0, arrays["u3d"], arrays["vh"], rmh, workspace["temout"], workspace["temin"],
             *common, u32(selection.west), u32(selection.east)),
            stream=stream,
        )
        compiled.kernels[8](one, one, (workspace["temout"], workspace["temin"], workspace["t3"], u32(nk)), stream=stream)
        (compiled.kernels[9] if apply_compiled is None else apply_compiled.kernels[0])(
            apply_grid, threads,
            (rho0, arrays["u3d"], workspace["t3"], *common, u32(selection.west), u32(selection.east)),
            stream=stream,
        )
    if selection.south == OPEN_RADIATIVE or selection.north == OPEN_RADIATIVE:
        compiled.kernels[7](
            reduce_grid, threads,
            (rho0, arrays["v3d"], arrays["uh"], rmh, workspace["temout"], workspace["temin"],
             *common, u32(selection.south), u32(selection.north)),
            stream=stream,
        )
        compiled.kernels[8](one, one, (workspace["temout"], workspace["temin"], workspace["t3"], u32(nk)), stream=stream)
        (compiled.kernels[10] if apply_compiled is None else apply_compiled.kernels[1])(
            apply_grid, threads,
            (rho0, arrays["v3d"], workspace["t3"], *common, u32(selection.south), u32(selection.north)),
            stream=stream,
        )


def enqueue_nudge_open_boundary(
    arrays: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """``src/solve2.F:726-822`` -- the nudgeobc inflow tendency."""

    cupy = importlib.import_module("cupy")
    numpy = importlib.import_module("numpy")
    u32, f32 = cupy.uint32, cupy.float32
    rtem = f32(numpy.divide(numpy.float32(1.0), numpy.float32(selection.alphobc), dtype=numpy.float32))
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    grid = ((max(ni, nj) * nk + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    common = (u32(ni), u32(nj), u32(nk), u32(ng))
    compiled.kernels[11](
        grid, threads,
        (arrays["u3d"], base["u0"], arrays["uten"], *common,
         u32(selection.west), u32(selection.east), rtem),
        stream=stream,
    )
    compiled.kernels[12](
        grid, threads,
        (arrays["v3d"], base["v0"], arrays["vten"], *common,
         u32(selection.south), u32(selection.north), rtem),
        stream=stream,
    )


def enqueue_periodic_face_pressure_gradient(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ng: int,
    dts: float,
    dx: float,
    dy: float,
    dz: float,
    cp: float,
    selection: LateralBoundarySelection,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """The boundary faces ``src/sound.F:388`` turns on when ``ib*`` is zero.

    ``src/param.F:6678-6681`` makes ``ibw`` zero exactly for ``wbc=1``, so this
    is a no-op on every open or rigid side and enqueues nothing at all when no
    side is periodic.
    """

    if PERIODIC not in selection.sides:
        return
    cupy = importlib.import_module("cupy")
    u32, f32 = cupy.uint32, cupy.float32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    words = 2 * nk * (ni + nj)
    grid = ((words + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    compiled.kernels[13](
        grid, threads,
        (arrays["ppd"], arrays["thv"], arrays["uten"], arrays["vten"],
         metrics["gxu"], metrics["gyv"], arrays["u3d"], arrays["v3d"],
         u32(ni), u32(nj), u32(nk), u32(ng),
         f32(dts), f32(1.0 / dx), f32(1.0 / dy), f32(dz), f32(cp),
         u32(selection.west), u32(selection.east), u32(selection.south), u32(selection.north)),
        stream=stream,
    )


def enqueue_hrdamp_tauf(
    tauf: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    i0: int,
    j0: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    xhd: float,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """``src/param.F:8736-8776`` on one tile's tauf, on the device."""

    cupy = importlib.import_module("cupy")
    u32, f32 = cupy.uint32, cupy.float32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    words = (nk + 1) * (nj + 2) * (ni + 2)
    grid = ((words + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    compiled.kernels[16](
        grid, threads,
        (tauf, u32(ni), u32(nj), u32(nk), u32(i0), u32(j0), u32(nx), u32(ny),
         f32(dx), f32(dy), f32(xhd)),
        stream=stream,
    )


def enqueue_irdamp2_operands(
    averages: Mapping[str, Any],
    th0: Any,
    tha: Any,
    targets: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    compiled: CompiledBoundaryFamilyKernels,
    stream: Any,
) -> None:
    """``src/solve1.F:427-438`` operand build, on the device."""

    cupy = importlib.import_module("cupy")
    u32 = cupy.uint32
    threads = (CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    words = max(nk * nj * (ni + 1), nk * (nj + 1) * ni)
    grid = ((words + CUDA_BOUNDARY_FAMILIES_THREADS - 1) // CUDA_BOUNDARY_FAMILIES_THREADS, 1, 1)
    compiled.kernels[17](
        grid, threads,
        (averages["uavg"], averages["vavg"], averages["thavg"], th0, tha,
         targets["u0"], targets["v0"], targets["tha"], u32(ni), u32(nj), u32(nk)),
        stream=stream,
    )


__all__ = [
    "CM1_ALPHOBC_DEFAULT",
    "CM1_CSTAR_DEFAULT",
    "CM1_CSTAR_FLOOR",
    "CUDA_BOUNDARY_FAMILIES_ABI",
    "CUDA_BOUNDARY_FAMILIES_KERNEL_NAMES",
    "CUDA_BOUNDARY_FAMILIES_SCHEMA",
    "CUDA_BOUNDARY_FAMILIES_SOURCE",
    "CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256",
    "CUDA_BOUNDARY_ROFLUX_CM1_ORDER_KERNEL_NAMES",
    "CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE",
    "CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE_SHA256",
    "CompiledBoundaryFamilyKernels",
    "DEFAULT_LATERAL_BOUNDARY_SELECTION",
    "KIND_SCALAR",
    "KIND_U",
    "KIND_V",
    "KIND_W",
    "LATERAL_VALUES",
    "OPEN_RADIATIVE",
    "PERIODIC",
    "RIGID_FREE_SLIP",
    "RIGID_NO_SLIP",
    "SOURCE_ANCHORS",
    "LateralBoundarySelection",
    "LateralBoundaryStep",
    "bind_lateral_families",
    "bound_lateral_sides",
    "compile_cuda_boundary_families",
    "compile_cuda_boundary_roflux_cm1_order",
    "cpu_domain_average_uvt",
    "cpu_family_sn_inplace",
    "cpu_family_we_inplace",
    "cpu_hrdamp_tauf_inplace",
    "cpu_irdamp2_operands",
    "cpu_nudgeobc_sn_inplace",
    "cpu_nudgeobc_we_inplace",
    "cpu_pressure_sn_inplace",
    "cpu_pressure_we_inplace",
    "cpu_radbcew_inplace",
    "cpu_radbcns_inplace",
    "cpu_restrict_openbc_sn_inplace",
    "cpu_restrict_openbc_we_inplace",
    "enqueue_hrdamp_tauf",
    "enqueue_irdamp2_operands",
    "enqueue_nudge_open_boundary",
    "enqueue_restore_non_open_faces",
    "enqueue_save_non_open_faces",
    "enqueue_periodic_face_pressure_gradient",
    "enqueue_radbc",
    "enqueue_restrict_outward_flux",
    "family_array_shape",
    "lateral_boundary_workspace_shapes",
]
