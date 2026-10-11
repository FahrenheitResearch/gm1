"""CM1 r21.1 moist terrain stage terms and post-acoustic flux bridge.

This module owns the source-ordered pieces that sit around the admitted
terrain/open acoustic solve:

* moist ``eqtset=2`` buoyancy, theta-rho, theta/pressure energy terms;
* the vertical buoyancy and terrain horizontal-buoyancy corrections; and
* post-sound ``getdiv`` followed by destructive ``getdivx`` mass-flux rebuild.

The base condensate ``qc0``/``qi0`` enters the buoyancy as CM1 writes it,
``g*(th3d*rth0 + repsm1*(qv-qv0) - (qli-qc0-qi0))`` (``solve2.F:404``).  A
caller that passes none gets exact-zero profiles, and ``(qli-0)-0`` is ``qli``
word for word, so every lane whose base state carries no condensate (the
initialized products, ``isnd`` 5, 7, 17 and the dry analytic soundings)
computes exactly the words it did before.  Both horizontally broadcast profiles and
terrain-varying physical base arrays are accepted so the bridge does not lock
the forecast driver to the earlier broadcast validation adapter.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


CUDA_MOIST_STAGE_BRIDGE_SCHEMA = "gpuwm-cm1.cuda-moist-stage-bridge/v1"
CUDA_MOIST_STAGE_BRIDGE_ABI = "gpuwm-cm1.cuda-moist-stage-bridge-kernels/v1"
CUDA_MOIST_STAGE_BRIDGE_THREADS = 128
CUDA_MOIST_STAGE_BRIDGE_CLAIM_LIMIT = (
    "CM1 r21.1 moist eqtset=2 stage thermodynamics, vertical buoyancy, terrain "
    "horizontal-buoyancy correction, and source-ordered post-sound getdiv then "
    "getdivx for Cartesian terrain, uniform horizontal spacing, uniform sigma, "
    "four open lateral boundaries, and base.F's base condensate (exact zero unless declared); "
    "Coriolis, SGS stress tendencies, damping, radiation, surface forcing, full "
    "large-step composition, forecast, storm, tornado, and validation claims are "
    "excluded"
)
SOURCE_ANCHORS = {
    "moist_accumulators": "src/solve2.F:300-392",
    "eqtset2_terms": "src/solve2.F:394-412",
    "vertical_buoyancy": "src/solve2.F:895-921",
    "terrain_buoyancy": "src/solve2.F:949-1015",
    "post_sound_getdiv": "src/misclibs.F:108-214",
    "post_sound_getdivx": "src/misclibs.F:222-370",
}
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/misclibs.F": "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
}


def cpu_eqtset2_cell_f32(
    *,
    th0: float,
    qv0: float,
    pi0: float,
    qc0: float = 0.0,
    qi0: float = 0.0,
    th3d: float,
    pp3d: float,
    qv: float,
    qc: float,
    qr: float,
    qi: float,
    qs: float,
    qg: float,
) -> dict[str, Any]:
    """Default-REAL scalar mirror of the selected solve2 eqtset=2 cell."""

    np = importlib.import_module("numpy")

    def f(value: Any) -> Any:
        return np.float32(value)

    def add(left: Any, right: Any) -> Any:
        return f(f(left) + f(right))

    def sub(left: Any, right: Any) -> Any:
        return f(f(left) - f(right))

    def mul(left: Any, right: Any) -> Any:
        return f(f(left) * f(right))

    def div(left: Any, right: Any) -> Any:
        return f(f(left) / f(right))

    rd, rv, cp, cpv = (f(287.04), f(461.5), f(1005.7), f(1870.0))
    cpl, cpi, gravity = (f(4190.0), f(2106.0), f(9.81))
    cv, cvv = sub(cp, rd), sub(cpv, rv)
    rovcp, reps = div(rd, cp), div(rv, rd)
    repsm1 = sub(reps, f(1.0))
    qv_live = np.maximum(f(qv), f(0.0))
    liquid = add(qc, qr)
    ice = add(add(qi, qs), qg)
    qli = np.maximum(add(liquid, ice), f(0.0))
    cpli = add(
        mul(cpl, np.maximum(liquid, f(0.0))),
        mul(cpi, np.maximum(ice, f(0.0))),
    )
    theta = add(th0, th3d)
    buoyancy = add(mul(th3d, div(f(1.0), th0)), mul(repsm1, sub(qv_live, qv0)))
    buoyancy = mul(gravity, sub(buoyancy, sub(sub(qli, qc0), qi0)))
    theta_rho = div(
        mul(theta, add(f(1.0), mul(reps, qv_live))),
        add(add(f(1.0), qv_live), qli),
    )
    cpm = add(add(cp, mul(cpv, qv_live)), cpli)
    cvm = div(f(1.0), add(add(cv, mul(cvv, qv_live)), cpli))
    theta_term = mul(
        mul(theta, sub(add(rd, mul(rv, qv_live)), mul(rovcp, cpm))), cvm
    )
    pressure_term = mul(mul(mul(add(pi0, pp3d), rovcp), cpm), cvm)
    return {
        "buoyancy": buoyancy,
        "theta_rho": theta_rho,
        "theta_term": theta_term,
        "pressure_term": pressure_term,
    }


CUDA_MOIST_STAGE_BRIDGE_KERNEL_NAMES = (
    "cm1_msb_terms_v1",
    "cm1_msb_piadv_v1",
    "cm1_msb_buoyancy_v1",
    "cm1_msb_getdiv_v1",
    "cm1_msb_getdivx_horizontal_v1",
    "cm1_msb_getdivx_vertical_v1",
    "cm1_msb_getdivx_divergence_v1",
)


CUDA_MOIST_STAGE_BRIDGE_SOURCE = r'''// gpuwm-cm1.cuda-moist-stage-bridge/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ unsigned long long pm(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long mi(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*nj+j)*ni+i;}
__device__ __forceinline__ float base(const float* a,unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int is3){return a[is3?mi(k,j,i,ni,nj):k];}

extern "C" __global__ void cm1_msb_terms_v1(
 const float* th0,const float* qv0,const float* qc0,const float* qi0,const float* pi0,const float* th3d,const float* q3d,const float* pp3d,
 float* thv,float* ppterm,float* dum8,float* thterm,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int base3d){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
 const float rd=287.04f,rv=461.5f,cp=1005.7f,cpv=1870.0f,cpl=4190.0f,cpi=2106.0f,g=9.81f;
 const float cv=s32(cp,rd),cvv=s32(cpv,rv),rovcp=d32(rd,cp),reps=d32(rv,rd),repsm1=s32(reps,1.0f);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj));unsigned long long p=pm(k+1u,j+1u,i+1u,ni,nj),s=i3(k+1u,j+ng,i+ng,nj+2u*ng,ni+2u*ng);
  float qv=fmaxf(q3d[z],0.0f),liq=a32(q3d[n+z],q3d[2ull*n+z]),ice=a32(a32(q3d[3ull*n+z],q3d[4ull*n+z]),q3d[5ull*n+z]),qli=fmaxf(a32(liq,ice),0.0f),cpli=a32(m32(cpl,fmaxf(liq,0.0f)),m32(cpi,fmaxf(ice,0.0f)));
  float bt=base(th0,k,j,i,ni,nj,base3d),bq=base(qv0,k,j,i,ni,nj,base3d),bp=base(pi0,k,j,i,ni,nj,base3d),rth=d32(1.0f,bt),theta=a32(bt,th3d[z]);
  float bc=base(qc0,k,j,i,ni,nj,base3d),bi=base(qi0,k,j,i,ni,nj,base3d);
  float b=a32(m32(th3d[z],rth),m32(repsm1,s32(qv,bq)));b=s32(b,s32(s32(qli,bc),bi));dum8[s]=m32(g,b);
  thv[p]=d32(m32(theta,a32(1.0f,m32(reps,qv))),a32(a32(1.0f,qv),qli));
  float cpm=a32(a32(cp,m32(cpv,qv)),cpli),cvm=d32(1.0f,a32(a32(cv,m32(cvv,qv)),cpli));
  thterm[z]=m32(m32(theta,s32(a32(rd,m32(rv,qv)),m32(rovcp,cpm))),cvm);
  float pi=a32(bp,pp3d[p]);ppterm[p]=m32(m32(m32(pi,rovcp),cpm),cvm);
 }}

extern "C" __global__ void cm1_msb_piadv_v1(const float* pi0,const float* pp3d,float* piadv,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int base3d){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*(nj+2u)*(ni+2u);
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%(ni+2u)),j=(unsigned int)((z/(ni+2u))%(nj+2u)),k=(unsigned int)(z/((unsigned long long)(ni+2u)*(nj+2u)))+1u;unsigned int ii=(i<1u)?0u:((i>ni)?ni-1u:i-1u),jj=(j<1u)?0u:((j>nj)?nj-1u:j-1u);piadv[pm(k,j,i,ni,nj)]=a32(base(pi0,k-1u,jj,ii,ni,nj,base3d),pp3d[pm(k,j,i,ni,nj)]);}}

__device__ __forceinline__ float bs(const float* b,unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){return b[i3(k,j+ng,i+ng,nj+2u*ng,ni+2u*ng)];}
__device__ __forceinline__ float bf(const float* b,const float* c1,const float* c2,unsigned int kf,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){if(kf==1u||kf==nk+1u)return 0.0f;unsigned long long p=mi(kf-1u,j,i,ni,nj);return a32(m32(c1[p],bs(b,kf-1u,j,i,ni,nj,ng)),m32(c2[p],bs(b,kf,j,i,ni,nj,ng)));}
extern "C" __global__ void cm1_msb_buoyancy_v1(const float* dum8,const float* c1,const float* c2,const float* gxu,const float* gyv,float* uten,float* vten,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nw=(unsigned long long)(nk-1u)*nj*ni,nu=(unsigned long long)nk*nj*(ni-1u),nv=(unsigned long long)nk*(nj-1u)*ni,total=nw+nu+nv;
 const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(unsigned long long z=first;z<total;z+=step){if(z<nw){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),kf=(unsigned int)(z/((unsigned long long)ni*nj))+2u;unsigned long long p=mi(kf-1u,j,i,ni,nj);float add=a32(m32(c1[p],bs(dum8,kf-1u,j,i,ni,nj,ng)),m32(c2[p],bs(dum8,kf,j,i,ni,nj,ng)));unsigned long long d=i3(kf,j+ng,i+ng,nyw,nxw);wten[d]=a32(wten[d],add);}
  else if(z<nw+nu){unsigned long long q=z-nw;unsigned int f=(unsigned int)(q%(ni-1u))+1u,j=(unsigned int)((q/(ni-1u))%nj),k=(unsigned int)(q/((unsigned long long)(ni-1u)*nj))+1u;float t=a32(a32(bf(dum8,c1,c2,k+1u,j,f,ni,nj,nk,ng),bf(dum8,c1,c2,k+1u,j,f-1u,ni,nj,nk,ng)),a32(bf(dum8,c1,c2,k,j,f,ni,nj,nk,ng),bf(dum8,c1,c2,k,j,f-1u,ni,nj,nk,ng)));t=m32(0.125f,t);t=s32(t,m32(0.25f,a32(bs(dum8,k,j,f,ni,nj,ng),bs(dum8,k,j,f-1u,ni,nj,ng))));t=m32(t,a32(gxu[mi(k-1u,j,f,ni+1u,nj)],gxu[mi(k,j,f,ni+1u,nj)]));unsigned long long d=i3(k,j+ng,f+ng,nyu,nxu);uten[d]=a32(uten[d],t);}
  else{unsigned long long q=z-nw-nu;unsigned int i=(unsigned int)(q%ni),f=(unsigned int)((q/ni)%(nj-1u))+1u,k=(unsigned int)(q/((unsigned long long)ni*(nj-1u)))+1u;float t=a32(a32(bf(dum8,c1,c2,k+1u,f,i,ni,nj,nk,ng),bf(dum8,c1,c2,k+1u,f-1u,i,ni,nj,nk,ng)),a32(bf(dum8,c1,c2,k,f,i,ni,nj,nk,ng),bf(dum8,c1,c2,k,f-1u,i,ni,nj,nk,ng)));t=m32(0.125f,t);t=s32(t,m32(0.25f,a32(bs(dum8,k,f,i,ni,nj,ng),bs(dum8,k,f-1u,i,ni,nj,ng))));t=m32(t,a32(gyv[mi(k-1u,f,i,ni,nj+1u)],gyv[mi(k,f,i,ni,nj+1u)]));unsigned long long d=i3(k,f+ng,i+ng,nyv,nxv);vten[d]=a32(vten[d],t);}}
}

__device__ __forceinline__ float xu(const float* u,const float* rgzu,unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj,unsigned int ng){return m32(u[i3(k,j+ng,f+ng,nj+2u*ng,ni+1u+2u*ng)],rgzu[(unsigned long long)j*(ni+1u)+f]);}
__device__ __forceinline__ float yv(const float* v,const float* rgzv,unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){return m32(v[i3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng)],rgzv[(unsigned long long)f*ni+i]);}
__device__ __forceinline__ float wt(const float* u,const float* v,const float* w,const float* rgzu,const float* rgzv,const float* dzdx,const float* dzdy,const float* gz,unsigned int kf,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){if(kf==1u||kf==nk+1u)return 0.0f;float rds=d32(1.0f,dz),sf=m32((float)(kf-1u),dz),sm=m32(s32((float)(kf-1u),0.5f),dz),r2=m32(s32(sf,sm),rds),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);float ux=a32(m32(r2,a32(xu(u,rgzu,kf,j,i,ni,nj,ng),xu(u,rgzu,kf,j,i+1u,ni,nj,ng))),m32(r1,a32(xu(u,rgzu,kf-1u,j,i,ni,nj,ng),xu(u,rgzu,kf-1u,j,i+1u,ni,nj,ng))));float vy=a32(m32(r2,a32(yv(v,rgzv,kf,j,i,ni,nj,ng),yv(v,rgzv,kf,j+1u,i,ni,nj,ng))),m32(r1,a32(yv(v,rgzv,kf-1u,j,i,ni,nj,ng),yv(v,rgzv,kf-1u,j+1u,i,ni,nj,ng))));float terrain=a32(m32(ux,dzdx[(unsigned long long)j*ni+i]),m32(vy,dzdy[(unsigned long long)j*ni+i]));float zt=m32((float)nk,dz);terrain=m32(m32(m32(terrain,s32(sf,zt)),gz[(unsigned long long)j*ni+i]),d32(1.0f,zt));return a32(w[i3(kf,j+ng,i+ng,nj+2u*ng,ni+2u*ng)],terrain);}
extern "C" __global__ void cm1_msb_getdiv_v1(const float* u,const float* v,const float* w,const float* rgzu,const float* rgzv,const float* dzdx,const float* dzdy,const float* gz,const float* thterm,float* div,float* theta_div,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,float smeps){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;float d=a32(m32(s32(xu(u,rgzu,k,j,i+1u,ni,nj,ng),xu(u,rgzu,k,j,i,ni,nj,ng)),rdx),m32(s32(yv(v,rgzv,k,j+1u,i,ni,nj,ng),yv(v,rgzv,k,j,i,ni,nj,ng)),rdy));d=a32(d,m32(s32(wt(u,v,w,rgzu,rgzv,dzdx,dzdy,gz,k+1u,j,i,ni,nj,nk,ng,dz),wt(u,v,w,rgzu,rgzv,dzdx,dzdy,gz,k,j,i,ni,nj,nk,ng,dz)),rdsf));if(fabsf(d)<smeps)d=0.0f;div[z]=d;theta_div[z]=m32(d,thterm[z]);}}

extern "C" __global__ void cm1_msb_getdivx_horizontal_v1(const float* rho0,const float* rgzu,const float* rgzv,float* rru,float* rrv,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,total=nu+nv;for(unsigned long long z=first;z<total;z+=step){if(z<nu){unsigned int f=(unsigned int)(z%(ni+1u)),j=(unsigned int)((z/(ni+1u))%nj),k=(unsigned int)(z/((unsigned long long)(ni+1u)*nj))+1u,l=(f==0u)?0u:f-1u,r=(f==ni)?ni-1u:f;float den=m32(0.5f,a32(rho0[mi(k-1u,j,l,ni,nj)],rho0[mi(k-1u,j,r,ni,nj)]));unsigned long long p=i3(k,j+ng,f+ng,nj+2u*ng,ni+1u+2u*ng);rru[p]=m32(m32(den,rru[p]),rgzu[(unsigned long long)j*(ni+1u)+f]);}else{unsigned long long q=z-nu;unsigned int i=(unsigned int)(q%ni),f=(unsigned int)((q/ni)%(nj+1u)),k=(unsigned int)(q/((unsigned long long)ni*(nj+1u)))+1u,s=(f==0u)?0u:f-1u,n=(f==nj)?nj-1u:f;float den=m32(0.5f,a32(rho0[mi(k-1u,s,i,ni,nj)],rho0[mi(k-1u,n,i,ni,nj)]));unsigned long long p=i3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng);rrv[p]=m32(m32(den,rrv[p]),rgzv[(unsigned long long)f*ni+i]);}}}

extern "C" __global__ void cm1_msb_getdivx_vertical_v1(const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;float rds=d32(1.0f,dz),zt=m32((float)nk,dz),rzt=d32(1.0f,zt);for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),kf=(unsigned int)(z/((unsigned long long)ni*nj))+1u,p=i3(kf,j+ng,i+ng,nj+2u*ng,ni+2u*ng);if(kf==1u||kf==nk+1u){rrw[p]=0.0f;}else{float sf=m32((float)(kf-1u),dz),sm=m32(s32((float)(kf-1u),0.5f),dz),r2=m32(s32(sf,sm),rds),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;float ux=a32(m32(r2,a32(rru[i3(kf,j+ng,i+ng,nyu,nxu)],rru[i3(kf,j+ng,i+1u+ng,nyu,nxu)])),m32(r1,a32(rru[i3(kf-1u,j+ng,i+ng,nyu,nxu)],rru[i3(kf-1u,j+ng,i+1u+ng,nyu,nxu)])));float vy=a32(m32(r2,a32(rrv[i3(kf,j+ng,i+ng,nyv,nxv)],rrv[i3(kf,j+1u+ng,i+ng,nyv,nxv)])),m32(r1,a32(rrv[i3(kf-1u,j+ng,i+ng,nyv,nxv)],rrv[i3(kf-1u,j+1u+ng,i+ng,nyv,nxv)])));float terrain=a32(m32(ux,dzdx[(unsigned long long)j*ni+i]),m32(vy,dzdy[(unsigned long long)j*ni+i]));terrain=m32(m32(m32(terrain,s32(sf,zt)),gz[(unsigned long long)j*ni+i]),rzt);rrw[p]=a32(m32(rf0[mi(kf-1u,j,i,ni,nj)],rrw[p]),terrain);}}}

extern "C" __global__ void cm1_msb_getdivx_divergence_v1(const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,float smeps){const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;float d=a32(m32(s32(rru[i3(k,j+ng,i+1u+ng,nyu,nxu)],rru[i3(k,j+ng,i+ng,nyu,nxu)]),rdx),m32(s32(rrv[i3(k,j+1u+ng,i+ng,nyv,nxv)],rrv[i3(k,j+ng,i+ng,nyv,nxv)]),rdy));d=a32(d,m32(s32(rrw[i3(k+1u,j+ng,i+ng,nyw,nxw)],rrw[i3(k,j+ng,i+ng,nyw,nxw)]),rdsf));divx[z]=(fabsf(d)<smeps)?0.0f:d;}}
'''
CUDA_MOIST_STAGE_BRIDGE_SOURCE_BYTES = CUDA_MOIST_STAGE_BRIDGE_SOURCE.encode("utf-8")
CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256 = (
    "a54763c96f18c6d2d41404615118315ffb35c8e5379d69f58e15a248db32ec3b"
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
class CompiledMoistStageBridgeKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_moist_stage_bridge() -> CompiledMoistStageBridgeKernels:
    digest = hashlib.sha256(CUDA_MOIST_STAGE_BRIDGE_SOURCE_BYTES).hexdigest()
    if digest != CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256:
        raise RuntimeError("packaged moist-stage bridge CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        options += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_MOIST_STAGE_BRIDGE_SOURCE, "cm1_moist_stage_bridge_v1_ptx.cu", method="ptx"
    ).compile(options + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_MOIST_STAGE_BRIDGE_SOURCE, "cm1_moist_stage_bridge_v1_cubin.cu", method="cubin"
    ).compile(options + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in CUDA_MOIST_STAGE_BRIDGE_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks moist-stage bridge entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled moist-stage bridge image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_MOIST_STAGE_BRIDGE_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", b"") if isinstance(props, dict) else b""
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(int(cupy.cuda.runtime.driverGetVersion()))),
        ("cuda_runtime_version", str(int(cupy.cuda.runtime.runtimeGetVersion()))),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledMoistStageBridgeKernels(module, kernels, ptx, cubin, arch, identity)


def _positive_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _spacing(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _cuda_f32(cupy: Any, value: Any, name: str, shapes: tuple[tuple[int, ...], ...], *, writable: bool = False) -> Any:
    if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must be a CUDA float32 array")
    if tuple(value.shape) not in shapes or not value.flags.c_contiguous:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected one of {shapes}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


@dataclass(slots=True)
class CudaMoistStageBridgeState:
    ni: int
    nj: int
    nk: int
    base3d: bool
    dum8: Any
    thterm: Any
    divergence: Any
    theta_divergence_term: Any
    divx: Any
    stream: Any
    device_id: int
    compiled: CompiledMoistStageBridgeKernels
    open_compiled: Any
    #: base.F's qc0/qi0 as the last stage read them: exact zero unless a
    #: caller passed a base state that carries condensate (receipt only).
    base_condensate: str = "exact-zero"
    #: Exact-zero base profile (or 3-D base) standing in for an absent qc0/qi0.
    zero_base: Any = None
    terms_ready: bool = False
    buoyancy_applied: bool = False
    post_acoustic_ready: bool = False


def create_cuda_moist_stage_bridge_state(
    *, ni: int, nj: int, nk: int, base3d: bool, stream: Any | None = None,
    compiled: CompiledMoistStageBridgeKernels | None = None, open_compiled: Any | None = None,
) -> CudaMoistStageBridgeState:
    ni, nj, nk = (_positive_int("ni", ni, 7), _positive_int("nj", nj, 7), _positive_int("nk", nk, 8))
    if type(base3d) is not bool:
        raise TypeError("base3d must be boolean")
    cupy = importlib.import_module("cupy")
    if open_compiled is None:
        from cm1core.cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary
        open_compiled = compile_cuda_sound3_open_boundary()
    compiled = compiled or compile_cuda_moist_stage_bridge()
    if compiled.target_arch != open_compiled.target_arch:
        raise ValueError("moist-stage and open-boundary images target different architectures")
    stream = stream or cupy.cuda.get_current_stream()
    physical = (nk, nj, ni)
    state = CudaMoistStageBridgeState(
        ni, nj, nk, base3d,
        cupy.zeros((nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY), dtype=cupy.float32),
        cupy.empty(physical, dtype=cupy.float32), cupy.empty(physical, dtype=cupy.float32),
        cupy.empty(physical, dtype=cupy.float32), cupy.empty(physical, dtype=cupy.float32),
        stream, int(cupy.cuda.Device().id), compiled, open_compiled,
    )
    state.zero_base = cupy.zeros(physical if base3d else (nk,), dtype=cupy.float32)
    return state


def _validate_common(state: CudaMoistStageBridgeState, arrays: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if not isinstance(state, CudaMoistStageBridgeState):
        raise TypeError("state must be CudaMoistStageBridgeState")
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.Device().id) != state.device_id:
        raise ValueError("moist-stage bridge is bound to another CUDA device")
    ni, nj, nk, ng = state.ni, state.nj, state.nk, CM1_OPEN_NGXY
    shapes = {
        "thv": (nk + 2, nj + 2, ni + 2), "ppterm": (nk + 2, nj + 2, ni + 2),
        "piadv": (nk + 2, nj + 2, ni + 2), "pp3d": (nk + 2, nj + 2, ni + 2),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "wten": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    a = {name: _cuda_f32(cupy, arrays[name], name, (shape,), writable=True) for name, shape in shapes.items()}
    metric_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    names = ("c1", "c2", "rho0_3d", "rf0", "gz", "rgzu", "rgzv", "dzdx", "dzdy", "gxu", "gyv")
    m = {name: _cuda_f32(cupy, metrics[name], name, (metric_shapes[name],)) for name in names}
    return cupy, a, m


def launch_cuda_eqtset2_stage_terms(
    state: CudaMoistStageBridgeState, arrays: Mapping[str, Any], metrics: Mapping[str, Any],
    *, th0: Any, qv0: Any, pi0: Any, th3d: Any, q3d: Any, stream: Any | None = None,
    qc0: Any | None = None, qi0: Any | None = None,
) -> Any:
    """Produce current-stage moist terms and fill source-required open halos.

    ``qc0``/``qi0`` are base.F's base-state cloud water and ice (``solve2.F:404``
    subtracts them from the live condensate); ``None`` is CM1's zero.
    """
    cupy, a, _ = _validate_common(state, arrays, metrics)
    if state.terms_ready or state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError("begin the next moist stage before recomputing terms")
    stream = stream or state.stream
    if int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("moist-stage bridge must remain on its bound stream")
    physical = (state.nk, state.nj, state.ni)
    base_shape = physical if state.base3d else (state.nk,)
    th0 = _cuda_f32(cupy, th0, "th0", (base_shape,))
    qv0 = _cuda_f32(cupy, qv0, "qv0", (base_shape,))
    pi0 = _cuda_f32(cupy, pi0, "pi0", (base_shape,))
    zero = state.zero_base
    if zero is None and (qc0 is None or qi0 is None):
        # A state built field by field (not by create_...) gets its zero base
        # on first use, on its own stream.
        with state.stream:
            zero = cupy.zeros(base_shape, dtype=cupy.float32)
        state.zero_base = zero
    qc0 = zero if qc0 is None else _cuda_f32(cupy, qc0, "qc0", (base_shape,))
    qi0 = zero if qi0 is None else _cuda_f32(cupy, qi0, "qi0", (base_shape,))
    state.base_condensate = "exact-zero" if (qc0 is zero and qi0 is zero) else "base.F qc0/qi0"

    th3d = _cuda_f32(cupy, th3d, "th3d", (physical,))
    q3d = _cuda_f32(cupy, q3d, "q3d", ((6, *physical),))
    n = state.ni * state.nj * state.nk
    pwords = state.nk * (state.nj + 2) * (state.ni + 2)
    grid = ((max(n, pwords) + CUDA_MOIST_STAGE_BRIDGE_THREADS - 1) // CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    block = (CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    u32 = cupy.uint32
    terms, piadv = state.compiled.kernels[:2]
    kpwe, kpsn = state.open_compiled.kernels[3:5]
    kfwe, kfsn = state.open_compiled.kernels[5:7]
    with stream:
        terms(grid, block, (th0, qv0, qc0, qi0, pi0, th3d, q3d, a["pp3d"], a["thv"], a["ppterm"], state.dum8, state.thterm, u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), u32(1 if state.base3d else 0)), stream=stream)
        piadv(grid, block, (pi0, a["pp3d"], a["piadv"], u32(state.ni), u32(state.nj), u32(state.nk), u32(1 if state.base3d else 0)), stream=stream)
        kpwe(grid, block, (a["thv"], u32(state.ni), u32(state.nj), u32(state.nk)), stream=stream)
        kpsn(grid, block, (a["thv"], u32(state.ni), u32(state.nj), u32(state.nk)), stream=stream)
        kfwe(grid, block, (state.dum8, u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
        kfsn(grid, block, (state.dum8, u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.terms_ready = True
    return event


def launch_cuda_moist_buoyancy(state: CudaMoistStageBridgeState, arrays: Mapping[str, Any], metrics: Mapping[str, Any], *, stream: Any | None = None) -> Any:
    """Accumulate vertical and terrain-horizontal buoyancy into live tendencies."""
    cupy, a, m = _validate_common(state, arrays, metrics)
    if not state.terms_ready or state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError("buoyancy requires one fresh current-stage term production")
    stream = stream or state.stream
    total = (state.nk - 1) * state.nj * state.ni + state.nk * state.nj * (state.ni - 1) + state.nk * (state.nj - 1) * state.ni
    grid = ((total + CUDA_MOIST_STAGE_BRIDGE_THREADS - 1) // CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    with stream:
        state.compiled.kernels[2](grid, (CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1), (state.dum8, m["c1"], m["c2"], m["gxu"], m["gyv"], a["uten"], a["vten"], a["wten"], cupy.uint32(state.ni), cupy.uint32(state.nj), cupy.uint32(state.nk), cupy.uint32(CM1_OPEN_NGXY)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.buoyancy_applied = True
    return event


@dataclass(frozen=True, slots=True)
class CudaPostAcousticFluxLaunch:
    state: CudaMoistStageBridgeState
    ready_event: Any
    receipt: dict[str, Any]


def launch_cuda_post_acoustic_flux_bridge(
    state: CudaMoistStageBridgeState, arrays: Mapping[str, Any], metrics: Mapping[str, Any],
    *, dx: float, dy: float, dz: float, stream: Any | None = None,
) -> CudaPostAcousticFluxLaunch:
    """Run current-stage getdiv, then destructively rebuild getdivx mass flux."""
    cupy, a, m = _validate_common(state, arrays, metrics)
    if not state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError("post-acoustic bridge requires applied current-stage buoyancy")
    dx, dy, dz = (_spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dz", dz)))
    stream = stream or state.stream
    n = state.ni * state.nj * state.nk
    horizontal = state.nk * state.nj * (state.ni + 1) + state.nk * (state.nj + 1) * state.ni
    vertical = (state.nk + 1) * state.nj * state.ni
    block = (CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    grids = tuple(((count + CUDA_MOIST_STAGE_BRIDGE_THREADS - 1) // CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1) for count in (n, horizontal, vertical))
    f32, u32 = cupy.float32, cupy.uint32
    getdiv, horizontal_kernel, vertical_kernel, divx_kernel = state.compiled.kernels[3:]
    with stream:
        getdiv(grids[0], block, (a["rru"], a["rrv"], a["rrw"], m["rgzu"], m["rgzv"], m["dzdx"], m["dzdy"], m["gz"], state.thterm, state.divergence, state.theta_divergence_term, u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), f32(dx), f32(dy), f32(dz), f32(1.0e-20)), stream=stream)
        horizontal_kernel(grids[1], block, (m["rho0_3d"], m["rgzu"], m["rgzv"], a["rru"], a["rrv"], u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY)), stream=stream)
        vertical_kernel(grids[2], block, (m["rf0"], m["dzdx"], m["dzdy"], m["gz"], a["rru"], a["rrv"], a["rrw"], u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), f32(dz)), stream=stream)
        divx_kernel(grids[0], block, (a["rru"], a["rrv"], a["rrw"], state.divx, u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), f32(dx), f32(dy), f32(dz), f32(1.0e-20)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.post_acoustic_ready = True
    receipt = {
        "schema": CUDA_MOIST_STAGE_BRIDGE_SCHEMA,
        "abi": CUDA_MOIST_STAGE_BRIDGE_ABI,
        "source_sha256": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
        "authority": {"tarball_sha256": AUTHORITY_TARBALL_SHA256, "source_sha256": dict(AUTHORITY_SOURCE_SHA256), "anchors": dict(SOURCE_ANCHORS)},
        "selection": {"eqtset": 2, "imoist": 1, "iice": 1, "base_condensate": state.base_condensate, "terrain": True, "axisymmetric": 0, "uniform_horizontal": True, "uniform_sigma": True, "lateral_boundaries": [2, 2, 2, 2], "base3d": state.base3d},
        "domain": {"ni": state.ni, "nj": state.nj, "nk": state.nk},
        "ordering": ["getdiv-on-acoustic-average-velocity", "getdivx-horizontal-mass-flux-in-place", "getdivx-terrain-vertical-mass-flux-in-place", "getdivx-divergence"],
        "target_arch": state.compiled.target_arch,
        "runtime": dict(state.compiled.runtime_identity),
        "claim_limit": CUDA_MOIST_STAGE_BRIDGE_CLAIM_LIMIT,
    }
    return CudaPostAcousticFluxLaunch(state, event, receipt)


def begin_cuda_moist_stage(state: CudaMoistStageBridgeState) -> None:
    """Advance the cadence guard after a completed post-acoustic bridge."""
    if not isinstance(state, CudaMoistStageBridgeState):
        raise TypeError("state must be CudaMoistStageBridgeState")
    if not state.post_acoustic_ready:
        raise ValueError("cannot begin another stage before post-acoustic getdiv/getdivx")
    state.terms_ready = False
    state.buoyancy_applied = False
    state.post_acoustic_ready = False


__all__ = [
    "AUTHORITY_SOURCE_SHA256", "AUTHORITY_TARBALL_SHA256", "CUDA_MOIST_STAGE_BRIDGE_ABI",
    "CUDA_MOIST_STAGE_BRIDGE_CLAIM_LIMIT", "CUDA_MOIST_STAGE_BRIDGE_SCHEMA",
    "CUDA_MOIST_STAGE_BRIDGE_SOURCE", "CUDA_MOIST_STAGE_BRIDGE_SOURCE_BYTES",
    "CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256", "CompiledMoistStageBridgeKernels",
    "CudaMoistStageBridgeState", "CudaPostAcousticFluxLaunch", "SOURCE_ANCHORS",
    "begin_cuda_moist_stage", "compile_cuda_moist_stage_bridge", "cpu_eqtset2_cell_f32",
    "create_cuda_moist_stage_bridge_state", "launch_cuda_eqtset2_stage_terms",
    "launch_cuda_moist_buoyancy", "launch_cuda_post_acoustic_flux_bridge",
]
