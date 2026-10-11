"""Bounded periodic flat-grid CUDA slice of CM1 r21.1 ``psolver=3``.

This module extends the independently ratified column specialization with the
source-ordered horizontal pressure-gradient updates for staggered ``u`` and
``v``, the horizontal divergence and pressure-advection contributions to
``fpk``, periodic pressure fills between acoustic small steps, and the
periodic/default RK-stage boundary cadence.  It is an operator oracle and an
executable layout seam, not a complete CM1 step or a production solver.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32


CUDA_SOUND3_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-sound3-periodic/v1"
CUDA_SOUND3_PERIODIC_ABI = "gpuwm-cm1.cuda-sound3-periodic-kernels/v1"
CUDA_SOUND3_PERIODIC_THREADS = 128
CM1_SOUND3_PERIODIC_ALPH_F32 = 0.60
CM1_SOUND3_PERIODIC_KDIV_F32 = 0.10
CM1_SOUND3_PERIODIC_CP_F32 = 1005.7
CM1_SOUND3_PERIODIC_SMEPS_F32 = 1.0e-30
CUDA_SOUND3_PERIODIC_CLAIM_LIMIT = (
    "CM1 r21.1 flat Cartesian serial-periodic psolver=3 acoustic/RK slice: "
    "horizontal pressure-gradient u/v updates, horizontal divergence and "
    "piadv contributions to fpk, vertically implicit column solve, pressure "
    "filter/update, native small-step pressure fills, and RK-stage periodic "
    "boundary cadence; held tendencies and prescribed base coefficients only; "
    "no terrain, MPI, open/radiative boundary, moisture/eqtset2 closure, "
    "complete solve2/solve3, dry-dynamics composition, forecast, or performance "
    "claim"
)

SOURCE_ANCHORS = {
    "solve2_dispatch": "src/solve2.F:1119-1138",
    "rk_small_step_schedule": "src/sound.F:81-113",
    "flat_vertical_coefficients": "src/sound.F:117-142",
    "stage_preparation": "src/sound.F:191-248",
    "horizontal_pressure_gradient": "src/sound.F:317-360",
    "flat_horizontal_fpk": "src/sound.F:487-509",
    "tridiagonal_solve": "src/sound.F:659-730",
    "pressure_update_and_fill": "src/sound.F:739-803",
    "velocity_time_average": "src/sound.F:806-834",
    "final_stage_ppx": "src/sound.F:842-851",
    "periodic_pressure_boundary": "src/bc.F:372-520",
    "periodic_velocity_boundary": "src/bc.F:528-1014",
    "rk_stage_boundary_calls": "src/solve2.F:1665-1691",
    "post_step_ppx_completion": "src/solve3.F:278-305",
    "serial_periodic_boundary_identity": "src/param.F:6546-6681",
    "default_face_timestep_weights": "src/param.F:9180-9221",
}

SCALAR_NAMES = (
    "mh",
    "rr0",
    "rf0",
    "c1",
    "c2",
    "mf",
    "piadv",
    "ppterm",
    "thv",
    "ppten",
    "ppi",
    "pp3d",
    "ppx",
    "ppd",
    "qk",
    "pk1",
    "pk2",
    "mm",
    "fpk",
    "divh",
    "hadvp",
    "elim_c",
    "elim_d",
    "tri_aa",
    "tri_bb",
    "tri_cc",
    "tri_dd",
    "tri_pivot",
    "tri_residual",
)
W_NAMES = ("wten", "wa", "w3d", "rrw")
U_NAMES = ("ua", "u3d", "uten", "rru")
V_NAMES = ("va", "v3d", "vten", "rrv")
METRIC_NAMES = ("dtu0", "dtv0", "uf", "vf", "uh", "vh")
ALL_ARRAY_NAMES = SCALAR_NAMES + W_NAMES + U_NAMES + V_NAMES + METRIC_NAMES
WRITABLE_NAMES = (
    "pp3d",
    "ppx",
    "ppd",
    "qk",
    "pk1",
    "pk2",
    "mm",
    "fpk",
    "divh",
    "hadvp",
    "elim_c",
    "elim_d",
    "tri_aa",
    "tri_bb",
    "tri_cc",
    "tri_dd",
    "tri_pivot",
    "tri_residual",
    "w3d",
    "rrw",
    "u3d",
    "rru",
    "v3d",
    "rrv",
)
COMPARE_OUTPUT_NAMES = (
    "u3d",
    "v3d",
    "w3d",
    "pp3d",
    "ppx",
    "ppd",
    "rru",
    "rrv",
    "rrw",
    "qk",
    "pk1",
    "pk2",
    "mm",
    "fpk",
    "divh",
    "hadvp",
    "elim_c",
    "elim_d",
    "tri_aa",
    "tri_bb",
    "tri_cc",
    "tri_dd",
    "tri_pivot",
    "tri_residual",
)


CUDA_SOUND3_PERIODIC_SOURCE = r"""// gpuwm-cm1.cuda-sound3-periodic-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float a32(const float a, const float b) { return __fadd_rn(a,b); }
__device__ __forceinline__ float s32(const float a, const float b) { return __fsub_rn(a,b); }
__device__ __forceinline__ float m32(const float a, const float b) { return __fmul_rn(a,b); }
__device__ __forceinline__ float d32(const float a, const float b) { return __fdiv_rn(a,b); }

__device__ __forceinline__ unsigned long long si(
    unsigned int k, unsigned int j, unsigned int i,
    unsigned int ni, unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj+2u)
          + (unsigned long long)j) * (unsigned long long)(ni+2u)
          + (unsigned long long)i;
}
__device__ __forceinline__ unsigned long long wi(
    unsigned int k, unsigned int j, unsigned int i,
    unsigned int ni, unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)nj
          + (unsigned long long)j) * (unsigned long long)ni
          + (unsigned long long)i;
}
__device__ __forceinline__ unsigned long long ui(
    unsigned int k, unsigned int j, unsigned int f,
    unsigned int ni, unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj+2u)
          + (unsigned long long)j) * (unsigned long long)(ni+1u)
          + (unsigned long long)f;
}
__device__ __forceinline__ unsigned long long vi(
    unsigned int k, unsigned int f, unsigned int i,
    unsigned int ni, unsigned int nj)
{
    return ((unsigned long long)k * (unsigned long long)(nj+1u)
          + (unsigned long long)f) * (unsigned long long)(ni+2u)
          + (unsigned long long)i;
}

extern "C" __global__
void cm1_sound3_periodic_prepare_scalar_v1(
    const float* mh, const float* rr0, const float* rf0,
    const float* c1, const float* c2, const float* mf,
    const float* piadv, const float* ppterm, const float* thv,
    const float* ppi, const float* ppx, const float* wa,
    float* w3d, float* pp3d, float* ppd, float* rrw,
    float* qk, float* pk1, float* pk2, float* mm,
    float* fpk, float* divh, float* hadvp,
    float* elim_c, float* elim_d,
    float* tri_aa, float* tri_bb, float* tri_cc, float* tri_dd,
    float* tri_pivot, float* tri_residual,
    unsigned int ni, unsigned int nj, unsigned int nk,
    unsigned int nrk, float dts, float rdz, float cp, float beta)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long step =
        (unsigned long long)blockDim.x * gridDim.x;
    const unsigned long long scalar_words =
        (unsigned long long)(nk+2u)*(nj+2u)*(ni+2u);
    for (unsigned long long x=first; x<scalar_words; x+=step) {
        const unsigned int i = (unsigned int)(x % (ni+2u));
        const unsigned long long q = x / (ni+2u);
        const unsigned int j = (unsigned int)(q % (nj+2u));
        const unsigned int k = (unsigned int)(q / (nj+2u));
        if (k>=1u && k<=nk) {
            ppd[x] = ppx[x];
            if (nrk != 1u) pp3d[x] = ppi[x];
        }
        if (i>=1u && i<=ni && j>=1u && j<=nj) {
            fpk[x]=0.0f; divh[x]=0.0f; hadvp[x]=0.0f;
            elim_c[x]=0.0f; elim_d[x]=0.0f;
            tri_aa[x]=0.0f; tri_bb[x]=0.0f; tri_cc[x]=0.0f;
            tri_dd[x]=0.0f; tri_pivot[x]=0.0f; tri_residual[x]=0.0f;
            if (k==0u || k==nk+1u) { qk[x]=0.0f; pk1[x]=0.0f; pk2[x]=0.0f; mm[x]=0.0f; }
        }
    }
    const unsigned long long ncol=(unsigned long long)ni*nj;
    for (unsigned long long col=first; col<ncol; col+=step) {
        const unsigned int j=(unsigned int)(col/ni)+1u;
        const unsigned int i=(unsigned int)(col%ni)+1u;
        for (unsigned int k=1u;k<=nk;++k) {
            const unsigned long long x=si(k,j,i,ni,nj);
            const unsigned long long b=si(k-1u,j,i,ni,nj);
            const unsigned long long t=si(k+1u,j,i,ni,nj);
            const float tem=m32(m32(dts,rdz),mh[x]);
            float r2=m32(m32(dts,rdz),mh[x]); r2=m32(r2,rr0[x]); r2=m32(r2,rf0[t]);
            float r1=m32(m32(dts,rdz),mh[x]); r1=m32(r1,rr0[x]); r1=m32(r1,rf0[x]);
            const float tem1=m32(m32(m32(dts,rdz),cp),mf[x]);
            qk[x]=m32(tem,ppterm[x]);
            pk2[x]=m32(r2,a32(m32(-c2[t],piadv[t]),m32(s32(1.0f,c1[t]),piadv[x])));
            pk1[x]=m32(r1,a32(m32(c1[x],piadv[b]),m32(s32(c2[x],1.0f),piadv[x])));
            mm[x]=m32(tem1,a32(m32(c2[x],thv[x]),m32(c1[x],thv[b])));
        }
        pk1[si(1u,j,i,ni,nj)]=0.0f;
        pk2[si(nk,j,i,ni,nj)]=0.0f;
        for (unsigned int k=1u;k<=nk;++k) {
            const unsigned long long w=wi(k,j-1u,i-1u,ni,nj);
            rrw[w]=m32(wa[w],beta);
            if (nrk != 1u) w3d[w]=wa[w];
        }
    }
}

extern "C" __global__
void cm1_sound3_periodic_prepare_uv_v1(
    const float* ua, const float* va, float* u3d, float* v3d,
    float* rru, float* rrv, unsigned int ni, unsigned int nj,
    unsigned int nk, unsigned int nrk)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long nu=(unsigned long long)(nk+2u)*(nj+2u)*(ni+1u);
    const unsigned long long nv=(unsigned long long)(nk+2u)*(nj+1u)*(ni+2u);
    for (unsigned long long x=first;x<nu;x+=step) { rru[x]=0.0f; if(nrk!=1u) u3d[x]=ua[x]; }
    for (unsigned long long x=first;x<nv;x+=step) { rrv[x]=0.0f; if(nrk!=1u) v3d[x]=va[x]; }
}

extern "C" __global__
void cm1_sound3_periodic_update_uv_v1(
    const float* ppd, const float* thv,
    const float* uten, const float* vten,
    const float* dtu0, const float* dtv0,
    const float* uf, const float* vf,
    float* u3d, float* v3d,
    unsigned int ni, unsigned int nj, unsigned int nk,
    float dts, float rdx, float rdy, float cp)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long nu=(unsigned long long)nk*(nj+1u)*(ni+1u);
    const unsigned long long nv=(unsigned long long)nk*(nj+1u)*(ni+1u);
    const float tem1=m32(m32(rdx,cp),0.5f);
    const float tem2=m32(m32(rdy,cp),0.5f);
    for (unsigned long long n=first;n<nu;n+=step) {
        const unsigned int f=(unsigned int)(n%(ni+1u));
        const unsigned long long q=n/(ni+1u);
        const unsigned int j=(unsigned int)(q%(nj+1u))+1u;
        const unsigned int k=(unsigned int)(q/(nj+1u))+1u;
        const unsigned long long x=ui(k,j,f,ni,nj);
        const unsigned long long pr=si(k,j,f+1u,ni,nj), pl=si(k,j,f,ni,nj);
        float pg=m32(tem1,s32(ppd[pr],ppd[pl])); pg=m32(pg,uf[f+1u]); pg=m32(pg,a32(thv[pr],thv[pl]));
        const float tend=s32(uten[x],pg);
        const float weight=dtu0[(unsigned long long)j*(ni+1u)+f];
        u3d[x]=a32(u3d[x],m32(m32(dts,weight),tend));
    }
    for (unsigned long long n=first;n<nv;n+=step) {
        const unsigned int i=(unsigned int)(n%(ni+1u))+1u;
        const unsigned long long q=n/(ni+1u);
        const unsigned int f=(unsigned int)(q%(nj+1u));
        const unsigned int k=(unsigned int)(q/(nj+1u))+1u;
        const unsigned long long x=vi(k,f,i,ni,nj);
        const unsigned long long pn=si(k,f+1u,i,ni,nj), ps=si(k,f,i,ni,nj);
        float pg=m32(tem2,s32(ppd[pn],ppd[ps])); pg=m32(pg,vf[f+1u]); pg=m32(pg,a32(thv[pn],thv[ps]));
        const float tend=s32(vten[x],pg);
        const float weight=dtv0[(unsigned long long)f*(ni+2u)+i];
        v3d[x]=a32(v3d[x],m32(m32(dts,weight),tend));
    }
}

extern "C" __global__
void cm1_sound3_periodic_column_v1(
    const float* piadv, const float* ppterm, const float* ppten,
    const float* wten, const float* u3d, const float* v3d,
    const float* uh, const float* vh,
    float* w3d, float* pp3d, float* ppx, float* ppd, float* rrw,
    const float* qk, const float* pk1, const float* pk2, const float* mm,
    float* fpk, float* divh, float* hadvp,
    float* elim_c, float* elim_d,
    float* tri_aa, float* tri_bb, float* tri_cc, float* tri_dd,
    float* tri_pivot, float* tri_residual,
    unsigned int ni, unsigned int nj, unsigned int nk,
    unsigned int nrk, unsigned int n, unsigned int nloop,
    float dts, float rdx, float rdy, float alph, float beta,
    float asq, float kdiv, float smeps)
{
    const unsigned int first=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int step=blockDim.x*gridDim.x;
    const unsigned int ncol=ni*nj;
    const float temx=m32(m32(dts,0.5f),rdx);
    const float temy=m32(m32(dts,0.5f),rdy);
    for (unsigned int col=first;col<ncol;col+=step) {
        const unsigned int j=col/ni+1u, i=col%ni+1u;
        for (unsigned int k=1u;k<=nk;++k) {
            const unsigned long long x=si(k,j,i,ni,nj);
            const float du=m32(m32(s32(u3d[ui(k,j,i,ni,nj)],u3d[ui(k,j,i-1u,ni,nj)]),rdx),uh[i]);
            const float dv=m32(m32(s32(v3d[vi(k,j,i,ni,nj)],v3d[vi(k,j-1u,i,ni,nj)]),rdy),vh[j]);
            float div=a32(du,dv); if(fabsf(div)<smeps) div=0.0f; divh[x]=div;
            const float gxw=s32(piadv[x],piadv[si(k,j,i-1u,ni,nj)]);
            const float gxe=s32(piadv[si(k,j,i+1u,ni,nj)],piadv[x]);
            const float gys=s32(piadv[x],piadv[si(k,j-1u,i,ni,nj)]);
            const float gyn=s32(piadv[si(k,j+1u,i,ni,nj)],piadv[x]);
            float hx=a32(m32(u3d[ui(k,j,i-1u,ni,nj)],gxw),m32(u3d[ui(k,j,i,ni,nj)],gxe));
            hx=m32(m32(-hx,temx),uh[i]);
            float hy=a32(m32(v3d[vi(k,j-1u,i,ni,nj)],gys),m32(v3d[vi(k,j,i,ni,nj)],gyn));
            hy=m32(m32(-hy,temy),vh[j]);
            const float hp=a32(hx,hy); hadvp[x]=hp;
            const float src=m32(dts,s32(ppten[x],m32(ppterm[x],div)));
            const float vert=a32(m32(w3d[wi(k,j-1u,i-1u,ni,nj)],a32(pk1[x],qk[x])),
                                  m32(w3d[wi(k+1u,j-1u,i-1u,ni,nj)],s32(pk2[x],qk[x])));
            fpk[x]=a32(a32(src,hp),m32(beta,vert));
        }
        for (unsigned int k=2u;k<=nk;++k) {
            const unsigned long long x=si(k,j,i,ni,nj), b=si(k-1u,j,i,ni,nj);
            float aa=0.0f, cc=0.0f;
            if(k>=3u) aa=m32(-asq,m32(mm[x],a32(pk1[b],qk[b])));
            if(k<nk) cc=m32(-asq,m32(-mm[x],s32(pk2[x],qk[x])));
            const float diag=a32(-a32(pk1[x],qk[x]),s32(pk2[b],qk[b]));
            const float bb=s32(1.0f,m32(asq,m32(mm[x],diag)));
            const float pd=s32(pp3d[x],pp3d[b]);
            const float fwk=a32(m32(dts,wten[wi(k,j-1u,i-1u,ni,nj)]),m32(beta,m32(-mm[x],pd)));
            const float upd=s32(a32(pp3d[x],fpk[x]),a32(pp3d[b],fpk[b]));
            const float dd=a32(a32(w3d[wi(k,j-1u,i-1u,ni,nj)],fwk),m32(alph,m32(-mm[x],upd)));
            float pivot=bb; if(k>=3u) pivot=a32(m32(aa,elim_c[b]),bb);
            const float inv=d32(1.0f,pivot);
            tri_aa[x]=aa; tri_bb[x]=bb; tri_cc[x]=cc; tri_dd[x]=dd; tri_pivot[x]=pivot;
            if(k<nk) { elim_c[x]=m32(-cc,inv); float num=dd; if(k>=3u) num=s32(dd,m32(aa,elim_d[b])); elim_d[x]=m32(num,inv); }
            else { w3d[wi(k,j-1u,i-1u,ni,nj)]=m32(s32(dd,m32(aa,elim_d[b])),inv); }
        }
        for (int k=(int)nk-1;k>=2;--k) {
            const unsigned long long x=si((unsigned int)k,j,i,ni,nj);
            w3d[wi((unsigned int)k,j-1u,i-1u,ni,nj)]=a32(
                m32(elim_c[x],w3d[wi((unsigned int)k+1u,j-1u,i-1u,ni,nj)]),elim_d[x]);
        }
        for (unsigned int k=2u;k<=nk;++k) {
            const unsigned long long x=si(k,j,i,ni,nj), b=si(k-1u,j,i,ni,nj), t=si(k+1u,j,i,ni,nj);
            float lhs=m32(tri_aa[x],w3d[wi(k-1u,j-1u,i-1u,ni,nj)]);
            lhs=a32(lhs,m32(tri_bb[x],w3d[wi(k,j-1u,i-1u,ni,nj)]));
            lhs=a32(lhs,m32(tri_cc[x],w3d[wi(k+1u,j-1u,i-1u,ni,nj)]));
            tri_residual[x]=s32(lhs,tri_dd[x]);
            (void)b; (void)t;
        }
        for (unsigned int k=1u;k<=nk;++k) {
            const unsigned long long x=si(k,j,i,ni,nj);
            const float old=pp3d[x];
            const float vert=a32(m32(w3d[wi(k,j-1u,i-1u,ni,nj)],a32(pk1[x],qk[x])),
                                  m32(w3d[wi(k+1u,j-1u,i-1u,ni,nj)],s32(pk2[x],qk[x])));
            float upd=a32(a32(old,fpk[x]),m32(alph,vert)); if(fabsf(upd)<smeps) upd=0.0f;
            pp3d[x]=upd; elim_c[x]=m32(kdiv,s32(upd,old)); ppd[x]=a32(upd,elim_c[x]);
            if(nrk==3u && n==nloop) ppx[x]=elim_c[x];
            const unsigned long long w=wi(k,j-1u,i-1u,ni,nj);
            if(n<nloop) rrw[w]=a32(rrw[w],w3d[w]);
            else rrw[w]=m32(a32(rrw[w],m32(w3d[w],alph)),d32(1.0f,(float)nloop));
        }
    }
}

extern "C" __global__
void cm1_sound3_periodic_average_uv_v1(
    const float* u3d, const float* v3d, float* rru, float* rrv,
    unsigned int ni, unsigned int nj, unsigned int nk,
    unsigned int n, unsigned int nloop)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long nu=(unsigned long long)nk*(nj+1u)*(ni+1u), nv=nu;
    const float tavg=d32(1.0f,(float)nloop);
    for(unsigned long long q=first;q<nu;q+=step){ unsigned int f=(unsigned int)(q%(ni+1u)); unsigned long long z=q/(ni+1u); unsigned int j=(unsigned int)(z%(nj+1u))+1u; unsigned int k=(unsigned int)(z/(nj+1u))+1u; unsigned long long x=ui(k,j,f,ni,nj); rru[x]=(n<nloop)?a32(rru[x],u3d[x]):m32(a32(rru[x],u3d[x]),tavg); }
    for(unsigned long long q=first;q<nv;q+=step){ unsigned int i=(unsigned int)(q%(ni+1u))+1u; unsigned long long z=q/(ni+1u); unsigned int f=(unsigned int)(z%(nj+1u)); unsigned int k=(unsigned int)(z/(nj+1u))+1u; unsigned long long x=vi(k,f,i,ni,nj); rrv[x]=(n<nloop)?a32(rrv[x],v3d[x]):m32(a32(rrv[x],v3d[x]),tavg); }
}

extern "C" __global__
void cm1_sound3_periodic_bcp_v1(float* p, unsigned int ni, unsigned int nj, unsigned int nk)
{
    if(blockIdx.x||threadIdx.x) return;
    for(unsigned int k=1u;k<=nk;++k){
        for(unsigned int j=0u;j<=nj+1u;++j){p[si(k,j,0u,ni,nj)]=p[si(k,j,ni,ni,nj)]; p[si(k,j,ni+1u,ni,nj)]=p[si(k,j,1u,ni,nj)];}
        for(unsigned int i=0u;i<=ni+1u;++i){p[si(k,0u,i,ni,nj)]=p[si(k,nj,i,ni,nj)]; p[si(k,nj+1u,i,ni,nj)]=p[si(k,1u,i,ni,nj)];}
    }
}

extern "C" __global__
void cm1_sound3_periodic_bcuv_v1(float* u, float* v, unsigned int ni, unsigned int nj, unsigned int nk)
{
    if(blockIdx.x||threadIdx.x) return;
    for(unsigned int k=1u;k<=nk;++k){
        for(unsigned int f=0u;f<=ni;++f){u[ui(k,0u,f,ni,nj)]=u[ui(k,nj,f,ni,nj)]; u[ui(k,nj+1u,f,ni,nj)]=u[ui(k,1u,f,ni,nj)];}
        for(unsigned int f=0u;f<=nj;++f){v[vi(k,f,0u,ni,nj)]=v[vi(k,f,ni,ni,nj)]; v[vi(k,f,ni+1u,ni,nj)]=v[vi(k,f,1u,ni,nj)];}
    }
}
"""

CUDA_SOUND3_PERIODIC_SOURCE_BYTES = CUDA_SOUND3_PERIODIC_SOURCE.encode("utf-8")
CUDA_SOUND3_PERIODIC_SOURCE_SHA256 = (
    "f766c90b9735008e86753e4731fda106b85bd198af13a536e378753537f93a2b"
)
CUDA_SOUND3_PERIODIC_KERNEL_NAMES = (
    "cm1_sound3_periodic_prepare_scalar_v1",
    "cm1_sound3_periodic_prepare_uv_v1",
    "cm1_sound3_periodic_update_uv_v1",
    "cm1_sound3_periodic_column_v1",
    "cm1_sound3_periodic_average_uv_v1",
    "cm1_sound3_periodic_bcp_v1",
    "cm1_sound3_periodic_bcuv_v1",
)


def scalar_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 2, ni + 2


def w_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj, ni


def u_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 2, ni + 1


def v_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    _validate_dimensions(ni, nj, nk)
    return nk + 2, nj + 1, ni + 2


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value, lower in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")


def array_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    scalar = scalar_shape_zyx(ni, nj, nk)
    shapes = {name: scalar for name in SCALAR_NAMES}
    shapes.update({name: w_shape_zyx(ni, nj, nk) for name in W_NAMES})
    shapes.update({name: u_shape_zyx(ni, nj, nk) for name in U_NAMES})
    shapes.update({name: v_shape_zyx(ni, nj, nk) for name in V_NAMES})
    shapes.update(
        {
            "dtu0": (nj + 2, ni + 1),
            "dtv0": (nj + 1, ni + 2),
            "uf": (ni + 2,),
            "vf": (nj + 2,),
            "uh": (ni + 2,),
            "vh": (nj + 2,),
        }
    )
    return shapes


def default_periodic_face_weights(ni: int, nj: int) -> tuple[Any, Any]:
    _validate_dimensions(ni, nj, 3)
    np = importlib.import_module("numpy")
    dtu0 = np.zeros((nj + 2, ni + 1), dtype=np.float32)
    dtv0 = np.zeros((nj + 1, ni + 2), dtype=np.float32)
    # param.F first fills every live face with one.  The ib* flags that zero
    # normal boundary faces remain zero for serial periodic boundaries.
    dtu0[1 : nj + 1, :] = np.float32(1.0)
    dtv0[:, 1 : ni + 1] = np.float32(1.0)
    return dtu0, dtv0


def fill_periodic_scalar_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
        raise TypeError("periodic scalar must be a float32 numpy.ndarray")
    if value.shape != scalar_shape_zyx(ni, nj, nk) or not value.flags.c_contiguous:
        raise ValueError("periodic scalar has an invalid shape or layout")
    for k in range(1, nk + 1):
        value[k, :, 0] = value[k, :, ni]
        value[k, :, ni + 1] = value[k, :, 1]
        value[k, 0, :] = value[k, nj, :]
        value[k, nj + 1, :] = value[k, 1, :]


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


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing periodic sound3 arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(
                f"{name} shape must be {shapes[name]}, received {value.shape}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name in WRITABLE_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("periodic sound3 arrays must not alias")
    return result


def _bits_equal(left: Any, right: Any) -> bool:
    np = importlib.import_module("numpy")
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def validate_periodic_contract(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    a = _require_host_arrays(arrays, ni, nj, nk)
    expected_u, expected_v = default_periodic_face_weights(ni, nj)
    if not _bits_equal(a["dtu0"], expected_u) or not _bits_equal(a["dtv0"], expected_v):
        raise ValueError(
            "dtu0/dtv0 differ from CM1 serial-periodic default face weights"
        )
    for name in ("uf", "vf", "uh", "vh"):
        if not _bits_equal(
            a[name], importlib.import_module("numpy").ones_like(a[name])
        ):
            raise ValueError(
                f"{name} must be exactly one for this uniform flat-grid slice"
            )
    for name in ("piadv", "thv", "ppi", "ppx"):
        value = a[name]
        if not _bits_equal(value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]):
            raise ValueError(f"{name} west periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]):
            raise ValueError(f"{name} east periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]):
            raise ValueError(f"{name} south periodic halo is invalid")
        if not _bits_equal(value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]):
            raise ValueError(f"{name} north periodic halo is invalid")
    if not _bits_equal(
        a["ua"][1 : nk + 1, 1 : nj + 1, 0], a["ua"][1 : nk + 1, 1 : nj + 1, ni]
    ):
        raise ValueError("ua duplicate periodic x faces differ")
    if not _bits_equal(
        a["va"][1 : nk + 1, 0, 1 : ni + 1], a["va"][1 : nk + 1, nj, 1 : ni + 1]
    ):
        raise ValueError("va duplicate periodic y faces differ")


def compare_sound3_periodic_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every declared output in strict raw-f32/no-null mode."""

    np = importlib.import_module("numpy")
    missing = [
        name
        for name in COMPARE_OUTPUT_NAMES
        if name not in expected or name not in candidate
    ]
    if missing:
        raise ValueError(f"missing comparison outputs: {', '.join(missing)}")
    fields: dict[str, Any] = {}
    for name in COMPARE_OUTPUT_NAMES:
        left = expected[name]
        right = candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison values must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32"):
            raise TypeError(f"{name} comparison values must be float32")
        if left.shape != right.shape:
            raise ValueError(f"{name} comparison shapes differ")
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


def _add(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _sub(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _mul(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


def _div(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) / np.float32(y))


@dataclass(frozen=True, slots=True)
class CpuSound3PeriodicStageDiagnostics:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    ppd_boundary_fills: int
    stage_velocity_boundary_fills: int
    stage_pressure_boundary_fills: int
    ppx_written: bool
    boundary_trace_sha256: str


def cpu_sound3_periodic_stage_inplace(
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
    cp: float = CM1_SOUND3_PERIODIC_CP_F32,
    alph: float = CM1_SOUND3_PERIODIC_ALPH_F32,
    kdiv: float = CM1_SOUND3_PERIODIC_KDIV_F32,
    smeps: float = CM1_SOUND3_PERIODIC_SMEPS_F32,
) -> CpuSound3PeriodicStageDiagnostics:
    """Execute one independent source-ordered CPU RK/acoustic stage."""

    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts = np.float32(dts_value)
    rdx = np.float32(rdx)
    rdy = np.float32(rdy)
    rdz = np.float32(rdz)
    cp = np.float32(cp)
    alph = np.float32(alph)
    beta = _sub(np, np.float32(1.0), alph)
    asq = _mul(np, alph, alph)
    kdiv = np.float32(kdiv)
    smeps = np.float32(smeps)
    for name in (
        "qk",
        "pk1",
        "pk2",
        "mm",
        "fpk",
        "divh",
        "hadvp",
        "elim_c",
        "elim_d",
        "tri_aa",
        "tri_bb",
        "tri_cc",
        "tri_dd",
        "tri_pivot",
        "tri_residual",
    ):
        a[name].fill(np.float32(0.0))
    a["ppd"][1 : nk + 1] = a["ppx"][1 : nk + 1]
    a["rru"].fill(np.float32(0.0))
    a["rrv"].fill(np.float32(0.0))
    if nrk != 1:
        np.copyto(a["u3d"], a["ua"])
        np.copyto(a["v3d"], a["va"])
        a["pp3d"][1 : nk + 1] = a["ppi"][1 : nk + 1]
    for j0 in range(nj):
        j = j0 + 1
        for i0 in range(ni):
            i = i0 + 1
            for k in range(1, nk + 1):
                tem = _mul(np, _mul(np, dts, rdz), a["mh"][k, j, i])
                r2 = _mul(np, _mul(np, dts, rdz), a["mh"][k, j, i])
                r2 = _mul(np, r2, a["rr0"][k, j, i])
                r2 = _mul(np, r2, a["rf0"][k + 1, j, i])
                r1 = _mul(np, _mul(np, dts, rdz), a["mh"][k, j, i])
                r1 = _mul(np, r1, a["rr0"][k, j, i])
                r1 = _mul(np, r1, a["rf0"][k, j, i])
                tem1 = _mul(np, _mul(np, _mul(np, dts, rdz), cp), a["mf"][k, j, i])
                a["qk"][k, j, i] = _mul(np, tem, a["ppterm"][k, j, i])
                a["pk2"][k, j, i] = _mul(
                    np,
                    r2,
                    _add(
                        np,
                        _mul(np, -a["c2"][k + 1, j, i], a["piadv"][k + 1, j, i]),
                        _mul(
                            np,
                            _sub(np, np.float32(1.0), a["c1"][k + 1, j, i]),
                            a["piadv"][k, j, i],
                        ),
                    ),
                )
                a["pk1"][k, j, i] = _mul(
                    np,
                    r1,
                    _add(
                        np,
                        _mul(np, a["c1"][k, j, i], a["piadv"][k - 1, j, i]),
                        _mul(
                            np,
                            _sub(np, a["c2"][k, j, i], np.float32(1.0)),
                            a["piadv"][k, j, i],
                        ),
                    ),
                )
                a["mm"][k, j, i] = _mul(
                    np,
                    tem1,
                    _add(
                        np,
                        _mul(np, a["c2"][k, j, i], a["thv"][k, j, i]),
                        _mul(np, a["c1"][k, j, i], a["thv"][k - 1, j, i]),
                    ),
                )
                a["rrw"][k, j0, i0] = _mul(np, a["wa"][k, j0, i0], beta)
                if nrk != 1:
                    a["w3d"][k, j0, i0] = a["wa"][k, j0, i0]
            a["pk1"][1, j, i] = np.float32(0.0)
            a["pk2"][nk, j, i] = np.float32(0.0)
    tem1 = _mul(np, _mul(np, rdx, cp), np.float32(0.5))
    tem2 = _mul(np, _mul(np, rdy, cp), np.float32(0.5))
    temx = _mul(np, _mul(np, dts, np.float32(0.5)), rdx)
    temy = _mul(np, _mul(np, dts, np.float32(0.5)), rdy)
    boundary_trace = hashlib.sha256()
    for n in range(1, nloop + 1):
        for k in range(1, nk + 1):
            for j in range(1, nj + 2):
                for f in range(ni + 1):
                    pg = _mul(
                        np, tem1, _sub(np, a["ppd"][k, j, f + 1], a["ppd"][k, j, f])
                    )
                    pg = _mul(np, pg, a["uf"][f + 1])
                    pg = _mul(
                        np, pg, _add(np, a["thv"][k, j, f + 1], a["thv"][k, j, f])
                    )
                    a["u3d"][k, j, f] = _add(
                        np,
                        a["u3d"][k, j, f],
                        _mul(
                            np,
                            _mul(np, dts, a["dtu0"][j, f]),
                            _sub(np, a["uten"][k, j, f], pg),
                        ),
                    )
            for f in range(nj + 1):
                for i in range(1, ni + 2):
                    pg = _mul(
                        np, tem2, _sub(np, a["ppd"][k, f + 1, i], a["ppd"][k, f, i])
                    )
                    pg = _mul(np, pg, a["vf"][f + 1])
                    pg = _mul(
                        np, pg, _add(np, a["thv"][k, f + 1, i], a["thv"][k, f, i])
                    )
                    a["v3d"][k, f, i] = _add(
                        np,
                        a["v3d"][k, f, i],
                        _mul(
                            np,
                            _mul(np, dts, a["dtv0"][f, i]),
                            _sub(np, a["vten"][k, f, i], pg),
                        ),
                    )
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                for k in range(1, nk + 1):
                    du = _mul(
                        np,
                        _mul(
                            np, _sub(np, a["u3d"][k, j, i], a["u3d"][k, j, i - 1]), rdx
                        ),
                        a["uh"][i],
                    )
                    dv = _mul(
                        np,
                        _mul(
                            np, _sub(np, a["v3d"][k, j, i], a["v3d"][k, j - 1, i]), rdy
                        ),
                        a["vh"][j],
                    )
                    div = _add(np, du, dv)
                    if bool(np.abs(div) < smeps):
                        div = np.float32(0.0)
                    a["divh"][k, j, i] = div
                    gxw = _sub(np, a["piadv"][k, j, i], a["piadv"][k, j, i - 1])
                    gxe = _sub(np, a["piadv"][k, j, i + 1], a["piadv"][k, j, i])
                    gys = _sub(np, a["piadv"][k, j, i], a["piadv"][k, j - 1, i])
                    gyn = _sub(np, a["piadv"][k, j + 1, i], a["piadv"][k, j, i])
                    hx = _add(
                        np,
                        _mul(np, a["u3d"][k, j, i - 1], gxw),
                        _mul(np, a["u3d"][k, j, i], gxe),
                    )
                    hx = _mul(np, _mul(np, -hx, temx), a["uh"][i])
                    hy = _add(
                        np,
                        _mul(np, a["v3d"][k, j - 1, i], gys),
                        _mul(np, a["v3d"][k, j, i], gyn),
                    )
                    hy = _mul(np, _mul(np, -hy, temy), a["vh"][j])
                    hp = _add(np, hx, hy)
                    a["hadvp"][k, j, i] = hp
                    src = _mul(
                        np,
                        dts,
                        _sub(
                            np, a["ppten"][k, j, i], _mul(np, a["ppterm"][k, j, i], div)
                        ),
                    )
                    vert = _add(
                        np,
                        _mul(
                            np,
                            a["w3d"][k, j0, i0],
                            _add(np, a["pk1"][k, j, i], a["qk"][k, j, i]),
                        ),
                        _mul(
                            np,
                            a["w3d"][k + 1, j0, i0],
                            _sub(np, a["pk2"][k, j, i], a["qk"][k, j, i]),
                        ),
                    )
                    a["fpk"][k, j, i] = _add(
                        np, _add(np, src, hp), _mul(np, beta, vert)
                    )
                for k in range(2, nk + 1):
                    aa = np.float32(0.0)
                    cc = np.float32(0.0)
                    if k >= 3:
                        aa = _mul(
                            np,
                            -asq,
                            _mul(
                                np,
                                a["mm"][k, j, i],
                                _add(np, a["pk1"][k - 1, j, i], a["qk"][k - 1, j, i]),
                            ),
                        )
                    if k < nk:
                        cc = _mul(
                            np,
                            -asq,
                            _mul(
                                np,
                                -a["mm"][k, j, i],
                                _sub(np, a["pk2"][k, j, i], a["qk"][k, j, i]),
                            ),
                        )
                    diag = _add(
                        np,
                        -_add(np, a["pk1"][k, j, i], a["qk"][k, j, i]),
                        _sub(np, a["pk2"][k - 1, j, i], a["qk"][k - 1, j, i]),
                    )
                    bb = _sub(
                        np,
                        np.float32(1.0),
                        _mul(np, asq, _mul(np, a["mm"][k, j, i], diag)),
                    )
                    pd = _sub(np, a["pp3d"][k, j, i], a["pp3d"][k - 1, j, i])
                    fwk = _add(
                        np,
                        _mul(np, dts, a["wten"][k, j0, i0]),
                        _mul(np, beta, _mul(np, -a["mm"][k, j, i], pd)),
                    )
                    upd = _sub(
                        np,
                        _add(np, a["pp3d"][k, j, i], a["fpk"][k, j, i]),
                        _add(np, a["pp3d"][k - 1, j, i], a["fpk"][k - 1, j, i]),
                    )
                    dd = _add(
                        np,
                        _add(np, a["w3d"][k, j0, i0], fwk),
                        _mul(np, alph, _mul(np, -a["mm"][k, j, i], upd)),
                    )
                    pivot = (
                        bb
                        if k < 3
                        else _add(np, _mul(np, aa, a["elim_c"][k - 1, j, i]), bb)
                    )
                    inv = _div(np, np.float32(1.0), pivot)
                    for name, value in (
                        ("tri_aa", aa),
                        ("tri_bb", bb),
                        ("tri_cc", cc),
                        ("tri_dd", dd),
                        ("tri_pivot", pivot),
                    ):
                        a[name][k, j, i] = value
                    if k < nk:
                        a["elim_c"][k, j, i] = _mul(np, -cc, inv)
                        num = (
                            dd
                            if k < 3
                            else _sub(np, dd, _mul(np, aa, a["elim_d"][k - 1, j, i]))
                        )
                        a["elim_d"][k, j, i] = _mul(np, num, inv)
                    else:
                        a["w3d"][k, j0, i0] = _mul(
                            np,
                            _sub(np, dd, _mul(np, aa, a["elim_d"][k - 1, j, i])),
                            inv,
                        )
                for k in range(nk - 1, 1, -1):
                    a["w3d"][k, j0, i0] = _add(
                        np,
                        _mul(np, a["elim_c"][k, j, i], a["w3d"][k + 1, j0, i0]),
                        a["elim_d"][k, j, i],
                    )
                for k in range(2, nk + 1):
                    lhs = _mul(np, a["tri_aa"][k, j, i], a["w3d"][k - 1, j0, i0])
                    lhs = _add(
                        np, lhs, _mul(np, a["tri_bb"][k, j, i], a["w3d"][k, j0, i0])
                    )
                    lhs = _add(
                        np, lhs, _mul(np, a["tri_cc"][k, j, i], a["w3d"][k + 1, j0, i0])
                    )
                    a["tri_residual"][k, j, i] = _sub(np, lhs, a["tri_dd"][k, j, i])
                for k in range(1, nk + 1):
                    old = a["pp3d"][k, j, i]
                    vert = _add(
                        np,
                        _mul(
                            np,
                            a["w3d"][k, j0, i0],
                            _add(np, a["pk1"][k, j, i], a["qk"][k, j, i]),
                        ),
                        _mul(
                            np,
                            a["w3d"][k + 1, j0, i0],
                            _sub(np, a["pk2"][k, j, i], a["qk"][k, j, i]),
                        ),
                    )
                    upd = _add(
                        np, _add(np, old, a["fpk"][k, j, i]), _mul(np, alph, vert)
                    )
                    if bool(np.abs(upd) < smeps):
                        upd = np.float32(0.0)
                    a["pp3d"][k, j, i] = upd
                    a["elim_c"][k, j, i] = _mul(np, kdiv, _sub(np, upd, old))
                    a["ppd"][k, j, i] = _add(np, upd, a["elim_c"][k, j, i])
                    if nrk == 3 and n == nloop:
                        a["ppx"][k, j, i] = a["elim_c"][k, j, i]
                    if n < nloop:
                        a["rrw"][k, j0, i0] = _add(
                            np, a["rrw"][k, j0, i0], a["w3d"][k, j0, i0]
                        )
                    else:
                        a["rrw"][k, j0, i0] = _mul(
                            np,
                            _add(
                                np,
                                a["rrw"][k, j0, i0],
                                _mul(np, a["w3d"][k, j0, i0], alph),
                            ),
                            _div(np, np.float32(1.0), np.float32(nloop)),
                        )
        if n < nloop:
            fill_periodic_scalar_inplace(a["ppd"], ni=ni, nj=nj, nk=nk)
            boundary_trace.update(b"ppd\0" + n.to_bytes(4, "little"))
            boundary_trace.update(a["ppd"].tobytes(order="C"))
        tavg = _div(np, np.float32(1.0), np.float32(nloop))
        for k in range(1, nk + 1):
            for j in range(1, nj + 2):
                for f in range(ni + 1):
                    a["rru"][k, j, f] = (
                        _add(np, a["rru"][k, j, f], a["u3d"][k, j, f])
                        if n < nloop
                        else _mul(
                            np, _add(np, a["rru"][k, j, f], a["u3d"][k, j, f]), tavg
                        )
                    )
            for f in range(nj + 1):
                for i in range(1, ni + 2):
                    a["rrv"][k, f, i] = (
                        _add(np, a["rrv"][k, f, i], a["v3d"][k, f, i])
                        if n < nloop
                        else _mul(
                            np, _add(np, a["rrv"][k, f, i], a["v3d"][k, f, i]), tavg
                        )
                    )
    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    boundary_trace.update(b"uv-stage\0")
    boundary_trace.update(a["u3d"].tobytes(order="C"))
    boundary_trace.update(a["v3d"].tobytes(order="C"))
    if nrk < 3:
        fill_periodic_scalar_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
        boundary_trace.update(b"pp3d-stage\0")
        boundary_trace.update(a["pp3d"].tobytes(order="C"))
    return CpuSound3PeriodicStageDiagnostics(
        nrk,
        nloop,
        float(dts),
        rk_stage_dt_f32(dt, nrk),
        nloop - 1,
        1,
        1 if nrk < 3 else 0,
        nrk == 3,
        boundary_trace.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CpuSound3PeriodicResult:
    cadence: tuple[CpuSound3PeriodicStageDiagnostics, ...]


def cpu_sound3_periodic_three_stage_inplace(
    arrays: MutableMapping[str, Any], **kwargs: Any
) -> CpuSound3PeriodicResult:
    return CpuSound3PeriodicResult(
        tuple(
            cpu_sound3_periodic_stage_inplace(arrays, nrk=nrk, **kwargs)
            for nrk in (1, 2, 3)
        )
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
    return common + (f"--gpu-architecture=compute_{arch}",), common + (
        f"--gpu-architecture=sm_{arch}",
    )


@dataclass(frozen=True, slots=True)
class CompiledSound3PeriodicKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_sound3_periodic() -> CompiledSound3PeriodicKernels:
    if (
        hashlib.sha256(CUDA_SOUND3_PERIODIC_SOURCE_BYTES).hexdigest()
        != CUDA_SOUND3_PERIODIC_SOURCE_SHA256
    ):
        raise RuntimeError("packaged periodic sound3 CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if arch != compiler_arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_SOUND3_PERIODIC_SOURCE, "cm1_sound3_periodic_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_SOUND3_PERIODIC_SOURCE, "cm1_sound3_periodic_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_SOUND3_PERIODIC_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks periodic sound3 entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_SOUND3_PERIODIC_KERNEL_NAMES
    )
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    name = (
        name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
    )
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", name),
    )
    return CompiledSound3PeriodicKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing periodic sound3 arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if (
            value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(x.data.ptr) for x in result.values()}) != len(result):
        raise ValueError("periodic sound3 device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaSound3PeriodicLaunch:
    ready_event: Any
    compiled: CompiledSound3PeriodicKernels
    receipt: dict[str, Any]


def launch_cuda_sound3_periodic_three_stage(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    rdx: float,
    rdy: float,
    rdz: float,
    cp: float = CM1_SOUND3_PERIODIC_CP_F32,
    alph: float = CM1_SOUND3_PERIODIC_ALPH_F32,
    kdiv: float = CM1_SOUND3_PERIODIC_KDIV_F32,
    stream: Any | None = None,
    compiled: CompiledSound3PeriodicKernels | None = None,
) -> CudaSound3PeriodicLaunch:
    """Enqueue one three-stage periodic acoustic transition without synchronizing."""
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    compiled = compile_cuda_sound3_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    kprep, kprepuv, kupdate, kcolumn, kavg, kbcp, kbcuv = compiled.kernels
    f32 = cupy.float32
    u32 = cupy.uint32
    alph32 = f32(alph)
    beta = f32(f32(1.0) - alph32)
    asq = f32(alph32 * alph32)
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    face_words = max((nk + 2) * (nj + 2) * (ni + 1), (nk + 2) * (nj + 1) * (ni + 2))
    work_words = max(scalar_words, ni * nj)
    grid_prep = (
        (work_words + CUDA_SOUND3_PERIODIC_THREADS - 1) // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    grid_faces = (
        (face_words + CUDA_SOUND3_PERIODIC_THREADS - 1) // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    grid_cols = (
        (ni * nj + CUDA_SOUND3_PERIODIC_THREADS - 1) // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    cadence = []
    for nrk in (1, 2, 3):
        nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
        kprep(
            grid_prep,
            (CUDA_SOUND3_PERIODIC_THREADS, 1, 1),
            (
                a["mh"],
                a["rr0"],
                a["rf0"],
                a["c1"],
                a["c2"],
                a["mf"],
                a["piadv"],
                a["ppterm"],
                a["thv"],
                a["ppi"],
                a["ppx"],
                a["wa"],
                a["w3d"],
                a["pp3d"],
                a["ppd"],
                a["rrw"],
                a["qk"],
                a["pk1"],
                a["pk2"],
                a["mm"],
                a["fpk"],
                a["divh"],
                a["hadvp"],
                a["elim_c"],
                a["elim_d"],
                a["tri_aa"],
                a["tri_bb"],
                a["tri_cc"],
                a["tri_dd"],
                a["tri_pivot"],
                a["tri_residual"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(nrk),
                f32(dts),
                f32(rdz),
                f32(cp),
                beta,
            ),
            stream=stream,
        )
        kprepuv(
            grid_faces,
            (CUDA_SOUND3_PERIODIC_THREADS, 1, 1),
            (
                a["ua"],
                a["va"],
                a["u3d"],
                a["v3d"],
                a["rru"],
                a["rrv"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(nrk),
            ),
            stream=stream,
        )
        for n in range(1, nloop + 1):
            kupdate(
                grid_faces,
                (CUDA_SOUND3_PERIODIC_THREADS, 1, 1),
                (
                    a["ppd"],
                    a["thv"],
                    a["uten"],
                    a["vten"],
                    a["dtu0"],
                    a["dtv0"],
                    a["uf"],
                    a["vf"],
                    a["u3d"],
                    a["v3d"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    f32(dts),
                    f32(rdx),
                    f32(rdy),
                    f32(cp),
                ),
                stream=stream,
            )
            kcolumn(
                grid_cols,
                (CUDA_SOUND3_PERIODIC_THREADS, 1, 1),
                (
                    a["piadv"],
                    a["ppterm"],
                    a["ppten"],
                    a["wten"],
                    a["u3d"],
                    a["v3d"],
                    a["uh"],
                    a["vh"],
                    a["w3d"],
                    a["pp3d"],
                    a["ppx"],
                    a["ppd"],
                    a["rrw"],
                    a["qk"],
                    a["pk1"],
                    a["pk2"],
                    a["mm"],
                    a["fpk"],
                    a["divh"],
                    a["hadvp"],
                    a["elim_c"],
                    a["elim_d"],
                    a["tri_aa"],
                    a["tri_bb"],
                    a["tri_cc"],
                    a["tri_dd"],
                    a["tri_pivot"],
                    a["tri_residual"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(nrk),
                    u32(n),
                    u32(nloop),
                    f32(dts),
                    f32(rdx),
                    f32(rdy),
                    alph32,
                    beta,
                    asq,
                    f32(kdiv),
                    f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
                ),
                stream=stream,
            )
            if n < nloop:
                kbcp(
                    (1, 1, 1),
                    (1, 1, 1),
                    (a["ppd"], u32(ni), u32(nj), u32(nk)),
                    stream=stream,
                )
            kavg(
                grid_faces,
                (CUDA_SOUND3_PERIODIC_THREADS, 1, 1),
                (
                    a["u3d"],
                    a["v3d"],
                    a["rru"],
                    a["rrv"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(n),
                    u32(nloop),
                ),
                stream=stream,
            )
        kbcuv(
            (1, 1, 1),
            (1, 1, 1),
            (a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk)),
            stream=stream,
        )
        if nrk < 3:
            kbcp(
                (1, 1, 1),
                (1, 1, 1),
                (a["pp3d"], u32(ni), u32(nj), u32(nk)),
                stream=stream,
            )
        cadence.append(
            {
                "nrk": nrk,
                "dttmp_f32": rk_stage_dt_f32(dt, nrk),
                "nloop": nloop,
                "dts_f32": float(f32(dts)),
                "ppd_boundary_fills": nloop - 1,
                "stage_velocity_boundary_fills": 1,
                "stage_pressure_boundary_fills": 1 if nrk < 3 else 0,
                "ppx_written": nrk == 3,
            }
        )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_SOUND3_PERIODIC_SCHEMA,
        "claim_limit": CUDA_SOUND3_PERIODIC_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "layout": {
                "scalar": "(nk+2,nj+2,ni+2), one periodic halo",
                "u": "(nk+2,nj+2,ni+1), transverse halo plus duplicate x faces",
                "v": "(nk+2,nj+1,ni+2), transverse halo plus duplicate y faces",
                "w": "(nk+2,nj,ni)",
            },
        },
        "cadence": cadence,
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "kernel": {
            "abi": CUDA_SOUND3_PERIODIC_ABI,
            "names": list(CUDA_SOUND3_PERIODIC_KERNEL_NAMES),
            "source_sha256": CUDA_SOUND3_PERIODIC_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "runtime": dict(compiled.runtime_identity),
        "composition_seam": {
            "dry_u": "u3d[1:nk+1,1:nj+1,:]",
            "dry_v": "v3d[1:nk+1,:,1:ni+1]",
            "status": "layout-executable-but-not-composed; dry trajectory holds velocity and omits the complete solve2 tendency set",
        },
    }
    return CudaSound3PeriodicLaunch(event, compiled, receipt)


__all__ = (
    "ALL_ARRAY_NAMES",
    "CM1_SOUND3_PERIODIC_ALPH_F32",
    "CM1_SOUND3_PERIODIC_CP_F32",
    "CM1_SOUND3_PERIODIC_KDIV_F32",
    "CUDA_SOUND3_PERIODIC_CLAIM_LIMIT",
    "CUDA_SOUND3_PERIODIC_SCHEMA",
    "CUDA_SOUND3_PERIODIC_SOURCE_SHA256",
    "COMPARE_OUTPUT_NAMES",
    "CpuSound3PeriodicResult",
    "CpuSound3PeriodicStageDiagnostics",
    "CudaSound3PeriodicLaunch",
    "array_shapes",
    "compile_cuda_sound3_periodic",
    "compare_sound3_periodic_raw",
    "cpu_sound3_periodic_stage_inplace",
    "cpu_sound3_periodic_three_stage_inplace",
    "default_periodic_face_weights",
    "fill_periodic_scalar_inplace",
    "fill_periodic_uv_inplace",
    "launch_cuda_sound3_periodic_three_stage",
    "scalar_shape_zyx",
    "u_shape_zyx",
    "v_shape_zyx",
    "validate_periodic_contract",
    "w_shape_zyx",
)
