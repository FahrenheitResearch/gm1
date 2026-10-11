"""CM1's ``terrain_flag=.false.`` closure statements for the stretched Cartesian lane.

The Cartesian stretch_z lane is flat: CM1 runs the flat branch of every
closure routine there, and those branches read the reference ``rdz`` times the
column metrics ``mh``/``mf`` (param.F:8467,8476) where the terrain branch
reads ``rds``/``rdsf``.  The terrain texts this lane used to bind are kept for
terrain runs; this module is a separate, statement-for-statement transcription
of the flat branches, including the routines both branches share, so the flat
lane reproduces CM1's words and not an equivalent regrouping of them:

* ``calcdef`` flat strain (turb.F:4766-4895) plus the open-boundary copies and
  the deformation reduction (turb.F:5250-5388);
* ``gettau`` (turb.F:4094-4190, :4281-4413) with the free-slip lower and upper
  faces of ``sfcstress`` (turb.F:4536-4545) and ``gettau`` (:4423-4433);
* ``turbu``/``turbuz``, ``turbv``/``turbvz`` flat branches with the two-part
  ``dot2p`` block (turbtend.F:1095-1361, :1438-1560, :1767-1802,
  :1860-2128, :2194-2310, :2424-2459);
* ``turbw`` (turbtend.F:2508-2885), ``turbt`` (turbtend.F:729-1044);
* ``turbs`` flat fluxes and ``turbsz`` (turbtend.F:60-434, :442-692);
* TKE ``advw`` with ``iamw=.false.`` (adv.F:3252-3265, :3346-3355) and the
  solve2 RK update (solve2.F:1708-1803).

``uh``/``uf``/``vh``/``vf`` are exactly one on this lane (param.F:7077-7081,
:7459-7463), so their factors are the identity and are not formed.
``vialpha`` and ``vibeta`` are compile-time 1 and 0 (constants.F:32-35); both
factors are formed literally so the ``temb`` terms keep CM1's signed zeros.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_stretched_sgs_tke import CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE
from .nvrtc_module import compile_nvrtc_module


GM1_FLAT_STRETCHED_CLOSURE_ABI = "gm1.cuda-flat-stretched-closure-f32/v1"
GM1_FLAT_STRETCHED_TKE_ADVECTION_ABI = "gm1.cuda-flat-stretched-tke-advection-f32/v1"
GM1_FLAT_STRETCHED_CLOSURE_THREADS = 128

GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES = (
    "gm1_flat_calcdef_mass_v1",
    "gm1_flat_calcdef_face_v1",
    "gm1_flat_calcdef_reduce_v1",
    "gm1_flat_gettau_normal_v1",
    "gm1_flat_gettau_t12_v1",
    "gm1_flat_gettau_t13_v1",
    "gm1_flat_gettau_t23_v1",
    "gm1_flat_turbu_v1",
    "gm1_flat_turbv_v1",
    "gm1_flat_turbw_v1",
    "gm1_flat_turbt_v1",
    "gm1_flat_turbs_xy_flux_v1",
    "gm1_flat_turbs_implicit_apply_v1",
)
GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME = "gm1_flat_tke_advw_rk_v1"

# ``GM1_FLAT_NK`` is the column workspace; ``flat_stretched_closure_source``
# defines it ahead of this text for the grid being compiled.
GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE = r'''// gm1.cuda-flat-stretched-closure-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);}
__device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);}
__device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);}
__device__ __forceinline__ float D(float x,float y){return __fdiv_rn(x,y);}
__device__ __forceinline__ int C(int x,int n){return x<0?0:(x>=n?n-1:x);}
__device__ __forceinline__ int CL(int x,int lo,int hi){return x<lo?lo:(x>hi?hi:x);}
__device__ __forceinline__ unsigned long long I3(int k,int j,int i,int ny,int nx){return ((unsigned long long)k*(unsigned long long)ny+(unsigned long long)j)*(unsigned long long)nx+(unsigned long long)i;}
// Mass (nk), face (nk+1) and staggered tile carriers with CM1's zero-gradient
// open-boundary halo (bc.F:528-1143) as an index clamp.
__device__ __forceinline__ float CELL(const float*a,int k,int j,int i,int nj,int ni){return a[I3(k,C(j,nj),C(i,ni),nj,ni)];}
__device__ __forceinline__ unsigned long long FIRST(){return (unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;}
__device__ __forceinline__ unsigned long long STEP(){return (unsigned long long)blockDim.x*gridDim.x;}

// turb.F:4772-4786 with the four open-side s12 copies of turb.F:5132-5214
// taken as a face-index clamp.  Tile k is CM1 k-1; mh is the direct-index
// profile, so rdz*mh(1,1,k) is M(rdz,mh[k+1]).
extern "C" __global__ void gm1_flat_calcdef_mass_v1(
 const float*u,const float*v,const float*w,const float*rho,const float*mh,
 float*s11,float*s22,float*s33,float*s12,unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz){
 const int NI=(int)ni,NJ=(int)nj;const unsigned long long total=(unsigned long long)nk*(nj+1u)*(ni+1u);
 for(unsigned long long q=FIRST();q<total;q+=STEP()){
  int ic=(int)(q%(ni+1u)),jc=(int)((q/(ni+1u))%(nj+1u)),k=(int)(q/((unsigned long long)(ni+1u)*(nj+1u)));
  if(ic<NI&&jc<NJ){unsigned long long p=I3(k,jc,ic,NJ,NI);float r=rho[p];
   s11[p]=M(M(r,S(u[I3(k,jc,ic+1,NJ,NI+1)],u[I3(k,jc,ic,NJ,NI+1)])),rdx);
   s22[p]=M(M(r,S(v[I3(k,jc+1,ic,NJ+1,NI)],v[I3(k,jc,ic,NJ+1,NI)])),rdy);
   s33[p]=M(M(r,S(w[I3(k+1,jc,ic,NJ,NI)],w[I3(k,jc,ic,NJ,NI)])),M(rdz,mh[k+1]));}
  int ii=CL(ic,1,NI-1),jj=CL(jc,1,NJ-1);
  float x=M(S(u[I3(k,jj,ii,NJ,NI+1)],u[I3(k,jj-1,ii,NJ,NI+1)]),rdy);
  float y=M(S(v[I3(k,jj,ii,NJ+1,NI)],v[I3(k,jj,ii-1,NJ+1,NI)]),rdx);
  float rs=A(A(rho[I3(k,jj-1,ii-1,NJ,NI)],rho[I3(k,jj,ii,NJ,NI)]),A(rho[I3(k,jj,ii-1,NJ,NI)],rho[I3(k,jj-1,ii,NJ,NI)]));
  s12[I3(k,jc,ic,NJ+1,NI+1)]=M(M(M(0.5f,A(x,y)),0.25f),rs);}}

// turb.F:4880-4895.  mf and rf are the tile face carriers.
extern "C" __global__ void gm1_flat_calcdef_face_v1(
 const float*u,const float*v,const float*w,const float*rf,const float*mf,float*s13,float*s23,
 unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz){
 const int NI=(int)ni,NJ=(int)nj;const unsigned long long total=(unsigned long long)(nk-1u)*(nj+1u)*(ni+1u);
 for(unsigned long long q=FIRST();q<total;q+=STEP()){
  int ic=(int)(q%(ni+1u)),jc=(int)((q/(ni+1u))%(nj+1u)),kf=(int)(q/((unsigned long long)(ni+1u)*(nj+1u)))+1;
  if(jc<NJ){int il=C(ic-1,NI),ir=C(ic,NI);
   float dw=M(S(w[I3(kf,jc,ir,NJ,NI)],w[I3(kf,jc,il,NJ,NI)]),rdx);
   float du=M(M(M(S(u[I3(kf,jc,ic,NJ,NI+1)],u[I3(kf-1,jc,ic,NJ,NI+1)]),rdz),0.5f),A(mf[I3(kf,jc,il,NJ,NI)],mf[I3(kf,jc,ir,NJ,NI)]));
   s13[I3(kf,jc,ic,NJ,NI+1)]=M(M(M(0.5f,A(dw,du)),0.5f),A(rf[I3(kf,jc,il,NJ,NI)],rf[I3(kf,jc,ir,NJ,NI)]));}
  if(ic<NI){int js=C(jc-1,NJ),jn=C(jc,NJ);
   float dw=M(S(w[I3(kf,jn,ic,NJ,NI)],w[I3(kf,js,ic,NJ,NI)]),rdy);
   float dv=M(M(M(S(v[I3(kf,jc,ic,NJ+1,NI)],v[I3(kf-1,jc,ic,NJ+1,NI)]),rdz),0.5f),A(mf[I3(kf,js,ic,NJ,NI)],mf[I3(kf,jn,ic,NJ,NI)]));
   s23[I3(kf,jc,ic,NJ+1,NI)]=M(M(M(0.5f,A(dw,dv)),0.5f),A(rf[I3(kf,js,ic,NJ,NI)],rf[I3(kf,jn,ic,NJ,NI)]));}}}

// turb.F:5361-5385.
extern "C" __global__ void gm1_flat_calcdef_reduce_v1(
 const float*rf,const float*c1,const float*c2,const float*s11,const float*s22,const float*s33,const float*s12,const float*s13,const float*s23,float*defv,float*defh,
 unsigned ni,unsigned nj,unsigned nk){
 const int NI=(int)ni,NJ=(int)nj;const unsigned long long total=(unsigned long long)ni*nj*(nk-1u),nxy=(unsigned long long)ni*nj;
 for(unsigned long long q=FIRST();q<total;q+=STEP()){
  int i=(int)(q%ni),j=(int)((q/ni)%nj),kf=(int)(q/nxy)+1;
  unsigned long long p=I3(kf,j,i,NJ,NI),lo=I3(kf-1,j,i,NJ,NI);float a=c1[p],b=c2[p];
  float t11=A(M(a,M(s11[lo],s11[lo])),M(b,M(s11[p],s11[p])));
  float t22=A(M(a,M(s22[lo],s22[lo])),M(b,M(s22[p],s22[p])));
  float t33=A(M(a,M(s33[lo],s33[lo])),M(b,M(s33[p],s33[p])));
  float l00=s12[I3(kf-1,j,i,NJ+1,NI+1)],l11=s12[I3(kf-1,j+1,i+1,NJ+1,NI+1)],l10=s12[I3(kf-1,j+1,i,NJ+1,NI+1)],l01=s12[I3(kf-1,j,i+1,NJ+1,NI+1)];
  float h00=s12[I3(kf,j,i,NJ+1,NI+1)],h11=s12[I3(kf,j+1,i+1,NJ+1,NI+1)],h10=s12[I3(kf,j+1,i,NJ+1,NI+1)],h01=s12[I3(kf,j,i+1,NJ+1,NI+1)];
  float t12=M(0.25f,A(M(a,A(A(M(l00,l00),M(l11,l11)),A(M(l10,l10),M(l01,l01)))),M(b,A(A(M(h00,h00),M(h11,h11)),A(M(h10,h10),M(h01,h01))))));
  float x0=s13[I3(kf,j,i,NJ,NI+1)],x1=s13[I3(kf,j,i+1,NJ,NI+1)],y0=s23[I3(kf,j,i,NJ+1,NI)],y1=s23[I3(kf,j+1,i,NJ+1,NI)];
  float t13=M(0.5f,A(M(x0,x0),M(x1,x1))),t23=M(0.5f,A(M(y0,y0),M(y1,y1))),rrf=D(1.0f,M(rf[p],rf[p]));
  defv[p]=M(M(4.0f,A(t13,t23)),rrf);
  defh[p]=M(A(M(2.0f,A(A(t11,t22),t33)),M(4.0f,t12)),rrf);}}

// turb.F:4107-4115: t11,t22,t33 times (kmh(k)+kmh(k+1)).
extern "C" __global__ void gm1_flat_gettau_normal_v1(const float*s11,const float*s22,const float*s33,const float*kmh,float*t11,float*t22,float*t33,unsigned ni,unsigned nj,unsigned nk){
 const unsigned long long n=(unsigned long long)ni*nj*nk,z=(unsigned long long)ni*nj;
 for(unsigned long long q=FIRST();q<n;q+=STEP()){float m=A(kmh[q],kmh[q+z]);t11[q]=M(s11[q],m);t22[q]=M(s22[q],m);t33[q]=M(s33[q],m);}}

// turb.F:4116-4118 then the open-side and corner copies of turb.F:4329-4388,
// which leave every edge face equal to the clamped interior face.
extern "C" __global__ void gm1_flat_gettau_t12_v1(const float*s12,const float*kmh,float*t12,unsigned ni,unsigned nj,unsigned nk){
 const int NI=(int)ni,NJ=(int)nj;const unsigned long long n=(unsigned long long)nk*(nj+1u)*(ni+1u);
 for(unsigned long long q=FIRST();q<n;q+=STEP()){
  int ic=(int)(q%(ni+1u)),jc=(int)((q/(ni+1u))%(nj+1u)),k=(int)(q/((unsigned long long)(ni+1u)*(nj+1u)));
  int ii=CL(ic,1,NI-1),jj=CL(jc,1,NJ-1);
  float lo=A(A(kmh[I3(k,jj-1,ii-1,NJ,NI)],kmh[I3(k,jj,ii,NJ,NI)]),A(kmh[I3(k,jj,ii-1,NJ,NI)],kmh[I3(k,jj-1,ii,NJ,NI)]));
  float hi=A(A(kmh[I3(k+1,jj-1,ii-1,NJ,NI)],kmh[I3(k+1,jj,ii,NJ,NI)]),A(kmh[I3(k+1,jj,ii-1,NJ,NI)],kmh[I3(k+1,jj-1,ii,NJ,NI)]));
  t12[I3(k,jc,ic,NJ+1,NI+1)]=M(M(s12[I3(k,jj,ii,NJ+1,NI+1)],0.25f),A(lo,hi));}}

// turb.F:4179-4182 for k=2..nk, the free-slip bottom copy of turb.F:4545
// (k=1 takes k=2) and the free-slip top of turb.F:4430 (k=nk+1 takes nk).
extern "C" __global__ void gm1_flat_gettau_t13_v1(const float*s13,const float*kmv,float*t13,unsigned ni,unsigned nj,unsigned nk){
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;const unsigned long long n=(unsigned long long)(nk+1u)*nj*(ni+1u);
 for(unsigned long long q=FIRST();q<n;q+=STEP()){
  int i=(int)(q%(ni+1u)),j=(int)((q/(ni+1u))%nj),f=(int)(q/((unsigned long long)(ni+1u)*nj));int ff=f==0?1:(f==K?K-1:f);
  t13[q]=M(s13[I3(ff,j,i,NJ,NI+1)],A(CELL(kmv,ff,j,i-1,NJ,NI),CELL(kmv,ff,j,i,NJ,NI)));}}
extern "C" __global__ void gm1_flat_gettau_t23_v1(const float*s23,const float*kmv,float*t23,unsigned ni,unsigned nj,unsigned nk){
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;const unsigned long long n=(unsigned long long)(nk+1u)*(nj+1u)*ni;
 for(unsigned long long q=FIRST();q<n;q+=STEP()){
  int i=(int)(q%ni),j=(int)((q/ni)%(nj+1u)),f=(int)(q/((unsigned long long)ni*(nj+1u)));int ff=f==0?1:(f==K?K-1:f);
  t23[q]=M(s23[I3(ff,j,i,NJ+1,NI)],A(CELL(kmv,ff,j-1,i,NJ,NI),CELL(kmv,ff,j,i,NJ,NI)));}}

// turbuz/turbvz implicit coefficients for one staggered column (turbtend.F:
// 1456-1560 and :2204-2310).  a and b are CM1's (i-1) and (i) [or (j-1) and
// (j)] tile cells; mass k is CM1 k+1 and face k is CM1 k+1.
#define PAIR(arr,k) A(CELL(arr,k,ja,ia,NJ,NI),CELL(arr,k,jb,ib,NJ,NI))
#define COLUMN_IMPLICIT(QK,WX,RD) \
 float rdt=D(0.5f,dt),tema=M(M(M(M(-0.0625f,dt),1.0f),rdz),rdz),temb=M(M(M(M(0.0625f,dt),0.0f),rdz),rdz),temc=M(M(0.5f,dt),rdz); \
 { int k=0;float rr0=D(1.0f,M(0.5f,PAIR(rho,k)));float tem=M(PAIR(mh,k),rr0); \
   float r2=M(M(M(PAIR(kmv,k+1),PAIR(mf,k+1)),PAIR(rf,k+1)),tem); \
   float cfc=M(tema,r2),cfb=S(1.0f,cfc),qk=QK(k),qp=QK(k+1); \
   float cfd=S(A(qk,M(temb,S(M(r2,qp),M(r2,qk)))),M(M(M(temc,TOP13(0)),PAIR(mh,0)),rr0)); \
   float inv=D(1.0f,cfb);d1[0]=-M(cfc,inv);d2[0]=M(cfd,inv);d7[0]=0.0f; } \
 for(int k=1;k<K-1;k++){ float rr0=D(1.0f,M(0.5f,PAIR(rho,k)));float tem=M(PAIR(mh,k),rr0); \
   float r1=M(M(M(PAIR(kmv,k),PAIR(mf,k)),PAIR(rf,k)),tem); \
   float r2=M(M(M(PAIR(kmv,k+1),PAIR(mf,k+1)),PAIR(rf,k+1)),tem); \
   float cfa=M(tema,r1),cfc=M(tema,r2),cfb=S(S(1.0f,cfa),cfc),qk=QK(k),qp=QK(k+1),qm=QK(k-1); \
   float cfd=A(qk,M(temb,A(S(M(r2,qp),M(A(r1,r2),qk)),M(r1,qm)))); \
   float inv=D(1.0f,A(M(cfa,d1[k-1]),cfb));d1[k]=-M(cfc,inv);d2[k]=M(S(cfd,M(cfa,d2[k-1])),inv); \
   d7[k]=M(M(M(M(S(CELL(w,k,jb,ib,NJ,NI),CELL(w,k,ja,ia,NJ,NI)),RD),0.25f),PAIR(kmv,k)),PAIR(rf,k)); } \
 { int k=K-1;float rr0=D(1.0f,M(0.5f,PAIR(rho,k)));float tem=M(PAIR(mh,k),rr0); \
   float r1=M(M(M(PAIR(kmv,k),PAIR(mf,k)),PAIR(rf,k)),tem); \
   float cfa=M(tema,r1),cfb=S(1.0f,cfa),qk=QK(k),qm=QK(k-1); \
   float cfd=A(A(qk,M(temb,A(-M(r1,qk),M(r1,qm)))),M(M(M(temc,TOP13(K)),PAIR(mh,k)),rr0)); \
   float inv=D(1.0f,A(M(cfa,d1[k-1]),cfb));d3[k]=M(S(cfd,M(cfa,d2[k-1])),inv); \
   tz[k]=M(M(PAIR(rho,k),S(d3[k],qk)),rdt); \
   d7[k]=M(M(M(M(S(CELL(w,k,jb,ib,NJ,NI),CELL(w,k,ja,ia,NJ,NI)),RD),0.25f),PAIR(kmv,k)),PAIR(rf,k)); \
   tz[k]=A(tz[k],M(M(M(S(0.0f,d7[k]),rdz),0.5f),PAIR(mh,k))); } \
 for(int k=K-2;k>=0;k--){ d3[k]=A(M(d1[k],d3[k+1]),d2[k]); \
   tz[k]=M(M(PAIR(rho,k),S(d3[k],QK(k))),rdt); \
   tz[k]=A(tz[k],M(M(M(S(d7[k+1],d7[k]),rdz),0.5f),PAIR(mh,k))); } \
 if(ntwk>0u){ for(int k=0;k<(int)ntwk;k++){ float tm=M(rdz,mhc[k+1]); \
   float dd=S(T2P(k+1),T2P(k)); \
   tz[k]=A(tz[k],M(tm,dd)); \
   float rr0=D(1.0f,M(0.5f,PAIR(rho,k))); \
   WX[I3COL(k)]=M(M(tm,dd),rr0); } }

// turbtend.F:1767-1802 / :2424-2459: dum2(k)=0.5*(rf(a)*fw(k)+rf(b)*fw(k)),
// zero at k=1 and k=ntwk+1.
#define T2P(wf) (((wf)==0||(wf)>=(int)ntwk)?0.0f:M(0.5f,A(M(CELL(rf,(wf),ja,ia,NJ,NI),tfw[(wf)]),M(CELL(rf,(wf),jb,ib,NJ,NI),tfw[(wf)]))))

// turbu (turbtend.F:1095-1111, :1200-1227, :1291-1300) with turbuz.
extern "C" __global__ void gm1_flat_turbu_v1(
 const float*t11,const float*t12,const float*t13,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,const float*mhc,
 const float*u,const float*w,float*uten,const float*tfw,float*t2pt,
 unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt,unsigned ntwk){
 const unsigned long long col=FIRST(),n=(unsigned long long)nj*(ni+1u);if(col>=n)return;
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;int f=(int)(col%(ni+1u)),j=(int)(col/(ni+1u));
 int ia=f-1,ib=f,ja=j,jb=j;
 float hx[GM1_FLAT_NK],hy[GM1_FLAT_NK],d1[GM1_FLAT_NK],d2[GM1_FLAT_NK],d3[GM1_FLAT_NK],d7[GM1_FLAT_NK],tz[GM1_FLAT_NK];
 for(int k=0;k<K;k++){
  hx[k]=M(S(CELL(t11,k,j,ib,NJ,NI),CELL(t11,k,j,ia,NJ,NI)),rdx);
  hy[k]=M(S(t12[I3(k,j+1,f,NJ+1,NI+1)],t12[I3(k,j,f,NJ+1,NI+1)]),rdy);
  if(f==0||f==NI)hx[k]=0.0f;
  if(j==0||j==NJ-1)hy[k]=0.0f;}
 #define QK(k) u[I3((k),j,f,NJ,NI+1)]
 #define TOP13(k) t13[I3((k),j,f,NJ,NI+1)]
 #define I3COL(k) I3((k),j,f,NJ,NI+1)
 COLUMN_IMPLICIT(QK,t2pt,rdx)
 #undef QK
 #undef TOP13
 #undef I3COL
 for(int k=0;k<K;k++){float rr0=D(1.0f,M(0.5f,PAIR(rho,k)));unsigned long long p=I3(k,j,f,NJ,NI+1);uten[p]=A(uten[p],M(A(A(hx[k],hy[k]),tz[k]),rr0));}}

// turbv (turbtend.F:1860-1876, :1962-1989, :2046-2056) with turbvz.
extern "C" __global__ void gm1_flat_turbv_v1(
 const float*t12,const float*t22,const float*t23,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,const float*mhc,
 const float*v,const float*w,float*vten,const float*tfw,float*t2pt,
 unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt,unsigned ntwk){
 const unsigned long long col=FIRST(),n=(unsigned long long)(nj+1u)*ni;if(col>=n)return;
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;int i=(int)(col%ni),f=(int)(col/ni);
 int ia=i,ib=i,ja=f-1,jb=f;
 float hx[GM1_FLAT_NK],hy[GM1_FLAT_NK],d1[GM1_FLAT_NK],d2[GM1_FLAT_NK],d3[GM1_FLAT_NK],d7[GM1_FLAT_NK],tz[GM1_FLAT_NK];
 for(int k=0;k<K;k++){
  hx[k]=M(S(t12[I3(k,f,i+1,NJ+1,NI+1)],t12[I3(k,f,i,NJ+1,NI+1)]),rdx);
  hy[k]=M(S(CELL(t22,k,jb,i,NJ,NI),CELL(t22,k,ja,i,NJ,NI)),rdy);
  if(i==0||i==NI-1)hx[k]=0.0f;
  if(f==0||f==NJ)hy[k]=0.0f;}
 #define QK(k) v[I3((k),f,i,NJ+1,NI)]
 #define TOP13(k) t23[I3((k),f,i,NJ+1,NI)]
 #define I3COL(k) I3((k),f,i,NJ+1,NI)
 COLUMN_IMPLICIT(QK,t2pt,rdy)
 #undef QK
 #undef TOP13
 #undef I3COL
 for(int k=0;k<K;k++){float rr0=D(1.0f,M(0.5f,PAIR(rho,k)));unsigned long long p=I3(k,f,i,NJ+1,NI);vten[p]=A(vten[p],M(A(A(hx[k],hy[k]),tz[k]),rr0));}}
#undef COLUMN_IMPLICIT
#undef T2P
#undef PAIR

// turbw (turbtend.F:2508-2535, :2572-2600, :2644-2760, :2808-2819).
extern "C" __global__ void gm1_flat_turbw_v1(
 const float*t13,const float*t23,const float*t33,const float*kmh,const float*rho,const float*rf,const float*mh,const float*mf,
 const float*w,float*wten,unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt){
 const unsigned long long col=FIRST(),n=(unsigned long long)nj*ni;if(col>=n)return;
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;int i=(int)(col%ni),j=(int)(col/ni);
 float hx[GM1_FLAT_NK],hy[GM1_FLAT_NK],d1[GM1_FLAT_NK],d2[GM1_FLAT_NK],d3[GM1_FLAT_NK],tz[GM1_FLAT_NK];
 #define WF(f) w[I3((f),j,i,NJ,NI)]
 #define PF(a,f) a[I3((f),j,i,NJ,NI)]
 for(int f=1;f<K;f++){
  hx[f]=M(S(t13[I3(f,j,i+1,NJ,NI+1)],t13[I3(f,j,i,NJ,NI+1)]),rdx);
  hy[f]=M(S(t23[I3(f,j+1,i,NJ+1,NI)],t23[I3(f,j,i,NJ+1,NI)]),rdy);
  if(i==0||i==NI-1)hx[f]=0.0f;
  if(j==0||j==NJ-1)hy[f]=0.0f;}
 float rdt=D(1.0f,dt),tema=M(M(M(M(-1.0f,dt),1.0f),rdz),rdz),temb=M(M(M(dt,0.0f),rdz),rdz),temc=M(dt,rdz);
 { int f=1;float rrf=D(PF(mf,f),PF(rf,f));
   float r2=M(M(M(A(PF(kmh,f),PF(kmh,f+1)),PF(mh,f)),PF(rho,f)),rrf);
   float cfc=M(tema,r2),cfb=S(1.0f,cfc);
   float cfd=S(A(WF(f),M(temb,S(M(r2,WF(f+1)),M(r2,WF(f))))),M(M(temc,PF(t33,f-1)),rrf));
   float inv=D(1.0f,cfb);d1[f]=-M(cfc,inv);d2[f]=M(cfd,inv); }
 for(int f=2;f<K-1;f++){ float rrf=D(PF(mf,f),PF(rf,f));
   float r1=M(M(M(A(PF(kmh,f-1),PF(kmh,f)),PF(mh,f-1)),PF(rho,f-1)),rrf);
   float r2=M(M(M(A(PF(kmh,f),PF(kmh,f+1)),PF(mh,f)),PF(rho,f)),rrf);
   float cfa=M(tema,r1),cfc=M(tema,r2),cfb=S(S(1.0f,cfa),cfc);
   float cfd=A(WF(f),M(temb,A(S(M(r2,WF(f+1)),M(A(r1,r2),WF(f))),M(r1,WF(f-1)))));
   float inv=D(1.0f,A(M(cfa,d1[f-1]),cfb));d1[f]=-M(cfc,inv);d2[f]=M(S(cfd,M(cfa,d2[f-1])),inv); }
 { int f=K-1;float rrf=D(PF(mf,f),PF(rf,f));
   float r1=M(M(M(A(PF(kmh,f-1),PF(kmh,f)),PF(mh,f-1)),PF(rho,f-1)),rrf);
   float cfa=M(tema,r1),cfb=S(1.0f,cfa);
   float cfd=A(A(WF(f),M(temb,A(-M(r1,WF(f)),M(r1,WF(f-1))))),M(M(temc,PF(t33,f)),rrf));
   float inv=D(1.0f,A(M(cfa,d1[f-1]),cfb));d3[f]=M(S(cfd,M(cfa,d2[f-1])),inv);
   tz[f]=M(M(PF(rf,f),S(d3[f],WF(f))),rdt); }
 for(int f=K-2;f>=1;f--){ d3[f]=A(M(d1[f],d3[f+1]),d2[f]);tz[f]=M(M(PF(rf,f),S(d3[f],WF(f))),rdt); }
 for(int f=1;f<K;f++){unsigned long long p=I3(f,j,i,NJ,NI);wten[p]=A(wten[p],D(A(A(hx[f],hy[f]),tz[f]),rf[p]));}
 #undef WF
 #undef PF
}

// turbt (turbtend.F:729-786, :856-883, :938-1036).  dumx(a) is the x face
// between cells a-1 and a: -0.25*(rf(a)+rf(a-1))*2*(kmh(a)+kmh(a-1))*(t(a)-t(a-1))*rdx.
__device__ __forceinline__ float TFX(const float*t,const float*km,const float*rf,int f,int j,int a,int NJ,int NI,float rdx){
 return -M(M(M(M(M(0.25f,A(CELL(rf,f,j,a,NJ,NI),CELL(rf,f,j,a-1,NJ,NI))),2.0f),A(CELL(km,f,j,a,NJ,NI),CELL(km,f,j,a-1,NJ,NI))),S(CELL(t,f,j,a,NJ,NI),CELL(t,f,j,a-1,NJ,NI))),rdx);}
__device__ __forceinline__ float TFY(const float*t,const float*km,const float*rf,int f,int b,int i,int NJ,int NI,float rdy){
 return -M(M(M(M(M(0.25f,A(CELL(rf,f,b,i,NJ,NI),CELL(rf,f,b-1,i,NJ,NI))),2.0f),A(CELL(km,f,b,i,NJ,NI),CELL(km,f,b-1,i,NJ,NI))),S(CELL(t,f,b,i,NJ,NI),CELL(t,f,b-1,i,NJ,NI))),rdy);}
extern "C" __global__ void gm1_flat_turbt_v1(
 const float*t,const float*kmh,const float*kmv,const float*rho,const float*rf,const float*mh,const float*mf,float*ten,
 unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt){
 const unsigned long long col=FIRST(),n=(unsigned long long)nj*ni;if(col>=n)return;
 const int NI=(int)ni,NJ=(int)nj,K=(int)nk;int i=(int)(col%ni),j=(int)(col/ni);
 float hx[GM1_FLAT_NK],hy[GM1_FLAT_NK],dx_[GM1_FLAT_NK],dy_[GM1_FLAT_NK],dz_[GM1_FLAT_NK],tz[GM1_FLAT_NK];
 #define TF(f) t[I3((f),j,i,NJ,NI)]
 #define PF(a,f) a[I3((f),j,i,NJ,NI)]
 for(int f=1;f<K;f++){
  hx[f]=-M(S(TFX(t,kmh,rf,f,j,i+1,NJ,NI,rdx),TFX(t,kmh,rf,f,j,i,NJ,NI,rdx)),rdx);
  hy[f]=-M(S(TFY(t,kmh,rf,f,j+1,i,NJ,NI,rdy),TFY(t,kmh,rf,f,j,i,NJ,NI,rdy)),rdy);
  if(i==0||i==NI-1)hx[f]=0.0f;
  if(j==0||j==NJ-1)hy[f]=0.0f;}
 float rdt=D(1.0f,dt),tema=M(M(M(M(-1.0f,dt),1.0f),rdz),rdz),temb=M(M(M(dt,0.0f),rdz),rdz);
 { int f=1;float rrf=D(PF(mf,f),PF(rf,f));
   float r2=M(M(M(A(PF(kmv,f),PF(kmv,f+1)),PF(mh,f)),PF(rho,f)),rrf);
   float cfc=M(tema,r2),cfb=S(1.0f,cfc);
   float cfd=A(TF(f),M(temb,S(M(r2,TF(f+1)),M(r2,TF(f)))));
   float inv=D(1.0f,cfb);dx_[f]=-M(cfc,inv);dy_[f]=M(cfd,inv); }
 for(int f=2;f<K-1;f++){ float rrf=D(PF(mf,f),PF(rf,f));
   float r1=M(M(M(A(PF(kmv,f-1),PF(kmv,f)),PF(mh,f-1)),PF(rho,f-1)),rrf);
   float r2=M(M(M(A(PF(kmv,f),PF(kmv,f+1)),PF(mh,f)),PF(rho,f)),rrf);
   float cfa=M(tema,r1),cfc=M(tema,r2),cfb=S(S(1.0f,cfa),cfc);
   float cfd=A(TF(f),M(temb,A(S(M(r2,TF(f+1)),M(A(r1,r2),TF(f))),M(r1,TF(f-1)))));
   float inv=D(1.0f,A(M(cfa,dx_[f-1]),cfb));dx_[f]=-M(cfc,inv);dy_[f]=M(S(cfd,M(cfa,dy_[f-1])),inv); }
 { int f=K-1;float rrf=D(PF(mf,f),PF(rf,f));
   float r1=M(M(M(A(PF(kmv,f-1),PF(kmv,f)),PF(mh,f-1)),PF(rho,f-1)),rrf);
   float cfa=M(tema,r1),cfb=S(1.0f,cfa);
   float cfd=A(TF(f),M(temb,A(-M(r1,TF(f)),M(r1,TF(f-1)))));
   float inv=D(1.0f,A(M(cfa,dx_[f-1]),cfb));dz_[f]=M(S(cfd,M(cfa,dy_[f-1])),inv);
   tz[f]=M(M(PF(rf,f),S(dz_[f],TF(f))),rdt); }
 for(int f=K-2;f>=1;f--){ dz_[f]=A(M(dx_[f],dz_[f+1]),dy_[f]);tz[f]=M(M(PF(rf,f),S(dz_[f],TF(f))),rdt); }
 for(int f=1;f<K;f++){unsigned long long p=I3(f,j,i,NJ,NI);ten[p]=A(ten[p],D(A(A(hx[f],hy[f]),tz[f]),rf[p]));}
 #undef TF
 #undef PF
}

// turbs flat fluxes (turbtend.F:71-84) on the one-cell open-halo copies
// gm1_turbs_pad_static_open/pack_scalar_open write: halo cell (j+1,i+1) is
// tile cell (j,i).
extern "C" __global__ void gm1_flat_turbs_xy_flux_v1(
 const float*sh,const float*rhoh,const float*khhh,float*fx,float*fy,unsigned ni,unsigned nj,unsigned nk,float dx,float dy){
 const unsigned long long q0=FIRST(),st=STEP();const int HX=(int)ni+2,HY=(int)nj+2;
 const unsigned long long nx=(unsigned long long)nk*nj*(ni+1u),ny=(unsigned long long)nk*(nj+1u)*ni;float rx=D(1.0f,dx),ry=D(1.0f,dy);
 for(unsigned long long z=q0;z<nx;z+=st){int f=(int)(z%(ni+1u)),j=(int)((z/(ni+1u))%nj),k=(int)(z/((unsigned long long)(ni+1u)*nj));
  unsigned long long l=I3(k,j+1,f,HY,HX),r=I3(k,j+1,f+1,HY,HX),lu=I3(k+1,j+1,f,HY,HX),ru=I3(k+1,j+1,f+1,HY,HX);
  fx[z]=M(M(M(M(-0.125f,A(rhoh[r],rhoh[l])),A(A(khhh[r],khhh[l]),A(khhh[ru],khhh[lu]))),S(sh[r],sh[l])),rx);}
 for(unsigned long long z=q0;z<ny;z+=st){int i=(int)(z%ni),f=(int)((z/ni)%(nj+1u)),k=(int)(z/((unsigned long long)ni*(nj+1u)));
  unsigned long long s=I3(k,f,i+1,HY,HX),nn=I3(k,f+1,i+1,HY,HX),su=I3(k+1,f,i+1,HY,HX),nu=I3(k+1,f+1,i+1,HY,HX);
  fy[z]=M(M(M(M(-0.125f,A(rhoh[nn],rhoh[s])),A(A(khhh[nn],khhh[s]),A(khhh[nu],khhh[su]))),S(sh[nn],sh[s])),ry);}}

// turbs flat tendency (turbtend.F:100-104, :300-327, :385-396) with turbsz's
// implicit solve (turbtend.F:545-690) and solve1.F:1171-1180's dum7/dum8.
extern "C" __global__ void gm1_flat_turbs_implicit_apply_v1(
 const float*sh,const float*rho,const float*rr,const float*rf,const float*mh,const float*mf,const float*khv,
 const float*fx,const float*fy,const float*sflux,float*cp,float*dp,float*target,
 unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz,float dt,unsigned iflux){
 const unsigned long long col=FIRST(),ncol=(unsigned long long)ni*nj;if(col>=ncol)return;
 const int NI=(int)ni,NJ=(int)nj,HX=(int)ni+2,HY=(int)nj+2;int i=(int)(col%ni),j=(int)(col/ni);
 float rx=D(1.0f,dx),ry=D(1.0f,dy),rz=D(1.0f,dz),rdt=D(1.0f,dt),tema=M(M(M(M(-1.0f,dt),1.0f),rz),rz),temb=M(M(M(dt,0.0f),rz),rz),temc=M(dt,rz);
 const unsigned long long nxy=ncol;
 #define SC(K) sh[I3((K),j+1,i+1,HY,HX)]
 #define COEF(F,K) M(M(M(M(khv[I3((F),j,i,NJ,NI)],mf[I3((F),j,i,NJ,NI)]),rf[I3((F),j,i,NJ,NI)]),mh[I3((K),j,i,NJ,NI)]),rr[I3((K),j,i,NJ,NI)])
 float r2=COEF(1,0),cfc=M(tema,r2),cfb=S(1.0f,cfc),ex=A(M(-r2,SC(0)),M(r2,SC(1)));
 float bc=M(M(M(temc,(iflux==1u)?M(sflux[col],rf[col]):0.0f),mh[col]),rr[col]);
 float cfd=A(A(SC(0),M(temb,ex)),bc),tem=D(1.0f,cfb);cp[col]=M(-cfc,tem);dp[col]=M(cfd,tem);
 for(int k=1;k+1<(int)nk;++k){float r1=COEF(k,k);r2=COEF(k+1,k);float cfa=M(tema,r1);cfc=M(tema,r2);cfb=S(S(1.0f,cfa),cfc);
  ex=A(S(M(r1,SC(k-1)),M(A(r1,r2),SC(k))),M(r2,SC(k+1)));cfd=A(SC(k),M(temb,ex));
  unsigned long long p=(unsigned long long)k*nxy+col,pm=p-nxy;tem=D(1.0f,A(M(cfa,cp[pm]),cfb));cp[p]=M(-cfc,tem);dp[p]=M(S(cfd,M(cfa,dp[pm])),tem);}
 int k=(int)nk-1;float r1=COEF(k,k),cfa=M(tema,r1);cfb=S(1.0f,cfa);ex=S(M(r1,SC(k-1)),M(r1,SC(k)));
 bc=M(M(M(temc,0.0f),mh[(unsigned long long)k*nxy+col]),rr[(unsigned long long)k*nxy+col]);cfd=S(A(SC(k),M(temb,ex)),bc);
 unsigned long long p=(unsigned long long)k*nxy+col,pm=p-nxy;tem=D(1.0f,A(M(cfa,cp[pm]),cfb));float sol=M(S(cfd,M(cfa,dp[pm])),tem);
 for(int kk=(int)nk-1;kk>=0;--kk){unsigned long long mass=(unsigned long long)kk*nxy+col;if(kk!=(int)nk-1)sol=A(M(cp[mass],sol),dp[mass]);
  float tz=M(M(rho[mass],S(sol,SC(kk))),rdt);
  unsigned long long xp=I3(kk,j,i,NJ,NI+1),yp=I3(kk,j,i,NJ+1,NI);
  float tx=-M(S(fx[xp+1u],fx[xp]),rx);if(i==0||i+1==NI)tx=0.0f;
  float ty=-M(S(fy[yp+(unsigned long long)ni],fy[yp]),ry);if(j==0||j+1==NJ)ty=0.0f;
  target[mass]=A(target[mass],M(A(A(tx,ty),tz),rr[mass]));}
 #undef SC
 #undef COEF
}
'''

GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE.encode()
).hexdigest()


def _replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> str:
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"flat TKE advection {label}: expected {count} site(s), found {found}")
    return source.replace(old, new)


def _flat_tke_advection_source() -> str:
    """adv.F advw (iamw=.false.) flat words, from the audited stretched text.

    The WENO/flux-form reconstructions, the inflow flux copies and the
    solve2 RK update are shared by both branches and are kept.  The flat
    branch changes three things: the vertical divergence is
    ``-(dumz(k)-dumz(k-1))*rdz*mf(1,1,k)`` (adv.F:3252-3262), the tendency is
    scaled by ``rrf0(1,1,k)`` with column ``c1``/``c2`` (adv.F:3346-3355), and
    the open-boundary divx piece is ``w*(..)*rdx*uh`` evaluated left to right
    (adv.F:3037-3095).
    """

    text = CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE
    head = text[: text.index("// turbt is one independent cell column")]
    start = text.index("__device__ __forceinline__ float RRU(")
    stop = text.index('extern "C" __global__ void cm1_tke_commit_v1(')
    body = text[start:stop]
    head = _replace_exact(
        head,
        "// gm1.cuda-child-stretched-sgs-tke-kernels/v1:tke",
        "// " + GM1_FLAT_STRETCHED_TKE_ADVECTION_ABI,
        count=1,
        label="header",
    )
    body = _replace_exact(
        body,
        'extern "C" __global__ void cm1_tke_advw_rk_v1(const float*q,const float*base,const float*ten,const float*c1,const float*c2,const float*rrf0,const float*gz,const float*rru,',
        'extern "C" __global__ void ' + GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME + '(const float*q,const float*base,const float*ten,const float*c1,const float*c2,const float*rrf0,const float*rru,',
        count=1,
        label="kernel name",
    )
    body = _replace_exact(
        body,
        "float dx,float dy,float dz,float dttmp,unsigned w,double ep,const float*rds){",
        "float dx,float dy,float dttmp,unsigned w,double ep,const float*mf,float rdz){",
        count=1,
        label="signature",
    )
    body = _replace_exact(
        body, "ax=S(ax,M(q[z],M(d,rx)));", "ax=S(ax,M(M(q[z],d),rx));", count=1, label="west/east divx piece"
    )
    body = _replace_exact(
        body, "ay=S(ay,M(q[z],M(d,ry)));", "ay=S(ay,M(M(q[z],d),ry));", count=1, label="south/north divx piece"
    )
    body = _replace_exact(
        body,
        "float az=M(-1,M(S(VF(q,rrw,f+1,j,i,nk,nj,ni,ng,w,ep),VF(q,rrw,f,j,i,nk,nj,ni,ng,w,ep)),rds[f+1])),tot=A(A(ax,ay),az);",
        "float az=M(M(-S(VF(q,rrw,f+1,j,i,nk,nj,ni,ng,w,ep),VF(q,rrw,f,j,i,nk,nj,ni,ng,w,ep)),rdz),mf[f+1]),tot=A(A(ax,ay),az);",
        count=1,
        label="vertical divergence",
    )
    body = _replace_exact(
        body,
        "tot=A(tot,M(q[z],A(M(cc2,divx[I3(f,j,i,nj,ni)]),M(cc1,divx[I3(f-1,j,i,nj,ni)]))));",
        "float c2c=c2[I3(f,0,0,nj,ni)],c1c=S(1,c2c);tot=A(tot,M(q[z],A(M(c2c,divx[I3(f,j,i,nj,ni)]),M(c1c,divx[I3(f-1,j,i,nj,ni)]))));",
        count=1,
        label="column c1/c2",
    )
    body = _replace_exact(
        body,
        "float wt=A(ten[z],M(tot,M(rrf0[z],gz[(unsigned long long)j*ni+i]))),",
        "float wt=A(ten[z],M(tot,rrf0[I3(f,0,0,nj,ni)])),",
        count=1,
        label="column rrf0",
    )
    for token in ("rds[", "gz[", "rgz"):
        if token in body:
            raise RuntimeError(f"flat TKE advection kept terrain token {token!r}")
    return head + body


GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE = _flat_tke_advection_source()
GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE.encode()
).hexdigest()

_NVRTC_OPTIONS = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_MAX_COLUMN = 4096


def flat_stretched_closure_source(*, max_nk: int) -> str:
    """The closure text with its column workspace sized for ``max_nk``."""

    if isinstance(max_nk, bool) or not isinstance(max_nk, int) or not 8 <= max_nk <= _MAX_COLUMN:
        raise ValueError(f"flat closure column capacity must be an integer in [8, {_MAX_COLUMN}]")
    return f"#define GM1_FLAT_NK {max_nk}\n" + GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE


@dataclass(frozen=True, slots=True)
class CompiledFlatStretchedClosure:
    module: Any
    kernels: Mapping[str, Any]
    tke_module: Any
    tke_kernel: Any
    max_nk: int
    source_sha256: str
    tke_source_sha256: str


def compile_cuda_gm1_flat_stretched_closure(*, max_nk: int) -> CompiledFlatStretchedClosure:
    if hashlib.sha256(GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE.encode()).hexdigest() != GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256:
        raise RuntimeError("flat closure source differs from its pin")
    if hashlib.sha256(GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE.encode()).hexdigest() != GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256:
        raise RuntimeError("flat TKE advection source differs from its pin")
    cupy = importlib.import_module("cupy")
    source = flat_stretched_closure_source(max_nk=max_nk)
    # Not a cupy.RawModule: CuPy appends -ftz=true after these options, so
    # the image would flush CM1's binary32 sub-normals (nvrtc_module).
    module = compile_nvrtc_module(cupy, code=source, options=_NVRTC_OPTIONS,
                                  name_expressions=GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES,
                                  filename="gm1_flat_stretched_closure.cu")
    tke_module = compile_nvrtc_module(
        cupy,
        code=GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE,
        options=_NVRTC_OPTIONS,
        name_expressions=(GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME,),
        filename="gm1_flat_stretched_tke_advection.cu",
    )
    return CompiledFlatStretchedClosure(
        module,
        MappingProxyType({name: module.get_function(name) for name in GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES}),
        tke_module,
        tke_module.get_function(GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME),
        max_nk,
        hashlib.sha256(source.encode()).hexdigest(),
        GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256,
    )


def _blocks(words: int) -> tuple[tuple[int], tuple[int]]:
    threads = GM1_FLAT_STRETCHED_CLOSURE_THREADS
    return ((max(1, words) + threads - 1) // threads,), (threads,)


def enqueue_flat_calcdef(
    kernels: Mapping[str, Any], arrays: Mapping[str, Any], *, mh_profile: Any,
    ni: int, nj: int, nk: int, rdx: Any, rdy: Any, rdz: Any, cupy: Any, stream: Any,
) -> None:
    """calcdef's flat strain and deformation on one tile (turb.F:4766-5388)."""

    u = cupy.uint32
    kernels["gm1_flat_calcdef_mass_v1"](
        *_blocks(nk * (nj + 1) * (ni + 1)),
        (arrays["u3d"], arrays["v3d"], arrays["w3d"], arrays["rho"], mh_profile,
         arrays["s11"], arrays["s22"], arrays["s33"], arrays["s12"], u(ni), u(nj), u(nk), rdx, rdy, rdz),
        stream=stream,
    )
    kernels["gm1_flat_calcdef_face_v1"](
        *_blocks((nk - 1) * (nj + 1) * (ni + 1)),
        (arrays["u3d"], arrays["v3d"], arrays["w3d"], arrays["rf"], arrays["mf"],
         arrays["s13"], arrays["s23"], u(ni), u(nj), u(nk), rdx, rdy, rdz),
        stream=stream,
    )
    kernels["gm1_flat_calcdef_reduce_v1"](
        *_blocks((nk - 1) * nj * ni),
        (arrays["rf"], *(arrays[name] for name in ("c1", "c2", "s11", "s22", "s33", "s12", "s13", "s23", "defv", "defh")),
         u(ni), u(nj), u(nk)),
        stream=stream,
    )


def enqueue_flat_gettau(
    kernels: Mapping[str, Any], arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, cupy: Any, stream: Any,
) -> None:
    """gettau with free-slip lower/upper faces (turb.F:4094-4433)."""

    u = cupy.uint32
    dims = (u(ni), u(nj), u(nk))
    kernels["gm1_flat_gettau_normal_v1"](
        *_blocks(ni * nj * nk),
        (arrays["s11"], arrays["s22"], arrays["s33"], arrays["kmh"], arrays["t11"], arrays["t22"], arrays["t33"], *dims),
        stream=stream,
    )
    kernels["gm1_flat_gettau_t12_v1"](
        *_blocks(nk * (nj + 1) * (ni + 1)), (arrays["s12"], arrays["kmh"], arrays["t12"], *dims), stream=stream,
    )
    kernels["gm1_flat_gettau_t13_v1"](
        *_blocks((nk + 1) * nj * (ni + 1)), (arrays["s13"], arrays["kmv"], arrays["t13"], *dims), stream=stream,
    )
    kernels["gm1_flat_gettau_t23_v1"](
        *_blocks((nk + 1) * (nj + 1) * ni), (arrays["s23"], arrays["kmv"], arrays["t23"], *dims), stream=stream,
    )


def enqueue_flat_momentum_stress(
    kernels: Mapping[str, Any], arrays: Mapping[str, Any], *, mh_profile: Any,
    ni: int, nj: int, nk: int, rdx: Any, rdy: Any, rdz: Any, dt: Any, cupy: Any, stream: Any,
    two_part: Any = None, directions: tuple[str, ...] = ("u", "v", "w"),
) -> None:
    """turbu, turbv and turbw (solve1.F:1138-1150) on one tile.

    ``two_part`` is the sgsmodel=3/4 binding; its ``tfu``/``tfv``, ``u2pt``/
    ``v2pt`` and ``ntwk`` switch on the ``dot2p`` block.  Without it the
    carriers passed in their place are never read (``ntwk=0``).
    """

    u = cupy.uint32
    dims = (u(ni), u(nj), u(nk))
    common = (arrays["kmv"], arrays["rho"], arrays["rf"], arrays["mh"], arrays["mf"], mh_profile)
    if two_part is None:
        u_extra = (mh_profile, arrays["uten"], u(0))
        v_extra = (mh_profile, arrays["vten"], u(0))
    else:
        ntwk = int(two_part.ntwk)
        if not 2 <= ntwk < nk:
            raise ValueError("two-part ntwk must satisfy 2 <= ntwk < nk")
        u_extra = (two_part.tfu, two_part.u2pt, u(ntwk))
        v_extra = (two_part.tfv, two_part.v2pt, u(ntwk))
    if "u" in directions:
        kernels["gm1_flat_turbu_v1"](
            *_blocks(nj * (ni + 1)),
            (arrays["t11"], arrays["t12"], arrays["t13"], *common, arrays["u3d"], arrays["w3d"], arrays["uten"],
             *u_extra[:2], *dims, rdx, rdy, rdz, dt, u_extra[2]),
            stream=stream,
        )
    if "v" in directions:
        kernels["gm1_flat_turbv_v1"](
            *_blocks((nj + 1) * ni),
            (arrays["t12"], arrays["t22"], arrays["t23"], *common, arrays["v3d"], arrays["w3d"], arrays["vten"],
             *v_extra[:2], *dims, rdx, rdy, rdz, dt, v_extra[2]),
            stream=stream,
        )
    if "w" in directions:
        kernels["gm1_flat_turbw_v1"](
            *_blocks(nj * ni),
            (arrays["t13"], arrays["t23"], arrays["t33"], arrays["kmh"], arrays["rho"], arrays["rf"], arrays["mh"],
             arrays["mf"], arrays["w3d"], arrays["wten"], *dims, rdx, rdy, rdz, dt),
            stream=stream,
        )


def enqueue_flat_turbt(
    kernels: Mapping[str, Any], arrays: Mapping[str, Any], *, kmh: Any, kmv: Any,
    ni: int, nj: int, nk: int, rdx: Any, rdy: Any, rdz: Any, dt: Any, cupy: Any, stream: Any,
) -> None:
    """turbt on the TKE (solve1.F:362-372)."""

    u = cupy.uint32
    kernels["gm1_flat_turbt_v1"](
        *_blocks(nj * ni),
        (arrays["tke3d"], kmh, kmv, arrays["rho"], arrays["rf"], arrays["mh"], arrays["mf"], arrays["tketen"],
         u(ni), u(nj), u(nk), rdx, rdy, rdz, dt),
        stream=stream,
    )


__all__ = [
    "CompiledFlatStretchedClosure",
    "GM1_FLAT_STRETCHED_CLOSURE_ABI",
    "GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES",
    "GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE",
    "GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256",
    "GM1_FLAT_STRETCHED_CLOSURE_THREADS",
    "GM1_FLAT_STRETCHED_TKE_ADVECTION_ABI",
    "GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME",
    "GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE",
    "GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256",
    "compile_cuda_gm1_flat_stretched_closure",
    "enqueue_flat_calcdef",
    "enqueue_flat_gettau",
    "enqueue_flat_momentum_stress",
    "enqueue_flat_turbt",
    "flat_stretched_closure_source",
]
