"""CM1's closure statements beside a rigid side wall, derived from the flat closure text.

The flat closure text (:mod:`cm1core.cuda_gm1_flat_stretched_closure`) is
CM1 r21.1's ``terrain_flag=.false.`` closure, the branch CM1 runs on every
Cartesian column, even or stretched.  It forms every tile edge as CM1's
*open* side: the strain and stress copies of turb.F:5173-5290 and
:4295-4410 as an index clamp, and the zero edge tendencies of
turbtend.F:84-104, :887-913, :1200-1227, :1966-1993 and :2604-2632.  CM1
does neither beside a rigid wall (``wbc``/``ebc``/``sbc``/``nbc`` = 3 or 4):
there it takes the wall branches of calcdef and gettau and forms the edge
tendency from the wall flux the mirrored halo gives.

This module derives the wall text from the flat text by exact substitutions
(each found exactly the expected number of times, so a change to the flat
text cannot slip through), and the flat text itself is not touched: a run
without a wall launches the images it always has.  Every derived kernel takes
one more argument, ``walls``, the tile's four sides as nibbles
``west | east<<4 | south<<8 | north<<12``; a nibble is 3 (free slip) or 4
(no slip) on a tile edge that is a domain wall and 0 everywhere else (an open
or periodic side, or a tile edge inside the domain), where the derived text
forms the flat text's own words.

What changes beside a wall, with CM1's lines:

* calcdef ``s12`` (turb.F:4788-4876, then the open copies :5217-5288 that
  follow them): free slip copies the next face in, as an open side does;
  no slip writes ``2.0*va(1,j,k)*rdx*uf(1)*0.25*((rho(0,j-1)+rho(1,j))+
  (rho(0,j)+rho(1,j-1)))`` (west; ``-2.0*va(ni,j,k)``, ``2.0*ua(i,1,k)``,
  ``-2.0*ua(i,nj,k)`` east, south, north).  The south and north walls are
  written after the west and east ones and the open copies after both, so
  a corner face takes the value that order leaves there.
* calcdef ``s13``/``s23`` (turb.F:4895-4970): free slip copies the next
  face in; no slip writes ``2.0*wa(1,j,k)*rdx*uf(1)`` (and its twins);
  CM1 copies no open side here, so an open side keeps the computed face.
* gettau ``t12``, ``t13``, ``t23`` (turb.F:4131-4232): free slip copies the
  next face in; no slip keeps the computed face, ``s12*0.25*(kmh sums)``,
  with the eddy coefficients' mirrored halo (``bcw(kmh,1)``, turb.F:3166) at
  the wall.
* turbu ``turby`` beside a south or north wall, turbv ``turbx`` beside a
  west or east wall, turbw, turbt and turbs on both axes: the edge cell's
  tendency is CM1's ``-(flux(2)-flux(1))*rdx`` (and its twins) with the wall
  flux the mirrored halo gives, not the open side's zero.
* the TKE advection (adv.F advw with ``iamw`` false): no inflow edit at a
  wall face (adv.F:3037-3101 runs for ``wbc = 2`` only).

``uh = uf = vh = vf = 1`` on every grid the Cartesian model admits, so their factors
are the identity and are not formed, as in the flat text.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .cuda_gm1_flat_stretched_closure import (
    GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES,
    GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE,
    GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256,
    GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME,
    GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE,
    GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256,
)
from .nvrtc_module import compile_nvrtc_module

GM1_FLAT_WALL_CLOSURE_ABI = "gm1.cuda-flat-wall-closure-f32/v1"
GM1_FLAT_WALL_TKE_ADVECTION_ABI = "gm1.cuda-flat-wall-tke-advection-f32/v1"
GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES = tuple(
    name.replace("gm1_flat_", "gm1_flatwall_", 1) for name in GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES
)
GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME = GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME.replace(
    "gm1_flat_", "gm1_flatwall_", 1
)
#: CM1's rigid side values (README.namelist:561-568).
FREE_SLIP = 3
NO_SLIP = 4

SOURCE_ANCHORS = MappingProxyType({
    "calcdef_s12_walls": "src/turb.F:4788-4876",
    "calcdef_s13_s23_walls": "src/turb.F:4895-4970",
    "calcdef_open_copies": "src/turb.F:5173-5290",
    "gettau_walls": "src/turb.F:4131-4232",
    "gettau_open_copies": "src/turb.F:4295-4410",
    "eddy_coefficient_halo": "src/turb.F:3166-3187; src/bc.F:942-1143",
    "turbs_open_edge": "src/turbtend.F:84-104",
    "turbt_open_edge": "src/turbtend.F:887-913",
    "turbu_open_edge": "src/turbtend.F:1200-1227",
    "turbv_open_edge": "src/turbtend.F:1966-1993",
    "turbw_open_edge": "src/turbtend.F:2604-2632",
    "tke_advection_inflow_edits": "src/adv.F:3037-3101",
})


def _replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> str:
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"wall closure {label}: expected {count} site(s), found {found}")
    return source.replace(old, new)


# The four wall formulas and the side lookup, ahead of the first kernel.
_HELPERS = r'''// The tile's rigid sides: west | east<<4 | south<<8 | north<<12, each 3 (free
// slip), 4 (no slip) or 0 (an open or periodic side, or a tile edge inside the
// domain, where the flat text's own open-side words are formed).
__device__ __forceinline__ unsigned WS(unsigned walls,int side){return (walls>>(4*side))&15u;}
// turb.F:4801-4809 (west, i=1, sign 2.0, va(1,j,k)) and :4822-4830 (east,
// i=ni+1, sign -2.0, va(ni,j,k)):
//   s12(i,j,k) = sign*va(iv,j,k)*rdx*uf(i)*0.25*( (rho(i-1,j-1,k)+rho(i,j,k))+(rho(i-1,j,k)+rho(i,j-1,k)) )
// f is the tile face (0 or ni), jf the tile row face, uf(i) = 1.
__device__ __forceinline__ float S12WE(const float*v,const float*rho,int k,int f,int jf,int iv,float sgn,int NJ,int NI,float rdx){
 float rs=A(A(CELL(rho,k,jf-1,f-1,NJ,NI),CELL(rho,k,jf,f,NJ,NI)),A(CELL(rho,k,jf,f-1,NJ,NI),CELL(rho,k,jf-1,f,NJ,NI)));
 return M(M(M(M(sgn,v[I3(k,jf,iv,NJ+1,NI)]),rdx),0.25f),rs);}
// turb.F:4845-4853 (south, j=1, 2.0, ua(i,1,k)) and :4866-4874 (north,
// j=nj+1, -2.0, ua(i,nj,k)), vf(j) = 1.
__device__ __forceinline__ float S12SN(const float*u,const float*rho,int k,int fi,int f,int ju,float sgn,int NJ,int NI,float rdy){
 float rs=A(A(CELL(rho,k,f-1,fi-1,NJ,NI),CELL(rho,k,f,fi,NJ,NI)),A(CELL(rho,k,f,fi-1,NJ,NI),CELL(rho,k,f-1,fi,NJ,NI)));
 return M(M(M(M(sgn,u[I3(k,ju,fi,NJ,NI+1)]),rdy),0.25f),rs);}

'''


def _derive_closure_template() -> str:
    text = GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE
    if hashlib.sha256(text.encode()).hexdigest() != GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256:
        raise RuntimeError("the flat closure text differs from its pin; derive the wall text again")
    text = _replace_exact(text, "// gm1.cuda-flat-stretched-closure-f32/v1\n",
                          f"// {GM1_FLAT_WALL_CLOSURE_ABI}\n", count=1, label="header")
    text = _replace_exact(text, "\n// turb.F:4772-4786 with the four open-side s12 copies",
                          "\n" + _HELPERS + "// turb.F:4772-4786 with the four open-side s12 copies",
                          count=1, label="helpers")
    text = _replace_exact(text, 'extern "C" __global__ void gm1_flat_', 'extern "C" __global__ void gm1_flatwall_',
                          count=len(GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES), label="kernel names")
    # Every kernel takes the tile's sides last.
    for old, count in (
        ("unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz){\n", 2),   # calcdef mass, face
        ("float*defv,float*defh,\n unsigned ni,unsigned nj,unsigned nk){\n", 1),       # calcdef reduce
        ("float*t11,float*t22,float*t33,unsigned ni,unsigned nj,unsigned nk){\n", 1),  # gettau normal
        ("float*t12,unsigned ni,unsigned nj,unsigned nk){\n", 1),                      # gettau t12
        ("float*t13,unsigned ni,unsigned nj,unsigned nk){\n", 1),                      # gettau t13
        ("float*t23,unsigned ni,unsigned nj,unsigned nk){\n", 1),                      # gettau t23
        ("float rdx,float rdy,float rdz,float dt,unsigned ntwk){\n", 2),               # turbu, turbv
        ("const float*w,float*wten,unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt){\n", 1),
        ("float*ten,\n unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float dt){\n", 1),  # turbt
        ("float*fx,float*fy,unsigned ni,unsigned nj,unsigned nk,float dx,float dy){\n", 1),  # turbs fluxes
        ("unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz,float dt,unsigned iflux){\n", 1),
    ):
        text = _replace_exact(text, old, old[: -len("){\n")] + ",unsigned walls){\n", count=count,
                              label="signature")

    # calcdef s12: turb.F:4788-4876 then :5217-5288, in CM1's order (see the module docstring).
    text = _replace_exact(
        text,
        "  int ii=CL(ic,1,NI-1),jj=CL(jc,1,NJ-1);\n"
        "  float x=M(S(u[I3(k,jj,ii,NJ,NI+1)],u[I3(k,jj-1,ii,NJ,NI+1)]),rdy);\n"
        "  float y=M(S(v[I3(k,jj,ii,NJ+1,NI)],v[I3(k,jj,ii-1,NJ+1,NI)]),rdx);\n"
        "  float rs=A(A(rho[I3(k,jj-1,ii-1,NJ,NI)],rho[I3(k,jj,ii,NJ,NI)]),A(rho[I3(k,jj,ii-1,NJ,NI)],rho[I3(k,jj-1,ii,NJ,NI)]));\n"
        "  s12[I3(k,jc,ic,NJ+1,NI+1)]=M(M(M(0.5f,A(x,y)),0.25f),rs);}}\n",
        "  int ii=CL(ic,1,NI-1),jj=CL(jc,1,NJ-1);\n"
        "  unsigned row=(jc==0)?WS(walls,2):((jc==NJ)?WS(walls,3):0u),col=(ic==0)?WS(walls,0):((ic==NI)?WS(walls,1):0u);\n"
        "  float s;\n"
        "  if(row==4u){int fi=col?ic:ii;s=(jc==0)?S12SN(u,rho,k,fi,0,0,2.0f,NJ,NI,rdy):S12SN(u,rho,k,fi,NJ,NJ-1,-2.0f,NJ,NI,rdy);}\n"
        "  else if(col==4u){s=(ic==0)?S12WE(v,rho,k,0,jj,0,2.0f,NJ,NI,rdx):S12WE(v,rho,k,NI,jj,NI-1,-2.0f,NJ,NI,rdx);}\n"
        "  else{\n"
        "  float x=M(S(u[I3(k,jj,ii,NJ,NI+1)],u[I3(k,jj-1,ii,NJ,NI+1)]),rdy);\n"
        "  float y=M(S(v[I3(k,jj,ii,NJ+1,NI)],v[I3(k,jj,ii-1,NJ+1,NI)]),rdx);\n"
        "  float rs=A(A(rho[I3(k,jj-1,ii-1,NJ,NI)],rho[I3(k,jj,ii,NJ,NI)]),A(rho[I3(k,jj,ii-1,NJ,NI)],rho[I3(k,jj-1,ii,NJ,NI)]));\n"
        "  s=M(M(M(0.5f,A(x,y)),0.25f),rs);}\n"
        "  s12[I3(k,jc,ic,NJ+1,NI+1)]=s;}}\n",
        count=1, label="calcdef s12",
    )
    # calcdef s13 and s23: turb.F:4895-4970.  Free slip forms the next face in;
    # no slip is 2.0*wa(1,j,k)*rdx*uf(1) and its twins.
    text = _replace_exact(
        text,
        "  if(jc<NJ){int il=C(ic-1,NI),ir=C(ic,NI);\n"
        "   float dw=M(S(w[I3(kf,jc,ir,NJ,NI)],w[I3(kf,jc,il,NJ,NI)]),rdx);\n"
        "   float du=M(M(M(S(u[I3(kf,jc,ic,NJ,NI+1)],u[I3(kf-1,jc,ic,NJ,NI+1)]),rdz),0.5f),A(mf[I3(kf,jc,il,NJ,NI)],mf[I3(kf,jc,ir,NJ,NI)]));\n"
        "   s13[I3(kf,jc,ic,NJ,NI+1)]=M(M(M(0.5f,A(dw,du)),0.5f),A(rf[I3(kf,jc,il,NJ,NI)],rf[I3(kf,jc,ir,NJ,NI)]));}\n",
        "  if(jc<NJ){unsigned side=(ic==0)?WS(walls,0):((ic==NI)?WS(walls,1):0u);\n"
        "   if(side==4u){s13[I3(kf,jc,ic,NJ,NI+1)]=(ic==0)?M(M(2.0f,w[I3(kf,jc,0,NJ,NI)]),rdx):M(M(-2.0f,w[I3(kf,jc,NI-1,NJ,NI)]),rdx);}\n"
        "   else{int fc=(side==3u)?((ic==0)?1:NI-1):ic;int il=C(fc-1,NI),ir=C(fc,NI);\n"
        "   float dw=M(S(w[I3(kf,jc,ir,NJ,NI)],w[I3(kf,jc,il,NJ,NI)]),rdx);\n"
        "   float du=M(M(M(S(u[I3(kf,jc,fc,NJ,NI+1)],u[I3(kf-1,jc,fc,NJ,NI+1)]),rdz),0.5f),A(mf[I3(kf,jc,il,NJ,NI)],mf[I3(kf,jc,ir,NJ,NI)]));\n"
        "   s13[I3(kf,jc,ic,NJ,NI+1)]=M(M(M(0.5f,A(dw,du)),0.5f),A(rf[I3(kf,jc,il,NJ,NI)],rf[I3(kf,jc,ir,NJ,NI)]));}}\n",
        count=1, label="calcdef s13",
    )
    text = _replace_exact(
        text,
        "  if(ic<NI){int js=C(jc-1,NJ),jn=C(jc,NJ);\n"
        "   float dw=M(S(w[I3(kf,jn,ic,NJ,NI)],w[I3(kf,js,ic,NJ,NI)]),rdy);\n"
        "   float dv=M(M(M(S(v[I3(kf,jc,ic,NJ+1,NI)],v[I3(kf-1,jc,ic,NJ+1,NI)]),rdz),0.5f),A(mf[I3(kf,js,ic,NJ,NI)],mf[I3(kf,jn,ic,NJ,NI)]));\n"
        "   s23[I3(kf,jc,ic,NJ+1,NI)]=M(M(M(0.5f,A(dw,dv)),0.5f),A(rf[I3(kf,js,ic,NJ,NI)],rf[I3(kf,jn,ic,NJ,NI)]));}}}\n",
        "  if(ic<NI){unsigned side=(jc==0)?WS(walls,2):((jc==NJ)?WS(walls,3):0u);\n"
        "   if(side==4u){s23[I3(kf,jc,ic,NJ+1,NI)]=(jc==0)?M(M(2.0f,w[I3(kf,0,ic,NJ,NI)]),rdy):M(M(-2.0f,w[I3(kf,NJ-1,ic,NJ,NI)]),rdy);}\n"
        "   else{int fc=(side==3u)?((jc==0)?1:NJ-1):jc;int js=C(fc-1,NJ),jn=C(fc,NJ);\n"
        "   float dw=M(S(w[I3(kf,jn,ic,NJ,NI)],w[I3(kf,js,ic,NJ,NI)]),rdy);\n"
        "   float dv=M(M(M(S(v[I3(kf,fc,ic,NJ+1,NI)],v[I3(kf-1,fc,ic,NJ+1,NI)]),rdz),0.5f),A(mf[I3(kf,js,ic,NJ,NI)],mf[I3(kf,jn,ic,NJ,NI)]));\n"
        "   s23[I3(kf,jc,ic,NJ+1,NI)]=M(M(M(0.5f,A(dw,dv)),0.5f),A(rf[I3(kf,js,ic,NJ,NI)],rf[I3(kf,jn,ic,NJ,NI)]));}}}}\n",
        count=1, label="calcdef s23",
    )
    # gettau t12: free slip and open sides take the next face in (turb.F:4131-
    # 4173, :4339-4410); a no-slip wall keeps its own face, with the mirrored
    # kmh halo (an index clamp) at the wall.
    text = _replace_exact(
        text,
        "  int ii=CL(ic,1,NI-1),jj=CL(jc,1,NJ-1);\n"
        "  float lo=A(A(kmh[I3(k,jj-1,ii-1,NJ,NI)],kmh[I3(k,jj,ii,NJ,NI)]),A(kmh[I3(k,jj,ii-1,NJ,NI)],kmh[I3(k,jj-1,ii,NJ,NI)]));\n"
        "  float hi=A(A(kmh[I3(k+1,jj-1,ii-1,NJ,NI)],kmh[I3(k+1,jj,ii,NJ,NI)]),A(kmh[I3(k+1,jj,ii-1,NJ,NI)],kmh[I3(k+1,jj-1,ii,NJ,NI)]));\n",
        "  int ii=(ic==0&&WS(walls,0)==4u)?0:((ic==NI&&WS(walls,1)==4u)?NI:CL(ic,1,NI-1));\n"
        "  int jj=(jc==0&&WS(walls,2)==4u)?0:((jc==NJ&&WS(walls,3)==4u)?NJ:CL(jc,1,NJ-1));\n"
        "  float lo=A(A(CELL(kmh,k,jj-1,ii-1,NJ,NI),CELL(kmh,k,jj,ii,NJ,NI)),A(CELL(kmh,k,jj,ii-1,NJ,NI),CELL(kmh,k,jj-1,ii,NJ,NI)));\n"
        "  float hi=A(A(CELL(kmh,k+1,jj-1,ii-1,NJ,NI),CELL(kmh,k+1,jj,ii,NJ,NI)),A(CELL(kmh,k+1,jj,ii-1,NJ,NI),CELL(kmh,k+1,jj-1,ii,NJ,NI)));\n",
        count=1, label="gettau t12",
    )
    # gettau t13/t23: a free-slip wall face takes the next face in (turb.F:
    # 4188-4232); the free-slip bottom and top copies then read it.
    text = _replace_exact(
        text,
        "  t13[q]=M(s13[I3(ff,j,i,NJ,NI+1)],A(CELL(kmv,ff,j,i-1,NJ,NI),CELL(kmv,ff,j,i,NJ,NI)));}}\n",
        "  int fc=(i==0&&WS(walls,0)==3u)?1:((i==NI&&WS(walls,1)==3u)?NI-1:i);\n"
        "  t13[q]=M(s13[I3(ff,j,fc,NJ,NI+1)],A(CELL(kmv,ff,j,fc-1,NJ,NI),CELL(kmv,ff,j,fc,NJ,NI)));}}\n",
        count=1, label="gettau t13",
    )
    text = _replace_exact(
        text,
        "  t23[q]=M(s23[I3(ff,j,i,NJ+1,NI)],A(CELL(kmv,ff,j-1,i,NJ,NI),CELL(kmv,ff,j,i,NJ,NI)));}}\n",
        "  int fc=(j==0&&WS(walls,2)==3u)?1:((j==NJ&&WS(walls,3)==3u)?NJ-1:j);\n"
        "  t23[q]=M(s23[I3(ff,fc,i,NJ+1,NI)],A(CELL(kmv,ff,fc-1,i,NJ,NI),CELL(kmv,ff,fc,i,NJ,NI)));}}\n",
        count=1, label="gettau t23",
    )
    # The edge tendencies: zero beside an open side only (turbtend.F:1200-1227,
    # :1966-1993, :2604-2632, :887-913, :84-104 test wbc.eq.2).
    text = _replace_exact(
        text,
        "  if(f==0||f==NI)hx[k]=0.0f;\n  if(j==0||j==NJ-1)hy[k]=0.0f;}\n",
        "  if(f==0||f==NI)hx[k]=0.0f;\n  if((j==0&&!WS(walls,2))||(j==NJ-1&&!WS(walls,3)))hy[k]=0.0f;}\n",
        count=1, label="turbu edges",
    )
    text = _replace_exact(
        text,
        "  if(i==0||i==NI-1)hx[k]=0.0f;\n  if(f==0||f==NJ)hy[k]=0.0f;}\n",
        "  if((i==0&&!WS(walls,0))||(i==NI-1&&!WS(walls,1)))hx[k]=0.0f;\n  if(f==0||f==NJ)hy[k]=0.0f;}\n",
        count=1, label="turbv edges",
    )
    text = _replace_exact(
        text,
        "  if(i==0||i==NI-1)hx[f]=0.0f;\n  if(j==0||j==NJ-1)hy[f]=0.0f;}\n",
        "  if((i==0&&!WS(walls,0))||(i==NI-1&&!WS(walls,1)))hx[f]=0.0f;\n"
        "  if((j==0&&!WS(walls,2))||(j==NJ-1&&!WS(walls,3)))hy[f]=0.0f;}\n",
        count=2, label="turbw and turbt edges",
    )
    text = _replace_exact(
        text,
        "  float tx=-M(S(fx[xp+1u],fx[xp]),rx);if(i==0||i+1==NI)tx=0.0f;\n"
        "  float ty=-M(S(fy[yp+(unsigned long long)ni],fy[yp]),ry);if(j==0||j+1==NJ)ty=0.0f;\n",
        "  float tx=-M(S(fx[xp+1u],fx[xp]),rx);if((i==0&&!WS(walls,0))||(i+1==NI&&!WS(walls,1)))tx=0.0f;\n"
        "  float ty=-M(S(fy[yp+(unsigned long long)ni],fy[yp]),ry);if((j==0&&!WS(walls,2))||(j+1==NJ&&!WS(walls,3)))ty=0.0f;\n",
        count=1, label="turbs edges",
    )
    for token in ("gm1_flat_", ):
        if token in text:
            raise RuntimeError(f"wall closure text kept {token!r}")
    return text


def _derive_tke_advection() -> str:
    text = GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE
    if hashlib.sha256(text.encode()).hexdigest() != GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE_SHA256:
        raise RuntimeError("the flat TKE advection text differs from its pin; derive the wall text again")
    text = _replace_exact(text, "// gm1.cuda-flat-stretched-tke-advection-f32/v1\n",
                          f"// {GM1_FLAT_WALL_TKE_ADVECTION_ABI}\n", count=1, label="TKE header")
    text = _replace_exact(
        text,
        f'extern "C" __global__ void {GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME}(',
        f'extern "C" __global__ void {GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME}(',
        count=1, label="TKE kernel name",
    )
    text = _replace_exact(text, "const float*mf,float rdz){", "const float*mf,float rdz,unsigned walls){",
                          count=1, label="TKE signature")
    # The fifth-order and WENO stencils read three cells beyond the tile.  CM1
    # fills the TKE's halo with bcw (bc.F:942-1143): a copy of the edge cell
    # on an open side, which the flat text's index clamp is, but a mirror on a
    # rigid wall, tke(1-i) = tke(i).  The derived text reads the mirror there.
    text = _replace_exact(
        text,
        "__device__ __forceinline__ float HF(const float*q,",
        "__device__ __forceinline__ unsigned WS(unsigned walls,int side){return (walls>>(4*side))&15u;}\n"
        "// bc.F:942-1143 beside a wall: index -1-x below the low edge, 2n-1-x above the high one.\n"
        "__device__ __forceinline__ int WR(int x,int n,unsigned lo,unsigned hi){"
        "if(x<0)return lo?-1-x:0;if(x>=n)return hi?2*n-1-x:n-1;return x;}\n"
        "__device__ __forceinline__ float QW(const float*a,int k,int j,int i,int nj,int ni,unsigned walls){"
        "return a[I3(k,WR(j,nj,WS(walls,2),WS(walls,3)),WR(i,ni,WS(walls,0),WS(walls,1)),nj,ni)];}\n"
        "__device__ __forceinline__ float HF(const float*q,",
        count=1, label="TKE mirrored reader",
    )
    text = _replace_exact(text, "int w,double ep,int ov,float*mass){float cc,m;",
                          "int w,double ep,int ov,float*mass,unsigned walls){float cc,m;", count=1,
                          label="HF signature")
    for face in ("1", "ni-1", "nj-1"):
        old = f"return HF(q,c2,rru,rrv,f,j,{face},axis,ni,nj,ng,w,ep,0,mass);"
        text = _replace_exact(text, old, old[:-2] + ",walls);", count=2 if face == "1" else 1,
                              label="HF recursion")
    head, rest = text.split("__device__ __forceinline__ float HFO(", 1)
    start = head.index("__device__ __forceinline__ float HF(const float*q,")
    body = head[start:]
    if body.count("CELL(q,") != 20:
        raise RuntimeError(f"wall TKE advection: expected 20 stencil reads in HF, found {body.count('CELL(q,')}")
    body, moved = re.subn(r"CELL\(q,(f,[a-z0-9+\-]+,[a-z0-9+\-]+),nj,ni\)", r"QW(q,\1,nj,ni,walls)", body)
    if moved != 20 or "CELL(q," in body:
        raise RuntimeError(f"wall TKE advection: moved {moved} of the 20 stencil reads to the mirrored reader")
    text = head[:start] + body + "__device__ __forceinline__ float HFO(" + rest
    text = _replace_exact(
        text,
        "int w,double ep,int ov,float*mass){float boundary_mass=HFM(c2,rru,rrv,f,j,face,axis,ni,nj,ng);"
        "float flux=HF(q,c2,rru,rrv,f,j,face,axis,ni,nj,ng,w,ep,ov,mass);",
        "int w,double ep,int ov,float*mass,unsigned walls){float boundary_mass=HFM(c2,rru,rrv,f,j,face,axis,ni,nj,ng);"
        "float flux=HF(q,c2,rru,rrv,f,j,face,axis,ni,nj,ng,w,ep,ov,mass,walls);",
        count=1, label="HFO",
    )
    # adv.F's doitw..doitn: the inflow face copy and the divx piece on an open side only.
    text = _replace_exact(
        text,
        "fx0=HF(q,c2,rru,rrv,f,j,i,0,ni,nj,ng,w,ep,1,&mw),fx1=HF(q,c2,rru,rrv,f,j,i+1,0,ni,nj,ng,w,ep,1,&me),"
        "fy0=HF(q,c2,rru,rrv,f,i,j,1,ni,nj,ng,w,ep,1,&ms),fy1=HF(q,c2,rru,rrv,f,i,j+1,1,ni,nj,ng,w,ep,1,&mn)",
        "fx0=HF(q,c2,rru,rrv,f,j,i,0,ni,nj,ng,w,ep,(i==0&&WS(walls,0))?0:1,&mw,walls),"
        "fx1=HF(q,c2,rru,rrv,f,j,i+1,0,ni,nj,ng,w,ep,(i+1==(int)ni&&WS(walls,1))?0:1,&me,walls),"
        "fy0=HF(q,c2,rru,rrv,f,i,j,1,ni,nj,ng,w,ep,(j==0&&WS(walls,2))?0:1,&ms,walls),"
        "fy1=HF(q,c2,rru,rrv,f,i,j+1,1,ni,nj,ng,w,ep,(j+1==(int)nj&&WS(walls,3))?0:1,&mn,walls)",
        count=1, label="TKE inflow face copies",
    )
    text = _replace_exact(text, "if((i==0&&mw>=0)||(i==(int)ni-1&&me<=0))",
                          "if((i==0&&!WS(walls,0)&&mw>=0)||(i==(int)ni-1&&!WS(walls,1)&&me<=0))",
                          count=1, label="TKE west/east divx piece")
    text = _replace_exact(text, "if((j==0&&ms>=0)||(j==(int)nj-1&&mn<=0))",
                          "if((j==0&&!WS(walls,2)&&ms>=0)||(j==(int)nj-1&&!WS(walls,3)&&mn<=0))",
                          count=1, label="TKE south/north divx piece")
    return text


GM1_FLAT_WALL_CLOSURE_TEMPLATE = _derive_closure_template()
GM1_FLAT_WALL_CLOSURE_TEMPLATE_SHA256 = hashlib.sha256(GM1_FLAT_WALL_CLOSURE_TEMPLATE.encode()).hexdigest()
GM1_FLAT_WALL_TKE_ADVECTION_SOURCE = _derive_tke_advection()
GM1_FLAT_WALL_TKE_ADVECTION_SOURCE_SHA256 = hashlib.sha256(GM1_FLAT_WALL_TKE_ADVECTION_SOURCE.encode()).hexdigest()

_NVRTC_OPTIONS = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_MAX_COLUMN = 4096


def flat_wall_closure_source(*, max_nk: int) -> str:
    """The wall closure text with its column workspace sized for ``max_nk``."""

    if isinstance(max_nk, bool) or not isinstance(max_nk, int) or not 8 <= max_nk <= _MAX_COLUMN:
        raise ValueError(f"wall closure column capacity must be an integer in [8, {_MAX_COLUMN}]")
    return f"#define GM1_FLAT_NK {max_nk}\n" + GM1_FLAT_WALL_CLOSURE_TEMPLATE


def has_rigid_wall(sides: Any) -> bool:
    """Is any of CM1's ``(wbc, ebc, sbc, nbc)`` a rigid wall (3 or 4)?"""

    return any(int(side) in (FREE_SLIP, NO_SLIP) for side in sides)


def wall_code(sides: Any, *, west: bool = True, east: bool = True, south: bool = True, north: bool = True) -> int:
    """The ``walls`` word of a tile: each side's nibble where that tile edge is a domain edge.

    ``sides`` is CM1's ``(wbc, ebc, sbc, nbc)``.  ``west`` and the rest say
    whether the tile's edge lies on the domain's edge; a tile edge inside the
    domain is 0, as is every side that is not a rigid wall.
    """

    code = 0
    for index, (side, edge) in enumerate(zip(sides, (west, east, south, north))):
        side = int(side)
        if edge and side in (FREE_SLIP, NO_SLIP):
            code |= side << (4 * index)
    return code


def tile_wall_code(sides: Any, spec: Any, *, nx: int, ny: int) -> int:
    """:func:`wall_code` for one closure tile (``spec`` is a ``Gm1TileSpec``)."""

    return wall_code(
        sides,
        west=int(spec.ci0) == 0,
        east=int(spec.ci0) + int(spec.cnx) == int(nx),
        south=int(spec.cj0) == 0,
        north=int(spec.cj0) + int(spec.cny) == int(ny),
    )


class _WallKernel:
    """A derived kernel with the tile's ``walls`` word appended to every launch.

    The flat launchers (``enqueue_flat_*``) and the budget replays call the
    flat kernels by name with the flat text's own arguments; this binds the
    one argument the derived text adds, so no call site changes.
    """

    __slots__ = ("_kernel", "_walls", "_cupy")

    def __init__(self, kernel: Any, walls: int, cupy: Any) -> None:
        self._kernel, self._walls, self._cupy = kernel, int(walls), cupy

    def __call__(self, grid: Any, block: Any, arguments: Any, **kwargs: Any) -> Any:
        return self._kernel(grid, block, (*tuple(arguments), self._cupy.uint32(self._walls)), **kwargs)


@dataclass
class CompiledFlatWallClosure:
    """The derived images, answering ``kernels``/``tke_kernel`` for the tile bound last.

    It has the shape of
    :class:`cm1core.cuda_gm1_flat_stretched_closure.CompiledFlatStretchedClosure`:
    ``kernels`` is keyed by the flat names, so the flat launchers run the
    derived text unchanged once :meth:`bind` has named the tile's sides.
    """

    module: Any
    raw_kernels: Mapping[str, Any]
    tke_module: Any
    raw_tke_kernel: Any
    max_nk: int
    source_sha256: str
    tke_source_sha256: str
    cupy: Any
    walls: int = 0
    _bound: dict[int, tuple[Mapping[str, Any], Any]] = field(default_factory=dict)

    def bind(self, walls: int) -> None:
        self.walls = int(walls)

    @property
    def kernels(self) -> Mapping[str, Any]:
        return self._views()[0]

    @property
    def tke_kernel(self) -> Any:
        return self._views()[1]

    def _views(self) -> tuple[Mapping[str, Any], Any]:
        views = self._bound.get(self.walls)
        if views is None:
            mapping = MappingProxyType({
                flat: _WallKernel(self.raw_kernels[wall], self.walls, self.cupy)
                for flat, wall in zip(GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES, GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES)
            })
            views = self._bound[self.walls] = (mapping, _WallKernel(self.raw_tke_kernel, self.walls, self.cupy))
        return views


def compile_cuda_gm1_flat_wall_closure(*, max_nk: int) -> CompiledFlatWallClosure:
    if hashlib.sha256(GM1_FLAT_WALL_CLOSURE_TEMPLATE.encode()).hexdigest() != GM1_FLAT_WALL_CLOSURE_TEMPLATE_SHA256:
        raise RuntimeError("wall closure source differs from its pin")
    cupy = importlib.import_module("cupy")
    source = flat_wall_closure_source(max_nk=max_nk)
    # Not a cupy.RawModule: CuPy appends -ftz=true after these options (nvrtc_module).
    module = compile_nvrtc_module(cupy, code=source, options=_NVRTC_OPTIONS,
                                  name_expressions=GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES,
                                  filename="gm1_flat_wall_closure.cu")
    tke_module = compile_nvrtc_module(
        cupy, code=GM1_FLAT_WALL_TKE_ADVECTION_SOURCE, options=_NVRTC_OPTIONS,
        name_expressions=(GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME,),
        filename="gm1_flat_wall_tke_advection.cu",
    )
    return CompiledFlatWallClosure(
        module,
        MappingProxyType({name: module.get_function(name) for name in GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES}),
        tke_module,
        tke_module.get_function(GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME),
        max_nk,
        hashlib.sha256(source.encode()).hexdigest(),
        GM1_FLAT_WALL_TKE_ADVECTION_SOURCE_SHA256,
        cupy,
    )


__all__ = [
    "CompiledFlatWallClosure",
    "FREE_SLIP",
    "GM1_FLAT_WALL_CLOSURE_ABI",
    "GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES",
    "GM1_FLAT_WALL_CLOSURE_TEMPLATE",
    "GM1_FLAT_WALL_CLOSURE_TEMPLATE_SHA256",
    "GM1_FLAT_WALL_TKE_ADVECTION_ABI",
    "GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME",
    "GM1_FLAT_WALL_TKE_ADVECTION_SOURCE",
    "GM1_FLAT_WALL_TKE_ADVECTION_SOURCE_SHA256",
    "NO_SLIP",
    "SOURCE_ANCHORS",
    "compile_cuda_gm1_flat_wall_closure",
    "flat_wall_closure_source",
    "has_rigid_wall",
    "tile_wall_code",
    "wall_code",
]
