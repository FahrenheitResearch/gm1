"""Profile-aware scalar advection and moisture pdef for the real GM1 child.

CM1's terrain scalar equations use ``rdsf(k)`` for every mass-level vertical
flux divergence.  The older CUDA slice reconstructed that value from one
uniform ``dz``.  That is bit-identical only on the legacy uniform grid and is
invalid for the admitted 15 km ``stretch_z=1`` child.  This module patches the
sealed legacy kernel texts at their exact vertical-metric seams, requires the
already-admitted device profiles, and exposes same-stream replacement calls.

The entry points intentionally do not advance a complete RK stage.  Packing,
open-boundary halos, A/current updates, and cadence remain owned by the sealed
scalar/moist RK state machine; this module replaces only its scalar advection
and final-stage pdef numerical families.  A coupled child composer must remain
fail-closed unless both replacements are wired at the official solve2 order.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_advection5 import (
    CM1_NRKMAX,
    scalar_open_boundary_code,
    CUDA_ADVECTION5_KERNEL_NAMES,
    CUDA_ADVECTION5_SOURCE,
    CUDA_ADVECTION5_SOURCE_SHA256,
    CUDA_ADVECTION5_THREADS,
    CompiledAdvection5Kernels,
    CudaScalarAdvection5Launch,
    _canonical_arch as _advection_arch,
    _require_cuda_array,
    _validated_grid,
    _validated_spacing,
    scalar_advection_uses_weno,
)
from cm1core.cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_PROOF_SHA256,
    generated_vertical_coordinate,
    require_generated_vertical,
    Gm1ChildStretchedMetrics,
    stretched_vertical_identity_defect,
)
from cm1core.cuda_scalar_moist_rk import (
    CM1_QMAG_F32,
    CUDA_SCALAR_MOIST_RK_KERNEL_NAMES,
    CUDA_SCALAR_MOIST_RK_SOURCE,
    CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
    CUDA_SCALAR_MOIST_RK_THREADS,
    CompiledScalarMoistRKKernels,
    _require_cuda_f32,
)
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


GM1_CHILD_STRETCHED_SCALAR_SCHEMA = "gm1.cuda-child-stretched-scalar/v1"
#: The ``final-rk-moisture-pdef`` receipt's shape.  v2 adds ``selection.numq``
#: (the axis ``species`` indexes) and ``selection.pdefweno``; the
#: ``scalar-advection5`` receipt is unchanged and stays on v1.
GM1_CHILD_STRETCHED_SCALAR_PDEF_SCHEMA = "gm1.cuda-child-stretched-scalar/v2"
GM1_CHILD_STRETCHED_SCALAR_ADVECTION_ABI = (
    "gm1.cuda-child-stretched-scalar-advection5-f32/v1"
)
GM1_CHILD_STRETCHED_SCALAR_PDEF_ABI = (
    "gm1.cuda-child-stretched-scalar-pdef-f32/v1"
)
CM1_ADV_F_SHA256 = "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
CM1_ADV_ROUTINES_F_SHA256 = (
    "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00"
)
CM1_PDEF_F_SHA256 = "815cb7729c441b07c8d171596543b15ec630ab0c87eda5ab59a95d4eab2ae670"
CM1_SOLVE2_F_SHA256 = "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"


def _replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"legacy CUDA source no longer matches {seam} seam")
    return source.replace(old, new, 1)


def _stretched_advection_source() -> str:
    source = _replace_once(
        CUDA_ADVECTION5_SOURCE,
        "// gpuwm-cm1.cuda-advection5-kernels/v1",
        f"// {GM1_CHILD_STRETCHED_SCALAR_ADVECTION_ABI}",
        seam="advection ABI",
    )
    source = _replace_once(
        source,
        "const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){",
        "const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,const float*sigma,const float*sigmaf,const float*rds,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float ztop){",
        seam="terrain vertical mass signature",
    )
    source = _replace_once(
        source,
        "float zt=m32((float)nk,dz),rzt=d32(1.0f,zt);",
        "float zt=ztop,rzt=d32(1.0f,zt);",
        seam="terrain physical top",
    )
    source = _replace_once(
        source,
        "float sigmaf=m32((float)(k-1u),dz),sigma=m32((float)k-1.5f,dz),r2=m32(s32(sigmaf,sigma),d32(1.0f,dz)),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);",
        "float sf=sigmaf[k],sm=sigma[k-1u],r2=m32(s32(sf,sm),rds[k]),r1=m32(0.5f,s32(1.0f,r2));r2=m32(0.5f,r2);",
        seam="terrain vertical interpolation",
    )
    source = _replace_once(
        source,
        "s32(sigmaf,zt)",
        "s32(sf,zt)",
        seam="terrain-height factor",
    )
    source = _replace_once(
        source,
        "const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz){",
        "const float* rru,const float* rrv,const float* rrw,const float*rdsf,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy){",
        seam="terrain divergence signature",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);\n for",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);\n for",
        seam="terrain divergence metric",
    )
    source = _replace_once(
        source,
        "),rdsf));divx[z]",
        "),rdsf[k]));divx[z]",
        seam="terrain divergence derivative",
    )
    source = _replace_once(
        source,
        "const float* a,const float* fx,const float* fy,const float* fz,const float* divx,const float* rr0,const float* gz,float* tendency,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,const float* rru,const float* rrv,unsigned int obc){",
        "const float* a,const float* fx,const float* fy,const float* fz,const float* divx,const float* rr0,const float* gz,const float*rdsf,float* tendency,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,const float* rru,const float* rrv,unsigned int obc){",
        seam="scalar tendency signature",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdsf=d32(1.0f,dz);for(;q<total;q+=step)",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<total;q+=step)",
        seam="scalar tendency metric",
    )
    source = source.replace(",rdsf);else if(k==nk)", ",rdsf[k]);else if(k==nk)", 1)
    source = source.replace(",rdsf);else az=", ",rdsf[k]);else az=", 1)
    source = source.replace(",rdsf);float total_adv=", ",rdsf[k]);float total_adv=", 1)
    if source.count("rdsf[k]") < 4:
        raise RuntimeError("scalar stretched derivative patch is incomplete")
    if "float dz" in source or "rdz" in source or "rdsf=d32" in source:
        raise RuntimeError("uniform-z arithmetic survived stretched scalar patch")
    return source


def _stretched_rk_source() -> str:
    source = _replace_once(
        CUDA_SCALAR_MOIST_RK_SOURCE,
        "// gpuwm-cm1.cuda-scalar-moist-rk-kernels/v1",
        f"// {GM1_CHILD_STRETCHED_SCALAR_PDEF_ABI}",
        seam="pdef ABI",
    )
    source = _replace_once(
        source,
        "const float* fx,const float* fy,const float* fz,float* ax,float* ay,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,const float* s,const float* rru,const float* rrv,unsigned int obc){",
        "const float* fx,const float* fy,const float* fz,const float*rdsf,float* ax,float* ay,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,const float* s,const float* rru,const float* rrv,unsigned int obc){",
        seam="pdef directional signature",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=step)",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<n;q+=step)",
        seam="pdef directional metric",
    )
    source = source.replace(",rdz):((k==nk)?", ",rdsf[k]):((k==nk)?", 1)
    source = source.replace(",rdz):m32(s32(fz", ",rdsf[k]):m32(s32(fz", 1)
    source = source.replace(",rdz));}}", ",rdsf[k]));}}", 1)
    source = _replace_once(
        source,
        "const float* a,const float* s,const float* rho0,const float* gz,const float* rrw,const float* az,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dz){",
        "const float* a,const float* s,const float* rho0,const float* gz,const float* rrw,const float* az,const float*rdsf,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt){",
        seam="pdef z candidate signature",
    )
    source = _replace_once(
        source,
        "unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float rdz=d32(1.0f,dz);for(;q<n;q+=step)",
        "unsigned int nx=ni+2u*ng,ny=nj+2u*ng;for(;q<n;q+=step)",
        seam="pdef z candidate metric",
    )
    source = _replace_once(
        source,
        ",dw),rdz))),gz[",
        ",dw),rdsf[k]))),gz[",
        seam="pdef z candidate derivative",
    )
    source = _replace_once(
        source,
        "const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rrw,const float* dum,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dz){",
        "const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rrw,const float* dum,const float*rdsf,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt){",
        seam="pdef z correction signature",
    )
    source = _replace_once(
        source,
        "unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float rdt=d32(1.0f,dt),rdz=d32(1.0f,dz);for(;q<n;q+=step)",
        "unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float rdt=d32(1.0f,dt);for(;q<n;q+=step)",
        seam="pdef z correction metric",
    )
    source = _replace_once(
        source,
        ",dw),rdz));}",
        ",dw),rdsf[k]));}",
        seam="pdef z correction derivative",
    )
    if "float dz" in source or "rdz" in source:
        raise RuntimeError("uniform-z arithmetic survived stretched pdef patch")
    return source


GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE = _stretched_advection_source()
GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE = _stretched_rk_source()
GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256 = (
    "b7deede6ae6fc4e10ff07c3a547c8e1b31e4797959c13e4fe5129f0fc58a455c"
)
GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256 = (
    "c6e5f8759f0f4f2e8276e6962a5d75c3b7566226db260ffe4ceeee50f063033b"
)

# ---------------------------------------------------------------------------
# The flat stretched column -- terrain_flag=.false.
# ---------------------------------------------------------------------------
#
# The two texts above are CM1's terrain branch.  GM1's Cartesian stretch_z
# lane is a flat column, where CM1 r21.1 forms
#
# * adv.F:458-478  advx = -((d*rdx)*uh(i)), advy likewise, and the vertical
#   term -((d*rdz)*mh(1,1,k)) where the terrain branch (:535-552) has
#   -(d*rdsf(k)); rdsf is 1.0 on a flat column (param.F:8288-8289);
# * adv.F:646-656  the tendency scaled by rr0(1,1,k) alone;
# * pdef.F:37-47, 146-178 (x), 263-275, 381-411 (y), 477-486, 548-557 (z)
#   the candidate rho0(1,1,k)*s0 + dt*(adv + ((s*d)*rdz)*mh(1,1,k)) and the
#   correction ((dum+mass)-rho0(1,1,k)*s0)*rdt - ((s*d)*rdz)*mh(1,1,k), with
#   no gz or rgz;
# * solve2.F:493-520, 605-617  the mass flux and divergence the advection
#   image also carries (the Cartesian lane takes both from getdivx instead).
#
# mh is the (nk+2,) CM1-indexed column, rdz = 1.0/dz of the metric reference
# (param.F:6327), rho0/rf0/rr0 are read at CM1's (1,1,k) and smeps is
# cm1.F:311's 1.0e-30.  uh = vh = 1 on this lane's uniform horizontal grid.
GM1_FLAT_STRETCHED_SCALAR_ADVECTION_ABI = "gm1.cuda-flat-stretched-scalar-advection5-f32/v1"
GM1_FLAT_STRETCHED_SCALAR_PDEF_ABI = "gm1.cuda-flat-stretched-scalar-pdef-f32/v1"


def _between(source: str, start: str, stop: str) -> tuple[int, int]:
    begin = source.index(start)
    return begin, source.index(stop, begin)


def _flat_stretched_advection_source() -> str:
    source = _replace_once(
        GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE,
        f"// {GM1_CHILD_STRETCHED_SCALAR_ADVECTION_ABI}",
        f"// {GM1_FLAT_STRETCHED_SCALAR_ADVECTION_ABI}",
        seam="flat advection ABI",
    )
    # solve2.F:493-506 (and the k=1/nk+1 zeros of :507-513).
    source = _replace_once(
        source,
        "const float* u,const float* v,const float* rho0,const float* rgzu,const float* rgzv,float* rru,float* rrv,float* rrw,",
        "const float* u,const float* v,const float* rho0,float* rru,float* rrv,float* rrw,",
        seam="flat horizontal mass signature",
    )
    source = _replace_once(
        source,
        "float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j)*ni+i-1u],rho0[((unsigned long long)(k-1u)*nj+j)*ni+i]);rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(m32(0.5f,rh),m32(u[i3(k,j+ng,f+ng,nyu,nxu)],rgzu[(unsigned long long)j*(ni+1u)+f]));",
        "rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(rho0[(unsigned long long)(k-1u)*nj*ni],u[i3(k,j+ng,f+ng,nyu,nxu)]);",
        seam="flat rru",
    )
    source = _replace_once(
        source,
        "float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j-1u)*ni+i],rho0[((unsigned long long)(k-1u)*nj+j)*ni+i]);rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(m32(0.5f,rh),m32(v[i3(k,f+ng,i+ng,nyv,nxv)],rgzv[(unsigned long long)f*ni+i]));",
        "rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(rho0[(unsigned long long)(k-1u)*nj*ni],v[i3(k,f+ng,i+ng,nyv,nxv)]);",
        seam="flat rrv",
    )
    # solve2.F:514-518.
    begin, end = _between(
        source,
        'extern "C" __global__ void cm1_adv5_terrain_vertical_mass_flux_v1(',
        'extern "C" __global__ void cm1_adv5_terrain_divergence_v1(',
    )
    source = (
        source[:begin]
        + 'extern "C" __global__ void cm1_adv5_terrain_vertical_mass_flux_v1(\n'
        " const float* w,const float* rf0,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){\n"
        " unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,"
        "step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*nj*ni;"
        "unsigned int nxw=ni+2u*ng,nyw=nj+2u*ng;\n"
        " for(unsigned long long z=q;z<total;z+=step){unsigned int i=(unsigned int)(z%ni),"
        "j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj))+1u;"
        "if(k>=2u&&i>=1u&&i+1u<ni&&j>=1u&&j+1u<nj){unsigned int x=i+ng,y=j+ng;"
        "rrw[i3(k,y,x,nyw,nxw)]=m32(rf0[(unsigned long long)(k-1u)*nj*ni],w[i3(k,y,x,nyw,nxw)]);}}\n"
        "}\n\n"
        + source[end:]
    )
    # solve2.F:605-617.
    source = _replace_once(
        source,
        "const float* rru,const float* rrv,const float* rrw,const float*rdsf,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy){",
        "const float* rru,const float* rrv,const float* rrw,const float*mh,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float rdz){",
        seam="flat divergence signature",
    )
    source = _replace_once(
        source,
        "d=a32(d,m32(s32(rrw[i3(k+1u,y,x,nyw,nxw)],rrw[i3(k,y,x,nyw,nxw)]),rdsf[k]));divx[z]=(fabsf(d)<1.0e-20f)?0.0f:d;",
        "d=a32(d,m32(m32(s32(rrw[i3(k+1u,y,x,nyw,nxw)],rrw[i3(k,y,x,nyw,nxw)]),rdz),mh[k]));divx[z]=(fabsf(d)<1.0e-30f)?0.0f:d;",
        seam="flat divergence",
    )
    # adv.F:458-478, 555-572, 646-656.
    source = _replace_once(
        source,
        "const float* a,const float* fx,const float* fy,const float* fz,const float* divx,const float* rr0,const float* gz,const float*rdsf,float* tendency,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,const float* rru,const float* rrv,unsigned int obc){",
        "const float* a,const float* fx,const float* fy,const float* fz,const float* divx,const float* rr0,const float*mh,float* tendency,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float rdz,const float* rru,const float* rrv,unsigned int obc){",
        seam="flat tendency signature",
    )
    # adv.F:273-281's -(dumx(i+1)-dumx(i)) on the (inflow-edited) faces fw/fe/fs/fn.
    source = _replace_once(
        source,
        "float ax=m32(s32(fw,fe),rdx),ay=m32(s32(fs,fn),rdy),az;",
        "float ax=m32(-s32(fe,fw),rdx),ay=m32(-s32(fn,fs),rdy),az;",
        seam="flat tendency horizontal derivatives",
    )
    source = _replace_once(
        source,
        "if(k==1u)az=m32(-fz[i3(2u,y,x,nyw,nxw)],rdsf[k]);else if(k==nk)az=m32(fz[i3(nk,y,x,nyw,nxw)],rdsf[k]);"
        "else az=m32(s32(fz[i3(k,y,x,nyw,nxw)],fz[i3(k+1u,y,x,nyw,nxw)]),rdsf[k]);",
        "if(k==1u)az=m32(m32(-fz[i3(2u,y,x,nyw,nxw)],rdz),mh[k]);else if(k==nk)az=m32(m32(fz[i3(nk,y,x,nyw,nxw)],rdz),mh[k]);"
        "else az=m32(m32(-s32(fz[i3(k+1u,y,x,nyw,nxw)],fz[i3(k,y,x,nyw,nxw)]),rdz),mh[k]);",
        seam="flat tendency vertical derivatives",
    )
    source = _replace_once(
        source,
        "tendency[q]=m32(m32(total_adv,rr0[q]),gz[(unsigned long long)j*ni+i]);",
        "tendency[q]=m32(total_adv,rr0[(unsigned long long)(k-1u)*nj*ni]);",
        seam="flat tendency scale",
    )
    for token in ("rdsf", "rds[", "sigma", "ztop", "dzdx", "dzdy", "rgz", "gz[", "1.0e-20"):
        if token in source:
            raise RuntimeError(f"terrain-branch operand {token!r} survived the flat scalar advection text")
    return source


def _flat_stretched_rk_source() -> str:
    source = _replace_once(
        GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE,
        f"// {GM1_CHILD_STRETCHED_SCALAR_PDEF_ABI}",
        f"// {GM1_FLAT_STRETCHED_SCALAR_PDEF_ABI}",
        seam="flat pdef ABI",
    )
    # adv.F:458-478 and 555-572 on the pdef fluxes.
    source = _replace_once(
        source,
        "const float* fx,const float* fy,const float* fz,const float*rdsf,float* ax,float* ay,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,const float* s,const float* rru,const float* rrv,unsigned int obc){",
        "const float* fx,const float* fy,const float* fz,const float*mh,float* ax,float* ay,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float rdz,const float* s,const float* rru,const float* rrv,unsigned int obc){",
        seam="flat directional signature",
    )
    source = _replace_once(
        source,
        "float axv=m32(s32(fw,fe),rdx),ayv=m32(s32(fs,fn),rdy);",
        "float axv=m32(-s32(fe,fw),rdx),ayv=m32(-s32(fn,fs),rdy);",
        seam="flat directional horizontal derivatives",
    )
    source = _replace_once(
        source,
        "az[q]=(k==1u)?m32(-fz[i3(2u,y,x,nys,nxs)],rdsf[k]):((k==nk)?m32(fz[i3(nk,y,x,nys,nxs)],rdsf[k]):m32(s32(fz[i3(k,y,x,nys,nxs)],fz[i3(k+1u,y,x,nys,nxs)]),rdsf[k]));",
        "az[q]=(k==1u)?m32(m32(-fz[i3(2u,y,x,nys,nxs)],rdz),mh[k]):((k==nk)?m32(m32(fz[i3(nk,y,x,nys,nxs)],rdz),mh[k]):m32(m32(-s32(fz[i3(k+1u,y,x,nys,nxs)],fz[i3(k,y,x,nys,nxs)]),rdz),mh[k]));",
        seam="flat directional vertical derivatives",
    )
    # pdef.F:37-47 and 263-275: rho0(1,1,k), no gz.
    for axis, flux, d, spacing in (("x", "rru", "du", "rdx"), ("y", "rrv", "dv", "rdy")):
        source = _replace_once(
            source,
            f"cm1_rk_pdef_candidate_{axis}_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* {flux},",
            f"cm1_rk_pdef_candidate_{axis}_v1(const float* a,const float* s,const float* rho0,const float* {flux},",
            seam=f"flat {axis} candidate signature",
        )
        source = _replace_once(
            source,
            f"?r0a:a32(r0a,m32(m32(dt,a32(a{axis}[q],m32(m32(s[i3(k,y,x,ny,nx)],{d}),{spacing}))),gz[(unsigned long long)j*ni+i]));",
            f"?r0a:a32(r0a,m32(dt,a32(a{axis}[q],m32(m32(s[i3(k,y,x,ny,nx)],{d}),{spacing}))));",
            seam=f"flat {axis} candidate",
        )
        source = _replace_once(
            source,
            f"cm1_rk_pdef_correct_{axis}_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* {flux},",
            f"cm1_rk_pdef_correct_{axis}_v1(const float* a,const float* s,const float* rho0,const float* {flux},",
            seam=f"flat {axis} correction signature",
        )
    # pdef.F:37-47 and 263-275's rho0(1,1,k)*s0, which the inflow candidate
    # (pdef.F:49-100) keeps unchanged.
    source, count = re.subn(
        re.escape("float r0a=m32(rho0[q],a[q]);"),
        "float r0a=m32(rho0[(unsigned long long)k0*nj*ni],a[q]);",
        source,
    )
    if count != 2:
        raise RuntimeError(f"flat horizontal pdef candidate base transform found {count}, expected 2")
    source, count = re.subn(
        re.escape("float first=m32(m32(s32(a32(dum[q],mass),m32(rho0[q],a[q])),rdt),rgz[(unsigned long long)j*ni+i]);"),
        "float first=m32(s32(a32(dum[q],mass),m32(rho0[(unsigned long long)k0*nj*ni],a[q])),rdt);",
        source,
    )
    if count != 2:
        raise RuntimeError(f"flat horizontal pdef correction transform found {count}, expected 2")
    # pdef.F:477-486 and 548-557.
    source = _replace_once(
        source,
        "cm1_rk_pdef_candidate_z_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rrw,const float* az,const float*rdsf,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt){",
        "cm1_rk_pdef_candidate_z_v1(const float* a,const float* s,const float* rho0,const float* rrw,const float* az,const float*mh,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float rdz){",
        seam="flat z candidate signature",
    )
    source = _replace_once(
        source,
        "dum[q]=a32(m32(rho0[q],a[q]),m32(m32(dt,a32(az[q],m32(m32(s[i3(k,y,x,ny,nx)],dw),rdsf[k]))),gz[(unsigned long long)j*ni+i]));",
        "dum[q]=a32(m32(rho0[(unsigned long long)k0*nj*ni],a[q]),m32(dt,a32(az[q],m32(m32(m32(s[i3(k,y,x,ny,nx)],dw),rdz),mh[k]))));",
        seam="flat z candidate",
    )
    source = _replace_once(
        source,
        "cm1_rk_pdef_correct_z_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rrw,const float* dum,const float*rdsf,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt){",
        "cm1_rk_pdef_correct_z_v1(const float* a,const float* s,const float* rho0,const float* rrw,const float* dum,const float*mh,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float rdz){",
        seam="flat z correction signature",
    )
    source = _replace_once(
        source,
        "float first=m32(m32(s32(a32(current,mass),m32(rho0[q],a[q])),rdt),rgz[(unsigned long long)j*ni+i]);",
        "float first=m32(s32(a32(current,mass),m32(rho0[(unsigned long long)k0*nj*ni],a[q])),rdt);",
        seam="flat z correction first",
    )
    source = _replace_once(
        source,
        "az[q]=s32(first,m32(m32(s[i3(k,y,x,ny,nx)],dw),rdsf[k]));",
        "az[q]=s32(first,m32(m32(m32(s[i3(k,y,x,ny,nx)],dw),rdz),mh[k]));",
        seam="flat z correction",
    )
    # adv.F:646-656.
    source = _replace_once(
        source,
        "cm1_rk_pdef_total_v1(const float* s,const float* ax,const float* ay,const float* az,const float* divx,const float* rr0,const float* gz,float* adv,",
        "cm1_rk_pdef_total_v1(const float* s,const float* ax,const float* ay,const float* az,const float* divx,const float* rr0,float* adv,",
        seam="flat total signature",
    )
    source = _replace_once(
        source,
        "adv[q]=m32(m32(t,rr0[q]),gz[(unsigned long long)j*ni+i]);",
        "adv[q]=m32(t,rr0[(unsigned long long)(k-1u)*nj*ni]);",
        seam="flat total scale",
    )
    for token in ("rdsf", "rds[", "sigma", "ztop", "rgz", "gz["):
        if token in source:
            raise RuntimeError(f"terrain-branch operand {token!r} survived the flat pdef text")
    return source


GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE = _flat_stretched_advection_source()
GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE = _flat_stretched_rk_source()
GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE.encode()
).hexdigest()
GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE.encode()
).hexdigest()


def _compile(
    source: str,
    source_pin: str,
    kernel_names: tuple[str, ...],
    stem: str,
    result_type: type[CompiledAdvection5Kernels] | type[CompiledScalarMoistRKKernels],
) -> Any:
    if hashlib.sha256(source.encode()).hexdigest() != source_pin:
        raise RuntimeError(f"packaged {stem} CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _advection_arch(device.compute_capability)
    if arch != _advection_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(source, f"{stem}_ptx.cu", method="ptx").compile(
        ptx_options
    )
    cubin, _ = compiler._NVRTCProgram(
        source, f"{stem}_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", decoded) for name in kernel_names):
        raise ValueError(f"compiled PTX lacks {stem} entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError(f"compiled {stem} image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in kernel_names)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return result_type(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


@dataclass(frozen=True, slots=True)
class CompiledFlatStretchedAdvection5Kernels(CompiledAdvection5Kernels):
    """The flat stretched column's scalar advection image (mh and rdz bound)."""


@dataclass(frozen=True, slots=True)
class CompiledFlatStretchedScalarPdefKernels(CompiledScalarMoistRKKernels):
    """The flat stretched column's moisture pdef image (mh and rdz bound)."""


def compile_cuda_gm1_flat_stretched_scalar_advection() -> CompiledFlatStretchedAdvection5Kernels:
    return _compile(
        GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE,
        GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256,
        CUDA_ADVECTION5_KERNEL_NAMES,
        "gm1_flat_stretched_scalar_advection",
        CompiledFlatStretchedAdvection5Kernels,
    )


def compile_cuda_gm1_flat_stretched_scalar_pdef() -> CompiledFlatStretchedScalarPdefKernels:
    return _compile(
        GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE,
        GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE_SHA256,
        CUDA_SCALAR_MOIST_RK_KERNEL_NAMES,
        "gm1_flat_stretched_scalar_pdef",
        CompiledFlatStretchedScalarPdefKernels,
    )


def _flat_column_binding(cupy: Any, vertical: Any, *, nk: int) -> tuple[Any, float] | None:
    if not isinstance(vertical, Gm1ChildStretchedMetrics) or not vertical.flat_column:
        return None
    from cm1core.cuda_gm1_child_stretched_metrics import require_flat_column

    mh, _mf, rdz = require_flat_column(vertical, nk=nk, cupy=cupy)
    return mh, rdz


def compile_cuda_gm1_child_stretched_scalar_advection() -> CompiledAdvection5Kernels:
    return _compile(
        GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE,
        GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256,
        CUDA_ADVECTION5_KERNEL_NAMES,
        "gm1_child_stretched_scalar_advection",
        CompiledAdvection5Kernels,
    )


def compile_cuda_gm1_child_stretched_scalar_pdef() -> CompiledScalarMoistRKKernels:
    return _compile(
        GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE,
        GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256,
        CUDA_SCALAR_MOIST_RK_KERNEL_NAMES,
        "gm1_child_stretched_scalar_pdef",
        CompiledScalarMoistRKKernels,
    )


def _require_vertical_identity(
    vertical: Any, *, nk: int, operator: str, allow_compact_validation: bool
) -> None:
    """Refuse a coordinate that does not describe this column, before CUDA.

    The domain is the caller's.  What would be silently wrong is binding
    ``rdsf`` generated for another column: ``init_terrain.F:292-298`` builds it
    from ``sigmaf``, ``param.F:8193`` makes the model top ``sigmaf(nk+1)``, and
    ``param.F:8170`` divides every ``mh`` word by the reference ``dz``, so a
    mismatched coordinate divides every vertical flux difference by the depth
    of a layer this run does not have.

    ``allow_compact_validation`` admits a hand-built fixture whose identity
    names neither the sealed child nor a generated coordinate.  A forecast may
    not bind one: there is then nothing in the identity to check this column
    against, and a profile array of the right *shape* for another grid would
    be accepted silently.
    """

    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    identity = vertical.identity
    defect = stretched_vertical_identity_defect(
        identity,
        nk=nk,
        ztop_f32=vertical.ztop_f32,
        dz_reference_f32=vertical.dz_reference_f32,
    )
    if defect is not None:
        raise ValueError(f"{operator} vertical profiles refused: {defect}")
    if not allow_compact_validation and not (
        "grid_sha256" in identity or "vertical_coordinate" in identity
    ):
        raise ValueError(
            f"{operator} outside compact validation requires a coordinate that "
            "names itself: the identity carries neither the sealed child's "
            "grid_sha256 nor a generated vertical_coordinate, so nothing in it "
            "can be checked against this nk, top and reference dz, and rdsf "
            "profiles built for a different column would be bound silently"
        )


def _validate_vertical(
    cupy: Any,
    vertical: Gm1ChildStretchedMetrics,
    *,
    nk: int,
    allow_compact_validation: bool = False,
) -> Any:
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    rdsf = _require_cuda_f32(
        cupy, vertical.rdsf, "rdsf", (nk + 3,), writable=False
    )
    # param.F:7814-8139 generates a coordinate from the declared param6 words.
    # One that carries its own record is held to param.F:8193's own model top;
    # the sealed 50 m child keeps its three hashes.
    if generated_vertical_coordinate(vertical) is not None:
        require_generated_vertical(vertical, nk=nk, cupy=cupy)
    elif not allow_compact_validation and (
        vertical.identity.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256
        or vertical.identity.get("proof_sha256") != GM1_CHILD_STRETCHED_PROOF_SHA256
        or vertical.ztop_f32 != 15000.0
    ):
        raise ValueError("production stretched scalar requires the sealed child profiles")
    return rdsf


def launch_cuda_gm1_child_stretched_scalar_advection5(
    scalar: Any,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    tendency: Any,
    fluxes: Mapping[str, Any],
    vertical: Gm1ChildStretchedMetrics,
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    weps: float,
    wflag: bool = True,
    stream: Any | None = None,
    compiled: CompiledAdvection5Kernels | None = None,
    allow_compact_validation: bool = False,
    lateral_sides: tuple[int, int, int, int] | None = None,
) -> CudaScalarAdvection5Launch:
    """Enqueue scalar fluxes/tendency with exact ``rdsf(k)`` derivatives.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)`` of a whole-domain
    launch, which advects every cell with adv.F:214-340's open-side inflow
    edits; ``None`` is a tile-local launch whose outer three cells are left
    to the tile schedule (:func:`cm1core.cuda_advection5.scalar_open_boundary_code`).
    """

    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    ni, nj, nk = _validated_grid(ni, nj, nk)
    dx, dy = _validated_spacing("dx", dx), _validated_spacing("dy", dy)
    _require_vertical_identity(
        vertical,
        nk=nk,
        operator="stretched scalar advection",
        allow_compact_validation=allow_compact_validation,
    )
    if isinstance(weps, bool) or not isinstance(weps, (int, float)):
        raise TypeError("weps must be numeric")
    weps = float(weps)
    if not math.isfinite(weps) or weps <= 0.0:
        raise ValueError("weps must be finite and > 0")
    use_weno = scalar_advection_uses_weno(nrk, wflag=wflag)
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    scalar = _require_cuda_array(
        cupy, scalar, "scalar", (nk + 2, nj + 2 * ng, ni + 2 * ng), writable=False
    )
    rru = _require_cuda_array(
        cupy, arrays["rru"], "rru", (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng), writable=False
    )
    rrv = _require_cuda_array(
        cupy, arrays["rrv"], "rrv", (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng), writable=False
    )
    rrw = _require_cuda_array(
        cupy, arrays["rrw"], "rrw", (nk + 3, nj + 2 * ng, ni + 2 * ng), writable=False
    )
    fx = _require_cuda_array(cupy, fluxes["x"], "flux_x", rru.shape)
    fy = _require_cuda_array(cupy, fluxes["y"], "flux_y", rrv.shape)
    fz = _require_cuda_array(cupy, fluxes["z"], "flux_z", rrw.shape)
    divx = _require_cuda_array(cupy, divx, "divx", (nk, nj, ni), writable=False)
    tendency = _require_cuda_array(cupy, tendency, "tendency", (nk, nj, ni))
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    c1 = _require_cuda_array(cupy, metrics["c1"], "c1", shapes["c1"], writable=False)
    c2 = _require_cuda_array(cupy, metrics["c2"], "c2", shapes["c2"], writable=False)
    rr0 = _require_cuda_array(cupy, metrics["rr0"], "rr0", shapes["rr0"], writable=False)
    gz = _require_cuda_array(cupy, metrics["gz"], "gz", shapes["gz"], writable=False)
    rdsf = _validate_vertical(cupy, vertical, nk=nk)
    from cm1core.cuda_immutable_metrics import require_cuda_argument_aliases

    require_cuda_argument_aliases(
        {
            **{
                f"state.{index}": value
                for index, value in enumerate(
                    (scalar, rru, rrv, rrw, fx, fy, fz, divx, tendency)
                )
            },
            "metric.c1": c1, "metric.c2": c2, "metric.rr0": rr0, "metric.gz": gz,
            "vertical.rdsf": rdsf,
        },
        immutable_metrics=metrics,
        message="stretched scalar inputs, vertical profiles, scratch and unproved metrics must not alias",
    )
    from cm1core.cuda_immutable_metrics import require_cuda_argument_aliases

    require_cuda_argument_aliases(
        {
            "state.scalar": scalar, "state.rru": rru, "state.rrv": rrv,
            "state.rrw": rrw, "scratch.fx": fx, "scratch.fy": fy,
            "scratch.fz": fz, "state.divx": divx, "state.tendency": tendency,
            "metric.c1": c1, "metric.c2": c2, "metric.rr0": rr0, "metric.gz": gz,
            "profile.rdsf": rdsf,
        },
        immutable_metrics=metrics,
        message="stretched scalar inputs, profiles, scratch and unproved metrics must not alias",
    )
    flat_column = _flat_column_binding(cupy, vertical, nk=nk)
    flat = flat_column is not None
    if compiled is None:
        compiled = (
            compile_cuda_gm1_flat_stretched_scalar_advection()
            if flat
            else compile_cuda_gm1_child_stretched_scalar_advection()
        )
    if isinstance(compiled, CompiledFlatStretchedAdvection5Kernels) != flat:
        raise ValueError(
            "the scalar advection image and the vertical column disagree: a flat "
            "column (terrain_flag=.false.) runs adv.F:458-478,555-572,646-656, a "
            "terrain column adv.F:535-552,631-642"
        )
    if compiled.target_arch != vertical.target_arch:
        raise ValueError("stretched scalar and vertical CUDA images target different architectures")
    stream = stream or cupy.cuda.get_current_stream()
    counts = (nk * (nj + 1) * (ni + 1), (nk - 1) * nj * ni, nk * nj * ni)
    block = (CUDA_ADVECTION5_THREADS, 1, 1)
    grids = tuple(
        ((count + CUDA_ADVECTION5_THREADS - 1) // CUDA_ADVECTION5_THREADS, 1, 1)
        for count in counts
    )
    flag, u32 = cupy.uint32(1 if use_weno else 0), cupy.uint32
    obc = u32(scalar_open_boundary_code(lateral_sides))
    with stream:
        compiled.kernels[4](
            grids[0], block,
            (scalar, rru, rrv, fx, fy, u32(ni), u32(nj), u32(nk), u32(ng), flag, cupy.float64(weps)),
            stream=stream,
        )
        compiled.kernels[5](
            grids[1], block,
            (scalar, rrw, c1, c2, fz, u32(ni), u32(nj), u32(nk), u32(ng), flag, cupy.float64(weps)),
            stream=stream,
        )
        if flat:
            compiled.kernels[6](
                grids[2], block,
                (scalar, fx, fy, fz, divx, rr0, flat_column[0], tendency, u32(ni), u32(nj), u32(nk), u32(ng),
                 cupy.float32(dx), cupy.float32(dy), cupy.float32(flat_column[1]), rru, rrv, obc),
                stream=stream,
            )
        else:
            compiled.kernels[6](
                grids[2], block,
                (scalar, fx, fy, fz, divx, rr0, gz, rdsf, tendency, u32(ni), u32(nj), u32(nk), u32(ng), cupy.float32(dx), cupy.float32(dy), rru, rrv, obc),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": GM1_CHILD_STRETCHED_SCALAR_SCHEMA,
        "operation": "scalar-advection5",
        "abi": GM1_FLAT_STRETCHED_SCALAR_ADVECTION_ABI if flat else GM1_CHILD_STRETCHED_SCALAR_ADVECTION_ABI,
        "source_sha256": (
            GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256
            if flat
            else GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256
        ),
        "legacy_source_sha256": CUDA_ADVECTION5_SOURCE_SHA256,
        "authority": {
            "src/adv.F": CM1_ADV_F_SHA256,
            "src/adv_routines.F": CM1_ADV_ROUTINES_F_SHA256,
            "anchors": "adv.F:535-577,631-659; solve2.F:490-627",
        },
        "vertical_metrics": dict(vertical.identity),
        "selection": {"terrain": not flat, "stretch_z": 1, "nrk": nrk, "uses_weno": use_weno, "compact_validation": allow_compact_validation},
        "lateral_sides": None if lateral_sides is None else list(lateral_sides),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "target_arch": compiled.target_arch,
        "uniform_dz_argument_present": False,
        "ready_event_recorded": True,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return CudaScalarAdvection5Launch(event, compiled, receipt)


@dataclass(frozen=True, slots=True)
class CudaGm1ChildStretchedPdefLaunch:
    ready_event: Any
    compiled: CompiledScalarMoistRKKernels
    receipt: dict[str, Any]


def launch_cuda_gm1_child_stretched_pdef_moisture(
    moisture_a: Any,
    scalar_padded: Any,
    mass_flux: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    fluxes: Mapping[str, Any],
    advx: Any,
    advy: Any,
    advz: Any,
    candidate: Any,
    tendency: Any,
    vertical: Gm1ChildStretchedMetrics,
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    dx: float,
    dy: float,
    weps: float,
    species: int,
    numq: int = len(CM1_QMAG_F32),
    stream: Any | None = None,
    compiled: CompiledScalarMoistRKKernels | None = None,
    allow_compact_validation: bool = False,
    lateral_sides: tuple[int, int, int, int] | None = None,
) -> CudaGm1ChildStretchedPdefLaunch:
    """Run CM1's source-ordered final-stage pdef chain with ``rdsf(k)``.

    ``src/solve2.F:1435-1533`` runs this chain for every ``n=1,numq`` on the
    final RK stage, number concentrations included; the only per-row inputs
    are ``weps = qmag(n)*epsilon`` (``:1472``, the caller's) and
    ``pdefweno``, which is 0 for ``nqv`` (``:1498``) and 1 for every other
    row (``:1520``).  ``species`` is the zero-based row and ``numq`` the
    axis it indexes; a caller that does not declare its axis is held to the
    six-row mass table, so a wider row is refused rather than guessed at.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)`` of a whole-domain
    launch: adv.F:214-340's inflow edits, pdef.F:49-100's inflow candidate and
    pdef.F:125-166/337-376's corrections over every cell with the ``bcs(dum)``
    halo.  ``None`` keeps a tile-local launch's interior-only schedule.
    """

    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    ni, nj, nk = _validated_grid(ni, nj, nk)
    if isinstance(numq, bool) or not isinstance(numq, int) or numq < 1:
        raise ValueError("numq must be CM1's positive q3d scalar count")
    if isinstance(species, bool) or not isinstance(species, int) or not 0 <= species < numq:
        raise ValueError(
            f"species must index one of the numq={numq} CM1 q3d scalars "
            "(solve2.F:1435 n=1,numq)"
        )
    dx, dy = _validated_spacing("dx", dx), _validated_spacing("dy", dy)
    dttmp = _validated_spacing("dttmp", dttmp)
    _require_vertical_identity(
        vertical,
        nk=nk,
        operator="stretched moisture pdef",
        allow_compact_validation=allow_compact_validation,
    )
    if isinstance(weps, bool) or not isinstance(weps, (int, float)):
        raise TypeError("weps must be numeric")
    weps = float(weps)
    if not math.isfinite(weps) or weps <= 0.0:
        raise ValueError("weps must be finite and > 0")
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    physical = (nk, nj, ni)
    padded_shape = (nk + 2, nj + 2 * ng, ni + 2 * ng)
    moisture_a = _require_cuda_f32(cupy, moisture_a, "moisture_a", physical, writable=False)
    scalar_padded = _require_cuda_f32(cupy, scalar_padded, "scalar_padded", padded_shape, writable=False)
    rru = _require_cuda_f32(cupy, mass_flux["rru"], "rru", (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng), writable=False)
    rrv = _require_cuda_f32(cupy, mass_flux["rrv"], "rrv", (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng), writable=False)
    rrw = _require_cuda_f32(cupy, mass_flux["rrw"], "rrw", (nk + 3, nj + 2 * ng, ni + 2 * ng), writable=False)
    fx = _require_cuda_f32(cupy, fluxes["x"], "flux_x", rru.shape)
    fy = _require_cuda_f32(cupy, fluxes["y"], "flux_y", rrv.shape)
    fz = _require_cuda_f32(cupy, fluxes["z"], "flux_z", rrw.shape)
    divx = _require_cuda_f32(cupy, divx, "divx", physical, writable=False)
    advx = _require_cuda_f32(cupy, advx, "advx", physical)
    advy = _require_cuda_f32(cupy, advy, "advy", physical)
    advz = _require_cuda_f32(cupy, advz, "advz", physical)
    candidate = _require_cuda_f32(cupy, candidate, "candidate", physical)
    tendency = _require_cuda_f32(cupy, tendency, "tendency", physical)
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    metric_names = ("c1", "c2", "rho0_3d", "rr0", "gz", "rgz")
    metric = {
        name: _require_cuda_f32(cupy, metrics[name], name, shapes[name], writable=False)
        for name in metric_names
    }
    rdsf = _validate_vertical(cupy, vertical, nk=nk, allow_compact_validation=allow_compact_validation)
    from cm1core.cuda_immutable_metrics import require_cuda_argument_aliases

    require_cuda_argument_aliases(
        {
            "state.moisture_a": moisture_a, "state.scalar_padded": scalar_padded,
            "state.rru": rru, "state.rrv": rrv, "state.rrw": rrw,
            "scratch.fx": fx, "scratch.fy": fy, "scratch.fz": fz,
            "state.divx": divx, "scratch.advx": advx, "scratch.advy": advy,
            "scratch.advz": advz, "scratch.candidate": candidate,
            "state.tendency": tendency,
            **{f"metric.{name}": value for name, value in metric.items()},
            "profile.rdsf": rdsf,
        },
        immutable_metrics=metrics,
        message="stretched pdef inputs, profiles, scratch and unproved metrics must not alias",
    )
    flat_column = _flat_column_binding(cupy, vertical, nk=nk)
    flat = flat_column is not None
    if compiled is None:
        compiled = (
            compile_cuda_gm1_flat_stretched_scalar_pdef()
            if flat
            else compile_cuda_gm1_child_stretched_scalar_pdef()
        )
    if isinstance(compiled, CompiledFlatStretchedScalarPdefKernels) != flat:
        raise ValueError(
            "the pdef image and the vertical column disagree: a flat column "
            "(terrain_flag=.false.) runs pdef.F's rho0(1,1,k)/rdz*mh(1,1,k) "
            "statements, a terrain column its gz/rgz/rdsf(k) ones"
        )
    if compiled.target_arch != vertical.target_arch:
        raise ValueError("stretched pdef and vertical CUDA images target different architectures")
    stream = stream or cupy.cuda.get_current_stream()
    counts = (nk * (nj + 1) * (ni + 1), (nk - 1) * nj * ni, nk * nj * ni)
    block = (CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
    grids = tuple(
        ((count + CUDA_SCALAR_MOIST_RK_THREADS - 1) // CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
        for count in counts
    )
    u32, f32 = cupy.uint32, cupy.float32
    obc = u32(scalar_open_boundary_code(lateral_sides))
    with stream:
        compiled.kernels[4](
            grids[0], block,
            (scalar_padded, rru, rrv, fx, fy, u32(ni), u32(nj), u32(nk), u32(ng), cupy.float64(weps)),
            stream=stream,
        )
        compiled.kernels[5](
            grids[1], block,
            (scalar_padded, rrw, metric["c1"], metric["c2"], fz, u32(ni), u32(nj), u32(nk), u32(ng), u32(0 if species == 0 else 1), cupy.float64(weps)),
            stream=stream,
        )
        if flat:
            mh_column, rdz = flat_column
            compiled.kernels[6](
                grids[2], block,
                (fx, fy, fz, mh_column, advx, advy, advz, u32(ni), u32(nj), u32(nk), u32(ng),
                 f32(dx), f32(dy), f32(rdz), scalar_padded, rru, rrv, obc),
                stream=stream,
            )
            tail = (u32(ni), u32(nj), u32(nk), u32(ng), f32(dttmp))
            for candidate_index, correction_index, flux, spacing, direction in (
                (7, 8, rru, dx, advx),
                (9, 10, rrv, dy, advy),
            ):
                head = (moisture_a, scalar_padded, metric["rho0_3d"], flux)
                compiled.kernels[candidate_index](
                    grids[2], block, (*head, direction, candidate, *tail, f32(spacing), obc), stream=stream,
                )
                compiled.kernels[correction_index](
                    grids[2], block, (*head, candidate, direction, *tail, f32(spacing), obc), stream=stream,
                )
            head = (moisture_a, scalar_padded, metric["rho0_3d"], rrw)
            compiled.kernels[11](
                grids[2], block, (*head, advz, mh_column, candidate, *tail, f32(rdz)), stream=stream,
            )
            compiled.kernels[12](
                grids[2], block, (*head, candidate, mh_column, advz, *tail, f32(rdz)), stream=stream,
            )
            compiled.kernels[13](
                grids[2], block,
                (scalar_padded, advx, advy, advz, divx, metric["rr0"], tendency, u32(ni), u32(nj), u32(nk), u32(ng), obc),
                stream=stream,
            )
        else:
            compiled.kernels[6](
                grids[2], block,
                (fx, fy, fz, rdsf, advx, advy, advz, u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), scalar_padded, rru, rrv, obc),
                stream=stream,
            )
            for candidate_index, correction_index, flux, spacing, direction, is_vertical in (
                (7, 8, rru, dx, advx, False),
                (9, 10, rrv, dy, advy, False),
                (11, 12, rrw, 0.0, advz, True),
            ):
                common = (
                    moisture_a, scalar_padded, metric["rho0_3d"], metric["gz"], flux,
                    direction,
                )
                tail = (u32(ni), u32(nj), u32(nk), u32(ng), f32(dttmp))
                candidate_args = (
                    (*common, rdsf, candidate, *tail)
                    if is_vertical
                    else (*common, candidate, *tail, f32(spacing), obc)
                )
                compiled.kernels[candidate_index](grids[2], block, candidate_args, stream=stream)
                correction_common = (
                    moisture_a, scalar_padded, metric["rho0_3d"], metric["gz"],
                    metric["rgz"], flux, candidate,
                )
                correction_args = (
                    (*correction_common, rdsf, direction, *tail)
                    if is_vertical
                    else (*correction_common, direction, *tail, f32(spacing), obc)
                )
                compiled.kernels[correction_index](grids[2], block, correction_args, stream=stream)
            compiled.kernels[13](
                grids[2], block,
                (scalar_padded, advx, advy, advz, divx, metric["rr0"], metric["gz"], tendency, u32(ni), u32(nj), u32(nk), u32(ng), obc),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": GM1_CHILD_STRETCHED_SCALAR_PDEF_SCHEMA,
        "operation": "final-rk-moisture-pdef",
        "abi": GM1_FLAT_STRETCHED_SCALAR_PDEF_ABI if flat else GM1_CHILD_STRETCHED_SCALAR_PDEF_ABI,
        "source_sha256": (
            GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE_SHA256
            if flat
            else GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256
        ),
        "legacy_source_sha256": CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
        "authority": {
            "src/adv.F": CM1_ADV_F_SHA256,
            "src/pdef.F": CM1_PDEF_F_SHA256,
            "src/solve2.F": CM1_SOLVE2_F_SHA256,
            "anchors": "adv.F:421-445,535-577,616-627; pdef.F:459-581; solve2.F:1430-1533",
        },
        "vertical_metrics": dict(vertical.identity),
        "selection": {"terrain": not flat, "stretch_z": 1, "pdscheme": 1, "pdef": 1, "nrk": CM1_NRKMAX, "species": species, "numq": numq, "pdefweno": 0 if species == 0 else 1, "compact_validation": allow_compact_validation},
        "lateral_sides": None if lateral_sides is None else list(lateral_sides),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "target_arch": compiled.target_arch,
        "uniform_dz_argument_present": False,
        "ready_event_recorded": True,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return CudaGm1ChildStretchedPdefLaunch(event, compiled, receipt)


__all__ = (
    "CompiledFlatStretchedAdvection5Kernels",
    "CompiledFlatStretchedScalarPdefKernels",
    "GM1_FLAT_STRETCHED_SCALAR_ADVECTION_ABI",
    "GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE",
    "GM1_FLAT_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256",
    "GM1_FLAT_STRETCHED_SCALAR_PDEF_ABI",
    "GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE",
    "GM1_FLAT_STRETCHED_SCALAR_PDEF_SOURCE_SHA256",
    "compile_cuda_gm1_flat_stretched_scalar_advection",
    "compile_cuda_gm1_flat_stretched_scalar_pdef",
    "CudaGm1ChildStretchedPdefLaunch",
    "GM1_CHILD_STRETCHED_SCALAR_ADVECTION_ABI",
    "GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE",
    "GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256",
    "GM1_CHILD_STRETCHED_SCALAR_PDEF_ABI",
    "GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE",
    "GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256",
    "GM1_CHILD_STRETCHED_SCALAR_SCHEMA",
    "GM1_CHILD_STRETCHED_SCALAR_PDEF_SCHEMA",
    "compile_cuda_gm1_child_stretched_scalar_advection",
    "compile_cuda_gm1_child_stretched_scalar_pdef",
    "launch_cuda_gm1_child_stretched_pdef_moisture",
    "launch_cuda_gm1_child_stretched_scalar_advection5",
)
