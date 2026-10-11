"""Fifth-order momentum advection on the sealed stretched GM1 child grid."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_PROOF_SHA256,
    Gm1ChildStretchedMetrics,
    generated_vertical_coordinate,
    require_generated_vertical,
    stretched_vertical_identity_defect,
)
from cm1core.cuda_momentum_advection5 import (
    CM1_ADV_F_SHA256,
    CM1_ADV_ROUTINES_F_SHA256,
    CM1_SMEPS_F32,
    CM1_SOLVE2_F_SHA256,
    CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES,
    CUDA_MOMENTUM_ADVECTION5_SOURCE,
    CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256,
    CUDA_MOMENTUM_ADVECTION5_THREADS,
    MOMENTUM_ALL_OPEN_SIDES,
    CompiledMomentumAdvection5Kernels,
    CudaMomentumAdvection5Launch,
    _canonical_arch,
    momentum_advection5_counts,
    momentum_boundary_faces,
    momentum_lateral_sides,
    momentum_open_boundary_code,
    _positive_int,
    _require_cuda,
    _spacing,
)
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA = "gm1.cuda-child-stretched-momentum5/v1"
GM1_CHILD_STRETCHED_MOMENTUM_ABI = "gm1.cuda-child-stretched-momentum5-f32/v1"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("legacy momentum source no longer matches the sealed stretch patch")
    return source.replace(old, new, 1)


def _stretched_source() -> str:
    source = _replace_once(
        CUDA_MOMENTUM_ADVECTION5_SOURCE,
        "// gpuwm-cm1.cuda-momentum-advection5-kernels/v1",
        "// gm1.cuda-child-stretched-momentum5-f32/v1",
    )
    source = _replace_once(
        source,
        "unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;for(;q<nu+nv;q+=s)",
        "unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxr=ni+2u*ng,nyr=nj+2u*ng;for(;q<nu+nv;q+=s)",
    )
    source = _replace_once(
        source,
        "k=(unsigned int)(z/nj)+1u,l=(f==0u)?0u:f-1u,r=(f==ni)?ni-1u:f;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+j)*ni+l],rho0[((unsigned long long)(k-1u)*nj+j)*ni+r]);",
        "k=(unsigned int)(z/nj)+1u;float rh=a32(rho0[i3(k,j+ng,f+ng-1u,nyr,nxr)],rho0[i3(k,j+ng,f+ng,nyr,nxr)]);",
    )
    source = _replace_once(
        source,
        "k=(unsigned int)(z/(nj+1u))+1u,b=(f==0u)?0u:f-1u,t=(f==nj)?nj-1u:f;float rh=a32(rho0[((unsigned long long)(k-1u)*nj+b)*ni+i],rho0[((unsigned long long)(k-1u)*nj+t)*ni+i]);",
        "k=(unsigned int)(z/(nj+1u))+1u;float rh=a32(rho0[i3(k,f+ng-1u,i+ng,nyr,nxr)],rho0[i3(k,f+ng,i+ng,nyr,nxr)]);",
    )
    source = _replace_once(
        source,
        "extern \"C\" __global__ void cm1_mom5_vertical_mass_v1(const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){",
        "extern \"C\" __global__ void cm1_mom5_vertical_mass_v1(const float* w,const float* rf0,const float* dzdx,const float* dzdy,const float* gz,const float* rru,const float* rrv,float* rrw,const float*sigma,const float*sigmaf,const float*rds,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float ztop){",
    )
    source = _replace_once(
        source,
        "float zt=m32((float)nk,dz),rzt=d32(1.0f,zt),rdz=d32(1.0f,dz);",
        "float zt=ztop,rzt=d32(1.0f,zt);",
    )
    source = _replace_once(
        source,
        "float sf=m32((float)(kf-1u),dz),sm=m32((float)kf-1.5f,dz),r2=m32(s32(sf,sm),rdz),",
        "float sf=sigmaf[kf],sm=sigma[kf-1u],r2=m32(s32(sf,sm),rds[kf]),",
    )
    source = _replace_once(
        source,
        "extern \"C\" __global__ void cm1_mom5_divergence_v1(const float* rru,const float* rrv,const float* rrw,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,float smeps){",
        "extern \"C\" __global__ void cm1_mom5_divergence_v1(const float* rru,const float* rrv,const float* rrw,const float*rdsf,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float smeps){",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;",
    )
    source = _replace_once(
        source,
        "),rdz));divx[q]",
        "),rdsf[k]));divx[q]",
    )
    source = _replace_once(
        source,
        "extern \"C\" __global__ void cm1_mom5_advu_v1(const float* u,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzu,const float* vh,float* uten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){",
        "extern \"C\" __global__ void cm1_mom5_advu_v1(const float* u,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzu,const float* vh,const float*rdsf,float* uten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,unsigned int obc){",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int f=",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<n;q+=s){unsigned int f=",
    )
    source = _replace_once(
        source,
        "uvz(u,rrw,c2a,2u,k,(int)j,(int)f,ni,nj,nk,ng,obc)),rdz));",
        "uvz(u,rrw,c2a,2u,k,(int)j,(int)f,ni,nj,nk,ng,obc)),rdsf[k]));",
    )
    source = source.replace(
        "uvz(u,rrw,c2a,2u,2u,(int)j,(int)f,ni,nj,nk,ng,obc),rdz)",
        "uvz(u,rrw,c2a,2u,2u,(int)j,(int)f,ni,nj,nk,ng,obc),rdsf[k])",
        1,
    ).replace(
        "uvz(u,rrw,c2a,2u,nk,(int)j,(int)f,ni,nj,nk,ng,obc),rdz)",
        "uvz(u,rrw,c2a,2u,nk,(int)j,(int)f,ni,nj,nk,ng,obc),rdsf[k])",
        1,
    )
    source = _replace_once(
        source,
        "extern \"C\" __global__ void cm1_mom5_advv_v1(const float* v,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzv,const float* uh,float* vten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){",
        "extern \"C\" __global__ void cm1_mom5_advv_v1(const float* v,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* rho0,const float* c2a,const float* gzv,const float* uh,const float*rdsf,float* vten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,unsigned int obc){",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni);unsigned long long z=q/ni;",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni);unsigned long long z=q/ni;",
    )
    source = _replace_once(
        source,
        "uvz(v,rrw,c2a,3u,k,(int)g,(int)i,ni,nj,nk,ng,obc)),rdz));",
        "uvz(v,rrw,c2a,3u,k,(int)g,(int)i,ni,nj,nk,ng,obc)),rdsf[k]));",
    )
    source = source.replace(
        "uvz(v,rrw,c2a,3u,2u,(int)g,(int)i,ni,nj,nk,ng,obc),rdz)",
        "uvz(v,rrw,c2a,3u,2u,(int)g,(int)i,ni,nj,nk,ng,obc),rdsf[k])",
        1,
    ).replace(
        "uvz(v,rrw,c2a,3u,nk,(int)g,(int)i,ni,nj,nk,ng,obc),rdz)",
        "uvz(v,rrw,c2a,3u,nk,(int)g,(int)i,ni,nj,nk,ng,obc),rdsf[k])",
        1,
    )
    source = _replace_once(
        source,
        "extern \"C\" __global__ void cm1_mom5_advw_v1(const float* w,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* c2a,const float* rrf0,const float* gz,const float* uh,const float* vh,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,unsigned int obc){",
        "extern \"C\" __global__ void cm1_mom5_advw_v1(const float* w,const float* rru,const float* rrv,const float* rrw,const float* divx,const float* c2a,const float* rrf0,const float* gz,const float* uh,const float* vh,const float*rds,float* wten,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,unsigned int obc){",
    )
    source = _replace_once(
        source,
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),kf=",
        "float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy);for(;q<n;q+=s){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),kf=",
    )
    source = _replace_once(
        source,
        "wz(w,rrw,kf-1u,(int)j,(int)i,ni,nj,nk,ng)),rdz),t=",
        "wz(w,rrw,kf-1u,(int)j,(int)i,ni,nj,nk,ng)),rds[kf]),t=",
    )
    if "float dz" in source or "rdz" in source:
        raise RuntimeError("uniform-z arithmetic survived the stretched momentum patch")
    return source


GM1_CHILD_STRETCHED_MOMENTUM_SOURCE = _stretched_source()
GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256 = (
    "fd9e490c5e75cc6005d252553d41c236adf71101d83719d4d26a7ec6dc1a0658"
)

# ---------------------------------------------------------------------------
# The flat stretched column -- terrain_flag=.false.
# ---------------------------------------------------------------------------
#
# The text above is CM1's terrain branch.  GM1's Cartesian stretch_z lane is a
# flat column (config.py admits grid.terrain only on the real-product lane),
# and there CM1 r21.1 runs different statements:
#
# * solve2.F:493-520  rru = rho0(1,1,k)*u3d, rrv = rho0(1,1,k)*v3d and
#   rrw = rf0(1,1,k)*w3d, with no terrain-following term added;
# * solve2.F:609-611  divx's vertical term is ((d*rdz)*mh(1,1,k)), where the
#   terrain branch (:621-623) forms d*rdsf(k) -- and rdsf stays 1.0 on a flat
#   column (param.F:8288-8289);
# * adv.F:1380-1391, 2321-2332  advz = -((d*rdz)*mh(1,1,k)) for u and v, and
#   the tendency is scaled by rr0(1,1,k) (:1484-1485, :2415-2416), not by
#   gzu/(0.5*(rho0(i-1)+rho0(i)));
# * adv.F:3299, 3354-3356  advz = -((d*rdz)*mf(1,1,k)) for w, scaled by
#   rrf0(1,1,k), not rrf0(i,j,k)*gz(i,j).
#
# mh and mf are the column's own param.F:8467,8476 words, bound as the
# (nk+2,) and (nk+3,) CM1-indexed profiles; rdz is param.F:6327's 1.0/dz of
# the metric reference dz; rho0/rf0/rr0/rrf0 are read at the first physical
# column, CM1's (1,1,k).  cm1.F:311 makes smeps 1.0e-30.
GM1_FLAT_STRETCHED_MOMENTUM_ABI = "gm1.cuda-flat-stretched-momentum5-f32/v1"
CM1_R21_SMEPS_F32 = 1.0e-30


def _flat_stretched_source() -> str:
    source = _replace_once(
        GM1_CHILD_STRETCHED_MOMENTUM_SOURCE,
        "// gm1.cuda-child-stretched-momentum5-f32/v1",
        f"// {GM1_FLAT_STRETCHED_MOMENTUM_ABI}",
    )
    # solve2.F:493-506.
    source = _replace_once(
        source,
        "cm1_mom5_horizontal_mass_v1(const float* u,const float* v,const float* rho0,const float* rgzu,const float* rgzv,float* rru,float* rrv,",
        "cm1_mom5_horizontal_mass_v1(const float* u,const float* v,const float* rho0,float* rru,float* rrv,",
    )
    source = _replace_once(
        source,
        "k=(unsigned int)(z/nj)+1u;float rh=a32(rho0[i3(k,j+ng,f+ng-1u,nyr,nxr)],rho0[i3(k,j+ng,f+ng,nyr,nxr)]);rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(m32(0.5f,rh),m32(u[i3(k,j+ng,f+ng,nyu,nxu)],rgzu[(unsigned long long)j*(ni+1u)+f]));",
        "k=(unsigned int)(z/nj)+1u;rru[i3(k,j+ng,f+ng,nyu,nxu)]=m32(rho0[i3(k,ng,ng,nyr,nxr)],u[i3(k,j+ng,f+ng,nyu,nxu)]);",
    )
    source = _replace_once(
        source,
        "k=(unsigned int)(z/(nj+1u))+1u;float rh=a32(rho0[i3(k,f+ng-1u,i+ng,nyr,nxr)],rho0[i3(k,f+ng,i+ng,nyr,nxr)]);rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(m32(0.5f,rh),m32(v[i3(k,f+ng,i+ng,nyv,nxv)],rgzv[(unsigned long long)f*ni+i]));",
        "k=(unsigned int)(z/(nj+1u))+1u;rrv[i3(k,f+ng,i+ng,nyv,nxv)]=m32(rho0[i3(k,ng,ng,nyr,nxr)],v[i3(k,f+ng,i+ng,nyv,nxv)]);",
    )
    # solve2.F:507-520.
    vertical_start = source.index('extern "C" __global__ void cm1_mom5_vertical_mass_v1(')
    vertical_end = source.index('extern "C" __global__ void cm1_mom5_divergence_v1(')
    source = (
        source[:vertical_start]
        + 'extern "C" __global__ void cm1_mom5_vertical_mass_v1(const float* w,const float* rf0,float* rrw,'
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){"
        "unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,"
        "s=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;"
        "unsigned int nx=ni+2u*ng,ny=nj+2u*ng;for(;q<n;q+=s){"
        "unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),"
        "kf=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;"
        "if(kf==1u||kf==nk+1u){rrw[i3(kf,y,x,ny,nx)]=0.0f;}"
        "else{rrw[i3(kf,y,x,ny,nx)]=m32(rf0[(unsigned long long)(kf-1u)*nj*ni],w[i3(kf,y,x,ny,nx)]);}}}\n"
        + source[vertical_end:]
    )
    # solve2.F:605-617.
    source = _replace_once(
        source,
        "cm1_mom5_divergence_v1(const float* rru,const float* rrv,const float* rrw,const float*rdsf,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float smeps){",
        "cm1_mom5_divergence_v1(const float* rru,const float* rrv,const float* rrw,const float*mh,float* divx,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float rdz,float smeps){",
    )
    source = _replace_once(
        source,
        "),rdsf[k]));divx[q]",
        "),rdz),mh[k]));divx[q]",
    )
    source = _replace_once(
        source,
        "d=a32(d,m32(s32(rrw[i3(k+1u,y,x,ny,nx)]",
        "d=a32(d,m32(m32(s32(rrw[i3(k+1u,y,x,ny,nx)]",
    )
    # adv.F:1380-1391, 1484-1485 (u) and 2321-2332, 2415-2416 (v).
    for stag, var, face, metric in (("2u", "u", "(int)j,(int)f", "gzu"), ("3u", "v", "(int)g,(int)i", "gzv")):
        source = _replace_once(
            source,
            f"const float* rho0,const float* c2a,const float* {metric},const float* {'vh' if var == 'u' else 'uh'},const float*rdsf,",
            f"const float* rr0,const float* c2a,const float* {'vh' if var == 'u' else 'uh'},const float*mh,",
        )
        source = _replace_once(
            source,
            f"float az=(k==1u)?m32(-uvz({var},rrw,c2a,{stag},2u,{face},ni,nj,nk,ng,obc),rdsf[k])"
            f":((k==nk)?m32(uvz({var},rrw,c2a,{stag},nk,{face},ni,nj,nk,ng,obc),rdsf[k])"
            f":m32(-s32(uvz({var},rrw,c2a,{stag},k+1u,{face},ni,nj,nk,ng,obc),uvz({var},rrw,c2a,{stag},k,{face},ni,nj,nk,ng,obc)),rdsf[k]));",
            f"float az=(k==1u)?m32(m32(-uvz({var},rrw,c2a,{stag},2u,{face},ni,nj,nk,ng,obc),rdz),mh[k])"
            f":((k==nk)?m32(m32(uvz({var},rrw,c2a,{stag},nk,{face},ni,nj,nk,ng,obc),rdz),mh[k])"
            f":m32(m32(-s32(uvz({var},rrw,c2a,{stag},k+1u,{face},ni,nj,nk,ng,obc),uvz({var},rrw,c2a,{stag},k,{face},ni,nj,nk,ng,obc)),rdz),mh[k]));",
        )
    source = _replace_once(
        source,
        "float sc=d32(gzu[(unsigned long long)j*(ni+1u)+f],m32(0.5f,a32(rho0[((unsigned long long)k0*nj+j)*ni+(unsigned int)fw],rho0[((unsigned long long)k0*nj+j)*ni+(unsigned int)fe])));",
        "float sc=rr0[(unsigned long long)k0*nj*ni];",
    )
    source = _replace_once(
        source,
        "float sc=d32(gzv[(unsigned long long)g*ni+i],m32(0.5f,a32(rho0[((unsigned long long)k0*nj+(unsigned int)gs)*ni+i],rho0[((unsigned long long)k0*nj+(unsigned int)gn)*ni+i])));",
        "float sc=rr0[(unsigned long long)k0*nj*ni];",
    )
    # adv.F:3299, 3354-3356 (w).
    source = _replace_once(
        source,
        "const float* c2a,const float* rrf0,const float* gz,const float* uh,const float* vh,const float*rds,float* wten,",
        "const float* c2a,const float* rrf0,const float* uh,const float* vh,const float*mf,float* wten,",
    )
    source = _replace_once(
        source,
        "wz(w,rrw,kf-1u,(int)j,(int)i,ni,nj,nk,ng)),rds[kf]),t=",
        "wz(w,rrw,kf-1u,(int)j,(int)i,ni,nj,nk,ng)),rdz),mf[kf]),t=",
    )
    source = _replace_once(
        source,
        "float az=m32(-s32(wz(w,rrw,kf,(int)j,(int)i,ni,nj,nk,ng)",
        "float az=m32(m32(-s32(wz(w,rrw,kf,(int)j,(int)i,ni,nj,nk,ng)",
    )
    source = _replace_once(
        source,
        "float sc=m32(rrf0[((unsigned long long)(kf-1u)*nj+j)*ni+i],gz[(unsigned long long)j*ni+i]);",
        "float sc=rrf0[(unsigned long long)(kf-1u)*nj*ni];",
    )
    # adv.F:1161-1185, 2093-2117, 3031-3099: the open-boundary corrections
    # are ((((q*0.5)*c)*rdy)*vh(j)) for u and v and (((w*c)*rdx)*uh(i)) for
    # w, and adv.F:3021-3022's w flux differences -((d*rdx)*uh(i)); the text
    # above grouped all three right to left.
    source, count = re.subn(
        r"m32\((m32\(ma\((?:[^()]|\(int\))*\),0\.5f\)|ma\((?:[^()]|\(int\))*\)),m32\(c,m32\((rd[xy]),([uv]h\[[a-z]+\])\)\)\)",
        r"m32(m32(m32(\1,c),\2),\3)",
        source,
    )
    if count != 8:
        raise RuntimeError(f"flat stretched momentum boundary transform found {count}, expected 8")
    source = _replace_once(source, "float ax=m32(-s32(wfx(", "float ax=m32(m32(-s32(wfx(")
    source = _replace_once(source, "),m32(rdx,uh[x])),ay=m32(-s32(wfy(", "),rdx),uh[x]),ay=m32(m32(-s32(wfy(")
    source = _replace_once(source, "),m32(rdy,vh[y]));float c2=c2a[", "),rdy),vh[y]);float c2=c2a[")
    # Every advection kernel takes rdz after its horizontal reciprocals.
    source, count = re.subn(
        r"(void cm1_mom5_adv[uvw]_v1\([^)]*unsigned int ng,float dx,float dy)(,unsigned int obc)\)\{",
        r"\1,float rdz\2){",
        source,
    )
    if count != 3:
        raise RuntimeError("flat stretched momentum rdz signature transform is incomplete")
    for token in ("rdsf", "rds[", "sigma", "ztop", "dzdx", "dzdy", "rgzu", "rgzv", "gz["):
        if token in source:
            raise RuntimeError(f"terrain-branch operand {token!r} survived the flat momentum text")
    return source


GM1_FLAT_STRETCHED_MOMENTUM_SOURCE = _flat_stretched_source()
GM1_FLAT_STRETCHED_MOMENTUM_SOURCE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_MOMENTUM_SOURCE.encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledFlatStretchedMomentumKernels(CompiledMomentumAdvection5Kernels):
    """The flat stretched column's image; its kernels take mh/mf and rdz."""


def compile_cuda_gm1_child_stretched_momentum() -> CompiledMomentumAdvection5Kernels:
    actual = hashlib.sha256(GM1_CHILD_STRETCHED_MOMENTUM_SOURCE.encode()).hexdigest()
    if actual != GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256:
        raise RuntimeError("packaged stretched-momentum CUDA source differs from its pin")
    return _compile_momentum_image(
        GM1_CHILD_STRETCHED_MOMENTUM_SOURCE, "gm1_child_stretched_momentum",
        CompiledMomentumAdvection5Kernels,
    )


def compile_cuda_gm1_flat_stretched_momentum() -> CompiledFlatStretchedMomentumKernels:
    """solve2.F:493-520,605-617 and adv.F's flat advu/advv/advw, compiled."""

    actual = hashlib.sha256(GM1_FLAT_STRETCHED_MOMENTUM_SOURCE.encode()).hexdigest()
    if actual != GM1_FLAT_STRETCHED_MOMENTUM_SOURCE_SHA256:
        raise RuntimeError("packaged flat stretched-momentum CUDA source differs from its pin")
    return _compile_momentum_image(
        GM1_FLAT_STRETCHED_MOMENTUM_SOURCE, "gm1_flat_stretched_momentum",
        CompiledFlatStretchedMomentumKernels,
    )


def _compile_momentum_image(source: str, stem: str, result_type: type) -> Any:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
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
    ptx, _ = compiler._NVRTCProgram(
        source,
        f"{stem}_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        source,
        f"{stem}_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks stretched-momentum entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled stretched-momentum image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_MOMENTUM_ADVECTION5_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
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


def launch_cuda_gm1_child_stretched_momentum5(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    vertical: Gm1ChildStretchedMetrics,
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    divx: Any | None = None,
    stream: Any | None = None,
    compiled: CompiledMomentumAdvection5Kernels | None = None,
    rrw_boundary_hook: Callable[..., Mapping[str, object]] | None = None,
    post_divergence_hook: Callable[..., Mapping[str, object]] | None = None,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    lateral_sides: tuple[int, int, int, int] = MOMENTUM_ALL_OPEN_SIDES,
) -> CudaMomentumAdvection5Launch:
    """Enqueue the selected terrain momentum family with exact stretch profiles.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)``: advu advances faces
    ``i1..i2`` and advv rows ``j1..j2`` of that family and the inflow edits run
    on its open sides only (:func:`~cm1core.cuda_momentum_advection5.momentum_open_boundary_code`).
    """

    ni, nj, nk = (
        _positive_int("ni", ni, 7),
        _positive_int("nj", nj, 7),
        _positive_int("nk", nk, 8),
    )
    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be boolean")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if allow_uniform_idealized_validation and not allow_compact_validation:
        raise ValueError(
            "the uniform idealized momentum fixture is a validation lane, not a "
            "forecast: it binds a coordinate whose layers are all one depth, so "
            "it must be declared with allow_compact_validation"
        )
    if rrw_boundary_hook is not None and not callable(rrw_boundary_hook):
        raise TypeError("rrw_boundary_hook must be callable or None")
    if post_divergence_hook is not None and not callable(post_divergence_hook):
        raise TypeError("post_divergence_hook must be callable or None")
    lateral_sides = momentum_lateral_sides(lateral_sides)
    obc = momentum_open_boundary_code(lateral_sides)
    # param.F:7814-8139 generates a level set from whatever param6 words a run
    # declares; the sealed 50 m child is one member of that family, not the
    # family.  A binding that carries its own generated coordinate record is
    # held to that coordinate (param.F:8193) and to no fixture's extent.
    generated = (
        generated_vertical_coordinate(vertical)
        if isinstance(vertical, Gm1ChildStretchedMetrics)
        else None
    )
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    dx, dy = _spacing("dx", dx), _spacing("dy", dy)
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    if allow_uniform_idealized_validation:
        if (
            vertical.identity.get("fixture_class")
            != "source-audited-flat-idealized-uniform-200m"
            or vertical.ztop_f32 != 15000.0
            or vertical.dz_reference_f32 != 200.0
        ):
            raise ValueError(
                "uniform idealized momentum requires the source-audited flat 200 m profiles"
            )
    else:
        # The domain is the caller's; what is refused is a coordinate that does
        # not describe this column.  param.F:8193 makes the model top
        # sigmaf(nk+1) and param.F:8170 divides every mh word by the reference
        # dz, so profiles generated for another nk, top or reference would put
        # every vertical flux difference and every terrain-height factor on a
        # column this run does not have.
        defect = stretched_vertical_identity_defect(
            vertical.identity,
            nk=nk,
            ztop_f32=vertical.ztop_f32,
            dz_reference_f32=vertical.dz_reference_f32,
        )
        if defect is not None:
            raise ValueError(f"stretched momentum vertical profiles refused: {defect}")
        if not allow_compact_validation and not (
            "grid_sha256" in vertical.identity
            or "vertical_coordinate" in vertical.identity
        ):
            raise ValueError(
                "stretched momentum outside compact validation requires a "
                "coordinate that names itself: the identity carries neither the "
                "sealed child's grid_sha256 nor a generated vertical_coordinate, "
                "so nothing in it can be checked against this nk, top and "
                "reference dz, and rds/rdsf profiles built for a different "
                "column would be bound silently"
            )
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    shapes = {
        "u3d": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "v3d": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "w3d": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "wten": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "uh": (ni + 2 * ng,),
        "vh": (nj + 2 * ng,),
    }
    writable = {"uten", "vten", "wten", "rru", "rrv", "rrw"}
    a = {
        name: _require_cuda(cupy, arrays[name], name, shape, writable=name in writable)
        for name, shape in shapes.items()
    }
    metric_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    metric_names = (
        "rho0_3d", "rf0", "rr0", "rrf0", "c1", "c2", "gz", "gzu",
        "gzv", "rgzu", "rgzv", "dzdx", "dzdy",
    )
    m = {
        name: _require_cuda(cupy, metrics[name], name, metric_shapes[name], writable=False)
        for name in metric_names
    }
    rho0_padded = _require_cuda(
        cupy,
        metrics["rho0_padded"],
        "rho0_padded",
        (nk + 2, nj + 2 * ng, ni + 2 * ng),
        writable=False,
    )
    profiles = {
        "sigma": _require_cuda(
            cupy, vertical.sigma, "sigma", (nk + 2,), writable=False
        ),
        "sigmaf": _require_cuda(
            cupy, vertical.sigmaf, "sigmaf", (nk + 3,), writable=False
        ),
        "rds": _require_cuda(
            cupy, vertical.rds, "rds", (nk + 2,), writable=False
        ),
        "rdsf": _require_cuda(
            cupy, vertical.rdsf, "rdsf", (nk + 3,), writable=False
        ),
    }
    if generated is not None:
        require_generated_vertical(vertical, nk=nk, cupy=cupy)
    elif allow_uniform_idealized_validation:
        if (
            vertical.identity.get("fixture_class")
            != "source-audited-flat-idealized-uniform-200m"
            or vertical.ztop_f32 != 15000.0
            or vertical.dz_reference_f32 != 200.0
        ):
            raise ValueError(
                "uniform idealized momentum requires the source-audited flat 200 m profiles"
            )
    elif not allow_compact_validation and (
        vertical.identity.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256
        or vertical.identity.get("proof_sha256") != GM1_CHILD_STRETCHED_PROOF_SHA256
        or vertical.ztop_f32 != 15000.0
    ):
        raise ValueError("production stretched momentum requires the sealed child profiles")
    flat = bool(vertical.flat_column)
    if flat:
        from cm1core.cuda_gm1_child_stretched_metrics import require_flat_column

        mh_column, mf_column, rdz = require_flat_column(vertical, nk=nk, cupy=cupy)
        profiles["mh"], profiles["mf"] = mh_column, mf_column
    divx = (
        cupy.empty((nk, nj, ni), dtype=cupy.float32)
        if divx is None
        else _require_cuda(cupy, divx, "divx", (nk, nj, ni), writable=True)
    )
    # The same rule the scalar-dz family applies: mutable state, scratch and
    # profiles own their storage outright, and two *metric* names may share a
    # pointer only where the immutable map carries a source proof that both
    # carriers are the same constant word (cuda_immutable_metrics).  A Rust
    # initializer that publishes rr0 and rrf0 as one constant plane is not an
    # aliasing defect, and refusing it here would refuse the whole lane.
    from cm1core.cuda_immutable_metrics import require_cuda_argument_aliases

    require_cuda_argument_aliases(
        {
            **{f"state.{name}": value for name, value in a.items()},
            **{f"metric.{name}": value for name, value in m.items()},
            "metric.rho0_padded": rho0_padded,
            **{f"profile.{name}": value for name, value in profiles.items()},
            "divx": divx,
        },
        immutable_metrics=metrics,
        message="stretched momentum arrays and unproved metrics must not alias",
    )
    if compiled is None:
        compiled = (
            compile_cuda_gm1_flat_stretched_momentum()
            if flat
            else compile_cuda_gm1_child_stretched_momentum()
        )
    if isinstance(compiled, CompiledFlatStretchedMomentumKernels) != flat:
        raise ValueError(
            "the momentum image and the vertical column disagree: a flat column "
            "(terrain_flag=.false.) runs solve2.F:493-520,605-617 and adv.F's "
            "rdz*mh/rdz*mf statements, a terrain column the rdsf/rds ones"
        )
    if compiled.target_arch != vertical.target_arch:
        raise ValueError("stretched momentum and vertical CUDA images target different architectures")
    stream = stream or cupy.cuda.get_current_stream()
    horizontal, vertical_mass, divergence, advu, advv, advw = compiled.kernels
    counts = (
        nk * nj * (ni + 1) + nk * (nj + 1) * ni,
        (nk + 1) * nj * ni,
        nk * nj * ni,
        *momentum_advection5_counts(ni, nj, nk, lateral_sides),
    )
    block = (CUDA_MOMENTUM_ADVECTION5_THREADS, 1, 1)
    grids = tuple(
        ((count + CUDA_MOMENTUM_ADVECTION5_THREADS - 1) // CUDA_MOMENTUM_ADVECTION5_THREADS, 1, 1)
        for count in counts
    )
    u32, f32 = cupy.uint32, cupy.float32
    rrw_boundary_receipt: Mapping[str, object] | None = None
    post_divergence_receipt: Mapping[str, object] | None = None
    with stream:
        if flat:
            horizontal(
                grids[0], block,
                (a["u3d"], a["v3d"], rho0_padded, a["rru"], a["rrv"],
                 u32(ni), u32(nj), u32(nk), u32(ng)),
                stream=stream,
            )
            vertical_mass(
                grids[1], block,
                (a["w3d"], m["rf0"], a["rrw"], u32(ni), u32(nj), u32(nk), u32(ng)),
                stream=stream,
            )
        else:
            horizontal(
                grids[0], block,
                (a["u3d"], a["v3d"], rho0_padded, m["rgzu"], m["rgzv"],
                 a["rru"], a["rrv"], u32(ni), u32(nj), u32(nk), u32(ng)),
                stream=stream,
            )
            vertical_mass(
                grids[1], block,
                (a["w3d"], m["rf0"], m["dzdx"], m["dzdy"], m["gz"], a["rru"],
                 a["rrv"], a["rrw"], profiles["sigma"], profiles["sigmaf"],
                 profiles["rds"],
                 u32(ni), u32(nj), u32(nk), u32(ng), f32(vertical.ztop_f32)),
                stream=stream,
            )
        if rrw_boundary_hook is not None:
            rrw_boundary_receipt = rrw_boundary_hook(
                a["rrw"], stream=stream, ni=ni, nj=nj, nk=nk
            )
            if not isinstance(rrw_boundary_receipt, Mapping):
                raise TypeError("rrw_boundary_hook must return a receipt mapping")
        if flat:
            divergence(
                grids[2], block,
                (a["rru"], a["rrv"], a["rrw"], profiles["mh"], divx, u32(ni),
                 u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(rdz),
                 f32(CM1_R21_SMEPS_F32)),
                stream=stream,
            )
        else:
            divergence(
                grids[2], block,
                (a["rru"], a["rrv"], a["rrw"], profiles["rdsf"], divx, u32(ni),
                 u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(CM1_SMEPS_F32)),
                stream=stream,
            )
        if post_divergence_hook is not None:
            post_divergence_receipt = post_divergence_hook(
                a,
                stream=stream,
                ni=ni,
                nj=nj,
                nk=nk,
                nrk=nrk,
                ng=ng,
            )
            if not isinstance(post_divergence_receipt, Mapping):
                raise TypeError("post_divergence_hook must return a receipt mapping")
        if flat:
            tail = (u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), f32(rdz), u32(obc))
            advu(
                grids[3], block,
                (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rr0"],
                 m["c2"], a["vh"], profiles["mh"], a["uten"], *tail),
                stream=stream,
            )
            advv(
                grids[4], block,
                (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rr0"],
                 m["c2"], a["uh"], profiles["mh"], a["vten"], *tail),
                stream=stream,
            )
            advw(
                grids[5], block,
                (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"],
                 m["rrf0"], a["uh"], a["vh"], profiles["mf"], a["wten"], *tail),
                stream=stream,
            )
        else:
            advu(
                grids[3], block,
                (a["u3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"],
                 m["c2"], m["gzu"], a["vh"], profiles["rdsf"], a["uten"], u32(ni),
                 u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), u32(obc)),
                stream=stream,
            )
            advv(
                grids[4], block,
                (a["v3d"], a["rru"], a["rrv"], a["rrw"], divx, m["rho0_3d"],
                 m["c2"], m["gzv"], a["uh"], profiles["rdsf"], a["vten"], u32(ni),
                 u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), u32(obc)),
                stream=stream,
            )
            advw(
                grids[5], block,
                (a["w3d"], a["rru"], a["rrv"], a["rrw"], divx, m["c2"],
                 m["rrf0"], m["gz"], a["uh"], a["vh"], profiles["rds"], a["wten"],
                 u32(ni), u32(nj), u32(nk), u32(ng), f32(dx), f32(dy), u32(obc)),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA,
        "abi": GM1_FLAT_STRETCHED_MOMENTUM_ABI if flat else GM1_CHILD_STRETCHED_MOMENTUM_ABI,
        "source_sha256": (
            GM1_FLAT_STRETCHED_MOMENTUM_SOURCE_SHA256
            if flat
            else GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256
        ),
        "legacy_horizontal_family_sha256": CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256,
        "authority": {
            "src/adv.F": CM1_ADV_F_SHA256,
            "src/adv_routines.F": CM1_ADV_ROUTINES_F_SHA256,
            "src/solve2.F": CM1_SOLVE2_F_SHA256,
            "anchors": "adv.F selected terrain u/v/w fifth-order branches; solve2.F:490-627",
        },
        "vertical_metrics": dict(vertical.identity),
        "selection": {
            "hadvordrv": 5, "vadvordrv": 5, "advwenov": 0, "terrain": not flat,
            "stretch_z": 1, "axisymmetric": 0,
            "four_open": lateral_sides == MOMENTUM_ALL_OPEN_SIDES,
            "lateral_boundaries": list(lateral_sides), "nrk": nrk,
            "compact_validation": allow_compact_validation,
            "uniform_idealized_validation": allow_uniform_idealized_validation,
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "boundary_faces": momentum_boundary_faces(lateral_sides),
        "target_arch": compiled.target_arch,
        "uniform_sigma_kernel_used": False,
        "rrw_open_boundary": (
            dict(rrw_boundary_receipt)
            if rrw_boundary_receipt is not None
            else {"performed": False}
        ),
        "ready_event_recorded": True,
    }
    if post_divergence_receipt is not None:
        receipt["post_divergence_hook"] = dict(post_divergence_receipt)
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return CudaMomentumAdvection5Launch(divx, event, compiled, receipt)


__all__ = (
    "CompiledFlatStretchedMomentumKernels",
    "GM1_FLAT_STRETCHED_MOMENTUM_ABI",
    "GM1_FLAT_STRETCHED_MOMENTUM_SOURCE",
    "GM1_FLAT_STRETCHED_MOMENTUM_SOURCE_SHA256",
    "compile_cuda_gm1_flat_stretched_momentum",
    "GM1_CHILD_STRETCHED_MOMENTUM_ABI",
    "GM1_CHILD_STRETCHED_MOMENTUM_SCHEMA",
    "GM1_CHILD_STRETCHED_MOMENTUM_SOURCE",
    "GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256",
    "compile_cuda_gm1_child_stretched_momentum",
    "launch_cuda_gm1_child_stretched_momentum5",
)
