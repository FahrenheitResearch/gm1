"""Profile-aware CM1 r21.1 psolver=3/all-open acoustic stage.

This is the stretched-grid successor to :mod:`cuda_psolver3_terrain_open`.
That operator reconstructs a uniform vertical coordinate inline from one
scalar ``dz`` -- ``sig(k,dz)``, ``sigf(k,dz)``, ``rds(dz)`` -- so it is bound
to ``stretch_z=0`` by construction.  This one binds the ``sigma``, ``sigmaf``,
``rds`` and ``rdsf`` device profiles of whatever coordinate the run declared.
No scalar-dz fallback exists.

The domain is the caller's.  ``nk``, ``ni``, ``nj``, ``dx``, ``dy`` and ``dt``
are admitted by their stencil and arithmetic requirements -- four scalar levels
for ``param.F:8985-9060``, three columns for the open lateral boundary, and
positive finite spacings -- not by a fixture's dimensions.  The sealed 15 km,
75-level, 50 m child keeps every identity check it has ever had, because a
vertical binding that claims to be that grid is still held to its three sealed
hashes.

Only an explicit RK stage is exposed.  The caller owns the large-step
coordinator, A/current carriers, held tendencies, and synchronization.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_psolver3_terrain_open import (
    CM1_ALPH_F32,
    CM1_CP_F32,
    CM1_KDIV_F32,
    CM1_SMEPS_F32,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_BYTES,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
    TerrainOpenSoundAblations,
    _canonical_arch,
    _positive,
    _require_device_arrays,
    _require_device_metrics,
    psolver3_terrain_open_source,
    terrain_open_acoustic_workspace_shapes,
    terrain_open_sound_shapes,
)
from cm1core.cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_open_boundary import (
    CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE,
    CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_BYTES,
    CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
)


CUDA_GM1_CHILD_PSOLVER3_OPEN_SCHEMA = "gm1.cuda-child-psolver3-open-stretch-z1/v1"
CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI = "gm1.cuda-child-psolver3-open-stretch-z1-kernels/v1"
CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA = (
    "gm1.cuda-child-psolver3-open-stretch-z1-stage/v1"
)
GM1_CHILD_STATIC_COMBINED_SHA256 = (
    "2ec8b6a5a48a81017527e66c4fa6b264f1476b737fa2b0a02f7795012877c0a9"
)
GM1_CHILD_STRETCHED_GRID_SHA256 = (
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
)
GM1_CHILD_STRETCHED_PROOF_SHA256 = (
    "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
)
GM1_CHILD_STRETCHED_PROFILE_SHA256 = {
    "sigma": "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3",
    "sigmaf": "ceb1624e4dc10eb31562c7e8003f939d6d157e1ecd73d1dee479de5459f2a90e",
    "rds": "b995cfc800e8b69c257d1f0e9e328c315d91652f7b0a86a748af49eadd4306fe",
    "rdsf": "008898d0ed25941c735f0f7a23746d3a7eea562d12d01ad7cc07b6d50ed6f416",
}
GM1_CHILD_ZTOP_F32 = 15000.0
GM1_CHILD_DZ_REFERENCE_F32 = 100.0

SOURCE_ANCHORS = {
    "terrain_coefficients": "src/sound.F:145-188",
    "terrain_pressure_gradient": "src/sound.F:364-410",
    "terrain_divergence": "src/sound.F:446-534",
    "terrain_pressure_advection": "src/sound.F:596-654",
    "implicit_vertical_solve": "src/sound.F:659-730",
    "open_consumption": "src/sound.F:253-302;src/bc.F:2392-2684",
    "radiation_history": "src/solve2.F:1258-1271;src/bc.F:2054-2172",
    "profile_authority": "src/param.F:8464-8492;8603-8627",
}

CUDA_GM1_CHILD_PSOLVER3_OPEN_CLAIM_LIMIT = (
    "one CM1 r21.1 serial terrain/all-four-open psolver=3 acoustic RK stage "
    "on the sealed 75-level 15 km stretch_z=1 GM1 child coordinate, using "
    "device sigma/sigmaf/rds/rdsf profiles, source-valid padded base density, "
    "native small-step cadence, transformed pressure operators, implicit "
    "column solve, radiation-history production, and family fills; the caller "
    "owns large-step physics/advection, state cadence, and trajectory claims"
)


@dataclass(frozen=True, slots=True)
class Gm1ChildPsolver3OpenSelection:
    psolver: int = 3
    boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)
    bottom_boundary: int = 1
    top_boundary: int = 1
    terrain: bool = True
    stretch_z: int = 1
    axisymmetric: int = 0
    mpi: bool = False
    source_backed_base_halos: bool = True
    product_admission: bool = True


DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION = Gm1ChildPsolver3OpenSelection()


def _validate_selection(selection: Gm1ChildPsolver3OpenSelection) -> None:
    if not isinstance(selection, Gm1ChildPsolver3OpenSelection):
        raise TypeError("selection must be Gm1ChildPsolver3OpenSelection")
    # The operator binds sigma/sigmaf/rds/rdsf and rebuilds no level from a
    # scalar dz, so every param.F:7814-8139 branch GM1 generates reaches it
    # unchanged; the CM1 words that do change the operator -- psolver, the six
    # boundary selectors, terrain, axisymm, MPI -- stay pinned.
    from .cm1_vertical_grid import ADMITTED_STRETCH_Z

    if selection.stretch_z not in ADMITTED_STRETCH_Z:
        raise ValueError(
            f"stretch_z must be one of {ADMITTED_STRETCH_Z}; param.F:7814-8139 "
            "defines no other branch this port generates"
        )
    from dataclasses import replace as _replace

    if _replace(selection, stretch_z=1) != DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION:
        raise ValueError(
            "only the sealed serial terrain/four-open child is admitted; "
            "psolver, the boundary selectors, terrain, axisymm and MPI are not "
            "widened here"
        )


def _replace_exact(source: str, old: str, new: str, *, count: int = 1) -> str:
    observed = source.count(old)
    if observed != count:
        raise RuntimeError(
            f"upstream sealed CUDA transform expected {count} occurrence(s), found {observed}"
        )
    return source.replace(old, new)


def _build_profile_core_source(*, retain_column_diagnostics: bool = True) -> str:
    """The profile-bound core, with the same diagnostic choice the legacy has.

    ``psolver3_terrain_open_source`` already selects whether the tridiagonal
    coefficient/pivot/residual arrays are written; that choice is upstream of
    every vertical-metric substitution below, so it is taken there and the
    substitutions then apply to whichever variant results.  A caller running
    under a device budget keeps the diagnostic-free variant, whose six column
    arrays are then never allocated at full extent -- 173 MB on the 300x300x75
    child, which is the difference between fitting and not.
    """

    if (
        hashlib.sha256(CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_BYTES).hexdigest()
        != CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256
    ):
        raise RuntimeError("legacy psolver3 source differs from its sealed pin")
    source = psolver3_terrain_open_source(
        retain_column_diagnostics=retain_column_diagnostics
    )
    source = _replace_exact(
        source,
        "// gpuwm-cm1.cuda-psolver3-terrain-open-kernels/v1",
        f"// {CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI}",
    )
    for old, new in (
        ("cm1_tos_prepare_v1", "cm1_child_szt_prepare_v1"),
        ("cm1_tos_pgrad_v1", "cm1_child_szt_pgrad_v1"),
        ("cm1_tos_column_v1", "cm1_child_szt_column_v1"),
        ("cm1_tos_average_v1", "cm1_child_szt_average_v1"),
    ):
        source = _replace_exact(source, old, new)
    for helper in (
        "__device__ __forceinline__ float sig(unsigned int k,float dz){return m32(s32((float)k,0.5f),dz);}\n",
        "__device__ __forceinline__ float sigf(unsigned int k,float dz){return m32((float)(k-1u),dz);}\n",
        "__device__ __forceinline__ float rds(float dz){return d32(1.0f,dz);}\n",
    ):
        source = _replace_exact(source, helper, "")

    source = _replace_exact(
        source,
        "unsigned int nk,unsigned int ng,float dz){\n if(kf==1u||kf==nk+1u)return 0.0f;float r2=m32(s32(sigf(kf,dz),sig(kf-1u,dz)),rds(dz)),r1=s32(1.0f,r2);",
        "unsigned int nk,unsigned int ng,const float* sigma,const float* sigmaf,const float* rds,float ztop){\n if(kf==1u||kf==nk+1u)return 0.0f;float r2=m32(s32(sigmaf[kf],sigma[kf-1u]),rds[kf]),r1=s32(1.0f,r2);",
        count=2,
    )
    source = _replace_exact(
        source,
        "float zt=m32((float)nk,dz),height=m32(s32(sigf(kf,dz),zt),d32(1.0f,zt));",
        "float height=m32(s32(sigmaf[kf],ztop),d32(1.0f,ztop));",
        count=2,
    )

    # Source-valid base density is padded; face coupling is exactly the
    # sound.F:604-613 arithmetic rather than the legacy broadcast ablation.
    source = _replace_exact(
        source,
        "__device__ __forceinline__ float xurho(const float* u,const float* rgzu,const float* rho,unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj,unsigned int ng){return m32(rho[(unsigned long long)(k-1u)*nj*ni],xu(u,rgzu,k,j,f,ni,nj,ng));}\n__device__ __forceinline__ float yvrho(const float* v,const float* rgzv,const float* rho,unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){return m32(rho[(unsigned long long)(k-1u)*nj*ni],yv(v,rgzv,k,f,i,ni,nj,ng));}",
        "__device__ __forceinline__ float xurho(const float* u,const float* rgzu,const float* rho,unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj,unsigned int ng){unsigned int n=ni+2u*ng,y=j+ng;float dens=m32(0.5f,a32(rho[idx3(k,y,f+ng-1u,nj+2u*ng,n)],rho[idx3(k,y,f+ng,nj+2u*ng,n)]));return m32(dens,xu(u,rgzu,k,j,f,ni,nj,ng));}\n__device__ __forceinline__ float yvrho(const float* v,const float* rgzv,const float* rho,unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){unsigned int n=ni+2u*ng,x=i+ng;float dens=m32(0.5f,a32(rho[idx3(k,f+ng-1u,x,nj+2u*ng,n)],rho[idx3(k,f+ng,x,nj+2u*ng,n)]));return m32(dens,yv(v,rgzv,k,f,i,ni,nj,ng));}",
    )

    source = _replace_exact(
        source,
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,float dts,float dz,float cp,float beta){",
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,float dts,const float* rds,const float* rdsf,float cp,float beta){",
    )
    source = _replace_exact(
        source, "float tem=m32(dts,rds(dz));", "float tem=m32(dts,rdsf[k]);"
    )
    source = _replace_exact(source, "rds(dz)),rr0[mi]", "rdsf[k]),rr0[mi]")
    source = _replace_exact(
        source,
        "float tem1=m32(m32(dts,rds(dz)),cp);",
        "float tem1=m32(m32(dts,rds[k]),cp);",
    )

    source = _replace_exact(
        source,
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dts,float rdx,float rdy,float dz,float cp){",
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dts,float rdx,float rdy,const float* rds,float cp){",
    )
    source = _replace_exact(
        source, "),rds(dz)):0.0f,dphl=", "),rds[k+1u]):0.0f,dphl=", count=2
    )
    source = _replace_exact(
        source, "),rds(dz)):0.0f,dpl=", "),rds[k+1u]):0.0f,dpl=", count=2
    )
    source = _replace_exact(
        source, "),rds(dz)):0.0f,dpll=", "),rds[k]):0.0f,dpll=", count=2
    )
    source = _replace_exact(
        source, "),rds(dz)):0.0f;float vs=", "),rds[k]):0.0f;float vs=", count=2
    )

    source = _replace_exact(
        source,
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,unsigned int nstep,unsigned int nloop,float dts,float rdx,float rdy,float dz,float alph,float beta,float kdiv,float smeps,unsigned int do_initial,unsigned int do_div,unsigned int do_adv,unsigned int do_implicit,unsigned int do_update){",
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,unsigned int nstep,unsigned int nloop,float dts,float rdx,float rdy,const float* sigma,const float* sigmaf,const float* rds,const float* rdsf,float ztop,float alph,float beta,float kdiv,float smeps,unsigned int do_initial,unsigned int do_div,unsigned int do_adv,unsigned int do_implicit,unsigned int do_update){",
    )
    source = _replace_exact(
        source,
        ",ni,nj,nk,ng,dz),d3div(",
        ",ni,nj,nk,ng,sigma,sigmaf,rds,ztop),d3div(",
    )
    source = _replace_exact(
        source,
        ",ni,nj,nk,ng,dz)),rds(dz));",
        ",ni,nj,nk,ng,sigma,sigmaf,rds,ztop)),rdsf[k]);",
    )
    source = _replace_exact(
        source,
        ",ni,nj,nk,ng,dz),m32(beta,w[",
        ",ni,nj,nk,ng,sigma,sigmaf,rds,ztop),m32(beta,w[",
        count=2,
    )
    if (
        "rds(dz)" in source
        or "sigf(" in source
        or "sig(" in source
        or "float dz" in source
    ):
        raise RuntimeError(
            "profile-aware core still contains a scalar vertical fallback"
        )
    source += r"""
extern "C" __global__ void cm1_child_szt_profile_probe_v1(
 const float* sigma,const float* sigmaf,const float* rds,const float* rdsf,
 float* out,unsigned int k,float ztop){
 if(blockIdx.x||threadIdx.x)return;
 float r2=m32(s32(sigmaf[k],sigma[k-1u]),rds[k]);
 out[0]=s32(1.0f,r2);out[1]=r2;
 out[2]=m32(s32(sigmaf[k],ztop),d32(1.0f,ztop));
 out[3]=rds[k];out[4]=rdsf[k];
}
"""
    return source


def _build_profile_boundary_source() -> str:
    if (
        hashlib.sha256(CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_BYTES).hexdigest()
        != CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256
    ):
        raise RuntimeError("legacy terrain-open source differs from its sealed pin")
    source = CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE
    source = _replace_exact(
        source,
        "// gpuwm-cm1.cuda-terrain-open-boundary-kernels/v1",
        "// gm1.cuda-child-psolver3-open-stretch-z1-boundary-kernels/v1",
    )
    for old, new in (
        ("cm1_terrain_ssopen_snapshot_v2", "cm1_child_szt_ssopen_snapshot_v1"),
        ("cm1_terrain_ssopen_update_v2", "cm1_child_szt_ssopen_update_v1"),
        ("cm1_terrain_bcwsfc_physical_v1", "cm1_child_szt_bcwsfc_physical_v1"),
    ):
        source = _replace_exact(source, old, new)
    source = _replace_exact(
        source,
        "unsigned int nk,unsigned int ny,unsigned int nx,float dz){",
        "unsigned int nk,unsigned int ny,unsigned int nx,const float* sigma,const float* sigmaf,const float* rds){",
    )
    source = _replace_exact(
        source,
        "const float sigmaf=m32((float)(kf-1u),dz),sigma=m32(s32((float)(kf-1u),0.5f),dz),rds=1.0f/dz;\n  const float r2=m32(s32(sigmaf,sigma),rds),r1=s32(1.0f,r2);",
        "const float r2=m32(s32(sigmaf[kf],sigma[kf-1u]),rds[kf]),r1=s32(1.0f,r2);",
    )
    source = _replace_exact(
        source,
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){",
        "unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,const float* sigma,const float* sigmaf,const float* rds){",
    )
    source = _replace_exact(
        source, ",nk,nyu,nxu,dz)", ",nk,nyu,nxu,sigma,sigmaf,rds)", count=4
    )
    source = _replace_exact(
        source, ",nk,nyv,nxv,dz)", ",nk,nyv,nxv,sigma,sigmaf,rds)", count=4
    )
    source = _replace_exact(
        source,
        "float dts,float rdx,float rdy,float dz){",
        "float dts,float rdx,float rdy,const float* rdsf){",
    )
    source = _replace_exact(source, "1.0f/dz", "rdsf[k]", count=4)
    if "float dz" in source or "1.0f/dz" in source:
        raise RuntimeError(
            "profile-aware boundary still contains a scalar vertical fallback"
        )
    return source


# ---------------------------------------------------------------------------
# The flat stretched column -- terrain_flag=.false.
# ---------------------------------------------------------------------------
#
# The two texts above are sound.F's terrain branch.  GM1's Cartesian
# stretch_z lane is a flat column, and there CM1 r21.1 runs
#
# * sound.F:119-143  qk = ((dts*rdz)*mh(1,1,k))*ppterm, pk2 and pk1 with
#   r2/r1 = ((((dts*rdz)*mh)*rr0)*rf0(k+1 or k)) and the column c1/c2, and
#   mm = (((dts*rdz)*cp)*mf(1,1,k))*(c2*thv(k)+c1*thv(k-1)) at every k;
# * sound.F:318-342  u3d = u3d + dtu(i,j)*(uten - ((tem1*dpp)*uf(i))*
#   (thv(i)+thv(i-1))) over i=1..ni+1, j=1..nj+1, with tem1 = (rdx*cp)*0.5
#   and dtu = dts*dtu0 (param.F:9182-9219: zero on an open side and on the
#   row outside the grid), and the same for v;
# * sound.F:491-509  fpk = ((dts*(ppten-ppterm*div)) + H) + beta*W, where
#   div has no vertical term, H is the flat pressure advection
#   -((X*temx)*uh) - ((Y*temy)*vh) and W = w(k)*(pk1+qk) + w(k+1)*(pk2-qk);
#   no terrain advection block (sound.F:569-641) and no bcwsfc;
# * sound.F:742-765  the update with w(k) and w(k+1) as they are, not zeroed
#   at k=1 and k=nk;
# * bc.F:2410-2419 and its three twins  u(1) = u(1) + dts*(((-radbcw*
#   (u(2)-u(1)))*rdx)*uh(1) + uten(1)), with no dum1 reconstruction.
#
# mh and mf are the column's (nk+2,) and (nk+3,) CM1-indexed profiles, rdz
# is param.F:6327's 1.0/dz of the metric reference, the base state is read
# at CM1's (1,1,k), smeps is cm1.F:311's 1.0e-30, and uh = vh = uf = vf = 1
# on this lane's uniform horizontal grid.
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_ABI = "gm1.cuda-flat-stretched-psolver3-open-kernels/v1"
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_ABI = (
    "gm1.cuda-flat-stretched-psolver3-open-boundary-kernels/v1"
)
CM1_R21_SMEPS_F32 = 1.0e-30

_FLAT_PREPARE_METRICS = (
    "const unsigned long long c0=(unsigned long long)(k-1u)*nj*ni,c1i=(unsigned long long)k*nj*ni;"
    "float tem=m32(m32(dts,rdz),mh[k]);"
    "float r2=m32(m32(m32(m32(dts,rdz),mh[k]),rr0[c0]),rf0[c1i]);"
    "float r1=m32(m32(m32(m32(dts,rdz),mh[k]),rr0[c0]),rf0[c0]);"
    "float tem1=m32(m32(m32(dts,rdz),cp),mf[k]);"
    "float c1a=c1[c0],c1b=c1[c1i],c2a=c2[c0],c2b=c2[c1i];"
    "qk[p]=m32(tem,ppterm[p]);"
    "pk2[p]=m32(r2,a32(m32(-c2b,piadv[pidx(k+1u,pj,pi,ni,nj)]),m32(s32(1.0f,c1b),piadv[p])));"
    "pk1[p]=m32(r1,a32(m32(c1a,piadv[pidx(k-1u,pj,pi,ni,nj)]),m32(s32(c2a,1.0f),piadv[p])));"
    "mm[p]=m32(tem1,a32(m32(c2a,thv[p]),m32(c1a,thv[pidx(k-1u,pj,pi,ni,nj)])));"
)

_FLAT_PGRAD = r'''extern "C" __global__ void cm1_child_szt_pgrad_v1(const float* ppd,const float* thv,const float* uten,const float* vten,float* u,float* v,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dts,float rdx,float rdy,float cp,unsigned int obc){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*(nj+1u)*(ni+1u);
 const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;const float tem1=m32(m32(rdx,cp),0.5f),tem2=m32(m32(rdy,cp),0.5f);
 const bool pw=(obc&15u)==1u,pe=((obc>>4)&15u)==1u,ps=((obc>>8)&15u)==1u,pn=((obc>>12)&15u)==1u;
 for(unsigned long long z=first;z<total;z+=step){unsigned int i=(unsigned int)(z%(ni+1u))+1u;unsigned long long q=z/(ni+1u);unsigned int j=(unsigned int)(q%(nj+1u))+1u,k=(unsigned int)(q/(nj+1u))+1u;
  float wu=(j<=nj&&(i>1u||pw)&&(i<ni+1u||pe))?1.0f:0.0f,wv=(i<=ni&&(j>1u||ps)&&(j<nj+1u||pn))?1.0f:0.0f;
  float pu=m32(tem1,s32(ppd[pidx(k,j,i,ni,nj)],ppd[pidx(k,j,i-1u,ni,nj)]));pu=m32(pu,a32(thv[pidx(k,j,i,ni,nj)],thv[pidx(k,j,i-1u,ni,nj)]));
  unsigned long long xu0=idx3(k,j-1u+ng,i-1u+ng,nyu,nxu);u[xu0]=a32(u[xu0],m32(m32(dts,wu),s32(uten[xu0],pu)));
  float pv=m32(tem2,s32(ppd[pidx(k,j,i,ni,nj)],ppd[pidx(k,j-1u,i,ni,nj)]));pv=m32(pv,a32(thv[pidx(k,j,i,ni,nj)],thv[pidx(k,j-1u,i,ni,nj)]));
  unsigned long long xv0=idx3(k,j-1u+ng,i-1u+ng,nyv,nxv);v[xv0]=a32(v[xv0],m32(m32(dts,wv),s32(vten[xv0],pv)));}
}
'''

_FLAT_COLUMN_SIGNATURE = (
    "const float* piadv,const float* ppterm,const float* ppten,const float* wten,\n"
    " const float* u,const float* v,float* w,float* pp3d,float* ppd,float* ppx,float* qk,float* pk1,float* pk2,float* mm,float* fpk,float* divh,float* hadvp,float* ec,float* ed,float* aa0,float* bb0,float* cc0,float* dd0,float* piv,float* res,\n"
    " unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,unsigned int nstep,unsigned int nloop,float dts,float rdx,float rdy,float alph,float beta,float kdiv,float smeps,unsigned int do_initial,unsigned int do_div,unsigned int do_adv,unsigned int do_implicit,unsigned int do_update){"
)


def _flat_initial(retain: bool) -> str:
    store = "divh[p]=div;hadvp[p]=hp;" if retain else ""
    return (
        "if(do_initial){unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;"
        "float div=0.0f;if(do_div){div=a32(m32(s32(u[idx3(k,y,x+1u,nyu,nxu)],u[idx3(k,y,x,nyu,nxu)]),rdx),"
        "m32(s32(v[idx3(k,y+1u,x,nyv,nxv)],v[idx3(k,y,x,nyv,nxv)]),rdy));if(fabsf(div)<smeps)div=0.0f;}"
        "float hx=a32(m32(u[idx3(k,y,x,nyu,nxu)],s32(piadv[p],piadv[pidx(k,pj,pi-1u,ni,nj)])),"
        "m32(u[idx3(k,y,x+1u,nyu,nxu)],s32(piadv[pidx(k,pj,pi+1u,ni,nj)],piadv[p])));hx=m32(-hx,temx);"
        "float hy=a32(m32(v[idx3(k,y,x,nyv,nxv)],s32(piadv[p],piadv[pidx(k,pj-1u,pi,ni,nj)])),"
        "m32(v[idx3(k,y+1u,x,nyv,nxv)],s32(piadv[pidx(k,pj+1u,pi,ni,nj)],piadv[p])));hy=m32(-hy,temy);"
        "float hp=a32(hx,hy),vert=a32(m32(w[idx3(k,y,x,nyw,nxw)],a32(pk1[p],qk[p])),m32(w[idx3(k+1u,y,x,nyw,nxw)],s32(pk2[p],qk[p])));"
        f"{store}fpk[p]=a32(a32(m32(dts,s32(ppten[p],m32(ppterm[p],div))),hp),m32(beta,vert));}}"
    )


def _build_flat_profile_core_source(*, retain_column_diagnostics: bool = True) -> str:
    source = _build_profile_core_source(retain_column_diagnostics=retain_column_diagnostics)
    source = _replace_exact(
        source,
        f"// {CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI}",
        f"// {CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_ABI}",
    )
    helpers = source.index("__device__ __forceinline__ float xu(")
    prepare = source.index('extern "C" __global__ void cm1_child_szt_prepare_v1(')
    source = source[:helpers] + source[prepare:]
    source = _replace_exact(
        source,
        " const float* rr0,const float* rf0,const float* c1,const float* c2,const float* gz,\n",
        " const float* rr0,const float* rf0,const float* c1,const float* c2,\n",
    )
    source = _replace_exact(
        source,
        "unsigned int nrk,float dts,const float* rds,const float* rdsf,float cp,float beta){",
        "unsigned int nrk,float dts,const float* mh,const float* mf,float rdz,float cp,float beta){",
    )
    metrics = source.index("  float tem=m32(dts,rdsf[k]);")
    tail = source.index("if(k==1u)pk1[p]=0.0f;", metrics)
    source = source[:metrics] + "  " + _FLAT_PREPARE_METRICS + source[tail:]
    pgrad = source.index('extern "C" __global__ void cm1_child_szt_pgrad_v1(')
    column = source.index('extern "C" __global__ void cm1_child_szt_column_v1(')
    source = source[:pgrad] + _FLAT_PGRAD + source[column:]
    column = source.index('extern "C" __global__ void cm1_child_szt_column_v1(')
    head = source.index("const float* piadv,const float* ppterm,const float* ppten,const float* wten,const float* rho,", column)
    body = source.index("unsigned int do_update){", head) + len("unsigned int do_update){")
    source = source[:head] + _FLAT_COLUMN_SIGNATURE + source[body:]
    initial = source.index("if(do_initial){")
    implicit = source.index(" if(do_implicit){", initial)
    source = source[:initial] + _flat_initial(retain_column_diagnostics) + "}\n" + source[implicit:]
    source = _replace_exact(
        source,
        "float w1=(k==1u)?0.0f:w[idx3(k,y,x,nyw,nxw)],w2=(k==nk)?0.0f:w[idx3(k+1u,y,x,nyw,nxw)],old=pp3d[p]",
        "float w1=w[idx3(k,y,x,nyw,nxw)],w2=w[idx3(k+1u,y,x,nyw,nxw)],old=pp3d[p]",
    )
    probe = source.index('extern "C" __global__ void cm1_child_szt_profile_probe_v1(')
    source = source[:probe] + (
        'extern "C" __global__ void cm1_child_szt_profile_probe_v1(\n'
        " const float* mh,const float* mf,float* out,unsigned int k,float rdz){\n"
        " if(blockIdx.x||threadIdx.x)return;\n"
        " out[0]=rdz;out[1]=mh[k];out[2]=mf[k];out[3]=m32(rdz,mh[k]);out[4]=m32(rdz,mf[k]);\n"
        "}\n"
    )
    for token in ("rdsf", "rds[", "rds(", "sigma", "ztop", "dzdx", "dzdy", "rgz", "gz[", "gxu", "gyv", "rrf0", "xurho"):
        if token in source:
            raise RuntimeError(f"terrain-branch operand {token!r} survived the flat acoustic text")
    return source


def _build_flat_profile_boundary_source() -> str:
    """bc.F:2410-2419, 2485-2494, 2560-2569 and 2635-2644: the flat ssopenbc."""

    return (
        f"// {CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_ABI}\n"
        'static_assert(sizeof(float)==4,"CM1 default REAL requires float32");\n'
        "__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}\n"
        "__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}\n"
        "__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}\n"
        "__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}\n"
        'extern "C" __global__ void cm1_child_szt_ssopen_update_v1(\n'
        " const float* uten,const float* vten,const float* uh,const float* vh,\n"
        " const float* rw,const float* re,const float* rs,const float* rn,\n"
        " float* u,float* v,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,\n"
        " float dts,float rdx,float rdy,unsigned int obc){\n"
        " const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;\n"
        " const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;\n"
        " const unsigned long long we=(unsigned long long)nk*nj,sn=(unsigned long long)nk*ni;\n"
        " for(unsigned long long q=first;q<we;q+=step){\n"
        "   const unsigned int j0=(unsigned int)(q%nj),k=(unsigned int)(q/nj)+1u,y=j0+ng,xw=ng,xi=ng+1u,xe=ni+ng,xin=ni+ng-1u;\n"
        "   unsigned long long b=i3(k,y,xw,nyu,nxu);float t;\n"
        "   if((obc&15u)==2u){t=a32(m32(m32(m32(-rw[(unsigned long long)k*(nj+2u*ng)+y],s32(u[i3(k,y,xi,nyu,nxu)],u[b])),rdx),uh[ng]),uten[b]);u[b]=a32(u[b],m32(dts,t));}\n"
        "   b=i3(k,y,xe,nyu,nxu);\n"
        "   if(((obc>>4)&15u)==2u){t=a32(m32(m32(m32(-re[(unsigned long long)k*(nj+2u*ng)+y],s32(u[b],u[i3(k,y,xin,nyu,nxu)])),rdx),uh[ni+ng-1u]),uten[b]);u[b]=a32(u[b],m32(dts,t));}\n"
        " }\n"
        " for(unsigned long long q=first;q<sn;q+=step){\n"
        "   const unsigned int i0=(unsigned int)(q%ni),k=(unsigned int)(q/ni)+1u,x=i0+ng,ys=ng,yi=ng+1u,yn=nj+ng,yin=nj+ng-1u;\n"
        "   unsigned long long b=i3(k,ys,x,nyv,nxv);float t;\n"
        "   if(((obc>>8)&15u)==2u){t=a32(m32(m32(m32(-rs[(unsigned long long)k*(ni+2u*ng)+x],s32(v[i3(k,yi,x,nyv,nxv)],v[b])),rdy),vh[ng]),vten[b]);v[b]=a32(v[b],m32(dts,t));}\n"
        "   b=i3(k,yn,x,nyv,nxv);\n"
        "   if(((obc>>12)&15u)==2u){t=a32(m32(m32(m32(-rn[(unsigned long long)k*(ni+2u*ng)+x],s32(v[b],v[i3(k,yin,x,nyv,nxv)])),rdy),vh[nj+ng-1u]),vten[b]);v[b]=a32(v[b],m32(dts,t));}\n"
        " }\n"
        "}\n"
    )


CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE = _build_profile_core_source()
CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_BYTES = CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE.encode(
    "utf-8"
)
CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_BYTES
).hexdigest()
# The same solve with the six column-diagnostic stores removed.  Nothing else
# differs: the kernel names, their parameter lists and every live equation are
# the sealed source's own words.
CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE = _build_profile_core_source(
    retain_column_diagnostics=False
)
CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_BYTES = (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE.encode("utf-8")
)
CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_BYTES
).hexdigest()
CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE = _build_profile_boundary_source()
CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_BYTES = (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE.encode("utf-8")
)
CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_BYTES
).hexdigest()

CORE_KERNEL_NAMES = (
    "cm1_child_szt_prepare_v1",
    "cm1_child_szt_pgrad_v1",
    "cm1_child_szt_column_v1",
    "cm1_child_szt_average_v1",
    "cm1_child_szt_profile_probe_v1",
)
BOUNDARY_KERNEL_NAMES = (
    "cm1_child_szt_ssopen_snapshot_v1",
    "cm1_child_szt_ssopen_update_v1",
    "cm1_child_szt_bcwsfc_physical_v1",
)

CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE = _build_flat_profile_core_source()
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE.encode("utf-8")
).hexdigest()
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE = _build_flat_profile_core_source(
    retain_column_diagnostics=False
)
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE.encode("utf-8")
).hexdigest()
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE = _build_flat_profile_boundary_source()
CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE.encode("utf-8")
).hexdigest()
#: The flat boundary image carries only the ssopenbc update: a flat column
#: reconstructs no u/v at w points (bc.F:2410-2419) and calls no bcwsfc
#: (solve2.F:484, 1676; solve3.F:430 are terrain-only).
FLAT_BOUNDARY_KERNEL_NAMES = ("cm1_child_szt_ssopen_update_v1",)
FLAT_CORE_SOURCE_SHA256 = frozenset(
    {
        CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE_SHA256,
        CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256,
    }
)


@dataclass(frozen=True, slots=True)
class CompiledGm1ChildPsolver3OpenImage:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class CompiledGm1ChildPsolver3Open:
    core: CompiledGm1ChildPsolver3OpenImage
    boundary: CompiledGm1ChildPsolver3OpenImage
    #: Whether this image writes the tridiagonal coefficient/pivot/residual
    #: arrays.  A launch reads it to decide which array inventory to require,
    #: exactly as the scalar-dz stage does.
    retain_column_diagnostics: bool = True

    @property
    def target_arch(self) -> str:
        return self.core.target_arch


def _compile_image(
    source: str, names: tuple[str, ...], stem: str
) -> CompiledGm1ChildPsolver3OpenImage:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(source, f"{stem}_ptx.cu", method="ptx").compile(
        common + (f"--gpu-architecture=compute_{arch}",)
    )
    cubin, _ = compiler._NVRTCProgram(
        source, f"{stem}_cubin.cu", method="cubin"
    ).compile(common + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) for name in names):
        raise ValueError("compiled PTX lacks a stretched child entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled stretched child image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in names)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ChildPsolver3OpenImage(
        module,
        kernels,
        ptx,
        cubin,
        arch,
        identity,
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def compile_cuda_gm1_child_psolver3_open(
    *, retain_column_diagnostics: bool = True
) -> CompiledGm1ChildPsolver3Open:
    if type(retain_column_diagnostics) is not bool:
        raise TypeError("retain_column_diagnostics must be bool")
    core = _compile_image(
        CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE
        if retain_column_diagnostics
        else _build_profile_core_source(retain_column_diagnostics=False),
        CORE_KERNEL_NAMES,
        "gm1_child_psolver3_open_core_v1"
        if retain_column_diagnostics
        else "gm1_child_psolver3_open_core_compact_v1",
    )
    boundary = _compile_image(
        CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE,
        BOUNDARY_KERNEL_NAMES,
        "gm1_child_psolver3_open_boundary_v1",
    )
    if core.target_arch != boundary.target_arch:
        raise ValueError(
            "stretched child core and boundary images target different devices"
        )
    return CompiledGm1ChildPsolver3Open(core, boundary, retain_column_diagnostics)


def compile_cuda_gm1_flat_stretched_psolver3_open(
    *, retain_column_diagnostics: bool = True
) -> CompiledGm1ChildPsolver3Open:
    """The flat stretched column's sound.F images (terrain_flag=.false.)."""

    if type(retain_column_diagnostics) is not bool:
        raise TypeError("retain_column_diagnostics must be bool")
    core = _compile_image(
        CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE
        if retain_column_diagnostics
        else CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE,
        CORE_KERNEL_NAMES,
        "gm1_flat_stretched_psolver3_open_core_v1"
        if retain_column_diagnostics
        else "gm1_flat_stretched_psolver3_open_core_compact_v1",
    )
    boundary = _compile_image(
        CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE,
        FLAT_BOUNDARY_KERNEL_NAMES,
        "gm1_flat_stretched_psolver3_open_boundary_v1",
    )
    if core.target_arch != boundary.target_arch:
        raise ValueError(
            "flat stretched core and boundary images target different devices"
        )
    return CompiledGm1ChildPsolver3Open(core, boundary, retain_column_diagnostics)


def _require_vertical(
    cupy: Any,
    vertical: Any,
    *,
    nk: int,
    allow_uniform_idealized_validation: bool = False,
) -> tuple[Any, Any, Any, Any, float]:
    try:
        from cm1core.cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
    except ImportError as exc:  # pragma: no cover - packaging gate exercises this
        raise RuntimeError("stretched child metric binding is not installed") from exc
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    expected_shapes = {
        "sigma": (nk + 2,),
        "sigmaf": (nk + 3,),
        "rds": (nk + 2,),
        "rdsf": (nk + 3,),
    }
    values = []
    for name, shape in expected_shapes.items():
        value = getattr(vertical, name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
            or int(value.device.id) != int(cupy.cuda.Device().id)
        ):
            raise ValueError(f"vertical {name} must be device C-f32 {shape}")
        values.append(value)
    if len({int(value.data.ptr) for value in values}) != 4:
        raise ValueError("vertical profiles must not alias")
    identity = vertical.identity
    if not isinstance(identity, Mapping) or not identity.get("schema"):
        raise ValueError("vertical identity must be a schema-bearing mapping")
    if allow_uniform_idealized_validation:
        if (
            identity.get("fixture_class")
            != "source-audited-flat-idealized-uniform-200m"
        ):
            raise ValueError("uniform idealized vertical fixture class changed")
        if float(vertical.dz_reference_f32) != 200.0:
            raise ValueError("vertical reference dz must be exactly 200 m")
        if float(vertical.ztop_f32) != GM1_CHILD_ZTOP_F32:
            raise ValueError("vertical top must be exactly 15000 m")
    elif "grid_sha256" in identity:
        # The sealed 15 km child keeps every check it has ever had.  An
        # identity that names a generated grid is held to that grid's own
        # words, not to this one's.
        if identity.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256:
            raise ValueError("vertical grid SHA-256 differs from the sealed child")
        if identity.get("proof_sha256") != GM1_CHILD_STRETCHED_PROOF_SHA256:
            raise ValueError("vertical proof SHA-256 differs from the sealed child")
        if (
            dict(identity.get("profile_sha256", {}))
            != GM1_CHILD_STRETCHED_PROFILE_SHA256
        ):
            raise ValueError("vertical profile SHA-256 inventory differs")
        if float(vertical.ztop_f32) != GM1_CHILD_ZTOP_F32:
            raise ValueError("vertical top must be exactly 15000 m")
        if float(vertical.dz_reference_f32) != GM1_CHILD_DZ_REFERENCE_F32:
            raise ValueError("vertical reference dz must be exactly 100 m")
    else:
        # A coordinate this run generated.  param.F:8193 makes the model top
        # sigmaf(nk+1); binding anything else would put the terrain-coordinate
        # height factor (init_terrain.F:175-196) on a different column than the one
        # the profiles describe.
        ztop = float(vertical.ztop_f32)
        dz_reference = float(vertical.dz_reference_f32)
        if not math.isfinite(ztop) or ztop <= 0.0:
            raise ValueError("vertical top must be finite and greater than zero")
        if not math.isfinite(dz_reference) or dz_reference <= 0.0:
            raise ValueError(
                "vertical reference dz must be finite and greater than zero"
            )
        top_face = float(values[1][nk + 1].get())
        if top_face != ztop:
            raise ValueError(
                f"vertical top {ztop} m is not this coordinate's own sigmaf(nk+1) "
                f"of {top_face} m (param.F:8193)"
            )
        declared = identity.get("vertical_coordinate")
        if isinstance(declared, Mapping):
            if int(declared.get("nk", nk)) != nk:
                raise ValueError("vertical coordinate level count differs from nk")
            if float(declared.get("ztop_m", ztop)) != ztop:
                raise ValueError("vertical coordinate top differs from the bound top")
            if float(declared.get("dz_reference_m", dz_reference)) != dz_reference:
                raise ValueError(
                    "vertical coordinate dz reference differs from the bound one"
                )
    return (*values, float(vertical.ztop_f32))


def _require_padded_rho0(
    cupy: Any, metrics: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> Any:
    if "rho0_padded" not in metrics:
        raise ValueError(
            "metrics must expose source-valid padded rho0 as rho0_padded; "
            "physical rho0_3d cannot source outer-face averages"
        )
    value = metrics["rho0_padded"]
    shape = (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY)
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != cupy.float32
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"rho0_padded must be device C-f32 {shape}")
    if int(value.device.id) != int(cupy.cuda.Device().id):
        raise ValueError("rho0_padded is resident on another CUDA device")
    return value


@dataclass(frozen=True, slots=True)
class CudaGm1ChildPsolver3OpenLaunch:
    ready_event: Any
    compiled: CompiledGm1ChildPsolver3Open
    open_compiled: Any
    lateral_compiled: Any
    retained_arrays: tuple[Any, ...]
    cadence: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def launch_cuda_gm1_child_psolver3_open_stage(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    vertical: Any,
    selection: Gm1ChildPsolver3OpenSelection = DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION,
    ablations: TerrainOpenSoundAblations | None = None,
    stream: Any | None = None,
    compiled: CompiledGm1ChildPsolver3Open | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
    workspace: Mapping[str, Any] | None = None,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    nest_substep: Any | None = None,
    alph: float | None = None,
    kdiv: float | None = None,
    lateral_sides: tuple[int, int, int, int] = (2, 2, 2, 2),
) -> CudaGm1ChildPsolver3OpenLaunch:
    """Enqueue one profile-aware child acoustic RK stage.

    ``lateral_sides`` is CM1's ``(wbc, ebc, sbc, nbc)``.  The flat
    (``terrain_flag=.false.``) stage takes it as sound.F does: the pressure
    gradient runs on the faces ``param.F:9182-9219``'s ``dtu0``/``dtv0`` mask
    in, which includes a periodic side's boundary face, and bc.F's
    ``ssopenbc*`` update runs on open sides only (sound.F:274-302).  The
    terrain child stage is sealed to four open sides.

    There is deliberately no ``dz`` parameter.  Every vertical derivative and
    terrain-coordinate interpolation consumes the sealed device profiles.

    ``alph`` and ``kdiv`` are CM1's ``param3`` reals (README.namelist:751-759)
    and are bound exactly as the uniform operator binds them
    (:func:`cm1core.cuda_psolver3_terrain_open.launch_cuda_psolver3_terrain_open_three_stage`):
    sound.F:81-82 forms ``beta=1.0-alph`` and ``asq=alph*alph`` and
    sound.F:756,782 scale the divergence damper by ``kdiv`` on every grid,
    stretched or not.  ``None`` binds CM1's own 0.60 and 0.10, which are the
    words this stage carried as constants before the document's values
    reached it.
    """

    _validate_selection(selection)
    for name, value in (("alph", alph), ("kdiv", kdiv)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric or None")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name}={value} is outside CM1's admitted [0, 1]")
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    # The stencils, not a fixture, set the floor.  param.F:8985-9060 fits its
    # extrapolation to four scalar levels at each end, and the open lateral
    # boundary reads three columns in from each side.
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if nk < 4:
        raise ValueError(
            "child psolver requires nk >= 4; param.F:8985-9060 extrapolates "
            "from four scalar levels at each end of the column"
        )
    if ni < 3 or nj < 3:
        raise ValueError("child psolver requires ni,nj >= 3")
    if allow_uniform_idealized_validation and not allow_compact_validation:
        raise ValueError(
            "uniform idealized psolver admission requires explicit compact "
            "validation"
        )
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    dt, dx, dy = (_positive("dt", dt), _positive("dx", dx), _positive("dy", dy))
    if isinstance(nsound, bool) or not isinstance(nsound, int) or nsound < 1:
        raise ValueError("nsound must be a positive integer")
    if ablations is None:
        ablations = TerrainOpenSoundAblations()
    if not isinstance(ablations, TerrainOpenSoundAblations):
        raise TypeError("ablations must be TerrainOpenSoundAblations")
    from cm1core.cuda_momentum_advection5 import momentum_lateral_sides, momentum_open_boundary_code

    lateral_sides = momentum_lateral_sides(lateral_sides)
    obc = momentum_open_boundary_code(lateral_sides)

    cupy = importlib.import_module("cupy")
    from cm1core.cuda_lateral_closure import compile_cuda_lateral_closure
    from cm1core.cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary

    retain_diagnostics = True if compiled is None else bool(
        getattr(compiled, "retain_column_diagnostics", True)
    )
    a = _require_device_arrays(
        cupy, arrays, ni=ni, nj=nj, nk=nk,
        retain_column_diagnostics=retain_diagnostics,
    )
    m = _require_device_metrics(cupy, metrics, ni=ni, nj=nj, nk=nk)
    rho0_padded = _require_padded_rho0(cupy, metrics, ni=ni, nj=nj, nk=nk)
    sigma, sigmaf, rds, rdsf, ztop = _require_vertical(
        cupy,
        vertical,
        nk=nk,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    if any(int(rho0_padded.data.ptr) == int(value.data.ptr) for value in a.values()):
        raise ValueError("rho0_padded must not alias a mutable acoustic array")
    # terrain_flag=.false.: sound.F's flat statements, on the column's own
    # mh/mf; the terrain child keeps the terrain statements above.
    flat = bool(vertical.flat_column)
    if not flat and lateral_sides != (2, 2, 2, 2):
        raise ValueError(
            f"lateral_sides {lateral_sides}: the terrain child acoustic stage runs "
            "bc.F's ssopenbc on all four sides and its sealed pressure gradient "
            "skips every boundary face; only the flat stage takes a lateral family"
        )
    flat_column = None
    if flat:
        from cm1core.cuda_gm1_child_stretched_metrics import require_flat_column

        flat_column = require_flat_column(vertical, nk=nk, cupy=cupy)
    if compiled is None:
        compiled = (
            compile_cuda_gm1_flat_stretched_psolver3_open(
                retain_column_diagnostics=retain_diagnostics
            )
            if flat
            else compile_cuda_gm1_child_psolver3_open()
        )
    if (compiled.core.source_sha256 in FLAT_CORE_SOURCE_SHA256) != flat:
        raise ValueError(
            "the acoustic image and the vertical column disagree: a flat column "
            "(terrain_flag=.false.) runs sound.F:119-143,318-342,491-509,742-765 "
            "and a terrain column runs sound.F:147-178,346-392,512-535,569-641; "
            "compile the image for the column this binding describes"
        )
    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    lateral_compiled = lateral_compiled or compile_cuda_lateral_closure()
    if not (
        compiled.target_arch
        == open_compiled.target_arch
        == lateral_compiled.target_arch
        == vertical.target_arch
    ):
        raise ValueError("child acoustic CUDA images target different architectures")
    stream = stream or cupy.cuda.get_current_stream()
    f32, u32 = cupy.float32, cupy.uint32
    prepare, pgrad, column, average, _profile_probe = compiled.core.kernels
    if not flat:
        snapshot, ssopen, bcwsfc = compiled.boundary.kernels
    _, _, _, kpwe, kpsn, kfwe, kfsn, kwrigid, kpack, kunpack = open_compiled.kernels
    threads = (128, 1, 1)
    grid = (
        (
            max(
                max(value.size for value in a.values()),
                max(value.size for value in m.values()),
            )
            + 127
        )
        // 128,
        1,
        1,
    )
    edge_grid = ((max(ni * nk, nj * nk) + 127) // 128, 1, 1)
    col_grid = ((ni * nj + 127) // 128, 1, 1)
    face_grid = ((max(nk * nj * (ni - 1), nk * (nj - 1) * ni) + 127) // 128, 1, 1)
    # The radiation packing and terrain-edge snapshot buffers are exactly the
    # uniform stage's, so a caller running under a device-memory budget can
    # supply them from its own registry instead of having this call reach the
    # pool behind the budget's back.
    workspace_shapes = terrain_open_acoustic_workspace_shapes(ni, nj, nk)
    if workspace is None:
        scratch = {
            name: cupy.empty(shape, dtype=cupy.float32)
            for name, shape in workspace_shapes.items()
        }
    else:
        if set(workspace) != set(workspace_shapes):
            raise ValueError(
                "supplied acoustic workspace has the wrong named buffer inventory"
            )
        scratch = dict(workspace)
        for name, shape in workspace_shapes.items():
            value = scratch[name]
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.float32
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"supplied acoustic workspace {name} has the wrong dtype, shape or layout"
                )
            if int(value.device.id) != int(cupy.cuda.Device().id):
                raise ValueError(
                    "supplied acoustic workspace belongs to another CUDA device"
                )
        live = {int(value.data.ptr) for value in (*a.values(), *m.values())}
        pointers = [int(scratch[name].data.ptr) for name in workspace_shapes]
        if len(set(pointers)) != len(pointers) or live & set(pointers):
            raise ValueError(
                "supplied acoustic workspace aliases live state or another workspace"
            )
    pua, pu, pva, pv, pruf, prvf, prw, pre, prs, prn, us, vs = (
        scratch[name] for name in workspace_shapes
    )
    retained = (
        pua,
        pu,
        pva,
        pv,
        pruf,
        prvf,
        prw,
        pre,
        prs,
        prn,
        us,
        vs,
        rho0_padded,
        sigma,
        sigmaf,
        rds,
        rdsf,
    )
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts = f32(dts_value)
    alph = f32(CM1_ALPH_F32 if alph is None else alph)
    kdiv_f32 = f32(CM1_KDIV_F32 if kdiv is None else kdiv)
    beta = f32(f32(1.0) - alph)
    def _enqueue_terrain() -> Any:
        with stream:
            prepare(
                grid,
                threads,
                (
                    a["ua"],
                    a["va"],
                    a["wa"],
                    a["ppi"],
                    a["ppx"],
                    m["rr0"],
                    m["rf0"],
                    m["c1"],
                    m["c2"],
                    m["gz"],
                    a["piadv"],
                    a["ppterm"],
                    a["thv"],
                    a["u3d"],
                    a["v3d"],
                    a["w3d"],
                    a["pp3d"],
                    a["ppd"],
                    a["rru"],
                    a["rrv"],
                    a["rrw"],
                    a["qk"],
                    a["pk1"],
                    a["pk2"],
                    a["mm"],
                    a["fpk"],
                    a["divh"],
                    a["hadvp"],
                    a["dum3"],
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
                    u32(CM1_OPEN_NGXY),
                    u32(nrk),
                    dts,
                    rds,
                    rdsf,
                    f32(CM1_CP_F32),
                    beta,
                ),
                stream=stream,
            )
            for n in range(1, nloop + 1):
                # A live nest's child re-imposes its specified zone inside every
                # small step, as WRF does (dyn_em/solve_em.F:1262-1604).
                if nest_substep is not None:
                    nest_substep.begin(nrk=nrk, n=n, dts=float(dts))
                if ablations.open_consumption:
                    snapshot(
                        edge_grid,
                        threads,
                        (
                            a["u3d"],
                            a["v3d"],
                            us,
                            vs,
                            u32(ni),
                            u32(nj),
                            u32(nk),
                            u32(CM1_OPEN_NGXY),
                            sigma,
                            sigmaf,
                            rds,
                        ),
                        stream=stream,
                    )
                    ssopen(
                        edge_grid,
                        threads,
                        (
                            a["uten"],
                            a["vten"],
                            a["uh"],
                            a["vh"],
                            a["radbcw"],
                            a["radbce"],
                            a["radbcs"],
                            a["radbcn"],
                            m["gz"],
                            m["rgzu"],
                            m["rgzv"],
                            m["gx"],
                            m["gy"],
                            us,
                            vs,
                            a["u3d"],
                            a["v3d"],
                            u32(ni),
                            u32(nj),
                            u32(nk),
                            u32(CM1_OPEN_NGXY),
                            dts,
                            f32(1.0 / dx),
                            f32(1.0 / dy),
                            rdsf,
                        ),
                        stream=stream,
                    )
                if ablations.terrain_pressure_gradient:
                    pgrad(
                        face_grid,
                        threads,
                        (
                            a["ppd"],
                            a["thv"],
                            a["uten"],
                            a["vten"],
                            m["gxu"],
                            m["gyv"],
                            a["u3d"],
                            a["v3d"],
                            u32(ni),
                            u32(nj),
                            u32(nk),
                            u32(CM1_OPEN_NGXY),
                            dts,
                            f32(1.0 / dx),
                            f32(1.0 / dy),
                            rds,
                            f32(CM1_CP_F32),
                        ),
                        stream=stream,
                    )
                # dyn_em/solve_em.F:1275 advance_uv, then :1341,1351 spec_bdyupdate.
                if nest_substep is not None:
                    nest_substep.after_momentum(nrk=nrk, n=n, dts=float(dts))
                common = (
                    a["piadv"],
                    a["ppterm"],
                    a["ppten"],
                    a["wten"],
                    rho0_padded,
                    m["rr0"],
                    m["rrf0"],
                    m["rf0"],
                    m["c1"],
                    m["c2"],
                    m["gz"],
                    m["rgzu"],
                    m["rgzv"],
                    m["dzdx"],
                    m["dzdy"],
                    a["u3d"],
                    a["v3d"],
                    a["w3d"],
                    a["pp3d"],
                    a["ppd"],
                    a["ppx"],
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
                    u32(CM1_OPEN_NGXY),
                    u32(nrk),
                    u32(n),
                    u32(nloop),
                    dts,
                    f32(1.0 / dx),
                    f32(1.0 / dy),
                    sigma,
                    sigmaf,
                    rds,
                    rdsf,
                    f32(ztop),
                    alph,
                    beta,
                    kdiv_f32,
                    f32(CM1_SMEPS_F32),
                )
                column(
                    col_grid,
                    threads,
                    common
                    + (
                        u32(1),
                        u32(ablations.transformed_divergence),
                        u32(0),
                        u32(0),
                        u32(0),
                    ),
                    stream=stream,
                )
                bcwsfc(
                    grid,
                    threads,
                    (
                        a["u3d"],
                        a["v3d"],
                        m["gz"],
                        m["dzdx"],
                        m["dzdy"],
                        a["w3d"],
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        u32(CM1_OPEN_NGXY),
                    ),
                    stream=stream,
                )
                column(
                    col_grid,
                    threads,
                    common
                    + (
                        u32(0),
                        u32(0),
                        u32(ablations.pressure_advection),
                        u32(ablations.implicit_vertical_solve),
                        u32(1),
                    ),
                    stream=stream,
                )
                # dyn_em/solve_em.F:1457,1467 and :1582-1604, after the column step.
                if nest_substep is not None:
                    nest_substep.after_pressure(nrk=nrk, n=n, dts=float(dts))
                if n < nloop and ablations.between_step_pressure_fill:
                    kpwe(
                        grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream
                    )
                    kpsn(
                        grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream
                    )
                average(
                    grid,
                    threads,
                    (
                        a["u3d"],
                        a["v3d"],
                        a["w3d"],
                        a["rru"],
                        a["rrv"],
                        a["rrw"],
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        u32(CM1_OPEN_NGXY),
                        u32(n),
                        u32(nloop),
                        alph,
                    ),
                    stream=stream,
                )
            if ablations.radiation_history_production:
                kpack(
                    grid,
                    threads,
                    (
                        a["ua"],
                        a["u3d"],
                        a["va"],
                        a["v3d"],
                        a["ruf"],
                        a["rvf"],
                        pua,
                        pu,
                        pva,
                        pv,
                        pruf,
                        prvf,
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        u32(CM1_OPEN_NGXY),
                    ),
                    stream=stream,
                )
                lateral_compiled.kernels[3](
                    edge_grid,
                    (256, 1, 1),
                    (
                        pruf,
                        prvf,
                        pua,
                        pu,
                        pva,
                        pv,
                        prw,
                        pre,
                        prs,
                        prn,
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        f32(dx),
                        f32(dy),
                        f32(rk_stage_dt_f32(dt, nrk)),
                        f32(350.0),
                    ),
                    stream=stream,
                )
                kunpack(
                    edge_grid,
                    threads,
                    (
                        prw,
                        pre,
                        prs,
                        prn,
                        a["radbcw"],
                        a["radbce"],
                        a["radbcs"],
                        a["radbcn"],
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        u32(CM1_OPEN_NGXY),
                    ),
                    stream=stream,
                )
            if ablations.stage_family_fills:
                kfwe(
                    grid,
                    threads,
                    (a["u3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)),
                    stream=stream,
                )
                kfsn(
                    grid,
                    threads,
                    (a["u3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)),
                    stream=stream,
                )
                kfsn(
                    grid,
                    threads,
                    (a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)),
                    stream=stream,
                )
                kfwe(
                    grid,
                    threads,
                    (a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)),
                    stream=stream,
                )
                kfwe(
                    grid,
                    threads,
                    (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)),
                    stream=stream,
                )
                kfsn(
                    grid,
                    threads,
                    (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)),
                    stream=stream,
                )
                kwrigid(
                    grid,
                    threads,
                    (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)),
                    stream=stream,
                )
                bcwsfc(
                    grid,
                    threads,
                    (
                        a["u3d"],
                        a["v3d"],
                        m["gz"],
                        m["dzdx"],
                        m["dzdy"],
                        a["w3d"],
                        u32(ni),
                        u32(nj),
                        u32(nk),
                        u32(CM1_OPEN_NGXY),
                    ),
                    stream=stream,
                )
                if nrk < 3:
                    kpwe(
                        grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream
                    )
                    kpsn(
                        grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream
                    )
                    kfwe(
                        grid,
                        threads,
                        (
                            a["scalar"],
                            u32(ni),
                            u32(nj),
                            u32(nk),
                            u32(CM1_OPEN_NGXY),
                            u32(0),
                        ),
                        stream=stream,
                    )
                    kfsn(
                        grid,
                        threads,
                        (
                            a["scalar"],
                            u32(ni),
                            u32(nj),
                            u32(nk),
                            u32(CM1_OPEN_NGXY),
                            u32(0),
                        ),
                        stream=stream,
                    )
            event = cupy.cuda.Event(disable_timing=True)
            event.record(stream)
        return event

    if flat:
        event = _enqueue_flat_stage(
            cupy, a, m, rho0_padded, compiled, open_compiled, lateral_compiled,
            flat_column, stream=stream, grid=grid, edge_grid=edge_grid,
            col_grid=col_grid, ni=ni, nj=nj, nk=nk, nrk=nrk, nloop=nloop, dts=dts,
            dx=dx, dy=dy, dt=dt, alph=alph, beta=beta, kdiv=kdiv_f32,
            ablations=ablations, nest_substep=nest_substep,
            radiation=(pua, pu, pva, pv, pruf, prvf, prw, pre, prs, prn),
            obc=obc,
        )
    else:
        event = _enqueue_terrain()
    cadence = (
        {
            "nrk": nrk,
            "nloop": nloop,
            "dts_f32": float(dts),
            "open_consumptions": nloop if ablations.open_consumption else 0,
            "pressure_fills": nloop - 1 if ablations.between_step_pressure_fill else 0,
            "radiation_productions": 1 if ablations.radiation_history_production else 0,
        },
    )
    receipt = {
        "schema": CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
        "abi": CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_ABI if flat else CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI,
        "claim_limit": CUDA_GM1_CHILD_PSOLVER3_OPEN_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "dx_m": dx,
            "dy_m": dy,
            "ztop_m": ztop,
            "compact_validation": allow_compact_validation,
        },
        "cadence": list(cadence),
        "stretch_z": selection.stretch_z,
        "vertical": dict(vertical.identity),
        "static_combined_sha256": GM1_CHILD_STATIC_COMBINED_SHA256,
        "source_valid_padded_rho0": True,
        "scalar_dz_fallback": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "column_diagnostics_retained": compiled.retain_column_diagnostics,
        # README.namelist:751-759, recorded by value as the uniform stage
        # records them, so a reader never has to guess which off-centering
        # weight and divergence damper produced these words.
        "acoustic_reals": {"alph": float(alph), "kdiv": float(kdiv_f32)},
        "kernels": {
            # The core image's own source, so a reader can tell which of the
            # two sealed variants ran without re-deriving it from the flag.
            "core_source_sha256": compiled.core.source_sha256,
            "boundary_source_sha256": (
                CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
                if flat
                else CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
            ),
            "core_ptx_sha256": hashlib.sha256(compiled.core.ptx).hexdigest(),
            "core_cubin_sha256": hashlib.sha256(compiled.core.cubin).hexdigest(),
            "boundary_ptx_sha256": hashlib.sha256(compiled.boundary.ptx).hexdigest(),
            "boundary_cubin_sha256": hashlib.sha256(
                compiled.boundary.cubin
            ).hexdigest(),
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.core.runtime_identity),
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
    }
    if flat:
        # Published only for a flat column, so the terrain child's receipt is
        # exactly what it was.
        receipt["vertical_column"] = {
            "terrain_flag": False,
            "metric": "rdz*mh(1,1,k), rdz*mf(1,1,k) (param.F:6327, 8467, 8476)",
            "rdz_f32": float(flat_column[2]),
            "smeps_f32": float(f32(CM1_R21_SMEPS_F32)),
            "anchors": "sound.F:119-143,318-342,491-509,742-765; bc.F:2410-2419,2485-2494,2560-2569,2635-2644",
            # sound.F:274-302 and param.F:9182-9219: which sides took the
            # ssopenbc update and which boundary faces the pressure gradient.
            "lateral_boundaries": list(lateral_sides),
        }
    if nest_substep is not None:
        receipt["nest_specified_zone"] = nest_substep.record()
    return CudaGm1ChildPsolver3OpenLaunch(
        event,
        compiled,
        open_compiled,
        lateral_compiled,
        retained,
        cadence,
        receipt,
    )


def _enqueue_flat_stage(
    cupy: Any,
    a: Mapping[str, Any],
    m: Mapping[str, Any],
    rho0_padded: Any,
    compiled: CompiledGm1ChildPsolver3Open,
    open_compiled: Any,
    lateral_compiled: Any,
    flat_column: tuple[Any, Any, float],
    *,
    stream: Any,
    grid: tuple[int, int, int],
    edge_grid: tuple[int, int, int],
    col_grid: tuple[int, int, int],
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    nloop: int,
    dts: Any,
    dx: float,
    dy: float,
    dt: float,
    alph: Any,
    beta: Any,
    kdiv: Any,
    ablations: TerrainOpenSoundAblations,
    nest_substep: Any,
    radiation: tuple[Any, ...],
    obc: int,
) -> Any:
    """One flat-column acoustic RK stage (sound.F with terrain_flag=.false.).

    The substep order is the terrain stage's -- open-boundary tendency,
    pressure gradient, pressure/w column, pressure fill, averaging -- less the
    two terrain-only calls: no u/v reconstruction at w points before the
    open-boundary update, and no bcwsfc between the column's two passes.
    """

    f32, u32 = cupy.float32, cupy.uint32
    mh, mf, rdz_value = flat_column
    rdz = f32(rdz_value)
    rdx = f32(f32(1.0) / f32(dx))
    rdy = f32(f32(1.0) / f32(dy))
    smeps = f32(CM1_R21_SMEPS_F32)
    prepare, pgrad, column, average, _profile_probe = compiled.core.kernels
    (ssopen,) = compiled.boundary.kernels
    _, _, _, kpwe, kpsn, kfwe, kfsn, kwrigid, kpack, kunpack = open_compiled.kernels
    pua, pu, pva, pv, pruf, prvf, prw, pre, prs, prn = radiation
    threads = (128, 1, 1)
    ng = u32(CM1_OPEN_NGXY)
    dims = (u32(ni), u32(nj), u32(nk), ng)
    face_grid = ((nk * (nj + 1) * (ni + 1) + 127) // 128, 1, 1)
    with stream:
        prepare(
            grid,
            threads,
            (
                a["ua"], a["va"], a["wa"], a["ppi"], a["ppx"],
                m["rr0"], m["rf0"], m["c1"], m["c2"],
                a["piadv"], a["ppterm"], a["thv"], a["u3d"], a["v3d"], a["w3d"],
                a["pp3d"], a["ppd"], a["rru"], a["rrv"], a["rrw"],
                a["qk"], a["pk1"], a["pk2"], a["mm"], a["fpk"], a["divh"], a["hadvp"],
                a["dum3"], a["elim_c"], a["elim_d"], a["tri_aa"], a["tri_bb"],
                a["tri_cc"], a["tri_dd"], a["tri_pivot"], a["tri_residual"],
                *dims, u32(nrk), dts, mh, mf, rdz, f32(CM1_CP_F32), beta,
            ),
            stream=stream,
        )
        for n in range(1, nloop + 1):
            if nest_substep is not None:
                nest_substep.begin(nrk=nrk, n=n, dts=float(dts))
            if ablations.open_consumption:
                ssopen(
                    edge_grid,
                    threads,
                    (
                        a["uten"], a["vten"], a["uh"], a["vh"],
                        a["radbcw"], a["radbce"], a["radbcs"], a["radbcn"],
                        a["u3d"], a["v3d"], *dims, dts, rdx, rdy, u32(obc),
                    ),
                    stream=stream,
                )
            if ablations.terrain_pressure_gradient:
                pgrad(
                    face_grid,
                    threads,
                    (
                        a["ppd"], a["thv"], a["uten"], a["vten"], a["u3d"], a["v3d"],
                        *dims, dts, rdx, rdy, f32(CM1_CP_F32), u32(obc),
                    ),
                    stream=stream,
                )
            if nest_substep is not None:
                nest_substep.after_momentum(nrk=nrk, n=n, dts=float(dts))
            common = (
                a["piadv"], a["ppterm"], a["ppten"], a["wten"], a["u3d"], a["v3d"],
                a["w3d"], a["pp3d"], a["ppd"], a["ppx"], a["qk"], a["pk1"], a["pk2"],
                a["mm"], a["fpk"], a["divh"], a["hadvp"], a["elim_c"], a["elim_d"],
                a["tri_aa"], a["tri_bb"], a["tri_cc"], a["tri_dd"], a["tri_pivot"],
                a["tri_residual"], *dims, u32(nrk), u32(n), u32(nloop), dts, rdx, rdy,
                alph, beta, kdiv, smeps,
            )
            column(
                col_grid,
                threads,
                common + (u32(1), u32(ablations.transformed_divergence), u32(0), u32(0), u32(0)),
                stream=stream,
            )
            column(
                col_grid,
                threads,
                common + (u32(0), u32(0), u32(0), u32(ablations.implicit_vertical_solve), u32(1)),
                stream=stream,
            )
            if nest_substep is not None:
                nest_substep.after_pressure(nrk=nrk, n=n, dts=float(dts))
            if n < nloop and ablations.between_step_pressure_fill:
                kpwe(grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
                kpsn(grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
            average(
                grid,
                threads,
                (
                    a["u3d"], a["v3d"], a["w3d"], a["rru"], a["rrv"], a["rrw"],
                    *dims, u32(n), u32(nloop), alph,
                ),
                stream=stream,
            )
        if ablations.radiation_history_production:
            kpack(
                grid,
                threads,
                (
                    a["ua"], a["u3d"], a["va"], a["v3d"], a["ruf"], a["rvf"],
                    pua, pu, pva, pv, pruf, prvf, *dims,
                ),
                stream=stream,
            )
            lateral_compiled.kernels[3](
                edge_grid,
                (256, 1, 1),
                (
                    pruf, prvf, pua, pu, pva, pv, prw, pre, prs, prn,
                    u32(ni), u32(nj), u32(nk), f32(dx), f32(dy),
                    f32(rk_stage_dt_f32(dt, nrk)), f32(350.0),
                ),
                stream=stream,
            )
            kunpack(
                edge_grid,
                threads,
                (
                    prw, pre, prs, prn, a["radbcw"], a["radbce"], a["radbcs"], a["radbcn"],
                    *dims,
                ),
                stream=stream,
            )
        if ablations.stage_family_fills:
            # solve2.F:1662-1676 on a flat column: bcw with its rigid faces
            # and no bcwsfc.
            for kernel, target, kind in (
                (kfwe, "u3d", 1), (kfsn, "u3d", 1), (kfsn, "v3d", 2),
                (kfwe, "v3d", 2), (kfwe, "w3d", 3), (kfsn, "w3d", 3),
            ):
                kernel(grid, threads, (a[target], *dims, u32(kind)), stream=stream)
            kwrigid(grid, threads, (a["w3d"], *dims), stream=stream)
            if nrk < 3:
                kpwe(grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream)
                kpsn(grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream)
                kfwe(grid, threads, (a["scalar"], *dims, u32(0)), stream=stream)
                kfsn(grid, threads, (a["scalar"], *dims, u32(0)), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return event


def profile_operator_probe_f32(
    *,
    sigma: Any,
    sigmaf: Any,
    rds: Any,
    rdsf: Any,
    k: int,
    ztop: float = GM1_CHILD_ZTOP_F32,
    nk: int | None = None,
) -> dict[str, float]:
    """Independent compact arithmetic probe for native/oracle tests.

    ``nk`` defaults to the sealed 75-level child, so every existing call is
    unchanged; supply it to probe any other column.
    """

    np = importlib.import_module("numpy")
    if nk is None:
        nk = 75
    if isinstance(nk, bool) or not isinstance(nk, int) or nk < 4:
        raise ValueError("nk must be an integer >= 4")
    for name, value, shape in (
        ("sigma", sigma, (nk + 2,)),
        ("sigmaf", sigmaf, (nk + 3,)),
        ("rds", rds, (nk + 2,)),
        ("rdsf", rdsf, (nk + 3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != shape
        ):
            raise ValueError(f"{name} must be host C-f32 {shape}")
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= nk:
        raise ValueError(f"k must be 2..{nk}")
    if not math.isfinite(float(ztop)) or float(ztop) <= 0.0:
        raise ValueError("ztop must be finite and greater than zero")
    # param.F:8193.  The height factor at init_terrain.F:175-196 divides by the
    # model top, and the model top of these profiles is their own top face;
    # any other number would probe a column these words do not describe.
    top_face = float(sigmaf[nk + 1])
    if float(ztop) != top_face:
        raise ValueError(
            f"ztop {float(ztop)} m is not this column's own sigmaf(nk+1) of "
            f"{top_face} m (param.F:8193)"
        )

    def sub(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) - np.float32(right))

    def mul(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) * np.float32(right))

    r2 = mul(sub(sigmaf[k], sigma[k - 1]), rds[k])
    r1 = sub(np.float32(1.0), r2)
    height = mul(sub(sigmaf[k], np.float32(ztop)), np.float32(1.0) / np.float32(ztop))
    return {
        "r1": float(r1),
        "r2": float(r2),
        "height": float(height),
        "rds": float(rds[k]),
        "rdsf": float(rdsf[k]),
    }


__all__ = (
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI",
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256",
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256",
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_SCHEMA",
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256",
    "CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA",
    "CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_ABI",
    "CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256",
    "CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256",
    "CUDA_GM1_FLAT_STRETCHED_PSOLVER3_OPEN_SOURCE_SHA256",
    "CompiledGm1ChildPsolver3Open",
    "CudaGm1ChildPsolver3OpenLaunch",
    "DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION",
    "Gm1ChildPsolver3OpenSelection",
    "compile_cuda_gm1_child_psolver3_open",
    "compile_cuda_gm1_flat_stretched_psolver3_open",
    "launch_cuda_gm1_child_psolver3_open_stage",
    "profile_operator_probe_f32",
    "terrain_open_sound_shapes",
)
