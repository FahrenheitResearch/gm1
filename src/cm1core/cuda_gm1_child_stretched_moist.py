"""Profile-aware post-acoustic ``getdiv``/``getdivx`` for the GM1 child.

The legacy bridge is exact only for uniform sigma.  These three replacement
kernels consume the already admitted CM1 ``stretch_z=1`` profiles; the
horizontal mass-flux kernel is reused because it has no vertical-coordinate
dependence.  Thermodynamic terms and buoyancy remain the independently sealed
base-3D kernels in :mod:`cm1core.cuda_moist_stage_bridge`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from cm1core.cuda_moist_stage_bridge import (
    CUDA_MOIST_STAGE_BRIDGE_THREADS,
    CudaMoistStageBridgeState,
    _spacing,
    _validate_common,
)
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY


GM1_CHILD_STRETCHED_MOIST_SCHEMA = "gm1.cuda-child-stretched-moist-stage/v1"
GM1_CHILD_STRETCHED_MOIST_ABI = "gm1.cuda-child-stretched-moist-stage-f32/v1"
GM1_CHILD_STRETCHED_MOIST_KERNEL_NAMES = (
    "gm1_child_stretch_getdiv_v1",
    "gm1_child_stretch_getdivx_horizontal_v1",
    "gm1_child_stretch_getdivx_vertical_v1",
    "gm1_child_stretch_getdivx_divergence_v1",
)
GM1_CHILD_STRETCHED_MOIST_SOURCE = r'''// gm1.cuda-child-stretched-moist-stage-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long I3(unsigned k,unsigned j,unsigned i,unsigned ny,unsigned nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ unsigned long long MI(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*nj+j)*ni+i;}
__device__ __forceinline__ float XU(const float*u,const float*rgzu,unsigned k,unsigned j,unsigned f,unsigned ni,unsigned nj,unsigned ng){return M(u[I3(k,j+ng,f+ng,nj+2u*ng,ni+1u+2u*ng)],rgzu[(unsigned long long)j*(ni+1u)+f]);}
__device__ __forceinline__ float YV(const float*v,const float*rgzv,unsigned k,unsigned f,unsigned i,unsigned ni,unsigned nj,unsigned ng){return M(v[I3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng)],rgzv[(unsigned long long)f*ni+i]);}
__device__ __forceinline__ float WT(const float*u,const float*v,const float*w,const float*rgzu,const float*rgzv,const float*dzdx,const float*dzdy,const float*gz,const float*sigma,const float*sigmaf,const float*rds,unsigned kf,unsigned j,unsigned i,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float ztop){
 if(kf==1u||kf==nk+1u)return 0.0f;float r2=M(S(sigmaf[kf],sigma[kf-1u]),rds[kf]),r1=M(0.5f,S(1.0f,r2));r2=M(0.5f,r2);
 float ux=A(M(r2,A(XU(u,rgzu,kf,j,i,ni,nj,ng),XU(u,rgzu,kf,j,i+1u,ni,nj,ng))),M(r1,A(XU(u,rgzu,kf-1u,j,i,ni,nj,ng),XU(u,rgzu,kf-1u,j,i+1u,ni,nj,ng))));
 float vy=A(M(r2,A(YV(v,rgzv,kf,j,i,ni,nj,ng),YV(v,rgzv,kf,j+1u,i,ni,nj,ng))),M(r1,A(YV(v,rgzv,kf-1u,j,i,ni,nj,ng),YV(v,rgzv,kf-1u,j+1u,i,ni,nj,ng))));
 float terrain=A(M(ux,dzdx[(unsigned long long)j*ni+i]),M(vy,dzdy[(unsigned long long)j*ni+i]));terrain=M(M(M(terrain,S(sigmaf[kf],ztop)),gz[(unsigned long long)j*ni+i]),D(1.0f,ztop));return A(w[I3(kf,j+ng,i+ng,nj+2u*ng,ni+2u*ng)],terrain);
}
extern "C" __global__ void gm1_child_stretch_getdiv_v1(const float*u,const float*v,const float*w,const float*rgzu,const float*rgzv,const float*dzdx,const float*dzdy,const float*gz,const float*sigma,const float*sigmaf,const float*rds,const float*rdsf,const float*thterm,float*div,float*theta_div,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float dx,float dy,float ztop,float smeps){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=D(1.0f,dx),rdy=D(1.0f,dy);
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),k=(unsigned)(q/((unsigned long long)ni*nj))+1u;float d=A(M(S(XU(u,rgzu,k,j,i+1u,ni,nj,ng),XU(u,rgzu,k,j,i,ni,nj,ng)),rdx),M(S(YV(v,rgzv,k,j+1u,i,ni,nj,ng),YV(v,rgzv,k,j,i,ni,nj,ng)),rdy));d=A(d,M(S(WT(u,v,w,rgzu,rgzv,dzdx,dzdy,gz,sigma,sigmaf,rds,k+1u,j,i,ni,nj,nk,ng,ztop),WT(u,v,w,rgzu,rgzv,dzdx,dzdy,gz,sigma,sigmaf,rds,k,j,i,ni,nj,nk,ng,ztop)),rdsf[k]));if(fabsf(d)<smeps)d=0.0f;div[q]=d;theta_div[q]=M(d,thterm[q]);}
}
extern "C" __global__ void gm1_child_stretch_getdivx_horizontal_v1(const float*rho0,const float*rgzu,const float*rgzv,float*rru,float*rrv,unsigned ni,unsigned nj,unsigned nk,unsigned ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,total=nu+nv;unsigned nxrho=ni+2u*ng,nyrho=nj+2u*ng;
 for(;q<total;q+=step){if(q<nu){unsigned f=(unsigned)(q%(ni+1u)),j=(unsigned)((q/(ni+1u))%nj),k=(unsigned)(q/((unsigned long long)(ni+1u)*nj))+1u;float den=M(0.5f,A(rho0[I3(k,j+ng,f+ng-1u,nyrho,nxrho)],rho0[I3(k,j+ng,f+ng,nyrho,nxrho)]));unsigned p=I3(k,j+ng,f+ng,nj+2u*ng,ni+1u+2u*ng);rru[p]=M(M(den,rru[p]),rgzu[(unsigned long long)j*(ni+1u)+f]);}else{unsigned long long z=q-nu;unsigned i=(unsigned)(z%ni),f=(unsigned)((z/ni)%(nj+1u)),k=(unsigned)(z/((unsigned long long)ni*(nj+1u)))+1u;float den=M(0.5f,A(rho0[I3(k,f+ng-1u,i+ng,nyrho,nxrho)],rho0[I3(k,f+ng,i+ng,nyrho,nxrho)]));unsigned p=I3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng);rrv[p]=M(M(den,rrv[p]),rgzv[(unsigned long long)f*ni+i]);}}
}
extern "C" __global__ void gm1_child_stretch_getdivx_vertical_v1(const float*rf0,const float*dzdx,const float*dzdy,const float*gz,const float*sigma,const float*sigmaf,const float*rds,const float*rru,const float*rrv,float*rrw,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float ztop){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;float rzt=D(1.0f,ztop);
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),kf=(unsigned)(q/((unsigned long long)ni*nj))+1u,p=I3(kf,j+ng,i+ng,nj+2u*ng,ni+2u*ng);if(kf==1u||kf==nk+1u){rrw[p]=0.0f;}else{float r2=M(S(sigmaf[kf],sigma[kf-1u]),rds[kf]),r1=M(0.5f,S(1.0f,r2));r2=M(0.5f,r2);unsigned nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;float ux=A(M(r2,A(rru[I3(kf,j+ng,i+ng,nyu,nxu)],rru[I3(kf,j+ng,i+1u+ng,nyu,nxu)])),M(r1,A(rru[I3(kf-1u,j+ng,i+ng,nyu,nxu)],rru[I3(kf-1u,j+ng,i+1u+ng,nyu,nxu)])));float vy=A(M(r2,A(rrv[I3(kf,j+ng,i+ng,nyv,nxv)],rrv[I3(kf,j+1u+ng,i+ng,nyv,nxv)])),M(r1,A(rrv[I3(kf-1u,j+ng,i+ng,nyv,nxv)],rrv[I3(kf-1u,j+1u+ng,i+ng,nyv,nxv)])));float terrain=A(M(ux,dzdx[(unsigned long long)j*ni+i]),M(vy,dzdy[(unsigned long long)j*ni+i]));terrain=M(M(M(terrain,S(sigmaf[kf],ztop)),gz[(unsigned long long)j*ni+i]),rzt);rrw[p]=A(M(rf0[MI(kf-1u,j,i,ni,nj)],rrw[p]),terrain);}}
}
extern "C" __global__ void gm1_child_stretch_getdivx_divergence_v1(const float*rru,const float*rrv,const float*rrw,const float*rdsf,float*divx,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float dx,float dy,float smeps){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=D(1.0f,dx),rdy=D(1.0f,dy);
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),k=(unsigned)(q/((unsigned long long)ni*nj))+1u;unsigned nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;float d=A(M(S(rru[I3(k,j+ng,i+1u+ng,nyu,nxu)],rru[I3(k,j+ng,i+ng,nyu,nxu)]),rdx),M(S(rrv[I3(k,j+1u+ng,i+ng,nyv,nxv)],rrv[I3(k,j+ng,i+ng,nyv,nxv)]),rdy));d=A(d,M(S(rrw[I3(k+1u,j+ng,i+ng,nyw,nxw)],rrw[I3(k,j+ng,i+ng,nyw,nxw)]),rdsf[k]));divx[q]=(fabsf(d)<smeps)?0.0f:d;}
}
'''
GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256 = (
    "9e899ae56709ec406b836bba0bcac8941dcab3b69ef690d555a4f65ba2d3bbdf"
)

#: The same four operators on a flat stretched column (terrain_flag=.false.).
#: misclibs.F:127-138 forms getdiv as ((du*rdx)*uh + (dv*rdy)*vh) +
#: ((dw*rdz)*mh(1,1,k)) straight from the small-step averages, with no
#: terrain-following w and no rdsf; misclibs.F:241-270 scales rru/rrv by
#: rho0(1,1,k) and rrw by rf0(1,1,k) and adds nothing; misclibs.F:324-336 is
#: getdiv's flat divergence again.  uh = vh = 1 on this lane's uniform
#: horizontal grid, so those two factors are the identity and are not
#: bound.  mh is the (nk+2,) CM1-indexed column, rdz = 1.0/dz of the metric
#: reference (param.F:6327), and smeps is cm1.F:311's 1.0e-30.
GM1_FLAT_STRETCHED_MOIST_ABI = "gm1.cuda-flat-stretched-moist-stage-f32/v1"
GM1_FLAT_STRETCHED_MOIST_SOURCE = r'''// gm1.cuda-flat-stretched-moist-stage-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long I3(unsigned k,unsigned j,unsigned i,unsigned ny,unsigned nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_child_stretch_getdiv_v1(const float*u,const float*v,const float*w,const float*mh,const float*thterm,float*div,float*theta_div,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float dx,float dy,float rdz,float smeps){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=D(1.0f,dx),rdy=D(1.0f,dy);
 unsigned nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),k=(unsigned)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;float d=A(M(S(u[I3(k,y,x+1u,nyu,nxu)],u[I3(k,y,x,nyu,nxu)]),rdx),M(S(v[I3(k,y+1u,x,nyv,nxv)],v[I3(k,y,x,nyv,nxv)]),rdy));d=A(d,M(M(S(w[I3(k+1u,y,x,nyw,nxw)],w[I3(k,y,x,nyw,nxw)]),rdz),mh[k]));if(fabsf(d)<smeps)d=0.0f;div[q]=d;theta_div[q]=M(d,thterm[q]);}
}
extern "C" __global__ void gm1_child_stretch_getdivx_horizontal_v1(const float*rho0,float*rru,float*rrv,unsigned ni,unsigned nj,unsigned nk,unsigned ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,total=nu+nv;unsigned nxrho=ni+2u*ng,nyrho=nj+2u*ng;
 for(;q<total;q+=step){if(q<nu){unsigned f=(unsigned)(q%(ni+1u)),j=(unsigned)((q/(ni+1u))%nj),k=(unsigned)(q/((unsigned long long)(ni+1u)*nj))+1u;unsigned long long p=I3(k,j+ng,f+ng,nj+2u*ng,ni+1u+2u*ng);rru[p]=M(rru[p],rho0[I3(k,ng,ng,nyrho,nxrho)]);}else{unsigned long long z=q-nu;unsigned i=(unsigned)(z%ni),f=(unsigned)((z/ni)%(nj+1u)),k=(unsigned)(z/((unsigned long long)ni*(nj+1u)))+1u;unsigned long long p=I3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng);rrv[p]=M(rrv[p],rho0[I3(k,ng,ng,nyrho,nxrho)]);}}
}
extern "C" __global__ void gm1_child_stretch_getdivx_vertical_v1(const float*rf0,float*rrw,unsigned ni,unsigned nj,unsigned nk,unsigned ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni;
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),kf=(unsigned)(q/((unsigned long long)ni*nj))+1u;unsigned long long p=I3(kf,j+ng,i+ng,nj+2u*ng,ni+2u*ng);if(kf==1u||kf==nk+1u){rrw[p]=0.0f;}else{rrw[p]=M(rrw[p],rf0[(unsigned long long)(kf-1u)*nj*ni]);}}
}
extern "C" __global__ void gm1_child_stretch_getdivx_divergence_v1(const float*rru,const float*rrv,const float*rrw,const float*mh,float*divx,unsigned ni,unsigned nj,unsigned nk,unsigned ng,float dx,float dy,float rdz,float smeps){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;float rdx=D(1.0f,dx),rdy=D(1.0f,dy);
 unsigned nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),k=(unsigned)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;float d=A(M(S(rru[I3(k,y,x+1u,nyu,nxu)],rru[I3(k,y,x,nyu,nxu)]),rdx),M(S(rrv[I3(k,y+1u,x,nyv,nxv)],rrv[I3(k,y,x,nyv,nxv)]),rdy));d=A(d,M(M(S(rrw[I3(k+1u,y,x,nyw,nxw)],rrw[I3(k,y,x,nyw,nxw)]),rdz),mh[k]));divx[q]=(fabsf(d)<smeps)?0.0f:d;}
}
extern "C" __global__ void gm1_flat_stretch_buoyancy_w_v1(const float*dum8,const float*c1,const float*c2,float*wten,unsigned ni,unsigned nj,unsigned nk,unsigned ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk-1u)*nj*ni;unsigned nx=ni+2u*ng,ny=nj+2u*ng;
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),kf=(unsigned)(q/((unsigned long long)ni*nj))+2u;unsigned long long p=((unsigned long long)(kf-1u)*nj+j)*ni+i;float add=A(M(c1[p],dum8[I3(kf-1u,j+ng,i+ng,ny,nx)]),M(c2[p],dum8[I3(kf,j+ng,i+ng,ny,nx)]));unsigned long long d=I3(kf,j+ng,i+ng,ny,nx);wten[d]=A(wten[d],add);}
}
'''
GM1_FLAT_STRETCHED_MOIST_KERNEL_NAMES = (
    *GM1_CHILD_STRETCHED_MOIST_KERNEL_NAMES,
    "gm1_flat_stretch_buoyancy_w_v1",
)
GM1_FLAT_STRETCHED_MOIST_SOURCE_SHA256 = hashlib.sha256(
    GM1_FLAT_STRETCHED_MOIST_SOURCE.encode()
).hexdigest()
#: cm1.F:311: CM1's single-precision smeps.  The terrain child keeps the
#: 1.0e-20 its sealed kernels were measured with.
CM1_R21_SMEPS_F32 = 1.0e-30


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
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
class CompiledGm1ChildStretchedMoist:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CompiledGm1FlatStretchedMoist(CompiledGm1ChildStretchedMoist):
    """The flat stretched column's getdiv/getdivx image (mh and rdz bound)."""


def compile_cuda_gm1_child_stretched_moist() -> CompiledGm1ChildStretchedMoist:
    actual = hashlib.sha256(GM1_CHILD_STRETCHED_MOIST_SOURCE.encode()).hexdigest()
    if actual != GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256:
        raise RuntimeError("packaged stretched-moist CUDA source differs from its pin")
    return _compile_moist_image(
        GM1_CHILD_STRETCHED_MOIST_SOURCE, "gm1_child_stretched_moist",
        CompiledGm1ChildStretchedMoist, GM1_CHILD_STRETCHED_MOIST_KERNEL_NAMES,
    )


def compile_cuda_gm1_flat_stretched_moist() -> CompiledGm1FlatStretchedMoist:
    """misclibs.F's flat getdiv/getdivx, compiled."""

    actual = hashlib.sha256(GM1_FLAT_STRETCHED_MOIST_SOURCE.encode()).hexdigest()
    if actual != GM1_FLAT_STRETCHED_MOIST_SOURCE_SHA256:
        raise RuntimeError("packaged flat stretched-moist CUDA source differs from its pin")
    return _compile_moist_image(
        GM1_FLAT_STRETCHED_MOIST_SOURCE, "gm1_flat_stretched_moist",
        CompiledGm1FlatStretchedMoist, GM1_FLAT_STRETCHED_MOIST_KERNEL_NAMES,
    )


def _compile_moist_image(
    source: str, stem: str, result_type: type, names: tuple[str, ...]
) -> Any:
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
        source,
        f"{stem}_ptx.cu",
        method="ptx",
    ).compile(options + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        source,
        f"{stem}_cubin.cu",
        method="cubin",
    ).compile(options + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in names
    ):
        raise ValueError("compiled PTX lacks stretched-moist entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled stretched-moist image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in names)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return result_type(module, kernels, ptx, cubin, arch, identity)


def launch_cuda_gm1_flat_stretched_buoyancy(
    state: CudaMoistStageBridgeState,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    stream: Any | None = None,
    compiled: CompiledGm1FlatStretchedMoist,
) -> Any:
    """solve2.F:897-906 on a flat column: w's buoyancy, and nothing else.

    The terrain branch (solve2.F:971-1008) adds the horizontal pressure-
    gradient correction ``(...)*(gxu(k)+gxu(k+1))`` to uten/vten; CM1 never
    runs it with terrain_flag=.false., so no word of uten or vten is written
    here -- not even the ``+-0`` that term becomes on a flat column.
    """

    cupy, a, m = _validate_common(state, arrays, metrics)
    if not state.terms_ready or state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError("buoyancy requires one fresh current-stage term production")
    if not isinstance(compiled, CompiledGm1FlatStretchedMoist):
        raise TypeError("the flat buoyancy runs from the flat stretched moist image")
    stream = state.stream if stream is None else stream
    words = (state.nk - 1) * state.nj * state.ni
    block = (CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    grid = ((words + CUDA_MOIST_STAGE_BRIDGE_THREADS - 1) // CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    u32 = cupy.uint32
    with stream:
        compiled.kernels[4](
            grid, block,
            (state.dum8, m["c1"], m["c2"], a["wten"], u32(state.ni), u32(state.nj),
             u32(state.nk), u32(CM1_OPEN_NGXY)),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.buoyancy_applied = True
    return event


@dataclass(frozen=True, slots=True)
class CudaGm1ChildStretchedPostAcousticLaunch:
    ready_event: Any
    receipt: Mapping[str, Any]
    compiled: CompiledGm1ChildStretchedMoist


def launch_cuda_gm1_child_stretched_post_acoustic(
    state: CudaMoistStageBridgeState,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    vertical: Gm1ChildStretchedMetrics,
    *,
    dx: float,
    dy: float,
    stream: Any | None = None,
    compiled: CompiledGm1ChildStretchedMoist | None = None,
) -> CudaGm1ChildStretchedPostAcousticLaunch:
    """Run post-sound ``getdiv`` then destructive stretched ``getdivx``."""

    cupy, a, m = _validate_common(state, arrays, metrics)
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    if not state.buoyancy_applied or state.post_acoustic_ready:
        raise ValueError("stretched post-acoustic bridge requires fresh buoyancy")
    dx, dy = _spacing("dx", dx), _spacing("dy", dy)
    stream = state.stream if stream is None else stream
    if int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("stretched moist stage must remain on its bound stream")
    flat = bool(vertical.flat_column)
    if compiled is None:
        compiled = (
            compile_cuda_gm1_flat_stretched_moist()
            if flat
            else compile_cuda_gm1_child_stretched_moist()
        )
    if isinstance(compiled, CompiledGm1FlatStretchedMoist) != flat:
        raise ValueError(
            "the getdiv image and the vertical column disagree: a flat column "
            "(terrain_flag=.false.) runs misclibs.F:127-138,241-270,324-336, a "
            "terrain column misclibs.F:156-208,275-311,356-367"
        )
    if compiled.target_arch != vertical.target_arch:
        raise ValueError("stretched moist and vertical CUDA images target different architectures")
    rho0_padded = metrics.get("rho0_padded")
    expected_rho0_shape = (
        state.nk + 2,
        state.nj + 2 * CM1_OPEN_NGXY,
        state.ni + 2 * CM1_OPEN_NGXY,
    )
    if (
        not isinstance(rho0_padded, cupy.ndarray)
        or rho0_padded.dtype != cupy.float32
        or rho0_padded.shape != expected_rho0_shape
        or not rho0_padded.flags.c_contiguous
    ):
        raise ValueError(
            f"rho0_padded must be admitted C-f32 CUDA {expected_rho0_shape}"
        )
    n = state.ni * state.nj * state.nk
    horizontal = state.nk * state.nj * (state.ni + 1) + state.nk * (state.nj + 1) * state.ni
    vertical_words = (state.nk + 1) * state.nj * state.ni
    block = (CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
    grids = tuple(
        ((count + CUDA_MOIST_STAGE_BRIDGE_THREADS - 1) // CUDA_MOIST_STAGE_BRIDGE_THREADS, 1, 1)
        for count in (n, horizontal, vertical_words)
    )
    f32, u32 = cupy.float32, cupy.uint32
    getdiv, horizontal_kernel, vertical_kernel, divergence = compiled.kernels[:4]
    if flat:
        from cm1core.cuda_gm1_child_stretched_metrics import require_flat_column

        mh_column, _mf_column, rdz = require_flat_column(vertical, nk=state.nk, cupy=cupy)
        with stream:
            common = (u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY))
            getdiv(
                grids[0], block,
                (a["rru"], a["rrv"], a["rrw"], mh_column, state.thterm,
                 state.divergence, state.theta_divergence_term, *common,
                 f32(dx), f32(dy), f32(rdz), f32(CM1_R21_SMEPS_F32)),
                stream=stream,
            )
            horizontal_kernel(
                grids[1], block, (rho0_padded, a["rru"], a["rrv"], *common), stream=stream,
            )
            vertical_kernel(
                grids[2], block, (m["rf0"], a["rrw"], *common), stream=stream,
            )
            divergence(
                grids[0], block,
                (a["rru"], a["rrv"], a["rrw"], mh_column, state.divx, *common,
                 f32(dx), f32(dy), f32(rdz), f32(CM1_R21_SMEPS_F32)),
                stream=stream,
            )
            event = cupy.cuda.Event(disable_timing=True)
            event.record(stream)
    else:
        with stream:
            getdiv(
                grids[0],
                block,
                (
                    a["rru"], a["rrv"], a["rrw"], m["rgzu"], m["rgzv"],
                    m["dzdx"], m["dzdy"], m["gz"], vertical.sigma,
                    vertical.sigmaf, vertical.rds, vertical.rdsf, state.thterm,
                    state.divergence, state.theta_divergence_term, u32(state.ni),
                    u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY), f32(dx),
                    f32(dy), f32(vertical.ztop_f32), f32(1.0e-20),
                ),
                stream=stream,
            )
            horizontal_kernel(
                grids[1],
                block,
                (
                    rho0_padded, m["rgzu"], m["rgzv"], a["rru"], a["rrv"],
                    u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY),
                ),
                stream=stream,
            )
            vertical_kernel(
                grids[2],
                block,
                (
                    m["rf0"], m["dzdx"], m["dzdy"], m["gz"], vertical.sigma,
                    vertical.sigmaf, vertical.rds, a["rru"], a["rrv"], a["rrw"],
                    u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY),
                    f32(vertical.ztop_f32),
                ),
                stream=stream,
            )
            divergence(
                grids[0],
                block,
                (
                    a["rru"], a["rrv"], a["rrw"], vertical.rdsf, state.divx,
                    u32(state.ni), u32(state.nj), u32(state.nk), u32(CM1_OPEN_NGXY),
                    f32(dx), f32(dy), f32(1.0e-20),
                ),
                stream=stream,
            )
            event = cupy.cuda.Event(disable_timing=True)
            event.record(stream)
    state.post_acoustic_ready = True
    receipt = MappingProxyType(
        {
            "schema": GM1_CHILD_STRETCHED_MOIST_SCHEMA,
            "abi": GM1_FLAT_STRETCHED_MOIST_ABI if flat else GM1_CHILD_STRETCHED_MOIST_ABI,
            "source_order": [
                "misclibs.F:108-214 getdiv",
                "misclibs.F:222-370 getdivx horizontal/vertical/divergence",
            ],
            "vertical_metrics": dict(vertical.identity),
            "kernel": {
                "source_sha256": (
                    GM1_FLAT_STRETCHED_MOIST_SOURCE_SHA256
                    if flat
                    else GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256
                ),
                "target_arch": compiled.target_arch,
            },
            "density_face_policy": "source-valid padded rho0; no clamp or broadcast",
            "uniform_sigma_kernel_used": False,
            "ready_event_recorded": True,
        }
    )
    json.dumps(dict(receipt), allow_nan=False, sort_keys=True)
    return CudaGm1ChildStretchedPostAcousticLaunch(event, receipt, compiled)


__all__ = (
    "CM1_R21_SMEPS_F32",
    "CompiledGm1FlatStretchedMoist",
    "GM1_FLAT_STRETCHED_MOIST_ABI",
    "GM1_FLAT_STRETCHED_MOIST_SOURCE",
    "GM1_FLAT_STRETCHED_MOIST_SOURCE_SHA256",
    "compile_cuda_gm1_flat_stretched_moist",
    "launch_cuda_gm1_flat_stretched_buoyancy",
    "GM1_FLAT_STRETCHED_MOIST_KERNEL_NAMES",
    "CompiledGm1ChildStretchedMoist",
    "CudaGm1ChildStretchedPostAcousticLaunch",
    "GM1_CHILD_STRETCHED_MOIST_ABI",
    "GM1_CHILD_STRETCHED_MOIST_KERNEL_NAMES",
    "GM1_CHILD_STRETCHED_MOIST_SCHEMA",
    "GM1_CHILD_STRETCHED_MOIST_SOURCE",
    "GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256",
    "compile_cuda_gm1_child_stretched_moist",
    "launch_cuda_gm1_child_stretched_post_acoustic",
)
