"""CM1's eqtset=2 latent-heating convention for the Cartesian lane's column schemes.

The defect this module fixes
----------------------------
CM1 r21.1 finishes an ``eqtset=2`` microphysics call by rebuilding pressure,
Exner and theta at *fixed density* from the scheme's new temperature
(``src/mp_driver.F:603-617``)::

    prs  = rho*(rd+rv*qv)*T
    pp3d = (prs*rp00)**rovcp - pi0
    th3d = T/(pi0+pp3d) - th0

That rebuild is only right for a temperature that was heated at *constant
volume*.  CM1 never hands it anything else:

* its Morrison scheme (``ptype=5``) replaces every ``L/cpm`` by the
  constant-volume factors ``ecnd = (XXLV-RV*T)*cvm``, ``edep = (XXLS-RV*T)*cvm``
  and ``efrz = (XXLS-XXLV)*cvm`` under ``eqtset=2`` (``src/morrison.F:1640-1650``)
  and replaces WRF's one-step constant-pressure condensation ``PCC`` by the
  iterated constant-volume adjustment ``satadj2`` (``src/morrison.F:2441-2480``,
  ``:4012-4050``, ``:6402-6504``);
* its Thompson scheme (``ptype=3``) does the same with ``rcvm``
  (``src/thompson.F:1874-1877``, ``:2865-2920``, ``:3184-3200``);
* the constant-pressure schemes (``ptype=4,7,50-53,55``, WSM6 among them) are
  never run at ``eqtset=2``: ``src/param.F:936-948`` forces ``eqtset=1`` and
  ``src/mp_driver.F:1569-1573`` then holds the pressure instead.

GM1's Cartesian components are ArWen's WRF-lineage schemes, which heat at
constant pressure with ``L/cpm`` (ArWen ``morrison.cu:522-551,829-870``,
``wsm6.cu:625-652``, ``thompson.cu``'s ``ocp``).  The lane handed that
temperature straight to the sealed fixed-density rebuild
(``cuda_gm1_tilestream_post_wsm6.py``'s ``gm1_tiled_post_wsm6_eos_qa_f32_v1``),
so once the pressure excess had radiated away theta kept only
``cv/cp + Rv*T/L`` (about 0.78) of every phase change's latent heat -- the
condensation that drives an updraft and the evaporation and melting that
drive a cold pool alike.  On CM1's own supercell deck that made the storm
about 40 % weaker than CM1's (``evidence/oracle/supercell-fidelity``).

What this module does
---------------------
Two kernels bracket the column component, on the same stream, inside
:meth:`cm1core.cuda_cartesian_finish.CartesianFinish.microphysics`:

1. **before** the component, ``gm1_cm1_eqtset2_rcvm_snapshot_f32_v1`` keeps
   CM1's ``cvm`` of the column's entry state -- the word
   ``src/morrison.F:1641-1642`` and ``src/thompson.F:1875-1876`` form from the
   scheme's input mixing ratios;
2. **after** it, ``gm1_cm1_eqtset2_moist_heating_f32_v1`` turns the
   component's constant-pressure step into CM1's constant-volume one.  The
   component's own heating is ``cpm_scheme*(T - Tb) = sum(L_i*dq_i)``; CM1's
   ``eqtset=2`` heating of the same phase changes is
   ``sum((L_i - Rv*T)*dq_i)*cvm`` over the vapour exchanges and
   ``Lf*cvm`` over freezing and melting, which is::

       T = Tb + (cpm_scheme*(T - Tb) + Rv*Tb*(qv - qv_b)) * cvm

   For Morrison it then runs CM1's own ``satadj2`` (``src/morrison.F:
   6402-6504``, statement for statement) at fixed density on the converted
   state, keeping the result only where CM1 keeps it (``|dT| >= 1e-4``,
   ``src/morrison.F:2474-2480``, ``:4045-4051``).

The sealed rebuild then runs unchanged.  Its bytes, its snapshot and its
receipts are the sealed module's; only the temperature (``thp``) and, for
Morrison, ``q3d`` rows ``qv``/``qc`` it is handed are CM1's.

A cell the component left alone (``|T-Tb| < tsmall`` and ``|qv-qv_b| <
qsmall``, CM1's own gate, ``src/mp_driver.F:607-608``) is not converted, so
the sealed gate sees exactly the words it saw before; ``satadj2`` still
visits every cell, as CM1's column loop does.

What is and is not claimed
--------------------------
* Morrison and Thompson: the heating is CM1's eqtset=2 heating of the
  component's own phase changes.  The path differs from CM1's in-scheme one:
  the conversion acts on the whole step's increment (``cvm`` from the entry
  state, ``Rv*T`` at the entry temperature), and Morrison's ``satadj2`` runs
  after ArWen's sedimentation rather than before it.  Each is first order in
  a quantity that is small over one step.
* WSM6: CM1 has **no** eqtset=2 WSM6 (``src/param.F:936-948``).  GM1 applies
  CM1's own eqtset=2 treatment of the other WRF constant-pressure scheme it
  carries, Thompson: the scheme's process amounts, heated at constant volume.
  The receipt says ``cm1_configuration: false``.
* The component's latent heats and ``cpm`` are ArWen's (WRF's) words; the
  conversion uses them to recover ``sum(L*dq)`` and CM1's ``Rv``, ``cv``,
  ``cvv``, ``cpl`` and ``cpi`` for the rest.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

CUDA_CARTESIAN_MOIST_HEATING_SCHEMA = "gm1.cuda-cartesian-cm1-eqtset2-moist-heating/v1"
CUDA_CARTESIAN_MOIST_HEATING_ABI = "gm1.cuda-cartesian-cm1-eqtset2-moist-heating-f32/v1"
RCVM_SNAPSHOT_KERNEL = "gm1_cm1_eqtset2_rcvm_snapshot_f32_v1"
MOIST_HEATING_KERNEL = "gm1_cm1_eqtset2_moist_heating_f32_v1"
THREADS = 256

#: Conversion modes the heating kernel takes.
MODE_CONSTANT_VOLUME = 1
MODE_CONSTANT_VOLUME_SATADJ2 = 2

#: How each ArWen component forms its ``cpm`` (the divisor of its own heating).
CPM_MORRISON = 0  # morrison.cu:1038,1046  CP*(1+0.887*max(qv,0)), CP = 7*RD/2 = 1004.5
CPM_WSM6 = 1      # wsm6.cu:365-366        1004.5*(1-v)+1846.4*v, v = max(qv,1e-15)
CPM_THOMPSON = 2  # thompson.cu:230,249    1004*(1+0.887*max(1e-10,qv))

CM1_TSMALL_F32 = 1.0e-4
CM1_QSMALL_F32 = 1.0e-14


@dataclass(frozen=True, slots=True)
class MoistHeatingSelection:
    """How one component's constant-pressure step becomes CM1's eqtset=2 step."""

    component_id: str
    mode: int
    cpm_kind: int
    scheme_cpm: str
    cm1_reference: tuple[str, ...]
    cm1_configuration: bool
    note: str

    def record(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "mode": {
                MODE_CONSTANT_VOLUME: "constant-volume heating of the component's phase changes",
                MODE_CONSTANT_VOLUME_SATADJ2: (
                    "constant-volume heating of the component's phase changes, then "
                    "CM1 satadj2 at fixed density"
                ),
            }[self.mode],
            "scheme_cpm": self.scheme_cpm,
            "cm1_reference": list(self.cm1_reference),
            "cm1_configuration": self.cm1_configuration,
            "note": self.note,
        }


_MORRISON_REFERENCE = (
    "src/morrison.F:1640-1650 ecnd/edep/efrz = (L-RV*T)*cvm, (XXLS-XXLV)*cvm under eqtset=2",
    "src/morrison.F:2441-2480 and :4012-4051 satadj2 in place of WRF's PCC, kept when |dT|>=1e-4",
    "src/morrison.F:6402-6504 satadj2, eqtset=2 branch",
    "src/mp_driver.F:603-617 fixed-rho rebuild (the sealed finish)",
)
_THOMPSON_REFERENCE = (
    "src/thompson.F:1874-1877 rcvm from the entry state",
    "src/thompson.F:2865-2920, :3184-3200 edep/efrz/econd = (L-Rv*T)*rcvm, lfus*rcvm under eqtset=2",
    "src/mp_driver.F:603-617 fixed-rho rebuild (the sealed finish)",
)

MOIST_HEATING_SELECTIONS: Mapping[str, MoistHeatingSelection] = MappingProxyType({
    "morrison-mp10": MoistHeatingSelection(
        component_id="morrison-mp10",
        mode=MODE_CONSTANT_VOLUME_SATADJ2,
        cpm_kind=CPM_MORRISON,
        scheme_cpm="ArWen morrison.cu:1046 CP*(1+0.887*qv), CP=1004.5",
        cm1_reference=_MORRISON_REFERENCE,
        cm1_configuration=True,
        note=(
            "CM1 ptype=5 at eqtset=2.  satadj2 runs after ArWen's sedimentation "
            "rather than before it, on the converted state"
        ),
    ),
    "thompson-mp8": MoistHeatingSelection(
        component_id="thompson-mp8",
        mode=MODE_CONSTANT_VOLUME,
        cpm_kind=CPM_THOMPSON,
        scheme_cpm="ArWen thompson.cu ocp = 1/(1004*(1+0.887*max(1e-10,qv)))",
        cm1_reference=_THOMPSON_REFERENCE,
        cm1_configuration=True,
        note="CM1 ptype=3 at eqtset=2",
    ),
    "thompson-aerosol-mp28": MoistHeatingSelection(
        component_id="thompson-aerosol-mp28",
        mode=MODE_CONSTANT_VOLUME,
        cpm_kind=CPM_THOMPSON,
        scheme_cpm="ArWen thompson_aerosol_*.cu ocp = 1/(1004*(1+0.887*max(1e-10,qv)))",
        cm1_reference=_THOMPSON_REFERENCE,
        cm1_configuration=True,
        note=(
            "CM1 r21.1 numbers no aerosol-aware Thompson ptype; its thompson.F "
            "carries the aerosol-aware branch under the same eqtset=2 statements"
        ),
    ),
    "wsm6-mp6": MoistHeatingSelection(
        component_id="wsm6-mp6",
        mode=MODE_CONSTANT_VOLUME,
        cpm_kind=CPM_WSM6,
        scheme_cpm="ArWen wsm6.cu:366 1004.5*(1-qv)+1846.4*qv",
        cm1_reference=(
            "src/param.F:936-948: CM1 forces eqtset=1 for ptype=7, so it has no eqtset=2 WSM6",
            "GM1 applies CM1's eqtset=2 rule for its other WRF scheme, src/thompson.F:2865-2920, :3184-3200",
            "src/mp_driver.F:603-617 fixed-rho rebuild (the sealed finish)",
        ),
        cm1_configuration=False,
        note=(
            "not a CM1 configuration: CM1 runs WSM6 only at eqtset=1 with "
            "mp_driver.F:1569-1573's constant-pressure closure"
        ),
    ),
})


def moist_heating_selection(component_id: str) -> MoistHeatingSelection:
    """The conversion a component's step takes, or a refusal naming the gap."""

    selection = MOIST_HEATING_SELECTIONS.get(component_id)
    if selection is None:
        from .errors import refuse

        refuse(
            code="GM1-MOIST-HEATING-CONVENTION-UNKNOWN",
            gate="column-component",
            option="physics.component",
            value=component_id,
            reason=(
                f"{component_id!r} has no recorded heating convention, so the "
                "fixed-density rebuild of src/mp_driver.F:603-617 would be handed "
                "a temperature heated under an unknown convention; a constant-"
                "pressure scheme keeps only about 0.78 of its latent heat there"
            ),
            remedy="record the component's cpm and CM1 eqtset=2 reference here first",
        )
    return selection


CUDA_CARTESIAN_MOIST_HEATING_SOURCE = r'''// gm1.cuda-cartesian-cm1-eqtset2-moist-heating-f32/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
/* CM1 r21.1 default-REAL constants: src/morrison.F:416-431 (satadj2's module
   words) and src/constants.F:113-153 (thompson.F's), which agree. */
struct Cm1Eqtset2 { float r, rv, cv, cvv, cpl, cpi, ep_2, lv1, lv2; };
__device__ __forceinline__ Cm1Eqtset2 cm1_eqtset2(){
 Cm1Eqtset2 c;
 c.r=287.04f; c.rv=461.5f;
 c.cv=S(1005.7f,c.r);                 /* cv = CP - R */
 c.cvv=1408.5f; c.cpl=4190.0f; c.cpi=2106.0f;
 c.ep_2=D(c.r,c.rv);                  /* EP_2 = R/RV */
 c.lv2=S(c.cpl,1870.0f);              /* lv2 = cpl-cpv */
 c.lv1=A(2501000.0f,M(c.lv2,273.15f)); /* lv1 = xlv0+(cpl-cpv)*273.15 */
 return c;
}
/* src/morrison.F:1641-1642, src/thompson.F:1875-1876:
   1/( cv+cvv*qv+cpl*(qc+qr)+cpi*(qi+qs+qg) ) of the scheme's entry state. */
extern "C" __global__ void gm1_cm1_eqtset2_rcvm_snapshot_f32_v1(
 unsigned long long n,const float*q3d,float*rcvm_before){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 const Cm1Eqtset2 c=cm1_eqtset2();
 const float qv=q3d[p],qc=q3d[n+p],qr=q3d[2ull*n+p],qi=q3d[3ull*n+p],qs=q3d[4ull*n+p],qg=q3d[5ull*n+p];
 rcvm_before[p]=D(1.0f,A(A(A(c.cv,M(c.cvv,qv)),M(c.cpl,A(qc,qr))),M(c.cpi,A(A(qi,qs),qg))));
}
/* The component's own cpm, from the vapour it was handed. */
__device__ __forceinline__ float scheme_cpm(unsigned int kind,float qvb){
 if(kind==0u) return M(1004.5f,A(1.0f,M(0.887f,fmaxf(qvb,0.0f))));
 if(kind==1u){ const float vap=fmaxf(qvb,1.0e-15f); return A(M(1004.5f,S(1.0f,vap)),M(vap,1846.4f)); }
 return M(1004.0f,A(1.0f,M(0.887f,fmaxf(1.0e-10f,qvb))));
}
/* src/morrison.F:6402-6504, subroutine satadj2, the eqtset=2 branch, with
   the call site's PRES(K) = RHO(K)*DUMT*(R+RV*DUMQV) (:2453, :4024).
   Returns the number of iterations (0: not entered; 100: CM1 stops). */
__device__ __forceinline__ int cm1_satadj2_eqtset2(const Cm1Eqtset2&c,float rho,
 float*tnew,float*qvnew,float*qcnew,float qrnew,float qinew,float qsnew,float qgnew){
 float prs=M(M(rho,*tnew),A(c.r,M(c.rv,*qvnew)));
 const float converge=M(2.0f,1.0e-4f);
 float esl=fminf(M(prs,0.5f),M(611.2f,expf(D(M(17.67f,S(*tnew,273.15f)),S(*tnew,29.65f)))));
 float qvs=D(M(c.ep_2,esl),S(prs,esl));
 int n=0;
 if(*qcnew>1.0e-14f||*qvnew>qvs){
  const float ql=fmaxf(0.0f,A(*qcnew,qrnew));
  const float qi=fmaxf(0.0f,A(A(qinew,qsnew),qgnew));
  const float cvml=A(A(A(c.cv,M(c.cvv,*qvnew)),M(c.cpl,ql)),M(c.cpi,qi));
  const float lhv=S(c.lv1,M(c.lv2,*tnew));
  const float t1=D(S(lhv,M(c.rv,*tnew)),cvml);
  const float d1=M(M(t1,17.67f),243.5f);
  float tlast=*tnew;
  bool doit=true;
  while(doit){
   n=n+1;
   const float tm=S(*tnew,29.65f);
   float dqv=D(S(qvs,*qvnew),A(1.0f,D(M(d1,qvs),M(tm,tm))));
   dqv=fminf(dqv,*qcnew);
   if(A(*qvnew,dqv)<1.0e-20f) dqv=S(1.0e-20f,*qvnew);
   *qvnew=A(*qvnew,dqv);
   *qcnew=S(*qcnew,dqv);
   *tnew=S(*tnew,M(dqv,t1));
   prs=M(M(rho,A(c.r,M(c.rv,*qvnew))),*tnew);
   doit=false;
   if(fabsf(S(*tnew,tlast))>converge){
    tlast=*tnew;
    esl=fminf(M(prs,0.5f),M(611.2f,expf(D(M(17.67f,S(*tnew,273.15f)),S(*tnew,29.65f)))));
    qvs=D(M(c.ep_2,esl),S(prs,esl));
    doit=true;
   }
   if(n==100) doit=false;
  }
 }
 return n;
}
/* One cell of the satadj2 routine alone, for the native word check. */
extern "C" __global__ void gm1_cm1_satadj2_eqtset2_probe_f32_v1(
 unsigned long long n,const float*rho,float*t,float*qv,float*qc,const float*qr,
 const float*qi,const float*qs,const float*qg,int*iterations){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 const Cm1Eqtset2 c=cm1_eqtset2();
 float tn=t[p],qvn=qv[p],qcn=qc[p];
 iterations[p]=cm1_satadj2_eqtset2(c,rho[p],&tn,&qvn,&qcn,qr[p],qi[p],qs[p],qg[p]);
 t[p]=tn;qv[p]=qvn;qc[p]=qcn;
}
extern "C" __global__ void gm1_cm1_eqtset2_moist_heating_f32_v1(
 unsigned long long n,unsigned int mode,unsigned int cpm_kind,
 const float*rho,const float*th0,float*thp,const float*pii,float*q3d,
 const float*temperature_before,const float*qv_before,const float*rcvm_before,
 unsigned int*nonconvergent){
 unsigned long long p=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
 const Cm1Eqtset2 c=cm1_eqtset2();
 const float tb=temperature_before[p],qvb=qv_before[p];
 /* The component's temperature, read exactly as the sealed finish reads it. */
 float t=M(A(th0[p],thp[p]),pii[p]);
 float qv=q3d[p],qc=q3d[n+p];
 const float dqv=S(qv,qvb);
 const bool touched=fabsf(S(t,tb))>=1.0e-4f||fabsf(dqv)>=1.0e-14f;
 bool changed=false;
 if(touched){
  /* cpm*(T-Tb) is the component's sum(L*dq); CM1's eqtset=2 heating of the
     same phase changes is (sum(L*dq) + Rv*T*dqv)*cvm. */
  t=A(tb,M(A(M(scheme_cpm(cpm_kind,qvb),S(t,tb)),M(M(c.rv,tb),dqv)),rcvm_before[p]));
  changed=true;
 }
 if(mode==2u){
  /* src/morrison.F:1823: T3D(K).GE.TMELT is the warm branch, whose call
     (:2446) hands satadj2 DUMQI = 0.0; the cold branch (:4012) hands it qi. */
  const float qi=(tb>=273.15f)?0.0f:q3d[3ull*n+p];
  float ts=t,qvs=qv,qcs=qc;
  const int iterations=cm1_satadj2_eqtset2(c,rho[p],&ts,&qvs,&qcs,q3d[2ull*n+p],qi,q3d[4ull*n+p],q3d[5ull*n+p]);
  if(iterations>=100) atomicAdd(nonconvergent,1u);
  /* src/morrison.F:2474-2480, :4045-4051: kept only when |DUMT-SAVET|>=1e-4. */
  if(fabsf(S(ts,t))>=1.0e-4f){ t=ts; qv=qvs; qc=qcs; changed=true; }
 }
 if(changed){
  thp[p]=S(D(t,pii[p]),th0[p]);
  q3d[p]=qv; q3d[n+p]=qc;
 }
}
'''
CUDA_CARTESIAN_MOIST_HEATING_SOURCE_BYTES = CUDA_CARTESIAN_MOIST_HEATING_SOURCE.encode("utf-8")
CUDA_CARTESIAN_MOIST_HEATING_SOURCE_SHA256 = (
    "12cdf453a229ff58c13152ef66e72bff2bcfe5e2282ccc835bfcc5dd38dedf34"
)


def source_pin_is_valid() -> bool:
    return (
        hashlib.sha256(CUDA_CARTESIAN_MOIST_HEATING_SOURCE_BYTES).hexdigest()
        == CUDA_CARTESIAN_MOIST_HEATING_SOURCE_SHA256
    )


@dataclass(frozen=True, slots=True)
class CompiledCartesianMoistHeating:
    module_owner: Any
    snapshot_kernel: Any
    heating_kernel: Any
    probe_kernel: Any
    cubin_sha256: str
    options: tuple[str, ...]


def compile_cuda_cartesian_moist_heating() -> CompiledCartesianMoistHeating:
    if not source_pin_is_valid():
        actual = hashlib.sha256(CUDA_CARTESIAN_MOIST_HEATING_SOURCE_BYTES).hexdigest()
        raise RuntimeError(f"packaged moist-heating CUDA source differs from pin: {actual}")
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_CARTESIAN_MOIST_HEATING_SOURCE,
        filename="gm1_cartesian_moist_heating.cu",
        options=("--std=c++17", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"),
        name_expressions=(
            RCVM_SNAPSHOT_KERNEL, MOIST_HEATING_KERNEL, "gm1_cm1_satadj2_eqtset2_probe_f32_v1",
        ),
    )
    return CompiledCartesianMoistHeating(
        module,
        module.get_function(RCVM_SNAPSHOT_KERNEL),
        module.get_function(MOIST_HEATING_KERNEL),
        module.get_function("gm1_cm1_satadj2_eqtset2_probe_f32_v1"),
        hashlib.sha256(module.cubin).hexdigest(),
        tuple(module.options),
    )


def _grid(n: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((n + THREADS - 1) // THREADS, 1, 1), (THREADS, 1, 1)


def enqueue_rcvm_snapshot(
    compiled: CompiledCartesianMoistHeating, *, q3d: Any, rcvm_before: Any, stream: Any,
) -> None:
    """Keep CM1's ``cvm`` of the scheme's entry state, before the component runs."""

    cupy = importlib.import_module("cupy")
    n = int(rcvm_before.size)
    if int(q3d.shape[0]) < 6 or int(q3d[0].size) != n:
        raise ValueError("the entry-state cvm reads the six CM1 mass rows of this block")
    grid, block = _grid(n)
    compiled.snapshot_kernel(grid, block, (cupy.uint64(n), q3d, rcvm_before), stream=stream)


def enqueue_moist_heating(
    compiled: CompiledCartesianMoistHeating,
    selection: MoistHeatingSelection,
    *,
    values: Mapping[str, Any],
    aux: Mapping[str, Any],
    rcvm_before: Any,
    nonconvergent: Any,
    stream: Any,
) -> dict[str, Any]:
    """Convert the component's step to CM1's eqtset=2 heating, before the sealed rebuild.

    ``values`` is the block's post-physics inventory: ``thp`` holds the
    component's theta perturbation and ``pp3d`` its Exner (the staging of
    ``cuda_gm1_tilestream_column``), exactly what the sealed finish reads.
    ``aux`` is the sealed snapshot's ``temperature_before``/``qv_before``.
    """

    cupy = importlib.import_module("cupy")
    n = int(rcvm_before.size)
    for name in ("rho", "th0", "thp", "pp3d"):
        if int(values[name].size) != n:
            raise ValueError(f"moist heating {name} does not cover this block")
    grid, block = _grid(n)
    compiled.heating_kernel(
        grid, block,
        (
            cupy.uint64(n), cupy.uint32(selection.mode), cupy.uint32(selection.cpm_kind),
            values["rho"], values["th0"], values["thp"], values["pp3d"], values["q3d"],
            aux["temperature_before"], aux["qv_before"], rcvm_before, nonconvergent,
        ),
        stream=stream,
    )
    record = {
        "schema": CUDA_CARTESIAN_MOIST_HEATING_SCHEMA,
        "abi": CUDA_CARTESIAN_MOIST_HEATING_ABI,
        "kernel_source_sha256": CUDA_CARTESIAN_MOIST_HEATING_SOURCE_SHA256,
        "cubin_sha256": compiled.cubin_sha256,
        "selection": selection.record(),
        "words": n,
    }
    record["operation_token_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def refuse_nonconvergent_satadj2(words: Any) -> None:
    """``src/morrison.F:6488-6501``: CM1 stops when satadj2 has not converged."""

    count = int(words[0])
    if count:
        raise RuntimeError(
            f"satadj2 did not converge in 100 iterations in {count} cells "
            "(src/morrison.F:6488-6501 stops CM1 here: 'Convergence cannot be "
            "reached in satadj2 subroutine ... Try decreasing the timestep')"
        )


__all__ = [
    "CM1_QSMALL_F32",
    "CM1_TSMALL_F32",
    "CPM_MORRISON",
    "CPM_THOMPSON",
    "CPM_WSM6",
    "CUDA_CARTESIAN_MOIST_HEATING_ABI",
    "CUDA_CARTESIAN_MOIST_HEATING_SCHEMA",
    "CUDA_CARTESIAN_MOIST_HEATING_SOURCE",
    "CUDA_CARTESIAN_MOIST_HEATING_SOURCE_SHA256",
    "CompiledCartesianMoistHeating",
    "MODE_CONSTANT_VOLUME",
    "MODE_CONSTANT_VOLUME_SATADJ2",
    "MOIST_HEATING_KERNEL",
    "MOIST_HEATING_SELECTIONS",
    "MoistHeatingSelection",
    "RCVM_SNAPSHOT_KERNEL",
    "compile_cuda_cartesian_moist_heating",
    "enqueue_moist_heating",
    "enqueue_rcvm_snapshot",
    "moist_heating_selection",
    "refuse_nonconvergent_satadj2",
    "source_pin_is_valid",
]
