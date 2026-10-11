"""Source-ordered GM1 LES closure cadence on one CUDA device.

GM1 is the user-facing GPU model.  CM1 r21.1 is the pinned mathematical
authority.  This module owns the persistent LES/TKE state and composes the
admitted operators at their actual model cadences:

* ``sfc_and_turb`` refresh after initialization and each completed large step;
* held Rayleigh, SGS stress, scalar diffusion, dissipative heating, and TKE
  diffusion once in ``solve1``;
* TKE advection/RK immediately after each stage mass-flux/divergence rebuild;
* final TKE commit in ``solve3``.

Python performs allocation, validation, launch ordering, and receipt assembly.
All evolving atmospheric arithmetic remains on the CUDA device.  The one host
calculation is the static Rayleigh coefficient construction from the already
hash-validated initialization payload.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from cm1core.cuda_gm1_scalar_turbulence import (
    CompiledGm1ScalarTurbulenceKernels,
    Gm1ScalarTurbulenceWorkspace,
    compile_cuda_gm1_scalar_turbulence,
    create_cuda_gm1_scalar_turbulence_workspace,
    launch_cuda_gm1_scalar_turbulence,
)
from cm1core.cuda_moist_nm import (
    MASK_NAME,
    CompiledMoistNmKernel,
    compile_cuda_moist_nm,
    launch_cuda_moist_nm,
)
from cm1core.cuda_rayleigh_held import (
    CompiledRayleighHeldKernel,
    compile_cuda_rayleigh_held,
    cpu_build_rayleigh_coefficients,
    launch_cuda_rayleigh_held,
)
from cm1core.cuda_sgs_stress_tendency import (
    CompiledSgsStressKernels,
    compile_cuda_sgs_stress_tendency,
    launch_cuda_sgs_stress_tendency,
    sgs_stress_shapes,
)
from cm1core.cuda_sgs_tke import (
    MUTABLE_ARRAY_NAMES,
    CompiledSgsTkeKernels,
    compile_cuda_sgs_tke,
    launch_cuda_sgs_tke,
)
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_terrain_deformation import (
    CompiledTerrainDeformationKernels,
    compile_cuda_terrain_deformation,
    launch_cuda_terrain_deformation,
)
from cm1core.cuda_tke_evolution import (
    CompiledTkeEvolutionKernels,
    compile_cuda_tke_evolution,
    launch_cuda_tke_commit,
    launch_cuda_tke_diffusion,
    launch_cuda_tke_rk_stage,
)
from cm1core.eqtset2_energy_terms import Eqtset2IdissCarriers, Eqtset2IdissConsumer


GM1_CLOSURE_SCHEMA = "gm1.cuda-les-closure/v1"
GM1_CLOSURE_SOURCE_ORDER = (
    "post-step:calcnm",
    "post-step:calcdef",
    "post-step:tkekm-tkebc-gettau-state",
    "solve1:rayleigh-held",
    "solve1:sgs-momentum-stress",
    "solve1:theta-and-six-moisture-scalar-turbulence",
    "solve1:eqtset2-idiss",
    "solve1:tke-diffusion-held",
    "solve2:nrk1-tke-advection-rk",
    "solve2:nrk2-tke-advection-rk",
    "solve2:nrk3-tke-advection-rk",
    "solve3:tkea-commit",
)


def _positive_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _host_export_carrier(product: Any, name: str) -> Any:
    """Read one static carrier directly from the validated Rust payload."""

    np = importlib.import_module("numpy")
    spec = product.export.carrier(name)
    mapped = np.memmap(
        product.export.payload_path,
        dtype=np.dtype("<f4"),
        mode="r",
        offset=spec.offset_nbytes,
        shape=spec.shape,
        order="C",
    )
    result = np.array(mapped, dtype=np.float32, copy=True, order="C")
    del mapped
    return result


@dataclass(slots=True)
class CudaGm1ClosureState:
    product: Any
    prepared: Any
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    dt: float
    stream: Any
    rayleigh: dict[str, Any]
    moist: dict[str, Any]
    defv: Any
    defh: Any
    deformation_scratch: Mapping[str, Any]
    sgs: dict[str, Any]
    stress: dict[str, Any]
    tke: dict[str, Any]
    tke_workspace: Any
    scalar_workspace: Gm1ScalarTurbulenceWorkspace
    idiss: Eqtset2IdissConsumer
    rayleigh_compiled: CompiledRayleighHeldKernel
    moist_compiled: CompiledMoistNmKernel
    deformation_compiled: CompiledTerrainDeformationKernels
    sgs_compiled: CompiledSgsTkeKernels
    stress_compiled: CompiledSgsStressKernels
    scalar_compiled: CompiledGm1ScalarTurbulenceKernels
    tke_compiled: CompiledTkeEvolutionKernels
    ready_event: Any
    refresh_index: int = 0
    completed_stage: int = 0
    retained: list[Any] = field(default_factory=list)
    last_refresh_receipt: dict[str, Any] | None = None

    @property
    def target_arches(self) -> set[str]:
        return {
            self.rayleigh_compiled.target_arch,
            self.moist_compiled.target_arch,
            self.deformation_compiled.target_arch,
            self.sgs_compiled.target_arch,
            self.stress_compiled.target_arch,
            self.scalar_compiled.target_arch,
            self.tke_compiled.target_arch,
        }

    def checkpoint_arrays(self) -> dict[str, Any]:
        """Return the persistent closure carriers needed at a safe boundary."""

        return {"tkea": self.sgs["tkea"]}


def _refresh_closure(state: CudaGm1ClosureState) -> dict[str, Any]:
    cupy = importlib.import_module("cupy")
    p = state.product
    with state.stream:
        moist_launch = launch_cuda_moist_nm(
            {
                "c1": p.field("c1"),
                "c2": p.field("c2"),
                "mf": p.field("mf"),
                "pi0": p.field("pi0"),
                "th0": p.field("th0"),
                "prs": p.field("prs"),
                "pp3d": p.field("pp3d"),
                "th3d": p.field("th3d"),
                "q3d": p.field("q3d"),
                **state.moist,
            },
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dz=state.dz,
            stream=state.stream,
            compiled=state.moist_compiled,
        )
        deformation = launch_cuda_terrain_deformation(
            {
                **{
                    name: p.field(name)
                    for name in (
                        "u3d",
                        "v3d",
                        "w3d",
                        "rho",
                        "rf",
                        "c1",
                        "c2",
                        "gz",
                        "rgz",
                        "gzu",
                        "rgzu",
                        "gzv",
                        "rgzv",
                        "gx",
                        "gy",
                    )
                },
                "defv": state.defv,
                "defh": state.defh,
            },
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            stream=state.stream,
            compiled=state.deformation_compiled,
        )
        state.deformation_scratch = deformation.scratch
        state.sgs["dissten"].fill(cupy.float32(0.0))
        state.sgs["tketen"].fill(cupy.float32(0.0))
        sgs_launch = launch_cuda_sgs_tke(
            state.sgs,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            stream=state.stream,
            compiled=state.sgs_compiled,
        )
        cupy.copyto(state.tke["tke3d"], state.sgs["tkea"])
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    state.refresh_index += 1
    state.completed_stage = 0
    state.ready_event = event
    state.retained.extend((moist_launch, deformation, sgs_launch))
    receipt = {
        "schema": GM1_CLOSURE_SCHEMA,
        "phase": "post-step-sfc-and-turb-refresh",
        "refresh_index": state.refresh_index,
        "source_order": list(GM1_CLOSURE_SOURCE_ORDER[:3]),
        "operators": {
            "moist_nm": moist_launch.receipt,
            "terrain_deformation": deformation.receipt,
            "sgs_tke": sgs_launch.receipt,
        },
        "persistent_tkea": True,
        "synthetic_atmospheric_fields": False,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    state.last_refresh_receipt = receipt
    return receipt


def create_cuda_gm1_closure_state(
    product: Any,
    prepared: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    stream: Any | None = None,
) -> CudaGm1ClosureState:
    """Allocate persistent closure state and execute the startup refresh."""

    ni = _positive_int("ni", ni)
    nj = _positive_int("nj", nj)
    nk = _positive_int("nk", nk, 3)
    dx = _positive_float("dx", dx)
    dy = _positive_float("dy", dy)
    dz = _positive_float("dz", dz)
    dt = _positive_float("dt", dt)
    if tuple(product.export.crop_count_yx) != (nj, ni):
        raise ValueError("closure geometry differs from initialized product")
    if product.export.carrier("pi0").shape != (nk,):
        raise ValueError("closure vertical geometry differs from initialized product")
    cupy = importlib.import_module("cupy")
    stream = stream or product.stream
    face = (nk + 1, nj, ni)
    mass = (nk, nj, ni)

    rayleigh_compiled = compile_cuda_rayleigh_held()
    moist_compiled = compile_cuda_moist_nm()
    deformation_compiled = compile_cuda_terrain_deformation()
    sgs_compiled = compile_cuda_sgs_tke()
    stress_compiled = compile_cuda_sgs_stress_tendency()
    scalar_compiled = compile_cuda_gm1_scalar_turbulence()
    tke_compiled = compile_cuda_tke_evolution()

    np = importlib.import_module("numpy")
    zh = _host_export_carrier(product, "zh")
    zf = _host_export_carrier(product, "zf")
    coefficients = cpu_build_rayleigh_coefficients(
        np.ascontiguousarray(np.pad(zh, ((0, 0), (1, 1), (1, 1)), mode="edge")),
        np.ascontiguousarray(np.pad(zf, ((0, 0), (1, 1), (1, 1)), mode="edge")),
    )
    del zh, zf
    with stream:
        rayleigh = {
            "tauh": cupy.asarray(coefficients.tauh),
            "taus": cupy.asarray(coefficients.taus),
            "tauf": cupy.asarray(coefficients.tauf),
            "u0": product.field("u0"),
            "v0": product.field("v0"),
            "ua": product.field("ua"),
            "va": product.field("va"),
            "wa": product.field("wa")[:nk],
            "tha": product.field("tha"),
            "uten1": cupy.empty((nk, nj, ni + 1), dtype=cupy.float32),
            "vten1": cupy.empty((nk, nj + 1, ni), dtype=cupy.float32),
            "wten1": cupy.empty(mass, dtype=cupy.float32),
            "thten1": cupy.empty(mass, dtype=cupy.float32),
        }
        moist = {
            "nm": cupy.empty(face, dtype=cupy.float32),
            "temperature": cupy.empty(mass, dtype=cupy.float32),
            "qt": cupy.empty(mass, dtype=cupy.float32),
            "thv": cupy.empty(mass, dtype=cupy.float32),
            "qvci": cupy.empty(mass, dtype=cupy.float32),
            MASK_NAME: cupy.empty(mass, dtype=cupy.uint8),
        }
        defv = cupy.empty(face, dtype=cupy.float32)
        defh = cupy.empty(face, dtype=cupy.float32)
        ng = CM1_OPEN_NGXY
        sgs = {
            "ruh": prepared.arrays["uh"][ng : ng + ni],
            "rvh": prepared.arrays["vh"][ng : ng + nj],
            "rmf": product.field("rmf"),
            "nm": moist["nm"],
            "defv": defv,
            "defh": defh,
            **{
                name: cupy.zeros(face, dtype=cupy.float32)
                for name in MUTABLE_ARRAY_NAMES
            },
        }
        stress_shapes = sgs_stress_shapes(ni, nj, nk)
        stress = {
            name: cupy.zeros(stress_shapes[name], dtype=cupy.float32)
            for name in ("t11", "t22", "t33", "t12", "t13", "t23", "uten", "vten", "wten")
        }
        tke = {
            "tkea": sgs["tkea"],
            "tke3d": cupy.zeros(face, dtype=cupy.float32),
            "tketen": sgs["tketen"],
            "kmh": sgs["kmh"],
            "kmv": sgs["kmv"],
            **{
                name: product.field(name)
                for name in (
                    "rho",
                    "rf",
                    "mh",
                    "mf",
                    "gz",
                    "rgz",
                    "rgzu",
                    "rgzv",
                    "c1",
                    "c2",
                    "rrf0",
                )
            },
            "rru": prepared.arrays["rru"],
            "rrv": prepared.arrays["rrv"],
            "rrw": prepared.arrays["rrw"],
            "divx": cupy.zeros(mass, dtype=cupy.float32),
        }
        tke_workspace = cupy.empty(face, dtype=cupy.float32)
        scalar_workspace = create_cuda_gm1_scalar_turbulence_workspace(
            ni=ni, nj=nj, nk=nk
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state = CudaGm1ClosureState(
        product=product,
        prepared=prepared,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        stream=stream,
        rayleigh=rayleigh,
        moist=moist,
        defv=defv,
        defh=defh,
        deformation_scratch={},
        sgs=sgs,
        stress=stress,
        tke=tke,
        tke_workspace=tke_workspace,
        scalar_workspace=scalar_workspace,
        idiss=Eqtset2IdissConsumer(nx=ni, ny=nj, nz=nk),
        rayleigh_compiled=rayleigh_compiled,
        moist_compiled=moist_compiled,
        deformation_compiled=deformation_compiled,
        sgs_compiled=sgs_compiled,
        stress_compiled=stress_compiled,
        scalar_compiled=scalar_compiled,
        tke_compiled=tke_compiled,
        ready_event=event,
    )
    _refresh_closure(state)
    if len(state.target_arches) != 1:
        raise ValueError("GM1 closure CUDA images target different architectures")
    return state


def launch_cuda_gm1_closure_solve1(
    state: CudaGm1ClosureState, carriers: Any
) -> dict[str, Any]:
    """Compose all admitted held closure tendencies exactly once."""

    if state.completed_stage != 0:
        raise ValueError("GM1 closure solve1 requires a fresh large-step cadence")
    # The forecast coordinator synchronizes the prior completed-step event before
    # entering solve1.  Drop last-step launch/event owners here so the allocator
    # can reuse deformation and operator scratch instead of growing per step.
    state.ready_event.synchronize()
    state.retained.clear()
    cupy = importlib.import_module("cupy")
    p = state.product
    ng = CM1_OPEN_NGXY
    with state.stream:
        rayleigh_launch = launch_cuda_rayleigh_held(
            state.rayleigh,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            stream=state.stream,
            compiled=state.rayleigh_compiled,
        )
        for name in ("uten", "vten", "wten"):
            state.stress[name].fill(cupy.float32(0.0))
        stress_launch = launch_cuda_sgs_stress_tendency(
            {
                **{
                    name: state.deformation_scratch[name]
                    for name in ("s11", "s22", "s33", "s12", "s13", "s23")
                },
                "kmh": state.sgs["kmh"],
                "kmv": state.sgs["kmv"],
                **{
                    name: p.field(name)
                    for name in (
                        "rho",
                        "rf",
                        "mh",
                        "mf",
                        "gz",
                        "rgz",
                        "gzu",
                        "rgzu",
                        "gzv",
                        "rgzv",
                        "gxu",
                        "gyv",
                        "u3d",
                        "v3d",
                        "w3d",
                    )
                },
                **state.stress,
            },
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            dt=state.dt,
            stream=state.stream,
            compiled=state.stress_compiled,
        )

        uview = carriers.momentum["uten"][
            1 : state.nk + 1,
            ng : ng + state.nj,
            ng : ng + state.ni + 1,
        ]
        vview = carriers.momentum["vten"][
            1 : state.nk + 1,
            ng : ng + state.nj + 1,
            ng : ng + state.ni,
        ]
        wview = carriers.momentum["wten"][
            1 : state.nk + 2,
            ng : ng + state.nj,
            ng : ng + state.ni,
        ]
        cupy.add(state.rayleigh["uten1"], state.stress["uten"], out=uview)
        cupy.add(state.rayleigh["vten1"], state.stress["vten"], out=vview)
        cupy.copyto(wview, state.stress["wten"])
        cupy.add(wview[: state.nk], state.rayleigh["wten1"], out=wview[: state.nk])
        cupy.copyto(carriers.theta, state.rayleigh["thten1"])

        scalar_launch = launch_cuda_gm1_scalar_turbulence(
            {
                "th0": p.field("th0"),
                "tha": p.field("tha"),
                "qa": p.field("qa"),
                "rho": p.field("rho"),
                "rr": p.field("rr"),
                "rf": p.field("rf"),
                "mh": p.field("mh"),
                "mf": p.field("mf"),
                "khh": state.sgs["khh"],
                "khv": state.sgs["khv"],
                "gz": p.field("gz"),
                "rgz": p.field("rgz"),
                "rgzu": p.field("rgzu"),
                "rgzv": p.field("rgzv"),
                "gx": p.field("gx"),
                "gy": p.field("gy"),
                "gxu": p.field("gxu"),
                "gyv": p.field("gyv"),
                "thten1": carriers.theta,
                "qten": carriers.moisture,
            },
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            dt=state.dt,
            stream=state.stream,
            compiled=state.scalar_compiled,
            workspace=state.scalar_workspace,
        )
        q = p.field("qa")
        idiss_launch = state.idiss.apply(
            Eqtset2IdissCarriers(
                thten1=carriers.theta,
                dissten=state.sgs["dissten"],
                pi0=p.field("pi0"),
                ppi=p.field("ppi"),
                qv=q[0],
                qc=q[1],
                qr=q[2],
                qi=q[3],
                qs=q[4],
                qg=q[5],
            )
        )
        diffusion = launch_cuda_tke_diffusion(
            state.tke,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            dx=state.dx,
            dy=state.dy,
            dz=state.dz,
            dt=state.dt,
            stream=state.stream,
            compiled=state.tke_compiled,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    state.ready_event = event
    state.retained.extend(
        (rayleigh_launch, stress_launch, scalar_launch, idiss_launch, diffusion)
    )
    receipt = {
        "schema": GM1_CLOSURE_SCHEMA,
        "phase": "solve1-held-closure",
        "source_order": list(GM1_CLOSURE_SOURCE_ORDER[3:8]),
        "operators": {
            "rayleigh": rayleigh_launch.receipt,
            "sgs_stress": dict(stress_launch.receipt),
            "scalar_turbulence": scalar_launch.receipt,
            "idiss": dict(idiss_launch.receipt),
            "tke_diffusion": diffusion.receipt,
        },
        "device_only_evolving_math": True,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return receipt


def launch_cuda_gm1_closure_stage(
    state: CudaGm1ClosureState, *, nrk: int, divx: Any
) -> dict[str, Any]:
    if nrk != state.completed_stage + 1 or nrk not in (1, 2, 3):
        raise ValueError(
            f"GM1 closure TKE cadence requires stage {state.completed_stage + 1}"
        )
    state.tke["divx"] = divx
    launch = launch_cuda_tke_rk_stage(
        state.tke,
        workspace=state.tke_workspace,
        ni=state.ni,
        nj=state.nj,
        nk=state.nk,
        nrk=nrk,
        dx=state.dx,
        dy=state.dy,
        dz=state.dz,
        dt=state.dt,
        stream=state.stream,
        compiled=state.tke_compiled,
    )
    state.completed_stage = nrk
    state.ready_event = launch.ready_event
    state.retained.append(launch)
    return dict(launch.receipt)


def finish_cuda_gm1_closure_step(state: CudaGm1ClosureState) -> dict[str, Any]:
    """Commit stage-three TKE, then refresh closure fields for the next step."""

    if state.completed_stage != 3:
        raise ValueError("GM1 closure commit requires completed RK stages 1, 2, and 3")
    commit = launch_cuda_tke_commit(
        state.tke,
        ni=state.ni,
        nj=state.nj,
        nk=state.nk,
        completed_nrk=3,
        stream=state.stream,
        compiled=state.tke_compiled,
    )
    state.retained.append(commit)
    return {
        "schema": GM1_CLOSURE_SCHEMA,
        "phase": "solve3-tkea-commit",
        "source_order": [GM1_CLOSURE_SOURCE_ORDER[-1]],
        "operator": dict(commit.receipt),
    }


def refresh_cuda_gm1_closure_for_next_step(
    state: CudaGm1ClosureState,
) -> dict[str, Any]:
    if state.completed_stage != 3:
        raise ValueError("GM1 closure refresh requires a committed completed step")
    return _refresh_closure(state)


__all__ = [
    "CudaGm1ClosureState",
    "GM1_CLOSURE_SCHEMA",
    "GM1_CLOSURE_SOURCE_ORDER",
    "create_cuda_gm1_closure_state",
    "finish_cuda_gm1_closure_step",
    "launch_cuda_gm1_closure_solve1",
    "launch_cuda_gm1_closure_stage",
    "refresh_cuda_gm1_closure_for_next_step",
]
