"""Fail-closed flat-periodic CM1 r21.1 ``psolver=3`` operator.

This module seals the existing source-ordered ``sound.F`` CUDA kernels behind
the exact stock-case pressure/grid envelope for which they are valid.  In
particular, CM1's flat-grid branch reads the base-state arrays at ``(1,1,k)``;
the executable contract therefore proves that those arrays are horizontally
homogeneous before an asynchronous stage launch may be used.

The one-stage API is the live cadence seam.  A caller may rebuild held
large-step tendencies between RK stages and launch stages 1, 2, and 3 in
source order.  The three-stage wrapper is intentionally labelled held-input.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from .cuda_sound3_periodic import (
    ALL_ARRAY_NAMES,
    CM1_SOUND3_PERIODIC_ALPH_F32,
    CM1_SOUND3_PERIODIC_CP_F32,
    CM1_SOUND3_PERIODIC_KDIV_F32,
    CM1_SOUND3_PERIODIC_SMEPS_F32,
    COMPARE_OUTPUT_NAMES,
    CUDA_SOUND3_PERIODIC_ABI,
    CUDA_SOUND3_PERIODIC_CLAIM_LIMIT,
    CUDA_SOUND3_PERIODIC_KERNEL_NAMES,
    CUDA_SOUND3_PERIODIC_SOURCE_SHA256,
    CUDA_SOUND3_PERIODIC_THREADS,
    CompiledSound3PeriodicKernels,
    CpuSound3PeriodicResult,
    CpuSound3PeriodicStageDiagnostics,
    _require_device_arrays,
    array_shapes,
    compare_sound3_periodic_raw,
    compile_cuda_sound3_periodic,
    cpu_sound3_periodic_stage_inplace,
    default_periodic_face_weights,
    fill_periodic_scalar_inplace,
    fill_periodic_uv_inplace,
    scalar_shape_zyx,
    u_shape_zyx,
    v_shape_zyx,
    validate_periodic_contract,
    w_shape_zyx,
)
from .errors import refuse


CUDA_PSOLVER3_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-psolver3-periodic/v1"
CUDA_PSOLVER3_PERIODIC_ABI = "gpuwm-cm1.cuda-psolver3-periodic-stage-dispatch/v1"
BASE_PROFILE_NAMES = ("mh", "rr0", "rf0", "c1", "c2", "mf")
SOURCE_ANCHORS = {
    "dispatch": "src/solve2.F:1119-1138",
    "schedule": "src/sound.F:81-113",
    "flat_base_profile": "src/sound.F:117-142",
    "stage_prepare": "src/sound.F:191-248",
    "flat_cartesian_uv": "src/sound.F:317-360",
    "flat_divergence_and_exner_advection": "src/sound.F:484-511",
    "implicit_tridiagonal_column": "src/sound.F:659-730",
    "pressure_filter": "src/sound.F:739-803",
    "acoustic_averages": "src/sound.F:806-834",
    "final_stage_ppx": "src/sound.F:842-853",
    "stage_boundaries": "src/solve2.F:1665-1691",
    "periodic_boundaries": "src/bc.F:372-1014",
}
CM1_PSOLVER3_PERIODIC_CLAIM_LIMIT = (
    "CM1 r21.1 sound.F vertically implicit split-compressible psolver=3 "
    "acoustic/RK operator for the exact flat Cartesian, single-process, "
    "four-periodic, eqtset=2, apmasscon=1 stock pressure profile; supplied "
    "nsound, held thv/piadv/ppterm/ppten/momentum tendencies, and proven "
    "horizontally homogeneous CM1 base arrays; stage-by-stage CUDA cadence "
    "is exposed; no adaptive nsound derivation, eqtset2 moisture closure, "
    "solve2/solve3 completion, terrain, MPI, axisymmetry, open or rigid "
    "lateral boundaries, immersed boundary, convergence/updraft forcing, "
    "coupled trajectory, forecast, meteorological product, or performance claim"
)


@dataclass(frozen=True, slots=True)
class Psolver3PeriodicSelection:
    """Exact admitted namelist/build envelope for this operator."""

    psolver: int = 3
    terrain_flag: bool = False
    axisymm: int = 0
    mpi: bool = False
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    eqtset: int = 2
    apmasscon: int = 1
    do_ib: bool = False
    convinit: int = 0
    wnudge: int = 0


def require_psolver3_periodic_selection(
    selection: Psolver3PeriodicSelection,
) -> None:
    """Refuse every branch outside the source-audited selected family."""

    checks = (
        ("psolver", selection.psolver, 3, "ARWEN-PS3P-PSOLVER"),
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-PS3P-TERRAIN"),
        ("axisymm", selection.axisymm, 0, "ARWEN-PS3P-AXISYMM"),
        ("mpi", selection.mpi, False, "ARWEN-PS3P-MPI"),
        ("wbc", selection.wbc, 1, "ARWEN-PS3P-WBC"),
        ("ebc", selection.ebc, 1, "ARWEN-PS3P-EBC"),
        ("sbc", selection.sbc, 1, "ARWEN-PS3P-SBC"),
        ("nbc", selection.nbc, 1, "ARWEN-PS3P-NBC"),
        ("irbc", selection.irbc, 4, "ARWEN-PS3P-IRBC"),
        ("roflux", selection.roflux, 0, "ARWEN-PS3P-ROFLUX"),
        ("nudgeobc", selection.nudgeobc, 0, "ARWEN-PS3P-NUDGE-OBC"),
        ("eqtset", selection.eqtset, 2, "ARWEN-PS3P-EQTSET"),
        ("apmasscon", selection.apmasscon, 1, "ARWEN-PS3P-APMASSCON"),
        ("do_ib", selection.do_ib, False, "ARWEN-PS3P-IMMERSED"),
        ("convinit", selection.convinit, 0, "ARWEN-PS3P-CONVINIT"),
        ("wnudge", selection.wnudge, 0, "ARWEN-PS3P-WNUDGE"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="psolver3-flat-periodic-selection",
                reason=(
                    f"the bounded psolver=3 family requires {option}={expected!r}; "
                    "the official alternate branch is not admitted here"
                ),
                remedy=(
                    "select the exact flat serial-periodic eqtset=2 pressure "
                    "profile or use a separately ratified psolver=3 operator"
                ),
                option=option,
                value=value,
            )


def _selection_payload(selection: Psolver3PeriodicSelection) -> dict[str, Any]:
    return {
        name: getattr(selection, name) for name in selection.__dataclass_fields__
    }


def _validate_scalars(*, cp: float, alph: float, kdiv: float) -> None:
    if not math.isfinite(cp) or cp <= 0.0:
        raise ValueError("cp must be finite and positive")
    if not math.isfinite(alph) or not 0.0 < alph <= 1.0:
        raise ValueError("alph must be finite and in (0, 1]")
    if not math.isfinite(kdiv) or kdiv < 0.0:
        raise ValueError("kdiv must be finite and nonnegative")


def validate_psolver3_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
) -> None:
    """Validate layouts, periodic state, and CM1's flat base-profile rule."""

    require_psolver3_periodic_selection(selection)
    validate_periodic_contract(arrays, ni=ni, nj=nj, nk=nk)
    np = importlib.import_module("numpy")
    for name in BASE_PROFILE_NAMES:
        live = arrays[name][:, 1 : nj + 1, 1 : ni + 1]
        reference = arrays[name][:, 1:2, 1:2]
        expected = np.broadcast_to(reference, live.shape)
        if not np.array_equal(live.view(np.uint32), expected.view(np.uint32)):
            raise ValueError(
                f"{name} must be horizontally homogeneous because flat sound.F "
                "reads the base profile at (1,1,k)"
            )


def cpu_psolver3_periodic_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
    **kwargs: Any,
) -> CpuSound3PeriodicStageDiagnostics:
    """Execute one selected CPU stage, preserving an external RK interleave."""

    ni, nj, nk = (int(kwargs[name]) for name in ("ni", "nj", "nk"))
    validate_psolver3_periodic_contract(
        arrays, ni=ni, nj=nj, nk=nk, selection=selection
    )
    _validate_scalars(
        cp=float(kwargs.get("cp", CM1_SOUND3_PERIODIC_CP_F32)),
        alph=float(kwargs.get("alph", CM1_SOUND3_PERIODIC_ALPH_F32)),
        kdiv=float(kwargs.get("kdiv", CM1_SOUND3_PERIODIC_KDIV_F32)),
    )
    return cpu_sound3_periodic_stage_inplace(arrays, **kwargs)


def cpu_psolver3_periodic_three_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
    **kwargs: Any,
) -> CpuSound3PeriodicResult:
    """Execute the held-input three-stage CPU convenience transition."""

    cadence = tuple(
        cpu_psolver3_periodic_stage_inplace(
            arrays, nrk=nrk, selection=selection, **kwargs
        )
        for nrk in (1, 2, 3)
    )
    return CpuSound3PeriodicResult(cadence)


@dataclass(frozen=True, slots=True)
class Psolver3PeriodicDeviceProof:
    """Device-bound proof for immutable selected-profile contract inputs."""

    ni: int
    nj: int
    nk: int
    device_id: int
    pointers: tuple[tuple[str, int], ...]
    contract_sha256: str


def prove_cuda_psolver3_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
) -> Psolver3PeriodicDeviceProof:
    """Synchronously prove periodic/layout/base preconditions once per allocation."""

    require_psolver3_periodic_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    digest = hashlib.sha256()

    def prove_exact(label: str, left: Any, right: Any, reason: str) -> None:
        if not bool(cupy.array_equal(left.view(cupy.uint32), right.view(cupy.uint32))):
            raise ValueError(reason)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(
            cupy.asnumpy(left).astype("<f4", copy=False).tobytes(order="C")
        )

    expected_u, expected_v = default_periodic_face_weights(ni, nj)
    prove_exact(
        "dtu0", a["dtu0"], cupy.asarray(expected_u),
        "dtu0 differs from CM1 serial-periodic default face weights",
    )
    prove_exact(
        "dtv0", a["dtv0"], cupy.asarray(expected_v),
        "dtv0 differs from CM1 serial-periodic default face weights",
    )
    for name in ("uf", "vf", "uh", "vh"):
        prove_exact(
            name,
            a[name],
            cupy.ones_like(a[name]),
            f"{name} must be exactly one for this uniform flat-grid slice",
        )
    for name in BASE_PROFILE_NAMES:
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        profile = a[name][:, 1:2, 1:2]
        exact = cupy.all(live.view(cupy.uint32) == profile.view(cupy.uint32)).item()
        if not bool(exact):
            raise ValueError(
                f"{name} must be horizontally homogeneous because flat sound.F "
                "reads the base profile at (1,1,k)"
            )
        digest.update(name.encode("ascii") + b"\0")
        digest.update(cupy.asnumpy(profile).astype("<f4", copy=False).tobytes(order="C"))
    for name in ("piadv", "thv", "ppi", "ppx"):
        value = a[name]
        prove_exact(
            f"{name}-west",
            value[1 : nk + 1, :, 0],
            value[1 : nk + 1, :, ni],
            f"{name} west periodic halo is invalid",
        )
        prove_exact(
            f"{name}-east",
            value[1 : nk + 1, :, ni + 1],
            value[1 : nk + 1, :, 1],
            f"{name} east periodic halo is invalid",
        )
        prove_exact(
            f"{name}-south",
            value[1 : nk + 1, 0, :],
            value[1 : nk + 1, nj, :],
            f"{name} south periodic halo is invalid",
        )
        prove_exact(
            f"{name}-north",
            value[1 : nk + 1, nj + 1, :],
            value[1 : nk + 1, 1, :],
            f"{name} north periodic halo is invalid",
        )
    prove_exact(
        "ua-duplicate-x-face",
        a["ua"][1 : nk + 1, 1 : nj + 1, 0],
        a["ua"][1 : nk + 1, 1 : nj + 1, ni],
        "ua duplicate periodic x faces differ",
    )
    prove_exact(
        "va-duplicate-y-face",
        a["va"][1 : nk + 1, 0, 1 : ni + 1],
        a["va"][1 : nk + 1, nj, 1 : ni + 1],
        "va duplicate periodic y faces differ",
    )
    device_id = int(cupy.cuda.Device().id)
    pointers = tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES)
    return Psolver3PeriodicDeviceProof(
        ni, nj, nk, device_id, pointers, digest.hexdigest()
    )


def _require_matching_base_proof(
    proof: Psolver3PeriodicDeviceProof,
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    cupy: Any,
) -> None:
    if (proof.ni, proof.nj, proof.nk) != (ni, nj, nk):
        raise ValueError("psolver3 periodic base proof dimensions do not match")
    if proof.device_id != int(cupy.cuda.Device().id):
        raise ValueError("psolver3 periodic base proof belongs to another CUDA device")
    current = tuple((name, int(arrays[name].data.ptr)) for name in ALL_ARRAY_NAMES)
    if current != proof.pointers:
        raise ValueError("psolver3 periodic base proof does not match these allocations")


@dataclass(frozen=True, slots=True)
class CudaPsolver3PeriodicLaunch:
    ready_event: Any
    compiled: CompiledSound3PeriodicKernels
    device_proof: Psolver3PeriodicDeviceProof
    receipt: dict[str, Any]


def launch_cuda_psolver3_periodic_stage(
    arrays: Mapping[str, Any],
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
    device_proof: Psolver3PeriodicDeviceProof,
    cp: float = CM1_SOUND3_PERIODIC_CP_F32,
    alph: float = CM1_SOUND3_PERIODIC_ALPH_F32,
    kdiv: float = CM1_SOUND3_PERIODIC_KDIV_F32,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledSound3PeriodicKernels | None = None,
) -> CudaPsolver3PeriodicLaunch:
    """Enqueue one acoustic RK stage without synchronizing.

    ``device_proof`` is deliberately mandatory so a live forecast can prove
    the selected periodic/base contract once, then interleave stages without
    hidden host syncs.
    """

    require_psolver3_periodic_selection(selection)
    _validate_scalars(cp=cp, alph=alph, kdiv=kdiv)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    _require_matching_base_proof(
        device_proof, a, ni=ni, nj=nj, nk=nk, cupy=cupy
    )
    compiled = compile_cuda_sound3_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    kprep, kprepuv, kupdate, kcolumn, kavg, kbcp, kbcuv = compiled.kernels
    f32, u32 = cupy.float32, cupy.uint32
    alph32 = f32(alph)
    beta = f32(f32(1.0) - alph32)
    asq = f32(alph32 * alph32)
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    face_words = max(
        (nk + 2) * (nj + 2) * (ni + 1),
        (nk + 2) * (nj + 1) * (ni + 2),
    )
    grid_prep = (
        (max(scalar_words, ni * nj) + CUDA_SOUND3_PERIODIC_THREADS - 1)
        // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    grid_faces = (
        (face_words + CUDA_SOUND3_PERIODIC_THREADS - 1)
        // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    grid_cols = (
        (ni * nj + CUDA_SOUND3_PERIODIC_THREADS - 1)
        // CUDA_SOUND3_PERIODIC_THREADS,
        1,
        1,
    )
    block = (CUDA_SOUND3_PERIODIC_THREADS, 1, 1)
    kprep(
        grid_prep,
        block,
        (
            a["mh"], a["rr0"], a["rf0"], a["c1"], a["c2"], a["mf"],
            a["piadv"], a["ppterm"], a["thv"], a["ppi"], a["ppx"], a["wa"],
            a["w3d"], a["pp3d"], a["ppd"], a["rrw"], a["qk"], a["pk1"],
            a["pk2"], a["mm"], a["fpk"], a["divh"], a["hadvp"],
            a["elim_c"], a["elim_d"], a["tri_aa"], a["tri_bb"],
            a["tri_cc"], a["tri_dd"], a["tri_pivot"], a["tri_residual"],
            u32(ni), u32(nj), u32(nk), u32(nrk), f32(dts), f32(rdz), f32(cp), beta,
        ),
        stream=stream,
    )
    kprepuv(
        grid_faces,
        block,
        (
            a["ua"], a["va"], a["u3d"], a["v3d"], a["rru"], a["rrv"],
            u32(ni), u32(nj), u32(nk), u32(nrk),
        ),
        stream=stream,
    )
    for n in range(1, nloop + 1):
        kupdate(
            grid_faces,
            block,
            (
                a["ppd"], a["thv"], a["uten"], a["vten"], a["dtu0"],
                a["dtv0"], a["uf"], a["vf"], a["u3d"], a["v3d"],
                u32(ni), u32(nj), u32(nk), f32(dts), f32(rdx), f32(rdy), f32(cp),
            ),
            stream=stream,
        )
        kcolumn(
            grid_cols,
            block,
            (
                a["piadv"], a["ppterm"], a["ppten"], a["wten"], a["u3d"],
                a["v3d"], a["uh"], a["vh"], a["w3d"], a["pp3d"], a["ppx"],
                a["ppd"], a["rrw"], a["qk"], a["pk1"], a["pk2"], a["mm"],
                a["fpk"], a["divh"], a["hadvp"], a["elim_c"], a["elim_d"],
                a["tri_aa"], a["tri_bb"], a["tri_cc"], a["tri_dd"],
                a["tri_pivot"], a["tri_residual"], u32(ni), u32(nj), u32(nk),
                u32(nrk), u32(n), u32(nloop), f32(dts), f32(rdx), f32(rdy),
                alph32, beta, asq, f32(kdiv), f32(CM1_SOUND3_PERIODIC_SMEPS_F32),
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
            block,
            (
                a["u3d"], a["v3d"], a["rru"], a["rrv"], u32(ni), u32(nj),
                u32(nk), u32(n), u32(nloop),
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
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    cadence = {
        "nrk": nrk,
        "dttmp_f32": rk_stage_dt_f32(dt, nrk),
        "nloop": nloop,
        "dts_f32": float(f32(dts)),
        "ppd_boundary_fills": nloop - 1,
        "stage_velocity_boundary_fills": 1,
        "stage_pressure_boundary_fills": 1 if nrk < 3 else 0,
        "ppx_written": nrk == 3,
    }
    receipt = {
        "schema": CUDA_PSOLVER3_PERIODIC_SCHEMA,
        "abi": CUDA_PSOLVER3_PERIODIC_ABI,
        "selection": _selection_payload(selection),
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": cadence,
        "device_contract_proof": {
            "names": list(BASE_PROFILE_NAMES),
            "contract_sha256": device_proof.contract_sha256,
            "device_id": device_proof.device_id,
            "precondition": "valid while the proven allocations and periodic/base contract inputs remain unchanged",
        },
        "kernel": {
            "upstream_abi": CUDA_SOUND3_PERIODIC_ABI,
            "names": list(CUDA_SOUND3_PERIODIC_KERNEL_NAMES),
            "source_sha256": CUDA_SOUND3_PERIODIC_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "claim_limit": CM1_PSOLVER3_PERIODIC_CLAIM_LIMIT,
        "underlying_operator_claim_limit": CUDA_SOUND3_PERIODIC_CLAIM_LIMIT,
    }
    return CudaPsolver3PeriodicLaunch(event, compiled, device_proof, receipt)


def launch_cuda_psolver3_periodic_three_stage(
    arrays: Mapping[str, Any],
    *,
    device_proof: Psolver3PeriodicDeviceProof | None = None,
    selection: Psolver3PeriodicSelection = Psolver3PeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledSound3PeriodicKernels | None = None,
    **kwargs: Any,
) -> CudaPsolver3PeriodicLaunch:
    """Run all stages with held inputs; live callers should use the stage API."""

    ni, nj, nk = (int(kwargs[name]) for name in ("ni", "nj", "nk"))
    if device_proof is None:
        device_proof = prove_cuda_psolver3_periodic_contract(
            arrays, ni=ni, nj=nj, nk=nk, selection=selection
        )
    last: CudaPsolver3PeriodicLaunch | None = None
    cadence: list[dict[str, Any]] = []
    for nrk in (1, 2, 3):
        last = launch_cuda_psolver3_periodic_stage(
            arrays,
            nrk=nrk,
            device_proof=device_proof,
            selection=selection,
            stream=stream,
            compiled=compiled,
            **kwargs,
        )
        compiled = last.compiled
        cadence.append(last.receipt["cadence"])
    assert last is not None
    receipt = dict(last.receipt)
    receipt["cadence"] = cadence
    receipt["transition"] = (
        "held-input three-stage wrapper; callers needing live tendency/RK "
        "interleaving must call launch_cuda_psolver3_periodic_stage after each rebuild"
    )
    return CudaPsolver3PeriodicLaunch(
        last.ready_event, last.compiled, device_proof, receipt
    )


__all__ = (
    "ALL_ARRAY_NAMES",
    "BASE_PROFILE_NAMES",
    "CM1_PSOLVER3_PERIODIC_CLAIM_LIMIT",
    "COMPARE_OUTPUT_NAMES",
    "CUDA_PSOLVER3_PERIODIC_ABI",
    "CUDA_PSOLVER3_PERIODIC_SCHEMA",
    "CUDA_SOUND3_PERIODIC_SOURCE_SHA256",
    "CudaPsolver3PeriodicLaunch",
    "Psolver3PeriodicDeviceProof",
    "Psolver3PeriodicSelection",
    "SOURCE_ANCHORS",
    "array_shapes",
    "compare_sound3_periodic_raw",
    "compile_cuda_sound3_periodic",
    "cpu_psolver3_periodic_stage_inplace",
    "cpu_psolver3_periodic_three_stage_inplace",
    "default_periodic_face_weights",
    "fill_periodic_scalar_inplace",
    "fill_periodic_uv_inplace",
    "launch_cuda_psolver3_periodic_stage",
    "launch_cuda_psolver3_periodic_three_stage",
    "prove_cuda_psolver3_periodic_contract",
    "require_psolver3_periodic_selection",
    "scalar_shape_zyx",
    "u_shape_zyx",
    "v_shape_zyx",
    "validate_psolver3_periodic_contract",
    "w_shape_zyx",
)
