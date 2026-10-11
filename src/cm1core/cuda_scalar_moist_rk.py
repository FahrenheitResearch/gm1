"""CM1 r21.1 scalar/moisture low-storage RK stage composition.

The source contract is deliberately split at the scheduler boundary.  A
caller supplies the A carriers, current stage carriers, pre-RK held
tendencies, the stage-local ``t11*thterm`` field, and mass fluxes formed after
that stage's acoustic work.  This module keeps all numerical work on device,
calls the pinned fifth-order scalar operator, applies CM1's final-stage
positive-definite repair for moisture, and writes the exact low-storage
``A + dttmp*tendency`` update.

It does not produce the held tendencies, acoustic state, mass flux, or
``eqtset=2`` divergence term.  Those are earlier source-ordered scheduler
responsibilities and remain separate restart inventory.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_advection5 import (
    CM1_NRKMAX,
    CompiledAdvection5Kernels,
    compile_cuda_advection5,
    launch_cuda_scalar_advection5,
    scalar_open_boundary_code,
    scalar_advection_uses_weno,
)
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    CompiledOpenBoundaryKernels,
    compile_cuda_sound3_open_boundary,
)
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


#: v2: the stage receipt names every row of the bound axis (``species``), counts
#: it (``numq``) and lists one ``weps`` per row.  v1 receipts always listed the
#: six WSM6 names and six weps, whatever axis the stage advanced.
#: v3: ``moisture_advection`` says what formed each moisture row's advection:
#: the plain fifth-order launch, the final stage's pdef chain alone, or (under
#: ``GM1_FINAL_STAGE_MOISTURE_ADVECTION=launch``) the plain launch and then the
#: chain that rewrites every word of it.
CUDA_SCALAR_MOIST_RK_SCHEMA = "gpuwm-cm1.cuda-scalar-moist-rk/v3"
CUDA_SCALAR_MOIST_RK_ABI = "gpuwm-cm1.cuda-scalar-moist-rk-kernels/v1"
CUDA_SCALAR_MOIST_RK_THREADS = 128
CM1_SCALAR_MOIST_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")
CM1_TH0R_F32 = 300.0
CM1_SMEPS_F32 = 1.0e-30
CM1_EPSILON_F32 = 1.0e-18
CM1_QMAG_F32 = (1.0e-2,) * 6

#: What the final RK stage enqueues for each moisture row before its pdef
#: chain.  ``pdef-only`` (the default) enqueues nothing: adv.F:421-445 with
#: pdef=1 (solve2.F:1451-1455) forms the row's advection from the chain alone,
#: and every word the plain fifth-order launch wrote -- the fx/fy faces,
#: fz(k=2..nk) and the tendency interior -- the chain rewrote before any read.
#: ``launch`` enqueues that launch anyway, as every run did before, so a device
#: measurement can time and compare both from one tree and one restart.  It
#: is not a configuration key and does not enter a document's sealed hash.
FINAL_STAGE_MOISTURE_ADVECTION_ENVIRONMENT = "GM1_FINAL_STAGE_MOISTURE_ADVECTION"
FINAL_STAGE_MOISTURE_ADVECTION_MODES = ("pdef-only", "launch")


def final_stage_moisture_advection_record(nrk: int, mode: str) -> str:
    """The receipt word for what formed a stage's moisture advection."""

    if nrk != CM1_NRKMAX:
        return "scalar-advection5"
    return "pdef-chain-only" if mode == "pdef-only" else "scalar-advection5-then-pdef-chain"


def final_stage_moisture_advection_mode(environment: Mapping[str, str] | None = None) -> str:
    """The final-stage moisture launch an environment selects; unknown values are refused."""

    value = (os.environ if environment is None else environment).get(
        FINAL_STAGE_MOISTURE_ADVECTION_ENVIRONMENT, "pdef-only"
    )
    if value not in FINAL_STAGE_MOISTURE_ADVECTION_MODES:
        raise ValueError(
            f"{FINAL_STAGE_MOISTURE_ADVECTION_ENVIRONMENT}={value!r} is not one of "
            f"{FINAL_STAGE_MOISTURE_ADVECTION_MODES}; an unrecognised value would "
            "measure a launch sequence nobody selected"
        )
    return value

SOURCE_ANCHORS = {
    "rk_schedule": "src/solve2.F:294-299",
    "theta_seed_and_transport": "src/solve2.F:1313-1399",
    "theta_update": "src/solve2.F:1412-1424",
    "moisture_seed_selector": "src/solve2.F:1430-1473",
    "moisture_advection": "src/solve2.F:1482-1517",
    "moisture_update": "src/solve2.F:1519-1531",
    "advection_accumulation": "src/adv.F:631-659",
    "positive_definite_order": "src/adv.F:421-445;618-627",
    "positive_definite_repair": "src/pdef.F:15-581",
    "weno_selector": "src/adv.F:75-78",
    "weno_pdef_horizontal": "src/adv_routines.F:204-430",
    "weno_pdef_vertical": "src/adv_routines.F:2359-2557;4038-4447",
    "scalar_boundaries": "src/solve2.F:1681-1706;src/bc.F:180-484",
}

CUDA_SCALAR_MOIST_RK_CLAIM_LIMIT = (
    "CM1 r21.1 three-stage scalar/moisture low-storage A-anchored update, "
    "theta relative-300-K transport construction, final-stage-only scalar "
    "WENO cadence, and source-independent-interior terrain pdef moisture "
    "repair for six semantic WSM6 carriers. Held-tendency production, live "
    "eqtset=2 t11/thterm production, acoustic/mass-flux evolution, source-"
    "external open-boundary values, MPI, diffusion/subsidence, dry-mass "
    "correction, microphysics, a coupled step, trajectory, forecast, storm, "
    "tornado, and model validation remain excluded"
)


def _f32(np: Any, value: Any) -> Any:
    return np.float32(value)


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def scalar_moist_rk_stage_timestep_f32(dt: float, nrk: int) -> Any:
    """Return CM1's default-REAL ``dt/(nrkmax+1-nrk)`` value."""

    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError("dt must be numeric")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    np = importlib.import_module("numpy")
    return _div(np, np.float32(dt), np.float32(CM1_NRKMAX + 1 - nrk))


def scalar_moist_rk_cadence(nrk: int) -> dict[str, object]:
    """Describe the sealed stage selector without importing CUDA."""

    d = scalar_moist_rk_stage_timestep_f32(1.0, nrk)
    return {
        "nrk": nrk,
        "divisor": CM1_NRKMAX + 1 - nrk,
        "unit_dt_fraction_f32": float(d),
        "uses_weno": scalar_advection_uses_weno(nrk),
        "moisture_positive_definite": nrk == CM1_NRKMAX,
        "refresh_scalar_halos_for_next_stage": nrk < CM1_NRKMAX,
    }


def moisture_weps_f64(species: int, qmag: tuple[float, ...] = CM1_QMAG_F32) -> float:
    """``src/solve2.F:1472``: ``weps = qmag(n)*epsilon``, in CM1's binary32.

    ``qmag`` is CM1's own per-scalar order-of-magnitude table, built once per
    ``ptype`` in ``src/param.F``.  It is ``1.0e-2`` for every mass mixing
    ratio and ``1.0e6`` for every number concentration
    (``src/param.F:3464-3473`` for ``ptype=5``, ``:3673-3678`` for
    ``ptype=7``), so a double-moment axis must not be given the mass value for
    its number rows: the WENO positive-definite epsilon would then be
    twenty-two orders of magnitude below the field it guards.
    """

    if isinstance(species, bool) or not isinstance(species, int):
        raise TypeError("species must be an integer")
    if species < 0 or species >= len(qmag):
        raise ValueError(f"species must index one of {len(qmag)} CM1 q3d scalars")
    np = importlib.import_module("numpy")
    return float(
        _mul(
            np,
            np.float32(qmag[species]),
            np.float32(CM1_EPSILON_F32),
        )
    )


def scalar_moist_rk_numq(state: Any) -> int:
    """``src/solve2.F:1435``: the number of rows the moisture RK loop advances.

    CM1 runs ``DO n=1,numq`` with one ``advs`` call per scalar and nothing in
    the loop distinguishes a number concentration from a mixing ratio beyond
    its own ``qmag(n)`` (``:1472``) and the ``nqv`` pdefweno switch
    (``:1498``, ``:1520``).  The count is the length of the state's own
    ``qmag`` table, and every moisture carrier the stage reads or writes must
    lead with exactly that many rows: a longer axis would leave its trailing
    scalars un-advected, holding the previous stage's words, and a shorter one
    would be indexed past its allocation.  The state's ``names`` (CM1's
    ``qname``) must describe the same axis, or the stage receipt would publish
    one scalar's weps under another scalar's name.
    """

    qmag = getattr(state, "qmag", None)
    if isinstance(qmag, (str, bytes)) or not isinstance(qmag, (tuple, list)) or not qmag:
        raise ValueError("scalar/moist RK state carries no CM1 qmag table")
    numq = len(qmag)
    names = getattr(state, "names", None)
    if (
        not isinstance(names, (tuple, list))
        or len(names) != numq
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != numq
    ):
        raise ValueError(
            f"scalar/moist RK state names {names!r} but its qmag table describes "
            f"numq={numq}; the stage receipt would attribute a row's weps to the "
            "wrong CM1 qname"
        )
    for name in ("moisture_a", "moisture_current", "moisture_held"):
        value = getattr(state, name)
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) != 4 or int(shape[0]) != numq:
            raise ValueError(
                f"{name} carries {shape[0] if shape else 'no'} scalar rows but "
                f"the bound qmag table describes numq={numq}; solve2.F:1435 "
                "advances every row the axis holds, so the stage would skip or "
                "overrun a moisture scalar"
            )
    return numq


def theta_weps_f64() -> float:
    np = importlib.import_module("numpy")
    return float(_mul(np, np.float32(10.0), np.float32(CM1_EPSILON_F32)))


def _require_host_f32(np: Any, value: Any, name: str, shape: tuple[int, ...]) -> Any:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def cpu_theta_transport_scalar_inplace(
    th0: Any,
    th3d: Any,
    padded: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    """Independent CPU transcription of ``(th0-th0r)+th3d`` packing."""

    np = importlib.import_module("numpy")
    th0 = _require_host_f32(np, th0, "th0", (nk,))
    th3d = _require_host_f32(np, th3d, "th3d", (nk, nj, ni))
    padded = _require_host_f32(
        np,
        padded,
        "padded",
        (nk + 2, nj + 2 * CM1_OPEN_NGXY, ni + 2 * CM1_OPEN_NGXY),
    )
    ng = CM1_OPEN_NGXY
    for k in range(nk):
        base = _sub(np, th0[k], np.float32(CM1_TH0R_F32))
        for j in range(nj):
            for i in range(ni):
                padded[k + 1, j + ng, i + ng] = _add(np, base, th3d[k, j, i])


def cpu_compose_scalar_moist_rk_stage_inplace(
    theta_a: Any,
    moisture_a: Any,
    theta_held: Any,
    moisture_held: Any,
    theta_divergence_term: Any,
    theta_advection: Any,
    moisture_advection: Any,
    theta_current: Any,
    moisture_current: Any,
    *,
    nrk: int,
    dt: float,
) -> None:
    """Compose one source-ordered A-anchored scalar/moisture stage on CPU.

    ``theta_divergence_term`` is the already-produced positive expression
    ``t11*thterm``.  It is subtracted from ``theta_held`` exactly where CM1
    seeds ``thten``.  Advection arrays are increments produced by ``advs``.
    """

    np = importlib.import_module("numpy")
    if not isinstance(theta_a, np.ndarray) or theta_a.dtype != np.float32:
        raise TypeError("theta_a must be a float32 numpy.ndarray")
    if theta_a.ndim != 3 or not theta_a.flags.c_contiguous:
        raise ValueError("theta_a must be a C-order 3-D array")
    shape = theta_a.shape
    qshape = (len(CM1_SCALAR_MOIST_SPECIES),) + shape
    theta_values = {
        name: _require_host_f32(np, value, name, shape)
        for name, value in (
            ("theta_a", theta_a),
            ("theta_held", theta_held),
            ("theta_divergence_term", theta_divergence_term),
            ("theta_advection", theta_advection),
            ("theta_current", theta_current),
        )
    }
    moisture_values = {
        name: _require_host_f32(np, value, name, qshape)
        for name, value in (
            ("moisture_a", moisture_a),
            ("moisture_held", moisture_held),
            ("moisture_advection", moisture_advection),
            ("moisture_current", moisture_current),
        )
    }
    pointers = [
        int(value.ctypes.data)
        for value in (*theta_values.values(), *moisture_values.values())
    ]
    if len(pointers) != len(set(pointers)):
        raise ValueError("RK inputs and outputs must not alias")
    dttmp = scalar_moist_rk_stage_timestep_f32(dt, nrk)
    theta_seed = np.float32(
        theta_values["theta_held"] - theta_values["theta_divergence_term"]
    )
    theta_total = np.float32(theta_seed + theta_values["theta_advection"])
    theta_out = np.float32(theta_values["theta_a"] + np.float32(dttmp * theta_total))
    theta_out[np.abs(theta_out) < np.float32(CM1_SMEPS_F32)] = np.float32(0.0)
    np.copyto(theta_values["theta_current"], theta_out)
    q_total = np.float32(
        moisture_values["moisture_held"] + moisture_values["moisture_advection"]
    )
    q_out = np.float32(moisture_values["moisture_a"] + np.float32(dttmp * q_total))
    q_out[np.abs(q_out) < np.float32(CM1_SMEPS_F32)] = np.float32(0.0)
    np.copyto(moisture_values["moisture_current"], q_out)


def cpu_pdef_direction_inplace(
    scalar_a: Any,
    scalar_padded: Any,
    rho0: Any,
    gz: Any,
    rgz: Any,
    mass_flux: Any,
    directional: Any,
    candidate: Any,
    *,
    axis: str,
    dt: float,
    spacing: float,
) -> None:
    """Independent CPU transcription of one terrain ``pdef*1/*2`` pair.

    Horizontal physical-edge cells are deliberately excluded because their
    CM1 values depend on boundary-family halo exchange.  The source-independent
    interior is exact.  Vertical zero end halos are part of ``pdefz`` and are
    transcribed directly.
    """

    np = importlib.import_module("numpy")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    if not isinstance(scalar_a, np.ndarray) or scalar_a.dtype != np.float32:
        raise TypeError("scalar_a must be a float32 numpy.ndarray")
    if scalar_a.ndim != 3 or not scalar_a.flags.c_contiguous:
        raise ValueError("scalar_a must be a C-order 3-D array")
    nk, nj, ni = scalar_a.shape
    physical = (nk, nj, ni)
    padded_shape = (
        nk + 2,
        nj + 2 * CM1_OPEN_NGXY,
        ni + 2 * CM1_OPEN_NGXY,
    )
    scalar_padded = _require_host_f32(np, scalar_padded, "scalar_padded", padded_shape)
    rho0 = _require_host_f32(np, rho0, "rho0", physical)
    gz = _require_host_f32(np, gz, "gz", (nj, ni))
    rgz = _require_host_f32(np, rgz, "rgz", (nj, ni))
    directional = _require_host_f32(np, directional, "directional", physical)
    candidate = _require_host_f32(np, candidate, "candidate", physical)
    ng = CM1_OPEN_NGXY
    flux_shapes = {
        "x": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "y": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "z": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    mass_flux = _require_host_f32(np, mass_flux, "mass_flux", flux_shapes[axis])
    dttmp = np.float32(dt)
    if not np.isfinite(dttmp) or dttmp <= np.float32(0.0):
        raise ValueError("dt must be finite and > 0")
    reciprocal_dt = _div(np, np.float32(1.0), dttmp)
    reciprocal_spacing = _div(np, np.float32(1.0), np.float32(spacing))
    if not np.isfinite(reciprocal_spacing) or reciprocal_spacing <= np.float32(0.0):
        raise ValueError("spacing must be finite and > 0")

    for k in range(nk):
        kp = k + 1
        for j in range(nj):
            yp = j + ng
            for i in range(ni):
                xp = i + ng
                if axis == "x":
                    difference = _sub(
                        np,
                        mass_flux[kp, yp, xp + 1],
                        mass_flux[kp, yp, xp],
                    )
                elif axis == "y":
                    difference = _sub(
                        np,
                        mass_flux[kp, yp + 1, xp],
                        mass_flux[kp, yp, xp],
                    )
                else:
                    difference = _sub(
                        np,
                        mass_flux[kp + 1, yp, xp],
                        mass_flux[kp, yp, xp],
                    )
                # pdef.F:67-68 (y :308-309, z :493-494): s*(rru(i+1)-rru(i))*rdx*uh
                # is ((s*d)*rdx)*uh, left to right; uh = mh = 1 here.
                divergence = _mul(
                    np, _mul(np, scalar_padded[kp, yp, xp], difference), reciprocal_spacing
                )
                total = _add(np, directional[k, j, i], divergence)
                increment = _mul(np, _mul(np, dttmp, total), gz[j, i])
                candidate[k, j, i] = _add(
                    np, _mul(np, rho0[k, j, i], scalar_a[k, j, i]), increment
                )

    def repair_line(values: Any) -> tuple[Any, Any]:
        length = int(values.size)
        mass = np.zeros(length, dtype=np.float32)
        flags = np.zeros(length, dtype=np.bool_)
        start = 0 if axis == "z" else 1
        stop = length if axis == "z" else length - 1
        for index in range(start, stop):
            center = values[index]
            if center >= np.float32(0.0):
                continue
            left = (
                np.float32(0.0)
                if index == 0
                else np.maximum(np.float32(0.0), values[index - 1])
            )
            right = (
                np.float32(0.0)
                if index + 1 == length
                else np.maximum(np.float32(0.0), values[index + 1])
            )
            donor_sum = _add(np, left, right)
            if donor_sum <= np.float32(CM1_SMEPS_F32):
                continue
            ratio = _div(
                np,
                np.maximum(center, _sub(np, np.float32(0.0), donor_sum)),
                donor_sum,
            )
            if index > 0:
                mass[index - 1] = _add(np, mass[index - 1], _mul(np, left, ratio))
                if values[index - 1] > np.float32(CM1_SMEPS_F32):
                    flags[index - 1] = True
            mass[index] = _sub(np, mass[index], _mul(np, donor_sum, ratio))
            flags[index] = True
            if index + 1 < length:
                mass[index + 1] = _add(np, mass[index + 1], _mul(np, right, ratio))
                if values[index + 1] > np.float32(CM1_SMEPS_F32):
                    flags[index + 1] = True
        return mass, flags

    if axis == "x":
        lines = ((k, j, candidate[k, j, :]) for k in range(nk) for j in range(nj))
    elif axis == "y":
        lines = ((k, i, candidate[k, :, i]) for k in range(nk) for i in range(ni))
    else:
        lines = ((j, i, candidate[:, j, i]) for j in range(nj) for i in range(ni))
    for first, second, values in lines:
        mass, flags = repair_line(values)
        for index in range(int(values.size)):
            if not flags[index]:
                continue
            if axis == "x":
                k, j, i = first, second, index
            elif axis == "y":
                k, j, i = first, index, second
            else:
                k, j, i = index, first, second
            kp, yp, xp = k + 1, j + ng, i + ng
            if axis == "x":
                difference = _sub(np, mass_flux[kp, yp, xp + 1], mass_flux[kp, yp, xp])
            elif axis == "y":
                difference = _sub(np, mass_flux[kp, yp + 1, xp], mass_flux[kp, yp, xp])
            else:
                difference = _sub(np, mass_flux[kp + 1, yp, xp], mass_flux[kp, yp, xp])
            # pdef.F:199-200 (y :438-439, z :568-569), left to right.
            divergence = _mul(
                np, _mul(np, scalar_padded[kp, yp, xp], difference), reciprocal_spacing
            )
            adjusted = _sub(
                np,
                _add(np, candidate[k, j, i], mass[index]),
                _mul(np, rho0[k, j, i], scalar_a[k, j, i]),
            )
            adjusted = _mul(np, _mul(np, adjusted, reciprocal_dt), rgz[j, i])
            directional[k, j, i] = _sub(np, adjusted, divergence)


def scalar_moist_rk_interior_bounds(
    ni: int, nj: int, *, positive_definite: bool
) -> dict[str, tuple[int, int]]:
    for name, value in (("ni", ni), ("nj", nj)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 10:
            raise ValueError(f"{name} must be an integer >= 10")
    pad = 4 if positive_definite else 3
    return {"i": (pad + 1, ni - pad), "j": (pad + 1, nj - pad)}


CUDA_SCALAR_MOIST_RK_KERNEL_NAMES = (
    "cm1_rk_pack_theta_v1",
    "cm1_rk_pack_scalar_v1",
    "cm1_rk_compose_theta_v1",
    "cm1_rk_compose_moisture_v1",
    "cm1_rk_pdef_horizontal_flux_v1",
    "cm1_rk_pdef_vertical_flux_v1",
    "cm1_rk_pdef_directional_v1",
    "cm1_rk_pdef_candidate_x_v1",
    "cm1_rk_pdef_correct_x_v1",
    "cm1_rk_pdef_candidate_y_v1",
    "cm1_rk_pdef_correct_y_v1",
    "cm1_rk_pdef_candidate_z_v1",
    "cm1_rk_pdef_correct_z_v1",
    "cm1_rk_pdef_total_v1",
)


CUDA_SCALAR_MOIST_RK_SOURCE = r"""// gpuwm-cm1.cuda-scalar-moist-rk-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
static_assert(sizeof(double)==8,"CM1 WENO epsilon requires float64");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float flx3(float s1,float s2,float s3){float r=m32(-1.0f/6.0f,s1);r=a32(r,m32(5.0f/6.0f,s2));return a32(r,m32(2.0f/6.0f,s3));}
__device__ __forceinline__ float weno5(float s1,float s2,float s3,float s4,float s5,double weps){
 float d1=a32(s32(s1,m32(2.0f,s2)),s3),e1=a32(s32(s1,m32(4.0f,s2)),m32(3.0f,s3));
 float b1r=a32(m32(13.0f/12.0f,m32(d1,d1)),m32(0.25f,m32(e1,e1)));
 float d2=a32(s32(s2,m32(2.0f,s3)),s4),e2=s32(s2,s4);
 float b2r=a32(m32(13.0f/12.0f,m32(d2,d2)),m32(0.25f,m32(e2,e2)));
 float d3=a32(s32(s3,m32(2.0f,s4)),s5),e3=a32(s32(m32(3.0f,s3),m32(4.0f,s4)),s5);
 float b3r=a32(m32(13.0f/12.0f,m32(d3,d3)),m32(0.25f,m32(e3,e3)));
 double b1=(double)b1r,b2=(double)b2r,b3=(double)b3r,tau=fabs(b1-b3);
 double q1=fmin(1.0e30,tau/(b1+weps)),q2=fmin(1.0e30,tau/(b2+weps)),q3=fmin(1.0e30,tau/(b3+weps));
 double w1=(double)0.1f*(1.0+q1*q1),w2=(double)0.6f*(1.0+q2*q2),w3=(double)0.3f*(1.0+q3*q3);
 float p1=a32(a32(m32(2.0f/6.0f,s1),m32(-7.0f/6.0f,s2)),m32(11.0f/6.0f,s3));
 float p2=a32(a32(m32(-1.0f/6.0f,s2),m32(5.0f/6.0f,s3)),m32(2.0f/6.0f,s4));
 float p3=a32(a32(m32(2.0f/6.0f,s3),m32(5.0f/6.0f,s4)),m32(-1.0f/6.0f,s5));
 return (float)((w1*(double)p1+w2*(double)p2+w3*(double)p3)/(w1+w2+w3));}
__device__ __forceinline__ float upstrpd(float s1,float s2,float s3,float weps){float dd=s32(s2,s1);float signed_eps=copysignf(fabsf(weps),dd);float rr=d32(s32(s3,s2),a32(signed_eps,dd));float phi=fmaxf(0.0f,fminf(rr,fminf(a32(1.0f/6.0f,m32(2.0f/6.0f,rr)),1.0f)));return a32(s2,m32(phi,s32(s2,s1)));}
__device__ __forceinline__ float max5(float a,float b,float c,float d,float e){return fmaxf(fmaxf(fabsf(a),fabsf(b)),fmaxf(fabsf(c),fmaxf(fabsf(d),fabsf(e))));}
// obc packs CM1 wbc|ebc<<4|sbc<<8|nbc<<12 (bc.F lateral values 1..4).  obc=0 is a
// tile-local launch: the outer three cells belong to a neighbour tile and are skipped.
__device__ __forceinline__ unsigned int obc_side(unsigned int obc,unsigned int side){return (obc>>(4u*side))&15u;}
__device__ __forceinline__ bool obc_owned(unsigned int i,unsigned int j,unsigned int ni,unsigned int nj,unsigned int obc){return obc!=0u||(i>=3u&&i+3u<ni&&j>=3u&&j+3u<nj);}
__device__ __forceinline__ bool obc_inflow_lo(unsigned int i,unsigned int obc,unsigned int side,float m){return i==0u&&obc_side(obc,side)==2u&&m>=0.0f;}
__device__ __forceinline__ bool obc_inflow_hi(unsigned int i,unsigned int n,unsigned int obc,unsigned int side,float m){return i+1u==n&&obc_side(obc,side)==2u&&m<=0.0f;}
// pdef.F:125-166 on a whole domain: row cell c in [-2,n+1] as bcs(dum) fills it
// (bc.F:180-364): 1 periodic, 2 zero-gradient, 3/4 mirror.
__device__ __forceinline__ long long pdef_cell(long long c,long long n,unsigned int lo,unsigned int hi){if(c<0){if(lo==1u)return c+n;if(lo==2u)return 0;return -c-1;}if(c>=n){if(hi==1u)return c-n;if(hi==2u)return n-1;return 2*n-1-c;}return c;}
__device__ __forceinline__ float pdef_at(const float* d,long long base,long long stride,long long c,long long n,unsigned int lo,unsigned int hi){return d[base+pdef_cell(c,n,lo,hi)*stride];}
__device__ __forceinline__ float pdef_ratio(const float* d,long long base,long long stride,long long c,long long n,unsigned int lo,unsigned int hi){float v=pdef_at(d,base,stride,c,n,lo,hi);if(v>=0.0f)return 0.0f;float f1=fmaxf(0.0f,pdef_at(d,base,stride,c-1,n,lo,hi)),f2=fmaxf(0.0f,pdef_at(d,base,stride,c+1,n,lo,hi)),sum=a32(f1,f2);return(sum>1.0e-30f)?d32(fmaxf(v,-sum),sum):0.0f;}
__device__ __forceinline__ void pdef_row_mass(const float* d,long long base,long long stride,long long c,long long n,unsigned int lo,unsigned int hi,float* mass,bool* flag){float v=pdef_at(d,base,stride,c,n,lo,hi);if(v<0.0f){float r=pdef_ratio(d,base,stride,c,n,lo,hi);if(r!=0.0f){float sum=a32(fmaxf(0.0f,pdef_at(d,base,stride,c-1,n,lo,hi)),fmaxf(0.0f,pdef_at(d,base,stride,c+1,n,lo,hi)));*mass=s32(0.0f,m32(sum,r));*flag=true;}}else if(v>1.0e-30f){float rl=pdef_ratio(d,base,stride,c-1,n,lo,hi),rr=pdef_ratio(d,base,stride,c+1,n,lo,hi);if(rl!=0.0f){*mass=a32(*mass,m32(v,rl));*flag=true;}if(rr!=0.0f){*mass=a32(*mass,m32(v,rr));*flag=true;}}}

extern "C" __global__ void cm1_rk_pack_theta_v1(const float* th0,const float* th,float* p,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng;for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj));p[i3(k+1u,j+ng,i+ng,ny,nx)]=a32(s32(th0[k],300.0f),th[q]);}}
extern "C" __global__ void cm1_rk_pack_scalar_v1(const float* s,float* p,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng;for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj));p[i3(k+1u,j+ng,i+ng,ny,nx)]=s[q];}}
extern "C" __global__ void cm1_rk_compose_theta_v1(const float* a,const float* held,const float* divterm,const float* adv,float* out,unsigned long long n,float dttmp){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=step){float seed=s32(held[q],divterm[q]),v=a32(a[q],m32(dttmp,a32(seed,adv[q])));out[q]=(fabsf(v)<1.0e-30f)?0.0f:v;}}
extern "C" __global__ void cm1_rk_compose_moisture_v1(const float* a,const float* held,const float* adv,float* out,unsigned long long n,float dttmp){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;for(;q<n;q+=step){float v=a32(a[q],m32(dttmp,a32(held[q],adv[q])));out[q]=(fabsf(v)<1.0e-30f)?0.0f:v;}}

extern "C" __global__ void cm1_rk_pdef_horizontal_flux_v1(const float* s,const float* rru,const float* rrv,float* fx,float* fy,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,double weps){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*(nj+1u)*(ni+1u);unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng;for(;q<n;q+=step){unsigned int i=(unsigned int)(q%(ni+1u)),j=(unsigned int)((q/(ni+1u))%(nj+1u)),k=(unsigned int)(q/((unsigned long long)(ni+1u)*(nj+1u)))+1u;if(j<nj){unsigned int x=i+ng,y=j+ng;float m=rru[i3(k,y,x,nyu,nxu)],s1,s2,s3,s4,s5;if(m>=0.0f){s1=s[i3(k,y,x-3u,nys,nxs)];s2=s[i3(k,y,x-2u,nys,nxs)];s3=s[i3(k,y,x-1u,nys,nxs)];s4=s[i3(k,y,x,nys,nxs)];s5=s[i3(k,y,x+1u,nys,nxs)];}else{s1=s[i3(k,y,x+2u,nys,nxs)];s2=s[i3(k,y,x+1u,nys,nxs)];s3=s[i3(k,y,x,nys,nxs)];s4=s[i3(k,y,x-1u,nys,nxs)];s5=s[i3(k,y,x-2u,nys,nxs)];}fx[i3(k,y,x,nyu,nxu)]=(max5(s1,s2,s3,s4,s5)>weps)?m32(m,weno5(s1,s2,s3,s4,s5,weps)):0.0f;}if(i<ni){unsigned int x=i+ng,y=j+ng;float m=rrv[i3(k,y,x,nyv,nxv)],s1,s2,s3,s4,s5;if(m>=0.0f){s1=s[i3(k,y-3u,x,nys,nxs)];s2=s[i3(k,y-2u,x,nys,nxs)];s3=s[i3(k,y-1u,x,nys,nxs)];s4=s[i3(k,y,x,nys,nxs)];s5=s[i3(k,y+1u,x,nys,nxs)];}else{s1=s[i3(k,y+2u,x,nys,nxs)];s2=s[i3(k,y+1u,x,nys,nxs)];s3=s[i3(k,y,x,nys,nxs)];s4=s[i3(k,y-1u,x,nys,nxs)];s5=s[i3(k,y-2u,x,nys,nxs)];}fy[i3(k,y,x,nyv,nxv)]=(max5(s1,s2,s3,s4,s5)>weps)?m32(m,weno5(s1,s2,s3,s4,s5,weps)):0.0f;}}}

extern "C" __global__ void cm1_rk_pdef_vertical_flux_v1(
    const float* s,const float* rrw,const float* c1,const float* c2,float* fz,
    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
    unsigned int pdefweno,double weps){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
  unsigned long long n=(unsigned long long)(nk-1u)*nj*ni;
  unsigned int nx=ni+2u*ng,ny=nj+2u*ng;
  for(;q<n;q+=step){
    unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj);
    unsigned int k=(unsigned int)(q/((unsigned long long)ni*nj))+2u;
    unsigned int x=i+ng,y=j+ng;
    float m=rrw[i3(k,y,x,ny,nx)],r;
    if(k==2u){
      r=(m>=0.0f)
        ?a32(m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],s[i3(k-1u,y,x,ny,nx)]),
             m32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],s[i3(k,y,x,ny,nx)]))
        :upstrpd(s[i3(k+1u,y,x,ny,nx)],s[i3(k,y,x,ny,nx)],
                 s[i3(k-1u,y,x,ny,nx)],(float)weps);
    }else if(k==nk){
      r=(m>0.0f)
        ?upstrpd(s[i3(k-2u,y,x,ny,nx)],s[i3(k-1u,y,x,ny,nx)],
                 s[i3(k,y,x,ny,nx)],(float)weps)
        :a32(m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],s[i3(k-1u,y,x,ny,nx)]),
             m32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],s[i3(k,y,x,ny,nx)]));
    }else{
      // pdefweno=1 overwrites k=3 for rrw>0 (vadv_lwr_weno_35_b); pdefweno=0
      // leaves k=3 to vadv_lwr_weno_35, which takes the upstream branch for
      // rrw>=0, so a zero rrw there is 0*upstrpd, not 0*weno5.
      bool low=(k==3u&&(pdefweno?m>0.0f:m>=0.0f))||(k==nk-1u&&m<0.0f);
      if(low){
        r=(k==3u)
          ?upstrpd(s[i3(k-2u,y,x,ny,nx)],s[i3(k-1u,y,x,ny,nx)],
                   s[i3(k,y,x,ny,nx)],(float)weps)
          :upstrpd(s[i3(k+1u,y,x,ny,nx)],s[i3(k,y,x,ny,nx)],
                   s[i3(k-1u,y,x,ny,nx)],(float)weps);
      }else{
        float s1,s2,s3,s4,s5;
        if(m>=0.0f){
          s1=s[i3(k-3u,y,x,ny,nx)];s2=s[i3(k-2u,y,x,ny,nx)];
          s3=s[i3(k-1u,y,x,ny,nx)];s4=s[i3(k,y,x,ny,nx)];
          s5=s[i3(k+1u,y,x,ny,nx)];
        }else{
          s1=s[i3(k+2u,y,x,ny,nx)];s2=s[i3(k+1u,y,x,ny,nx)];
          s3=s[i3(k,y,x,ny,nx)];s4=s[i3(k-1u,y,x,ny,nx)];
          s5=s[i3(k-2u,y,x,ny,nx)];
        }
        // vadv_weno5's gate stores dumz=0.0, never rrw*0.0 (-0 for rrw<0).
        if(pdefweno&&max5(s1,s2,s3,s4,s5)<=weps){fz[i3(k,y,x,ny,nx)]=0.0f;continue;}
        r=weno5(s1,s2,s3,s4,s5,weps);
      }
    }
    fz[i3(k,y,x,ny,nx)]=m32(m,r);
  }
}

extern "C" __global__ void cm1_rk_pdef_directional_v1(const float* fx,const float* fy,const float* fz,float* ax,float* ay,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dx,float dy,float dz,const float* s,const float* rru,const float* rrv,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nxs=ni+2u*ng,nys=nj+2u*ng,nxu=ni+1u+2u*ng,nyu=nys,nxv=nxs,nyv=nj+1u+2u*ng;float rdx=d32(1.0f,dx),rdy=d32(1.0f,dy),rdz=d32(1.0f,dz);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj)),k=k0+1u,x=i+ng,y=j+ng;float fw=fx[i3(k,y,x,nyu,nxu)],fe=fx[i3(k,y,x+1u,nyu,nxu)],fs=fy[i3(k,y,x,nyv,nxv)],fn=fy[i3(k,y+1u,x,nyv,nxv)];float uw=rru[i3(k,y,x,nyu,nxu)],ue=rru[i3(k,y,x+1u,nyu,nxu)],vs=rrv[i3(k,y,x,nyv,nxv)],vn=rrv[i3(k,y+1u,x,nyv,nxv)];bool iw=obc_inflow_lo(i,obc,0u,uw),ie=obc_inflow_hi(i,ni,obc,1u,ue),js=obc_inflow_lo(j,obc,2u,vs),jn=obc_inflow_hi(j,nj,obc,3u,vn);if(iw)fw=fe;if(ie)fe=fw;if(js)fs=fn;if(jn)fn=fs;float axv=m32(s32(fw,fe),rdx),ayv=m32(s32(fs,fn),rdy);if(iw||ie||js||jn){float sv=s[i3(k,y,x,nys,nxs)];if(iw)axv=s32(axv,m32(m32(sv,s32(ue,uw)),rdx));if(ie)axv=s32(axv,m32(m32(sv,s32(ue,uw)),rdx));if(js)ayv=s32(ayv,m32(m32(sv,s32(vn,vs)),rdy));if(jn)ayv=s32(ayv,m32(m32(sv,s32(vn,vs)),rdy));}ax[q]=axv;ay[q]=ayv;az[q]=(k==1u)?m32(-fz[i3(2u,y,x,nys,nxs)],rdz):((k==nk)?m32(fz[i3(nk,y,x,nys,nxs)],rdz):m32(s32(fz[i3(k,y,x,nys,nxs)],fz[i3(k+1u,y,x,nys,nxs)]),rdz));}}

extern "C" __global__ void cm1_rk_pdef_candidate_x_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rru,const float* ax,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dx,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nxu=ni+1u+2u*ng;float rdx=d32(1.0f,dx);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj)),k=k0+1u,x=i+ng,y=j+ng;float du=s32(rru[i3(k,y,x+1u,ny,nxu)],rru[i3(k,y,x,ny,nxu)]);float r0a=m32(rho0[q],a[q]);dum[q]=(obc_inflow_lo(i,obc,0u,rru[i3(k,y,x,ny,nxu)])||obc_inflow_hi(i,ni,obc,1u,rru[i3(k,y,x+1u,ny,nxu)]))?r0a:a32(r0a,m32(m32(dt,a32(ax[q],m32(m32(s[i3(k,y,x,ny,nx)],du),rdx))),gz[(unsigned long long)j*ni+i]));}}
__device__ __forceinline__ float ratio1(const float* d,long long c,long long lo,long long hi,long long stride){if(c<=lo||c>=hi||d[c]>=0.0f)return 0.0f;float f1=fmaxf(0.0f,d[c-stride]),f2=fmaxf(0.0f,d[c+stride]),sum=a32(f1,f2);return(sum>1.0e-30f)?d32(fmaxf(d[c],-sum),sum):0.0f;}
extern "C" __global__ void cm1_rk_pdef_correct_x_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rru,const float* dum,float* ax,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dx,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nxu=ni+1u+2u*ng;float rdt=d32(1.0f,dt),rdx=d32(1.0f,dx);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj));float mass=0.0f;bool flag=false;if(obc!=0u){pdef_row_mass(dum,(long long)(q-i),1ll,(long long)i,(long long)ni,obc_side(obc,0u),obc_side(obc,1u),&mass,&flag);}else if(i>0u&&i+1u<ni){if(dum[q]<0.0f){float r=ratio1(dum,(long long)q,(long long)q-(long long)i,(long long)q+(long long)(ni-1u-i),1);if(r!=0.0f){float sum=a32(fmaxf(0.0f,dum[q-1u]),fmaxf(0.0f,dum[q+1u]));mass=s32(0.0f,m32(sum,r));flag=true;}}else if(dum[q]>1.0e-30f){float rl=ratio1(dum,(long long)q-1,(long long)q-(long long)i,(long long)q+(long long)(ni-1u-i),1),rr=ratio1(dum,(long long)q+1,(long long)q-(long long)i,(long long)q+(long long)(ni-1u-i),1);if(rl!=0.0f){mass=a32(mass,m32(dum[q],rl));flag=true;}if(rr!=0.0f){mass=a32(mass,m32(dum[q],rr));flag=true;}}}if(flag){unsigned int x=i+ng,y=j+ng,k=k0+1u;float first=m32(m32(s32(a32(dum[q],mass),m32(rho0[q],a[q])),rdt),rgz[(unsigned long long)j*ni+i]);float du=s32(rru[i3(k,y,x+1u,ny,nxu)],rru[i3(k,y,x,ny,nxu)]);ax[q]=s32(first,m32(m32(s[i3(k,y,x,ny,nx)],du),rdx));}}}

extern "C" __global__ void cm1_rk_pdef_candidate_y_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rrv,const float* ay,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dy,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nyv=nj+1u+2u*ng;float rdy=d32(1.0f,dy);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj)),k=k0+1u,x=i+ng,y=j+ng;float dv=s32(rrv[i3(k,y+1u,x,nyv,nx)],rrv[i3(k,y,x,nyv,nx)]);float r0a=m32(rho0[q],a[q]);dum[q]=(obc_inflow_lo(j,obc,2u,rrv[i3(k,y,x,nyv,nx)])||obc_inflow_hi(j,nj,obc,3u,rrv[i3(k,y+1u,x,nyv,nx)]))?r0a:a32(r0a,m32(m32(dt,a32(ay[q],m32(m32(s[i3(k,y,x,ny,nx)],dv),rdy))),gz[(unsigned long long)j*ni+i]));}}
extern "C" __global__ void cm1_rk_pdef_correct_y_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rrv,const float* dum,float* ay,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dy,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng,nyv=nj+1u+2u*ng;float rdt=d32(1.0f,dt),rdy=d32(1.0f,dy);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj));long long stride=(long long)ni;float mass=0.0f;bool flag=false;if(obc!=0u){pdef_row_mass(dum,(long long)q-(long long)j*stride,stride,(long long)j,(long long)nj,obc_side(obc,2u),obc_side(obc,3u),&mass,&flag);}else if(j>0u&&j+1u<nj){if(dum[q]<0.0f){float r=ratio1(dum,(long long)q,(long long)q-(long long)j*stride,(long long)q+(long long)(nj-1u-j)*stride,stride);if(r!=0.0f){float sum=a32(fmaxf(0.0f,dum[q-ni]),fmaxf(0.0f,dum[q+ni]));mass=s32(0.0f,m32(sum,r));flag=true;}}else if(dum[q]>1.0e-30f){float rl=ratio1(dum,(long long)q-stride,(long long)q-(long long)j*stride,(long long)q+(long long)(nj-1u-j)*stride,stride),rr=ratio1(dum,(long long)q+stride,(long long)q-(long long)j*stride,(long long)q+(long long)(nj-1u-j)*stride,stride);if(rl!=0.0f){mass=a32(mass,m32(dum[q],rl));flag=true;}if(rr!=0.0f){mass=a32(mass,m32(dum[q],rr));flag=true;}}}if(flag){unsigned int x=i+ng,y=j+ng,k=k0+1u;float first=m32(m32(s32(a32(dum[q],mass),m32(rho0[q],a[q])),rdt),rgz[(unsigned long long)j*ni+i]);float dv=s32(rrv[i3(k,y+1u,x,nyv,nx)],rrv[i3(k,y,x,nyv,nx)]);ay[q]=s32(first,m32(m32(s[i3(k,y,x,ny,nx)],dv),rdy));}}}

extern "C" __global__ void cm1_rk_pdef_candidate_z_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rrw,const float* az,float* dum,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dz){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float rdz=d32(1.0f,dz);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/((unsigned long long)ni*nj)),k=k0+1u,x=i+ng,y=j+ng;float dw=s32(rrw[i3(k+1u,y,x,ny,nx)],rrw[i3(k,y,x,ny,nx)]);dum[q]=a32(m32(rho0[q],a[q]),m32(m32(dt,a32(az[q],m32(m32(s[i3(k,y,x,ny,nx)],dw),rdz))),gz[(unsigned long long)j*ni+i]));}}
__device__ __forceinline__ float zval(const float* d,long long q,int k,unsigned long long slab,unsigned int nk){return(k<0||k>=(int)nk)?0.0f:d[q+(long long)k*slab];}
__device__ __forceinline__ float zratio(const float* d,long long base,int k,unsigned long long slab,unsigned int nk){float c=zval(d,base,k,slab,nk);if(c>=0.0f)return 0.0f;float f1=fmaxf(0.0f,zval(d,base,k-1,slab,nk)),f2=fmaxf(0.0f,zval(d,base,k+1,slab,nk)),sum=a32(f1,f2);return(sum>1.0e-30f)?d32(fmaxf(c,-sum),sum):0.0f;}
extern "C" __global__ void cm1_rk_pdef_correct_z_v1(const float* a,const float* s,const float* rho0,const float* gz,const float* rgz,const float* rrw,const float* dum,float* az,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dt,float dz){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni,slab=(unsigned long long)ni*nj;unsigned int nx=ni+2u*ng,ny=nj+2u*ng;float rdt=d32(1.0f,dt),rdz=d32(1.0f,dz);for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k0=(unsigned int)(q/slab);long long base=(long long)(q-(unsigned long long)k0*slab);float current=dum[q],mass=0.0f;bool flag=false;if(current<0.0f){float r=zratio(dum,base,(int)k0,slab,nk);if(r!=0.0f){float sum=a32(fmaxf(0.0f,zval(dum,base,(int)k0-1,slab,nk)),fmaxf(0.0f,zval(dum,base,(int)k0+1,slab,nk)));mass=s32(0.0f,m32(sum,r));flag=true;}}else if(current>1.0e-30f){float rl=zratio(dum,base,(int)k0-1,slab,nk),rr=zratio(dum,base,(int)k0+1,slab,nk);if(rl!=0.0f){mass=a32(mass,m32(current,rl));flag=true;}if(rr!=0.0f){mass=a32(mass,m32(current,rr));flag=true;}}if(flag){unsigned int k=k0+1u,x=i+ng,y=j+ng;float first=m32(m32(s32(a32(current,mass),m32(rho0[q],a[q])),rdt),rgz[(unsigned long long)j*ni+i]);float dw=s32(rrw[i3(k+1u,y,x,ny,nx)],rrw[i3(k,y,x,ny,nx)]);az[q]=s32(first,m32(m32(s[i3(k,y,x,ny,nx)],dw),rdz));}}}

extern "C" __global__ void cm1_rk_pdef_total_v1(const float* s,const float* ax,const float* ay,const float* az,const float* divx,const float* rr0,const float* gz,float* adv,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int obc){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;unsigned int nx=ni+2u*ng,ny=nj+2u*ng;for(;q<n;q+=step){unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj))+1u,x=i+ng,y=j+ng;if(obc_owned(i,j,ni,nj,obc)){float t=a32(a32(ax[q],ay[q]),az[q]);t=a32(t,m32(s[i3(k,y,x,ny,nx)],divx[q]));adv[q]=m32(m32(t,rr0[q]),gz[(unsigned long long)j*ni+i]);}}}
"""

CUDA_SCALAR_MOIST_RK_SOURCE_BYTES = CUDA_SCALAR_MOIST_RK_SOURCE.encode("utf-8")
CUDA_SCALAR_MOIST_RK_SOURCE_SHA256 = (
    "64132be2c80fe950b3f794de2e4a8d76b183948ea24d90a3714cd2a7faf16809"
)
#: The source the sealed ``cuda-scalar-moist-rk-v1`` / ``real-scalar-moist-rk-v1``
#: GPU receipts were measured with.  The receipts' words are that source's, not
#: the live source's and not CM1's.  It differs from the live source in these
#: spellings only:
#:
#: * ``weno5`` multiplied its weights by binary64 ``0.1``/``0.6``/``0.3`` where
#:   CM1's ``adv_routines.F:6189-6191`` promotes binary32 literals, which moves
#:   final-stage WENO5 face values;
#: * ``cm1_rk_pdef_vertical_flux_v1`` reconstructed ``k=3``'s ``rrw=+-0`` face
#:   with ``weno5`` under ``pdefweno=0`` (``low`` tested ``m>0.0f`` and picked
#:   the stencil by ``m>0.0f``) where ``vadv_lwr_weno_35`` takes ``upstrpd`` for
#:   ``rrw >= 0``, and wrote a face closed by ``vadv_weno5``'s ``pdefweno=1``
#:   gate as ``rrw*0.0`` (``-0`` for ``rrw < 0``) where CM1 stores ``0.0`` --
#:   both differences in the sign of a zero only;
#: * the six pdef candidate/correction kernels formed ``s*(d*rdx)`` where CM1's
#:   ``pdef.F:68,200`` (and the y/z twins) evaluate ``(s*d)*rdx`` left to
#:   right, which moves interior words.  Those receipts compared the CUDA
#:   kernels with :func:`cpu_pdef_direction_inplace` as it then was, so they
#:   describe the old association, not the live one;
#: * the ``ratio1`` neighbour-row bounds of ``cm1_rk_pdef_correct_x_v1`` and
#:   ``cm1_rk_pdef_correct_y_v1``: the pre-fix bounds admitted a row's first or
#:   last cell as a repair center and so read ``dum`` one cell (x) or one row
#:   (y) outside that row -- before the start of the allocation at the domain's
#:   first row of the lowest level, which faulted a 600x600x75 open-lateral run
#:   with CUDA_ERROR_ILLEGAL_ADDRESS.  The affected cells are i or j = 1 and
#:   n-2, which ``cm1_rk_pdef_total_v1`` never writes (it writes 3 <= i,j <
#:   n-3), so this one alone leaves the receipts' interior comparisons as they
#:   would be under the live bounds.
#:
#: ``tools/oracle/cm1_pdef_native.right_grouped_pdef_source``, then
#: ``tools/oracle/cm1_native_weno5_differential.pre_fix_spelling``, then
#: ``tests/test_cuda_scalar_moist_rk_pdef_bounds._prefix_source`` rebuilds it.
CUDA_SCALAR_MOIST_RK_EVIDENCE_SOURCE_SHA256 = (
    "ed55eb171393408c5844cdc03797c8e69cb4d2b3d077ce91e05d1de396ff5ef6"
)


def _source_pin_is_valid() -> bool:
    return (
        hashlib.sha256(CUDA_SCALAR_MOIST_RK_SOURCE_BYTES).hexdigest()
        == CUDA_SCALAR_MOIST_RK_SOURCE_SHA256
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
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledScalarMoistRKKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_scalar_moist_rk() -> CompiledScalarMoistRKKernels:
    if not _source_pin_is_valid():
        raise ValueError("scalar/moist RK CUDA source hash differs from the sealed pin")
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
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_SCALAR_MOIST_RK_SOURCE, "cm1_scalar_moist_rk_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_SCALAR_MOIST_RK_SOURCE,
        "cm1_scalar_moist_rk_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_SCALAR_MOIST_RK_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks scalar/moist RK entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled scalar/moist RK image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_SCALAR_MOIST_RK_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledScalarMoistRKKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


def _require_cuda_f32(
    cupy: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool = True
) -> Any:
    if not isinstance(value, cupy.ndarray):
        raise TypeError(f"{name} must be a cupy.ndarray")
    if value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


@dataclass(slots=True)
class CudaScalarMoistRKState:
    ni: int
    nj: int
    nk: int
    th0: Any
    theta_a: Any
    theta_current: Any
    moisture_a: Any
    moisture_current: Any
    theta_held: Any
    moisture_held: Any
    theta_padded: Any
    moisture_padded: Any
    fluxes: dict[str, Any]
    advection: Any
    advx: Any
    advy: Any
    advz: Any
    pdef_candidate: Any
    ready_event: Any
    stream: Any
    device_id: int
    next_nrk: int
    completed: bool
    #: CM1's ``qmag`` for the bound ``ptype`` (``src/param.F``), one entry per
    #: transported scalar.  Defaulted to the six-mass table so every existing
    #: constructor keeps the axis it already had.
    qmag: tuple[float, ...] = CM1_QMAG_F32
    #: CM1's ``qname`` for the same rows, in the same order (the component
    #: profile's ``names``); receipt-only, and defaulted with ``qmag``.
    names: tuple[str, ...] = CM1_SCALAR_MOIST_SPECIES


def create_cuda_scalar_moist_rk_state(
    carriers: Mapping[str, Any],
    held: Mapping[str, Any],
    th0: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any | None = None,
    qmag: tuple[float, ...] = CM1_QMAG_F32,
    names: tuple[str, ...] = CM1_SCALAR_MOIST_SPECIES,
) -> CudaScalarMoistRKState:
    """Bind live device carriers and allocate reusable stage scratch."""

    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 10:
            raise ValueError(f"{name} must be an integer >= 10")
    if not qmag or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0
        for value in qmag
    ):
        raise ValueError("qmag must be CM1's positive per-scalar magnitude table")
    if len(names) != len(qmag):
        raise ValueError(
            f"names {tuple(names)!r} must give CM1's qname for each of the "
            f"{len(qmag)} rows qmag describes"
        )
    cupy = importlib.import_module("cupy")
    physical = (nk, nj, ni)
    moisture = (len(qmag), nk, nj, ni)
    th0 = _require_cuda_f32(cupy, th0, "th0", (nk,), writable=False)
    theta_a = _require_cuda_f32(cupy, carriers["tha"], "tha", physical, writable=False)
    theta_current = _require_cuda_f32(cupy, carriers["th3d"], "th3d", physical)
    moisture_a = _require_cuda_f32(cupy, carriers["qa"], "qa", moisture, writable=False)
    moisture_current = _require_cuda_f32(cupy, carriers["q3d"], "q3d", moisture)
    theta_held = _require_cuda_f32(
        cupy, held["thten1"], "thten1", physical, writable=False
    )
    moisture_held = _require_cuda_f32(
        cupy, held["qten"], "qten", moisture, writable=False
    )
    bound = (
        th0,
        theta_a,
        theta_current,
        moisture_a,
        moisture_current,
        theta_held,
        moisture_held,
    )
    if len({int(value.data.ptr) for value in bound}) != len(bound):
        raise ValueError("live scalar/moisture RK carriers must not alias")
    ng = CM1_OPEN_NGXY
    theta_padded = cupy.zeros((nk + 2, nj + 2 * ng, ni + 2 * ng), dtype=cupy.float32)
    moisture_padded = cupy.zeros_like(theta_padded)
    shapes = {
        "x": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "y": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "z": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    fluxes = {
        name: cupy.zeros(shape, dtype=cupy.float32) for name, shape in shapes.items()
    }
    advection = cupy.zeros(physical, dtype=cupy.float32)
    advx = cupy.zeros_like(advection)
    advy = cupy.zeros_like(advection)
    advz = cupy.zeros_like(advection)
    pdef_candidate = cupy.zeros_like(advection)
    scratch = (
        theta_padded,
        moisture_padded,
        *fluxes.values(),
        advection,
        advx,
        advy,
        advz,
        pdef_candidate,
    )
    pointers = [int(value.data.ptr) for value in (*bound, *scratch)]
    if len(pointers) != len(set(pointers)):
        raise ValueError("scalar/moisture RK state and scratch must not alias")
    stream = stream or cupy.cuda.get_current_stream()
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return CudaScalarMoistRKState(
        ni,
        nj,
        nk,
        th0,
        theta_a,
        theta_current,
        moisture_a,
        moisture_current,
        theta_held,
        moisture_held,
        theta_padded,
        moisture_padded,
        fluxes,
        advection,
        advx,
        advy,
        advz,
        pdef_candidate,
        event,
        stream,
        int(cupy.cuda.Device().id),
        1,
        False,
        tuple(float(value) for value in qmag),
        tuple(str(name) for name in names),
    )


def begin_cuda_scalar_moist_rk_step(state: CudaScalarMoistRKState) -> None:
    """Reset only the cadence guard after the caller advances all A/held fields."""

    if not isinstance(state, CudaScalarMoistRKState):
        raise TypeError("state must be a CudaScalarMoistRKState")
    if not state.completed or state.next_nrk != 4:
        raise ValueError("cannot begin a new large step before stage 3 completes")
    state.next_nrk = 1
    state.completed = False


@dataclass(frozen=True, slots=True)
class CudaScalarMoistRKStageLaunch:
    state: CudaScalarMoistRKState
    ready_event: Any
    compiled: CompiledScalarMoistRKKernels
    advection_compiled: CompiledAdvection5Kernels
    receipt: dict[str, Any]


def _fill_scalar_open_halos(
    value: Any,
    *,
    state: CudaScalarMoistRKState,
    stream: Any,
    open_compiled: CompiledOpenBoundaryKernels,
) -> None:
    cupy = importlib.import_module("cupy")
    words = int(value.size)
    grid = (
        (words + CUDA_SCALAR_MOIST_RK_THREADS - 1) // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    threads = (CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
    u32 = cupy.uint32
    open_compiled.kernels[5](
        grid,
        threads,
        (
            value,
            u32(state.ni),
            u32(state.nj),
            u32(state.nk),
            u32(CM1_OPEN_NGXY),
            u32(0),
        ),
        stream=stream,
    )
    open_compiled.kernels[6](
        grid,
        threads,
        (
            value,
            u32(state.ni),
            u32(state.nj),
            u32(state.nk),
            u32(CM1_OPEN_NGXY),
            u32(0),
        ),
        stream=stream,
    )


def _stretched_pdef_source_sha256() -> str:
    from .cuda_gm1_child_stretched_scalar import (
        GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256,
    )

    return GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256


def launch_cuda_scalar_moist_rk_stage(
    state: CudaScalarMoistRKState,
    mass_flux: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    theta_divergence_term: Any,
    *,
    nrk: int,
    dt: float,
    dx: float,
    dy: float,
    dz: float | None = None,
    vertical: Any | None = None,
    stream: Any | None = None,
    compiled: CompiledScalarMoistRKKernels | None = None,
    advection_compiled: CompiledAdvection5Kernels | None = None,
    open_compiled: CompiledOpenBoundaryKernels | None = None,
) -> CudaScalarMoistRKStageLaunch:
    """Enqueue one exact scalar/moisture large-stage slice.

    The caller must invoke stages in order and must supply the mass flux and
    ``theta_divergence_term=t11*thterm`` belonging to the same stage.  The
    function performs no synchronization or host numerical work.

    The vertical coordinate reaches this stage one of two ways and never both.
    ``dz`` is ``param.F:7789-7811``'s uniform column, for which every layer is
    ``dz`` deep and ``adv.F:562-572``'s ``rdz*mh`` collapses to ``1/dz``.
    ``vertical`` is a
    :class:`~cm1core.cuda_gm1_child_stretched_metrics.Gm1ChildStretchedMetrics`
    carrying this run's own ``rdsf``, which is what ``adv.F:551`` and
    ``adv.F:710`` actually multiply by; the stage then drives the stretched
    advection and pdef families instead.  Passing ``dz`` alongside it is
    refused rather than ignored: on a stretched column ``dz`` is CM1's metric
    *reference* (``param.F:8170``), not any layer's depth, and a scalar
    divided by it would be advected at the wrong rate in every layer whose
    thickness is not ``dz``.

    No runtime path passes ``vertical`` today: the Cartesian lane
    (:mod:`cm1core.cuda_cartesian_dynamics`) drives a stretched column through
    the composer's ``_launch_cuda_gm1_child_stretched_scalar_stage``, and every
    other caller (``cuda_gm1_forecast``, ``cuda_gm1_child_large_step``, the
    ``tools/`` runners) passes ``dz``.  The branch is kept word-identical to the
    composer stage (``tools/oracle/stretched_scalar_species_identity.py``,
    ``tests/test_stretched_scalar_species_generic.py``) so the keyword does not
    silently advect a bound coordinate's moisture rows by a uniform ``dz``.
    """

    if (dz is None) == (vertical is None):
        raise ValueError(
            "the scalar/moist stage takes exactly one vertical coordinate: "
            "dz for param.F:7789-7811's uniform column, or vertical for the "
            "generated one; dz on a stretched column is the metric reference "
            "of param.F:8170, not a layer depth"
        )
    if not isinstance(state, CudaScalarMoistRKState):
        raise TypeError("state must be a CudaScalarMoistRKState")
    if nrk != state.next_nrk or state.completed:
        raise ValueError(
            f"scalar/moist RK cadence requires stage {state.next_nrk}, received {nrk}"
        )
    numq = scalar_moist_rk_numq(state)
    dttmp = scalar_moist_rk_stage_timestep_f32(dt, nrk)
    spacings = (("dx", dx), ("dy", dy)) if dz is None else (("dx", dx), ("dy", dy), ("dz", dz))
    for name, value in spacings:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.Device().id) != state.device_id:
        raise ValueError("scalar/moist RK state is bound to another CUDA device")
    stream = stream or state.stream
    if int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("scalar/moist RK state must remain on its bound CUDA stream")
    physical = (state.nk, state.nj, state.ni)
    theta_divergence_term = _require_cuda_f32(
        cupy,
        theta_divergence_term,
        "theta_divergence_term",
        physical,
        writable=False,
    )
    ng = CM1_OPEN_NGXY
    rru = _require_cuda_f32(
        cupy,
        mass_flux["rru"],
        "rru",
        (state.nk + 2, state.nj + 2 * ng, state.ni + 1 + 2 * ng),
        writable=False,
    )
    rrv = _require_cuda_f32(
        cupy,
        mass_flux["rrv"],
        "rrv",
        (state.nk + 2, state.nj + 1 + 2 * ng, state.ni + 2 * ng),
        writable=False,
    )
    rrw = _require_cuda_f32(
        cupy,
        mass_flux["rrw"],
        "rrw",
        (state.nk + 3, state.nj + 2 * ng, state.ni + 2 * ng),
        writable=False,
    )
    divx = _require_cuda_f32(cupy, divx, "divx", physical, writable=False)
    shapes = terrain_static_shapes(ni=state.ni, nj=state.nj, nk=state.nk)
    required_metrics = ("c1", "c2", "rho0_3d", "rr0", "gz", "rgz")
    from .cuda_immutable_metrics import immutable_metric_subset, require_cuda_argument_aliases
    metric = immutable_metric_subset(metrics, {
        name: _require_cuda_f32(cupy, metrics[name], name, shapes[name], writable=False)
        for name in required_metrics
    })
    stage_values = (
            theta_divergence_term,
            rru,
            rrv,
            rrw,
            divx,
            state.advection,
            state.advx,
            state.advy,
            state.advz,
            state.pdef_candidate,
            *state.fluxes.values(),
    )
    require_cuda_argument_aliases(
        {**{f"state.{index}": value for index, value in enumerate(stage_values)},
         **{f"metric.{name}": value for name, value in metric.items()}},
        immutable_metrics=metric,
        message="stage inputs, reusable scratch and unproved metrics must not alias",
    )
    if vertical is None:
        compiled = compiled or compile_cuda_scalar_moist_rk()
        advection_compiled = advection_compiled or compile_cuda_advection5()
    else:
        from .cuda_gm1_child_stretched_scalar import (
            compile_cuda_gm1_child_stretched_scalar_advection,
            compile_cuda_gm1_child_stretched_scalar_pdef,
            launch_cuda_gm1_child_stretched_pdef_moisture,
            launch_cuda_gm1_child_stretched_scalar_advection5,
        )

        compiled = compiled or compile_cuda_gm1_child_stretched_scalar_pdef()
        advection_compiled = (
            advection_compiled or compile_cuda_gm1_child_stretched_scalar_advection()
        )
    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    if not (
        compiled.target_arch
        == advection_compiled.target_arch
        == open_compiled.target_arch
    ):
        raise ValueError(
            "scalar/moist, advection, and boundary CUDA images target different architectures"
        )
    if vertical is not None and vertical.target_arch != compiled.target_arch:
        raise ValueError(
            "scalar/moist and vertical CUDA images target different architectures"
        )
    words = state.ni * state.nj * state.nk
    horizontal_words = state.nk * (state.nj + 1) * (state.ni + 1)
    vertical_words = (state.nk - 1) * state.nj * state.ni
    threads = (CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
    grid = (
        (words + CUDA_SCALAR_MOIST_RK_THREADS - 1) // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    grid_h = (
        (horizontal_words + CUDA_SCALAR_MOIST_RK_THREADS - 1)
        // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    grid_v = (
        (vertical_words + CUDA_SCALAR_MOIST_RK_THREADS - 1)
        // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    u32 = cupy.uint32
    # adv.F:214-340 and pdef.F:49-166 read the lateral family the halo fills
    # below apply; the sides come from those fills' own binding.
    from .cuda_boundary_families import bound_lateral_sides

    lateral_sides = bound_lateral_sides(open_compiled)
    obc = u32(scalar_open_boundary_code(lateral_sides))

    def _advect(padded: Any, weps: float) -> None:
        """adv.F's fifth-order scalar slice on this run's own column."""

        if vertical is None:
            launch_cuda_scalar_advection5(
                padded,
                {"rru": rru, "rrv": rrv, "rrw": rrw},
                metric,
                divx,
                state.advection,
                state.fluxes,
                ni=state.ni,
                nj=state.nj,
                nk=state.nk,
                nrk=nrk,
                dx=dx,
                dy=dy,
                dz=dz,
                weps=weps,
                stream=stream,
                compiled=advection_compiled,
                lateral_sides=lateral_sides,
            )
            return
        launch_cuda_gm1_child_stretched_scalar_advection5(
            padded,
            {"rru": rru, "rrv": rrv, "rrw": rrw},
            metric,
            divx,
            state.advection,
            state.fluxes,
            vertical,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            nrk=nrk,
            dx=dx,
            dy=dy,
            weps=weps,
            stream=stream,
            compiled=advection_compiled,
            lateral_sides=lateral_sides,
        )

    with stream:
        stream.wait_event(state.ready_event)
        compiled.kernels[0](
            grid,
            threads,
            (
                state.th0,
                state.theta_current,
                state.theta_padded,
                u32(state.ni),
                u32(state.nj),
                u32(state.nk),
                u32(ng),
            ),
            stream=stream,
        )
        _fill_scalar_open_halos(
            state.theta_padded,
            state=state,
            stream=stream,
            open_compiled=open_compiled,
        )
        state.advection.fill(cupy.float32(0.0))
        _advect(state.theta_padded, theta_weps_f64())
        compiled.kernels[2](
            grid,
            threads,
            (
                state.theta_a,
                state.theta_held,
                theta_divergence_term,
                state.advection,
                state.theta_current,
                cupy.uint64(words),
                cupy.float32(dttmp),
            ),
            stream=stream,
        )
        pdef = nrk == CM1_NRKMAX
        final_advection = final_stage_moisture_advection_mode()
        qmag = state.qmag
        for species in range(numq):
            current = state.moisture_current[species]
            compiled.kernels[1](
                grid,
                threads,
                (
                    current,
                    state.moisture_padded,
                    u32(state.ni),
                    u32(state.nj),
                    u32(state.nk),
                    u32(ng),
                ),
                stream=stream,
            )
            _fill_scalar_open_halos(
                state.moisture_padded,
                state=state,
                stream=stream,
                open_compiled=open_compiled,
            )
            state.advection.fill(cupy.float32(0.0))
            weps = moisture_weps_f64(species, qmag)
            if not pdef or final_advection == "launch":
                # On the final stage adv.F:421-445 (pdef=1, solve2.F:1451-1455)
                # forms the row's advection from the pdef chain alone.  The
                # plain launch wrote only words that chain rewrites before any
                # read -- the same fx/fy faces, fz(k=2..nk), and the
                # 3<=i<ni-3, 3<=j<nj-3 tendency interior its total kernel
                # assigns with "=" -- so by default it is not enqueued there.
                _advect(state.moisture_padded, weps)
            if pdef and vertical is not None:
                # 7d93095 routed the moisture rows through the profile-bound
                # families; the 11f3ac3 merge kept the uniform calls here, so a
                # bound coordinate advected theta by rdsf(k) and then refused
                # (dz=None) at the first moisture row.
                launch_cuda_gm1_child_stretched_pdef_moisture(
                    state.moisture_a[species],
                    state.moisture_padded,
                    {"rru": rru, "rrv": rrv, "rrw": rrw},
                    metric,
                    divx,
                    state.fluxes,
                    state.advx,
                    state.advy,
                    state.advz,
                    state.pdef_candidate,
                    state.advection,
                    vertical,
                    ni=state.ni,
                    nj=state.nj,
                    nk=state.nk,
                    dttmp=float(dttmp),
                    dx=dx,
                    dy=dy,
                    weps=weps,
                    species=species,
                    numq=numq,
                    stream=stream,
                    compiled=compiled,
                    lateral_sides=lateral_sides,
                )
            elif pdef:
                compiled.kernels[4](
                    grid_h,
                    threads,
                    (
                        state.moisture_padded,
                        rru,
                        rrv,
                        state.fluxes["x"],
                        state.fluxes["y"],
                        u32(state.ni),
                        u32(state.nj),
                        u32(state.nk),
                        u32(ng),
                        cupy.float64(weps),
                    ),
                    stream=stream,
                )
                compiled.kernels[5](
                    grid_v,
                    threads,
                    (
                        state.moisture_padded,
                        rrw,
                        metric["c1"],
                        metric["c2"],
                        state.fluxes["z"],
                        u32(state.ni),
                        u32(state.nj),
                        u32(state.nk),
                        u32(ng),
                        u32(0 if species == 0 else 1),
                        cupy.float64(weps),
                    ),
                    stream=stream,
                )
                compiled.kernels[6](
                    grid,
                    threads,
                    (
                        state.fluxes["x"],
                        state.fluxes["y"],
                        state.fluxes["z"],
                        state.advx,
                        state.advy,
                        state.advz,
                        u32(state.ni),
                        u32(state.nj),
                        u32(state.nk),
                        u32(ng),
                        cupy.float32(dx),
                        cupy.float32(dy),
                        cupy.float32(dz),
                        state.moisture_padded,
                        rru,
                        rrv,
                        obc,
                    ),
                    stream=stream,
                )
                for candidate_kernel, correction_kernel, flux, spacing, direction in (
                    (7, 8, rru, dx, state.advx),
                    (9, 10, rrv, dy, state.advy),
                    (11, 12, rrw, dz, state.advz),
                ):
                    compiled.kernels[candidate_kernel](
                        grid,
                        threads,
                        (
                            state.moisture_a[species],
                            state.moisture_padded,
                            metric["rho0_3d"],
                            metric["gz"],
                            flux,
                            direction,
                            state.pdef_candidate,
                            u32(state.ni),
                            u32(state.nj),
                            u32(state.nk),
                            u32(ng),
                            cupy.float32(dttmp),
                            cupy.float32(spacing),
                            *(() if candidate_kernel == 11 else (obc,)),
                        ),
                        stream=stream,
                    )
                    compiled.kernels[correction_kernel](
                        grid,
                        threads,
                        (
                            state.moisture_a[species],
                            state.moisture_padded,
                            metric["rho0_3d"],
                            metric["gz"],
                            metric["rgz"],
                            flux,
                            state.pdef_candidate,
                            direction,
                            u32(state.ni),
                            u32(state.nj),
                            u32(state.nk),
                            u32(ng),
                            cupy.float32(dttmp),
                            cupy.float32(spacing),
                            *(() if correction_kernel == 12 else (obc,)),
                        ),
                        stream=stream,
                    )
                compiled.kernels[13](
                    grid,
                    threads,
                    (
                        state.moisture_padded,
                        state.advx,
                        state.advy,
                        state.advz,
                        divx,
                        metric["rr0"],
                        metric["gz"],
                        state.advection,
                        u32(state.ni),
                        u32(state.nj),
                        u32(state.nk),
                        u32(ng),
                        obc,
                    ),
                    stream=stream,
                )
            compiled.kernels[3](
                grid,
                threads,
                (
                    state.moisture_a[species],
                    state.moisture_held[species],
                    state.advection,
                    current,
                    cupy.uint64(words),
                    cupy.float32(dttmp),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.ready_event = event
    state.next_nrk = nrk + 1
    state.completed = nrk == CM1_NRKMAX
    return CudaScalarMoistRKStageLaunch(
        state,
        event,
        compiled,
        advection_compiled,
        {
            "schema": CUDA_SCALAR_MOIST_RK_SCHEMA,
            "claim_limit": CUDA_SCALAR_MOIST_RK_CLAIM_LIMIT,
            "domain": {"ni": state.ni, "nj": state.nj, "nk": state.nk},
            "stage": {
                **scalar_moist_rk_cadence(nrk),
                "dt_f32": float(cupy.float32(dt)),
                "dttmp_f32": float(dttmp),
            },
            # solve2.F:1435 advances all numq rows, named by the bound
            # component's qname (param.F), e.g. nci,ncs,ncr,ncg for ptype=5.
            "species": list(state.names),
            "numq": numq,
            "moisture_advection": final_stage_moisture_advection_record(
                nrk, final_advection
            ),
            "moisture_weps_f64": [
                moisture_weps_f64(i, state.qmag) for i in range(numq)
            ],
            "theta_weps_f64": theta_weps_f64(),
            "interior_bounds_one_based": scalar_moist_rk_interior_bounds(
                state.ni, state.nj, positive_definite=nrk == CM1_NRKMAX
            ),
            "source_anchors": dict(SOURCE_ANCHORS),
            # Which column this stage actually divided by.  A reader that
            # cannot tell a uniform dz from a generated rdsf cannot tell a
            # correct stretched run from a silently wrong one.
            "vertical_coordinate": (
                {"stretch_z": 0, "uniform_dz_m": float(dz)}
                if vertical is None
                else {
                    "stretch_z": int(
                        (vertical.identity.get("vertical_coordinate") or {}).get(
                            "stretch_z", 1
                        )
                    ),
                    "identity_schema": vertical.identity.get("schema"),
                    "ztop_m": float(vertical.ztop_f32),
                    "dz_reference_m": float(vertical.dz_reference_f32),
                    "sha256": (
                        vertical.identity.get("vertical_coordinate") or {}
                    ).get("sha256"),
                    "rdsf_bound": True,
                }
            ),
            "kernel": {
                "abi": CUDA_SCALAR_MOIST_RK_ABI,
                "source_sha256": (
                    CUDA_SCALAR_MOIST_RK_SOURCE_SHA256
                    if vertical is None
                    else _stretched_pdef_source_sha256()
                ),
                "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
                "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
                "target_arch": compiled.target_arch,
                "ptx_options": list(compiled.ptx_options),
                "cubin_options": list(compiled.cubin_options),
            },
            "dependency_kernels": {
                "advection5_cubin_sha256": hashlib.sha256(
                    advection_compiled.cubin
                ).hexdigest(),
                "open_boundary_cubin_sha256": hashlib.sha256(
                    open_compiled.cubin
                ).hexdigest(),
            },
            "runtime": dict(compiled.runtime_identity),
        },
    )


__all__ = [
    "CM1_EPSILON_F32",
    "CM1_QMAG_F32",
    "CM1_SCALAR_MOIST_SPECIES",
    "CM1_SMEPS_F32",
    "CM1_TH0R_F32",
    "CUDA_SCALAR_MOIST_RK_ABI",
    "CUDA_SCALAR_MOIST_RK_CLAIM_LIMIT",
    "CUDA_SCALAR_MOIST_RK_KERNEL_NAMES",
    "CUDA_SCALAR_MOIST_RK_SCHEMA",
    "CUDA_SCALAR_MOIST_RK_SOURCE",
    "CUDA_SCALAR_MOIST_RK_SOURCE_SHA256",
    "CUDA_SCALAR_MOIST_RK_EVIDENCE_SOURCE_SHA256",
    "FINAL_STAGE_MOISTURE_ADVECTION_ENVIRONMENT",
    "FINAL_STAGE_MOISTURE_ADVECTION_MODES",
    "CompiledScalarMoistRKKernels",
    "CudaScalarMoistRKStageLaunch",
    "CudaScalarMoistRKState",
    "SOURCE_ANCHORS",
    "begin_cuda_scalar_moist_rk_step",
    "compile_cuda_scalar_moist_rk",
    "cpu_compose_scalar_moist_rk_stage_inplace",
    "cpu_pdef_direction_inplace",
    "cpu_theta_transport_scalar_inplace",
    "create_cuda_scalar_moist_rk_state",
    "final_stage_moisture_advection_mode",
    "final_stage_moisture_advection_record",
    "launch_cuda_scalar_moist_rk_stage",
    "moisture_weps_f64",
    "scalar_moist_rk_cadence",
    "scalar_moist_rk_numq",
    "scalar_moist_rk_interior_bounds",
    "scalar_moist_rk_stage_timestep_f32",
    "theta_weps_f64",
]
