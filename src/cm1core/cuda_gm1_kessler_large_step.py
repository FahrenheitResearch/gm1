"""Persistent final-large-step GM1 Kessler transition.

GM1 is the product name; CM1 r21.1 is the pinned mathematical authority.
This adapter composes the admitted Kessler driver at its native once-per-large-
step location between the final RK solve and ``solve3``.  It also preserves
the equation-set-2 held-tendency lifecycle and the persistent rain, rate, and
budget ledgers required across steps and restarts.

This module accepts only an already Kessler-initialized three-species state.
It does not convert a six-species WSM6 product and does not invent a mapping
for frozen species.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_kessler_driver import (
    CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256,
    CompiledGm1KesslerDriverKernels,
    Gm1KesslerDriverSelection,
    Gm1KesslerDriverWorkspace,
    compile_cuda_gm1_kessler_driver,
    cpu_gm1_kessler_driver_inplace,
    create_cuda_gm1_kessler_driver_workspace,
    launch_cuda_gm1_kessler_driver,
)
from cm1core.cuda_gm1_kessler_efall import (
    CompiledGm1KesslerEfallKernels,
    compile_cuda_gm1_kessler_efall,
)
from cm1core.cuda_gm1_kessler_local import CompiledGm1KesslerLocalKernels
from cm1core.errors import refuse


GM1_KESSLER_LARGE_STEP_SEMANTIC_ID = "kessler-ptype1"
GM1_KESSLER_LARGE_STEP_SCHEMA = "gm1.cuda-kessler-large-step/v1"
GM1_KESSLER_LARGE_STEP_RESTART_SCHEMA = "gm1.kessler-large-step-restart/v1"
GM1_KESSLER_LARGE_STEP_CLAIM_LIMIT = (
    "Device-resident once-per-large-step lifecycle for an already ptype=1 "
    "initialized GM1 state: equation-set-2 final-RK held-rate remove/snapshot, "
    "pre-physics EOS, optional dry-mass hook, complete admitted Kessler operator, "
    "equation-set-2 held-rate recapture, and persistent rain/prate/asq/qbudget "
    "restart state. Conversion "
    "from WSM6, real ptype=1 initialization, reflectivity, forecast skill, and "
    "tornado skill are not claimed."
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
    "src/solve1.F": "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    "src/solve3.F": "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393",
    "src/mp_driver.F": "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134",
    "src/kessler.F": "f232f840e425d8c9abd28841e0a46a06e0dbf641460224fa61b6a82e5a650472",
}
SOURCE_ANCHORS = {
    "solve1_held_consume": "src/solve1.F:1735-1753",
    "final_rk_remove_snapshot": "src/solve2.F:1561-1579",
    "pre_physics_eos": "src/solve2.F:1582-1595",
    "dry_mass_hook": "src/solve2.F:1742-1796,2093-2145",
    "microphysics_cadence": "src/cm1.F:3405-3432",
    "ptype1_driver": "src/mp_driver.F:99-149,190-232",
    "held_rate_recapture": "src/solve3.F:319-337",
    "native_initialization": "src/cm1.F:1689-1692; src/init3d.F:279-319; src/param.F:3169-3220",
}

_STATE_HELD = (
    ("pp3d", "qpten"),
    ("th3d", "qtten"),
    ("qv", "qvten"),
    ("qc", "qcten"),
)
_RESTART_NAMES = (
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "persistent.rain",
    "persistent.prate",
    "persistent.asq",
    "persistent.qbudget",
)


def _binary32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


# CM1 constants.F default-real statement order.  The ratio is deliberately
# rounded after the default-real operands are rounded, matching ``cpdrd=cp/rd``.
_CM1_P00_F32 = _binary32(100000.0)
_CM1_RD_F32 = _binary32(287.04)
_CM1_RV_F32 = _binary32(461.5)
_CM1_CP_F32 = _binary32(1005.7)
_CM1_CPDRD_F32 = _binary32(_CM1_CP_F32 / _CM1_RD_F32)


@dataclass(slots=True)
class Gm1KesslerHeldCarriers:
    qpten: Any
    qtten: Any
    qvten: Any
    qcten: Any


@dataclass(frozen=True, slots=True)
class Gm1KesslerLargeStepRestart:
    schema: str
    completed_transitions: int
    arrays: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Gm1KesslerLargeStepResult:
    ready_event: Any
    completed_transitions: int
    receipt: Mapping[str, Any]
    restart: Gm1KesslerLargeStepRestart
    driver_launch: Any


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _strict_geometry(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def _restart_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    mass = (nk, nj, ni)
    return {
        "held.qpten": mass,
        "held.qtten": mass,
        "held.qvten": mass,
        "held.qcten": mass,
        "persistent.rain": (1, nj, ni),
        "persistent.prate": (nj, ni),
        "persistent.asq": (3,),
        "persistent.qbudget": (10,),
    }


def _validate_device_array(
    cupy: Any,
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != cupy.dtype(dtype)
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            f"{name} must be C-order CUDA {cupy.dtype(dtype)} with shape {shape}"
        )


def cpu_gm1_kessler_large_step_inplace(
    arrays: Mapping[str, Any],
    held: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerDriverSelection,
    pre_component_hook: Callable[[Mapping[str, Any]], object] | None = None,
) -> dict[str, Any]:
    """Independent CPU transcription of the source-ordered composition.

    This is an operator/scheduler oracle helper, not an atmospheric product
    initializer.  ``arrays`` must already carry native ptype=1 qv/qc/qr state.
    """

    _strict_geometry(ni, nj, nk)
    dx, dy, dz, dt = (
        _positive(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt))
    )
    if not isinstance(selection, Gm1KesslerDriverSelection):
        raise TypeError("selection must be a Gm1KesslerDriverSelection")
    if pre_component_hook is not None and not callable(pre_component_hook):
        raise TypeError("pre_component_hook must be callable or None")
    np = importlib.import_module("numpy")
    mass = (nk, nj, ni)
    required = (
        "pi0", "th0", "rho", "rr", "pp3d", "th3d", "prs", "qv", "qc",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError("missing CPU Kessler large-step arrays: " + ", ".join(missing))
    held_missing = [name for _, name in _STATE_HELD if name not in held]
    if held_missing:
        raise ValueError("missing CPU Kessler held arrays: " + ", ".join(held_missing))
    for name in required:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != mass
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CPU float32 with shape {mass}")
    for _, name in _STATE_HELD:
        value = held[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != mass
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CPU float32 with shape {mass}")

    a = arrays
    dt32 = np.float32(dt)
    work = np.empty(mass, dtype=np.float32)
    if selection.equation_set == 2:
        for state_name, held_name in _STATE_HELD:
            state, rate = a[state_name], held[held_name]
            np.multiply(rate, dt32, out=work)
            np.subtract(state, work, out=state)
            np.copyto(rate, state)

    np.add(a["pi0"], a["pp3d"], out=work)
    np.power(work, np.float32(_CM1_CPDRD_F32), out=a["prs"])
    np.multiply(a["prs"], np.float32(_CM1_P00_F32), out=a["prs"])
    np.add(a["th0"], a["th3d"], out=a["rho"])
    np.multiply(a["rho"], work, out=a["rho"])
    np.maximum(a["qv"], np.float32(0.0), out=work)
    np.multiply(work, np.float32(_CM1_RV_F32), out=work)
    np.add(work, np.float32(_CM1_RD_F32), out=work)
    np.multiply(a["rho"], work, out=a["rho"])
    np.divide(a["prs"], a["rho"], out=a["rho"])

    hook_result = None if pre_component_hook is None else pre_component_hook(held)
    np.copyto(a["rr"], a["rho"])
    driver = cpu_gm1_kessler_driver_inplace(
        a,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        selection=selection,
    )
    if selection.equation_set == 2:
        for state_name, held_name in _STATE_HELD:
            state, snapshot = a[state_name], held[held_name]
            np.subtract(state, snapshot, out=work)
            np.divide(work, dt32, out=snapshot)
    return {
        "driver": driver,
        "pre_component_hook_result": hook_result,
        "equation_set_2_held_lifecycle": selection.equation_set == 2,
        "source_order": (
            "solve2-held-remove-snapshot",
            "solve2-pre-physics-eos",
            "solve2-dry-mass-hook",
            "cm1-mp_driver-ptype1",
            "solve3-held-recapture",
        ),
    }


class CudaGm1KesslerLargeStepTransition:
    """Persistent source-ordered Kessler transition for one CUDA state."""

    def __init__(
        self,
        *,
        arrays: dict[str, Any],
        carriers: Gm1KesslerHeldCarriers,
        ni: int,
        nj: int,
        nk: int,
        dx: float,
        dy: float,
        dz: float,
        dt: float,
        selection: Gm1KesslerDriverSelection,
        stream: Any,
        compiled: CompiledGm1KesslerDriverKernels,
        local_compiled: CompiledGm1KesslerLocalKernels,
        efall_compiled: CompiledGm1KesslerEfallKernels | None,
        workspace: Gm1KesslerDriverWorkspace,
        work: Any,
        completed_transitions: int,
    ) -> None:
        self.arrays = arrays
        self.carriers = carriers
        self.ni, self.nj, self.nk = ni, nj, nk
        self.dx, self.dy, self.dz, self.dt = dx, dy, dz, dt
        self.selection = selection
        self.stream = stream
        self.compiled = compiled
        self.local_compiled = local_compiled
        self.efall_compiled = efall_compiled
        self.workspace = workspace
        self.work = work
        self.completed_transitions = completed_transitions
        self._last_receipt: dict[str, Any] | None = None

    @property
    def target_arches(self) -> set[str]:
        result = {self.compiled.target_arch, self.local_compiled.target_arch}
        if self.efall_compiled is not None:
            result.add(self.efall_compiled.target_arch)
        return result

    def _restart_arrays(self) -> dict[str, Any]:
        return {
            "held.qpten": self.carriers.qpten,
            "held.qtten": self.carriers.qtten,
            "held.qvten": self.carriers.qvten,
            "held.qcten": self.carriers.qcten,
            "persistent.rain": self.arrays["rain"],
            "persistent.prate": self.arrays["prate"],
            "persistent.asq": self.arrays["asq"],
            "persistent.qbudget": self.arrays["qbudget"],
        }

    def capture_restart(self) -> Gm1KesslerLargeStepRestart:
        values = self._restart_arrays()
        if tuple(values) != _RESTART_NAMES:
            raise RuntimeError("GM1 Kessler restart inventory drifted")
        return Gm1KesslerLargeStepRestart(
            GM1_KESSLER_LARGE_STEP_RESTART_SCHEMA,
            self.completed_transitions,
            MappingProxyType(values),
        )

    def clone_restart(self) -> Gm1KesslerLargeStepRestart:
        cupy = importlib.import_module("cupy")
        with self.stream:
            values = {
                name: cupy.array(value, dtype=value.dtype, order="C", copy=True)
                for name, value in self._restart_arrays().items()
            }
        return Gm1KesslerLargeStepRestart(
            GM1_KESSLER_LARGE_STEP_RESTART_SCHEMA,
            self.completed_transitions,
            MappingProxyType(values),
        )

    def receipt_dict(self) -> dict[str, Any]:
        if self._last_receipt is None:
            raise RuntimeError("GM1 Kessler large-step transition has not completed")
        return json.loads(json.dumps(self._last_receipt, allow_nan=False, sort_keys=True))

    def apply(
        self,
        *,
        final_rk: bool = True,
        refl_10cm_due: bool = False,
        pre_component_hook: Callable[[object], object] | None = None,
    ) -> Gm1KesslerLargeStepResult:
        if type(final_rk) is not bool or not final_rk:
            refuse(
                code="GM1-KESSLER-LARGE-STEP-CADENCE",
                gate="physics-seam",
                option="final_rk",
                value=final_rk,
                reason="CM1 r21.1 calls mp_driver once after the completed RK solve",
                remedy="invoke the transition only after final RK stage 3",
            )
        if type(refl_10cm_due) is not bool:
            raise TypeError("refl_10cm_due must be boolean")
        if refl_10cm_due:
            refuse(
                code="GM1-KESSLER-REFLECTIVITY-UNPORTED",
                gate="physics-seam",
                option="refl_10cm_due",
                value=True,
                reason="CM1 does not admit the selected reflectivity consumer for ptype=1",
                remedy="disable reflectivity or select an admitted microphysics consumer",
            )
        if pre_component_hook is not None and not callable(pre_component_hook):
            raise TypeError("pre_component_hook must be callable or None")
        cupy = importlib.import_module("cupy")
        a, held = self.arrays, self.carriers
        dt = cupy.float32(self.dt)
        zero = cupy.float32(0.0)
        with self.stream:
            if self.selection.equation_set == 2:
                # solve2.F:1561-1579: consume the previous held rate, then
                # snapshot the exact pre-microphysics state into that carrier.
                for state_name, held_name in _STATE_HELD:
                    state = a[state_name]
                    rate = getattr(held, held_name)
                    cupy.multiply(rate, dt, out=self.work)
                    cupy.subtract(state, self.work, out=state)
                    cupy.copyto(rate, state)

            # solve2.F:1582-1595 pre-physics pressure and moist density.
            cupy.add(a["pi0"], a["pp3d"], out=self.work)
            cupy.power(self.work, cupy.float32(_CM1_CPDRD_F32), out=a["prs"])
            cupy.multiply(a["prs"], cupy.float32(_CM1_P00_F32), out=a["prs"])
            cupy.add(a["th0"], a["th3d"], out=a["rho"])
            cupy.multiply(a["rho"], self.work, out=a["rho"])
            cupy.maximum(a["qv"], zero, out=self.work)
            cupy.multiply(self.work, cupy.float32(_CM1_RV_F32), out=self.work)
            cupy.add(self.work, cupy.float32(_CM1_RD_F32), out=self.work)
            cupy.multiply(a["rho"], self.work, out=a["rho"])
            cupy.divide(a["prs"], a["rho"], out=a["rho"])

            hook_result = (
                None
                if pre_component_hook is None
                else pre_component_hook(self.carriers)
            )

            # mp_driver.F:113-130: dum3 is the post-correction density for the
            # Cartesian budget reductions. The native Kessler dummy is named
            # rr but is not reciprocal density.
            cupy.copyto(a["rr"], a["rho"])
            driver = launch_cuda_gm1_kessler_driver(
                a,
                ni=self.ni,
                nj=self.nj,
                nk=self.nk,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                dt=self.dt,
                selection=self.selection,
                stream=self.stream,
                compiled=self.compiled,
                local_compiled=self.local_compiled,
                efall_compiled=self.efall_compiled,
                workspace=self.workspace,
            )
            self.efall_compiled = driver.efall_compiled

            if self.selection.equation_set == 2:
                # solve3.F:319-337: snapshots become next-step held rates.
                for state_name, held_name in _STATE_HELD:
                    state = a[state_name]
                    snapshot = getattr(held, held_name)
                    cupy.subtract(state, snapshot, out=self.work)
                    cupy.divide(self.work, dt, out=snapshot)
            event = cupy.cuda.Event(disable_timing=True)
            event.record(self.stream)

        self.completed_transitions += 1
        restart = self.capture_restart()
        source = Path(__file__).read_bytes()
        receipt = {
            "schema": GM1_KESSLER_LARGE_STEP_SCHEMA,
            "semantic_id": GM1_KESSLER_LARGE_STEP_SEMANTIC_ID,
            "claim_limit": GM1_KESSLER_LARGE_STEP_CLAIM_LIMIT,
            "completed_transitions": self.completed_transitions,
            "configuration": {
                "ni": self.ni,
                "nj": self.nj,
                "nk": self.nk,
                "dx_m": self.dx,
                "dy_m": self.dy,
                "dz_m": self.dz,
                "dt_s": self.dt,
                "ptype": 1,
                "equation_set": self.selection.equation_set,
                "efall": self.selection.efall,
                "final_rk": True,
            },
            "source_order": [
                (
                    "held-rate remove/snapshot"
                    if self.selection.equation_set == 2
                    else "held-rate lifecycle inactive for equation_set=1"
                ),
                "pre-physics EOS",
                "dry-mass correction hook" if pre_component_hook is not None else "dry-mass hook absent",
                "mp_driver ptype=1",
                (
                    "held-rate recapture"
                    if self.selection.equation_set == 2
                    else "held-rate recapture inactive for equation_set=1"
                ),
            ],
            "authority": {
                "math_release": "CM1 r21.1",
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
                "anchors": dict(SOURCE_ANCHORS),
            },
            "operator": driver.receipt,
            "operator_source_sha256": CUDA_GM1_KESSLER_DRIVER_SOURCE_SHA256,
            "restart": {
                "schema": restart.schema,
                "completed_transitions": restart.completed_transitions,
                "inventory": list(restart.arrays),
                "zero_copy_capture": True,
                "persistent": ["rain", "prate", "asq", "qbudget"],
                "recomputable": ["fall_velocity", "driver workspace", "EOS work"],
            },
            "synchronization": {
                "host_scientific_copy_or_sync": False,
                "stream_pointer": int(self.stream.ptr),
                "event_pointer": int(event.ptr),
            },
            "source": {
                "module": __name__,
                "bytes": len(source),
                "sha256": hashlib.sha256(source).hexdigest(),
            },
            "pre_component_hook": {
                "selected": pre_component_hook is not None,
                "result_retained_by_caller": hook_result is not None,
            },
            "claim_boundaries": {
                "native_ptype1_initialization": False,
                "wsm6_to_kessler_conversion": False,
                "real_forecast": False,
                "tornado": False,
            },
        }
        json.dumps(receipt, allow_nan=False, sort_keys=True)
        self._last_receipt = receipt
        return Gm1KesslerLargeStepResult(
            event,
            self.completed_transitions,
            MappingProxyType(self.receipt_dict()),
            restart,
            driver,
        )


def create_cuda_gm1_kessler_large_step_transition(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    selection: Gm1KesslerDriverSelection,
    stream: Any | None = None,
    compiled: CompiledGm1KesslerDriverKernels | None = None,
    local_compiled: CompiledGm1KesslerLocalKernels | None = None,
    efall_compiled: CompiledGm1KesslerEfallKernels | None = None,
    workspace: Gm1KesslerDriverWorkspace | None = None,
    restart: Gm1KesslerLargeStepRestart | None = None,
) -> CudaGm1KesslerLargeStepTransition:
    """Bind an already Kessler-initialized device state without converting it."""

    _strict_geometry(ni, nj, nk)
    dx, dy, dz, dt = (
        _positive(name, value)
        for name, value in (("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt))
    )
    if not isinstance(selection, Gm1KesslerDriverSelection):
        raise TypeError("selection must be a Gm1KesslerDriverSelection")
    cupy = importlib.import_module("cupy")
    stream = stream or cupy.cuda.get_current_stream()
    mass = (nk, nj, ni)
    base_shapes = {
        "ruh": (ni,), "rvh": (nj,), "rmh": mass, "mh": mass,
        "pi0": mass, "th0": mass, "rho": mass, "pp3d": mass,
        "th3d": mass, "prs": mass,
        "qv": mass, "qc": mass, "qr": mass,
    }
    if selection.efall == 1:
        base_shapes["mf"] = (nk + 1, nj, ni)
    missing = [name for name in base_shapes if name not in arrays]
    if missing:
        raise ValueError(
            "missing GM1 Kessler large-step arrays: " + ", ".join(missing)
        )
    bound: dict[str, Any] = {}
    for name, shape in base_shapes.items():
        value = arrays[name]
        _validate_device_array(
            cupy, value, name=name, shape=shape, dtype=cupy.float32
        )
        bound[name] = value

    completed = 0
    if restart is not None:
        if (
            not isinstance(restart, Gm1KesslerLargeStepRestart)
            or restart.schema != GM1_KESSLER_LARGE_STEP_RESTART_SCHEMA
            or type(restart.completed_transitions) is not int
            or restart.completed_transitions < 0
            or tuple(restart.arrays) != _RESTART_NAMES
        ):
            raise ValueError("GM1 Kessler large-step restart contract differs")
        restart_arrays = dict(restart.arrays)
        completed = restart.completed_transitions
    else:
        restart_arrays = {
            "held.qpten": cupy.zeros(mass, dtype=cupy.float32),
            "held.qtten": cupy.zeros(mass, dtype=cupy.float32),
            "held.qvten": cupy.zeros(mass, dtype=cupy.float32),
            "held.qcten": cupy.zeros(mass, dtype=cupy.float32),
            "persistent.rain": (
                arrays["rain"] if "rain" in arrays
                else cupy.zeros((1, nj, ni), dtype=cupy.float32)
            ),
            "persistent.prate": (
                arrays["prate"] if "prate" in arrays
                else cupy.zeros((nj, ni), dtype=cupy.float32)
            ),
            "persistent.asq": (
                arrays["asq"] if "asq" in arrays
                else cupy.zeros((3,), dtype=cupy.float64)
            ),
            "persistent.qbudget": (
                arrays["qbudget"] if "qbudget" in arrays
                else cupy.zeros((10,), dtype=cupy.float64)
            ),
        }
    for name, shape in _restart_shapes(ni, nj, nk).items():
        dtype = cupy.float64 if name in ("persistent.asq", "persistent.qbudget") else cupy.float32
        _validate_device_array(
            cupy, restart_arrays[name], name=name, shape=shape, dtype=dtype
        )
    carriers = Gm1KesslerHeldCarriers(
        restart_arrays["held.qpten"],
        restart_arrays["held.qtten"],
        restart_arrays["held.qvten"],
        restart_arrays["held.qcten"],
    )
    bound.update(
        {
            "fall_velocity": cupy.empty(mass, dtype=cupy.float32),
            "rr": cupy.empty(mass, dtype=cupy.float32),
            "rain": restart_arrays["persistent.rain"],
            "prate": restart_arrays["persistent.prate"],
            "asq": restart_arrays["persistent.asq"],
            "qbudget": restart_arrays["persistent.qbudget"],
        }
    )
    pointers = [int(value.data.ptr) for value in bound.values()]
    pointers.extend(
        int(getattr(carriers, name).data.ptr)
        for name in ("qpten", "qtten", "qvten", "qcten")
    )
    if len(set(pointers)) != len(pointers):
        raise ValueError("GM1 Kessler large-step state and restart arrays alias")
    compiled = compiled or compile_cuda_gm1_kessler_driver()
    from cm1core.cuda_gm1_kessler_local import compile_cuda_gm1_kessler_local

    local_compiled = local_compiled or compile_cuda_gm1_kessler_local()
    if selection.efall == 1:
        efall_compiled = efall_compiled or compile_cuda_gm1_kessler_efall()
    workspace = workspace or create_cuda_gm1_kessler_driver_workspace(
        ni=ni, nj=nj, nk=nk
    )
    work = cupy.empty(mass, dtype=cupy.float32)
    return CudaGm1KesslerLargeStepTransition(
        arrays=bound,
        carriers=carriers,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        selection=selection,
        stream=stream,
        compiled=compiled,
        local_compiled=local_compiled,
        efall_compiled=efall_compiled,
        workspace=workspace,
        work=work,
        completed_transitions=completed,
    )


__all__ = [
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CudaGm1KesslerLargeStepTransition",
    "GM1_KESSLER_LARGE_STEP_CLAIM_LIMIT",
    "GM1_KESSLER_LARGE_STEP_RESTART_SCHEMA",
    "GM1_KESSLER_LARGE_STEP_SCHEMA",
    "GM1_KESSLER_LARGE_STEP_SEMANTIC_ID",
    "Gm1KesslerHeldCarriers",
    "Gm1KesslerLargeStepRestart",
    "Gm1KesslerLargeStepResult",
    "SOURCE_ANCHORS",
    "cpu_gm1_kessler_large_step_inplace",
    "create_cuda_gm1_kessler_large_step_transition",
]
