"""Reusable full-domain admission wrapper for the terrain/open acoustic step.

The wrapper keeps the corrected-real initialized arena, padded acoustic state,
open-radiation histories, and compiled CUDA operators alive across a coupling
boundary.  It never silently resets histories or re-arms a consumed large-step
contract.  CUDA remains a lazy import.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_initialized_product import CudaInitializedProduct
from cm1core.cuda_initialized_product_terrain_open import (
    CudaInitializedTerrainOpenState,
    prepare_cuda_initialized_product_terrain_open,
)
from cm1core.cuda_psolver3_terrain_open import (
    CudaPsolver3TerrainOpenLaunch,
    launch_cuda_psolver3_terrain_open_three_stage,
    terrain_open_sound_shapes,
)


CUDA_TERRAIN_OPEN_FULL_STATE_SCHEMA = "gpuwm-cm1.cuda-terrain-open-full-state/v1"
FULL_NI = 300
FULL_NJ = 300
FULL_NK = 75
FULL_PRODUCT_PAYLOAD_BYTES = 1_169_829_300
DEFAULT_DEVICE_RESERVE_BYTES = 2 * 1024**3
COLD_START_HISTORY_MODE = "cold-start-zero"
COLD_START_HISTORY_SOURCE_ANCHOR = "src/cm1.F:1559-1566"
RADIATION_HISTORY_NAMES = ("radbcw", "radbce", "radbcs", "radbcn")
TENDENCY_NAMES = ("uten", "vten", "wten", "ppten")

CUDA_TERRAIN_OPEN_FULL_STATE_CLAIM_LIMIT = (
    "one reusable full-domain corrected-real CM1 r21.1 terrain/all-open "
    "psolver=3 acoustic state and source-ordered three-stage step; cold-start "
    "radiation histories are exact zero only at cm1.F:1559-1566, later steps "
    "must preserve or explicitly import histories; terrain-varying hydrostatic "
    "base admission, source-valid external halos, nonzero large-step "
    "tendencies, physics, forecast, product, tornado, supercell, and "
    "performance claims remain excluded"
)


def _words(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _acoustic_scratch_bytes(ni: int, nj: int, nk: int) -> int:
    u = (nk, nj + 2, ni + 3)
    v = (nk, nj + 3, ni + 2)
    shapes = (
        u,
        u,
        v,
        v,
        (ni + 3,),
        (nj + 3,),
        (nk, nj),
        (nk, nj),
        (nk, ni),
        (nk, ni),
        ((nk + 1) * nj, 4),
        ((nk + 1) * ni, 4),
    )
    return sum(_words(shape) for shape in shapes) * 4


@dataclass(frozen=True, slots=True)
class FullStateMemoryPlan:
    ni: int
    nj: int
    nk: int
    product_arena_bytes: int
    prepared_array_bytes: int
    acoustic_scratch_bytes: int
    device_reserve_bytes: int

    @property
    def projected_peak_live_bytes(self) -> int:
        return (
            self.product_arena_bytes
            + self.prepared_array_bytes
            + self.acoustic_scratch_bytes
        )

    @property
    def required_free_after_product_bytes(self) -> int:
        return (
            self.prepared_array_bytes
            + self.acoustic_scratch_bytes
            + self.device_reserve_bytes
        )

    def receipt_dict(self) -> dict[str, int]:
        return {
            "acoustic_scratch_bytes": self.acoustic_scratch_bytes,
            "device_reserve_bytes": self.device_reserve_bytes,
            "prepared_array_bytes": self.prepared_array_bytes,
            "product_arena_bytes": self.product_arena_bytes,
            "projected_peak_live_bytes": self.projected_peak_live_bytes,
            "required_free_after_product_bytes": (
                self.required_free_after_product_bytes
            ),
        }


def full_state_memory_plan(
    *,
    ni: int = FULL_NI,
    nj: int = FULL_NJ,
    nk: int = FULL_NK,
    product_arena_bytes: int = FULL_PRODUCT_PAYLOAD_BYTES,
    device_reserve_bytes: int = DEFAULT_DEVICE_RESERVE_BYTES,
) -> FullStateMemoryPlan:
    """Return the allocation bound without importing CUDA."""

    for name, value in (
        ("ni", ni),
        ("nj", nj),
        ("nk", nk),
        ("product_arena_bytes", product_arena_bytes),
        ("device_reserve_bytes", device_reserve_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    prepared = sum(
        _words(shape) for shape in terrain_open_sound_shapes(ni, nj, nk).values()
    ) * 4
    return FullStateMemoryPlan(
        ni=ni,
        nj=nj,
        nk=nk,
        product_arena_bytes=product_arena_bytes,
        prepared_array_bytes=prepared,
        acoustic_scratch_bytes=_acoustic_scratch_bytes(ni, nj, nk),
        device_reserve_bytes=device_reserve_bytes,
    )


def _validate_cold_start_history_request(
    history_mode: str, imported_histories: object | None
) -> None:
    if history_mode != COLD_START_HISTORY_MODE:
        raise ValueError(
            "full-state initialization currently admits only cold-start-zero "
            "radiation histories anchored at cm1.F:1559-1566; resumed state "
            "must import all four history carriers before this gate is widened"
        )
    if imported_histories is not None:
        raise ValueError(
            "cold-start-zero refuses imported radiation histories; select an "
            "explicit resumed-history path once the authority ABI is available"
        )


@dataclass(frozen=True, slots=True)
class FullStateArm:
    epoch: int
    a_current_provenance: str
    tendency_provenance: str
    radiation_history_provenance: str


@dataclass(slots=True)
class CudaTerrainOpenFullState:
    prepared: CudaInitializedTerrainOpenState
    memory_plan: FullStateMemoryPlan
    history_mode: str
    memory_receipt: dict[str, int]
    armed: FullStateArm | None
    completed_steps: int = 0
    last_ready_event: Any | None = None
    last_acoustic: CudaPsolver3TerrainOpenLaunch | None = None

    @property
    def arrays(self) -> dict[str, Any]:
        return self.prepared.arrays

    @property
    def metrics(self) -> dict[str, Any]:
        return self.prepared.metrics


@dataclass(frozen=True, slots=True)
class CudaTerrainOpenFullStep:
    state: CudaTerrainOpenFullState
    acoustic: CudaPsolver3TerrainOpenLaunch
    arm: FullStateArm
    receipt: dict[str, Any]


def initialize_cuda_terrain_open_full_state(
    product: CudaInitializedProduct,
    *,
    history_mode: str = COLD_START_HISTORY_MODE,
    imported_histories: object | None = None,
    device_reserve_bytes: int = DEFAULT_DEVICE_RESERVE_BYTES,
    stream: Any | None = None,
) -> CudaTerrainOpenFullState:
    """Prepare the exact 300x300x75 product without hiding memory or history."""

    if not isinstance(product, CudaInitializedProduct):
        raise TypeError("product must be a CudaInitializedProduct")
    if not product.export.is_full_domain:
        raise ValueError("full-state API refuses cropped initialized products")
    ni, nj = product.export.crop_count_yx[1], product.export.crop_count_yx[0]
    nk = product.export.carrier("pi0").shape[0]
    if (ni, nj, nk) != (FULL_NI, FULL_NJ, FULL_NK):
        raise ValueError("full-state API requires the exact 300x300x75 product")
    if product.export.payload_nbytes != FULL_PRODUCT_PAYLOAD_BYTES:
        raise ValueError("full-state product arena byte count changed")
    _validate_cold_start_history_request(history_mode, imported_histories)

    cupy = importlib.import_module("cupy")
    stream = stream or product.stream
    product.ready_event.synchronize()
    free_before, total = cupy.cuda.runtime.memGetInfo()
    plan = full_state_memory_plan(device_reserve_bytes=device_reserve_bytes)
    if free_before < plan.required_free_after_product_bytes:
        raise MemoryError(
            "full terrain/open acoustic state refused: free device bytes after "
            f"product={free_before}, required including reserve="
            f"{plan.required_free_after_product_bytes}"
        )
    prepared = prepare_cuda_initialized_product_terrain_open(
        product,
        allow_cropped_validation=False,
        stream=stream,
    )
    prepared.ready_event.synchronize()
    zero_counts = {
        name: int(cupy.count_nonzero(prepared.arrays[name]).item())
        for name in RADIATION_HISTORY_NAMES
    }
    if any(zero_counts.values()):
        raise ValueError(
            "cold-start radiation histories differ from exact source-initialized zero"
        )
    tendency_counts = {
        name: int(cupy.count_nonzero(prepared.arrays[name]).item())
        for name in TENDENCY_NAMES
    }
    if any(tendency_counts.values()):
        raise ValueError("initial acoustic validation tendencies must be exact zero")
    free_after, total_after = cupy.cuda.runtime.memGetInfo()
    if total_after != total:
        raise RuntimeError("CUDA total memory changed during full-state preparation")
    if free_after < plan.acoustic_scratch_bytes + plan.device_reserve_bytes:
        raise MemoryError(
            "full terrain/open acoustic scratch plus reserve no longer fits after "
            "state preparation"
        )
    arm = FullStateArm(
        epoch=0,
        a_current_provenance="corrected-real initialized-product step-zero carriers",
        tendency_provenance="held-exact-zero acoustic-validation seam",
        radiation_history_provenance=(
            f"exact-zero cold start at {COLD_START_HISTORY_SOURCE_ANCHOR}"
        ),
    )
    return CudaTerrainOpenFullState(
        prepared=prepared,
        memory_plan=plan,
        history_mode=history_mode,
        memory_receipt={
            **plan.receipt_dict(),
            "free_bytes_after_product": int(free_before),
            "free_bytes_after_preparation": int(free_after),
            "total_device_bytes": int(total),
        },
        armed=arm,
        last_ready_event=prepared.ready_event,
    )


def arm_cuda_terrain_open_full_state_step(
    state: CudaTerrainOpenFullState,
    *,
    epoch: int,
    a_current_provenance: str,
    tendency_provenance: str,
    radiation_history_provenance: str,
) -> FullStateArm:
    """Arm a later coupling epoch without resetting any resident carrier."""

    if not isinstance(state, CudaTerrainOpenFullState):
        raise TypeError("state must be a CudaTerrainOpenFullState")
    if state.armed is not None:
        raise RuntimeError("full-state acoustic step is already armed")
    if epoch != state.completed_steps or epoch < 1:
        raise ValueError("later full-state epoch must equal completed step count")
    values = {
        "a/current provenance": a_current_provenance,
        "tendency provenance": tendency_provenance,
        "radiation-history provenance": radiation_history_provenance,
    }
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    history_text = radiation_history_provenance.lower()
    if any(
        marker in history_text
        for marker in ("reset", "cold-start-zero", "exact-zero", "zero histories")
    ):
        raise ValueError(
            "later full-state epochs refuse zero/reset radiation-history provenance"
        )
    if state.last_ready_event is not None:
        state.last_ready_event.synchronize()
    state.last_acoustic = None
    arm = FullStateArm(
        epoch=epoch,
        a_current_provenance=a_current_provenance.strip(),
        tendency_provenance=tendency_provenance.strip(),
        radiation_history_provenance=radiation_history_provenance.strip(),
    )
    state.armed = arm
    return arm


def launch_cuda_terrain_open_full_state_step(
    state: CudaTerrainOpenFullState,
    *,
    dt: float,
    nsound: int,
    stream: Any | None = None,
    compiled: Any | None = None,
    lateral_compiled: Any | None = None,
) -> CudaTerrainOpenFullStep:
    """Consume one explicit coupling arm and enqueue the native RK3 cadence."""

    if not isinstance(state, CudaTerrainOpenFullState):
        raise TypeError("state must be a CudaTerrainOpenFullState")
    arm = state.armed
    if arm is None:
        raise RuntimeError(
            "full-state step is not armed; coupling must supply A/current, "
            "tendency, and radiation-history provenance"
        )
    stream = stream or state.prepared.product.stream
    acoustic = launch_cuda_psolver3_terrain_open_three_stage(
        state.arrays,
        state.metrics,
        ni=FULL_NI,
        nj=FULL_NJ,
        nk=FULL_NK,
        dt=dt,
        nsound=nsound,
        dx=100.0,
        dy=100.0,
        dz=200.0,
        stream=stream,
        compiled=compiled,
        terrain_compiled=state.prepared.terrain_compiled,
        open_compiled=state.prepared.open_compiled,
        lateral_compiled=lateral_compiled,
    )
    state.armed = None
    state.completed_steps += 1
    state.last_ready_event = acoustic.ready_event
    state.last_acoustic = acoustic
    receipt = {
        "arm": {
            "a_current_provenance": arm.a_current_provenance,
            "epoch": arm.epoch,
            "radiation_history_provenance": arm.radiation_history_provenance,
            "tendency_provenance": arm.tendency_provenance,
        },
        "claim_limit": CUDA_TERRAIN_OPEN_FULL_STATE_CLAIM_LIMIT,
        "cold_start_history_source_anchor": COLD_START_HISTORY_SOURCE_ANCHOR,
        "memory": dict(state.memory_receipt),
        "schema": CUDA_TERRAIN_OPEN_FULL_STATE_SCHEMA,
        "source_acoustic_receipt": acoustic.receipt,
    }
    return CudaTerrainOpenFullStep(state, acoustic, arm, receipt)


__all__ = [
    "COLD_START_HISTORY_MODE",
    "COLD_START_HISTORY_SOURCE_ANCHOR",
    "CUDA_TERRAIN_OPEN_FULL_STATE_CLAIM_LIMIT",
    "CUDA_TERRAIN_OPEN_FULL_STATE_SCHEMA",
    "CudaTerrainOpenFullState",
    "CudaTerrainOpenFullStep",
    "DEFAULT_DEVICE_RESERVE_BYTES",
    "FULL_NI",
    "FULL_NJ",
    "FULL_NK",
    "FULL_PRODUCT_PAYLOAD_BYTES",
    "FullStateArm",
    "FullStateMemoryPlan",
    "RADIATION_HISTORY_NAMES",
    "TENDENCY_NAMES",
    "arm_cuda_terrain_open_full_state_step",
    "full_state_memory_plan",
    "initialize_cuda_terrain_open_full_state",
    "launch_cuda_terrain_open_full_state_step",
]
