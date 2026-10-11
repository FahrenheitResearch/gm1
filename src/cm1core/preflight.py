"""Pure, allocation-free runtime admission."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CM1Config, Lane
from .engine_pin import GPUWM_MEASURED_VERSION, GPUWM_REQUIREMENT, inspect
from .errors import refuse


@dataclass(frozen=True, slots=True)
class Blocker:
    code: str
    gate: str
    reason: str
    remedy: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "gate": self.gate,
            "reason": self.reason,
            "remedy": self.remedy,
        }


def blockers(config: CM1Config, *, device_physics: bool = True) -> tuple[Blocker, ...]:
    """``device_physics=False`` for a door that compiles and launches nothing.

    CM1's ``iconly=1`` (``cm1.F:4341-4346``) builds the initial condition and
    terminates.  No microphysics kernel is compiled, no device is opened and no
    ArWen symbol is read, so the byte-admission of the substituted component is
    not a condition of that door: refusing it there would fire on valid input.
    """
    result: list[Blocker] = []
    # The Cartesian lane composes and executes the complete source-ordered
    # large step on device and makes no native-trajectory, instrument or
    # Rust history/checkpoint claim, so none of the three lane-independent
    # blockers below is true of it.  Every other lane still carries them.
    composed_trajectory_lane = config.run.lane is Lane.CARTESIAN
    # Keep the first refusal on the first absent executable composition seam.
    # Structural state, bounded dry operators and pressure columns now exist;
    # none of those isolated slices is a complete large-step trajectory.
    if not composed_trajectory_lane:
        result.append(
            Blocker(
                "CM1-CUDA-TRAJECTORY-INCOMPLETE",
                "composed-trajectory",
                "audited CUDA structural initialization, dry RK/advection, vertical operators and a psolver=3 implicit column now pass bounded CPU/GPU gates, but horizontal acoustic coupling, family boundaries, closures and a complete source-ordered large step have not yet been composed into one forecast trajectory",
                "compose the horizontally varying psolver=3 path with dry dynamics, boundaries, damping, mass and SGS/TKE at native cadence, then pass one-step and multi-step trajectory decks",
            )
        )
        result.append(
            Blocker(
                "CM1-INSTRUMENTS-MISSING",
                "reference-instruments",
                "the operator ledger is partial; bounded grid/base and 46-field post-GETSET decks pass two optimization builds, repeats, and all four controls, but omitted initialization fields, complete 3-D grid metrics, and every time-dependent operator deck remain absent",
                "preserve O-GRID-BASE-V1 and O-POST-GETSET-V1, complete the source ledger/live-state inventory, and capture negative-controlled boundary, advection/WENO, pressure/acoustic, RK/step, mass, closure and restart decks",
            )
        )
    if device_physics and config.run.lane in {Lane.PRODUCT, Lane.REAL_PRODUCT, Lane.CARTESIAN}:
        # The Cartesian lane executes the substituted component, so it is
        # admitted against the exact capacity surface its [runtime] selects.
        profile = None
        if config.runtime is not None and config.runtime.capacity_profile != "default":
            profile = config.runtime.capacity_profile
        estate = inspect(capacity_profile=profile)
        if not estate.admitted:
            detail = (
                f"installed={estate.version!r}, moved={list(estate.moved)!r}, "
                f"absent={list(estate.absent)!r}"
            )
            result.append(
                Blocker(
                    "CM1-ARWEN-ESTATE-DRIFT",
                    "engine-pin",
                    "the audited ArWen 2.8.0 WSM6 candidate surface is not byte-admitted: "
                    + detail,
                    f"install {GPUWM_REQUIREMENT} and re-pin only from a measured {GPUWM_MEASURED_VERSION} tree",
                )
            )
    # CM1's radiation, PBL and cumulus families.  A selected family GM1
    # cannot execute blocks here, on the host, naming the concrete wrong
    # result -- it is never silently downgraded to the neutral profile.
    result.extend(physics_component_blockers(config))
    if config.run.lane is Lane.DRY_DYCORE:
        result.append(
            Blocker(
                "CM1-DRY-PSOLVER2-ORACLE-MISSING",
                "dry-psolver2-bootstrap",
                "the isotropic periodic dry profile is only a 42-key configuration contract; it has no complete derived native namelist/options inventory or instrumented psolver=2 oracle",
                "capture the complete derived native fixture and pass O-PSOLVER2 with negative controls before implementing or allocating the bootstrap",
            )
        )
    if config.run.lane is Lane.REAL_PRODUCT:
        # The four numerical gaps this blocker used to name are closed and are
        # no longer refused: the Rust HRRR-to-CM1 mapper performs the
        # terrain-aware horizontal remap (`rust/cm1-state/src/hrrr.rs`), VVEL
        # omega converts to geometric w with no zero-w fallback
        # (`hrrr::VERTICAL_MOTION_CONTRACT`), the CM1 carriers are published as
        # a hash-bound CM1IPR1 initialized product
        # (`rust/cm1-state/src/product_restart.rs`), and time-varying Davies
        # lateral forcing is driven by `cm1core.cuda_gm1_hrrr_lbc`.  What
        # remains true is only the last clause, and the refusal now says just
        # that: this distribution installs no door that executes the
        # real-product lane, so reporting such a configuration as runnable
        # would promise an executable path that does not exist.
        result.append(
            Blocker(
                "CM1-REAL-PRODUCT-NO-INSTALLED-DOOR",
                "real-initialization",
                "the real HRRR-to-CM1 mapper, geometric-w conversion, CM1IPR1 carrier publication and time-varying Davies lateral forcing are implemented and hash-admitted, but this distribution installs no door that drives them as one forecast; the real-product lane runs only through the recorded producers in tools/",
                "run the recorded real-product producers directly, or select run.lane = 'cartesian' for the installed executable door",
            )
        )
    if config.run.lane in {Lane.PRODUCT, Lane.REAL_PRODUCT}:
        result.append(
            Blocker(
                "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING",
                "eqtset2-coupling",
                "the source-ordered eqtset=2 tendency lifecycle and public ArWen WSM6 bridge pass raw-word CPU and repeat-exact sm86 gates, but idiss, efall, reflectivity consumption, restart import and a coupled multi-step trajectory remain unimplemented",
                "complete the remaining eqtset=2 energy terms and restart carrier import, then pass coupled water/energy/positivity and uninterrupted/resumed trajectory gates",
            )
        )
    if config.run.lane is Lane.NATIVE_CONTEXT:
        result.append(
            Blocker(
                "CM1-NATIVE-CONTEXT-NOT-A-PRODUCT-DOOR",
                "claim-boundary",
                "native CM1 physics is an oracle/context lane and is not executed by this distribution",
                "run the recorded pristine native reference campaign outside the package, or select product only after the grid, dynamics, boundary and physics gates pass",
            )
        )
    if not composed_trajectory_lane:
        result.append(
            Blocker(
                "CM1-RUST-LIVE-WIRING-MISSING",
                "rust-io-render",
                "the sealed wheel installs hash/ABI-verified Rust state and renderer executables, and the renderer directly ingests independently anchored CM1FHS1/CM1FCP1 envelopes; complete live CUDA capture/restore, semantic carrier completeness and uninterrupted/resumed identity remain open",
                "bind every live forecast, boundary, closure and physics carrier to CM1FCP1/CM1FHS1 and prove uninterrupted/resumed identity with identical history schedules",
            )
        )
    return tuple(result)


def physics_component_blockers(config: CM1Config) -> tuple[Blocker, ...]:
    """One blocker per selected radiation/PBL/cumulus family GM1 cannot run.

    Separate from :func:`blockers` so ``doctor`` can list the dispositions of
    every family without asking whether the whole run is admissible.
    """

    from .arwen_physics_component import (
        ARWEN_PHYSICS_SEAM_GATE,
        UNCOUPLED,
        UNPORTED,
        resolve_component,
    )

    radopt = 0 if config.radiation is None else config.radiation.radopt
    codes = {
        UNPORTED: "CM1-PHYSICS-COMPONENT-UNPORTED",
        UNCOUPLED: "CM1-PHYSICS-COMPONENT-UNCOUPLED",
    }
    options = {
        "radiation": "radiation.radopt",
        "pbl": "closures.ipbl",
        "cumulus": "physics.cuparam",
    }
    found: list[Blocker] = []
    for family, value in (
        ("radiation", radopt),
        ("pbl", config.closures.ipbl),
        ("cumulus", config.physics.cuparam_value),
    ):
        component = resolve_component(family, value)
        if component.runnable:
            continue
        remedy = f"set {options[family]} to 0"
        if component.missing_carriers:
            remedy += (
                "; the coupling this needs is "
                + ", ".join(component.missing_carriers)
                + " on the device"
            )
        if component.minimum_engine is not None:
            remedy += f"; the component itself needs {component.minimum_engine}"
        found.append(
            Blocker(
                codes[component.disposition],
                ARWEN_PHYSICS_SEAM_GATE,
                f"{options[family]}={value} selects {component.cm1_label}, which "
                f"CM1 runs at {component.cm1_source}, and {component.reason}",
                remedy,
            )
        )
    return tuple(found)


def require_runnable(config: CM1Config, *, device_physics: bool = True) -> None:
    gaps = blockers(config, device_physics=device_physics)
    if not gaps:
        return
    first = gaps[0]
    refuse(
        code=first.code,
        gate=first.gate,
        reason=first.reason,
        remedy=first.remedy,
    )


__all__ = ["Blocker", "blockers", "physics_component_blockers", "require_runnable"]
