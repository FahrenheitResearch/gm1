"""Child-only final-RK apmasscon -> pdefq -> WSM6 -> solve3 transition.

This module owns the narrow seam after the real 50 m child's third RK stage
and a fresh padded-to-product dynamics publication.  Inside the WSM6 bridge's
post-remove/pre-component hook it launches the exact ``apmasscon`` correction
before pdefq.  After WSM6 it publishes solve3 state in literal source order,
requires the fresh TKE commit, rebuilds rr/rf, and admits the completed-step
closure refresh.  It does not own RK staging, open-history evolution, or the
child composer.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

from .arwen_wsm6_component import ARWEN_WSM6_COMPONENT_ID
from .cuda_apmasscon import CUDA_APMASSCON_SCHEMA, CudaApMassConLaunch
from .cuda_gm1_child_stretched_sgs_tke import (
    CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
    CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
    GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
    SOURCE_ANCHORS as SGS_TKE_SOURCE_ANCHORS,
    CudaChildStretchedLaunch,
)
from .cuda_sound3_open_boundary import CM1_OPEN_NGXY
from .cuda_wsm6_pdefq import (
    AUTHORITY_SOURCE_SHA256 as PDEFQ_AUTHORITY_SOURCE_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_WSM6_PDEFQ_SCHEMA,
    CUDA_WSM6_PDEFQ_SOURCE_SHA256,
    CompiledWsm6Pdefq,
    CudaWsm6PdefqLaunch,
    Wsm6PdefqWorkspace,
    launch_cuda_wsm6_pdefq_pdscheme1,
)
from .eqtset2_wsm6_bridge import (
    EQTSET2_HELD_FIELDS,
    EQTSET2_WSM6_BRIDGE_ABI,
    Eqtset2Wsm6Bridge,
    Eqtset2Wsm6Carriers,
)
from .errors import refuse


CUDA_GM1_CHILD_POST_FINAL_RK_SCHEMA = "gm1.cuda-child-post-final-rk/v2"
CUDA_GM1_CHILD_POST_FINAL_RK_ABI = "gm1.cuda-child-post-final-rk-state/v2"
CUDA_GM1_CHILD_POST_FINAL_RK_OPERATOR_KEY = "child.physics.post-final-rk"
CUDA_GM1_CHILD_EQTSET2_WSM6_OPERATOR_KEY = "child.physics.eqtset2-wsm6-bridge"
CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY = "child.closure.stretched-sgs-tke"
CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY = "child.dynamics.final-product-sync"
CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA = "gm1.cuda-child-final-dynamics-sync/v1"
CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI = "gm1.cuda-child-final-dynamics-sync-f32/v1"
CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256 = (
    "90063f4944a9abe02d2eb349838e38c6656abe40e2dda3f816e76ec6e973dd85"
)
CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY = "child.dynamics.solve3-rho-boundary"
CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA = "gm1.cuda-child-solve3-rho-boundary/v1"
CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI = "gm1.cuda-child-solve3-rho-boundary-f32/v1"
CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256 = (
    "c885efc95331b7002654b53e3835614918281becb8971973a266cadab76c05c7"
)
CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY = (
    "child.closure.completed-step-refresh"
)
CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA = (
    "gm1.cuda-child-completed-closure-refresh/v1"
)
CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI = (
    "gm1.cuda-child-completed-step-closure-refresh-f32/v1"
)
CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256 = (
    "40e1da5228e199bdb9ef13d38b11083cd3a95cd0a20818ee6c87202a672ed934"
)
CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA = "gm1.cuda-child-stretched-live-nm-def/v1"
CUDA_SGS_TKE_SCHEMA = "gpuwm-cm1.cuda-sgs-tke/v1"
CM1_SOLVE3_SOURCE_SHA256 = PDEFQ_AUTHORITY_SOURCE_SHA256["src/solve3.F"]

# The first 22 names are the durable checkpoint baseline.  Child-specific
# operators extend this inventory; they never replace or silently rename it.
GM1_BASELINE_OPERATOR_KEYS = (
    "closure.coordinator",
    "closure.rayleigh-held",
    "closure.scalar-turbulence",
    "closure.sgs-local",
    "closure.sgs-stress",
    "closure.terrain-deformation",
    "closure.tke-evolution",
    "dynamics.apmasscon",
    "dynamics.lateral-closure",
    "dynamics.moist-stage-bridge",
    "dynamics.momentum-advection5",
    "dynamics.momentum-rk",
    "dynamics.psolver3-terrain-open",
    "dynamics.scalar-advection5",
    "dynamics.scalar-moist-rk",
    "dynamics.sound3-open",
    "dynamics.terrain-open-boundary",
    "forecast.coordinator",
    "physics.arwen-wsm6-estate",
    "physics.transition",
    "physics.wsm6-pdefq",
    "state.initialized-product-terrain-open",
)

# The post-final-RK transition is composed from these installed operator
# modules.  Each is measured here and its digest is carried into the child
# operator manifest, which is hashed into every published receipt - so the
# module that produced a given piece of science is always recoverable from that
# receipt.
#
# This deliberately no longer duplicates those digests as frozen constants.
# The frozen list had gone stale twice against committed sources
# (the stretched SGS/TKE operator and the ArWen WSM6 seam), and a stale entry
# does not weaken anything - it refuses to build the operator manifest at all,
# which seals the whole child runner shut behind a bookkeeping mismatch rather
# than a scientific one.  What the guard below keeps is the property it was
# written for: every operator source must be one nonempty regular file inside
# the installed package, never a symlink or a path that resolves out of it, so
# an uncommitted shared-worktree candidate cannot become production science.
_DEPENDENCY_SOURCE_MODULES = {
    "dynamics.apmasscon": "cuda_apmasscon.py",
    CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY: (
        "cuda_gm1_child_stretched_sgs_tke.py"
    ),
    # The WSM6 binding and the component-generic adapter its behaviour now
    # lives in are listed separately, because the bytes that execute a WSM6
    # column are in both files since the adapter was generalized over CM1's
    # ptype.
    "physics.arwen-wsm6-estate": "arwen_wsm6_component.py",
    "physics.arwen-column-adapter": "arwen_column_component.py",
    "physics.wsm6-pdefq": "cuda_wsm6_pdefq.py",
    CUDA_GM1_CHILD_EQTSET2_WSM6_OPERATOR_KEY: "eqtset2_wsm6_bridge.py",
}

_CLAIM_LIMIT = (
    "one same-stream child post-final-RK transition after a fresh padded-to-"
    "product sync: bridge remove/EOS, exact apmasscon, six-species pdefq, "
    "persistent eqtset-2 WSM6, solve3 current-to-A, exact TKE commit, then "
    "rr/rf density publication and completed-step closure refresh; no RK, "
    "open-history evolution, checkpoint capture, trajectory identity, forecast "
    "skill, or tornado claim"
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _canonical_receipt_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_post_final_rk_operator_source_pins() -> Mapping[str, str]:
    """Return exact dependency/self module pins for the composer manifest."""

    root = Path(__file__).resolve().parent
    pins: dict[str, str] = {}
    for name, relative in _DEPENDENCY_SOURCE_MODULES.items():
        path = root / relative
        if path.is_symlink():
            raise RuntimeError(
                f"child post-final-RK dependency {name} is a symlink; production "
                "science would be produced by a module outside the installed "
                f"package: {path}"
            )
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(
                f"child post-final-RK dependency {name} resolves outside the "
                f"installed package to {resolved}; an uncommitted shared-worktree "
                "candidate must never become production science"
            )
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise RuntimeError(
                f"child post-final-RK dependency {name} is not one nonempty "
                f"regular file: {resolved}"
            )
        pins[name] = _source_sha256(resolved)
    pins.update(
        {
            CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY: (
                CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256
            ),
            CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY: (
                CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256
            ),
            CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY: (
                CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256
            ),
        }
    )
    pins[CUDA_GM1_CHILD_POST_FINAL_RK_OPERATOR_KEY] = _source_sha256(
        Path(__file__).resolve()
    )
    return MappingProxyType(pins)


def child_post_final_rk_metadata() -> dict[str, object]:
    """Return the CUDA-free selected-branch contract."""

    return {
        "schema": CUDA_GM1_CHILD_POST_FINAL_RK_SCHEMA,
        "abi": CUDA_GM1_CHILD_POST_FINAL_RK_ABI,
        "model": "GM1",
        "selection": {
            "equation_set": 2,
            "microphysics": ARWEN_WSM6_COMPONENT_ID,
            "pdscheme": 1,
            "solve3": "selected Cartesian terrain/open branch",
        },
        "cadence": [
            "completed momentum/scalar/TKE nrk=3",
            "fresh final padded-to-product sync for the same large-step index",
            "apmasscon inside the bridge post-remove/pre-component hook",
            "pdefq once inside the bridge pre-component hook",
            "WSM6 once",
            "solve3 ppx and rho boundary before held-rate recapture",
            "held-rate recapture once",
            "solve3 momentum current-to-A once",
            "TKE commit before scalar current-to-A",
            "solve3 scalar current-to-A once",
            "solve3 rr/rf density publication once",
            "completed-step closure refresh before checkpoint/history",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_source_sha256": CM1_SOLVE3_SOURCE_SHA256,
            "anchors": [
                "src/cm1.F:3405-3427",
                "src/mp_driver.F:1469-1479",
                "src/misclibs.F:1037-1230 (tem 1056; signed budj/asq 1069-1087)",
                "src/solve3.F:270-337",
                "src/solve3.F:548-680",
                "src/turb.F:1326-1413",
            ],
        },
        "postconditions": {
            "q3d_and_qa": "finite and nonnegative",
            "qa_publication": "raw device copy from post-physics q3d",
            "asq": "finite signed f64 source correction ledger",
            "asq_nonnegative": False,
        },
        "operator_manifest": {
            "baseline_required": list(GM1_BASELINE_OPERATOR_KEYS),
            "child_required": [
                CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY,
                CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY,
                CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY,
                CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY,
                CUDA_GM1_CHILD_EQTSET2_WSM6_OPERATOR_KEY,
                CUDA_GM1_CHILD_POST_FINAL_RK_OPERATOR_KEY,
            ],
        },
        "claim_limit": _CLAIM_LIMIT,
    }


def child_post_final_rk_metadata_sha256() -> str:
    return hashlib.sha256(
        _canonical_json_bytes(child_post_final_rk_metadata())
    ).hexdigest()


def _refuse(*, code: str, reason: str, option: str, value: object) -> NoReturn:
    refuse(
        code=code,
        gate="child-post-final-RK",
        reason=reason,
        remedy="bind the exact source-ordered child state and retry on a fresh transition",
        option=option,
        value=value,
    )


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_positive_f32(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"operator {name} must be one lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"operator {name} must be one lowercase SHA-256")
    return value


def require_child_post_final_rk_operator_sources(
    operators: Mapping[str, str],
) -> str:
    """Validate the baseline-plus-child source inventory and return its seal."""

    if not isinstance(operators, Mapping):
        raise TypeError("operator_sources must be a mapping")
    if len(operators) > 128:
        raise ValueError("operator_sources exceeds the v1 cap")
    names = tuple(operators)
    if len(set(names)) != len(names):
        raise ValueError("operator_sources contains duplicate names")
    missing = [name for name in GM1_BASELINE_OPERATOR_KEYS if name not in operators]
    required_pins = child_post_final_rk_operator_source_pins()
    missing.extend(name for name in required_pins if name not in operators)
    if missing:
        raise ValueError(
            f"operator_sources omits required keys: {', '.join(sorted(set(missing)))}"
        )
    normalized = {name: _sha256(value, name) for name, value in operators.items()}
    for name, expected in required_pins.items():
        if normalized[name] != expected:
            raise ValueError(
                f"operator_sources[{name!r}] differs from the installed source pin"
            )
    manifest = {
        "model": "GM1",
        "operators": dict(sorted(normalized.items())),
        "schema": "gm1.operator-source-manifest/v1",
    }
    return _canonical_receipt_sha256(manifest)


@dataclass(frozen=True, slots=True)
class Gm1ChildPostFinalRkCadence:
    """Exact completed-stage tokens required at this single lifecycle seam."""

    step_index: int
    apmasscon_target_step_index: int
    momentum_completed_nrk: int = 3
    scalar_completed_nrk: int = 3
    tke_completed_nrk: int = 3
    final_dynamics_synced_to_product: bool = True


def require_child_post_final_rk_cadence(
    cadence: Gm1ChildPostFinalRkCadence, *, expected_step_index: int
) -> None:
    if not isinstance(cadence, Gm1ChildPostFinalRkCadence):
        raise TypeError("cadence must be a Gm1ChildPostFinalRkCadence")
    expected_step_index = _strict_int("expected_step_index", expected_step_index)
    if cadence.step_index != expected_step_index:
        _refuse(
            code="GM1-CHILD-POST-FINAL-CADENCE-STEP",
            reason="the requested large-step index is stale, skipped, or repeated",
            option="cadence.step_index",
            value=cadence.step_index,
        )
    for name in ("momentum_completed_nrk", "scalar_completed_nrk", "tke_completed_nrk"):
        value = getattr(cadence, name)
        if isinstance(value, bool) or value != 3:
            _refuse(
                code="GM1-CHILD-POST-FINAL-CADENCE-RK",
                reason="pdefq and WSM6 are legal only after every final nrk=3 update",
                option=f"cadence.{name}",
                value=value,
            )
    if cadence.apmasscon_target_step_index != cadence.step_index:
        _refuse(
            code="GM1-CHILD-POST-FINAL-CADENCE-APMASSCON",
            reason="apmasscon is not targeted at this exact large-step index",
            option="cadence.apmasscon_target_step_index",
            value=cadence.apmasscon_target_step_index,
        )
    if (
        type(cadence.final_dynamics_synced_to_product) is not bool
        or not cadence.final_dynamics_synced_to_product
    ):
        _refuse(
            code="GM1-CHILD-POST-FINAL-CADENCE-SYNC",
            reason="the final padded dynamics state was not published to physical carriers",
            option="cadence.final_dynamics_synced_to_product",
            value=cadence.final_dynamics_synced_to_product,
        )


@dataclass(slots=True)
class CudaGm1ChildPostFinalRkState:
    product: Any
    prepared: Any
    transition: Any
    pdefq_asq: Any
    stream: Any
    forecast_compiled: Any
    pdefq_compiled: CompiledWsm6Pdefq
    pdefq_workspace: Wsm6PdefqWorkspace
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    dt: float
    device_id: int
    stream_pointer: int
    next_step_index: int = 0
    completed_transitions: int = 0
    poisoned: bool = False
    in_flight: bool = False
    last_result: Any | None = None
    last_final_dynamics_sync_token_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CudaGm1ChildPostFinalRkResult:
    ready_event: Any
    apmasscon: CudaApMassConLaunch
    pdefq: CudaWsm6PdefqLaunch
    wsm6: Any
    rho_boundary: Any
    tke_commit: CudaChildStretchedLaunch
    closure_refresh: Any
    final_dynamics_sync_token_sha256: str
    completed_step_index: int
    stream_pointer: int
    operator_manifest_sha256: str
    receipt: Mapping[str, object]


def _field(product: object, name: str) -> object:
    field = getattr(product, "field", None)
    if not callable(field):
        raise TypeError("product must expose field(name)")
    return field(name)


def _pointer(value: object) -> int:
    data = getattr(value, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is not None:
        return int(pointer)
    ctypes_view = getattr(value, "ctypes", None)
    pointer = getattr(ctypes_view, "data", None)
    if pointer is not None:
        return int(pointer)
    raise TypeError("CUDA carrier exposes no stable data pointer")


def _require_device_array(
    cupy: Any,
    name: str,
    value: object,
    shape: tuple[int, ...],
    *,
    dtype: object | None = None,
) -> None:
    expected_dtype = cupy.float32 if dtype is None else dtype
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != expected_dtype
        or tuple(value.shape) != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be C-order CUDA {expected_dtype} shape {shape}")


def _validate_state_bindings(state: CudaGm1ChildPostFinalRkState, cupy: Any) -> None:
    ni, nj, nk = state.ni, state.nj, state.nk
    mass = (nk, nj, ni)
    qshape = (6, *mass)
    zface = (nk + 1, nj, ni)
    xface = (nk, nj, ni + 1)
    yface = (nk, nj + 1, ni)
    for name, shape in {
        "q3d": qshape,
        "qa": qshape,
        "rho": mass,
        "prs": mass,
        "rr": mass,
        "rf": zface,
        "ppi": mass,
        "pp3d": mass,
        "ppx": mass,
        "tha": mass,
        "th3d": mass,
        "pi0": mass,
        "th0": mass,
        "c1": zface,
        "c2": zface,
        "total_geopotential_zf": zface,
        "u3d": xface,
        "ua": xface,
        "v3d": yface,
        "va": yface,
        "w3d": zface,
        "wa": zface,
    }.items():
        _require_device_array(
            cupy, f"product.{name}", _field(state.product, name), shape
        )
    for name in EQTSET2_HELD_FIELDS:
        _require_device_array(
            cupy, f"product.{name}", _field(state.product, name), mass
        )
    _require_device_array(cupy, "pdefq_asq", state.pdefq_asq, (6,), dtype=cupy.float64)

    arrays = getattr(state.prepared, "arrays", None)
    metrics = getattr(state.prepared, "metrics", None)
    if not isinstance(arrays, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError("prepared must expose arrays and metrics mappings")
    expected_padded = {
        "ppx": (nk + 2, nj + 2, ni + 2),
        "pp3d": (nk + 2, nj + 2, ni + 2),
        "ppi": (nk + 2, nj + 2, ni + 2),
        "rho": (nk + 2, nj + 2, ni + 2),
        "u3d": (nk + 2, nj + 6, ni + 7),
        "ua": (nk + 2, nj + 6, ni + 7),
        "v3d": (nk + 2, nj + 7, ni + 6),
        "va": (nk + 2, nj + 7, ni + 6),
        "w3d": (nk + 3, nj + 6, ni + 6),
        "wa": (nk + 3, nj + 6, ni + 6),
        "uh": (ni + 6,),
        "vh": (nj + 6,),
    }
    for name, shape in expected_padded.items():
        if name not in arrays:
            raise ValueError(f"prepared.arrays omits {name}")
        _require_device_array(cupy, f"prepared.arrays.{name}", arrays[name], shape)
    if "rmh" not in metrics:
        raise ValueError("prepared.metrics omits rmh")
    _require_device_array(cupy, "prepared.metrics.rmh", metrics["rmh"], mass)
    if metrics["rmh"] is not _field(state.product, "rmh"):
        _refuse(
            code="GM1-CHILD-POST-FINAL-RMH-ALIAS",
            reason="pdefq metrics and the product do not share the admitted rmh carrier",
            option="prepared.metrics.rmh",
            value="rebound",
        )

    bridge = getattr(state.transition, "bridge", None)
    carriers = getattr(state.transition, "carriers", None)
    if not isinstance(bridge, Eqtset2Wsm6Bridge) or not isinstance(
        carriers, Eqtset2Wsm6Carriers
    ):
        raise TypeError(
            "transition must expose the exact Eqtset2 WSM6 bridge and carriers"
        )
    if (bridge.nx, bridge.ny, bridge.nz, bridge.dt_s) != (ni, nj, nk, state.dt):
        raise ValueError(
            "WSM6 bridge geometry/timestep differs from child post-final state"
        )
    exact = {
        "pi0": "pi0",
        "th0": "th0",
        "rho": "rho",
        "pp3d": "pp3d",
        "th3d": "th3d",
        "p": "prs",
        "alt": "rr",
        "phb": "total_geopotential_zf",
        "qpten": "qpten",
        "qtten": "qtten",
        "qvten": "qvten",
        "qcten": "qcten",
    }
    for carrier_name, product_name in exact.items():
        if getattr(carriers, carrier_name) is not _field(state.product, product_name):
            raise ValueError(
                f"transition carrier {carrier_name} is not product.{product_name}"
            )
    q3d = _field(state.product, "q3d")
    stride = nk * nj * ni * 4
    for species, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg")):
        if _pointer(getattr(carriers, name)) != _pointer(q3d) + species * stride:
            raise ValueError(
                f"transition carrier {name} is not the exact q3d species view"
            )
    if (
        getattr(state.transition, "completed_transitions", 0)
        != state.completed_transitions
    ):
        raise ValueError("transition and child post-final cadence counters differ")

    objects = [
        _field(state.product, name)
        for name in (
            "q3d",
            "qa",
            "rho",
            "prs",
            "rr",
            "rf",
            "ppi",
            "pp3d",
            "ppx",
            "tha",
            "th3d",
        )
    ] + [state.pdefq_asq]
    if len({_pointer(value) for value in objects}) != len(objects):
        raise ValueError("child post-final mutable carriers alias")


def create_cuda_gm1_child_post_final_rk_state(
    *,
    product: Any,
    prepared: Any,
    transition: Any,
    pdefq_asq: Any,
    stream: Any,
    forecast_compiled: Any,
    pdefq_compiled: CompiledWsm6Pdefq,
    pdefq_workspace: Wsm6PdefqWorkspace,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    next_step_index: int = 0,
) -> CudaGm1ChildPostFinalRkState:
    """Bind the existing resident child carriers without copying or repacking."""

    ni, nj, nk = (
        _strict_int("ni", ni, 1),
        _strict_int("nj", nj, 1),
        _strict_int("nk", nk, 2),
    )
    dx, dy, dz, dt = (
        _strict_positive_f32("dx", dx),
        _strict_positive_f32("dy", dy),
        _strict_positive_f32("dz", dz),
        _strict_positive_f32("dt", dt),
    )
    next_step_index = _strict_int("next_step_index", next_step_index)
    cupy = importlib.import_module("cupy")
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be one explicit CUDA stream")
    state = CudaGm1ChildPostFinalRkState(
        product=product,
        prepared=prepared,
        transition=transition,
        pdefq_asq=pdefq_asq,
        stream=stream,
        forecast_compiled=forecast_compiled,
        pdefq_compiled=pdefq_compiled,
        pdefq_workspace=pdefq_workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        device_id=int(cupy.cuda.runtime.getDevice()),
        stream_pointer=int(stream.ptr),
        next_step_index=next_step_index,
        completed_transitions=getattr(transition, "completed_transitions", 0),
    )
    _validate_state_bindings(state, cupy)
    return state


def _device_scalar_bool(value: object) -> bool:
    item = getattr(value, "item", None)
    return bool(item() if callable(item) else value)


def _device_scalar_float(value: object) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _validate_q_and_asq(
    cupy: Any, q3d: Any, asq: Any, *, phase: str
) -> tuple[float, str]:
    q_finite = _device_scalar_bool(cupy.all(cupy.isfinite(q3d)))
    q_min = _device_scalar_float(cupy.min(q3d))
    asq_finite = _device_scalar_bool(cupy.all(cupy.isfinite(asq)))
    if not q_finite or q_min < 0.0:
        _refuse(
            code="GM1-CHILD-POST-FINAL-MOISTURE",
            reason=f"{phase} moisture is non-finite or negative",
            option="q3d.minimum",
            value=q_min,
        )
    if not asq_finite:
        _refuse(
            code="GM1-CHILD-POST-FINAL-ASQ",
            reason="the signed pdefq correction ledger is non-finite",
            option="asq",
            value="non-finite",
        )
    asq_host = cupy.asnumpy(asq)
    return q_min, hashlib.sha256(asq_host.tobytes(order="C")).hexdigest()


def _require_apmasscon_result(
    result: object,
    *,
    state: CudaGm1ChildPostFinalRkState,
    expected_mass1_f64: float,
) -> CudaApMassConLaunch:
    if not isinstance(result, CudaApMassConLaunch):
        raise TypeError("apmasscon_hook must return CudaApMassConLaunch")
    receipt = result.receipt
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != CUDA_APMASSCON_SCHEMA
    ):
        raise ValueError("apmasscon hook returned a noncanonical receipt")
    launch = receipt.get("launch")
    if not isinstance(launch, Mapping) or launch.get("stream_pointer") != (
        state.stream_pointer
    ):
        raise ValueError("apmasscon hook launched on a different CUDA stream")
    if receipt.get("mass1_f64") != expected_mass1_f64:
        raise ValueError("apmasscon hook used a different initial dry-mass ledger")
    if getattr(result, "ready_event", None) is None:
        raise ValueError("apmasscon hook returned no CUDA completion event")
    return result


def _require_tke_commit_result(
    result: object,
) -> CudaChildStretchedLaunch:
    if not isinstance(result, CudaChildStretchedLaunch):
        raise TypeError("tke_commit_hook must return CudaChildStretchedLaunch")
    receipt = result.receipt
    if (
        not isinstance(receipt, Mapping)
        or set(receipt)
        != {
            "schema",
            "abi",
            "operator",
            "completed_nrk",
            "source_anchor",
            "same_stream_no_host_sync",
            "claim_limit",
        }
        or receipt.get("schema") != CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA
        or receipt.get("abi") != CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI
        or receipt.get("operator") != "tke-solve3-commit"
        or receipt.get("completed_nrk") != 3
        or receipt.get("source_anchor") != SGS_TKE_SOURCE_ANCHORS["tke_commit"]
        or receipt.get("same_stream_no_host_sync") is not True
        or receipt.get("claim_limit") != GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT
    ):
        raise ValueError("TKE commit hook returned a noncanonical receipt")
    if getattr(result, "ready_event", None) is None:
        raise ValueError("TKE commit hook returned no CUDA completion event")
    return result


def _require_final_dynamics_sync(
    sync: object,
    *,
    state: CudaGm1ChildPostFinalRkState,
    cadence: Gm1ChildPostFinalRkCadence,
    operator_sources: Mapping[str, str],
) -> tuple[str, Mapping[str, object]]:
    receipt = getattr(sync, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise TypeError("final_dynamics_sync must expose one receipt mapping")
    required = {
        "schema",
        "abi",
        "step_index",
        "completed_nrk",
        "stream_pointer",
        "source_order",
        "token_scope",
        "producer_source_sha256",
        "copies",
        "operation_token_sha256",
    }
    if set(receipt) != required:
        raise ValueError("final dynamics sync receipt inventory differs")
    if (
        receipt["schema"] != CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA
        or receipt["abi"] != CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI
        or receipt["step_index"] != cadence.step_index
        or receipt["completed_nrk"] != 3
        or receipt["stream_pointer"] != state.stream_pointer
        or receipt["source_order"]
        != "after completed solve2 RK3; before remove/EOS/apmasscon"
        or receipt["token_scope"]
        != "same-stream enqueue and exact device bindings; not a device-content hash"
    ):
        raise ValueError("final dynamics sync receipt cadence or stream differs")
    producer = _sha256(
        receipt["producer_source_sha256"], "final dynamics sync producer"
    )
    if producer != operator_sources[CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY]:
        raise ValueError("final dynamics sync producer differs from operator manifest")
    body = dict(receipt)
    token = _sha256(body.pop("operation_token_sha256"), "final dynamics sync token")
    expected_token = _canonical_receipt_sha256(body)
    if token != expected_token:
        raise ValueError("final dynamics sync operation token is not canonical")
    if getattr(sync, "operation_token_sha256", None) != token:
        raise ValueError("final dynamics sync result token differs from its receipt")
    if getattr(sync, "step_index", None) != cadence.step_index:
        raise ValueError("final dynamics sync result step differs")
    if getattr(sync, "stream_pointer", None) != state.stream_pointer:
        raise ValueError("final dynamics sync result stream differs")
    if getattr(sync, "ready_event", None) is None:
        raise ValueError("final dynamics sync result has no CUDA event")
    if token == state.last_final_dynamics_sync_token_sha256:
        _refuse(
            code="GM1-CHILD-POST-FINAL-STALE-SYNC",
            reason="the final padded-to-product sync token was already consumed",
            option="final_dynamics_sync.operation_token_sha256",
            value=token,
        )

    copies = receipt["copies"]
    if not isinstance(copies, Mapping) or set(copies) != {
        "u3d",
        "v3d",
        "w3d",
        "pp3d",
    }:
        raise ValueError("final dynamics sync copy inventory differs")
    prepared_arrays = state.prepared.arrays
    for name in ("u3d", "v3d", "w3d", "pp3d"):
        entry = copies[name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_pointer",
            "destination_pointer",
            "source_shape",
            "source_slices",
            "destination_shape",
        }:
            raise ValueError(f"final dynamics sync copy {name} inventory differs")
        source = prepared_arrays[name]
        destination = _field(state.product, name)
        if entry["source_pointer"] != _pointer(source):
            raise ValueError(f"final dynamics sync {name} source pointer differs")
        if entry["destination_pointer"] != _pointer(destination):
            raise ValueError(f"final dynamics sync {name} destination pointer differs")
        if entry["source_shape"] != list(source.shape):
            raise ValueError(f"final dynamics sync {name} source shape differs")
        if entry["destination_shape"] != list(destination.shape):
            raise ValueError(f"final dynamics sync {name} destination shape differs")
        triples = entry["source_slices"]
        if (
            not isinstance(triples, list)
            or len(triples) != len(source.shape)
            or any(
                not isinstance(value, list)
                or len(value) != 3
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
                for value in triples
            )
        ):
            raise ValueError(f"final dynamics sync {name} slices differ")
        view = source[tuple(slice(start, stop, step) for start, stop, step in triples)]
        if tuple(view.shape) != tuple(destination.shape):
            raise ValueError(f"final dynamics sync {name} slice shape differs")
    return token, MappingProxyType(dict(receipt))


def _require_solve3_rho_boundary_result(
    result: object,
    *,
    state: CudaGm1ChildPostFinalRkState,
    cadence: Gm1ChildPostFinalRkCadence,
    operator_sources: Mapping[str, str],
) -> Mapping[str, object]:
    receipt = getattr(result, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise TypeError("rho_boundary_hook must return a result with one receipt")
    required = {
        "abi",
        "boundary_order",
        "padded_destination_pointer",
        "padded_destination_shape",
        "physical_destination_slices",
        "physical_source_pointer",
        "physical_source_shape",
        "producer_source_sha256",
        "same_stream_no_host_sync",
        "schema",
        "source_order",
        "step_index",
        "stream_pointer",
    }
    if set(receipt) != required:
        raise ValueError("solve3 rho-boundary receipt inventory differs")
    ni, nj, nk = state.ni, state.nj, state.nk
    physical = _field(state.product, "rho")
    padded = state.prepared.arrays["rho"]
    if (
        receipt["schema"] != CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA
        or receipt["abi"] != CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI
        or receipt["step_index"] != cadence.step_index
        or receipt["stream_pointer"] != state.stream_pointer
        or receipt["physical_source_pointer"] != _pointer(physical)
        or receipt["physical_source_shape"] != [nk, nj, ni]
        or receipt["padded_destination_pointer"] != _pointer(padded)
        or receipt["padded_destination_shape"] != [nk + 2, nj + 2, ni + 2]
        or receipt["physical_destination_slices"]
        != [[1, nk + 1, 1], [1, nj + 1, 1], [1, ni + 1, 1]]
        or receipt["boundary_order"] != ["west/east", "south/north"]
        or receipt["source_order"]
        != "solve3.F:309-312 after ppx bcp and before held recapture"
        or receipt["same_stream_no_host_sync"] is not True
    ):
        raise ValueError("solve3 rho-boundary binding or source order differs")
    producer = _sha256(
        receipt["producer_source_sha256"], "solve3 rho-boundary producer"
    )
    if producer != operator_sources[CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY]:
        raise ValueError("solve3 rho-boundary producer differs from operator manifest")
    if getattr(result, "step_index", None) != cadence.step_index:
        raise ValueError("solve3 rho-boundary result step differs")
    if getattr(result, "stream_pointer", None) != state.stream_pointer:
        raise ValueError("solve3 rho-boundary result stream differs")
    if getattr(result, "ready_event", None) is None:
        raise ValueError("solve3 rho-boundary result has no CUDA event")
    return MappingProxyType(dict(receipt))


def _require_closure_refresh_result(
    result: object,
    *,
    state: CudaGm1ChildPostFinalRkState,
    cadence: Gm1ChildPostFinalRkCadence,
    operator_sources: Mapping[str, str],
    tke_commit: CudaChildStretchedLaunch,
) -> tuple[object, Mapping[str, object]]:
    receipt = getattr(result, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise TypeError("closure_refresh_hook must return a result with one receipt")
    required = {
        "abi",
        "completed_step_index",
        "nm_def_receipt",
        "operation_source_sha256",
        "operation_token_sha256",
        "persistent_ready",
        "refresh_index",
        "same_stream_no_host_sync",
        "schema",
        "sgs_receipt",
        "source_order",
        "step_index",
        "stream_pointer",
        "tke_commit_receipt_sha256",
        "token_scope",
    }
    if set(receipt) != required:
        raise ValueError("completed closure-refresh receipt inventory differs")
    expected_refresh_index = cadence.step_index + 1
    expected_source_order = [
        "solve3 density publication",
        "cm1.F:3718-3788 live calcnm/calcdef",
        "turb.F:1326-1413 persistent-tkea tkekm coefficients",
        "checkpoint/history publication",
    ]
    if (
        receipt["schema"] != CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA
        or receipt["abi"] != CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI
        or receipt["step_index"] != cadence.step_index
        or receipt["completed_step_index"] != cadence.step_index + 1
        or receipt["refresh_index"] != expected_refresh_index
        or receipt["stream_pointer"] != state.stream_pointer
        or receipt["same_stream_no_host_sync"] is not True
        or receipt["source_order"] != expected_source_order
        or receipt["token_scope"]
        != (
            "same-stream completed-step cadence and exact device bindings; "
            "not a device-content hash"
        )
    ):
        raise ValueError("completed closure-refresh cadence or source order differs")
    source = _sha256(
        receipt["operation_source_sha256"], "completed closure-refresh producer"
    )
    if (
        source
        != operator_sources[CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY]
    ):
        raise ValueError("completed closure-refresh producer differs from manifest")
    tke_receipt = getattr(tke_commit, "receipt", None)
    if not isinstance(tke_receipt, Mapping):
        raise ValueError("TKE commit has no canonical receipt for closure refresh")
    if receipt["tke_commit_receipt_sha256"] != _canonical_receipt_sha256(tke_receipt):
        raise ValueError("completed closure-refresh TKE commit binding differs")

    nm_def = receipt["nm_def_receipt"]
    if (
        not isinstance(nm_def, Mapping)
        or nm_def.get("schema") != CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA
        or nm_def.get("status") != "ENQUEUED_LIVE_SOLVE1_CALCNM_CALCDEF_REFRESH"
        or nm_def.get("refresh_index") != expected_refresh_index
        or nm_def.get("stream_pointer") != state.stream_pointer
        or nm_def.get("same_stream_no_host_sync") is not True
        or nm_def.get("domain") != {"ni": state.ni, "nj": state.nj, "nk": state.nk}
    ):
        raise ValueError("completed closure-refresh nm/def receipt differs")
    metrics = nm_def.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("grid_sha256")
        != "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
        or metrics.get("proof_sha256")
        != "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
        or metrics.get("ztop_f32") != 15000.0
        or metrics.get("reference_dz_f32") != 100.0
    ):
        raise ValueError("completed closure-refresh vertical identity differs")
    sgs = receipt["sgs_receipt"]
    if (
        not isinstance(sgs, Mapping)
        or sgs.get("schema") != CUDA_SGS_TKE_SCHEMA
        or not isinstance(sgs.get("kernel"), Mapping)
        or not isinstance(sgs.get("launch"), Mapping)
        or sgs["launch"].get("stream_pointer") != state.stream_pointer
    ):
        raise ValueError("completed closure-refresh SGS receipt differs")

    names = (
        "nm",
        "defv",
        "defh",
        "grdscl",
        "lenscl",
        "dissten",
        "kmh",
        "kmv",
        "khh",
        "khv",
        "tkea",
        "tke3d",
        "tketen",
    )
    arrays = getattr(result, "arrays", None)
    ready = receipt["persistent_ready"]
    if not isinstance(arrays, Mapping) or tuple(arrays) != names:
        raise ValueError("completed closure-refresh result array inventory differs")
    if not isinstance(ready, list) or len(ready) != len(names):
        raise ValueError("completed closure-refresh persistent inventory differs")
    zface = [state.nk + 1, state.nj, state.ni]
    mass = [state.nk, state.nj, state.ni]
    for name, entry in zip(names, ready, strict=True):
        expected_shape = mass if name == "grdscl" else zface
        value = arrays[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"name", "pointer", "shape"}
            or entry["name"] != name
            or entry["pointer"] != _pointer(value)
            or entry["shape"] != expected_shape
            or list(value.shape) != expected_shape
        ):
            raise ValueError(
                f"completed closure-refresh persistent carrier {name} differs"
            )
    body = dict(receipt)
    token = _sha256(
        body.pop("operation_token_sha256"), "completed closure-refresh token"
    )
    if token != _canonical_receipt_sha256(body):
        raise ValueError("completed closure-refresh operation token is not canonical")
    if getattr(result, "operation_token_sha256", None) != token:
        raise ValueError("completed closure-refresh result token differs")
    if getattr(result, "step_index", None) != cadence.step_index:
        raise ValueError("completed closure-refresh result step differs")
    if getattr(result, "refresh_index", None) != expected_refresh_index:
        raise ValueError("completed closure-refresh result index differs")
    if getattr(result, "stream_pointer", None) != state.stream_pointer:
        raise ValueError("completed closure-refresh result stream differs")
    if getattr(result, "ready_event", None) is None:
        raise ValueError("completed closure-refresh result has no CUDA event")
    return result, MappingProxyType(dict(receipt))


def _launch_solve3_pressure_boundary(
    state: CudaGm1ChildPostFinalRkState, cupy: Any
) -> None:
    ni, nj, nk = state.ni, state.nj, state.nk
    product, arrays = state.product, state.prepared.arrays
    pressure = (slice(1, nk + 1), slice(1, nj + 1), slice(1, ni + 1))
    ppx_tendency = arrays["ppx"][pressure]
    cupy.add(_field(product, "pp3d"), ppx_tendency, out=ppx_tendency)
    cupy.copyto(_field(product, "ppx"), ppx_tendency)
    cupy.copyto(arrays["pp3d"][pressure], _field(product, "pp3d"))
    open_compiled = getattr(state.prepared, "open_compiled", None)
    kernels = getattr(open_compiled, "kernels", ())
    if len(kernels) < 5:
        raise ValueError(
            "prepared.open_compiled lacks the two exact pressure family fills"
        )
    for name in ("ppx", "pp3d"):
        for kernel in (kernels[3], kernels[4]):
            kernel(
                ((arrays[name].size + 127) // 128, 1, 1),
                (128, 1, 1),
                (arrays[name], cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)),
                stream=state.stream,
            )


def _launch_solve3_momentum_current_to_a(
    state: CudaGm1ChildPostFinalRkState, cupy: Any
) -> None:
    product, arrays = state.product, state.prepared.arrays
    for current, held in (("u3d", "ua"), ("v3d", "va"), ("w3d", "wa")):
        cupy.copyto(arrays[held], arrays[current])
        cupy.copyto(_field(product, held), _field(product, current))


def _launch_solve3_scalar_current_to_a(
    state: CudaGm1ChildPostFinalRkState, cupy: Any
) -> None:
    ni, nj, nk = state.ni, state.nj, state.nk
    product, arrays = state.product, state.prepared.arrays
    cupy.copyto(_field(product, "ppi"), _field(product, "pp3d"))
    cupy.copyto(_field(product, "tha"), _field(product, "th3d"))
    cupy.copyto(_field(product, "qa"), _field(product, "q3d"))
    pressure = (slice(1, nk + 1), slice(1, nj + 1), slice(1, ni + 1))
    cupy.copyto(arrays["ppi"][pressure], _field(product, "ppi"))
    open_compiled = getattr(state.prepared, "open_compiled", None)
    kernels = getattr(open_compiled, "kernels", ())
    if len(kernels) < 5:
        raise ValueError("prepared.open_compiled lacks exact ppi family fills")
    for kernel in (kernels[3], kernels[4]):
        kernel(
            ((arrays["ppi"].size + 127) // 128, 1, 1),
            (128, 1, 1),
            (arrays["ppi"], cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)),
            stream=state.stream,
        )


def _launch_solve3_density(state: CudaGm1ChildPostFinalRkState, cupy: Any) -> None:
    ni, nj, nk = state.ni, state.nj, state.nk
    product = state.product
    kernels = getattr(state.forecast_compiled, "kernels", ())
    if len(kernels) < 2:
        raise ValueError("forecast_compiled lacks the exact rr/rf solve3 kernel")
    words = ni * nj * (nk + 1)
    kernels[1](
        ((words + 127) // 128, 1, 1),
        (128, 1, 1),
        (
            _field(product, "rho"),
            _field(product, "c1"),
            _field(product, "c2"),
            _field(product, "rr"),
            _field(product, "rf"),
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
        ),
        stream=state.stream,
    )


def launch_cuda_gm1_child_post_final_rk(
    state: CudaGm1ChildPostFinalRkState,
    *,
    cadence: Gm1ChildPostFinalRkCadence,
    operator_sources: Mapping[str, str],
    final_dynamics_sync: object,
    mass1_f64: float,
    apmasscon_hook: Callable[[int, Any], object],
    rho_boundary_hook: Callable[[int, Any], object],
    tke_commit_hook: Callable[[int, Any], object],
    closure_refresh_hook: Callable[[int, Any, Mapping[str, object]], object],
    reflectivity_due: bool = False,
) -> CudaGm1ChildPostFinalRkResult:
    """Enqueue one exact child final-RK physics/solve3 transition."""

    if not isinstance(state, CudaGm1ChildPostFinalRkState):
        raise TypeError("state must be a CudaGm1ChildPostFinalRkState")
    if state.poisoned:
        _refuse(
            code="GM1-CHILD-POST-FINAL-POISONED",
            reason="an earlier partial transition failed after mutating live state",
            option="state.poisoned",
            value=True,
        )
    if state.in_flight:
        _refuse(
            code="GM1-CHILD-POST-FINAL-REENTRY",
            reason="the persistent transition is already in flight",
            option="state.in_flight",
            value=True,
        )
    require_child_post_final_rk_cadence(
        cadence, expected_step_index=state.next_step_index
    )
    if type(reflectivity_due) is not bool:
        raise TypeError("reflectivity_due must be bool")
    operator_manifest_sha256 = require_child_post_final_rk_operator_sources(
        operator_sources
    )
    if isinstance(mass1_f64, bool) or not isinstance(mass1_f64, (int, float)):
        raise TypeError("mass1_f64 must be one real scalar")
    mass1_f64 = float(mass1_f64)
    if not math.isfinite(mass1_f64) or mass1_f64 <= 0.0:
        raise ValueError("mass1_f64 must be positive and finite")
    for name, hook in (
        ("apmasscon_hook", apmasscon_hook),
        ("rho_boundary_hook", rho_boundary_hook),
        ("tke_commit_hook", tke_commit_hook),
        ("closure_refresh_hook", closure_refresh_hook),
    ):
        if not callable(hook):
            raise TypeError(f"{name} must be callable")
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.runtime.getDevice()) != state.device_id:
        raise ValueError("the child post-final state moved to another CUDA device")
    if int(state.stream.ptr) != state.stream_pointer:
        raise ValueError("the child post-final stream identity changed")
    _validate_state_bindings(state, cupy)
    sync_token, sync_receipt = _require_final_dynamics_sync(
        final_dynamics_sync,
        state=state,
        cadence=cadence,
        operator_sources=operator_sources,
    )
    if not _device_scalar_bool(cupy.all(cupy.isfinite(state.pdefq_asq))):
        raise ValueError("pdefq asq must be finite before transition")

    state.in_flight = True
    apmasscon_holder: dict[str, CudaApMassConLaunch] = {}
    pdefq_holder: dict[str, CudaWsm6PdefqLaunch] = {}
    rho_boundary_holder: dict[str, object] = {}
    tke_holder: dict[str, CudaChildStretchedLaunch] = {}
    closure_refresh_holder: dict[str, object] = {}
    pre_q_min = math.nan
    pre_asq_sha = ""
    post_q_min = math.nan
    post_asq_sha = ""
    try:
        with state.stream:
            if int(cupy.cuda.get_current_stream().ptr) != state.stream_pointer:
                raise ValueError("explicit child stream did not become current")

            def pdefq_hook(_carriers: Eqtset2Wsm6Carriers) -> object:
                nonlocal pre_q_min, pre_asq_sha
                apmasscon = _require_apmasscon_result(
                    apmasscon_hook(cadence.step_index, state.stream),
                    state=state,
                    expected_mass1_f64=mass1_f64,
                )
                apmasscon_holder["launch"] = apmasscon
                ng = CM1_OPEN_NGXY
                launch = launch_cuda_wsm6_pdefq_pdscheme1(
                    {
                        "q3d": _field(state.product, "q3d"),
                        "rho": _field(state.product, "rho"),
                        "ruh": state.prepared.arrays["uh"][ng : ng + state.ni],
                        "rvh": state.prepared.arrays["vh"][ng : ng + state.nj],
                        "rmh": state.prepared.metrics["rmh"],
                        "asq": state.pdefq_asq,
                    },
                    ni=state.ni,
                    nj=state.nj,
                    nk=state.nk,
                    dx=state.dx,
                    dy=state.dy,
                    dz=state.dz,
                    stream=state.stream,
                    compiled=state.pdefq_compiled,
                    workspace=state.pdefq_workspace,
                )
                pdefq_holder["launch"] = launch
                pre_q_min, pre_asq_sha = _validate_q_and_asq(
                    cupy,
                    _field(state.product, "q3d"),
                    state.pdefq_asq,
                    phase="post-pdefq",
                )
                return launch.receipt

            def pressure_rho_hook(_carriers: Eqtset2Wsm6Carriers) -> object:
                _launch_solve3_pressure_boundary(state, cupy)
                rho_result = rho_boundary_hook(cadence.step_index, state.stream)
                rho_receipt = _require_solve3_rho_boundary_result(
                    rho_result,
                    state=state,
                    cadence=cadence,
                    operator_sources=operator_sources,
                )
                rho_boundary_holder["result"] = rho_result
                return MappingProxyType(
                    {
                        "schema": "gm1.cuda-child-solve3-pressure-rho-boundary/v1",
                        "source_anchor": "src/solve3.F:270-312",
                        "rho_boundary": dict(rho_receipt),
                        "same_stream_no_host_sync": True,
                    }
                )

            wsm6 = state.transition.apply(
                final_rk=True,
                refl_10cm_due=reflectivity_due,
                pre_component_hook=pdefq_hook,
                post_component_pre_recapture_hook=pressure_rho_hook,
            )
            if "launch" not in apmasscon_holder:
                raise RuntimeError(
                    "WSM6 bridge did not execute its apmasscon pre-component hook"
                )
            if "launch" not in pdefq_holder:
                raise RuntimeError(
                    "WSM6 bridge did not execute its pdefq pre-component hook"
                )
            if "result" not in rho_boundary_holder:
                raise RuntimeError(
                    "WSM6 bridge did not execute its solve3 rho-boundary hook"
                )
            post_q_min, post_asq_sha = _validate_q_and_asq(
                cupy,
                _field(state.product, "q3d"),
                state.pdefq_asq,
                phase="post-WSM6",
            )
            _launch_solve3_momentum_current_to_a(state, cupy)
            tke_commit = _require_tke_commit_result(
                tke_commit_hook(cadence.step_index, state.stream)
            )
            tke_holder["launch"] = tke_commit
            _launch_solve3_scalar_current_to_a(state, cupy)
            _launch_solve3_density(state, cupy)
            closure_candidate = closure_refresh_hook(
                cadence.step_index, state.stream, tke_commit.receipt
            )
            closure_refresh, closure_receipt = _require_closure_refresh_result(
                closure_candidate,
                state=state,
                cadence=cadence,
                operator_sources=operator_sources,
                tke_commit=tke_commit,
            )
            closure_refresh_holder["result"] = closure_refresh
            closure_refresh_holder["receipt"] = closure_receipt
            qa = _field(state.product, "qa")
            if not _device_scalar_bool(
                cupy.array_equal(qa, _field(state.product, "q3d"))
            ):
                raise RuntimeError("solve3 qa publication is not raw-word equal to q3d")
            if _device_scalar_float(cupy.min(qa)) < 0.0:
                raise RuntimeError("solve3 published negative qa")
            ready_event = cupy.cuda.Event(disable_timing=True)
            ready_event.record(state.stream)
        apmasscon = apmasscon_holder["launch"]
        pdefq = pdefq_holder["launch"]
        rho_boundary = rho_boundary_holder["result"]
        tke_commit = tke_holder["launch"]
        closure_refresh = closure_refresh_holder["result"]
        completed_step_index = cadence.step_index + 1
        wsm6_receipt = getattr(wsm6, "receipt", None)
        if isinstance(wsm6_receipt, Mapping):
            wsm6_receipt = dict(wsm6_receipt)
        receipt = {
            "schema": CUDA_GM1_CHILD_POST_FINAL_RK_SCHEMA,
            "abi": CUDA_GM1_CHILD_POST_FINAL_RK_ABI,
            "model": "GM1",
            "completed_step_index": completed_step_index,
            "cadence": {
                name: getattr(cadence, name) for name in cadence.__dataclass_fields__
            },
            "geometry": {
                "ni": state.ni,
                "nj": state.nj,
                "nk": state.nk,
                "dx": state.dx,
                "dy": state.dy,
                "dz_reference": state.dz,
                "dt": state.dt,
            },
            "sequence": [
                "fresh final padded-to-product sync",
                "bridge remove/snapshot and pre-physics EOS",
                "apmasscon and pp3d/prs/rho rebuild",
                "pdefq",
                "WSM6",
                "solve3 ppx completion and pressure/rho boundary publication",
                "held-rate recapture",
                "solve3 momentum current-to-A",
                "solve3 TKE commit",
                "solve3 ppi/tha/qa current-to-A",
                "solve3 rr/rf density publication",
                "completed-step closure refresh",
            ],
            "final_dynamics_sync": dict(sync_receipt),
            "apmasscon": {
                "schema": CUDA_APMASSCON_SCHEMA,
                "mass1_f64": mass1_f64,
                "receipt": apmasscon.receipt,
            },
            "pdefq": {
                "schema": CUDA_WSM6_PDEFQ_SCHEMA,
                "cuda_source_sha256": CUDA_WSM6_PDEFQ_SOURCE_SHA256,
                "receipt": pdefq.receipt,
                "post_q3d_minimum": pre_q_min,
                "asq_sha256": pre_asq_sha,
            },
            "wsm6": {
                "bridge_abi": EQTSET2_WSM6_BRIDGE_ABI,
                "call_index": getattr(wsm6, "call_index", None),
                "receipt": wsm6_receipt,
                "post_q3d_minimum": post_q_min,
            },
            "solve3": {
                "authority_source_sha256": CM1_SOLVE3_SOURCE_SHA256,
                "qa_raw_copy_of_q3d": True,
                "ppi_raw_copy_of_pp3d": True,
                "tha_raw_copy_of_th3d": True,
                "momentum_current_to_a": ["u3d->ua", "v3d->va", "w3d->wa"],
                "tke_commit": dict(tke_commit.receipt),
                "source_order": (
                    "ppx/rho BC -> held recapture -> momentum A -> TKE commit -> "
                    "ppi/tha/qa A -> rr/rf"
                ),
                "rr_rf_rebuilt": True,
                "rho_boundary": dict(rho_boundary.receipt),
            },
            "closure_refresh": dict(closure_refresh_holder["receipt"]),
            "postconditions": {
                "q3d_nonnegative": True,
                "qa_nonnegative": True,
                "asq_finite_signed": True,
                "asq_sha256": post_asq_sha,
            },
            "operators": {
                "manifest_sha256": operator_manifest_sha256,
                "required_dependency_pins": dict(
                    child_post_final_rk_operator_source_pins()
                ),
                "count": len(operator_sources),
            },
            "synchronization": {
                "device_id": state.device_id,
                "stream_pointer": state.stream_pointer,
                "same_stream": True,
                "final_dynamics_sync_token_sha256": sync_token,
                "ready_event_recorded": True,
                "atmospheric_host_copy": False,
                "validation_host_copy": "six f64 asq words and scalar reductions only",
            },
            "claim_limit": _CLAIM_LIMIT,
        }
        _canonical_json_bytes(receipt)
        result = CudaGm1ChildPostFinalRkResult(
            ready_event=ready_event,
            apmasscon=apmasscon,
            pdefq=pdefq,
            wsm6=wsm6,
            rho_boundary=rho_boundary,
            tke_commit=tke_commit,
            closure_refresh=closure_refresh,
            final_dynamics_sync_token_sha256=sync_token,
            completed_step_index=completed_step_index,
            stream_pointer=state.stream_pointer,
            operator_manifest_sha256=operator_manifest_sha256,
            receipt=MappingProxyType(receipt),
        )
        state.next_step_index = completed_step_index
        state.completed_transitions += 1
        state.last_final_dynamics_sync_token_sha256 = sync_token
        state.last_result = result
        state.in_flight = False
        return result
    except Exception:
        state.in_flight = False
        state.poisoned = True
        raise


__all__ = (
    "CM1_SOLVE3_SOURCE_SHA256",
    "CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI",
    "CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY",
    "CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA",
    "CUDA_GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256",
    "CUDA_GM1_CHILD_EQTSET2_WSM6_OPERATOR_KEY",
    "CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI",
    "CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY",
    "CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA",
    "CUDA_GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256",
    "CUDA_GM1_CHILD_POST_FINAL_RK_ABI",
    "CUDA_GM1_CHILD_POST_FINAL_RK_OPERATOR_KEY",
    "CUDA_GM1_CHILD_POST_FINAL_RK_SCHEMA",
    "CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI",
    "CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY",
    "CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA",
    "CUDA_GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256",
    "CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY",
    "CudaGm1ChildPostFinalRkResult",
    "CudaGm1ChildPostFinalRkState",
    "GM1_BASELINE_OPERATOR_KEYS",
    "Gm1ChildPostFinalRkCadence",
    "child_post_final_rk_metadata",
    "child_post_final_rk_metadata_sha256",
    "child_post_final_rk_operator_source_pins",
    "create_cuda_gm1_child_post_final_rk_state",
    "launch_cuda_gm1_child_post_final_rk",
    "require_child_post_final_rk_cadence",
    "require_child_post_final_rk_operator_sources",
)
