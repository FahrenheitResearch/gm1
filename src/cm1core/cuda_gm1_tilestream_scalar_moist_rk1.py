"""Exact tile-streamed CM1 r21.1 theta/moisture RK1 seam.

CM1 advances theta and all six WSM6 moisture carriers after the RK1 acoustic
solve has produced two distinct live products: density-weighted mass fluxes
and the eqtset-2 ``t11*thterm`` contribution.  This module joins those typed
products to the already admitted solve1 held tendencies, evaluates stretched
fifth-order scalar advection, and performs the exact A-anchored RK1 compose.

The mutable physical carrier is updated through the sealed old-value ring
lifecycle.  A tile may publish only its unique ``owned_rect('mass')`` after
the CUDA event completes.  The resulting token binds physical ``th3d`` and
``q3d`` *before* the separate CM1 ``bcp/bcs`` boundary phase.  It deliberately
does not claim pressure/density, TKE, RK2/RK3, global pressure tiling, a full
forecast, or a runnable five-metre meteorological product.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_stretched_scalar import (
    GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256,
    compile_cuda_gm1_child_stretched_scalar_advection,
    launch_cuda_gm1_child_stretched_scalar_advection5,
)
from .cuda_gm1_tilestream import (
    CM1_OPEN_NGXY,
    Gm1CanonicalHostCarrierStore,
    Gm1TileLocalSourceBinding,
    Gm1TilePlan,
    Gm1TileSpec,
    fill_gm1_tile_local_source,
    tile_host_buffer_shapes,
)
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from .cuda_gm1_tilestream_scalar_post_sound_getdiv import (
    CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA,
    Gm1TiledScalarThetaDivergenceToken,
)
from .cuda_gm1_tilestream_solve1_scalar_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS,
    Gm1TiledSolve1ScalarHeldToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_scalar_moist_rk import (
    CM1_SCALAR_MOIST_SPECIES,
    CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
    CUDA_SCALAR_MOIST_RK_THREADS,
    CompiledScalarMoistRKKernels,
    compile_cuda_scalar_moist_rk,
    moisture_weps_f64,
    scalar_moist_rk_stage_timestep_f32,
    theta_weps_f64,
)


CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCHEMA = (
    "gm1.cuda-tilestream-scalar-moist-rk1/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SOURCE_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk1-tile-source-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCATTER_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk1-owned-scatter/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk1-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK1"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_ADV_SHA256 = (
    "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
)
AUTHORITY_ADV_ROUTINES_SHA256 = (
    "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1277-1343",
    "src/solve2.F:1344-1427",
    "src/solve2.F:1430-1533",
    "src/solve2.F:1664-1705",
    "src/adv.F:535-659",
    "src/adv_routines.F:204-430,2359-2557",
)
SCALAR_ORDER = ("theta", *CM1_SCALAR_MOIST_SPECIES)
SOURCE_ORDER = (
    "live solve1 held thten1/qten",
    "global RK1 sound returns small-step-averaged velocities",
    "tile-local post-sound getdiv emits t11*thterm",
    "tile-local post-acoustic getdivx emits rru/rrv/rrw/divx",
    "theta stretched advs then A+dttmp*(held-t11*thterm+advection)",
    "qv/qc/qr/qi/qs/qg stretched advs then A+dttmp*(held+advection)",
    "publish every physical cell before the separate bcp/bcs boundary phase",
)
GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT = (
    "one CM1 r21.1 serial terrain/stretch_z=1 theta and six-species moisture "
    "RK1 physical update over exact fixed-halo tiles, consuming typed live "
    "solve1 held, post-sound theta-divergence, and same-stage post-acoustic "
    "mass-flux tokens with old-value ring restoration and exact owned-rectangle "
    "scatter. Scalar bcp/bcs, pressure/density, TKE, RK2/RK3, global psolver "
    "tilestreaming, MPI, open-history production, full forecast, and the "
    "five-metre meteorological product remain refused"
)

_BASE_HOST_NAMES = tuple(
    tile_host_buffer_shapes(tile_nx=6, tile_ny=6, nz=5)
)
_DEVICE_INPUT_NAMES = (
    "scalar",
    "rru",
    "rrv",
    "rrw",
    "divx",
    "c1",
    "c2",
    "rr0",
    "gz",
    "a",
    "held",
    "theta_divergence_term",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _inventory_sha(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "little"))
        digest.update(encoded)
        digest.update(arrays[name].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def scalar_moist_rk1_host_tile_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact pinned host inventory for one scalar-family tile pass."""

    shapes = dict(
        tile_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nk)
    )
    cnx = _positive_int("tile_nx", tile_nx) + 6
    cny = _positive_int("tile_ny", tile_ny) + 6
    physical = (_positive_int("nk", nk), cny, cnx)
    shapes.update(
        {
            "a": physical,
            "held": physical,
            "theta_divergence_term": physical,
            "current": physical,
        }
    )
    return MappingProxyType(shapes)


def scalar_moist_rk1_device_tile_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact device inventory: numerical inputs, current output, and scratch."""

    host = scalar_moist_rk1_host_tile_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    shapes = {name: host[name] for name in _DEVICE_INPUT_NAMES}
    shapes.update(
        {
            "current": host["current"],
            "advection": host["tendency"],
            "fx": host["rru"],
            "fy": host["rrv"],
            "fz": host["rrw"],
        }
    )
    return MappingProxyType(shapes)


def _require_tokens(
    held_token: object,
    flux_token: object,
    theta_token: object,
    *,
    plan: Gm1TilePlan | None = None,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if not isinstance(held_token, Gm1TiledSolve1ScalarHeldToken):
        raise TypeError("held_token must be the exact solve1 scalar-held token")
    if not isinstance(flux_token, Gm1TiledSolve2PostAcousticFluxToken):
        raise TypeError("flux_token must be the exact post-acoustic mass-flux token")
    if not isinstance(theta_token, Gm1TiledScalarThetaDivergenceToken):
        raise TypeError("theta_token must be the exact theta-divergence token")
    held, flux, theta = (
        held_token.receipt,
        flux_token.receipt,
        theta_token.receipt,
    )
    if (
        held.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_SCHEMA
        or held.get("status")
        != CUDA_GM1_TILESTREAM_SOLVE1_SCALAR_HELD_TOKEN_STATUS
        or held.get("completed_nrk") != 0
        or held.get("operation_token_sha256") != _operation_token(held)
        or flux.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or flux.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or flux.get("nrk") != 1
        or flux.get("operation_token_sha256") != _operation_token(flux)
        or theta.get("schema")
        != CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA
        or theta.get("status") != "ADMITTED_LIVE_RK1_SCALAR_THETA_DIVERGENCE"
        or theta.get("nrk") != 1
        or theta.get("operation_token_sha256") != _operation_token(theta)
    ):
        raise ValueError("scalar RK1 prerequisite token identity changed")
    streams = {
        held.get("stream_pointer"),
        flux.get("stream_pointer"),
        theta.get("stream_pointer"),
    }
    if len(streams) != 1 or not isinstance(next(iter(streams)), int):
        raise ValueError("scalar RK1 prerequisite streams differ")
    if (
        theta.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or theta.get("post_sound_velocity_token_sha256")
        != flux.get("post_sound_velocity_token_sha256")
    ):
        raise ValueError("theta divergence and mass flux belong to different sound stages")
    _lower_sha("stage_identity_sha256", flux.get("stage_identity_sha256"))
    if plan is not None:
        if not isinstance(plan, Gm1TilePlan):
            raise TypeError("plan must be Gm1TilePlan")
        if any(
            receipt.get("tile_plan_sha256") != plan.ownership_sha256
            for receipt in (held, flux, theta)
        ):
            raise ValueError("scalar RK1 prerequisite tile plans differ")
    return held, flux, theta


def _require_host_output(
    np: Any,
    outputs: Mapping[str, Any],
    *,
    expected: Mapping[str, tuple[int, ...]],
    receipt: Mapping[str, object],
    label: str,
) -> dict[str, Any]:
    if not isinstance(outputs, Mapping) or set(outputs) != set(expected):
        raise ValueError(f"{label} host output inventory changed")
    receipt_shapes = receipt.get("output_shapes")
    receipt_hashes = receipt.get("output_sha256")
    if not isinstance(receipt_shapes, Mapping) or not isinstance(receipt_hashes, Mapping):
        raise ValueError(f"{label} token lacks exact output bindings")
    checked: dict[str, Any] = {}
    for name, shape in expected.items():
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{label} {name} must be finite host C-<f4 {shape}")
        raw_sha = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if receipt_shapes.get(name) != list(shape) or receipt_hashes.get(name) != raw_sha:
            raise ValueError(f"{label} {name} bytes differ from their live token")
        checked[name] = value
    return checked


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarMoistRk1HostSources:
    held_outputs: Mapping[str, Any]
    flux_outputs: Mapping[str, Any]
    theta_outputs: Mapping[str, Any]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarMoistRk1TileSourceToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledScalarMoistRk1Launch:
    ready_event: Any
    current: Any
    advection: Any
    upstream: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarMoistRk1Token:
    receipt: Mapping[str, object]


def bind_gm1_tiled_scalar_moist_rk1_host_sources(
    held_token: Gm1TiledSolve1ScalarHeldToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledScalarThetaDivergenceToken,
    *,
    plan: Gm1TilePlan,
    held_outputs: Mapping[str, Any],
    flux_outputs: Mapping[str, Any],
    theta_outputs: Mapping[str, Any],
) -> Gm1TiledScalarMoistRk1HostSources:
    """Verify every full-domain prerequisite byte once before tile transport."""

    held, flux, theta = _require_tokens(
        held_token, flux_token, theta_token, plan=plan
    )
    if (plan.nx, plan.ny) != (300, 300):
        raise ValueError("scalar RK1 host binding requires the admitted 300 grid")
    np = importlib.import_module("numpy")
    held_checked = _require_host_output(
        np,
        held_outputs,
        expected={"thten1": (75, 300, 300), "qten": (6, 75, 300, 300)},
        receipt=held,
        label="solve1 held",
    )
    flux_checked = _require_host_output(
        np,
        flux_outputs,
        expected={
            "rru": (75, 300, 301),
            "rrv": (75, 301, 300),
            "rrw": (76, 300, 300),
            "divx": (75, 300, 300),
        },
        receipt=flux,
        label="post-acoustic flux",
    )
    theta_checked = _require_host_output(
        np,
        theta_outputs,
        expected={
            "divergence": (75, 300, 300),
            "theta_divergence_term": (75, 300, 300),
        },
        receipt=theta,
        label="theta divergence",
    )
    receipt: dict[str, object] = {
        "schema": "gm1.tilestream-scalar-moist-rk1-host-sources/v1",
        "status": "ADMITTED_EXACT_LIVE_RK1_SCALAR_HOST_SOURCES",
        "solve1_scalar_held_token_sha256": held["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": flux["stream_pointer"],
        "tile_plan_sha256": plan.ownership_sha256,
        "output_sha256": {
            "held": dict(held["output_sha256"]),
            "flux": dict(flux["output_sha256"]),
            "theta": dict(theta["output_sha256"]),
        },
        "full_domain_bytes_rehashed": True,
        "claim_limit": GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarMoistRk1HostSources(
        MappingProxyType(held_checked),
        MappingProxyType(flux_checked),
        MappingProxyType(theta_checked),
        MappingProxyType(receipt),
    )


def _require_sources(
    sources: object,
    held: Mapping[str, object],
    flux: Mapping[str, object],
    theta: Mapping[str, object],
    plan: Gm1TilePlan,
) -> Gm1TiledScalarMoistRk1HostSources:
    if not isinstance(sources, Gm1TiledScalarMoistRk1HostSources):
        raise TypeError("sources must be the exact scalar RK1 host-source binding")
    receipt = sources.receipt
    if (
        receipt.get("schema") != "gm1.tilestream-scalar-moist-rk1-host-sources/v1"
        or receipt.get("status") != "ADMITTED_EXACT_LIVE_RK1_SCALAR_HOST_SOURCES"
        or receipt.get("solve1_scalar_held_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("theta_divergence_token_sha256")
        != theta.get("operation_token_sha256")
        or receipt.get("tile_plan_sha256") != plan.ownership_sha256
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scalar RK1 host-source binding changed")
    return sources


def fill_gm1_tiled_scalar_moist_rk1_host_tile(
    binding: Gm1TileLocalSourceBinding,
    sources: Gm1TiledScalarMoistRk1HostSources,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
    *,
    scalar: str,
) -> Mapping[str, object]:
    """Fill one pre-boundary RK1 tile from exact physical live sources."""

    if scalar not in SCALAR_ORDER or binding.scalar != scalar:
        raise ValueError("scalar and tile-source binding differ")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    expected = scalar_moist_rk1_host_tile_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=75
    )
    if not isinstance(target, Mapping) or set(target) != set(expected):
        raise ValueError("scalar RK1 host tile inventory changed")
    np = importlib.import_module("numpy")
    for name, shape in expected.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"host scalar RK1 {name} must be writable C-<f4 {shape}")
    base = fill_gm1_tile_local_source(
        binding,
        spec,
        {name: target[name] for name in _BASE_HOST_NAMES},
    )
    ng, y0, x0, cny, cnx = (
        CM1_OPEN_NGXY,
        spec.cj0,
        spec.ci0,
        spec.cny,
        spec.cnx,
    )
    flux = sources.flux_outputs
    target["rru"].fill(np.float32(0.0))
    target["rrv"].fill(np.float32(0.0))
    target["rrw"].fill(np.float32(0.0))
    np.copyto(
        target["rru"][1:76, ng : ng + cny, ng : ng + cnx + 1],
        flux["rru"][:, y0 : y0 + cny, x0 : x0 + cnx + 1],
    )
    np.copyto(
        target["rrv"][1:76, ng : ng + cny + 1, ng : ng + cnx],
        flux["rrv"][:, y0 : y0 + cny + 1, x0 : x0 + cnx],
    )
    np.copyto(
        target["rrw"][1:77, ng : ng + cny, ng : ng + cnx],
        flux["rrw"][:, y0 : y0 + cny, x0 : x0 + cnx],
    )
    np.copyto(
        target["divx"],
        flux["divx"][:, y0 : y0 + cny, x0 : x0 + cnx],
    )
    canonical = binding.canonical.arrays
    if scalar == "theta":
        a = canonical["tha"][:, y0 : y0 + cny, x0 : x0 + cnx]
        held = sources.held_outputs["thten1"][
            :, y0 : y0 + cny, x0 : x0 + cnx
        ]
        divterm = sources.theta_outputs["theta_divergence_term"][
            :, y0 : y0 + cny, x0 : x0 + cnx
        ]
    else:
        species = CM1_SCALAR_MOIST_SPECIES.index(scalar)
        a = canonical["qa"][species, :, y0 : y0 + cny, x0 : x0 + cnx]
        held = sources.held_outputs["qten"][
            species, :, y0 : y0 + cny, x0 : x0 + cnx
        ]
        divterm = np.zeros_like(target["theta_divergence_term"])
    np.copyto(target["a"], a)
    np.copyto(target["held"], held)
    np.copyto(target["theta_divergence_term"], divterm)
    target["current"].fill(np.float32(0.0))
    target["tendency"].fill(np.float32(0.0))
    receipt: dict[str, object] = {
        "schema": "gm1.tilestream-scalar-moist-rk1-host-fill/v1",
        "status": "FILLED_LIVE_PRE_BOUNDARY_RK1_SCALAR_TILE",
        "scalar": scalar,
        "tile": spec.record(),
        "canonical_source_identity_sha256": binding.source_identity_sha256,
        "host_sources_token_sha256": sources.receipt["operation_token_sha256"],
        "base_fill_sha256": hashlib.sha256(_canonical(base)).hexdigest(),
        "same_stage_mass_flux_overlaid": True,
        "theta_divergence_applied_only_to_theta": True,
        "ring_patch_applied": False,
        "source_order": list(SOURCE_ORDER),
        "claim_limit": GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return MappingProxyType(receipt)


def bind_gm1_tiled_scalar_moist_rk1_tile_source(
    held_token: Gm1TiledSolve1ScalarHeldToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledScalarThetaDivergenceToken,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    scalar: str,
    fill_receipt: Mapping[str, object],
    ring_save_receipt: Mapping[str, object],
    ring_patch_receipt: Mapping[str, object],
    device_h2d_inventory_sha256: str | None = None,
) -> Gm1TiledScalarMoistRk1TileSourceToken:
    """Hash-bind one patched old-stage host tile and its exact H2D bytes."""

    held, flux, theta = _require_tokens(held_token, flux_token, theta_token)
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if scalar not in SCALAR_ORDER:
        raise ValueError("scalar must be theta,qv,qc,qr,qi,qs,qg")
    expected = scalar_moist_rk1_device_tile_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=75
    )
    input_shapes = {name: expected[name] for name in _DEVICE_INPUT_NAMES}
    if not isinstance(arrays, Mapping) or set(arrays) != set(input_shapes):
        raise ValueError("scalar RK1 H2D input inventory changed")
    np = importlib.import_module("numpy")
    for name, shape in input_shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"host scalar RK1 {name} must be finite C-<f4 {shape}")
    if (
        not isinstance(fill_receipt, Mapping)
        or fill_receipt.get("schema")
        != "gm1.tilestream-scalar-moist-rk1-host-fill/v1"
        or fill_receipt.get("scalar") != scalar
        or fill_receipt.get("tile") != spec.record()
        or fill_receipt.get("operation_token_sha256")
        != _operation_token(fill_receipt)
        or not isinstance(ring_save_receipt, Mapping)
        or ring_save_receipt.get("schema")
        != "gm1.hybrid-tilestream-scalar-ring-save/v1"
        or ring_save_receipt.get("tile") != spec.index
        or not isinstance(ring_patch_receipt, Mapping)
        or ring_patch_receipt.get("schema")
        != "gm1.hybrid-tilestream-scalar-ring-patch/v1"
        or ring_patch_receipt.get("tile") != spec.index
        or ring_patch_receipt.get("source_stage_values_restored") is not True
        or ring_patch_receipt.get("canonical_state_mutated_by_patch") is not False
    ):
        raise ValueError("scalar RK1 fill or ring lifecycle changed")
    host_sha = _inventory_sha(arrays)
    if device_h2d_inventory_sha256 is not None:
        if _lower_sha(
            "device_h2d_inventory_sha256", device_h2d_inventory_sha256
        ) != host_sha:
            raise ValueError("scalar RK1 H2D bytes differ from the patched host tile")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SOURCE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILE_SCALAR_MOIST_RK1_SOURCE",
        "completed_nrk": 0,
        "target_nrk": 1,
        "scalar": scalar,
        "tile": spec.record(),
        "solve1_scalar_held_token_sha256": held["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": flux["stream_pointer"],
        "fill_receipt_sha256": fill_receipt["operation_token_sha256"],
        "ring_save_receipt_sha256": hashlib.sha256(
            _canonical(ring_save_receipt)
        ).hexdigest(),
        "ring_patch_receipt_sha256": hashlib.sha256(
            _canonical(ring_patch_receipt)
        ).hexdigest(),
        "ring_schedule_sha256": ring_save_receipt.get("ring_schedule_sha256"),
        "host_inventory_sha256": host_sha,
        "device_h2d_inventory_sha256": device_h2d_inventory_sha256,
        "h2d_raw_bit_verified": device_h2d_inventory_sha256 is not None,
        "source_halo": spec.halo,
        "source_stage_values_restored": True,
        "claim_limit": GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarMoistRk1TileSourceToken(MappingProxyType(receipt))


def _require_source_token(
    token: object,
    held: Mapping[str, object],
    flux: Mapping[str, object],
    theta: Mapping[str, object],
    spec: Gm1TileSpec,
    scalar: str,
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledScalarMoistRk1TileSourceToken):
        raise TypeError("source_token must be the exact scalar RK1 tile source")
    receipt = token.receipt
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SOURCE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILE_SCALAR_MOIST_RK1_SOURCE"
        or receipt.get("completed_nrk") != 0
        or receipt.get("target_nrk") != 1
        or receipt.get("scalar") != scalar
        or receipt.get("tile") != spec.record()
        or receipt.get("solve1_scalar_held_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("theta_divergence_token_sha256")
        != theta.get("operation_token_sha256")
        or receipt.get("h2d_raw_bit_verified") is not True
        or receipt.get("host_inventory_sha256")
        != receipt.get("device_h2d_inventory_sha256")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scalar RK1 tile source identity or transport changed")
    return receipt


def launch_cuda_gm1_tiled_scalar_moist_rk1(
    held_token: Gm1TiledSolve1ScalarHeldToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledScalarThetaDivergenceToken,
    source_token: Gm1TiledScalarMoistRk1TileSourceToken,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    scalar: str,
    vertical: Any,
    dt: float,
    dx: float,
    dy: float,
    stream: Any,
    advection_compiled: Any | None = None,
    rk_compiled: CompiledScalarMoistRKKernels | None = None,
) -> CudaGm1TiledScalarMoistRk1Launch:
    """Enqueue one scalar family advection and exact pre-boundary RK1 compose."""

    held, flux, theta = _require_tokens(held_token, flux_token, theta_token)
    source = _require_source_token(
        source_token, held, flux, theta, spec, scalar
    )
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    if int(stream.ptr) != held.get("stream_pointer"):
        raise ValueError("scalar RK1 left its live prerequisite CUDA stream")
    for name, value in (("dt", dt), ("dx", dx), ("dy", dy)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    cupy = importlib.import_module("cupy")
    expected = scalar_moist_rk1_device_tile_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=75
    )
    if not isinstance(arrays, Mapping) or set(arrays) != set(expected):
        raise ValueError("scalar RK1 device tile inventory changed")
    checked: dict[str, Any] = {}
    for name, shape in expected.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"device scalar RK1 {name} must be C-f32 {shape}")
        checked[name] = value
    pointers = [int(value.data.ptr) for value in checked.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("scalar RK1 device arrays must not alias")
    advection_compiled = (
        advection_compiled or compile_cuda_gm1_child_stretched_scalar_advection()
    )
    rk_compiled = rk_compiled or compile_cuda_scalar_moist_rk()
    targets = {
        advection_compiled.target_arch,
        rk_compiled.target_arch,
        vertical.target_arch,
    }
    if len(targets) != 1:
        raise ValueError("scalar RK1 CUDA dependencies target different architectures")
    words = 75 * spec.cny * spec.cnx
    block = (CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
    grid = (
        (words + CUDA_SCALAR_MOIST_RK_THREADS - 1)
        // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    dttmp = scalar_moist_rk_stage_timestep_f32(float(dt), 1)
    weps = theta_weps_f64() if scalar == "theta" else moisture_weps_f64(
        CM1_SCALAR_MOIST_SPECIES.index(scalar)
    )
    with stream:
        checked["advection"].fill(cupy.float32(0.0))
        checked["current"].fill(cupy.float32(0.0))
        upstream = launch_cuda_gm1_child_stretched_scalar_advection5(
            checked["scalar"],
            {name: checked[name] for name in ("rru", "rrv", "rrw")},
            {name: checked[name] for name in ("c1", "c2", "rr0", "gz")},
            checked["divx"],
            checked["advection"],
            {
                axis: checked[name]
                for axis, name in (("x", "fx"), ("y", "fy"), ("z", "fz"))
            },
            vertical,
            ni=spec.cnx,
            nj=spec.cny,
            nk=75,
            nrk=1,
            dx=float(dx),
            dy=float(dy),
            weps=weps,
            stream=stream,
            compiled=advection_compiled,
            allow_compact_validation=True,
        )
        if scalar == "theta":
            rk_compiled.kernels[2](
                grid,
                block,
                (
                    checked["a"],
                    checked["held"],
                    checked["theta_divergence_term"],
                    checked["advection"],
                    checked["current"],
                    cupy.uint64(words),
                    cupy.float32(dttmp),
                ),
                stream=stream,
            )
        else:
            rk_compiled.kernels[3](
                grid,
                block,
                (
                    checked["a"],
                    checked["held"],
                    checked["advection"],
                    checked["current"],
                    cupy.uint64(words),
                    cupy.float32(dttmp),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCHEMA,
        "status": "ENQUEUED_LIVE_TILE_SCALAR_MOIST_RK1",
        "completed_nrk": 1,
        "scalar": scalar,
        "tile": spec.record(),
        "source_token_sha256": source["operation_token_sha256"],
        "solve1_scalar_held_token_sha256": held["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": int(stream.ptr),
        "dttmp_f32": float(dttmp),
        "weps_f64": weps,
        "same_stream_after_live_sound_products": True,
        "old_value_ring_restored": True,
        "scatter_disposition": "unique physical owned mass rectangle only",
        "boundary_disposition": "pre-bcp/bcs physical current; no halo mutation",
        "operator_source_sha256": {
            "stretched_scalar_advection": (
                GM1_CHILD_STRETCHED_SCALAR_ADVECTION_SOURCE_SHA256
            ),
            "scalar_moist_rk": CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
        },
        "source_order": list(SOURCE_ORDER),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "adv_routines_sha256": AUTHORITY_ADV_ROUTINES_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledScalarMoistRk1Launch(
        event,
        checked["current"],
        checked["advection"],
        upstream,
        MappingProxyType(receipt),
    )


def scatter_gm1_tiled_scalar_moist_rk1(
    canonical: Gm1CanonicalHostCarrierStore,
    spec: Gm1TileSpec,
    tile_current: Any,
    *,
    scalar: str,
    source_token: Gm1TiledScalarMoistRk1TileSourceToken,
) -> Mapping[str, object]:
    """Publish one exact pre-boundary physical owned rectangle."""

    if not isinstance(canonical, Gm1CanonicalHostCarrierStore):
        raise TypeError("canonical must be Gm1CanonicalHostCarrierStore")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if scalar not in SCALAR_ORDER:
        raise ValueError("scalar must be theta,qv,qc,qr,qi,qs,qg")
    receipt = source_token.receipt if isinstance(
        source_token, Gm1TiledScalarMoistRk1TileSourceToken
    ) else None
    if (
        receipt is None
        or receipt.get("scalar") != scalar
        or receipt.get("tile") != spec.record()
        or receipt.get("h2d_raw_bit_verified") is not True
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scatter source is not the exact admitted tile")
    np = importlib.import_module("numpy")
    expected = (canonical.nz, spec.cny, spec.cnx)
    if (
        not isinstance(tile_current, np.ndarray)
        or tile_current.dtype != np.dtype("<f4")
        or tile_current.shape != expected
        or not tile_current.flags.c_contiguous
        or not bool(np.all(np.isfinite(tile_current)))
    ):
        raise ValueError(f"tile_current must be finite host C-<f4 {expected}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    ty0, tx0 = y0 - spec.cj0, x0 - spec.ci0
    ty1, tx1 = ty0 + (y1 - y0), tx0 + (x1 - x0)
    if scalar == "theta":
        destination = canonical.arrays["th3d"][:, y0:y1, x0:x1]
    else:
        species = CM1_SCALAR_MOIST_SPECIES.index(scalar)
        destination = canonical.arrays["q3d"][species, :, y0:y1, x0:x1]
    np.copyto(destination, tile_current[:, ty0:ty1, tx0:tx1])
    record: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCATTER_SCHEMA,
        "status": "PUBLISHED_PRE_BOUNDARY_PHYSICAL_RK1_OWNERSHIP",
        "scalar": scalar,
        "tile": spec.record(),
        "source_token_sha256": receipt["operation_token_sha256"],
        "owned_yx": [y0, y1, x0, x1],
        "published_words": int(destination.size),
        "unique_full_physical_ownership": True,
        "scalar_bcp_or_bcs_performed": False,
        "padded_halos_published": False,
    }
    record["operation_token_sha256"] = _operation_token(record)
    return MappingProxyType(record)


def finish_cuda_gm1_tiled_scalar_moist_rk1(
    held_token: Gm1TiledSolve1ScalarHeldToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledScalarThetaDivergenceToken,
    launches: Mapping[str, Sequence[CudaGm1TiledScalarMoistRk1Launch]],
    scatters: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    plan: Gm1TilePlan,
    outputs: Mapping[str, Any],
) -> Gm1TiledScalarMoistRk1Token:
    """Admit all seven exact physical RK1 publications before boundaries."""

    held, flux, theta = _require_tokens(
        held_token, flux_token, theta_token, plan=plan
    )
    if (plan.nx, plan.ny) != (300, 300):
        raise ValueError("scalar RK1 completion requires the admitted 300 grid")
    if tuple(launches) != SCALAR_ORDER or tuple(scatters) != SCALAR_ORDER:
        raise ValueError("scalar RK1 family inventory or order changed")
    launch_tokens: dict[str, list[str]] = {}
    scatter_tokens: dict[str, list[str]] = {}
    streams: set[int] = set()
    owned_words = 0
    for scalar in SCALAR_ORDER:
        family_launches = launches[scalar]
        family_scatters = scatters[scalar]
        if len(family_launches) != plan.ntiles or len(family_scatters) != plan.ntiles:
            raise ValueError(f"scalar RK1 {scalar} requires every exact tile")
        launch_tokens[scalar], scatter_tokens[scalar] = [], []
        for spec, launch, scatter in zip(
            plan.specs, family_launches, family_scatters, strict=True
        ):
            if not isinstance(launch, CudaGm1TiledScalarMoistRk1Launch):
                raise TypeError("scalar RK1 completion requires typed CUDA launches")
            receipt = launch.receipt
            if (
                receipt.get("schema") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCHEMA
                or receipt.get("status") != "ENQUEUED_LIVE_TILE_SCALAR_MOIST_RK1"
                or receipt.get("completed_nrk") != 1
                or receipt.get("scalar") != scalar
                or receipt.get("tile") != spec.record()
                or receipt.get("solve1_scalar_held_token_sha256")
                != held.get("operation_token_sha256")
                or receipt.get("post_acoustic_flux_token_sha256")
                != flux.get("operation_token_sha256")
                or receipt.get("theta_divergence_token_sha256")
                != theta.get("operation_token_sha256")
                or receipt.get("operation_token_sha256")
                != _operation_token(receipt)
            ):
                raise ValueError("scalar RK1 tile identity or cadence changed")
            if (
                not isinstance(scatter, Mapping)
                or scatter.get("schema")
                != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCATTER_SCHEMA
                or scatter.get("status")
                != "PUBLISHED_PRE_BOUNDARY_PHYSICAL_RK1_OWNERSHIP"
                or scatter.get("scalar") != scalar
                or scatter.get("tile") != spec.record()
                or scatter.get("source_token_sha256")
                != receipt.get("source_token_sha256")
                or scatter.get("operation_token_sha256")
                != _operation_token(scatter)
            ):
                raise ValueError("scalar RK1 owned scatter identity changed")
            y0, y1, x0, x1 = spec.owned_rect("mass")
            words = 75 * (y1 - y0) * (x1 - x0)
            if scatter.get("published_words") != words:
                raise ValueError("scalar RK1 scatter word count changed")
            owned_words += words
            streams.add(int(receipt["stream_pointer"]))
            launch_tokens[scalar].append(str(receipt["operation_token_sha256"]))
            scatter_tokens[scalar].append(str(scatter["operation_token_sha256"]))
    expected_words = 7 * 75 * 300 * 300
    if owned_words != expected_words:
        raise ValueError("scalar RK1 physical scatter ownership is not exact once")
    if len(streams) != 1 or next(iter(streams)) != held.get("stream_pointer"):
        raise ValueError("scalar RK1 launches left their prerequisite stream")
    np = importlib.import_module("numpy")
    expected_outputs = {
        "th3d": (75, 300, 300),
        "q3d": (6, 75, 300, 300),
    }
    if not isinstance(outputs, Mapping) or tuple(outputs) != ("th3d", "q3d"):
        raise ValueError("scalar RK1 output inventory or order changed")
    output_sha: dict[str, str] = {}
    for name, shape in expected_outputs.items():
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{name} must be finite host C-<f4 {shape}")
        output_sha[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS,
        "completed_nrk": 1,
        "stream_pointer": next(iter(streams)),
        "solve1_scalar_held_token_sha256": held["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": {
            scalar: hashlib.sha256(_canonical(launch_tokens[scalar])).hexdigest()
            for scalar in SCALAR_ORDER
        },
        "scatter_operation_tokens_sha256": {
            scalar: hashlib.sha256(_canonical(scatter_tokens[scalar])).hexdigest()
            for scalar in SCALAR_ORDER
        },
        "host_scatter_exact_once": True,
        "output_sha256": output_sha,
        "output_shapes": {
            name: list(shape) for name, shape in expected_outputs.items()
        },
        "output_words": expected_words,
        "boundary_disposition": "physical pre-bcp/bcs th3d/q3d",
        "padded_halos_are_outputs": False,
        "source_order": list(SOURCE_ORDER),
        "ready_for": (
            "typed RK1 pressure/density plus bcp(rho,pp3d) and "
            "bcs(th3d,q3d) boundary seam only"
        ),
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "frozen or foreign solve1 held tendencies",
            "mass flux or theta divergence from another sound stage",
            "missing old-value ring save/patch or partial physical scatter",
            "scalar bcp/bcs or open-history mutation inside RK1",
            "RK2/RK3 without fresh same-stage prerequisites",
            "global psolver tilestream claim",
            "full RK, forecast, or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "adv_routines_sha256": AUTHORITY_ADV_ROUTINES_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarMoistRk1Token(MappingProxyType(receipt))


def five_meter_scalar_moist_rk1_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    """Return exact local memory and the unresolved 5 m live prerequisites."""

    host_shapes = scalar_moist_rk1_host_tile_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    device_shapes = scalar_moist_rk1_device_tile_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    host_bytes = 4 * sum(math.prod(shape) for shape in host_shapes.values())
    device_bytes = 4 * sum(math.prod(shape) for shape in device_shapes.values())
    record: dict[str, object] = {
        "schema": "gm1.hybrid-tilestream-scalar-moist-rk1-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "compute_tile": [tile_ny + 6, tile_nx + 6],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "scalar_families_sequential": list(SCALAR_ORDER),
        "host_buffer_bytes_each": host_bytes,
        "double_pinned_host_buffer_bytes": 2 * host_bytes,
        "device_buffer_bytes_each": device_bytes,
        "double_device_buffer_bytes": 2 * device_bytes,
        "one_reusable_old_value_ring_per_scalar_family": True,
        "tile_streamable_after_live_prerequisites": True,
        "five_meter_solve1_scalar_held_token_available": False,
        "five_meter_post_acoustic_flux_token_available": False,
        "five_meter_theta_divergence_token_available": False,
        "scalar_boundaries_available": False,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ADV_ROUTINES_SHA256",
    "AUTHORITY_ADV_SHA256",
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SCATTER_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_SOURCE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS",
    "CudaGm1TiledScalarMoistRk1Launch",
    "GM1_TILESTREAM_SCALAR_MOIST_RK1_CLAIM_LIMIT",
    "Gm1TiledScalarMoistRk1HostSources",
    "Gm1TiledScalarMoistRk1TileSourceToken",
    "Gm1TiledScalarMoistRk1Token",
    "SCALAR_ORDER",
    "SOURCE_ORDER",
    "bind_gm1_tiled_scalar_moist_rk1_host_sources",
    "bind_gm1_tiled_scalar_moist_rk1_tile_source",
    "fill_gm1_tiled_scalar_moist_rk1_host_tile",
    "finish_cuda_gm1_tiled_scalar_moist_rk1",
    "five_meter_scalar_moist_rk1_inventory",
    "launch_cuda_gm1_tiled_scalar_moist_rk1",
    "scalar_moist_rk1_device_tile_shapes",
    "scalar_moist_rk1_host_tile_shapes",
    "scatter_gm1_tiled_scalar_moist_rk1",
)
