"""Tile-local public-ArWen column-microphysics boundary, generic over ptype.

This module begins after the admitted, source-ordered ``pdefq`` finish.  It
reconstructs only the scheme-ready thermodynamic carriers, invokes the
CM1-owned public ArWen component for the selected
:class:`~cm1core.arwen_column_component.ColumnComponentSpec`, and publishes
that component's transported scalars plus its persistence inventory.  ``thp``
is an ephemeral output that must be drained into the subsequent CM1
post-physics rebuild before a tile buffer is reused.

Every admitted column scheme owns complete vertical columns and has no
horizontal stencil, so a fixed-size edge tile executes pad columns that are
never scattered.

This is not the post-scheme CM1 EOS rebuild, ``qa`` publication, solve3,
pressure, boundary-history update, MPI reduction or a coupled RK step.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import engine_pin
from .arwen_column_component import (
    ARWEN_PUBLIC_REFLECTIVITY_CALL,
    ColumnComponentSpec,
)
from .engine_capacity import (
    capacity_level_bounds,
    capacity_profile_metadata,
    capacity_profile_sha256,
)


#: Per-component record identity.  A schema string is an artifact contract, so
#: each component carries its own rather than sharing one that would make two
#: different scalar axes indistinguishable to a reader.
TILESTREAM_RECORD_IDENTITY: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "wsm6-mp6": MappingProxyType({
        "schema": "gm1.cuda-tilestream-arwen-wsm6/v1",
        "abi": "gm1.cuda-tilestream-arwen-wsm6-f32/v1",
        "status": "ENQUEUED_TILE_LOCAL_PUBLIC_WSM6",
    }),
    "morrison-mp10": MappingProxyType({
        "schema": "gm1.cuda-tilestream-arwen-morrison/v1",
        "abi": "gm1.cuda-tilestream-arwen-morrison-f32/v1",
        "status": "ENQUEUED_TILE_LOCAL_PUBLIC_MORRISON",
    }),
    "thompson-mp8": MappingProxyType({
        "schema": "gm1.cuda-tilestream-arwen-thompson/v1",
        "abi": "gm1.cuda-tilestream-arwen-thompson-f32/v1",
        "status": "ENQUEUED_TILE_LOCAL_PUBLIC_THOMPSON",
    }),
    "thompson-aerosol-mp28": MappingProxyType({
        "schema": "gm1.cuda-tilestream-arwen-thompson-aerosol/v1",
        "abi": "gm1.cuda-tilestream-arwen-thompson-aerosol-f32/v1",
        "status": "ENQUEUED_TILE_LOCAL_PUBLIC_THOMPSON_AEROSOL",
    }),
})

#: The reflectivity formula each component publishes is its own scheme's.
#: WSM6's is fixed-intercept exponential rain, the temperature-dependent snow
#: intercept, fixed-intercept graupel-or-hail, and the Blahak melting-particle
#: integration.  It is not interchangeable with another scheme's, which is why
#: CM1 refuses ``output_dbz`` per ``ptype`` in ``param.F:1583-1603``; a
#: component absent here has no admitted formula and a due call is refused.
TILESTREAM_REFLECTIVITY_AUTHORITY: Mapping[str, str] = MappingProxyType({
    "wsm6-mp6": (
        "WRF v4.6.1 phys/module_mp_wsm6.F90:2275-2444 refl10cm_wsm6 and "
        "phys/module_mp_radar.F:74-190 radar_init, through the public ArWen seam"
    ),
})

#: The measured native-oracle receipt each component's execution can cite.
#: ``None`` means no independent oracle has been captured for that component
#: yet, and the record says so rather than borrowing another scheme's.
TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256: Mapping[str, str | None] = MappingProxyType({
    "wsm6-mp6": "cbecf65c42095031c4d7210ac0c0693daa49e2f2f4363e05efe2b36d7dd2a482",
    "morrison-mp10": None,
    "thompson-mp8": None,
    "thompson-aerosol-mp28": None,
})
#: The estate-delta audit that admitted the estate this record's bytes come
#: from: the 2.7.4 -> 2.8.0 audit that tools/audit_arwen_wsm6_estate_delta.py
#: regenerates.  It named the 2.5.8 document
#: (`cd3bf5048a9922f3a2b7507c7453c0d676159866df13274621427c8f03bf1603`) until
#: the 2.7.4 re-pin, and the 2.7.4 document
#: (`55dd76687aef8aaf9f37909248d342d2176020f46369eee6dc804e5765a59c6d`) until
#: the 2.8.0 re-pin; a receipt that cites a superseded audit tells an operator
#: the running bytes were admitted by a document that never saw them.
ARWEN_WSM6_ESTATE_AUDIT_SHA256 = (
    "b92c2f6308f188a229016295ba77dd40fb423fbe2fdabb8dd0704da256ab812e"
)

#: The GPU-free dense-ice fall-speed oracle a Morrison execution can cite.
#: WSM6 has a measured component oracle and Morrison does not (R-019), so the
#: two are separate fields and neither borrows the other's.
MORRISON_DENSE_ICE_ORACLE_RECEIPT_SHA256 = (
    "5c20d1848bc880ca80a810c2c97df324c01b160d8e79123c9b468e65311002eb"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_f32(name: str, value: object) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


MASS_INPUT_NAMES: tuple[str, ...] = ("pi0", "th0", "pp3d", "th3d", "p", "rho")


def tiled_column_buffer_order(spec: ColumnComponentSpec) -> dict[str, tuple[str, ...]]:
    """The exact, ordered host and device inventories for one component."""

    volume_persistence = tuple(
        name
        for name in spec.component_persistence
        if name not in spec.surface_persistence
    )
    host_inputs = (
        *MASS_INPUT_NAMES, "phb", *spec.extra_interface_carriers, "q3d",
        *volume_persistence, *spec.surface_persistence,
    )
    host_buffers = (*host_inputs, "thp")
    return {
        "volume_persistence": volume_persistence,
        "host_inputs": host_inputs,
        "host_buffers": host_buffers,
        "device_buffers": (*host_buffers, "alt", "php"),
        "d2h": ("q3d", "thp", *spec.component_persistence),
    }


def tiled_column_host_buffer_shapes(
    spec: ColumnComponentSpec, *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact pinned-host union of scheme inputs and drainable outputs."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    order = tiled_column_buffer_order(spec)
    mass = (nk, tile_ny, tile_nx)
    shapes: dict[str, tuple[int, ...]] = {name: mass for name in MASS_INPUT_NAMES}
    shapes["phb"] = (nk + 1, tile_ny, tile_nx)
    # WRF's grid%w_2 lower full-level slice: the same (nk+1, ny, nx) stagger
    # GM1's own w3d already carries, staged unchanged.
    for name in spec.extra_interface_carriers:
        shapes[name] = (nk + 1, tile_ny, tile_nx)
    shapes["q3d"] = (spec.profile.numq, *mass)
    shapes.update({name: mass for name in order["volume_persistence"]})
    shapes.update({name: (tile_ny, tile_nx) for name in spec.surface_persistence})
    shapes["thp"] = mass
    if tuple(shapes) != order["host_buffers"]:  # pragma: no cover - import-time guard
        raise RuntimeError("tiled column host inventory order drifted")
    return MappingProxyType(shapes)


def tiled_column_device_buffer_shapes(
    spec: ColumnComponentSpec, *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact exposed device arrays; public-component scratch is separate."""

    shapes = dict(
        tiled_column_host_buffer_shapes(
            spec, tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
        )
    )
    shapes["alt"] = (nk, tile_ny, tile_nx)
    shapes["php"] = (nk + 1, tile_ny, tile_nx)
    order = tiled_column_buffer_order(spec)
    if tuple(shapes) != order["device_buffers"]:  # pragma: no cover
        raise RuntimeError("tiled column device inventory order drifted")
    return MappingProxyType(shapes)


def require_tiled_column_device_arrays(
    cupy: Any, spec: ColumnComponentSpec, arrays: Mapping[str, Any],
    *, nx: int, ny: int, nk: int,
) -> dict[str, Any]:
    shapes = tiled_column_device_buffer_shapes(spec, tile_nx=nx, tile_ny=ny, nk=nk)
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError(
            f"CUDA tiled {spec.component_id} inventory/order changed"
        )
    result: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"device {name} must be C-order CUDA float32 {shape}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError(f"CUDA tiled {spec.component_id} arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaGm1TiledColumnLaunch:
    ready_event: Any
    component: Any
    component_result: Any
    arrays: Mapping[str, Any]
    receipt: Mapping[str, Any]
    #: The scheme's own 10 cm reflectivity for this block, in dBZ on the
    #: [nk, ny, nx] tile, or ``None`` when this call was not an output step.
    #: It is component-owned transient scratch consumed exactly once by
    #: ``gpuwm.core.refl.consume_refl_10cm``; it is not model state and is
    #: not restored, so a caller must drain it before the next block reuses
    #: the component.  The launch refuses a due call that produced none.
    reflectivity_dbz: Any = None


def _component_class(component_id: str) -> Any:
    """The ArwenColumnComponent subclass bound to one admitted component id."""

    if component_id == "wsm6-mp6":
        from .arwen_wsm6_component import ArwenWsm6Component

        return ArwenWsm6Component
    if component_id == "morrison-mp10":
        from .arwen_morrison_component import ArwenMorrisonComponent

        return ArwenMorrisonComponent
    if component_id == "thompson-mp8":
        from .arwen_scheme_components import ArwenThompsonComponent

        return ArwenThompsonComponent
    if component_id == "thompson-aerosol-mp28":
        from .arwen_scheme_components import ArwenThompsonAerosolComponent

        return ArwenThompsonAerosolComponent
    raise ValueError(
        f"component {component_id!r} has no CM1-owned tiled binding; a tile "
        "launched without one would stage carriers for a scalar axis nothing "
        "here declares"
    )


def launch_cuda_gm1_tiled_column_block(
    spec: ColumnComponentSpec,
    arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    source_identity_sha256: str,
    pdefq_receipt_sha256: str,
    completed_calls: int,
    block_index: int,
    active_nx: int,
    active_ny: int,
    ihail: int,
    claim_limit: str,
    stream: Any | None = None,
    capacity_profile: str | None = None,
    refl_10cm_due: bool = False,
) -> CudaGm1TiledColumnLaunch:
    """Stage and enqueue one independent fixed-shape column block.

    ``refl_10cm_due`` is CM1's ``getdbz`` (``cm1.F:3214-3237``): the scheduler
    raises it on the step whose state a frame will carry, and the component
    then produces the reflectivity diagnostic from the same columns.
    """

    if type(refl_10cm_due) is not bool:
        raise TypeError("refl_10cm_due must be a strict boolean schedule decision")
    if refl_10cm_due and spec.component_id not in TILESTREAM_REFLECTIVITY_AUTHORITY:
        raise ValueError(
            f"component {spec.component_id!r} has no admitted reflectivity "
            "formula; publishing another scheme's dBZ under these hydrometeors "
            "would report an echo the simulated storm never scattered"
        )

    nx = _positive_int("nx", nx)
    ny = _positive_int("ny", ny)
    nk = _positive_int("nk", nk)
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if active_nx > nx or active_ny > ny:
        raise ValueError("active column rectangle exceeds the fixed block")
    block_index = _nonnegative_int("block_index", block_index)
    completed_calls = _nonnegative_int("completed_calls", completed_calls)
    source_identity_sha256 = _require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    pdefq_receipt_sha256 = _require_sha256(
        "pdefq_receipt_sha256", pdefq_receipt_sha256
    )
    spacing = tuple(_positive_f32(name, value) for name, value in (
        ("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)
    ))
    if capacity_profile is None:
        minimum, maximum = spec.minimum_levels, spec.maximum_levels
    else:
        minimum, maximum = capacity_level_bounds(capacity_profile)
    if not minimum <= nk <= maximum:
        raise ValueError(
            f"tiled {spec.component_id} requires {minimum} <= nk <= {maximum}"
        )
    inspection = engine_pin.inspect(
        capacity_profile=capacity_profile, component_id=spec.component_id
    )
    if not inspection.admitted:
        raise RuntimeError(
            f"installed ArWen {spec.component_id} estate differs from the audited bytes"
        )
    cupy = importlib.import_module("cupy")
    values = require_tiled_column_device_arrays(
        cupy, spec, arrays, nx=nx, ny=ny, nk=nk
    )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    q = {
        name: values["q3d"][index]
        for index, name in enumerate(spec.species)
    }
    carriers = spec.carriers_type(
        thp=values["thp"],
        thb=values["th0"],
        p=values["p"],
        alt=values["alt"],
        phb=values["phb"],
        php=values["php"],
        **{name: values[name] for name in spec.extra_interface_carriers},
        **q,
    )
    persistence = {name: values[name] for name in spec.component_persistence}
    component = _component_class(spec.component_id)(
        component_id=spec.component_id,
        nx=nx,
        ny=ny,
        nz=nk,
        dx_m=spacing[0],
        dy_m=spacing[1],
        dz_m=spacing[2],
        dt_s=spacing[3],
        capacity_profile=capacity_profile,
        ihail=ihail,
    )
    with stream:
        # eqtset2_wsm6_bridge.py:807-819 after the pre-component hook.
        # ``alt`` temporarily holds CM1 total Exner, then its final 1/rho.
        cupy.add(values["pi0"], values["pp3d"], out=values["alt"])
        cupy.add(values["th0"], values["th3d"], out=values["thp"])
        cupy.multiply(values["thp"], values["alt"], out=values["thp"])
        cupy.reciprocal(values["rho"], out=values["alt"])
        cupy.multiply(values["p"], cupy.float32(1.0 / 100000.0), out=values["pp3d"])
        cupy.power(values["pp3d"], cupy.float32(2.0 / 7.0), out=values["pp3d"])
        cupy.divide(values["thp"], values["pp3d"], out=values["thp"])
        cupy.subtract(values["thp"], values["th0"], out=values["thp"])
        values["php"].fill(cupy.float32(0.0))
        component.restore_persistent_arrays(
            carriers, persistence, completed_calls=completed_calls
        )
        result = component.apply(carriers, refl_10cm_due=refl_10cm_due)
    if int(result.stream_pointer) != int(stream.ptr):
        raise RuntimeError("public component changed the caller stream")
    identity = TILESTREAM_RECORD_IDENTITY[spec.component_id]
    record: dict[str, Any] = {
        "schema": identity["schema"],
        "abi": identity["abi"],
        "status": identity["status"],
        "block_index": block_index,
        "fixed_shape_zyx": [nk, ny, nx],
        "active_shape_yx": [active_ny, active_nx],
        "source_identity_sha256": source_identity_sha256,
        "pdefq_receipt_sha256": pdefq_receipt_sha256,
        "component_call_index": result.call_index,
        "completed_calls_before_block": completed_calls,
        "source_order": [
            "accepted pdefq q3d",
            "CM1 temperature and ArWen Exner/thp staging",
            f"public gpuwm.core.microphysics.apply mp_physics={spec.mp_physics}",
            (
                f"drain q3d, thp, and {len(spec.component_persistence)} "
                "persistence carriers"
            ),
        ],
        "staging": {
            "pp3d_tile_disposition": "destructive scratch for ArWen pii after temperature",
            "rho_fixed_across_component": True,
            "php_positive_zero": True,
            "statement_order_anchor": "src/cm1core/eqtset2_wsm6_bridge.py:807-819",
        },
        "component_receipt": dict(result.receipt),
        "arwen": {
            "source_identity": engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY,
            "audited_surface": engine_pin.audited_surface(
                capacity_profile, spec.component_id
            ),
            "native_oracle_receipt_sha256": TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256[
                spec.component_id
            ],
            "estate_audit_sha256": ARWEN_WSM6_ESTATE_AUDIT_SHA256,
            "dense_ice_fallspeed_oracle_receipt_sha256": (
                MORRISON_DENSE_ICE_ORACLE_RECEIPT_SHA256
            ),
        },
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "mutated": [
            "thp",
            f"q3d[0:{spec.profile.numq}]",
            *spec.component_persistence,
        ],
        "refusals": {
            "reflectivity": not refl_10cm_due,
            "qa_mutation": True,
            "cm1_post_physics_eos": True,
            "solve3": True,
            "pressure": True,
            "halo_or_open_history_mutation": True,
            "mpi_reduction": True,
            "five_meter_meteorological_product": True,
        },
        "claim_limit": claim_limit,
    }
    if capacity_profile is not None:
        record["capacity_profile"] = capacity_profile_metadata(capacity_profile)
        record["capacity_profile_sha256"] = capacity_profile_sha256(capacity_profile)
    reflectivity = None
    if refl_10cm_due:
        reflectivity = result.reflectivity_10cm
        if reflectivity is None:
            raise RuntimeError(
                f"the public {spec.component_id} component reported no "
                "reflectivity on a due output step; the frame would publish an "
                "empty dBZ carrier"
            )
        if (
            not isinstance(reflectivity, cupy.ndarray)
            or reflectivity.dtype != cupy.float32
            or tuple(reflectivity.shape) != (nk, ny, nx)
            or not reflectivity.flags.c_contiguous
        ):
            raise RuntimeError(
                f"the public {spec.component_id} reflectivity result is not the "
                "C-order float32 [nk, ny, nx] block this tile staged, so "
                "scattering it would place one column's dBZ words at another "
                "column's position"
            )
        # These blocks travel with the record so a reader can see that the
        # published dBZ came from the scheme that owns the size distributions
        # it depends on, and not from a generic formula applied afterwards.
        # They are added only on a due call, so a not-due record hashes to
        # exactly the bytes it always did.
        record["refl_10cm_due"] = True
        record["reflectivity"] = {
            "producer": ARWEN_PUBLIC_REFLECTIVITY_CALL,
            "units": "dBZ",
            "shape_zyx": [nk, ny, nx],
            "lifetime": "consumed exactly once by this call; not restart state",
            "formula_authority": TILESTREAM_REFLECTIVITY_AUTHORITY[spec.component_id],
            "cm1_output_due_authority": "src/cm1.F:3214-3237 getdbz",
        }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledColumnLaunch(
        result.ready_event,
        component,
        result,
        MappingProxyType(values),
        MappingProxyType(record),
        reflectivity,
    )


__all__ = [
    "ARWEN_WSM6_ESTATE_AUDIT_SHA256",
    "MORRISON_DENSE_ICE_ORACLE_RECEIPT_SHA256",
    "MASS_INPUT_NAMES",
    "TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256",
    "TILESTREAM_RECORD_IDENTITY",
    "CudaGm1TiledColumnLaunch",
    "column_component_class",
    "launch_cuda_gm1_tiled_column_block",
    "require_tiled_column_device_arrays",
    "tiled_column_buffer_order",
    "tiled_column_device_buffer_shapes",
    "tiled_column_host_buffer_shapes",
]

#: Public spelling for callers outside the tile launcher (the runtime warm-up).
#: The launcher itself looks ``_component_class`` up at call time.
def column_component_class(component_id: str) -> Any:
    return _component_class(component_id)
