"""Typed structural state contract for the first CM1 C-grid slice.

This is not a scientific CM1 initializer.  It turns source-known allocation
bounds into a backend-neutral plan and can materialize a deterministic host
sentinel state for schema/layout tests.  Only the physical interior is written
with ``+0.0f``; every unconfirmed halo word remains a quiet-NaN bit sentinel.
That distinction prevents this scaffolding from silently becoming a generic
halo-fill implementation before the native sentinels exist.

The plan is suitable for a future CUDA allocator and Rust zero-step writer.
The included bytearray materializer is evidence of structural reachability
only and must never be presented as device residency, base-state
initialization, or numerical parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

from . import CM1_AUTHORITY_SHA256
from .config import CM1Config, Lane
from .layout import (
    GRID_COORDINATES,
    STRUCTURAL_DYCORE_LAYOUTS,
    TARGET_HALO,
    FieldGeometry,
    FieldLayout,
    GridShape,
    Halo,
    Stagger,
)


STRUCTURAL_STATE_PLAN_SCHEMA = "gpuwm-cm1.structural-state-plan/v1"
STRUCTURAL_STATE_SCHEMA = "gpuwm-cm1.structural-state/v1"
STRUCTURAL_CLAIM_LIMIT = (
    "layout/interior-sentinel self-test only; not CM1 base-state or "
    "perturbation initialization, CUDA residency, a native oracle, or forecast output"
)
FLOAT32_LE = "<f4"
FLOAT32_ITEMSIZE = 4
CONFIRMED_INTERIOR_WORD = 0x00000000
UNCONFIRMED_HALO_WORD = 0x7FC00000
DEFAULT_STRUCTURAL_ALLOCATION_LIMIT = 64 * 1024 * 1024
_MAX_SIGNED_64 = (1 << 63) - 1
_ZERO_WORD_BYTES = struct.pack("<I", CONFIRMED_INTERIOR_WORD)
_HALO_WORD_BYTES = struct.pack("<I", UNCONFIRMED_HALO_WORD)
_STRUCTURAL_PROFILE_GEOMETRY: dict[
    str, tuple[tuple[int, int, int], tuple[float, float, float]]
] = {
    Lane.DRY_DYCORE.value: ((16, 16, 16), (1000.0, 1000.0, 1000.0)),
    Lane.PRODUCT.value: ((120, 120, 40), (1000.0, 1000.0, 500.0)),
    Lane.REAL_PRODUCT.value: ((300, 300, 75), (100.0, 100.0, 200.0)),
    Lane.NATIVE_CONTEXT.value: ((120, 120, 40), (1000.0, 1000.0, 500.0)),
}


def structural_profile_geometry(
    lane: str,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Return the fixed v1 dimensions and spacing for one sealed lane."""

    try:
        return _STRUCTURAL_PROFILE_GEOMETRY[lane]
    except (KeyError, TypeError) as error:
        raise ValueError("state plan lane is not a sealed CM1 lane") from error


@dataclass(frozen=True, slots=True)
class DeferredStateFamily:
    name: str
    native_names: tuple[str, ...]
    gate: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "name": self.name,
            "native_names": list(self.native_names),
            "reason": self.reason,
        }


DEFERRED_STATE_FAMILIES: tuple[DeferredStateFamily, ...] = (
    DeferredStateFamily(
        name="grid-metrics-and-base-state-values",
        native_names=(
            "xh",
            "xf",
            "yh",
            "yf",
            "zh",
            "zf",
            "mh",
            "rmh",
            "mf",
            "rmf",
            "pi0",
            "rho0",
            "prs0",
            "th0",
            "qv0",
            "rr0",
            "rf0",
            "u0",
            "v0",
        ),
        gate="O-GRID",
        reason=(
            "allocation geometry is source-known, but coordinate rounding, sounding, "
            "hydrostatic base values, and exact native live regions lack an executable oracle"
        ),
    ),
    DeferredStateFamily(
        name="packed-moisture",
        native_names=("qa", "q3d", "qten", "qpten", "qtten", "qvten", "qcten"),
        gate="O-ARWEN-WSM6",
        reason=(
            "species order, energy carriers, and component persistence await the public "
            "CM1-neutral ArWen seam"
        ),
    ),
    DeferredStateFamily(
        name="sgs-tke",
        native_names=("tkea", "tke3d", "tketen", "kmh", "kmv", "khh", "khv"),
        gate="O-CLOSURES",
        reason="closure-specific bounds and live regions have no native sentinel fixture",
    ),
    DeferredStateFamily(
        name="tendencies-boundary-history-and-clocks",
        native_names=(
            "uten",
            "uten1",
            "vten",
            "vten1",
            "wten",
            "wten1",
            "ppten",
            "thten",
            "thten1",
            "radbcw",
            "radbce",
            "radbcs",
            "radbcn",
            "nrk",
            "acoustic_substep",
        ),
        gate="O-INSTRUMENT",
        reason="exact lifetime, halo access, and restart cadence remain uninstrumented",
    ),
)


@dataclass(frozen=True, slots=True)
class PlannedField:
    layout: FieldLayout
    geometry: FieldGeometry
    dtype: str = FLOAT32_LE
    itemsize: int = FLOAT32_ITEMSIZE

    def __post_init__(self) -> None:
        if self.layout.precision != "float32":
            raise ValueError(
                f"structural field {self.layout.native_name!r} is not declared float32"
            )
        if self.dtype != FLOAT32_LE or self.itemsize != FLOAT32_ITEMSIZE:
            raise ValueError("the structural wire dtype is fixed to little-endian float32")
        if self.layout.stagger is not self.geometry.stagger:
            raise ValueError("field layout and geometry staggering disagree")
        if self.nbytes > _MAX_SIGNED_64:
            raise OverflowError("field byte count exceeds the signed-64 wire contract")

    @property
    def element_count(self) -> int:
        return self.geometry.allocation.element_count

    @property
    def nbytes(self) -> int:
        return self.element_count * self.itemsize

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "geometry": self.geometry.as_dict(),
            "initialization": {
                "confirmed_interior_word": f"0x{CONFIRMED_INTERIOR_WORD:08x}",
                "outside_confirmed_interior_word": f"0x{UNCONFIRMED_HALO_WORD:08x}",
                "status": "structural-sentinel-only",
            },
            "itemsize": self.itemsize,
            "live_region_policy": self.layout.live_region.value,
            "meaning": self.layout.meaning,
            "name": self.layout.native_name,
            "nbytes": self.nbytes,
            "output_name": self.layout.output_name,
            "persistence": self.layout.persistence,
            "units": self.layout.units,
        }


@dataclass(frozen=True, slots=True)
class StructuralStatePlan:
    config_sha256: str
    lane: str
    grid: GridShape
    spacing_m: tuple[float, float, float]
    halo: Halo
    fields: tuple[PlannedField, ...]
    deferred: tuple[DeferredStateFamily, ...] = DEFERRED_STATE_FAMILIES
    authority_sha256: str = CM1_AUTHORITY_SHA256
    schema: str = STRUCTURAL_STATE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STRUCTURAL_STATE_PLAN_SCHEMA:
            raise ValueError("the v1 structural state-plan schema is fixed")
        if len(self.config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_sha256
        ):
            raise ValueError("config_sha256 must be a lower-case SHA-256 hex digest")
        if len(self.authority_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.authority_sha256
        ):
            raise ValueError("authority_sha256 must be a lower-case SHA-256 hex digest")
        if self.authority_sha256 != CM1_AUTHORITY_SHA256:
            raise ValueError("state plan authority does not match the packaged CM1 authority")
        expected_shape, expected_spacing = structural_profile_geometry(self.lane)
        if self.halo != TARGET_HALO:
            raise ValueError("the v1 structural state plan is fixed to the target halo")
        if len(self.spacing_m) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in self.spacing_m
        ):
            raise ValueError("spacing_m must contain three finite positive values")
        if (self.grid.nx, self.grid.ny, self.grid.nz) != expected_shape:
            raise ValueError("state plan dimensions differ from the fixed lane profile")
        if self.spacing_m != expected_spacing:
            raise ValueError("state plan spacing differs from the fixed lane profile")
        names = tuple(field.layout.native_name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("planned field names must be unique")
        if names != tuple(row.native_name for row in STRUCTURAL_DYCORE_LAYOUTS):
            raise ValueError("the v1 structural field inventory and order are fixed")
        for field, expected_layout in zip(self.fields, STRUCTURAL_DYCORE_LAYOUTS):
            if field.layout != expected_layout:
                raise ValueError("structural field metadata differs from the fixed v1 layout")
            if field.geometry != expected_layout.geometry(self.grid, self.halo):
                raise ValueError("structural field geometry differs from native CM1 bounds")
        if self.deferred != DEFERRED_STATE_FAMILIES:
            raise ValueError("the v1 deferred-state gate inventory is fixed")
        if self.total_nbytes > _MAX_SIGNED_64:
            raise OverflowError("state byte count exceeds the signed-64 wire contract")

    @property
    def total_nbytes(self) -> int:
        return sum(field.nbytes for field in self.fields)

    def field(self, name: str) -> PlannedField:
        for field in self.fields:
            if field.layout.native_name == name:
                return field
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        coordinate_rows = []
        for coordinate in GRID_COORDINATES:
            bounds = coordinate.native_bounds(self.grid, self.halo)
            coordinate_rows.append(
                {
                    "axis": coordinate.axis.value,
                    "count": bounds.count,
                    "location": coordinate.location.value,
                    "name": coordinate.native_name,
                    "native_bounds": bounds.as_dict(),
                    "status": "allocation-only-values-deferred-to-O-GRID",
                    "units": coordinate.units,
                }
            )
        stagger_geometry = {
            stagger.value: self.grid.geometry(stagger, self.halo).as_dict()
            for stagger in Stagger
        }
        return {
            "authority_sha256": self.authority_sha256,
            "claim_limit": STRUCTURAL_CLAIM_LIMIT,
            "config_sha256": self.config_sha256,
            "deferred_state_families": [row.as_dict() for row in self.deferred],
            "fields": [field.as_dict() for field in self.fields],
            "grid": {
                "axis_order": "zyx",
                "coordinates": coordinate_rows,
                "dimensions": self.grid.as_dict(),
                "fortran_bounds": "inclusive",
                "halo": self.halo.as_dict(),
                "spacing_m": list(self.spacing_m),
                "stagger_geometry": stagger_geometry,
            },
            "lane": self.lane,
            "schema": self.schema,
            "status": "structural-allocation-contract-only",
            "total_nbytes": self.total_nbytes,
        }

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def build_structural_state_plan(config: CM1Config) -> StructuralStatePlan:
    """Build the backend-neutral v1 plan from one already sealed config."""

    if config.grid.terrain or config.grid.stretch_x or config.grid.stretch_y or config.grid.stretch_z:
        raise ValueError("the structural v1 plan is uniform, flat-grid only")
    selected_orders = (
        config.dynamics.scalar_horizontal_order,
        config.dynamics.scalar_vertical_order,
        config.dynamics.velocity_horizontal_order,
        config.dynamics.velocity_vertical_order,
        config.dynamics.weno_order,
    )
    if selected_orders != (5, 5, 5, 5, 5):
        raise ValueError("the structural v1 halo is sealed to fifth-order/WENO5")
    grid = GridShape(config.grid.nx, config.grid.ny, config.grid.nz)
    fields = tuple(
        PlannedField(layout=row, geometry=row.geometry(grid, TARGET_HALO))
        for row in STRUCTURAL_DYCORE_LAYOUTS
    )
    return StructuralStatePlan(
        config_sha256=config.sha256,
        lane=config.run.lane.value,
        grid=grid,
        spacing_m=(config.grid.dx_m, config.grid.dy_m, config.grid.dz_m),
        halo=TARGET_HALO,
        fields=fields,
    )


@dataclass(frozen=True, slots=True)
class StructuralFieldBuffer:
    field: PlannedField
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("structural field payloads must be immutable bytes")
        if len(self.data) != self.field.nbytes:
            raise ValueError(
                f"{self.field.layout.native_name} payload has {len(self.data)} bytes; "
                f"expected {self.field.nbytes}"
            )
        validate_structural_field_payload(self.field, self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def word_bits(self, *, i: int, j: int, k: int) -> int:
        """Read one native-indexed word without interpreting it as a float."""

        allocation = self.field.geometry.allocation
        x = allocation.x.offset(i)
        y = allocation.y.offset(j)
        z = allocation.z.offset(k)
        _, y_count, x_count = allocation.shape_zyx
        offset = ((z * y_count + y) * x_count + x) * FLOAT32_ITEMSIZE
        return struct.unpack_from("<I", self.data, offset)[0]


def validate_structural_field_payload(field: PlannedField, data: bytes) -> None:
    """Require the exact interior-zero/halo-sentinel word pattern."""

    if not isinstance(data, bytes):
        raise TypeError("structural field payloads must be immutable bytes")
    if len(data) != field.nbytes:
        raise ValueError(
            f"{field.layout.native_name} payload has {len(data)} bytes; "
            f"expected {field.nbytes}"
        )
    z_slice, y_slice, x_slice = field.geometry.confirmed_slices_zyx
    allocated_z, allocated_y, allocated_x = field.geometry.allocated_shape_zyx
    halo_row = _HALO_WORD_BYTES * allocated_x
    interior_row = (
        _HALO_WORD_BYTES * x_slice.start
        + _ZERO_WORD_BYTES * (x_slice.stop - x_slice.start)
        + _HALO_WORD_BYTES * (allocated_x - x_slice.stop)
    )
    row_nbytes = allocated_x * FLOAT32_ITEMSIZE
    for z in range(allocated_z):
        z_is_interior = z_slice.start <= z < z_slice.stop
        for y in range(allocated_y):
            expected = (
                interior_row
                if z_is_interior and y_slice.start <= y < y_slice.stop
                else halo_row
            )
            start = (z * allocated_y + y) * row_nbytes
            if data[start : start + row_nbytes] != expected:
                raise ValueError(
                    f"{field.layout.native_name} payload violates the structural "
                    f"sentinel policy at allocated z/y row {z}/{y}"
                )


@dataclass(frozen=True, slots=True)
class StructuralState:
    plan: StructuralStatePlan
    buffers: tuple[StructuralFieldBuffer, ...]
    schema: str = STRUCTURAL_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STRUCTURAL_STATE_SCHEMA:
            raise ValueError("the v1 structural state schema is fixed")
        planned = tuple(field.layout.native_name for field in self.plan.fields)
        actual = tuple(buffer.field.layout.native_name for buffer in self.buffers)
        if actual != planned:
            raise ValueError("buffer inventory/order must exactly match the state plan")
        if any(buffer.field != field for buffer, field in zip(self.buffers, self.plan.fields)):
            raise ValueError("buffer metadata must exactly match the state plan")
        if self.total_nbytes != self.plan.total_nbytes:
            raise ValueError("buffer byte count must exactly match the state plan")

    @property
    def total_nbytes(self) -> int:
        return sum(len(buffer.data) for buffer in self.buffers)

    def buffer(self, name: str) -> StructuralFieldBuffer:
        for buffer in self.buffers:
            if buffer.field.layout.native_name == name:
                return buffer
        raise KeyError(name)


def _initialize_field(field: PlannedField) -> StructuralFieldBuffer:
    payload = bytearray(_HALO_WORD_BYTES * field.element_count)
    z_slice, y_slice, x_slice = field.geometry.confirmed_slices_zyx
    _, allocated_y, allocated_x = field.geometry.allocated_shape_zyx
    row_bytes = _ZERO_WORD_BYTES * (x_slice.stop - x_slice.start)
    for z in range(z_slice.start, z_slice.stop):
        for y in range(y_slice.start, y_slice.stop):
            start = ((z * allocated_y + y) * allocated_x + x_slice.start) * FLOAT32_ITEMSIZE
            payload[start : start + len(row_bytes)] = row_bytes
    return StructuralFieldBuffer(field=field, data=bytes(payload))


def initialize_structural_state(
    plan: StructuralStatePlan,
    *,
    max_bytes: int = DEFAULT_STRUCTURAL_ALLOCATION_LIMIT,
) -> StructuralState:
    """Materialize the deterministic interior-zero/halo-sentinel test state."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if plan.total_nbytes > max_bytes:
        raise MemoryError(
            f"structural state needs {plan.total_nbytes} bytes; limit is {max_bytes}"
        )
    buffers = tuple(_initialize_field(field) for field in plan.fields)
    return StructuralState(plan=plan, buffers=buffers)


__all__ = [
    "CONFIRMED_INTERIOR_WORD",
    "DEFAULT_STRUCTURAL_ALLOCATION_LIMIT",
    "DEFERRED_STATE_FAMILIES",
    "DeferredStateFamily",
    "FLOAT32_ITEMSIZE",
    "FLOAT32_LE",
    "PlannedField",
    "STRUCTURAL_CLAIM_LIMIT",
    "STRUCTURAL_STATE_PLAN_SCHEMA",
    "STRUCTURAL_STATE_SCHEMA",
    "StructuralFieldBuffer",
    "StructuralState",
    "StructuralStatePlan",
    "UNCONFIRMED_HALO_WORD",
    "build_structural_state_plan",
    "initialize_structural_state",
    "structural_profile_geometry",
    "validate_structural_field_payload",
]
