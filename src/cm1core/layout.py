"""CM1-owned C-grid allocation and live-region metadata.

The pristine CM1 allocation bounds are expressible before a numerical backend
exists.  This module records those bounds in native one-based index space and
provides their C-order ``(z, y, x)`` translation.  It deliberately confirms
only the physical interior as writable: source-adjacent native sentinels must
still prove every ghost read/write region before a boundary or solver kernel
may use it.

In particular, A time-level arrays and current 3-D arrays remain distinct even
when their allocated Fortran bounds match.  A caller cannot turn matching
shapes into permission to share a halo-fill policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod


class Stagger(StrEnum):
    SCALAR = "scalar"
    X_FACE = "x-face"
    Y_FACE = "y-face"
    Z_FACE = "z-face"


class Axis(StrEnum):
    X = "x"
    Y = "y"


class CoordinateLocation(StrEnum):
    MASS = "mass"
    FACE = "face"


class LiveRegionPolicy(StrEnum):
    """Scientifically distinct valid-region families from CM1 source."""

    SCALAR_A_TIME_LEVEL = "scalar-a-time-level"
    SCALAR_3D = "scalar-current-3d"
    X_FACE_A_TIME_LEVEL = "x-face-a-time-level"
    X_FACE_3D = "x-face-current-3d"
    Y_FACE_A_TIME_LEVEL = "y-face-a-time-level"
    Y_FACE_3D = "y-face-current-3d"
    Z_FACE_A_TIME_LEVEL = "z-face-a-time-level"
    Z_FACE_3D = "z-face-current-3d"
    MOIST_A_TIME_LEVEL = "moist-a-time-level"
    MOIST_3D = "moist-current-3d"
    W_LEVEL_SGS_A_TIME_LEVEL = "w-level-sgs-a-time-level"
    W_LEVEL_SGS_3D = "w-level-sgs-current-3d"


# These descriptions intentionally do not guess an exact live slice. The
# allocations are source-known, but source-adjacent sentinels must still prove
# each read/write region before the state can allocate or fill it.
LIVE_REGION_CONTRACTS: dict[LiveRegionPolicy, str] = {
    LiveRegionPolicy.SCALAR_A_TIME_LEVEL: "A carrier on scalar allocation; never share a current-3d halo fill",
    LiveRegionPolicy.SCALAR_3D: "current scalar 3d state; distinct valid ghosts from A carrier",
    LiveRegionPolicy.X_FACE_A_TIME_LEVEL: "A carrier on x faces; distinct boundary/live policy from u3d",
    LiveRegionPolicy.X_FACE_3D: "current x-face 3d state; distinct boundary/live policy from ua",
    LiveRegionPolicy.Y_FACE_A_TIME_LEVEL: "A carrier on y faces; distinct boundary/live policy from v3d",
    LiveRegionPolicy.Y_FACE_3D: "current y-face 3d state; distinct boundary/live policy from va",
    LiveRegionPolicy.Z_FACE_A_TIME_LEVEL: "A carrier on z faces; includes extra interface and differs from w3d",
    LiveRegionPolicy.Z_FACE_3D: "current z-face 3d state; includes extra interface and differs from wa",
    LiveRegionPolicy.MOIST_A_TIME_LEVEL: "packed moisture A carrier; semantic species map and live slice are seam gates",
    LiveRegionPolicy.MOIST_3D: "packed current moisture; never assume the A-carrier live slice",
    LiveRegionPolicy.W_LEVEL_SGS_A_TIME_LEVEL: "TKE A carrier on closure-specific w levels",
    LiveRegionPolicy.W_LEVEL_SGS_3D: "current TKE on closure-specific w levels; distinct from tkea",
}


@dataclass(frozen=True, slots=True)
class NativeRange:
    """Inclusive native-Fortran index range."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise TypeError("native bounds must be integers, not booleans")
        if not isinstance(self.lower, int) or not isinstance(self.upper, int):
            raise TypeError("native bounds must be integers")
        if self.upper < self.lower:
            raise ValueError("native upper bound must not precede lower bound")

    @property
    def count(self) -> int:
        return self.upper - self.lower + 1

    def contains(self, other: "NativeRange") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def offset(self, native_index: int) -> int:
        if isinstance(native_index, bool) or not isinstance(native_index, int):
            raise TypeError("native indices must be integers, not booleans")
        if native_index < self.lower or native_index > self.upper:
            raise IndexError(
                f"native index {native_index} outside [{self.lower}, {self.upper}]"
            )
        return native_index - self.lower

    def as_dict(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class NativeBox:
    """Three-dimensional inclusive bounds, named in physical axis order."""

    x: NativeRange
    y: NativeRange
    z: NativeRange

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.z.count, self.y.count, self.x.count

    @property
    def element_count(self) -> int:
        return prod(self.shape_zyx)

    def contains(self, other: "NativeBox") -> bool:
        return (
            self.x.contains(other.x)
            and self.y.contains(other.y)
            and self.z.contains(other.z)
        )

    def slices_for(self, inner: "NativeBox") -> tuple[slice, slice, slice]:
        """Translate an inner native box to half-open C-order slices."""

        if not self.contains(inner):
            raise ValueError("inner native box is outside the allocation")
        return (
            slice(self.z.offset(inner.z.lower), self.z.offset(inner.z.upper) + 1),
            slice(self.y.offset(inner.y.lower), self.y.offset(inner.y.upper) + 1),
            slice(self.x.offset(inner.x.lower), self.x.offset(inner.x.upper) + 1),
        )

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "x": self.x.as_dict(),
            "y": self.y.as_dict(),
            "z": self.z.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Halo:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in (self.x, self.y, self.z)):
            raise TypeError("halo widths must be integers, not booleans")
        if not all(isinstance(value, int) for value in (self.x, self.y, self.z)):
            raise TypeError("halo widths must be integers")
        if min(self.x, self.y, self.z) < 0:
            raise ValueError("halo widths must be non-negative")

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "z": self.z}


# src/cm1.F:744-777 for the selected fifth-order/WENO5 campaign.
TARGET_HALO = Halo(x=3, y=3, z=1)


@dataclass(frozen=True, slots=True)
class FieldGeometry:
    """Source-known allocation plus the currently confirmed safe interior."""

    stagger: Stagger
    allocation: NativeBox
    confirmed_interior: NativeBox

    def __post_init__(self) -> None:
        if not self.allocation.contains(self.confirmed_interior):
            raise ValueError("confirmed interior must lie inside its allocation")

    @property
    def allocated_shape_zyx(self) -> tuple[int, int, int]:
        return self.allocation.shape_zyx

    @property
    def interior_shape_zyx(self) -> tuple[int, int, int]:
        return self.confirmed_interior.shape_zyx

    @property
    def confirmed_slices_zyx(self) -> tuple[slice, slice, slice]:
        return self.allocation.slices_for(self.confirmed_interior)

    def as_dict(self) -> dict[str, object]:
        slices = self.confirmed_slices_zyx
        return {
            "allocated_shape_zyx": list(self.allocated_shape_zyx),
            "allocation_native_bounds": self.allocation.as_dict(),
            "confirmed_interior_native_bounds": self.confirmed_interior.as_dict(),
            "confirmed_interior_slices_zyx": [
                {"start": item.start, "stop": item.stop} for item in slices
            ],
            "halo_authority": "unconfirmed-outside-physical-interior",
            "stagger": self.stagger.value,
        }


@dataclass(frozen=True, slots=True)
class GridShape:
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in (self.nx, self.ny, self.nz)):
            raise TypeError("CM1 dimensions must be integers, not booleans")
        if not all(isinstance(value, int) for value in (self.nx, self.ny, self.nz)):
            raise TypeError("CM1 dimensions must be integers")
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError("CM1 dimensions must all be positive")

    def interior(self, stagger: Stagger) -> tuple[int, int, int]:
        """Return C-order ``(z, y, x)`` interior shape for one staggering."""

        if stagger is Stagger.SCALAR:
            return self.nz, self.ny, self.nx
        if stagger is Stagger.X_FACE:
            return self.nz, self.ny, self.nx + 1
        if stagger is Stagger.Y_FACE:
            return self.nz, self.ny + 1, self.nx
        if stagger is Stagger.Z_FACE:
            return self.nz + 1, self.ny, self.nx
        raise AssertionError(f"unhandled stagger {stagger!r}")

    def allocated(self, stagger: Stagger, halo: Halo) -> tuple[int, int, int]:
        return self.geometry(stagger, halo).allocated_shape_zyx

    def native_interior(self, stagger: Stagger) -> NativeBox:
        """Return physical-domain bounds in CM1's native one-based indices."""

        x_upper = self.nx + (1 if stagger is Stagger.X_FACE else 0)
        y_upper = self.ny + (1 if stagger is Stagger.Y_FACE else 0)
        z_upper = self.nz + (1 if stagger is Stagger.Z_FACE else 0)
        return NativeBox(
            x=NativeRange(1, x_upper),
            y=NativeRange(1, y_upper),
            z=NativeRange(1, z_upper),
        )

    def native_allocation(self, stagger: Stagger, halo: Halo) -> NativeBox:
        """Return source-known single-rank allocation bounds.

        This implements ``ib=1-ngxy`` through ``ie=nx+ngxy`` and the
        corresponding y/z bounds from ``src/cm1.F:744-777``.  The staggered
        axis receives its native extra face/interface.
        """

        interior = self.native_interior(stagger)
        return NativeBox(
            x=NativeRange(interior.x.lower - halo.x, interior.x.upper + halo.x),
            y=NativeRange(interior.y.lower - halo.y, interior.y.upper + halo.y),
            z=NativeRange(interior.z.lower - halo.z, interior.z.upper + halo.z),
        )

    def geometry(self, stagger: Stagger, halo: Halo) -> FieldGeometry:
        return FieldGeometry(
            stagger=stagger,
            allocation=self.native_allocation(stagger, halo),
            confirmed_interior=self.native_interior(stagger),
        )

    def horizontal_coordinate_bounds(
        self,
        axis: Axis,
        location: CoordinateLocation,
        halo: Halo,
    ) -> NativeRange:
        """Return exact 1-D ``xh/xf/yh/yf`` allocation bounds."""

        if axis is Axis.X:
            width = self.nx
            halo_width = halo.x
        elif axis is Axis.Y:
            width = self.ny
            halo_width = halo.y
        else:  # pragma: no cover - StrEnum construction prevents this.
            raise AssertionError(f"unhandled axis {axis!r}")
        extra_face = 1 if location is CoordinateLocation.FACE else 0
        return NativeRange(1 - halo_width, width + halo_width + extra_face)

    def as_dict(self) -> dict[str, int]:
        return {"nx": self.nx, "ny": self.ny, "nz": self.nz}


@dataclass(frozen=True, slots=True)
class CoordinateLayout:
    native_name: str
    axis: Axis
    location: CoordinateLocation
    units: str = "m"

    def native_bounds(self, grid: GridShape, halo: Halo) -> NativeRange:
        return grid.horizontal_coordinate_bounds(self.axis, self.location, halo)


GRID_COORDINATES: tuple[CoordinateLayout, ...] = (
    CoordinateLayout("xh", Axis.X, CoordinateLocation.MASS),
    CoordinateLayout("xf", Axis.X, CoordinateLocation.FACE),
    CoordinateLayout("yh", Axis.Y, CoordinateLocation.MASS),
    CoordinateLayout("yf", Axis.Y, CoordinateLocation.FACE),
)


@dataclass(frozen=True, slots=True)
class FieldLayout:
    native_name: str
    meaning: str
    units: str
    stagger: Stagger
    live_region: LiveRegionPolicy
    persistence: str
    output_name: str
    precision: str = "float32"

    def interior_shape(self, grid: GridShape) -> tuple[int, int, int]:
        return grid.interior(self.stagger)

    def geometry(self, grid: GridShape, halo: Halo = TARGET_HALO) -> FieldGeometry:
        return grid.geometry(self.stagger, halo)


CORE_LAYOUTS: tuple[FieldLayout, ...] = (
    FieldLayout("ua", "zonal velocity A carrier", "m s-1", Stagger.X_FACE, LiveRegionPolicy.X_FACE_A_TIME_LEVEL, "restart", "u"),
    FieldLayout("u3d", "current zonal velocity", "m s-1", Stagger.X_FACE, LiveRegionPolicy.X_FACE_3D, "stage", "u"),
    FieldLayout("va", "meridional velocity A carrier", "m s-1", Stagger.Y_FACE, LiveRegionPolicy.Y_FACE_A_TIME_LEVEL, "restart", "v"),
    FieldLayout("v3d", "current meridional velocity", "m s-1", Stagger.Y_FACE, LiveRegionPolicy.Y_FACE_3D, "stage", "v"),
    FieldLayout("wa", "vertical velocity A carrier", "m s-1", Stagger.Z_FACE, LiveRegionPolicy.Z_FACE_A_TIME_LEVEL, "restart", "w"),
    FieldLayout("w3d", "current vertical velocity", "m s-1", Stagger.Z_FACE, LiveRegionPolicy.Z_FACE_3D, "stage", "w"),
    FieldLayout("ppi", "Exner perturbation A carrier", "1", Stagger.SCALAR, LiveRegionPolicy.SCALAR_A_TIME_LEVEL, "restart", "ppi"),
    FieldLayout("pp3d", "current Exner perturbation", "1", Stagger.SCALAR, LiveRegionPolicy.SCALAR_3D, "stage", "ppi"),
    FieldLayout("tha", "theta perturbation A carrier", "K", Stagger.SCALAR, LiveRegionPolicy.SCALAR_A_TIME_LEVEL, "restart", "thpert"),
    FieldLayout("th3d", "current theta perturbation", "K", Stagger.SCALAR, LiveRegionPolicy.SCALAR_3D, "stage", "thpert"),
    FieldLayout("qa", "packed moisture A carrier", "kg kg-1", Stagger.SCALAR, LiveRegionPolicy.MOIST_A_TIME_LEVEL, "restart", "semantic species"),
    FieldLayout("q3d", "packed current moisture", "kg kg-1", Stagger.SCALAR, LiveRegionPolicy.MOIST_3D, "stage", "semantic species"),
    FieldLayout("tkea", "TKE A carrier", "m2 s-2", Stagger.Z_FACE, LiveRegionPolicy.W_LEVEL_SGS_A_TIME_LEVEL, "restart", "tke"),
    FieldLayout("tke3d", "current TKE", "m2 s-2", Stagger.Z_FACE, LiveRegionPolicy.W_LEVEL_SGS_3D, "stage", "tke"),
)


# These ten scalar/momentum carriers use the source-known standard dycore
# allocations.  Moisture adds an unresolved species dimension and SGS/TKE uses
# closure-specific bounds; neither family is structurally allocated here.
STRUCTURAL_DYCORE_NAMES: tuple[str, ...] = (
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "tha",
    "th3d",
)
STRUCTURAL_DYCORE_LAYOUTS: tuple[FieldLayout, ...] = tuple(
    row for name in STRUCTURAL_DYCORE_NAMES for row in CORE_LAYOUTS if row.native_name == name
)
if tuple(row.native_name for row in STRUCTURAL_DYCORE_LAYOUTS) != STRUCTURAL_DYCORE_NAMES:
    raise AssertionError("structural dycore inventory does not resolve exactly")


def layout(name: str) -> FieldLayout:
    for row in CORE_LAYOUTS:
        if row.native_name == name:
            return row
    raise KeyError(name)


__all__ = [
    "Axis",
    "CORE_LAYOUTS",
    "CoordinateLayout",
    "CoordinateLocation",
    "FieldLayout",
    "FieldGeometry",
    "GRID_COORDINATES",
    "GridShape",
    "Halo",
    "LIVE_REGION_CONTRACTS",
    "LiveRegionPolicy",
    "NativeBox",
    "NativeRange",
    "Stagger",
    "STRUCTURAL_DYCORE_LAYOUTS",
    "STRUCTURAL_DYCORE_NAMES",
    "TARGET_HALO",
    "layout",
]
