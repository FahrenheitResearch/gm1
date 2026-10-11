"""Immutable uniform Cartesian C-grid dimensions and storage geometry.

Shapes use Python ``(z, y, x)`` order.  A physical shape says nothing about
which stored halo words are source-live.  Callers choose the actual array
family's halo widths explicitly; this module never fills or extrapolates a
scientific field.  CM1 r21.1 ``cm1.F:744-777`` establishes three lateral
ghost cells for fifth-order advection and one vertical ghost cell.  Its
staggered allocations add the respective u/v/w face independently.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cm1_vertical_grid import VerticalLevels


Stagger = Literal["mass", "u", "v", "w"]
_STAGGER_ZYX = {"mass": (0, 0, 0), "u": (0, 0, 1), "v": (0, 1, 0), "w": (1, 0, 0)}


def _extent(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class CartesianGridGeometry:
    """Physical cell counts, horizontal spacing and one vertical coordinate.

    ``dz_m`` is CM1's namelist ``dz``: the *metric reference* every ``mh``/
    ``mf`` word divides by (``param.F:8170,8176``), not necessarily any layer's
    thickness.  With ``vertical`` left unset the coordinate is the uniform
    ``stretch_z=0`` one, for which ``dz_m`` is also the layer thickness and
    ``ztop_m`` is ``nz*dz_m`` (``param.F:2608``); supply a
    :class:`~cm1core.cm1_vertical_grid.VerticalLevels` to carry a stretched
    grid instead.
    """

    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float
    vertical: "VerticalLevels | None" = None

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = _extent(name, getattr(self, name))
            # Existing CUDA dimension arguments are unsigned 32-bit.  Leave
            # room for the largest source halo plus the staggered extra face.
            if value > (1 << 32) - 8:
                raise ValueError(f"{name} plus source halos exceeds CUDA uint32")
        for name in ("dx_m", "dy_m", "dz_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        for count, spacing in zip((self.nx, self.ny, self.nz), self.spacing_m):
            if not math.isfinite(count * spacing):
                raise ValueError("Cartesian domain extent must be finite")
        if math.prod((self.nz + 3, self.ny + 7, self.nx + 7)) > sys.maxsize // 4:
            raise ValueError("Cartesian float32 allocation exceeds host size_t")
        if self.vertical is None:
            # param.F:7789-7811.  The uniform coordinate is a pure function of
            # nz and dz_m, so it is built on demand by ``levels`` rather than
            # here: this constructor still has to admit the sub-stencil grids
            # that existing callers build only to assert a stencil refusal,
            # and the Fornberg weights of param.F:8985-9060 need four levels.
            return
        from .cm1_vertical_grid import VerticalLevels

        if not isinstance(self.vertical, VerticalLevels):
            raise TypeError("vertical must be cm1_vertical_grid.VerticalLevels")
        if self.vertical.nk != self.nz:
            raise ValueError("declared vertical level count must equal grid.nz")
        if self.vertical.dz_reference_m != self.dz_m:
            raise ValueError(
                "declared vertical dz reference must equal grid.dz_m; "
                "param.F:8170 divides every mh word by exactly this number"
            )

    @property
    def dx(self) -> float:
        return self.dx_m

    @property
    def dy(self) -> float:
        return self.dy_m

    @property
    def dz(self) -> float:
        return self.dz_m

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        """Return ``(dx, dy, dz)`` in metres."""

        return self.dx_m, self.dy_m, self.dz_m

    @property
    def ztop_m(self) -> float:
        """Model top: ``param.F:2608``'s ``nz * dz`` unless the grid stretches."""

        if self.vertical is None:
            return self.nz * self.dz_m
        return self.vertical.ztop_m

    @property
    def stretch_z(self) -> int:
        """CM1 ``input.F:283``'s branch selector for this grid."""

        return 0 if self.vertical is None else self.vertical.stretch_z

    @property
    def uniform_z(self) -> bool:
        return self.stretch_z == 0

    @property
    def levels(self) -> "VerticalLevels":
        """This grid's complete CM1 vertical coordinate.

        An undeclared coordinate is ``param.F:7789-7811``'s uniform one, built
        here rather than stored so that a uniform geometry keeps the exact
        six-word identity, hash and ``replace()`` behaviour it has always had.
        """

        if self.vertical is not None:
            return self.vertical
        from .cm1_vertical_grid import VerticalGridError, uniform_vertical_levels

        try:
            return uniform_vertical_levels(self.nz, self.dz_m)
        except VerticalGridError as error:
            raise ValueError(str(error)) from None

    @property
    def mass_heights_m(self) -> tuple[float, ...]:
        """``zh(1..nz)`` in metres above ground."""

        return self.levels.mass_heights_m

    @property
    def face_heights_m(self) -> tuple[float, ...]:
        """``zf(1..nz+1)`` in metres above ground."""

        return self.levels.face_heights_m

    @property
    def spacings_m(self) -> tuple[float, ...]:
        """``zf(k+1)-zf(k)`` for ``k=1..nz``: the physical layer depths."""

        return self.levels.spacings_m

    @property
    def lowest_mass_level_m(self) -> float:
        """``zh(1)``.  ``0.5*dz_m`` on a uniform grid, far lower on a stretched one."""

        if self.vertical is None:
            return 0.5 * self.dz_m
        return self.vertical.lowest_mass_level_m

    @property
    def highest_mass_level_m(self) -> float:
        """``zh(nz)``."""

        if self.vertical is None:
            return self.ztop_m - 0.5 * self.dz_m
        return self.vertical.highest_mass_level_m

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.nz, self.ny, self.nx

    @property
    def mass_shape(self) -> tuple[int, int, int]:
        return self.shape_zyx

    @property
    def u_shape(self) -> tuple[int, int, int]:
        return self.nz, self.ny, self.nx + 1

    @property
    def v_shape(self) -> tuple[int, int, int]:
        return self.nz, self.ny + 1, self.nx

    @property
    def w_shape(self) -> tuple[int, int, int]:
        return self.nz + 1, self.ny, self.nx

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    def physical_shape(self, stagger: Stagger = "mass") -> tuple[int, int, int]:
        try:
            extra = _STAGGER_ZYX[stagger]
        except (KeyError, TypeError):
            raise ValueError(f"unknown C-grid stagger {stagger!r}") from None
        return tuple(size + face for size, face in zip(self.shape_zyx, extra))

    def padded_shape(
        self, stagger: Stagger = "mass", *, halo_xy: int, halo_z: int
    ) -> tuple[int, int, int]:
        """Stored shape for an explicitly selected symmetric halo family."""

        halo_xy = _extent("halo_xy", halo_xy, minimum=0)
        halo_z = _extent("halo_z", halo_z, minimum=0)
        nz, ny, nx = self.physical_shape(stagger)
        return nz + 2 * halo_z, ny + 2 * halo_xy, nx + 2 * halo_xy

    def physical_slices(
        self, stagger: Stagger = "mass", *, halo_xy: int, halo_z: int
    ) -> tuple[slice, slice, slice]:
        """Physical rectangle in a named caller-selected storage family.

        Pass ``halo_xy=3, halo_z=1`` for the established scalar/velocity
        allocations and ``halo_xy=1, halo_z=1`` for the compact pressure
        family.  This is an extraction rectangle, not a source-live halo mask.
        """

        self.padded_shape(stagger, halo_xy=halo_xy, halo_z=halo_z)
        nz, ny, nx = self.physical_shape(stagger)
        return (
            slice(halo_z, halo_z + nz),
            slice(halo_xy, halo_xy + ny),
            slice(halo_xy, halo_xy + nx),
        )

    def require_stencil(self, *, radius_xy: int, minimum_nz: int) -> None:
        """Require the selected stencil's domain extents before allocation."""

        radius_xy = _extent("radius_xy", radius_xy, minimum=0)
        minimum_nz = _extent("minimum_nz", minimum_nz)
        if min(self.nx, self.ny) < 2 * radius_xy + 1:
            raise ValueError("Cartesian horizontal domain is smaller than the stencil")
        if self.nz < minimum_nz:
            raise ValueError(f"Cartesian vertical stencil requires nz >= {minimum_nz}")

    def record(self) -> dict[str, object]:
        """The receipt words.

        A ``stretch_z=0`` grid publishes exactly the six words and ``ztop_m``
        it always has, so every uniform receipt keeps its established hash; a
        stretched grid adds its own coordinate under ``vertical``.
        """

        body: dict[str, object] = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "dz_m": self.dz_m,
            "ztop_m": self.ztop_m,
        }
        if self.vertical is not None and self.vertical.stretch_z != 0:
            body["vertical"] = self.vertical.record()
        return body


__all__ = ("CartesianGridGeometry", "Stagger")
