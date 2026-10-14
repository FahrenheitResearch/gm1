"""A hurricane's rings: means around its pressure centre, by radius and height.

The method is the one ``records/GM1-HURRICANE-RUNS.json`` measured GM1's and CM1's hurricane runs with (one script
read every run's CM1-format frames), so a picture of a run and the record say the same numbers:

* **the centre**: the lowest pressure at the lowest model level after two passes of a five-point smoother (each
  cell averaged with its four neighbours; cells on the area's edge repeat themselves), at that cell's centre;
* **the rings**: one grid spacing wide.  Ring ``n`` holds the cells whose centres lie ``n dx`` to ``(n + 1) dx``
  from the centre (its middle radius is ``(n + 0.5) dx``, so the first ring is the centre cell alone); the rings
  reach half the area's shorter side, and a ring the area's edge cuts averages the cells it has;
* **the swirling wind**: the wind across each cell's line to the centre, ``-u sin(a) + v cos(a)`` with ``a`` the
  cell's direction from the centre counted counter-clockwise from east (so the swirl of a northern-hemisphere
  hurricane is positive), from the ground-relative wind at the cell's centre; the **radial wind**,
  ``u cos(a) + v sin(a)``, is negative where the air flows in;
* **the radius of maximum wind**: the middle radius of the ring whose mean swirling wind at the lowest model level
  is strongest;
* **the warm core**: at each level, the ring-mean potential temperature minus its mean over the rings between 300
  and 450 km from the centre (:data:`FAR_RING`; an area too small for them uses the outer third of the rings it
  has, and says so).  Its number is the mean of the rings within 40 km of the centre, at the level where that is
  largest, with that level's height.  The base state is the same at every place on a level, so θ′ (the frame's
  θ minus the base state) gives the same differences as θ.

Plain numpy, one level at a time, on the CPU.  :func:`of_frame` reads a quick-look frame; :func:`of_levels` takes
any source of levels, which is how ``tools/generate_hurricane_rings.py`` reads CM1's own netCDF output the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

#: Two passes of the five-point smoother before the lowest pressure is taken (the record's method).
SMOOTH_PASSES = 2
#: The warm core's number is the mean of the rings whose middle lies within this radius (m) of the centre.
CORE_RADIUS = 40_000.0
#: The rings the warm core is measured against: their middles lie between these radii (m), exclusive.
FAR_RING = (300_000.0, 450_000.0)


def smooth(field: np.ndarray, passes: int = SMOOTH_PASSES) -> np.ndarray:
    """The five-point smoother: each cell the mean of itself and its four neighbours, ``passes`` times, in double
    precision; a cell on the area's edge stands in for its missing neighbour."""

    a = np.asarray(field, dtype=np.float64)
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 5.0
    return a


def pressure_centre(pressure: np.ndarray) -> tuple[int, int]:
    """(j, i) of the lowest smoothed pressure of one level."""

    s = smooth(pressure)
    j, i = np.unravel_index(int(np.argmin(s)), s.shape)
    return int(j), int(i)


@dataclass
class Rings:
    """The ring means of one saved time.  Arrays are ``(level, ring)``; radii and heights in metres."""

    centre_ji: tuple[int, int]
    #: The centre in the area's own metres from its south-west corner (the cell centre).
    centre_m: tuple[float, float]
    dr: float
    radius_m: np.ndarray
    count: np.ndarray
    z_m: np.ndarray
    swirl: np.ndarray
    radial: np.ndarray
    theta: np.ndarray
    #: The radii (m) whose rings the warm core is measured against, and whether they are :data:`FAR_RING`.
    far_m: tuple[float, float]
    far_is_standard: bool

    @property
    def rings(self) -> int:
        return int(self.radius_m.size)

    @property
    def warm(self) -> np.ndarray:
        """The ring-mean potential temperature minus its mean over the far rings, at each level (K)."""

        far = (self.radius_m > self.far_m[0]) & (self.radius_m < self.far_m[1])
        return self.theta - self.theta[:, far].mean(axis=1, keepdims=True)

    @property
    def rmw_m(self) -> float:
        """The radius of maximum wind: the middle radius of the ring with the strongest mean swirling wind at the
        lowest level."""

        return float(self.radius_m[int(np.argmax(self.swirl[0]))])

    @property
    def swirl_max_lowest(self) -> float:
        return float(self.swirl[0].max())

    def strongest_swirl(self) -> tuple[float, float, float]:
        """(m/s, radius m, height m) of the strongest mean swirling wind anywhere."""

        k, n = np.unravel_index(int(np.argmax(self.swirl)), self.swirl.shape)
        return float(self.swirl[k, n]), float(self.radius_m[n]), float(self.z_m[k])

    def warm_core(self) -> tuple[float, float, int]:
        """(K, height m, level) of the warm core: the mean of the rings within :data:`CORE_RADIUS` at the level
        where it is largest."""

        core = self.radius_m < CORE_RADIUS
        profile = self.warm[:, core].mean(axis=1)
        k = int(np.argmax(profile))
        return float(profile[k]), float(self.z_m[k]), k


def far_ring(rings: int, dr: float) -> tuple[tuple[float, float], bool]:
    """The radii of the rings the warm core is measured against: :data:`FAR_RING` when the rings reach past it,
    else the outer third of the rings there are."""

    reach = rings * dr
    if reach > FAR_RING[1]:
        return FAR_RING, True
    return (reach * 2.0 / 3.0, reach), False


def of_levels(level: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]], *, nz: int, ny: int, nx: int,
              dx: float, dy: float, z_m: np.ndarray, centre_ji: tuple[int, int]) -> Rings:
    """The ring means about ``centre_ji`` of the levels ``level(k) -> (u, v, theta)``, each ``(ny, nx)`` at the
    cells' centres: the ground-relative wind (m/s) and the potential temperature or its departure from the base
    state (K)."""

    j0, i0 = centre_ji
    x = (np.arange(nx, dtype=np.float64) - i0) * dx
    y = (np.arange(ny, dtype=np.float64) - j0) * dy
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)
    dr = float(dx)
    rings = int(min((nx - 1) * dx, (ny - 1) * dy) / 2.0 / dr)
    if rings < 2:
        raise ValueError("the area is too small for rings around the centre")
    which = np.clip((R / dr).astype(np.int64), 0, rings).ravel()
    count = np.bincount(which, minlength=rings + 1)[:rings]
    safe = np.maximum(R, 1e-6)
    cos, sin = X / safe, Y / safe
    swirl = np.zeros((nz, rings))
    radial = np.zeros((nz, rings))
    theta = np.zeros((nz, rings))

    def mean(values: np.ndarray) -> np.ndarray:
        return np.bincount(which, np.asarray(values, dtype=np.float64).ravel(), minlength=rings + 1)[:rings] / count

    for k in range(nz):
        u, v, th = level(k)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        swirl[k] = mean(-u * sin + v * cos)
        radial[k] = mean(u * cos + v * sin)
        theta[k] = mean(th)
    far, standard = far_ring(rings, dr)
    return Rings(centre_ji=(int(j0), int(i0)), centre_m=((i0 + 0.5) * dx, (j0 + 0.5) * dy), dr=dr,
                 radius_m=(np.arange(rings) + 0.5) * dr, count=count, z_m=np.asarray(z_m, dtype=np.float64)[:nz],
                 swirl=swirl, radial=radial, theta=theta, far_m=far, far_is_standard=standard)


def of_frame(qf: Any) -> Rings | None:
    """The ring means of a quick-look frame (:class:`.frame.QuickFrame`) about its pressure centre, or None when
    the frame carries no pressure."""

    p0 = qf.pressure_level(0)
    if p0 is None:
        return None
    centre = pressure_centre(p0)

    def level(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, v = qf.uv_level(k, ground=True)
        return u, v, qf.theta_prime(k)

    return of_levels(level, nz=qf.nz, ny=qf.ny, nx=qf.nx, dx=qf.dx, dy=qf.dy, z_m=qf.zh, centre_ji=centre)


__all__ = ["CORE_RADIUS", "FAR_RING", "Rings", "far_ring", "of_frame", "of_levels", "pressure_centre", "smooth"]
