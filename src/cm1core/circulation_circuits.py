"""Circulation about material circuits, integrated inline.

What this is
------------
The Markowski/Richardson circuit diagnostic: seed a closed ring of parcels,
let the flow carry it, and integrate

.. math::  C = \\oint_{\\mathcal C} \\mathbf V \\cdot d\\boldsymbol\\ell

about it at every model step.  Because the circuit is *material*, Kelvin's
theorem says :math:`C` is conserved for barotropic, inviscid flow, and
Bjerknes' theorem says that otherwise

.. math::  \\frac{dC}{dt}
           = \\oint \\frac{1}{\\rho^2}\\,\\nabla\\rho\\times\\nabla p \\cdot d\\mathbf A
             + \\oint \\mathbf F \\cdot d\\boldsymbol\\ell

so the change in a tornado's circulation is attributed, term by term, to
baroclinity and to friction.  That attribution is the question "where did this
tornado's rotation come from", and it is the reason to compute it inline: the
circuit's shape between history dumps is exactly what is lost when it is
reconstructed post hoc.

Authority
---------
**GM1 EXTENSION.**  CM1 r21.1 has no circuit or circulation diagnostic.  The
parcels that carry the circuit are CM1's (``src/parcel.F:10``), the sampling
is CM1's (``src/parcel.F:656``); the line integral and its budget are not.
It is therefore validated analytically, not differentially: solid-body
rotation has a closed-form circulation for the discrete N-gon, and a
barotropic inviscid flow must conserve it.

Discretisation
--------------
``C`` is the trapezoidal line integral over the closed polygon whose vertices
are the circuit's parcels in seeding order:

.. math::  C = \\sum_n \\tfrac12 (\\mathbf V_n + \\mathbf V_{n+1})
                    \\cdot (\\mathbf x_{n+1} - \\mathbf x_n)

which is exact for a velocity field linear along each segment.  The sum is
carried in float64 -- a 512-vertex circuit accumulates 512 cancelling terms of
similar size, and binary32 loses the answer's leading digits doing that; the
*terms* are the run's own binary32 words, only the accumulation widens.  That
is the same rule :mod:`cm1core.storm_diagnostics` already applies to its
domain reductions.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CIRCULATION_SCHEMA = "gpuwm-cm1.material-circulation/v1"

SOURCE_ANCHORS: Mapping[str, str] = {
    "parcel_carrier": "src/parcel.F:10-620",
    "parcel_sampling": "src/parcel.F:656-1690",
    "circulation": "GM1 EXTENSION: CM1 r21.1 carries no circuit diagnostic",
}


@dataclass(frozen=True, slots=True)
class CircuitDefinition:
    """One closed material circuit: a contiguous slice of the parcel array."""

    name: str
    first_parcel: int
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a circuit needs a name")
        for field_name in ("first_parcel", "count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.first_parcel < 0:
            raise ValueError("first_parcel must be >= 0")
        if self.count < 3:
            raise ValueError(
                "a closed circuit needs at least three parcels; two parcels "
                "enclose no area and its circulation is identically zero"
            )

    @property
    def stop(self) -> int:
        return self.first_parcel + self.count

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first_parcel": self.first_parcel,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class CirculationSample:
    name: str
    circulation_m2_s: float
    enclosed_area_m2: float
    perimeter_m: float
    centroid_m: tuple[float, float, float]
    mean_radius_m: float
    baroclinic_tendency_m2_s2: float
    friction_tendency_m2_s2: float

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "circulation_m2_s": self.circulation_m2_s,
            "enclosed_area_m2": self.enclosed_area_m2,
            "perimeter_m": self.perimeter_m,
            "centroid_m": list(self.centroid_m),
            "mean_radius_m": self.mean_radius_m,
            "baroclinic_tendency_m2_s2": self.baroclinic_tendency_m2_s2,
            "friction_tendency_m2_s2": self.friction_tendency_m2_s2,
        }


def circuit_circulation(
    positions: Any,
    velocities: Any,
    circuit: CircuitDefinition,
    *,
    baroclinic: Any = None,
    friction: Any = None,
) -> CirculationSample:
    """Close the line integral about one circuit.

    ``positions`` and ``velocities`` are ``(3, nparcels)`` arrays -- the parcel
    positions and the velocities already sampled at them, which the parcel
    stage produces anyway.  ``baroclinic`` and ``friction``, when given, are
    ``(3, nparcels)`` vector fields sampled at the same points; their line
    integrals are the two Bjerknes tendency terms.
    """

    np = importlib.import_module("numpy")
    lo, hi = circuit.first_parcel, circuit.stop
    if positions.shape[0] != 3 or velocities.shape[0] != 3:
        raise ValueError("positions and velocities must be (3, nparcels)")
    if hi > positions.shape[1] or hi > velocities.shape[1]:
        raise ValueError(f"circuit {circuit.name!r} runs past the parcel array")
    x = np.asarray(positions[:, lo:hi], dtype=np.float64)
    v = np.asarray(velocities[:, lo:hi], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(
            f"circuit {circuit.name!r} has a parcel with no position; a circuit "
            "whose vertices are not all placed encloses no defined area"
        )
    nxt = np.roll(x, -1, axis=1)
    dl = nxt - x
    mid_v = 0.5 * (v + np.roll(v, -1, axis=1))
    circulation = float(np.sum(mid_v * dl))
    perimeter = float(np.sum(np.sqrt(np.sum(dl * dl, axis=0))))
    # Projected area of the closed polygon, by the shoelace formula on each
    # coordinate plane; the magnitude of the vector area is the circuit's own
    # enclosed area whatever plane it has tilted into.
    area_vec = 0.5 * np.array([
        float(np.sum(x[1] * nxt[2] - x[2] * nxt[1])),
        float(np.sum(x[2] * nxt[0] - x[0] * nxt[2])),
        float(np.sum(x[0] * nxt[1] - x[1] * nxt[0])),
    ])
    area = float(np.linalg.norm(area_vec))
    centroid = tuple(float(value) for value in x.mean(axis=1))
    radius = float(np.mean(np.sqrt(np.sum(
        (x - np.asarray(centroid, dtype=np.float64)[:, None]) ** 2, axis=0
    ))))

    def line_integral(vector: Any) -> float:
        if vector is None:
            return 0.0
        block = np.asarray(vector[:, lo:hi], dtype=np.float64)
        mid = 0.5 * (block + np.roll(block, -1, axis=1))
        return float(np.sum(mid * dl))

    return CirculationSample(
        circuit.name, circulation, area, perimeter, centroid, radius,
        line_integral(baroclinic), line_integral(friction),
    )


def solid_body_circulation(*, omega_s_inv: float, radius_m: float, count: int) -> float:
    """The exact circulation of a regular ``count``-gon in solid-body rotation.

    For :math:`\\mathbf V = \\Omega\\hat z \\times \\mathbf r` the trapezoidal
    integral closes in closed form: the polygon's area is
    :math:`\\tfrac{N}{2}R^2\\sin(2\\pi/N)` and the circulation is
    :math:`2\\Omega` times it, approaching :math:`2\\pi R^2\\Omega` as
    :math:`N\\to\\infty`.  This is the analytic answer the circuit diagnostic is
    validated against; it is stated here rather than in the test so that a
    reader of the module can check the claim without reading the test.
    """

    if count < 3:
        raise ValueError("a polygon needs at least three vertices")
    area = 0.5 * count * radius_m * radius_m * math.sin(2.0 * math.pi / count)
    return 2.0 * omega_s_inv * area


def circulation_report(
    positions: Any,
    velocities: Any,
    circuits: Sequence[CircuitDefinition],
    *,
    model_time_s: float,
    baroclinic: Any = None,
    friction: Any = None,
) -> dict[str, Any]:
    """Every circuit's sample at one model time, ready for a receipt row."""

    samples = [
        circuit_circulation(positions, velocities, circuit,
                            baroclinic=baroclinic, friction=friction)
        for circuit in circuits
    ]
    return {
        "schema": CIRCULATION_SCHEMA,
        "model_time_s": float(model_time_s),
        "circuits": [sample.record() for sample in samples],
        "tendency_terms_supplied": {
            "baroclinic": baroclinic is not None,
            "friction": friction is not None,
        },
        "cm1_authority": dict(SOURCE_ANCHORS),
    }


__all__ = [
    "CIRCULATION_SCHEMA", "CircuitDefinition", "CirculationSample",
    "SOURCE_ANCHORS", "circuit_circulation", "circulation_report",
    "solid_body_circulation",
]
