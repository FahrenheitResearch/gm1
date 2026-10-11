"""Circulation about material circuits, against closed-form and Kelvin.

CM1 r21.1 has no circuit diagnostic, so there is no differential.  Two exact
statements stand in for one:

* a regular N-gon in solid-body rotation has circulation ``2*omega`` times its
  own polygon area, in closed form;
* Kelvin's theorem: carried by the flow of a barotropic, inviscid field, a
  material circuit's circulation does not change.  The parcels here are
  integrated by the real :mod:`cm1core.lagrangian_parcels` RK2, so this is a
  statement about the whole stack and not about the line integral alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cm1core.circulation_circuits import (
    CircuitDefinition,
    circuit_circulation,
    circulation_report,
    solid_body_circulation,
)
from cm1core.cuda_psolver3_terrain_open import CM1_OPEN_NGXY, terrain_open_sound_shapes
from cm1core.lagrangian_parcels import (
    ParcelGrid,
    ParcelVelocityField,
    _sample_uvw,
    cpu_parcel_advance,
    parcel_circuit_positions,
)

NX, NY, NZ = 40, 40, 20
DX = DY = 250.0
DZ = 250.0


def _grid() -> ParcelGrid:
    return ParcelGrid(nx=NX, ny=NY, nz=NZ, dx_m=DX, dy_m=DY, dz_m=DZ, iorigin=2)


def _rotation_field(grid: ParcelGrid, omega: float) -> ParcelVelocityField:
    ng = CM1_OPEN_NGXY
    shapes = terrain_open_sound_shapes(NX, NY, NZ)
    u = np.zeros(shapes["u3d"], dtype=np.float32)
    v = np.zeros(shapes["v3d"], dtype=np.float32)
    w = np.zeros(shapes["w3d"], dtype=np.float32)
    for k in range(u.shape[0]):
        for j in range(u.shape[1]):
            y = grid.miny_m + (j - ng + 0.5) * grid.dy_m
            u[k, j, :] = np.float32(-omega * y)
    for k in range(v.shape[0]):
        for i in range(v.shape[2]):
            x = grid.minx_m + (i - ng + 0.5) * grid.dx_m
            v[k, :, i] = np.float32(omega * x)
    return ParcelVelocityField(u, v, w)


def _sample_all(field, grid, positions):
    np_ = np
    out = np_.empty_like(positions)
    for p in range(positions.shape[1]):
        u, v, w = _sample_uvw(np_, field, grid,
                              positions[0, p], positions[1, p], positions[2, p])
        out[0, p], out[1, p], out[2, p] = u, v, w
    return out


def test_the_ngon_circulation_is_the_closed_form_two_omega_times_area():
    omega, radius, count = 0.02, 500.0, 64
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=1000.0,
        radius_m=radius, count=count,
    )
    velocities = np.empty_like(positions)
    velocities[0] = -omega * positions[1]
    velocities[1] = omega * positions[0]
    velocities[2] = 0.0
    sample = circuit_circulation(
        positions, velocities, CircuitDefinition("ring", 0, count)
    )
    exact = solid_body_circulation(omega_s_inv=omega, radius_m=radius, count=count)
    assert sample.circulation_m2_s == pytest.approx(exact, rel=1.0e-6)
    assert sample.enclosed_area_m2 == pytest.approx(
        0.5 * count * radius * radius * math.sin(2.0 * math.pi / count), rel=1.0e-6
    )
    # ...and it converges to the continuum answer as the circuit is refined.
    assert exact / (2.0 * math.pi * radius * radius * omega) == pytest.approx(1.0, rel=2e-3)


def test_kelvins_theorem_holds_for_a_circuit_carried_by_the_real_integrator():
    grid = _grid()
    omega = 0.01
    field = _rotation_field(grid, omega)
    count = 48
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=1000.0,
        radius_m=1500.0, count=count,
    )
    circuit = CircuitDefinition("ring", 0, count)
    first = circuit_circulation(positions, _sample_all(field, grid, positions), circuit)
    for _ in range(60):
        positions = cpu_parcel_advance(positions, field, grid, dt_s=2.0)
    last = circuit_circulation(positions, _sample_all(field, grid, positions), circuit)
    assert last.circulation_m2_s == pytest.approx(first.circulation_m2_s, rel=2.0e-4)
    assert last.mean_radius_m == pytest.approx(first.mean_radius_m, rel=2.0e-4)


def test_a_contracting_circuit_obeys_stokes_and_the_exact_contraction_law():
    """A converging column with constant vorticity: two exact statements.

    The field is ``u = -omega*y - a*x``, ``v = omega*x - a*y``, ``w = 2*a*z``:
    non-divergent, with ``zeta = 2*omega`` everywhere.  A material circle
    contracts as ``r(t) = r0*exp(-a t)`` in closed form, and Stokes' theorem
    on a constant-vorticity field says the circuit's circulation must equal
    ``2*omega`` times its own enclosed area at *every* time.  Both are checked,
    because together they prove the line integral and the parcel transport are
    consistent with each other.

    Kelvin's theorem is deliberately **not** asserted here.  This velocity
    field is frozen rather than evolved, so ``zeta`` is held constant while the
    area falls, and the circulation genuinely decreases; a port that claimed
    conservation here would be claiming something untrue.  The Kelvin check
    lives in the solid-body test above, where it does hold exactly.
    """

    grid = _grid()
    omega, a = 0.01, 1.2e-3
    ng = CM1_OPEN_NGXY
    shapes = terrain_open_sound_shapes(NX, NY, NZ)
    u = np.zeros(shapes["u3d"], dtype=np.float32)
    v = np.zeros(shapes["v3d"], dtype=np.float32)
    w = np.zeros(shapes["w3d"], dtype=np.float32)
    for j in range(u.shape[1]):
        y = grid.miny_m + (j - ng + 0.5) * grid.dy_m
        for i in range(u.shape[2]):
            x = grid.minx_m + (i - ng) * grid.dx_m
            u[:, j, i] = np.float32(-omega * y - a * x)
    for j in range(v.shape[1]):
        y = grid.miny_m + (j - ng) * grid.dy_m
        for i in range(v.shape[2]):
            x = grid.minx_m + (i - ng + 0.5) * grid.dx_m
            v[:, j, i] = np.float32(omega * x - a * y)
    for k in range(w.shape[0]):
        w[k, :, :] = np.float32(2.0 * a * (k - 1) * grid.dz_m)
    field = ParcelVelocityField(u, v, w)
    count = 48
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=1000.0,
        radius_m=1500.0, count=count,
    )
    circuit = CircuitDefinition("ring", 0, count)
    first = circuit_circulation(positions, _sample_all(field, grid, positions), circuit)
    for _ in range(80):
        positions = cpu_parcel_advance(positions, field, grid, dt_s=2.0)
    last = circuit_circulation(positions, _sample_all(field, grid, positions), circuit)
    elapsed = 80 * 2.0
    assert last.mean_radius_m == pytest.approx(
        first.mean_radius_m * math.exp(-a * elapsed), rel=1.0e-3
    )
    for sample in (first, last):
        assert sample.circulation_m2_s / sample.enclosed_area_m2 == pytest.approx(
            2.0 * omega, rel=1.0e-4
        )
    assert last.circulation_m2_s == pytest.approx(
        first.circulation_m2_s * math.exp(-2.0 * a * elapsed), rel=2.0e-3
    )


def test_a_circuit_with_an_unplaced_vertex_is_refused_by_name():
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=100.0, radius_m=100.0, count=8
    )
    positions[:, 3] = np.float32(np.nan)
    with pytest.raises(ValueError, match="encloses no defined area"):
        circuit_circulation(positions, np.zeros_like(positions),
                            CircuitDefinition("ring", 0, 8))


def test_a_circuit_that_runs_past_the_parcel_array_is_refused():
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=100.0, radius_m=100.0, count=8
    )
    with pytest.raises(ValueError, match="past the parcel array"):
        circuit_circulation(positions, np.zeros_like(positions),
                            CircuitDefinition("ring", 4, 8))


def test_a_two_vertex_circuit_is_refused_because_it_encloses_nothing():
    with pytest.raises(ValueError, match="at least three parcels"):
        CircuitDefinition("degenerate", 0, 2)


def test_the_report_carries_the_unvalidated_marker_and_every_circuit():
    count = 12
    positions = parcel_circuit_positions(
        center_x_m=0.0, center_y_m=0.0, center_z_m=500.0, radius_m=300.0, count=count
    )
    doubled = np.concatenate([positions, positions], axis=1)
    circuits = [CircuitDefinition("a", 0, count), CircuitDefinition("b", count, count)]
    report = circulation_report(doubled, np.zeros_like(doubled), circuits,
                                model_time_s=12.0)
    assert [c["name"] for c in report["circuits"]] == ["a", "b"]
    assert report["model_time_s"] == 12.0
    assert report["tendency_terms_supplied"] == {"baroclinic": False, "friction": False}
