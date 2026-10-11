"""The Eulerian vorticity budget, checked against closed-form answers.

CM1 r21.1 has no vorticity budget, so there is no differential to run.  What
there is instead is exactness: every derivative here is a centred difference,
so for a velocity field **linear** in ``(x, y, z)`` every component and every
term of the budget has a closed form, and the port must reproduce it to
binary32 rounding.  Solid-body rotation -- non-divergent, no shear, no
solenoid -- is the negative control whose whole budget is identically zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from cm1core.cuda_psolver3_terrain_open import CM1_OPEN_NGXY, terrain_open_sound_shapes
from cm1core.vorticity_budget import (
    VORTICITY_BUDGET_FIELD_NAMES,
    VORTICITY_COMPONENT_NAMES,
    cpu_vorticity_budget,
)

NX, NY, NZ = 14, 13, 12
DX, DY, DZ = 500.0, 400.0, 300.0
MINX, MINY = -0.5 * DX * NX, -0.5 * DY * NY


def _linear_carriers(matrix, offset):
    """Fill the three padded carriers with ``V_i = A_ij x_j + b_i``."""

    ng = CM1_OPEN_NGXY
    shapes = terrain_open_sound_shapes(NX, NY, NZ)
    u = np.zeros(shapes["u3d"], dtype=np.float32)
    v = np.zeros(shapes["v3d"], dtype=np.float32)
    w = np.zeros(shapes["w3d"], dtype=np.float32)
    for array, row, stagger in ((u, 0, "u"), (v, 1, "v"), (w, 2, "w")):
        for k in range(array.shape[0]):
            for j in range(array.shape[1]):
                for i in range(array.shape[2]):
                    ci, cj, ck = i - ng + 1, j - ng + 1, k
                    x = MINX + ((ci - 1) if stagger == "u" else (ci - 0.5)) * DX
                    y = MINY + ((cj - 1) if stagger == "v" else (cj - 0.5)) * DY
                    z = ((ck - 1) if stagger == "w" else (ck - 0.5)) * DZ
                    array[k, j, i] = np.float32(
                        matrix[row, 0] * x + matrix[row, 1] * y
                        + matrix[row, 2] * z + offset[row]
                    )
    return u, v, w


def test_every_component_and_term_is_exact_for_a_linear_velocity_field():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(3, 3)) * 1.0e-3
    offset = rng.normal(size=3)
    u, v, w = _linear_carriers(matrix, offset)
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ
    )
    xv = matrix[2, 1] - matrix[1, 2]
    yv = matrix[0, 2] - matrix[2, 0]
    zv = matrix[1, 0] - matrix[0, 1]
    div = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    for name, exact in (("xvort", xv), ("yvort", yv), ("zvort", zv)):
        seen = getattr(result.components, name)
        assert np.max(np.abs(seen - exact)) / abs(exact) < 1.0e-5, name
    expected = {
        "stretching_x": xv * matrix[0, 0],
        "tilting_x": yv * matrix[0, 1] + zv * matrix[0, 2],
        "divergence_x": -xv * div,
        "stretching_y": yv * matrix[1, 1],
        "tilting_y": xv * matrix[1, 0] + zv * matrix[1, 2],
        "divergence_y": -yv * div,
        "stretching_z": zv * matrix[2, 2],
        "tilting_z": xv * matrix[2, 0] + yv * matrix[2, 1],
        "divergence_z": -zv * div,
    }
    scale = max(abs(value) for value in expected.values())
    for name, value in expected.items():
        assert np.max(np.abs(result.terms[name] - value)) / scale < 1.0e-5, name


def test_solid_body_rotation_has_an_identically_zero_budget():
    omega = 0.01
    matrix = np.array([[0.0, -omega, 0.0], [omega, 0.0, 0.0], [0.0, 0.0, 0.0]])
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ
    )
    assert np.max(np.abs(result.components.zvort - 2.0 * omega)) < 1.0e-8
    assert np.max(np.abs(result.components.xvort)) < 1.0e-10
    assert np.max(np.abs(result.components.yvort)) < 1.0e-10
    for name in VORTICITY_BUDGET_FIELD_NAMES:
        assert np.max(np.abs(result.terms[name])) < 1.0e-10, name


def test_pure_vertical_stretching_amplifies_exactly_as_the_closed_form_says():
    """The tornado's own term: a non-divergent converging column.

    ``u = a x``, ``v = a y``, ``w = -2 a z`` plus solid-body rotation is
    non-divergent, so the whole vertical budget is the stretching term
    ``zeta * dw/dz = -2 a * 2 omega`` and nothing else.
    """

    a, omega = 2.0e-3, 5.0e-3
    matrix = np.array([[a, -omega, 0.0], [omega, a, 0.0], [0.0, 0.0, -2.0 * a]])
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ
    )
    assert np.max(np.abs(result.terms["stretching_z"] - 2.0 * omega * (-2.0 * a))) < 1e-9
    assert np.max(np.abs(result.terms["tilting_z"])) < 1.0e-10
    assert np.max(np.abs(result.terms["divergence_z"])) < 1.0e-10


def test_tilting_converts_horizontal_vorticity_exactly():
    """Environmental streamwise vorticity tilted by a horizontal w gradient."""

    shear, gradient = 4.0e-3, 1.0e-3
    matrix = np.zeros((3, 3))
    matrix[0, 2] = shear       # u = S z  -> yvort = S
    matrix[2, 0] = gradient    # w = G x  -> yvort = S - G, tilting_z = yvort*dw/dy
    matrix[2, 1] = gradient    # w += G y -> xvort = -G ... both cross terms live
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ
    )
    xv, yv = matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0]
    exact = xv * matrix[2, 0] + yv * matrix[2, 1]
    assert np.max(np.abs(result.terms["tilting_z"] - exact)) / abs(exact) < 1.0e-5
    assert np.max(np.abs(result.terms["stretching_z"])) < 1.0e-12


def test_the_solenoid_is_the_baroclinic_cross_product_it_claims_to_be():
    matrix = np.zeros((3, 3))
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    rho = np.empty((NZ, NY, NX), dtype=np.float32)
    prs = np.empty((NZ, NY, NX), dtype=np.float32)
    alpha, beta = 1.0e-4, 5.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                x = MINX + (i + 0.5) * DX
                y = MINY + (j + 0.5) * DY
                rho[k, j, i] = np.float32(1.0 + alpha * x)
                prs[k, j, i] = np.float32(1.0e5 + beta * y)
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, rho=rho, prs=prs,
        nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
    )
    inner = (slice(1, NZ - 1), slice(1, NY - 1), slice(1, NX - 1))
    exact = alpha * beta / (rho[inner] ** 2)
    assert np.max(np.abs(result.terms["solenoid_z"][inner] - exact)) / float(
        np.max(np.abs(exact))
    ) < 1.0e-4
    assert result.one_sided_words > 0
    assert result.record()["one_sided_solenoid_words"] == result.one_sided_words


def test_friction_is_the_curl_of_the_supplied_momentum_tendency():
    matrix = np.zeros((3, 3))
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    tend_u = np.zeros((NZ, NY, NX), dtype=np.float32)
    tend_v = np.zeros((NZ, NY, NX), dtype=np.float32)
    tend_w = np.zeros((NZ, NY, NX), dtype=np.float32)
    slope = 3.0e-4
    for j in range(NY):
        tend_u[:, j, :] = np.float32(slope * (MINY + (j + 0.5) * DY))
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, uten=tend_u, vten=tend_v, wten=tend_w,
        nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ,
    )
    # curl_z F = dFv/dx - dFu/dy = -slope
    assert np.max(np.abs(result.terms["friction_z"] + slope)) / slope < 1.0e-4


def test_omitting_rho_prs_or_the_tendency_publishes_zero_and_says_so():
    matrix = np.zeros((3, 3))
    u, v, w = _linear_carriers(matrix, np.zeros(3))
    result = cpu_vorticity_budget(
        u=u, v=v, w=w, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ
    )
    assert result.one_sided_words == 0
    for axis in ("x", "y", "z"):
        assert not result.terms[f"solenoid_{axis}"].any()
        assert not result.terms[f"friction_{axis}"].any()


def test_the_published_field_inventory_is_three_components_and_fifteen_terms():
    assert VORTICITY_COMPONENT_NAMES == ("xvort", "yvort", "zvort")
    assert len(VORTICITY_BUDGET_FIELD_NAMES) == 15
    assert VORTICITY_BUDGET_FIELD_NAMES[:5] == (
        "stretching_x", "tilting_x", "divergence_x", "solenoid_x", "friction_x",
    )


def test_the_corner_vorticity_is_exact_and_is_not_the_scalar_point_average():
    """``parcel.F:1064-1066`` builds zvort at corners and samples it there.

    ``misclibs.F:2053``'s four-point average to scalar points is a *different*
    field; averaging first and interpolating second is not the same number as
    interpolating the corners, which is why CM1 keeps both and why this port
    does too.  The corner field is exact for a linear wind, which pins the
    index mapping between CM1's one-based ``(i, j)`` and the padded carrier.
    """

    from cm1core.vorticity_budget import corner_vertical_vorticity

    matrix = np.zeros((3, 3))
    matrix[0, 1] = -3.0e-3   # du/dy
    matrix[1, 0] = 7.0e-3    # dv/dx
    u, v, _ = _linear_carriers(matrix, np.zeros(3))
    corner = corner_vertical_vorticity(
        u, v, nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, xp=np
    )
    assert corner.shape == (NZ, NY + 1, NX + 1)
    exact = matrix[1, 0] - matrix[0, 1]
    assert np.max(np.abs(corner - exact)) / abs(exact) < 1.0e-5
