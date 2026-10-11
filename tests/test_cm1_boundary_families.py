"""The CM1 r21.1 lateral boundary families, checked against ``src/bc.F``.

The sealed all-open operator in :mod:`cm1core.cuda_sound3_open_boundary`
already carries a source-exact CPU transcription of the ``wbc=ebc=sbc=nbc=2``
family.  These tests hold the general operator to it word for word, then check
the three families CM1 has that this port could not previously express.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cm1core.cuda_boundary_families import (
    CUDA_BOUNDARY_FAMILIES_KERNEL_NAMES,
    CUDA_BOUNDARY_FAMILIES_SOURCE,
    CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256,
    KIND_SCALAR,
    KIND_U,
    KIND_V,
    KIND_W,
    OPEN_RADIATIVE,
    PERIODIC,
    RIGID_FREE_SLIP,
    RIGID_NO_SLIP,
    LateralBoundarySelection,
    cpu_family_sn_inplace,
    cpu_family_we_inplace,
    cpu_nudgeobc_we_inplace,
    cpu_pressure_sn_inplace,
    cpu_pressure_we_inplace,
    cpu_radbcew_inplace,
    cpu_radbcns_inplace,
    cpu_restrict_openbc_sn_inplace,
    cpu_restrict_openbc_we_inplace,
    family_array_shape,
)
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    cpu_bcp_open_inplace,
    cpu_bcs_open_inplace,
    cpu_bcu_open_inplace,
    cpu_bcv_open_inplace,
    cpu_bcw_open_inplace,
)


NI, NJ, NK = 9, 8, 6
NG = CM1_OPEN_NGXY


def _noise(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal(shape).astype(np.float32))


def test_packaged_source_matches_its_pin_and_declares_every_entry_point() -> None:
    assert hashlib.sha256(CUDA_BOUNDARY_FAMILIES_SOURCE.encode()).hexdigest() == (
        CUDA_BOUNDARY_FAMILIES_SOURCE_SHA256
    )
    for name in CUDA_BOUNDARY_FAMILIES_KERNEL_NAMES:
        assert f"void {name}(" in CUDA_BOUNDARY_FAMILIES_SOURCE


@pytest.mark.parametrize(
    "kind, sealed",
    [
        (KIND_SCALAR, cpu_bcs_open_inplace),
        (KIND_U, cpu_bcu_open_inplace),
        (KIND_W, cpu_bcw_open_inplace),
    ],
)
def test_open_family_reproduces_the_sealed_transcription_word_for_word(kind, sealed) -> None:
    shape = family_array_shape(kind, ni=NI, nj=NJ, nk=NK, ng=NG)
    general = _noise(shape, 11 + kind)
    reference = general.copy()
    cpu_family_we_inplace(general, ni=NI, nj=NJ, nk=NK, ng=NG, kind=kind, wbc=2, ebc=2)
    cpu_family_sn_inplace(general, ni=NI, nj=NJ, nk=NK, ng=NG, kind=kind, sbc=2, nbc=2)
    if kind == KIND_W:
        sealed(reference, ni=NI, nj=NJ, nk=NK, flag=0)
    else:
        sealed(reference, ni=NI, nj=NJ, nk=NK)
    assert general.tobytes() == reference.tobytes()


def test_open_v_family_reproduces_the_sealed_transcription_in_cm1_order() -> None:
    # src/bc.F:722-934 does south and north first, then west and east; the
    # port's launch order already matches, so the general operator is held to
    # the same sequence rather than to a reordered one.
    shape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    general = _noise(shape, 21)
    reference = general.copy()
    cpu_family_sn_inplace(general, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_V, sbc=2, nbc=2)
    cpu_family_we_inplace(general, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_V, wbc=2, ebc=2)
    cpu_bcv_open_inplace(reference, ni=NI, nj=NJ, nk=NK)
    assert general.tobytes() == reference.tobytes()


def test_open_pressure_family_reproduces_the_sealed_transcription() -> None:
    general = _noise((NK + 2, NJ + 2, NI + 2), 31)
    reference = general.copy()
    cpu_pressure_we_inplace(general, ni=NI, nj=NJ, nk=NK, wbc=2, ebc=2)
    cpu_pressure_sn_inplace(general, ni=NI, nj=NJ, nk=NK, sbc=2, nbc=2)
    cpu_bcp_open_inplace(reference, ni=NI, nj=NJ, nk=NK)
    assert general.tobytes() == reference.tobytes()


def test_periodic_scalar_wraps_exactly_as_bc_f_180_364() -> None:
    shape = family_array_shape(KIND_SCALAR, ni=NI, nj=NJ, nk=NK, ng=NG)
    a = _noise(shape, 41)
    cpu_family_we_inplace(a, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, wbc=1, ebc=1)
    for k in range(1, NK + 1):
        for i in range(1, NG + 1):
            # s(1-i) = s(ni+1-i) and s(ni+i) = s(i)
            assert a[k, :, NG - 1 + (1 - i)].tobytes() == a[k, :, NG - 1 + (NI + 1 - i)].tobytes()
            assert a[k, :, NG - 1 + (NI + i)].tobytes() == a[k, :, NG - 1 + i].tobytes()


def test_periodic_u_wraps_on_the_staggered_face_as_bc_f_528_714() -> None:
    shape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    a = _noise(shape, 51)
    cpu_family_we_inplace(a, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_U, wbc=1, ebc=1)
    for i in range(1, NG + 1):
        assert a[1:NK + 1, :, NG - 1 + (1 - i)].tobytes() == a[1:NK + 1, :, NG - 1 + (NI + 1 - i)].tobytes()
        assert a[1:NK + 1, :, NG - 1 + (NI + 1 + i)].tobytes() == a[1:NK + 1, :, NG - 1 + (1 + i)].tobytes()


def test_periodic_fill_reaches_the_full_padded_row_range_open_never_touches() -> None:
    # src/bc.F's periodic branch loops j=jb,je; the open branch loops j=0,nj+1.
    # The outermost padded rows are the difference, and a stencil that reads a
    # corner halo reads exactly those.
    shape = family_array_shape(KIND_SCALAR, ni=NI, nj=NJ, nk=NK, ng=NG)
    a = _noise(shape, 61)
    open_copy = a.copy()
    cpu_family_we_inplace(a, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, wbc=1, ebc=1)
    cpu_family_we_inplace(open_copy, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, wbc=2, ebc=2)
    assert not np.array_equal(a[1:NK + 1, 0, 0], open_copy[1:NK + 1, 0, 0])


def test_rigid_free_slip_mirrors_scalars_and_rigid_u_is_antisymmetric_about_a_zero_wall() -> None:
    shape = family_array_shape(KIND_SCALAR, ni=NI, nj=NJ, nk=NK, ng=NG)
    s = _noise(shape, 71)
    cpu_family_we_inplace(s, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, wbc=3, ebc=3)
    # src/bc.F's rigid branch loops j=0,nj+1; rows outside it are untouched.
    srows = slice(NG - 1, NG - 1 + NJ + 2)
    for i in range(1, NG + 1):
        assert s[1:NK + 1, srows, NG - 1 + (1 - i)].tobytes() == s[1:NK + 1, srows, NG - 1 + i].tobytes()
        assert (
            s[1:NK + 1, srows, NG - 1 + (NI + i)].tobytes()
            == s[1:NK + 1, srows, NG - 1 + (NI + 1 - i)].tobytes()
        )
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    u = _noise(ushape, 81)
    cpu_family_we_inplace(u, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_U, wbc=4, ebc=4)
    rows = slice(NG - 1, NG - 1 + NJ + 2)
    assert np.all(u[1:NK + 1, rows, NG] == np.float32(0.0))
    assert np.all(u[1:NK + 1, rows, NI + NG] == np.float32(0.0))
    for i in range(1, NG + 1):
        np.testing.assert_array_equal(
            u[1:NK + 1, rows, NG - 1 + (1 - i)], -u[1:NK + 1, rows, NG - 1 + (1 + i)]
        )
        np.testing.assert_array_equal(
            u[1:NK + 1, rows, NG - 1 + (NI + 1 + i)], -u[1:NK + 1, rows, NG - 1 + (NI + 1 - i)]
        )


def test_rigid_v_wall_is_zero_and_antisymmetric_on_its_own_staggered_axis() -> None:
    shape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    v = _noise(shape, 91)
    cpu_family_sn_inplace(v, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_V, sbc=3, nbc=3)
    cols = slice(NG - 1, NG - 1 + NI + 2)
    assert np.all(v[1:NK + 1, NG, cols] == np.float32(0.0))
    assert np.all(v[1:NK + 1, NJ + NG, cols] == np.float32(0.0))
    for j in range(1, NG + 1):
        np.testing.assert_array_equal(
            v[1:NK + 1, NG - 1 + (1 - j), cols], -v[1:NK + 1, NG - 1 + (1 + j), cols]
        )


def test_mixed_family_leaves_each_axis_with_its_own_branch() -> None:
    # CM1's own squall-line deck: open in x, periodic in y.
    shape = family_array_shape(KIND_SCALAR, ni=NI, nj=NJ, nk=NK, ng=NG)
    a = _noise(shape, 101)
    cpu_family_we_inplace(a, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, wbc=2, ebc=2)
    cpu_family_sn_inplace(a, ni=NI, nj=NJ, nk=NK, ng=NG, kind=KIND_SCALAR, sbc=1, nbc=1)
    ks = slice(1, NK + 1)
    # x is a zero-gradient copy of the first interior column
    np.testing.assert_array_equal(a[ks, NG, 0], a[ks, NG, NG])
    # y wraps
    np.testing.assert_array_equal(a[ks, NG - 1, NG], a[ks, NG - 1 + NJ, NG])


def test_periodic_pressure_wraps_and_open_pressure_copies() -> None:
    p = _noise((NK + 2, NJ + 2, NI + 2), 111)
    q = p.copy()
    cpu_pressure_we_inplace(p, ni=NI, nj=NJ, nk=NK, wbc=1, ebc=1)
    cpu_pressure_we_inplace(q, ni=NI, nj=NJ, nk=NK, wbc=2, ebc=2)
    ks = slice(1, NK + 1)
    np.testing.assert_array_equal(p[ks, :, 0], p[ks, :, NI])
    np.testing.assert_array_equal(p[ks, :, NI + 1], p[ks, :, 1])
    np.testing.assert_array_equal(q[ks, :, 0], q[ks, :, 1])
    np.testing.assert_array_equal(q[ks, :, NI + 1], q[ks, :, NI])


def test_radbcew_is_signed_and_clamped_exactly_as_bc_f_1948() -> None:
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    ua = _noise(ushape, 121) * np.float32(40.0)
    radbcw = np.zeros((NK + 2, NJ + 2 * NG), dtype=np.float32)
    radbce = np.zeros((NK + 2, NJ + 2 * NG), dtype=np.float32)
    cpu_radbcew_inplace(
        ua, radbcw, radbce, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2, cstar=30.0
    )
    ks = slice(1, NK + 1)
    rows = slice(NG, NG + NJ)
    assert np.all(radbcw[ks, rows] <= np.float32(0.0))
    assert np.all(radbce[ks, rows] >= np.float32(0.0))
    expected_w = np.minimum(ua[ks, rows, NG] - np.float32(30.0), np.float32(0.0))
    np.testing.assert_array_equal(radbcw[ks, rows], expected_w)
    expected_e = np.maximum(ua[ks, rows, NI + NG] + np.float32(30.0), np.float32(0.0))
    np.testing.assert_array_equal(radbce[ks, rows], expected_e)


def test_radbcns_writes_nothing_on_a_side_that_is_not_open() -> None:
    vshape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    va = _noise(vshape, 131) * np.float32(40.0)
    radbcs = np.full((NK + 2, NI + 2 * NG), np.float32(7.0))
    radbcn = np.full((NK + 2, NI + 2 * NG), np.float32(7.0))
    cpu_radbcns_inplace(
        va, radbcs, radbcn, ni=NI, nj=NJ, nk=NK, ng=NG, sbc=1, nbc=1, cstar=30.0
    )
    assert np.all(radbcs == np.float32(7.0))
    assert np.all(radbcn == np.float32(7.0))


def test_restrict_outward_flux_makes_outflow_no_larger_than_inflow() -> None:
    # src/bc.F:2180-2278.  Build a boundary that leaks: outward on both faces.
    rho0 = np.full((NK, NJ, NI), np.float32(1.0), dtype=np.float32)
    rmh = np.full((NK, NJ, NI), np.float32(1.0), dtype=np.float32)
    rvh = np.full((NJ + 2 * NG,), np.float32(1.0), dtype=np.float32)
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    u3d = np.zeros(ushape, dtype=np.float32)
    ks = slice(1, NK + 1)
    rows = slice(NG, NG + NJ)
    u3d[ks, rows, NG] = np.float32(-3.0)  # outward at the west face
    u3d[ks, rows, NI + NG] = np.float32(5.0)  # outward at the east face
    before_out = float(
        -np.sum(np.minimum(u3d[ks, rows, NG], 0.0)) + np.sum(np.maximum(u3d[ks, rows, NI + NG], 0.0))
    )
    t3 = cpu_restrict_openbc_we_inplace(
        rho0, u3d, rvh, rmh, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2
    )
    after_out = float(
        -np.sum(np.minimum(u3d[ks, rows, NG], 0.0)) + np.sum(np.maximum(u3d[ks, rows, NI + NG], 0.0))
    )
    assert before_out > 0.0
    assert t3 == pytest.approx(0.0, abs=1e-12)
    assert after_out < before_out
    assert after_out == pytest.approx(0.0, abs=1e-9)


def test_restrict_outward_flux_is_the_identity_on_a_balanced_boundary() -> None:
    rho0 = np.full((NK, NJ, NI), np.float32(1.25), dtype=np.float32)
    rmh = np.full((NK, NJ, NI), np.float32(1.0), dtype=np.float32)
    ruh = np.full((NI + 2 * NG,), np.float32(1.0), dtype=np.float32)
    vshape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    v3d = np.zeros(vshape, dtype=np.float32)
    ks = slice(1, NK + 1)
    cols = slice(NG, NG + NI)
    v3d[ks, NG, cols] = np.float32(4.0)  # inflow at south
    v3d[ks, NJ + NG, cols] = np.float32(4.0)  # outflow at north
    before = v3d.copy()
    t3 = cpu_restrict_openbc_sn_inplace(
        rho0, v3d, ruh, rmh, ni=NI, nj=NJ, nk=NK, ng=NG, sbc=2, nbc=2
    )
    assert t3 == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_array_equal(v3d, before)


def test_nudgeobc_acts_only_where_the_flow_is_inflow() -> None:
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    u3d = np.zeros(ushape, dtype=np.float32)
    uten = np.zeros(ushape, dtype=np.float32)
    u0 = np.full((NK,), np.float32(10.0))
    ks = slice(1, NK + 1)
    rows = slice(NG, NG + NJ)
    u3d[ks, rows, NG] = np.float32(14.0)  # west inflow: u>0
    u3d[ks, rows, NI + NG] = np.float32(14.0)  # east outflow: u>0, untouched
    cpu_nudgeobc_we_inplace(
        u3d, u0, uten, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2, alphobc=60.0
    )
    expected = -np.float32(4.0) * np.divide(np.float32(1.0), np.float32(60.0), dtype=np.float32)
    np.testing.assert_allclose(uten[ks, rows, NG], expected, rtol=0, atol=0)
    assert np.all(uten[ks, rows, NI + NG] == np.float32(0.0))


def test_selection_rewrites_an_unset_cstar_exactly_as_param_f_722() -> None:
    assert LateralBoundarySelection(cstar=0.0).cstar == 30.0
    assert LateralBoundarySelection(cstar=0.05).cstar == 30.0
    assert LateralBoundarySelection(cstar=45.0).cstar == 45.0


def test_selection_publishes_cm1_boundary_index_flags() -> None:
    periodic = LateralBoundarySelection(west=1, east=1, south=1, north=1)
    assert (periodic.ibw, periodic.ibe, periodic.ibs, periodic.ibn) == (0, 0, 0, 0)
    channel = LateralBoundarySelection(west=2, east=2, south=1, north=1)
    assert (channel.ibw, channel.ibe, channel.ibs, channel.ibn) == (1, 1, 0, 0)
    assert not channel.all_open and channel.any_open


def test_selection_refuses_a_value_cm1_does_not_have() -> None:
    with pytest.raises(ValueError):
        LateralBoundarySelection(west=5)
    with pytest.raises(ValueError):
        LateralBoundarySelection(irbc=3)
    with pytest.raises(ValueError):
        LateralBoundarySelection(nudgeobc=1, alphobc=0.0)


def test_a_domain_no_wider_than_the_halo_is_refused_rather_than_wrapped_from_ghosts() -> None:
    with pytest.raises(ValueError):
        family_array_shape(KIND_SCALAR, ni=3, nj=8, nk=4, ng=3)
