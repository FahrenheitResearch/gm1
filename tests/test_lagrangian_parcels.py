"""CM1 r21.1 ``src/parcel.F`` parcel integration, checked against exact answers.

There is no CM1 differential in this file: it runs no model.  What it does is
pin the transcription (attribute inventory, boundary fill, periodic exit) and
prove the integrator against two closed-form answers -- uniform translation
and solid-body rotation, for which the RK2 map is exactly
``(1 - theta^2/2 + i*theta)`` per step.  The CM1 differential lives in
``tools/oracle`` and is recorded in the branch report.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from cm1core.cuda_psolver3_terrain_open import CM1_OPEN_NGXY, terrain_open_sound_shapes
from cm1core.lagrangian_parcels import (
    CM1_CGS1_F32,
    CM1_CGS2_F32,
    CM1_CGS3_F32,
    PARCEL_BASE_ATTRIBUTES,
    PARCEL_SWITCH_ORDER,
    PARCEL_UNFILLED_ATTRIBUTES,
    ParcelGrid,
    ParcelSeeding,
    ParcelVelocityField,
    cpu_parcel_advance,
    cm1_parcel_switch_context,
    fill_parcel_vertical_boundaries,
    parcel_attribute_layout,
    parcel_circuit_positions,
    seed_parcel_positions,
)
from cm1core.moisture_species import WSM6_PROFILE

NX, NY, NZ = 24, 24, 20
DX = DY = 1000.0
DZ = 500.0


def wsm6_switch_context(*, bbc: int = 1):
    """GM1's shipped LES deck (WSM6, cm1setup=1, sgsmodel=1, ipbl=0) on ``bbc``."""

    return cm1_parcel_switch_context(imoist=1, profile=WSM6_PROFILE, cm1setup=1, sgsmodel=1,
                                     ipbl=0, sfcmodel=1 if bbc == 3 else 0, oceanmodel=1,
                                     bbc=bbc, iptra=0, terrain=False)


def widest_switch_context():
    """The one CM1 deck that keeps every column this port can lay out.

    ``cm1setup=4`` with ``sgsmodel=1`` and ``ipbl=1`` is the only branch where
    ``param.F:6273-6276`` keeps ``prcl_km``/``prcl_kh``, and ``bbc=3`` keeps
    ``prcl_sfc`` (the CM1 oracle's ``les-inside-mesoscale`` case).
    """

    return cm1_parcel_switch_context(imoist=1, profile=WSM6_PROFILE, cm1setup=4, sgsmodel=1,
                                     ipbl=1, sfcmodel=1, oceanmodel=1, bbc=3, iptra=0,
                                     terrain=False)


def _grid(iorigin: int = 2) -> ParcelGrid:
    return ParcelGrid(nx=NX, ny=NY, nz=NZ, dx_m=DX, dy_m=DY, dz_m=DZ, iorigin=iorigin)


def _empty_field() -> ParcelVelocityField:
    shapes = terrain_open_sound_shapes(NX, NY, NZ)
    return ParcelVelocityField(
        np.zeros(shapes["u3d"], dtype=np.float32),
        np.zeros(shapes["v3d"], dtype=np.float32),
        np.zeros(shapes["w3d"], dtype=np.float32),
    )


def _fill_linear(field: ParcelVelocityField, grid: ParcelGrid, fn) -> None:
    """Fill the three carriers from ``fn(x, y, z) -> (u, v, w)`` at their points."""

    ng = CM1_OPEN_NGXY
    for name, array in (("u", field.u), ("v", field.v), ("w", field.w)):
        for k in range(array.shape[0]):
            for j in range(array.shape[1]):
                for i in range(array.shape[2]):
                    ci, cj, ck = i - ng + 1, j - ng + 1, k
                    if name == "u":
                        x = grid.minx_m + (ci - 1) * grid.dx_m
                        y = grid.miny_m + (cj - 0.5) * grid.dy_m
                        z = (ck - 0.5) * grid.dz_m
                        array[k, j, i] = np.float32(fn(x, y, z)[0])
                    elif name == "v":
                        x = grid.minx_m + (ci - 0.5) * grid.dx_m
                        y = grid.miny_m + (cj - 1) * grid.dy_m
                        z = (ck - 0.5) * grid.dz_m
                        array[k, j, i] = np.float32(fn(x, y, z)[1])
                    else:
                        x = grid.minx_m + (ci - 0.5) * grid.dx_m
                        y = grid.miny_m + (cj - 0.5) * grid.dy_m
                        z = (ck - 1) * grid.dz_m
                        array[k, j, i] = np.float32(fn(x, y, z)[2])


def test_attribute_inventory_is_param_F_6385_6500_switch_order():
    layout = parcel_attribute_layout({name: 1 for name in PARCEL_SWITCH_ORDER},
                                     widest_switch_context())
    assert layout.names[:7] == PARCEL_BASE_ATTRIBUTES
    # param.F:6391 -- seven unconditional, then each switch's block in order.
    assert layout.names[7:10] == ("th", "t", "prs")
    assert layout.names[10:16] == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert layout.names[16:20] == ("kmh", "kmv", "khh", "khv")
    assert layout.index("zvort") > layout.index("vpg")
    assert layout.npvals == len(set(layout.names))
    off = parcel_attribute_layout({name: 0 for name in PARCEL_SWITCH_ORDER},
                                  widest_switch_context())
    assert off.names == PARCEL_BASE_ATTRIBUTES and off.npvals == 7


def test_unfilled_attributes_are_named_rather_than_published_as_zero():
    layout = parcel_attribute_layout({name: 1 for name in PARCEL_SWITCH_ORDER},
                                     widest_switch_context())
    assert set(layout.unfilled) == set(PARCEL_UNFILLED_ATTRIBUTES)
    record = layout.record()
    for name in layout.unfilled:
        assert name in record["unfilled_attributes"]
        assert len(record["unfilled_attributes"][name]) > 30


def test_a_switch_that_is_off_removes_exactly_its_own_attributes():
    switches = {name: 1 for name in PARCEL_SWITCH_ORDER}
    full = parcel_attribute_layout(switches, widest_switch_context())
    switches["prcl_q"] = 0
    fewer = parcel_attribute_layout(switches, widest_switch_context())
    assert set(full.names) - set(fewer.names) == {"qv", "qc", "qr", "qi", "qs", "qg"}
    assert fewer.npvals == full.npvals - 6


def test_free_slip_ghost_fill_is_param_F_9038_quadratic():
    field = _empty_field()
    rng = np.random.default_rng(3)
    field.u[...] = rng.normal(size=field.u.shape).astype(np.float32)
    field.v[...] = rng.normal(size=field.v.shape).astype(np.float32)
    expected = (
        np.float32(CM1_CGS1_F32) * field.u[1]
        + np.float32(CM1_CGS2_F32) * field.u[2]
        + np.float32(CM1_CGS3_F32) * field.u[3]
    )
    fill_parcel_vertical_boundaries(field, nx=NX, ny=NY, nz=NZ, bbc=1, tbc=1)
    assert np.array_equal(field.u[0], expected)
    assert np.array_equal(field.w[NZ + 2], np.zeros_like(field.w[NZ + 2]))


def test_no_slip_ghost_fill_carries_the_translating_frame():
    field = _empty_field()
    fill_parcel_vertical_boundaries(
        field, nx=NX, ny=NY, nz=NZ, bbc=2, tbc=1,
        u_move_m_s=12.5, v_move_m_s=3.0,
    )
    # init3d.F:32-33 -- the stored wind is frame-relative, so ground-fixed zero
    # is -umove here.
    assert float(field.u[0].max()) == pytest.approx(-12.5)
    assert float(field.v[0].max()) == pytest.approx(-3.0)


def test_an_unsupported_boundary_is_refused_by_value():
    field = _empty_field()
    with pytest.raises(ValueError, match="parcel.F lower-boundary"):
        fill_parcel_vertical_boundaries(field, nx=NX, ny=NY, nz=NZ, bbc=9, tbc=1)


def test_uniform_flow_translates_a_parcel_exactly():
    grid = _grid()
    field = _empty_field()
    field.u[...] = np.float32(10.0)
    field.v[...] = np.float32(-4.0)
    field.w[...] = np.float32(0.25)
    positions = np.array([[0.0], [0.0], [2000.0]], dtype=np.float32)
    dt = 2.0
    steps = 30
    out = positions
    for _ in range(steps):
        out = cpu_parcel_advance(out, field, grid, dt_s=dt)
    assert float(out[0, 0]) == pytest.approx(10.0 * dt * steps, abs=1e-3)
    assert float(out[1, 0]) == pytest.approx(-4.0 * dt * steps, abs=1e-3)
    assert float(out[2, 0]) == pytest.approx(2000.0 + 0.25 * dt * steps, abs=1e-3)


def test_solid_body_rotation_matches_the_exact_rk2_map():
    """RK2 on a linear field is exactly ``(1 - theta^2/2 + i*theta)`` per step.

    Trilinear interpolation reproduces a linear field exactly, so the discrete
    parcel path is the RK2 map on the continuum field and both its amplitude
    and its phase are known in closed form.  This is the decisive analytic
    check on the integrator, the sampler and the weights together.
    """

    grid = _grid()
    field = _empty_field()
    omega = 0.004
    _fill_linear(field, grid, lambda x, y, z: (-omega * y, omega * x, 0.0))
    dt, steps = 5.0, 40
    theta = omega * dt
    amplification = complex(1.0 - 0.5 * theta * theta, theta)
    start = complex(3000.0, 0.0)
    expected = start * amplification ** steps
    out = np.array([[start.real], [start.imag], [2000.0]], dtype=np.float32)
    for _ in range(steps):
        out = cpu_parcel_advance(out, field, grid, dt_s=dt)
    seen = complex(float(out[0, 0]), float(out[1, 0]))
    assert abs(seen - expected) / abs(expected) < 2.0e-5
    # And the two closed-form statements separately, so a failure says which.
    assert abs(seen) / abs(start) == pytest.approx(
        abs(amplification) ** steps, rel=2.0e-5
    )
    assert cmath.phase(seen) == pytest.approx(
        steps * cmath.phase(amplification), abs=2.0e-5
    )


def test_backward_integration_returns_a_parcel_to_where_it_started():
    """GM1 EXTENSION: ``direction=-1`` is the same RK2 with ``-dt``.

    Run forward then backward through the *same* frozen field: the round trip
    is not the identity -- RK2 is not time-symmetric -- but it closes to the
    scheme's own third-order local error, which for these numbers is metres
    out of kilometres and is what a backward trajectory is worth.
    """

    grid = _grid()
    field = _empty_field()
    omega = 0.004
    _fill_linear(field, grid, lambda x, y, z: (-omega * y, omega * x, 0.0))
    start = np.array([[3000.0], [0.0], [2000.0]], dtype=np.float32)
    out = start
    for _ in range(20):
        out = cpu_parcel_advance(out, field, grid, dt_s=5.0)
    for _ in range(20):
        out = cpu_parcel_advance(out, field, grid, dt_s=5.0, direction=-1)
    drift = math.hypot(float(out[0, 0]) - 3000.0, float(out[1, 0]))
    assert drift < 5.0


def test_a_parcel_that_exits_is_re_entered_periodically():
    grid = _grid()
    field = _empty_field()
    field.u[...] = np.float32(600.0)
    out = np.array([[grid.maxx_m - 100.0], [0.0], [2000.0]], dtype=np.float32)
    out = cpu_parcel_advance(out, field, grid, dt_s=1.0)
    # parcel.F:577-592 wraps no matter what the lateral conditions are.
    assert float(out[0, 0]) < grid.minx_m + grid.dx_m * NX
    assert float(out[0, 0]) == pytest.approx(grid.minx_m + 500.0, abs=1.0)


def test_a_parcel_driven_below_ground_is_clamped_to_parcel_F_1e_6():
    grid = _grid()
    field = _empty_field()
    field.w[...] = np.float32(-50.0)
    out = np.array([[0.0], [0.0], [100.0]], dtype=np.float32)
    out = cpu_parcel_advance(out, field, grid, dt_s=10.0)
    assert float(out[2, 0]) == pytest.approx(1.0e-6, abs=1e-9)


def test_unplaced_parcels_keep_init3d_F_363_sentinel_and_are_not_integrated():
    grid = _grid()
    field = _empty_field()
    field.u[...] = np.float32(10.0)
    out = np.full((3, 4), np.float32(-1.0e30), dtype=np.float32)
    out[:, 0] = (0.0, 0.0, 2000.0)
    out = cpu_parcel_advance(out, field, grid, dt_s=1.0)
    assert float(out[0, 0]) == pytest.approx(10.0)
    assert float(out[0, 1]) == pytest.approx(-1.0e30)


def test_grid_seeding_reproduces_init3d_F_381_387_exactly():
    grid = _grid(iorigin=2)
    seeding = ParcelSeeding(mode="grid", nparcels=60 * 60 * 10)
    assert seeding.is_cm1_default_grid
    seeded = seed_parcel_positions(seeding, grid)
    assert seeded.shape == (3, 36000)
    # init3d.F:385-387 -- minx + 2000*(i-1), miny + 2000*(j-1), zh(1)+1000*(k-1)
    assert float(seeded[0, 0]) == pytest.approx(grid.minx_m)
    assert float(seeded[1, 0]) == pytest.approx(grid.miny_m)
    assert float(seeded[2, 0]) == pytest.approx(grid.lowest_mass_level_m)
    assert float(seeded[0, 1]) == pytest.approx(grid.minx_m + 2000.0)
    assert float(seeded[1, 60]) == pytest.approx(grid.miny_m + 2000.0)
    assert float(seeded[2, 3600]) == pytest.approx(grid.lowest_mass_level_m + 1000.0)


def test_seeding_that_overruns_nparcels_is_refused_like_init3d_F_375():
    with pytest.raises(ValueError, match="init3d.F:375-383"):
        seed_parcel_positions(ParcelSeeding(mode="grid", nparcels=10), _grid())


def test_vorticity_threshold_seeding_places_only_the_cells_over_the_threshold():
    grid = _grid()
    zeta = np.zeros((NZ, NY, NX), dtype=np.float32)
    zeta[3, 4, 5] = 0.05
    zeta[3, 4, 6] = -0.05
    zeta[3, 4, 7] = 0.001
    seeded = seed_parcel_positions(
        ParcelSeeding(mode="vorticity-threshold", nparcels=8,
                      vorticity_threshold_s_inv=0.01),
        grid, vertical_vorticity=zeta,
    )
    placed = seeded[:, seeded[0] > -1.0e29]
    assert placed.shape[1] == 2
    assert float(placed[0, 0]) == pytest.approx(grid.minx_m + 5.5 * DX)
    assert float(placed[2, 0]) == pytest.approx(3.5 * DZ)


def test_threshold_seeding_without_a_vorticity_field_is_refused_by_name():
    with pytest.raises(ValueError, match="model-time decision"):
        seed_parcel_positions(
            ParcelSeeding(mode="vorticity-threshold", nparcels=4), _grid()
        )


def test_iorigin_moves_every_published_position_by_half_a_domain():
    centred, corner = _grid(2), _grid(1)
    assert centred.minx_m == pytest.approx(-0.5 * NX * DX)
    assert corner.minx_m == 0.0
    assert centred.maxx_m - centred.minx_m == corner.maxx_m - corner.minx_m


def test_an_unadmitted_iorigin_is_refused_by_value():
    with pytest.raises(ValueError, match="param.F:6666"):
        ParcelGrid(nx=NX, ny=NY, nz=NZ, dx_m=DX, dy_m=DY, dz_m=DZ, iorigin=3)


def test_circuit_seeding_is_a_closed_ring_in_seeding_order():
    ring = parcel_circuit_positions(
        center_x_m=100.0, center_y_m=-50.0, center_z_m=750.0,
        radius_m=400.0, count=8,
    )
    assert ring.shape == (3, 8)
    radii = np.hypot(ring[0] - 100.0, ring[1] + 50.0)
    assert float(radii.min()) == pytest.approx(400.0, rel=1e-5)
    assert float(radii.max()) == pytest.approx(400.0, rel=1e-5)
    assert float(ring[0, 0]) == pytest.approx(500.0, rel=1e-5)
    assert np.allclose(ring[2], 750.0)


# ---------------------------------------------------------------------------
# The parent parcels are parcel.F on the parent's own deck.
# ---------------------------------------------------------------------------
from cm1core import nest_parcels as npc  # noqa: E402
from cm1core.cm1_vertical_grid import build_vertical_levels  # noqa: E402
from cm1core.lagrangian_config import parse_lagrangian_table  # noqa: E402
from cm1core.lagrangian_stage import LagrangianStage  # noqa: E402


def test_the_top_face_w_zero_is_cm1_level_nk_plus_1():
    """parcel.F:124-128 zeroes wa(i,j,nk+1); storage index nz+1, not the ghost above it."""

    field = _empty_field()
    rng = np.random.default_rng(4)
    field.w[...] = rng.normal(size=field.w.shape).astype(np.float32)
    above = field.w[NZ + 2].copy()
    fill_parcel_vertical_boundaries(field, nx=NX, ny=NY, nz=NZ, bbc=1, tbc=1)
    assert not field.w[NZ + 1].any()
    assert np.array_equal(field.w[NZ + 2], above)
    deck = _grid().deck()
    w = rng.normal(size=field.w.shape).astype(np.float32)
    npc.fill_velocity_ghosts(np, field.u, field.v, w, deck, bbc=3, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)
    assert not w[NZ + 1].any() and w[NZ + 2].any()


def _linear_field(grid, fn):
    """``fn(x, y, z)`` at every padded C-grid word, from the grid's own deck."""

    deck = grid.deck()
    ng = CM1_OPEN_NGXY
    shapes = terrain_open_sound_shapes(grid.nx, grid.ny, grid.nz)
    out = []
    for name, comp in (("u3d", 0), ("v3d", 1), ("w3d", 2)):
        k_n, j_n, i_n = shapes[name]
        zc = (deck.zf if name == "w3d" else deck.zh)[:k_n].astype(np.float64)
        cj = np.arange(j_n) + 1 - ng
        ci = np.arange(i_n) + 1 - ng
        yc = deck.miny_m + ((cj - 1) if name == "v3d" else (cj - 0.5)) * deck.dy_m
        xc = deck.minx_m + ((ci - 1) if name == "u3d" else (ci - 0.5)) * deck.dx_m
        z3, y3, x3 = np.meshgrid(zc, yc, xc, indexing="ij")
        out.append(np.asarray(fn(x3, y3, z3)[comp] * np.ones_like(x3), dtype=np.float32))
    return ParcelVelocityField(*out)


def test_the_first_rk_stage_is_neither_wrapped_nor_clamped():
    """parcel.F:526-537 moves the midpoint without the exit or the clamp (:553-567)."""

    grid = _grid()
    a, c0, c1 = 0.01, -2.0, -0.5
    field = _linear_field(grid, lambda x, y, z: (a * x, 0.0 * y, c0 + c1 * z))
    x0, z0, dt = grid.maxx_m - 50.0, 3.0, 1.0
    out = cpu_parcel_advance(np.array([[x0], [0.0], [z0]], dtype=np.float32), field, grid, dt_s=dt)
    u1 = a * x0
    x1 = x0 + dt * u1                      # beyond maxx: sampled there, not wrapped back
    x = x0 + 0.5 * dt * (u1 + a * x1)
    width = grid.maxx_m - grid.minx_m
    assert float(out[0, 0]) == pytest.approx(x - width if x > grid.maxx_m else x, abs=2e-3)
    w1 = c0 + c1 * z0
    z1 = z0 + dt * w1                      # below ground: sampled there, not clamped to 1e-6
    assert z1 < 0.0
    assert float(out[2, 0]) == pytest.approx(z0 + 0.5 * dt * (w1 + c0 + c1 * z1), abs=2e-4)


def _stretched_grid():
    # 5 layers of 100 m, 10 stretched layers averaging 500 m, 5 layers of 900 m.
    levels = build_vertical_levels(stretch_z=1, nk=NZ, dz_m=500.0, ztop_m=10000.0, str_bot_m=500.0,
                                   str_top_m=5500.0, dz_bot_m=100.0, dz_top_m=900.0)
    return ParcelGrid(nx=NX, ny=NY, nz=NZ, dx_m=DX, dy_m=DY, dz_m=500.0, iorigin=2, levels=levels)


def test_a_stretched_deck_is_searched_on_its_own_faces():
    """parcel.F:251-260 scans zf; a uniform closed form picks the wrong layer on stretch_z=1."""

    grid = _stretched_grid()
    assert grid.maxz_m == float(grid.levels.zf[NZ + 1])
    assert grid.lowest_mass_level_m == float(grid.levels.zh[1])
    c = 0.01
    field = _linear_field(grid, lambda x, y, z: (0.0 * x, 0.0 * y, c * z))
    heights = [60.0, 260.0, 830.0, 3100.0, 7400.0]
    start = np.array([[0.0] * 5, [0.0] * 5, heights], dtype=np.float32)
    out = cpu_parcel_advance(start, field, grid, dt_s=2.0)
    for p, z0 in enumerate(heights):
        z1 = z0 + 2.0 * c * z0
        assert float(out[2, p]) == pytest.approx(z0 + 1.0 * (c * z0 + c * z1), rel=2e-6)
    seeds = seed_parcel_positions(ParcelSeeding(mode="grid", nparcels=36000), grid)
    assert float(seeds[2, 0]) == float(grid.levels.zh[1])


@pytest.mark.parametrize("stretched", [False, True])
def test_a_semi_slip_stage_steps_with_zntmp_and_refuses_without_it(stretched):
    grid = _stretched_grid() if stretched else _grid()
    table = parse_lagrangian_table({"iprcl": 1, "nparcels": 64, "seeding": "box", "seed_nx": 8,
                                    "seed_ny": 4, "seed_nz": 2, "seed_dx_m": 900.0, "seed_dy_m": 900.0,
                                    "seed_dz_m": 20.0, "seed_x0_m": -3000.0, "seed_y0_m": -1500.0,
                                    "seed_z0_m": 5.0})
    stage = LagrangianStage.build(table, nx=NX, ny=NY, nz=NZ, dx_m=DX, dy_m=DY, dz_m=grid.dz_m, dt_s=2.0,
                                  xp=np, iorigin=2, levels=grid.levels, bbc=3, u_move_m_s=4.0,
                                  v_move_m_s=-1.0, lateral=(1, 1, 2, 2),
                                  switch_context=wsm6_switch_context(bbc=3))
    stage.seed()
    rng = np.random.default_rng(8)
    shapes = terrain_open_sound_shapes(NX, NY, NZ)
    arrays = {"rru": rng.normal(0, 8, shapes["u3d"]).astype(np.float32),
              "rrv": rng.normal(0, 8, shapes["v3d"]).astype(np.float32),
              "rrw": rng.normal(0, 0.05, shapes["w3d"]).astype(np.float32)}
    with pytest.raises(RuntimeError, match="parcel.F:470-515"):
        stage.capture_advecting_velocity(arrays)
    znt = rng.uniform(1e-3, 0.5, (NY, NX)).astype(np.float32)
    start = np.array(stage.positions, copy=True)
    velocity = stage.capture_advecting_velocity(arrays, roughness=znt)
    stage.advance(velocity)
    deck = grid.deck((1, 1, 2, 2))
    surface = npc.SemiSlip(npc.pad_roughness(np, znt, deck, ng=CM1_OPEN_NGXY), 1, 4.0, -1.0)
    n = start.shape[1]
    want, _ = npc.reference_rk2(np, start, np.ones(n, np.int32), np.zeros(n, np.int32), code=1,
                                u=velocity.u, v=velocity.v, w=velocity.w, deck=deck, dt_s=2.0,
                                ng=CM1_OPEN_NGXY, surface=surface)
    assert stage.positions.tobytes() == want.tobytes()
    plain, _ = npc.reference_rk2(np, start, np.ones(n, np.int32), np.zeros(n, np.int32), code=1,
                                 u=velocity.u, v=velocity.v, w=velocity.w, deck=deck, dt_s=2.0,
                                 ng=CM1_OPEN_NGXY)
    low = start[2] < deck.zh[1]
    assert low.any() and np.any(plain[:, low] != want[:, low])
    record = stage.record()["integrator"]
    assert record["bbc"] == 3 and "parcel.F:470-515" in record["below_lowest_level"]
