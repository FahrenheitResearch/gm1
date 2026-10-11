"""Parcels in the live nest child, held to the kernel text, closed forms and their rules.

No CUDA context is opened.  The packaged kernel string is compiled with g++
(``tools/oracle/nest_parcel_oracle.py``) and compared raw-word against the host
reference; the tracker, the frame sampler, the hand-off and the door run on
NumPy through the same code the device path takes for everything but the one
kernel launch.
"""

from __future__ import annotations

import cmath
import importlib.util
import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from cm1core import nest_parcels as npc  # noqa: E402
from cm1core.cm1_vertical_grid import build_vertical_levels  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.lagrangian_config import (  # noqa: E402
    NEST_CHILD_KEYS,
    LagrangianConfigError,
    parse_lagrangian_table,
)
from cm1core.lagrangian_parcels import parcel_attribute_layout  # noqa: E402
from cm1core.nest_geometry import NestGeometry  # noqa: E402

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
NG = 3


def child_levels():
    """Five metres at the ground under a 1500 m top: the child's own deck."""

    return build_vertical_levels(stretch_z=1, nk=27, dz_m=100.0, ztop_m=1500.0, str_bot_m=90.0,
                                 str_top_m=500.0, dz_bot_m=5.0, dz_top_m=200.0)


def parent_levels():
    return build_vertical_levels(stretch_z=1, nk=12, dz_m=100.0, ztop_m=1500.0, str_bot_m=60.0,
                                 str_top_m=500.0, dz_bot_m=20.0, dz_top_m=200.0)


def geometry(**overrides) -> NestGeometry:
    values = dict(parent_nx=18, parent_ny=16, parent_dx_m=600.0, parent_dy_m=600.0,
                  refinement_ratio=3, time_refinement_ratio=3, parent_i0=6, parent_j0=5,
                  parent_span_x=6, parent_span_y=5, spec_zone=1, relax_zone=4, feedback=1)
    values.update(overrides)
    return NestGeometry(**values)


def child_deck(g: NestGeometry) -> npc.ParcelDeck:
    return npc.deck_from_levels(name="child", nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m,
                                minx_m=g.origin_x_m, miny_m=g.origin_y_m, levels=child_levels(),
                                lateral=(2, 2, 2, 2), periodic=False)


def parent_deck() -> npc.ParcelDeck:
    return npc.deck_from_levels(name="parent", nx=18, ny=16, dx_m=600.0, dy_m=600.0, minx_m=0.0,
                                miny_m=0.0, levels=parent_levels(), lateral=(2, 2, 2, 2), periodic=True)


def padded_winds(deck: npc.ParcelDeck, fn):
    """Evaluate ``fn(x, y, z) -> (u, v, w)`` at every padded C-grid word, ghosts included."""

    nx, ny, nz = deck.nx, deck.ny, deck.nz
    zh = deck.zh.astype(np.float64)
    zf = deck.zf.astype(np.float64)

    def axis(n, origin, spacing, face):
        cm1 = np.arange(n) + 1 - NG
        return origin + ((cm1 - 1) if face else (cm1 - 0.5)) * spacing

    def build(kz, jn, jface, in_, iface, component):
        z = kz[:, None, None]
        y = axis(jn, deck.miny_m, deck.dy_m, jface)[None, :, None]
        x = axis(in_, deck.minx_m, deck.dx_m, iface)[None, None, :]
        x, y, z = np.broadcast_arrays(x, y, z)
        return np.asarray(fn(x, y, z)[component], dtype=np.float64).astype(np.float32) * np.ones(x.shape, np.float32)

    u = build(zh, ny + 2 * NG, False, nx + 1 + 2 * NG, True, 0)
    v = build(zh, ny + 1 + 2 * NG, True, nx + 2 * NG, False, 1)
    w = build(zf, ny + 2 * NG, False, nx + 2 * NG, False, 2)
    return u, v, w


# ---------------------------------------------------------------------------
# The kernel text is the host reference.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def harness():
    from tools.oracle.nest_parcel_oracle import open_harness

    return open_harness()


def _hostile_positions(rng, deck: npc.ParcelDeck, n: int):
    lo_x, hi_x = deck.minx_m - 2 * deck.dx_m, deck.minx_m + (deck.nx + 2) * deck.dx_m
    lo_y, hi_y = deck.miny_m - 2 * deck.dy_m, deck.miny_m + (deck.ny + 2) * deck.dy_m
    pos = np.stack([rng.uniform(lo_x, hi_x, n), rng.uniform(lo_y, hi_y, n),
                    rng.uniform(-20.0, deck.maxz_m + 40.0, n)]).astype(np.float32)
    # Positions exactly on faces, half levels and deck faces: where a search
    # that is not parcel.F's scan would pick the neighbouring cell.
    q = n // 4
    pos[0, :q] = [deck.face_x(int(i)) for i in rng.integers(1, deck.nx + 2, q)]
    pos[1, q:2 * q] = [deck.face_y(int(j)) for j in rng.integers(1, deck.ny + 2, q)]
    pos[2, 2 * q:3 * q] = deck.zf[rng.integers(0, deck.nz + 3, q)]
    pos[2, 3 * q:3 * q + q // 2] = deck.zh[rng.integers(0, deck.nz + 2, q // 2)]
    pos[:, -1] = np.float32(-1.0e30)
    pos[0, -2] = np.float32("nan")
    return pos


@requires_gxx
@pytest.mark.parametrize("order", ["linear", "shuffled"])
@pytest.mark.parametrize("dt_s", [0.7, -1.3])
@pytest.mark.parametrize("which", ["child", "parent"])
def test_the_kernel_text_is_the_host_reference_raw_word(harness, order, dt_s, which):
    rng = np.random.default_rng(20260915)
    deck = child_deck(geometry()) if which == "child" else parent_deck()
    shapes = ((deck.nz + 2, deck.ny + 2 * NG, deck.nx + 1 + 2 * NG),
              (deck.nz + 2, deck.ny + 1 + 2 * NG, deck.nx + 2 * NG),
              (deck.nz + 3, deck.ny + 2 * NG, deck.nx + 2 * NG))
    scale = (40.0 if which == "child" else 400.0)
    u, v, w = (rng.normal(0.0, scale, shape).astype(np.float32) for shape in shapes)
    n = 1200
    positions = _hostile_positions(rng, deck, n)
    owner = rng.integers(0, 3, n).astype(np.int32)
    status = np.where(rng.random(n) < 0.05, 2, 0).astype(np.int32)
    ref_pos, ref_status = npc.reference_rk2(np, positions, owner, status, code=1, u=u, v=v, w=w,
                                            deck=deck, dt_s=dt_s, ng=NG)
    ker_pos, ker_status = harness.rk2(positions=positions, owner=owner, status=status, u=u, v=v,
                                      w=w, deck=deck, code=1, dt_s=dt_s, ng=NG, order=order)
    assert ref_pos.tobytes() == ker_pos.tobytes()
    assert np.array_equal(ref_status, ker_status)
    # The case is not vacuous: parcels moved, some left the lattice, and the
    # pre-set statuses and non-finite starts were left alone as the kernel does.
    moved = (owner == 1) & (status == 0) & (ref_status == 0) & (positions[0] > -1.0e29)
    assert int(moved.sum()) > 100
    assert {0, 1, 2} <= set(np.unique(ref_status[owner == 1]).tolist())
    assert ref_pos[:, owner != 1].tobytes() == positions[:, owner != 1].tobytes()


def test_the_search_returns_parcel_F_s_linear_scan_index():
    rng = np.random.default_rng(7)
    deck = child_deck(geometry())
    coords = np.concatenate([
        rng.uniform(deck.minx_m - 500.0, deck.minx_m + (deck.nx + 3) * deck.dx_m, 4000),
        [deck.face_x(i) for i in range(-1, deck.nx + 4)],
    ]).astype(np.float32)
    for lo, hi in ((1, deck.nx), (0, deck.nx + 1)):
        got = npc.reference_search(np, coords, deck.minx_m, deck.dx_m, deck.nx, lo, hi)
        for c, g in zip(coords, got):
            want = -100
            i = hi + 1
            while want < 0 and i > lo:  # parcel.F:161-167, :209-215
                i -= 1
                if c >= np.float32(deck.face_x(i)) and c <= np.float32(deck.face_x(i + 1)):
                    want = i
            assert int(g) == want


# ---------------------------------------------------------------------------
# Closed forms on a child with its own deck.
# ---------------------------------------------------------------------------
def test_solid_body_rotation_on_a_child_with_its_own_deck_is_the_exact_rk2_map():
    """RK2 on a linear field is ``(1 - theta^2/2 + i*theta)`` per step.

    Trilinear interpolation reproduces a linear wind, so the discrete path on
    the child's lattice and stretched deck is the RK2 map itself.  Parcels sit
    in the 5 m layers and aloft, with the child's origin 3.6 km from the
    parent's: a sampler that read child indices as parent metres, or the
    parent's deck, would miss by kilometres.
    """

    g = geometry()
    deck = child_deck(g)
    xc, yc, omega = 5400.0, 4500.0, 0.01
    u, v, w = padded_winds(deck, lambda x, y, z: (-omega * (y - yc), omega * (x - xc), 0.0 * z))
    heights = (2.5, 7.0, 43.0, 810.0)
    start = [complex(300.0, 0.0), complex(0.0, -250.0), complex(-180.0, 90.0), complex(120.0, 200.0)]
    positions = np.array([[xc + s.real for s in start], [yc + s.imag for s in start], heights],
                         dtype=np.float32)
    owner = np.ones(4, np.int32)
    status = np.zeros(4, np.int32)
    dt, steps = 1.0, 60
    for _ in range(steps):
        positions, status = npc.reference_rk2(np, positions, owner, status, code=1, u=u, v=v, w=w,
                                              deck=deck, dt_s=dt, ng=NG)
    assert not status.any()
    theta = omega * dt
    amplification = complex(1.0 - 0.5 * theta * theta, theta) ** steps
    for p, s in enumerate(start):
        seen = complex(float(positions[0, p]) - xc, float(positions[1, p]) - yc)
        assert abs(seen - s * amplification) / abs(s) < 5.0e-5
        assert cmath.phase(seen / s) == pytest.approx(cmath.phase(amplification), abs=5.0e-5)
    assert positions[2].tolist() == list(np.asarray(heights, np.float32))


def test_uniform_flow_on_a_child_with_its_own_deck_crosses_stretched_layers_exactly():
    g = geometry()
    deck = child_deck(g)
    u, v, w = padded_winds(deck, lambda x, y, z: (7.0 + 0 * x, -3.0 + 0 * y, 0.5 + 0 * z))
    positions = np.array([[4900.0, 5500.0], [4700.0, 4300.0], [1.0, 60.0]], dtype=np.float32)
    owner = np.ones(2, np.int32)
    status = np.zeros(2, np.int32)
    for _ in range(80):
        positions, status = npc.reference_rk2(np, positions, owner, status, code=1, u=u, v=v, w=w,
                                              deck=deck, dt_s=1.0, ng=NG)
    assert not status.any()
    assert positions[0].tolist() == pytest.approx([4900.0 + 560.0, 5500.0 + 560.0], abs=2e-3)
    assert positions[1].tolist() == pytest.approx([4700.0 - 240.0, 4300.0 - 240.0], abs=2e-3)
    # From inside the 5 m layers up through the stretching zone.
    assert positions[2].tolist() == pytest.approx([41.0, 100.0], abs=2e-4)
    down = padded_winds(deck, lambda x, y, z: (0 * x, 0 * y, -30.0 + 0 * z))
    pos = np.array([[5000.0], [4500.0], [40.0]], dtype=np.float32)
    pos, st = npc.reference_rk2(np, pos, np.ones(1, np.int32), np.zeros(1, np.int32), code=1,
                                u=down[0], v=down[1], w=down[2], deck=deck, dt_s=2.0, ng=NG)
    assert float(pos[2, 0]) == pytest.approx(1.0e-6, abs=1e-9)  # parcel.F:562-565


# ---------------------------------------------------------------------------
# The tracker: hand-off across a move, termination, frames, hooks.
# ---------------------------------------------------------------------------
BOUNDARY = SimpleNamespace(bbc=1, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)


def _settings(**overrides):
    table = {"seed_frame": "parent-grid-origin", "nparcels": 3, "prclfrq_s": 3.0,
             "seed_nx": 1, "seed_ny": 1, "seed_nz": 1, "seed_dx_m": 10.0, "seed_dy_m": 10.0,
             "seed_dz_m": 10.0, "seed_x0_m": 5000.0, "seed_y0_m": 4500.0, "seed_z0_m": 500.0}
    table.update(overrides)
    return parse_lagrangian_table({"nest_child": table}).nest_child


def _runtime(deck: npc.ParcelDeck, winds, *, zvort_fn=None):
    nz, ny, nx = deck.nz, deck.ny, deck.nx
    mass = (nz, ny, nx)
    face = (nz + 1, ny, nx)
    full = lambda value, shape: np.full(shape, np.float32(value), dtype=np.float32)  # noqa: E731
    u, v, w = winds
    return SimpleNamespace(
        config=SimpleNamespace(bbc=1, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0),
        dynamics=SimpleNamespace(
            arrays={"u3d": u, "v3d": v, "w3d": w, "rru": u, "rrv": v, "rrw": w},
            fields={"th3d": full(1.0, mass), "th0": full(300.0, mass), "pi0": full(0.9, mass),
                    "pp3d": full(0.0, mass), "qv0": full(0.0, mass), "prs": full(9.0e4, mass),
                    "rho": full(1.1, mass), "q3d": np.zeros((6, *mass), np.float32)},
            tke={"tke3d": full(0.25, face)},
            metrics={"c1": full(0.5, face), "c2": full(0.5, face)},
        ),
    )


class _Clock:
    def __init__(self):
        self.completed_steps = 0
        self.model_seconds = 0.0


def _tracker(settings, g):
    from test_lagrangian_parcels import wsm6_switch_context

    layout = parcel_attribute_layout({"prcl_th": 1, "prcl_b": 1, "prcl_vpg": 1, "prcl_vort": 1,
                                      "prcl_tke": 1, "prcl_rho": 1}, wsm6_switch_context())
    tracker = npc.NestParcelTracker(settings=settings, layout=layout, child_deck=child_deck(g),
                                    parent_deck=parent_deck(), child_dt_s=1.0, parent_dt_s=3.0,
                                    xp=np, ng=NG)
    tracker.seed(g, model_time_s=0.0, parent_step=0)
    return tracker


def _parent_step(tracker, driver, child_runtime, parent_runtime, *, child_steps=3):
    clock = driver.parent.clock
    tracker.capture_parent(parent_runtime.dynamics.arrays, BOUNDARY)
    for _ in range(child_steps):
        tracker.capture_child(child_runtime.dynamics.arrays, BOUNDARY)
        tracker.advance_child()
    clock.completed_steps += 1
    clock.model_seconds += 3.0
    return tracker.after_parent_step(driver)


def test_parcels_are_handed_off_at_the_child_free_interior_and_across_a_nest_move(tmp_path):
    g = geometry()
    x_lo, x_hi, _, _ = npc.free_interior_m(g)
    assert (x_lo, x_hi) == (4600.0, 6200.0)
    settings = _settings(nparcels=3, seed_nx=3, seed_dx_m=725.0, seed_x0_m=4700.0)  # 4700, 5425, 6150
    tracker = _tracker(settings, g)
    wind = lambda x, y, z: (20.0 + 0 * x, 0 * y, 0 * z)  # noqa: E731
    child = _runtime(child_deck(g), padded_winds(child_deck(g), wind))
    parent = _runtime(parent_deck(), padded_winds(parent_deck(), wind))
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=_Clock(), **vars(parent)), child=child)
    tracker.open_track(tmp_path / npc.NEST_CHILD_TRACK_DIRECTORY)

    _parent_step(tracker, driver, child, parent)
    owner = np.asarray(tracker.owner)
    x = np.asarray(tracker.positions)[0]
    assert x.tolist() == pytest.approx([4760.0, 5485.0, 6210.0], abs=1e-3)
    assert owner.tolist() == [npc.OWNER_CHILD, npc.OWNER_CHILD, npc.OWNER_PARENT]

    # The nest moves one parent cell east (nest_move.execute); both domains are
    # at one model time and the child's footprint is now [5200, 6800].
    moved = geometry(parent_i0=7)
    driver.geometry = moved
    child = _runtime(child_deck(moved), padded_winds(child_deck(moved), wind))
    driver.child = child
    before = np.asarray(tracker.positions).copy()
    row = _parent_step(tracker, driver, child, parent, child_steps=3)
    after = np.asarray(tracker.positions)
    # Child parcels took three child steps (on the old lattice, before the
    # move); the parent-owned one took one parent step of 3 s.  Nobody's
    # position was touched by the move or the hand-off.
    assert (after[0] - before[0]).tolist() == pytest.approx([60.0, 60.0, 60.0], abs=2e-3)
    owner = np.asarray(tracker.owner)
    assert owner.tolist() == [npc.OWNER_PARENT, npc.OWNER_CHILD, npc.OWNER_CHILD]
    assert tracker.child_deck.minx_m == 4200.0
    assert tracker.handoffs == {"child_to_parent": 2, "parent_to_child": 1}
    events = [(e["parcel"], e["from"], e["to"]) for e in tracker.events]
    assert events == [(2, "child", "parent"), (0, "child", "parent"), (2, "parent", "child")]
    assert row["child_placement"]["parent_i0"] == 7
    assert row["counts"] == {"child": 2, "parent": 1, "terminated": 0}

    _parent_step(tracker, driver, child, parent)
    assert np.asarray(tracker.positions)[0].tolist() == pytest.approx([4880.0, 5605.0, 6330.0], abs=3e-3)

    receipt = tracker.publish_track()
    root = tmp_path / npc.NEST_CHILD_TRACK_DIRECTORY
    frames, npvals, nparcels = receipt["arena_shape_frames_npvals_nparcels"]
    arena = np.frombuffer((root / npc.NEST_PARCEL_ARENA_FILENAME).read_bytes(), "<f4").reshape(frames, npvals, nparcels)
    names = receipt["attribute_order"]
    assert frames == 3 and names[-1] == "grid"
    assert arena[:, names.index("grid")].tolist() == [[1, 1, 2], [2, 1, 1], [2, 1, 1]]
    # Every sample on either grid reads the same 20 m/s wind and 301 K.
    assert np.allclose(arena[:, names.index("u")], 20.0, atol=1e-4)
    assert np.allclose(arena[:, names.index("th")], 301.0, atol=1e-3)
    assert np.allclose(arena[:, names.index("zvort")], 0.0, atol=1e-6)
    events_tsv = (root / npc.NEST_PARCEL_EVENTS_FILENAME).read_text().splitlines()
    assert len(events_tsv) == 4 and "free interior" in events_tsv[1]
    on_disk = json.loads((root / npc.NEST_PARCEL_RECEIPT_FILENAME).read_text())
    assert on_disk["tracker"]["changes_the_simulated_storm"] is False
    assert "parent grid origin" in on_disk["tracker"]["coordinate_frame"]


def test_a_frame_samples_vorticity_buoyancy_and_winds_on_the_owning_grid(tmp_path):
    g = geometry()
    omega = 0.01
    deck = child_deck(g)
    rotation = lambda x, y, z: (-omega * (y - 4500.0), omega * (x - 5400.0), 0 * z)  # noqa: E731
    tracker = _tracker(_settings(nparcels=2, seed_nx=2, seed_dx_m=200.0, seed_x0_m=5300.0,
                                 seed_z0_m=3.0), g)
    child = _runtime(deck, padded_winds(deck, rotation))
    parent = _runtime(parent_deck(), padded_winds(parent_deck(), rotation))
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=_Clock(), **vars(parent)), child=child)
    tracker.open_track(tmp_path / "track")
    driver.parent.clock.completed_steps = 3
    positions = np.asarray(tracker.positions)
    row = tracker.record_frame(driver, step=3, model_time_s=9.0, positions=positions,
                               owner=np.asarray(tracker.owner))
    assert row["counts"]["child"] == 2
    receipt = tracker.publish_track()
    names = receipt["attribute_order"]
    arena = np.frombuffer((tmp_path / "track" / npc.NEST_PARCEL_ARENA_FILENAME).read_bytes(), "<f4").reshape(1, len(names), 2)
    # parcel.F:1066-1067's corner vorticity of solid-body rotation is 2*omega,
    # sampled at 3 m -- below zh(1), through the parcel_interp:902-959 ghost.
    assert arena[0, names.index("zvort")].tolist() == pytest.approx([2 * omega] * 2, rel=2e-4)
    assert arena[0, names.index("v")].tolist() == pytest.approx([omega * (-100.0), omega * 100.0], abs=2e-4)
    g_cm1 = np.float32(9.81) * (np.float32(1.0) * (np.float32(1.0) / np.float32(300.0)))
    assert arena[0, names.index("b")].tolist() == pytest.approx([float(g_cm1)] * 2, rel=1e-5)
    assert arena[0, names.index("tke")].tolist() == pytest.approx([0.25, 0.25])
    assert arena[0, names.index("mtime")].tolist() == [9.0, 9.0]
    assert "kmh" not in names or np.all(arena[0, names.index("kmh")] == np.float32(-1.0e30))


def test_a_parcel_that_leaves_the_child_between_boundaries_is_terminated_with_its_reason(tmp_path):
    g = geometry()
    tracker = _tracker(_settings(seed_x0_m=6100.0, nparcels=1), g)
    wind = lambda x, y, z: (500.0 + 0 * x, 0 * y, 0 * z)  # noqa: E731
    child = _runtime(child_deck(g), padded_winds(child_deck(g), wind))
    parent = _runtime(parent_deck(), padded_winds(parent_deck(), wind))
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=_Clock(), **vars(parent)), child=child)
    _parent_step(tracker, driver, child, parent)
    assert np.asarray(tracker.owner).tolist() == [npc.OWNER_TERMINATED]
    assert float(np.asarray(tracker.positions)[0, 0]) == pytest.approx(7100.0, abs=1e-3)
    assert "left the lattice" in tracker.events[0]["reason"]
    assert tracker.record()["terminations"]


def test_an_attribute_hook_appends_named_columns_and_is_fixed_before_the_track_opens(tmp_path):
    g = geometry()
    tracker = _tracker(_settings(), g)

    class Budget:
        names = ("zeta_tilting",)
        units = ("1/s^2",)
        marker = "a test hook"

        def sample(self, context):
            out = np.full(context.positions.shape[1], np.float32(-1.0e30), np.float32)
            chosen = context.indices[npc.OWNER_CHILD]
            out[chosen] = np.float32(0.125)
            return {"zeta_tilting": out}

    tracker.register_hook(Budget())
    with pytest.raises(ValueError, match="already published"):
        tracker.register_hook(Budget())
    wind = lambda x, y, z: (0 * x, 0 * y, 0 * z)  # noqa: E731
    child = _runtime(child_deck(g), padded_winds(child_deck(g), wind))
    parent = _runtime(parent_deck(), padded_winds(parent_deck(), wind))
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=_Clock(), **vars(parent)), child=child)
    tracker.open_track(tmp_path / "track")
    with pytest.raises(RuntimeError, match="before the track opens"):
        tracker.register_hook(Budget())
    _parent_step(tracker, driver, child, parent)
    receipt = tracker.publish_track()
    assert receipt["attribute_order"][-1] == "zeta_tilting"
    table = (tmp_path / "track" / npc.NEST_PARCEL_TABLE_FILENAME).read_text()
    assert "zeta_tilting\t1/s^2" in table
    shape = receipt["arena_shape_frames_npvals_nparcels"]
    arena = np.frombuffer((tmp_path / "track" / npc.NEST_PARCEL_ARENA_FILENAME).read_bytes(), "<f4").reshape(shape)
    # Parcel 0 is the one placed, child-owned parcel; 1 and 2 are unplaced.
    assert arena[0, -1].tobytes() == np.array([0.125, -1.0e30, -1.0e30], dtype="<f4").tobytes()


# ---------------------------------------------------------------------------
# Configuration, seeding refusal and the door.
# ---------------------------------------------------------------------------
def _door_fixture():
    path = Path(__file__).with_name("test_nest_config_to_call_site.py")
    spec = importlib.util.spec_from_file_location("_nest_door_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nest_child_toml(**overrides) -> str:
    values = {"seed_frame": '"parent-grid-origin"', "nparcels": 20460, "prclfrq_s": 10.0,
              "seed_nx": 44, "seed_ny": 31, "seed_nz": 15, "seed_dx_m": 150.0, "seed_dy_m": 150.0,
              "seed_dz_m": 100.0, "seed_x0_m": 19500.0, "seed_y0_m": 20000.0, "seed_z0_m": 5.0}
    values.update(overrides)
    return "\n[lagrangian.nest_child]\n" + "\n".join(f"{k} = {v}" for k, v in values.items()) + "\n"


def _config(nest: bool = True, **overrides):
    from cm1core.config import parse_mapping

    fixture = _door_fixture()
    text = fixture.BASE_TOML + (fixture.NEST_TABLE if nest else "") + _nest_child_toml(**overrides)
    return parse_mapping(tomllib.loads(text))


def test_every_nest_child_key_is_required_and_unknown_ones_are_refused():
    full = {"seed_frame": "parent-grid-origin", "nparcels": 8, "prclfrq_s": 1.0, "seed_nx": 2,
            "seed_ny": 2, "seed_nz": 2, "seed_dx_m": 1.0, "seed_dy_m": 1.0, "seed_dz_m": 1.0,
            "seed_x0_m": 0.0, "seed_y0_m": 0.0, "seed_z0_m": 0.0}
    assert set(full) == NEST_CHILD_KEYS
    for key in sorted(NEST_CHILD_KEYS):
        partial = {k: v for k, v in full.items() if k != key}
        with pytest.raises(LagrangianConfigError, match=f"nest_child.{key} must be declared"):
            parse_lagrangian_table({"nest_child": partial})
    with pytest.raises(LagrangianConfigError, match="seed_origin_m is not a child-parcel selector"):
        parse_lagrangian_table({"nest_child": {**full, "seed_origin_m": 0.0}})
    with pytest.raises(LagrangianConfigError, match="not from the domain centre"):
        parse_lagrangian_table({"nest_child": {**full, "seed_frame": "domain-centre"}})
    with pytest.raises(LagrangianConfigError, match="init3d.F:375-383"):
        parse_lagrangian_table({"nest_child": {**full, "nparcels": 7}})
    with pytest.raises(LagrangianConfigError, match="must be positive"):
        parse_lagrangian_table({"nest_child": {**full, "seed_dx_m": 0.0}})


def test_the_sub_table_is_projected_only_when_declared():
    without = parse_lagrangian_table({"iprcl": 1})
    assert without.nest_child is None and "nest_child" not in without.projection()
    declared = parse_lagrangian_table({"iprcl": 1, "nest_child": {
        "seed_frame": "parent-grid-origin", "nparcels": 1, "prclfrq_s": 1.0, "seed_nx": 1,
        "seed_ny": 1, "seed_nz": 1, "seed_dx_m": 1.0, "seed_dy_m": 1.0, "seed_dz_m": 1.0,
        "seed_x0_m": 19500.0, "seed_y0_m": 30000.0, "seed_z0_m": 5.0}})
    assert declared.projection()["nest_child"]["seed_x0_m"] == 19500.0
    assert declared.projection()["nest_child"]["seed_frame"] == "parent-grid-origin"
    from test_lagrangian_parcels import wsm6_switch_context

    record = declared.record(wsm6_switch_context())
    assert "nest_child" in record["gm1_extension_keys"]
    # param.F:6257-6282 resolves the child's layout exactly as the parent's.
    assert record["nest_child_attributes"] == record["parcel_attributes"]
    assert record["nest_child_attributes"]["switches"]["prcl_nc"] == 0
    assert "parent grid origin" in record["nest_child"]["frame"]
    assert without.projection() == {k: v for k, v in declared.projection().items() if k != "nest_child"}


def test_the_door_admits_a_child_box_and_seals_it_into_the_document():
    from cm1core.cartesian_door import require_nest

    config = _config()
    assert config.canonical_dict()["lagrangian"]["nest_child"]["nparcels"] == 20460
    binding = require_nest(config)
    assert binding is not None
    other = _config(seed_y0_m=20150.0)
    assert other.sha256 != config.sha256


def test_the_door_refuses_last_nights_box_where_it_leaves_this_childs_free_interior():
    """y 30-34.5 km on a child whose free interior ends at 34 km: refused, not clipped."""

    from cm1core.cartesian_door import require_nest

    with pytest.raises(CM1Refusal) as excinfo:
        require_nest(_config(seed_y0_m=30000.0))
    assert excinfo.value.detail.code == "CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT"
    assert "parent grid origin" in excinfo.value.detail.reason
    assert "34000.0" in excinfo.value.detail.reason


@pytest.mark.parametrize("overrides, code", [
    ({"prclfrq_s": 10.25}, "CM1-CARTESIAN-NEST-PARCELS-CADENCE"),
    ({"seed_x0_m": 15500.0}, "CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT"),
    ({"seed_z0_m": -1.0}, "CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT"),
])
def test_the_door_refuses_each_inadmissible_child_box_by_name(overrides, code):
    from cm1core.cartesian_door import require_nest

    with pytest.raises(CM1Refusal) as excinfo:
        require_nest(_config(**overrides))
    assert excinfo.value.detail.code == code


def test_child_parcels_without_a_nest_are_refused():
    from cm1core.cartesian_door import require_nest

    with pytest.raises(CM1Refusal) as excinfo:
        require_nest(_config(nest=False))
    assert excinfo.value.detail.code == "CM1-CARTESIAN-NEST-PARCELS-NO-NEST"


def test_a_runtime_placement_that_no_longer_holds_the_box_is_refused_at_seeding():
    tracker_settings = _settings(seed_x0_m=4700.0)
    # Its own ValueError subclass: nest_live refuses exactly this as a box to move.
    with pytest.raises(npc.NestParcelSeedExtentError, match="free interior"):
        _tracker(tracker_settings, geometry(parent_i0=7))


# ---------------------------------------------------------------------------
# Plumbing: the child's own capture and step, the parent's capture, the sidecar.
# ---------------------------------------------------------------------------
def test_the_child_runtime_captures_and_steps_its_parcels_after_its_own_hooks():
    from cm1core.nest_live import make_child_runtime_class

    calls = []

    class Base:
        def __init__(self):
            self.dynamics = SimpleNamespace(arrays={"rru": 1})
            self.config = "child-config"

        def _capture_parcel_velocity(self):
            calls.append("base-capture")

        def _advance_lagrangian(self):
            calls.append("base-advance")

    runtime = make_child_runtime_class(Base)()
    runtime._capture_parcel_velocity()
    runtime.nest_parcels = SimpleNamespace(
        capture_child=lambda arrays, config: calls.append(("capture", config)),
        advance_child=lambda: calls.append("advance"),
    )
    runtime._capture_parcel_velocity()
    runtime._advance_lagrangian()
    assert calls == ["base-capture", "base-capture", ("capture", "child-config"), "base-advance", "advance"]


def test_the_parent_captures_its_average_for_handed_off_parcels_and_publishes_the_track(tmp_path):
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

    seen = []
    tracker = SimpleNamespace(
        capture_parent=lambda arrays, config: seen.append(("capture", arrays, config)),
        open_track=lambda path: seen.append(("open", path.name)),
        publish_track=lambda: {"schema": "x"},
    )
    fake = SimpleNamespace(nest_driver=SimpleNamespace(parcels=tracker), lagrangian=None,
                           dynamics=SimpleNamespace(arrays="arrays"), config="config")
    CudaCartesianRuntime._capture_parcel_velocity(fake)
    assert seen == [("capture", "arrays", "config")]
    assert CudaCartesianRuntime._nest_parcel_track(fake, tmp_path) == {}
    assert seen[-1] == ("open", npc.NEST_CHILD_TRACK_DIRECTORY)
    assert CudaCartesianRuntime._nest_parcel_track(fake) == {"nest_child_lagrangian": {"schema": "x"}}
    bare = SimpleNamespace(nest_driver=None, lagrangian=None)
    assert CudaCartesianRuntime._nest_parcel_track(bare) == {}


def test_each_child_frame_gets_its_placement_and_deck_beside_it(tmp_path):
    from cm1core.grid_geometry import CartesianGridGeometry

    g = geometry(parent_i0=7)
    grid = CartesianGridGeometry(g.nx, g.ny, 27, g.dx_m, g.dy_m, 100.0, child_levels())
    child_config = SimpleNamespace(grid=grid, bbc=1, tbc=1, u_move_m_s=12.5, v_move_m_s=3.0)
    document = npc.child_placement_document(index=4, step=36, model_time_s=180.0, geometry=g,
                                            child_config=child_config, domainlocx_m=2250.0,
                                            domainlocy_m=540.0)
    path = npc.write_child_placement(tmp_path, document)
    assert path == tmp_path / "placements" / "frame-000004.json"
    loaded = json.loads(path.read_text())
    assert loaded["origin_x_m"] == 4200.0 and loaded["parent_i0"] == 7
    assert loaded["zf_cm1_indexed"] == [float(v) for v in child_levels().zf]
    assert loaded["vertical_sha256"] == child_levels().sha256
    with pytest.raises(FileExistsError):
        npc.write_child_placement(tmp_path, document)


# ---------------------------------------------------------------------------
# bbc = 3: parcel.F:470-515 (parcel_driver) and :1515-1583 (parcel_interp).
# ---------------------------------------------------------------------------
from cm1core.cm1_vertical_grid import uniform_vertical_levels  # noqa: E402


def uniform_parent_deck() -> npc.ParcelDeck:
    """A uniform deck in the domain-centred frame with mixed periodic/open laterals."""

    return npc.deck_from_levels(name="parent", nx=18, ny=16, dx_m=600.0, dy_m=600.0,
                                minx_m=-5400.0, miny_m=-4800.0, levels=uniform_vertical_levels(12, 125.0),
                                lateral=(1, 1, 2, 2), periodic=True)


def _deck(which: str) -> npc.ParcelDeck:
    return {"child": lambda: child_deck(geometry()), "parent": parent_deck,
            "parent-uniform": uniform_parent_deck}[which]()


def _libm_logf():
    import ctypes
    import ctypes.util

    library = ctypes.CDLL(ctypes.util.find_library("m"))
    function = library.logf
    function.argtypes = [ctypes.c_float]
    function.restype = ctypes.c_float
    return lambda value: np.float32(function(float(np.float32(value))))


def cm1_zntmp(znt, deck):
    """cm1.F:2192-2194, bc.F:12-172 bc2d (serial) and turb.F:1233-1237, loop for loop.

    Returned as a dict from CM1 (i, j) to the word; an independent oracle for
    nest_parcels.pad_roughness.
    """

    ni, nj, ngxy = deck.nx, deck.ny, NG
    tem = np.float32(1.0e-6)
    s = {(i, j): tem for i in range(1 - ngxy, ni + ngxy + 1) for j in range(1 - ngxy, nj + ngxy + 1)}
    for j in range(1, nj + 1):
        for i in range(1, ni + 1):
            s[(i, j)] = np.float32(znt[j - 1, i - 1])
    wbc, ebc, sbc, nbc = deck.lateral
    ib, ie, jb, je = 1 - ngxy, ni + ngxy, 1 - ngxy, nj + ngxy
    for j in range(jb, je + 1):
        if wbc == 1:
            for i in range(1, ngxy + 1):
                s[(1 - i, j)] = s[(ni + 1 - i, j)]
        else:
            for i in range(1 - ngxy, 1):
                s[(i, j)] = s[(1, j)]
    for j in range(jb, je + 1):
        if ebc == 1:
            for i in range(1, ngxy + 1):
                s[(ni + i, j)] = s[(i, j)]
        else:
            for i in range(ni + 1, ni + ngxy + 1):
                s[(i, j)] = s[(ni, j)]
    if sbc == 1:
        for j in range(1, ngxy + 1):
            for i in range(ib, ie + 1):
                s[(i, 1 - j)] = s[(i, nj + 1 - j)]
    else:
        for j in range(1 - ngxy, 1):
            for i in range(ib, ie + 1):
                s[(i, j)] = s[(i, 1)]
    if nbc == 1:
        for j in range(1, ngxy + 1):
            for i in range(ib, ie + 1):
                s[(i, nj + j)] = s[(i, j)]
    else:
        for j in range(nj + 1, nj + ngxy + 1):
            for i in range(ib, ie + 1):
                s[(i, j)] = s[(i, nj)]
    zntmp = {key: tem for key in s}
    for j in range(0, nj + 2):
        for i in range(0, ni + 2):
            zntmp[(i, j)] = s[(i, j)]
    return zntmp


def test_pad_roughness_is_bc2d_then_turb_F_s_first_ring_copy():
    rng = np.random.default_rng(11)
    for deck in (child_deck(geometry()), parent_deck(), uniform_parent_deck()):
        znt = rng.uniform(1.0e-4, 0.8, (deck.ny, deck.nx)).astype(np.float32)
        padded = npc.pad_roughness(np, znt, deck, ng=NG)
        oracle = cm1_zntmp(znt, deck)
        for (i, j), word in oracle.items():
            assert padded[j - 1 + NG, i - 1 + NG].tobytes() == np.float32(word).tobytes(), (deck.name, i, j)


def _semi_slip(deck, rng, imove):
    znt = rng.uniform(1.0e-4, 0.6, (deck.ny, deck.nx)).astype(np.float32)
    moves = (12.5, -3.25) if imove else (0.0, 0.0)
    return znt, npc.SemiSlip(npc.pad_roughness(np, znt, deck, ng=NG), imove, *moves)


def _random_winds(deck, rng, scale):
    shapes = ((deck.nz + 2, deck.ny + 2 * NG, deck.nx + 1 + 2 * NG),
              (deck.nz + 2, deck.ny + 1 + 2 * NG, deck.nx + 2 * NG),
              (deck.nz + 3, deck.ny + 2 * NG, deck.nx + 2 * NG))
    u, v = (rng.normal(0.0, scale, shape).astype(np.float32) for shape in shapes[:2])
    w = rng.normal(0.0, 0.05, shapes[2]).astype(np.float32)
    return u, v, w


@requires_gxx
@pytest.mark.parametrize("dt_s", [0.7, -1.3])
@pytest.mark.parametrize("imove", [0, 1])
@pytest.mark.parametrize("which", ["child", "parent", "parent-uniform"])
def test_the_semi_slip_kernel_text_is_the_host_reference_raw_word(harness, which, imove, dt_s):
    rng = np.random.default_rng(20260916)
    deck = _deck(which)
    scale = 40.0 if which == "child" else 400.0
    u, v, w = _random_winds(deck, rng, scale)
    n = 1500
    positions = _hostile_positions(rng, deck, n)
    zh1 = float(deck.zh[1])
    third = np.arange(n) % 3 == 0
    positions[2, third] = rng.uniform(0.02 * zh1, zh1, int(third.sum())).astype(np.float32)
    positions[2, 7] = np.float32(deck.zh[1])  # exactly zh(1): parcel.F:474 is strict
    positions[:, -1] = np.float32(-1.0e30)
    owner = np.ones(n, np.int32)
    status = np.zeros(n, np.int32)
    _, surface = _semi_slip(deck, rng, imove)
    for order in ("linear", "shuffled"):
        ref_pos, ref_status = npc.reference_rk2(np, positions, owner, status, code=1, u=u, v=v, w=w,
                                                deck=deck, dt_s=dt_s, ng=NG, surface=surface)
        ker_pos, ker_status = harness.rk2(positions=positions, owner=owner, status=status, u=u, v=v,
                                          w=w, deck=deck, code=1, dt_s=dt_s, ng=NG, order=order,
                                          surface=surface)
        assert ref_pos.tobytes() == ker_pos.tobytes()
        assert np.array_equal(ref_status, ker_status)
    # Not vacuous: the log layer moved the low parcels and nothing else.
    plain, _ = npc.reference_rk2(np, positions, owner, status, code=1, u=u, v=v, w=w, deck=deck,
                                 dt_s=dt_s, ng=NG)
    ok = ref_status == 0
    differs = ok & np.any(plain != ref_pos, axis=0)
    assert int((differs & third).sum()) > 100
    high = ok & (positions[2] > np.float32(3.0 * zh1 + abs(dt_s) * 5.0))
    assert high.any() and plain[:, high].tobytes() == ref_pos[:, high].tobytes()


def _cm1_first_stage_flags(deck, x, y, z):
    """parcel.F:161-180 and :251-255 as written (first stage)."""

    ni, nj = deck.nx, deck.ny
    xf = lambda i: np.float32(deck.face_x(i))  # noqa: E731
    yf = lambda j: np.float32(deck.face_y(j))  # noqa: E731
    iflag = jflag = -100
    i = ni + 1
    while iflag < 0 and i > 1:
        i -= 1
        if x >= xf(i) and x <= xf(i + 1):
            iflag = i
    j = nj + 1
    while jflag < 0 and j > 1:
        j -= 1
        if y >= yf(j) and y <= yf(j + 1):
            jflag = j
    kflag = 1
    while z >= deck.zf[kflag + 1]:
        kflag += 1
    return iflag, jflag, kflag


def _halves(deck):
    f32 = np.float32
    xh = lambda i: f32(deck.minx_m) + (f32(i) - f32(0.5)) * f32(deck.dx_m)  # noqa: E731
    yh = lambda j: f32(deck.miny_m) + (f32(j) - f32(0.5)) * f32(deck.dy_m)  # noqa: E731
    return xh, yh


def _get2d_scalar(s, i, j, x, y, xs, ys, deck):
    """parcel.F:1719-1787 get2d, statement by statement."""

    f32 = np.float32
    xh, yh = _halves(deck)
    xf = lambda k: f32(deck.face_x(k))  # noqa: E731
    yf = lambda k: f32(deck.face_y(k))  # noqa: E731
    rx = (x - xf(i)) / (xf(i + 1) - xf(i)) if xs == 1 else (x - xh(i)) / (xh(i + 1) - xh(i))
    ry = (y - yf(j)) / (yf(j + 1) - yf(j)) if ys == 1 else (y - yh(j)) / (yh(j + 1) - yh(j))
    w1 = (f32(1.0) - rx) * (f32(1.0) - ry)
    w2 = rx * (f32(1.0) - ry)
    w3 = (f32(1.0) - rx) * ry
    w7 = rx * ry
    return s(i, j) * w1 + s(i + 1, j) * w2 + s(i, j + 1) * w3 + s(i + 1, j + 1) * w7


@pytest.mark.parametrize("imove", [0, 1])
@pytest.mark.parametrize("which", ["child", "parent", "parent-uniform"])
def test_the_log_layer_is_parcel_F_470_515_statement_by_statement(which, imove):
    """An independent transcription of parcel_driver's bbc=3 branch, one parcel at a time."""

    rng = np.random.default_rng(5)
    deck = _deck(which)
    u, v, w = _random_winds(deck, rng, 30.0)
    znt, surface = _semi_slip(deck, rng, imove)
    zntmp = cm1_zntmp(znt, deck)
    logf = _libm_logf()
    f32 = np.float32
    n = 300
    x = rng.uniform(deck.minx_m + 0.5 * deck.dx_m, deck.minx_m + (deck.nx - 0.5) * deck.dx_m, n).astype(np.float32)
    y = rng.uniform(deck.miny_m + 0.5 * deck.dy_m, deck.miny_m + (deck.ny - 0.5) * deck.dy_m, n).astype(np.float32)
    z = rng.uniform(0.0, float(deck.zh[1]), n).astype(np.float32)
    got_u, got_v, got_w, status = npc.reference_sample_uvw(np, u, v, w, deck, x, y, z, second_stage=False,
                                                           ng=NG, surface=surface)
    plain_w = npc.reference_sample_uvw(np, u, v, w, deck, x, y, z, second_stage=False, ng=NG)[2]
    assert not status.any() and got_w.tobytes() == plain_w.tobytes()
    xh, yh = _halves(deck)
    ua = lambda i, j: u[1, j - 1 + NG, i - 1 + NG]  # noqa: E731
    va = lambda i, j: v[1, j - 1 + NG, i - 1 + NG]  # noqa: E731
    zn = lambda i, j: zntmp[(i, j)]  # noqa: E731
    umove, vmove = f32(surface.u_move_m_s), f32(surface.v_move_m_s)
    for p in range(n):
        x3d, y3d, z3d = x[p], y[p], z[p]
        iflag, jflag, _ = _cm1_first_stage_flags(deck, x3d, y3d, z3d)
        assert z3d < deck.zh[1]
        i, j = iflag, jflag
        if y3d < yh(j):
            j = j - 1
        uval = _get2d_scalar(ua, i, j, x3d, y3d, 1, 0, deck)
        i, j = iflag, jflag
        if x3d < xh(i):
            i = i - 1
        vval = _get2d_scalar(va, i, j, x3d, y3d, 0, 1, deck)
        i, j = iflag, jflag
        if x3d < xh(i):
            i = i - 1
        if y3d < yh(j):
            j = j - 1
        z0 = _get2d_scalar(zn, i, j, x3d, y3d, 0, 0, deck)
        rznt = f32(1.0) / z0
        var = logf((z3d + z0) * rznt) / logf((deck.zh[1] + z0) * rznt)
        if imove == 1:
            uval = (uval + umove) * var - umove
            vval = (vval + vmove) * var - vmove
        else:
            uval = uval * var
            vval = vval * var
        assert np.float32(got_u[p]).tobytes() == np.float32(uval).tobytes(), p
        assert np.float32(got_v[p]).tobytes() == np.float32(vval).tobytes(), p


def _semi_slip_runtime(deck, winds, znt, moves):
    u, v, w = winds
    return SimpleNamespace(
        config=SimpleNamespace(bbc=3, tbc=1, u_move_m_s=moves[0], v_move_m_s=moves[1],
                               imove=int(moves != (0.0, 0.0))),
        closure=SimpleNamespace(surface_state={"znt": znt}),
        dynamics=SimpleNamespace(arrays={"u3d": u, "v3d": v, "w3d": w}),
    )


@pytest.mark.parametrize("moves", [(0.0, 0.0), (12.5, -3.25)])
@pytest.mark.parametrize("which", ["child", "parent-uniform"])
def test_the_recorded_winds_and_zvort_below_zh1_are_parcel_interp_s_log_layer(which, moves):
    """parcel.F:1515-1581: (uval+umove)*var, and the log-layer corner vorticity through get2d."""

    rng = np.random.default_rng(9)
    deck = _deck(which)
    winds = _random_winds(deck, rng, 20.0)
    znt = rng.uniform(1.0e-3, 0.4, (deck.ny, deck.nx)).astype(np.float32)
    carriers = npc.RuntimeCarriers(_semi_slip_runtime(deck, winds, znt, moves), deck, np, NG)
    zntmp = cm1_zntmp(znt, deck)
    logf = _libm_logf()
    f32 = np.float32
    n = 120
    x = rng.uniform(deck.minx_m + 1.5 * deck.dx_m, deck.minx_m + (deck.nx - 1.5) * deck.dx_m, n)
    y = rng.uniform(deck.miny_m + 1.5 * deck.dy_m, deck.miny_m + (deck.ny - 1.5) * deck.dy_m, n)
    z = rng.uniform(0.0, float(deck.zh[1]), n)
    z[:10] = rng.uniform(float(deck.zh[1]), float(deck.zh[3]), 10)  # above: untouched
    positions = np.array([x, y, z], dtype=np.float32)
    got = npc.sample_parcel_attributes(np, carriers, positions, ["u", "v", "w", "zvort"])
    plain_rt = _semi_slip_runtime(deck, winds, znt, moves)
    plain_rt.config.bbc = 1
    u3d, v3d, _ = winds
    xh, yh = _halves(deck)
    umove, vmove = f32(moves[0]), f32(moves[1])
    zh1 = deck.zh[1]
    rdx, rdy = f32(1.0) / f32(deck.dx_m), f32(1.0) / f32(deck.dy_m)
    U = lambda i, j: u3d[1, j - 1 + NG, i - 1 + NG]  # noqa: E731
    V = lambda i, j: v3d[1, j - 1 + NG, i - 1 + NG]  # noqa: E731
    for p in range(n):
        x3d, y3d, z3d = positions[:, p]
        if not z3d < zh1:
            continue
        iflag, jflag, _ = _cm1_first_stage_flags(deck, x3d, y3d, z3d)
        i, j = iflag, jflag
        if x3d < xh(i):
            i = i - 1
        if y3d < yh(j):
            j = j - 1
        z0 = _get2d_scalar(lambda a, b: zntmp[(a, b)], i, j, x3d, y3d, 0, 0, deck)
        i, j = iflag, jflag
        if y3d < yh(j):
            j = j - 1
        uval = _get2d_scalar(U, i, j, x3d, y3d, 1, 0, deck)
        i, j = iflag, jflag
        if x3d < xh(i):
            i = i - 1
        vval = _get2d_scalar(V, i, j, x3d, y3d, 0, 1, deck)
        rznt = f32(1.0) / z0
        var = logf((z3d + z0) * rznt) / logf((zh1 + z0) * rznt)
        assert got["u"][p].tobytes() == ((uval + umove) * var).tobytes()
        assert got["v"][p].tobytes() == ((vval + vmove) * var).tobytes()
        dum3, dum4, dum7 = {}, {}, {}
        for j in range(jflag - 1, jflag + 2):
            for i in range(iflag, iflag + 2):
                z0 = f32(0.5) * (zntmp[(i - 1, j)] + zntmp[(i, j)])
                rznt = f32(1.0) / z0
                dum3[(i, j)] = (U(i, j) + umove) * logf((z3d + z0) * rznt) / logf((zh1 + z0) * rznt)
        for j in range(jflag, jflag + 2):
            for i in range(iflag - 1, iflag + 2):
                z0 = f32(0.5) * (zntmp[(i, j - 1)] + zntmp[(i, j)])
                rznt = f32(1.0) / z0
                dum4[(i, j)] = (V(i, j) + vmove) * logf((z3d + z0) * rznt) / logf((zh1 + z0) * rznt)
        for j in range(jflag, jflag + 2):
            for i in range(iflag, iflag + 2):
                dum7[(i, j)] = (dum4[(i, j)] - dum4[(i - 1, j)]) * rdx * f32(1.0) \
                    - (dum3[(i, j)] - dum3[(i, j - 1)]) * rdy * f32(1.0)
        zvort = _get2d_scalar(lambda a, b: dum7[(a, b)], iflag, jflag, x3d, y3d, 1, 1, deck)
        assert got["zvort"][p].tobytes() == zvort.tobytes(), p
    plain = npc.sample_parcel_attributes(np, npc.RuntimeCarriers(plain_rt, deck, np, NG), positions[:, :10],
                                         ["u", "v", "w", "zvort"])
    for name in ("u", "v", "w", "zvort"):
        assert got[name][:10].tobytes() == plain[name].tobytes()
    assert got["w"].tobytes() == npc.sample_parcel_attributes(
        np, npc.RuntimeCarriers(plain_rt, deck, np, NG), positions, ["w"])["w"].tobytes()


def test_a_semi_slip_tracker_needs_the_grids_roughness_and_steps_with_it():
    g = geometry()
    tracker = _tracker(_settings(nparcels=2, seed_nx=2, seed_dx_m=150.0, seed_x0_m=5300.0, seed_z0_m=1.5), g)
    deck = child_deck(g)
    rng = np.random.default_rng(3)
    u, v, w = _random_winds(deck, rng, 5.0)
    boundary = SimpleNamespace(bbc=3, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)
    arrays = {"rru": u, "rrv": v, "rrw": w}
    with pytest.raises(RuntimeError, match="parcel.F:470-515"):
        tracker.capture_child(arrays, boundary)
    znt = rng.uniform(1.0e-3, 0.3, (deck.ny, deck.nx)).astype(np.float32)
    start = np.asarray(tracker.positions).copy()
    tracker.capture_child(arrays, boundary, roughness=znt)
    tracker.advance_child()
    surface = npc.SemiSlip(npc.pad_roughness(np, znt, deck, ng=NG), 0, 0.0, 0.0)
    uf, vf, wf = (a.copy() for a in (u, v, w))
    npc.fill_velocity_halos(np, uf, vf, wf, deck, ng=NG)
    npc.fill_velocity_ghosts(np, uf, vf, wf, deck, bbc=3, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)
    want, _ = npc.reference_rk2(np, start, np.asarray(tracker.owner), np.zeros(start.shape[1], np.int32), code=1,
                                u=uf, v=vf, w=wf, deck=deck, dt_s=1.0, ng=NG, surface=surface)
    assert np.asarray(tracker.positions).tobytes() == want.tobytes()


def test_the_door_admits_child_parcels_on_a_semi_slip_grid():
    defect = npc.nest_parcel_admission_defect(
        settings=_settings(), geometry=geometry(), child_grid=SimpleNamespace(nz=27, vertical=child_levels()),
        parent_dt_s=3.0, bbc=3, tbc=1, lateral=(2, 2, 2, 2), terrain=False,
    )
    assert defect is None
