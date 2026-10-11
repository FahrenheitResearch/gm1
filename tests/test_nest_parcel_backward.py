"""The offline backward-trajectory diagnostic, against closed forms and its refusals."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from cm1core import nest_parcel_backward as nb  # noqa: E402
from cm1core import nest_parcels as npc  # noqa: E402
from cm1core.lagrangian_config import parse_lagrangian_table  # noqa: E402

from test_nest_parcels import child_levels, geometry  # noqa: E402


def _seeding(**overrides):
    table = {"seed_frame": "parent-grid-origin", "nparcels": 4, "prclfrq_s": 1.0, "seed_nx": 2,
             "seed_ny": 2, "seed_nz": 1, "seed_dx_m": 100.0, "seed_dy_m": 80.0, "seed_dz_m": 1.0,
             "seed_x0_m": 5600.0, "seed_y0_m": 4400.0, "seed_z0_m": 30.0}
    table.update(overrides)
    return parse_lagrangian_table({"nest_child": table}).nest_child


def _frame(index, t, g, fn):
    deck = npc.deck_from_levels(name="child-frame", nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m,
                                minx_m=g.origin_x_m, miny_m=g.origin_y_m, levels=child_levels(),
                                lateral=(2, 2, 2, 2), periodic=False)
    nz, ny, nx = deck.nz, deck.ny, deck.nx
    u = np.full((nz, ny, nx + 1), np.float32(fn(t)[0]), np.float32)
    v = np.full((nz, ny + 1, nx), np.float32(fn(t)[1]), np.float32)
    w = np.full((nz + 1, ny, nx), np.float32(fn(t)[2]), np.float32)
    w[0] = 0.0
    w[nz] = 0.0
    placement = {"spec_zone": g.spec_zone, "relax_zone": g.relax_zone, "origin_x_m": g.origin_x_m,
                 "origin_y_m": g.origin_y_m, "dx_m": g.dx_m, "dy_m": g.dy_m, "bbc": 1, "tbc": 1,
                 "u_move_m_s": 0.0, "v_move_m_s": 0.0, "parent_i0": g.parent_i0,
                 "parent_j0": g.parent_j0}
    return nb.frame_from_arrays(index=index, model_time_s=t, placement=placement, deck=deck, u=u, v=v, w=w)


def test_a_steady_wind_is_reversed_exactly_across_a_nest_move_with_a_zero_error_estimate(tmp_path):
    """Frames 0-1 on the first placement, 2-3 one parent cell east: the blend samples each on its own lattice."""

    steady = lambda t: (-5.0, -2.0, 0.0)  # noqa: E731
    first, moved = geometry(parent_i0=6), geometry(parent_i0=7)
    frames = [_frame(0, 0.0, first, steady), _frame(1, 10.0, first, steady),
              _frame(2, 20.0, moved, steady), _frame(3, 30.0, moved, steady)]
    request = nb.BackwardRequest(_seeding(), start_time_s=30.0, end_time_s=0.0, dt_s=1.0)
    result = nb.integrate_backward(frames, request)
    rows = result.rows
    assert rows.shape == (31, len(nb.BACKWARD_ROWS), 4)
    assert (result.status == nb.STATUS_ACTIVE).all()
    start = rows[0, 0:2]
    end = rows[-1, 0:2]
    # Backward through a wind toward -x, -y: the air came from +x, +y.
    assert (end[0] - start[0]).tolist() == pytest.approx([150.0] * 4, abs=2e-3)
    assert (end[1] - start[1]).tolist() == pytest.approx([60.0] * 4, abs=2e-3)
    assert rows[-1, 9].tolist() == [0.0] * 4
    assert result.frames_used == [0, 1, 2, 3]
    receipt = nb.write_backward_trajectories(tmp_path / "out", result, frames, request)
    assert receipt["kind"] == "offline-diagnostic"
    assert receipt["label"].startswith("OFFLINE DIAGNOSTIC")
    assert any("Seibert" in c for c in receipt["citations"])
    assert receipt["frames_used"][2]["parent_i0"] == 7


def test_the_error_estimate_bounds_the_actual_error_of_a_wind_quadratic_in_time():
    """u(t) = -(a + b t^2): linear interpolation errs by b*T^3/6 per interval; the estimate is b*T^2/4 per second."""

    a, b = 5.0, 0.002
    wind = lambda t: (-(a + b * t * t), 0.0, 0.0)  # noqa: E731
    g = geometry()
    frames = [_frame(i, 10.0 * i, g, wind) for i in range(4)]
    request = nb.BackwardRequest(_seeding(seed_x0_m=4700.0, seed_nx=1, seed_ny=1, nparcels=1),
                                 start_time_s=30.0, end_time_s=0.0, dt_s=1.0)
    result = nb.integrate_backward(frames, request)
    x_start = float(result.rows[0, 0, 0])
    x_end = float(result.rows[-1, 0, 0])
    exact = x_start + (a * 30.0 + b * 30.0 ** 3 / 3.0)
    actual = abs(x_end - exact)
    estimate = float(result.rows[-1, 9, 0])
    assert actual == pytest.approx(3 * b * 10.0 ** 3 / 6.0, rel=2e-3)   # 1.0 m of interpolation error
    # The curvature is a divided difference of binary32 wind samples, so it is
    # exact only to their rounding.
    assert estimate == pytest.approx(30.0 * (10.0 ** 2 / 8.0) * 2 * b, rel=1e-4)  # 1.5 m estimated
    assert actual <= estimate
    assert result.rows[5, 8, 0] == pytest.approx((10.0 ** 2 / 8.0) * 2 * b, rel=1e-4)


@pytest.mark.parametrize("change, match", [
    ({"frames": 2}, "at least three frames"),
    ({"dt_s": 0.7}, "whole number of steps"),
    ({"start_time_s": 40.0}, "not inside the frames"),
    ({"seed_x0_m": 3700.0}, "outside frame"),
])
def test_requests_the_frames_cannot_answer_are_refused(change, match):
    steady = lambda t: (-5.0, 0.0, 0.0)  # noqa: E731
    g = geometry()
    frames = [_frame(i, 10.0 * i, g, steady) for i in range(change.get("frames", 4))]
    seeding = _seeding(**({"seed_x0_m": change["seed_x0_m"]} if "seed_x0_m" in change else {}))
    request = nb.BackwardRequest(seeding, start_time_s=change.get("start_time_s", 30.0),
                                 end_time_s=0.0, dt_s=change.get("dt_s", 1.0))
    with pytest.raises(nb.BackwardTrajectoryError, match=match):
        nb.integrate_backward(frames, request)


def test_the_tool_reads_published_child_frames_and_their_placements_from_disk(tmp_path):
    from cm1core.grid_geometry import CartesianGridGeometry

    import tools.nest_backward_trajectories as cli

    child_root = tmp_path / "nest-child"
    g = geometry()
    grid = CartesianGridGeometry(g.nx, g.ny, 27, g.dx_m, g.dy_m, 100.0, child_levels())
    child_config = SimpleNamespace(grid=grid, bbc=1, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)
    nz, ny, nx = 27, g.ny, g.nx
    for index, t in enumerate((0.0, 10.0, 20.0)):
        frame_dir = child_root / f"frame-{index:06d}"
        frame_dir.mkdir(parents=True)
        carriers = {"u3d": np.full((nz, ny, nx + 1), -4.0, "<f4"),
                    "v3d": np.full((nz, ny + 1, nx), 1.0, "<f4"),
                    "w3d": np.zeros((nz + 1, ny, nx), "<f4")}
        offset, rows, blob = 0, [], b""
        for name, array in carriers.items():
            shape = ",".join(str(s) for s in array.shape)
            rows.append("\t".join([name, "m", "m s-1", "u", shape, "1", "/stale/staging/path", str(offset)]))
            blob += array.tobytes()
            offset += array.nbytes
        (frame_dir / nb.HISTORY_ARENA).write_bytes(blob)
        (frame_dir / nb.HISTORY_TABLE).write_text("\n".join(rows) + "\n", encoding="ascii")
        npc.write_child_placement(child_root, npc.child_placement_document(
            index=index, step=index * 10, model_time_s=t, geometry=g, child_config=child_config,
            domainlocx_m=0.0, domainlocy_m=0.0))
    frames = nb.read_child_frames(child_root)
    assert [f.model_time_s for f in frames] == [0.0, 10.0, 20.0]
    code = cli.main(["--child-root", str(child_root), "--out", str(tmp_path / "back"),
                     "--seed-frame", "parent-grid-origin", "--seed-box", "5600,4400,30,50,50,10,2,2,1",
                     "--start-time-s", "20", "--end-time-s", "0", "--dt-s", "2"])
    assert code == 0
    receipt = json.loads((tmp_path / "back" / nb.BACKWARD_RECEIPT_FILENAME).read_text())
    assert receipt["kind"] == "offline-diagnostic"
    arena = np.frombuffer((tmp_path / "back" / nb.BACKWARD_ARENA_FILENAME).read_bytes(), "<f4")
    arena = arena.reshape(receipt["arena_shape_steps_rows_nparcels"])
    assert (arena[-1, 0] - arena[0, 0]).tolist() == pytest.approx([80.0] * 4, abs=2e-3)
    refused = cli.main(["--child-root", str(child_root), "--out", str(tmp_path / "again"),
                        "--seed-frame", "domain-centre", "--seed-box", "0,0,30,50,50,10,2,2,1",
                        "--start-time-s", "20", "--end-time-s", "0", "--dt-s", "2"])
    assert refused == 2


def test_a_frame_without_its_placement_is_refused_by_name(tmp_path):
    (tmp_path / "frame-000000").mkdir()
    with pytest.raises(nb.BackwardTrajectoryError, match="placement"):
        nb.read_child_frames(tmp_path)


def test_a_semi_slip_child_frame_carries_its_roughness_and_applies_the_log_layer(tmp_path):
    """parcel.F:470-515 offline: a steady wind below zh(1) is scaled by the log-layer ratio."""

    import math

    g = geometry()
    steady = lambda t: (-5.0, 0.0, 0.0)  # noqa: E731
    znt_word = 0.05
    frames = []
    for index, t in enumerate((0.0, 10.0, 20.0, 30.0)):
        frame = _frame(index, t, g, steady)
        placement = {**frame.placement, "bbc": 3, "imove": 0}
        deck = frame.deck
        roughness = np.full((deck.ny, deck.nx), np.float32(znt_word), np.float32)
        nz, ny, nx = deck.nz, deck.ny, deck.nx
        u = np.full((nz, ny, nx + 1), np.float32(-5.0), np.float32)
        v = np.zeros((nz, ny + 1, nx), np.float32)
        w = np.zeros((nz + 1, ny, nx), np.float32)
        with pytest.raises(nb.BackwardTrajectoryError, match="parcel.F:470-515"):
            nb.frame_from_arrays(index=index, model_time_s=t, placement=placement, deck=deck, u=u, v=v, w=w)
        frames.append(nb.frame_from_arrays(index=index, model_time_s=t, placement=placement, deck=deck,
                                           u=u, v=v, w=w, roughness=roughness))
    zh1 = float(frames[0].deck.zh[1])
    z = 1.0
    request = nb.BackwardRequest(_seeding(seed_z0_m=z), start_time_s=30.0, end_time_s=0.0, dt_s=1.0)
    result = nb.integrate_backward(frames, request)
    assert (result.status == nb.STATUS_ACTIVE).all()
    ratio = math.log((z + znt_word) / znt_word) / math.log((zh1 + znt_word) / znt_word)
    moved = result.rows[-1, 0] - result.rows[0, 0]
    assert moved.tolist() == pytest.approx([5.0 * ratio * 30.0] * 4, rel=2e-5)
    assert ratio < 1.0


def test_a_semi_slip_placement_writes_and_pins_the_frames_znt(tmp_path):
    from cm1core.grid_geometry import CartesianGridGeometry

    g = geometry(parent_i0=7)
    grid = CartesianGridGeometry(g.nx, g.ny, 27, g.dx_m, g.dy_m, 100.0, child_levels())
    child_config = SimpleNamespace(grid=grid, bbc=3, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0)
    document = npc.child_placement_document(index=2, step=12, model_time_s=60.0, geometry=g,
                                            child_config=child_config, domainlocx_m=0.0, domainlocy_m=0.0)
    with pytest.raises(ValueError, match="parcel.F:470-515"):
        npc.write_child_placement(tmp_path / "a", document)
    znt = np.random.default_rng(1).uniform(1e-3, 0.2, (g.ny, g.nx)).astype(np.float32)
    path = npc.write_child_placement(tmp_path / "b", document, znt)
    loaded = json.loads(path.read_text())
    sidecar = path.parent / loaded["roughness"]["file"]
    assert np.frombuffer(sidecar.read_bytes(), "<f4").reshape(g.ny, g.nx).tobytes() == znt.tobytes()
    import hashlib

    assert loaded["roughness"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert loaded["imove"] == 0
