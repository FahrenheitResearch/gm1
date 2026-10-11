"""A live nest child's own run-long surface swaths (sws, svs, sps, ...).

The child used to be built with ``diagnostics=None``, so a 20 m or 4 m child
published no swath at all and the 4 m swath visualisation had to rebuild one
from window records.  These tests hold the child's swaths to CM1's rules on
the host, through the real call sites with NumPy in CuPy's place:

* the child keeps exactly the parent's swath switches (``solve3.F:855-1136``
  on the child's own grid and step) and nothing else;
* a stationary child's swath is the running extremum of its per-step fields;
* a move shifts the swath with the child and fills the exposed strip from the
  parent's swath through the same capture operator (``nest_move``, THE SHIFT
  AND THE FILL), so a fixed ground point never decreases across a move;
* the child restart carries the planes (``restart_write.F:692-757``) and a
  resumed child continues from them word for word;
* the child frame publishes them, frame-relative, with no ground twin.

The device run of the same path is ``tools/gpu_check_child_swaths.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import nest_move as nm
from cm1core.cartesian_door import build_runtime_config
from cm1core.cartesian_history import cartesian_history_profile
from cm1core.cartesian_tornado_diagnostics import CartesianTornadoDiagnostics
from cm1core.config import load_config
from cm1core.errors import refuse
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.nest_live import NestDriver, child_runtime_config, child_swath_selection
from cm1core.nest_restart import SWATH_CARRIER, nest_child_restart_profile
from cm1core.tornado_diagnostics import (
    OUTPUT_SWITCH_NAMES,
    SWATH_INITIAL_VALUES,
    SWATH_NAMES,
    TornadoDiagnosticSelection,
)

from test_tornado_diagnostics import _BASE_DOCUMENT, _DIAGNOSTICS_BLOCK, _host_runtime

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")

_SWATHS_ONLY = """
[diagnostics]
output_uh = 0
output_vort = 0
output_pv = 0
output_sws = 1
output_svs = 1
output_sps = 1
output_srs = 0
output_sgs = 0
output_sus = 0
output_shs = 0
tornado_report = false
"""


def _runtime(tmp_path: Path, block: str, name: str):
    path = tmp_path / name
    path.write_text(_BASE_DOCUMENT + block, encoding="utf-8")
    return build_runtime_config(load_config(path))


def _child(parent, grid=None):
    return child_runtime_config(
        parent, parent.grid if grid is None else grid,
        child_dt_s=parent.dt_s / 5, acoustic_substeps=parent.acoustic_substeps,
    )


# ---------------------------------------------------------------------------
# 1. The child's contract.
# ---------------------------------------------------------------------------
def test_the_child_keeps_exactly_the_parents_swaths(tmp_path) -> None:
    parent = _runtime(tmp_path, _DIAGNOSTICS_BLOCK, "all.toml")
    assert parent.diagnostics.active and parent.diagnostics.tornado_report
    child = _child(parent)
    selection = child.diagnostics
    assert selection.swath_names == SWATH_NAMES == parent.diagnostics.swath_names
    # The instantaneous rotation fields and the report stay the parent's.
    assert (selection.output_uh, selection.output_vort, selection.output_pv) == (0, 0, 0)
    assert selection.tornado_report is False
    assert child.record()["diagnostics"]["output_sws"] == 1

    partial = _child(_runtime(tmp_path, _SWATHS_ONLY, "three.toml"))
    assert partial.diagnostics.swath_names == ("sws", "svs", "sps")

    none = _child(_runtime(tmp_path, "", "none.toml"))
    assert none.diagnostics.active is False
    assert child_swath_selection(None).active is False
    with pytest.raises(TypeError):
        child_swath_selection(object())


def test_a_swath_the_child_grid_cannot_carry_is_refused_by_name(tmp_path) -> None:
    parent = _runtime(tmp_path, _DIAGNOSTICS_BLOCK, "all.toml")
    g = parent.grid
    shallow = CartesianGridGeometry(g.nx, g.ny, 8, g.dx_m, g.dy_m, 500.0)  # 4 km top
    # The check RuntimeConfig.__post_init__ runs on the child contract.
    from cm1core.tornado_diagnostics import validate_tornado_diagnostic_selection

    with pytest.raises(ValueError, match="output_shs"):
        validate_tornado_diagnostic_selection(
            child_swath_selection(parent.diagnostics), grid=shallow,
            microphysics=parent.microphysics, bbc=parent.bbc,
        )
    # The deep parent grid carries every swath, so the child contract builds.
    assert _child(parent).diagnostics.output_sus == 1


# ---------------------------------------------------------------------------
# 2. A stationary child: the running extremum of every per-step field.
# ---------------------------------------------------------------------------
def test_a_stationary_childs_swath_is_the_running_extremum(tmp_path) -> None:
    child = _child(_runtime(tmp_path, _SWATHS_ONLY, "three.toml"))
    assert child.u_move_m_s != 0.0  # a translating frame: the twin would exist
    host = _host_runtime(child, seed=11)
    tornado = CartesianTornadoDiagnostics(host, total_steps=6, ground=False)
    rng = np.random.default_rng(5)
    fields = host.dynamics.fields
    per_step = {name: [] for name in ("sws", "svs", "sps")}
    for step in range(1, 7):
        fields["u3d"][...] = rng.standard_normal(fields["u3d"].shape).astype(np.float32) * 9
        fields["v3d"][...] = rng.standard_normal(fields["v3d"].shape).astype(np.float32) * 9
        fields["prs"][...] = (np.float32(95000.0) + rng.standard_normal(
            fields["prs"].shape).astype(np.float32) * np.float32(300.0))
        plane = np.empty(child.grid.surface_shape, np.float32)
        per_step["sws"].append(tornado.swaths.surface_wind_speed(
            u=fields["u3d"], v=fields["v3d"], out=plane).copy())
        per_step["svs"].append(tornado.swaths.surface_vertical_vorticity(
            u=fields["u3d"], v=fields["v3d"], out=plane).copy())
        per_step["sps"].append(fields["prs"][0] - np.float32(95000.0))
        host.clock.domainlocx_m = child.u_move_m_s * child.dt_s * step
        host.clock.domainlocy_m = child.v_move_m_s * child.dt_s * step
        tornado.update(step)
        frame = tornado.swaths.frame
        assert np.array_equal(frame["sws"], np.maximum.reduce(per_step["sws"]))
        assert np.array_equal(frame["svs"], np.maximum.reduce(per_step["svs"]))
        assert np.array_equal(frame["sps"], np.minimum.reduce(per_step["sps"]))
    # No ground twin: not allocated, not published, not priced.
    assert tornado.swaths.ground == {}
    published = tornado.published_carriers()
    assert not any(key.startswith("ground.") for key in published)
    names = [row.name for row in tornado.carrier_table()]
    assert names == ["sws", "svs", "sps"]
    assert not any(name.endswith("_ground") for name in host.registry.buffers)
    record = tornado.record()["swaths"]
    assert record["ground_slot"] is False and "ground_slot_omitted" in record
    profile = cartesian_history_profile(child, tornado_carriers=tornado.carrier_table())
    carried = {row.name for row in profile.carriers}
    assert {"sws", "svs", "sps"} <= carried and not {"sws2", "svs2", "sps2"} & carried
    # The parent's own swaths still keep their twin.
    parent_host = _host_runtime(_runtime(tmp_path, _SWATHS_ONLY, "p.toml"))
    assert CartesianTornadoDiagnostics(parent_host, total_steps=6).swaths.ground


# ---------------------------------------------------------------------------
# 3. Opening the child's swaths, fresh and restored.
# ---------------------------------------------------------------------------
def _fake_driver(child_config, *, restored=None):
    child = _host_runtime(child_config, seed=2)
    return SimpleNamespace(child=child, q=5, restored_swaths=restored)


def test_the_child_swaths_open_with_the_runs_child_step_count(tmp_path) -> None:
    child_config = _child(_runtime(tmp_path, _SWATHS_ONLY, "three.toml"))
    driver = _fake_driver(child_config)
    tornado = NestDriver.open_child_swaths(driver, parent_total_steps=4)
    assert driver.child.tornado is tornado
    assert tornado.anchor.total_steps == 20 and tornado.ground is False
    for name in ("sws", "svs", "sps"):
        assert np.all(tornado.swaths.frame[name] == np.float32(SWATH_INITIAL_VALUES[name]))
    with pytest.raises(RuntimeError, match="already open"):
        NestDriver.open_child_swaths(driver, parent_total_steps=4)
    off = _fake_driver(_child(_runtime(tmp_path, "", "none.toml")))
    assert NestDriver.open_child_swaths(off, parent_total_steps=4) is None
    assert off.child.tornado is None


def test_a_restored_child_continues_its_swaths_word_for_word(tmp_path) -> None:
    child_config = _child(_runtime(tmp_path, _SWATHS_ONLY, "three.toml"))
    rng = np.random.default_rng(9)
    shape = child_config.grid.surface_shape
    carried = {name: rng.standard_normal(shape).astype(np.float32) for name in ("sws", "svs", "sps")}
    driver = _fake_driver(child_config, restored={k: v.copy() for k, v in carried.items()})
    tornado = NestDriver.open_child_swaths(driver, parent_total_steps=4)
    assert driver.restored_swaths is None
    for name, words in carried.items():
        assert tornado.swaths.frame[name].tobytes() == words.tobytes()
    assert tornado.record()["swaths"]["restored_from_restart"] is True
    # A fold after the resume keeps the carried extremum where it is larger.
    driver.child.clock.domainlocx_m = driver.child.clock.domainlocy_m = 0.0
    tornado.update(1)
    assert np.all(tornado.swaths.frame["sws"] >= carried["sws"])
    assert np.all(tornado.swaths.frame["sps"] <= carried["sps"])

    missing = _fake_driver(child_config, restored={"sws": carried["sws"]})
    with pytest.raises(ValueError, match="sentinel"):
        NestDriver.open_child_swaths(missing, parent_total_steps=4)
    wrong = _fake_driver(child_config, restored={
        name: np.zeros((3, 3), np.float32) for name in ("sws", "svs", "sps")})
    with pytest.raises(ValueError, match="extent"):
        NestDriver.open_child_swaths(wrong, parent_total_steps=4)
    late = CartesianTornadoDiagnostics(_host_runtime(child_config), total_steps=4, ground=False)
    late.update(1)
    with pytest.raises(RuntimeError, match="already folded"):
        late.restore_swaths(carried)


def test_a_resumed_child_opens_its_swaths_at_its_restart_step(tmp_path) -> None:
    """A resumed child's swaths open at its own restart step and fold from the next one.

    Since restart profile v2 the fold refuses a skipped or repeated step, so a resumed child
    whose swaths opened at step 0 would refuse its first fold.  NestDriver
    aligns the child's clock with the restored parent's (parent step x q)
    when it is built, before the integration opens the swaths, and the
    swaths open at that step: slots before it stay unwritten.
    """

    import math

    child_config = _child(_runtime(tmp_path, _SWATHS_ONLY, "resumed.toml"))
    rng = np.random.default_rng(11)
    shape = child_config.grid.surface_shape
    carried = {name: rng.standard_normal(shape).astype(np.float32) for name in ("sws", "svs", "sps")}
    driver = _fake_driver(child_config, restored={k: v.copy() for k, v in carried.items()})
    driver.child.clock.completed_steps = 3 * driver.q  # the restored parent stands at step 3
    tornado = NestDriver.open_child_swaths(driver, parent_total_steps=4)
    swaths = tornado.swaths
    assert (swaths.opened_step, swaths.last_step, swaths.anchor.total_steps) == (15, 15, 20)
    for name, words in carried.items():
        assert swaths.frame[name].tobytes() == words.tobytes()
    with pytest.raises(ValueError, match="stand at step 15 and were asked to fold step 15"):
        tornado.update(15)
    tornado.update(16)
    assert np.all(swaths.frame["sws"] >= carried["sws"])
    assert np.all(swaths.frame["sps"] <= carried["sps"])
    series = swaths.host_series()["sws_max"]
    assert len(series) == 16 and all(math.isnan(value) for value in series[:15])
    assert not math.isnan(series[15])
    with pytest.raises(ValueError, match="folds every large step"):
        tornado.update(18)
    row = tornado.record()["swaths"]
    # The child restart carries the planes, not the step their lineage began at.
    assert row["first_step"] == {"sws": None, "svs": None, "sps": None}
    assert row["restored_from_restart"] is True and row["opened_step"] == 15


def test_the_child_restart_prices_one_plane_per_kept_swath(tmp_path) -> None:
    child_config = _child(_runtime(tmp_path, _SWATHS_ONLY, "three.toml"))
    profile = nest_child_restart_profile(child_config)
    rows = {c.name: c for c in profile.carriers if c.name.startswith("nest.swath.")}
    assert list(rows) == [SWATH_CARRIER.format(name=n) for n in ("sws", "svs", "sps")]
    for name, row in zip(("sws", "svs", "sps"), rows.values()):
        assert row.shape == tuple(child_config.grid.surface_shape)
        assert (row.group, row.key, row.stagger) == ("nest", f"swath.{name}", "surface")
    bare = nest_child_restart_profile(_child(_runtime(tmp_path, "", "none.toml")))
    assert not any(c.name.startswith("nest.swath.") for c in bare.carriers)
    assert bare.payload_nbytes < profile.payload_nbytes


# ---------------------------------------------------------------------------
# 4. The move.
# ---------------------------------------------------------------------------
def test_every_child_swath_buffer_has_a_move_rule(tmp_path) -> None:
    child_config = _child(_runtime(tmp_path, _DIAGNOSTICS_BLOCK, "all.toml"))
    host = _host_runtime(child_config)
    tornado = CartesianTornadoDiagnostics(host, total_steps=4, ground=False)
    tornado.update(1)
    tornado.refresh(1)
    shapes = {name: tuple(value.shape) for name, value in host.registry.buffers.items()}
    shifted, kept, unknown = nm.classify_carriers(shapes, grid=child_config.grid)
    assert unknown == []
    moved = {rule.name for rule in shifted}
    assert moved == {f"tornado.swath.{name}" for name in SWATH_NAMES}
    assert all(rule.plane and rule.stagger == "mass" for rule in shifted)
    assert kept["tornado.prs0"].startswith("invariant")
    assert kept["tornado.uh"].startswith("rebuilt")
    assert kept["tornado.swath.series.sws_max"].startswith("per-step series")
    # A ground twin on a child has no rule, so it stops the move by name.
    _, _, unknown = nm.classify_carriers(
        {**shapes, "tornado.swath.sws_ground": (9, 9)}, grid=child_config.grid)
    assert unknown == ["tornado.swath.sws_ground"]


@pytest.fixture(scope="module")
def move_harness():
    from tools.oracle.nest_move_oracle import open_harness

    return open_harness()


@pytest.fixture(scope="module")
def live_harness():
    from tools.oracle.nest_live_oracle import open_harness

    return open_harness()


@requires_gxx
@pytest.mark.parametrize("move", ((1, 0), (0, -1), (-1, 1)))
def test_a_move_keeps_every_swath_word_over_ground_the_child_still_covers(
    move_harness, live_harness, monkeypatch, move
) -> None:
    """Interior words are the child's own, shifted; the strip is the parent's swath."""

    import cm1core.nest_live as nest_live
    from cm1core.nest_live_kernels import reference_capture
    from tools.oracle.nest_move_oracle import GxxModule

    from test_nest_move import RATIO, _geometry, _settings, _small_driver, _Stream

    rng = np.random.default_rng(21)
    g = _geometry()
    driver, _ = _small_driver(rng, g)
    driver.module = GxxModule(live=live_harness)
    monkeypatch.setattr(nest_live, "_refill_halos_and_thermodynamics", lambda d, **kw: None)
    cg, pg = driver.child.config.grid, driver.parent.config.grid
    for name, sign in (("sws", 1.0), ("sps", -1.0)):
        driver.child.registry.buffers[f"tornado.swath.{name}"] = (
            sign * rng.uniform(10, 90, size=cg.surface_shape)).astype(np.float32)
        driver.parent.registry.buffers[f"tornado.swath.{name}"] = (
            sign * rng.uniform(10, 90, size=pg.surface_shape)).astype(np.float32)
    before = {n: driver.child.registry.buffers[f"tornado.swath.{n}"].copy() for n in ("sws", "sps")}
    mover = nm.MovingNest(settings=_settings(tracker="cm1-pressure-centre", smoothing_radius_m=None),
                          cupy=np, stream=_Stream(), refuse=refuse,
                          module=GxxModule(move=move_harness))
    mi, mj = move
    new = mover.execute(driver, move_i=mi, move_j=mj)
    assert "tornado.swath.sws" in mover.shifted_carriers
    ny, nx = cg.surface_shape
    si, sj = RATIO * mi, RATIO * mj
    for name in ("sws", "sps"):
        after = driver.child.registry.buffers[f"tornado.swath.{name}"]
        strip = reference_capture(
            driver.parent.registry.buffers[f"tornado.swath.{name}"].reshape(1, *pg.surface_shape),
            child_lattice=(1, ny, nx), child_volume=None, side="full", width=5, ratio=RATIO,
            parent_i0=new.parent_i0, parent_j0=new.parent_j0, stagger="mass", parent_dt_s=0.5,
        )[0][0]
        covered = 0
        for j in range(ny):
            for i in range(nx):
                oj, oi = j + sj, i + si
                if 0 <= oj < ny and 0 <= oi < nx:
                    # The same ground: the word is the child's own, unchanged,
                    # so a swath read at a fixed ground point never decreases.
                    assert after[j, i] == before[name][oj, oi]
                    covered += 1
                else:
                    assert after[j, i] == strip[j, i]
        assert 0 < covered < ny * nx


def test_the_output_switch_names_are_the_ones_the_child_filters() -> None:
    # child_swath_selection builds a selection from the swath switches alone;
    # every other switch is the dataclass default, off.
    selection = child_swath_selection(TornadoDiagnosticSelection(**{
        name: 1 for name in OUTPUT_SWITCH_NAMES}, tornado_report=False))
    on = {name for name in OUTPUT_SWITCH_NAMES if getattr(selection, name)}
    assert on == {f"output_{name}" for name in SWATH_NAMES}


def test_the_rain_and_hail_swaths_follow_cm1s_rows_not_one_scheme(tmp_path) -> None:
    """solve3.F:993-1047 folds qa(i,j,1,nqr) and qa(i,j,1,nqg) for any scheme with those rows (param.F:4727-4728
    turns them off where nqr or nqg is 0). Every component GM1 binds carries CM1's six mass rows first, so Morrison
    keeps both swaths as WSM6 does; a component with no transcribed axis is still refused by name."""

    from cm1core.moisture_species import MOISTURE_COMPONENT_IDS, moisture_species_profile
    from cm1core.tornado_diagnostics import (GRAUPEL_INDEX, RAINWATER_INDEX,
                                             validate_tornado_diagnostic_selection)

    parent = _runtime(tmp_path, _DIAGNOSTICS_BLOCK, "all.toml")
    assert parent.diagnostics.output_srs and parent.diagnostics.output_sgs
    for component in MOISTURE_COMPONENT_IDS:
        names = moisture_species_profile(component).mass_names
        assert (names[RAINWATER_INDEX], names[GRAUPEL_INDEX]) == ("qr", "qg"), component
        validate_tornado_diagnostic_selection(parent.diagnostics, grid=parent.grid, microphysics=component,
                                              bbc=parent.bbc)
    with pytest.raises(ValueError, match="output_srs/output_sgs"):
        validate_tornado_diagnostic_selection(parent.diagnostics, grid=parent.grid, microphysics="kessler-like",
                                              bbc=parent.bbc)
