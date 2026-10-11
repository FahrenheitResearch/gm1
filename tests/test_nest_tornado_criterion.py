"""The fixed nest tornado criterion, held to its host reference and to its own rules.

No test here opens a CUDA context unless ``GM1_NEST_DEVICE_TESTS=1`` asks for
one by name.  The packaged kernel text is compiled with ``g++``
(``tools/oracle/nest_tornado_criterion_oracle.py``) and its raw float64 words
are compared with :func:`cm1core.nest_tornado_criterion.reference_sample`; the
orchestration (:class:`NestTornadoCriterion`) then runs unchanged on NumPy
arrays through a RawModule stand-in, over synthetic Rankine vortices placed
on a 20 m child.  CM1 r21.1 has no nesting and makes no tornado claim: what is
traced to CM1 is the formulas a sample reads (``solve3.F:864-865``
ground-relative speed, ``parcel.F:1066-1067`` corner vorticity), checked
against closed forms below.

The live sampling path itself (CuPy words on a stream, ``words.get(stream=)``)
runs on the host through ``tools/oracle/host_cupy.py``, whose ``RawModule``
compiles the same packaged kernel text.

Device verification, on a node with a CUDA device::

    GM1_NEST_DEVICE_TESTS=1 CM1_R21_1_PRISTINE=<pristine> nice -n 10 \\
        python3 -m pytest -q -p no:cacheprovider tests/test_nest_tornado_criterion.py -k device
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

numpy = pytest.importorskip("numpy")

from cm1core import nest_tornado_criterion as tc  # noqa: E402
from cm1core.config import NestTornadoCriterionConfig, load_config, parse_mapping  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.nest_geometry import NestGeometry  # noqa: E402
from test_cartesian_restart import forecast_publisher  # noqa: E402,F401 - the session fixture

REPO = Path(__file__).resolve().parents[1]
requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
F32 = numpy.float32
NG = tc.NG
#: The child's frame velocity (umove, vmove) in every driven test.
FRAME = (8.0, 6.0)
COLUMN = {name: n for n, name in enumerate(tc.TIME_SERIES_COLUMNS)}


@pytest.fixture(scope="module")
def harness():
    from tools.oracle.nest_tornado_criterion_oracle import open_harness

    return open_harness()


def _plan(**overrides) -> tc.SamplePlan:
    values = dict(nx=37, ny=31, nz=4, dx_m=20.0, dy_m=20.0, ku0=1, ku1=2, w0=0.5, w1=0.5,
                  radius_m=90.0, u_move_m_s=8.5, v_move_m_s=6.0, lo_i=5, hi_i=32, lo_j=5, hi_j=26)
    values.update(overrides)
    return tc.SamplePlan(**values)


def _random_winds(rng, plan: tc.SamplePlan):
    shapes = plan.owner_shapes()
    return (rng.normal(0.0, 10.0, shapes["u"]).astype(F32),
            rng.normal(0.0, 10.0, shapes["v"]).astype(F32))


def _column(row, name):
    return row[COLUMN[name]]


# ---------------------------------------------------------------------------
# Rankine vortices on a padded C grid.
# ---------------------------------------------------------------------------
def _rankine(r, vmax: float, rc: float):
    """Solid-body rotation inside ``rc``, a 1/r potential vortex outside: peak ``vmax`` at ``rc``."""

    return numpy.where(r <= rc, vmax * r / rc, vmax * rc / numpy.maximum(r, 1e-12))


def _padded_winds(nx: int, ny: int, nz: int, dx: float, vortices, background):
    """Padded ``u3d``/``v3d`` of Rankine vortices ``(x, y, vmax, rc)`` in child metres.

    Every padded word, ghosts included, is the analytic wind at its own face,
    and every level holds the same plane.
    """

    xu = (numpy.arange(nx + 1 + 2 * NG) - NG) * dx
    yu = (numpy.arange(ny + 2 * NG) - NG + 0.5) * dx
    xv = (numpy.arange(nx + 2 * NG) - NG + 0.5) * dx
    yv = (numpy.arange(ny + 1 + 2 * NG) - NG) * dx
    uu = numpy.full((yu.size, xu.size), float(background[0]))
    vv = numpy.full((yv.size, xv.size), float(background[1]))
    for xc, yc, vmax, rc in vortices:
        rx, ry = xu[None, :] - xc, yu[:, None] - yc
        r = numpy.hypot(rx, ry)
        uu += -_rankine(r, vmax, rc) * ry / numpy.maximum(r, 1e-12)
        rx, ry = xv[None, :] - xc, yv[:, None] - yc
        r = numpy.hypot(rx, ry)
        vv += _rankine(r, vmax, rc) * rx / numpy.maximum(r, 1e-12)
    u = numpy.empty((nz + 2,) + uu.shape, dtype=F32)
    v = numpy.empty((nz + 2,) + vv.shape, dtype=F32)
    u[:] = uu.astype(F32)
    v[:] = vv.astype(F32)
    return u, v


class _Driver:
    """The attributes NestTornadoCriterion reads from a NestDriver, on host arrays.

    A 200 x 200, 20 m child (4 km) on a 100 m parent, whose frame moves at
    :data:`FRAME`; ``zh(1) = dz/2``.
    """

    def __init__(self, *, dz: float = 20.0, frame=FRAME, vertical=None, nz: int = 4) -> None:
        self.geometry = NestGeometry(
            parent_nx=100, parent_ny=100, parent_dx_m=100.0, parent_dy_m=100.0,
            refinement_ratio=5, time_refinement_ratio=5, parent_i0=30, parent_j0=30,
            parent_span_x=40, parent_span_y=40, spec_zone=1, relax_zone=4, feedback=1,
        )
        g = self.geometry
        self.frame = tuple(float(f) for f in frame)
        grid = SimpleNamespace(nx=g.nx, ny=g.ny, nz=nz, dx_m=g.dx_m, dy_m=g.dy_m, dz_m=dz,
                               vertical=vertical)
        shapes = _plan(nx=g.nx, ny=g.ny, nz=nz).owner_shapes()
        self.child = SimpleNamespace(
            config=SimpleNamespace(grid=grid, u_move_m_s=self.frame[0], v_move_m_s=self.frame[1]),
            dynamics=SimpleNamespace(arrays={"u3d": numpy.zeros(shapes["u"], dtype=F32),
                                             "v3d": numpy.zeros(shapes["v"], dtype=F32)}),
        )
        self.clock = SimpleNamespace(completed_steps=0, model_seconds=0.0, domainlocx_m=0.0,
                                     domainlocy_m=0.0)
        self.parent = SimpleNamespace(clock=self.clock)

    def at(self, step: int, dt: float) -> float:
        t = step * dt
        self.clock.completed_steps = step
        self.clock.model_seconds = t
        self.clock.domainlocx_m = self.frame[0] * t
        self.clock.domainlocy_m = self.frame[1] * t
        return t

    def local(self, ground_x: float, ground_y: float) -> tuple[float, float]:
        g = self.geometry
        return (ground_x - self.clock.domainlocx_m - g.origin_x_m,
                ground_y - self.clock.domainlocy_m - g.origin_y_m)

    def place(self, vortices, *, ground_background=(0.0, 0.0)) -> None:
        """Rankine vortices at ground positions, in a ground-relative background wind."""

        g, c = self.geometry, self.child
        local = [(*self.local(x, y), vmax, rc) for x, y, vmax, rc in vortices]
        background = (ground_background[0] - self.frame[0], ground_background[1] - self.frame[1])
        u, v = _padded_winds(g.nx, g.ny, c.config.grid.nz, g.dx_m, local, background)
        c.dynamics.arrays["u3d"][:] = u
        c.dynamics.arrays["v3d"][:] = v


def _criterion(driver: _Driver, settings=None, *, module=None, root: Path | None = None):
    from tools.oracle.nest_tornado_criterion_oracle import ReferenceModule

    criterion = tc.NestTornadoCriterion(
        settings=NestTornadoCriterionConfig() if settings is None else settings,
        driver=driver, xp=numpy, stream=None,
        module=ReferenceModule() if module is None else module,
    )
    if root is not None:
        criterion.open(root)
    return criterion


# ---------------------------------------------------------------------------
# 1. The kernel text is the host reference, word for word.
# ---------------------------------------------------------------------------
@requires_gxx
@pytest.mark.parametrize("seed", range(4))
def test_the_kernel_text_is_the_host_reference_word_for_word(harness, seed) -> None:
    from tools.oracle.nest_tornado_criterion_oracle import GxxModule

    rng = numpy.random.default_rng(seed)
    plan = _plan(
        ku0=[1, 1, 2, 3][seed], ku1=[2, 2, 3, 4][seed],
        w0=[0.5, 1.0, 0.25, 0.7][seed], w1=[0.5, 0.0, 0.75, 0.3][seed],
        radius_m=[90.0, 45.0, 21.0, 700.0][seed],
        # seed 3's disk and box reach every edge: the clamps and the ghost rows.
        lo_i=[5, 5, 5, 0][seed], hi_i=[32, 32, 32, 37][seed],
        lo_j=[5, 5, 5, 0][seed], hi_j=[26, 26, 26, 31][seed],
    )
    u, v = _random_winds(rng, plan)
    if seed == 1:
        u[1, 10, 12] = numpy.nan  # a non-finite corner is skipped, not selected
        v[2, 20, 20] = numpy.inf
    module = GxxModule(harness)
    out = numpy.full(tc.SAMPLE_WORDS, 7.0)
    tc.launch_tornado_sample(numpy, module, u=u, v=v, out=out, plan=plan)
    expected = tc.reference_sample(u, v, plan=plan)
    assert out.tobytes() == expected.tobytes()
    assert module.calls == 1 and out[0] == 1.0 and out[7] > 0 and out[8] > 0


@requires_gxx
def test_ties_and_an_all_nonfinite_search_follow_the_raster_rule(harness) -> None:
    from tools.oracle.nest_tornado_criterion_oracle import GxxModule

    module = GxxModule(harness)
    plan = _plan()
    shapes = plan.owner_shapes()
    u = numpy.zeros(shapes["u"], dtype=F32)
    v = numpy.zeros(shapes["v"], dtype=F32)
    out = numpy.zeros(tc.SAMPLE_WORDS)
    tc.launch_tornado_sample(numpy, module, u=u, v=v, out=out, plan=plan)
    # A flat field: the first corner of the search box in DO j / DO i order.
    assert list(out[:4]) == [1.0, float(plan.lo_i), float(plan.lo_j), 0.0]
    assert out.tobytes() == tc.reference_sample(u, v, plan=plan).tobytes()
    u[:] = numpy.nan
    tc.launch_tornado_sample(numpy, module, u=u, v=v, out=out, plan=plan)
    assert out[0] == 0.0 and list(out[1:3]) == [-1.0, -1.0]
    assert out.tobytes() == tc.reference_sample(u, v, plan=plan).tobytes()


@requires_gxx
def test_the_orchestration_launches_the_kernel_text_on_a_rankine_vortex(harness) -> None:
    from tools.oracle.nest_tornado_criterion_oracle import GxxModule

    driver = _Driver(dz=10.0)  # zh = 5, 15: the 10 m winds are the two levels' mean
    module = GxxModule(harness)
    criterion = _criterion(driver, module=module)
    assert (criterion.plan.ku0, criterion.plan.ku1, criterion.plan.w0, criterion.plan.w1) == (
        1, 2, 0.5, 0.5)
    driver.at(7, 1.0)
    g = driver.geometry
    driver.place([(g.origin_x_m + 2013.0 + 56.0, g.origin_y_m + 1987.0 + 42.0, 45.0, 80.0)])
    driver.child.dynamics.arrays["u3d"][2] += F32(3.0)  # level 2 differs from level 1
    row = criterion.after_parent_step(driver)
    assert module.calls == 1 and criterion.launches == 1
    words = tc.reference_sample(driver.child.dynamics.arrays["u3d"],
                                driver.child.dynamics.arrays["v3d"], plan=criterion.plan)
    assert _column(row, "centre_i") == int(words[1]) and _column(row, "centre_j") == int(words[2])
    assert _column(row, "zeta_max_s_inv") == words[3]
    assert _column(row, "ground_relative_wind_max_m_s") == words[4]
    assert _column(row, "frame_relative_wind_max_m_s") == words[5]
    assert _column(row, "core_corners") == int(words[6])
    assert _column(row, "conditions") == "ZWCM" and _column(row, "state") == "run"


def test_the_launcher_refuses_layouts_the_kernel_would_misread() -> None:
    from tools.oracle.nest_tornado_criterion_oracle import ReferenceModule

    plan = _plan()
    u, v = _random_winds(numpy.random.default_rng(0), plan)
    out = numpy.zeros(tc.SAMPLE_WORDS)
    module = ReferenceModule()
    with pytest.raises(ValueError, match="another layout"):
        tc.launch_tornado_sample(numpy, module, u=v, v=v, out=out, plan=plan)
    with pytest.raises(ValueError, match="another layout"):
        tc.launch_tornado_sample(numpy, module, u=u.astype(numpy.float64), v=v, out=out, plan=plan)
    with pytest.raises(ValueError, match="float64 words"):
        tc.launch_tornado_sample(numpy, module, u=u, v=v, out=numpy.zeros(3), plan=plan)
    with pytest.raises(ValueError, match="height bracket"):
        tc.launch_tornado_sample(numpy, module, u=u, v=v, out=out, plan=replace(plan, ku1=5))
    assert module.calls == 0


# ---------------------------------------------------------------------------
# 2. The reference reads CM1's formulas.
# ---------------------------------------------------------------------------
def test_the_reference_is_cm1s_corner_vorticity_and_ground_relative_speed() -> None:
    """parcel.F:1066-1067 and solve3.F:864-865 on solid-body rotation, interpolated in height."""

    plan = _plan(nx=40, ny=40, nz=3, w0=0.25, w1=0.75, radius_m=100.0,
                 lo_i=10, hi_i=30, lo_j=10, hi_j=30)
    shapes = plan.owner_shapes()
    # u = -a*(y - yc), v = a*(x - xc) about corner (20, 20): vorticity 2a at
    # every corner.  Level 1 turns at a1, level 2 at a2; the 10 m plane at
    # 0.25*a1 + 0.75*a2.
    a1, a2 = 0.125, 0.5
    xc = yc = 20 * plan.dx_m
    u = numpy.zeros(shapes["u"], dtype=F32)
    v = numpy.zeros(shapes["v"], dtype=F32)
    y_u = (numpy.arange(shapes["u"][1]) - NG + 0.5) * plan.dy_m
    x_v = (numpy.arange(shapes["v"][2]) - NG + 0.5) * plan.dx_m
    for k, a in ((1, a1), (2, a2)):
        u[k] = (-a * (y_u - yc))[:, None].astype(F32)
        v[k] = (a * (x_v - xc))[None, :].astype(F32)
    words = tc.reference_sample(u, v, plan=plan)
    a = 0.25 * a1 + 0.75 * a2
    assert words[0] == 1.0 and words[3] == 2 * a
    # Every corner ties, so the centre is the first corner of the search box.
    assert (words[1], words[2]) == (10.0, 10.0)
    zeta = tc.reference_corner_vorticity(u, v, plan)
    assert zeta.shape == (41, 41) and numpy.all(zeta == 2 * a)
    # Scalar speeds a*r at (i+0.5, j+0.5) over the 100 m disk about X = (200, 200).
    xs = (numpy.arange(plan.nx) + 0.5) * plan.dx_m
    X, Y = numpy.meshgrid(xs, xs)
    disk = (X - 200.0) ** 2 + (Y - 200.0) ** 2 <= 100.0 ** 2
    uf, vf = -a * (Y - yc), a * (X - xc)
    assert words[4] == pytest.approx(numpy.hypot(uf + 8.5, vf + 6.0)[disk].max(), abs=1e-5)
    assert words[5] == pytest.approx(numpy.hypot(uf, vf)[disk].max(), abs=1e-5)
    corners = (numpy.arange(41) * 20.0 - 200.0) ** 2
    in_disk = int(numpy.count_nonzero(corners[:, None] + corners[None, :] <= 100.0 ** 2))
    assert words[7] == float(numpy.count_nonzero(disk)) and words[8] == float(in_disk)
    # Uniform vorticity: every disk corner is at least half the maximum.
    assert words[6] == words[8]


def test_height_brackets_search_boxes_and_the_core_radius() -> None:
    from cm1core.cm1_vertical_grid import build_vertical_levels, uniform_vertical_levels

    assert tc.height_bracket(uniform_vertical_levels(10, 20.0), 10.0) == (1, 2, 1.0, 0.0)
    assert tc.height_bracket(uniform_vertical_levels(10, 10.0), 10.0) == (1, 2, 0.5, 0.5)
    assert tc.height_bracket(uniform_vertical_levels(10, 25.0), 10.0) is None
    # The 20 m child deck of the 100 m parent runs, with a 5 m lowest layer
    # (nest.child_dz_bot_m = 5).
    levels = build_vertical_levels(stretch_z=1, nk=102, dz_m=100.0, ztop_m=15000.0,
                                   str_bot_m=170.0, str_top_m=7320.0, dz_bot_m=5.0,
                                   dz_top_m=320.0)
    assert [float(z) for z in levels.zh[1:4]] == [2.5, 7.5, 12.5]
    assert tc.height_bracket(levels, 10.0) == (2, 3, 0.5, 0.5)
    assert tc.interior_corner_box(200, 20.0, 1000.0) == (50, 150)
    assert tc.interior_corner_box(450, 20.0, 1000.0) == (50, 400)
    lo, hi = tc.interior_corner_box(99, 20.0, 1000.0)
    assert lo > hi
    assert tc.interior_corner_box(10, 0.1, 0.3) == (3, 7)
    assert tc.core_radius_m(20.0, 0) == 0.0
    assert tc.core_radius_m(20.0, 13) == pytest.approx(20.0 * math.sqrt(13 / math.pi))
    # 40 m needs 13 corners at 20 m: 12 is 39.09 m.
    assert tc.core_radius_m(20.0, 12) < 40.0 <= tc.core_radius_m(20.0, 13)


# ---------------------------------------------------------------------------
# 3. Synthetic Rankine vortices through the live orchestration.
# ---------------------------------------------------------------------------
TRANSLATION = (12.0, 3.0)


def test_a_translating_rankine_vortex_is_a_tornado_once_it_qualifies_for_sixty_seconds(
    tmp_path,
) -> None:
    """45 m/s, 80 m core, moving (12, 3) m/s over the ground in an (8, 6) m/s frame.

    For its first 30 s the vortex turns at 15 m/s: 0.375 s-1 passes the
    vorticity bar, but the ground-relative wind (15 + 12.4) does not reach 35.
    From 30 s it turns at 45 m/s and every condition holds, so the run of
    qualifying samples starts at 30 s and becomes a tornado at 90 s.
    """

    driver = _Driver()
    criterion = _criterion(driver, root=tmp_path)
    g = driver.geometry
    x0, y0 = g.origin_x_m + 1503.0, g.origin_y_m + 2011.0
    for step in range(0, 151):
        t = driver.at(step, 1.0)
        vmax = 15.0 if t < 30.0 else 45.0
        gx, gy = x0 + TRANSLATION[0] * t, y0 + TRANSLATION[1] * t
        driver.place([(gx, gy, vmax, 80.0)], ground_background=TRANSLATION)
        row = criterion.after_parent_step(driver)
        # The centre is a corner of the core, on the ground where the vortex is.
        assert math.hypot(_column(row, "centre_ground_x_m") - gx,
                          _column(row, "centre_ground_y_m") - gy) <= 80.0
        if t < 30.0:
            assert _column(row, "conditions") == "Z-CM" and _column(row, "state") == "-"
            assert _column(row, "ground_relative_wind_max_m_s") < 35.0
            continue
        assert _column(row, "conditions") == "ZWCM", row
        assert _column(row, "run_span_s") == t - 30.0
        assert _column(row, "state") == ("run" if t < 90.0 else "episode")
        assert _column(row, "episodes") == (0 if t < 90.0 else 1)
        # The grid's peak of a Rankine wind: at most 45 m/s rotation plus the
        # translation (ground) or plus what the frame leaves of it (frame-relative);
        # the 20 m scalar points miss the core-edge cusp by up to a few m/s.
        ground = 45.0 + math.hypot(*TRANSLATION)
        frame = 45.0 + math.hypot(TRANSLATION[0] - FRAME[0], TRANSLATION[1] - FRAME[1])
        assert ground - 6.0 <= _column(row, "ground_relative_wind_max_m_s") <= ground + 1e-3
        assert frame - 6.0 <= _column(row, "frame_relative_wind_max_m_s") <= frame + 1e-3
        assert (_column(row, "ground_relative_wind_max_m_s")
                > _column(row, "frame_relative_wind_max_m_s") + 3.0)
        assert _column(row, "core_radius_m") == pytest.approx(80.0, abs=5.0)
    record = criterion.close()
    assert record["verdict"] == "TORNADO"
    assert record["genesis_model_time_s"] == 30.0
    assert record["tornado_declared_model_time_s"] == 90.0
    assert record["samples"] == 151 and record["device_launches"] == 151
    assert record["condition_counts"]["qualifying"] == 121
    assert record["condition_counts"]["ground_relative_wind"] == 121
    assert record["longest_qualifying_run_s"] == 120.0
    [episode] = record["episodes"]
    assert episode["in_progress_at_last_sample"] is True
    assert (episode["start_model_time_s"], episode["end_model_time_s"]) == (30.0, 150.0)
    velocity = episode["mean_centre_ground_velocity_m_s"]
    assert velocity == pytest.approx(TRANSLATION, abs=40.0 / 120.0 * 2)
    assert record["criterion"] == dict(tc.CRITERION)
    assert record["criterion_sha256"] == tc.CRITERION_SHA256
    # The sidecar: a header, then one compact row per parent step.
    [series] = record["time_series"]
    path = tmp_path / "nest-child" / tc.CRITERION_JSONL_FILENAME
    raw = path.read_bytes()
    assert series == {"path": os.fspath(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "rows": 151, "bytes": len(raw)}
    assert len(raw) / 151 < 400  # small: bytes per sample, header included
    header, rows = tc.read_time_series(path)
    assert header["columns"] == list(tc.TIME_SERIES_COLUMNS)
    assert header["plan"]["levels_cm1_indexed"] == [1, 2]
    assert len(rows) == 151 and rows[-1][COLUMN["episodes"]] == 1
    # The sidecar alone re-decides everything the receipt says.
    replay = tc.replay_time_series(path)
    assert replay["rows_identical"] is True and replay["mismatches"] == []
    for key in ("verdict", "genesis_model_time_s", "tornado_declared_model_time_s", "samples",
                "condition_counts", "longest_qualifying_run_s", "episodes", "episode_count"):
        assert replay[key] == record[key], key


def test_the_ground_relative_wind_qualifies_a_vortex_whose_frame_relative_wind_does_not() -> None:
    """solve3.F:864's umove/vmove: a 30 m/s Rankine vortex carried at the frame's 10 m/s."""

    for frame, verdict in ((FRAME, "TORNADO"), ((0.0, 0.0), "NO_TORNADO")):
        driver = _Driver(frame=frame)
        criterion = _criterion(driver)
        g = driver.geometry
        for step in range(0, 61):
            t = driver.at(step, 1.0)
            # At rest in the grid: carried over the ground at the frame velocity.
            driver.place([(g.origin_x_m + 2000.0 + frame[0] * t,
                           g.origin_y_m + 2000.0 + frame[1] * t, 30.0, 80.0)],
                         ground_background=frame)
            row = criterion.after_parent_step(driver)
            assert _column(row, "frame_relative_wind_max_m_s") < 30.0 + 1e-3
            assert _column(row, "centre_move_m") in (None, pytest.approx(math.hypot(*frame), abs=1e-6))
        record = criterion.close()
        assert record["verdict"] == verdict, frame
        wind = row[COLUMN["ground_relative_wind_max_m_s"]]
        if frame == FRAME:
            assert 35.0 <= wind <= 40.0 + 1e-3 and _column(row, "conditions") == "ZWCM"
        else:
            assert wind < 35.0 and _column(row, "conditions") == "Z-CM"


def test_a_rankine_core_under_forty_metres_is_not_a_tornado_however_strong() -> None:
    driver = _Driver()
    criterion = _criterion(driver)
    g = driver.geometry
    for step in range(0, 70):
        driver.at(step, 1.0)
        driver.place([(g.origin_x_m + 2000.0 + FRAME[0] * step,
                       g.origin_y_m + 2000.0 + FRAME[1] * step, 60.0, 30.0)],
                     ground_background=FRAME)
        row = criterion.after_parent_step(driver)
        assert _column(row, "zeta_max_s_inv") >= 3.0
        assert _column(row, "core_radius_m") < 40.0
        assert _column(row, "conditions") == "ZW-M"
    assert criterion.close()["verdict"] == "NO_TORNADO"


def test_a_weak_rankine_vortex_in_a_strong_wind_fails_on_vorticity() -> None:
    driver = _Driver()
    criterion = _criterion(driver)
    g = driver.geometry
    for step in range(0, 70):
        t = driver.at(step, 1.0)
        # 10 m/s over an 80 m core is 0.25 s-1; the 40 m/s ground wind is not a vortex.
        driver.place([(g.origin_x_m + 1500.0 + 40.0 * t, g.origin_y_m + 2000.0, 10.0, 80.0)],
                     ground_background=(40.0, 0.0))
        row = criterion.after_parent_step(driver)
        assert _column(row, "zeta_max_s_inv") < 0.30
        assert _column(row, "conditions")[:2] == "-W"
    record = criterion.close()
    assert record["verdict"] == "NO_TORNADO" and record["condition_counts"]["zeta"] == 0


def test_a_stronger_vortex_inside_the_edge_margin_is_never_the_centre() -> None:
    driver = _Driver()
    criterion = _criterion(driver)
    driver.at(1, 1.0)
    g = driver.geometry
    strong = (g.origin_x_m + FRAME[0] + 610.0, g.origin_y_m + FRAME[1] + 2000.0, 60.0, 60.0)
    weak = (g.origin_x_m + FRAME[0] + 2400.0, g.origin_y_m + FRAME[1] + 2300.0, 40.0, 80.0)
    driver.place([strong, weak])
    row = criterion.after_parent_step(driver)
    zeta = tc.reference_corner_vorticity(driver.child.dynamics.arrays["u3d"],
                                         driver.child.dynamics.arrays["v3d"], criterion.plan)
    assert zeta.max() >= 1.9 * _column(row, "zeta_max_s_inv")
    assert abs(_column(row, "centre_i") * 20.0 - 2400.0) <= 80.0
    assert abs(_column(row, "centre_j") * 20.0 - 2300.0) <= 80.0
    assert (criterion.plan.lo_i, criterion.plan.hi_i) == (50, 150)


def test_a_jump_to_another_rankine_vortex_ends_the_run() -> None:
    """Two tornado-strength vortices 1 km apart, trading the vorticity maximum every 40 s."""

    driver = _Driver()
    criterion = _criterion(driver)
    g = driver.geometry
    for step in range(0, 200):
        t = driver.at(step, 1.0)
        first_leads = t < 40.0 or 80.0 <= t < 120.0
        a = (g.origin_x_m + 1500.0 + FRAME[0] * t, g.origin_y_m + 2000.0 + FRAME[1] * t,
             50.0 if first_leads else 45.0, 80.0)
        b = (g.origin_x_m + 2500.0 + FRAME[0] * t, g.origin_y_m + 2000.0 + FRAME[1] * t,
             45.0 if first_leads else 50.0, 80.0)
        driver.place([a, b], ground_background=FRAME)
        row = criterion.after_parent_step(driver)
        if step in (40, 80, 120):
            assert _column(row, "conditions") == "ZWC-"
            assert _column(row, "centre_move_m") > 900.0
    record = criterion.close()
    # Runs 0-39, 41-79, 81-119 never span 60 s; 121-199 does, from 121 s.
    assert record["qualifying_runs_started"] == 4
    assert record["verdict"] == "TORNADO" and record["genesis_model_time_s"] == 121.0
    assert record["tornado_declared_model_time_s"] == 181.0


# ---------------------------------------------------------------------------
# 4. The ledger's rules at their boundaries.
# ---------------------------------------------------------------------------
def _sample(t, *, x=0.0, y=0.0, zeta=1.0, wind=40.0, corners=20, found=True, step=None):
    return tc.CriterionSample(
        step=int(round(t * 2)) if step is None else step, model_time_s=float(t), found=found,
        centre_i=0 if found else -1, centre_j=0 if found else -1,
        ground_x_m=x if found else math.nan, ground_y_m=y if found else math.nan,
        zeta_max_s_inv=zeta if found else math.nan, ground_relative_wind_m_s=wind,
        frame_relative_wind_m_s=wind - 5.0, core_corners=corners,
        core_radius_m=tc.core_radius_m(20.0, corners),
    )


def _times(start, stop, dt=0.5):
    return [start + i * dt for i in range(int(round((stop - start) / dt)) + 1)]


def test_every_threshold_is_inclusive_at_its_fixed_value() -> None:
    ledger = tc.CriterionLedger()
    assert ledger.observe(_sample(0.0, zeta=0.30, wind=35.0, corners=13))[0] == "ZWCM"
    ledger = tc.CriterionLedger()
    assert ledger.observe(_sample(0.0, zeta=math.nextafter(0.30, 0.0)))[0] == "-WCM"
    assert ledger.observe(_sample(0.5, wind=math.nextafter(35.0, 0.0)))[0] == "Z-CM"
    assert ledger.observe(_sample(1.0, corners=12))[0] == "ZW-M"
    # 60 m/s * 0.5 s + 40 m = 70 m exactly passes; a hair more fails.
    ledger = tc.CriterionLedger()
    ledger.observe(_sample(0.0))
    conditions, move, state, span = ledger.observe(_sample(0.5, x=70.0))
    assert (conditions, move, state, span) == ("ZWCM", 70.0, "run", 0.5)
    conditions, move, _, _ = ledger.observe(_sample(1.0, x=140.0 + 1e-9))
    assert conditions == "ZWC-" and move > 70.0
    # The limit scales with the model time between samples.
    ledger = tc.CriterionLedger()
    ledger.observe(_sample(0.0))
    assert ledger.observe(_sample(2.0, y=160.0))[0] == "ZWCM"


def test_a_non_finite_maximum_fails_its_condition() -> None:
    """The sidecar stores a non-finite word as null; the live ledger decides what a replay can."""

    for value in (math.inf, -math.inf, math.nan):
        ledger = tc.CriterionLedger()
        assert ledger.observe(_sample(0.0, wind=value))[0] == "Z-CM"
        assert ledger.observe(_sample(0.5, zeta=value))[0] == "-WCM"
        assert ledger.record()["condition_counts"]["qualifying"] == 0


def test_sixty_seconds_is_first_to_last_sample_and_any_failure_restarts_it() -> None:
    ledger = tc.CriterionLedger()
    for t in _times(0.0, 59.5):
        ledger.observe(_sample(t))
    assert not ledger.tornado and ledger.longest_run_s == 59.5
    assert ledger.observe(_sample(60.0))[2] == "episode"
    assert ledger.tornado and ledger.genesis_s == 0.0 and ledger.declared_s == 60.0
    # One failing sample ends the episode at its last qualifying sample.
    ledger = tc.CriterionLedger()
    for t in _times(0.0, 100.0):
        ledger.observe(_sample(t))
    ledger.observe(_sample(100.5, wind=1.0))
    for t in _times(101.0, 130.0):
        ledger.observe(_sample(t))
    ledger.observe(_sample(130.5, zeta=0.1))
    for t in _times(131.0, 200.0):
        ledger.observe(_sample(t))
    record = ledger.record()
    assert record["episode_count"] == 2 and record["qualifying_runs_started"] == 3
    assert [(e["start_model_time_s"], e["end_model_time_s"]) for e in record["episodes"]] == [
        (0.0, 100.0), (131.0, 200.0)]
    assert "in_progress_at_last_sample" in record["episodes"][1]
    assert record["genesis_model_time_s"] == 0.0 and ledger.last_episode_end_s == 100.0
    # A run of samples 1/3 s apart reaches 60 s within the clock's own tolerance.
    ledger = tc.CriterionLedger()
    t = 0.0
    for _ in range(180):
        ledger.observe(_sample(t))
        t += 1.0 / 3.0
    ledger.observe(_sample(t))
    assert ledger.tornado and abs(ledger.declared_s - 60.0) < 1e-9


def test_a_sample_without_a_centre_breaks_the_run_and_the_next_move() -> None:
    ledger = tc.CriterionLedger()
    first = ledger.observe(_sample(0.0))
    assert first == ("ZWCM", None, "run", 0.0)  # the first sample has no previous centre
    ledger.observe(_sample(0.5))
    assert ledger.observe(_sample(1.0, found=False)) == ("----", None, "-", None)
    assert ledger.observe(_sample(1.5))[0] == "ZWC-"
    assert ledger.observe(_sample(2.0)) == ("ZWCM", 0.0, "run", 0.0)
    assert ledger.record()["condition_counts"]["no_centre"] == 1


def test_stop_after_episode_waits_its_interval_and_for_any_run_in_progress() -> None:
    rules = NestTornadoCriterionConfig(stop_after_episode_s=300.0)
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 100.0):
        ledger.observe(_sample(t))
    for t in _times(100.5, 399.5):
        ledger.observe(_sample(t, wind=20.0))
        assert ledger.stop_request is None, t
    ledger.observe(_sample(400.0, wind=20.0))
    assert ledger.stop_request == {
        "rule": "stop_after_episode_s", "threshold_s": 300.0,
        "last_episode_end_model_time_s": 100.0, "requested_at_step": 800,
        "requested_at_model_time_s": 400.0,
    }
    # A qualifying run in progress holds the request off, however short.
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 100.0):
        ledger.observe(_sample(t))
    for t in _times(100.5, 399.5):
        ledger.observe(_sample(t, wind=20.0))
    ledger.observe(_sample(400.0))
    assert ledger.stop_request is None
    ledger.observe(_sample(400.5, wind=20.0))
    assert ledger.stop_request["requested_at_model_time_s"] == 400.5
    # A run that never spanned 60 s is no episode and starts no interval.
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 30.0):
        ledger.observe(_sample(t))
    for t in _times(30.5, 2000.0, 10.0):
        ledger.observe(_sample(t, wind=20.0))
    assert ledger.stop_request is None and not ledger.tornado


def test_stop_if_no_tornado_by_fires_at_its_time_and_never_after_a_tornado() -> None:
    rules = NestTornadoCriterionConfig(stop_if_no_tornado_by_s=5400.0)
    ledger = tc.CriterionLedger(rules)
    for t in _times(5300.0, 5399.5):
        ledger.observe(_sample(t, wind=34.9))
        assert ledger.stop_request is None
    ledger.observe(_sample(5400.0, wind=34.9))
    assert ledger.stop_request["rule"] == "stop_if_no_tornado_by_s"
    assert ledger.stop_request["requested_at_step"] == 10800
    # A qualifying run in progress at the cutoff is given the chance to become one.
    ledger = tc.CriterionLedger(rules)
    for t in _times(5380.0, 5460.0):
        ledger.observe(_sample(t))
    assert ledger.stop_request is None and ledger.tornado
    for t in _times(5460.5, 9000.0, 10.0):
        ledger.observe(_sample(t, wind=1.0))
    assert ledger.stop_request is None
    # A run that ends before 60 s releases the request at the next sample.
    ledger = tc.CriterionLedger(rules)
    for t in _times(5390.0, 5420.0):
        ledger.observe(_sample(t))
    assert ledger.stop_request is None
    ledger.observe(_sample(5420.5, wind=1.0))
    assert ledger.stop_request["requested_at_model_time_s"] == 5420.5
    # Absent rules never request.
    ledger = tc.CriterionLedger(NestTornadoCriterionConfig())
    for t in _times(0.0, 100.0):
        ledger.observe(_sample(t))
    for t in _times(100.5, 9000.0, 10.0):
        ledger.observe(_sample(t, wind=1.0))
    assert ledger.stop_request is None


def test_a_pending_request_is_withdrawn_when_a_qualifying_run_begins() -> None:
    """Request at the cutoff, a vortex one sample later: the run is not cut at the next frame."""

    rules = NestTornadoCriterionConfig(stop_if_no_tornado_by_s=10.0)
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 10.0):
        ledger.observe(_sample(t, wind=20.0))
    request = ledger.stop_request
    assert request["requested_at_model_time_s"] == 10.0
    assert ledger.observe(_sample(10.5))[2] == "run"
    assert ledger.stop_request is None
    record = ledger.record()
    assert record["stop_requests_withdrawn"] == 1
    assert record["last_withdrawn_stop_request"] == {
        **request, "withdrawn_at_step": 21, "withdrawn_at_model_time_s": 10.5,
        "withdrawn_because": tc.STOP_WITHDRAWN_BY_RUN,
    }
    assert tc.STOP_WITHDRAWN_BY_RUN not in {tc.STOP_HONOURED, tc.STOP_UNUSED_AT_DURATION,
                                            tc.STOP_UNUSED_AT_COURANT}
    for t in _times(11.0, 70.5):
        ledger.observe(_sample(t))
        assert ledger.stop_request is None, t
    assert ledger.tornado and ledger.genesis_s == 10.5
    # After the tornado the cutoff never fires again, however quiet it gets.
    for t in _times(71.0, 300.0):
        ledger.observe(_sample(t, wind=1.0))
    assert ledger.stop_request is None
    assert ledger.record()["stop_requests_withdrawn"] == 1
    # A run that ends short of an episode re-raises the request at its first
    # failing sample, as a run in progress at the cutoff always did.
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 10.0):
        ledger.observe(_sample(t, wind=20.0))
    for t in _times(10.5, 30.0):
        ledger.observe(_sample(t))
        assert ledger.stop_request is None, t
    ledger.observe(_sample(30.5, wind=1.0))
    assert ledger.stop_request["requested_at_model_time_s"] == 30.5
    assert ledger.record()["stop_requests_withdrawn"] == 1
    # Every withdrawal is counted; the last is kept.
    ledger.observe(_sample(31.0))
    ledger.observe(_sample(31.5, wind=1.0))
    ledger.observe(_sample(32.0))
    record = ledger.record()
    assert record["stop_request"] is None and record["stop_requests_withdrawn"] == 3
    assert record["last_withdrawn_stop_request"]["requested_at_model_time_s"] == 31.5
    assert record["last_withdrawn_stop_request"]["withdrawn_at_model_time_s"] == 32.0
    # stop_after_episode_s: a second episode after the withdrawal restarts its interval.
    rules = NestTornadoCriterionConfig(stop_after_episode_s=2.0)
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 61.0):
        ledger.observe(_sample(t))
    for t in _times(61.5, 63.0):
        ledger.observe(_sample(t, wind=1.0))
    assert ledger.stop_request["rule"] == "stop_after_episode_s"
    assert ledger.stop_request["requested_at_model_time_s"] == 63.0
    ledger.observe(_sample(63.5))
    assert ledger.stop_request is None
    for t in _times(64.0, 123.5):
        ledger.observe(_sample(t))
    assert ledger.record()["episode_count"] == 2
    ledger.observe(_sample(124.0, wind=1.0))
    assert ledger.stop_request is None and ledger.last_episode_end_s == 123.5
    ledger.observe(_sample(125.0, wind=1.0))
    assert ledger.stop_request is None
    ledger.observe(_sample(125.5, wind=1.0))
    assert ledger.stop_request["last_episode_end_model_time_s"] == 123.5
    assert ledger.stop_request["requested_at_model_time_s"] == 125.5
    assert ledger.record()["last_withdrawn_stop_request"]["last_episode_end_model_time_s"] == 61.0
    # Without a pending request a run withdraws nothing.
    ledger = tc.CriterionLedger(rules)
    for t in _times(0.0, 10.0):
        ledger.observe(_sample(t))
    assert ledger.record()["stop_requests_withdrawn"] == 0
    assert ledger.record()["last_withdrawn_stop_request"] is None


def test_the_stop_block_names_the_frame_published_before_the_end() -> None:
    event = {"request": {"rule": "stop_after_episode_s", "threshold_s": 300.0},
             "frame_index": 38, "restart_number": None, "step": 9120, "model_time_s": 4560.0}
    block = tc.criterion_stop_record(event, frame_directory="/h/frame-000038",
                                     frames_requested=50, frames_published=38)
    assert block["rule"] == "stop_after_episode_s" and block["request"] == event["request"]
    assert (block["frame_index"], block["frame_directory"], block["restart_number"]) == (
        38, "/h/frame-000038", None)
    assert (block["step"], block["frames_published"]) == (9120, 38)


# ---------------------------------------------------------------------------
# 5. The sidecar survives a run that dies.
# ---------------------------------------------------------------------------
def test_a_failed_run_keeps_its_samples_and_a_torn_last_row_still_replays(tmp_path) -> None:
    driver = _Driver()
    criterion = _criterion(driver, root=tmp_path)
    g = driver.geometry
    for step in range(0, 75):
        t = driver.at(step, 1.0)
        driver.place([(g.origin_x_m + 2000.0 + FRAME[0] * t, g.origin_y_m + 2000.0 + FRAME[1] * t,
                       45.0, 80.0)], ground_background=FRAME)
        criterion.after_parent_step(driver)
    criterion.abandon()
    path = tmp_path / "nest-child" / tc.CRITERION_JSONL_FILENAME
    assert criterion.writer.closed and criterion.time_series[0]["rows"] == 75
    with path.open("ab") as handle:
        handle.write(b'[75,75.0,true,10')  # killed mid-row
    replay = tc.replay_time_series(path)
    assert replay["verdict"] == "TORNADO" and replay["samples"] == 75
    assert replay["genesis_model_time_s"] == 0.0 and replay["rows_identical"] is True
    # A sidecar that took no sample leaves nothing behind to block the retry.
    empty_root = tmp_path / "empty"
    empty = _criterion(_Driver(), root=empty_root)
    assert (empty_root / "nest-child" / tc.CRITERION_JSONL_FILENAME).is_file()
    empty.abandon()
    assert not (empty_root / "nest-child").exists()
    assert empty.close()["time_series"] == []


def test_an_infinite_wind_in_the_disk_fails_live_and_the_replay_agrees(tmp_path) -> None:
    """A blown-up word breaks the run live exactly as the sidecar's null breaks it in a replay."""

    # zh = 5, 15 m: both weights 0.5, so an infinite face stays infinite at 10 m
    # (a zero weight would make it 0*inf = NaN, which the maximum skips).
    driver = _Driver(dz=10.0)
    criterion = _criterion(driver, root=tmp_path)
    g = driver.geometry
    for step in range(0, 70):
        t = driver.at(step, 1.0)
        driver.place([(g.origin_x_m + 2000.0 + FRAME[0] * t, g.origin_y_m + 2000.0 + FRAME[1] * t,
                       45.0, 80.0)], ground_background=FRAME)
        if 20 <= step < 25:
            # The u face at (j, i) = (100, 105), 110 m from the centre corner (100, 100).
            driver.child.dynamics.arrays["u3d"][:, NG + 100, NG + 105] = numpy.inf
        row = criterion.after_parent_step(driver)
        if 20 <= step < 25:
            assert _column(row, "found") and _column(row, "ground_relative_wind_max_m_s") is None
            assert _column(row, "conditions") == "Z-CM" and _column(row, "state") == "-"
        else:
            assert _column(row, "conditions") == "ZWCM"
    record = criterion.close()
    # Runs 0-19 s and 25-69 s: neither spans 60 s.
    assert record["verdict"] == "NO_TORNADO" and record["qualifying_runs_started"] == 2
    replay = tc.replay_time_series(tmp_path / "nest-child" / tc.CRITERION_JSONL_FILENAME)
    assert replay["rows_identical"] is True and replay["mismatches"] == []
    for key in ("verdict", "samples", "condition_counts", "qualifying_runs_started",
                "longest_qualifying_run_s", "episodes", "episode_count"):
        assert replay[key] == record[key], key


class _CountingModule:
    """A module whose get_function calls are counted (CuPy compiles at the first one)."""

    def __init__(self, inner) -> None:
        self.inner, self.taken = inner, 0

    def get_function(self, name):
        self.taken += 1
        return self.inner.get_function(name)


def test_the_kernel_is_taken_once_when_the_criterion_is_built() -> None:
    from tools.oracle.nest_tornado_criterion_oracle import ReferenceModule

    driver = _Driver()
    module = _CountingModule(ReferenceModule())
    criterion = _criterion(driver, module=module)
    assert module.taken == 1 and criterion.launches == 0
    for step in range(3):
        driver.at(step, 1.0)
        criterion.after_parent_step(driver)
    assert module.taken == 1 and criterion.launches == 3

    class Broken:
        def get_function(self, name):
            raise RuntimeError("NVRTC_ERROR_COMPILATION")

    # A kernel that does not compile refuses when the nest is attached, not a parent step later.
    with pytest.raises(RuntimeError, match="NVRTC_ERROR_COMPILATION"):
        _criterion(_Driver(), module=Broken())


def test_the_live_sample_reads_the_runtime_owners_through_a_cupy_stream_on_the_host(tmp_path) -> None:
    """NestTornadoCriterion.sample as the runtime runs it, with no sample() override.

    ``tools/oracle/host_cupy.py`` stands in for CuPy: ``RawModule`` compiles the
    packaged kernel text with g++, the child's ``u3d``/``v3d`` are allocated at
    the padded shapes the dynamics allocates them with
    (``terrain_open_sound_shapes``), the launch goes through the stream, and the
    nine words come back through ``words.get(stream=...)``.
    """

    from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
    from tools.oracle import host_cupy

    cupy = host_cupy.build()
    stream = cupy.cuda.Stream(non_blocking=True)
    driver = _Driver(dz=10.0)
    g = driver.geometry
    nz = driver.child.config.grid.nz
    owners = terrain_open_sound_shapes(g.nx, g.ny, nz)
    host_cupy.COUNTS.clear()
    criterion = tc.NestTornadoCriterion(settings=NestTornadoCriterionConfig(), driver=driver,
                                        xp=cupy, stream=stream)
    assert host_cupy.COUNTS["raw_module_compile"] == 1 and criterion.launches == 0
    assert criterion.plan.owner_shapes() == {"u": owners["u3d"], "v": owners["v3d"]}
    assert isinstance(criterion.words, host_cupy.ndarray)
    criterion.open(tmp_path)
    rows = []
    for step in (40, 41):
        t = driver.at(step, 0.5)
        driver.place([(g.origin_x_m + 2013.0 + FRAME[0] * t, g.origin_y_m + 1987.0 + FRAME[1] * t,
                       45.0, 80.0)], ground_background=FRAME)
        host = {name: driver.child.dynamics.arrays[name].copy() for name in ("u3d", "v3d")}
        host["u3d"][2] += F32(3.0)  # the two bracketing levels differ
        device = {name: cupy.zeros(owners[name], dtype=cupy.float32) for name in ("u3d", "v3d")}
        for name in device:
            device[name][...] = host[name]
        live = driver.child.dynamics.arrays
        driver.child.dynamics.arrays = device
        launches, reads = host_cupy.COUNTS["raw_kernel_launch"], host_cupy.COUNTS["device_to_host_read"]
        row = criterion.after_parent_step(driver)
        driver.child.dynamics.arrays = live
        assert host_cupy.COUNTS["raw_kernel_launch"] == launches + 1
        assert host_cupy.COUNTS["device_to_host_read"] == reads + 1
        words = tc.reference_sample(host["u3d"], host["v3d"], plan=criterion.plan)
        assert words[0] == 1.0
        assert (_column(row, "centre_i"), _column(row, "centre_j")) == (int(words[1]), int(words[2]))
        assert _column(row, "zeta_max_s_inv") == words[3]
        assert _column(row, "ground_relative_wind_max_m_s") == words[4]
        assert _column(row, "frame_relative_wind_max_m_s") == words[5]
        assert _column(row, "core_corners") == int(words[6])
        assert _column(row, "conditions") == "ZWCM"
        rows.append(row)
    assert host_cupy.COUNTS["raw_module_compile"] == 1
    record = criterion.close()
    assert record["device_launches"] == 2 and record["time_series"][0]["rows"] == 2
    assert tc.replay_time_series(tmp_path / "nest-child" / tc.CRITERION_JSONL_FILENAME)[
        "rows_identical"] is True


def test_a_replay_refuses_a_sidecar_decided_by_another_criterion(tmp_path) -> None:
    driver = _Driver()
    criterion = _criterion(driver, root=tmp_path)
    criterion.close()
    path = tmp_path / "nest-child" / tc.CRITERION_JSONL_FILENAME
    header, _ = tc.read_time_series(path)
    header["criterion_sha256"] = "0" * 64
    path.write_text(json.dumps(header) + "\n")
    with pytest.raises(ValueError, match="another rule"):
        tc.replay_time_series(path)


# ---------------------------------------------------------------------------
# 6. Where the driver samples, and how the runtime consumes it.
# ---------------------------------------------------------------------------
def test_the_driver_samples_after_the_child_steps_and_before_feedback_and_any_move() -> None:
    from cm1core.nest_live import NestDriver

    order: list[str] = []
    driver = object.__new__(NestDriver)
    clock = SimpleNamespace(completed_steps=4, model_seconds=2.0, domainlocx_m=16.0,
                            domainlocy_m=12.0)
    child_clock = SimpleNamespace(completed_steps=12, domainlocx_m=0.0, domainlocy_m=0.0)

    def child_step():
        order.append("child.step")
        child_clock.completed_steps += 1

    driver.initialized_from_parent = True
    driver.q = 3
    driver.parent = SimpleNamespace(clock=clock)
    driver.child = SimpleNamespace(clock=child_clock, step=child_step,
                                   synchronize=lambda: order.append("child.synchronize"))
    driver.forcing = SimpleNamespace(refresh=lambda p, c: order.append("forcing.refresh"),
                                     advance_clock=lambda: None)
    driver.geometry = SimpleNamespace(feedback=1)
    driver.feed_back = lambda: order.append("feed_back")
    driver.mover = SimpleNamespace(after_parent_step=lambda d: order.append("mover"))
    driver.parcels = SimpleNamespace(after_parent_step=lambda d: order.append("parcels"))
    driver.window = SimpleNamespace(after_parent_step=lambda d: order.append("window"))
    driver.child_steps = driver.parent_steps = 0

    def sample(d):
        assert d.child.clock.domainlocx_m == 16.0 and d.child.clock.completed_steps == 15
        order.append("criterion")

    driver.criterion = SimpleNamespace(after_parent_step=sample)
    driver.parent.dynamics = driver.child.dynamics = None
    clock.completed_steps = 5
    NestDriver.after_parent_step(driver)
    assert order == ["forcing.refresh", "child.step", "child.step", "child.step",
                     "child.synchronize", "criterion", "feed_back", "mover", "parcels",
                     "window"]
    driver.criterion = None
    order.clear()
    clock.completed_steps = 6
    NestDriver.after_parent_step(driver)
    assert "criterion" not in order


def test_the_runtime_consumes_the_criterion_only_through_the_live_driver() -> None:
    from cm1core.cuda_cartesian_runtime import (
        INTEGRATION_COURANT_STOP_STATUS,
        INTEGRATION_CRITERION_STOP_STATUS,
        INTEGRATION_STATUS,
        CudaCartesianRuntime,
    )

    runtime = SimpleNamespace(nest_driver=None)
    bound = lambda name: getattr(CudaCartesianRuntime, name).__get__(runtime)  # noqa: E731
    runtime._nest_criterion = bound("_nest_criterion")
    assert bound("_nest_tornado_criterion")() == {}
    assert bound("_nest_tornado_criterion")(Path("/nowhere")) == {}
    assert bound("_nest_criterion_stop_request")() is None
    runtime.nest_driver = SimpleNamespace(criterion=None)
    assert bound("_nest_tornado_criterion")() == {}
    criterion = SimpleNamespace(stop_request={"rule": "x"}, opened=None,
                                open=lambda root: setattr(criterion, "opened", root),
                                close=lambda: {"closed": True})
    runtime.nest_driver = SimpleNamespace(criterion=criterion)
    assert bound("_nest_criterion_stop_request")() == {"rule": "x"}
    assert bound("_nest_tornado_criterion")(Path("/h")) == {} and criterion.opened == Path("/h")
    assert bound("_nest_tornado_criterion")() == {"nest_tornado_criterion": {"closed": True}}
    assert len({INTEGRATION_STATUS, INTEGRATION_COURANT_STOP_STATUS,
                INTEGRATION_CRITERION_STOP_STATUS}) == 3


def test_the_timing_harness_refuses_a_nest_that_declares_the_criterion(monkeypatch, tmp_path) -> None:
    """benchmark() would sample every step, warm-up included, and publish no row of them."""

    import tomllib

    import cm1core.cartesian_door as cartesian_door
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

    runtime = SimpleNamespace(nest_driver=SimpleNamespace(criterion=object()))
    runtime._nest_criterion = CudaCartesianRuntime._nest_criterion.__get__(runtime)
    with pytest.raises(ValueError, match="timing harness does not sample"):
        CudaCartesianRuntime.benchmark(runtime, warmup_steps=1, measured_steps=1)
    # The door refuses it by name before anything is resolved or a device opened.
    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    config = parse_mapping(tomllib.loads(BASE_TOML + _criterion_toml()))
    with pytest.raises(CM1Refusal) as caught:
        cartesian_door.run_cartesian(config, config_label="criterion", output=tmp_path / "r.json",
                                     integrate=False)
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-TORNADO-CRITERION-INTEGRATION"
    assert caught.value.detail.option == "nest.tornado_criterion" and caught.value.detail.remedy
    assert not (tmp_path / "r.json").exists()
    # --integrate is not refused for the criterion (it is refused next for its missing root).
    with pytest.raises(CM1Refusal) as caught:
        cartesian_door.run_cartesian(config, config_label="criterion", output=tmp_path / "r.json",
                                     integrate=True)
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-DIRECTORY"


class _ScriptedCriterion(tc.NestTornadoCriterion):
    """The shipped orchestration and ledger over scripted kernel words (no winds needed)."""

    def __init__(self, settings, words_at) -> None:
        self.settings, self.xp, self.stream = settings, numpy, None
        self.plan = _plan(nx=9, ny=11, nz=8, dx_m=500.0 / 3.0, dy_m=500.0 / 3.0, lo_i=0, hi_i=9,
                          lo_j=0, hi_j=11)
        self.bracket_heights_m = [10.0, 30.0]
        self.time_refinement_ratio = 3
        self.words = numpy.zeros(tc.SAMPLE_WORDS)
        self.module = self.kernel = None
        self.ledger = tc.CriterionLedger(settings)
        self.writer = None
        self.time_series = []
        self.launches = 0
        self.sample_wall = tc.SampleWall()
        self.stop_request_disposition = None
        self.carried_rows = None
        self.carried = None
        self.words_at = words_at

    def sample(self, driver):
        self.launches += 1
        return self.words_at(float(driver.parent.clock.model_seconds))


def _nested_double(criterion):
    from test_background_publication import _nested

    parent = _nested(seed=5)
    driver = parent.nest_driver
    driver.criterion = criterion
    stepping = parent.step

    def step():
        stepping()
        criterion.after_parent_step(driver)

    parent.step = step
    return parent


QUALIFYING = [1.0, 4.0, 5.0, 1.0, 40.0, 30.0, 20.0, 50.0, 60.0]
QUIET = [1.0, 4.0, 5.0, 0.1, 10.0, 8.0, 1.0, 50.0, 60.0]


def test_a_stop_rule_ends_the_integration_after_the_next_frame(tmp_path) -> None:
    """The shipped integrate on the host double: qualifying to 61 s, a 2 s quiet, stop at 70 s."""

    from cm1core.cuda_cartesian_runtime import INTEGRATION_CRITERION_STOP_STATUS

    criterion = _ScriptedCriterion(
        NestTornadoCriterionConfig(stop_after_episode_s=2.0),
        lambda t: QUALIFYING if t <= 61.0 else QUIET,
    )
    parent = _nested_double(criterion)
    root = tmp_path / "frames"
    receipt = parent.integrate(duration_s=100.0, history_interval_s=10.0, history_root=root)
    assert receipt["status"] == INTEGRATION_CRITERION_STOP_STATUS
    stop = receipt["criterion_stop"]
    assert stop["rule"] == "stop_after_episode_s"
    assert stop["request"]["requested_at_model_time_s"] == 63.0
    assert stop["request"]["last_episode_end_model_time_s"] == 61.0
    assert (stop["frame_index"], stop["step"], stop["model_time_s"]) == (7, 140, 70.0)
    assert stop["frame_directory"] == receipt["frames"][-1]["directory"]
    assert (stop["frames_requested"], stop["frames_published"]) == (10, 7)
    assert parent.clock.completed_steps == 140
    block = receipt["nest_tornado_criterion"]
    assert block["verdict"] == "TORNADO" and block["genesis_model_time_s"] == 0.5
    assert block["tornado_declared_model_time_s"] == 60.5 and block["samples"] == 140
    [series] = block["time_series"]
    assert series["rows"] == 140
    assert Path(series["path"]) == root / "nest-child" / tc.CRITERION_JSONL_FILENAME
    assert "nest.tornado_criterion stop rule" in receipt["claim_limit"]
    assert block["stop_request_disposition"] == tc.STOP_HONOURED
    replay = tc.replay_time_series(series["path"])
    assert replay["rows_identical"] and replay["stop_request"] == block["stop_request"]


def test_a_criterion_stop_past_the_frame_row_budget_still_names_its_directory(tmp_path) -> None:
    """frame_row_budget drops the row; the directory is the formatted one, as courant_stop's."""

    from cm1core.cuda_cartesian_runtime import INTEGRATION_CRITERION_STOP_STATUS

    criterion = _ScriptedCriterion(
        NestTornadoCriterionConfig(stop_after_episode_s=2.0),
        lambda t: QUALIFYING if t <= 61.0 else QUIET,
    )
    parent = _nested_double(criterion)
    root = tmp_path / "frames"
    receipt = parent.integrate(duration_s=100.0, history_interval_s=10.0, history_root=root,
                               frame_row_budget=1)
    assert receipt["status"] == INTEGRATION_CRITERION_STOP_STATUS
    assert [row["index"] for row in receipt["frames"]] == [1]
    stop = receipt["criterion_stop"]
    assert stop["frame_index"] == 7
    assert stop["frame_directory"] == os.fspath(root / "frame-000007")
    assert Path(stop["frame_directory"]).is_dir()


@pytest.mark.parametrize(("cutoff", "why"), [
    # Admitted by the ledger alone (config.py now refuses it): raised by the last sample.
    (4.0, "the cutoff is the run's duration"),
    # Admitted by config.py: raised inside the last history interval.
    (3.5, "the request is raised after the last non-final history event"),
])
def test_a_stop_request_pending_at_the_final_event_ends_nothing(tmp_path, cutoff, why) -> None:
    """The run integrated its whole duration: completed status, the request recorded unused."""

    from cm1core.cuda_cartesian_runtime import INTEGRATION_STATUS

    criterion = _ScriptedCriterion(NestTornadoCriterionConfig(stop_if_no_tornado_by_s=cutoff),
                                   lambda t: QUIET)
    parent = _nested_double(criterion)
    receipt = parent.integrate(duration_s=4.0, history_interval_s=1.0,
                               history_root=tmp_path / "frames")
    assert receipt["status"] == INTEGRATION_STATUS, why
    assert receipt["criterion_stop"] is None
    assert parent.clock.model_seconds == 4.0 and len(receipt["frames"]) == 4
    assert "not integrated" not in receipt["claim_limit"]
    block = receipt["nest_tornado_criterion"]
    assert block["stop_request"]["rule"] == "stop_if_no_tornado_by_s"
    assert block["stop_request"]["requested_at_model_time_s"] == cutoff
    assert block["stop_request_disposition"] == tc.STOP_UNUSED_AT_DURATION
    # One history interval earlier the same rule does end the run, at the next frame.
    criterion = _ScriptedCriterion(NestTornadoCriterionConfig(stop_if_no_tornado_by_s=cutoff - 1.0),
                                   lambda t: QUIET)
    parent = _nested_double(criterion)
    receipt = parent.integrate(duration_s=4.0, history_interval_s=1.0,
                               history_root=tmp_path / "earlier")
    assert receipt["criterion_stop"]["model_time_s"] == 3.0
    assert receipt["nest_tornado_criterion"]["stop_request_disposition"] == tc.STOP_HONOURED


def test_a_run_that_begins_after_the_request_keeps_the_integration_alive(tmp_path) -> None:
    """Cutoff 5 s, quiet to 5 s, qualifying from 5.5 s: the frame at 10 s ends nothing.

    Before the withdrawal the request raised at 5 s was honoured at the 10 s
    frame and the vortex that began at 5.5 s was cut with NO_TORNADO; now the
    run integrates its 80 s and the vortex becomes a TORNADO at 65.5 s.
    """

    from cm1core.cuda_cartesian_runtime import INTEGRATION_STATUS

    criterion = _ScriptedCriterion(
        NestTornadoCriterionConfig(stop_if_no_tornado_by_s=5.0),
        lambda t: QUIET if t <= 5.0 else QUALIFYING,
    )
    parent = _nested_double(criterion)
    root = tmp_path / "frames"
    receipt = parent.integrate(duration_s=80.0, history_interval_s=10.0, history_root=root)
    assert receipt["status"] == INTEGRATION_STATUS and receipt["criterion_stop"] is None
    assert parent.clock.model_seconds == 80.0 and len(receipt["frames"]) == 8
    block = receipt["nest_tornado_criterion"]
    assert block["verdict"] == "TORNADO" and block["genesis_model_time_s"] == 5.5
    assert block["tornado_declared_model_time_s"] == 65.5 and block["samples"] == 160
    assert block["stop_request"] is None and block["stop_request_disposition"] is None
    assert block["stop_requests_withdrawn"] == 1
    withdrawn = block["last_withdrawn_stop_request"]
    assert (withdrawn["rule"], withdrawn["requested_at_model_time_s"]) == (
        "stop_if_no_tornado_by_s", 5.0)
    assert (withdrawn["withdrawn_at_step"], withdrawn["withdrawn_at_model_time_s"]) == (11, 5.5)
    assert withdrawn["withdrawn_because"] == tc.STOP_WITHDRAWN_BY_RUN
    assert block["sample_wall_seconds"]["count"] == 160
    assert block["sample_wall_seconds"]["total"] >= block["sample_wall_seconds"]["maximum"] > 0.0
    # The sidecar alone reaches the same withdrawal.
    [series] = block["time_series"]
    replay = tc.replay_time_series(series["path"])
    assert replay["rows_identical"] and replay["stop_request"] is None
    assert replay["stop_requests_withdrawn"] == 1
    assert replay["last_withdrawn_stop_request"] == withdrawn
    assert "nest.tornado_criterion stop rule" not in receipt["claim_limit"]


def test_the_door_receipt_says_a_criterion_stop_ended_the_run(monkeypatch, tmp_path) -> None:
    """The door's own status and claim, never the completed-run ones; exit is success."""

    import cm1core.cartesian_door as cartesian_door
    from cm1core.cuda_cartesian_runtime import INTEGRATION_CRITERION_STOP_STATUS
    from test_storm_lane import ANALYTIC, _STUB_SOUNDING_REPORT

    library = tmp_path / "libcm1_state.so"
    library.write_bytes(b"")
    frames = tmp_path / "frames"
    receipt_path = tmp_path / "receipt.json"
    stop = tc.criterion_stop_record(
        {"request": {"rule": "stop_if_no_tornado_by_s", "threshold_s": 60.0},
         "frame_index": 2, "restart_number": None, "step": 120, "model_time_s": 60.0},
        frame_directory=str(frames / "frame-000002"),
        frames_requested=4, frames_published=2,
    )
    integration = {"status": INTEGRATION_CRITERION_STOP_STATUS, "courant_stop": None,
                   "criterion_stop": stop, "executed_steps": 120, "model_seconds_at_end": 60.0}
    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    monkeypatch.setattr(cartesian_door, "resolve_initializer_library", lambda argument: library)
    monkeypatch.setattr(cartesian_door, "resolve_history_publisher",
                        lambda history_dir, argument=None: None)
    monkeypatch.setattr(cartesian_door, "require_cuda_runtime", lambda: None)
    monkeypatch.setattr(cartesian_door, "require_sounding_report",
                        lambda *a, **k: _STUB_SOUNDING_REPORT)
    monkeypatch.setattr(cartesian_door, "_execute", lambda **kwargs: (dict(integration), None))
    assert cartesian_door.run_cartesian(
        load_config(ANALYTIC), config_label=str(ANALYTIC), output=receipt_path,
        history_dir=frames, integrate=True,
    ) == 0
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["status"] == "stopped-at-nest-tornado-criterion-stop-rule"
    assert written["criterion_stop"] == stop and written["courant_stop"] is None
    assert "not integrated, by design" in written["claim_limit"]
    # A receipt whose integration carries no criterion keeps exactly its keys.
    del integration["criterion_stop"]
    other = tmp_path / "other.json"
    cartesian_door.run_cartesian(load_config(ANALYTIC), config_label=str(ANALYTIC), output=other,
                                 history_dir=tmp_path / "frames2", integrate=True)
    written = json.loads(other.read_text(encoding="utf-8"))
    assert "criterion_stop" not in written
    assert written["status"] == "executed-cartesian-integration-to-requested-duration"


def test_without_a_stop_rule_the_integration_runs_to_its_duration(tmp_path) -> None:
    from cm1core.cuda_cartesian_runtime import INTEGRATION_STATUS

    criterion = _ScriptedCriterion(NestTornadoCriterionConfig(), lambda t: QUIET)
    parent = _nested_double(criterion)
    receipt = parent.integrate(duration_s=4.0, history_interval_s=1.0,
                               history_root=tmp_path / "frames")
    assert receipt["status"] == INTEGRATION_STATUS and receipt["criterion_stop"] is None
    assert receipt["nest_tornado_criterion"]["verdict"] == "NO_TORNADO"
    assert receipt["nest_tornado_criterion"]["samples"] == 8
    assert receipt["nest_tornado_criterion"]["stop_request_disposition"] is None


def test_a_run_that_fails_mid_integration_keeps_the_rows_it_sampled(tmp_path) -> None:
    def words_at(t):
        if t >= 3.0:
            raise RuntimeError("the child went non-finite")
        return QUALIFYING

    criterion = _ScriptedCriterion(NestTornadoCriterionConfig(), words_at)
    parent = _nested_double(criterion)
    root = tmp_path / "frames"
    with pytest.raises(RuntimeError, match="non-finite"):
        parent.integrate(duration_s=4.0, history_interval_s=1.0, history_root=root)
    path = root / "nest-child" / tc.CRITERION_JSONL_FILENAME
    assert criterion.writer.closed and criterion.time_series[0]["rows"] == 5
    assert hashlib.sha256(path.read_bytes()).hexdigest() == criterion.time_series[0]["sha256"]
    assert tc.replay_time_series(path)["samples"] == 5


# ---------------------------------------------------------------------------
# 7. The configuration: fixed thresholds, optional stop rules, sealed identity.
# ---------------------------------------------------------------------------
BASE_TOML = """
[run]
lane = "cartesian"
duration_s = 6000.0
dt_s = 0.5
history_interval_s = 60.0
restart_interval_s = 600.0

[initialization]
isnd = 5
iwnd = 2

[grid]
nx = 160
ny = 160
nz = 60
dx_m = 100.0
dy_m = 100.0
dz_m = 20.0
terrain = false
stretch_x = false
stretch_y = false
stretch_z = false

[dynamics]
psolver = 3
adaptive_timestep = false
scalar_horizontal_order = 5
scalar_vertical_order = 5
velocity_horizontal_order = 5
velocity_vertical_order = 5
scalar_weno = 2
velocity_weno = 0
weno_order = 5
dry_mass_correction = true

[boundaries]
west = 2
east = 2
south = 2
north = 2
bottom = 1
top = 1
radiative_order = 4

[closures]
sgs_model = 1
vertical_implicit_diffusion = true
upper_damping = 1
upper_damping_depth_m = 300.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = true
u_move_m_s = 8.5
v_move_m_s = 6.0

[runtime]
acoustic_substeps = 6
tile_nx = 64
tile_ny = 64
warmup_steps = 0
memory_budget_mib = 8192
capacity_profile = "default"

[physics]
component = "wsm6-mp6"
equation_set = 2

[nest]
refinement_ratio = 5
time_refinement_ratio = 3
parent_i0 = 35
parent_j0 = 35
parent_span_x = 90
parent_span_y = 90
spec_zone = 1
relax_zone = 4
feedback = 1
acoustic_substeps = 6
follow_storm = true
follow_interval_s = 30.0
follow_max_speed_m_s = 40.0
follow_tracker = "wrf-surface-circulation"
follow_smoothing_radius_m = 1000.0
follow_corral_m = 3000.0
"""


def _criterion_toml(**keys) -> str:
    lines = ["", "[nest.tornado_criterion]"]
    lines += [f"{key} = {json.dumps(value)}" for key, value in keys.items()]
    return "\n".join(lines) + "\n"


def _parse(text: str):
    import tomllib

    return parse_mapping(tomllib.loads(text))


def test_the_declared_criterion_is_bound_projected_and_sealed() -> None:
    from cm1core.cartesian_door import require_nest

    flat = _parse(BASE_TOML)
    assert flat.nest.tornado_criterion is None
    assert "tornado_criterion" not in flat.canonical_dict()["nest"]
    bare = _parse(BASE_TOML + _criterion_toml())
    assert bare.nest.tornado_criterion == NestTornadoCriterionConfig()
    assert bare.canonical_dict()["nest"]["tornado_criterion"] == {}
    stops = _parse(BASE_TOML + _criterion_toml(stop_after_episode_s=900.0,
                                               stop_if_no_tornado_by_s=5400.0))
    assert stops.canonical_dict()["nest"]["tornado_criterion"] == {
        "stop_after_episode_s": 900.0, "stop_if_no_tornado_by_s": 5400.0}
    one = _parse(BASE_TOML + _criterion_toml(stop_after_episode_s=900.0))
    assert one.canonical_dict()["nest"]["tornado_criterion"] == {"stop_after_episode_s": 900.0}
    assert len({flat.sha256, bare.sha256, stops.sha256, one.sha256}) == 4
    binding = require_nest(stops)
    assert binding.tornado_criterion == stops.nest.tornado_criterion
    plan = binding.record(parent_dt_s=0.5)["tornado_criterion"]
    assert plan["criterion"] == dict(tc.CRITERION)
    assert plan["criterion_sha256"] == tc.CRITERION_SHA256
    assert plan["kernel_sha256"] == tc.NEST_TORNADO_KERNEL_SHA256
    assert plan["stop_rules"] == {"stop_after_episode_s": 900.0, "stop_if_no_tornado_by_s": 5400.0}
    assert "tornado_criterion" not in require_nest(flat).record(parent_dt_s=0.5)


def test_a_nest_without_the_criterion_keeps_its_sealed_identity() -> None:
    # Measured at fbd129e, before the table existed.
    config = load_config(REPO / "examples" / "cartesian-nested-storm-following-mesh.toml")
    assert config.nest.tornado_criterion is None
    assert "tornado_criterion" not in config.canonical_dict()["nest"]
    assert config.sha256 == "be470b625b3487a8b759818968161ef9652d44a28539c5d60a61ae75b9004ef5"


@pytest.mark.parametrize(
    ("text", "code", "fragment", "option"),
    [
        (_criterion_toml(zeta_min_s_inv=0.2), "CM1-CFG-NEST-TORNADO-CRITERION-FIXED",
         "tuned after a run", "nest.tornado_criterion.zeta_min_s_inv"),
        (_criterion_toml(sustain_s=30.0), "CM1-CFG-NEST-TORNADO-CRITERION-FIXED",
         "fixed tornado criterion", "nest.tornado_criterion.sustain_s"),
        (_criterion_toml(ground_relative_wind_m_s=30.0), "CM1-CFG-NEST-TORNADO-CRITERION-FIXED",
         "fixed tornado criterion", "nest.tornado_criterion.ground_relative_wind_m_s"),
        (_criterion_toml(sample_every=2), "CM1-CFG-UNKNOWN", "unconsumed",
         "nest.tornado_criterion.sample_every"),
        (_criterion_toml(stop_after_episode_s=0.0), "CM1-CFG-NUMERIC-RANGE", "non-positive",
         "nest.tornado_criterion.stop_after_episode_s"),
        (_criterion_toml(stop_if_no_tornado_by_s=-5.0), "CM1-CFG-NUMERIC-RANGE", "non-positive",
         "nest.tornado_criterion.stop_if_no_tornado_by_s"),
        (_criterion_toml(stop_if_no_tornado_by_s=6000.5), "CM1-CFG-NEST-TORNADO-STOP",
         "could never act", "nest.tornado_criterion.stop_if_no_tornado_by_s"),
        # At the duration: raised by the last sample, at the final event.
        (_criterion_toml(stop_if_no_tornado_by_s=6000.0), "CM1-CFG-NEST-TORNADO-STOP",
         "final scheduled event", "nest.tornado_criterion.stop_if_no_tornado_by_s"),
        (None, None, None, None),
    ],
)
def test_the_table_is_refused_by_key_and_by_value(text, code, fragment, option) -> None:
    if text is None:
        # [nest.tornado_criterion] written as a scalar in [nest].
        import tomllib

        raw = tomllib.loads(BASE_TOML)
        raw["nest"]["tornado_criterion"] = 1.0
        with pytest.raises(CM1Refusal) as caught:
            parse_mapping(raw)
        assert caught.value.detail.code == "CM1-CFG-SECTION-TYPE"
        assert caught.value.detail.option == "nest.tornado_criterion"
        return
    with pytest.raises(CM1Refusal) as caught:
        _parse(BASE_TOML + text)
    detail = caught.value.detail
    assert detail.code == code, detail.reason
    assert fragment in detail.reason, detail.reason
    assert detail.option == option
    assert detail.remedy


@pytest.mark.parametrize(
    ("edits", "code", "fragment"),
    [
        # zh(1) = 12.5 m: 10 m would be extrapolated.
        ({"dz_m = 20.0": "dz_m = 25.0"}, "CM1-CFG-NEST-TORNADO-HEIGHT", "extrapolated"),
        # A 1.9 km child holds no corner 1000 m from both edges.
        ({"parent_span_x = 90": "parent_span_x = 19", "follow_storm = true": "follow_storm = false",
          "follow_interval_s = 30.0": "follow_interval_s = 0.0",
          "follow_max_speed_m_s = 40.0": "follow_max_speed_m_s = 0.0",
          'follow_tracker = "wrf-surface-circulation"\n': "",
          "follow_smoothing_radius_m = 1000.0\n": "", "follow_corral_m = 3000.0\n": ""},
         "CM1-CFG-NEST-TORNADO-EXTENT", "no corner could ever be a centre"),
        # 35 forced rows of 20 m reach the winds a 300 m disk reads 1000 m in.
        ({"relax_zone = 4": "relax_zone = 34"}, "CM1-CFG-NEST-TORNADO-EDGE-MARGIN",
         "parent wind"),
    ],
)
def test_a_child_the_fixed_criterion_cannot_sample_is_refused(edits, code, fragment) -> None:
    text = BASE_TOML
    for old, new in edits.items():
        assert old in text
        text = text.replace(old, new)
    _parse(text)  # the nest alone is admitted
    with pytest.raises(CM1Refusal) as caught:
        _parse(text + _criterion_toml())
    assert caught.value.detail.code == code, caught.value.detail.reason
    assert fragment in caught.value.detail.reason


def test_a_criterion_outside_nest_is_an_unknown_section() -> None:
    without_nest = BASE_TOML.split("\n[nest]\n")[0]
    with pytest.raises(CM1Refusal) as caught:
        _parse(without_nest + _criterion_toml().replace("[nest.tornado_criterion]",
                                                        "[tornado_criterion]"))
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"


# ---------------------------------------------------------------------------
# 8. Offline: the tool replays the sidecar and recomputes published frames.
# ---------------------------------------------------------------------------
def _write_child_frame(child_root: Path, index: int, driver: _Driver, *,
                       child_step_offset: int = 0) -> None:
    from cm1core.cm1_vertical_grid import uniform_vertical_levels
    from cm1core.nest_parcels import NEST_CHILD_PLACEMENT_DIRECTORY, NEST_CHILD_PLACEMENT_SCHEMA

    g, c = driver.geometry, driver.child
    nz = c.config.grid.nz
    levels = uniform_vertical_levels(nz, c.config.grid.dz_m)
    frame = child_root / f"frame-{index:06d}"
    frame.mkdir(parents=True)
    u, v = c.dynamics.arrays["u3d"], c.dynamics.arrays["v3d"]
    carriers = [
        ("u3d", u[1:nz + 1, NG:NG + g.ny, NG:NG + g.nx + 1]),
        ("v3d", v[1:nz + 1, NG:NG + g.ny + 1, NG:NG + g.nx]),
        ("w3d", numpy.zeros((nz + 1, g.ny, g.nx), dtype=F32)),
    ]
    arena, rows, offset = b"", [], 0
    for name, array in carriers:
        words = numpy.ascontiguousarray(array, dtype="<f4").tobytes()
        rows.append("\t".join([name, "m", "u", "s", ",".join(map(str, array.shape)), "b", "r",
                               str(offset)]))
        arena += words
        offset += len(words)
    (frame / "history-carriers.f32").write_bytes(arena)
    (frame / "history-field-table.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")
    placements = child_root / NEST_CHILD_PLACEMENT_DIRECTORY
    placements.mkdir(exist_ok=True)
    # The placement names the CHILD's step, as the runtime writes it
    # (NestDriver.snapshot_child_frame -> child._snapshot_frame): the parent's
    # completed steps times the time refinement ratio.
    child_step = driver.clock.completed_steps * g.time_refinement_ratio + child_step_offset
    (placements / f"frame-{index:06d}.json").write_text(json.dumps({
        "schema": NEST_CHILD_PLACEMENT_SCHEMA, "index": index,
        "step": child_step, "model_time_s": driver.clock.model_seconds,
        "parent_i0": g.parent_i0, "parent_j0": g.parent_j0,
        "nx": g.nx, "ny": g.ny, "nz": nz, "dx_m": g.dx_m, "dy_m": g.dy_m,
        "origin_x_m": g.origin_x_m, "origin_y_m": g.origin_y_m,
        "dz_reference_m": float(levels.dz_reference_m),
        "zf_cm1_indexed": [float(z) for z in levels.zf], "zh_cm1_indexed": [float(z) for z in levels.zh],
        "mf_cm1_indexed": [float(m) for m in levels.mf], "cgs": list(levels.cgs),
        "cgt": list(levels.cgt), "vertical_sha256": levels.sha256, "bbc": 1, "tbc": 1,
        "u_move_m_s": driver.frame[0], "v_move_m_s": driver.frame[1],
        "domainlocx_m": driver.clock.domainlocx_m, "domainlocy_m": driver.clock.domainlocy_m,
        "spec_zone": 1, "relax_zone": 4,
    }), encoding="utf-8")


def test_the_tool_replays_the_sidecar_and_recomputes_it_from_published_frames(tmp_path) -> None:
    """Frames keyed by the child's step (q = 5) find the rows of their parent steps.

    The frame velocity and step carry low-order bits (float32 8.3 m/s, dt =
    0.4 s), so the ground origin recovered from a row in floating point would
    not round-trip; the placement is compared as integers instead.
    """

    driver = _Driver(frame=(float(numpy.float32(8.3)), 6.0))
    assert driver.geometry.time_refinement_ratio == 5
    root = tmp_path / "history"
    criterion = _criterion(driver, root=root)
    child_root = root / "nest-child"
    g = driver.geometry
    dt = 0.4
    frames_on_the_sampled_placement = 0
    for step in range(0, 61):
        t = driver.at(step, dt)
        driver.place([(g.origin_x_m + 1800.0 + TRANSLATION[0] * t,
                       g.origin_y_m + 2100.0 + TRANSLATION[1] * t, 45.0, 80.0)],
                      ground_background=TRANSLATION)
        criterion.after_parent_step(driver)
        if step % 5 == 0:
            frames_on_the_sampled_placement += 1
            _write_child_frame(child_root, step // 5 + 1, driver)
    # A frame published after a nest move: the child stands elsewhere.
    driver.geometry = replace(g, parent_i0=g.parent_i0 + 1)
    _write_child_frame(child_root, 14, driver)
    driver.geometry = g
    # A placement whose child step is not a parent-step boundary has no row.
    _write_child_frame(child_root, 15, driver, child_step_offset=2)
    criterion.close()
    header, rows = tc.read_time_series(child_root / tc.CRITERION_JSONL_FILENAME)
    assert header["time_refinement_ratio"] == 5
    assert (_column(rows[7], "step"), _column(rows[7], "parent_i0"),
            _column(rows[7], "parent_j0")) == (7, g.parent_i0, g.parent_j0)
    report = tc.recompute_from_frames(child_root)
    assert frames_on_the_sampled_placement == 13
    assert (report["frames_read"], report["frames_compared"], report["frames_identical_to_jsonl"],
            report["frames_after_a_move"], report["frames_without_a_row"]) == (15, 13, 13, 1, 1)
    by_index = {entry["index"]: entry for entry in report["frames"]}
    assert (by_index[3]["child_step"], by_index[3]["parent_step"]) == (50, 10)
    assert by_index[14]["placement_sampled"] == [g.parent_i0, g.parent_j0]
    assert by_index[14]["placement_published"] == [g.parent_i0 + 1, g.parent_j0]
    assert by_index[15]["parent_step"] is None and by_index[15]["jsonl_row"] is False
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{REPO / 'src'}"}
    tool = REPO / "tools" / "tornado_criterion.py"
    done = subprocess.run([sys.executable, str(tool), "replay",
                           str(child_root / tc.CRITERION_JSONL_FILENAME)],
                          capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout)
    assert summary["verdict"] == "NO_TORNADO" and summary["rows_identical"] is True
    assert summary["samples"] == 61
    out = tmp_path / "frames.json"
    done = subprocess.run([sys.executable, str(tool), "frames", str(child_root), "--out", str(out)],
                          capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["frames_identical_to_jsonl"] == 13
    assert json.loads(done.stdout)["frames_without_a_row"] == 1
    assert json.loads(out.read_text())["kind"] == "offline-recompute-from-frames"
    done = subprocess.run([sys.executable, str(tool), "replay", str(tmp_path / "missing.jsonl")],
                          capture_output=True, text=True, env=env)
    assert done.returncode == 2 and "refused" in done.stderr


def test_a_qualifying_run_survives_a_nest_move_and_the_frames_tool_counts_the_moved_frame(
    tmp_path,
) -> None:
    """The nest moves one parent cell east under a translating vortex: no condition breaks.

    The sample is taken on the placement the child was integrated on and the
    move follows it (``NestDriver.after_parent_step`` order); the next sample's
    corner index drops by the refinement ratio, its ground position does not
    jump, so ``M`` holds and the run goes on to an episode.  The frame written
    at the step of the move carries the new placement against a row sampled
    on the old one, and ``recompute_from_frames`` counts it instead of
    comparing it; the frames on either side compare identical.
    """

    driver = _Driver()
    root = tmp_path / "history"
    criterion = _criterion(driver, root=root)
    child_root = root / "nest-child"
    g = driver.geometry
    r = g.refinement_ratio
    x0, y0 = g.origin_x_m + 1500.0, g.origin_y_m + 2000.0
    rows = []
    for step in range(0, 71):
        t = driver.at(step, 1.0)
        driver.place([(x0 + TRANSLATION[0] * t, y0 + TRANSLATION[1] * t, 45.0, 80.0)],
                     ground_background=TRANSLATION)
        rows.append(criterion.after_parent_step(driver))
        if step in (25, 30, 35):
            _write_child_frame(child_root, step // 5, driver)
        if step == 30:
            # After the sample and the frame of this step: the mover shifts the
            # child one parent cell east.  The fixture's arrays are re-placed
            # from the new origin at the next step, as the move's copy and
            # capture leave them on the device.
            driver.geometry = replace(g, parent_i0=g.parent_i0 + 1)
            _write_child_frame(child_root, 40, driver)
    before, after = rows[30], rows[31]
    assert (_column(before, "parent_i0"), _column(after, "parent_i0")) == (
        g.parent_i0, g.parent_i0 + 1)
    # In the child the vortex drifts (12 - 8) m/s = 0.2 cells per second; the
    # move takes the refinement ratio off the corner index in one step.
    assert abs(_column(after, "centre_i") - (_column(before, "centre_i") - r)) <= 1
    assert _column(after, "centre_move_m") <= math.hypot(*TRANSLATION) + 2 * g.dx_m
    assert _column(after, "conditions") == "ZWCM" and _column(after, "state") == "run"
    assert all(_column(row, "state") != "-" for row in rows[1:])
    record = criterion.close()
    assert record["verdict"] == "TORNADO" and record["genesis_model_time_s"] == 0.0
    assert record["condition_counts"]["centre_move"] == 71
    report = tc.recompute_from_frames(child_root)
    assert (report["frames_read"], report["frames_compared"], report["frames_identical_to_jsonl"],
            report["frames_after_a_move"], report["frames_without_a_row"]) == (4, 3, 3, 1, 0)
    by_index = {entry["index"]: entry for entry in report["frames"]}
    assert by_index[40]["placement_sampled"] == [g.parent_i0, g.parent_j0]
    assert by_index[40]["placement_published"] == [g.parent_i0 + 1, g.parent_j0]
    assert by_index[7]["same_placement_as_sample"] and by_index[7]["identical_to_jsonl"]
    replay = tc.replay_time_series(child_root / tc.CRITERION_JSONL_FILENAME)
    assert replay["rows_identical"] and replay["verdict"] == "TORNADO"


# ---------------------------------------------------------------------------
# 9. Device verification (opt-in, on a GPU node).
# ---------------------------------------------------------------------------
requires_opted_in_device = pytest.mark.skipif(
    os.environ.get("GM1_NEST_DEVICE_TESTS") != "1",
    reason="GM1_NEST_DEVICE_TESTS=1 is not set; no CUDA context is opened unless asked for by name",
)


@requires_opted_in_device
def test_the_device_sample_words_equal_the_host_reference() -> None:
    cupy = pytest.importorskip("cupy")
    module = tc.compile_nest_tornado_criterion_module(cupy)
    stream = cupy.cuda.Stream(non_blocking=True)
    rng = numpy.random.default_rng(2026)
    plan = _plan(nx=450, ny=450, nz=12, ku0=2, ku1=3, w0=0.5, w1=0.5, radius_m=300.0,
                 lo_i=50, hi_i=400, lo_j=50, hi_j=400)
    u, v = _padded_winds(450, 450, 12, 20.0, [(4512.0, 4488.0, 45.0, 80.0)], (4.0, -3.0))
    cases = [(u, v)]
    ru, rv = _random_winds(rng, plan)
    ru[2, 120, 200] = numpy.nan
    cases.append((ru, rv))
    for host_u, host_v in cases:
        with stream:
            du, dv = cupy.asarray(host_u), cupy.asarray(host_v)
            words = cupy.full(tc.SAMPLE_WORDS, 7.0, dtype=cupy.float64)
            tc.launch_tornado_sample(cupy, module, u=du, v=dv, out=words, plan=plan, stream=stream)
        host = words.get(stream=stream)
        assert host.tobytes() == tc.reference_sample(host_u, host_v, plan=plan).tobytes()


@requires_opted_in_device
def test_the_device_live_sample_reads_cupy_owners_through_its_stream() -> None:
    """NestTornadoCriterion on CuPy: NVRTC at construction, one launch and one stream read."""

    cupy = pytest.importorskip("cupy")
    stream = cupy.cuda.Stream(non_blocking=True)
    driver = _Driver(dz=10.0)
    g = driver.geometry
    criterion = tc.NestTornadoCriterion(settings=NestTornadoCriterionConfig(), driver=driver,
                                        xp=cupy, stream=stream)
    t = driver.at(40, 0.5)
    driver.place([(g.origin_x_m + 2013.0 + FRAME[0] * t, g.origin_y_m + 1987.0 + FRAME[1] * t,
                   45.0, 80.0)], ground_background=FRAME)
    host = {name: driver.child.dynamics.arrays[name].copy() for name in ("u3d", "v3d")}
    host["u3d"][2] += F32(3.0)
    with stream:
        driver.child.dynamics.arrays = {name: cupy.asarray(host[name]) for name in host}
    row = criterion.after_parent_step(driver)
    words = tc.reference_sample(host["u3d"], host["v3d"], plan=criterion.plan)
    assert (_column(row, "centre_i"), _column(row, "centre_j")) == (int(words[1]), int(words[2]))
    assert _column(row, "zeta_max_s_inv") == words[3]
    assert _column(row, "ground_relative_wind_max_m_s") == words[4]
    assert _column(row, "frame_relative_wind_max_m_s") == words[5]
    assert _column(row, "core_corners") == int(words[6])
    assert criterion.launches == 1


# ---------------------------------------------------------------------------
# Restarts: the ledger is nest child restart state (cm1core.nest_restart).
# ---------------------------------------------------------------------------
_NAME = __import__("re").compile(r"^[a-z][a-z0-9._-]{0,63}$")


def _ledger_sample(step: int, t: float, *, found: bool = True, zeta: float = 0.5, wind: float = 40.0,
            x: float = 0.0, y: float = 0.0, corners: int = 20) -> tc.CriterionSample:
    return tc.CriterionSample(
        step=step, model_time_s=t, parent_i0=1, parent_j0=1, found=found,
        centre_i=5 if found else -1, centre_j=5 if found else -1,
        ground_x_m=x if found else math.nan, ground_y_m=y if found else math.nan,
        zeta_max_s_inv=zeta if found else math.nan, ground_relative_wind_m_s=wind,
        frame_relative_wind_m_s=wind - 5.0, core_corners=corners,
        core_radius_m=tc.core_radius_m(20.0, corners),
    )


def test_the_ledger_checkpoint_round_trips_every_field_and_continues_identically() -> None:
    """A run in progress, a closed episode, a withdrawn request and a pending one all cross.

    Episode 0.5-70 s, quiet to 110 s (the after-episode rule requests a stop
    at 100 s), qualifying from 110 s (withdraws it): checkpointed at 120 s with
    a run in progress, restored into a fresh ledger, and both ledgers fed the
    same samples to 300 s -- the second episode is declared at 170 s by both,
    and the stop request re-raised at 230 s by both.
    """

    rules = SimpleNamespace(stop_after_episode_s=30.0, stop_if_no_tornado_by_s=None)
    whole = tc.CriterionLedger(rules)
    for step in range(1, 241):
        t = step * 0.5
        whole.observe(_ledger_sample(step, t, zeta=0.5 if (t <= 70.0 or t >= 110.0) else 0.1))
    assert whole.stop_requests_withdrawn == 1 and whole.run is not None and not whole.run.episode
    rows = whole.checkpoint()
    assert all(_NAME.match(name) for name in rows), [n for n in rows if not _NAME.match(n)]
    assert all(kind in ("u64", "bool", "f64-bits") for kind, _ in rows.values())
    assert rows["nest.tornado.samples"] == ("u64", "240")
    assert rows["nest.tornado.run"] == ("bool", "true")
    assert rows["nest.tornado.episodes"] == ("u64", "1")
    assert rows["nest.tornado.withdrawn"] == ("bool", "true")
    assert rows["nest.tornado.stop_request"] == ("bool", "false")
    # An untouched peak is -inf: carried bit for bit, not as null.
    untouched = tc._Run(number=1, start_s=0.0, last_s=0.0, start_xy=(0.0, -0.0), last_xy=(0.0, 0.0))
    encoded = tc._run_rows("x.", untouched)
    assert encoded["x.peak_zeta.value"] == ("f64-bits", "fff0000000000000")
    assert encoded["x.start_y_m"] == ("f64-bits", "8000000000000000")  # the sign of zero
    decoded = tc._run_from_rows(tc._ScalarReader(encoded), "x.")
    assert decoded == untouched and math.isinf(decoded.peak_wf[0])

    resumed = tc.CriterionLedger(rules)
    carried = resumed.restore(rows)
    assert resumed.checkpoint() == rows and resumed.record() == whole.record()
    assert carried["verdict_at_restart"] == "TORNADO" and carried["samples"] == 240
    assert carried["qualifying_run_in_progress"]["start_model_time_s"] == 110.0
    for step in range(241, 601):
        t = step * 0.5
        sample = _ledger_sample(step, t, zeta=0.5 if t <= 200.0 else 0.1)
        whole.observe(sample)
        resumed.observe(sample)
    assert resumed.record() == whole.record()
    record = whole.record()
    assert record["episode_count"] == 2 and record["stop_request"]["requested_at_model_time_s"] == 230.0
    assert record["episodes"][1]["start_model_time_s"] == 110.0
    # A missing or mistyped row is refused by name; a restore over samples too.
    missing = dict(rows)
    missing.pop("nest.tornado.run.peak_wg.at_s")
    with pytest.raises(ValueError, match="nest.tornado.run.peak_wg.at_s"):
        tc.CriterionLedger(rules).restore(missing)
    mistyped = dict(rows)
    mistyped["nest.tornado.samples"] = ("bool", "true")
    with pytest.raises(ValueError, match="nest.tornado.samples is bool, not u64"):
        tc.CriterionLedger(rules).restore(mistyped)
    with pytest.raises(ValueError, match="already observed"):
        whole.restore(rows)
    other = dict(rows)
    other["nest.tornado.schema"] = ("u64", "2")
    with pytest.raises(ValueError, match="schema 2"):
        tc.CriterionLedger(rules).restore(other)


def test_an_empty_and_a_no_centre_ledger_checkpoint_too() -> None:
    empty = tc.CriterionLedger()
    rows = empty.checkpoint()
    back = tc.CriterionLedger()
    assert back.restore(rows)["samples"] == 0 and back.record() == empty.record()
    lost = tc.CriterionLedger(SimpleNamespace(stop_after_episode_s=None, stop_if_no_tornado_by_s=3.0))
    for step in range(1, 9):
        lost.observe(_ledger_sample(step, step * 0.5, found=False))
    assert lost.stop_request is not None and lost.previous[1] is False
    back = tc.CriterionLedger(SimpleNamespace(stop_after_episode_s=None, stop_if_no_tornado_by_s=3.0))
    back.restore(lost.checkpoint())
    assert back.record() == lost.record()
    # The previous centre of a sample that found none is NaN: carried as bits.
    assert back.previous[:2] == lost.previous[:2] == (4.0, False)
    assert all(math.isnan(v) for v in (*back.previous[2:], *lost.previous[2:]))
    assert back.checkpoint() == lost.checkpoint()


def test_a_resumed_nested_run_carries_the_ledger_through_the_child_restart(
    forecast_publisher, tmp_path
) -> None:
    """The shipped integrate and nest restart on the host double, resumed mid-run.

    Qualifying to 61 s (an episode), quiet to 70 s, qualifying again from 70 s:
    the restart at 80 s is written with a 10 s run in progress and a closed
    episode.  The uninterrupted run declares the second episode at 130 s; the
    resumed run, whose ledger starts from the child restart's scalar rows,
    declares it at 130 s too, and its receipt's ledger is the whole run's
    (160 s in all, restarts every 40 s).  A resume that starts the ledger
    fresh at 80 s declares nothing before 140.5 s and reports one episode: the
    negative control.
    """

    from cm1core.cartesian_restart import _parse_scalar_table, read_cartesian_restart
    from cm1core.cuda_cartesian_runtime import INTEGRATION_STATUS
    from cm1core.nest_restart import NestRestartState, nest_child_restart_profile

    publisher = forecast_publisher
    words = lambda t: QUALIFYING if (t <= 61.0 or t >= 70.0) else QUIET  # noqa: E731

    def run(parent, root: Path, *, duration: float, **keywords):
        return parent.integrate(
            duration_s=duration, history_interval_s=10.0, history_root=root / "frames",
            history_publisher=publisher, restart_root=root / "restarts",
            restart_interval_s=40.0, restart_publisher=publisher, **keywords,
        )

    whole_criterion = _ScriptedCriterion(NestTornadoCriterionConfig(), words)
    whole = run(_nested_double(whole_criterion), tmp_path / "whole", duration=160.0)
    assert whole["status"] == INTEGRATION_STATUS
    whole_block = whole["nest_tornado_criterion"]
    assert whole_block["episode_count"] == 2 and whole_block["samples"] == 320
    assert whole_block["episodes"][1]["start_model_time_s"] == 70.0
    assert whole_block["carried_from_restart"] is None

    first = run(_nested_double(_ScriptedCriterion(NestTornadoCriterionConfig(), words)),
                tmp_path / "part1", duration=80.0)
    restart = tmp_path / "part1" / "restarts" / "restart-000002"
    child_rows = _parse_scalar_table(
        (restart / "nest-child" / "restart-scheduler-scalars.tsv").read_text(encoding="ascii")
    )
    ledger_rows = {name: row for name, row in child_rows.items() if name.startswith("nest.tornado.")}
    assert ledger_rows["nest.tornado.samples"] == ("u64", "160")
    assert ledger_rows["nest.tornado.run"] == ("bool", "true")
    assert ledger_rows["nest.tornado.episodes"] == ("u64", "1")
    assert NestRestartState.from_scalars(child_rows).criterion == ledger_rows
    assert first["nest_tornado_criterion"]["episode_count"] == 1

    # Resume: the parent's words from its restart (the double carries no child
    # words to restore; the scripted sample reads only the clock), the ledger
    # from the child restart's rows, then the same 70 s.
    resumed_criterion = _ScriptedCriterion(NestTornadoCriterionConfig(), words)
    carried = resumed_criterion.restore_from_restart(ledger_rows)["carried_from_restart"]
    assert carried["samples"] == 160 and carried["qualifying_run_in_progress"]["start_model_time_s"] == 70.0
    resumed = _nested_double(resumed_criterion)
    resumed.restore_restart(read_cartesian_restart(restart, publisher=publisher,
                                                   into=tmp_path / "restored"))
    child = resumed.nest_driver.child
    child.restore_restart(
        read_cartesian_restart(restart / "nest-child", publisher=publisher,
                               into=tmp_path / "restored-child"),
        profile=nest_child_restart_profile(child.config, nparcels=None),
    )
    assert resumed.clock.model_seconds == 80.0
    assert resumed.nest_driver.child.clock.completed_steps == 160
    second = run(resumed, tmp_path / "part2", duration=160.0,
                 first_history_index=9, first_restart_number=3)
    block = second["nest_tornado_criterion"]
    for key in ("verdict", "genesis_model_time_s", "tornado_declared_model_time_s", "samples",
                "first_sample_model_time_s", "last_sample_model_time_s", "condition_counts",
                "qualifying_runs_started", "longest_qualifying_run_s", "episode_count",
                "episodes", "episodes_dropped", "stop_request", "stop_requests_withdrawn",
                "last_withdrawn_stop_request"):
        assert block[key] == whole_block[key], key
    assert block["carried_from_restart"]["samples"] == 160
    assert "carried" in block["verdict_scope"]
    [series] = block["time_series"]
    assert series["rows"] == 160
    header, rows = tc.read_time_series(series["path"])
    assert header["carried_from_restart"] == {name: list(row) for name, row in ledger_rows.items()}
    replay = tc.replay_time_series(series["path"])
    assert replay["rows_identical"] and replay["carried_from_restart"]["samples"] == 160
    for key in ("verdict", "genesis_model_time_s", "episode_count", "episodes", "condition_counts"):
        assert replay[key] == whole_block[key], key
    # The next child restart carries the whole run's ledger, as the
    # uninterrupted run's does.
    third = _parse_scalar_table((tmp_path / "part2" / "restarts" / "restart-000003" / "nest-child"
                                 / "restart-scheduler-scalars.tsv").read_text(encoding="ascii"))
    whole_third = _parse_scalar_table((tmp_path / "whole" / "restarts" / "restart-000003" / "nest-child"
                                       / "restart-scheduler-scalars.tsv").read_text(encoding="ascii"))
    assert {n: r for n, r in third.items() if n.startswith("nest.tornado.")} == \
        {n: r for n, r in whole_third.items() if n.startswith("nest.tornado.")}

    # Negative control: a fresh ledger at the resume.
    fresh = _nested_double(_ScriptedCriterion(NestTornadoCriterionConfig(), words))
    fresh.restore_restart(read_cartesian_restart(restart, publisher=publisher,
                                                 into=tmp_path / "restored-fresh"))
    fresh.nest_driver.child.restore_restart(
        read_cartesian_restart(restart / "nest-child", publisher=publisher,
                               into=tmp_path / "restored-fresh-child"),
        profile=nest_child_restart_profile(fresh.nest_driver.child.config, nparcels=None),
    )
    control = run(fresh, tmp_path / "control", duration=160.0,
                  first_history_index=9, first_restart_number=3)["nest_tornado_criterion"]
    assert control["episode_count"] == 1 and control["genesis_model_time_s"] == 80.5
    assert control["tornado_declared_model_time_s"] == 140.5 and control["samples"] == 160
    assert control["carried_from_restart"] is None
