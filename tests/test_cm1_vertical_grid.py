"""CM1 r21.1 vertical coordinate: one generator, four admitted branches.

Every assertion below is either a raw-word identity against a direct
transcription of ``param.F``, or a refusal that names a CM1 stop.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core.cartesian_door import build_runtime_config, require_grid_extent
from cm1core.cm1_vertical_grid import (
    ADMITTED_STRETCH_Z,
    PINNED_FORNBERG_F32,
    STRETCH_Z2_REFUSAL,
    VerticalGridError,
    build_vertical_levels,
    fornberg_extrapolation_weights,
    uniform_vertical_levels,
)
from cm1core.config import load_config, require_vertical_coordinate
from cm1core.errors import CM1Refusal
from cm1core.cuda_terrain_open_boundary import terrain_coordinate_profiles
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.stretch_z1_child_15km import (
    STRETCH_Z1_CHILD_15KM_PARAMETERS,
    STRETCH_Z1_CHILD_15KM_PROFILE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
UNIFORM_CASES = ((4, 1.0), (16, 100.0), (40, 500.0), (75, 200.0), (237, 75.0), (48, 333.0))


def _native_uniform(nk: int, dz: float) -> dict[str, np.ndarray]:
    """param.F:7789-7811 plus the :8145-8210 tail, written straight out."""

    f = np.float32
    dz32 = f(dz)
    zf = np.zeros(nk + 3, dtype=np.float32)
    for k in range(nk + 3):
        zf[k] = f(f(k - 1) * dz32)
    # param.F:8165-8166
    zf[0] = f(-zf[2])
    zf[nk + 2] = f(f(zf[nk + 1]) + f(f(zf[nk + 1]) - f(zf[nk])))
    zh = np.empty(nk + 2, dtype=np.float32)
    mh = np.empty(nk + 2, dtype=np.float32)
    for k in range(nk + 2):
        zh[k] = f(f(0.5) * f(f(zf[k + 1]) + f(zf[k])))
        mh[k] = f(dz32 / f(f(zf[k + 1]) - f(zf[k])))
    zh[0] = f(-zh[1])
    zh[nk + 1] = f(f(zh[nk]) + f(f(2.0) * f(f(zf[nk + 1]) - f(zh[nk]))))
    mf = np.empty(nk + 3, dtype=np.float32)
    for k in range(1, nk + 2):
        mf[k] = f(dz32 / f(f(zh[k]) - f(zh[k - 1])))
    mf[0], mf[nk + 2] = mf[1], mf[nk + 1]
    sigmaf = zf.copy()
    sigma = np.empty(nk + 2, dtype=np.float32)
    for k in range(nk + 2):
        sigma[k] = f(f(0.5) * f(f(sigmaf[k]) + f(sigmaf[k + 1])))
    return {
        "zf": zf, "zh": zh, "sigmaf": sigmaf, "sigma": sigma, "mh": mh, "mf": mf,
        "rmh": np.array([f(f(1.0) / f(value)) for value in mh], dtype=np.float32),
        "rmf": np.array([f(f(1.0) / f(value)) for value in mf], dtype=np.float32),
    }


@pytest.mark.parametrize("nk,dz", UNIFORM_CASES)
def test_stretch_z0_is_raw_bit_identical_to_the_uniform_source(nk: int, dz: float) -> None:
    """The whole point of the change: today's grid must not move one bit."""

    levels = uniform_vertical_levels(nk, dz)
    native = _native_uniform(nk, dz)
    for name, expected in native.items():
        assert getattr(levels, name).tobytes() == expected.tobytes(), name
    # param.F:7793,7802 set mf and mh to exactly one on the unstretched grid.
    assert np.all(np.asarray(levels.mh) == np.float32(1.0))
    assert np.all(np.asarray(levels.mf) == np.float32(1.0))
    assert np.all(np.asarray(levels.rmh) == np.float32(1.0))
    assert np.all(np.asarray(levels.rmf) == np.float32(1.0))
    # param.F:2608.
    assert levels.ztop_m == float(np.float32(np.float32(dz) * np.float32(nk)))
    assert levels.lowest_mass_level_m == float(np.float32(0.5 * np.float32(dz)))


@pytest.mark.parametrize("nk,dz", UNIFORM_CASES)
def test_uniform_fornberg_weights_are_computed_not_assumed(nk: int, dz: float) -> None:
    """(1.875, -1.25, 0.375) is a result of param.F:8985-9060, not a constant."""

    levels = uniform_vertical_levels(nk, dz)
    assert levels.cgs == PINNED_FORNBERG_F32
    assert levels.cgt == PINNED_FORNBERG_F32
    assert levels.fornberg_matches_pinned_kernels()
    profiles = terrain_coordinate_profiles(nk, dz)
    assert tuple(map(float, profiles["cgs"])) == PINNED_FORNBERG_F32
    assert tuple(map(float, profiles["cgt"])) == PINNED_FORNBERG_F32


def test_the_sealed_15km_child_grid_reproduces_its_pinned_profiles() -> None:
    """stretch_z=1 through the one generator is the already-sealed child."""

    p = STRETCH_Z1_CHILD_15KM_PARAMETERS
    levels = build_vertical_levels(
        stretch_z=1, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop, str_bot_m=p.str_bot,
        str_top_m=p.str_top, dz_bot_m=p.dz_bot, dz_top_m=p.dz_top,
    )
    import hashlib

    for name, expected in STRETCH_Z1_CHILD_15KM_PROFILE_SHA256.items():
        actual = hashlib.sha256(
            np.ascontiguousarray(getattr(levels, name), dtype=np.float32).tobytes()
        ).hexdigest()
        assert actual == expected, name
    # The tornado-grade properties the whole task exists for.
    assert levels.lowest_mass_level_m == 10.0
    assert levels.spacings_m[:11] == (20.0,) * 11
    assert levels.levels_below_m(1000.0) == 22
    assert levels.ztop_m == 15000.0
    # The bottom and top segments stay uniform, so the compiled kernels'
    # inline quadratic is still this grid's own.
    assert levels.fornberg_matches_pinned_kernels()


def test_the_explicit_branches_reproduce_the_analytic_one_word_for_word() -> None:
    """param.F:8005-8132 and :7819-7957 must agree on the same column."""

    p = STRETCH_Z1_CHILD_15KM_PARAMETERS
    analytic = build_vertical_levels(
        stretch_z=1, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop, str_bot_m=p.str_bot,
        str_top_m=p.str_top, dz_bot_m=p.dz_bot, dz_top_m=p.dz_top,
    )
    # param.F:8085-8089 reads the w levels themselves.
    w_levels = build_vertical_levels(
        stretch_z=4, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop,
        levels_m=analytic.face_heights_m,
    )
    assert w_levels.sha256 == analytic.sha256
    # param.F:8019-8026 rebuilds the faces from the scalar levels by the
    # alternating recursion sigmaf(k+1) = 2*sigma(k) - sigmaf(k).  That
    # recursion carries every rounding error upward with alternating sign, so
    # on this 75-level column it does NOT return the faces it came from: the
    # generated top is 14999.998 m, not 15000 m.  CM1 r19.8 added stretch_z=4
    # for exactly this reason, and param.F:8037 then makes that drifted number
    # the model top rather than the declared one.  A deck that means 15000 m
    # must say so with w levels.
    with pytest.raises(VerticalGridError, match=r"param\.F:8037"):
        build_vertical_levels(
            stretch_z=3, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop,
            levels_m=analytic.mass_heights_m,
        )
    drifted = build_vertical_levels(
        stretch_z=3, nk=p.nk, dz_m=p.dz, ztop_m=14999.998046875,
        levels_m=analytic.mass_heights_m,
    )
    assert drifted.ztop_m == 14999.998046875
    assert drifted.sha256 != analytic.sha256
    # On a column short enough for the recursion to stay exact the two agree.
    short = build_vertical_levels(stretch_z=0, nk=8, dz_m=250.0, ztop_m=2000.0)
    rebuilt = build_vertical_levels(
        stretch_z=3, nk=8, dz_m=250.0, ztop_m=2000.0, levels_m=short.mass_heights_m,
    )
    assert rebuilt.sha256 == short.sha256


def test_a_column_that_stretches_at_the_ground_leaves_the_pinned_quadratic() -> None:
    """This is the numerics the recomputation exists for.

    ``param.F:8985-9060`` fits a quadratic through the lowest three scalar
    levels.  Equally spaced, that quadratic is (1.875, -1.25, 0.375).  Once the
    first layers differ in depth it is not, and reusing the old triple would
    extrapolate ``rf0`` at the ground from the wrong parabola.
    """

    faces = [0.0, 20.0, 60.0, 140.0, 300.0, 620.0, 940.0, 1260.0, 1580.0, 1900.0, 2220.0, 2540.0]
    levels = build_vertical_levels(stretch_z=4, nk=11, dz_m=100.0, ztop_m=2540.0, levels_m=faces)
    assert levels.cgs != PINNED_FORNBERG_F32
    assert not levels.fornberg_matches_pinned_kernels()
    # Fornberg's zeroth-derivative weights interpolate: they sum to one and
    # reproduce the fitted function at the extrapolation point, the ground.
    assert abs(sum(levels.cgs) - 1.0) < 1e-5
    ground = sum(
        weight * height for weight, height in zip(levels.cgs, levels.mass_heights_m[:3])
    )
    assert abs(ground) < 1e-3
    profiles = terrain_coordinate_profiles(11, 100.0, levels=levels)
    assert tuple(map(float, profiles["cgs"])) == levels.cgs
    assert tuple(map(float, profiles["cgt"])) == levels.cgt
    # param.F:8170: mh is dz over the layer depth, no longer one.
    assert float(levels.mh[1]) == pytest.approx(100.0 / 20.0)
    assert float(levels.mh[11]) == pytest.approx(100.0 / 320.0)


def test_terrain_open_profiles_keep_their_uniform_words_and_take_declared_ones() -> None:
    uniform = terrain_coordinate_profiles(75, 200.0)
    declared = terrain_coordinate_profiles(75, 200.0, levels=uniform_vertical_levels(75, 200.0))
    for name in ("sigma", "sigmaf", "rds", "rdsf", "cgs", "cgt"):
        assert uniform[name].tobytes() == declared[name].tobytes(), name
    p = STRETCH_Z1_CHILD_15KM_PARAMETERS
    levels = build_vertical_levels(
        stretch_z=1, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop, str_bot_m=p.str_bot,
        str_top_m=p.str_top, dz_bot_m=p.dz_bot, dz_top_m=p.dz_top,
    )
    stretched = terrain_coordinate_profiles(p.nk, p.dz, levels=levels)
    # init_terrain.F:292-298 over this column's own spacings.
    assert float(stretched["rdsf"][1]) == pytest.approx(1.0 / 20.0)
    assert float(stretched["rdsf"][75]) == pytest.approx(1.0 / 320.0)
    assert float(stretched["sigma"][1]) == 10.0
    with pytest.raises(ValueError, match="dz reference"):
        terrain_coordinate_profiles(p.nk, 50.0, levels=levels)
    with pytest.raises(ValueError, match="nk scalar levels"):
        terrain_coordinate_profiles(p.nk - 1, p.dz, levels=levels)


def test_every_refusal_names_a_cm1_stop() -> None:
    assert ADMITTED_STRETCH_Z == (0, 1, 3, 4)
    with pytest.raises(VerticalGridError) as caught:
        build_vertical_levels(stretch_z=2, nk=75, dz_m=200.0, ztop_m=15000.0)
    assert str(caught.value) == STRETCH_Z2_REFUSAL
    assert "param.F:9897-9937" in STRETCH_Z2_REFUSAL
    with pytest.raises(VerticalGridError, match="branches"):
        build_vertical_levels(stretch_z=5, nk=75, dz_m=200.0, ztop_m=15000.0)
    # param.F:2608 owns the unstretched top.
    with pytest.raises(VerticalGridError, match="param.F:2608"):
        build_vertical_levels(stretch_z=0, nk=40, dz_m=500.0, ztop_m=19000.0)
    # param.F:7828-7877 feasibility: 200 m does not divide by 30 m.
    with pytest.raises(VerticalGridError, match="0.01 m feasibility"):
        build_vertical_levels(
            stretch_z=1, nk=75, dz_m=100.0, ztop_m=15000.0, str_bot_m=200.0,
            str_top_m=7000.0, dz_bot_m=30.0, dz_top_m=320.0,
        )
    # param.F:8037 and :8107 own the explicit branches' tops.
    with pytest.raises(VerticalGridError, match="param.F:8105"):
        build_vertical_levels(
            stretch_z=4, nk=4, dz_m=100.0, ztop_m=999.0,
            levels_m=[0.0, 100.0, 200.0, 300.0, 400.0],
        )
    with pytest.raises(VerticalGridError, match="explicit"):
        build_vertical_levels(stretch_z=3, nk=4, dz_m=100.0, ztop_m=400.0, levels_m=[50.0])
    with pytest.raises(VerticalGridError, match="ground"):
        build_vertical_levels(
            stretch_z=4, nk=4, dz_m=100.0, ztop_m=400.0,
            levels_m=[10.0, 100.0, 200.0, 300.0, 400.0],
        )
    # param.F:8212-8224 dummy check.
    with pytest.raises(VerticalGridError, match="non-positive"):
        build_vertical_levels(
            stretch_z=4, nk=4, dz_m=100.0, ztop_m=400.0,
            levels_m=[0.0, 200.0, 200.0, 300.0, 400.0],
        )


def test_fornberg_helper_refuses_a_column_it_cannot_fit() -> None:
    levels = uniform_vertical_levels(8, 100.0)
    with pytest.raises(VerticalGridError, match="four scalar levels"):
        fornberg_extrapolation_weights(levels.sigma, levels.sigmaf, nk=3, top=False)


def test_geometry_keeps_its_uniform_identity_and_carries_a_declared_one() -> None:
    uniform = CartesianGridGeometry(96, 96, 40, 500.0, 500.0, 500.0)
    assert uniform.vertical is None
    assert uniform.stretch_z == 0 and uniform.uniform_z
    assert uniform.record() == {
        "nx": 96, "ny": 96, "nz": 40, "dx_m": 500.0, "dy_m": 500.0,
        "dz_m": 500.0, "ztop_m": 20000.0,
    }
    assert uniform.lowest_mass_level_m == 250.0
    assert uniform.highest_mass_level_m == 19750.0
    assert uniform.levels.sha256 == uniform_vertical_levels(40, 500.0).sha256
    p = STRETCH_Z1_CHILD_15KM_PARAMETERS
    levels = build_vertical_levels(
        stretch_z=1, nk=p.nk, dz_m=p.dz, ztop_m=p.ztop, str_bot_m=p.str_bot,
        str_top_m=p.str_top, dz_bot_m=p.dz_bot, dz_top_m=p.dz_top,
    )
    stretched = CartesianGridGeometry(300, 300, 75, 50.0, 50.0, 100.0, levels)
    assert stretched.ztop_m == 15000.0
    assert stretched.lowest_mass_level_m == 10.0
    assert stretched.record()["vertical"]["levels_below_1000m"] == 22
    with pytest.raises(ValueError, match="level count"):
        CartesianGridGeometry(300, 300, 74, 50.0, 50.0, 100.0, levels)
    with pytest.raises(ValueError, match="dz reference"):
        CartesianGridGeometry(300, 300, 75, 50.0, 50.0, 200.0, levels)
    with pytest.raises(TypeError, match="VerticalLevels"):
        CartesianGridGeometry(300, 300, 75, 50.0, 50.0, 100.0, object())
    # replace() on a uniform grid still rebuilds, so the stencil refusals that
    # existed before the coordinate did still fire in their own order.
    assert replace(uniform, nz=3).stretch_z == 0


def test_the_shipped_tornado_grade_deck_is_admitted_and_is_tornado_grade() -> None:
    config = load_config(ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml")
    assert config.grid.stretch_z == 1
    assert (config.grid.ztop_m, config.grid.dz_bot_m, config.grid.dz_top_m) == (
        15000.0, 20.0, 320.0,
    )
    grid = require_grid_extent(config)
    assert grid.ztop_m == 15000.0
    assert grid.lowest_mass_level_m == 10.0
    assert grid.spacings_m[0] == 20.0
    assert sum(1 for z in grid.mass_heights_m if z < 1000.0) == 22
    assert (config.grid.dx_m, config.grid.dy_m) == (50.0, 50.0)
    assert config.grid.nx * config.grid.dx_m == 60000.0


def test_the_cartesian_runtime_door_admits_the_tornado_grade_grid() -> None:
    """The door that refused CM1-CARTESIAN-UNIFORM-DZ-OPERATORS now admits it.

    Its four named operators each have a profile-bound successor, so the guard
    is retired and this deck reaches a RuntimeConfig carrying its coordinate.
    """

    from cm1core.cartesian_door import cartesian_vertical_operator_record

    config = load_config(ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml")
    runtime = build_runtime_config(config)
    grid = runtime.grid
    assert grid.stretch_z == 1
    assert grid.vertical is not None
    assert grid.ztop_m == 15000.0
    assert grid.spacings_m[0] == 20.0
    # param.F:8170: dz is the metric reference, not any layer's thickness.
    assert grid.dz_m == 100.0 != grid.spacings_m[0]
    record = cartesian_vertical_operator_record(grid)
    assert record["family"] == "profile-bound"
    selected = {row["selected"] for row in record["operators"]}
    assert selected == {
        "cuda_gm1_child_psolver3_open",
        "cuda_gm1_child_stretched_momentum",
        "cuda_gm1_child_stretched_scalar",
        "cuda_gm1_child_stretched_moist",
        # The Cartesian lane is terrain_flag=.false., so its closure runs
        # CM1's flat calcdef/turb*/TKE-advection statements too.
        "cuda_gm1_flat_stretched_closure",
    }
    # The unstretched decks are untouched and keep the scalar-dz family.
    uniform = load_config(ROOT / "examples" / "cartesian-uniform-grid.toml")
    uniform_grid = build_runtime_config(uniform).grid
    assert uniform_grid.vertical is None
    uniform_record = cartesian_vertical_operator_record(uniform_grid)
    assert uniform_record["family"] == "uniform-dz"
    assert {row["selected"] for row in uniform_record["operators"]} == {
        "cuda_psolver3_terrain_open",
        "cuda_momentum_advection5",
        "cuda_scalar_moist_rk",
        "cuda_moist_stage_bridge",
        # param.F:8170,8176 make every mh/mf word exactly one on a uniform
        # column, so its closure keeps the rds/rdsf image it has always run.
        "cuda_gm1_child_stretched_sgs_tke",
    }


def test_the_door_refuses_a_rectangular_grid_by_the_kernel_that_is_wrong() -> None:
    """turb.F:4548-4605 scales the y stress divergence by rdy, not rdx."""

    import tomllib

    from cm1core.config import parse_mapping

    raw = tomllib.loads((ROOT / "examples" / "cartesian-uniform-grid.toml").read_text())
    raw["grid"]["dy_m"] = 250.0
    config = parse_mapping(raw)
    with pytest.raises(CM1Refusal) as caught:
        build_runtime_config(config)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-SQUARE-HORIZONTAL-SPACING"
    assert detail.option == "grid.dy_m"
    assert "cm1_sgs_turbu" in detail.reason and "turb.F:4548-4605" in detail.reason


@pytest.mark.parametrize(
    "key,value,code,option",
    [
        ("stretch_z", 2, "CM1-CFG-UNSUPPORTED-VALUE", "grid.stretch_z"),
        ("stretch_z", 7, "CM1-CFG-UNSUPPORTED-VALUE", "grid.stretch_z"),
        ("stretch_z", True, "CM1-CFG-INTEGER", "grid.stretch_z"),
        ("stretch_z", 0, "CM1-CFG-STRETCH-CONTROL-UNUSED", "grid.ztop_m"),
        ("dz_bot_m", 0.0, "CM1-CFG-STRETCH-CONTROL-MISSING", "grid.dz_bot_m"),
        ("dz_top_m", -5.0, "CM1-CFG-NUMERIC-RANGE", "grid.dz_top_m"),
        ("levels_m", [100.0], "CM1-CFG-STRETCH-CONTROL-UNUSED", "grid.levels_m"),
        ("levels_m", [100.0, 50.0], "CM1-CFG-LEVEL-ARRAY", "grid.levels_m"),
        # param.F:7867-7877: nk1+nk2+nk3 must equal nk.
        ("str_top_m", 20000.0, "CM1-CFG-VERTICAL-COORDINATE", "grid.stretch_z"),
    ],
)
def test_the_grid_table_refuses_by_key_and_value(
    key: str, value: object, code: str, option: str
) -> None:
    import tomllib

    raw = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text(
            encoding="utf-8"
        )
    )
    raw["grid"][key] = value
    from cm1core.config import parse_mapping

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == code
    assert caught.value.detail.option == option


def test_stretch_z2_is_refused_at_the_door_with_its_own_reason() -> None:
    import tomllib

    raw = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text(
            encoding="utf-8"
        )
    )
    raw["grid"]["stretch_z"] = 2
    from cm1core.config import parse_mapping

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert "param.F:9897-9937" in caught.value.detail.reason


def test_the_module_opens_no_numerical_runtime_on_import() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import cm1core.cm1_vertical_grid; "
            "assert 'cupy' not in sys.modules; assert 'gpuwm' not in sys.modules",
        ],
        check=True,
        env=env,
    )


def test_a_coordinate_the_compiled_kernels_cannot_extrapolate_is_named() -> None:
    """The guard that makes the recomputation safe to ship.

    Six compiled GM1 kernel families carry ``(1.875, -1.25, 0.375)`` inline as
    the lower and upper boundary extrapolation.  That triple is only this
    grid's own ``param.F:8985-9060`` answer while the three levels at each end
    are equally spaced.  A deck whose column stretches inside those three is
    refused by name rather than silently extrapolated from the wrong parabola.
    """

    import tomllib

    from cm1core.config import parse_mapping

    raw = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text(
            encoding="utf-8"
        )
    )
    nk = 12
    faces = [0.0, 20.0, 60.0, 140.0]
    while len(faces) < nk + 1:
        faces.append(faces[-1] + 320.0)
    raw["grid"].update(
        nx=64, ny=64, nz=nk, dz_m=100.0, stretch_z=4, ztop_m=0.0,
        str_bot_m=0.0, str_top_m=0.0, dz_bot_m=0.0, dz_top_m=0.0, levels_m=faces,
    )
    raw["closures"]["upper_damping_depth_m"] = 1000.0
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-VERTICAL-FORNBERG"
    assert "cuda_gm1_forecast" in detail.reason
    assert "rf0" in detail.reason
    assert "1.875" in detail.reason
    # The same column with its lowest three layers equal is admitted.
    faces = [0.0, 20.0, 40.0, 60.0]
    while len(faces) < nk + 1:
        faces.append(faces[-1] + 320.0)
    raw["grid"]["levels_m"] = faces
    config = parse_mapping(raw)
    assert config.grid.stretch_z == 4
    assert require_grid_extent(config).lowest_mass_level_m == 10.0


@pytest.mark.parametrize(
    "dz",
    [500.0, 333.3, 100.0, 125.0, 250.0, 1000.0 / 3, 1000.0, 41.6666666, 0.1, 7.7],
)
def test_a_uniform_column_is_admitted_whatever_binary32_spelling_its_dz_has(
    dz: float,
) -> None:
    """param.F:8990-8993 divides the spacing out, so it cannot decide this.

    On a uniform column the Fornberg nodes are ``alpha = (0.5, 1.5, 2.5)`` and
    the extrapolation point is the origin for *every* ``dz``, so the weights
    are analytically ``(1.875, -1.25, 0.375)`` and the grid is exactly the one
    the compiled kernels' inline quadratic was written for.

    This is the pin for a guard that used to compare the recomputed f32 words
    to that triple bit for bit.  ``dz = 333.3`` has no exact binary32
    spelling, so a face near 13 km rounds by an ulp, the recomputation returns
    seven digits of representation noise instead of the analytic answer, and
    a correct uniform grid was refused while ``dz = 500.0`` -- the same
    geometry -- passed.
    """

    import tomllib

    from cm1core.config import parse_mapping

    levels = uniform_vertical_levels(40, dz)
    assert levels.lowest_layers_equally_spaced
    assert levels.highest_layers_equally_spaced
    assert levels.fornberg_matches_pinned_kernels()
    spread = max(levels.spacings_m) - min(levels.spacings_m)
    assert spread <= 2.0 * np.spacing(np.float32(dz * 40))  # one depth, respelled

    raw = tomllib.loads(
        (ROOT / "examples" / "cartesian-uniform-grid.toml").read_text(encoding="utf-8")
    )
    raw["grid"]["dz_m"] = dz
    # param.F:8633-8649 stops CM1 when the Rayleigh layer covers more than half
    # the column, so keep this deck's damper inside the shortest top tested.
    raw["closures"]["upper_damping_depth_m"] = 0.25 * dz * raw["grid"]["nz"]
    config = parse_mapping(raw)
    assert config.grid.dz_m == dz
    assert require_grid_extent(config).levels.fornberg_matches_pinned_kernels()


def test_the_recomputed_weights_stay_cm1s_own_rounded_words() -> None:
    """The admission above is not bought by rewriting ``cgs``/``cgt``.

    ``cuda_terrain_open_boundary``'s CPU reference consumes those words and has
    to keep matching native CM1, which computes them in f32 from its own f32
    coordinate.  The guard reads a separate, geometric predicate instead.
    """

    levels = uniform_vertical_levels(40, 333.3)
    assert levels.cgs == PINNED_FORNBERG_F32
    assert levels.cgt != PINNED_FORNBERG_F32
    assert levels.cgt == pytest.approx(PINNED_FORNBERG_F32, rel=1e-4)
    assert levels.fornberg_matches_pinned_kernels()
    profiles = terrain_coordinate_profiles(40, 333.3, levels=levels)
    assert tuple(map(float, profiles["cgt"])) == levels.cgt


def test_the_pinned_quadratic_follows_the_stretch_z1_segment_depths() -> None:
    """param.F:7925-7937's three segments decide each end, not the f32 words.

    ``zf(k)=(k-1)*dz_bot`` for ``k<=nk1+1`` and ``zf(k)=zf(k-1)+dz_top`` above
    ``nk1+nk2+1``; in between ``:7929-7930`` gives layer ``j`` the depth
    ``dz_bot + 2*c2*nominal**2*(j-1)``, whose first member is exactly
    ``dz_bot``.  So the lowest three layers share one depth once ``nk1 >= 2``
    and the highest three once ``nk3 >= 2``, and the remedy's ``>= 3`` recipe
    is a sufficient recipe rather than the boundary.
    """

    def column(**kwargs: float) -> object:
        return build_vertical_levels(stretch_z=1, dz_m=100.0, **kwargs)  # type: ignore[arg-type]

    # nk1=10, nk3=25: the shipped tornado-grade column.
    admitted = column(nk=75, ztop_m=15000.0, str_bot_m=200.0, str_top_m=7000.0,
                      dz_bot_m=20.0, dz_top_m=320.0)
    assert admitted.fornberg_matches_pinned_kernels()
    # nk1=2: the third layer is the first stretched one, and it is dz_bot.
    edge = column(nk=67, ztop_m=14840.0, str_bot_m=40.0, str_top_m=6840.0,
                  dz_bot_m=20.0, dz_top_m=320.0)
    assert edge.spacings_m[:3] == (20.0, 20.0, 20.0)
    assert edge.fornberg_matches_pinned_kernels()
    # nk1=1: the third layer is the second stretched one, and it is not.
    graded_bottom = column(nk=66, ztop_m=14820.0, str_bot_m=20.0, str_top_m=6820.0,
                           dz_bot_m=20.0, dz_top_m=320.0)
    assert not graded_bottom.lowest_layers_equally_spaced
    assert graded_bottom.highest_layers_equally_spaced
    assert not graded_bottom.fornberg_matches_pinned_kernels()
    # nk3=1: the third layer down is 7.7 m shallower than the two above it.
    graded_top = column(nk=51, ztop_m=7320.0, str_bot_m=200.0, str_top_m=7000.0,
                        dz_bot_m=20.0, dz_top_m=320.0)
    assert graded_top.lowest_layers_equally_spaced
    assert not graded_top.highest_layers_equally_spaced
    assert not graded_top.fornberg_matches_pinned_kernels()


def test_a_graded_end_triplet_is_still_refused_by_the_result_it_prevents() -> None:
    """The named defect survives the fix: the guard still fires on real grading."""

    import tomllib

    from cm1core.config import parse_mapping

    raw = tomllib.loads(
        (ROOT / "examples" / "cartesian-uniform-grid.toml").read_text(encoding="utf-8")
    )
    nk = 51
    raw["grid"].update(
        nz=nk, dz_m=100.0, stretch_z=1, ztop_m=7320.0, str_bot_m=200.0,
        str_top_m=7000.0, dz_bot_m=20.0, dz_top_m=320.0,
    )
    raw["closures"]["upper_damping_depth_m"] = 1000.0
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-VERTICAL-FORNBERG"
    assert detail.option == "grid.stretch_z"
    # The refusal names the concrete wrong result and this column's own depths.
    for fragment in (
        "highest three layers",
        "320.0, 320.0 and 312.30712890625 m deep",
        "cuda_gm1_forecast",
        "cuda_storm_diagnostics",
        "rf0",
        "surface stress",
        "(1.875, -1.25, 0.375)",
    ):
        assert fragment in detail.reason, fragment
    assert "str_bot_m >= 3*dz_bot_m" in detail.remedy


# ---------------------------------------------------------------------------
# The generator against CM1 r21.1 itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_name,evidence_name",
    [
        ("stretched-column-32", "stretched-column-32"),
        ("explicit-w-levels-32", "explicit-w-levels-32"),
        ("explicit-scalar-levels-32", "explicit-scalar-levels-32"),
    ],
)
def test_every_stretched_branch_reproduces_cm1s_own_written_column(
    case_name: str, evidence_name: str
) -> None:
    """One check, three branches: param.F:7819-7953, :8072-8132 and :8005-8064.

    Each ``evidence/oracle/<case>`` holds the GrADS descriptors a real
    ``cm1.exe`` run of that case wrote beside its history.  ``stretch_z=3`` is
    the interesting one: CM1 reaches its faces through ``param.F:8025``'s
    alternating recursion rather than reading them, so this holds the
    recursion's own output and not the list that was fed to it.
    """

    from cm1core.oracle.grads import read_ctl

    evidence = ROOT / "evidence" / "oracle" / evidence_name
    grid = require_grid_extent(_config_from_case(case_name, evidence))
    assert grid.stretch_z != 0
    for name, heights in (
        ("cm1out_s.ctl", grid.mass_heights_m),
        ("cm1out_w.ctl", grid.face_heights_m),
    ):
        dataset = read_ctl(evidence / name)
        assert dataset.levels, f"{name} carries no explicit levels"
        cm1_m = np.array(dataset.levels, dtype=np.float64) * 1000.0
        gm1_m = np.array(heights, dtype=np.float64)[: len(cm1_m)]
        assert len(cm1_m) == (grid.nz if name.endswith("s.ctl") else grid.nz + 1)
        printed = (gm1_m / 1000.0).astype(np.float32).astype(np.float64) * 1000.0
        assert np.abs(printed - cm1_m).max() < 0.002, (case_name, name)


def _config_from_case(case_name: str, evidence: Path):
    """The gpuwm-cm1 document one differential case translates to."""

    import tomllib

    from cm1core.config import parse_mapping
    from cm1core.oracle.spec import load_case

    case = load_case(ROOT / "examples" / "oracle" / f"{case_name}.differential.toml")
    return parse_mapping(tomllib.loads(case.gm1_toml(evidence / "namelist.input")))


def test_generated_stretch_z1_column_is_cm1s_own_written_column() -> None:
    """CM1 r21.1 wrote these heights; this port's generator reproduces them.

    ``evidence/oracle/stretched-column-32`` holds the GrADS descriptors a real
    ``cm1.exe`` run of ``examples/oracle/stretched-column-32.differential.toml``
    wrote beside its history: ``zdef 40 levels`` for the scalar levels
    (``param.F:8162-8168``'s ``zh``) and 41 for the w levels (``:7925-7934``'s
    ``zf``).  They are the oracle for the coordinate itself, and they cost no
    GPU to check.

    The descriptor prints kilometres in seven significant digits, so the
    comparison is held to that -- half a millimetre at four kilometres -- and
    not to raw words CM1 never wrote.
    """

    from cm1core.oracle.grads import read_ctl

    evidence = ROOT / "evidence" / "oracle" / "stretched-column-32"
    config = load_config_from_case()
    grid = require_grid_extent(config)
    assert grid.stretch_z == 1

    for name, heights in (
        ("cm1out_s.ctl", grid.mass_heights_m),
        ("cm1out_w.ctl", grid.face_heights_m),
    ):
        dataset = read_ctl(evidence / name)
        assert dataset.levels, f"{name} carries no explicit levels"
        cm1_m = np.array(dataset.levels, dtype=np.float64) * 1000.0
        gm1_m = np.array(heights, dtype=np.float64)[: len(cm1_m)]
        assert len(cm1_m) == (grid.nz if name.endswith("s.ctl") else grid.nz + 1)
        # What the descriptor can express: the km value rounded to binary32.
        printed = (gm1_m / 1000.0).astype(np.float32).astype(np.float64) * 1000.0
        assert np.abs(printed - cm1_m).max() < 0.002, name

    # The descriptor is not uniform, which is the whole point: a zdef ...
    # linear column would have carried no levels at all.
    scalar = read_ctl(evidence / "cm1out_s.ctl")
    spacings = np.diff(np.array(scalar.levels)) * 1000.0
    assert round(float(spacings.min()), 3) == 100.0
    assert round(float(spacings.max()), 3) == 300.0


def load_config_from_case():
    """The gpuwm-cm1 document the differential case translates to.

    The translation starts from the same CM1 deck the run started from, which
    is recorded beside the descriptors, so this reads one configuration and
    not two that have to be kept in step by hand.
    """

    import tomllib

    from cm1core.config import parse_mapping
    from cm1core.oracle.spec import load_case

    case = load_case(ROOT / "examples" / "oracle" / "stretched-column-32.differential.toml")
    base = ROOT / "evidence" / "oracle" / "stretched-column-32" / "namelist.input"
    return parse_mapping(tomllib.loads(case.gm1_toml(base)))
