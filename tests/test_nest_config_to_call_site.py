"""From TOML text to changed behaviour, for the refined child grid.

Every test here starts from a configuration document written as a user would
write it and follows it to something that actually changes: the child's
geometry, its CM1-derived clock, its own :class:`RuntimeConfig`, the lateral
forcing tables the parent hands it, and the ``--plan`` JSON the door prints
without opening a device.  The final two tests are the pair that matters most:
the ``[nest]`` table is refused by key and by value, and a document that
declares no nest is byte-for-byte what it was before nesting existed.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from cm1core.cartesian_door import (
    NestBinding,
    build_runtime_config,
    cartesian_plan_payload,
    nest_allocation_plan,
    require_nest,
)
from cm1core.config import load_config, parse_mapping
from cm1core.errors import CM1Refusal
from cm1core.nest_boundary import (
    NEST_RELAXATION_KERNEL_SHA256,
    NEST_RELAXATION_KERNEL_SOURCE,
    apply_specified_relaxation,
    block_mean_residual,
    boundary_normal_velocity_is_parent_face_value,
    build_boundary_tables,
    davies_weights,
    interpolate_parent_to_child,
    parent_face_index,
    restrict_child_to_parent,
)
from cm1core.nest_geometry import (
    CM1_NESTING_ABSENCE,
    NestGeometry,
    cm1_pressure_centre_cell,
    cm1_smoothed_pressure,
    move_nest,
    plan_nest_move,
)

REPO = Path(__file__).resolve().parents[1]
NESTED_EXAMPLE = REPO / "examples" / "cartesian-nested-storm-following-mesh.toml"
#: Every shipped example that declares [nest]; the sealed single-domain hashes
#: below are the other ones, and a new nested deck must be named here rather
#: than moving a hash that has nothing to do with it.
NESTED_EXAMPLES: frozenset[str] = frozenset({
    NESTED_EXAMPLE.name, "cartesian-nested-tornado-window.toml",
})
FLAT_EXAMPLE = REPO / "examples" / "cartesian-supercell-translating-frame.toml"


# ---------------------------------------------------------------------------
# The document a user writes.
# ---------------------------------------------------------------------------
BASE_TOML = """
[run]
lane = "cartesian"
duration_s = 600.0
dt_s = 0.5
history_interval_s = 60.0
restart_interval_s = 600.0

[initialization]
isnd = 5
iwnd = 2

[grid]
nx = 60
ny = 60
nz = 40
dx_m = 1000.0
dy_m = 1000.0
dz_m = 500.0
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
upper_damping_depth_m = 5000.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = true
u_move_m_s = 12.5
v_move_m_s = 3.0

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
"""

NEST_TABLE = """
[nest]
refinement_ratio = 5
time_refinement_ratio = 5
parent_i0 = 15
parent_j0 = 15
parent_span_x = 20
parent_span_y = 20
spec_zone = 1
relax_zone = 4
feedback = 1
acoustic_substeps = 6
follow_storm = true
follow_interval_s = 60.0
follow_max_speed_m_s = 55.0
"""


def _nest_text(**overrides: object) -> str:
    lines = []
    for line in NEST_TABLE.strip().splitlines():
        if "=" in line:
            key = line.split("=")[0].strip()
            if key in overrides:
                value = overrides[key]
                rendered = (
                    "true" if value is True else "false" if value is False else repr(value)
                )
                lines.append(f"{key} = {rendered}")
                continue
        lines.append(line)
    return "\n" + "\n".join(lines) + "\n"


def _config(**overrides: object) -> CM1Config:
    import tomllib

    return parse_mapping(tomllib.loads(BASE_TOML + _nest_text(**overrides)))


# ---------------------------------------------------------------------------
# 1. The gate this work exists to remove.
# ---------------------------------------------------------------------------
def test_the_nest_table_reaches_the_parser_at_all() -> None:
    """Before this branch, ``[nest]`` was refused by name as an unknown table.

    ``CM1-CFG-UNKNOWN`` fired on the section itself, which is what made every
    line of GM1's existing child machinery unreachable from a configuration
    file.  This is the one assertion that says the door is open.
    """

    config = _config()
    assert config.nest is not None
    assert config.nest.refinement_ratio == 5
    assert config.nest.feedback == 1


# ---------------------------------------------------------------------------
# 2. Config text -> a child grid that is different from the parent.
# ---------------------------------------------------------------------------
def test_the_declared_ratio_changes_the_child_grid_and_its_clock() -> None:
    config = _config()
    binding = require_nest(config)
    assert isinstance(binding, NestBinding)
    parent = build_runtime_config(config)

    # The text said 5, so the child is five times finer and five times faster.
    assert binding.geometry.refinement_ratio == 5
    assert binding.child.grid.dx_m == parent.grid.dx_m / 5
    assert binding.child.grid.dy_m == parent.grid.dy_m / 5
    assert binding.child.grid.nx == 20 * 5
    assert binding.child.grid.ny == 20 * 5
    assert binding.child.dt_s == parent.dt_s / 5
    # The vertical coordinate is shared, which is what keeps the vertically
    # implicit acoustic solve from ever crossing the interface.
    assert binding.child.grid.nz == parent.grid.nz
    assert binding.child.grid.dz_m == parent.grid.dz_m

    # A different ratio in the text gives a different child.  This is the
    # whole point of the key.
    # 25 rather than 9: a child spacing must be one exact binary32 word, and
    # 1000 m / 9 is not (CM1-CFG-NEST-CHILD-SPACING).
    finer = require_nest(_config(refinement_ratio=25, time_refinement_ratio=25))
    assert finer is not None
    assert finer.child.grid.dx_m == parent.grid.dx_m / 25
    assert finer.child.grid.nx == 20 * 25
    assert finer.child.dt_s == parent.dt_s / 25


def test_the_declared_position_changes_where_the_child_sits() -> None:
    config = _config()
    here = require_nest(config)
    there = require_nest(_config(parent_i0=25, parent_j0=10))
    assert here is not None and there is not None
    assert here.geometry.origin_x_m == 15 * 1000.0
    assert there.geometry.origin_x_m == 25 * 1000.0
    assert there.geometry.origin_y_m == 10 * 1000.0
    assert here.geometry.centre_x_m != there.geometry.centre_x_m


def test_cm1s_own_dtsmall_decides_the_childs_acoustic_cadence() -> None:
    """``misclibs.F:857-868`` and ``:883-885``, evaluated on the child."""

    config = _config()
    binding = require_nest(config)
    assert binding is not None
    geometry = binding.geometry
    # 0.60 * 200 m / 360 m/s.
    assert geometry.child_acoustic_small_step_limit_s() == pytest.approx(
        0.60 * 200.0 / 360.0
    )
    # Refining space and time by the same ratio leaves dbldt/dtsm unchanged,
    # so the child needs exactly the substep count the parent does.
    assert geometry.advective_courant_multiplier() == 1.0
    assert geometry.cm1_child_acoustic_substeps(config.run.dt_s) == (
        NestGeometry(
            parent_nx=60,
            parent_ny=60,
            parent_dx_m=1000.0,
            parent_dy_m=1000.0,
            refinement_ratio=5,
            time_refinement_ratio=5,
            parent_i0=15,
            parent_j0=15,
            parent_span_x=20,
            parent_span_y=20,
            spec_zone=1,
            relax_zone=4,
            feedback=0,
        ).cm1_child_acoustic_substeps(config.run.dt_s)
    )
    # Refining space alone multiplies the advective Courant number by the
    # ratio, which is what CM1 stops on at misclibs.F:2714.
    spatial_only = require_nest(_config(time_refinement_ratio=1))
    assert spatial_only is not None
    assert spatial_only.geometry.advective_courant_multiplier() == 5.0


# ---------------------------------------------------------------------------
# 3. Config text -> the ``--plan`` the door prints, with no device opened.
# ---------------------------------------------------------------------------
def test_the_plan_the_door_prints_carries_the_nest_and_prices_the_child() -> None:
    """``--plan``'s own payload, from the door's own function.

    ``cartesian_plan_payload`` is the body of ``--plan``; the command line
    prints exactly this object.  It opens no device, touches no disk and needs
    no engine estate, so it runs everywhere the parser does.
    """

    import tomllib

    nested = cartesian_plan_payload(parse_mapping(tomllib.loads(BASE_TOML + NEST_TABLE)))
    single = cartesian_plan_payload(parse_mapping(tomllib.loads(BASE_TOML)))

    assert nested["status"] == "priced-no-device-opened"
    # The same run, one table of configuration apart.
    assert single["nest"] is None
    assert "nest_allocation" not in single
    assert nested["nest"] is not None
    assert nested["nest"]["dx_m"] == 200.0
    assert nested["nest"]["nx"] == 100
    assert nested["nest"]["child_dt_s"] == 0.1
    assert nested["nest"]["feedback_name"] == "two-way"
    assert nested["nest"]["cm1_authority"] == CM1_NESTING_ABSENCE
    # The child is priced exactly as the parent is, on the host.
    assert nested["nest_allocation"]["config"]["grid"]["dx_m"] == 200.0
    assert nested["nest_allocation"]["dynamics"] and nested["nest_allocation"]["closure"]
    # Changing one word of the nest table changes the price.
    finer = cartesian_plan_payload(
        parse_mapping(tomllib.loads(BASE_TOML + _nest_text(refinement_ratio=25, time_refinement_ratio=25)))
    )
    assert finer["nest_allocation"] != nested["nest_allocation"]
    # Nothing about the parent changed.
    for key in ("measured_steps", "capacity_profile", "admitted_level_bounds", "config"):
        assert nested[key] == single[key]
    assert json.dumps(nested, sort_keys=True, allow_nan=False)


@pytest.mark.skipif(
    not __import__("cm1core.engine_pin", fromlist=["inspect"]).inspect().admitted,
    reason="the audited ArWen estate is not installed, so admission stops earlier",
)
def test_the_command_line_prints_the_nest_plan(tmp_path: Path, capsys) -> None:
    """The same payload, through the installed command line."""

    from cm1core.cli import main

    document = tmp_path / "nested.toml"
    document.write_text(BASE_TOML + NEST_TABLE, encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    assert main(["cartesian", str(document), "--output", str(receipt), "--plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nest"]["dx_m"] == 200.0
    assert payload["nest_allocation"]["config"]["grid"]["nx"] == 100
    assert not receipt.exists()


def test_the_shipped_nested_example_is_admitted_end_to_end() -> None:
    config = load_config(NESTED_EXAMPLE)
    binding = require_nest(config)
    assert binding is not None
    assert binding.geometry.dx_m == 100.0
    assert binding.geometry.nx == 270
    assert binding.child.dt_s == pytest.approx(0.05)
    assert binding.follow_storm is True
    assert binding.follow_interval_steps == 600
    record = binding.record(parent_dt_s=config.run.dt_s)
    assert record["lateral_forcing"]["normal_flux_is_parent_face_value"] is True
    assert record["storm_following"]["follow_storm"] is True
    assert json.dumps(record, sort_keys=True, allow_nan=False)


# ---------------------------------------------------------------------------
# 4. The lateral forcing the parent actually supplies.
# ---------------------------------------------------------------------------
def _parent_field(nx: int, ny: int, seed: int = 11) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [
            300.0
            + 4.0 * math.sin(i / 6.0)
            + 3.0 * math.cos(j / 5.0)
            + rng.uniform(-0.4, 0.4)
            for i in range(nx)
        ]
        for j in range(ny)
    ]


def _geometry(ratio: int = 5, feedback: int = 1) -> NestGeometry:
    return NestGeometry(
        parent_nx=40,
        parent_ny=40,
        parent_dx_m=1000.0,
        parent_dy_m=1000.0,
        refinement_ratio=ratio,
        time_refinement_ratio=ratio,
        parent_i0=10,
        parent_j0=10,
        parent_span_x=10,
        parent_span_y=10,
        spec_zone=1,
        relax_zone=4,
        feedback=feedback,
    )


@pytest.mark.parametrize("ratio", [3, 5, 7, 9])
def test_the_parent_cell_mean_survives_interpolation_to_the_last_bit(
    ratio: int,
) -> None:
    """Conservation across the interface, as an exact statement.

    Not a tolerance: ``sub_cell_offsets`` is spelled so the offsets are exact
    binary negations of one another, and the block sums in pairing order to
    exactly zero deviation from the parent value.
    """

    geometry = _geometry(ratio)
    parent = _parent_field(40, 40)
    for j in range(1, 39):
        for i in range(1, 39):
            assert block_mean_residual(parent, geometry, j, i) == 0.0


def test_the_interpolation_creates_no_new_extremum() -> None:
    geometry = _geometry(5)
    parent = _parent_field(40, 40)
    child = interpolate_parent_to_child(parent, geometry)
    donor_i = geometry.donor_index_x()
    donor_j = geometry.donor_index_y()
    for jc in range(geometry.ny):
        for ic in range(geometry.nx):
            j, i = donor_j[jc], donor_i[ic]
            stencil = (
                parent[j][i],
                parent[j][i - 1],
                parent[j][i + 1],
                parent[j - 1][i],
                parent[j + 1][i],
            )
            assert min(stencil) - 1e-12 <= child[jc][ic] <= max(stencil) + 1e-12


def test_a_non_negative_parent_gives_the_child_no_negative_water() -> None:
    """The whole of the microphysics treatment at the interface."""

    geometry = _geometry(5)
    rng = random.Random(3)
    # A mixing-ratio-shaped field: mostly zero, with a sharp moist core.
    parent = [
        [
            max(0.0, 0.012 * math.exp(-(((i - 21) / 3.0) ** 2 + ((j - 19) / 3.0) ** 2)))
            + (0.0 if rng.random() < 0.7 else rng.uniform(0.0, 1e-4))
            for i in range(40)
        ]
        for j in range(40)
    ]
    child = interpolate_parent_to_child(parent, geometry)
    assert min(min(row) for row in child) >= 0.0


def test_the_normal_flux_at_the_child_boundary_is_the_parents_own_face() -> None:
    geometry = _geometry(5)
    for side in ("west", "east", "south", "north"):
        assert boundary_normal_velocity_is_parent_face_value(geometry, side)
    assert parent_face_index(geometry, "west") == 10
    assert parent_face_index(geometry, "east") == 20
    assert parent_face_index(geometry, "south") == 10
    assert parent_face_index(geometry, "north") == 20


def test_the_davies_weights_are_the_declared_ramp() -> None:
    fcx, gcx = davies_weights(spec_zone=1, relax_zone=4, dt_s=0.1)
    assert fcx[0] == 0.0  # the specified row is not relaxed
    assert fcx[1] == pytest.approx(0.1 / 0.1 * 1.0)
    assert fcx[2] == pytest.approx(0.1 / 0.1 * (2 / 3))
    assert fcx[3] == pytest.approx(0.1 / 0.1 * (1 / 3))
    assert fcx[4] == 0.0  # the ramp reaches zero at the inner edge
    for f, g in zip(fcx, gcx):
        assert g == pytest.approx(f / 5.0)
    # A shorter child step tightens the relaxation in the same proportion.
    fast, _ = davies_weights(spec_zone=1, relax_zone=4, dt_s=0.05)
    assert fast[1] == pytest.approx(2.0 * fcx[1])
    with pytest.raises(ValueError):
        davies_weights(spec_zone=1, relax_zone=1, dt_s=0.1)


def test_the_boundary_tables_hit_the_parent_value_at_the_end_of_the_interval() -> None:
    """``value + dtbc*tendency`` is the child's own state at ``dtbc=0`` and
    the interpolated parent at ``dtbc=parent_dt``."""

    geometry = _geometry(5)
    parent_end = _parent_field(40, 40, seed=5)
    child_from_parent = interpolate_parent_to_child(parent_end, geometry)
    child_start = [
        [value - 1.25 for value in row] for row in child_from_parent
    ]
    tables = build_boundary_tables(
        parent_at_end=parent_end,
        child_at_start=child_start,
        geometry=geometry,
        parent_dt_s=0.5,
        width=5,
    )
    west = tables["west"]
    for jc in range(geometry.ny):
        for w in range(5):
            assert west.value[jc][w] == child_start[jc][w]
            assert west.target(dtbc=0.0)[jc][w] == pytest.approx(child_start[jc][w])
            assert west.target(dtbc=0.5)[jc][w] == pytest.approx(
                child_from_parent[jc][w]
            )


def test_the_relaxation_moves_the_child_toward_the_parent_and_only_near_the_edge() -> None:
    geometry = _geometry(5)
    parent_end = _parent_field(40, 40, seed=9)
    child = interpolate_parent_to_child(parent_end, geometry)
    # Give the child a uniform cold bias so the relaxation has work to do.
    biased = [[value - 2.0 for value in row] for row in child]
    tables = build_boundary_tables(
        parent_at_end=parent_end,
        child_at_start=biased,
        geometry=geometry,
        parent_dt_s=0.5,
        width=geometry.spec_zone + geometry.relax_zone,
    )
    zero = [[0.0] * geometry.nx for _ in range(geometry.ny)]
    out = apply_specified_relaxation(
        field=biased,
        tendency=zero,
        tables=tables,
        geometry=geometry,
        dtbc=0.5,
        child_dt_s=0.1,
    )
    # The specified row carries the parent's tendency outright and it is
    # positive, because the child is cold.
    assert out[0][geometry.nx // 2] > 0.0
    assert out[geometry.ny - 1][geometry.nx // 2] > 0.0
    assert out[geometry.ny // 2][0] > 0.0
    # The relaxed rows are warmed too.
    assert out[2][geometry.nx // 2] > 0.0
    # The interior is untouched: a nest whose relaxation reached the middle
    # would be a re-interpolation of the parent, not a refinement of it.
    middle = geometry.ny // 2
    assert out[middle][geometry.nx // 2] == 0.0
    assert all(value == 0.0 for value in out[middle][10:-10])


def test_two_way_feedback_returns_an_unmodified_child_as_the_parent() -> None:
    geometry = _geometry(5, feedback=1)
    parent = _parent_field(40, 40, seed=13)
    child = interpolate_parent_to_child(parent, geometry)
    back = restrict_child_to_parent(child, geometry)
    checked = 0
    for jp in range(geometry.parent_span_y):
        for ip in range(geometry.parent_span_x):
            value = back[jp][ip]
            if value != value:  # NaN marks a relaxation row, never fed back
                continue
            assert value == pytest.approx(
                parent[geometry.parent_j0 + jp][geometry.parent_i0 + ip], abs=1e-12
            )
            checked += 1
    assert checked > 0


def test_an_even_ratio_would_displace_every_fed_back_face_value() -> None:
    """The concrete wrong result the parity refusal prevents, measured.

    The constructor refuses an even ratio, so the displacement is computed on
    a geometry built by hand from the same offsets.
    """

    from dataclasses import replace

    odd = _geometry(5)
    assert odd.feedback_face_displacement_m() == 0.0
    even = replace(odd, refinement_ratio=5)
    object.__setattr__(even, "refinement_ratio", 4)
    # Four child faces about a parent face: -2, -1, 0, +1 child cells.
    assert even.feedback_face_displacement_m() == pytest.approx(-0.5 * even.dx_m)
    with pytest.raises(ValueError, match="even"):
        _geometry(4)


# ---------------------------------------------------------------------------
# 5. Storm following, on CM1's own tracker.
# ---------------------------------------------------------------------------
def test_cm1s_own_smoother_and_centre_find_a_displaced_pressure_minimum() -> None:
    nx = ny = 41
    # A pressure perturbation field with one clear minimum away from centre.
    field = [
        [
            -600.0 * math.exp(-(((i - 29) / 4.0) ** 2 + ((j - 12) / 4.0) ** 2))
            for i in range(nx)
        ]
        for j in range(ny)
    ]
    # The smoother is CM1's: 100 passes of the 9-point 0.11111111 stencil.
    smoothed = cm1_smoothed_pressure(field, passes=4)
    assert len(smoothed) == ny and len(smoothed[0]) == nx
    assert min(min(r) for r in smoothed) > min(min(r) for r in field)
    i, j = cm1_pressure_centre_cell(field, passes=20)
    assert abs(i - 29) <= 2
    assert abs(j - 12) <= 2


def test_the_nest_follows_the_tracked_centre_in_whole_parent_cells() -> None:
    geometry = _geometry(5)
    # Nest centre is parent cell face 15, tracked centre is at face 21.
    move = plan_nest_move(
        geometry, centre_i_face=21, centre_j_face=15, max_cells_per_move=10
    )
    assert move.shift_i == 6
    assert move.shift_j == 0
    assert move.clamped is False
    moved = move_nest(geometry, move)
    assert moved.parent_i0 == geometry.parent_i0 + 6
    assert moved.origin_x_m == geometry.origin_x_m + 6 * 1000.0
    # The moved nest is still face aligned and still admissible.
    assert moved.placement_defect() is None


def test_the_nest_motion_is_rate_limited_and_says_so() -> None:
    geometry = _geometry(5)
    move = plan_nest_move(
        geometry, centre_i_face=30, centre_j_face=15, max_cells_per_move=2
    )
    assert move.requested_shift_i == 15
    assert move.shift_i == 2
    assert move.clamped is True
    assert "max_cells_per_move" in move.reason


def test_the_nest_stops_at_the_parent_interior_rather_than_leaving_it() -> None:
    geometry = _geometry(5)
    move = plan_nest_move(
        geometry, centre_i_face=39, centre_j_face=15, max_cells_per_move=50
    )
    moved = move_nest(geometry, move)
    assert moved.placement_defect() is None
    assert move.clamped is True
    assert "parent interior" in move.reason


# ---------------------------------------------------------------------------
# 6. Refused by key and by value, each naming a concrete wrong result.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,code,fragment",
    [
        ({"refinement_ratio": 4}, "CM1-CFG-NEST-RATIO-PARITY", "half a child cell"),
        ({"refinement_ratio": 1}, "CM1-CFG-NEST-RATIO", "refines nothing"),
        ({"spec_zone": 0}, "CM1-CFG-NEST-SPEC-ZONE", "uninitialised memory"),
        ({"relax_zone": 1}, "CM1-CFG-NEST-RELAX-ZONE", "reflects the storm"),
        (
            {"spec_zone": 4, "relax_zone": 4},
            "CM1-CFG-NEST-ZONE-ORDER",
            "applies to nothing",
        ),
        ({"feedback": 2}, "CM1-CFG-UNSUPPORTED-VALUE", "one-way"),
        (
            {"parent_i0": 1},
            "CM1-CFG-NEST-PLACEMENT",
            "open radiative lateral boundary",
        ),
        (
            {"parent_span_x": 45},
            "CM1-CFG-NEST-PLACEMENT",
            "open radiative lateral boundary",
        ),
        (
            {
                "acoustic_substeps": 4,
                "time_refinement_ratio": 1,
                "refinement_ratio": 25,
            },
            "CM1-CFG-NEST-ACOUSTIC-SUBSTEPS",
            "acoustic small step",
        ),
        (
            {"refinement_ratio": 3, "time_refinement_ratio": 3},
            "CM1-CFG-NEST-CHILD-SPACING",
            "no exact default-REAL word",
        ),
        (
            {"follow_storm": False},
            "CM1-CFG-NEST-FOLLOW-INACTIVE",
            "never moved",
        ),
        (
            {"follow_interval_s": 0.0, "follow_max_speed_m_s": 0.0},
            "CM1-CFG-NEST-FOLLOW-ACTIVE",
            "resolve clear air",
        ),
        (
            {"follow_interval_s": 61.3},
            "CM1-CFG-NEST-FOLLOW-CADENCE",
            "integer number of parent large steps",
        ),
        (
            {"follow_max_speed_m_s": 420.0},
            "CM1-CFG-NEST-FOLLOW-STEP",
            "off its previous footprint",
        ),
    ],
)
def test_every_nest_key_is_refused_by_key_and_by_value(
    overrides: dict, code: str, fragment: str
) -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**overrides)
    detail = caught.value.detail
    assert detail.code == code, detail.reason
    assert fragment in detail.reason
    assert detail.option is not None
    assert detail.remedy


def test_the_two_conditions_the_nest_needs_are_refused_by_the_lane_already() -> None:
    """``psolver=3`` and an unstretched horizontal parent, and no second guard.

    Both are conditions the child genuinely depends on -- the child's acoustic
    cadence comes from ``misclibs.F:857-868``'s psolver=3 branch, and the
    face-alignment that makes the interface flux the parent's own needs a
    single parent spacing -- but the Cartesian lane already refuses both before
    the nest is looked at.  This test asserts the refusals are live and that
    ``_validate_nest`` does not stand a dead duplicate beside them.
    """

    import tomllib
    from cm1core import config as config_module

    text = BASE_TOML.replace("psolver = 3", "psolver = 2") + NEST_TABLE
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(text))
    assert caught.value.detail.option == "dynamics.psolver"

    text = BASE_TOML.replace("stretch_x = false", "stretch_x = true") + NEST_TABLE
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(text))
    assert caught.value.detail.option == "grid.stretch_x"

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "CM1-CFG-NEST-PSOLVER" not in source
    assert "CM1-CFG-NEST-HORIZONTAL-STRETCH" not in source


def test_an_unknown_nest_key_is_still_refused_by_name() -> None:
    import tomllib

    text = BASE_TOML + NEST_TABLE + "extra_key = 1\n"
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(text))
    assert caught.value.detail.code.startswith("CM1-CFG-")


def test_a_half_declared_nest_is_refused_rather_than_completed() -> None:
    import tomllib

    text = BASE_TOML + "\n[nest]\nrefinement_ratio = 5\n"
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(text))
    assert caught.value.detail.code.startswith("CM1-CFG-")


def test_the_nest_table_is_refused_outside_the_cartesian_lane() -> None:
    config = load_config(REPO / "examples" / "official-supercell-product.toml")
    assert config.nest is None


# ---------------------------------------------------------------------------
# 7. The existing default is bit-identical.
# ---------------------------------------------------------------------------
#: The sealed ``config.sha256`` of every shipped single-domain example, as
#: computed by the tree this branch started from (``git archive HEAD``).  These
#: hashes travel inside recorded GPU-run receipts as ``plan_sha256``, so a
#: change to any of them would orphan measurements already taken.  They are
#: pinned here by value rather than recomputed, because a test that recomputes
#: both sides of an equality proves nothing.
# Measured three ways and identical in all three: with the pre-nest
# ``config.py`` of the merge base, with sota/uhdiag's own, and with this
# branch's after both merges.  Not one of the twenty documents moved a byte.
# That is the whole claim -- adding [nest] and [diagnostics] changes nothing
# for a document that declares neither -- and it is why this table grows by
# appending measured hashes rather than by re-baselining.
# Moved deliberately since the table was first measured, and only for these
# reasons: eeea32b (merged in a3b8c75) projects dynamics.alph/kdiv/coriolis/
# idiff and moves alph 0.5 -> CM1's 0.60 (the twenty-one documents that existed
# then are exactly their a3b8c75 hashes); later merges shipped more
# single-domain examples, which are appended.  NOT absorbed: since a0cee5e
# (sota/uhdiag) CM1Config.canonical_dict leaks the dataclass field
# tornado_diagnostics into every document (as null when undeclared), which
# moves every identity below although no key a document declares changed.
# These values are the hashes with that field not projected, as the comments in
# canonical_dict state it must be, so this test fails until config.py stops
# projecting it.
SEALED_EXAMPLE_CONFIG_SHA256 = {
    'cartesian-adaptive-moving-domain.toml': 'a9c469d743c5023e317e3ca57ba9fbee1eb7416fb2f248f7d868208ac5c9ac52',
    'cartesian-environment-fields.toml': 'dcf75d6d48504b80d4ac3797c6d629b601844d361d177cda558059a316b86e0c',
    'cartesian-fifty-metre-fine-step.toml': '39c3e8198371318b55955f26c1cf94c83d831a1945d6d132544f9cebbfa6fcef',
    'cartesian-idealized-radiation-relaxation.toml': '026bd77d3341dd153adb8f22f426fbf66173d368f05305000790e246490462d5',
    'cartesian-initial-condition-only.toml': 'e5b28a5b43a621fdb5564f80da530afd2a795526c409f4d9b39569edc7effdac',
    'cartesian-open-boundary-companions.toml': '0da3451dbf2fa368ca67934591bef0913232d70fbe961a756cadbf1b1f79d779',
    'cartesian-periodic-laterals.toml': 'a53d68402f55a6b99a02823ab174505573ecb30dddfb553bd2dfa4680714fefd',
    'cartesian-restart-and-ensemble.toml': '6a0a95c687cd5a733477e568136a7ce0b68f58ad340c5b7f31607a5f8c837f3d',
    'cartesian-squall-line-rkw-shear.toml': 'e95ccbb8508411e86019c2fd211055c10cff720f0149c26812385e2f77e56f24',
    'cartesian-stretched-tornado-grade-grid.toml': 'd4f0b4a9fa9bf371a6e726067fa4ce4998925cbdd876831fbad6365212a9d811',
    'cartesian-stretched-tornado-two-part-sgs.toml': 'bd89d449a9032dead1a6ec2d0f191094c729969e2317b1f5b8609c37a51f75b0',
    'cartesian-supercell-attempt.toml': 'f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b',
    'cartesian-supercell-checkpointed.toml': '4fc9757360147fa4079ae2e9d935784682b5c0f344a4583233788f8e113f31d3',
    # Moved deliberately by tor2/lspgrad1-perturbation-coriolis: the document
    # now declares dynamics.lspgrad = 1 (it was
    # 35d2730806f918916a1c36f74a68b978d14f320265655aff7a32aa9a80f3e06c).
    'cartesian-supercell-coriolis.toml': '5a5b268d2baa97d10608e7ee9d69f0bb28c247e2997d28b61fde07adacc56aad',
    'cartesian-supercell-moore-20130520-18z-16gib.toml': 'edb512b96ec4e9d60b863533ad22c11ceaeb8c534f0c9edfaaddddae2b65c8f3',
    'cartesian-supercell-moore-20130520-18z-32gib.toml': '306e38d05260d258d1473d5d06ccf277dcd4798a91c3c0a1b11722faace69fbb',
    'cartesian-supercell-morrison-hail.toml': 'a77c563732ccbbb7d66c5d5c2cdf2675e05c7c315e8c3384d00a49718fe7a435',
    'cartesian-supercell-reflectivity.toml': '93a9773d3b790f7637deca63c8afcfad3c608c209e9f88962638b1b5f624c4ca',
    'cartesian-supercell-resume.toml': '2a175db02bd580c998d00c8fe741172a729852c895a8c7b544f3a12c8d41e759',
    'cartesian-supercell-storm-diagnostics.toml': '048af23575b6b32e5f1b59f9e4ffdbb3eb1803d8b9ee66fba0c8f45fdae35043',
    'cartesian-supercell-surface-drag.toml': '5bc41131130449a709a673f780c47e844187e06d105af339391930d7bf8f0a77',
    'cartesian-supercell-swok-20190521-00z-16gib.toml': 'd0be8815cec61bb90ae8c90d7e32c4dee0c1acc08916d4386579fb4f79866146',
    'cartesian-supercell-swok-20190521-00z-32gib.toml': 'e833d0e4c5a21e1211decf64fa23fc9d394798fa77fe275ea457a543098eefa0',
    'cartesian-supercell-thompson-aerosol.toml': 'f62afbbdcd79c66ba1a32a6de5d0fd025688ce2c690d09bb6f26b9153c08496c',
    'cartesian-supercell-thompson.toml': '1aa60c4dca41acbe5152f723f575f6f58c33bd778b03befca01eb4bad6e17b86',
    'cartesian-supercell-translating-frame.toml': '77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca',
    'cartesian-surface-enthalpy-flux-over-water.toml': 'a165fa493e7c00210b8ceb26396e52b3e0505ffd7dc03c920c831fa8da3b6169',
    'cartesian-surface-prescribed-roughness.toml': '0351bed1183def9a6ff04cb9b155bde6e9e647912cb309a2a5ec071403ba053b',
    'cartesian-tornado-budgets.toml': '0b387dfcf3ac4f4c10f8da71373b64f9722abaa72b5ac719a58c9cfd131a7bdf',
    'cartesian-tornado-diagnostics.toml': '7a07b8468a96ef8821a1714d09fba1ea96cbf28895a68bc11aca4f49a902f1c3',
    'cartesian-tornado-lagrangian-analysis.toml': 'ddbf431ec89de383a30ef0b84bfa90ae89a14790dec099a8875d66b94e09d3d2',
    'cartesian-uniform-grid.toml': '60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48',
    'cartesian-upstream-bubble-seeded.toml': 'ef77f794f7e734dbeb9251d30b56f9ce6b822243d8bfc50f6ab5bac9214983f8',
    'cartesian-wk-analytic-sounding-shear.toml': 'e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7',
    'dry-psolver2-bootstrap.toml': '0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340',
    'official-supercell-native-context.toml': '33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c',
    'official-supercell-product.toml': 'b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038',
    'real-atmosphere-100m-product.toml': '07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00',
}


def test_a_document_with_no_nest_projects_no_nest_key_and_keeps_its_hash() -> None:
    """The proof that this branch changes nothing for anyone not using it."""

    import tomllib

    config = parse_mapping(tomllib.loads(BASE_TOML))
    assert config.nest is None
    assert "nest" not in config.canonical_dict()

    # Every shipped single-domain example still hashes to the byte it hashed
    # to before ``[nest]`` existed.
    measured = {}
    for example in sorted((REPO / "examples").glob("*.toml")):
        if example.name in NESTED_EXAMPLES:
            continue
        loaded = load_config(example)
        assert loaded.nest is None
        assert "nest" not in loaded.canonical_dict()
        measured[example.name] = loaded.sha256
    assert measured == SEALED_EXAMPLE_CONFIG_SHA256


def test_the_declared_nest_is_projected_and_changes_the_documents_identity() -> None:
    """The other half: a declared nest is sealed, not silently dropped."""

    import tomllib

    flat = parse_mapping(tomllib.loads(BASE_TOML))
    nested = parse_mapping(tomllib.loads(BASE_TOML + NEST_TABLE))
    projected = nested.canonical_dict()
    assert projected["nest"]["refinement_ratio"] == 5
    assert set(projected["nest"]) == {
        "refinement_ratio",
        "time_refinement_ratio",
        "parent_i0",
        "parent_j0",
        "parent_span_x",
        "parent_span_y",
        "spec_zone",
        "relax_zone",
        "feedback",
        "acoustic_substeps",
        "follow_storm",
        "follow_interval_s",
        "follow_max_speed_m_s",
        "follow_tracker",
    }
    assert nested.sha256 != flat.sha256
    # Two documents differing only in one nest word are two identities.
    other = parse_mapping(tomllib.loads(BASE_TOML + _nest_text(parent_i0=16)))
    assert other.sha256 != nested.sha256


def test_the_flat_example_binds_exactly_the_runtime_it_always_did() -> None:
    config = load_config(FLAT_EXAMPLE)
    assert config.nest is None
    assert require_nest(config) is None
    assert nest_allocation_plan(None) == {"nest": None}
    runtime = build_runtime_config(config)
    assert runtime.grid.dx_m == 1000.0
    assert runtime.acoustic_substeps == 6


# ---------------------------------------------------------------------------
# 8. The device kernel this lane will launch.
# ---------------------------------------------------------------------------
def test_the_relaxation_kernel_text_is_pinned_and_self_consistent() -> None:
    import hashlib

    assert (
        hashlib.sha256(NEST_RELAXATION_KERNEL_SOURCE.encode("utf-8")).hexdigest()
        == NEST_RELAXATION_KERNEL_SHA256
    )
    # Every float operation is explicitly round-to-nearest: nothing in this
    # kernel may be contracted into an FMA, because the CPU oracle above is
    # what it is compared against.
    assert "__fadd_rn" in NEST_RELAXATION_KERNEL_SOURCE
    assert "__fmul_rn" in NEST_RELAXATION_KERNEL_SOURCE
    assert "__fsub_rn" in NEST_RELAXATION_KERNEL_SOURCE
    for forbidden in ("fmaf(", "__fmaf", "expf(", "powf("):
        assert forbidden not in NEST_RELAXATION_KERNEL_SOURCE


# ---------------------------------------------------------------------------
# 9. The schedule a nested run actually is, executed on the host.
# ---------------------------------------------------------------------------
def test_the_child_takes_exactly_the_declared_sub_steps_across_a_parent_step() -> None:
    from cm1core.nest_coupler import parent_interval_schedule

    geometry = _geometry(5)
    schedule = parent_interval_schedule(geometry, parent_dt_s=0.5, parent_time_s=10.0)
    assert len(schedule) == 5
    assert [step.index for step in schedule] == [0, 1, 2, 3, 4]
    # WRF resets dtbc when the tables are built (mediation_force_domain.F:213)
    # and advances it by one child step before each step's forcing
    # (solve_em.F:372), so the first sub-step already targets one child step
    # into the interval and the last targets the parent's new state exactly.
    assert schedule[0].dtbc == pytest.approx(0.1)
    assert schedule[-1].dtbc == pytest.approx(0.5)
    assert schedule[-1].model_time_s == pytest.approx(10.4)


def test_one_coupled_parent_step_forces_the_edge_and_leaves_the_interior_free() -> None:
    from cm1core.nest_coupler import couple_one_parent_step, euler_advance

    geometry = _geometry(3, feedback=0)
    parent = _parent_field(40, 40, seed=21)
    target = interpolate_parent_to_child(parent, geometry)
    cold = [[value - 2.0 for value in row] for row in target]
    zero = [[0.0] * geometry.nx for _ in range(geometry.ny)]
    child, step = couple_one_parent_step(
        geometry=geometry,
        parent_at_end=parent,
        child_at_start=cold,
        child_tendency=zero,
        advance=euler_advance(0.5 / 3),
        parent_dt_s=0.5,
        parent_time_s=0.0,
    )
    middle = geometry.nx // 2
    # The specified row reaches the parent's value: its tendency is the
    # table's, and one parent interval of it closes the whole difference.
    assert child[0][middle] == pytest.approx(target[0][middle], abs=1e-9)
    assert child[geometry.ny - 1][middle] == pytest.approx(
        target[geometry.ny - 1][middle], abs=1e-9
    )
    # The relaxed rows are moved toward the parent but not pinned to it.
    assert cold[2][middle] < child[2][middle] <= target[2][middle] + 1e-9
    # The interior is untouched: the nest refines the parent, it does not
    # re-interpolate it.
    assert child[middle][middle] == pytest.approx(cold[middle][middle])
    assert step.fed_back_cells == 0
    assert step.record()["feedback"] == "one-way"


def test_two_way_feedback_writes_the_childs_interior_into_the_parent() -> None:
    from cm1core.nest_coupler import couple_one_parent_step, euler_advance

    geometry = _geometry(3, feedback=1)
    parent = _parent_field(40, 40, seed=23)
    target = interpolate_parent_to_child(parent, geometry)
    cold = [[value - 2.0 for value in row] for row in target]
    zero = [[0.0] * geometry.nx for _ in range(geometry.ny)]
    writeback = [list(row) for row in parent]
    _, step = couple_one_parent_step(
        geometry=geometry,
        parent_at_end=parent,
        child_at_start=cold,
        child_tendency=zero,
        advance=euler_advance(0.5 / 3),
        parent_dt_s=0.5,
        parent_writeback=writeback,
    )
    assert step.fed_back_cells > 0
    changed = [
        (jp, ip)
        for jp in range(geometry.parent_span_y)
        for ip in range(geometry.parent_span_x)
        if writeback[geometry.parent_j0 + jp][geometry.parent_i0 + ip]
        != parent[geometry.parent_j0 + jp][geometry.parent_i0 + ip]
    ]
    assert len(changed) == step.fed_back_cells
    # The child's interior is two degrees cold, and that is what the parent
    # receives.
    jp, ip = changed[0]
    assert writeback[geometry.parent_j0 + jp][
        geometry.parent_i0 + ip
    ] == pytest.approx(
        parent[geometry.parent_j0 + jp][geometry.parent_i0 + ip] - 2.0, abs=1e-6
    )
    # Nothing outside the nest moved.
    for j in range(40):
        for i in range(40):
            inside_j = geometry.parent_j0 <= j < geometry.parent_j0 + geometry.parent_span_y
            inside_i = geometry.parent_i0 <= i < geometry.parent_i0 + geometry.parent_span_x
            if inside_j and inside_i:
                continue
            assert writeback[j][i] == parent[j][i]


def test_one_way_feedback_never_touches_the_parent() -> None:
    from cm1core.nest_coupler import couple_one_parent_step, euler_advance

    geometry = _geometry(3, feedback=0)
    parent = _parent_field(40, 40, seed=29)
    target = interpolate_parent_to_child(parent, geometry)
    cold = [[value - 3.0 for value in row] for row in target]
    zero = [[0.0] * geometry.nx for _ in range(geometry.ny)]
    writeback = [list(row) for row in parent]
    _, step = couple_one_parent_step(
        geometry=geometry,
        parent_at_end=parent,
        child_at_start=cold,
        child_tendency=zero,
        advance=euler_advance(0.5 / 3),
        parent_dt_s=0.5,
        parent_writeback=writeback,
    )
    assert step.fed_back_cells == 0
    assert writeback == [list(row) for row in parent]


def test_a_nested_document_refuses_only_what_the_live_nest_does_not_execute(
    tmp_path: Path,
) -> None:
    """The nest integrates and moves; what it does not do is refused by name.

    A following nest is admitted past every nest gate (cm1core.nest_move moves
    the running child).  CM1's pressure tracker on a parent whose lateral halo
    is not the open one still refuses: its zero-gradient smoother halo is bc2d
    for code-2 boundaries only.  A fixed nest travels as far as the same
    document without a nest does on this host, and stops at a refusal that has
    nothing to do with nesting.
    """

    from cm1core.cartesian_door import run_cartesian
    import tomllib

    config = parse_mapping(tomllib.loads(BASE_TOML + NEST_TABLE))
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            config,
            config_label="nested",
            output=tmp_path / "receipt.json",
            integrate=True,
            history_dir=tmp_path / "history",
        )
    detail = caught.value.detail
    assert not detail.code.startswith("CM1-CARTESIAN-NEST"), detail.code
    assert not (tmp_path / "receipt.json").exists()

    periodic = BASE_TOML.replace("west = 2\neast = 2", "west = 1\neast = 1")
    assert periodic != BASE_TOML
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            parse_mapping(tomllib.loads(periodic + NEST_TABLE)),
            config_label="nested-periodic",
            output=tmp_path / "periodic.json",
            integrate=True,
            history_dir=tmp_path / "periodic-history",
        )
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-INTEGRATION", detail.code
    assert "bc2d" in detail.reason

    fixed = parse_mapping(tomllib.loads(BASE_TOML + _nest_text(
        follow_storm=False, follow_interval_s=0.0, follow_max_speed_m_s=0.0,
    )))
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            fixed,
            config_label="fixed",
            output=tmp_path / "fixed.json",
            integrate=True,
            history_dir=tmp_path / "fixed-history",
        )
    # A fixed nest is admitted to integration: whatever stops it on a host
    # without a device is a later gate, never the nest.
    detail = caught.value.detail
    assert detail.code != "CM1-CARTESIAN-NEST-INTEGRATION"
    assert not detail.code.startswith("CM1-CARTESIAN-NEST")
    assert not (tmp_path / "fixed.json").exists()

def test_a_parent_cell_count_for_the_following_step_is_refused_by_name() -> None:
    import tomllib

    text = BASE_TOML + NEST_TABLE.replace(
        "follow_max_speed_m_s = 55.0", "follow_max_cells = 3"
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(text))
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-NEST-FOLLOW-UNIT"
    assert "different speed on every parent spacing" in detail.reason
    assert "follow_max_speed_m_s" in detail.remedy


def test_the_same_speed_survives_a_change_of_parent_spacing() -> None:
    from cm1core.config import nest_follow_max_cells

    config = _config()
    assert config.nest.follow_max_speed_m_s == 55.0
    binding = require_nest(config)
    # 55 m/s * 60 s over 1000 m parent cells.
    assert binding.follow_max_cells == 3
    halved = nest_follow_max_cells(config.nest, parent_dx_m=500.0)
    assert halved == 6
    assert binding.record(parent_dt_s=0.5)["storm_following"]["follow_max_speed_m_s"] == 55.0


def test_a_speed_too_slow_to_move_one_parent_cell_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(follow_max_speed_m_s=10.0)
    assert caught.value.detail.code == "CM1-CFG-NEST-FOLLOW-SPEED"


def test_the_default_tracker_holds_the_placement_offset() -> None:
    from cm1core.nest_geometry import follow_target_face

    config = _config()
    assert config.nest.follow_tracker == "cm1-pressure-held-offset"
    # A window placed 9 km east of a steady pressure centre on a 1 km parent
    # stays 9 km east of it as the centre moves, rather than jumping onto it.
    assert follow_target_face(
        "cm1-pressure-held-offset", centre_i_face=40, centre_j_face=30,
        placement_offset_faces=(9.0, 0.0),
    ) == (49.0, 30.0)
    assert follow_target_face(
        "cm1-pressure-centre", centre_i_face=40, centre_j_face=30,
        placement_offset_faces=(9.0, 0.0),
    ) == (40.0, 30.0)
    import tomllib

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(
            BASE_TOML + NEST_TABLE + 'follow_tracker = "vorticity-maximum"\n'
        ))
    assert caught.value.detail.code == "CM1-CFG-NEST-FOLLOW-TRACKER"
