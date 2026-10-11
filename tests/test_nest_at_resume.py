"""Can a ``[nest]`` table be added AT RESUME to a checkpointed run?

This is the "roll back five minutes and drop a nest on the storm" flow, and
the answer has two halves that are worth keeping apart, because only one of
them is a gap.

**The restart layer admits it.**  ``cartesian_restart.require_resumable``
draws README.restart:20-23's line between a minor configuration change and a
major one by value: it compares the restart *profile* (grid extent and carrier
inventory), ``dt_s``, the initializer's source identity, the compiled operator
set and the frame velocity.  A ``[nest]`` table moves none of those -- it adds
a second domain, it does not reshape the first -- so a restart written by a
single-domain run prices the identical profile as the same document with a
nest bolted on.  ``config_sha256`` does change, and is recorded in the restart
receipt, but it is recorded rather than compared, precisely so that the run
length and the publication cadences may change across a resume.

**The integration admits it too**, now that the live nest integrates a fixed
window concurrently with its parent (``cm1core.nest_live``): the child is
built from the restored parent's state, and a following nest moves the running
child (``cm1core.nest_move``).  A nested run now writes restarts too: the
child's own restart lands beside the parent's and a resume of it restores the
child instead of rebuilding it (``cm1core.nest_restart``,
``tests/test_nest_restart.py``), so ``CM1-CARTESIAN-NEST-RESTART-WRITE`` is
retired.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from cm1core.cartesian_door import build_runtime_config
from cm1core.cartesian_restart import cartesian_restart_profile
from cm1core.config import parse_mapping

BASE_TOML = """
[run]
lane = "cartesian"
duration_s = 600.0
dt_s = 0.5
history_interval_s = 60.0
restart_interval_s = 300.0

[initialization]
isnd = 5
iwnd = 2

[grid]
nx = 60
ny = 60
nz = 40
dx_m = 250.0
dy_m = 250.0
dz_m = 250.0
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
tile_nx = 32
tile_ny = 32
warmup_steps = 0
memory_budget_mib = 4096
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
follow_max_speed_m_s = 13.0
"""

RESUME_KEYS = "irst = 1\nrstnum = 1\n"


def _config(text: str):
    return parse_mapping(tomllib.loads(text))


def _with_resume(text: str) -> str:
    return text.replace("[initialization]", RESUME_KEYS + "\n[initialization]", 1)


def test_the_nest_table_does_change_the_document_identity() -> None:
    """The control: the nest is genuinely in the canonical document."""

    plain = _config(BASE_TOML)
    nested = _config(BASE_TOML + NEST_TABLE)
    assert plain.nest is None
    assert nested.nest is not None
    assert plain.sha256 != nested.sha256
    assert "nest" not in plain.canonical_dict()
    assert "nest" in nested.canonical_dict()


def test_the_restart_profile_is_identical_with_and_without_a_nest() -> None:
    """The half that is not a gap.

    ``require_resumable`` refuses a restart whose profile does not match, and
    the profile is the grid extent and the carrier inventory.  A nest adds a
    second domain; it does not reshape the first.  If this ever stops holding,
    a nest added at resume would start being refused for a reason nobody
    intended, and the message would talk about carrier shapes.
    """

    plain = cartesian_restart_profile(build_runtime_config(_config(BASE_TOML)))
    nested = cartesian_restart_profile(
        build_runtime_config(_config(BASE_TOML + NEST_TABLE))
    )
    assert plain.sha256 == nested.sha256
    assert plain.geometry == nested.geometry
    assert plain.shapes() == nested.shapes()


def test_none_of_the_other_resume_comparisons_move_either() -> None:
    """The rest of ``require_resumable``'s list, by name.

    dt_s, the frame velocity and the initializer selection are the remaining
    things a resume compares that a document can change.  A nest changes none
    of them, and the child deliberately inherits the frame
    (``cartesian_door.require_nest`` passes the parent's umove/vmove into the
    child's RuntimeConfig) rather than carrying one of its own.
    """

    plain = _config(_with_resume(BASE_TOML))
    nested = _config(_with_resume(BASE_TOML + NEST_TABLE))
    assert plain.run.dt_s == nested.run.dt_s
    assert plain.closures.u_move_m_s == nested.closures.u_move_m_s
    assert plain.closures.v_move_m_s == nested.closures.v_move_m_s
    assert plain.initialization.isnd == nested.initialization.isnd
    assert plain.initialization.iwnd == nested.initialization.iwnd
    # Both documents are asking to continue a previous integration.
    assert plain.run.restarting and nested.run.restarting
    assert plain.run.rstnum == nested.run.rstnum == 1


FIXED_NEST_TABLE = NEST_TABLE.replace("follow_storm = true", "follow_storm = false").replace(
    "follow_interval_s = 60.0", "follow_interval_s = 0.0"
).replace("follow_max_speed_m_s = 13.0", "follow_max_speed_m_s = 0.0")


def test_a_fixed_nest_added_at_resume_is_admitted_past_every_nest_gate() -> None:
    """A fixed nest added at resume is admitted; the first refusal is not about the nest.

    The device resume was run on an RTX 5090.  On this host the document stops at
    a later gate (its output path does not exist), never at a nest refusal.
    """

    from cm1core.cartesian_door import run_cartesian
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal) as excinfo:
        run_cartesian(
            _config(_with_resume(BASE_TOML + FIXED_NEST_TABLE)),
            config_label="resume-nested",
            output=Path("/nonexistent/receipt.json"),
            integrate=True,
        )
    detail = excinfo.value.detail
    assert not detail.code.startswith("CM1-CARTESIAN-NEST")


def test_a_resumed_nest_is_admitted_past_every_nest_gate_with_or_without_restarts() -> None:
    """A following nest, and one that writes restarts, pass every nest gate.

    Both documents are admitted with no --restart-from, no device, no disk and
    no estate; the first refusal each reaches is not the nest's.
    """

    from cm1core.cartesian_door import run_cartesian
    from cm1core.errors import CM1Refusal

    # A following nest added at resume is admitted past every nest gate: its
    # tracker starts from WRF's "first time through" (no previous centre,
    # share/mediation_nest_move.F:471-476), exactly as a fresh following nest.
    with pytest.raises(CM1Refusal) as excinfo:
        run_cartesian(
            _config(_with_resume(BASE_TOML + NEST_TABLE)),
            config_label="resume-with-following-nest",
            output=Path("/nonexistent/receipt.json"),
            integrate=True,
        )
    detail = excinfo.value.detail
    assert not detail.code.startswith("CM1-CARTESIAN-NEST"), detail.code

    with pytest.raises(CM1Refusal) as excinfo:
        run_cartesian(
            _config(_with_resume(BASE_TOML + FIXED_NEST_TABLE)),
            config_label="resume-with-nest-writing-restarts",
            output=Path("/nonexistent/receipt.json"),
            integrate=True,
            restart_dir=Path("/nonexistent/restarts"),
        )
    detail = excinfo.value.detail
    assert not detail.code.startswith("CM1-CARTESIAN-NEST"), detail.code
