"""The installed door that reaches the configurable Cartesian CUDA runtime.

Every scientific, resource and platform admission happens here, on the host,
before a CUDA context is opened or an output path is created.  The numerical
work belongs to the device-resident runtime; this module admits it, drives it
and records what it actually did.  Nothing here imports CuPy at module scope.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import importlib.util
import json
import os
from pathlib import Path
from types import MappingProxyType
import sys
import time
from typing import TYPE_CHECKING, Any, Mapping

from . import CM1_AUTHORITY_SHA256, CM1_RELEASE, MATURITY, __version__
from .cartesian_history import (
    CARTESIAN_HISTORY_CARRIER_PROFILES,
    COMPLETE_CARRIER_PROFILE,
    require_history_carrier_profile,
)
from .boundary_validation_register import unvalidated_for as boundary_unvalidated_for
from .column_components import column_component_spec
from .config import CM1Config, Lane, load_config, require_vertical_coordinate
from .dense_ice import default_ihail
from .config import (
    CM1Config,
    Lane,
    load_config,
    nest_follow_max_cells,
    require_vertical_coordinate,
    tornado_diagnostic_selection,
)
from .engine_capacity import capacity_level_bounds
from .errors import CM1Refusal, refuse
from .grid_geometry import CartesianGridGeometry
from .dynamics_validation_register import dynamics_validation_record
from .gm1_runtime import (
    DOMAIN_LOCATION_AUTHORITY,
    DOMAIN_LOCATION_UNITS,
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    FIRST_RESTART_NUMBER,
    HistoryCadence,
    RunSchedule,
    RuntimeConfig,
    history_cadence,
    run_schedule,
)
from .preflight import require_runnable
from .run_geometry import BUBBLE_HORIZONTAL_RADIUS_M
from .unvalidated import receipt as unvalidated_receipt
from .unvalidated import selected_for_run as unvalidated_selected_for_run
from .initialization_validation import unvalidated_initialization
from .sounding_report import (
    SoundingReportSelection,
    format_sounding_report,
    initial_sounding_report,
)

if TYPE_CHECKING:
    from .gm1_idealized_residency import InitializerSelection, TranslatingFrame
    from .nest_geometry import NestGeometry
    from .run_geometry import Hodograph


CARTESIAN_GATE = "cartesian-runtime"
# The cm1-state cdylib is the Rust idealized initializer the runtime loads
# through ctypes; rust_tools.EXPECTED_LIBRARIES keys it by this name.
CARTESIAN_INITIALIZER_LIBRARY = "cm1-state"
# The wheel-sealed Rust writer that turns a raw carrier arena into a CM1FHS1
# envelope.  rust_tools seals it under this name.
CARTESIAN_HISTORY_PUBLISHER = "cm1-forecast-publish"
CARTESIAN_RECEIPT_SCHEMA = "gpuwm-cm1.cartesian/v1"
CARTESIAN_INTEGRATION_RECEIPT_SCHEMA = "gpuwm-cm1.cartesian-integration/v1"
# cm1.F:4341-4350.  A run that builds the initial condition and stops is not a
# forecast and does not wear a forecast receipt's schema.
CARTESIAN_INITIAL_CONDITION_RECEIPT_SCHEMA = "gpuwm-cm1.cartesian-initial-condition/v1"
DEFAULT_CAPACITY_PROFILE = "default"
# The installed idealized sounding.  cm1core.data keys it by a provenance name
# and verifies its recorded SHA-256 on the way out, so resolve it through that
# accessor rather than reading a bare package resource: a wheel whose sounding
# bytes drifted must refuse, not initialize a column from an unknown profile.
CARTESIAN_SOUNDING_RESOURCE = "cm1r21.1-cpm-radconvequil-sounding"
# gm1_runtime.RuntimeConfig requires this stencil before any device memory is
# priced; naming it here keeps the refusal a door refusal, not a TypeError.
STENCIL_RADIUS_XY = 3
STENCIL_MINIMUM_NZ = 8


from .cartesian_adaptive_frame import (  # noqa: E402
    AdaptiveFrameBinding,
    cartesian_frame_mode_record,
    require_adaptive_frame,
)


def require_cartesian_lane(config: CM1Config) -> None:
    if config.run.lane is not Lane.CARTESIAN:
        refuse(
            code="CM1-CARTESIAN-LANE",
            gate=CARTESIAN_GATE,
            reason="this door drives the configurable Cartesian runtime; another lane's sealed profile would be executed under a grid contract it was never admitted against",
            remedy=f"pass a configuration whose run.lane is {Lane.CARTESIAN.value!r}, or use the door named by that lane",
            option="run.lane",
            value=config.run.lane.value,
        )


def require_listed_component(config: CM1Config) -> None:
    from .column_components import UNLISTED_COMPONENTS

    component = config.physics.component
    switch = UNLISTED_COMPONENTS.get(component)
    if switch is not None and os.environ.get(switch) != "1":
        refuse(
            code="CM1-CARTESIAN-UNLISTED-COMPONENT",
            gate=CARTESIAN_GATE,
            reason=(
                f"physics.component = {component!r} reaches WRF Thompson-aerosol "
                "routines adapted from Numerical Recipes, whose licence does not "
                "permit free redistribution, and it has no CM1 counterpart (no "
                "ptype selects it), so this release does not offer it"
            ),
            remedy="select 'thompson-mp8' (CM1 ptype = 3), 'morrison-mp10' (ptype = 5) or 'wsm6-mp6' (ptype = 7)",
            option="physics.component",
            value=component,
        )


def selected_capacity_profile(config: CM1Config) -> str | None:
    """Return the explicit ArWen capacity profile, or None for the default."""

    runtime = config.runtime
    if runtime is None or runtime.capacity_profile == DEFAULT_CAPACITY_PROFILE:
        return None
    return runtime.capacity_profile


def require_level_capacity(config: CM1Config) -> tuple[int, int]:
    """Admit grid.nz against the column capacity the run actually selected.

    The bound is the selected component's own compiled column depth, not one
    scheme's: WSM6's launcher is compiled at 64 and 80 slots and the explicit
    overlay extends it, while Morrison's sedimentation kernel is compiled at
    64 and 256 slots and reads neither of the two files that overlay patches.
    """

    if config.physics.component == "off":
        # CM1's imoist = 0 (config.py admits "off" only beside imoist = 0 and
        # run.iconly = 1): no column is bound, so no compiled column depth
        # applies; the stencil floor is this lane's only level bound.
        return STENCIL_MINIMUM_NZ, max(STENCIL_MINIMUM_NZ, int(config.grid.nz))
    spec = column_component_spec(config.physics.component)
    profile = selected_capacity_profile(config)
    if profile is None:
        lower, upper = spec.minimum_levels, spec.maximum_levels
    else:
        if not spec.capacity_profile_supported:
            refuse(
                code="CM1-CARTESIAN-CAPACITY-PROFILE-SCOPE",
                gate=CARTESIAN_GATE,
                reason=(
                    f"the explicit column-capacity overlay rebuilds WSM6's own "
                    f"constants and vertical contract, which the "
                    f"{spec.component_id} launcher never reads; selecting it "
                    "here would admit a column depth nothing on this scheme's "
                    "path was rebuilt for"
                ),
                remedy=(
                    'set runtime.capacity_profile = "default"; this component '
                    f"already admits {spec.minimum_levels}..{spec.maximum_levels} levels"
                ),
                option="runtime.capacity_profile",
                value=profile,
            )
        lower, upper = capacity_level_bounds(profile)
    if config.grid.nz > upper:
        remedy = (
            f"reduce grid.nz to at most {upper}"
            if profile is not None or not spec.capacity_profile_supported
            else f"reduce grid.nz to at most {upper}, or select an explicit runtime.capacity_profile whose columns are compiled deeper"
        )
        refuse(
            code="CM1-CARTESIAN-LEVEL-CAPACITY",
            gate=CARTESIAN_GATE,
            reason=f"the selected {spec.component_id} column capacity is compiled for {lower}..{upper} levels; a deeper column would index per-thread scheme arrays past their compiled extent and return physics computed on foreign memory",
            remedy=remedy,
            option="grid.nz",
            value=config.grid.nz,
        )
    # The stencil floor is the door's own; report the interval this lane
    # actually admits rather than the component's unqualified bounds.
    return max(lower, STENCIL_MINIMUM_NZ), upper


def require_grid_extent(config: CM1Config) -> CartesianGridGeometry:
    """Admit the bound stencil's own extents, then build the geometry."""

    if min(config.grid.nx, config.grid.ny) < 2 * STENCIL_RADIUS_XY + 1:
        refuse(
            code="CM1-CARTESIAN-GRID-EXTENT",
            gate=CARTESIAN_GATE,
            reason=f"the bound fifth-order horizontal stencil reads {STENCIL_RADIUS_XY} cells either side; a shorter row would advect from ghost words that no boundary fills",
            remedy=f"use grid.nx and grid.ny of at least {2 * STENCIL_RADIUS_XY + 1}",
            option="grid",
            value={"nx": config.grid.nx, "ny": config.grid.ny},
        )
    if config.grid.nz < STENCIL_MINIMUM_NZ:
        refuse(
            code="CM1-CARTESIAN-GRID-EXTENT",
            gate=CARTESIAN_GATE,
            reason=f"the bound vertical closure and implicit pressure columns require {STENCIL_MINIMUM_NZ} levels; a shorter column would solve a tridiagonal system that has no interior",
            remedy=f"use grid.nz of at least {STENCIL_MINIMUM_NZ}",
            option="grid.nz",
            value=config.grid.nz,
        )
    # config.py has already built and admitted the coordinate; rebuilding it
    # here keeps one generator and one set of refusals.  A stretch_z=0 grid
    # carries None so that its geometry, receipt and hash stay exactly what
    # they were before the vertical-coordinate words existed; param.F:7789-7811
    # is then rebuilt on demand from nz and dz_m.
    vertical = require_vertical_coordinate(config)
    if vertical.stretch_z == 0:
        vertical = None
    try:
        return CartesianGridGeometry(
            config.grid.nx,
            config.grid.ny,
            config.grid.nz,
            config.grid.dx_m,
            config.grid.dy_m,
            config.grid.dz_m,
            vertical,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-GRID-EXTENT",
            gate=CARTESIAN_GATE,
            reason=f"the Cartesian geometry refuses this grid: {error}",
            remedy="select a uniform grid whose float32 allocation and CUDA dimensions are representable",
            option="grid",
            value=config.grid.nx * config.grid.ny * config.grid.nz,
        )


def measured_steps(config: CM1Config) -> int:
    """Complete large steps implied by the sealed clock.

    Under ``dynamics.adaptive_timestep`` this is the count the *declared*
    step would take, not the count the run will take: misclibs.F:593-810
    chooses every step from the state, so the number is not knowable before
    the run.  It is still the right figure for a benchmark that asks how many
    steps to warm up and measure, which is this function's only caller.
    """

    return int(round(config.run.duration_s / config.run.dt_s))


#: How the Cartesian lane selects its vertical-metric operator family.
#:
#: Left is the operator that reconstructs the column inline from the scalar
#: ``dz`` -- correct for ``param.F:7789-7811``'s uniform grid and for nothing
#: else.  Right is the profile-bound successor this lane runs whenever
#: ``grid.stretch_z != 0``.  The Cartesian lane is ``terrain_flag=.false.``, so
#: the successor runs CM1's flat statements: it binds the generated column's
#: ``mh(1,1,k)``/``mf(1,1,k)`` and multiplies a vertical difference by the
#: reference ``rdz`` and then by that word, ``((d*rdz)*mh(1,1,k))``, and has no
#: scalar-dz fallback.  The third column is the CM1 term whose vertical metric
#: separates the two, and it is the reason the selection is by coordinate and
#: never by preference: on a stretched grid the left-hand operator would form
#: these terms on a column the run does not have.
CARTESIAN_VERTICAL_OPERATOR_FAMILIES = (
    (
        "cuda_psolver3_terrain_open",
        "cuda_gm1_child_psolver3_open",
        "sound.F:119-143,318-342,491-509,742-765 -- the flat vertical pressure "
        "gradient, divergence and implicit column coefficients on rdz*mh(1,1,k) "
        "and rdz*mf(1,1,k), which the uniform operator rebuilds from a scalar dz",
    ),
    (
        "cuda_momentum_advection5",
        "cuda_gm1_child_stretched_momentum",
        "adv.F:1380-1391,3299 -- the vertical momentum flux difference, "
        "((d*rdz)*mh(1,1,k)) and ((d*rdz)*mf(1,1,k)), not d*rdz alone",
    ),
    (
        "cuda_scalar_moist_rk",
        "cuda_gm1_child_stretched_scalar",
        "adv.F:562-572 -- the scalar vertical flux divergence "
        "((d*rdz)*mh(1,1,k)), and pdef.F:482-483,554-555's final-stage chain "
        "on the same column",
    ),
    (
        "cuda_gm1_child_stretched_sgs_tke",
        "cuda_gm1_flat_stretched_closure",
        "turb.F:4766-4895, turbtend.F:60-2885, adv.F:3252-3355 -- calcdef, the "
        "SGS stress and scalar diffusion and TKE advection, on the flat "
        "statements rather than the terrain ones",
    ),
    (
        "cuda_moist_stage_bridge",
        "cuda_gm1_child_stretched_moist",
        "misclibs.F:133-135,329-332 -- the post-acoustic getdiv/getdivx "
        "divergence, ((d*rdz)*mh(1,1,k)) rather than d*rdz",
    ),
)
#: Which compiled operator family each Cartesian large-step phase reaches,
#: by CM1 ``stretch_z``.  ``param.F:7789-7811``'s uniform column is the one
#: where ``rdz*mh`` collapses to ``1/dz`` (``param.F:8170`` makes every ``mh``
#: word exactly one there), so the uniform family binds a scalar ``dz`` and is
#: exact.  Every other ``param.F:7814-8139`` branch reaches the family that
#: binds the generated column's ``mh``/``mf`` with CM1's flat
#: (``terrain_flag=.false.``) statements and carries no ``dz`` argument at
#: all; ``rdz`` is ``1/dz`` of the metric reference (``param.F:6327``).  This
#: table is published in the run receipt so a reader can tell, without the
#: kernels, which column the run actually divided by.
CARTESIAN_OPERATOR_FAMILIES = MappingProxyType({
    "uniform": MappingProxyType({
        "pre_acoustic": "cuda_momentum_advection5.cm1_mom5_*",
        "acoustic": "cuda_psolver3_terrain_open.cm1_tos_*",
        "post_acoustic": "cuda_moist_stage_bridge.cm1_msb_*",
        "scalar_moist": "cuda_advection5.cm1_adv5_* + cuda_scalar_moist_rk.cm1_rk_*",
        "vertical_metric": "scalar dz (param.F:7789-7811; mh=mf=1 at param.F:8170,8176)",
    }),
    "stretched": MappingProxyType({
        "pre_acoustic": "cuda_gm1_child_stretched_momentum.cm1_mom5_* (flat, mh/mf bound)",
        "acoustic": "cuda_gm1_child_psolver3_open.gm1_child_* (flat, mh/mf bound)",
        "post_acoustic": "cuda_gm1_child_stretched_moist.gm1_child_stretch_* (flat, mh bound)",
        "scalar_moist": (
            "cuda_gm1_child_stretched_scalar.cm1_adv5_* + cm1_rk_* (flat, mh bound)"
        ),
        "closure": "cuda_gm1_flat_stretched_closure.gm1_flat_* (flat, mh/mf bound)",
        "vertical_metric": (
            "rdz*mh(1,1,k), rdz*mf(1,1,k) on the generated column "
            "(param.F:6327,8467,8476; terrain_flag=.false.)"
        ),
    }),
})


def cartesian_operator_family(stretch_z: int) -> dict[str, str]:
    """The operator family this ``stretch_z`` reaches, by phase."""

    return dict(CARTESIAN_OPERATOR_FAMILIES["uniform" if stretch_z == 0 else "stretched"])


def cartesian_vertical_operator_record(grid: CartesianGridGeometry) -> dict[str, Any]:
    """Which vertical-metric operator family this grid selects, and why.

    The receipt carries this so a reader can tell, from the document alone,
    which of the two families integrated the run.
    """

    stretched = grid.stretch_z != 0
    return {
        "stretch_z": grid.stretch_z,
        "family": "profile-bound" if stretched else "uniform-dz",
        "authority": "src/param.F:7789-7811" if not stretched else "src/param.F:7814-8209",
        "operators": [
            {
                "uniform_dz": uniform,
                "profile_bound": profile,
                "selected": profile if stretched else uniform,
                "cm1_term": term,
            }
            for uniform, profile, term in CARTESIAN_VERTICAL_OPERATOR_FAMILIES
        ],
    }


def cartesian_output_selection(config: CM1Config) -> Any:
    """The &param9 selection this run executes, by CM1's own key names.

    CM1 has one ``output_dbz`` namelist word and one of each getcape plane
    switch, and this port spells them in the tables that own the things they
    describe: ``physics.output_dbz`` beside the component whose
    scheme-specific formula computes reflectivity, and the four
    ``diagnostics.output_*`` planes beside the ``getcape.F`` parcel controls
    that decide what they mean.  ``[output]`` therefore cannot declare them --
    two spellings of one word could be made to contradict each other -- but
    the &param9 machinery still branches on them, so they are bound here from
    the tables that do own them.

    Rebinding at the door rather than at parse time is deliberate: it leaves
    ``config.output`` exactly as the document wrote it, so a document that
    declares no ``[output]`` table still projects no ``[output]`` key and every
    sealed canonical identity is unchanged.
    """

    from .cm1_output_selection import DEFAULT_OUTPUT_SELECTION

    selection = DEFAULT_OUTPUT_SELECTION if config.output is None else config.output

    environment = config.environment_fields
    return selection.with_values(
        {
            "output_dbz": int(config.physics.output_dbz),
            "output_cape": int(environment.output_cape),
            "output_cin": int(environment.output_cin),
            "output_lcl": int(environment.output_lcl),
            "output_lfc": int(environment.output_lfc),
            # misclibs.F's rotation diagnostics (uh, xvort/yvort/zvort, pv)
            # and solve3.F's swaths (sws..shs) are declared in [diagnostics]
            # beside the tornado_report they feed, and they have ONE producer:
            # cartesian_tornado_diagnostics, which the runtime opens from that
            # table.  It is the superset -- stretched columns, the bbc lower
            # closure, zvort_sfc and CM1's ground-fixed "2" swath twins under
            # a moving frame (writeout.F:568-720), which the &param9 engine
            # does not transcribe.  Binding the same switches here as well
            # made both engines compute and publish uh, xvort, ..., shs, so
            # every deck that selected one refused at frame pricing with two
            # carriers under one name.  The &param9 cross-checks
            # (config.py, CM1-param9) still read these switches from
            # [diagnostics] directly, so no refusal is lost.
            **{key: 0 for key in _TORNADO_OWNED_OUTPUT_KEYS},
        },
        sources=_output_binding_reasons(config),
    )


def _output_binding_reasons(config: CM1Config) -> dict[str, str]:
    """Which table decided each rebound &param9 word, as the receipt states it.

    Without these the record said every such word was "declared in this run's
    [output] table", which no document can do: [output] refuses them by key.
    """

    environment = config.environment_fields
    reasons = {
        "output_dbz": (
            f"bound from physics.output_dbz = {int(config.physics.output_dbz)}: CM1 "
            "has one output_dbz word and this port declares it in [physics], beside "
            "the component whose operator computes it"
        ),
    }
    for key in ("output_cape", "output_cin", "output_lcl", "output_lfc"):
        reasons[key] = (
            f"bound from diagnostics.{key} = {int(getattr(environment, key))}: the "
            "getcape planes are declared in [diagnostics] beside the parcel "
            "controls that define them"
            + ("" if config.diagnostics is not None else
               "; this run declares no [diagnostics] table, which leaves them off "
               "as CM1's distributed deck does")
        )
    declared = config.tornado_diagnostics
    for key in _TORNADO_OWNED_OUTPUT_KEYS:
        if declared is None:
            reasons[key] = (
                f"not published: {key} is declared only in [diagnostics] (its "
                "rotation and swath family) and this run declares no such family; "
                f"declare diagnostics.{key} = 1 to publish it as CM1's deck does"
            )
        elif int(getattr(declared, key)):
            reasons[key] = (
                f"published by the [diagnostics] rotation and swath producer "
                f"(diagnostics.{key} = 1), not by the &param9 engine, whose copy of "
                "the word is 0 so the field is not written twice"
            )
        else:
            reasons[key] = (
                f"diagnostics.{key} = 0 in this run's [diagnostics] table, so the "
                "field is not published"
            )
    return reasons


#: &param9 words whose products cartesian_tornado_diagnostics alone publishes.
_TORNADO_OWNED_OUTPUT_KEYS = (
    "output_uh", "output_vort", "output_pv", "output_sws", "output_svs",
    "output_sps", "output_srs", "output_sgs", "output_sus", "output_shs",
)


def build_runtime_config(config: CM1Config) -> RuntimeConfig:
    """Bind the admitted configuration to the runtime's own contract."""

    grid = require_grid_extent(config)
    # param.F:7814-8139's stretched coordinates are integrated by the
    # profile-bound operator family named in CARTESIAN_VERTICAL_OPERATOR_FAMILIES,
    # which cuda_cartesian_dynamics selects from grid.stretch_z alone.  The
    # refusal that once stood here (CM1-CARTESIAN-UNIFORM-DZ-OPERATORS) named
    # four operators that bound a scalar dz; all four now have a successor
    # bound to the generated column (CM1's flat rdz*mh/rdz*mf statements), so
    # the guard is retired.
    if config.grid.dx_m != config.grid.dy_m:
        refuse(
            code="CM1-CARTESIAN-SQUARE-HORIZONTAL-SPACING",
            gate=CARTESIAN_GATE,
            reason=(
                f"grid.dx_m={config.grid.dx_m} and grid.dy_m={config.grid.dy_m} differ, "
                "but this lane's SGS stress divergence "
                "(cuda_sgs_stress_tendency's cm1_sgs_turbu/turbv/turbw_terrain_v1) "
                "takes one horizontal spacing and forms turb.F:4548-4605's x and y "
                "stress divergences from the same reciprocal, so every subgrid "
                "momentum tendency would be scaled by dx where CM1 scales it by dy"
            ),
            remedy=(
                "set grid.dy_m equal to grid.dx_m, or admit a rectangular-grid "
                "stress kernel that carries rdx and rdy separately"
            ),
            option="grid.dy_m",
            value=config.grid.dy_m,
        )
    runtime = config.runtime
    if runtime is None:
        refuse(
            code="CM1-CARTESIAN-RUNTIME-TABLE",
            gate=CARTESIAN_GATE,
            reason="the Cartesian lane carries no [runtime] table, so tiles, substeps and the device budget would fall back to implicit defaults",
            remedy="declare every [runtime] key explicitly",
            option="runtime",
            value=None,
        )
    limit = grid.ztop_m * 0.5
    if config.closures.upper_damping_depth_m > limit:
        refuse(
            code="CM1-CARTESIAN-DAMPING-DEPTH",
            gate=CARTESIAN_GATE,
            reason=f"the Rayleigh layer would reach below half of this grid's {grid.ztop_m} m top, so damping would act on the troposphere this run is meant to resolve",
            remedy=f"set closures.upper_damping_depth_m to at most {limit}, or raise grid.nz * grid.dz_m",
            option="closures.upper_damping_depth_m",
            value=config.closures.upper_damping_depth_m,
        )
    try:
        bound = RuntimeConfig(
            grid,
            dt_s=config.run.dt_s,
            acoustic_substeps=runtime.acoustic_substeps,
            # Both keys undeclared is one whole-domain closure tile; it is
            # priced against the declared budget by resolve_closure_tiles
            # below, after every other key has been bound and checked.
            tile_nx=grid.nx if runtime.tile_nx is None else runtime.tile_nx,
            tile_ny=grid.ny if runtime.tile_ny is None else runtime.tile_ny,
            memory_budget_bytes=runtime.memory_budget_mib * 1024**2,
            microphysics=config.physics.component,
            # CM1 param.F:1567-1580.  An undeclared ihail binds the branch
            # recorded for the selected component: WSM6 keeps the graupel
            # branch this lane has always executed, Morrison takes CM1's own
            # ptype=5 deck value (run/config_files/supercell/namelist.input:55).
            # The runtime record names the resulting constants by value either way.
            ihail=(
                default_ihail(config.physics.component)
                if config.physics.ihail is None
                else config.physics.ihail
            ),
            damping_depth_m=config.closures.upper_damping_depth_m,
            # src/param.F:724.  CM1 zeroes rdalpha when neither damper is on,
            # so a document that leaves a rate beside irdamp=0 -- which three
            # of CM1's own shipped decks do -- integrates no damping, and the
            # receipt records the rate the run actually used.
            damping_rate_s_inv=(
                config.closures.upper_damping_rate_s_inv
                if (config.closures.upper_damping >= 1 or config.closures.hrdamp >= 1)
                else 0.0
            ),
            # base.F:2661-2674.  config.py has already refused nonzero speeds
            # with closures.moving_frame off, mirroring param.F:701-702, so
            # an inactive frame is exactly (0.0, 0.0) here.
            u_move_m_s=config.closures.u_move_m_s,
            v_move_m_s=config.closures.v_move_m_s,
            # param.F:7212-7213, the one frame the door's extent checks and the
            # runtime's published parcel positions share.
            iorigin=1 if config.initialization is None else config.initialization.iorigin,
            # CM1 param2/param3 rotation, by CM1's own namelist names.
            # solve2.F:631-720 applies the term between divx and advu; with
            # icor=0 nothing is enqueued and the translating frame's exact
            # Galilean invariance is untouched.  With icor=1 the f-plane
            # branch adds the frame velocity back (solve2.F:645,647), which
            # is what keeps a translating run's rotation ground-relative.
            # config.py has already refused betaplane=1 without icor=1, an
            # |fcor| outside asin's domain, and betaplane=1 in a translating
            # frame, so this is a rename-free binding.
            icor=config.dynamics.coriolis,
            betaplane=config.dynamics.beta_plane,
            fcor_s_inv=config.dynamics.fcor_s_inv,
            # README.namelist:531-538.  config.py has already refused 2/3/4 by
            # name, lspgrad=1 without an f-plane icor=1, and lspgrad=1 beside
            # an adaptive moving domain, so this is a rename-free binding.
            lspgrad=config.dynamics.lspgrad_value,
            # init3d.F:1303-1396: iinit = 9, 10 or 12 start the storm with a
            # forcing applied during the run (cm1core.initiation_forcing).  A
            # nest child carries none: CM1 has no nest, and the forcing is the
            # parent's, over the first minutes (nest_forcing_window refusal).
            initiation_forcing=_initiation_forcing(config, grid),
            # base.F's base-state cloud (isnd = 4, 10, 11): solve2.F:404
            # subtracts qc0/qi0 from the live condensate in the buoyancy, so
            # the dynamics holds both base volumes for these soundings.
            base_condensate=(
                config.initialization is not None
                and int(config.initialization.isnd) in BASE_CONDENSATE_ISND
            ),
            # CM1 param3 acoustic reals (README.namelist:751-759).
            alph=config.dynamics.alph,
            kdiv=config.dynamics.kdiv,
            # CM1 param2 adapt_dt (README.namelist:222-225), the controller of
            # misclibs.F:593-810.  run.dt_s is CM1's dtl: the step the run
            # starts at and, by misclibs.F:654, twice the largest the
            # controller may ever choose.
            adaptive_timestep=config.dynamics.adaptive_timestep,
            dtl_s=config.run.dt_s,
            # CM1's lower boundary and surface layer, by CM1's own namelist
            # names.  config.py has already applied every param.F cross-check
            # (bbc range, bbc=3 requires sfcmodel, sfcmodel requires bbc=3,
            # cecd range, cnstcd range), so this is a rename-free binding.
            bbc=config.closures.bbc,
            sfcmodel=config.closures.sfcmodel,
            isfcflx=config.closures.isfcflx,
            cecd=config.closures.cecd,
            cnstcd=config.closures.cnstcd,
            # CM1 README.namelist:1366-1369.  config.py has already refused
            # output_dbz under a component with no admitted reflectivity
            # formula (param.F:1583-1603's own refusal) and a level above the
            # column, so this is a rename-free binding.
            output_dbz=config.physics.output_dbz,
            dbz_level_m=config.physics.dbz_level_m,
            # misclibs.F:1037-1117.  An undeclared key keeps the runtime
            # contract's own default, which is admissible here precisely
            # because the two layouts are raw-word identical: nothing
            # scientific is inherited, only how much of the device the
            # pre-WSM6 repair occupies.
            # mp_driver.F:1469-1479.  Same rule for the WSM6 block extents:
            # an undeclared key follows the closure tile, which is the
            # decomposition every document already ran.
            **{
                name: value
                for name, value in (
                    ("pdefq_launch", runtime.pdefq_launch),
                    ("physics_block_nx", runtime.physics_block_nx),
                    ("physics_block_ny", runtime.physics_block_ny),
                )
                if value is not None
            },
            # CM1's lateral family and its open-boundary companions, by CM1's
            # own names.  config.py has already applied every param.F
            # cross-check, so this is a rename-free binding.
            wbc=config.boundaries.west,
            ebc=config.boundaries.east,
            sbc=config.boundaries.south,
            nbc=config.boundaries.north,
            irbc=config.boundaries.radiative_order,
            roflux=config.boundaries.roflux,
            nudgeobc=config.boundaries.nudgeobc,
            alphobc=config.boundaries.alphobc,
            cstar=config.boundaries.cstar,
            irdamp=config.closures.upper_damping,
            hrdamp=config.closures.hrdamp,
            xhd_m=config.closures.xhd_m,
            cnstce=config.closures.cnstce,
            set_znt=config.closures.set_znt,
            cnst_znt=config.closures.cnst_znt,
            set_ust=config.closures.set_ust,
            cnst_ust=config.closures.cnst_ust,
            # CM1's param12 initial surface conditions.  An absent [surface]
            # table is CM1's own shipped deck (run/namelist.input:130-136),
            # which is what the runtime contract already defaults to, so the
            # binding is the declared table or nothing.
            **(
                {
                    "initsfc": config.surface.initsfc,
                    "tsk0_k": config.surface.tsk0_k,
                    "tmn0_k": config.surface.tmn0_k,
                    "xland0": config.surface.xland0,
                    "lu0": config.surface.lu0,
                    "season": config.surface.season,
                    "oceanmodel": config.surface.oceanmodel,
                }
                if config.surface is not None
                else {}
            ),
            # CM1's radiation, PBL and cumulus families, by CM1's own names.
            # config.py has already applied every cross-check in
            # cm1_physics_families, so this is a rename-free binding.
            rterm=config.closures.rterm_value,
            cm1setup=config.closures.cm1setup,
            ipbl=config.closures.ipbl,
            # README.namelist:334-352 and :1089-1092; config.py has refused
            # every value this lane would integrate as another.
            sgsmodel=config.closures.sgs_model,
            t2p_avg=config.closures.t2p_avg_value,
            # README.namelist:1081-1087; absent is src/cm1.F:355-356's 0.
            ramp_sgs=config.closures.ramp_sgs,
            ramp_time_s=config.closures.ramp_time,
            cuparam=config.physics.cuparam_value,
            radiation=_radiation_selection(config),
            # README.namelist:583-588.  config.py:1521-1525 seals this lane's
            # boundary family, so boundaries.top is CM1's tbc and is carried
            # here by name rather than re-derived at the calcvort call site.
            tbc=config.boundaries.top,
            # CM1 &param9.  An absent [output] table is this lane's own
            # defaults; cm1_output_selection.DEFAULT_MOVES states every one of
            # them against CM1 r21.1's distributed run/namelist.input.
            output=cartesian_output_selection(config),
            # CM1 writeout.F:2895-3030, by CM1's own namelist names.  An absent
            # [diagnostics] table is writeout.F:2895's shipped state -- all four
            # switches off, no column sweep, and a history frame that is byte for
            # byte the one this lane has always published.
            output_cape=config.environment_fields.output_cape,
            output_cin=config.environment_fields.output_cin,
            output_lcl=config.environment_fields.output_lcl,
            output_lfc=config.environment_fields.output_lfc,
            parcel_source=config.environment_fields.parcel_source,
            adiabat=config.environment_fields.adiabat,
            mixed_layer_depth_m=config.environment_fields.mixed_layer_depth_m,
            pressure_increment_pa=config.environment_fields.pressure_increment_pa,
            storm_motion=config.environment_fields.storm_motion,
            # CM1's output_* diagnostic switches.  config.py has already
            # applied every grid, microphysics and lower-boundary cross-check
            # (_validate_diagnostics), so this is a rename-free binding; an
            # absent [diagnostics] table binds the all-off selection that
            # every earlier run carried.
            diagnostics=tornado_diagnostic_selection(config),
            # The optional [lagrangian] table, already bound and range-checked
            # by lagrangian_config.parse_lagrangian_table.  What is checked
            # here instead is the part that needs the *grid*: a high-frequency
            # window and a circuit seeded outside the domain.
            lagrangian=_require_lagrangian_extent(config, grid),
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-RUNTIME-ENVELOPE",
            gate=CARTESIAN_GATE,
            reason=f"the Cartesian runtime contract refuses this configuration: {error}",
            remedy="correct the named option; the runtime contract is not widened from the door",
            option="runtime",
            value=config.sha256,
        )
    if runtime.tile_nx is None:
        return resolve_closure_tiles(
            bound, physics_block_declared=runtime.physics_block_nx is not None
        )
    return bound


#: The explicit tiles an untiled refusal may recommend, largest first.
_MEASURED_CLOSURE_TILES = (128, 64, 32, 16)


def closure_decomposition_peak_bytes(runtime_config: RuntimeConfig) -> dict[str, int]:
    """The registry peak one closure decomposition prices, CUDA-cold.

    The dynamics phase peak (its acoustic workspace survives into the next
    step's closure, ``CartesianFinish.microphysics`` is what releases it) is
    held together with the persistent closure profiles, the surface state, the
    reflectivity residents, and the largest of the three phase workspaces that
    never coexist: the held-tendency tile, the TKE stage and one physics
    block.  Component scratch outside the registry is reserved separately by
    the runtime against measured free memory, not against this budget.
    """

    from .cuda_cartesian_closure import cartesian_closure_allocation_plan
    from .cuda_cartesian_dynamics import cartesian_dynamics_allocation_plan
    from .cuda_cartesian_finish import (
        cartesian_physics_block_payload_bytes,
        cartesian_reflectivity_allocation_plan,
    )

    dynamics = cartesian_dynamics_allocation_plan(runtime_config)
    closure = cartesian_closure_allocation_plan(runtime_config)
    physics = cartesian_physics_block_payload_bytes(runtime_config)
    rows = {
        "dynamics_phase_peak_bytes": int(dynamics["maximum_listed_phase_payload_bytes"]),
        "closure_persistent_bytes": int(closure["persistent_profile_payload_bytes"])
        + int(closure.get("surface_state_payload_bytes", 0)),
        "reflectivity_resident_bytes": int(
            cartesian_reflectivity_allocation_plan(runtime_config)["resident_total_bytes"]
        ),
        "held_phase_bytes": int(closure["held_phase_payload_bytes"]),
        "tke_phase_bytes": int(closure["tke_phase_payload_bytes"]),
        "physics_block_bytes": int(physics),
    }
    rows["priced_peak_bytes"] = (
        rows["dynamics_phase_peak_bytes"]
        + rows["closure_persistent_bytes"]
        + rows["reflectivity_resident_bytes"]
        + max(rows["held_phase_bytes"], rows["tke_phase_bytes"], rows["physics_block_bytes"])
    )
    return rows


def resolve_closure_tiles(
    bound: RuntimeConfig, *, physics_block_declared: bool
) -> RuntimeConfig:
    """Admit one whole-domain closure tile for a document that declares none.

    The decomposition decides no word the model computes: the closure's
    operators read at most three cells either side of the cell they write,
    each tile gathers that halo from the resident state and scatters only the
    cells it owns, and ``tools/oracle/cartesian_tiling_identity.py`` runs the
    real closure both ways and compares every published carrier raw word for
    raw word.  What the decomposition does decide is how much workspace one
    tile holds and how many times per large step the host launches the
    closure's ~170 operations.  One tile launches them once.

    It is admitted only when the priced peak fits the declared budget.  A
    budget it does not fit is refused, naming the largest explicit tile that
    does, rather than falling back to a tile silently: CM1's cecd=2/3 water
    roughness iteration is the one path whose tiled words differ from the
    untiled ones (docs/GM1-CARTESIAN-RUNTIME.md), so a silent fallback would
    change what such a run computes behind a document that asked for neither.
    """

    from dataclasses import replace

    from .gm1_runtime import MAX_PHYSICS_BLOCK_CELLS

    grid = bound.grid
    budget = int(bound.memory_budget_bytes)
    if not physics_block_declared and max(grid.nx, grid.ny) > MAX_PHYSICS_BLOCK_CELLS:
        refuse(
            code="CM1-CARTESIAN-UNTILED-PHYSICS-BLOCK",
            gate=CARTESIAN_GATE,
            reason=(
                f"runtime.tile_nx/tile_ny are undeclared, so the closure runs as one "
                f"{grid.nx} x {grid.ny} tile and the undeclared WSM6 block follows it; "
                f"a block wider than {MAX_PHYSICS_BLOCK_CELLS} cells would stage its "
                "column volumes past the admitted block size and could fail to "
                "allocate after the step has advanced past the pre-physics seam"
            ),
            remedy=(
                f"declare runtime.physics_block_nx and physics_block_ny of at most "
                f"{MAX_PHYSICS_BLOCK_CELLS}, or declare runtime.tile_nx and tile_ny"
            ),
            option="runtime.tile_nx",
            value=[grid.ny, grid.nx],
        )
    priced = closure_decomposition_peak_bytes(bound)
    if priced["priced_peak_bytes"] > budget:
        fitting = None
        for tile in _MEASURED_CLOSURE_TILES:
            candidate = replace(
                bound, tile_nx=tile, tile_ny=tile,
                **({} if physics_block_declared else {
                    "physics_block_nx": tile, "physics_block_ny": tile,
                }),
            )
            if closure_decomposition_peak_bytes(candidate)["priced_peak_bytes"] <= budget:
                fitting = tile
                break
        mib = -(-priced["priced_peak_bytes"] // 1024**2)
        refuse(
            code="CM1-CARTESIAN-UNTILED-BUDGET",
            gate=CARTESIAN_GATE,
            reason=(
                "runtime.tile_nx/tile_ny are undeclared, so the closure would run as "
                f"one {grid.nx} x {grid.ny} tile, and its priced device peak of "
                f"{priced['priced_peak_bytes']} bytes exceeds runtime.memory_budget_mib "
                f"({budget} bytes); the registry would refuse the held-tendency "
                "workspace inside the first large step"
            ),
            remedy=(
                f"raise runtime.memory_budget_mib to at least {mib}"
                + (
                    f", or declare runtime.tile_nx = runtime.tile_ny = {fitting}"
                    if fitting is not None
                    else ""
                )
            ),
            option="runtime.memory_budget_mib",
            value=budget // 1024**2,
        )
    return replace(
        bound,
        tile_selection={
            "declared": False,
            "decomposition": "one whole-domain closure tile",
            "tile_shape_yx": [grid.ny, grid.nx],
            "priced_peak_bytes": priced["priced_peak_bytes"],
            "memory_budget_bytes": budget,
            "rule": (
                "runtime.tile_nx and tile_ny undeclared: one whole-domain tile, "
                "admitted because its priced peak fits the declared budget"
            ),
        },
    )


def _radiation_selection(config: CM1Config) -> Any:
    """CM1's ``param11`` selection for this document, or its radopt=0 identity."""

    from .cm1_physics_families import Cm1RadiationSelection, NO_RADIATION

    table = config.radiation
    if table is None:
        return NO_RADIATION
    return Cm1RadiationSelection(
        radopt=table.radopt,
        dtrad_s=table.dtrad_s,
        ctrlat=table.ctrlat,
        ctrlon=table.ctrlon,
        year=table.year,
        month=table.month,
        day=table.day,
        hour=table.hour,
        minute=table.minute,
        second=table.second,
    )


@dataclass(frozen=True, slots=True)
class NestBinding:
    """One admitted nest: its geometry, its child runtime and its record.

    This is what turns ``[nest]`` in a TOML into something the integration
    entry point can drive.  The child is a full
    :class:`~cm1core.gm1_runtime.RuntimeConfig` in its own right -- the same
    contract the parent is bound to -- because a child of a CM1 parent is a CM1
    grid and must be admitted by CM1's own rules, not by a looser child-only
    envelope.
    """

    geometry: "NestGeometry"
    child: RuntimeConfig
    follow_storm: bool
    follow_interval_s: float
    follow_max_cells: int
    follow_interval_steps: int
    follow_max_speed_m_s: float = 0.0
    follow_tracker: str = "cm1-pressure-held-offset"
    follow_smoothing_radius_m: float | None = None
    follow_corral_m: float | None = None
    follow_on_edge: str | None = None
    follow_hold_limit: int | None = None
    follow_relocation_radius_m: float | None = None
    #: ``[nest.tornado_criterion]`` (cm1core.nest_tornado_criterion), or None.
    tornado_criterion: "object | None" = None

    def follow_settings(self) -> "object | None":
        """The live nest's following contract (:class:`cm1core.nest_move.FollowSettings`)."""

        if not self.follow_storm:
            return None
        from .nest_move import FollowSettings, corral_cells

        return FollowSettings(
            tracker=self.follow_tracker,
            interval_steps=self.follow_interval_steps,
            interval_s=self.follow_interval_s,
            max_speed_m_s=self.follow_max_speed_m_s,
            smoothing_radius_m=self.follow_smoothing_radius_m,
            corral_cells_x=corral_cells(self.follow_corral_m, self.geometry.parent_dx_m),
            corral_cells_y=corral_cells(self.follow_corral_m, self.geometry.parent_dy_m),
            on_edge=self.follow_on_edge,
            hold_limit=self.follow_hold_limit,
            relocation_radius_m=self.follow_relocation_radius_m,
        )

    def record(self, *, parent_dt_s: float) -> dict[str, object]:
        from .nest_boundary import nest_boundary_record
        from .nest_coupler import NEST_COUPLER_SCHEMA, parent_interval_schedule

        payload = dict(self.geometry.record(parent_dt_s=parent_dt_s))
        payload["child_runtime"] = self.child.record()
        payload["coupling"] = {
            "schema": NEST_COUPLER_SCHEMA,
            "order": (
                "one parent large step completes; the boundary value and "
                "tendency tables are built once from the parent's state at the "
                "END of that interval and dtbc is reset "
                "(share/interp_fcn.F:2583-2584, share/mediation_force_domain.F:213); "
                "the child then takes its sub-steps, each first advancing "
                "dtbc by one child step (dyn_em/solve_em.F:372), so sub-step n "
                "relaxes toward the target at (n+1)*child_dt; two-way feedback, "
                "when selected, writes the child's interior block means into the "
                "parent's covered cells afterwards "
                "(frame/module_integrate.F:393-443)"
            ),
            "dtbc_s": [
                step.dtbc
                for step in parent_interval_schedule(
                    self.geometry, parent_dt_s=parent_dt_s
                )
            ],
        }
        payload["lateral_forcing"] = nest_boundary_record(
            self.geometry, child_dt_s=self.child.dt_s
        )
        from .nest_geometry import (
            NEST_PSMTH_KERNEL_NAME,
            NEST_PSMTH_KERNEL_SHA256,
        )

        payload["storm_following"] = {
            "follow_storm": self.follow_storm,
            "tracker_kernel": NEST_PSMTH_KERNEL_NAME,
            "tracker_kernel_sha256": NEST_PSMTH_KERNEL_SHA256,
            "follow_interval_s": self.follow_interval_s,
            "follow_interval_parent_steps": self.follow_interval_steps,
            "follow_max_speed_m_s": self.follow_max_speed_m_s,
            "follow_max_parent_cells_per_recentring": self.follow_max_cells,
            "follow_tracker": self.follow_tracker,
            "follow_smoothing_radius_m": self.follow_smoothing_radius_m,
            "follow_corral_m": self.follow_corral_m,
            "live_motion": (
                "a running child moves (cm1core.nest_move): WRF's med_nest_move "
                "after med_nest_feedback (frame/module_integrate.F:442-449), at most "
                "one parent cell per axis per parent step toward the tracked centre "
                "(share/mediation_nest_move.F:122-127,659-704), stopped by name at "
                "the corral (:816-859); the exposed strip is captured from the "
                "parent and every other child word is shifted "
                "(dyn_em/shift_domain_em.F:52-80)"
                if self.follow_storm else None
            ),
            "tracker_authority": (
                "CM1 r21.1 src/azimavg.F:4071-4107 (100 passes of the 9-point "
                "0.11111111 box smoother on the lowest level of pp3d, in CM1's "
                "own summation order), src/azimavg.F:4122-4139 (corner "
                "interpolation to zeta points and the minimum) and "
                "src/azimavg.F:4159-4163 (the centre is reported as a grid "
                "face). The cadence follows src/azimavg.F:4421-4435's "
                "adapt_move_frq. Moving a sub-domain rather than the whole "
                "domain is this port's own extension: CM1's own adaptive move "
                "(src/azimavg.F:4376-4528) translates the single domain."
            ),
            "quantisation": (
                "whole parent cells. A fractional shift would take the child's "
                "outer u and v face columns off the parent's face columns, so "
                "the interface flux would stop being the parent's own"
            ),
        }
        if self.follow_on_edge is not None:
            # Projected only when declared, so a receipt of a document written
            # before the keys existed keeps its bytes.
            following = payload["storm_following"]
            following["follow_on_edge"] = self.follow_on_edge
            following["follow_hold_limit"] = self.follow_hold_limit
            following["follow_relocation_radius_m"] = self.follow_relocation_radius_m
            if self.follow_on_edge == "hold":
                following["live_motion"] = (
                    "a running child moves (cm1core.nest_move): WRF's med_nest_move "
                    "after med_nest_feedback (frame/module_integrate.F:442-449), at "
                    "most one parent cell per axis per parent step toward the "
                    "tracked centre (share/mediation_nest_move.F:122-127,659-704); "
                    "a tracker sample that finds no cyclonic circulation, whose "
                    "maximum is on the search-window edge, or whose centroid lies "
                    "beyond the search reach keeps the previous centre and is "
                    "listed; the reach is follow_max_speed_m_s*follow_interval_s "
                    "times the follow intervals since the last accepted centre and "
                    "the window one child cell wider than it; after follow_hold_limit "
                    "consecutive held samples the next one relocates the centre to "
                    "the strongest circulation within follow_relocation_radius_m, "
                    "listed, and the nest moves toward it one parent cell per parent "
                    "step, or the run stops by name when none is found there; a "
                    "move into the corral is cancelled per axis (:849-859) for at "
                    "most follow_hold_limit*follow_interval parent steps; the "
                    "exposed strip is captured from the parent and every other "
                    "child word is shifted (dyn_em/shift_domain_em.F:52-80)"
                )
        if self.tornado_criterion is not None:
            # Present only when declared, so every other nest plan keeps its bytes.
            from .nest_tornado_criterion import (
                CM1_AUTHORITY,
                CRITERION,
                CRITERION_SHA256,
                NEST_TORNADO_CRITERION_SCHEMA,
                NEST_TORNADO_KERNEL_SHA256,
                WORDING,
            )

            payload["tornado_criterion"] = {
                "schema": NEST_TORNADO_CRITERION_SCHEMA,
                "criterion": dict(CRITERION),
                "criterion_sha256": CRITERION_SHA256,
                "kernel_sha256": NEST_TORNADO_KERNEL_SHA256,
                "stop_rules": self.tornado_criterion.record(),
                "sampled": "every parent step, before feedback and any nest move",
                "cm1_authority": dict(CM1_AUTHORITY),
                "wording": WORDING,
            }
        return payload


def require_nest(config: CM1Config) -> "NestBinding | None":
    """Bind ``[nest]`` to a geometry and a child runtime, or refuse.

    ``None`` means the document declares no nest, which is the single-domain
    run this lane has always been: nothing is allocated, nothing is priced and
    the canonical document projects no nest key, so the existing default stays
    bit-identical.

    ``config.py`` has already applied every cross-field check that needs the
    whole document -- parity, zone order, placement, CM1's ``dtsmall`` verdict
    on the child, the follow cadence -- so this is a binding, not a second
    validator.  What it adds is the child's own
    :class:`~cm1core.gm1_runtime.RuntimeConfig`, which is where a wrong child
    would finally be caught by the same contract the parent passes.
    """

    from .nest_child_window import require_nest_child_window_admission
    from .nest_parcels import require_nest_parcel_admission

    nest = config.nest
    if nest is None:
        require_nest_parcel_admission(config, None)
        require_nest_child_window_admission(config, None)
        return None
    from .nest_geometry import NestGeometry

    initialization = config.initialization
    if initialization is not None and int(initialization.iinit) in (9, 10, 12):
        refuse(
            code="CM1-CARTESIAN-NEST-FORCING",
            gate=CARTESIAN_GATE,
            reason=(
                f"initialization.iinit = {initialization.iinit} starts the storm with a "
                "forcing applied to the parent during the first minutes (init3d.F:1303-1396); "
                "a live nest's child exists from the first step and carries no such term, so "
                "the child would integrate the unforced air under the parent's forced updraft "
                "and the two would disagree across the specified zone"
            ),
            remedy=(
                "run the forced start single-domain, write a restart after the forcing "
                "window, and nest from that restart"
            ),
            option="nest",
            value=initialization.iinit,
        )
    if initialization is not None and int(initialization.isnd) in BASE_CONDENSATE_ISND:
        refuse(
            code="CM1-CARTESIAN-NEST-BASE-CONDENSATE",
            gate=CARTESIAN_GATE,
            reason=(
                f"initialization.isnd = {initialization.isnd} builds a base state with cloud "
                "(base.F qc0/qi0), which solve2.F:404 subtracts in the buoyancy; a live nest's "
                "child is initialized from the parent's pi0/th0/qv0 columns only and would carry "
                "no qc0/qi0, so its buoyancy would differ from the parent's by the base cloud's "
                "weight at every point"
            ),
            remedy="run this sounding single-domain (CM1 r21.1 has no nest)",
            option="nest",
            value=initialization.isnd,
        )
    parent = require_grid_extent(config)
    geometry = NestGeometry(
        parent_nx=config.grid.nx,
        parent_ny=config.grid.ny,
        parent_dx_m=config.grid.dx_m,
        parent_dy_m=config.grid.dy_m,
        refinement_ratio=nest.refinement_ratio,
        time_refinement_ratio=nest.time_refinement_ratio,
        parent_i0=nest.parent_i0,
        parent_j0=nest.parent_j0,
        parent_span_x=nest.parent_span_x,
        parent_span_y=nest.parent_span_y,
        spec_zone=nest.spec_zone,
        relax_zone=nest.relax_zone,
        feedback=nest.feedback,
    )
    from .config import nest_child_vertical_coordinate

    # The child's own deck when [nest] declares one (config.py has already
    # refused a different top or a child coarser than its parent), else the
    # parent's column word for word.
    child_levels = nest_child_vertical_coordinate(config)
    child_nz = parent.nz if child_levels is None else child_levels.nk
    try:
        child_grid = CartesianGridGeometry(
            geometry.nx,
            geometry.ny,
            child_nz,
            geometry.dx_m,
            geometry.dy_m,
            parent.dz_m,
            parent.vertical if child_levels is None else child_levels,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-NEST-GRID",
            gate=CARTESIAN_GATE,
            reason=f"the refined child geometry is inadmissible: {error}",
            remedy=(
                "reduce nest.refinement_ratio or the nest span; the child is "
                "allocated as a full CM1 grid and is priced as one"
            ),
            option="nest.refinement_ratio",
            value=[geometry.nx, geometry.ny, child_nz],
        )
    runtime = config.runtime
    assert runtime is not None  # build_runtime_config refuses an absent table
    try:
        child = RuntimeConfig(
            child_grid,
            dt_s=geometry.child_dt_s(config.run.dt_s),
            acoustic_substeps=nest.acoustic_substeps,
            tile_nx=child_grid.nx if runtime.tile_nx is None else runtime.tile_nx,
            tile_ny=child_grid.ny if runtime.tile_ny is None else runtime.tile_ny,
            memory_budget_bytes=runtime.memory_budget_mib * 1024**2,
            microphysics=config.physics.component,
            damping_depth_m=config.closures.upper_damping_depth_m,
            damping_rate_s_inv=config.closures.upper_damping_rate_s_inv,
            # The child lives in the parent's frame.  base.F:2667 subtracts the
            # same umove/vmove from both base states, so a child with a
            # different frame would be handed lateral forcing whose winds were
            # relative to a frame its interior did not use.
            u_move_m_s=config.closures.u_move_m_s,
            v_move_m_s=config.closures.v_move_m_s,
            bbc=config.closures.bbc,
            sfcmodel=config.closures.sfcmodel,
            isfcflx=config.closures.isfcflx,
            cecd=config.closures.cecd,
            cnstcd=config.closures.cnstcd,
            # The child integrates the parent's rotation and its lspgrad
            # forcing (nest_live.child_runtime_config); the latter holds two
            # per-level profiles for the life of the run, so the price names it.
            icor=config.dynamics.coriolis,
            betaplane=config.dynamics.beta_plane,
            fcor_s_inv=config.dynamics.fcor_s_inv,
            lspgrad=config.dynamics.lspgrad_value,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-NEST-RUNTIME",
            gate=CARTESIAN_GATE,
            reason=(
                "the Cartesian runtime contract refuses the child this nest "
                f"implies: {error}"
            ),
            remedy=(
                "correct the named nest key; the child is admitted by the same "
                "contract as the parent and the contract is not widened for it"
            ),
            option="nest",
            value={
                "nx": geometry.nx,
                "ny": geometry.ny,
                "dx_m": geometry.dx_m,
                "dt_s": geometry.child_dt_s(config.run.dt_s),
                "acoustic_substeps": nest.acoustic_substeps,
            },
        )
    if runtime.tile_nx is None:
        # The child is priced as its own full CM1 grid under the same budget
        # the parent's registry uses, so it is admitted by the same rule.
        child = resolve_closure_tiles(child, physics_block_declared=False)
    steps = (
        int(round(nest.follow_interval_s / config.run.dt_s))
        if nest.follow_storm
        else 0
    )
    binding = NestBinding(
        geometry=geometry,
        child=child,
        follow_storm=nest.follow_storm,
        follow_interval_s=nest.follow_interval_s,
        follow_max_cells=(
            nest_follow_max_cells(nest, parent_dx_m=config.grid.dx_m)
            if nest.follow_storm else 0
        ),
        follow_interval_steps=steps,
        follow_max_speed_m_s=nest.follow_max_speed_m_s,
        follow_tracker=nest.follow_tracker,
        follow_smoothing_radius_m=nest.follow_smoothing_radius_m,
        follow_corral_m=nest.follow_corral_m,
        follow_on_edge=nest.follow_on_edge,
        follow_hold_limit=nest.follow_hold_limit,
        follow_relocation_radius_m=nest.follow_relocation_radius_m,
        tornado_criterion=nest.tornado_criterion,
    )
    require_nest_parcel_admission(config, binding)
    require_nest_child_window_admission(config, binding)
    return binding


def require_nest_execution(
    config: CM1Config, *, initial_condition_only: bool
) -> None:
    """Refuse the nested runs the live nest does not execute, each by name.

    A nest over a fixed window is integrated concurrently with its parent
    (:mod:`cm1core.nest_live`).  What is refused here is exactly what that
    driver does not do, and each refusal names the wrong result it prevents.
    """

    nest = config.nest
    if nest is None:
        return
    value = {
        "refinement_ratio": nest.refinement_ratio,
        "child_nx": nest.parent_span_x * nest.refinement_ratio,
        "child_ny": nest.parent_span_y * nest.refinement_ratio,
    }
    lateral = (
        config.boundaries.west, config.boundaries.east,
        config.boundaries.south, config.boundaries.north,
    )
    if (
        nest.follow_storm
        and nest.follow_tracker != "wrf-surface-circulation"
        and lateral != (2, 2, 2, 2)
    ):
        # The moving nest itself is implemented (cm1core.nest_move).  What is
        # not is CM1's pressure tracker on a parent whose lateral halo is not
        # the open one: device_pressure_centre_cell fills the smoother's halo
        # by zero-gradient extension, which is bc2d for code-2 boundaries only
        # (src/bc.F:12-172); a periodic or rigid parent wraps or mirrors it.
        refuse(
            code="CM1-CARTESIAN-NEST-FOLLOW-INTEGRATION",
            gate=CARTESIAN_GATE,
            reason=(
                f"nest.follow_tracker = {nest.follow_tracker!r} runs CM1's "
                "getpsmth smoother with the open boundary's zero-gradient halo, "
                f"but the parent's lateral boundaries are {list(lateral)}; CM1's "
                "own bc2d (src/bc.F:12-172) fills that halo differently there, so "
                "near the parent's edges the tracker would reach a pressure "
                "minimum CM1 would not and move the running child toward it"
            ),
            remedy=(
                "use four open (code 2) parent boundaries, or select "
                "nest.follow_tracker = \"wrf-surface-circulation\", which runs on "
                "the child's own interior"
            ),
            option="nest.follow_tracker",
            value={**value, "lateral_boundaries": list(lateral)},
        )
    # A nested run writes restarts: the child's own restart is written beside
    # the parent's at every restart time and a resume reads it back
    # (cm1core.nest_restart, WRF share/mediation_integrate.F:861-892,1157-1182).
    # CM1-CARTESIAN-NEST-RESTART-WRITE is retired with the defect it named.
    if initial_condition_only:
        refuse(
            code="CM1-CARTESIAN-NEST-INITIAL-CONDITION",
            gate=CARTESIAN_GATE,
            reason=(
                "run.iconly = 1 publishes the parent's initial condition without "
                "opening a device, and the child's initial condition is the "
                "parent's captured onto the child on the device; the publication "
                "would name a nest whose child was never built"
            ),
            remedy="drop run.iconly for the nested document, or drop the [nest] table",
            option="run.iconly",
            value=config.run.iconly,
        )
    if config.dynamics.adaptive_timestep:
        refuse(
            code="CM1-CARTESIAN-NEST-ADAPTIVE-STEP",
            gate=CARTESIAN_GATE,
            reason=(
                "dynamics.adaptive_timestep changes the parent's step between "
                "steps, and the child's step is the parent's divided by "
                "nest.time_refinement_ratio; the child would keep integrating at "
                "the first step's clock while its boundary tables were built for "
                "another interval"
            ),
            remedy="set dynamics.adaptive_timestep = false for the nested run",
            option="dynamics.adaptive_timestep",
            value=True,
        )
    if config.closures.do_adapt_move:
        refuse(
            code="CM1-CARTESIAN-NEST-ADAPTIVE-FRAME",
            gate=CARTESIAN_GATE,
            reason=(
                "closures.do_adapt_move changes the parent's umove/vmove during "
                "the run, and the child is bound to the parent's frame at build "
                "time (base.F:2667 subtracts it from the child's base state); the "
                "child would be forced with winds relative to a frame its own "
                "base state no longer uses"
            ),
            remedy="use a fixed translating frame for the nested run",
            option="closures.do_adapt_move",
            value=True,
        )
    if config.dynamics.beta_plane:
        refuse(
            code="CM1-CARTESIAN-NEST-BETA-PLANE",
            gate=CARTESIAN_GATE,
            reason=(
                "betaplane = 1 evaluates f about each grid's own centery "
                "(param.F:8966), so the child would carry the parent-centre "
                "Coriolis parameter at its own centre rather than at its "
                "position in the parent"
            ),
            remedy="use the f-plane (dynamics.beta_plane = 0) for the nested run",
            option="dynamics.beta_plane",
            value=1,
        )


def nest_allocation_plan(binding: "NestBinding | None") -> dict[str, object]:
    """Price the child exactly as the parent is priced, or say there is none."""

    if binding is None:
        return {"nest": None}
    return {
        "nest": binding.record(parent_dt_s=binding.child.dt_s * binding.geometry.time_refinement_ratio),
        "nest_allocation": allocation_plan(binding.child),
    }


def require_sounding_report(
    config: CM1Config,
    *,
    library: Path,
    sounding_bytes: bytes,
    selection: "InitializerSelection",
) -> dict[str, object]:
    """This run's own t=0 environment parameters, or a refusal naming why not.

    ``base.F:3143-3189`` prints this for every moist CM1 run and gates it on
    nothing, so neither does this: the parameters of the atmosphere a run was
    given belong in its receipt whether or not the 2-D planes are switched on.
    The ``getcape.F`` source parameters and the storm motion come from the
    ``[diagnostics]`` table when the document declares one and from CM1's own
    shipped values when it does not.
    """

    fields = config.environment_fields
    grid = require_grid_extent(config)
    return initial_sounding_report(
        library,
        sounding_bytes,
        isnd=selection.isnd,
        iwnd=selection.iwnd,
        nz=grid.nz,
        dz_m=grid.dz_m,
        levels=grid.vertical,
        selection=SoundingReportSelection(
            adiabat=fields.adiabat,
            mixed_layer_depth_m=fields.mixed_layer_depth_m,
            pressure_increment_pa=fields.pressure_increment_pa,
            storm_motion=fields.storm_motion,
        ),
        # base.F:2663-2669.  The report adds this frame back before reading the
        # hodograph: a Bunkers motion measured in a frame that already
        # translates with the storm would report a storm-relative helicity near
        # zero for any sounding at all.
        u_move_m_s=config.closures.u_move_m_s,
        v_move_m_s=config.closures.v_move_m_s,
        # base.F's words the column depends on beyond isnd/iwnd.
        testcase=selection.testcase,
        imoist=selection.imoist,
        thec_mb_k=selection.thec_mb_k,
        qt_mb=selection.qt_mb,
        tsk0_k=selection.tsk0_k,
        eqtset=selection.eqtset,
    )


def _require_lagrangian_extent(config: CM1Config, grid: CartesianGridGeometry):
    """Refuse a Lagrangian selector that names cells outside this grid.

    ``lagrangian_config`` refuses every value that is wrong on its own terms.
    Two are only wrong against a grid, and both are refused here rather than
    clipped, because clipping would publish a different volume -- or a
    different circuit -- from the one the sealed document names.
    """

    lagrangian = getattr(config, "lagrangian", None)
    if lagrangian is None:
        return None
    from .high_frequency_output import HighFrequencyWindow

    values = lagrangian.values
    if lagrangian.hifrq_enabled and not (
        values["hifrq_i1"] == 0 and values["hifrq_j1"] == 0 and values["hifrq_k1"] == 0
    ):
        try:
            window = HighFrequencyWindow(
                int(values["hifrq_i0"]), int(values["hifrq_i1"]),
                int(values["hifrq_j0"]), int(values["hifrq_j1"]),
                int(values["hifrq_k0"]), int(values["hifrq_k1"]),
            )
            window.require_within(nx=grid.nx, ny=grid.ny, nz=grid.nz)
        except ValueError as error:
            refuse(
                code="CM1-CARTESIAN-HIFRQ-WINDOW",
                gate=CARTESIAN_GATE,
                reason=str(error),
                remedy=(
                    "set lagrangian.hifrq_i0/i1/j0/j1/k0/k1 inside the grid, or "
                    "leave all three upper bounds at 0 for hifrq.F's own "
                    "whole-domain extent"
                ),
                option="lagrangian.hifrq_i1",
                value=int(values["hifrq_i1"]),
            )
    if lagrangian.pdcomp_enabled:
        from .cuda_pdcomp_open import PdcompSelection, pdcomp_selection_defect

        defect = pdcomp_selection_defect(PdcompSelection(
            ni=grid.nx, nj=grid.ny, nk=grid.nz,
            dx=grid.dx_m, dy=grid.dy_m, dz=grid.dz_m, dt=float(config.run.dt_s),
            wbc=config.boundaries.west, ebc=config.boundaries.east,
            sbc=config.boundaries.south, nbc=config.boundaries.north,
            icor=config.dynamics.coriolis, psolver=config.dynamics.psolver,
            terrain=bool(config.grid.terrain),
        ))
        if defect is not None:
            refuse(
                code="CM1-CARTESIAN-PDCOMP",
                gate=CARTESIAN_GATE,
                reason=defect,
                remedy="set lagrangian.output_pdcomp = 0, or change the named word",
                option="lagrangian.output_pdcomp",
                value=1,
            )
    if lagrangian.circuits_enabled:
        parcel_grid = lagrangian.parcel_grid(
            nx=grid.nx, ny=grid.ny, nz=grid.nz,
            dx_m=grid.dx_m, dy_m=grid.dy_m, dz_m=grid.dz_m,
            # param.F:7212-7213's frame, declared once in [initialization].
            iorigin=(
                1 if config.initialization is None else config.initialization.iorigin
            ),
        )
        radius = float(values["circuit_radius_m"])
        cx, cy = float(values["circuit_center_x_m"]), float(values["circuit_center_y_m"])
        top = float(values["circuit_center_z_m"]) + float(values["circuit_spacing_z_m"]) * (
            int(values["circuits"]) - 1
        )
        outside = (
            cx - radius < parcel_grid.minx_m or cx + radius > parcel_grid.maxx_m
            or cy - radius < parcel_grid.miny_m or cy + radius > parcel_grid.maxy_m
            or float(values["circuit_center_z_m"]) <= 0.0
            or top >= parcel_grid.maxz_m
        )
        if outside:
            refuse(
                code="CM1-CARTESIAN-CIRCUIT-EXTENT",
                gate=CARTESIAN_GATE,
                reason=(
                    "a material circuit seeded outside the domain has vertices "
                    f"that parcel.F:577-592 wraps periodically to the far side "
                    "on the first step, so its line integral would close across "
                    "the whole domain and report a circulation that is not the "
                    "circuit's; this grid spans x "
                    f"[{parcel_grid.minx_m}, {parcel_grid.maxx_m}], y "
                    f"[{parcel_grid.miny_m}, {parcel_grid.maxy_m}], z "
                    f"(0, {parcel_grid.maxz_m})"
                ),
                remedy=(
                    "move lagrangian.circuit_center_x_m/y_m/z_m or shrink "
                    "lagrangian.circuit_radius_m so every ring lies inside the grid"
                ),
                option="lagrangian.circuit_radius_m",
                value=radius,
            )
    return lagrangian


def moving_frame_record(frame: "TranslatingFrame") -> dict[str, object]:
    """What a reader needs to map this run's coordinates back to the ground.

    Without it every position a viewer reports is wrong by ``u_move_m_s``
    times the model time, with nothing in the artifact to say so.

    This is the run's constant, declared before a step is taken.  Where the
    domain actually is at a given model time is not left to the reader: the
    runtime integrates ``domainlocx``/``domainlocy`` exactly as
    ``cm1.F:3512-3513`` does and publishes the pair with every history frame.
    """

    return {
        "imove": frame.imove,
        "u_move_m_s": frame.u_move_m_s,
        "v_move_m_s": frame.v_move_m_s,
        "authority": "CM1 r21.1 base.F:2663-2669",
        "domain_location_authority": DOMAIN_LOCATION_AUTHORITY,
        "convention": (
            "every published wind is frame-relative; the ground-relative wind is "
            "u_move_m_s + u and v_move_m_s + v (init3d.F:32-33). The domain's own "
            "ground position at each published model time is carried per frame as "
            f"{DOMAIN_LOCATION_X_NAME}/{DOMAIN_LOCATION_Y_NAME} in "
            f"{DOMAIN_LOCATION_UNITS}, the running integral of cm1.F:3512-3513"
        ),
    }


def require_moving_frame(config: CM1Config) -> "TranslatingFrame":
    """Bind ``closures`` to CM1's ``base.F:2661-2674`` frame, or refuse.

    ``closures.moving_frame`` is CM1's ``imove``.  With it off the frame is
    exactly fixed, because ``config.py`` already refuses nonzero speeds
    alongside it (``CM1-CFG-MOVING-FRAME-INACTIVE``), which is the same
    coupling ``param.F:701-702`` expresses by silently zeroing them.
    """

    from .gm1_idealized_residency import FIXED_FRAME, TranslatingFrame

    if not config.closures.moving_frame:
        return FIXED_FRAME
    try:
        return TranslatingFrame(
            u_move_m_s=config.closures.u_move_m_s,
            v_move_m_s=config.closures.v_move_m_s,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-MOVING-FRAME",
            gate=CARTESIAN_GATE,
            reason=f"the base.F:2663-2669 translating frame refuses this velocity: {error}",
            remedy="declare closures.u_move_m_s and closures.v_move_m_s as finite storm-motion speeds in m/s",
            option="closures.u_move_m_s",
            value=[config.closures.u_move_m_s, config.closures.v_move_m_s],
        )


def _component_cm1_eqtset(component: str) -> int:
    """CM1's eqtset for the component's ptype after param.F:936-950."""

    from .moisture_species import moisture_species_profile

    try:
        return int(moisture_species_profile(component).cm1_eqtset)
    except Exception:  # noqa: BLE001 - an unknown component is refused elsewhere, by name
        return 2


def require_initializer_selection(config: CM1Config) -> tuple[InitializerSelection, bool]:
    """Bind the declared CM1 base.F selectors to the initializer's contract.

    A configuration that declares no [initialization] table binds the
    single-sounding lane -- base.F isnd=7 with the iwnd=0 that param.F:836-852
    forces for it -- which is the only lane this door has ever driven.  The
    bound pair is named by value in the receipt either way.
    """

    from .gm1_idealized_residency import InitializerSelection, SINGLE_SOUNDING_SELECTION

    declared = config.initialization
    if declared is None:
        return SINGLE_SOUNDING_SELECTION, False
    try:
        return (
            InitializerSelection(
                isnd=declared.isnd,
                iwnd=declared.iwnd,
                iinit=declared.iinit,
                irst=declared.irst,
                imoist=declared.imoist,
                testcase=declared.testcase,
                ibalance=declared.ibalance,
                irandp=declared.irandp,
                random_seed=declared.random_seed,
                random_generator=1 if declared.random_generator == "cm1" else 0,
                # init3d.F:697-1256 reads fcor for the vortex's gradient wind,
                # which param.F:1985 zeroes when icor = 0, and makes images
                # across repeating sides.
                fcor_s_inv=float(config.dynamics.fcor_s_inv) if int(config.dynamics.coriolis) == 1 else 0.0,
                periodic_x=int(config.boundaries.west == 1 and config.boundaries.east == 1),
                periodic_y=int(config.boundaries.south == 1 and config.boundaries.north == 1),
                ric_m=declared.bubble_center_x_m,
                # init3d.F:548-550: the iinit = 3 line's first bubble sits at
                # the literal rjc = 3000 m (bubble_first_center_y_m), not at
                # centery.  A declared bubble_center_y_m still moves it, as it
                # did before that literal was carried (the Settings screen sets
                # the bubble's x and y together); config.py refuses both at once.
                rjc_m=(declared.bubble_first_center_y_m
                       if int(declared.iinit) == 3 and "bubble_center_y_m" not in declared.declared
                       else declared.bubble_center_y_m),
                zc_m=declared.zc_m,
                # init3d.F:1355-1356: the iinit = 11 envelope's horizontal
                # scale is its own word (half_width_m, 5 km), not the bubble's.
                bhrad_m=declared.half_width_m if int(declared.iinit) == 11 else declared.bhrad_m,
                bvrad_m=declared.bvrad_m,
                bptpert_k=declared.bptpert_k,
                maintain_rh=declared.maintain_rh,
                nbub=declared.nbub,
                bubble_spacing_y_m=declared.bubble_spacing_y_m,
                zdep_m=declared.zdep_m,
                random_line_amplitude_k=declared.random_line_amplitude_k,
                hurr_vg_m_per_s=declared.hurr_vg_m_per_s,
                hurr_rotate_deg=declared.hurr_rotate_deg,
                thec_mb_k=declared.thec_mb_k,
                qt_mb=declared.qt_mb,
                # base.F:1204/1849 read the namelist tsk0: the one the
                # [initialization] table declares, else the [surface] table's,
                # else run/namelist.input's.
                tsk0_k=(declared.tsk0_k if declared.tsk0_k is not None
                        else float(config.surface.tsk0_k) if config.surface is not None
                        else 299.28),
                # param.F:5178-5205: poiss mirrors an axis with an open (2) or
                # free-slip (3) side; ibalance = 2 reads it.
                imirror=int(config.boundaries.west in (2, 3) or config.boundaries.east in (2, 3)),
                jmirror=int(config.boundaries.south in (2, 3) or config.boundaries.north in (2, 3)),
                # param.F:936-950: the eqtset CM1 runs this component's ptype
                # with, which base.F's isnd = 10 lapse rate reads.
                eqtset=_component_cm1_eqtset(config.physics.component),
            ),
            True,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-INITIALIZER-SELECTION",
            gate=CARTESIAN_GATE,
            reason=f"the idealized initializer refuses this base-state selection: {error}",
            remedy="declare an [initialization] table the transcribed base.F / init3d.F initializer implements",
            option="initialization",
            value={"isnd": declared.isnd, "iwnd": declared.iwnd, "iinit": declared.iinit},
        )


def require_bubble_center(
    config: CM1Config, grid: CartesianGridGeometry
) -> tuple[float, float]:
    """``init3d.F:456-457``'s ``ric``/``rjc`` for this run, by value.

    An ``[initialization]`` table that declares neither word binds
    ``param.F:7408``'s ``centerx``/``centery``, which is what ``init3d.F``
    itself assigns, so a document written before the bubble could be moved
    initializes exactly the perturbation it always did.  ``config.py`` has
    already refused a half-declared pair and a disc that does not fit inside
    the grid, so this only has to say which two numbers the initializer gets.

    The pair is in the ``iorigin`` frame (``param.F:7212-7213``), exactly as
    :func:`bubble_center_m` -- the authority the initializer is handed --
    gives it: metres from the south-west corner for iorigin=1, from the
    domain centre for iorigin=2.  ``grid`` must be this configuration's own.
    """

    if (grid.nx, grid.ny) != (config.grid.nx, config.grid.ny):
        raise ValueError("require_bubble_center needs the configuration's own grid")
    return bubble_center_m(config)


def bubble_center_record(
    config: CM1Config, center_xy_m: tuple[float, float]
) -> dict[str, object]:
    """Where the warm bubble was put, and whether the document said so."""

    declared = (
        config.initialization is not None
        and config.initialization.bubble_center_declared
    )
    record: dict[str, object] = {
        "authority": "CM1 r21.1 init3d.F:456-457 ric/rjc; param.F:7408 centerx/centery",
        "declared": declared,
        "bubble_center_x_m": center_xy_m[0],
        "bubble_center_y_m": center_xy_m[1],
        "horizontal_radius_m": BUBBLE_HORIZONTAL_RADIUS_M,
        "convention": (
            "metres in the iorigin frame (param.F:7212-7213): from the "
            "south-west corner for iorigin=1, from the domain centre for "
            "iorigin=2; an undeclared pair is param.F:7408's domain centre"
        ),
    }
    # A forcing arm starts the storm where its push reaches: whether its
    # centre is in the domain and how many grid points it touches, with the
    # plain warning when it touches none or is centred outside.  Only a run
    # with a forcing arm carries the key, so every other receipt is unchanged.
    reach = initiation_forcing_reach(config)
    if reach is not None:
        record["forcing_reach"] = reach
    return record


def initiation_forcing_reach(config: CM1Config) -> dict[str, object] | None:
    """``initiation_forcing.forcing_reach`` for this document, or None without a forcing arm.

    Host arithmetic on the resolved words and the run's own vertical
    coordinate; a coordinate that cannot be built is refused elsewhere, and
    here it only means the reach is not reported.
    """

    from .initiation_forcing import forcing_reach, resolve_initiation_forcing

    grid = config.grid
    forcing = resolve_initiation_forcing(
        config.initialization, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m
    )
    if forcing is None:
        return None
    try:
        levels = require_vertical_coordinate(config)
    except CM1Refusal:
        return None
    return forcing_reach(
        forcing, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m,
        zh=levels.mass_heights_m, zf=levels.face_heights_m,
    )


#: The ``isnd`` lanes the run-geometry admission refuses on.  Both read a
#: user's ``input_sounding`` (base.F:495): isnd=7 takes its winds from the
#: file, isnd=17 takes them from ``iwnd`` (base.F:2263, param.F:849-851).  The
#: analytic Weisman-Klemp lane (isnd=5) is evaluated the same way and reported,
#: not refused: it is CM1's own validation profile, and this repository's
#: sub-hour bubble tests on 32-60 km grids -- where no storm organises and the
#: organising-storm model has nothing to say -- run on it.  The estimator
#: (``gpuwm-cm1 estimate``) shows the analytic lane's result before a run.
RUN_GEOMETRY_ADMITTED_ISND = frozenset({7, 17})
#: base.F's soundings whose base state carries cloud water or ice
#: (qc0 = qt_mb - qv0, base.F:347; isnd = 10's qcval, base.F:917-930;
#: isnd = 11's ql2 is zero for its pseudoadiabatic ttype = 2, but the arm
#: writes qc0 and CM1 copies it, so it is carried).  isnd = 9 zeroes qc0.
BASE_CONDENSATE_ISND = frozenset({4, 10, 11})
#: The soundings a run (not only run.iconly = 1) may start from: each has a time integration that matched CM1
#: r21.1's like for like (records/CM1-INITIALIZATION-VALIDATION.json).  5, 7 and 17 are the ones every earlier
#: release ran; 9 matched CM1 over the first 15 minutes, both models on Morrison at eqtset = 2.  Every other
#: analytic sounding is built word for word as CM1 builds it at model time zero and is shown with run.iconly = 1:
#: 4, 10 and 11 were compared and did not match, and 1, 8, 12-14 and 18-23 have not been compared.
INTEGRATION_CHECKED_ISND = frozenset({5, 7, 9, 17})
#: The analytic soundings whose 15-minute integration was compared with CM1's like for like and did not match it.
INTEGRATION_FAILED_ISND = frozenset({4, 10, 11})


def sounding_integration_checked(isnd: int) -> bool:
    """Whether a run from this ``isnd`` has an integration that matched CM1's (else it is shown only)."""

    return int(isnd) in INTEGRATION_CHECKED_ISND


def run_geometry_hodograph(
    config: CM1Config,
    selection: "InitializerSelection",
    sounding_bytes: bytes,
) -> tuple["Hodograph", dict[str, object]]:
    """The base-state wind this run integrates, and where it comes from.

    ``base.F:2263`` fills the wind from ``iwnd`` for every ``isnd`` but 7, so
    the file's winds are the profile only on the isnd=7 lane; isnd=17 reads
    its thermodynamics from the same file and its winds from ``iwnd``
    (param.F:849-851), and isnd=5 reads no file at all.  The analytic branch
    is evaluated at the run's own scalar levels, as base.F evaluates it.
    Raises ``TypeError``/``ValueError`` when no usable profile results.
    """

    from .run_geometry import hodograph_from_analytic, hodograph_from_input_sounding

    if selection.isnd == 7:
        return hodograph_from_input_sounding(sounding_bytes), {
            "wind_source": "input_sounding (base.F:495-607, isnd=7)",
        }
    levels = require_vertical_coordinate(config)
    nz = config.grid.nz
    heights = [float(value) for value in levels.zh[1 : nz + 1]]
    hodograph = hodograph_from_analytic(
        selection.iwnd,
        heights,
        testcase=selection.testcase,
        hurr_vg_m_s=selection.hurr_vg_m_per_s,
        hurr_rotate_deg=selection.hurr_rotate_deg,
    )
    return hodograph, {
        "wind_source": (
            f"base.F:2263-2650 iwnd={selection.iwnd} evaluated at the run's {nz} "
            f"scalar levels (isnd={selection.isnd}"
            + (", thermodynamics from input_sounding" if selection.isnd == 17 else "")
            + ")"
        ),
    }


def open_lateral_sides(config: CM1Config) -> tuple[bool, bool, bool, bool]:
    """(west, east, south, north): which sides a storm can be lost at.

    README.namelist wbc/ebc/sbc/nbc: 1 is periodic, so what leaves re-enters;
    2 (open radiative) and 3 (rigid wall) are boundaries a storm meets.
    """

    b = config.boundaries
    return tuple(int(side) != 1 for side in (b.west, b.east, b.south, b.north))  # type: ignore[return-value]


def require_run_geometry(
    config: CM1Config,
    grid: CartesianGridGeometry,
    *,
    selection: "InitializerSelection",
    moving_frame: "TranslatingFrame",
    center_xy_m: tuple[float, float],
    sounding_label: str | None,
    sounding_bytes: bytes,
    adaptive_frame: AdaptiveFrameBinding | None = None,
) -> dict[str, object]:
    """Refuse a run whose storm leaves the domain before the clock does.

    This is the admission that the two high-resolution runs of 2026-09-13
    needed and did not have.  Both declared a translating frame at their
    sounding's mature right-mover motion and put CM1's ``iinit=1`` bubble at
    the domain centre of a 44.8 km domain.  An unorganised bubble does not
    move at a mature supercell's motion -- it advects with the wind through
    its own depth -- so the perturbation crossed the domain at the difference
    and reached the radiative lateral boundary at minute 16, before it could
    organise.  Every later domain maximum was a boundary artifact, and five
    GPU-hours produced no storm.

    ``run_geometry`` computes the whole track on the host from the profile
    this run will actually integrate: the file's winds on isnd=7, and
    base.F's closed-form ``iwnd`` branch at the run's own levels on isnd=5
    and isnd=17.  Nothing here opens a device.  The lanes in
    :data:`RUN_GEOMETRY_ADMITTED_ISND` are refused on the result; the analytic
    isnd=5 lane is evaluated and reported (``tracked-member-leaves-reported``
    when the track loses its clearance), for the reason that constant gives.
    """

    from .run_geometry import (
        MEMBER_LEFT,
        MEMBER_RIGHT,
        evaluate_run_geometry,
    )

    # README.namelist:1789-1794.  With the adaptive controller selected there
    # is no constant to aim: azimavg.F:4431-4435 re-aims the frame at the
    # storm every cadence.  What is still a fixed-frame question, and is still
    # refused below, is the spin-up: param.F:706-707 starts the run at
    # umove = vmove = 0 and cm1.F:3642-3651 makes the first correction at
    # model time adapt_move_frq, so the storm has to survive that first
    # interval on a stationary grid or the controller never gets a storm to
    # follow.
    window_s = config.run.duration_s
    if adaptive_frame is not None:
        # A run that ends before the first correction is a fixed-frame run for
        # its whole length, and there is nothing beyond its own end to admit.
        window_s = min(adaptive_frame.adapt_move_frq_s, config.run.duration_s)
    if getattr(config, "chain", None) is not None:
        # A domain driven by a WRF or ArWen history (records/GM1-ARWEN-CHAIN.md)
        # starts from that history's state, not from an initializer's bubble,
        # and its lateral rows are specified from the driving run every step:
        # what reaches the boundary is the driving model's air, not a
        # radiative-boundary artifact.  There is no initializer track to aim.
        return {
            "status": "not-evaluated-chain",
            "reason": (
                "[chain]: the initial state and the lateral boundaries come from the "
                "driving WRF or ArWen history, so no initializer storm is tracked"
            ),
            "moving_frame": moving_frame_record(moving_frame),
        }
    run = getattr(config, "run", None)
    if run is not None and getattr(run, "iconly", None) == 1:
        return {
            "status": "not-evaluated-iconly",
            "reason": (
                "run.iconly = 1: CM1 writes the initial condition and stops (cm1.F:4341), "
                "so no storm is integrated and none can leave the domain"
            ),
            "moving_frame": moving_frame_record(moving_frame),
        }
    if int(getattr(selection, "iinit", 1)) == 0 and int(getattr(selection, "irandp", 0)) == 0:
        # README.namelist iinit = 0 with irandp = 0: init3d.F adds no initial
        # perturbation at all, so there is no storm whose track could leave the
        # domain.  (A downscaled run's namelist is one: its state comes from the
        # driving history, records/GM1-ARWEN-CHAIN.md.)
        return {
            "status": "not-evaluated-no-initial-perturbation",
            "reason": "iinit = 0 and irandp = 0: CM1 adds no initial perturbation, so no storm is tracked",
            "moving_frame": moving_frame_record(moving_frame),
        }
    sides = open_lateral_sides(config)
    if not any(sides):
        # README.namelist wbc/ebc/sbc/nbc = 1: a doubly periodic domain has no
        # lateral boundary for a storm to reach, so there is no exit time to
        # refuse on (the no-storm balance column is the case that needs it).
        return {
            "status": "not-evaluated-doubly-periodic",
            "reason": (
                "wbc = ebc = sbc = nbc = 1: every lateral boundary is periodic, "
                "so a storm cannot leave the domain"
            ),
            "bubble_center": bubble_center_record(config, center_xy_m),
            "moving_frame": moving_frame_record(moving_frame),
        }
    admitted = selection.isnd in RUN_GEOMETRY_ADMITTED_ISND
    # ``center_xy_m`` is in the iorigin frame; the track is formed on the
    # corner-origin rectangle [0, nx*dx) x [0, ny*dy), so shift by the frame's
    # origin (0 for iorigin=1, minus half the domain for iorigin=2).
    origin_x, origin_y = horizontal_origin_m(config)
    corner_center = (center_xy_m[0] - origin_x, center_xy_m[1] - origin_y)
    try:
        hodograph, wind_source = run_geometry_hodograph(config, selection, sounding_bytes)
        geometry = evaluate_run_geometry(
            hodograph=hodograph,
            nx=grid.nx,
            ny=grid.ny,
            dx_m=grid.dx_m,
            dy_m=grid.dy_m,
            duration_s=window_s,
            frame_m_s=(moving_frame.u_move_m_s, moving_frame.v_move_m_s),
            bubble_center_xy_m=corner_center,
            open_sides=sides,
        )
    except (TypeError, ValueError) as error:
        if not admitted:
            return {
                "status": "not-evaluated-no-usable-profile",
                "reason": (
                    f"base.F isnd={selection.isnd} / iwnd={selection.iwnd} gives no "
                    f"profile the storm-track model can use: {error}"
                ),
                "bubble_center": bubble_center_record(config, center_xy_m),
                "moving_frame": moving_frame_record(moving_frame),
            }
        refuse(
            code="CM1-CARTESIAN-RUN-GEOMETRY-UNREADABLE",
            gate=CARTESIAN_GATE,
            reason=f"the declared profile carries no usable hodograph: {error}",
            remedy="supply a CM1 r21.1 input_sounding whose records are height, theta, qv, u, v and which reaches 6 km",
            option="sounding",
            value=sounding_label,
        )
    tracked = geometry.members[MEMBER_RIGHT]
    if not tracked.stays_inside and adaptive_frame is not None and admitted:
        exit_s = tracked.exit_time_s or 0.0
        refuse(
            code="CM1-CARTESIAN-ADAPT-MOVE-SPINUP",
            gate=CARTESIAN_GATE,
            reason=(
                f"the tracked storm loses its {geometry.admission_clearance_m:.0f} m "
                f"clearance from a lateral boundary at model time {exit_s:.0f} s "
                f"({exit_s / 60.0:.1f} min), before the adaptive controller's "
                f"first correction at "
                f"{adaptive_frame.adapt_move_frq_s:.0f} s. param.F:706-707 starts "
                "every do_adapt_move run at umove = vmove = 0 and "
                "cm1.F:3642-3651 makes the first update one adapt_move_frq "
                "later, so on this grid the bubble reaches the radiative "
                "boundary while the domain is still stationary -- the "
                "controller would then steer the frame by the pressure minimum "
                "of a boundary artefact instead of a storm"
            ),
            remedy=(
                "shorten closures.adapt_move_frq so the first correction comes "
                "before the storm reaches the boundary, widen the grid, or "
                "declare initialization.bubble_center_x_m / "
                "initialization.bubble_center_y_m upwind of the domain centre"
            ),
            option="closures.adapt_move_frq",
            value=adaptive_frame.adapt_move_frq_s,
        )
    if not tracked.stays_inside and admitted:
        exit_s = tracked.exit_time_s or 0.0
        # One remedy, chosen so that it is itself admissible: a remedy the
        # same door would refuse leaves the operator with nothing to do.
        plan = geometry.recommendation()
        # Back into the iorigin frame the document declares ric/rjc in.
        recommended = (
            plan["bubble_center_xy_m"][0] + origin_x,
            plan["bubble_center_xy_m"][1] + origin_y,
        )
        frame = plan["frame_m_s"]
        extent = plan["required_domain_extent_m"]
        if plan["action"] == "move-the-bubble":
            remedy = (
                f"declare initialization.bubble_center_x_m = {recommended[0]:.0f} and "
                f"initialization.bubble_center_y_m = {recommended[1]:.0f}, which "
                "centres the tracked member's whole excursion in this grid"
            )
        elif plan["action"] == "change-the-frame":
            remedy = (
                f"declare closures.u_move_m_s = {frame[0]:.2f}, closures.v_move_m_s = "
                f"{frame[1]:.2f}, initialization.bubble_center_x_m = "
                f"{recommended[0]:.0f} and initialization.bubble_center_y_m = "
                f"{recommended[1]:.0f}; that frame is the one that minimizes the "
                "tracked member's excursion over this run's own duration"
            )
        else:
            remedy = (
                f"widen the grid to at least {extent[0]:.0f} x {extent[1]:.0f} m and "
                f"declare closures.u_move_m_s = {frame[0]:.2f}, closures.v_move_m_s = "
                f"{frame[1]:.2f} with the bubble at "
                f"({recommended[0]:.0f}, {recommended[1]:.0f}) m; or shorten "
                "run.duration_s so the run ends while the storm is still inside"
            )
        refuse(
            code="CM1-CARTESIAN-RUN-GEOMETRY",
            gate=CARTESIAN_GATE,
            reason=(
                f"the tracked storm loses its {geometry.admission_clearance_m:.0f} m "
                f"clearance from a lateral boundary at model time {exit_s:.0f} s "
                f"({exit_s / 60.0:.1f} min) of a {config.run.duration_s:.0f} s run, "
                f"on a {geometry.domain_extent_m[0]:.0f} x "
                f"{geometry.domain_extent_m[1]:.0f} m domain. The warm bubble at "
                f"({center_xy_m[0]:.0f}, {center_xy_m[1]:.0f}) m advects with its own "
                f"steering wind ({geometry.steering_m_s[0]:.2f}, "
                f"{geometry.steering_m_s[1]:.2f}) m/s while the declared frame travels "
                f"at ({moving_frame.u_move_m_s:.2f}, {moving_frame.v_move_m_s:.2f}) m/s, "
                "so from that minute on every domain maximum this run publishes is a "
                "radiative-boundary artifact and not a storm"
            ),
            remedy=remedy,
            option="grid",
            value={
                "nx": grid.nx,
                "ny": grid.ny,
                "dx_m": grid.dx_m,
                "dy_m": grid.dy_m,
                "duration_s": config.run.duration_s,
            },
        )
    if tracked.stays_inside:
        status = (
            "tracked-member-stays-inside"
            if adaptive_frame is None
            else "adaptive-frame-spin-up-clear"
        )
    else:
        # Only the analytic lane reaches here with a lost clearance.
        status = "tracked-member-leaves-reported"
    record: dict[str, object] = {"status": status}
    if not tracked.stays_inside:
        record["reported_not_admitted"] = (
            f"the tracked storm loses its {geometry.admission_clearance_m:.0f} m "
            f"clearance at model time {tracked.exit_time_s or 0.0:.0f} s. The "
            f"analytic isnd={selection.isnd} lane is reported here, not refused: "
            "it is CM1's own validation profile and carries sub-hour bubble tests "
            "on small grids where no storm organises"
        )
    if adaptive_frame is not None:
        record["evaluated_window_s"] = window_s
        record["adaptive_frame_note"] = (
            "the track above is the stationary grid this run starts on, "
            f"evaluated over the {window_s:.0f} s before azimavg.F:4431-4435's "
            "first correction. After that the frame velocity is a model "
            "variable and no host-side constant-frame track describes it; "
            "where the domain actually is at each published model time is "
            f"carried per frame as {DOMAIN_LOCATION_X_NAME}/"
            f"{DOMAIN_LOCATION_Y_NAME}"
        )
    record.update(geometry.record())
    record.update(wind_source)
    record["track_coordinates"] = (
        "metres from the domain's south-west corner; the iorigin frame's origin "
        f"is at ({origin_x:.0f}, {origin_y:.0f}) m in them"
    )
    record["open_sides"] = dict(zip(("west", "east", "south", "north"), sides))
    record["bubble_center"] = bubble_center_record(config, center_xy_m)
    # The left mover is reported, never refused on.  A domain that holds both
    # members of a split for two hours is about 70 km wider than one that
    # holds the right mover, which at these resolutions is past the history
    # envelope's payload cap; saying so is better than pretending otherwise.
    left = geometry.members[MEMBER_LEFT]
    record["left_mover_disposition"] = (
        "inside for the whole run"
        if left.stays_inside
        else f"leaves the domain at model time {left.exit_time_s:.0f} s; the split's "
        "left member is reported, not admitted on"
    )
    return record



def _plan_sounding(
    selection: "InitializerSelection", sounding: Path | None
) -> dict[str, object]:
    """Resolve the profile for a plan, or say why the geometry is unknown.

    Only isnd=7 takes its winds from the file (base.F:2263); isnd=17 reads the
    file for thermodynamics alone, so its track needs no profile here.
    """

    if selection.isnd != 7:
        return {"sounding_label": None, "sounding_bytes": b""}
    try:
        label, payload = resolve_sounding(sounding)
    except CM1Refusal as refusal:
        refuse(
            code="CM1-CARTESIAN-SOUNDING-UNREADABLE",
            gate=CARTESIAN_GATE,
            reason=(
                "no input_sounding could be resolved for this plan, so the storm "
                f"track cannot be formed: {refusal.detail.reason}"
            ),
            remedy="pass --sounding with the profile this run will read",
            option="sounding",
            value=None if sounding is None else str(sounding),
        )
    return {"sounding_label": label, "sounding_bytes": payload}


def cartesian_history_plan(runtime_config: RuntimeConfig) -> dict[str, object]:
    """Price one published history frame against the envelope's own cap.

    ``cartesian_history.cartesian_history_profile`` is the authority for both
    numbers and refuses a grid the CM1FHS1 contract cannot carry; this only
    reports the two the operator has to plan disk and cadence around.
    """

    from .cartesian_history import (
        MAX_FORECAST_PAYLOAD_BYTES,
        DuplicateHistoryCarrierError,
        cartesian_history_profile,
    )

    from .cartesian_tornado_diagnostics import tornado_carrier_table

    try:
        # Price the frame the runtime will actually publish: the rotation and
        # swath carriers are part of it (cuda_cartesian_runtime passes the same
        # table), so a plan that left them out would admit a frame the run
        # then refuses.  The ground-fixed swath twins are sized by the run's
        # own track and are priced when the runtime opens them.
        profile = cartesian_history_profile(
            runtime_config, tornado_carriers=tornado_carrier_table(runtime_config)
        )
    except DuplicateHistoryCarrierError as error:
        # A repeated carrier name is a carrier-table defect: no grid size
        # makes it publishable, so the remedy must not point at the grid.
        refuse(
            code="CM1-CARTESIAN-HISTORY-CARRIER-NAME",
            gate=CARTESIAN_GATE,
            reason=f"this configuration has no publishable history frame: {error}",
            remedy=(
                "deselect one of the switches that publishes "
                f"{', '.join(error.names)} (each name may come from one "
                "[output], [physics] or [diagnostics] switch only), and report "
                "the configuration: two producers of one CM1 output name is a "
                "gpuwm-cm1 carrier-table defect"
            ),
            option="history.carriers",
            value=list(error.names),
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-HISTORY-PAYLOAD",
            gate=CARTESIAN_GATE,
            reason=f"this grid has no publishable history frame: {error}",
            remedy="reduce grid.nx, grid.ny or grid.nz until one frame fits the CM1FHS1 payload cap",
            option="grid",
            value={
                "nx": runtime_config.grid.nx,
                "ny": runtime_config.grid.ny,
                "nz": runtime_config.grid.nz,
            },
        )
    return {
        "payload_bytes": profile.payload_nbytes,
        "maximum_payload_bytes": MAX_FORECAST_PAYLOAD_BYTES,
        "staging_allocation_bytes": profile.staging_bytes,
        "carrier_names": [carrier.name for carrier in profile.carriers],
    }


def horizontal_origin_m(config: CM1Config) -> tuple[float, float]:
    """``param.F:7212-7213``: the domain corner (iorigin=1) or centre (2).

    CM1 multiplies its horizontal coordinate array by ``mult``, 0.0 for
    iorigin=1 and 0.5 for iorigin=2, so the centred frame runs from
    ``-nx*dx/2`` to ``+nx*dx/2``.  Every absolute-coordinate initialization
    parameter -- ``ric``, ``rjc``, the Skamarock-Klemp wave centre -- is read
    in this frame, and so is every coordinate a published frame reports.
    """

    iorigin = 1 if config.initialization is None else config.initialization.iorigin
    if iorigin == 1:
        return (0.0, 0.0)
    grid = config.grid
    return (
        -0.5 * grid.nx * grid.dx_m,
        -0.5 * grid.ny * grid.dy_m,
    )


def bubble_center_m(config: CM1Config) -> tuple[float, float]:
    """``init3d.F:456-457``: ``ric``/``rjc``, defaulting to the domain centre.

    CM1 writes ``ric = centerx`` and ``rjc = centery``.  A declared value is
    an absolute coordinate in the ``iorigin`` frame, so it is taken as written
    rather than offset again.
    """

    origin_x, origin_y = horizontal_origin_m(config)
    grid = config.grid
    centre = (
        origin_x + 0.5 * grid.nx * grid.dx_m,
        origin_y + 0.5 * grid.ny * grid.dy_m,
    )
    declared = config.initialization
    if declared is None:
        return centre
    # The forcing arms start the storm where their forcing is centred
    # (init3d.F:1372-1376 xc/yc_wnudge, :1313-1314 xcent/ycent, :1339
    # xc_uforce with the line uniform in y), so the storm-track check and the
    # Domain screen place the storm there.  They build no tha, so handing the
    # same pair to the initializer as ric/rjc changes no word.
    from .initiation_forcing import resolve_initiation_forcing

    forcing = resolve_initiation_forcing(
        declared, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m
    )
    if forcing is not None:
        if forcing.iinit == 12:
            return (forcing["xc_wnudge"], forcing["yc_wnudge"])
        if forcing.iinit == 9:
            return (forcing["xcent"], forcing["ycent"])
        return (forcing["xc_uforce"], centre[1])
    return (
        centre[0] if declared.bubble_center_x_m is None else float(declared.bubble_center_x_m),
        centre[1] if declared.bubble_center_y_m is None else float(declared.bubble_center_y_m),
    )


def require_history_cadence(config: CM1Config) -> HistoryCadence:
    """Admit the requested integration length and frame interval, CUDA-cold."""

    try:
        return history_cadence(
            dt_s=config.run.dt_s,
            duration_s=config.run.duration_s,
            interval_s=config.run.history_interval_s,
            adaptive=config.dynamics.adaptive_timestep,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-HISTORY-CADENCE",
            gate=CARTESIAN_GATE,
            reason=f"the requested integration cannot be placed on the large-step clock: {error}",
            remedy="set run.duration_s and run.history_interval_s to whole numbers of run.dt_s steps, with the interval dividing the duration exactly",
            option="run.history_interval_s",
            value=config.run.history_interval_s,
        )


def resume_selection(config: CM1Config, restart_from: Path | None) -> tuple[Path, dict] | None:
    """Locate the restart this configuration asks to continue from.

    ``cm1.F:2660`` reads a restart only under ``irst=1`` and
    README.namelist:229-231 makes ``rstnum`` the number of the file to read.
    Everything this needs is on disk, so the whole selection -- including the
    schedule the resumed run will follow -- is admitted here, before a CUDA
    context exists.
    """

    from .cartesian_restart import (
        RESTART_DIRECTORY_FORMAT,
        require_restart_receipt,
    )

    if not config.run.restarting:
        if restart_from is not None:
            refuse(
                code="CM1-CARTESIAN-RESTART-SELECTION",
                gate=CARTESIAN_GATE,
                reason="a restart directory was supplied but run.irst is not 1, so cm1.F:2660 would never read it and the run would integrate from the initial condition while its receipt named a restart",
                remedy="declare run.irst = 1 and run.rstnum = N, or drop --restart-from",
                option="restart-from",
                value=str(restart_from),
            )
        return None
    if restart_from is None:
        refuse(
            code="CM1-CARTESIAN-RESTART-SELECTION",
            gate=CARTESIAN_GATE,
            reason="run.irst = 1 asks to continue a previous run and nothing names where that run's restarts are; without the directory the resume would silently become a fresh start",
            remedy="pass --restart-from with the previous run's --restart-dir",
            option="restart-from",
            value=None,
        )
    root = Path(restart_from).expanduser()
    if not root.is_dir():
        refuse(
            code="CM1-CARTESIAN-RESTART-SELECTION",
            gate=CARTESIAN_GATE,
            reason="the restart directory does not exist, so there is no carried state to continue from",
            remedy="pass --restart-from with the directory the previous run wrote its restarts into",
            option="restart-from",
            value=str(root),
        )
    name = RESTART_DIRECTORY_FORMAT.format(number=config.run.rstnum)
    selected = root / name
    if not selected.is_dir():
        present = sorted(
            entry.name for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
        )
        refuse(
            code="CM1-CARTESIAN-RESTART-SELECTION",
            gate=CARTESIAN_GATE,
            reason=f"{root} holds no {name}; it holds {present or 'no restarts at all'}. Resuming from a restart that is not there would start the run over from its initial condition after the operator believed it had been continued",
            remedy="set run.rstnum to one of the numbers present, or point --restart-from at the directory that holds the intended restart",
            option="run.rstnum",
            value=config.run.rstnum,
        )
    try:
        receipt, _ = require_restart_receipt(selected)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        refuse(
            code="CM1-CARTESIAN-RESTART-RECEIPT",
            gate=CARTESIAN_GATE,
            reason=f"the selected restart does not carry an admissible publication receipt: {error}",
            remedy="select a restart written by this lane; a directory without its receipt cannot name the contract its envelope must be read back under",
            option="run.rstnum",
            value=config.run.rstnum,
        )
    # A nested run's restart names its child's own restart beside it
    # (cm1core.nest_restart).  When the document declares a [nest], the child's
    # receipt is admitted here, before a device exists: a declared child whose
    # receipt is missing, unreadable, of another profile or step, not the one
    # written with this parent, or without its coupling state would otherwise
    # be found only after the parent had been restored on the device.  The
    # envelope's words are decoded later, by Rust; a torn one stops the run
    # there.  A document with no [nest] resumes the parent alone from its own
    # restart, whatever the child directory holds or fails to hold (the
    # ``nest_child_restart`` row of the resume record says so): a child that is
    # gone must not bar the parent from continuing.
    from .nest_restart import nest_child_receipt, nest_resume_differences

    child = None
    if config.nest is not None:
        try:
            child = nest_child_receipt(receipt, selected)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            refuse(
                code="CM1-CARTESIAN-RESTART-RECEIPT",
                gate=CARTESIAN_GATE,
                reason=(
                    "the selected restart was written by a nested run and declares its "
                    f"nest child's restart, which is not admissible: {error}. Resuming "
                    "without it would rebuild the child from the parent, and the resumed "
                    "trajectory would not be the interrupted one"
                ),
                remedy=(
                    "select a restart whose nest-child directory is complete (a nested "
                    "restart is published parent and child together), or drop the "
                    "[nest] table to continue the parent alone from its own restart"
                ),
                option="run.rstnum",
                value=config.run.rstnum,
            )
    if child is not None:
        try:
            refused, _admitted = nest_resume_differences(child[1], config)
        except (ValueError, TypeError, KeyError) as error:
            refused = {"nest_coupling": f"unreadable: {error}"}
        if refused:
            refuse(
                code="CM1-CARTESIAN-NEST-RESTART-MISMATCH",
                gate=CARTESIAN_GATE,
                reason=(
                    "the selected restart's nest child was written with coupling the "
                    f"document does not declare: {refused} (restart, document). The "
                    "child's lattice and clock are its refinement ratios and spans, so "
                    "its words would be read onto another grid; parcels the restart "
                    "carries would end without a termination, or parcels it never held "
                    "would be invented; and a restored placement the document's zones "
                    "cannot hold would force the child from the parent's own boundary"
                ),
                remedy=(
                    "resume with the [nest] ratios and spans and the "
                    "[lagrangian.nest_child] nparcels the restart was written with; "
                    "spec_zone, relax_zone, feedback and the follow settings may change "
                    "and are recorded in the receipt"
                ),
                option="nest",
                value=refused,
            )
    if (
        config.ensemble is not None and config.nest is not None
        and receipt.get("nest_child") is not None
    ):
        refuse(
            code="CM1-CARTESIAN-NEST-RESTART-ENSEMBLE",
            gate=CARTESIAN_GATE,
            reason=(
                "the ensemble branch perturbs the parent's potential temperature "
                "(init3d.F:1446-1483 on the parent grid), and this restart carries the "
                "nest child's own state, which is restored rather than rebuilt from the "
                "perturbed parent; the member's child would start unperturbed inside a "
                "perturbed parent"
            ),
            remedy=(
                "branch the member from a single-domain restart with the [nest] table "
                "declared, which rebuilds the child from the perturbed parent, or drop "
                "the [ensemble] table"
            ),
            option="ensemble",
            value={
                "member": config.ensemble.member,
                "seed": config.ensemble.seed,
                "run.rstnum": config.run.rstnum,
            },
        )
    return selected, dict(receipt)


def require_run_schedule(
    config: CM1Config, *, restart: dict | None = None, write_restarts: bool = False
) -> RunSchedule:
    """Admit this run's complete publication schedule, CUDA-cold.

    ``cm1.F:3180-3190`` compares the model clock against ``taptim`` and
    ``rsttim`` every step.  Both cadences are whole large steps here, so the
    entire schedule -- including the one a resumed run will follow from the
    step its restart stopped at -- is computed and refused on the host, before
    a device is opened or an output path is created.
    """

    start_step = 0 if restart is None else int(restart["step"])
    first_history_index = 1 if restart is None else int(restart["next_history_index"])
    first_restart_number = (
        FIRST_RESTART_NUMBER if restart is None else int(restart["restart_number"]) + 1
    )
    try:
        return run_schedule(
            dt_s=config.run.dt_s,
            duration_s=config.run.duration_s,
            history_interval_s=config.run.history_interval_s,
            restart_interval_s=config.run.restart_interval_s if write_restarts else None,
            start_step=start_step,
            reset_frqtim=config.run.reset_frqtim,
            first_history_index=first_history_index,
            first_restart_number=first_restart_number,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-HISTORY-CADENCE",
            gate=CARTESIAN_GATE,
            reason=f"the requested integration cannot be placed on the large-step clock: {error}",
            remedy="set run.duration_s, run.history_interval_s and run.restart_interval_s to whole numbers of run.dt_s steps that each divide the steps remaining after the resume point, or set run.restart_reset_frqtim = false to keep the schedule the restart carried",
            option="run.history_interval_s",
            value=config.run.history_interval_s,
        )


def require_new_history_root(
    directory: Path | None, *, option: str, kind: str = "history",
    code: str = "CM1-CARTESIAN-HISTORY-DIRECTORY",
    collision: str = "frames written into it would be read alongside another run's frames as one trajectory",
) -> Path | None:
    """An output root must be new: one run's artifacts are never mixed in.

    ``kind``, ``code`` and ``collision`` let the restart root refuse under its
    own name and its own consequence.  A restart directory that already held
    another run's restarts is worse than a mixed history directory, because
    the number a later resume selects would name a state from a different
    trajectory.
    """

    if directory is None:
        return None
    requested = Path(directory).expanduser()
    if requested.exists() or requested.is_symlink():
        refuse(
            code=code,
            gate=CARTESIAN_GATE,
            reason=f"the {kind} directory already exists; {collision}",
            remedy=f"pass --{option} with a new directory under an existing parent",
            option=option,
            value=str(requested),
        )
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        refuse(
            code=code,
            gate=CARTESIAN_GATE,
            reason=f"the {kind} directory parent cannot be resolved exactly: {error}",
            remedy=f"create the intended parent and pass --{option} with a new directory inside it",
            option=option,
            value=str(requested),
        )
    if not parent.is_dir():
        refuse(
            code=code,
            gate=CARTESIAN_GATE,
            reason=f"the {kind} directory parent is not a directory",
            remedy=f"create the intended parent and pass --{option} with a new directory inside it",
            option=option,
            value=str(parent),
        )
    return parent / requested.name


def require_ensemble(config: CM1Config) -> dict[str, object] | None:
    """Bind ``[ensemble]`` to one reproducible member, or return the control.

    ``None`` means the document declares no member, which is the unperturbed
    control the lane has always run.  Nothing here touches a device: the
    perturbation is priced and named on the host, so a receipt says which
    member a run was before the run starts.
    """

    from .ensemble_branch import (
        CM1_IRANDP_AMPLITUDE_K,
        ENSEMBLE_BRANCH_AUTHORITY,
        ENSEMBLE_BRANCH_CLAIM_LIMIT,
        ENSEMBLE_PERTURBED_CARRIERS,
    )

    table = config.ensemble
    if table is None:
        return None
    grid = config.grid
    return {
        "authority": ENSEMBLE_BRANCH_AUTHORITY,
        "branch_point": (
            "the restart named by run.rstnum"
            if config.run.restarting
            else "the initialized state before the first large step"
        ),
        "cells": int(grid.nx) * int(grid.ny) * int(grid.nz),
        "claim_limit": ENSEMBLE_BRANCH_CLAIM_LIMIT,
        "cm1_irandp_amplitude_k": CM1_IRANDP_AMPLITUDE_K,
        "member": table.member,
        "perturbed_carriers": list(ENSEMBLE_PERTURBED_CARRIERS),
        "seed": table.seed,
        "theta_amplitude_k": table.theta_amplitude_k,
    }


def resolve_initializer_library(argument: Path | None) -> Path:
    """Resolve the Rust idealized initializer, installed or explicit."""

    if argument is not None:
        return _require_regular_file(
            argument, "CM1-CARTESIAN-INITIALIZER-PATH", "library"
        )
    from . import rust_tools

    # rust_tools seals the cdylib under the same name its Cargo package uses,
    # which is also the key in EXPECTED_LIBRARIES.  Do not guess it here: a
    # wrong name reads as "no installed library" and silently sends the user
    # to --library when a perfectly good one is installed.
    resolver = getattr(rust_tools, "native_library_path", None)
    if resolver is not None:
        try:
            candidate = resolver(CARTESIAN_INITIALIZER_LIBRARY)
        except (OSError, TypeError, ValueError, rust_tools.NativeToolError) as error:
            refuse(
                code="CM1-CARTESIAN-INITIALIZER-MISSING",
                gate=CARTESIAN_GATE,
                reason=f"the installed native initializer library is not admitted: {error}",
                remedy="pass --library with a verified gm1 idealized initializer, or reinstall a platform wheel whose native library passes its manifest",
            )
        if candidate is not None:
            return _require_regular_file(
                candidate, "CM1-CARTESIAN-INITIALIZER-PATH", "library"
            )
    refuse(
        code="CM1-CARTESIAN-INITIALIZER-MISSING",
        gate=CARTESIAN_GATE,
        reason="this installation exposes no native initializer library; the model column can only be built by the Rust CM1 initializer, never by a Python substitute",
        remedy="pass --library with the built gm1 idealized initializer, or install a platform wheel that ships it",
        option="library",
        value=None,
    )


def resolve_sounding(argument: Path | None) -> tuple[str, bytes]:
    """Return the labelled origin and exact bytes of the input sounding."""

    if argument is not None:
        path = _require_regular_file(
            argument, "CM1-CARTESIAN-SOUNDING-PATH", "sounding"
        )
        try:
            payload = path.read_bytes()
        except OSError as error:
            refuse(
                code="CM1-CARTESIAN-SOUNDING-PATH",
                gate=CARTESIAN_GATE,
                reason=f"the input sounding cannot be read exactly: {error}",
                remedy="supply a readable CM1 r21.1 input_sounding",
                option="sounding",
                value=str(path),
            )
        return _label_supplied_sounding(path, payload), payload
    # The sounding is a CM1 r21.1 runtime asset. CM1-AUTHORITY.json's
    # redistribution record names the tar-only runtime assets as a blocker and
    # its package policy is to ship identities, not CM1 runtime files, so this
    # package records what is required and refuses rather than bundling it.
    from .data import PackagedInputError, required_input_identity

    try:
        identity = required_input_identity(CARTESIAN_SOUNDING_RESOURCE)
    except (PackagedInputError, OSError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-SOUNDING-MISSING",
            gate=CARTESIAN_GATE,
            reason=f"this installation carries no required-input identity for the idealized sounding: {error}",
            remedy="pass --sounding with a CM1 r21.1 input_sounding covering the model column",
            option="sounding",
            value=None,
        )
    refuse(
        code="CM1-CARTESIAN-SOUNDING-NOT-BUNDLED",
        gate=CARTESIAN_GATE,
        reason=(
            "this package does not redistribute CM1 runtime assets, so no input "
            "sounding is installed, and the Rust initializer never invents or "
            "extrapolates a profile"
        ),
        remedy=(
            "pass --sounding with "
            f"{identity['source_release']}/{identity['source_path']} from your own "
            "authority extraction (tools/authority/fetch_cm1_21_1.py retrieves the "
            f"release); the expected file is {identity['bytes']} bytes, sha256 "
            f"{identity['sha256']}"
        ),
        option="sounding",
        value=None,
    )


def _label_supplied_sounding(path: Path, payload: bytes) -> str:
    """Name the supplied profile, and say so when it is a recorded input.

    The package records the identity of the CM1 assets it needs without
    redistributing them, so a caller who supplies the recorded file gets that
    recorded name in the receipt instead of a machine-local path.
    """

    from .data import PackagedInputError, required_inputs

    digest = _sha256(payload)
    try:
        recorded = required_inputs()
    except (PackagedInputError, OSError, ValueError):
        return str(path)
    for name, identity in recorded.items():
        if identity.get("sha256") == digest:
            return f"{name} (supplied: {path})"
    return str(path)


def require_sounding_extent(
    label: str, payload: bytes, grid: CartesianGridGeometry
) -> tuple[float, float]:
    """Prove the profile covers every mass level before the GPU is opened.

    The initializer prepends CM1's surface record at z=0 before interpolating
    (``rust/cm1-state/src/idealized_initialization.rs:357``), so its admitted
    interval is ``(0, z[-1]]`` and not ``(first data height, z[-1]]``.  Only
    the top bound can bind for a positive spacing.

    Checking the first data height instead refused every grid whose first mass
    level sits below it: with this 40 km profile, whose first record is at
    124 m, that is every ``dz`` below 248 m.  The initializer accepts those
    grids, and the 125 m configurations in the evidence tree are exactly
    them.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        refuse(
            code="CM1-CARTESIAN-SOUNDING-FORMAT",
            gate=CARTESIAN_GATE,
            reason=f"the input sounding is not UTF-8 text: {error}",
            remedy="supply the pristine CM1 r21.1 input_sounding text",
            option="sounding",
            value=label,
        )
    heights: list[float] = []
    # Line one is CM1's surface record; every later line begins with a height.
    for number, line in enumerate(text.splitlines()[1:], start=2):
        if not line.split():
            continue
        try:
            heights.append(float(line.split()[0]))
        except ValueError:
            refuse(
                code="CM1-CARTESIAN-SOUNDING-FORMAT",
                gate=CARTESIAN_GATE,
                reason=f"line {number} of the input sounding does not begin with a height in metres",
                remedy="supply a CM1 r21.1 input_sounding whose records are height, theta, qv, u, v",
                option="sounding",
                value=label,
            )
    if len(heights) < 2:
        refuse(
            code="CM1-CARTESIAN-SOUNDING-FORMAT",
            gate=CARTESIAN_GATE,
            reason="the input sounding carries fewer than two levels, so no height bracket exists",
            remedy="supply a complete CM1 r21.1 input_sounding",
            option="sounding",
            value=label,
        )
    # zh(1) and zh(nk) of this grid's own coordinate.  On a uniform grid these
    # are 0.5*dz and ztop-0.5*dz; on a stretched one they are the generated
    # levels, and reconstructing them from a scalar dz would refuse every
    # tornado-grade column whose top layer is far thicker than dz.
    lowest_mass = grid.lowest_mass_level_m
    highest_mass = grid.highest_mass_level_m
    if heights[-1] < highest_mass:
        refuse(
            code="CM1-CARTESIAN-SOUNDING-EXTENT",
            gate=CARTESIAN_GATE,
            reason="the sounding top does not reach the highest mass level of this column; the initializer would refuse after the device context, component warmup and model allocation were already paid for",
            remedy=f"supply a profile whose top reaches {highest_mass} m, this grid's highest scalar level, or lower the model top",
            option="sounding",
            value={
                "source": label,
                "sounding_interval_m": [heights[0], heights[-1]],
                "required_mass_interval_m": [lowest_mass, highest_mass],
            },
        )
    return heights[0], heights[-1]


#: ``0`` stops runs from adding their measured step time to this machine's
#: rate cache (``cm1run.rates``); anything else, or unset, keeps it on.
LEARN_RATES_ENV = "GPUWM_CM1_LEARN_RATES"


def learn_step_rate(receipt: Mapping[str, Any] | None, output: Path) -> bool:
    """After every run, keep its mean step time for the next estimate.

    After every run, the measured rate is stored in
    ~/.cache/gpuwm-cm1/rates.json and used next time.  Best
    effort and after the receipt is durable: a cache that cannot be written,
    or a receipt with no step time, never touches the run or its exit status.
    """

    if os.environ.get(LEARN_RATES_ENV, "1") == "0":
        return False
    try:
        from .cm1run.rates import learn_receipt

        # From the published file, so the point is keyed by the same bytes
        # "Measure speed" or a later ``gpuwm-cm1 estimate RUNDIR`` would read.
        return learn_receipt(Path(output), source="run")[1]
    except Exception:  # noqa: BLE001 - the cache is a convenience, never a failure
        return False


def require_new_receipt_path(argument: Path) -> Path:
    """Sealed, non-overwriting receipt target under an existing parent."""

    requested = Path(argument).expanduser()
    if requested.exists() or requested.is_symlink():
        refuse(
            code="CM1-CARTESIAN-OUTPUT",
            gate=CARTESIAN_GATE,
            reason="the run receipt already exists or is a symlink; overwriting it would destroy the independent record of an earlier execution",
            remedy="choose a new .json path inside an existing directory",
            option="output",
            value=str(requested),
        )
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-CARTESIAN-OUTPUT",
            gate=CARTESIAN_GATE,
            reason=f"the receipt parent cannot be resolved exactly: {error}",
            remedy="create the intended directory and pass a new .json path inside it",
            option="output",
            value=str(requested),
        )
    output = parent / requested.name
    if not parent.is_dir() or output.suffix.lower() != ".json":
        refuse(
            code="CM1-CARTESIAN-OUTPUT",
            gate=CARTESIAN_GATE,
            reason="the receipt must be a new .json file in an existing directory",
            remedy="choose a non-existent .json path; existing receipts are never overwritten",
            option="output",
            value=str(output),
        )
    return output


def require_cuda_runtime():
    """Import CuPy and prove a device exists, naming exactly what is absent."""

    if importlib.util.find_spec("cupy") is None:
        refuse(
            code="CM1-CARTESIAN-CUPY-MISSING",
            gate=CARTESIAN_GATE,
            reason="CuPy is not installed, so this distribution cannot drive a single CUDA kernel of the Cartesian runtime",
            remedy='install the CUDA extra, for example pip install "gpuwm-cm1[gpu]"',
            option="cupy",
            value=None,
        )
    try:
        import cupy
    except ImportError as error:
        refuse(
            code="CM1-CARTESIAN-CUPY-MISSING",
            gate=CARTESIAN_GATE,
            reason=f"CuPy is discoverable but cannot be imported: {error}",
            remedy="install a CuPy build matching the installed CUDA toolkit and driver",
            option="cupy",
            value=None,
        )
    try:
        devices = int(cupy.cuda.runtime.getDeviceCount())
    except (OSError, RuntimeError) as error:
        refuse(
            code="CM1-CARTESIAN-CUDA-DEVICE",
            gate=CARTESIAN_GATE,
            reason=f"the CUDA runtime cannot enumerate a device: {error}",
            remedy="run on a host with a working NVIDIA driver and a visible device",
            option="cuda_device_count",
            value=None,
        )
    if devices < 1:
        refuse(
            code="CM1-CARTESIAN-CUDA-DEVICE",
            gate=CARTESIAN_GATE,
            reason="no CUDA device is visible, so the device-resident runtime has nowhere to allocate the model state",
            remedy="expose exactly one device, for example through CUDA_VISIBLE_DEVICES",
            option="cuda_device_count",
            value=devices,
        )
    return cupy


def allocation_plan(runtime_config: RuntimeConfig) -> dict[str, object]:
    """CPU-side pricing of the run; allocation-free and CUDA-cold."""

    from .cuda_cartesian_closure import cartesian_closure_allocation_plan
    from .cuda_cartesian_dynamics import cartesian_dynamics_allocation_plan

    return {
        "config": runtime_config.record(),
        "dynamics": cartesian_dynamics_allocation_plan(runtime_config),
        "closure": cartesian_closure_allocation_plan(runtime_config),
    }


def _require_regular_file(argument: object, code: str, option: str) -> Path:
    try:
        path = Path(argument).expanduser().resolve(strict=True)
    except (OSError, TypeError) as error:
        refuse(
            code=code,
            gate=CARTESIAN_GATE,
            reason=f"the {option} path cannot be resolved exactly: {error}",
            remedy=f"pass --{option} with an existing regular file",
            option=option,
            value=str(argument),
        )
    if path.is_symlink() or not path.is_file():
        refuse(
            code=code,
            gate=CARTESIAN_GATE,
            reason=f"the {option} path is not a regular file",
            remedy=f"pass --{option} with an existing regular non-symlink file",
            option=option,
            value=str(path),
        )
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_profile_level(value: object) -> str:
    """``--profile``: how much of the run's performance is recorded."""

    from .runtime_profiler import PROFILE_LEVELS, require_profile_level as admitted

    try:
        return admitted(value)
    except ValueError:
        refuse(
            code="CM1-CARTESIAN-PROFILE",
            gate=CARTESIAN_GATE,
            reason=(
                f"--profile {value!r} is not a profiling level; an unrecognised "
                "level would either silently record nothing or install stream "
                "markers and NVTX ranges the operator did not ask for"
            ),
            remedy=(
                f"pass one of {', '.join(PROFILE_LEVELS)}: 'off' records nothing, "
                "'summary' (the default) records per-phase device/host time, "
                "launches, copies and I/O stages, 'detailed' adds per-carrier I/O "
                "rows and NVTX ranges for nsys"
            ),
            option="profile",
            value=value,
        )


def require_cuda_graph_mode(value: object, config: Any = None) -> str:
    """``--cuda-graphs``: whether large steps are captured once and replayed.

    An execution control, like ``--profile``: it is not part of the sealed
    configuration and changes no word (cm1core.cuda_graphs).  ``verify``
    captures every step and compares each graph with the one ``on`` would
    have replayed, argument byte by argument byte; it replays nothing.
    """

    from .cuda_graphs import CUDA_GRAPH_MODES

    if not isinstance(value, str) or value not in CUDA_GRAPH_MODES:
        refuse(
            code="CM1-CARTESIAN-CUDA-GRAPHS",
            gate=CARTESIAN_GATE,
            reason=(
                f"--cuda-graphs {value!r} is not a graph mode; an unrecognised "
                "mode would either replay steps the operator did not ask to "
                "replay or silently run every step in Python"
            ),
            remedy=(
                "pass 'off' (the default: every step runs its schedule in "
                "Python), 'on' (capture a step once per key and replay it) or "
                "'verify' (capture every step and compare it with the graph "
                "'on' would replay)"
            ),
            option="cuda-graphs",
            value=value,
        )
    if value != "off" and config is not None:
        refusal = cuda_graph_configuration_refusal(config)
        if refusal is not None:
            refuse(
                code="CM1-CARTESIAN-CUDA-GRAPHS-UNPROVEN",
                gate=CARTESIAN_GATE,
                reason=(
                    f"--cuda-graphs {value} on {refusal}: no capture of this "
                    "configuration has been compared word for word with its "
                    "Python schedule, so a replay could launch a step with host "
                    "state frozen at another step"
                ),
                remedy="run with --cuda-graphs off",
                option="cuda-graphs",
                value=value,
            )
    return value


def cuda_graph_configuration_refusal(config: Any) -> str | None:
    """The first configuration word ``--cuda-graphs`` is not proven on, or None."""

    if getattr(config, "chain", None) is not None:
        return "an ArWen chain run ([chain]), whose boundary tables are replaced between steps"
    initialization = getattr(config, "initialization", None)
    if initialization is not None and int(getattr(initialization, "iinit", 1)) in (9, 10, 12):
        return (
            f"initialization.iinit = {initialization.iinit}, whose forcing ramps with "
            "the model time (init3d.F:1303-1396) and switches itself off; a replayed "
            "step would freeze the ramp at the captured time"
        )
    return None


def _initiation_forcing(config: Any, grid: Any) -> Any:
    """init3d.F:1303-1396's forcing for this document, or None."""

    from .initiation_forcing import resolve_initiation_forcing

    return resolve_initiation_forcing(
        config.initialization, nx=grid.nx, ny=grid.ny, dx_m=grid.dx_m, dy_m=grid.dy_m
    )


def require_event_parts(value: object) -> bool:
    """``--event-parts``: whether a frame and its step's restart are written beside each other.

    Returns ``integrate``'s ``concurrent_event_parts``.  ``beside`` (the
    default) has the background worker write the two parts of a same-step
    event at once, renamed frame first; ``in-turn`` writes them one after the
    other, for a disk whose two writes contend, a nearly full disk, or a host
    short of memory for both publishers' payloads at once.
    """

    from .background_publication import EVENT_PARTS_BESIDE, EVENT_PARTS_IN_TURN

    if isinstance(value, str) and value == EVENT_PARTS_BESIDE:
        return True
    if isinstance(value, str) and value == EVENT_PARTS_IN_TURN:
        return False
    refuse(
        code="CM1-CARTESIAN-EVENT-PARTS",
        gate=CARTESIAN_GATE,
        reason=(
            f"--event-parts {value!r} is not a way of committing a frame and the "
            "restart of the same step; an unrecognised value would either write "
            "them beside each other on a disk or host the operator knows cannot "
            "take both at once, or serialise them and cost the stepping thread "
            "the frame's whole write on every restart"
        ),
        remedy=(
            f"pass {EVENT_PARTS_BESIDE!r} (the default: the restart is written "
            "while the frame is, renamed after it) or "
            f"{EVENT_PARTS_IN_TURN!r} (one after the other: for storage whose two "
            "writes contend, a nearly full disk, or a host without memory for "
            "both publishers' payloads at once)"
        ),
        option="event-parts",
        value=value,
    )


def require_progress_interval(value: object) -> float:
    """``--progress-interval``: seconds between live progress lines."""

    from .runtime_profiler import require_progress_interval as admitted

    try:
        return admitted(value)
    except ValueError:
        refuse(
            code="CM1-CARTESIAN-PROGRESS-INTERVAL",
            gate=CARTESIAN_GATE,
            reason=(
                f"--progress-interval {value!r} is not a finite, nonnegative number "
                "of seconds; a negative or non-numeric cadence cannot be scheduled "
                "and would either print on every step or never"
            ),
            remedy="pass a number of wall-clock seconds such as 30, or 0 to print no progress lines",
            option="progress-interval",
            value=value,
        )


def perf_door(arguments: argparse.Namespace) -> int:
    """``gpuwm-cm1 perf RECEIPT``: print a run receipt's performance section."""

    from .runtime_profiler import format_performance

    path = Path(arguments.receipt)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        refuse(
            code="CM1-PERF-RECEIPT",
            gate=CARTESIAN_GATE,
            reason=f"the run receipt could not be read as JSON: {error}",
            remedy="pass the receipt a Cartesian run wrote with --output",
            option="receipt",
            value=str(path),
        )
    performance = None
    if isinstance(document, dict):
        benchmark = document.get("runtime_benchmark")
        if isinstance(benchmark, dict):
            performance = benchmark.get("performance")
        if performance is None:
            performance = document.get("performance")
    if not isinstance(performance, dict):
        refuse(
            code="CM1-PERF-RECEIPT",
            gate=CARTESIAN_GATE,
            reason=(
                "the receipt carries no performance section; it was written by a "
                "run with --profile off or by a version that did not record one, "
                "so there is nothing measured to summarise"
            ),
            remedy="re-run the document with --profile summary (the default) or --profile detailed",
            option="receipt",
            value=str(path),
        )
    if getattr(arguments, "json", False):
        print(json.dumps(performance, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_performance(performance), end="")
    return 0


def _frame_progress(row: dict[str, object]) -> None:
    """One line per published frame, so a long run is followable live.

    This goes to stderr: --json puts the canonical receipt on stdout, and a
    progress line mixed into it would make the receipt unparseable.
    """

    label = (
        f"frame {row['index']}" if "index" in row
        else f"restart {row['number']}"
    )
    written = row.get("restart_written")
    print(
        f"{label} step {row['step']} t={row['model_time_s']}s -> {row['directory']}"
        + ("" if written is None else f" (+restart {written})"),
        file=sys.stderr,
        flush=True,
    )


HISTORY_ENVELOPE_CHOICES = ("on", "off")
#: The door's receipt status for an integration a run-control file ended.
CONTROL_STOP_STATUS = "stopped-at-run-control-request"
CONTROL_PAUSE_STATUS = "paused-at-run-control-request"


def require_history_envelope(value: object) -> bool:
    """``--history-envelope``: whether each frame also gets its CM1FHS1 envelope.

    Returns ``True`` for ``on``.  ``off`` publishes the raw arena and its
    tables only: the envelope is a byte-for-byte second copy of the arena
    (443,189,440 B arena and 443,198,452 B envelope per 256x256x80 Morrison
    frame, measured), which halves a run's frame disk.  Restarts are
    unaffected: their envelope is the artifact a resume reads.
    """

    if isinstance(value, str) and value in HISTORY_ENVELOPE_CHOICES:
        return value == "on"
    refuse(
        code="CM1-CARTESIAN-HISTORY-ENVELOPE",
        gate=CARTESIAN_GATE,
        reason=(
            f"--history-envelope {value!r} is neither on nor off; an unrecognised "
            "value would either write every frame twice or publish frames with no "
            "envelope a reader expected"
        ),
        remedy=(
            "pass 'on' (the default: each frame also gets history.cm1fhs, which "
            "gpuwm-cm1 render and cm1-render read) or 'off' (the raw arena and "
            "its field table only, half the disk)"
        ),
        option="history-envelope",
        value=value,
    )


def open_run_events(path: Path) -> Any:
    """Open ``--events PATH`` as a new file, or refuse naming why not."""

    from .run_events import RunEventLog

    try:
        return RunEventLog(path)
    except FileExistsError:
        refuse(
            code="CM1-CARTESIAN-EVENTS",
            gate=CARTESIAN_GATE,
            reason=(
                f"{path} already exists; a second run's records appended to it would "
                "be read as one run whose steps went backwards"
            ),
            remedy="pass --events with a new file (one log per run segment)",
            option="events",
            value=str(path),
        )
    except OSError as error:
        refuse(
            code="CM1-CARTESIAN-EVENTS",
            gate=CARTESIAN_GATE,
            reason=f"the event log could not be created: {error}",
            remedy="pass --events with a new file in a writable directory",
            option="events",
            value=str(path),
        )


class _RunReporting:
    """The event log and the run-control directory of one door call.

    It writes the ``frame``/``restart`` records as publications land, the
    ``admitted`` record once admission is over, and the ``end`` record from
    however the call ended.  With neither an event log nor a control
    directory it is inert and the door behaves exactly as it always has.
    """

    def __init__(self, events: Any = None, control: Any = None) -> None:
        self.events = events
        self.control = control
        self.admitted = False
        self.receipt: str | None = None
        self.reason: str | None = None
        self.request: str | None = None
        #: Why a run that ended without an error ended early, when no control
        #: request explains it (a [nest.tornado_criterion] stop rule).
        self.message: str | None = None
        self.last_step: int | None = None
        self.last_model_time_s: float | None = None
        self.frames = 0
        self.restarts = 0

    def frame_progress(self, row: dict[str, object]) -> None:
        _frame_progress(row)
        self.last_step = int(row["step"])  # type: ignore[arg-type]
        self.last_model_time_s = float(row["model_time_s"])  # type: ignore[arg-type]
        from .run_events import directory_bytes

        if "index" in row:
            self.frames += 1
            if self.events is not None:
                self.events.emit(
                    "frame",
                    index=int(row["index"]),  # type: ignore[arg-type]
                    step=int(row["step"]),  # type: ignore[arg-type]
                    model_time_s=float(row["model_time_s"]),  # type: ignore[arg-type]
                    directory=str(row["directory"]),
                    bytes=directory_bytes(str(row["directory"])),
                    initial_condition=int(row["index"]) == 0,  # type: ignore[arg-type]
                    restart_written=row.get("restart_written"),
                )
        restart = row.get("restart")
        if isinstance(restart, dict):
            self.restarts += 1
            if self.events is not None:
                self.events.emit(
                    "restart",
                    number=int(restart["number"]),
                    step=int(restart["step"]),
                    model_time_s=float(restart["model_time_s"]),
                    directory=str(restart["directory"]),
                    bytes=directory_bytes(str(restart["directory"])),
                    next_history_index=int(restart["next_history_index"]),
                )

    def admit(self, **fields: Any) -> None:
        self.admitted = True
        if self.events is not None:
            self.events.emit("admitted", **fields)
            self.events.set_phase("starting")
            self.events.start_heartbeat()

    def phase(self, name: str) -> None:
        if self.events is not None:
            self.events.set_phase(name)

    def end(self, error: BaseException | None, rc: int | None = None) -> None:
        """Write the ``end`` record for however the door call ended."""

        events = self.events
        if events is None or events.ended:
            return
        from .background_publication import TerminationRequested

        signal_name = message = None
        if error is None:
            reason = self.reason or "completed"
            rc = 0 if rc is None else int(rc)
            message = self.message
        elif isinstance(error, TerminationRequested):
            reason, rc, signal_name = "stopped", 143, "SIGTERM"
        elif isinstance(error, KeyboardInterrupt):
            reason, rc, signal_name = "stopped", 130, "SIGINT"
        elif isinstance(error, CM1Refusal):
            detail = error.detail
            rc = 2
            message = detail.reason
            if detail.code == "CM1-CARTESIAN-COURANT-STOP":
                reason = "courant-stop"
            else:
                reason = "error" if self.admitted else "refused"
                events.emit(
                    "refused", code=detail.code, gate=detail.gate,
                    reason=detail.reason, remedy=detail.remedy,
                    option=getattr(detail, "option", None),
                )
        elif isinstance(error, SystemExit):
            code = error.code
            reason = "error"
            rc = code if isinstance(code, int) else 1
            message = None if code is None or isinstance(code, int) else str(code)
        else:
            reason, rc = "error", (2 if isinstance(error, (OSError, UnicodeError, ValueError)) else 1)
            message = f"{type(error).__name__}: {error}"
        # Where the model clock stands now (a host read, no device call);
        # before stepping began, the last published event's, if any.
        step, model_time_s = self.last_step, self.last_model_time_s
        runtime = None if self.control is None else self.control.runtime
        if runtime is not None:
            step = int(runtime.clock.completed_steps)
            model_time_s = float(runtime.clock.model_seconds)
        events.emit(
            "end",
            reason=reason,
            rc=int(rc),
            receipt=self.receipt,
            step=step,
            model_time_s=model_time_s,
            frames_published=self.frames,
            restarts_published=self.restarts,
            signal=signal_name,
            request=self.request,
            message=message,
        )


def _execute(
    *,
    cupy,
    library: Path,
    sounding_bytes: bytes,
    selection: InitializerSelection,
    moving_frame: "TranslatingFrame",
    adaptive_frame_binding: "AdaptiveFrameBinding | None",
    origin_xy_m: tuple[float, float],
    bubble_center_xy_m: tuple[float, float],
    grid: CartesianGridGeometry,
    runtime_config: RuntimeConfig,
    capacity_profile: str | None,
    warmup_steps: int,
    steps: int,
    trace_phases: bool,
    history_dir: Path | None,
    history_carrier_profile: str,
    history_publisher: Path | None,
    schedule: RunSchedule | None,
    frame_publisher: Any = "same",
    restart_dir: Path | None,
    restart_interval_s: float | None,
    reset_frqtim: bool,
    resume_from: Path | None,
    resume_scratch: Path | None,
    initial_condition_only: bool,
    gpu_uuid: str | None,
    lock: Path | None,
    identity_sha256: str | None = None,
    config_sha256: str | None = None,
    ensemble: dict[str, object] | None = None,
    nest: "NestBinding | None" = None,
    chain: Any = None,
    profile: str = "off",
    progress_interval_s: float = 0.0,
    concurrent_event_parts: bool = True,
    cuda_graphs: str = "off",
    history_initial: bool = False,
    control: Any = None,
    reporting: "_RunReporting | None" = None,
) -> tuple[dict[str, object], str | None]:
    """Drive the real runtime under the ecosystem GPU mutex when named.

    Three entry points, selected before the device opens.
    ``initial_condition_only`` is CM1's ``iconly`` (cm1.F:4341-4350): build the
    state, write it, integrate nothing.  A ``schedule`` integrates to the
    sealed duration, publishing a frame at every history interval and a
    complete restart at every restart interval.  Neither runs the timing
    harness, which is what this door has always run without them.

    ``resume_from`` is ``cm1.F:2666`` under ``irst=1``: the carried state is
    read back through the Rust boundary that wrote it and put into the freshly
    initialized runtime before a single large step is taken.
    """

    from .cartesian_restart import (
        cartesian_restart_profile,
        read_cartesian_restart,
    )
    from .cuda_cartesian_runtime import CartesianCourantStop, CudaCartesianRuntime
    from .gm1_idealized_residency import IdealizedSource

    # ``history_publisher`` reads and writes restarts; ``frame_publisher`` seals
    # history frames, and is None under --history-envelope off.
    if frame_publisher == "same":
        frame_publisher = history_publisher
    if gpu_uuid is None:
        guard = nullcontext(None)
    else:
        from .runtime_gpu_guard import exclusive_runtime_gpu

        guard = exclusive_runtime_gpu(gpu_uuid, directory_lock=lock)
        if reporting is not None:
            reporting.phase("waiting-for-gpu")
    with guard as token, ExitStack() as nest_sources:
        if reporting is not None:
            reporting.phase("starting")
        profiler = None
        if profile != "off":
            from .runtime_profiler import RuntimeProfiler

            # Installed before anything compiles, so every NVRTC program is
            # timed and every kernel launch is counted, and removed when this
            # block exits however it exits.  Nothing CuPy returns is wrapped,
            # and a hook that fails turns itself off rather than stop the run.
            profiler = RuntimeProfiler(
                profile, cupy=cupy, progress_interval_s=progress_interval_s,
            ).install_hooks()
            nest_sources.callback(profiler.uninstall_hooks)
        if cuda_graphs == "off":
            stream = cupy.cuda.Stream(non_blocking=True)
        else:
            from .cuda_graphs import capture_stream_class

            stream = capture_stream_class(cupy)(non_blocking=True)
        if profiler is not None:
            profiler.bind_stream(stream)
        source_started = time.perf_counter()
        with IdealizedSource(
            library,
            sounding_bytes,
            grid,
            selection=selection,
            moving_frame=moving_frame,
            origin_xy_m=origin_xy_m,
            bubble_center_xy_m=bubble_center_xy_m,
        ) as source:
            if profiler is not None:
                profiler.startup_stage(
                    "initializer.rust_idealized_source", time.perf_counter() - source_started,
                )
            with stream:
                runtime_class = CudaCartesianRuntime
                if chain is not None:
                    from .arwen_chain_runtime import make_chain_runtime_class

                    runtime_class = make_chain_runtime_class(CudaCartesianRuntime)
                construction_started = time.perf_counter()
                runtime = runtime_class(
                    source,
                    runtime_config,
                    cupy=cupy,
                    stream=stream,
                    capacity_profile=capacity_profile,
                )
                if profiler is not None:
                    profiler.startup_stage(
                        "runtime.construction", time.perf_counter() - construction_started,
                    )
                    runtime.profiler = profiler
                runtime.trace_phases = trace_phases
                # cm1.F:3669-3679 runs inside the completed large step, so the
                # controller is attached before the first one -- and before a
                # restart is restored, so restart_read.F:397-421's fifteen
                # persisted controller words land in it rather than in a
                # controller that does not exist yet.
                # A fixed-frame run binds nothing, so it asks nothing of the
                # runtime it drives; only an adaptive run needs the hook.
                if adaptive_frame_binding is not None:
                    runtime.attach_adaptive_frame(adaptive_frame_binding)
                restart_resume: dict[str, object] | None = None
                ensemble_receipt: dict[str, object] | None = None
                try:
                    resume = None
                    restored_child = None
                    if resume_from is not None:
                        # cm1.F:2666.  The restore directory is a sibling of the
                        # run's own output, created here and never reused, so the
                        # words that reach the device came out of this envelope.
                        restore_started = time.perf_counter()
                        restart_profile = cartesian_restart_profile(runtime_config)
                        restored = read_cartesian_restart(
                            resume_from, publisher=history_publisher, into=resume_scratch,
                        )
                        if nest is not None:
                            # WRF share/mediation_integrate.F:861-892: the nest
                            # reads its own restart.  Decoded here, on the host,
                            # into a sibling of the parent's scratch, before the
                            # parent's words reach the device: a torn child
                            # envelope stops the run before any are written.
                            from .nest_restart import read_nest_child_restart

                            restored_child = read_nest_child_restart(
                                restored.receipt, resume_from, publisher=history_publisher,
                                into=resume_scratch.parent / f"{resume_scratch.name}-nest-child",
                            )
                        resume = runtime.restore_restart(restored, profile=restart_profile)
                        if nest is None and restored.receipt.get("nest_child") is not None:
                            resume["nest_child_restart"] = (
                                "not resumed: the document declares no [nest], so the "
                                "parent continues alone from its own restart"
                            )
                        if profiler is not None:
                            profiler.startup_stage(
                                "restart.read_and_restore",
                                time.perf_counter() - restore_started,
                            )
                        print(
                            f"resumed {resume['step']} completed large steps "
                            f"(t={resume['model_time_s']}s) from {resume_from}",
                            file=sys.stderr, flush=True,
                        )
                        # restart_read.F:1064-1126 continues every swath; one
                        # this run keeps and the restart does not carry starts
                        # here, and the run says so where a reader will see it.
                        swath_warning = (resume.get("swaths") or {}).get("warning")
                        if swath_warning:
                            print(f"warning: {swath_warning}", file=sys.stderr, flush=True)
                            events = None if reporting is None else reporting.events
                            if events is not None:
                                events.emit(
                                    "warning", code="GM1-SWATHS-RESTART-HERE",
                                    message=swath_warning,
                                )
                    if ensemble is not None:
                        # init3d.F:1446-1483 perturbs before the model advances.
                        # After a restore, that is this point: the carriers are
                        # the branch point's, and no step has run on them yet.
                        # It must follow restore_restart -- applied before it,
                        # the restore overwrote the perturbation and every
                        # "branched" member ran unperturbed.
                        from .ensemble_branch import apply_ensemble_branch

                        ensemble_receipt = apply_ensemble_branch(
                            runtime,
                            member=int(ensemble["member"]),
                            seed=int(ensemble["seed"]),
                            theta_amplitude_k=float(ensemble["theta_amplitude_k"]),
                        ).as_dict()
                    nest_initialization: dict[str, object] | None = None
                    if nest is not None:
                        # frame/module_integrate.F:393-443.  The child is built
                        # after the parent is restored and branched, so its
                        # initial condition is the state the parent will
                        # actually integrate from (med_nest_initial).
                        attach_started = time.perf_counter()
                        nest_initialization = _attach_live_nest(
                            runtime, nest, stack=nest_sources, cupy=cupy,
                            stream=stream, library=library,
                            sounding_bytes=sounding_bytes, selection=selection,
                            moving_frame=moving_frame,
                            capacity_profile=capacity_profile,
                            resumed=resume is not None,
                            restored_child=restored_child,
                        )
                        if profiler is not None:
                            profiler.startup_stage(
                                "nest.attach", time.perf_counter() - attach_started,
                            )
                    graph_runtimes = []
                    if cuda_graphs != "off":
                        # After the restore and the nest attach, so the first
                        # step each runtime takes (eager, cm1core.cuda_graphs)
                        # builds its lazy workspaces from the state it runs on.
                        graph_runtimes.append(runtime.enable_cuda_graphs(
                            cuda_graphs, role="domain" if nest is None else "parent",
                        ))
                        if nest is not None:
                            graph_runtimes.append(runtime.nest_driver.child.enable_cuda_graphs(
                                cuda_graphs, role="nest-child",
                                memory=graph_runtimes[0].memory,
                            ))
                    chain_initialization: dict[str, object] | None = None
                    if chain is not None:
                        # ndown's initial state and wrfbdy tables, written
                        # after the initializer built the sounding base state.
                        from .arwen_chain_runtime import attach_arwen_chain

                        chain_initialization = attach_arwen_chain(
                            runtime, chain, cupy=cupy, stream=stream,
                        )
                    if initial_condition_only:
                        benchmark = runtime.initial_condition(
                            restart_root=restart_dir,
                            restart_publisher=history_publisher,
                            history_root=history_dir,
                            history_publisher=frame_publisher,
                            history_carrier_profile=history_carrier_profile,
                        )
                    elif schedule is None:
                        benchmark = runtime.benchmark(
                            warmup_steps=warmup_steps,
                            measured_steps=steps,
                            history_directory=history_dir,
                            history_publisher=frame_publisher,
                            history_carrier_profile=history_carrier_profile,
                        )
                    else:
                        benchmark = runtime.integrate(
                            duration_s=schedule.cadence.duration_s,
                            history_interval_s=schedule.cadence.interval_s,
                            history_root=history_dir,
                            history_publisher=frame_publisher,
                            history_carrier_profile=history_carrier_profile,
                            restart_root=restart_dir,
                            restart_interval_s=restart_interval_s,
                            restart_publisher=(
                                None if restart_dir is None else history_publisher
                            ),
                            restart_reset_frqtim=reset_frqtim,
                            first_history_index=schedule.first_history_index,
                            first_restart_number=schedule.first_restart_number,
                            progress=(
                                _frame_progress if reporting is None
                                else reporting.frame_progress
                            ),
                            concurrent_event_parts=concurrent_event_parts,
                            history_initial=history_initial,
                            control=control,
                        )
                    if resume is not None:
                        benchmark["resume"] = resume
                    if graph_runtimes:
                        benchmark["cuda_graphs"] = [graphs.record() for graphs in graph_runtimes]
                    if nest is not None:
                        benchmark["nest_execution"] = {
                            **runtime.nest_driver.record(),
                            "child_initialization": nest_initialization,
                        }
                    if chain is not None:
                        benchmark["arwen_chain"] = {
                            **runtime.chain_driver.record(),
                            "initialization": chain_initialization,
                        }
                    # What the controller did, beside what the document asked
                    # for: the branch, the cadence, the frame velocity it
                    # reached and the state a restart would carry.
                    if runtime.adaptive_frame is not None:
                        benchmark["adaptive_frame"] = runtime.adaptive_frame.record()
                except CartesianCourantStop as stop:
                    # CM1 writes the tripping state's output and then stops
                    # (cm1.F:3517-3520, cm1.F:4385-4392).  The frames and any
                    # restart are already on disk; the receipt that names the
                    # step and the model time must reach the caller with them,
                    # so it is finished here exactly as a completed one would be.
                    _stamp_environment(stop.receipt, cupy)
                    stop.gpu_mutex_owner = token
                    stop.restart_resume = restart_resume
                    stop.ensemble_branch = ensemble_receipt
                    raise
            _stamp_environment(benchmark, cupy)
            if restart_resume is not None:
                benchmark["restart_resume"] = restart_resume
            if ensemble_receipt is not None:
                benchmark["ensemble_branch"] = ensemble_receipt
    return benchmark, token


def _attach_live_nest(
    runtime, nest: "NestBinding", *, stack, cupy, stream, library: Path,
    sounding_bytes: bytes, selection: InitializerSelection,
    moving_frame: "TranslatingFrame", capacity_profile: str | None, resumed: bool,
    restored_child: Any = None,
) -> dict[str, object]:
    """Build the child runtime, fill it from the parent, and bind the driver.

    The child's own source builds the same base state the parent's does (the
    same sounding, the same frame, the same vertical deck); its perturbation
    words are then replaced by the parent's, captured onto the child's lattice
    on the device, so the child starts from the state the parent is in.

    ``restored_child`` -- the child restart a nested restart carries and its
    coupling state (:func:`cm1core.nest_restart.read_nest_child_restart`) --
    replaces that capture with the child's own words, at the placement and
    tracker state the interrupted run had reached (WRF
    ``share/mediation_integrate.F:861-892``).
    """

    from dataclasses import replace as _replace

    from .cuda_cartesian_runtime import CudaCartesianRuntime
    from .gm1_idealized_residency import IdealizedSource
    from .nest_live import NestDriver, child_runtime_config, make_child_runtime_class

    child_config = child_runtime_config(
        runtime.config, nest.child.grid, child_dt_s=nest.child.dt_s,
        acoustic_substeps=nest.child.acoustic_substeps,
    )
    source = stack.enter_context(IdealizedSource(
        library, sounding_bytes, child_config.grid,
        selection=_replace(selection, iinit=0), moving_frame=moving_frame,
    ))
    geometry = nest.geometry
    restored = state = None
    if restored_child is not None:
        restored, state = restored_child
        # The ratios and spans must be the document's; the placement is the
        # restart's, because a following nest has moved since.
        geometry = state.geometry(nest.geometry)
    with stream:
        child = make_child_runtime_class(CudaCartesianRuntime)(
            source, child_config, cupy=cupy, stream=stream,
            capacity_profile=capacity_profile,
        )
        child.dynamics.source = source
        driver = NestDriver(
            parent=runtime, child=child, geometry=geometry, cupy=cupy, stream=stream,
            follow=nest.follow_settings(), criterion=nest.tornado_criterion,
            restoring=state is not None,
        )
        if state is None:
            initialization = driver.initialize_child()
        else:
            initialization = driver.restore_child(restored, state)
    runtime.nest_driver = driver
    initialization["parent_state"] = (
        "restored restart" if resumed else "initialized parent at step 0"
    )
    if state is not None:
        initialization["child_state"] = "restored from the nest child's own restart"
    initialization["parent_completed_steps"] = int(runtime.clock.completed_steps)
    return initialization


def _stamp_environment(benchmark: dict[str, object], cupy) -> None:
    """Name the device and the software that produced one runtime record."""

    benchmark["gpu_name"] = cupy.cuda.runtime.getDeviceProperties(0)[
        "name"
    ].decode()
    benchmark["software"] = {
        "python": sys.version,
        "cupy": cupy.__version__,
        "cuda_driver_version": cupy.cuda.runtime.driverGetVersion(),
        "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "producer_sha256": _sha256(Path(__file__).read_bytes()),
    }


def _write_initial_condition(
    *,
    config: CM1Config,
    config_label: str,
    output: Path,
    library: Path,
    sounding_label: str | None,
    sounding_bytes: bytes,
    sounding_interval: object,
    selection: "InitializerSelection",
    selection_declared: bool,
    moving_frame: "TranslatingFrame",
    frame_mode: dict[str, Any],
    origin_xy_m: tuple[float, float],
    bubble_center_xy_m: tuple[float, float],
    grid: CartesianGridGeometry,
    runtime_config: RuntimeConfig,
    level_bounds: object,
    capacity_profile: str | None,
    history_dir: Path | None,
    unvalidated: dict[str, object],
    emit_json: bool,
) -> int:
    """CM1 ``iconly=1``: build the initial condition, write it, and stop.

    No CUDA module is imported and no device is opened.  The initial state is
    the Rust initializer's own host words, published in the runtime's frame
    layout at model time zero.
    """

    from .cartesian_initial_condition import (
        INITIAL_CONDITION_CLAIM_LIMIT,
        INITIAL_CONDITION_STATUS,
        publish_initial_condition,
    )
    from .gm1_idealized_residency import IdealizedSource

    if history_dir is None:
        refuse(
            code="CM1-CARTESIAN-HISTORY-DIRECTORY",
            gate=CARTESIAN_GATE,
            reason="run.iconly = 1 builds the initial condition so a user can look at it; with no history directory there would be nowhere to write it and the door would do nothing at all",
            remedy="pass --history-dir with a new directory to receive the model-time-zero frame",
            option="history-dir",
            value=None,
        )
    started = time.perf_counter()
    try:
        with IdealizedSource(
            library,
            sounding_bytes,
            grid,
            selection=selection,
            moving_frame=moving_frame,
            origin_xy_m=origin_xy_m,
            bubble_center_xy_m=bubble_center_xy_m,
        ) as source:
            frame = publish_initial_condition(
                source,
                runtime_config,
                history_dir / "frame-000000",
                moving_frame=moving_frame,
            )
    except CM1Refusal:
        raise
    except Exception as error:
        refuse(
            code="CM1-CARTESIAN-INITIAL-CONDITION",
            gate=CARTESIAN_GATE,
            reason=f"the initial condition could not be built ({type(error).__name__}): {error}",
            remedy="correct the [initialization] table or the supplied sounding; no partial initial condition is published",
            option="initialization",
            value=selection.record(),
        )
    receipt = {
        "schema": CARTESIAN_INITIAL_CONDITION_RECEIPT_SCHEMA,
        "status": INITIAL_CONDITION_STATUS,
        "authority_sha256": CM1_AUTHORITY_SHA256,
        "cm1_release": CM1_RELEASE,
        "distribution": "gpuwm-cm1",
        "version": __version__,
        "maturity": MATURITY,
        "lane": config.run.lane.value,
        "config": config_label,
        "config_sha256": config.sha256,
        "admitted_level_bounds": list(level_bounds),
        "capacity_profile": capacity_profile,
        "initialization": selection.record(),
        "initialization_declared": selection_declared,
        "initialization_origin_xy_m": list(origin_xy_m),
        "initialization_bubble_center_xy_m": list(bubble_center_xy_m),
        "unvalidated": unvalidated,
        "moving_frame": moving_frame_record(moving_frame),
        "frame_mode": frame_mode,
        "initializer_library": str(library),
        "initializer_library_sha256": _sha256(library.read_bytes()),
        "sounding": sounding_label,
        "sounding_sha256": None if sounding_label is None else _sha256(sounding_bytes),
        "sounding_interval_m": None if sounding_interval is None else list(sounding_interval),
        "grid": grid.record(),
        "history_directory": str(history_dir),
        "initial_condition": frame,
        "device_opened": False,
        "output": str(output),
        "process_wall_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "claim_limit": INITIAL_CONDITION_CLAIM_LIMIT,
    }
    document = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False, default=dict)
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(document + "\n")
    except OSError as error:
        refuse(
            code="CM1-CARTESIAN-OUTPUT",
            gate=CARTESIAN_GATE,
            reason=f"the initial-condition receipt could not be published without overwriting: {error}",
            remedy="re-run against a new receipt path on a writable filesystem",
            option="output",
            value=str(output),
        )
    if emit_json:
        print(document)
    else:
        print(f"initial condition written to {history_dir / 'frame-000000'}", file=sys.stderr)
    return 0


def cartesian_plan_payload(
    config: CM1Config,
    *,
    cadence: "HistoryCadence | None" = None,
    sounding: Path | None = None,
    schedule: "RunSchedule | None" = None,
    restart_dir: Path | None = None,
    resume_directory: Path | None = None,
    resume_receipt: dict | None = None,
) -> dict[str, object]:
    """Price one document on the host: no device, no disk, no engine estate.

    This is the body of ``--plan``, factored out so a caller -- and a test --
    reaches exactly what the command line prints rather than a second
    assembly of the same keys.  Every function it calls is pure and host-side,
    so it may be called as many times as a caller likes.

    The keyword arguments are the host context ``--plan`` has already resolved
    by the time it prints: the profile this run would read, the large-step
    schedule an integration proved, and the restart selection the filesystem
    answered.  They default to ``None`` so a caller that only wants the
    document priced -- a test, or a pricing tool -- reaches the same keys the
    command line prints with the same code, and the run-geometry entry says
    by value that no profile was resolvable rather than going silent.
    """

    require_cartesian_lane(config)
    require_listed_component(config)
    runtime_config = build_runtime_config(config)
    grid = runtime_config.grid
    selection, selection_declared = require_initializer_selection(config)
    moving_frame = require_moving_frame(config)
    bubble_center = require_bubble_center(config, grid)
    assert config.runtime is not None  # build_runtime_config refuses otherwise
    # Price the geometry too, and by the same rules.  A plan whose grid
    # cannot hold its own storm is not a plan worth executing, and this is
    # the cheapest moment anyone will ever learn it.  A plan may be taken
    # without a resolvable profile, so the failure to read one is reported
    # here rather than refused; the run path refuses.
    try:
        plan_geometry: dict[str, object] = require_run_geometry(
            config,
            grid,
            selection=selection,
            moving_frame=moving_frame,
            center_xy_m=bubble_center,
            **_plan_sounding(selection, sounding),
        )
    except CM1Refusal as refusal:
        if refusal.detail.code != "CM1-CARTESIAN-SOUNDING-UNREADABLE":
            raise
        plan_geometry = {
            "status": "not-evaluated-no-readable-profile",
            "reason": refusal.detail.reason,
            "bubble_center": bubble_center_record(config, bubble_center),
        }
    return {
        "schema": "gpuwm-cm1.cartesian-plan/v1",
        "status": "priced-no-device-opened",
        "config_sha256": config.sha256,
        "admitted_level_bounds": list(require_level_capacity(config)),
        "capacity_profile": selected_capacity_profile(config),
        "measured_steps": measured_steps(config),
        "warmup_steps": config.runtime.warmup_steps,
        "initialization": selection.record(),
        "initialization_declared": selection_declared,
        "moving_frame": moving_frame_record(moving_frame),
        "vertical_operator_family": cartesian_vertical_operator_record(grid),
        # Every capability is selectable, and anything this
        # run selected that nothing has measured is named here rather than
        # left to look validated by silence -- the same rows a run's receipt
        # lists, the flagged initialization selectors included.
        "unvalidated": unvalidated_receipt(unvalidated_selected_for_run(
            config, initialization=unvalidated_initialization(selection, config),
        )),
        "bubble_center": bubble_center_record(config, bubble_center),
        "run_geometry": plan_geometry,
        "history_frame": cartesian_history_plan(runtime_config),
        "integration_cadence": None if cadence is None else cadence.record(),
        # The restart plan is priced without a directory: --plan opens no
        # device and creates no path, so it reports the selectors the document
        # declares and the cadence they would fire on, not a resolved tool.
        "ensemble": require_ensemble(config),
        "restart_selectors": {
            "authority": (
                "run/namelist.input:30 rstfrq; README.namelist:227-232 "
                "irst/rstnum; README.namelist:1488-1510 restart_reset_frqtim"
            ),
            "interval_s": float(config.run.restart_interval_s),
            # CM1's restart selectors live in [run] here, beside the cadence
            # they are read with, so they are reported from there.
            "irst": config.run.irst or 0,
            "restart_reset_frqtim": config.run.reset_frqtim,
            "rstnum": config.run.rstnum or FIRST_RESTART_NUMBER,
            "steps_per_restart": int(
                round(config.run.restart_interval_s / config.run.dt_s)
            ),
            "declared": config.run.irst is not None or config.run.rstnum is not None,
        },
        **allocation_plan(runtime_config),
        **nest_allocation_plan(require_nest(config)),
    }


def run_cartesian(
    config: CM1Config,
    *,
    config_label: str,
    output: Path,
    library: Path | None = None,
    sounding: Path | None = None,
    gpu_uuid: str | None = None,
    lock: Path | None = None,
    plan: bool = False,
    trace_phases: bool = False,
    history_dir: Path | None = None,
    history_publisher_argument: Path | None = None,
    history_carriers: str = COMPLETE_CARRIER_PROFILE,
    restart_dir: Path | None = None,
    restart_from: Path | None = None,
    integrate: bool = False,
    emit_json: bool = False,
    profile: object = "summary",
    progress_interval: object = 30.0,
    event_parts: object = "beside",
    cuda_graphs: object = "off",
    events: Path | None = None,
    control_dir: Path | None = None,
    history_envelope: object = "on",
    history_initial: bool = False,
) -> int:
    """Admit, drive and record one Cartesian run from a parsed configuration.

    ``events`` (``--events``) writes the run's ``gm1.run-events/v1`` log,
    ``control_dir`` (``--control-dir``) is read for stop and pause requests
    once per large step (:mod:`cm1core.run_control`), ``history_envelope``
    (``--history-envelope on|off``) decides whether each frame also gets its
    CM1FHS1 envelope, and ``history_initial`` (``--history-initial``)
    publishes the initialized state as frame 0.  Each is off by default, and
    with all four at their defaults the door does exactly what it did before
    they existed.  The first three need ``--integrate``; SIGTERM ends a run
    given either of the first two with an ``end`` record and status 143 at
    any point, not only while it steps.
    """

    envelope_on = require_history_envelope(history_envelope)
    if not isinstance(history_initial, bool):
        raise TypeError("history_initial must be a bool")
    for flag, value in (
        ("events", events), ("control-dir", control_dir),
        ("history-initial", history_initial or None),
    ):
        if value is None:
            continue
        if plan:
            refuse(
                code="CM1-CARTESIAN-RUN-FLAG",
                gate=CARTESIAN_GATE,
                reason=(
                    f"--{flag} belongs to a run, and --plan opens no device and runs "
                    "nothing; the plan would be read as a run that did not happen"
                ),
                remedy=f"drop --{flag} from the --plan call, or drop --plan",
                option=flag,
                value=str(value),
            )
        if not integrate:
            refuse(
                code="CM1-CARTESIAN-RUN-FLAG",
                gate=CARTESIAN_GATE,
                reason=(
                    f"--{flag} applies to an integration (its frames, its steps, its "
                    "restarts); without --integrate this door runs the timing "
                    "harness or builds an initial condition, which it would silently "
                    "not apply to"
                ),
                remedy=f"pass --integrate, or drop --{flag}",
                option=flag,
                value=str(value),
            )
    log = None if events is None else open_run_events(Path(events))
    reporting = _RunReporting(log)
    terminating = nullcontext()
    if log is not None or control_dir is not None:
        from .background_publication import termination_as_exception

        # SIGTERM as TerminationRequested for the whole call, so a run stopped
        # while it compiles or waits for the GPU still writes its end record
        # and exits 143 (the integration installs the same handler itself).
        terminating = termination_as_exception()
    try:
        with terminating:
            rc = _run_cartesian(
                config, config_label=config_label, output=output, library=library,
                sounding=sounding, gpu_uuid=gpu_uuid, lock=lock, plan=plan,
                trace_phases=trace_phases, history_dir=history_dir,
                history_publisher_argument=history_publisher_argument,
                history_carriers=history_carriers, restart_dir=restart_dir,
                restart_from=restart_from, integrate=integrate, emit_json=emit_json,
                profile=profile, progress_interval=progress_interval,
                event_parts=event_parts, cuda_graphs=cuda_graphs,
                reporting=reporting, control_dir=control_dir,
                envelope_on=envelope_on, history_initial=history_initial,
                events_path=None if events is None else Path(events),
            )
    except BaseException as error:
        reporting.end(error)
        raise
    else:
        reporting.end(None, rc)
        return rc
    finally:
        if log is not None:
            log.close()


def _run_cartesian(
    config: CM1Config,
    *,
    config_label: str,
    output: Path,
    library: Path | None = None,
    sounding: Path | None = None,
    gpu_uuid: str | None = None,
    lock: Path | None = None,
    plan: bool = False,
    trace_phases: bool = False,
    history_dir: Path | None = None,
    history_publisher_argument: Path | None = None,
    history_carriers: str = COMPLETE_CARRIER_PROFILE,
    restart_dir: Path | None = None,
    restart_from: Path | None = None,
    integrate: bool = False,
    emit_json: bool = False,
    profile: object = "summary",
    progress_interval: object = 30.0,
    event_parts: object = "beside",
    cuda_graphs: object = "off",
    reporting: _RunReporting,
    control_dir: Path | None,
    envelope_on: bool,
    history_initial: bool,
    events_path: Path | None,
) -> int:
    require_cartesian_lane(config)
    require_listed_component(config)
    # Execution controls with no scientific meaning, refused by value before
    # anything is resolved: they decide what is measured, or how an event's
    # bytes reach the disk, never a word.
    profile = require_profile_level(profile)
    progress_interval_s = require_progress_interval(progress_interval)
    concurrent_event_parts = require_event_parts(event_parts)
    cuda_graph_mode = require_cuda_graph_mode(cuda_graphs, config)
    # Do not resolve/create/touch the output above this line.
    # A nested document is refused here only for what the live nest does not
    # execute (require_nest_execution names each).  The checks need no engine,
    # no device and no disk, so they run before the engine estate is inspected
    # and before anything is resolved, created or touched.
    if config.nest is not None and not plan:
        require_nest_execution(
            config, initial_condition_only=bool(config.run.initial_condition_only),
        )
    if config.chain is not None and not plan:
        from .arwen_chain import require_chain_execution

        require_chain_execution(
            config, restart_dir=restart_dir,
            initial_condition_only=bool(config.run.initial_condition_only),
        )
    # README.namelist:233-236: iconly=1 builds the initial condition and stops,
    # so this door compiles no kernel, opens no device and reads no ArWen
    # symbol.  Admitting it against the device-physics estate would refuse a
    # host-only inspection for a reason that cannot apply to it.
    # With --restart-dir iconly runs on the device (below, before the host
    # door), which builds the microphysics finish and so needs the estate.
    iconly = config.run.initial_condition_only and restart_dir is None
    require_runnable(config, device_physics=not iconly)
    level_bounds = require_level_capacity(config)
    runtime_config = build_runtime_config(config)
    grid = runtime_config.grid
    steps = measured_steps(config)
    capacity_profile = selected_capacity_profile(config)
    selection, selection_declared = require_initializer_selection(config)
    moving_frame = require_moving_frame(config)
    # README.namelist:1789-1797.  With closures.do_adapt_move declared the
    # frame velocity stops being a constant and becomes the controller's own
    # state; this binds the sealed operator or refuses the grid/boundary
    # combination it was never validated on, before a device exists.
    adaptive_frame_binding = require_adaptive_frame(config)
    # param.F:7212-7213's frame, then init3d.F:456-457's centre inside it.
    # ``bubble_center_m`` is the iorigin-aware authority for both, so the
    # value the run_geometry track is evaluated against is the same one the
    # initializer is handed.
    origin_xy_m = horizontal_origin_m(config)
    bubble_center_xy_m = bubble_center_m(config)
    bubble_center = bubble_center_xy_m
    unvalidated = unvalidated_initialization(selection, config)
    # cm1.F:2660 and cm1.F:4341: which of the three things this run is.  Both
    # are decided from the configuration and the filesystem, before a device
    # exists, so an unreachable restart or an impossible schedule refuses at
    # once instead of after the initializer has run.
    initial_condition_only = config.run.initial_condition_only
    resume = resume_selection(config, restart_from)
    resume_directory = None if resume is None else resume[0]
    resume_receipt = None if resume is None else resume[1]
    if initial_condition_only and integrate:
        refuse(
            code="CM1-CARTESIAN-INITIAL-CONDITION",
            gate=CARTESIAN_GATE,
            reason="cm1.F:4341-4350 stops the model before it integrates, and --integrate asks it to step to run.duration_s; running both would spend the GPU hours the caller asked to avoid",
            remedy="drop --integrate, or set run.iconly = 0",
            option="run.iconly",
            value=config.run.iconly,
        )
    if initial_condition_only and restart_dir is None and history_dir is None:
        refuse(
            code="CM1-CARTESIAN-INITIAL-CONDITION",
            gate=CARTESIAN_GATE,
            reason="run.iconly = 1 builds the initial condition and stops, and with neither a restart nor a history directory there is nowhere to write it; the run would initialize the atmosphere and discard it",
            remedy="pass --restart-dir with a new directory to receive the nstep=0 restart (built on the device), or --history-dir alone to inspect the model-time-zero frame on the host",
            option="restart-dir",
            value=None,
        )
    if restart_dir is not None and not (integrate or initial_condition_only):
        refuse(
            code="CM1-CARTESIAN-RESTART-DIRECTORY",
            gate=CARTESIAN_GATE,
            reason="the timing harness runs a fixed number of steps and writes no restart, so a restart directory declared here would be created and left empty",
            remedy="pass --integrate to run run.duration_s with restarts at run.restart_interval_s, or set run.iconly = 1, or drop --restart-dir",
            option="restart-dir",
            value=str(restart_dir),
        )
    # The nest is bound on the host, before a device is opened: a child whose
    # geometry, clock or acoustic cadence CM1 would refuse must stop here,
    # not after the parent has been allocated.
    nest = require_nest(config)
    if (
        nest is not None and nest.tornado_criterion is not None
        and not integrate and not initial_condition_only and not plan
    ):
        # cm1core.nest_tornado_criterion samples every parent step and
        # publishes every sample to <history>/nest-child/tornado-criterion.jsonl,
        # which only the integration opens.  The timing harness would sample
        # its warm-up and measured steps and publish none of them.
        refuse(
            code="CM1-CARTESIAN-NEST-TORNADO-CRITERION-INTEGRATION",
            gate=CARTESIAN_GATE,
            reason=(
                "[nest.tornado_criterion] is sampled every parent step and every sample is "
                "published to the integration's nest-child/tornado-criterion.jsonl; the timing "
                "harness opens no such file, so its receipt would carry a verdict with no row "
                "to replay it from"
            ),
            remedy=(
                "pass --integrate (with --history-dir) to sample the criterion, or drop "
                "[nest.tornado_criterion] to time the nest"
            ),
            option="nest.tornado_criterion",
            value=nest.tornado_criterion.record(),
        )
    # The integration entry point needs a frame root and an admissible frame
    # schedule.  Both are proved here, on the host, before anything else: a
    # duration or interval the large-step clock cannot place must refuse
    # before a device is opened, not after two hours of stepping.
    cadence = None
    schedule = None
    if integrate:
        schedule = require_run_schedule(
            config, restart=resume_receipt, write_restarts=restart_dir is not None
        )
        cadence = schedule.cadence
        if history_dir is None:
            refuse(
                code="CM1-CARTESIAN-HISTORY-DIRECTORY",
                gate=CARTESIAN_GATE,
                reason="an integration with no history directory would step to the requested duration and publish none of the model states it computed",
                remedy="pass --history-dir with a new directory to receive one frame per run.history_interval_s",
                option="history-dir",
                value=None,
            )
    elif resume is not None:
        refuse(
            code="CM1-CARTESIAN-RESTART-SELECTION",
            gate=CARTESIAN_GATE,
            reason="run.irst = 1 continues a previous integration and the timing harness runs a fixed number of steps from whatever state it is handed; the restored hours would be measured as a benchmark and then thrown away",
            remedy="pass --integrate to continue the run to run.duration_s",
            option="run.irst",
            value=config.run.irst,
        )
    if history_initial and resume is not None:
        refuse(
            code="CM1-CARTESIAN-RUN-FLAG",
            gate=CARTESIAN_GATE,
            reason=(
                "--history-initial publishes the initialized state as frame 0, and "
                "run.irst = 1 continues from a restart: this run never holds an "
                "initialized state, and frame 0 belongs to the run that started it"
            ),
            remedy="drop --history-initial when continuing from a restart",
            option="history-initial",
            value=True,
        )
    if not envelope_on and history_dir is None:
        refuse(
            code="CM1-CARTESIAN-HISTORY-ENVELOPE",
            gate=CARTESIAN_GATE,
            reason=(
                "--history-envelope off names what each published frame carries, but "
                "this run publishes no history; the receipt would record a frame "
                "format no frame of the run has"
            ),
            remedy="pass --history-dir, or drop --history-envelope",
            option="history-envelope",
            value="off",
        )
    history_dir = require_new_history_root(history_dir, option="history-dir")
    history_carriers = require_cartesian_history_carriers(history_carriers)
    if history_dir is None and history_carriers != COMPLETE_CARRIER_PROFILE:
        # Symmetric with the publisher-without-a-directory refusal above: a
        # run that publishes nothing must not record a carrier inventory.
        refuse(
            code="CM1-CARTESIAN-HISTORY-CARRIERS-UNPUBLISHED",
            gate=CARTESIAN_GATE,
            reason=(
                f"--history-carriers {history_carriers} names the carrier inventory each published frame carries, but this run publishes no history; the receipt would record an inventory no frame in the run carries"
            ),
            remedy="pass --history-dir with a new directory to publish frames under this inventory, or drop --history-carriers",
            option="history-carriers",
            value=history_carriers,
        )
    restart_dir = require_new_history_root(
        restart_dir, option="restart-dir", kind="restart",
        code="CM1-CARTESIAN-RESTART-DIRECTORY",
        collision=(
            "a later resume selects a restart by number, and a directory holding "
            "two runs' restarts would hand it a state from the wrong trajectory"
        ),
    )
    ensemble = require_ensemble(config)
    control = None
    if control_dir is not None or reporting.events is not None:
        from .run_control import RunControl, require_control_directory

        if control_dir is not None:
            try:
                control_dir = require_control_directory(control_dir)
            except (OSError, ValueError) as error:
                refuse(
                    code="CM1-CARTESIAN-CONTROL-DIRECTORY",
                    gate=CARTESIAN_GATE,
                    reason=f"the run-control directory cannot be used: {error}",
                    remedy=(
                        "pass --control-dir with a directory that holds no stop or "
                        "pause file (it is created if it does not exist)"
                    ),
                    option="control-dir",
                    value=str(control_dir),
                )
        control = RunControl(control_dir, reporting.events)
        reporting.control = control
    if plan:
        # Price the geometry too, and by the same rules.  A plan whose grid
        # cannot hold its own storm is not a plan worth executing, and this
        # is the cheapest moment anyone will ever learn it.  A plan may be
        # taken without a resolvable profile, so the failure to read one is
        # reported here rather than refused; the run path below refuses.
        try:
            plan_geometry: dict[str, object] = require_run_geometry(
                config,
                grid,
                selection=selection,
                moving_frame=moving_frame,
                center_xy_m=bubble_center,
                adaptive_frame=adaptive_frame_binding,
                **_plan_sounding(selection, sounding),
            )
        except CM1Refusal as refusal:
            if refusal.detail.code != "CM1-CARTESIAN-SOUNDING-UNREADABLE":
                raise
            plan_geometry = {
                "status": "not-evaluated-no-readable-profile",
                "reason": refusal.detail.reason,
                "bubble_center": bubble_center_record(config, bubble_center),
            }
        payload = {
            "schema": "gpuwm-cm1.cartesian-plan/v1",
            "status": "priced-no-device-opened",
            "config_sha256": config.sha256,
            "admitted_level_bounds": list(level_bounds),
            "capacity_profile": capacity_profile,
            "measured_steps": steps,
            "warmup_steps": config.runtime.warmup_steps,
            "initialization": selection.record(),
            "initialization_declared": selection_declared,
            "initialization_origin_xy_m": list(origin_xy_m),
            "initialization_bubble_center_xy_m": list(bubble_center_xy_m),
            "initialization_validation": unvalidated,
            "moving_frame": moving_frame_record(moving_frame),
            "frame_mode": cartesian_frame_mode_record(moving_frame, adaptive_frame_binding),
            "vertical_operator_family": cartesian_vertical_operator_record(runtime_config.grid),
            # Every capability is selectable, and anything
            # this run selected that nothing has measured is named here rather
            # than left to look validated by silence -- the capability rows,
            # the column component, the [dynamics] words at the values this
            # run gives them, and the initialization selectors, in one list.
            "unvalidated": unvalidated_receipt(
                unvalidated_selected_for_run(config, initialization=unvalidated)
            ),
            "bubble_center": bubble_center_record(config, bubble_center),
            "run_geometry": plan_geometry,
            "history_frame": cartesian_history_plan(runtime_config),
            # CM1's lateral family by CM1's own names, and -- loudly -- every
            # boundary capability this run turns on for which no CM1-vs-GM1
            # differential has been executed.
            "lateral_boundaries": runtime_config.lateral_boundary_selection.record(),
            "boundary_validation": boundary_unvalidated_for(config.boundaries, config.closures),
            "integration_cadence": None if cadence is None else cadence.record(),
            "run_schedule": None if schedule is None else schedule.record(),
            "restart": _restart_plan(config, restart_dir, resume_directory, resume_receipt),
            "initial_condition_only": initial_condition_only,
            "history_directory": None if history_dir is None else str(history_dir),
            "history_carriers": None if history_dir is None else history_carriers,
            # Off, a frame is its raw arena and tables only; on, it also holds
            # history.cm1fhs, a second copy of history_frame.payload_bytes.
            "history_envelope": "on" if envelope_on else "off",
            **allocation_plan(runtime_config),
        }
        # int/lifecycle factored the plan body into cartesian_plan_payload and
        # added the nest, ensemble and restart-selector blocks there.  One plan
        # is one JSON document, so its keys are folded in rather than printed
        # as a second object after this one: two objects on stdout are not
        # parseable as a plan at all.  This dict is the authority for the keys
        # both produce, because it is the one built from the resolved run.
        payload = {**cartesian_plan_payload(config, cadence=cadence), **payload}
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0

    library = resolve_initializer_library(library)
    if selection.reads_input_sounding:
        sounding_label, sounding_bytes = resolve_sounding(sounding)
        sounding_interval = require_sounding_extent(sounding_label, sounding_bytes, grid)
    else:
        # base.F:466 and base.F:363-470: the analytic sounding lane computes
        # its own profile and reads no file.  Admitting one here would record
        # a profile identity in the receipt that the run never used.
        if sounding is not None:
            refuse(
                code="CM1-CARTESIAN-SOUNDING-UNREAD",
                gate=CARTESIAN_GATE,
                reason=f"initialization.isnd={selection.isnd} computes its base state analytically and never reads an input_sounding; the supplied profile would be recorded in the receipt as this run's thermodynamic identity and never used",
                remedy="omit --sounding for this initializer lane, or declare initialization.isnd = 7 to run the single-sounding lane",
                option="sounding",
                value=str(sounding),
            )
        sounding_label, sounding_bytes, sounding_interval = None, b"", None
    # The ArWen chain's prepared products are opened and hash-checked against
    # this document and its sounding on the host, before a device is opened.
    from .arwen_chain import require_chain

    chain = require_chain(config, sounding_bytes=sounding_bytes)
    # The geometry admission goes here, after the profile is in hand and
    # before anything is created or a device is opened.
    run_geometry = require_run_geometry(
        config,
        grid,
        selection=selection,
        moving_frame=moving_frame,
        center_xy_m=bubble_center,
        sounding_label=sounding_label,
        sounding_bytes=sounding_bytes,
        adaptive_frame=adaptive_frame_binding,
    )
    # base.F:3143-3189.  CM1 prints the thermodynamic properties of its own
    # base-state sounding for every moist run, before it integrates anything.
    # This is that block, plus the three kinematic parameters CM1 has no
    # equivalent for, and it runs here -- on the host, before a device is
    # opened -- so a run that cannot describe the atmosphere it was handed
    # stops before it burns a GPU rather than after.
    sounding_report = require_sounding_report(
        config, library=library, sounding_bytes=sounding_bytes, selection=selection
    )
    print(format_sounding_report(sounding_report), file=sys.stderr)
    output = require_new_receipt_path(output)
    # Two iconly doors.  With --restart-dir the caller needs CM1's nstep=0
    # restart (cm1.F:3110-3115 / :4341-4351), which only the device runtime
    # writes -- and its frame materializes every &param9 product from the live
    # accumulators.  The host door writes a frame and nothing else, so routing
    # a restart request to it dropped the restart and refused every product.
    if config.run.initial_condition_only and restart_dir is None:
        return _write_initial_condition(
            config=config,
            config_label=config_label,
            output=output,
            library=library,
            sounding_label=sounding_label,
            sounding_bytes=sounding_bytes,
            sounding_interval=sounding_interval,
            selection=selection,
            selection_declared=selection_declared,
            moving_frame=moving_frame,
            frame_mode=cartesian_frame_mode_record(
                moving_frame, adaptive_frame_binding
            ),
            origin_xy_m=origin_xy_m,
            bubble_center_xy_m=bubble_center_xy_m,
            grid=grid,
            runtime_config=runtime_config,
            level_bounds=level_bounds,
            capacity_profile=capacity_profile,
            history_dir=history_dir,
            unvalidated=unvalidated,
            emit_json=emit_json,
        )
    # Resolve the publisher before the device opens: a run that cannot publish
    # what the caller asked for must stop before it burns a GPU, not after.
    # A restart directory needs it as much as a history directory does -- more,
    # because a restart has no publisher-less form -- and so does a resume,
    # which reads its envelope back through the same executable.
    # With --history-envelope off no frame is sealed, so the publisher is
    # needed only to write or read restarts.
    history_publisher = resolve_history_publisher(
        (history_dir if envelope_on else None) or restart_dir or resume_directory,
        history_publisher_argument,
    )
    frame_publisher = history_publisher if envelope_on else None
    cupy = require_cuda_runtime()
    if schedule is not None:
        from .run_events import host_name

        grid_record = runtime_config.grid
        reporting.admit(
            config=config_label,
            config_sha256=config.sha256,
            grid={
                "nx": int(grid_record.nx), "ny": int(grid_record.ny), "nz": int(grid_record.nz),
                "dx_m": float(grid_record.dx_m), "dy_m": float(grid_record.dy_m),
                "dz_m": (
                    float(grid_record.dz_m) if int(runtime_config.grid.stretch_z) == 0
                    else None
                ),
                "stretch_z": int(runtime_config.grid.stretch_z),
                "ztop_m": float(runtime_config.grid.ztop_m),
            },
            microphysics=str(runtime_config.microphysics),
            steps=None if runtime_config.adaptive_timestep else int(schedule.final_step),
            start_step=int(schedule.start_step),
            dt_s=float(config.run.dt_s),
            nsound=int(runtime_config.acoustic_substeps),
            adaptive_timestep=bool(runtime_config.adaptive_timestep),
            duration_s=float(schedule.cadence.duration_s),
            history_interval_s=float(schedule.cadence.interval_s),
            frames_expected=int(schedule.history_count) + int(bool(history_initial)),
            restart_interval_s=(
                None if restart_dir is None else config.run.restart_interval_s
            ),
            restarts_expected=int(schedule.restart_count),
            history_directory=str(history_dir),
            restart_directory=None if restart_dir is None else str(restart_dir),
            resumed_from=None if resume_directory is None else str(resume_directory),
            receipt=str(output),
            control_directory=None if control_dir is None else str(control_dir),
            history_envelope="on" if envelope_on else "off",
            history_initial=bool(history_initial),
            host=host_name(),
            gpu_uuid_requested=gpu_uuid,
        )
    # A forcing arm centred outside the domain, or touching no grid
    # point, runs exactly as CM1 runs it -- and says so where the user looks,
    # before the first step (a resumed run has no forcing: param.F:891-903).
    reach = initiation_forcing_reach(config) if resume_directory is None else None
    if reach is not None and reach.get("warning"):
        print(f"warning: {reach['warning']}", file=sys.stderr, flush=True)
        reach_events = None if reporting is None else reporting.events
        if reach_events is not None:
            reach_events.emit("warning", code=reach["code"], message=reach["warning"])

    # Imported here, not at module scope, for the same reason CuPy is: the
    # runtime module pulls the whole device stack behind it.
    from .cuda_cartesian_runtime import CartesianCourantStop

    started = time.perf_counter()
    courant_stop: dict[str, object] | None = None
    try:
        benchmark, token = _execute(
            cupy=cupy,
            library=library,
            sounding_bytes=sounding_bytes,
            selection=selection,
            moving_frame=moving_frame,
            adaptive_frame_binding=adaptive_frame_binding,
            origin_xy_m=origin_xy_m,
            bubble_center_xy_m=bubble_center_xy_m,
            grid=grid,
            runtime_config=runtime_config,
            capacity_profile=capacity_profile,
            warmup_steps=config.runtime.warmup_steps,
            steps=steps,
            trace_phases=bool(trace_phases),
            history_dir=history_dir,
            history_publisher=history_publisher,
            history_carrier_profile=history_carriers,
            # The schedule carries the cadence (``RunSchedule.cadence``) and
            # ``_execute`` reads it from there.  Passing the cadence a second
            # time under its own name is what made every --integrate run die
            # with a TypeError the moment it opened the device.
            schedule=schedule,
            restart_dir=restart_dir,
            restart_interval_s=(
                None if restart_dir is None or initial_condition_only
                else config.run.restart_interval_s
            ),
            reset_frqtim=config.run.reset_frqtim,
            resume_from=resume_directory,
            # The envelope is decoded into a new directory beside this run's
            # own receipt rather than inside the restart being read, so a
            # restart can be resumed from more than once and a read-only
            # archive of restarts still resumes.
            resume_scratch=(
                None if resume_directory is None
                else output.parent / f".{output.name}.resume-source"
            ),
            initial_condition_only=initial_condition_only,
            gpu_uuid=gpu_uuid,
            lock=lock,
            identity_sha256=config.restart_identity_sha256,
            config_sha256=config.sha256,
            ensemble=ensemble,
            nest=nest,
            chain=chain,
            profile=profile,
            progress_interval_s=progress_interval_s,
            concurrent_event_parts=concurrent_event_parts,
            cuda_graphs=cuda_graph_mode,
            frame_publisher=frame_publisher,
            history_initial=history_initial,
            control=control,
            reporting=reporting,
        )
    except CM1Refusal:
        raise
    except CartesianCourantStop as stop:
        # CM1's own advective Courant stop, and the one failure whose record is
        # written rather than discarded.  misclibs.F:2714 sets ``stopit`` and
        # cm1.F:3517-3520 forces ``dostat``/``dowriteout`` precisely so the
        # statistics and history output for the state that tripped it survive;
        # cm1.F:4385-4392 only then stops the model.  The frames are already
        # published, so the receipt that names the step, the model time and the
        # frame they are in is written with them -- under a status of its own,
        # never the completed-run status, and followed by a refusal so the exit
        # status can never read as success.
        benchmark = stop.receipt
        token = getattr(stop, "gpu_mutex_owner", None)
        # A run that ends at CM1's stop still has to say where it came from and
        # which member it was; these are the two records a reader needs to
        # reproduce the leg that diverged.
        resume = getattr(stop, "restart_resume", None)
        if resume is not None:
            benchmark["restart_resume"] = resume
        branch = getattr(stop, "ensemble_branch", None)
        if branch is not None:
            benchmark["ensemble_branch"] = branch
        courant_stop = dict(stop.receipt["courant_stop"])
    except Exception as error:
        # The CUDA runtime, the compiler and the component each raise their own
        # exception types.  Every one of them is named here rather than
        # printed as a traceback, and no receipt is written for a run that
        # stopped: a partial trajectory must never read as a completed one.
        refuse(
            code="CM1-CARTESIAN-RUN-FAILED",
            gate=CARTESIAN_GATE,
            reason=f"the Cartesian runtime stopped during execution ({type(error).__name__}): {error}",
            remedy="preserve the console output, reduce the grid or the device budget, and re-run; a partial trajectory is never published as a receipt",
            option="output",
            value=str(output),
        )
    if initial_condition_only:
        schema = CARTESIAN_INITIAL_CONDITION_RECEIPT_SCHEMA
        status = "published-cartesian-initial-condition-without-integrating"
    elif cadence is None:
        schema = CARTESIAN_RECEIPT_SCHEMA
        status = "executed-cartesian-large-steps"
    else:
        schema = CARTESIAN_INTEGRATION_RECEIPT_SCHEMA
        status = (
            "executed-cartesian-integration-to-requested-duration"
            if resume is None
            else "resumed-and-executed-cartesian-integration-to-requested-duration"
        )
    # A declared [nest.tornado_criterion] stop rule ended the integration on
    # purpose (cm1core.nest_tornado_criterion).  The integration receipt names
    # it only when the nest declares the criterion, and so does this one, so
    # every other receipt keeps exactly the keys it had.
    criterion_stop = (
        benchmark.get("criterion_stop") if isinstance(benchmark, dict) else None
    )
    control_stop = (
        benchmark.get("control_stop") if isinstance(benchmark, dict) else None
    )
    if courant_stop is not None:
        # CM1's advective Courant stop outranks every other outcome this run
        # could report: whatever was asked for, the run ended early.
        status = "stopped-at-cm1-advective-courant-stop"
    elif criterion_stop is not None:
        status = "stopped-at-nest-tornado-criterion-stop-rule"
        # Not "completed": the requested duration was not integrated.  No
        # control file asked for it, so the end record's request stays null
        # and its message names the rule.
        reporting.reason = "stopped"
        reporting.message = (
            f"stopped by nest.tornado_criterion.{criterion_stop['rule']} at "
            f"{criterion_stop['model_time_s']} model seconds"
        )
    elif control_stop is not None:
        status = (
            CONTROL_STOP_STATUS if control_stop["request"] == "stop"
            else CONTROL_PAUSE_STATUS
        )
        reporting.reason = "stopped" if control_stop["request"] == "stop" else "paused"
        reporting.request = control_stop["request"]
    receipt = {
        "schema": schema,
        "status": status,
        "courant_stop": courant_stop,
        **(
            {"criterion_stop": criterion_stop}
            if isinstance(benchmark, dict) and "criterion_stop" in benchmark else {}
        ),
        "authority_sha256": CM1_AUTHORITY_SHA256,
        "cm1_release": CM1_RELEASE,
        "distribution": "gpuwm-cm1",
        "version": __version__,
        "maturity": MATURITY,
        "lane": config.run.lane.value,
        "config": config_label,
        "config_sha256": config.sha256,
        "admitted_level_bounds": list(level_bounds),
        "capacity_profile": capacity_profile,
        # Which compiled family every large step actually ran, chosen by
        # grid.stretch_z alone.  Without it a reader cannot tell a stretched
        # run that multiplied by rdz*mh(1,1,k) from one that used a scalar dz.
        "vertical_coordinate": runtime_config.grid.levels.record(),
        "operator_family": cartesian_operator_family(runtime_config.grid.stretch_z),
        "measured_steps": steps,
        "warmup_steps": config.runtime.warmup_steps,
        "initialization": selection.record(),
        "initialization_declared": selection_declared,
        "initialization_origin_xy_m": list(origin_xy_m),
        "initialization_bubble_center_xy_m": list(bubble_center_xy_m),
        "initialization_validation": unvalidated,
        "moving_frame": moving_frame_record(moving_frame),
        "frame_mode": cartesian_frame_mode_record(moving_frame, adaptive_frame_binding),
        "vertical_operator_family": cartesian_vertical_operator_record(runtime_config.grid),
        # Every capability is selectable, and anything this
        # run selected that nothing has measured is named here rather than
        # left to look validated by silence -- the capability rows, the column
        # component, the [dynamics] words at the values this run gives them,
        # and the initialization selectors, in one list a reader sees first.
        "unvalidated": unvalidated_receipt(
            unvalidated_selected_for_run(config, initialization=unvalidated)
        ),
        "bubble_center": bubble_center_record(config, bubble_center),
        "run_geometry": run_geometry,
        # Which of CM1's param2/param3 dynamical selectors this run used, and
        # what each one rests on.  At the top level, beside the lane and the
        # configuration hash, because a reader who opens one receipt must be
        # able to separate a measured number from an unmeasured one without
        # going looking for it.
        "dynamics_validation": dynamics_validation_record(config.dynamics),
        # CM1's lateral family by CM1's own names, and -- loudly -- every
        # boundary capability this run turned on for which no CM1-vs-GM1
        # differential has been executed.  A reader never has to infer from a
        # lane name which half of a boundary was measured.
        "lateral_boundaries": runtime_config.lateral_boundary_selection.record(),
        "boundary_validation": boundary_unvalidated_for(config.boundaries, config.closures),
        # ``None`` for every single-domain run, so a receipt written before
        # nesting existed and one written now for the same document differ in
        # exactly one null-valued key and in nothing else.
        "nest": None if nest is None else nest.record(parent_dt_s=config.run.dt_s),
        "integration_cadence": None if cadence is None else cadence.record(),
        "run_schedule": None if schedule is None else schedule.record(),
        "restart": _restart_plan(config, restart_dir, resume_directory, resume_receipt),
        "initial_condition_only": initial_condition_only,
        "initializer_library": str(library),
        "initializer_library_sha256": _sha256(library.read_bytes()),
        "sounding": sounding_label,
        "sounding_sha256": None if sounding_label is None else _sha256(sounding_bytes),
        "sounding_interval_m": None if sounding_interval is None else list(sounding_interval),
        # base.F:3143-3189 over getcape.F:14-594, plus Bunkers et al. (2000).
        # A receipt that does not say what atmosphere was integrated cannot be
        # read after the fact; this block is that statement.
        "initial_sounding_parameters": sounding_report,
        "gpu_mutex_owner": token,
        "output": str(output),
        "process_wall_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "history_directory": None if history_dir is None else str(history_dir),
        "history_layout": (
            None if cadence is None
            else "one complete frame directory per run.history_interval_s, named frame-NNNNNN and renamed into the history directory only once written"
        ),
        "history_publisher": None if history_publisher is None else str(history_publisher),
        # Null when the run publishes nothing: a receipt must not record a
        # carrier inventory no frame of the run carries.
        "history_carriers": None if history_dir is None else history_carriers,
        "history_envelope": None if history_dir is None else ("on" if envelope_on else "off"),
        "history_initial": bool(history_initial),
        "events": None if events_path is None else str(events_path),
        "control_directory": None if control_dir is None else str(control_dir),
        "restart_directory": None if restart_dir is None else str(restart_dir),
        "restart_layout": (
            None if restart_dir is None
            else "one complete restart directory per run.restart_interval_s, named restart-NNNNNN and renamed into the restart directory only once written"
        ),
        "runtime_benchmark": benchmark,
        # What this run measured about itself.  The per-phase, I/O and startup
        # tables are runtime_benchmark.performance; `gpuwm-cm1 perf` prints them.
        "profiling": {
            "level": profile,
            "progress_interval_s": progress_interval_s,
            "summary_command": f"gpuwm-cm1 perf {output}",
        },
        "claim_limit": (
            "abandoned at CM1's advective Courant stop after publishing the "
            "output that reached it; the requested model duration was not "
            "integrated and nothing in this record is a completed run"
            if courant_stop is not None else
            "ended by the declared nest.tornado_criterion stop rule after "
            "publishing the frame of the event that followed it; the requested "
            "model duration was not integrated, by design, and the criterion's "
            "verdict covers the integrated interval only"
            if criterion_stop is not None else
            "ended at a run-control request after publishing every event taken "
            "before it; the requested model duration was not integrated"
            + (
                "" if control_stop["request"] != "pause"
                else f", and restart {control_stop['restart_number']} continues it"
            )
            if control_stop is not None else
            "executed large steps and measured resource use on one admitted "
            "grid; not native-CM1 trajectory identity, restart identity, or "
            "forecast skill"
        ),
    }
    try:
        document = json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False, default=dict
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CARTESIAN-RECEIPT",
            gate=CARTESIAN_GATE,
            reason=f"the completed run produced a receipt that is not canonical JSON: {error}",
            remedy="preserve the console output and report the runtime receipt defect; an uncanonical receipt is never written",
            option="output",
            value=str(output),
        )
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(document + "\n")
        reporting.receipt = str(output)
        learn_step_rate(receipt, output)
    except OSError as error:
        refuse(
            code="CM1-CARTESIAN-OUTPUT",
            gate=CARTESIAN_GATE,
            reason=f"the completed run receipt could not be published without overwriting: {error}",
            remedy="re-run against a new receipt path on a writable filesystem; the measured run is lost, never overwritten",
            option="output",
            value=str(output),
        )
    if courant_stop is not None:
        if emit_json:
            print(document)
        refuse(
            code="CM1-CARTESIAN-COURANT-STOP",
            gate=CARTESIAN_GATE,
            reason=(
                f"{courant_stop['reason']}; CM1 abandons a run there "
                "(misclibs.F:2714, cm1.F:4385-4392) rather than integrate a "
                "diverging wind field to the requested duration"
            ),
            remedy=(
                f"read frame {courant_stop['frame_index']} in "
                f"{courant_stop['frame_directory']} and the receipt {output}, "
                "which are written for exactly this purpose, then lower "
                "run.dt_s or coarsen grid.dx_m/dy_m/dz_m so the same wind "
                "stays below a Courant number of 1.50 and re-run against a new "
                "history directory"
            ),
            option="run.dt_s",
            value=config.run.dt_s,
        )
    if emit_json:
        print(document)
    elif initial_condition_only:
        print(
            "built the initial condition and stopped without integrating "
            f"(run.iconly = 1); restart {benchmark['restart_publication']['directory']}; "
            f"receipt {output}"
        )
    elif cadence is None:
        print(
            f"executed {benchmark['completed_steps']} Cartesian large steps; "
            f"receipt {output}"
        )
    elif control_stop is not None:
        print(
            ("stopped" if control_stop["request"] == "stop" else "paused")
            + f" at the run-control request ({control_stop.get('file')}) after "
            f"{benchmark['executed_steps']} Cartesian large steps at "
            f"{benchmark['model_seconds_at_end']} model seconds; "
            f"{control_stop['frames_published']} of {control_stop['frames_requested']} "
            f"history frames in {history_dir}"
            + (
                "" if control_stop["restart_number"] is None
                else f"; last restart {control_stop['restart_number']} in {restart_dir}"
            )
            + f"; receipt {output}"
        )
    elif criterion_stop is not None:
        print(
            f"stopped by nest.tornado_criterion.{criterion_stop['rule']} after "
            f"{benchmark['executed_steps']} Cartesian large steps at "
            f"{criterion_stop['model_time_s']} model seconds; "
            f"{criterion_stop['frames_published']} of {criterion_stop['frames_requested']} "
            f"history frames in {history_dir}; receipt {output}"
        )
    else:
        print(
            f"integrated {benchmark['executed_steps']} Cartesian large steps to "
            f"{benchmark['model_seconds_at_end']} model seconds; "
            + ("frame 0 and " if history_initial else "")
            + f"{schedule.history_count} history frames in {history_dir}"
            + (
                "" if restart_dir is None
                else f"; {schedule.restart_count} restarts in {restart_dir}"
            )
            + f"; receipt {output}"
        )
    return 0


def require_cartesian_history_carriers(value: str) -> str:
    """Admit one published carrier inventory, refusing any other spelling."""

    try:
        return require_history_carrier_profile(value)
    except ValueError as error:
        refuse(
            code="CM1-CARTESIAN-HISTORY-CARRIERS",
            gate=CARTESIAN_GATE,
            reason=str(error),
            remedy=(
                "pass --history-carriers "
                f"{' or '.join(CARTESIAN_HISTORY_CARRIER_PROFILES)}"
            ),
            option="history-carriers",
            value=value,
        )


def resolve_history_publisher(
    history_dir: Path | None, argument: Path | None = None
) -> Path | None:
    """The wheel-sealed forecast publisher, or a named refusal."""

    if history_dir is None:
        return None
    if argument is not None:
        # Symmetric with --library and --sounding: a source checkout has no
        # installed native-tools.json, so without this a run from a checkout
        # can never publish a frame.
        return _require_regular_file(
            argument, "CM1-CARTESIAN-HISTORY-PUBLISHER-PATH", "history-publisher"
        )
    from . import rust_tools

    try:
        return rust_tools.verify_native_tools()[CARTESIAN_HISTORY_PUBLISHER]
    except (OSError, KeyError, ValueError, rust_tools.NativeToolError) as error:
        refuse(
            code="CM1-CARTESIAN-HISTORY-PUBLISHER-MISSING",
            gate=CARTESIAN_GATE,
            reason=f"history frames were requested but the installed forecast publisher is not admitted: {error}",
            remedy=f"install a platform wheel whose {CARTESIAN_HISTORY_PUBLISHER} passes its manifest, or omit --history-dir",
            option="history-dir",
            value=str(history_dir),
        )


def _restart_plan(
    config: CM1Config, restart_dir: Path | None,
    resume_directory: Path | None, resume_receipt: dict | None,
) -> dict[str, object]:
    """What this run does about restarts, in CM1's own selector names."""

    return {
        "schema": "gpuwm-cm1.cartesian-restart-selection/v1",
        "authority": "CM1 r21.1 cm1.F:2660,3186,4247-4252,4341-4350; README.namelist:103-104,227-235,1488-1510",
        "irst": config.run.irst,
        "rstnum": config.run.rstnum,
        "iconly": config.run.iconly,
        "restart_reset_frqtim": config.run.restart_reset_frqtim,
        "effective_restart_reset_frqtim": config.run.reset_frqtim,
        "restart_interval_s": config.run.restart_interval_s,
        "restart_directory": None if restart_dir is None else str(restart_dir),
        "resumed_from": None if resume_directory is None else str(resume_directory),
        "resumed_step": None if resume_receipt is None else int(resume_receipt["step"]),
        "resumed_model_time_s": (
            None if resume_receipt is None else float(resume_receipt["model_time_s"])
        ),
        "resumed_restart_sha256": (
            None if resume_receipt is None
            else str(resume_receipt["forecast_envelope"]["contract_sha256"])
        ),
    }


def cartesian_door(arguments: argparse.Namespace) -> int:
    return run_cartesian(
        load_config(arguments.config),
        config_label=str(arguments.config),
        output=arguments.output,
        library=arguments.library,
        sounding=arguments.sounding,
        gpu_uuid=arguments.gpu_uuid,
        lock=arguments.lock,
        plan=arguments.plan,
        trace_phases=arguments.trace_phases,
        history_dir=arguments.history_dir,
        history_publisher_argument=arguments.history_publisher,
        history_carriers=arguments.history_carriers,
        restart_dir=arguments.restart_dir,
        restart_from=arguments.restart_from,
        integrate=arguments.integrate,
        emit_json=arguments.json,
        profile=arguments.profile,
        progress_interval=arguments.progress_interval,
        event_parts=arguments.event_parts,
        cuda_graphs=getattr(arguments, "cuda_graphs", "off"),
        events=getattr(arguments, "events", None),
        control_dir=getattr(arguments, "control_dir", None),
        history_envelope=getattr(arguments, "history_envelope", "on"),
        history_initial=bool(getattr(arguments, "history_initial", False)),
    )


__all__ = [
    "CARTESIAN_GATE",
    "CARTESIAN_INTEGRATION_RECEIPT_SCHEMA",
    "CARTESIAN_RECEIPT_SCHEMA",
    "allocation_plan",
    "bubble_center_record",
    "CARTESIAN_OPERATOR_FAMILIES",
    "NestBinding",
    "build_runtime_config",
    "cartesian_operator_family",
    "CARTESIAN_INITIAL_CONDITION_SCHEMA",
    "cartesian_plan_payload",
    "cartesian_door",
    "cartesian_history_plan",
    "measured_steps",
    "cartesian_frame_mode_record",
    "moving_frame_record",
    "require_cartesian_history_carriers",
    "require_cartesian_lane",
    "require_cuda_runtime",
    "require_grid_extent",
    "require_history_cadence",
    "require_initializer_selection",
    "require_level_capacity",
    "require_bubble_center",
    "require_adaptive_frame",
    "require_moving_frame",
    "require_nest",
    "nest_allocation_plan",
    "require_new_history_root",
    "require_ensemble",
    "require_new_receipt_path",
    "learn_step_rate",
    "LEARN_RATES_ENV",
    "require_run_schedule",
    "resume_selection",
    "resolve_history_publisher",
    "require_run_geometry",
    "RUN_GEOMETRY_ADMITTED_ISND",
    "run_geometry_hodograph",
    "open_lateral_sides",
    "horizontal_origin_m",
    "bubble_center_m",
    "closure_decomposition_peak_bytes",
    "require_sounding_extent",
    "require_sounding_report",
    "resolve_initializer_library",
    "run_cartesian",
    "perf_door",
    "require_profile_level",
    "require_event_parts",
    "require_cuda_graph_mode",
    "require_progress_interval",
    "resolve_sounding",
    "selected_capacity_profile",
]
