"""Strict, sealed configuration for the deliberately narrow first slice."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

from .config_types import LaneName
from .cm1_output_selection import (
    GM1_DEFAULTS as _OUTPUT_GM1_DEFAULTS,
    OUTPUT_KEYS,
    PRODUCT as _OUTPUT_PRODUCT,
    STATE as _OUTPUT_STATE,
    SWITCH_BY_KEY as _OUTPUT_SWITCH_BY_KEY,
    UNSUPPORTED as _OUTPUT_UNSUPPORTED,
    OutputSelection,
)
from .cm1_vertical_grid import ADMITTED_STRETCH_Z, STRETCH_Z2_REFUSAL, VerticalLevels
from .ensemble_branch import CM1_IRANDP_AMPLITUDE_K
from .errors import refuse
from .lagrangian_config import (
    LAGRANGIAN_KEYS,
    LagrangianConfig,
    LagrangianConfigError,
    parse_lagrangian_table,
)
from .nest_move import EDGE_POLICIES as _NEST_EDGE_POLICIES
from .gm1_runtime import (
    CM1_MAX_ACOUSTIC_SUBSTEPS,
    CM1_MIN_ACOUSTIC_SUBSTEPS,
    CM1_MIN_NORMAL_DT_S,
    MAX_FRAME_SPEED_M_S,
    acoustic_small_step_defect,
    cm1_diagnostic_acoustic_substeps,
    cm1_largest_clock_aligned_large_step_s,
    cm1_max_large_step_s,
    cm1_real,
    positive_normal_f32_bits,
)


CONFIG_SCHEMA = "gpuwm-cm1.config/v1"


class Lane(StrEnum):
    DRY_DYCORE = LaneName.DRY_DYCORE
    PRODUCT = LaneName.PRODUCT
    REAL_PRODUCT = LaneName.REAL_PRODUCT
    NATIVE_CONTEXT = LaneName.NATIVE_CONTEXT
    # The configurable Cartesian CUDA lane. Its identity string is bound to
    # the installed profile manifest by test_cartesian_door, so the record and
    # this enumeration cannot drift apart.
    CARTESIAN = "cartesian"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The run clock, and CM1's restart selection surface.

    ``irst``, ``rstnum``, ``iconly`` and ``restart_reset_frqtim`` keep CM1's
    own namelist spellings (README.namelist:227-235, 1488-1510) so the
    transcribed source line is findable from the configuration.  All four are
    optional: a document that does not declare them has the exact canonical
    identity it had before they existed, and :meth:`CM1Config.canonical_dict`
    drops an undeclared key rather than projecting a null.
    """

    lane: Lane
    duration_s: float
    dt_s: float
    history_interval_s: float
    restart_interval_s: float
    # README.namelist:227, cm1.F:2660.  0 = build the initial condition,
    # 1 = read a restart file and continue from it.
    irst: int | None = None
    # README.namelist:229-231.  Which restart file number to continue from.
    rstnum: int | None = None
    # README.namelist:233-235, cm1.F:4341-4350.  1 = build the initial
    # condition, write it and stop without integrating.
    iconly: int | None = None
    # README.namelist:1488-1510, cm1.F:2711-2715.
    restart_reset_frqtim: bool | None = None

    @property
    def restarting(self) -> bool:
        """CM1's ``irst.eq.1`` (cm1.F:2660)."""

        return self.irst == 1

    @property
    def initial_condition_only(self) -> bool:
        """CM1's ``iconly.eq.1`` (cm1.F:4341)."""

        return self.iconly == 1

    @property
    def reset_frqtim(self) -> bool:
        """CM1's ``restart_reset_frqtim`` with its cm1.F:363 default.

        ``cm1.F:363`` sets it ``.true.``; this lane's default is the opposite,
        ``.false.``, and the reason is README.namelist:1505-1510's own: the
        ``.false.`` branch keeps the schedule the restart file carried, which
        is the option CM1 names as the one that "may be needed to obtain
        bit-identical restarts".  A lane whose product is bit-exactness
        defaults to the branch that delivers it and lets a caller ask for the
        other by name.
        """

        return bool(self.restart_reset_frqtim)


@dataclass(frozen=True, slots=True)
class GridConfig:
    """CM1 ``param0``/``param6`` grid words.

    ``stretch_z`` is CM1's integer selector (``input.F:283``), not a flag:
    0 uniform, 1 Wilhelmson-Chen, 3 user scalar levels, 4 user w levels
    (``param.F:7814-8139``).  ``param.F``'s fifth branch, the COMMAS geometric
    ``stretch_z=2``, is refused by value; ``cm1_vertical_grid`` holds the
    reason.  ``dz_m`` stays CM1's ``dz``: the metric reference every ``mh`` and
    ``mf`` word divides by (``param.F:8170,8176``).  On a stretched grid that
    is no longer any layer's thickness, which is why ``ztop_m`` is declared
    separately instead of derived from ``nz*dz_m``.
    """

    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float
    terrain: bool
    stretch_x: bool
    stretch_y: bool
    stretch_z: int
    ztop_m: float
    str_bot_m: float
    str_top_m: float
    dz_bot_m: float
    dz_top_m: float
    levels_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    """CM1's ``param2``/``param3`` dynamical selectors, by CM1's own names.

    ``coriolis``/``fcor_s_inv``/``beta_plane`` are ``icor``/``fcor``/
    ``betaplane`` (README.namelist:518-530, solve2.F:631-720); ``alph`` and
    ``kdiv`` are the two ``param3`` reals the psolver=3 acoustic step reads
    (README.namelist:751-759); ``idiff``/``mdiff``/``difforder``/``kdiff2``/
    ``kdiff6`` are the explicit artificial diffusion of
    README.namelist:281-306,743-749.  Every one carries CM1 r21.1's own
    distributed ``run/namelist.input`` value when a document leaves it out,
    so a document written before these keys existed keeps the run and the
    ``sha256`` it was sealed with.
    """

    psolver: int
    adaptive_timestep: bool
    scalar_horizontal_order: int
    scalar_vertical_order: int
    velocity_horizontal_order: int
    velocity_vertical_order: int
    scalar_weno: int
    velocity_weno: int
    weno_order: int
    dry_mass_correction: bool
    coriolis: int = 0
    beta_plane: int = 0
    fcor_s_inv: float = 5.0e-5
    alph: float = 0.60
    kdiv: float = 0.10
    idiff: int = 0
    mdiff: int = 0
    difforder: int = 6
    kdiff2: float = 75.0
    kdiff6: float = 0.040
    # CM1's ``lspgrad`` (README.namelist:531-538, src/solve1.F:536-681).
    # Optional and defaulting to CM1's own ``lspgrad=0``
    # (run/namelist.input:61): a document that does not declare it projects
    # nothing and keeps the exact sealed identity it had before the key
    # existed.  Only 1 is admitted -- the geostrophic balance of the base
    # state, solve1.F:590-612 -- and _validate_dynamics_words refuses it
    # without an f-plane icor=1 or beside an adaptive moving domain.
    lspgrad: int | None = None

    @property
    def lspgrad_value(self) -> int:
        return 0 if self.lspgrad is None else self.lspgrad


@dataclass(frozen=True, slots=True)
class OpenBoundaryCompanionConfig:
    """CM1's open-boundary companions, by CM1's own namelist names.

    ``roflux`` is ``README.namelist:595-602`` and ``src/bc.F:2180-2391``:
    without it an open-boundary run can develop a runaway outward mass flux,
    domain-total mass loss and a spurious pressure fall.  ``nudgeobc`` and
    ``alphobc`` are ``README.namelist:603-610,770-771`` and
    ``src/solve2.F:726-822``: the inflow wind is pulled back toward the
    declared base state so a long run keeps the sounding it was given.
    ``cstar`` is ``README.namelist:826-829``, the prescribed outward phase
    speed ``src/bc.F:1964`` uses for ``irbc=1`` and ``irbc=2``.

    A document declares all four or none: a half-declared set would take the
    rest from this port's defaults, which is the thing this grammar exists to
    prevent.
    """

    roflux: int
    nudgeobc: int
    alphobc: float
    cstar: float


# ``src/cm1.F:327-346`` -- CM1's own initial values for the four.  An absent
# table is exactly this, and is not projected into the canonical document.
CM1_OPEN_BOUNDARY_COMPANIONS = OpenBoundaryCompanionConfig(
    roflux=0, nudgeobc=0, alphobc=60.0, cstar=0.0
)


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    west: int
    east: int
    south: int
    north: int
    bottom: int
    top: int
    radiative_order: int
    # Absent in every document written before the open-boundary companions
    # became configurable.  An absent table is CM1's own cm1.F initialization
    # and is not projected into the canonical document, so such a document
    # keeps the exact sealed identity it had before these keys existed.
    open_companions: OpenBoundaryCompanionConfig | None = None

    @property
    def roflux(self) -> int:
        return (self.open_companions or CM1_OPEN_BOUNDARY_COMPANIONS).roflux

    @property
    def nudgeobc(self) -> int:
        return (self.open_companions or CM1_OPEN_BOUNDARY_COMPANIONS).nudgeobc

    @property
    def alphobc(self) -> float:
        return (self.open_companions or CM1_OPEN_BOUNDARY_COMPANIONS).alphobc

    @property
    def cstar(self) -> float:
        return (self.open_companions or CM1_OPEN_BOUNDARY_COMPANIONS).cstar

    @property
    def lateral(self) -> tuple[int, int, int, int]:
        return (self.west, self.east, self.south, self.north)


@dataclass(frozen=True, slots=True)
class LowerBoundaryConfig:
    """CM1's own lower-boundary and surface-layer selectors.

    Every name here is the CM1 r21.1 namelist name.  ``bbc`` is the wind
    lower boundary condition of ``README.namelist:574-580`` -- 1 free slip,
    2 no slip, 3 semi slip -- and ``sfcmodel``, ``isfcflx``, ``cecd`` and
    ``cnstcd`` are the ``param12`` surface options of
    ``README.namelist:914-920,1017-1042``.  A document declares all five or
    none: a half-declared surface would let one of them fall back to an
    implicit scientific default, which is the thing this grammar exists to
    prevent.
    """

    bbc: int
    sfcmodel: int
    isfcflx: int
    cecd: int
    cnstcd: float
    # ``run/namelist.input:139,148-151``.  The remaining ``sfcmodel=1``
    # options of ``README.namelist:1037-1038,1072-1078``: the constant
    # enthalpy exchange coefficient, and the two switches that impose a
    # roughness length or a friction velocity instead of a drag coefficient.
    # Optional as their own group so a document written before they existed
    # keeps the exact canonical bytes it had; declaring any requires all
    # five, for the same reason the lower boundary does.
    surface_layer: "SurfaceLayerConfig | None" = None

    @property
    def cnstce(self) -> float:
        return (self.surface_layer or DEFAULT_SURFACE_LAYER).cnstce

    @property
    def set_znt(self) -> int:
        return (self.surface_layer or DEFAULT_SURFACE_LAYER).set_znt

    @property
    def cnst_znt(self) -> float:
        return (self.surface_layer or DEFAULT_SURFACE_LAYER).cnst_znt

    @property
    def set_ust(self) -> int:
        return (self.surface_layer or DEFAULT_SURFACE_LAYER).set_ust

    @property
    def cnst_ust(self) -> float:
        return (self.surface_layer or DEFAULT_SURFACE_LAYER).cnst_ust


@dataclass(frozen=True, slots=True)
class SurfaceLayerConfig:
    """CM1's remaining ``sfcmodel=1`` surface-layer options."""

    cnstce: float
    set_znt: int
    cnst_znt: float
    set_ust: int
    cnst_ust: float


# CM1 r21.1 ``run/namelist.input:139,148-151`` -- the shipped deck's own
# values.  ``set_znt`` and ``set_ust`` are both off there, so an undeclared
# group is the constant-drag surface CM1 itself runs by default and
# ``cnst_znt``/``cnst_ust`` are inert.
DEFAULT_SURFACE_LAYER = SurfaceLayerConfig(
    cnstce=0.001, set_znt=0, cnst_znt=0.16, set_ust=0, cnst_ust=0.25
)


@dataclass(frozen=True, slots=True)
class SurfaceConfig:
    """CM1's ``param12`` initial surface conditions and land surface.

    Every name is CM1's own (``README.namelist:971-1015``).  ``initsfc``
    chooses the spatial pattern, ``lu0``/``season`` choose the LANDUSE.TBL
    row that sets the roughness length and moisture availability, and
    ``oceanmodel`` chooses whether the water surface evolves.  Declaring the
    table requires a surface layer or a semi-slip bottom, because
    ``src/init_surface.F:40`` is the only writer of these fields and it runs
    only then.
    """

    initsfc: int
    tsk0_k: float
    tmn0_k: float
    xland0: float
    lu0: int
    season: int
    oceanmodel: int


# CM1 r21.1 ``run/namelist.input:130-136``, the shipped deck's own surface.
DEFAULT_SURFACE = SurfaceConfig(
    initsfc=1,
    tsk0_k=299.28,
    tmn0_k=297.28,
    xland0=2.0,
    lu0=16,
    season=1,
    oceanmodel=1,
)


# CM1 r21.1 ``run/namelist.input:70,128-129`` -- the supercell deck's own
# lower boundary: free slip with no surface model and no surface fluxes.
# ``cecd``/``cnstcd`` are inert in that deck because ``getcecd`` only runs
# for ``sfcmodel>=1`` (``src/turb.F:614-616``); the port names the single
# ``getcecd`` branch it implements rather than a branch it would refuse.
FREE_SLIP_LOWER_BOUNDARY = LowerBoundaryConfig(
    bbc=1, sfcmodel=0, isfcflx=0, cecd=1, cnstcd=0.001
)


@dataclass(frozen=True, slots=True)
class HorizontalDampingConfig:
    """CM1's lateral Rayleigh sponge, by CM1's own namelist names.

    ``hrdamp`` is ``README.namelist:401-403`` and the four-side ``tauf``
    construction of ``src/param.F:8687-8802``; ``xhd`` is the distance from
    each lateral boundary over which it acts (``README.namelist:767-768``).
    ``hrdamp=1`` damps toward the base state and ``hrdamp=2`` toward the
    evolving horizontal average, matching the two ``src/solve1.F:404-438``
    branches.  Since cm1r19.6 the lateral sponge acts on ``w`` only
    (``src/param.F:8736``).

    Declared as a pair or not at all: an ``hrdamp`` with no ``xhd`` would
    build a sponge of this port's width rather than the document's.
    """

    hrdamp: int
    xhd_m: float


# ``src/cm1.F`` leaves both at zero: no lateral sponge.  An absent table is
# exactly that and is not projected into the canonical document.
NO_HORIZONTAL_DAMPING = HorizontalDampingConfig(hrdamp=0, xhd_m=0.0)


@dataclass(frozen=True, slots=True)
class TurbulenceScaleConfig:
    """CM1's ``cm1setup`` and ``ipbl``: which closure mixes the column.

    They are optional as a pair, and absent they are CM1's LES deck --
    ``cm1setup=1`` with ``ipbl=0`` -- which is what GM1's Cartesian storm
    lane has always integrated (``run/namelist.input`` ships ``cm1setup=1``,
    ``ipbl=0``).  Declaring one requires declaring the other, because the
    pair is the whole of CM1's scale selection (``src/param.F:752-753``) and
    a half-declared one would take the other from an implicit default.
    """

    cm1setup: int
    ipbl: int


@dataclass(frozen=True, slots=True)
class RadiationConfig:
    """CM1's ``param11`` table: a radiation scheme, its cadence and its sun.

    Absent, the run has no sun -- ``radopt=0`` -- and stays location-free,
    which is exactly what every GM1 document written before this table
    existed asked for.
    """

    radopt: int
    dtrad_s: float
    ctrlat: float
    ctrlon: float
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


@dataclass(frozen=True, slots=True)
class AdaptiveMoveConfig:
    """CM1's adaptive moving-domain selectors, by CM1's own namelist names.

    ``do_adapt_move`` and ``adapt_move_frq`` are the two words CM1 reads in
    the ``param15`` group beside ``doazimavg``/``azimavgfrq``
    (``src/input.F:337``, ``README.namelist:1789-1797``).  ``do_adapt_move``
    finds the storm from a smoothed field of the lowest model level's pressure
    and adjusts ``umove``/``vmove`` to keep it near the domain centre;
    ``adapt_move_frq`` is how often, in seconds, that happens.

    A document declares both or neither.  A cadence without a controller is a
    number nothing reads, and a controller without a cadence would take its
    update frequency from an implicit default -- which is the one thing this
    grammar exists to prevent.
    """

    do_adapt_move: bool
    adapt_move_frq: float


# ``src/cm1.F:371`` and ``run/namelist.input:360-361``: CM1's own distributed
# values.  A document that declares neither key is exactly this, and is not
# projected into the canonical document, so its sealed sha256 -- and the
# plan_sha256 inside recorded GPU-run receipts -- is what it was before these
# keys existed.
FIXED_FRAME_ADAPTIVE_MOVE = AdaptiveMoveConfig(
    do_adapt_move=False, adapt_move_frq=3600.0
)


@dataclass(frozen=True, slots=True)
class SgsStartupRampConfig:
    """CM1's ``ramp_sgs``/``ramp_time`` (``README.namelist:1081-1087``).

    ``ramp_sgs=1`` ramps the LES subgrid model in over the first
    ``ramp_time`` seconds (``src/turb.F:240-262``).  Every deck CM1 ships
    declares ``ramp_sgs = 1`` and ``ramp_time = 1800.``; CM1's compiled
    default is ``0``/``0.0`` (``src/cm1.F:355-356``), which is also what this
    lane integrated before the pair existed.  A document declares both or
    neither.
    """

    ramp_sgs: int
    ramp_time: float


# ``src/cm1.F:355-356``: CM1's compiled default, and every document written
# before these keys existed.  Not projected into the canonical document.
NO_SGS_STARTUP_RAMP = SgsStartupRampConfig(ramp_sgs=0, ramp_time=0.0)


@dataclass(frozen=True, slots=True)
class ClosureConfig:
    sgs_model: int
    vertical_implicit_diffusion: bool
    upper_damping: int
    upper_damping_depth_m: float
    upper_damping_rate_s_inv: float
    moving_frame: bool
    u_move_m_s: float
    v_move_m_s: float
    # Absent in every document written before the lower boundary became
    # configurable.  An absent table is CM1's free-slip deck and is not
    # projected into the canonical document, so such a document keeps the
    # exact sealed identity it had before these keys existed -- the same
    # rule the optional [initialization] table already follows.
    lower_boundary: LowerBoundaryConfig | None = None
    # Absent in every document written before the lateral sponge became
    # configurable; an absent table is CM1's hrdamp=0 and projects nothing.
    horizontal_damping: HorizontalDampingConfig | None = None

    @property
    def hrdamp(self) -> int:
        return (self.horizontal_damping or NO_HORIZONTAL_DAMPING).hrdamp

    @property
    def xhd_m(self) -> float:
        return (self.horizontal_damping or NO_HORIZONTAL_DAMPING).xhd_m
    # CM1's ``rterm`` (README.namelist:555-558, src/solve1.F:1003-1054).
    # Optional and defaulting to CM1's own ``rterm=0``: a document that does
    # not declare it projects nothing and keeps the exact sealed identity it
    # had before the key existed.
    rterm: int | None = None
    # CM1's ``t2p_avg`` (README.namelist:1089-1092), read only by the
    # two-part models sgsmodel=3/4.  Absent is CM1's shipped 1
    # (run/config_files/namelist.input:154) and is not projected.
    t2p_avg: int | None = None
    # CM1's cm1setup/ipbl pair, by the same rule.
    turbulence_scale: TurbulenceScaleConfig | None = None

    @property
    def cm1setup(self) -> int:
        """CM1's ``cm1setup``; absent, the LES deck GM1 has always run."""

        return (self.turbulence_scale or DEFAULT_TURBULENCE_SCALE).cm1setup

    @property
    def ipbl(self) -> int:
        return (self.turbulence_scale or DEFAULT_TURBULENCE_SCALE).ipbl

    @property
    def rterm_value(self) -> int:
        return 0 if self.rterm is None else self.rterm

    @property
    def t2p_avg_value(self) -> int:
        return 1 if self.t2p_avg is None else self.t2p_avg
    # Absent in every document written before the moving domain became
    # adaptive.  An absent pair is CM1's own ``do_adapt_move = .false.``
    # (cm1.F:371) with the 3600 s cadence run/namelist.input:361 ships, and is
    # not projected into the canonical document, so such a document keeps the
    # exact sealed identity it had before these keys existed.
    adaptive_move: AdaptiveMoveConfig | None = None

    @property
    def do_adapt_move(self) -> bool:
        """CM1's ``do_adapt_move`` (README.namelist:1789-1794)."""

        return (self.adaptive_move or FIXED_FRAME_ADAPTIVE_MOVE).do_adapt_move

    @property
    def adapt_move_frq(self) -> float:
        """CM1's ``adapt_move_frq`` in seconds (README.namelist:1796-1797)."""

        return (self.adaptive_move or FIXED_FRAME_ADAPTIVE_MOVE).adapt_move_frq
    # CM1's SGS start-up ramp pair.  Absent is src/cm1.F:355-356's compiled
    # ramp_sgs=0 and is not projected, so an earlier document keeps its sealed
    # identity.
    sgs_startup_ramp: SgsStartupRampConfig | None = None

    @property
    def ramp_sgs(self) -> int:
        """CM1's ``ramp_sgs`` (README.namelist:1081-1082)."""

        return (self.sgs_startup_ramp or NO_SGS_STARTUP_RAMP).ramp_sgs

    @property
    def ramp_time(self) -> float:
        """CM1's ``ramp_time`` in seconds (README.namelist:1085-1086)."""

        return (self.sgs_startup_ramp or NO_SGS_STARTUP_RAMP).ramp_time

    @property
    def bbc(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).bbc

    @property
    def sfcmodel(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).sfcmodel

    @property
    def isfcflx(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).isfcflx

    @property
    def cecd(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).cecd

    @property
    def cnstcd(self) -> float:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).cnstcd

    @property
    def cnstce(self) -> float:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).cnstce

    @property
    def set_znt(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).set_znt

    @property
    def cnst_znt(self) -> float:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).cnst_znt

    @property
    def set_ust(self) -> int:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).set_ust

    @property
    def cnst_ust(self) -> float:
        return (self.lower_boundary or FREE_SLIP_LOWER_BOUNDARY).cnst_ust


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    """CM1's moisture-scheme selector and its output-side reflectivity word.

    ``output_dbz`` is CM1's own namelist name (``README.namelist:1366-1369``):
    "reflectivity (dBZ)?", available only for the ``ptype`` values whose
    scheme carries a reflectivity routine, and also writing the composite
    (column-maximum) reflectivity ``cref``.  ``param.F:1583-1603`` refuses it
    by ``ptype`` for exactly that reason -- the formula depends on the
    scheme's own assumed rain, snow and graupel size distributions.

    ``dbz_level_m`` has no CM1 namelist word: CM1 writes the whole 3-D ``dbz``
    field and leaves a user to slice it.  This port publishes one chosen level
    as a plan-view carrier, so the level has to be named, and it is named in
    metres and resolved through CM1's own descending level scan
    (``misclibs.F:2454-2466``).

    Both are optional keys with a stated default, like the ``param6``
    vertical-grid words above them: a document that declares neither is the
    document it has always been, byte for byte, and keeps its sealed
    ``sha256``.
    """

    """The selected column component and the CM1 options that reach it.

    ``ihail`` and ``ndcnst`` keep CM1's own ``&param2``/``&param3`` names.
    Both are optional keys, and an undeclared one binds the value recorded for
    the SELECTED component (``cm1core.dense_ice.default_ihail``), because CM1
    supplies no default of its own -- ``src/param.F:1567-1580`` stops the model
    outside ``0..1``.  The canonical document omits an undeclared key rather
    than nulling it, so a sealed configuration identity does not change because
    a key became declarable.
    """

    component: str
    equation_set: int
    output_dbz: bool = False
    dbz_level_m: float | None = None
    #: ``src/param.F:1567-1580``; 0 = graupel, 1 = hail.  None = undeclared.
    ihail: int | None = None
    #: ``&param3 ndcnst`` (cm-3).  None = undeclared.  Consumed by ptype=5.
    ndcnst: float | None = None
    # CM1's ``cuparam`` (README.namelist:513-517).  Optional, defaulting to
    # CM1's own ``cuparam=0``, and projected only when declared.
    cuparam: int | None = None

    @property
    def cuparam_value(self) -> int:
        return 0 if self.cuparam is None else self.cuparam
    #: ``src/param.F:1567-1580``; 0 = graupel, 1 = hail.  ``None`` is
    #: undeclared, which resolves per component at the door and is omitted
    #: from the canonical document rather than written as null.
    ihail: int | None = None
    #: ``src/morrison.F:329``'s constant cloud-droplet concentration, cm-3.
    #: ``None`` is undeclared; only ptype=5 reads it.
    ndcnst: float | None = None


from .initiation_forcing import (  # noqa: E402
    FORCING_DEFAULTS as _FORCING_DEFAULTS,
    FORCING_IINIT as _FORCING_IINIT,
    FORCING_KEYS as _FORCING_KEYS,
)


#: The ``[initialization]`` keys beyond ``isnd``/``iwnd``, with CM1's own
#: distributed value for each (``run/namelist.input:29,33,39,45,75-83`` and the
#: named parameter blocks of ``init3d.F:456-461,494,525-530,1275``).  ``bubble_center_x_m``
#: and ``bubble_center_y_m`` are ``None`` for CM1's ``centerx``/``centery``
#: (``init3d.F:456-457``), which is the domain centre.
_INITIALIZATION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "iinit": 1,
        "irst": 0,
        "imoist": 1,
        "testcase": 0,
        "ibalance": 0,
        "irandp": 0,
        "random_seed": 0,
        # Which stream iinit=8 and irandp draw from: "gm1" is GM1's own seeded
        # stream (what every run before this key drew from); "cm1" is CM1's
        # own numbers, libgfortran's random_number as init3d.F:427-440 seeds it.
        "random_generator": "gm1",
        # DELIBERATE DIVERGENCE FROM CM1'S DISTRIBUTED DECK, stated rather
        # than silent: run/namelist.input:81 ships iorigin = 2, but this
        # port's Rust initializer builds xh/yh from an origin of (0, 0)
        # (gm1_idealized_residency.IdealizedSource), which is param.F:7212's
        # iorigin = 1 frame, and every run this port has ever made was in
        # it.  Defaulting to 2 here would move the frame -- and therefore
        # every absolute-coordinate word read in it -- under every document
        # written before the key existed.  Declare iorigin = 2 to get CM1's
        # own centred frame.
        "iorigin": 1,
        "bubble_center_x_m": None,
        "bubble_center_y_m": None,
        "zc_m": 1400.0,
        "bhrad_m": 10000.0,
        "bvrad_m": 1400.0,
        "bptpert_k": 1.0,
        "maintain_rh": False,
        "nbub": 3,
        "bubble_first_center_y_m": 3000.0,
        "bubble_spacing_y_m": 30000.0,
        "zdep_m": 2500.0,
        "half_width_m": 5000.0,
        "random_line_amplitude_k": 0.20,
        "hurr_vg_m_per_s": 40.0,
        "hurr_rotate_deg": 0.0,
        # base.F:267-268: the moist benchmark's wet-equivalent potential
        # temperature and total water, set in source by the isnd = 4 arm and
        # read again by init3d.F:625-645's iinit = 4 bubble; param.F:6335-6336
        # zeroes both for every other sounding.
        "thec_mb_k": 320.0,
        "qt_mb": 0.020,
        # The namelist tsk0 base.F's isnd = 11 (:1204) and isnd = 21 (:1849)
        # build their surface air from; None is [surface].tsk0_k when that
        # table is declared, else run/namelist.input's 299.28 K.
        "tsk0_k": None,
        # init3d.F:1303-1396: the words of the three forcing arms (9, 10, 12),
        # by CM1's own names and with CM1's own values; see
        # cm1core.initiation_forcing.  Each is read by one arm only.
        # init3d.F:1372-1385, iinit=12
        "xc_wnudge": None,
        "xr_wnudge": 10000.0,
        "yc_wnudge": None,
        "yr_wnudge": 10000.0,
        "zc_wnudge": 1500.0,
        "zr_wnudge": 1500.0,
        "alpha_wnudge": 0.5,
        "wmax_wnudge": 10.0,
        "t1_wnudge": 900.0,
        "t2_wnudge": 1200.0,
        # init3d.F:1309-1315, iinit=9
        "dmax": -0.001,
        "zdeep": 2000.0,
        "lamx": 10000.0,
        "lamy": 10000.0,
        "xcent": 0.0,
        "ycent": 0.0,
        "convtime": 900.0,
        # init3d.F:1339-1344, iinit=10
        "xc_uforce": None,
        "xr_uforce": 10000.0,
        "zr_uforce": 10000.0,
        "alpha_uforce": 0.1,
        "t1_uforce": 3300.0,
        "t2_uforce": 3600.0,
    }
)
_INITIALIZATION_OPTIONAL_KEYS: frozenset[str] = frozenset(_INITIALIZATION_DEFAULTS)

#: base.F's sounding branches this port builds: every analytic arm
#: (rust analytic_soundings.rs, storm_initialization.rs for 5) and the two
#: input_sounding lanes.  6 stops CM1; 15 needs testcase 4/5's constants.
_ADMITTED_ISND: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23}
)

#: ``init3d.F``'s own per-branch named parameter block.  CM1 does not carry
#: one set of perturbation constants: each ``iinit`` arm assigns its own
#: before it builds ``tha``, so the value an absent key means is a function of
#: the declared branch and not of the key alone.  Taking one flat default
#: across every arm would build a ``+1.0 K`` warm bubble where ``iinit=2``
#: declares a ``-6.0 K`` dam-break cold pool (``init3d.F:499-502``) and where
#: ``iinit=5`` declares a ``-15.0 K`` density-current blob
#: (``init3d.F:662-667``) -- the sign of the buoyancy the run is triggered by,
#: changed silently and with no refusal to announce it.
#:
#: Verified line by line against ``work/native-reference/cm1r21.1``:
#: ``init3d.F:456-466`` (1), ``:495-502`` (2), ``:527-537`` (3),
#: ``:662-667`` (5), ``:1264-1275`` (8), ``:1355-1356`` (11) and
#: ``:1400,1408`` (13).  A key an arm never assigns keeps the module-level
#: value below, which is CM1's own initialization of the variable.
_INITIALIZATION_BRANCH_DEFAULTS: Mapping[int, Mapping[str, Any]] = MappingProxyType(
    {
        0: MappingProxyType({}),
        1: MappingProxyType(
            {
                "zc_m": 1400.0,
                "bhrad_m": 10000.0,
                "bvrad_m": 1400.0,
                "bptpert_k": 1.0,
                "maintain_rh": False,
            }
        ),
        2: MappingProxyType(
            {"zdep_m": 2500.0, "bptpert_k": -6.0, "maintain_rh": True}
        ),
        3: MappingProxyType(
            {
                "bubble_center_x_m": 30000.0,
                "zc_m": 1400.0,
                "bhrad_m": 10000.0,
                "bvrad_m": 1400.0,
                "bptpert_k": 2.0,
                "maintain_rh": False,
                "nbub": 3,
                # init3d.F:548-550 writes the three centres as literals
                # 3000, 33000 and 63000: a first centre and a 30 km step.
                "bubble_first_center_y_m": 3000.0,
                "bubble_spacing_y_m": 30000.0,
            }
        ),
        # init3d.F:585-589: the Bryan-Fritsch bubble, uniform in y.
        4: MappingProxyType(
            {
                "bubble_center_x_m": 0.0,
                "zc_m": 2000.0,
                "bhrad_m": 2000.0,
                "bvrad_m": 2000.0,
                "bptpert_k": 2.0,
                "maintain_rh": False,
            }
        ),
        5: MappingProxyType(
            {
                "bubble_center_x_m": 0.0,
                "bubble_center_y_m": 0.0,
                "zc_m": 3000.0,
                "bhrad_m": 4000.0,
                "bvrad_m": 2000.0,
                "bptpert_k": -15.0,
            }
        ),
        8: MappingProxyType(
            {
                "zc_m": 1500.0,
                "bhrad_m": 10000.0,
                "bvrad_m": 1500.0,
                "bptpert_k": 2.0,
                "maintain_rh": False,
                "random_line_amplitude_k": 0.20,
            }
        ),
        # init3d.F:1355-1356 writes the Skamarock-Klemp wave as literals:
        # 0.01 K over a 10 km depth (bvrad_m, the arm's vertical scale),
        # centred at 100 km with a 5 km envelope half width (half_width_m,
        # its horizontal scale).
        11: MappingProxyType(
            {
                "bptpert_k": 0.01,
                "bvrad_m": 10000.0,
                "bubble_center_x_m": 100000.0,
                "half_width_m": 5000.0,
            }
        ),
        # init3d.F:1400 sets bvrad=1000 as the layer depth; :1408's amplitude
        # is the literal 1.0.
        13: MappingProxyType({"bptpert_k": 1.0, "bvrad_m": 1000.0}),
    }
)


def _initialization_word(table: Mapping[str, Any], name: str, iinit: int) -> Any:
    """One ``[initialization]`` word: declared, else CM1's own branch value.

    The lookup order is the order CM1 itself resolves in: a declared value
    wins; otherwise the ``iinit`` arm's own assignment; otherwise the value
    the variable carries when no arm assigns it.
    """

    if name in table:
        return table[name]
    branch = _INITIALIZATION_BRANCH_DEFAULTS.get(iinit, {})
    if name in branch:
        return branch[name]
    return _INITIALIZATION_DEFAULTS[name]


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    """CM1 ``base.F`` and ``init3d.F`` selectors for the idealized lane.

    ``isnd`` is base.F's sounding branch and ``iwnd`` its base-state wind
    branch; every other key is one of CM1's own namelist or named-parameter
    words, so the transcribed source line is findable from the configuration.
    A document that declares only ``isnd`` and ``iwnd`` gets CM1's own
    distributed value for the rest and projects exactly the two keys it wrote.
    branch; both keep CM1's own namelist names so the transcribed source line
    is findable from the configuration.

    ``bubble_center_x_m``/``bubble_center_y_m`` are ``init3d.F:456-457``'s
    ``ric``/``rjc``: where the ``iinit=1`` warm bubble is centred, in domain
    metres from the ``(0, 0)`` corner this port's initializer builds from.
    ``init3d.F`` assigns them ``centerx``/``centery`` (``param.F:7408``), the
    domain centre, and that is exactly what an absent pair binds.  They are
    declared together or not at all, and an absent pair is not projected into
    the canonical document, so every configuration written before the bubble
    could be moved keeps the identity it had.
    """

    isnd: int
    iwnd: int
    iinit: int = 1
    irst: int = 0
    imoist: int = 1
    testcase: int = 0
    ibalance: int = 0
    irandp: int = 0
    random_seed: int = 0
    random_generator: str = "gm1"
    iorigin: int = 1
    #: ``init3d.F:456-457``'s ``ric``/``rjc``.  Spelled by what they do rather
    #: than by CM1's local variable name, because that is the spelling this
    #: port already sealed into shipped documents.
    bubble_center_x_m: float | None = None
    bubble_center_y_m: float | None = None
    zc_m: float = 1400.0
    bhrad_m: float = 10000.0
    bvrad_m: float = 1400.0
    bptpert_k: float = 1.0
    maintain_rh: bool = False
    nbub: int = 3
    bubble_first_center_y_m: float = 3000.0
    bubble_spacing_y_m: float = 30000.0
    zdep_m: float = 2500.0
    half_width_m: float = 5000.0
    random_line_amplitude_k: float = 0.20
    hurr_vg_m_per_s: float = 40.0
    hurr_rotate_deg: float = 0.0
    #: base.F:267-268 (isnd = 4; also init3d.F:625-645's iinit = 4).
    thec_mb_k: float = 320.0
    qt_mb: float = 0.020
    #: The tsk0 base.F's isnd = 11 and 21 read; None resolves at the door
    #: (``[surface].tsk0_k`` if declared, else CM1's 299.28 K).
    tsk0_k: float | None = None
    # init3d.F:1370-1396, iinit=12 updraft nudging (cm1core.initiation_forcing).
    # A ``None`` centre is CM1's centerx/centery, the domain centre.
    xc_wnudge: float | None = None
    xr_wnudge: float = 10000.0
    yc_wnudge: float | None = None
    yr_wnudge: float = 10000.0
    zc_wnudge: float = 1500.0
    zr_wnudge: float = 1500.0
    alpha_wnudge: float = 0.5
    wmax_wnudge: float = 10.0
    t1_wnudge: float = 900.0
    t2_wnudge: float = 1200.0
    # init3d.F:1303-1333, iinit=9 forced convergence.
    dmax: float = -1.0e-3
    zdeep: float = 2000.0
    lamx: float = 10000.0
    lamy: float = 10000.0
    xcent: float = 0.0
    ycent: float = 0.0
    convtime: float = 900.0
    # init3d.F:1334-1347, iinit=10 momentum forcing.  ``None`` is
    # minx+0.5*(maxx-minx), the domain centre.
    xc_uforce: float | None = None
    xr_uforce: float = 10000.0
    zr_uforce: float = 10000.0
    alpha_uforce: float = 0.1
    t1_uforce: float = 3300.0
    t2_uforce: float = 3600.0
    #: Which optional keys the document actually wrote.  It is not part of the
    #: canonical projection; it decides how much of the table that projection
    #: carries, exactly as ``grid.stretch_z`` does for the param6 words.
    declared: frozenset[str] = frozenset()

    @property
    def bubble_center_declared(self) -> bool:
        return self.bubble_center_x_m is not None

    def bubble_center_xy_m(
        self, *, nx: int, ny: int, dx_m: float, dy_m: float
    ) -> tuple[float, float]:
        """``ric``/``rjc``, declared or defaulted to ``param.F:7408``.

        The port's initializer builds ``xh``/``yh`` from an origin of
        ``(0, 0)`` (``gm1_idealized_residency.IdealizedSource``), so
        ``minx = 0`` and ``param.F:7408``'s ``centerx`` is ``0.5*nx*dx``.
        """

        if self.bubble_center_x_m is None or self.bubble_center_y_m is None:
            return 0.5 * nx * dx_m, 0.5 * ny * dy_m
        return self.bubble_center_x_m, self.bubble_center_y_m


@dataclass(frozen=True, slots=True)
class ChainConfig:
    """A standalone Cartesian domain driven by a saved ArWen history.

    CM1 r21.1 has no lateral boundary forcing of any kind; this is a declared
    GM1 extension whose boundary authority is WRF's specified (``wrfbdy`` /
    ``ndown``) lane, recorded in :mod:`cm1core.arwen_chain` and
    ``records/GM1-ARWEN-CHAIN.md``.  Declared whole or not at all: every word
    below changes the air the domain is built from or forced with.
    """

    history_directory: str
    domain_id: int
    start_time: str
    end_time: str
    parent_i0: int
    parent_j0: int
    refinement_ratio: int
    spec_zone: int
    relax_zone: int
    terrain_relief_limit_m: float
    map_factor_departure_limit: float
    prepared_directory: str
    #: ``all-species`` (absent: the default, not projected) or ``vapour-only`` (WRF's
    #: ``have_bcs_moist = .false.``; ``arwen_chain.MOISTURE_LANES``).
    moisture: str | None = None


@dataclass(frozen=True, slots=True)
class NestConfig:
    """One refined child grid inside the Cartesian parent.

    Every name here is this port's own, because **CM1 r21.1 has no grid
    nesting at all**: no ``nest`` namelist word, no nesting section in
    ``README``/``README.namelist``, no ``CHANGES`` entry and no parent/child
    code in ``src`` -- its only moving-mesh capability is the whole-domain
    adaptive move of ``src/azimavg.F:4376-4528``.  Nesting is a declared
    extension beyond CM1, recorded as one in
    ``records/CM1-NEST-SEAM.md``, and :mod:`cm1core.nest_geometry` carries the
    same sentence so a receipt can never imply otherwise.

    What is **not** this port's own is every number the child's clock is
    admitted against: ``src/misclibs.F:857-868`` sets the psolver=3 acoustic
    small step from the horizontal spacings, ``src/misclibs.F:883-885`` derives
    the substep count, ``src/misclibs.F:835`` caps it at 192, and
    ``src/misclibs.F:2714`` is the advective Courant stop the refinement moves
    the child toward.  The child is a CM1 grid and CM1's own rules decide what
    it may run.

    The table is declared whole or not at all.  A half-declared nest would let
    one of the zone widths or the feedback direction fall back to an implicit
    value, which is the thing this grammar exists to prevent; and an absent
    table is not projected into the canonical document, so a configuration
    written before nesting existed keeps exactly the sealed identity it had.
    """

    refinement_ratio: int
    time_refinement_ratio: int
    parent_i0: int
    parent_j0: int
    parent_span_x: int
    parent_span_y: int
    spec_zone: int
    relax_zone: int
    feedback: int
    acoustic_substeps: int
    follow_storm: bool
    follow_interval_s: float
    follow_max_speed_m_s: float
    follow_tracker: str = "cm1-pressure-held-offset"
    # Declared only for a following nest; None is "not declared" and is not
    # projected, so a document written before these keys keeps its identity.
    follow_smoothing_radius_m: float | None = None
    follow_corral_m: float | None = None
    #: What the surface tracker does when its maximum lands on the search-window
    #: edge, it finds no circulation, its centroid lies beyond reach, or a move
    #: would enter the corral: ``"stop"`` or ``"hold"`` (cm1core.nest_move, THE
    #: HOLD POLICY).  ``None`` is undeclared, which is ``stop``, and is not
    #: projected.
    follow_on_edge: str | None = None
    #: Consecutive held tracker samples before ``hold`` relocates; declared
    #: exactly when ``follow_on_edge = "hold"``, not projected otherwise.
    follow_hold_limit: int | None = None
    #: Half-width in metres of the window a relocation searches around the kept
    #: centre; declared exactly when ``follow_on_edge = "hold"``, not projected
    #: otherwise.
    follow_relocation_radius_m: float | None = None
    #: The child's own vertical deck, or ``None`` for the parent's.  Declared
    #: through the optional ``nest.child_*`` keys; absent, it is not projected
    #: into the canonical document, so every nest document sealed before
    #: vertical refinement existed keeps its bytes.
    child_vertical: "NestChildVerticalConfig | None" = None
    #: ``[nest.tornado_criterion]``: the fixed tornado criterion sampled on the
    #: live child every parent step (:mod:`cm1core.nest_tornado_criterion`), or
    #: ``None``.  Absent, nothing is sampled and nothing is projected, so every
    #: nest document sealed before the criterion existed keeps its bytes.
    tornado_criterion: "NestTornadoCriterionConfig | None" = None


@dataclass(frozen=True, slots=True)
class NestTornadoCriterionConfig:
    """``[nest.tornado_criterion]``: sample the fixed tornado criterion on the live child.

    GM1 EXTENSION.  CM1 r21.1 has no nesting and makes no tornado claim, so no
    key here has a namelist word.  Declaring the table (it may be empty) is
    what turns the criterion on.  Its thresholds are not keys: they are the
    constants of :mod:`cm1core.nest_tornado_criterion`, fixed before any run
    and never tuned, so a TORNADO verdict means the same thing in every
    receipt that carries the same criterion schema.  The two keys are stop
    rules only, and each is optional; absent, the criterion never ends a run.
    """

    stop_after_episode_s: float | None = None
    stop_if_no_tornado_by_s: float | None = None

    def record(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in NEST_TORNADO_CRITERION_KEYS_ORDERED}


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """One reproducible ensemble member, by CM1's own perturbation.

    ``irandp`` (``README.namelist:692-695``) is CM1's ensemble switch and
    ``src/init3d.F:1446-1483`` is what it does: a uniform potential-temperature
    perturbation of plus or minus one amplitude added to ``tha`` over the whole
    domain.  ``src/init3d.F:1466`` sets that amplitude in the Fortran source --
    "set magnitude of perturbations in init3d.F" -- and CM1's own value is
    0.25 K, which is what ``theta_amplitude_k`` carries here.

    ``member`` and ``seed`` are this port's own, and they exist because CM1's
    ``random_number`` stream is unseeded: a CM1 member cannot be re-run, so it
    cannot be bisected, and the question an ensemble is for -- was this outcome
    inevitable, or did it turn on this perturbation -- cannot be asked of it.

    A run that declares no ``[ensemble]`` table is the unperturbed control, and
    projects no ensemble key, so every document sealed before ensembles existed
    keeps exactly the ``sha256`` it had.
    """

    member: int
    seed: int
    theta_amplitude_k: float


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """CM1 ``writeout.F`` environment-field switches and ``getcape.F`` controls.

    ``output_cape``/``output_cin``/``output_lcl``/``output_lfc`` keep CM1's own
    namelist names (``input.F:122``, ``param.F:4684-4687``) so the transcribed
    source line is findable from the configuration.  Absent, the whole table is
    CM1 r21.1's own shipped state: ``run/namelist.input`` leaves all four at 0
    and ``writeout.F:2895`` then runs no column sweep at all.

    The remaining four keys are ``getcape.F``'s own source parameters --
    ``getcape.F:65-68`` (source), ``getcape.F:73-77`` (adiabat),
    ``getcape.F:70`` (ml_depth) and ``getcape.F:60`` (pinc) -- with the values
    CM1 itself ships, so a document that declares the table and changes nothing
    else gets exactly CM1's arithmetic.  ``storm_motion`` is not CM1: CM1 r21.1
    computes no storm motion anywhere, so the t=0 report names the published
    method it uses instead of leaving a reader to guess.
    """

    output_cape: bool
    output_cin: bool
    output_lcl: bool
    output_lfc: bool
    parcel_source: int
    adiabat: int
    mixed_layer_depth_m: float
    pressure_increment_pa: float
    storm_motion: str


#: ``writeout.F:2895``.  All four switches off is CM1's own shipped state and
#: the state every document that declares no [diagnostics] table is in.
NO_ENVIRONMENT_FIELDS = DiagnosticsConfig(
    output_cape=False,
    output_cin=False,
    output_lcl=False,
    output_lfc=False,
    parcel_source=3,
    adiabat=1,
    mixed_layer_depth_m=500.0,
    pressure_increment_pa=200.0,
    storm_motion="bunkers-right",
)


@dataclass(frozen=True, slots=True)
class TornadoDiagnosticsConfig:
    """CM1's ``output_*`` diagnostic switches for the Cartesian lane.

    Every name but ``tornado_report`` is CM1 r21.1's own namelist name
    (``src/input.F:120``) and carries CM1's own 0/1 integer, so a reader can
    find the branch a document selected in ``src/writeout.F`` and
    ``src/solve3.F`` by name.  ``tornado_report`` has no CM1 counterpart
    because CM1 makes no tornado claim; it is GM1's own and it is a boolean.

    The table is optional as a whole and mandatory key by key.  Absent, the
    run computes, allocates and publishes exactly what it did before these
    diagnostics existed, and the canonical document does not project the
    table -- so every identity sealed from a document written before it is
    byte-for-byte what it was.
    """

    output_uh: int
    output_vort: int
    output_pv: int
    output_sws: int
    output_svs: int
    output_sps: int
    output_srs: int
    output_sgs: int
    output_sus: int
    output_shs: int
    tornado_report: bool


@dataclass(frozen=True, slots=True)
class CartesianRuntimeConfig:
    """The Cartesian lane's execution table; absent in every other lane."""

    acoustic_substeps: int
    # The closure tile.  ``None`` for both means the document declared
    # neither key: the closure runs as ONE whole-domain tile, which the door
    # admits only when the priced device peak fits runtime.memory_budget_mib
    # (cartesian_door.resolve_closure_tiles).  The decomposition is an
    # execution decision -- tools/oracle/cartesian_tiling_identity.py proves
    # the tiled and untiled closures raw-word identical -- so an undeclared
    # pair inherits no scientific default and is not projected, and every
    # document that declares the pair keeps its canonical bytes.
    tile_nx: int | None
    tile_ny: int | None
    warmup_steps: int
    memory_budget_mib: int
    capacity_profile: str
    # How the pre-WSM6 positive-definite repair (CM1 src/misclibs.F:1037-1117)
    # is spread over the device.  ``None`` means the document did not declare
    # it and the runtime's own default stands.  This is an execution layout,
    # not a CM1 selector: both admitted values run misclibs.F's per-column k
    # order and its left-associated west-to-east budj carry serially, and are
    # raw-word identical, so an undeclared key inherits no scientific default
    # and the canonical document keeps exactly the bytes it had before the key
    # existed.
    pdefq_launch: str | None = None
    # The rectangle one WSM6 call covers (CM1 src/mp_driver.F:1469-1479 is
    # called once per physical column, and no column reads another).  ``None``
    # means the document did not declare it and the WSM6 block follows the
    # closure tile, exactly as it always has.  Like ``pdefq_launch`` this is an
    # execution decision -- it changes how many times per-call staging and
    # drain are paid and how much component scratch one call needs, not what
    # any column computes -- so an undeclared key inherits no scientific
    # default and the canonical document keeps its bytes.
    physics_block_nx: int | None = None
    physics_block_ny: int | None = None


@dataclass(frozen=True, slots=True)
class CM1Config:
    run: RunConfig
    grid: GridConfig
    dynamics: DynamicsConfig
    boundaries: BoundaryConfig
    closures: ClosureConfig
    physics: PhysicsConfig
    runtime: CartesianRuntimeConfig | None = None
    initialization: InitializationConfig | None = None
    # CM1's param12 initial surface conditions.  Optional as a table for the
    # same reason [initialization] is: a document that does not declare it
    # keeps the exact sealed identity it had before the table existed, and
    # gets CM1's own shipped deck (run/namelist.input:130-136).
    surface: SurfaceConfig | None = None
    radiation: RadiationConfig | None = None
    #: CM1 ``&param9``.  Absent means this lane's own defaults, which
    #: ``cm1_output_selection.DEFAULT_MOVES`` states key by key against CM1
    #: r21.1's distributed ``run/namelist.input``.
    output: "OutputSelection | None" = None
    #: CM1 has no nesting; this is a declared GM1 extension, recorded in
    #: records/CM1-NEST-SEAM.md.  Absent is the single-domain run.
    nest: "NestConfig | None" = None
    #: Reproducible ensemble branching on CM1's irandp theta perturbation.
    ensemble: "EnsembleConfig | None" = None
    diagnostics: DiagnosticsConfig | None = None
    #: misclibs.F's rotation diagnostics and solve3.F's swaths, declared in
    #: the same [diagnostics] table but as their own all-or-none family.
    tornado_diagnostics: "TornadoDiagnosticsConfig | None" = None
    # CM1's parcel, budget, circuit, pressure-decomposition and
    # high-frequency-output selectors.  Optional as a whole: a document that
    # omits [lagrangian] projects nothing for it and keeps the exact sealed
    # sha256 it had before the table existed, which is what every recorded
    # receipt's plan_sha256 names.
    lagrangian: "LagrangianConfig | None" = None
    schema: str = CONFIG_SCHEMA
    #: A saved ArWen history driving this domain (cm1core.arwen_chain).  Absent
    #: is not projected, so every document sealed before it keeps its bytes.
    chain: "ChainConfig | None" = None

    @property
    def environment_fields(self) -> "DiagnosticsConfig":
        """CM1's ``output_*`` switches and getcape controls for this document.

        An absent ``[diagnostics]`` table is ``writeout.F:2895``'s own shipped
        state: no plane is written and no column sweep runs.
        """

        return self.diagnostics or NO_ENVIRONMENT_FIELDS

    def canonical_dict(self) -> dict[str, object]:
        from .profiles import profile_identity

        # OutputSelection holds read-only mappings, which asdict cannot deep
        # copy; it is projected below from the document's own declared words.
        payload = asdict(self if self.output is None else replace(self, output=None))
        payload["run"]["lane"] = self.run.lane.value
        # An optional [run] key that was not declared keeps the exact sealed
        # identity the document had before the key existed; an absent selector
        # is not projected as a null.
        for key in _OPTIONAL_RUN_KEYS:
            if payload["run"].get(key) is None:
                payload["run"].pop(key, None)
        # A lane that declares no [runtime] table keeps the exact sealed
        # identity it had before the table existed; an absent section is not
        # projected as a null key.
        if self.runtime is None:
            del payload["runtime"]
        if self.chain is None:
            del payload["chain"]
        elif payload["chain"].get("moisture") is None:
            # An undeclared moisture lane keeps the sealed identity chained documents had
            # before the key existed.
            del payload["chain"]["moisture"]
        # The same rule holds for the optional [initialization] table: a
        # document that does not declare it keeps the exact sealed identity it
        # had before the table existed.  The two ``init3d.F:456-457``
        # ``ric``/``rjc`` words inside it follow the same rule one level down:
        # a table that does not move the bubble projects exactly the two keys
        # it always projected, so its canonical bytes -- and the plan_sha256
        # that recorded GPU-run receipts carry -- do not move either.
        if self.initialization is None:
            del payload["initialization"]
        else:
            # The same rule the param6 vertical words follow: a document that
            # declares none of the optional CM1 selectors projects exactly the
            # two keys it has always projected, so every identity sealed from
            # those bytes -- including the plan_sha256 inside recorded GPU-run
            # receipts -- is unchanged.  A document that declares any of them
            # projects the whole resolved table, because at that point the
            # remaining defaults are load-bearing and must be stated.  The
            # bubble-centre pair is one of those optional keys, so an
            # undeclared pair is dropped here exactly as it was before.
            table = payload["initialization"]
            declared = table.pop("declared")
            if not declared:
                for name in _INITIALIZATION_OPTIONAL_KEYS:
                    del table[name]
            else:
                # A forcing arm's words are projected only beside that arm,
                # or where the document wrote them, so every document sealed
                # before they existed keeps its bytes.
                branch_words = set(_FORCING_DEFAULTS.get(self.initialization.iinit, {}))
                for name in (_FORCING_KEYS | {"random_generator"}) - branch_words - set(declared):
                    del table[name]
                # base.F's own words follow the same rule: projected
                # beside the sounding that reads them, or where declared.
                sounding_words = set()
                if self.initialization.isnd == 4:
                    sounding_words |= {"thec_mb_k", "qt_mb"}
                if self.initialization.isnd in (11, 21):
                    sounding_words |= {"tsk0_k"}
                for name in {"thec_mb_k", "qt_mb", "tsk0_k"} - sounding_words - set(declared):
                    del table[name]
                table["declared"] = sorted(declared)
        # The same rule for the [surface] table.
        if self.surface is None:
            del payload["surface"]
        # The same rule for the optional [physics] keys: an undeclared CM1
        # selector is absent from the canonical document, not present as null.
        for key in ("ihail", "ndcnst"):
            if payload["physics"][key] is None:
                del payload["physics"][key]
        # The same rule again for [radiation]: an absent table is CM1's
        # radopt=0 -- no sun, no place, no date -- and projects nothing, so
        # every document sealed before this table existed keeps its bytes.
        if self.radiation is None:
            del payload["radiation"]
        # &param9.  A document that declares no [output] table keeps the exact
        # sealed identity it had before the table existed; a declared table
        # projects only the keys the document actually wrote, so a run that
        # asks for one extra field does not re-hash every other key.
        if self.output is None:
            del payload["output"]
        else:
            payload["output"] = {
                key: self.output[key] for key in self.output.declared
            }
        # The same rule for the optional [nest] table.  A document that
        # declares no nest is the single-domain run it has always been and
        # projects no nest key, so its canonical bytes -- and the plan_sha256
        # every recorded GPU receipt carries -- are exactly what they were
        # before grid refinement became configurable.
        if self.nest is None:
            del payload["nest"]
        else:
            for key in (
                "follow_smoothing_radius_m",
                "follow_corral_m",
                "follow_on_edge",
                "follow_hold_limit",
                "follow_relocation_radius_m",
            ):
                if payload["nest"][key] is None:
                    del payload["nest"][key]
            if self.nest.child_vertical is None:
                # A nest on the parent's deck projects exactly the keys it did
                # before a child could declare its own.
                del payload["nest"]["child_vertical"]
            # The same rule for [nest.tornado_criterion]: absent projects
            # nothing, and a declared table projects its stop rules only when
            # the document wrote them (an empty table projects as {}).
            if self.nest.tornado_criterion is None:
                del payload["nest"]["tornado_criterion"]
            else:
                criterion = payload["nest"]["tornado_criterion"]
                for key in NEST_TORNADO_CRITERION_KEYS_ORDERED:
                    if criterion[key] is None:
                        del criterion[key]
        # And for the optional [ensemble] table: a document that declares none
        # is the unperturbed control every sealed identity was measured on.
        if self.ensemble is None:
            del payload["ensemble"]
        # And again for the optional [diagnostics] table.  A document that
        # declares none of CM1's output_cape/output_cin/output_lcl/output_lfc
        # switches is in writeout.F:2895's own shipped state -- no column sweep
        # runs and no plane is written -- which is exactly the state every
        # sealed document was in before this table existed, so its canonical
        # bytes and every identity taken from them are unchanged.
        # [tornado_diagnostics] is not a table of its own: it is parsed out of
        # [diagnostics] and projected there below.  asdict still emits the
        # dataclass field at the top level -- a null for every document that
        # declares none, which moved every sealed identity when the field was
        # added -- so it is removed here and never projected twice.
        del payload["tornado_diagnostics"]
        if self.diagnostics is None and self.tornado_diagnostics is None:
            del payload["diagnostics"]
        else:
            # Two families share one table; each projects only what it parsed,
            # so a document that declares one does not gain the other's keys.
            table = payload["diagnostics"] = {}
            for family in (self.diagnostics, self.tornado_diagnostics):
                if family is not None:
                    table.update(asdict(family))
        # The same rule for the optional [diagnostics] table.  A document
        # that declares no CM1 output_* switches projects none of them, so
        # its canonical bytes -- and the plan_sha256 every recorded GPU-run
        # receipt carries -- are exactly what they were before the switches
        # existed.
        # ...and for [lagrangian].  When it IS declared, every key it resolves
        # is projected -- declared or defaulted -- so the sealed document names
        # the whole diagnostic configuration instead of inheriting one.
        if self.lagrangian is None:
            del payload["lagrangian"]
        else:
            payload["lagrangian"] = self.lagrangian.projection()
        # The five CM1 lower-boundary keys live in [closures] under CM1's own
        # names.  A document that declares none of them is CM1's free-slip
        # deck and projects none of them, so its canonical bytes -- and every
        # identity sealed from them -- are exactly what they were before the
        # lower boundary became configurable.  A document that declares them
        # projects them flat, as the five keys it actually wrote.
        lower_boundary = payload["closures"].pop("lower_boundary")
        if lower_boundary is not None:
            # The same rule again for the five sfcmodel=1 surface-layer
            # options: a document that declares none projects none, so every
            # identity sealed from a document written before they existed is
            # exactly what it was.  They are nested inside the lower boundary
            # and must be lifted out of it before it is flattened, whether or
            # not any other optional group is declared.
            surface_layer = lower_boundary.pop("surface_layer")
            payload["closures"].update(lower_boundary)
            if surface_layer is not None:
                payload["closures"].update(surface_layer)
        # The same rule for CM1's open-boundary companions and its lateral
        # Rayleigh sponge.  A document that declares neither group projects
        # neither, so the canonical bytes -- and every identity sealed from
        # them, including the plan_sha256 recorded GPU-run receipts carry --
        # are exactly what they were before these keys existed.
        companions = payload["boundaries"].pop("open_companions")
        if companions is not None:
            payload["boundaries"].update(companions)
        sponge = payload["closures"].pop("horizontal_damping")
        if sponge is not None:
            payload["closures"].update(sponge)
        # closures.rterm and the cm1setup/ipbl pair follow the same rule as
        # the lower boundary: a document that does not declare them is
        # CM1's own rterm=0 LES deck and projects neither, so its canonical
        # bytes -- and the plan_sha256 every recorded receipt carries -- are
        # exactly what they were before these keys existed.
        turbulence_scale = payload["closures"].pop("turbulence_scale")
        if turbulence_scale is not None:
            payload["closures"].update(turbulence_scale)
        if payload["closures"]["rterm"] is None:
            del payload["closures"]["rterm"]
        if payload["closures"]["t2p_avg"] is None:
            del payload["closures"]["t2p_avg"]
        if payload["physics"]["cuparam"] is None:
            del payload["physics"]["cuparam"]
        # README.namelist:1789-1797's two adaptive moving-domain words follow
        # the same rule.  A document that declares neither is CM1's own
        # ``do_adapt_move = .false.`` deck and projects neither, so its
        # canonical bytes -- and every identity sealed from them -- are exactly
        # what they were before the moving domain became adaptive.
        adaptive_move = payload["closures"].pop("adaptive_move")
        if adaptive_move is not None:
            payload["closures"].update(adaptive_move)
        # README.namelist:1081-1087's ramp pair, by the same rule.
        sgs_startup_ramp = payload["closures"].pop("sgs_startup_ramp")
        if sgs_startup_ramp is not None:
            payload["closures"].update(sgs_startup_ramp)
        # The same rule for the param6 vertical-grid words.  An unstretched
        # grid projects exactly the one word it has always projected, in the
        # form it has always had: before these keys existed, ``stretch_z`` was
        # the boolean "is this grid stretched", and for stretch_z=0 that is
        # still precisely what the document says.  The five analytic lengths
        # and the level list are not projected, because param.F:2608 derives
        # this branch's model top and param.F:7814 reads none of the rest.
        # Keeping the projection fixed keeps every sealed identity fixed --
        # including the plan_sha256 that recorded GPU-run receipts carry.
        # The pdefq launch layout is projected only by a document that
        # declares it.  It changes no word the model computes -- the two
        # layouts are raw-word identical -- so a document that says nothing
        # about it keeps the identity it had before the key existed.
        if self.runtime is not None:
            for name in (
                "pdefq_launch", "physics_block_nx", "physics_block_ny", "tile_nx", "tile_ny",
            ):
                if getattr(self.runtime, name) is None:
                    del payload["runtime"][name]
        grid = payload["grid"]
        if self.grid.stretch_z == 0:
            for name in _VERTICAL_GRID_WORDS:
                del grid[name]
            grid["stretch_z"] = False
        else:
            grid["levels_m"] = list(self.grid.levels_m)
        # README.namelist:1366-1369.  A run that asks for no reflectivity
        # projects exactly the two physics words it always has, so every
        # example document's sealed sha256 -- and the plan_sha256 inside
        # recorded GPU-run receipts -- is what it was before this key existed.
        physics = payload["physics"]
        if not self.physics.output_dbz:
            del physics["output_dbz"]
        if self.physics.dbz_level_m is None:
            del physics["dbz_level_m"]
        # The CM1 param2/param3 dynamical words follow the same rule: a word
        # no operator reads under this document's own selectors is not
        # projected, so it cannot move an identity without moving a result.
        # ``fcor``/``betaplane`` are read only under icor=1 (solve2.F:632) and
        # the four diffusion reals only under idiff>=1 (adv.F:962,
        # cm1.F:3791).  ``alph`` and ``kdiv`` are read by every psolver=3
        # small step (sound.F:79-80 and the divergence damper), so they are
        # always projected: this port carried alph=0.5 and now carries CM1's
        # own 0.60, which changes every acoustic result, and an identity that
        # did not move with it would name two different runs.
        dynamics = payload["dynamics"]
        if self.dynamics.coriolis == 0:
            for name in ("beta_plane", "fcor_s_inv"):
                del dynamics[name]
        if self.dynamics.idiff == 0:
            for name in ("mdiff", "difforder", "kdiff2", "kdiff6"):
                del dynamics[name]
        # dynamics.lspgrad follows closures.rterm's rule: a document that does
        # not declare it is CM1's own lspgrad=0 deck and projects nothing, so
        # its canonical bytes -- and the plan_sha256 every recorded receipt
        # carries -- are exactly what they were before the key existed.
        if self.dynamics.lspgrad is None:
            del dynamics["lspgrad"]
        payload["profile_identity"] = profile_identity(self.run.lane.value)
        return payload

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def restart_identity(self) -> dict[str, object]:
        """The part of this document a restart's carriers depend on.

        ``README.restart:16-24`` is explicit that a user may change the
        namelist on a restart -- "minor changes in model configuration, like
        the advection scheme or time step, could probably be changed without
        any problems" -- and ``src/cm1.F:2711-2714`` exists precisely so a
        resumed run may take new output and restart frequencies.  Binding a
        resume to the whole sealed document would therefore refuse the normal
        case: continuing a run to a longer ``run.duration_s``, which is what a
        second leg is.

        What may not change is what the restored words *mean*.  Every key
        except the run clock, the restart selectors and the execution table is
        part of that meaning, so this projection removes exactly those three
        and keeps everything else, including ``run.dt_s``: the held rate
        carriers are formed with the step (``src/solve2.F:145``) and
        ``cm1.F:3512``'s domain-location integral is only exact for the step
        it accumulated at.
        """

        payload = self.canonical_dict()
        run = dict(payload["run"])  # type: ignore[arg-type]
        for name in ("duration_s", "history_interval_s", "restart_interval_s"):
            run.pop(name, None)
        payload["run"] = run
        payload.pop("restart", None)
        payload.pop("runtime", None)
        # A member branches from the control's restart, so the ensemble table
        # is outside the identity too: it changes the state after the restore,
        # not what the restored words mean.
        payload.pop("ensemble", None)
        payload["restart_identity_schema"] = "gpuwm-cm1.restart-identity/v1"
        return payload

    @property
    def restart_identity_sha256(self) -> str:
        """SHA-256 of :meth:`restart_identity`, in canonical bytes."""

        return hashlib.sha256(
            json.dumps(
                self.restart_identity(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


# CM1's lower-boundary and surface-layer selectors, by CM1's own namelist
# names.  They are optional as a group: absent means CM1's free-slip deck
# (README.namelist:574-580, run/namelist.input:70,128-129), and declaring any
# one of them requires declaring all five so no surface selector is ever
# implicit.
_LOWER_BOUNDARY_KEYS: frozenset[str] = frozenset(
    {"bbc", "sfcmodel", "isfcflx", "cecd", "cnstcd"}
)

# CM1's two adaptive moving-domain selectors, by CM1's own namelist names
# (README.namelist:1789-1797, src/input.F:337).  Optional as a pair for the
# same reason the five lower-boundary keys are: absent is CM1's own
# ``do_adapt_move = .false.`` deck, and every example document's canonical
# sha256 already travels inside recorded GPU-run receipts as ``plan_sha256``.
_ADAPT_MOVE_KEYS: frozenset[str] = frozenset({"do_adapt_move", "adapt_move_frq"})

# CM1's SGS start-up ramp pair (README.namelist:1081-1087, src/input.F:139,196).
# Optional as a pair: absent is src/cm1.F:355-356's compiled ramp_sgs=0.
_SGS_RAMP_KEYS: frozenset[str] = frozenset({"ramp_sgs", "ramp_time"})

#: ``init3d.F:456-457``'s ``ric``/``rjc``.  Optional as a pair: absent is
#: ``param.F:7408``'s domain centre, which is what ``init3d.F`` assigns.
#: Declaring one without the other would leave the other axis on an implicit
#: default while the document read as though it had placed the bubble.
_BUBBLE_CENTER_KEYS: frozenset[str] = frozenset(
    {"bubble_center_x_m", "bubble_center_y_m"}
)

#: ``writeout.F``'s four environment planes and ``getcape.F``'s own parcel
#: controls.  A named group rather than a bare literal because ``[diagnostics]``
#: holds more than one independent family of CM1 words, and each has to be
#: all-or-none on its own: a document that selects updraft helicity must not be
#: forced to name a parcel adiabat, and one that asks for CAPE must not be
#: forced to name a swath.  ``_DIAGNOSTICS_GROUPS`` below is what enforces that.
_ENVIRONMENT_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "output_cape",
        "output_cin",
        "output_lcl",
        "output_lfc",
        "parcel_source",
        "adiabat",
        "mixed_layer_depth_m",
        "pressure_increment_pa",
        "storm_motion",
    }
)


#: ``misclibs.F``'s rotation diagnostics and ``solve3.F``'s run-long surface
#: swaths.  A second, independent family inside ``[diagnostics]``: a document
#: that wants updraft helicity must not be made to name a parcel adiabat.
_TORNADO_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "output_uh",
        "output_vort",
        "output_pv",
        "output_sws",
        "output_svs",
        "output_sps",
        "output_srs",
        "output_sgs",
        "output_sus",
        "output_shs",
        "tornado_report",
    }
)

#: &param9 words this port spells in another table.  CM1 has one namelist word
#: for each, so ``[output]`` must not offer a second, separately declarable
#: spelling that a document could contradict.
_OUTPUT_KEYS_OWNED_ELSEWHERE: frozenset[str] = frozenset(
    {
        "output_dbz", "output_cape", "output_cin", "output_lcl", "output_lfc",
        # pdcomp's forcings and Laplacian sources are produced by the
        # [lagrangian] machinery, so the switch is declared beside them.
        "output_pdcomp",
    }
) | (_TORNADO_DIAGNOSTIC_KEYS - {"tornado_report"})

#: Which table owns each &param9 word this port spells outside ``[output]``.
_OUTPUT_KEY_OWNER: Mapping[str, str] = MappingProxyType(
    {
        "output_dbz": "physics",
        **{key: "diagnostics" for key in _OUTPUT_KEYS_OWNED_ELSEWHERE - {"output_dbz"}},
    }
)

#: Each independent family of ``[diagnostics]`` keys, with the label its
#: partial-declaration refusal names.  Declaring any key of a family requires
#: declaring all of it; declaring none of a family is CM1's own shipped state
#: for that family.
_DIAGNOSTICS_GROUPS: tuple[tuple[frozenset[str], str], ...] = (
    (_ENVIRONMENT_FIELD_KEYS, "getcape environment field"),
    (_TORNADO_DIAGNOSTIC_KEYS, "rotation diagnostic and surface swath"),
)


#: CM1's ``param2``/``param3`` dynamical words, by CM1's own namelist names.
#: Optional in the mandatory ``[dynamics]`` table for the same reason the
#: ``param6`` vertical words are optional in ``[grid]``: the canonical
#: document of every shipped example is sealed by ``sha256`` and that hash
#: travels inside recorded GPU-run receipts as ``plan_sha256``.  Absent, each
#: binds CM1 r21.1's own ``run/namelist.input`` value, stated below by value.
_DYNAMICS_CM1_WORDS: frozenset[str] = frozenset(
    {
        "coriolis",
        "beta_plane",
        "fcor_s_inv",
        "alph",
        "kdiv",
        "idiff",
        "mdiff",
        "difforder",
        "kdiff2",
        "kdiff6",
    }
)
#: run/namelist.input:29-30,42-44,59-60,92-96.  ``alph`` is README.namelist:
#: 756-759's documented default and the value all 23 shipped decks set.
_DYNAMICS_CM1_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "coriolis": 0,
        "beta_plane": 0,
        "fcor_s_inv": 5.0e-5,
        "alph": 0.60,
        "kdiv": 0.10,
        "idiff": 0,
        "mdiff": 0,
        "difforder": 6,
        "kdiff2": 75.0,
        "kdiff6": 0.040,
    }
)

# CM1's open-boundary companions (README.namelist:595-610,826-829) and its
# lateral Rayleigh sponge (README.namelist:401-403,767-768).  Each group is
# declared whole or not at all, for the same reason the lower boundary is.
_OPEN_COMPANION_KEYS: frozenset[str] = frozenset(
    {"roflux", "nudgeobc", "alphobc", "cstar"}
)
_HORIZONTAL_DAMPING_KEYS: frozenset[str] = frozenset({"hrdamp", "xhd"})

# The remaining sfcmodel=1 options of README.namelist:1037-1038,1072-1078.
# Optional as their own group, for the same reason and with the same rule.
_SURFACE_LAYER_KEYS: frozenset[str] = frozenset(
    {"cnstce", "set_znt", "cnst_znt", "set_ust", "cnst_ust"}
)

# CM1's param12 initial surface conditions (README.namelist:971-1015).  A
# [surface] table declares all seven or none.
_SURFACE_KEYS: frozenset[str] = frozenset(
    {"initsfc", "tsk0_k", "tmn0_k", "xland0", "lu0", "season", "oceanmodel"}
)

# CM1's scale selection, by CM1's own names.  Optional as a pair: absent
# means CM1's LES deck (cm1setup=1, ipbl=0), which is what GM1's Cartesian
# lane has always integrated, and declaring one requires declaring both so
# neither is ever implicit.
_TURBULENCE_SCALE_KEYS: frozenset[str] = frozenset({"cm1setup", "ipbl"})
DEFAULT_TURBULENCE_SCALE = TurbulenceScaleConfig(cm1setup=1, ipbl=0)

#: CM1's ``param11`` words, in the order ``input.F:321`` reads them.  The
#: whole table is optional; ``radopt`` is its only mandatory key when it is
#: declared, because README.namelist:832-834 says nothing else in param11 is
#: read for radopt=0 and cm1_physics_families refuses a declared place or
#: date with the sun switched off.
_RADIATION_KEYS: tuple[str, ...] = (
    "radopt",
    "dtrad_s",
    "ctrlat",
    "ctrlon",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
)
_RADIATION_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "dtrad_s": 0.0,
        "ctrlat": 0.0,
        "ctrlon": 0.0,
        "year": 0,
        "month": 0,
        "day": 0,
        "hour": 0,
        "minute": 0,
        "second": 0,
    }
)

#: Every key of the optional [chain] table (cm1core.arwen_chain).  All required except
#: ``moisture``, whose absence is the all-species lane every chain had before it existed.
_CHAIN_KEYS: frozenset[str] = frozenset(
    {
        "history_directory", "domain_id", "start_time", "end_time", "parent_i0",
        "parent_j0", "refinement_ratio", "spec_zone", "relax_zone",
        "terrain_relief_limit_m", "map_factor_departure_limit", "prepared_directory",
        "moisture",
    }
)

#: Every key of the optional [nest] table.  Declared whole or not at all.
_NEST_KEYS: frozenset[str] = frozenset(
    {
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
        "follow_smoothing_radius_m",
        "follow_corral_m",
        "follow_on_edge",
        "follow_hold_limit",
        "follow_relocation_radius_m",
        # [nest.tornado_criterion], a sub-table (NEST_TORNADO_CRITERION_KEYS).
        "tornado_criterion",
        *(
            "child_nz",
            "child_stretch_z",
            "child_str_bot_m",
            "child_str_top_m",
            "child_dz_bot_m",
            "child_dz_top_m",
            "child_levels_m",
        ),
    }
)

#: The optional keys that give the child its own vertical deck.  CM1's own
#: param6 words (param.F:7814-8139) with a ``child_`` prefix; the model top and
#: the ``dz`` metric reference are the parent's.
NEST_CHILD_VERTICAL_KEYS: tuple[str, ...] = (
    "child_nz",
    "child_stretch_z",
    "child_str_bot_m",
    "child_str_top_m",
    "child_dz_bot_m",
    "child_dz_top_m",
    "child_levels_m",
)

#: ``[nest.tornado_criterion]`` (cm1core.nest_tornado_criterion).  Only stop
#: rules are keys; every threshold is a constant of the criterion module.
NEST_TORNADO_CRITERION_TABLE = "tornado_criterion"
NEST_TORNADO_CRITERION_KEYS_ORDERED: tuple[str, ...] = (
    "stop_after_episode_s",
    "stop_if_no_tornado_by_s",
)
NEST_TORNADO_CRITERION_KEYS: frozenset[str] = frozenset(NEST_TORNADO_CRITERION_KEYS_ORDERED)
#: Sub-tables whose keys the support matrix documents one row each, beside
#: the row for the sub-table itself.
SUBTABLE_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {f"nest.{NEST_TORNADO_CRITERION_TABLE}": NEST_TORNADO_CRITERION_KEYS}
)

#: The storm trackers a following nest may re-centre on.  Both run CM1's own
#: getpsmth/getpcent pressure centre (src/azimavg.F:4071-4163); they differ in
#: what the nest is moved toward.
#:
#: ``cm1-pressure-held-offset`` (the default) keeps the displacement between
#: the nest's centre and CM1's pressure centre that the document placed it at.
#: CM1's pressure centre is the mesocyclone's, and on a mature supercell it is
#: steady while the strongest low-level vorticity sits 9 to 11 km from it
#: (measured on the archived frames of the 100 m storm, commit dca8ab6), so a
#: window placed to hold the surface vortex stays on it as the storm moves.
#:
#: ``cm1-pressure-centre`` re-centres the nest on the pressure centre itself,
#: which is CM1's own adaptive-move target; a small window centred there sits
#: on the mesocyclone aloft and can leave the surface vortex outside it.
NEST_FOLLOW_TRACKERS: frozenset[str] = frozenset(
    {"cm1-pressure-held-offset", "cm1-pressure-centre", "wrf-surface-circulation"}
)
#: ``wrf-surface-circulation`` runs WRF's vortex centre
#: (share/mediation_nest_move.F:352-606) on the child's own lowest-level
#: area-averaged vertical vorticity -- circulation per unit area over a box of
#: ``follow_smoothing_radius_m`` -- so the nest follows the surface vortex
#: rather than the mesocyclone's pressure centre 9-11 km away, and the box
#: mean does not jump to short-lived gust-front sub-vortices
#: (cm1core.nest_move).
NEST_SURFACE_TRACKER = "wrf-surface-circulation"
NEST_FOLLOW_TRACKER_DEFAULT = "cm1-pressure-held-offset"
#: ``nest.follow_on_edge``: ``stop`` (what an undeclared key does) or ``hold``
#: (cm1core.nest_move, THE HOLD POLICY).  Surface tracker only.  The set is
#: the executor's own, so a policy the executor learns is admitted here.
NEST_FOLLOW_ON_EDGE_POLICIES: frozenset[str] = frozenset(_NEST_EDGE_POLICIES)

#: Every key of the optional [ensemble] table.  Declared whole or not at all:
#: a member with an implicit seed could not be re-run, and a seed with an
#: implicit member could not be told apart from another member in a receipt.
_ENSEMBLE_KEYS: frozenset[str] = frozenset({"member", "seed", "theta_amplitude_k"})

_SCHEMA: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "run": frozenset(
            {
                "lane",
                "duration_s",
                "dt_s",
                "history_interval_s",
                "restart_interval_s",
                "irst",
                "rstnum",
                "iconly",
                "restart_reset_frqtim",
            }
        ),
        "grid": frozenset(
            {
                "nx",
                "ny",
                "nz",
                "dx_m",
                "dy_m",
                "dz_m",
                "terrain",
                "stretch_x",
                "stretch_y",
                "stretch_z",
                "ztop_m",
                "str_bot_m",
                "str_top_m",
                "dz_bot_m",
                "dz_top_m",
                "levels_m",
            }
        ),
        "dynamics": frozenset(
            {
                "psolver",
                "adaptive_timestep",
                "scalar_horizontal_order",
                "scalar_vertical_order",
                "velocity_horizontal_order",
                "velocity_vertical_order",
                "scalar_weno",
                "velocity_weno",
                "weno_order",
                "dry_mass_correction",
                "lspgrad",
            }
            | _DYNAMICS_CM1_WORDS
        ),
        "boundaries": frozenset(
            {"west", "east", "south", "north", "bottom", "top", "radiative_order"}
            | _OPEN_COMPANION_KEYS
        ),
        "closures": frozenset(
            {
                "sgs_model",
                "rterm",
                "t2p_avg",
                "vertical_implicit_diffusion",
                "upper_damping",
                "upper_damping_depth_m",
                "upper_damping_rate_s_inv",
                "moving_frame",
                "u_move_m_s",
                "v_move_m_s",
            }
            | _LOWER_BOUNDARY_KEYS
            | _HORIZONTAL_DAMPING_KEYS
            | _SURFACE_LAYER_KEYS
            | _TURBULENCE_SCALE_KEYS
            | _ADAPT_MOVE_KEYS
            | _SGS_RAMP_KEYS
        ),
        "physics": frozenset(
            {
                "component", "equation_set", "output_dbz", "dbz_level_m",
                "ihail", "ndcnst", "cuparam",
            }
        ),
        "initialization": frozenset({"isnd", "iwnd"}) | _INITIALIZATION_OPTIONAL_KEYS,
        "surface": _SURFACE_KEYS,
        "radiation": frozenset(_RADIATION_KEYS),
        # CM1's &param9 history-selection words.  ``output_dbz`` is NOT one of
        # them here: CM1 has a single output_dbz namelist word and this port
        # already ships it as physics.output_dbz, beside the component whose
        # scheme-specific formula computes it and beside dbz_level_m.  Two TOML
        # keys for one namelist word would let a document say false in one
        # table and 1 in the other with nothing to reconcile them, so the
        # &param9 selection follows physics.output_dbz instead of redeclaring
        # it.
        # ``output_dbz`` belongs to [physics], beside the component whose
        # scheme-specific formula computes it; the four getcape planes
        # belong to [diagnostics], beside the parcel controls that decide
        # what they mean.  CM1 has one namelist word for each, so &param9
        # carries them but does not offer a second spelling.
        "output": frozenset(OUTPUT_KEYS) - _OUTPUT_KEYS_OWNED_ELSEWHERE,
        # ``getcape.F``'s environment planes and their parcel controls.
        # Optional as a table; see _DIAGNOSTICS_GROUPS for why its key groups
        # are all-or-none within the table rather than across it.
        "diagnostics": frozenset(_ENVIRONMENT_FIELD_KEYS) | _TORNADO_DIAGNOSTIC_KEYS,
        "lagrangian": LAGRANGIAN_KEYS,
        "runtime": frozenset(
            {
                "acoustic_substeps",
                "tile_nx",
                "tile_ny",
                "warmup_steps",
                "memory_budget_mib",
                "capacity_profile",
                "pdefq_launch",
                "physics_block_nx",
                "physics_block_ny",
            }
        ),
        "nest": _NEST_KEYS,
        "chain": _CHAIN_KEYS,
        "ensemble": _ENSEMBLE_KEYS,
    }
)

# ``_SCHEMA`` is the complete inventory of keys this parser consumes and the
# support matrix documents.  Sections listed here are declared by one lane
# only; every other lane keeps the six mandatory tables, so an unconsumed
# [runtime] table elsewhere still refuses by name through CM1-CFG-UNKNOWN.
_LANE_SCHEMA: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        Lane.CARTESIAN.value: frozenset(
            {
                "runtime", "initialization", "surface", "nest", "ensemble",
                "diagnostics", "lagrangian", "chain",
            }
        )
    }
)
_LANE_ONLY_SECTIONS: frozenset[str] = frozenset().union(*_LANE_SCHEMA.values())
# A section that a lane may declare but need not.  An absent [initialization]
# table binds base.F isnd=7 with the iwnd=0 that param.F:836-852 forces for it,
# which is the only initializer lane this door has ever driven and which every
# receipt names by value.  The table is optional rather than mandatory only so
# that the sealed semantic_config_fixture identity recorded in
# records/CM1-PROFILE-MANIFEST.json does not change; promoting it to mandatory
# is a one-line change there and in the shipped example.
# The [surface] table follows the same rule: absent, it is CM1's own shipped
# deck (run/namelist.input:130-136) and it is not projected, so every
# document sealed before the table existed keeps its exact canonical bytes.
_OPTIONAL_SECTIONS: frozenset[str] = frozenset({"initialization", "surface", "radiation", "output", "nest", "ensemble", "diagnostics", "lagrangian", "chain"})
# CM1's restart selection surface (README.namelist:227-235, 1488-1510).  These
# keys are optional rather than mandatory for the same reason [initialization]
# is: every sealed semantic identity recorded in
# records/CM1-PROFILE-MANIFEST.json predates them, and a document that does not
# declare one hashes exactly as it did before the key existed.
_OPTIONAL_RUN_KEYS: tuple[str, ...] = ("irst", "rstnum", "iconly", "restart_reset_frqtim")

#: CM1's ``param6`` vertical-grid words, in the order ``param.F`` reads them.
#: They are the only optional keys in a mandatory table.  Absent, they are
#: CM1 r21.1's own distributed values -- ``run/namelist.input:174-179`` ships
#: ``stretch_z = 0`` with the four lengths and the model top unread by that
#: branch -- so an undeclared grid is the unstretched one it has always been.
#:
#: They are optional rather than mandatory for one concrete reason: the
#: canonical document of every shipped example is sealed by ``sha256`` and
#: that hash travels inside recorded GPU-run receipts as ``plan_sha256``
#: (``state.py:345``).  Promoting these keys would change documents whose
#: runs are already measured, and a receipt that no longer matches the
#: document it names is worse than an implicit CM1 default that the
#: projection below states by value.
_VERTICAL_GRID_WORDS: tuple[str, ...] = (
    "stretch_z", "ztop_m", "str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m",
    "levels_m",
)
#: input.F:283 / run/namelist.input:174-179.
_VERTICAL_GRID_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "stretch_z": 0,
        "ztop_m": 0.0,
        "str_bot_m": 0.0,
        "str_top_m": 0.0,
        "dz_bot_m": 0.0,
        "dz_top_m": 0.0,
        "levels_m": (),
    }
)
#: README.namelist:1366-1369.  ``output_dbz`` absent is CM1's own "no
#: reflectivity in this run" -- ``param.F:4665`` clamps the namelist word into
#: {0,1} and ``run/config_files/les_ib_windtunnel/namelist.input:246`` ships
#: it at 0 -- and ``dbz_level_m`` absent is "publish the volume and the
#: composite, and no single level".  They are optional for the same reason the
#: vertical-grid words are: every shipped example's canonical document is
#: sealed by ``sha256`` and that hash travels inside recorded GPU-run receipts.
_REFLECTIVITY_WORDS: tuple[str, ...] = ("output_dbz", "dbz_level_m")

# One table, not one per capability: a second ``_OPTIONAL_KEYS`` assignment
# further down this module would shadow the first in silence and make every
# key the first one names mandatory again.
# Keys a section may declare but need not.  An absent key is not an unknown
# key: it binds the value the lane has always executed, and every receipt
# names that value.  ``physics.ihail`` and ``physics.ndcnst`` are optional so
# that the sealed configuration identities recorded in
# records/CM1-PROFILE-MANIFEST.json and pinned by tests do not change because
# CM1's own selectors became declarable; the param6 grid words above are
# optional for the same reason.  Both tables live in this one mapping: a
# second assignment would silently shadow the first and turn an optional key
# back into a mandatory one, refusing every document that omits it.
_OPTIONAL_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "run": frozenset(_OPTIONAL_RUN_KEYS),
        "grid": frozenset(_VERTICAL_GRID_WORDS),
        "physics": frozenset(_REFLECTIVITY_WORDS) | {"ihail", "ndcnst", "cuparam"},
        "initialization": _INITIALIZATION_OPTIONAL_KEYS,
        # ``lspgrad`` is optional without a bound default: absent is CM1's own
        # lspgrad=0 and is not projected (DynamicsConfig.lspgrad).
        "dynamics": _DYNAMICS_CM1_WORDS | {"lspgrad"},
        "boundaries": _OPEN_COMPANION_KEYS,
        "closures": _HORIZONTAL_DAMPING_KEYS | frozenset({"rterm", "t2p_avg"}) | _TURBULENCE_SCALE_KEYS,
        "radiation": frozenset(_RADIATION_KEYS) - {"radopt"},
        # Every [lagrangian] key is individually optional.
        "lagrangian": LAGRANGIAN_KEYS,
        # ``chain.moisture`` is optional: absent is the all-species lane, not projected.
        "chain": frozenset({"moisture"}),
        # Every &param9 word is individually optional inside its table.
        # ``output_dbz`` belongs to [physics], beside the component whose
        # scheme-specific formula computes it; the four getcape planes
        # belong to [diagnostics], beside the parcel controls that decide
        # what they mean.  CM1 has one namelist word for each, so &param9
        # carries them but does not offer a second spelling.
        "output": frozenset(OUTPUT_KEYS) - _OUTPUT_KEYS_OWNED_ELSEWHERE,
        # ``runtime.pdefq_launch`` selects between two raw-word identical
        # launch layouts for CM1's pre-WSM6 positive-definite repair.  It is
        # optional because leaving it out inherits no scientific default:
        # whichever layout runs, every published word is the same.  What it
        # changes is how much of the device the repair occupies.
        # ``runtime.physics_block_nx``/``_ny`` size the WSM6 call; leaving
        # them out follows the closure tile, which is what every document
        # already did before the keys existed.
        # ``runtime.tile_nx``/``_ny`` are optional as a pair: leaving both out
        # runs the closure as one whole-domain tile, which is raw-word
        # identical to any declared tiling (see CartesianRuntimeConfig).
        "runtime": frozenset({
            "pdefq_launch", "physics_block_nx", "physics_block_ny", "tile_nx", "tile_ny",
        }),
        # The tracker a following nest re-centres on; absent is the one that
        # stays on the surface vortex region (NEST_FOLLOW_TRACKER_DEFAULT).
        # follow_smoothing_radius_m is required by, and only by, the surface
        # tracker; follow_corral_m defaults to the placement halo every nest
        # already keeps.  Both are refused when they would act on nothing.
        # follow_on_edge is absent for "stop"; follow_hold_limit and
        # follow_relocation_radius_m are required by, and only by,
        # follow_on_edge = "hold".
        # ... and the child's own vertical deck; absent is the parent's deck.
        # ... and the tornado criterion sub-table; absent is no criterion.
        "nest": frozenset({
            "follow_tracker", "follow_smoothing_radius_m", "follow_corral_m",
            "follow_on_edge", "follow_hold_limit", "follow_relocation_radius_m",
            NEST_TORNADO_CRITERION_TABLE,
        })
        | frozenset(NEST_CHILD_VERTICAL_KEYS),
    }

)


def _dynamics_word(table: Mapping[str, Any], name: str) -> Any:
    """One CM1 ``param2``/``param3`` word, or CM1's own distributed value."""

    if name in table:
        return table[name]
    return _DYNAMICS_CM1_DEFAULTS[name]


#: README.namelist:531-538 and src/solve1.F:590-678, by value.  Each unported
#: branch is refused with the wrong result it would integrate on this lane.
_LSPGRAD_UNPORTED_REASONS: Mapping[int, str] = MappingProxyType(
    {
        2: (
            "solve1.F:614-631 forces with -fcor*(vg+vmove) and fcor*(ug+umove) "
            "from ug/vg profiles this lane's closure does not carry.  CM1 "
            "allocates them zero (cm1.F:1316-1319) and base.F fills them only "
            "for iwnd=8 (base.F:2523-2540) and testcases 3, 7, 11 and 13 "
            "(base.F:2792, 2919, 2940, 2980); for every other base state "
            "base.F:2990-2993 leaves ug=-umove, so the forcing CM1 adds is "
            "exactly zero and the run would be lspgrad=0 under a geostrophic "
            "label, and for iwnd=8 the forcing would be missing altogether"
        ),
        3: (
            "solve1.F:633-659 is the gradient-wind balance of a hurricane "
            "boundary layer (Bryan et al. 2017): it forces with vg and "
            "vg*vg/hurr_rad rotated by param.F:8971-8974's hurr_angle, and this "
            "lane's closure carries neither a vg profile nor hurr_rad (the "
            "iwnd=8 initializer uses hurr_vg only to build u0/v0), so the "
            "forcing would be missing while the receipt named it"
        ),
        4: (
            "solve1.F:661-678 adds ulspg/vlspg, which CM1 allocates zero "
            "(cm1.F:1334-1337) and base.F sets only for testcases 12 and 13 "
            "(base.F:2961-2985) or by editing the source; this lane's closure "
            "carries no such profile, so for every other deck the run would be "
            "lspgrad=0 under a label naming a specified pressure gradient, and "
            "for those two the forcing would be missing altogether"
        ),
    }
)


def _parse_lspgrad(table: Mapping[str, Any]) -> int | None:
    """CM1's ``lspgrad``; only the base-state geostrophic branch is ported."""

    if "lspgrad" not in table:
        return None
    value = _integer("dynamics.lspgrad", table["lspgrad"])
    if value in _LSPGRAD_UNPORTED_REASONS:
        refuse(
            code="CM1-CFG-LSPGRAD-UNPORTED",
            gate="support-matrix",
            reason=_LSPGRAD_UNPORTED_REASONS[value],
            remedy=(
                "use dynamics.lspgrad = 1 (Coriolis on the perturbation from the "
                "base-state wind, solve1.F:590-612) or dynamics.lspgrad = 0"
            ),
            option="dynamics.lspgrad",
            value=value,
        )
    return _integer_choice(
        "dynamics.lspgrad",
        value,
        {0, 1},
        "solve1.F:538 enters the block for lspgrad>=1 but dispatches only on "
        "1, 2, 3 and 4 (solve1.F:590-678), so any other value adds nothing "
        "and would be the lspgrad=0 identity wearing a label",
    )


def _lane_sections(lane: str | None) -> Mapping[str, frozenset[str]]:
    """Return the exact tables one lane must declare, and nothing else."""

    admitted = _LANE_SCHEMA.get(lane, frozenset()) if lane is not None else frozenset()
    return MappingProxyType(
        {
            section: keys
            for section, keys in _SCHEMA.items()
            if section not in _LANE_ONLY_SECTIONS or section in admitted
        }
    )


def _unknown(option: str, value: object) -> NoReturn:
    refuse(
        code="CM1-CFG-UNKNOWN",
        gate="support-matrix",
        reason=(
            "an unconsumed configuration value could select a different CM1 "
            "runtime branch while the port continued with its default"
        ),
        remedy="remove the key or add a source-mapped support-matrix row and validator",
        option=option,
        value=value,
    )


def _strict_sections(raw: Mapping[str, Any], schema: Mapping[str, frozenset[str]]) -> None:
    for section, value in raw.items():
        if section not in schema:
            _unknown(section, value)
        if not isinstance(value, Mapping):
            refuse(
                code="CM1-CFG-SECTION-TYPE",
                gate="config-grammar",
                reason="a non-table section cannot be bound to named CM1 options",
                remedy=f"write [{section}] as a TOML table",
                option=section,
                value=value,
            )
        for key, item in value.items():
            if key not in schema[section]:
                _unknown(f"{section}.{key}", item)
    missing = set(schema) - set(raw) - _OPTIONAL_SECTIONS
    if missing:
        name = sorted(missing)[0]
        refuse(
            code="CM1-CFG-MISSING-SECTION",
            gate="config-grammar",
            reason="the sealed run would otherwise inherit an implicit scientific default",
            remedy=f"add an explicit [{name}] table",
            option=name,
            value=None,
        )


def _require_exact_keys(raw: Mapping[str, Any], schema: Mapping[str, frozenset[str]]) -> None:
    for section, expected in schema.items():
        if section not in raw:
            continue  # only an optional section can be absent here
        for group, label, why in (
            (_OPEN_COMPANION_KEYS, "boundaries", "open-boundary companion"),
            (_HORIZONTAL_DAMPING_KEYS, "closures", "lateral Rayleigh sponge"),
        ):
            if section != label:
                continue
            declared = group & set(raw[section])
            if declared and declared != group:
                key = sorted(group - declared)[0]
                refuse(
                    code="CM1-CFG-BOUNDARY-COMPANION-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        f"a half-declared {why} would take its remaining "
                        "selectors from this port's defaults, so the run would "
                        "integrate a boundary the document does not name"
                    ),
                    remedy=(
                        "declare all of "
                        + ", ".join(f"{label}.{name}" for name in sorted(group))
                        + ", or none of them"
                    ),
                    option=f"{label}.{key}",
                    value=None,
                )
            expected = expected - group
        if section == "closures":
            layer = _SURFACE_LAYER_KEYS & set(raw[section])
            if layer and layer != _SURFACE_LAYER_KEYS:
                key = sorted(_SURFACE_LAYER_KEYS - layer)[0]
                refuse(
                    code="CM1-CFG-SURFACE-LAYER-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "a half-declared surface layer would take the "
                        "remaining getcecd options from CM1's shipped deck, "
                        "so the run would impose a roughness length, a "
                        "friction velocity or an enthalpy coefficient the "
                        "document does not name"
                    ),
                    remedy=(
                        "declare all of "
                        + ", ".join(
                            f"closures.{name}" for name in sorted(_SURFACE_LAYER_KEYS)
                        )
                        + ", or none of them"
                    ),
                    option=f"closures.{key}",
                    value=None,
                )
            declared = _LOWER_BOUNDARY_KEYS & set(raw[section])
            if declared and declared != _LOWER_BOUNDARY_KEYS:
                key = sorted(_LOWER_BOUNDARY_KEYS - declared)[0]
                refuse(
                    code="CM1-CFG-LOWER-BOUNDARY-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "a half-declared lower boundary would take the "
                        "remaining surface selectors from this port's "
                        "free-slip default, so the run would integrate a "
                        "different boundary from the one the document names"
                    ),
                    remedy=(
                        "declare all of "
                        + ", ".join(f"closures.{name}" for name in sorted(_LOWER_BOUNDARY_KEYS))
                        + ", or none of them"
                    ),
                    option=f"closures.{key}",
                    value=None,
                )
            declared_scale = _TURBULENCE_SCALE_KEYS & set(raw[section])
            if declared_scale and declared_scale != _TURBULENCE_SCALE_KEYS:
                key = sorted(_TURBULENCE_SCALE_KEYS - declared_scale)[0]
                refuse(
                    code="CM1-CFG-TURBULENCE-SCALE-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "cm1setup and ipbl are one decision -- which closure "
                        "mixes the column (src/param.F:752-753) -- so a "
                        "half-declared pair would take the other half from "
                        "this port's LES default and the run could end up "
                        "with an LES closure and a PBL scheme both acting, "
                        "or with neither"
                    ),
                    remedy=(
                        "declare both closures.cm1setup and closures.ipbl, "
                        "or neither"
                    ),
                    option=f"closures.{key}",
                    value=None,
                )
            expected = expected - _LOWER_BOUNDARY_KEYS
            expected = expected - _LOWER_BOUNDARY_KEYS - _SURFACE_LAYER_KEYS
            declared = _ADAPT_MOVE_KEYS & set(raw[section])
            if declared and declared != _ADAPT_MOVE_KEYS:
                key = sorted(_ADAPT_MOVE_KEYS - declared)[0]
                refuse(
                    code="CM1-CFG-ADAPT-MOVE-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "closures.do_adapt_move and closures.adapt_move_frq are "
                        "one selection: a controller declared without a cadence "
                        "would re-centre the domain on run/namelist.input:361's "
                        "implicit 3600 s rather than the interval the document "
                        "means, and a cadence declared without the controller is "
                        "a number nothing reads while the frame stays fixed"
                    ),
                    remedy=(
                        "declare both "
                        + " and ".join(
                            f"closures.{name}" for name in sorted(_ADAPT_MOVE_KEYS)
                        )
                        + ", or neither"
                    ),
                    option=f"closures.{key}",
                    value=None,
                )
            expected = expected - _ADAPT_MOVE_KEYS
            declared = _SGS_RAMP_KEYS & set(raw[section])
            if declared and declared != _SGS_RAMP_KEYS:
                key = sorted(_SGS_RAMP_KEYS - declared)[0]
                refuse(
                    code="CM1-CFG-SGS-RAMP-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "closures.ramp_sgs and closures.ramp_time are one "
                        "selection (README.namelist:1081-1087): a ramp declared "
                        "without its duration would ramp over src/cm1.F:356's "
                        "compiled 0 s -- param.F:750 makes that 1e-10 s, no "
                        "ramp at all -- while the document named one"
                    ),
                    remedy=(
                        "declare both closures.ramp_sgs and closures.ramp_time "
                        "(every CM1 deck ships 1 and 1800.), or neither"
                    ),
                    option=f"closures.{key}",
                    value=None,
                )
            expected = expected - _SGS_RAMP_KEYS
        if section == "diagnostics":
            # [diagnostics] carries two independent families of CM1 words --
            # getcape.F's environment planes with their parcel controls, and
            # misclibs.F/solve3.F's rotation diagnostics and swaths.  Each is
            # all-or-none on its own, so a document that wants updraft
            # helicity is not made to name a parcel adiabat and one that wants
            # CAPE is not made to name a swath.  Declaring none of a family is
            # CM1's own shipped state for it.
            for group, label in _DIAGNOSTICS_GROUPS:
                declared_group = group & set(raw[section])
                if declared_group and declared_group != group:
                    key = sorted(group - declared_group)[0]
                    refuse(
                        code="CM1-CFG-DIAGNOSTICS-PARTIAL",
                        gate="config-grammar",
                        reason=(
                            f"a half-declared {label} group would take the rest "
                            "of its selectors from this port's defaults, so the "
                            "run would publish a diagnostic the document did not "
                            "describe while the receipt named the declared half"
                        ),
                        remedy=(
                            f"declare every {label} key ("
                            + ", ".join(f"diagnostics.{name}" for name in sorted(group))
                            + "), or none of them"
                        ),
                        option=f"diagnostics.{key}",
                        value=None,
                    )
            expected = expected - frozenset().union(*(g for g, _ in _DIAGNOSTICS_GROUPS))
        if section == "initialization":
            declared = _BUBBLE_CENTER_KEYS & set(raw[section])
            # Only ``iinit=1`` reads both axes.  ``init3d.F``'s dam-break cold
            # pool (:489), line of bubbles (:521), Straka blob (:654) and
            # Skamarock-Klemp wave (:1348) read ``ric`` alone -- they are
            # uniform in y by construction -- so demanding ``rjc`` beside it
            # would make the x centre of those four branches undeclarable
            # while the geometry they build is perfectly well defined.  An
            # absent ``iinit`` is CM1's own warm bubble, so the pair rule
            # still holds for every document written before this key existed.
            branch = raw[section].get("iinit", _INITIALIZATION_DEFAULTS["iinit"])
            if branch == 1 and declared and declared != _BUBBLE_CENTER_KEYS:
                key = sorted(_BUBBLE_CENTER_KEYS - declared)[0]
                refuse(
                    code="CM1-CFG-BUBBLE-CENTER-PARTIAL",
                    gate="config-grammar",
                    reason=(
                        "a warm bubble moved on one axis only would be placed "
                        "on the other axis at param.F:7408's domain centre, so "
                        "init3d.F:456-457 would build a perturbation the "
                        "document does not describe and the run's geometry "
                        "would be half the declared one"
                    ),
                    remedy=(
                        "declare both "
                        + " and ".join(
                            f"initialization.{name}" for name in sorted(_BUBBLE_CENTER_KEYS)
                        )
                        + ", or neither"
                    ),
                    option=f"initialization.{key}",
                    value=None,
                )
        missing = expected - set(raw[section]) - _OPTIONAL_KEYS.get(section, frozenset())
        if missing:
            key = sorted(missing)[0]
            refuse(
                code="CM1-CFG-MISSING-KEY",
                gate="config-grammar",
                reason="the sealed run would otherwise inherit an implicit scientific default",
                remedy=f"set [{section}] {key} explicitly",
                option=f"{section}.{key}",
                value=None,
            )


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(
            code="CM1-CFG-NUMERIC-TYPE",
            gate="config-grammar",
            reason="a non-numeric grid or clock value has no CM1 arithmetic meaning",
            remedy="supply a finite positive number",
            option=name,
            value=value,
            independent=True,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value  # type: ignore[return-value]  # reached only while refusals are collected
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="a non-positive or non-finite grid/clock value makes the schedule undefined",
            remedy="supply a finite value greater than zero",
            option=name,
            value=value,
            independent=True,
        )
    return result


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(
            code="CM1-CFG-NUMERIC-TYPE",
            gate="config-grammar",
            reason="a non-numeric value cannot be sealed into CM1 arithmetic",
            remedy="supply a finite number",
            option=name,
            value=value,
            independent=True,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value  # type: ignore[return-value]  # reached only while refusals are collected
    result = float(value)
    if not math.isfinite(result):
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="a non-finite value cannot be represented in a run receipt",
            remedy="supply a finite number",
            option=name,
            value=value,
            independent=True,
        )
    # JSON distinguishes -0.0 textually even though CM1/IEEE comparisons do
    # not. Normalize it so one semantic fixed profile has one canonical hash.
    return 0.0 if result == 0.0 else result


def _finite_nonnegative(name: str, value: object) -> float:
    """A CM1 length that may be exactly zero (an unused stretch control)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(
            code="CM1-CFG-NUMERIC-TYPE",
            gate="config-grammar",
            reason="a non-numeric grid length has no CM1 arithmetic meaning",
            remedy="supply a finite number of metres",
            option=name,
            value=value,
            independent=True,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value  # type: ignore[return-value]  # reached only while refusals are collected
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="a negative or non-finite grid length would place a model level below the ground",
            remedy="supply a finite number of metres that is zero or greater",
            option=name,
            value=value,
            independent=True,
        )
    return 0.0 if result == 0.0 else result


def _level_array(name: str, value: object) -> tuple[float, ...]:
    """CM1's ``input_grid_z`` contents, carried inside the sealed document.

    ``param.F:8023`` and ``:8087`` read the levels from a side file.  GM1
    carries them here instead so the run hash covers the vertical coordinate;
    a file the hash does not cover could be edited between a run and its
    restart and silently move every model level.
    """

    if not isinstance(value, (list, tuple)):
        refuse(
            code="CM1-CFG-LEVEL-ARRAY",
            gate="config-grammar",
            reason="explicit vertical levels must be an array of heights in metres",
            remedy="write grid.levels_m as a TOML array of numbers, or leave it undeclared when stretch_z is 0 or 1",
            option=name,
            value=value,
            independent=True,
        )
        return value  # type: ignore[return-value]  # reached only while refusals are collected
    levels: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            refuse(
                code="CM1-CFG-LEVEL-ARRAY",
                gate="config-grammar",
                reason=f"level {index} is not a number, so it has no height",
                remedy="supply finite heights in metres, ascending",
                option=name,
                value=item,
                independent=True,
            )
            continue  # reached only while refusals are collected
        height = float(item)
        if not math.isfinite(height) or height < 0.0:
            refuse(
                code="CM1-CFG-LEVEL-ARRAY",
                gate="config-grammar",
                reason=f"level {index} is negative or non-finite, so it lies below the ground or has no height",
                remedy="supply finite heights in metres, ascending",
                option=name,
                value=item,
                independent=True,
            )
        if levels and height <= levels[-1]:
            refuse(
                code="CM1-CFG-LEVEL-ARRAY",
                gate="config-grammar",
                reason=f"level {index} does not rise above level {index - 1}; param.F:8212-8224 stops CM1 on a non-positive layer depth",
                remedy="supply strictly ascending heights in metres",
                option=name,
                value=item,
                independent=True,
            )
        levels.append(0.0 if height == 0.0 else height)
    return tuple(levels)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        refuse(
            code="CM1-CFG-INTEGER",
            gate="config-grammar",
            reason="the CM1 branch selector or dimension requires an integer in range",
            remedy=f"supply an integer >= {minimum}",
            option=name,
            value=value,
            independent=True,
        )
    return value


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        refuse(
            code="CM1-CFG-BOOLEAN",
            gate="config-grammar",
            reason="integer-like truth values can collide with CM1 branch selectors",
            remedy="use the TOML literal true or false",
            option=name,
            value=value,
            independent=True,
        )
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        refuse(
            code="CM1-CFG-STRING",
            gate="config-grammar",
            reason="a semantic component selector must be an exact string",
            remedy="supply a quoted TOML string",
            option=name,
            value=value,
            independent=True,
        )
    return value


def _optional_integer(
    name: str, section: Mapping[str, Any], *, minimum: int, reason: str
) -> int | None:
    """One optional integer selector, refused by value when it is declared."""

    key = name.split(".", 1)[1]
    if key not in section:
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        refuse(
            code="CM1-CFG-INTEGER",
            gate="config-grammar",
            reason=reason,
            remedy=f"supply an integer >= {minimum}, or remove {name}",
            option=name,
            value=value,
            independent=True,
        )
    return value


def _optional_integer_choice(
    name: str, section: Mapping[str, Any], allowed: set[int], reason: str
) -> int | None:
    key = name.split(".", 1)[1]
    if key not in section:
        return None
    return int(_choice(name, _integer(name, section[key]), set(allowed), reason))


def _optional_boolean(name: str, section: Mapping[str, Any]) -> bool | None:
    key = name.split(".", 1)[1]
    if key not in section:
        return None
    return _boolean(name, section[key])


def _choice(name: str, value: object, allowed: set[object], reason: str) -> object:
    try:
        admitted = value in allowed
    except TypeError:
        admitted = False
    if not admitted:
        refuse(
            code="CM1-CFG-UNSUPPORTED-VALUE",
            gate="support-matrix",
            reason=reason,
            remedy="select one of " + ", ".join(map(repr, sorted(allowed, key=str))),
            option=name,
            value=value,
            independent=True,
        )
    return value


def _integer_choice(
    name: str, value: object, allowed: set[int], reason: str
) -> int:
    return int(_choice(name, _integer(name, value), set(allowed), reason))


def _finite_real(name: str, value: object) -> float:
    """A CM1 ``real`` that may be negative -- ``fcor`` south of the equator."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(
            code="CM1-CFG-NUMERIC-TYPE",
            gate="config-grammar",
            reason="a non-numeric CM1 real has no arithmetic meaning",
            remedy="supply a finite number",
            option=name,
            value=value,
            independent=True,
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value  # type: ignore[return-value]  # reached only while refusals are collected
    result = float(value)
    if not math.isfinite(result):
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason=(
                "a non-finite CM1 real reaches a kernel as NaN or Inf and "
                "makes every word it multiplies non-finite from the first "
                "stage onward"
            ),
            remedy="supply a finite number",
            option=name,
            value=value,
            independent=True,
        )
    return result


def _unit_interval(name: str, value: object, reason: str) -> float:
    """A CM1 weight that ``sound.F`` only has meaning for inside [0, 1]."""

    result = _finite_real(name, value)
    if not 0.0 <= result <= 1.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason=reason,
            remedy=f"set {name} to a value between 0.0 and 1.0 inclusive",
            option=name,
            value=value,
            independent=True,
        )
    return result


def _document_lane(raw: Mapping[str, Any]) -> Lane | None:
    """Admit the lane by value before any section is validated.

    The lane selects which optional tables the document may declare, so an
    unknown lane is refused first; otherwise a lane typo would surface as an
    unknown section and hide the defect that actually selected the branch.
    """

    run = raw.get("run")
    if not isinstance(run, Mapping):
        return None
    value = run.get("lane")
    if not isinstance(value, str):
        return None
    try:
        return Lane(value)
    except ValueError:
        _choice(
            "run.lane",
            value,
            {member.value for member in Lane},
            "an unknown lane could compare substituted physics against the wrong oracle",
        )
        raise AssertionError("unreachable")


def _acoustic_substeps(name: str, value: object) -> int:
    """Admit CM1's own ``nsound`` range, not a shorter enumerated list.

    ``misclibs.F:883-885`` derives an even count of at least four and
    ``misclibs.F:835`` caps it at 192, above which ``misclibs.F:887-911``
    stops the run.  This lane declares the count once instead of rederiving it
    per step, so the same three bounds are checked here, and
    ``_require_cartesian_envelope`` then refuses any (dt, grid, count) triple
    whose small step exceeds CM1's ``dtsm``.
    """

    count = _integer(name, value)
    if (
        count < CM1_MIN_ACOUSTIC_SUBSTEPS
        or count > CM1_MAX_ACOUSTIC_SUBSTEPS
        or count % 2
    ):
        refuse(
            code="CM1-CFG-CARTESIAN-ACOUSTIC-SUBSTEPS",
            gate="cartesian-runtime",
            reason=(
                "sound.F:95 takes the second RK stage's small-step count as "
                "0.5*nsound in integer arithmetic while sound.F:96 keeps "
                "dts=dt/nsound, so an odd count makes that stage's acoustic "
                "integration span (nsound-1)/2*dt/nsound seconds instead of "
                "the dt/2 the stage advances, and the pressure and momentum "
                "it hands the third stage are short of the stage timestep; "
                "misclibs.F:883 never derives fewer than four small steps, "
                "and misclibs.F:835 with misclibs.F:887-911 stops the run "
                f"above {CM1_MAX_ACOUSTIC_SUBSTEPS} rather than take that many"
            ),
            remedy=(
                "declare an even runtime.acoustic_substeps between "
                f"{CM1_MIN_ACOUSTIC_SUBSTEPS} and {CM1_MAX_ACOUSTIC_SUBSTEPS}"
            ),
            option=name,
            value=value,
        )
    return count


#: The explicit closure tile extents a document may declare.
CLOSURE_TILE_EXTENTS: frozenset[int] = frozenset({16, 32, 64, 128})


def _optional_closure_tile(option: str, table: Mapping[str, Any]) -> int | None:
    """One declared closure tile extent, or ``None`` for one whole-domain tile.

    The pair is all-or-none.  Declaring one extent and leaving the other out
    names no decomposition: it could mean a strip of full-domain width or a
    tile whose other side silently inherited a default, and the device
    workspace the closure prices differs by the whole domain between the two.
    A spelling for "untiled" other than leaving both keys out -- ``0``,
    ``"auto"``, the grid extent -- is refused rather than guessed at, so there
    is exactly one way to write each decomposition and one canonical document
    for it.
    """

    key = option.split(".", 1)[1]
    other = "tile_ny" if key == "tile_nx" else "tile_nx"
    if key not in table:
        if other in table:
            refuse(
                code="CM1-CFG-CARTESIAN-TILE-PAIR",
                gate="cartesian-runtime",
                reason=(
                    f"runtime.{other} is declared and {option} is not, so the "
                    "closure decomposition is half-specified: the missing side "
                    "could be the whole domain or a default, and the closure "
                    "workspace priced against the device differs by the full "
                    "domain extent between those readings"
                ),
                remedy=(
                    f"declare {option} beside runtime.{other}, or leave both out "
                    "to run the closure as one whole-domain tile"
                ),
                option=option,
                value=None,
            )
        return None
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value not in CLOSURE_TILE_EXTENTS:
        refuse(
            code="CM1-CFG-UNSUPPORTED-VALUE",
            gate="cartesian-runtime",
            reason=(
                "only the measured closure tile extents 16, 32, 64 and 128 are "
                "admitted; any other value, including a spelling such as 0 or "
                "\"auto\" for an untiled closure, would size the closure "
                "workspace against the device budget for a decomposition nobody "
                "declared"
            ),
            remedy=(
                f"set {option} to 16, 32, 64 or 128, or leave both runtime.tile_nx "
                "and runtime.tile_ny out to run the closure as one whole-domain tile"
            ),
            option=option,
            value=value,
        )
    return int(value)


def _optional_physics_block(option: str, table: Mapping[str, Any]) -> int | None:
    """One WSM6 block extent, or ``None`` to follow the closure tile."""

    from .gm1_runtime import MAX_PHYSICS_BLOCK_CELLS

    key = option.split(".", 1)[1]
    if key not in table:
        return None
    value = _integer(option, table[key], minimum=1)
    if value > MAX_PHYSICS_BLOCK_CELLS:
        refuse(
            code="CM1-CFG-UNSUPPORTED-VALUE",
            gate="cartesian-runtime",
            reason=(
                "one WSM6 call stages five mass volumes and an interface volume "
                "of its own block; a block this wide would fail to allocate "
                "part-way through a large step that has already advanced the "
                "state past the pre-WSM6 seam, where it cannot be re-run"
            ),
            remedy=f"set {option} to at most {MAX_PHYSICS_BLOCK_CELLS}, or leave it out to follow the closure tile",
            option=option,
            value=value,
        )
    return value



def _parse_output(raw: Mapping[str, Any]) -> OutputSelection | None:
    """CM1 ``&param9``, refused by key and by value.

    Every one of CM1's 64 ``output_*`` words is admitted here.  A word this
    lane cannot honour is refused *by value*, and the refusal names the
    concrete wrong result: a field that would be published identically zero
    and read as physics, or a prognostic carrier that would go missing from a
    frame the resume path reads by name.  A word this lane can honour is
    carried through to :class:`~cm1core.gm1_runtime.RuntimeConfig` and changes
    the published frame.
    """

    table = raw.get("output")
    if table is None:
        return None
    values = dict(_OUTPUT_GM1_DEFAULTS)
    declared: list[str] = []
    for key in OUTPUT_KEYS:
        if key not in table:
            continue
        switch = _OUTPUT_SWITCH_BY_KEY[key]
        value = _integer_choice(
            f"output.{key}",
            table[key],
            set(switch.allowed),
            f"CM1 admits only {switch.allowed} for this word ({switch.cm1_ref})",
        )
        if switch.disposition == _OUTPUT_UNSUPPORTED and value != 0:
            refuse(
                code="CM1-OUTPUT-UNSUPPORTED",
                gate="CM1-param9",
                reason=(
                    f"{key}={value} selects CM1 fields "
                    f"{', '.join(switch.fields) or '(see ' + switch.cm1_ref + ')'} "
                    f"that this lane has no operator for: {switch.defect}"
                ),
                remedy=f"set output.{key} = 0",
                option=f"output.{key}",
                value=value,
            )
        if switch.disposition == _OUTPUT_STATE and value != switch.gm1_default:
            refuse(
                code="CM1-OUTPUT-STATE-CARRIER",
                gate="CM1-param9",
                reason=(
                    f"{key}={value} would change a field this lane publishes "
                    f"unconditionally as {', '.join(switch.carried_by)}: "
                    f"{switch.defect}"
                ),
                remedy=f"set output.{key} = {switch.gm1_default}",
                option=f"output.{key}",
                value=value,
            )
        values[key] = value
        declared.append(key)
    try:
        return OutputSelection(values=values, declared=tuple(declared))
    except ValueError as error:
        refuse(
            code="CM1-OUTPUT-SELECTION",
            gate="CM1-param9",
            reason=f"the &param9 selection is not admissible: {error}",
            remedy="correct the named output_* word",
            option="output",
            value=None,
        )


def _validate_output_against_lane(config: "CM1Config") -> None:
    """Refuse a selection the *rest* of this document cannot produce.

    These are cross-checks, not key checks: the word is admissible on its own
    and inadmissible beside another word in the same document.
    """

    from .cm1_budget_terms import refuse_inadmissible_budgets

    refuse_inadmissible_budgets(config, refuse=refuse)
    selection = config.output
    if selection is None and config.tornado_diagnostics is None and config.diagnostics is None:
        return

    def declared(key: str) -> int:
        """The value this document actually selects for one &param9 word.

        Five of CM1's &param9 words are spelled in the table that owns the
        thing they describe rather than in ``[output]``, so reading the
        ``[output]`` selection alone would check a default where the document
        declared something else -- and a cross-check that reads the wrong
        value is worse than none.
        """

        owner = _OUTPUT_KEY_OWNER.get(key)
        if owner == "physics":
            return int(config.physics.output_dbz)
        if owner == "diagnostics":
            for table in (config.diagnostics, config.tornado_diagnostics):
                if table is not None and hasattr(table, key):
                    return int(getattr(table, key))
            return 0
        return 0 if selection is None else int(selection[key])

    def where(key: str) -> str:
        return f"{_OUTPUT_KEY_OWNER.get(key, 'output')}.{key}"

    if config.physics.component == "off":
        for key in ("output_rain", "output_dbz", "output_srs", "output_sgs",
                    "output_cape", "output_cin", "output_lcl", "output_lfc",
                    "output_lwp", "output_pwat", "output_qvpert"):
            if declared(key):
                refuse(
                    code="CM1-OUTPUT-DRY-LANE",
                    gate="CM1-param9",
                    reason=(
                        f"{key}=1 names a moist field, but physics.component is "
                        "'off' and this run carries no water substance; the "
                        "published plane would be identically zero and read as "
                        "a storm that produced no rain, no echo and no CAPE"
                    ),
                    remedy=f"set {where(key)} = 0, or select a moist physics.component",
                    option=where(key),
                    value=declared(key),
                )
    if config.grid.stretch_z != 0:
        # Only calcvort's horizontal components carry the vertical metric
        # (misclibs.F:1622-1683, :1855-1898: rdz*mf).  calcuh's no-terrain
        # branch (misclibs.F:1451-1459, :1512-1530) and the svs/shs swaths
        # (solve3.F:885-894, :1100-1110) use only rdx*uf and rdy*vf plus the
        # column's own zh/zf heights, which the runtime passes from the
        # generated stretched column; they are admitted on stretch_z != 0.
        for key in ("output_vort", "output_pv"):
            if declared(key):
                refuse(
                    code="CM1-OUTPUT-STRETCHED-METRIC",
                    gate="CM1-param9",
                    reason=(
                        f"{key}=1 selects a calcvort product, but "
                        f"grid.stretch_z={config.grid.stretch_z} builds a "
                        "non-uniform column, and calcvort's xvort/yvort carry "
                        "CM1's rdz*mf vertical metric (misclibs.F:1622-1683). "
                        "The &param9 transcription in cuda_cm1_output_diagnostics "
                        "binds mh/mf to one (param.F:8170); the producer this lane "
                        "actually runs for [diagnostics] (tornado_diagnostics) "
                        "carries mh/mf, but its stretched branch has no CM1 check "
                        "yet, so a stretched xvort/yvort would be published under "
                        "CM1's names unverified"
                    ),
                    remedy=f"set {where(key)} = 0, or set grid.stretch_z = 0",
                    option=where(key),
                    value=declared(key),
                )


def _parse_runtime(raw: Mapping[str, Any]) -> CartesianRuntimeConfig:
    from .cuda_wsm6_pdefq import WSM6_PDEFQ_LAYOUTS
    from .engine_capacity import WSM6_256_CAPACITY_PROFILE

    t = raw["runtime"]
    return CartesianRuntimeConfig(
        acoustic_substeps=_acoustic_substeps(
            "runtime.acoustic_substeps", t["acoustic_substeps"]
        ),
        tile_nx=_optional_closure_tile("runtime.tile_nx", t),
        tile_ny=_optional_closure_tile("runtime.tile_ny", t),
        warmup_steps=_integer("runtime.warmup_steps", t["warmup_steps"], minimum=0),
        memory_budget_mib=_integer(
            "runtime.memory_budget_mib", t["memory_budget_mib"], minimum=256
        ),
        capacity_profile=str(
            _choice(
                "runtime.capacity_profile",
                _string("runtime.capacity_profile", t["capacity_profile"]),
                {"default", WSM6_256_CAPACITY_PROFILE},
                "an unpinned column capacity could overrun per-thread WSM6 arrays",
            )
        ),
        pdefq_launch=(
            None
            if "pdefq_launch" not in t
            else str(
                _choice(
                    "runtime.pdefq_launch",
                    _string("runtime.pdefq_launch", t["pdefq_launch"]),
                    set(WSM6_PDEFQ_LAYOUTS),
                    "an unadmitted layout would regroup misclibs.F:1082's "
                    "left-associated west-to-east budj carry or "
                    "misclibs.F:1065-1081's per-column k order and publish an "
                    "asq moisture-correction ledger that differs from the "
                    "authority in its trailing bits",
                )
            )
        ),
        physics_block_nx=_optional_physics_block("runtime.physics_block_nx", t),
        physics_block_ny=_optional_physics_block("runtime.physics_block_ny", t),
    )


def _parse_initialization(raw: Mapping[str, Any]) -> InitializationConfig | None:
    """Admit CM1's own base.F and init3d.F initialization selectors.

    ``isnd``: 5 is base.F:363-470's analytic Weisman-Klemp sounding, 7 is
    base.F:495-607's ``input_sounding`` lane, and 17 is that same file read for
    thermodynamics only, with the base-state wind taken from ``iwnd``
    (param.F:849-851).  ``iwnd`` is base.F:2263-2650's eleven branches; 7 is
    the one number CM1's chain has no arm for.  ``iinit`` is
    init3d.F:450-1416's perturbation chain.  Every other key names one word of
    CM1's own parameter blocks and is optional; absent, it is CM1's own
    distributed value.
    """

    if "initialization" not in raw:
        return None
    t = raw["initialization"]
    declared = frozenset(_INITIALIZATION_OPTIONAL_KEYS & set(t))

    # The branch has to be known before any geometry word is resolved: CM1
    # assigns its perturbation constants inside the iinit arm, so what an
    # absent key means depends on which arm was declared.
    branch = _integer_choice(
        "initialization.iinit",
        t["iinit"] if "iinit" in t else _INITIALIZATION_DEFAULTS["iinit"],
        {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13},
        "init3d.F:450-1416's arms are 0 (undisturbed), 1 (warm bubble), 2 (dam-break cold pool), 3 (line of bubbles), 4 (Bryan-Fritsch moist benchmark bubble), 5 (Straka cold blob), 7 (tropical-cyclone vortex), 8 (line thermal with random noise), 9 (forced convergence), 10 (momentum forcing), 11 (Skamarock-Klemp IG wave), 12 (updraft nudging) and 13 (near-surface theta wave); the chain has no else arm, so any other value would start the run undisturbed under a receipt naming a trigger",
    )

    def word(name: str) -> Any:
        return _initialization_word(t, name, branch)

    def length(name: str) -> float:
        return _finite(f"initialization.{name}", word(name))

    def positive_length(name: str) -> float:
        return _finite_positive(f"initialization.{name}", word(name))

    def optional_position(name: str) -> float | None:
        value = word(name)
        if value is None:
            return None
        return _finite(f"initialization.{name}", value)

    random_seed = _integer("initialization.random_seed", word("random_seed"))
    if random_seed < 0 or random_seed >= 2**64:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="the irandp seed crosses the initializer ABI as an unsigned 64-bit word",
            remedy="declare initialization.random_seed in 0..2**64-1",
            option="initialization.random_seed",
            value=word("random_seed"),
        )
    random_generator = _choice(
        "initialization.random_generator",
        _string("initialization.random_generator", word("random_generator")),
        {"gm1", "cm1"},
        "the perturbation streams are GM1's own seeded splitmix64 ('gm1') and CM1's own "
        "libgfortran xoshiro256** as init3d.F:427-440 seeds it ('cm1'); any other name "
        "would draw from one of them under a receipt naming neither",
    )
    if random_generator == "cm1" and random_seed != 0:
        refuse(
            code="CM1-CFG-RANDOM-SEED-UNREAD",
            gate="config-grammar",
            reason="CM1's own random stream is seeded by init3d.F:427-440's fixed scramble; the declared seed would be recorded in the receipt and never read",
            remedy="drop initialization.random_seed, or select random_generator = \"gm1\" to draw from GM1's stream with that seed",
            option="initialization.random_seed",
            value=random_seed,
        )
    irandp = _integer_choice(
        "initialization.irandp",
        word("irandp"),
        {0, 1},
        "README.namelist:692-695 admits irandp=0 or 1; init3d.F's chain has no else arm, so any other value would add no perturbation at all and the run would stay laminar under a receipt naming a random field",
    )
    if irandp == 1 and random_seed == 0 and random_generator == "gm1":
        refuse(
            code="CM1-CFG-RANDOM-SEED-MISSING",
            gate="config-grammar",
            reason="init3d.F:416-443 seeds one generator for the whole run; with no recorded seed the perturbation field, and therefore every storm that grows out of it, could not be reproduced from the receipt that describes it, and CM1's own alternative (init3d.F:417, use_truly_random_pert) reseeds from the wall clock",
            remedy="declare a nonzero initialization.random_seed beside irandp = 1, or random_generator = \"cm1\" for CM1's own numbers",
            option="initialization.random_seed",
            value=0,
        )
    maintain_rh = word("maintain_rh")
    if type(maintain_rh) is not bool:
        refuse(
            code="CM1-CFG-BOOLEAN-TYPE",
            gate="config-grammar",
            reason="init3d.F:465 maintain_rh is a Fortran logical; a non-boolean has no branch",
            remedy="declare initialization.maintain_rh = true or false",
            option="initialization.maintain_rh",
            value=maintain_rh,
        )
    # init3d.F:548-550 writes the iinit = 3 line's first centre as the literal
    # rjc = 3000 m; GM1 moves it with bubble_first_center_y_m, or (as before
    # that literal was carried, and as the Settings screen writes it beside
    # bubble_center_x_m) with bubble_center_y_m.  Both at once would record two
    # places and use one.
    if branch == 3 and {"bubble_center_y_m", "bubble_first_center_y_m"} <= declared:
        refuse(
            code="CM1-CFG-INIT-WORD-TWICE",
            gate="config-grammar",
            reason=(
                "initialization.bubble_center_y_m and initialization.bubble_first_center_y_m both place the "
                "first bubble of the iinit = 3 line (init3d.F:548's rjc); the receipt would record two places "
                "and the run would use one"
            ),
            remedy="declare one of them",
            option="initialization.bubble_first_center_y_m",
            value=t["bubble_first_center_y_m"],
        )
    # init3d.F:1355-1356 gives the iinit = 11 wave a literal 5 km envelope half
    # width, which GM1 sets with half_width_m; bhrad_m is the bubble's radius
    # and this arm does not read it.
    if branch == 11 and "bhrad_m" in declared:
        refuse(
            code="CM1-CFG-INIT-WORD-UNREAD",
            gate="config-grammar",
            reason=(
                "initialization.bhrad_m is not read by iinit = 11: init3d.F:1355-1356 gives the wave a 5 km "
                "envelope half width, which GM1 sets with initialization.half_width_m; the declared radius would "
                "be recorded in the receipt and the run would ignore it"
            ),
            remedy="declare initialization.half_width_m instead, or leave both undeclared for CM1's 5 km",
            option="initialization.bhrad_m",
            value=t["bhrad_m"],
        )
    if branch == 7 and int(word("iorigin")) != 2:
        refuse(
            code="CM1-CFG-VORTEX-IORIGIN",
            gate="config-grammar",
            reason="init3d.F:763-771 takes the vortex's reference radii from xhref(nx/2+i), which are radii only in param.F:7212-7213's centred frame (iorigin = 2); with the corner origin every point near the vortex centre falls inside the first radius and init3d.F:1083 reads rref(0), outside the array",
            remedy="declare initialization.iorigin = 2 with iinit = 7, as CM1's own hurricane decks do",
            option="initialization.iorigin",
            value=word("iorigin"),
        )
    if branch == 7:
        # init3d.F:800-806: CM1 stops an iinit=7 run with ibalance /= 0, because
        # the vortex's own gradient-wind pressure is the initial pressure.
        if int(word("ibalance")) != 0:
            refuse(
                code="CM1-CFG-VORTEX-IBALANCE",
                gate="config-grammar",
                reason="init3d.F:800-806 stops an iinit=7 run with ibalance /= 0: the vortex brings its own balanced pressure, and a second balance would replace it",
                remedy="declare initialization.ibalance = 0 with iinit = 7",
                option="initialization.ibalance",
                value=word("ibalance"),
            )
        if maintain_rh:
            refuse(
                code="CM1-CFG-VORTEX-MAINTAIN-RH",
                gate="config-grammar",
                reason="init3d.F:1424-1441 would reset the vortex's own vapour (init3d.F:895-966 holds it at the environment's relative humidity in the vortex's temperature) to the base relative humidity; that pairing is not transcribed here",
                remedy="leave initialization.maintain_rh false with iinit = 7, as CM1's own arm does",
                option="initialization.maintain_rh",
                value=True,
            )
    if branch == 4 and maintain_rh:
        refuse(
            code="CM1-CFG-INIT-WORD-UNREAD",
            gate="config-grammar",
            reason="init3d.F:579-647 never sets maintain_rh: the moist benchmark arm holds its own relative humidity, and init3d.F:1422-1441's rescaling after it would replace the saturated bubble's vapour with another field",
            remedy="leave initialization.maintain_rh false with iinit = 4, as CM1's own arm does",
            option="initialization.maintain_rh",
            value=True,
        )
    isnd = _integer_choice(
        "initialization.isnd",
        t["isnd"],
        set(_ADMITTED_ISND),
        "base.F:134-2036's sounding branches are isnd = 1-5, 7-15 and 17-23; 6 stops CM1 ('isnd = 6 code has been removed', base.F:455-460), 15 (DYCOMS-II) runs only with testcase = 4 or 5 (base.F:1491-1510), whose constants.F:78-98 changes rd, cp and xlv for the whole model while GM1's kernels and ArWen's physics are built with the default constants, and any other value has no arm, so base.F would leave the base state at its allocation zero",
    )
    imoist_word = word("imoist")
    if isnd in (2, 3) and imoist_word == 1:
        refuse(
            code="CM1-CFG-ISND-DRY",
            gate="config-grammar",
            reason=(
                f"param.F:1501-1515 stops CM1 for isnd = {isnd} with imoist = 1 ('For this value of isnd, "
                "imoist must be 0'): the dry isothermal (2) and constant-lapse (3) soundings carry no vapour "
                "and base.F has no moist branch for them"
            ),
            remedy="declare initialization.imoist = 0 with this sounding (GM1 builds it for run.iconly = 1; "
                   "the Cartesian large step is moist only), or choose another isnd",
            option="initialization.isnd",
            value=isnd,
        )
    thec_mb_k = _finite_positive("initialization.thec_mb_k", word("thec_mb_k"))
    qt_mb = _finite_positive("initialization.qt_mb", word("qt_mb"))
    if qt_mb >= 1.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="base.F:268's qt_mb is a total water mixing ratio in kg/kg; 1 or more is not a sounding",
            remedy="declare initialization.qt_mb between 0 and 1 (CM1: 0.020)",
            option="initialization.qt_mb",
            value=qt_mb,
        )
    for key, readers in (("thec_mb_k", (4,)), ("qt_mb", (4,)), ("tsk0_k", (11, 21))):
        if key in declared and isnd not in readers:
            refuse(
                code="CM1-CFG-INIT-WORD-UNREAD",
                gate="config-grammar",
                reason=(
                    f"initialization.{key} is read only by base.F's isnd = {' and '.join(map(str, readers))} "
                    f"(param.F:6335-6336 zeroes the moist benchmark's words for every other sounding); with isnd = "
                    f"{isnd} the declared value would be recorded in the receipt and never read"
                ),
                remedy=f"drop initialization.{key}, or select isnd = {readers[0]}",
                option=f"initialization.{key}",
                value=t[key],
            )
    tsk0_word = word("tsk0_k")
    tsk0_k = None if tsk0_word is None else _finite(f"initialization.tsk0_k", tsk0_word)
    if tsk0_k is not None and tsk0_k < 100.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason="base.F:1204 and :1849 build the surface air from tsk0 in kelvin; below 100 K it is not a temperature this lane can hold",
            remedy="declare initialization.tsk0_k in kelvin (CM1: 299.28)",
            option="initialization.tsk0_k",
            value=tsk0_k,
        )
    testcase = _integer_choice(
        "initialization.testcase",
        word("testcase"),
        set(range(16)) - {4, 5},
        "param.F reads testcase in 0..15 (README.namelist:216-232), and an unlisted value falls through base.F's "
        "and init3d.F's else arms and would silently select the default deck under another name; testcase = 4 "
        "and 5 make constants.F:78-98 set rd = 287.0, cp = 1015.0 or 1004.0 and xlv = 2.47e6 or 2.50e6 (and "
        "constants.F:161-164 lv1 = xlv, lv2 = 0) for the whole CM1 run, while GM1's base state, dynamics kernels "
        "and ArWen's physics are built with constants.F's defaults (rd = 287.04, cp = 1005.7, xlv = 2.501e6), so "
        "such a run would carry other thermodynamics than the CM1 run its receipt names",
    )
    # param.F:1044-1072: CM1 resets imoist for five test cases without
    # stopping (1 and 2 run dry, 3, 7 and 8 run moist; 4 and 5 are refused
    # above).  A document that says otherwise would build and integrate a
    # different atmosphere from the CM1 run its receipt names.
    forced_imoist = 0 if testcase in (1, 2) else 1 if testcase in (3, 7, 8) else None
    if forced_imoist is not None and imoist_word in (0, 1) and imoist_word != forced_imoist:
        refuse(
            code="CM1-CFG-TESTCASE-IMOIST",
            gate="config-grammar",
            reason=(
                f"param.F:{'1044-1057' if forced_imoist == 0 else '1059-1072'} sets imoist to "
                f"{forced_imoist} for testcase = {testcase} ('... setting imoist to {forced_imoist} ...'), so "
                f"CM1 would run this document {'dry' if forced_imoist == 0 else 'moist'} while GM1 built and "
                f"integrated it with imoist = {imoist_word}"
            ),
            remedy=f"declare initialization.imoist = {forced_imoist} with testcase = {testcase}, as CM1 does",
            option="initialization.imoist",
            value=imoist_word,
        )
    forcing_words = _parse_forcing_words(t, branch)
    return InitializationConfig(
        isnd=isnd,
        iwnd=_integer_choice(
            "initialization.iwnd",
            t["iwnd"],
            {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11},
            "base.F:2286-2640 implements these eleven wind branches and no else arm; iwnd=7 does not exist upstream, and any unimplemented value would leave base.F:2263-2280's zero wind in place while the receipt named a shear profile",
        ),
        iinit=branch,
        irst=_integer_choice(
            "initialization.irst",
            word("irst"),
            {0, 1},
            "param.F:891-903 reads irst as 0 or 1 and forces iinit=0 on a restart",
        ),
        imoist=_integer_choice(
            "initialization.imoist",
            word("imoist"),
            {0, 1},
            "README.namelist:308 admits imoist=0 or 1; any other value leaves base.F:389-397's rh0 and base.F:592-609's qvsnd undefined, so the base state would carry no defined moisture",
        ),
        testcase=testcase,
        ibalance=_integer_choice(
            "initialization.ibalance",
            word("ibalance"),
            {0, 1, 2},
            "README.namelist:697-705 and init3d.F:1735-1917 implement ibalance=0 (initial pressure perturbation exactly zero, which is what all 23 shipped CM1 decks select), 1 (hydrostatic top-down integration) and 2 (the anelastic buoyancy pressure, init3d.F:1893's poiss); the chain has no else arm, so any other value would leave the pressure at zero under a receipt naming a balance",
        ),
        irandp=irandp,
        random_seed=random_seed,
        random_generator=random_generator,
        iorigin=_integer_choice(
            "initialization.iorigin",
            word("iorigin"),
            {1, 2},
            "param.F:7212-7213 places the horizontal origin at the domain corner (1) or its centre (2); any other value would leave xh/yh at the corner while every absolute-coordinate parameter in the document -- ric, rjc, the IG wave centre -- was written for a different frame",
        ),
        bubble_center_x_m=optional_position("bubble_center_x_m"),
        bubble_center_y_m=optional_position("bubble_center_y_m"),
        zc_m=length("zc_m"),
        bhrad_m=positive_length("bhrad_m"),
        bvrad_m=positive_length("bvrad_m"),
        bptpert_k=length("bptpert_k"),
        maintain_rh=maintain_rh,
        nbub=_integer("initialization.nbub", word("nbub")),
        bubble_first_center_y_m=length("bubble_first_center_y_m"),
        bubble_spacing_y_m=length("bubble_spacing_y_m"),
        zdep_m=positive_length("zdep_m"),
        half_width_m=positive_length("half_width_m"),
        random_line_amplitude_k=length("random_line_amplitude_k"),
        hurr_vg_m_per_s=length("hurr_vg_m_per_s"),
        hurr_rotate_deg=length("hurr_rotate_deg"),
        thec_mb_k=thec_mb_k,
        qt_mb=qt_mb,
        tsk0_k=tsk0_k,
        **forcing_words,
        declared=declared,
    )


def _parse_forcing_words(t: Mapping[str, Any], branch: int) -> dict[str, Any]:
    """init3d.F:1303-1396's forcing words, declared or CM1's own values.

    CM1 itself checks none of them; the checks here name the division or the
    empty window each bad value would produce at run time.
    """

    words: dict[str, Any] = {}
    for name in _FORCING_KEYS:
        value = t[name] if name in t else _INITIALIZATION_DEFAULTS[name]
        if value is None and name in ("xc_wnudge", "yc_wnudge", "xc_uforce"):
            words[name] = None
            continue
        words[name] = _finite(f"initialization.{name}", value)
    positive = {
        "xr_wnudge": "misclibs.F:571-573 divides by it (rxrwnudge); zero or less has no ellipsoid",
        "yr_wnudge": "misclibs.F:571-573 divides by it (ryrwnudge); zero or less has no ellipsoid",
        "zr_wnudge": "misclibs.F:571-573 divides by it (rzrwnudge); zero or less has no ellipsoid",
        "zdeep": "misclibs.F:478 divides by it; zero or less forces no layer",
        "lamx": "init3d.F:1328 and misclibs.F:480-481 divide by its square",
        "lamy": "init3d.F:1328 and misclibs.F:482 divide by its square",
        "xr_uforce": "solve1.F:1258-1259 divides by it; zero or less forces no box",
        "zr_uforce": "solve1.F:1258-1260 divides by it; zero or less forces no box",
    }
    for name, why in positive.items():
        if words[name] <= 0.0:
            refuse(
                code="CM1-CFG-FORCING-RADIUS",
                gate="config-grammar",
                reason=f"initialization.{name} must be positive: {why}",
                remedy=f"declare a positive initialization.{name}, or leave it undeclared for CM1's own value",
                option=f"initialization.{name}",
                value=words[name],
            )
    # t2 <= t1 is admitted, as CM1 runs it: sound.F:569 (mtime+dt <= t2_wnudge)
    # and solve1.F:1245 (mtime < t2_uforce) stop the forcing before mtime
    # reaches t1, so misclibs.F:557-561 / solve1.F:1247-1251 keep gamm = 1 and
    # never divide by t2 - t1: the forcing acts at full strength and stops at
    # t2 without a ramp.
    return words


def _parse_surface_layer(c: Mapping[str, Any]) -> SurfaceLayerConfig | None:
    """Bind CM1's remaining five ``sfcmodel=1`` options, or leave the deck."""

    if not (_SURFACE_LAYER_KEYS & set(c)):
        return None
    return SurfaceLayerConfig(
        cnstce=_finite("closures.cnstce", c["cnstce"]),
        set_znt=_integer("closures.set_znt", c["set_znt"]),
        cnst_znt=_finite("closures.cnst_znt", c["cnst_znt"]),
        set_ust=_integer("closures.set_ust", c["set_ust"]),
        cnst_ust=_finite("closures.cnst_ust", c["cnst_ust"]),
    )


def _parse_surface(raw: Mapping[str, Any]) -> SurfaceConfig | None:
    """Bind CM1's ``param12`` initial surface conditions, or leave the deck."""

    table = raw.get("surface")
    if table is None:
        return None
    return SurfaceConfig(
        initsfc=_integer("surface.initsfc", table["initsfc"]),
        tsk0_k=_finite("surface.tsk0_k", table["tsk0_k"]),
        tmn0_k=_finite("surface.tmn0_k", table["tmn0_k"]),
        xland0=_finite("surface.xland0", table["xland0"]),
        lu0=_integer("surface.lu0", table["lu0"]),
        season=_integer("surface.season", table["season"]),
        oceanmodel=_integer("surface.oceanmodel", table["oceanmodel"]),
    )


def _parse_turbulence_scale(c: Mapping[str, Any]) -> TurbulenceScaleConfig | None:
    """Bind CM1's ``cm1setup``/``ipbl`` pair, or leave CM1's LES deck.

    ``_require_exact_keys`` has already refused a half-declared pair, so
    either both keys are present or neither is.  The range and cross-checks
    belong to :func:`cm1core.cm1_physics_families.validate_turbulence_scale`,
    which names the wrong result each combination would produce.
    """

    if not (_TURBULENCE_SCALE_KEYS & set(c)):
        return None
    return TurbulenceScaleConfig(
        cm1setup=_integer("closures.cm1setup", c["cm1setup"]),
        ipbl=_integer("closures.ipbl", c["ipbl"]),
    )


def _parse_radiation(raw: Mapping[str, Any]) -> RadiationConfig | None:
    """Bind CM1's ``param11`` table, or leave the run without a sun.

    README.namelist:832-834 -- "You do not need to set anything here unless
    radopt >= 1" -- is why every key but ``radopt`` is optional and defaults
    to zero.  ``cm1_physics_families.validate_radiation_selection`` then
    refuses a declared latitude, longitude, date or cadence with radiation
    off, so a receipt never carries a place the run did not have.
    """

    if "radiation" not in raw:
        return None
    table = raw["radiation"]

    def word(name: str) -> Any:
        return table[name] if name in table else _RADIATION_DEFAULTS[name]

    return RadiationConfig(
        radopt=_integer("radiation.radopt", table["radopt"]),
        dtrad_s=_finite_nonnegative("radiation.dtrad_s", word("dtrad_s")),
        ctrlat=_finite("radiation.ctrlat", word("ctrlat")),
        ctrlon=_finite("radiation.ctrlon", word("ctrlon")),
        year=_integer("radiation.year", word("year")),
        month=_integer("radiation.month", word("month")),
        day=_integer("radiation.day", word("day")),
        hour=_integer("radiation.hour", word("hour")),
        minute=_integer("radiation.minute", word("minute")),
        second=_integer("radiation.second", word("second")),
    )


def _parse_ensemble(raw: Mapping[str, Any]) -> EnsembleConfig | None:
    """Admit one ensemble member by key and by value."""

    if "ensemble" not in raw:
        return None
    t = raw["ensemble"]
    member = _integer("ensemble.member", t["member"], minimum=0)
    seed = _integer("ensemble.seed", t["seed"], minimum=0)
    if seed > 0xFFFFFFFFFFFFFFFF:
        refuse(
            code="CM1-CFG-ENSEMBLE-SEED",
            gate="config-grammar",
            reason=(
                "an ensemble seed drives a 64-bit generator; a wider value "
                "would be truncated and two documents naming different seeds "
                "would produce the same member"
            ),
            remedy="declare ensemble.seed below 2**64",
            option="ensemble.seed",
            value=seed,
        )
    amplitude = _finite("ensemble.theta_amplitude_k", t["theta_amplitude_k"])
    if amplitude <= 0.0:
        refuse(
            code="CM1-CFG-ENSEMBLE-AMPLITUDE",
            gate="config-grammar",
            reason=(
                f"ensemble.theta_amplitude_k={amplitude} perturbs nothing. "
                "src/init3d.F:1466 sets CM1's own amplitude to 0.25 K and "
                "src/init3d.F:1479 adds amplitude*(2*rand-1) to tha, so a zero "
                "or negative amplitude would make this member the exact "
                "trajectory it branched from while every receipt named it as an "
                "ensemble member; the unperturbed control is the run that "
                "declares no [ensemble] table at all"
            ),
            remedy=(
                "declare a positive ensemble.theta_amplitude_k in kelvin "
                f"({CM1_IRANDP_AMPLITUDE_K} is CM1's own), or remove the "
                "[ensemble] table to run the control"
            ),
            option="ensemble.theta_amplitude_k",
            value=amplitude,
        )
    return EnsembleConfig(member=member, seed=seed, theta_amplitude_k=amplitude)




def _parse_chain(raw: Mapping[str, Any]) -> ChainConfig | None:
    """Admit the optional ArWen chain table, by key and by value.

    The checks that need the ArWen history itself (projection, spacing,
    terrain, cadence, scheme) are :func:`cm1core.arwen_chain.admit_frames`;
    the ones that need the rest of this document are
    :func:`cm1core.arwen_chain.validate_chain_document`.
    """

    if "chain" not in raw:
        return None
    from .arwen_chain import parse_wrf_time

    t = raw["chain"]
    times = {}
    for key in ("start_time", "end_time"):
        text = _string(f"chain.{key}", t[key])
        try:
            times[key] = parse_wrf_time(text)
        except ValueError:
            refuse(
                code="CM1-CFG-CHAIN-TIME",
                gate="config-grammar",
                reason=(
                    f"chain.{key} = {text!r} is not a WRF Times string; the run would be "
                    "matched to history frames by a time nobody wrote"
                ),
                remedy="spell it as ArWen's history does, YYYY-MM-DD_HH:MM:SS (UTC)",
                option=f"chain.{key}",
                value=text,
            )
    if times["end_time"] <= times["start_time"]:
        refuse(
            code="CM1-CFG-CHAIN-TIME",
            gate="config-grammar",
            reason="chain.end_time is not after chain.start_time, so no boundary interval exists",
            remedy="declare an end_time at least one ArWen history interval after start_time",
            option="chain.end_time",
            value=[t["start_time"], t["end_time"]],
        )
    ratio = _integer("chain.refinement_ratio", t["refinement_ratio"], minimum=1)
    if ratio < 3 or ratio % 2 == 0:
        refuse(
            code="CM1-CFG-CHAIN-RATIO",
            gate="config-grammar",
            reason=(
                f"chain.refinement_ratio = {ratio}: the capture operator's coincident faces and "
                "block-mean identity (records/CM1-NEST-SEAM.md 5.1) hold only for an odd ratio "
                "of 3 or more; an even ratio places no GM1 face on an ArWen face, so the normal "
                "flux forced through the GM1 boundary would not be ArWen's"
            ),
            remedy="declare an odd chain.refinement_ratio (3, 5, 7, ...)",
            option="chain.refinement_ratio",
            value=ratio,
        )
    spec_zone = _integer("chain.spec_zone", t["spec_zone"], minimum=1)
    relax_zone = _integer("chain.relax_zone", t["relax_zone"], minimum=0)
    if relax_zone < 2 or relax_zone <= spec_zone:
        refuse(
            code="CM1-CFG-CHAIN-ZONES",
            gate="config-grammar",
            reason=(
                f"chain.spec_zone = {spec_zone} and chain.relax_zone = {relax_zone}: WRF's "
                "relaxation ramp divides by relax_zone-1 and relaxes rows spec_zone..relax_zone-1 "
                "(dyn_em/module_bc_em.F:1311-1322), so a zone no wider than the specified one "
                "relaxes nothing and the specified rows would reflect every outgoing wave"
            ),
            remedy="declare relax_zone > spec_zone >= 1 (WRF's real-data default is 1 and 5)",
            option="chain.relax_zone",
            value=[spec_zone, relax_zone],
        )
    return ChainConfig(
        history_directory=_string("chain.history_directory", t["history_directory"]),
        domain_id=_integer("chain.domain_id", t["domain_id"], minimum=1),
        start_time=str(t["start_time"]),
        end_time=str(t["end_time"]),
        parent_i0=_integer("chain.parent_i0", t["parent_i0"], minimum=0),
        parent_j0=_integer("chain.parent_j0", t["parent_j0"], minimum=0),
        refinement_ratio=ratio,
        spec_zone=spec_zone,
        relax_zone=relax_zone,
        terrain_relief_limit_m=_finite_nonnegative(
            "chain.terrain_relief_limit_m", t["terrain_relief_limit_m"]
        ),
        map_factor_departure_limit=_finite_nonnegative(
            "chain.map_factor_departure_limit", t["map_factor_departure_limit"]
        ),
        prepared_directory=_string("chain.prepared_directory", t["prepared_directory"]),
        moisture=_chain_moisture(t.get("moisture")),
    )


def _chain_moisture(value: object) -> str | None:
    """``chain.moisture``: absent, ``all-species`` or ``vapour-only`` (``arwen_chain.MOISTURE_LANES``)."""

    if value is None:
        return None
    from .arwen_chain import MOISTURE_LANES

    if not isinstance(value, str) or value not in MOISTURE_LANES:
        refuse(
            code="CM1-CFG-CHAIN-MOISTURE",
            gate="config-grammar",
            reason=(
                f"chain.moisture = {value!r} names no moisture lane; the run would take the "
                "history's species by a rule no receipt names"
            ),
            remedy="declare \"all-species\" (the history ran GM1's scheme) or \"vapour-only\" "
                   "(WRF's have_bcs_moist = .false.)",
            option="chain.moisture",
            value=value,
        )
    return value


def _parse_nest(raw: Mapping[str, Any]) -> NestConfig | None:
    """Admit the optional refined child grid, by key and by value.

    Every refusal below names the concrete wrong result it prevents.  The
    cross-field checks that need the parent's own grid and clock live in
    :func:`_validate_nest`, which runs after the whole document is bound.
    """

    if "nest" not in raw:
        return None
    t = raw["nest"]
    ratio = _integer("nest.refinement_ratio", t["refinement_ratio"], minimum=1)
    if ratio < 3:
        refuse(
            code="CM1-CFG-NEST-RATIO",
            gate="support-matrix",
            reason=(
                f"nest.refinement_ratio={ratio} refines nothing: a ratio of 1 "
                "gives the child the parent's own spacing, so the run would "
                "allocate and integrate a second domain of identical "
                "resolution whose only difference from the parent is that its "
                "laterals are specified and reflecting instead of CM1's open "
                "radiative boundary (README.namelist:561-575)"
            ),
            remedy="declare an odd nest.refinement_ratio of 3 or more",
            option="nest.refinement_ratio",
            value=ratio,
        )
    if ratio % 2 == 0:
        refuse(
            code="CM1-CFG-NEST-RATIO-PARITY",
            gate="support-matrix",
            reason=(
                f"nest.refinement_ratio={ratio} is even. CM1 is an Arakawa C "
                "grid (src/cm1.F:744-777 allocates u, v and w on their own "
                "faces). The r child faces nearest a parent u face are "
                "symmetric about it -- offsets -(r-1)/2..+(r-1)/2 child cells "
                "-- only when r is odd; for an even r the block is displaced "
                "by half a child cell, so two-way feedback would write into "
                "the parent's u at x a mean of the child's u at x - dx/2, and "
                "every fed-back momentum value would carry that systematic "
                "upstream shift for the whole run. An even ratio also leaves "
                "no child column coincident with the parent column, so no "
                "parent value survives the round trip unchanged"
            ),
            remedy="declare an odd nest.refinement_ratio (3, 5, 7, ...)",
            option="nest.refinement_ratio",
            value=ratio,
        )
    spec_zone = _integer("nest.spec_zone", t["spec_zone"], minimum=0)
    if spec_zone < 1:
        refuse(
            code="CM1-CFG-NEST-SPEC-ZONE",
            gate="support-matrix",
            reason=(
                "nest.spec_zone=0 leaves the child no row whose tendency the "
                "parent sets outright, so the child's outermost row would be "
                "advanced by its own fifth-order stencil "
                "(dynamics.scalar_horizontal_order=5 reads three cells either "
                "side, src/cm1.F:744-777) reading ghost words nothing fills; "
                "the nest would integrate its boundary from uninitialised "
                "memory rather than from the parent"
            ),
            remedy="declare nest.spec_zone = 1 or more",
            option="nest.spec_zone",
            value=spec_zone,
        )
    relax_zone = _integer("nest.relax_zone", t["relax_zone"], minimum=0)
    if relax_zone < 2:
        refuse(
            code="CM1-CFG-NEST-RELAX-ZONE",
            gate="support-matrix",
            reason=(
                f"nest.relax_zone={relax_zone} carries no Davies ramp: the "
                "weight is (spec_zone + relax_zone - loop)/(relax_zone - 1), "
                "which is undefined at 1 and empty at 0. With no ramp the "
                "child's forcing steps discontinuously from fully specified to "
                "fully free in one cell, and a specified boundary with no "
                "relaxation reflects the storm's own gravity waves and the "
                "psolver=3 acoustic modes straight back into the refined "
                "domain"
            ),
            remedy="declare nest.relax_zone = 4 or another width of at least 2",
            option="nest.relax_zone",
            value=relax_zone,
        )
    if relax_zone <= spec_zone:
        refuse(
            code="CM1-CFG-NEST-ZONE-ORDER",
            gate="support-matrix",
            reason=(
                f"nest.relax_zone={relax_zone} does not exceed "
                f"nest.spec_zone={spec_zone}, so every row the relaxation "
                "would act on is already specified outright and the declared "
                "relaxation applies to nothing; the receipt would name a "
                "Davies zone the run never evaluated"
            ),
            remedy="declare nest.relax_zone strictly greater than nest.spec_zone",
            option="nest.relax_zone",
            value=[spec_zone, relax_zone],
        )
    feedback = _integer_choice(
        "nest.feedback",
        t["feedback"],
        {0, 1},
        "only one-way (0: the parent forces the child and reads nothing back) "
        "and two-way (1: the child's interior block means replace the parent's "
        "covered cells) are implemented; another value would run one-way while "
        "the receipt named a feedback direction the run did not use",
    )
    follow_storm = _boolean("nest.follow_storm", t["follow_storm"])
    follow_interval_s = _finite_nonnegative(
        "nest.follow_interval_s", t["follow_interval_s"]
    )
    follow_max_speed_m_s = _finite_nonnegative(
        "nest.follow_max_speed_m_s", t["follow_max_speed_m_s"]
    )
    follow_tracker = t.get("follow_tracker", NEST_FOLLOW_TRACKER_DEFAULT)
    if follow_tracker not in NEST_FOLLOW_TRACKERS:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-TRACKER",
            gate="support-matrix",
            reason=(
                f"nest.follow_tracker={follow_tracker!r} names no implemented "
                "tracker; the nest would be moved toward a centre nothing "
                "computes while the receipt named it"
            ),
            remedy="declare one of " + ", ".join(sorted(NEST_FOLLOW_TRACKERS)),
            option="nest.follow_tracker",
            value=follow_tracker,
        )
    smoothing_raw = t.get("follow_smoothing_radius_m")
    corral_raw = t.get("follow_corral_m")
    follow_smoothing_radius_m = (
        None if smoothing_raw is None
        else _finite_positive("nest.follow_smoothing_radius_m", smoothing_raw)
    )
    follow_corral_m = (
        None if corral_raw is None else _finite_positive("nest.follow_corral_m", corral_raw)
    )
    if follow_storm and follow_tracker == NEST_SURFACE_TRACKER and follow_smoothing_radius_m is None:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-SMOOTHING",
            gate="support-matrix",
            reason=(
                "nest.follow_tracker = \"wrf-surface-circulation\" averages the "
                "lowest-level vorticity over a box before it looks for the "
                "vortex, and the box radius is what keeps it from jumping to a "
                "short-lived gust-front sub-vortex; there is no scientific "
                "default for it, and a guessed one would name a tracker whose "
                "robustness nobody chose"
            ),
            remedy=(
                "declare nest.follow_smoothing_radius_m, the half-width in metres "
                "of the circulation box (larger than the sub-vortices to be "
                "ignored, smaller than the child's interior)"
            ),
            option="nest.follow_smoothing_radius_m",
            value=None,
        )
    if follow_smoothing_radius_m is not None and (
        not follow_storm or follow_tracker != NEST_SURFACE_TRACKER
    ):
        refuse(
            code="CM1-CFG-NEST-FOLLOW-SMOOTHING",
            gate="support-matrix",
            reason=(
                "nest.follow_smoothing_radius_m is declared but no surface-vortex "
                "tracker runs (follow_storm = false, or a CM1 pressure tracker is "
                "selected); the receipt would name a circulation box the run "
                "never evaluated"
            ),
            remedy=(
                "drop nest.follow_smoothing_radius_m, or select "
                "nest.follow_tracker = \"wrf-surface-circulation\" with "
                "follow_storm = true"
            ),
            option="nest.follow_smoothing_radius_m",
            value=[follow_storm, follow_tracker, follow_smoothing_radius_m],
        )
    if follow_corral_m is not None and not follow_storm:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-INACTIVE",
            gate="support-matrix",
            reason=(
                "nest.follow_corral_m is declared but nest.follow_storm is false; "
                "a fixed nest never moves, so the receipt would name a corral no "
                "move was ever checked against"
            ),
            remedy="drop nest.follow_corral_m, or set nest.follow_storm = true",
            option="nest.follow_corral_m",
            value=follow_corral_m,
        )
    if not follow_storm and (follow_interval_s != 0.0 or follow_max_speed_m_s != 0.0):
        refuse(
            code="CM1-CFG-NEST-FOLLOW-INACTIVE",
            gate="support-matrix",
            reason=(
                "nest.follow_storm is false but a re-centring cadence or step "
                "is declared; the nest would stay where it was placed while the "
                "receipt named a cadence at which it never moved, so a reader "
                "would believe a fixed nest had tracked the storm"
            ),
            remedy=(
                "set nest.follow_storm = true, or declare "
                "nest.follow_interval_s = 0.0 and nest.follow_max_speed_m_s = 0.0"
            ),
            option="nest.follow_storm",
            value=[follow_storm, follow_interval_s, follow_max_speed_m_s],
        )
    if follow_storm and (follow_interval_s <= 0.0 or follow_max_speed_m_s <= 0.0):
        refuse(
            code="CM1-CFG-NEST-FOLLOW-ACTIVE",
            gate="support-matrix",
            reason=(
                "nest.follow_storm is true but the re-centring cadence or the "
                "permitted step is zero, so the tracker would run and the nest "
                "would never move; the refined mesh would stay behind a "
                "translating storm and resolve clear air while the receipt "
                "claimed a storm-following nest"
            ),
            remedy=(
                "declare a positive nest.follow_interval_s and "
                "nest.follow_max_speed_m_s"
            ),
            option="nest.follow_interval_s",
            value=[follow_interval_s, follow_max_speed_m_s],
        )
    follow_on_edge, follow_hold_limit, follow_relocation_radius_m = _parse_nest_follow_on_edge(
        t, follow_storm=follow_storm, follow_tracker=follow_tracker
    )
    return NestConfig(
        refinement_ratio=ratio,
        time_refinement_ratio=_integer(
            "nest.time_refinement_ratio", t["time_refinement_ratio"], minimum=1
        ),
        parent_i0=_integer("nest.parent_i0", t["parent_i0"], minimum=0),
        parent_j0=_integer("nest.parent_j0", t["parent_j0"], minimum=0),
        parent_span_x=_integer("nest.parent_span_x", t["parent_span_x"], minimum=1),
        parent_span_y=_integer("nest.parent_span_y", t["parent_span_y"], minimum=1),
        spec_zone=spec_zone,
        relax_zone=relax_zone,
        feedback=feedback,
        acoustic_substeps=_acoustic_substeps(
            "nest.acoustic_substeps", t["acoustic_substeps"]
        ),
        follow_storm=follow_storm,
        follow_interval_s=follow_interval_s,
        follow_max_speed_m_s=follow_max_speed_m_s,
        follow_tracker=follow_tracker,
        follow_smoothing_radius_m=follow_smoothing_radius_m,
        follow_corral_m=follow_corral_m,
        follow_on_edge=follow_on_edge,
        follow_hold_limit=follow_hold_limit,
        follow_relocation_radius_m=follow_relocation_radius_m,
        child_vertical=_parse_nest_child_vertical(t),
        tornado_criterion=_parse_nest_tornado_criterion(t),
    )


def _parse_nest_follow_on_edge(
    t: Mapping[str, Any], *, follow_storm: bool, follow_tracker: str
) -> tuple[str | None, int | None, float | None]:
    """``nest.follow_on_edge``, ``follow_hold_limit``, ``follow_relocation_radius_m``.

    All three absent is ``stop``.  No CM1 counterpart: CM1 r21.1 has no nesting
    (its only moving mesh is ``src/azimavg.F:4376-4528``).  The authority for
    ``hold`` is WRF's moving nest, which does not stop on a search-window edge
    maximum (``share/mediation_nest_move.F:471-487``) and cancels a move into
    the corral and keeps integrating (``:849-859``); GM1 bounds holding by
    ``follow_hold_limit`` and then relocates within
    ``follow_relocation_radius_m`` (``cm1core.nest_move``, THE HOLD POLICY).
    The radius is checked against the search reach in :func:`_validate_nest`.
    """

    on_edge_raw = t.get("follow_on_edge")
    limit_raw = t.get("follow_hold_limit")
    radius_raw = t.get("follow_relocation_radius_m")
    if on_edge_raw is not None:
        on_edge_raw = _string("nest.follow_on_edge", on_edge_raw)
        if on_edge_raw not in NEST_FOLLOW_ON_EDGE_POLICIES:
            refuse(
                code="CM1-CFG-NEST-FOLLOW-ON-EDGE",
                gate="support-matrix",
                reason=(
                    f"nest.follow_on_edge={on_edge_raw!r} names no implemented "
                    "policy; a tracker maximum on the search-window edge, a lost "
                    "vortex or a move into the corral would be handled by a rule "
                    "the receipt did not name"
                ),
                remedy="declare \"stop\" (or leave the key out) or \"hold\"",
                option="nest.follow_on_edge",
                value=on_edge_raw,
            )
        if not follow_storm or follow_tracker != NEST_SURFACE_TRACKER:
            refuse(
                code="CM1-CFG-NEST-FOLLOW-ON-EDGE",
                gate="support-matrix",
                reason=(
                    "nest.follow_on_edge is declared but no surface-vortex tracker "
                    "runs (follow_storm = false, or a CM1 pressure tracker is "
                    "selected); the policy acts only on wrf-surface-circulation's "
                    "search window, so the receipt would name an edge policy the "
                    "run never evaluated"
                ),
                remedy=(
                    "drop nest.follow_on_edge, or select nest.follow_tracker = "
                    "\"wrf-surface-circulation\" with follow_storm = true"
                ),
                option="nest.follow_on_edge",
                value=[follow_storm, follow_tracker, on_edge_raw],
            )
    if on_edge_raw == "hold" and limit_raw is None:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-HOLD-LIMIT",
            gate="support-matrix",
            reason=(
                "nest.follow_on_edge = \"hold\" keeps the previous centre while the "
                "vortex is out of the tracker's reach and cancels moves into the "
                "corral; with no limit a nest that lost its storm would hold for "
                "the rest of the run under a storm-following receipt, and there "
                "is no scientific default for how long a hold may last"
            ),
            remedy=(
                "declare nest.follow_hold_limit, the consecutive held tracker "
                "samples (follow intervals) after which the tracker relocates"
            ),
            option="nest.follow_hold_limit",
            value=None,
        )
    if limit_raw is not None and on_edge_raw != "hold":
        refuse(
            code="CM1-CFG-NEST-FOLLOW-HOLD-LIMIT",
            gate="support-matrix",
            reason=(
                "nest.follow_hold_limit is declared but nest.follow_on_edge is not "
                "\"hold\", so every edge maximum, lost vortex and corral hit stops "
                "the run at once; the receipt would name a hold limit no sample "
                "was ever counted against"
            ),
            remedy=(
                "drop nest.follow_hold_limit, or declare nest.follow_on_edge = \"hold\""
            ),
            option="nest.follow_hold_limit",
            value=[on_edge_raw, limit_raw],
        )
    if on_edge_raw == "hold" and radius_raw is None:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-RELOCATION",
            gate="support-matrix",
            reason=(
                "nest.follow_on_edge = \"hold\" relocates the tracked centre to the "
                "strongest circulation near it once follow_hold_limit samples have "
                "been held; with no radius the relocation would either search "
                "nothing beyond the window that just failed or the whole child, "
                "where another storm's circulation may win, and there is no "
                "scientific default for how far a storm's vortex may be sought"
            ),
            remedy=(
                "declare nest.follow_relocation_radius_m, the half-width in metres "
                "searched around the kept centre, beyond "
                "follow_max_speed_m_s * follow_interval_s"
            ),
            option="nest.follow_relocation_radius_m",
            value=None,
        )
    if radius_raw is not None and on_edge_raw != "hold":
        refuse(
            code="CM1-CFG-NEST-FOLLOW-RELOCATION",
            gate="support-matrix",
            reason=(
                "nest.follow_relocation_radius_m is declared but nest.follow_on_edge "
                "is not \"hold\", so no sample is ever held and no relocation is "
                "ever searched; the receipt would name a radius the run never "
                "evaluated"
            ),
            remedy=(
                "drop nest.follow_relocation_radius_m, or declare "
                "nest.follow_on_edge = \"hold\" with nest.follow_hold_limit"
            ),
            option="nest.follow_relocation_radius_m",
            value=[on_edge_raw, radius_raw],
        )
    limit = (
        None if limit_raw is None
        else _integer("nest.follow_hold_limit", limit_raw, minimum=1)
    )
    radius = (
        None if radius_raw is None
        else _finite_positive("nest.follow_relocation_radius_m", radius_raw)
    )
    return on_edge_raw, limit, radius


def _parse_nest_tornado_criterion(t: Mapping[str, Any]) -> NestTornadoCriterionConfig | None:
    """Admit ``[nest.tornado_criterion]`` by key and by value.

    The cross-field checks that need the child's grid, its vertical deck and
    the run's clock live in :func:`_validate_nest_tornado_criterion`.
    """

    if NEST_TORNADO_CRITERION_TABLE not in t:
        return None
    prefix = f"nest.{NEST_TORNADO_CRITERION_TABLE}"
    table = t[NEST_TORNADO_CRITERION_TABLE]
    if not isinstance(table, Mapping):
        refuse(
            code="CM1-CFG-SECTION-TYPE",
            gate="config-grammar",
            reason=(
                f"{prefix} must be a TOML table (empty to sample the criterion with no "
                "stop rule); a scalar there declares neither the criterion nor a stop rule, "
                "so whether the run was sampled would rest on a value that means nothing"
            ),
            remedy=f"write [{prefix}] as a table, empty or with its stop-rule keys",
            option=prefix,
            value=table,
        )
    for key in sorted(table):
        if key not in NEST_TORNADO_CRITERION_KEYS:
            from .nest_tornado_criterion import CRITERION, FIXED_THRESHOLD_NAMES

            if key in FIXED_THRESHOLD_NAMES:
                refuse(
                    code="CM1-CFG-NEST-TORNADO-CRITERION-FIXED",
                    gate="config-grammar",
                    reason=(
                        f"{prefix}.{key} names a threshold of the fixed tornado criterion "
                        f"({dict(CRITERION)!r}); a threshold set per document would "
                        "let two receipts carrying the same TORNADO verdict mean different "
                        "vortices, and a threshold tuned after a run would decide the "
                        "verdict from the answer"
                    ),
                    remedy=(
                        f"remove {prefix}.{key}; the criterion's thresholds are constants of "
                        "cm1core.nest_tornado_criterion, and only stop_after_episode_s and "
                        "stop_if_no_tornado_by_s are keys"
                    ),
                    option=f"{prefix}.{key}",
                    value=table[key],
                )
            _unknown(f"{prefix}.{key}", table[key])
    return NestTornadoCriterionConfig(**{
        key: None if key not in table else _finite_positive(f"{prefix}.{key}", table[key])
        for key in NEST_TORNADO_CRITERION_KEYS_ORDERED
    })


@dataclass(frozen=True)
class NestChildVerticalConfig:
    """The child's own CM1 vertical deck (``param.F:7814-8139``).

    The model top and the ``dz`` metric reference are the parent's: a child
    with another top would leave parent air without a child level or fill
    child levels by CM1's constant extension, and
    ``nest_vertical.vertical_refinement_defect`` refuses both that and a child
    coarser than its parent at any height.
    """

    nz: int
    stretch_z: int
    str_bot_m: float
    str_top_m: float
    dz_bot_m: float
    dz_top_m: float
    levels_m: tuple[float, ...]


def _parse_nest_child_vertical(t: Mapping[str, Any]) -> NestChildVerticalConfig | None:
    declared = [key for key in NEST_CHILD_VERTICAL_KEYS if key in t]
    if not declared:
        return None
    missing = [key for key in ("child_nz", "child_stretch_z") if key not in t]
    if missing:
        refuse(
            code="CM1-CFG-NEST-CHILD-VERTICAL",
            gate="support-matrix",
            reason=(
                f"[nest] declares {declared} but not {missing}; a child deck "
                "without its level count or its stretch_z branch would fall back "
                "to the parent's column while the document named lengths of its own"
            ),
            remedy=(
                "declare nest.child_nz and nest.child_stretch_z with the lengths "
                "that branch reads, or none of the nest.child_* keys"
            ),
            option=f"nest.{missing[0]}",
            value=declared,
        )
    return NestChildVerticalConfig(
        nz=_integer("nest.child_nz", t["child_nz"], minimum=7),
        stretch_z=_integer_choice(
            "nest.child_stretch_z",
            t["child_stretch_z"],
            {1, 3, 4},
            "a child deck is one of CM1 r21.1's generated branches (param.F:7814-8139); "
            "stretch_z=0 fixes its top at dz*nk (param.F:2608) and would need a child "
            "dz_m of its own, and stretch_z=2 is refused for every grid",
        ),
        str_bot_m=_finite_nonnegative("nest.child_str_bot_m", t.get("child_str_bot_m", 0.0)),
        str_top_m=_finite_nonnegative("nest.child_str_top_m", t.get("child_str_top_m", 0.0)),
        dz_bot_m=_finite_nonnegative("nest.child_dz_bot_m", t.get("child_dz_bot_m", 0.0)),
        dz_top_m=_finite_nonnegative("nest.child_dz_top_m", t.get("child_dz_top_m", 0.0)),
        levels_m=_level_array("nest.child_levels_m", t.get("child_levels_m", ())),
    )


def nest_child_vertical_coordinate(config: "CM1Config") -> "VerticalLevels | None":
    """Build the child's declared deck by the rule the parent's is held to.

    ``None`` when the child carries the parent's deck.  Every refusal
    :func:`require_vertical_coordinate` raises for a grid is raised for the
    child too, renamed onto the ``nest.child_*`` key that caused it.
    """

    from .errors import CM1Refusal, fail_fast

    nest = config.nest
    if nest is None or nest.child_vertical is None:
        return None
    declared = nest.child_vertical
    parent_levels = require_vertical_coordinate(config)
    grid = replace(
        config.grid,
        nz=declared.nz,
        stretch_z=declared.stretch_z,
        ztop_m=parent_levels.ztop_m if declared.stretch_z == 1 else 0.0,
        str_bot_m=declared.str_bot_m,
        str_top_m=declared.str_top_m,
        dz_bot_m=declared.dz_bot_m,
        dz_top_m=declared.dz_top_m,
        levels_m=declared.levels_m,
    )
    try:
        # The refusal is translated to the child's own option names below, so
        # it has to be raised here even while refusals are being collected.
        with fail_fast():
            return require_vertical_coordinate(replace(config, grid=grid))
    except CM1Refusal as refusal:
        detail = refusal.detail
        option = detail.option or "grid.stretch_z"
        refuse(
            code="CM1-CFG-NEST-CHILD-VERTICAL-DECK",
            gate=detail.gate,
            reason=(
                "the child's declared vertical deck is refused by the rule every "
                f"grid is held to ({detail.code}): {detail.reason}"
            ),
            remedy=detail.remedy.replace("grid.", "nest.child_"),
            option=option.replace("grid.", "nest.child_"),
            value=detail.value,
        )


def _parse_diagnostics(raw: Mapping[str, Any]) -> DiagnosticsConfig | None:
    """Admit CM1's four ``output_*`` environment fields and getcape's controls.

    The table is optional as a whole and mandatory key by key once declared:
    ``_require_exact_keys`` has already refused a partial one, so a document
    that names the table names every switch it turns on and every parcel
    control it lifts with.  An absent table is ``writeout.F:2895``'s own
    shipped state, in which no column sweep runs.
    """

    if "diagnostics" not in raw:
        return None
    t = raw["diagnostics"]
    # [diagnostics] holds more than one independent family; a document that
    # declares only the rotation diagnostics and swaths has not declared this
    # one, and _require_exact_keys has already refused a half-declared group.
    if not (_ENVIRONMENT_FIELD_KEYS & set(t)):
        return None
    return DiagnosticsConfig(
        output_cape=_boolean("diagnostics.output_cape", t["output_cape"]),
        output_cin=_boolean("diagnostics.output_cin", t["output_cin"]),
        output_lcl=_boolean("diagnostics.output_lcl", t["output_lcl"]),
        output_lfc=_boolean("diagnostics.output_lfc", t["output_lfc"]),
        parcel_source=_integer_choice(
            "diagnostics.parcel_source",
            t["parcel_source"],
            {1, 2, 3},
            "getcape.F:242-250 stops CM1 on any other source; an unrecognized "
            "parcel would leave kmax at whatever the previous column left it and "
            "lift a parcel from an arbitrary level, so every published CAPE, CIN, "
            "LCL and LFC would describe a parcel this document never named. "
            "writeout.F:2921 uses 3, the mixed-layer parcel",
        ),
        adiabat=_integer_choice(
            "diagnostics.adiabat",
            t["adiabat"],
            {1, 2, 3, 4},
            "getcape.F:419-423 stops CM1 on any other formulation; an "
            "unrecognized adiabat would leave the condensate neither removed nor "
            "carried, so the lifted parcel would follow neither a pseudoadiabat "
            "nor a reversible adiabat and the published CAPE would belong to no "
            "thermodynamic path. getcape.F:73 ships 1, pseudoadiabatic with "
            "liquid only",
        ),
        mixed_layer_depth_m=_finite_positive(
            "diagnostics.mixed_layer_depth_m", t["mixed_layer_depth_m"]
        ),
        pressure_increment_pa=_finite_positive(
            "diagnostics.pressure_increment_pa", t["pressure_increment_pa"]
        ),
        storm_motion=_choice(
            "diagnostics.storm_motion",
            _string("diagnostics.storm_motion", t["storm_motion"]),
            {"bunkers-right", "bunkers-left"},
            "storm-relative helicity is a line integral taken against one storm "
            "motion, and CM1 r21.1 computes none anywhere in its source tree. An "
            "unnamed or unimplemented method would publish a helicity that cannot "
            "be reproduced or compared with any other estimate, and the sign of "
            "the reported SRH flips between the right and left mover. Only "
            "Bunkers et al. (2000) internal dynamics is transcribed",
        ),
    )


def _parse_tornado_diagnostics(
    raw: Mapping[str, Any]
) -> TornadoDiagnosticsConfig | None:
    """Admit CM1's rotation/swath ``output_*`` switches, or leave them off.

    Each switch is CM1's own 0/1 integer.  ``param.F:4677`` clamps the
    namelist value with ``max(0,min(1,output_uh))``, so CM1 would silently
    turn a 2 into a 1 and a -1 into a 0 and run a branch the namelist did not
    name; this door refuses instead, because a receipt that records ``2``
    beside a field CM1 computed under ``1`` describes a run nobody made.
    """

    if "diagnostics" not in raw:
        return None
    t = raw["diagnostics"]
    # [diagnostics] holds more than one independent family of CM1 words; a
    # document that declares only the getcape planes has not declared this
    # one, and _require_exact_keys has already refused a half-declared group.
    if not (_TORNADO_DIAGNOSTIC_KEYS & set(t)):
        return None
    switches = {}
    for name in (
        "output_uh", "output_vort", "output_pv", "output_sws", "output_svs",
        "output_sps", "output_srs", "output_sgs", "output_sus", "output_shs",
    ):
        switches[name] = _integer_choice(
            f"diagnostics.{name}",
            t[name],
            {0, 1},
            "CM1 r21.1 param.F:4677 clamps this switch with max(0,min(1,...)), "
            "so any other value would select the same branch as 0 or 1 while "
            "the sealed document recorded a number CM1 never ran",
        )
    return TornadoDiagnosticsConfig(
        **switches,
        tornado_report=_boolean("diagnostics.tornado_report", t["tornado_report"]),
    )


def _parse_lagrangian(raw: Mapping[str, Any]) -> LagrangianConfig | None:
    """Bind the optional ``[lagrangian]`` table, or leave it absent.

    ``_strict_sections`` has already refused any key this table does not
    declare, so every refusal raised here is a refusal by *value*.
    """

    if "lagrangian" not in raw:
        return None
    try:
        return parse_lagrangian_table(raw["lagrangian"])
    except LagrangianConfigError as error:
        refuse(
            code="CM1-CFG-LAGRANGIAN-VALUE",
            gate="config-grammar",
            reason=str(error),
            remedy="correct the named [lagrangian] value",
            option="lagrangian",
            value=None,
        )


def _parse_lower_boundary(c: Mapping[str, Any]) -> LowerBoundaryConfig | None:
    """Bind CM1's five lower-boundary selectors, or leave the free-slip deck.

    ``_require_exact_keys`` has already refused a partial declaration, so
    either all five keys are present or none is.
    """

    if not (_LOWER_BOUNDARY_KEYS & set(c)) and not (_SURFACE_LAYER_KEYS & set(c)):
        return None
    if not (_LOWER_BOUNDARY_KEYS & set(c)):
        refuse(
            code="CM1-CFG-SURFACE-LAYER-WITHOUT-LOWER-BOUNDARY",
            gate="config-grammar",
            reason=(
                "the sfcmodel=1 surface-layer options are read only by "
                "getcecd (src/sfcphys.F:81-236), which src/turb.F:614-616 "
                "calls only for sfcmodel>=1 with bbc=3; with the lower "
                "boundary left at its free-slip default the declared "
                "roughness, friction velocity and enthalpy coefficient reach "
                "nothing at all"
            ),
            remedy=(
                "declare closures.bbc, sfcmodel, isfcflx, cecd and cnstcd "
                "alongside them"
            ),
            option="closures.sfcmodel",
            value=None,
        )
    return LowerBoundaryConfig(
        surface_layer=_parse_surface_layer(c),
        # The CM1 range checks belong to _validate_lower_boundary, which
        # names the wrong result each out-of-range value would produce; this
        # only refuses a value that is not an integer at all.
        bbc=_integer("closures.bbc", c["bbc"]),
        sfcmodel=_integer("closures.sfcmodel", c["sfcmodel"]),
        isfcflx=_integer("closures.isfcflx", c["isfcflx"]),
        cecd=_integer("closures.cecd", c["cecd"]),
        cnstcd=_finite("closures.cnstcd", c["cnstcd"]),
    )

def _parse_open_companions(b: Mapping[str, Any]) -> OpenBoundaryCompanionConfig | None:
    """Bind CM1's four open-boundary companions, or leave cm1.F's own values.

    ``_require_exact_keys`` has already refused a partial declaration.  The
    CM1 range checks belong to ``_validate_boundary_family``, which names the
    wrong result each out-of-range value would produce.
    """

    if not (_OPEN_COMPANION_KEYS & set(b)):
        return None
    return OpenBoundaryCompanionConfig(
        roflux=_integer("boundaries.roflux", b["roflux"]),
        nudgeobc=_integer("boundaries.nudgeobc", b["nudgeobc"]),
        alphobc=_finite("boundaries.alphobc", b["alphobc"]),
        cstar=_finite("boundaries.cstar", b["cstar"]),
    )


def _parse_horizontal_damping(c: Mapping[str, Any]) -> HorizontalDampingConfig | None:
    """Bind CM1's lateral Rayleigh sponge, or leave hrdamp=0."""

    if not (_HORIZONTAL_DAMPING_KEYS & set(c)):
        return None
    return HorizontalDampingConfig(
        hrdamp=_integer("closures.hrdamp", c["hrdamp"]),
        xhd_m=_finite("closures.xhd", c["xhd"]),
    )



def _parse_adaptive_move(c: Mapping[str, Any]) -> AdaptiveMoveConfig | None:
    """Bind CM1's two adaptive moving-domain selectors, or leave the fixed frame.

    ``_require_exact_keys`` has already refused a half-declared pair, so
    either both keys are present or neither is.  The CM1 cross-checks belong
    to :func:`_validate_adaptive_move`, which names the wrong result each
    inadmissible value would produce; this only refuses a value that is not of
    the type CM1 declares (``input.F:200`` reads ``adapt_move_frq`` as a
    ``real`` and ``input.F:40`` reads ``do_adapt_move`` as a ``logical``).
    """

    if not (_ADAPT_MOVE_KEYS & set(c)):
        return None
    return AdaptiveMoveConfig(
        do_adapt_move=_boolean("closures.do_adapt_move", c["do_adapt_move"]),
        adapt_move_frq=_finite("closures.adapt_move_frq", c["adapt_move_frq"]),
    )

def _parse_sgs_startup_ramp(c: Mapping[str, Any]) -> SgsStartupRampConfig | None:
    """Bind CM1's ``ramp_sgs``/``ramp_time`` pair, or leave the compiled default.

    ``src/input.F:139,196`` read ``ramp_sgs`` as an integer and ``ramp_time``
    as a ``real``; the value checks are :func:`_validate_sgs_startup_ramp`'s.
    """

    if not (_SGS_RAMP_KEYS & set(c)):
        return None
    return SgsStartupRampConfig(
        ramp_sgs=_integer("closures.ramp_sgs", c["ramp_sgs"]),
        ramp_time=_finite("closures.ramp_time", c["ramp_time"]),
    )


def _vertical_word(grid: Mapping[str, Any], name: str) -> Any:
    """One ``param6`` vertical word, or CM1's own distributed default."""

    if name not in grid:
        return _VERTICAL_GRID_DEFAULTS[name]
    value = grid[name]
    if name == "stretch_z" and value is False:
        # ``stretch_z = false`` is the spelling this port shipped while it
        # generated only param.F:7789-7811's uniform coordinate, and it names
        # exactly CM1's ``stretch_z = 0``.  It is still read so that every
        # example document whose sha256 already travels inside a recorded
        # GPU-run receipt stays byte-identical.  ``true`` is not read, because
        # it names no branch: param.F:7814-8139 dispatches on 1, 2, 3 and 4.
        return 0
    return value


def _derived_top(grid: GridConfig) -> float:
    """The model top CM1 overwrites for the branches that do not read one."""

    import numpy as _np

    if grid.stretch_z == 0:
        return float(_np.float32(_np.float32(grid.dz_m) * _np.float32(grid.nz)))
    if grid.stretch_z == 3:
        # param.F:8019-8026, then :8037 makes that last face the model top.
        face = _np.float32(0.0)
        for value in grid.levels_m:
            face = _np.float32(_np.float32(2.0) * _np.float32(value)) - face
        return float(face)
    return float(_np.float32(grid.levels_m[-1])) if grid.levels_m else 0.0


def require_vertical_coordinate(config: "CM1Config") -> "VerticalLevels":
    """Build the declared vertical coordinate, or refuse by key and value.

    Four of CM1 r21.1's five branches are generated (``param.F:7814-8139``);
    ``cm1_vertical_grid.STRETCH_Z2_REFUSAL`` says why the fifth is not.  The
    refusals below are the ones CM1 itself raises -- ``stopcm1`` on an
    infeasible segment decomposition (``param.F:7828-7877``) or a non-positive
    layer depth (``param.F:8212-8224``) -- plus two this port owns: a stretch
    control that the selected branch does not read, and a coordinate whose
    lowest or highest three layers do not share one depth, which is the
    condition under which ``param.F:8985-9060``'s quadratic is the one the
    compiled kernels carry inline; it is named rather than silently
    mis-extrapolated.
    """

    from .cm1_vertical_grid import (
        PINNED_FORNBERG_F32,
        VerticalGridError,
        build_vertical_levels,
    )

    g = config.grid
    # param.F:7819-7953 is the one branch that reads the analytic controls.
    analytic = g.stretch_z == 1
    unused = [
        name
        for name, value in (
            ("grid.ztop_m", g.ztop_m),
            ("grid.str_bot_m", g.str_bot_m),
            ("grid.str_top_m", g.str_top_m),
            ("grid.dz_bot_m", g.dz_bot_m),
            ("grid.dz_top_m", g.dz_top_m),
        )
        if value != 0.0
    ]
    if not analytic and unused:
        derived = {
            0: "nz*dz_m (param.F:2608)",
            3: "its own top scalar level (param.F:8037)",
            4: "its own top w level (param.F:8105)",
        }[g.stretch_z]
        refuse(
            code="CM1-CFG-STRETCH-CONTROL-UNUSED",
            gate="config-grammar",
            reason=f"stretch_z={g.stretch_z} derives the model top from {derived} and reads no analytic stretch control, so {unused[0]} would travel in the run receipt describing a coordinate this run does not have",
            remedy=f"set {unused[0]} to 0.0, or select stretch_z=1",
            option=unused[0],
            value=getattr(g, unused[0].split(".", 1)[1]),
        )
    if analytic:
        for name in ("ztop_m", "dz_bot_m", "dz_top_m"):
            if getattr(g, name) <= 0.0:
                refuse(
                    code="CM1-CFG-STRETCH-CONTROL-MISSING",
                    gate="config-grammar",
                    reason=f"stretch_z={g.stretch_z} divides by grid.{name}; a zero would make every generated level non-finite",
                    remedy=f"set grid.{name} to its value in metres",
                    option=f"grid.{name}",
                    value=getattr(g, name),
                )
    if g.stretch_z in (3, 4) and not g.levels_m:
        refuse(
            code="CM1-CFG-STRETCH-CONTROL-MISSING",
            gate="config-grammar",
            reason=f"stretch_z={g.stretch_z} is param.F:{'8005-8070' if g.stretch_z == 3 else '8072-8137'}'s explicit level list; with no levels the run would fall back to the uniform coordinate",
            remedy=f"set grid.levels_m to the {g.nz if g.stretch_z == 3 else g.nz + 1} ascending {'scalar' if g.stretch_z == 3 else 'w'} levels in metres",
            option="grid.levels_m",
            value=[],
        )
    if analytic and g.levels_m:
        refuse(
            code="CM1-CFG-STRETCH-CONTROL-UNUSED",
            gate="config-grammar",
            reason=f"stretch_z={g.stretch_z} generates its own levels from param.F:7819-7953, so grid.levels_m would travel in the run receipt describing a coordinate this run does not have",
            remedy="remove grid.levels_m, or select stretch_z=3 or 4",
            option="grid.levels_m",
            value=list(g.levels_m),
        )
    try:
        levels = build_vertical_levels(
            stretch_z=g.stretch_z,
            nk=g.nz,
            dz_m=g.dz_m,
            ztop_m=g.ztop_m if analytic else _derived_top(g),
            str_bot_m=g.str_bot_m,
            str_top_m=g.str_top_m,
            dz_bot_m=g.dz_bot_m,
            dz_top_m=g.dz_top_m,
            levels_m=g.levels_m,
        )
    except VerticalGridError as error:
        refuse(
            code="CM1-CFG-VERTICAL-COORDINATE",
            gate="config-grammar",
            reason=f"CM1 r21.1 param.F:7814-8225 stops on this vertical grid: {error}",
            remedy="correct the named grid word; the vertical generator is a transcription and is not widened",
            option="grid.stretch_z",
            value=g.stretch_z,
        )
    # param.F:8985-9060 fits a quadratic through the lowest three scalar levels
    # and another through the highest three, and param.F:8990-8993 and
    # :9004-9007 divide every node by that end's own layer depth.  The spacing
    # therefore cancels: on any column whose relevant three layers share one
    # depth the weights are analytically (1.875, -1.25, 0.375), whatever that
    # depth is.  So the question worth asking of a coordinate is whether those
    # three layers *are* one depth -- not whether a fresh f32 recomputation of
    # the weights returns the pinned words bit for bit.  It was the second
    # question that stood here, and it refused a plain uniform grid whenever dz
    # had no exact binary32 spelling: at 40 x 333.3 m the top faces round by
    # one f32 ulp, the recomputation returns (1.8750014, -1.2500057, 0.3750043)
    # and the grid was turned away, while the arithmetically identical 500.0 m
    # column passed.  A refusal that fires on valid input is a defect, so the
    # test now reads cm1_vertical_grid's construction-level predicate, which
    # is decided on the heights the generator was asked for, before any of them
    # is rounded into the f32 coordinate.  The defect the refusal exists to
    # prevent is unchanged and is named below.
    if not levels.fornberg_matches_pinned_kernels():
        edge = "lowest" if not levels.lowest_layers_equally_spaced else "highest"
        depths = (
            levels.spacings_m[:3]
            if edge == "lowest"
            else tuple(reversed(levels.spacings_m[-3:]))
        )
        refuse(
            code="CM1-CFG-VERTICAL-FORNBERG",
            gate="support-matrix",
            reason=(
                f"this coordinate's {edge} three layers are {depths[0]}, "
                f"{depths[1]} and {depths[2]} m deep, so its own "
                "param.F:8985-9060 quadratic is fitted through three levels "
                "that are not equally spaced, but the compiled kernels of "
                "cuda_gm1_forecast, cuda_terrain_open_boundary, "
                "cuda_terrain_deformation, cuda_gm1_scalar_turbulence, "
                "cuda_gm1_child_stretched_nm_def and cuda_storm_diagnostics all "
                f"carry {PINNED_FORNBERG_F32} inline, which is the quadratic "
                "through three *equally* spaced nodes; rf0 at the "
                f"{edge} face, the w-level u/v averages, the scalar and "
                "momentum w-level interpolations and the surface stress would "
                "each be extrapolated from a parabola fitted to the wrong "
                "three levels"
            ),
            remedy=(
                f"keep the {edge} three scalar levels uniformly spaced -- for stretch_z=1 "
                "that is str_bot_m >= 3*dz_bot_m and ztop_m - str_top_m >= 3*dz_top_m"
            ),
            option="grid.stretch_z",
            value=g.stretch_z,
        )
    return levels


def parse_mapping(raw: Mapping[str, Any]) -> CM1Config:
    if not isinstance(raw, Mapping):
        refuse(
            code="CM1-CFG-ROOT-TYPE",
            gate="config-grammar",
            reason="the configuration root must be a set of named TOML tables",
            remedy="supply a TOML document whose root contains the six required tables",
            option="config-root",
            value=type(raw).__name__,
        )
    lane = _document_lane(raw)
    schema = _lane_sections(None if lane is None else lane.value)
    nest_table = raw.get("nest")
    if isinstance(nest_table, Mapping) and "follow_max_cells" in nest_table:
        refuse(
            code="CM1-CFG-NEST-FOLLOW-UNIT",
            gate="config-grammar",
            reason=(
                "nest.follow_max_cells counted PARENT cells per re-centring, so "
                "the same number is a different speed on every parent spacing: "
                "4 cells every 270 s is 13 m/s on a 900 m parent and 1.5 m/s on "
                "a 100 m one, and a deck carried from one to the other would "
                "silently stop following any real storm (measured: an 88-cell "
                "request clamped to 4, 22 re-centrings to catch up)"
            ),
            remedy=(
                "declare nest.follow_max_speed_m_s, the fastest the nest may "
                "follow in metres per second; the parent-cell limit per "
                "re-centring is derived from it and the parent spacing"
            ),
            option="nest.follow_max_cells",
            value=nest_table["follow_max_cells"],
        )
    _strict_sections(raw, schema)
    _require_exact_keys(raw, schema)
    r = raw["run"]
    g = raw["grid"]
    d = raw["dynamics"]
    b = raw["boundaries"]
    c = raw["closures"]
    p = raw["physics"]

    # The grammar above proved the lane string exists; keep its type refusal.
    lane = Lane(_string("run.lane", r["lane"]))

    config = CM1Config(
        run=RunConfig(
            lane=lane,
            duration_s=_finite_positive("run.duration_s", r["duration_s"]),
            dt_s=_finite_positive("run.dt_s", r["dt_s"]),
            history_interval_s=_finite_positive(
                "run.history_interval_s", r["history_interval_s"]
            ),
            restart_interval_s=_finite(
                "run.restart_interval_s", r["restart_interval_s"]
            ),
            irst=_optional_integer_choice(
                "run.irst", r, {0, 1},
                "cm1.F:2660 reads a restart file only when irst=1; any other value "
                "is neither CM1's fresh start nor its restart, and the run would "
                "silently take the branch the port happened to default to",
            ),
            rstnum=_optional_integer(
                "run.rstnum", r, minimum=1,
                reason="README.restart numbers restart files from one and counts "
                "upward, so a number below one names no restart and the resume "
                "would fall back to the initial condition with the receipt still "
                "recording a restart",
            ),
            iconly=_optional_integer_choice(
                "run.iconly", r, {0, 1},
                "README.namelist:233-235 makes iconly a two-valued switch; any "
                "other value would leave it ambiguous whether cm1.F:4341's stop "
                "applies, and the run would integrate for hours when the caller "
                "asked only to see the initial condition",
            ),
            restart_reset_frqtim=_optional_boolean(
                "run.restart_reset_frqtim", r,
            ),
        ),
        grid=GridConfig(
            nx=_integer("grid.nx", g["nx"], minimum=7),
            ny=_integer("grid.ny", g["ny"], minimum=7),
            nz=_integer("grid.nz", g["nz"], minimum=7),
            dx_m=_finite_positive("grid.dx_m", g["dx_m"]),
            dy_m=_finite_positive("grid.dy_m", g["dy_m"]),
            dz_m=_finite_positive("grid.dz_m", g["dz_m"]),
            terrain=_boolean("grid.terrain", g["terrain"]),
            stretch_x=_boolean("grid.stretch_x", g["stretch_x"]),
            stretch_y=_boolean("grid.stretch_y", g["stretch_y"]),
            # input.F:283 declares stretch_z in param6 as an integer branch
            # selector, so it is admitted by value and not as a flag.
            stretch_z=_integer_choice(
                "grid.stretch_z",
                _vertical_word(g, "stretch_z"),
                set(ADMITTED_STRETCH_Z),
                "CM1 r21.1 param.F:7814-8139 dispatches on 1, 2, 3 and 4 and treats everything below 1 as the uniform coordinate; true names no branch and an out-of-range value would leave the uniform coordinate in place while the receipt claimed a stretched one, and for stretch_z=2 specifically: "
                + STRETCH_Z2_REFUSAL,
            ),
            # param.F:2608 and :8040/:8107 overwrite ztop for stretch_z 0, 3
            # and 4, so only branch 1 reads a declared model top; the others
            # must leave it at 0.0 and take the generated one.
            ztop_m=_finite_nonnegative("grid.ztop_m", _vertical_word(g, "ztop_m")),
            str_bot_m=_finite_nonnegative("grid.str_bot_m", _vertical_word(g, "str_bot_m")),
            str_top_m=_finite_nonnegative("grid.str_top_m", _vertical_word(g, "str_top_m")),
            dz_bot_m=_finite_nonnegative("grid.dz_bot_m", _vertical_word(g, "dz_bot_m")),
            dz_top_m=_finite_nonnegative("grid.dz_top_m", _vertical_word(g, "dz_top_m")),
            levels_m=_level_array("grid.levels_m", _vertical_word(g, "levels_m")),
        ),
        dynamics=DynamicsConfig(
            psolver=_integer_choice(
                    "dynamics.psolver",
                    d["psolver"],
                    {2, 3},
                    "only the internal dry bootstrap and product pressure paths are in scope",
            ),
            adaptive_timestep=_boolean(
                "dynamics.adaptive_timestep", d["adaptive_timestep"]
            ),
            scalar_horizontal_order=_integer_choice(
                    "dynamics.scalar_horizontal_order",
                    d["scalar_horizontal_order"],
                    {5},
                    "other scalar stencil orders have no source-mapped operator deck",
            ),
            scalar_vertical_order=_integer_choice(
                    "dynamics.scalar_vertical_order",
                    d["scalar_vertical_order"],
                    {5},
                    "other scalar stencil orders have no source-mapped operator deck",
            ),
            velocity_horizontal_order=_integer_choice(
                    "dynamics.velocity_horizontal_order",
                    d["velocity_horizontal_order"],
                    {5},
                    "other velocity stencil orders have no source-mapped operator deck",
            ),
            velocity_vertical_order=_integer_choice(
                    "dynamics.velocity_vertical_order",
                    d["velocity_vertical_order"],
                    {5},
                    "other velocity stencil orders have no source-mapped operator deck",
            ),
            scalar_weno=_integer_choice(
                    "dynamics.scalar_weno",
                    d["scalar_weno"],
                    {0, 2},
                    "only off for a reduced bootstrap or final-stage target cadence is mapped",
            ),
            velocity_weno=_integer_choice(
                    "dynamics.velocity_weno",
                    d["velocity_weno"],
                    {0},
                    "the first profile does not select velocity WENO",
            ),
            weno_order=_integer_choice(
                    "dynamics.weno_order",
                    d["weno_order"],
                    {5},
                    "only fifth-order WENO is in the first source/oracle campaign",
            ),
            dry_mass_correction=_boolean(
                "dynamics.dry_mass_correction", d["dry_mass_correction"]
            ),
            # README.namelist:518-530; solve2.F:631-720; param.F:8945-8969.
            coriolis=_integer_choice(
                "dynamics.coriolis",
                _dynamics_word(d, "coriolis"),
                {0, 1},
                "solve2.F:632 dispatches on icor=1 and nothing else; any "
                "other value would silently integrate no rotation",
            ),
            beta_plane=_integer_choice(
                "dynamics.beta_plane",
                _dynamics_word(d, "beta_plane"),
                {0, 1},
                "param.F:8955 builds f2d for betaplane=1 and leaves the "
                "f-plane constant otherwise; there is no third branch",
            ),
            fcor_s_inv=_finite_real(
                "dynamics.fcor_s_inv", _dynamics_word(d, "fcor_s_inv")
            ),
            # README.namelist:751-759; sound.F:78-82.
            alph=_unit_interval(
                "dynamics.alph",
                _dynamics_word(d, "alph"),
                "sound.F:79-80 forms beta=1.0-alph and asq=alph*alph; an "
                "off-centering weight outside the unit interval makes the "
                "vertically implicit acoustic step amplify its own solution "
                "every small step instead of damping it",
            ),
            kdiv=_unit_interval(
                "dynamics.kdiv",
                _dynamics_word(d, "kdiv"),
                "sound.F's divergence damper adds kdiv times the pressure "
                "change back into the pressure the next small step advects; "
                "outside [0, 1] that feedback grows without bound",
            ),
            # README.namelist:281-306,743-749.
            idiff=_integer_choice(
                "dynamics.idiff",
                _dynamics_word(d, "idiff"),
                {0, 1, 2},
                "README.namelist:281-284 defines 0, 1 (all variables) and 2 "
                "(winds only) and adv.F/diff2.F have no other branch",
            ),
            mdiff=_integer_choice(
                "dynamics.mdiff",
                _dynamics_word(d, "mdiff"),
                {0, 1},
                "README.namelist:290-296 defines the Xue monotonic variant "
                "as on or off and nothing between",
            ),
            difforder=_integer_choice(
                "dynamics.difforder",
                _dynamics_word(d, "difforder"),
                {2, 6},
                "README.namelist:297-307 defines second and sixth order "
                "only; adv.F:962 tests difforder==6 and cm1.F:3791 tests "
                "difforder==2, so any other value diffuses nothing",
            ),
            kdiff2=_finite_nonnegative(
                "dynamics.kdiff2", _dynamics_word(d, "kdiff2")
            ),
            kdiff6=_finite_nonnegative(
                "dynamics.kdiff6", _dynamics_word(d, "kdiff6")
            ),
            lspgrad=_parse_lspgrad(d),
        ),
        boundaries=BoundaryConfig(
            west=_integer("boundaries.west", b["west"]),
            east=_integer("boundaries.east", b["east"]),
            south=_integer("boundaries.south", b["south"]),
            north=_integer("boundaries.north", b["north"]),
            bottom=_integer("boundaries.bottom", b["bottom"]),
            top=_integer("boundaries.top", b["top"]),
            radiative_order=_integer("boundaries.radiative_order", b["radiative_order"]),
            open_companions=_parse_open_companions(b),
        ),
        closures=ClosureConfig(
            sgs_model=_integer("closures.sgs_model", c["sgs_model"]),
            vertical_implicit_diffusion=_boolean(
                "closures.vertical_implicit_diffusion", c["vertical_implicit_diffusion"]
            ),
            upper_damping=_integer("closures.upper_damping", c["upper_damping"]),
            upper_damping_depth_m=_finite(
                "closures.upper_damping_depth_m", c["upper_damping_depth_m"]
            ),
            upper_damping_rate_s_inv=_finite(
                "closures.upper_damping_rate_s_inv", c["upper_damping_rate_s_inv"]
            ),
            moving_frame=_boolean("closures.moving_frame", c["moving_frame"]),
            u_move_m_s=_finite("closures.u_move_m_s", c["u_move_m_s"]),
            v_move_m_s=_finite("closures.v_move_m_s", c["v_move_m_s"]),
            lower_boundary=_parse_lower_boundary(c),
            horizontal_damping=_parse_horizontal_damping(c),
            rterm=(
                _integer_choice(
                    "closures.rterm",
                    c["rterm"],
                    {0, 1},
                    "src/solve1.F:1006 tests rterm against 1 and nothing else, so any other value is the rterm=0 identity wearing a label",
                )
                if "rterm" in c
                else None
            ),
            # README.namelist:1089-1092.  Ranges, the time-average branch and
            # the sgsmodel cross-check are cuda_gm1_two_part_sgs's refusals.
            t2p_avg=(
                _integer("closures.t2p_avg", c["t2p_avg"])
                if "t2p_avg" in c
                else None
            ),
            turbulence_scale=_parse_turbulence_scale(c),
            adaptive_move=_parse_adaptive_move(c),
            sgs_startup_ramp=_parse_sgs_startup_ramp(c),
        ),
        physics=PhysicsConfig(
            component=_string("physics.component", p["component"]),
            equation_set=_integer("physics.equation_set", p["equation_set"], minimum=1),
            # README.namelist:1366-1369.  CM1 spells this as the integer 0/1
            # its namelist carries; a TOML document spells a yes/no question
            # as a boolean, and param.F:4665's max(0,min(1,...)) clamp means
            # the two are the same two states.
            output_dbz=(
                False if "output_dbz" not in p
                else _boolean("physics.output_dbz", p["output_dbz"])
            ),
            dbz_level_m=(
                None if "dbz_level_m" not in p
                else _finite_positive("physics.dbz_level_m", p["dbz_level_m"])
            ),
            ihail=(
                _integer_choice(
                    "physics.ihail",
                    p["ihail"],
                    {0, 1},
                    "CM1 param.F:1567-1580 stops the model outside 0..1; the value "
                    "selects which dense-ice constants the scheme is initialized "
                    "with, so an unadmitted value would silently execute the "
                    "unselected category's fall speeds",
                )
                if "ihail" in p
                else None
            ),
            ndcnst=(
                _finite_positive("physics.ndcnst", p["ndcnst"])
                if "ndcnst" in p
                else None
            ),
            cuparam=(
                _integer("physics.cuparam", p["cuparam"]) if "cuparam" in p else None
            ),
        ),
        runtime=_parse_runtime(raw) if lane is Lane.CARTESIAN else None,
        initialization=_parse_initialization(raw) if lane is Lane.CARTESIAN else None,
        surface=_parse_surface(raw),
        radiation=_parse_radiation(raw),
        output=_parse_output(raw),
        nest=_parse_nest(raw) if lane is Lane.CARTESIAN else None,
        chain=_parse_chain(raw) if lane is Lane.CARTESIAN else None,
        ensemble=_parse_ensemble(raw) if lane is Lane.CARTESIAN else None,
        diagnostics=_parse_diagnostics(raw) if lane is Lane.CARTESIAN else None,
        tornado_diagnostics=(
            _parse_tornado_diagnostics(raw) if lane is Lane.CARTESIAN else None
        ),
        lagrangian=_parse_lagrangian(raw) if lane is Lane.CARTESIAN else None,
    )
    # &param9 before the lane profile: a document that asks for a product this
    # geometry or this physics cannot make should hear about THAT, not about
    # the first profile word that happens to differ.
    _validate_output_against_lane(config)
    _validate_physics_families(config)
    _validate_first_slice(config)
    return config


def _validate_reflectivity(config: CM1Config) -> None:
    """Admit CM1's ``output_dbz`` by key and by value.

    ``param.F:1583-1603`` is CM1's own refusal: with ``imoist=1`` and
    ``output_dbz=1`` under a ``ptype`` whose scheme has no reflectivity
    routine, CM1 prints the admitted list and either stops (``ptype=4``) or
    silently drops the request.  This port refuses instead of dropping,
    because a run whose document says "reflectivity" and whose frames carry
    none is worse than a run that will not start.

    The formula is scheme-specific: it is a moment of the scheme's own assumed
    rain, snow and graupel size distributions.  This port drives WSM6 through
    the ArWen seam and publishes WRF's ``refl10cm_wsm6``
    (``mp_wsm6.F90:2275-2444``); applying it to another scheme's condensate
    would publish dBZ values the simulated hydrometeors never scattered.
    """

    from .gm1_runtime import REFLECTIVITY_COMPONENTS

    physics = config.physics
    if physics.output_dbz and physics.component not in REFLECTIVITY_COMPONENTS:
        refuse(
            code="CM1-CFG-DBZ-COMPONENT",
            gate="support-matrix",
            reason=(
                f"physics.component={physics.component!r} has no admitted "
                "reflectivity formula in this port; reflectivity is a moment of "
                "the microphysics scheme's own assumed rain, snow and graupel "
                "size distributions, so publishing another scheme's formula "
                "under this run's hydrometeors would report a dBZ field the "
                "simulated storm never scattered, and CM1 param.F:1583-1603 "
                "refuses output_dbz by ptype for the same reason"
            ),
            remedy=(
                "select a component with an admitted reflectivity formula ("
                + ", ".join(sorted(REFLECTIVITY_COMPONENTS))
                + "), or set physics.output_dbz = false"
            ),
            option="physics.output_dbz",
            value=physics.output_dbz,
        )
    if physics.dbz_level_m is not None and not physics.output_dbz:
        refuse(
            code="CM1-CFG-DBZ-LEVEL-INACTIVE",
            gate="support-matrix",
            reason=(
                "a reflectivity level is declared while output_dbz is off, so no "
                "reflectivity is computed at all and the level this document "
                "names would never appear in a published frame"
            ),
            remedy="set physics.output_dbz = true, or remove physics.dbz_level_m",
            option="physics.dbz_level_m",
            value=physics.dbz_level_m,
        )
    if physics.dbz_level_m is None:
        return
    levels = require_vertical_coordinate(config)
    # misclibs.F:2454-2466 scans k=nk,1,-1 and keeps the lowest level at or
    # above the requested height; a height the column never reaches is left
    # unset and CM1's kcheck guard skips the report.
    nz = config.grid.nz
    if not any(float(levels.zh[k]) >= physics.dbz_level_m for k in range(1, nz + 1)):
        top = float(levels.zh[nz])
        refuse(
            code="CM1-CFG-DBZ-LEVEL-ABOVE-TOP",
            gate="support-matrix",
            reason=(
                f"physics.dbz_level_m={physics.dbz_level_m!r} m is above this "
                f"grid's highest scalar level at {top!r} m; misclibs.F:2454-2466 "
                "leaves such a request unset, so the published plane would be the "
                "model top under a label naming a height it is not"
            ),
            remedy=(
                f"set physics.dbz_level_m to at most {top!r}, or raise the model "
                "top with grid.nz and grid.dz_m"
            ),
            option="physics.dbz_level_m",
            value=physics.dbz_level_m,
        )


def _validate_boundary_family(config: CM1Config) -> None:
    """Refuse every lateral boundary this port would integrate as another one.

    Each rule is one CM1 r21.1 stop, forced rewrite or absent operator.  The
    four lateral values are CM1's own (``README.namelist:561-568``): 1
    periodic, 2 open-radiative, 3 rigid free-slip, 4 rigid no-slip, and every
    combination of them, including CM1's own mixed squall-line family
    ``wbc=ebc=2`` with ``sbc=nbc=1``, reaches one operator.
    """

    b = config.boundaries
    lane_is_dry = config.run.lane is Lane.DRY_DYCORE
    if lane_is_dry:
        # The dry bootstrap's periodic operators are bound to an unpadded
        # layout of their own; they cannot read the ngxy ghost planes the
        # general family writes, so a non-periodic side there would advect
        # from words nothing filled.
        if b.lateral != (1, 1, 1, 1):
            refuse(
                code="CM1-CFG-BOUNDARY-COMBINATION",
                gate="boundary-oracles",
                reason=(
                    "the dry bootstrap lane binds cuda_sound3_periodic, whose "
                    "carriers have no ngxy ghost planes; a non-periodic side "
                    "would leave its boundary stencil reading uninitialized "
                    "words rather than a filled halo"
                ),
                remedy="use (1, 1, 1, 1) on the dry lane, or the cartesian lane",
                option="boundaries",
                value=b.lateral,
            )
    for name, value in (
        ("boundaries.west", b.west),
        ("boundaries.east", b.east),
        ("boundaries.south", b.south),
        ("boundaries.north", b.north),
    ):
        _integer_choice(
            name,
            value,
            {1, 2, 3, 4},
            "src/bc.F:180-1143 has exactly four lateral branches; any other "
            "value would leave that side's ngxy ghost planes holding whatever "
            "the previous step left there, and the 5th-order advection stencil "
            "reads three of them",
        )
    # src/param.F:6678-6681 -- ib* is what selects the pressure-gradient face
    # range and the ssopenbc/radbc calls; a lane that cannot express one of
    # them would integrate a side it did not name.
    _integer_choice(
        "boundaries.radiative_order",
        b.radiative_order,
        {1, 2, 4},
        "README.namelist:589-594 names exactly three radiative schemes: "
        "Klemp-Wilhelmson on large steps (1), on small steps (2), and "
        "Durran-Klemp (4); another value would leave radbcw/e/s/n at their "
        "previous words and the open faces would advance on the tendency "
        "alone, which is a reflecting boundary wearing a radiative name",
    )
    # README.namelist:583-588.  tbc=2 and tbc=3 replace the t13/t23 upper face
    # and the tkebc top branch; this port's closure clamps that face to free
    # slip with no branch (cuda_sgs_stress_tendency.py t13/t23 build, and
    # cuda_sgs_tke.py:99-109), so admitting them would integrate a free-slip
    # lid while the receipt named a wall.
    _integer_choice(
        "boundaries.top",
        b.top,
        {1},
        "src/turb.F:3311-3360 and :4515-4702 give tbc=2/3 their own upper "
        "t13/t23 face and tkebc branch; this port's stress kernel clamps the "
        "top face to the level below it with no branch, so a declared no-slip "
        "or semi-slip lid would produce no drag at all while the receipt named "
        "one",
    )
    _integer_choice(
        "boundaries.roflux",
        b.roflux,
        {0, 1},
        "README.namelist:595-602 is a flag; src/bc.F:2180-2391 either rescales "
        "the outward normal mass flux or does not",
    )
    _integer_choice(
        "boundaries.nudgeobc",
        b.nudgeobc,
        {0, 1},
        "README.namelist:603-610 is a flag; src/solve2.F:726-822 either adds "
        "the inflow nudging tendency or does not",
    )
    if b.nudgeobc == 1:
        # src/solve2.F:728 forms 1.0/alphobc.
        if not (b.alphobc > 0.0):
            refuse(
                code="CM1-CFG-ALPHOBC-RANGE",
                gate="support-matrix",
                reason=(
                    "src/solve2.F:728 forms 1.0/alphobc; a zero or negative "
                    "time scale would make the nudging tendency infinite or "
                    "push the inflow wind away from the declared sounding "
                    "instead of toward it"
                ),
                remedy="declare a positive boundaries.alphobc; CM1's own default is 60.0 s",
                option="boundaries.alphobc",
                value=b.alphobc,
            )
        if 2 not in b.lateral:
            refuse(
                code="CM1-CFG-NUDGEOBC-WITHOUT-OPEN-SIDE",
                gate="support-matrix",
                reason=(
                    "src/solve2.F:726-822 adds the inflow nudging tendency only "
                    "on a side whose value is 2, so with no open side the "
                    "receipt would name an inflow nudging the run never applied"
                ),
                remedy="declare an open-radiative side, or set boundaries.nudgeobc=0",
                option="boundaries.nudgeobc",
                value=b.nudgeobc,
            )
    if b.cstar < 0.0:
        refuse(
            code="CM1-CFG-CSTAR-RANGE",
            gate="support-matrix",
            reason=(
                "src/bc.F:1964 forms ua-cstar as the west outward phase speed; "
                "a negative cstar would make an outflow face radiate inward and "
                "reflect the wave it was written to absorb"
            ),
            remedy="declare a nonnegative boundaries.cstar; src/param.F:722 reads anything below 0.1 as unset and uses 30.0",
            option="boundaries.cstar",
            value=b.cstar,
        )
    # src/param.F:1906-1950.  Expressible for the first time now that a rigid
    # wall can be declared, and CM1 stops rather than translate a wall.
    if config.closures.moving_frame:
        for axis, speed, sides, names in (
            ("u_move_m_s", config.closures.u_move_m_s, (b.west, b.east), ("west", "east")),
            ("v_move_m_s", config.closures.v_move_m_s, (b.south, b.north), ("south", "north")),
        ):
            if abs(speed) >= 1.0e-10 and any(side in (3, 4) for side in sides):
                refuse(
                    code="CM1-CFG-MOVING-FRAME-RIGID-WALL",
                    gate="support-matrix",
                    reason=(
                        f"src/param.F:1906-1950 stops CM1 for this exact pair: a "
                        f"rigid wall is fixed in the ground frame, so translating "
                        f"the domain at {speed} m/s would slide the wall through "
                        "the fluid and the declared no-slip or free-slip surface "
                        "would move relative to the air it is meant to bound"
                    ),
                    remedy=(
                        f"set closures.{axis} to 0.0, or make boundaries."
                        f"{names[0]}/{names[1]} periodic or open"
                    ),
                    option=f"closures.{axis}",
                    value=speed,
                )


def _validate_horizontal_damping(config: CM1Config) -> None:
    """Refuse every lateral Rayleigh sponge CM1 itself stops for."""

    closures = config.closures
    _integer_choice(
        "closures.hrdamp",
        closures.hrdamp,
        {0, 1, 2},
        "README.namelist:401-403 and src/solve1.F:404-438 have exactly three "
        "branches: off, damp toward the base state, damp toward the evolving "
        "horizontal average",
    )
    if closures.hrdamp == 0:
        if closures.xhd_m < 0.0:
            refuse(
                code="CM1-CFG-NUMERIC-RANGE",
                gate="config-grammar",
                reason="a damping width cannot be negative",
                remedy="use zero when the lateral sponge is disabled",
                option="closures.xhd",
                value=closures.xhd_m,
            )
        return
    if not (closures.xhd_m > 0.0):
        refuse(
            code="CM1-CFG-XHD-RANGE",
            gate="support-matrix",
            reason=(
                "src/param.F:8736-8776 divides by xhd to build the lateral "
                "cosine taper; with xhd zero or negative every tauf word would "
                "stay at its vertical value and the declared lateral sponge "
                "would absorb nothing"
            ),
            remedy="declare a positive closures.xhd, or set closures.hrdamp=0",
            option="closures.xhd",
            value=closures.xhd_m,
        )
    if not (closures.upper_damping_rate_s_inv > 0.0):
        refuse(
            code="CM1-CFG-RDALPHA-RANGE",
            gate="support-matrix",
            reason=(
                "src/solve1.F:404-438 multiplies every Rayleigh coefficient by "
                "rdalpha, so a zero rate makes the declared lateral sponge a "
                "no-op that the receipt would still name"
            ),
            remedy="declare a positive closures.upper_damping_rate_s_inv; CM1 recommends about 1/300",
            option="closures.upper_damping_rate_s_inv",
            value=closures.upper_damping_rate_s_inv,
        )
    # src/param.F:8689-8722 stops the model when the sponge covers more than
    # half of either horizontal extent.
    for name, extent in (
        ("grid.nx * grid.dx_m", config.grid.nx * config.grid.dx_m),
        ("grid.ny * grid.dy_m", config.grid.ny * config.grid.dy_m),
    ):
        if closures.xhd_m > 0.5 * extent:
            refuse(
                code="CM1-CFG-XHD-DEPTH",
                gate="support-matrix",
                reason=(
                    f"src/param.F:8689-8722 stops CM1 here: a {closures.xhd_m} m "
                    f"sponge on a {extent} m side damps w over more than half "
                    "the domain, so the interior this run exists to resolve "
                    "would be inside the sponge"
                ),
                remedy=f"set closures.xhd to at most {0.5 * extent}, or widen {name}",
                option="closures.xhd",
                value=closures.xhd_m,
            )


def _validate_surface(config: CM1Config) -> None:
    """Refuse every ``[surface]`` table CM1 would stop on, by CM1's rule.

    ``src/init_surface.F:40`` runs the whole block only for ``sfcmodel>=1``
    or ``bbc=3``, and ``:270-322`` then stops CM1 on an impossible land/water
    flag, land-use category or skin temperature.  A document that declares
    the table without a surface is refused here rather than silently
    building fields nothing reads.
    """

    surface = config.surface
    if surface is None:
        return
    closures = config.closures
    # src/init_surface.F:40
    if closures.sfcmodel < 1 and closures.bbc != 3:
        refuse(
            code="CM1-CFG-SURFACE-WITHOUT-SURFACE-LAYER",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:40 fills tsk, xland, lu_index and znt "
                "only for sfcmodel>=1 or bbc=3; with a free-slip or no-slip "
                "bottom and no surface layer nothing reads them, so the "
                "declared sea-surface temperature and land use would not "
                "reach the run at all"
            ),
            remedy=(
                "declare closures.bbc=3 with closures.sfcmodel=1, or remove "
                "the [surface] table"
            ),
            option="closures.sfcmodel",
            value=closures.sfcmodel,
        )
    # README.namelist:982-989; src/init_surface.F:67-145
    if surface.initsfc not in (1, 2, 3, 4):
        refuse(
            code="CM1-CFG-INITSFC-RANGE",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:67-145 has four branches and its final "
                "ELSE writes nothing at all (:135-145), so any other value "
                "leaves every column at the tsk0/xland0/lu0 defaults while "
                "the receipt names a spatially varying surface"
            ),
            remedy=(
                "set surface.initsfc to 1 (constant), 2 (sea breeze), 3 "
                "(rough west, smooth east) or 4 (coastline)"
            ),
            option="surface.initsfc",
            value=surface.initsfc,
        )
    # src/init_surface.F:279-289
    if surface.xland0 not in (1.0, 2.0):
        refuse(
            code="CM1-CFG-XLAND0-RANGE",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:279-289 stops CM1 unless every column is "
                "land (1) or water (2), and getcecd branches on xland>1.5 "
                "(src/sfcphys.F:141); a third value silently takes the land "
                "branch while the receipt names a water surface"
            ),
            remedy="set surface.xland0 to 1.0 for land or 2.0 for water",
            option="surface.xland0",
            value=surface.xland0,
        )
    # src/init_surface.F:290-300; LANDUSE.TBL carries 33 categories.
    if not 1 <= surface.lu0 <= 33:
        refuse(
            code="CM1-CFG-LU0-RANGE",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:446-459 indexes LANDUSE.TBL with this "
                "category to set the roughness length, moisture availability "
                "and albedo of every column; outside [1,33] there is no row, "
                "so the roughness the surface drag is built from would come "
                "from unallocated memory"
            ),
            remedy="set surface.lu0 to a LANDUSE.TBL category in [1, 33]",
            option="surface.lu0",
            value=surface.lu0,
        )
    # src/init_surface.F:442; the table carries two seasons.
    if surface.season not in (1, 2):
        refuse(
            code="CM1-CFG-SEASON-RANGE",
            gate="support-matrix",
            reason=(
                "LANDUSE.TBL carries exactly two seasons "
                "(src/init_surface.F:378,442); any other index reads past the "
                "table, so the roughness length, albedo and moisture "
                "availability of every column would come from unallocated "
                "memory"
            ),
            remedy="set surface.season to 1 (summer) or 2 (winter)",
            option="surface.season",
            value=surface.season,
        )
    # src/init_surface.F:301-309
    if not surface.tsk0_k >= 100.0:
        refuse(
            code="CM1-CFG-TSK0-RANGE",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:301-309 stops CM1 below 100 K; sfcflux "
                "divides the skin temperature by the surface Exner function "
                "(src/sfcphys.F:341) and takes its saturation mixing ratio "
                "(src/cm1libs.F:38), and below 100 K that saturation pressure "
                "underflows to zero, so the moisture flux draws the full "
                "lowest-level mixing ratio into the ground every step"
            ),
            remedy="declare a physical surface.tsk0_k in kelvin",
            option="surface.tsk0_k",
            value=surface.tsk0_k,
        )
    if not surface.tmn0_k >= 100.0:
        refuse(
            code="CM1-CFG-TMN0-RANGE",
            gate="support-matrix",
            reason=(
                "src/init_surface.F:310-318 stops CM1 below 100 K; the soil "
                "slab interpolates between the skin and deep-layer "
                "temperatures (src/init_surface.F:160-166), so a deep layer "
                "below 100 K drives the whole soil column below it"
            ),
            remedy="declare a physical surface.tmn0_k in kelvin",
            option="surface.tmn0_k",
            value=surface.tmn0_k,
        )
    # src/init_surface.F:327-336, src/oml.F; README.namelist:971-977.
    if surface.oceanmodel not in (0, 1):
        refuse(
            code="CM1-CFG-OCEANMODEL-UNSUPPORTED",
            gate="support-matrix",
            reason=(
                "oceanmodel=2 is the Pollard et al. ocean mixed layer "
                "(src/oml.F, src/init_surface.F:329-336, src/turb.F:1305): it "
                "makes the sea-surface temperature prognostic so it cools "
                "under the storm.  No GM1 carrier evolves it, so the run "
                "would hold the sea surface at its initial temperature for "
                "the whole integration while the receipt named an interactive "
                "ocean"
            ),
            remedy=(
                "set surface.oceanmodel=1, the fixed sea-surface temperature "
                "CM1 requires with sfcmodel=1 (README.namelist:946)"
            ),
            option="surface.oceanmodel",
            value=surface.oceanmodel,
        )


def _validate_physics_families(config: CM1Config) -> None:
    """Apply every CM1 cross-check that ties radiation, PBL and cumulus.

    The rules themselves live in :mod:`cm1core.cm1_physics_families` with
    their CM1 source anchors; this is where a parsed document meets them.
    """

    from .cm1_physics_families import (
        Cm1CumulusSelection,
        Cm1RadiationSelection,
        Cm1TurbulenceScale,
        NO_RADIATION,
        validate_cumulus_selection,
        validate_radiation_selection,
        validate_turbulence_scale,
    )

    validate_turbulence_scale(
        Cm1TurbulenceScale(
            cm1setup=config.closures.cm1setup,
            sgsmodel=config.closures.sgs_model,
            ipbl=config.closures.ipbl,
        )
    )
    radiation = config.radiation
    selection = (
        NO_RADIATION
        if radiation is None
        else Cm1RadiationSelection(
            radopt=radiation.radopt,
            dtrad_s=radiation.dtrad_s,
            ctrlat=radiation.ctrlat,
            ctrlon=radiation.ctrlon,
            year=radiation.year,
            month=radiation.month,
            day=radiation.day,
            hour=radiation.hour,
            minute=radiation.minute,
            second=radiation.second,
        )
    )
    validate_radiation_selection(selection, rterm=config.closures.rterm_value)
    validate_cumulus_selection(
        Cm1CumulusSelection(
            cuparam=config.physics.cuparam_value,
            dx_m=config.grid.dx_m,
            dy_m=config.grid.dy_m,
        )
    )
    # The lane's own wall, named rather than left to the generic support
    # matrix.  cuda_cartesian_closure.held_tendencies runs the prognostic-TKE
    # closure -- TKE source, turbt diffusion, SGS stress divergence and
    # scalar turbulence -- on every large step and consults no sgsmodel
    # selector, so this runtime has no way to be anything but cm1setup=1.
    if config.run.lane is not Lane.DRY_DYCORE and config.closures.cm1setup != 1:
        refuse(
            code="CM1-CFG-CLOSURE-HAS-NO-OFF-SWITCH",
            gate="cartesian-runtime",
            reason=(
                f"cm1setup={config.closures.cm1setup} asks this lane not to run "
                "an LES subgrid closure, but cm1core.cuda_cartesian_closure "
                "integrates the prognostic-TKE closure on every large step and "
                "reads no sgsmodel selector; the run would be recorded as "
                + (
                    "an adiabatic, inviscid integration"
                    if config.closures.cm1setup == 0
                    else "a mesoscale run with a PBL parameterization"
                )
                + " while the LES closure went on mixing every column"
            ),
            remedy=(
                "use closures.cm1setup = 1 in this lane; making the chain "
                "conditional means gating the TKE source, turbt diffusion, SGS "
                "stress and scalar turbulence in "
                "cuda_cartesian_closure.held_tendencies on CM1's idoles"
            ),
            option="closures.cm1setup",
            value=config.closures.cm1setup,
        )


def tornado_diagnostic_selection(config: CM1Config) -> Any:
    """Bind the ``[diagnostics]`` table to the runtime's own selection type.

    An absent table is the all-off default, which is the state every run
    integrated before these switches existed.
    """

    from .tornado_diagnostics import (
        DEFAULT_TORNADO_DIAGNOSTICS,
        TornadoDiagnosticSelection,
    )

    declared = config.tornado_diagnostics
    if declared is None:
        return DEFAULT_TORNADO_DIAGNOSTICS
    return TornadoDiagnosticSelection(
        output_uh=declared.output_uh,
        output_vort=declared.output_vort,
        output_pv=declared.output_pv,
        output_sws=declared.output_sws,
        output_svs=declared.output_svs,
        output_sps=declared.output_sps,
        output_srs=declared.output_srs,
        output_sgs=declared.output_sgs,
        output_sus=declared.output_sus,
        output_shs=declared.output_shs,
        tornado_report=declared.tornado_report,
    )


def _validate_diagnostics(config: CM1Config) -> None:
    """Refuse every diagnostic selection this port would publish as another.

    The rules live beside the operator that runs them
    (``tornado_diagnostics.validate_tornado_diagnostic_selection``) so a
    programmatic caller that bypasses this door cannot select a diagnostic the
    grid cannot support either; the door names the same defect first, before a
    word of device memory is priced.
    """

    if config.tornado_diagnostics is None:
        return
    from .tornado_diagnostics import validate_tornado_diagnostic_selection

    selection = tornado_diagnostic_selection(config)
    levels = require_vertical_coordinate(config)

    class _GridView:
        nx = config.grid.nx
        ny = config.grid.ny
        nz = config.grid.nz
        ztop_m = levels.ztop_m
        mass_heights_m = levels.mass_heights_m
        face_heights_m = levels.face_heights_m

    try:
        validate_tornado_diagnostic_selection(
            selection,
            grid=_GridView(),
            microphysics=config.physics.component,
            bbc=config.closures.bbc,
            option_prefix="diagnostics",
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CFG-DIAGNOSTIC-SELECTION",
            gate="support-matrix",
            reason=str(error),
            remedy=(
                "turn the named switch off, or change the grid, microphysics "
                "or lower boundary it needs; the diagnostic contract is not "
                "widened from the door"
            ),
            option="diagnostics",
            value=config.tornado_diagnostics.__class__.__name__,
        )


def _validate_lower_boundary(config: CM1Config) -> None:
    """Refuse every lower boundary this port would integrate as another one.

    Each rule is one CM1 r21.1 ``param.F`` stop or forced rewrite; where CM1
    silently rewrites the user's namelist this port refuses instead, so the
    sealed receipt never names a boundary the run did not integrate.
    """

    # The operator that actually runs getcecd owns the arithmetic bound; the
    # door names the same one so a document is refused before anything is
    # allocated.  Imported here, like gm1_runtime does, so the configuration
    # grammar keeps no module-level dependency on a CUDA operator.
    from .cuda_surface_drag import (
        CNSTCD_BINARY32_ROUGHNESS_CEILING,
        CNSTCD_BINARY32_ROUGHNESS_FLOOR,
    )

    closures = config.closures
    # src/param.F:1518-1532
    if closures.bbc not in (1, 2, 3):
        refuse(
            code="CM1-CFG-BBC-RANGE",
            gate="support-matrix",
            reason=(
                "CM1 has exactly three wind lower boundary conditions; any "
                "other value would leave t13 and t23 at k=1 holding the "
                "free-slip copy of the level above while the receipt claimed "
                "a surface the run never applied"
            ),
            remedy="set closures.bbc to 1 (free slip), 2 (no slip) or 3 (semi slip)",
            option="closures.bbc",
            value=closures.bbc,
        )
    # boundaries.bottom is this port's earlier spelling of the same CM1
    # variable (records/CM1-SUPPORT-MATRIX.csv, row SM-BC-BOTTOM).
    if config.boundaries.bottom != closures.bbc:
        refuse(
            code="CM1-CFG-BBC-DISAGREEMENT",
            gate="support-matrix",
            reason=(
                "boundaries.bottom and closures.bbc are two names for CM1's "
                "single bbc selector; with different values the momentum "
                "tendency would use one of them and the receipt would name "
                "the other"
            ),
            remedy=f"set boundaries.bottom to {closures.bbc}, or change closures.bbc",
            option="boundaries.bottom",
            value=config.boundaries.bottom,
        )
    # src/param.F:2073-2088
    if closures.bbc == 3 and closures.sfcmodel < 1:
        refuse(
            code="CM1-CFG-SEMI-SLIP-WITHOUT-SURFACE-LAYER",
            gate="support-matrix",
            reason=(
                "the semi-slip surface stress is ust**2 times the surface "
                "wind direction (src/turb.F:4596-4605) and only a surface "
                "layer computes ust; with no surface layer ust stays zero, so "
                "t13 and t23 at k=1 would be exactly zero and the run would be "
                "frictionless while the receipt reported drag"
            ),
            remedy="set closures.sfcmodel=1 alongside closures.bbc=3, or use closures.bbc=2",
            option="closures.sfcmodel",
            value=closures.sfcmodel,
        )
    # src/param.F:873-889 rewrites the user's bbc; this port refuses instead.
    if closures.sfcmodel >= 1 and closures.bbc != 3:
        refuse(
            code="CM1-CFG-SURFACE-LAYER-WITHOUT-SEMI-SLIP",
            gate="support-matrix",
            reason=(
                "CM1 forces bbc=3 whenever a surface layer is on; with any "
                "other bbc the cd, znt and ust the surface layer computes "
                "would reach no momentum tendency at all, so the run would pay "
                "for a surface layer and integrate a different boundary"
            ),
            remedy="set closures.bbc=3 alongside closures.sfcmodel=1, or set closures.sfcmodel=0",
            option="closures.bbc",
            value=closures.bbc,
        )
    if closures.sfcmodel not in (0, 1):
        refuse(
            code="CM1-CFG-SFCMODEL-UNSUPPORTED",
            gate="support-matrix",
            reason=(
                "sfcmodel=2,3,4,6,7 are the WRF surface-layer schemes and "
                "sfcmodel=5 is CM1's Monin-Obukhov scheme; none is ported, so "
                "selecting one would silently run the neutral constant-Cd drag "
                "while the receipt named a stability-dependent surface"
            ),
            remedy="use closures.sfcmodel=1, CM1's simple neutral scheme, or 0",
            option="closures.sfcmodel",
            value=closures.sfcmodel,
        )
    # src/param.F:1349-1365
    if closures.sfcmodel == 1 and closures.cecd not in (1, 2, 3):
        refuse(
            code="CM1-CFG-CECD-RANGE",
            gate="support-matrix",
            reason=(
                "getcecd has three drag branches (src/sfcphys.F:120-195); any "
                "other value falls through all of them, so cd, znt and ust "
                "keep whatever they held from the previous step and the "
                "surface stress is built from a drag coefficient nothing in "
                "this run computed"
            ),
            remedy=(
                "set closures.cecd to 1 (constant closures.cnstcd), 2 "
                "(Deacon) or 3 (Fairall/Donelan)"
            ),
            option="closures.cecd",
            value=closures.cecd,
        )
    if closures.isfcflx not in (0, 1):
        refuse(
            code="CM1-CFG-ISFCFLX-RANGE",
            gate="config-grammar",
            reason=(
                "CM1's surface-flux switch is 0 or 1 only "
                "(README.namelist:914); any other value would fall through "
                "every branch in sfcphys.F and leave the surface enthalpy "
                "flux silently zero while the receipt named a third setting "
                "that never existed"
            ),
            remedy="set closures.isfcflx to 0 or 1",
            option="closures.isfcflx",
            value=closures.isfcflx,
        )
    # src/param.F:2090-2106
    if closures.isfcflx == 1 and closures.sfcmodel < 1:
        refuse(
            code="CM1-CFG-ISFCFLX-WITHOUT-SURFACE-LAYER",
            gate="support-matrix",
            reason=(
                "src/turb.F:645-649 calls sfcflux only inside the sfcmodel=1 "
                "block, and the enthalpy exchange coefficients it multiplies "
                "are written by getcecd inside the same block "
                "(src/sfcphys.F:197-236); with no surface layer ch and cq "
                "stay zero, so the heat and moisture fluxes are exactly zero "
                "while the receipt claims a surface energy budget"
            ),
            remedy=(
                "set closures.sfcmodel=1 alongside closures.isfcflx=1, or "
                "set closures.isfcflx=0"
            ),
            option="closures.sfcmodel",
            value=closures.sfcmodel,
        )
    # src/param.F:1330-1346, and the simultaneous case src/param.F:1311-1327
    # leaves commented out.
    for name in ("set_znt", "set_ust"):
        value = getattr(closures, name)
        if value not in (0, 1):
            refuse(
                code="CM1-CFG-SET-SWITCH-RANGE",
                gate="support-matrix",
                reason=(
                    "src/sfcphys.F:81-118 tests this switch against 1 "
                    "exactly; any other value silently falls through to the "
                    "cecd branch, so the declared surface is never imposed "
                    "and the drag coefficient comes from somewhere else"
                ),
                remedy=f"set closures.{name} to 0 or 1",
                option=f"closures.{name}",
                value=value,
            )
    if closures.set_znt == 1 and closures.set_ust == 1:
        refuse(
            code="CM1-CFG-SET-ZNT-AND-UST",
            gate="support-matrix",
            reason=(
                "src/sfcphys.F:81 tests set_znt first, so with both imposed "
                "the roughness branch runs and the declared friction velocity "
                "never reaches the surface: ust comes from the log law while "
                "the receipt names cnst_ust"
            ),
            remedy=(
                "impose closures.cnst_znt or closures.cnst_ust, not both"
            ),
            option="closures.set_ust",
            value=closures.set_ust,
        )
    if (closures.set_znt == 1 or closures.set_ust == 1) and closures.sfcmodel != 1:
        refuse(
            code="CM1-CFG-SET-SWITCH-WITHOUT-SURFACE-LAYER",
            gate="support-matrix",
            reason=(
                "getcecd is the only reader of set_znt and set_ust "
                "(src/sfcphys.F:81-118) and src/turb.F:614-616 calls it only "
                "for sfcmodel=1; with any other surface model the declared "
                "roughness or friction velocity is never applied and the "
                "surface stress is built from an unset drag coefficient"
            ),
            remedy=(
                "set closures.sfcmodel=1 alongside closures.set_znt=1 or "
                "closures.set_ust=1"
            ),
            option="closures.sfcmodel",
            value=closures.sfcmodel,
        )
    if closures.set_znt == 1 and not closures.cnst_znt > 0.0:
        refuse(
            code="CM1-CFG-CNST-ZNT-RANGE",
            gate="support-matrix",
            reason=(
                "src/sfcphys.F:85-88 divides by the roughness length and "
                "takes the logarithm of ten over it; at zero the reciprocal "
                "is infinite and the drag coefficient is exactly zero, and at "
                "a negative length the logarithm is not-a-number, so every "
                "surface stress word built from it is not-a-number"
            ),
            remedy=(
                "declare a positive closures.cnst_znt in metres; measured "
                "land values are 0.01 to 1"
            ),
            option="closures.cnst_znt",
            value=closures.cnst_znt,
        )
    if closures.set_ust == 1 and not closures.cnst_ust > 0.0:
        refuse(
            code="CM1-CFG-CNST-UST-RANGE",
            gate="support-matrix",
            reason=(
                "src/sfcphys.F:106 clamps the friction velocity with "
                "max(1.0e-10,ust) before dividing by it, so a non-positive "
                "declaration never reaches the surface: the exponent becomes "
                "the lowest-level wind times 4e9, its exponential overflows, "
                "and the roughness length za/inf is exactly zero, making the "
                "drag coefficient and the 10 m wind ratio not-a-number"
            ),
            remedy=(
                "declare a positive closures.cnst_ust in m/s; measured values "
                "are 0.1 to 2"
            ),
            option="closures.cnst_ust",
            value=closures.cnst_ust,
        )
    if closures.isfcflx == 1 and not closures.cnstce >= 0.0:
        refuse(
            code="CM1-CFG-CNSTCE-RANGE",
            gate="support-matrix",
            reason=(
                "src/sfcphys.F:203-205 assigns this value straight into ch "
                "and cq over water, and src/sfcphys.F:341,349 multiplies the "
                "heat and moisture fluxes by it; a negative coefficient makes "
                "the surface cool the air it is warmer than"
            ),
            remedy="declare a non-negative closures.cnstce, for example 0.001",
            option="closures.cnstce",
            value=closures.cnstce,
        )
    # CM1 bounds cnstcd nowhere -- src/param.F only broadcasts and prints it
    # and README.namelist:1040-1041 gives no range -- so this port refuses
    # only the two ends where src/sfcphys.F:127-130 stops producing a
    # roughness length at all.  Both bounds and their reasons are the
    # operator's own (cm1core.cuda_surface_drag), named once here for the
    # document that declares the key.
    if closures.sfcmodel == 1 and not closures.cnstcd > 0.0:
        refuse(
            code="CM1-CFG-CNSTCD-RANGE",
            gate="support-matrix",
            reason=(
                "src/sfcphys.F:127 clamps the drag coefficient with "
                "max(1.0e-10,cnstcd) before anything reads it, so a "
                "non-positive declaration never reaches the surface: the run "
                "integrates 1e-10 while the receipt names the declared value, "
                "and that clamped Cd overflows src/sfcphys.F:128's "
                "exp(karman/sqrt(Cd)) to a roughness length of exactly zero, "
                "making every word src/sfcphys.F:129-130 builds from 1/znt "
                "not-a-number"
            ),
            remedy="declare a positive drag coefficient, for example 0.001",
            option="closures.cnstcd",
            value=closures.cnstcd,
        )
    if (
        closures.sfcmodel == 1
        and closures.cnstcd < CNSTCD_BINARY32_ROUGHNESS_FLOOR
    ):
        refuse(
            code="CM1-CFG-CNSTCD-RANGE",
            gate="support-matrix",
            reason=(
                "at this drag coefficient karman/sqrt(Cd) is past binary32's "
                "exponential range, so src/sfcphys.F:128's "
                "exp(karman/sqrt(Cd)) overflows and the roughness length "
                "10/inf is exactly zero; the 10 m wind ratio built from 1/znt "
                "is not-a-number and the bbc=3 surface dissipation rate "
                "log((dz1+znt)/znt) at src/solve1.F:956 is infinite, so the "
                "lowest model level's heating is infinite"
            ),
            remedy=(
                "declare a drag coefficient of at least "
                f"{CNSTCD_BINARY32_ROUGHNESS_FLOOR:.6g}; measured surface "
                "values are 1e-3 to 1e-2"
            ),
            option="closures.cnstcd",
            value=closures.cnstcd,
        )
    if (
        closures.sfcmodel == 1
        and closures.cnstcd >= CNSTCD_BINARY32_ROUGHNESS_CEILING
    ):
        refuse(
            code="CM1-CFG-CNSTCD-RANGE",
            gate="support-matrix",
            reason=(
                "at this drag coefficient karman/sqrt(Cd) is below binary32's "
                "2**-24, so src/sfcphys.F:128's exp(karman/sqrt(Cd)) rounds to "
                "exactly 1.0, the subtraction of 1.0 gives zero and the "
                "roughness length 10/0 is infinite; the 10 m wind ratio and "
                "the friction velocity the semi-slip surface stress squares "
                "are then both not-a-number"
            ),
            remedy=(
                "declare a drag coefficient below "
                f"{CNSTCD_BINARY32_ROUGHNESS_CEILING:.6g}; measured surface "
                "values are 1e-3 to 1e-2"
            ),
            option="closures.cnstcd",
            value=closures.cnstcd,
        )
    if closures.bbc != 1 and config.closures.sgs_model == 0:
        refuse(
            code="CM1-CFG-LOWER-BOUNDARY-WITHOUT-SGS",
            gate="support-matrix",
            reason=(
                "the no-slip and semi-slip lower boundaries are boundary "
                "values of the SGS stress tensor (src/turb.F:4515-4700); with "
                "the subgrid model off there is no stress divergence to carry "
                "them, so the surface would exert no force at all"
            ),
            remedy="set closures.sgs_model=1, or keep closures.bbc=1",
            option="closures.sgs_model",
            value=config.closures.sgs_model,
        )
    if closures.bbc != 1 and config.grid.terrain:
        refuse(
            code="CM1-CFG-LOWER-BOUNDARY-TERRAIN",
            gate="support-matrix",
            reason=(
                "over terrain CM1's surface stress is rotated into the "
                "terrain-following metric before it enters the momentum "
                "tendency; applying the flat-ground lower face on a sloped "
                "lower boundary would put the drag vector in the wrong "
                "direction by the terrain slope"
            ),
            remedy="use a flat lower boundary with closures.bbc=2 or 3",
            option="grid.terrain",
            value=config.grid.terrain,
        )


def _validate_restart_selection(config: CM1Config) -> None:
    """CM1's irst/rstnum/iconly surface, refused by key and by value.

    Each of these keys exists in CM1 to select a branch that either runs the
    model or does not.  A declared selector that no lane reads is worse than
    an unknown key, because the receipt records it and the run behaves as if
    it had not been written.
    """

    run = config.run
    for name, value in (("run.irst", run.irst), ("run.iconly", run.iconly)):
        if value is not None and config.run.lane is not Lane.CARTESIAN:
            refuse(
                code="CM1-CFG-RESTART-LANE",
                gate="cartesian-runtime",
                reason=(
                    f"{name} selects CM1's restart/initial-condition branch and only "
                    f"the cartesian lane has one; in lane {config.run.lane.value!r} "
                    "the value would be recorded in the receipt and the run would "
                    "integrate from the initial condition regardless"
                ),
                remedy=f"remove {name}, or declare run.lane = \"cartesian\"",
                option=name,
                value=value,
            )
    if run.rstnum is not None and not run.restarting:
        refuse(
            code="CM1-CFG-RESTART-SELECTION",
            gate="cartesian-runtime",
            reason=(
                "cm1.F:2660 reads the restart file only under irst=1, so "
                f"run.rstnum={run.rstnum} would select a restart nothing opens and "
                "the run would start from the initial condition while its receipt "
                "recorded a restart number"
            ),
            remedy="declare run.irst = 1 beside run.rstnum, or remove run.rstnum",
            option="run.rstnum",
            value=run.rstnum,
        )
    if run.restarting and run.rstnum is None:
        refuse(
            code="CM1-CFG-RESTART-SELECTION",
            gate="cartesian-runtime",
            reason=(
                "README.namelist:229-231 makes rstnum the number of the restart to "
                "continue from, and a restart directory holds every restart the "
                "previous run wrote; without it the resume would have to guess "
                "which model time to continue from"
            ),
            remedy="declare run.rstnum = N beside run.irst = 1",
            option="run.irst",
            value=run.irst,
        )
    if run.restarting and run.initial_condition_only:
        refuse(
            code="CM1-CFG-RESTART-SELECTION",
            gate="cartesian-runtime",
            reason=(
                "cm1.F:4342 stops for iconly only when the run is a startup or an "
                "interpolated restart, so on a plain restart CM1 ignores iconly and "
                "integrates; a caller who declared both would ask to see an initial "
                "condition and get the whole run instead"
            ),
            remedy="declare run.iconly = 1 with run.irst = 0, or drop run.iconly",
            option="run.iconly",
            value=run.iconly,
        )


def _validate_dynamics_words(config: CM1Config) -> None:
    """Cross-checks CM1 makes between its own ``param2``/``param3`` words.

    Every refusal here names a wrong result, not a taste.  None of them can
    fire on a configuration CM1 r21.1 itself would integrate correctly.
    """

    from .gm1_runtime import CM1_MAX_ABS_FCOR_S_INV

    d = config.dynamics
    if d.beta_plane == 1 and d.coriolis != 1:
        refuse(
            code="CM1-CFG-BETAPLANE-WITHOUT-CORIOLIS",
            gate="config-grammar",
            reason=(
                "param.F:8955 builds a latitude-varying f2d, but solve2.F:632 "
                "only evaluates the Coriolis term at all under icor=1; this "
                "document declares a beta plane for a term that would never "
                "be applied, so the run would integrate no rotation while its "
                "receipt named one"
            ),
            remedy="set dynamics.coriolis = 1, or dynamics.beta_plane = 0",
            option="dynamics.beta_plane",
            value=d.beta_plane,
        )
    if d.beta_plane == 1 and abs(d.fcor_s_inv) > CM1_MAX_ABS_FCOR_S_INV:
        refuse(
            code="CM1-CFG-BETAPLANE-LATITUDE",
            gate="config-grammar",
            reason=(
                "param.F:8952 takes phi = asin( fcor/(2*omega) ) with omega = "
                f"{CM1_MAX_ABS_FCOR_S_INV / 2:.6e} 1/s (constants.F:87); "
                f"|fcor| = {abs(d.fcor_s_inv):.6e} is outside asin's domain, "
                "so phi, beta0 and every f2d word would be NaN and the whole "
                "momentum field would be non-finite after the first stage"
            ),
            remedy=(
                "set |dynamics.fcor_s_inv| to at most "
                f"{CM1_MAX_ABS_FCOR_S_INV!r}, which is 2*omega, the Coriolis "
                "parameter at the pole"
            ),
            option="dynamics.fcor_s_inv",
            value=d.fcor_s_inv,
        )
    frame = (config.closures.u_move_m_s, config.closures.v_move_m_s)
    if d.adaptive_timestep and config.closures.moving_frame and frame != (0.0, 0.0):
        # Measured, not assumed: a run of this combination is refused by the
        # Rust history publisher partway through, after GPU time has been
        # spent.  rust/cm1-state/src/domain_location.rs:254-282 proves
        # cm1.F:3512's running sum exact by showing it equals n*dt*umove, and
        # that identity needs one dt for the whole run.  misclibs.F:785 gives
        # every step its own length, so the sum is still CM1's integral but is
        # no longer that product, and the publisher stops the run rather than
        # place a frame on the wrong patch of ground.  None of CM1 r21.1's 23
        # shipped decks sets both: imove=1 appears only in supercell and
        # squall_line, and both set adapt_dt=0.
        refuse(
            code="CM1-CFG-ADAPTIVE-TRANSLATING-FRAME",
            gate="config-grammar",
            reason=(
                "dynamics.adaptive_timestep gives every large step its own "
                "length (misclibs.F:785), and the published domain location's "
                "exactness proof -- rust/cm1-state/src/domain_location.rs, "
                "which checks cm1.F:3512's running sum against "
                "completed_steps*dt*umove -- holds only for one dt.  The "
                "history publisher refuses the mismatch mid-run, so this "
                "combination costs GPU time and publishes nothing"
            ),
            remedy=(
                "set closures.moving_frame = false with zero frame speeds and "
                "size the domain for the storm's ground track, or set "
                "dynamics.adaptive_timestep = false; teaching "
                "domain_location.rs to carry the running sum without the "
                "constant-dt identity would retire this refusal"
            ),
            option="dynamics.adaptive_timestep",
            value=d.adaptive_timestep,
        )
    if d.lspgrad_value == 1 and (d.coriolis != 1 or d.beta_plane != 0):
        # CM1 itself does not check this pair.  solve1.F:606-609 subtract
        # fcor*(v0+vmove) and add fcor*(u0+umove) whatever icor says, which
        # is the base state's own Coriolis force taken away; only the f-plane
        # term of solve2.F:639-653, fcor*(V+Vmove), puts it back.
        refuse(
            code="CM1-CFG-LSPGRAD-WITHOUT-FPLANE-CORIOLIS",
            gate="config-grammar",
            reason=(
                "lspgrad=1 subtracts fcor x (V0 + Vmove) from the held u/v "
                "tendency at every point (solve1.F:606-609) so that the "
                "f-plane Coriolis term fcor x (V + Vmove) (solve2.F:639-653) "
                "acts only on the departure from the base state.  "
                + (
                    "With icor=0 there is no Coriolis term to balance, so the "
                    "base-state wind itself would be accelerated at "
                    "fcor*|V0+Vmove| for the whole run"
                    if d.coriolis != 1
                    else "The beta-plane branch (solve2.F:655-673) uses f2d(y) "
                    "and omits the frame add-back, so a uniform fcor profile "
                    "cancels it only on the domain's central latitude and a "
                    "residual acceleration of (f2d-fcor)*V0 would remain "
                    "everywhere else"
                )
            ),
            remedy=(
                "set dynamics.coriolis = 1 with dynamics.beta_plane = 0, or "
                "dynamics.lspgrad = 0"
            ),
            option="dynamics.lspgrad",
            value=d.lspgrad,
        )
    if d.lspgrad_value == 1 and config.closures.do_adapt_move:
        # azimavg.F:4446-4449 rebases ug/vg and change_uvmove rebases u0/v0
        # when the adaptive controller moves the frame.
        refuse(
            code="CM1-CFG-LSPGRAD-ADAPTIVE-MOVE",
            gate="config-grammar",
            reason=(
                "lspgrad=1 forces with fcor*(V0 + Vmove) from the frame-relative "
                "base-state wind (base.F:2661-2674).  An adaptive moving domain "
                "changes umove/vmove mid-run and CM1 rebases u0/v0 and ug/vg "
                "with it (azimavg.F:4440-4449); this lane computes the forcing "
                "profile once from the initial frame, so after the first frame "
                "update the forcing and the carried winds would name different "
                "frames"
            ),
            remedy=(
                "declare a fixed frame (closures.do_adapt_move = false), or set "
                "dynamics.lspgrad = 0"
            ),
            option="dynamics.lspgrad",
            value=d.lspgrad,
        )
    if d.beta_plane == 1 and config.closures.moving_frame and frame != (0.0, 0.0):
        refuse(
            code="CM1-CFG-BETAPLANE-TRANSLATING-FRAME",
            gate="config-grammar",
            reason=(
                "solve2.F:645,647 add +vmove/+umove inside the f-plane "
                "Coriolis term because base.F:2667 already subtracted the "
                "frame velocity from every carried wind; solve2.F:667-672 "
                "omit that add-back in the beta-plane branch.  In this frame "
                f"({frame[0]}, {frame[1]}) m/s the u and v accelerations "
                f"would be short by {abs(d.fcor_s_inv * frame[1]):.3e} and "
                f"{abs(d.fcor_s_inv * frame[0]):.3e} m/s^2 at every point for "
                "the whole run -- a steady spurious acceleration, not a "
                "rounding difference"
            ),
            remedy=(
                "set closures.moving_frame = false with zero frame speeds, "
                "or use dynamics.beta_plane = 0, whose f-plane branch carries "
                "CM1's frame add-back exactly"
            ),
            option="dynamics.beta_plane",
            value=d.beta_plane,
        )


def _validate_sgs_startup_ramp(config: CM1Config) -> None:
    """Admit CM1's ``ramp_sgs``/``ramp_time`` by value (README.namelist:1081-1087)."""

    closures = config.closures
    if closures.ramp_sgs not in (0, 1):
        refuse(
            code="CM1-CFG-SGS-RAMP",
            gate="support-matrix",
            reason=(
                f"ramp_sgs={closures.ramp_sgs} is neither of README.namelist:1082's "
                "0 (no) and 1 (yes); src/turb.F:240 tests ramp_sgs.ge.1, so the "
                "document would name a ramp CM1 does not define"
            ),
            remedy="declare closures.ramp_sgs = 0 or 1",
            option="closures.ramp_sgs",
            value=closures.ramp_sgs,
        )
    if closures.ramp_time < 0.0:
        refuse(
            code="CM1-CFG-SGS-RAMP-TIME",
            gate="support-matrix",
            reason=(
                f"a {closures.ramp_time} s ramp is no duration: src/param.F:750 "
                "raises it to 1e-10 s, so the subgrid model would start at full "
                "strength while the document named a ramp"
            ),
            remedy="declare a nonnegative closures.ramp_time in seconds (CM1 ships 1800.)",
            option="closures.ramp_time",
            value=closures.ramp_time,
        )
    if closures.ramp_sgs == 1 and config.run.lane is not Lane.CARTESIAN:
        refuse(
            code="CM1-CFG-SGS-RAMP-LANE",
            gate="support-matrix",
            reason=(
                f"the {config.run.lane.value} lane's closure carries no "
                "src/turb.F:240-262 start-up ramp, so the run would mix at full "
                "subgrid strength from the first step while the document named "
                "ramp_sgs=1"
            ),
            remedy="use the cartesian lane, or declare closures.ramp_sgs = 0",
            option="closures.ramp_sgs",
            value=closures.ramp_sgs,
        )


def _validate_adaptive_move(config: CM1Config) -> None:
    """Admit CM1's ``do_adapt_move``/``adapt_move_frq`` by key and by value.

    ``src/param.F:703-717`` is CM1's own coupling: an adaptive moving domain
    zeroes ``umove``/``vmove`` and raises ``imove`` to 1, because the whole
    point of the controller is that the frame velocity is a model variable and
    not a namelist constant.  ``src/cm1.F:3641-3651`` is the other coupling
    this port has to state: CM1 finds the storm centre only on a step that is
    already writing output, so a cadence that never lands on one leaves
    ``xcenter``/``ycenter`` at ``cm1.F:2527-2529``'s domain centre and the
    controller computes a zero correction forever.
    """

    closures = config.closures
    frequency = closures.adapt_move_frq
    if frequency <= 0.0:
        refuse(
            code="CM1-CFG-ADAPT-MOVE-FRQ",
            gate="support-matrix",
            reason=(
                f"{frequency} s is not a re-centring interval; cm1.F:3671-3679 "
                "advances adaptmovetim by exactly this number each time the "
                "controller fires, so a zero or negative cadence leaves "
                "adaptmovetim at or below the model time forever and "
                "azimavg.F:4431-4435 divides the centre departure by it -- the "
                "frame velocity would be updated on every large step from a "
                "division by a nonpositive interval, not once per cadence"
            ),
            remedy=(
                "declare a positive closures.adapt_move_frq in seconds; CM1's "
                "own deck ships run/namelist.input:361 adapt_move_frq = 3600.0"
            ),
            option="closures.adapt_move_frq",
            value=frequency,
        )
    # input.F:200 reads adapt_move_frq into a CM1 ``real`` and param.F:603
    # broadcasts it as MPI_REAL, so the number azimavg.F:4431 divides by is
    # the binary32 one whatever the document wrote.
    if cm1_real(frequency) != frequency:
        refuse(
            code="CM1-CFG-ADAPT-MOVE-FRQ-PRECISION",
            gate="support-matrix",
            reason=(
                f"{frequency} s is not a CM1 `real`: azimavg.F:4431-4435 would "
                f"divide the centre departure by {cm1_real(frequency)} s while "
                "cm1.F:3679 advanced the double-precision adaptmovetim by "
                f"{frequency} s, so the controller's response gain and its own "
                "firing clock would disagree and the two would drift apart for "
                "the rest of the integration"
            ),
            remedy=(
                "declare a cadence exactly representable in binary32, such as "
                f"{cm1_real(frequency)!r}"
            ),
            option="closures.adapt_move_frq",
            value=frequency,
        )
    if not closures.do_adapt_move:
        if frequency != FIXED_FRAME_ADAPTIVE_MOVE.adapt_move_frq:
            refuse(
                code="CM1-CFG-ADAPT-MOVE-INACTIVE",
                gate="support-matrix",
                reason=(
                    f"a {frequency} s re-centring cadence would be silently "
                    "ignored while the controller is off: cm1.F:3669 reads "
                    "adapt_move_frq only inside `IF( do_adapt_move )`, so the "
                    "document would name a storm-following frame and the run "
                    "would integrate a fixed one"
                ),
                remedy=(
                    "set closures.do_adapt_move = true, or leave "
                    "closures.adapt_move_frq at CM1's own "
                    f"{FIXED_FRAME_ADAPTIVE_MOVE.adapt_move_frq!r}"
                ),
                option="closures.do_adapt_move",
                value=closures.do_adapt_move,
            )
        return
    if not closures.moving_frame:
        refuse(
            code="CM1-CFG-ADAPT-MOVE-IMOVE",
            gate="support-matrix",
            reason=(
                "param.F:711 raises imove to 1 for every do_adapt_move run, so "
                "a document that also says the frame is fixed describes a run "
                "that cannot exist: azimavg.F:4351-4366 rebases every wind word "
                "in the domain at each cadence while base.F:2663-2669's "
                "convention -- published wind = ground wind minus the frame -- "
                "would be recorded as inactive, and every wind the run "
                "published would be wrong by the accumulated frame velocity "
                "with nothing in the artifact to say so"
            ),
            remedy="set closures.moving_frame = true beside closures.do_adapt_move",
            option="closures.moving_frame",
            value=closures.moving_frame,
        )
    for name, value in (
        ("closures.u_move_m_s", closures.u_move_m_s),
        ("closures.v_move_m_s", closures.v_move_m_s),
    ):
        if value != 0.0:
            refuse(
                code="CM1-CFG-ADAPT-MOVE-FRAME-SPEED",
                gate="support-matrix",
                reason=(
                    f"param.F:706-707 sets umove and vmove to 0.0 for every "
                    f"do_adapt_move run, so the declared {value} m/s would "
                    "never reach base.F:2667: the base-state winds would not be "
                    "shifted into the frame the document names, the controller "
                    "would start from zero regardless, and every published wind "
                    f"would be wrong by {value} m/s against the document that "
                    "describes it"
                ),
                remedy=(
                    f"set {name} = 0.0; the adaptive controller derives the "
                    "frame velocity from the storm and a declared constant is "
                    "not an initial guess CM1 reads"
                ),
                option=name,
                value=value,
            )
    if config.dynamics.adaptive_timestep:
        # This refusal used to name the port's own defect: ``integrate()``
        # scheduled ``dowriteout``/``getdbz`` on the *fixed-step* positions of
        # the output times, which under adapt_dt are not the steps that write.
        # ``CudaCartesianRuntime.schedule_adaptive_output_step`` fixed that --
        # it applies cm1.F:3177,3215's own test after getnewdt has chosen the
        # step, and ``history_frame_steps`` now holds the steps the run really
        # publishes at.  What is left is CM1's side, below: the cadence itself
        # is the one schedule getnewdt is not given.
        refuse(
            code="CM1-CFG-ADAPT-MOVE-ADAPTIVE-TIMESTEP",
            gate="fixed-clock",
            reason=(
                "dynamics.adaptive_timestep gives every large step its own "
                "length (misclibs.F:785), chosen from that code's own cflmax, "
                "and adaptmovetim is the one output-like time getnewdt is "
                "never asked to land on: misclibs.F:593-596 passes it stattim, "
                "taptim, rsttim, prcltim, diagtim, azimavgtim and hifrqtim, "
                "and not the move cadence.  cm1.F:3671's test is also the one "
                "with no tolerance -- a bare mtime >= adaptmovetim, where "
                "cm1.F:3177 allows 0.1*dbldt -- so which step crosses the "
                "cadence is not known until the run takes it, and it need not "
                "be a step that writes a frame.  On a step that does not, "
                "cm1.F:3642-3643 skips the centre diagnosis while "
                "cm1.F:3671-3679 updates the frame velocity anyway, and the "
                "domain is steered from the previous cadence's centre or from "
                "cm1.F:2529's domain centre -- the wrong result "
                "CM1-CFG-ADAPT-MOVE-CADENCE-UNWRITTEN prevents under a "
                "declared dt, whose guarantee (a cadence that is a whole "
                "number of history intervals is crossed on a step that "
                "publishes) is read off the declared step schedule that "
                "adapt_dt replaces.  Nothing has measured the pair: no run of "
                "do_adapt_move with adapt_dt exists in either code here"
            ),
            remedy=(
                "set dynamics.adaptive_timestep = false beside "
                "closures.do_adapt_move, or declare a fixed frame; a "
                "differential of the pair against CM1 r21.1, showing both "
                "codes crossing the cadence on the same step and diagnosing "
                "the centre there, would retire this refusal"
            ),
            option="dynamics.adaptive_timestep",
            value=config.dynamics.adaptive_timestep,
        )
    # cm1.F:3641-3651.  The centre diagnosis runs only on a step that writes
    # history, high-frequency or azimuthal-average output, and this port's
    # Cartesian lane writes history frames and nothing else, so a cadence that
    # is not a whole number of history intervals fires on steps that publish
    # nothing.
    interval = config.run.history_interval_s
    multiples = frequency / interval
    if not math.isfinite(multiples) or multiples < 1.0 or not multiples.is_integer():
        refuse(
            code="CM1-CFG-ADAPT-MOVE-CADENCE-UNWRITTEN",
            gate="fixed-clock",
            reason=(
                f"a {frequency} s cadence is not a whole number of this run's "
                f"{interval} s history intervals. cm1.F:3642-3651 diagnoses the "
                "storm centre only on a step that is already writing output, so "
                "on every cadence that lands between two frames getpsmth and "
                "getpcent are skipped and azimavg.F:4431 is evaluated with "
                "xcenter still at cm1.F:2529's domain centre -- the correction "
                "is identically zero, umove and vmove never leave 0.0, and the "
                "run integrates a fixed frame while its document names an "
                "adaptive one"
            ),
            remedy=(
                "declare closures.adapt_move_frq as a positive integer multiple "
                f"of run.history_interval_s = {interval!r}, or shorten "
                "run.history_interval_s so the cadence lands on a frame"
            ),
            option="closures.adapt_move_frq",
            value=frequency,
        )


def _validate_first_slice(config: CM1Config) -> None:
    restart_clock = abs(config.run.restart_interval_s)
    if restart_clock == 0.0:
        refuse(
            code="CM1-CFG-NUMERIC-RANGE",
            gate="config-grammar",
            reason=(
                "src/cm1.F:3110-3114 reads rstfrq=0 as 'write one restart file "
                "and stop', with dosolve set false, so CM1 writes it from the "
                "initialized state at nstep=0 -- the branch its own comment "
                "marks as useful for ensemble DA. This port's restart boundary "
                "is the post-solve3 state of a completed large step "
                "(src/cm1.F:4246), and there is no completed large step at "
                "nstep=0, so a zero cadence has no capture point here and the "
                "run would advance to its full duration having written nothing"
            ),
            remedy=(
                "use a positive interval to write restarts during the run, or "
                "-3600 only in native-context; there is no GM1 equivalent of "
                "CM1's rstfrq=0 initial-state restart"
            ),
            option="run.restart_interval_s",
            value=config.run.restart_interval_s,
        )
    if config.run.restart_interval_s < 0.0 and config.run.lane is not Lane.NATIVE_CONTEXT:
        refuse(
            code="CM1-CFG-RESTART-SIGN",
            gate="fixed-clock",
            reason="a negative restart frequency is native CM1's disable sentinel, not a product checkpoint cadence",
            remedy="use a positive interval for product/dry lanes; only native-context preserves -3600",
            option="run.restart_interval_s",
            value=config.run.restart_interval_s,
        )
    # misclibs.F:659-681.  With adapt_dt on, getnewdt ramps the step down so
    # it lands exactly on the next output, restart or statistics time: the
    # events set the steps, not the other way round, so requiring an integer
    # multiple of the declared dt would refuse exactly the configurations
    # CM1's own controller exists to serve.  What must still hold is that
    # each event divides the run, or the completed state is never published.
    if config.dynamics.adaptive_timestep:
        for name, interval in (
            ("run.history_interval_s", config.run.history_interval_s),
            ("run.restart_interval_s", restart_clock),
        ):
            frames = config.run.duration_s / interval
            if not math.isfinite(frames) or frames < 1.0 or not frames.is_integer():
                refuse(
                    code="CM1-CFG-CLOCK-ALIGNMENT",
                    gate="fixed-clock",
                    reason=(
                        "with dynamics.adaptive_timestep the events set the "
                        "steps, not the other way round, but an interval that "
                        "does not divide run.duration_s still leaves the "
                        "completed state of the run unpublished"
                    ),
                    remedy=f"set {name} to a value that divides run.duration_s exactly",
                    option=name,
                    value=interval,
                )
    else:
        for name, interval in (
            ("run.duration_s", config.run.duration_s),
            ("run.history_interval_s", config.run.history_interval_s),
            ("run.restart_interval_s", restart_clock),
        ):
            steps = interval / config.run.dt_s
            if not math.isfinite(steps) or steps < 1.0 or not steps.is_integer():
                refuse(
                    code="CM1-CFG-CLOCK-ALIGNMENT",
                    gate="fixed-clock",
                    reason="the fixed-timestep scheduler cannot place this event on an exact large-step boundary",
                    remedy=f"set {name} to an integer multiple of run.dt_s",
                    option=name,
                    value=interval,
                )
    for name, interval in (
        ("run.history_interval_s", config.run.history_interval_s),
        ("run.restart_interval_s", restart_clock),
    ):
        if interval > config.run.duration_s:
            refuse(
                code="CM1-CFG-EVENT-OUTSIDE-RUN",
                gate="fixed-clock",
                reason=(
                    "the first sealed profile requires each declared event to "
                    "occur during the run. src/param.F:6338 starts rsttim at "
                    "rstfrq and src/cm1.F:3186 writes only once the model time "
                    "reaches it, so an interval longer than run.duration_s "
                    "would end the run before its first event and leave nothing "
                    "to resume from while the receipt named a cadence"
                ),
                remedy=f"set {name} less than or equal to run.duration_s",
                option=name,
                value=interval,
            )
    _validate_restart_selection(config)
    if config.grid.terrain and config.run.lane is not Lane.REAL_PRODUCT:
        _choice(
            "grid.terrain",
            True,
            {False},
            "terrain changes CM1 metrics and lower-boundary operators and is admitted only by the separately identified real-atmosphere profile",
        )
    for axis, enabled in (
        ("x", config.grid.stretch_x),
        ("y", config.grid.stretch_y),
    ):
        if enabled:
            _choice(
                f"grid.stretch_{axis}",
                True,
                {False},
                "horizontal stretching changes uh/uf and vh/vf, which every advection, "
                "deformation and open-boundary operator in this port carries as exactly "
                "one; a stretched row would be advected and damped with the wrong "
                "horizontal metric at every column",
            )
    require_vertical_coordinate(config)
    _validate_reflectivity(config)
    _validate_lower_boundary(config)
    _validate_dynamics_words(config)
    _validate_surface(config)
    _validate_diagnostics(config)
    # boundaries.bottom is CM1's bbc under this port's earlier spelling; it is
    # checked against closures.bbc just above, so what remains is CM1's own
    # lateral family, its top boundary and its radiation branch.
    _validate_boundary_family(config)
    _validate_horizontal_damping(config)
    if config.run.lane is Lane.DRY_DYCORE:
        if config.closures.sgs_model != 0:
            _choice(
                "closures.sgs_model",
                config.closures.sgs_model,
                {0},
                "the periodic dry bootstrap disables SGS",
            )
    elif config.closures.sgs_model not in (1, 3, 4):
        _choice(
            "closures.sgs_model",
            config.closures.sgs_model,
            {1, 3, 4},
            "README.namelist:334-352: this lane integrates the Deardorff TKE "
            "closure (1) and CM1's two two-part near-wall variants (3, 4); the "
            "Smagorinsky and NBA models have no kernels, so the TKE closure "
            "would mix every column while the receipt named another model",
        )
    from .cuda_gm1_two_part_sgs import validate_two_part_selection

    validate_two_part_selection(
        sgsmodel=config.closures.sgs_model if config.run.lane is not Lane.DRY_DYCORE else 1,
        t2p_avg=config.closures.t2p_avg_value,
        t2p_avg_declared=config.closures.t2p_avg is not None,
        terrain=bool(config.grid.terrain),
        bbc=config.closures.bbc,
        sfcmodel=config.closures.sfcmodel,
    )
    # README.namelist:391-398 and src/solve1.F:404-438.  irdamp=2 damps toward
    # the evolving horizontal average rather than the frozen base state, which
    # is what CM1 recommends for a long run and what two of its own shipped
    # decks use (cpm_RadConvEquil, les_HurrCoast).
    if config.run.lane is Lane.DRY_DYCORE:
        _choice(
            "closures.upper_damping",
            config.closures.upper_damping,
            {0},
            "the dry operator bootstrap builds no Rayleigh coefficients at all",
        )
    else:
        _integer_choice(
            "closures.upper_damping",
            config.closures.upper_damping,
            {0, 1, 2},
            "src/solve1.F:404-438 has exactly three branches: off, damp toward "
            "the base state, damp toward the horizontal average; another value "
            "would leave uten1/vten1/wten1/thten1 holding the previous step's "
            "words while the receipt named a damper",
        )
    # src/param.F:725.  CM1 rewrites hrdamp to irdamp when both are on; a
    # document that names a different pair would be integrated as one it does
    # not say.
    if config.closures.upper_damping >= 1 and config.closures.hrdamp >= 1:
        if config.closures.hrdamp != config.closures.upper_damping:
            refuse(
                code="CM1-CFG-HRDAMP-FOLLOWS-IRDAMP",
                gate="support-matrix",
                reason=(
                    "src/param.F:725 sets hrdamp=irdamp whenever both are on, so "
                    f"CM1 would integrate hrdamp={config.closures.upper_damping} "
                    f"while this document names {config.closures.hrdamp}; "
                    "src/solve1.F:404 then selects one tendency branch for both "
                    "dampers, not one each"
                ),
                remedy=f"set closures.hrdamp to {config.closures.upper_damping}, or set closures.upper_damping to {config.closures.hrdamp}",
                option="closures.hrdamp",
                value=config.closures.hrdamp,
            )
    target_implicit_diffusion = config.run.lane is not Lane.DRY_DYCORE
    if config.closures.vertical_implicit_diffusion is not target_implicit_diffusion:
        _choice(
            "closures.vertical_implicit_diffusion",
            config.closures.vertical_implicit_diffusion,
            {target_implicit_diffusion},
            "the dry operator bootstrap disables turbulence diffusion while the official SGS branch requires it",
        )
    for name, value in (
        ("closures.upper_damping_depth_m", config.closures.upper_damping_depth_m),
        ("closures.upper_damping_rate_s_inv", config.closures.upper_damping_rate_s_inv),
    ):
        if value < 0.0 or (config.closures.upper_damping and value == 0.0):
            refuse(
                code="CM1-CFG-NUMERIC-RANGE",
                gate="config-grammar",
                reason="a damping magnitude cannot be negative",
                remedy="use zero only when damping is disabled, otherwise use a positive value",
                option=name,
                value=value,
            )
    try:
        domain_top = config.grid.nz * config.grid.dz_m
    except OverflowError:
        domain_top = math.inf
    if not math.isfinite(domain_top):
        refuse(
            code="CM1-CFG-GRID-EXTENT",
            gate="support-matrix",
            reason="the derived uniform-grid top is not finite",
            remedy="use one of the checked fixed profiles",
            option="grid.nz*grid.dz_m",
            value=domain_top,
        )
    if config.closures.upper_damping and (
        config.closures.upper_damping_depth_m >= domain_top
    ):
        refuse(
            code="CM1-CFG-DAMPING-DEPTH",
            gate="support-matrix",
            reason="active damping requires positive depth/rate and a layer beginning above the model bottom",
            remedy="use positive depth/rate with depth below grid.nz * grid.dz_m",
            option="closures.upper_damping_depth_m",
            value=config.closures.upper_damping_depth_m,
        )
    # CM1 base.F:2663-2669's frame velocity.
    #
    # param.F:1906-1950 also stops CM1 when a nonzero umove/vmove meets a
    # rigid lateral wall (wbc/ebc/sbc/nbc of 3 or 4).  No guard is written
    # for that here because this configuration cannot express one: every
    # lane pins its lateral boundaries to open (2) or periodic (1), and both
    # are frame-invariant.  A guard that can never fire is not a guard.
    if config.closures.moving_frame:
        for name, value in (
            ("closures.u_move_m_s", config.closures.u_move_m_s),
            ("closures.v_move_m_s", config.closures.v_move_m_s),
        ):
            if abs(value) > MAX_FRAME_SPEED_M_S:
                refuse(
                    code="CM1-CFG-MOVING-FRAME-SPEED",
                    gate="support-matrix",
                    reason=f"{value} m/s is not a storm motion; base.F:2667 subtracts the frame velocity from every base-state wind word, so the declared hodograph would be replaced by a uniform frame-relative jet of that magnitude through the whole depth of the domain",
                    remedy=f"declare a frame velocity within {MAX_FRAME_SPEED_M_S} m/s; CM1's own supercell deck uses umove=12.5, vmove=3.0",
                    option=name,
                    value=value,
                )
            # umove/vmove are CM1 `real` (base.F:2661-2674) and GM1's frame
            # crosses to Rust as binary32 for the same reason.  A declared
            # speed that does not survive binary32 is subtracted from the base
            # winds as one number and integrated by cm1.F:3512 as another, so
            # the published domain location walks away from the carriers it
            # describes at the difference times the model time.
            if cm1_real(value) != value:
                refuse(
                    code="CM1-CFG-MOVING-FRAME-PRECISION",
                    gate="support-matrix",
                    reason=f"{value} m/s is not a CM1 `real`: base.F:2667 would subtract {cm1_real(value)} m/s from every base-state wind word while cm1.F:3512 integrated {value} m/s into domainlocx/domainlocy, so every published domain location would drift from the carriers it describes by {abs(cm1_real(value) - value)} m/s times the model time",
                    remedy=f"declare a frame velocity exactly representable in binary32, such as {cm1_real(value)!r}; CM1's own supercell deck uses umove=12.5, vmove=3.0",
                    option=name,
                    value=value,
                )
    _validate_adaptive_move(config)
    _validate_sgs_startup_ramp(config)
    if not config.closures.moving_frame and (
        config.closures.u_move_m_s != 0.0 or config.closures.v_move_m_s != 0.0
    ):
        refuse(
            code="CM1-CFG-MOVING-FRAME-INACTIVE",
            gate="support-matrix",
            reason="nonzero translation speeds would be silently ignored while the moving-frame branch is off",
            remedy="set both speeds to 0.0 or enable closures.moving_frame",
            option="closures.moving_frame",
            value=config.closures.moving_frame,
        )
    # CM1's own microphysics selectors are consumed by the configurable
    # Cartesian lane only.  Every other lane is a sealed profile whose
    # component and constants are fixed by its own record, so a declared value
    # there would be written into that run's identity and receipts while the
    # sealed profile continued to execute the value it was sealed with.
    if config.run.lane is not Lane.CARTESIAN:
        for name, value in (
            ("physics.ihail", config.physics.ihail),
            ("physics.ndcnst", config.physics.ndcnst),
        ):
            if value is not None:
                refuse(
                    code="CM1-CFG-PHYSICS-OPTION-SCOPE",
                    gate="support-matrix",
                    reason=(
                        f"{name} is consumed by the configurable Cartesian lane "
                        f"only; lane {config.run.lane.value!r} runs a sealed "
                        "profile whose column component and its constants are "
                        "fixed by records/CM1-PROFILE-MANIFEST.json, so this "
                        "value would be sealed into the run's configuration "
                        "identity and its receipts while the profile continued "
                        "to execute the value it was sealed with"
                    ),
                    remedy=(
                        f"remove {name}, or set run.lane = "
                        f"{Lane.CARTESIAN.value!r}"
                    ),
                    option=name,
                    value=value,
                )
    if config.run.lane is Lane.DRY_DYCORE:
        _choice(
            "physics.component",
            config.physics.component,
            {"off"},
            "a dry-dycore comparison cannot include substituted column physics",
        )
        _choice(
            "physics.equation_set",
            config.physics.equation_set,
            {1},
            "the dry bootstrap is a separately identified eqtset=1 profile",
        )
        _require_dry_bootstrap_profile(config)
    elif config.run.lane is Lane.CARTESIAN:
        # README.namelist:233-236 / cm1.F:4341-4346: iconly=1 builds the
        # initial condition and stops, so no column component is compiled or
        # launched and a dry initial state (imoist=0, component='off') is a
        # legitimate thing to inspect.  Every integrating run still binds the
        # one component this runtime executes.
        initialization = config.initialization
        # CM1 has one iconly word (README.namelist:233-236).  run.iconly is
        # the spelling this port already ships and seals, so the dry
        # inspection lane reads it rather than a second key for the same
        # namelist entry.
        inspecting_only = config.run.initial_condition_only
        from .moisture_species import MOISTURE_COMPONENT_IDS

        _choice(
            "physics.component",
            config.physics.component,
            (
                set(MOISTURE_COMPONENT_IDS) | {"off"}
                if inspecting_only
                else set(MOISTURE_COMPONENT_IDS)
            ),
            "the Cartesian runtime binds only the column components whose CM1 "
            "q3d scalar table, persistence inventory and dense-ice branch this "
            "port has transcribed"
            + (
                "; with run.iconly = 1 it also admits 'off', because "
                "nothing is compiled or launched"
                if inspecting_only
                else ""
            ),
        )
        _choice(
            "physics.equation_set",
            config.physics.equation_set,
            {1, 2} if inspecting_only else {2},
            "the Cartesian large step executes the eqtset=2 source order only",
        )
        if initialization is not None and initialization.imoist == 0 and not inspecting_only:
            refuse(
                code="CM1-CFG-CARTESIAN-DRY-INTEGRATION",
                gate="cartesian-runtime",
                reason="initialization.imoist = 0 builds base.F:592-609's dry base state, but this lane's large step is bound to the wsm6-mp6 column and would run six-species microphysics on it every step; the run would carry moisture the document declared absent",
                remedy="declare run.iconly = 1 to inspect the dry initial condition, or initialization.imoist = 1 to integrate",
                option="initialization.imoist",
                value=0,
            )
        # A run starts only from a sounding whose time integration matched CM1 r21.1's like for like (isnd = 5,
        # 7 and 17, and 9).  The integration from base.F's isnd = 4, 10 and 11 and from the moist benchmark bubble
        # (init3d.F:579-647) was compared with CM1 over the first 15 minutes, both models on Morrison at
        # eqtset = 2, and did not match it (w and theta' differ from the first minute;
        # records/CM1-INITIALIZATION-VALIDATION.json has the numbers); isnd = 11 has no cloud in its base state,
        # so the departure is not the base cloud's alone.  The integration from the other analytic soundings
        # (isnd = 1, 8, 12-14, 18-23) has not been compared.  All their initial states are CM1's word for word,
        # so they stay open to inspect with run.iconly = 1; integrating them is refused.  isnd = 2 and 3 are
        # refused above already (imoist = 0).
        if initialization is not None and not inspecting_only:
            from .cartesian_door import INTEGRATION_FAILED_ISND, sounding_integration_checked

            if int(initialization.isnd) in (4, 10):
                refuse(
                    code="CM1-CFG-BASE-CLOUD-INTEGRATION",
                    gate="cartesian-runtime",
                    reason=(
                        f"initialization.isnd = {initialization.isnd} builds a base state that carries cloud "
                        "(base.F qc0/qi0, subtracted in solve2.F:404's buoyancy); GM1 builds that initial state "
                        "word for word as CM1 r21.1 does, but its time integration from it does not match CM1's "
                        "(w and theta' differ from the first minute, both models on Morrison at eqtset = 2), so a "
                        "run from it would not be the CM1 run the receipt names"
                    ),
                    remedy="declare run.iconly = 1 to inspect the initial state, or choose isnd = 5 or 9, or "
                           "isnd = 7 with an input_sounding file",
                    option="initialization.isnd",
                    value=int(initialization.isnd),
                )
            elif int(initialization.isnd) not in (2, 3) and not sounding_integration_checked(initialization.isnd):
                compared = int(initialization.isnd) in INTEGRATION_FAILED_ISND
                refuse(
                    code="CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION",
                    gate="cartesian-runtime",
                    reason=(
                        f"initialization.isnd = {initialization.isnd} builds one of base.F's analytic soundings; "
                        "GM1 builds that initial state word for word as CM1 r21.1 does, but "
                        + ("its time integration from it does not match CM1's (w and theta' differ from the first "
                           "minute, both models on Morrison at eqtset = 2, with no cloud in the base state)"
                           if compared else
                           "no time integration from it has been compared with CM1's (of the analytic soundings "
                           "only isnd = 5 and 9 matched CM1 over 15 minutes, and isnd = 4, 10 and 11 did not)")
                        + ", so a run from it would not be the CM1 run the receipt names"
                    ),
                    remedy="declare run.iconly = 1 to inspect the initial state, or choose isnd = 5 or 9, or "
                           "isnd = 7 with an input_sounding file",
                    option="initialization.isnd",
                    value=int(initialization.isnd),
                )
            if int(initialization.iinit) == 4:
                refuse(
                    code="CM1-CFG-MOIST-BENCHMARK-INTEGRATION",
                    gate="cartesian-runtime",
                    reason=(
                        "initialization.iinit = 4 builds Bryan and Fritsch's moist benchmark bubble "
                        "(init3d.F:579-647) word for word as CM1 r21.1 does, but GM1's time integration from it "
                        "does not match CM1's (w and theta' differ from the first minute, both models on Morrison "
                        "at eqtset = 2), so a run from it would not be the CM1 run the receipt names"
                    ),
                    remedy="declare run.iconly = 1 to inspect the initial state, or start the storm with "
                           "iinit = 1 (a warm bubble)",
                    option="initialization.iinit",
                    value=4,
                )
        if (
            initialization is not None
            and initialization.imoist == 0
            and config.physics.component != "off"
            and inspecting_only
        ):
            refuse(
                code="CM1-CFG-CARTESIAN-DRY-COMPONENT",
                gate="cartesian-runtime",
                reason="initialization.imoist = 0 and a moist physics.component name the same CM1 selector two different ways, so the receipt would record a moist column over a dry initial state",
                remedy="declare physics.component = 'off' beside initialization.imoist = 0",
                option="physics.component",
                value=config.physics.component,
            )
        _require_cartesian_physics(config)
        _require_cartesian_envelope(config)
        _validate_nest(config)
        if config.chain is not None:
            from .arwen_chain import validate_chain_document

            validate_chain_document(config)
    elif config.run.lane in {Lane.PRODUCT, Lane.REAL_PRODUCT}:
        _choice(
            "physics.component",
            config.physics.component,
            {"wsm6-mp6"},
            "the first product seam is measured only for the semantic ArWen WSM6 component",
        )
        _choice(
            "physics.equation_set",
            config.physics.equation_set,
            {2},
            "the product target is CM1 eqtset=2 and remains red until its energy oracle passes",
        )
        if config.run.lane is Lane.PRODUCT:
            _require_official_profile(config)
        else:
            _require_real_product_profile(config)
    else:
        _choice(
            "physics.component",
            config.physics.component,
            {"native-cm1-ptype5"},
            "native context preserves CM1's own component identity and never enters production",
        )
        _choice(
            "physics.equation_set",
            config.physics.equation_set,
            {2},
            "the official native context selects eqtset=2",
        )
        _require_official_profile(config)


def _require_cartesian_physics(config: CM1Config) -> None:
    """Admit the component-scoped CM1 physics options this lane executes.

    ``ihail`` is admitted for every component whose scheme reads a dense-ice
    branch: ``src/param.F:3407`` passes it to Morrison, ``src/param.F:3697-3706``
    passes it to WSM6, and ArWen's WDM6 takes the same five constants through
    ``wdm6_hail_opt``.  A component whose scheme diagnoses its own dense ice --
    either Thompson -- has no such branch, and a declared value there is
    refused by name rather than recorded and ignored.  ``ndcnst`` is consumed
    by ``ptype=5`` only (``src/morrison.F:329``), and the admitted ArWen kernel
    writes its conversion as a literal, so declaring it anywhere else, or
    declaring any other value, is refused by name too.
    """

    from .arwen_morrison_component import MORRISON_NDCNST_CM3, require_ndcnst
    from .dense_ice import dense_ice_selection
    from .moisture_species import MORRISON_PROFILE

    physics = config.physics
    # dense_ice refuses a declared ihail for a component with no branch to
    # select, and refuses a value outside CM1's own 0..1 for one that has.
    dense_ice_selection(physics.component, physics.ihail)
    if physics.ndcnst is not None:
        if physics.component != MORRISON_PROFILE.component_id:
            refuse(
                code="CM1-CFG-CARTESIAN-NDCNST-SCOPE",
                gate="cartesian-runtime",
                reason=(
                    f"ndcnst is CM1's constant cloud-droplet concentration for "
                    f"ptype=5 (src/morrison.F:329); component "
                    f"{physics.component!r} never reads it, so the declared "
                    "value would be sealed into this run's configuration and "
                    "receipts while changing nothing the scheme computed"
                ),
                remedy=(
                    "remove physics.ndcnst, or select physics.component = "
                    f"{MORRISON_PROFILE.component_id!r}"
                ),
                option="physics.ndcnst",
                value=physics.ndcnst,
            )
        require_ndcnst(physics.ndcnst)
    elif physics.component == MORRISON_PROFILE.component_id:
        # Undeclared binds CM1's own namelist value, which is also the only
        # one the admitted kernel executes; naming it here keeps the refusal
        # and the default one number.
        require_ndcnst(MORRISON_NDCNST_CM3)
def nest_follow_max_cells(nest: NestConfig, *, parent_dx_m: float) -> int:
    """Whole parent cells one re-centring may move, from the declared speed.

    Derived rather than declared, so a deck keeps its physical following speed
    when its parent spacing changes.
    """

    return int(math.floor(nest.follow_max_speed_m_s * nest.follow_interval_s / parent_dx_m))


def _validate_nest(config: CM1Config) -> None:
    """Cross-check the declared nest against the parent grid and clock.

    Everything checked here needs the whole document: the parent's spacing,
    its stretching selectors, its pressure solver and its large step.  CM1's
    own rules decide every clock question, by line:
    ``src/misclibs.F:857-868`` (the psolver=3 acoustic small step),
    ``src/misclibs.F:883-885`` (the substep count), ``src/misclibs.F:835``
    with ``src/misclibs.F:887-911`` (max_nsound and the stop above it), and
    ``src/misclibs.F:2714`` (the advective Courant stop the refinement moves
    the child toward).
    """

    nest = config.nest
    if nest is None:
        return
    from .nest_geometry import (
        PARENT_STENCIL_HALO as STENCIL_RADIUS_XY,
        NestGeometry,
        child_acoustic_defect,
    )

    # Two conditions this nest depends on are not checked here, and must not
    # be: the Cartesian lane has already refused both before this function is
    # reached, and a second refusal standing beside a live one is dead code
    # wearing a guard's clothes.
    #
    #   * ``dynamics.psolver`` -- ``_require_cartesian_envelope`` admits only
    #     3.  That matters to the child rather than being incidental: the
    #     child's acoustic cadence is derived from misclibs.F:857-868's
    #     psolver=3 branch, where ``dtsm`` is 0.60*min(dx,dy)/360 and depends
    #     on the horizontal spacings alone, so refining the horizontal is
    #     exactly what shortens it.  misclibs.F:846-855's psolver=2 branch
    #     takes 0.50*min(dx,dy,dz)/360 instead, and on a grid whose ``dz`` is
    #     the smallest spacing -- every tornado-scale nest -- refining the
    #     horizontal would not shorten ``dtsm`` at all.
    #   * ``grid.stretch_x``/``grid.stretch_y`` -- ``_validate_first_slice``
    #     admits only false.  That matters to the child too: its spacing is
    #     the parent's divided by the ratio and its boundary sits on whole
    #     parent cells, which is what puts the child's outer u and v columns
    #     on the parent's own faces.  With no single parent spacing the
    #     derived child cells would drift out of registration across the nest
    #     and the flux entering the child would not be the flux that left the
    #     parent.
    #
    # Both facts are recorded in records/CM1-NEST-SEAM.md; neither is
    # re-refused.
    try:
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
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CFG-NEST-GEOMETRY",
            gate="support-matrix",
            reason=f"the nest geometry refuses this placement: {error}",
            remedy="correct the named nest key; the geometry contract is not widened from the door",
            option="nest",
            value={
                "refinement_ratio": nest.refinement_ratio,
                "parent_span_x": nest.parent_span_x,
                "parent_span_y": nest.parent_span_y,
                "spec_zone": nest.spec_zone,
                "relax_zone": nest.relax_zone,
            },
        )
    import struct as _struct

    for axis, spacing in (("x", geometry.dx_m), ("y", geometry.dy_m)):
        word = _struct.unpack("<f", _struct.pack("<f", spacing))[0]
        if word != spacing:
            refuse(
                code="CM1-CFG-NEST-CHILD-SPACING",
                gate="support-matrix",
                reason=(
                    f"the child's {axis} spacing is the parent's divided by "
                    f"nest.refinement_ratio={nest.refinement_ratio}, "
                    f"{spacing!r} m, which has no exact default-REAL word "
                    f"(it would be stored as {word!r} m).  CM1 carries dx and "
                    "dy as one binary32 word each (param.F), so the child's "
                    "faces would drift off the parent's across the window -- "
                    f"by {abs(word - spacing) * (geometry.nx if axis == 'x' else geometry.ny)!r} m "
                    "at its far edge -- and the flux the child's boundary "
                    "carries would stop being the flux through the parent's "
                    "face; the initializer refuses the same word"
                ),
                remedy=(
                    "choose a parent spacing and ratio whose quotient is exact "
                    "in binary32 (e.g. 600 m / 3 = 200 m, 100 m / 5 = 20 m, "
                    "250 m / 5 = 50 m)"
                ),
                option="nest.refinement_ratio",
                value=[config.grid.dx_m, nest.refinement_ratio],
            )
    child_levels = nest_child_vertical_coordinate(config)
    if child_levels is not None:
        from .nest_vertical import require_vertical_refinement

        # A different model top, or a child layer thicker than every parent
        # layer it overlaps, is refused here by what it would do to the column.
        require_vertical_refinement(
            require_vertical_coordinate(config),
            child_levels,
            code="CM1-CFG-NEST-VERTICAL-REFINEMENT",
            gate="support-matrix",
            option="nest.child_stretch_z",
        )
    placement = geometry.placement_defect()
    if placement is not None:
        refuse(
            code="CM1-CFG-NEST-PLACEMENT",
            gate="support-matrix",
            reason=placement,
            remedy=(
                "move the nest inward with nest.parent_i0/nest.parent_j0, or "
                "shrink nest.parent_span_x/nest.parent_span_y"
            ),
            option="nest.parent_i0",
            value=[
                nest.parent_i0,
                nest.parent_j0,
                nest.parent_span_x,
                nest.parent_span_y,
            ],
        )
    if min(geometry.nx, geometry.ny) < 2 * STENCIL_RADIUS_XY + 1:
        refuse(
            code="CM1-CFG-NEST-EXTENT",
            gate="support-matrix",
            reason=(
                f"the child is {geometry.nx} by {geometry.ny} cells, and the "
                f"bound fifth-order horizontal stencil reads "
                f"{STENCIL_RADIUS_XY} cells either side; a shorter row would "
                "advect from ghost words that no boundary fills"
            ),
            remedy=(
                "raise nest.parent_span_x/nest.parent_span_y or "
                "nest.refinement_ratio so the child has at least "
                f"{2 * STENCIL_RADIUS_XY + 1} cells on each axis"
            ),
            option="nest.parent_span_x",
            value=[geometry.nx, geometry.ny],
        )
    clock = geometry.clock_defect(config.run.dt_s)
    if clock is not None:
        refuse(
            code="CM1-CFG-NEST-CLOCK",
            gate="fixed-clock",
            reason=clock,
            remedy=(
                "raise nest.time_refinement_ratio, or lower run.dt_s, so the "
                "child's step shrinks with its spacing"
            ),
            option="nest.time_refinement_ratio",
            value=nest.time_refinement_ratio,
        )
    defect = child_acoustic_defect(
        geometry,
        parent_dt_s=config.run.dt_s,
        acoustic_substeps=nest.acoustic_substeps,
    )
    if defect is not None:
        refuse(
            code="CM1-CFG-NEST-ACOUSTIC-SUBSTEPS",
            gate="support-matrix",
            reason=(
                f"the child's {geometry.dx_m!r} m spacing at "
                f"{geometry.child_dt_s(config.run.dt_s)!r} s is inadmissible: "
                + defect
            ),
            remedy=(
                "declare nest.acoustic_substeps of at least "
                f"{geometry.cm1_child_acoustic_substeps(config.run.dt_s)}, "
                "which is what CM1's own dtsmall derives for this child"
            ),
            option="nest.acoustic_substeps",
            value=nest.acoustic_substeps,
        )
    if nest.follow_storm:
        child_dt = geometry.child_dt_s(config.run.dt_s)
        cadence = nest.follow_interval_s / config.run.dt_s
        if not math.isfinite(cadence) or cadence < 1.0 or not cadence.is_integer():
            refuse(
                code="CM1-CFG-NEST-FOLLOW-CADENCE",
                gate="fixed-clock",
                reason=(
                    "nest.follow_interval_s is not an integer number of parent "
                    "large steps, so a re-centring would fall between two "
                    "parent states and the nest would be handed lateral "
                    "forcing built from a parent time the run never reached; "
                    "CM1's own adaptive move fires on its own cadence at a "
                    "completed step (src/cm1.F:3641-3679)"
                ),
                remedy=(
                    f"set nest.follow_interval_s to an integer multiple of "
                    f"run.dt_s = {config.run.dt_s!r}"
                ),
                option="nest.follow_interval_s",
                value=nest.follow_interval_s,
            )
        if nest.follow_interval_s > config.run.duration_s:
            refuse(
                code="CM1-CFG-NEST-FOLLOW-CADENCE",
                gate="fixed-clock",
                reason=(
                    "nest.follow_interval_s is longer than the run, so the "
                    "tracker would never fire and the nest would never move "
                    "while the receipt named it storm-following"
                ),
                remedy=(
                    "set nest.follow_interval_s to at most "
                    f"run.duration_s = {config.run.duration_s!r}"
                ),
                option="nest.follow_interval_s",
                value=nest.follow_interval_s,
            )
        # A nest that may jump further than its own relaxation zone in one
        # event exposes freshly uncovered child cells that no relaxation has
        # ever touched.
        cells = nest_follow_max_cells(nest, parent_dx_m=config.grid.dx_m)
        if cells < 1:
            refuse(
                code="CM1-CFG-NEST-FOLLOW-SPEED",
                gate="support-matrix",
                reason=(
                    f"nest.follow_max_speed_m_s={nest.follow_max_speed_m_s!r} over "
                    f"nest.follow_interval_s={nest.follow_interval_s!r} is less "
                    f"than one {config.grid.dx_m!r} m parent cell per re-centring, "
                    "and the nest moves by whole parent cells only; it could "
                    "never move while the receipt named it storm-following"
                ),
                remedy=(
                    "raise nest.follow_max_speed_m_s or nest.follow_interval_s so "
                    "their product is at least one parent cell"
                ),
                option="nest.follow_max_speed_m_s",
                value=nest.follow_max_speed_m_s,
            )
        reach = cells * nest.refinement_ratio
        if reach > geometry.nx or reach > geometry.ny:
            refuse(
                code="CM1-CFG-NEST-FOLLOW-STEP",
                gate="support-matrix",
                reason=(
                    f"nest.follow_max_speed_m_s={nest.follow_max_speed_m_s!r} is "
                    f"{cells} parent cells per re-centring, {reach} child cells, "
                    "which is at least the "
                    f"child's own {geometry.nx} by {geometry.ny} extent; one "
                    "re-centring would move the nest entirely off its previous "
                    "footprint, so every cell of the moved child would be "
                    "filled by interpolation from the parent and the refined "
                    "detail the nest exists to produce would be discarded at "
                    "each move"
                ),
                remedy=(
                    "lower nest.follow_max_speed_m_s, or raise "
                    "nest.parent_span_x/nest.parent_span_y"
                ),
                option="nest.follow_max_speed_m_s",
                value=nest.follow_max_speed_m_s,
            )
        from .nest_move import corral_cells, surface_tracker_defect

        if nest.follow_tracker == NEST_SURFACE_TRACKER:
            spacing = min(geometry.dx_m, geometry.dy_m)
            h = int(math.floor(nest.follow_smoothing_radius_m / spacing))
            tracker_defect = surface_tracker_defect(
                nx=geometry.nx, ny=geometry.ny,
                width=nest.spec_zone + nest.relax_zone, h=h,
            )
            if tracker_defect is not None:
                refuse(
                    code="CM1-CFG-NEST-FOLLOW-SMOOTHING",
                    gate="support-matrix",
                    reason=(
                        f"nest.follow_smoothing_radius_m={nest.follow_smoothing_radius_m!r} "
                        f"on the child's {spacing!r} m spacing is a box half-width of "
                        f"{h} child cells: " + tracker_defect
                    ),
                    remedy=(
                        "choose a smoothing radius of at least one child cell whose "
                        "box fits inside the child's interior"
                    ),
                    option="nest.follow_smoothing_radius_m",
                    value=[nest.follow_smoothing_radius_m, geometry.nx, geometry.ny],
                )
            if nest.follow_relocation_radius_m is not None and nest.follow_hold_limit is not None:
                # Under hold the search reach grows with the held samples
                # (cm1core.nest_move, THE HOLD POLICY): the sample that
                # relocates spans follow_hold_limit + 1 follow intervals.
                reach_m = nest.follow_max_speed_m_s * nest.follow_interval_s
                intervals = int(nest.follow_hold_limit) + 1
                needed_m = intervals * reach_m + spacing
                if nest.follow_relocation_radius_m < needed_m:
                    refuse(
                        code="CM1-CFG-NEST-FOLLOW-RELOCATION",
                        gate="support-matrix",
                        reason=(
                            f"nest.follow_relocation_radius_m="
                            f"{nest.follow_relocation_radius_m!r} does not reach one "
                            f"{spacing!r} m child cell beyond the search reach of the "
                            f"sample that relocates: after follow_hold_limit = "
                            f"{nest.follow_hold_limit} held samples that window spans "
                            f"{intervals} follow intervals, {intervals} * "
                            f"follow_max_speed_m_s * follow_interval_s = "
                            f"{intervals * reach_m!r} m, so a relocation would search no "
                            "further than the window whose sample was just held and "
                            "would find the same edge, or nothing"
                        ),
                        remedy=(
                            "declare nest.follow_relocation_radius_m of at least "
                            f"{needed_m!r} m (twice that, {2.0 * needed_m!r} m, is "
                            "recommended: the held window widens at the declared speed "
                            "only, so a vortex a few per cent over "
                            "follow_max_speed_m_s is found by the radius or not at all; "
                            "records/CM1-NEST-SEAM.md 5.8), or lower "
                            "nest.follow_hold_limit"
                        ),
                        option="nest.follow_relocation_radius_m",
                        value=[nest.follow_relocation_radius_m, nest.follow_hold_limit,
                               reach_m, spacing],
                    )
        for axis, spacing, start, span, count in (
            ("x", config.grid.dx_m, nest.parent_i0, nest.parent_span_x, config.grid.nx),
            ("y", config.grid.dy_m, nest.parent_j0, nest.parent_span_y, config.grid.ny),
        ):
            corral = corral_cells(nest.follow_corral_m, spacing)
            if corral < STENCIL_RADIUS_XY:
                refuse(
                    code="CM1-CFG-NEST-FOLLOW-CORRAL",
                    gate="support-matrix",
                    reason=(
                        f"nest.follow_corral_m={nest.follow_corral_m!r} is {corral} "
                        f"parent cells on the {axis} axis, fewer than the "
                        f"{STENCIL_RADIUS_XY} the parent's fifth-order stencil reads; "
                        "a moved child would build its lateral forcing from parent "
                        "halo words the open radiative boundary writes "
                        "(src/bc.F:528-934) instead of from parent air"
                    ),
                    remedy=(
                        f"declare nest.follow_corral_m of at least "
                        f"{STENCIL_RADIUS_XY * spacing!r} m, or leave it out"
                    ),
                    option="nest.follow_corral_m",
                    value=nest.follow_corral_m,
                )
            clear = min(start, count - start - span)
            if clear < corral:
                refuse(
                    code="CM1-CFG-NEST-FOLLOW-CORRAL",
                    gate="support-matrix",
                    reason=(
                        f"the nest is placed {clear} parent cells from the parent's "
                        f"{axis} boundary, inside its own {corral}-cell corral; every "
                        f"move on the {axis} axis would end inside the corral too "
                        "(share/mediation_nest_move.F:833-859), so the first "
                        "re-centring along it would stop the run"
                    ),
                    remedy=(
                        "place the nest further from the parent boundary, or lower "
                        "nest.follow_corral_m"
                    ),
                    option="nest.follow_corral_m",
                    value=[clear, corral],
                )
        del child_dt
    if nest.tornado_criterion is not None:
        _validate_nest_tornado_criterion(
            config, geometry,
            child_levels if child_levels is not None else require_vertical_coordinate(config),
        )


def _validate_nest_tornado_criterion(config: CM1Config, geometry: Any, levels: Any) -> None:
    """Cross-check ``[nest.tornado_criterion]`` against the child it samples.

    The criterion is fixed (:data:`cm1core.nest_tornado_criterion.CRITERION`),
    so what can be wrong is the child: a deck on which 10 m is below the
    lowest wind level, a child too narrow to hold a centre 1000 m from every
    edge, a coarse child whose 300 m disk would reach the specified and
    relaxation rows, or a stop rule that could never act.
    """

    from .nest_tornado_criterion import CRITERION, height_bracket, interior_corner_box

    nest = config.nest
    criterion = nest.tornado_criterion
    prefix = "nest.tornado_criterion"
    height = CRITERION["height_agl_m"]
    zh1 = float(levels.zh[1])
    if height_bracket(levels, height) is None:
        refuse(
            code="CM1-CFG-NEST-TORNADO-HEIGHT",
            gate="support-matrix",
            reason=(
                f"the criterion samples the wind at {height!r} m AGL, below the child's "
                f"lowest scalar level zh(1) = {zh1!r} m; that wind would be extrapolated "
                "from the levels above rather than interpolated between two of them, a "
                "number no model level carries"
            ),
            remedy=(
                f"give the child a lowest layer no thicker than {2.0 * height!r} m "
                "(nest.child_dz_bot_m or the parent deck), or drop [nest.tornado_criterion]"
            ),
            option=prefix,
            value=[height, zh1],
        )
    margin, radius = CRITERION["edge_margin_m"], CRITERION["disk_radius_m"]
    for axis, n, d in (("x", geometry.nx, geometry.dx_m), ("y", geometry.ny, geometry.dy_m)):
        lo, hi = interior_corner_box(n, d, margin)
        if lo > hi:
            refuse(
                code="CM1-CFG-NEST-TORNADO-EXTENT",
                gate="support-matrix",
                reason=(
                    f"the child is {n * d!r} m across in {axis}; a vortex centre must lie "
                    f"{margin!r} m from every child edge, so no corner could ever be a centre "
                    "and every sample would find nothing"
                ),
                remedy=f"widen the child to more than {2.0 * margin!r} m in {axis}",
                option=prefix,
                value=[axis, n, d, margin],
            )
    rows = nest.spec_zone + nest.relax_zone
    forced_m = rows * max(geometry.dx_m, geometry.dy_m)
    clearance_m = margin - radius - max(geometry.dx_m, geometry.dy_m)
    if clearance_m <= forced_m:
        refuse(
            code="CM1-CFG-NEST-TORNADO-EDGE-MARGIN",
            gate="support-matrix",
            reason=(
                f"a centre {margin!r} m from the child edge with its {radius!r} m disk and "
                f"one cell of vorticity stencil reads winds {clearance_m!r} m from the edge, "
                f"inside the child's {rows} specified and relaxation rows ({forced_m!r} m); "
                "the ground-relative wind that decides the verdict could be a parent wind "
                "interpolated onto the child"
            ),
            remedy="refine the child or narrow nest.spec_zone + nest.relax_zone",
            option=prefix,
            value=[margin, radius, rows, forced_m],
        )
    # Both stop rules are model times or intervals of this run's absolute
    # clock (run.duration_s is CM1's timax), so only the cutoff can be checked
    # against the run before it starts.  A cutoff AT the duration is refused
    # with one after it: its request is raised by the last sample, which is
    # the final scheduled event, where the run ends having integrated the
    # whole duration -- the rule could end nothing.
    cutoff = criterion.stop_if_no_tornado_by_s
    if cutoff is not None:
        if cutoff >= config.run.duration_s:
            refuse(
                code="CM1-CFG-NEST-TORNADO-STOP",
                gate="fixed-clock",
                reason=(
                    f"{prefix}.stop_if_no_tornado_by_s={cutoff!r} is not before the end of the "
                    f"run (run.duration_s={config.run.duration_s!r}); its request would be "
                    "raised no earlier than the final scheduled event, where the run ends "
                    "anyway, so the stop rule could never act while the receipt named it"
                ),
                remedy=(
                    f"declare {prefix}.stop_if_no_tornado_by_s before run.duration_s (at least "
                    "one history interval before it, for the stop to save any stepping), "
                    "or drop it"
                ),
                option=f"{prefix}.stop_if_no_tornado_by_s",
                value=[cutoff, config.run.duration_s],
            )


def _acoustic_small_step_remedy(config: CM1Config, acoustic_substeps: int) -> str:
    """Name only the routes this same door will actually admit.

    Two of them are traps unless they are computed rather than asserted.

    ``misclibs.F:887-911`` with ``adapt_dt=0`` has no substep route above
    ``max_nsound``: CM1 prints "Time step (dtl) needs to be smaller" and calls
    ``stopcm1``.  Naming a count above 192 here would send the operator to a
    value ``_acoustic_substeps`` refuses by a different code, so above that
    ceiling the count is named as unavailable and only the step is offered.

    And the largest admissible step has to satisfy the inequality this door
    checks -- ``dt/nsound > dtsm`` -- not merely the product ``nsound*dtsm``,
    and it has to fall on the declared event clock, or the operator sets the
    printed value and is refused a second time.  ``cm1_max_large_step_s`` gives
    the first and ``cm1_largest_clock_aligned_large_step_s`` the second, so the
    step this message prints is one the very next parse admits.
    """

    grid = config.grid
    needed = cm1_diagnostic_acoustic_substeps(
        dt_s=config.run.dt_s, dx_m=grid.dx_m, dy_m=grid.dy_m
    )
    maximum = cm1_max_large_step_s(
        dx_m=grid.dx_m, dy_m=grid.dy_m, acoustic_substeps=acoustic_substeps
    )
    aligned = cm1_largest_clock_aligned_large_step_s(
        dx_m=grid.dx_m,
        dy_m=grid.dy_m,
        acoustic_substeps=acoustic_substeps,
        event_seconds=(
            config.run.duration_s,
            config.run.history_interval_s,
            abs(config.run.restart_interval_s),
        ),
    )
    if aligned is None:
        step_clause = (
            f"lower run.dt_s to at most {maximum!r} seconds, and to a value "
            "that still divides run.duration_s, run.history_interval_s and "
            "run.restart_interval_s exactly"
        )
    else:
        step_clause = (
            f"set run.dt_s to {aligned!r} seconds, the largest step this grid "
            "and this substep count admit that also places run.duration_s, "
            "run.history_interval_s and run.restart_interval_s on exact "
            "large-step boundaries"
        )
    if needed > CM1_MAX_ACOUSTIC_SUBSTEPS:
        return (
            f"{step_clause}. Raising runtime.acoustic_substeps is not "
            f"available here: this step needs {needed} of them, above CM1's "
            f"own max_nsound={CM1_MAX_ACOUSTIC_SUBSTEPS}, where "
            "misclibs.F:887-911 stops the run instead -- a smaller large step "
            "is CM1's only answer above that ceiling"
        )
    return (
        f"raise runtime.acoustic_substeps to {needed}, the count CM1's own "
        f"dtsmall derives for this grid and step, or {step_clause}"
    )


def _require_cartesian_envelope(config: CM1Config) -> None:
    """Admit the configurable Cartesian envelope by value, not by one point.

    This lane is a range, not a sealed profile: the grid is a user choice.
    Every option outside the range the bound large-step schedule actually
    executes is refused by name here, before any device memory is priced.
    """

    # The large step is a free parameter in this lane, exactly as ``dtl`` is
    # in CM1's namelist: ``solve2.F:145`` declares ``dt`` a default-REAL
    # variable and every GM1 stage, the final-RK eqtset2 held-rate removal
    # included, now takes it as an argument.  What is not free is the acoustic
    # small step it implies, which CM1 derives in ``misclibs.F:818-915`` and
    # never accepts from a namelist.
    runtime = config.runtime
    if runtime is None:
        # _LANE_SCHEMA makes [runtime] mandatory in this lane, so reaching
        # here would mean the parser admitted a document without the table
        # the acoustic cadence is derived from.
        raise AssertionError("the Cartesian lane always declares [runtime]")
    if positive_normal_f32_bits(config.run.dt_s) is None:
        refuse(
            code="CM1-CFG-CARTESIAN-TIMESTEP",
            gate="cartesian-runtime",
            reason=(
                "solve2.F:145 carries the large step as a default-REAL "
                "word and solve2.F:1569-1576 multiplies the four held "
                "diabatic rates by it; a step that is not a positive "
                "normal binary32 number flushes the final-RK held-rate "
                "removal to zero inside the kernel while every other "
                "stage advances by the declared step, and the "
                "(state-snapshot)/dt inverse after microphysics then "
                "divides by that same unusable word"
            ),
            remedy=(
                "declare run.dt_s as a finite step of at least "
                f"{CM1_MIN_NORMAL_DT_S!r} seconds"
            ),
            option="run.dt_s",
            value=config.run.dt_s,
        )
    if config.dynamics.idiff != 0:
        # Not taste: adv.F:962,1177,1622,1899,2115,2541,2832,3109,3433 and
        # turb.F:6804-6927 are every place CM1 applies this term, and the
        # Cartesian lane's advection images (cuda_advection5,
        # cuda_momentum_advection5) contain no difforder branch at all.  A
        # document that declared idiff=1 would be admitted, publish a receipt
        # naming a diffusion coefficient, and integrate exactly none of it.
        # All 23 of CM1's own shipped decks set idiff=0, so this refusal
        # cannot fire on a configuration CM1 itself distributes.
        refuse(
            code="CM1-CFG-CARTESIAN-EXPLICIT-DIFFUSION",
            gate="cartesian-runtime",
            reason=(
                f"dynamics.idiff={config.dynamics.idiff} selects CM1's "
                "explicit artificial diffusion (adv.F:962 for difforder=6, "
                "cm1.F:3791 for difforder=2), and this lane's advection "
                "kernels carry no difforder branch; the declared "
                f"difforder={config.dynamics.difforder} with "
                f"kdiff2={config.dynamics.kdiff2}/"
                f"kdiff6={config.dynamics.kdiff6} would appear in the run "
                "receipt while the integration applied no diffusion at all"
            ),
            remedy=(
                "set dynamics.idiff = 0, which is what all 23 of CM1 r21.1's "
                "shipped configuration decks set; the sub-grid closure "
                "selected by closures.sgs_model is this lane's diffusion"
            ),
            option="dynamics.idiff",
            value=config.dynamics.idiff,
        )
    defect = acoustic_small_step_defect(
        dt_s=config.run.dt_s,
        dx_m=config.grid.dx_m,
        dy_m=config.grid.dy_m,
        acoustic_substeps=runtime.acoustic_substeps,
    )
    if defect is not None:
        refuse(
            code="CM1-CFG-CARTESIAN-ACOUSTIC-SMALL-STEP",
            gate="cartesian-runtime",
            reason=defect,
            remedy=_acoustic_small_step_remedy(config, runtime.acoustic_substeps),
            option="run.dt_s",
            value=config.run.dt_s,
        )
    # param.F:836-852 resets iwnd to 0 whenever isnd=7, without stopping.
    initialization = config.initialization
    if initialization is not None and initialization.isnd == 7 and initialization.iwnd != 0:
        refuse(
            code="CM1-CFG-CARTESIAN-BASE-WIND",
            gate="cartesian-runtime",
            reason="CM1 param.F:836-852 forces iwnd=0 whenever isnd=7, so the declared analytic base-state wind would be discarded and the run would carry the input_sounding winds while this sealed configuration recorded a different base state",
            remedy="declare initialization.iwnd = 0 with isnd = 7, select isnd = 17 to pair that same file's thermodynamics with this analytic hodograph (param.F:849-851), or select isnd = 5 for base.F's analytic sounding",
            option="initialization.iwnd",
            value=initialization.iwnd,
        )
    # Only the warm bubble's centre, and only where the document wrote it,
    # is priced.  Every other arm's placement is CM1's own and CM1 builds it
    # whatever the domain: init3d.F:548-550 puts iinit=3's bubbles at y = 3,
    # 33 and 63 km, :662-663 the Straka blob at x = 0 and :1356 the
    # Skamarock-Klemp wave at x = 100 km, each cut off by the grid without a
    # word.  Refusing them would refuse CM1 decks CM1 runs.
    if (
        initialization is not None
        and initialization.iinit == 1
        and initialization.bubble_center_declared
        and _BUBBLE_CENTER_KEYS & initialization.declared
    ):
        _require_bubble_inside_domain(config, initialization)
    for name, value, allowed, reason in (
        (
            "dynamics.psolver",
            config.dynamics.psolver,
            {3},
            "the Cartesian dynamics composes the vertically implicit acoustic path only",
        ),
        (
            "dynamics.scalar_weno",
            config.dynamics.scalar_weno,
            {2},
            "the Cartesian scalar stage is bound to the final-RK-stage WENO cadence",
        ),
        (
            "dynamics.dry_mass_correction",
            config.dynamics.dry_mass_correction,
            {True},
            "the Cartesian large step always closes with the dry-air mass correction",
        ),
    ):
        _choice(name, value, set(allowed), reason)
    # cm1.F:3186 fires a restart whenever the model clock reaches rsttim and
    # cm1.F:4247-4252 advances rsttim by rstfrq, so an intermediate cadence is
    # an ordinary CM1 configuration.  It is now an ordinary one here too: the
    # refusal that stood in this place said "this lane writes no checkpoint",
    # and the lane writes one.  What remains is the arithmetic that keeps the
    # last restart on the run's own final state.
    restarts = config.run.restart_interval_s
    if config.run.duration_s % restarts:
        refuse(
            code="CM1-CFG-CARTESIAN-RESTART-CADENCE",
            gate="cartesian-runtime",
            reason=(
                f"run.duration_s={config.run.duration_s} is not a whole number of "
                f"{restarts} s restart intervals, so the last restart this run "
                "wrote would be up to one interval older than the run itself and a "
                "resume would silently redo model time the run had already computed"
            ),
            remedy="set run.restart_interval_s to a value that divides run.duration_s exactly",
            option="run.restart_interval_s",
            value=restarts,
        )
    # The history cadence is a real integration cadence in this lane: frames
    # land at every multiple of run.history_interval_s up to run.duration_s.
    # An interval that does not divide the duration would stop the frames
    # short of the run's own completed state, so it is refused by value here,
    # with the same arithmetic the runtime uses.
    from .gm1_runtime import history_cadence

    try:
        history_cadence(
            adaptive=config.dynamics.adaptive_timestep,
            dt_s=config.run.dt_s,
            duration_s=config.run.duration_s,
            interval_s=config.run.history_interval_s,
        )
    except (TypeError, ValueError) as error:
        refuse(
            code="CM1-CFG-CARTESIAN-CADENCE",
            gate="cartesian-runtime",
            reason=f"the history cadence is not an exact large-step schedule: {error}",
            remedy="set run.history_interval_s to a whole number of run.dt_s steps that divides run.duration_s exactly",
            option="run.history_interval_s",
            value=config.run.history_interval_s,
        )
    # CM1 places no rule between rstfrq and tapfrq: cm1.F:3177 and cm1.F:3186
    # compare the model clock against taptim and rsttim independently, and
    # cm1.F:3992 and cm1.F:4250 advance each by its own interval.
    # A restart between two frames is an ordinary CM1 configuration, and the
    # resume schedule (gm1_runtime.run_schedule) keeps the frames on the
    # original absolute grid under restart_reset_frqtim=false and refuses by
    # name when .true. would move them off it, so no refusal is needed here.


def _require_bubble_inside_domain(
    config: CM1Config, initialization: InitializationConfig
) -> None:
    """``init3d.F:456-476``'s bubble must fit in the domain it perturbs.

    ``init3d.F:468-482`` evaluates ``beta`` only at the grid's own ``xh``/
    ``yh``, so a bubble whose ``bhrad`` disc reaches past a lateral edge is
    not wrapped or reflected -- it is simply cut off.  Two things then go
    wrong at once and neither announces itself: the initial perturbation
    carries less integrated buoyancy than the ``1.0 K`` / ``10 km`` /
    ``1.4 km`` deck the receipt records, and its truncated flank sits on the
    radiative lateral boundary (``boundaries.west``/``east``/``south``/
    ``north`` = 2) at t=0, where ``bc.F``'s radiation condition is applied to
    a buoyancy gradient that has no exterior to radiate into.
    """

    from .run_geometry import BUBBLE_HORIZONTAL_RADIUS_M

    # param.F:7212-7213's frame.  iorigin=2 runs the coordinate from
    # -nx*dx/2 to +nx*dx/2, and every absolute-coordinate initialization
    # word -- including init3d.F:662-663's ric=rjc=0.0 for the Straka blob --
    # is read in it.  Pricing the disc against 0..nx*dx regardless would
    # refuse CM1's own shipped centre for that arm.
    width = config.grid.nx * config.grid.dx_m
    depth = config.grid.ny * config.grid.dy_m
    iorigin = initialization.iorigin
    if iorigin == 2:
        x_lo, x_hi = -0.5 * width, 0.5 * width
        y_lo, y_hi = -0.5 * depth, 0.5 * depth
    else:
        x_lo, x_hi = 0.0, width
        y_lo, y_hi = 0.0, depth
    # init3d.F:459's own bhrad, now that the document can declare it; absent,
    # it is the 10 km deck this constant names, so an undeclared radius is
    # priced exactly as it was before the key existed.
    radius = (
        BUBBLE_HORIZONTAL_RADIUS_M
        if initialization.bhrad_m is None
        else float(initialization.bhrad_m)
    )
    for name, value, lo, hi in (
        ("initialization.bubble_center_x_m", initialization.bubble_center_x_m, x_lo, x_hi),
        ("initialization.bubble_center_y_m", initialization.bubble_center_y_m, y_lo, y_hi),
    ):
        # Only iinit=1 reads both axes; the branches that are uniform in y
        # declare x alone, so an undeclared axis is CM1's own domain centre
        # and is inside the domain by construction.
        if value is None:
            continue
        if value < lo + radius or value > hi - radius:
            refuse(
                code="CM1-CFG-BUBBLE-CENTER-OUTSIDE",
                gate="cartesian-runtime",
                reason=(
                    f"init3d.F:459 gives the warm bubble a {radius} m horizontal "
                    f"radius, and this centre puts its disc outside the "
                    f"{lo}..{hi} m the grid covers on that axis; init3d.F:468-482 "
                    "would build a truncated perturbation with less buoyancy than "
                    "the deck the receipt names, sitting on the radiative lateral "
                    "boundary at the first step"
                ),
                remedy=(
                    f"declare {name} between {lo + radius} and {hi - radius}, or "
                    "widen the grid on that axis"
                ),
                option=name,
                value=value,
            )

def _require_official_profile(config: CM1Config) -> None:
    expected: dict[str, object] = {
        "run.duration_s": 7200.0,
        "run.dt_s": 6.0,
        "run.history_interval_s": 900.0,
        "run.restart_interval_s": -3600.0 if config.run.lane is Lane.NATIVE_CONTEXT else 3600.0,
        "grid.nx": 120,
        "grid.ny": 120,
        "grid.nz": 40,
        "grid.dx_m": 1000.0,
        "grid.dy_m": 1000.0,
        "grid.dz_m": 500.0,
        "dynamics.psolver": 3,
        "dynamics.scalar_weno": 2,
        "dynamics.dry_mass_correction": True,
        "closures.upper_damping_depth_m": 5000.0,
        "closures.upper_damping_rate_s_inv": 0.0033333333333,
        "closures.moving_frame": True,
        "closures.u_move_m_s": 12.5,
        "closures.v_move_m_s": 3.0,
    }
    actual: dict[str, object] = {
        "run.duration_s": config.run.duration_s,
        "run.dt_s": config.run.dt_s,
        "run.history_interval_s": config.run.history_interval_s,
        "run.restart_interval_s": config.run.restart_interval_s,
        "grid.nx": config.grid.nx,
        "grid.ny": config.grid.ny,
        "grid.nz": config.grid.nz,
        "grid.dx_m": config.grid.dx_m,
        "grid.dy_m": config.grid.dy_m,
        "grid.dz_m": config.grid.dz_m,
        "dynamics.psolver": config.dynamics.psolver,
        "dynamics.scalar_weno": config.dynamics.scalar_weno,
        "dynamics.dry_mass_correction": config.dynamics.dry_mass_correction,
        "closures.upper_damping_depth_m": config.closures.upper_damping_depth_m,
        "closures.upper_damping_rate_s_inv": config.closures.upper_damping_rate_s_inv,
        "closures.moving_frame": config.closures.moving_frame,
        "closures.u_move_m_s": config.closures.u_move_m_s,
        "closures.v_move_m_s": config.closures.v_move_m_s,
    }
    for name, wanted in expected.items():
        if actual[name] != wanted:
            refuse(
                code="CM1-CFG-PROFILE-DRIFT",
                gate="fixed-default",
                reason="product/native-context runs are sealed to the one official first profile with an oracle campaign",
                remedy=f"set {name} to {wanted!r}; derived experiments require a separately named profile",
                option=name,
                value=actual[name],
            )


def _require_real_product_profile(config: CM1Config) -> None:
    expected: dict[str, object] = {
        "run.duration_s": 3600.0,
        "run.dt_s": 0.5,
        "run.history_interval_s": 300.0,
        "run.restart_interval_s": 900.0,
        "grid.nx": 300,
        "grid.ny": 300,
        "grid.nz": 75,
        "grid.dx_m": 100.0,
        "grid.dy_m": 100.0,
        "grid.dz_m": 200.0,
        "grid.terrain": True,
        "dynamics.psolver": 3,
        "dynamics.scalar_weno": 2,
        "dynamics.dry_mass_correction": True,
        "closures.upper_damping_depth_m": 3000.0,
        "closures.upper_damping_rate_s_inv": 0.0033333333333,
        "closures.moving_frame": False,
        "closures.u_move_m_s": 0.0,
        "closures.v_move_m_s": 0.0,
    }
    actual: dict[str, object] = {
        "run.duration_s": config.run.duration_s,
        "run.dt_s": config.run.dt_s,
        "run.history_interval_s": config.run.history_interval_s,
        "run.restart_interval_s": config.run.restart_interval_s,
        "grid.nx": config.grid.nx,
        "grid.ny": config.grid.ny,
        "grid.nz": config.grid.nz,
        "grid.dx_m": config.grid.dx_m,
        "grid.dy_m": config.grid.dy_m,
        "grid.dz_m": config.grid.dz_m,
        "grid.terrain": config.grid.terrain,
        "dynamics.psolver": config.dynamics.psolver,
        "dynamics.scalar_weno": config.dynamics.scalar_weno,
        "dynamics.dry_mass_correction": config.dynamics.dry_mass_correction,
        "closures.upper_damping_depth_m": config.closures.upper_damping_depth_m,
        "closures.upper_damping_rate_s_inv": config.closures.upper_damping_rate_s_inv,
        "closures.moving_frame": config.closures.moving_frame,
        "closures.u_move_m_s": config.closures.u_move_m_s,
        "closures.v_move_m_s": config.closures.v_move_m_s,
    }
    for name, wanted in expected.items():
        if actual[name] != wanted:
            refuse(
                code="CM1-CFG-PROFILE-DRIFT",
                gate="fixed-default",
                reason="the real-atmosphere product is sealed to the GPU-first 100 m profile",
                remedy=f"set {name} to {wanted!r}; add a separately named profile before widening",
                option=name,
                value=actual[name],
            )


def _require_dry_bootstrap_profile(config: CM1Config) -> None:
    expected: dict[str, object] = {
        "run.duration_s": 60.0,
        "run.dt_s": 1.0,
        "run.history_interval_s": 60.0,
        "run.restart_interval_s": 60.0,
        "grid.nx": 16,
        "grid.ny": 16,
        "grid.nz": 16,
        "grid.dx_m": 1000.0,
        "grid.dy_m": 1000.0,
        "grid.dz_m": 1000.0,
        "dynamics.psolver": 2,
        "dynamics.scalar_weno": 0,
        "dynamics.dry_mass_correction": True,
        "boundaries.west": 1,
        "boundaries.east": 1,
        "boundaries.south": 1,
        "boundaries.north": 1,
        "boundaries.bottom": 1,
        "boundaries.top": 1,
        "boundaries.radiative_order": 1,
        "closures.sgs_model": 0,
        "closures.vertical_implicit_diffusion": False,
        "closures.upper_damping": 0,
        "closures.upper_damping_depth_m": 0.0,
        "closures.upper_damping_rate_s_inv": 0.0,
        "closures.moving_frame": False,
        "closures.u_move_m_s": 0.0,
        "closures.v_move_m_s": 0.0,
    }
    actual: dict[str, object] = {
        "run.duration_s": config.run.duration_s,
        "run.dt_s": config.run.dt_s,
        "run.history_interval_s": config.run.history_interval_s,
        "run.restart_interval_s": config.run.restart_interval_s,
        "grid.nx": config.grid.nx,
        "grid.ny": config.grid.ny,
        "grid.nz": config.grid.nz,
        "grid.dx_m": config.grid.dx_m,
        "grid.dy_m": config.grid.dy_m,
        "grid.dz_m": config.grid.dz_m,
        "dynamics.psolver": config.dynamics.psolver,
        "dynamics.scalar_weno": config.dynamics.scalar_weno,
        "dynamics.dry_mass_correction": config.dynamics.dry_mass_correction,
        "boundaries.west": config.boundaries.west,
        "boundaries.east": config.boundaries.east,
        "boundaries.south": config.boundaries.south,
        "boundaries.north": config.boundaries.north,
        "boundaries.bottom": config.boundaries.bottom,
        "boundaries.top": config.boundaries.top,
        "boundaries.radiative_order": config.boundaries.radiative_order,
        "closures.sgs_model": config.closures.sgs_model,
        "closures.vertical_implicit_diffusion": config.closures.vertical_implicit_diffusion,
        "closures.upper_damping": config.closures.upper_damping,
        "closures.upper_damping_depth_m": config.closures.upper_damping_depth_m,
        "closures.upper_damping_rate_s_inv": config.closures.upper_damping_rate_s_inv,
        "closures.moving_frame": config.closures.moving_frame,
        "closures.u_move_m_s": config.closures.u_move_m_s,
        "closures.v_move_m_s": config.closures.v_move_m_s,
    }
    for name, wanted in expected.items():
        if actual[name] != wanted:
            refuse(
                code="CM1-CFG-PROFILE-DRIFT",
                gate="fixed-default",
                reason="the dry bootstrap is sealed to its separately identified reduced oracle campaign",
                remedy=f"set {name} to {wanted!r}; add a separately named profile before widening",
                option=name,
                value=actual[name],
            )


def load_config(path: str | Path) -> CM1Config:
    try:
        source = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-CFG-PATH",
            gate="config-grammar",
            reason=f"the configuration path cannot be resolved exactly: {error}",
            remedy="supply an existing regular UTF-8 TOML file",
            option="config",
            value=str(path),
        )
    if not source.is_file():
        refuse(
            code="CM1-CFG-PATH",
            gate="config-grammar",
            reason="the configuration path is not a regular file",
            remedy="supply an existing regular UTF-8 TOML file",
            option="config",
            value=str(source),
        )
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        refuse(
            code="CM1-CFG-PARSE",
            gate="config-grammar",
            reason=f"the configuration cannot be parsed exactly: {error}",
            remedy="supply a UTF-8 TOML document with the declared v1 tables",
            option="config",
            value=str(source),
        )
    return parse_mapping(raw)


__all__ = [
    "BoundaryConfig",
    "CM1Config",
    "DiagnosticsConfig",
    "CartesianRuntimeConfig",
    "ClosureConfig",
    "CONFIG_SCHEMA",
    "tornado_diagnostic_selection",
    "DynamicsConfig",
    "GridConfig",
    "FREE_SLIP_LOWER_BOUNDARY",
    "InitializationConfig",
    "LagrangianConfig",
    "Lane",
    "LowerBoundaryConfig",
    "PhysicsConfig",
    "RunConfig",
    "load_config",
    "parse_mapping",
    "require_vertical_coordinate",
]
