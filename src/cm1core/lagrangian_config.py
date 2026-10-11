"""The ``[lagrangian]`` configuration table: CM1's names, GM1's extensions.

One optional TOML table carries every Lagrangian and budget selector.  It is
optional as a whole -- a document that omits it is exactly the document it was
before this table existed, byte for byte, so every sealed ``sha256`` that
travels inside a recorded receipt as ``plan_sha256`` is unchanged -- and every
key inside it is optional too.  What is **not** optional is the projection:
when the table is present, :meth:`LagrangianConfig.record` states *every*
value, declared or defaulted, so the sealed document names the whole
configuration rather than inheriting an implicit scientific default.  That is
the same rule ``config.py``'s ``param6`` vertical words already follow.

Every key that CM1 has a word for keeps CM1's word:

============================  =========================================
key                           CM1 authority
============================  =========================================
``iprcl``                     ``run/namelist.input:87``
``nparcels``                  ``run/namelist.input:88``
``prclfrq_s``                 ``run/namelist.input:24`` (``prclfrq``)
``prcl_*`` (sixteen)          ``input.F:330-333`` ``&param13``
``output_pdcomp``             ``run/namelist.input:277``
``dohifrq`` / ``hifrqfrq_s``  ``cm1.F:2645-2647``
============================  =========================================

The rest are GM1 extensions and are named so in their docstrings and in the
receipt: the seeding modes, the backward direction, the vorticity budget, the
material circuits, the high-frequency window and the vortex-following child
sub-volume ``[lagrangian.nest_child_window]``.

CM1's ``iorigin`` is not here: it is ``[initialization].iorigin``, spelled once,
and every parcel coordinate this table names is in its frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .lagrangian_parcels import (
    PARCEL_SWITCH_ORDER,
    SEEDING_MODES,
    ParcelGrid,
    ParcelSeeding,
    ParcelSwitchContext,
    parcel_attribute_layout,
)

LAGRANGIAN_CONFIG_SCHEMA = "gpuwm-cm1.lagrangian-config/v1"

#: Non-``prcl_*`` keys and their CM1-stated defaults.  ``iprcl=0`` is CM1's
#: own shipped value (``run/namelist.input:87``), so an empty ``[lagrangian]``
#: table integrates no parcels and changes no result.
_DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "iprcl": 0,
    "nparcels": 1,
    "prclfrq_s": 60.0,
    "direction": "forward",
    "seeding": "grid",
    "seed_nx": 60,
    "seed_ny": 60,
    "seed_nz": 10,
    "seed_dx_m": 2000.0,
    "seed_dy_m": 2000.0,
    "seed_dz_m": 1000.0,
    "seed_x0_m": 0.0,
    "seed_y0_m": 0.0,
    "seed_z0_m": 0.0,
    "seed_vorticity_threshold_s_inv": 0.01,
    "vorticity_budget": False,
    "vorticity_budget_frq_s": 60.0,
    "vorticity_budget_history": False,
    "parcel_vorticity_budget": False,
    "circuits": 0,
    "circuit_parcels": 0,
    "circuit_radius_m": 0.0,
    "circuit_center_x_m": 0.0,
    "circuit_center_y_m": 0.0,
    "circuit_center_z_m": 0.0,
    "circuit_spacing_z_m": 0.0,
    "output_pdcomp": 0,
    "dohifrq": False,
    "hifrqfrq_s": 60.0,
    "hifrq_i0": 0, "hifrq_i1": 0,
    "hifrq_j0": 0, "hifrq_j1": 0,
    "hifrq_k0": 0, "hifrq_k1": 0,
})

#: ``&param13``'s sixteen switches.  CM1's own deck sets every one of them to
#: 1 (``run/namelist.input:332-349``); this table defaults them the same way.
#: These are the *requested* values, and the sealed projection keeps them so:
#: ``param.F:6257-6282`` resolves them against the run's other words, which
#: this table does not hold, so the columns are laid out by
#: :meth:`LagrangianConfig.attribute_layout` with the run's
#: :class:`~cm1core.lagrangian_parcels.ParcelSwitchContext`.
_SWITCH_DEFAULTS: Mapping[str, int] = MappingProxyType({
    "prcl_th": 1, "prcl_t": 1, "prcl_prs": 1, "prcl_ptra": 1,
    "prcl_q": 1, "prcl_nc": 1, "prcl_km": 1, "prcl_kh": 1,
    "prcl_tke": 1, "prcl_dbz": 1, "prcl_b": 1, "prcl_vpg": 1,
    "prcl_vort": 1, "prcl_rho": 1, "prcl_qsat": 1, "prcl_sfc": 1,
})
# Written out rather than built from PARCEL_SWITCH_ORDER so the sixteen keys
# this table accepts are literal text: tools/build_cm1_gm1_keymap.py reads
# them without importing this module, and a key set that only exists after a
# comprehension runs cannot be checked against config.py's schema.
assert tuple(_SWITCH_DEFAULTS) == PARCEL_SWITCH_ORDER

#: GM1 EXTENSION: ``[lagrangian.nest_child]``, parcels integrated in a live
#: nest child (:mod:`cm1core.nest_parcels`).  A sub-table, projected only when
#: declared, so every document sealed before it keeps its bytes.
NEST_CHILD_TABLE = "nest_child"

#: GM1 EXTENSION: ``[lagrangian.nest_child_window]``, the vortex-following
#: sub-volume of the live child written every child step
#: (:mod:`cm1core.nest_child_window`).  Also a sub-table, also projected only
#: when declared.
NEST_CHILD_WINDOW_TABLE = "nest_child_window"

LAGRANGIAN_KEYS: frozenset[str] = (
    frozenset(_DEFAULTS) | frozenset(_SWITCH_DEFAULTS)
    | frozenset({NEST_CHILD_TABLE, NEST_CHILD_WINDOW_TABLE})
)

#: Every ``[lagrangian.nest_child]`` key, each one required: a child seed box
#: that inherited an origin, a spacing or a frame would be a box the document
#: does not name, and last night's box was exactly such a misreading.
NEST_CHILD_KEYS: frozenset[str] = frozenset({
    "seed_frame", "nparcels", "prclfrq_s",
    "seed_nx", "seed_ny", "seed_nz",
    "seed_dx_m", "seed_dy_m", "seed_dz_m",
    "seed_x0_m", "seed_y0_m", "seed_z0_m",
})
NEST_CHILD_SEED_FRAME = "parent-grid-origin"

#: Every ``[lagrangian.nest_child_window]`` key, each one required, for the
#: same reason the child seed box requires every one of its own: a sub-volume
#: that inherited an extent, a cadence, a centre rule or a disk budget would
#: be a volume the sealed document does not name, and the one thing a
#: high-cadence stream must never do is surprise a run's disk.
NEST_CHILD_WINDOW_KEYS: frozenset[str] = frozenset({
    "frame", "follow", "center_x_m", "center_y_m",
    "half_width_x_m", "half_width_y_m", "top_m",
    "stride_child_steps", "start_s", "end_s",
    "budget_bytes_per_model_minute",
})
NEST_CHILD_WINDOW_FRAME = "parent-grid-origin"
#: ``tornado-criterion``: the window centres on the centre
#: :mod:`cm1core.nest_tornado_criterion` found at the last parent step (the
#: first maximum of the child's 10 m corner vorticity).  ``fixed``: on the
#: declared ``center_x_m``/``center_y_m``.
#: ``nest-centre`` centres the window on the child's own centre after every
#: nest move: the window rides a storm-following nest (``nest.follow_storm``),
#: whose tracker keeps the vortex within half a parent cell of that centre.  It
#: is the mode for a child too small to hold the tornado criterion's 1000 m
#: edge margin (a 1 km, 2 m child), where ``tornado-criterion`` has no centre.
NEST_CHILD_WINDOW_FOLLOW: tuple[str, ...] = ("tornado-criterion", "fixed", "nest-centre")

#: GM1 extensions added after documents declaring this table were sealed.
_PROJECTED_WHEN_TRUE: frozenset[str] = frozenset({
    "vorticity_budget_history", "parcel_vorticity_budget",
})

DIRECTIONS: tuple[str, ...] = ("forward", "backward")


class LagrangianConfigError(ValueError):
    """A ``[lagrangian]`` value that names no runnable diagnostic."""


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LagrangianConfigError(f"lagrangian.{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LagrangianConfigError(f"lagrangian.{name} must be finite")
    return 0.0 if result == 0.0 else result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LagrangianConfigError(
            f"lagrangian.{name} must be an integer >= {minimum}"
        )
    return value


def _flag(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise LagrangianConfigError(
            f"lagrangian.{name} is a CM1 integer selector, not a TOML boolean; "
            "CM1 tests it with .eq.1, so write 0 or 1"
        )
    if not isinstance(value, int) or value not in (0, 1):
        raise LagrangianConfigError(f"lagrangian.{name} must be 0 or 1")
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise LagrangianConfigError(
            f"lagrangian.{name} must be the TOML literal true or false"
        )
    return value


@dataclass(frozen=True, slots=True)
class NestChildParcelConfig:
    """``[lagrangian.nest_child]``: a box of parcels in the live nest child.

    GM1 EXTENSION (CM1 r21.1 has no nesting).  Every coordinate is metres from
    the **parent grid's origin** -- the parent's west face ``xf(1)``, south
    face ``yf(1)`` and the ground -- in the parent's translating frame, which
    is the frame the runtime publishes parent parcels in when
    ``[initialization].iorigin = 1``.  It is neither the
    domain centre (CM1 ``iorigin = 2``) nor the child's corner, and the one
    admitted ``seed_frame`` value says so in the sealed document.  The box is
    ``init3d.F:381-387``'s triple loop over explicit counts and spacings.
    """

    seed_frame: str
    nparcels: int
    prclfrq_s: float
    seed_nx: int
    seed_ny: int
    seed_nz: int
    seed_dx_m: float
    seed_dy_m: float
    seed_dz_m: float
    seed_x0_m: float
    seed_y0_m: float
    seed_z0_m: float

    @property
    def seed_count(self) -> int:
        return self.seed_nx * self.seed_ny * self.seed_nz

    def projection(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(NEST_CHILD_KEYS)}

    def record(self) -> dict[str, Any]:
        return {
            **self.projection(),
            "seed_count": self.seed_count,
            "extent_m": {
                "x": [self.seed_x0_m, self.seed_x0_m + self.seed_dx_m * (self.seed_nx - 1)],
                "y": [self.seed_y0_m, self.seed_y0_m + self.seed_dy_m * (self.seed_ny - 1)],
                "z": [self.seed_z0_m, self.seed_z0_m + self.seed_dz_m * (self.seed_nz - 1)],
            },
            "frame": (
                "metres from the parent grid origin (parent xf(1), yf(1), ground), in the "
                "parent's translating frame; not the domain centre and not the child corner"
            ),
            "gm1_extension": "CM1 r21.1 has no nesting; cm1core.nest_parcels",
        }


def _parse_nest_child(value: Any) -> NestChildParcelConfig:
    prefix = f"lagrangian.{NEST_CHILD_TABLE}"
    if not isinstance(value, Mapping):
        raise LagrangianConfigError(
            f"{prefix} must be a TOML table ([{prefix}]); a scalar there names no seed box"
        )
    unknown = sorted(set(value) - NEST_CHILD_KEYS)
    if unknown:
        raise LagrangianConfigError(
            f"{prefix}.{unknown[0]} is not a child-parcel selector; an unconsumed key could "
            "name a seeding or cadence the run would not use"
        )
    missing = sorted(NEST_CHILD_KEYS - set(value))
    if missing:
        raise LagrangianConfigError(
            f"{prefix}.{missing[0]} must be declared; every child-parcel selector is "
            "explicit, so no seed box inherits an origin, spacing, cadence or frame"
        )
    if value["seed_frame"] != NEST_CHILD_SEED_FRAME:
        raise LagrangianConfigError(
            f"{prefix}.seed_frame must be \"{NEST_CHILD_SEED_FRAME}\": seed coordinates are "
            "metres from the parent grid's origin (its west and south faces and the ground), "
            "not from the domain centre (iorigin = 2) or the child's corner; a box read in "
            "the other frame lands half a domain away from the vortex"
        )
    counts = {name: _integer(f"{NEST_CHILD_TABLE}.{name}", value[name], minimum=1)
              for name in ("nparcels", "seed_nx", "seed_ny", "seed_nz")}
    lengths: dict[str, float] = {}
    for name in ("prclfrq_s", "seed_dx_m", "seed_dy_m", "seed_dz_m"):
        lengths[name] = _number(f"{NEST_CHILD_TABLE}.{name}", value[name])
        if lengths[name] <= 0.0:
            raise LagrangianConfigError(
                f"{prefix}.{name} must be positive; a zero spacing stacks a row of parcels "
                "on one point and a zero cadence never records a frame"
            )
    for name in ("seed_x0_m", "seed_y0_m", "seed_z0_m"):
        lengths[name] = _number(f"{NEST_CHILD_TABLE}.{name}", value[name])
    config = NestChildParcelConfig(seed_frame=NEST_CHILD_SEED_FRAME, **counts, **lengths)  # type: ignore[arg-type]
    if config.seed_count > config.nparcels:
        raise LagrangianConfigError(
            f"{prefix} seeds {config.seed_count} parcels but nparcels is {config.nparcels}; "
            "init3d.F:375-383 stops CM1 when its seeding loop overruns the parcel array"
        )
    return config


@dataclass(frozen=True, slots=True)
class NestChildWindowConfig:
    """``[lagrangian.nest_child_window]``: the child sub-volume written every child step.

    GM1 EXTENSION (CM1 r21.1 has no nesting, and ``hifrq.F`` writes the whole
    domain at a fixed place).  This table names a box that **follows the
    vortex** and whose words are written at the child's own step rather than
    at a history cadence, because that is the one thing an accurate trajectory
    into a 100 m/s core needs and the one thing no saved history frame can be
    made to give: between two frames ``T`` apart the offline wind is an
    interpolation whose error grows like ``T**2``
    (:mod:`cm1core.nest_parcel_backward`).

    ``center_x_m``/``center_y_m`` and ``half_width_x_m``/``half_width_y_m``
    are metres from the **parent grid's origin** -- the parent's west face
    ``xf(1)``, south face ``yf(1)`` -- in the parent's translating frame,
    exactly :class:`NestChildParcelConfig`'s frame.  ``top_m`` is height above
    the ground; the window always starts at the ground, because the air a
    tornado ingests is in the lowest few hundred metres and a sub-volume that
    began above it could not integrate one trajectory through it.
    """

    frame: str
    follow: str
    center_x_m: float
    center_y_m: float
    half_width_x_m: float
    half_width_y_m: float
    top_m: float
    stride_child_steps: int
    start_s: float
    end_s: float
    budget_bytes_per_model_minute: int

    @property
    def follows_criterion(self) -> bool:
        return self.follow == "tornado-criterion"

    @property
    def follows_nest(self) -> bool:
        return self.follow == "nest-centre"

    def projection(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(NEST_CHILD_WINDOW_KEYS)}

    def record(self) -> dict[str, Any]:
        return {
            **self.projection(),
            "frame_meaning": (
                "metres from the parent grid origin (parent xf(1), yf(1)) in the parent's "
                "translating frame; top_m is height above the ground and the window always "
                "reaches the ground"
            ),
            "centre_rule": (
                "the tornado criterion's own centre at the last parent step "
                "(cm1core.nest_tornado_criterion), held at the last found centre when a "
                "sample finds none"
                if self.follows_criterion else
                "the child's own centre after every nest move (a storm-following nest keeps "
                "its tracked vortex within half a parent cell of it); center_x_m/center_y_m "
                "are not read"
                if self.follows_nest else
                f"fixed at ({self.center_x_m}, {self.center_y_m}) m from the parent grid origin"
            ),
            "gm1_extension": (
                "CM1 r21.1 has no nesting and hifrq.F writes the whole domain at a fixed "
                "place; cm1core.nest_child_window"
            ),
        }


def _parse_nest_child_window(value: Any) -> NestChildWindowConfig:
    prefix = f"lagrangian.{NEST_CHILD_WINDOW_TABLE}"
    if not isinstance(value, Mapping):
        raise LagrangianConfigError(
            f"{prefix} must be a TOML table ([{prefix}]); a scalar there names no sub-volume"
        )
    unknown = sorted(set(value) - NEST_CHILD_WINDOW_KEYS)
    if unknown:
        raise LagrangianConfigError(
            f"{prefix}.{unknown[0]} is not a child-window selector; an unconsumed key could "
            "name an extent, a cadence or a disk budget the run would not use"
        )
    missing = sorted(NEST_CHILD_WINDOW_KEYS - set(value))
    if missing:
        raise LagrangianConfigError(
            f"{prefix}.{missing[0]} must be declared; every child-window selector is explicit, "
            "so no sub-volume inherits an extent, a centre rule, a cadence, an interval or a "
            "disk budget"
        )
    if value["frame"] != NEST_CHILD_WINDOW_FRAME:
        raise LagrangianConfigError(
            f'{prefix}.frame must be "{NEST_CHILD_WINDOW_FRAME}": the centre and the half-widths '
            "are metres from the parent grid's origin (its west and south faces), not from the "
            "domain centre (iorigin = 2) and not from the child's corner; a window read in the "
            "other frame streams a volume half a domain from the vortex"
        )
    if value["follow"] not in NEST_CHILD_WINDOW_FOLLOW:
        raise LagrangianConfigError(
            f"{prefix}.follow must be one of " + ", ".join(NEST_CHILD_WINDOW_FOLLOW)
            + "; an unadmitted mode would leave the window where it was seeded while the vortex "
            "moved out of it, and the stream would hold the air beside the tornado"
        )
    counts = {"stride_child_steps": _integer(
        f"{NEST_CHILD_WINDOW_TABLE}.stride_child_steps", value["stride_child_steps"], minimum=1)}
    budget = _integer(f"{NEST_CHILD_WINDOW_TABLE}.budget_bytes_per_model_minute",
                      value["budget_bytes_per_model_minute"], minimum=1)
    lengths: dict[str, float] = {}
    for name in ("half_width_x_m", "half_width_y_m", "top_m"):
        lengths[name] = _number(f"{NEST_CHILD_WINDOW_TABLE}.{name}", value[name])
        if lengths[name] <= 0.0:
            raise LagrangianConfigError(
                f"{prefix}.{name} must be positive; a window with a zero extent holds no cell, "
                "and no trajectory can be integrated through no cells"
            )
    for name in ("center_x_m", "center_y_m", "start_s", "end_s"):
        lengths[name] = _number(f"{NEST_CHILD_WINDOW_TABLE}.{name}", value[name])
    if lengths["start_s"] < 0.0:
        raise LagrangianConfigError(
            f"{prefix}.start_s must be at or after the start of the run; a negative model time "
            "names no step the run takes"
        )
    if lengths["end_s"] <= lengths["start_s"]:
        raise LagrangianConfigError(
            f"{prefix}.end_s must be later than start_s; an empty interval opens the stream and "
            "closes it without a single record, and the trajectories it was declared for could "
            "not be integrated"
        )
    follow = str(value["follow"])
    if follow in ("tornado-criterion", "nest-centre") and (
            lengths["center_x_m"] != 0.0 or lengths["center_y_m"] != 0.0):
        raise LagrangianConfigError(
            f"{prefix}.center_x_m and center_y_m must be 0.0 when follow = \"{follow}\"; "
            "a declared centre the run then ignores would name a sub-volume in the sealed "
            "document that is not the one on disk"
        )
    return NestChildWindowConfig(
        frame=NEST_CHILD_WINDOW_FRAME, follow=follow,
        budget_bytes_per_model_minute=budget, **counts, **lengths,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class LagrangianConfig:
    """Every Lagrangian and budget selector, resolved to a value.

    The selectors are held as an immutable tuple of ``(key, value)`` pairs in
    sorted key order rather than a mapping, for one concrete reason:
    ``CM1Config.canonical_dict`` reaches this object through
    ``dataclasses.asdict``, which deep-copies what it finds, and a
    ``MappingProxyType`` cannot be deep-copied -- the sealed hash of every
    document that declares this table would raise instead of being computed.
    """

    items: tuple[tuple[str, Any], ...]

    @property
    def values(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.items))

    def get(self, name: str) -> Any:
        return self.values[name]

    @property
    def parcels_enabled(self) -> bool:
        return int(self.values["iprcl"]) == 1

    @property
    def budget_enabled(self) -> bool:
        return bool(self.values["vorticity_budget"])

    @property
    def vorticity_budget_history(self) -> bool:
        """GM1 extension: the vorticity budget as history carriers."""

        return bool(self.values.get("vorticity_budget_history", False))

    @property
    def parcel_vorticity_budget(self) -> bool:
        """GM1 extension: the vorticity budget sampled along every parcel."""

        return bool(self.values.get("parcel_vorticity_budget", False))

    @property
    def circuits_enabled(self) -> bool:
        return int(self.values["circuits"]) > 0

    @property
    def pdcomp_enabled(self) -> bool:
        return int(self.values["output_pdcomp"]) == 1

    @property
    def hifrq_enabled(self) -> bool:
        return bool(self.values["dohifrq"])

    @property
    def backward(self) -> bool:
        return self.values["direction"] == "backward"

    @property
    def nest_child(self) -> NestChildParcelConfig | None:
        """``[lagrangian.nest_child]``, or ``None`` when the sub-table is absent."""

        pairs = self.values.get(NEST_CHILD_TABLE)
        return None if pairs is None else NestChildParcelConfig(**dict(pairs))

    @property
    def nest_child_window(self) -> NestChildWindowConfig | None:
        """``[lagrangian.nest_child_window]``, or ``None`` when the sub-table is absent."""

        pairs = self.values.get(NEST_CHILD_WINDOW_TABLE)
        return None if pairs is None else NestChildWindowConfig(**dict(pairs))

    def seeding(self) -> ParcelSeeding:
        v = self.values
        return ParcelSeeding(
            mode=v["seeding"], nparcels=int(v["nparcels"]),
            seed_nx=int(v["seed_nx"]), seed_ny=int(v["seed_ny"]),
            seed_nz=int(v["seed_nz"]),
            seed_dx_m=float(v["seed_dx_m"]), seed_dy_m=float(v["seed_dy_m"]),
            seed_dz_m=float(v["seed_dz_m"]),
            seed_x0_m=float(v["seed_x0_m"]), seed_y0_m=float(v["seed_y0_m"]),
            seed_z0_m=float(v["seed_z0_m"]),
            vorticity_threshold_s_inv=float(v["seed_vorticity_threshold_s_inv"]),
        )

    def attribute_layout(self, context: ParcelSwitchContext) -> Any:
        """The attribute list CM1 lays out for this table on the run ``context`` names.

        ``param.F:6257-6282`` overrides the requested switches from the run's
        own words (``iptra``, ``imoist``, ``ptype``, ``idm``, ``sgsmodel``,
        ``ipbl``, ``iusetke``, ``bbc``), which this table alone cannot know: the
        caller passes :func:`cm1core.lagrangian_parcels.parcel_switch_context`
        of its runtime contract, and the one resolver in
        :func:`cm1core.lagrangian_parcels.parcel_attribute_layout` applies them.
        """

        return parcel_attribute_layout(
            {name: int(self.values[name]) for name in PARCEL_SWITCH_ORDER},
            context,
        )

    def parcel_grid(self, *, nx: int, ny: int, nz: int,
                    dx_m: float, dy_m: float, dz_m: float,
                    iorigin: int, levels: Any = None) -> ParcelGrid:
        """The frame every published parcel position is measured in.

        CM1 has one ``iorigin`` (``param.F:7212-7213``) and this port spells
        it once, in ``[initialization]``.  The caller passes it here rather
        than the parcel table carrying a second, separately declarable copy
        that a document could set to the other value -- and it has no default,
        so the door's extent checks and the runtime cannot resolve it
        differently.  ``levels`` is the grid's CM1 vertical coordinate
        (``cm1_vertical_grid.VerticalLevels``); ``None`` is the uniform deck.
        """

        return ParcelGrid(nx=nx, ny=ny, nz=nz, dx_m=dx_m, dy_m=dy_m, dz_m=dz_m,
                          iorigin=int(iorigin), levels=levels)

    def projection(self) -> dict[str, Any]:
        """The canonical projection: every key, declared or defaulted.

        ``nest_child`` is present only when declared, and then as its whole
        table.

        The two vorticity-budget selectors are the one exception: they are
        projected only when true, so every document sealed before they existed
        keeps its exact ``plan_sha256`` and a document that turns one on names
        it in its sealed bytes.
        """

        out = {
            name: self.values[name] for name in sorted(self.values)
            if name not in _PROJECTED_WHEN_TRUE or self.values[name]
        }
        for table in (NEST_CHILD_TABLE, NEST_CHILD_WINDOW_TABLE):
            if table in out:
                out[table] = dict(out[table])
        return out

    def record(self, context: ParcelSwitchContext) -> dict[str, Any]:
        """The configuration record; ``context`` as for :meth:`attribute_layout`.

        ``selectors`` is the sealed projection -- the switches as requested --
        and ``parcel_attributes`` / ``nest_child_attributes`` are the layouts
        CM1 would write for them on this run: resolved switches, the rule that
        turned each one off, and the columns the track actually carries.
        """

        layout = self.attribute_layout(context) if self.parcels_enabled else None
        payload: dict[str, Any] = {
            "schema": LAGRANGIAN_CONFIG_SCHEMA,
            "selectors": self.projection(),
            "parcels_enabled": self.parcels_enabled,
            "vorticity_budget_enabled": self.budget_enabled,
            "circuits_enabled": self.circuits_enabled,
            "pdcomp_enabled": self.pdcomp_enabled,
            "high_frequency_enabled": self.hifrq_enabled,
            "cm1_named_keys": sorted(
                ["iprcl", "nparcels", "prclfrq_s", "output_pdcomp",
                 "dohifrq", "hifrqfrq_s", *PARCEL_SWITCH_ORDER]
            ),
            "gm1_extension_keys": sorted(
                set(_DEFAULTS)
                - {"iprcl", "nparcels", "prclfrq_s", "output_pdcomp",
                   "dohifrq", "hifrqfrq_s"}
            ),
        }
        if layout is not None:
            payload["parcel_attributes"] = layout.record()
        child = self.nest_child
        if child is not None:
            payload["gm1_extension_keys"] = sorted([*payload["gm1_extension_keys"], NEST_CHILD_TABLE])
            payload["nest_child"] = child.record()
            payload["nest_child_attributes"] = self.attribute_layout(context).record()
        window = self.nest_child_window
        if window is not None:
            payload["gm1_extension_keys"] = sorted(
                [*payload["gm1_extension_keys"], NEST_CHILD_WINDOW_TABLE]
            )
            payload["nest_child_window"] = window.record()
        return payload


def parse_lagrangian_table(table: Mapping[str, Any] | None) -> LagrangianConfig | None:
    """Bind one ``[lagrangian]`` table, or return ``None`` when absent.

    Unknown keys are the caller's refusal to raise (``config.py`` owns
    ``CM1-CFG-UNKNOWN``); this function refuses by *value*, and every refusal
    names the wrong result it prevents.
    """

    if table is None:
        return None
    values: dict[str, Any] = {}
    for name, default in _DEFAULTS.items():
        values[name] = table.get(name, default)
    for name, default in _SWITCH_DEFAULTS.items():
        values[name] = table.get(name, default)

    values["iprcl"] = _flag("iprcl", values["iprcl"])
    values["nparcels"] = _integer("nparcels", values["nparcels"], minimum=1)
    values["prclfrq_s"] = _number("prclfrq_s", values["prclfrq_s"])
    if values["prclfrq_s"] <= 0.0:
        raise LagrangianConfigError(
            "lagrangian.prclfrq_s must be positive; CM1's negative prclfrq "
            "sentinel (param.F) disables the write, and a run that declares "
            "parcels and never records them produces no trajectory at all"
        )
    if values["direction"] not in DIRECTIONS:
        raise LagrangianConfigError(
            "lagrangian.direction must be 'forward' or 'backward'; a backward "
            "run integrates the same RK2 with -dt"
        )
    if values["seeding"] not in SEEDING_MODES:
        raise LagrangianConfigError(
            "lagrangian.seeding must be one of " + ", ".join(SEEDING_MODES)
        )
    for name in ("seed_nx", "seed_ny", "seed_nz"):
        values[name] = _integer(name, values[name], minimum=1)
    for name in ("seed_dx_m", "seed_dy_m", "seed_dz_m"):
        values[name] = _number(name, values[name])
        if values[name] <= 0.0:
            raise LagrangianConfigError(
                f"lagrangian.{name} must be positive; a zero seeding spacing "
                "stacks every parcel of that row on one point and the "
                "trajectories are duplicates of one trajectory"
            )
    for name in ("seed_x0_m", "seed_y0_m", "seed_z0_m",
                 "seed_vorticity_threshold_s_inv",
                 "circuit_radius_m", "circuit_center_x_m", "circuit_center_y_m",
                 "circuit_center_z_m", "circuit_spacing_z_m",
                 "vorticity_budget_frq_s", "hifrqfrq_s"):
        values[name] = _number(name, values[name])
    values["vorticity_budget"] = _boolean("vorticity_budget", values["vorticity_budget"])
    if values["vorticity_budget"] and values["vorticity_budget_frq_s"] <= 0.0:
        raise LagrangianConfigError(
            "lagrangian.vorticity_budget_frq_s must be positive when the "
            "budget is on; a non-positive cadence would compute the budget and "
            "never publish it"
        )
    for name in ("vorticity_budget_history", "parcel_vorticity_budget"):
        values[name] = _boolean(name, values[name])
    if values["parcel_vorticity_budget"] and values["iprcl"] != 1:
        raise LagrangianConfigError(
            "lagrangian.parcel_vorticity_budget requires lagrangian.iprcl = 1; "
            "the vorticity-budget attributes are sampled at parcel positions, and "
            "with no integrator there are no positions to sample"
        )
    values["circuits"] = _integer("circuits", values["circuits"], minimum=0)
    values["circuit_parcels"] = _integer("circuit_parcels", values["circuit_parcels"], minimum=0)
    if values["circuits"] > 0:
        if values["circuit_parcels"] < 3:
            raise LagrangianConfigError(
                "lagrangian.circuit_parcels must be at least 3 when circuits "
                "are requested; two vertices enclose no area and the "
                "circulation of such a circuit is identically zero"
            )
        if values["circuit_radius_m"] <= 0.0:
            raise LagrangianConfigError(
                "lagrangian.circuit_radius_m must be positive when circuits are "
                "requested; a zero-radius circuit encloses no area, so its "
                "circulation is zero for every flow and measures nothing"
            )
        if values["iprcl"] != 1:
            raise LagrangianConfigError(
                "lagrangian.circuits requires lagrangian.iprcl = 1; a material "
                "circuit is carried by parcels, and without the integrator the "
                "circuit would stay at its seed position while the flow moved "
                "past it, reporting the circulation of a fixed ring as if it "
                "were a material one"
            )
        needed = values["circuits"] * values["circuit_parcels"]
        if needed > values["nparcels"]:
            raise LagrangianConfigError(
                f"lagrangian.circuits x circuit_parcels is {needed} but "
                f"nparcels is {values['nparcels']}; the last circuit would run "
                "past the parcel array and close its line integral over "
                "parcels that belong to another circuit"
            )
    values["output_pdcomp"] = _flag("output_pdcomp", values["output_pdcomp"])
    values["dohifrq"] = _boolean("dohifrq", values["dohifrq"])
    if values["dohifrq"] and values["hifrqfrq_s"] <= 0.0:
        raise LagrangianConfigError(
            "lagrangian.hifrqfrq_s must be positive when dohifrq is true; "
            "cm1.F:3229's hifrqfrq<=0 sentinel writes on every single step, "
            "which for a tornado-grade run is terabytes nobody asked for"
        )
    for name in ("hifrq_i0", "hifrq_i1", "hifrq_j0", "hifrq_j1",
                 "hifrq_k0", "hifrq_k1"):
        values[name] = _integer(name, values[name], minimum=0)
    for name in PARCEL_SWITCH_ORDER:
        values[name] = _flag(name, values[name])
    if NEST_CHILD_TABLE in table:
        child = _parse_nest_child(table[NEST_CHILD_TABLE])
        values[NEST_CHILD_TABLE] = tuple(sorted(child.projection().items()))
    if NEST_CHILD_WINDOW_TABLE in table:
        window = _parse_nest_child_window(table[NEST_CHILD_WINDOW_TABLE])
        values[NEST_CHILD_WINDOW_TABLE] = tuple(sorted(window.projection().items()))
    return LagrangianConfig(tuple(sorted(values.items())))


def lagrangian_defaults() -> dict[str, Any]:
    """The complete resolved default table, for a document that declares none."""

    return {**_DEFAULTS, **_SWITCH_DEFAULTS}


__all__ = [
    "DIRECTIONS", "LAGRANGIAN_CONFIG_SCHEMA", "LAGRANGIAN_KEYS",
    "LagrangianConfig", "LagrangianConfigError", "NEST_CHILD_KEYS",
    "NEST_CHILD_SEED_FRAME", "NEST_CHILD_TABLE",
    "NEST_CHILD_WINDOW_FOLLOW", "NEST_CHILD_WINDOW_FRAME",
    "NEST_CHILD_WINDOW_KEYS", "NEST_CHILD_WINDOW_TABLE",
    "NestChildParcelConfig", "NestChildWindowConfig",
    "lagrangian_defaults", "parse_lagrangian_table",
]
