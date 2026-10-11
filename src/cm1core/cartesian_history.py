"""Observational history frames published from the Cartesian CUDA runtime.

This module owns one thing: turning the completed device state of
``CudaCartesianRuntime`` into the raw binary32 arena and semantic field table
that the Rust ``cm1-forecast-publish`` boundary admits as a CM1FHS1 forecast
history envelope.  It performs no transform, regrid, render or restore; it
never reads an envelope back into a live runtime.

That is a division of labour, not a missing capability.  CM1 writes two kinds
of file and this lane writes the same two: an observational history that must
not change the trajectory, and a restart that carries everything needed to
continue it.  The restart is ``cm1core.cartesian_restart``, it is a CM1FCP1
envelope, and it is the one that is read back.  A history frame carries
twelve carriers chosen for a viewer; resuming from one would silently restart
the run from a subset of its own state.

Three properties of this runtime shape the profile:

* ``ua/u3d/va/v3d/wa/w3d`` are strided physical views into their padded
  owners, and ``zf`` is a broadcast of one source-proven vertical profile.
  None of the four can be copied in address order, so each is materialized
  into one registry-budgeted staging allocation before the transfer.
* There is no ``total_geopotential_zf`` carrier anywhere in this runtime.  It
  is omitted by name rather than substituted with a Python ``g*zf`` product,
  which would publish an invented field under a CM1 name.
* Publication is valid only at the post-solve3 quiescent boundary, where the
  current/A carriers agree and the padded owners have been synchronized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from types import MappingProxyType
from typing import Any

from .cuda_memory_inventory import shape_bytes
from .cuda_surface_drag import surface_validation_markers
from .cm1_output_selection import OUTPUT_GROUP, output_carrier_rows
from .forecast_arena import (
    ForecastArenaSnapshot,
    ForecastArenaWrite,
    HostArena,
    forecast_arena_nbytes,
    snapshot_forecast_arena,
    write_forecast_arena_snapshot_new,
)
from .cuda_composite_reflectivity import CM1_CREF_SEED_DBZ
from .gm1_runtime import (
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    MOVE_SPEED_X_NAME,
    MOVE_SPEED_Y_NAME,
    REFLECTIVITY_AUTHORITY,
    RuntimeClock,
    RuntimeConfig,
    cm1_real,
)


CARTESIAN_HISTORY_PROFILE_SCHEMA = "gm1.cartesian-history-profile/v1"
CARTESIAN_HISTORY_PUBLICATION_SCHEMA = "gm1.cartesian-history-publication/v1"
CARTESIAN_HISTORY_PROFILE_ID = "gm1.cartesian-history/v1"
CARTESIAN_HISTORY_STATUS = "PUBLISHED_CARTESIAN_HISTORY_FRAME"
# Two claims, because the run makes two different ones.  Without a publisher
# no Rust executes, so nothing has been finite-checked and the receipt must not
# say it has.
CARTESIAN_HISTORY_CLAIM_LIMIT = (
    "one observational frame of this runtime's completed device state, "
    "byte-exact and finite-checked by Rust; not a restartable checkpoint, a "
    "NetCDF product, a rendered field, or a validated forecast"
)
CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT = (
    "one observational frame of this runtime's completed device state, "
    "written as a raw carrier arena and a scheduler-scalar table with "
    "per-carrier SHA-256 and no forecast envelope; no Rust ran, so the "
    "payload is not finite-checked and the cm1.F:3512-3513 domain location "
    "was not re-derived; not a restartable checkpoint, a NetCDF product, a "
    "rendered field, or a validated forecast"
)
# rust/cm1-state/src/forecast.rs:52 and :56.  One forecast payload is capped
# and the cap is enforced here, before a single device word is copied to the
# host.  The number is a host-memory ceiling, not a container limit: the
# CM1FHS1 prefix carries a 64-bit header length and every payload offset is an
# exact decimal JSON integer, so the envelope has always addressed payloads
# past 2^32 bytes.  What it bounds is admission, which reads the whole
# artifact and then materializes one float32 copy of every carrier.
MAX_FORECAST_PAYLOAD_BYTES = 16 * 1024 * 1024 * 1024
FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE = 2
FORECAST_HISTORY_MAGIC = "CM1FHS1\0"
# v2 is the publisher that takes the scheduler-scalar table, which is how
# CM1's per-history-time umove/vmove/domainlocx/domainlocy reach the envelope.
# A v1 publisher would write a frame with no domain location in it and no sign
# that one was meant to be there, so it is refused by ABI rather than fallen
# back to.  v3 adds `restore`, the reading half of CM1's restart pair
# (restart_read.F, cm1.F:2666): a v2 publisher can write this lane's
# checkpoint and cannot read it back, so a run that intended to be resumable
# would produce envelopes nothing could continue from.
FORECAST_PUBLISH_ABI = "gpuwm-cm1-forecast-publish/v3"

# Which carriers one frame publishes.  Unknown values are refused by name
# below; there is no third behaviour reachable by spelling.
COMPLETE_CARRIER_PROFILE = "complete"
RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE = "reconstructible-omitted"
CARTESIAN_HISTORY_CARRIER_PROFILES = (
    COMPLETE_CARRIER_PROFILE, RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
)
CARTESIAN_HISTORY_REDUCED_PROFILE_ID = "gm1.cartesian-history-reconstructible-omitted/v1"

ARENA_FILENAME = "history-carriers.f32"
FIELD_TABLE_FILENAME = "history-field-table.tsv"
# The table handed to cm1-forecast-publish, whose raw-path column must be
# absolute.  Written beside the published table, used for the one publisher
# call, and removed before the frame is renamed into place.
PUBLISHER_FIELD_TABLE_FILENAME = ".history-field-table.publisher.tsv"
SCALAR_TABLE_FILENAME = "history-scheduler-scalars.tsv"
RECEIPT_FILENAME = "history-publication.json"
# The installed render door maps the suffix to the envelope format and
# accepts ".cm1fhs" (doors.py:291).  Naming the file after the CM1FHS1
# magic instead would make the shipped renderer refuse its own producer
# unless the caller also passed --format.
ENVELOPE_FILENAME = "history.cm1fhs"

STAGING_BUFFER_NAME = "history.stage"

# A published carrier that is a leading-axis window of one packed runtime
# array, and the window it takes.  ``q3d`` is CM1's mass prefix and ``nc3d``
# the number suffix that follows it (src/param.F:3441-3449).
_PACKED_SCALAR_WINDOWS = {
    "q3d": lambda carrier, value: (0, carrier.shape[0]),
    "nc3d": lambda carrier, value: (value.shape[0] - carrier.shape[0], value.shape[0]),
}

# The admitted child render history carries thirteen names.  This runtime has
# no geopotential carrier, so twelve are published and the thirteenth is
# refused by name below rather than reconstructed in Python.
OMITTED_CARRIERS = MappingProxyType({
    "total_geopotential_zf": (
        "the Cartesian runtime holds no geopotential carrier; publishing a "
        "Python g*zf product under this CM1 name would present an invented "
        "field as model output"
    ),
})

# The opt-in reduced profile drops exactly the three-dimensional carriers a
# reader rebuilds from the frame's own remaining words by a CM1 formula.  Each
# entry is that formula and its authority; a carrier whose reconstruction
# needs anything this frame does not carry is not listed and stays published.
RECONSTRUCTIBLE_CARRIERS = MappingProxyType({
    "prs": (
        "CM1 r21.1 solve2.F:1588 prs = p00*((pi0+pp3d)**(cp/rd)), with p00 from "
        "constants.F:17, rd and cp from constants.F:113-114 and the exponent "
        "cpdrd = cp/rd from constants.F:137.  That is the expression this "
        "runtime's own pressure kernel evaluates (cuda_cartesian_startup and "
        "cuda_gm1_tilestream_rk3_eqtset2_remove_eos: p00=100000, "
        "pow(pi0+pp3d, cp/rd)).  The frame carries pp3d at full extent and pi0 "
        "as the base.pi0 column this runtime broadcasts it from, so every input "
        "of the formula is published; cm1_state::reduced_history is the reader "
        "that applies it.  Reproducing the published words bit for bit "
        "additionally requires the same single-precision pow this runtime pins"
    ),
    "zf": (
        "metrics.zf_profile is the single vertical profile this runtime "
        "broadcasts over every horizontal plane, and the broadcast is proved "
        "plane by plane before upload (cuda_immutable_metrics."
        "upload_immutable_source_metrics); zf[k, j, i] = metrics.zf_profile[k] "
        "at every i and j"
    ),
})
# rf is deliberately absent from that table: CM1 builds it from rho with the
# vertical interpolation weights c1/c2 (solve3.F:666) and the second-order
# edge extrapolations at k=1 and k=nk+1 (solve3.F:678-679), none of which this
# frame carries.  Omitting it would lose the w-face density, not compress it.
REDUCED_PROFILE_READER_COST = (
    "a reduced frame carries no prs or zf words; a reader rebuilds them from "
    "the columns it does carry, by the formulas recorded in "
    "reconstructible_carriers.  rust/cm1-state/src/reduced_history.rs is that "
    "reader and cm1-render --format cm1fhs applies it, so --field prs and "
    "--field zf resolve against a reduced frame exactly as against a complete "
    "one.  A consumer that reads the carrier inventory directly and does not "
    "apply the formulas sees a frame two carriers short"
)


@dataclass(frozen=True, slots=True)
class CartesianHistoryCarrier:
    """One published float32 carrier and where the live runtime holds it."""

    name: str
    meaning: str
    units: str
    stagger: str
    shape: tuple[int, ...]
    group: str
    key: str
    staged: bool
    # True when the published carrier is the one vertical column of a source
    # that the runtime broadcasts over every horizontal plane.  Staging takes
    # [:, 0, 0] and only after the runtime's own plane-by-plane proof; without
    # that proof the column would misplace every height in the frame.
    horizontal_profile_column: bool = False

    @property
    def nbytes(self) -> int:
        return shape_bytes(self.shape)

    @property
    def words(self) -> int:
        return self.nbytes // 4

    def bounds(self) -> tuple[int, ...]:
        """CM1 one-based inclusive index bounds in the declared axis order."""

        return tuple(value for extent in self.shape for value in (1, extent))

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name, "meaning": self.meaning, "units": self.units,
            "stagger": self.stagger, "shape": list(self.shape),
            "bounds": list(self.bounds()), "nbytes": self.nbytes,
            "runtime_group": self.group, "runtime_key": self.key,
            "staged_through_registry": self.staged,
            "horizontal_profile_column": self.horizontal_profile_column,
        }


def _reflectivity_rows(config: RuntimeConfig) -> tuple[tuple[Any, ...], ...]:
    """CM1's ``output_dbz`` carriers, under CM1's own output names.

    ``writeout.F:4089-4107`` writes the three-dimensional field as ``dbz`` and
    ``writeout.F:1081-1091`` writes the composite beside it as ``cref``; both
    appear only when ``output_dbz=1`` (README.namelist:1366-1369), which is
    why this port publishes them per run rather than always.  The chosen-level
    plane has no CM1 name because CM1 leaves a user to slice the volume, so it
    travels under a name of this port's own and says which level it is.

    Cost, stated rather than left to be counted: the frame carries twelve
    carriers and about sixteen float32 words per gridpoint without this.
    ``dbz`` adds exactly one more word per gridpoint; ``cref`` and the level
    plane add one word per *column* each, which is ``1/nz`` of a word per
    gridpoint apiece.  :func:`reflectivity_envelope_cost` computes the exact
    numbers for one geometry.
    """

    if not config.output_dbz:
        return ()
    g = config.grid
    rows: list[tuple[Any, ...]] = [
        ("dbz", "simulated 10 cm radar reflectivity", "dBZ", "mass",
         g.mass_shape, "reflectivity", "dbz", False),
        ("cref", "composite (column-maximum) reflectivity", "dBZ", "surface",
         g.surface_shape, "reflectivity", "cref", False),
    ]
    if config.dbz_level_m is not None:
        # Name the level the plane is actually taken at, not only the one the
        # configuration asked for.  misclibs.F:2454-2466 answers a request
        # with the lowest model level at or above it, which is generally not
        # the requested height, and a carrier labelled with the request alone
        # would be read as a plane at a height it is not at.
        level = config.reflectivity_level_index
        height = float(config.grid.levels.zh[level + 1])
        rows.append(
            ("dbz_level",
             f"reflectivity at model level {level + 1} (zh={height!r} m), the "
             f"lowest level at or above the requested {config.dbz_level_m!r} m",
             "dBZ", "surface", g.surface_shape, "reflectivity", "dbz_level", False)
        )
    return tuple(rows)


class DuplicateHistoryCarrierError(ValueError):
    """Two producers selected one carrier name.

    This is a defect in the carrier table, not a payload-size problem, so a
    door that reports it must not offer a smaller grid as the remedy.
    """

    def __init__(self, message: str, names: Sequence[str]) -> None:
        super().__init__(message)
        self.names = tuple(names)


def _carrier_table(
    config: RuntimeConfig, *, carrier_profile: str = COMPLETE_CARRIER_PROFILE
) -> tuple[CartesianHistoryCarrier, ...]:
    g = config.grid
    m, u, v, w = g.mass_shape, g.u_shape, g.v_shape, g.w_shape
    profile = config.profile
    # CM1 gives every scalar on its q3d axis a unit (``qunit(n)``), and the two
    # admitted tables do not share one: ptype=7 is six kg/kg rows, ptype=5 is
    # those six plus four kg-1 number concentrations (src/param.F:3455-3463,
    # :3667-3672).  One field-table row carries one unit, so the mass prefix
    # and the number suffix are published as two named carriers rather than as
    # one array whose declared unit would be wrong for four of its rows.
    # Both are contiguous leading-axis views of the same q3d allocation.
    mass_rows = len(profile.mass_names)
    number_rows = len(profile.number_names)
    rows = (
        ("closure.tkea", "subgrid turbulence kinetic energy", "m2 s-2", "w", w, "tke", "tkea", False),
        ("pp3d", "current perturbation nondimensional pressure", "nondimensional", "mass", m, "fields", "pp3d", False),
        ("prs", "current air pressure", "Pa", "mass", m, "fields", "prs", False),
        ("q3d", "current " + ",".join(profile.mass_names) + " mixing ratios", "kg kg-1", "mass-species", (mass_rows, *m), "fields", "q3d", False),
        *((
            ("nc3d", "current " + ",".join(profile.number_names) + " number concentrations", "kg-1", "mass-species", (number_rows, *m), "fields", "q3d", False),
        ) if number_rows else ()),
        ("rf", "current air density on w faces", "kg m-3", "w", w, "fields", "rf", False),
        ("rho", "current air density", "kg m-3", "mass", m, "fields", "rho", False),
        ("terrain_msl", "terrain height above mean sea level", "m", "surface", g.surface_shape, "metrics", "terrain_msl", False),
        # th3d is CM1's `tha`: the potential-temperature PERTURBATION.  CM1
        # forms total theta as th0+tha (src/solve1.F:893,897,1031) and this
        # runtime does the same (cuda_gm1_tilestream_wsm6.py:598).  The
        # meaning string is what consumers display -- rust/gm1-view labels the
        # field from it (rust/gm1-view/src/input.rs:499-518) -- so calling
        # this "current potential temperature" made a roughly +/-10 K
        # perturbation read as a roughly 300 K absolute field.  Select
        # output_basestate (this lane's default) to get th0 beside it, or
        # output_th for the absolute field under CM1's own name.
        ("th3d", "current potential-temperature perturbation (CM1 tha; total theta is th0+tha)",
         "K", "mass", m, "fields", "th3d", False),
        ("u3d", "current west-east velocity", "m s-1", "u", u, "fields", "u3d", True),
        ("v3d", "current south-north velocity", "m s-1", "v", v, "fields", "v3d", True),
        ("w3d", "current vertical velocity", "m s-1", "w", w, "fields", "w3d", True),
        ("zf", "model w-face height above mean sea level", "m", "w", w, "metrics", "zf", True),
    ) + _reflectivity_rows(config)
    # CM1 &param9 products, appended AFTER the fixed carriers so the arena
    # byte offsets of the twelve that were always published do not move.
    from .cartesian_environment import ENVIRONMENT_GROUP
    from .cuda_getcape import (
        ADIABAT_NAMES, GETCAPE_FIELD_META, GETCAPE_FIELD_SWITCHES, PARCEL_SOURCE_NAMES,
    )

    environment = config.environment_field_selection
    selection = getattr(config, "output", None)
    if selection is not None:
        # One row per name.  CM1 has one dbz and one cref, and two producers
        # reach them here: the reflectivity rows above, which also carry the
        # single-level dbz_level plane, and the &param9 rows below, which the
        # same physics.output_dbz selects.  Publishing both would put two
        # carriers under one name in the arena, and a reader would get
        # whichever was written last -- so the fixed rows win and &param9
        # contributes only what they do not already publish.
        published = {row[0] for row in rows}
        param9 = []
        for row in output_carrier_rows(
            selection, mass=m, w=w, u=u, v=v,
            surface=g.surface_shape, profile=(g.nz,),
        ):
            if row[0] in published:
                continue
            if row[0] in GETCAPE_FIELD_META:
                # cape/cin/lcl/lfc have exactly one producer: the &param9
                # engine (cartesian_output_products._getcape), whose kernel is
                # differenced against CM1's own compiled getcape.F.  It lifts
                # the parcel [diagnostics] names, so the description says
                # which one rather than assuming writeout.F:2921's.
                absent = {
                    "lcl": "; -1 where the parcel never saturated",
                    "lfc": "; -1 where the parcel never reached one",
                }.get(row[0], "")
                row = (
                    row[0],
                    f"{GETCAPE_FIELD_META[row[0]][0]} "
                    f"({GETCAPE_FIELD_SWITCHES[row[0]]}, "
                    f"{PARCEL_SOURCE_NAMES[environment.parcel_source]} parcel, "
                    f"{ADIABAT_NAMES[environment.adiabat]} adiabat{absent})",
                    *row[2:],
                )
            param9.append(row)
        rows = rows + tuple(param9)
    rows = rows + _budget_rows(config, mass=m, u=u, v=v, w=w)
    # CM1 writeout.F:2937-3028.  Each plane is appended only when its own
    # output_* switch is on, which is exactly writeout.F's own rule: a document
    # that declares no [diagnostics] table publishes the carriers above and
    # nothing else, byte for byte as it always has.  A runtime that carries an
    # &param9 selection has already published these names from the &param9
    # engine above (cartesian_door binds the four switches into it), so the
    # environment sweep contributes a plane only where no &param9 selection
    # exists; it never publishes a second carrier under a name already taken.
    published = {row[0] for row in rows}
    for name in environment.selected():
        if name in published:
            continue
        meaning, units = GETCAPE_FIELD_META[name]
        rows += (
            (
                name,
                f"{meaning} ({GETCAPE_FIELD_SWITCHES[name]}, "
                f"{environment.record()['parcel_source_name']} parcel)",
                units,
                "surface",
                g.surface_shape,
                ENVIRONMENT_GROUP,
                name,
                False,
            ),
        )
    if carrier_profile == COMPLETE_CARRIER_PROFILE:
        return tuple(CartesianHistoryCarrier(*row) for row in rows)
    require_history_carrier_profile(carrier_profile)
    # Each omitted field is replaced by the compact column its CM1
    # reconstruction reads, so the reduced frame still carries every word the
    # formula needs.  Both are one column; neither scales with nx or ny.
    carriers = [
        CartesianHistoryCarrier(*row) for row in rows
        if row[0] not in RECONSTRUCTIBLE_CARRIERS
    ]
    carriers.append(CartesianHistoryCarrier(
        "base.pi0", "base-state nondimensional pressure column", "nondimensional",
        "mass-column", (g.nz,), "base", "pi0", False,
    ))
    carriers.append(CartesianHistoryCarrier(
        "metrics.zf_profile",
        "model w-face height above mean sea level, one column broadcast over "
        "every horizontal plane",
        "m", "w-column", (g.nz + 1,), "metrics", "zf", True,
        horizontal_profile_column=True,
    ))
    return tuple(sorted(carriers, key=lambda carrier: carrier.name))


def _budget_rows(config: RuntimeConfig, *, mass: Any, u: Any, v: Any, w: Any) -> tuple[Any, ...]:
    """CM1 ``output_*budget`` terms and the GM1 vorticity budget, when selected.

    A run that selects neither publishes exactly the carriers it always did.
    """

    from .cm1_budget_terms import (
        budget_carrier_rows, facts_from_runtime, vorticity_carrier_rows,
    )

    selection = getattr(config, "output", None)
    lagrangian = getattr(config, "lagrangian", None)
    history_vorticity = bool(
        lagrangian is not None and lagrangian.values.get("vorticity_budget_history", False)
    )
    if selection is None and not history_vorticity:
        return ()
    facts = facts_from_runtime(config)
    rows: tuple[Any, ...] = ()
    if selection is not None:
        rows = budget_carrier_rows(selection, facts, mass=mass, u=u, v=v, w=w, group=OUTPUT_GROUP)
    if history_vorticity:
        rows = rows + vorticity_carrier_rows(facts, mass=mass, group=OUTPUT_GROUP)
    return rows


@dataclass(frozen=True, slots=True)
class CartesianHistoryProfile:
    """A priced, contract-admissible history frame for one runtime geometry."""

    profile_id: str
    geometry: Mapping[str, Any]
    carriers: tuple[CartesianHistoryCarrier, ...]
    payload_nbytes: int
    staging_words: int
    carrier_profile: str = COMPLETE_CARRIER_PROFILE

    @property
    def staging_bytes(self) -> int:
        return self.staging_words * 4

    @property
    def admission_host_bytes(self) -> int:
        """Peak host bytes a reader spends admitting this frame."""

        return forecast_admission_host_bytes(self.payload_nbytes)

    @property
    def reconstructible_carriers(self) -> Mapping[str, str]:
        if self.carrier_profile == COMPLETE_CARRIER_PROFILE:
            return MappingProxyType({})
        return RECONSTRUCTIBLE_CARRIERS

    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        """The ordered carrier-name to shape mapping the arena writer takes."""

        return MappingProxyType({carrier.name: carrier.shape for carrier in self.carriers})

    def staged_names(self) -> tuple[str, ...]:
        return tuple(carrier.name for carrier in self.carriers if carrier.staged)

    def record(self) -> dict[str, Any]:
        return {
            "schema": CARTESIAN_HISTORY_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "geometry": dict(self.geometry),
            "carriers": [carrier.record() for carrier in self.carriers],
            "carrier_names": [carrier.name for carrier in self.carriers],
            "payload_nbytes": self.payload_nbytes,
            "maximum_forecast_payload_bytes": MAX_FORECAST_PAYLOAD_BYTES,
            "admission_host_bytes": self.admission_host_bytes,
            "staging_allocation_bytes": self.staging_bytes,
            "staged_carriers": list(self.staged_names()),
            "carrier_profile": self.carrier_profile,
            "omitted_carriers": dict(OMITTED_CARRIERS),
            "reconstructible_carriers": dict(self.reconstructible_carriers),
            "reduced_profile_reader_cost": (
                None if self.carrier_profile == COMPLETE_CARRIER_PROFILE
                else REDUCED_PROFILE_READER_COST
            ),
            # CM1 &param9 products in this frame, by CM1's own writeout.F
            # name_output strings.  A consumer reads this rather than
            # subtracting the twelve prognostic carriers by hand.
            "selected_output_names": [
                carrier.name for carrier in self.carriers
                if carrier.group == OUTPUT_GROUP
            ],
            "artifact_magic": FORECAST_HISTORY_MAGIC,
        }


def reflectivity_envelope_cost(config: RuntimeConfig) -> dict[str, Any]:
    """What the CM1 ``output_dbz`` carriers add to one frame, in words.

    Stated per gridpoint because that is the unit the frame budget is quoted
    in: twelve carriers and sixteen float32 words per gridpoint without the
    diagnostic.
    """

    g = config.grid
    base = _carrier_table(_without_reflectivity(config))
    full = _carrier_table(config)
    points = g.nz * g.ny * g.nx
    base_words = sum(carrier.words for carrier in base)
    full_words = sum(carrier.words for carrier in full)
    return {
        "output_dbz": bool(config.output_dbz),
        "dbz_level_m": config.dbz_level_m,
        "carriers_without_reflectivity": len(base),
        "carriers_with_reflectivity": len(full),
        "words_per_gridpoint_without_reflectivity": base_words / points,
        "words_per_gridpoint_with_reflectivity": full_words / points,
        "added_words_per_gridpoint": (full_words - base_words) / points,
        "added_payload_nbytes": 4 * (full_words - base_words),
        "optional_per_run": True,
        "authority": "CM1 r21.1 README.namelist:1366-1369 output_dbz",
    }


def _without_reflectivity(config: RuntimeConfig) -> RuntimeConfig:
    from dataclasses import replace

    if not config.output_dbz:
        return config
    # physics.output_dbz is also the &param9 output_dbz word, which publishes
    # dbz and cref from the output selection.  Leaving it on here would keep
    # both carriers in the "without" frame and quote only the level plane as
    # the cost of reflectivity.
    selection = getattr(config, "output", None)
    if selection is not None and selection.on("output_dbz"):
        return replace(
            config, output_dbz=False, dbz_level_m=None,
            output=selection.with_values({"output_dbz": 0}),
        )
    return replace(config, output_dbz=False, dbz_level_m=None)


#: Why a model-time-zero frame of an ``output_dbz`` run carries no dBZ.
INITIAL_FRAME_REFLECTIVITY_OMITTED = (
    "cm1.F:3214-3237 computes dBZ inside the microphysics of the step whose "
    "state is written, and this runtime computes it only there (the scheme's "
    "own 10 cm operator, called with refl_10cm_due); the model-time-zero frame "
    "is taken before any large step, so the reflectivity carriers still hold "
    "their -1000 seed, and they are left out of this one frame rather than "
    "published as a field nothing computed"
)


def initial_frame_history_profile(
    config: RuntimeConfig, *, carrier_profile: str = COMPLETE_CARRIER_PROFILE,
    tornado_carriers: Sequence[Any] = (),
) -> "CartesianHistoryProfile":
    """The profile of an integration's model-time-zero frame (``--history-initial``).

    The integration's own profile, less the reflectivity carriers when the
    run declares ``output_dbz`` (:data:`INITIAL_FRAME_REFLECTIVITY_OMITTED`).
    Every other carrier sits at the same name, shape and meaning, so a reader
    that opens frame 0 and frame 1 finds the same fields but dbz and cref.
    """

    return cartesian_history_profile(
        _without_reflectivity(config), carrier_profile=carrier_profile,
        tornado_carriers=tornado_carriers,
    )


def forecast_admission_host_bytes(payload_nbytes: int) -> int:
    """Peak host bytes required to admit one payload of ``payload_nbytes``.

    Mirrors ``forecast_admission_host_bytes`` at
    rust/cm1-state/src/forecast.rs:830.  A reader holds the whole artifact and
    one float32 copy of every carrier at once, so this, not the payload alone,
    is what a publishing host must be able to give back to a reader.
    """

    if type(payload_nbytes) is not int or payload_nbytes < 0:
        raise ValueError("payload byte count must be a non-negative integer")
    return payload_nbytes * FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE


def require_history_carrier_profile(value: str) -> str:
    """Admit one of the two published carrier inventories, by name."""

    if value not in CARTESIAN_HISTORY_CARRIER_PROFILES:
        raise ValueError(
            f"history carrier profile {value!r} is not one of "
            f"{list(CARTESIAN_HISTORY_CARRIER_PROFILES)}; an unrecognized "
            "profile would publish a frame whose carrier inventory no reader "
            "in this distribution declares, and every reader would refuse it "
            "by name after the whole device-to-host transfer had been paid for"
        )
    return value


def cartesian_history_profile(
    config: RuntimeConfig, *, profile_id: str | None = None,
    carrier_profile: str = COMPLETE_CARRIER_PROFILE,
    tornado_carriers: Sequence[Any] = (),
) -> CartesianHistoryProfile:
    """Price one history frame and refuse a grid the host cannot admit.

    Nothing here touches CUDA.  The size decision is made before allocation
    so a caller never pays for a full device-to-host transfer that the Rust
    contract would then reject.

    ``tornado_carriers`` are the CM1 ``output_*`` diagnostic fields the
    document's ``[diagnostics]`` table selected, already priced by
    :func:`cm1core.cartesian_tornado_diagnostics.tornado_carrier_table`.  An
    empty sequence -- which is what an absent table gives -- leaves the
    twelve-carrier profile this lane has always published exactly as it was,
    down to the arena byte layout and the field table's SHA-256.
    """

    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    require_history_carrier_profile(carrier_profile)
    if profile_id is None:
        profile_id = (
            CARTESIAN_HISTORY_PROFILE_ID
            if carrier_profile == COMPLETE_CARRIER_PROFILE
            else CARTESIAN_HISTORY_REDUCED_PROFILE_ID
        )
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("history profile id must be a nonempty string")
    # One table, carrying the declared carrier profile.  The swath rows are
    # appended after it, so the byte offsets of the carriers a frame has
    # always published do not move when a diagnostic is selected.
    carriers = _carrier_table(config, carrier_profile=carrier_profile) + tuple(
        CartesianHistoryCarrier(
            row.name, row.meaning, row.units, row.stagger, tuple(row.shape),
            "tornado", row.key, False,
        )
        for row in tornado_carriers
    )
    names = [carrier.name for carrier in carriers]
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise DuplicateHistoryCarrierError(
            f"carrier name(s) {', '.join(repeated)} are published by more than "
            "one producer in this profile; two carriers under one name would "
            "overwrite each other in the arena and the reader would get "
            "whichever was written last",
            repeated,
        )
    g = config.grid
    for carrier in carriers:
        if carrier.nbytes > MAX_FORECAST_PAYLOAD_BYTES:
            raise ValueError(
                f"history carrier {carrier.name} alone needs {carrier.nbytes} bytes "
                f"at nx={g.nx}, ny={g.ny}, nz={g.nz}; one CM1FHS1 array is capped "
                f"at {MAX_FORECAST_PAYLOAD_BYTES} bytes, the ceiling that keeps a "
                "reader from allocating the artifact plus one float32 copy of it "
                "without bound, so Rust would refuse the frame after the whole "
                "transfer"
            )
    payload = forecast_arena_nbytes({carrier.name: carrier.shape for carrier in carriers})
    if payload > MAX_FORECAST_PAYLOAD_BYTES:
        # Name every carrier and its cost, and name the ones that would have
        # to go, in declared order, to fit.  A frame is not silently reduced:
        # this lane publishes the complete state of the component it ran or it
        # publishes nothing, because a consumer cannot tell a dropped species
        # from a species that was zero.
        running = 0
        dropped: list[str] = []
        for carrier in carriers:
            if running + carrier.nbytes <= MAX_FORECAST_PAYLOAD_BYTES:
                running += carrier.nbytes
            else:
                dropped.append(f"{carrier.name} ({carrier.nbytes} bytes)")
        inventory = ", ".join(
            f"{carrier.name}={carrier.nbytes}" for carrier in carriers
        )
        raise ValueError(
            f"history payload for component {config.microphysics} is {payload} "
            f"bytes at nx={g.nx}, ny={g.ny}, nz={g.nz}; the CM1FHS1 contract "
            f"caps one forecast payload at {MAX_FORECAST_PAYLOAD_BYTES} bytes, "
            f"so this grid has no publishable history frame in the existing "
            f"envelope format.  The declared carriers are {inventory}; fitting "
            f"the cap would require omitting {', '.join(dropped)}, and this "
            "lane will not publish a frame that silently drops a carrier.  "
            f"Publish the {RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE} profile "
            "instead, which omits only the carriers cm1-render rebuilds from "
            "the columns the frame still carries, or reduce grid.nx, grid.ny "
            "or grid.nz until one frame fits"
        )
    # One allocation, carved into disjoint slices, is reused by every frame:
    # the transfer then validates and streams the whole arena in one pass
    # without an aliased or lazily materialized carrier map.
    staging_words = sum(carrier.words for carrier in carriers if carrier.staged)
    return CartesianHistoryProfile(
        profile_id, MappingProxyType(dict(g.record())), carriers, payload,
        staging_words, carrier_profile,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb", buffering=0) as stream:
        stream.write(payload)
        os.fsync(stream.fileno())


def _regular_executable(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("the forecast publisher must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("the forecast publisher must be a nonempty regular file")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise ValueError("the forecast publisher is not executable")
    return resolved


def _run_checked(arguments: Sequence[str | os.PathLike[str]], label: str,
                 io: Any = None, cancel: Any = None) -> str:
    """Run one publisher command; with ``io``, also collect its stage timings.

    The timings arrive on stderr only when the environment asks for them, so
    stdout -- which receipts embed as ``rust_output`` -- is the same bytes
    either way.

    With ``cancel`` (a :class:`~cm1core.background_publication.PublicationCancel`)
    the publisher runs in its own session, so a terminal interrupt aimed at
    the run does not kill a checkpoint the run is still waiting to finish; the
    cancel token, or any exception on the waiting thread, kills it instead,
    and on Linux so does the end of the run's process, however it ends.
    """

    environment = None
    started = 0.0
    if io is not None:
        from .runtime_profiler import publish_timing_environment

        environment = publish_timing_environment()
        started = io.begin("publisher")
    command = [os.fspath(value) for value in arguments]
    if cancel is None:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, env=environment,
        )
    else:
        completed = _run_cancellable(command, environment, cancel)
    if io is not None:
        from .runtime_profiler import parse_publish_timing

        io.end("publisher", started)
        for stage, seconds, nbytes in parse_publish_timing(completed.stderr):
            io.add(f"publisher.{stage}", seconds, nbytes)
    if completed.returncode != 0:
        detail = _publisher_detail(completed.stderr) or completed.stdout.strip()
        raise RuntimeError(f"{label} failed with exit {completed.returncode}: {detail}")
    return completed.stdout.strip()


# Run in the publisher's own process before it execs: ask Linux for SIGKILL
# when the thread that started it dies (prctl PR_SET_PDEATHSIG = 1, SIGKILL =
# 9), and exit if that already happened.  The flag survives execve.
_PARENT_DEATH_EXEC = (
    "import ctypes, os, sys\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "if libc.prctl(1, 9, 0, 0, 0) != 0:\n"
    "    raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG, SIGKILL) failed')\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit('the run that started this publisher has already exited')\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)


def _tied_to_this_process(command: list[str]) -> list[str]:
    """``command`` behind a Linux exec wrapper that dies with this process.

    A publisher in its own session (``start_new_session``) no longer shares
    the run's process group, so a terminal hangup or ``kill -- -PGID`` that
    ends the run would leave it writing a multi-gibibyte envelope into staging
    and holding its payload in memory.  The parent-death signal ends it with
    the run however the run ends, SIGKILL included.  The wrapper replaces
    itself with the publisher (same pid, same session and process group, the
    same pipes), so killing the group and reading its output are unchanged.
    Elsewhere, or without an interpreter to run the wrapper, ``command`` as is.
    """

    import sys

    if not sys.platform.startswith("linux") or not sys.executable:
        return command
    return [
        sys.executable, "-I", "-S", "-c", _PARENT_DEATH_EXEC, str(os.getpid()), *command,
    ]


def _run_cancellable(command: list[str], environment: Any, cancel: Any) -> Any:
    """``subprocess.run`` with the same captured output, but stoppable.

    The publisher runs in its own session and is killed with this process
    (:func:`_tied_to_this_process`).
    """

    cancel.check()
    process = subprocess.Popen(
        _tied_to_this_process(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True,
    )
    cancel.attach(process)
    try:
        # A cancel from another thread kills the process, which ends this.
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.communicate()
        raise
    finally:
        cancel.detach(process)
    cancel.check()
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _publisher_detail(stderr: str) -> str:
    """The publisher's own error text, without any timing rows around it."""

    from .runtime_profiler import PUBLISH_TIMING_PREFIX

    return "\n".join(
        line for line in (stderr or "").splitlines()
        if not line.startswith(PUBLISH_TIMING_PREFIX + "\t")
    ).strip()


def active_io_event(runtime: Any) -> Any:
    """The I/O event a profiled runtime opened around this publication, if any."""

    return getattr(runtime, "active_io_event", None)


def require_forecast_publisher(publisher: Path | str) -> Path:
    """Admit exactly one executable that speaks this profile's field table."""

    executable = _regular_executable(Path(publisher))
    abi = _run_checked([executable, "--abi"], "forecast publisher ABI query")
    if abi != FORECAST_PUBLISH_ABI:
        raise ValueError(
            f"the forecast publisher reports ABI {abi!r}, not "
            f"{FORECAST_PUBLISH_ABI!r}; its field table columns are not the "
            "ones this profile emits"
        )
    return executable


def require_new_history_directory(directory: Path | str) -> Path:
    """Refuse an existing frame directory before any device work is queued."""

    directory = Path(directory)
    if directory.exists():
        raise ValueError(
            f"history frame directory {directory} already exists; publishing "
            "into it would mix two model times in one frame"
        )
    return directory


def require_cartesian_publication_boundary(
    runtime: Any, *, initial_condition: bool = False
) -> int:
    """Admit only the post-solve3 quiescent state of a completed large step.

    Anywhere else, the current and A carriers belong to different RK stages
    and the padded wind owners have not been refreshed from their physical
    rectangles, so the frame would mix two model times under one clock.

    ``initial_condition`` admits step zero, and only a caller that has said
    so in as many words gets it.  ``cm1.F:4341-4350``'s ``iconly`` builds the
    initial condition, writes it and stops, which is how a CM1 user looks at a
    sounding, a bubble and a grid before spending the GPU hours; the frame it
    produces is the initialized source state and the receipt says so, rather
    than presenting it under a completed-forecast clock.
    """

    clock, dynamics, finish = runtime.clock, runtime.dynamics, runtime.finish
    if clock.failed:
        raise RuntimeError(
            "a failed CUDA step leaves the device state undefined; there is no "
            "completed model time to publish a history frame at"
        )
    if not isinstance(initial_condition, bool):
        raise TypeError("initial_condition must be a bool")
    step = int(clock.completed_steps)
    if step < 1 and not initial_condition:
        raise RuntimeError(
            "no large step has completed; publishing here would emit the "
            "initialized source state under a completed-forecast clock. Pass "
            "initial_condition=True (cm1.F:4341-4350, iconly) to publish it as "
            "what it is"
        )
    if step > 0 and initial_condition:
        raise RuntimeError(
            f"initial_condition was declared at {step} completed large steps; the "
            "frame would be labelled an initial condition while carrying an "
            "integrated state"
        )
    if int(finish.completed_physics_step) != step or int(finish.completed_solve3_step) != step:
        raise RuntimeError(
            f"the runtime completed {step} steps but stands at physics step "
            f"{int(finish.completed_physics_step)} and solve3 step "
            f"{int(finish.completed_solve3_step)}; a frame taken here would mix "
            "pre- and post-physics carriers at one model time"
        )
    # cuda_cartesian_dynamics sets this token in sync_padded_from_physical,
    # the last device work of solve3.
    if dynamics._expected_phase != ("held", 1):
        raise RuntimeError(
            f"the dynamics cadence is at {dynamics._expected_phase!r}, not the "
            "post-solve3 quiescent boundary; the padded wind owners still hold "
            "the previous stage's words"
        )
    # cm1.F:3214-3237 raises getdbz on the step whose state is written, so a
    # published frame's reflectivity is always this step's.  A frame taken
    # after a step the flag was not raised on would carry the dBZ field of an
    # earlier model time beside this one's winds, and nothing in the artifact
    # would say the two disagree.
    if getattr(finish, "output_dbz", False):
        computed = int(getattr(finish, "completed_reflectivity_step", 0))
        if computed != step:
            raise RuntimeError(
                f"this run declares output_dbz and stands at completed step {step}, "
                f"but its reflectivity carriers were last written at step {computed}; "
                "publishing here would put one model time's dBZ field beside another "
                "model time's winds. Schedule the reflectivity output step "
                "(CudaCartesianRuntime.schedule_reflectivity) for every step a frame "
                "is published at, as cm1.F:3214-3237 does"
            )
    return step


def _horizontal_profile_column(dynamics: Any, carrier: CartesianHistoryCarrier, value: Any) -> Any:
    """Take one column of a broadcast source, only on the runtime's own proof.

    ``upload_immutable_source_metrics`` records ``kind == "horizontal-profile"``
    for a metric only after checking every horizontal plane of the source.
    Without that proof this column would be one arbitrary sample of a field
    that varies in x and y, and every height in the published frame would be
    wrong away from i=j=0.
    """

    proofs = getattr(getattr(dynamics, carrier.group, None), "proofs", None)
    proof = None if proofs is None else proofs.get(carrier.key)
    if proof is None or proof.get("kind") != "horizontal-profile":
        raise ValueError(
            f"history carrier {carrier.name} publishes one column of "
            f"{carrier.group}[{carrier.key!r}], but this runtime carries no "
            "horizontal-profile proof for it; the column would be one sample "
            "of a field that varies in x and y and would misplace every height "
            "in the frame away from the first column"
        )
    return value[:, 0, 0]


def _stage_carriers(runtime: Any, profile: CartesianHistoryProfile) -> Mapping[str, Any]:
    """Materialize strided and broadcast carriers into one budgeted buffer."""

    from .cartesian_environment import ENVIRONMENT_GROUP, compute_environment_planes

    cupy, registry = runtime.cp, runtime.registry
    dynamics = runtime.dynamics
    tornado = getattr(runtime, "tornado", None)
    # One mapping, not four.  Each branch that added a carrier group wrote its
    # own `groups = {...}`, and in a function body the last one wins -- so the
    # reflectivity, base-state, &param9 and environment groups were all being
    # dropped, and a frame that declared any of their carriers would have
    # refused by name at publication time.
    groups = {
        "fields": dynamics.fields,
        "tke": dynamics.tke,
        "metrics": dynamics.metrics,
        # CM1 output_dbz.  Empty for a run that declared it off, so a carrier
        # that names this group in such a run refuses by name below instead of
        # publishing an absent field.
        "reflectivity": getattr(runtime.finish, "reflectivity", {}),
        "base": getattr(dynamics, "base_profiles", {}),
        # CM1 &param9 products, materialised by the runtime's diagnostic
        # engine immediately before this call.  An empty mapping here with a
        # selection that asks for products is a defect, and the KeyError below
        # names the missing carrier rather than publishing a zero plane.
        OUTPUT_GROUP: getattr(runtime, "output_fields", {}),
        # writeout.F:2895: the column sweep runs only when a plane wants it,
        # and returns an empty mapping otherwise without allocating anything.
        # Only when this profile actually takes a plane from the sweep: with an
        # &param9 selection the four names come from the &param9 engine, and
        # running the sweep anyway would allocate and fill column scratch
        # nothing reads.
        ENVIRONMENT_GROUP: (
            compute_environment_planes(
                runtime, runtime.config.environment_field_selection
            )
            if any(carrier.group == ENVIRONMENT_GROUP for carrier in profile.carriers)
            else {}
        ),
        # misclibs.F's rotation diagnostics and solve3.F's run-long swaths.
        "tornado": {} if tornado is None else dict(tornado.published_carriers()),
    }
    staging = registry.allocate(STAGING_BUFFER_NAME, (profile.staging_words,))
    carriers: dict[str, Any] = {}
    offset = 0
    with runtime.stream:
        for carrier in profile.carriers:
            try:
                value = groups[carrier.group][carrier.key]
            except KeyError:
                raise ValueError(
                    f"history carrier {carrier.name} is not present as "
                    f"{carrier.group}[{carrier.key!r}]; this runtime cannot "
                    "publish the declared profile"
                ) from None
            # Two carriers may name the same runtime key when one CM1 q3d
            # allocation carries two units; each takes its own contiguous
            # leading-axis window, in declared order.  A carrier that already
            # spans its whole runtime array keeps that exact object, so a
            # single-moment frame publishes the words it always did.
            if (
                carrier.name in _PACKED_SCALAR_WINDOWS
                and value.ndim == 4
                and int(value.shape[0]) != int(carrier.shape[0])
            ):
                start, stop = _PACKED_SCALAR_WINDOWS[carrier.name](carrier, value)
                value = value[start:stop]
            if carrier.staged:
                # The arena gate runs on the staged *view*, which is float32
                # and correctly shaped by construction, so a wrong-dtype or
                # wrong-extent source would be silently cast or broadcast into
                # it and published as if it had passed.  Check the source.
                if value.dtype != cupy.dtype("float32"):
                    raise ValueError(
                        f"history carrier {carrier.name} is {value.dtype} in "
                        f"{carrier.group}[{carrier.key!r}]; staging would cast it "
                        "to float32 and publish a frame the source never held"
                    )
                # A column carrier's source is the whole broadcast field: one
                # horizontal plane per published level.  Checking the column
                # shape here would accept a source of any horizontal extent.
                expected = (
                    (*carrier.shape, *runtime.config.grid.surface_shape)
                    if carrier.horizontal_profile_column else carrier.shape
                )
                if tuple(value.shape) != tuple(expected):
                    raise ValueError(
                        f"history carrier {carrier.name} has extent "
                        f"{tuple(value.shape)} in {carrier.group}[{carrier.key!r}] "
                        f"but the profile declares {tuple(expected)}; staging "
                        "would broadcast it and publish a frame of the wrong grid"
                    )
                if carrier.horizontal_profile_column:
                    value = _horizontal_profile_column(dynamics, carrier, value)
                view = staging[offset:offset + carrier.words].reshape(carrier.shape)
                cupy.copyto(view, value)
                offset += carrier.words
                value = view
            carriers[carrier.name] = value
    if offset != profile.staging_words:
        raise RuntimeError(
            f"history staging used {offset} of {profile.staging_words} budgeted "
            "words; the profile and the staged carriers disagree"
        )
    return MappingProxyType(carriers)


def _field_table_bytes(
    profile: CartesianHistoryProfile, arena: ForecastArenaWrite, *,
    absolute_arena_path: bool = False,
) -> bytes:
    """The eight-column field table of one frame.

    Column 7 names the arena.  The published table carries the arena's own
    file name, relative to the frame directory that holds the table: a frame
    is written under ``.incomplete/`` and renamed into place, and it may be
    copied to another machine, so an absolute path written at staging time
    names a file that no longer exists once the frame is published
    (gm1-history-grads-export resolves the name against the frame directory).
    ``absolute_arena_path`` is for ``cm1-forecast-publish`` alone, whose ABI
    v3 admits only an absolute raw path; that table is handed to the
    publisher and never published.
    """

    arena_name = (
        os.fspath(arena.path.resolve()) if absolute_arena_path else arena.path.name
    )
    rows = []
    for carrier in profile.carriers:
        columns = (
            carrier.name, carrier.meaning, carrier.units, carrier.stagger,
            ",".join(str(extent) for extent in carrier.shape),
            ",".join(str(bound) for bound in carrier.bounds()),
            arena_name,
            str(arena.carrier_offset_bytes[carrier.name]),
        )
        if any("\t" in column or "\n" in column for column in columns):
            raise ValueError(
                f"history carrier {carrier.name} carries a tab or newline; the "
                "Rust field table is strictly tab-separated and would split the row"
            )
        rows.append("\t".join(columns))
    return ("\n".join(rows) + "\n").encode("ascii")


# writeout.F:9614-9648 writes the four together under the imove guard, and
# this is the order the envelope carries them in.
_DOMAIN_LOCATION_SCALAR_NAMES = (
    DOMAIN_LOCATION_X_NAME, DOMAIN_LOCATION_Y_NAME,
    MOVE_SPEED_X_NAME, MOVE_SPEED_Y_NAME,
)


def _scalar_table_bytes(runtime: Any) -> bytes:
    """CM1's per-history-time moving-frame scalars, in the publisher's table.

    ``writeout_nc.F:281-291`` and ``writeout_nc.F:559-567`` write ``umove``
    and ``vmove`` once per history time; ``writeout.F:9614-9648`` writes the
    same two beside ``domainlocx`` and ``domainlocy``.  All four travel here
    under CM1's own names, and the domain location travels as binary64 bits
    because ``cm1.F:266`` declares it double precision and a decimal round
    trip through the command line would not be the same number.

    Rust re-derives ``cm1.F:3512-3513`` from the frame's own step and dt and
    refuses the table if the value it is handed is not what that integral
    gives, so a drifted or reset accumulator cannot reach an artifact.
    """

    clock = runtime.clock
    rows = (
        (DOMAIN_LOCATION_X_NAME, float(clock.domainlocx_m)),
        (DOMAIN_LOCATION_Y_NAME, float(clock.domainlocy_m)),
        (MOVE_SPEED_X_NAME, float(cm1_real(runtime.config.u_move_m_s))),
        (MOVE_SPEED_Y_NAME, float(cm1_real(runtime.config.v_move_m_s))),
    )
    return "".join(
        f"{name}\tf64-bits\t{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}\n"
        for name, value in rows
    ).encode("ascii")


def domain_location_record(runtime: Any) -> dict[str, Any]:
    """CM1's moving-frame block for one published frame.

    Every wind carrier in the arena is frame-relative and every x/y in the
    field table is a frame coordinate.  ``umove``/``vmove`` map the winds back
    (``init3d.F:32-33``); ``domainlocx``/``domainlocy`` are where the domain
    itself is, in ground metres, at this frame's model time -- the running
    integral of ``cm1.F:3512-3513``, carried by the runtime and published
    here, not a product a reader is left to form.
    """

    # The clock owns the values, their CM1 names, units and descriptions.
    # What a published frame adds is what the envelope did with them.
    return {
        **runtime.clock.domain_location_record(),
        "ground_relative_wind": (
            f"{MOVE_SPEED_X_NAME} + u, {MOVE_SPEED_Y_NAME} + v (init3d.F:32-33)"
        ),
        "envelope_scalars": list(_DOMAIN_LOCATION_SCALAR_NAMES),
    }


def resume_clock_from_publication(
    config: RuntimeConfig, receipt: Mapping[str, Any] | Path | str
) -> RuntimeClock:
    """Continue this lane's domain-location integral from a published frame.

    ``restart_write.F:399-400`` writes ``domainlocx``/``domainlocy`` into CM1's
    restart file and ``restart_read.F:419-420`` reads them straight back into
    the ``intent(inout)`` pair, so a resumed CM1 run continues the integral.
    The Cartesian lane restores no device state, so this restores the part it
    does persist: the clock.  An integral that restarted at zero would
    mislocate every frame after the resume by the whole distance the domain
    had already travelled.

    The persisted frame velocity is checked against the one being resumed
    under, because a value accumulated at one ``umove`` is not the same
    integral at another and continuing it would place every later frame on a
    patch of ground the carriers never covered.
    """

    if isinstance(receipt, (str, Path)):
        receipt = json.loads(Path(receipt).read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping or a path to one")
    if receipt.get("schema") != CARTESIAN_HISTORY_PUBLICATION_SCHEMA:
        raise ValueError(
            f"a {CARTESIAN_HISTORY_PUBLICATION_SCHEMA} receipt is the only artifact "
            "this lane persists the domain-location integral in; nothing else names "
            "the completed step count the integral belongs to"
        )
    frame = receipt.get("moving_frame")
    if not isinstance(frame, Mapping):
        raise ValueError(
            "the receipt carries no moving_frame block, so it predates the "
            "cm1.F:3512-3513 domain-location integral; resuming from it would "
            "restart the integral at zero and mislocate every later frame"
        )
    missing = [
        name for name in
        (DOMAIN_LOCATION_X_NAME, DOMAIN_LOCATION_Y_NAME, MOVE_SPEED_X_NAME, MOVE_SPEED_Y_NAME)
        if name not in frame
    ]
    if missing:
        raise ValueError(
            f"the receipt's moving_frame block carries no {', '.join(missing)}; "
            "writeout.F:9614-9648 writes the four together and the integral cannot "
            "be continued from a partial set"
        )
    for name, declared in (
        (MOVE_SPEED_X_NAME, cm1_real(config.u_move_m_s)),
        (MOVE_SPEED_Y_NAME, cm1_real(config.v_move_m_s)),
    ):
        if float(frame[name]) != declared:
            raise ValueError(
                f"the receipt was written at {name}={float(frame[name])!r} m/s and "
                f"this run declares {declared!r} m/s; cm1.F:3512 integrates one "
                "constant frame velocity, so continuing the persisted value under a "
                "different one would place every later frame on ground the carriers "
                "never covered"
            )
    return RuntimeClock.resume(
        config,
        completed_steps=int(receipt["step"]),
        domainlocx_m=float(frame[DOMAIN_LOCATION_X_NAME]),
        domainlocy_m=float(frame[DOMAIN_LOCATION_Y_NAME]),
    )


@dataclass(frozen=True, slots=True)
class CartesianHistorySnapshot:
    """One frame's words and every runtime value its receipt reads, off the device.

    Everything :func:`commit_cartesian_history` needs, and nothing it would
    have to ask the runtime for: by the time a background commit runs the
    runtime has stepped on, so a clock, a config record or a diagnostic read
    then would describe a later model time than the words beside it.
    """

    directory: Path
    executable: Path | None
    profile: CartesianHistoryProfile
    arena: ForecastArenaSnapshot
    scalars: bytes
    step: int
    model_time_s: float
    dt_s: float
    config_record: Mapping[str, Any]
    execution_identity: str
    moving_frame: Mapping[str, Any]
    lower_boundary: Mapping[str, Any]
    surface: Mapping[str, Any]
    tornado_diagnostics: Mapping[str, Any] | None
    reflectivity: Mapping[str, Any] | None
    initial_condition: bool


def snapshot_cartesian_history(
    runtime: Any, directory: Path | str, *, publisher: Path | str | None = None,
    profile: CartesianHistoryProfile | None = None,
    carrier_profile: str = COMPLETE_CARRIER_PROFILE,
    initial_condition: bool = False, host: HostArena | None = None,
) -> CartesianHistorySnapshot:
    """The device half of a frame: every check, the staging and one host copy.

    Runs where the device state is quiescent -- the stepping thread, at the
    post-solve3 boundary.  With a ``host`` arena it leaves nothing on the
    device: the staging buffer and the environment planes are released before
    it returns, exactly as a completed publication releases them.  Without
    one nothing is copied: the staged carriers stay on the device for
    :func:`commit_cartesian_history` to stream on this thread at once, and it
    releases them after the arena write (or ``snapshot.arena.release()`` does,
    if the frame is abandoned first).  No directory is created.
    """

    step = require_cartesian_publication_boundary(
        runtime, initial_condition=initial_condition
    )
    require_history_carrier_profile(carrier_profile)
    tornado = getattr(runtime, "tornado", None)
    if tornado is not None:
        # The rotation fields are instantaneous, so they are recomputed for
        # this frame's own state rather than carried from the last step that
        # happened to touch them.
        io = active_io_event(runtime)
        tick = None if io is None else io.begin("diagnostics.tornado_refresh")
        tornado.refresh(step)
        if io is not None:
            io.end("diagnostics.tornado_refresh", tick)
    if profile is None:
        profile = cartesian_history_profile(
            runtime.config,
            carrier_profile=carrier_profile,
            tornado_carriers=() if tornado is None else tornado.carrier_table(),
        )
    if profile.carrier_profile != carrier_profile:
        raise ValueError(
            f"the supplied history profile publishes the {profile.carrier_profile!r} "
            f"carrier inventory, not the requested {carrier_profile!r}; the receipt "
            "would name an inventory the frame does not carry"
        )
    if profile.geometry != runtime.config.grid.record():
        raise ValueError(
            "the supplied history profile prices another geometry than this "
            "runtime; its carrier shapes would misdescribe every published word"
        )
    directory = require_new_history_directory(directory)
    io = active_io_event(runtime)
    tick = None if io is None else io.begin("publisher_admission")
    executable = None if publisher is None else require_forecast_publisher(publisher)
    if io is not None:
        io.end("publisher_admission", tick)
    config_record = runtime.config.record()
    carriers: Mapping[str, Any] = {}
    words: ForecastArenaSnapshot | None = None
    try:
        tick = None if io is None else io.begin("device_staging_enqueue")
        carriers = _stage_carriers(runtime, profile)
        if io is not None:
            io.end("device_staging_enqueue", tick)
            tick = io.begin("device_staging_sync")
        runtime.stream.synchronize()
        if io is not None:
            io.end("device_staging_sync", tick)
            io.set_bytes(profile.payload_nbytes)
        # Without a host arena the staged carriers stay on the device until
        # the commit that follows at once streams them, and are released then.
        words = snapshot_forecast_arena(
            carriers, shapes=profile.shapes(), cupy=runtime.cp, host=host, io=io,
            release=lambda: _release_frame_staging(runtime),
        )
    finally:
        carriers = {}
        if words is None or not words.on_device:
            _release_frame_staging(runtime)
    try:
        return _history_snapshot(
            runtime, directory, executable=executable, profile=profile, words=words,
            step=step, config_record=config_record, tornado=tornado,
            initial_condition=initial_condition,
        )
    except BaseException:
        words.release()
        raise


def _release_frame_staging(runtime: Any) -> None:
    """Free one frame's device staging, whether it was published or refused."""

    # The staged views die with the registry reference they alias, and a
    # refused frame must not strand the staging budget on the device.
    if STAGING_BUFFER_NAME in runtime.registry.buffers:
        runtime.registry.release([STAGING_BUFFER_NAME])
    # The getcape column scratch and its four planes are per-frame for the
    # same reason: a refused frame must not strand them either.
    from .cartesian_environment import release_environment_planes

    release_environment_planes(runtime)


def _history_snapshot(
    runtime: Any, directory: Path, *, executable: Path | None,
    profile: CartesianHistoryProfile, words: ForecastArenaSnapshot, step: int,
    config_record: Mapping[str, Any], tornado: Any, initial_condition: bool,
) -> CartesianHistorySnapshot:
    config = runtime.config
    reflectivity = None
    # README.namelist:1366-1369.  A run without the diagnostic writes exactly
    # the receipt it has always written; a run with it says where the numbers
    # came from, because a dBZ field is only meaningful beside the scheme
    # whose size distributions produced it.
    published_reflectivity = [
        carrier.name for carrier in profile.carriers
        if carrier.group == "reflectivity"
        or (carrier.group == OUTPUT_GROUP and carrier.name in ("dbz", "cref"))
    ]
    if config.output_dbz and initial_condition and not published_reflectivity:
        # initial_frame_history_profile left dbz and cref out of this frame.
        reflectivity = {
            "output_dbz": True,
            "carriers": [],
            "omitted_at_model_time_zero": INITIAL_FRAME_REFLECTIVITY_OMITTED,
            "authority": REFLECTIVITY_AUTHORITY,
        }
    elif config.output_dbz:
        level = config.reflectivity_level_index
        reflectivity = {
            "output_dbz": True,
            "carriers": [
                carrier.name for carrier in profile.carriers
                if carrier.group == "reflectivity"
            ],
            "computed_at_step": int(
                getattr(runtime.finish, "completed_reflectivity_step", 0)
            ),
            "level_index": level,
            "level_m": config.dbz_level_m,
            "level_height_m": (
                None if level is None
                else float(config.grid.levels.zh[level + 1])
            ),
            "composite_seed_dbz": CM1_CREF_SEED_DBZ,
            "envelope_cost": reflectivity_envelope_cost(config),
            "authority": REFLECTIVITY_AUTHORITY,
        }
    return CartesianHistorySnapshot(
        directory=directory,
        executable=executable,
        profile=profile,
        arena=words,
        scalars=_scalar_table_bytes(runtime),
        step=step,
        model_time_s=float(runtime.clock.model_seconds),
        dt_s=float(config.dt_s),
        config_record=config_record,
        execution_identity=str(runtime.runtime_source_identity["aggregate_sha256"]),
        # CM1 base.F:2663-2669 and cm1.F:3512-3513.  Every wind carrier in the
        # arena is frame-relative and every x/y in the field table is a frame
        # coordinate; this block is the whole of what maps them back, and it
        # carries the running domain-location integral rather than asking the
        # reader to form one.
        moving_frame=domain_location_record(runtime),
        # CM1 README.namelist:574-580.  A frame with bbc=1 exerts no surface
        # stress at all (src/turb.F:4536-4545), so any near-ground rotation
        # in this frame was produced without surface friction.  A reader
        # cannot tell that from the fields, so the frame says it.
        lower_boundary={
            "bbc": int(config.bbc),
            "sfcmodel": int(config.sfcmodel),
            "isfcflx": int(config.isfcflx),
            "cecd": int(config.cecd),
            "cnstcd": float(config.cnstcd),
            "cnstce": float(config.cnstce),
            "set_znt": int(config.set_znt),
            "cnst_znt": float(config.cnst_znt),
            "set_ust": int(config.set_ust),
            "cnst_ust": float(config.cnst_ust),
            "label": {1: "free slip", 2: "no slip", 3: "semi slip"}[
                int(config.bbc)
            ],
            "surface_stress": bool(config.bbc != 1),
            # src/turb.F:645-649 and src/turbtend.F:595.  Without this the
            # lowest model level's heat and moisture tendency has no surface
            # term at all, and a reader of the fields cannot tell.
            "surface_enthalpy_flux": bool(
                int(config.isfcflx) == 1
                and int(config.sfcmodel) >= 1
            ),
            "authority": "CM1 r21.1 turb.F:4515-4702 sfcstress",
        },
        # CM1 README.namelist:971-1015.  The initial surface a run integrated,
        # and -- loudly -- what about it was measured and what was not.  A
        # selectable capability with no evidence marker reads as a measured
        # one, which is the failure this block exists to prevent.
        surface={
            "initsfc": int(config.initsfc),
            "tsk0_k": float(config.tsk0_k),
            "tmn0_k": float(config.tmn0_k),
            "xland0": float(config.xland0),
            "lu0": int(config.lu0),
            "season": int(config.season),
            "oceanmodel": int(config.oceanmodel),
            "authority": "CM1 r21.1 init_surface.F:40-343",
            **surface_validation_markers(config.surface_drag_selection),
        },
        # CM1's output_* diagnostics for this run, when the document selected
        # any.  A frame from a run that declared no [diagnostics] table
        # carries no such key, exactly as every frame before them did.
        tornado_diagnostics=None if tornado is None else tornado.record(),
        reflectivity=reflectivity,
        initial_condition=bool(initial_condition),
    )


def commit_cartesian_history(
    snapshot: CartesianHistorySnapshot, *, io: Any = None, cancel: Any = None,
) -> dict[str, Any]:
    """The host half of a frame: the arena file, the tables, Rust, the receipt.

    Reads nothing but ``snapshot``, so it may run on a worker thread while
    the runtime steps on.  The files are the files the one-call publication
    has always written.
    """

    directory = snapshot.directory
    profile = snapshot.profile
    executable = snapshot.executable
    created_directory: Path | None = None
    try:
        directory.mkdir(parents=True)
        created_directory = directory
        arena = write_forecast_arena_snapshot_new(
            directory / ARENA_FILENAME, snapshot.arena, io=io, fsync=True, cancel=cancel,
        )
    except BaseException:
        # An empty frame directory would survive the refusal and then block
        # every retry at the same path, because the writer demands a new one.
        if created_directory is not None:
            try:
                next(created_directory.iterdir())
            except StopIteration:
                created_directory.rmdir()
            except OSError:
                pass
        raise
    finally:
        # Words still on the device are needed by nothing after the arena.
        snapshot.arena.release()
    tick = None if io is None else io.begin("tables")
    table = _field_table_bytes(profile, arena)
    table_path = directory / FIELD_TABLE_FILENAME
    _write_new_fsynced(table_path, table)
    scalars = snapshot.scalars
    scalar_path = directory / SCALAR_TABLE_FILENAME
    _write_new_fsynced(scalar_path, scalars)
    if io is not None:
        io.end("tables", tick, len(table) + len(scalars))
    model_time_s = snapshot.model_time_s
    dt_s = snapshot.dt_s
    config_record = dict(snapshot.config_record)
    step = snapshot.step
    initial_condition = snapshot.initial_condition
    envelope: dict[str, Any] = {
        "published": False,
        "reason": (
            "no cm1-forecast-publish executable was supplied; the raw arena, "
            "field table and scheduler-scalar table are the published artifacts "
            "of this frame"
        ),
    }
    if executable is not None:
        publisher_table = directory / PUBLISHER_FIELD_TABLE_FILENAME
        _write_new_fsynced(
            publisher_table, _field_table_bytes(profile, arena, absolute_arena_path=True),
        )
        try:
            output = _run_checked([
                executable, "history", directory / ENVELOPE_FILENAME, profile.profile_id,
                _canonical_sha256(config_record), snapshot.execution_identity,
                str(step), repr(model_time_s), repr(dt_s), publisher_table, scalar_path,
            ], "forecast history publication", io=io, cancel=cancel)
        finally:
            publisher_table.unlink(missing_ok=True)
        envelope = {
            "published": True,
            "abi": FORECAST_PUBLISH_ABI,
            "path": os.fspath(directory / ENVELOPE_FILENAME),
            "rust_output": output,
        }
    receipt = {
        "schema": CARTESIAN_HISTORY_PUBLICATION_SCHEMA,
        "status": CARTESIAN_HISTORY_STATUS,
        "directory": os.fspath(directory),
        "step": step,
        "model_time_s": model_time_s,
        "dt_s": dt_s,
        "config": config_record,
        "config_sha256": _canonical_sha256(config_record),
        "moving_frame": dict(snapshot.moving_frame),
        "lower_boundary": dict(snapshot.lower_boundary),
        "surface": dict(snapshot.surface),
        "execution_identity_sha256": snapshot.execution_identity,
        "profile": profile.record(),
        "arena": arena.record(),
        **(
            {} if snapshot.tornado_diagnostics is None
            else {"tornado_diagnostics": snapshot.tornado_diagnostics}
        ),
        "field_table_path": os.fspath(table_path),
        "field_table_sha256": hashlib.sha256(table).hexdigest(),
        "scalar_table_path": os.fspath(scalar_path),
        "scalar_table_sha256": hashlib.sha256(scalars).hexdigest(),
        "quiescent_boundary": (
            "initialized source state before any large step (cm1.F:4341-4350, iconly)"
            if initial_condition else "post-solve3 completed large step"
        ),
        "initial_condition": bool(initial_condition),
        "forecast_envelope": envelope,
        # A history frame is observational and carries twelve carriers chosen
        # for a viewer.  cm1core.cartesian_restart writes the artifact a run
        # is continued from, and its receipt says restore_supported true.
        "restore_supported": False,
        "claim_limit": (
            (
                CARTESIAN_HISTORY_CLAIM_LIMIT
                if executable is not None
                else CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT
            )
            + (
                "" if profile.carrier_profile == COMPLETE_CARRIER_PROFILE
                else (
                    f"; this frame omits {sorted(RECONSTRUCTIBLE_CARRIERS)}, which a "
                    "reader rebuilds by the formulas recorded in "
                    "profile.reconstructible_carriers, and which "
                    "cm1_state::reduced_history is the shipped reader for"
                )
            )
        ),
        "finite_checked_by_rust": executable is not None,
    }
    if snapshot.reflectivity is not None:
        receipt["reflectivity"] = dict(snapshot.reflectivity)
    tick = None if io is None else io.begin("receipt")
    _write_new_fsynced(
        directory / RECEIPT_FILENAME,
        (json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if io is not None:
        io.end("receipt", tick)
    return receipt


def publish_cartesian_history(
    runtime: Any, directory: Path | str, *, publisher: Path | str | None = None,
    profile: CartesianHistoryProfile | None = None,
    carrier_profile: str = COMPLETE_CARRIER_PROFILE,
    initial_condition: bool = False,
) -> dict[str, Any]:
    """Write one history frame from the live device state into a new directory.

    The frame is the raw arena plus the semantic field table.  When an
    explicit ``cm1-forecast-publish`` executable is supplied, the CM1FHS1
    envelope is produced from those exact bytes; without one the raw pair is
    the published artifact and the receipt says so.  This artifact is never
    read back into a runtime: ``cartesian_restart`` writes the one that is.

    This is :func:`snapshot_cartesian_history` followed at once by
    :func:`commit_cartesian_history` on the calling thread.
    """

    snapshot = snapshot_cartesian_history(
        runtime, directory, publisher=publisher, profile=profile,
        carrier_profile=carrier_profile, initial_condition=initial_condition,
    )
    try:
        return commit_cartesian_history(snapshot, io=active_io_event(runtime))
    finally:
        snapshot.arena.release()


__all__ = (
    "CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT",
    "CARTESIAN_HISTORY_CARRIER_PROFILES",
    "CARTESIAN_HISTORY_CLAIM_LIMIT",
    "CARTESIAN_HISTORY_PROFILE_ID",
    "CARTESIAN_HISTORY_PROFILE_SCHEMA",
    "CARTESIAN_HISTORY_PUBLICATION_SCHEMA",
    "CARTESIAN_HISTORY_REDUCED_PROFILE_ID",
    "CARTESIAN_HISTORY_STATUS",
    "COMPLETE_CARRIER_PROFILE",
    "FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE",
    "MAX_FORECAST_PAYLOAD_BYTES",
    "OMITTED_CARRIERS",
    "RECONSTRUCTIBLE_CARRIERS",
    "RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE",
    "REDUCED_PROFILE_READER_COST",
    "SCALAR_TABLE_FILENAME",
    "CartesianHistoryCarrier",
    "CartesianHistoryProfile",
    "CartesianHistorySnapshot",
    "cartesian_history_profile",
    "commit_cartesian_history",
    "domain_location_record",
    "forecast_admission_host_bytes",
    "INITIAL_FRAME_REFLECTIVITY_OMITTED",
    "initial_frame_history_profile",
    "publish_cartesian_history",
    "reflectivity_envelope_cost",
    "require_cartesian_publication_boundary",
    "require_forecast_publisher",
    "require_history_carrier_profile",
    "require_new_history_directory",
    "resume_clock_from_publication",
    "snapshot_cartesian_history",
)
