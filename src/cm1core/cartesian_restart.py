"""CM1's restart pair for the Cartesian CUDA lane: write one, continue from it.

CM1 r21.1 carries a simulation across a stop with two subroutines.
``src/restart_write.F`` writes every array and scalar the model carries from
one timestep to the next -- README.restart says so in as many words -- and
``src/restart_read.F``, called from ``src/cm1.F:2666`` under ``irst.eq.1``,
reads the identical inventory straight back into the same arrays.  Nothing in
between is recomputed.  That is the whole design, and this module is that
design for this lane.

What CM1 writes, and where this writes the same thing
----------------------------------------------------

* The prognostic volumes.  ``restart_write.F:767-795`` writes ``rho``,
  ``prs``, ``ua``, ``va``, ``wa``, ``ppi``, ``tha`` and ``ppx``;
  ``restart_write.F:809-819`` writes every moisture species ``qa(n)``;
  ``restart_write.F:839`` writes ``tkea`` under ``idoles .and. iusetke``,
  which is this lane's ``sgsmodel=1`` prognostic-TKE closure.
  ``restart_write.F:820-837`` writes the ``eqtset=2`` rate carriers
  ``qpten``/``qtten``/``qvten``/``qcten``.
* The surface and rain accumulators.  ``restart_write.F:692-757`` writes
  ``rain``, ``sws``, ``svs``, ``sps``, ``srs``, ``sgs``, ``sus`` and ``shs``,
  and under ``nrain = 2`` (a moving domain) the translated ``sws2``..``shs2``.
  The ArWen WSM6 seam owns this lane's rain equivalents and names them
  itself: :data:`~cm1core.arwen_wsm6_component.WSM6_RESTART_REQUIRED`.  The
  swaths a run keeps travel in the ``swath`` group
  (:class:`cm1core.tornado_diagnostics.SwathRestartLayout`): each selected
  plane, its ground-anchored canvas with the canvas's origin and extent, and
  the per-step series the tornado report reads.  They are opened with the
  run's length, after the restore, so the restored words wait on the runtime
  (``restored_swaths``) until then; see "Across a restart" below.
* The radiative lateral-boundary history ``radbcw``/``radbce``/``radbcs``/
  ``radbcn``, which this lane's open laterals carry from step to step.
* The scalars.  ``restart_write.F:364`` writes ``nstep``;
  ``restart_write.F:403`` writes ``mtime``; ``restart_write.F:378`` and
  ``restart_write.F:383`` write the output and restart-file counters ``nwrite``
  and ``nrst``; ``restart_write.F:412`` writes the dry-mass reference
  ``mass1``; ``restart_write.F:399-400`` write ``domainlocx``/``domainlocy``
  beside ``umove``/``vmove`` at ``restart_write.F:393-398``;
  ``restart_write.F:656-658`` write the ``qbudget``/``asq``/``bsq`` budget
  accumulators.

What CM1 does not write, and this does not either
-------------------------------------------------

README.namelist:1512-1516 records that the base-state fields in CM1's restart
files -- ``th0``, ``prs0``, ``pi0``, ``rho0``, ``qv0``, ``u0``, ``v0``, ``zs``,
``zh``, ``zf`` -- exist "for specialized applications only" and that "None of
the following are required for bit-identical restarts".  They are rebuilt from
the same sounding and geometry, and this port rebuilds them the same way: the
restart receipt pins the sounding by SHA-256 and the initializer by identity,
so the rebuild is the same rebuild or the resume is refused by name.

Across a restart: the swath group and the envelope version
----------------------------------------------------------

Restart profile ``gm1.cartesian-restart/v2`` adds the ``swath`` group.  Its
shapes follow the length of the run that wrote it (the ground canvas and the
per-step series are sized from it), so a resume compares the *state* part of
the profile -- every other carrier, exactly the v1 profile -- and applies the
swaths by their recorded layout: a resume planned longer or shorter than the
run that wrote the restart re-places the canvas by its origin.  A v1 restart
(written before the swaths were carried) still resumes; a swath the run keeps
and the restart does not carry starts at the restart step, and the resume
says so ("swaths restart here").

What is deliberately not carried, and why that is sound
------------------------------------------------------

Phase scratch.  The acoustic workspace, the moist stage bridge, the scalar RK
workspace and the padded non-persistent arrays are allocated inside a large
step and released inside it (``cuda_cartesian_dynamics._ensure_bridge`` and
``_drop_bridge`` allocate zeroed and release per stage;
``release_phase_workspace`` and ``_drop_padded`` release the rest), and
``sync_padded_from_physical`` resets the moist-stage and scalar-RK stage state
through ``begin_cuda_moist_stage`` and ``begin_cuda_scalar_moist_rk_step`` as
the last device work of solve3.  So at the boundary a restart is taken at,
none of it holds a value the next step reads before writing.

That last sentence is the one claim here that host reasoning cannot finish,
because it is a property of the CUDA kernels rather than of this module.  The
evidence for it is the lane's existing run-to-run bit reproducibility: two
independent processes, with independent CUDA contexts and independent memory
pools, produced bit-identical float payloads in every frame of a 120-step
run.  Uninitialized scratch that reached a result would have to be identical
across two driver allocations for that to hold.
``tests/test_cuda_cartesian_restart.py`` is the direct test, and it is
GPU-gated.

The artifact
------------

One ``CM1FCP1`` forecast checkpoint envelope written by the same Rust boundary
that writes this lane's history frames, through the same two tables.  Rust
validates the complete byte inventory, re-derives ``cm1.F:3512-3513`` for the
frame's own clock, and publishes atomically; ``cm1-forecast-publish restore``
reads it back into the identical arena and tables, so the bytes that reach the
device on a resume are the bytes the envelope holds and not a re-encoding of
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from .arwen_wsm6_component import (
    WSM6_RESTART_REQUIRED,
    WSM6_SURFACE_SCRATCH,
)
from .column_components import column_component_spec
from .cartesian_history import (
    FORECAST_PUBLISH_ABI,
    MAX_FORECAST_PAYLOAD_BYTES,
    require_forecast_publisher,
    require_new_history_directory,
)
from .cuda_cartesian_dynamics import cartesian_dynamics_restart_state
from .cuda_memory_inventory import shape_bytes
from .forecast_arena import (
    FORECAST_ARENA_TRANSFER_CHUNK_WORDS,
    ForecastArenaSnapshot,
    ForecastArenaWrite,
    HostArena,
    forecast_arena_nbytes,
    snapshot_forecast_arena,
    write_forecast_arena_snapshot_new,
)
from .cartesian_adaptive_frame import ADAPTIVE_MOVE_RESTART_PREFIX
from .tornado_diagnostics import (
    SERIES_NAMES,
    SWATH_RESTART_CANVAS,
    SWATH_RESTART_FIRST,
    SWATH_RESTART_GROUP,
    SWATH_RESTART_INDEX,
    SWATH_RESTART_LOCATION,
    SWATH_RESTART_OPENED,
    SWATH_RESTART_ORIGIN_X,
    SWATH_RESTART_ORIGIN_Y,
    SWATH_RESTART_PLANE,
    SWATH_RESTART_RESIDUAL_X,
    SWATH_RESTART_RESIDUAL_Y,
    SWATH_RESTART_SERIES,
    SWATH_RESTART_UPDATES,
    SwathRestartLayout,
    SwathRestartState,
    binary32_chunks_as_binary64,
)
from .gm1_runtime import (
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    FIRST_RESTART_NUMBER,
    MOVE_SPEED_X_NAME,
    MOVE_SPEED_Y_NAME,
    RuntimeClock,
    RuntimeConfig,
    cm1_real,
)


CARTESIAN_RESTART_PROFILE_SCHEMA = "gm1.cartesian-restart-profile/v1"
# v2 carries the swaths a run keeps (the ``swath`` group) with their layout.
# Its state part -- every carrier outside that group -- is the v1 profile
# word for word, which is what a resume compares (state_profile_sha256), so a
# v1 restart still resumes, with its swaths restarting at the restart step.
CARTESIAN_RESTART_PROFILE_ID = "gm1.cartesian-restart/v2"
CARTESIAN_RESTART_PROFILE_ID_V1 = "gm1.cartesian-restart/v1"
CARTESIAN_RESTART_PROFILE_IDS = (CARTESIAN_RESTART_PROFILE_ID_V1, CARTESIAN_RESTART_PROFILE_ID)
# v2 is the one-copy restart directory: restart-carriers.f32 is no longer
# kept beside the envelope.  A v2 receipt always carries ``arena.retained``;
# when it is false ``arena_filename`` and ``arena.path`` are null and
# ``arena.disposition`` and ``arena.envelope_verification`` say what happened
# to the raw arena.  Every v1 key keeps its meaning, the arena digests are
# unchanged, and a resume reads only the envelope in either version, so both
# are admitted; a reader that followed ``arena.path`` to a v1 directory's raw
# arena must check the schema (or ``arena.retained``) first.
CARTESIAN_RESTART_PUBLICATION_SCHEMA = "gm1.cartesian-restart-publication/v2"
CARTESIAN_RESTART_PUBLICATION_SCHEMA_V1 = "gm1.cartesian-restart-publication/v1"
CARTESIAN_RESTART_PUBLICATION_SCHEMAS = (
    CARTESIAN_RESTART_PUBLICATION_SCHEMA_V1,
    CARTESIAN_RESTART_PUBLICATION_SCHEMA,
)
CARTESIAN_RESTART_RESUME_SCHEMA = "gm1.cartesian-restart-resume/v1"
CARTESIAN_RESTART_STATUS = "PUBLISHED_CARTESIAN_RESTART"
CARTESIAN_RESUME_STATUS = "RESTORED_CARTESIAN_RESTART"
CARTESIAN_RESTART_AUTHORITY = (
    "CM1 r21.1 restart_write.F:324-416,656-658,692-757,767-840 and "
    "restart_read.F:337-430 (cm1.F:2666 irst=1, cm1.F:4252 rstfrq)"
)
FORECAST_CHECKPOINT_MAGIC = "CM1FCP1\0"

# CM1's remaining restart namelist surface, answered rather than exposed.
# Each of these selects between alternatives CM1 has and this lane does not,
# so declaring one would be a configuration key with a single admissible
# value.  The answer travels in the artifact instead, where a CM1 user reading
# a GM1 restart can find it; a key nobody consumes is still refused by name
# through CM1-CFG-UNKNOWN, so nobody sets one of these and is quietly ignored.
CM1_RESTART_NAMELIST_ANSWERS = MappingProxyType({
    "restart_format": (
        "neither of CM1's two (README.namelist:1459-1462): this lane writes one "
        "CM1FCP1 envelope, a self-describing contract-validated artifact, not a "
        "GrADS direct-access set and not a netCDF file"
    ),
    "restart_filetype": (
        "CM1's 2 (README.namelist:1464-1486): one restart per restart time, in "
        "its own directory named restart-NNNNNN, renamed into place only once "
        "every file in it is on disk"
    ),
    "restart_reset_frqtim": (
        "run.restart_reset_frqtim, defaulting to CM1's .false. branch rather "
        "than its .true. one, because README.namelist:1505-1510 names .false. "
        "as the option that may be needed for bit-identical restarts"
    ),
    "restart_file_theta/dbz/th0/prs0/pi0/rho0/qv0/u0/v0/zs/zh/zf/diags": (
        "all .false. (README.namelist:1512-1546): README.namelist:1512-1516 "
        "records that none of them is required for a bit-identical restart, and "
        "the base state and the heights are rebuilt from the sounding and the "
        "geometry this restart's receipt pins"
    ),
    "restart_use_theta": (
        "CM1's .false. default (README.namelist:1550-1573): perturbation "
        "potential temperature is carried and used as it is, which is the "
        "branch CM1 names as needed for bit-identical restarts; the .true. "
        "branch adds and then subtracts the base state and loses exactness to "
        "round-off"
    ),
})

# cm1.F:301 / README.restart: cm1rst_000001_*.dat counts upward from one.
RESTART_DIRECTORY_FORMAT = "restart-{number:06d}"
ARENA_FILENAME = "restart-carriers.f32"
FIELD_TABLE_FILENAME = "restart-field-table.tsv"
SCALAR_TABLE_FILENAME = "restart-scheduler-scalars.tsv"
RECEIPT_FILENAME = "restart-publication.json"
ENVELOPE_FILENAME = "restart.cm1fcp"
# Written by `cm1-forecast-publish restore` into the directory it publishes.
RESTORE_ARENA_FILENAME = "restart-carriers.f32"
RESTORE_FIELD_TABLE_FILENAME = "restart-field-table.tsv"
RESTORE_SCALAR_TABLE_FILENAME = "restart-scheduler-scalars.tsv"
RESTORE_RECEIPT_FILENAME = "restart-restore.json"

CARTESIAN_RESTART_CLAIM_LIMIT = (
    "one complete carried state of this runtime at a completed large-step "
    "boundary, byte-exact and validated by Rust, sufficient to continue the "
    "integration where it stopped; the base state, grid metrics and uniform "
    "grid coefficients are rebuilt from the pinned sounding and geometry "
    "rather than carried (README.namelist:1512-1516)"
)

# ---------------------------------------------------------------- scalars
#
# CM1 names every one of these; the comment on each is where restart_write.F
# writes it.  The f64 scalars cross the Rust boundary as sixteen hexadecimal
# characters, so a dry-mass reference or a domain location arrives with the
# bits the run accumulated and not a decimal round trip.
MASS1_SCALAR_NAME = "mass1"
ASQ_SCALAR_FORMAT = "asq{index}"
ASQ_LENGTH = 6
HISTORY_COUNTER_NAME = "nwrite"
RESTART_COUNTER_NAME = "nrst"
PHYSICS_STEP_NAME = "completed_physics_step"
SOLVE3_STEP_NAME = "completed_solve3_step"
# The closure's own cadence token.  CartesianClosure.closure refuses any step
# that is not `_held_step + 1`, so a runtime restored with the prognostic state
# of step N and a closure still at step 0 refuses its own next large step --
# after the device has been opened, the atmosphere built and the whole arena
# uploaded.  It is carried for the same reason the physics and solve3 tokens
# are: the cadence is part of the carried state, not a consequence of it.
CLOSURE_STEP_NAME = "completed_closure_step"
RESTART_SCALAR_AUTHORITY = MappingProxyType({
    DOMAIN_LOCATION_X_NAME: "restart_write.F:399",
    DOMAIN_LOCATION_Y_NAME: "restart_write.F:400",
    MOVE_SPEED_X_NAME: "restart_write.F:393",
    MOVE_SPEED_Y_NAME: "restart_write.F:396",
    MASS1_SCALAR_NAME: "restart_write.F:412",
    HISTORY_COUNTER_NAME: "restart_write.F:378",
    RESTART_COUNTER_NAME: "restart_write.F:383",
    "asq": "restart_write.F:657",
    PHYSICS_STEP_NAME: "gm1 cuda_cartesian_finish physics cadence token",
    SOLVE3_STEP_NAME: "gm1 cuda_cartesian_finish solve3 cadence token",
    CLOSURE_STEP_NAME: "gm1 cuda_cartesian_closure held-tendency cadence token",
})

# ---------------------------------------------------------------- carriers

_GROUP_SOURCE = MappingProxyType({
    "padded": ("dynamics", "arrays"),
    "fields": ("dynamics", "fields"),
    "held": ("dynamics", "held"),
    "tke": ("dynamics", "tke"),
    "component": ("finish", "persistence"),
})
_MEANINGS = MappingProxyType({
    "ua": "previous-step west-east velocity, padded owner",
    "u3d": "current west-east velocity, padded owner",
    "va": "previous-step south-north velocity, padded owner",
    "v3d": "current south-north velocity, padded owner",
    "wa": "previous-step vertical velocity, padded owner",
    "w3d": "current vertical velocity, padded owner",
    "ppi": "previous-step perturbation nondimensional pressure",
    "pp3d": "current perturbation nondimensional pressure",
    "ppx": "accumulated acoustic pressure change",
    "ppten": "held pressure tendency",
    "rru": "west-east mass flux",
    "rrv": "south-north mass flux",
    "rrw": "vertical mass flux",
    "radbcw": "west radiative lateral-boundary phase speed history",
    "radbce": "east radiative lateral-boundary phase speed history",
    "radbcs": "south radiative lateral-boundary phase speed history",
    "radbcn": "north radiative lateral-boundary phase speed history",
    "rf": "current air density on w faces",
    "rr": "current inverse air density",
    "rho": "current air density",
    "prs": "current air pressure",
    "tha": "previous-step potential temperature",
    "th3d": "current potential temperature",
    "qa": "previous-step qv,qc,qr,qi,qs,qg mixing ratios",
    "q3d": "current qv,qc,qr,qi,qs,qg mixing ratios",
    "qpten": "eqtset=2 pressure rate carrier",
    "qtten": "eqtset=2 potential-temperature rate carrier",
    "qvten": "eqtset=2 water-vapour rate carrier",
    "qcten": "eqtset=2 cloud-water rate carrier",
    "uten": "held west-east momentum tendency",
    "vten": "held south-north momentum tendency",
    "wten": "held vertical momentum tendency",
    "thten1": "held potential-temperature tendency",
    "qten": "held moisture tendencies",
    "tkea": "previous-step subgrid turbulence kinetic energy",
    "tke3d": "current subgrid turbulence kinetic energy",
    "tketen": "subgrid turbulence kinetic energy tendency",
    "h_diabatic": "microphysics diabatic heating rate",
    "effc": "cloud-droplet effective radius",
    "effi": "cloud-ice effective radius",
    "effs": "snow effective radius",
    "mp_rainnc": "accumulated total precipitation",
    "mp_rainncv": "one-step total precipitation",
    "mp_snownc": "accumulated snow precipitation",
    "mp_snowncv": "one-step snow precipitation",
    "effr": "rain effective radius",
    "nc": "cloud-droplet number concentration, reassigned from ndcnst each call",
    "nwfa2d": "surface water-friendly aerosol emission rate",
    "nifa2d": "surface ice-friendly aerosol emission rate",
    "mp_graupelnc": "accumulated graupel precipitation",
    "mp_graupelncv": "one-step graupel precipitation",
    "mp_sr": "frozen fraction of one-step precipitation",
})
_UNITS = MappingProxyType({
    "ua": "m s-1", "u3d": "m s-1", "va": "m s-1", "v3d": "m s-1",
    "wa": "m s-1", "w3d": "m s-1",
    "ppi": "nondimensional", "pp3d": "nondimensional", "ppx": "nondimensional",
    "ppten": "nondimensional s-1",
    "rru": "kg m-2 s-1", "rrv": "kg m-2 s-1", "rrw": "kg m-2 s-1",
    "radbcw": "m s-1", "radbce": "m s-1", "radbcs": "m s-1", "radbcn": "m s-1",
    "rf": "kg m-3", "rr": "m3 kg-1", "rho": "kg m-3", "prs": "Pa",
    "tha": "K", "th3d": "K", "qa": "kg kg-1", "q3d": "kg kg-1",
    "qpten": "Pa s-1", "qtten": "K s-1", "qvten": "kg kg-1 s-1", "qcten": "kg kg-1 s-1",
    "uten": "m s-2", "vten": "m s-2", "wten": "m s-2",
    "thten1": "K s-1", "qten": "kg kg-1 s-1",
    "tkea": "m2 s-2", "tke3d": "m2 s-2", "tketen": "m2 s-3",
    "h_diabatic": "K s-1", "effc": "m", "effi": "m", "effs": "m",
    "effr": "m", "nc": "kg-1", "nwfa2d": "kg-1 s-1", "nifa2d": "kg-1 s-1",
    "mp_rainnc": "mm", "mp_rainncv": "mm", "mp_snownc": "mm", "mp_snowncv": "mm",
    "mp_graupelnc": "mm", "mp_graupelncv": "mm", "mp_sr": "1",
})


def _species_axis_text(profile: Any) -> dict[str, tuple[str, str]] | None:
    """Meaning and units of the q3d-axis carriers for a double-moment axis.

    ``None`` for an all-mass axis, whose rows keep the text WSM6 restarts have
    always carried.  An axis with number concentrations names both halves: CM1
    gives the four Morrison rows ``qunit = '#/kg'`` (``src/param.F:3470-3473``)
    and one field-table row carries one units string, so it states both.
    """

    if not profile.number_names:
        return None
    mass, number = ",".join(profile.mass_names), ",".join(profile.number_names)
    rows = f"rows {mass} kg kg-1 then {number} kg-1"
    return {
        "qa": (f"previous-step {mass} mixing ratios then {number} number concentrations", "kg kg-1; kg-1"),
        "q3d": (f"current {mass} mixing ratios then {number} number concentrations", "kg kg-1; kg-1"),
        "qten": (f"held moisture tendencies, {rows}", "kg kg-1 s-1; kg-1 s-1"),
    }


def _stagger(
    group: str, key: str, shape: tuple[int, ...], geometry: Any, numq: int = 6
) -> str:
    if group == "padded":
        return f"padded-{key}"
    if shape == geometry.surface_shape:
        return "surface"
    if shape == geometry.w_shape or shape == (6, *geometry.w_shape):
        return "w"
    if shape == geometry.u_shape:
        return "u"
    if shape == geometry.v_shape:
        return "v"
    if shape == geometry.mass_shape:
        return "mass"
    # CM1 writes one qa(n) per scalar on the component's axis
    # (restart_write.F:809-819), so the species extent is that axis's numq.
    if shape == (6, *geometry.mass_shape) or shape == (numq, *geometry.mass_shape):
        return "mass-species"
    return "runtime"


@dataclass(frozen=True, slots=True)
class CartesianRestartCarrier:
    """One persisted float32 carrier and where the live runtime holds it."""

    name: str
    meaning: str
    units: str
    stagger: str
    shape: tuple[int, ...]
    group: str
    key: str

    @property
    def nbytes(self) -> int:
        return shape_bytes(self.shape)

    @property
    def words(self) -> int:
        return self.nbytes // 4

    def bounds(self) -> tuple[int, ...]:
        return tuple(value for extent in self.shape for value in (1, extent))

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name, "meaning": self.meaning, "units": self.units,
            "stagger": self.stagger, "shape": list(self.shape),
            "bounds": list(self.bounds()), "nbytes": self.nbytes,
            "runtime_group": self.group, "runtime_key": self.key,
        }


@dataclass(frozen=True, slots=True)
class CartesianRestartProfile:
    """The complete carried state of one runtime geometry, priced."""

    profile_id: str
    geometry: Mapping[str, Any]
    carriers: tuple[CartesianRestartCarrier, ...]
    payload_nbytes: int
    rebuilt: Mapping[str, str]
    #: The swaths a v2 parent restart carries (the ``swath`` group), if any.
    swaths: SwathRestartLayout | None = None

    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType({c.name: c.shape for c in self.carriers})

    def record(self) -> dict[str, Any]:
        row = {
            "schema": CARTESIAN_RESTART_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "authority": CARTESIAN_RESTART_AUTHORITY,
            "geometry": dict(self.geometry),
            "carriers": [carrier.record() for carrier in self.carriers],
            "carrier_names": [carrier.name for carrier in self.carriers],
            "payload_nbytes": self.payload_nbytes,
            "maximum_forecast_payload_bytes": MAX_FORECAST_PAYLOAD_BYTES,
            "rebuilt_rather_than_carried": dict(self.rebuilt),
            "scheduler_scalar_authority": dict(RESTART_SCALAR_AUTHORITY),
            "cm1_restart_namelist": dict(CM1_RESTART_NAMELIST_ANSWERS),
            "artifact_magic": FORECAST_CHECKPOINT_MAGIC,
        }
        if self.profile_id == CARTESIAN_RESTART_PROFILE_ID:
            row["swaths"] = None if self.swaths is None else self.swaths.record()
        return row

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.record())

    @property
    def state_sha256(self) -> str:
        """The digest a resume compares: the carried state without the swaths."""

        return state_profile_sha256(self.record())

    def state_only(self) -> "CartesianRestartProfile":
        """This profile without its swath group: what a restore uploads directly."""

        if self.swaths is None:
            return self
        carriers = tuple(c for c in self.carriers if c.group != SWATH_RESTART_GROUP)
        return CartesianRestartProfile(
            self.profile_id, self.geometry, carriers,
            forecast_arena_nbytes({c.name: c.shape for c in carriers}), self.rebuilt,
        )


def state_profile_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A restart profile record reduced to its carried state, in v1 form.

    Drops the ``swath`` group and the layout, and names a v2 parent profile by
    the v1 id, so a v1 restart's recorded profile and the same run's v2
    profile reduce to the same bytes.  The swath group's shapes follow the
    run's length, and a resume may run longer or shorter than the run that
    wrote the restart; the state carriers' may not change at all.
    """

    row = dict(record)
    row.pop("swaths", None)
    if row.get("profile_id") == CARTESIAN_RESTART_PROFILE_ID:
        row["profile_id"] = CARTESIAN_RESTART_PROFILE_ID_V1
    carriers = [
        carrier for carrier in row["carriers"]
        if carrier.get("runtime_group") != SWATH_RESTART_GROUP
    ]
    if len(carriers) != len(row["carriers"]):
        row["carriers"] = carriers
        row["carrier_names"] = [carrier["name"] for carrier in carriers]
        row["payload_nbytes"] = sum(int(carrier["nbytes"]) for carrier in carriers)
    return row


def state_profile_sha256(record: Mapping[str, Any]) -> str:
    """The SHA-256 of :func:`state_profile_record`."""

    return _canonical_sha256(state_profile_record(record))


def cartesian_restart_profile(
    config: RuntimeConfig, *, profile_id: str = CARTESIAN_RESTART_PROFILE_ID,
    swaths: SwathRestartLayout | None = None,
) -> CartesianRestartProfile:
    """Price one complete restart and refuse a grid the envelope cannot carry.

    Nothing here touches CUDA.  The refusal happens before a device word is
    read, so a grid whose carried state exceeds the envelope contract costs no
    GPU time and, more to the point, is discovered before the run rather than
    three hours into it when the first restart is due.

    ``swaths`` is the layout of the swaths the run keeps
    (:meth:`cm1core.cartesian_tornado_diagnostics.CartesianTornadoDiagnostics.restart_layout`,
    or :func:`~cm1core.cartesian_tornado_diagnostics.swath_restart_layout`
    before a device exists); they are appended as the ``swath`` group.
    """

    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("restart profile id must be a nonempty string")
    if swaths is not None:
        if not isinstance(swaths, SwathRestartLayout):
            raise TypeError("swaths must be a SwathRestartLayout")
        if profile_id != CARTESIAN_RESTART_PROFILE_ID:
            raise ValueError(
                f"only a {CARTESIAN_RESTART_PROFILE_ID} parent restart carries the "
                f"swath group; {profile_id} keeps its swaths in its own carriers"
            )
        if tuple(swaths.surface_shape) != tuple(config.grid.surface_shape):
            raise ValueError(
                f"the swath layout is {tuple(swaths.surface_shape)} and this grid's "
                f"surface is {tuple(config.grid.surface_shape)}"
            )
    geometry = config.grid
    state = cartesian_dynamics_restart_state(config)
    groups: dict[str, dict[str, tuple[int, ...]]] = dict(state["carried"])
    # The selected component's own persistence inventory, in its own order.
    # For WSM6 that is exactly WSM6_RESTART_REQUIRED with WSM6_SURFACE_SCRATCH
    # on the surface grid; Morrison adds effr and its component-owned nc, and
    # aerosol-aware Thompson its two surface emission carriers.  A restart that
    # kept only WSM6's list would resume a double-moment run with those
    # carriers at their fresh-start values.
    spec = column_component_spec(config.microphysics)
    groups["component"] = {
        name: (
            geometry.surface_shape if name in spec.surface_persistence
            else geometry.mass_shape
        )
        for name in spec.component_persistence
    }
    profile_axis = config.profile
    species_text = _species_axis_text(profile_axis)
    carriers: list[CartesianRestartCarrier] = []
    for group, shapes in groups.items():
        for key, shape in shapes.items():
            shape = tuple(int(extent) for extent in shape)
            name = f"{group}.{key}"
            try:
                meaning, units = _MEANINGS[key], _UNITS[key]
                if species_text is not None and key in species_text:
                    meaning, units = species_text[key]
            except KeyError:
                raise ValueError(
                    f"restart carrier {name} has no declared meaning and units; a "
                    "carrier published without them would reach the envelope as an "
                    "unnamed block of words that no consumer could identify"
                ) from None
            carriers.append(CartesianRestartCarrier(
                name, meaning, units,
                _stagger(group, key, shape, geometry, profile_axis.numq),
                shape, group, key,
            ))
    if swaths is not None:
        carriers.extend(
            CartesianRestartCarrier(name, meaning, units, stagger, shape, SWATH_RESTART_GROUP, key)
            for name, meaning, units, stagger, shape, key in swaths.carriers()
        )
    for carrier in carriers:
        if carrier.nbytes > MAX_FORECAST_PAYLOAD_BYTES:
            raise ValueError(
                f"restart carrier {carrier.name} alone needs {carrier.nbytes} bytes "
                f"at nx={geometry.nx}, ny={geometry.ny}, nz={geometry.nz}; the "
                f"CM1FCP1 array contract caps one carrier at "
                f"{MAX_FORECAST_PAYLOAD_BYTES} bytes, so this run could never write "
                "a restart and a stop would lose the whole integration"
            )
    payload = forecast_arena_nbytes({c.name: c.shape for c in carriers})
    if payload > MAX_FORECAST_PAYLOAD_BYTES:
        raise ValueError(
            f"the carried state is {payload} bytes at nx={geometry.nx}, "
            f"ny={geometry.ny}, nz={geometry.nz}; the CM1FCP1 contract caps one "
            f"forecast payload at {MAX_FORECAST_PAYLOAD_BYTES} bytes, so this grid "
            "has no writable restart and a run on it could never be continued"
        )
    return CartesianRestartProfile(
        profile_id, MappingProxyType(dict(geometry.record())), tuple(carriers),
        payload, MappingProxyType(dict(state["rebuilt"])), swaths,
    )


def runtime_restart_profile(runtime: Any) -> CartesianRestartProfile:
    """The restart profile of a live runtime: its state plus the swaths it keeps."""

    tornado = getattr(runtime, "tornado", None)
    layout = None if tornado is None else tornado.restart_layout()
    return cartesian_restart_profile(runtime.config, swaths=layout)


# ---------------------------------------------------------------- boundary


def require_cartesian_restart_boundary(runtime: Any) -> int:
    """Admit the quiescent state of a completed large step, or the source state.

    ``cm1.F:3186`` fires a restart between completed large steps, where the
    current and A carriers agree and the padded owners have been refreshed.
    Anywhere else the two families belong to different RK stages and a resume
    would continue from a state the model never held.

    Step zero is admitted, and it is the only difference from the history
    boundary: ``cm1.F:3110-3115`` writes a restart at ``nstep=0`` when
    ``rstfrq`` is zero, and ``cm1.F:4341-4350``'s ``iconly`` stops the model
    there.  That state is the initial condition, which is exactly what a user
    checking a sounding, a bubble and a grid wants to hold.
    """

    clock, dynamics, finish = runtime.clock, runtime.dynamics, runtime.finish
    if clock.failed:
        raise RuntimeError(
            "a failed CUDA step leaves the device state undefined; a restart "
            "written here would resume onto words no completed model time produced"
        )
    step = int(clock.completed_steps)
    if int(finish.completed_physics_step) != step or int(finish.completed_solve3_step) != step:
        raise RuntimeError(
            f"the runtime completed {step} steps but stands at physics step "
            f"{int(finish.completed_physics_step)} and solve3 step "
            f"{int(finish.completed_solve3_step)}; a restart taken here would "
            "carry pre- and post-physics words at one model time and resume onto "
            "a state that is neither"
        )
    if dynamics._expected_phase != ("held", 1):
        raise RuntimeError(
            f"the dynamics cadence is at {dynamics._expected_phase!r}, not the "
            "post-solve3 quiescent boundary; the padded wind owners still hold the "
            "previous stage's words and a resume would start mid-RK-stage"
        )
    closure = runtime.closure
    if (
        int(closure._held_step) != step
        or closure._pending_step is not None
        or int(closure._tke_stage) != 3
    ):
        raise RuntimeError(
            f"the closure stands at held step {closure._held_step!r}, pending step "
            f"{closure._pending_step!r} and TKE stage {closure._tke_stage!r} at "
            f"{step} completed large steps; the held tendencies and the TKE stages "
            "of this step are not both finished, so the state is mid-step"
        )
    return step


# ---------------------------------------------------------------- helpers


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")).hexdigest()


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb", buffering=0) as stream:
        stream.write(payload)
        os.fsync(stream.fileno())


def _f64_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def _from_f64_bits(text: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(text, 16)))[0]


def _restart_carriers(
    runtime: Any, profile: CartesianRestartProfile,
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Bind every declared carrier to the live allocation that holds it.

    Nothing is staged.  Every carrier in this profile is a contiguous owner:
    the wind and pressure fields are persisted as their padded owners, not as
    the strided physical views into them, precisely so that the halo words the
    next large step's fifth-order stencils read are carried too.

    ``extra`` binds the carriers of a group this module does not own by name
    (a nest child's coupling state, :mod:`cm1core.nest_restart`); a declared
    carrier of such a group that ``extra`` does not bind is refused like any
    other missing carrier.
    """

    carriers: dict[str, Any] = {}
    for carrier in profile.carriers:
        if carrier.group not in _GROUP_SOURCE:
            value = None if extra is None else extra.get(carrier.name)
            if value is None:
                raise ValueError(
                    f"restart carrier {carrier.name} belongs to group "
                    f"{carrier.group!r}, which no live allocation was bound for; "
                    "a resume from a partial restart would continue from words "
                    "this run never held"
                )
            carriers[carrier.name] = value
            continue
        holder, attribute = _GROUP_SOURCE[carrier.group]
        try:
            container = getattr(runtime, holder)
            value = getattr(container, attribute)[carrier.key]
        except (AttributeError, KeyError):
            raise ValueError(
                f"restart carrier {carrier.name} is not present as "
                f"{holder}.{attribute}[{carrier.key!r}]; this runtime cannot write "
                "the complete carried state and a resume from a partial restart "
                "would continue from words this run never held"
            ) from None
        carriers[carrier.name] = value
    return MappingProxyType(carriers)


def _field_table_bytes(profile: CartesianRestartProfile, arena: ForecastArenaWrite) -> bytes:
    rows = []
    for carrier in profile.carriers:
        columns = (
            carrier.name, carrier.meaning, carrier.units, carrier.stagger,
            ",".join(str(extent) for extent in carrier.shape),
            ",".join(str(bound) for bound in carrier.bounds()),
            os.fspath(arena.path.resolve()),
            str(arena.carrier_offset_bytes[carrier.name]),
        )
        if any("\t" in column or "\n" in column for column in columns):
            raise ValueError(
                f"restart carrier {carrier.name} carries a tab or newline; the "
                "Rust field table is strictly tab-separated and would split the row"
            )
        rows.append("\t".join(columns))
    return ("\n".join(rows) + "\n").encode("ascii")


def restart_scalars(
    runtime: Any, *, next_history_index: int, restart_number: int
) -> dict[str, tuple[str, str]]:
    """CM1's carried scalars for this model time, by CM1's own names.

    Every value that the next large step reads and no operator recomputes.
    ``mass1`` is the one whose absence would be silent: ``cm1.F`` fixes the
    dry-air mass reference once from the initial state and
    ``apmasscon`` corrects every later step back to it, so a resumed run that
    recomputed it from the restarted state would correct the rest of the
    integration towards a different total mass than the run it is continuing.
    """

    clock, config, finish = runtime.clock, runtime.config, runtime.finish
    asq = runtime.cp.asnumpy(finish.asq).reshape(-1)
    if int(asq.size) != ASQ_LENGTH:
        raise ValueError(
            f"the positive-definite moisture accumulator has {int(asq.size)} "
            f"entries; restart_write.F:657 writes one per moisture species and "
            f"this lane carries {ASQ_LENGTH}"
        )
    rows: dict[str, tuple[str, str]] = {
        DOMAIN_LOCATION_X_NAME: ("f64-bits", _f64_bits(clock.domainlocx_m)),
        DOMAIN_LOCATION_Y_NAME: ("f64-bits", _f64_bits(clock.domainlocy_m)),
        MOVE_SPEED_X_NAME: ("f64-bits", _f64_bits(cm1_real(config.u_move_m_s))),
        MOVE_SPEED_Y_NAME: ("f64-bits", _f64_bits(cm1_real(config.v_move_m_s))),
        MASS1_SCALAR_NAME: ("f64-bits", _f64_bits(finish.initial_mass1_f64)),
        HISTORY_COUNTER_NAME: ("u64", str(int(next_history_index))),
        RESTART_COUNTER_NAME: ("u64", str(int(restart_number))),
        PHYSICS_STEP_NAME: ("u64", str(int(finish.completed_physics_step))),
        SOLVE3_STEP_NAME: ("u64", str(int(finish.completed_solve3_step))),
        CLOSURE_STEP_NAME: ("u64", str(int(runtime.closure._held_step))),
    }
    for index in range(ASQ_LENGTH):
        rows[ASQ_SCALAR_FORMAT.format(index=index)] = (
            "f64-bits", _f64_bits(float(asq[index]))
        )
    # restart_write.F:377-401.  CM1's adaptive moving domain carries fifteen
    # more words from one timestep to the next, and a resume that rebuilt them
    # would restart the controller at param.F:706-707's zero velocity with
    # xcenter back at the domain centre -- the frame would jump to a stop, the
    # winds would be rebased by the whole accumulated velocity at the next
    # cadence, and the domain-location integral would lose everything the
    # interrupted run had travelled.  A fixed-frame run writes none of these
    # rows, so its restart is byte-identical to the one it wrote before the
    # moving domain could be adaptive.
    adaptive = getattr(runtime, "adaptive_frame", None)
    if adaptive is not None:
        rows.update(adaptive.restart_scalars())
    return rows


def _scalar_table_bytes(rows: Mapping[str, tuple[str, str]]) -> bytes:
    return "".join(
        f"{name}\t{kind}\t{value}\n" for name, (kind, value) in rows.items()
    ).encode("ascii")


def _run_checked(arguments: Sequence[str | os.PathLike[str]], label: str,
                 io: Any = None, cancel: Any = None) -> str:
    from .cartesian_history import _run_checked as run_publisher

    return run_publisher(arguments, label, io=io, cancel=cancel)


# ---------------------------------------------------------------- write


@dataclass(frozen=True, slots=True)
class CartesianRestartSnapshot:
    """One restart's carried words and every runtime value its receipt reads."""

    directory: Path
    location: Path
    executable: Path
    profile: CartesianRestartProfile
    arena: ForecastArenaSnapshot
    scalars: Mapping[str, tuple[str, str]]
    step: int
    model_time_s: float
    dt_s: float
    config_record: Mapping[str, Any]
    execution_identity: str
    source_identity: Mapping[str, Any]
    restart_number: int
    next_history_index: int
    # Receipt sections a nested run adds: the parent's declaration of its
    # child restart, the child's coupling state (cm1core.nest_restart).  None
    # -- and no receipt key -- for every single-domain run.
    receipt_extra: Mapping[str, Any] | None = None
    # What the swath group carries (counters and layout), for the receipt's
    # "swaths" section; None when the profile carries no swath.
    swaths: Mapping[str, Any] | None = None


def snapshot_cartesian_restart(
    runtime: Any, directory: Path | str, *, publisher: Path | str,
    profile: CartesianRestartProfile | None = None,
    restart_number: int = FIRST_RESTART_NUMBER,
    next_history_index: int = 1,
    published_directory: Path | str | None = None,
    host: HostArena | None = None,
    extra_carriers: Mapping[str, Any] | None = None,
    extra_scalars: Mapping[str, tuple[str, str]] | None = None,
) -> CartesianRestartSnapshot:
    """The device half of a restart: every refusal, the scalars, one host copy.

    Runs at the quiescent post-solve3 boundary on the stepping thread.  No
    directory is created; nothing on the device is allocated or left behind.
    Without a ``host`` arena no copy is taken either: the snapshot names the
    live carriers, and :func:`commit_cartesian_restart` must follow at once on
    this thread, streaming them as the one-call writer always has.

    ``extra_carriers`` and ``extra_scalars`` carry state this runtime does not
    own by itself -- a nest child's coupling (:mod:`cm1core.nest_restart`) --
    beside its own inventory; neither may rename a carrier or scalar the
    runtime already writes.
    """

    step = require_cartesian_restart_boundary(runtime)
    if int(getattr(runtime.config, "sgsmodel", 1)) == 3:
        raise ValueError(
            "sgsmodel=3 carries gamk from one step to the next and CM1 writes it "
            "(src/restart_write.F:1598); this restart inventory has no profile "
            "carrier, so the resumed step would scale kmh/kmv by gamk=1 instead "
            "of the persisted isotropy factor.  Use sgsmodel=4 or 1 for a run "
            "that must be continued"
        )
    if profile is None:
        profile = runtime_restart_profile(runtime)
    if profile.geometry != runtime.config.grid.record():
        raise ValueError(
            "the supplied restart profile prices another geometry than this "
            "runtime; its carrier shapes would misdescribe every persisted word "
            "and a resume would read the state onto the wrong grid"
        )
    for name, value in (
        ("restart_number", restart_number), ("next_history_index", next_history_index),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    directory = require_new_history_directory(directory)
    io = getattr(runtime, "active_io_event", None)
    tick = None if io is None else io.begin("publisher_admission")
    executable = require_forecast_publisher(publisher)
    if io is not None:
        io.end("publisher_admission", tick)
    config_record = runtime.config.record()
    tick = None if io is None else io.begin("scheduler_scalars")
    scalars = restart_scalars(
        runtime, next_history_index=next_history_index, restart_number=restart_number
    )
    swath_record = None
    if profile.swaths is not None:
        # restart_write.F:696-758: the swaths the run keeps, beside the state.
        swath_carriers, swath_scalars, swath_record = _swath_restart_words(
            runtime, profile.swaths, step=step,
        )
        extra_carriers = _merge_extra(extra_carriers, swath_carriers, "carriers")
        extra_scalars = _merge_extra(extra_scalars, swath_scalars, "scalars")
    if extra_scalars:
        clash = sorted(set(scalars) & set(extra_scalars))
        if clash:
            raise ValueError(
                f"extra restart scalars {clash} rename scalars this runtime already "
                "writes; one table row would hold two meanings"
            )
        scalars.update(extra_scalars)
    if io is not None:
        io.end("scheduler_scalars", tick)
    carriers = _restart_carriers(runtime, profile, extra_carriers)
    tick = None if io is None else io.begin("device_sync")
    runtime.stream.synchronize()
    if io is not None:
        io.end("device_sync", tick)
        io.set_bytes(sum(4 * _words(shape) for shape in profile.shapes().values()))
    words = snapshot_forecast_arena(
        carriers, shapes=profile.shapes(), cupy=runtime.cp, host=host, io=io,
    )
    return CartesianRestartSnapshot(
        directory=directory,
        location=directory if published_directory is None else Path(published_directory),
        executable=executable,
        profile=profile,
        arena=words,
        scalars=MappingProxyType(dict(scalars)),
        step=step,
        model_time_s=float(runtime.clock.model_seconds),
        dt_s=float(runtime.config.dt_s),
        config_record=config_record,
        execution_identity=str(runtime.runtime_source_identity["aggregate_sha256"]),
        source_identity=MappingProxyType(dict(runtime.source_identity)),
        restart_number=int(restart_number),
        next_history_index=int(next_history_index),
        swaths=swath_record,
    )


def _merge_extra(
    given: Mapping[str, Any] | None, added: Mapping[str, Any], what: str,
) -> dict[str, Any]:
    merged = dict(given or {})
    clash = sorted(set(merged) & set(added))
    if clash:
        raise ValueError(
            f"restart {what} {clash} are bound twice, by the caller and by the swath "
            "group; one name would hold two meanings"
        )
    merged.update(added)
    return merged


def _swath_restart_words(
    runtime: Any, layout: SwathRestartLayout, *, step: int,
) -> tuple[dict[str, Any], dict[str, tuple[str, str]], dict[str, Any]]:
    """The live swaths' carriers, scalar rows and receipt section for one restart."""

    tornado = getattr(runtime, "tornado", None)
    live = None if tornado is None else tornado.restart_layout()
    if live is None:
        raise RuntimeError(
            "the restart profile carries swaths and none are open at this restart; "
            "a restart without them would resume the run-long extrema from CM1's "
            "initial values and publish part of the run as the whole"
        )
    if live != layout:
        raise ValueError(
            f"the restart profile prices swaths {layout.record()} and this runtime "
            f"keeps {live.record()}; their words would be written under another "
            "layout's shapes"
        )
    arrays, counters = tornado.restart_words()
    if int(counters["step"]) != int(step):
        raise RuntimeError(
            f"the swaths stand at step {counters['step']} and the restart at {step}; "
            "a swath that missed a completed step is not the run's extremum"
        )
    scalars: dict[str, tuple[str, str]] = {
        SWATH_RESTART_UPDATES: ("u64", str(int(counters["updates"]))),
        SWATH_RESTART_OPENED: ("u64", str(int(counters["opened_step"]))),
        SWATH_RESTART_RESIDUAL_X: ("f64-bits", _f64_bits(counters["residual_m"][0])),
        SWATH_RESTART_RESIDUAL_Y: ("f64-bits", _f64_bits(counters["residual_m"][1])),
    }
    for name in layout.names:
        scalars[SWATH_RESTART_FIRST.format(name=name)] = (
            "u64", str(int(counters["first_step"][name]))
        )
    origin = counters["canvas_origin_cells_xy"]
    if origin is not None:
        scalars[SWATH_RESTART_ORIGIN_X] = ("i64", str(int(origin[0])))
        scalars[SWATH_RESTART_ORIGIN_Y] = ("i64", str(int(origin[1])))
    record = {
        "carried": list(layout.names),
        "layout": layout.record(),
        "updates": int(counters["updates"]),
        "opened_step": int(counters["opened_step"]),
        "first_step": {name: int(value) for name, value in counters["first_step"].items()},
        "whole_run": all(int(value) == 1 for value in counters["first_step"].values()),
        "authority": "restart_write.F:696-758 (sws..shs; sws2..shs2 under nrain=2)",
    }
    return arrays, scalars, record


# The CM1FCP1/CM1FHS1 prefix: eight magic bytes and the header length as a
# little-endian u64 (rust/cm1-state/src/forecast.rs PREFIX_BYTES), and the
# header cap the writer enforces (MAX_FORECAST_HEADER_BYTES).
_ENVELOPE_PREFIX_BYTES = 16
_ENVELOPE_HEADER_LIMIT_BYTES = 1024 * 1024


def restart_arena_disposition(*, staged: bool) -> str:
    """What became of a restart's raw arena, for ``arena.disposition``.

    ``staged`` is whether the restart was assembled in a staging directory and
    renamed into place (``integrate``), or written straight into its final
    directory (``write_cartesian_restart`` without ``published_directory``, as
    ``CudaCartesianRuntime.publish_restart`` and the storm driver call it).
    """

    where = (
        "inside the staging directory and deleted there, before the directory was "
        "renamed into place,"
        if staged else "beside the envelope in this directory and deleted"
    )
    return (
        f"not retained: {ARENA_FILENAME} was the publisher's input, written unsynced "
        f"{where} once the CM1FCP1 envelope was on disk and its "
        "header's per-array SHA-256 matched every carrier digest recorded here. "
        "The envelope holds the same words and is the only file a resume reads "
        "(cm1-forecast-publish restore). The path column of "
        f"{FIELD_TABLE_FILENAME} names that deleted publisher input, not a file of "
        "this restart"
    )


def _restart_refusal_note(snapshot: CartesianRestartSnapshot) -> str:
    if snapshot.location != snapshot.directory:
        return (
            f"the restart was assembled in the staging directory {snapshot.directory}, "
            "which is discarded; nothing is renamed into place"
        )
    return (
        f"the restart was written in place: {snapshot.directory} keeps its raw arena, "
        f"tables and envelope but no {RECEIPT_FILENAME}, so no resume admits it"
    )


def read_envelope_header(envelope: Path) -> tuple[int, bytes]:
    """A CM1FCP1 envelope's declared header length and header bytes; nothing else is read."""

    envelope = Path(envelope)
    with envelope.open("rb") as stream:
        prefix = stream.read(_ENVELOPE_PREFIX_BYTES)
        if len(prefix) != _ENVELOPE_PREFIX_BYTES or prefix[:8] != FORECAST_CHECKPOINT_MAGIC.encode("ascii"):
            raise RuntimeError(
                f"{envelope} does not begin with the CM1FCP1 magic; nothing proves "
                "the checkpoint holds the arena's words, so the restart is refused "
                "before its arena is deleted"
            )
        header_nbytes = int.from_bytes(prefix[8:], "little")
        if not 0 < header_nbytes <= _ENVELOPE_HEADER_LIMIT_BYTES:
            raise RuntimeError(
                f"{envelope} declares a {header_nbytes}-byte header; the publisher "
                f"caps it at {_ENVELOPE_HEADER_LIMIT_BYTES}"
            )
        header_bytes = stream.read(header_nbytes)
    if len(header_bytes) != header_nbytes:
        raise RuntimeError(
            f"{envelope} holds {len(header_bytes)} of the {header_nbytes} header bytes it declares"
        )
    return header_nbytes, header_bytes


def envelope_header_sha256(envelope: Path) -> str:
    """The SHA-256 of an envelope's header, as its publication receipt recorded it.

    The header holds Rust's SHA-256 of every carrier's words, the contract, the
    step and the model time, so this digest names one publication's words
    without reading its payload (``cm1core.nest_restart`` binds a nest child
    restart to its parent by it).
    """

    return hashlib.sha256(read_envelope_header(envelope)[1]).hexdigest()


def verify_restart_envelope_carries_arena(
    envelope: Path, arena: ForecastArenaWrite, *, contract_sha256: str,
) -> dict[str, Any]:
    """Prove a published checkpoint holds exactly the words the arena held.

    Reads only the envelope's prefix and header.  The header's ``arrays`` rows
    are Rust's SHA-256 of each carrier's payload window, computed from the
    bytes Rust read and wrote; this compares them with the digests the arena
    writer took from the host words, carrier by carrier.  Equal digests for
    every carrier, the same carrier set, the same byte counts and an envelope
    of exactly prefix + header + payload bytes is what licenses deleting the
    raw arena.
    """

    envelope = Path(envelope)
    header_nbytes, header_bytes = read_envelope_header(envelope)
    header = json.loads(header_bytes)
    problems = []
    if header.get("contract_sha256") != contract_sha256:
        problems.append(
            f"contract {header.get('contract_sha256')!r} is not the reported {contract_sha256!r}"
        )
    if int(header.get("payload_nbytes", -1)) != arena.nbytes:
        problems.append(
            f"payload {header.get('payload_nbytes')!r} bytes is not the arena's {arena.nbytes}"
        )
    rows = {row["name"]: row for row in header.get("arrays", [])}
    if set(rows) != set(arena.carrier_sha256):
        problems.append(
            f"carriers {sorted(set(rows) ^ set(arena.carrier_sha256))} are in only one of "
            "the envelope and the arena"
        )
    for name, digest in arena.carrier_sha256.items():
        row = rows.get(name)
        if row is not None and str(row.get("sha256")) != digest:
            problems.append(f"{name} has envelope SHA-256 {row.get('sha256')} and arena SHA-256 {digest}")
    expected_size = _ENVELOPE_PREFIX_BYTES + header_nbytes + arena.nbytes
    size = envelope.stat().st_size
    if size != expected_size:
        problems.append(f"the envelope is {size} bytes, not prefix + header + payload = {expected_size}")
    if problems:
        raise RuntimeError(
            "the published checkpoint does not verifiably hold the arena's words, so "
            "the restart is refused before its arena is deleted and no receipt is "
            "written: "
            + "; ".join(problems)
        )
    return {
        "arrays_verified": len(rows),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "payload_nbytes": arena.nbytes,
    }


def commit_cartesian_restart(
    snapshot: CartesianRestartSnapshot, *, io: Any = None, cancel: Any = None,
    defer_receipt: bool = False,
) -> dict[str, Any]:
    """The host half of a restart: tables, envelope, verification, receipt.

    Reads nothing but ``snapshot``.  The raw arena is written only as the
    publisher's input and is removed once the envelope is verified to hold
    every one of its words: the directory keeps one copy of the state, the
    envelope, which is the only file a resume reads.

    ``defer_receipt`` returns the receipt without writing it, for a caller
    that must add to it what only a later publication knows (a nested run's
    parent declares the child restart written after it) and then writes it
    with :func:`write_restart_receipt`.  Until then the directory holds no
    receipt and admits no resume.
    """

    directory = snapshot.directory
    profile = snapshot.profile
    created: Path | None = None
    try:
        directory.mkdir(parents=True)
        created = directory
        # Unsynced: this file is deleted below once the envelope holding the
        # same words has been fsynced by the publisher and verified.
        arena = write_forecast_arena_snapshot_new(
            directory / ARENA_FILENAME, snapshot.arena, io=io, fsync=False, cancel=cancel,
        )
    except BaseException:
        if created is not None:
            try:
                next(created.iterdir())
            except StopIteration:
                created.rmdir()
            except OSError:
                pass
        raise
    finally:
        snapshot.arena.release()
    tick = None if io is None else io.begin("tables")
    table = _field_table_bytes(profile, arena)
    table_path = directory / FIELD_TABLE_FILENAME
    _write_new_fsynced(table_path, table)
    scalar_bytes = _scalar_table_bytes(snapshot.scalars)
    scalar_path = directory / SCALAR_TABLE_FILENAME
    _write_new_fsynced(scalar_path, scalar_bytes)
    if io is not None:
        io.end("tables", tick, len(table) + len(scalar_bytes))
    model_time_s = snapshot.model_time_s
    dt_s = snapshot.dt_s
    config_record = dict(snapshot.config_record)
    execution_identity = snapshot.execution_identity
    output = _run_checked([
        snapshot.executable, "checkpoint", directory / ENVELOPE_FILENAME, profile.profile_id,
        _canonical_sha256(config_record), execution_identity,
        str(snapshot.step), repr(model_time_s), repr(dt_s), table_path, scalar_path,
    ], "forecast checkpoint publication", io=io, cancel=cancel)
    contract_sha256 = next(
        (line.split("=", 1)[1] for line in output.splitlines()
         if line.startswith("contract_sha256=")),
        None,
    )
    if contract_sha256 is None:
        raise RuntimeError(
            "the forecast publisher did not report a contract SHA-256; without it "
            "the envelope cannot be read back and this restart is unusable"
        )
    if cancel is not None:
        cancel.check()
    tick = None if io is None else io.begin("envelope_verify")
    try:
        verified = verify_restart_envelope_carries_arena(
            directory / ENVELOPE_FILENAME, arena, contract_sha256=contract_sha256,
        )
    except Exception as refusal:
        refusal.add_note(_restart_refusal_note(snapshot))
        raise
    if io is not None:
        io.end("envelope_verify", tick)
        tick = io.begin("arena.discard")
    (directory / ARENA_FILENAME).unlink()
    if io is not None:
        io.end("arena.discard", tick, arena.nbytes)
    location = snapshot.location
    arena_record = arena.record()
    arena_record["path"] = None
    arena_record["retained"] = False
    arena_record["disposition"] = restart_arena_disposition(staged=location != directory)
    arena_record["envelope_verification"] = verified
    receipt = {
        "schema": CARTESIAN_RESTART_PUBLICATION_SCHEMA,
        "status": CARTESIAN_RESTART_STATUS,
        "authority": CARTESIAN_RESTART_AUTHORITY,
        "directory": os.fspath(location),
        "staged_directory": os.fspath(directory),
        "arena_filename": None,
        "field_table_filename": FIELD_TABLE_FILENAME,
        "scalar_table_filename": SCALAR_TABLE_FILENAME,
        "envelope_filename": ENVELOPE_FILENAME,
        "restart_number": snapshot.restart_number,
        "step": snapshot.step,
        "model_time_s": model_time_s,
        "dt_s": dt_s,
        "config": config_record,
        "config_sha256": _canonical_sha256(config_record),
        "execution_identity_sha256": execution_identity,
        "source_identity": dict(snapshot.source_identity),
        "profile": profile.record(),
        "profile_sha256": profile.sha256,
        "arena": arena_record,
        "field_table_path": os.fspath(location / FIELD_TABLE_FILENAME),
        "field_table_sha256": hashlib.sha256(table).hexdigest(),
        "scalar_table_path": os.fspath(location / SCALAR_TABLE_FILENAME),
        "scalar_table_sha256": hashlib.sha256(scalar_bytes).hexdigest(),
        "scheduler_scalars": {name: value for name, (_, value) in snapshot.scalars.items()},
        "scheduler_scalar_types": {name: kind for name, (kind, _) in snapshot.scalars.items()},
        "next_history_index": snapshot.next_history_index,
        "quiescent_boundary": "post-solve3 completed large step",
        "forecast_envelope": {
            "published": True,
            "abi": FORECAST_PUBLISH_ABI,
            "path": os.fspath(location / ENVELOPE_FILENAME),
            "magic": FORECAST_CHECKPOINT_MAGIC,
            "contract_sha256": contract_sha256,
            "rust_output": output,
        },
        "restore_supported": True,
        "finite_checked_by_rust": True,
        "claim_limit": CARTESIAN_RESTART_CLAIM_LIMIT,
    }
    if snapshot.swaths is not None:
        receipt["swaths"] = dict(snapshot.swaths)
    if snapshot.receipt_extra:
        clash = sorted(set(receipt) & set(snapshot.receipt_extra))
        if clash:
            raise ValueError(
                f"receipt sections {clash} would overwrite the restart's own keys"
            )
        receipt.update(dict(snapshot.receipt_extra))
    if not defer_receipt:
        write_restart_receipt(directory, receipt, io=io)
    return receipt


def write_restart_receipt(directory: Path, receipt: Mapping[str, Any], *, io: Any = None) -> None:
    """The last file of a restart, fsynced: a directory without it admits no resume."""

    tick = None if io is None else io.begin("receipt")
    _write_new_fsynced(
        Path(directory) / RECEIPT_FILENAME,
        (json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if io is not None:
        io.end("receipt", tick)


def write_cartesian_restart(
    runtime: Any, directory: Path | str, *, publisher: Path | str,
    profile: CartesianRestartProfile | None = None,
    restart_number: int = FIRST_RESTART_NUMBER,
    next_history_index: int = 1,
    published_directory: Path | str | None = None,
) -> dict[str, Any]:
    """Write one complete restart of the live device state into a new directory.

    Unlike a history frame, this has no publisher-less mode.  A restart that
    Rust never validated is a restart nobody can prove is complete, and the
    first time anyone would find out is three GPU-hours into a resume.

    ``published_directory`` is where the finished restart will finally live
    when the caller writes it into a staging directory and renames it into
    place.  Every path the receipt records is that directory's, so a receipt
    read after the rename names files that exist; the staging path is recorded
    beside them under its own key rather than silently substituted.

    This is :func:`snapshot_cartesian_restart` followed at once by
    :func:`commit_cartesian_restart` on the calling thread.
    """

    snapshot = snapshot_cartesian_restart(
        runtime, directory, publisher=publisher, profile=profile,
        restart_number=restart_number, next_history_index=next_history_index,
        published_directory=published_directory,
    )
    try:
        return commit_cartesian_restart(snapshot, io=getattr(runtime, "active_io_event", None))
    finally:
        snapshot.arena.release()


def _words(shape: Sequence[int]) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


# ---------------------------------------------------------------- read


@dataclass(frozen=True, slots=True)
class RestoredCartesianRestart:
    """One validated restart, unpacked from its envelope and not yet applied."""

    receipt: Mapping[str, Any]
    restore_receipt: Mapping[str, Any]
    directory: Path
    arena_path: Path
    offsets: Mapping[str, int]
    shapes: Mapping[str, tuple[int, ...]]
    scalars: Mapping[str, tuple[str, str]]
    step: int
    model_time_s: float

    def f64(self, name: str) -> float:
        kind, value = self.scalars[name]
        if kind != "f64-bits":
            raise ValueError(f"restart scalar {name} is {kind}, not f64-bits")
        return _from_f64_bits(value)

    def u64(self, name: str) -> int:
        kind, value = self.scalars[name]
        if kind != "u64":
            raise ValueError(f"restart scalar {name} is {kind}, not u64")
        return int(value)

    def i64(self, name: str) -> int:
        kind, value = self.scalars[name]
        if kind != "i64":
            raise ValueError(f"restart scalar {name} is {kind}, not i64")
        return int(value)

    @property
    def profile_id(self) -> str:
        return str(self.receipt["profile"]["profile_id"])

    @property
    def swath_layout(self) -> SwathRestartLayout | None:
        """The swaths this restart carries, or None (a v1 restart carries none)."""

        record = self.receipt["profile"].get("swaths")
        return None if record is None else SwathRestartLayout.from_record(record)

    def record(self) -> dict[str, Any]:
        return {
            "schema": CARTESIAN_RESTART_RESUME_SCHEMA,
            "status": CARTESIAN_RESUME_STATUS,
            "authority": CARTESIAN_RESTART_AUTHORITY,
            "restart_directory": os.fspath(Path(self.receipt["directory"])),
            "restart_envelope_sha256": str(self.restore_receipt["artifact_sha256"]),
            "restore_directory": os.fspath(self.directory),
            "restart_number": int(self.receipt["restart_number"]),
            "step": self.step,
            "model_time_s": self.model_time_s,
            "dt_s": float(self.receipt["dt_s"]),
            "carrier_count": len(self.shapes),
            "arena_nbytes": int(self.restore_receipt["arena_nbytes"]),
            "array_payload_sha256": str(self.restore_receipt["array_payload_sha256"]),
            "artifact_sha256": str(self.restore_receipt["artifact_sha256"]),
            "contract_sha256": str(self.restore_receipt["contract_sha256"]),
            "profile_sha256": str(self.receipt["profile_sha256"]),
            "execution_identity_sha256": str(self.receipt["execution_identity_sha256"]),
            "scheduler_scalars": {name: value for name, (_, value) in self.scalars.items()},
        }


def require_restart_receipt(
    receipt: Mapping[str, Any] | Path | str,
) -> tuple[Mapping[str, Any], Path]:
    """Admit one restart publication receipt and the directory it lives in.

    The directory comes from where the receipt was found, not from what the
    receipt says about itself, so a restart copied or moved after it was
    written still resumes and a receipt pointing at some other run's envelope
    cannot redirect the read.
    """

    if isinstance(receipt, (str, Path)):
        path = Path(receipt)
        if path.is_dir():
            path = path / RECEIPT_FILENAME
        directory = path.parent
        receipt = json.loads(path.read_text(encoding="utf-8"))
    else:
        directory = None
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping, a directory, or a path to one")
    if receipt.get("schema") not in CARTESIAN_RESTART_PUBLICATION_SCHEMAS:
        raise ValueError(
            f"a {' or '.join(CARTESIAN_RESTART_PUBLICATION_SCHEMAS)} receipt is the "
            "only artifact this lane can resume from; nothing else names the "
            "contract the envelope must be read back under, and reading a CM1FCP1 "
            "envelope under the wrong contract decodes every carrier as another one"
        )
    if directory is None:
        directory = Path(receipt["directory"])
    return receipt, directory


def _parse_field_table(text: str) -> tuple[dict[str, int], dict[str, tuple[int, ...]], set[Path]]:
    offsets: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    paths: set[Path] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        columns = line.split("\t")
        if len(columns) != 8:
            raise ValueError(
                f"restored field table line {number} has {len(columns)} columns; "
                "the Rust boundary writes exactly eight and a short row would "
                "misplace every carrier after it"
            )
        name = columns[0]
        shapes[name] = tuple(int(extent) for extent in columns[4].split(","))
        offsets[name] = int(columns[7])
        paths.add(Path(columns[6]))
    return offsets, shapes, paths


def _parse_scalar_table(text: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        columns = line.split("\t")
        if len(columns) != 3:
            raise ValueError(
                f"restored scalar table line {number} has {len(columns)} columns; "
                "the Rust boundary writes exactly three"
            )
        rows[columns[0]] = (columns[1], columns[2])
    return rows


def read_cartesian_restart(
    receipt: Mapping[str, Any] | Path | str, *, publisher: Path | str,
    into: Path | str,
) -> RestoredCartesianRestart:
    """Read one restart envelope back through Rust into a new directory.

    ``restart_read.F`` is where CM1 does this, and it reads the file it wrote
    rather than a summary of it.  Same here: the envelope's own bytes are
    decoded by the boundary that validated them, anchored to the contract
    SHA-256 the publication receipt recorded, and written back out as the
    exact arena and the two tables.  Nothing between the envelope and the
    device re-encodes a word.
    """

    receipt, home = require_restart_receipt(receipt)
    executable = require_forecast_publisher(publisher)
    envelope = home / str(receipt.get("envelope_filename", ENVELOPE_FILENAME))
    if not envelope.is_file():
        raise ValueError(
            f"the restart directory {home} holds no {envelope.name}; the CM1FCP1 "
            "envelope is the whole of the carried state and a resume without it "
            "would silently continue from the initial condition"
        )
    contract_sha256 = str(receipt["forecast_envelope"]["contract_sha256"])
    into = require_new_history_directory(into)
    _run_checked(
        [executable, "restore", envelope, contract_sha256, into],
        "forecast checkpoint restore",
    )
    restore_receipt = json.loads(
        (into / RESTORE_RECEIPT_FILENAME).read_text(encoding="utf-8")
    )
    offsets, shapes, paths = _parse_field_table(
        (into / RESTORE_FIELD_TABLE_FILENAME).read_text(encoding="ascii")
    )
    arena_path = into / RESTORE_ARENA_FILENAME
    if paths != {arena_path}:
        raise ValueError(
            f"the restored field table names {sorted(map(os.fspath, paths))} rather "
            f"than {os.fspath(arena_path)}; a restart read through a path this "
            "process did not just create could be any other run's state"
        )
    scalars = _parse_scalar_table(
        (into / RESTORE_SCALAR_TABLE_FILENAME).read_text(encoding="ascii")
    )
    declared = {carrier["name"] for carrier in receipt["profile"]["carriers"]}
    if set(shapes) != declared:
        missing = sorted(declared - set(shapes))
        extra = sorted(set(shapes) - declared)
        raise ValueError(
            f"the restored envelope carries {len(shapes)} arrays and the receipt's "
            f"profile declares {len(declared)}; missing {missing}, unexpected "
            f"{extra}. A resume from a partial state would continue from words "
            "this run never held"
        )
    return RestoredCartesianRestart(
        receipt, restore_receipt, into, arena_path,
        MappingProxyType(offsets), MappingProxyType(shapes),
        MappingProxyType(scalars),
        int(restore_receipt["step"]),
        _from_f64_bits(str(restore_receipt["model_time_f64_bits"])),
    )


# ---------------------------------------------------------------- restore


def require_resumable(
    restored: RestoredCartesianRestart, runtime: Any,
    profile: CartesianRestartProfile,
) -> None:
    """Refuse a restart whose words do not mean what this runtime would read.

    README.restart:20-23 says minor configuration changes across a restart are
    fine and major ones "could cause very bad things to happen".  This is that
    line drawn by value: the run length and the publication cadences may
    change, because they name when to stop and what to write, not what the
    carriers are.  The grid, the carrier inventory, the timestep, the frame
    velocity, the sounding, the initializer and the compiled operator set may
    not, because each of them changes what the persisted words mean.
    """

    receipt = restored.receipt
    # A parent run resumes a v1 or a v2 parent restart; any other profile (a
    # nest child's) resumes only a restart of its own id.
    admitted = (
        CARTESIAN_RESTART_PROFILE_IDS if profile.profile_id == CARTESIAN_RESTART_PROFILE_ID
        else (profile.profile_id,)
    )
    if receipt["profile"].get("profile_id") not in admitted:
        raise ValueError(
            f"the restart was written under profile {receipt['profile'].get('profile_id')!r}; "
            f"this run resumes {' or '.join(admitted)} restarts, and any other carries "
            "an inventory it would read under the wrong names"
        )
    # The swath group's shapes follow the run's length and are applied by
    # their own layout (read_restored_swaths); the state must not change.
    persisted = state_profile_sha256(receipt["profile"])
    if persisted != profile.state_sha256:
        raise ValueError(
            f"the restart was written under restart profile {receipt['profile_sha256']} "
            f"(state {persisted}) and this run prices state {profile.state_sha256}; the "
            "grid extent or the carrier inventory has changed, so the persisted arena "
            "would be read onto differently shaped allocations and every word after "
            "the first difference would land under another carrier's name"
        )
    if float(receipt["dt_s"]) != float(runtime.config.dt_s):
        raise ValueError(
            f"the restart was written at dt_s={float(receipt['dt_s'])!r} and this "
            f"run declares {float(runtime.config.dt_s)!r}; the model clock and the "
            "cm1.F:3512 domain-location integral are both a step count times this "
            "number, so the resumed run would place every later frame at a model "
            "time the carriers do not belong to"
        )
    # base.F's base state, the grid metrics and the compact profiles are the
    # part of the model this restart deliberately does not carry
    # (README.namelist:1512-1516), on the understanding that the initializer
    # rebuilds them.  That only holds while the initializer and its input are
    # the same ones: a resume under another sounding would put carriers that
    # evolved above one base state on top of another, and every later
    # buoyancy, pressure and moisture term would be computed against a profile
    # the carriers have never seen.
    persisted_source = dict(receipt["source_identity"])
    live_source = dict(runtime.source_identity)
    if persisted_source != live_source:
        differing = sorted(
            name for name in set(persisted_source) | set(live_source)
            if persisted_source.get(name) != live_source.get(name)
        )
        raise ValueError(
            f"the restart was written from source identity {persisted_source} and "
            f"this run builds {live_source}; {differing} differ. This restart does "
            "not carry the base state, the grid metrics or the compact profiles, "
            "because README.namelist:1512-1516 records that CM1 does not need "
            "them for a bit-identical restart -- but only when the initializer "
            "rebuilds the same ones, and here it would not"
        )
    execution = str(runtime.runtime_source_identity["aggregate_sha256"])
    if str(receipt["execution_identity_sha256"]) != execution:
        raise ValueError(
            f"the restart was written by operator set "
            f"{receipt['execution_identity_sha256']} and this process is {execution}; "
            "a restart continues a trajectory word for word, and a changed kernel "
            "produces a different next step from the same state, so the continued "
            "run would not be the run that was interrupted"
        )
    for name, declared in (
        (MOVE_SPEED_X_NAME, cm1_real(runtime.config.u_move_m_s)),
        (MOVE_SPEED_Y_NAME, cm1_real(runtime.config.v_move_m_s)),
    ):
        persisted = restored.f64(name)
        if persisted != declared:
            raise ValueError(
                f"the restart was written at {name}={persisted!r} m/s and this run "
                f"declares {declared!r} m/s; base.F:2667 subtracted the persisted "
                "value from every base-state wind word, so the restored carriers "
                "are relative to a frame this run is not in"
            )


def restored_cadence_tokens(restored: RestoredCartesianRestart) -> dict[str, int]:
    """The three cadence tokens, each required to equal the restart's own step."""

    tokens = {
        name: restored.u64(name)
        for name in (PHYSICS_STEP_NAME, SOLVE3_STEP_NAME, CLOSURE_STEP_NAME)
    }
    wrong = {name: value for name, value in tokens.items() if value != restored.step}
    if wrong:
        raise ValueError(
            f"the restart's cadence tokens {wrong} do not agree with its step "
            f"{restored.step}; the next large step would refuse its own cadence "
            "check after the device had been opened and the whole arena uploaded"
        )
    return tokens


def upload_restored_carriers(
    runtime: Any, restored: RestoredCartesianRestart,
    profile: CartesianRestartProfile, extra_carriers: Mapping[str, Any] | None = None,
) -> int:
    """Write every restored carrier into its live allocation, word for word.

    ``restart_read.F`` reading into the arrays ``restart_write.F`` wrote from.
    Returns the bytes written, which must be the profile's whole payload.
    """

    import numpy

    cp = runtime.cp
    live = _restart_carriers(runtime, profile, extra_carriers)
    float32 = cp.dtype("float32")
    restored_bytes = 0
    with runtime.stream:
        with restored.arena_path.open("rb", buffering=0) as stream:
            for carrier in profile.carriers:
                target = live[carrier.name]
                shape = tuple(restored.shapes[carrier.name])
                if shape != carrier.shape:
                    raise ValueError(
                        f"restored carrier {carrier.name} has extent {shape} and this "
                        f"runtime holds {carrier.shape}; writing it would overrun or "
                        "underfill the live allocation"
                    )
                if tuple(target.shape) != carrier.shape:
                    raise ValueError(
                        f"live carrier {carrier.name} holds {tuple(target.shape)} and "
                        f"the restart profile declares {carrier.shape}; restoring would "
                        "fill only part of the allocation and leave the rest -- for a "
                        "double-moment q3d, its number concentrations -- at stale words"
                    )
                if target.dtype != float32:
                    raise ValueError(
                        f"live carrier {carrier.name} is {target.dtype}, not float32; "
                        "the envelope's binary32 words would be reinterpreted"
                    )
                if not bool(target.flags.c_contiguous):
                    raise ValueError(
                        f"live carrier {carrier.name} is a strided view; restoring "
                        "words in address order would write halo and neighbouring "
                        "carrier storage"
                    )
                stream.seek(restored.offsets[carrier.name])
                flat = target.reshape(-1)
                words = carrier.words
                for start in range(0, words, FORECAST_ARENA_TRANSFER_CHUNK_WORDS):
                    stop = min(start + FORECAST_ARENA_TRANSFER_CHUNK_WORDS, words)
                    raw = stream.read((stop - start) * 4)
                    if len(raw) != (stop - start) * 4:
                        raise ValueError(
                            f"the restored arena ended inside carrier {carrier.name}; "
                            "a truncated restart would resume from uninitialized words"
                        )
                    host = numpy.frombuffer(raw, dtype="<f4")
                    flat[start:stop] = cp.asarray(host)
                    restored_bytes += len(raw)
    runtime.stream.synchronize()
    if restored_bytes != profile.payload_nbytes:
        raise RuntimeError(
            f"restored {restored_bytes} of {profile.payload_nbytes} declared carrier "
            "bytes; the restart and the runtime disagree about the carried state"
        )
    return restored_bytes


def read_restored_swaths(restored: RestoredCartesianRestart) -> SwathRestartState | None:
    """The swath words a restart carries, read from its restored arena to the host.

    ``restart_read.F:1064-1126``.  They are not uploaded here: the swaths are
    opened with the run's length at the start of the integration
    (``CudaCartesianRuntime.open_tornado_diagnostics``), and the words wait on
    the runtime until then.  None for a restart that carries none.
    """

    import numpy

    layout = restored.swath_layout
    if layout is None:
        return None
    rows = layout.carriers()
    missing = sorted(name for name, *_ in rows if name not in restored.shapes)
    if missing:
        raise ValueError(
            f"the restart's profile lays out swath carriers {missing} its envelope does "
            "not hold; a swath resumed from part of its words is not the run's extremum"
        )
    words: dict[str, Any] = {}
    with restored.arena_path.open("rb", buffering=0) as stream:
        for name, _meaning, _units, _stagger, shape, _key in rows:
            if tuple(restored.shapes[name]) != tuple(shape):
                raise ValueError(
                    f"restored swath carrier {name} has extent {tuple(restored.shapes[name])} "
                    f"and its layout {tuple(shape)}"
                )
            count = _words(shape)
            stream.seek(restored.offsets[name])
            raw = stream.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"the restored arena ended inside swath carrier {name}")
            words[name] = numpy.frombuffer(raw, dtype="<f4").reshape(shape).astype(numpy.float32)
    origin = layout.canvas_origin_cells_xy
    if origin is not None:
        carried = (restored.i64(SWATH_RESTART_ORIGIN_X), restored.i64(SWATH_RESTART_ORIGIN_Y))
        if carried != tuple(origin):
            raise ValueError(
                f"the restart's ground canvas origin is {carried} in its envelope and "
                f"{tuple(origin)} in its profile; the canvas cannot be placed on the ground"
            )
    index = words[SWATH_RESTART_INDEX]
    if not bool(numpy.all(index == numpy.floor(index))):
        raise ValueError("the restored per-step vortex index holds a word that is not a whole cell")
    return SwathRestartState(
        layout=layout,
        step=int(restored.step),
        planes=MappingProxyType({
            name: words[SWATH_RESTART_PLANE.format(name=name)] for name in layout.names
        }),
        canvases=MappingProxyType({} if origin is None else {
            name: words[SWATH_RESTART_CANVAS.format(name=name)] for name in layout.names
        }),
        series=MappingProxyType({
            name: words[SWATH_RESTART_SERIES.format(name=name)] for name in SERIES_NAMES
        }),
        vortex_index=index.astype(numpy.int32),
        domain_location=binary32_chunks_as_binary64(words[SWATH_RESTART_LOCATION]),
        updates=restored.u64(SWATH_RESTART_UPDATES),
        opened_step=restored.u64(SWATH_RESTART_OPENED),
        first_step=MappingProxyType({
            name: restored.u64(SWATH_RESTART_FIRST.format(name=name)) for name in layout.names
        }),
        residual_m=(restored.f64(SWATH_RESTART_RESIDUAL_X), restored.f64(SWATH_RESTART_RESIDUAL_Y)),
    )


def swath_resume_record(
    restored: RestoredCartesianRestart, config: Any, state: SwathRestartState | None,
) -> dict[str, Any]:
    """What a resume does with the swaths, and the warning when one restarts here."""

    selection = getattr(config, "diagnostics", None)
    wanted = tuple(selection.swath_names) if selection is not None and selection.active else ()
    carried = () if state is None else tuple(state.layout.names)
    step = int(restored.step)
    restart_here = [name for name in wanted if name not in carried] if step > 0 else []
    record: dict[str, Any] = {
        "restart_profile": restored.profile_id,
        "carried": list(carried),
        "selected": list(wanted),
        "continued": [name for name in wanted if name in carried],
        "restart_here": restart_here,
        "not_continued": [name for name in carried if name not in wanted],
        "carried_state": None if state is None else state.record(),
        "warning": None,
    }
    if restart_here:
        number = int(restored.receipt.get("restart_number", 0))
        why = (
            f"restart {number:06d} was written before GM1 carried the swaths in its "
            f"restarts ({CARTESIAN_RESTART_PROFILE_ID_V1})"
            if restored.profile_id == CARTESIAN_RESTART_PROFILE_ID_V1
            else f"the run that wrote restart {number:06d} did not keep them"
        )
        record["warning"] = (
            f"swaths restart here: {', '.join(restart_here)} hold the extremes since "
            f"t={restored.model_time_s:g} s (step {step}), not since the start of the "
            f"run, because {why}"
        )
    return record


def restore_cartesian_restart(
    runtime: Any, restored: RestoredCartesianRestart, *,
    profile: CartesianRestartProfile | None = None,
) -> dict[str, Any]:
    """Put one restored state back into a live runtime, word for word.

    This is ``restart_read.F`` reading into the arrays ``restart_write.F``
    wrote from: the same allocations, the same words, no reconstruction.  The
    clock is continued rather than restarted (``restart_read.F:419-420``), the
    dry-mass reference is overridden rather than recomputed
    (``restart_write.F:412``), and the positive-definite moisture accumulator
    is restored rather than zeroed (``restart_write.F:657``).
    """

    import numpy

    if profile is None:
        profile = cartesian_restart_profile(runtime.config)
    # The swaths are applied by their own layout once they are opened; the
    # state is uploaded here, word for word.
    profile = profile.state_only()
    if int(runtime.clock.completed_steps) != 0:
        raise RuntimeError(
            f"this runtime has already completed {int(runtime.clock.completed_steps)} "
            "large steps; restoring a restart over a stepped state would discard "
            "the trajectory it had already computed"
        )
    require_cartesian_restart_boundary(runtime)
    require_resumable(restored, runtime, profile)
    # Every scalar is decoded and admitted before one device word is touched.
    # restart_read.F:419-420's pair is the one that can refuse -- RuntimeClock
    # checks the running integral against n*dt*umove -- and a refusal after the
    # carriers had been overwritten would leave a runtime holding half a
    # restored state and half an initial condition.
    adaptive = getattr(runtime, "adaptive_frame", None)
    persisted_controller = any(
        name.startswith(ADAPTIVE_MOVE_RESTART_PREFIX) for name in restored.scalars
    )
    if (adaptive is not None) != persisted_controller:
        raise ValueError(
            "this restart "
            + ("carries" if persisted_controller else "carries no")
            + " CM1 adaptive moving-domain controller state and this run "
            + ("does not drive one" if adaptive is None else "drives one")
            + "; restart_write.F:377-401's fifteen words are the frame every "
            "restored wind is relative to, so continuing across that change "
            "would put carriers shifted into one frame under a run that "
            "believes it is in another"
        )
    clock = RuntimeClock.resume(
        runtime.config,
        completed_steps=restored.step,
        domainlocx_m=restored.f64(DOMAIN_LOCATION_X_NAME),
        domainlocy_m=restored.f64(DOMAIN_LOCATION_Y_NAME),
        adaptive_frame=adaptive,
    )
    mass1 = restored.f64(MASS1_SCALAR_NAME)
    asq = [restored.f64(ASQ_SCALAR_FORMAT.format(index=index)) for index in range(ASQ_LENGTH)]
    tokens = restored_cadence_tokens(restored)
    swaths = read_restored_swaths(restored)
    swath_record = swath_resume_record(restored, runtime.config, swaths)
    cp = runtime.cp
    restored_bytes = upload_restored_carriers(runtime, restored, profile)

    # restart_read.F:397-421: the controller's own fifteen words, restored
    # before the clock is installed so the integral the clock publishes is the
    # controller's and not a fresh zero.  cm1.F:3646 also skips the centre
    # diagnosis on the step a run resumes on, which the restored step number
    # arms.
    if adaptive is not None:
        adaptive.restore(restored)
        clock.domainlocx_m = float(adaptive.domainlocx_m)
        clock.domainlocy_m = float(adaptive.domainlocy_m)
    # restart_read.F:419-420: the integral is continued, not restarted.
    runtime.clock = clock
    # restart_write.F:412: mass1 is the reference apmasscon corrects every
    # later step back to.  Recomputing it here would silently retarget the
    # rest of the integration.
    runtime.finish.initial_mass1_f64 = mass1
    runtime.dynamics.mass1_f64 = mass1
    # restart_write.F:657.
    with runtime.stream:
        runtime.finish.asq[...] = cp.asarray(numpy.asarray(asq, dtype="<f8"))
    runtime.finish.completed_physics_step = tokens[PHYSICS_STEP_NAME]
    runtime.finish.completed_solve3_step = tokens[SOLVE3_STEP_NAME]
    # cuda_cartesian_closure refuses any step that is not `_held_step + 1`.
    runtime.closure._held_step = tokens[CLOSURE_STEP_NAME]
    runtime.closure._pending_step = None
    runtime.closure._tke_stage = 3
    # src/cm1.F:2663 and src/turb.F:1460-1466: the step a run resumes on
    # skips the two-part profiles, exactly as CM1's restarted flag does.
    two_part = getattr(runtime.closure, "two_part", None)
    if two_part is not None:
        two_part.mark_resumed()
    runtime.stream.synchronize()
    require_cartesian_restart_boundary(runtime)
    # restart_read.F:1064-1126: continued when the swaths open
    # (CudaCartesianRuntime.open_tornado_diagnostics), sized for the run.
    runtime.restored_swaths = swaths if swath_record["continued"] else None
    record = restored.record()
    record["swaths"] = swath_record
    record["restored_carrier_bytes"] = restored_bytes
    record["mass1"] = mass1
    record["asq"] = asq
    record["next_history_index"] = restored.u64(HISTORY_COUNTER_NAME)
    record["restart_number"] = restored.u64(RESTART_COUNTER_NAME)
    record["claim_limit"] = CARTESIAN_RESTART_CLAIM_LIMIT
    return record


__all__ = (
    "ARENA_FILENAME",
    "ASQ_LENGTH",
    "ASQ_SCALAR_FORMAT",
    "CARTESIAN_RESTART_AUTHORITY",
    "CARTESIAN_RESTART_CLAIM_LIMIT",
    "CARTESIAN_RESTART_PROFILE_ID",
    "CARTESIAN_RESTART_PROFILE_IDS",
    "CARTESIAN_RESTART_PROFILE_ID_V1",
    "CARTESIAN_RESTART_PROFILE_SCHEMA",
    "CARTESIAN_RESTART_PUBLICATION_SCHEMA",
    "CARTESIAN_RESTART_PUBLICATION_SCHEMAS",
    "CARTESIAN_RESTART_PUBLICATION_SCHEMA_V1",
    "CARTESIAN_RESTART_RESUME_SCHEMA",
    "CARTESIAN_RESTART_STATUS",
    "CARTESIAN_RESUME_STATUS",
    "CLOSURE_STEP_NAME",
    "CM1_RESTART_NAMELIST_ANSWERS",
    "ENVELOPE_FILENAME",
    "FIELD_TABLE_FILENAME",
    "FORECAST_CHECKPOINT_MAGIC",
    "HISTORY_COUNTER_NAME",
    "MASS1_SCALAR_NAME",
    "PHYSICS_STEP_NAME",
    "RECEIPT_FILENAME",
    "RESTART_COUNTER_NAME",
    "RESTART_DIRECTORY_FORMAT",
    "RESTART_SCALAR_AUTHORITY",
    "SCALAR_TABLE_FILENAME",
    "SOLVE3_STEP_NAME",
    "CartesianRestartCarrier",
    "CartesianRestartProfile",
    "CartesianRestartSnapshot",
    "RestoredCartesianRestart",
    "cartesian_restart_profile",
    "commit_cartesian_restart",
    "envelope_header_sha256",
    "read_cartesian_restart",
    "read_envelope_header",
    "read_restored_swaths",
    "require_cartesian_restart_boundary",
    "require_restart_receipt",
    "require_resumable",
    "restart_arena_disposition",
    "restart_scalars",
    "restore_cartesian_restart",
    "restored_cadence_tokens",
    "runtime_restart_profile",
    "snapshot_cartesian_restart",
    "state_profile_record",
    "state_profile_sha256",
    "swath_resume_record",
    "upload_restored_carriers",
    "verify_restart_envelope_carries_arena",
    "write_cartesian_restart",
    "write_restart_receipt",
)
