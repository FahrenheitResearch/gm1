"""The vortex-following child sub-volume, written every child step.

WHY THIS EXISTS
---------------
A trajectory into a tornado core is an integral of the wind along a moving
point, and the accuracy of that integral is set by how often the wind is
known.  Everything GM1 could offer before this module knows the wind at the
*history* cadence: the child frames :mod:`cm1core.nest_parcel_backward` reads
are tens of seconds apart, and between two of them the offline wind is a
linear interpolation whose error is bounded by ``(T**2/8) max|u_tt|`` -- on a
20 m child around a 117 m/s core that measured out at 800-1200 m of position
error, which is twenty core radii.  Forward parcels seeded on a lattice do not
close the gap either: a 40 m seed lattice does not sample a 40-60 m core, and
three measured attempts (100k, 140k and 135k parcels) put at most a few dozen
parcels inside it, none of which ever recorded more than half the grid's own
vorticity.

The remedy every study that has answered this question uses -- Orf et al.
(2017) is the best known -- is to write a small sub-volume around the vortex
at, or near, the model's own step and integrate backward through *that*.  This
module is that sub-volume, with one addition: the words it writes are not the
completed state a history frame holds but the **advecting velocity the model
itself used**, the acoustic small-step average CM1 saves at
``nrk == nrkmax`` (``src/solve2.F:1290-1301``) and hands to ``parcel.F``
(``src/solve3.F:768-769``).  Written every child step, that stream lets the
offline integrator step backward through the model's own wind words with **no
time interpolation at all** (:mod:`cm1core.nest_parcel_backward`'s
``wind_source = "model-step"``), which removes the dominant error term rather
than estimating it.

AUTHORITY
---------
CM1 r21.1 has no nesting (``src/cm1core/records/CM1-NEST-SEAM.md`` section 1)
and its own high-frequency writer ``src/hifrq.F:21`` writes the whole domain
at a fixed place, driven by two words ``src/cm1.F:2645-2647`` hard-codes.  The
*placement* and the *cadence* here are therefore GM1's, and every one of them
is in :data:`GM1_CHOICES` and in the stream's receipt.  What is CM1's, and is
cited rather than re-derived:

* ``src/solve2.F:1290-1301`` and ``src/solve3.F:438-469``: the advecting
  velocity this stream carries is the acoustic small-step average saved at the
  final RK stage -- the same words :meth:`cm1core.nest_parcels.NestParcelTracker.capture_child`
  snapshots, taken at the same moment in the same child step.
* ``src/parcel.F:59-96``: the ``k = 0`` ghost of ``u`` and ``v``, which reads
  only levels 1, 2 and 3 and is therefore local to the bottom of any block
  that reaches the ground (:func:`cm1core.nest_parcels.fill_velocity_bottom_ghost`).
* ``src/parcel.F:153-461``: the sampler the offline integrator runs on this
  stream, unchanged, on the sub-volume's own :class:`cm1core.nest_parcels.ParcelDeck`.
* ``src/turb.F:1161,1233-1237`` with ``src/cm1.F:2192-2194``: ``zntmp``, which
  a ``bbc = 3`` child needs for ``parcel.F:470-515``'s log layer below
  ``zh(1)`` -- exactly where a tornado's inflow parcels are.

WHAT IS GM1'S, AND WHY
----------------------
* **The window is a pure slice.**  Every field is a strided copy of the
  child's own padded carrier: no interpolation, no boundary rule, no kernel.
  That is the point -- a stream that re-derived anything would no longer be
  the model's words.  It is bought by two refusals:

  - the window, *including its three ghost columns each way*, must lie inside
    the child's free interior (the lattice less ``spec_zone + relax_zone``
    rows), so every word it holds came from the child's own integration and
    none from the parent's interpolated boundary rows;
  - the window's top must be strictly below the child's, so the level above
    its top is a real child level rather than ``tbc``'s free-slip
    extrapolation.  A lid where the child has none would bend every
    trajectory that reached it.

  The one word that is not a slice is the ``k = 0`` ghost of ``u`` and ``v``,
  which the child's live carriers do not hold and which ``parcel.F:59-96``
  builds locally from levels 1-3.

* **The window reaches the ground.**  ``k0`` is not a key.  The air a tornado
  ingests is in the lowest few hundred metres, and a sub-volume that began
  above the surface could not integrate one of those trajectories.

* **The centre.**  With ``follow = "tornado-criterion"`` the window centres on
  the centre :mod:`cm1core.nest_tornado_criterion` found at the last parent
  step -- the first maximum of the child's 10 m corner vorticity -- which is
  one parent step of lag (a few tens of metres of vortex motion against a
  kilometre-scale half-width; the lag is recorded).  A sample that finds no
  centre holds the last one, and the run records how many records were held.
  Before the first centre is found nothing is written, and that too is counted:
  a window cannot follow a vortex that has not been located.

* **Clamping, not refusal, at run time.**  A centre that would push the window
  past the free interior is clamped, and the record says so.  Refusing would
  end a run because the vortex approached the child's edge; silently moving it
  would be worse, so every record carries its own index origin and ground
  origin and nothing is ever placed by inference.

* **One stream file, not one directory per record.**  Ten records a second is
  six hundred directories a model minute; the stream is one append-only arena
  with a fixed record stride, one TSV index row per record and one receipt.

* **The stream is bounded in model time and in bytes.**  ``start_s``/``end_s``
  bound the interval and ``budget_bytes_per_model_minute`` is checked at the
  door against the size the declared geometry actually produces, so a run
  cannot discover its disk cost by filling the disk.

UNVALIDATED
-----------
See :data:`UNVALIDATED_MARKER`.  CM1 r21.1 has nothing to difference this
against; what is proved is that the published words are the child's own words
(the cut is an index identity, checked against a whole-child reference) and
that the offline integrator run on the stream reproduces a closed-form
trajectory to the scheme's own order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import cuda_graphs
from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY
from .nest_parcels import ParcelDeck, fill_velocity_bottom_ghost

#: v2: ``uadv``/``vadv``/``wadv``; ``rho``/``prs``/``th``/``qv``/``qhyd`` on CM1
#: levels ``1..nz+1`` with one ring; the held ``uten``/``vten``/``wten``;
#: ``znt`` when ``bbc = 3``.  v1 carried a padded ``rho``/``prs`` the live
#: child does not hold (its ``rho``/``prs`` are physical ``dynamics.fields``).
#: v2 is what every component but Morrison writes, byte for byte as before v3.
NEST_CHILD_WINDOW_SCHEMA = "gm1.nest-child-window-stream/v2"
#: v3 is v2 plus either or both of two additions, each declared in the
#: receipt so a reader never infers it:
#:
#: * **species** (Morrison, ``morrison-mp10``): ``qhyd``'s slot holds the four
#:   condensate species ``qc``, ``qr``, ``qis = qi + qs`` and ``qg`` (hail when
#:   ``ihail = 1``, morrison.F:169-170,381-384), so a reader can tell cloud from
#:   rain from ice from hail and form reflectivity at the window's cadence.
#:   ``qhyd`` is rebuilt by the reader (:func:`rebuild_qhyd`).  The receipt
#:   carries ``species``, ``schema_version`` and ``field_names``.
#: * **rotation**: a ``rotation`` header, written only when the child
#:   integrates CM1's ``icor = 1`` Coriolis term: the f-plane ``fcor``, the
#:   frame velocity ``solve2.F:645,647`` adds back inside the term, and -- with
#:   ``lspgrad = 1`` -- the per-level large-scale pressure-gradient words
#:   ``solve1.F:590-612`` puts into the held tendency.  A v2 reader handed a
#:   rotating stream would drop the planetary terms from every budget without a
#:   word, which is why the name changes.
#:
#: A child with neither (every component but Morrison, no Coriolis) still
#: writes v2 byte for byte; a v3 receipt with neither is refused, because it
#: would be a v2 stream under the wrong name.
NEST_CHILD_WINDOW_SCHEMA_V3 = "gm1.nest-child-window-stream/v3"
#: The rotating stream's name: the same v3 (a Morrison child with Coriolis
#: writes one v3 stream carrying both the species and the rotation header).
NEST_CHILD_WINDOW_ROTATING_SCHEMA = NEST_CHILD_WINDOW_SCHEMA_V3
NEST_CHILD_WINDOW_SCHEMAS: tuple[str, ...] = (NEST_CHILD_WINDOW_SCHEMA,
                                              NEST_CHILD_WINDOW_SCHEMA_V3)
NEST_CHILD_WINDOW_DIRECTORY = "nest-child-window"
WINDOW_ARENA_FILENAME = "window-stream.f32"
WINDOW_INDEX_FILENAME = "window-records.tsv"
WINDOW_RECEIPT_FILENAME = "window-stream.json"

#: The lateral ghost width of the child's padded carriers, which the window
#: inherits word for word (``cuda_cartesian_dynamics`` pads ``halo_xy = 3``,
#: :data:`cm1core.cuda_psolver3_terrain_open.CM1_OPEN_NGXY`).
NG = CM1_OPEN_NGXY

#: The fields one record carries, in the arena's own order: ``(name, stagger,
#: units, meaning)``.  ``stagger`` names the record layout
#: (:meth:`WindowBox.shapes`): ``u``/``v``/``w`` are the padded velocity blocks
#: with three ghost columns each way and the ``parcel.F`` vertical ghosts;
#: ``s`` is a scalar block of CM1 levels ``1..nz+1`` with one ring of the
#: child's own neighbouring cells; ``tu``/``tv``/``tw`` are the held momentum
#: tendencies on their own faces over the same ring and levels.
WINDOW_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("uadv", "u", "m/s",
     "solve2.F:1290-1301 acoustic small-step average at nrk==nrkmax (rru), the wind parcel.F advects with"),
    ("vadv", "v", "m/s",
     "solve2.F:1290-1301 acoustic small-step average at nrk==nrkmax (rrv)"),
    ("wadv", "w", "m/s",
     "solve2.F:1290-1301 acoustic small-step average at nrk==nrkmax (rrw)"),
    ("rho", "s", "kg/m3", "the child's live dry-air density rho (CM1 levels 1..nz+1, one ring)"),
    ("prs", "s", "Pa", "the child's live pressure prs (CM1 levels 1..nz+1, one ring)"),
    ("th", "s", "K",
     "the child's tha, the potential-temperature perturbation at the start of the step "
     "(th = th0 + tha, parcel.F:763)"),
    ("qv", "s", "kg/kg", "the child's qa(:,:,:,nqv) at the start of the step"),
    ("qhyd", "s", "kg/kg",
     "qa(nqc)+qa(nqr)+qa(nqi)+qa(nqs)+qa(nqg) at the start of the step: the condensate "
     "parcel.F:745-775 subtracts as loading (qt - qv), summed over the same five species"),
    ("uten", "tu", "m/s2",
     "the held momentum tendency uten1 of this step (solve1.F:414-1620): with no Rayleigh "
     "layer in the window, turbu's subgrid stress plus any idiff diffusion"),
    ("vten", "tv", "m/s2", "the held momentum tendency vten1 of this step (solve1.F)"),
    ("wten", "tw", "m/s2", "the held momentum tendency wten1 of this step (solve1.F)"),
)
#: Written only when ``bbc = 3``.
WINDOW_SURFACE_FIELD = (
    "znt", "2d", "m",
    "turb.F:1161,1233-1237 zntmp, which parcel.F:470-515's log layer below zh(1) reads",
)
#: The moisture rows ``qhyd`` sums, in CM1's WSM6/Morrison mass-species order
#: (``nqv = 1`` .. ``nqg = 6``), which is the order parcel.F:745-775 reads.
CONDENSATE_ROWS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Schema v3's species, which take ``qhyd``'s slot in the record: ``(name,
#: stagger, units, meaning)`` exactly as :data:`WINDOW_FIELDS`.
WINDOW_SPECIES_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("qc", "s", "kg/kg",
     "qa(nqc), cloud water, at the start of the step (param.F:3414-3473 row 2)"),
    ("qr", "s", "kg/kg", "qa(nqr), rain, at the start of the step (param.F:3414-3473 row 3)"),
    ("qis", "s", "kg/kg",
     "qa(nqi)+qa(nqs), cloud ice plus snow, at the start of the step (param.F:3414-3473 "
     "rows 4 and 5); one word because both are the low-density ice the window's readers "
     "treat alike, and the pair costs a whole field"),
    ("qg", "s", "kg/kg",
     "qa(nqg) at the start of the step (param.F:3414-3473 row 6): the dense precipitating "
     "ice, hail when ihail=1 and graupel when ihail=0 (morrison.F:169-170,381-384)"),
)
#: Which CM1 moisture scalars each species word sums, by name and in CM1's own
#: row order, so a component's rows are found by name and never by position.
WINDOW_SPECIES_SOURCES: Mapping[str, tuple[str, ...]] = {
    "qc": ("qc",), "qr": ("qr",), "qis": ("qi", "qs"), "qg": ("qg",),
}
#: The components whose window is schema v3.  Every other admitted component
#: keeps schema v2 byte for byte.
WINDOW_SPECIES_COMPONENTS: frozenset[str] = frozenset({"morrison-mp10"})
#: Schema v3's fields, in the arena's own order: v2's with ``qhyd``'s slot
#: taken by the species.
WINDOW_FIELDS_V3: tuple[tuple[str, str, str, str], ...] = tuple(
    field for entry in WINDOW_FIELDS
    for field in (WINDOW_SPECIES_FIELDS if entry[0] == "qhyd" else (entry,))
)
#: The CM1 mass prefix both schemas read (param.F:3414-3473, :3641-3678).
_CM1_MASS_NAMES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")
CM1_AUTHORITY: Mapping[str, str] = {
    "advecting_velocity": "src/solve2.F:1290-1301; src/solve3.F:438-469",
    "parcel_call_site": "src/solve3.F:768-769",
    "bottom_ghost": "src/parcel.F:59-96",
    "sampler": "src/parcel.F:153-461",
    "semi_slip_log_layer": "src/parcel.F:470-515",
    "roughness_carrier": "src/turb.F:1161,1233-1237; src/cm1.F:2192-2194",
    "whole_domain_high_frequency_writer": "src/hifrq.F:21-1790; src/cm1.F:2645-2647",
    "deck": "src/param.F:8162-8209 (zf, zh, mf ghosts)",
    "nesting": "GM1 EXTENSION: CM1 r21.1 has no nesting (records/CM1-NEST-SEAM.md section 1)",
}

GM1_CHOICES: Mapping[str, str] = {
    "coordinates": (
        "metres from the parent grid origin (parent xf(1), yf(1), ground) in the parent's "
        "translating frame, the frame [lagrangian.nest_child] parcels are published in"
    ),
    "cut": (
        "every field is a strided copy of the child's own padded carrier; the only word "
        "formed here is the k=0 ghost of u and v (parcel.F:59-96), which is local to the "
        "bottom of any block that reaches the ground"
    ),
    "extent": (
        "the window and its three ghost columns each way lie inside the child's free "
        "interior, and its top is strictly below the child's, so no published word is a "
        "parent-interpolated boundary row or a tbc extrapolation"
    ),
    "ground": "k0 is fixed at the ground; a sub-volume above it could integrate no inflow trajectory",
    "centre": (
        "the tornado criterion's centre at the last parent step, held when a sample finds "
        "none, clamped (and recorded as clamped) when it would push the window past the "
        "free interior"
    ),
    "cadence": "every stride_child_steps completed child steps, between start_s and end_s inclusive",
    "layout": "one append-only arena with a fixed record stride, one TSV row per record, one receipt",
    "write": (
        "the device-to-host copy runs on the stepping thread and the write, the SHA-256 and "
        "the index row on one worker thread with a bounded buffer pool"
    ),
}

UNVALIDATED_MARKER = (
    "UNVALIDATED against CM1: CM1 r21.1 has no nesting and no sub-volume writer, so a "
    "vortex-following child window has no CM1 counterpart.  What is checked here is that "
    "the cut is an index identity against the whole child's own carriers, that the k=0 "
    "ghost is parcel.F:59-96's own words, and that an offline integration through the "
    "stream reproduces a closed-form trajectory to the RK2 scheme's order."
)


class NestChildWindowError(ValueError):
    """A window that cannot hold the words it claims to."""


@dataclass(frozen=True)
class WindowLayout:
    """Which schema a component's window writes, and the moisture rows it reads.

    ``species_rows`` maps each v3 species word to the zero-based ``qa`` rows it
    sums, found by name in the component's own CM1 table; it is empty for v2.
    ``numq`` is the component's own row count, which the live carrier must
    match exactly for v3.
    """

    component_id: str
    schema: str
    fields: tuple[tuple[str, str, str, str], ...]
    species_rows: Mapping[str, tuple[int, ...]]
    numq: int
    axis_authority: str

    @property
    def species(self) -> bool:
        return bool(self.species_rows)

    @property
    def version(self) -> int:
        return 3 if self.species else 2

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"component_id": self.component_id, "schema": self.schema,
                               "schema_version": self.version}
        if self.species:
            out.update({
                "qa_rows_zero_based": {k: list(v) for k, v in self.species_rows.items()},
                "sources": {k: list(v) for k, v in WINDOW_SPECIES_SOURCES.items()},
                "numq": self.numq,
                "axis_authority": self.axis_authority,
                "qhyd": (
                    "not published: a reader rebuilds it as ((qc + qr) + qis) + qg in float32 "
                    "(cm1core.nest_child_window.rebuild_qhyd), which differs from parcel.F:745-775's "
                    "sequential qc+qr+qi+qs+qg only by float rounding"
                ),
            })
        return out


def window_layout(component_id: Any) -> WindowLayout:
    """The window layout for one microphysics component, refusing what it cannot cut.

    A component in :data:`WINDOW_SPECIES_COMPONENTS` writes schema v3 with its
    species rows found **by name**; a species it does not carry is refused,
    because a word published under ``qc`` would otherwise hold another row.
    Every other component writes schema v2 and must carry CM1's six mass rows
    as its leading rows, which is what ``qhyd``'s fixed rows 1..5 assume.
    """

    from .moisture_species import moisture_species_profile

    if not isinstance(component_id, str):
        raise NestChildWindowError(
            f"the child names no microphysics component ({component_id!r}); the window cannot "
            "know which qa rows hold which species, and a cut by position would publish "
            "another component's rows under these names"
        )
    from .errors import CM1Refusal

    try:
        profile = moisture_species_profile(component_id)
    except CM1Refusal as error:
        # A dry child (``off``) or an unadmitted component holds no species
        # table; the window refuses it under its own code, at the door,
        # rather than failing at the first capture.
        raise NestChildWindowError(
            f"component {component_id!r} has no CM1 species table the window can read "
            f"({error.detail.code}); a window needs qv, qc, qr, qi, qs and qg by name"
        ) from error
    names = tuple(profile.names)
    if component_id in WINDOW_SPECIES_COMPONENTS:
        rows: dict[str, tuple[int, ...]] = {}
        for word, sources in WINDOW_SPECIES_SOURCES.items():
            missing = [name for name in sources if name not in profile.mass_names]
            if missing:
                raise NestChildWindowError(
                    f"component {component_id!r} carries no {', '.join(missing)} mass row, so "
                    f"schema v3's {word!r} (the sum of {' + '.join(sources)}) cannot be formed; "
                    "publishing it without them would name a species the model did not hold"
                )
            rows[word] = tuple(names.index(name) for name in sources)
        return WindowLayout(component_id, NEST_CHILD_WINDOW_SCHEMA_V3, WINDOW_FIELDS_V3,
                            rows, int(profile.numq), str(profile.axis_authority))
    if names[:len(_CM1_MASS_NAMES)] != _CM1_MASS_NAMES:
        raise NestChildWindowError(
            f"component {component_id!r} does not lead with CM1's qv, qc, qr, qi, qs, qg rows "
            f"({list(names[:6])}); schema v2's qhyd sums rows 1..5 in parcel.F:745-775's order "
            "and would sum other words as the condensate loading"
        )
    return WindowLayout(component_id, NEST_CHILD_WINDOW_SCHEMA, WINDOW_FIELDS, {},
                        int(profile.numq), str(profile.axis_authority))


def rebuild_qhyd(words: Mapping[str, Any]) -> Any:
    """Schema v3's condensate loading ``((qc + qr) + qis) + qg`` in float32.

    parcel.F:745-775 sums ``qc + qr + qi + qs + qg`` one row at a time; the
    stream has already added ``qi + qs`` into one word, so this sum groups the
    same five terms differently and agrees with parcel.F's to float rounding
    (a few units in the last place of the total), not bit for bit.
    """

    np = _np()
    f = np.float32
    qc, qr, qis, qg = (np.asarray(words[name], dtype=np.float32)
                       for name in ("qc", "qr", "qis", "qg"))
    return f(f(f(qc + qr) + qis) + qg)


def _np() -> Any:
    import numpy

    return numpy


def _host(value: Any) -> Any:
    return value.get() if hasattr(value, "get") else value


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WindowBox:
    """A sub-volume of the child: index origin, cell counts, published shapes.

    ``i0``/``j0`` are zero-based indices into the child's **physical**
    rectangle; ``nx``/``ny``/``nz`` are the window's own cell counts, with the
    vertical always starting at the ground (``k0 = 0``).
    """

    i0: int
    j0: int
    nx: int
    ny: int
    nz: int

    @property
    def i1(self) -> int:
        return self.i0 + self.nx

    @property
    def j1(self) -> int:
        return self.j0 + self.ny

    def shapes(self, *, surface: bool, species: bool = False) -> dict[str, tuple[int, ...]]:
        """Every published field's shape; ``species`` is schema v3's four words for ``qhyd``."""

        nx, ny, nz = self.nx, self.ny, self.nz
        out: dict[str, tuple[int, ...]] = {
            "uadv": (nz + 2, ny + 2 * NG, nx + 1 + 2 * NG),
            "vadv": (nz + 2, ny + 1 + 2 * NG, nx + 2 * NG),
            "wadv": (nz + 3, ny + 2 * NG, nx + 2 * NG),
        }
        ring = (nz + 1, ny + 2, nx + 2)
        condensate = tuple(name for name, *_ in WINDOW_SPECIES_FIELDS) if species else ("qhyd",)
        for name in ("rho", "prs", "th", "qv", *condensate):
            out[name] = ring
        out["uten"] = (nz + 1, ny + 2, nx + 3)
        out["vten"] = (nz + 1, ny + 3, nx + 2)
        out["wten"] = (nz + 2, ny + 2, nx + 2)
        if surface:
            out["znt"] = (ny + 2 * NG, nx + 2 * NG)
        return out

    def record_words(self, *, surface: bool, species: bool = False) -> int:
        return sum(math.prod(shape)
                   for shape in self.shapes(surface=surface, species=species).values())

    def record_bytes(self, *, surface: bool, species: bool = False) -> int:
        return 4 * self.record_words(surface=surface, species=species)

    def record(self) -> dict[str, Any]:
        return {"i0": self.i0, "j0": self.j0, "i1": self.i1, "j1": self.j1,
                "nx": self.nx, "ny": self.ny, "nz": self.nz, "k0": 0}


def window_cells(settings: Any, *, dx_m: float, dy_m: float, zf: Any) -> tuple[int, int, int]:
    """``(nx, ny, nz)``: the half-widths and top in cells of the child's deck.

    The horizontal count is the odd number of cells that covers the declared
    half-width each way, so the window is symmetric about its centre cell and
    a centre that lands on a cell boundary does not shift the volume by half a
    cell between two records.  The vertical count is the number of whole child
    layers at or below ``top_m``.
    """

    half_i = int(math.ceil(float(settings.half_width_x_m) / float(dx_m)))
    half_j = int(math.ceil(float(settings.half_width_y_m) / float(dy_m)))
    nx, ny = 2 * half_i + 1, 2 * half_j + 1
    nz = 0
    for k in range(1, len(zf) - 1):
        if float(zf[k + 1]) <= float(settings.top_m):
            nz = k
        else:
            break
    if nz < 4:
        raise NestChildWindowError(
            f"lagrangian.nest_child_window.top_m = {settings.top_m} m covers {nz} whole child "
            "layers; parcel.F:59-96's k=0 ghost reads levels 1, 2 and 3 and the sampler needs a "
            "layer above them, so a window shallower than four layers cannot be sampled at all"
        )
    return nx, ny, nz


def place_window(*, centre_x_m: float, centre_y_m: float, geometry: Any,
                 nx: int, ny: int) -> tuple[int, int, bool]:
    """``(i0, j0, clamped)`` for a window of ``nx`` by ``ny`` cells on ``centre``.

    The admissible index range keeps the window and its ``NG`` ghost columns
    inside the child's free interior.  A centre outside that range is clamped
    to it and the caller records the clamp.
    """

    width = int(geometry.spec_zone) + int(geometry.relax_zone) + NG
    lo_i, hi_i = width, int(geometry.nx) - width - nx
    lo_j, hi_j = width, int(geometry.ny) - width - ny
    if hi_i < lo_i or hi_j < lo_j:
        raise NestChildWindowError(
            f"a {nx} by {ny} cell window does not fit inside the child's free interior less "
            f"{NG} ghost columns ({geometry.nx} by {geometry.ny} cells, "
            f"{geometry.spec_zone} specified and {geometry.relax_zone} relaxation rows); every "
            "word it published would be a parent-interpolated boundary row"
        )
    want_i = int(math.floor((float(centre_x_m) - float(geometry.origin_x_m)) / float(geometry.dx_m))) - nx // 2
    want_j = int(math.floor((float(centre_y_m) - float(geometry.origin_y_m)) / float(geometry.dy_m))) - ny // 2
    i0 = min(max(want_i, lo_i), hi_i)
    j0 = min(max(want_j, lo_j), hi_j)
    return i0, j0, bool(i0 != want_i or j0 != want_j)


def window_deck(child: ParcelDeck, box: WindowBox) -> ParcelDeck:
    """The sub-volume as a CM1 grid in its own right.

    The horizontal origin moves to the window's own west and south faces; the
    vertical words are the child's own, truncated at the window's top, so
    ``zf``, ``zh`` and ``mf`` are the child's numbers and not a new deck.
    ``cgt`` is carried unchanged and is never used: the window's top ghost is a
    real child level, not an extrapolation, which is why ``top_m`` must stay
    below the child's top.  There is no periodic exit: a parcel that leaves the
    window leaves the volume the stream holds and stops there.
    """

    np = _np()
    nz = box.nz
    return ParcelDeck(
        name="child-window", nx=box.nx, ny=box.ny, nz=nz,
        dx_m=child.dx_m, dy_m=child.dy_m,
        minx_m=float(np.float32(child.minx_m + box.i0 * child.dx_m)),
        miny_m=float(np.float32(child.miny_m + box.j0 * child.dy_m)),
        dz_reference_m=child.dz_reference_m,
        zf=np.ascontiguousarray(child.zf[: nz + 3]),
        zh=np.ascontiguousarray(child.zh[: nz + 2]),
        mf=np.ascontiguousarray(child.mf[: nz + 3]),
        cgs=child.cgs, cgt=child.cgt, lateral=(2, 2, 2, 2), periodic=False,
        vertical_sha256=child.vertical_sha256,
    )


def cut_window(xp: Any, arrays: Mapping[str, Any], box: WindowBox, *,
               surface: Any = None,
               species_rows: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """The window's fields cut out of the child's carriers.

    ``arrays`` maps the child's carriers by name:

    * ``rru``/``rrv``/``rrw`` -- the child's live **padded** ``dynamics.arrays``,
      which at ``nrk == nrkmax`` hold the acoustic small-step average
      (``solve2.F:1290-1301``);
    * ``rho``/``prs``/``tha`` -- the child's **physical** ``(nz, ny, nx)``
      ``dynamics.fields``, and ``qa`` its physical ``(numq, nz, ny, nx)`` rows;
    * ``uten``/``vten``/``wten`` -- the child's **physical** held momentum
      tendencies ``dynamics.held`` (``u``: ``(nz, ny, nx+1)``, ``v``:
      ``(nz, ny+1, nx)``, ``w``: ``(nz+1, ny, nx)``).

    The scalar and tendency blocks keep CM1 levels ``1..nz+1`` (the level above
    the window's top is a real child level, because the door keeps the window
    below the child's top) and one ring of the child's own neighbouring cells,
    which every centred derivative needs and all it needs.  The window keeps
    ``NG`` columns clear of the child's own edge, so the ring is always inside
    the physical rectangle.

    ``surface`` is the child's **physical** ``(ny, nx)`` ``znt`` for a
    ``bbc = 3`` child, or ``None``; the window's ``zntmp`` is cut straight out
    of it, because :func:`cm1core.nest_parcels.pad_roughness`'s padded word at
    ``[j-1+ng, i-1+ng]`` is the child's own ``znt(i, j)`` everywhere the window
    can reach -- the window keeps ``NG`` columns clear of the child's own edge,
    so no published word is one of ``bc.F``'s boundary copies or
    ``cm1.F:2194``'s ``1.0e-6`` filler.

    Every returned array except ``qhyd`` is a **view**: the caller copies it out
    and must never write through one, because the words belong to the running
    model.  ``qhyd`` is the one word formed here, a sum of five species rows
    in parcel.F's own order, on the array module of the carriers.

    With ``species_rows`` (schema v3, :func:`window_layout`) ``qhyd`` is not
    formed; each species word is its ``qa`` row -- a view -- or, for ``qis``,
    the sum ``qi + qs`` of its two rows in CM1's row order.
    """

    i0, j0, nx, ny, nz = box.i0, box.j0, box.nx, box.ny, box.nz
    ks = slice(0, nz + 1)
    js, is_ = slice(j0 - 1, j0 + ny + 1), slice(i0 - 1, i0 + nx + 1)
    out = {
        "uadv": arrays["rru"][0:nz + 2, j0:j0 + ny + 2 * NG, i0:i0 + nx + 1 + 2 * NG],
        "vadv": arrays["rrv"][0:nz + 2, j0:j0 + ny + 1 + 2 * NG, i0:i0 + nx + 2 * NG],
        "wadv": arrays["rrw"][0:nz + 3, j0:j0 + ny + 2 * NG, i0:i0 + nx + 2 * NG],
        "rho": arrays["rho"][ks, js, is_],
        "prs": arrays["prs"][ks, js, is_],
        "th": arrays["tha"][ks, js, is_],
        "qv": arrays["qa"][0, ks, js, is_],
        "uten": arrays["uten"][ks, js, slice(i0 - 1, i0 + nx + 2)],
        "vten": arrays["vten"][ks, slice(j0 - 1, j0 + ny + 2), is_],
        "wten": arrays["wten"][slice(0, nz + 2), js, is_],
    }
    qa = arrays["qa"]
    if species_rows:
        for word, rows in species_rows.items():
            block = qa[rows[0], ks, js, is_]
            for row in rows[1:]:
                block = block + qa[row, ks, js, is_]
            out[word] = block
    else:
        total = qa[CONDENSATE_ROWS[0], ks, js, is_].copy()
        for row in CONDENSATE_ROWS[1:]:
            total = total + qa[row, ks, js, is_]
        out["qhyd"] = total
    if surface is not None:
        out["znt"] = surface[j0 - NG:j0 - NG + ny + 2 * NG, i0 - NG:i0 - NG + nx + 2 * NG]
    _ = xp
    return out


def fill_ring_bottom_ghost(block: Any, zh: Any) -> Any:
    """A ringed block of CM1 levels ``1..nz+1`` with a ``k = 0`` ghost below it.

    GM1 CHOICE.  The scalar and tendency blocks hold real child levels only,
    and a centred vertical derivative at level 1 needs a level below the
    ground.  The ghost is the linear extrapolation in the deck's own ``zh``
    through levels 1 and 2, so the centred derivative at level 1 is exactly
    the one-sided difference ``(s2 - s1)/(zh2 - zh1)`` -- exact for a field
    linear in height and inventing nothing beyond that.
    """

    np = _np()
    block = np.asarray(block, dtype=np.float32)
    z0, z1, z2 = (float(zh[k]) for k in (0, 1, 2))
    ratio = np.float32((z1 - z0) / (z2 - z1))
    ghost = np.float32(block[0] - ratio * np.float32(block[1] - block[0]))
    return np.concatenate([ghost[None], block], axis=0)


def fill_window_w_ghost(w: Any) -> None:
    """``w[0] = -w[2]``: the below-ground ``w`` ghost, by reflection.

    GM1 CHOICE, and a narrow one.  ``parcel.F:57-130`` fills ``ua(i,j,0)`` and
    ``va(i,j,0)`` and the top faces, and leaves ``wa(i,j,0)`` alone, because
    the parcel sampler never reads it: ``parcel.F:251-260``'s ``kflag`` is at
    least 1, and the ``w`` stencil reads ``kflag`` and ``kflag+1``.  **No
    trajectory in this stream depends on this word.**

    It is written for one consumer: the vorticity budget
    (:func:`cm1core.vorticity_budget.cpu_vorticity_budget`) forms
    ``dw/dz`` at the lowest scalar level from the ``w`` faces at index 0 and
    2, and reading a word the model leaves undefined there would put an
    arbitrary number into the stretching term of exactly the near-ground
    parcels the whole tool exists to follow.  The reflection ``w(zf(0)) =
    -w(zf(2))`` is the flat rigid lower boundary's own antisymmetry about
    ``zf(1) = 0``, and it is exact for a ``w`` linear in height, which is what
    the centred difference above it assumes anyway.
    """

    np = _np()
    w[0] = np.float32(0.0) - w[2]


def stream_bytes_per_model_minute(box: WindowBox, *, child_dt_s: float, stride: int,
                                  surface: bool, species: bool = False) -> float:
    """What the declared window costs on disk, per minute of model time.

    ``species`` prices schema v3 (four species words in ``qhyd``'s place).
    """

    records = 60.0 / (float(child_dt_s) * int(stride))
    return float(box.record_bytes(surface=surface, species=species)) * records


# ---------------------------------------------------------------------------
# Rotation: the words a v3 stream's header carries.
# ---------------------------------------------------------------------------
ROTATION_AUTHORITY: Mapping[str, str] = {
    "coriolis_force": (
        "src/solve2.F:633-653: uten += fcor*(0.25*((v(i,j)+v(i,j+1))+(v(i-1,j)+v(i-1,j+1))) "
        "+ vmove); vten -= fcor*(0.25*((u(i,j)+u(i+1,j))+(u(i,j-1)+u(i+1,j-1))) + umove), "
        "added to the stage tendency at every RK stage, never to the held uten1/vten1"
    ),
    "fcor": "src/param.F:1985 (fcor = 0 when icor = 0); README.namelist fcor",
    "frame": "src/base.F:2667 (winds are frame-relative); src/solve2.F:645,647 (the add-back)",
    "lspgrad_1": (
        "src/solve1.F:590-612: uten1 -= fcor*(0.25*((v0+v0)+(v0+v0)) + vmove); "
        "vten1 += fcor*(0.25*((u0+u0)+(u0+u0)) + umove), inside the held tendency"
    ),
}


def lspgrad1_profiles(*, fcor_s_inv: float, u_move_m_s: float, v_move_m_s: float,
                      u0: Any, v0: Any) -> tuple[Any, Any]:
    """The per-level accelerations ``solve1.F:590-612`` adds to the held tendency.

    ``(lspg_u, lspg_v)``: ``-fcor*(vbar0 + vmove)`` on the ``u`` faces and
    ``+fcor*(ubar0 + umove)`` on the ``v`` faces, one word per level, in CM1's
    float32 operation order.  ``u0``/``v0`` are the child's compact
    frame-relative base columns; a flat child's base state is horizontally
    uniform, so the four-point average of ``solve1.F:606-609`` is the level's
    own word (``0.25*((a+a)+(a+a)) == a`` exactly) and is kept anyway, in
    CM1's order, so nothing here depends on that.
    """

    np = _np()
    f = np.float32
    fcor, umove, vmove = f(fcor_s_inv), f(u_move_m_s), f(v_move_m_s)
    u0 = np.asarray(_host(u0), dtype=np.float32)
    v0 = np.asarray(_host(v0), dtype=np.float32)
    vbar = f(f(0.25) * f(f(v0 + v0) + f(v0 + v0)))
    ubar = f(f(0.25) * f(f(u0 + u0) + f(u0 + u0)))
    lspg_u = f(f(0.0) - f(fcor * f(vbar + vmove)))
    lspg_v = f(fcor * f(ubar + umove))
    return lspg_u, lspg_v


def rotation_header(*, fcor_s_inv: float, u_move_m_s: float, v_move_m_s: float,
                    lspgrad: int, lspg_u: Any = None, lspg_v: Any = None) -> dict[str, Any]:
    """The ``rotation`` block of a v3 stream's receipt."""

    np = _np()
    f = np.float32
    if int(lspgrad) not in (0, 1):
        raise NestChildWindowError(
            f"lspgrad={lspgrad}: a window header records lspgrad=1's per-level words only; any "
            "other large-scale pressure gradient would stay inside the held tendency the budget "
            "names friction"
        )
    out: dict[str, Any] = {
        "icor": 1,
        "betaplane": 0,
        "fcor_s_inv": float(f(fcor_s_inv)),
        "fcor_f32_bits": int(np.asarray(f(fcor_s_inv)).view(np.uint32)),
        "u_move_m_s": float(f(u_move_m_s)),
        "v_move_m_s": float(f(v_move_m_s)),
        "lspgrad": int(lspgrad),
        "winds": (
            "uadv/vadv are frame-relative (base.F:2667); the Coriolis force is evaluated on "
            "the ground-relative wind by adding u_move_m_s/v_move_m_s back, as solve2.F:645,647 "
            "does"
        ),
        "held_tendency": (
            "uten/vten/wten are CM1's uten1/vten1/wten1: the Coriolis force is NOT in them "
            "(solve2.F adds it to the stage tendency); with lspgrad=1 they DO hold "
            "lspg_u_m_s2/lspg_v_m_s2 (solve1.F:590-612), which the budget removes from friction "
            "and publishes as its own term"
        ),
        "cm1_authority": dict(ROTATION_AUTHORITY),
    }
    if int(lspgrad) == 1:
        if lspg_u is None or lspg_v is None:
            raise NestChildWindowError(
                "lspgrad=1 but no per-level lspg words were formed; the held tendency would "
                "carry the large-scale pressure gradient and the budget would publish it as "
                "friction"
            )
        out["lspg_u_m_s2"] = [float(v) for v in np.asarray(lspg_u, dtype=np.float32)]
        out["lspg_v_m_s2"] = [float(v) for v in np.asarray(lspg_v, dtype=np.float32)]
        out["lspg_levels"] = "CM1 k = 1..nz+1 of the window, the held tendency blocks' own levels"
    return out


def check_rotation_header(rotation: Any, *, nz: int, where: str) -> None:
    """Refuse a ``rotation`` block that could not produce the terms it names."""

    if not isinstance(rotation, Mapping):
        raise NestChildWindowError(f"{where}: a v3 stream must carry a rotation header")
    if int(rotation.get("icor", 0)) != 1 or int(rotation.get("betaplane", -1)) != 0:
        raise NestChildWindowError(
            f"{where}: the rotation header is not the f plane (icor=1, betaplane=0); the budget "
            "evaluates one fcor and would be wrong everywhere else"
        )
    for key in ("fcor_s_inv", "u_move_m_s", "v_move_m_s", "lspgrad"):
        if key not in rotation:
            raise NestChildWindowError(f"{where}: the rotation header names no {key!r}")
    lspgrad = int(rotation["lspgrad"])
    if lspgrad not in (0, 1):
        raise NestChildWindowError(f"{where}: lspgrad={lspgrad} is not one a window records")
    if lspgrad == 1:
        for key in ("lspg_u_m_s2", "lspg_v_m_s2"):
            words = rotation.get(key)
            if not isinstance(words, list) or len(words) != int(nz) + 1:
                raise NestChildWindowError(
                    f"{where}: lspgrad=1 needs {key} on the window's {int(nz) + 1} held levels; "
                    "without it the large-scale pressure gradient would be read as friction"
                )


# ---------------------------------------------------------------------------
# Admission.
# ---------------------------------------------------------------------------
NEST_CHILD_WINDOW_GATE = "cartesian-runtime"


def nest_child_window_admission_defect(*, settings: Any, geometry: Any, child_grid: Any,
                                       child_dt_s: float, duration_s: float, bbc: int,
                                       terrain: bool,
                                       criterion_declared: bool,
                                       nest_moves: bool,
                                       damping_bottom_m: float | None = None,
                                       hrdamp: int = 0,
                                       coriolis: int = 0,
                                       beta_plane: int = 0,
                                       lspgrad: int = 0,
                                       component: Any = None) -> tuple[str, str, str] | None:
    """``(code_suffix, option, reason)`` for the first inadmissible choice, or ``None``.

    ``component`` is the microphysics component id: it picks the stream's
    layout (:func:`window_layout`) and therefore the bytes the budget prices.
    """

    from .nest_vertical import levels_of_grid

    if terrain:
        return ("TERRAIN", "grid.terrain",
                "parcel.F's terrain branch is not transcribed for a child, so a window cut from "
                "a sigma-coordinate child would be sampled at the wrong heights by every offline "
                "integration that read it")
    if bbc not in (1, 2, 3):
        return ("BOUNDARY", "closures.bbc",
                f"bbc={bbc} has no parcel.F lower-boundary branch (parcel.F:59-96), so the "
                "window's k=0 ghost of u and v could not be formed and every parcel below zh(1) "
                "would read an unfilled level")
    if settings.follows_criterion and not criterion_declared:
        return ("NO-CRITERION", "lagrangian.nest_child_window.follow",
                'follow = "tornado-criterion" takes the window\'s centre from '
                "[nest.tornado_criterion], which this document does not declare; with no centre "
                "the window would never open and the stream would be empty")
    if getattr(settings, "follows_nest", False) and not nest_moves:
        return ("NEST-STATIC", "lagrangian.nest_child_window.follow",
                'follow = "nest-centre" rides a storm-following nest\'s moves, but this '
                "document's nest does not move (nest.follow_storm = false); the window would sit "
                "on the child's fixed centre while its receipt named a centre that follows the "
                'storm; declare follow = "fixed" with that centre instead')
    if float(settings.end_s) > float(duration_s):
        return ("INTERVAL", "lagrangian.nest_child_window.end_s",
                f"end_s = {settings.end_s} s is after run.duration_s = {duration_s} s; the stream "
                "would name an interval the run does not reach and the trajectories declared for "
                "its last seconds could not be integrated")
    levels = levels_of_grid(child_grid)
    child_top = float(levels.zf[int(child_grid.nz) + 1])
    try:
        nx, ny, nz = window_cells(settings, dx_m=float(geometry.dx_m), dy_m=float(geometry.dy_m),
                                  zf=levels.zf)
    except NestChildWindowError as error:
        return ("EXTENT", "lagrangian.nest_child_window.top_m", str(error))
    if nz >= int(child_grid.nz):
        return ("TOP", "lagrangian.nest_child_window.top_m",
                f"top_m = {settings.top_m} m reaches the child's own top ({child_top} m); the "
                "level above the window would then be tbc's free-slip extrapolation "
                "(parcel.F:98-126) rather than a child level, and every trajectory that touched "
                "it would feel a lid the child's interior does not have")
    try:
        place_window(centre_x_m=float(geometry.origin_x_m) + 0.5 * int(geometry.nx) * float(geometry.dx_m),
                     centre_y_m=float(geometry.origin_y_m) + 0.5 * int(geometry.ny) * float(geometry.dy_m),
                     geometry=geometry, nx=nx, ny=ny)
    except NestChildWindowError as error:
        return ("EXTENT", "lagrangian.nest_child_window.half_width_x_m", str(error))
    window_top = float(levels.zf[nz + 1])
    if damping_bottom_m is not None and float(damping_bottom_m) <= window_top:
        return ("DAMPING", "lagrangian.nest_child_window.top_m",
                f"the Rayleigh layer begins at {damping_bottom_m} m and the window's top face is "
                f"{window_top} m; the held momentum tendency the window publishes "
                "(solve1.F:414-446's uten1) would then carry the damping, and the trajectory "
                "budget would report the damping as friction")
    if int(hrdamp) != 0:
        return ("DAMPING", "closures.hrdamp",
                f"hrdamp={hrdamp} adds the lateral Rayleigh sponge to the held momentum tendency "
                "(solve1.F:414-446); the window would publish it inside the tendency the "
                "trajectory budget names friction")
    if int(coriolis) != 0 and int(beta_plane) != 0:
        # icor = 1 on the f plane is admitted: the stream's v3 header carries
        # fcor and the frame velocity, and the trajectory budget adds the
        # Coriolis force of solve2.F:639-653 and its curl (f's tilting and
        # stretching).  The beta plane is not: f then varies with yh
        # (param.F:8966) and the header carries one f.
        return ("CORIOLIS-BETA-PLANE", "dynamics.beta_plane",
                f"coriolis={coriolis} with beta_plane={beta_plane}: solve2.F:655-673 makes f a "
                "function of the child's yh, and the window's header carries one fcor, so the "
                "trajectory budget's Coriolis term would be evaluated with the wrong f at every "
                "row but one")
    if int(lspgrad) not in (0, 1):
        return ("LSPGRAD", "dynamics.lspgrad",
                f"lspgrad={lspgrad}: only lspgrad=1's per-level geostrophic words "
                "(solve1.F:590-612) are recorded in the window's header; any other large-scale "
                "pressure gradient would stay inside the held tendency the budget names "
                "friction")
    try:
        layout = window_layout(component)
    except NestChildWindowError as error:
        return ("SPECIES", "physics.component", str(error))
    box = WindowBox(0, 0, nx, ny, nz)
    cost = stream_bytes_per_model_minute(box, child_dt_s=child_dt_s,
                                         stride=int(settings.stride_child_steps),
                                         surface=bbc == 3, species=layout.species)
    if cost > float(settings.budget_bytes_per_model_minute):
        return ("DISK-BUDGET", "lagrangian.nest_child_window.budget_bytes_per_model_minute",
                f"this {'species schema-v3 ' if layout.species else ''}window is "
                f"{nx}x{ny}x{nz} child cells and writes "
                f"{box.record_bytes(surface=bbc == 3, species=layout.species)} bytes every "
                f"{float(child_dt_s) * int(settings.stride_child_steps)} s of model time, which "
                f"is {cost:.0f} bytes per model minute against the declared budget of "
                f"{settings.budget_bytes_per_model_minute}; a stream that discovered its cost by "
                "filling the disk would lose the run's next history frame and its restart with it")
    return None


def _damping_bottom_m(config: Any) -> float | None:
    """Where CM1's upper Rayleigh layer begins (``zd = ztop - depth``), or ``None``."""

    closures = config.closures
    if int(getattr(closures, "upper_damping", 0) or 0) == 0:
        return None
    grid = config.grid
    # param.F:2608: an unstretched grid's top is nz*dz, and its ztop_m is
    # refused unless 0.0 (CM1-CFG-STRETCH-CONTROL-UNUSED), so reading ztop_m
    # there would put the Rayleigh layer below the ground and refuse every
    # window on an unstretched child.
    if int(getattr(grid, "stretch_z", 0) or 0) == 0:
        ztop = float(grid.nz) * float(grid.dz_m)
    else:
        ztop = float(grid.ztop_m)
    return ztop - float(closures.upper_damping_depth_m)


def _config_lspgrad(dynamics: Any) -> int:
    """The document's ``dynamics.lspgrad``, 0 where the key does not exist or is absent."""

    value = getattr(dynamics, "lspgrad_value", None)
    if value is None:
        value = getattr(dynamics, "lspgrad", 0)
    return int(value or 0)


def require_nest_child_window_admission(config: Any, binding: Any) -> None:
    """Refuse ``[lagrangian.nest_child_window]`` where it cannot mean what it says."""

    from .errors import refuse

    lagrangian = getattr(config, "lagrangian", None)
    settings = None if lagrangian is None else lagrangian.nest_child_window
    if settings is None:
        return
    if binding is None:
        refuse(
            code="CM1-CARTESIAN-NEST-WINDOW-NO-NEST",
            gate=NEST_CHILD_WINDOW_GATE,
            reason=(
                "[lagrangian.nest_child_window] streams a sub-volume of a nest child, but the "
                "document declares no [nest]; there is no child lattice to cut a window from, "
                "and a window cut from the parent would hold 100 m words while its receipt "
                "named the child's"
            ),
            remedy="declare [nest], or remove [lagrangian.nest_child_window]",
            option="lagrangian.nest_child_window", value=None,
        )
    nest = getattr(config, "nest", None)
    defect = nest_child_window_admission_defect(
        settings=settings, geometry=binding.geometry, child_grid=binding.child.grid,
        child_dt_s=float(binding.child.dt_s), duration_s=float(config.run.duration_s),
        bbc=int(config.closures.bbc), terrain=bool(config.grid.terrain),
        criterion_declared=getattr(nest, "tornado_criterion", None) is not None,
        nest_moves=bool(getattr(nest, "follow_storm", False)),
        damping_bottom_m=_damping_bottom_m(config),
        hrdamp=int(getattr(config.closures, "hrdamp", 0) or 0),
        coriolis=int(getattr(getattr(config, "dynamics", None), "coriolis", 0) or 0),
        beta_plane=int(getattr(getattr(config, "dynamics", None), "beta_plane", 0) or 0),
        lspgrad=_config_lspgrad(getattr(config, "dynamics", None)),
        component=config.physics.component,
    )
    if defect is not None:
        suffix, option, reason = defect
        refuse(
            code=f"CM1-CARTESIAN-NEST-WINDOW-{suffix}",
            gate=NEST_CHILD_WINDOW_GATE,
            reason=reason,
            remedy=(
                "declare a window that fits inside the child's free interior with three ghost "
                "columns to spare, below the child's top and its Rayleigh layer, over a flat "
                "free-slip, no-slip or semi-slip child without hrdamp, with Coriolis only on "
                "the f plane and lspgrad 0 or 1, inside the run's own duration, within a disk "
                "budget the document states, with a microphysics component whose species rows "
                "the window can name; follow = \"nest-centre\" only on a storm-following nest "
                "(nest.follow_storm = true)"
            ),
            option=option, value=settings.record(),
        )


# ---------------------------------------------------------------------------
# The writer.
# ---------------------------------------------------------------------------
INDEX_COLUMNS: tuple[str, ...] = (
    "index", "child_step", "parent_step", "model_time_s", "offset_bytes", "bytes",
    "i0", "j0", "nx", "ny", "nz", "origin_x_m", "origin_y_m",
    "centre_x_m", "centre_y_m", "centre_held", "clamped", "sha256",
)


@dataclass
class _Pending:
    row: dict[str, Any]
    buffer: Any


class _StreamWriter:
    """One worker thread: arena append, SHA-256, index row, buffer recycling.

    The stepping thread never waits on a write; it waits only when every
    buffer in the pool is still in flight, and that wait is measured and
    published (``blocked_seconds``), because a stream whose cadence the disk
    cannot keep up with must say so rather than quietly stealing the run's
    step rate.
    """

    def __init__(self, root: Path, *, depth: int) -> None:
        self.root = root
        self.arena = (root / WINDOW_ARENA_FILENAME).open("xb", buffering=0)
        self.index = (root / WINDOW_INDEX_FILENAME).open("xb", buffering=0)
        self.index.write(("\t".join(INDEX_COLUMNS) + "\n").encode("ascii"))
        self.free: queue.Queue[Any] = queue.Queue()
        self.work: queue.Queue[Any] = queue.Queue()
        self.depth = int(depth)
        self._made: list[Any] = []
        self.rows: list[dict[str, Any]] = []
        self.written_bytes = 0
        self.blocked_seconds = 0.0
        self.error: BaseException | None = None
        # A daemon thread on purpose.  A run that dies leaves a stream with no
        # receipt, and read_window_stream refuses one of those by name, so a
        # truncated arena can never be read as anything; what a non-daemon
        # writer WOULD do is hang the interpreter on a failed run, waiting on a
        # queue nothing will fill.  The success path joins it in close().
        self.thread = threading.Thread(target=self._run, name="gm1-child-window", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.work.get()
            if item is None:
                return
            try:
                raw = item.buffer.tobytes(order="C")
                self.arena.write(raw)
                item.row["sha256"] = hashlib.sha256(raw).hexdigest()
                self.index.write(("\t".join(str(item.row[name]) for name in INDEX_COLUMNS)
                                  + "\n").encode("ascii"))
                self.rows.append(item.row)
                self.written_bytes += len(raw)
            except BaseException as error:  # surfaced on the stepping thread
                if self.error is None:
                    self.error = error
            finally:
                self.free.put(item.buffer)

    def take(self, make: Any) -> Any:
        self.check()
        try:
            return self.free.get_nowait()
        except queue.Empty:
            pass
        if len(self._made) < self.depth:
            buffer = make()
            self._made.append(buffer)
            return buffer
        started = time.perf_counter()
        buffer = self.free.get()
        self.blocked_seconds += time.perf_counter() - started
        self.check()
        return buffer

    def submit(self, row: dict[str, Any], buffer: Any) -> None:
        self.check()
        self.work.put(_Pending(row, buffer))

    def check(self) -> None:
        if self.error is not None:
            error, self.error = self.error, None
            raise error

    def abandon(self) -> None:
        """Stop the worker and leave the stream unreadable, for a failed run."""

        self.work.put(None)
        self.thread.join(timeout=30.0)

    def close(self) -> None:
        self.work.put(None)
        self.thread.join()
        self.arena.flush()
        os.fsync(self.arena.fileno())
        self.arena.close()
        self.index.flush()
        os.fsync(self.index.fileno())
        self.index.close()
        self.check()


# ---------------------------------------------------------------------------
# The live stream.
# ---------------------------------------------------------------------------
@dataclass
class NestChildWindowStream:
    """The child's vortex-following sub-volume, captured every child step."""

    settings: Any
    child_deck: ParcelDeck
    geometry: Any
    child_dt_s: float
    parent_dt_s: float
    bbc: int
    tbc: int
    u_move_m_s: float
    v_move_m_s: float
    imove: int
    nx: int
    ny: int
    nz: int
    xp: Any = None
    stream: Any = None
    buffer_depth: int = 3
    layout: WindowLayout | None = None

    centre: tuple[float, float] | None = None
    centre_step: int | None = None
    centre_found_step: int | None = None
    parent_step: int = 0
    records: int = 0
    held_records: int = 0
    clamped_records: int = 0
    skipped_no_centre: int = 0
    skipped_outside_interval: int = 0
    device_copy_seconds: float = 0.0
    _writer: Any = None
    _box: WindowBox | None = None
    _clamped: bool = False
    _held: bool = False
    _offsets: tuple[tuple[str, int, int], ...] = ()
    _record_words: int = 0
    _carriers_checked: bool = False
    base_columns: dict[str, Any] | None = None
    #: ``None`` for a non-rotating child (a v2 stream); otherwise the f-plane
    #: selection, completed with lspgrad=1's words at the first capture.
    icor: int = 0
    fcor_s_inv: float = 0.0
    lspgrad: int = 0
    rotation: dict[str, Any] | None = None

    @classmethod
    def build(cls, *, settings: Any, geometry: Any, child_deck: ParcelDeck, child_config: Any,
              parent_dt_s: float, xp: Any = None, stream: Any = None) -> "NestChildWindowStream":
        nx, ny, nz = window_cells(settings, dx_m=float(geometry.dx_m), dy_m=float(geometry.dy_m),
                                  zf=child_deck.zf)
        if nz >= child_deck.nz:
            raise NestChildWindowError(
                "the window's top reaches the child's own; the level above it would be tbc's "
                "extrapolation rather than a child level"
            )
        layout = window_layout(getattr(child_config, "microphysics", None))
        imove = getattr(child_config, "imove", None)
        u_move, v_move = float(child_config.u_move_m_s), float(child_config.v_move_m_s)
        icor = int(getattr(child_config, "icor", 0) or 0)
        lspgrad = int(getattr(child_config, "lspgrad", 0) or 0)
        if icor == 1 and int(getattr(child_config, "betaplane", 0) or 0) != 0:
            raise NestChildWindowError(
                "the child integrates solve2.F:655-673's beta plane; f then varies with yh and "
                "the window's header carries one fcor"
            )
        if lspgrad not in (0, 1):
            raise NestChildWindowError(
                f"the child integrates lspgrad={lspgrad}; a window header records lspgrad=1's "
                "per-level words only"
            )
        self = cls(
            settings=settings, child_deck=child_deck, geometry=geometry,
            child_dt_s=float(child_config.dt_s), parent_dt_s=float(parent_dt_s),
            bbc=int(child_config.bbc), tbc=int(getattr(child_config, "tbc", 1)),
            u_move_m_s=u_move, v_move_m_s=v_move,
            imove=int(imove) if imove is not None else int(u_move != 0.0 or v_move != 0.0),
            nx=nx, ny=ny, nz=nz, xp=xp, stream=stream, layout=layout,
            icor=icor,
            fcor_s_inv=float(getattr(child_config, "fcor_s_inv", 0.0) or 0.0) if icor == 1 else 0.0,
            lspgrad=lspgrad if icor == 1 else 0,
        )
        if icor == 1 and lspgrad == 0:
            self.rotation = rotation_header(fcor_s_inv=self.fcor_s_inv, u_move_m_s=u_move,
                                            v_move_m_s=v_move, lspgrad=0)
        self._layout()
        if settings.follows_nest:
            # The child's own centre is known before the first step too.
            self.centre = (float(geometry.centre_x_m), float(geometry.centre_y_m))
            self._box = self._place()
        elif not settings.follows_criterion:
            # A declared centre is known before the first step, so the window
            # is placed at once and no record is skipped waiting for one.
            self.centre = (float(settings.center_x_m), float(settings.center_y_m))
            self._box = self._place()
        return self

    # -- layout ---------------------------------------------------------------
    @property
    def ng_lateral(self) -> int:
        return NG

    @property
    def surface(self) -> bool:
        return self.bbc == 3

    @property
    def species(self) -> bool:
        return self.layout is not None and self.layout.species

    @property
    def fields_table(self) -> tuple[tuple[str, str, str, str], ...]:
        return WINDOW_FIELDS if self.layout is None else self.layout.fields

    @property
    def schema(self) -> str:
        """v3 when the stream carries species or a rotation header, else v2."""

        if self.species or self.icor == 1:
            return NEST_CHILD_WINDOW_SCHEMA_V3
        return NEST_CHILD_WINDOW_SCHEMA

    def _layout(self) -> None:
        box = WindowBox(0, 0, self.nx, self.ny, self.nz)
        shapes = box.shapes(surface=self.surface, species=self.species)
        order = [name for name, *_ in self.fields_table]
        if self.surface:
            order.append(WINDOW_SURFACE_FIELD[0])
        offsets: list[tuple[str, int, int]] = []
        at = 0
        for name in order:
            words = math.prod(shapes[name])
            offsets.append((name, at, words))
            at += words
        self._offsets = tuple(offsets)
        self._record_words = at

    @property
    def record_bytes(self) -> int:
        return 4 * self._record_words

    def bytes_per_model_minute(self) -> float:
        return stream_bytes_per_model_minute(
            WindowBox(0, 0, self.nx, self.ny, self.nz), child_dt_s=self.child_dt_s,
            stride=int(self.settings.stride_child_steps), surface=self.surface,
            species=self.species,
        )

    # -- the centre -----------------------------------------------------------
    def after_parent_step(self, driver: Any) -> None:
        """Take the parent step's centre and re-place the window on it.

        Called from :meth:`cm1core.nest_live.NestDriver.after_parent_step`
        after the criterion has sampled and after any nest move, so the index
        origin the next child steps are cut at belongs to the placement they
        will run on.
        """

        self.geometry = driver.geometry
        self.relocate_child(driver.geometry)
        self.parent_step = int(driver.parent.clock.completed_steps)
        if self.settings.follows_nest:
            # After any nest move, so the centre is the placement the next child
            # steps run on; the parent-grid-origin frame is the geometry's own.
            self.centre = (float(driver.geometry.centre_x_m), float(driver.geometry.centre_y_m))
            self.centre_found_step = self.parent_step
            self._held = False
        elif self.settings.follows_criterion:
            criterion = getattr(driver, "criterion", None)
            sample = None if criterion is None else getattr(criterion, "last_sample", None)
            if sample is not None and bool(sample.found):
                # The criterion samples BEFORE the feedback and before any nest
                # move (CM1-NEST-SEAM.md 5.9), so its corner index belongs to
                # the placement the child was integrated on, not to the one the
                # window is about to be cut from.  Its own ground position
                # already carries that placement's origin
                # (nest_tornado_criterion.sample_from_words), so the centre is
                # taken from there and put back in the parent-grid-origin frame
                # by removing the domain's ground position.  Re-deriving it from
                # the current geometry would move the centre by exactly the
                # cells the nest just moved.
                clock = driver.parent.clock
                self.centre = (float(sample.ground_x_m) - float(clock.domainlocx_m),
                               float(sample.ground_y_m) - float(clock.domainlocy_m))
                self.centre_found_step = self.parent_step
                self._held = False
            else:
                self._held = self.centre is not None
        self.centre_step = self.parent_step
        self._box = None if self.centre is None else self._place()

    def relocate_child(self, geometry: Any) -> bool:
        """Put the child deck's ``xf(1)``/``yf(1)`` on ``geometry``; ``True`` if it moved.

        A following nest (:mod:`cm1core.nest_move`) re-places the child by whole
        parent cells between parent steps, and every record's published ground
        origin is ``child xf(1) + i0*dx``.  Without this the origin would stay
        at the placement the stream was built on and every trajectory read out
        of the stream after the first move would be put on the wrong ground --
        the one failure mode a moving nest makes silent, because the words
        themselves would still be the child's.
        """

        from dataclasses import replace

        np = _np()
        minx = float(np.float32(geometry.origin_x_m))
        miny = float(np.float32(geometry.origin_y_m))
        if minx == self.child_deck.minx_m and miny == self.child_deck.miny_m:
            return False
        self.child_deck = replace(self.child_deck, minx_m=minx, miny_m=miny)
        return True

    def _place(self) -> WindowBox:
        assert self.centre is not None
        i0, j0, clamped = place_window(centre_x_m=self.centre[0], centre_y_m=self.centre[1],
                                       geometry=self.geometry, nx=self.nx, ny=self.ny)
        self._clamped = clamped
        return WindowBox(i0, j0, self.nx, self.ny, self.nz)

    # -- lifecycle ------------------------------------------------------------
    def open(self, root: Path | str) -> None:
        directory = Path(root) / NEST_CHILD_WINDOW_DIRECTORY
        directory.mkdir(parents=True, exist_ok=False)
        self._writer = _StreamWriter(directory, depth=self.buffer_depth)

    def _make_buffer(self) -> Any:
        np = _np()
        return np.empty(self._record_words, dtype="<f4")

    # -- one child step -------------------------------------------------------
    def _check_carriers(self, arrays: Mapping[str, Any]) -> None:
        """Every carrier the cut reads, present and of the child's own shape.

        The cut is pure indexing, so a carrier of the wrong family (padded
        where it assumes physical, or the reverse) would not fail: it would
        publish the word three cells or one level away from the one named.
        """

        nx, ny, nz = self.child_deck.nx, self.child_deck.ny, self.child_deck.nz
        want = {
            "rru": (nz + 2, ny + 2 * NG, nx + 1 + 2 * NG),
            "rrv": (nz + 2, ny + 1 + 2 * NG, nx + 2 * NG),
            "rrw": (nz + 3, ny + 2 * NG, nx + 2 * NG),
            "rho": (nz, ny, nx), "prs": (nz, ny, nx), "tha": (nz, ny, nx),
            "uten": (nz, ny, nx + 1), "vten": (nz, ny + 1, nx), "wten": (nz + 1, ny, nx),
        }
        for name, shape in want.items():
            if name not in arrays:
                raise NestChildWindowError(
                    f"the child hands the window no {name!r} carrier; a record without it "
                    "would hold another field under that name"
                )
            if tuple(arrays[name].shape) != shape:
                raise NestChildWindowError(
                    f"the child's {name!r} carrier is {tuple(arrays[name].shape)}, not the "
                    f"{shape} the cut indexes; every word cut from it would be displaced"
                )
        qa = arrays.get("qa")
        if self.species and (qa is None or int(qa.shape[0]) != self.layout.numq):
            raise NestChildWindowError(
                f"the child's qa has {None if qa is None else int(qa.shape[0])} rows, not the "
                f"{self.layout.numq} of {self.layout.component_id!r} ({self.layout.axis_authority}); "
                "schema v3 finds its species rows by name in that table, and a carrier of "
                "another layout would publish other rows under the species' names"
            )
        if qa is None or tuple(qa.shape[1:]) != (nz, ny, nx) or int(qa.shape[0]) < 6:
            raise NestChildWindowError(
                "the child's qa must be (numq >= 6, nz, ny, nx) with qv, qc, qr, qi, qs, qg in "
                "its first six rows (parcel.F:745-775's order); a window cut from another "
                "species layout would publish a different sum as the condensate loading"
            )
        self._carriers_checked = True

    def graph_state(self, *, child_step: int, model_time_s: float) -> str:
        """Which branch :meth:`capture` takes at this child step, deciding nothing.

        ``"due"`` cuts and streams a record (a host read, so that child step
        runs eagerly under ``--cuda-graphs``); every other state is a host
        branch a captured step freezes, so it is part of the graph key.
        """

        if self._writer is None:
            return "closed"
        if not (float(self.settings.start_s) <= model_time_s <= float(self.settings.end_s)):
            return "outside-interval"
        if int(child_step) % int(self.settings.stride_child_steps) != 0:
            return "off-stride"
        if self._box is None:
            return "no-centre"
        return "due"

    def _count_outside_interval(self, _step: int) -> None:
        self.skipped_outside_interval += 1

    def _count_no_centre(self, _step: int) -> None:
        self.skipped_no_centre += 1

    def capture(self, arrays: Mapping[str, Any], *, child_step: int, model_time_s: float,
                roughness: Any = None) -> bool:
        """Cut and enqueue one record; ``True`` when one was written.

        Called at the child's ``nrk == nrkmax``, in
        ``cm1core.nest_live``'s child runtime, immediately after
        :meth:`cm1core.nest_parcels.NestParcelTracker.capture_child` -- the
        same words at the same moment of the same step.  ``child_step`` is the
        step this capture completes and ``model_time_s`` the model time at its
        end, which is the time a backward integrator steps *from* when it
        reverses this record.
        """

        state = self.graph_state(child_step=child_step, model_time_s=model_time_s)
        if state == "outside-interval":
            cuda_graphs.bookkeeping(self._count_outside_interval, child_step)
        elif state == "no-centre":
            cuda_graphs.bookkeeping(self._count_no_centre, child_step)
        if state != "due":
            return False
        np, xp = _np(), self.xp
        box = self._box
        surface = None
        if self.surface:
            if roughness is None:
                raise NestChildWindowError(
                    "bbc=3 but the child carries no zntmp for this step; parcel.F:470-515's log "
                    "layer below zh(1) is exactly where a tornado's inflow parcels are, and a "
                    "stream without it could not sample one of them"
                )
            surface = roughness
        if not self._carriers_checked:
            self._check_carriers(arrays)
        if self.base_columns is None:
            # th0 and qv0 are horizontally uniform in a flat child (base.F), so
            # one column holds them; the budget's buoyancy needs them and they
            # are recorded once, in the receipt, rather than streamed.
            self.base_columns = {
                name: [float(v) for v in np.asarray(
                    _host(arrays[f"{name}_profile"][0:self.nz + 1]), dtype=np.float32)]
                for name in ("th0", "qv0") if f"{name}_profile" in arrays
            }
        if self.icor == 1 and self.rotation is None:
            # lspgrad=1: solve1.F:590-612's per-level words, formed once from
            # the child's own compact frame-relative base columns.
            if "u0_profile" not in arrays or "v0_profile" not in arrays:
                raise NestChildWindowError(
                    "lspgrad=1 but the child hands the window no u0/v0 base column; the held "
                    "tendency carries solve1.F:590-612's geostrophic words and without them "
                    "the budget would publish the large-scale pressure gradient as friction"
                )
            lspg_u, lspg_v = lspgrad1_profiles(
                fcor_s_inv=self.fcor_s_inv, u_move_m_s=self.u_move_m_s,
                v_move_m_s=self.v_move_m_s,
                u0=arrays["u0_profile"][0:self.nz + 1], v0=arrays["v0_profile"][0:self.nz + 1])
            self.rotation = rotation_header(
                fcor_s_inv=self.fcor_s_inv, u_move_m_s=self.u_move_m_s,
                v_move_m_s=self.v_move_m_s, lspgrad=1, lspg_u=lspg_u, lspg_v=lspg_v)
        if self.stream is not None and hasattr(self.stream, "__enter__"):
            # qhyd is the one word formed here; it must be ordered on the
            # child's own stream, after the step that wrote qa.
            with self.stream:
                fields = cut_window(xp, arrays, box, surface=surface,
                                    species_rows=self._species_rows())
        else:
            fields = cut_window(xp, arrays, box, surface=surface,
                                species_rows=self._species_rows())
        buffer = self._writer.take(self._make_buffer)
        started = time.perf_counter()
        shapes = box.shapes(surface=self.surface, species=self.species)
        for name, at, words in self._offsets:
            block = fields[name]
            host = np.asarray(block.get(stream=self.stream) if hasattr(block, "get") else block,
                              dtype=np.float32)
            buffer[at:at + words] = host.reshape(-1)
        self.device_copy_seconds += time.perf_counter() - started
        # parcel.F:59-96's k = 0 ghost, the one word the cut cannot carry: the
        # child's live rru/rrv hold no filled lower ghost, and the branch reads
        # only levels 1, 2 and 3, so it is formed here on the window's own block.
        place = {name: (at, words) for name, at, words in self._offsets}

        def block(name: str) -> Any:
            at, words = place[name]
            return buffer[at:at + words].reshape(shapes[name])

        fill_velocity_bottom_ghost(np, block("uadv"), block("vadv"), self.child_deck,
                                   bbc=self.bbc, u_move_m_s=self.u_move_m_s,
                                   v_move_m_s=self.v_move_m_s)
        fill_window_w_ghost(block("wadv"))
        row = {
            "index": self.records, "child_step": int(child_step), "parent_step": self.parent_step,
            "model_time_s": repr(float(model_time_s)),
            "offset_bytes": self.records * self.record_bytes, "bytes": self.record_bytes,
            "i0": box.i0, "j0": box.j0, "nx": box.nx, "ny": box.ny, "nz": box.nz,
            "origin_x_m": repr(float(np.float32(self.child_deck.minx_m + box.i0 * self.child_deck.dx_m))),
            "origin_y_m": repr(float(np.float32(self.child_deck.miny_m + box.j0 * self.child_deck.dy_m))),
            "centre_x_m": repr(float(self.centre[0])), "centre_y_m": repr(float(self.centre[1])),
            "centre_held": int(self._held), "clamped": int(self._clamped), "sha256": "",
        }
        self._writer.submit(row, buffer)
        self.records += 1
        self.held_records += int(self._held)
        self.clamped_records += int(self._clamped)
        return True

    def _species_rows(self) -> Mapping[str, tuple[int, ...]] | None:
        return self.layout.species_rows if self.species else None

    # -- close ----------------------------------------------------------------
    def close(self) -> dict[str, Any] | None:
        if self._writer is None:
            return None
        writer, self._writer = self._writer, None
        writer.close()
        receipt = self.record()
        receipt["arena_bytes"] = writer.written_bytes
        receipt["blocked_seconds"] = writer.blocked_seconds
        receipt["records_written"] = len(writer.rows)
        path = writer.root / WINDOW_RECEIPT_FILENAME
        with path.open("xb", buffering=0) as handle:
            handle.write((json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
                         .encode("utf-8"))
            os.fsync(handle.fileno())
        fd = os.open(writer.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        receipt["directory"] = str(writer.root)
        return receipt

    def abandon(self) -> None:
        """Stop the writer without a receipt; the stream is then unreadable by name.

        A run that died wrote no receipt for anything else either, and
        :func:`read_window_stream` refuses a directory without one, so what is
        on disk cannot be mistaken for a complete stream.
        """

        writer, self._writer = self._writer, None
        if writer is not None:
            writer.abandon()

    def record(self) -> dict[str, Any]:
        # The shape-only box: every record's real index origin, and therefore
        # its ground origin, is in that record's own index row, because the
        # window follows the vortex and the child may have moved under it.
        box = WindowBox(0, 0, self.nx, self.ny, self.nz)
        deck = window_deck(self.child_deck, box)
        fields = [
            {"name": name, "stagger": stagger, "units": units, "meaning": meaning,
             "shape_zyx": list(box.shapes(surface=self.surface, species=self.species)[name]),
             "offset_words": at, "words": words}
            for (name, at, words), (_n, stagger, units, meaning) in zip(
                self._offsets,
                list(self.fields_table) + ([WINDOW_SURFACE_FIELD] if self.surface else []),
            )
        ]
        # Schema v2's receipt is unchanged key for key; v3 adds its version and
        # its field names in arena order, and -- for species -- the species'
        # rows and sources (the rotation header is added below).
        v3 = {} if self.schema == NEST_CHILD_WINDOW_SCHEMA else {
            "schema_version": 3,
            "field_names": [entry["name"] for entry in fields],
        }
        if self.species:
            v3["species"] = self.layout.record()
        out = {
            "schema": self.schema,
            **v3,
            "kind": "model-step-window-stream",
            "label": (
                "GM1 EXTENSION -- a vortex-following sub-volume of the live nest child, holding "
                "the model's own advecting velocity words (solve2.F:1290-1301) at the child's "
                "own step; not a CM1 product and not a history frame"
            ),
            "selectors": self.settings.record(),
            "window": box.record(),
            "ghost_columns": NG,
            "record_bytes": self.record_bytes,
            "record_words": self._record_words,
            "bytes_per_model_minute": self.bytes_per_model_minute(),
            "fields": fields,
            "deck": {
                **deck.record(),
                "dz_reference_m": float(self.child_deck.dz_reference_m),
                "xf1_m": "per record: the index row's origin_x_m",
                "yf1_m": "per record: the index row's origin_y_m",
            },
            "zf_cm1_indexed": [float(v) for v in deck.zf],
            "zh_cm1_indexed": [float(v) for v in deck.zh],
            "mf_cm1_indexed": [float(v) for v in deck.mf],
            "cgs": list(deck.cgs),
            "child": {
                "dt_s": self.child_dt_s, "dx_m": self.child_deck.dx_m,
                "dy_m": self.child_deck.dy_m, "nx": self.child_deck.nx,
                "ny": self.child_deck.ny, "nz": self.child_deck.nz,
                "bbc": self.bbc, "tbc": self.tbc, "imove": self.imove,
                "u_move_m_s": self.u_move_m_s, "v_move_m_s": self.v_move_m_s,
                "spec_zone": int(self.geometry.spec_zone),
                "relax_zone": int(self.geometry.relax_zone),
            },
            "parent_dt_s": self.parent_dt_s,
            "base_columns": {
                "levels": "CM1 k = 1..nz+1 at the window's first record",
                **(self.base_columns or {}),
            },
            "held_tendency": (
                "uten/vten/wten are the step's held momentum tendency (CM1 uten1/vten1/wten1, "
                "solve1.F); the door admits a window only below the Rayleigh layer and without "
                "hrdamp or large-scale forcing, so inside the window it is the subgrid stress "
                "(turbtend.F) plus any idiff diffusion, which is what the budget names friction"
            ),
            "counts": {
                "records": self.records, "centre_held": self.held_records,
                "clamped": self.clamped_records,
                "skipped_before_a_centre_was_found": self.skipped_no_centre,
                "skipped_outside_the_declared_interval": self.skipped_outside_interval,
            },
            "restart": (
                "the window's centre is not restart state: a resumed run reopens the stream and "
                "takes its first centre from the criterion's first sample after the resume, one "
                "parent step later, and counts the records it skipped waiting for it"
                if self.settings.follows_criterion else
                "the window's centre is not restart state: a resumed run reopens the stream on "
                "the resumed child's centre"
                if self.settings.follows_nest else
                "the window's centre is the declared one; a resumed run reopens the stream there"
            ),
            "centre_lag": (
                "the centre is the tornado criterion's from the previous parent step, so a "
                f"record lags the vortex by at most one parent step ({self.parent_dt_s} s)"
                if self.settings.follows_criterion else
                "the child's centre after the parent step's nest move; the nest's tracker "
                "decides where that is (nest.follow_interval_s cadence, whole parent cells)"
                if self.settings.follows_nest else "fixed centre; no lag"
            ),
            "device_copy_seconds": self.device_copy_seconds,
            "cm1_authority": dict(CM1_AUTHORITY),
            "gm1_choices": dict(GM1_CHOICES),
            "unvalidated": UNVALIDATED_MARKER,
        }
        if self.icor == 1:
            if self.rotation is None:
                raise NestChildWindowError(
                    "lspgrad=1 but no record was captured, so no lspg words were formed; the "
                    "stream would name a rotation it cannot describe"
                )
            out["schema"] = NEST_CHILD_WINDOW_ROTATING_SCHEMA
            out["rotation"] = dict(self.rotation)
            out["held_tendency"] = (
                "uten/vten/wten are the step's held momentum tendency (CM1 uten1/vten1/wten1, "
                "solve1.F); the door admits a window only below the Rayleigh layer and without "
                "hrdamp, so inside the window it is the subgrid stress (turbtend.F) plus any "
                "idiff diffusion, plus -- with lspgrad=1 -- the per-level large-scale pressure "
                "gradient of solve1.F:590-612 recorded in rotation.lspg_*_m_s2, which the "
                "budget removes before naming the rest friction.  The Coriolis force itself is "
                "not in it: solve2.F:633-653 adds it to the stage tendency, and the budget "
                "re-forms it from uadv/vadv and rotation.fcor_s_inv"
            )
        return out


# ---------------------------------------------------------------------------
# The reader.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowRecord:
    index: int
    child_step: int
    parent_step: int
    model_time_s: float
    offset_bytes: int
    bytes: int
    i0: int
    j0: int
    nx: int
    ny: int
    nz: int
    origin_x_m: float
    origin_y_m: float
    centre_x_m: float
    centre_y_m: float
    centre_held: bool
    clamped: bool
    sha256: str


def read_window_index(root: Path | str) -> tuple[dict[str, Any], list[WindowRecord]]:
    """The stream's receipt and every index row, in model order."""

    directory = Path(root)
    receipt_path = directory / WINDOW_RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise NestChildWindowError(
            f"{receipt_path} does not exist; a window stream without its receipt carries no deck, "
            "no field layout and no child words, so its bytes cannot be read as anything"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = receipt.get("schema")
    if schema not in NEST_CHILD_WINDOW_SCHEMAS:
        raise NestChildWindowError(
            f"{receipt_path} is not a {' or '.join(NEST_CHILD_WINDOW_SCHEMAS)} document")
    species = "species" in receipt
    rotating = receipt.get("rotation") is not None
    names = [entry.get("name") for entry in receipt.get("fields", [])]
    want = [name for name, *_ in (WINDOW_FIELDS_V3 if schema == NEST_CHILD_WINDOW_SCHEMA_V3
                                  and species else WINDOW_FIELDS)]
    if names[:len(want)] != want or names[len(want):] not in ([], [WINDOW_SURFACE_FIELD[0]]):
        raise NestChildWindowError(
            f"{receipt_path} says {schema}{' with species' if species else ''} but lists "
            f"fields {names}, not {want} (plus znt when bbc = 3); a reader that trusted the "
            "schema over the layout would read one field as another"
        )
    if schema == NEST_CHILD_WINDOW_SCHEMA and (species or "rotation" in receipt):
        raise NestChildWindowError(
            f"{receipt_path} is a {schema} document with a species or rotation header; v2 "
            "streams carry qhyd and are non-rotating by definition, and one of the two is wrong"
        )
    if schema == NEST_CHILD_WINDOW_SCHEMA_V3 and not (species or rotating):
        raise NestChildWindowError(
            f"{receipt_path}: a {schema} stream without species must carry a rotation header; "
            "with neither it is a v2 stream under the wrong name, or a rotating stream that "
            "lost the header its budget needs"
        )
    if rotating:
        check_rotation_header(receipt.get("rotation"), nz=int(receipt["window"]["nz"]),
                              where=str(receipt_path))
    if schema == NEST_CHILD_WINDOW_SCHEMA_V3 and (
            receipt.get("schema_version") != 3 or receipt.get("field_names") != names):
        raise NestChildWindowError(
            f"{receipt_path} is schema v3 but its schema_version/field_names header does not "
            "agree with its field table")
    lines = (directory / WINDOW_INDEX_FILENAME).read_text(encoding="ascii").splitlines()
    header = lines[0].split("\t")
    if tuple(header) != INDEX_COLUMNS:
        raise NestChildWindowError(
            f"{directory / WINDOW_INDEX_FILENAME} has columns {header}, not {list(INDEX_COLUMNS)}"
        )
    rows: list[WindowRecord] = []
    for line in lines[1:]:
        if not line:
            continue
        cells = dict(zip(INDEX_COLUMNS, line.split("\t")))
        rows.append(WindowRecord(
            index=int(cells["index"]), child_step=int(cells["child_step"]),
            parent_step=int(cells["parent_step"]), model_time_s=float(cells["model_time_s"]),
            offset_bytes=int(cells["offset_bytes"]), bytes=int(cells["bytes"]),
            i0=int(cells["i0"]), j0=int(cells["j0"]), nx=int(cells["nx"]),
            ny=int(cells["ny"]), nz=int(cells["nz"]),
            origin_x_m=float(cells["origin_x_m"]), origin_y_m=float(cells["origin_y_m"]),
            centre_x_m=float(cells["centre_x_m"]), centre_y_m=float(cells["centre_y_m"]),
            centre_held=bool(int(cells["centre_held"])), clamped=bool(int(cells["clamped"])),
            sha256=cells["sha256"],
        ))
    rows.sort(key=lambda r: r.model_time_s)
    return receipt, rows


@dataclass
class WindowStream:
    """A published stream, opened for reading: the receipt, the index, the words."""

    directory: Path
    receipt: dict[str, Any]
    records: list[WindowRecord]
    _raw: Any = None

    @property
    def dt_s(self) -> float:
        """The model time one record steps: the child's step times the stride."""

        return float(self.receipt["child"]["dt_s"]) * int(
            self.receipt["selectors"]["stride_child_steps"])

    @property
    def times_s(self) -> list[float]:
        return [r.model_time_s for r in self.records]

    @property
    def rotation(self) -> dict[str, Any] | None:
        """The stream's f-plane header, or ``None`` for a non-rotating stream."""

        rotation = self.receipt.get("rotation")
        return None if rotation is None else dict(rotation)

    @property
    def schema_version(self) -> int:
        return 3 if self.receipt["schema"] == NEST_CHILD_WINDOW_SCHEMA_V3 else 2

    @property
    def has_species(self) -> bool:
        """``True`` when records carry ``qc``/``qr``/``qis``/``qg`` (schema v3 species)."""

        return "species" in self.receipt

    def deck_for(self, record: WindowRecord) -> ParcelDeck:
        np = _np()
        r = self.receipt
        child = r["child"]
        return ParcelDeck(
            name="child-window", nx=record.nx, ny=record.ny, nz=record.nz,
            dx_m=float(np.float32(child["dx_m"])), dy_m=float(np.float32(child["dy_m"])),
            minx_m=float(np.float32(record.origin_x_m)), miny_m=float(np.float32(record.origin_y_m)),
            dz_reference_m=float(r["deck"]["dz_reference_m"]),
            zf=np.asarray(r["zf_cm1_indexed"], dtype=np.float32),
            zh=np.asarray(r["zh_cm1_indexed"], dtype=np.float32),
            mf=np.asarray(r["mf_cm1_indexed"], dtype=np.float32),
            cgs=tuple(float(v) for v in r["cgs"]),  # type: ignore[arg-type]
            cgt=tuple(float(v) for v in r["cgs"]),  # type: ignore[arg-type]
            lateral=(2, 2, 2, 2), periodic=False,
            vertical_sha256=str(r["deck"]["vertical_sha256"]),
        )

    def fields(self, index: int, *, verify: bool = False) -> dict[str, Any]:
        """One record's words, reshaped, as read-only views of the arena.

        A schema-v3 record carries no ``qhyd``; the returned mapping adds it,
        rebuilt from the species (:func:`rebuild_qhyd`), so every consumer of
        the condensate loading -- the trajectory budget's total density and
        buoyancy -- reads either schema unchanged.  It is the one returned
        array that is not a view of the arena.
        """

        np = _np()
        record = self.records[index]
        if self._raw is None:
            self._raw = np.memmap(self.directory / WINDOW_ARENA_FILENAME, dtype="<f4", mode="r")
        base = record.offset_bytes // 4
        if verify:
            raw = np.ascontiguousarray(
                self._raw[base:base + record.bytes // 4]).tobytes(order="C")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != record.sha256:
                raise NestChildWindowError(
                    f"record {record.index} of {self.directory} hashes to {digest}, not the "
                    f"{record.sha256} its index row states; the arena has been truncated or "
                    "rewritten and the trajectories through it would be through other words"
                )
        out: dict[str, Any] = {}
        for entry in self.receipt["fields"]:
            shape = tuple(int(v) for v in entry["shape_zyx"])
            at = base + int(entry["offset_words"])
            out[entry["name"]] = self._raw[at:at + int(entry["words"])].reshape(shape)
        if "qhyd" not in out:
            out["qhyd"] = rebuild_qhyd(out)
        return out

    def surface_for(self, index: int, deck: ParcelDeck) -> Any:
        """The record's :class:`cm1core.nest_parcels.SemiSlip`, or ``None``."""

        child = self.receipt["child"]
        if int(child["bbc"]) != 3:
            return None
        from .nest_parcels import SemiSlip

        return SemiSlip(self.fields(index)["znt"], int(child["imove"]),
                        float(child["u_move_m_s"]), float(child["v_move_m_s"]))

    def frame(self, index: int, *, verify: bool = False, time_shift_s: float = 0.0) -> Any:
        """One record as a :class:`cm1core.nest_parcel_backward.ChildFrame`.

        The frame's ``free_interior`` is the window's whole physical rectangle:
        its ``spec_zone`` and ``relax_zone`` are zero because the *window* has
        neither -- its boundary rows are child interior rows, and the three
        ghost columns beyond them give every sample inside the rectangle its
        full ``parcel.F`` stencil.

        ``time_shift_s`` is subtracted from the record's label.  A record is
        labelled with the END of the child step whose average wind it holds;
        a reader that blends records in time (``history-blend`` on a strided
        stream) must place each at the step's MIDPOINT, ``child_dt/2``
        earlier, or every blended wind would be half a step late.
        """

        from .nest_parcel_backward import ChildFrame

        record = self.records[index]
        deck = self.deck_for(record)
        words = self.fields(index, verify=verify)
        child = self.receipt["child"]
        placement = {
            "spec_zone": 0, "relax_zone": 0,
            "origin_x_m": record.origin_x_m, "origin_y_m": record.origin_y_m,
            "dx_m": deck.dx_m, "dy_m": deck.dy_m, "nx": record.nx, "ny": record.ny,
            "nz": record.nz, "bbc": int(child["bbc"]), "tbc": int(child["tbc"]),
            "u_move_m_s": float(child["u_move_m_s"]), "v_move_m_s": float(child["v_move_m_s"]),
            "imove": int(child["imove"]), "parent_i0": None, "parent_j0": None,
            "winds_published": "advecting velocity (solve2.F:1290-1301)",
            "child_step": record.child_step, "centre_held": record.centre_held,
            "clamped": record.clamped,
        }
        return ChildFrame(
            index=record.index, model_time_s=record.model_time_s - float(time_shift_s),
            placement=placement,
            deck=deck, u=words["uadv"], v=words["vadv"], w=words["wadv"],
            arena_sha256=record.sha256, directory=str(self.directory),
            surface=self.surface_for(index, deck),
        )

    def frames(self, *, verify: bool = False, time_shift_s: float = 0.0) -> list[Any]:
        return [self.frame(q, verify=verify, time_shift_s=time_shift_s)
                for q in range(len(self.records))]

    @property
    def midpoint_shift_s(self) -> float:
        """Half a child step: record label minus the midpoint of the step it averages."""

        return 0.5 * float(self.receipt["child"]["dt_s"])


def read_window_stream(root: Path | str) -> WindowStream:
    """Open a published window stream for reading."""

    directory = Path(root)
    receipt, records = read_window_index(directory)
    if not records:
        raise NestChildWindowError(
            f"{directory} holds no record; the stream opened and closed without the criterion "
            "ever finding a centre, or outside its declared interval, and there is nothing to "
            "integrate a trajectory through"
        )
    return WindowStream(directory, receipt, records)


__all__ = [
    "CM1_AUTHORITY", "GM1_CHOICES", "INDEX_COLUMNS", "NEST_CHILD_WINDOW_DIRECTORY",
    "NEST_CHILD_WINDOW_GATE", "NEST_CHILD_WINDOW_ROTATING_SCHEMA", "NEST_CHILD_WINDOW_SCHEMA",
    "NEST_CHILD_WINDOW_SCHEMAS", "NEST_CHILD_WINDOW_SCHEMA_V3", "NG", "NestChildWindowError",
    "ROTATION_AUTHORITY", "check_rotation_header", "lspgrad1_profiles", "rotation_header",
    "NestChildWindowStream", "UNVALIDATED_MARKER", "WINDOW_ARENA_FILENAME",
    "WINDOW_FIELDS", "WINDOW_FIELDS_V3", "WINDOW_INDEX_FILENAME", "WINDOW_RECEIPT_FILENAME",
    "WINDOW_SPECIES_COMPONENTS", "WINDOW_SPECIES_FIELDS", "WINDOW_SPECIES_SOURCES",
    "WINDOW_SURFACE_FIELD", "WindowBox", "WindowLayout", "WindowRecord", "WindowStream",
    "cut_window", "fill_window_w_ghost", "nest_child_window_admission_defect", "place_window",
    "read_window_index", "read_window_stream", "rebuild_qhyd",
    "require_nest_child_window_admission", "window_layout",
    "stream_bytes_per_model_minute", "window_cells", "window_deck",
]
