"""Capabilities this port exposes and has not measured against CM1.

A capability becomes selectable the moment it is reachable.  Anything exposed
without a measurement carries a row here, and that row is loud in two places a
user actually reads: ``gpuwm-cm1 doctor`` lists every row, and a run receipt
lists the rows the run's own document selected.  Silence is the failure mode
this module exists to prevent -- a capability that is neither refused nor
marked reads as validated, and nothing downstream can tell the difference.

A row is not a refusal.  A refusal names a wrong result and stops the run; a
row here names a result nobody has *checked*, and the run proceeds.  The two
are written differently on purpose: ``not_checked`` says what was not measured
and ``would_show`` says what a wrong answer would look like in the output, so a
reader can decide whether this run's conclusion depends on it.

Retiring a row is the same act as retiring a guard: when the measurement is
made, delete the row and record the measurement.  A row that survives its own
validation is as misleading as a missing one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any


UNVALIDATED_REGISTER_SCHEMA = "gm1.unvalidated-capability-register/v1"

#: The marker string.  Machine-readable, greppable, and the same word in the
#: receipt, in doctor and in this module.
UNVALIDATED = "unvalidated"


@dataclass(frozen=True, slots=True)
class UnvalidatedCapability:
    """One exposed, unmeasured capability."""

    #: The configuration key a user types, dotted exactly as the TOML spells it.
    option: str
    #: The values this row covers.  Empty means every value the key admits.
    values: tuple[Any, ...]
    #: What was not measured.  Not "this is untested" -- the specific check.
    not_checked: str
    #: What a wrong answer would look like in the run's own output.
    would_show: str
    #: CM1 r21.1 file:line the capability is implemented from.
    authority: str
    #: A further condition on the whole record, for a row that covers some
    #: documents with this value and not others.  ``None`` selects on the
    #: value alone.  Not part of the receipt record: ``not_checked`` says in
    #: words which documents the row covers.
    applies: Callable[[Mapping[str, Any]], bool] | None = field(default=None, compare=False)
    #: The row's words for one record, for a row whose reason depends on the
    #: record: ``(not_checked, would_show)``, or ``None`` to keep the row's
    #: own.  ``selected`` hands a run the row in these words, so its receipt
    #: says why this run carries it.  Not part of the receipt record.
    explain: Callable[[Mapping[str, Any]], tuple[str, str] | None] | None = field(default=None, compare=False)

    def covers(self, value: Any) -> bool:
        if not self.values:
            return True
        return any(value == candidate and type(value) is type(candidate) for candidate in self.values)

    def selects(self, record: Mapping[str, Any]) -> bool:
        if self.option not in record or not self.covers(record[self.option]):
            return False
        return self.applies is None or bool(self.applies(record))

    def for_record(self, record: Mapping[str, Any]) -> "UnvalidatedCapability":
        """This row in the words that fit ``record`` (itself when it has one set of words)."""

        words = self.explain(record) if self.explain is not None else None
        if not words:
            return self
        return replace(self, not_checked=words[0], would_show=words[1])

    def record(self) -> dict[str, Any]:
        return {
            "marker": UNVALIDATED,
            "option": self.option,
            "values": list(self.values),
            "not_checked": self.not_checked,
            "would_show": self.would_show,
            "authority": self.authority,
        }

    def line(self) -> str:
        values = "any value" if not self.values else ", ".join(repr(v) for v in self.values)
        return f"{UNVALIDATED} {self.option} ({values}): {self.not_checked}; a wrong answer would show as {self.would_show}"


#: How far apart three layer depths may be, as a fraction of the deepest of
#: them, and still be one depth for this register.  A column read from CM1's
#: ``input_grid_z`` carries binary32 heights: rebuilt into faces, the compared
#: 53-level column's three top layers come out 500.0039, 499.9961 and
#: 500.0039 m, one depth written three ways (1.6e-5 apart).  Stretching moves a
#: layer by percents (CM1's hurricane column: 50, 73.7 and 97.4 m at the
#: ground).  The kernels' own choice of weights stays exact
#: (``VerticalLevels.fornberg_matches_pinned_kernels``).
EQUAL_LAYERS_FRACTION = 1e-3

#: How far a w level may sit from a compared column's, as a fraction of that
#: column's thinnest layer, for the levels to be that column.  The compared
#: 53 levels written as ``input_grid_z`` heights agree with it to 3 mm.
SAME_COLUMN_FRACTION = 1e-3

#: The stretched column compared with CM1 in a storm with cloud: CM1's stock
#: supercell with Morrison for two hours on these 53 levels (100 m layers at
#: the ground growing to 500 m above 9.3 km, under a 19.3 km top), all ten
#: mature-phase statistics inside the release bands
#: (``evidence/oracle/stretched-supercell``).
COMPARED_STRETCHED_COLUMN: Mapping[str, Any] = MappingProxyType({
    "grid.stretch_z": 1, "grid.nz": 53, "grid.dz_m": 500.0, "grid.ztop_m": 19300.0,
    "grid.str_bot_m": 300.0, "grid.str_top_m": 9300.0, "grid.dz_bot_m": 100.0, "grid.dz_top_m": 500.0,
})

#: The hurricane presets' levels: 59, three equal 50 m layers at the ground and
#: 500 m ones above 4 km, under a 25 km top.  CM1 r21.1 ran the 8 km hurricane
#: on them with the presets' changed settings (CM1's hurricane vortex,
#: ``iinit = 7``; Morrison; a constant ocean drag, ``bbc = 3`` and
#: ``sfcmodel = 1``), and GM1's run of it was compared with CM1's over four
#: days (``records/GM1-HURRICANE-RUNS.json``; the cards say what it showed).
HURRICANE_PRESET_COLUMN: Mapping[str, Any] = MappingProxyType({
    "grid.stretch_z": 1, "grid.nz": 59, "grid.dz_m": 500.0, "grid.ztop_m": 25000.0,
    "grid.str_bot_m": 150.0, "grid.str_top_m": 4000.0, "grid.dz_bot_m": 50.0, "grid.dz_top_m": 500.0,
})

#: The one scheme stretched levels were compared with in a storm.
COMPARED_STRETCHED_SCHEME = "morrison-mp10"

#: Why a stretched column carries the row (``stretched_levels_reasons``).
REASON_ENDS = "ends"
REASON_DEEPER = "deeper"
REASON_LEVELS = "levels"
REASON_SCHEME = "scheme"

#: The schemes other than Morrison, as the row names them, with CM1's ptype.
STRETCHED_SCHEME_WORDS: Mapping[str, tuple[str, str]] = MappingProxyType({
    "wsm6-mp6": ("WSM6", "ptype = 7"),
    "thompson-mp8": ("Thompson", "ptype = 3"),
    "thompson-aerosol-mp28": ("Thompson with aerosols", "no CM1 ptype"),
})

_NOT_CHECKED_BY_REASON: Mapping[str, str] = MappingProxyType({
    REASON_ENDS: (
        "levels whose lowest or highest three layers are not equally deep "
        "(str_bot = 0, or str_top = ztop) run, and were compared with CM1 in "
        "one storm, CM1's stock supercell with Morrison for two hours on "
        "CM1's 59-level hurricane column: with ground friction (bbc = 3, "
        "sfcmodel = 1, cecd = 1) all ten mature-phase statistics are inside "
        "the release bands (0.987 to 1.086 of CM1's); with a free-slip ground "
        "nine are within 2.4 % and max vertical vorticity came out 15.6 % "
        "high, outside its 10 % band"
    ),
    REASON_DEEPER: (
        "these levels reach higher than the one stretched column compared with "
        "CM1 in a storm with cloud (CM1's stock supercell with Morrison on 53 "
        "levels under a 19.3 km top, all ten statistics inside the release "
        "bands); on a 25.15 km column with three equal 50 m layers at the "
        "ground (nz = 62, str_bot = 150, str_top = 5650) the same supercell with "
        "a free-slip ground had max vertical vorticity 11.9 % high, outside its "
        "10 % band"
    ),
    REASON_LEVELS: (
        "these are not the stretched levels compared with CM1 in a storm with "
        "cloud (CM1's stock supercell with Morrison on nz = 53, ztop = 19300, "
        "str_bot = 300, str_top = 9300, dz_bot = 100, dz_top = 500; all ten "
        "statistics inside the release bands), and no storm on these levels "
        "has been compared with CM1"
    ),
    REASON_SCHEME: (
        "{scheme} ({ptype}) has not been run on stretched levels: every storm "
        "compared with CM1 on stretched levels used Morrison (ptype = 5)"
    ),
})

_WOULD_SHOW_BY_REASON: Mapping[str, str] = MappingProxyType({
    REASON_ENDS: (
        "a largest vertical vorticity aloft late in the storm above CM1's "
        "(12 to 14 km in the last half hour of that run); the first rain at "
        "the ground, the rain total and the other statistics agreed with "
        "CM1's in both runs"
    ),
    REASON_DEEPER: (
        "a largest vertical vorticity aloft late in the storm above CM1's "
        "(13 to 14 km, almost all of it in the last frame of that run); the "
        "other nine statistics were within 2.2 % of CM1's"
    ),
    REASON_LEVELS: (
        "storm statistics (the updraft, the vorticity, the cold pool, the "
        "rain) further from CM1's than the release bands allow"
    ),
    REASON_SCHEME: (
        "rain and hail reaching the ground early or late, or in the wrong "
        "amount, where they fall through layers of changing depth"
    ),
})


def _stretched(record: Mapping[str, Any]) -> bool:
    stretch_z = record.get("grid.stretch_z")
    return stretch_z in (1, 3, 4) and not isinstance(stretch_z, bool)


def _record_levels(record: Mapping[str, Any]) -> Any:
    """The column the record's ``grid.*`` words build, or ``None``.

    A record whose column cannot be built (a missing or refused word) gives
    ``None``: the row reports what a run chose, and such a document does not
    run.
    """

    from types import SimpleNamespace

    from .cm1_vertical_grid import VerticalGridError, build_vertical_levels
    from .config import _derived_top

    if not _stretched(record):
        return None
    stretch_z = record["grid.stretch_z"]
    try:
        nz = record["grid.nz"]
        dz_m = float(record["grid.dz_m"])
        levels_m = tuple(float(value) for value in (record.get("grid.levels_m") or ()))
        if stretch_z == 1:
            words = {
                "ztop_m": float(record.get("grid.ztop_m", 0.0)),
                "str_bot_m": float(record.get("grid.str_bot_m", 0.0)),
                "str_top_m": float(record.get("grid.str_top_m", 0.0)),
                "dz_bot_m": float(record.get("grid.dz_bot_m", 0.0)),
                "dz_top_m": float(record.get("grid.dz_top_m", 0.0)),
            }
        else:
            top = _derived_top(SimpleNamespace(stretch_z=stretch_z, dz_m=dz_m, nz=nz, levels_m=levels_m))
            words = {"ztop_m": top, "levels_m": levels_m}
        return build_vertical_levels(stretch_z=stretch_z, nk=nz, dz_m=dz_m, **words)
    except (KeyError, TypeError, ValueError, IndexError, VerticalGridError):
        return None


@lru_cache(maxsize=None)
def _compared_levels(which: str) -> Any:
    words = COMPARED_STRETCHED_COLUMN if which == "supercell" else HURRICANE_PRESET_COLUMN
    return _record_levels(words)


def _one_depth(depths: Sequence[float]) -> bool:
    return max(depths) - min(depths) <= EQUAL_LAYERS_FRACTION * max(depths)


def _end_layers_equal(levels: Any) -> bool:
    faces = levels.face_heights_m
    if len(faces) < 4:
        return bool(levels.fornberg_matches_pinned_kernels())
    bottom = [faces[index + 1] - faces[index] for index in range(3)]
    top = [faces[-1 - index] - faces[-2 - index] for index in range(3)]
    return _one_depth(bottom) and _one_depth(top)


def _same_column(levels: Any, compared: Any) -> bool:
    faces, reference = levels.face_heights_m, compared.face_heights_m
    if len(faces) != len(reference):
        return False
    slack = SAME_COLUMN_FRACTION * min(compared.spacings_m)
    return all(abs(face - other) <= slack for face, other in zip(faces, reference))


def _hurricane_case(record: Mapping[str, Any]) -> bool:
    """CM1's hurricane vortex over surface drag: the case CM1 ran on the presets' levels."""

    return (record.get("initialization.iinit") == 7 and record.get("closures.bbc") == 3
            and record.get("closures.sfcmodel") == 1)


def end_layers_unequal(record: Mapping[str, Any]) -> bool:
    """Whether the record's column has unequally deep end layers.

    True when the lowest three or the highest three layers of the column the
    record's ``grid.*`` words build are not one depth within
    :data:`EQUAL_LAYERS_FRACTION` (``str_bot = 0``, or ``str_top = ztop``, or
    an explicit level list that stretches at an end): the columns whose face
    density takes the column's own ``cgs`` and ``cgt`` instead of the sealed
    1.875, -1.25 and 0.375, apart from binary32 heights of an equally spaced
    end, which this test reads as the equal layers they were written from.  A
    record whose column cannot be built is False.
    """

    levels = _record_levels(record)
    return levels is not None and not _end_layers_equal(levels)


def stretched_levels_reasons(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Why the record's stretched column is not one compared with CM1 in a storm.

    Empty when it is, or when the record is not on stretched levels.  In order:

    * ``"ends"``: the lowest or highest three layers are not equally deep, in
      any run (the 0.1.7 row);
    * ``"deeper"`` or ``"levels"``, in a run with cloud: equal end layers, but
      not the 53 levels compared with CM1 in a storm, and reaching higher
      than they do, or not.  The hurricane presets' own levels with CM1's
      hurricane vortex over surface drag are not either: CM1 ran that storm on
      them;
    * ``"scheme"``, in a run with cloud: a scheme other than Morrison, which no
      storm on stretched levels has used.

    A dry run (``physics.component = "off"``) on equal end layers carries no
    row: stretched levels are compared with CM1 without cloud, word for word.
    A record that does not carry ``physics.component`` is read as dry.
    """

    levels = _record_levels(record)
    if levels is None:
        return ()
    component = record.get("physics.component")
    cloud = isinstance(component, str) and component != "off"
    reasons: list[str] = []
    if not _end_layers_equal(levels):
        reasons.append(REASON_ENDS)
    elif cloud:
        supercell = _compared_levels("supercell")
        hurricane = _compared_levels("hurricane")
        if _same_column(levels, supercell):
            pass
        elif _same_column(levels, hurricane) and _hurricane_case(record):
            pass
        else:
            highest = supercell.face_heights_m[-1] + SAME_COLUMN_FRACTION * min(supercell.spacings_m)
            reasons.append(REASON_DEEPER if levels.face_heights_m[-1] > highest else REASON_LEVELS)
    if cloud and component != COMPARED_STRETCHED_SCHEME:
        reasons.append(REASON_SCHEME)
    return tuple(reasons)


def stretched_levels_not_compared(record: Mapping[str, Any]) -> bool:
    """Whether the record's stretched column carries the row (any reason)."""

    return bool(stretched_levels_reasons(record))


def stretched_scheme_words(component: Any) -> tuple[str, str]:
    """``(name, CM1 ptype)`` of a scheme other than Morrison, as the row says them."""

    return STRETCHED_SCHEME_WORDS.get(str(component), (str(component), "no CM1 ptype"))


def _stretched_levels_words(record: Mapping[str, Any]) -> tuple[str, str] | None:
    reasons = stretched_levels_reasons(record)
    if not reasons:
        return None
    scheme, ptype = stretched_scheme_words(record.get("physics.component"))
    not_checked = "; ".join(
        _NOT_CHECKED_BY_REASON[reason].format(scheme=scheme, ptype=ptype) for reason in reasons
    )
    would_show = "; or ".join(_WOULD_SHOW_BY_REASON[reason] for reason in reasons)
    return not_checked, would_show


#: Grid geometry.  ``grid.stretch_z`` 1, 3 and 4 are each measured against CM1
#: r21.1 in a dry run (``evidence/oracle/stretched-column-32``,
#: ``.../explicit-w-levels-32`` and ``.../explicit-scalar-levels-32``), and one
#: column has been compared with CM1 once cloud forms:
#: ``evidence/oracle/stretched-supercell`` holds CM1's stock supercell on 53
#: stretched levels (100 m layers at the ground growing to 500 m) for two
#: hours with Morrison microphysics, all ten statistics inside the release
#: bands.  That column with Morrison carries no row, and nor do the hurricane
#: presets' levels with CM1's hurricane vortex over surface drag, on which CM1
#: ran the 8 km hurricane (``records/GM1-HURRICANE-RUNS.json``).
#:
#: Any other stretched column carries the row in a run with cloud, and so does
#: a column whose end layers are not equally deep in any run, each with its
#: reason (``stretched_levels_reasons``).  The same supercell on CM1's
#: hurricane column (``str_bot = 0``,
#: ``evidence/oracle/stretched-ground-supercell``) is inside the bands with
#: ground friction, but with a free-slip ground one statistic is outside its
#: band, and on a 25 km column with three equal layers at the ground too.
#: The row says so; it does not stop the run.
UNVALIDATED_CAPABILITIES: tuple[UnvalidatedCapability, ...] = (
    UnvalidatedCapability(
        option="grid.stretch_z",
        values=(1, 3, 4),
        not_checked=(
            "stretched levels run, and one column has been compared with CM1 "
            "in a storm with cloud: CM1's stock supercell with Morrison for two "
            "hours on 53 levels (100 m layers at the ground growing to 500 m "
            "above 9.3 km, under a 19.3 km top), all ten mature-phase "
            "statistics inside the release bands. That column with Morrison "
            "carries no row, and nor do the hurricane presets' levels with "
            "CM1's hurricane vortex over surface drag and Morrison, on which "
            "CM1 ran the 8 km hurricane. A run with cloud on any other "
            "stretched column carries this row, and so does every column whose "
            "lowest or highest three layers are not equally deep (str_bot = 0, "
            "or str_top = ztop): on CM1's 59-level hurricane column the same "
            "supercell with ground friction (bbc = 3, sfcmodel = 1, cecd = 1) "
            "has all ten statistics inside the release bands (0.987 to 1.086 "
            "of CM1's), and with a free-slip ground max vertical vorticity came "
            "out 15.6 % high, outside its 10 % band; on a 25.15 km column with "
            "three equal 50 m layers at the ground it came out 11.9 % high. "
            "WSM6 and Thompson have not been run on stretched levels"
        ),
        would_show=(
            "a largest vertical vorticity aloft late in the storm above CM1's "
            "(12 to 14 km in the last half hour of the runs on 25 km columns), "
            "or, with WSM6 or Thompson, rain and hail reaching the ground early "
            "or late; the first rain at the ground, the rain total and the "
            "other statistics agreed with CM1's in the Morrison runs"
        ),
        authority="src/param.F:8985-9060;src/solve3.F:678-679",
        applies=stretched_levels_not_compared,
        explain=_stretched_levels_words,
    ),
)


def selected(record: Mapping[str, Any]) -> tuple[UnvalidatedCapability, ...]:
    """The rows one configuration record selects, each in the words that fit it.

    ``record`` is a flat mapping of dotted option to value, as a caller builds
    it from the document it admitted.  A key the caller does not carry selects
    nothing: this reports what a run chose, never what it might have chosen.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of dotted option to value")
    return tuple(row.for_record(record) for row in UNVALIDATED_CAPABILITIES if row.selects(record))


def selected_by_document(config: Any) -> tuple[UnvalidatedCapability, ...]:
    """The rows one admitted ``CM1Config`` selects.

    The document is flattened through its own canonical projection, so the
    keys this reads are exactly the keys the run's ``config_sha256`` covers.
    """

    flat: dict[str, Any] = {}
    for table, fields in config.canonical_dict().items():
        if isinstance(fields, Mapping):
            for name, value in fields.items():
                flat[f"{table}.{name}"] = value
    return selected(flat)


#: CM1 r21.1's own source for each column component's scheme, for the
#: component rows below.  A component with no CM1 counterpart says so.
_COMPONENT_AUTHORITY = {
    # CM1 r21.1 src/param.F: each ptype's setup block, ELSEIF to ELSEIF.
    "morrison-mp10": "src/morrison.F; src/param.F:3400-3631 (ptype=5)",
    "thompson-mp8": "src/thompson.F; src/param.F:3322-3399 (ptype=3)",
    "thompson-aerosol-mp28": (
        "none: src/param.F:3141-4509 numbers no aerosol-aware Thompson scheme"
    ),
    "wsm6-mp6": "src/mp_wsm6.F; src/param.F:3632-3713 (ptype=7)",
}

#: What a wrong answer would look like, for the dynamics rows the value-aware
#: register can report.  The register itself says what was not checked.
_DYNAMICS_WOULD_SHOW = {
    "dynamics.kdiv": (
        "acoustic divergence damping of a strength CM1 would not apply at this "
        "kdiv: pressure and vertical-velocity noise left in the fields, or "
        "divergent motions damped that CM1 keeps"
    ),
    "dynamics.lspgrad": (
        "a near-surface wind and 0-1 km storm-relative helicity that drift from "
        "CM1's below about 700 m while the large-scale pressure gradient acts"
    ),
}


def component_row(component_id: str) -> UnvalidatedCapability | None:
    """The row for a column component with no measurement, else ``None``."""

    from .component_validation import UNVALIDATED_MARKER, component_validation

    state = component_validation(component_id)
    if state.validated:
        return None
    return UnvalidatedCapability(
        option="physics.component",
        values=(component_id,),
        not_checked=(
            f"{UNVALIDATED_MARKER} {component_id} ({state.state}): "
            + "; ".join(state.not_checked)
        ),
        would_show=(
            "a moist trajectory that departs from native CM1's scheme at this "
            "ptype: hydrometeor mass and number, latent heating and so updraft "
            "and downdraft strength, the cold pool and the surface rain"
        ),
        authority=_COMPONENT_AUTHORITY.get(component_id, "not stated"),
    )


def dynamics_rows(dynamics: Any) -> tuple[UnvalidatedCapability, ...]:
    """The ``[dynamics]`` rows unvalidated at the values this run uses."""

    from .dynamics_validation_register import unvalidated_dynamics_capabilities

    return tuple(
        UnvalidatedCapability(
            option=item.key,
            values=(value,) if value is not None else (),
            not_checked=item.not_checked,
            would_show=_DYNAMICS_WOULD_SHOW.get(
                item.key,
                f"results that differ from CM1 r21.1's wherever {item.cm1_name} acts",
            ),
            authority=item.authority,
        )
        for item, value in unvalidated_dynamics_capabilities(dynamics)
    )


def initialization_rows(initialization: Mapping[str, Any] | None) -> tuple[UnvalidatedCapability, ...]:
    """The initialization selectors the run's own validation block flags."""

    if not initialization:
        return ()
    return tuple(
        UnvalidatedCapability(
            option=f"initialization.{item['selector']}",
            values=(item["value"],),
            not_checked=str(item.get("not_checked") or "not stated"),
            would_show=(
                "an initial state that differs from the one CM1 builds for this "
                "selector value: the sounding, the perturbation or the balance "
                "the run starts from"
            ),
            authority=str(item.get("evidence") or item.get("method") or "not stated"),
        )
        for item in initialization.get("items", ())
    )


def selected_for_run(config: Any, *, initialization: Mapping[str, Any] | None = None,
                     ) -> tuple[UnvalidatedCapability, ...]:
    """Every unvalidated capability one run uses: the receipt's top-level rows.

    The capability rows the document selects (:func:`selected_by_document`),
    then the column component when its state is not ``measured``
    (``component_validation``), then the ``[dynamics]`` rows unvalidated at
    the values the document gives them (``kdiv`` at CM1's 0.10 is not one),
    then the initialization selectors ``initialization`` (the run's
    ``unvalidated_initialization`` block) flags.  A reader of the top-level
    register sees every one of them without opening the nested blocks.
    """

    rows = list(selected_by_document(config))
    physics = getattr(config, "physics", None)
    component = getattr(physics, "component", None)
    # "off" is a dry run: no column component, nothing to have measured.
    if isinstance(component, str) and component != "off":
        row = component_row(component)
        if row is not None:
            rows.append(row)
    dynamics = getattr(config, "dynamics", None)
    if dynamics is not None:
        rows.extend(dynamics_rows(dynamics))
    rows.extend(initialization_rows(initialization))
    return tuple(rows)


def receipt(rows: Sequence[UnvalidatedCapability] | None = None) -> dict[str, Any]:
    """The receipt block.  Present even when empty, so absence is a statement."""

    rows = UNVALIDATED_CAPABILITIES if rows is None else tuple(rows)
    return {
        "schema": UNVALIDATED_REGISTER_SCHEMA,
        "marker": UNVALIDATED,
        "count": len(rows),
        "capabilities": [row.record() for row in rows],
    }


__all__ = [
    "COMPARED_STRETCHED_COLUMN",
    "EQUAL_LAYERS_FRACTION",
    "HURRICANE_PRESET_COLUMN",
    "SAME_COLUMN_FRACTION",
    "UNVALIDATED",
    "UNVALIDATED_CAPABILITIES",
    "UNVALIDATED_REGISTER_SCHEMA",
    "UnvalidatedCapability",
    "component_row",
    "dynamics_rows",
    "end_layers_unequal",
    "initialization_rows",
    "receipt",
    "selected",
    "selected_by_document",
    "selected_for_run",
    "stretched_levels_not_compared",
    "stretched_levels_reasons",
    "stretched_scheme_words",
]
