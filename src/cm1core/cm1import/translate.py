"""The one CM1 namelist -> gpuwm-cm1 TOML translator.

Both callers use it:

* the differential harness (:mod:`cm1core.oracle.spec`), which starts from a
  CM1 base deck, applies a case's ``[cm1]`` overrides and forces its own
  ``&param9`` selection (``HARNESS_OUTPUT``) on both models;
* the namelist importer (:func:`translate_parsed`), which starts from a
  user's namelist, fills the keys it omits with CM1 r21.1's own code defaults,
  maps ``ptype`` to a physics component, and keeps the user's ``&param9``
  selection.

Every key goes through ``records/CM1-GM1-KEYMAP.json``.  The transforms CM1
applies implicitly are applied here once: ``zd`` becomes the damping depth
below the model top, ``nsound`` is derived with CM1's own ``dtsmall``,
all-or-none groups are completed at CM1's shipped values, ``bbc`` is rewritten
for a surface layer as ``param.F`` does, and the stretch and radiation
branches drop the words their branch does not read.

An unmapped physical key is judged against what GM1 actually integrates
(``records/CM1-GM1-KEYMAP.json``'s ``gm1_behaviour``), not against CM1's
distributed deck: a deck that sets a key to CM1's own shipped value is still a
difference when GM1 runs something else, and a key the r21.1 inventory does
not contain is named rather than skipped.  Nothing here needs a CM1 build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from . import keymap as _keymap
from . import defaults as _defaults

#: The gpuwm-cm1 tables, in the order a generated document writes them.  One
#: entry per table, in read order.
TABLE_ORDER = (
    "run",
    "initialization",
    "grid",
    "dynamics",
    "boundaries",
    "closures",
    "surface",
    "runtime",
    "physics",
    "radiation",
    "output",
    "diagnostics",
    "lagrangian",
)

#: The &param9 rotation products gpuwm-cm1 refuses on a non-uniform column.
UNIFORM_METRIC_ONLY = frozenset(
    {"output_vort", "output_pv", "output_uh", "output_shs", "output_svs"}
)

#: The &param9 swaths of CM1's rainwater and graupel/hail rows: the most qr and
#: qg at the lowest level, every step (solve3.F:993-1047, ``qa(i,j,1,nqr)`` and
#: ``qa(i,j,1,nqg)``).  CM1 writes them for every scheme with those rows and
#: turns them off where ``nqr`` or ``nqg`` is 0 (param.F:4727-4728).  Every GM1
#: component carries CM1's six mass rows first (moisture_species.py,
#: ``require_cm1_mass_prefix``), so qr and qg are rows 3 and 6 for each of them,
#: as they are for CM1's ptype 3, 5 and 7.
RAIN_GRAUPEL_SWATHS = frozenset({"output_srs", "output_sgs"})

#: The owning table of each &param9 word gpuwm-cm1 spells outside ``[output]``.
GM1_OUTPUT_OWNERS: Mapping[str, str] = {
    "output_dbz": "physics",
    "output_cape": "diagnostics",
    "output_cin": "diagnostics",
    "output_lcl": "diagnostics",
    "output_lfc": "diagnostics",
    "output_uh": "diagnostics",
    "output_vort": "diagnostics",
    "output_pv": "diagnostics",
    "output_sws": "diagnostics",
    "output_svs": "diagnostics",
    "output_sps": "diagnostics",
    "output_srs": "diagnostics",
    "output_sgs": "diagnostics",
    "output_sus": "diagnostics",
    "output_shs": "diagnostics",
}

#: [diagnostics] takes its switches as CM1's own 0/1 integers except
#: output_cape/cin/lcl/lfc, which rel/capediag admits as TOML booleans.
_BOOLEAN_DIAGNOSTICS = frozenset({"output_cape", "output_cin", "output_lcl", "output_lfc"})

#: Each all-or-none key group a generated deck has to complete.  The door
#: refuses a half-declared group by name, so a deck that sets one word of a
#: group is emitted with the rest of it at CM1's own shipped values.  The key
#: is the group; the section is its first "_"-separated part.
GROUP_DEFAULTS: dict[str, dict[str, object]] = {
    "closures": {
        "bbc": 1, "sfcmodel": 0, "isfcflx": 0, "cecd": 3, "cnstcd": 0.001,
    },
    "closures_surface_layer": {
        "cnstce": 0.001, "set_znt": 0, "cnst_znt": 0.16,
        "set_ust": 0, "cnst_ust": 0.25,
    },
    "surface": {
        "initsfc": 1, "tsk0_k": 299.28, "tmn0_k": 297.28, "xland0": 2.0,
        "lu0": 16, "season": 1, "oceanmodel": 1,
    },
    # getcape.F's own source parameters (:60, :65-68, :70, :73-77) and the
    # mover the t=0 SRH is taken against; CM1's shipped values, so a deck that
    # asks for a plane and nothing else gets exactly CM1's arithmetic.
    "diagnostics": {
        "output_cape": False, "output_cin": False,
        "output_lcl": False, "output_lfc": False,
        "parcel_source": 3, "adiabat": 1,
        "mixed_layer_depth_m": 500.0, "pressure_increment_pa": 200.0,
        "storm_motion": "bunkers-right",
    },
    # misclibs.F's rotation diagnostics and solve3.F's run-long swaths, off as
    # CM1's own run/namelist.input:252-268 ships them.
    "diagnostics_rotation": {
        "output_uh": 0, "output_vort": 0, "output_pv": 0,
        "output_sws": 0, "output_svs": 0, "output_sps": 0,
        "output_srs": 0, "output_sgs": 0, "output_sus": 0, "output_shs": 0,
        "tornado_report": False,
    },
}

#: GM1 keys a component may or may not read, and the components that read
#: them.  CM1's distributed deck declares ``ihail`` and ``ndcnst``
#: unconditionally, so a deck that selects a component which reads neither
#: would otherwise be handed a value gpuwm-cm1 refuses by name.  The
#: translator records the fact instead of emitting the key.
COMPONENT_SCOPED_KEYS: Mapping[str, frozenset[str]] = {
    "physics.ihail": frozenset({"wsm6-mp6", "morrison-mp10", "wdm6-mp16"}),
    "physics.ndcnst": frozenset({"morrison-mp10"}),
}

#: ``physics.component`` <-> CM1 ``(imoist, ptype)``.
#:
#: README.namelist:458-495 and src/param.F:3141-4509 are the table.  WSM6 is
#: ``ptype=7`` (``src/param.F:3632``, the branch CM1 labels WSM6);
#: Morrison is ``ptype=5`` (``src/param.F:3400``).  ``native-cm1-ptype5`` keeps
#: ``ptype=5``: that is what its name says it is.  ``wdm6-mp16`` has no entry
#: at all -- CM1 r21.1 has no WDM6 scheme.  Both callers read the selectors an
#: unmapped key hangs on with the component's own (imoist, ptype).
COMPONENT_TO_CM1: Mapping[str, tuple[int, int]] = {
    "off": (0, 0),
    "wsm6-mp6": (1, 7),
    "morrison-mp10": (1, 5),
    "thompson-mp8": (1, 3),
    "native-cm1-ptype5": (1, 5),
}

#: src/param.F:936-948: the ptypes CM1 integrates at eqtset=1 whatever the
#: deck says.  GM1 has one equation set, eqtset=2.
EQTSET1_PTYPES = frozenset({4, 7, 50, 51, 52, 53, 55})

#: Where ``translate`` parks the GM1-side level list so ``model_top_m`` can
#: read it.  Deliberately not a CM1 namelist name: nothing emits it.
DECLARED_LEVELS = "__gm1_declared_levels_m"

_SKIP = object()


class TranslationError(ValueError):
    """A CM1 configuration that cannot be translated, named by key."""


@dataclass(frozen=True)
class Translation:
    """What crossed, what did not, and what the reader must not trust."""

    gm1: Mapping[str, Mapping[str, Any]]
    namelist_overrides: Mapping[str, Any]
    carried: tuple[Mapping[str, Any], ...]
    unrepresentable: tuple[Mapping[str, Any], ...]
    #: Keys both models carry and the selected scheme reads in neither, so not
    #: emitting them changes nothing either side computes.  Recorded, never
    #: counted as unvalidated.
    component_inert: tuple[Mapping[str, Any], ...] = ()
    #: &param9 words of the projected selection the door does not take at the
    #: selected value, so the document leaves them at gpuwm-cm1's own value.
    #: Each row says what the model writes instead.
    output_not_admitted: tuple[Mapping[str, Any], ...] = ()
    #: Unmapped keys CM1 does not read under this configuration (their
    #: selector is off on both sides).  Recorded, never counted.
    inert_unmapped: tuple[Mapping[str, Any], ...] = ()
    #: What the translation changed about the CM1 deck itself (the harness
    #: replaces its ``ptype`` by the component's), so a reader is not
    #: surprised by it.
    notices: tuple[Mapping[str, Any], ...] = ()

    @property
    def unvalidated(self) -> bool:
        return bool(self.unrepresentable)

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema": "gm1.differential-translation/v1",
            "carried": list(self.carried),
            "unrepresentable_overrides": list(self.unrepresentable),
            "component_inert_selectors": list(self.component_inert),
            "inert_unmapped_selectors": list(self.inert_unmapped),
            "notices": list(self.notices),
            "unvalidated": self.unvalidated,
            "unvalidated_reason": (
                None
                if not self.unvalidated
                else (
                    "the CM1 deck carries "
                    f"{len(self.unrepresentable)} physical selector(s) whose value "
                    "gpuwm-cm1 does not integrate (no TOML key, a value GM1 does not "
                    "run, or a key outside the r21.1 inventory); CM1 will honour them "
                    "and gpuwm-cm1 will run whatever it transcribed instead, so the two "
                    "runs are not the same physical problem for those terms"
                )
            ),
        }


def _gm1_output_words() -> Any:
    from ..cm1_output_selection import SWITCH_BY_KEY

    return SWITCH_BY_KEY


def _gm1_output_owned_elsewhere() -> Any:
    from ..config import _OUTPUT_KEYS_OWNED_ELSEWHERE

    return _OUTPUT_KEYS_OWNED_ELSEWHERE


def gm1_output_keys() -> frozenset[str]:
    """The &param9 words gpuwm-cm1 admits in its own ``[output]`` table.

    Five of CM1's are spelled in the table that owns the thing they describe
    -- ``physics.output_dbz`` beside the component whose formula computes it,
    and the four getcape planes and the rotation/swath switches beside the
    controls that define them -- so they are emitted into their own tables
    rather than into ``[output]``, which would refuse them by name.
    """

    return frozenset(_gm1_output_words()) - _gm1_output_owned_elsewhere()


def gm1_output_admits(key: str, value: int) -> bool:
    """Whether the gpuwm-cm1 door would take this &param9 word at this value.

    Where it would not -- a product it has no operator for, or a prognostic
    carrier it will not drop -- the emitted document leaves the word out and
    keeps gpuwm-cm1's own value.
    """

    switch = _gm1_output_words()[key]
    if switch.disposition == "unsupported":
        return value == 0
    if switch.disposition == "state":
        return value == switch.gm1_default
    return value in switch.allowed


def _as_int_word(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _output_refusal(key: str, value: Any) -> Mapping[str, Any]:
    switch = _gm1_output_words()[key]
    if switch.disposition == "unsupported":
        instead = "gpuwm-cm1 has no operator for this field and does not write it"
    elif switch.disposition == "state":
        instead = (
            f"gpuwm-cm1 always writes this carrier (its value {switch.gm1_default}); "
            "the namelist's value is not taken"
        )
    else:
        instead = f"gpuwm-cm1 admits {list(switch.allowed)} for this word"
    return {"cm1_key": key, "value": value, "disposition": switch.disposition, "instead": instead}


def distributed_reference() -> Mapping[str, Any]:
    """CM1 r21.1's own distributed ``run/namelist.input``, flattened.

    It is the closest thing CM1 has to a default: the values a user who
    changed nothing would run.  Read from the generated record, so no CM1
    tree or build is needed.  The translator no longer judges unmapped keys
    against it (see ``gm1_behaviour`` in the key map); it stays for readers.
    """

    return _defaults.load().distributed_flat()


def model_top_m(effective: Mapping[str, Any]) -> float:
    """CM1's ``maxz``: the top w level, ``param.F:8193``.

    ``stretch_z=0`` makes that exactly ``nz*dz`` (``param.F:2608``).
    ``stretch_z=1`` generates its column analytically from the five
    ``param6`` words the namelist does carry, so the top is determined and is
    taken from this port's one generator rather than assumed equal to the
    declared ``ztop`` -- ``param.F:7932-7934`` reaches the top by accumulating
    ``dz_top`` over ``nk3`` layers, and ``param.F:7828-7877``'s feasibility
    check is what makes that land on ``ztop``.  ``stretch_z=3`` and ``4`` read
    their levels from a file (``param.F:8019-8026,8085-8089``), which a
    namelist alone does not determine, so those still refuse.
    """

    stretch_z = int(effective.get("stretch_z", 0))
    if stretch_z == 0:
        return float(effective["nz"]) * float(effective["dz"])
    if stretch_z == 1:
        from ..cm1_vertical_grid import VerticalGridError, build_vertical_levels

        try:
            levels = build_vertical_levels(
                stretch_z=1,
                nk=int(effective["nz"]),
                dz_m=float(effective["dz"]),
                ztop_m=float(effective["ztop"]),
                str_bot_m=float(effective["str_bot"]),
                str_top_m=float(effective["str_top"]),
                dz_bot_m=float(effective["dz_bot"]),
                dz_top_m=float(effective["dz_top"]),
            )
        except (KeyError, TypeError, ValueError, VerticalGridError) as error:
            raise TranslationError(
                "CM1 zd is a height and gpuwm-cm1 upper_damping_depth_m is a depth "
                "below the model top, so the translation must build this case's "
                f"stretch_z=1 column to place the damper, and it does not build: {error}"
            ) from error
        return float(levels.ztop_m)
    declared = effective.get(DECLARED_LEVELS)
    if declared:
        from ..cm1_vertical_grid import VerticalGridError, build_vertical_levels

        try:
            from ..config import GridConfig, _derived_top

            declared = tuple(float(value) for value in declared)
            # param.F:8037 and :8105 overwrite ztop with the generated top
            # face, so the top is derived exactly as config.py derives it and
            # is not taken from the namelist word this branch never reads.
            probe = GridConfig(
                nx=1, ny=1, nz=int(effective["nz"]),
                dx_m=1.0, dy_m=1.0, dz_m=float(effective["dz"]),
                terrain=False, stretch_x=False, stretch_y=False,
                stretch_z=stretch_z, ztop_m=0.0, str_bot_m=0.0, str_top_m=0.0,
                dz_bot_m=0.0, dz_top_m=0.0, levels_m=declared,
            )
            levels = build_vertical_levels(
                stretch_z=stretch_z,
                nk=int(effective["nz"]),
                dz_m=float(effective["dz"]),
                ztop_m=_derived_top(probe),
                levels_m=declared,
            )
        except (KeyError, TypeError, ValueError, VerticalGridError) as error:
            raise TranslationError(
                "CM1 zd is a height and gpuwm-cm1 upper_damping_depth_m is a depth "
                "below the model top, so the translation must build this case's "
                f"stretch_z={stretch_z} column to place the damper, and it does not "
                f"build: {error}"
            ) from error
        return float(levels.ztop_m)
    raise TranslationError(
        "CM1 zd is a height and gpuwm-cm1 upper_damping_depth_m is a depth below the "
        f"model top; stretch_z={stretch_z} reads its levels from a file "
        "(param.F:8013,8083), so declare them as [gm1] \"grid.levels_m\" and both "
        "sides will run -- and place the damper on -- the same column."
    )


def _transform(kind: str, value: Any, component: str, effective: Mapping[str, Any]) -> Any:
    if kind in ("identity", "seconds", "metres"):
        return value
    if kind == "int-to-bool":
        if value not in (0, 1):
            raise TranslationError(f"a boolean gpuwm-cm1 key takes CM1 0 or 1 only; got {value!r}")
        return bool(value)
    if kind == "logical-to-bool":
        return bool(value)
    if kind == "abs-seconds":
        return abs(float(value))
    if kind == "cm1-zd-to-depth":
        return model_top_m(effective) - float(value)
    if kind == "derived-component":
        return component
    if kind == "none":
        return _SKIP
    raise TranslationError(f"unknown key-map transform {kind!r}")


def _same_word(value: Any, expected: Any) -> bool:
    """A namelist value against a recorded one; CM1 logicals compare as 0/1."""

    if isinstance(value, str) or isinstance(expected, str):
        return value == expected
    return bool(value == expected)


def _selector_holds(alternative: Mapping[str, Any], selectors: Mapping[str, Any]) -> bool:
    """Whether every selector of one ``read_when`` alternative has a listed value.

    A selector the deck does not carry is read as holding: the translation
    cannot show the key is inert, so it names it.
    """

    for key, values in alternative.items():
        if key not in selectors:
            continue
        if not any(_same_word(selectors[key], value) for value in values):
            return False
    return True


def translate(
    effective: Mapping[str, Any],
    *,
    component: str,
    gm1_only: Mapping[str, Any],
    output: Mapping[str, Any],
    declared: Collection[str] = (),
    groups: Mapping[str, str] | None = None,
) -> Translation:
    """Translate one flat CM1 configuration into a gpuwm-cm1 document.

    ``effective`` is every CM1 key with the value CM1 would run.
    ``component`` is the gpuwm-cm1 physics component standing for
    ``imoist``/``ptype``.  ``gm1_only`` holds the dotted keys CM1 has no word
    for.  ``output`` is the ``&param9`` selection to project onto the
    document (the harness's own, or the namelist's).  ``declared`` names the
    keys a caller set explicitly; they are reported as ``carried``.
    ``groups`` names each key's CM1 namelist group (the harness passes its
    base deck's); a key it does not name takes its group from CM1 r21.1's
    inventory, and a key in neither has none.
    """

    effective = dict(effective)
    # param.F:8013,8083 read the level list from a file, so it is declared
    # on the GM1 side and is not a namelist word.  The zd transform still
    # needs the model top it implies (param.F:8193), so it travels with the
    # effective deck under a name no CM1 group has.
    levels = gm1_only.get("grid.levels_m")
    if levels:
        effective[DECLARED_LEVELS] = tuple(float(value) for value in levels)
    declared = frozenset(declared)

    table = _keymap.load()
    gm1: dict[str, dict[str, Any]] = {name: {} for name in TABLE_ORDER}
    carried: list[dict[str, Any]] = []
    unrepresentable: list[dict[str, Any]] = []
    component_inert: list[dict[str, Any]] = []
    inert_unmapped: list[dict[str, Any]] = []

    if component not in COMPONENT_TO_CM1:
        raise TranslationError(f"unknown physics.component {component!r}; known: {sorted(COMPONENT_TO_CM1)}")
    # Both sides run the component's scheme, so the selectors an unmapped key
    # hangs on are read with the component's own imoist/ptype.
    imoist, ptype = COMPONENT_TO_CM1[component]
    selectors = dict(effective)
    selectors.update({"imoist": imoist, "ptype": ptype})
    inventory = _defaults.load()

    def group_of(key: str) -> str | None:
        if groups is not None and key in groups:
            return groups[key]
        found = inventory.groups_of(key)
        return found[0] if found else None

    non_physical = {
        row["cm1_group"] for row in table.non_physical_families
    } | set(table.harness_owned_groups)

    for cm1_key, value in sorted((k, v) for k, v in effective.items() if k != DECLARED_LEVELS):
        entry = table.by_cm1(cm1_key)
        if entry is None:
            group = group_of(cm1_key)
            if group in non_physical:
                continue  # output, statistics, parcel-output and restart selectors
            # A key the r21.1 inventory does not hold -- a newer CM1's, or a
            # misspelling -- used to vanish here without a word.
            unrepresentable.append({
                "cm1_key": cm1_key,
                "cm1_group": group,
                "effective": value,
                "declared_by_case": cm1_key in declared,
                "note": (
                    "not in CM1 r21.1's physical-key inventory "
                    "(records/CM1-GM1-KEYMAP.json); neither model's r21.1 "
                    "grammar reads it here, so what it selects is not run"
                ),
            })
            continue
        scope = COMPONENT_SCOPED_KEYS.get(entry.gm1_key or "")
        if scope is not None and component not in scope:
            # The key exists on both sides and the selected scheme reads it in
            # NEITHER: src/param.F:3407 hands ndcnst to ptype=5 only, and ihail
            # reaches ptype=2, 5 and 7 only.  Not emitting it therefore
            # changes nothing either model computes, so this is recorded
            # rather than counted against the comparison.  It is not optional
            # either way: gpuwm-cm1 refuses a declared value for a component
            # that does not read it, so emitting the key would make a valid
            # configuration unrunnable.
            component_inert.append({
                "cm1_key": cm1_key,
                "cm1_group": entry.cm1_group,
                "effective": value,
                "declared_by_case": cm1_key in declared,
                "gm1_key": entry.gm1_key,
                "note": (
                    f"{entry.gm1_key} is read by "
                    f"{', '.join(sorted(scope))} only; CM1 does not read "
                    f"its own {cm1_key} for this ptype either, so the term "
                    "is inert on both sides"
                ),
            })
            continue
        if entry.gm1_key is not None and entry.gm1_key.startswith("output."):
            # [output] is projected from ``output`` below, so both callers get
            # the same words through the same admission test.
            continue
        if entry.gm1_key is None:
            if not entry.affects_state:
                continue
            # Judged against what GM1 integrates, not against CM1's
            # distributed deck: every shipped deck sets ramp_sgs=1, which GM1
            # did not run, and this used to report nothing for it.
            behaviour = dict(entry.gm1_behaviour or {})
            row = {
                "cm1_key": cm1_key,
                "cm1_group": entry.cm1_group,
                "effective": value,
                "declared_by_case": cm1_key in declared,
                "note": entry.note,
            }
            # read_when first: a key CM1 does not read under these selectors
            # is inert on both sides whatever GM1 would run in its place.
            reading = None
            if "read_when" in behaviour:
                reading = [
                    alternative for alternative in behaviour["read_when"]
                    if _selector_holds(alternative, selectors)
                ]
                if not reading:
                    inert_unmapped.append({
                        "cm1_key": cm1_key,
                        "effective": value,
                        "read_when": behaviour["read_when"],
                        "anchor": behaviour.get("anchor"),
                    })
                    continue
                row["cm1_reads_it_because"] = reading
            if "gm1_runs" in behaviour:
                if _same_word(value, behaviour["gm1_runs"]):
                    continue
                row["gm1_runs"] = behaviour["gm1_runs"]
            elif reading is None:
                row["gm1_behaviour"] = "unrecorded"
            row["anchor"] = behaviour.get("anchor")
            unrepresentable.append(row)
            continue
        section, name = entry.gm1_key.split(".", 1)
        try:
            translated = _transform(entry.transform, value, component, effective)
        except TranslationError as error:
            raise TranslationError(f"CM1 {cm1_key} = {value!r} ({entry.gm1_key}): {error}") from error
        if translated is _SKIP:
            continue
        gm1[section][name] = translated
        if cm1_key in declared:
            carried.append({
                "cm1_key": cm1_key,
                "gm1_key": entry.gm1_key,
                "transform": entry.transform,
                "value": translated,
            })

    # &param9 on the gpuwm-cm1 side: the words the door admits at their
    # value; the rest keep gpuwm-cm1's own value and are recorded.
    output_keys = gm1_output_keys()
    not_admitted: list[Mapping[str, Any]] = []
    gm1["output"] = {}
    for key, value in output.items():
        if key not in output_keys:
            continue
        if gm1_output_admits(key, value):
            gm1["output"][key] = value
        else:
            not_admitted.append(_output_refusal(key, value))
    if not gm1["output"]:
        del gm1["output"]

    # GM1-only keys, and the alias config.py:1228 requires.
    for dotted, value in sorted(gm1_only.items()):
        section, name = dotted.split(".", 1)
        gm1.setdefault(section, {})[name] = value
    for key, section in GM1_OUTPUT_OWNERS.items():
        if key not in output:
            continue
        value = output[key]
        if not gm1_output_admits(key, value):
            not_admitted.append(_output_refusal(key, value))
            continue
        if section == "physics":
            # config.py's CM1-CFG-DBZ-COMPONENT: GM1 publishes only the
            # reflectivity of a component whose own formula it carries.
            # Another component (Morrison) is emitted with output_dbz false --
            # the key defaults on, so leaving it out would be refused.
            from ..gm1_runtime import REFLECTIVITY_COMPONENTS

            admitted = gm1[section].get("component") in REFLECTIVITY_COMPONENTS
            gm1[section][key] = bool(value) and admitted
            if bool(value) and not admitted:
                not_admitted.append({
                    "cm1_key": key, "value": value, "disposition": "component",
                    "instead": (
                        f"{gm1[section].get('component')} carries no reflectivity operator "
                        "inside the model, so the document writes output_dbz = false"
                    ),
                })
        elif key in _BOOLEAN_DIAGNOSTICS:
            gm1[section][key] = bool(value)
        else:
            gm1[section][key] = int(value)
    # config.py's optional key groups are all-or-none, so a deck that names
    # one member has to emit the whole group; the missing members take CM1's
    # own shipped values, which is what the absent group means.
    for group, defaults in GROUP_DEFAULTS.items():
        section = group.split("_", 1)[0]
        present = gm1.get(section, {})
        if any(name in present for name in defaults):
            for name, fallback in defaults.items():
                present.setdefault(name, fallback)
            gm1[section] = present
    # src/param.F:873-889 rewrites bbc for a surface layer; gpuwm-cm1 refuses
    # the disagreement instead, so the emitted deck states the boundary CM1
    # would have integrated rather than the one it declared.
    if gm1.get("closures", {}).get("sfcmodel", 0) >= 1:
        gm1["closures"]["bbc"] = 3
    # A [surface] table is only read for a surface layer or a semi-slip bottom
    # (src/init_surface.F:40); emitting one without either is refused by the
    # door and would read as a configuration error.
    if gm1.get("surface") and not (
        gm1.get("closures", {}).get("sfcmodel", 0) >= 1
        or gm1.get("closures", {}).get("bbc", 1) == 3
    ):
        gm1.pop("surface")
    # base.F:1204 (isnd = 11) and :1849 (isnd = 21) build the surface air from
    # the namelist tsk0 whether or not a surface layer runs.  Without a
    # [surface] table the word travels in [initialization].
    if (
        _as_int_word(effective.get("isnd")) in (11, 21)
        and "surface" not in gm1
        and effective.get("tsk0") is not None
        and "initialization" in gm1
    ):
        gm1["initialization"]["tsk0_k"] = float(effective["tsk0"])
    gm1["boundaries"]["bottom"] = gm1["closures"]["bbc"]

    # misclibs.F:818-915 with README.namelist:453-455: since cm1r19 CM1 never
    # reads nsound from a namelist.  It derives it from dt and the grid before
    # the first step (cm1.F:3054) and again on every change of dt.  gpuwm-cm1
    # declares the count instead, so it is derived here, once.
    from ..cm1_adaptive_timestep import cm1_dtsmall  # noqa: PLC0415

    try:
        _, nsound = cm1_dtsmall(
            dt_s=float(effective["dtl"]),
            nx=int(effective["nx"]),
            ny=int(effective["ny"]),
            dx_m=float(effective["dx"]),
            dy_m=float(effective["dy"]),
            dz_m=float(effective["dz"]),
            psolver=int(effective.get("psolver", 3)),
            csound_m_s=float(effective.get("csound", 300.0)),
            adapt_dt=bool(int(effective.get("adapt_dt", 0))),
        )
    except (TypeError, ValueError) as error:
        spelled = ", ".join(
            f"{key} = {effective.get(key)!r}"
            for key in ("dtl", "nx", "ny", "dx", "dy", "dz", "psolver", "csound", "adapt_dt")
            if key in effective
        )
        raise TranslationError(
            "CM1 derives the acoustic step count from the large step and the grid "
            f"(misclibs.F:818-915), and cannot from {spelled}: {error}"
        ) from error
    stated = gm1.get("runtime", {}).get("acoustic_substeps")
    if stated is not None and int(stated) != nsound:
        raise TranslationError(
            f"[gm1] runtime.acoustic_substeps = {stated}, but CM1's own "
            f"dtsmall derives {nsound} for dtl={effective['dtl']} on this "
            "grid and never accepts the count from a namelist "
            "(README.namelist:453-455).  Declaring a different one makes "
            "the two codes take a different number of acoustic small "
            "steps per large step, so the divergence curve measures the "
            "schedule rather than the term.  Remove the key."
        )
    gm1.setdefault("runtime", {})["acoustic_substeps"] = nsound

    # CM1 ignores the param6 stretch words on stretch_z=0 (param.F:8037
    # overwrites ztop outright); gpuwm-cm1 refuses a declared value the branch
    # does not read, because a receipt naming a model top the run does not
    # have is worse than no value at all.  Honour the branch.  README.namelist
    # :832-834 -- CM1 reads nothing else in param11 for radopt=0 -- and
    # config.py refuses a declared place, date or cadence with radiation off.
    if int(effective.get("radopt", 0)) == 0:
        gm1["radiation"].clear()
    if int(effective.get("stretch_z", 0)) == 0:
        for name in ("stretch_z", "ztop_m", "str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m", "levels_m"):
            gm1["grid"].pop(name, None)
    else:
        if int(effective.get("stretch_z", 0)) in (3, 4):
            # param.F:8005-8132 reads the level list and nothing else: it overwrites ztop
            # with the generated top (:8037, :8105) and never reads str_bot, str_top, dz_bot
            # or dz_top.  Honour the branch, as for stretch_z=0 above.
            for name in ("ztop_m", "str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m"):
                if name in gm1["grid"]:
                    gm1["grid"][name] = 0.0
        # calcvort and calcuh bind CM1's mh/mf vertical metric to one
        # (param.F:8170), so gpuwm-cm1 refuses these five on a non-uniform
        # column.  Set to 0 rather than removed: the group is all-or-none.
        table_ = gm1.get("diagnostics")
        if table_:
            for name in sorted(UNIFORM_METRIC_ONLY):
                if name in table_:
                    if table_[name] and name in output:
                        not_admitted.append({
                            "cm1_key": name, "value": output[name], "disposition": "stretched-column",
                            "instead": "gpuwm-cm1 computes this product on a uniform column only",
                        })
                    table_[name] = 0
    # The rain and graupel swaths read CM1's qr and qg rows; param.F:4727-4728
    # turns each off where the scheme has no such row, and so does this, set to
    # 0 for the same all-or-none reason.  Every component GM1 binds carries the
    # six mass rows (qv, qc, qr, qi, qs, qg), so this changes nothing for them.
    component = gm1.get("physics", {}).get("component")
    if component:
        from ..moisture_species import moisture_species_profile

        try:
            mass = moisture_species_profile(component).mass_names
        except Exception:  # an unknown component is refused by the door, by name
            mass = ()
        table_ = gm1.get("diagnostics")
        if table_ and (mass[2:3] != ("qr",) or mass[5:6] != ("qg",)):
            for name in sorted(RAIN_GRAUPEL_SWATHS):
                if name in table_:
                    if table_[name] and name in output:
                        not_admitted.append({
                            "cm1_key": name, "value": output[name], "disposition": "component",
                            "instead": "this scheme has no rainwater or graupel row (CM1 param.F:4727-4728 "
                                       "turns the swath off the same way)",
                        })
                    table_[name] = 0

    # [lagrangian] carries CM1's parcel, &param13 and pdcomp selectors.  It is
    # emitted only when the run integrates parcels or decomposes the pressure:
    # a document that omits it keeps the exact sealed sha256 it had before the
    # table existed.
    selectors = gm1.get("lagrangian", {})
    if int(selectors.get("iprcl", 0)) != 1 and int(selectors.get("output_pdcomp", 0)) != 1:
        gm1["lagrangian"] = {}

    # dynamics.lspgrad is optional and absent binds CM1's 0 unprojected
    # (config.py), so an lspgrad=0 deck -- every one CM1 ships -- is emitted
    # without it and keeps the sha256 its receipts already name.
    if gm1.get("dynamics", {}).get("lspgrad") == 0:
        del gm1["dynamics"]["lspgrad"]

    # src/param.F:936-948: CM1 forces eqtset=1 for ptype 4, 7, 50-53 and 55
    # (WSM6 among them), so CM1 integrates the constant-pressure equation set
    # for such a scheme whatever the deck says.  GM1 has one equation set,
    # eqtset=2, and heats WSM6's phase changes at constant volume
    # (cuda_cartesian_moist_heating) -- not a CM1 configuration.
    if ptype in EQTSET1_PTYPES and int(effective.get("eqtset", 2)) == 2:
        unrepresentable.append({
            "cm1_key": "eqtset",
            "cm1_group": group_of("eqtset"),
            "effective": effective.get("eqtset"),
            "declared_by_case": "eqtset" in declared,
            "cm1_runs": 1,
            "note": (
                f"src/param.F:936-948 sets eqtset=1 for ptype={ptype}; gpuwm-cm1 "
                "integrates eqtset=2 with the scheme's phase changes heated at "
                "constant volume, so the equation sets differ"
            ),
        })

    gm1 = {k: v for k, v in gm1.items() if v}
    return Translation(
        gm1=gm1,
        namelist_overrides={},
        carried=tuple(carried),
        unrepresentable=tuple(unrepresentable),
        component_inert=tuple(component_inert),
        output_not_admitted=tuple(not_admitted),
        inert_unmapped=tuple(inert_unmapped),
    )


def toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def render_toml(translation: Translation, header: tuple[str, ...]) -> str:
    """The gpuwm-cm1 document text: comment header, then tables in read order."""

    lines = list(header)
    if translation.unvalidated:
        lines.append("# UNVALIDATED: " + (translation.receipt()["unvalidated_reason"] or ""))
        for row in translation.unrepresentable:
            lines.append(f"#   CM1 {row['cm1_key']} = {row['effective']!r}: {row['note']}")
    for section in TABLE_ORDER:
        if section not in translation.gm1:
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key in sorted(translation.gm1[section]):
            lines.append(f"{key} = {toml_scalar(translation.gm1[section][key])}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The importer: a user's namelist, as CM1 r21.1 would run it
# ---------------------------------------------------------------------------

#: CM1 r21.1's ptype table (README.namelist:458-495).
PTYPE_SCHEMES: Mapping[int, str] = {
    0: "no microphysics (vapor only)",
    1: "Kessler",
    2: "NASA-Goddard LFO",
    3: "Thompson",
    4: "Gilmore/Straka/Rasmussen LFO",
    5: "Morrison double-moment",
    6: "Rotunno-Emanuel (1987) water-only",
    7: "WSM6",
    26: "NSSL 2-moment (graupel only)",
    27: "NSSL 2-moment (graupel and hail)",
    28: "NSSL single-moment",
    50: "P3 1-ice category, 1-moment cloud water",
    51: "P3 1-ice category, 2-moment cloud water",
    52: "P3 2-ice categories, 2-moment cloud water",
    53: "P3 1-ice category, 3-moment ice, 2-moment cloud water",
    55: "Jensen ISHMAEL",
}

#: The ptypes gpuwm-cm1 runs, and how.  Thompson has run only a 120-s
#: column, so it runs only after the user acknowledges that.
PTYPE_COMPONENTS: Mapping[int, tuple[str, str]] = {
    5: ("morrison-mp10", "runs"),
    7: ("wsm6-mp6", "runs"),
    3: ("thompson-mp8", "acknowledge"),
}

#: The runtime keys an imported deck starts from.  CM1 has no word for them;
#: the estimator replaces the memory budget with what the card has free.
PRODUCT_GM1_ONLY: Mapping[str, Any] = {
    "run.lane": "cartesian",
    "runtime.capacity_profile": "default",
    "runtime.memory_budget_mib": 8192,
    "runtime.warmup_steps": 0,
}


@dataclass(frozen=True)
class ComponentChoice:
    """``ptype`` (and ``imoist``) as a gpuwm-cm1 physics component.

    ``status`` is ``runs``, ``acknowledge`` (runs once the user accepts the
    stated limit) or ``refused`` (``component`` is None).
    """

    imoist: int
    ptype: int
    component: str | None
    status: str
    scheme: str
    reason: str


def component_for(imoist: int, ptype: int) -> ComponentChoice:
    if int(imoist) == 0:
        return ComponentChoice(
            imoist=0, ptype=int(ptype), component="off", status="runs",
            scheme="no moisture (imoist = 0)",
            reason="imoist = 0 is gpuwm-cm1's component 'off'; CM1 ignores ptype without moisture",
        )
    scheme = PTYPE_SCHEMES.get(int(ptype))
    if int(ptype) in PTYPE_COMPONENTS:
        component, status = PTYPE_COMPONENTS[int(ptype)]
        reason = {
            "morrison-mp10": "ArWen's WRF-lineage Morrison (mp_physics = 10) with CM1's ihail and ndcnst, not CM1's morrison.F",
            "wsm6-mp6": "ArWen's WSM6 (mp_physics = 6)",
            "thompson-mp8": "ArWen's Thompson (mp_physics = 8); it has run only a 120-s column, so a run needs your acknowledgment",
        }[component]
        return ComponentChoice(imoist=1, ptype=int(ptype), component=component, status=status,
                               scheme=scheme or "", reason=reason)
    if scheme is None:
        return ComponentChoice(
            imoist=1, ptype=int(ptype), component=None, status="refused", scheme="",
            reason=f"ptype = {ptype} is not a CM1 r21.1 microphysics scheme (README.namelist:458-495)",
        )
    return ComponentChoice(
        imoist=1, ptype=int(ptype), component=None, status="refused", scheme=scheme,
        reason=(
            f"ptype = {ptype} is {scheme}, which gpuwm-cm1 does not have; "
            "use ptype = 5 (Morrison) or 7 (WSM6)"
        ),
    )


def cm1_startup_stops(group_order: Sequence[str]) -> tuple[str, ...]:
    """Why CM1 r21.1 itself stops reading this file before param.F runs.

    cm1.F:480-482 reads ``&param0``, ``&param2`` and ``&param8`` one after
    another with no ``end=`` branch and no rewind between them.  A file
    without one of them, or with them in another order, ends that read at the
    end of the file and CM1 stops.  Every other group is read after a rewind
    and may be absent.
    """

    required = _defaults.load().required_groups
    order = list(group_order)
    stops = [
        f"CM1 r21.1 stops at start-up: cm1.F:480-482 reads &{group} with no end= "
        f"branch, and this file has no &{group}"
        for group in required
        if group not in order
    ]
    present = [group for group in required if group in order]
    if [order.index(group) for group in present] != sorted(order.index(group) for group in present):
        stops.append(
            "CM1 r21.1 stops at start-up: cm1.F:480-482 reads "
            + ", ".join(f"&{group}" for group in required)
            + " in that order without rewinding, and this file opens them as "
            + ", ".join(f"&{group}" for group in sorted(present, key=order.index))
        )
    return tuple(stops)


@dataclass(frozen=True)
class NamelistImport:
    """One namelist, read as CM1 r21.1 would run it, and its translation.

    ``effective`` holds every r21.1 key: the file's value, an older key's
    reading (``aliases``), an r22.0 key's r21.1 meaning (``r22``), or CM1
    r21.1's code default (``defaulted``).  ``unknown`` holds the assignments
    no record explains.  ``cm1_stops`` says why CM1 r21.1 itself would stop
    reading the file (a missing or misordered start-up group).
    ``translation`` is None when the physics component is refused or CM1
    would not start.
    """

    parsed: Any
    version: str
    effective: Mapping[str, Any]
    stated: frozenset[str]
    defaulted: tuple[Any, ...]
    aliases: tuple[Any, ...]
    r22: tuple[Any, ...]
    r22_settings: Mapping[str, str]
    unknown: tuple[Any, ...]
    component: ComponentChoice
    output_writer: Mapping[str, Any]
    translation: Translation | None
    cm1_stops: tuple[str, ...] = ()

    def toml(self) -> str | None:
        if self.translation is None:
            return None
        name = self.parsed.path.rsplit("/", 1)[-1] if self.parsed.path else "namelist"
        return render_toml(self.translation, (
            f"# gpuwm-cm1 configuration imported from CM1 namelist {name}",
            f"# (sha256 {self.parsed.sha256}) by cm1core.cm1import through records/CM1-GM1-KEYMAP.json.",
            "# Do not hand-edit: edit the namelist and import it again.",
        ))


def translate_parsed(
    parsed: Any,
    *,
    gm1_only: Mapping[str, Any] | None = None,
    translate_now: bool = True,
) -> NamelistImport:
    """Read a parsed namelist as CM1 r21.1 would run it, and translate it.

    ``translate_now=False`` stops after reading: ``translation`` is None, and
    the caller translates (the coverage report does, setting refused values
    aside so one pass reaches every setting).
    """

    from . import aliases as _aliases
    from . import versions as _versions

    stated_values = parsed.values()
    stated_flat = parsed.flat()
    version = _versions.detect(parsed.unknown)
    alias_readings, rest = _aliases.resolve(parsed.unknown, stated_flat)
    added = _aliases.assignments(alias_readings, stated_flat)

    r22_settings: dict[str, Any] = {}
    r22_reasons: dict[str, str] = {}
    r22_output_vort = None
    if version == _versions.R22:
        r22_settings, r22_reasons = _versions.r21_settings(rest, stated_flat)
        r22_output_vort = stated_flat.get("output_vort")
    view: dict[str, dict[str, Any]] = {group: dict(rows) for group, rows in stated_values.items()}
    defaults = _defaults.load()
    for key, value in {**added, **r22_settings}.items():
        group = defaults.groups_of(key)[0]
        view.setdefault(group, {})[key] = value
    effective, filled = defaults.fill(view)
    r22_readings, rest = _versions.classify(rest, effective, r22_output_vort=r22_output_vort)
    if version == _versions.R22:
        meaning = _versions.load().changed_meaning["irandp"]
        if effective.get("irandp") not in meaning["r21_values"]:
            r22_readings = r22_readings + (_versions.R22Reading(
                group="param2", key="irandp", text=str(effective.get("irandp")), value=effective.get("irandp"),
                klass="physics", status="differs", read_by_r22=True,
                message=f"irandp = {effective.get('irandp')} in CM1 r22.0: {meaning['r22_meaning']}; CM1 r21.1: {meaning['r21_meaning']}",
            ),)

    choice = component_for(int(effective.get("imoist", 0)), int(effective.get("ptype", 0)))
    stated = frozenset(stated_flat) | frozenset(added) | frozenset(r22_settings)
    param9 = defaults.fields["param9"]
    selection = {key: effective[key] for key in param9}
    writer = {key: effective[key] for key in ("output_format", "output_filetype")}
    stops = cm1_startup_stops(parsed.group_order)
    translation = None
    if choice.component is not None and not stops and translate_now:
        runtime = dict(PRODUCT_GM1_ONLY)
        runtime.update(gm1_only or {})
        translation = translate(
            effective,
            component=choice.component,
            gm1_only=runtime,
            output=selection,
            declared=stated,
        )
    return NamelistImport(
        parsed=parsed,
        version=version,
        effective=effective,
        stated=stated,
        defaulted=filled,
        aliases=alias_readings,
        r22=r22_readings,
        r22_settings=r22_reasons,
        unknown=rest,
        component=choice,
        output_writer=writer,
        translation=translation,
        cm1_stops=stops,
    )


def import_namelist(
    path: Any,
    *,
    tool: Any = None,
    gm1_only: Mapping[str, Any] | None = None,
    translate_now: bool = True,
) -> NamelistImport:
    """Parse ``path`` with the native parser and translate it (see :func:`translate_parsed`)."""

    from . import namelist as _namelist

    return translate_parsed(_namelist.inspect(path, tool=tool), gm1_only=gm1_only, translate_now=translate_now)
