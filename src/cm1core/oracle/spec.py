"""One physical configuration, expressed once.

A *case* is a TOML document with two tables:

``[case]``
    identity: ``name``, the ``term`` under test (which the horizon calculator
    reads), and free-text ``notes``.

``[cm1]``
    the physical configuration, in CM1 r21.1's own namelist words.  CM1 is the
    mathematical authority, so its vocabulary is the one the case speaks; the
    gpuwm-cm1 TOML is *derived* from it through the checked key map.  Any key
    the authority inventory does not contain refuses by name.

``[gm1]``
    the dotted gpuwm-cm1 keys CM1 has no word for -- the lane, the CUDA tile
    shape, the memory budget, the named physics component.  Only keys listed
    as ``gm1_only`` in the key map are admitted here; anything else would be a
    second, unchecked spelling of a CM1 key.

Emission is deliberately total: :meth:`DifferentialCase.gm1_toml` always
produces a complete gpuwm-cm1 document, filling anything the case does not
state from the CM1 base deck, so the two sides can never differ because one of
them fell back to a default the other did not.

The translation receipt names every ``[cm1]`` key the port cannot carry.  That
list is the *concrete* gap for this case: CM1 will honour the value and GM1
will silently run its own, so a comparison that ignores it is measuring two
different physical problems.  "Cannot carry" is judged against what GM1
actually integrates (``records/CM1-GM1-KEYMAP.json``'s ``gm1_behaviour``), not
against CM1's distributed deck: a deck that sets a key to CM1's own shipped
value is still a difference when GM1 runs something else, and a key the r21.1
inventory does not contain is named rather than skipped.
"""

from __future__ import annotations

import tomllib
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ..cm1import import translate as _translate
from . import keymap as _keymap
from . import namelist as _namelist

CASE_SCHEMA = "gm1.differential-case/v1"

#: CM1 output selectors the harness owns outright: they make CM1 write exactly
#: the fields GM1's history publisher emits, in the raw GrADS form this
#: harness reads, and nothing else.  ``output_format=1`` is CM1's own
#: direct-access binary32; it carries no container and no library.
HARNESS_OUTPUT: Mapping[str, Any] = {
    "output_format": 1,
    "output_filetype": 2,
    "output_interp": 0,
    "output_th": 1,
    "output_thpert": 1,
    "output_prs": 1,
    "output_rho": 1,
    "output_tke": 1,
    "output_qv": 1,
    "output_q": 1,
    "output_u": 1,
    "output_v": 1,
    "output_w": 1,
    "output_uinterp": 0,
    "output_vinterp": 0,
    "output_winterp": 0,
    # CM1 &param9 products this port now publishes under CM1's own names, so
    # the differential compares the diagnostic operators as well as the state.
    "output_vort": 1,
    "output_pv": 1,
    "output_uh": 1,
    # On, so the run exercises the reflectivity path end to end.  ``dbz`` and
    # ``cref`` are deliberately NOT in compare.FIELD_MAP: CM1 forms
    # reflectivity with its own src/mp_wsm6.F operator and gpuwm-cm1 with
    # ArWen's audited WSM6 10 cm operator, so pairing them would report the
    # difference between two transcriptions as a model difference.  The
    # cm1_output_selection UNVALIDATED marker says exactly that.
    "output_dbz": 1,
    "output_rain": 1,
    "output_sws": 1,
    "output_svs": 1,
    "output_sps": 1,
    "output_srs": 1,
    "output_sgs": 1,
    "output_sus": 1,
    "output_shs": 1,
    "output_cape": 1,
    "output_cin": 1,
    "output_lcl": 1,
    "output_lfc": 1,
    "output_pwat": 1,
    "output_lwp": 1,
    "output_km": 0,
    "output_kh": 0,
    "output_basestate": 0,
    "output_zh": 0,
    "output_zs": 0,
}

#: ``physics.component`` <-> CM1 ``(imoist, ptype)``: the translator's table
#: (README.namelist:458-495, src/param.F:3141-4509).  WSM6 is ``ptype=7``; it
#: was once mapped to 5, which is Morrison, so every CM1 half of a WSM6
#: differential ran Morrison.  ``wdm6-mp16`` has no entry: CM1 r21.1 has no
#: WDM6 scheme, so a case that names it is refused by key.
COMPONENT_TO_CM1: Mapping[str, tuple[int, int]] = _translate.COMPONENT_TO_CM1

#: A case document that cannot be translated, named by key.  The harness and
#: the importer share one translator, so they share its error too.
CaseError = _translate.TranslationError
Translation = _translate.Translation

#: The translator's own names, kept here for the harness's tests and tools.
_TABLE_ORDER = _translate.TABLE_ORDER
_GROUP_DEFAULTS = _translate.GROUP_DEFAULTS
_COMPONENT_SCOPED_KEYS = _translate.COMPONENT_SCOPED_KEYS
_GM1_OUTPUT_OWNERS = _translate.GM1_OUTPUT_OWNERS
_UNIFORM_METRIC_ONLY = _translate.UNIFORM_METRIC_ONLY
_DECLARED_LEVELS = _translate.DECLARED_LEVELS
_gm1_output_admits = _translate.gm1_output_admits
_model_top_m = _translate.model_top_m
_toml_scalar = _translate.toml_scalar
_reference_deck = _translate.distributed_reference


@dataclass(frozen=True)
class DifferentialCase:
    name: str
    term: str
    notes: str
    cm1: Mapping[str, Any]
    gm1_only: Mapping[str, Any]
    source: Path | None = None

    # -- CM1 side ---------------------------------------------------------
    def namelist_deck(self, base: Path) -> "OrderedDict[str, MutableMapping[str, Any]]":
        """CM1's shipped deck at ``base``, with this case applied."""

        deck = _namelist.read(base)
        unknown = _namelist.apply(deck, dict(self.cm1))
        if unknown:
            raise CaseError(
                f"case [cm1] names keys the base deck {base} has no group for: "
                f"{sorted(unknown)}; CM1 reads only the groups its deck declares"
            )
        unknown = _namelist.apply(deck, HARNESS_OUTPUT)
        if unknown:
            raise CaseError(f"base deck {base} is missing harness output selectors: {sorted(unknown)}")
        return deck

    # -- GM1 side ---------------------------------------------------------
    def translate(self, base: Path) -> Translation:
        """The case on the gpuwm-cm1 side, through the one shared translator.

        The base deck plus the case's ``[cm1]`` overrides is the effective CM1
        configuration; the harness forces ``HARNESS_OUTPUT`` as the &param9
        selection so both models write the same fields.
        """

        deck = _namelist.read(base)
        effective: dict[str, Any] = {}
        for fields in deck.values():
            effective.update(fields)
        effective.update(self.cm1)
        component = self.gm1_only.get("physics.component")
        if component is None:
            raise CaseError("[gm1] must declare physics.component; CM1's ptype is an in-tree scheme number and GM1 names the component behind its ArWen seam")
        if component not in COMPONENT_TO_CM1:
            raise CaseError(f"unknown physics.component {component!r}; known: {sorted(COMPONENT_TO_CM1)}")
        # Both sides run the component's scheme: the CM1 namelist's
        # imoist/ptype are overridden from it (_component_overrides), and the
        # receipt says so when the base deck named another scheme.
        imoist, ptype = COMPONENT_TO_CM1[component]
        notices: list[dict[str, Any]] = []
        if (int(effective.get("imoist", imoist)), int(effective.get("ptype", ptype))) != (imoist, ptype):
            notices.append({
                "cm1_keys": ["imoist", "ptype"],
                "deck": [effective.get("imoist"), effective.get("ptype")],
                "run": [imoist, ptype],
                "note": (
                    f"physics.component = {component!r} selects CM1 ptype={ptype}; the "
                    "deck's own scheme is replaced on the CM1 side too, so both codes "
                    "run the component's scheme family"
                ),
            })
        translation = _translate.translate(
            effective,
            component=component,
            gm1_only=self.gm1_only,
            output=HARNESS_OUTPUT,
            declared=frozenset(self.cm1),
            groups={key: group for group, fields in deck.items() for key in fields},
        )
        overrides = dict(self.cm1)
        overrides.update(_component_overrides(component))
        return replace(translation, namelist_overrides=overrides, notices=tuple(notices))

    def gm1_toml(self, base: Path) -> str:
        return _translate.render_toml(self.translate(base), (
            f"# gpuwm-cm1 configuration derived from differential case {self.name!r}",
            f"# by cm1core.oracle.spec through records/CM1-GM1-KEYMAP.json.",
            "# Do not hand-edit: edit the case and re-emit, or the two sides drift.",
        ))


def _component_overrides(component: str) -> Mapping[str, int]:
    imoist, ptype = COMPONENT_TO_CM1[component]
    return {"imoist": imoist, "ptype": ptype}


def load_case(path: Path) -> DifferentialCase:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(payload) - {"case", "cm1", "gm1"})
    if unknown:
        raise CaseError(f"a differential case declares [case], [cm1] and [gm1] only; got extra {unknown}")
    case = payload.get("case", {})
    for required in ("name", "term"):
        if required not in case:
            raise CaseError(f"[case] must declare {required}")
    table = _keymap.load()
    cm1 = payload.get("cm1", {})
    owned = sorted(
        key for key in cm1
        if (entry := table.by_cm1(key)) is not None
        and entry.gm1_key is not None
        and entry.gm1_key.startswith("output.")
    )
    if owned:
        raise CaseError(
            f"[cm1] declares the &param9 output selectors {owned}, which this harness "
            "owns on both sides: HARNESS_OUTPUT makes CM1 write exactly the fields "
            "gpuwm-cm1 publishes, and moving one here would compare a field one model "
            "wrote against one the other did not"
        )
    stray = sorted(k for k in cm1 if table.by_cm1(k) is None)
    if stray:
        raise CaseError(
            f"[cm1] names keys absent from the CM1 r21.1 physical-configuration inventory: {stray}; "
            "output, statistics, parcel and restart selectors are owned by the harness"
        )
    gm1_only = payload.get("gm1", {})
    admitted = {row["gm1_key"] for row in table.gm1_only}
    stray = sorted(k for k in gm1_only if k not in admitted)
    if stray:
        raise CaseError(
            f"[gm1] admits only the keys CM1 has no word for: {stray} are not in the key map's gm1_only list; "
            "a key CM1 does have must be declared in [cm1] so both sides get it"
        )
    return DifferentialCase(
        name=case["name"],
        term=case["term"],
        notes=case.get("notes", ""),
        cm1=cm1,
        gm1_only=gm1_only,
        source=path,
    )
