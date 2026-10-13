"""The value-aware rows of the coverage report that are not refusals.

Each function here looks at the namelist's own values and says, for the
settings GM1 runs, where it runs them differently from CM1 (``differs``),
where nobody has compared them with CM1 at this value (``not_checked``) and
which settings a one-GPU run has no use for (``not_needed``).  The rows come
from the records the runtime itself keeps, so the report and a run's receipt
say the same thing:

* ``unvalidated.py`` and ``dynamics_validation_register.py`` (the receipt's
  ``unvalidated`` block): a stretched column with cloud, ``kdiv`` other than
  0.10, ``lspgrad = 1``, and the others they name;
* ``boundary_validation_register.py``: rigid walls, ``irbc = 1``, and the
  periodic sides' SGS-closure caveat (the fix removes it, and this row with it);
* ``cm1_output_selection.py`` and the translation's ``output_not_admitted``:
  the ``&param9`` words GM1 does not write, or always writes;
* ``records/GM1-STOCK-DECK-RATIOS.json``: the microphysics rows.

A row that no longer holds is removed where its fact is recorded, not here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .coverage import (
    ACK_THOMPSON,
    NOT_NEEDED,
    NOT_NEEDED_PREFIXES,
    STATISTICS_GROUP,
    Row,
    fortran,
    stock_deck_ratios,
)

if TYPE_CHECKING:  # pragma: no cover
    from .coverage import _Builder

#: Short CM1-words sentences for the &param9 words GM1 does not write.
PARAM9_WORDS: Mapping[str, str] = {
    "output_interp": "height-interpolated output (output_interp)",
    "output_coldpool": "cold-pool intensity and depth",
    "output_sfcflx": "surface fluxes",
    "output_sfcparams": "surface parameters",
    "output_sfcdiags": "surface diagnostics (2 m and 10 m values)",
    "output_psfc": "surface pressure",
    "output_zs": "terrain height",
    "output_buoyancy": "buoyancy",
    "output_pblten": "PBL tendencies",
    "output_dissten": "dissipative-heating tendency",
    "output_fallvel": "hydrometeor fall speeds",
    "output_nm": "squared Brunt-Vaisala frequency",
    "output_def": "deformation",
    "output_radten": "radiative tendencies",
    "output_pdcomp": "pressure decomposition",
}

#: Scheme names for the texts.
SCHEME_NAMES: Mapping[str, str] = {
    "morrison-mp10": "Morrison",
    "wsm6-mp6": "WSM6",
    "thompson-mp8": "Thompson",
}

#: Short texts for the runtime register's options, in CM1 words.
UNVALIDATED_TEXT: Mapping[str, str] = {
    "grid.stretch_z": (
        "A stretched vertical grid matches CM1 in dry runs, but no stretched run has been compared "
        "with CM1 once cloud forms: the rain and hail fall through layers whose depth the uniform "
        "grid never tested"
    ),
    "dynamics.kdiv": (
        "Divergence damping has been compared with CM1 only at kdiv = 0.10, the value every CM1 "
        "deck ships"
    ),
    "dynamics.lspgrad": (
        "The large-scale pressure gradient (lspgrad = 1) has been compared with CM1 only in a "
        "15-minute 100 m bubble and a column; no storm run"
    ),
}


#: Plain words for the same rows.
UNVALIDATED_PLAIN: Mapping[str, str] = {
    "grid.stretch_z": "A stretched vertical grid has not been checked against CM1 once clouds form.",
    "dynamics.kdiv": "This sound-wave damping setting has not been checked against CM1.",
    "dynamics.lspgrad": "The large-scale pressure gradient has not been checked against CM1 in a storm.",
}


def _flat(document: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {f"{table}.{name}": value for table, rows in document.items() if isinstance(rows, Mapping)
            for name, value in rows.items()}


def _int(effective: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(effective.get(key, default))
    except (TypeError, ValueError):
        return default


def _stated(builder: "_Builder", keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(key for key in keys if key in builder.stated)


def microphysics_rows(builder: "_Builder", component: str) -> list[Row]:
    choice = builder.imported.component
    if choice.status == "refused" or choice.component in (None, "off"):
        return []
    record = stock_deck_ratios()
    rows: list[Row] = []
    if choice.component == "thompson-mp8":
        rows.append(Row(
            category="not_checked",
            keys=("ptype",),
            value=builder.written("ptype"),
            text=(
                "ArWen's Thompson (mp_physics = 8), not CM1's thompson.F. It has run only a 120-s "
                "column, on an earlier ArWen; no storm has been compared with CM1 ptype = 3. ArWen "
                "2.8.9 corrected a table read in Thompson (rain collecting graupel), so a Thompson storm "
                "differs from GM1 0.1.0's, and it has not been compared with CM1 or WRF on this engine. "
                "The run waits until you acknowledge this"
            ),
            code="CM1-IMPORT-THOMPSON",
            fix="import again with --acknowledge-thompson, or tick 'Run with Thompson' on Check",
            plain="Rain and hail: Thompson. It has been checked only in a single column, not in a storm.",
            acknowledge=ACK_THOMPSON,
            evidence="src/cm1core/component_validation.py",
            stage="register",
        ))
        return rows
    scheme = record["schemes"].get(str(choice.component))
    if scheme is None:
        return rows
    low, high = scheme["ratio_range"]
    twin_low, twin_high = scheme["twin_range"]
    if choice.component == "wsm6-mp6":
        keys = ("ptype",) + _stated(builder, ("eqtset",))
        rows.append(Row(
            category="differs",
            keys=keys,
            value=builder.values(keys) if len(keys) > 1 else builder.written("ptype"),
            text=(
                "CM1 runs WSM6 with eqtset = 1 (param.F:936-948). GM1 runs eqtset = 2 dynamics with "
                "WSM6's heating converted to constant volume. On CM1's stock supercell, GM1/CM1 "
                f"(ptype = 7) mature-phase ratios are {low:.2f}-{high:.2f} (CM1 against itself: "
                f"{twin_low:.2f}-{twin_high:.2f})"
            ),
            plain=("Rain and hail: WSM6. GM1 solves WSM6 with slightly different equations from CM1's, and on "
                   "CM1's own storm GM1's run comes out stronger than CM1's (measured)."),
            evidence="records/GM1-STOCK-DECK-RATIOS.json",
            expand="ratios",
            stage="register",
        ))
        return rows
    # The category follows the release bands (the record's gate). The 5 % target is printed, not a gate: a
    # statistic past it is named in the row and in the plain sentence.
    same = scheme["all_inside_band"]
    past_target = [row for row in scheme["rows"] if not row["inside_target"]]
    worst = max(abs(row["ratio"] - 1.0) for row in scheme["rows"]) * 100.0
    rows.append(Row(
        category="same" if same else "differs",
        keys=("ptype",),
        value=builder.written("ptype"),
        text=(
            "ArWen's WRF-lineage Morrison, not CM1's morrison.F. On CM1's stock supercell, GM1/CM1 "
            f"mature-phase ratios are {low:.3f}-{high:.3f} (CM1 against itself: {twin_low:.3f}-"
            f"{twin_high:.3f})"
            + ("" if same else "; outside the release band on at least one statistic")
            + ("" if not past_target else "; past the 5 % target: " + ", ".join(
                f"{row['label']} {row['ratio']:.4f}" for row in past_target))
        ),
        evidence="records/GM1-STOCK-DECK-RATIOS.json",
        expand="ratios",
        stage="register",
        family="physics",
        plain=("Rain and hail: Morrison. On CM1's own storm it differs from CM1 by more than its release band on "
               "at least one measure (measured)." if not same else
               "Rain and hail: Morrison. On CM1's own storm it matches CM1 within 5 % (measured)." if not past_target else
               f"Rain and hail: Morrison. On CM1's own storm it matches CM1 within {int(worst * 10.0 + 0.999) / 10.0:.1f} % "
               "(measured)."),
    ))
    return rows


def output_rows(builder: "_Builder", component: str) -> list[Row]:
    translation = builder.translation
    if translation is None:
        return []
    from ..cm1_output_selection import SWITCH_BY_KEY

    rows: list[Row] = []
    by_disposition: dict[str, list[Mapping[str, Any]]] = {}
    for item in translation.output_not_admitted:
        by_disposition.setdefault(item["disposition"], []).append(item)
    unsupported = [item["cm1_key"] for item in by_disposition.get("unsupported", ())]
    pair = [key for key in ("output_km", "output_kh") if key in unsupported]
    if pair:
        rows.append(Row(
            category="differs",
            keys=tuple(pair),
            value=builder.values(pair),
            text=(
                "Not saved: eddy viscosity (kmh, kmv) and eddy diffusivity (khh, khv). GM1's "
                "turbulence closure does not keep them between steps"
            ),
            plain="Two turbulence fields (eddy viscosity and diffusivity) are not saved.",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    for key in unsupported:
        if key in pair:
            continue
        switch = SWITCH_BY_KEY.get(key)
        rows.append(Row(
            category="differs",
            keys=(key,),
            value=builder.written(key),
            text=f"Not saved: {PARAM9_WORDS.get(key, key)}. GM1 has no operator for it",
            plain="This output is not saved: GM1 cannot compute it yet.",
            detail=None if switch is None else switch.defect or None,
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    state = [item["cm1_key"] for item in by_disposition.get("state", ())
             if item["cm1_key"] not in ("output_format", "output_filetype")]
    if state:
        rows.append(Row(
            category="differs",
            keys=tuple(state),
            value=builder.values(state),
            text=(
                "Written in every GM1 frame whatever the namelist says: GM1's resume and viewer read "
                "these fields, so a 0 here is not taken"
            ),
            plain="Some fields are always saved, even where the namelist turns them off.",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    component_words = [item["cm1_key"] for item in by_disposition.get("component", ())]
    if "output_dbz" in component_words:
        rows.append(Row(
            category="differs",
            keys=("output_dbz",),
            value=builder.written("output_dbz"),
            text=(
                f"GM1 has no reflectivity operator inside the {SCHEME_NAMES.get(component, component)} step, so frames carry no "
                "dbz. It is computed after each frame with CM1's calc_refl10cm formula on the CPU "
                "by gpuwm-cm1 cm1out"
            ),
            plain="Radar reflectivity is worked out after each saved frame, with CM1's own formula.",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    swaths = [key for key in component_words if key in ("output_srs", "output_sgs")]
    if swaths:
        rows.append(Row(
            category="differs",
            keys=tuple(swaths),
            value=builder.values(swaths),
            text=(
                "Not saved: this scheme has no rainwater or graupel row, and CM1 turns these swaths "
                "off the same way (param.F:4727-4728)"
            ),
            plain="This rain and hail scheme has no rain or hail to keep a swath of.",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    stretched = [item["cm1_key"] for item in by_disposition.get("stretched-column", ())]
    if stretched:
        rows.append(Row(
            category="differs",
            keys=tuple(stretched),
            value=builder.values(stretched),
            text=(
                "Not saved on a stretched vertical grid: CM1's calcvort and calcuh take the vertical "
                "metric as one (param.F:8170), so GM1 computes these on a uniform column only"
            ),
            plain="Spin and updraft-helicity output needs evenly spaced levels.",
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    return rows


def run_rows(builder: "_Builder", component: str) -> list[Row]:
    effective = builder.effective
    rows: list[Row] = []
    statfrq = effective.get("statfrq")
    stat_keys = [key for key in sorted(builder.stated, key=lambda k: builder.order.get(k, 10**9))
                 if builder.group_of(key) == STATISTICS_GROUP]
    if isinstance(statfrq, (int, float)) and statfrq > 0:
        keys = _stated(builder, ("statfrq",)) + tuple(stat_keys)
        if keys:
            tapfrq = effective.get("tapfrq")
            rows.append(Row(
                category="differs",
                keys=keys,
                display=("statfrq, stat_* (" + str(len(stat_keys)) + " words)") if len(stat_keys) > 3 else None,
                value=builder.written("statfrq"),
                text=(
                    f"CM1 writes the &param10 statistics every {fortran(statfrq)} s. GM1 takes them "
                    f"from the history frames, every {fortran(tapfrq)} s; it does not compute "
                    "statistics during the run"
                ),
                plain="Summary statistics come from the saved frames, not from every step.",
                evidence="records/CM1-GM1-KEYMAP.json",
                stage="register",
            ))
    at_end = builder.restart_at_end()
    if at_end is not None and "rstfrq" in builder.stated:
        rows.append(Row(
            category="differs",
            keys=("rstfrq",),
            value=builder.written("rstfrq"),
            text=(
                "CM1 writes no restart files for a negative rstfrq. GM1 writes one restart at the end "
                f"(timax = {fortran(at_end)} s) so the run can be continued"
            ),
            plain="GM1 saves one restart point at the end so the run can be continued; CM1 would save none.",
            evidence="records/CM1-CADENCE-MAP.md",
            stage="register",
        ))
    # CM1's adaptive large step on a stretched column.  Measured on CM1's hurricane case (59 stretched levels,
    # 50 m at the bottom, 8 km across): with adapt_dt = 1 GM1's storm parts from CM1's once rain forms, runs
    # stopped early (hours 20.8 and 29.1) and the one that ended made a far weaker storm; with adapt_dt = 0 at
    # 7.5 s it stays close to CM1's until the last day (records/GM1-HURRICANE-RUNS.json).
    # A uniform column with adapt_dt = 1 has not shown this and gets no row.
    if _int(effective, "adapt_dt") == 1 and _int(effective, "stretch_z") >= 1 and "adapt_dt" in builder.stated:
        rows.append(Row(
            category="differs",
            keys=("adapt_dt",),
            value=builder.written("adapt_dt"),
            text=(
                "CM1's adaptive time step on stretched levels. On CM1's hurricane case GM1's storm parts from "
                "CM1's once rain forms (rain an hour early, much colder air near the sea, a far weaker storm) "
                "and runs stopped early; with a fixed step (adapt_dt = 0, dtl = 7.5) GM1's storm stays close to "
                "the one CM1 makes on its own adaptive step until the last day, when GM1's holds its strength "
                "and CM1's weakens (measured; the numbers are in the evidence record)"
            ),
            fix="set adapt_dt = 0 and a dtl the storm's fastest wind allows (the hurricane presets use 7.5 s)",
            plain=("The time step that changes with the storm runs differently from CM1's on stretched levels: "
                   "on the hurricane case it stopped early or made a much weaker storm (measured)."),
            evidence="records/GM1-HURRICANE-RUNS.json",
            stage="register",
        ))
    seed = builder.settings.get("initialization", {}).get("random_seed")
    # With no seed the run draws CM1's own random numbers (random_generator =
    # "cm1"), which match CM1 word for word; only a chosen seed differs.
    if _int(effective, "irandp") == 1 and seed is not None:
        rows.append(Row(
            category="differs",
            keys=("irandp",),
            value=builder.written("irandp"),
            text=(
                "GM1 draws the random perturbations from its own seeded stream"
                + (f" (seed {seed}, in settings.toml)" if seed is not None else "")
                + ", so they differ from CM1's; compare runs statistically, not point by point"
            ),
            plain=("The random starting bumps come from GM1's own random numbers, so the details "
                   "differ from CM1's."),
            evidence="records/CM1-GM1-KEYMAP.json",
            stage="register",
        ))
    if _int(effective, "iinit") == 8 and seed is not None:
        rows.append(Row(
            category="differs",
            keys=("iinit",),
            value=builder.written("iinit"),
            text=(
                f"The line thermal's random noise comes from GM1's own random stream (seed {seed}, chosen "
                "on import), so the details differ from CM1's as they would with another seed; import "
                "without --seed for CM1's own numbers"
            ),
            plain="The random noise that starts the storm line comes from GM1's own random numbers.",
            evidence="records/CM1-INITIALIZATION-VALIDATION.json",
            stage="register",
        ))
    scratch = [f"var{index}" for index in range(1, 21)]
    live = [key for key in scratch if key in builder.stated
            and isinstance(effective.get(key), (int, float)) and float(effective[key]) != 0.0]
    if live:
        rows.append(Row(
            category="differs",
            keys=tuple(live),
            value=builder.values(live),
            text=(
                "Stock CM1 r21.1 reads var1-var20 only in hand-edited Fortran. GM1 runs stock CM1, "
                "so whatever your edited code does with them is not run"
            ),
            plain="Settings used only by edited CM1 code are not run.",
            evidence="records/CM1-GM1-KEYMAP.json",
            stage="register",
        ))
    for key, what in (("dodomaindiag", "domain diagnostics (&param14)"), ("doazimavg", "azimuthal averages (&param15)")):
        if key in builder.stated and effective.get(key) is True:
            rows.append(Row(
                category="differs",
                keys=(key,),
                value=builder.written(key),
                text=f"Not saved: CM1's {what}. GM1 has no writer for them",
                plain="This extra CM1 output is not saved.",
                evidence="records/CM1-GM1-KEYMAP.json",
                stage="register",
            ))
    if "outunits" in builder.stated and _int(effective, "outunits", 1) != 1:
        rows.append(Row(
            category="differs",
            keys=("outunits",),
            value=builder.written("outunits"),
            text="The cm1out files carry coordinates in km, CM1's outunits = 1",
            plain="Coordinates are saved in kilometres.",
            evidence="records/CM1-GM1-KEYMAP.json",
            stage="register",
        ))
    return rows


def not_checked_rows(builder: "_Builder", component: str) -> list[Row]:
    translation = builder.translation
    if translation is None:
        return []
    from .. import unvalidated
    from ..boundary_validation_register import register as boundary_register
    from ..boundary_validation_register import unvalidated_for

    document = translation.gm1
    flat = _flat(document)
    rows: list[Row] = []
    seen: set[str] = set()

    def option_row(option: str, text: str, evidence: str | None, detail: str | None = None) -> None:
        keys = builder.cm1_keys(option) or (option,)
        if option in seen:
            return
        seen.add(option)
        rows.append(Row(
            category="not_checked",
            keys=keys,
            value=builder.values(keys),
            text=text,
            detail=detail,
            evidence=evidence,
            option=option,
            stage="register",
            plain=UNVALIDATED_PLAIN.get(option, "This setting has not been checked against CM1 at this value."),
        ))

    for capability in unvalidated.selected(flat):
        option_row(capability.option, UNVALIDATED_TEXT.get(capability.option, capability.not_checked),
                   "src/cm1core/unvalidated.py", capability.not_checked)
    dynamics = SimpleNamespace(**document.get("dynamics", {}))
    for capability in unvalidated.dynamics_rows(dynamics):
        option_row(capability.option, UNVALIDATED_TEXT.get(capability.option, capability.not_checked),
                   "src/cm1core/dynamics_validation_register.py", capability.not_checked)
    boundaries = document.get("boundaries", {})
    closures = document.get("closures", {})
    lateral = [boundaries.get(side) for side in ("west", "east", "south", "north")]
    if all(isinstance(value, int) for value in lateral):
        selection = unvalidated_for(
            SimpleNamespace(
                west=lateral[0], east=lateral[1], south=lateral[2], north=lateral[3],
                radiative_order=boundaries.get("radiative_order", 4),
                roflux=boundaries.get("roflux", 0), nudgeobc=boundaries.get("nudgeobc", 0),
            ),
            SimpleNamespace(upper_damping=closures.get("upper_damping", 0), hrdamp=closures.get("hrdamp", 0)),
        )
        sides = ("wbc", "ebc", "sbc", "nbc")
        for name, entry in selection["unvalidated"].items():
            if name.startswith("wbc/ebc/sbc/nbc=3,4"):
                keys = tuple(side for side, value in zip(sides, lateral) if value in (3, 4))
                text = ("Rigid walls: raw-word exact against CM1's bc.F, but no CM1-vs-GM1 run has been "
                        "compared for a walled domain")
            elif name.startswith("irbc=1"):
                keys = ("irbc",)
                text = ("irbc = 1 (Klemp-Wilhelmson on the large step): no CM1-vs-GM1 run has been "
                        "compared; only irbc = 2's small-step position was")
            else:
                continue
            plain = ("Solid side walls have not been checked against CM1." if keys != ("irbc",) else
                     "This open-side setting has not been checked against CM1.")
            rows.append(Row(category="not_checked", keys=keys, value=builder.values(keys), text=text,
                            detail=entry["not_checked"], evidence="records/CM1-SUPPORT-MATRIX.csv",
                            stage="register", plain=plain))
        periodic = [side for side, value in zip(sides, lateral) if value == 1]
        caveat = boundary_register()["wbc/ebc/sbc/nbc=1 (periodic)"].get("caveat") or ""
        if periodic and "SGS closure" in caveat:
            rows.append(Row(
                category="not_checked",
                keys=tuple(periodic),
                value=builder.values(periodic),
                text=(
                    "Periodic sides: the momentum faces match CM1's, but GM1's turbulence scheme "
                    "still applies the open-side edge treatment on a periodic side, where CM1 does not "
                    "(turbtend.F:67-102)"
                ),
                plain="Repeating sides have not been fully checked against CM1 yet.",
                detail=caveat,
                evidence="records/CM1-SUPPORT-MATRIX.csv",
                stage="register",
            ))
    if closures.get("bbc") == 3 and closures.get("sfcmodel") == 1:
        keys = _stated(builder, ("bbc", "sfcmodel")) or ("bbc", "sfcmodel")
        rows.append(Row(
            category="not_checked",
            keys=keys,
            value=builder.values(keys),
            text=(
                "The semi-slip bottom with the surface layer (bbc = 3, sfcmodel = 1) has been compared with "
                "CM1 in one storm on one card: CM1's supercell with cecd = 1 on an RTX 4090, where nine "
                "mature-phase statistics are 0.986 to 1.018 of CM1's and max vertical vorticity is 1.072 "
                "(measured). It has not been compared in any other storm or on any other card, and the "
                "15-minute 100 m bubble check at these settings is still to run"
            ),
            plain=("Ground friction has been compared with CM1 in one storm on one card only: CM1's supercell "
                   "on an RTX 4090 (measured)."),
            evidence="records/CM1-SUPPORT-MATRIX.csv",
            stage="register",
        ))
    return rows


def not_needed_rows(builder: "_Builder", component: str) -> list[Row]:
    rows: list[Row] = []
    grouped: dict[str, list[str]] = {}
    for key in sorted(builder.stated, key=lambda k: builder.order.get(k, 10**9)):
        if key in NOT_NEEDED:
            rows.append(Row(category="not_needed", keys=(key,), value=builder.written(key), text=NOT_NEEDED[key],
                            stage="register", plain="Not needed for a run on one GPU."))
            continue
        for prefix, text in NOT_NEEDED_PREFIXES.items():
            if key.startswith(prefix):
                grouped.setdefault(text, []).append(key)
    for text, keys in grouped.items():
        prefix = next(prefix for prefix, value in NOT_NEEDED_PREFIXES.items() if value == text)
        rows.append(Row(category="not_needed", keys=tuple(keys), value=builder.values(keys), text=text,
                        display=f"{prefix}* ({len(keys)})" if len(keys) > 3 else None, stage="register",
                        plain="Not needed for a run on one GPU."))
    return rows


def known_rows(builder: "_Builder", component: str) -> list[Row]:
    return (
        microphysics_rows(builder, component)
        + output_rows(builder, component)
        + run_rows(builder, component)
        + not_checked_rows(builder, component)
        + not_needed_rows(builder, component)
    )


__all__ = ["PARAM9_WORDS", "UNVALIDATED_TEXT", "known_rows"]
