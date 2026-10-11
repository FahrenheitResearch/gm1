"""Fresh-install diagnostics that never open a CUDA context."""

from __future__ import annotations

import os

import importlib.util
import json
import sys
from dataclasses import dataclass
from importlib import resources

from . import CM1_AUTHORITY_SHA256, MATURITY, __version__
from .engine_pin import GPUWM_REQUIREMENT, inspect
from .rust_tools import NativeToolError, verify_native_tools


OK = "ok"
PRESENT = "present"
MISSING = "missing"
INFO = "info"
#: A capability a document can select that nothing has measured.  Not blocking
#: -- it is selectable on purpose -- but never silent.  See cm1core.unvalidated.
UNVALIDATED = "unvalidated"
#: The four findings below describe the source-anchored whole-model product
#: lane behind ``gpuwm-cm1 init``/``forecast``, which refuses at its trajectory
#: gate.  They stay in the report, but they say nothing about whether this
#: installation can run the Cartesian lane (``gpuwm-cm1 cartesian``), so they
#: are informational and never set doctor's exit status.
LEGACY_LANE_PREFIX = (
    "[legacy init/forecast product lane; informational, does not affect the "
    "Cartesian lane] "
)
#: The token every UNVALIDATED finding's detail carries, so a reader
#: scanning an install for what is unproven greps one word.
UNVALIDATED_MARKER = "NOT checked"


@dataclass(frozen=True, slots=True)
class Finding:
    subject: str
    status: str
    detail: str
    required: bool = False
    remedy: str | None = None

    def __post_init__(self) -> None:
        """Guarantee the one word a reader greps for.

        Every capability this distribution exposes without a CM1 comparison
        behind it has to be findable by scanning for a single token; that is
        the whole reason UNVALIDATED is a status of its own rather than INFO.
        The producers of these findings were written independently and spell
        the marker five different ways -- "UNVALIDATED", "NOT VALIDATED",
        "NOT checked", "NOT performed" -- and three of them state the limit in
        prose without any marker at all.  Rather than depend on every future
        producer remembering the convention, the invariant is enforced here:
        an UNVALIDATED finding whose detail does not already carry the token
        gets it prefixed.
        """

        if self.status == UNVALIDATED and UNVALIDATED_MARKER not in self.detail:
            object.__setattr__(
                self, "detail", f"{UNVALIDATED_MARKER}: {self.detail}"
            )

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "subject": self.subject,
            "status": self.status,
            "detail": self.detail,
            "required": self.required,
        }
        if self.remedy is not None:
            payload["remedy"] = self.remedy
        return payload


def _boundary_validation_findings() -> list[Finding]:
    """Name every lateral capability that is selectable and not yet measured.

    ``gpuwm-cm1 doctor`` is where a user looks before trusting a run, so the
    boundary capabilities this distribution exposes without a CM1-vs-GM1
    differential are listed here by CM1's own names, one finding each, with the
    sentence saying what was not checked.  They are informational, not
    blocking: the capability runs, and the claim about it is bounded.
    """

    from .boundary_validation_register import register

    findings: list[Finding] = []
    for name, entry in register().items():
        if entry["differential"] is not None:
            findings.append(
                Finding(
                    f"CM1 lateral boundary: {name}",
                    OK,
                    f"differential against CM1 r21.1: {entry['differential']}; checked: {entry['checked']}",
                    required=False,
                )
            )
            continue
        findings.append(
            Finding(
                f"CM1 lateral boundary: {name}",
                UNVALIDATED,
                f"selectable and running; checked: {entry['checked']}; NOT checked: {entry['not_checked']}",
                required=False,
                remedy=(
                    "run tools/oracle/differential.py against CM1 r21.1 with this "
                    f"capability selected ({entry['cm1']}) and record the comparison"
                ),
            )
        )
    return findings


def _output_selection_findings() -> list[Finding]:
    """CM1 &param9: what is selectable, what moved, and what is unproven.

    Read from the selection table itself, so a product added later without a
    validation note shows up here rather than quietly shipping as checked.
    """

    from .cm1_output_selection import (
        CM1_OUTPUT_AUTHORITY,
        DEFAULT_MOVES,
        DEFAULT_OUTPUT_SELECTION,
        PRODUCT,
        STATE,
        SWITCHES,
        UNSUPPORTED as _UNSUPPORTED,
    )

    counts = {PRODUCT: 0, STATE: 0, _UNSUPPORTED: 0}
    for switch in SWITCHES:
        counts[switch.disposition] += 1
    findings = [
        Finding(
            "CM1 &param9 output selection",
            OK,
            (
                f"{len(SWITCHES)} output_* words admitted from {CM1_OUTPUT_AUTHORITY}; "
                f"{counts[PRODUCT]} select a field this lane computes, "
                f"{counts[STATE]} name a carrier every frame already holds, "
                f"{counts[_UNSUPPORTED]} are refused by value with the wrong "
                "result they prevent named; "
                f"default selection publishes "
                f"{len(DEFAULT_OUTPUT_SELECTION.selected_names())} CM1-named fields"
            ),
            required=False,
        ),
        Finding(
            "CM1 &param9 defaults moved from CM1's distributed deck",
            INFO,
            "; ".join(
                f"{key}: CM1 {DEFAULT_OUTPUT_SELECTION.record()['cm1_distributed_defaults'][key]} "
                f"-> this lane {DEFAULT_OUTPUT_SELECTION[key]} ({DEFAULT_MOVES[key].split('.')[0]})"
                for key in DEFAULT_OUTPUT_SELECTION.moved_from_cm1
            )
            or "none; every default matches run/namelist.input",
            required=False,
        ),
    ]
    for row in DEFAULT_OUTPUT_SELECTION.unvalidated_markers():
        findings.append(
            Finding(
                f"UNVALIDATED {row['option']} ({', '.join(row['cm1_output_names'])})",
                UNVALIDATED,
                row["not_checked"],
                required=False,
                remedy=(
                    "the field is selectable and is published; nothing in this "
                    "distribution has checked it, and every run receipt carries "
                    "the same marker under output_products.unvalidated"
                ),
            )
        )
    return findings


def _records_finding() -> Finding:
    required = {
        "CM1-AUTHORITY.json",
        "CM1-SOURCE-MAP.md",
        "CM1-STATE-MAP.csv",
        "CM1-CADENCE-MAP.md",
        "CM1-SUPPORT-MATRIX.csv",
        "CM1-ARITHMETIC-CONTRACT.md",
        "CM1-ARWEN-SEAM.md",
        "CM1-NEST-SEAM.md",
        "CM1-ORACLE-MANIFEST.json",
        "CM1-PROFILE-MANIFEST.json",
        "CM1-RISK-REGISTER.md",
    }
    root = resources.files("cm1core.records")
    missing = sorted(name for name in required if not root.joinpath(name).is_file())
    if missing:
        return Finding(
            "installed evidence records",
            MISSING,
            f"missing {missing}",
            required=True,
            remedy="reinstall a complete gpuwm-cm1 artifact built from a checkout that passes packaging tests",
        )
    return Finding(
        "installed evidence records",
        OK,
        f"all {len(required)} required and profile records are reachable",
        required=True,
    )


def _cartesian_inputs_finding() -> Finding:
    """Name what the Cartesian door would have to be given explicitly.

    Neither input is required for the other doors, so an absent one is not a
    blocking finding; it is the difference between one command and one command
    with two paths typed out.
    """

    from . import rust_tools
    from .cartesian_door import (
        CARTESIAN_INITIALIZER_LIBRARY,
        CARTESIAN_SOUNDING_RESOURCE,
    )
    from .data import PackagedInputError, required_input_identity

    # Resolve both the way the door does, so this finding cannot report a
    # healthy installation that the door then refuses, or the reverse.
    missing: list[str] = []
    try:
        rust_tools.native_library_path(CARTESIAN_INITIALIZER_LIBRARY)
    except (OSError, ValueError, rust_tools.NativeToolError):
        missing.append("--library (this wheel installs no admitted initializer library)")
    # The sounding is never installed: CM1-AUTHORITY.json's package policy is
    # to ship identities rather than CM1 runtime assets, so --sounding is
    # always required.  Reporting it as resolving would contradict the door.
    try:
        identity = required_input_identity(CARTESIAN_SOUNDING_RESOURCE)
        missing.append(
            "--sounding (CM1 runtime assets are not redistributed; supply "
            f"{identity['source_release']}/{identity['source_path']}, "
            f"{identity['bytes']} bytes, sha256 {identity['sha256']})"
        )
    except (PackagedInputError, OSError, ValueError) as error:
        missing.append(f"--sounding (no recorded input identity: {error})")
    if missing:
        return Finding(
            "GM1 Cartesian runtime inputs",
            INFO,
            "`gpuwm-cm1 cartesian` requires " + " and ".join(missing),
            required=False,
            remedy="pass the named arguments; tools/authority/fetch_cm1_21_1.py retrieves the CM1 release the sounding comes from",
        )
    return Finding(
        "GM1 Cartesian runtime inputs",
        PRESENT,
        "the installed initializer library and input sounding both resolve without arguments",
        required=False,
    )


def _unvalidated_findings() -> list[Finding]:
    """One finding per exposed, unmeasured capability.

    These are not blocking: a capability is selectable the moment it is
    reachable.  They are loud so that a user who
    selects one learns it here rather than from a result nobody checked.
    """

    from .unvalidated import UNVALIDATED_CAPABILITIES

    return [
        Finding(
            f"unvalidated capability {row.option}",
            UNVALIDATED,
            row.line(),
            required=False,
            remedy=f"CM1 authority {row.authority}; measure it differentially against CM1 and delete the row",
        )
        for row in UNVALIDATED_CAPABILITIES
    ]


def _initialization_validation_findings() -> list[Finding]:
    """Say, by value, which initialization selectors are unvalidated.

    Every capability is selectable now, and
    anything not validated in the pass that exposed it carries a loud,
    machine-readable marker naming what was not checked.  This is that marker
    in ``doctor``: one line per selector value whose row in
    ``records/CM1-INITIALIZATION-VALIDATION.json`` is not ``validated``.
    """

    from .initialization_validation import UNVALIDATED_MARKER, validated_rows

    try:
        rows = validated_rows()
    except (OSError, ValueError) as error:
        return [
            Finding(
                "initialization validation record",
                MISSING,
                f"unreadable: {error}",
                required=True,
                remedy="reinstall a complete gpuwm-cm1 artifact",
            )
        ]
    unvalidated = [row for row in rows if row.get("status") != "validated"]
    findings = [
        Finding(
            "initialization selectors validated against CM1",
            OK,
            f"{len(rows) - len(unvalidated)} of {len(rows)} selector values are checked "
            "against the real CM1 r21.1 binary at model time zero "
            "(tools/oracle/initialization_differential.py)",
        )
    ]
    for row in unvalidated:
        findings.append(
            Finding(
                f"{UNVALIDATED_MARKER}: initialization.{row['selector']} = {row['value']}",
                INFO,
                f"not checked: {row.get('not_checked') or 'not stated'}"
                + (f" | what was checked instead: {row['method']}" if row.get("method") else ""),
                remedy=(
                    "this capability is selectable and its receipt carries the same "
                    "marker; treat any result that depends on it as unconfirmed"
                ),
            )
        )
    return findings


def _dynamics_findings() -> list[Finding]:
    """One line per CM1 dynamical selector, and what stands behind it.

    A capability that is reachable but unmeasured is the thing an operator is
    most likely to trust by accident, so it is printed here beside the ones
    that were measured rather than left to a receipt they may never read.
    """

    from .dynamics_validation_register import (
        DYNAMICS_CAPABILITIES,
        UNVALIDATED,
        VALIDATED,
    )

    findings: list[Finding] = []
    for item in DYNAMICS_CAPABILITIES:
        if item.status == VALIDATED:
            findings.append(
                Finding(
                    f"CM1 {item.cm1_name} ({item.key})",
                    OK,
                    f"{item.authority}; {item.validation}",
                )
            )
        elif item.status == UNVALIDATED:
            findings.append(
                Finding(
                    f"CM1 {item.cm1_name} ({item.key})",
                    UNVALIDATED,
                    f"{item.authority}; selectable, NOT VALIDATED: {item.not_checked}",
                    remedy=(
                        "the key is admitted and reaches the kernel; treat any "
                        "result that depends on it as unmeasured until this "
                        "line says otherwise"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    f"CM1 {item.cm1_name} ({item.key})",
                    INFO,
                    f"{item.authority}; {item.validation}",
                )
            )
    return findings


def _surface_layer_finding() -> Finding:
    """What the surface layer is, and exactly what was not measured.

    ``src/sfcphys.F:81-236`` getcecd, ``src/turb.F:292`` psfc,
    ``src/sfcphys.F:341-361`` sfcflux and ``src/turbtend.F:588-596`` are each
    raw-word identical to CM1 r21.1's own statements compiled by gfortran.
    That is not the same as a measured storm, and this finding says so by
    name rather than letting a run receipt imply otherwise.
    """

    from .cuda_surface_drag import SURFACE_UNVALIDATED

    return Finding(
        "CM1 surface layer and land surface",
        UNVALIDATED,
        json.dumps(
            {
                "selectable": [
                    "closures.bbc=1/2/3",
                    "closures.sfcmodel=0/1",
                    "closures.cecd=1/2/3",
                    "closures.cnstcd",
                    "closures.cnstce",
                    "closures.set_znt/cnst_znt",
                    "closures.set_ust/cnst_ust",
                    "closures.isfcflx=0/1",
                    "surface.initsfc=1/2/3/4",
                    "surface.tsk0_k",
                    "surface.tmn0_k",
                    "surface.xland0",
                    "surface.lu0",
                    "surface.season",
                    "surface.oceanmodel=1",
                ],
                "validated_raw_word_against_gfortran_cm1": [
                    "src/sfcphys.F:83-99 getcecd set_znt",
                    "src/sfcphys.F:102-118 getcecd set_ust",
                    "src/sfcphys.F:120-136 getcecd cecd=1",
                    "src/sfcphys.F:138-194 getcecd cecd=2/3 and the land branch",
                    "src/sfcphys.F:197-236 getcecd ch/cq",
                    "src/turb.F:292 psfc",
                    "src/sfcphys.F:341-361 sfcflux with src/cm1libs.F:37-41 rslf",
                    "src/turbtend.F:588-596,625-690 the scalar lower flux",
                ],
                "validated_on_a_device": {
                    "card": "NVIDIA GeForce RTX 5070 Ti (sm_120)",
                    "bit_identical": [
                        "cecd=1 constant drag",
                        "set_znt=1 prescribed roughness",
                        "isfcflx=1 enthalpy flux over them",
                    ],
                    "within_3_ulp": [
                        "set_ust=1 prescribed friction velocity",
                        "cecd=2/3 wind-dependent water drag",
                    ],
                    "receipt": (
                        "evidence/validation/gm1-surface-layer-device-v1/"
                        "receipt.json"
                    ),
                },
                "unvalidated": dict(SURFACE_UNVALIDATED),
                "refused_by_name": [
                    "closures.sfcmodel=2..7 (the WRF and MOST surface layers)",
                    "surface.oceanmodel=2 (the Pollard ocean mixed layer)",
                    "set_flx=1 prescribed surface fluxes",
                    "use_avg_sfc, pertflx, sfcdiags and every PBL scheme",
                ],
            },
            sort_keys=True,
        ),
        required=False,
        remedy=(
            "run the surface selectors on a GPU and against cm1.exe over a "
            "storm before reporting a measured surface trajectory"
        ),
    )


def _column_component_findings() -> list[Finding]:
    """One row per selectable column component, plus one per blocked scheme.

    Every capability is selectable, and
    anything unvalidated carries a loud, machine-readable marker naming what
    was not checked.  These rows are that marker in ``doctor``; the same words
    come from the same module into every run receipt.
    """

    from .column_component_blockers import blocked_component_records
    from .component_validation import (
        UNVALIDATED_MARKER,
        component_validation,
        validation_records,
    )
    from .column_components import COLUMN_COMPONENT_SPECS

    from .column_components import UNLISTED_COMPONENTS

    findings: list[Finding] = []
    for component_id, spec in COLUMN_COMPONENT_SPECS.items():
        if component_id in UNLISTED_COMPONENTS and os.environ.get(UNLISTED_COMPONENTS[component_id]) != "1":
            # Not offered by this release (the door refuses it by value), so it
            # is not advertised here either; one line says why.  A developer
            # tree with the switch on sees the component's own row.
            findings.append(
                Finding(
                    f"column component {component_id}",
                    INFO,
                    "not offered in this release: pending a licence review of "
                    "routines it reaches, and no CM1 ptype selects it",
                    required=False,
                )
            )
            continue
        state = component_validation(component_id)
        ptype = (
            f"CM1 ptype={spec.cm1_ptype}"
            if spec.cm1_ptype is not None
            else "no CM1 ptype"
        )
        findings.append(
            Finding(
                f"column component {component_id}",
                OK if state.validated else UNVALIDATED,
                f"{spec.scheme_name}, {ptype}, ArWen mp_physics={spec.mp_physics}, "
                f"{spec.profile.numq} transported scalars. {state.sentence()}",
                required=False,
                remedy=None
                if state.validated
                else (
                    "run it and compare: a CM1 differential where CM1 has the same "
                    "scheme, otherwise a WRF or analytic check.  Until then every "
                    f"receipt from this component carries {UNVALIDATED_MARKER}"
                ),
            )
        )
    blocked = blocked_component_records()
    if blocked:
        findings.append(
            Finding(
                "column components not yet selectable",
                INFO,
                "; ".join(
                    f"{record['component_id']} ({record['scheme_name']}, ArWen "
                    f"mp_physics={record['arwen_mp_physics']}): {record['blocker']}"
                    for record in blocked
                ),
                required=False,
                remedy=(
                    "src/cm1core/column_component_blockers.py names the exact change "
                    "each one needs"
                ),
            )
        )
    _ = validation_records
    return findings


def _physics_family_findings() -> list[Finding]:
    """Say, for every CM1 radiation/PBL/cumulus value, what GM1 would do.

    This is the loud half of "wire everything; flag the unvalidated": a
    reader who wants to know whether ``radopt=2`` will run, and what has
    been measured about it, reads it here rather than from a run that
    refuses two minutes in.
    """

    from .arwen_physics_component import (
        ADMITTED,
        IDENTITY,
        UNCOUPLED,
        UNPORTED,
        RRTMG_LOOKUP_TABLES,
        component_engine_report,
        component_inventory,
    )

    findings: list[Finding] = []
    inventory = component_inventory()
    for family in ("radiation", "pbl", "cumulus"):
        rows = inventory[family]
        runnable = [row for row in rows if row["disposition"] in (ADMITTED, IDENTITY)]
        uncoupled = [row for row in rows if row["disposition"] == UNCOUPLED]
        unported = [row for row in rows if row["disposition"] == UNPORTED]
        selector = rows[0]["cm1_selector"]
        detail = json.dumps(
            {
                "selector": selector,
                "runnable_values": [row["cm1_value"] for row in runnable],
                "uncoupled_values": {
                    str(row["cm1_value"]): row.get("missing_carriers")
                    or row.get("minimum_engine")
                    for row in uncoupled
                },
                "unported_values": [row["cm1_value"] for row in unported],
                "unvalidated": [
                    row["cm1_value"] for row in rows if row["cm1_value"] != 0
                ],
            },
            sort_keys=True,
        )
        findings.append(
            Finding(
                f"CM1 {family} family ({selector})",
                INFO,
                detail,
                required=False,
                remedy=(
                    f"every nonzero {selector} is UNVALIDATED: no CM1-vs-GM1 "
                    "differential and no offline oracle has measured it in a "
                    "coupled GM1 run; uncoupled values name the carriers they "
                    "still need and refuse at preflight rather than running "
                    "the neutral profile"
                ),
            )
        )
    report = component_engine_report("rrtmg-legacy-ra4")
    findings.append(
        Finding(
            "ArWen RRTMG lookup tables",
            OK if report["admitted"] else INFO,
            json.dumps(
                {
                    "tables": dict(RRTMG_LOOKUP_TABLES),
                    "engine": report,
                },
                sort_keys=True,
            ),
            required=False,
            remedy=None
            if report["admitted"]
            else "install an ArWen estate carrying gpuwm/data/wrf_radiation to reach radopt=2",
        )
    )
    findings.append(
        Finding(
            "CM1 rterm=1 idealized radiation",
            OK,
            "the capped Rotunno-Emanuel theta tendency (src/solve1.F:1003-1054) "
            "is live in the Cartesian closure and selectable as closures.rterm. "
            "MEASURED: 0 of 11339 words differ from the evidenced CPU operator on "
            "an RTX 5070 Ti, and a paired CM1 differential reads a regression "
            "coefficient of 1.0001-1.0015 over 18 frames "
            "(evidence/radpbl/receipt.json)",
            required=False,
        )
    )
    return findings


def _cartesian_restart_finding() -> Finding:
    """Report the Cartesian lane's restart capability and what is unchecked.

    CM1 saves "all information that is carried from one timestep to the next"
    (``README.restart:12-15``) and resumes it with ``irst``/``rstnum``.  The
    Cartesian lane now writes the equivalent at ``run.restart_interval_s``
    through the Rust ``gm1-cartesian-restart`` tool.  Whatever this pass could
    not check is named here by subject, so a fresh installation reads the
    limits without running anything.
    """

    from . import rust_tools
    from .cartesian_restart import CARTESIAN_RESTART_CLAIM_LIMIT

    # What the restart carries, and therefore what it does not: the base state,
    # the grid metrics and the uniform-grid coefficients are rebuilt from the
    # pinned sounding and geometry rather than written into the artifact
    # (README.namelist:1512-1516).  A reader needs that limit before trusting a
    # resumed leg, so it is reported here rather than only in a run receipt.
    unchecked = CARTESIAN_RESTART_CLAIM_LIMIT
    try:
        rust_tools.verify_native_tools()["gm1-cartesian-restart"]
    except (OSError, ValueError, rust_tools.NativeToolError) as error:
        return Finding(
            "GM1 Cartesian restart",
            MISSING,
            (
                "the gm1-cartesian-restart executable that owns every durable "
                f"restart byte is not admitted: {error}. CLAIM LIMIT: {unchecked}"
            ),
            required=False,
            remedy=(
                "reinstall a sealed platform wheel, or pass --restart-tool with "
                "the built gm1-cartesian-restart executable"
            ),
        )
    return Finding(
        "GM1 Cartesian restart",
        PRESENT,
        (
            "`gpuwm-cm1 cartesian --integrate --restart-dir DIR` writes one "
            "CM1FCP1 restart every run.restart_interval_s (src/cm1.F:3186,4246) "
            "and run.irst = 1 with run.rstnum = N resumes file set N "
            "(README.restart:28-49). CLAIM LIMIT: " + unchecked
        ),
        required=False,
        remedy=(
            "read the run receipt's restart block for what each published "
            "artifact does and does not carry"
        ),
    )


def _ensemble_finding() -> Finding:
    """Report the ensemble member switch and what it does not perturb."""

    from .ensemble_branch import (
        CM1_IRANDP_AMPLITUDE_K,
        ENSEMBLE_PERTURBED_CARRIERS,
    )

    return Finding(
        "GM1 ensemble members",
        PRESENT,
        (
            "an [ensemble] table with member/seed/theta_amplitude_k runs CM1's "
            "irandp perturbation (src/init3d.F:1446-1483, amplitude "
            f"{CM1_IRANDP_AMPLITUDE_K} K at src/init3d.F:1466) on "
            f"{', '.join(ENSEMBLE_PERTURBED_CARRIERS)} at the initialized state "
            "or at the restart named by [restart] rstnum, from a seeded "
            "splitmix64 stream so a member can be re-run. UNVALIDATED: the "
            "perturbation reaches the theta carriers only -- the pressure and "
            "density carriers are not re-derived, so a member starts "
            "momentarily out of exact thermodynamic balance and the first "
            "acoustic step adjusts it; no spread, calibration or skill claim "
            "of any kind is made about the resulting members"
        ),
        required=False,
        remedy=(
            "read runtime_benchmark.ensemble_branch in the run receipt for the "
            "SHA-256 of the field a member was actually given"
        ),
    )


def _lagrangian_findings() -> list[Finding]:
    """What the [lagrangian] table can select, and what carries no CM1 proof.

    Every capability is listed whether or not any document currently selects
    it, because ``doctor`` is read before a run is configured, not after.
    """

    from .lagrangian_config import LAGRANGIAN_KEYS
    from .lagrangian_stage import UNVALIDATED_MARKERS

    findings = [
        Finding(
            "GM1 inline Lagrangian and budget analysis",
            PRESENT,
            (
                "the [lagrangian] table selects "
                f"{len(LAGRANGIAN_KEYS)} keys: CM1 parcels (iprcl, nparcels, "
                "prclfrq_s and the sixteen &param13 prcl_* switches, positions in "
                "[initialization].iorigin's frame; "
                "src/parcel.F:10, src/init3d.F:355-411, src/param.F:6239-6282 "
                "resolving the switches, src/param.F:6385-6500), "
                "the Eulerian vorticity budget, material circuits and their "
                "circulation, CM1's pdcomp decomposition (run/namelist.input:277; "
                "pipb/pipdl/pipdn/pipc inverted with poiss.F's mirrored open-lateral "
                "transform, src/param.F:5178-5205) and "
                "the high-frequency window (src/cm1.F:2645-2647). Every "
                "selector is a diagnostic: none of them writes a prognostic "
                "carrier, so none can change the simulated storm"
            ),
            required=False,
        ),
        Finding(
            "GM1 parcel trajectories vs CM1",
            OK,
            (
                "the RK2 integrator, the face searches on the grid's own deck, "
                "the staggered trilinear stencil, the parcel.F:59-130 vertical "
                "boundary fill, the bbc=3 log layer (parcel.F:470-515), the "
                "periodic exit and the attribute inventory are transcribed from CM1 r21.1 "
                "src/parcel.F and are advected with solve3.F:444-465's "
                "acoustic small-step-averaged rru/rrv/rrw, which is the "
                "velocity CM1 itself uses for psolver=3"
            ),
            required=False,
        ),
    ]
    for subject, detail in sorted(UNVALIDATED_MARKERS.items()):
        findings.append(
            Finding(
                f"GM1 lagrangian: {subject}",
                UNVALIDATED,
                detail,
                required=False,
                remedy=(
                    "select it and read the marker in the run receipt under "
                    "lagrangian.unvalidated; it is exposed deliberately and "
                    "the marker travels with every run that uses it"
                ),
            )
        )
    return findings


def _cupy_finding() -> Finding:
    """Every Cartesian kernel runs through CuPy; without it nothing runs."""

    present = importlib.util.find_spec("cupy") is not None
    return Finding(
        "CuPy import surface",
        PRESENT if present else MISSING,
        "module discoverable; CUDA runtime deliberately not imported by doctor"
        if present
        else (
            "module not discoverable; the Cartesian lane compiles and launches "
            "every kernel through CuPy/NVRTC, so this installation cannot run "
            "`gpuwm-cm1 cartesian --integrate`"
        ),
        required=True,
        remedy=None
        if present
        else 'pip install "gpuwm-cm1[gpu]" (CuPy for CUDA 13 with NVRTC; NVIDIA driver R580 or newer)',
    )


def collect() -> list[Finding]:
    try:
        native_tools = verify_native_tools()
        native_error = None
    except (OSError, NativeToolError) as error:
        native_tools = None
        native_error = str(error)
    findings = [
        Finding(
            "Python",
            OK if sys.version_info >= (3, 11) else MISSING,
            sys.version.split()[0],
            required=True,
            remedy="install Python 3.11 or newer"
            if sys.version_info < (3, 11)
            else None,
        ),
        Finding(
            "gpuwm-cm1 distribution",
            PRESENT,
            f"version={__version__}, maturity={MATURITY}, CM1 authority={CM1_AUTHORITY_SHA256}",
            required=True,
        ),
        _records_finding(),
    ]
    findings.extend(_initialization_validation_findings())
    findings.extend(_dynamics_findings())
    findings.extend(_column_component_findings())
    estate = inspect()
    findings.append(
        Finding(
            "ArWen audited WSM6 surface",
            OK if estate.admitted else MISSING,
            json.dumps(estate.as_dict(), sort_keys=True),
            required=True,
            remedy=None
            if estate.admitted
            else f'pip install "{GPUWM_REQUIREMENT}" and re-run doctor',
        )
    )
    findings.extend(
        (
            Finding(
                "CM1-neutral ArWen microphysics seam",
                OK,
                "the CM1-owned wsm6-mp6 adapter reaches gpuwm.core.microphysics.apply without importing private WSM6 or the full MPAS phase1+phase2 seam; three RTX 3080 product-mode columns pass the direct WRF v4.6.1 oracle",
                required=True,
            ),
            Finding(
                "CM1 reference instruments",
                MISSING,
                LEGACY_LANE_PREFIX
                + "the source ledger is partial; O-GRID-BASE-V1 and O-POST-GETSET-V1 pass raw-bit O0/O2/repeat plus all four controls, and O-TERRAIN-DEFORMATION-SGS-V1 passes one selected first-step closure slice, but post-GETSET covers only 46 named allocations (prs scientific comparison is physical-interior-only), omits rf0/rr0/rrf0/qc0/qi0/surface/late-turbulence state, and most time-dependent decks remain absent",
                required=False,
                remedy="preserve both bounded startup decks and source-live masks, capture omitted initialization state, complete the source ledger, and finish the remaining negative-controlled instrument campaign",
            ),
            Finding(
                "CM1 CUDA dycore",
                MISSING,
                LEGACY_LANE_PREFIX
                + "audited structural initialization, dry RK/advection, vertical operators, a psolver=3 implicit column, and the selected first-step terrain calcdef/local SGS source pass bounded CPU/GPU comparisons; horizontal acoustic composition, family boundaries, evolved/general closure fluxes and a complete source-ordered large-step trajectory remain open",
                required=False,
                remedy="compose the horizontal psolver=3 path and the boundary closures with the existing slices, then pass complete-step and multi-step native trajectory gates",
            ),
            Finding(
                "eqtset=2 coupled trajectory",
                MISSING,
                LEGACY_LANE_PREFIX
                + "the eqtset=2 lifecycle and public ArWen WSM6 bridge pass bounded CPU/sm86 gates; the idiss consumer is native/sm86 raw-word exact and receives live SGS dissten over the full corrected-real source-initialized first step, public reflectivity is consumed exactly once, and a full-domain 21-array Rust restart/resume transition is exact; evolved SGS/general turbulence, efall=1, complete forecast restart, and coupled multi-step budgets remain red",
                required=False,
                remedy="preserve the first-step live-dissten cadence, implement evolved SGS/general turbulence, keep efall=0 unless public terminal velocities become available, and pass coupled water/energy/positivity plus complete restart trajectory gates",
            ),
            Finding(
                "CM1 Rust history/restart engine",
                MISSING,
                LEGACY_LANE_PREFIX
                + (
                    "the bundled cm1-state executable passes its installed hash/ABI gate and the 21-array WSM6 transition-local restart/resume is exact, but complete coupled live-state capture/restore remains open"
                    if native_tools is not None
                    else f"the bundled cm1-state executable is not admitted: {native_error}"
                ),
                required=False,
                remedy="bind every live CUDA, boundary, closure and physics carrier, then prove uninterrupted/resumed identity",
            ),
            Finding(
                "CM1 Rust renderer",
                OK if native_tools is not None else MISSING,
                (
                    "the bundled cm1-render executable passes its installed hash/ABI gate and directly decodes independently anchored CM1FHS1/CM1FCP1 envelopes"
                    if native_tools is not None
                    else f"the bundled cm1-render executable is not admitted: {native_error}"
                ),
                required=True,
                remedy=None
                if native_tools is not None
                else "reinstall a sealed platform wheel and pass the real CM1FHS1 installed render gate",
            ),
            _cartesian_inputs_finding(),
            *_unvalidated_findings(),
            _surface_layer_finding(),
            *_physics_family_findings(),
            *_output_selection_findings(),
            _cartesian_restart_finding(),
            _ensemble_finding(),
            *_lagrangian_findings(),
            _cupy_finding(),
        )
    )
    findings.extend(_boundary_validation_findings())
    return findings


def blocking(findings: list[Finding]) -> list[Finding]:
    return [item for item in findings if item.required and item.status == MISSING]


def unvalidated(findings: list[Finding]) -> list[Finding]:
    """Exposed, running, and not compared against CM1.  Never blocking."""

    return [item for item in findings if item.status == UNVALIDATED]


def render(findings: list[Finding], *, explain: bool = False) -> str:
    labels = {
        OK: "OK     ",
        PRESENT: "PRESENT",
        MISSING: "MISSING",
        INFO: "INFO   ",
        # Deliberately as wide and as loud as MISSING: a capability that is
        # reachable but unmeasured must not read as a quieter kind of fine.
        UNVALIDATED: "UNVALID",
    }
    lines: list[str] = []
    for item in findings:
        lines.append(f"{labels[item.status]} {item.subject}: {item.detail}")
        if explain and item.remedy:
            lines.append(f"        remedy: {item.remedy}")
    lines.append(f"blocking required findings: {len(blocking(findings))}")
    lines.append(
        "Cartesian lane runnable on this installation: "
        + ("no" if blocking(findings) else "yes")
    )
    # One line, not two: branches spelled this summary differently and the
    # union emitted both, reporting the same number twice.
    lines.append(f"unvalidated exposed capabilities: {len(unvalidated(findings))}")
    return "\n".join(lines)


__all__ = ["Finding", "UNVALIDATED", "blocking", "collect", "render", "unvalidated"]
__all__ = ["Finding", "blocking", "collect", "render", "unvalidated"]
__all__ = ["Finding", "UNVALIDATED", "blocking", "collect", "render", "unvalidated"]
