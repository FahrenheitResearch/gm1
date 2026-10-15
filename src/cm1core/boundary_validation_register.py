"""What this distribution's lateral boundaries are, and what was measured.

Every CM1 r21.1 lateral capability this port can select is listed here with the
check that was actually performed on it.  A capability whose ``differential``
entry is ``None`` is *exposed but not differentially validated*: it runs, it is
selectable from a document, and no CM1-vs-GM1 comparison has been executed for
it.  A capability with a differential and a ``not_checked`` sentence was
compared with CM1, and the comparison found a part that does not give CM1's
words; it is not checked either, and is published as such.
``unvalidated_for`` turns one run's selection into the machine-readable list a
receipt and ``gpuwm-cm1 doctor`` both publish, so a reader never has to infer
from a lane name which parts of a boundary were measured.

Silence is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


BOUNDARY_VALIDATION_SCHEMA = "gpuwm-cm1.boundary-validation-register/v1"

#: One entry per selectable capability.  ``checked`` is what was measured and
#: ``differential`` names the CM1-vs-GM1 comparison, or is ``None`` when none
#: has been run.  ``not_checked`` says, in one sentence, what a reader must not
#: assume; it is ``None`` only when nothing the capability runs is known to
#: differ from CM1 beyond the stated caveat.
_REGISTER: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "wbc/ebc/sbc/nbc=2 (open-radiative)": MappingProxyType(
            {
                "cm1": "src/bc.F:180-1143,2392-2684",
                "checked": "the port's sealed all-open operator, with its recorded sm86 full-state evidence; a CM1 differential to 600 s; the scalar advection and pdef edge cells (adv.F:214-340, pdef.F:49-166) against CM1's own compiled advs on every cell; and a 900 s CM1 differential with cold outflow leaving through the open edges, at 100 m on a stretched column with a translating frame, lspgrad=1 and Coriolis",
                "differential": "evidence/oracle/open-boundary-ring (edge-coldpool-100); evidence/oracle/acoustic-warm-bubble-32; evidence/oracle/boundary-families/acoustic-warm-bubble-600",
                "caveat": "the acoustic-warm-bubble evidence predates the scalar ring fix, and its east-edge scalar divergence was that defect; while a density current leaves through an open edge the edge-band pressure differs from CM1 by several times the interior level for a few minutes, and a 1e-4 K perturbation of GM1 against itself shows the same edge amplification, so that part is the radiative condition's own sensitivity",
                "not_checked": None,
            }
        ),
        "wbc/ebc/sbc/nbc=1 (periodic)": MappingProxyType(
            {
                "cm1": "src/bc.F:180-1143 periodic branches; src/sound.F:388 with param.F:6678 ibw=0",
                "checked": "raw-word against the src/bc.F transcription for every carrier; a CM1 differential at 140 s (1.5 acoustic transits) and 600 s (6.75 transits), which agrees with CM1 to within 2% of the all-open control's own disagreement on every field; advu/advv/advw's boundary faces (adv.F i1/i2, j1/j2, doitw..doitn) word for word against CM1's compiled routines on periodic and mixed families (the uniform text's open-side inflow edits differ by rounding, evidence/oracle/periodic-momentum-faces); and horizontally uniform periodic columns whose columns stay bit-identical on both vertical families (8x8x115 stretched and 12x12x115 uniform free-slip to 1800 s, 32x32x40 uniform to 600 s), once the uniform lane's acoustic step advances the periodic faces before it forms the divergence (sound.F:386-406 before :446-530); the flat (stretched) lane's small step word for word against CM1's compiled sound on periodic and mixed families (ssopenbc on open sides only, sound.F:274-302; param.F:9182-9219's dtu0/dtv0 pressure-gradient masks), and a bubble beside a periodic corner whose two boundary-face copies stay one word on both vertical families",
                "differential": "evidence/oracle/boundary-families/boundary-periodic-32, boundary-periodic-600, boundary-open-x-periodic-y-32; evidence/oracle/periodic-momentum-faces",
                "caveat": "the 32x32x40 differentials at dx=1000 m predate the momentum boundary-face fix (faces 1 and ni+1 got no advection there) and the uniform lane's acoustic face order (the divergence read the previous small step's boundary face); no LES-resolution or long-equilibration periodic run has been compared with CM1; the SGS closure still applies the open-side edge treatment on every family (turbtend.F:67-102 copies the boundary flux only for wbc=2), so on a periodic side the turbulent flux divergence of the edge columns and rows is dropped: a warm bubble moved by whole cells moves every step-1 held tendency except the w and theta words in the edge column or row beside it (evidence/oracle/periodic-momentum-faces, tests/test_cartesian_periodic_column_gpu.py); compared word for word with CM1's compiled turb.F, turbtend.F and adv.F on 24x20x20 periodic-in-x grids, the deformation, the stresses, the stress divergence, the scalar diffusion and the TKE advection all differ from CM1 in the whole edge column beside a periodic side (760 to 840 words a call, on both column families, with or without walls on the other sides; evidence/oracle/walls-operators)",
                "not_checked": "the turbulence scheme's edge column beside a periodic side: GM1's closure still applies the open-side edge treatment there, where CM1 does not (turbtend.F:67-102), and compared word for word with CM1's compiled turb.F, turbtend.F and adv.F on 24x20x20 periodic-in-x grids the deformation, the stresses, the stress divergence, the scalar diffusion and the TKE advection differ from CM1 in the whole edge column beside a periodic side, 760 to 840 words a call, on both column families, with or without walls on the other sides (evidence/oracle/walls-operators); no whole run has measured what this does to a storm",
            }
        ),
        "wbc/ebc/sbc/nbc=3,4 (rigid walls)": MappingProxyType(
            {
                "cm1": "src/bc.F:180-1143 rigid branches; src/turb.F calcdef and gettau wall branches; src/turbtend.F; src/adv.F; src/init3d.F:2147-2151",
                "checked": "every operator a walled domain runs, word for word against CM1 r21.1's compiled routines on 24x20x20 domains with free-slip, no-slip and mixed walls on even and stretched columns: the deformation, the stresses and the stress divergence (turb.F calcdef and gettau, turbtend.F turbu, turbv, turbw, turbt, turbs), the eddy coefficients, the TKE advection, the u, v, w and scalar advection, the positive-definite step, the lateral fills and the whole acoustic RK stage (evidence/oracle/walls-operators); the initial fill of the wall faces (init3d.F:2147-2151); and 15-minute dry warm bubbles in a 64 km box with a 10 m/s wind into the walls, against CM1 run on the CPU: with free-slip or no-slip walls in a stable layer GM1 stays within 4e-5 of each field's peak, and with free-slip, no-slip, or both on a stretched column in a neutral layer, where the thermal turns turbulent against the walls, within 3.4 times CM1's own spread from 1e-4 K added to its sounding, and within 2.3 times it in root-mean-square (evidence/oracle/walls-and-lateral-runs)",
                "differential": None,
                "caveat": None,
                "not_checked": "a storm in a walled box: in CM1's two-hour supercell with free-slip walls west and east and no-slip walls south and north, GM1 follows CM1 to within 2 cells for the first 70 minutes, but the mature-phase max 2-5 km updraft helicity is 1.26 times CM1's (band 10 %), the max-w column is 12 cells from CM1's at 75 minutes and farther later, and the other nine statistics are 0.964 to 1.110; CM1 against its 1e-4 K twin is 0.982 to 1.031 within 2 cells (evidence/oracle/walls-and-lateral-runs)",
            }
        ),
        "irbc=2 (Klemp-Wilhelmson on small steps) with cstar": MappingProxyType(
            {
                "cm1": "src/bc.F:1948-2053; src/sound.F:260-268",
                "checked": "raw-word radbcew/radbcns against the src/bc.F transcription, then a CM1 differential at 280 s (three acoustic transits) with cstar=100 m/s; the stretched column's acoustic stage, which was handed no lateral step before, word for word against CM1's compiled sound with irbc=2 on open, walled and periodic sides; and CM1's two-hour supercell on 53 stretched levels with irbc=2, whose ten mature-phase statistics are 0.963 to 1.010 of CM1's",
                "differential": "evidence/oracle/boundary-families/boundary-irbc2-32; evidence/oracle/walls-operators; evidence/oracle/walls-and-lateral-runs",
                "caveat": "the option's own effect on this case is at the two codes' baseline rounding disagreement, so the differential shows GM1's response is the same order and sign as CM1's rather than resolving it to a fraction",
                "not_checked": None,
            }
        ),
        "irbc=1 (Klemp-Wilhelmson on large steps) with cstar": MappingProxyType(
            {
                "cm1": "src/bc.F:1948-2053; src/solve1.F:402-409",
                "checked": "the same raw-word radbcew/radbcns kernels irbc=2 uses, called once per large step from the A-level winds, one 140 s GPU integration that completes, and the acoustic stage beside it word for word against CM1's compiled sound on even and stretched columns: the stage keeps irbc=1's phase speeds, where the stretched column's stage used to overwrite them with the Durran-Klemp ones (evidence/oracle/walls-operators)",
                "differential": None,
                "caveat": None,
                "not_checked": "no CM1-vs-GM1 run has been compared with irbc=1; the large-step call itself is checked against the transcription of bc.F, not against CM1's own run",
            }
        ),
        "roflux=1 (restrict outward mass flux)": MappingProxyType(
            {
                "cm1": "src/bc.F:2180-2391; src/sound.F:283-302",
                "checked": "raw-word against the src/bc.F transcription including the double-precision per-level reduction and the t3 ratio, then a CM1 differential at 280 s and 600 s; the stretched column's acoustic stage, which was handed no lateral step before, word for word against CM1's compiled sound with roflux=1 on open, walled and periodic sides, the rescale in CM1's own order (bc.F:2254, rho0*u3d rounded to binary32 first); and CM1's two-hour supercell on 53 stretched levels with roflux=1, whose ten mature-phase statistics are 0.955 to 1.011 of CM1's",
                "differential": "evidence/oracle/boundary-families/boundary-roflux-32, boundary-roflux-600; evidence/oracle/walls-operators; evidence/oracle/walls-and-lateral-runs",
                "caveat": "the rescale is a near-identity on this case (CM1's own response is 2e-5 of the field), so the differential shows the option is live and does not disturb the agreement; the switch-response residual is dominated by amplified rounding at the boundary face; on an even column the acoustic stage forms rho0*u3d in double precision where CM1 rounds it to binary32 first, which agrees with CM1 to rounding (with the lateral settings, at most 2.2e-3 of a field's peak over one stage) and not word for word; in a storm the rescale moves no statistic beyond CM1's own spread, so the storm alone cannot tell it from a no-op",
                "not_checked": None,
            }
        ),
        "nudgeobc=1 with alphobc": MappingProxyType(
            {
                "cm1": "src/solve2.F:726-822",
                "checked": "raw-word tendency against the src/solve2.F transcription, then a CM1 differential at 280 s = 4.7 alphobc e-foldings, where GM1's response is 1.2x CM1's",
                "differential": "evidence/oracle/boundary-families/boundary-nudgeobc-32",
                "caveat": "the term is summed into the held tendency once the stage's uten and vten are assembled rather than inside CM1's U- and V-equation blocks: the same sum in exact arithmetic, not the same rounding",
                "not_checked": None,
            }
        ),
        "irdamp=2 (damp toward the horizontal average)": MappingProxyType(
            {
                "cm1": "src/solve1.F:427-438; src/testcase_simple_phys.F:307-375",
                "checked": "raw-word against the transcription of the three replaced operands, with the held assignment left as the unchanged already evidenced kernel, then a CM1 differential at 280 s",
                "differential": "evidence/oracle/boundary-families/boundary-irdamp2-32",
                "caveat": "the level mean is a device tree reduction, not CM1's serial i-then-j double accumulation, so it is accurate rather than raw-word identical",
                "not_checked": None,
            }
        ),
        "hrdamp=1,2 with xhd (lateral Rayleigh sponge)": MappingProxyType(
            {
                "cm1": "src/param.F:8687-8802; src/solve1.F:404-438",
                "checked": "the four-side taper built on the device from src/param.F:8736-8776, agreeing with the transcription to 2e-6, entering the already evidenced held assignment through tauf, which acts on w only; then a CM1 differential at 280 s",
                "differential": "evidence/oracle/boundary-families/boundary-hrdamp-32",
                "caveat": "device cosf is not bit-identical to the host libm cos CM1 links, so the coefficient agrees to rounding and not to the word",
                "not_checked": None,
            }
        ),
    }
)


def register() -> Mapping[str, Mapping[str, Any]]:
    return _REGISTER


def _selected(boundaries: Any, closures: Any) -> list[str]:
    """Name every register entry this configuration actually turns on."""

    lateral = (boundaries.west, boundaries.east, boundaries.south, boundaries.north)
    names: list[str] = []
    if 2 in lateral:
        names.append("wbc/ebc/sbc/nbc=2 (open-radiative)")
    if 1 in lateral:
        names.append("wbc/ebc/sbc/nbc=1 (periodic)")
    if 3 in lateral or 4 in lateral:
        names.append("wbc/ebc/sbc/nbc=3,4 (rigid walls)")
    if 2 in lateral and boundaries.radiative_order == 2:
        names.append("irbc=2 (Klemp-Wilhelmson on small steps) with cstar")
    if 2 in lateral and boundaries.radiative_order == 1:
        names.append("irbc=1 (Klemp-Wilhelmson on large steps) with cstar")
    # CM1 rescales the outward flow through open sides only (bc.F:2180-2391).
    if 2 in lateral and boundaries.roflux == 1:
        names.append("roflux=1 (restrict outward mass flux)")
    if boundaries.nudgeobc == 1:
        names.append("nudgeobc=1 with alphobc")
    if closures.upper_damping == 2:
        names.append("irdamp=2 (damp toward the horizontal average)")
    if closures.hrdamp >= 1:
        names.append("hrdamp=1,2 with xhd (lateral Rayleigh sponge)")
    return names


def is_unvalidated(name: str) -> bool:
    """True when part of the capability has not been checked against CM1.

    That is a capability with no CM1-vs-GM1 differential, and one whose
    differential found a part that does not give CM1's words (``not_checked``
    stated beside a differential).
    """

    entry = _REGISTER[name]
    return entry["differential"] is None or entry["not_checked"] is not None


def unvalidated_for(boundaries: Any, closures: Any) -> dict[str, Any]:
    """The receipt block for one run's lateral boundaries.

    ``capabilities`` is every selectable boundary capability this run turned
    on; ``unvalidated`` is the subset with a part not checked against CM1 (no
    CM1-vs-GM1 differential, or one that found a part differing from CM1),
    each with the sentence naming what was not checked, and the differential
    and its caveat when there is one; ``validated`` is the rest.
    """

    selected = _selected(boundaries, closures)
    unvalidated = {
        name: {
            "cm1": _REGISTER[name]["cm1"],
            "checked": _REGISTER[name]["checked"],
            "not_checked": _REGISTER[name]["not_checked"],
            **(
                {"differential": _REGISTER[name]["differential"], "caveat": _REGISTER[name]["caveat"]}
                if _REGISTER[name]["differential"] is not None
                else {}
            ),
        }
        for name in selected
        if is_unvalidated(name)
    }
    return {
        "schema": BOUNDARY_VALIDATION_SCHEMA,
        "capabilities": selected,
        "validated": {
            name: {
                "cm1": _REGISTER[name]["cm1"],
                "differential": _REGISTER[name]["differential"],
                # What the differential does not settle, stated beside it.
                "caveat": _REGISTER[name]["caveat"],
            }
            for name in selected
            if not is_unvalidated(name)
        },
        "unvalidated": unvalidated,
        "unvalidated_count": len(unvalidated),
    }


__all__ = [
    "BOUNDARY_VALIDATION_SCHEMA",
    "is_unvalidated",
    "register",
    "unvalidated_for",
]
