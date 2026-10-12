"""Plain words for the refusals the importer meets most often.

Every refusal carries its reason in CM1's words (the key, the value, the CM1
source line, the wrong result it prevents).  Guided mode and every tooltip
lead with plain words instead, with CM1's name as a grey hint:
short sentences, "your" not "the device", what happened and what to do, one
line each, no marketing words and no product names beyond GM1 and CM1.

:func:`describe` returns, for one refused (or not-read) row:

``cm1``        a one-line statement in CM1's words that replaces a long
               reason (the long reason is kept as the row's detail), or None
               to keep the reason as it is;
``fix``        the fix in CM1's words, or None to keep the computed one;
``plain``      what happened, in plain words;
``plain_fix``  what to do, in plain words.

The tables cover every refusal met across CM1's 23 shipped decks and a corpus
of 21 real namelists (36 settings, 30 codes and 5 code-and-setting pairs: the
30 or so most frequent and the rest of what those namelists meet); anything else gets a plain sentence for its
category.
Nothing here imports anything beyond the standard library and cm1core's own
pure-Python importer (for the ptype table).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class RefusalText:
    cm1: str | None
    fix: str | None
    plain: str
    plain_fix: str | None


Context = Mapping[str, Any]
Entry = Callable[[Any, Context], "RefusalText | None"]

_NSSL = {26, 27, 28, 29}
_P3 = {50, 51, 52, 53, 60, 61, 62, 63}

#: CM1 r21.1's iinit arms (README.namelist:679-690).
IINIT_NAMES: Mapping[int, str] = {
    0: "no perturbation", 1: "a warm bubble", 2: "a cold pool", 3: "a line of warm bubbles",
    4: "the Bryan-Fritsch moist benchmark bubble", 5: "a cold blob", 7: "a tropical cyclone vortex",
    8: "a line thermal with random noise", 9: "forced convergence", 10: "momentum forcing",
    11: "the Skamarock-Klemp gravity-wave test", 12: "updraft nudging", 13: "a near-surface theta wave",
}

#: CM1 r21.1's isnd options (README.namelist:612-651).
ISND_NAMES: Mapping[int, str] = {
    1: "dry adiabatic", 2: "dry isothermal", 3: "dry, constant lapse rate",
    4: "saturated neutral, the Bryan-Fritsch moist benchmark", 5: "Weisman-Klemp",
    6: "removed from CM1", 7: "the input_sounding file",
    8: "dry, constant d(theta)/dz", 9: "dry, constant stability", 10: "saturated, constant stability",
    11: "saturated, constant equivalent potential temperature", 12: "dry, adiabatic near the surface",
    13: "dry, three stability layers", 14: "convective boundary-layer test", 15: "stratocumulus (DYCOMS-II) test",
    17: "the input_sounding file's thermodynamics", 18: "sheared boundary-layer test",
    19: "shallow cumulus (BOMEX) test", 20: "precipitating shallow cumulus (RICO) test",
    21: "radiative-convective equilibrium (RCEMIP)", 22: "stable boundary-layer test",
    23: "shallow cumulus over land test",
}

#: CM1 r21.1's psolver options (README.namelist:404-456).
PSOLVER_NAMES: Mapping[int, str] = {
    1: "fully explicit compressible", 2: "time-split, explicit", 4: "anelastic", 5: "incompressible",
    6: "compressible-Boussinesq", 7: "modified compressible",
}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ptype(value: Any, context: Context) -> RefusalText:
    from ..cm1import.translate import PTYPE_SCHEMES

    ptype = _int(value)
    scheme = PTYPE_SCHEMES.get(ptype if ptype is not None else -1)
    if ptype in _NSSL:
        plain = "NSSL microphysics is not available in GM1."
    elif ptype in _P3:
        plain = "P3 microphysics is not available in GM1."
    elif ptype == 0:
        plain = "GM1 needs a rain and hail scheme for a moist run; this one has none."
    else:
        plain = "This rain and hail scheme is not available in GM1."
    return RefusalText(
        cm1=(f"ptype = {value} is {scheme}, which GM1 does not have." if scheme else
             f"ptype = {value} is not a CM1 r21.1 microphysics scheme (README.namelist:458-495)."),
        fix="Use ptype = 5 (Morrison) or ptype = 7 (WSM6).",
        plain=plain,
        plain_fix="Pick Morrison or WSM6 for rain and hail.",
    )


def _iinit(value: Any, context: Context) -> RefusalText:
    iinit = _int(value)
    if iinit == 4:
        # Refused only beside a word the arm never reads (config.py:
        # CM1-CFG-INIT-WORD-UNREAD, maintain_rh).
        return RefusalText(
            cm1="iinit = 4 (the Bryan-Fritsch moist benchmark bubble) sets its own humidity "
                "(init3d.F:579-647); CM1 never applies maintain_rh to it.",
            fix="Leave maintain_rh off with iinit = 4.",
            plain="The moist benchmark bubble sets its own humidity; the keep-humidity switch does not apply.",
            plain_fix="Turn the keep-humidity switch off.",
        )
    return RefusalText(
        cm1=f"iinit = {value} is not one of CM1 r21.1's ways of starting a run; GM1 has every one of them: "
            "iinit = 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 and 13 (init3d.F:450-1416), 4 to show only "
            "(iconly = 1).",
        fix="Use iinit = 1 (warm bubble); its centre, size and amplitude are on Settings.",
        plain="This is not one of CM1's ways of starting the storm.",
        plain_fix="Start the storm with a warm bubble instead.",
    )


def _isnd(value: Any, context: Context) -> RefusalText:
    isnd = _int(value)
    name = ISND_NAMES.get(isnd if isnd is not None else -1, f"option {value}")
    if isnd in (2, 3):
        return RefusalText(
            cm1=f"isnd = {value} ({name}) is a dry profile: CM1 stops unless imoist = 0 (param.F:1501-1515), "
                "and GM1 runs a dry state only to show the initial condition (run.iconly = 1).",
            fix="Set imoist = 0 and iconly = 1 to look at it, or use a moist profile such as isnd = 5 to run.",
            plain=f"This built-in profile ({name}) has no water vapour, so it can be shown but not run as a storm.",
            plain_fix="Use a moist profile to run a storm.",
        )
    if isnd == 15:
        return RefusalText(
            cm1="isnd = 15 (the DYCOMS-II stratocumulus test) runs only with testcase = 4 or 5, which change "
                "CM1's gas constants and latent heat for the whole model (constants.F:78-98); GM1 is built with "
                "CM1's default constants.",
            fix="Write the DYCOMS-II profile to input_sounding and use isnd = 7 with testcase = 0.",
            plain="This test profile needs different physical constants than GM1 is built with.",
            plain_fix="Give the profile as an input_sounding file instead.",
        )
    return RefusalText(
        cm1=f"isnd = {value} ({name}) is not a sounding CM1 r21.1 can build"
            + (" (base.F:455-460 stops CM1: 'isnd = 6 code has been removed')" if isnd == 6 else "")
            + "; CM1's soundings are isnd = 1-5, 7-15 and 17-23, and GM1 builds all of them but 15 "
            "(2 and 3 only with imoist = 0); GM1 runs isnd = 5, 7, 9 and 17 and shows the others only, with "
            "iconly = 1.",
        fix="Use isnd = 5, or write the profile to input_sounding and use isnd = 7.",
        plain=f"This built-in weather profile ({name}) is not available.",
        plain_fix="Use the standard storm profile, or give the profile as an input_sounding file.",
    )


def _testcase(value: Any, context: Context) -> RefusalText:
    if _int(value) not in (4, 5):
        return RefusalText(
            cm1=f"testcase = {value} is not one of CM1 r21.1's test cases (0-15, README.namelist:216-232).",
            fix="Use testcase = 0.",
            plain="This is not one of CM1's test cases.",
            plain_fix="Use the standard test case setting.",
        )
    return RefusalText(
        cm1=f"testcase = {value} changes CM1's gas constants and latent heat for the whole model "
            "(constants.F:78-98, 161-164); GM1's dynamics and physics are built with CM1's default constants.",
        fix="Use testcase = 0.",
        plain="This test case needs different physical constants than GM1 is built with.",
        plain_fix="Use the standard test case setting.",
    )


def _psolver(value: Any, context: Context) -> RefusalText:
    psolver = _int(value)
    name = PSOLVER_NAMES.get(psolver or -1, f"option {value}")
    return RefusalText(
        cm1=f"psolver = {value} ({name}) is not available; GM1 runs CM1's time-split, vertically "
            "implicit solver, psolver = 3.",
        fix="Set psolver = 3.",
        plain="This pressure solver is not available in GM1.",
        plain_fix="Use the standard pressure solver.",
    )


def _stretch_z(value: Any, context: Context) -> RefusalText:
    level = _int(value)
    if level == 2:
        cm1 = ("stretch_z = 2 is not available: CM1's own generator leaves its top face and ztop at "
               "two different heights (param.F:9897-9937), and a GM1 grid has one model top.")
    elif level in (3, 4):
        cm1 = (f"stretch_z = {value} reads the vertical levels from the input_grid_z file "
               "(param.F:8013, 8083), which the importer does not read yet.")
    else:
        cm1 = None
    return RefusalText(
        cm1=cm1,
        fix="Use stretch_z = 1 (CM1's analytic stretching), or 0 for even levels.",
        plain="This way of stretching the vertical grid is not available in GM1.",
        plain_fix="Use even levels or the standard stretching.",
    )


def _stretch(axis: str) -> Entry:
    def entry(value: Any, context: Context) -> RefusalText:
        spacing = context.get("dx" if axis == "x" else "dy")
        at = ""
        if isinstance(spacing, (int, float)) and abs(float(spacing) - 100.0) < 1e-6:
            at = " At dx = 100 m a 16 GB card fits about 60 x 60 km x 80 levels."
        return RefusalText(
            cm1=f"GM1 has no horizontal grid stretching in v0.1 (stretch_{axis} = {value}).",
            fix=f"Use a uniform grid (stretch_{axis} = 0).{at}",
            plain="Stretched horizontal grids are not available yet.",
            plain_fix="Use the same grid spacing everywhere.",
        )
    return entry


def _fixed(cm1: str | None, fix: str | None, plain: str, plain_fix: str | None) -> Entry:
    def entry(value: Any, context: Context) -> RefusalText:
        return RefusalText(
            cm1=None if cm1 is None else cm1.format(value=value),
            fix=None if fix is None else fix.format(value=value),
            plain=plain.format(value=value),
            plain_fix=None if plain_fix is None else plain_fix.format(value=value),
        )
    return entry


#: By CM1 key: refusals of that setting whatever the code.
BY_KEY: Mapping[str, Entry] = {
    "ptype": _ptype,
    "iinit": _iinit,
    "isnd": _isnd,
    "testcase": _testcase,
    "psolver": _psolver,
    "stretch_x": _stretch("x"),
    "stretch_z": _stretch_z,
    "stretch_y": _stretch("y"),
    "idiss": _fixed(
        "GM1 always includes dissipative heating (idiss = 1); idiss = {value} turns it off in CM1.",
        "Set idiss = 1.",
        "GM1 always includes the heating made by friction; this setup turns it off.",
        "Leave that heating on.",
    ),
    "advwenov": _fixed(
        "WENO advection of the winds (advwenov = {value}) is not available; GM1 runs CM1's "
        "fifth-order advection for the winds.",
        "Set advwenov = 0.",
        "This way of moving the winds around the grid (WENO) is not available in GM1.",
        "Use the standard wind advection.",
    ),
    "weno_order": _fixed(
        "weno_order = {value}: GM1 runs fifth-order WENO only (weno_order = 5); with WENO on, CM1 r21.1 "
        "itself stops on any value but 3, 5, 7 and 9 (param.F:1257-1275).",
        "Set weno_order = 5.",
        "This WENO order is not available in GM1.",
        "Use fifth order.",
    ),
    "advwenos": _fixed(
        "GM1 advects scalars with WENO on the final stage only (advwenos = 2).",
        "Set advwenos = 2.",
        "This scalar advection choice is not available in GM1.",
        "Use the standard setting.",
    ),
    "tconfig": _fixed(
        "GM1 runs tconfig = 1 (one eddy viscosity in every direction).",
        "Set tconfig = 1.",
        "This turbulence setting is not available in GM1.",
        "Use the standard turbulence setting.",
    ),
    "horizturb": _fixed(
        "A separate horizontal turbulence length scale (horizturb = 1) is not available in GM1.",
        "Set horizturb = 0.",
        "This turbulence setting is not available in GM1.",
        "Use the standard turbulence setting.",
    ),
    "set_flx": _fixed(
        "Prescribed surface fluxes (set_flx = 1) are not available in GM1.",
        "Set set_flx = 0.",
        "Fixed surface heat and moisture fluxes are not available in GM1.",
        "Let the surface fluxes follow the model.",
    ),
    "iptra": _fixed(
        "Passive tracers (iptra = 1) are not available in GM1.",
        "Set iptra = 0.",
        "Passive tracers are not available in GM1.",
        "Turn the tracers off.",
    ),
    "apmasscon": _fixed(
        "GM1 always applies the dry-air mass correction (apmasscon = 1).",
        "Set apmasscon = 1.",
        "GM1 always keeps the total dry-air mass fixed; this setup turns that off.",
        "Leave the mass correction on.",
    ),
    "lspgrad": _fixed(
        "lspgrad = {value} is not available; GM1 has lspgrad = 0 and 1.",
        "Set lspgrad = 0, or 1 for the base-state geostrophic balance.",
        "This large-scale pressure-gradient option is not available in GM1.",
        "Use the standard setting.",
    ),
    "sfcmodel": _fixed(
        "The surface layer sfcmodel = {value} is not available; GM1 has sfcmodel = 0 and 1.",
        "Set sfcmodel = 1 (CM1's own surface layer), or 0.",
        "This surface-layer scheme is not available in GM1.",
        "Use the standard surface layer.",
    ),
    "radopt": _fixed(
        "Radiation (radopt = {value}) is not coupled to GM1's GPU step yet.",
        "Set radopt = 0.",
        "Sunlight and infrared heating (radiation) are not available yet.",
        "Run without radiation.",
    ),
    "ipbl": _fixed(
        None,
        None,
        "A boundary-layer scheme is not available in GM1.",
        "Run without it.",
    ),
    "cuparam": _fixed(
        "Cumulus parameterization (cuparam = {value}) is not coupled to GM1's GPU step.",
        "Set cuparam = 0.",
        "Cumulus parameterization is not available in GM1.",
        "Run without it.",
    ),
    "terrain_flag": _fixed(
        "Terrain is not available in GM1 v0.1.",
        "Set terrain_flag = .false.",
        "Terrain is not available yet.",
        "Use flat ground.",
    ),
    "ndcnst": _fixed(
        "GM1's Morrison uses a fixed cloud-droplet number of 250 per cubic centimetre (ndcnst = 250).",
        "Set ndcnst = 250.0.",
        "This cloud-droplet number is not available with the Morrison scheme in GM1.",
        "Use the standard droplet number.",
    ),
    "adapt_dt": _fixed(
        None,
        None,
        "An adjustable time step is not available with this setup.",
        "Use a fixed time step.",
    ),
    "iorigin": _fixed(
        None,
        "Set iorigin = 1 (domain corner) or 2 (domain centre).",
        "Say where the grid's origin is.",
        "Pick the corner or the centre.",
    ),
    "eqtset": _fixed(
        "GM1 runs CM1's eqtset = 2 equation set only.",
        "Set eqtset = 2.",
        "This equation set is not available in GM1.",
        "Use the standard equation set.",
    ),
    "doimpl": _fixed(
        "GM1 always runs the vertically implicit turbulence diffusion (doimpl = 1).",
        "Set doimpl = 1.",
        "This turbulence setting is not available in GM1.",
        "Use the standard turbulence setting.",
    ),
    "cm1setup": _fixed(
        None,
        None,
        "GM1 runs CM1's cloud-scale setup only (large-eddy turbulence everywhere).",
        "Use the standard setup.",
    ),
    "tbc": _fixed(
        None,
        "Set tbc = 1 (free slip at the model top).",
        "This model-top boundary is not available in GM1.",
        "Use the standard free-slip top.",
    ),
    "sgsmodel": _fixed(
        None,
        None,
        "This turbulence scheme setting is not available in GM1.",
        "Use the standard turbulence scheme.",
    ),
    "bbc": _fixed(
        None,
        None,
        "This ground boundary setting is not available in GM1.",
        "Use free slip, semi-slip or no slip.",
    ),
    "zd": _fixed(
        None,
        None,
        "The damping layer at the model top does not fit this grid.",
        "Start the damping layer below the model top.",
    ),
    "t2p_avg": _fixed(
        None,
        None,
        "This near-ground turbulence averaging is not available in GM1.",
        "Use the standard averaging.",
    ),
    "imoist": _fixed(
        "GM1 runs moist storms only (imoist = 1); imoist = 0 runs no water at all.",
        "Set imoist = 1 and pick ptype = 5 or 7.",
        "GM1 runs storms with moisture only.",
        "Turn moisture on.",
    ),
    "output_pdcomp": _fixed(
        None,
        None,
        "The pressure decomposition output is not available with this setup.",
        "Turn that output off.",
    ),
    "input_sounding": _fixed(
        None,
        None,
        "The weather profile file (input_sounding) has a problem.",
        None,
    ),
}

#: By refusal code: whatever the key.
BY_CODE: Mapping[str, Entry] = {
    "CM1-CFG-BASE-CLOUD-INTEGRATION": lambda value, context: RefusalText(
        cm1=f"isnd = {value} ({ISND_NAMES.get(_int(value) or -1, 'this sounding')}) starts with cloud in the base "
            "state (base.F qc0/qi0). GM1 builds that starting state exactly as CM1 does, but its run from it does "
            "not yet match CM1's: in a 15-minute comparison the updrafts and temperatures differed from the first "
            "minute. GM1 shows this sounding only (iconly = 1).",
        fix="Set iconly = 1 to look at the starting state, or use isnd = 5 or 9, or isnd = 7 with an input_sounding "
            "file.",
        plain="This built-in profile starts with cloud everywhere. GM1 can show how it starts, but cannot run it "
              "yet: its runs from it did not match CM1's.",
        plain_fix="Show the starting state only, or use the standard storm profile.",
    ),
    "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION": lambda value, context: RefusalText(
        cm1=f"isnd = {value} ({ISND_NAMES.get(_int(value) or -1, 'this sounding')}) is one of CM1's built-in "
            "profiles. GM1 builds its starting state exactly as CM1 does, but "
            + ("its run from it does not yet match CM1's: in a 15-minute comparison the updrafts and temperatures "
               "differed from the first minute, with no cloud in the starting state."
               if _int(value) == 11 else
               "no run from it has been compared with CM1's yet. Of CM1's built-in profiles, GM1 runs only those "
               "whose runs matched CM1's (isnd = 5 and 9).")
            + " GM1 shows this sounding only (iconly = 1).",
        fix="Set iconly = 1 to look at the starting state, or use isnd = 5 or 9, or isnd = 7 with an input_sounding "
            "file.",
        plain="GM1 can show how this built-in profile starts, but cannot run it yet: "
              + ("its runs from it did not match CM1's." if _int(value) == 11 else
                 "its runs from it have not been checked against CM1's."),
        plain_fix="Show the starting state only, or use the standard storm profile.",
    ),
    "CM1-CFG-MOIST-BENCHMARK-INTEGRATION": _fixed(
        "iinit = 4 (the Bryan-Fritsch moist benchmark bubble) is built exactly as CM1 builds it, but GM1's run from "
        "it does not yet match CM1's: in a 15-minute comparison the updrafts and temperatures differed from the "
        "first minute. GM1 shows this start only (iconly = 1).",
        "Set iconly = 1 to look at the starting state, or use iinit = 1 (warm bubble).",
        "GM1 can show how the moist benchmark bubble starts, but cannot run it yet: its runs from it did not match "
        "CM1's.",
        "Show the starting state only, or start the storm with a warm bubble.",
    ),
    "CM1-IMPORT-SOUNDING-MISSING": _fixed(
        None, None,
        "This setup reads its starting weather profile from a file, and none was given.",
        "Add the input_sounding file next to the namelist, or choose it on the Sounding screen.",
    ),
    "CM1-IMPORT-SOUNDING-TOP": _fixed(
        None, None,
        "The weather profile stops below the top of the model.",
        "Extend the profile higher, or make the model shallower.",
    ),
    "CM1-IMPORT-SOUNDING-FORMAT": _fixed(
        None, None,
        "The weather profile file is not in CM1's input_sounding layout.",
        "Write it as CM1 expects: a surface line, then one line per height.",
    ),
    "CM1-IMPORT-SOUNDING-HEIGHTS": _fixed(
        None, None,
        "The heights in the weather profile must rise from the ground up.",
        "List each height once, lowest first.",
    ),
    "CM1-IMPORT-SOUNDING-LEVELS": _fixed(
        None, None,
        "The weather profile needs at least two heights above the surface.",
        "Add more heights to the profile.",
    ),
    "CM1-IMPORT-SOUNDING-UNREADABLE": _fixed(
        None, None,
        "The weather profile file cannot be read.",
        "Check the file and its permissions.",
    ),
    "CM1-IMPORT-STARTUP-GROUPS": _fixed(
        None, None,
        "The namelist is missing a section CM1 needs to start, or has them out of order.",
        "Add the missing section, in CM1's usual order.",
    ),
    "CM1-IMPORT-UNREAD-PHYSICS": _fixed(
        None, None,
        "GM1 runs this physics setting only at CM1's usual value.",
        "Use the value the fix names.",
    ),
    "CM1-IMPORT-UNKNOWN-KEY": _fixed(
        None, None,
        "This setting is not one CM1 r21.1 knows, and it sits among settings that change the storm.",
        "Remove it, or rename it to the CM1 r21.1 setting it stands for.",
    ),
    "CM1-IMPORT-R22-PHYSICS": _fixed(
        None, None,
        "This setting comes from CM1 r22.0 and changes the storm in a way CM1 r21.1 cannot.",
        "Set it to its CM1 r21.1 value, or remove it.",
    ),
    "CM1-IMPORT-OLD-KEY": _fixed(
        None, None,
        "This setting comes from an older CM1 and has no CM1 r21.1 equivalent.",
        "Remove it, or set the CM1 r21.1 setting that replaced it.",
    ),
    "CM1-IMPORT-TRANSLATION": _fixed(
        None, None,
        "GM1 cannot turn this setting into a run.",
        "Change it as the fix says.",
    ),
    "CM1-CFG-ADAPTIVE-TRANSLATING-FRAME": _fixed(
        "GM1 cannot yet combine an adaptive time step (adapt_dt = 1) with a moving domain (imove = 1).",
        "Set adapt_dt = 0 with a fixed dtl, or imove = 0.",
        "An adjustable time step with a moving domain is not available yet.",
        "Use a fixed time step, or keep the domain still.",
    ),
    "CM1-CFG-CLOCK-ALIGNMENT": _fixed(
        None, None,
        "An output or restart time does not land on a whole number of time steps.",
        "Make the output and restart intervals whole multiples of the time step.",
    ),
    "CM1-CFG-EVENT-OUTSIDE-RUN": _fixed(
        None, None,
        "An output or restart interval is longer than the run itself.",
        "Shorten the interval, or run longer.",
    ),
    "CM1-CFG-DIAGNOSTIC-SELECTION": _fixed(
        None, None,
        "This output cannot be worked out correctly on this grid.",
        "Turn that output off, or change the grid it needs.",
    ),
    "CM1-CFG-PBL-WITHOUT-MESOSCALE-SETUP": _fixed(
        None, None,
        "A boundary-layer scheme needs a setup GM1 does not have.",
        "Run without the boundary-layer scheme.",
    ),
    "CM1-OUTPUT-DRY-LANE": _fixed(
        None, None,
        "This rain or cloud output needs moisture, and the run is dry.",
        "Turn the output off, or turn moisture on.",
    ),
    "CM1-OUTPUT-BUDGET": _fixed(
        None, None,
        "This budget output is not available with this rain and hail scheme.",
        "Turn the budget output off.",
    ),
    "CM1-CFG-VERTICAL-FORNBERG": _fixed(
        None, None,
        "This stretched vertical grid is not supported: its bottom or top layers change depth too fast.",
        "Use a uniform vertical grid, or stretch it more gently.",
    ),
    "CM1-CFG-INTEGER": _fixed(
        None, None,
        "This setting needs a whole number in range.",
        "Use a whole number in range.",
    ),
    "CM1-CFG-MOVING-FRAME-INACTIVE": _fixed(
        None, None,
        "A moving-domain speed is set, but the moving domain is off.",
        "Set the speeds to zero, or turn the moving domain on.",
    ),
    "CM1-PHYSICS-COMPONENT-UNCOUPLED": _fixed(
        None, None,
        "This physics scheme is not coupled to GM1's GPU step yet.",
        "Turn it off.",
    ),
    "CM1-PHYSICS-COMPONENT-UNPORTED": _fixed(
        None, None,
        "This physics scheme is not available in GM1.",
        "Turn it off.",
    ),
    "CM1-CFG-SFCMODEL-UNSUPPORTED": _fixed(
        None, None,
        "This surface-layer scheme is not available in GM1.",
        "Use the standard surface layer.",
    ),
    "CM1-CARTESIAN-HISTORY-PAYLOAD": _fixed(
        "This grid is too large for one saved frame (nx x ny x nz).",
        "Reduce nx, ny or nz.",
        "This grid is too large to save a single frame of it.",
        "Use a smaller grid.",
    ),
    "CM1-CARTESIAN-LEVEL-CAPACITY": _fixed(
        None,
        None,
        "The grid has more levels than this rain and hail scheme is built for.",
        "Use fewer levels.",
    ),
    "CM1-CARTESIAN-RUN-GEOMETRY": _fixed(
        None, None,
        "The storm would drift too close to the side of the area before the run ends.",
        "Make the area larger, start the storm elsewhere, or move the area with the storm.",
    ),
    "CM1-CARTESIAN-RUN-GEOMETRY-UNREADABLE": _fixed(
        None, None,
        "The storm's track cannot be worked out from this weather profile.",
        "Give a profile that reaches at least 6 km.",
    ),
    "CM1-IMPORT-CECD-UNSUPPORTED": _fixed(
        "Only cecd = 1 (constant drag coefficient) is implemented; cecd = {value} is not.",
        "Set cecd = 1.",
        "Only a constant ground-drag coefficient is available in this version.",
        "Use the constant drag coefficient.",
    ),
    "CM1-CFG-CLOSURE-HAS-NO-OFF-SWITCH": _fixed(
        None, None,
        "GM1 always runs its large-eddy turbulence scheme; this setup turns it off.",
        "Use the standard setup.",
    ),
}

#: Codes many settings share: a key's own words come before these.
GENERIC_CODES = frozenset({
    "CM1-CFG-UNSUPPORTED-VALUE", "CM1-CFG-INTEGER", "CM1-IMPORT-UNREAD-PHYSICS", "CM1-IMPORT-TRANSLATION",
    "CM1-PHYSICS-COMPONENT-UNCOUPLED", "CM1-PHYSICS-COMPONENT-UNPORTED", "CM1-IMPORT-UNKNOWN-KEY",
    "CM1-IMPORT-R22-PHYSICS", "CM1-IMPORT-OLD-KEY",
})

#: A refused setting the tables above do not name, and the other categories.
GENERIC: Mapping[str, RefusalText] = {
    "refused": RefusalText(None, None, "GM1 cannot run this setting as CM1 does.", "Change it as the fix says."),
    "not_read": RefusalText(
        None, None,
        "GM1 does not read this output setting; the run goes ahead without it once you agree.",
        "Tick 'Run without these settings', or remove it.",
    ),
}

#: Specific (code, key) pairs, checked first.
BY_CODE_AND_KEY: Mapping[tuple[str, str], Entry] = {
    ("CM1-CFG-INTEGER", "nx"): _fixed(
        "GM1 runs three-dimensional domains only: nx must be at least 7.",
        "Set nx to 7 or more.",
        "Two-dimensional runs are not available; the grid needs at least 7 points each way.",
        "Make the grid at least 7 points wide.",
    ),
    ("CM1-CFG-INTEGER", "ny"): _fixed(
        "GM1 runs three-dimensional domains only: ny must be at least 7.",
        "Set ny to 7 or more.",
        "Two-dimensional runs are not available; the grid needs at least 7 points each way.",
        "Make the grid at least 7 points wide.",
    ),
    ("CM1-CFG-INTEGER", "nz"): _fixed(
        "GM1 needs at least 7 vertical levels (nz).",
        "Set nz to 7 or more.",
        "The grid needs at least 7 levels.",
        "Add levels.",
    ),
    # The diagnostics module's selection check (validate_*_diagnostic_selection)
    # names the words it refuses; the importer reads them off the reason
    # (coverage._keys_from_reason).
    ("CM1-CFG-DIAGNOSTIC-SELECTION", "output_uh"): _fixed(
        None, None,
        "Updraft helicity needs a model at least 5 km deep.",
        "Turn updraft helicity off, or make the model deeper.",
    ),
    ("CM1-CFG-DIAGNOSTIC-SELECTION", "output_sus"): _fixed(
        None, None,
        "The updraft swath at 5 km needs a model at least 5 km deep.",
        "Turn the updraft swath off, or make the model deeper.",
    ),
}


def describe(
    *,
    code: str | None,
    keys: tuple[str, ...],
    value: Any,
    context: Context,
    category: str,
) -> RefusalText | None:
    """Plain (and curated CM1) words for one row, or None to keep the row as it is."""

    if category not in ("refused", "not_read"):
        return None
    key = keys[0] if keys else ""
    specific = code is not None and code in BY_CODE and code not in GENERIC_CODES
    for lookup in (
        (BY_CODE_AND_KEY.get((code or "", key)) if code else None),
        BY_CODE.get(code or "") if specific else None,
        BY_KEY.get(key),
        BY_CODE.get(code or ""),
    ):
        if lookup is not None:
            found = lookup(value, context)
            if found is not None:
                return found
    return GENERIC.get(category)


def covered() -> dict[str, list[str]]:
    """The codes and keys with their own words (for the tests and the Commands page)."""

    return {
        "keys": sorted(BY_KEY),
        "codes": sorted(BY_CODE),
        "code_and_key": sorted(f"{code}:{key}" for code, key in BY_CODE_AND_KEY),
    }


__all__ = ["BY_CODE", "BY_KEY", "GENERIC", "IINIT_NAMES", "ISND_NAMES", "RefusalText", "covered", "describe"]
