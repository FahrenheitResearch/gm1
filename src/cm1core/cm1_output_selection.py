"""CM1 r21.1 ``&param9``: the 64 ``output_*`` switches, by CM1's own names.

CM1 selects what a history file contains with 64 namelist words
(``run/namelist.input:213-278``; declared at ``src/input.F:107-124`` and read
at ``src/param.F`` through the ``param9`` namelist) that drive roughly 340
``name_output`` assignments in ``src/writeout.F``.  Before this module the
Cartesian lane published twelve fixed carriers and offered no switch at all,
so every CM1 output selector was refused as an unknown key.

Nothing here imports CuPy, numpy or :mod:`gpuwm`.  The selection is a
document-level decision and must stay diagnosable without a CUDA context --
``gpuwm-cm1 doctor`` reads it.

Three dispositions, and each one is a promise:

``PRODUCT``
    The switch selects a field this lane computes and publishes.  Setting it
    changes the frame.

``STATE``
    The field is already in every frame as a prognostic-state carrier, so the
    switch cannot be turned off: this lane's history frame is also the input
    to :mod:`cm1core.gm1_resume` and to ``rust/gm1-view``'s reader, both of
    which address the carrier by name.  ``0`` is refused, and the refusal
    names the carrier that would go missing.

``UNSUPPORTED``
    No operator exists.  ``1`` is refused and the refusal names the concrete
    wrong result -- an absent field silently read as zero, or an invented one.

A ``PRODUCT`` switch may additionally be marked ``unvalidated``: it is
exposed, it runs, and the run receipt and ``doctor`` both carry a loud marker
naming exactly what was not checked.  The rule is: wire everything, flag
what is not proven.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from types import MappingProxyType
from typing import Any, Mapping

CM1_OUTPUT_SELECTION_SCHEMA = "gm1.cm1-output-selection/v1"

#: Where CM1 r21.1 declares and ships these words.
CM1_OUTPUT_AUTHORITY = (
    "src/input.F:107-124 (declaration), run/namelist.input:213-278 "
    "(&param9 as CM1 r21.1 distributes it), src/writeout.F:20-8752 "
    "(the name_output assignments each switch selects)"
)

PRODUCT = "product"
STATE = "state"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class OutputSwitch:
    """One CM1 ``output_*`` word and what this lane does with it."""

    key: str
    cm1_default: int
    gm1_default: int
    disposition: str
    #: CM1 ``name_output`` strings this switch selects here.
    fields: tuple[str, ...] = ()
    #: CM1 file:line for the block that writes them.
    cm1_ref: str = ""
    #: For STATE: the carrier that already carries it.
    carried_by: tuple[str, ...] = ()
    #: For UNSUPPORTED: the concrete wrong result refusing prevents.
    defect: str = ""
    #: For PRODUCT: what a differential against CM1 has not established.
    unvalidated: str | None = None
    #: CM1's own admissible values for this word.  Every ``output_*`` switch
    #: is a 0/1 flag except ``output_format`` (1 GrADS, 2 NetCDF-4) and
    #: ``output_filetype`` (1 one file, 2 one per time, 3 one per MPI rank).
    allowed: tuple[int, ...] = (0, 1)

    @property
    def default_moved(self) -> bool:
        return self.gm1_default != self.cm1_default

    def record(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "key": self.key,
            "disposition": self.disposition,
            "cm1_distributed_default": self.cm1_default,
            "gm1_default": self.gm1_default,
            "cm1_ref": self.cm1_ref,
            "admissible_values": list(self.allowed),
        }
        if self.fields:
            row["cm1_output_names"] = list(self.fields)
        if self.carried_by:
            row["carried_by"] = list(self.carried_by)
        if self.defect:
            row["refused_defect"] = self.defect
        if self.unvalidated:
            row["unvalidated"] = self.unvalidated
        return row


def _s(key, cm1, gm1, disp, **kw) -> OutputSwitch:
    return OutputSwitch(key=key, cm1_default=cm1, gm1_default=gm1, disposition=disp, **kw)


_W = "src/writeout.F"

#: Every ``&param9`` word, in ``run/namelist.input``'s own order.
SWITCHES: tuple[OutputSwitch, ...] = (
    # ---------------------------------------------------------- file layout
    _s("output_format", 1, 1, STATE,
       cm1_ref="README.namelist:1237-1250",
       allowed=(1, 2),
       carried_by=("history-carriers.f32",),
       defect=(
           "output_format=2 is NetCDF-4/HDF5; this lane writes the raw CM1FHS1 "
           "arena, and declaring format 2 would name a container no writer here "
           "produces while the run still wrote binary32 records"
       )),
    _s("output_filetype", 1, 2, STATE,
       cm1_ref="README.namelist:1258-1272",
       allowed=(1, 2, 3),
       carried_by=("step-NNNNNNNN/",),
       defect=(
           "this lane writes one directory per history time (CM1's "
           "output_filetype=2); filetype 1 would name a single growing file and "
           "filetype 3 one file per MPI rank, neither of which this publisher "
           "creates, so a reader following the declared layout would find nothing"
       )),
    _s("output_interp", 0, 0, UNSUPPORTED,
       cm1_ref="README.namelist:1274-1281; src/writeout.F:10111",
       defect=(
           "output_interp writes a second, height-interpolated copy of every "
           "field for a terrain run; this lane refuses terrain, so the '_i' file "
           "set would be a byte-identical duplicate presented as an independent "
           "height-interpolated product"
       )),
    # --------------------------------------------------- surface accumulation
    _s("output_rain", 1, 1, PRODUCT,
       fields=("rain", "prate"),
       cm1_ref=f"{_W}:361-404",
       unvalidated=None),
    # ------------------------------------------------------ run-long extrema
    _s("output_sws", 1, 1, PRODUCT, fields=("sws",),
       cm1_ref=f"{_W}:408-426; src/solve3.F:857-881"),
    _s("output_svs", 1, 1, PRODUCT, fields=("svs",),
       cm1_ref=f"{_W}:428-446; src/solve3.F:885-965"),
    _s("output_sps", 1, 1, PRODUCT, fields=("sps",),
       cm1_ref=f"{_W}:448-466; src/solve3.F:967-991"),
    _s("output_srs", 1, 1, PRODUCT, fields=("srs",),
       cm1_ref=f"{_W}:468-486; src/solve3.F:993-1019"),
    _s("output_sgs", 1, 1, PRODUCT, fields=("sgs",),
       cm1_ref=f"{_W}:488-506; src/solve3.F:1022-1047"),
    _s("output_sus", 1, 1, PRODUCT, fields=("sus",),
       cm1_ref=f"{_W}:508-526; src/solve3.F:1049-1102"),
    _s("output_shs", 1, 1, PRODUCT, fields=("shs",),
       cm1_ref=f"{_W}:528-546; src/solve3.F:1106-1130"),
    # ------------------------------------------------------------- surface 2d
    _s("output_coldpool", 0, 0, UNSUPPORTED,
       cm1_ref=f"{_W}:772-800; src/misclibs.F calccpch",
       defect=(
           "cold-pool intensity C and depth h come from calccpch's integral of "
           "buoyancy below the cold-pool top; publishing a zero plane under the "
           "name 'cpc' would read as a storm with no cold pool at all"
       )),
    _s("output_sfcflx", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:224",
       defect=(
           "hfx/qfx are surface-layer fluxes; this lane's admitted lower "
           "boundary computes drag, not heat and moisture fluxes, so the plane "
           "would be identically zero and read as a run with no surface forcing"
       )),
    _s("output_sfcparams", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:225",
       defect=(
           "xland/znt/mavail describe a land-surface model this lane does not "
           "run; a constant plane would present an unrun surface as a uniform "
           "one"
       )),
    _s("output_sfcdiags", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:226",
       defect=(
           "t2/q2/u10/v10 are surface-layer similarity diagnostics; without "
           "sfcmodel they would be the lowest model level relabelled as 2 m and "
           "10 m values, which is wrong by the whole surface-layer profile"
       )),
    _s("output_psfc", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:227",
       defect=(
           "psfc is pressure extrapolated to the surface below the lowest model "
           "level; publishing prs(k=1) under the name would be low by roughly "
           "half a layer of hydrostatic pressure"
       )),
    _s("output_zs", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:228",
       defect=(
           "zs is terrain height; this lane refuses terrain and already "
           "publishes the identically zero terrain_msl carrier, so a second "
           "zero plane under CM1's name adds a field and no information"
       )),
    _s("output_zh", 0, 1, PRODUCT, fields=("zhval",),
       cm1_ref=f"{_W}:3179"),
    _s("output_basestate", 0, 1, PRODUCT,
       fields=("pi0", "th0", "prs0", "qv0", "u0", "v0"),
       cm1_ref=f"{_W}:4430-4520, :6888, :7349"),
    # ------------------------------------------------- perturbation/absolute
    _s("output_th", 1, 1, PRODUCT, fields=("th",),
       cm1_ref=f"{_W} name_output 'th'; src/solve1.F:893 (th = th0 + tha)"),
    _s("output_thpert", 0, 1, STATE, carried_by=("th3d",),
       cm1_ref=f"{_W} name_output 'thpert'",
       defect=(
           "the th3d carrier IS CM1's tha, the potential-temperature "
           "perturbation (solve1.F:893 forms total theta as th0+tha); it is the "
           "prognostic carrier this runtime integrates and resumes from, so it "
           "cannot be dropped"
       )),
    _s("output_prs", 1, 1, STATE, carried_by=("prs",),
       cm1_ref=f"{_W} name_output 'prs'",
       defect="prs is a published prognostic carrier and the resume path reads it by name"),
    _s("output_prspert", 0, 0, PRODUCT, fields=("prspert",),
       cm1_ref=f"{_W} name_output 'prspert'"),
    _s("output_pi", 0, 0, PRODUCT, fields=("pi",),
       cm1_ref=f"{_W} name_output 'pi'"),
    _s("output_pipert", 0, 1, STATE, carried_by=("pp3d",),
       cm1_ref=f"{_W} name_output 'pipert'",
       defect=(
           "the pp3d carrier IS CM1's ppi, the perturbation nondimensional "
           "pressure the acoustic solver integrates; it cannot be dropped"
       )),
    _s("output_rho", 0, 1, STATE, carried_by=("rho", "rf"),
       cm1_ref=f"{_W} name_output 'rho'",
       defect="rho and rf are published prognostic carriers the resume path reads by name"),
    _s("output_rhopert", 0, 0, PRODUCT, fields=("rhopert",),
       cm1_ref=f"{_W} name_output 'rhopert'"),
    _s("output_tke", 1, 1, STATE, carried_by=("closure.tkea",),
       cm1_ref=f"{_W} name_output 'tke'",
       defect="closure.tkea is the prognostic sgsmodel=1 carrier and cannot be dropped"),
    _s("output_km", 1, 0, UNSUPPORTED,
       cm1_ref=f"{_W} name_output 'kmh','kmv','kmw'",
       defect=(
           "kmh/kmv are the closure's eddy viscosities; this lane holds them in "
           "closure scratch that is released between large steps, so a frame "
           "written at publication time would carry the last tile's values over "
           "the whole domain"
       )),
    _s("output_kh", 1, 0, UNSUPPORTED,
       cm1_ref=f"{_W} name_output 'khh','khv'",
       defect=(
           "khh/khv are the closure's eddy diffusivities and share the released "
           "scratch above; publishing them would spread one tile's values across "
           "the domain"
       )),
    _s("output_qv", 1, 1, STATE, carried_by=("q3d",),
       cm1_ref=f"{_W} name_output 'qv'",
       defect="qv is q3d[0], a published prognostic carrier the resume path reads by name"),
    _s("output_qvpert", 0, 0, PRODUCT, fields=("qvpert",),
       cm1_ref=f"{_W} name_output 'qvpert'"),
    _s("output_q", 1, 1, STATE, carried_by=("q3d",),
       cm1_ref=f"{_W} name_output 'qc','qr','qi','qs','qg'",
       defect="the six WSM6 species are q3d, published prognostic carriers"),
    _s("output_dbz", 1, 1, PRODUCT, fields=("dbz", "cref"),
       cm1_ref=f"{_W}:4085-4108 (dbz), :1081-1104 (cref via misclibs.F calccref)",
       unvalidated=(
           "the reflectivity operator is ArWen's audited WSM6 10 cm forward "
           "operator (gpuwm.core.refl.consume_refl_10cm), not CM1's "
           "src/mp_wsm6.F/src/mp_radar.F operator; the two assume the same WSM6 "
           "size distributions but are separate transcriptions, so a CM1 "
           "differential on 'dbz' measures both the state and the operator and "
           "cannot separate them.  The column-maximum reduction ('cref') IS "
           "CM1's, transcribed from src/misclibs.F calccref and tested against a "
           "CPU oracle."
       )),
    _s("output_buoyancy", 0, 0, UNSUPPORTED,
       cm1_ref=f"{_W} name_output 'buoyancy'",
       defect=(
           "CM1's buoyancy output is the w-equation buoyancy term as the solver "
           "forms it, including the eqtset=2 pressure contribution; a Python or "
           "post-hoc g*(thv'/thv0) would be a different quantity published under "
           "the solver's name"
       )),
    _s("output_u", 1, 1, STATE, carried_by=("u3d",),
       cm1_ref=f"{_W} name_output 'u'",
       defect="u3d is a published prognostic carrier the resume path reads by name"),
    _s("output_upert", 0, 0, PRODUCT, fields=("upert",),
       cm1_ref=f"{_W} name_output 'upert'"),
    _s("output_uinterp", 1, 0, PRODUCT, fields=("uinterp",),
       cm1_ref=f"{_W} name_output 'uinterp'"),
    _s("output_v", 1, 1, STATE, carried_by=("v3d",),
       cm1_ref=f"{_W} name_output 'v'",
       defect="v3d is a published prognostic carrier the resume path reads by name"),
    _s("output_vpert", 0, 0, PRODUCT, fields=("vpert",),
       cm1_ref=f"{_W} name_output 'vpert'"),
    _s("output_vinterp", 1, 0, PRODUCT, fields=("vinterp",),
       cm1_ref=f"{_W} name_output 'vinterp'"),
    _s("output_w", 1, 1, STATE, carried_by=("w3d",),
       cm1_ref=f"{_W} name_output 'w'",
       defect="w3d is a published prognostic carrier the resume path reads by name"),
    _s("output_winterp", 1, 0, PRODUCT, fields=("winterp",),
       cm1_ref=f"{_W} name_output 'winterp'; src/misclibs.F:1579-1587"),
    # ------------------------------------------------------------- rotation
    _s("output_vort", 0, 0, PRODUCT, fields=("xvort", "yvort", "zvort"),
       cm1_ref=f"{_W}:4330-4400; src/misclibs.F:1540-2174 calcvort"),
    _s("output_pv", 0, 0, PRODUCT, fields=("pv",),
       cm1_ref=f"{_W}:4403-4427; src/misclibs.F:1540-2174 calcvort getpv1/2/3"),
    _s("output_uh", 0, 0, PRODUCT, fields=("uh",),
       cm1_ref=f"{_W}:724-769; src/misclibs.F:1417-1532 calcuh"),
    # ------------------------------------------------------- physics tendency
    _s("output_pblten", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:256",
       defect=(
           "this lane runs no PBL scheme; a zero tendency field would read as a "
           "boundary layer whose scheme ran and did nothing"
       )),
    _s("output_dissten", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:257",
       defect=(
           "dissipative heating is accumulated inside the RK schedule; there is "
           "no retained accumulator, so a publication-time recomputation would "
           "be one instantaneous rate presented as the step's heating"
       )),
    _s("output_fallvel", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:258",
       defect=(
           "vtr/vts/vtg/vti are hydrometeor fall velocities computed inside the "
           "microphysics sedimentation sweep and not retained; recomputing them "
           "outside the component would use a different size distribution from "
           "the one that actually fell"
       )),
    _s("output_nm", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:259",
       defect=(
           "the Brunt-Vaisala frequency CM1 outputs is the closure's own nm "
           "array, formed with the moist/ice branch the turbulence scheme used; "
           "a dry recomputation would differ wherever there is condensate"
       )),
    _s("output_def", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:260",
       defect=(
           "defh/defv are the closure's deformation scratch, released between "
           "large steps; a publication-time frame would carry one tile's values"
       )),
    _s("output_radten", 0, 0, UNSUPPORTED,
       cm1_ref="run/namelist.input:261",
       defect=(
           "this lane runs no radiation scheme; a zero tendency would read as a "
           "run whose radiation was computed and found negligible"
       )),
    # ------------------------------------------------------- thermodynamics
    _s("output_cape", 0, 0, PRODUCT, fields=("cape",),
       cm1_ref=f"{_W}:2894-2930; src/getcape.F:14"),
    _s("output_cin", 0, 0, PRODUCT, fields=("cin",),
       cm1_ref=f"{_W}:2894-2930; src/getcape.F:14"),
    _s("output_lcl", 0, 0, PRODUCT, fields=("lcl",),
       cm1_ref=f"{_W}:2894-2930; src/getcape.F:14"),
    _s("output_lfc", 0, 0, PRODUCT, fields=("lfc",),
       cm1_ref=f"{_W}:2894-2930; src/getcape.F:14"),
    _s("output_pwat", 0, 0, PRODUCT, fields=("pwat",),
       cm1_ref=f"{_W} name_output 'pwat'"),
    _s("output_lwp", 0, 0, PRODUCT, fields=("lwp", "cwp"),
       cm1_ref=f"{_W} name_output 'lwp','cwp'"),
    # ------------------------------------------------------------- budgets
    # cm1_budget_terms resolves which of these names a run publishes from the
    # rest of its configuration (param.F:5302-5875); ``fields`` lists every
    # name this lane can reach.  cartesian_budgets takes each at CM1's own
    # placement by replaying GM1's fused kernel with the other direction zeroed.
    # Not yet differenced against a CM1 run: the UNVALIDATED marker travels in
    # every run receipt's budgets block (cartesian_budgets.UNVALIDATED_BUDGETS)
    # rather than here, because the sealed param9 evidence receipt predates it.
    _s("output_thbudget", 0, 0, PRODUCT,
       fields=("ptb_hadv", "ptb_vadv", "ptb_hidiff", "ptb_vidiff", "ptb_hturb",
               "ptb_vturb", "ptb_mp", "ptb_rdamp", "ptb_rad", "ptb_div", "ptb_diss"),
       cm1_ref="run/namelist.input:274; README.namelist:1432-1443; src/param.F:5302-5419; src/writeout.F:5144-5713"),
    _s("output_qvbudget", 0, 0, PRODUCT,
       fields=("qvb_hadv", "qvb_vadv", "qvb_hidiff", "qvb_vidiff", "qvb_hturb",
               "qvb_vturb", "qvb_mp"),
       cm1_ref="run/namelist.input:275; src/param.F:5488-5570; src/writeout.F:5846-6107"),
    _s("output_ubudget", 0, 0, PRODUCT,
       fields=("ub_hadv", "ub_vadv", "ub_hidiff", "ub_vidiff", "ub_hturb", "ub_vturb",
               "ub_pgrad", "ub_rdamp", "ub_cor"),
       cm1_ref="run/namelist.input:276; src/param.F:5590-5693; src/writeout.F:6901-7210"),
    _s("output_vbudget", 0, 0, PRODUCT,
       fields=("vb_hadv", "vb_vadv", "vb_hidiff", "vb_vidiff", "vb_hturb", "vb_vturb",
               "vb_pgrad", "vb_rdamp", "vb_cor"),
       cm1_ref="run/namelist.input:277; src/param.F:5695-5795; src/writeout.F:7362-7671"),
    _s("output_wbudget", 0, 0, PRODUCT,
       fields=("wb_hadv", "wb_vadv", "wb_hidiff", "wb_vidiff", "wb_hturb", "wb_vturb",
               "wb_pgrad", "wb_rdamp", "wb_buoy"),
       cm1_ref="run/namelist.input:278; src/param.F:5797-5875; src/writeout.F:8209-8457"),
    _s("output_pdcomp", 0, 0, UNSUPPORTED,
       cm1_ref="src/pdcomp.F:10; README.namelist:1449-1454",
       defect=(
           "CM1 has one output_pdcomp word and this port declares it as "
           "lagrangian.output_pdcomp, beside the other analysis selectors; a "
           "second [output] spelling could contradict the one that runs"
       )),
)

SWITCH_BY_KEY: Mapping[str, OutputSwitch] = MappingProxyType(
    {switch.key: switch for switch in SWITCHES}
)
OUTPUT_KEYS: tuple[str, ...] = tuple(switch.key for switch in SWITCHES)

if len(SWITCH_BY_KEY) != len(SWITCHES):  # pragma: no cover - table integrity
    raise RuntimeError("duplicate CM1 output switch key")


#: The defaults this lane applies when no ``[output]`` table is declared.
GM1_DEFAULTS: Mapping[str, int] = MappingProxyType(
    {switch.key: switch.gm1_default for switch in SWITCHES}
)
#: CM1 r21.1's own distributed ``run/namelist.input`` values, for comparison.
CM1_DISTRIBUTED_DEFAULTS: Mapping[str, int] = MappingProxyType(
    {switch.key: switch.cm1_default for switch in SWITCHES}
)

#: Why each default was moved away from CM1's distributed deck.  Every entry
#: here is a deliberate, recorded divergence, not an oversight.
DEFAULT_MOVES: Mapping[str, str] = MappingProxyType({
    "output_filetype": (
        "this publisher writes one directory per history time, which is CM1's "
        "output_filetype=2, not the single growing file of filetype=1"
    ),
    "output_zh": (
        "moved ON.  CM1 defaults it off because its zh is recoverable from the "
        "namelist grid words; this lane admits generated stretched columns "
        "whose level heights are not recoverable from dz alone, and the "
        "carrier is one nz-long profile"
    ),
    "output_basestate": (
        "moved ON.  The th3d carrier is CM1's tha, a PERTURBATION; without th0 "
        "beside it a consumer cannot recover absolute potential temperature at "
        "all, and rust/gm1-view labels the field from the carrier's meaning "
        "string.  The base state here is nz-long profiles, so the cost is "
        "kilobytes per frame"
    ),
    "output_thpert": (
        "moved ON.  It is not a new field: the th3d carrier this lane has "
        "always published IS CM1's thpert, and the switch now says so"
    ),
    "output_pipert": "moved ON.  The pp3d carrier this lane publishes IS CM1's pipert",
    "output_rho": "moved ON.  rho and rf are published prognostic carriers",
    "output_km": (
        "moved OFF.  CM1's deck asks for the eddy viscosities; this lane "
        "releases the closure scratch that holds them between large steps, and "
        "publishing the surviving tile would be wrong over the whole domain"
    ),
    "output_kh": "moved OFF, for the same released-scratch reason as output_km",
    "output_uinterp": (
        "moved OFF.  CM1's deck writes the mass-point winds as a convenience; "
        "they are an exact 2-point average of the published u3d carrier, and "
        "defaulting them on would add one mass volume per component per frame"
    ),
    "output_vinterp": "moved OFF, for the same recoverable-from-v3d reason as output_uinterp",
    "output_winterp": "moved OFF, for the same recoverable-from-w3d reason as output_uinterp",
})


@dataclass(frozen=True, slots=True)
class OutputSelection:
    """One admitted ``&param9`` selection, by CM1's own key names."""

    values: Mapping[str, int] = _field(default_factory=lambda: MappingProxyType(dict(GM1_DEFAULTS)))
    declared: tuple[str, ...] = ()
    #: Words whose value another table decided (:meth:`with_values`), each
    #: with the sentence that says which table and why.
    bound_from: Mapping[str, str] = _field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        missing = [key for key in OUTPUT_KEYS if key not in self.values]
        if missing:
            raise ValueError(f"output selection is missing {', '.join(missing)}")
        extra = [key for key in self.values if key not in SWITCH_BY_KEY]
        if extra:
            raise ValueError(f"output selection carries unknown keys {', '.join(sorted(extra))}")
        for key, value in self.values.items():
            allowed = SWITCH_BY_KEY[key].allowed
            if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
                allowed_text = " or ".join(str(item) for item in allowed)
                raise ValueError(
                    f"{key} must be the integer {allowed_text}, not {value!r}"
                )
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        unknown = [key for key in self.bound_from if key not in SWITCH_BY_KEY]
        if unknown:
            raise ValueError(f"output selection binds unknown keys {', '.join(sorted(unknown))}")
        object.__setattr__(self, "bound_from", MappingProxyType(dict(self.bound_from)))

    def __getitem__(self, key: str) -> int:
        return self.values[key]

    def on(self, key: str) -> bool:
        return bool(self.values[key])

    def with_values(self, overrides: Mapping[str, int],
                    sources: Mapping[str, str] | None = None) -> "OutputSelection":
        """The same selection with some words rebound from another key.

        Used for ``output_dbz``, which CM1 spells once and this port ships as
        ``physics.output_dbz``: the &param9 machinery still needs the word,
        but it must not be separately declarable in ``[output]`` or a
        document could assert both values at once.  The rebound word is not
        added to ``declared``, because the document did not declare it here.

        ``sources`` names, per rebound word, the table that decided it; the
        record's ``differs_from_cm1_distributed_deck`` reason says that rather
        than claim the word came from ``[output]``.
        """

        merged = dict(self.values)
        merged.update(overrides)
        bound = dict(self.bound_from)
        bound.update(sources or {})
        return OutputSelection(values=merged, declared=self.declared, bound_from=bound)

    def difference_reason(self, key: str) -> str:
        """Why this word differs from CM1's distributed deck, truthfully.

        A word the document declared in ``[output]`` says so; a word another
        table decided names that table; an undeclared word at a moved GM1
        default gives the recorded reason for the move.
        """

        if key in self.declared:
            return "declared in this run's [output] table"
        if key in self.bound_from:
            return self.bound_from[key]
        if key in DEFAULT_MOVES:
            return DEFAULT_MOVES[key]
        return "this lane's default for a word the document did not declare"

    @property
    def moved_from_cm1(self) -> tuple[str, ...]:
        return tuple(
            key for key in OUTPUT_KEYS
            if self.values[key] != CM1_DISTRIBUTED_DEFAULTS[key]
        )

    def unvalidated_markers(self) -> tuple[Mapping[str, Any], ...]:
        """Loud, machine-readable markers for every selected-but-unproven field."""

        rows = []
        for switch in SWITCHES:
            if switch.disposition != PRODUCT or not self.values[switch.key]:
                continue
            if switch.unvalidated is None:
                continue
            rows.append(MappingProxyType({
                "marker": "UNVALIDATED",
                "option": switch.key,
                "cm1_output_names": list(switch.fields),
                "not_checked": switch.unvalidated,
            }))
        return tuple(rows)

    def record(self) -> dict[str, Any]:
        return {
            "schema": CM1_OUTPUT_SELECTION_SCHEMA,
            "authority": CM1_OUTPUT_AUTHORITY,
            "switch_count": len(SWITCHES),
            "values": dict(self.values),
            "declared_in_document": list(self.declared),
            "cm1_distributed_defaults": dict(CM1_DISTRIBUTED_DEFAULTS),
            "differs_from_cm1_distributed_deck": {
                key: {
                    "cm1": CM1_DISTRIBUTED_DEFAULTS[key],
                    "gm1": self.values[key],
                    "reason": self.difference_reason(key),
                }
                for key in self.moved_from_cm1
            },
            "selected_cm1_output_names": list(self.selected_names()),
            "unvalidated": [dict(row) for row in self.unvalidated_markers()],
            "dispositions": [switch.record() for switch in SWITCHES],
        }

    def selected_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for switch in SWITCHES:
            if switch.disposition == PRODUCT and self.values[switch.key]:
                names.extend(switch.fields)
        return tuple(names)



#: The runtime group every ``&param9`` product is staged from.  The runtime
#: fills one dictionary under this name immediately before a frame is staged;
#: :mod:`cm1core.cartesian_history` reads it exactly as it reads ``fields``,
#: ``tke`` and ``metrics``.
OUTPUT_GROUP = "output"


def output_carrier_rows(
    selection: "OutputSelection",
    *,
    mass: tuple[int, int, int],
    w: tuple[int, int, int],
    u: tuple[int, int, int],
    v: tuple[int, int, int],
    surface: tuple[int, int],
    profile: tuple[int],
) -> tuple[tuple[Any, ...], ...]:
    """The carrier rows one ``&param9`` selection adds to a history frame.

    Each row is ``(name, meaning, units, stagger, shape, group, key, staged)``
    in :class:`cm1core.cartesian_history.CartesianHistoryCarrier`'s own order.
    Names are CM1 ``writeout.F`` ``name_output`` strings wherever CM1 has one,
    so a frame opened by a CM1 tool carries the names that tool expects.
    """

    rows: list[tuple[Any, ...]] = []

    def add(name, meaning, units, stagger, shape):
        rows.append((name, meaning, units, stagger, shape, OUTPUT_GROUP, name, False))

    if selection.on("output_rain"):
        add("rain", "accumulated surface rainfall", "cm", "surface", surface)
        add("prate", "surface precipitation rate", "kg m-2 s-1", "surface", surface)
        # CM1 r21.1 writeout.F publishes no snow/graupel accumulation of its
        # own; the WSM6 component this lane runs produces all three and the
        # snow ratio, and they are named here for what they are.
        add("snownc", "accumulated surface snowfall (WSM6 SNOWNC)", "cm", "surface", surface)
        add("graupelnc", "accumulated surface graupel (WSM6 GRAUPELNC)", "cm", "surface", surface)
        add("sr", "snow fraction of the last step's surface precipitation", "1", "surface", surface)
    if selection.on("output_sws"):
        add("sws", "run-long max ground-relative horizontal wind speed at the lowest model level",
            "m s-1", "surface", surface)
    if selection.on("output_svs"):
        add("svs", "run-long max vertical vorticity at the lowest model level",
            "s-1", "surface", surface)
    if selection.on("output_sps"):
        add("sps", "run-long min pressure perturbation at the lowest model level",
            "Pa", "surface", surface)
    if selection.on("output_srs"):
        add("srs", "run-long max qr at the lowest model level", "kg kg-1", "surface", surface)
    if selection.on("output_sgs"):
        add("sgs", "run-long max qg at the lowest model level", "kg kg-1", "surface", surface)
    if selection.on("output_sus"):
        add("sus", "run-long max w at 5 km AGL", "m s-1", "surface", surface)
    if selection.on("output_shs"):
        add("shs", "run-long max integrated updraft helicity", "m2 s-2", "surface", surface)
    if selection.on("output_dbz"):
        add("dbz", "simulated 10 cm radar reflectivity", "dBZ", "mass", mass)
        add("cref", "composite (column-maximum) reflectivity", "dBZ", "surface", surface)
    if selection.on("output_uh"):
        add("uh", "integrated (2-5 km AGL) updraft helicity", "m2 s-2", "surface", surface)
    if selection.on("output_vort"):
        add("xvort", "horizontal vorticity (x)", "s-1", "mass", mass)
        add("yvort", "horizontal vorticity (y)", "s-1", "mass", mass)
        add("zvort", "vertical vorticity", "s-1", "mass", mass)
    if selection.on("output_pv"):
        add("pv", "potential vorticity", "K m2 kg-1 s-1", "mass", mass)
    if selection.on("output_cape"):
        add("cape", "convective available potential energy (mixed-layer parcel)",
            "J kg-1", "surface", surface)
    if selection.on("output_cin"):
        add("cin", "convective inhibition (mixed-layer parcel)", "J kg-1", "surface", surface)
    if selection.on("output_lcl"):
        add("lcl", "lifting condensation level, -1 where the parcel never saturated",
            "m AGL", "surface", surface)
    if selection.on("output_lfc"):
        add("lfc", "level of free convection, -1 where the parcel never reached one",
            "m AGL", "surface", surface)
    if selection.on("output_th"):
        add("th", "potential temperature (th0 + th3d)", "K", "mass", mass)
    if selection.on("output_prspert"):
        add("prspert", "pressure perturbation", "Pa", "mass", mass)
    if selection.on("output_pi"):
        add("pi", "nondimensional pressure (pi0 + pp3d)", "nondimensional", "mass", mass)
    if selection.on("output_rhopert"):
        add("rhopert", "density perturbation", "kg m-3", "mass", mass)
    if selection.on("output_qvpert"):
        add("qvpert", "water vapour mixing ratio perturbation", "kg kg-1", "mass", mass)
    if selection.on("output_upert"):
        add("upert", "west-east velocity perturbation", "m s-1", "u", u)
    if selection.on("output_vpert"):
        add("vpert", "south-north velocity perturbation", "m s-1", "v", v)
    if selection.on("output_uinterp"):
        add("uinterp", "west-east velocity at scalar points", "m s-1", "mass", mass)
    if selection.on("output_vinterp"):
        add("vinterp", "south-north velocity at scalar points", "m s-1", "mass", mass)
    if selection.on("output_winterp"):
        add("winterp", "vertical velocity at scalar points", "m s-1", "mass", mass)
    if selection.on("output_pwat"):
        add("pwat", "precipitable water", "m", "surface", surface)
    if selection.on("output_lwp"):
        add("lwp", "liquid water path", "kg m-2", "surface", surface)
        add("cwp", "cloud water path", "kg m-2", "surface", surface)
    if selection.on("output_zh"):
        add("zhval", "height of the scalar levels", "m", "profile", profile)
    if selection.on("output_basestate"):
        add("pi0", "base-state nondimensional pressure", "nondimensional", "profile", profile)
        add("th0", "base-state potential temperature", "K", "profile", profile)
        add("prs0", "base-state pressure", "Pa", "profile", profile)
        add("qv0", "base-state water vapor mixing ratio", "kg kg-1", "profile", profile)
        add("u0", "base-state west-east velocity", "m s-1", "profile", profile)
        add("v0", "base-state south-north velocity", "m s-1", "profile", profile)
    del w
    return tuple(rows)


#: Products whose device work must run on every large step, not only at
#: publication time: a run-long extremum sampled at frame times is not a
#: run-long extremum.  ``solve3.F:857-1130`` updates all of these per step.
PER_STEP_PRODUCTS: Mapping[str, str] = MappingProxyType({
    "output_sws": "src/solve3.F:857-881",
    "output_svs": "src/solve3.F:885-965",
    "output_sps": "src/solve3.F:967-991",
    "output_srs": "src/solve3.F:993-1019",
    "output_sgs": "src/solve3.F:1022-1047",
    "output_sus": "src/solve3.F:1049-1102",
    "output_shs": "src/solve3.F:1106-1130",
})


def requires_per_step_work(selection: "OutputSelection") -> bool:
    return any(selection.on(key) for key in PER_STEP_PRODUCTS)


def requires_updraft_helicity(selection: "OutputSelection") -> bool:
    """``shs`` is the run-long maximum of the same field ``uh`` publishes."""

    return selection.on("output_uh") or selection.on("output_shs")


DEFAULT_OUTPUT_SELECTION = OutputSelection()


def default_selection() -> OutputSelection:
    return DEFAULT_OUTPUT_SELECTION


__all__ = [
    "CM1_OUTPUT_AUTHORITY",
    "OUTPUT_GROUP",
    "PER_STEP_PRODUCTS",
    "output_carrier_rows",
    "requires_per_step_work",
    "requires_updraft_helicity",
    "CM1_OUTPUT_SELECTION_SCHEMA",
    "CM1_DISTRIBUTED_DEFAULTS",
    "DEFAULT_MOVES",
    "DEFAULT_OUTPUT_SELECTION",
    "GM1_DEFAULTS",
    "OUTPUT_KEYS",
    "OutputSelection",
    "OutputSwitch",
    "PRODUCT",
    "STATE",
    "SWITCHES",
    "SWITCH_BY_KEY",
    "UNSUPPORTED",
    "default_selection",
]
