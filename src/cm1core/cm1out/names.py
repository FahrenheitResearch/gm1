"""CM1 r21.1's history-file names, and where each one comes from in a GM1 frame.

CM1 decides what a history file holds in ``src/writeout.F``: each block tests
its ``&param9`` switch and then sets ``name_output``, ``desc_output``,
``unit_output`` and ``grid_output`` for one variable.  :data:`CM1_VARIABLES`
transcribes those four strings for every variable a GM1 run can write or a
CM1 user's namelist can ask for, with the ``writeout.F`` line of its
``name_output`` statement; a file lists its variables in that line order,
which is CM1's own order.

:func:`selected` gives the variables CM1 writes for one namelist's
settings; :func:`cm1core.cm1out.fields.plan` pairs each with its reader from
a frame, and :func:`not_produced_reason` says in plain words why GM1 does
not write one.

Nothing here imports CuPy.  numpy is imported by the readers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: The CM1 source every string below is copied from.
CM1_WRITEOUT = "CM1 r21.1 src/writeout.F"


@dataclass(frozen=True, slots=True)
class Cm1Variable:
    """One CM1 history variable: its four ``writeout.F`` strings and its line."""

    name: str
    long_name: str
    units: str
    #: ``2`` (yh, xh), ``s`` (zh, yh, xh), ``u`` (zh, yh, xf), ``v`` (zh, yf, xh)
    #: or ``w`` (zf, yh, xh), as ``writeout_nc.F:305-339`` maps them.
    grid: str
    #: The line of ``name_output(n_out) = '<name>'`` in writeout.F.
    line: float


def _v(line: float, name: str, long_name: str, units: str, grid: str) -> Cm1Variable:
    return Cm1Variable(name, long_name, units, grid, line)


# Transcribed from writeout.F (name_output / desc_output / unit_output /
# grid_output).  Where writeout.F picks the description by a condition, the
# branch CM1 takes for the configurations GM1 runs is used and the condition
# is stated beside it.
_TABLE = (
    _v(363, "rain", "accumulated surface rainfall", "cm", "2"),
    _v(389, "prate", "surface precipitation rate", "kg/m2/s", "2"),
    _v(410, "sws", "max horiz wind speed at lowest model level", "m/s", "2"),
    _v(430, "svs", "max vert vorticity at lowest model level", "1/s", "2"),
    _v(450, "sps", "min pressure at lowest model level", "Pa", "2"),
    _v(470, "srs", "max qr at lowest model level", "kg/kg", "2"),
    _v(490, "sgs", "max qg at lowest model level", "kg/kg", "2"),
    _v(510, "sus", "max w at 5 km AGL", "m/s", "2"),
    _v(530, "shs", "max integrated updraft helicity", "m2/s2", "2"),
    _v(551, "rain2", "translated surface rainfall", "cm", "2"),
    _v(573, "sws2", "translated max horiz wspd at lowest model level", "m/s", "2"),
    _v(595, "svs2", "translated max vert vort at lowest model level", "1/s", "2"),
    _v(617, "sps2", "translated min pressure at lowest model level", "Pa", "2"),
    _v(639, "srs2", "translated max qr at lowest model level", "kg/kg", "2"),
    _v(661, "sgs2", "translated max qg at lowest model level", "kg/kg", "2"),
    _v(683, "sus2", "translated max w at 5 km AGL", "m/s", "2"),
    _v(705, "shs2", "translated max integrated updraft helicity", "m2/s2", "2"),
    _v(727, "uh", "integrated (2-5 km) AGL) updraft helicity", "m2/s2", "2"),
    _v(784, "cpc", "cold pool intensity C", "m/s", "2"),
    _v(806, "cph", "cold pool depth h", "m AGL", "2"),
    _v(1018, "psfc", "surface pressure", "Pa", "2"),
    _v(1062, "zs", "terrain height", "m", "2"),
    _v(1084, "cref", "composite reflectivity", "dBZ", "2"),
    _v(2785, "cwp", "cloud water path", "kg/m^2", "2"),
    _v(2822, "lwp", "liquid water path", "kg/m^2", "2"),
    _v(2859, "pwat", "precipitable water", "m", "2"),
    _v(2940, "cape", "convective available potential energy", "J/kg", "2"),
    _v(2964, "cin", "convective inhibition", "J/kg", "2"),
    _v(2988, "lcl", "lifted condensation level", "m", "2"),
    _v(3012, "lfc", "level of free convection", "m", "2"),
    _v(3179, "zhval", "height on model levels", "m", "s"),
    _v(3212, "th", "potential temperature", "K", "s"),
    _v(3259, "thpert", "potential temperature perturbation", "K", "s"),
    _v(3280, "prs", "pressure", "Pa", "s"),
    _v(3301, "prspert", "pressure perturbation", "Pa", "s"),
    _v(3322, "pi", "nondimensional pressure", "nondimensional", "s"),
    _v(3355, "pipert", "nondimensional pressure perturbation", "nondimensional", "s"),
    _v(3418, "rho", "dry-air density", "kg/m^3", "s"),
    _v(3439, "rhopert", "dry-air density perturbation", "kg/m^3", "s"),
    _v(3500, "qv", "water vapor mixing ratio", "kg/kg", "s"),
    _v(3546, "qvpert", "water vapor mixing ratio perturbation", "kg/kg", "s"),
    # 3547.xx: the output_q loop over every q3d scalar but qv (writeout.F
    # after :3546), one variable per species, named and described by qname
    # and qunit -- see species_variables().
    _v(4092, "dbz", "reflectivity", "dBZ", "s"),
    _v(4115, "buoyancy", "buoyancy", "m/s^2", "s"),
    _v(4162, "uinterp", "u interpolated to scalar points (grid-relative)", "m/s", "s"),
    _v(4183, "vinterp", "v interpolated to scalar points (grid-relative)", "m/s", "s"),
    _v(4204, "winterp", "w interpolated to scalar points", "m/s", "s"),
    _v(4342, "xvort", "horizontal vorticity (x)", "1/s", "s"),
    _v(4363, "yvort", "horizontal vorticity (y)", "1/s", "s"),
    _v(4384, "zvort", "vertical vorticity", "1/s", "s"),
    _v(4405, "pv", "potential vorticity", "K m2/kg/s", "s"),
    _v(4433, "pi0", "base-state nondimensional pressure", "nondimensional", "s"),
    _v(4454, "th0", "base-state potential temperature", "K", "s"),
    _v(4475, "prs0", "base-state pressure", "Pa", "s"),
    _v(4496, "qv0", "base-state water vapor mixing ratio", "kg/kg", "s"),
    # Budget terms.  :5144/:5171 (and the qv, u, v, w twins) describe the
    # advection terms as "(non-diff component)" when the advection is odd
    # order or WENO, which is every scheme GM1 runs.
    _v(5144, "ptb_hadv", "pt budget: horiz advection (non-diff component)", "K/s", "s"),
    _v(5171, "ptb_vadv", "pt budget: vert advection (non-diff component)", "K/s", "s"),
    _v(5198, "ptb_hidiff", "pot temp budget: horiz implicit diffusion", "K/s", "s"),
    _v(5221, "ptb_vidiff", "pot temp budget: vert implicit diffusion", "K/s", "s"),
    _v(5244, "ptb_hediff", "pot temp budget: horiz explicit diffusion", "K/s", "s"),
    _v(5267, "ptb_vediff", "pot temp budget: vert explicit diffusion", "K/s", "s"),
    _v(5290, "ptb_hturb", "pot temp budget: horiz parameterized turbulence", "K/s", "s"),
    _v(5313, "ptb_vturb", "pot temp budget: vert parameterized turbulence", "K/s", "s"),
    _v(5336, "ptb_mp", "pot temp budget: microphysics scheme", "K/s", "s"),
    _v(5359, "ptb_rdamp", "pot temp budget: Rayleigh damper", "K/s", "s"),
    _v(5382, "ptb_nudge", "pot temp budget: nudging", "K/s", "s"),
    _v(5405, "ptb_rad", "pot temp budget: radiation scheme", "K/s", "s"),
    _v(5428, "ptb_div", "pot temp budget: moist divergence term", "K/s", "s"),
    # :5451/:5478 name the PBL scheme in the description when ipbl is 3, 4 or 5
    # (see _describe).
    _v(5451, "ptb_diss", "pot temp budget: dissipative heating", "K/s", "s"),
    _v(5478, "ptb_pbl", "pot tem. budget: PBL scheme", "K/s", "s"),
    _v(5505, "ptb_lsw", "pot temp budget: advection by large-scale w", "K/s", "s"),
    # :5529-5679: Morrison (ptype = 5) process rates of the theta budget.
    _v(5529, "tt_cond", "theta tendency: condensation", "K/s", "s"),
    _v(5554, "tt_evac", "theta tendency: cloudwater evaporation", "K/s", "s"),
    _v(5579, "tt_evar", "theta tendency: rainwater evaporation", "K/s", "s"),
    _v(5604, "tt_dep", "theta tendency: deposition", "K/s", "s"),
    _v(5629, "tt_subl", "theta tendency: sublimation", "K/s", "s"),
    _v(5654, "tt_melt", "theta tendency: melting", "K/s", "s"),
    _v(5679, "tt_frz", "theta tendency: freezing", "K/s", "s"),
    _v(5846, "qvb_hadv", "qv budget: horizontal advection (non-diff component)", "kg/kg/s", "s"),
    _v(5873, "qvb_vadv", "qv budget: vertical advection (non-diff component)", "kg/kg/s", "s"),
    _v(5899, "qvb_hidiff", "qv budget: horiz implicit diffusion", "kg/kg/s", "s"),
    _v(5922, "qvb_vidiff", "qv budget: vert implicit diffusion", "kg/kg/s", "s"),
    _v(5945, "qvb_hediff", "qv budget: horiz explicit diffusion", "kg/kg/s", "s"),
    _v(5968, "qvb_vediff", "qv budget: vert explicit diffusion", "kg/kg/s", "s"),
    _v(5991, "qvb_hturb", "qv budget: horizontal parameterized turbulence", "kg/kg/s", "s"),
    _v(6014, "qvb_vturb", "qv budget: vertical parameterized turbulence", "kg/kg/s", "s"),
    _v(6038, "qvb_mp", "qv budget: microphysics scheme", "kg/kg/s", "s"),
    _v(6061, "qvb_nudge", "qv budget: nudging", "kg/kg/s", "s"),
    _v(6084, "qvb_pbl", "qv budget: PBL scheme", "kg/kg/s", "s"),
    _v(6107, "qvb_lsw", "qv budget: advection by large-scale w", "kg/kg/s", "s"),
    # :6131-6231: Morrison (ptype = 5) process rates of the qv budget.
    _v(6131, "qt_cond", "qv tendency: condensation", "kg/kg/s", "s"),
    _v(6156, "qt_evac", "qv tendency: cloudwater evaporation", "kg/kg/s", "s"),
    _v(6181, "qt_evar", "qv tendency: rainwater evaporation", "kg/kg/s", "s"),
    _v(6206, "qt_dep", "qv tendency: deposition", "kg/kg/s", "s"),
    _v(6231, "qt_subl", "qv tendency: sublimation", "kg/kg/s", "s"),
    # :6255-6320: the pressure decomposition (output_pdcomp).
    _v(6255, "pipb", "diagnosed pi-prime: buoyancy component", "nondimensional", "s"),
    _v(6276, "pipdl", "diagnosed pi-prime: linear dynamic component", "nondimensional", "s"),
    _v(6297, "pipdn", "diagnosed pi-prime: nonlinear dynamic component", "nondimensional", "s"),
    _v(6320, "pipc", "diagnosed pi-prime: Coriolis component", "nondimensional", "s"),
    _v(6829, "u", "E-W (x) velocity (grid-relative)", "m/s", "u"),
    _v(6867, "upert", "u perturbation (grid-relative)", "m/s", "u"),
    _v(6888, "u0", "base-state u (grid-relative)", "m/s", "u"),
    _v(6901, "ub_hadv", "u budget: horizontal advection (non-diff component)", "m/s/s", "u"),
    _v(6928, "ub_vadv", "u budget: vertical advection (non-diff component)", "m/s/s", "u"),
    _v(6954, "ub_hidiff", "u budget: horiz implicit diffusion", "m/s/s", "u"),
    _v(6977, "ub_vidiff", "u budget: vert implicit diffusion", "m/s/s", "u"),
    _v(7000, "ub_hediff", "u budget: horiz explicit diffusion", "m/s/s", "u"),
    _v(7023, "ub_vediff", "u budget: vert explicit diffusion", "m/s/s", "u"),
    _v(7046, "ub_hturb", "u budget: horizontal parameterized turbulence", "m/s/s", "u"),
    _v(7069, "ub_vturb", "u budget: vertical parameterized turbulence", "m/s/s", "u"),
    _v(7093, "ub_pgrad", "u budget: pressure gradient", "m/s/s", "u"),
    _v(7116, "ub_rdamp", "u budget: Rayleigh damper", "m/s/s", "u"),
    _v(7139, "ub_cor", "u budget: Coriolis acceleration", "m/s/s", "u"),
    _v(7163, "ub_cent", "u budget: centrifugal acceleration", "m/s/s", "u"),
    _v(7187, "ub_pbl", "u budget: PBL scheme", "m/s/s", "u"),
    _v(7210, "ub_lsw", "u budget: advection by large-scale w", "m/s/s", "u"),
    _v(7290, "v", "N-S (y) velocity (grid-relative)", "m/s", "v"),
    _v(7328, "vpert", "v perturbation (grid-relative)", "m/s", "v"),
    _v(7349, "v0", "base-state v (grid-relative)", "m/s", "v"),
    _v(7362, "vb_hadv", "v budget: horizontal advection (non-diff component)", "m/s/s", "v"),
    _v(7389, "vb_vadv", "v budget: vertical advection (non-diff component)", "m/s/s", "v"),
    _v(7416, "vb_hidiff", "v budget: horiz implicit diffusion", "m/s/s", "v"),
    _v(7439, "vb_vidiff", "v budget: vert implicit diffusion", "m/s/s", "v"),
    _v(7462, "vb_hediff", "v budget: horiz explicit diffusion", "m/s/s", "v"),
    _v(7485, "vb_vediff", "v budget: vert explicit diffusion", "m/s/s", "v"),
    _v(7508, "vb_hturb", "v budget: horizontal parameterized turbulence", "m/s/s", "v"),
    _v(7531, "vb_vturb", "v budget: vertical parameterized turbulence", "m/s/s", "v"),
    _v(7554, "vb_pgrad", "v budget: pressure gradient", "m/s/s", "v"),
    _v(7577, "vb_rdamp", "v budget: Rayleigh damper", "m/s/s", "v"),
    _v(7600, "vb_cor", "v budget: Coriolis acceleration", "m/s/s", "v"),
    _v(7624, "vb_cent", "v budget: centrifugal acceleration", "m/s/s", "v"),
    _v(7648, "vb_pbl", "v budget: PBL scheme", "m/s/s", "v"),
    _v(7671, "vb_lsw", "v budget: advection by large-scale w", "m/s/s", "v"),
    _v(7751, "w", "vertical velocity", "m/s", "w"),
    _v(7790, "tke", "subgrid turbulence kinetic energy", "m^2/s^2", "w"),
    # :7829/:7862 add "(from 2D Smagorinsky scheme)" only when ipbl is 1, 3,
    # 4, 5 or 6, none of which GM1 runs.
    _v(7829, "kmh", "horizontal eddy viscosity for momentum", "m^2/s", "w"),
    _v(7847, "kmv", "vertical eddy viscosity for momentum", "m^2/s", "w"),
    _v(7862, "khh", "horizontal eddy diffusivity for scalars", "m^2/s", "w"),
    _v(7880, "khv", "vertical eddy diffusivity for scalars", "m^2/s", "w"),
    _v(8143, "dissten", "dissipation rate", "m^2/s^3", "w"),
    _v(8156, "nm", "squared Brunt-Vaisala frequency", "1/s^2", "w"),
    _v(8169, "defv", "vertical deformation", "1/s^2", "w"),
    _v(8182, "defh", "horizontal deformation", "1/s^2", "w"),
    _v(8209, "wb_hadv", "w budget: horizontal advection (non-diff component)", "m/s/s", "w"),
    _v(8237, "wb_vadv", "w budget: vertical advection (non-diff component)", "m/s/s", "w"),
    _v(8265, "wb_hidiff", "w budget: horiz implicit diffusion", "m/s/s", "w"),
    _v(8289, "wb_vidiff", "w budget: vert implicit diffusion", "m/s/s", "w"),
    _v(8313, "wb_hediff", "w budget: horiz explicit diffusion", "m/s/s", "w"),
    _v(8337, "wb_vediff", "w budget: vert explicit diffusion", "m/s/s", "w"),
    _v(8361, "wb_hturb", "w budget: horizontal parameterized turbulence", "m/s/s", "w"),
    _v(8385, "wb_vturb", "w budget: vertical parameterized turbulence", "m/s/s", "w"),
    _v(8409, "wb_pgrad", "w budget: pressure gradient", "m/s/s", "w"),
    _v(8433, "wb_rdamp", "w budget: Rayleigh damper", "m/s/s", "w"),
    _v(8457, "wb_buoy", "w budget: buoyancy", "m/s/s", "w"),
    # :8481-8561: the pressure decomposition (output_pdcomp).
    _v(8481, "pgradb", "vert pres grad: buoyancy component", "m/s/s", "w"),
    _v(8514, "pgraddl", "vert pres grad: linear dynamic component", "m/s/s", "w"),
    _v(8537, "pgraddn", "vert pres grad: nonlinear dynamic component", "m/s/s", "w"),
    _v(8561, "pgradc", "vert pres grad: Coriolis component", "m/s/s", "w"),
)

CM1_VARIABLES: Mapping[str, Cm1Variable] = {row.name: row for row in _TABLE}

#: writeout.F's species loop sits right after qvpert (:3546).
_SPECIES_LINE = 3546.5


def species_variables(names: tuple[str, ...], units: tuple[str, ...]) -> tuple[Cm1Variable, ...]:
    """The ``output_q`` loop: every q3d scalar but qv, named by ``qname``.

    writeout.F writes ``text1 = qname(n)`` as both the name and the
    description and ``qunit(n)`` as the unit (``kg/kg`` for a mixing ratio,
    ``#/kg`` for a number concentration).
    """

    rows = []
    for index, (name, unit) in enumerate(zip(names, units)):
        if name == "qv":
            continue
        rows.append(Cm1Variable(name, name, unit, "s", _SPECIES_LINE + index / 1000.0))
    return tuple(rows)


#: The &param9 switch and any further CM1 condition behind each variable.
#: A callable gets the CM1 settings (namelist values plus the derived ones
#: named in :func:`cm1_settings`) and returns whether CM1 writes the variable.
_GATES: Mapping[str, Callable[[Mapping[str, Any]], bool]] = {}


def _gate(names: str, rule: Callable[[Mapping[str, Any]], bool]) -> None:
    for name in names.split():
        _GATES[name] = rule  # type: ignore[index]


def _on(key: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda cm1: int(cm1.get(key, 0) or 0) == 1


def _moving(cm1: Mapping[str, Any]) -> bool:
    return int(cm1.get("imove", 0) or 0) == 1


_gate("rain prate", _on("output_rain"))
for _swath in ("sws", "svs", "sps", "srs", "sgs", "sus", "shs"):
    _gate(_swath, _on(f"output_{_swath}"))
    # writeout.F:549-720: the translated twins exist when nrain = 2, which
    # param.F:4697-4698 sets exactly when imove = 1.
    _gate(f"{_swath}2", (lambda key: lambda cm1: _on(key)(cm1) and _moving(cm1))(f"output_{_swath}"))
_gate("rain2", lambda cm1: _on("output_rain")(cm1) and _moving(cm1))
_gate("uh", _on("output_uh"))
_gate("cpc cph", _on("output_coldpool"))
_gate("psfc", _on("output_psfc"))
_gate("zs", _on("output_zs"))
_gate("cref dbz", _on("output_dbz"))
_gate("cwp lwp", lambda cm1: int(cm1.get("imoist", 0)) == 1 and _on("output_lwp")(cm1))
_gate("pwat", lambda cm1: int(cm1.get("imoist", 0)) == 1 and _on("output_pwat")(cm1))
for _key in ("cape", "cin", "lcl", "lfc"):
    _gate(_key, (lambda key: lambda cm1: int(cm1.get("imoist", 0)) == 1 and _on(key)(cm1))(f"output_{_key}"))
_gate("zhval", _on("output_zh"))
for _key in ("th", "thpert", "prs", "prspert", "pi", "pipert", "rho", "rhopert", "qv", "qvpert",
             "buoyancy", "uinterp", "vinterp", "winterp", "pv", "u", "upert", "v", "vpert", "w",
             "dissten", "nm"):
    _gate(_key, _on(f"output_{_key}"))
_gate("xvort yvort zvort", _on("output_vort"))
_gate("pi0 th0 prs0 qv0 u0 v0", _on("output_basestate"))
_gate("defv defh", _on("output_def"))


def _int(cm1: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = cm1.get(key, default)
    return default if value is None else int(value)


def _idoles(cm1: Mapping[str, Any]) -> bool:
    # param.F:752
    return _int(cm1, "cm1setup", 1) in (1, 4)


def _idopbl(cm1: Mapping[str, Any]) -> bool:
    # param.F:753
    return _int(cm1, "cm1setup", 1) in (2, 4)


def _iusetke(cm1: Mapping[str, Any]) -> bool:
    # param.F:4989-4997 (MYNN) and :5102-5108 (the LES closures that carry tke).
    return ((_idopbl(cm1) and _int(cm1, "ipbl") in (4, 5)) or _int(cm1, "sfcmodel") == 6
            or (_idoles(cm1) and _int(cm1, "sgsmodel") in (1, 3, 4, 5)))


# writeout.F:7790: output_tke .and. idoles .and. iusetke.
_gate("tke", lambda cm1: _on("output_tke")(cm1) and _idoles(cm1) and _iusetke(cm1))
_GATES["kmh"] = _on("output_km")
_GATES["kmv"] = lambda cm1: _on("output_km")(cm1) and (int(cm1.get("sgsmodel", 0)) >= 1 or int(cm1.get("ipbl", 0)) == 2)
_GATES["khh"] = _on("output_kh")
_GATES["khv"] = lambda cm1: _on("output_kh")(cm1) and (int(cm1.get("sgsmodel", 0)) >= 1 or int(cm1.get("ipbl", 0)) == 2)


def _odd(order: int, weno: int) -> bool:
    return order in (3, 5, 7, 9) or weno >= 1


def budget_names(cm1: Mapping[str, Any]) -> frozenset[str]:
    """The budget terms CM1 writes: param.F:5302-5875's td_/qd_/ud_/vd_/wd_ slots.

    writeout.F writes a budget term exactly when its slot is set (``if(
    td_hadv.ge.1 )`` ...), so this is the slot decision, transcribed.
    """

    cm1setup, ipbl, testcase = _int(cm1, "cm1setup", 1), _int(cm1, "ipbl"), _int(cm1, "testcase")
    imoist, ptype = _int(cm1, "imoist"), _int(cm1, "ptype")
    idoles, idopbl = _idoles(cm1), _idopbl(cm1)
    dohturb = idoles or cm1setup == 3 or (cm1setup == 2 and _int(cm1, "horizturb") == 1)  # :3062-3074
    dovturb = idoles or cm1setup == 3
    use_pbl = idopbl and ipbl >= 1                                                           # :4984-4986
    dolsw = testcase in (3, 4, 5, 6)                                                         # :4860-4865
    lsnudge = bool(cm1.get("do_lsnudge", False))
    idiff, difforder = _int(cm1, "idiff"), _int(cm1, "difforder")
    scalar_h = _odd(_int(cm1, "hadvordrs", 5), _int(cm1, "advwenos"))
    scalar_v = _odd(_int(cm1, "vadvordrs", 5), _int(cm1, "advwenos"))
    velocity_h = _odd(_int(cm1, "hadvordrv", 5), _int(cm1, "advwenov"))
    velocity_v = _odd(_int(cm1, "vadvordrv", 5), _int(cm1, "advwenov"))
    rdamp = _int(cm1, "irdamp") >= 1 or _int(cm1, "hrdamp") >= 1
    names: set[str] = set()

    def turbulence(terms: set[str], gate: bool) -> None:
        if gate:
            if dohturb:
                terms.add("hturb")
            if dovturb:
                terms.add("vturb")

    def diffusion(terms: set[str], on: bool) -> None:
        if on:
            terms.add("hediff")
            if difforder == 2:
                terms.add("vediff")

    if _on("output_thbudget")(cm1):                                                         # :5302-5419
        t = {"hadv", "vadv"}
        turbulence(t, cm1setup >= 1)
        if scalar_h:
            t.add("hidiff")
        if scalar_v:
            t.add("vidiff")
        diffusion(t, idiff == 1)
        if imoist >= 1 and ptype >= 1:
            t.add("mp")
        if _int(cm1, "irdamp") >= 1:
            t.add("rdamp")
        if testcase == 10 or lsnudge:
            t.add("nudge")
        if _int(cm1, "rterm") == 1 or _int(cm1, "radopt") >= 1:
            t.add("rad")
        if imoist == 1 and _int(cm1, "eqtset", 2) == 2:
            t.add("div")
        if _int(cm1, "idiss") == 1 or ipbl in (3, 4, 5):
            t.add("diss")
        if use_pbl:
            t.add("pbl")
        if dolsw:
            t.add("lsw")
        names |= {f"ptb_{term}" for term in t}
        if imoist == 1 and ptype == 5:
            names |= {f"tt_{term}" for term in ("cond", "evac", "evar", "dep", "subl", "melt", "frz")}
    if _on("output_qvbudget")(cm1) and imoist == 1:                                         # :5488-5570
        q = {"hadv", "vadv"}
        turbulence(q, cm1setup >= 1 or ipbl >= 1)
        if scalar_h:
            q.add("hidiff")
        if scalar_v:
            q.add("vidiff")
        diffusion(q, idiff == 1)
        if ptype >= 1:
            q.add("mp")
        if testcase == 10 or lsnudge:
            q.add("nudge")
        if use_pbl:
            q.add("pbl")
        if dolsw:
            q.add("lsw")
        names |= {f"qvb_{term}" for term in q}
        if ptype == 5:
            names |= {f"qt_{term}" for term in ("cond", "evac", "evar", "dep", "subl")}
    for switch, prefix in (("output_ubudget", "ub"), ("output_vbudget", "vb"), ("output_wbudget", "wb")):
        if not _on(switch)(cm1):                                                             # :5590-5875
            continue
        m = {"hadv", "vadv", "pgrad"}
        turbulence(m, cm1setup >= 1 or ipbl >= 1)
        if velocity_h:
            m.add("hidiff")
        if velocity_v:
            m.add("vidiff")
        diffusion(m, idiff >= 1)
        if rdamp:
            m.add("rdamp")
        if prefix == "wb":
            m.add("buoy")
        else:
            if _int(cm1, "icor") == 1:
                m.add("cor")
            if _int(cm1, "axisymm") == 1:
                m.add("cent")
            if use_pbl:
                m.add("pbl")
            if dolsw:
                m.add("lsw")
        names |= {f"{prefix}_{term}" for term in m}
    return frozenset(names)


_BUDGET_PREFIXES = ("ptb_", "qvb_", "ub_", "vb_", "wb_", "tt_", "qt_")
for _row in _TABLE:
    if _row.name.startswith(_BUDGET_PREFIXES):
        _GATES[_row.name] = (lambda name: lambda cm1: name in budget_names(cm1))(_row.name)
# writeout.F:6252-6340, :8478-8580: pdcomp = (output_pdcomp = 1), param.F:727-729;
# the Coriolis components only with icor = 1.
_gate("pipb pipdl pipdn pgradb pgraddl pgraddn", _on("output_pdcomp"))
_gate("pipc pgradc", lambda cm1: _on("output_pdcomp")(cm1) and _int(cm1, "icor") == 1)


def _describe(row: Cm1Variable, cm1: Mapping[str, Any]) -> Cm1Variable:
    """The description writeout.F picks by a condition, for these settings."""

    name = row.name
    if name.endswith(("_hadv", "_vadv")):
        scalar = name.startswith(("ptb_", "qvb_"))
        order_key = ("hadvordr" if name.endswith("_hadv") else "vadvordr") + ("s" if scalar else "v")
        if not _odd(_int(cm1, order_key, 5), _int(cm1, "advwenos" if scalar else "advwenov")):
            plain = {
                "ptb_hadv": "pot temp budget: horiz advection", "ptb_vadv": "pot temp budget: vert advection",
                "qvb_hadv": "qv budget: horizontal advection", "qvb_vadv": "qv budget: vertical advection",
            }.get(name) or row.long_name.replace(" (non-diff component)", "")
            return Cm1Variable(name, plain, row.units, row.grid, row.line)
    if name in ("ptb_diss", "ptb_pbl") and _int(cm1, "ipbl") in (3, 4, 5):
        text = ("pot temp budget: diss. heating (from PBL)" if name == "ptb_diss"
                else "pot tem. budget: PBL scheme (excluding diss. heating)")
        return Cm1Variable(name, text, row.units, row.grid, row.line)
    return row


def selected(cm1: Mapping[str, Any], species: tuple[Cm1Variable, ...] = ()) -> tuple[Cm1Variable, ...]:
    """The variables CM1 r21.1 writes for these settings, in its file order."""

    rows = [_describe(row, cm1) for row in _TABLE if _GATES.get(row.name, lambda _cm1: False)(cm1)]
    if _on("output_q")(cm1) and int(cm1.get("imoist", 0)) == 1:
        rows.extend(species)
    return tuple(sorted(rows, key=lambda row: row.line))


# --------------------------------------------------------------------------
# Why a selected variable is not in a GM1 file.  Plain words; each names what
# is missing and, where one exists, what to do.

NOT_PRODUCED_REASONS: Mapping[str, str] = {
    "kmh": "GM1 does not save the eddy viscosity (output_km is not read).",
    "kmv": "GM1 does not save the eddy viscosity (output_km is not read).",
    "khh": "GM1 does not save the eddy diffusivity (output_kh is not read).",
    "khv": "GM1 does not save the eddy diffusivity (output_kh is not read).",
    "rain2": "GM1 does not keep the ground-relative (translated) rainfall.",
    "dbz": ("This microphysics has no reflectivity in the model; GM1 computes it after the run only for "
            "Morrison (CM1's calc_refl10cm)."),
    "cref": ("This microphysics has no reflectivity in the model; GM1 computes it after the run only for "
             "Morrison (CM1's calc_refl10cm)."),
    "srs": "The run did not keep the rain swath (output_srs = 0).",
    "sgs": "The run did not keep the hail swath (output_sgs = 0).",
    "srs2": "The run did not keep the rain swath (output_srs = 0).",
    "sgs2": "The run did not keep the hail swath (output_sgs = 0).",
}

GENERIC_NOT_PRODUCED = "The run did not write this field (the setting was not read, or GM1 has no operator for it)."

#: &param9 switches (and iptra) whose CM1 fields GM1 has no operator for, so
#: none of their variables is in a GM1 file.  CM1 writes a family of fields
#: for each, chosen by further settings, so the switch is named instead.
SWITCHES_NOT_PRODUCED: Mapping[str, str] = {
    "output_interp": "CM1 writes a second, height-interpolated file (cm1out_i) for a terrain run; GM1 runs no terrain and writes none.",
    "output_sfcflx": "GM1 computes no surface heat and moisture fluxes, so their fields (thflux, qvflux, tsk and the rest) are not written.",
    "output_sfcparams": "GM1 runs no land-surface model, so its parameters (xland, znt, mavail and the rest) are not written.",
    "output_sfcdiags": "GM1 has no surface-layer diagnostics, so t2, q2, u10, v10 and the rest are not written.",
    "output_pblten": "GM1 runs no PBL scheme, so there are no PBL tendencies to write.",
    "output_fallvel": "GM1 does not keep the hydrometeor fall speeds (vtc, vtr, vts, vtg, vti).",
    "output_radten": "GM1 runs no radiation scheme, so there are no radiative tendencies to write.",
    "iptra": "GM1 does not run passive tracers, so pt1, pt2 ... are not written.",
}


def switches_not_produced(cm1: Mapping[str, Any]) -> dict[str, str]:
    """The switches that are on (after param.F's own rewrites) whose fields GM1 never writes."""

    return {key: reason for key, reason in SWITCHES_NOT_PRODUCED.items() if _on(key)(cm1)}


#: The rotation fields GM1 computes only on a uniform vertical grid
#: (cm1import.translate.UNIFORM_METRIC_ONLY: output_vort, output_pv,
#: output_uh, output_shs, output_svs).
UNIFORM_GRID_ONLY = frozenset({"xvort", "yvort", "zvort", "pv", "uh", "shs", "shs2", "svs", "svs2"})


def not_produced_reason(name: str, *, uniform: bool = True) -> str:
    if name in UNIFORM_GRID_ONLY and not uniform:
        return ("GM1 computes vorticity, updraft helicity and their swaths only on a uniform vertical grid, "
                "and this run's grid is stretched.")
    return NOT_PRODUCED_REASONS.get(name, GENERIC_NOT_PRODUCED)


__all__ = [
    "CM1_VARIABLES",
    "CM1_WRITEOUT",
    "Cm1Variable",
    "GENERIC_NOT_PRODUCED",
    "NOT_PRODUCED_REASONS",
    "SWITCHES_NOT_PRODUCED",
    "budget_names",
    "not_produced_reason",
    "switches_not_produced",
    "selected",
    "species_variables",
]
