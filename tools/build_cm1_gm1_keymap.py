#!/usr/bin/env python3
"""Emit the explicit CM1 r21.1 namelist key -> gpuwm-cm1 TOML key map.

The map below is authored by hand, key by key, against CM1's own
``README.namelist`` and ``src/input.F``.  The generator then *checks* it
against the packaged authority inventory
(``rust/cm1-state/schema/cm1-r21.1-namelist-support.json``, itself derived
from the pinned CM1 sources) so that:

  * every CM1 namelist field in a physical-configuration group is accounted
    for -- either mapped to a gpuwm-cm1 TOML key or explicitly recorded as
    having no equivalent, with a reason;
  * no entry names a CM1 field that does not exist;
  * no entry names a gpuwm-cm1 key outside ``config.py``'s ``_SCHEMA``.

The emitted "unmapped" list is therefore a *derived* gap register: it is what
the authority inventory contains minus what this port can express, rather
than an assertion about coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "gm1.cm1-r21.1-to-gpuwm-cm1-keymap/v1"

#: Groups whose fields describe the *physical configuration* of a run.  The
#: remaining groups (param9/10/13/14/15/16 and the two NSSL microphysics
#: tables) select output, restart and scheme-internal tuning; they are
#: summarised by family rather than mapped key by key, because the harness
#: owns the output selectors outright.
PHYSICAL_GROUPS = (
    "param0", "param1", "param2", "param3", "param4", "param5", "param6",
    "param7", "param8",
    # &param9 selects what a history file contains.  It became a physical
    # group here when gpuwm-cm1 grew a key for all 64 of its words
    # (cm1_output_selection); before that every one of them was an unmapped
    # gap.  The differential harness still OWNS the selection -- it forces both
    # models to write the same list of fields -- so a case document may not
    # move one, and load_case refuses it by name.
    "param9",
    "param11", "param12", "param17", "param18",
    "param19", "param20", "param21",
)

#: CM1 groups that gpuwm-cm1 maps key-for-key but the differential harness
#: sets itself on both sides.
HARNESS_OWNED_GROUPS = frozenset({"param9"})

# --------------------------------------------------------------------------
# The map.  Each entry is
#     cm1_key: (gm1_dotted_key | None, transform, note)
# ``transform`` names how the value crosses, and is one of:
#   identity          - same number, same meaning
#   seconds           - CM1 seconds -> GM1 ``*_s`` seconds (same number)
#   metres            - CM1 metres  -> GM1 ``*_m`` metres  (same number)
#   cm1-zd-to-depth   - CM1 zd (height of the damping-layer base) -> GM1
#                       upper_damping_depth_m (its depth below the model top)
#   int-to-bool       - CM1 0/1 integer -> GM1 strict boolean
#   logical-to-bool   - CM1 .true./.false. -> GM1 strict boolean
#   abs-seconds       - CM1 signed rstfrq -> GM1 positive cadence
#   derived           - GM1 spells the same physics with a different word
#   alias             - a second GM1 spelling of a key already mapped
# --------------------------------------------------------------------------
MAP: dict[str, tuple[str | None, str, str]] = {
    # ---- param0 ----------------------------------------------------------
    "nx": ("grid.nx", "identity", "README.namelist:20-24"),
    "ny": ("grid.ny", "identity", "README.namelist:20-24"),
    "nz": ("grid.nz", "identity", "README.namelist:20-24"),
    "ppnode": (None, "none", "MPI cores-per-node; this port is single-device CUDA and has no MPI decomposition"),
    "timeformat": (None, "none", "stdout time formatting only; no effect on state"),
    "timestats": (None, "none", "stdout timer report only; no effect on state"),
    "terrain_flag": ("grid.terrain", "logical-to-bool", "README.namelist:49-50"),
    "procfiles": (None, "none", "per-MPI-rank text files; no MPI in this port"),
    "outunits": (None, "none", "stdout units label only; no effect on state"),
    # ---- param1 ----------------------------------------------------------
    "dx": ("grid.dx_m", "metres", "uniform x spacing; read only when stretch_x=0"),
    "dy": ("grid.dy_m", "metres", "uniform y spacing; read only when stretch_y=0"),
    "dz": ("grid.dz_m", "metres", "uniform z spacing; read only when stretch_z=0"),
    "dtl": ("run.dt_s", "seconds", "large time step"),
    "timax": ("run.duration_s", "seconds", "model run length"),
    "run_time": (None, "none", "wall-clock stop; a scheduler convenience, not physics"),
    "tapfrq": ("run.history_interval_s", "seconds", "history write cadence"),
    "rstfrq": ("run.restart_interval_s", "abs-seconds", "CM1 uses a negative rstfrq as the disable sentinel; GM1 carries a positive cadence and a lane that selects Rust checkpointing"),
    "statfrq": (None, "none", "CM1 domain-statistics cadence; GM1 has no statpack writer"),
    "prclfrq": (None, "none", "parcel write cadence; GM1 has no parcel integrator"),
    # ---- param9 (the 64 output_* words) ----------------------------------
    "output_format": ("output.output_format", "identity", "the field is a prognostic carrier every frame already holds; README.namelist:1237-1250"),
    "output_filetype": ("output.output_filetype", "identity", "the field is a prognostic carrier every frame already holds; README.namelist:1258-1272"),
    "output_interp": ("output.output_interp", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; README.namelist:1274-1281; src/writeout.F:10111"),
    "output_rain": ("output.output_rain", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F:361-404"),
    "output_sws": ("diagnostics.output_sws", "identity", "README.namelist:1340; solve3.F:857-1130 run-long max surface wind swath"),
    "output_svs": ("diagnostics.output_svs", "identity", "README.namelist:1341; solve3.F max surface vertical vorticity swath"),
    "output_sps": ("diagnostics.output_sps", "identity", "README.namelist:1342; solve3.F min surface pressure swath"),
    "output_srs": ("diagnostics.output_srs", "identity", "README.namelist:1343; solve3.F max surface rainwater swath"),
    "output_sgs": ("diagnostics.output_sgs", "identity", "README.namelist:1344; solve3.F max surface graupel swath"),
    "output_sus": ("diagnostics.output_sus", "identity", "README.namelist:1345; solve3.F max 5 km updraft swath"),
    "output_shs": ("diagnostics.output_shs", "identity", "README.namelist:1346; solve3.F max updraft helicity swath"),
    "output_coldpool": ("output.output_coldpool", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; src/writeout.F:772-800; src/misclibs.F calccpch"),
    "output_sfcflx": ("output.output_sfcflx", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:224"),
    "output_sfcparams": ("output.output_sfcparams", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:225"),
    "output_sfcdiags": ("output.output_sfcdiags", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:226"),
    "output_psfc": ("output.output_psfc", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:227"),
    "output_zs": ("output.output_zs", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:228"),
    "output_zh": ("output.output_zh", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F:3179"),
    "output_basestate": ("output.output_basestate", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F:4430-4520, :6888, :7349"),
    "output_th": ("output.output_th", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'th'; src/solve1.F:893 (th = th0 + tha)"),
    "output_thpert": ("output.output_thpert", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'thpert'"),
    "output_prs": ("output.output_prs", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'prs'"),
    "output_prspert": ("output.output_prspert", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'prspert'"),
    "output_pi": ("output.output_pi", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'pi'"),
    "output_pipert": ("output.output_pipert", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'pipert'"),
    "output_rho": ("output.output_rho", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'rho'"),
    "output_rhopert": ("output.output_rhopert", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'rhopert'"),
    "output_tke": ("output.output_tke", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'tke'"),
    "output_km": ("output.output_km", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; src/writeout.F name_output 'kmh','kmv','kmw'"),
    "output_kh": ("output.output_kh", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; src/writeout.F name_output 'khh','khv'"),
    "output_qv": ("output.output_qv", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'qv'"),
    "output_qvpert": ("output.output_qvpert", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'qvpert'"),
    "output_q": ("output.output_q", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'qc','qr','qi','qs','qg'"),
    "output_dbz": ("output.output_dbz", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F:4085-4108 (dbz), :1081-1104 (cref via misclibs.F calccref)"),
    "output_buoyancy": ("output.output_buoyancy", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; src/writeout.F name_output 'buoyancy'"),
    "output_u": ("output.output_u", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'u'"),
    "output_upert": ("output.output_upert", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'upert'"),
    "output_uinterp": ("output.output_uinterp", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'uinterp'"),
    "output_v": ("output.output_v", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'v'"),
    "output_vpert": ("output.output_vpert", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'vpert'"),
    "output_vinterp": ("output.output_vinterp", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'vinterp'"),
    "output_w": ("output.output_w", "identity", "the field is a prognostic carrier every frame already holds; src/writeout.F name_output 'w'"),
    "output_winterp": ("output.output_winterp", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'winterp'; src/misclibs.F:1579-1587"),
    "output_vort": ("diagnostics.output_vort", "identity", "README.namelist:1380; misclibs.F:1540-2174 calcvort"),
    "output_pv": ("diagnostics.output_pv", "identity", "README.namelist:1381; potential vorticity from calcvort"),
    "output_uh": ("diagnostics.output_uh", "identity", "README.namelist:1382; misclibs.F:1417-1532 calcuh updraft helicity"),
    "output_pblten": ("output.output_pblten", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:256"),
    "output_dissten": ("output.output_dissten", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:257"),
    "output_fallvel": ("output.output_fallvel", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:258"),
    "output_nm": ("output.output_nm", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:259"),
    "output_def": ("output.output_def", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:260"),
    "output_radten": ("output.output_radten", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:261"),
    "output_cape": ("diagnostics.output_cape", "identity", "README.namelist:1372; writeout.F:2937-2949 publishes the CAPE plane"),
    "output_cin": ("diagnostics.output_cin", "identity", "README.namelist:1373; writeout.F:2951-2963"),
    "output_lcl": ("diagnostics.output_lcl", "identity", "README.namelist:1374; writeout.F:2965-2990"),
    "output_lfc": ("diagnostics.output_lfc", "identity", "README.namelist:1375; writeout.F:2992-3028"),
    "output_pwat": ("output.output_pwat", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'pwat'"),
    "output_lwp": ("output.output_lwp", "identity", "selects a field this lane computes and publishes under CM1's name; src/writeout.F name_output 'lwp','cwp'"),
    "output_thbudget": ("output.output_thbudget", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:274; README.namelist:1432-1443"),
    "output_qvbudget": ("output.output_qvbudget", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:275"),
    "output_ubudget": ("output.output_ubudget", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:276"),
    "output_vbudget": ("output.output_vbudget", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:277"),
    "output_wbudget": ("output.output_wbudget", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; run/namelist.input:278"),
    "output_pdcomp": ("output.output_pdcomp", "identity", "admitted by key; value 1 is refused with the wrong result it prevents named; src/pdcomp.F:10; README.namelist:1445-1450"),
    "prclfrq": ("lagrangian.prclfrq_s", "seconds", "README.namelist:129-133; parcel write cadence"),
    # ---- param2 ----------------------------------------------------------
    "cm1setup": ("closures.cm1setup", "identity", "which closure mixes the column (src/param.F:752-753); GM1 declares cm1setup and ipbl as one pair and refuses every value but 1 in the Cartesian lane, whose prognostic-TKE closure has no off switch"),
    "testcase": ("initialization.testcase", "identity", "run/namelist.input:29; selects base.F:2467-2510's constant wind and init3d.F:1459-1620's random amplitude and depth limit"),
    "adapt_dt": ("dynamics.adaptive_timestep", "int-to-bool", "adaptive large step"),
    "irst": ("initialization.irst", "identity", "param.F:891-903 forces iinit=0 on a restart; GM1 restarts state through its own Rust checkpoint envelope but reads the selector for that rule"),
    "rstnum": (None, "none", "restart file number; see irst"),
    "iconly": ("run.iconly", "identity", "README.namelist:233-236 / cm1.F:4341-4346: build the initial condition, write it, stop"),
    "hadvordrs": ("dynamics.scalar_horizontal_order", "identity", "README.namelist"),
    "vadvordrs": ("dynamics.scalar_vertical_order", "identity", "README.namelist"),
    "hadvordrv": ("dynamics.velocity_horizontal_order", "identity", "README.namelist"),
    "vadvordrv": ("dynamics.velocity_vertical_order", "identity", "README.namelist"),
    "advwenos": ("dynamics.scalar_weno", "identity", "README.namelist"),
    "advwenov": ("dynamics.velocity_weno", "identity", "README.namelist"),
    "weno_order": ("dynamics.weno_order", "identity", "README.namelist"),
    "apmasscon": ("dynamics.dry_mass_correction", "int-to-bool", "dry-mass conservation fix"),
    "idiff": ("dynamics.idiff", "identity", "explicit artificial diffusion selector; the Cartesian lane admits 0 only and names the missing difforder branch"),
    "mdiff": ("dynamics.mdiff", "identity", "Xue monotonic variant of the sixth-order diffusion; read only when idiff>=1"),
    "difforder": ("dynamics.difforder", "identity", "artificial-diffusion order; read only when idiff>=1"),
    "imoist": ("physics.component", "derived-component", "GM1 spells moisture as the physics component: component='off' is imoist=0, any microphysics component is imoist=1"),
    "ipbl": ("closures.ipbl", "identity", "PBL scheme; ipbl=1/4/6 and the GM1 extension 11 resolve to byte-audited ArWen components and refuse at preflight naming the surface-layer carriers GM1 does not have"),
    "sgsmodel": ("closures.sgs_model", "identity", "subgrid turbulence closure"),
    "tconfig": (None, "none", "turbulence configuration; GM1 pins CM1's default"),
    "bcturbs": (None, "none", "turbulence lateral boundary treatment; GM1 pins CM1's default"),
    "horizturb": (None, "none", "separate horizontal turbulence length scale; no TOML key"),
    "doimpl": ("closures.vertical_implicit_diffusion", "int-to-bool", "vertically implicit diffusion"),
    "irdamp": ("closures.upper_damping", "identity", "upper Rayleigh damping"),
    "hrdamp": ("closures.hrdamp", "identity", "lateral Rayleigh damping (param.F:8687-8802); acts on w only since cm1r19.6"),
    "psolver": ("dynamics.psolver", "identity", "pressure solver"),
    "ptype": ("physics.component", "derived-component", "microphysics scheme number; GM1 names the scheme: ptype=5 is morrison-mp10, 7 is wsm6-mp6, 3 is thompson-mp8 (src/param.F:2715-2730, README.namelist:458-495; cm1import.translate.PTYPE_COMPONENTS)"),
    "ihail": ("physics.ihail", "identity", "dense-ice branch: src/param.F:3407 into Morrison, :3697-3706 into WSM6; optional in the TOML, and an undeclared value binds the component's own recorded default (CM1's shipped 1 for both schemes CM1 numbers)"),
    "iautoc": (None, "none", "autoconversion option; no TOML key"),
    "output_dbz": ("physics.output_dbz", "identity", "README.namelist:1366-1369; GM1 spells it in [physics], beside the component whose scheme-specific formula computes it and beside dbz_level_m"),
    "cuparam": ("physics.cuparam", "identity", "cumulus parameterization; cuparam=1 is the same CCPP/MMM new Tiedtke ArWen ships as ntiedtke.cu, refused below 4000 m spacing and on an estate older than ArWen 2.7.4"),
    "icor": ("dynamics.coriolis", "identity", "Coriolis switch; solve2.F:631-720 between divx and advu"),
    "betaplane": ("dynamics.beta_plane", "identity", "beta-plane Coriolis; param.F:8945-8969 builds f2d"),
    "lspgrad": ("dynamics.lspgrad", "identity", "large-scale pressure gradient; 1 (solve1.F:590-612, Coriolis on the perturbation from the base state) is live in the Cartesian closure, 2/3/4 are refused by name"),
    "eqtset": ("physics.equation_set", "identity", "mass/energy equation set"),
    "idiss": (None, "none", "dissipative heating; eqtset2_surface_dissipation.py exists but no TOML key selects it"),
    "efall": (None, "none", "fallout energy term; cuda_gm1_kessler_efall.py exists but no TOML key selects it"),
    "rterm": ("closures.rterm", "identity", "Rotunno-Emanuel Newtonian relaxation that mimics radiation, capped at 2 K/day (src/solve1.F:1003-1054); live in the Cartesian closure"),
    "wbc": ("boundaries.west", "identity", "west lateral boundary"),
    "ebc": ("boundaries.east", "identity", "east lateral boundary"),
    "sbc": ("boundaries.south", "identity", "south lateral boundary"),
    "nbc": ("boundaries.north", "identity", "north lateral boundary"),
    "bbc": ("closures.bbc", "identity", "lower boundary wind condition; boundaries.bottom is a second GM1 spelling of the same selector and must agree"),
    "tbc": ("boundaries.top", "identity", "upper boundary"),
    "irbc": ("boundaries.radiative_order", "identity", "open lateral radiative scheme"),
    "roflux": ("boundaries.roflux", "identity", "restrict outward mass flux at open boundaries (bc.F:2180-2391)"),
    "nudgeobc": ("boundaries.nudgeobc", "identity", "nudge inflow winds toward the base state (solve2.F:726-822)"),
    "isnd": ("initialization.isnd", "identity", "base.F sounding branch"),
    "iwnd": ("initialization.iwnd", "identity", "base.F base-state wind branch"),
    "itern": (None, "none", "analytic terrain shape; GM1 terrain comes from real data only"),
    "iinit": ("initialization.iinit", "identity", "init3d.F:450-1416 3-D initialization branch; GM1 builds every arm, 0-5 and 7-13 (9, 10 and 12 as the forcings CM1 applies during the run)"),
    "irandp": ("initialization.irandp", "identity", "init3d.F:1446-1600; GM1 additionally requires an explicit initialization.random_seed, because CM1's generator is a property of one libgfortran build and cannot be reproduced"),
    "ibalance": ("initialization.ibalance", "identity", "init3d.F:1735-1917; GM1 builds 0, 1 and 2 (the anelastic buoyancy pressure, poiss.F with singleton.F transcribed)"),
    "iorigin": ("initialization.iorigin", "identity", "param.F:7212-7213: horizontal origin at the domain corner (1) or centre (2)"),
    "axisymm": (None, "none", "axisymmetric geometry; GM1 is Cartesian only"),
    "imove": ("closures.moving_frame", "int-to-bool", "translating reference frame"),
    "iptra": (None, "none", "passive tracers; positive_definite_tracer.py exists but no TOML key selects tracers"),
    "npt": (None, "none", "number of passive tracers; see iptra"),
    "pdtra": (None, "none", "positive-definite tracer advection; see iptra"),
    "iprcl": ("lagrangian.iprcl", "identity", "README.namelist:734-740; src/parcel.F:10"),
    "nparcels": ("lagrangian.nparcels", "identity", "README.namelist:734-740; src/init3d.F:355-411"),
    # ---- param3 ----------------------------------------------------------
    "kdiff2": ("dynamics.kdiff2", "identity", "second-order artificial-diffusion coefficient; read only when idiff>=1 and difforder=2"),
    "kdiff6": ("dynamics.kdiff6", "identity", "sixth-order artificial-diffusion coefficient; read only when idiff>=1 and difforder=6"),
    "fcor": ("dynamics.fcor_s_inv", "identity", "Coriolis parameter in 1/s; read only when icor=1"),
    "kdiv": ("dynamics.kdiv", "identity", "divergence-damper coefficient; psolver=2,3 only"),
    "alph": ("dynamics.alph", "identity", "vertically implicit acoustic off-centering; README.namelist:756-759 default 0.60"),
    "rdalpha": ("closures.upper_damping_rate_s_inv", "identity", "upper damping inverse timescale"),
    "zd": ("closures.upper_damping_depth_m", "cm1-zd-to-depth", "CM1 zd (param.F:8655) is the HEIGHT OF THE BASE of the Rayleigh layer; GM1 upper_damping_depth_m (rust/cm1-state/src/rayleigh_profiles.rs:186-188) is its DEPTH below the model top. depth = ztop - zd, and CM1's shipped supercell deck (zd=15000 on a 20 km top) is GM1's 5000. A naive identity map puts the damper in the wrong place."),
    "xhd": ("closures.xhd", "metres", "lateral Rayleigh sponge width from each boundary (param.F:8736-8776)"),
    "alphobc": ("boundaries.alphobc", "seconds", "nudgeobc time scale (solve2.F:728 forms 1/alphobc)"),
    "umove": ("closures.u_move_m_s", "identity", "frame translation speed, x"),
    "vmove": ("closures.v_move_m_s", "identity", "frame translation speed, y"),
    "do_adapt_move": ("closures.do_adapt_move", "logical-to-bool", "CM1's adaptive moving domain (README.namelist:1789-1794): the storm is found from the smoothed lowest-level pressure and umove/vmove are adjusted to keep it near the domain centre. param.F:704-717 zeroes umove/vmove and raises imove when it is on, so a document that declares it declares those three together."),
    "adapt_move_frq": ("closures.adapt_move_frq", "seconds", "seconds between storm-centre diagnoses (README.namelist:1796-1797); cm1.F:3679 advances adaptmovetim by exactly it and azimavg.F:4431 divides the centre departure by it."),
    "v_t": (None, "none", "prescribed terminal fall speed; no TOML key"),
    "l_h": (None, "none", "horizontal turbulence length scale; no TOML key"),
    "lhref1": (None, "none", "turbulence length-scale reference; no TOML key"),
    "lhref2": (None, "none", "turbulence length-scale reference; no TOML key"),
    "l_inf": (None, "none", "asymptotic turbulence length scale; no TOML key"),
    "ndcnst": ("physics.ndcnst", "identity", "constant cloud-droplet concentration in cm-3, src/morrison.F:329; read by ptype=5 only and admitted at CM1's own 250.0, which is the value the audited ArWen kernel writes as a literal"),
    "nt_c": (None, "none", "constant cloud-droplet concentration for Thompson, cm-3 (README.namelist:816); no TOML key, and ArWen's classic Thompson runs 100"),
    "csound": (None, "none", "reference sound speed for psolver=4/5; GM1 runs psolver 2/3 only"),
    "cstar": ("boundaries.cstar", "identity", "prescribed outward phase speed for irbc=1,2 (bc.F:1964; param.F:722 rewrites anything below 0.1 to 30.0)"),
    # ---- param4/5/6 ------------------------------------------------------
    "stretch_x": ("grid.stretch_x", "int-to-bool", "GM1 admits only stretch_x=0 (false)"),
    "dx_inner": (None, "none", "x-stretch inner spacing; unreachable while stretch_x is refused"),
    "dx_outer": (None, "none", "x-stretch outer spacing; unreachable while stretch_x is refused"),
    "nos_x_len": (None, "none", "x-stretch uniform-region length; unreachable"),
    "tot_x_len": (None, "none", "x-stretch total length; unreachable"),
    "stretch_y": ("grid.stretch_y", "int-to-bool", "GM1 admits only stretch_y=0 (false)"),
    "dy_inner": (None, "none", "y-stretch inner spacing; unreachable"),
    "dy_outer": (None, "none", "y-stretch outer spacing; unreachable"),
    "nos_y_len": (None, "none", "y-stretch uniform-region length; unreachable"),
    "tot_y_len": (None, "none", "y-stretch total length; unreachable"),
    "stretch_z": ("grid.stretch_z", "identity", "GM1 admits 0, 1, 3 and 4; CM1's stretch_z=2 is refused"),
    "ztop": ("grid.ztop_m", "metres", "model top; read by stretch_z=1"),
    "str_bot": ("grid.str_bot_m", "metres", "stretch layer base; read by stretch_z=1"),
    "str_top": ("grid.str_top_m", "metres", "stretch layer top; read by stretch_z=1"),
    "dz_bot": ("grid.dz_bot_m", "metres", "spacing at the stretch layer base; read by stretch_z=1"),
    "dz_top": ("grid.dz_top_m", "metres", "spacing at the stretch layer top; read by stretch_z=1"),
    # ---- param7 (DNS) ----------------------------------------------------
    "bc_temp": (None, "none", "DNS temperature boundary condition; GM1 has no DNS lane in its TOML"),
    "ptc_top": (None, "none", "DNS top potential temperature; no TOML key"),
    "ptc_bot": (None, "none", "DNS bottom potential temperature; no TOML key"),
    "viscosity": (None, "none", "DNS molecular viscosity; dns_temperature_diffusion.py exists but no TOML key selects it"),
    "pr_num": (None, "none", "DNS Prandtl number; no TOML key"),
    # ---- param8 (user variables) ----------------------------------------
    **{f"var{i}": (None, "none", "CM1 user-defined scratch variable read by hand-edited test cases only; no GM1 equivalent by design")
       for i in range(1, 21)},
    # ---- param11 (radiation / clock) ------------------------------------
    "radopt": ("radiation.radopt", "identity", "radiation scheme; radopt=2 resolves to ArWen legacy-RRTMG with its sha256-admitted AER tables and refuses at preflight naming the surface state GM1 does not carry; radopt=1 (NASA-Goddard) has no ArWen counterpart"),
    "dtrad": ("radiation.dtrad_s", "identity", "radiation call interval; GM1 carries CM1 own radtim walk (src/cm1.F:3196-3199,3708-3713)"),
    "ctrlat": ("radiation.ctrlat", "identity", "domain centre latitude; required as a group with radopt>=1 and refused with radopt=0, because GM1 idealized lane is otherwise location-free"),
    "ctrlon": ("radiation.ctrlon", "identity", "domain centre longitude; same group rule as ctrlat"),
    "year": ("radiation.year", "identity", "simulation start year; CM1 turns the start date into the Julian day the solar declination needs"),
    "month": ("radiation.month", "identity", "simulation start month; CM1 turns the start date into the Julian day the solar declination needs"),
    "day": ("radiation.day", "identity", "simulation start day; CM1 turns the start date into the Julian day the solar declination needs"),
    "hour": ("radiation.hour", "identity", "simulation start hour; CM1 turns the start date into the Julian day the solar declination needs"),
    "minute": ("radiation.minute", "identity", "simulation start minute; CM1 turns the start date into the Julian day the solar declination needs"),
    "second": ("radiation.second", "identity", "simulation start second; CM1 turns the start date into the Julian day the solar declination needs"),
    # ---- param12 (surface) ----------------------------------------------
    "isfcflx": ("closures.isfcflx", "identity", "surface flux switch"),
    "sfcmodel": ("closures.sfcmodel", "identity", "surface-layer scheme"),
    "oceanmodel": ("surface.oceanmodel", "identity", "ocean/water surface model; GM1 admits 1 (fixed SST) and refuses 2 by name"),
    "initsfc": ("surface.initsfc", "identity", "initial surface-condition branch"),
    "tsk0": ("surface.tsk0_k", "identity", "initial skin temperature in kelvin"),
    "tmn0": ("surface.tmn0_k", "identity", "initial deep-soil temperature in kelvin"),
    "xland0": ("surface.xland0", "identity", "land (1.0) or water (2.0) flag"),
    "lu0": ("surface.lu0", "identity", "LANDUSE.TBL land-use category"),
    "season": ("surface.season", "identity", "LANDUSE.TBL season index"),
    "cecd": ("closures.cecd", "identity", "surface exchange-coefficient formulation"),
    "pertflx": (None, "none", "perturbed surface fluxes; no TOML key"),
    "cnstce": ("closures.cnstce", "identity", "constant enthalpy exchange coefficient for isfcflx=1 over water with cecd=1"),
    "cnstcd": ("closures.cnstcd", "identity", "constant drag coefficient"),
    "isftcflx": (None, "none", "high-wind exchange-coefficient option; no TOML key"),
    "iz0tlnd": (None, "none", "thermal roughness over land; no TOML key"),
    "oml_hml0": (None, "none", "ocean mixed-layer depth; unreachable with oceanmodel"),
    "oml_gamma": (None, "none", "ocean lapse rate; unreachable with oceanmodel"),
    "set_flx": (None, "none", "prescribed surface fluxes; prescribed_surface_flux.py exists but no TOML key selects it"),
    "cnst_shflx": (None, "none", "prescribed sensible heat flux; unreachable with set_flx"),
    "cnst_lhflx": (None, "none", "prescribed latent heat flux; unreachable with set_flx"),
    "set_znt": ("closures.set_znt", "identity", "impose the roughness length instead of deriving it"),
    "cnst_znt": ("closures.cnst_znt", "identity", "imposed roughness length in metres"),
    "set_ust": ("closures.set_ust", "identity", "impose the friction velocity instead of deriving it"),
    "cnst_ust": ("closures.cnst_ust", "identity", "imposed friction velocity in m/s"),
    "ramp_sgs": ("closures.ramp_sgs", "identity", "README.namelist:1081-1082; src/turb.F:240-262 ramps cme/ce1/ce2/csm in over ramp_time (cuda_cartesian_sgs_ramp); optional pair, absent is src/cm1.F:355's compiled 0"),
    "ramp_time": ("closures.ramp_time", "seconds", "README.namelist:1085-1086; src/param.F:750 floors it at 1e-10 s; read only when ramp_sgs=1"),
    "t2p_avg": ("closures.t2p_avg", "identity", "two-part sgsmodel=3/4 averaging (README.namelist:1089-1092), read only when dot2p; 1 is ported and is CM1's shipped value"),
    # ---- param17 (LES subdomain) ----------------------------------------
    "les_subdomain_shape": (None, "none", "LES subdomain nesting; no TOML key"),
    "les_subdomain_xlen": (None, "none", "LES subdomain length; no TOML key"),
    "les_subdomain_ylen": (None, "none", "LES subdomain length; no TOML key"),
    "les_subdomain_dlen": (None, "none", "LES subdomain length; no TOML key"),
    "les_subdomain_trnslen": (None, "none", "LES subdomain transition length; no TOML key"),
    # ---- param18 (turbulence recycling) ---------------------------------
    "do_recycle_w": (None, "none", "turbulence recycling; eddy_recycle.F has no GM1 counterpart"),
    "do_recycle_s": (None, "none", "turbulence recycling; no GM1 counterpart"),
    "do_recycle_e": (None, "none", "turbulence recycling; no GM1 counterpart"),
    "do_recycle_n": (None, "none", "turbulence recycling; no GM1 counterpart"),
    "recycle_width_dx": (None, "none", "recycling geometry; no GM1 counterpart"),
    "recycle_depth_m": (None, "none", "recycling geometry; no GM1 counterpart"),
    "recycle_cap_loc_m": (None, "none", "recycling geometry; no GM1 counterpart"),
    "recycle_inj_loc_m": (None, "none", "recycling geometry; no GM1 counterpart"),
    # ---- param19 (large-scale nudging) ----------------------------------
    "do_lsnudge": (None, "none", "large-scale nudging; large_scale_nudging.py exists but no TOML key selects it"),
    "do_lsnudge_u": (None, "none", "large-scale nudging component; unreachable"),
    "do_lsnudge_v": (None, "none", "large-scale nudging component; unreachable"),
    "do_lsnudge_th": (None, "none", "large-scale nudging component; unreachable"),
    "do_lsnudge_qv": (None, "none", "large-scale nudging component; unreachable"),
    "lsnudge_tau": (None, "none", "nudging timescale; unreachable"),
    "lsnudge_start": (None, "none", "nudging window; unreachable"),
    "lsnudge_end": (None, "none", "nudging window; unreachable"),
    "lsnudge_ramp_time": (None, "none", "nudging ramp; unreachable"),
    # ---- param20 (immersed boundaries) ----------------------------------
    "do_ib": (None, "none", "immersed boundary method; no GM1 counterpart"),
    "ib_init": (None, "none", "immersed boundary shape; no GM1 counterpart"),
    "top_cd": (None, "none", "immersed boundary drag; no GM1 counterpart"),
    "side_cd": (None, "none", "immersed boundary drag; no GM1 counterpart"),
    # ---- param21 (idealized hurricane) ----------------------------------
    "hurr_vg": (None, "none", "idealized hurricane vortex; GM1 has no axisymmetric/hurricane initializer"),
    "hurr_rad": (None, "none", "idealized hurricane vortex; no GM1 counterpart"),
    "hurr_vgpl": (None, "none", "idealized hurricane vortex; no GM1 counterpart"),
    "hurr_rotate": (None, "none", "idealized hurricane vortex; no GM1 counterpart"),
}

# ---- param13 (parcel attributes) and the pdcomp selector -----------------
# These live in CM1's non-physical groups, but they are not "output
# formatting": each one decides whether a physical quantity is recorded along
# the trajectory at all, and the port carries every one of them under CM1's
# own name.  param.F:6257-6282 resolves them against the run's other words and
# param.F:6385-6500 builds npvals from what it resolved.
_PARCEL_SWITCHES = (
    "prcl_th", "prcl_t", "prcl_prs", "prcl_ptra", "prcl_q", "prcl_nc",
    "prcl_km", "prcl_kh", "prcl_tke", "prcl_dbz", "prcl_b", "prcl_vpg",
    "prcl_vort", "prcl_rho", "prcl_qsat", "prcl_sfc",
)
MAP.update({
    name: (f"lagrangian.{name}", "identity",
           "src/input.F:330-333 &param13; src/param.F:6257-6282 resolves it "
           "against the run's other words and src/param.F:6385-6500 builds the "
           "attribute offsets from the result")
    for name in _PARCEL_SWITCHES
})
MAP["output_pdcomp"] = (
    "lagrangian.output_pdcomp", "identity",
    "run/namelist.input:277; src/param.F:727-729; src/pdcomp.F:9",
)

#: CM1 keys that change no prognostic state: stdout formatting, timer reports,
#: MPI decomposition, and write cadences for products GM1 does not produce.
#: They are still in the map -- silence is not an option -- but a differential
#: receipt must not report them as physics the two codes disagree about.
NO_EFFECT_ON_STATE = frozenset({
    "ppnode", "timeformat", "timestats", "procfiles", "outunits", "run_time",
    "statfrq", "prclfrq", "iprcl", "nparcels", "rstnum",
    "output_pdcomp", *_PARCEL_SWITCHES,
})

#: What GM1 does for a CM1 physical key it has no TOML word for.  The
#: translator reads this, not CM1's distributed deck: a key the deck sets to
#: CM1's own shipped value is still a difference when GM1 runs something else
#: (every shipped deck sets ramp_sgs=1, which GM1 did not run until the key
#: existed, and the translator reported nothing).
#:
#: ``gm1_runs``  -- the one value GM1 always integrates; any other effective
#:                  value is a difference the translation must name.
#: ``read_when`` -- the CM1 statements that read the key, as alternatives
#:                  (each a {selector: [values]} that must all hold).  When
#:                  none holds, CM1 does not read the key and it is inert on
#:                  both sides; when one holds, CM1 reads a value GM1 cannot
#:                  express and the translation names it.  An empty list
#:                  means no statement of the r21.1 source reads the key.
#: A key with both is read by CM1 only under ``read_when``, and there GM1
#: runs ``gm1_runs``.  A key with neither is named whenever it is set, because
#: what GM1 does for it is not recorded.
UNMAPPED_BEHAVIOUR: dict[str, dict[str, Any]] = {
    "axisymm": {"gm1_runs": 0, "anchor": "src/adv.F:113; GM1 is Cartesian only"},
    "bcturbs": {"gm1_runs": 1, "anchor": "src/turbtend.F:501-579; GM1 pins bcturbs=1"},
    "tconfig": {"gm1_runs": 1, "anchor": "src/solve1.F:321-333; GM1 pins tconfig=1"},
    "idiss": {"gm1_runs": 1, "anchor": "src/solve1.F:744,919; the Cartesian closure always applies dissipative heating"},
    "efall": {"gm1_runs": 0, "anchor": "src/mp_driver.F:96,570; no Cartesian component runs getefall"},
    "horizturb": {"gm1_runs": 0, "anchor": "src/turb.F:1550; GM1 has no turbparam_horiz"},
    "iptra": {"gm1_runs": 0, "anchor": "src/solve2.F:1906-2066; GM1 carries no passive tracers"},
    "pertflx": {"gm1_runs": 0, "anchor": "src/turb.F:309"},
    "set_flx": {"gm1_runs": 0, "anchor": "src/turb.F; GM1 has no prescribed-flux key"},
    "do_ib": {"gm1_runs": False, "anchor": "src/cm1.F; GM1 has no immersed boundaries"},
    "do_lsnudge": {"gm1_runs": False, "anchor": "src/solve1.F:278; GM1 has no nudging key"},
    "do_recycle_w": {"gm1_runs": False, "anchor": "src/eddy_recycle.F"},
    "do_recycle_s": {"gm1_runs": False, "anchor": "src/eddy_recycle.F"},
    "do_recycle_e": {"gm1_runs": False, "anchor": "src/eddy_recycle.F"},
    "do_recycle_n": {"gm1_runs": False, "anchor": "src/eddy_recycle.F"},
    "bc_temp": {"read_when": [{"cm1setup": [3]}], "anchor": "src/turbtend.F:523-533 (DNS only)"},
    "pr_num": {"read_when": [{"cm1setup": [3]}], "anchor": "src/cm1.F:1709-1711, src/turbtend.F:466"},
    "ptc_bot": {"read_when": [{"cm1setup": [3]}], "anchor": "src/turbtend.F:466-478"},
    "ptc_top": {"read_when": [{"cm1setup": [3]}], "anchor": "src/turbtend.F:466-478"},
    "viscosity": {"read_when": [{"cm1setup": [3]}], "anchor": "src/cm1.F:1709-1711"},
    "cnst_shflx": {"read_when": [{"set_flx": [1]}], "anchor": "README.namelist set_flx"},
    "cnst_lhflx": {"read_when": [{"set_flx": [1]}], "anchor": "README.namelist set_flx"},
    "csound": {"read_when": [{"psolver": [6, 7]}], "anchor": "src/misclibs.F:869-879"},
    "ib_init": {"read_when": [{"do_ib": [True]}], "anchor": "immersed boundaries"},
    "top_cd": {"read_when": [{"do_ib": [True]}], "anchor": "immersed boundaries"},
    "side_cd": {"read_when": [{"do_ib": [True]}], "anchor": "immersed boundaries"},
    **{
        name: {"read_when": [{"do_lsnudge": [True]}], "anchor": "src/solve1.F:278-281"}
        for name in (
            "do_lsnudge_u", "do_lsnudge_v", "do_lsnudge_th", "do_lsnudge_qv",
            "lsnudge_tau", "lsnudge_start", "lsnudge_end", "lsnudge_ramp_time",
        )
    },
    **{
        name: {
            "read_when": [{side: [True]} for side in ("do_recycle_w", "do_recycle_s", "do_recycle_e", "do_recycle_n")],
            "anchor": "src/eddy_recycle.F",
        }
        for name in ("recycle_width_dx", "recycle_depth_m", "recycle_cap_loc_m", "recycle_inj_loc_m")
    },
    **{
        name: {"read_when": [{"stretch_x": [1, 2]}], "anchor": "src/param.F stretch_x"}
        for name in ("dx_inner", "dx_outer", "nos_x_len", "tot_x_len")
    },
    **{
        name: {"read_when": [{"stretch_y": [1, 2]}], "anchor": "src/param.F stretch_y"}
        for name in ("dy_inner", "dy_outer", "nos_y_len", "tot_y_len")
    },
    "hurr_vg": {"read_when": [{"iwnd": [8]}], "anchor": "src/base.F:2525-2545"},
    "hurr_vgpl": {"read_when": [{"iwnd": [8]}], "anchor": "src/base.F:2844-2845"},
    "hurr_rad": {"read_when": [{"iwnd": [8]}, {"lspgrad": [3]}], "anchor": "src/base.F:2845, src/solve1.F:633-646"},
    "hurr_rotate": {"read_when": [{"lspgrad": [3]}], "anchor": "src/param.F:8971-8974"},
    "iautoc": {"read_when": [{"ptype": [2, 4]}], "anchor": "src/goddard.F:578"},
    # CM1 reads nt_c only for Thompson; ArWen's classic Thompson (mp_physics
    # = 8) pins Nt_c = 100e6 m-3 (gpuwm/core/thompson_aerosol_constants.py:25,
    # module_mp_thompson.F:88), which is nt_c = 100 cm-3.
    "nt_c": {"read_when": [{"ptype": [3]}], "gm1_runs": 100.0,
             "anchor": "src/param.F:1711, :9130 thompson_init; ArWen NT_C = 100e6 m-3"},
    "v_t": {"read_when": [{"ptype": [6]}], "anchor": "src/mp_driver.F:699-754"},
    "npt": {"read_when": [{"iptra": [1]}], "anchor": "src/solve2.F:2066"},
    "pdtra": {"read_when": [{"iptra": [1]}], "anchor": "src/solve2.F:1906,2066"},
    "itern": {"read_when": [{"terrain_flag": [True]}], "anchor": "src/init_terrain.F:64-83"},
    "isftcflx": {"read_when": [{"sfcmodel": [2, 3, 4, 6, 7]}], "anchor": "src/turb.F:726-746"},
    "iz0tlnd": {"read_when": [{"sfcmodel": [2, 3, 4, 6, 7]}], "anchor": "src/turb.F:726-746"},
    "l_h": {"read_when": [{"cm1setup": [2, 4], "horizturb": [1]}], "anchor": "src/turb.F:1550, :3683-3711"},
    "lhref1": {"read_when": [{"cm1setup": [2, 4], "horizturb": [1]}], "anchor": "src/turb.F:1550, :3683-3711"},
    "lhref2": {"read_when": [{"cm1setup": [2, 4], "horizturb": [1]}], "anchor": "src/turb.F:1550, :3683-3711"},
    "l_inf": {"read_when": [{"cm1setup": [2, 4], "ipbl": [2]}], "anchor": "src/turb.F:1560, :3898-3901"},
    **{
        name: {"read_when": [{"cm1setup": [4]}], "anchor": "src/init3d.F:2621-2650"}
        for name in (
            "les_subdomain_shape", "les_subdomain_xlen", "les_subdomain_ylen",
            "les_subdomain_dlen", "les_subdomain_trnslen",
        )
    },
    "oml_hml0": {"read_when": [{"oceanmodel": [2]}], "anchor": "src/init_surface.F:330"},
    "oml_gamma": {"read_when": [{"oceanmodel": [2]}], "anchor": "src/oml.F"},
    **{
        f"var{n}": {"read_when": [], "anchor": "read by no statement of the r21.1 source outside input.F and its echo"}
        for n in range(1, 21)
    },
}


#: GM1 TOML keys that CM1 has no namelist word for.  The reverse direction of
#: the map, so both lists are checkable.
GM1_ONLY: dict[str, str] = {
    # CM1 r21.1 has no nesting and no ensemble concept; both are declared GM1
    # extensions, recorded in records/CM1-NEST-SEAM.md.  They map to no CM1
    # namelist word because there is none to map to.
    "nest.refinement_ratio": "grid refinement ratio of the child mesh; CM1 has one domain.",
    "nest.time_refinement_ratio": "child large-step refinement; CM1 has one clock.",
    "nest.parent_i0": "parent scalar index of the child's SW corner.",
    "nest.parent_j0": "parent scalar index of the child's SW corner.",
    "nest.parent_span_x": "whole parent cells the child covers in x.",
    "nest.parent_span_y": "whole parent cells the child covers in y.",
    "nest.spec_zone": "Davies specified-boundary width, in child cells.",
    "nest.relax_zone": "Davies relaxation width, in child cells.",
    "nest.feedback": "one-way or two-way child-to-parent feedback.",
    "nest.acoustic_substeps": "the child's own nsound; CM1 derives one domain's.",
    "nest.follow_storm": "whether the running nest moves with the tracked vortex.",
    "nest.follow_interval_s": "tracker cadence (WRF vortex_interval).",
    "nest.follow_max_speed_m_s": "search-window speed bound (WRF max_vortex_speed); parent cells per move are derived.",
    "nest.follow_tracker": "wrf-surface-circulation (child's surface vortex) or a CM1 pressure-centre target.",
    "nest.follow_smoothing_radius_m": "half-width of the surface tracker's circulation box.",
    "nest.follow_corral_m": "minimum clearance kept from the parent boundary (WRF corral_dist).",
    "nest.follow_on_edge": "stop or hold when the surface tracker's sample is out of reach (window edge, no circulation, centroid too far) or its move enters the corral.",
    "nest.follow_hold_limit": "consecutive held tracker samples before hold relocates the centre.",
    "nest.follow_relocation_radius_m": "half-width hold's relocation searches around the kept centre.",
    # [nest.tornado_criterion] (cm1core.nest_tornado_criterion): CM1 r21.1 has
    # no nesting and makes no tornado claim; the sample formulas are solve3.F's.
    "nest.tornado_criterion": "fixed tornado criterion sampled on the live child every parent step; CM1 makes no tornado claim.",
    # A child that declares its own vertical deck (c9063b6, recorded in
    # records/CM1-NEST-SEAM.md 5.6 by acdcbcf).
    "nest.child_dz_bot_m": "child's own param6 dz_bot (stretch_z=1); the parent's top is kept.",
    "nest.child_dz_top_m": "child's own param6 dz_top (stretch_z=1); the parent's top is kept.",
    "nest.child_levels_m": "child's own explicit levels (stretch_z=3 or 4).",
    "nest.child_nz": "child's own level count when it declares its own vertical deck.",
    "nest.child_str_bot_m": "child's own param6 str_bot (stretch_z=1).",
    "nest.child_str_top_m": "child's own param6 str_top (stretch_z=1).",
    "nest.child_stretch_z": "child's own stretch_z branch (1, 3 or 4) over the parent's model top.",
    # [chain] drives one Cartesian domain from a saved ArWen history (WRF's
    # ndown lane, cm1core.arwen_chain, records/GM1-ARWEN-CHAIN.md).  CM1 r21.1
    # has no lateral boundary forcing, so none of these has a namelist word.
    "chain.history_directory": "directory of the saved ArWen (WRF) history frames the domain is driven from.",
    "chain.domain_id": "ArWen domain whose history frames are read.",
    "chain.start_time": "first history frame: the initial state and first boundary time.",
    "chain.end_time": "last history frame the lateral boundary tendencies run to.",
    "chain.parent_i0": "parent-history scalar index of the domain's SW corner.",
    "chain.parent_j0": "parent-history scalar index of the domain's SW corner.",
    "chain.refinement_ratio": "horizontal refinement from the history grid to this domain.",
    "chain.spec_zone": "Davies specified-boundary width, in domain cells (WRF spec_zone).",
    "chain.relax_zone": "Davies relaxation width, in domain cells (WRF relax_zone).",
    "chain.terrain_relief_limit_m": "largest terrain relief the flat Cartesian domain admits from the history window.",
    "chain.map_factor_departure_limit": "largest map-factor departure from 1 the Cartesian domain admits.",
    "chain.moisture": "which of the history's moist fields drive the domain: every species of GM1's scheme (all-species) or water vapour alone (vapour-only, WRF have_bcs_moist = .false.).",
    "chain.prepared_directory": "output of tools/arwen_chain_prepare.py: sounding and boundary tables.",
    "ensemble.member": "ensemble member index; CM1 runs one member per deck.",
    "ensemble.seed": "seed for the reproducible irandp perturbation stream.",
    "ensemble.theta_amplitude_k": "amplitude of the member's theta perturbation.",
    # getcape.F's own parcel controls are source parameters in CM1, not
    # namelist words: getcape.F:60,65-68,70,73-77 sets them in the routine.
    "diagnostics.parcel_source": "getcape.F:65-68 source parcel; a source parameter in CM1.",
    "diagnostics.adiabat": "getcape.F:73-77 adiabat; a source parameter in CM1.",
    "diagnostics.mixed_layer_depth_m": "getcape.F:70 ml_depth; a source parameter in CM1.",
    "diagnostics.pressure_increment_pa": "getcape.F:60 pinc; a source parameter in CM1.",
    "diagnostics.storm_motion": "which Bunkers mover the t=0 SRH is taken against; CM1 r21.1 computes no storm motion.",
    "diagnostics.tornado_report": "GM1's own statement of what the run produced; CM1 r21.1 makes no tornado claim.",
    "run.lane": "GM1 lane selector; separates the dry validation lane, the sealed product profiles and the configurable Cartesian runtime. CM1 has one executable and no lane concept.",
    "grid.levels_m": "explicit user level list for stretch_z=3/4; CM1 reads these from a separate file rather than the namelist.",
    "boundaries.bottom": "a second GM1 spelling of CM1's bbc; config.py:1228 refuses a disagreement between the two.",
    "physics.component": "names the ArWen microphysics component behind the declared seam; CM1's ptype is an integer into its own in-tree schemes.",
    "runtime.acoustic_substeps": "CM1 derives nsound from dtl and dtsmall (misclibs.F:883-885) rather than accepting it; GM1 declares it.",
    "runtime.tile_nx": "CUDA tile shape; a device execution parameter with no CM1 equivalent.",
    "runtime.tile_ny": "CUDA tile shape; a device execution parameter with no CM1 equivalent.",
    "runtime.warmup_steps": "CUDA timing-harness warmup; no CM1 equivalent.",
    "runtime.memory_budget_mib": "device memory budget; no CM1 equivalent.",
    "runtime.capacity_profile": "device capacity profile identity; no CM1 equivalent.",
    # init3d.F's named parameter blocks are source edits in CM1, not namelist
    # words: README.namelist:679-690 says "additional variables need to be set
    # in init3d.F file".  GM1 makes them configuration under CM1's own names,
    # so there is no CM1 namelist key to map them to.
    "initialization.bubble_center_x_m": "init3d.F:456 ric, the perturbation centre in x; a source edit in CM1, configuration here.",
    "initialization.bubble_center_y_m": "init3d.F:457 rjc, the perturbation centre in y; a source edit in CM1, configuration here.",
    "initialization.zc_m": "init3d.F:458 zc, the perturbation centre height; a source edit in CM1.",
    "initialization.bubble_first_center_y_m": "init3d.F:548-550 writes the three iinit=3 centres as literals 3000/33000/63000; this is the first of them, a source edit in CM1.",
    "initialization.half_width_m": "init3d.F:1356 writes the Skamarock-Klemp envelope half width as the literal 5000.0; a source edit in CM1.",
    "physics.dbz_level_m": "the single height whose reflectivity plane is published beside the column maximum; CM1 users slice the 3-D dbz field themselves, so there is no namelist word.",
    "run.irst": "README.namelist:227 irst, spelled in GM1's [run] table because the restart pair is a run-level lifecycle choice here.",
    "run.rstnum": "README.namelist:229 rstnum, spelled in GM1's [run] table beside irst.",
    "run.restart_reset_frqtim": "README.namelist:1505-1510 restart_reset_frqtim, spelled in GM1's [run] table beside irst.",
    "runtime.pdefq_launch": "which of two raw-word identical launch layouts runs CM1's pre-WSM6 positive-definite repair; a device execution parameter with no CM1 equivalent.",
    "runtime.physics_block_nx": "the x extent of one WSM6 call; a device decomposition parameter with no CM1 equivalent.",
    "runtime.physics_block_ny": "the y extent of one WSM6 call; a device decomposition parameter with no CM1 equivalent.",
    "initialization.bhrad_m": "init3d.F:459 bhrad, the horizontal radius; a source edit in CM1.",
    "initialization.bvrad_m": "init3d.F:460 bvrad, the vertical radius; a source edit in CM1.",
    "initialization.bptpert_k": "init3d.F:461 bptpert, the perturbation amplitude; a source edit in CM1.",
    "initialization.maintain_rh": "init3d.F:465 maintain_rh; a source edit in CM1.",
    "initialization.nbub": "init3d.F:525 nbub, the number of bubbles in the iinit=3 line; a source edit in CM1.",
    "initialization.bubble_spacing_y_m": "the y step between the iinit=3 bubbles; init3d.F:527-529 writes its three centres as literals.",
    "initialization.zdep_m": "init3d.F:494 zdep, the iinit=2 cold-pool depth; a source edit in CM1.",
    "initialization.random_line_amplitude_k": "init3d.F:1275, the iinit=8 random amplitude; a source edit in CM1.",
    "initialization.hurr_vg_m_per_s": "run/namelist.input:403 hurr_vg, read by base.F:2534's iwnd=8 branch; CM1 spells it in param21 rather than param6.",
    "initialization.hurr_rotate_deg": "run/namelist.input:406 hurr_rotate, which param.F:8973 turns into base.F:2545's hurr_angle.",
    "initialization.thec_mb_k": "base.F:267 thec_mb, the moist benchmark's wet-equivalent potential temperature (isnd = 4); a source word in CM1, configuration here.",
    "initialization.qt_mb": "base.F:268 qt_mb, the moist benchmark's total water (isnd = 4, and init3d.F:625-645's iinit = 4); a source word in CM1, configuration here.",
    "initialization.tsk0_k": "CM1's namelist tsk0 as base.F:1204 (isnd = 11) and :1849 (isnd = 21) read it, for a run with no [surface] table; with one, tsk0 is surface.tsk0_k.",
    "initialization.random_seed": "the seed for GM1's own irandp generator (random_generator = \"gm1\"). CM1 has no equivalent key: init3d.F:427-440 seeds gfortran's generator from a fixed scramble, which random_generator = \"cm1\" reproduces.",
    "initialization.random_generator": "which stream iinit=8 and irandp draw from: \"cm1\" is CM1's own numbers (libgfortran's xoshiro256** random_number as init3d.F:427-440 seeds it, checked word for word against gfortran 15.2), \"gm1\" GM1's seeded stream. CM1 has no key: its stream is fixed.",
    "initialization.xc_wnudge": "init3d.F:1372 xc_wnudge, the updraft-nudging centre in x (default centerx); a source edit in CM1.",
    "initialization.xr_wnudge": "init3d.F:1373 xr_wnudge, the nudging radius in x; a source edit in CM1.",
    "initialization.yc_wnudge": "init3d.F:1375 yc_wnudge, the nudging centre in y (default centery); a source edit in CM1.",
    "initialization.yr_wnudge": "init3d.F:1376 yr_wnudge, the nudging radius in y; a source edit in CM1.",
    "initialization.zc_wnudge": "init3d.F:1378 zc_wnudge, the nudging centre height; a source edit in CM1.",
    "initialization.zr_wnudge": "init3d.F:1379 zr_wnudge, the nudging radius in z; a source edit in CM1.",
    "initialization.alpha_wnudge": "init3d.F:1381 alpha_wnudge, the nudging rate (1/s); a source edit in CM1.",
    "initialization.wmax_wnudge": "init3d.F:1382 wmax_wnudge, the w the nudging pulls toward (m/s); a source edit in CM1.",
    "initialization.t1_wnudge": "init3d.F:1384 t1_wnudge, when the nudging starts to ramp down (s); a source edit in CM1.",
    "initialization.t2_wnudge": "init3d.F:1385 t2_wnudge, when the nudging stops (s); a source edit in CM1.",
    "initialization.dmax": "init3d.F:1309 Dmax, the forced-convergence divergence (1/s); a source edit in CM1.",
    "initialization.zdeep": "init3d.F:1310 zdeep, the depth of forced convergence (m); a source edit in CM1.",
    "initialization.lamx": "init3d.F:1311 lamx, Loftus et al.'s lambda_x (m); a source edit in CM1.",
    "initialization.lamy": "init3d.F:1312 lamy, Loftus et al.'s lambda_y (m); a source edit in CM1.",
    "initialization.xcent": "init3d.F:1313 xcent, the convergence centre in x (m); a source edit in CM1.",
    "initialization.ycent": "init3d.F:1314 ycent, the convergence centre in y (m); a source edit in CM1.",
    "initialization.convtime": "init3d.F:1315 convtime, how long the convergence is applied (s); a source edit in CM1.",
    "initialization.xc_uforce": "init3d.F:1339 xc_uforce, the momentum-forcing centre in x (default the domain centre); a source edit in CM1.",
    "initialization.xr_uforce": "init3d.F:1340 xr_uforce, the forcing radius in x; a source edit in CM1.",
    "initialization.zr_uforce": "init3d.F:1341 zr_uforce, the forcing depth scale; a source edit in CM1.",
    "initialization.alpha_uforce": "init3d.F:1342 alpha_uforce, the forcing's peak (m/s/s); a source edit in CM1.",
    "initialization.t1_uforce": "init3d.F:1343 t1_uforce, when the forcing starts to ramp down (s); a source edit in CM1.",
    "initialization.t2_uforce": "init3d.F:1344 t2_uforce, when the forcing stops (s); a source edit in CM1.",
    "initialization.imoist": "a second GM1 spelling of CM1's imoist, read by the initializer. physics.component carries the same selector for the runtime; config.py refuses a disagreement.",
    # ---- [lagrangian]: what CM1 r21.1 has no namelist word for -----------
    "lagrangian.nest_child": "GM1 extension: parcels integrated inside a live nest child (cm1core.nest_parcels). CM1 r21.1 has no nesting, so it has no word for a child seeding; the per-step arithmetic is src/parcel.F's.",
    "lagrangian.nest_child_window": "GM1 extension: a vortex-following sub-volume of a live nest child, written every child step with the model's own advecting velocity words (cm1core.nest_child_window). CM1 r21.1 has no nesting and its own high-frequency writer src/hifrq.F:21 writes the whole domain at a fixed place, from two words src/cm1.F:2645-2647 hard-codes, so CM1 has no name for either the window or its cadence.",
    "lagrangian.dohifrq": "CM1 r21.1 hard-codes dohifrq=.false. at src/cm1.F:2645 and reads it from no namelist; turning hifrq.F on there means editing and rebuilding CM1. GM1 makes it configuration, under CM1's own name.",
    "lagrangian.hifrqfrq_s": "CM1 r21.1 hard-codes hifrqfrq=3600.0 at src/cm1.F:2647 and reads it from no namelist.",
    "lagrangian.hifrq_i0": "GM1 extension: hifrq.F writes the whole domain, so CM1 has no window words. A tornado occupies a few km of a domain hundreds of km across.",
    "lagrangian.hifrq_i1": "GM1 extension: see lagrangian.hifrq_i0.",
    "lagrangian.hifrq_j0": "GM1 extension: see lagrangian.hifrq_i0.",
    "lagrangian.hifrq_j1": "GM1 extension: see lagrangian.hifrq_i0.",
    "lagrangian.hifrq_k0": "GM1 extension: see lagrangian.hifrq_i0.",
    "lagrangian.hifrq_k1": "GM1 extension: see lagrangian.hifrq_i0.",
    "lagrangian.direction": "GM1 extension: backward trajectories; CM1's parcel_driver integrates forward only.",
    "lagrangian.seeding": "GM1 extension: CM1 hard-codes one 60x60x10 seeding grid at src/init3d.F:381-387 with no namelist control; seeding='grid' with GM1's default parameters reproduces it exactly.",
    "lagrangian.seed_nx": "GM1 extension: src/init3d.F:381's hard-coded 60.",
    "lagrangian.seed_ny": "GM1 extension: src/init3d.F:382's hard-coded 60.",
    "lagrangian.seed_nz": "GM1 extension: src/init3d.F:380's hard-coded 10.",
    "lagrangian.seed_dx_m": "GM1 extension: src/init3d.F:385's hard-coded 2000.0 m.",
    "lagrangian.seed_dy_m": "GM1 extension: src/init3d.F:386's hard-coded 2000.0 m.",
    "lagrangian.seed_dz_m": "GM1 extension: src/init3d.F:387's hard-coded 1000.0 m.",
    "lagrangian.seed_x0_m": "GM1 extension: offset of the seeding origin from minx.",
    "lagrangian.seed_y0_m": "GM1 extension: offset of the seeding origin from miny.",
    "lagrangian.seed_z0_m": "GM1 extension: offset of the seeding origin from zh(1).",
    "lagrangian.seed_vorticity_threshold_s_inv": "GM1 extension: seed only where |zeta| exceeds this, a model-time decision CM1's namelist cannot express.",
    "lagrangian.vorticity_budget": "GM1 extension: CM1 r21.1 computes no vorticity budget anywhere.",
    "lagrangian.vorticity_budget_frq_s": "GM1 extension: see lagrangian.vorticity_budget.",
    "lagrangian.vorticity_budget_history": "GM1 extension: CM1 r21.1 has no vorticity budget; this publishes calcvort of every output_*budget momentum term, and the stretching/tilting decomposition, as history carriers.",
    "lagrangian.parcel_vorticity_budget": "GM1 extension: CM1's parcel.F records no vorticity budget; this samples the vorticity budget along every parcel after CM1's attributes.",
    "lagrangian.circuits": "GM1 extension: CM1 r21.1 has no material-circuit or circulation diagnostic.",
    "lagrangian.circuit_parcels": "GM1 extension: see lagrangian.circuits.",
    "lagrangian.circuit_radius_m": "GM1 extension: see lagrangian.circuits.",
    "lagrangian.circuit_center_x_m": "GM1 extension: see lagrangian.circuits.",
    "lagrangian.circuit_center_y_m": "GM1 extension: see lagrangian.circuits.",
    "lagrangian.circuit_center_z_m": "GM1 extension: see lagrangian.circuits.",
    "lagrangian.circuit_spacing_z_m": "GM1 extension: see lagrangian.circuits.",
}

#: The non-physical CM1 groups, summarised by family.
NON_PHYSICAL_FAMILIES = {
    "param10": "domain-statistics selectors; GM1 has no statpack writer.",
    "param13": "parcel-attribute selectors; all sixteen are mapped individually above, because each decides whether a physical quantity is recorded along the trajectory.",
    "param14": "domain-diagnostic selectors; GM1 publishes its own diagnostics.",
    # param15 stays non-physical so the harness keeps refusing doazimavg,
    # azimavgfrq and rlen by name: a case that moved one would make CM1 write
    # an azimuthal average GM1 has no writer for.
    "param15": "azimuthal-average selectors; GM1 has no azimavg writer, so doazimavg, azimavgfrq and rlen are unreachable. do_adapt_move and adapt_move_frq are mapped individually above, because they move the domain; the controller is driven from the completed-step frame cadence, which is cm1.F:3642-3651's other dowriteout branch.",
    "param16": "native-restart selectors; GM1 restarts through its own Rust checkpoint envelope.",
    "nssl2mom_params": "NSSL two-moment microphysics tuning; ArWen ships an nssl2 kernel but no TOML key selects it.",
    "nssl_mp_params": "NSSL microphysics internals; unreachable with the above.",
    "logging": "runtime logging only; no effect on state.",
}


def build(schema_path: Path, config_path: Path) -> dict[str, Any]:
    authority = json.loads(schema_path.read_text(encoding="utf-8"))
    groups = {g["name"]: g for g in authority["groups"]}

    # 1. every mapped CM1 key must exist in the authority inventory.
    inventory: dict[str, str] = {}
    for name, group in groups.items():
        for field in group["fields"]:
            inventory.setdefault(field["name"], name)
    unknown = sorted(k for k in MAP if k not in inventory)
    if unknown:
        raise SystemExit(f"map names CM1 fields absent from the authority inventory: {unknown}")

    # 2. every physical-group CM1 key must appear in the map.
    physical: list[str] = []
    for name in PHYSICAL_GROUPS:
        physical.extend(f["name"] for f in groups[name]["fields"])
    missing = sorted(set(physical) - set(MAP))
    if missing:
        raise SystemExit(f"physical CM1 fields absent from the map: {missing}")

    # 3. every GM1 key named must be in config.py's schema.
    gm1_schema = _gm1_schema(config_path)
    named = {v[0] for v in MAP.values() if v[0]} | set(GM1_ONLY)
    stray = sorted(k for k in named if k not in gm1_schema)
    if stray:
        raise SystemExit(f"map names gpuwm-cm1 keys absent from config.py _SCHEMA: {stray}")
    uncovered = sorted(gm1_schema - named)
    if uncovered:
        raise SystemExit(f"gpuwm-cm1 keys neither mapped nor listed as GM1-only: {uncovered}")

    # 4. every unmapped state-affecting key says what GM1 does for it, and
    #    only an unmapped key does.
    stray = sorted(k for k in UNMAPPED_BEHAVIOUR if k not in MAP or MAP[k][0] is not None)
    if stray:
        raise SystemExit(f"UNMAPPED_BEHAVIOUR names mapped or unknown keys: {stray}")
    for key, record in UNMAPPED_BEHAVIOUR.items():
        if not (("gm1_runs" in record) or ("read_when" in record)) or "anchor" not in record:
            raise SystemExit(f"UNMAPPED_BEHAVIOUR[{key!r}] needs gm1_runs, read_when or both, and an anchor")
        for alternative in record.get("read_when", ()):
            unknown = sorted(k for k in alternative if k not in inventory)
            if unknown:
                raise SystemExit(f"UNMAPPED_BEHAVIOUR[{key!r}] reads selectors absent from the inventory: {unknown}")
    unrecorded = sorted(
        k for k, v in MAP.items()
        if v[0] is None and k not in NO_EFFECT_ON_STATE and k not in UNMAPPED_BEHAVIOUR
    )

    entries = []
    for key in sorted(MAP):
        gm1, transform, note = MAP[key]
        entry = {
            "cm1_key": key,
            "cm1_group": inventory[key],
            "cm1_type": next(f["type"] for f in groups[inventory[key]]["fields"] if f["name"] == key),
            "gm1_key": gm1,
            "transform": transform,
            "note": note,
            "affects_state": key not in NO_EFFECT_ON_STATE,
        }
        if key in UNMAPPED_BEHAVIOUR:
            entry["gm1_behaviour"] = UNMAPPED_BEHAVIOUR[key]
        entries.append(entry)

    mapped = [e for e in entries if e["gm1_key"]]
    unmapped = [e for e in entries if not e["gm1_key"]]
    return {
        "schema": SCHEMA,
        "authority_schema": authority["schema"],
        "authority_tarball_sha256": authority["authority"]["tarball_sha256"],
        "counts": {
            "cm1_fields_total": authority["counts"]["fields"],
            "cm1_fields_physical": len(physical),
            # The map also carries the sixteen &param13 parcel-attribute
            # switches, output_pdcomp, and &param15's do_adapt_move and
            # adapt_move_frq, which live in non-physical groups but each
            # decide a physical quantity or where the domain is.  They
            # are counted separately so cm1_fields_physical stays the count of
            # PHYSICAL_GROUPS and the two together are the map's own size.
            "cm1_fields_mapped_from_non_physical_groups":
                len([e for e in entries if e["cm1_group"] not in PHYSICAL_GROUPS]),
            "cm1_fields_mapped": len(mapped),
            "cm1_fields_unmapped": len(unmapped),
            "gm1_keys_total": len(gm1_schema),
            "gm1_keys_only": len(GM1_ONLY),
        },
        "entries": entries,
        "gm1_only": [{"gm1_key": k, "note": GM1_ONLY[k]} for k in sorted(GM1_ONLY)],
        "non_physical_families": [
            {"cm1_group": k, "cm1_field_count": len(groups[k]["fields"]), "note": v}
            for k, v in sorted(NON_PHYSICAL_FAMILIES.items())
        ],
        "harness_owned_groups": sorted(HARNESS_OWNED_GROUPS),
        "derived_gap_register": sorted(e["cm1_key"] for e in unmapped),
        # Unmapped state-affecting keys whose GM1 behaviour is not recorded:
        # the translator names each of them whenever a deck sets it.
        "gm1_behaviour_unrecorded": unrecorded,
    }


_KEY_GROUPS = ("_LOWER_BOUNDARY_KEYS", "_OPEN_COMPANION_KEYS", "_HORIZONTAL_DAMPING_KEYS")


def _gm1_schema(config_path: Path) -> set[str]:
    """Read ``config.py``'s ``_SCHEMA`` without importing CUDA-adjacent code."""

    import ast

    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    # Two passes, so a section that unions in a named key group is read
    # whatever order the module defines the two in.  Naming one group here by
    # hand -- as this did for _LOWER_BOUNDARY_KEYS -- meant the next group
    # added to a section silently vanished from the map instead of failing.
    groups: dict[str, set[str]] = {}
    nodes: dict[str, ast.AST] = {}
    schema_node: ast.AST | None = None
    # config.py declares several key groups as named frozensets and then folds
    # them into _SCHEMA by name; each one is resolved here so a key that lives
    # in a group is not read as absent from the grammar.
    groups: dict[str, set[str]] = {}
    lower: set[str] = set()
    # config.py imports LAGRANGIAN_KEYS from lagrangian_config, so it is a
    # bare Name in the _SCHEMA literal; read the sibling module the same way
    # _LOWER_BOUNDARY_KEYS is read, rather than importing either.
    lagrangian = _named_key_set(
        config_path.with_name("lagrangian_config.py"),
        ("_DEFAULTS", "_SWITCH_DEFAULTS"),
    )
    schema: set[str] = set()
    group_names = ("_LOWER_BOUNDARY_KEYS", "_TURBULENCE_SCALE_KEYS", "_RADIATION_KEYS")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        name = getattr(target, "id", None)
        if name is None or node.value is None:
            continue
        if name == "_SCHEMA":
            schema_node = node.value
            continue
        strings = {
            c.value
            for c in ast.walk(node.value)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        # Remember the expression too: a group can be built from another
        # group by name (``A | (B - {...})``), and collecting only the string
        # literals would silently under-count it.
        nodes[name] = node.value
        if strings:
            groups[name] = strings
            continue
        # A group derived from another table, e.g.
        # ``_INITIALIZATION_OPTIONAL_KEYS = frozenset(_INITIALIZATION_DEFAULTS)``.
        # Its members are the keys of the table it is built from, which this
        # pass has already read, so resolve it by name rather than by literal.
        if isinstance(node.value, ast.Call):
            for arg in node.value.args:
                if isinstance(arg, ast.Name) and arg.id in groups:
                    groups[name] = set(groups[arg.id])
                    break
    # ``OUTPUT_KEYS`` is CM1's &param9 inventory; config.py imports it from
    # cm1_output_selection rather than spelling 63 names again, so the AST
    # pass cannot see its members and has to read them from the module that
    # owns them.
    # Key groups config.py imports rather than spells: reading them from the
    # module that owns them keeps one inventory instead of two.
    for group_name, module in (
        ("OUTPUT_KEYS", "cm1core.cm1_output_selection"),
        ("LAGRANGIAN_KEYS", "cm1core.lagrangian_config"),
    ):
        if group_name in groups:
            continue
        import importlib
        import sys as _sys

        root = str(config_path.resolve().parents[2])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        groups[group_name] = set(getattr(importlib.import_module(module), group_name))
    if "OUTPUT_KEYS" not in groups:
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(config_path.resolve().parents[2]))
        groups["OUTPUT_KEYS"] = set(
            importlib.import_module("cm1core.cm1_output_selection").OUTPUT_KEYS
        )
    schema: set[str] = set()
    if schema_node is None:
        raise SystemExit("config.py defines no _SCHEMA")
    mapping = schema_node.args[0] if isinstance(schema_node, ast.Call) else schema_node
    assert isinstance(mapping, ast.Dict)
    resolving: set[str] = set()

    def _members(node: ast.AST) -> set[str]:
        """The key names one _SCHEMA value denotes.

        Evaluated rather than walked, so that a section spelled as a
        difference -- ``frozenset(OUTPUT_KEYS) - {"output_dbz"}``, which is
        how &param9 avoids offering a second spelling of a word [physics]
        already owns -- reports the keys the parser actually admits instead
        of every name that appears in the expression.
        """

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            if node.id in nodes and node.id not in resolving:
                resolving.add(node.id)
                try:
                    return _members(nodes[node.id])
                finally:
                    resolving.discard(node.id)
            return set(groups.get(node.id, ()))
        if isinstance(node, ast.Dict):
            # ``frozenset(SOME_DEFAULTS)`` is the table's key set.
            return {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        if isinstance(node, ast.Starred):
            # ``{..., *("child_nz", ...)}``: _NEST_KEYS unpacks its optional
            # child-deck words into the set literal.
            return _members(node.value)
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            return set().union(*(_members(e) for e in node.elts)) if node.elts else set()
        if isinstance(node, ast.Call):
            return set().union(*(_members(a) for a in node.args)) if node.args else set()
        if isinstance(node, ast.BinOp):
            left, right = _members(node.left), _members(node.right)
            if isinstance(node.op, ast.Sub):
                return left - right
            return left | right
        return set()

    for key_node, value_node in zip(mapping.keys, mapping.values):
        schema.update(f"{key_node.value}.{k}" for k in _members(value_node))
        if name == "_LOWER_BOUNDARY_KEYS":
            lower = {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        elif name == "_SCHEMA":
            call = node.value
            mapping = call.args[0] if isinstance(call, ast.Call) else call
            assert isinstance(mapping, ast.Dict)
            for key_node, value_node in zip(mapping.keys, mapping.values):
                section = key_node.value
                for c in ast.walk(value_node):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        schema.add(f"{section}.{c.value}")
                if any(isinstance(c, ast.Name) and c.id == "_LOWER_BOUNDARY_KEYS" for c in ast.walk(value_node)):
                    schema.update(f"{section}.{k}" for k in lower)
                if any(isinstance(c, ast.Name) and c.id == "LAGRANGIAN_KEYS" for c in ast.walk(value_node)):
                    schema.update(f"{section}.{k}" for k in lagrangian)
    return schema


def _named_key_set(path: Path, names: tuple[str, ...]) -> set[str]:
    """The literal dict keys of the named module-level mappings."""

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if getattr(target, "id", None) not in names:
            continue
        for child in ast.walk(node.value):
            if isinstance(child, ast.Dict):
                for key_node in child.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        out.add(key_node.value)
            elif isinstance(child, ast.DictComp):
                # {name: 1 for name in PARCEL_SWITCH_ORDER}
                out.update(_named_tuple_members(path, child))
    return out


def _named_tuple_members(path: Path, comp) -> set[str]:
    """Resolve a dict comprehension over one module-level string tuple."""

    import ast

    source = comp.generators[0].iter
    if not isinstance(source, ast.Name):
        return set()
    for module in (path, path.with_name("lagrangian_parcels.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.Assign):
                target = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                target = node.target
            if getattr(target, "id", None) != source.id:
                continue
            return {
                c.value for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            }
    return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--schema", type=Path, default=root / "rust/cm1-state/schema/cm1-r21.1-namelist-support.json")
    parser.add_argument("--config", type=Path, default=root / "src/cm1core/config.py")
    parser.add_argument("--output", type=Path, default=root / "src/cm1core/records/CM1-GM1-KEYMAP.json")
    arguments = parser.parse_args()
    payload = build(arguments.schema, arguments.config)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
