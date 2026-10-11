"""The ArWen -> GM1 chain: a GM1 Cartesian domain driven by an ArWen history.

WHAT THIS IS
------------
WRF's ``ndown`` in GM1's terms: a finer, standalone GM1 Cartesian domain whose
initial state and **time-varying** lateral boundaries are taken from a saved
sequence of ArWen (GPU WRF) history frames.  It is the live nest
(:mod:`cm1core.nest_live`) with an **offline** parent: the parent state at
each boundary time is read from a frame instead of a running parent domain,
and the boundary target is linear in time between two frames.

AUTHORITIES, EACH FOR WHAT IT DECIDES
-------------------------------------
* **GM1's dynamics: CM1 r21.1.**  Nothing in the chain changes a GM1 operator.
  The domain is an ordinary Cartesian GM1 grid whose base state is built by
  CM1's own ``base.F`` ``isnd = 7`` branch from a sounding this module writes.
* **Lateral boundary forcing: WRF** (CM1 r21.1 has none -- ``nest_boundary``
  module docstring).  The specified and relaxation zones, their placement in
  the step and the Davies weights are the live nest's
  (``records/CM1-NEST-SEAM.md`` section 5.2).  What changes is where the table
  pair comes from, and that is WRF's specified (``wrfbdy``) lane, which
  ``ndown`` writes exactly as ``real.exe`` does:

  - ``main/ndown_em.F:846-848`` ``stuff_bdy``: the boundary **value** at the
    start of an interval is the driving model's own state at that time (not
    the nest's own value, which is the live-nest ``med_nest_force`` rule);
  - ``main/ndown_em.F:969-972`` with ``share/module_bc.F:2938``
    ``stuff_bdytend_new``: the **tendency** is ``(new - old)/time_diff`` over
    the boundary interval (``new_bdy_frq = interval_seconds``, ``:321``);
  - ``share/module_bc.F:1300-1320`` ``relax_bdytend_core``: the relaxation
    target is ``bdy + dtbc*bdy_tend``;
  - ``share/input_wrf.F:1833-1835`` resets ``dtbc = 0`` when the boundary
    record is read and ``dyn_em/solve_em.F:377`` advances ``dtbc = dtbc + dt``
    at the start of every step;
  - ``dyn_em/module_bc_em.F:1311-1322`` ``lbc_fcx_gcx`` specified branch: the
    nested-branch weights times ``exp(-(loop-(spec_zone+1))*spec_exp)``; the
    chain admits only ``spec_exp = 0``, for which the two branches are the
    same numbers, so the live nest's :func:`nest_boundary.davies_weights`
    are WRF's specified weights exactly.
* **Vertical transfer between decks: CM1's ``interp_on_restart``**
  (``src/restart_read.F:564-609, 3302-3470``), per column, in WRF's order --
  vertically on a parent-resolution window, then horizontally
  (``nest_vertical`` module docstring).  ArWen's levels are terrain-following
  mass levels whose heights move with every column and every frame, so the
  CM1 search is run on each column's own heights above ground; the CM1 ghost
  rules (``src/param.F:7952-7953, 8165-8173``) extend each column below its
  first and above its last level exactly as CM1 extends its own.
* **Horizontal transfer: the live nest's capture operator**
  (``nest_downscale.downscale_level_staggered``, ``CM1-NEST-SEAM.md`` 5.1).
  ArWen is an Arakawa C grid like CM1, so ``U``/``V``/``W`` sit on the same
  staggers as ``u3d``/``v3d``/``w3d``, the GM1 domain is placed on whole ArWen
  cells with an odd ratio, and the normal velocity on the GM1 boundary is the
  ArWen face value copied.

GM1 CHOICES (NOT TRANSCRIPTIONS), RECORDED IN EVERY RECEIPT
-----------------------------------------------------------
See :data:`GM1_CHOICES`.  In short: full theta and full Exner pressure are
remapped and GM1's own base state is subtracted after the remap; the base
state is the domain-mean t0 column; TKE is not transferred; no hydrostatic
rebalance is applied (CM1's ``interp_on_restart`` applies none).

SCOPE, AND WHAT IS REFUSED
--------------------------
GM1's Cartesian lane is flat and has no map-factor terms.  The chain admits a
region only where that is an honest description of the ArWen grid: terrain
relief under a declared limit; GM1's spacing the history's true spacing over the
window (``DX/(ratio*m)``, ``m`` the window-mean map factor) and the map factor's
variation over the window under a declared limit; one projection in metres;
frames at one cadence that divides the GM1 step and is no longer than the GM1
domain's crossing time; either the GM1 microphysics scheme's own species by name
(``chain.moisture = "all-species"``) or water vapour alone (``"vapour-only"``,
WRF's ``have_bcs_moist = .false.``, :data:`MOISTURE_LANES`); and a fixed
(non-translating) GM1 frame.  Every refusal names the wrong result it prevents;
:data:`REFUSAL_CODES` lists them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHAIN_SCHEMA = "gm1.arwen-chain/v1"
FRAME_SCHEMA = "gm1.arwen-history-frame/v1"
PREPARED_SCHEMA = "gm1.arwen-chain-prepared/v1"
CHAIN_GATE = "arwen-chain"

#: WRF ``share/module_model_constants.F``: ``g`` (:17), ``t0`` (:37),
#: ``EOMEG`` (:83), ``r_d`` (:19), ``r_v`` (461.6).
WRF_GRAVITY_M_S2 = 9.81
WRF_T0_K = 300.0
WRF_EOMEG_S_INV = 7.2921e-5
WRF_RV_OVER_RD = 461.6 / 287.0
#: CM1 r21.1 ``constants.incl``, as GM1's startup kernel carries them
#: (``cuda_cartesian_startup.CM1_RD_F32``/``CM1_CP_F32``/``CM1_P00_F32``).
CM1_RD = 287.04
CM1_CP = 1005.7
CM1_P00_PA = 1.0e5

#: GM1 microphysics component -> WRF ``mp_physics`` option it transcribes.
WRF_MP_PHYSICS = {
    "wsm6-mp6": 6,
    "thompson-mp8": 8,
    "morrison-mp10": 10,
    "wdm6-mp16": 16,
    "thompson-aerosol-mp28": 28,
}
#: GM1 ``q3d`` scalar name -> WRF history variable (Registry.EM_COMMON names).
WRF_SPECIES = {
    "qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN", "qi": "QICE", "qs": "QSNOW",
    "qg": "QGRAUP", "nci": "QNICE", "ncs": "QNSNOW", "ncr": "QNRAIN",
    "ncg": "QNGRAUPEL", "ncc": "QNCLOUD", "nwfa": "QNWFA", "nifa": "QNIFA",
    "ncn": "QNCCN",
}
#: Projections whose ``DX`` is metres on the map plane with an isotropic map
#: factor (WRF ``MAP_PROJ`` 1 Lambert, 2 polar stereographic, 3 Mercator).
METRIC_PROJECTIONS = {1: "Lambert conformal", 2: "polar stereographic", 3: "Mercator"}
#: Every projection the chain admits: the metric three, and ``MAP_PROJ = 0``, an
#: idealized WRF or ArWen grid, which is GM1's own geometry (flat, no map).
PROJECTION_NAMES = {0: "idealized (no map)", **METRIC_PROJECTIONS}

#: WRF ``mp_physics`` options by name (WRF 4.x ``Registry``/``module_physics_init``),
#: so a refusal names the scheme the history ran rather than a bare number.
WRF_MP_SCHEME_NAMES = {
    0: "no microphysics", 1: "Kessler", 2: "Purdue Lin", 3: "WSM3", 4: "WSM5", 5: "Ferrier",
    6: "WSM6", 7: "Goddard", 8: "Thompson", 9: "Milbrandt-Yau", 10: "Morrison", 11: "CAM 5.1",
    13: "Stony Brook Lin", 14: "WDM5", 16: "WDM6", 17: "NSSL", 18: "NSSL", 19: "NSSL",
    21: "NSSL", 22: "NSSL", 24: "WSM7", 26: "WDM7", 28: "Thompson aerosol-aware",
    30: "HUJI spectral bin", 32: "HUJI spectral bin", 38: "Thompson hail-aware",
    40: "Morrison with CESM aerosol", 50: "P3", 51: "P3", 52: "P3", 53: "P3",
    55: "Jensen ISHMAEL", 56: "NTU",
}
#: The history schemes the chain offers, and the GM1 component that transcribes each
#: (6, 8 and 10; Thompson runs only after the import's acknowledgment).
CHAIN_SCHEMES = {6: "wsm6-mp6", 8: "thompson-mp8", 10: "morrison-mp10"}
#: Components a chained run must acknowledge first (the import's own rule).
ACKNOWLEDGED_COMPONENTS = {
    "thompson-mp8": "Thompson microphysics has run only a 120 s column in GM1; its storm-scale "
                    "results are not yet compared with CM1's",
}

#: How the history's moisture enters the chain (``chain.moisture``).
#:
#: ``all-species`` (the default): every species of GM1's scheme from the history, which must have run
#: that scheme (WRF ``have_bcs_moist = .true.``, or a nest: ``solve_em.F:2256-2258, 4622-4623``).
#: ``vapour-only``: WRF's own default real-data lane, ``have_bcs_moist = .false.`` (WRF v4.6.1
#: ``Registry/Registry.EM_COMMON:2945``): only water vapour is relaxed and specified from the
#: driving run (``dyn_em/solve_em.F:2256-2258`` ``relax_bdy_scalar``/``spec_bdy_scalar`` and
#: ``:4622-4623`` ``spec_bdy_final`` for ``P_QV`` only); every other moist species takes
#: ``flow_dep_bdy`` on the specified rows after each RK stage (``solve_em.F:2337-2349``,
#: ``share/module_bc.F:2335-2456``: zero on inflow, the first interior row's value on outflow)
#: and starts at zero, as a ``real.exe`` start from an analysis without hydrometeors does.  It is
#: the honest lane for a history whose scheme GM1 does not run (P3, NSSL, ...): no species is
#: poured into another scheme's categories.
MOISTURE_LANES = {
    "all-species": "every species of GM1's scheme, from a history that ran that scheme",
    "vapour-only": (
        "water vapour only (WRF have_bcs_moist = .false.): hydrometeors start at zero and take "
        "WRF's flow-dependent boundary (zero inflow, zero-gradient outflow)"
    ),
}
DEFAULT_MOISTURE = "all-species"
#: The history's hydrometeor mass variables, for the record of what the vapour-only lane leaves out.
HYDROMETEOR_MASS_VARIABLES = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL")

#: The GM1 carriers the chain fills, in table order, and their staggers.
DYNAMIC_CARRIERS = (("u", "u"), ("v", "v"), ("w", "w"), ("pi", "mass"), ("theta", "mass"))

AUTHORITY = {
    "boundary_value": "WRF main/ndown_em.F:846-848 stuff_bdy (driving state at the interval start)",
    "boundary_tendency": (
        "WRF main/ndown_em.F:969-972 stuff_bdytend with new_bdy_frq=interval_seconds (:321); "
        "share/module_bc.F:2938 (new-old)/time_diff"
    ),
    "relaxation_target": "WRF share/module_bc.F:1300-1320 relax_bdytend_core bdy + dtbc*bdy_tend",
    "dtbc_reset": "WRF share/input_wrf.F:1833-1835 dtbc=0 on reading the boundary record",
    "dtbc_advance": "WRF dyn_em/solve_em.F:377 dtbc = dtbc + dt",
    "davies_weights": (
        "WRF dyn_em/module_bc_em.F:1297-1341 lbc_fcx_gcx, specified branch with spec_exp=0, "
        "identical to the nested branch cm1core.nest_boundary.davies_weights transcribes"
    ),
    "placement_in_step": "records/CM1-NEST-SEAM.md section 5.2 (the live nest's WRF placement)",
    "vertical_rule": "CM1 r21.1 src/restart_read.F:564-609, 3302-3470 interp_on_restart, per column",
    "vertical_ghosts": "CM1 r21.1 src/param.F:7952-7953 (sigmaf), 8165-8173 (zf, zh)",
    "vertical_order": "WRF external/RSL_LITE/force_domain_em_part2.F:126,141-147 (vertical first)",
    "horizontal_operator": "cm1core.nest_downscale.downscale_level_staggered (CM1-NEST-SEAM.md 5.1)",
    "heights": "WRF geopotential (PH+PHB)/g on w levels; mass level height the mean of its two faces",
    "base_state": "CM1 r21.1 src/base.F:495-607 isnd=7 input_sounding",
    "coriolis": "WRF share/module_model_constants.F:83 EOMEG; f = 2*EOMEG*sin(XLAT)",
    "arwen_theta": "gpuwm/io/wrfout.py:822 T = total_theta - 300 (gpuwm/core/state.py:958-961)",
    "theta_m_attribute": "WRF share/output_wrf.F:679 writes USE_THETA_M to the history file",
}

GM1_CHOICES = {
    "remapped_thermodynamics": (
        "full potential temperature (T+300) and full Exner function "
        "((P+PB)/p00)**(rd/cp) with CM1's constants are remapped; GM1's own th0 and pi0 "
        "are subtracted on GM1 levels afterwards. CM1's interp_on_restart moves "
        "perturbations between two CM1 base states; ArWen carries no CM1 base state, so the "
        "full field is the one quantity both models share"
    ),
    "base_state": (
        "the GM1 base state is CM1's isnd=7 construction from the domain-mean t0 column of "
        "the remapped state on GM1's scalar levels, with the domain-mean PSFC, TH2 (or T2 "
        "and PSFC) and Q2 as the surface record"
    ),
    "tke": (
        "not transferred: CM1's restart-grid path does not read tkea (restart_read.F:1207) "
        "and WRF's TKE advance has no boundary call (solve_em.F:2378-2416); GM1's own "
        "initial TKE is kept"
    ),
    "no_rebalance": (
        "no hydrostatic rebalance of the remapped pressure, as CM1's interp_on_restart "
        "does none; the domain-mean pi' left by the difference between WRF's and CM1's "
        "discrete hydrostatics is measured and recorded"
    ),
    "grid_axes": (
        "GM1 x/y are ArWen's grid axes; U and V are ArWen's grid-relative components and "
        "are copied without rotation"
    ),
    "map_scale": (
        "GM1's spacing is the history's true spacing over the window: DX/(ratio*m), m the "
        "window-mean map factor MAPFAC_M (WRF's physical length of a DX step is DX/m; "
        "m = 1 for MAP_PROJ = 0).  The map-factor check then measures what GM1 still gets "
        "wrong, max |m/m_window - 1| over the window, against the unchanged limit.  GM1 carries "
        "no map-factor terms, so a window over which m varies more than the limit is refused"
    ),
    "coriolis": (
        "GM1's f-plane fcor must match 2*EOMEG*sin(latitude) at the domain centre. CM1 "
        "r21.1 applies fcor to the full wind plus the frame velocity (src/solve2.F:633-650), "
        "and ArWen's pi' carries the large-scale gradient that balances it, so no "
        "large-scale pressure-gradient forcing is added (CM1's lspgrad and lsnudge are not "
        "selectable from a Cartesian document)"
    ),
    "upper_damping": (
        "CM1's Rayleigh upper damping, when the document selects it, relaxes toward the "
        "base state (the t0 domain-mean column) for the whole run; over hours that is a "
        "stale target and is a named limit of this first version, not a transcription of "
        "WRF's damp_opt"
    ),
    "number_concentrations": "WRF QN* and GM1 nc*/nwfa/nifa both # per kg of dry air; copied",
}

REFUSAL_CODES = {
    "GM1-CHAIN-FRAME": "a frame directory is malformed, unhashed or its words changed",
    "GM1-CHAIN-FRAME-COUNT": "fewer than two frames: no time interpolation exists",
    "GM1-CHAIN-FRAME-GRID": "frames disagree on grid, projection, terrain or scheme",
    "GM1-CHAIN-CADENCE": "frame times are not one strictly increasing constant cadence",
    "GM1-CHAIN-WINDOW": "the frames do not start at start_time and end at end_time",
    "GM1-CHAIN-INTERVAL-STEP": "the boundary interval is not a whole number of GM1 steps",
    "GM1-CHAIN-DURATION": "run.duration_s reaches past the last frame",
    "GM1-CHAIN-CROSSING": "the interval is longer than the GM1 domain's crossing time",
    "GM1-CHAIN-PROJECTION": "the projection's DX is not an isotropic metric spacing",
    "GM1-CHAIN-SPACING": "GM1 dx is not the history's true spacing DX/(ratio*m) as a binary32 word",
    "GM1-CHAIN-PLACEMENT": "the GM1 window reaches ArWen boundary or halo words",
    "GM1-CHAIN-TERRAIN": "terrain relief over the window exceeds the declared limit",
    "GM1-CHAIN-MAP-FACTOR": "the map factor varies over the window by more than the declared limit",
    "GM1-CHAIN-CORIOLIS": "GM1's f-plane does not match the domain's latitude",
    "GM1-CHAIN-MICROPHYSICS": "ArWen's mp_physics is not GM1's scheme (all-species lane)",
    "GM1-CHAIN-SPECIES": "a GM1 scalar has no ArWen history variable",
    "GM1-CHAIN-VARIABLE": "a required history variable is absent",
    "GM1-CHAIN-THETA-M": "T may carry moist potential temperature",
    "GM1-CHAIN-COLUMN": "ArWen level heights are not strictly increasing in a column",
    "GM1-CHAIN-MODEL-TOP": "GM1 levels lie above ArWen's last level",
    "GM1-CHAIN-PREPARED": "the prepared products do not match this document or their hashes",
    "GM1-CHAIN-RESTART": "a restart would not carry the chain's interval position",
    "GM1-CHAIN-INITIAL-CONDITION": "iconly publishes a state the chain never wrote",
    "GM1-CHAIN-SOUNDING": "the base-state sounding is not the prepared t0 column",
}


def _np() -> Any:
    import numpy

    return numpy


def _refuse(code: str, reason: str, remedy: str, *, option: str | None = None,
            value: object = None) -> None:
    from .errors import refuse

    if code not in REFUSAL_CODES:
        raise KeyError(code)
    refuse(code=code, gate=CHAIN_GATE, reason=reason, remedy=remedy, option=option, value=value)


def parse_wrf_time(text: str) -> float:
    """A WRF ``Times`` string as UTC seconds since the epoch.

    ``YYYY-MM-DD_HH:MM:SS`` is WRF's own spelling; ``YYYY-MM-DD HH:MM:SS`` and
    ``YYYY-MM-DDTHH:MM:SS`` (written by some post-processors) name the same time.
    """

    from .wrf_history import wrf_spelling

    stamp = _dt.datetime.strptime(wrf_spelling(text), "%Y-%m-%d_%H:%M:%S")
    return stamp.replace(tzinfo=_dt.timezone.utc).timestamp()


def wrf_time(seconds: float) -> str:
    """UTC seconds as WRF's ``YYYY-MM-DD_HH:MM:SS``."""

    stamp = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=float(seconds))
    # Spelled out: strftime drops the leading zeros of WRF's idealized year 0001 on some C libraries.
    return (f"{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}_"
            f"{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}")


def wrf_scheme_name(mp: object) -> str:
    """``"WSM3 (3)"``: a WRF ``mp_physics`` option by its scheme's name."""

    if mp is None:
        return "no MP_PHYSICS attribute"
    try:
        option = int(mp)
    except (TypeError, ValueError):
        return f"mp_physics {mp!r}"
    return f"{WRF_MP_SCHEME_NAMES.get(option, 'an unlisted scheme')} ({option})"


def component_for_mp_physics(mp: object) -> str:
    """The GM1 component a history's ``MP_PHYSICS`` is chained with, or a refusal naming the scheme."""

    try:
        option = int(mp)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        option = None
    if option not in CHAIN_SCHEMES:
        offered = ", ".join(wrf_scheme_name(k) for k in sorted(CHAIN_SCHEMES))
        _refuse(
            "GM1-CHAIN-MICROPHYSICS",
            f"the history ran {wrf_scheme_name(mp)} microphysics and GM1 has no component "
            "transcribing it; its hydrometeors would be handed to another scheme's "
            "size distributions, fall speeds and densities from the first step",
            f"run the WRF or ArWen case with one of {offered}",
            option="MP_PHYSICS", value=mp,
        )
    return CHAIN_SCHEMES[option]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Frames: the Rust extractor's output (rust/cm1-state/src/arwen_history.rs).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArwenFrame:
    """One extracted ArWen history record."""

    directory: Path
    header: Mapping[str, Any]

    @classmethod
    def open(cls, directory: Path | str) -> "ArwenFrame":
        root = Path(directory)
        try:
            header = json.loads((root / "frame.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _refuse(
                "GM1-CHAIN-FRAME",
                f"{root} has no readable frame.json ({error}); the chain would have no "
                "identity for the words it remaps",
                "extract the frame with gm1-arwen-history-extract WRFOUT RECORD NEW_DIR",
                value=str(root),
            )
        if header.get("schema") != FRAME_SCHEMA:
            _refuse(
                "GM1-CHAIN-FRAME",
                f"{root}/frame.json is schema {header.get('schema')!r}, not {FRAME_SCHEMA}",
                "re-extract the frame with this release's gm1-arwen-history-extract",
                value=str(root),
            )
        frame = cls(root, header)
        size = (root / "fields.f32le").stat().st_size
        end = max((int(f["offset"]) + int(f["nbytes"]) for f in header["fields"]), default=0)
        if size != end:
            _refuse(
                "GM1-CHAIN-FRAME",
                f"{root}/fields.f32le holds {size} bytes and its table ends at {end}; a "
                "truncated or appended arena would be read as other words",
                "re-extract the frame into a new directory",
                value=str(root),
            )
        return frame

    @property
    def times(self) -> str:
        return str(self.header["times"])

    @property
    def valid_s(self) -> float:
        return parse_wrf_time(self.times)

    @property
    def attributes(self) -> Mapping[str, Any]:
        return self.header["attributes"]

    @property
    def dimensions(self) -> Mapping[str, int]:
        return self.header["dimensions"]

    def has(self, name: str) -> bool:
        return any(f["name"] == name for f in self.header["fields"])

    def entry(self, name: str) -> Mapping[str, Any]:
        for f in self.header["fields"]:
            if f["name"] == name:
                return f
        _refuse(
            "GM1-CHAIN-VARIABLE",
            f"the ArWen frame at {self.times} carries no {name}; the chain would have to "
            "invent the field",
            f"write {name} to the ArWen history (for P and PB: include_diagnostic_pressure)",
            option=name,
            value=str(self.directory),
        )
        raise AssertionError  # pragma: no cover

    def array(self, name: str) -> Any:
        np = _np()
        entry = self.entry(name)
        with open(self.directory / "fields.f32le", "rb") as handle:
            handle.seek(int(entry["offset"]))
            raw = handle.read(int(entry["nbytes"]))
        if _sha256(raw) != entry["sha256"]:
            _refuse(
                "GM1-CHAIN-FRAME",
                f"{name} in {self.directory} no longer hashes to its extraction record; the "
                "chain would remap words the extractor did not write",
                "re-extract the frame into a new directory",
                option=name,
                value=str(self.directory),
            )
        return np.frombuffer(raw, dtype="<f4").reshape(tuple(int(n) for n in entry["shape"]))


def write_frame(directory: Path | str, *, times: str, attributes: Mapping[str, Any],
                fields: Mapping[str, Any], dims: Mapping[str, Sequence[str]] | None = None,
                source: Mapping[str, Any] | None = None) -> ArwenFrame:
    """Write a frame in the extractor's format (test fixtures and tooling).

    The Rust extractor is the producer of real frames; this writes the same
    two files so a synthetic ArWen-like state can drive the whole chain.
    """

    np = _np()
    root = Path(directory)
    root.mkdir(parents=True)
    rows = []
    offset = 0
    dimensions: dict[str, int] = {}
    with open(root / "fields.f32le", "wb") as handle:
        for name, value in fields.items():
            array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
            names = list((dims or {}).get(name, [f"d{a}_{n}" for a, n in enumerate(array.shape)]))
            for d, n in zip(names, array.shape):
                dimensions[d] = int(n)
            raw = array.tobytes()
            handle.write(raw)
            rows.append({
                "dims": names, "name": name, "nbytes": len(raw), "offset": offset,
                "sha256": _sha256(raw), "shape": list(array.shape), "stagger": "", "units": "",
            })
            offset += len(raw)
    header = {
        "absent": [], "attributes": dict(attributes), "dimensions": dimensions,
        "fields": rows, "schema": FRAME_SCHEMA,
        "source": dict(source or {"format": "synthetic", "path": "", "record": 0, "sha256": ""}),
        "times": times,
    }
    (root / "frame.json").write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
    return ArwenFrame(root, header)


# ---------------------------------------------------------------------------
# Placement.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainPlacement:
    """Where the GM1 domain sits on the ArWen mass grid."""

    parent_i0: int
    parent_j0: int
    refinement_ratio: int
    spec_zone: int
    relax_zone: int

    def geometry(self, frame: ArwenFrame, *, nx: int, ny: int) -> Any:
        from .nest_geometry import NestGeometry

        attrs = frame.attributes
        ratio = int(self.refinement_ratio)
        if nx % ratio or ny % ratio:
            _refuse(
                "GM1-CHAIN-PLACEMENT",
                f"the GM1 grid is {nx}x{ny} and chain.refinement_ratio is {ratio}; a domain that "
                "is not a whole number of ArWen cells has boundary faces between ArWen faces, so "
                "the normal flux it is forced with would be interpolated rather than ArWen's own",
                "declare grid.nx and grid.ny as multiples of chain.refinement_ratio",
                option="grid.nx", value=[nx, ny, ratio],
            )
        try:
            geometry = NestGeometry(
                parent_nx=int(frame.dimensions["west_east"]),
                parent_ny=int(frame.dimensions["south_north"]),
                parent_dx_m=float(attrs["DX"]), parent_dy_m=float(attrs["DY"]),
                refinement_ratio=ratio, time_refinement_ratio=1,
                parent_i0=int(self.parent_i0), parent_j0=int(self.parent_j0),
                parent_span_x=nx // ratio, parent_span_y=ny // ratio,
                spec_zone=int(self.spec_zone), relax_zone=int(self.relax_zone), feedback=0,
            )
        except (KeyError, TypeError, ValueError) as error:
            _refuse(
                "GM1-CHAIN-PLACEMENT",
                f"the GM1 window cannot be placed on the ArWen grid: {error}",
                "place the window on whole ArWen cells with an odd ratio and spec/relax zones "
                "that leave an interior",
                option="chain.parent_i0", value=[self.parent_i0, self.parent_j0, ratio],
            )
        defect = geometry.placement_defect()
        if defect is not None:
            _refuse(
                "GM1-CHAIN-PLACEMENT",
                f"{defect}; ArWen's own lateral boundary rows are specified from its driver and "
                "the capture stencil would read them as interior air",
                "move the window at least three ArWen cells inside the ArWen domain",
                option="chain.parent_i0", value=[self.parent_i0, self.parent_j0],
            )
        return geometry


# ---------------------------------------------------------------------------
# CM1's interp_on_restart rule on columns with their own heights.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMap:
    """``interp_on_restart``'s planes and ``rz`` for every new level of every column."""

    lower: Any  # (nk, J, I) int, 0-based old data level held by datk1
    upper: Any  # (nk, J, I) int, datk2
    fraction: Any  # (nk, J, I) float32 rz
    below_support: int
    above_support: int


def build_column_map(old_ghosted: Any, new_levels: Any, *, nzold: int) -> ColumnMap:
    """:func:`nest_vertical.build_cm1_vertical_map`, for every column at once.

    ``old_ghosted`` is ``(nzold+2, J, I)`` float32: each column's CM1-indexed
    coordinate ``0..nzold+1`` (ghost, data levels, ghost).  ``new_levels`` is
    GM1's ``(nk,)`` float32 interior coordinate.  The search
    ``do while(sigma0(kold+1) < sigma(k))`` over a strictly increasing column
    stops at the count of old levels ``1..nzold+1`` below the target; the
    ``prep_for_interp`` plane state machine is carried per column.  Raw-word
    identity with the scalar transliteration is tested.
    """

    np = _np()
    old = np.asarray(old_ghosted)
    new = np.asarray(new_levels)
    if old.dtype != np.float32 or new.dtype != np.float32 or old.ndim != 3 or new.ndim != 1:
        raise ValueError("column maps take a float32 (nzold+2,J,I) coordinate and (nk,) levels")
    if old.shape[0] != nzold + 2 or not bool(np.all(np.isfinite(old))):
        raise ValueError("the old coordinate must carry nzold data levels and two finite ghosts")
    if not bool(np.all(old[1:] > old[:-1])):
        raise ValueError("an old column's heights are not strictly increasing")
    shape = (new.size,) + old.shape[1:]
    lower = np.empty(shape, dtype=np.int64)
    upper = np.empty(shape, dtype=np.int64)
    rz = np.empty(shape, dtype=np.float32)
    datk1 = np.full(old.shape[1:], -1, dtype=np.int64)
    datk2 = np.full(old.shape[1:], -1, dtype=np.int64)
    kold2 = np.zeros(old.shape[1:], dtype=np.int64)
    alldone = np.zeros(old.shape[1:], dtype=bool)
    for k in range(new.size):
        target = new[k]
        iz = np.count_nonzero(old[1:] < target, axis=0).astype(np.int64)
        if bool(np.any(iz >= nzold + 1)):
            raise ValueError(
                f"new level {k + 1} at {float(target)} m lies above an old column's last ghost "
                "level; CM1's search would read past sigma0"
            )
        below = np.take_along_axis(old, iz[None], axis=0)[0]
        above = np.take_along_axis(old, (iz + 1)[None], axis=0)[0]
        numerator = (target - below).astype(np.float32)
        denominator = (above - below).astype(np.float32)
        rz[k] = (numerator / denominator).astype(np.float32)
        need = (iz + 1) > kold2
        read = need & ((iz + 1) <= nzold)
        past = need & ~read
        # The read loop leaves kold1 = iz, datk2 = iz+1 and datk1 = iz; with
        # kold1 == 0 CM1 duplicates the first plane (restart_read.F:3302-3406).
        datk2 = np.where(read, iz + 1, datk2)
        datk1 = np.where(read, np.where(iz == 0, iz + 1, iz), datk1)
        kold2 = np.where(read, iz + 1, kold2)
        datk1 = np.where(past & ~alldone, datk2, datk1)
        alldone = alldone | past
        if bool(np.any(datk1 < 1)) or bool(np.any(datk2 < 1)):
            raise ValueError("CM1's plane state machine never read an old level")
        lower[k] = datk1 - 1
        upper[k] = datk2 - 1
    below_support = int(np.count_nonzero(new[:, None, None] < old[1][None]))
    above_support = int(np.count_nonzero(new[:, None, None] > old[nzold][None]))
    return ColumnMap(lower, upper, rz, below_support, above_support)


def interpolate_columns(volume: Any, cmap: ColumnMap) -> Any:
    """``do_interp`` at ``rx = ry = 0`` on every column; float32 out.

    The same eight REAL weights and the same written sum order as
    :func:`nest_vertical.cm1_vertical_interpolate`.
    """

    np = _np()
    from .nest_vertical import cm1_vertical_weights

    data = np.asarray(volume)
    if data.dtype != np.float32:
        raise ValueError("the carriers interp_on_restart reads are default REAL words")
    w1, w2, w3, w4, w5, w6, w7, w8 = cm1_vertical_weights(cmap.fraction)
    lo = np.take_along_axis(data, cmap.lower, axis=0).astype(np.float64)
    hi = np.take_along_axis(data, cmap.upper, axis=0).astype(np.float64)
    result = lo * w1
    result = result + lo * w2
    result = result + lo * w3
    result = result + hi * w4
    result = result + hi * w5
    result = result + hi * w6
    result = result + lo * w7
    result = result + hi * w8
    return result.astype(np.float32)


def _ghost_mass(np: Any, zm: Any, zw_top: Any) -> Any:
    """``param.F:8172-8173`` on a column of scalar heights with its top face."""

    f = np.float32
    below = (-zm[0]).astype(f)
    above = (zm[-1] + f(2.0) * (zw_top - zm[-1]).astype(f)).astype(f)
    return np.concatenate([below[None], zm, above[None]], axis=0)


def _ghost_face(np: Any, zw: Any) -> Any:
    """``param.F:8165-8166`` (and ``:7952-7953`` for sigmaf) on a column of faces."""

    f = np.float32
    below = (-zw[1]).astype(f)
    above = (zw[-1] + (zw[-1] - zw[-2]).astype(f)).astype(f)
    return np.concatenate([below[None], zw, above[None]], axis=0)


def column_heights(frame: ArwenFrame, window: Any, stagger: str) -> tuple[Any, int]:
    """Ghosted heights above ground for one stagger's window, float32.

    Face heights are ``(PH+PHB)/g - HGT`` (float64, narrowed once); a scalar
    level is the mean of its two faces in float32 as CM1 forms ``zh``; a
    ``u`` (``v``) column is the float32 mean of the two mass columns it sits
    between, which is how WRF places a staggered point's height.
    """

    np = _np()
    f = np.float32
    j0, i0, nj, ni = window.j0, window.i0, window.nj, window.ni
    xs = 1 if stagger == "u" else 0
    ys = 1 if stagger == "v" else 0
    jlo, ilo = j0 - ys, i0 - xs
    ph = frame.array("PH")[:, jlo:j0 + nj, ilo:i0 + ni].astype(np.float64)
    phb = frame.array("PHB")[:, jlo:j0 + nj, ilo:i0 + ni].astype(np.float64)
    hgt = frame.array("HGT")[jlo:j0 + nj, ilo:i0 + ni].astype(np.float64)
    zw = ((ph + phb) / WRF_GRAVITY_M_S2 - hgt[None]).astype(f)
    if xs:
        zw = (f(0.5) * (zw[:, :, :-1] + zw[:, :, 1:])).astype(f)
    if ys:
        zw = (f(0.5) * (zw[:, :-1, :] + zw[:, 1:, :])).astype(f)
    if zw.shape[1:] != (nj, ni):
        raise ValueError(f"the {stagger} window reaches past the ArWen grid")
    if stagger == "w":
        return _ghost_face(np, zw), zw.shape[0]
    zm = (f(0.5) * (zw[:-1] + zw[1:])).astype(f)
    return _ghost_mass(np, zm, zw[-1]), zm.shape[0]


def _species_names(component: str) -> tuple[str, ...]:
    from .moisture_species import moisture_species_profile

    return tuple(moisture_species_profile(component).names)


def _parent_carrier(frame: ArwenFrame, name: str, window: Any) -> Any:
    """One carrier on its window of ArWen points, float32."""

    np = _np()
    sl = (slice(None), slice(window.j0, window.j0 + window.nj), slice(window.i0, window.i0 + window.ni))
    if name in ("u", "v", "w"):
        return np.ascontiguousarray(frame.array(name.upper())[sl], dtype=np.float32)
    if name == "theta":
        return (frame.array("T")[sl].astype(np.float64) + WRF_T0_K).astype(np.float32)
    if name == "pi":
        p = frame.array("P")[sl].astype(np.float64) + frame.array("PB")[sl].astype(np.float64)
        return ((p / CM1_P00_PA) ** (CM1_RD / CM1_CP)).astype(np.float32)
    return np.ascontiguousarray(frame.array(WRF_SPECIES[name])[sl], dtype=np.float32)


def transferred_species(component: str, moisture: str = DEFAULT_MOISTURE) -> tuple[str, ...]:
    """The GM1 species the chain takes from the history: all of them, or water vapour alone."""

    names = _species_names(component)
    if moisture == "vapour-only":
        return tuple(name for name in names if name == "qv")
    return names


def carriers_of(component: str, moisture: str = DEFAULT_MOISTURE) -> tuple[tuple[str, str], ...]:
    """Every carrier the chain fills: dynamics, then GM1's q3d axis in order (per moisture lane)."""

    return DYNAMIC_CARRIERS + tuple((name, "mass") for name in transferred_species(component, moisture))


def remap_frame(frame: ArwenFrame, geometry: Any, levels: Any, component: str,
                moisture: str = DEFAULT_MOISTURE) -> dict[str, Any]:
    """Every chain carrier on the GM1 lattice: vertical window, then horizontal.

    Returns float32 volumes: ``u`` ``(nk, ny, nx+1)``, ``v`` ``(nk, ny+1, nx)``,
    ``w`` ``(nk+1, ny, nx)``, the rest ``(nk, ny, nx)``.  ``theta`` and ``pi``
    are full fields.
    """

    np = _np()
    from .nest_downscale import downscale_level_staggered
    from .nest_vertical import VerticalWindow, _WindowGeometry

    nk = int(levels.nk)
    mass_levels = np.asarray(levels.zh[1:nk + 1], dtype=np.float32)
    face_levels = np.asarray(levels.zf[1:nk + 2], dtype=np.float32)
    view = _WindowGeometry(geometry)
    maps: dict[str, ColumnMap] = {}
    out: dict[str, Any] = {}
    for name, stagger in carriers_of(component, moisture):
        window = VerticalWindow.for_stagger(geometry, stagger)
        if stagger not in maps:
            ghosted, nzold = column_heights(frame, window, stagger)
            try:
                maps[stagger] = build_column_map(
                    ghosted, face_levels if stagger == "w" else mass_levels, nzold=nzold
                )
            except ValueError as error:
                code = "GM1-CHAIN-MODEL-TOP" if "above" in str(error) else "GM1-CHAIN-COLUMN"
                _refuse(
                    code,
                    f"ArWen's {stagger} columns at {frame.times} cannot carry GM1's levels: {error}",
                    "lower GM1's model top below ArWen's lowest-column top level, or check the "
                    "frame's PH/PHB/HGT",
                    option="grid.ztop_m", value=float(face_levels[-1]),
                )
        column = interpolate_columns(_parent_carrier(frame, name, window), maps[stagger])
        x_lattice = "face" if stagger == "u" else "mass"
        y_lattice = "face" if stagger == "v" else "mass"
        volume = np.empty(
            (column.shape[0], geometry.ny + (stagger == "v"), geometry.nx + (stagger == "u")),
            dtype=np.float32,
        )
        for k in range(column.shape[0]):
            volume[k] = downscale_level_staggered(
                column[k], view, x_lattice=x_lattice, y_lattice=y_lattice
            ).astype(np.float32)
        out[name] = volume
    out["_support"] = {s: {"below": m.below_support, "above": m.above_support} for s, m in maps.items()}
    return out


# ---------------------------------------------------------------------------
# Boundary strips and interval tables.
# ---------------------------------------------------------------------------
SIDES = ("west", "east", "south", "north")


def strip(volume: Any, side: str, width: int) -> Any:
    """One side's rows in the live nest's table layout (distance 0 outermost)."""

    np = _np()
    from .nest_live_kernels import _side_selector

    return np.ascontiguousarray(volume[_side_selector(side, tuple(volume.shape), width)])


def perturbation(full: Any, base_column: Any) -> Any:
    """``full - base`` per level in float32 (``th3d = theta - th0``, ``pp3d = pi - pi0``)."""

    np = _np()
    base = np.asarray(base_column, dtype=np.float32).reshape((-1,) + (1,) * (np.ndim(full) - 1))
    return (np.asarray(full, dtype=np.float32) - base).astype(np.float32)


def interval_tables(start: Any, end: Any, interval_s: float) -> tuple[Any, Any]:
    """WRF's ``stuff_bdy``/``stuff_bdytend`` pair for one boundary interval.

    ``value = start`` (``ndown_em.F:846-848``) and
    ``tend = (end - start)/interval`` (``module_bc.F:2938``), formed as the
    live nest's capture forms its tendency: the difference in float32 and the
    reciprocal interval in float64, narrowed once
    (``nest_live_kernels.gm1_nest_capture``).
    """

    np = _np()
    here = np.ascontiguousarray(start, dtype=np.float32)
    diff = (np.asarray(end, dtype=np.float32) - here).astype(np.float32)
    rdt = 1.0 / float(interval_s)
    return here, (rdt * diff.astype(np.float64)).astype(np.float32)


def relaxation_target(value: Any, tend: Any, dtbc: float) -> Any:
    """``bdy + dtbc*bdy_tend`` as the kernel forms it (``gm1_nest_target``)."""

    np = _np()
    return (np.asarray(value, np.float32) + (np.float32(dtbc) * np.asarray(tend, np.float32)).astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Admission.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainRequest:
    """Everything the admission reads from the GM1 document."""

    nx: int
    ny: int
    dx_m: float
    dy_m: float
    levels: Any
    dt_s: float
    duration_s: float
    component: str
    coriolis: int
    fcor_s_inv: float
    start_time: str
    end_time: str
    placement: ChainPlacement
    terrain_relief_limit_m: float
    map_factor_departure_limit: float
    moisture: str = DEFAULT_MOISTURE

    def contract(self) -> dict[str, Any]:
        # The moisture lane joins the contract only when it is not the default, so every chain
        # prepared before the lane existed keeps its contract.
        lane = {} if self.moisture == DEFAULT_MOISTURE else {"moisture": self.moisture}
        return {**lane,
            "nx": self.nx, "ny": self.ny, "dx_m": self.dx_m, "dy_m": self.dy_m,
            "vertical_sha256": self.levels.sha256, "nk": int(self.levels.nk),
            "dt_s": self.dt_s, "duration_s": self.duration_s, "component": self.component,
            "coriolis": self.coriolis, "fcor_s_inv": self.fcor_s_inv,
            "start_time": self.start_time, "end_time": self.end_time,
            "parent_i0": self.placement.parent_i0, "parent_j0": self.placement.parent_j0,
            "refinement_ratio": self.placement.refinement_ratio,
            "spec_zone": self.placement.spec_zone, "relax_zone": self.placement.relax_zone,
            "terrain_relief_limit_m": self.terrain_relief_limit_m,
            "map_factor_departure_limit": self.map_factor_departure_limit,
        }


_STATIC_ATTRIBUTES = ("DX", "DY", "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MP_PHYSICS")

#: The admission's checks, in the order they run (and the order ``admit_frames``
#: raises the first refusal in).  Each is one named question with one plain
#: answer; :func:`admission_report` runs every one it can.
ADMISSION_CHECKS = (
    "frame_count", "frame_grid", "cadence", "window", "interval_step", "duration",
    "projection", "placement", "spacing", "terrain", "map_factor", "coriolis",
    "microphysics", "species", "variables", "theta_m", "model_top", "crossing",
)
#: Checks that need the window placed on the grid first.
_NEEDS_GEOMETRY = frozenset({"spacing", "terrain", "map_factor", "coriolis", "theta_m", "model_top", "crossing"})
#: Checks that need the frame cadence first.
_NEEDS_INTERVAL = frozenset({"interval_step", "duration", "crossing"})
#: Checks whose question is only asked of a scheme GM1 has.
_NEEDS_SCHEME = frozenset({"species"})


#: Ground height (HGT) may differ between saved times by float rounding only.  Stock WRF 3.7 rewrote HGT
#: with differences of 2.5 mm between records of one file (the Katrina test file); a centimetre is far
#: below any height the transfer resolves and far above a float32 word of a few-km height.
HGT_TOLERANCE_M = 0.01


def _field_digest(frame: Any, name: str) -> str:
    """The SHA-256 of one field's words: the extractor's record, else hashed here."""

    np = _np()
    try:
        entry = frame.entry(name)
    except Exception:  # noqa: BLE001 - the caller's own entry() refuses by name
        raise
    digest = entry.get("sha256") if isinstance(entry, Mapping) else None
    if digest:
        return str(digest)
    return _sha256(np.ascontiguousarray(frame.array(name), dtype="<f4").tobytes())


def frame_source(frame: Any) -> dict[str, Any]:
    """Where a frame's words came from (an extracted directory, or a history file's record)."""

    history = getattr(frame, "history", None)
    if history is not None and hasattr(history, "source"):
        # A record read straight from a history file (wrf_history.HistoryFrame): its header would read
        # every field of the record, so the file's own source line is taken instead.
        source = dict(history.source(frame.record))
    else:
        header = getattr(frame, "header", None)
        source = dict((header or {}).get("source", {})) if isinstance(header, Mapping) else {}
    extra = getattr(frame, "source", None)
    if isinstance(extra, Mapping):
        source.update(extra)
    directory = getattr(frame, "directory", None)
    if directory is not None:
        source.setdefault("directory", str(directory))
    return source


def producer_of(attributes: Mapping[str, Any]) -> str:
    """Who wrote a history: "ArWen" (gpuwm's GPUWM_VERSION seal or a TITLE naming it), "GM1"
    (``chain frames-from-gm1``), else "WRF" (as :attr:`cm1core.wrf_history.WrfHistory.producer`)."""

    title = str(attributes.get("TITLE", "")).strip().lower()
    if title.startswith("gm1"):
        return "GM1"
    if attributes.get("GPUWM_VERSION") is not None or title.startswith("gpuwm") or "arwen" in title \
            or "gpuwm" in title:
        return "ArWen"
    return "WRF"


def _f_window(frame: Any, msl: Any, centre: tuple[int, int]) -> tuple[Any, float, str]:
    """The Coriolis parameter over the window, its centre value and where it came from.

    ``MAP_PROJ`` 1-3: ``2*EOMEG*sin(XLAT)`` (the record's rule).  ``MAP_PROJ`` 0
    (an idealized WRF or ArWen run): the history's own ``F`` when it has
    one, else ``XLAT``; with neither, the history carries no rotation and ``f``
    is zero, as it is in the idealized WRF and ArWen cases that write neither.
    """

    np = _np()
    projection = int(frame.attributes.get("MAP_PROJ", -1))
    if projection == 0 and frame.has("F"):
        field = frame.array("F")
        return field[msl].astype(np.float64), float(field[centre]), "F"
    if frame.has("XLAT"):
        lat = frame.array("XLAT")
        f_field = 2.0 * WRF_EOMEG_S_INV * np.sin(np.radians(lat[msl].astype(np.float64)))
        return f_field, 2.0 * WRF_EOMEG_S_INV * math.sin(math.radians(float(lat[centre]))), "XLAT"
    if projection == 0:
        return np.zeros((1, 1)), 0.0, "none (no F and no XLAT in an idealized history)"
    frame.entry("XLAT")
    raise AssertionError  # pragma: no cover


def _mapfac_window(frame: Any, msl: Any) -> tuple[Any, str]:
    np = _np()
    if frame.has("MAPFAC_M"):
        return frame.array("MAPFAC_M")[msl].astype(np.float64), "MAPFAC_M"
    if int(frame.attributes.get("MAP_PROJ", -1)) == 0:
        return np.ones((1, 1)), "1 (MAP_PROJ = 0 has no map)"
    frame.entry("MAPFAC_M")
    raise AssertionError  # pragma: no cover


def window_map_scale(frame: Any, msl: Any) -> tuple[float, str]:
    """The window-mean map factor ``m`` (float64 mean of the ``MAPFAC_M`` words), and its source.

    WRF's physical length of one ``DX`` step is ``DX/m`` (the map factor is map distance over
    earth distance), so GM1's true spacing over the window is ``DX/(ratio*m)``.  ``MAP_PROJ = 0``
    has no map: ``m = 1`` exactly, and GM1's spacing is ``DX/ratio`` as before.
    """

    mapfac, source = _mapfac_window(frame, msl)
    return float(mapfac.mean()), source


def box_mass_slice(i0: int, j0: int, ni: int, nj: int) -> tuple[slice, slice]:
    """The history mass cells a box's admission reads: the box and the one cell around it.

    The same cells as ``nest_vertical.VerticalWindow.for_stagger(geometry, "mass")``, the window the
    transfer reads (and the terrain, map-factor and spacing checks measure).
    """

    return (slice(int(j0) - 1, int(j0) + int(nj) + 1), slice(int(i0) - 1, int(i0) + int(ni) + 1))


def chain_spacing_m(frame: Any, ratio: int, msl: Any) -> float:
    """GM1's spacing over the window, ``DX/(ratio*m)``, as the binary32 word the document declares."""

    np = _np()
    scale, _ = window_map_scale(frame, msl)
    spacing = float(frame.attributes["DX"])
    if scale == 1.0:
        return spacing / int(ratio)
    return float(np.float32(spacing / (int(ratio) * scale)))


def _required_variables(frame: Any) -> tuple[str, ...]:
    names = ["U", "V", "W", "T", "PH", "PHB", "P", "PB", "HGT"]
    if int(frame.attributes.get("MAP_PROJ", -1)) != 0:
        names += ["MAPFAC_M", "XLAT"]
    return tuple(names)


def left_out_condensate(frame: Any, msl: Any) -> dict[str, Any]:
    """The hydrometeor water the vapour-only lane leaves out of the t0 state, measured.

    Column paths over the window's history columns, kg m-2: ``sum(rho q dz)`` with the layer
    depth from the geopotential faces and ``rho = p/(Rd T)`` from the full pressure and dry
    theta (WRF's own variables; a measurement, not a transcription), and the vapour path the
    same way for comparison.
    """

    np = _np()
    sl = (slice(None),) + tuple(msl)
    ph = frame.array("PH")[sl].astype(np.float64) + frame.array("PHB")[sl].astype(np.float64)
    dz = np.diff(ph / WRF_GRAVITY_M_S2, axis=0)
    p = frame.array("P")[sl].astype(np.float64) + frame.array("PB")[sl].astype(np.float64)
    temperature = (frame.array("T")[sl].astype(np.float64) + WRF_T0_K) * (p / CM1_P00_PA) ** (CM1_RD / CM1_CP)
    rho_dz = p / (CM1_RD * temperature) * dz
    present = [name for name in HYDROMETEOR_MASS_VARIABLES if frame.has(name)]
    paths = {name: float((rho_dz * frame.array(name)[sl].astype(np.float64)).sum(axis=0).mean())
             for name in present}
    vapour = float((rho_dz * frame.array("QVAPOR")[sl].astype(np.float64)).sum(axis=0).mean()) \
        if frame.has("QVAPOR") else float("nan")
    total = sum(paths.values())
    return {
        "times": frame.times, "window_mean_path_kg_m2": paths, "hydrometeor_path_kg_m2": total,
        "vapour_path_kg_m2": vapour,
        "fraction_of_total_water": total / (total + vapour) if vapour == vapour and total + vapour > 0 else None,
        "basis": "sum(rho*q*dz), rho = p/(Rd*T), dz from (PH+PHB)/g, mean over the window's history columns",
    }


class _Admission:
    """The admission's measured facts, filled check by check."""

    def __init__(self, frames: Sequence[Any], request: ChainRequest) -> None:
        self.frames = frames
        self.request = request
        self.first = frames[0] if frames else None
        self.interval: float | None = None
        self.steps: float | None = None
        self.geometry: Any = None
        self.msl: Any = None
        self.facts: dict[str, Any] = {}

    # -- containers ---------------------------------------------------------
    def frame_count(self) -> None:
        frames = self.frames
        self.facts["frame_count"] = len(frames)
        if len(frames) < 2:
            _refuse(
                "GM1-CHAIN-FRAME-COUNT",
                f"{len(frames)} history frame(s) given; with one frame there is no boundary tendency, "
                "and the domain would be forced toward a frozen t0 state for the whole run",
                "select at least two history frames spanning the run",
                value=len(frames),
            )

    def frame_grid(self) -> None:
        first = self.first
        ground_change = 0.0
        for frame in self.frames[1:]:
            for key in _STATIC_ATTRIBUTES:
                if first.attributes.get(key) != frame.attributes.get(key):
                    _refuse(
                        "GM1-CHAIN-FRAME-GRID",
                        f"history frames {first.times} and {frame.times} disagree on {key} "
                        f"({first.attributes.get(key)!r} vs {frame.attributes.get(key)!r}); two "
                        "frames on different grids or schemes would be interpolated as one field",
                        "take every frame from one domain of one run",
                        option=key,
                    )
            for dim in ("west_east", "south_north", "bottom_top"):
                if first.dimensions.get(dim) != frame.dimensions.get(dim):
                    _refuse(
                        "GM1-CHAIN-FRAME-GRID",
                        f"history frames disagree on {dim}",
                        "take every frame from one domain of one run",
                        option=dim,
                    )
            if _field_digest(first, "HGT") != _field_digest(frame, "HGT"):
                np = _np()
                change = float(np.max(np.abs(frame.array("HGT").astype(np.float64)
                                             - first.array("HGT").astype(np.float64))))
                ground_change = max(ground_change, change)
                if change > HGT_TOLERANCE_M:
                    _refuse(
                        "GM1-CHAIN-FRAME-GRID",
                        f"terrain HGT differs between {first.times} and {frame.times} by up to {change:.3g} m; "
                        "heights above ground would be measured from different ground in consecutive frames",
                        "take every frame from one domain of one run; if only the first saved time's ground "
                        "differs (WRF's first record can), start at the next saved time",
                        option="HGT", value=[first.times, frame.times, change],
                    )
        self.facts["grid"] = {
            "west_east": int(first.dimensions["west_east"]),
            "south_north": int(first.dimensions["south_north"]),
            "bottom_top": int(first.dimensions["bottom_top"]),
            "ground_change_m": ground_change,
            "ground_tolerance_m": HGT_TOLERANCE_M,
        }

    # -- time --------------------------------------------------------------
    def cadence(self) -> None:
        frames = self.frames
        times = [f.valid_s for f in frames]
        intervals = [b - a for a, b in zip(times, times[1:])]
        if any(i <= 0 for i in intervals) or any(i != intervals[0] for i in intervals):
            _refuse(
                "GM1-CHAIN-CADENCE",
                f"frame times {[f.times for f in frames]} are not strictly increasing at one "
                "cadence; WRF's boundary file holds one interval_seconds (ndown_em.F:321) and an "
                "uneven sequence would give each interval's tendency a different time base",
                "take frames at the run's own history cadence, without gaps",
                value=[f.times for f in frames],
            )
        self.interval = float(intervals[0])
        self.facts["interval_s"] = self.interval

    def window(self) -> None:
        frames, request = self.frames, self.request
        if (frames[0].valid_s != parse_wrf_time(request.start_time)
                or frames[-1].valid_s != parse_wrf_time(request.end_time)):
            _refuse(
                "GM1-CHAIN-WINDOW",
                f"the frames run {frames[0].times} to {frames[-1].times} and the document declares "
                f"{request.start_time} to {request.end_time}; the run would be initialized or forced "
                "at a time the document does not name",
                "take exactly the frames from chain.start_time to chain.end_time",
                option="chain.start_time", value=[frames[0].times, frames[-1].times],
            )

    def interval_step(self) -> None:
        interval, request = self.interval, self.request
        steps = interval / request.dt_s
        if not float(steps).is_integer() or round(steps) * request.dt_s != interval:
            _refuse(
                "GM1-CHAIN-INTERVAL-STEP",
                f"the boundary interval {interval} s is not a whole number of {request.dt_s} s GM1 "
                "steps; the tables would be replaced in the middle of a step and the relaxation "
                "target would jump by the unelapsed part of the tendency",
                "choose run.dt_s dividing the history interval",
                option="run.dt_s", value=[interval, request.dt_s],
            )
        self.steps = steps

    def duration(self) -> None:
        frames, request = self.frames, self.request
        span = frames[-1].valid_s - frames[0].valid_s
        if request.duration_s > span:
            _refuse(
                "GM1-CHAIN-DURATION",
                f"run.duration_s = {request.duration_s} s reaches past the last frame "
                f"({span} s after the first); the boundary tendency would be "
                "extrapolated past the data",
                "shorten run.duration_s or extend chain.end_time",
                option="run.duration_s", value=request.duration_s,
            )

    # -- geometry ------------------------------------------------------------
    def projection(self) -> None:
        first = self.first
        projection = int(first.attributes.get("MAP_PROJ", -1))
        if (projection not in METRIC_PROJECTIONS and projection != 0) or \
                first.attributes.get("DX") != first.attributes.get("DY"):
            _refuse(
                "GM1-CHAIN-PROJECTION",
                f"MAP_PROJ = {projection} with DX = {first.attributes.get('DX')} and DY = "
                f"{first.attributes.get('DY')}: GM1 needs an isotropic metric spacing, and a "
                "latitude-longitude or anisotropic grid would be integrated with the wrong lengths",
                "run on a Lambert, polar-stereographic or Mercator grid, or an idealized "
                "(MAP_PROJ = 0) grid, with DX = DY",
                option="MAP_PROJ", value=projection,
            )
        self.facts["projection"] = {
            "MAP_PROJ": projection, "name": PROJECTION_NAMES[projection],
            **{k: first.attributes.get(k) for k in ("TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON")},
        }

    def placement(self) -> None:
        request = self.request
        self.geometry = request.placement.geometry(self.first, nx=request.nx, ny=request.ny)
        from .nest_vertical import VerticalWindow

        mass = VerticalWindow.for_stagger(self.geometry, "mass")
        self.msl = (slice(mass.j0, mass.j0 + mass.nj), slice(mass.i0, mass.i0 + mass.ni))
        g = self.geometry
        self.facts["window"] = {
            "parent_i0": g.parent_i0, "parent_j0": g.parent_j0,
            "parent_span_x": g.parent_span_x, "parent_span_y": g.parent_span_y,
            "origin_x_m": g.origin_x_m, "origin_y_m": g.origin_y_m,
        }

    def spacing(self) -> None:
        np = _np()
        first, request = self.first, self.request
        ratio = request.placement.refinement_ratio
        scale, source = window_map_scale(first, self.msl)
        self.facts["arwen_dx_m"] = float(first.attributes["DX"])
        self.facts["map_scale"] = {"window_mean_map_factor": scale, "source": source,
                                   "gm1_dx_m": float(np.float32(float(first.attributes["DX"]) / (ratio * scale)))
                                   if scale != 1.0 else float(first.attributes["DX"]) / ratio}
        for axis, spacing, declared in (("x", float(first.attributes["DX"]), request.dx_m),
                                        ("y", float(first.attributes["DY"]), request.dy_m)):
            true = spacing / (ratio * scale)
            word = np.float32(true)
            exact = scale != 1.0 or float(word) == true
            if not exact or np.float32(declared).tobytes() != word.tobytes():
                scaled = "" if scale == 1.0 else f" / map factor {scale:.9g} (the window mean)"
                _refuse(
                    "GM1-CHAIN-SPACING",
                    f"history d{axis} {spacing} m / ratio {ratio}{scaled} = {true} m and grid.d{axis}_m "
                    f"= {declared} m; GM1 cells would not be the history's cells on the ground, so "
                    "every horizontal length (advection, pressure gradient, diffusion) and every "
                    "boundary face would be off by the difference",
                    f"declare grid.d{axis}_m = {float(word)!r}" + ("" if scale != 1.0 else
                                                                   " with a ratio for which it is exact"),
                    option=f"grid.d{axis}_m", value=[spacing, ratio, scale, declared],
                )

    def terrain(self) -> None:
        np = _np()
        request = self.request
        hgt = self.first.array("HGT")[self.msl].astype(np.float64)
        relief = float(hgt.max() - hgt.min())
        self.facts["terrain_relief_m"] = relief
        self.facts["terrain_mean_m"] = float(hgt.mean())
        if not relief <= request.terrain_relief_limit_m:
            _refuse(
                "GM1-CHAIN-TERRAIN",
                f"terrain relief over the GM1 window is {relief} m, above chain."
                f"terrain_relief_limit_m = {request.terrain_relief_limit_m} m; GM1's Cartesian lane is "
                "flat, so air at one height above ground would sit at different absolute heights "
                "under the same GM1 level and the flow over and around the terrain would be absent",
                "move the window over flatter ground, or declare the relief the experiment accepts",
                option="chain.terrain_relief_limit_m", value=relief,
            )

    def map_factor(self) -> None:
        """GM1's remaining length error: ``max |m/m_window - 1|`` (``GM1_CHOICES['map_scale']``).

        GM1's spacing is the history's true spacing at the window-mean map factor (the spacing
        check), so a cell where ``m`` differs from that mean is that fraction too long or too short
        on the ground.  The limit is the same one; what it is held against is GM1's actual error.
        """

        np = _np()
        request = self.request
        mapfac, source = _mapfac_window(self.first, self.msl)
        scale = float(mapfac.mean())
        departure = float(np.max(np.abs(mapfac / scale - 1.0)))
        self.facts["map_factor_departure"] = departure
        self.facts["map_factor_source"] = source
        self.facts["map_factor_unit_departure"] = float(np.max(np.abs(mapfac - 1.0)))
        self.facts["map_factor_range"] = [float(mapfac.min()), float(mapfac.max())]
        if not departure <= request.map_factor_departure_limit:
            _refuse(
                "GM1-CHAIN-MAP-FACTOR",
                f"the map factor over the window ranges {float(mapfac.min()):.6g}..{float(mapfac.max()):.6g} "
                f"about its mean {scale:.6g}, so GM1's lengths (true at the mean) are off by up to "
                f"{departure} somewhere, above chain.map_factor_departure_limit = "
                f"{request.map_factor_departure_limit}; GM1 has no map-factor terms, so advection, "
                "pressure gradients and diffusion would be wrong by that fraction there",
                "make the window smaller, or place it where the map factor changes less",
                option="chain.map_factor_departure_limit", value=departure,
            )

    def coriolis(self) -> None:
        request, geometry = self.request, self.geometry
        centre = (geometry.parent_j0 + geometry.parent_span_y // 2,
                  geometry.parent_i0 + geometry.parent_span_x // 2)
        f_field, f_centre, source = _f_window(self.first, self.msl, centre)
        f_tolerance = max(0.5 * float(f_field.max() - f_field.min()), 1.0e-6 * abs(f_centre))
        self.facts["coriolis"] = {
            "window_centre_f_s_inv": f_centre, "tolerance_s_inv": f_tolerance,
            "declared_f_s_inv": request.fcor_s_inv, "declared_coriolis": request.coriolis,
            "source": source,
        }
        declared = request.fcor_s_inv if request.coriolis == 1 else 0.0
        rotating = request.coriolis == 1 or f_centre != 0.0
        if not rotating:
            return
        if request.coriolis != 1 or abs(declared - f_centre) > f_tolerance:
            _refuse(
                "GM1-CHAIN-CORIOLIS",
                f"GM1 declares coriolis = {request.coriolis}, fcor = {request.fcor_s_inv} s-1; the "
                f"window centre's f is {f_centre} s-1 (from {source}; range across the window "
                f"{float(f_field.min())}..{float(f_field.max())}). The history's winds and pressure "
                "are in rotating balance, so a GM1 domain with another f would turn that balance "
                "into a spurious acceleration from the first step",
                f"declare dynamics.coriolis = 1 and dynamics.fcor_s_inv = {f_centre!r}"
                if f_centre != 0.0 else "declare dynamics.coriolis = 0 (the history has no rotation)",
                option="dynamics.fcor_s_inv", value=[request.fcor_s_inv, f_centre],
            )

    # -- physics -----------------------------------------------------------
    def microphysics(self) -> None:
        request = self.request
        expected_mp = WRF_MP_PHYSICS.get(request.component)
        mp = self.first.attributes.get("MP_PHYSICS")
        if request.moisture == "vapour-only":
            self._vapour_only(mp)
            return
        if expected_mp is None or mp is None or int(mp) != expected_mp:
            _refuse(
                "GM1-CHAIN-MICROPHYSICS",
                f"the history ran {wrf_scheme_name(mp)} and GM1 runs {request.component} (WRF "
                f"option {expected_mp}); hydrometeors carry one scheme's size-distribution and "
                "density assumptions, and another scheme would fall, melt and reflect them "
                "differently from the first step",
                "run GM1 with the component transcribing the history's mp_physics "
                f"({', '.join(wrf_scheme_name(k) for k in sorted(CHAIN_SCHEMES))} are offered), "
                "or take water vapour alone (chain.moisture = \"vapour-only\", WRF's "
                "have_bcs_moist = .false.)",
                option="physics.component", value=[mp, request.component],
            )
        self.facts["microphysics"] = {
            "mp_physics": int(mp), "scheme": wrf_scheme_name(mp), "component": request.component,
            "species": list(_species_names(request.component)), "moisture": request.moisture,
            "transferred": list(transferred_species(request.component, request.moisture)),
        }

    def _vapour_only(self, mp: Any) -> None:
        """The vapour-only lane: any history scheme; its hydrometeors are measured, not transferred."""

        request = self.request
        self.facts["microphysics"] = {
            "mp_physics": None if mp is None else int(mp), "scheme": wrf_scheme_name(mp),
            "component": request.component, "species": list(_species_names(request.component)),
            "moisture": request.moisture,
            "transferred": list(transferred_species(request.component, request.moisture)),
            "not_transferred": [name for name in _species_names(request.component) if name != "qv"],
            "lane": MOISTURE_LANES["vapour-only"],
            "left_out_at_t0": left_out_condensate(self.first, self.msl),
        }

    def species(self) -> None:
        wanted = transferred_species(self.request.component, self.request.moisture)
        missing = [(name, WRF_SPECIES.get(name, "?")) for name in wanted
                   if name not in WRF_SPECIES or not self.first.has(WRF_SPECIES[name])]
        if missing:
            name, wrf_name = missing[0]
            others = [w for _, w in missing[1:]]
            _refuse(
                "GM1-CHAIN-SPECIES",
                f"GM1 carries {name} and the history frame has no {wrf_name}"
                + (f" (nor {', '.join(others)})" if others else "")
                + "; the scalar would start and be forced as zero",
                "write every species of the scheme to the history",
                option=name, value=[w for _, w in missing],
            )

    def variables(self) -> None:
        first = self.first
        for name in _required_variables(first):
            first.entry(name)
        self.facts["surface_record"] = surface_sources(first)

    def theta_m(self) -> None:
        np = _np()
        first, msl = self.first, self.msl
        theta_m = first.attributes.get("USE_THETA_M")
        record: dict[str, Any] = {"USE_THETA_M": theta_m}
        self.facts["theta"] = record
        if theta_m is None or int(theta_m) == 0:
            return
        if not first.has("THM"):
            _refuse(
                "GM1-CHAIN-THETA-M",
                "the frame declares USE_THETA_M = 1 and carries no THM, so T cannot be shown to "
                "be the dry potential temperature GM1's th3d holds; a moist theta would warm "
                "every moist column by about 0.6 qv theta",
                "write THM beside T, or run with use_theta_m = 0",
                option="USE_THETA_M", value=theta_m,
            )
        t = first.array("T")[(slice(None),) + msl].astype(np.float64) + WRF_T0_K
        qv = first.array("QVAPOR")[(slice(None),) + msl].astype(np.float64)
        thm = first.array("THM")[(slice(None),) + msl].astype(np.float64) + WRF_T0_K
        residual = float(np.max(np.abs(t * (1.0 + WRF_RV_OVER_RD * qv) - thm)))
        record["dry_T_residual_K"] = residual
        if residual > 1.0e-2:
            _refuse(
                "GM1-CHAIN-THETA-M",
                f"with USE_THETA_M = 1, (T+300)(1+Rv/Rd qv) differs from THM+300 by {residual} K, "
                "so T is not the dry potential temperature; GM1 would start moist columns warm",
                "write the dry potential temperature as T",
                option="USE_THETA_M", value=residual,
            )

    def model_top(self) -> None:
        """GM1's highest levels against every window column's own last level (record section 2).

        A GM1 scalar level above the history's last mass level, or a GM1 top face above its top face,
        would be filled by CM1's constant extension (``interp_on_restart``'s last plane): harmless for
        theta, winds and moisture, but an Exner function held constant with height is out of
        hydrostatic balance by g dz / (cp theta) per metre of extension, which the model would then
        adjust with a burst of vertical motion at its top.  The history must reach GM1's top.
        """

        np = _np()
        from .nest_vertical import VerticalWindow

        levels = self.request.levels
        nk = int(levels.nk)
        tops = {"mass": float(levels.zh[nk]), "w": float(levels.zf[nk + 1])}
        lowest: dict[str, float] = {}
        for stagger, top in tops.items():
            window = VerticalWindow.for_stagger(self.geometry, stagger)
            ghosted, nzold = column_heights(self.first, window, stagger)
            lowest[stagger] = float(np.min(ghosted[nzold]))
            if top > lowest[stagger]:
                _refuse(
                    "GM1-CHAIN-MODEL-TOP",
                    f"GM1's highest {'scalar level' if stagger == 'mass' else 'face'} is {top} m above ground "
                    f"and the lowest window column of the history has its last "
                    f"{'mass level' if stagger == 'mass' else 'face'} at {lowest[stagger]} m; GM1's levels "
                    "above it would take CM1's constant extension, and an Exner function held constant with "
                    "height is out of hydrostatic balance",
                    "lower GM1's model top under the history's last level",
                    option="grid.ztop_m", value=[top, lowest[stagger]],
                )
        self.facts["model_top"] = {"gm1_top_m": tops, "history_lowest_last_level_m": lowest}

    def crossing(self) -> None:
        np = _np()
        from .nest_vertical import VerticalWindow

        request, geometry, interval = self.request, self.geometry, self.interval
        umax = vmax = 0.0
        uw = VerticalWindow.for_stagger(geometry, "u")
        vw = VerticalWindow.for_stagger(geometry, "v")
        for frame in self.frames:
            umax = max(umax, float(np.max(np.abs(_parent_carrier(frame, "u", uw)))))
            vmax = max(vmax, float(np.max(np.abs(_parent_carrier(frame, "v", vw)))))
        crossing = min(
            request.nx * request.dx_m / umax if umax > 0 else math.inf,
            request.ny * request.dy_m / vmax if vmax > 0 else math.inf,
        )
        self.facts.update({"max_abs_u_m_s": umax, "max_abs_v_m_s": vmax, "crossing_time_s": crossing})
        if interval > crossing:
            _refuse(
                "GM1-CHAIN-CROSSING",
                f"the boundary interval is {interval} s and the fastest wind in the window "
                f"({max(umax, vmax)} m/s) crosses the GM1 domain in {crossing} s; a disturbance could "
                "enter and leave the domain between two frames, so linear interpolation would force "
                "the boundary with air that was never there",
                "write history more often, or enlarge the GM1 domain",
                option="chain.end_time", value=[interval, crossing],
            )


def admission_report(frames: Sequence[Any], request: ChainRequest) -> dict[str, Any]:
    """Every admission check, each passed, refused (with its refusal) or not reached.

    Runs the checks :func:`admit_frames` runs, in the same order, but goes on
    past a refusal wherever a later check does not depend on the refused one:
    ``{"checks": [{"id", "status": "pass"|"refused"|"not-checked", "refusal"?,
    "after"?}], "facts": {...}, "admitted": bool}``.
    """

    from .errors import CM1Refusal

    state = _Admission(frames, request)
    rows: list[dict[str, Any]] = []
    failed: set[str] = set()
    for name in ADMISSION_CHECKS:
        blocked = None
        if name != "frame_count" and "frame_count" in failed:
            blocked = "frame_count"
        elif name in _NEEDS_INTERVAL and state.interval is None:
            blocked = "cadence"
        elif name in _NEEDS_GEOMETRY and state.geometry is None:
            blocked = "placement"
        elif name in _NEEDS_SCHEME and "microphysics" in failed:
            blocked = "microphysics"
        if blocked is not None:
            rows.append({"id": name, "status": "not-checked", "after": blocked})
            continue
        try:
            getattr(state, name)()
        except CM1Refusal as refusal:
            failed.add(name)
            rows.append({"id": name, "status": "refused", "refusal": refusal.detail.as_dict()})
            continue
        rows.append({"id": name, "status": "pass"})
    return {"checks": rows, "facts": state.facts,
            "admitted": all(row["status"] == "pass" for row in rows)}


def admit_frames(frames: Sequence[Any], request: ChainRequest) -> dict[str, Any]:
    """Refuse, by name, every frame sequence the chain would map wrongly.

    Returns the measured admission record.  Order: container facts, time,
    geometry, then physics (:data:`ADMISSION_CHECKS`).
    """

    state = _Admission(frames, request)
    for name in ADMISSION_CHECKS:
        getattr(state, name)()
    facts = state.facts
    first = frames[0]
    return {
        "frames": [{"times": f.times, "directory": str(getattr(f, "directory", "") or ""),
                    "source": frame_source(f)} for f in frames],
        "producer": producer_of(first.attributes),
        "title": str(first.attributes.get("TITLE", "")),
        "interval_s": state.interval,
        "steps_per_interval": int(round(state.steps)),
        "projection": facts["projection"],
        "arwen_dx_m": facts["arwen_dx_m"],
        "terrain_relief_m": facts["terrain_relief_m"],
        "terrain_mean_m": facts["terrain_mean_m"],
        "map_factor_departure": facts["map_factor_departure"],
        "map_factor_source": facts["map_factor_source"],
        "map_factor_unit_departure": facts["map_factor_unit_departure"],
        "map_scale": facts["map_scale"],
        "coriolis": facts["coriolis"],
        "max_abs_u_m_s": facts["max_abs_u_m_s"], "max_abs_v_m_s": facts["max_abs_v_m_s"],
        "crossing_time_s": facts["crossing_time_s"],
        "microphysics": facts["microphysics"],
        "theta": facts["theta"],
        "surface_record": facts["surface_record"],
        "model_top": facts["model_top"],
        "hybrid_opt": first.attributes.get("HYBRID_OPT"),
        "window": facts["window"],
    }


# ---------------------------------------------------------------------------
# The t0 sounding.
# ---------------------------------------------------------------------------
def surface_sources(frame: Any) -> dict[str, str]:
    """Which history field gives each word of the sounding's surface line.

    WRF and ArWen real-data histories carry ``PSFC``, ``TH2`` or ``T2`` and
    ``Q2``.  Idealized histories often carry none of them; then the surface
    pressure is WRF's own extrapolation of the full pressure to the ground
    (``module_big_step_utilities_em.F:5566-5578`` ``phy_prep``, linear in height
    from the two lowest levels) and the surface theta and qv are the lowest
    level's, which is CM1's constant extension below the first level
    (``param.F:8165-8173``, the rule the vertical transfer uses).
    """

    return {
        "pressure": "PSFC" if frame.has("PSFC") else "P+PB extrapolated to the ground (WRF phy_prep)",
        "theta": ("TH2" if frame.has("TH2") else "T2 with the surface pressure" if frame.has("T2")
                  else "lowest-level T+300 (CM1's constant extension)"),
        "qv": "Q2" if frame.has("Q2") else "lowest-level QVAPOR (CM1's constant extension)",
    }


def _surface_fields(frame: Any, sl: Any) -> tuple[Any, Any, Any]:
    """Surface pressure (Pa), theta (K) and qv (kg/kg) over the window, float64."""

    np = _np()
    level0 = (0,) + tuple(sl)
    if frame.has("PSFC"):
        psfc = frame.array("PSFC")[sl].astype(np.float64)
    else:
        ph = frame.array("PH")[(slice(0, 3),) + tuple(sl)].astype(np.float64)
        phb = frame.array("PHB")[(slice(0, 3),) + tuple(sl)].astype(np.float64)
        z_if = (ph + phb) / WRF_GRAVITY_M_S2
        z_mid = 0.5 * (z_if[:-1] + z_if[1:])
        p = (frame.array("P")[(slice(0, 2),) + tuple(sl)].astype(np.float64)
             + frame.array("PB")[(slice(0, 2),) + tuple(sl)].astype(np.float64))
        w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
        psfc = w1 * p[0] + (1.0 - w1) * p[1]
    if frame.has("TH2"):
        th2 = frame.array("TH2")[sl].astype(np.float64)
    elif frame.has("T2"):
        th2 = frame.array("T2")[sl].astype(np.float64) * (CM1_P00_PA / psfc) ** (CM1_RD / CM1_CP)
    else:
        th2 = frame.array("T")[level0].astype(np.float64) + WRF_T0_K
    if frame.has("Q2"):
        q2 = frame.array("Q2")[sl].astype(np.float64)
    else:
        q2 = frame.array("QVAPOR")[level0].astype(np.float64)
    return psfc, th2, q2


def domain_mean_sounding(initial: Mapping[str, Any], frame: Any, geometry: Any,
                         levels: Any) -> bytes:
    """CM1 ``input_sounding`` (``base.F:495-607``) from the remapped t0 state.

    Row ``k``: GM1 scalar height ``zh(k)``, domain-mean theta (K), qv (g/kg),
    u and v (m/s; the means of each staggered lattice).  Header: domain-mean
    surface pressure (mb), surface theta and qv (g/kg) over the history cells
    the window covers (:func:`surface_sources` names where each comes from).
    """

    np = _np()
    sl = (slice(geometry.parent_j0, geometry.parent_j0 + geometry.parent_span_y),
          slice(geometry.parent_i0, geometry.parent_i0 + geometry.parent_span_x))
    psfc, th2, q2 = _surface_fields(frame, sl)
    lines = [f"{psfc.mean() / 100.0:.9g} {th2.mean():.9g} {q2.mean() * 1000.0:.9g}"]
    zh = np.asarray(levels.zh[1:int(levels.nk) + 1], dtype=np.float64)
    theta = initial["theta"].astype(np.float64).mean(axis=(1, 2))
    qv = initial["qv"].astype(np.float64).mean(axis=(1, 2))
    u = initial["u"].astype(np.float64).mean(axis=(1, 2))
    v = initial["v"].astype(np.float64).mean(axis=(1, 2))
    for k in range(zh.size):
        lines.append(
            f"{zh[k]:.9g} {theta[k]:.9g} {max(qv[k], 0.0) * 1000.0:.9g} {u[k]:.9g} {v[k]:.9g}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


# ---------------------------------------------------------------------------
# Prepared products.
# ---------------------------------------------------------------------------
INITIAL_ARENA = "chain-initial-state.f32le"
BOUNDARY_ARENA = "chain-boundary-frames.f32le"
SOUNDING = "input_sounding"
RECEIPT = "chain-receipt.json"


def prepare_chain(frames: Sequence[ArwenFrame], request: ChainRequest, out: Path | str,
                  *, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Admit, remap and write the initial state, boundary strips and sounding."""

    np = _np()
    root = Path(out)
    admission = admit_frames(frames, request)
    geometry = request.placement.geometry(frames[0], nx=request.nx, ny=request.ny)
    width = request.placement.spec_zone + request.placement.relax_zone
    root.mkdir(parents=True)
    carriers = carriers_of(request.component, request.moisture)
    initial_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    support = None
    with open(root / INITIAL_ARENA, "wb") as initial_arena, open(root / BOUNDARY_ARENA, "wb") as boundary_arena:
        for index, frame in enumerate(frames):
            state = remap_frame(frame, geometry, request.levels, request.component, request.moisture)
            support = state.pop("_support")
            if index == 0:
                sounding = domain_mean_sounding(state, frame, geometry, request.levels)
                (root / SOUNDING).write_bytes(sounding)
                for name, _ in carriers:
                    raw = np.ascontiguousarray(state[name], dtype="<f4").tobytes()
                    initial_rows.append({"name": name, "shape": list(state[name].shape),
                                         "offset": initial_arena.tell(), "nbytes": len(raw),
                                         "sha256": _sha256(raw)})
                    initial_arena.write(raw)
            for name, _ in carriers:
                for side in SIDES:
                    rows = strip(state[name], side, width)
                    raw = np.ascontiguousarray(rows, dtype="<f4").tobytes()
                    boundary_rows.append({"frame": index, "name": name, "side": side,
                                          "shape": list(rows.shape), "offset": boundary_arena.tell(),
                                          "nbytes": len(raw), "sha256": _sha256(raw)})
                    boundary_arena.write(raw)
    receipt = {
        "schema": PREPARED_SCHEMA,
        "chain_schema": CHAIN_SCHEMA,
        "contract": request.contract(),
        "admission": admission,
        "vertical_support": support,
        "table_width": width,
        "carriers": [{"name": n, "stagger": s} for n, s in carriers],
        "moisture": {"lane": request.moisture, "meaning": MOISTURE_LANES[request.moisture]},
        "initial_state": {"arena": INITIAL_ARENA, "rows": initial_rows,
                          "arena_sha256": _sha256((root / INITIAL_ARENA).read_bytes())},
        "boundary_frames": {"arena": BOUNDARY_ARENA, "rows": boundary_rows,
                            "arena_sha256": _sha256((root / BOUNDARY_ARENA).read_bytes()),
                            "count": len(frames)},
        "sounding": {"file": SOUNDING, "sha256": _sha256((root / SOUNDING).read_bytes())},
        "authority": dict(AUTHORITY),
        "gm1_choices": dict(GM1_CHOICES),
        "document": dict(document or {}),
    }
    (root / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


@dataclass(frozen=True)
class PreparedChain:
    """A prepared chain directory, hash-checked on open."""

    root: Path
    receipt: Mapping[str, Any]

    @classmethod
    def open(cls, directory: Path | str, *, contract: Mapping[str, Any] | None = None) -> "PreparedChain":
        root = Path(directory)
        try:
            receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _refuse(
                "GM1-CHAIN-PREPARED",
                f"{root} has no readable {RECEIPT} ({error}); the run would start from and be "
                "forced by products with no admission record",
                "run tools/arwen_chain_prepare.py for this document into chain.prepared_directory",
                option="chain.prepared_directory", value=str(root),
            )
        if receipt.get("schema") != PREPARED_SCHEMA:
            _refuse("GM1-CHAIN-PREPARED", f"{root}/{RECEIPT} is not {PREPARED_SCHEMA}",
                    "re-run the prepare tool", option="chain.prepared_directory", value=str(root))
        for key, arena in (("initial_state", INITIAL_ARENA), ("boundary_frames", BOUNDARY_ARENA)):
            if _sha256((root / arena).read_bytes()) != receipt[key]["arena_sha256"]:
                _refuse("GM1-CHAIN-PREPARED",
                        f"{root}/{arena} does not hash to its receipt; the run would read words the "
                        "admission never saw", "re-run the prepare tool into a new directory",
                        option="chain.prepared_directory", value=str(root))
        if _sha256((root / SOUNDING).read_bytes()) != receipt["sounding"]["sha256"]:
            _refuse("GM1-CHAIN-PREPARED", f"{root}/{SOUNDING} does not hash to its receipt",
                    "re-run the prepare tool into a new directory",
                    option="chain.prepared_directory", value=str(root))
        if contract is not None:
            recorded = receipt["contract"]
            for key, value in contract.items():
                if recorded.get(key) != value:
                    _refuse(
                        "GM1-CHAIN-PREPARED",
                        f"the prepared chain was admitted for {key} = {recorded.get(key)!r} and this "
                        f"document declares {value!r}; its boundary tables and initial state belong "
                        "to another grid, clock or window",
                        "re-run the prepare tool for this document",
                        option=f"chain.{key}", value=[recorded.get(key), value],
                    )
        return cls(root, receipt)

    @property
    def interval_s(self) -> float:
        return float(self.receipt["admission"]["interval_s"])

    @property
    def steps_per_interval(self) -> int:
        return int(self.receipt["admission"]["steps_per_interval"])

    @property
    def frame_count(self) -> int:
        return int(self.receipt["boundary_frames"]["count"])

    @property
    def width(self) -> int:
        return int(self.receipt["table_width"])

    @property
    def moisture(self) -> str:
        return str((self.receipt.get("moisture") or {}).get("lane", DEFAULT_MOISTURE))

    @property
    def sounding_bytes(self) -> bytes:
        return (self.root / SOUNDING).read_bytes()

    def _read(self, arena: str, row: Mapping[str, Any]) -> Any:
        np = _np()
        with open(self.root / arena, "rb") as handle:
            handle.seek(int(row["offset"]))
            raw = handle.read(int(row["nbytes"]))
        return np.frombuffer(raw, dtype="<f4").reshape(tuple(row["shape"])).astype(np.float32)

    def initial(self, name: str) -> Any:
        for row in self.receipt["initial_state"]["rows"]:
            if row["name"] == name:
                return self._read(INITIAL_ARENA, row)
        raise KeyError(name)

    def strip(self, frame: int, name: str, side: str) -> Any:
        for row in self.receipt["boundary_frames"]["rows"]:
            if row["frame"] == frame and row["name"] == name and row["side"] == side:
                return self._read(BOUNDARY_ARENA, row)
        raise KeyError((frame, name, side))


def host_interval_tables(prepared: PreparedChain, interval: int, *, th0: Any, pi0: Any,
                         species: Sequence[str],
                         transferred: Sequence[str] | None = None) -> dict[str, dict[str, tuple[Any, Any]]]:
    """The live nest's table set for boundary interval ``interval``, on the host.

    Keys are :class:`nest_live.NestLateralForcing` carrier names (``u``,
    ``v``, ``w``, ``pp``, ``th``, ``q0``..).  ``pp`` and ``th`` are GM1
    perturbations from ``pi0``/``th0``; the rest are the remapped fields.
    """

    if not 0 <= interval < prepared.frame_count - 1:
        _refuse(
            "GM1-CHAIN-DURATION",
            f"boundary interval {interval} is requested and the chain holds "
            f"{prepared.frame_count - 1}; the tables would be extrapolated past the last frame",
            "shorten run.duration_s", value=interval,
        )
    names = {"u": "u", "v": "v", "w": "w", "pp": "pi", "th": "theta"}
    names.update({f"q{s}": name for s, name in enumerate(species)
                  if transferred is None or name in transferred})
    out: dict[str, dict[str, tuple[Any, Any]]] = {}
    for carrier, source in names.items():
        sides = {}
        for side in SIDES:
            start = prepared.strip(interval, source, side)
            end = prepared.strip(interval + 1, source, side)
            if carrier == "pp":
                start, end = perturbation(start, pi0), perturbation(end, pi0)
            elif carrier == "th":
                start, end = perturbation(start, th0), perturbation(end, th0)
            sides[side] = interval_tables(start, end, prepared.interval_s)
        out[carrier] = sides
    return out


# ---------------------------------------------------------------------------
# The document.
# ---------------------------------------------------------------------------
def _cfg_refuse(code: str, reason: str, remedy: str, option: str, value: object) -> None:
    from .errors import refuse

    refuse(code=code, gate="config-grammar", reason=reason, remedy=remedy, option=option,
           value=value)


def validate_chain_document(config: Any) -> None:
    """The ``[chain]`` checks that need the rest of the document.

    Each refusal is a GM1 runtime path the chain does not execute, or a
    setting under which the ArWen state would be integrated wrongly.
    """

    c = config.chain
    if c is None:
        return
    if config.nest is not None:
        _cfg_refuse(
            "CM1-CFG-CHAIN-NEST",
            "[chain] and [nest] both declared: the chain domain's lateral tables are the nest "
            "forcing's own, and a live nest inside a chain domain would need the chain domain "
            "to be a parent while it is itself being forced; that composition has no receipt",
            "run the chain domain alone, or chain a finer domain from its published history",
            "chain", True,
        )
    g = config.grid
    if g.nx % c.refinement_ratio or g.ny % c.refinement_ratio:
        _cfg_refuse(
            "CM1-CFG-CHAIN-GRID",
            f"grid.nx x grid.ny = {g.nx} x {g.ny} is not a whole number of ArWen cells at ratio "
            f"{c.refinement_ratio}; the GM1 boundary faces would fall between ArWen faces",
            "declare grid.nx and grid.ny as multiples of chain.refinement_ratio",
            "grid.nx", [g.nx, g.ny, c.refinement_ratio],
        )
    if int(g.stretch_x) or int(g.stretch_y):
        _cfg_refuse(
            "CM1-CFG-CHAIN-GRID",
            "horizontal stretching places GM1 cells off ArWen/ratio spacing, so the capture "
            "operator's donor cells would not be the ArWen cells under them",
            "declare grid.stretch_x = 0 and grid.stretch_y = 0", "grid.stretch_x",
            [g.stretch_x, g.stretch_y],
        )
    b = config.boundaries
    lateral = (b.west, b.east, b.south, b.north)
    if lateral != (2, 2, 2, 2) or b.roflux or b.nudgeobc:
        _cfg_refuse(
            "CM1-CFG-CHAIN-LATERAL",
            f"lateral boundaries {list(lateral)} with roflux = {b.roflux}, nudgeobc = "
            f"{b.nudgeobc}: the specified-zone hooks run only in CM1's open (code 2) acoustic "
            "path; a periodic or rigid domain would wrap or wall the rows ArWen forces, and "
            "nudgeobc/roflux would pull them toward the base state or rescale ArWen's outflow",
            "declare wbc = ebc = sbc = nbc = 2 with roflux = 0 and nudgeobc = 0",
            "boundaries.west", {"lateral": list(lateral), "roflux": b.roflux, "nudgeobc": b.nudgeobc},
        )
    closures = config.closures
    if closures.moving_frame or closures.do_adapt_move:
        _cfg_refuse(
            "CM1-CFG-CHAIN-MOVING-FRAME",
            "a translating GM1 frame moves the domain over ground ArWen's fixed grid does not "
            "follow: the boundary strips were sampled at one ground position and GM1's winds "
            "would be frame-relative while ArWen's are ground-relative",
            "declare closures.moving_frame = false and do_adapt_move = false",
            "closures.moving_frame", [closures.moving_frame, closures.do_adapt_move],
        )
    d = config.dynamics
    if d.adaptive_timestep:
        _cfg_refuse(
            "CM1-CFG-CHAIN-ADAPTIVE-STEP",
            "an adaptive step would stop the boundary interval being a whole number of steps, "
            "so tables would be reloaded mid-step",
            "declare dynamics.adaptive_timestep = false", "dynamics.adaptive_timestep", True,
        )
    if d.beta_plane:
        _cfg_refuse(
            "CM1-CFG-CHAIN-CORIOLIS",
            "betaplane = 1 varies f about GM1's own domain centre with CM1's beta, not ArWen's "
            "map latitude; the rotation would disagree with the balance ArWen's state carries",
            "declare dynamics.beta_plane = 0 with fcor at the window centre", "dynamics.beta_plane", 1,
        )
    isnd = 7 if config.initialization is None else int(config.initialization.isnd)
    if isnd != 7:
        _cfg_refuse(
            "CM1-CFG-CHAIN-INITIALIZATION",
            f"initialization.isnd = {isnd}: the chain's base state is CM1's isnd=7 sounding of "
            "the ArWen t0 column; another sounding lane would subtract a base state the "
            "remapped perturbations were not taken from",
            "declare initialization.isnd = 7 and pass the prepared input_sounding", "initialization.isnd", isnd,
        )
    span = parse_wrf_time(c.end_time) - parse_wrf_time(c.start_time)
    if config.run.duration_s > span:
        _cfg_refuse(
            "CM1-CFG-CHAIN-DURATION",
            f"run.duration_s = {config.run.duration_s} s and the chain window is {span} s; the "
            "boundary would be extrapolated past chain.end_time",
            "shorten run.duration_s or extend chain.end_time", "run.duration_s", config.run.duration_s,
        )


def chain_request_from_config(config: Any) -> ChainRequest:
    from .config import require_vertical_coordinate

    c = config.chain
    return ChainRequest(
        nx=int(config.grid.nx), ny=int(config.grid.ny), dx_m=float(config.grid.dx_m),
        dy_m=float(config.grid.dy_m), levels=require_vertical_coordinate(config),
        dt_s=float(config.run.dt_s), duration_s=float(config.run.duration_s),
        component=str(config.physics.component), coriolis=int(config.dynamics.coriolis),
        fcor_s_inv=float(config.dynamics.fcor_s_inv), start_time=c.start_time,
        end_time=c.end_time,
        placement=ChainPlacement(
            parent_i0=c.parent_i0, parent_j0=c.parent_j0, refinement_ratio=c.refinement_ratio,
            spec_zone=c.spec_zone, relax_zone=c.relax_zone,
        ),
        terrain_relief_limit_m=float(c.terrain_relief_limit_m),
        map_factor_departure_limit=float(c.map_factor_departure_limit),
        moisture=str(getattr(c, "moisture", None) or DEFAULT_MOISTURE),
    )


@dataclass(frozen=True)
class ChainBinding:
    prepared: PreparedChain
    spec_zone: int
    relax_zone: int


def require_chain_execution(config: Any, *, restart_dir: Any, initial_condition_only: bool) -> None:
    """Refuse the chained runs this first version does not execute (host, no disk)."""

    if config.chain is None:
        return
    if restart_dir is not None or config.run.restarting:
        _refuse(
            "GM1-CHAIN-RESTART",
            "a restart of a chain domain carries its state but not its boundary interval "
            "position or its prepared tables' identity, so a resume would re-write the t0 "
            "initial state over the restored one or force it from the wrong interval",
            "drop --restart-dir and run.irst for the chained run", option="restart-dir",
            value=None if restart_dir is None else str(restart_dir),
        )
    if initial_condition_only:
        _refuse(
            "GM1-CHAIN-INITIAL-CONDITION",
            "run.iconly = 1 publishes the host initializer's state without opening a device, and "
            "the chain writes its ArWen initial state on the device; the frame would be the "
            "sounding's horizontally uniform state labelled as the chain's",
            "drop run.iconly; inspect the prepared chain-initial-state instead",
            option="run.iconly", value=1,
        )


def require_chain(config: Any, *, sounding_bytes: bytes) -> ChainBinding | None:
    """Open the prepared chain for this document and check its base-state sounding."""

    if config.chain is None:
        return None
    request = chain_request_from_config(config)
    prepared = PreparedChain.open(config.chain.prepared_directory, contract=request.contract())
    if _sha256(bytes(sounding_bytes)) != prepared.receipt["sounding"]["sha256"]:
        _refuse(
            "GM1-CHAIN-SOUNDING",
            "the sounding this run builds its base state from is not the prepared chain's t0 "
            "column; th3d and pp3d would be ArWen's state minus another base state",
            f"pass --sounding {Path(config.chain.prepared_directory) / SOUNDING}",
            option="sounding", value=_sha256(bytes(sounding_bytes)),
        )
    return ChainBinding(prepared, int(config.chain.spec_zone), int(config.chain.relax_zone))


__all__ = [
    "ChainBinding", "chain_request_from_config", "require_chain", "require_chain_execution",
    "validate_chain_document",
    "AUTHORITY", "CHAIN_SCHEMA", "FRAME_SCHEMA", "GM1_CHOICES", "PREPARED_SCHEMA",
    "REFUSAL_CODES", "ADMISSION_CHECKS", "CHAIN_SCHEMES", "DEFAULT_MOISTURE", "MOISTURE_LANES",
    "box_mass_slice", "chain_spacing_m", "left_out_condensate", "transferred_species", "window_map_scale", "PROJECTION_NAMES", "WRF_MP_SCHEME_NAMES",
    "admission_report", "component_for_mp_physics", "frame_source", "producer_of",
    "surface_sources", "wrf_scheme_name", "wrf_time",
    "ArwenFrame", "ChainPlacement", "ChainRequest", "ColumnMap",
    "PreparedChain", "admit_frames", "build_column_map", "carriers_of", "column_heights",
    "domain_mean_sounding", "host_interval_tables", "interpolate_columns", "interval_tables",
    "parse_wrf_time", "perturbation", "prepare_chain", "relaxation_target", "remap_frame",
    "strip", "write_frame",
]
