"""CM1 r21.1 lower boundary condition and surface drag on the CUDA device.

GM1's Cartesian lane has run with CM1's free-slip rigid bottom (``bbc=1``),
which is what CM1's own supercell deck uses.  A free-slip lower boundary
generates no surface horizontal vorticity, so a near-ground vortex has
nothing to tilt and stretch.  This module owns the three other pieces CM1
changes when the lower boundary stops being free-slip, in CM1's own source
order, entirely on the device:

* ``getcecd`` with ``cecd=1`` -- the constant drag coefficient that idealized
  surface-drag studies use -- producing ``cd``, ``znt`` and ``ust`` from the
  lowest model level wind (``src/sfcphys.F:120-136``);
* ``tkebc``'s ``bbc=2/3`` lower face, which replaces the free-slip zero
  eddy viscosity at ``k=1`` with ``karman*znt*ust``
  (``src/turb.F:3244-3306``);
* ``sfcstress``, the ``t13``/``t23`` lower boundary values that the momentum
  stress divergence consumes, for no-slip and semi-slip
  (``src/turb.F:4548-4605``), with ``stau`` (``src/turb.F:4692-4700``);
* the ``bbc=3`` surface dissipation rate that replaces the interior
  ``dissten`` average in dissipative heating (``src/solve1.F:950-962``).

Every wind that reaches the surface here is ground-relative.  CM1 forms
``ugr = ua + umove`` and ``vgr = va + vmove`` at ``src/turb.F:296-307``
before any surface quantity is computed, and ``u1``/``v1``/``s1`` are built
from those at ``src/turb.F:539-541``.  In a translating frame the carried
wind is not the wind the ground sees; applying drag to it would drag the
storm's frame-relative wind instead of the wind blowing over the surface.
The frame velocity is therefore part of this operator's selection and is
refused, not defaulted.

Python allocates, validates and orders launches.  Every evolving word stays
on the device.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


SURFACE_DRAG_SCHEMA = "gm1.cuda-surface-drag/v1"
SURFACE_DRAG_ABI = "gm1.cuda-surface-drag-kernels/v1"
SURFACE_DRAG_THREADS = 128
SURFACE_DRAG_GATE = "surface-drag"

SURFACE_DRAG_KERNEL_NAMES = (
    "cm1_sfc_ground_relative_wind_v1",
    "cm1_sfc_getcecd_cnstcd_v1",
    "cm1_sfc_tkebc_lower_v1",
    "cm1_sfc_stress_no_slip_v1",
    "cm1_sfc_stress_semi_slip_v1",
    "cm1_sfc_stau_v1",
    "cm1_sfc_dissipation_v1",
    # src/sfcphys.F:83-99,102-118,138-195,197-236 -- the getcecd branches CM1
    # reaches for set_znt=1, set_ust=1 and the wind-dependent water drag, and
    # the isfcflx=1 enthalpy exchange coefficients.
    "cm1_sfc_getcecd_set_znt_v1",
    "cm1_sfc_getcecd_set_ust_v1",
    "cm1_sfc_getcecd_water_v1",
    "cm1_sfc_getcecd_chcq_v1",
    # src/turb.F:292 surface pressure and src/sfcphys.F:341-361 sfcflux.
    "cm1_sfc_psfc_v1",
    "cm1_sfc_flux_v1",
)

SURFACE_DRAG_CLAIM_LIMIT = (
    "CM1 r21.1 bbc=1/2/3 lower boundary with sfcmodel=1 cecd=1 constant-Cd "
    "getcecd, tkebc lower face, sfcstress and the bbc=3 surface dissipation "
    "rate, for serial flat-ground Cartesian axisymm=0 sgsmodel=1 grids with "
    "four open lateral boundaries and use_avg_sfc false; supplied live "
    "momentum, density, metric, eddy-coefficient and height carriers only; "
    "the roughness length and log-layer ratio pass through CUDA expf/logf "
    "rather than the platform libm, so words downstream of them are a few "
    "units in the last place from a native default-REAL build; terrain, "
    "pertflx, external surface-layer models, oceanmodel, "
    "PBL schemes, axisymmetry, MPI halos, forecast, storm and tornado claims "
    "are excluded"
)

SURFACE_LAYER_CLAIM_LIMIT = (
    "CM1 r21.1 sfcmodel=1 getcecd in all four source branches -- set_znt=1 "
    "prescribed roughness, set_ust=1 prescribed friction velocity, cecd=1 "
    "constant drag, and cecd=2 Deacon / cecd=3 Fairall-Donelan wind-dependent "
    "water drag with the land branch -- plus the isfcflx=1 enthalpy exchange "
    "coefficients, src/turb.F:292 surface pressure, the sfcflux computed "
    "branch and its scalar lower flux; supplied live surface carriers only; "
    "set_flx=1 prescribed fluxes, the testcase overrides, use_avg_sfc, "
    "pertflx, the moisture budget accumulator, sfcdiags, oceanmodel=2, "
    "sfcmodel=2..7 and every PBL scheme are excluded"
)

# CM1 r21.1 source anchors.  Every formula below cites one of these.
SOURCE_ANCHORS = MappingProxyType(
    {
        "namelist_documentation": "README.namelist:574-581,1016-1042",
        "bbc_range_check": "src/param.F:1518-1532",
        "bbc3_requires_sfcmodel": "src/param.F:2073-2088",
        "sfcmodel_forces_bbc3": "src/param.F:873-889",
        "sfcmodel1_cecd_range": "src/param.F:1349-1365",
        "isfcflx_requires_sfcmodel": "src/param.F:2090-2106",
        "surface_enthalpy_flux": "src/sfcphys.F:275-453",
        "surface_enthalpy_flux_consumer": "src/turbtend.F:517-519",
        "ground_relative_wind": "src/turb.F:296-307",
        "lowest_level_wind_and_height": "src/turb.F:532-555",
        "getcecd_dispatch": "src/turb.F:614-616",
        "getcecd_constant_drag": "src/sfcphys.F:120-136",
        "getcecd_prescribed_roughness": "src/sfcphys.F:83-99",
        "getcecd_prescribed_friction_velocity": "src/sfcphys.F:102-118",
        "getcecd_wind_dependent_water_drag": "src/sfcphys.F:138-195",
        "getcecd_enthalpy_exchange": "src/sfcphys.F:197-236",
        "surface_pressure": "src/turb.F:290-293",
        "surface_pressure_weights": "src/param.F:9038-9040",
        "surface_enthalpy_flux_normal": "src/sfcphys.F:341-361",
        "saturation_mixing_ratio": "src/cm1libs.F:16-62",
        "surface_flux_scalar_consumer": "src/turbtend.F:517-521,587-596",
        "surface_flux_scalar_boundary": "src/turbtend.F:625-633",
        "surface_flux_dispatch": "src/turb.F:645-649",
        "surface_flux_compatibility": "src/init_surface.F:207-227",
        "landuse_table": "src/init_surface.F:365-466",
        "landuse_call": "src/init_surface.F:239-243",
        "initsfc_branches": "src/init_surface.F:40-145",
        "prescribed_roughness_override": "src/init_surface.F:246-254",
        "surface_reality_checks": "src/init_surface.F:270-322",
        "surface_halo_zero_gradient": "src/turb.F:1138-1178; src/bc.F:12-172",
        "tkebc_lower_face": "src/turb.F:3244-3306",
        "sfcstress_free_slip": "src/turb.F:4536-4546",
        "sfcstress_no_slip": "src/turb.F:4548-4564",
        "sfcstress_semi_slip": "src/turb.F:4583-4605",
        "surface_stress_magnitude": "src/turb.F:4692-4700",
        "momentum_lower_flux_consumer": "src/turbtend.F:1442-1460,1454",
        "momentum_lower_flux_consumer_v": "src/turbtend.F:2212-2230,2224",
        "momentum_lower_flux_explicit": "src/turbtend.F:1411-1421,1417",
        "diff2_no_slip_ground_relative": "src/diff2.F:940-951",
        "diff2_semi_slip": "src/diff2.F:952-965",
        "surface_dissipative_heating": "src/solve1.F:917-988",
        "karman": "src/constants.F:23",
        "deck_defaults": "run/namelist.input:70-71,128-129,137-140",
        # bbc=2 with no surface layer: CM1 allocates znt and ust as zero and
        # init_surface only fills them for sfcmodel>=1 or bbc=3, so tkebc's
        # karman*znt*ust lower face is exactly zero there.  This port keeps
        # those carriers zero for the same selection rather than inventing a
        # roughness the source never assigns.
        "surface_state_zero_initialization": "src/cm1.F:1506-1514",
        "surface_state_filled_only_for_surface_layer": "src/init_surface.F:40,239-243",
    }
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/constants.F": (
            "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/sfcphys.F": (
            "7cb54de579b8bb038285430890e075ab19ed2e6d93158bd50d656209d64696a7"
        ),
        "src/turb.F": (
            "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d"
        ),
        "src/turbtend.F": (
            "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
        "src/diff2.F": (
            "00ebe1dc8fc3060758e99eb80f8167108fc81e5bd534781ddcfff0a1c46c1696"
        ),
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
    }
)

# src/constants.F:23
CM1_KARMAN_F32 = 0.40

# src/sfcphys.F:127-128.  getcecd's cecd=1 branch is
#
#     cd  = max(1.0e-10, cnstcd)
#     znt = 10.0/(exp(karman/sqrt(cd)) - 1.0)
#
# and CM1 bounds cnstcd nowhere: src/param.F only broadcasts and prints it
# (src/param.F:580, src/param.F:2948) and README.namelist:1040-1041 gives no
# range.  A drag coefficient above one is not unphysical arithmetic either:
# it makes znt larger than the 10 m reference height, and both logarithms at
# src/sfcphys.F:130 still take arguments strictly above one for any znt > 0,
# so the log-layer wind ratio stays strictly positive.  This port therefore
# admits every Cd whose roughness length exists, and refuses only the two
# ends where it does not:
#
# * below CNSTCD_BINARY32_ROUGHNESS_FLOOR, karman/sqrt(Cd) exceeds binary32's
#   exponential range, exp overflows, and znt = 10/inf is exactly zero, so
#   1/znt is infinite;
# * at or above CNSTCD_BINARY32_ROUGHNESS_CEILING, karman/sqrt(Cd) falls to
#   binary32's 2**-24, exp of it rounds to exactly 1.0, the subtraction gives
#   zero and znt = 10/0 is infinite.
#
# Both are the smallest/largest binary32 value on the wrong side, bracketed
# one unit in the last place by
# tests/test_cuda_surface_drag.py::test_the_drag_coefficient_bounds_are_exactly_where_the_roughness_stops.
CNSTCD_BINARY32_ROUGHNESS_FLOOR = 2.0325873265392147e-05
CNSTCD_BINARY32_ROUGHNESS_CEILING = 45036003655680.0

FREE_SLIP = 1
NO_SLIP = 2
SEMI_SLIP = 3

# src/sfcphys.F:5-9.  The Fairall/Donelan water-drag line, as CM1's own
# binary32 parameters: Cd rises linearly from dcd1 at dwsp1 to dcd2 at dwsp2
# and is clamped to that interval.
CM1_DCD1_F32 = 1.0e-3
CM1_DCD2_F32 = 2.4e-3
CM1_DWSP1_F32 = 5.0
CM1_DWSP2_F32 = 25.0

# src/constants.F:107-160 as CM1 builds them for its own defaults.  They are
# written here as the two-operand quotients CM1 forms so the binary32 word is
# the one the compiled model carries, not a decimal literal near it.
CM1_RD_F32 = 287.04
CM1_RV_F32 = 461.5
CM1_CP_F32 = 1005.7
CM1_P00_F32 = 1.0e5

# CM1 r21.1 ``getcecd`` reaches exactly one of four mutually exclusive
# branches (src/sfcphys.F:81-195), in this order.
GETCECD_SET_ZNT = "set_znt"
GETCECD_SET_UST = "set_ust"
GETCECD_CONSTANT_CD = "cnstcd"
GETCECD_WIND_DEPENDENT = "wind_dependent"


@dataclass(frozen=True, slots=True)
class SurfaceDragSelection:
    """CM1's own selector names for one lower-boundary configuration.

    ``bbc``, ``sfcmodel``, ``isfcflx``, ``cecd`` and ``cnstcd`` are the
    namelist names in ``README.namelist:574-581`` and
    ``README.namelist:1016-1042``.  ``u_move_m_s``/``v_move_m_s`` are
    ``base.F``'s ``umove``/``vmove``: they are not a surface option, they are
    what makes the surface wind ground-relative at ``src/turb.F:303-304``.
    """

    bbc: int = FREE_SLIP
    sfcmodel: int = 0
    isfcflx: int = 0
    cecd: int = 1
    cnstcd: float = 0.001
    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0
    # run/namelist.input:139,148-151 -- the remaining sfcmodel=1 options.
    # ``cnstce`` is the constant enthalpy exchange coefficient isfcflx=1
    # uses over water with cecd=1 (src/sfcphys.F:203-205); ``set_znt`` and
    # ``set_ust`` are the two branches that take precedence over ``cecd``
    # entirely (src/sfcphys.F:81-118).
    cnstce: float = 0.001
    set_znt: int = 0
    cnst_znt: float = 0.16
    set_ust: int = 0
    cnst_ust: float = 0.25

    @property
    def active(self) -> bool:
        """True when CM1 replaces the free-slip lower boundary values."""

        return self.bbc != FREE_SLIP

    @property
    def uses_surface_layer(self) -> bool:
        """True when ``getcecd`` runs, i.e. when ``ust``/``znt`` are needed."""

        return self.sfcmodel >= 1

    @property
    def getcecd_branch(self) -> str:
        """Which of ``getcecd``'s four branches this selection reaches.

        ``src/sfcphys.F:81-195`` tests ``set_znt`` first, then ``set_ust``,
        then ``cecd.eq.1``, and falls through to the wind-dependent water
        drag.  The order is the source's, not a preference: a document that
        declares both ``set_znt=1`` and ``cecd=2`` gets the roughness branch
        in CM1, and gets it here.
        """

        if self.set_znt == 1:
            return GETCECD_SET_ZNT
        if self.set_ust == 1:
            return GETCECD_SET_UST
        if self.cecd == 1:
            return GETCECD_CONSTANT_CD
        return GETCECD_WIND_DEPENDENT

    @property
    def uses_surface_enthalpy_flux(self) -> bool:
        """True when ``sfcflux`` runs and the scalar lower flux is nonzero.

        ``src/turb.F:645-649`` calls ``sfcflux`` only inside the
        ``sfcmodel.eq.1`` block, and ``src/init_surface.F:212-214`` only sets
        ``dosfcflx`` when ``isfcflx=1``; ``src/turbtend.F:519,595`` is the
        one place the resulting flux reaches a scalar tendency.
        """

        return self.isfcflx == 1 and self.sfcmodel >= 1

    def record(self) -> dict[str, object]:
        return {
            "bbc": self.bbc,
            "sfcmodel": self.sfcmodel,
            "isfcflx": self.isfcflx,
            "cecd": self.cecd,
            "cnstcd": self.cnstcd,
            "cnstce": self.cnstce,
            "set_znt": self.set_znt,
            "cnst_znt": self.cnst_znt,
            "set_ust": self.set_ust,
            "cnst_ust": self.cnst_ust,
            "getcecd_branch": self.getcecd_branch,
            "surface_enthalpy_flux": self.uses_surface_enthalpy_flux,
            "u_move_m_s": self.u_move_m_s,
            "v_move_m_s": self.v_move_m_s,
            "authority": dict(SOURCE_ANCHORS),
            "validation": surface_validation_markers(self),
        }


FREE_SLIP_SELECTION = SurfaceDragSelection()


# --------------------------------------------------------------------------
# What was measured, and what was not.
#
# Every selector below is reachable from a document.  This table says, per
# selector, how far the evidence goes, so a run receipt and ``doctor`` can
# say it too rather than leaving a reader to assume that "selectable" and
# "measured" are the same word.  ``VALIDATED`` means a raw-word comparison
# against CM1 r21.1's own statements compiled by gfortran; ``UNVALIDATED``
# names the specific thing that was not checked.
# --------------------------------------------------------------------------

VALIDATED = "validated"
UNVALIDATED = "unvalidated"

_NATIVE_RAW_WORD = (
    "raw-word identical to CM1 r21.1's own statements compiled by gfortran "
    "at -O0 and -O2 with -ffp-contract=off, for both the host transcription "
    "and the packaged CUDA kernel text compiled for the host "
    "(tools/oracle/surface_layer_harness.f90, "
    "tools/oracle/surface_layer_kernel_oracle.py)"
)
_DEVICE_EXACT = (
    "bit-identical on an RTX 5070 Ti (sm_120), "
    "evidence/validation/gm1-surface-layer-device-v1/receipt.json"
)
_DEVICE_ULP = (
    "within 3 units in the last place on an RTX 5070 Ti (sm_120) through "
    "CUDA's expf and logf, "
    "evidence/validation/gm1-surface-layer-device-v1/receipt.json"
)
_TURBSZ_RAW_WORD = (
    "raw-word identical to CM1 r21.1 turbsz compiled by gfortran, with the "
    "surface flux on and off, through the whole implicit column "
    "(tools/oracle/turbsz_lower_flux_harness.f90, "
    "tools/oracle/turbsz_lower_flux_kernel_oracle.py)"
)

SURFACE_VALIDATION = MappingProxyType(
    {
        "bbc": (VALIDATED, "src/turb.F:4548-4605 sfcstress, sealed before this work"),
        "cnstcd": (VALIDATED, _NATIVE_RAW_WORD + "; " + _DEVICE_EXACT),
        "set_znt": (VALIDATED, _NATIVE_RAW_WORD + "; " + _DEVICE_EXACT),
        "set_ust": (VALIDATED, _NATIVE_RAW_WORD + "; " + _DEVICE_ULP),
        "cecd": (VALIDATED, _NATIVE_RAW_WORD + "; " + _DEVICE_ULP),
        "cnstce": (VALIDATED, _NATIVE_RAW_WORD + "; " + _DEVICE_EXACT),
        "isfcflx": (
            VALIDATED,
            _NATIVE_RAW_WORD + "; " + _TURBSZ_RAW_WORD + "; " + _DEVICE_EXACT,
        ),
        "initsfc": (
            VALIDATED,
            "src/init_surface.F:40-227 through the sealed "
            "cm1core.surface_initialization operator and its native harness",
        ),
        "lu0": (
            VALIDATED,
            "src/init_surface.F:441-466 against the packaged LANDUSE.TBL "
            "record, whose bytes are pinned to CM1 r21.1's own run/LANDUSE.TBL",
        ),
        "season": (VALIDATED, "the same packaged LANDUSE.TBL record"),
    }
)

# Named once, so the receipt, doctor and the report say the same words.
SURFACE_UNVALIDATED = MappingProxyType(
    {
        "device_transcendentals": (
            "the device comparison covers one card and one grid size.  On an "
            "RTX 5070 Ti (sm_120, "
            "evidence/validation/gm1-surface-layer-device-v1) the cecd=1 "
            "constant-drag branch, the set_znt=1 prescribed-roughness branch "
            "and the isfcflx=1 enthalpy flux over them are bit-identical to "
            "the host words this port's gfortran oracle validates; no other "
            "architecture, driver or CUDA runtime has been compared, and "
            "logf/expf/powf are not specified to the last place across them."
        ),
        "device_last_place_difference": (
            "this branch's device words are not bit-identical to the host "
            "reference: on an RTX 5070 Ti they differ by up to 3 units in "
            "the last place through CUDA's expf and logf, and the "
            "wind-dependent water drag carries that difference round its "
            "convergence loop.  A trajectory through it is reproducible "
            "against the same device, not against a CPU reference."
        ),
        "integrated_surface_trajectory": (
            "no multi-step cm1.exe-versus-GM1 differential was run with a "
            "surface layer on.  Each surface term is raw-word exact against "
            "CM1's own statements, and the lower boundary they feed was "
            "sealed before this work, but the composition of drag, surface "
            "fluxes and the storm over many steps was not measured against "
            "the reference model."
        ),
        "spatially_varying_surface_in_a_storm": (
            "initsfc=2 and initsfc=4 build a land/water contrast and cecd=2/3 "
            "make the drag depend on the 10 m wind over water; both are "
            "raw-word exact per column, but no run has been made in which "
            "the contrast interacts with a developing storm."
        ),
        "surface_moisture_budget": (
            "src/sfcphys.F:439-447 accumulates qvflux into qbudget(8); this "
            "port does not carry that accumulator, so a moisture budget "
            "closed over a run with isfcflx=1 will not balance by the "
            "surface flux term."
        ),
        "surface_diagnostics": (
            "src/turb.F:651-656 sfcdiags and the hfx/qfx/t2/q2/br/zol "
            "diagnostics it writes are not ported; nothing reads them in "
            "this lane, and no output carries them."
        ),
    }
)


def surface_validation_markers(selection: SurfaceDragSelection) -> dict[str, object]:
    """What a receipt has to say about this selection's evidence."""

    if not selection.active:
        return {
            "schema": "gm1.surface-validation/v1",
            "surface": False,
            "validated": {},
            "unvalidated": {},
        }
    keys = ["bbc"]
    if selection.uses_surface_layer:
        keys.append(selection.getcecd_branch.replace("wind_dependent", "cecd"))
        keys.append("initsfc")
        keys.append("lu0")
        keys.append("season")
    if selection.isfcflx == 1:
        keys.extend(("isfcflx", "cnstce"))
    unvalidated = {"device_transcendentals", "integrated_surface_trajectory"}
    if selection.uses_surface_layer and selection.getcecd_branch in (
        GETCECD_SET_UST,
        GETCECD_WIND_DEPENDENT,
    ):
        unvalidated.add("device_last_place_difference")
    if selection.isfcflx == 1:
        unvalidated.update(("surface_moisture_budget", "surface_diagnostics"))
    if selection.cecd in (2, 3) or selection.getcecd_branch == GETCECD_WIND_DEPENDENT:
        unvalidated.add("spatially_varying_surface_in_a_storm")
    return {
        "schema": "gm1.surface-validation/v1",
        "surface": True,
        "validated": {
            key: {"status": SURFACE_VALIDATION[key][0], "evidence": SURFACE_VALIDATION[key][1]}
            for key in dict.fromkeys(keys)
            if key in SURFACE_VALIDATION
        },
        "unvalidated": {
            key: {"status": UNVALIDATED, "not_checked": SURFACE_UNVALIDATED[key]}
            for key in sorted(unvalidated)
        },
    }


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_surface_drag_selection(
    selection: SurfaceDragSelection, *, option_prefix: str = "closures"
) -> SurfaceDragSelection:
    """Refuse every lower-boundary configuration this operator cannot run.

    Each refusal names the concrete wrong result it prevents, not the rule
    it enforces.
    """

    if not isinstance(selection, SurfaceDragSelection):
        raise TypeError("selection must be a SurfaceDragSelection")
    for name in ("bbc", "sfcmodel", "isfcflx", "cecd"):
        _integer(f"{option_prefix}.{name}", getattr(selection, name))
    if isinstance(selection.cnstcd, bool) or not isinstance(
        selection.cnstcd, (int, float)
    ):
        raise TypeError(f"{option_prefix}.cnstcd must be a finite real number")
    # src/turb.F:303-304 adds the frame velocity to every carried wind before
    # the surface sees it, so the frame is checked before anything reads it.
    for name in ("u_move_m_s", "v_move_m_s"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{option_prefix}.{name} must be a finite real number")
        if not math.isfinite(float(value)):
            refuse(
                code="CM1-SFC-FRAME-VELOCITY",
                gate=SURFACE_DRAG_GATE,
                option=f"{option_prefix}.{name}",
                value=value,
                reason=(
                    "turb.F:303-304 adds the frame velocity to every carried "
                    "wind before the surface sees it; a non-finite frame "
                    "velocity makes the whole surface wind, its friction "
                    "velocity and every surface stress word not-a-number, and "
                    "the momentum tendency it feeds with them"
                ),
                remedy="declare a finite storm-motion speed in m/s",
            )

    # src/param.F:1518-1532
    if selection.bbc not in (FREE_SLIP, NO_SLIP, SEMI_SLIP):
        refuse(
            code="CM1-SFC-BBC-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.bbc",
            value=selection.bbc,
            reason=(
                "CM1 has exactly three wind lower boundary conditions; any "
                "other value would leave t13 and t23 at k=1 holding the "
                "free-slip copy of the level above while the run claimed a "
                "surface it never applied"
            ),
            remedy="set bbc to 1 (free slip), 2 (no slip) or 3 (semi slip)",
        )
    # src/param.F:2073-2088
    if selection.bbc == SEMI_SLIP and selection.sfcmodel < 1:
        refuse(
            code="CM1-SFC-SEMI-SLIP-WITHOUT-SURFACE-LAYER",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.sfcmodel",
            value=selection.sfcmodel,
            reason=(
                "the semi-slip surface stress is ust**2 times the surface "
                "wind direction, and only a surface layer computes ust; with "
                "no surface layer ust stays zero, so t13 and t23 at k=1 would "
                "be exactly zero and the run would be frictionless while "
                "reporting drag"
            ),
            remedy="set sfcmodel=1 with bbc=3, or use bbc=2 for no slip",
        )
    # src/param.F:873-889 changes bbc for the user; a port that silently
    # rewrote a declared key would hide the change from the receipt.
    if selection.sfcmodel >= 1 and selection.bbc != SEMI_SLIP:
        refuse(
            code="CM1-SFC-SURFACE-LAYER-WITHOUT-SEMI-SLIP",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.bbc",
            value=selection.bbc,
            reason=(
                "CM1 forces bbc=3 whenever a surface layer is on; with any "
                "other bbc the cd, znt and ust this surface layer computes "
                "would reach no momentum tendency at all, so the run would "
                "pay for a surface layer and integrate a different boundary"
            ),
            remedy="set bbc=3 alongside sfcmodel=1, or set sfcmodel=0",
        )
    if selection.sfcmodel not in (0, 1):
        refuse(
            code="CM1-SFC-SFCMODEL-UNSUPPORTED",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.sfcmodel",
            value=selection.sfcmodel,
            reason=(
                "sfcmodel=2,3,4,6,7 are the WRF surface-layer schemes and "
                "sfcmodel=5 is CM1's Monin-Obukhov scheme; none is ported, so "
                "selecting one would silently run the neutral constant-Cd "
                "drag while the receipt named a stability-dependent surface"
            ),
            remedy="use sfcmodel=1, CM1's simple neutral scheme, or sfcmodel=0",
        )
    # src/param.F:1349-1363
    if selection.sfcmodel == 1 and selection.cecd not in (1, 2, 3):
        refuse(
            code="CM1-SFC-CECD-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cecd",
            value=selection.cecd,
            reason=(
                "getcecd has three drag branches (sfcphys.F:120-195); any "
                "other value falls through all of them, so cd, znt and ust "
                "keep whatever they held from the previous step and the "
                "surface stress is built from a drag coefficient nothing in "
                "this run computed"
            ),
            remedy=(
                "set cecd to 1 (constant cnstcd), 2 (Deacon) or 3 "
                "(Fairall/Donelan)"
            ),
        )
    # src/param.F:1330-1346.  CM1 stops above 1; the simultaneous check at
    # src/param.F:1311-1327 is commented out in r21.1, but the wrong result
    # it names is real, so this port refuses it rather than silently
    # preferring one of the two declared surfaces.
    for name in ("set_znt", "set_ust"):
        value = getattr(selection, name)
        if value not in (0, 1):
            refuse(
                code="CM1-SFC-SET-SWITCH-RANGE",
                gate=SURFACE_DRAG_GATE,
                option=f"{option_prefix}.{name}",
                value=value,
                reason=(
                    "sfcphys.F:81-118 tests this switch against 1 exactly; "
                    "any other value silently falls through to the cecd "
                    "branch, so the declared surface is never imposed and the "
                    "drag coefficient comes from somewhere else entirely"
                ),
                remedy=f"set {name} to 0 or 1",
            )
    if selection.set_znt == 1 and selection.set_ust == 1:
        refuse(
            code="CM1-SFC-SET-ZNT-AND-UST",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.set_ust",
            value=selection.set_ust,
            reason=(
                "sfcphys.F:81 tests set_znt first, so with both imposed the "
                "roughness branch runs and the declared friction velocity "
                "never reaches the surface: ust comes from the log law "
                "instead, while the receipt names cnst_ust"
            ),
            remedy="impose the roughness length or the friction velocity, not both",
        )
    # README.namelist:920
    if (selection.set_znt == 1 or selection.set_ust == 1) and selection.sfcmodel != 1:
        refuse(
            code="CM1-SFC-SET-SWITCH-WITHOUT-SURFACE-LAYER",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.sfcmodel",
            value=selection.sfcmodel,
            reason=(
                "getcecd is the only reader of set_znt and set_ust "
                "(sfcphys.F:81-118) and turb.F:614-616 calls it only for "
                "sfcmodel=1; with any other surface model the declared "
                "roughness or friction velocity is never applied and the "
                "surface stress is built from an unset drag coefficient"
            ),
            remedy="set sfcmodel=1 alongside set_znt=1 or set_ust=1",
        )
    for name, value in (
        (f"{option_prefix}.cnst_znt", selection.cnst_znt),
        (f"{option_prefix}.cnst_ust", selection.cnst_ust),
        (f"{option_prefix}.cnstce", selection.cnstce),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite real number")
    if selection.set_znt == 1 and not (
        math.isfinite(selection.cnst_znt) and selection.cnst_znt > 0.0
    ):
        refuse(
            code="CM1-SFC-CNST-ZNT-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnst_znt",
            value=selection.cnst_znt,
            reason=(
                "sfcphys.F:85-88 divides by the roughness length and takes "
                "the logarithm of ten over it; at zero the reciprocal is "
                "infinite and the drag coefficient is exactly zero, and at a "
                "negative length the logarithm is not-a-number, so every "
                "surface stress word built from it is not-a-number"
            ),
            remedy=(
                "declare a positive roughness length in metres; measured "
                "land values are 0.01 to 1"
            ),
        )
    if selection.set_ust == 1 and not (
        math.isfinite(selection.cnst_ust) and selection.cnst_ust > 0.0
    ):
        refuse(
            code="CM1-SFC-CNST-UST-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnst_ust",
            value=selection.cnst_ust,
            reason=(
                "sfcphys.F:106 clamps the friction velocity with "
                "max(1.0e-10,ust) before dividing by it, so a non-positive "
                "declaration never reaches the surface: the exponent becomes "
                "the lowest-level wind times 4e9, its exponential overflows, "
                "and the roughness length za/inf is exactly zero, making the "
                "drag coefficient and the 10 m wind ratio not-a-number"
            ),
            remedy=(
                "declare a positive friction velocity in m/s; measured "
                "values are 0.1 to 2"
            ),
        )
    # src/param.F:2090-2106
    if selection.isfcflx not in (0, 1):
        refuse(
            code="CM1-SFC-ISFCFLX-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.isfcflx",
            value=selection.isfcflx,
            reason=(
                "CM1's surface-flux switch is 0 or 1 only "
                "(README.namelist:914); any other value would fall through "
                "every branch in sfcphys.F and leave the surface enthalpy "
                "flux silently zero while the receipt named a third setting "
                "that never existed"
            ),
            remedy="set isfcflx to 0 or 1",
        )
    if selection.isfcflx == 1 and selection.sfcmodel < 1:
        refuse(
            code="CM1-SFC-ENTHALPY-FLUX-WITHOUT-SURFACE-LAYER",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.sfcmodel",
            value=selection.sfcmodel,
            reason=(
                "turb.F:645-649 calls sfcflux only inside the sfcmodel=1 "
                "block, and the enthalpy exchange coefficients it multiplies "
                "are written by getcecd (sfcphys.F:197-236), which the same "
                "block calls; with no surface layer ch and cq stay zero, so "
                "the heat and moisture fluxes are exactly zero while the "
                "receipt claims a surface energy budget"
            ),
            remedy="set sfcmodel=1 alongside isfcflx=1, or set isfcflx=0",
        )
    if selection.isfcflx == 1 and not (
        math.isfinite(selection.cnstce) and selection.cnstce >= 0.0
    ):
        refuse(
            code="CM1-SFC-CNSTCE-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnstce",
            value=selection.cnstce,
            reason=(
                "sfcphys.F:203-205 assigns this value straight into ch and "
                "cq over water, and sfcphys.F:341,349 multiplies the heat and "
                "moisture fluxes by it; a negative coefficient makes the "
                "surface cool the air it is warmer than, and a non-finite one "
                "makes every scalar tendency at the lowest level not-a-number"
            ),
            remedy=(
                "declare a non-negative enthalpy exchange coefficient, for "
                "example 0.001"
            ),
        )
    if selection.sfcmodel == 1 and not (
        math.isfinite(selection.cnstcd) and selection.cnstcd > 0.0
    ):
        refuse(
            code="CM1-SFC-CNSTCD-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnstcd",
            value=selection.cnstcd,
            reason=(
                "sfcphys.F:127 clamps the drag coefficient with "
                "max(1.0e-10,cnstcd) before anything reads it, so a "
                "non-positive declaration never reaches the surface at all: "
                "the run integrates 1e-10 while the receipt names the "
                "declared value. That clamped Cd is itself below the "
                "roughness floor -- karman/sqrt(1e-10) is 40000, "
                "sfcphys.F:128's exp of it overflows, the roughness length is "
                "exactly zero and every word sfcphys.F:129-130 builds from "
                "1/znt is not-a-number"
            ),
            remedy="declare a positive drag coefficient, for example 0.001",
        )
    if (
        selection.sfcmodel == 1
        and selection.cnstcd < CNSTCD_BINARY32_ROUGHNESS_FLOOR
    ):
        refuse(
            code="CM1-SFC-CNSTCD-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnstcd",
            value=selection.cnstcd,
            reason=(
                "at this drag coefficient karman/sqrt(Cd) is past binary32's "
                "exponential range, so sfcphys.F:128's exp(karman/sqrt(Cd)) "
                "overflows and the roughness length 10/inf is exactly zero; "
                "1/znt at sfcphys.F:129 is then infinite, the 10 m wind ratio "
                "is not-a-number, and the bbc=3 surface dissipation rate "
                "log((dz1+znt)/znt) at solve1.F:956 is infinite, so the "
                "lowest model level's heating is infinite"
            ),
            remedy=(
                "declare a drag coefficient of at least "
                f"{CNSTCD_BINARY32_ROUGHNESS_FLOOR:.6g}; measured surface "
                "values are 1e-3 to 1e-2"
            ),
        )
    if (
        selection.sfcmodel == 1
        and selection.cnstcd >= CNSTCD_BINARY32_ROUGHNESS_CEILING
    ):
        refuse(
            code="CM1-SFC-CNSTCD-RANGE",
            gate=SURFACE_DRAG_GATE,
            option=f"{option_prefix}.cnstcd",
            value=selection.cnstcd,
            reason=(
                "at this drag coefficient karman/sqrt(Cd) is below binary32's "
                "2**-24, so sfcphys.F:128's exp(karman/sqrt(Cd)) rounds to "
                "exactly 1.0, the subtraction of 1.0 gives zero and the "
                "roughness length 10/0 is infinite; every word "
                "sfcphys.F:129-130 builds from 1/znt, and the friction "
                "velocity sfcphys.F:134 takes from the same logarithm that "
                "the semi-slip surface stress squares, is then not-a-number. "
                "CM1 imposes no upper bound and neither does this port below "
                "that value"
            ),
            remedy=(
                "declare a drag coefficient below "
                f"{CNSTCD_BINARY32_ROUGHNESS_CEILING:.6g}; measured surface "
                "values are 1e-3 to 1e-2"
            ),
        )
    return selection


def surface_drag_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    """Return every carrier this operator reads or writes."""

    for name, value, minimum in (("ni", ni, 3), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max(ni, nj, nk) > (1 << 31) - 8:
        raise ValueError("surface dimensions plus faces exceed signed CUDA indexing")
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    surface = (nj, ni)
    return {
        "u3d": (nk, nj, ni + 1),
        "v3d": (nk, nj + 1, ni),
        "zh": mass,
        "terrain_msl": surface,
        "zf": zface,
        "rf": zface,
        "mf": zface,
        "kmh": zface,
        "kmv": zface,
        "khh": zface,
        "khv": zface,
        "tkea": zface,
        "t13": (nk + 1, nj, ni + 1),
        "t23": (nk + 1, nj + 1, ni),
        "u1": surface,
        "v1": surface,
        "s1": surface,
        "za": surface,
        "cd": surface,
        "znt": surface,
        "ust": surface,
        "u10": surface,
        "v10": surface,
        "s10": surface,
        "stau": surface,
        "epsd_sfc": surface,
        # src/init_surface.F:40-65,239-243 surface fields the surface layer
        # reads rather than derives, and the sfcmodel=1 enthalpy carriers of
        # src/sfcphys.F:197-236 and src/sfcphys.F:341-361.
        "xland": surface,
        "mavail": surface,
        "tsk": surface,
        "ch": surface,
        "cq": surface,
        "psfc": surface,
        "qsfc": surface,
        "thflux": surface,
        "qvflux": surface,
        # src/turb.F:292 reads the lowest three mass pressures; sfcflux reads
        # the lowest base and perturbation potential temperature and the
        # lowest water-vapour mixing ratio.
        "prs": mass,
        "th0": mass,
        "tha": mass,
        "qva": mass,
    }


# The per-tile scratch CM1's lower boundary owns when it is not free slip.
SURFACE_STATE_NAMES = (
    "u1",
    "v1",
    "s1",
    "za",
    "cd",
    "znt",
    "ust",
    "u10",
    "v10",
    "s10",
    "stau",
    "epsd_sfc",
)

# src/init_surface.F:40-65 fills these once, before the first step, and
# src/sfcphys.F:155-158 then carries ``znt`` forward from step to step as the
# first guess of the wind-dependent water-roughness iteration.  They are
# therefore whole-domain model state, not per-tile scratch: a per-tile
# rebuild would restart that iteration from the roughness of whichever tile
# ran last.
SURFACE_PERSISTENT_NAMES = (
    "znt",
    "xland",
    "mavail",
    "tsk",
)

# The extra per-tile scratch the isfcflx=1 surface enthalpy flux owns.
SURFACE_FLUX_STATE_NAMES = (
    "ch",
    "cq",
    "psfc",
    "qsfc",
    "thflux",
    "qvflux",
)


# --------------------------------------------------------------------------
# Default-REAL host transcription.
#
# This is the reference the CUDA image is written from and the oracle the
# tests compare against.  Every arithmetic statement is one binary32
# operation, in CM1's own statement order, so the host and device results
# differ only where a libm and a CUDA transcendental differ in the last unit
# in the last place.
# --------------------------------------------------------------------------


def _f32_ops(np: Any) -> tuple[Any, Any, Any, Any]:
    def add(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) + np.float32(right))

    def sub(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) - np.float32(right))

    def mul(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) * np.float32(right))

    def div(left: Any, right: Any) -> Any:
        return np.float32(np.float32(left) / np.float32(right))

    return add, sub, mul, div


def _clamp(index: int, extent: int) -> int:
    """CM1's ``bc2d`` zero-gradient halo for the four open lateral edges.

    ``src/bc.F:12-172`` fills the halo of every surface field with the edge
    value when the lateral boundary condition is 2 (open radiative), and
    ``src/turb.F:1138-1178`` calls it for ust, u1, v1, s1 and znt whenever
    bbc=3.  Index clamping is that halo, without storing it.
    """

    if index < 0:
        return 0
    if index >= extent:
        return extent - 1
    return index


def cpu_surface_drag_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    selection: SurfaceDragSelection,
) -> dict[str, object]:
    """Run CM1's lower-boundary source order once, in place, on the host."""

    np = importlib.import_module("numpy")
    validate_surface_drag_selection(selection)
    shapes = surface_drag_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing surface carriers: " + ", ".join(sorted(missing)))
    for name, shape in shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if tuple(value.shape) != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order float32 with shape {shape}")
    if isinstance(dz, bool) or not isinstance(dz, (int, float)):
        raise TypeError("dz must be numeric")
    dz = float(dz)
    if not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("dz must be finite and > 0")
    if not selection.active:
        return {"applied": False, "bbc": selection.bbc}

    add, sub, mul, div = _f32_ops(np)
    f32 = np.float32
    karman = f32(CM1_KARMAN_F32)
    umove = f32(selection.u_move_m_s)
    vmove = f32(selection.v_move_m_s)
    rdz = div(f32(1.0), f32(dz))

    a = arrays
    logf, expf, sqrtf = _host_libm_f32()

    # src/turb.F:296-307 ground-relative wind, then src/turb.F:539-543 and
    # 547-555 for the lowest level wind and its height above the surface.
    for j in range(nj):
        for i in range(ni):
            left = add(a["u3d"][0, j, i], umove)
            right = add(a["u3d"][0, j, i + 1], umove)
            south = add(a["v3d"][0, j, i], vmove)
            north = add(a["v3d"][0, j + 1, i], vmove)
            u1 = mul(f32(0.5), add(left, right))
            v1 = mul(f32(0.5), add(south, north))
            a["u1"][j, i] = u1
            a["v1"][j, i] = v1
            a["s1"][j, i] = sqrtf(np, add(mul(u1, u1), mul(v1, v1)))
            a["za"][j, i] = sub(a["zh"][0, j, i], a["terrain_msl"][j, i])

    nonconvergent = False
    if selection.uses_surface_layer:
        # src/sfcphys.F:81-195 getcecd, in the source's own branch order.
        branch = selection.getcecd_branch
        dcd1 = f32(CM1_DCD1_F32)
        dcd2 = f32(CM1_DCD2_F32)
        dwsp1 = f32(CM1_DWSP1_F32)
        dwsp2 = f32(CM1_DWSP2_F32)
        dfac = div(sub(dcd2, dcd1), sub(dwsp2, dwsp1))
        for j in range(nj):
            for i in range(ni):
                za = a["za"][j, i]
                s1 = a["s1"][j, i]
                if branch == GETCECD_SET_ZNT:
                    # src/sfcphys.F:83-99: the roughness length is imposed.
                    rznt = div(f32(1.0), a["znt"][j, i])
                    reference = logf(np, mul(f32(10.0), rznt))
                    root = div(karman, reference)
                    a["cd"][j, i] = mul(root, root)
                    level = logf(np, mul(za, rznt))
                    var = div(reference, level)
                    a["ust"][j, i] = max(
                        div(mul(s1, karman), level), f32(1.0e-6)
                    )
                elif branch == GETCECD_SET_UST:
                    # src/sfcphys.F:102-118: the friction velocity is imposed.
                    a["ust"][j, i] = f32(selection.cnst_ust)
                    exponent = div(
                        mul(max(f32(1.0e-10), s1), karman),
                        max(f32(1.0e-10), a["ust"][j, i]),
                    )
                    znt = div(za, sub(expf(np, exponent), f32(1.0)))
                    a["znt"][j, i] = znt
                    rznt = div(f32(1.0), znt)
                    reference = logf(np, mul(add(f32(10.0), znt), rznt))
                    level = logf(np, mul(add(za, znt), rznt))
                    root = div(karman, reference)
                    a["cd"][j, i] = mul(root, root)
                    var = div(reference, level)
                elif branch == GETCECD_CONSTANT_CD:
                    # src/sfcphys.F:120-136: the drag coefficient is imposed.
                    cd = max(f32(1.0e-10), f32(selection.cnstcd))
                    a["cd"][j, i] = cd
                    exponent = div(karman, sqrtf(np, cd))
                    znt = div(f32(10.0), sub(expf(np, exponent), f32(1.0)))
                    a["znt"][j, i] = znt
                    rznt = div(f32(1.0), znt)
                    reference = logf(np, mul(add(f32(10.0), znt), rznt))
                    level = logf(np, mul(add(za, znt), rznt))
                    var = div(reference, level)
                    a["ust"][j, i] = max(
                        div(mul(s1, karman), level), f32(1.0e-6)
                    )
                else:
                    # src/sfcphys.F:138-195: Cd, z0 and u* from the 10 m wind
                    # over water; the prescribed roughness length over land.
                    znt = a["znt"][j, i]
                    rznt = div(f32(1.0), znt)
                    reference = logf(np, mul(add(f32(10.0), znt), rznt))
                    level = logf(np, mul(add(za, znt), rznt))
                    var = div(reference, level)
                    speed = mul(s1, var)
                    if a["xland"][j, i] > f32(1.5):
                        wlast = f32(-1000.0)
                        iterations = 0
                        while abs(sub(speed, wlast)) > f32(0.001):
                            iterations += 1
                            wlast = speed
                            if selection.cecd == 2:
                                cd = add(
                                    f32(1.1e-3), mul(f32(4.0e-5), speed)
                                )
                            else:
                                cd = add(dcd1, mul(sub(speed, dwsp1), dfac))
                                cd = min(cd, dcd2)
                                cd = max(cd, dcd1)
                            a["cd"][j, i] = cd
                            exponent = div(karman, sqrtf(np, cd))
                            znt = max(
                                f32(1.0e-20),
                                div(
                                    f32(10.0),
                                    sub(expf(np, exponent), f32(1.0)),
                                ),
                            )
                            a["znt"][j, i] = znt
                            rznt = div(f32(1.0), znt)
                            reference = logf(
                                np, mul(add(f32(10.0), znt), rznt)
                            )
                            level = logf(np, mul(add(za, znt), rznt))
                            var = div(reference, level)
                            speed = mul(s1, var)
                            if iterations > 20:
                                nonconvergent = True
                                break
                    else:
                        root = div(karman, reference)
                        a["cd"][j, i] = mul(root, root)
                    a["ust"][j, i] = max(
                        div(mul(s1, karman), level), f32(1.0e-6)
                    )
                a["u10"][j, i] = mul(a["u1"][j, i], var)
                a["v10"][j, i] = mul(a["v1"][j, i], var)
                a["s10"][j, i] = mul(a["s1"][j, i], var)
        if selection.isfcflx == 1:
            # src/sfcphys.F:197-236 getcecd, the enthalpy exchange
            # coefficients: Ce=Cd over land, and the declared or
            # wind-dependent value over water.
            for j in range(nj):
                for i in range(ni):
                    if a["xland"][j, i] > f32(1.5):
                        if selection.cecd == 1:
                            coefficient = f32(selection.cnstce)
                        elif selection.cecd == 2:
                            coefficient = add(
                                f32(1.1e-3), mul(f32(4.0e-5), a["s10"][j, i])
                            )
                        else:
                            coefficient = f32(1.20e-3)
                    else:
                        coefficient = a["cd"][j, i]
                    a["ch"][j, i] = coefficient
                    a["cq"][j, i] = coefficient

    # src/turb.F:3244-3306 tkebc lower face for bbc=2 and bbc=3.
    for j in range(nj):
        for i in range(ni):
            a["tkea"][0, j, i] = f32(0.0)
            a["kmh"][0, j, i] = mul(mul(karman, a["znt"][j, i]), a["ust"][j, i])
            a["kmv"][0, j, i] = a["kmh"][0, j, i]
            ratio = div(a["khv"][1, j, i], add(f32(1.0e-10), a["kmv"][1, j, i]))
            a["khv"][0, j, i] = mul(a["kmv"][0, j, i], ratio)
            ratio = div(a["khh"][1, j, i], add(f32(1.0e-10), a["kmh"][1, j, i]))
            a["khh"][0, j, i] = mul(a["kmh"][0, j, i], ratio)

    if selection.bbc == NO_SLIP:
        # src/turb.F:4555-4564
        for j in range(nj):
            for i in range(ni + 1):
                left = _clamp(i - 1, ni)
                right = _clamp(i, ni)
                term = mul(f32(2.0), add(a["u3d"][0, j, i], umove))
                term = mul(term, rdz)
                term = mul(term, f32(0.5))
                term = mul(term, add(a["mf"][0, j, left], a["mf"][0, j, right]))
                term = mul(term, f32(0.5))
                term = mul(term, add(a["rf"][0, j, left], a["rf"][0, j, right]))
                term = mul(term, f32(0.5))
                term = mul(term, add(a["kmv"][1, j, left], a["kmv"][1, j, right]))
                a["t13"][0, j, i] = term
        for j in range(nj + 1):
            south = _clamp(j - 1, nj)
            north = _clamp(j, nj)
            for i in range(ni):
                term = mul(f32(2.0), add(a["v3d"][0, j, i], vmove))
                term = mul(term, rdz)
                term = mul(term, f32(0.5))
                term = mul(term, add(a["mf"][0, south, i], a["mf"][0, north, i]))
                term = mul(term, f32(0.5))
                term = mul(term, add(a["rf"][0, south, i], a["rf"][0, north, i]))
                term = mul(term, f32(0.5))
                term = mul(term, add(a["kmv"][1, south, i], a["kmv"][1, north, i]))
                a["t23"][0, j, i] = term
    else:
        # src/turb.F:4596-4605
        for j in range(nj):
            for i in range(ni + 1):
                left = _clamp(i - 1, ni)
                right = _clamp(i, ni)
                west = mul(
                    mul(a["ust"][j, left], a["ust"][j, left]),
                    div(a["u1"][j, left], max(a["s1"][j, left], f32(0.01))),
                )
                east = mul(
                    mul(a["ust"][j, right], a["ust"][j, right]),
                    div(a["u1"][j, right], max(a["s1"][j, right], f32(0.01))),
                )
                term = mul(f32(0.25), add(west, east))
                a["t13"][0, j, i] = mul(
                    term, add(a["rf"][0, j, left], a["rf"][0, j, right])
                )
        for j in range(nj + 1):
            south = _clamp(j - 1, nj)
            north = _clamp(j, nj)
            for i in range(ni):
                lower = mul(
                    mul(a["ust"][south, i], a["ust"][south, i]),
                    div(a["v1"][south, i], max(a["s1"][south, i], f32(0.01))),
                )
                upper = mul(
                    mul(a["ust"][north, i], a["ust"][north, i]),
                    div(a["v1"][north, i], max(a["s1"][north, i], f32(0.01))),
                )
                term = mul(f32(0.25), add(lower, upper))
                a["t23"][0, j, i] = mul(
                    term, add(a["rf"][0, south, i], a["rf"][0, north, i])
                )

    # src/turb.F:4692-4700
    for j in range(nj):
        for i in range(ni):
            x = mul(f32(0.5), add(a["t13"][0, j, i], a["t13"][0, j, i + 1]))
            y = mul(f32(0.5), add(a["t23"][0, j, i], a["t23"][0, j + 1, i]))
            a["stau"][j, i] = div(
                sqrtf(np, add(mul(x, x), mul(y, y))), a["rf"][0, j, i]
            )

    if selection.bbc == SEMI_SLIP:
        # src/solve1.F:956-957 and 982-983.
        for j in range(nj):
            for i in range(ni):
                dz1 = sub(a["zf"][1, j, i], a["zf"][0, j, i])
                znt = a["znt"][j, i]
                ust = a["ust"][j, i]
                cubed = mul(mul(ust, ust), ust)
                numerator = mul(cubed, logf(np, div(add(dz1, znt), znt)))
                a["epsd_sfc"][j, i] = div(numerator, mul(karman, dz1))
    else:
        a["epsd_sfc"][:, :] = np.float32(0.0)

    return {
        "applied": True,
        "bbc": selection.bbc,
        "getcecd_branch": (
            selection.getcecd_branch if selection.uses_surface_layer else None
        ),
        "water_roughness_nonconvergent": nonconvergent,
        "surface_words": nj * ni,
        "t13_words": nj * (ni + 1),
        "t23_words": (nj + 1) * ni,
    }


def cpu_surface_enthalpy_flux_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    moisture: bool,
    surface_pressure_weights: tuple[float, float, float],
) -> dict[str, object]:
    """``src/turb.F:292`` then ``src/sfcphys.F:341-361``, on the host.

    This is the reference the two enthalpy kernels are written from.  The
    only transcendental CM1 uses here that the host and the device evaluate
    differently is ``(psfc*rp00)**rovcp`` and ``rslf``'s exponential, so the
    device words are a few units in the last place from these.
    """

    np = importlib.import_module("numpy")
    add, sub, mul, div = _f32_ops(np)
    f32 = np.float32
    logf, expf, sqrtf = _host_libm_f32()
    powf = _host_libm_pow_f32()
    cgs1, cgs2, cgs3 = (f32(value) for value in surface_pressure_weights)
    rp00 = div(f32(1.0), f32(CM1_P00_F32))
    rovcp = div(f32(CM1_RD_F32), f32(CM1_CP_F32))
    eps = div(f32(CM1_RD_F32), f32(CM1_RV_F32))
    a = arrays
    for j in range(nj):
        for i in range(ni):
            # src/turb.F:292
            pressure = add(
                add(
                    mul(cgs1, a["prs"][0, j, i]),
                    mul(cgs2, a["prs"][1, j, i]),
                ),
                mul(cgs3, a["prs"][2, j, i]),
            )
            a["psfc"][j, i] = pressure
            skin = a["tsk"][j, i]
            # src/sfcphys.F:340-341
            pisfc = powf(np, mul(pressure, rp00), rovcp)
            a["thflux"][j, i] = mul(
                mul(a["ch"][j, i], a["s10"][j, i]),
                sub(sub(div(skin, pisfc), a["th0"][0, j, i]), a["tha"][0, j, i]),
            )
            if moisture:
                # src/cm1libs.F:37-41 rslf, then src/sfcphys.F:346-350
                exponent = div(
                    mul(f32(17.67), sub(skin, f32(273.15))),
                    sub(skin, f32(29.65)),
                )
                esl = mul(f32(611.2), expf(np, exponent))
                esl = min(esl, mul(pressure, f32(0.5)))
                saturation = div(mul(eps, esl), sub(pressure, esl))
                a["qsfc"][j, i] = saturation
                a["qvflux"][j, i] = mul(
                    mul(
                        mul(a["cq"][j, i], a["s10"][j, i]),
                        sub(saturation, a["qva"][0, j, i]),
                    ),
                    a["mavail"][j, i],
                )
    return {
        "applied": True,
        "moisture": bool(moisture),
        "surface_words": nj * ni,
        "surface_pressure_weights": [float(value) for value in (cgs1, cgs2, cgs3)],
    }


def _host_libm_pow_f32() -> Any:
    """``powf`` from the platform libm, for the Exner ratio at the ground."""

    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("m") or "libm.so.6"
    libm = ctypes.CDLL(name)
    libm.powf.restype = ctypes.c_float
    libm.powf.argtypes = (ctypes.c_float, ctypes.c_float)

    def powf(np: Any, base: Any, exponent: Any) -> Any:
        return np.float32(libm.powf(float(base), float(exponent)))

    return powf


def _host_libm_f32() -> tuple[Any, Any, Any]:
    """Return binary32 log, exp and sqrt bound to the platform libm.

    ``numpy`` evaluates float32 transcendentals through its own SIMD kernels,
    which are not the same function CM1's compiled Fortran calls.  The host
    reference uses libm directly so it can be compared against the already
    native-verified ``surface_exchange_coefficients`` operator word for word.
    """

    import ctypes
    import ctypes.util
    import os

    if os.name == "nt":
        library = ctypes.CDLL("ucrtbase")
    else:
        name = ctypes.util.find_library("m")
        library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
    bound = []
    for name in ("logf", "expf", "sqrtf"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
        bound.append(function)

    def wrap(function: Any) -> Any:
        def call(np: Any, value: Any) -> Any:
            return np.float32(function(ctypes.c_float(float(np.float32(value)))))

        return call

    return wrap(bound[0]), wrap(bound[1]), wrap(bound[2])


# --------------------------------------------------------------------------
# CUDA transcription.  One statement per source statement; round-to-nearest
# intrinsics and --fmad=false keep the statement boundaries the host
# reference was written from.
# --------------------------------------------------------------------------

CUDA_SURFACE_DRAG_SOURCE = r'''// gm1.cuda-surface-drag-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float x,float y){return __fadd_rn(x,y);}
__device__ __forceinline__ float S(float x,float y){return __fsub_rn(x,y);}
__device__ __forceinline__ float M(float x,float y){return __fmul_rn(x,y);}
__device__ __forceinline__ float D(float x,float y){return __fdiv_rn(x,y);}
__device__ __forceinline__ int CI(int x,int n){return x<0?0:(x>=n?n-1:x);}
__device__ __forceinline__ unsigned long long IX(int k,int j,int i,int ny,int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float C2(const float*a,int j,int i,int nj,int ni){return a[(unsigned long long)CI(j,nj)*ni+CI(i,ni)];}
__device__ __forceinline__ float C3(const float*a,int k,int j,int i,int nj,int ni){return a[IX(k,CI(j,nj),CI(i,ni),nj,ni)];}
#define KARMAN 0.40f

// src/turb.F:296-307 ground-relative wind; src/turb.F:539-543 lowest model
// level wind at the scalar point; src/turb.F:547-555 its height above ground.
extern "C" __global__ void cm1_sfc_ground_relative_wind_v1(const float*u3d,const float*v3d,const float*zh,const float*zs,float*u1,float*v1,float*s1,float*za,unsigned ni,unsigned nj,float umove,float vmove){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){int i=(int)(q%ni),j=(int)(q/ni);
    float ul=A(u3d[IX(0,j,i,nj,ni+1)],umove),ur=A(u3d[IX(0,j,i+1,nj,ni+1)],umove);
    float vs=A(v3d[IX(0,j,i,nj+1,ni)],vmove),vn=A(v3d[IX(0,j+1,i,nj+1,ni)],vmove);
    float uu=M(0.5f,A(ul,ur)),vv=M(0.5f,A(vs,vn));
    u1[q]=uu;v1[q]=vv;s1[q]=sqrtf(A(M(uu,uu),M(vv,vv)));
    za[q]=S(zh[IX(0,j,i,nj,ni)],zs[q]);}}

// src/sfcphys.F:120-136 getcecd, cecd=1 constant drag coefficient.
extern "C" __global__ void cm1_sfc_getcecd_cnstcd_v1(const float*u1,const float*v1,const float*s1,const float*za,float*znt,float*ust,float*cd,float*u10,float*v10,float*s10,unsigned ni,unsigned nj,float cnstcd){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){
    float c=fmaxf(1.0e-10f,cnstcd);cd[q]=c;
    float e=D(KARMAN,sqrtf(c));
    float z0=D(10.0f,S(expf(e),1.0f));znt[q]=z0;
    float rz=D(1.0f,z0);
    float ref=logf(M(A(10.0f,z0),rz)),lev=logf(M(A(za[q],z0),rz));
    float var=D(ref,lev);
    u10[q]=M(u1[q],var);v10[q]=M(v1[q],var);s10[q]=M(s1[q],var);
    ust[q]=fmaxf(D(M(s1[q],KARMAN),lev),1.0e-6f);}}

// src/turb.F:3244-3306 tkebc, the bbc=2/3 lower face.
extern "C" __global__ void cm1_sfc_tkebc_lower_v1(float*tkea,float*kmh,float*kmv,float*khh,float*khv,const float*znt,const float*ust,unsigned ni,unsigned nj){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){
    tkea[q]=0.0f;
    kmh[q]=M(M(KARMAN,znt[q]),ust[q]);
    kmv[q]=kmh[q];
    unsigned long long up=n+q;
    khv[q]=M(kmv[q],D(khv[up],A(1.0e-10f,kmv[up])));
    khh[q]=M(kmh[q],D(khh[up],A(1.0e-10f,kmh[up])));}}

// src/turb.F:4555-4564 sfcstress, bbc=2 (no slip).
extern "C" __global__ void cm1_sfc_stress_no_slip_v1(const float*u3d,const float*v3d,const float*mf,const float*rf,const float*kmv,float*t13,float*t23,unsigned ni,unsigned nj,float rdz,float umove,float vmove){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,nu=(unsigned long long)(ni+1u)*nj,nv=(unsigned long long)ni*(nj+1u),st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<nu+nv;q+=st){
    if(q<nu){int i=(int)(q%(ni+1u)),j=(int)(q/(ni+1u)),l=CI(i-1,(int)ni),r=CI(i,(int)ni);
      float t=M(2.0f,A(u3d[IX(0,j,i,nj,ni+1)],umove));
      t=M(t,rdz);t=M(t,0.5f);t=M(t,A(mf[IX(0,j,l,nj,ni)],mf[IX(0,j,r,nj,ni)]));
      t=M(t,0.5f);t=M(t,A(rf[IX(0,j,l,nj,ni)],rf[IX(0,j,r,nj,ni)]));
      t=M(t,0.5f);t=M(t,A(kmv[IX(1,j,l,nj,ni)],kmv[IX(1,j,r,nj,ni)]));
      t13[q]=t;}
    else{unsigned long long p=q-nu;int i=(int)(p%ni),j=(int)(p/ni),s=CI(j-1,(int)nj),r=CI(j,(int)nj);
      float t=M(2.0f,A(v3d[IX(0,j,i,nj+1,ni)],vmove));
      t=M(t,rdz);t=M(t,0.5f);t=M(t,A(mf[IX(0,s,i,nj,ni)],mf[IX(0,r,i,nj,ni)]));
      t=M(t,0.5f);t=M(t,A(rf[IX(0,s,i,nj,ni)],rf[IX(0,r,i,nj,ni)]));
      t=M(t,0.5f);t=M(t,A(kmv[IX(1,s,i,nj,ni)],kmv[IX(1,r,i,nj,ni)]));
      t23[p]=t;}}}

// src/turb.F:4596-4605 sfcstress, bbc=3 (semi slip).  This is where drag is set.
extern "C" __global__ void cm1_sfc_stress_semi_slip_v1(const float*u1,const float*v1,const float*s1,const float*ust,const float*rf,float*t13,float*t23,unsigned ni,unsigned nj){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,nu=(unsigned long long)(ni+1u)*nj,nv=(unsigned long long)ni*(nj+1u),st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<nu+nv;q+=st){
    if(q<nu){int i=(int)(q%(ni+1u)),j=(int)(q/(ni+1u)),l=CI(i-1,(int)ni),r=CI(i,(int)ni);
      unsigned long long pl=(unsigned long long)j*ni+l,pr=(unsigned long long)j*ni+r;
      float w=M(M(ust[pl],ust[pl]),D(u1[pl],fmaxf(s1[pl],0.01f)));
      float e=M(M(ust[pr],ust[pr]),D(u1[pr],fmaxf(s1[pr],0.01f)));
      t13[q]=M(M(0.25f,A(w,e)),A(rf[IX(0,j,l,nj,ni)],rf[IX(0,j,r,nj,ni)]));}
    else{unsigned long long p=q-nu;int i=(int)(p%ni),j=(int)(p/ni),s=CI(j-1,(int)nj),r=CI(j,(int)nj);
      unsigned long long ps=(unsigned long long)s*ni+i,pn=(unsigned long long)r*ni+i;
      float lo=M(M(ust[ps],ust[ps]),D(v1[ps],fmaxf(s1[ps],0.01f)));
      float hi=M(M(ust[pn],ust[pn]),D(v1[pn],fmaxf(s1[pn],0.01f)));
      t23[p]=M(M(0.25f,A(lo,hi)),A(rf[IX(0,s,i,nj,ni)],rf[IX(0,r,i,nj,ni)]));}}}

// src/turb.F:4692-4700 surface stress magnitude.
extern "C" __global__ void cm1_sfc_stau_v1(const float*t13,const float*t23,const float*rf,float*stau,unsigned ni,unsigned nj){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){int i=(int)(q%ni),j=(int)(q/ni);
    float x=M(0.5f,A(t13[IX(0,j,i,nj,ni+1)],t13[IX(0,j,i+1,nj,ni+1)]));
    float y=M(0.5f,A(t23[IX(0,j,i,nj+1,ni)],t23[IX(0,j+1,i,nj+1,ni)]));
    stau[q]=D(sqrtf(A(M(x,x),M(y,y))),rf[IX(0,j,i,nj,ni)]);}}

// src/solve1.F:956-957 and 982-983: the bbc=3 surface dissipation rate that
// replaces the interior dissten average in dissipative heating at k=1.
extern "C" __global__ void cm1_sfc_dissipation_v1(const float*zf,const float*znt,const float*ust,float*epsd,unsigned ni,unsigned nj){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){int i=(int)(q%ni),j=(int)(q/ni);
    float dz1=S(zf[IX(1,j,i,nj,ni)],zf[IX(0,j,i,nj,ni)]);
    float z0=znt[q],u=ust[q];
    float cubed=M(M(u,u),u);
    epsd[q]=D(M(cubed,logf(D(A(dz1,z0),z0))),M(KARMAN,dz1));}}

// src/sfcphys.F:83-99 getcecd, set_znt=1: the roughness length is imposed by
// init_surface and the drag coefficient follows from it.  Note the reference
// and level logarithms here take 10*rznt and za*rznt, with no znt added --
// that is the source, and it is not the cecd=1 form.
extern "C" __global__ void cm1_sfc_getcecd_set_znt_v1(const float*u1,const float*v1,const float*s1,const float*za,const float*znt,float*ust,float*cd,float*u10,float*v10,float*s10,unsigned ni,unsigned nj){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){
    float rz=D(1.0f,znt[q]);
    float ref=logf(M(10.0f,rz));
    float root=D(KARMAN,ref);cd[q]=M(root,root);
    float lev=logf(M(za[q],rz));
    float var=D(ref,lev);
    u10[q]=M(u1[q],var);v10[q]=M(v1[q],var);s10[q]=M(s1[q],var);
    ust[q]=fmaxf(D(M(s1[q],KARMAN),lev),1.0e-6f);}}

// src/sfcphys.F:102-118 getcecd, set_ust=1: the friction velocity is imposed
// and the roughness length is inverted out of the log law that would have
// produced it.  ust is not recomputed afterwards.
extern "C" __global__ void cm1_sfc_getcecd_set_ust_v1(const float*u1,const float*v1,const float*s1,const float*za,float*znt,float*ust,float*cd,float*u10,float*v10,float*s10,unsigned ni,unsigned nj,float cnst_ust){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){
    ust[q]=cnst_ust;
    float e=D(M(fmaxf(1.0e-10f,s1[q]),KARMAN),fmaxf(1.0e-10f,ust[q]));
    float z0=D(za[q],S(expf(e),1.0f));znt[q]=z0;
    float rz=D(1.0f,z0);
    float ref=logf(M(A(10.0f,z0),rz)),lev=logf(M(A(za[q],z0),rz));
    float root=D(KARMAN,ref);cd[q]=M(root,root);
    float var=D(ref,lev);
    u10[q]=M(u1[q],var);v10[q]=M(v1[q],var);s10[q]=M(s1[q],var);}}

// src/sfcphys.F:138-195 getcecd, the wind-dependent water drag (cecd=2
// Deacon, cecd=3 Fairall/Donelan) with the land branch that reads the
// roughness length init_surface left in znt.  src/sfcphys.F:189-191 stops
// CM1 after twenty iterations; this port raises the same stop through a flag
// the host reads after the launch, because a kernel cannot stop the model.
extern "C" __global__ void cm1_sfc_getcecd_water_v1(const float*u1,const float*v1,const float*s1,const float*za,const float*xland,float*znt,float*ust,float*cd,float*u10,float*v10,float*s10,unsigned*nonconvergent,unsigned ni,unsigned nj,unsigned cecd){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  const float dcd1=1.0e-3f,dcd2=2.4e-3f,dwsp1=5.0f,dwsp2=25.0f,dfac=D(S(dcd2,dcd1),S(dwsp2,dwsp1));
  for(;q<n;q+=st){
    float z0=znt[q],rz=D(1.0f,z0);
    float ref=logf(M(A(10.0f,z0),rz)),lev=logf(M(A(za[q],z0),rz));
    float var=D(ref,lev),sp=M(s1[q],var);
    if(xland[q]>1.5f){
      float wlast=-1000.0f;int it=0;
      while(fabsf(S(sp,wlast))>0.001f){
        it=it+1;wlast=sp;
        float c;
        if(cecd==2u){c=A(1.1e-3f,M(4.0e-5f,sp));}
        else{c=A(dcd1,M(S(sp,dwsp1),dfac));c=fminf(c,dcd2);c=fmaxf(c,dcd1);}
        cd[q]=c;
        z0=fmaxf(1.0e-20f,D(10.0f,S(expf(D(KARMAN,sqrtf(c))),1.0f)));znt[q]=z0;
        rz=D(1.0f,z0);
        ref=logf(M(A(10.0f,z0),rz));lev=logf(M(A(za[q],z0),rz));
        var=D(ref,lev);sp=M(s1[q],var);
        if(it>20){atomicOr(nonconvergent,1u);break;}}}
    else{
      float root=D(KARMAN,ref);cd[q]=M(root,root);}
    u10[q]=M(u1[q],var);v10[q]=M(v1[q],var);s10[q]=M(s1[q],var);
    ust[q]=fmaxf(D(M(s1[q],KARMAN),lev),1.0e-6f);}}

// src/sfcphys.F:197-236 getcecd, the isfcflx=1 enthalpy exchange
// coefficients.  Over land CM1 sets Ce to Cd; over water it is the declared
// constant, Deacon's line, or Drennan's constant.
extern "C" __global__ void cm1_sfc_getcecd_chcq_v1(const float*xland,const float*s10,const float*cd,float*ch,float*cq,unsigned ni,unsigned nj,unsigned cecd,float cnstce){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){
    float c;
    if(xland[q]>1.5f){
      if(cecd==1u)c=cnstce;
      else if(cecd==2u)c=A(1.1e-3f,M(4.0e-5f,s10[q]));
      else c=1.20e-3f;}
    else c=cd[q];
    ch[q]=c;cq[q]=c;}}

// src/turb.F:290-293 surface pressure, from the three lowest mass levels
// through the src/param.F:9038-9040 Fornberg weights.
extern "C" __global__ void cm1_sfc_psfc_v1(const float*prs,float*psfc,unsigned ni,unsigned nj,float cgs1,float cgs2,float cgs3){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  for(;q<n;q+=st){int i=(int)(q%ni),j=(int)(q/ni);
    psfc[q]=A(A(M(cgs1,prs[IX(0,j,i,nj,ni)]),M(cgs2,prs[IX(1,j,i,nj,ni)])),M(cgs3,prs[IX(2,j,i,nj,ni)]));}}

// src/sfcphys.F:341-361 sfcflux, the computed (set_flx=0) branch, with
// src/cm1libs.F:37-41 rslf for the saturation mixing ratio at the skin
// temperature.  eps, rovcp and rp00 are built here as the two-operand
// quotients src/constants.F:129-148 forms, so the binary32 words are CM1's.
extern "C" __global__ void cm1_sfc_flux_v1(const float*ch,const float*cq,const float*s10,const float*psfc,const float*tsk,const float*th0,const float*tha,const float*qva,const float*mavail,float*thflux,float*qvflux,float*qsfc,unsigned ni,unsigned nj,unsigned imoist){
  unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)ni*nj,st=(unsigned long long)blockDim.x*gridDim.x;
  const float rd=287.04f,rv=461.5f,cp=1005.7f,rp00=D(1.0f,1.0e5f),rovcp=D(rd,cp),eps=D(rd,rv);
  for(;q<n;q+=st){
    float p=psfc[q],t=tsk[q];
    float pisfc=powf(M(p,rp00),rovcp);
    thflux[q]=M(M(ch[q],s10[q]),S(S(D(t,pisfc),th0[q]),tha[q]));
    if(imoist==1u){
      float esl=M(611.2f,expf(D(M(17.67f,S(t,273.15f)),S(t,29.65f))));
      esl=fminf(esl,M(p,0.5f));
      float qvsat=D(M(eps,esl),S(p,esl));
      qsfc[q]=qvsat;
      qvflux[q]=M(M(M(cq[q],s10[q]),S(qvsat,qva[q])),mavail[q]);}}}
'''

CUDA_SURFACE_DRAG_SOURCE_SHA256 = hashlib.sha256(
    CUDA_SURFACE_DRAG_SOURCE.encode()
).hexdigest()


def _source_pin_is_valid() -> bool:
    return (
        bool(re.fullmatch(r"[0-9a-f]{64}", CUDA_SURFACE_DRAG_SOURCE_SHA256))
        and hashlib.sha256(CUDA_SURFACE_DRAG_SOURCE.encode()).hexdigest()
        == CUDA_SURFACE_DRAG_SOURCE_SHA256
    )


def build_cuda_surface_drag_source() -> str:
    if not _source_pin_is_valid():
        raise RuntimeError("surface drag CUDA source differs from its pin")
    return CUDA_SURFACE_DRAG_SOURCE


@dataclass(frozen=True, slots=True)
class CompiledSurfaceDragKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: Mapping[str, object]
    source_sha256: str = CUDA_SURFACE_DRAG_SOURCE_SHA256


def compile_cuda_surface_drag() -> CompiledSurfaceDragKernels:
    source = build_cuda_surface_drag_source()
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    device = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(device.id)
    major = int(props["major"])
    minor = int(props["minor"])
    # One architecture rule for the package: the clamped target every GM1 image
    # and every CuPy compile uses, not getDeviceProperties' unclamped pair.
    target = importlib.import_module("cm1core.nvrtc_module").nvrtc_target_arch(cupy)
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    nvrtc = tuple(compiler._get_nvrtc_version())
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        source, "cm1_surface_drag_v1_ptx.cu", method="ptx"
    ).compile(common + (f"--gpu-architecture=compute_{target}",))
    cubin, _ = compiler._NVRTCProgram(
        source, "cm1_surface_drag_v1_cubin.cu", method="cubin"
    ).compile(common + (f"--gpu-architecture=sm_{target}",))
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    # The image that runs is the cubin compiled above (``--ftz=false``), not a
    # RawModule, which CuPy compiles with ``-ftz=true`` appended.
    module = importlib.import_module("cupy.cuda.function").Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in SURFACE_DRAG_KERNEL_NAMES)
    name = (
        props["name"].decode()
        if isinstance(props["name"], bytes)
        else str(props["name"])
    )
    return CompiledSurfaceDragKernels(
        module,
        kernels,
        ptx,
        cubin,
        target,
        {
            "device_id": device.id,
            "device_name": name,
            "compute_capability": f"{major}.{minor}",
            "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        },
    )


@dataclass(frozen=True, slots=True)
class SurfaceDragLaunch:
    ready_event: Any
    receipt: Mapping[str, object]
    compiled: CompiledSurfaceDragKernels


def _grid(total: int) -> tuple[int]:
    return (max(1, (total + SURFACE_DRAG_THREADS - 1) // SURFACE_DRAG_THREADS),)


def _device_arrays(
    arrays: Mapping[str, Any], names: tuple[str, ...], shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, Any]:
    cupy = importlib.import_module("cupy")
    out: dict[str, Any] = {}
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing surface carrier: {name}")
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if (
            value.dtype != cupy.float32
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-order float32 with shape {shapes[name]}"
            )
        out[name] = value
    return out


def launch_cuda_surface_layer(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: SurfaceDragSelection,
    stream: Any | None = None,
    compiled: CompiledSurfaceDragKernels | None = None,
) -> SurfaceDragLaunch:
    """Refresh the surface layer and the TKE lower face once per large step.

    This is CM1's ``getsfc`` cadence: ``src/turb.F:522-616`` computes the
    lowest-level wind and the exchange coefficients, and ``src/turb.F:1378``
    applies ``tkebc`` later in the same call, before the step's ``gettau``.
    ``tkebc`` rescales the scalar lower face from the ``k=2`` eddy
    coefficients, so the supplied ``kmh``/``kmv``/``khh``/``khv`` must
    already hold the values ``tkekm`` (``src/turb.F:1351``) wrote.  A
    runtime that computes them between the two calls enqueues
    :func:`enqueue_surface_layer` and :func:`enqueue_surface_tke_face`
    separately instead.
    """

    validate_surface_drag_selection(selection)
    if not selection.active:
        raise ValueError(
            "bbc=1 keeps CM1's free-slip lower boundary; no surface layer runs"
        )
    cupy = importlib.import_module("cupy")
    compiled = compiled or compile_cuda_surface_drag()
    stream = stream or cupy.cuda.get_current_stream()
    shapes = surface_drag_shapes(ni, nj, nk)
    names = (
        "u3d", "v3d", "zh", "terrain_msl",
        *SURFACE_STATE_NAMES,
        "tkea", "kmh", "kmv", "khh", "khv", "zf", "rf", "mf", "t13", "t23",
    )
    values = _device_arrays(arrays, names, shapes)
    binding = SurfaceDragBinding(
        selection=selection, compiled=compiled, arrays=values, dz=1.0
    )
    with stream:
        launched = enqueue_surface_layer(binding, {}, ni=ni, nj=nj, nk=nk, stream=stream)
        launched += enqueue_surface_tke_face(
            binding, {}, ni=ni, nj=nj, nk=nk, stream=stream
        )
        event = cupy.cuda.Event()
        event.record(stream)
    return SurfaceDragLaunch(
        event,
        {
            **binding.record(),
            "phase": "post-step-getsfc-and-tkebc",
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "kernel": {"names": list(launched), "target_arch": compiled.target_arch},
        },
        compiled,
    )


def launch_cuda_surface_stress(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    selection: SurfaceDragSelection,
    stream: Any | None = None,
    compiled: CompiledSurfaceDragKernels | None = None,
) -> SurfaceDragLaunch:
    """Replace ``t13``/``t23`` at the lowest face, CM1's ``sfcstress``.

    ``gettau`` builds the interior stresses and then calls ``sfcstress`` at
    ``src/turb.F:4417``, so this must run after the ``t13``/``t23`` build and
    before the momentum stress divergence reads ``t13(i,j,1)`` at
    ``src/turbtend.F:1454``.
    """

    validate_surface_drag_selection(selection)
    if not selection.active:
        raise ValueError(
            "bbc=1 keeps the free-slip copy gettau already wrote at k=1"
        )
    cupy = importlib.import_module("cupy")
    compiled = compiled or compile_cuda_surface_drag()
    stream = stream or cupy.cuda.get_current_stream()
    shapes = surface_drag_shapes(ni, nj, nk)
    names = (
        "u3d", "v3d", "zh", "terrain_msl",
        *SURFACE_STATE_NAMES,
        "tkea", "kmh", "kmv", "khh", "khv", "zf", "rf", "mf", "t13", "t23",
    )
    values = _device_arrays(arrays, names, shapes)
    binding = SurfaceDragBinding(
        selection=selection, compiled=compiled, arrays=values, dz=dz
    )
    with stream:
        launched = enqueue_surface_stress(binding, {}, ni=ni, nj=nj, nk=nk, stream=stream)
        event = cupy.cuda.Event()
        event.record(stream)
    return SurfaceDragLaunch(
        event,
        {
            **binding.record(),
            "phase": "gettau-sfcstress",
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "kernel": {"names": list(launched), "target_arch": compiled.target_arch},
        },
        compiled,
    )


def launch_cuda_surface_dissipation(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: SurfaceDragSelection,
    stream: Any | None = None,
    compiled: CompiledSurfaceDragKernels | None = None,
) -> SurfaceDragLaunch:
    """Form the ``bbc=3`` surface dissipation rate, CM1's ``solve1`` epsd.

    ``src/solve1.F:955-957`` computes it inside the dissipative-heating
    block, so this must run after ``getcecd`` has written ``ust``/``znt``
    and before the lowest model level's heating reads it.
    """

    validate_surface_drag_selection(selection)
    if selection.bbc != SEMI_SLIP:
        raise ValueError(
            "only bbc=3 replaces the interior dissten average at k=1 "
            "(src/solve1.F:932-935)"
        )
    cupy = importlib.import_module("cupy")
    compiled = compiled or compile_cuda_surface_drag()
    stream = stream or cupy.cuda.get_current_stream()
    shapes = surface_drag_shapes(ni, nj, nk)
    names = (*SURFACE_STATE_NAMES, "zf")
    values = _device_arrays(arrays, names, shapes)
    binding = SurfaceDragBinding(
        selection=selection, compiled=compiled, arrays=values, dz=1.0
    )
    with stream:
        launched = enqueue_surface_dissipation(
            binding, {}, ni=ni, nj=nj, nk=nk, stream=stream
        )
        event = cupy.cuda.Event()
        event.record(stream)
    return SurfaceDragLaunch(
        event,
        {
            **binding.record(),
            "phase": "solve1-surface-dissipation-rate",
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "kernel": {"names": list(launched), "target_arch": compiled.target_arch},
        },
        compiled,
    )


# --------------------------------------------------------------------------
# Stream-level enqueue helpers.
#
# ``gettau`` calls ``sfcstress`` between the interior stress build and the
# momentum stress divergence (``src/turb.F:4417``), so the lower-face
# replacement has to be enqueued in the middle of an existing launch
# sequence rather than as its own stream episode.  These helpers assume the
# caller has already entered the target stream; the ``launch_*`` functions
# above are thin wrappers that enter one and record an event.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurfaceDragBinding:
    """One lower-boundary selection bound to its device carriers.

    ``arrays`` holds the surface carriers this operator owns -- the lowest
    model level wind, the exchange coefficients derived from it, and the
    surface dissipation rate.  The momentum carriers it also needs (``rf``,
    ``mf``, ``kmv``, ``u3d``, ``v3d``, ``t13``, ``t23``) come from the caller
    that already owns them, so nothing is copied and nothing is duplicated.
    """

    selection: SurfaceDragSelection
    compiled: CompiledSurfaceDragKernels
    arrays: Mapping[str, Any]
    dz: float
    # src/param.F:9038-9040 cgs1,cgs2,cgs3 -- the Fornberg weights that
    # extrapolate the three lowest mass pressures to the ground at
    # src/turb.F:292.  They belong to the grid, not to the surface, so they
    # are supplied rather than derived here; isfcflx=1 is the only consumer.
    surface_pressure_weights: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        validate_surface_drag_selection(self.selection)
        if not self.selection.active:
            raise ValueError(
                "bbc=1 keeps CM1's free-slip lower boundary and binds no "
                "surface operator"
            )
        if isinstance(self.dz, bool) or not isinstance(self.dz, (int, float)):
            raise TypeError("dz must be numeric")
        if not math.isfinite(float(self.dz)) or float(self.dz) <= 0.0:
            raise ValueError("dz must be finite and > 0")
        missing = [name for name in SURFACE_STATE_NAMES if name not in self.arrays]
        if missing:
            raise ValueError(
                "missing surface carriers: " + ", ".join(sorted(missing))
            )
        if self.selection.uses_surface_enthalpy_flux:
            weights = self.surface_pressure_weights
            if (
                not isinstance(weights, tuple)
                or len(weights) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in weights
                )
            ):
                raise ValueError(
                    "isfcflx=1 needs the three src/param.F:9038-9040 surface "
                    "extrapolation weights; without them src/turb.F:292's "
                    "surface pressure is undefined and every enthalpy flux "
                    "built from it is undefined with it"
                )

    def record(self) -> dict[str, object]:
        return {
            "schema": SURFACE_DRAG_SCHEMA,
            "abi": SURFACE_DRAG_ABI,
            "claim_limit": SURFACE_DRAG_CLAIM_LIMIT,
            "authority": dict(AUTHORITY_SOURCE_SHA256),
            "authority_tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": self.compiled.source_sha256,
            "selection": self.selection.record(),
            "spacing_m": {"dz": float(self.dz)},
            "surface_pressure_weights": (
                [float(value) for value in self.surface_pressure_weights]
                if self.surface_pressure_weights is not None
                else None
            ),
        }


def _require_enqueued_carriers(
    arrays: Mapping[str, Any], names: tuple[str, ...], *, ni: int, nj: int, nk: int
) -> None:
    """Check every carrier this operator is about to index, before it does.

    The stream-level helpers are handed live device arrays a caller already
    owns, so nothing else checks them.  A carrier of the wrong extent would
    not fail: it would index into a neighbouring row and write a surface
    stress built from the wrong column, silently.
    """

    shapes = surface_drag_shapes(ni, nj, nk)
    for name in names:
        value = arrays.get(name)
        if value is None:
            raise ValueError(f"missing surface carrier: {name}")
        shape = tuple(int(size) for size in getattr(value, "shape", ()))
        if shape != shapes[name]:
            raise ValueError(
                f"{name} must have shape {shapes[name]}, not {shape}"
            )
        dtype = getattr(value, "dtype", None)
        if getattr(dtype, "name", None) != "float32":
            raise TypeError(f"{name} must be binary32, not {dtype}")
        flags = getattr(value, "flags", None)
        if flags is not None and not getattr(flags, "c_contiguous", True):
            raise ValueError(f"{name} must be C-order")


_SURFACE_LAYER_CARRIERS = (
    "u3d", "v3d", "zh", "terrain_msl",
    "u1", "v1", "s1", "za", "znt", "ust", "cd", "u10", "v10", "s10",
)
_SURFACE_WATER_CARRIERS = _SURFACE_LAYER_CARRIERS + ("xland",)
_SURFACE_ENTHALPY_COEFFICIENT_CARRIERS = ("xland", "s10", "cd", "ch", "cq")
_SURFACE_FLUX_CARRIERS = (
    "prs", "psfc", "ch", "cq", "s10", "tsk", "th0", "tha", "qva", "mavail",
    "thflux", "qvflux", "qsfc",
)
_SURFACE_TKE_FACE_CARRIERS = (
    "tkea", "kmh", "kmv", "khh", "khv", "znt", "ust",
)
_SURFACE_STRESS_CARRIERS = {
    NO_SLIP: ("u3d", "v3d", "mf", "rf", "kmv", "t13", "t23", "stau"),
    SEMI_SLIP: ("u1", "v1", "s1", "ust", "rf", "t13", "t23", "stau"),
}
_SURFACE_DISSIPATION_CARRIERS = ("zf", "znt", "ust", "epsd_sfc")


def enqueue_surface_layer(
    binding: SurfaceDragBinding,
    carriers: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
    nonconvergent: Any = None,
) -> tuple[str, ...]:
    """``getsfc``, on an already-entered stream.

    ``src/turb.F:522-616`` builds the lowest model level wind from the
    ground-relative wind and calls ``getcecd``.  ``tkebc`` is a separate CM1
    call later in the same driver (``src/turb.F:1378``, after ``tkekm`` at
    ``src/turb.F:1351``) and belongs to :func:`enqueue_surface_tke_face`;
    launching it here would read the eddy coefficients before ``tkekm`` has
    written them.
    """

    cupy = importlib.import_module("cupy")
    selection = binding.selection
    kernels = binding.compiled.kernels
    a = dict(binding.arrays)
    a.update(carriers)
    _require_enqueued_carriers(a, _SURFACE_LAYER_CARRIERS, ni=ni, nj=nj, nk=nk)
    u32, f32 = cupy.uint32, cupy.float32
    block = (SURFACE_DRAG_THREADS,)
    grid = _grid(ni * nj)
    launched: list[str] = [SURFACE_DRAG_KERNEL_NAMES[0]]
    kernels[0](
        grid,
        block,
        (
            a["u3d"], a["v3d"], a["zh"], a["terrain_msl"],
            a["u1"], a["v1"], a["s1"], a["za"],
            u32(ni), u32(nj),
            f32(selection.u_move_m_s), f32(selection.v_move_m_s),
        ),
        stream=stream,
    )
    if selection.uses_surface_layer:
        branch = selection.getcecd_branch
        if branch == GETCECD_SET_ZNT:
            index = 7
            arguments = (
                a["u1"], a["v1"], a["s1"], a["za"], a["znt"],
                a["ust"], a["cd"], a["u10"], a["v10"], a["s10"],
                u32(ni), u32(nj),
            )
        elif branch == GETCECD_SET_UST:
            index = 8
            arguments = (
                a["u1"], a["v1"], a["s1"], a["za"],
                a["znt"], a["ust"], a["cd"], a["u10"], a["v10"], a["s10"],
                u32(ni), u32(nj), f32(selection.cnst_ust),
            )
        elif branch == GETCECD_CONSTANT_CD:
            index = 1
            arguments = (
                a["u1"], a["v1"], a["s1"], a["za"],
                a["znt"], a["ust"], a["cd"], a["u10"], a["v10"], a["s10"],
                u32(ni), u32(nj), f32(selection.cnstcd),
            )
        else:
            index = 9
            _require_enqueued_carriers(
                a, ("xland",), ni=ni, nj=nj, nk=nk
            )
            if nonconvergent is None:
                raise ValueError(
                    "the wind-dependent water drag iterates and needs the "
                    "nonconvergence flag sfcphys.F:189-191 stops CM1 on"
                )
            arguments = (
                a["u1"], a["v1"], a["s1"], a["za"], a["xland"],
                a["znt"], a["ust"], a["cd"], a["u10"], a["v10"], a["s10"],
                nonconvergent, u32(ni), u32(nj), u32(selection.cecd),
            )
        kernels[index](grid, block, arguments, stream=stream)
        launched.append(SURFACE_DRAG_KERNEL_NAMES[index])
        if selection.isfcflx == 1:
            # src/sfcphys.F:197-236, inside the same getcecd call.
            _require_enqueued_carriers(
                a, _SURFACE_ENTHALPY_COEFFICIENT_CARRIERS, ni=ni, nj=nj, nk=nk
            )
            kernels[10](
                grid,
                block,
                (
                    a["xland"], a["s10"], a["cd"], a["ch"], a["cq"],
                    u32(ni), u32(nj), u32(selection.cecd), f32(selection.cnstce),
                ),
                stream=stream,
            )
            launched.append(SURFACE_DRAG_KERNEL_NAMES[10])
    return tuple(launched)


def enqueue_surface_enthalpy_flux(
    binding: SurfaceDragBinding,
    carriers: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    moisture: bool,
    stream: Any,
) -> tuple[str, ...]:
    """``sfcflux``'s computed branch, on an already-entered stream.

    ``src/turb.F:645-649`` calls ``sfcflux`` immediately after ``getcecd``
    inside the ``sfcmodel.eq.1`` block, so the enthalpy exchange coefficients
    it multiplies are the ones the same step just wrote.  The surface
    pressure it needs is ``src/turb.F:292``, extrapolated to the ground from
    the three lowest mass levels through the ``src/param.F:9038-9040``
    Fornberg weights; this port launches that extrapolation here, in front of
    its only consumer, rather than leaving it to a caller that would have to
    know the weights.
    """

    cupy = importlib.import_module("cupy")
    selection = binding.selection
    if not selection.uses_surface_enthalpy_flux:
        raise ValueError(
            "isfcflx=0 leaves the scalar lower flux at zero "
            "(src/turbtend.F:517-521) and launches nothing here"
        )
    kernels = binding.compiled.kernels
    a = dict(binding.arrays)
    a.update(carriers)
    _require_enqueued_carriers(a, _SURFACE_FLUX_CARRIERS, ni=ni, nj=nj, nk=nk)
    u32, f32 = cupy.uint32, cupy.float32
    block = (SURFACE_DRAG_THREADS,)
    grid = _grid(ni * nj)
    weights = binding.surface_pressure_weights
    kernels[11](
        grid,
        block,
        (
            a["prs"], a["psfc"], u32(ni), u32(nj),
            f32(weights[0]), f32(weights[1]), f32(weights[2]),
        ),
        stream=stream,
    )
    kernels[12](
        grid,
        block,
        (
            a["ch"], a["cq"], a["s10"], a["psfc"], a["tsk"],
            a["th0"], a["tha"], a["qva"], a["mavail"],
            a["thflux"], a["qvflux"], a["qsfc"],
            u32(ni), u32(nj), u32(1 if moisture else 0),
        ),
        stream=stream,
    )
    return (SURFACE_DRAG_KERNEL_NAMES[11], SURFACE_DRAG_KERNEL_NAMES[12])


def enqueue_surface_tke_face(
    binding: SurfaceDragBinding,
    carriers: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
) -> tuple[str, ...]:
    """``tkebc``'s ``bbc=2/3`` lower face, on an already-entered stream.

    ``src/turb.F:1378`` calls ``tkebc`` after ``tkekm``
    (``src/turb.F:1351``), which is the call that writes ``kmh``/``kmv`` and
    ``khh``/``khv`` at every interior face.  ``tkebc``'s ``bbc=2/3`` branch
    then replaces the free-slip zero eddy viscosity at ``k=1`` with
    ``karman*znt*ust`` and rescales the scalar face from the ``k=2`` ratio
    (``src/turb.F:3244-3260``), so it has to run after the coefficient
    operators, not before them, and exactly once.
    """

    cupy = importlib.import_module("cupy")
    kernels = binding.compiled.kernels
    a = dict(binding.arrays)
    a.update(carriers)
    _require_enqueued_carriers(
        a, _SURFACE_TKE_FACE_CARRIERS, ni=ni, nj=nj, nk=nk
    )
    u32 = cupy.uint32
    kernels[2](
        _grid(ni * nj),
        (SURFACE_DRAG_THREADS,),
        (
            a["tkea"], a["kmh"], a["kmv"], a["khh"], a["khv"],
            a["znt"], a["ust"], u32(ni), u32(nj),
        ),
        stream=stream,
    )
    return (SURFACE_DRAG_KERNEL_NAMES[2],)


def enqueue_surface_stress(
    binding: SurfaceDragBinding,
    carriers: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
) -> tuple[str, ...]:
    """``sfcstress`` plus ``stau``, on an already-entered stream.

    Enqueue this after the interior ``t13``/``t23`` build and before the
    momentum stress divergence reads ``t13(i,j,1)``
    (``src/turb.F:4417``, then ``src/turbtend.F:1454`` and
    ``src/turbtend.F:2224``).  ``sfcstress`` computes no dissipation rate:
    CM1 forms that inside ``solve1`` at the point of use
    (``src/solve1.F:956-957``), which is :func:`enqueue_surface_dissipation`.
    """

    cupy = importlib.import_module("cupy")
    selection = binding.selection
    kernels = binding.compiled.kernels
    a = dict(binding.arrays)
    a.update(carriers)
    _require_enqueued_carriers(
        a, _SURFACE_STRESS_CARRIERS[selection.bbc], ni=ni, nj=nj, nk=nk
    )
    u32, f32 = cupy.uint32, cupy.float32
    block = (SURFACE_DRAG_THREADS,)
    faces = _grid((ni + 1) * nj + ni * (nj + 1))
    surface = _grid(ni * nj)
    launched: list[str] = []
    if selection.bbc == NO_SLIP:
        kernels[3](
            faces,
            block,
            (
                a["u3d"], a["v3d"], a["mf"], a["rf"], a["kmv"],
                a["t13"], a["t23"], u32(ni), u32(nj),
                f32(1.0 / float(binding.dz)),
                f32(selection.u_move_m_s), f32(selection.v_move_m_s),
            ),
            stream=stream,
        )
        launched.append(SURFACE_DRAG_KERNEL_NAMES[3])
    else:
        kernels[4](
            faces,
            block,
            (
                a["u1"], a["v1"], a["s1"], a["ust"], a["rf"],
                a["t13"], a["t23"], u32(ni), u32(nj),
            ),
            stream=stream,
        )
        launched.append(SURFACE_DRAG_KERNEL_NAMES[4])
    kernels[5](
        surface,
        block,
        (a["t13"], a["t23"], a["rf"], a["stau"], u32(ni), u32(nj)),
        stream=stream,
    )
    launched.append(SURFACE_DRAG_KERNEL_NAMES[5])
    return tuple(launched)


def enqueue_surface_dissipation(
    binding: SurfaceDragBinding,
    carriers: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
) -> tuple[str, ...]:
    """The ``bbc=3`` surface dissipation rate, on an already-entered stream.

    ``src/solve1.F:955-957`` forms the rate from the depth ``dz1`` of the
    lowest model level: ``ust`` cubed, times the natural log of
    ``(dz1 + zntmp)/zntmp``, divided by ``karman*dz1``,

    inside the dissipative-heating block itself, from the ``ust`` and
    ``zntmp`` the surface layer already produced.  This operator materializes
    that one temporary, so it has to be enqueued between ``getcecd`` and the
    lowest-level heating that reads it -- never after, or the heating reads a
    carrier nothing has written yet and the lowest model level gets exactly
    zero frictional heat.  ``src/solve1.F:932-935`` runs it for ``bbc=3``
    only; every other lower boundary heats ``k=1`` from the interior
    ``dissten`` average instead and this launches nothing.
    """

    if binding.selection.bbc != SEMI_SLIP:
        return ()
    cupy = importlib.import_module("cupy")
    kernels = binding.compiled.kernels
    a = dict(binding.arrays)
    a.update(carriers)
    _require_enqueued_carriers(
        a, _SURFACE_DISSIPATION_CARRIERS, ni=ni, nj=nj, nk=nk
    )
    u32 = cupy.uint32
    kernels[6](
        _grid(ni * nj),
        (SURFACE_DRAG_THREADS,),
        (a["zf"], a["znt"], a["ust"], a["epsd_sfc"], u32(ni), u32(nj)),
        stream=stream,
    )
    return (SURFACE_DRAG_KERNEL_NAMES[6],)


def surface_drag_selection_from_config(config: Any) -> SurfaceDragSelection:
    """Bind a parsed ``[closures]`` table to CM1's lower-boundary selectors.

    The configuration keys carry CM1's own namelist names, so this is a
    rename-free binding.  The frame velocity comes with them because
    ``src/turb.F:303-304`` makes every surface wind ground-relative.
    """

    closures = config.closures
    return validate_surface_drag_selection(
        SurfaceDragSelection(
            bbc=closures.bbc,
            sfcmodel=closures.sfcmodel,
            isfcflx=closures.isfcflx,
            cecd=closures.cecd,
            cnstcd=closures.cnstcd,
            cnstce=closures.cnstce,
            set_znt=closures.set_znt,
            cnst_znt=closures.cnst_znt,
            set_ust=closures.set_ust,
            cnst_ust=closures.cnst_ust,
            u_move_m_s=closures.u_move_m_s,
            v_move_m_s=closures.v_move_m_s,
        )
    )


__all__ = [
    "enqueue_surface_enthalpy_flux",
    "cpu_surface_enthalpy_flux_inplace",
    "UNVALIDATED",
    "VALIDATED",
    "GETCECD_WIND_DEPENDENT",
    "GETCECD_CONSTANT_CD",
    "GETCECD_SET_UST",
    "GETCECD_SET_ZNT",
    "SURFACE_LAYER_CLAIM_LIMIT",
    "SURFACE_FLUX_STATE_NAMES",
    "SURFACE_PERSISTENT_NAMES",
    "surface_validation_markers",
    "SURFACE_UNVALIDATED",
    "SURFACE_VALIDATION",
    "surface_drag_selection_from_config",
    "enqueue_surface_stress",
    "enqueue_surface_layer",
    "SurfaceDragBinding",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_KARMAN_F32",
    "CUDA_SURFACE_DRAG_SOURCE",
    "CUDA_SURFACE_DRAG_SOURCE_SHA256",
    "CompiledSurfaceDragKernels",
    "FREE_SLIP",
    "FREE_SLIP_SELECTION",
    "NO_SLIP",
    "SEMI_SLIP",
    "SOURCE_ANCHORS",
    "SURFACE_DRAG_ABI",
    "SURFACE_DRAG_CLAIM_LIMIT",
    "SURFACE_DRAG_KERNEL_NAMES",
    "SURFACE_DRAG_SCHEMA",
    "SURFACE_STATE_NAMES",
    "SurfaceDragLaunch",
    "SurfaceDragSelection",
    "build_cuda_surface_drag_source",
    "compile_cuda_surface_drag",
    "cpu_surface_drag_inplace",
    "launch_cuda_surface_layer",
    "launch_cuda_surface_stress",
    "surface_drag_shapes",
    "validate_surface_drag_selection",
]
