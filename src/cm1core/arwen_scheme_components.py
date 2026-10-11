"""Two further ArWen column dispatches bound as CM1 components.

The adapter machinery lives in :mod:`cm1core.arwen_column_component` and the
scalar axes in :mod:`cm1core.moisture_species`; this module is the binding for
classic Thompson (CM1 ``ptype=3``, ArWen ``mp_physics=8``) and WDM6 (ArWen
``mp_physics=16``, which CM1 r21.1 has no scheme for) and nothing else.
Importing it imports neither ``gpuwm`` nor CuPy, inspects no CUDA device and
creates no CUDA context.

Both axes carry CM1's own six mass mixing ratios, in CM1's own order, as their
leading rows.  That is not decoration: the sealed post-physics EOS/``qa``
publication (``cuda_gm1_tilestream_post_wsm6``) is compiled for exactly six
mass species, so a component whose mass prefix is a different length has no
publication path and is refused by name in
:mod:`cm1core.column_component_blockers` rather than bound here.

Authority (read-only tree ``work/native-reference/cm1r21.1``):

* ``src/param.F:3322-3398`` -- the ``ptype=3`` eight-scalar table: six mass
  rows then ``nci`` and ``ncr``, with ``iice=1`` and ``idm=1``.
* ``src/param.F:936-950`` -- ``ptype=3`` is absent from the list CM1
  force-downgrades to ``eqtset=1``, so it keeps ``eqtset=2``.
* ``src/param.F:3141-4509`` -- the complete ``ptype`` dispatch.  It never
  names WDM6, which is why ``WDM6_PROFILE`` cites ArWen for its axis.

ArWen (admitted estate, ``cm1core.engine_pin``):

* ``gpuwm/core/microphysics.py`` ``_apply_thompson`` -- ``required = ("qc",
  "qi", "ni", "qs", "qg", "qr", "nr")``, which is exactly CM1's ``ptype=3``
  axis under ArWen's names for the two moments.
* ``gpuwm/core/wdm6.py:151`` -- ``required = ("qi", "qs", "qg", "nn", "nc",
  "nr", "effc", "effi", "effs")``.
* ``gpuwm/core/wdm6_constants.py:39`` -- WDM6 imports
  ``rimed_ice_constants`` from ``wsm6_constants``, so ``wdm6_hail_opt``
  selects the same five constants ``ihail`` selects for WSM6.
* ``gpuwm/core/refl.py:610-637`` -- which species each scheme's 10 cm
  reflectivity diagnostic consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .arwen_column_component import (
    ArwenColumnComponent,
    ArwenColumnStepResult,
    ColumnComponentSpec,
    component_metadata as _component_metadata,
    component_metadata_bytes as _component_metadata_bytes,
    component_metadata_sha256 as _component_metadata_sha256,
)
from .dense_ice import DEFAULT_IHAIL_BY_COMPONENT, default_ihail
from .moisture_species import (
    THOMPSON_AEROSOL_PROFILE,
    THOMPSON_PROFILE,
    WDM6_PROFILE,
)


ARWEN_THOMPSON_COMPONENT_ID = "thompson-mp8"
ARWEN_THOMPSON_AEROSOL_COMPONENT_ID = "thompson-aerosol-mp28"
ARWEN_WDM6_COMPONENT_ID = "wdm6-mp16"

#: ``gpuwm/core/kernels/wsm6.cu:663-673`` and
#: ``gpuwm/core/kernels/wdm6.cu:1286-1317`` write these three microns for an
#: empty category; ``gpuwm/core/kernels/thompson.cu:335-337,379-381`` writes
#: the same three numbers, in metres, and multiplies by 1e6 as the last
#: operation, so the stored words are ``float32`` 2.49, 4.99 and 9.99 exactly.
_EFFECTIVE_RADIUS_INITIAL_MICRONS: tuple[tuple[str, float], ...] = (
    ("effc", 2.49),
    ("effi", 4.99),
    ("effs", 9.99),
)

#: The seven ArWen surface accumulators every one of these schemes names.
_SURFACE_SCRATCH: tuple[str, ...] = (
    "mp_rainnc",
    "mp_rainncv",
    "mp_snownc",
    "mp_snowncv",
    "mp_graupelnc",
    "mp_graupelncv",
    "mp_sr",
)


@dataclass(frozen=True, slots=True)
class ArwenThompsonCarriers:
    """CM1 ``ptype=3`` carriers, in CM1's own ``q3d`` order.

    ``nci`` and ``ncr`` are CM1's names (``src/param.F:3353-3354``); ArWen's
    launcher calls the same two moments ``ni`` and ``nr``, and the rename is
    performed by :data:`THOMPSON_SPEC`'s carrier map rather than positionally.
    """

    thp: object
    thb: object
    p: object
    alt: object
    phb: object
    php: object
    #: WRF's ``grid%w_2``.  ``gpuwm/core/microphysics.py`` ``_apply_thompson``
    #: reads ``state.w[:nz]`` for its cloud sedimentation, which is WRF's own
    #: lower full-level slice rather than a mass-level average.  GM1 stages its
    #: dycore ``w3d`` here unchanged: the same quantity on the same stagger.
    w: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    nci: object
    ncr: object


@dataclass(frozen=True, slots=True)
class ArwenWdm6Carriers:
    """WDM6 carriers.  ``ncn`` is the prognostic CCN number ArWen calls ``nn``."""

    thp: object
    thb: object
    p: object
    alt: object
    phb: object
    php: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    ncn: object
    ncc: object
    ncr: object


#: ``src/param.F:3353-3354`` against ``gpuwm/core/microphysics.py``'s
#: ``required`` tuple.  CM1's seventh scalar is cloud-ice number and ArWen
#: calls it ``ni``; CM1's eighth is rain number and ArWen calls it ``nr``.
THOMPSON_CARRIER_NAMES: Mapping[str, str] = MappingProxyType({
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "nci": "ni", "ncr": "nr",
})

#: ``gpuwm/core/wdm6.py:151``.  CM1 numbers no WDM6 scalar, so the three
#: number rows take CM1's own ``nc``-prefixed spelling for a number
#: concentration of a named category and map onto ArWen's launcher keywords.
WDM6_CARRIER_NAMES: Mapping[str, str] = MappingProxyType({
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "ncn": "nn", "ncc": "nc", "ncr": "nr",
})


THOMPSON_SPEC = ColumnComponentSpec(
    component_id=ARWEN_THOMPSON_COMPONENT_ID,
    scheme_name="Thompson",
    component_abi="gpuwm-cm1.arwen-thompson-component/v1",
    metadata_schema="gpuwm-cm1.arwen-thompson-metadata/v1",
    receipt_schema="gpuwm-cm1.arwen-thompson-receipt/v1",
    reflectivity_consumer_abi="gpuwm-cm1.arwen-thompson-reflectivity-consumer/v1",
    mp_physics=8,
    cm1_ptype=THOMPSON_PROFILE.cm1_ptype,
    profile=THOMPSON_PROFILE,
    carrier_names=THOMPSON_CARRIER_NAMES,
    component_owned_carriers=MappingProxyType({}),
    effective_radius_initial_microns=_EFFECTIVE_RADIUS_INITIAL_MICRONS,
    surface_scratch=_SURFACE_SCRATCH,
    # gpuwm/physics_vertical_contract.py THOMPSON_VERTICAL_LEVEL_BOUNDS.
    minimum_levels=2,
    maximum_levels=256,
    # gpuwm/core/refl.py:610-628: qv is the vapour argument, then qr, nr, qs
    # and qg; the graupel number is the scheme's own same-call shadow, not a
    # transported scalar, so it is not listed as a species here.
    reflectivity_species=("qv", "qr", "ncr", "qs", "qg"),
    reflectivity_absent_species=("qc", "qi", "nci"),
    claim_limit=(
        "public ArWen classic Thompson component execution and CM1 "
        "carrier/persistence binding only; not CM1 eqtset=2 coupling, dycore "
        "parity, a coupled trajectory, restart identity, or model validation"
    ),
    code_token="THOM",
    gate="ArWen-Thompson-component",
    carriers_type=ArwenThompsonCarriers,
    capacity_profile_supported=False,
    extra_interface_carriers=("w",),
    extra_execution=MappingProxyType({
        # gpuwm/core/microphysics.py _apply_thompson resolves its packaged
        # classic tables before any GPU setup and byte-validates every asset.
        "classic_tables": (
            "gpuwm/core/thompson_runtime.py load_classic_device_tables "
            "byte-validates the packaged classic tables; GPUWM_THOMPSON_TABLE_"
            "ROOT overrides the root and a wrong or missing root fails closed"
        ),
        # src/param.F:3322-3325.
        "cm1_iice": THOMPSON_PROFILE.iice,
        "cm1_idm": THOMPSON_PROFILE.idm,
        "dense_ice_selector": "absent; see cm1core.dense_ice.DENSE_ICE_SELECTOR_ABSENT",
    }),
)


WDM6_SPEC = ColumnComponentSpec(
    component_id=ARWEN_WDM6_COMPONENT_ID,
    scheme_name="WDM6",
    component_abi="gpuwm-cm1.arwen-wdm6-component/v1",
    metadata_schema="gpuwm-cm1.arwen-wdm6-metadata/v1",
    receipt_schema="gpuwm-cm1.arwen-wdm6-receipt/v1",
    reflectivity_consumer_abi="gpuwm-cm1.arwen-wdm6-reflectivity-consumer/v1",
    mp_physics=16,
    cm1_ptype=WDM6_PROFILE.cm1_ptype,
    profile=WDM6_PROFILE,
    carrier_names=WDM6_CARRIER_NAMES,
    component_owned_carriers=MappingProxyType({}),
    effective_radius_initial_microns=_EFFECTIVE_RADIUS_INITIAL_MICRONS,
    surface_scratch=_SURFACE_SCRATCH,
    # gpuwm/core/wdm6.py:64 -> wdm6_constants.WDM6_VERTICAL_LEVEL_BOUNDS,
    # whose deepest compiled WDM6_KMAX tier is 80.
    minimum_levels=2,
    maximum_levels=80,
    # gpuwm/core/refl.py:629-637.
    reflectivity_species=("qv", "qr", "ncr", "qs", "qg"),
    reflectivity_absent_species=("qc", "qi", "ncn", "ncc"),
    claim_limit=(
        "public ArWen WDM6 component execution and CM1 carrier/persistence "
        "binding only; CM1 r21.1 has no WDM6 ptype, so this is not a CM1 "
        "scheme and no CM1 differential exists for it; not CM1 eqtset=2 "
        "coupling, dycore parity, a coupled trajectory, restart identity, or "
        "model validation"
    ),
    code_token="WDM6",
    gate="ArWen-WDM6-component",
    carriers_type=ArwenWdm6Carriers,
    capacity_profile_supported=False,
    extra_execution=MappingProxyType({
        "cm1_scheme_present": False,
        "cm1_absence": (
            "src/param.F:3141-4509 dispatches ptype 0,1,2,3,4,5,6,7,26,27,28,"
            "50,51,52,53,55 and names no WDM6 scheme; this component has no "
            "CM1 counterpart and therefore no CM1 differential"
        ),
        # gpuwm/core/wdm6_constants.py:39.
        "dense_ice_constants_shared_with": "wsm6-mp6",
        "cm1_iice": WDM6_PROFILE.iice,
        "cm1_idm": WDM6_PROFILE.idm,
    }),
)


@dataclass(frozen=True, slots=True)
class ArwenThompsonAerosolCarriers:
    """Aerosol-aware Thompson carriers.

    Eleven transported scalars: the classic eight plus cloud-droplet number and
    the water- and ice-friendly aerosol numbers the scheme activates droplets
    and nucleates ice from.  CM1 r21.1 numbers no aerosol-aware Thompson
    ``ptype``, so the ``nc``-prefixed spellings are CM1's naming convention
    applied to ArWen's axis, and the carrier map performs the rename.
    """

    thp: object
    thb: object
    p: object
    alt: object
    phb: object
    php: object
    #: WRF's ``grid%w_2``.  ``gpuwm/core/microphysics_aerosol.py:436,463`` reads
    #: ``state.w[:-1]`` for CCN activation and cloud sedimentation.
    w: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    nci: object
    ncr: object
    ncc: object
    nwfa: object
    nifa: object


#: ``gpuwm/core/microphysics_aerosol.py``'s own state reads.
THOMPSON_AEROSOL_CARRIER_NAMES: Mapping[str, str] = MappingProxyType({
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "nci": "ni", "ncr": "nr", "ncc": "nc", "nwfa": "nwfa", "nifa": "nifa",
})

#: ``gpuwm/core/microphysics_aerosol.py:220`` requires both by name.  WRF
#: carries them as two-dimensional surface emission rates, not as column
#: scalars, so they are component-owned surface persistence rather than rows on
#: the transported axis.
THOMPSON_AEROSOL_SURFACE_CARRIERS: Mapping[str, str] = MappingProxyType({
    "nwfa2d": (
        "WRF's surface water-friendly aerosol emission rate, kg-1 s-1; a "
        "two-dimensional field the scheme reads and CM1 has no equivalent for"
    ),
    "nifa2d": (
        "WRF's surface ice-friendly aerosol emission rate, kg-1 s-1; the same"
    ),
})


THOMPSON_AEROSOL_SPEC = ColumnComponentSpec(
    component_id=ARWEN_THOMPSON_AEROSOL_COMPONENT_ID,
    scheme_name="Thompson aerosol-aware",
    component_abi="gpuwm-cm1.arwen-thompson-aerosol-component/v1",
    metadata_schema="gpuwm-cm1.arwen-thompson-aerosol-metadata/v1",
    receipt_schema="gpuwm-cm1.arwen-thompson-aerosol-receipt/v1",
    reflectivity_consumer_abi=(
        "gpuwm-cm1.arwen-thompson-aerosol-reflectivity-consumer/v1"
    ),
    mp_physics=28,
    cm1_ptype=THOMPSON_AEROSOL_PROFILE.cm1_ptype,
    profile=THOMPSON_AEROSOL_PROFILE,
    carrier_names=THOMPSON_AEROSOL_CARRIER_NAMES,
    component_owned_carriers=MappingProxyType({}),
    component_owned_surface_carriers=THOMPSON_AEROSOL_SURFACE_CARRIERS,
    effective_radius_initial_microns=_EFFECTIVE_RADIUS_INITIAL_MICRONS,
    surface_scratch=_SURFACE_SCRATCH,
    # gpuwm/physics_vertical_contract.py THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS.
    minimum_levels=2,
    maximum_levels=256,
    # gpuwm/core/refl.py:610-628 -- one branch for mp=8 and mp=28, because
    # WRF's calc_refl10cm has no aerosol-aware arm and reads no droplet number.
    reflectivity_species=("qv", "qr", "ncr", "qs", "qg"),
    reflectivity_absent_species=("qc", "qi", "nci", "ncc", "nwfa", "nifa"),
    claim_limit=(
        "public ArWen aerosol-aware Thompson component execution and CM1 "
        "carrier/persistence binding only; CM1 r21.1 has no aerosol-aware "
        "Thompson ptype, so this is not a CM1 scheme and no CM1 differential "
        "exists for it; the aerosol initial state is ArWen's own declared "
        "fallback rather than a WRF global-emission profile; not CM1 eqtset=2 "
        "coupling, dycore parity, a coupled trajectory, restart identity, or "
        "model validation"
    ),
    code_token="AERO",
    gate="ArWen-Thompson-aerosol-component",
    carriers_type=ArwenThompsonAerosolCarriers,
    capacity_profile_supported=False,
    extra_interface_carriers=("w",),
    extra_execution=MappingProxyType({
        "cm1_scheme_present": False,
        "cm1_absence": (
            "src/param.F:3141-4509 dispatches ptype 0,1,2,3,4,5,6,7,26,27,28,"
            "50,51,52,53,55; its ptype=3 is the CLASSIC Thompson scheme and it "
            "numbers no aerosol-aware variant.  This component has no CM1 "
            "counterpart and therefore no CM1 differential"
        ),
        "aerosol_initial_state": (
            "this seam reaches the scheme through gpuwm.core.microphysics.apply "
            "without calling gpuwm.core.microphysics.microphysics_init, so the "
            "nwfa/nifa profile is ArWen's own declared fallback rather than a "
            "WRF global-emission initial state; ArWen fills it on the first "
            "call from aerosol_profile_needs_fill"
        ),
        "dense_ice_selector": "absent; see cm1core.dense_ice.DENSE_ICE_SELECTOR_ABSENT",
    }),
)


class ArwenThompsonAerosolComponent(ArwenColumnComponent):
    """A persistent CM1-owned binding to ArWen's aerosol-aware Thompson dispatch."""

    SPEC = THOMPSON_AEROSOL_SPEC



class ArwenThompsonComponent(ArwenColumnComponent):
    """A persistent CM1-owned binding to ArWen's public classic Thompson dispatch."""

    SPEC = THOMPSON_SPEC


class ArwenWdm6Component(ArwenColumnComponent):
    """A persistent CM1-owned binding to ArWen's public WDM6 dispatch."""

    SPEC = WDM6_SPEC


#: The step result is component-independent.
ArwenSchemeStepResult = ArwenColumnStepResult

THOMPSON_DEFAULT_IHAIL = default_ihail(ARWEN_THOMPSON_COMPONENT_ID)
WDM6_DEFAULT_IHAIL = DEFAULT_IHAIL_BY_COMPONENT[ARWEN_WDM6_COMPONENT_ID]


def thompson_component_metadata(
    *, capacity_profile: str | None = None
) -> dict[str, object]:
    """The stable, CUDA-free semantic contract for the Thompson binding."""

    return _component_metadata(
        THOMPSON_SPEC, capacity_profile=capacity_profile, ihail=THOMPSON_DEFAULT_IHAIL
    )


def thompson_component_metadata_bytes(
    *, capacity_profile: str | None = None
) -> bytes:
    return _component_metadata_bytes(
        THOMPSON_SPEC, capacity_profile=capacity_profile, ihail=THOMPSON_DEFAULT_IHAIL
    )


def thompson_component_metadata_sha256(
    *, capacity_profile: str | None = None
) -> str:
    return _component_metadata_sha256(
        THOMPSON_SPEC, capacity_profile=capacity_profile, ihail=THOMPSON_DEFAULT_IHAIL
    )


def wdm6_component_metadata(
    *, capacity_profile: str | None = None, ihail: int = WDM6_DEFAULT_IHAIL
) -> dict[str, object]:
    """The stable, CUDA-free semantic contract for the WDM6 binding."""

    return _component_metadata(
        WDM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def wdm6_component_metadata_bytes(
    *, capacity_profile: str | None = None, ihail: int = WDM6_DEFAULT_IHAIL
) -> bytes:
    return _component_metadata_bytes(
        WDM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def wdm6_component_metadata_sha256(
    *, capacity_profile: str | None = None, ihail: int = WDM6_DEFAULT_IHAIL
) -> str:
    return _component_metadata_sha256(
        WDM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def create_arwen_thompson_component(
    *,
    component_id: str,
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    dt_s: float,
    capacity_profile: str | None = None,
    ihail: int | None = THOMPSON_DEFAULT_IHAIL,
) -> ArwenThompsonComponent:
    """Create a validated, still CUDA-free persistent Thompson component."""

    return ArwenThompsonComponent(
        component_id=component_id, nx=nx, ny=ny, nz=nz,
        dx_m=dx_m, dy_m=dy_m, dz_m=dz_m, dt_s=dt_s,
        capacity_profile=capacity_profile, ihail=ihail,
    )


def create_arwen_wdm6_component(
    *,
    component_id: str,
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    dt_s: float,
    capacity_profile: str | None = None,
    ihail: int = WDM6_DEFAULT_IHAIL,
) -> ArwenWdm6Component:
    """Create a validated, still CUDA-free persistent WDM6 component."""

    return ArwenWdm6Component(
        component_id=component_id, nx=nx, ny=ny, nz=nz,
        dx_m=dx_m, dy_m=dy_m, dz_m=dz_m, dt_s=dt_s,
        capacity_profile=capacity_profile, ihail=ihail,
    )


__all__ = [
    "ARWEN_THOMPSON_AEROSOL_COMPONENT_ID",
    "ARWEN_THOMPSON_COMPONENT_ID",
    "ArwenThompsonAerosolCarriers",
    "ArwenThompsonAerosolComponent",
    "THOMPSON_AEROSOL_CARRIER_NAMES",
    "THOMPSON_AEROSOL_SPEC",
    "THOMPSON_AEROSOL_SURFACE_CARRIERS",
    "ARWEN_WDM6_COMPONENT_ID",
    "ArwenSchemeStepResult",
    "ArwenThompsonCarriers",
    "ArwenThompsonComponent",
    "ArwenWdm6Carriers",
    "ArwenWdm6Component",
    "THOMPSON_CARRIER_NAMES",
    "THOMPSON_DEFAULT_IHAIL",
    "THOMPSON_SPEC",
    "WDM6_CARRIER_NAMES",
    "WDM6_DEFAULT_IHAIL",
    "WDM6_SPEC",
    "create_arwen_thompson_component",
    "create_arwen_wdm6_component",
    "thompson_component_metadata",
    "thompson_component_metadata_bytes",
    "thompson_component_metadata_sha256",
    "wdm6_component_metadata",
    "wdm6_component_metadata_bytes",
    "wdm6_component_metadata_sha256",
]
