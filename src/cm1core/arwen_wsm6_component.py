"""CM1 ``ptype=7`` bound to ArWen's public WSM6 dispatch (``mp_physics=6``).

The adapter machinery lives in :mod:`cm1core.arwen_column_component`; this
module is the WSM6 binding and nothing else.  Importing it imports neither
``gpuwm`` nor CuPy, inspects no CUDA device, and creates no CUDA context.

The sole physics execution edge is ``gpuwm.core.microphysics.apply`` with
``mp_physics=6``.  CM1 owns the adapter and carrier binding; ArWen owns the
microphysics preparation, WSM6 dispatch, latent-heating finish, and compiled
kernel lifecycle behind that public function.  The adapter never imports a
scheme-private WSM6 module and never copies an ArWen kernel.

Authority: ``src/param.F:3632-3678`` is the ``ptype=7`` scalar table and
``src/param.F:3697-3706`` is the ``mp_wsm6_init`` call that carries CM1's
``ihail`` into the scheme as ``hail_opt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .arwen_column_component import (
    ARWEN_KERNEL_MANIFEST_MODULE,
    ARWEN_PUBLIC_MICROPHYSICS_CALL,
    ARWEN_PUBLIC_MICROPHYSICS_MODULE,
    ARWEN_PUBLIC_REFLECTIVITY_CALL,
    ARWEN_PUBLIC_REFLECTIVITY_MODULE,
    ARWEN_RUN_CONFIG_MODULE,
    ArwenColumnComponent,
    ArwenColumnStepResult,
    ColumnComponentSpec,
    component_metadata as _component_metadata,
    component_metadata_bytes as _component_metadata_bytes,
    component_metadata_sha256 as _component_metadata_sha256,
)
from .dense_ice import DEFAULT_IHAIL
from .moisture_species import WSM6_PROFILE


ARWEN_WSM6_COMPONENT_ID = "wsm6-mp6"
ARWEN_WSM6_COMPONENT_ABI = "gpuwm-cm1.arwen-wsm6-component/v1"
ARWEN_WSM6_METADATA_SCHEMA = "gpuwm-cm1.arwen-wsm6-metadata/v1"
ARWEN_WSM6_RECEIPT_SCHEMA = "gpuwm-cm1.arwen-wsm6-receipt/v1"
ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI = (
    "gpuwm-cm1.arwen-wsm6-reflectivity-consumer/v1"
)

WSM6_MIN_LEVELS = 2
WSM6_MAX_LEVELS = 80
WSM6_SPECIES: tuple[str, ...] = WSM6_PROFILE.names
WSM6_MUTABLE_CARRIERS: tuple[str, ...] = ("thp",) + WSM6_SPECIES
WSM6_VOLUME_CARRIERS: tuple[str, ...] = ("thp", "p", "alt") + WSM6_SPECIES
WSM6_BASE_CARRIERS: tuple[str, ...] = ("thb", "phb")
WSM6_INTERFACE_CARRIERS: tuple[str, ...] = ("php",)
WSM6_ALL_CARRIERS: tuple[str, ...] = (
    "thp", "thb", "p", "alt", "phb", "php",
) + WSM6_SPECIES

WSM6_SURFACE_SCRATCH: tuple[str, ...] = (
    "mp_rainnc",
    "mp_rainncv",
    "mp_snownc",
    "mp_snowncv",
    "mp_graupelnc",
    "mp_graupelncv",
    "mp_sr",
)
WSM6_EFFECTIVE_RADIUS_FIELDS: tuple[str, ...] = ("effc", "effi", "effs")
_EFFECTIVE_RADIUS_INITIAL_MICRONS: tuple[tuple[str, float], ...] = (
    ("effc", 2.49),
    ("effi", 4.99),
    ("effs", 9.99),
)
WSM6_COMPONENT_PERSISTENCE: tuple[str, ...] = (
    ("h_diabatic",) + WSM6_EFFECTIVE_RADIUS_FIELDS + WSM6_SURFACE_SCRATCH
)
WSM6_RESTART_REQUIRED: tuple[str, ...] = WSM6_COMPONENT_PERSISTENCE

_CLAIM_LIMIT = (
    "public ArWen WSM6 component execution and CM1 carrier/persistence binding only; "
    "not CM1 eqtset=2 coupling, dycore parity, a coupled trajectory, restart identity, "
    "or model validation"
)


@dataclass(frozen=True, slots=True)
class ArwenWsm6Carriers:
    """Caller-owned device carriers consumed or updated by WSM6.

    The dataclass is deliberately backend-neutral.  Array validation belongs
    to :meth:`ArwenWsm6Component.apply`, where it can happen before CuPy or
    ArWen is imported.
    """

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


#: WSM6 is single-moment, so every CM1 scalar maps onto the identically named
#: ArWen carrier and the mapping is the identity (``src/param.F:3646-3651``).
WSM6_SPEC = ColumnComponentSpec(
    component_id=ARWEN_WSM6_COMPONENT_ID,
    scheme_name="WSM6",
    component_abi=ARWEN_WSM6_COMPONENT_ABI,
    metadata_schema=ARWEN_WSM6_METADATA_SCHEMA,
    receipt_schema=ARWEN_WSM6_RECEIPT_SCHEMA,
    reflectivity_consumer_abi=ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI,
    mp_physics=6,
    cm1_ptype=WSM6_PROFILE.cm1_ptype,
    profile=WSM6_PROFILE,
    carrier_names=MappingProxyType({name: name for name in WSM6_SPECIES}),
    component_owned_carriers=MappingProxyType({}),
    effective_radius_initial_microns=_EFFECTIVE_RADIUS_INITIAL_MICRONS,
    surface_scratch=WSM6_SURFACE_SCRATCH,
    minimum_levels=WSM6_MIN_LEVELS,
    maximum_levels=WSM6_MAX_LEVELS,
    reflectivity_species=("qv", "qr", "qs", "qg"),
    reflectivity_absent_species=("qc", "qi"),
    capacity_profile_supported=True,
    extra_execution=MappingProxyType({}),
    claim_limit=_CLAIM_LIMIT,
    code_token="WSM6",
    gate="ArWen-WSM6-component",
    carriers_type=ArwenWsm6Carriers,
)


class ArwenWsm6Component(ArwenColumnComponent):
    """A persistent CM1-owned binding to ArWen's public WSM6 dispatch."""

    SPEC = WSM6_SPEC


#: The step result is component-independent; the historical name is kept
#: because the eqtset=2 bridge annotates its field with it.
ArwenWsm6StepResult = ArwenColumnStepResult


def component_metadata(
    *, capacity_profile: str | None = None, ihail: int = DEFAULT_IHAIL
) -> dict[str, object]:
    """Return the stable, CUDA-free semantic contract for the WSM6 binding."""

    return _component_metadata(
        WSM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def component_metadata_bytes(
    *, capacity_profile: str | None = None, ihail: int = DEFAULT_IHAIL
) -> bytes:
    """Canonical LF-terminated JSON bytes for :func:`component_metadata`."""

    return _component_metadata_bytes(
        WSM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def component_metadata_sha256(
    *, capacity_profile: str | None = None, ihail: int = DEFAULT_IHAIL
) -> str:
    """SHA-256 of the canonical component metadata bytes."""

    return _component_metadata_sha256(
        WSM6_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def create_arwen_wsm6_component(
    *,
    component_id: str,
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    dt_s: float,
    ztop_m: float | None = None,
    capacity_profile: str | None = None,
    ihail: int = DEFAULT_IHAIL,
) -> ArwenWsm6Component:
    """Create a validated, still CUDA-free persistent WSM6 component."""

    return ArwenWsm6Component(
        component_id=component_id,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
        dt_s=dt_s,
        ztop_m=ztop_m,
        capacity_profile=capacity_profile,
        ihail=ihail,
    )


__all__ = [
    "ARWEN_KERNEL_MANIFEST_MODULE",
    "ARWEN_PUBLIC_MICROPHYSICS_CALL",
    "ARWEN_PUBLIC_MICROPHYSICS_MODULE",
    "ARWEN_PUBLIC_REFLECTIVITY_CALL",
    "ARWEN_PUBLIC_REFLECTIVITY_MODULE",
    "ARWEN_RUN_CONFIG_MODULE",
    "ARWEN_WSM6_COMPONENT_ABI",
    "ARWEN_WSM6_COMPONENT_ID",
    "ARWEN_WSM6_METADATA_SCHEMA",
    "ARWEN_WSM6_RECEIPT_SCHEMA",
    "ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI",
    "ArwenWsm6Carriers",
    "ArwenWsm6Component",
    "ArwenWsm6StepResult",
    "WSM6_ALL_CARRIERS",
    "WSM6_COMPONENT_PERSISTENCE",
    "WSM6_EFFECTIVE_RADIUS_FIELDS",
    "WSM6_MAX_LEVELS",
    "WSM6_MIN_LEVELS",
    "WSM6_RESTART_REQUIRED",
    "WSM6_SPEC",
    "WSM6_SPECIES",
    "WSM6_SURFACE_SCRATCH",
    "component_metadata",
    "component_metadata_bytes",
    "component_metadata_sha256",
    "create_arwen_wsm6_component",
]
