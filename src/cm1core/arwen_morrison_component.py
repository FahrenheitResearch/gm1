"""CM1 ``ptype=5`` bound to ArWen's public Morrison dispatch (``mp_physics=10``).

The adapter machinery lives in :mod:`cm1core.arwen_column_component`; this
module is the Morrison binding and nothing else.  Importing it imports neither
``gpuwm`` nor CuPy, inspects no CUDA device, and creates no CUDA context.

Morrison is double moment: CM1 transports four number concentrations beside
the six mass mixing ratios (``src/param.F:3417-3473``), and CM1's own order
for them is ``nci, ncs, ncr, ncg`` while ArWen's launcher takes
``nc, nr, ni, ns, ng``.  Nothing here is bound positionally; ``MORRISON_SPEC``
carries the name map and every consumer goes through it.

Cloud-droplet number is the one moment CM1 does not transport at ``inum=1``
(``src/morrison.F:325`` fixes ``INUM = 1``, so the ``numq = 11`` table at
``src/param.F:3476-3582`` is unreachable).  CM1 instead reassigns
``NC3D(K)`` to ``NDCNST`` times 1.E6 over ``RHO(K)`` inside the column loop before the size
distributions are formed (``src/morrison.F:1830-1833`` and ``:2770-2773``,
with ``RHO(K) = RHO3D(K)`` at ``:1673`` -- CM1 uses the model's prognostic
density, not WRF's diagnosed one).  This binding owns that carrier and
reproduces exactly that statement, from CM1's prognostic density, immediately
before every dispatch: see
:meth:`ArwenMorrisonComponent.prepare_component_owned_carriers`.

Authority (read-only tree ``work/native-reference/cm1r21.1``):

* ``src/param.F:3400-3473`` -- the ``ptype=5`` scalar table and ``iice``/``idm``.
* ``src/param.F:3407`` -- the ``graupel_init`` call, with ``ihail``, ``inum`` and ``ndcnst``.
* ``src/morrison.F:397-412,441-476`` -- what ``ihail`` selects.
* ``src/mp_driver.F:505-520`` -- the six-scalar positive-definite repair.
* ``src/mp_driver.F:546-558`` -- the ``MP_GRAUPEL`` argument order this map
  reproduces: ``q3d(7)=NNI, q3d(8)=NS, q3d(9)=NR, q3d(10)=NG``.
* ``src/morrison.F:686-706`` -- ``MP_GRAUPEL``'s own declared argument names.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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
    _Cm1ArwenColumnState,
    component_metadata as _component_metadata,
    component_metadata_bytes as _component_metadata_bytes,
    component_metadata_sha256 as _component_metadata_sha256,
)
from .dense_ice import DEFAULT_IHAIL_BY_COMPONENT

#: src/param.F:3407 with run/config_files/supercell/namelist.input:55.
MORRISON_DEFAULT_IHAIL = DEFAULT_IHAIL_BY_COMPONENT["morrison-mp10"]
from .errors import refuse
from .moisture_species import MORRISON_PROFILE


#: ``src/param.F:3407`` with ``run/config_files/supercell/namelist.input:55``:
#: the dense-ice branch CM1 pairs ``ptype=5`` with.


ARWEN_MORRISON_COMPONENT_ID = "morrison-mp10"
ARWEN_MORRISON_COMPONENT_ABI = "gpuwm-cm1.arwen-morrison-component/v1"
ARWEN_MORRISON_METADATA_SCHEMA = "gpuwm-cm1.arwen-morrison-metadata/v1"
ARWEN_MORRISON_RECEIPT_SCHEMA = "gpuwm-cm1.arwen-morrison-receipt/v1"
ARWEN_MORRISON_REFLECTIVITY_CONSUMER_ABI = (
    "gpuwm-cm1.arwen-morrison-reflectivity-consumer/v1"
)

#: ``gpuwm/core/morrison.py`` ``VERTICAL_LEVEL_BOUNDS``: the sedimentation
#: kernel is compiled at 64 and 256 column slots, so the launcher admits
#: 2..256 levels without any capacity overlay.
MORRISON_MIN_LEVELS = 2
MORRISON_MAX_LEVELS = 256

MORRISON_SPECIES: tuple[str, ...] = MORRISON_PROFILE.names
MORRISON_MASS_SPECIES: tuple[str, ...] = MORRISON_PROFILE.mass_names
MORRISON_NUMBER_SPECIES: tuple[str, ...] = MORRISON_PROFILE.number_names

#: ``src/mp_driver.F:546-558`` against ``src/morrison.F:687,701-705``.
#: CM1's seventh scalar is cloud-ice number and ArWen calls it ``ni``; CM1's
#: ninth is rain number and ArWen calls it ``nr``.  Swapping the two is
#: exactly the defect this table exists to prevent.
MORRISON_CARRIER_NAMES: Mapping[str, str] = MappingProxyType({
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "nci": "ni", "ncs": "ns", "ncr": "nr", "ncg": "ng",
})

#: ``src/morrison.F:1830-1833``.  The one moment CM1 does not transport.
MORRISON_COMPONENT_OWNED_CARRIERS: Mapping[str, str] = MappingProxyType({
    "nc": (
        "cloud-droplet number concentration; src/morrison.F:325 fixes INUM=1, "
        "so CM1 never transports it and reassigns it from ndcnst and the "
        "prognostic density inside the column loop (src/morrison.F:1830-1833, "
        ":2770-2773) before the size distributions are formed"
    ),
})

#: ``src/morrison.F:5153-5178``: every empty category reports 25 microns.
#: ArWen's kernel writes the same constant for an empty category, so these
#: are the values a column with no condensate carries.
MORRISON_EFFECTIVE_RADIUS_INITIAL_MICRONS: tuple[tuple[str, float], ...] = (
    ("effc", 25.0),
    ("effr", 25.0),
    ("effi", 25.0),
    ("effs", 25.0),
)
MORRISON_EFFECTIVE_RADIUS_FIELDS: tuple[str, ...] = tuple(
    name for name, _ in MORRISON_EFFECTIVE_RADIUS_INITIAL_MICRONS
)

#: The same seven ArWen scratch slots WSM6 uses; ``gpuwm/core/morrison.py``
#: names them identically.
MORRISON_SURFACE_SCRATCH: tuple[str, ...] = (
    "mp_rainnc",
    "mp_rainncv",
    "mp_snownc",
    "mp_snowncv",
    "mp_graupelnc",
    "mp_graupelncv",
    "mp_sr",
)

#: ``&param3 ndcnst`` in ``run/namelist.input:108`` and in CM1's own supercell
#: deck (``run/config_files/supercell/namelist.input:105``).  ArWen's kernel
#: writes ``nc = 250.0e6/rhoa`` as a literal, so this is the only value GM1
#: can execute.
MORRISON_NDCNST_CM3 = 250.0
_MORRISON_NC_PER_KG_TIMES_RHO = MORRISON_NDCNST_CM3 * 1.0e6

MORRISON_COMPONENT_PERSISTENCE: tuple[str, ...] = (
    ("h_diabatic",)
    + MORRISON_EFFECTIVE_RADIUS_FIELDS
    + tuple(MORRISON_COMPONENT_OWNED_CARRIERS)
    + MORRISON_SURFACE_SCRATCH
)
MORRISON_RESTART_REQUIRED: tuple[str, ...] = MORRISON_COMPONENT_PERSISTENCE

MORRISON_ALL_CARRIERS: tuple[str, ...] = (
    "thp", "thb", "p", "alt", "phb", "php",
) + MORRISON_SPECIES

_CLAIM_LIMIT = (
    "public ArWen Morrison double-moment component execution and CM1 "
    "carrier/persistence binding only; not CM1 eqtset=2 coupling, dycore "
    "parity, a coupled trajectory, restart identity, or model validation"
)


@dataclass(frozen=True, slots=True)
class ArwenMorrisonCarriers:
    """Caller-owned device carriers consumed or updated by Morrison.

    Field names are CM1's ``q3d`` names in CM1's own axis order, so a caller
    binds what ``src/param.F:3446-3449`` transports and the adapter performs
    the rename onto ArWen's launcher keywords.
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
    nci: object
    ncs: object
    ncr: object
    ncg: object


MORRISON_SPEC = ColumnComponentSpec(
    component_id=ARWEN_MORRISON_COMPONENT_ID,
    scheme_name="Morrison",
    component_abi=ARWEN_MORRISON_COMPONENT_ABI,
    metadata_schema=ARWEN_MORRISON_METADATA_SCHEMA,
    receipt_schema=ARWEN_MORRISON_RECEIPT_SCHEMA,
    reflectivity_consumer_abi=ARWEN_MORRISON_REFLECTIVITY_CONSUMER_ABI,
    mp_physics=10,
    cm1_ptype=MORRISON_PROFILE.cm1_ptype,
    profile=MORRISON_PROFILE,
    carrier_names=MORRISON_CARRIER_NAMES,
    component_owned_carriers=MORRISON_COMPONENT_OWNED_CARRIERS,
    effective_radius_initial_microns=MORRISON_EFFECTIVE_RADIUS_INITIAL_MICRONS,
    surface_scratch=MORRISON_SURFACE_SCRATCH,
    minimum_levels=MORRISON_MIN_LEVELS,
    maximum_levels=MORRISON_MAX_LEVELS,
    # gpuwm/core/refl.py:599-609 consumes qr/nr/qs/ns/qg/ng at mp_physics=10;
    # qv is the vapour argument and qc/qi are not arguments to it.
    reflectivity_species=("qv", "qr", "ncr", "qs", "ncs", "qg", "ncg"),
    reflectivity_absent_species=("qc", "qi", "nci"),
    claim_limit=_CLAIM_LIMIT,
    code_token="MORR",
    gate="ArWen-Morrison-component",
    carriers_type=ArwenMorrisonCarriers,
    capacity_profile_supported=False,
    extra_execution=MappingProxyType({
        # src/morrison.F:325 fixes INUM=1 in an unmodified r21.1, which is why
        # cloud-droplet number is not on the transported axis.
        "inum": 1,
        "inum_authority": "src/morrison.F:314-325,656",
        # src/morrison.F:329,1832 and gpuwm/core/kernels/morrison.cu:124,:1310.
        "ndcnst_cm3": MORRISON_NDCNST_CM3,
        "ndcnst_authority": (
            "src/morrison.F:329,1832; gpuwm/core/kernels/morrison.cu:124,:1310"
        ),
        "cloud_droplet_number_refresh": (
            "GM1 reassigns nc = ndcnst*1e6/rho from CM1's prognostic density "
            "immediately before dispatch (src/morrison.F:1830-1833,2770-2773 "
            "with :1673), ahead of the admitted kernel's own end-of-call "
            "refresh from its diagnosed density"
        ),
    }),
)


def require_ndcnst(value: object, *, option: str = "physics.ndcnst") -> float:
    """Admit CM1's ``ndcnst`` only where ArWen's kernel actually reads it.

    ``src/morrison.F:329`` stores the namelist value and ``:1832`` converts it
    from cm-3 to kg-1.  ArWen's transcription writes the conversion out as the
    literal ``nc[idx] = 250.0e6f / rhoa``
    (``gpuwm/core/kernels/morrison.cu:124,:1310``), so no other droplet
    concentration is executable through this seam.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        numeric = None
    else:
        numeric = float(value)
    if numeric != MORRISON_NDCNST_CM3:
        refuse(
            code="GM1-MORRISON-NDCNST",
            gate="ArWen-Morrison-component",
            reason=(
                "the admitted ArWen Morrison kernel writes the constant droplet "
                "concentration as a literal 250e6 kg-1 times density "
                "(gpuwm/core/kernels/morrison.cu:124,:1310); a different ndcnst would "
                "be recorded in the run's configuration and receipts while every "
                "column continued to activate 250 cm-3 of cloud droplets, so the "
                "autoconversion rate and the cloud effective radius the run "
                "reported would belong to a droplet spectrum nobody selected"
            ),
            remedy=f"set {option} = {MORRISON_NDCNST_CM3}",
            option=option,
            value=value,
        )
    return MORRISON_NDCNST_CM3


class ArwenMorrisonComponent(ArwenColumnComponent):
    """A persistent CM1-owned binding to ArWen's public Morrison dispatch."""

    SPEC = MORRISON_SPEC

    def prepare_component_owned_carriers(
        self, cupy: object, carriers: object, state: _Cm1ArwenColumnState
    ) -> None:
        """Reassign ``nc`` from ``ndcnst`` and CM1's prognostic density.

        ``src/morrison.F:1830-1833`` and ``:2770-2773`` do exactly this at the
        top of each column's work, before the size distributions are formed,
        and ``:1673`` makes ``RHO(K)`` the model's prognostic density rather
        than the diagnosed ``PRES/(R*T)`` of the WRF original.  The admitted
        ArWen kernel instead refreshes ``nc`` at the END of a call from its own
        diagnosed density (``gpuwm/core/kernels/morrison.cu:124,:1310``), which
        would leave the value one large step stale and on the wrong density;
        writing it here, from ``alt = 1/rho``, restores CM1's own statement
        order and CM1's own density.
        """

        # alt is CM1's specific volume 1/rho, already bound read-only, so this
        # is one multiply rather than a divide and needs no extra carrier.
        cupy.multiply(
            carriers.alt,
            cupy.float32(_MORRISON_NC_PER_KG_TIMES_RHO),
            out=state.nc,
        )


#: The step result is component-independent.
ArwenMorrisonStepResult = ArwenColumnStepResult


def component_metadata(
    *, capacity_profile: str | None = None, ihail: int = MORRISON_DEFAULT_IHAIL
) -> dict[str, object]:
    """Return the stable, CUDA-free semantic contract for the Morrison binding."""

    return _component_metadata(
        MORRISON_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def component_metadata_bytes(
    *, capacity_profile: str | None = None, ihail: int = MORRISON_DEFAULT_IHAIL
) -> bytes:
    """Canonical LF-terminated JSON bytes for :func:`component_metadata`."""

    return _component_metadata_bytes(
        MORRISON_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def component_metadata_sha256(
    *, capacity_profile: str | None = None, ihail: int = MORRISON_DEFAULT_IHAIL
) -> str:
    """SHA-256 of the canonical component metadata bytes."""

    return _component_metadata_sha256(
        MORRISON_SPEC, capacity_profile=capacity_profile, ihail=ihail
    )


def create_arwen_morrison_component(
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
    ihail: int = MORRISON_DEFAULT_IHAIL,
) -> ArwenMorrisonComponent:
    """Create a validated, still CUDA-free persistent Morrison component."""

    return ArwenMorrisonComponent(
        component_id=component_id,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
        dt_s=dt_s,
        capacity_profile=capacity_profile,
        ihail=ihail,
    )


__all__ = [
    "ARWEN_KERNEL_MANIFEST_MODULE",
    "ARWEN_MORRISON_COMPONENT_ABI",
    "ARWEN_MORRISON_COMPONENT_ID",
    "ARWEN_MORRISON_METADATA_SCHEMA",
    "ARWEN_MORRISON_RECEIPT_SCHEMA",
    "ARWEN_MORRISON_REFLECTIVITY_CONSUMER_ABI",
    "ARWEN_PUBLIC_MICROPHYSICS_CALL",
    "ARWEN_PUBLIC_MICROPHYSICS_MODULE",
    "ARWEN_PUBLIC_REFLECTIVITY_CALL",
    "ARWEN_PUBLIC_REFLECTIVITY_MODULE",
    "ARWEN_RUN_CONFIG_MODULE",
    "MORRISON_ALL_CARRIERS",
    "MORRISON_CARRIER_NAMES",
    "MORRISON_COMPONENT_OWNED_CARRIERS",
    "MORRISON_COMPONENT_PERSISTENCE",
    "MORRISON_EFFECTIVE_RADIUS_FIELDS",
    "MORRISON_MASS_SPECIES",
    "MORRISON_MAX_LEVELS",
    "MORRISON_MIN_LEVELS",
    "MORRISON_NDCNST_CM3",
    "MORRISON_NUMBER_SPECIES",
    "MORRISON_RESTART_REQUIRED",
    "MORRISON_SPEC",
    "MORRISON_SPECIES",
    "MORRISON_SURFACE_SCRATCH",
    "ArwenMorrisonCarriers",
    "ArwenMorrisonComponent",
    "ArwenMorrisonStepResult",
    "component_metadata",
    "component_metadata_bytes",
    "component_metadata_sha256",
    "create_arwen_morrison_component",
    "require_ndcnst",
]
