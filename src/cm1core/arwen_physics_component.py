"""CM1-owned adapter for ArWen's radiation, PBL and cumulus components.

This is the sibling of :mod:`cm1core.arwen_wsm6_component`: CM1 owns the
selector, the carrier binding and the receipt; ArWen owns the scheme, its
kernels and its lookup tables.  Like that module it is import-light -- it
never imports ``gpuwm``, CuPy or numpy at module scope, so a configuration
can be admitted, refused or reported on a machine with no GPU and no
installed physics estate.

What this module is for
-----------------------
CM1 names three families GM1 had no key for: ``radopt`` (``param11``),
``ipbl`` (``param2``) and ``cuparam`` (``param2``).  ArWen ships components
for most of them.  The mapping is not one-to-one and the gaps are the
interesting part, so every CM1 selector value resolves here to exactly one
of three dispositions:

``admitted``
    ArWen ships the scheme and GM1 can reach it: the component id, its
    ArWen selector values, its public entry point and the audited bytes of
    every file it executes are recorded.
``unported``
    ArWen ships no equivalent.  The refusal names the CM1 source that would
    have run and the concrete field that would be left at zero.
``uncoupled``
    ArWen ships the scheme but GM1 cannot yet supply its inputs.  The
    refusal names the exact missing carriers.  This is the honest state of
    every PBL scheme today: all of them read surface-layer products
    (``hfx``, ``qfx``, ``psim``, ``psih``, ``br``, ``wspd``, ``u10``,
    ``v10``) that CM1 forms in ``src/sfcphys.F`` and GM1 does not yet carry
    on the device.

Nothing here silently substitutes a scheme for another one.  A selector
that cannot be executed refuses by key and value; it is never quietly
downgraded to the neutral profile, because a PBL scheme handed zero surface
fluxes and zero stability functions returns a plausible, finite, neutral
answer -- the failure mode that produces a wrong forecast rather than a
stopped run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import util as _import_util
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .cm1_physics_families import (
    CUPARAM_LABELS,
    IPBL_GM1_EXTENSIONS,
    IPBL_LABELS,
    RADOPT_LABELS,
)
from .errors import refuse


ARWEN_PHYSICS_COMPONENT_ABI = "gpuwm-cm1.arwen-physics-component/v1"
ARWEN_PHYSICS_COMPONENT_SCHEMA = "gpuwm-cm1.arwen-physics-component/v1"
ARWEN_PHYSICS_SEAM_GATE = "cartesian-runtime"

#: Dispositions, in the order a reader should think about them.
ADMITTED = "admitted"
UNPORTED = "unported"
UNCOUPLED = "uncoupled"
IDENTITY = "identity"

#: The surface-layer products every WRF-lineage PBL scheme reads, by the
#: names ArWen's launchers use.  CM1 forms them in ``src/sfcphys.F`` under
#: ``sfcmodel>=1``; GM1's device state carries none of them today, which is
#: why every ``ipbl>=1`` value below is ``uncoupled`` rather than
#: ``admitted``.  Naming them here means the refusal names the work, and
#: means this list is what has to shrink for the disposition to change.
PBL_REQUIRED_SURFACE_CARRIERS: tuple[str, ...] = (
    "hfx",
    "qfx",
    "psim",
    "psih",
    "br",
    "wspd",
    "u10",
    "v10",
    "xland",
    "psfc",
)
#: The radiation inputs CM1 gathers before ``radiation_driver``
#: (``src/cm1.F:3693-3707``) that GM1's idealized lane has no carrier for.
RADIATION_REQUIRED_SURFACE_CARRIERS: tuple[str, ...] = (
    "tsk",
    "albd",
    "emiss",
    "xland",
    "glw",
    "gsw",
)


def _audited(entries: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(sorted(entries.items())))


#: Measured on the published gpuwm 2.8.9 wheel, the estate
#: :mod:`cm1core.engine_pin` admits.  These components were first measured on
#: ArWen 2.5.8 (audited commit f47f177d4b2e0da95568d70230f5ce9785e632c7) and
#: were not re-measured at the 2.7.4 pin; the 2.8.0 re-pin brought every entry
#: onto the admitted estate and the 2.8.9 re-pin re-measured them (19 of the
#: 28 files changed: YSU, Shin-Hong, MYNN, New Tiedtke and the RRTMG drivers
#: and kernels; MYJ, the RRTMG longwave chain and the lookup tables did not).
#: The New Tiedtke unit and, from 2.8.9, the YSU and Shin-Hong units compile
#: with device headers this table does not list (glibc_flt32.cuh, and
#: ysu_topo.cuh for YSU); they join the table before any stage couples one of
#: these schemes.  The digests are byte admission only: no
#: Cartesian stage couples these components, and none has a GPU validation on
#: 2.8.9.  Every file an admitted component executes is listed with the bytes
#: GM1 measured, so a drifting estate refuses instead of running unknown
#: physics.
ARWEN_COMPONENT_SURFACE: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "ysu-pbl1": _audited(
            {
                "gpuwm/core/ysu.py": "bde629698c989bae0571dec6e1623b430200c62c6cf92e6b51c2d7bc42d75f57",
                "gpuwm/core/kernels/ysu.cu": "46f4642c73cdc00ad08822bb89788d576fc4bc6ca7a9d8a68aa2272d66c7a669",
            }
        ),
        "shinhong-pbl11": _audited(
            {
                "gpuwm/core/shinhong.py": "3d5c1758e8e789c4b9c23ace408f18b7af0fce5b6b9eadff6068f8ab238540fd",
                "gpuwm/core/kernels/shinhong.cu": "26a78c8f8b24a2c783f6780db4b6b4a8fc52eb30a1e51bc71a7919a33ee8f6ed",
            }
        ),
        "myj-pbl2": _audited(
            {
                "gpuwm/core/myjpbl.py": "dabfda3bf88f9f49798da6043efb7a1d4fdbb0617ed7b5be9b023f32863a04ff",
                "gpuwm/core/kernels/myjpbl.cu": "d0e0b3dde6ba1729460694a3bf730ac82420e85d3979327953a7e45bf85719f1",
            }
        ),
        "mynn-pbl5": _audited(
            {
                "gpuwm/core/mynn_pbl.py": "631f32c9c8f71581f2941c26e08f5f112b78b63cfb9693e988f211dad0a42ffe",
                "gpuwm/core/mynn_pbl_gpu.py": "f1a97ec8c787b2247368c996e3fb7dca233e3a9ab7ccb9d44541861eb312a35c",
                "gpuwm/core/mynn_pbl_runtime.py": "69362728ef2ee0a33669cfa1928532a1f0693ff35e1e43f32e7da5d7f3c513ab",
                "gpuwm/core/kernels/mynn_pbl.cu": "176d00a656786395753cd254921fe74968448c8daafc7c7f254b97c484338cda",
            }
        ),
        "rrtmg-legacy-ra4": _audited(
            {
                "gpuwm/core/rrtmg_legacy.py": "ea3adcfdc84b2df75fc4c952cdb3ae0b103ce8d4cc4ded3dda221de23cc8f6c9",
                "gpuwm/core/rrtmg_legacy_prep.py": "93f230eff3d226539656dd32f955ab077cb17f8ff3ff57dddbc97bb63bd2828e",
                "gpuwm/core/rrtmg_mcica.py": "3947f739003363e0902d1c058e80ca0111545c4e0b2acc0548891041fa0be7de",
                "gpuwm/core/rrtmg_lw.py": "e899ce92bb2e09ed8d70a5368795fed21eb1bbd0c4faf8d3cd0195d3182bc6cd",
                "gpuwm/core/rrtmg_sw.py": "349c9d0d0d6380ae1951aa4ee5c8ecee29831cf8ab8dcede54ca5c5ad28dad8a",
                "gpuwm/core/kernels/rrtmg_lw.cu": "241382acf34bc312e6361822acc057ec3fedf2f31514b477ff939ef723632592",
                "gpuwm/core/kernels/rrtmg_lw_chain.cu": "a71a779ea3a733e422414ee5d5d885cb904972fe32ac2bfe6438d6c5214c1252",
                "gpuwm/core/kernels/rrtmg_lw_taugb02_10_11_12.cu": "fe5d57d1eb2649005d2748ae40f849c340b03a5f3d3a61c7fb506d0c69792485",
                "gpuwm/core/kernels/rrtmg_lw_taugb03_05.cu": "5de3f0c7c700cff993d5d5a3caa868bbd68d810ea30a902e6e92d652e5612cf7",
                "gpuwm/core/kernels/rrtmg_lw_taugb06_09.cu": "ab677eb0de0faeb2b7a3f4a5923c6777e11128ab7950d9b63abc9cdb61125157",
                "gpuwm/core/kernels/rrtmg_lw_taugb13_16.cu": "104c3157451dc878e3f9d751db1cf01a4368b1a2892825678370c5159a787ff6",
                "gpuwm/core/kernels/rrtmg_sw.cu": "43134d6d6d2be4b3e668c7ab92d5850f1f992a09812dd09c9536fce5d9ab791e",
                "gpuwm/core/kernels/rrtmg_mcica_wrf.cu": "80cd32a2feee70f1d0fe2db3eaf02386de4d970bfc9dc8e22d71fafbab5564b6",
                # The AER lookup tables themselves, as WRF distributes them.
                "gpuwm/data/wrf_radiation/RRTMG_LW_DATA": "bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97",
                "gpuwm/data/wrf_radiation/RRTMG_SW_DATA": "a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf",
                "gpuwm/data/wrf_radiation/rrtmg_lw_statics.npz": "edd2508db89180667b0f80b4cdd991f4aa1447e711bbfe3a6db8de5fdb778d62",
            }
        ),
        # First shipped in ArWen 2.7.4; absent from the 2.5.8 estate, which
        # is what made cuparam=1 an engine question rather than a
        # configuration one.
        "ntiedtke-cu16": _audited(
            {
                "gpuwm/core/ntiedtke.py": "ad24fc80bce9998e84e73d5cc87c8241831a11e8fc67cc21d2b37505bc4b1fcf",
                "gpuwm/core/kernels/ntiedtke.cu": "17fe4aac03ed6b5191bda49db9eebe6c51c29d0e542bafc8ea33989f363caa42",
            }
        ),
    }
)

#: The AER lookup tables RRTMG reads, separated out because "get their
#: admission right" is a question about these two files and not about the
#: Python around them.  ``RRTMG_LW_DATA``/``RRTMG_SW_DATA`` are the
#: unformatted-Fortran absorption-coefficient tables WRF distributes under
#: the AER BSD-3-Clause licence; ``rrtmg_lw_statics.npz`` is ArWen's
#: pre-reduced form of the longwave table.
RRTMG_LOOKUP_TABLES: Mapping[str, str] = _audited(
    {
        "gpuwm/data/wrf_radiation/RRTMG_LW_DATA": "bcfdee24b63a4c909522a329b8e16c539f0173c7e5aea2caf933ab4fe28c5c97",
        "gpuwm/data/wrf_radiation/RRTMG_SW_DATA": "a7d25f5b4d33be8629cbef7ecacc1ff413bf398a021297793e843ba1cc627baf",
        "gpuwm/data/wrf_radiation/rrtmg_lw_statics.npz": "edd2508db89180667b0f80b4cdd991f4aa1447e711bbfe3a6db8de5fdb778d62",
    }
)
RRTMG_LOOKUP_TABLE_LICENCE = (
    "gpuwm/data/wrf_radiation/LICENSE-AER-RRTMG-BSD-3-Clause.txt with "
    "gpuwm/data/wrf_radiation/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt"
)


@dataclass(frozen=True, slots=True)
class ArwenPhysicsComponent:
    """One CM1 selector value resolved against ArWen's component surface."""

    family: str
    cm1_selector: str
    cm1_value: int
    cm1_label: str
    disposition: str
    component_id: str | None = None
    arwen_selectors: Mapping[str, int] | None = None
    entry_point: str | None = None
    audited_files: Mapping[str, str] | None = None
    missing_carriers: tuple[str, ...] = ()
    reason: str = ""
    cm1_source: str = ""
    minimum_engine: str | None = None

    @property
    def runnable(self) -> bool:
        return self.disposition in (ADMITTED, IDENTITY)

    def record(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "family": self.family,
            "cm1_selector": self.cm1_selector,
            "cm1_value": self.cm1_value,
            "cm1_label": self.cm1_label,
            "cm1_source": self.cm1_source,
            "disposition": self.disposition,
            "runnable": self.runnable,
        }
        if self.component_id is not None:
            payload["component_id"] = self.component_id
        if self.arwen_selectors is not None:
            payload["arwen_selectors"] = dict(self.arwen_selectors)
        if self.entry_point is not None:
            payload["entry_point"] = self.entry_point
        if self.audited_files is not None:
            payload["audited_files"] = dict(self.audited_files)
            payload["audited_surface_sha256"] = hashlib.sha256(
                json.dumps(
                    dict(self.audited_files), sort_keys=True, separators=(",", ":")
                ).encode("ascii")
            ).hexdigest()
        if self.missing_carriers:
            payload["missing_carriers"] = list(self.missing_carriers)
        if self.reason:
            payload["reason"] = self.reason
        if self.minimum_engine is not None:
            payload["minimum_engine"] = self.minimum_engine
        return payload


def _pbl_uncoupled(value: int, component: str, entry: str, selectors: Mapping[str, int]) -> ArwenPhysicsComponent:
    return ArwenPhysicsComponent(
        family="pbl",
        cm1_selector="ipbl",
        cm1_value=value,
        cm1_label=IPBL_LABELS[value],
        cm1_source=(
            "src/turb.F:1885-2620 (ipbl dispatch), "
            "src/solve1.F:1086-1113 (thpten applied to thten1)"
        ),
        disposition=UNCOUPLED,
        component_id=component,
        arwen_selectors=MappingProxyType(dict(selectors)),
        entry_point=entry,
        audited_files=ARWEN_COMPONENT_SURFACE[component],
        missing_carriers=PBL_REQUIRED_SURFACE_CARRIERS,
        reason=(
            "ArWen ships this scheme and GM1 can address it, but every "
            "WRF-lineage PBL scheme reads surface-layer products that GM1 "
            "does not yet carry on the device; CM1 forms them in "
            "src/sfcphys.F under sfcmodel>=1"
        ),
    )


_PBL_COMPONENTS: Mapping[int, ArwenPhysicsComponent] = MappingProxyType(
    {
        0: ArwenPhysicsComponent(
            family="pbl",
            cm1_selector="ipbl",
            cm1_value=0,
            cm1_label=IPBL_LABELS[0],
            cm1_source="src/param.F:752-753 leaves idopbl false",
            disposition=IDENTITY,
            reason="no PBL tendency exists, so nothing is added to thten1",
        ),
        1: _pbl_uncoupled(1, "ysu-pbl1", "gpuwm.core.ysu.launch_ysu", {"bl_pbl_physics": 1}),
        2: ArwenPhysicsComponent(
            family="pbl",
            cm1_selector="ipbl",
            cm1_value=2,
            cm1_label=IPBL_LABELS[2],
            cm1_source="src/turb.F:2071-2123 (the Louis-type scheme, inline)",
            disposition=UNPORTED,
            reason=(
                "this is CM1's own scheme and has no WRF or ArWen counterpart; "
                "it is roughly fifty lines inline in src/turb.F and is the "
                "cheapest of the six to transcribe, but nothing transcribes it "
                "today, so selecting it would leave thpten at zero while "
                "src/solve1.F:1086-1113 added that zero to thten1"
            ),
        ),
        3: ArwenPhysicsComponent(
            family="pbl",
            cm1_selector="ipbl",
            cm1_value=3,
            cm1_label=IPBL_LABELS[3],
            cm1_source="src/module_bl_gfsedmf.F (2327 lines), dispatched at src/turb.F:2124",
            disposition=UNPORTED,
            reason=(
                "ArWen ships no eddy-diffusivity-mass-flux component at all; "
                "selecting it would leave the boundary layer unmixed while the "
                "receipt named GFS-EDMF"
            ),
        ),
        4: _pbl_uncoupled(4, "mynn-pbl5", "gpuwm.core.mynn_pbl_runtime", {"bl_pbl_physics": 5}),
        5: ArwenPhysicsComponent(
            family="pbl",
            cm1_selector="ipbl",
            cm1_value=5,
            cm1_label=IPBL_LABELS[5],
            cm1_source="src/module_bl_mynn.F dispatched at src/turb.F:2192 with the level-3 branch",
            disposition=UNPORTED,
            reason=(
                "ArWen's MYNN component is the level-2.5 closure with EDMF "
                "(physics_registry_v2.json pins bl_mynn_closure=2.6); it has no "
                "level-3 branch, and running 2.5 while the receipt said 3 would "
                "misreport the closure that produced the profile"
            ),
        ),
        6: _pbl_uncoupled(6, "myj-pbl2", "gpuwm.core.myjpbl.launch_myj_pbl", {"bl_pbl_physics": 2}),
        11: _pbl_uncoupled(
            11, "shinhong-pbl11", "gpuwm.core.shinhong.launch_shinhong", {"bl_pbl_physics": 11}
        ),
    }
)

_RADIATION_COMPONENTS: Mapping[int, ArwenPhysicsComponent] = MappingProxyType(
    {
        0: ArwenPhysicsComponent(
            family="radiation",
            cm1_selector="radopt",
            cm1_value=0,
            cm1_label=RADOPT_LABELS[0],
            cm1_source="src/param.F:6347 sets radtim to 1.0d60",
            disposition=IDENTITY,
            reason="no radiative tendency exists, so nothing is added to thten1",
        ),
        1: ArwenPhysicsComponent(
            family="radiation",
            cm1_selector="radopt",
            cm1_value=1,
            cm1_label=RADOPT_LABELS[1],
            cm1_source="src/radiation_driver.F:119, src/irrad3d.F, src/sorad3d.F",
            disposition=UNPORTED,
            reason=(
                "the NASA-Goddard longwave and shortwave codes CM1 adapted from "
                "ARPS have no ArWen counterpart; ArWen's radiation components are "
                "RRTMG, RTE+RRTMGP, RRTM+Dudhia and an analytic clear-sky proxy. "
                "Selecting radopt=1 would leave lwten/swten/thraten at zero while "
                "src/solve1.F:1056-1085 added that zero to thten1, so the run "
                "would report the Goddard scheme and integrate no radiation"
            ),
        ),
        2: ArwenPhysicsComponent(
            family="radiation",
            cm1_selector="radopt",
            cm1_value=2,
            cm1_label=RADOPT_LABELS[2],
            cm1_source=(
                "src/radiation_driver.F:314, src/module_ra_rrtmg_lw.F, "
                "src/module_ra_rrtmg_sw.F, cadence at src/cm1.F:3196-3199,3691-3713"
            ),
            disposition=UNCOUPLED,
            component_id="rrtmg-legacy-ra4",
            arwen_selectors=MappingProxyType({"ra_lw_physics": 4, "ra_sw_physics": 4}),
            entry_point="gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation",
            audited_files=ARWEN_COMPONENT_SURFACE["rrtmg-legacy-ra4"],
            missing_carriers=RADIATION_REQUIRED_SURFACE_CARRIERS,
            reason=(
                "CM1's RRTMG and ArWen's legacy-RRTMG adapter are the same WRF "
                "code, and ArWen ships the AER lookup tables, but "
                "src/cm1.F:3693-3707 hands radiation_driver a surface state "
                "(skin temperature, albedo, emissivity, land mask and the two "
                "surface flux carriers) that GM1's location-free idealized lane "
                "does not carry; run without it the scheme would see a 0 K "
                "ground with zero albedo and return a longwave cooling profile "
                "that is wrong everywhere in the column"
            ),
        ),
    }
)

_CUMULUS_COMPONENTS: Mapping[int, ArwenPhysicsComponent] = MappingProxyType(
    {
        0: ArwenPhysicsComponent(
            family="cumulus",
            cm1_selector="cuparam",
            cm1_value=0,
            cm1_label=CUPARAM_LABELS[0],
            cm1_source="src/solve1.F:1510 guards the whole block on cuparam>=1",
            disposition=IDENTITY,
            reason="no cumulus tendency exists, so nothing is added to thten1",
        ),
        1: ArwenPhysicsComponent(
            family="cumulus",
            cm1_selector="cuparam",
            cm1_value=1,
            cm1_label=CUPARAM_LABELS[1],
            cm1_source=(
                "src/cu_ntiedtke.F, src/init_physics.F:141-156 (cu_ntiedtke_init), "
                "src/solve1.F:1508-1541"
            ),
            disposition=UNCOUPLED,
            component_id="ntiedtke-cu16",
            arwen_selectors=MappingProxyType({"cu_physics": 16}),
            entry_point="gpuwm.core.ntiedtke",
            audited_files=ARWEN_COMPONENT_SURFACE["ntiedtke-cu16"],
            minimum_engine="gpuwm>=2.7.4",
            reason=(
                "CM1 r21.1 and ArWen ship the same scheme -- both are the CCPP/MMM "
                "cu_ntiedtke, and src/cu_ntiedtke.F:221-230 and ArWen's "
                "kernels/ntiedtke.cu:175-191 carry the identical scale-dependency "
                "factors -- but the ntiedtke kernel first appears in ArWen 2.7.4, "
                "and no Cartesian stage couples it"
            ),
        ),
    }
)

_FAMILY_TABLES: Mapping[str, Mapping[int, ArwenPhysicsComponent]] = MappingProxyType(
    {
        "pbl": _PBL_COMPONENTS,
        "radiation": _RADIATION_COMPONENTS,
        "cumulus": _CUMULUS_COMPONENTS,
    }
)
_FAMILY_KEYS: Mapping[str, str] = MappingProxyType(
    {"pbl": "closures.ipbl", "radiation": "radiation.radopt", "cumulus": "physics.cuparam"}
)


def resolve_component(family: str, value: int) -> ArwenPhysicsComponent:
    """Resolve one CM1 selector value; raises KeyError for an unknown value."""

    return _FAMILY_TABLES[family][value]


def component_inventory() -> dict[str, list[dict[str, object]]]:
    """Every selector value in every family, as one machine-readable table."""

    return {
        family: [table[value].record() for value in sorted(table)]
        for family, table in _FAMILY_TABLES.items()
    }


def require_runnable_component(family: str, value: int) -> ArwenPhysicsComponent:
    """Admit one selector value or refuse it by key, value and named defect."""

    component = resolve_component(family, value)
    if component.runnable:
        return component
    option = _FAMILY_KEYS[family]
    codes = {
        UNPORTED: "CM1-PHYSICS-COMPONENT-UNPORTED",
        UNCOUPLED: "CM1-PHYSICS-COMPONENT-UNCOUPLED",
    }
    remedy = f"set {option} to 0"
    if component.missing_carriers:
        remedy += (
            "; the coupling this needs is "
            + ", ".join(component.missing_carriers)
            + " on the device"
        )
    if component.minimum_engine is not None:
        remedy += f"; the component itself needs {component.minimum_engine}"
    refuse(
        code=codes[component.disposition],
        gate=ARWEN_PHYSICS_SEAM_GATE,
        reason=(
            f"{option}={value} selects {component.cm1_label}, which CM1 runs at "
            f"{component.cm1_source}, and {component.reason}"
        ),
        remedy=remedy,
        option=option,
        value=value,
    )


def installed_component_files(component_id: str) -> dict[str, str | None]:
    """Measured sha256 of every audited file of one component, or ``None``.

    Returns ``{}`` when no ``gpuwm`` distribution is importable at all, so a
    machine with no physics estate reports "not installed" rather than
    "drifted".
    """

    try:
        spec = _import_util.find_spec("gpuwm")
    except (ImportError, ValueError):
        return {}
    if spec is None or spec.origin is None:
        return {}
    root = Path(spec.origin).resolve().parent.parent
    measured: dict[str, str | None] = {}
    for name in ARWEN_COMPONENT_SURFACE[component_id]:
        path = root / name
        measured[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return measured


def component_engine_report(component_id: str) -> dict[str, object]:
    """Whether the installed estate carries this component, byte for byte."""

    expected = ARWEN_COMPONENT_SURFACE[component_id]
    measured = installed_component_files(component_id)
    if not measured:
        return {
            "component_id": component_id,
            "engine_installed": False,
            "admitted": False,
            "absent": sorted(expected),
            "moved": [],
        }
    absent = sorted(name for name, value in measured.items() if value is None)
    moved = sorted(
        name
        for name, value in measured.items()
        if value is not None and value != expected[name]
    )
    return {
        "component_id": component_id,
        "engine_installed": True,
        "admitted": not absent and not moved,
        "absent": absent,
        "moved": moved,
    }


def unvalidated_markers(
    *, radopt: int, ipbl: int, cuparam: int, rterm: int
) -> list[dict[str, object]]:
    """Machine-readable "this was not checked" rows for a run receipt.

    One row per family that is selected but whose numbers GM1 has not
    measured against an oracle.  A family that is off produces no row; a
    family that is validated produces no row.  Silence here means checked,
    which is why every row states what specifically was not checked.
    """

    markers: list[dict[str, object]] = []
    # rterm is the one family in this group that has been measured: a
    # raw-word device oracle against the evidenced CPU operator, and a paired
    # CM1 differential. It therefore produces no marker when it is on.
    if rterm not in (0, 1):
        markers.append(
            {
                "schema": "gpuwm-cm1.unvalidated-physics/v1",
                "family": "idealized-radiation",
                "selector": "rterm",
                "value": rterm,
                "component_id": None,
                "not_checked": "rterm is outside CM1's 0/1 and nothing measured it",
            }
        )
    not_checked = {
        "radiation": (
            "the RRTMG lookup tables are admitted offline -- CM1 r21.1's own "
            "run/RRTMG_LW_DATA and run/RRTMG_SW_DATA are byte-identical to "
            "ArWen's copies, 16 and 14 Fortran records exactly "
            "(evidence/radpbl/rrtmg-tables) -- but no heating rate has been "
            "compared against CM1, and no solar-geometry, ozone, trace-gas or "
            "cloud-optical-property path has been measured at all"
        ),
        "pbl": (
            "no tendency from this scheme has been compared against CM1's own "
            "ipbl dispatch, and the surface-layer inputs it reads are not yet "
            "carried on the device, so nothing about its profile is measured"
        ),
        "cumulus": (
            "CM1's src/cu_ntiedtke.F and ArWen's kernels/ntiedtke.cu are "
            "established as the same CCPP/MMM scheme with identical "
            "scale-dependency factors, but no tendency from either has been "
            "compared against the other"
        ),
    }
    for family, value in (("radiation", radopt), ("pbl", ipbl), ("cumulus", cuparam)):
        if value == 0:
            continue
        component = resolve_component(family, value)
        markers.append(
            {
                "schema": "gpuwm-cm1.unvalidated-physics/v1",
                "family": family,
                "selector": component.cm1_selector,
                "value": value,
                "component_id": component.component_id,
                "disposition": component.disposition,
                "not_checked": not_checked[family],
            }
        )
    return markers


def seam_record() -> dict[str, object]:
    """The complete, CUDA-free seam contract, for a receipt or for doctor."""

    return {
        "schema": ARWEN_PHYSICS_COMPONENT_SCHEMA,
        "component_abi": ARWEN_PHYSICS_COMPONENT_ABI,
        "claim_limit": (
            "CM1 selector to ArWen component resolution, byte admission of the "
            "component surface, and the named coupling each component still "
            "needs; not a coupled trajectory, a validated scheme, or a "
            "native-CM1 parity claim for any of these three families"
        ),
        "families": component_inventory(),
        "rrtmg_lookup_tables": dict(RRTMG_LOOKUP_TABLES),
        "rrtmg_lookup_table_licence": RRTMG_LOOKUP_TABLE_LICENCE,
        "pbl_required_surface_carriers": list(PBL_REQUIRED_SURFACE_CARRIERS),
        "radiation_required_surface_carriers": list(
            RADIATION_REQUIRED_SURFACE_CARRIERS
        ),
        "gm1_extension_ipbl_values": sorted(IPBL_GM1_EXTENSIONS),
    }


__all__ = [
    "ADMITTED",
    "ARWEN_COMPONENT_SURFACE",
    "ARWEN_PHYSICS_COMPONENT_ABI",
    "ARWEN_PHYSICS_COMPONENT_SCHEMA",
    "ARWEN_PHYSICS_SEAM_GATE",
    "ArwenPhysicsComponent",
    "IDENTITY",
    "PBL_REQUIRED_SURFACE_CARRIERS",
    "RADIATION_REQUIRED_SURFACE_CARRIERS",
    "RRTMG_LOOKUP_TABLES",
    "RRTMG_LOOKUP_TABLE_LICENCE",
    "UNCOUPLED",
    "UNPORTED",
    "component_engine_report",
    "component_inventory",
    "installed_component_files",
    "require_runnable_component",
    "resolve_component",
    "seam_record",
    "unvalidated_markers",
]
