"""Source-ordered CM1 eqtset=2 bridge to the public ArWen WSM6 component.

Importing this module imports neither CuPy nor :mod:`gpuwm`.  The production
bridge executes all carrier arithmetic through CuPy only after the complete
backend-neutral ABI, lifecycle, and audited-engine gates pass.  Its order is
the bounded CM1 r21.1 lifecycle proven by ``O-EQTSET2-LIFECYCLE-V1``:

* final-RK removal and snapshot (``solve2.F:1561-1579``);
* pre-physics pressure/density rebuild (``solve2.F:1582-1595``);
* the public microphysics seam (``cm1.F:3405-3432``);
* ptype-5 eqtset-2 pressure/theta rebuild (``mp_driver.F:603-617``); and
* next-step held-rate recapture (``solve3.F:319-337``).

The bridge intentionally leaves pre-RK dissipative heating to its separately
admitted CM1 consumer and refuses fallout heating because public ArWen WSM6
does not expose the required terminal-velocity fields.  It exports its held
arrays and public-component persistence and can bind the
same exact device arrays on a fresh bridge before physics executes.  Durable
restart serialization remains a separate Rust-owned gate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, NoReturn

from . import engine_pin
from .arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SURFACE_SCRATCH,
    ArwenWsm6Carriers,
    ArwenWsm6Component,
    ArwenWsm6StepResult,
    create_arwen_wsm6_component,
)
from .errors import refuse
from .eqtset2_energy_terms import (
    EQTSET2_IDISS_LIVE_REFUSAL,
    require_efall_contract,
)


EQTSET2_WSM6_BRIDGE_ABI = "gpuwm-cm1.eqtset2-wsm6-bridge/v1"
EQTSET2_WSM6_RECEIPT_SCHEMA = "gpuwm-cm1.eqtset2-wsm6-receipt/v1"
EQTSET2_WSM6_RESTART_SCHEMA = "gpuwm-cm1.eqtset2-wsm6-restart/v1"

CM1_P00_F32 = 100000.0
CM1_RP00_F32 = 1.0 / CM1_P00_F32
CM1_RD_F32 = 287.04
CM1_RV_F32 = 461.5
CM1_CP_F32 = 1005.7
CM1_ROVCP_F32 = CM1_RD_F32 / CM1_CP_F32
CM1_CPDRD_F32 = CM1_CP_F32 / CM1_RD_F32

# ArWen WSM6 is the WRF-constant component: gpuwm/core/constants.py pins
# RD=287.0, CP=7*RD/2, hence RCP=2/7.  The bridge stages CM1 temperature into
# that theta convention, then converts the returned temperature back through
# CM1's ptype-5 equation-set rebuild.  Neither constant estate is blurred.
ARWEN_P00_F32 = 100000.0
ARWEN_RCP_F32 = 2.0 / 7.0

EQTSET2_HELD_FIELDS: tuple[str, ...] = (
    "qpten",
    "qtten",
    "qvten",
    "qcten",
)
EQTSET2_RESTART_FIELDS: tuple[str, ...] = EQTSET2_HELD_FIELDS + tuple(
    f"component.{name}" for name in WSM6_COMPONENT_PERSISTENCE
)
_BASE_FIELDS: tuple[str, ...] = (
    "pi0",
    "th0",
)
_VOLUME_FIELDS: tuple[str, ...] = (
    "rho",
    "pp3d",
    "th3d",
    "p",
    "alt",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
) + EQTSET2_HELD_FIELDS
_MUTABLE_FIELDS: tuple[str, ...] = (
    "rho",
    "pp3d",
    "th3d",
    "p",
    "alt",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
) + EQTSET2_HELD_FIELDS
_ALL_FIELDS: tuple[str, ...] = _BASE_FIELDS + _VOLUME_FIELDS + ("phb", "php")
_CLAIM_LIMIT = (
    "final-RK eqtset=2 held-tendency lifecycle and public ArWen WSM6 device "
    "binding with exact pre-apply restart restore and one-shot reflectivity "
    "consumption; live pre-RK idiss wiring, efall=1, coupled dycore trajectory, "
    "and product forecast admission remain red"
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _refuse(
    *, code: str, reason: str, remedy: str, option: str, value: object
) -> NoReturn:
    refuse(
        code=code,
        gate="CM1-eqtset2-ArWen-bridge",
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _refuse(
            code="CM1-EQTSET2-DIMENSION",
            reason=(
                f"{name} must be a strict integer >= {minimum}; accepting it would "
                "mis-size the source-ordered scalar lifecycle"
            ),
            remedy=f"pass {name} as an integer >= {minimum}",
            option=name,
            value=value,
        )
    return value


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            code="CM1-EQTSET2-SCALAR",
            reason=f"{name} must be a finite positive real scalar",
            remedy=f"pass {name} as a finite number > 0",
            option=name,
            value=value,
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            code="CM1-EQTSET2-SCALAR",
            reason=f"{name} must be finite and positive",
            remedy=f"pass {name} as a finite number > 0",
            option=name,
            value=value,
        )
    return result


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        _refuse(
            code="CM1-EQTSET2-BOOLEAN",
            reason=f"{name} must be a strict boolean lifecycle selection",
            remedy=f"pass {name} as true or false",
            option=name,
            value=value,
        )
    return value


def _shape(value: object) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_f32(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    return bool(
        dtype is not None
        and getattr(dtype, "name", None) == "float32"
        and getattr(dtype, "kind", None) == "f"
        and getattr(dtype, "itemsize", None) == 4
    )


def _is_c_contiguous(value: object) -> bool:
    flags = getattr(value, "flags", None)
    if flags is None:
        return False
    result = getattr(flags, "c_contiguous", None)
    if result is None:
        try:
            result = flags["C_CONTIGUOUS"]
        except (KeyError, TypeError, IndexError):
            return False
    return bool(result)


@dataclass(frozen=True, slots=True)
class Eqtset2Wsm6Carriers:
    """CM1 scalar carriers at the completed final RK stage, in C z-y-x order."""

    pi0: object
    th0: object
    rho: object
    pp3d: object
    th3d: object
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
    qpten: object
    qtten: object
    qvten: object
    qcten: object


@dataclass(frozen=True, slots=True)
class Eqtset2Wsm6StepResult:
    """Asynchronous completion token and receipts for one bridge lifecycle."""

    component_result: ArwenWsm6StepResult
    ready_event: object
    call_index: int
    stream_pointer: int
    receipt: Mapping[str, object]
    restart_arrays: Mapping[str, object]


def bridge_metadata() -> dict[str, object]:
    """Return the CUDA-free semantic and red-gate contract."""

    return {
        "schema": "gpuwm-cm1.eqtset2-wsm6-metadata/v1",
        "bridge_abi": EQTSET2_WSM6_BRIDGE_ABI,
        "component_id": ARWEN_WSM6_COMPONENT_ID,
        "claim_limit": _CLAIM_LIMIT,
        "lifecycle": [
            {
                "ordinal": 1,
                "operation": "final-RK remove and snapshot",
                "source": "src/solve2.F:1561-1579",
            },
            {
                "ordinal": 2,
                "operation": "pre-physics pressure/density rebuild",
                "source": "src/solve2.F:1582-1595",
            },
            {
                "ordinal": 3,
                "operation": "public ArWen WSM6 apply",
                "source": "src/cm1.F:3405-3432",
            },
            {
                "ordinal": 4,
                "operation": "ptype-5 eqtset-2 pressure/theta rebuild",
                "source": "src/mp_driver.F:603-617",
            },
            {
                "ordinal": 5,
                "operation": "held-rate recapture",
                "source": "src/solve3.F:319-337",
            },
        ],
        "constants": {
            "cm1": {
                "p00": CM1_P00_F32,
                "rd": CM1_RD_F32,
                "rv": CM1_RV_F32,
                "cp": CM1_CP_F32,
                "rovcp": CM1_ROVCP_F32,
                "cpdrd": CM1_CPDRD_F32,
            },
            "arwen_wsm6_staging": {"p00": ARWEN_P00_F32, "rcp": ARWEN_RCP_F32},
        },
        "restart": {
            "export_inventory": list(EQTSET2_RESTART_FIELDS),
            "import_status": "exact-pre-apply-device-binding-implemented",
            "durable_serialization": "not implemented",
        },
        "red_gates": {
            "idiss": "separate supplied-input pre-RK consumer; live dissten wiring required",
            "efall": "refused: public WSM6 exposes no terminal-velocity fields",
            "reflectivity_consumer": "implemented as an exactly-once transient diagnostic",
            "durable_restart_serialization": "external CM1W6R1 envelope",
            "product_admission": False,
        },
    }


def bridge_metadata_bytes() -> bytes:
    return _canonical_json_bytes(bridge_metadata())


def bridge_metadata_sha256() -> str:
    return hashlib.sha256(bridge_metadata_bytes()).hexdigest()


def _validate_carriers_without_cuda(
    carriers: Eqtset2Wsm6Carriers, *, nx: int, ny: int, nz: int
) -> None:
    if not isinstance(carriers, Eqtset2Wsm6Carriers):
        _refuse(
            code="CM1-EQTSET2-CARRIER-SET",
            reason="the bridge requires one complete named CM1 carrier set",
            remedy="construct Eqtset2Wsm6Carriers with every declared field",
            option="carriers",
            value=type(carriers).__name__,
        )
    volume = (nz, ny, nx)
    interface = (nz + 1, ny, nx)
    expected: dict[str, tuple[tuple[int, ...], ...]] = {
        name: (volume,) for name in _VOLUME_FIELDS
    }
    for name in _BASE_FIELDS:
        expected[name] = ((nz,), volume)
    expected["phb"] = ((nz + 1,), interface)
    expected["php"] = (interface,)
    for name in _ALL_FIELDS:
        value = getattr(carriers, name)
        actual_shape = _shape(value)
        if actual_shape not in expected[name]:
            _refuse(
                code="CM1-EQTSET2-CARRIER-SHAPE",
                reason=(
                    f"carrier {name} has shape {actual_shape}, not {expected[name]}; "
                    "accepting it would transpose or broadcast source cells"
                ),
                remedy="bind the exact C-contiguous z-y-x CM1 scalar/interface view",
                option=f"carriers.{name}.shape",
                value=actual_shape,
            )
        if not _is_f32(value):
            _refuse(
                code="CM1-EQTSET2-CARRIER-DTYPE",
                reason=f"carrier {name} is not IEEE-754 binary32",
                remedy="bind a float32 carrier without an adapter-side cast",
                option=f"carriers.{name}.dtype",
                value=str(getattr(value, "dtype", None)),
            )
        if not _is_c_contiguous(value):
            _refuse(
                code="CM1-EQTSET2-CARRIER-STRIDES",
                reason=f"carrier {name} is not C-contiguous z-y-x",
                remedy="prepare a C-contiguous device carrier before bridge apply",
                option=f"carriers.{name}.flags.c_contiguous",
                value=False,
            )
    mutable = [getattr(carriers, name) for name in _MUTABLE_FIELDS]
    if len({id(value) for value in mutable}) != len(mutable):
        _refuse(
            code="CM1-EQTSET2-CARRIER-ALIAS",
            reason=(
                "two mutable state/held arrays alias; remove/snapshot would overwrite "
                "a live prognostic or a different retained tendency"
            ),
            remedy="bind distinct mutable and held array objects",
            option="carriers.mutable_identity",
            value="aliased",
        )


def _validate_cupy_residency(cupy: object, carriers: Eqtset2Wsm6Carriers) -> int:
    ndarray = getattr(cupy, "ndarray", None)
    if not isinstance(ndarray, type):
        raise RuntimeError("installed CuPy module exposes no ndarray type")
    current_device = int(cupy.cuda.runtime.getDevice())
    for name in _ALL_FIELDS:
        value = getattr(carriers, name)
        if not isinstance(value, ndarray):
            _refuse(
                code="CM1-EQTSET2-NONCUDA-CARRIER",
                reason=f"carrier {name} is not a CuPy device array",
                remedy="bind the resident CuPy array owned by the CM1 state",
                option=f"carriers.{name}",
                value=type(value).__name__,
            )
        device = getattr(value, "device", None)
        device_id = getattr(device, "id", None)
        if device_id is None or int(device_id) != current_device:
            _refuse(
                code="CM1-EQTSET2-DEVICE-OWNERSHIP",
                reason=(
                    f"carrier {name} is not owned by current CUDA device "
                    f"{current_device}"
                ),
                remedy="make the carrier-owning device current before bridge apply",
                option=f"carriers.{name}.device",
                value=device_id,
            )
    return current_device


class Eqtset2Wsm6Bridge:
    """Persistent source-ordered device bridge around one WSM6 component."""

    def __init__(
        self,
        *,
        component: ArwenWsm6Component,
        nx: int,
        ny: int,
        nz: int,
        dt_s: float,
        idiss: bool,
        efall: bool,
        product_admission: bool,
    ) -> None:
        self.nx = _strict_int("nx", nx, 1)
        self.ny = _strict_int("ny", ny, 1)
        self.nz = _strict_int("nz", nz, 2)
        self.dt_s = _positive_finite("dt_s", dt_s)
        self.idiss = _strict_bool("idiss", idiss)
        self.efall = _strict_bool("efall", efall)
        self.product_admission = _strict_bool("product_admission", product_admission)
        component_contract = {
            "component_id": ARWEN_WSM6_COMPONENT_ID,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dt_s": self.dt_s,
        }
        actual_contract = {
            name: getattr(component, name, None) for name in component_contract
        }
        if actual_contract != component_contract:
            _refuse(
                code="CM1-EQTSET2-COMPONENT-CONTRACT",
                reason=(
                    "the persistent public WSM6 component geometry, timestep, or "
                    "semantic identity differs from the bridge contract"
                ),
                remedy="construct both objects with one exact admitted configuration",
                option="component_contract",
                value=actual_contract,
            )
        if self.idiss:
            _refuse(
                code=EQTSET2_IDISS_LIVE_REFUSAL,
                reason=(
                    "CM1 dissipative heating belongs to solve1's pre-RK cadence and "
                    "requires the live SGS/diffusion dissten z-face carrier; the "
                    "post-final-RK WSM6 bridge has no such producer input"
                ),
                remedy=(
                    "run the source-ordered Eqtset2IdissConsumer before the solve1 "
                    "held-tendency consume once a live dissten producer is composed, "
                    "and keep this post-final-RK bridge idiss=false"
                ),
                option="idiss",
                value=True,
            )
        require_efall_contract(self.efall)
        if self.product_admission:
            _refuse(
                code="CM1-EQTSET2-PRODUCT-RED",
                reason="this bounded bridge has not passed the coupled product trajectory/restart gates",
                remedy="use product_admission=false for component evidence only",
                option="product_admission",
                value=True,
            )
        self._component = component
        self._scratch: dict[str, object] = {}
        self._held: dict[str, object] | None = None
        self._stream_pointer: int | None = None
        self._device_id: int | None = None
        self._call_count = 0
        self._last_receipt: dict[str, object] | None = None
        self._pending_restart: Mapping[str, object] | None = None
        self._pending_completed_calls: int | None = None
        self._restored_completed_calls: int | None = None

    @property
    def component(self) -> ArwenWsm6Component:
        return self._component

    @property
    def call_count(self) -> int:
        return self._call_count

    def _scratch_like(self, cupy: object, name: str, like: object) -> object:
        value = self._scratch.get(name)
        if value is None:
            value = cupy.empty_like(like)
            self._scratch[name] = value
        elif _shape(value) != _shape(like) or getattr(value, "dtype", None) != getattr(
            like, "dtype", None
        ):
            raise RuntimeError(f"bridge scratch {name!r} drifted from carrier ABI")
        return value

    def _restart_arrays(self) -> dict[str, object]:
        if self._held is None:
            raise RuntimeError("eqtset2 bridge has not completed a lifecycle call")
        arrays = dict(self._held)
        component_arrays = self._component.persistent_arrays()
        if tuple(component_arrays) != WSM6_COMPONENT_PERSISTENCE:
            raise RuntimeError("public component persistence inventory drifted")
        arrays.update(
            {f"component.{name}": value for name, value in component_arrays.items()}
        )
        if tuple(arrays) != EQTSET2_RESTART_FIELDS:
            raise RuntimeError("bridge restart inventory ordering drifted")
        return arrays

    def restart_arrays(self) -> Mapping[str, object]:
        """Expose held rates plus public component persistence without copying."""

        return MappingProxyType(self._restart_arrays())

    def restore_restart_arrays(
        self, arrays: Mapping[str, object], *, completed_calls: int
    ) -> None:
        """Stage exact held/component arrays for binding before the next apply."""

        if (
            self._held is not None
            or self._call_count != 0
            or self._pending_restart is not None
        ):
            _refuse(
                code="CM1-EQTSET2-RESTORE-LIFETIME",
                reason="restart restore is only valid on one fresh bridge instance",
                remedy="construct a fresh bridge and stage one complete restart",
                option="bridge.call_count",
                value=self._call_count,
            )
        if (
            isinstance(completed_calls, bool)
            or not isinstance(completed_calls, int)
            or completed_calls < 0
        ):
            _refuse(
                code="CM1-EQTSET2-RESTORE-CADENCE",
                reason="completed_calls must be a strict nonnegative integer",
                remedy="restore the exact completed bridge-call ordinal",
                option="completed_calls",
                value=completed_calls,
            )
        if not isinstance(arrays, Mapping) or tuple(arrays) != EQTSET2_RESTART_FIELDS:
            _refuse(
                code="CM1-EQTSET2-RESTORE-INVENTORY",
                reason="restart fields or order differ from the bridge restart ABI",
                remedy="restore all held and component arrays in exact ABI order",
                option="restart_fields",
                value=list(arrays)
                if isinstance(arrays, Mapping)
                else type(arrays).__name__,
            )
        volume = (self.nz, self.ny, self.nx)
        surface = (self.ny, self.nx)
        for name in EQTSET2_RESTART_FIELDS:
            value = arrays[name]
            component_name = name.removeprefix("component.")
            expected = (
                surface
                if name.startswith("component.")
                and component_name in WSM6_SURFACE_SCRATCH
                else volume
            )
            if _shape(value) != expected:
                _refuse(
                    code="CM1-EQTSET2-RESTORE-SHAPE",
                    reason=f"restart field {name} has shape {_shape(value)}, not {expected}",
                    remedy="restore the exact bridge-domain array",
                    option=f"restart_fields.{name}.shape",
                    value=_shape(value),
                )
            if not _is_f32(value) or not _is_c_contiguous(value):
                _refuse(
                    code="CM1-EQTSET2-RESTORE-LAYOUT",
                    reason=f"restart field {name} is not C-contiguous binary32",
                    remedy="restore serialized raw words without conversion",
                    option=f"restart_fields.{name}.layout",
                    value=str(getattr(value, "dtype", None)),
                )
        if len({id(arrays[name]) for name in EQTSET2_RESTART_FIELDS}) != len(
            EQTSET2_RESTART_FIELDS
        ):
            _refuse(
                code="CM1-EQTSET2-RESTORE-ALIAS",
                reason="two bridge restart fields share one array object",
                remedy="restore distinct held and component storage",
                option="restart_fields.identity",
                value="aliased",
            )
        self._pending_restart = MappingProxyType(dict(arrays))
        self._pending_completed_calls = completed_calls

    def receipt_dict(self) -> dict[str, object]:
        if self._last_receipt is None:
            raise RuntimeError("eqtset2 bridge has not completed a lifecycle call")
        return json.loads(_canonical_json_bytes(self._last_receipt))

    def receipt_bytes(self) -> bytes:
        return _canonical_json_bytes(self.receipt_dict())

    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_bytes()).hexdigest()

    def apply(
        self,
        carriers: Eqtset2Wsm6Carriers,
        *,
        final_rk: bool,
        refl_10cm_due: bool = False,
        pre_component_hook: Callable[[Eqtset2Wsm6Carriers], object] | None = None,
        post_component_pre_recapture_hook: (
            Callable[[Eqtset2Wsm6Carriers], object] | None
        ) = None,
    ) -> Eqtset2Wsm6StepResult:
        """Enqueue one complete final-RK bridge lifecycle on the current stream."""

        final_rk = _strict_bool("final_rk", final_rk)
        if not final_rk:
            _refuse(
                code="CM1-EQTSET2-CADENCE",
                reason="remove/snapshot and physics are final-RK-only in CM1 r21.1",
                remedy="call this bridge only after nrk==nrkmax",
                option="final_rk",
                value=False,
            )
        refl_10cm_due = _strict_bool("refl_10cm_due", refl_10cm_due)
        _validate_carriers_without_cuda(carriers, nx=self.nx, ny=self.ny, nz=self.nz)
        inspection = engine_pin.inspect()
        if not inspection.admitted:
            _refuse(
                code="CM1-ARWEN-ENGINE-DRIFT",
                reason="the installed ArWen WSM6 execution surface differs from the audited estate",
                remedy="install the measured ArWen estate or audit and version a new bridge ABI",
                option="engine_inspection",
                value=inspection.as_dict(),
            )

        cupy = importlib.import_module("cupy")
        _validate_carriers_without_cuda(carriers, nx=self.nx, ny=self.ny, nz=self.nz)
        device_id = _validate_cupy_residency(cupy, carriers)
        stream = cupy.cuda.get_current_stream()
        stream_pointer = int(stream.ptr)
        if self._device_id is not None and self._device_id != device_id:
            _refuse(
                code="CM1-EQTSET2-PERSISTENCE-DEVICE",
                reason="bridge persistence cannot move between CUDA devices",
                remedy="keep one bridge on one device",
                option="device_id",
                value=device_id,
            )
        if self._stream_pointer is not None and self._stream_pointer != stream_pointer:
            _refuse(
                code="CM1-EQTSET2-PERSISTENCE-STREAM",
                reason="bridge persistence and held arrays are ordered on a different CUDA stream",
                remedy="keep one bridge on one stream or establish a new bridge after an explicit dependency",
                option="stream_pointer",
                value=stream_pointer,
            )
        if self._held is not None and any(
            self._held[name] is not getattr(carriers, name)
            for name in EQTSET2_HELD_FIELDS
        ):
            _refuse(
                code="CM1-EQTSET2-HELD-REBIND",
                reason="held-tendency object identity changed within a persistent bridge",
                remedy="retain the same four device arrays for the bridge lifetime",
                option="held_identity",
                value="changed",
            )
        if self._pending_restart is not None and any(
            self._pending_restart[name] is not getattr(carriers, name)
            for name in EQTSET2_HELD_FIELDS
        ):
            _refuse(
                code="CM1-EQTSET2-RESTORE-HELD-IDENTITY",
                reason="restored held arrays are not the carrier objects bound for apply",
                remedy="construct carriers with the exact staged held restart arrays",
                option="restart_fields.held_identity",
                value="changed",
            )

        dt = cupy.float32(self.dt_s)
        cm1_p00 = cupy.float32(CM1_P00_F32)
        cm1_rp00 = cupy.float32(CM1_RP00_F32)
        cm1_rd = cupy.float32(CM1_RD_F32)
        cm1_rv = cupy.float32(CM1_RV_F32)
        cm1_rovcp = cupy.float32(CM1_ROVCP_F32)
        cm1_cpdrd = cupy.float32(CM1_CPDRD_F32)
        arwen_rp00 = cupy.float32(1.0 / ARWEN_P00_F32)
        arwen_rcp = cupy.float32(ARWEN_RCP_F32)
        zero = cupy.float32(0.0)

        # The corrected Rust product retains CM1's native one-dimensional
        # base profiles.  Reshaping to z-1-1 is an object view only; CuPy then
        # performs the same per-level broadcast as the Fortran scalar loops.
        pi0_zyx = (
            carriers.pi0.reshape((self.nz, 1, 1))
            if len(carriers.pi0.shape) == 1
            else carriers.pi0
        )
        th0_zyx = (
            carriers.th0.reshape((self.nz, 1, 1))
            if len(carriers.th0.shape) == 1
            else carriers.th0
        )

        work = self._scratch_like(cupy, "work", carriers.pp3d)
        cm1_pi = self._scratch_like(cupy, "cm1_pi", carriers.pp3d)
        arwen_pii = self._scratch_like(cupy, "arwen_pii", carriers.pp3d)
        temperature = self._scratch_like(cupy, "temperature", carriers.pp3d)
        arwen_thp = self._scratch_like(cupy, "arwen_thp", carriers.pp3d)

        if self._pending_restart is not None:
            component_restore_carriers = ArwenWsm6Carriers(
                thp=arwen_thp,
                thb=carriers.th0,
                p=carriers.p,
                alt=carriers.alt,
                phb=carriers.phb,
                php=carriers.php,
                qv=carriers.qv,
                qc=carriers.qc,
                qr=carriers.qr,
                qi=carriers.qi,
                qs=carriers.qs,
                qg=carriers.qg,
            )
            component_restart = {
                name: self._pending_restart[f"component.{name}"]
                for name in WSM6_COMPONENT_PERSISTENCE
            }
            completed_calls = self._pending_completed_calls
            if completed_calls is None:
                raise RuntimeError("staged bridge restart lost completed call count")
            self._component.restore_persistent_arrays(
                component_restore_carriers,
                component_restart,
                completed_calls=completed_calls,
            )
            self._held = {
                name: self._pending_restart[name] for name in EQTSET2_HELD_FIELDS
            }
            self._call_count = completed_calls
            self._restored_completed_calls = completed_calls
            self._pending_restart = None
            self._pending_completed_calls = None

        # solve2.F:1561-1579 -- each multiply and subtract remains a separate
        # binary32 device operation, preventing an unmeasured FMA contraction.
        for state_name, held_name in (
            ("pp3d", "qpten"),
            ("th3d", "qtten"),
            ("qv", "qvten"),
            ("qc", "qcten"),
        ):
            state = getattr(carriers, state_name)
            held = getattr(carriers, held_name)
            cupy.multiply(held, dt, out=work)
            cupy.subtract(state, work, out=state)
            cupy.copyto(held, state)

        # solve2.F:1582-1595 -- pre-physics EOS.  temperature retains CM1's
        # physical T while arwen_pii expresses the same T in WSM6's WRF theta
        # convention; the scheme never receives a mixed-constant temperature.
        cupy.add(pi0_zyx, carriers.pp3d, out=cm1_pi)
        cupy.power(cm1_pi, cm1_cpdrd, out=carriers.p)
        cupy.multiply(carriers.p, cm1_p00, out=carriers.p)
        cupy.add(th0_zyx, carriers.th3d, out=work)
        cupy.multiply(work, cm1_pi, out=temperature)
        cupy.maximum(carriers.qv, zero, out=work)
        cupy.multiply(work, cm1_rv, out=work)
        cupy.add(work, cm1_rd, out=work)
        cupy.multiply(temperature, work, out=work)
        cupy.divide(carriers.p, work, out=carriers.rho)
        cupy.reciprocal(carriers.rho, out=carriers.alt)

        # solve2's selected final-stage dry-mass correction is ordered after
        # the remove/snapshot EOS above and before mp_driver.  GM1 supplies
        # that already-admitted device operation through this narrow hook.
        # Because apmasscon mutates pp3d, pressure, and rho, rebuild every
        # pressure-dependent staging value before the public component call.
        _pre_component_result = None
        if pre_component_hook is not None:
            if not callable(pre_component_hook):
                raise TypeError("pre_component_hook must be callable or None")
            _pre_component_result = pre_component_hook(carriers)
            cupy.add(pi0_zyx, carriers.pp3d, out=cm1_pi)
            cupy.add(th0_zyx, carriers.th3d, out=work)
            cupy.multiply(work, cm1_pi, out=temperature)
            cupy.reciprocal(carriers.rho, out=carriers.alt)

        cupy.multiply(carriers.p, arwen_rp00, out=arwen_pii)
        cupy.power(arwen_pii, arwen_rcp, out=arwen_pii)
        cupy.divide(temperature, arwen_pii, out=arwen_thp)
        cupy.subtract(arwen_thp, th0_zyx, out=arwen_thp)

        component_carriers = ArwenWsm6Carriers(
            thp=arwen_thp,
            thb=carriers.th0,
            p=carriers.p,
            alt=carriers.alt,
            phb=carriers.phb,
            php=carriers.php,
            qv=carriers.qv,
            qc=carriers.qc,
            qr=carriers.qr,
            qi=carriers.qi,
            qs=carriers.qs,
            qg=carriers.qg,
        )
        component_result = self._component.apply(
            component_carriers, refl_10cm_due=refl_10cm_due
        )
        if int(component_result.stream_pointer) != stream_pointer:
            raise RuntimeError("public component returned a different CUDA stream")

        # mp_driver.F:603-617 -- recover post-WSM6 temperature in ArWen's
        # pinned theta convention, then rebuild CM1 pressure, Exner, and theta
        # with CM1 constants while rho remains fixed across physics.
        cupy.add(th0_zyx, arwen_thp, out=work)
        cupy.multiply(work, arwen_pii, out=temperature)
        cupy.multiply(carriers.qv, cm1_rv, out=work)
        cupy.add(work, cm1_rd, out=work)
        cupy.multiply(carriers.rho, work, out=work)
        cupy.multiply(work, temperature, out=carriers.p)
        cupy.multiply(carriers.p, cm1_rp00, out=work)
        cupy.power(work, cm1_rovcp, out=carriers.pp3d)
        cupy.subtract(carriers.pp3d, pi0_zyx, out=carriers.pp3d)
        cupy.add(pi0_zyx, carriers.pp3d, out=work)
        cupy.divide(temperature, work, out=carriers.th3d)
        cupy.subtract(carriers.th3d, th0_zyx, out=carriers.th3d)

        # solve3.F:270-312 precedes the held-rate recapture at 319-337.  The
        # selected child supplies ppx completion plus pressure/rho boundary
        # publication through this exact post-component seam.
        _post_component_pre_recapture_result = None
        if post_component_pre_recapture_hook is not None:
            if not callable(post_component_pre_recapture_hook):
                raise TypeError(
                    "post_component_pre_recapture_hook must be callable or None"
                )
            _post_component_pre_recapture_result = post_component_pre_recapture_hook(
                carriers
            )

        # solve3.F:319-337 -- snapshots become next-step rates.
        for state_name, held_name in (
            ("pp3d", "qpten"),
            ("th3d", "qtten"),
            ("qv", "qvten"),
            ("qc", "qcten"),
        ):
            state = getattr(carriers, state_name)
            held = getattr(carriers, held_name)
            cupy.subtract(state, held, out=work)
            cupy.divide(work, dt, out=held)

        ready_event = cupy.cuda.Event(disable_timing=True)
        ready_event.record(stream)
        self._held = {name: getattr(carriers, name) for name in EQTSET2_HELD_FIELDS}
        self._stream_pointer = stream_pointer
        self._device_id = device_id
        self._call_count += 1

        restart_arrays = self._restart_arrays()
        bridge_source = Path(__file__).read_bytes()
        component_receipt = self._component.receipt_dict()
        receipt: dict[str, object] = {
            "schema": EQTSET2_WSM6_RECEIPT_SCHEMA,
            "bridge_abi": EQTSET2_WSM6_BRIDGE_ABI,
            "bridge_metadata_sha256": bridge_metadata_sha256(),
            "claim_limit": _CLAIM_LIMIT,
            "call_index": self._call_count,
            "configuration": {
                "nx": self.nx,
                "ny": self.ny,
                "nz": self.nz,
                "dt_s": self.dt_s,
                "equation_set": 2,
                "native_ptype_context": 5,
                "semantic_component_id": ARWEN_WSM6_COMPONENT_ID,
                "idiss": self.idiss,
                "efall": self.efall,
                "refl_10cm_due": refl_10cm_due,
                "product_admission": False,
            },
            "source": {
                "module": __name__,
                "bytes": len(bridge_source),
                "sha256": hashlib.sha256(bridge_source).hexdigest(),
                "anchors": [item["source"] for item in bridge_metadata()["lifecycle"]],
            },
            "lifecycle": {
                "final_rk_required": True,
                "operation_ordinals": [1, 2, 3, 4, 5],
                "pre_component_hook": pre_component_hook is not None,
                "post_component_pre_recapture_hook": (
                    post_component_pre_recapture_hook is not None
                ),
                "temperature_staging": (
                    "CM1 theta*pi -> physical T -> ArWen theta; returned ArWen "
                    "theta*pi -> physical T -> CM1 ptype-5 EOS"
                ),
                "rho_fixed_across_component": True,
                "held_fields": list(EQTSET2_HELD_FIELDS),
                "reflectivity_lifetime": (
                    "public WSM6 output-due field consumed exactly once inside the "
                    "component call and excluded from restart state"
                ),
            },
            "component": {
                "receipt_sha256": self._component.receipt_sha256(),
                "receipt": component_receipt,
            },
            "restart": {
                "schema": EQTSET2_WSM6_RESTART_SCHEMA,
                "export_zero_copy": True,
                "inventory": list(restart_arrays),
                "shapes": {
                    name: list(getattr(value, "shape"))
                    for name, value in restart_arrays.items()
                },
                "import_status": "exact-pre-apply-device-binding-implemented",
                "restored_before_apply": self._restored_completed_calls is not None,
                "restored_completed_calls": self._restored_completed_calls,
                "durable_serialization": False,
            },
            "synchronization": {
                "device_id": device_id,
                "stream_pointer": stream_pointer,
                "component_ready_event_precedes_bridge_rebuild": True,
                "bridge_ready_event_recorded": True,
                "explicit_synchronize": False,
                "host_state_copy": False,
            },
            "red_gates": {
                "idiss": True,
                "efall": True,
                "reflectivity_consumer": False,
                "restart_import": False,
                "durable_restart_serialization": True,
                "coupled_trajectory": True,
                "product_admission": True,
            },
        }
        _canonical_json_bytes(receipt)
        self._last_receipt = receipt
        detached = self.receipt_dict()
        return Eqtset2Wsm6StepResult(
            component_result=component_result,
            ready_event=ready_event,
            call_index=self._call_count,
            stream_pointer=stream_pointer,
            receipt=MappingProxyType(detached),
            restart_arrays=MappingProxyType(restart_arrays),
        )


def create_eqtset2_wsm6_bridge(
    *,
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    dt_s: float,
    equation_set: int = 2,
    native_ptype_context: int = 5,
    idiss: bool = False,
    efall: bool = False,
    product_admission: bool = False,
) -> Eqtset2Wsm6Bridge:
    """Create a validated bridge and still leave both numerical runtimes cold."""

    if isinstance(equation_set, bool) or equation_set != 2:
        _refuse(
            code="CM1-EQTSET2-SELECTION",
            reason="this bridge implements only CM1 equation_set=2",
            remedy="select equation_set=2 or use the distinct dry eqtset=1 lane",
            option="equation_set",
            value=equation_set,
        )
    if isinstance(native_ptype_context, bool) or native_ptype_context != 5:
        _refuse(
            code="CM1-EQTSET2-PTYPE-CONTEXT",
            reason=(
                "the audited post-physics rebuild is the ptype=5 native context; "
                "this integer is provenance and is never an ArWen selector"
            ),
            remedy="use native_ptype_context=5 with semantic component wsm6-mp6",
            option="native_ptype_context",
            value=native_ptype_context,
        )
    component = create_arwen_wsm6_component(
        component_id=ARWEN_WSM6_COMPONENT_ID,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
        dt_s=dt_s,
    )
    return Eqtset2Wsm6Bridge(
        component=component,
        nx=nx,
        ny=ny,
        nz=nz,
        dt_s=dt_s,
        idiss=idiss,
        efall=efall,
        product_admission=product_admission,
    )


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


_POWF: object | None = None


def _powf(base: float, exponent: float) -> float:
    global _POWF
    if _POWF is None:
        candidates = ["ucrtbase"] if hasattr(ctypes, "WinDLL") else []
        library = ctypes.util.find_library("m")
        if library:
            candidates.append(library)
        candidates.append(None)
        for candidate in candidates:
            try:
                handle = ctypes.CDLL(candidate) if candidate else ctypes.CDLL(None)
                function = handle.powf
            except (OSError, AttributeError):
                continue
            function.argtypes = [ctypes.c_float, ctypes.c_float]
            function.restype = ctypes.c_float
            _POWF = function
            break
        else:
            raise RuntimeError("platform C runtime exposes no binary32 powf")
    return _f32(_POWF(ctypes.c_float(base), ctypes.c_float(exponent)))


def _same_length(*values: tuple[float, ...]) -> None:
    if not values or len({len(value) for value in values}) != 1 or not values[0]:
        raise ValueError("CPU lifecycle vectors must be nonempty and equal length")


def _tuple_f32(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(_f32(value) for value in values)


def cpu_consume_prior_f32(
    base: Iterable[float], held: Iterable[float]
) -> tuple[float, ...]:
    """Default-real transcription of one solve1 held-rate consume."""

    left = _tuple_f32(base)
    right = _tuple_f32(held)
    _same_length(left, right)
    return tuple(_f32(a + b) for a, b in zip(left, right, strict=True))


def cpu_remove_snapshot_f32(
    state: Iterable[float], held: Iterable[float], dt_s: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Default-real transcription of final-RK remove plus snapshot."""

    current = _tuple_f32(state)
    estimate = _tuple_f32(held)
    _same_length(current, estimate)
    dt = _f32(dt_s)
    removed = tuple(
        _f32(value - _f32(dt * rate))
        for value, rate in zip(current, estimate, strict=True)
    )
    return removed, removed


def cpu_ptype5_rebuild_f32(
    *,
    rho: Iterable[float],
    temperature: Iterable[float],
    qv: Iterable[float],
    pi0: Iterable[float],
    th0: Iterable[float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Default-real ptype-5 eqtset-2 EOS rebuild used after physics."""

    rho_v = _tuple_f32(rho)
    temp_v = _tuple_f32(temperature)
    qv_v = _tuple_f32(qv)
    pi0_v = _tuple_f32(pi0)
    th0_v = _tuple_f32(th0)
    _same_length(rho_v, temp_v, qv_v, pi0_v, th0_v)
    rd = _f32(CM1_RD_F32)
    rv = _f32(CM1_RV_F32)
    rp00 = _f32(CM1_RP00_F32)
    rovcp = _f32(CM1_ROVCP_F32)
    pressure: list[float] = []
    pp3d: list[float] = []
    th3d: list[float] = []
    for density, temp, vapor, base_pi, base_theta in zip(
        rho_v, temp_v, qv_v, pi0_v, th0_v, strict=True
    ):
        gas = _f32(rd + _f32(rv * vapor))
        prs = _f32(_f32(density * gas) * temp)
        pp = _f32(_powf(_f32(prs * rp00), rovcp) - base_pi)
        theta = _f32(_f32(temp / _f32(base_pi + pp)) - base_theta)
        pressure.append(prs)
        pp3d.append(pp)
        th3d.append(theta)
    return tuple(pressure), tuple(pp3d), tuple(th3d)


def cpu_recapture_f32(
    state: Iterable[float], snapshot: Iterable[float], dt_s: float
) -> tuple[float, ...]:
    """Default-real transcription of solve3 held-rate recapture."""

    current = _tuple_f32(state)
    prior = _tuple_f32(snapshot)
    _same_length(current, prior)
    dt = _f32(dt_s)
    return tuple(
        _f32(_f32(value - old) / dt) for value, old in zip(current, prior, strict=True)
    )


__all__ = [
    "ARWEN_RCP_F32",
    "CM1_CPDRD_F32",
    "CM1_CP_F32",
    "CM1_P00_F32",
    "CM1_RD_F32",
    "CM1_ROVCP_F32",
    "CM1_RV_F32",
    "EQTSET2_HELD_FIELDS",
    "EQTSET2_RESTART_FIELDS",
    "EQTSET2_WSM6_BRIDGE_ABI",
    "EQTSET2_WSM6_RECEIPT_SCHEMA",
    "EQTSET2_WSM6_RESTART_SCHEMA",
    "Eqtset2Wsm6Bridge",
    "Eqtset2Wsm6Carriers",
    "Eqtset2Wsm6StepResult",
    "bridge_metadata",
    "bridge_metadata_bytes",
    "bridge_metadata_sha256",
    "cpu_consume_prior_f32",
    "cpu_ptype5_rebuild_f32",
    "cpu_recapture_f32",
    "cpu_remove_snapshot_f32",
    "create_eqtset2_wsm6_bridge",
]
