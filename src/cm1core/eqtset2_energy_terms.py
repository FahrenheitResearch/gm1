"""Bounded CM1 r21.1 equation-set-2 energy-term contracts.

The production path in this module is a supplied-input CUDA consumer.  It
does not manufacture SGS dissipation and it is deliberately separate from
the post-final-RK WSM6 transition: CM1 applies dissipative heating in the
pre-RK ``solve1`` cadence before it consumes the prior microphysics rates.

CM1's optional fallout-energy term cannot currently be connected to public
ArWen WSM6.  The upstream formula consumes one terminal-velocity field for
each condensed species, while ArWen's public WSM6 result exposes no such
fields.  ``require_efall_contract`` therefore admits the fixed real-product
selection ``efall=False`` and gives ``efall=True`` one stable named refusal.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, NoReturn

from .errors import refuse


EQTSET2_IDISS_CONSUMER_ABI = "gpuwm-cm1.eqtset2-idiss-consumer/v1"
EQTSET2_IDISS_RECEIPT_SCHEMA = "gpuwm-cm1.eqtset2-idiss-consumer-receipt/v1"
EQTSET2_ENERGY_METADATA_SCHEMA = "gpuwm-cm1.eqtset2-energy-terms/v1"
EQTSET2_EFALL_REFUSAL = "CM1-EQTSET2-EFALL-PUBLIC-VT-MISSING"
EQTSET2_IDISS_LIVE_REFUSAL = "CM1-EQTSET2-IDISS-LIVE-DISSTEN-MISSING"
EQTSET2_SPECIES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_bits(value: float) -> str:
    return struct.pack("<f", _f32(value)).hex()


# constants.F:111-148, evaluated in the source assignment order as default REAL.
CM1_G = _f32(9.81)
CM1_RD = _f32(287.04)
CM1_CP = _f32(1005.7)
CM1_CPV = _f32(1870.0)
CM1_RV = _f32(461.5)
CM1_CV = _f32(CM1_CP - CM1_RD)
CM1_CVV = _f32(CM1_CPV - CM1_RV)
CM1_CPDCV = _f32(CM1_CP / CM1_CV)
CM1_CPL = _f32(4190.0)
CM1_CPI = _f32(2106.0)

_IDISS_FIELDS: tuple[str, ...] = (
    "thten1",
    "dissten",
    "pi0",
    "ppi",
) + EQTSET2_SPECIES
_CLAIM_LIMIT = (
    "source-ordered CM1 r21.1 moist eqtset=2 idiss heating from an explicitly "
    "supplied live dissten z-face field; no SGS producer, held-tendency consume, "
    "WSM6 call, efall=1, coupled step, trajectory, forecast, or validation claim"
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _refuse(*, code: str, reason: str, remedy: str, option: str, value: object) -> NoReturn:
    refuse(
        code=code,
        gate="CM1-eqtset2-energy-terms",
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )


def require_efall_contract(efall: object) -> bool:
    """Admit the official real-product ``efall=0`` selection only."""

    if type(efall) is not bool:
        _refuse(
            code="CM1-EQTSET2-EFALL-BOOLEAN",
            reason="efall must be a strict boolean fixed-profile selection",
            remedy="pass the exact profile value false",
            option="efall",
            value=efall,
        )
    if efall:
        _refuse(
            code=EQTSET2_EFALL_REFUSAL,
            reason=(
                "CM1 r21.1 mp_driver.F:570-600 and kessler.F:485-522 require "
                "qc/qr/qi/qs/qg terminal-velocity fields, but public ArWen WSM6 "
                "does not return them; deriving or substituting them would change "
                "the fallout-energy equation"
            ),
            remedy=(
                "keep the sealed real-product efall=false selection, or version a "
                "future public ArWen terminal-velocity output contract and native oracle"
            ),
            option="efall",
            value=True,
        )
    return False


def energy_terms_metadata() -> dict[str, object]:
    return {
        "schema": EQTSET2_ENERGY_METADATA_SCHEMA,
        "idiss_consumer_abi": EQTSET2_IDISS_CONSUMER_ABI,
        "claim_limit": _CLAIM_LIMIT,
        "source": {
            "constants": "src/constants.F:111-148",
            "heat_capacity": "src/solve1.F:744-848",
            "idiss": "src/solve1.F:917-975",
            "idiss_interior_first_level": "src/solve1.F:932-935",
            "efall_dispatch": "src/mp_driver.F:570-600",
            "efall_equation": "src/kessler.F:485-522",
        },
        "cadence": {
            "idiss": "pre-RK solve1 before prior microphysics tendency consume",
            "efall": "inside microphysics driver after scheme and before ptype-5 EOS rebuild",
        },
        "selection": {
            "equation_set": 2,
            "imoist": 1,
            "iice": 1,
            "bottom_boundary": [1, 2, 3],
            "interior_first_level": {"1": 1, "2": 1, "3": 2},
            "species_order": list(EQTSET2_SPECIES),
            "liquid_sum_order": ["qc", "qr"],
            "ice_sum_order": ["qi", "qs", "qg"],
        },
        "constants_f32_bits": {
            name: _f32_bits(value)
            for name, value in (
                ("g", CM1_G),
                ("rd", CM1_RD),
                ("cp", CM1_CP),
                ("cpv", CM1_CPV),
                ("rv", CM1_RV),
                ("cv", CM1_CV),
                ("cvv", CM1_CVV),
                ("cpdcv", CM1_CPDCV),
                ("cpl", CM1_CPL),
                ("cpi", CM1_CPI),
            )
        },
        "efall": {
            "real_product_selection": False,
            "true_selection": "refused until public terminal-velocity contract exists",
            "refusal_code": EQTSET2_EFALL_REFUSAL,
            "required_terminal_velocity_order": ["qc", "qr", "qi", "qs", "qg"],
        },
    }


def energy_terms_metadata_bytes() -> bytes:
    return _canonical_json_bytes(energy_terms_metadata())


def energy_terms_metadata_sha256() -> str:
    return hashlib.sha256(energy_terms_metadata_bytes()).hexdigest()


def cpu_idiss_cell_f32(
    *,
    thten1: float,
    dissten_lower: float,
    dissten_upper: float,
    pi0: float,
    ppi: float,
    qv: float,
    qc: float,
    qr: float,
    qi: float,
    qs: float,
    qg: float,
) -> Mapping[str, float]:
    """Default-REAL scalar mirror of solve1.F:744-848,937-949."""

    ql = _f32(_f32(qc) + _f32(qr))
    qice = _f32(_f32(qi) + _f32(qs))
    qice = _f32(qice + _f32(qg))
    cvm = _f32(CM1_CV + _f32(CM1_CVV * _f32(qv)))
    cvm = _f32(cvm + _f32(CM1_CPL * ql))
    cvm = _f32(cvm + _f32(CM1_CPI * qice))
    epsd = _f32(_f32(dissten_lower) + _f32(dissten_upper))
    epsd = _f32(_f32(0.5) * epsd)
    total_pi = _f32(_f32(pi0) + _f32(ppi))
    denominator = _f32(CM1_CPDCV * cvm)
    denominator = _f32(denominator * total_pi)
    dheat = _f32(epsd / denominator)
    updated = _f32(_f32(thten1) + dheat)
    return MappingProxyType(
        {
            "ql": ql,
            "qice": qice,
            "cvm": cvm,
            "epsd": epsd,
            "dheat": dheat,
            "thten1": updated,
        }
    )


@dataclass(frozen=True, slots=True)
class Eqtset2IdissCarriers:
    thten1: object
    dissten: object
    pi0: object
    ppi: object
    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object


@dataclass(frozen=True, slots=True)
class Eqtset2IdissResult:
    ready_event: object
    call_index: int
    stream_pointer: int
    receipt: Mapping[str, object]


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


def _is_c(value: object) -> bool:
    flags = getattr(value, "flags", None)
    return bool(flags is not None and getattr(flags, "c_contiguous", False))


class Eqtset2IdissConsumer:
    """Persistent scratch owner for the supplied-live-dissten CUDA consumer."""

    def __init__(self, *, nx: int, ny: int, nz: int, bottom_boundary: int = 1):
        for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _refuse(
                    code="CM1-EQTSET2-IDISS-DIMENSION",
                    reason=f"{name} must be a strict positive integer",
                    remedy="bind the exact physical-domain dimensions",
                    option=name,
                    value=value,
                )
        # src/solve1.F:917-988.  This consumer is the interior loop
        # `do k=k1,nk`, whose arithmetic is the same for every lower
        # boundary; bbc only moves `k1`, and CM1's own bbc=3 branch heats
        # k=1 from the surface-layer dissipation rate instead
        # (src/solve1.F:932-935, 950-962), which
        # cm1core.eqtset2_surface_dissipation owns and the caller drives by
        # handing this consumer the k=2..nk slice.  The boundary is recorded
        # so the receipt names the one the run integrated.
        if (
            isinstance(bottom_boundary, bool)
            or not isinstance(bottom_boundary, int)
            or bottom_boundary not in (1, 2, 3)
        ):
            _refuse(
                code="CM1-EQTSET2-IDISS-BBC",
                reason=(
                    "CM1 has exactly three wind lower boundary conditions "
                    "(src/param.F:1518-1532); any other value would make this "
                    "heating receipt name a lower boundary the run cannot have "
                    "integrated, so a reader could not tell which levels the "
                    "interior dissten average actually covered"
                ),
                remedy="record bottom_boundary 1 (free slip), 2 (no slip) or 3 (semi slip)",
                option="bottom_boundary",
                value=bottom_boundary,
            )
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.bottom_boundary = bottom_boundary
        self._scratch: dict[str, object] = {}
        self._device_id: int | None = None
        self._stream_pointer: int | None = None
        self._call_count = 0
        self._last_receipt: dict[str, object] | None = None

    def _receipt_configuration(self) -> dict[str, object]:
        """What this consumer integrated, including which levels it covered.

        ``src/solve1.F:932-935`` starts the interior loop at ``k=2`` for the
        semi-slip bottom, so the boundary alone does not say which levels the
        supplied ``dissten`` average reached; both are recorded.
        """

        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "bottom_boundary": self.bottom_boundary,
            "interior_first_level": 2 if self.bottom_boundary == 3 else 1,
            "equation_set": 2,
            "imoist": 1,
            "iice": 1,
            "species_order": list(EQTSET2_SPECIES),
        }

    def _validate(self, carriers: Eqtset2IdissCarriers) -> None:
        if not isinstance(carriers, Eqtset2IdissCarriers):
            _refuse(
                code="CM1-EQTSET2-IDISS-CARRIERS",
                reason="idiss requires one complete named supplied-input carrier set",
                remedy="construct Eqtset2IdissCarriers with all ten fields",
                option="carriers",
                value=type(carriers).__name__,
            )
        volume = (self.nz, self.ny, self.nx)
        expected = {name: volume for name in _IDISS_FIELDS}
        expected["dissten"] = (self.nz + 1, self.ny, self.nx)
        expected["pi0"] = ((self.nz,), volume)
        for name in _IDISS_FIELDS:
            value = getattr(carriers, name)
            shapes = expected[name]
            admitted = shapes if isinstance(shapes[0], tuple) else (shapes,)
            if _shape(value) not in admitted:
                _refuse(
                    code="CM1-EQTSET2-IDISS-SHAPE",
                    reason=f"{name} has shape {_shape(value)}, not {admitted}",
                    remedy="bind exact C-contiguous z-y-x mass/z-face carriers",
                    option=f"carriers.{name}.shape",
                    value=_shape(value),
                )
            if not _is_f32(value) or not _is_c(value):
                _refuse(
                    code="CM1-EQTSET2-IDISS-LAYOUT",
                    reason=f"{name} is not C-contiguous IEEE-754 binary32",
                    remedy="bind raw binary32 device storage without conversion",
                    option=f"carriers.{name}.layout",
                    value=str(getattr(value, "dtype", None)),
                )
        if any(carriers.thten1 is getattr(carriers, name) for name in _IDISS_FIELDS[1:]):
            _refuse(
                code="CM1-EQTSET2-IDISS-ALIAS",
                reason="mutable thten1 aliases an idiss input field",
                remedy="bind distinct tendency and input arrays",
                option="carriers.thten1.identity",
                value="aliased",
            )

    def apply(self, carriers: Eqtset2IdissCarriers) -> Eqtset2IdissResult:
        # One validation per call: it inspects shapes, layouts and aliases and
        # launches nothing, and it ran twice back to back on every closure tile.
        self._validate(carriers)
        cupy = importlib.import_module("cupy")
        ndarray = getattr(cupy, "ndarray", None)
        current_device = int(cupy.cuda.runtime.getDevice())
        for name in _IDISS_FIELDS:
            value = getattr(carriers, name)
            if not isinstance(ndarray, type) or not isinstance(value, ndarray):
                _refuse(
                    code="CM1-EQTSET2-IDISS-NONCUDA",
                    reason=f"{name} is not a CuPy device array",
                    remedy="bind the live CM1 CUDA carrier",
                    option=f"carriers.{name}",
                    value=type(value).__name__,
                )
            if int(value.device.id) != current_device:
                _refuse(
                    code="CM1-EQTSET2-IDISS-DEVICE",
                    reason=f"{name} is not on current CUDA device {current_device}",
                    remedy="make the carrier-owning device current",
                    option=f"carriers.{name}.device",
                    value=int(value.device.id),
                )
        stream = cupy.cuda.get_current_stream()
        stream_pointer = int(stream.ptr)
        if self._device_id not in (None, current_device) or self._stream_pointer not in (
            None,
            stream_pointer,
        ):
            _refuse(
                code="CM1-EQTSET2-IDISS-OWNERSHIP",
                reason="persistent idiss scratch cannot move across device or stream",
                remedy="retain one consumer on one CUDA stream",
                option="device_stream",
                value=[current_device, stream_pointer],
            )

        def scratch(name: str) -> object:
            value = self._scratch.get(name)
            if value is None:
                value = cupy.empty_like(carriers.thten1)
                self._scratch[name] = value
            return value

        ql = scratch("ql")
        qice = scratch("qice")
        cvm = scratch("cvm")
        work = scratch("work")
        epsd = scratch("epsd")
        pi0 = (
            carriers.pi0.reshape((self.nz, 1, 1))
            if len(carriers.pi0.shape) == 1
            else carriers.pi0
        )
        cv = cupy.float32(CM1_CV)
        cvv = cupy.float32(CM1_CVV)
        cpl = cupy.float32(CM1_CPL)
        cpi = cupy.float32(CM1_CPI)
        cpdcv = cupy.float32(CM1_CPDCV)
        half = cupy.float32(0.5)

        # solve1.F:750-834, exact species and accumulation order.
        cupy.copyto(ql, carriers.qc)
        cupy.add(ql, carriers.qr, out=ql)
        cupy.copyto(qice, carriers.qi)
        cupy.add(qice, carriers.qs, out=qice)
        cupy.add(qice, carriers.qg, out=qice)
        cupy.multiply(carriers.qv, cvv, out=cvm)
        cupy.add(cvm, cv, out=cvm)
        cupy.multiply(ql, cpl, out=work)
        cupy.add(cvm, work, out=cvm)
        cupy.multiply(qice, cpi, out=work)
        cupy.add(cvm, work, out=cvm)

        # solve1.F:937-949.  Each operation is a separate binary32 device op.
        cupy.add(carriers.dissten[:-1], carriers.dissten[1:], out=epsd)
        cupy.multiply(epsd, half, out=epsd)
        cupy.add(pi0, carriers.ppi, out=work)
        cupy.multiply(cvm, cpdcv, out=cvm)
        cupy.multiply(cvm, work, out=cvm)
        cupy.divide(epsd, cvm, out=epsd)
        cupy.add(carriers.thten1, epsd, out=carriers.thten1)

        ready = cupy.cuda.Event(disable_timing=True)
        ready.record(stream)
        self._device_id = current_device
        self._stream_pointer = stream_pointer
        self._call_count += 1
        source = Path(__file__).read_bytes()
        receipt = {
            "schema": EQTSET2_IDISS_RECEIPT_SCHEMA,
            "consumer_abi": EQTSET2_IDISS_CONSUMER_ABI,
            "metadata_sha256": energy_terms_metadata_sha256(),
            "claim_limit": _CLAIM_LIMIT,
            "call_index": self._call_count,
            "configuration": self._receipt_configuration(),
            "source": {
                "module": __name__,
                "bytes": len(source),
                "sha256": hashlib.sha256(source).hexdigest(),
                "anchors": ["src/solve1.F:744-848", "src/solve1.F:917-975"],
            },
            "binding": {
                "supplied_live_dissten": True,
                "dissten_shape": list(carriers.dissten.shape),
                "thten1_mutated_in_place": True,
                "six_species_order": list(EQTSET2_SPECIES),
                "host_copy": False,
            },
            "cadence": "pre-RK before prior microphysics tendency consume",
            "synchronization": {
                "device_id": current_device,
                "stream_pointer": stream_pointer,
                "ready_event_recorded": True,
                "explicit_synchronize": False,
            },
            "red_gates": {
                "live_sgs_dissten_producer_missing": True,
                "efall_1": True,
                "coupled_step": True,
                "trajectory": True,
            },
        }
        _canonical_json_bytes(receipt)
        self._last_receipt = receipt
        return Eqtset2IdissResult(
            ready_event=ready,
            call_index=self._call_count,
            stream_pointer=stream_pointer,
            receipt=MappingProxyType(json.loads(_canonical_json_bytes(receipt))),
        )

    def receipt_dict(self) -> dict[str, object]:
        if self._last_receipt is None:
            raise RuntimeError("idiss consumer has not completed an apply")
        return json.loads(_canonical_json_bytes(self._last_receipt))


__all__ = [
    "CM1_CP",
    "CM1_CPDCV",
    "CM1_CPI",
    "CM1_CPL",
    "CM1_CV",
    "CM1_CVV",
    "EQTSET2_EFALL_REFUSAL",
    "EQTSET2_ENERGY_METADATA_SCHEMA",
    "EQTSET2_IDISS_CONSUMER_ABI",
    "EQTSET2_IDISS_LIVE_REFUSAL",
    "EQTSET2_IDISS_RECEIPT_SCHEMA",
    "EQTSET2_SPECIES",
    "Eqtset2IdissCarriers",
    "Eqtset2IdissConsumer",
    "Eqtset2IdissResult",
    "cpu_idiss_cell_f32",
    "energy_terms_metadata",
    "energy_terms_metadata_bytes",
    "energy_terms_metadata_sha256",
    "require_efall_contract",
]
