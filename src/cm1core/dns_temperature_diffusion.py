"""Source-exact CM1 r21.1 DNS potential-temperature vertical diffusion.

This bounded CPU operator owns the complete ``cm1setup=3``, ``ivar=1``
``bc_temp=1/2`` slice of ``turbsz`` for both explicit and implicit vertical
diffusion.  It deliberately remains an operator-only port: the production
turbulence dispatcher and coupled time-step integration are not admitted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_DNS_TEMPERATURE_DIFFUSION_SCHEMA = (
    "gpuwm-cm1.dns-temperature-diffusion/v1"
)
CM1_DNS_TEMPERATURE_DIFFUSION_STATUS = (
    "ADMITTED_CM1_R21_1_DNS_TEMPERATURE_DIFFUSION_OPERATOR_ONLY"
)
CM1_DNS_TEMPERATURE_DIFFUSION_CLAIM_LIMIT = (
    "CM1 r21.1 turbsz cm1setup=3, ivar=1 potential-temperature vertical "
    "diffusion with bc_temp=1/2 and doimpl=0/1 on supplied compact carriers "
    "only; no water-vapor ivar=2 saturation boundary, other scalar, "
    "horizontal turbulence, turbulence dispatch, coupled large step, "
    "production configuration, trajectory, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_DNS_TEMPERATURE_DIFFUSION_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/constants.F": (
            "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/turbtend.F": (
            "07857b0ee315c8b1d4552c3a5c049ba7a5aac16ec9c624b4afacba66e5a240de"
        ),
    }
)
CM1_DNS_TEMPERATURE_DIFFUSION_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1177-1197",
        "selection_validation": "src/param.F:1550-1565",
        "theta_boundary_values": "src/turbtend.F:466-482",
        "explicit_operator": "src/turbtend.F:484-556",
        "implicit_operator": "src/turbtend.F:558-690",
        "default_real_constants": "src/constants.F:14,22,49-52",
    }
)


@dataclass(frozen=True, slots=True)
class DnsTemperatureDiffusionSelection:
    """Exact outer selectors and default-REAL scalars for the bounded slice."""

    cm1setup: int
    ivar: int
    bc_temp: int
    doimpl: int
    dt_seconds: float
    rdz_m_inv: float
    viscosity_m2_s: float
    prandtl_number: float
    ptc_bot_k: float
    ptc_top_k: float


@dataclass(frozen=True, slots=True)
class DnsTemperatureDiffusionResult:
    """Completed source-order operator and immutable receipt."""

    turbz: Any
    dumz: Any
    receipt: Mapping[str, object]


def dns_temperature_diffusion_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    """Return compact C-order state and vertical-face carrier shapes."""

    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    state = (nk, nj, ni)
    face = (nk + 1, nj, ni)
    return {
        "mh": state,
        "mf": face,
        "rho": state,
        "rr": state,
        "rf": face,
        "s": state,
        "khv": face,
        "dum7": state,
        "dum8": state,
        "turbz": state,
        "dum1": state,
        "dum2": state,
        "dumz": face,
    }


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(
    np: Any, selection: DnsTemperatureDiffusionSelection
) -> dict[str, Any]:
    if not isinstance(selection, DnsTemperatureDiffusionSelection):
        raise TypeError("selection must be a DnsTemperatureDiffusionSelection")
    for name in ("cm1setup", "ivar", "bc_temp", "doimpl"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.cm1setup != 3 or selection.ivar != 1:
        refuse(
            code="CM1-DNS-TEMPERATURE-DIFFUSION-OUTER-DISPATCH",
            gate="GM1-PHYSICS-SUPPORT",
            option="cm1setup/ivar",
            value=(selection.cm1setup, selection.ivar),
            reason="this bounded operator owns only DNS potential temperature",
            remedy="use cm1setup=3 and ivar=1 or add a separately evidenced scalar slice",
        )
    if selection.bc_temp not in (1, 2):
        refuse(
            code="CM1-DNS-TEMPERATURE-DIFFUSION-BC",
            gate="GM1-PHYSICS-SUPPORT",
            option="bc_temp",
            value=selection.bc_temp,
            reason="CM1 r21.1 DNS admits bc_temp=1 or bc_temp=2",
            remedy="select an evidenced DNS temperature boundary branch",
        )
    if selection.doimpl not in (0, 1):
        refuse(
            code="CM1-DNS-TEMPERATURE-DIFFUSION-IMPLICIT",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="doimpl",
            value=selection.doimpl,
            reason="the bounded operator owns explicit 0 and implicit 1 only",
            remedy="select doimpl=0/1 or add separately evidenced semantics",
        )

    values = {
        "dt": _f32(np, "dt_seconds", selection.dt_seconds),
        "rdz": _f32(np, "rdz_m_inv", selection.rdz_m_inv),
        "viscosity": _f32(
            np, "viscosity_m2_s", selection.viscosity_m2_s
        ),
        "prandtl": _f32(np, "prandtl_number", selection.prandtl_number),
        "ptc_bot": _f32(np, "ptc_bot_k", selection.ptc_bot_k),
        "ptc_top": _f32(np, "ptc_top_k", selection.ptc_top_k),
    }
    for name in ("dt", "rdz", "viscosity", "prandtl"):
        if values[name] <= np.float32(0.0):
            raise ValueError(f"{name} must be positive")
    return values


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = dns_temperature_diffusion_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(
            "missing DNS temperature-diffusion arrays: " + ", ".join(missing)
        )
    values: list[tuple[str, Any]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must have native float32 dtype")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        values.append((name, value))
    for name in ("turbz", "dum1", "dum2", "dumz"):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "DNS temperature-diffusion arrays must not alias: "
                    f"{left_name}, {right_name}"
                )


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _boundary_fluxes(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    bc_temp: int,
    kappa: Any,
    ptcb: Any,
    ptct: Any,
    ptc_bot: Any,
    ptc_top: Any,
    rdz: Any,
) -> None:
    two = np.float32(2.0)
    for j in range(nj):
        for i in range(ni):
            if bc_temp == 1:
                bottom = _mul(np, np.negative(kappa), two)
                bottom = _mul(np, bottom, _sub(np, arrays["s"][0, j, i], ptcb))
                bottom = _mul(np, bottom, rdz)
                bottom = _mul(np, bottom, arrays["mf"][0, j, i])
                bottom = _mul(np, bottom, arrays["rf"][0, j, i])
                arrays["dumz"][0, j, i] = bottom

                top = _mul(np, np.negative(kappa), two)
                top = _mul(np, top, _sub(np, ptct, arrays["s"][nk - 1, j, i]))
                top = _mul(np, top, rdz)
                top = _mul(np, top, arrays["mf"][nk, j, i])
                top = _mul(np, top, arrays["rf"][nk, j, i])
                arrays["dumz"][nk, j, i] = top
            else:
                bottom = _mul(np, kappa, ptc_bot)
                arrays["dumz"][0, j, i] = _mul(
                    np, bottom, arrays["rf"][0, j, i]
                )
                top = _mul(np, kappa, ptc_top)
                arrays["dumz"][nk, j, i] = _mul(
                    np, top, arrays["rf"][nk, j, i]
                )


def cpu_dns_temperature_diffusion_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: DnsTemperatureDiffusionSelection,
) -> DnsTemperatureDiffusionResult:
    """Apply the source-ordered DNS theta vertical diffusion slice."""

    np = importlib.import_module("numpy")
    scalar = _validate_selection(np, selection)
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)

    one = np.float32(1.0)
    zero = np.float32(0.0)
    th0r = np.float32(300.0)
    dt = scalar["dt"]
    rdz = scalar["rdz"]
    kappa = np.float32(scalar["viscosity"] / scalar["prandtl"])
    ptcb = _sub(np, scalar["ptc_bot"], th0r)
    ptct = _sub(np, scalar["ptc_top"], th0r)

    if selection.doimpl == 0:
        for j in range(nj):
            for k in range(1, nk):
                for i in range(ni):
                    flux = np.negative(arrays["khv"][k, j, i])
                    flux = _mul(
                        np,
                        flux,
                        _sub(
                            np,
                            arrays["s"][k, j, i],
                            arrays["s"][k - 1, j, i],
                        ),
                    )
                    flux = _mul(np, flux, rdz)
                    flux = _mul(np, flux, arrays["mf"][k, j, i])
                    flux = _mul(np, flux, arrays["rf"][k, j, i])
                    arrays["dumz"][k, j, i] = flux
        _boundary_fluxes(
            np,
            arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            bc_temp=selection.bc_temp,
            kappa=kappa,
            ptcb=ptcb,
            ptct=ptct,
            ptc_bot=scalar["ptc_bot"],
            ptc_top=scalar["ptc_top"],
            rdz=rdz,
        )
        for j in range(nj):
            for k in range(nk):
                for i in range(ni):
                    tendency = np.negative(
                        _sub(
                            np,
                            arrays["dumz"][k + 1, j, i],
                            arrays["dumz"][k, j, i],
                        )
                    )
                    tendency = _mul(np, tendency, rdz)
                    arrays["turbz"][k, j, i] = _mul(
                        np, tendency, arrays["mh"][k, j, i]
                    )
    else:
        rdt = np.float32(one / dt)
        tema = _mul(np, np.float32(-1.0), dt)
        tema = _mul(np, tema, one)  # vialpha = 1.0
        tema = _mul(np, tema, rdz)
        tema = _mul(np, tema, rdz)
        temb = _mul(np, dt, zero)  # vibeta = 0.0
        temb = _mul(np, temb, rdz)
        temb = _mul(np, temb, rdz)
        temc = _mul(np, dt, rdz)
        _boundary_fluxes(
            np,
            arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            bc_temp=selection.bc_temp,
            kappa=kappa,
            ptcb=ptcb,
            ptct=ptct,
            ptc_bot=scalar["ptc_bot"],
            ptc_top=scalar["ptc_top"],
            rdz=rdz,
        )

        k = 0
        for j in range(nj):
            for i in range(ni):
                r2 = arrays["dum8"][k, j, i]
                cfc = _mul(np, tema, r2)
                cfb = _sub(np, one, cfc)
                inner = _add(
                    np,
                    _mul(np, np.negative(r2), arrays["s"][k, j, i]),
                    _mul(np, r2, arrays["s"][k + 1, j, i]),
                )
                cfd = _add(np, arrays["s"][k, j, i], _mul(np, temb, inner))
                boundary = _mul(np, temc, arrays["dumz"][0, j, i])
                boundary = _mul(np, boundary, arrays["mh"][0, j, i])
                boundary = _mul(np, boundary, arrays["rr"][0, j, i])
                cfd = _add(np, cfd, boundary)
                tem = np.float32(one / cfb)
                arrays["dum1"][k, j, i] = _mul(np, np.negative(cfc), tem)
                arrays["dum2"][k, j, i] = _mul(np, cfd, tem)

        for k in range(1, nk - 1):
            for j in range(nj):
                for i in range(ni):
                    r1 = arrays["dum7"][k, j, i]
                    r2 = arrays["dum8"][k, j, i]
                    cfa = _mul(np, tema, r1)
                    cfc = _mul(np, tema, r2)
                    cfb = _sub(np, _sub(np, one, cfa), cfc)
                    inner = _mul(np, r1, arrays["s"][k - 1, j, i])
                    rsum = _add(np, r1, r2)
                    inner = _sub(
                        np, inner, _mul(np, rsum, arrays["s"][k, j, i])
                    )
                    inner = _add(
                        np, inner, _mul(np, r2, arrays["s"][k + 1, j, i])
                    )
                    cfd = _add(
                        np, arrays["s"][k, j, i], _mul(np, temb, inner)
                    )
                    denom = _add(
                        np,
                        _mul(np, cfa, arrays["dum1"][k - 1, j, i]),
                        cfb,
                    )
                    tem = np.float32(one / denom)
                    arrays["dum1"][k, j, i] = _mul(
                        np, np.negative(cfc), tem
                    )
                    numerator = _sub(
                        np,
                        cfd,
                        _mul(np, cfa, arrays["dum2"][k - 1, j, i]),
                    )
                    arrays["dum2"][k, j, i] = _mul(np, numerator, tem)

        k = nk - 1
        for j in range(nj):
            for i in range(ni):
                r1 = arrays["dum7"][k, j, i]
                cfa = _mul(np, tema, r1)
                cfb = _sub(np, one, cfa)
                inner = _sub(
                    np,
                    _mul(np, r1, arrays["s"][k - 1, j, i]),
                    _mul(np, r1, arrays["s"][k, j, i]),
                )
                cfd = _add(np, arrays["s"][k, j, i], _mul(np, temb, inner))
                boundary = _mul(np, temc, arrays["dumz"][nk, j, i])
                boundary = _mul(np, boundary, arrays["mh"][k, j, i])
                boundary = _mul(np, boundary, arrays["rr"][k, j, i])
                cfd = _sub(np, cfd, boundary)
                denom = _add(
                    np,
                    _mul(np, cfa, arrays["dum1"][k - 1, j, i]),
                    cfb,
                )
                tem = np.float32(one / denom)
                numerator = _sub(
                    np,
                    cfd,
                    _mul(np, cfa, arrays["dum2"][k - 1, j, i]),
                )
                arrays["dumz"][k, j, i] = _mul(np, numerator, tem)
                delta = _sub(
                    np, arrays["dumz"][k, j, i], arrays["s"][k, j, i]
                )
                tendency = _mul(np, arrays["rho"][k, j, i], delta)
                arrays["turbz"][k, j, i] = _mul(np, tendency, rdt)

        for k in range(nk - 2, -1, -1):
            for j in range(nj):
                for i in range(ni):
                    solution = _mul(
                        np,
                        arrays["dum1"][k, j, i],
                        arrays["dumz"][k + 1, j, i],
                    )
                    solution = _add(np, solution, arrays["dum2"][k, j, i])
                    arrays["dumz"][k, j, i] = solution
                    delta = _sub(np, solution, arrays["s"][k, j, i])
                    tendency = _mul(np, arrays["rho"][k, j, i], delta)
                    arrays["turbz"][k, j, i] = _mul(np, tendency, rdt)

    output_names = ("turbz", "dum1", "dum2", "dumz")
    receipt: dict[str, object] = {
        "schema": CM1_DNS_TEMPERATURE_DIFFUSION_SCHEMA,
        "status": CM1_DNS_TEMPERATURE_DIFFUSION_STATUS,
        "claim_limit": CM1_DNS_TEMPERATURE_DIFFUSION_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_DNS_TEMPERATURE_DIFFUSION_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_DNS_TEMPERATURE_DIFFUSION_SOURCE_ANCHORS),
        },
        "selection": {
            "cm1setup": selection.cm1setup,
            "ivar": selection.ivar,
            "bc_temp": selection.bc_temp,
            "doimpl": selection.doimpl,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
            },
            "kappa_raw_bits": int(np.asarray(kappa).view(np.uint32)),
            "ptcb_raw_bits": int(np.asarray(ptcb).view(np.uint32)),
            "ptct_raw_bits": int(np.asarray(ptct).view(np.uint32)),
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "output_sha256": {
            name: _sha256_array(arrays[name]) for name in output_names
        },
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return DnsTemperatureDiffusionResult(
        turbz=arrays["turbz"],
        dumz=arrays["dumz"],
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_DNS_TEMPERATURE_DIFFUSION_AUTHORITY_SHA256",
    "CM1_DNS_TEMPERATURE_DIFFUSION_CLAIM_LIMIT",
    "CM1_DNS_TEMPERATURE_DIFFUSION_SCHEMA",
    "CM1_DNS_TEMPERATURE_DIFFUSION_SOURCE_ANCHORS",
    "CM1_DNS_TEMPERATURE_DIFFUSION_STATUS",
    "DnsTemperatureDiffusionResult",
    "DnsTemperatureDiffusionSelection",
    "cpu_dns_temperature_diffusion_inplace",
    "dns_temperature_diffusion_shapes",
]
