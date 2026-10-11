"""Typed solve1 delivery for source-exact CM1 ``rterm=1`` radiation.

The independently evidenced CPU operator owns the bounded source slices in
``solve1.F:744-850,1002-1056``.  This module prepares its exact full-domain
default-REAL theta increment and exposes the one legal tiled enqueue point:
after ``idiss`` and before SGS stress/scalar turbulence.  The live binding is
deliberately narrower than the CPU operator: it admits only the Cartesian,
six-category, eqtset=2 idealized configuration used by the current flat-grid
GM1 path.  Budget and pressure diagnostics remain sealed in the CPU receipt
but are not coupled into the live trajectory.

Omitting the binding is the unchanged ``rterm=0`` path.  No CUDA-vs-native
raw-bit or coupled-trajectory claim is made here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .idealized_radiation import (
    CM1_IDEALIZED_RADIATION_STATUS,
    IdealizedRadiationSelection,
    cpu_idealized_radiation_inplace,
    idealized_radiation_shapes,
)


CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_SCHEMA = (
    "gm1.cuda-solve1-idealized-radiation-binding/v1"
)
CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_STATUS = (
    "PREPARED_LIVE_SOLVE1_IDEALIZED_RADIATION_AFTER_IDISS"
)
CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_SCHEMA = (
    "gm1.cuda-solve1-idealized-radiation-tile/v1"
)
CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_STATUS = (
    "ENQUEUED_LIVE_SOLVE1_IDEALIZED_RADIATION_AFTER_IDISS"
)
CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_CLAIM_LIMIT = (
    "typed 300x300x75/50 m/flat-uniform-200 m CUDA delivery of the "
    "independently source-exact CPU rterm=1 theta increment at the solve1.F "
    "post-idiss/pre-SGS boundary; CPU pressure/budget outputs are sealed but "
    "not coupled, and there is no CUDA-vs-native raw-bit oracle, diagnostic, "
    "axisymmetric coordinate system, simultaneous lspgrad/Coriolis, complete "
    "solve1/RK trajectory, production configuration, forecast, or product"
)
CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_ORDER = (
    "eqtset=2 idiss completes live closure-dissipation heating",
    "active rterm=1 adds the exact capped idealized-radiation theta tendency",
    "stretched SGS stress then scalar turbulence follow",
)
IDEALIZED_RADIATION_CPU_OPERATOR_SOURCE_SHA256 = (
    "1b1deb3b29f9d679310d90ee1f5a1580d15bb26f7ded998b240925e6528c7134"
)
_SOURCE_NAMES = (
    "tha",
    "pi0",
    "ppi",
    "th0",
    "rho",
    "ruh",
    "rvh",
    "rmh",
    "xf",
    "qa",
)


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _f32_bits(np: Any, value: float) -> int:
    return int(np.asarray(np.float32(value), dtype=np.float32).view(np.uint32))


def _require_operator_source() -> None:
    installed = Path(__file__).with_name("idealized_radiation.py")
    actual = hashlib.sha256(installed.read_bytes()).hexdigest()
    if actual != IDEALIZED_RADIATION_CPU_OPERATOR_SOURCE_SHA256:
        raise RuntimeError(
            "installed idealized-radiation CPU operator differs from its source pin"
        )


def _validate_scalar(name: str, value: object, *, expected: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if expected is not None and result != expected:
        raise ValueError(f"live idealized radiation requires exact {name}={expected}")
    return result


def _validate_geometry(
    *,
    nx: int,
    ny: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    completed_nrk: int,
    allow_compact_validation: bool,
) -> None:
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    for name, value in (("nx", nx), ("ny", ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    _validate_scalar("dx_m", dx)
    _validate_scalar("dy_m", dy)
    _validate_scalar("dz_m", dz)
    if not allow_compact_validation and (nx, ny, nk) != (300, 300, 75):
        raise ValueError(
            "live idealized radiation requires the admitted 300x300x75 geometry"
        )
    if not allow_compact_validation:
        _validate_scalar("dx_m", dx, expected=50.0)
        _validate_scalar("dy_m", dy, expected=50.0)
        _validate_scalar("dz_m", dz, expected=200.0)
    if completed_nrk != 0:
        raise ValueError("solve1 idealized radiation is admitted only before RK stage 1")


def _validate_live_selection(selection: IdealizedRadiationSelection) -> None:
    if type(selection) is not IdealizedRadiationSelection:
        raise TypeError("selection must be the exact idealized-radiation selection type")
    exact = {
        "rterm": 1,
        "radopt": 0,
        "eqtset": 2,
        "imoist": 1,
        "iice": 1,
        "nqv": 1,
        "nql1": 2,
        "nql2": 3,
        "nqs1": 4,
        "nqs2": 6,
        "axisymmetric": 0,
        "diagnostic": False,
        "mpi": False,
    }
    changed = {
        name: getattr(selection, name)
        for name, expected in exact.items()
        if getattr(selection, name) != expected
    }
    if changed:
        raise ValueError(
            "live idealized radiation requires the exact Cartesian eqtset=2 "
            f"six-category non-diagnostic selection; changed={changed}"
        )


def _validate_companions(
    *,
    lspgrad: int,
    icor: int,
    adaptive_dt: bool,
    axisymmetric_outer_coordinate: bool,
) -> None:
    if lspgrad != 0:
        raise ValueError("active idealized radiation does not admit simultaneous lspgrad")
    if icor != 0:
        raise ValueError("active idealized radiation does not admit simultaneous Coriolis")
    if adaptive_dt is not False:
        raise ValueError("active idealized radiation does not admit adaptive dt")
    if axisymmetric_outer_coordinate is not False:
        raise ValueError(
            "active idealized radiation does not admit the axisymmetric outer coordinate"
        )


def _selection_record(np: Any, selection: IdealizedRadiationSelection) -> dict[str, object]:
    return {
        "rterm": 1,
        "radopt": 0,
        "eqtset": 2,
        "imoist": 1,
        "iice": 1,
        "categories": {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6},
        "axisymmetric": 0,
        "diagnostic": False,
        "mpi": False,
        "dt_f32_bits": _f32_bits(np, selection.dt_seconds),
        "dx_f32_bits": _f32_bits(np, selection.dx_m),
        "dy_f32_bits": _f32_bits(np, selection.dy_m),
        "dz_f32_bits": _f32_bits(np, selection.dz_m),
    }


@dataclass(frozen=True, slots=True)
class Gm1Solve1IdealizedRadiationCpuIncrement:
    """Exact CPU operator outputs needed by the bounded live projection."""

    thrad: Any
    prad: Any
    qbudget_delta: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1Solve1IdealizedRadiationBinding:
    """One full-domain theta increment shared by every solve1 tile."""

    thrad: Any
    ready_event: Any
    stream: Any
    receipt: Mapping[str, object]


def build_gm1_solve1_idealized_radiation_cpu_increment(
    sources: Mapping[str, Any],
    *,
    selection: IdealizedRadiationSelection,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    completed_nrk: int = 0,
    lspgrad: int = 0,
    icor: int = 0,
    adaptive_dt: bool = False,
    axisymmetric_outer_coordinate: bool = False,
    allow_compact_validation: bool = False,
) -> Gm1Solve1IdealizedRadiationCpuIncrement:
    """Run the exact CPU family and project its theta increment for live use."""

    _validate_live_selection(selection)
    _validate_geometry(
        nx=nx,
        ny=ny,
        nk=nk,
        dx=selection.dx_m,
        dy=selection.dy_m,
        dz=selection.dz_m,
        completed_nrk=completed_nrk,
        allow_compact_validation=allow_compact_validation,
    )
    _validate_scalar("dt_seconds", selection.dt_seconds)
    _validate_companions(
        lspgrad=lspgrad,
        icor=icor,
        adaptive_dt=adaptive_dt,
        axisymmetric_outer_coordinate=axisymmetric_outer_coordinate,
    )
    if not isinstance(sources, Mapping) or tuple(sources) != _SOURCE_NAMES:
        raise ValueError(
            "idealized-radiation source inventory/order changed; expected "
            + ",".join(_SOURCE_NAMES)
        )
    _require_operator_source()

    np = importlib.import_module("numpy")
    shapes = idealized_radiation_shapes(nx, ny, nk, 6)
    workspace: dict[str, Any] = {name: sources[name] for name in _SOURCE_NAMES}
    for name in ("thten1", "cvm", "mass_weight", "thrad", "prad", "tdiag"):
        workspace[name] = np.zeros(shapes[name], dtype=np.float32)
    workspace["bud"] = np.zeros(shapes["bud"], dtype=np.float64)
    workspace["qbudget"] = np.zeros(shapes["qbudget"], dtype=np.float64)
    result = cpu_idealized_radiation_inplace(
        workspace,
        ni=nx,
        nj=ny,
        nk=nk,
        numq=6,
        selection=selection,
    )
    if result.receipt.get("status") != CM1_IDEALIZED_RADIATION_STATUS:
        raise ValueError("source-exact idealized-radiation receipt status changed")
    thrad = np.ascontiguousarray(workspace["thrad"], dtype=np.float32)
    prad = np.ascontiguousarray(workspace["prad"], dtype=np.float32)
    qbudget_delta = np.ascontiguousarray(workspace["qbudget"], dtype=np.float64)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_SCHEMA,
        "status": "PREPARED_CPU_EXACT_SOLVE1_IDEALIZED_RADIATION_INCREMENT",
        "completed_nrk": 0,
        "domain": {"nx": nx, "ny": ny, "nk": nk, "numq": 6},
        "selection": _selection_record(np, selection),
        "companion_selection": {
            "lspgrad": 0,
            "icor": 0,
            "adaptive_dt": False,
            "axisymmetric_outer_coordinate": False,
        },
        "required_source_names": list(_SOURCE_NAMES),
        "source_sha256": {name: _array_sha256(sources[name]) for name in _SOURCE_NAMES},
        "output_shapes": {
            "thrad": list(thrad.shape),
            "prad": list(prad.shape),
            "qbudget_delta": list(qbudget_delta.shape),
        },
        "output_sha256": {
            "thrad": _array_sha256(thrad),
            "prad": _array_sha256(prad),
            "qbudget_delta": _array_sha256(qbudget_delta),
        },
        "cpu_operator_operation_token_sha256": result.receipt[
            "operation_token_sha256"
        ],
        "cpu_operator_source_sha256": (
            IDEALIZED_RADIATION_CPU_OPERATOR_SOURCE_SHA256
        ),
        "source_order": list(CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_ORDER),
        "pressure_tendency_coupled": False,
        "energy_budget_coupled": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Solve1IdealizedRadiationCpuIncrement(
        thrad=thrad,
        prad=prad,
        qbudget_delta=qbudget_delta,
        receipt=MappingProxyType(receipt),
    )


def prepare_cuda_gm1_solve1_idealized_radiation(
    sources: Mapping[str, Any],
    *,
    selection: IdealizedRadiationSelection,
    stream: Any,
    nx: int = 300,
    ny: int = 300,
    nk: int = 75,
    completed_nrk: int = 0,
    lspgrad: int = 0,
    icor: int = 0,
    adaptive_dt: bool = False,
    axisymmetric_outer_coordinate: bool = False,
) -> CudaGm1Solve1IdealizedRadiationBinding:
    """Upload the exact full-domain theta increment on the explicit stream."""

    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    cpu = build_gm1_solve1_idealized_radiation_cpu_increment(
        sources,
        selection=selection,
        nx=nx,
        ny=ny,
        nk=nk,
        completed_nrk=completed_nrk,
        lspgrad=lspgrad,
        icor=icor,
        adaptive_dt=adaptive_dt,
        axisymmetric_outer_coordinate=axisymmetric_outer_coordinate,
    )
    cupy = importlib.import_module("cupy")
    with stream:
        thrad = cupy.asarray(cpu.thrad)
        ready_event = cupy.cuda.Event(disable_timing=True)
        ready_event.record(stream)
    receipt = dict(cpu.receipt)
    receipt["status"] = CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_STATUS
    receipt["stream_pointer"] = int(stream.ptr)
    receipt["host_to_device_on_source_stream"] = True
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Solve1IdealizedRadiationBinding(
        thrad=thrad,
        ready_event=ready_event,
        stream=stream,
        receipt=MappingProxyType(receipt),
    )


def require_cuda_gm1_solve1_idealized_radiation_binding(
    binding: CudaGm1Solve1IdealizedRadiationBinding,
    *,
    spec: Gm1TileSpec,
    stream: Any,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> CudaGm1Solve1IdealizedRadiationBinding:
    """Validate exact geometry, cadence, stream, and device bytes pre-enqueue."""

    if type(binding) is not CudaGm1Solve1IdealizedRadiationBinding:
        raise TypeError(
            "binding must be the exact live solve1 idealized-radiation binding"
        )
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (300, 300):
        raise ValueError("idealized-radiation tile must belong to the 300 grid")
    if stream is not binding.stream or int(stream.ptr) != binding.receipt.get(
        "stream_pointer"
    ):
        raise ValueError(
            "idealized-radiation tile stream differs from the source-owning stream"
        )
    receipt = binding.receipt
    np = importlib.import_module("numpy")
    expected_selection = dict(receipt.get("selection", {}))
    for name, value in (("dt", dt), ("dx", dx), ("dy", dy), ("dz", dz)):
        _validate_scalar(name, value)
    if (
        expected_selection.get("dt_f32_bits") != _f32_bits(np, dt)
        or expected_selection.get("dx_f32_bits") != _f32_bits(np, dx)
        or expected_selection.get("dy_f32_bits") != _f32_bits(np, dy)
        or expected_selection.get("dz_f32_bits") != _f32_bits(np, dz)
    ):
        raise ValueError("idealized-radiation binding geometry or timestep changed")
    if (
        receipt.get("schema")
        != CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_STATUS
        or receipt.get("completed_nrk") != 0
        or receipt.get("domain") != {"nx": 300, "ny": 300, "nk": 75, "numq": 6}
        or receipt.get("companion_selection")
        != {
            "lspgrad": 0,
            "icor": 0,
            "adaptive_dt": False,
            "axisymmetric_outer_coordinate": False,
        }
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("idealized-radiation binding identity or cadence changed")
    cupy = importlib.import_module("cupy")
    if (
        not isinstance(binding.thrad, cupy.ndarray)
        or binding.thrad.dtype != cupy.float32
        or tuple(binding.thrad.shape) != (75, 300, 300)
        or not binding.thrad.flags.c_contiguous
    ):
        raise ValueError(
            "binding thrad must be C-order CUDA float32 shape (75, 300, 300)"
        )
    return binding


def enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss(
    binding: CudaGm1Solve1IdealizedRadiationBinding,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    stream: Any,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> Mapping[str, object]:
    """Add one exact global ``thrad`` slice after idiss and before SGS."""

    binding = require_cuda_gm1_solve1_idealized_radiation_binding(
        binding, spec=spec, stream=stream, dt=dt, dx=dx, dy=dy, dz=dz
    )
    cupy = importlib.import_module("cupy")
    value = arrays.get("thten1") if isinstance(arrays, Mapping) else None
    shape = (75, spec.cny, spec.cnx)
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != cupy.float32
        or tuple(value.shape) != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"thten1 must be C-order CUDA float32 shape {shape}")
    with stream:
        stream.wait_event(binding.ready_event)
        cupy.add(
            value,
            binding.thrad[
                :, spec.cj0 : spec.cj0 + spec.cny, spec.ci0 : spec.ci0 + spec.cnx
            ],
            out=value,
        )
    receipt = binding.receipt
    tile_receipt: dict[str, object] = {
        "schema": CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_SCHEMA,
        "status": CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_STATUS,
        "completed_nrk": 0,
        "tile": spec.record(),
        "stream_pointer": int(stream.ptr),
        "binding_operation_token_sha256": receipt["operation_token_sha256"],
        "cpu_operator_operation_token_sha256": receipt[
            "cpu_operator_operation_token_sha256"
        ],
        "selection": dict(receipt["selection"]),
        "companion_selection": dict(receipt["companion_selection"]),
        "source_order": list(CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_ORDER),
        "same_stream_after_idiss_before_sgs": True,
        "pressure_tendency_coupled": False,
        "energy_budget_coupled": False,
        "active_cuda_raw_bit_oracle": False,
        "coupled_trajectory_admitted": False,
        "claim_limit": CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_CLAIM_LIMIT,
    }
    tile_receipt["operation_token_sha256"] = _operation_token(tile_receipt)
    return MappingProxyType(tile_receipt)


# ---------------------------------------------------------------------------
# Live device operator for CM1 r21.1 ``rterm=1``.
#
# Everything above delivers one precomputed full-domain increment for the
# sealed 300x300x75 evidence geometry.  A production run needs the tendency
# formed on the device from the live ``tha`` every large step, on any grid,
# so the three capped regions of ``src/solve1.F:1026-1031`` are transcribed
# once here as a kernel and validated raw-word against the CPU operator above
# (which is itself the evidenced transcription of the same source lines).
#
# Only ``thten1`` changes.  ``src/solve1.F:1033-1040`` forms ``prad`` and the
# ``qbudget(10)`` accumulation from ``thrad``, but neither is a prognostic:
# the pressure tendency is a diagnostic of the same increment and the budget
# is a scalar reduction.  Transcribing them into the hot kernel would cost a
# device reduction per step for a number no carrier reads, so the live
# operator computes exactly the state change CM1 makes, and the receipt says
# so rather than implying a budget that was never accumulated.
# ---------------------------------------------------------------------------

#: ``12.0*3600.0`` is exact in binary32, so the two caps are the correctly
#: rounded ``+-1.0f/43200.0f`` on both sides of the seam.
#:
#: EVERY OPERATION IS AN ``_rn`` INTRINSIC, and the division is the reason.
#: Written as plain ``-tha / cap_denominator`` this kernel disagreed with the
#: CPU operator on 3 of 11339 uniformly sampled cells, by exactly one unit in
#: the last place, and the three differences were all the signature of
#: multiply-by-reciprocal: ``x * (1/43200f)`` rather than ``x / 43200f``.
#: NVRTC strength-reduces a division by a compile-time constant even with
#: ``--prec-div=true`` in the option list.  CM1's Fortran does a real divide
#: (``src/solve1.F:1028``), so the reciprocal form is wrong by one ULP
#: wherever the two disagree -- rare, silent, and permanent in a trajectory.
#: ``__fdiv_rn`` is NVIDIA-guaranteed to be the correctly rounded quotient
#: and cannot be strength-reduced; ``__fadd_rn`` and ``__fmul_rn`` are here
#: for the same reason, so no later edit can reintroduce a contraction.
CUDA_GM1_RTERM_SOURCE = r"""
extern "C" __global__ void gm1_solve1_rterm_theta_tendency(
    float *__restrict__ thten1,
    const float *__restrict__ tha,
    const unsigned int count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    /* src/solve1.F:1026-1031 */
    const float cap_denominator = __fmul_rn(12.0f, 3600.0f);
    const float negative_cap = __fdiv_rn(-1.0f, cap_denominator);
    const float positive_cap = __fdiv_rn(1.0f, cap_denominator);
    const float theta_perturbation = tha[index];
    float thrad = __fdiv_rn(-theta_perturbation, cap_denominator);
    if (theta_perturbation > 1.0f) {
        thrad = negative_cap;
    }
    if (theta_perturbation < -1.0f) {
        thrad = positive_cap;
    }
    /* src/solve1.F:1032 */
    thten1[index] = __fadd_rn(thten1[index], thrad);
}
"""
CUDA_GM1_RTERM_KERNEL_NAME = "gm1_solve1_rterm_theta_tendency"
CUDA_GM1_RTERM_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_RTERM_SOURCE.encode("ascii")
).hexdigest()
CUDA_GM1_RTERM_THREADS = 256
#: IEEE divide, no contraction, no flush-to-zero: the same three operations
#: numpy performs in :func:`cm1core.idealized_radiation.cpu_idealized_radiation_inplace`.
CUDA_GM1_RTERM_COMPILE_OPTIONS = (
    "--std=c++14",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
CUDA_GM1_RTERM_AUTHORITY = (
    "CM1 r21.1 src/solve1.F:1003-1054 (Rotunno-Emanuel rterm=1), "
    "README.namelist:555-558, src/param.F:2283-2294 (radopt and rterm are "
    "mutually exclusive)"
)


@dataclass(frozen=True, slots=True)
class CudaGm1RtermCompiled:
    """One compiled live ``rterm=1`` kernel and the text it was built from."""

    kernel: Any
    source_sha256: str


def compile_cuda_gm1_rterm() -> CudaGm1RtermCompiled:
    """Compile the live capped idealized-radiation tendency kernel."""

    if (
        hashlib.sha256(CUDA_GM1_RTERM_SOURCE.encode("ascii")).hexdigest()
        != CUDA_GM1_RTERM_SOURCE_SHA256
    ):
        raise RuntimeError("local rterm kernel source differs from its pin")
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_RTERM_SOURCE,
        options=CUDA_GM1_RTERM_COMPILE_OPTIONS,
        name_expressions=(CUDA_GM1_RTERM_KERNEL_NAME,),
        filename="gm1_rterm.cu",
    )
    return CudaGm1RtermCompiled(
        kernel=module.get_function(CUDA_GM1_RTERM_KERNEL_NAME),
        source_sha256=CUDA_GM1_RTERM_SOURCE_SHA256,
    )


def launch_cuda_gm1_rterm(
    thten1: Any,
    tha: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
    compiled: CudaGm1RtermCompiled,
) -> None:
    """Add CM1's capped ``rterm=1`` theta tendency to one live tile."""

    if type(compiled) is not CudaGm1RtermCompiled:
        raise TypeError("compiled must be the live rterm kernel binding")
    cupy = importlib.import_module("cupy")
    shape = (nk, nj, ni)
    for name, value in (("thten1", thten1), ("tha", tha)):
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 shape {shape}")
    if int(thten1.data.ptr) == int(tha.data.ptr):
        raise ValueError("thten1 and tha must not alias")
    count = nk * nj * ni
    blocks = (count + CUDA_GM1_RTERM_THREADS - 1) // CUDA_GM1_RTERM_THREADS
    compiled.kernel(
        (blocks, 1, 1),
        (CUDA_GM1_RTERM_THREADS, 1, 1),
        (thten1, tha, cupy.uint32(count)),
        stream=stream,
    )


def cpu_gm1_rterm_theta_tendency(tha: Any) -> Any:
    """The CPU oracle for :func:`launch_cuda_gm1_rterm`, in numpy binary32.

    Written as the same three source lines rather than as a vectorized
    rewrite, so a reader can check it against ``src/solve1.F:1026-1031``
    without reasoning about broadcast order.
    """

    np = importlib.import_module("numpy")
    values = np.asarray(tha, dtype=np.float32)
    cap_denominator = np.float32(np.float32(12.0) * np.float32(3600.0))
    negative_cap = np.float32(np.float32(-1.0) / cap_denominator)
    positive_cap = np.float32(np.float32(1.0) / cap_denominator)
    thrad = np.empty_like(values)
    flat_in = values.reshape(-1)
    flat_out = thrad.reshape(-1)
    for index in range(flat_in.shape[0]):
        theta_perturbation = flat_in[index]
        value = np.float32(np.float32(-theta_perturbation) / cap_denominator)
        if theta_perturbation > np.float32(1.0):
            value = negative_cap
        if theta_perturbation < np.float32(-1.0):
            value = positive_cap
        flat_out[index] = value
    return thrad


CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = [
    "CUDA_GM1_RTERM_AUTHORITY",
    "CUDA_GM1_RTERM_COMPILE_OPTIONS",
    "CUDA_GM1_RTERM_KERNEL_NAME",
    "CUDA_GM1_RTERM_SOURCE",
    "CUDA_GM1_RTERM_SOURCE_SHA256",
    "CUDA_GM1_RTERM_THREADS",
    "CudaGm1RtermCompiled",
    "compile_cuda_gm1_rterm",
    "cpu_gm1_rterm_theta_tendency",
    "launch_cuda_gm1_rterm",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_SCHEMA",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_BINDING_STATUS",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_CLAIM_LIMIT",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_ORDER",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_SOURCE_SHA256",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_SCHEMA",
    "CUDA_GM1_SOLVE1_IDEALIZED_RADIATION_TILE_STATUS",
    "CudaGm1Solve1IdealizedRadiationBinding",
    "Gm1Solve1IdealizedRadiationCpuIncrement",
    "IDEALIZED_RADIATION_CPU_OPERATOR_SOURCE_SHA256",
    "build_gm1_solve1_idealized_radiation_cpu_increment",
    "enqueue_cuda_gm1_solve1_idealized_radiation_after_idiss",
    "prepare_cuda_gm1_solve1_idealized_radiation",
    "require_cuda_gm1_solve1_idealized_radiation_binding",
]
