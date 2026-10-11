"""CM1 r21.1 six-species positive-definite repair before WSM6.

Official CM1 applies ``pdefq``/``pdefqtest`` to the live ``q3d`` state once
per completed large step, after final-RK dynamics (including ``apmasscon``)
and immediately before microphysics.  With hard-wired ``pdscheme=1`` it
clips values below the species threshold and rescales every retained level in
the same vertical column so the pre-repair rho-weighted column sum is kept.
``solve3`` later copies the repaired/post-physics ``q3d`` into ``qa``.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .cuda_sound3_periodic import _canonical_arch, _image_options
from .errors import refuse


CUDA_WSM6_PDEFQ_SCHEMA = "gpuwm-cm1.cuda-wsm6-pdefq/v1"
CUDA_WSM6_PDEFQ_ABI = "gpuwm-cm1.cuda-wsm6-pdefq-kernels/v1"
CUDA_WSM6_PDEFQ_KERNEL_NAMES = (
    "cm1_wsm6_pdefq_columns_f32_v1",
    "cm1_wsm6_pdefq_finish_f64_v1",
)
CUDA_WSM6_PDEFQ_COLUMN_ABI = "gpuwm-cm1.cuda-wsm6-pdefq-column-kernels/v1"
CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES = (
    "cm1_wsm6_pdefq_column_f32_v1",
    "cm1_wsm6_pdefq_rowscan_f64_v1",
)
WSM6_SPECIES_ORDER = ("qv", "qc", "qr", "qi", "qs", "qg")
CM1_WSM6_PDEFQ_CLAIM_LIMIT = (
    "CM1 r21.1 Cartesian six-species pdscheme=1 pdefq/pdefqtest repair on "
    "supplied live q3d/rho/ruh/rvh/rmh carriers and asq ledgers, once after "
    "final-RK dynamics/apmasscon and before WSM6; no direct qa repair, "
    "microphysics, solve3 copy, halos, axisymmetric/MPI budget reduction, "
    "pdscheme alternatives, full trajectory, forecast, or skill claim"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/param.F": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    "src/solve3.F": "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393",
    "src/mp_driver.F": "f70684282df49338dbb6336c7f3ca538c66e8cc99933fe5fb52325303c394134",
    "src/misclibs.F": "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1",
    "src/restart_write.F": "d9c21d7071a4548cb9d3ffee1658550ce0e2a4ee7db9c67758ac8c61c07cddba",
    "src/restart_read.F": "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
}
# CM1's ``pdscheme=1`` branch (src/misclibs.F:1058-1088) carries exactly three
# ordered reductions, and nothing else in it is ordered:
#
#   * the four per-column accumulations over ``k`` -- ``t1``, ``a1``, ``t2``,
#     ``a2`` -- which misclibs.F:1065-1068 resets at every ``i`` and which
#     therefore belong to one ``(i,j)`` column alone;
#   * the west-to-east carry ``budj(j)=budj(j)+a2-a1`` at misclibs.F:1082.
#     Fortran parses that left-associated as ``(budj+a2)-a1``, so it is a
#     recurrence over ``i`` rather than a sum and cannot be regrouped;
#   * the south-to-north ``asq=asq+budj(j)*tem`` at misclibs.F:1086-1088.
#
# Every ``(i,j)`` column reads only read-only carriers plus its own ``q3d``
# column and writes only its own ``q3d`` column, so the columns are mutually
# independent: misclibs.F's ``i`` loop carries nothing but ``budj``.  A layout
# may therefore evaluate the columns in any order, and only the two carries
# above have to stay serial.  Both layouts below do exactly that, with the same
# operations in the same per-column order, so they are raw-word identical by
# construction -- which is why the default may move between them without a
# named profile.
#
# ``row-serial`` is the pinned transcription: one thread per physical row ``j``,
# which walks ``i`` and ``k`` itself.  Its parallelism is ``nj`` alone, so its
# cost is proportional to ``ni*nk`` and independent of ``nj``, and on a 448-row
# domain it resolves to four thread blocks.
# ``column-parallel`` launches one thread per ``(i,j)`` column -- ``ni*nj`` of
# them, indexed so that a warp reads consecutive ``i`` at one ``(k,j)`` -- and
# then runs misclibs.F:1082's carry as a separate ordered west-to-east scan over
# the per-column ``(a1,a2)`` pairs the column kernel wrote.
WSM6_PDEFQ_ROW_SERIAL_LAYOUT = "row-serial"
WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT = "column-parallel"
WSM6_PDEFQ_LAYOUTS = (
    WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT,
    WSM6_PDEFQ_ROW_SERIAL_LAYOUT,
)
DEFAULT_WSM6_PDEFQ_LAYOUT = WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT
WSM6_PDEFQ_BLOCK_THREADS = 128
# CUDA's maximum ``gridDim.y``.  ``column-parallel`` indexes the physical row
# by ``blockIdx.y``, so a taller domain would have to be launched differently.
CUDA_MAX_GRID_DIM_Y = 65535


SOURCE_ANCHORS = {
    "hardwired_pdscheme": "src/param.F:731-732",
    "completed_step_microphysics": "src/cm1.F:3405-3432",
    "ptype5_repair": "src/mp_driver.F:505-519",
    "wsm6_repair": "src/mp_driver.F:1469-1479",
    "pdefq": "src/misclibs.F:1037-1117",
    "pdefqtest": "src/misclibs.F:1125-1230",
    "solve3_publish_qa": "src/solve3.F:625-639",
    "restart_budget_ledgers": "src/restart_write.F:618-666; src/restart_read.F:894-912",
}


@dataclass(frozen=True, slots=True)
class Wsm6PdefqSelection:
    semantic_component_id: str = "wsm6-mp6"
    pdscheme: int = 1
    axisymm: int = 0
    species_order: tuple[str, ...] = WSM6_SPECIES_ORDER
    qsmall: float = 1.0e-14


DEFAULT_WSM6_PDEFQ_SELECTION = Wsm6PdefqSelection()


def require_wsm6_pdefq_selection(
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
) -> None:
    if not isinstance(selection, Wsm6PdefqSelection):
        raise TypeError("selection must be a Wsm6PdefqSelection")
    expected = DEFAULT_WSM6_PDEFQ_SELECTION
    for name in selection.__dataclass_fields__:
        value, wanted = getattr(selection, name), getattr(expected, name)
        if type(value) is not type(wanted) or value != wanted:  # noqa: E721
            refuse(
                code="ARWEN-WSM6-PDEFQ-SELECTION",
                gate="pre-wsm6-positive-definite-repair",
                option=name,
                value=value,
                reason="the bounded repair is sealed to CM1's hard-wired six-species pdscheme=1 branch",
                remedy=f"select {name}={wanted!r} or add an independently evidenced branch",
            )


def require_wsm6_pdefq_layout(layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT) -> str:
    """Admit one of the two transcribed launch layouts, by value."""

    if not isinstance(layout, str):
        raise TypeError("pdefq launch layout must be a string")
    if layout not in WSM6_PDEFQ_LAYOUTS:
        refuse(
            code="ARWEN-WSM6-PDEFQ-LAYOUT",
            gate="pre-wsm6-positive-definite-repair",
            option="pdefq_launch",
            value=layout,
            reason=(
                "only the two transcribed layouts keep misclibs.F:1082's "
                "left-associated west-to-east budj carry and "
                "misclibs.F:1065-1081's per-column k order serial; an "
                "unadmitted layout would regroup one of those two sums and "
                "publish an asq moisture-correction ledger -- and the rescaled "
                "q3d that the same t3 produces -- differing from the authority "
                "in their trailing bits"
            ),
            remedy=f"select one of {', '.join(sorted(WSM6_PDEFQ_LAYOUTS))}",
        )
    return layout


def wsm6_pdefq_launch_plan(
    *, ni: int, nj: int, nk: int, layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT
) -> dict[str, Any]:
    """Decide, on the host, the exact CUDA grids this layout will launch.

    The launcher calls this and nothing else to size its grids, so a test that
    reads this plan is reading the geometry the device actually gets.
    """

    require_wsm6_pdefq_layout(layout)
    wsm6_pdefq_shapes(ni, nj, nk)
    threads = WSM6_PDEFQ_BLOCK_THREADS
    if layout == WSM6_PDEFQ_ROW_SERIAL_LAYOUT:
        stages = (
            {
                "kernel": CUDA_WSM6_PDEFQ_KERNEL_NAMES[0],
                "grid": [(nj + threads - 1) // threads, 1, 1],
                "block": [threads, 1, 1],
                "launches": len(WSM6_SPECIES_ORDER),
                "carries": "per-column k, then the west-to-east budj recurrence, in one thread per row",
            },
        )
    else:
        if nj > CUDA_MAX_GRID_DIM_Y:
            refuse(
                code="ARWEN-WSM6-PDEFQ-ROWS",
                gate="pre-wsm6-positive-definite-repair",
                option="nj",
                value=nj,
                reason=(
                    f"the column-parallel layout indexes the physical row by blockIdx.y, "
                    f"which CUDA caps at {CUDA_MAX_GRID_DIM_Y}; a taller domain would launch "
                    "no block at all for the rows past the cap, leaving their q3d unrepaired "
                    "and their budj row of the asq ledger reading uninitialised device memory"
                ),
                remedy=(
                    f"use nj of at most {CUDA_MAX_GRID_DIM_Y}, or select "
                    f"pdefq_launch={WSM6_PDEFQ_ROW_SERIAL_LAYOUT!r}"
                ),
            )
        stages = (
            {
                "kernel": CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES[0],
                "grid": [(ni + threads - 1) // threads, nj, 1],
                "block": [threads, 1, 1],
                "launches": len(WSM6_SPECIES_ORDER),
                "carries": "per-column k, one thread per (i,j) column",
            },
            {
                "kernel": CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES[1],
                "grid": [(nj + threads - 1) // threads, 1, 1],
                "block": [threads, 1, 1],
                "launches": len(WSM6_SPECIES_ORDER),
                "carries": "the west-to-east budj recurrence, serial over i in one thread per row",
            },
        )
    finish = {
        "kernel": CUDA_WSM6_PDEFQ_KERNEL_NAMES[1],
        "grid": [1, 1, 1],
        "block": [1, 1, 1],
        "launches": 1,
        "carries": "asq species then south-to-north j, serial in one thread",
    }
    columns = ni * nj
    return {
        "layout": layout,
        "geometry": {"ni": ni, "nj": nj, "nk": nk},
        "stages": [dict(stage) for stage in (*stages, finish)],
        # The number that made the pinned layout a defect: a 448-row domain
        # resolves row-serial to four blocks whatever ni and nk are.
        "blocks_per_species": stages[0]["grid"][0] * stages[0]["grid"][1],
        "threads_per_species": stages[0]["grid"][0] * stages[0]["grid"][1] * threads,
        "independent_columns": columns,
        "authority": SOURCE_ANCHORS["pdefq"],
    }


def wsm6_pdefq_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "q3d": (6, nk, nj, ni),
        "rho": (nk, nj, ni),
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": (nk, nj, ni),
        "asq": (6,),
    }


def _require_host_arrays(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    shapes = wsm6_pdefq_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing WSM6 pdefq arrays: {', '.join(missing)}")
    result = {}
    for name, shape in shapes.items():
        value = arrays[name]
        dtype = np.float64 if name == "asq" else np.float32
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order {np.dtype(dtype)} shape {shape}")
        if name in ("q3d", "asq") and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    if bool(np.any(result["rho"] <= np.float32(0.0))):
        raise ValueError("rho must be positive")
    for name in ("ruh", "rvh", "rmh"):
        if bool(np.any(result[name] <= np.float32(0.0))):
            raise ValueError(f"{name} must be positive")
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("WSM6 pdefq arrays must not alias")
    return result


def _m(np: Any, left: Any, right: Any) -> Any:
    return np.multiply(left, right, dtype=np.float32)


def _pdefq_column(
    np: Any,
    a: Mapping[str, Any],
    q: Any,
    *,
    i: int,
    j: int,
    nk: int,
    threshold: Any,
) -> tuple[float, float, int]:
    """One ``(i,j)`` column of misclibs.F:1065-1081, in the authority's k order.

    This is every statement inside CM1's ``do i`` body except its last line.
    It reads ``rho``/``ruh``/``rvh``/``rmh`` and this column of ``q3d``, and it
    writes this column of ``q3d`` and nothing else -- which is exactly why the
    columns can be evaluated in any order.  It returns misclibs.F:1071's
    ``a1`` and misclibs.F:1080's ``a2``, the only two words the ``do i`` loop
    carries out of one column, and how many words it clipped.
    """

    t1 = t2 = a1 = a2 = 0.0
    corrected = 0
    for k in range(nk):
        p = (k, j, i)
        t1 += float(_m(np, a["rho"][p], q[p]))
        cell = _m(np, _m(np, _m(np, _m(np, a["rho"][p], q[p]), a["ruh"][i]), a["rvh"][j]), a["rmh"][p])
        a1 += float(cell)
        if q[p] < threshold:
            q[p] = np.float32(0.0)
            corrected += 1
        t2 += float(_m(np, a["rho"][p], q[p]))
    t3 = (t1 + 1.0e-20) / (t2 + 1.0e-20)
    if t3 < 0.0:
        t3 = 1.0
    for k in range(nk):
        p = (k, j, i)
        q[p] = np.float32(t3 * float(q[p]))
        cell = _m(np, _m(np, _m(np, _m(np, a["rho"][p], q[p]), a["ruh"][i]), a["rvh"][j]), a["rmh"][p])
        a2 += float(cell)
    return a1, a2, corrected


def cpu_wsm6_pdefq_pdscheme1_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
) -> dict[str, Any]:
    """Source-order CPU oracle for all six sequential pdefq calls."""

    require_wsm6_pdefq_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    values = tuple(np.float32(value) for value in (dx, dy, dz))
    if not all(bool(np.isfinite(value)) and value > np.float32(0.0) for value in values):
        raise ValueError("dx, dy, and dz must be positive finite float32")
    volume32 = _m(np, _m(np, values[0], values[1]), values[2])
    budgets = np.empty((6, nj), dtype=np.float64)
    # misclibs.F:1071 ``a1`` and misclibs.F:1080 ``a2`` for every column, kept
    # with the physical row fastest so this is the same object the
    # column-parallel device workspace holds.
    pairs = np.empty((6, 2, ni, nj), dtype=np.float64)
    corrected_words = 0
    for species in range(6):
        q = a["q3d"][species]
        threshold = np.float32(0.0 if species == 0 else selection.qsmall)
        for j in range(nj):
            budj = 0.0
            for i in range(ni):
                a1, a2, corrected = _pdefq_column(
                    np, a, q, i=i, j=j, nk=nk, threshold=threshold
                )
                corrected_words += corrected
                pairs[species, 0, i, j] = a1
                pairs[species, 1, i, j] = a2
                # ``budj=budj+a2-a1`` is left-associated in the authority.
                budj = (budj + a2) - a1
            budgets[species, j] = budj
        for j in range(nj):
            a["asq"][species] += budgets[species, j] * float(volume32)
    return {
        "schema": CUDA_WSM6_PDEFQ_SCHEMA,
        "corrected_input_words": corrected_words,
        "budget_columns": budgets,
        "column_pairs": pairs,
        "source_order": [f"pdefq({name})" for name in WSM6_SPECIES_ORDER],
        "claim_limit": CM1_WSM6_PDEFQ_CLAIM_LIMIT,
    }


def wsm6_pdefq_column_order(*, ni: int, nj: int, stride: int = 1) -> tuple[tuple[int, int], ...]:
    """A deterministic permutation of every ``(i,j)`` column.

    ``stride=1`` is CM1's own order.  Any other stride coprime with ``ni*nj``
    walks the same columns in an order the authority never uses, which is what
    a test needs to stand in for a device that schedules blocks arbitrarily.
    """

    for name, value in (("ni", ni), ("nj", nj), ("stride", stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    count = ni * nj
    if math.gcd(stride, count) != 1:
        raise ValueError("stride must be coprime with ni*nj to visit every column once")
    # The linear index is CM1's own: ``j`` outer, ``i`` inner, so stride 1
    # reproduces misclibs.F:1062-1064 exactly.
    return tuple(
        tuple(reversed(divmod((index * stride) % count, ni)))
        for index in range(count)
    )


def _foreign_column_stride(*, ni: int, nj: int) -> int:
    """The smallest stride above one that still visits every column once."""

    count = ni * nj
    for stride in range(2, count):
        if math.gcd(stride, count) == 1:
            return stride
    return 1


def cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
    column_order: Any = None,
) -> dict[str, Any]:
    """CPU oracle for the column-parallel device layout, in three phases.

    Phase one evaluates every ``(i,j)`` column of misclibs.F:1065-1081 in
    ``column_order`` -- by default an order the authority never uses -- and
    records the ``(a1,a2)`` pair each one produced.  Phase two runs
    misclibs.F:1082's carry as a west-to-east scan, serial over ``i`` and still
    left-associated.  Phase three is misclibs.F:1086-1088, species then
    south-to-north ``j``.  Phases two and three keep the authority's order
    exactly; only phase one is reordered, and only because a column carries
    nothing to the next one.
    """

    require_wsm6_pdefq_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni=ni, nj=nj, nk=nk)
    values = tuple(np.float32(value) for value in (dx, dy, dz))
    if not all(bool(np.isfinite(value)) and value > np.float32(0.0) for value in values):
        raise ValueError("dx, dy, and dz must be positive finite float32")
    volume32 = _m(np, _m(np, values[0], values[1]), values[2])
    order = (
        wsm6_pdefq_column_order(ni=ni, nj=nj, stride=_foreign_column_stride(ni=ni, nj=nj))
        if column_order is None
        else tuple((int(i), int(j)) for i, j in column_order)
    )
    if sorted(order) != sorted((i, j) for j in range(nj) for i in range(ni)):
        raise ValueError("column_order must visit every (i,j) column exactly once")
    budgets = np.empty((6, nj), dtype=np.float64)
    pairs = np.empty((6, 2, ni, nj), dtype=np.float64)
    corrected_words = 0
    for species in range(6):
        q = a["q3d"][species]
        threshold = np.float32(0.0 if species == 0 else selection.qsmall)
        for i, j in order:
            a1, a2, corrected = _pdefq_column(
                np, a, q, i=i, j=j, nk=nk, threshold=threshold
            )
            corrected_words += corrected
            pairs[species, 0, i, j] = a1
            pairs[species, 1, i, j] = a2
        for j in range(nj):
            budj = 0.0
            for i in range(ni):
                budj = (budj + pairs[species, 1, i, j]) - pairs[species, 0, i, j]
            budgets[species, j] = budj
    for species in range(6):
        for j in range(nj):
            a["asq"][species] += budgets[species, j] * float(volume32)
    return {
        "schema": CUDA_WSM6_PDEFQ_SCHEMA,
        "corrected_input_words": corrected_words,
        "budget_columns": budgets,
        "column_pairs": pairs,
        "column_order_is_source_order": tuple(order)
        == wsm6_pdefq_column_order(ni=ni, nj=nj),
        "source_order": [f"pdefq({name})" for name in WSM6_SPECIES_ORDER],
        "claim_limit": CM1_WSM6_PDEFQ_CLAIM_LIMIT,
    }


CUDA_WSM6_PDEFQ_SOURCE = r'''// gpuwm-cm1.cuda-wsm6-pdefq-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(double)==8,"float64 required");
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long I(unsigned s,unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj,unsigned nk){return (((unsigned long long)s*nk+k)*nj+j)*ni+i;}
extern "C" __global__ void cm1_wsm6_pdefq_columns_f32_v1(unsigned species,unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,const float*rmh,const float*rho,float*q3d,double*budgets){unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nj)return;float rmax=species?1.0e-14f:0.0f;double budj=0.0;for(unsigned i=0;i<ni;++i){double t1=0.0,t2=0.0,a1=0.0,a2=0.0;for(unsigned k=0;k<nk;++k){unsigned long long z=I(species,k,j,i,ni,nj,nk),p=((unsigned long long)k*nj+j)*ni+i;float q=q3d[z];t1=__dadd_rn(t1,double(M(rho[p],q)));float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh[p]);a1=__dadd_rn(a1,double(cell));if(q<rmax){q=0.0f;q3d[z]=q;}t2=__dadd_rn(t2,double(M(rho[p],q)));}double t3=__ddiv_rn(__dadd_rn(t1,1.0e-20),__dadd_rn(t2,1.0e-20));if(t3<0.0)t3=1.0;for(unsigned k=0;k<nk;++k){unsigned long long z=I(species,k,j,i,ni,nj,nk),p=((unsigned long long)k*nj+j)*ni+i;float q=__double2float_rn(__dmul_rn(t3,double(q3d[z])));q3d[z]=q;float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh[p]);a2=__dadd_rn(a2,double(cell));}budj=__dsub_rn(__dadd_rn(budj,a2),a1);}budgets[(unsigned long long)species*nj+j]=budj;}
extern "C" __global__ void cm1_wsm6_pdefq_finish_f64_v1(float volume,unsigned nj,const double*budgets,double*asq){if(blockIdx.x||threadIdx.x)return;for(unsigned s=0;s<6u;++s)for(unsigned j=0;j<nj;++j)asq[s]=__dadd_rn(asq[s],__dmul_rn(budgets[(unsigned long long)s*nj+j],double(volume)));}
'''
CUDA_WSM6_PDEFQ_SOURCE_BYTES = CUDA_WSM6_PDEFQ_SOURCE.encode("utf-8")
CUDA_WSM6_PDEFQ_SOURCE_SHA256 = "38fb779397c513a4ceb5288e1519188b11315fea636208430c4189bb4dde7522"


CUDA_WSM6_PDEFQ_COLUMN_SOURCE = r'''// gpuwm-cm1.cuda-wsm6-pdefq-column-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(double)==8,"float64 required");
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long I(unsigned s,unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj,unsigned nk){return (((unsigned long long)s*nk+k)*nj+j)*ni+i;}
extern "C" __global__ void cm1_wsm6_pdefq_column_f32_v1(unsigned species,unsigned ni,unsigned nj,unsigned nk,const float*ruh,const float*rvh,const float*rmh,const float*rho,float*q3d,double*pairs){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ni)return;unsigned j=blockIdx.y;float rmax=species?1.0e-14f:0.0f;double t1=0.0,t2=0.0,a1=0.0,a2=0.0;for(unsigned k=0;k<nk;++k){unsigned long long z=I(species,k,j,i,ni,nj,nk),p=((unsigned long long)k*nj+j)*ni+i;float q=q3d[z];t1=__dadd_rn(t1,double(M(rho[p],q)));float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh[p]);a1=__dadd_rn(a1,double(cell));if(q<rmax){q=0.0f;q3d[z]=q;}t2=__dadd_rn(t2,double(M(rho[p],q)));}double t3=__ddiv_rn(__dadd_rn(t1,1.0e-20),__dadd_rn(t2,1.0e-20));if(t3<0.0)t3=1.0;for(unsigned k=0;k<nk;++k){unsigned long long z=I(species,k,j,i,ni,nj,nk),p=((unsigned long long)k*nj+j)*ni+i;float q=__double2float_rn(__dmul_rn(t3,double(q3d[z])));q3d[z]=q;float cell=M(M(M(M(rho[p],q),ruh[i]),rvh[j]),rmh[p]);a2=__dadd_rn(a2,double(cell));}unsigned long long o=(unsigned long long)i*nj+j;pairs[o]=a1;pairs[(unsigned long long)ni*nj+o]=a2;}
extern "C" __global__ void cm1_wsm6_pdefq_rowscan_f64_v1(unsigned species,unsigned ni,unsigned nj,const double*pairs,double*budgets){unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nj)return;unsigned long long n=(unsigned long long)ni*nj;double budj=0.0;for(unsigned i=0;i<ni;++i){unsigned long long o=(unsigned long long)i*nj+j;budj=__dsub_rn(__dadd_rn(budj,pairs[n+o]),pairs[o]);}budgets[(unsigned long long)species*nj+j]=budj;}
'''
CUDA_WSM6_PDEFQ_COLUMN_SOURCE_BYTES = CUDA_WSM6_PDEFQ_COLUMN_SOURCE.encode("utf-8")
CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256 = "e7b8e12924163bfefe9a17181b6fb2ade5401b6e1661f443bf932fce88a6f36c"


@dataclass(frozen=True, slots=True)
class CompiledWsm6Pdefq:
    module_owner: Any
    kernels: tuple[Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]
    # The column-parallel layout's own translation unit.  It is a second
    # module rather than two more kernels in the pinned one so that the
    # transcription every recorded receipt names by ``source_sha256`` stays
    # byte-identical; the pinned ``finish`` kernel above is shared, because
    # misclibs.F:1086-1088's south-to-north asq sum is the same in both.
    column_module_owner: Any = None
    column_kernels: tuple[Any, Any] | None = None
    column_ptx: bytes = b""
    column_cubin: bytes = b""


def _compile_unit(
    compiler: Any,
    source: str,
    stem: str,
    kernel_names: tuple[str, ...],
    ptx_options: Any,
    cubin_options: Any,
) -> tuple[bytes, bytes]:
    """Build one translation unit and prove every named entry survived it."""

    ptx, _ = compiler._NVRTCProgram(source, f"{stem}_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(source, f"{stem}_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    for name in kernel_names:
        if not re.search(rf"\.entry\s+{name}(?:\s|\()", decoded):
            raise ValueError(f"compiled PTX lacks {name}")
    return ptx, cubin


def compile_cuda_wsm6_pdefq() -> CompiledWsm6Pdefq:
    for source_bytes, pinned, label in (
        (CUDA_WSM6_PDEFQ_SOURCE_BYTES, CUDA_WSM6_PDEFQ_SOURCE_SHA256, "WSM6 pdefq"),
        (CUDA_WSM6_PDEFQ_COLUMN_SOURCE_BYTES, CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256, "WSM6 pdefq column"),
    ):
        actual = hashlib.sha256(source_bytes).hexdigest()
        if actual != pinned:
            raise RuntimeError(f"packaged {label} CUDA source differs from pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, cubin = _compile_unit(compiler, CUDA_WSM6_PDEFQ_SOURCE, "cm1_wsm6_pdefq", CUDA_WSM6_PDEFQ_KERNEL_NAMES, ptx_options, cubin_options)
    column_ptx, column_cubin = _compile_unit(compiler, CUDA_WSM6_PDEFQ_COLUMN_SOURCE, "cm1_wsm6_pdefq_column", CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES, ptx_options, cubin_options)
    module = function.Module()
    module.load(cubin)
    column_module = function.Module()
    column_module.load(column_cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledWsm6Pdefq(
        module,
        tuple(module.get_function(name) for name in CUDA_WSM6_PDEFQ_KERNEL_NAMES),
        ptx,
        cubin,
        arch,
        identity,
        column_module,
        tuple(column_module.get_function(name) for name in CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES),
        column_ptx,
        column_cubin,
    )


@dataclass(frozen=True, slots=True)
class Wsm6PdefqWorkspace:
    ni: int
    nj: int
    nk: int
    budget_columns: Any
    layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT
    # ``column-parallel`` only.  ``(2, ni, nj)`` float64: the per-column ``a1``
    # then the per-column ``a2`` of misclibs.F:1071 and 1080, laid out with the
    # physical row fastest so the ordered west-to-east scan that consumes them
    # reads one coalesced float64 per thread.
    column_pairs: Any = None


def wsm6_pdefq_workspace_bytes(*, ni: int, nj: int, nk: int, layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT) -> int:
    """Device bytes one layout's workspace costs, priced without a device."""

    require_wsm6_pdefq_layout(layout)
    wsm6_pdefq_shapes(ni, nj, nk)
    budget = 6 * nj * 8
    if layout == WSM6_PDEFQ_ROW_SERIAL_LAYOUT:
        return budget
    return budget + 2 * ni * nj * 8


def create_cuda_wsm6_pdefq_workspace(
    *, ni: int, nj: int, nk: int, layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT
) -> Wsm6PdefqWorkspace:
    wsm6_pdefq_shapes(ni, nj, nk)
    # Refuses a domain the layout cannot launch before any device memory is
    # taken for it.
    wsm6_pdefq_launch_plan(ni=ni, nj=nj, nk=nk, layout=layout)
    cupy = importlib.import_module("cupy")
    pairs = (
        None
        if layout == WSM6_PDEFQ_ROW_SERIAL_LAYOUT
        else cupy.empty((2, ni, nj), dtype=cupy.float64)
    )
    return Wsm6PdefqWorkspace(
        ni, nj, nk, cupy.empty((6, nj), dtype=cupy.float64), layout, pairs
    )


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    shapes = wsm6_pdefq_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA WSM6 pdefq arrays: {', '.join(missing)}")
    result = {}
    for name, shape in shapes.items():
        value = arrays[name]
        dtype = cupy.float64 if name == "asq" else cupy.float32
        if not isinstance(value, cupy.ndarray) or value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-order CUDA {dtype} shape {shape}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA WSM6 pdefq arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaWsm6PdefqLaunch:
    ready_event: Any
    compiled: CompiledWsm6Pdefq
    workspace: Wsm6PdefqWorkspace
    receipt: dict[str, Any]


def launch_cuda_wsm6_pdefq_pdscheme1(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: Wsm6PdefqSelection = DEFAULT_WSM6_PDEFQ_SELECTION,
    layout: str = DEFAULT_WSM6_PDEFQ_LAYOUT,
    stream: Any | None = None,
    compiled: CompiledWsm6Pdefq | None = None,
    workspace: Wsm6PdefqWorkspace | None = None,
) -> CudaWsm6PdefqLaunch:
    """Mutate live q3d and asq at the exact pre-WSM6 large-step seam."""

    require_wsm6_pdefq_selection(selection)
    require_wsm6_pdefq_layout(layout)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    values = tuple(cupy.float32(value) for value in (dx, dy, dz))
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
        raise ValueError("dx, dy, and dz must be positive finite float32")
    volume = cupy.float32(cupy.float32(values[0] * values[1]) * values[2])
    plan = wsm6_pdefq_launch_plan(ni=ni, nj=nj, nk=nk, layout=layout)
    compiled = compile_cuda_wsm6_pdefq() if compiled is None else compiled
    workspace = create_cuda_wsm6_pdefq_workspace(ni=ni, nj=nj, nk=nk, layout=layout) if workspace is None else workspace
    if not isinstance(workspace, Wsm6PdefqWorkspace) or (workspace.ni, workspace.nj, workspace.nk) != (ni, nj, nk):
        raise ValueError("WSM6 pdefq workspace geometry differs")
    if workspace.layout != layout:
        raise ValueError(f"WSM6 pdefq workspace was built for layout {workspace.layout!r}, not {layout!r}")
    if not isinstance(workspace.budget_columns, cupy.ndarray) or workspace.budget_columns.dtype != cupy.float64 or workspace.budget_columns.shape != (6, nj) or not workspace.budget_columns.flags.c_contiguous:
        raise ValueError("WSM6 pdefq budget workspace has wrong layout")
    input_pointers = {int(value.data.ptr) for value in a.values()}
    if int(workspace.budget_columns.data.ptr) in input_pointers:
        raise ValueError("WSM6 pdefq workspace aliases a live carrier")
    pairs = workspace.column_pairs
    if layout == WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT:
        if not isinstance(pairs, cupy.ndarray) or pairs.dtype != cupy.float64 or pairs.shape != (2, ni, nj) or not pairs.flags.c_contiguous:
            raise ValueError("WSM6 pdefq column-pair workspace has wrong layout")
        if int(pairs.data.ptr) in input_pointers or int(pairs.data.ptr) == int(workspace.budget_columns.data.ptr):
            raise ValueError("WSM6 pdefq workspace aliases a live carrier")
        if compiled.column_kernels is None:
            raise ValueError("this compiled WSM6 pdefq module carries no column-parallel kernels")
    elif pairs is not None:
        raise ValueError("the row-serial layout takes no column-pair workspace")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    finish = compiled.kernels[1]
    species_words = tuple(
        (cupy.uint32(species), cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk))
        for species in range(6)
    )
    with stream:
        if layout == WSM6_PDEFQ_ROW_SERIAL_LAYOUT:
            rows = compiled.kernels[0]
            grid, block = tuple(plan["stages"][0]["grid"]), tuple(plan["stages"][0]["block"])
            for species in range(6):
                rows(
                    grid,
                    block,
                    (
                        *species_words[species],
                        a["ruh"], a["rvh"], a["rmh"], a["rho"], a["q3d"], workspace.budget_columns,
                    ),
                    stream=stream,
                )
        else:
            column, rowscan = compiled.column_kernels
            column_grid, column_block = tuple(plan["stages"][0]["grid"]), tuple(plan["stages"][0]["block"])
            scan_grid, scan_block = tuple(plan["stages"][1]["grid"]), tuple(plan["stages"][1]["block"])
            for species in range(6):
                column(
                    column_grid,
                    column_block,
                    (
                        *species_words[species],
                        a["ruh"], a["rvh"], a["rmh"], a["rho"], a["q3d"], pairs,
                    ),
                    stream=stream,
                )
                # misclibs.F:1082's carry, still west-to-east and still
                # left-associated, over the pairs the columns just wrote.
                rowscan(
                    scan_grid,
                    scan_block,
                    (cupy.uint32(species), cupy.uint32(ni), cupy.uint32(nj), pairs, workspace.budget_columns),
                    stream=stream,
                )
        finish((1, 1, 1), (1, 1, 1), (volume, cupy.uint32(nj), workspace.budget_columns, a["asq"]), stream=stream)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_WSM6_PDEFQ_SCHEMA,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "geometry": {"ni": ni, "nj": nj, "nk": nk, "dx": float(values[0]), "dy": float(values[1]), "dz": float(values[2])},
        "cadence": "once after final-RK dynamics/apmasscon and before WSM6; solve3 later copies q3d to qa",
        "mutated": ["q3d[0:6]", "asq[0:6]"],
        "kernel": {"abi": CUDA_WSM6_PDEFQ_ABI, "source_sha256": CUDA_WSM6_PDEFQ_SOURCE_SHA256, "target_arch": compiled.target_arch},
        "launch": plan,
        "layout_kernel": (
            {"abi": CUDA_WSM6_PDEFQ_COLUMN_ABI, "source_sha256": CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256}
            if layout == WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT
            else {"abi": CUDA_WSM6_PDEFQ_ABI, "source_sha256": CUDA_WSM6_PDEFQ_SOURCE_SHA256}
        ),
        "workspace_bytes": wsm6_pdefq_workspace_bytes(ni=ni, nj=nj, nk=nk, layout=layout),
        "runtime": dict(compiled.runtime_identity),
        "authority": {"tarball_sha256": AUTHORITY_TARBALL_SHA256, "source_sha256": AUTHORITY_SOURCE_SHA256, "anchors": SOURCE_ANCHORS},
        "claim_limit": CM1_WSM6_PDEFQ_CLAIM_LIMIT,
    }
    return CudaWsm6PdefqLaunch(event, compiled, workspace, receipt)


__all__ = (
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_WSM6_PDEFQ_CLAIM_LIMIT",
    "CUDA_MAX_GRID_DIM_Y",
    "CUDA_WSM6_PDEFQ_ABI",
    "CUDA_WSM6_PDEFQ_COLUMN_ABI",
    "CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES",
    "CUDA_WSM6_PDEFQ_COLUMN_SOURCE",
    "CUDA_WSM6_PDEFQ_COLUMN_SOURCE_BYTES",
    "CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256",
    "CUDA_WSM6_PDEFQ_KERNEL_NAMES",
    "CUDA_WSM6_PDEFQ_SCHEMA",
    "CUDA_WSM6_PDEFQ_SOURCE_BYTES",
    "CUDA_WSM6_PDEFQ_SOURCE_SHA256",
    "CompiledWsm6Pdefq",
    "CudaWsm6PdefqLaunch",
    "DEFAULT_WSM6_PDEFQ_LAYOUT",
    "DEFAULT_WSM6_PDEFQ_SELECTION",
    "SOURCE_ANCHORS",
    "WSM6_PDEFQ_BLOCK_THREADS",
    "WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT",
    "WSM6_PDEFQ_LAYOUTS",
    "WSM6_PDEFQ_ROW_SERIAL_LAYOUT",
    "WSM6_SPECIES_ORDER",
    "Wsm6PdefqSelection",
    "Wsm6PdefqWorkspace",
    "compile_cuda_wsm6_pdefq",
    "cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace",
    "cpu_wsm6_pdefq_pdscheme1_inplace",
    "create_cuda_wsm6_pdefq_workspace",
    "launch_cuda_wsm6_pdefq_pdscheme1",
    "require_wsm6_pdefq_layout",
    "require_wsm6_pdefq_selection",
    "wsm6_pdefq_column_order",
    "wsm6_pdefq_launch_plan",
    "wsm6_pdefq_shapes",
    "wsm6_pdefq_workspace_bytes",
)
