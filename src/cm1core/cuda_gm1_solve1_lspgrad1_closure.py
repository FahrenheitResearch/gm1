"""CM1 ``lspgrad=1`` in the live Cartesian closure, parent and nest child.

``src/solve1.F:590-612`` is the whole operator.  At every ``u`` point it
subtracts ``fcor`` times (the four-point average of ``v0`` around the point,
plus ``vmove``) from ``uten1``, and at every ``v`` point it adds ``fcor`` times
(the four-point average of ``u0`` around the point, plus ``umove``) to
``vten1``; each average is a quarter of the sum of two pair sums.  It runs
over ``k=1..nk``, ``j=1..nj+1``, ``i=1..ni+1``, immediately after the Rayleigh
block has assigned ``uten1``/``vten1`` (``solve1.F:414-446``) and before any
diffusion, dissipative heating or turbulence adds to them (``solve1.F:686``,
``:920-1001``, ``:1116`` onward).  With the f-plane Coriolis term of
``solve2.F:639-653``, ``fcor*(V + Vmove)``, the base state then feels no net
rotation: the run integrates ``f x (V - V0)``.  ``u0``/``v0`` are frame-relative
(``base.F:2661-2674``), which is why both lines add the frame velocity back.

Every sounding this lane initializes (isnd=5 and isnd=7, ``base.F``) builds
``u0``/``v0`` from one column, so they are horizontally uniform including
their halos, and the four-point average of one word ``a`` is
``0.25*((a+a)+(a+a))`` -- one word per level.  This module builds those two
per-level profiles ONCE on the host, in CM1's own default-REAL operation
order, and refuses to hand them to a kernel unless they are raw-word identical
to :func:`cm1core.large_scale_pressure_gradient.cpu_large_scale_pressure_gradient_inplace`
-- the operator ``evidence/oracles/lspgrad-source-exact-v1`` proves against
gfortran -- applied to a uniform compact window of the same column.  The
per-step work is the kernel below: one subtraction on every u face and one
addition on every v face of the tile.

Nothing here is a claim that the coupled trajectory matches CM1.  That is the
no-storm balance-column differential (``tools/balance_column_differential.py``),
which needs a device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .large_scale_pressure_gradient import (
    LargeScalePressureGradientSelection,
    cpu_large_scale_pressure_gradient_inplace,
)


CARTESIAN_LSPGRAD1_PROFILE_SCHEMA = "gm1.cartesian-lspgrad1-profiles/v1"
CARTESIAN_LSPGRAD1_AUTHORITY = (
    "CM1 r21.1 src/solve1.F:590-612 (lspgrad=1: Coriolis force of the "
    "base-state wind removed from uten1/vten1), src/solve1.F:414-446 (the "
    "Rayleigh assignment it follows), README.namelist:531-538, "
    "base.F:2661-2674 (frame-relative u0/v0)"
)
CARTESIAN_LSPGRAD1_CLAIM_LIMIT = (
    "per-level lspgrad=1 profiles raw-word identical to the gfortran-proven "
    "CPU operator on a horizontally uniform column, and one CUDA kernel that "
    "applies them on every owned u/v face after the Rayleigh assignment; no "
    "CM1 coupled-trajectory differential (the balance column is a node run)"
)
#: The source order CM1 puts this term in, as the closure receipt records it.
SOURCE_ORDER_ENTRY = "lspgrad=1 perturbation-Coriolis large-scale pressure gradient"

CUDA_GM1_LSPGRAD1_KERNEL_NAME = "gm1_solve1_lspgrad1_base_state_f32_v1"
#: Every operation is an ``_rn`` intrinsic, so no contraction or
#: reassociation can reach it.  The profile index is the level of the flat
#: word: ``uten1`` is ``(nk, nj, ni+1)`` and ``vten1`` is ``(nk, nj+1, ni)``,
#: the tile carriers the Rayleigh kernel just assigned.
CUDA_GM1_LSPGRAD1_SOURCE = r"""
__device__ __forceinline__ float cm1_add(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float cm1_sub(const float a,const float b){return __fsub_rn(a,b);}

extern "C" __global__ void gm1_solve1_lspgrad1_base_state_f32_v1(
 const float* __restrict__ ul,const float* __restrict__ vl,
 float* __restrict__ uten1,float* __restrict__ vten1,
 unsigned int ni,unsigned int nj,unsigned int nk)
{
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned long long plane_u=(unsigned long long)nj*(ni+1u);
 const unsigned long long plane_v=(unsigned long long)(nj+1u)*ni;
 const unsigned long long nu=(unsigned long long)nk*plane_u;
 const unsigned long long nv=(unsigned long long)nk*plane_v;
 /* src/solve1.F:606-607: uten1 = uten1 - fcor*(0.25*(...) + vmove) */
 for(unsigned long long z=first;z<nu;z+=step){
  uten1[z]=cm1_sub(uten1[z],ul[(unsigned int)(z/plane_u)]);
 }
 /* src/solve1.F:608-609: vten1 = vten1 + fcor*(0.25*(...) + umove) */
 for(unsigned long long z=first;z<nv;z+=step){
  vten1[z]=cm1_add(vten1[z],vl[(unsigned int)(z/plane_v)]);
 }
}
"""
CUDA_GM1_LSPGRAD1_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_LSPGRAD1_SOURCE.encode("ascii")
).hexdigest()
CUDA_GM1_LSPGRAD1_THREADS = 256
#: The options every neighbouring closure kernel compiles with: no FMA, no
#: flush-to-zero, IEEE division.
CUDA_GM1_LSPGRAD1_COMPILE_OPTIONS = (
    "--std=c++14",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


@dataclass(frozen=True, slots=True)
class CartesianLspgrad1Profiles:
    """``ul[k]`` subtracted from ``uten1`` and ``vl[k]`` added to ``vten1``."""

    ul: Any
    vl: Any
    receipt: Mapping[str, object]


def _np() -> Any:
    return importlib.import_module("numpy")


def _sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _profile(np: Any, name: str, value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional float32 profile")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} carries a non-finite base-state wind word")
    return np.ascontiguousarray(array)


def build_cartesian_lspgrad1_profiles(
    u0: Any,
    v0: Any,
    *,
    fcor_s_inv: float,
    u_move_m_s: float,
    v_move_m_s: float,
) -> CartesianLspgrad1Profiles:
    """``solve1.F:606-609`` on one horizontally uniform, frame-relative column.

    ``u0``/``v0`` are the compact ``(nk,)`` base-state winds the closure damps
    toward.  Each profile word is formed in CM1's order with default-REAL
    operands -- ``s = a + a; s = s + s; s = 0.25*s; s = s + move; fcor*s`` --
    and the pair is then checked, raw word for raw word, against the proven
    CPU operator on a uniform ``(nk, 3, 3)`` window, including the
    application to a nonzero held tendency.  A mismatch is a refusal: the
    kernel would otherwise integrate a forcing no oracle has seen.
    """

    np = _np()
    u0 = _profile(np, "u0", u0)
    v0 = _profile(np, "v0", v0)
    if u0.shape != v0.shape:
        raise ValueError("u0 and v0 must share one vertical level count")
    nk = int(u0.shape[0])
    f32 = np.float32
    fcor, umove, vmove = f32(fcor_s_inv), f32(u_move_m_s), f32(v_move_m_s)
    for name, value in (("fcor", fcor), ("umove", umove), ("vmove", vmove)):
        if not np.isfinite(value):
            raise ValueError(f"{name} is not a finite CM1 default REAL")

    def column(base: Any, move: Any) -> Any:
        out = np.empty_like(base)
        np.add(base, base, out=out)
        np.add(out, out, out=out)
        np.multiply(f32(0.25), out, out=out)
        np.add(out, move, out=out)
        np.multiply(fcor, out, out=out)
        return out

    ul = column(v0, vmove)
    vl = column(u0, umove)

    # The authority check.  ni=nj=1 is the smallest window whose loop visits
    # both staggered stencils; the initial tendencies are nonzero so the
    # subtraction and addition themselves are compared too.
    ni = nj = 1
    ramp = np.arange(nk * (nj + 1) * (ni + 1), dtype=np.float64).reshape(nk, nj + 1, ni + 1)
    initial_u = (1.0e-3 * np.sin(ramp + 0.25)).astype(np.float32)
    initial_v = (1.0e-3 * np.cos(ramp + 0.75)).astype(np.float32)
    workspace = {
        "uten1": initial_u.copy(),
        "vten1": initial_v.copy(),
        "ulspg": np.zeros((nk,), dtype=np.float32),
        "vlspg": np.zeros((nk,), dtype=np.float32),
        "u0_window": np.ascontiguousarray(
            np.broadcast_to(u0[:, None, None], (nk, nj + 2, ni + 2)), dtype=np.float32
        ),
        "v0_window": np.ascontiguousarray(
            np.broadcast_to(v0[:, None, None], (nk, nj + 2, ni + 2)), dtype=np.float32
        ),
    }
    operator = cpu_large_scale_pressure_gradient_inplace(
        workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=LargeScalePressureGradientSelection(
            lspgrad=1, fcor=float(fcor), umove=float(umove), vmove=float(vmove)
        ),
    )
    expected_u = np.subtract(initial_u, ul[:, None, None])
    expected_v = np.add(initial_v, vl[:, None, None])
    mismatched = int(
        np.count_nonzero(workspace["uten1"].view(np.uint32) != expected_u.view(np.uint32))
        + np.count_nonzero(workspace["vten1"].view(np.uint32) != expected_v.view(np.uint32))
    )
    if mismatched:
        raise RuntimeError(
            f"the lspgrad=1 profiles differ from solve1.F:590-612's operator in "
            f"{mismatched} words; the closure will not integrate a forcing the "
            "source-exact operator does not produce"
        )
    if not (np.all(np.isfinite(ul)) and np.all(np.isfinite(vl))):
        raise ValueError("the lspgrad=1 profiles are not finite")
    receipt: dict[str, object] = {
        "schema": CARTESIAN_LSPGRAD1_PROFILE_SCHEMA,
        "lspgrad": 1,
        "authority": CARTESIAN_LSPGRAD1_AUTHORITY,
        "claim_limit": CARTESIAN_LSPGRAD1_CLAIM_LIMIT,
        "nk": nk,
        "fcor_f32_bits": int(fcor.view(np.uint32)),
        "umove_f32_bits": int(umove.view(np.uint32)),
        "vmove_f32_bits": int(vmove.view(np.uint32)),
        "base_state_sha256": {"u0": _sha256(u0), "v0": _sha256(v0)},
        "profile_sha256": {"ul": _sha256(ul), "vl": _sha256(vl)},
        "operator_identity": {
            "operator": "cm1core.large_scale_pressure_gradient.cpu_large_scale_pressure_gradient_inplace",
            "window_zyx": [nk, nj + 2, ni + 2],
            "mismatched_words": 0,
            "operator_operation_token_sha256": operator.receipt["operation_token_sha256"],
        },
        "horizontal_uniformity": (
            "u0/v0 are the compact column every admitted sounding builds "
            "(isnd=5/7), so the four-point average is exact per level"
        ),
        "kernel_source_sha256": CUDA_GM1_LSPGRAD1_SOURCE_SHA256,
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return CartesianLspgrad1Profiles(ul=ul, vl=vl, receipt=MappingProxyType(receipt))


def cpu_lspgrad1_apply(uten1: Any, vten1: Any, ul: Any, vl: Any) -> None:
    """The host statement of what the kernel does to one tile, in place."""

    np = _np()
    np.subtract(uten1, ul[:, None, None], out=uten1)
    np.add(vten1, vl[:, None, None], out=vten1)


@dataclass(frozen=True, slots=True)
class CudaGm1Lspgrad1Compiled:
    """One compiled ``lspgrad=1`` kernel and the text it was built from."""

    kernel: Any
    source_sha256: str


def compile_cuda_gm1_lspgrad1() -> CudaGm1Lspgrad1Compiled:
    """Compile the live ``lspgrad=1`` kernel from its pinned text."""

    if (
        hashlib.sha256(CUDA_GM1_LSPGRAD1_SOURCE.encode("ascii")).hexdigest()
        != CUDA_GM1_LSPGRAD1_SOURCE_SHA256
    ):
        raise RuntimeError("local lspgrad=1 kernel source differs from its pin")
    # nvrtc_module: cupy.RawModule appends -ftz=true behind the caller's
    # options, so every closure image is built through the NVRTC program
    # directly with gradual underflow, as rterm's is.
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_LSPGRAD1_SOURCE,
        options=CUDA_GM1_LSPGRAD1_COMPILE_OPTIONS,
        name_expressions=(CUDA_GM1_LSPGRAD1_KERNEL_NAME,),
        filename="gm1_lspgrad1.cu",
    )
    return CudaGm1Lspgrad1Compiled(
        kernel=module.get_function(CUDA_GM1_LSPGRAD1_KERNEL_NAME),
        source_sha256=CUDA_GM1_LSPGRAD1_SOURCE_SHA256,
    )


def launch_cuda_gm1_lspgrad1(
    uten1: Any,
    vten1: Any,
    ul: Any,
    vl: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
    compiled: Any,
) -> None:
    """Apply ``solve1.F:606-609`` to one tile's held u/v tendency."""

    cupy = importlib.import_module("cupy")
    for name, value, shape in (
        ("uten1", uten1, (nk, nj, ni + 1)),
        ("vten1", vten1, (nk, nj + 1, ni)),
        ("ul", ul, (nk,)),
        ("vl", vl, (nk,)),
    ):
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 shape {shape}")
    count = max(nk * nj * (ni + 1), nk * (nj + 1) * ni)
    blocks = (count + CUDA_GM1_LSPGRAD1_THREADS - 1) // CUDA_GM1_LSPGRAD1_THREADS
    compiled.kernel(
        (blocks, 1, 1),
        (CUDA_GM1_LSPGRAD1_THREADS, 1, 1),
        (ul, vl, uten1, vten1, cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)),
        stream=stream,
    )


CUDA_GM1_SOLVE1_LSPGRAD1_CLOSURE_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()


__all__ = [
    "CARTESIAN_LSPGRAD1_AUTHORITY",
    "CARTESIAN_LSPGRAD1_CLAIM_LIMIT",
    "CARTESIAN_LSPGRAD1_PROFILE_SCHEMA",
    "CUDA_GM1_LSPGRAD1_COMPILE_OPTIONS",
    "CUDA_GM1_LSPGRAD1_KERNEL_NAME",
    "CUDA_GM1_LSPGRAD1_SOURCE",
    "CUDA_GM1_LSPGRAD1_SOURCE_SHA256",
    "CUDA_GM1_LSPGRAD1_THREADS",
    "CUDA_GM1_SOLVE1_LSPGRAD1_CLOSURE_SOURCE_SHA256",
    "CartesianLspgrad1Profiles",
    "CudaGm1Lspgrad1Compiled",
    "SOURCE_ORDER_ENTRY",
    "build_cartesian_lspgrad1_profiles",
    "compile_cuda_gm1_lspgrad1",
    "cpu_lspgrad1_apply",
    "launch_cuda_gm1_lspgrad1",
]
