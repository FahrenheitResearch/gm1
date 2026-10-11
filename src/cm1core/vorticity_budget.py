"""Eulerian vorticity components and their budget, evaluated inline.

Authority and what is new
-------------------------
The three vorticity *components* are CM1's own: ``src/misclibs.F:1542``
``calcvort``, flat-Cartesian branch -- ``xvort`` at :1618-1683, ``yvort`` at
:1854-1900, ``zvort`` at :2035-2056.  Those lines are transcribed.

The *budget* is not CM1's.  CM1 r21.1 has no tilting, stretching, solenoidal
or frictional vorticity tendency anywhere: ``grep -rn "tilt" src/`` in the
r21.1 tree returns nothing.  It is the standard analysis of every
tornadogenesis paper and it is exactly what a post-hoc calculation from
ten-minute history frames cannot reconstruct, so GM1 computes it at the model
timestep from the resident state.  Every term below is therefore marked
**GM1 EXTENSION** and is validated against an analytic field rather than
against CM1.

The equation, for the compressible vorticity vector
:math:`\\boldsymbol\\omega = \\nabla \\times \\mathbf{V}`:

.. math::

    \\frac{D\\boldsymbol\\omega}{Dt}
      = (\\boldsymbol\\omega\\cdot\\nabla)\\mathbf{V}
      - \\boldsymbol\\omega (\\nabla\\cdot\\mathbf{V})
      + \\frac{1}{\\rho^{2}}\\nabla\\rho\\times\\nabla p
      + \\nabla\\times\\mathbf{F}

published term by term, per component:

``stretching``   the like-with-like part of :math:`(\\omega\\cdot\\nabla)V`
``tilting``      the cross terms of :math:`(\\omega\\cdot\\nabla)V`
``divergence``   :math:`-\\omega_i(\\nabla\\cdot V)`
``solenoid``     the baroclinic generation
``friction``     :math:`\\nabla\\times\\mathbf{F}` for a supplied momentum
                 tendency -- the runtime supplies the held subgrid tendency

Exactness
---------
Every derivative is a centred difference on the C-grid, so for a velocity
field linear in ``(x, y, z)`` every component and every term is exact to
rounding.  That is the validation: :mod:`tests.test_vorticity_budget` builds
a random linear field, computes each term in closed form, and requires the
port to reproduce it.  A solid-body rotation, whose entire budget is
identically zero, is the negative control.

Lateral and vertical edges
--------------------------
The velocity carriers are GM1's padded ones and carry three lateral ghost
cells and the ``parcel.F:57-126`` vertical ghost fill, so the velocity
derivatives are centred everywhere.  ``rho`` and ``prs`` are the physical
``(nz, ny, nx)`` rectangles with no halo, so the solenoid's outermost ring and
its top and bottom rows are one-sided.  The receipt counts both, exactly as
:mod:`cm1core.cuda_storm_diagnostics` does for its own display ring; a reader
never has to guess which words are centred.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY

VORTICITY_BUDGET_SCHEMA = "gpuwm-cm1.vorticity-budget/v1"
VORTICITY_BUDGET_ABI = "gpuwm-cm1.vorticity-budget-kernels/v1"
VORTICITY_BUDGET_THREADS = 128

SOURCE_ANCHORS: Mapping[str, str] = {
    "calcvort": "src/misclibs.F:1542-2100",
    "xvort_flat": "src/misclibs.F:1618-1683",
    "yvort_flat": "src/misclibs.F:1854-1900",
    "zvort_flat": "src/misclibs.F:2035-2056",
    "parcel_zvort": "src/parcel.F:1061-1072",
    "vertical_ghost_fill": "src/parcel.F:57-126",
    "budget": "GM1 EXTENSION: CM1 r21.1 carries no vorticity budget",
}

#: The published field names, in the order the arena writes them.
VORTICITY_COMPONENT_NAMES: tuple[str, ...] = ("xvort", "yvort", "zvort")
VORTICITY_BUDGET_TERM_NAMES: tuple[str, ...] = (
    "stretching", "tilting", "divergence", "solenoid", "friction",
)
VORTICITY_BUDGET_FIELD_NAMES: tuple[str, ...] = tuple(
    f"{term}_{axis}"
    for axis in ("x", "y", "z")
    for term in VORTICITY_BUDGET_TERM_NAMES
)


def _f32(np: Any, value: Any) -> Any:
    return np.float32(value)


@dataclass(frozen=True, slots=True)
class EulerianVorticity:
    xvort: Any
    yvort: Any
    zvort: Any


@dataclass(frozen=True, slots=True)
class VorticityBudget:
    components: EulerianVorticity
    terms: Mapping[str, Any]
    centred_words: int
    one_sided_words: int

    def record(self) -> dict[str, Any]:
        return {
            "schema": VORTICITY_BUDGET_SCHEMA,
            "components": list(VORTICITY_COMPONENT_NAMES),
            "terms": list(VORTICITY_BUDGET_FIELD_NAMES),
            "centred_words": int(self.centred_words),
            "one_sided_solenoid_words": int(self.one_sided_words),
            "cm1_authority": dict(SOURCE_ANCHORS),
        }


def _scalar_u(np: Any, u: Any, ng: int, nz: int, ny: int, nx: int) -> Any:
    """``u`` averaged to scalar points, with one lateral ghost ring kept.

    Shape ``(nz+2, ny+2, nx+2)``: levels ``k=0..nz+1`` and one ghost cell each
    way horizontally, which is what the centred derivatives below need.
    """

    # u(i,j,k) lives at [k, j-1+ng, i-1+ng]; scalar i wants faces i and i+1.
    left = u[:, ng - 1:ng + ny + 1, ng - 1:ng + nx + 1]
    right = u[:, ng - 1:ng + ny + 1, ng:ng + nx + 2]
    return np.float32(np.float32(0.5) * np.float32(left + right))


def _scalar_v(np: Any, v: Any, ng: int, nz: int, ny: int, nx: int) -> Any:
    low = v[:, ng - 1:ng + ny + 1, ng - 1:ng + nx + 1]
    high = v[:, ng:ng + ny + 2, ng - 1:ng + nx + 1]
    return np.float32(np.float32(0.5) * np.float32(low + high))


def _scalar_w(np: Any, w: Any, ng: int, nz: int, ny: int, nx: int) -> Any:
    low = w[0:nz + 2, ng - 1:ng + ny + 1, ng - 1:ng + nx + 1]
    high = w[1:nz + 3, ng - 1:ng + ny + 1, ng - 1:ng + nx + 1]
    return np.float32(np.float32(0.5) * np.float32(low + high))


def _ddx(np: Any, s: Any, rdx2: Any) -> Any:
    """Centred x derivative of a ``(nz+2, ny+2, nx+2)`` ghosted scalar block."""

    return np.float32(np.float32(s[1:-1, 1:-1, 2:] - s[1:-1, 1:-1, :-2]) * rdx2)


def _ddy(np: Any, s: Any, rdy2: Any) -> Any:
    return np.float32(np.float32(s[1:-1, 2:, 1:-1] - s[1:-1, :-2, 1:-1]) * rdy2)


def _ddz(np: Any, s: Any, rdz2: Any, zh: Any = None) -> Any:
    """Centred z derivative; ``zh`` gives the deck's own half levels.

    With ``zh`` absent the spacing is the uniform ``dz`` the caller passed,
    which is what a uniform deck wants.  With ``zh`` -- CM1-indexed
    ``0..nz+1``, so ``zh[k+1] - zh[k-1]`` is the centred span at level ``k`` --
    the derivative is taken over the deck's own levels, which is what CM1's own
    ``calcvort`` does through its sigma metric ``rdz*mf``
    (``src/misclibs.F:1628``).  Either way it is exact for a field linear in
    height.
    """

    difference = np.float32(s[2:, 1:-1, 1:-1] - s[:-2, 1:-1, 1:-1])
    if zh is None:
        return np.float32(difference * rdz2)
    span = np.float32(np.asarray(zh[2:], dtype=np.float32)
                      - np.asarray(zh[:-2], dtype=np.float32))
    return np.float32(difference / span[:, None, None])


def _one_sided_grad(np: Any, s: Any, spacing: Any, axis: int) -> Any:
    """Centred where both neighbours exist, one-sided at the two faces."""

    out = np.empty_like(s)
    n = s.shape[axis]
    if n < 2:
        out[...] = np.float32(0.0)
        return out
    two = np.float32(2.0)
    sl = [slice(None)] * 3

    def take(lo: int, hi: int) -> Any:
        local = list(sl)
        local[axis] = slice(lo, hi)
        return s[tuple(local)]

    def put(lo: int, hi: int, value: Any) -> None:
        local = list(sl)
        local[axis] = slice(lo, hi)
        out[tuple(local)] = value

    put(1, n - 1, np.float32(np.float32(take(2, n) - take(0, n - 2))
                             / np.float32(two * spacing)))
    put(0, 1, np.float32(np.float32(take(1, 2) - take(0, 1)) / spacing))
    put(n - 1, n, np.float32(np.float32(take(n - 1, n) - take(n - 2, n - 1)) / spacing))
    return out


def corner_vertical_vorticity(
    u: Any, v: Any, *, nx: int, ny: int, nz: int, dx: float, dy: float, xp: Any
) -> Any:
    """``src/parcel.F:1064-1066``: ``dv/dx - du/dy`` at the C-grid corners.

    Returns the ``(nz, ny+1, nx+1)`` corner field CM1's ``parcel_interp``
    samples directly, without the four-point average
    ``src/misclibs.F:2053`` takes for the scalar-point ``zvort``.  Taking the
    average first and interpolating second is a different number, so the
    corner field is built and sampled as CM1 builds and samples it.

    ``u`` and ``v`` are the padded carriers: CM1's row ``j`` is y index
    ``j-1+ng`` and CM1's face ``i`` is x index ``i-1+ng``, and levels
    ``1..nz`` are z indices ``1..nz``.
    """

    ng = CM1_OPEN_NGXY
    du = xp.subtract(u[1:nz + 1, ng:ng + ny + 1, ng:ng + nx + 1],
                     u[1:nz + 1, ng - 1:ng + ny, ng:ng + nx + 1])
    dv = xp.subtract(v[1:nz + 1, ng:ng + ny + 1, ng:ng + nx + 1],
                     v[1:nz + 1, ng:ng + ny + 1, ng - 1:ng + nx])
    return xp.subtract(xp.divide(dv, xp.float32(dx)),
                       xp.divide(du, xp.float32(dy)))


def cpu_vorticity_budget(
    *,
    u: Any,
    v: Any,
    w: Any,
    rho: Any = None,
    prs: Any = None,
    uten: Any = None,
    vten: Any = None,
    wten: Any = None,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    zh: Any = None,
) -> VorticityBudget:
    """Components and budget on the physical ``(nz, ny, nx)`` scalar rectangle.

    ``u``/``v``/``w`` are the padded C-grid carriers with the
    ``parcel.F:57-126`` vertical ghost fill already applied.  ``rho``/``prs``
    are the physical scalar rectangles; omit them and the solenoid is zero and
    the receipt says so.  ``uten``/``vten``/``wten`` are the momentum tendency
    whose curl the friction term is; omit them and the friction term is zero.

    ``zh`` is the deck's CM1-indexed half levels (``0..nz+1``).  Given, the
    three vertical velocity derivatives are taken over the deck's own levels
    rather than over one uniform ``dz``, which is what a stretched deck needs
    and what CM1's ``calcvort`` itself does through ``rdz*mf``
    (``src/misclibs.F:1628``).  It is refused together with ``rho``/``prs``:
    the solenoid's vertical gradient below is one-sided over a uniform ``dz``,
    so a stretched deck would mis-scale it while the velocity terms were
    correct, and a budget whose terms carried two different vertical metrics
    would be worse than one that refuses.
    """

    if zh is not None and (rho is not None or prs is not None):
        raise ValueError(
            "cpu_vorticity_budget's solenoid is formed over one uniform dz, so it cannot be "
            "combined with a stretched deck's zh; form the solenoid on the deck's own levels "
            "instead (cm1core.nest_window_budget.centred_solenoid)"
        )

    np = importlib.import_module("numpy")
    ng = CM1_OPEN_NGXY
    f = np.float32
    dxf, dyf, dzf = f(dx), f(dy), f(dz)
    rdx, rdy, rdz = f(f(1.0) / dxf), f(f(1.0) / dyf), f(f(1.0) / dzf)
    rdx2, rdy2, rdz2 = f(f(0.5) * rdx), f(f(0.5) * rdy), f(f(0.5) * rdz)

    us = _scalar_u(np, u, ng, nz, ny, nx)
    vs = _scalar_v(np, v, ng, nz, ny, nx)
    ws = _scalar_w(np, w, ng, nz, ny, nx)

    dudx = _ddx(np, us, rdx2)
    dudy = _ddy(np, us, rdy2)
    dudz = _ddz(np, us, rdz2, zh)
    dvdx = _ddx(np, vs, rdx2)
    dvdy = _ddy(np, vs, rdy2)
    dvdz = _ddz(np, vs, rdz2, zh)
    dwdx = _ddx(np, ws, rdx2)
    dwdy = _ddy(np, ws, rdy2)
    dwdz = _ddz(np, ws, rdz2, zh)

    # misclibs.F:1618-1683 / :1854-1900 / :2035-2056, at scalar points.
    xvort = f(dwdy - dvdz)
    yvort = f(dudz - dwdx)
    zvort = f(dvdx - dudy)
    divergence = f(f(dudx + dvdy) + dwdz)

    terms: dict[str, Any] = {}
    # (omega . grad) V, split like-with-like / cross.
    terms["stretching_x"] = f(xvort * dudx)
    terms["tilting_x"] = f(f(yvort * dudy) + f(zvort * dudz))
    terms["divergence_x"] = f(-f(xvort * divergence))
    terms["stretching_y"] = f(yvort * dvdy)
    terms["tilting_y"] = f(f(xvort * dvdx) + f(zvort * dvdz))
    terms["divergence_y"] = f(-f(yvort * divergence))
    terms["stretching_z"] = f(zvort * dwdz)
    terms["tilting_z"] = f(f(xvort * dwdx) + f(yvort * dwdy))
    terms["divergence_z"] = f(-f(zvort * divergence))

    zeros = np.zeros((nz, ny, nx), dtype=np.float32)
    one_sided = 0
    if rho is None or prs is None:
        for axis in ("x", "y", "z"):
            terms[f"solenoid_{axis}"] = zeros.copy()
    else:
        rho = np.ascontiguousarray(rho, dtype=np.float32)
        prs = np.ascontiguousarray(prs, dtype=np.float32)
        if rho.shape != (nz, ny, nx) or prs.shape != (nz, ny, nx):
            raise ValueError("rho and prs must be the (nz, ny, nx) scalar rectangles")
        drdx = _one_sided_grad(np, rho, dxf, 2)
        drdy = _one_sided_grad(np, rho, dyf, 1)
        drdz = _one_sided_grad(np, rho, dzf, 0)
        dpdx = _one_sided_grad(np, prs, dxf, 2)
        dpdy = _one_sided_grad(np, prs, dyf, 1)
        dpdz = _one_sided_grad(np, prs, dzf, 0)
        inv = f(f(1.0) / f(rho * rho))
        terms["solenoid_x"] = f(inv * f(f(drdy * dpdz) - f(drdz * dpdy)))
        terms["solenoid_y"] = f(inv * f(f(drdz * dpdx) - f(drdx * dpdz)))
        terms["solenoid_z"] = f(inv * f(f(drdx * dpdy) - f(drdy * dpdx)))
        ring = nz * ny * nx - max(nz - 2, 0) * max(ny - 2, 0) * max(nx - 2, 0)
        one_sided = int(ring)

    if uten is None or vten is None or wten is None:
        for axis in ("x", "y", "z"):
            terms[f"friction_{axis}"] = zeros.copy()
    else:
        fu = _one_sided_grad
        tu = np.ascontiguousarray(uten, dtype=np.float32)
        tv = np.ascontiguousarray(vten, dtype=np.float32)
        tw = np.ascontiguousarray(wten, dtype=np.float32)
        for name, value in (("uten", tu), ("vten", tv), ("wten", tw)):
            if value.shape != (nz, ny, nx):
                raise ValueError(
                    f"{name} must be interpolated to the (nz, ny, nx) scalar rectangle"
                )
        terms["friction_x"] = f(fu(np, tw, dyf, 1) - fu(np, tv, dzf, 0))
        terms["friction_y"] = f(fu(np, tu, dzf, 0) - fu(np, tw, dxf, 2))
        terms["friction_z"] = f(fu(np, tv, dxf, 2) - fu(np, tu, dyf, 1))

    return VorticityBudget(
        EulerianVorticity(xvort, yvort, zvort),
        {name: terms[name] for name in VORTICITY_BUDGET_FIELD_NAMES},
        centred_words=nz * ny * nx,
        one_sided_words=one_sided,
    )


CUDA_VORTICITY_BUDGET_SOURCE = r'''// gpuwm-cm1.vorticity-budget-kernels/v1
// Components: CM1 r21.1 src/misclibs.F calcvort, flat Cartesian branch.
// Budget terms: GM1 extension; CM1 r21.1 has none.
static_assert(sizeof(float) == 4, "CM1 default real must be binary32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long ix3(int k,int j,int i,int ny,int nx){return ((unsigned long long)k*(unsigned long long)ny+(unsigned long long)j)*(unsigned long long)nx+(unsigned long long)i;}

// scalar-point averages of the padded staggered carriers, CM1 one-based (i,j,k)
__device__ __forceinline__ float sU(const float*u,int i,int j,int k,int nx,int ny,int ng){
  const int nxu=nx+1+2*ng,nyu=ny+2*ng;
  return m32(0.5f,a32(u[ix3(k,j-1+ng,i-1+ng,nyu,nxu)],u[ix3(k,j-1+ng,i+ng,nyu,nxu)]));}
__device__ __forceinline__ float sV(const float*v,int i,int j,int k,int nx,int ny,int ng){
  const int nxv=nx+2*ng,nyv=ny+1+2*ng;
  return m32(0.5f,a32(v[ix3(k,j-1+ng,i-1+ng,nyv,nxv)],v[ix3(k,j+ng,i-1+ng,nyv,nxv)]));}
__device__ __forceinline__ float sW(const float*w,int i,int j,int k,int nx,int ny,int ng){
  const int nxw=nx+2*ng,nyw=ny+2*ng;
  return m32(0.5f,a32(w[ix3(k,j-1+ng,i-1+ng,nyw,nxw)],w[ix3(k+1,j-1+ng,i-1+ng,nyw,nxw)]));}

extern "C" __global__ void cm1_vorticity_budget_v1(
    const float* __restrict__ u,const float* __restrict__ v,const float* __restrict__ w,
    float* __restrict__ xvort,float* __restrict__ yvort,float* __restrict__ zvort,
    float* __restrict__ terms,
    int nx,int ny,int nz,int ng,float dx,float dy,float dz){
  const unsigned long long words=(unsigned long long)nx*ny*nz;
  const float rdx2=m32(0.5f,d32(1.0f,dx)),rdy2=m32(0.5f,d32(1.0f,dy)),rdz2=m32(0.5f,d32(1.0f,dz));
  for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;q<words;q+=(unsigned long long)blockDim.x*gridDim.x){
    const int i0=(int)(q%(unsigned long long)nx),j0=(int)((q/(unsigned long long)nx)%(unsigned long long)ny),k0=(int)(q/((unsigned long long)nx*ny));
    const int i=i0+1,j=j0+1,k=k0+1;   // CM1 one-based
    const float dudx=m32(s32(sU(u,i+1,j,k,nx,ny,ng),sU(u,i-1,j,k,nx,ny,ng)),rdx2);
    const float dudy=m32(s32(sU(u,i,j+1,k,nx,ny,ng),sU(u,i,j-1,k,nx,ny,ng)),rdy2);
    const float dudz=m32(s32(sU(u,i,j,k+1,nx,ny,ng),sU(u,i,j,k-1,nx,ny,ng)),rdz2);
    const float dvdx=m32(s32(sV(v,i+1,j,k,nx,ny,ng),sV(v,i-1,j,k,nx,ny,ng)),rdx2);
    const float dvdy=m32(s32(sV(v,i,j+1,k,nx,ny,ng),sV(v,i,j-1,k,nx,ny,ng)),rdy2);
    const float dvdz=m32(s32(sV(v,i,j,k+1,nx,ny,ng),sV(v,i,j,k-1,nx,ny,ng)),rdz2);
    const float dwdx=m32(s32(sW(w,i+1,j,k,nx,ny,ng),sW(w,i-1,j,k,nx,ny,ng)),rdx2);
    const float dwdy=m32(s32(sW(w,i,j+1,k,nx,ny,ng),sW(w,i,j-1,k,nx,ny,ng)),rdy2);
    const float dwdz=m32(s32(sW(w,i,j,k+1,nx,ny,ng),sW(w,i,j,k-1,nx,ny,ng)),rdz2);
    const float xv=s32(dwdy,dvdz),yv=s32(dudz,dwdx),zv=s32(dvdx,dudy);
    const float dvg=a32(a32(dudx,dvdy),dwdz);
    xvort[q]=xv; yvort[q]=yv; zvort[q]=zv;
    // slot order matches VORTICITY_BUDGET_FIELD_NAMES, five terms per axis
    terms[0u*words+q]=m32(xv,dudx);
    terms[1u*words+q]=a32(m32(yv,dudy),m32(zv,dudz));
    terms[2u*words+q]=-m32(xv,dvg);
    terms[5u*words+q]=m32(yv,dvdy);
    terms[6u*words+q]=a32(m32(xv,dvdx),m32(zv,dvdz));
    terms[7u*words+q]=-m32(yv,dvg);
    terms[10u*words+q]=m32(zv,dwdz);
    terms[11u*words+q]=a32(m32(xv,dwdx),m32(yv,dwdy));
    terms[12u*words+q]=-m32(zv,dvg);
  }
}
'''
CUDA_VORTICITY_BUDGET_SOURCE_BYTES = CUDA_VORTICITY_BUDGET_SOURCE.encode("utf-8")
CUDA_VORTICITY_BUDGET_SOURCE_SHA256 = hashlib.sha256(
    CUDA_VORTICITY_BUDGET_SOURCE_BYTES
).hexdigest()

VORTICITY_BUDGET_KERNEL_NAMES: tuple[str, ...] = ("cm1_vorticity_budget_v1",)


@dataclass(frozen=True, slots=True)
class CompiledVorticityBudgetKernels:
    module: Any
    kernels: tuple[Any, ...]
    source_sha256: str
    abi: str = VORTICITY_BUDGET_ABI


def compile_cuda_vorticity_budget() -> CompiledVorticityBudgetKernels:
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    module = compile_nvrtc_module(cupy, code=CUDA_VORTICITY_BUDGET_SOURCE,
                                  options=("--std=c++17", "-lineinfo"),
                                  filename="gm1_vorticity_budget.cu")
    kernels = tuple(module.get_function(n) for n in VORTICITY_BUDGET_KERNEL_NAMES)
    return CompiledVorticityBudgetKernels(
        module, kernels, CUDA_VORTICITY_BUDGET_SOURCE_SHA256
    )


def launch_cuda_vorticity_budget(
    *, u: Any, v: Any, w: Any, xvort: Any, yvort: Any, zvort: Any, terms: Any,
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float,
    stream: Any, compiled: CompiledVorticityBudgetKernels, cupy: Any,
) -> Mapping[str, Any]:
    """Fill the three components and the nine velocity-gradient budget terms.

    ``terms`` is a ``(15, nz, ny, nx)`` float32 device array whose slot order
    is :data:`VORTICITY_BUDGET_FIELD_NAMES`.  The solenoid and friction slots
    are left untouched by this kernel; the runtime fills them from ``rho``,
    ``prs`` and the held momentum tendency, or leaves them zero and records
    that in the receipt.
    """

    words = nx * ny * nz
    blocks = ((words + VORTICITY_BUDGET_THREADS - 1) // VORTICITY_BUDGET_THREADS, 1, 1)
    compiled.kernels[0](
        blocks, (VORTICITY_BUDGET_THREADS, 1, 1),
        (u, v, w, xvort, yvort, zvort, terms,
         cupy.int32(nx), cupy.int32(ny), cupy.int32(nz), cupy.int32(CM1_OPEN_NGXY),
         cupy.float32(dx), cupy.float32(dy), cupy.float32(dz)),
        stream=stream,
    )
    return {
        "schema": VORTICITY_BUDGET_SCHEMA,
        "abi": compiled.abi,
        "source_sha256": compiled.source_sha256,
        "fields": list(VORTICITY_BUDGET_FIELD_NAMES),
        "filled_by_this_kernel": [
            n for n in VORTICITY_BUDGET_FIELD_NAMES
            if n.startswith(("stretching", "tilting", "divergence"))
        ],
        "cm1_authority": dict(SOURCE_ANCHORS),
    }


__all__ = [
    "CUDA_VORTICITY_BUDGET_SOURCE", "CUDA_VORTICITY_BUDGET_SOURCE_SHA256",
    "CompiledVorticityBudgetKernels", "EulerianVorticity", "SOURCE_ANCHORS",
    "VORTICITY_BUDGET_ABI", "VORTICITY_BUDGET_FIELD_NAMES",
    "VORTICITY_BUDGET_KERNEL_NAMES", "VORTICITY_BUDGET_SCHEMA",
    "VORTICITY_BUDGET_TERM_NAMES", "VORTICITY_COMPONENT_NAMES",
    "VorticityBudget", "compile_cuda_vorticity_budget",
    "corner_vertical_vorticity", "cpu_vorticity_budget",
    "launch_cuda_vorticity_budget",
]
