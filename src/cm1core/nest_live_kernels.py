"""Device kernels for the live nest, and the CPU references they are held to.

Four operators, each a declared GM1 extension (CM1 r21.1 has no nesting;
``records/CM1-NEST-SEAM.md`` section 1) and each written so its CPU reference
below reproduces its raw words:

``gm1_nest_capture``
    The parent -> child reconstruction on a staggered lattice, evaluated either
    over the whole child (the child's initial condition, WRF's
    ``med_nest_initial`` role) or over the four boundary strips of width
    ``spec_zone + relax_zone`` (the boundary tables, WRF's ``bdy_interp1``
    role, ``share/interp_fcn.F:2423-2626``).  With ``want_tend`` it also
    writes WRF's pair ``bdy_xs = nfld`` and ``bdy_txs = rdt*(psca - nfld)``
    (``share/interp_fcn.F:2583-2584``): the child's own value at the start of
    the parent interval, and the tendency that carries it to the interpolated
    parent at the end of the interval, with ``rdt = 1.D0/cdt`` formed in
    double precision (``share/interp_fcn.F:2480,2500``).

    The reconstruction is ``nest_downscale.downscale_level_staggered`` -- the
    same limited five-point slopes, evaluated at each child point's offset
    from its nearest parent point on that point's own lattice -- promoted to
    float64, computed, and narrowed to float32 once.  ``u`` is a face lattice
    in ``x`` and a centre lattice in ``y``; ``v`` the reverse; ``w``, the
    scalars and the pressure are centre lattices in both.  Why that operator is
    admissible for a re-imposed boundary tendency, which its docstring used to
    deny, is established by measurement in ``records/CM1-NEST-SEAM.md``
    section 5.1 and tested in ``tests/test_nest_live_operators.py``.

``gm1_nest_spec_gather`` / ``gm1_nest_spec_apply``
    WRF's ``spec_bdyupdate`` (``share/module_bc.F:1955-2064``): inside the
    specified zone the field advances by the boundary tendency alone,
    ``field = field + dt*field_tend``.  GM1's acoustic operator writes every
    face it owns, the specified rows included, so the pre-advance words of the
    specified rows are gathered first and the update is written from them.
    The loop ownership is ``spec_bdyupdate``'s: the ``y`` sides own the
    corners (``b_limit = b_dist``) and the ``x`` sides start one row in
    (``b_dist+jbs+1 .. jbe-b_dist-1``).  ``gm1_nest_spec_apply`` in mode 1 is
    the RK scalar form, ``scalar = scalar_old + dt_rk*tend``
    (``dyn_em/solve_em.F:2305`` ``rk_update_scalar`` with the specified-zone
    tendency set by ``spec_bdy_scalar``, ``:2277``).  Modes 2 and 3 are WRF's
    end-of-step ``spec_bdy_final`` (``share/module_bc.F:2066-2216``, called at
    the end of ``solve_em``, WRF v4.6.1 ``dyn_em/solve_em.F:4531-4716``):
    ``field = bdy + dtbc*bdy_tend`` on the specified rows, from the boundary
    value table itself; mode 3 holds a species at zero or above.

``gm1_nest_restrict``
    Two-way feedback, this port's own (``CM1-NEST-SEAM.md`` section 3.3): each
    covered parent point becomes the mean of its ``r*r`` child block, summed
    in float64 in raster order.  Blocks that reach into the child's
    relaxation zone are not fed back.

Every single-precision operation is an explicit ``_rn`` intrinsic and every
double-precision one is a plain IEEE operation, compiled with
``--fmad=false`` so nothing is contracted.
"""

from __future__ import annotations

import hashlib
from typing import Any

NEST_LIVE_KERNEL_SCHEMA = "gm1.cartesian-nest-live-kernels/v1"

#: The sides, in the order every table and receipt lists them, and the integer
#: each kernel is handed.  ``0`` is the whole child extent.
SIDE_CODES = {"full": 0, "west": 1, "east": 2, "south": 3, "north": 4}

NEST_LIVE_KERNEL_NAMES = (
    "gm1_nest_capture",
    "gm1_nest_spec_gather",
    "gm1_nest_spec_apply",
    "gm1_nest_restrict",
)

NEST_LIVE_KERNEL_SOURCE = r'''// gm1.cartesian-nest-live-kernels/v1
// CM1 r21.1 has no nesting; every operator here is a declared GM1 extension.
// Placement and time interpolation follow WRF: share/interp_fcn.F:2423-2626
// (bdy_interp1), share/module_bc.F:1955-2064 (spec_bdyupdate),
// dyn_em/solve_em.F:2305 (rk_update_scalar on the specified zone).
#define GM1_LN_IDX(k, j, i, nj, ni) \
    ((((unsigned long long)(k) * (unsigned long long)(nj)) + (unsigned long long)(j)) \
     * (unsigned long long)(ni) + (unsigned long long)(i))

// numpy.minimum / numpy.maximum on non-NaN operands: the second operand is
// returned unless the first is strictly smaller / larger.  That choice fixes
// the sign of a zero, which is what a raw-word comparison sees.
__device__ __forceinline__ double gm1_ln_min(double a, double b) { return (a < b) ? a : b; }
__device__ __forceinline__ double gm1_ln_max(double a, double b) { return (a > b) ? a : b; }
__device__ __forceinline__ int gm1_ln_clamp(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// nest_downscale._axis_donors, one axis.  A centre lattice sits at the
// symmetric sub-offset (2m+1-r)/(2r) of parent cell c//r; a face lattice at
// parent-face coordinate c/r, nearest face floor(c/r+0.5), and the offset of
// a coincident face is exactly zero.
__device__ __forceinline__ void gm1_ln_axis(int c, int ratio, int origin, int face,
                                           int* donor, double* off)
{
    if (face) {
        double exact = (double)c / (double)ratio;
        double nearest = floor(exact + 0.5);
        *donor = origin + (int)nearest;
        *off = (c % ratio == 0) ? 0.0 : (exact - nearest);
    } else {
        int m = c % ratio;
        *donor = origin + c / ratio;
        *off = (double)(2 * m + 1 - ratio) / (double)(2 * ratio);
    }
}

// Side layouts: full (nzl, cl_nj, cl_ni); west/east (nzl, cl_nj, width);
// south/north (nzl, width, cl_ni).  Distance index 0 is the outermost row.
extern "C" __global__ void gm1_nest_capture(
    const float* __restrict__ parent, const float* __restrict__ child,
    float* __restrict__ out_value, float* __restrict__ out_tend,
    int p_nj_owner, int p_ni_owner, int p_k0, int p_j0, int p_i0,
    int pl_nj, int pl_ni,
    int c_nj_owner, int c_ni_owner, int c_k0, int c_j0, int c_i0,
    int nzl, int cl_nj, int cl_ni, int side, int width,
    int ratio, int parent_i0, int parent_j0, int x_face, int y_face,
    double rdt, int want_tend)
{
    int na = cl_nj, nb = cl_ni;
    if (side == 1 || side == 2) nb = width;
    if (side == 3 || side == 4) na = width;
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)na * (unsigned long long)nb;
    if (tid >= (unsigned long long)nzl * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int a = (int)(rem / (unsigned long long)nb);
    int b = (int)(rem - (unsigned long long)a * nb);
    int j = a, i = b;
    if (side == 2) i = cl_ni - 1 - b;
    if (side == 4) j = cl_nj - 1 - a;

    int di, dj;
    double ax, ay;
    gm1_ln_axis(i, ratio, parent_i0, x_face, &di, &ax);
    gm1_ln_axis(j, ratio, parent_j0, y_face, &dj, &ay);
    int im = gm1_ln_clamp(di - 1, pl_ni - 1), ip = gm1_ln_clamp(di + 1, pl_ni - 1);
    int jm = gm1_ln_clamp(dj - 1, pl_nj - 1), jp = gm1_ln_clamp(dj + 1, pl_nj - 1);
    int ic = gm1_ln_clamp(di, pl_ni - 1), jc = gm1_ln_clamp(dj, pl_nj - 1);
    int kk = k + p_k0;
    double c = (double)parent[GM1_LN_IDX(kk, jc + p_j0, ic + p_i0, p_nj_owner, p_ni_owner)];
    double w = (double)parent[GM1_LN_IDX(kk, jc + p_j0, im + p_i0, p_nj_owner, p_ni_owner)];
    double e = (double)parent[GM1_LN_IDX(kk, jc + p_j0, ip + p_i0, p_nj_owner, p_ni_owner)];
    double s = (double)parent[GM1_LN_IDX(kk, jm + p_j0, ic + p_i0, p_nj_owner, p_ni_owner)];
    double n = (double)parent[GM1_LN_IDX(kk, jp + p_j0, ic + p_i0, p_nj_owner, p_ni_owner)];
    double sx = 0.5 * (e - w);
    double sy = 0.5 * (n - s);
    double dev = 0.5 * fabs(sx) + 0.5 * fabs(sy);
    double lo = gm1_ln_min(gm1_ln_min(gm1_ln_min(c, w), gm1_ln_min(e, s)), n);
    double hi = gm1_ln_max(gm1_ln_max(gm1_ln_max(c, w), gm1_ln_max(e, s)), n);
    int flat = (dev == 0.0);
    double safe = flat ? 1.0 : dev;
    double phi = gm1_ln_min(1.0, gm1_ln_min((hi - c) / safe, (c - lo) / safe));
    if (phi < 0.0) phi = 0.0;
    if (flat) phi = 0.0;
    double slx = phi * sx;
    double sly = phi * sy;
    float target = (float)(c + (slx * ax + sly * ay));
    if (!want_tend) {
        out_value[tid] = target;
        return;
    }
    float here = child[GM1_LN_IDX(k + c_k0, j + c_j0, i + c_i0, c_nj_owner, c_ni_owner)];
    out_value[tid] = here;
    float diff = __fsub_rn(target, here);
    out_tend[tid] = (float)(rdt * (double)diff);
}

// Launched over one side's specified band: west/east (nzl, nyl, sz), south/
// north (nzl, sz, nxl).  Points the side does not own are left untouched.
__device__ __forceinline__ int gm1_ln_owned(int side, int d, int along, int nyl, int nxl)
{
    if (side == 3 || side == 4) return along >= d && along < nxl - d;
    return along >= d + 1 && along < nyl - d - 1;
}

extern "C" __global__ void gm1_nest_spec_gather(
    const float* __restrict__ field, float* __restrict__ snap,
    int nj_owner, int ni_owner, int k0, int j0, int i0,
    int nzl, int nyl, int nxl, int side, int sz)
{
    int na = nyl, nb = sz;
    if (side == 3 || side == 4) { na = sz; nb = nxl; }
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)na * (unsigned long long)nb;
    if (tid >= (unsigned long long)nzl * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int a = (int)(rem / (unsigned long long)nb);
    int b = (int)(rem - (unsigned long long)a * nb);
    int j, i, d, along;
    if (side == 3) { d = a; j = a; i = b; along = i; }
    else if (side == 4) { d = a; j = nyl - 1 - a; i = b; along = i; }
    else if (side == 1) { d = b; j = a; i = b; along = j; }
    else { d = b; j = a; i = nxl - 1 - b; along = j; }
    if (!gm1_ln_owned(side, d, along, nyl, nxl)) return;
    snap[tid] = field[GM1_LN_IDX(k + k0, j + j0, i + i0, nj_owner, ni_owner)];
}

// mode 0: field = snap + dt*tend (spec_bdyupdate, from the gathered words).
// mode 1: field = base + dt*tend  (rk_update_scalar, from the stage's
//         time-level-n field, which has the same owner layout as field).
// mode 2: field = value + dt*tend (spec_bdy_final: snap is the boundary value
//         table, laid out as tend, and dt is dtbc).
// mode 3: mode 2 held at zero or above (a species after pdefq).
// tend is the boundary table: west/east (nzl, nyl, width), south/north
// (nzl, width, nxl).
extern "C" __global__ void gm1_nest_spec_apply(
    float* __restrict__ field, const float* __restrict__ snap,
    const float* __restrict__ base, const float* __restrict__ tend,
    int nj_owner, int ni_owner, int k0, int j0, int i0,
    int nzl, int nyl, int nxl, int side, int sz, int width,
    float dt, int mode)
{
    int na = nyl, nb = sz;
    if (side == 3 || side == 4) { na = sz; nb = nxl; }
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)na * (unsigned long long)nb;
    if (tid >= (unsigned long long)nzl * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int a = (int)(rem / (unsigned long long)nb);
    int b = (int)(rem - (unsigned long long)a * nb);
    int j, i, d, along;
    unsigned long long t;
    if (side == 3) { d = a; j = a; i = b; along = i;
        t = GM1_LN_IDX(k, d, i, width, nxl); }
    else if (side == 4) { d = a; j = nyl - 1 - a; i = b; along = i;
        t = GM1_LN_IDX(k, d, i, width, nxl); }
    else if (side == 1) { d = b; j = a; i = b; along = j;
        t = GM1_LN_IDX(k, j, d, nyl, width); }
    else { d = b; j = a; i = nxl - 1 - b; along = j;
        t = GM1_LN_IDX(k, j, d, nyl, width); }
    if (!gm1_ln_owned(side, d, along, nyl, nxl)) return;
    unsigned long long q = GM1_LN_IDX(k + k0, j + j0, i + i0, nj_owner, ni_owner);
    float start = (mode == 0) ? snap[tid] : (mode == 1) ? base[q] : snap[t];
    float out = __fadd_rn(start, __fmul_rn(dt, tend[t]));
    if (mode == 3 && out < 0.0f) out = 0.0f;
    field[q] = out;
}

// One covered parent lattice point per thread: (nzl, pnl_j, pnl_i), with
// pnl = span (+1 on a face axis).  Centre axis: child block c0=p*r .. c0+r-1,
// fed back iff c0 >= relax and c0+r <= n_child - relax.  Face axis: child
// faces p*r-h .. p*r+h with h=(r-1)/2, fed back iff p*r-h >= relax and
// p*r+h <= n_faces-1-relax.
extern "C" __global__ void gm1_nest_restrict(
    const float* __restrict__ child, float* __restrict__ parent,
    int c_nj_owner, int c_ni_owner, int c_k0, int c_j0, int c_i0,
    int p_nj_owner, int p_ni_owner, int p_k0, int p_j0, int p_i0,
    int nzl, int cl_nj, int cl_ni, int pnl_j, int pnl_i,
    int ratio, int relax, int parent_i0, int parent_j0, int x_face, int y_face)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)pnl_j * (unsigned long long)pnl_i;
    if (tid >= (unsigned long long)nzl * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int jp = (int)(rem / (unsigned long long)pnl_i);
    int ip = (int)(rem - (unsigned long long)jp * pnl_i);
    int h = (ratio - 1) / 2;
    int i0 = x_face ? ip * ratio - h : ip * ratio;
    int j0 = y_face ? jp * ratio - h : jp * ratio;
    if (i0 < relax || j0 < relax) return;
    if (i0 + ratio > cl_ni - relax) return;
    if (j0 + ratio > cl_nj - relax) return;
    double acc = 0.0;
    for (int aa = 0; aa < ratio; ++aa) {
        for (int bb = 0; bb < ratio; ++bb) {
            acc = acc + (double)child[GM1_LN_IDX(k + c_k0, j0 + aa + c_j0,
                                                 i0 + bb + c_i0,
                                                 c_nj_owner, c_ni_owner)];
        }
    }
    parent[GM1_LN_IDX(k + p_k0, jp + parent_j0 + p_j0, ip + parent_i0 + p_i0,
                      p_nj_owner, p_ni_owner)] =
        (float)(acc / (double)(ratio * ratio));
}
'''
NEST_LIVE_KERNEL_SHA256 = hashlib.sha256(
    NEST_LIVE_KERNEL_SOURCE.encode("utf-8")
).hexdigest()

#: The options every GM1 production image is compiled with
#: (``cuda_cartesian_startup._image_options``): no contraction, no
#: tiny-to-zero, correctly rounded division and square root.
NEST_LIVE_NVRTC_OPTIONS = (
    "--std=c++17",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
NEST_LIVE_THREADS = 256


def compile_nest_live_module(cupy: Any) -> Any:
    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy,
        code=NEST_LIVE_KERNEL_SOURCE,
        options=NEST_LIVE_NVRTC_OPTIONS,
        filename="gm1_nest_live.cu",
    )


# ---------------------------------------------------------------------------
# Lattices.
# ---------------------------------------------------------------------------
#: carrier stagger -> (x_face, y_face, extra vertical levels)
LATTICE_OF_STAGGER = {
    "u": (1, 0, 0),
    "v": (0, 1, 0),
    "w": (0, 0, 1),
    "mass": (0, 0, 0),
}


def lattice_shape(stagger: str, nz: int, ny: int, nx: int) -> tuple[int, int, int]:
    x_face, y_face, z_extra = LATTICE_OF_STAGGER[stagger]
    return nz + z_extra, ny + y_face, nx + x_face


def side_shape(side: str, lattice: tuple[int, int, int], width: int) -> tuple[int, int, int]:
    nzl, nyl, nxl = lattice
    if side == "full":
        return lattice
    if side in ("west", "east"):
        return nzl, nyl, width
    if side in ("south", "north"):
        return nzl, width, nxl
    raise ValueError(f"unknown side {side!r}")


def _launch_grid(count: int) -> tuple[tuple[int], tuple[int]]:
    return ((max(count, 1) + NEST_LIVE_THREADS - 1) // NEST_LIVE_THREADS,), (NEST_LIVE_THREADS,)


def _require_f32(cupy: Any, *arrays: Any) -> None:
    for array in arrays:
        if array.dtype != cupy.float32:
            raise TypeError(
                "every live-nest operand must be float32; a float64 array would "
                "be read as twice as many float32 words"
            )


def _owner_layout(owner: Any, origin: tuple[int, int, int],
                  lattice: tuple[int, int, int]) -> tuple[int, int, int, int, int]:
    """``(nj_owner, ni_owner, k0, j0, i0)`` after checking the rectangle fits."""

    if owner.ndim != 3 or not owner.flags.c_contiguous:
        raise ValueError("a live-nest owner must be a C-contiguous 3-D array")
    nk, nj, ni = (int(v) for v in owner.shape)
    k0, j0, i0 = (int(v) for v in origin)
    nzl, nyl, nxl = lattice
    if k0 < 0 or j0 < 0 or i0 < 0 or k0 + nzl > nk or j0 + nyl > nj or i0 + nxl > ni:
        raise ValueError(
            f"a {lattice} lattice at origin {origin} does not fit its "
            f"{tuple(owner.shape)} owner; the kernel would index past it"
        )
    return nj, ni, k0, j0, i0


def launch_capture(
    cupy: Any, module: Any, *, parent: Any, parent_origin: tuple[int, int, int],
    parent_lattice: tuple[int, int, int], child: Any | None,
    child_origin: tuple[int, int, int], child_lattice: tuple[int, int, int],
    out_value: Any, out_tend: Any | None, side: str, width: int, ratio: int,
    parent_i0: int, parent_j0: int, stagger: str, parent_dt_s: float,
    stream: Any = None,
) -> None:
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    want = out_tend is not None
    if parent_lattice[0] != child_lattice[0]:
        raise ValueError("the nest refines the horizontal only; level counts must match")
    shape = side_shape(side, child_lattice, width)
    if tuple(out_value.shape) != shape or (want and tuple(out_tend.shape) != shape):
        raise ValueError(f"{side} output must be {shape}")
    p = _owner_layout(parent, parent_origin, parent_lattice)
    if want:
        if child is None:
            raise ValueError("a boundary table needs the child's current state")
        c = _owner_layout(child, child_origin, child_lattice)
        _require_f32(cupy, parent, child, out_value, out_tend)
    else:
        c = (1, 1, 0, 0, 0)
        _require_f32(cupy, parent, out_value)
    if not out_value.flags.c_contiguous or (want and not out_tend.flags.c_contiguous):
        raise ValueError("live-nest outputs must be C-contiguous")
    count = int(out_value.size)
    blocks, threads = _launch_grid(count)
    placeholder = out_value if not want else out_tend
    kernel = module.get_function("gm1_nest_capture")
    i32 = cupy.int32
    kernel(
        blocks, threads,
        (
            parent, child if want else parent, out_value, placeholder,
            i32(p[0]), i32(p[1]), i32(p[2]), i32(p[3]), i32(p[4]),
            i32(parent_lattice[1]), i32(parent_lattice[2]),
            i32(c[0]), i32(c[1]), i32(c[2]), i32(c[3]), i32(c[4]),
            i32(child_lattice[0]), i32(child_lattice[1]), i32(child_lattice[2]),
            i32(SIDE_CODES[side]), i32(width), i32(ratio),
            i32(parent_i0), i32(parent_j0), i32(x_face), i32(y_face),
            cupy.float64(1.0 / float(parent_dt_s)), i32(1 if want else 0),
        ),
        stream=stream,
    )


def launch_spec_gather(
    cupy: Any, module: Any, *, field: Any, origin: tuple[int, int, int],
    lattice: tuple[int, int, int], snap: Any, side: str, spec_zone: int,
    stream: Any = None,
) -> None:
    lay = _owner_layout(field, origin, lattice)
    _require_f32(cupy, field, snap)
    shape = side_shape(side, lattice, spec_zone)
    if tuple(snap.shape) != shape:
        raise ValueError(f"{side} snapshot must be {shape}")
    blocks, threads = _launch_grid(int(snap.size))
    i32 = cupy.int32
    module.get_function("gm1_nest_spec_gather")(
        blocks, threads,
        (field, snap, i32(lay[0]), i32(lay[1]), i32(lay[2]), i32(lay[3]), i32(lay[4]),
         i32(lattice[0]), i32(lattice[1]), i32(lattice[2]),
         i32(SIDE_CODES[side]), i32(spec_zone)),
        stream=stream,
    )


def launch_spec_apply(
    cupy: Any, module: Any, *, field: Any, origin: tuple[int, int, int],
    lattice: tuple[int, int, int], tend: Any, side: str, spec_zone: int,
    width: int, dt: float, snap: Any | None = None, base: Any | None = None,
    value: Any | None = None, nonnegative: bool = False, stream: Any = None,
) -> None:
    """``gm1_nest_spec_apply``: ``snap`` (mode 0), ``base`` (mode 1) or ``value`` (mode 2, 3 with ``nonnegative``)."""

    lay = _owner_layout(field, origin, lattice)
    if sum(x is not None for x in (snap, base, value)) != 1:
        raise ValueError("exactly one of snap (mode 0), base (mode 1) and value (modes 2, 3) is required")
    if nonnegative and value is None:
        raise ValueError("only spec_bdy_final (value=) holds a species at zero or above")
    if base is not None and (tuple(base.shape) != tuple(field.shape) or not base.flags.c_contiguous):
        raise ValueError("the RK base must share the field's owner layout")
    band = side_shape(side, lattice, spec_zone)
    if snap is not None and tuple(snap.shape) != band:
        raise ValueError(f"{side} snapshot must be {band}")
    if tuple(tend.shape) != side_shape(side, lattice, width):
        raise ValueError(f"{side} tendency table must be {side_shape(side, lattice, width)}")
    if value is not None and (tuple(value.shape) != tuple(tend.shape) or not value.flags.c_contiguous):
        raise ValueError(f"{side} value table must be laid out as its tendency table {tuple(tend.shape)}")
    first = snap if snap is not None else (value if value is not None else base)
    _require_f32(cupy, field, tend, first)
    mode = 0 if snap is not None else 1 if base is not None else (3 if nonnegative else 2)
    count = band[0] * band[1] * band[2]
    blocks, threads = _launch_grid(count)
    i32 = cupy.int32
    module.get_function("gm1_nest_spec_apply")(
        blocks, threads,
        (field, snap if snap is not None else (value if value is not None else tend),
         base if base is not None else tend, tend,
         i32(lay[0]), i32(lay[1]), i32(lay[2]), i32(lay[3]), i32(lay[4]),
         i32(lattice[0]), i32(lattice[1]), i32(lattice[2]),
         i32(SIDE_CODES[side]), i32(spec_zone), i32(width),
         cupy.float32(dt), i32(mode)),
        stream=stream,
    )


def launch_restrict(
    cupy: Any, module: Any, *, child: Any, child_origin: tuple[int, int, int],
    child_lattice: tuple[int, int, int], parent: Any,
    parent_origin: tuple[int, int, int], parent_lattice: tuple[int, int, int],
    span_x: int, span_y: int, ratio: int, relax_zone: int, parent_i0: int,
    parent_j0: int, stagger: str, stream: Any = None,
) -> None:
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    c = _owner_layout(child, child_origin, child_lattice)
    p = _owner_layout(parent, parent_origin, parent_lattice)
    _require_f32(cupy, child, parent)
    pnl_j, pnl_i = span_y + y_face, span_x + x_face
    if parent_j0 + pnl_j > parent_lattice[1] or parent_i0 + pnl_i > parent_lattice[2]:
        raise ValueError("the covered parent rectangle does not fit the parent lattice")
    count = child_lattice[0] * pnl_j * pnl_i
    blocks, threads = _launch_grid(count)
    i32 = cupy.int32
    module.get_function("gm1_nest_restrict")(
        blocks, threads,
        (child, parent,
         i32(c[0]), i32(c[1]), i32(c[2]), i32(c[3]), i32(c[4]),
         i32(p[0]), i32(p[1]), i32(p[2]), i32(p[3]), i32(p[4]),
         i32(child_lattice[0]), i32(child_lattice[1]), i32(child_lattice[2]),
         i32(pnl_j), i32(pnl_i), i32(ratio), i32(relax_zone),
         i32(parent_i0), i32(parent_j0), i32(x_face), i32(y_face)),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# CPU references.  NumPy, float64 where the kernel is float64 and float32
# where it is float32, in the kernel's own order.
# ---------------------------------------------------------------------------
def _np() -> Any:
    import numpy

    return numpy


def reference_capture(
    parent_lattice_volume: Any, *, child_lattice: tuple[int, int, int],
    child_volume: Any | None, side: str, width: int, ratio: int,
    parent_i0: int, parent_j0: int, stagger: str, parent_dt_s: float,
) -> tuple[Any, Any | None]:
    """``gm1_nest_capture`` on the host.

    ``parent_lattice_volume`` is the parent's physical lattice rectangle and
    ``child_volume`` the child's; both float32.  The reconstruction is
    :func:`cm1core.nest_downscale.downscale_level_staggered` on each level --
    literally that function -- narrowed to float32.
    """

    numpy = _np()
    from dataclasses import dataclass

    from .nest_downscale import downscale_level_staggered

    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    nzl, nyl, nxl = child_lattice
    nx = nxl - x_face
    ny = nyl - y_face

    @dataclass(frozen=True)
    class _Window:
        refinement_ratio: int
        parent_i0: int
        parent_j0: int
        nx: int
        ny: int

        def sub_cell_offsets(self) -> tuple[float, ...]:
            r = self.refinement_ratio
            return tuple((2 * m + 1 - r) / (2 * r) for m in range(r))

    window = _Window(ratio, parent_i0, parent_j0, nx, ny)
    parent = numpy.asarray(parent_lattice_volume, dtype=numpy.float32)
    full = numpy.empty((nzl, ny + y_face, nx + x_face), dtype=numpy.float32)
    for k in range(nzl):
        level = downscale_level_staggered(
            parent[k], window,  # type: ignore[arg-type]
            x_lattice="face" if x_face else "mass",
            y_lattice="face" if y_face else "mass",
        )
        full[k] = level.astype(numpy.float32)
    selector = _side_selector(side, child_lattice, width)
    target = numpy.ascontiguousarray(full[selector])
    if child_volume is None:
        return target, None
    here = numpy.ascontiguousarray(numpy.asarray(child_volume, dtype=numpy.float32)[selector])
    diff = (target - here).astype(numpy.float32)
    rdt = 1.0 / float(parent_dt_s)
    tend = (rdt * diff.astype(numpy.float64)).astype(numpy.float32)
    return here, tend


def _side_selector(side: str, lattice: tuple[int, int, int], width: int) -> Any:
    numpy = _np()
    nzl, nyl, nxl = lattice
    if side == "full":
        return (slice(None), slice(None), slice(None))
    if side == "west":
        return (slice(None), slice(None), slice(0, width))
    if side == "east":
        return (slice(None), slice(None), numpy.arange(nxl - 1, nxl - 1 - width, -1))
    if side == "south":
        return (slice(None), slice(0, width), slice(None))
    if side == "north":
        return (slice(None), numpy.arange(nyl - 1, nyl - 1 - width, -1), slice(None))
    raise ValueError(side)


def owned_band_mask(side: str, lattice: tuple[int, int, int], spec_zone: int) -> Any:
    """Boolean mask over one side's band: which entries that side writes."""

    numpy = _np()
    nzl, nyl, nxl = lattice
    shape = side_shape(side, lattice, spec_zone)
    mask = numpy.zeros(shape[1:], dtype=bool)
    for d in range(spec_zone):
        if side in ("south", "north"):
            mask[d, d:nxl - d] = True
        else:
            mask[d + 1:nyl - d - 1, d] = True
    return numpy.broadcast_to(mask, shape)


def reference_spec_apply(
    field: Any, *, lattice: tuple[int, int, int], tend: Any, side: str,
    spec_zone: int, width: int, dt: float, snap: Any | None = None,
    base: Any | None = None, value: Any | None = None, nonnegative: bool = False,
) -> Any:
    """``gm1_nest_spec_apply`` on a physical lattice volume (origin 0); ``value`` is modes 2 and 3."""

    numpy = _np()
    out = numpy.array(field, dtype=numpy.float32, copy=True)
    nzl, nyl, nxl = lattice
    dt32 = numpy.float32(dt)
    mask = owned_band_mask(side, lattice, spec_zone)
    band = side_shape(side, lattice, spec_zone)
    for k in range(band[0]):
        for a in range(band[1]):
            for b in range(band[2]):
                if not mask[k, a, b]:
                    continue
                if side in ("south", "north"):
                    d = a
                    j, i = (a, b) if side == "south" else (nyl - 1 - a, b)
                    at = (k, d, i)
                else:
                    d = b
                    j, i = (a, b) if side == "west" else (a, nxl - 1 - b)
                    at = (k, j, d)
                t = tend[at]
                if snap is not None:
                    start = snap[k, a, b]
                elif value is not None:
                    start = value[at]
                else:
                    start = base[k, j, i]
                result = numpy.float32(
                    numpy.float32(start) + numpy.float32(dt32 * numpy.float32(t))
                )
                if nonnegative and result < numpy.float32(0.0):
                    result = numpy.float32(0.0)
                out[k, j, i] = result
    return out


def reference_restrict(
    child: Any, parent: Any, *, child_lattice: tuple[int, int, int], span_x: int,
    span_y: int, ratio: int, relax_zone: int, parent_i0: int, parent_j0: int,
    stagger: str,
) -> Any:
    """``gm1_nest_restrict`` on physical lattice volumes (origin 0)."""

    numpy = _np()
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    out = numpy.array(parent, dtype=numpy.float32, copy=True)
    nzl, cl_nj, cl_ni = child_lattice
    h = (ratio - 1) // 2
    child = numpy.asarray(child, dtype=numpy.float32)
    for k in range(nzl):
        for jp in range(span_y + y_face):
            for ip in range(span_x + x_face):
                i0 = ip * ratio - h if x_face else ip * ratio
                j0 = jp * ratio - h if y_face else jp * ratio
                if i0 < relax_zone or j0 < relax_zone:
                    continue
                if i0 + ratio > cl_ni - relax_zone or j0 + ratio > cl_nj - relax_zone:
                    continue
                acc = 0.0
                for aa in range(ratio):
                    for bb in range(ratio):
                        acc = acc + float(child[k, j0 + aa, i0 + bb])
                out[k, jp + parent_j0, ip + parent_i0] = numpy.float32(acc / float(ratio * ratio))
    return out


__all__ = [
    "LATTICE_OF_STAGGER",
    "NEST_LIVE_KERNEL_NAMES",
    "NEST_LIVE_KERNEL_SCHEMA",
    "NEST_LIVE_KERNEL_SHA256",
    "NEST_LIVE_KERNEL_SOURCE",
    "NEST_LIVE_NVRTC_OPTIONS",
    "SIDE_CODES",
    "compile_nest_live_module",
    "lattice_shape",
    "launch_capture",
    "launch_restrict",
    "launch_spec_apply",
    "launch_spec_gather",
    "owned_band_mask",
    "reference_capture",
    "reference_restrict",
    "reference_spec_apply",
    "side_shape",
]
