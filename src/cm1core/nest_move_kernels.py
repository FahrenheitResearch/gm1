"""Device kernels for the moving live nest, and the CPU references they are held to.

CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md`` section 1).  A moving
nest is a declared GM1 extension whose authority is WRF, and each operator
below names the WRF routine it stands in for.

``gm1_nest_shift``
    WRF's whole-domain shift of a moving nest's state
    (``dyn_em/shift_domain_em.F:52-80`` looping
    ``frame/loop_based_x_shift_code.h`` /
    ``frame/loop_based_y_shift_code.h``: ``field(ips:ipf) =
    field(ips+px:ipf+px)`` with ``px = isign(parent_grid_ratio, disp_x)``, one
    parent cell at a time).  The child's lattice point ``(k, j, i)`` takes the
    word the unmoved child held at ``(k, j + r*sj, i + r*si)``; a point whose
    source lies outside the unmoved lattice is left untouched, because it is
    the newly exposed strip and has already been written from the parent by
    ``gm1_nest_capture`` (WRF's ``med_interp_domain`` restricted to
    ``imask == 1``, ``share/mediation_nest_move.F:188`` with the mask set by
    ``shift_domain_em.F:64-72``).  Words are copied, not recomputed, so the
    shifted interior is the child's own state to the last bit.

``gm1_nest_zeta_box``
    The surface-vortex tracker's field: the mean of the ``(2h+1)^2`` corner
    vertical vorticities around one corner, summed in float64 in raster order.
    ``dx*dy`` times that sum is the circulation around the box's outline
    (Stokes on the C grid: interior corner differences telescope), so the
    tracked quantity is an area-averaged vorticity -- a circulation per unit
    area -- not a point value.  The corner vorticity itself is CM1's
    ``src/parcel.F:1064-1066`` formula, evaluated by
    :func:`cm1core.vorticity_budget.corner_vertical_vorticity` before this
    kernel runs.

``gm1_nest_vortex_centroid``
    WRF's vortex centre (``share/mediation_nest_move.F:471-606``,
    ``time_for_move2`` under ``VORTEX_CENTER``), with the tracked field
    mirrored: WRF looks for the minimum of geopotential height and weights the
    centroid by ``vmax - height``; the vortex tracker looks for the maximum of
    area-averaged vorticity and weights by ``zeta - vmin``.  Serial (one
    thread) so the float64 accumulation order is WRF's ``DO j / DO i`` order
    and the CPU reference reproduces it word for word.

Every double-precision operation is a plain IEEE operation compiled with
``--fmad=false`` so nothing is contracted.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .nest_live_kernels import NEST_LIVE_NVRTC_OPTIONS, NEST_LIVE_THREADS

NEST_MOVE_KERNEL_SCHEMA = "gm1.cartesian-nest-move-kernels/v1"

NEST_MOVE_KERNEL_NAMES = (
    "gm1_nest_shift",
    "gm1_nest_zeta_box",
    "gm1_nest_vortex_centroid",
)

#: Number of doubles ``gm1_nest_vortex_centroid`` writes.
CENTROID_WORDS = 12

NEST_MOVE_KERNEL_SOURCE = r'''// gm1.cartesian-nest-move-kernels/v1
// CM1 r21.1 has no nesting; every operator here is a declared GM1 extension.
// Authority: WRF dyn_em/shift_domain_em.F:52-80 with
// frame/loop_based_{x,y}_shift_code.h (the shift), and
// share/mediation_nest_move.F:471-606 time_for_move2 (the vortex centre).
#define GM1_NM_IDX(k, j, i, nj, ni) \
    ((((unsigned long long)(k) * (unsigned long long)(nj)) + (unsigned long long)(j)) \
     * (unsigned long long)(ni) + (unsigned long long)(i))

// One thread per lattice point of the moved child (nzl, nyl, nxl).  snap and
// out share the owner layout (nj_owner, ni_owner) and lattice origin.
extern "C" __global__ void gm1_nest_shift(
    const float* __restrict__ snap, float* __restrict__ out,
    int nj_owner, int ni_owner, int k0, int j0, int i0,
    int nzl, int nyl, int nxl, int shift_j, int shift_i)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long plane = (unsigned long long)nyl * (unsigned long long)nxl;
    if (tid >= (unsigned long long)nzl * plane) return;
    int k = (int)(tid / plane);
    unsigned long long rem = tid - (unsigned long long)k * plane;
    int j = (int)(rem / (unsigned long long)nxl);
    int i = (int)(rem - (unsigned long long)j * nxl);
    int sj = j + shift_j;
    int si = i + shift_i;
    if (sj < 0 || sj >= nyl || si < 0 || si >= nxl) return;
    out[GM1_NM_IDX(k + k0, j + j0, i + i0, nj_owner, ni_owner)] =
        snap[GM1_NM_IDX(k + k0, sj + j0, si + i0, nj_owner, ni_owner)];
}

// One thread per corner of the (ny1, nx1) corner plane.  Only centres whose
// whole box lies in [lo, hi] on both axes are written.
extern "C" __global__ void gm1_nest_zeta_box(
    const float* __restrict__ zeta, double* __restrict__ out,
    int ny1, int nx1, int h, int lo_j, int hi_j, int lo_i, int hi_i)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (unsigned long long)ny1 * (unsigned long long)nx1) return;
    int j = (int)(tid / (unsigned long long)nx1);
    int i = (int)(tid - (unsigned long long)j * nx1);
    if (j - h < lo_j || j + h > hi_j || i - h < lo_i || i + h > hi_i) return;
    double acc = 0.0;
    for (int a = -h; a <= h; ++a) {
        for (int b = -h; b <= h; ++b) {
            acc = acc + (double)zeta[GM1_NM_IDX(0, j + a, i + b, ny1, nx1)];
        }
    }
    int side = 2 * h + 1;
    out[tid] = acc / (double)(side * side);
}

// Serial: launch with exactly one thread.  zbar is the (ny1, nx1) plane from
// gm1_nest_zeta_box.  [is0, ie0] x [js0, je0] is the search window
// (mediation_nest_move.F:471-487); [clo, chi] the admissible centres; d the
// centroid half-width in cells (rsmooth/dx, :556).
//
// out: 0 found, 1 imax, 2 jmax, 3 zmax, 4 is, 5 ie, 6 js, 7 je, 8 pbar,
//      9 xi, 10 xj, 11 vmin.  xi/xj are written only when pbar != 0 (:590).
extern "C" __global__ void gm1_nest_vortex_centroid(
    const double* __restrict__ zbar, double* __restrict__ out,
    int nx1, int is0, int ie0, int js0, int je0,
    int clo_i, int chi_i, int clo_j, int chi_j, double d)
{
    unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0ULL) return;
    // :527-542, mirrored: the first strict maximum in DO j / DO i order.
    int found = 0, im = -1, jm = -1;
    double best = 0.0;
    for (int j = js0; j <= je0; ++j) {
        for (int i = is0; i <= ie0; ++i) {
            double v = zbar[GM1_NM_IDX(0, j, i, 1, nx1)];
            if (!found || v > best) {
                if (v == v) { best = v; im = i; jm = j; found = 1; }
            }
        }
    }
    out[0] = (double)found;
    out[1] = (double)im;
    out[2] = (double)jm;
    out[3] = best;
    if (!found) return;
    // :558-565: REAL bounds assigned to INTEGER, i.e. truncated.
    double fis = (double)im - d, fie = (double)im + d;
    double fjs = (double)jm - d, fje = (double)jm + d;
    if (fis < (double)clo_i) fis = (double)clo_i;
    if (fie > (double)chi_i) fie = (double)chi_i;
    if (fjs < (double)clo_j) fjs = (double)clo_j;
    if (fje > (double)chi_j) fje = (double)chi_j;
    int is = (int)fis, ie = (int)fie, js = (int)fjs, je = (int)fje;
    out[4] = (double)is; out[5] = (double)ie;
    out[6] = (double)js; out[7] = (double)je;
    // :567-575, mirrored: the box minimum, first strict in DO j / DO i order.
    int seen = 0;
    double vmin = 0.0;
    for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
            double v = zbar[GM1_NM_IDX(0, j, i, 1, nx1)];
            if (!seen || v < vmin) { vmin = v; seen = 1; }
        }
    }
    // :577-588: fact = vmax - height, mirrored to zbar - vmin.
    double pbar = 0.0, ipbar = 0.0, jpbar = 0.0;
    for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
            double fact = zbar[GM1_NM_IDX(0, j, i, 1, nx1)] - vmin;
            pbar = pbar + fact;
            ipbar = ipbar + fact * (double)(i - is);
            jpbar = jpbar + fact * (double)(j - js);
        }
    }
    out[8] = pbar;
    out[11] = vmin;
    if (pbar != 0.0) {
        // :597-600 with a = 0, b = 1: xi = is + ipbar/pbar.
        out[9] = (double)is + ipbar / pbar;
        out[10] = (double)js + jpbar / pbar;
    }
}
'''
NEST_MOVE_KERNEL_SHA256 = hashlib.sha256(
    NEST_MOVE_KERNEL_SOURCE.encode("utf-8")
).hexdigest()


def compile_nest_move_module(cupy: Any) -> Any:
    from .nvrtc_module import compile_nvrtc_module

    return compile_nvrtc_module(
        cupy,
        code=NEST_MOVE_KERNEL_SOURCE,
        options=NEST_LIVE_NVRTC_OPTIONS,
        filename="gm1_nest_move.cu",
    )


def _launch_grid(count: int) -> tuple[tuple[int], tuple[int]]:
    return ((max(count, 1) + NEST_LIVE_THREADS - 1) // NEST_LIVE_THREADS,), (NEST_LIVE_THREADS,)


# ---------------------------------------------------------------------------
# Launchers.
# ---------------------------------------------------------------------------
def launch_shift(
    cupy: Any, module: Any, *, snap: Any, out: Any, origin: tuple[int, int, int],
    lattice: tuple[int, int, int], shift_j: int, shift_i: int, stream: Any = None,
) -> None:
    """``out[lattice] <- snap[lattice + (shift_j, shift_i)]`` where the source exists."""

    from .nest_live_kernels import _owner_layout, _require_f32

    if tuple(snap.shape) != tuple(out.shape):
        raise ValueError("the snapshot must share the moved carrier's owner layout")
    if snap is out:
        raise ValueError(
            "the shift reads the unmoved words while writing the moved ones; "
            "an in-place launch would read words other threads already moved"
        )
    lay = _owner_layout(out, origin, lattice)
    if not snap.flags.c_contiguous:
        raise ValueError("the snapshot must be C-contiguous")
    _require_f32(cupy, snap, out)
    blocks, threads = _launch_grid(lattice[0] * lattice[1] * lattice[2])
    i32 = cupy.int32
    module.get_function("gm1_nest_shift")(
        blocks, threads,
        (snap, out, i32(lay[0]), i32(lay[1]), i32(lay[2]), i32(lay[3]), i32(lay[4]),
         i32(lattice[0]), i32(lattice[1]), i32(lattice[2]), i32(shift_j), i32(shift_i)),
        stream=stream,
    )


def launch_zeta_box(
    cupy: Any, module: Any, *, zeta: Any, out: Any, h: int,
    lo_j: int, hi_j: int, lo_i: int, hi_i: int, stream: Any = None,
) -> None:
    if zeta.ndim != 2 or not zeta.flags.c_contiguous or zeta.dtype != cupy.float32:
        raise ValueError("the corner vorticity plane must be one C-contiguous float32 level")
    if tuple(out.shape) != tuple(zeta.shape) or out.dtype != cupy.float64:
        raise ValueError("the box-mean plane must be float64 on the corner lattice")
    ny1, nx1 = (int(v) for v in zeta.shape)
    blocks, threads = _launch_grid(ny1 * nx1)
    i32 = cupy.int32
    module.get_function("gm1_nest_zeta_box")(
        blocks, threads,
        (zeta, out, i32(ny1), i32(nx1), i32(h), i32(lo_j), i32(hi_j), i32(lo_i), i32(hi_i)),
        stream=stream,
    )


def launch_vortex_centroid(
    cupy: Any, module: Any, *, zbar: Any, out: Any, window: tuple[int, int, int, int],
    centres: tuple[int, int, int, int], d: float, stream: Any = None,
) -> None:
    if zbar.ndim != 2 or not zbar.flags.c_contiguous or zbar.dtype != cupy.float64:
        raise ValueError("the tracked plane must be one C-contiguous float64 level")
    if tuple(out.shape) != (CENTROID_WORDS,) or out.dtype != cupy.float64:
        raise ValueError(f"the centroid record is {CENTROID_WORDS} float64 words")
    is0, ie0, js0, je0 = window
    clo_i, chi_i, clo_j, chi_j = centres
    i32 = cupy.int32
    module.get_function("gm1_nest_vortex_centroid")(
        (1,), (1,),
        (zbar, out, i32(zbar.shape[1]), i32(is0), i32(ie0), i32(js0), i32(je0),
         i32(clo_i), i32(chi_i), i32(clo_j), i32(chi_j), cupy.float64(d)),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# CPU references, in the kernels' own order.
# ---------------------------------------------------------------------------
def reference_shift(snap: Any, out: Any, *, origin: tuple[int, int, int],
                    lattice: tuple[int, int, int], shift_j: int, shift_i: int) -> Any:
    import numpy

    result = numpy.array(out, dtype=numpy.float32, copy=True)
    snap = numpy.asarray(snap, dtype=numpy.float32)
    k0, j0, i0 = origin
    nzl, nyl, nxl = lattice
    # Destination rows/columns whose source lies inside the unmoved lattice.
    dj0, dj1 = max(0, -shift_j), min(nyl, nyl - shift_j)
    di0, di1 = max(0, -shift_i), min(nxl, nxl - shift_i)
    if dj0 < dj1 and di0 < di1:
        result[k0:k0 + nzl, j0 + dj0:j0 + dj1, i0 + di0:i0 + di1] = snap[
            k0:k0 + nzl, j0 + dj0 + shift_j:j0 + dj1 + shift_j,
            i0 + di0 + shift_i:i0 + di1 + shift_i,
        ]
    return result


def reference_zeta_box(zeta: Any, *, h: int, lo_j: int, hi_j: int, lo_i: int,
                       hi_i: int, out: Any | None = None) -> Any:
    import numpy

    zeta = numpy.asarray(zeta, dtype=numpy.float32)
    ny1, nx1 = zeta.shape
    result = (numpy.zeros((ny1, nx1), dtype=numpy.float64) if out is None
              else numpy.array(out, dtype=numpy.float64, copy=True))
    side = 2 * h + 1
    divisor = float(side * side)
    for j in range(max(lo_j + h, 0), min(hi_j - h, ny1 - 1) + 1):
        for i in range(max(lo_i + h, 0), min(hi_i - h, nx1 - 1) + 1):
            acc = 0.0
            for a in range(-h, h + 1):
                row = zeta[j + a]
                for b in range(-h, h + 1):
                    acc = acc + float(row[i + b])
            result[j, i] = acc / divisor
    return result


def reference_vortex_centroid(zbar: Any, *, window: tuple[int, int, int, int],
                              centres: tuple[int, int, int, int], d: float) -> list[float]:
    import numpy

    zbar = numpy.asarray(zbar, dtype=numpy.float64)
    is0, ie0, js0, je0 = window
    clo_i, chi_i, clo_j, chi_j = centres
    out = [0.0] * CENTROID_WORDS
    found, im, jm, best = False, -1, -1, 0.0
    for j in range(js0, je0 + 1):
        for i in range(is0, ie0 + 1):
            v = float(zbar[j, i])
            if (not found or v > best) and v == v:
                best, im, jm, found = v, i, j, True
    out[0], out[1], out[2], out[3] = float(found), float(im), float(jm), best
    if not found:
        return out
    fis, fie = float(im) - d, float(im) + d
    fjs, fje = float(jm) - d, float(jm) + d
    fis = float(clo_i) if fis < float(clo_i) else fis
    fie = float(chi_i) if fie > float(chi_i) else fie
    fjs = float(clo_j) if fjs < float(clo_j) else fjs
    fje = float(chi_j) if fje > float(chi_j) else fje
    is_, ie, js, je = int(fis), int(fie), int(fjs), int(fje)
    out[4], out[5], out[6], out[7] = float(is_), float(ie), float(js), float(je)
    seen, vmin = False, 0.0
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            v = float(zbar[j, i])
            if not seen or v < vmin:
                vmin, seen = v, True
    pbar = ipbar = jpbar = 0.0
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            fact = float(zbar[j, i]) - vmin
            pbar = pbar + fact
            ipbar = ipbar + fact * float(i - is_)
            jpbar = jpbar + fact * float(j - js)
    out[8], out[11] = pbar, vmin
    if pbar != 0.0:
        out[9] = float(is_) + ipbar / pbar
        out[10] = float(js) + jpbar / pbar
    return out


def fortran_int(value: float) -> int:
    """Fortran ``INT``/REAL-to-INTEGER assignment: truncation toward zero."""

    if not math.isfinite(value):
        raise ValueError("a non-finite tracker coordinate has no integer cell")
    return int(value)


__all__ = [
    "CENTROID_WORDS",
    "NEST_MOVE_KERNEL_NAMES",
    "NEST_MOVE_KERNEL_SCHEMA",
    "NEST_MOVE_KERNEL_SHA256",
    "NEST_MOVE_KERNEL_SOURCE",
    "compile_nest_move_module",
    "fortran_int",
    "launch_shift",
    "launch_vortex_centroid",
    "launch_zeta_box",
    "reference_shift",
    "reference_vortex_centroid",
    "reference_zeta_box",
]
