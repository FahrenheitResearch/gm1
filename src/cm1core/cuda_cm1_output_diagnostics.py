"""CM1 r21.1 ``&param9`` diagnostic operators on one CUDA device.

Every kernel here is a transcription of a named CM1 r21.1 routine, for the
Cartesian, no-terrain, uniform-horizontal-grid branch this lane integrates:

================  ==================================================
``calcuh``        ``src/misclibs.F:1417-1532`` -- 2-5 km AGL updraft
                  helicity, CM1's own integrand, layer and units
``calcvort``      ``src/misclibs.F:1540-2174`` -- xvort/yvort/zvort at
                  scalar points and the three-term potential vorticity
``calccref``      ``src/misclibs.F`` calccref -- column-maximum dBZ
surface swaths    ``src/solve3.F:857-1130`` -- the run-long extrema
                  ``sws``/``svs``/``sps``/``srs``/``sgs``/``sus``/``shs``
================  ==================================================

**Grid assumptions, and why they are not hidden.**  ``uf``/``vf`` are CM1's
stretched-horizontal-grid factors; ``param.F``'s uniform branch sets them to
one and the Cartesian door refuses ``stretch_x``/``stretch_y``.  ``mh``/``mf``
are the vertical metric; ``param.F:8170`` makes both one when ``stretch_z=0``,
which ``cartesian_door.py`` requires for this lane.  Both are therefore bound
to one here, and :func:`require_uniform_diagnostic_grid` refuses a geometry
where that is false rather than returning a silently wrong vorticity.  Only
``xvort``/``yvort`` (and the potential vorticity built from them) carry
``mh``/``mf``; ``calcuh``'s no-terrain branch and the surface swaths use the
horizontal factors and the column's own ``zh``/``zf`` heights, so they are
exact on a stretched column and are not refused there.

**Halo treatment.**  CM1 reads one halo cell beyond the physical rectangle in
every horizontal derivative.  Winds are read from the *padded* owners, whose
halos this lane's own boundary operators filled.  Scalars have no padded
owner, so their halo is reproduced exactly as ``src/bc.F:202-207,245-252``
fills it for open laterals (``wbc=ebc=sbc=nbc=2``): ``s(0,j,k)=s(1,j,k)`` and
``s(ni+1,j,k)=s(ni,j,k)``, i.e. an index clamp.  That is CM1's value, not an
approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import re
from typing import Any

CUDA_CM1_OUTPUT_DIAGNOSTICS_SCHEMA = "gpuwm-cm1.cuda-cm1-output-diagnostics/v1"
CUDA_CM1_OUTPUT_DIAGNOSTICS_ABI = "gpuwm-cm1.cuda-cm1-output-diagnostics-kernels/v1"
CUDA_CM1_OUTPUT_DIAGNOSTICS_THREADS = 256

SOURCE_ANCHORS = {
    "updraft_helicity": "src/misclibs.F:1417-1532",
    "vorticity_and_pv": "src/misclibs.F:1540-2174",
    "composite_reflectivity": "src/misclibs.F calccref",
    "surface_swaths": "src/solve3.F:857-1130",
    "swath_initial_values": "src/init3d.F:181-201",
    "open_lateral_scalar_halo": "src/bc.F:202-207,245-252",
    "uniform_horizontal_factors": "src/param.F:7789-7811",
    "uniform_vertical_metric": "src/param.F:8170",
    "fornberg_boundary_derivative": "src/param.F:9042-9052",
}

CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES = (
    "cm1_out_zeta_corner_v1",
    "cm1_out_calcuh_v1",
    "cm1_out_xvort_v1",
    "cm1_out_yvort_v1",
    "cm1_out_zvort_v1",
    "cm1_out_pv_v1",
    "cm1_out_calccref_v1",
    "cm1_out_swath_update_v1",
    "cm1_out_swath_init_v1",
)

CLAIM_LIMIT = (
    "CM1 r21.1 Cartesian no-terrain diagnostic operators on the completed "
    "post-solve3 state; uniform horizontal grid; xvort/yvort/pv on uniform dz "
    "only (calcuh and the swaths also on a stretched column); no "
    "terrain metric branch, no axisymmetric branch, no MPI reduction, and no "
    "forecast-skill claim"
)

#: ``src/init3d.F:184-198``.  Positive-definite swaths start at zero, the
#: minimum-seeking pressure swath at an absurdly large number, and the two
#: sign-free maxima at an absurdly low one.
CM1_SWATH_INITIAL: dict[str, float] = {
    "sws": 0.0,
    "svs": -1000.0,
    "sps": 200000.0,
    "srs": 0.0,
    "sgs": 0.0,
    "sus": -1000.0,
    "shs": 0.0,
}

#: ``src/misclibs.F`` calccref seeds the column maximum here.
CM1_CREF_INITIAL = -1000.0

#: ``src/misclibs.F:1443-1444``.  The integration layer is fixed in CM1.
CM1_UH_LAYER_BOTTOM_M = 2000.0
CM1_UH_LAYER_TOP_M = 5000.0


CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE = r'''
// CM1 r21.1 &param9 diagnostic operators.  Cartesian, no terrain, uniform
// horizontal grid (uf=vf=1, param.F:7789-7811) and uniform dz (mh=mf=1,
// param.F:8170).  Index conventions:
//
//   padded u  (nz+2, ny+6, nx+7)  physical [1:nz+1, 3:ny+3, 3:nx+4]
//   padded v  (nz+2, ny+7, nx+6)  physical [1:nz+1, 3:ny+4, 3:nx+3]
//   padded w  (nz+3, ny+6, nx+6)  physical [1:nz+2, 3:ny+3, 3:nx+3]
//
// so a CM1 one-based (i,j,k) maps to the zero-based triple (k-1, j-1, i-1)
// plus the halo offsets below.  Scalars carry no halo: SCLAMP reproduces
// bc.F's open-lateral copy.

#define UPAD(kz,jj,ii) u[ (size_t)((kz)+1)*(size_t)((ny+6)*(nx+7)) \
                        + (size_t)((jj)+3)*(size_t)(nx+7) + (size_t)((ii)+3) ]
#define VPAD(kz,jj,ii) v[ (size_t)((kz)+1)*(size_t)((ny+7)*(nx+6)) \
                        + (size_t)((jj)+3)*(size_t)(nx+6) + (size_t)((ii)+3) ]
#define WPAD(kf,jj,ii) w[ (size_t)((kf)+1)*(size_t)((ny+6)*(nx+6)) \
                        + (size_t)((jj)+3)*(size_t)(nx+6) + (size_t)((ii)+3) ]
#define M3(a,kz,jj,ii) a[ (size_t)(kz)*(size_t)(ny*nx) + (size_t)(jj)*(size_t)(nx) + (size_t)(ii) ]
#define CLAMPI(x,lo,hi) ((x)<(lo)?(lo):((x)>(hi)?(hi):(x)))

// ---------------------------------------------------------------------------
// misclibs.F:1451-1459 and :2033-2047.  Vertical vorticity on the cell corner
// (CM1 i=1..ni+1, j=1..nj+1), zero-based corner (ci,cj) = (i-1,j-1).
__device__ __forceinline__ float zeta_corner(
    const float* __restrict__ u, const float* __restrict__ v,
    int kz, int cj, int ci, int nx, int ny, float rdx, float rdy) {
  const float dvdx = (VPAD(kz, cj, ci) - VPAD(kz, cj, ci - 1)) * rdx;
  const float dudy = (UPAD(kz, cj, ci) - UPAD(kz, cj - 1, ci)) * rdy;
  return dvdx - dudy;
}

extern "C" __global__ void cm1_out_zeta_corner_v1(
    const float* __restrict__ u, const float* __restrict__ v,
    float* __restrict__ zeta, int nx, int ny, int nz, float rdx, float rdy) {
  const size_t words = (size_t)nz * (size_t)(ny + 1) * (size_t)(nx + 1);
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < words;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ci = (int)(q % (size_t)(nx + 1));
    const int cj = (int)((q / (size_t)(nx + 1)) % (size_t)(ny + 1));
    const int kz = (int)(q / ((size_t)(nx + 1) * (size_t)(ny + 1)));
    zeta[q] = zeta_corner(u, v, kz, cj, ci, nx, ny, rdx, rdy);
  }
}

// ---------------------------------------------------------------------------
// misclibs.F:1512-1530.  Updraft helicity over the fixed 2-5 km AGL layer.
// Only cyclonically rotating updrafts contribute: w and zeta are separately
// floored at zero before the product.
extern "C" __global__ void cm1_out_calcuh_v1(
    const float* __restrict__ zeta, const float* __restrict__ w,
    const float* __restrict__ zh, const float* __restrict__ zf,
    float* __restrict__ uh, int nx, int ny, int nz,
    float zz0, float zzt) {
  const size_t cols = (size_t)nx * (size_t)ny;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < cols;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)(q / (size_t)nx);
    float acc = 0.0f;
    for (int kz = 0; kz < nz; ++kz) {
      const float zc = zh[kz];
      if (zc >= zz0 && zc <= zzt) {
        float wbar = 0.5f * (WPAD(kz, jj, ii) + WPAD(kz + 1, jj, ii));
        wbar = wbar > 0.0f ? wbar : 0.0f;
        const size_t base = (size_t)kz * (size_t)((ny + 1) * (nx + 1));
        float zbar = 0.25f * (
            zeta[base + (size_t)jj * (size_t)(nx + 1) + (size_t)ii]
          + zeta[base + (size_t)jj * (size_t)(nx + 1) + (size_t)(ii + 1)]
          + zeta[base + (size_t)(jj + 1) * (size_t)(nx + 1) + (size_t)ii]
          + zeta[base + (size_t)(jj + 1) * (size_t)(nx + 1) + (size_t)(ii + 1)]);
        zbar = zbar > 0.0f ? zbar : 0.0f;
        const float zu = zf[kz + 1] < zzt ? zf[kz + 1] : zzt;
        const float zl = zf[kz] > zz0 ? zf[kz] : zz0;
        acc = acc + (zu - zl) * wbar * zbar;
      }
    }
    uh[q] = acc;
  }
}

// ---------------------------------------------------------------------------
// misclibs.F:1622-1683.  x-vorticity: the (i, j-face, k-face) stencil, its
// bbc/tbc closures, then the four-point average onto scalar points.
// ``bbc``/``tbc`` are CM1's own selectors; 1 is free slip (copy the interior
// value down/up) and 2 is no slip (the image-point form, with the frame
// velocity added back when imove=1, misclibs.F:1650-1658).
__device__ __forceinline__ float xvort_tem(
    const float* __restrict__ v, const float* __restrict__ w,
    int kidx, int jf, int ii, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float vmove) {
  if (kidx <= 0) {
    if (bbc == 2) return -2.0f * (VPAD(0, jf, ii) + vmove) * rdz;
    // bbc=3: misclibs.F leaves tem(:,:,1) at the `tem=0.0` fill and
    // overwrites xvort/yvort at k=1 afterwards with the log-layer form
    // (misclibs.F:2160-2168).  That overwrite needs ust/znt/u1/v1/s1, which
    // this diagnostic is not given; the zero below is CM1's literal
    // pre-overwrite value and the selection receipt marks it UNVALIDATED.
    if (bbc == 3) return 0.0f;
    kidx = 1;
  } else if (kidx >= nz) {
    if (tbc == 2) return 2.0f * VPAD(nz - 1, jf, ii) * rdz;
    kidx = nz - 1;
  }
  const float dwdy = (WPAD(kidx, jf, ii) - WPAD(kidx, jf - 1, ii)) * rdy;
  const float dvdz = (VPAD(kidx, jf, ii) - VPAD(kidx - 1, jf, ii)) * rdz;
  return dwdy - dvdz;
}

__device__ __forceinline__ float xvort_at(
    const float* __restrict__ v, const float* __restrict__ w,
    int kz, int jj, int ii, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float vmove) {
  return 0.25f * (
      xvort_tem(v, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove)
    + xvort_tem(v, w, kz, jj + 1, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove)
    + xvort_tem(v, w, kz + 1, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove)
    + xvort_tem(v, w, kz + 1, jj + 1, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove));
}

extern "C" __global__ void cm1_out_xvort_v1(
    const float* __restrict__ v, const float* __restrict__ w,
    float* __restrict__ out, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float vmove) {
  const size_t words = (size_t)nz * (size_t)ny * (size_t)nx;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < words;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)((q / (size_t)nx) % (size_t)ny);
    const int kz = (int)(q / ((size_t)nx * (size_t)ny));
    out[q] = xvort_at(v, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove);
  }
}

// misclibs.F:1855-1898.  y-vorticity on the (i-face, j, k-face) stencil.
__device__ __forceinline__ float yvort_tem(
    const float* __restrict__ u, const float* __restrict__ w,
    int kidx, int jj, int iface, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float umove) {
  if (kidx <= 0) {
    if (bbc == 2) return 2.0f * (UPAD(0, jj, iface) + umove) * rdz;
    if (bbc == 3) return 0.0f;   // see the xvort note above
    kidx = 1;
  } else if (kidx >= nz) {
    if (tbc == 2) return -2.0f * UPAD(nz - 1, jj, iface) * rdz;
    kidx = nz - 1;
  }
  const float dudz = (UPAD(kidx, jj, iface) - UPAD(kidx - 1, jj, iface)) * rdz;
  const float dwdx = (WPAD(kidx, jj, iface) - WPAD(kidx, jj, iface - 1)) * rdx;
  return dudz - dwdx;
}

__device__ __forceinline__ float yvort_at(
    const float* __restrict__ u, const float* __restrict__ w,
    int kz, int jj, int ii, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float umove) {
  return 0.25f * (
      yvort_tem(u, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove)
    + yvort_tem(u, w, kz, jj, ii + 1, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove)
    + yvort_tem(u, w, kz + 1, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove)
    + yvort_tem(u, w, kz + 1, jj, ii + 1, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove));
}

extern "C" __global__ void cm1_out_yvort_v1(
    const float* __restrict__ u, const float* __restrict__ w,
    float* __restrict__ out, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc, float umove) {
  const size_t words = (size_t)nz * (size_t)ny * (size_t)nx;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < words;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)((q / (size_t)nx) % (size_t)ny);
    const int kz = (int)(q / ((size_t)nx * (size_t)ny));
    out[q] = yvort_at(u, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove);
  }
}

// misclibs.F:2033-2053.  z-vorticity: the corner value averaged to scalar pts.
__device__ __forceinline__ float zvort_at(
    const float* __restrict__ u, const float* __restrict__ v,
    int kz, int jj, int ii, int nx, int ny, float rdx, float rdy) {
  return 0.25f * (
      zeta_corner(u, v, kz, jj, ii, nx, ny, rdx, rdy)
    + zeta_corner(u, v, kz, jj, ii + 1, nx, ny, rdx, rdy)
    + zeta_corner(u, v, kz, jj + 1, ii, nx, ny, rdx, rdy)
    + zeta_corner(u, v, kz, jj + 1, ii + 1, nx, ny, rdx, rdy));
}

extern "C" __global__ void cm1_out_zvort_v1(
    const float* __restrict__ u, const float* __restrict__ v,
    float* __restrict__ out, int nx, int ny, int nz, float rdx, float rdy) {
  const size_t words = (size_t)nz * (size_t)ny * (size_t)nx;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < words;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)((q / (size_t)nx) % (size_t)ny);
    const int kz = (int)(q / ((size_t)nx * (size_t)ny));
    out[q] = zvort_at(u, v, kz, jj, ii, nx, ny, rdx, rdy);
  }
}

// ---------------------------------------------------------------------------
// misclibs.F:1687-1706, :1928-1946, :2131-2155.  Potential vorticity as the
// sum of three products of a vorticity component with the matching potential
// temperature gradient, divided by density (CM1 multiplies by rr = 1/rho).
// bc.F's open-lateral scalar halo is the index clamp in SCLAMP.
__device__ __forceinline__ float theta_at(
    const float* __restrict__ th0, const float* __restrict__ th3d,
    int kz, int jj, int ii, int nx, int ny) {
  const int ic = CLAMPI(ii, 0, nx - 1);
  const int jc = CLAMPI(jj, 0, ny - 1);
  const size_t q = (size_t)kz * (size_t)(ny * nx) + (size_t)jc * (size_t)nx + (size_t)ic;
  return th0[q] + th3d[q];
}

extern "C" __global__ void cm1_out_pv_v1(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ w, const float* __restrict__ th0,
    const float* __restrict__ th3d, const float* __restrict__ rho,
    float* __restrict__ out, int nx, int ny, int nz,
    float rdx, float rdy, float rdz, int bbc, int tbc,
    float umove, float vmove,
    float dgs1, float dgs2, float dgs3, float dgt1, float dgt2, float dgt3) {
  const size_t words = (size_t)nz * (size_t)ny * (size_t)nx;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < words;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)((q / (size_t)nx) % (size_t)ny);
    const int kz = (int)(q / ((size_t)nx * (size_t)ny));
    // pv1: x-vorticity times the x-gradient of theta, averaged onto the
    // scalar point from its two bounding u faces (misclibs.F:1697-1704).
    const float th_c = theta_at(th0, th3d, kz, jj, ii, nx, ny);
    const float dthdx_w = (th_c - theta_at(th0, th3d, kz, jj, ii - 1, nx, ny)) * rdx;
    const float dthdx_e = (theta_at(th0, th3d, kz, jj, ii + 1, nx, ny) - th_c) * rdx;
    float pv = xvort_at(v, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, vmove)
             * 0.5f * (dthdx_w + dthdx_e);
    // pv2: y-vorticity times the y-gradient (misclibs.F:1938-1944).
    const float dthdy_s = (th_c - theta_at(th0, th3d, kz, jj - 1, ii, nx, ny)) * rdy;
    const float dthdy_n = (theta_at(th0, th3d, kz, jj + 1, ii, nx, ny) - th_c) * rdy;
    pv = pv + yvort_at(u, w, kz, jj, ii, nx, ny, nz, rdx, rdy, rdz, bbc, tbc, umove)
            * 0.5f * (dthdy_s + dthdy_n);
    // pv3: z-vorticity times the vertical gradient, the two bounding w-level
    // values averaged.  The surface and top values use CM1's Fornberg
    // one-sided derivative weights (param.F:9042-9052).
    float dz_lo, dz_hi;
    if (kz == 0) {
      dz_lo = (dgs3 * theta_at(th0, th3d, 2, jj, ii, nx, ny)
             + dgs2 * theta_at(th0, th3d, 1, jj, ii, nx, ny)
             + dgs1 * th_c) * rdz;
    } else {
      dz_lo = (th_c - theta_at(th0, th3d, kz - 1, jj, ii, nx, ny)) * rdz;
    }
    if (kz == nz - 1) {
      dz_hi = (dgt3 * theta_at(th0, th3d, nz - 3, jj, ii, nx, ny)
             + dgt2 * theta_at(th0, th3d, nz - 2, jj, ii, nx, ny)
             + dgt1 * th_c) * rdz;
    } else {
      dz_hi = (theta_at(th0, th3d, kz + 1, jj, ii, nx, ny) - th_c) * rdz;
    }
    pv = pv + zvort_at(u, v, kz, jj, ii, nx, ny, rdx, rdy) * 0.5f * (dz_lo + dz_hi);
    // solve3.F forms rr as 1.0/rho and misclibs.F:2152 multiplies by it.
    const float rr = 1.0f / rho[q];
    out[q] = pv * rr;
  }
}

// ---------------------------------------------------------------------------
// misclibs.F calccref.  Column maximum of the reflectivity volume.
extern "C" __global__ void cm1_out_calccref_v1(
    const float* __restrict__ dbz, float* __restrict__ cref,
    int nx, int ny, int nz, float seed) {
  const size_t cols = (size_t)nx * (size_t)ny;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < cols;
       q += (size_t)blockDim.x * gridDim.x) {
    float best = seed;
    for (int kz = 0; kz < nz; ++kz) {
      const float value = dbz[(size_t)kz * cols + q];
      if (value > best) best = value;
    }
    cref[q] = best;
  }
}

// ---------------------------------------------------------------------------
// init3d.F:184-198.  Every swath starts at its own sentinel.
extern "C" __global__ void cm1_out_swath_init_v1(
    float* __restrict__ plane, int words, float value) {
  for (int q = (int)(blockIdx.x * blockDim.x + threadIdx.x); q < words;
       q += (int)(blockDim.x * gridDim.x)) {
    plane[q] = value;
  }
}

// solve3.F:857-1130.  One large step of every run-long surface extremum.
// ``sws`` includes the frame velocity because CM1 does (solve3.F:864-865):
// the swath is a ground-relative wind speed.
extern "C" __global__ void cm1_out_swath_update_v1(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ w, const float* __restrict__ prs,
    const float* __restrict__ prs0_profile, const float* __restrict__ qr,
    const float* __restrict__ qg, const float* __restrict__ uh,
    const float* __restrict__ zf,
    float* __restrict__ sws, float* __restrict__ svs,
    float* __restrict__ sps, float* __restrict__ srs,
    float* __restrict__ sgs, float* __restrict__ sus,
    float* __restrict__ shs,
    int nx, int ny, int nz, float rdx, float rdy,
    float umove, float vmove, int sus_face) {
  const size_t cols = (size_t)nx * (size_t)ny;
  for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < cols;
       q += (size_t)blockDim.x * gridDim.x) {
    const int ii = (int)(q % (size_t)nx);
    const int jj = (int)(q / (size_t)nx);
    if (sws != 0) {
      // solve3.F:862-866: ground-relative horizontal speed at k=1.
      const float ubar = umove + 0.5f * (UPAD(0, jj, ii) + UPAD(0, jj, ii + 1));
      const float vbar = vmove + 0.5f * (VPAD(0, jj, ii) + VPAD(0, jj + 1, ii));
      const float speed = sqrtf(ubar * ubar + vbar * vbar);
      if (speed > sws[q]) sws[q] = speed;
    }
    if (svs != 0) {
      // solve3.F:889-894: the corner vorticity at the lowest model level.
      const float tem = zeta_corner(u, v, 0, jj, ii, nx, ny, rdx, rdy);
      if (tem > svs[q]) svs[q] = tem;
    }
    if (sps != 0) {
      // solve3.F:975-979: MINIMUM pressure PERTURBATION at k=1.
      const float tem = prs[q] - prs0_profile[0];
      if (tem < sps[q]) sps[q] = tem;
    }
    if (srs != 0) {
      const float tem = qr[q];
      if (tem > srs[q]) srs[q] = tem;
    }
    if (sgs != 0) {
      const float tem = qg[q];
      if (tem > sgs[q]) sgs[q] = tem;
    }
    if (sus != 0) {
      // solve3.F:1076-1086: w on the first w face at or above 5 km AGL.
      const float tem = WPAD(sus_face, jj, ii);
      if (tem > sus[q]) sus[q] = tem;
    }
    if (shs != 0) {
      const float tem = uh[q];
      if (tem > shs[q]) shs[q] = tem;
    }
  }
}
'''

CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_BYTES = CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE.encode("utf-8")
CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_SHA256 = (
    "9fea1f6fefa0a2131de8c6dd72e95900225a50425bb47760cf03b2a9f49ac43a"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledCm1OutputDiagnostics:
    module_owner: Any
    kernels: tuple[Any, ...]
    by_name: dict[str, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    source_sha256: str


def require_uniform_diagnostic_grid(geometry: Any) -> None:
    """Refuse a geometry whose metric words are not the ones bound here.

    The concrete wrong result: every horizontal derivative in ``calcvort`` and
    ``calcuh`` carries CM1's ``uf(i)``/``vf(j)`` factor and every vertical one
    carries ``mh``/``mf``.  This transcription binds all four to one.  On a
    stretched column the vertical vorticity would still be right and the
    horizontal components wrong by the local metric -- a plausible-looking
    field that is wrong by up to the stretching ratio, published under CM1's
    names.
    """

    from .errors import refuse

    stretch_z = int(getattr(geometry, "stretch_z", 0))
    if stretch_z != 0:
        refuse(
            code="CM1-OUTPUT-DIAG-UNIFORM-METRIC",
            gate="cm1-output-diagnostics",
            reason=(
                f"grid.stretch_z={stretch_z} builds a non-uniform column, but the "
                "calcvort transcription in cuda_cm1_output_diagnostics binds "
                "CM1's mh/mf vertical metric to one (param.F:8170); xvort and yvort "
                "would be wrong by the local stretching ratio at every level and "
                "would still be published under CM1's names"
            ),
            remedy=(
                "set grid.stretch_z=0 for this lane, or turn output_vort and "
                "output_pv off until the stretched metric branch is transcribed"
            ),
            option="grid.stretch_z",
            value=stretch_z,
        )


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return (
        common + (f"--gpu-architecture=compute_{arch}",),
        common + (f"--gpu-architecture=sm_{arch}",),
    )


def compile_cuda_cm1_output_diagnostics() -> CompiledCm1OutputDiagnostics:
    digest = hashlib.sha256(CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_BYTES).hexdigest()
    if digest != CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_SHA256:
        raise RuntimeError(
            "packaged CM1 output-diagnostic CUDA source differs from its pin"
        )
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE, "cm1_out_diag_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE, "cm1_out_diag_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    ptx_text = ptx.decode("utf-8")
    for name in CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES:
        if re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) is None:
            raise ValueError(f"compiled PTX lacks the {name} entry point")
    module = function.Module()
    module.load(bytes(cubin))
    kernels = tuple(
        module.get_function(name) for name in CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES
    )
    return CompiledCm1OutputDiagnostics(
        module_owner=module,
        kernels=kernels,
        by_name=dict(zip(CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES, kernels, strict=True)),
        ptx=bytes(ptx),
        cubin=bytes(cubin),
        target_arch=arch,
        source_sha256=digest,
    )


def diagnostic_receipt() -> dict[str, Any]:
    return {
        "schema": CUDA_CM1_OUTPUT_DIAGNOSTICS_SCHEMA,
        "abi": CUDA_CM1_OUTPUT_DIAGNOSTICS_ABI,
        "kernels": list(CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES),
        "source_sha256": CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_SHA256,
        "source_anchors": dict(SOURCE_ANCHORS),
        "swath_initial_values": dict(CM1_SWATH_INITIAL),
        "uh_layer_m": [CM1_UH_LAYER_BOTTOM_M, CM1_UH_LAYER_TOP_M],
        "claim_limit": CLAIM_LIMIT,
    }


__all__ = [
    "CLAIM_LIMIT",
    "CM1_CREF_INITIAL",
    "CM1_SWATH_INITIAL",
    "CM1_UH_LAYER_BOTTOM_M",
    "CM1_UH_LAYER_TOP_M",
    "CUDA_CM1_OUTPUT_DIAGNOSTICS_ABI",
    "CUDA_CM1_OUTPUT_DIAGNOSTICS_KERNEL_NAMES",
    "CUDA_CM1_OUTPUT_DIAGNOSTICS_SCHEMA",
    "CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE",
    "CUDA_CM1_OUTPUT_DIAGNOSTICS_SOURCE_SHA256",
    "CompiledCm1OutputDiagnostics",
    "SOURCE_ANCHORS",
    "compile_cuda_cm1_output_diagnostics",
    "diagnostic_receipt",
    "require_uniform_diagnostic_grid",
]
