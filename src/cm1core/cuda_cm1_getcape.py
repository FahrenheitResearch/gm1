"""CM1 r21.1 ``src/getcape.F`` as one CUDA kernel, one thread per column.

``getcape`` is CM1's self-contained parcel-ascent diagnostic: it returns
CAPE, CIN and the heights of the LCL, LFC and EL for one sounding.  CM1 calls
it twice -- once per column at output time to make the 2-D ``cape``/``cin``/
``lcl``/``lfc`` fields (``src/writeout.F:2894-2930``), and once at
initialization on the base-state sounding to print the three parcel reports
(``src/base.F:3155-3180``).  Both call sites are reproduced here; the second
launches the same kernel on a one-column grid rather than re-deriving the
thermodynamics in Python.

Everything is transcribed statement-for-statement from ``src/getcape.F``,
including its own constants block (``:98-117``), which deliberately differs
from ``src/constants.F`` -- ``getcape`` carries ``cp=1005.7``, ``rd=287.04``
and ``g=9.81`` of its own, and reproducing CM1 means using *those*.

The three user options at ``src/getcape.F:60-76`` are CM1's compiled-in
values and are exposed as arguments rather than re-chosen: ``pinc=200`` Pa,
``ml_depth=500`` m and ``adiabat=1`` (pseudoadiabatic, liquid only).
``source`` is the parcel selector CM1's call sites pass: ``writeout.F`` uses
3 (mixed layer) and ``base.F`` loops 1, 2, 3.

Accumulators: ``avgth``/``avgqv`` are DOUBLE PRECISION in the authority
(``:92``) and are double here; everything else is CM1 default REAL, which is
binary32 because ``src/Makefile`` selects gfortran without ``-DDP``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import re
from typing import Any

CUDA_CM1_GETCAPE_SCHEMA = "gpuwm-cm1.cuda-cm1-getcape/v1"
CUDA_CM1_GETCAPE_ABI = "gpuwm-cm1.cuda-cm1-getcape-kernels/v1"
CUDA_CM1_GETCAPE_KERNEL_NAMES = ("cm1_getcape_columns_v1", "cm1_getcape_sounding_v1")

SOURCE_ANCHORS = {
    "getcape": "src/getcape.F:14-520",
    "getqvl": "src/getcape.F:525-539",
    "getqvi": "src/getcape.F:544-558",
    "getthx": "src/getcape.F:563-578",
    "gettd": "src/getcape.F:583-596",
    "field_call_site": "src/writeout.F:2894-2930",
    "sounding_call_site": "src/base.F:3140-3180",
    "surface_extrapolation": "src/writeout.F:2912-2915 (cgs1..3, param.F:9038-9040)",
}

#: ``src/getcape.F:60-76``.
CM1_GETCAPE_PINC_PA = 200.0
CM1_GETCAPE_ML_DEPTH_M = 500.0
CM1_GETCAPE_ADIABAT = 1
CM1_GETCAPE_CONVERGE = 0.0002

#: ``src/getcape.F`` returns -1.0 for a level the parcel never reached.
CM1_GETCAPE_ABSENT_LEVEL = -1.0

CM1_GETCAPE_SOURCES = {1: "surface", 2: "most-unstable", 3: "mixed-layer"}

CLAIM_LIMIT = (
    "CM1 r21.1 getcape.F parcel ascent, one thread per column; CM1's own "
    "constants block, pinc=200 Pa, ml_depth=500 m, adiabat=1; no ice "
    "adiabat, no reversible branch, and no forecast-skill claim"
)


CUDA_CM1_GETCAPE_SOURCE = r'''
// CM1 r21.1 src/getcape.F, one CUDA thread per column.
//
// Scratch layout: every per-level array is a (nlev, ncol) global slab so
// neighbouring threads read neighbouring words.  nlev = nz + 1, because
// CM1's call sites prepend one Fornberg-extrapolated surface level.

#define SC(a,k) a[(size_t)(k) * (size_t)ncol + (size_t)col]

__device__ __forceinline__ float gc_getqvl(float p, float t) {
  // getcape.F:525-539
  const float eps = 287.04f / 461.5f;
  float es = 611.2f * expf(17.67f * (t - 273.15f) / (t - 29.65f));
  const float half = p * 0.5f;
  if (es > half) es = half;
  return eps * es / (p - es);
}

__device__ __forceinline__ float gc_getqvi(float p, float t) {
  // getcape.F:544-558
  const float eps = 287.04f / 461.5f;
  float es = 611.2f * expf(21.8745584f * (t - 273.15f) / (t - 7.66f));
  const float half = p * 0.5f;
  if (es > half) es = half;
  return eps * es / (p - es);
}

__device__ __forceinline__ float gc_getthx(float p, float t, float td, float q) {
  // getcape.F:563-578
  float tlcl;
  if ((td - t) >= -0.1f) {
    tlcl = t;
  } else {
    tlcl = 56.0f + 1.0f / (1.0f / (td - 56.0f) + 0.00125f * logf(t / td));
  }
  return t * powf(100000.0f / p, 0.2854f * (1.0f - 0.28f * q))
           * expf(((3376.0f / tlcl) - 2.54f) * q * (1.0f + 0.81f * q));
}

__device__ __forceinline__ float gc_gettd(float p, float t, float q) {
  // getcape.F:583-596
  const float eps = 287.04f / 461.5f;
  const float el = logf((q / eps) * p / 100.0f / (1.0f + (q / eps)));
  return 273.15f + (243.5f * el - 440.8f) / (19.48f - el);
}

// getcape.F:14-520 for one column.  p_in is mb, t_in is Celsius, q_in kg/kg;
// the arrays are the (nlev, ncol) slabs described above.
__device__ void gc_column(
    int source, int nlev, int ncol, int col,
    float* __restrict__ p, float* __restrict__ t, float* __restrict__ td,
    float* __restrict__ pi, float* __restrict__ q, float* __restrict__ th,
    float* __restrict__ thv, float* __restrict__ z,
    float pinc, float ml_depth, int adiabat,
    float* cape_out, float* cin_out, float* zlcl_out, float* zlfc_out,
    float* zel_out, float* psrc_out, float* tsrc_out, float* qsrc_out) {
  const float g = 9.81f, p00 = 100000.0f, cp = 1005.7f, rd = 287.04f, rv = 461.5f;
  const float xlv = 2501000.0f, xls = 2836017.0f, t0 = 273.15f;
  const float cpv = 1875.0f, cpl = 4190.0f, cpi = 2118.636f;
  const float lv1 = xlv + (cpl - cpv) * t0, lv2 = cpl - cpv;
  const float ls1 = xls + (cpi - cpv) * t0, ls2 = cpi - cpv;
  const float rp00 = 1.0f / p00, eps = rd / rv, reps = rv / rd;
  const float rddcp = rd / cp, cpdg = cp / g;
  const float converge = 0.0002f;

  // ---- convert to mks; get pi, q, th, thv (getcape.F:132-144) ----
  for (int k = 0; k < nlev; ++k) {
    const float pk = 100.0f * SC(p, k);
    const float tk = 273.15f + SC(t, k);
    float qk = SC(q, k);
    if (!(qk > 1.0e-10f)) qk = 1.0e-10f;
    const float ee = logf((qk / eps) * pk / 100.0f / (1.0f + (qk / eps)));
    SC(p, k) = pk;
    SC(t, k) = tk;
    SC(q, k) = qk;
    SC(td, k) = 273.15f + (243.5f * ee - 440.8f) / (19.48f - ee);
    const float pik = powf(pk * rp00, rddcp);
    SC(pi, k) = pik;
    const float thk = tk / pik;
    SC(th, k) = thk;
    SC(thv, k) = thk * (1.0f + reps * qk) / (1.0f + qk);
  }

  // ---- hydrostatic heights (getcape.F:148-153) ----
  SC(z, 0) = 0.0f;
  for (int k = 1; k < nlev; ++k) {
    const float dz = -cpdg * 0.5f * (SC(thv, k) + SC(thv, k - 1))
                   * (SC(pi, k) - SC(pi, k - 1));
    SC(z, k) = SC(z, k - 1) + dz;
  }

  // ---- source parcel (getcape.F:157-247) ----
  int kmax = 0;
  double avgth = 0.0, avgqv = 0.0;
  if (source == 1) {
    kmax = 0;
  } else if (source == 2) {
    if (SC(p, 0) < 50000.0f) {
      kmax = 0;
    } else {
      float maxthe = 0.0f;
      for (int k = 0; k < nlev; ++k) {
        if (SC(p, k) >= 50000.0f) {
          const float the = gc_getthx(SC(p, k), SC(t, k), SC(td, k), SC(q, k));
          if (the > maxthe) { maxthe = the; kmax = k; }
        }
      }
    }
  } else {
    if ((SC(z, 1) - SC(z, 0)) > ml_depth) {
      avgth = SC(th, 0); avgqv = SC(q, 0); kmax = 0;
    } else if (SC(z, nlev - 1) < ml_depth) {
      avgth = SC(th, nlev - 1); avgqv = SC(q, nlev - 1); kmax = nlev - 1;
    } else {
      int k = 1;
      while (k < nlev && SC(z, k) <= ml_depth) {
        avgth += 0.5 * (double)(SC(z, k) - SC(z, k - 1))
               * (double)(SC(th, k) + SC(th, k - 1));
        avgqv += 0.5 * (double)(SC(z, k) - SC(z, k - 1))
               * (double)(SC(q, k) + SC(q, k - 1));
        k = k + 1;
      }
      if (k >= nlev) k = nlev - 1;
      const float frac = (ml_depth - SC(z, k - 1)) / (SC(z, k) - SC(z, k - 1));
      const float th2i = SC(th, k - 1) + (SC(th, k) - SC(th, k - 1)) * frac;
      const float qv2i = SC(q, k - 1) + (SC(q, k) - SC(q, k - 1)) * frac;
      avgth += 0.5 * (double)(ml_depth - SC(z, k - 1)) * (double)(th2i + SC(th, k - 1));
      avgqv += 0.5 * (double)(ml_depth - SC(z, k - 1)) * (double)(qv2i + SC(q, k - 1));
      avgth = avgth / (double)ml_depth;
      avgqv = avgqv / (double)ml_depth;
      kmax = 0;
    }
  }

  // ---- parcel properties at the initial location (getcape.F:251-281) ----
  float narea = 0.0f;
  int k = kmax;
  float th2, pi2, p2, t2, thv2, qv2, b2;
  if (source == 1 || source == 2) {
    th2 = SC(th, kmax); pi2 = SC(pi, kmax); p2 = SC(p, kmax);
    t2 = SC(t, kmax); thv2 = SC(thv, kmax); qv2 = SC(q, kmax); b2 = 0.0f;
  } else {
    th2 = (float)avgth; qv2 = (float)avgqv;
    thv2 = th2 * (1.0f + reps * qv2) / (1.0f + qv2);
    pi2 = SC(pi, kmax); p2 = SC(p, kmax); t2 = th2 * pi2;
    b2 = g * (thv2 - SC(thv, kmax)) / SC(thv, kmax);
  }
  *psrc_out = p2; *tsrc_out = t2; *qsrc_out = qv2;

  float ql2 = 0.0f, qi2 = 0.0f, qt = qv2;
  float cape = 0.0f, cin = 0.0f;
  int doit = 1, cloud = 0;
  const int ice = (adiabat == 1 || adiabat == 2) ? 0 : 1;
  float zlcl = -1.0f, zlfc = -1.0f, zel = -1.0f, plcl = 0.0f;

  // ---- ascent (getcape.F:316-512) ----
  while (doit && (k < nlev - 1)) {
    k = k + 1;
    const float b1 = b2;
    float dp = SC(p, k - 1) - SC(p, k);
    int nloop;
    if (dp < pinc) { nloop = 1; }
    else { nloop = 1 + (int)(dp / pinc); dp = dp / (float)nloop; }

    for (int n = 1; n <= nloop; ++n) {
      const float p1 = p2, t1 = t2, pi1 = pi2, th1 = th2;
      const float qv1 = qv2, ql1 = ql2, qi1 = qi2;
      (void)pi1; (void)t1;
      p2 = p2 - dp;
      pi2 = powf(p2 * rp00, rddcp);
      float thlast = th1;
      int it = 0, not_converged = 1;
      while (not_converged) {
        it = it + 1;
        t2 = thlast * pi2;
        float fliq, fice;
        if (ice) {
          fliq = (t2 - 233.15f) / (273.15f - 233.15f);
          if (fliq > 1.0f) fliq = 1.0f;
          if (fliq < 0.0f) fliq = 0.0f;
          fice = 1.0f - fliq;
        } else { fliq = 1.0f; fice = 0.0f; }
        float qs = fliq * gc_getqvl(p2, t2) + fice * gc_getqvi(p2, t2);
        qv2 = qt < qs ? qt : qs;
        qi2 = fice * (qt - qv2); if (qi2 < 0.0f) qi2 = 0.0f;
        ql2 = qt - qv2 - qi2;   if (ql2 < 0.0f) ql2 = 0.0f;
        const float tbar = 0.5f * (t1 + t2), qvbar = 0.5f * (qv1 + qv2);
        const float qlbar = 0.5f * (ql1 + ql2), qibar = 0.5f * (qi1 + qi2);
        const float lhv = lv1 - lv2 * tbar, lhs = ls1 - ls2 * tbar;
        const float rm = rd + rv * qvbar;
        const float cpm = cp + cpv * qvbar + cpl * qlbar + cpi * qibar;
        th2 = th1 * expf(lhv * (ql2 - ql1) / (cpm * tbar)
                       + lhs * (qi2 - qi1) / (cpm * tbar)
                       + (rm / cpm - rd / cp) * logf(p2 / p1));
        if (it > 100) {
          // getcape.F:376-388: lack of convergence returns zeros.
          *cape_out = 0.0f; *cin_out = 0.0f;
          *zlcl_out = -1.0f; *zlfc_out = -1.0f; *zel_out = -1.0f;
          *psrc_out = 0.0f; *tsrc_out = 0.0f; *qsrc_out = 0.0f;
          return;
        }
        if (fabsf(th2 - thlast) > converge) { thlast = thlast + 0.3f * (th2 - thlast); }
        else { not_converged = 0; }
      }
      if (ql2 >= 1.0e-10f) cloud = 1;
      if (cloud && zlcl < 0.0f) {
        zlcl = SC(z, k - 1) + (SC(z, k) - SC(z, k - 1)) * (float)n / (float)nloop;
        plcl = SC(p, k - 1) + (SC(p, k) - SC(p, k - 1)) * (float)n / (float)nloop;
      }
      if (adiabat == 1 || adiabat == 3) { qt = qv2; ql2 = 0.0f; qi2 = 0.0f; }
    }
    (void)plcl;

    thv2 = th2 * (1.0f + reps * qv2) / (1.0f + qv2 + ql2 + qi2);
    b2 = g * (thv2 - SC(thv, k)) / SC(thv, k);
    const float dz = -cpdg * 0.5f * (SC(thv, k) + SC(thv, k - 1))
                   * (SC(pi, k) - SC(pi, k - 1));

    if (zlcl > 0.0f && zlfc < 0.0f && b2 > 0.0f) {
      if (b1 > 0.0f) { zlfc = zlcl; }
      else { zlfc = SC(z, k - 1) + (SC(z, k) - SC(z, k - 1)) * (0.0f - b1) / (b2 - b1); }
    }
    if (zlfc > 0.0f && zel < 0.0f && b2 < 0.0f) {
      zel = SC(z, k - 1) + (SC(z, k) - SC(z, k - 1)) * (0.0f - b1) / (b2 - b1);
    }

    float parea;
    if ((b2 >= 0.0f) && (b1 < 0.0f)) {
      const float frac = b2 / (b2 - b1);
      parea = 0.5f * b2 * dz * frac;
      narea = narea - 0.5f * b1 * dz * (1.0f - frac);
      cin = cin + narea;
      narea = 0.0f;
    } else if ((b2 < 0.0f) && (b1 > 0.0f)) {
      const float frac = b1 / (b1 - b2);
      parea = 0.5f * b1 * dz * frac;
      narea = -0.5f * b2 * dz * (1.0f - frac);
    } else if (b2 < 0.0f) {
      parea = 0.0f;
      narea = narea - 0.5f * dz * (b1 + b2);
    } else {
      parea = 0.5f * dz * (b1 + b2);
      narea = 0.0f;
    }
    cape = cape + (parea > 0.0f ? parea : 0.0f);

    if ((SC(p, k) <= 10000.0f) && (b2 < 0.0f)) doit = 0;
  }

  *cape_out = cape; *cin_out = cin;
  *zlcl_out = zlcl; *zlfc_out = zlfc; *zel_out = zel;
}

// writeout.F:2894-2930.  Build each column from the model state, prepend the
// Fornberg-extrapolated surface level, then run the ascent.
extern "C" __global__ void cm1_getcape_columns_v1(
    const float* __restrict__ prs, const float* __restrict__ th0,
    const float* __restrict__ th3d, const float* __restrict__ pi0,
    const float* __restrict__ pp3d, const float* __restrict__ qv,
    float* __restrict__ sp, float* __restrict__ st, float* __restrict__ std_,
    float* __restrict__ spi, float* __restrict__ sq, float* __restrict__ sth,
    float* __restrict__ sthv, float* __restrict__ sz,
    float* __restrict__ cape, float* __restrict__ cin,
    float* __restrict__ lcl, float* __restrict__ lfc,
    int nx, int ny, int nz, int source,
    float pinc, float ml_depth, int adiabat,
    float cgs1, float cgs2, float cgs3) {
  const int ncol = nx * ny;
  const int nlev = nz + 1;
  for (int col = (int)(blockIdx.x * blockDim.x + threadIdx.x); col < ncol;
       col += (int)(blockDim.x * gridDim.x)) {
    for (int k = 0; k < nz; ++k) {
      const size_t q = (size_t)k * (size_t)ncol + (size_t)col;
      SC(sp, k + 1) = 0.01f * prs[q];
      SC(st, k + 1) = (th0[q] + th3d[q]) * (pi0[q] + pp3d[q]) - 273.15f;
      SC(sq, k + 1) = qv[q];
    }
    SC(sp, 0) = cgs1 * SC(sp, 1) + cgs2 * SC(sp, 2) + cgs3 * SC(sp, 3);
    SC(st, 0) = cgs1 * SC(st, 1) + cgs2 * SC(st, 2) + cgs3 * SC(st, 3);
    SC(sq, 0) = cgs1 * SC(sq, 1) + cgs2 * SC(sq, 2) + cgs3 * SC(sq, 3);
    float c, i, l, f, e, ps, ts, qs;
    gc_column(source, nlev, ncol, col, sp, st, std_, spi, sq, sth, sthv, sz,
              pinc, ml_depth, adiabat, &c, &i, &l, &f, &e, &ps, &ts, &qs);
    cape[col] = c; cin[col] = i; lcl[col] = l; lfc[col] = f;
  }
}

// base.F:3140-3180.  One already-assembled sounding, reported in full.
extern "C" __global__ void cm1_getcape_sounding_v1(
    float* __restrict__ sp, float* __restrict__ st, float* __restrict__ std_,
    float* __restrict__ spi, float* __restrict__ sq, float* __restrict__ sth,
    float* __restrict__ sthv, float* __restrict__ sz,
    float* __restrict__ out, int nlev, int source,
    float pinc, float ml_depth, int adiabat) {
  const int ncol = 1, col = 0;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float c, i, l, f, e, ps, ts, qs;
  gc_column(source, nlev, ncol, col, sp, st, std_, spi, sq, sth, sthv, sz,
            pinc, ml_depth, adiabat, &c, &i, &l, &f, &e, &ps, &ts, &qs);
  out[0] = c; out[1] = i; out[2] = l; out[3] = f; out[4] = e;
  out[5] = ps; out[6] = ts; out[7] = qs;
}
'''

CUDA_CM1_GETCAPE_SOURCE_BYTES = CUDA_CM1_GETCAPE_SOURCE.encode("utf-8")
CUDA_CM1_GETCAPE_SOURCE_SHA256 = (
    "297cc88c58664cff57e3a128b21f3b41d50e94cd661c64feaa165aa98dc5c9de"
)

#: Per-column scratch slabs the kernel needs, each ``(nz + 1, ny * nx)``.
GETCAPE_SCRATCH_NAMES = ("p", "t", "td", "pi", "q", "th", "thv", "z")


def getcape_scratch_bytes(nx: int, ny: int, nz: int) -> int:
    return len(GETCAPE_SCRATCH_NAMES) * (nz + 1) * ny * nx * 4


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
class CompiledCm1Getcape:
    module_owner: Any
    kernels: tuple[Any, ...]
    by_name: dict[str, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    source_sha256: str


def compile_cuda_cm1_getcape() -> CompiledCm1Getcape:
    digest = hashlib.sha256(CUDA_CM1_GETCAPE_SOURCE_BYTES).hexdigest()
    if digest != CUDA_CM1_GETCAPE_SOURCE_SHA256:
        raise RuntimeError("packaged getcape CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    arch = _canonical_arch(cupy.cuda.Device().compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14", "--fmad=false", "--ftz=false",
        "--prec-div=true", "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_CM1_GETCAPE_SOURCE, "cm1_getcape_v1_ptx.cu", method="ptx"
    ).compile(common + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_CM1_GETCAPE_SOURCE, "cm1_getcape_v1_cubin.cu", method="cubin"
    ).compile(common + (f"--gpu-architecture=sm_{arch}",))
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    ptx_text = ptx.decode("utf-8")
    for name in CUDA_CM1_GETCAPE_KERNEL_NAMES:
        if re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) is None:
            raise ValueError(f"compiled PTX lacks the {name} entry point")
    module = function.Module()
    module.load(bytes(cubin))
    kernels = tuple(module.get_function(name) for name in CUDA_CM1_GETCAPE_KERNEL_NAMES)
    return CompiledCm1Getcape(
        module_owner=module,
        kernels=kernels,
        by_name=dict(zip(CUDA_CM1_GETCAPE_KERNEL_NAMES, kernels, strict=True)),
        ptx=bytes(ptx), cubin=bytes(cubin), target_arch=arch, source_sha256=digest,
    )


def getcape_receipt() -> dict[str, Any]:
    return {
        "schema": CUDA_CM1_GETCAPE_SCHEMA,
        "abi": CUDA_CM1_GETCAPE_ABI,
        "kernels": list(CUDA_CM1_GETCAPE_KERNEL_NAMES),
        "source_sha256": CUDA_CM1_GETCAPE_SOURCE_SHA256,
        "source_anchors": dict(SOURCE_ANCHORS),
        "pinc_pa": CM1_GETCAPE_PINC_PA,
        "ml_depth_m": CM1_GETCAPE_ML_DEPTH_M,
        "adiabat": CM1_GETCAPE_ADIABAT,
        "converge": CM1_GETCAPE_CONVERGE,
        "absent_level_value": CM1_GETCAPE_ABSENT_LEVEL,
        "claim_limit": CLAIM_LIMIT,
    }


__all__ = [
    "CLAIM_LIMIT",
    "CM1_GETCAPE_ABSENT_LEVEL",
    "CM1_GETCAPE_ADIABAT",
    "CM1_GETCAPE_CONVERGE",
    "CM1_GETCAPE_ML_DEPTH_M",
    "CM1_GETCAPE_PINC_PA",
    "CM1_GETCAPE_SOURCES",
    "CUDA_CM1_GETCAPE_ABI",
    "CUDA_CM1_GETCAPE_KERNEL_NAMES",
    "CUDA_CM1_GETCAPE_SCHEMA",
    "CUDA_CM1_GETCAPE_SOURCE",
    "CUDA_CM1_GETCAPE_SOURCE_SHA256",
    "CompiledCm1Getcape",
    "GETCAPE_SCRATCH_NAMES",
    "SOURCE_ANCHORS",
    "compile_cuda_cm1_getcape",
    "getcape_receipt",
    "getcape_scratch_bytes",
]
