"""Bounded CUDA column specialization of CM1 r21.1 ``psolver=3`` sound.

The implementation retains the native three-stage small-step cadence,
flat-grid vertical coefficient construction, sequential tridiagonal solve,
pressure update/filter, time-averaged vertical velocity, and final-stage
``ppx`` hold.  Each batch member represents an independent horizontally
uniform column.  Horizontal pressure-gradient momentum updates, horizontal
divergence, boundaries, terrain, moisture, and MPI are outside this slice.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping


CUDA_SOUND3_COLUMN_SCHEMA = "gpuwm-cm1.cuda-sound3-column/v1"
CUDA_SOUND3_COLUMN_ABI = "gpuwm-cm1.cuda-sound3-column-kernels/v1"
CUDA_SOUND3_COLUMN_THREADS = 128
CM1_SOUND3_DEFAULT_ALPH_F32 = 0.60
CM1_SOUND3_DEFAULT_KDIV_F32 = 0.10
CM1_SOUND3_DEFAULT_CP_F32 = 1005.7
CM1_SOUND3_SMEPS_F32 = 1.0e-30
CUDA_SOUND3_COLUMN_CLAIM_LIMIT = (
    "CM1 r21.1 psolver=3 flat horizontally uniform independent-column "
    "specialization with native RK-stage nloop/dts cadence, source-ordered "
    "vertical implicit tridiagonal solve, pressure update/filter, and held "
    "final-stage ppx; no horizontal pressure-gradient momentum, horizontal "
    "divergence, terrain, moisture, boundary, MPI, full sound, trajectory, "
    "or forecast claim"
)

SOURCE_ANCHORS = {
    "solve2_dispatch": "src/solve2.F:1119-1138",
    "rk_small_step_schedule": "src/sound.F:81-113",
    "flat_vertical_coefficients": "src/sound.F:117-142",
    "coefficient_boundary": "src/sound.F:181-188",
    "stage_preparation": "src/sound.F:191-248",
    "small_step_loop": "src/sound.F:255-837",
    "flat_uniform_fpk": "src/sound.F:487-509",
    "tridiagonal_solve": "src/sound.F:659-730",
    "flat_pressure_update": "src/sound.F:739-761",
    "vertical_time_average": "src/sound.F:806-834",
    "final_stage_ppx": "src/sound.F:842-851",
    "nrkmax": "src/constants.F:61",
    "default_cp": "src/constants.F:109-116",
    "official_profile_parameters": "run/namelist.input:95-96",
}

INPUT_ARRAY_NAMES = (
    "mh",
    "rr0",
    "rf0",
    "c1",
    "c2",
    "mf",
    "piadv",
    "ppterm",
    "thv",
    "ppten",
    "wten",
    "wa",
    "ppi",
)
STATE_ARRAY_NAMES = ("w3d", "pp3d", "ppx", "ppd", "rrw")
SCRATCH_ARRAY_NAMES = (
    "qk",
    "pk1",
    "pk2",
    "mm",
    "fpk",
    "elim_c",
    "elim_d",
    "tri_aa",
    "tri_bb",
    "tri_cc",
    "tri_dd",
    "tri_pivot",
    "tri_residual",
    "ppx_stage1",
    "ppx_stage2",
    "ppx_stage3",
)
ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + STATE_ARRAY_NAMES + SCRATCH_ARRAY_NAMES
CUDA_SOUND3_COLUMN_KERNEL_NAMES = (
    "cm1_sound3_column_stage_v1",
    "cm1_copy_f32_v1",
)

CUDA_SOUND3_COLUMN_SOURCE = r"""// gpuwm-cm1.cuda-sound3-column-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float cm1_add(const float a, const float b)
{
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float cm1_sub(const float a, const float b)
{
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float cm1_mul(const float a, const float b)
{
    return __fmul_rn(a, b);
}

__device__ __forceinline__ float cm1_div(const float a, const float b)
{
    return __fdiv_rn(a, b);
}

__device__ __forceinline__ unsigned long long cm1_idx(
    const unsigned int column,
    const unsigned int k,
    const unsigned int stride)
{
    return (unsigned long long)column * (unsigned long long)stride
        + (unsigned long long)k;
}

extern "C" __global__
void cm1_copy_f32_v1(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const unsigned long long count)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long step =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
    for (unsigned long long index = first; index < count; index += step) {
        destination[index] = source[index];
    }
}

extern "C" __global__
void cm1_sound3_column_stage_v1(
    const float* __restrict__ mh,
    const float* __restrict__ rr0,
    const float* __restrict__ rf0,
    const float* __restrict__ c1,
    const float* __restrict__ c2,
    const float* __restrict__ mf,
    const float* __restrict__ piadv,
    const float* __restrict__ ppterm,
    const float* __restrict__ thv,
    const float* __restrict__ ppten,
    const float* __restrict__ wten,
    const float* __restrict__ wa,
    const float* __restrict__ ppi,
    float* __restrict__ w3d,
    float* __restrict__ pp3d,
    float* __restrict__ ppx,
    float* __restrict__ ppd,
    float* __restrict__ rrw,
    float* __restrict__ qk,
    float* __restrict__ pk1,
    float* __restrict__ pk2,
    float* __restrict__ mm,
    float* __restrict__ fpk,
    float* __restrict__ elim_c,
    float* __restrict__ elim_d,
    float* __restrict__ tri_aa,
    float* __restrict__ tri_bb,
    float* __restrict__ tri_cc,
    float* __restrict__ tri_dd,
    float* __restrict__ tri_pivot,
    float* __restrict__ tri_residual,
    const unsigned int ncol,
    const unsigned int nk,
    const unsigned int stride,
    const unsigned int nrk,
    const unsigned int nloop,
    const float dts,
    const float rdz,
    const float cp,
    const float alph,
    const float beta,
    const float asq,
    const float kdiv,
    const float smeps)
{
    const unsigned int first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_step = blockDim.x * gridDim.x;
    for (unsigned int column = first; column < ncol; column += grid_step) {
        const unsigned long long base =
            (unsigned long long)column * (unsigned long long)stride;

        for (unsigned int k = 0u; k < stride; ++k) {
            const unsigned long long index = base + (unsigned long long)k;
            qk[index] = 0.0f;
            pk1[index] = 0.0f;
            pk2[index] = 0.0f;
            mm[index] = 0.0f;
            fpk[index] = 0.0f;
            elim_c[index] = 0.0f;
            elim_d[index] = 0.0f;
            tri_aa[index] = 0.0f;
            tri_bb[index] = 0.0f;
            tri_cc[index] = 0.0f;
            tri_dd[index] = 0.0f;
            tri_pivot[index] = 0.0f;
            tri_residual[index] = 0.0f;
        }

        for (unsigned int k = 1u; k <= nk; ++k) {
            const unsigned long long index = cm1_idx(column, k, stride);
            const unsigned long long below = cm1_idx(column, k - 1u, stride);
            const unsigned long long above = cm1_idx(column, k + 1u, stride);
            const float tem = cm1_mul(cm1_mul(dts, rdz), mh[index]);
            float r2 = cm1_mul(cm1_mul(dts, rdz), mh[index]);
            r2 = cm1_mul(r2, rr0[index]);
            r2 = cm1_mul(r2, rf0[above]);
            float r1 = cm1_mul(cm1_mul(dts, rdz), mh[index]);
            r1 = cm1_mul(r1, rr0[index]);
            r1 = cm1_mul(r1, rf0[index]);
            const float tem1 = cm1_mul(cm1_mul(cm1_mul(dts, rdz), cp), mf[index]);
            qk[index] = cm1_mul(tem, ppterm[index]);
            const float pk2_left = cm1_mul(-c2[above], piadv[above]);
            const float pk2_right = cm1_mul(cm1_sub(1.0f, c1[above]), piadv[index]);
            pk2[index] = cm1_mul(r2, cm1_add(pk2_left, pk2_right));
            const float pk1_left = cm1_mul(c1[index], piadv[below]);
            const float pk1_right = cm1_mul(cm1_sub(c2[index], 1.0f), piadv[index]);
            pk1[index] = cm1_mul(r1, cm1_add(pk1_left, pk1_right));
            const float theta_face = cm1_add(
                cm1_mul(c2[index], thv[index]),
                cm1_mul(c1[index], thv[below]));
            mm[index] = cm1_mul(tem1, theta_face);
        }
        pk1[cm1_idx(column, 1u, stride)] = 0.0f;
        pk2[cm1_idx(column, nk, stride)] = 0.0f;

        for (unsigned int k = 1u; k <= nk; ++k) {
            const unsigned long long index = cm1_idx(column, k, stride);
            ppd[index] = ppx[index];
            rrw[index] = cm1_mul(wa[index], beta);
            if (nrk != 1u) {
                w3d[index] = wa[index];
                pp3d[index] = ppi[index];
            }
        }

        for (unsigned int n = 1u; n <= nloop; ++n) {
            for (unsigned int k = 1u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                const unsigned long long above = cm1_idx(column, k + 1u, stride);
                const float zero_divergence = 0.0f;
                const float source = cm1_sub(
                    ppten[index], cm1_mul(ppterm[index], zero_divergence));
                const float vertical = cm1_add(
                    cm1_mul(w3d[index], cm1_add(pk1[index], qk[index])),
                    cm1_mul(w3d[above], cm1_sub(pk2[index], qk[index])));
                fpk[index] = cm1_add(
                    cm1_mul(dts, source), cm1_mul(beta, vertical));
            }

            for (unsigned int k = 2u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                const unsigned long long below = cm1_idx(column, k - 1u, stride);
                float aa = 0.0f;
                float cc = 0.0f;
                if (k >= 3u) {
                    aa = cm1_mul(
                        -asq,
                        cm1_mul(mm[index], cm1_add(pk1[below], qk[below])));
                }
                if (k < nk) {
                    cc = cm1_mul(
                        -asq,
                        cm1_mul(-mm[index], cm1_sub(pk2[index], qk[index])));
                }
                const float diagonal_sum = cm1_add(
                    -cm1_add(pk1[index], qk[index]),
                    cm1_sub(pk2[below], qk[below]));
                const float bb = cm1_sub(
                    1.0f, cm1_mul(asq, cm1_mul(mm[index], diagonal_sum)));
                const float pressure_difference = cm1_sub(pp3d[index], pp3d[below]);
                const float fwk = cm1_add(
                    cm1_mul(dts, wten[index]),
                    cm1_mul(beta, cm1_mul(-mm[index], pressure_difference)));
                const float updated_difference = cm1_sub(
                    cm1_add(pp3d[index], fpk[index]),
                    cm1_add(pp3d[below], fpk[below]));
                float dd = cm1_add(w3d[index], fwk);
                dd = cm1_add(dd, cm1_mul(alph, cm1_mul(-mm[index], updated_difference)));
                float pivot = bb;
                if (k >= 3u) {
                    pivot = cm1_add(cm1_mul(aa, elim_c[below]), bb);
                }
                const float inverse = cm1_div(1.0f, pivot);
                tri_aa[index] = aa;
                tri_bb[index] = bb;
                tri_cc[index] = cc;
                tri_dd[index] = dd;
                tri_pivot[index] = pivot;
                if (k < nk) {
                    elim_c[index] = cm1_mul(-cc, inverse);
                    float numerator = dd;
                    if (k >= 3u) {
                        numerator = cm1_sub(dd, cm1_mul(aa, elim_d[below]));
                    }
                    elim_d[index] = cm1_mul(numerator, inverse);
                } else {
                    const float numerator = cm1_sub(dd, cm1_mul(aa, elim_d[below]));
                    w3d[index] = cm1_mul(numerator, inverse);
                }
            }

            for (int k = (int)nk - 1; k >= 2; --k) {
                const unsigned long long index = cm1_idx(column, (unsigned int)k, stride);
                const unsigned long long above = cm1_idx(column, (unsigned int)k + 1u, stride);
                w3d[index] = cm1_add(
                    cm1_mul(elim_c[index], w3d[above]), elim_d[index]);
            }

            for (unsigned int k = 2u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                const unsigned long long below = cm1_idx(column, k - 1u, stride);
                const unsigned long long above = cm1_idx(column, k + 1u, stride);
                float lhs = cm1_mul(tri_aa[index], w3d[below]);
                lhs = cm1_add(lhs, cm1_mul(tri_bb[index], w3d[index]));
                lhs = cm1_add(lhs, cm1_mul(tri_cc[index], w3d[above]));
                tri_residual[index] = cm1_sub(lhs, tri_dd[index]);
            }

            for (unsigned int k = 1u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                const unsigned long long above = cm1_idx(column, k + 1u, stride);
                const float old_pressure = pp3d[index];
                const float vertical = cm1_add(
                    cm1_mul(w3d[index], cm1_add(pk1[index], qk[index])),
                    cm1_mul(w3d[above], cm1_sub(pk2[index], qk[index])));
                float updated = cm1_add(old_pressure, fpk[index]);
                updated = cm1_add(updated, cm1_mul(alph, vertical));
                if (fabsf(updated) < smeps) {
                    updated = 0.0f;
                }
                pp3d[index] = updated;
                elim_c[index] = cm1_mul(kdiv, cm1_sub(updated, old_pressure));
                ppd[index] = cm1_add(updated, elim_c[index]);
            }

            for (unsigned int k = 1u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                if (n < nloop) {
                    rrw[index] = cm1_add(rrw[index], w3d[index]);
                } else {
                    const float tavg = cm1_div(1.0f, (float)nloop);
                    rrw[index] = cm1_mul(
                        cm1_add(rrw[index], cm1_mul(w3d[index], alph)), tavg);
                }
            }
        }

        if (nrk == 3u) {
            for (unsigned int k = 1u; k <= nk; ++k) {
                const unsigned long long index = cm1_idx(column, k, stride);
                ppx[index] = elim_c[index];
            }
        }
    }
}
"""

CUDA_SOUND3_COLUMN_SOURCE_BYTES = CUDA_SOUND3_COLUMN_SOURCE.encode("utf-8")
CUDA_SOUND3_COLUMN_SOURCE_SHA256 = (
    "ee01ba1cad35539f2f9a06972aee4da12acf9f2afe9c2401ec2b379fb50f6bc0"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_SOUND3_COLUMN_SOURCE_BYTES).hexdigest() == (
        CUDA_SOUND3_COLUMN_SOURCE_SHA256
    )


def column_shape(ncol: int, nk: int) -> tuple[int, int]:
    if isinstance(ncol, bool) or not isinstance(ncol, int) or ncol < 1:
        raise ValueError("ncol must be an integer >= 1")
    if isinstance(nk, bool) or not isinstance(nk, int) or nk < 3:
        raise ValueError("nk must be an integer >= 3")
    return ncol, nk + 2


def sound3_stage_schedule_f32(dt: float, nsound: int, nrk: int) -> tuple[int, float]:
    """Transcribe ``sound.F:84-108`` for the fixed three-stage RK scheme."""

    if isinstance(nsound, bool) or not isinstance(nsound, int) or nsound < 4:
        raise ValueError("nsound must be an integer >= 4")
    if nsound % 2:
        raise ValueError("native diagnostic cadence requires even nsound")
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be one of 1, 2, 3")
    np = importlib.import_module("numpy")
    dt32 = np.float32(dt)
    if nrk == 1:
        nloop = int(np.floor(np.float32(nsound) / np.float32(3.0) + np.float32(0.5)))
        denominator = np.float32(np.float32(nloop) * np.float32(3.0))
        dts = np.float32(dt32 / denominator)
        target = np.float32(dt32 / np.float32(nsound))
        if bool(dts > target):
            nloop += 1
            denominator = np.float32(np.float32(nloop) * np.float32(3.0))
            dts = np.float32(dt32 / denominator)
    elif nrk == 2:
        nloop = nsound // 2
        dts = np.float32(dt32 / np.float32(nsound))
    else:
        nloop = nsound
        dts = np.float32(dt32 / np.float32(nsound))
    return nloop, float(dts)


def rk_stage_dt_f32(dt: float, nrk: int) -> float:
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be one of 1, 2, 3")
    np = importlib.import_module("numpy")
    return float(np.float32(np.float32(dt) / np.float32(4 - nrk)))


def _require_host_arrays(
    arrays: Mapping[str, Any], ncol: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shape = column_shape(ncol, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing sound3 arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name not in INPUT_ARRAY_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("sound3 column arrays must not alias")
    return result


def _f32_add(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) + np.float32(b))


def _f32_sub(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) - np.float32(b))


def _f32_mul(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) * np.float32(b))


def _f32_div(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) / np.float32(b))


@dataclass(frozen=True, slots=True)
class CpuSound3StageDiagnostics:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    ppx_written: bool


def cpu_sound3_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ncol: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    rdz: float,
    cp: float = CM1_SOUND3_DEFAULT_CP_F32,
    alph: float = CM1_SOUND3_DEFAULT_ALPH_F32,
    kdiv: float = CM1_SOUND3_DEFAULT_KDIV_F32,
    smeps: float = CM1_SOUND3_SMEPS_F32,
) -> CpuSound3StageDiagnostics:
    """Execute one source-ordered, horizontally uniform ``sound`` RK stage."""

    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ncol, nk)
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts = np.float32(dts_value)
    rdz32 = np.float32(rdz)
    cp32 = np.float32(cp)
    alph32 = np.float32(alph)
    beta = _f32_sub(np, np.float32(1.0), alph32)
    asq = _f32_mul(np, alph32, alph32)
    kdiv32 = np.float32(kdiv)
    smeps32 = np.float32(smeps)

    for name in SCRATCH_ARRAY_NAMES[:13]:
        a[name].fill(np.float32(0.0))

    for column in range(ncol):
        for k in range(1, nk + 1):
            tem = _f32_mul(np, _f32_mul(np, dts, rdz32), a["mh"][column, k])
            r2 = _f32_mul(np, _f32_mul(np, dts, rdz32), a["mh"][column, k])
            r2 = _f32_mul(np, r2, a["rr0"][column, k])
            r2 = _f32_mul(np, r2, a["rf0"][column, k + 1])
            r1 = _f32_mul(np, _f32_mul(np, dts, rdz32), a["mh"][column, k])
            r1 = _f32_mul(np, r1, a["rr0"][column, k])
            r1 = _f32_mul(np, r1, a["rf0"][column, k])
            tem1 = _f32_mul(
                np, _f32_mul(np, _f32_mul(np, dts, rdz32), cp32), a["mf"][column, k]
            )
            a["qk"][column, k] = _f32_mul(np, tem, a["ppterm"][column, k])
            pk2_left = _f32_mul(np, -a["c2"][column, k + 1], a["piadv"][column, k + 1])
            pk2_right = _f32_mul(
                np,
                _f32_sub(np, np.float32(1.0), a["c1"][column, k + 1]),
                a["piadv"][column, k],
            )
            a["pk2"][column, k] = _f32_mul(np, r2, _f32_add(np, pk2_left, pk2_right))
            pk1_left = _f32_mul(np, a["c1"][column, k], a["piadv"][column, k - 1])
            pk1_right = _f32_mul(
                np,
                _f32_sub(np, a["c2"][column, k], np.float32(1.0)),
                a["piadv"][column, k],
            )
            a["pk1"][column, k] = _f32_mul(np, r1, _f32_add(np, pk1_left, pk1_right))
            theta_face = _f32_add(
                np,
                _f32_mul(np, a["c2"][column, k], a["thv"][column, k]),
                _f32_mul(np, a["c1"][column, k], a["thv"][column, k - 1]),
            )
            a["mm"][column, k] = _f32_mul(np, tem1, theta_face)
        a["pk1"][column, 1] = np.float32(0.0)
        a["pk2"][column, nk] = np.float32(0.0)

        for k in range(1, nk + 1):
            a["ppd"][column, k] = a["ppx"][column, k]
            a["rrw"][column, k] = _f32_mul(np, a["wa"][column, k], beta)
            if nrk != 1:
                a["w3d"][column, k] = a["wa"][column, k]
                a["pp3d"][column, k] = a["ppi"][column, k]

        for small_step in range(1, nloop + 1):
            for k in range(1, nk + 1):
                source = _f32_sub(
                    np,
                    a["ppten"][column, k],
                    _f32_mul(np, a["ppterm"][column, k], np.float32(0.0)),
                )
                vertical = _f32_add(
                    np,
                    _f32_mul(
                        np,
                        a["w3d"][column, k],
                        _f32_add(np, a["pk1"][column, k], a["qk"][column, k]),
                    ),
                    _f32_mul(
                        np,
                        a["w3d"][column, k + 1],
                        _f32_sub(np, a["pk2"][column, k], a["qk"][column, k]),
                    ),
                )
                a["fpk"][column, k] = _f32_add(
                    np, _f32_mul(np, dts, source), _f32_mul(np, beta, vertical)
                )

            for k in range(2, nk + 1):
                aa = np.float32(0.0)
                cc = np.float32(0.0)
                if k >= 3:
                    aa = _f32_mul(
                        np,
                        -asq,
                        _f32_mul(
                            np,
                            a["mm"][column, k],
                            _f32_add(
                                np, a["pk1"][column, k - 1], a["qk"][column, k - 1]
                            ),
                        ),
                    )
                if k < nk:
                    cc = _f32_mul(
                        np,
                        -asq,
                        _f32_mul(
                            np,
                            -a["mm"][column, k],
                            _f32_sub(np, a["pk2"][column, k], a["qk"][column, k]),
                        ),
                    )
                diagonal_sum = _f32_add(
                    np,
                    -_f32_add(np, a["pk1"][column, k], a["qk"][column, k]),
                    _f32_sub(np, a["pk2"][column, k - 1], a["qk"][column, k - 1]),
                )
                bb = _f32_sub(
                    np,
                    np.float32(1.0),
                    _f32_mul(np, asq, _f32_mul(np, a["mm"][column, k], diagonal_sum)),
                )
                pressure_difference = _f32_sub(
                    np, a["pp3d"][column, k], a["pp3d"][column, k - 1]
                )
                fwk = _f32_add(
                    np,
                    _f32_mul(np, dts, a["wten"][column, k]),
                    _f32_mul(
                        np, beta, _f32_mul(np, -a["mm"][column, k], pressure_difference)
                    ),
                )
                updated_difference = _f32_sub(
                    np,
                    _f32_add(np, a["pp3d"][column, k], a["fpk"][column, k]),
                    _f32_add(np, a["pp3d"][column, k - 1], a["fpk"][column, k - 1]),
                )
                dd = _f32_add(np, a["w3d"][column, k], fwk)
                dd = _f32_add(
                    np,
                    dd,
                    _f32_mul(
                        np,
                        alph32,
                        _f32_mul(np, -a["mm"][column, k], updated_difference),
                    ),
                )
                pivot = bb
                if k >= 3:
                    pivot = _f32_add(
                        np, _f32_mul(np, aa, a["elim_c"][column, k - 1]), bb
                    )
                inverse = _f32_div(np, np.float32(1.0), pivot)
                a["tri_aa"][column, k] = aa
                a["tri_bb"][column, k] = bb
                a["tri_cc"][column, k] = cc
                a["tri_dd"][column, k] = dd
                a["tri_pivot"][column, k] = pivot
                if k < nk:
                    a["elim_c"][column, k] = _f32_mul(np, -cc, inverse)
                    numerator = dd
                    if k >= 3:
                        numerator = _f32_sub(
                            np, dd, _f32_mul(np, aa, a["elim_d"][column, k - 1])
                        )
                    a["elim_d"][column, k] = _f32_mul(np, numerator, inverse)
                else:
                    numerator = _f32_sub(
                        np, dd, _f32_mul(np, aa, a["elim_d"][column, k - 1])
                    )
                    a["w3d"][column, k] = _f32_mul(np, numerator, inverse)

            for k in range(nk - 1, 1, -1):
                a["w3d"][column, k] = _f32_add(
                    np,
                    _f32_mul(np, a["elim_c"][column, k], a["w3d"][column, k + 1]),
                    a["elim_d"][column, k],
                )

            for k in range(2, nk + 1):
                lhs = _f32_mul(np, a["tri_aa"][column, k], a["w3d"][column, k - 1])
                lhs = _f32_add(
                    np, lhs, _f32_mul(np, a["tri_bb"][column, k], a["w3d"][column, k])
                )
                lhs = _f32_add(
                    np,
                    lhs,
                    _f32_mul(np, a["tri_cc"][column, k], a["w3d"][column, k + 1]),
                )
                a["tri_residual"][column, k] = _f32_sub(np, lhs, a["tri_dd"][column, k])

            for k in range(1, nk + 1):
                old_pressure = a["pp3d"][column, k]
                vertical = _f32_add(
                    np,
                    _f32_mul(
                        np,
                        a["w3d"][column, k],
                        _f32_add(np, a["pk1"][column, k], a["qk"][column, k]),
                    ),
                    _f32_mul(
                        np,
                        a["w3d"][column, k + 1],
                        _f32_sub(np, a["pk2"][column, k], a["qk"][column, k]),
                    ),
                )
                updated = _f32_add(np, old_pressure, a["fpk"][column, k])
                updated = _f32_add(np, updated, _f32_mul(np, alph32, vertical))
                if bool(np.abs(updated) < smeps32):
                    updated = np.float32(0.0)
                a["pp3d"][column, k] = updated
                a["elim_c"][column, k] = _f32_mul(
                    np, kdiv32, _f32_sub(np, updated, old_pressure)
                )
                a["ppd"][column, k] = _f32_add(np, updated, a["elim_c"][column, k])

            for k in range(1, nk + 1):
                if small_step < nloop:
                    a["rrw"][column, k] = _f32_add(
                        np, a["rrw"][column, k], a["w3d"][column, k]
                    )
                else:
                    tavg = _f32_div(np, np.float32(1.0), np.float32(nloop))
                    a["rrw"][column, k] = _f32_mul(
                        np,
                        _f32_add(
                            np,
                            a["rrw"][column, k],
                            _f32_mul(np, a["w3d"][column, k], alph32),
                        ),
                        tavg,
                    )

        if nrk == 3:
            a["ppx"][column, 1 : nk + 1] = a["elim_c"][column, 1 : nk + 1]

    snapshot = a[f"ppx_stage{nrk}"]
    np.copyto(snapshot, a["ppx"])
    return CpuSound3StageDiagnostics(
        nrk=nrk,
        nloop=nloop,
        dts_f32=float(dts),
        dttmp_f32=rk_stage_dt_f32(dt, nrk),
        ppx_written=nrk == 3,
    )


@dataclass(frozen=True, slots=True)
class CpuSound3TrajectoryResult:
    cadence: tuple[CpuSound3StageDiagnostics, ...]


def cpu_sound3_three_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ncol: int,
    nk: int,
    dt: float,
    nsound: int,
    rdz: float,
    cp: float = CM1_SOUND3_DEFAULT_CP_F32,
    alph: float = CM1_SOUND3_DEFAULT_ALPH_F32,
    kdiv: float = CM1_SOUND3_DEFAULT_KDIV_F32,
) -> CpuSound3TrajectoryResult:
    cadence = tuple(
        cpu_sound3_stage_inplace(
            arrays,
            ncol=ncol,
            nk=nk,
            dt=dt,
            nsound=nsound,
            nrk=nrk,
            rdz=rdz,
            cp=cp,
            alph=alph,
            kdiv=kdiv,
        )
        for nrk in (1, 2, 3)
    )
    return CpuSound3TrajectoryResult(cadence=cadence)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        major, minor = text.split(".", 1)
        text = major + minor
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(
    arch: str, nvrtc_major: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
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


@dataclass(frozen=True, slots=True)
class CompiledSound3ColumnKernels:
    module_owner: Any
    kernels: tuple[Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_sound3_column() -> CompiledSound3ColumnKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged sound3-column CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if compiler_arch != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx_program = compiler._NVRTCProgram(
        CUDA_SOUND3_COLUMN_SOURCE, "cm1_sound3_column_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_SOUND3_COLUMN_SOURCE, "cm1_sound3_column_v1_cubin.cu", method="cubin"
    )
    cubin, _ = cubin_program.compile(cubin_options)
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")
    elif not isinstance(ptx, bytes):
        ptx = bytes(ptx)
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    elif not isinstance(cubin, bytes):
        cubin = bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_SOUND3_COLUMN_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more sound3 entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled sound3 image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_SOUND3_COLUMN_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc_parts[0])}.{int(nvrtc_parts[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledSound3ColumnKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ncol: int, nk: int
) -> dict[str, Any]:
    shape = column_shape(ncol, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing sound3 arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("sound3 column arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaSound3ColumnLaunch:
    ready_event: Any
    compiled: CompiledSound3ColumnKernels
    retained_events: tuple[Any, ...]
    receipt: dict[str, Any]


def launch_cuda_sound3_three_stage(
    arrays: Mapping[str, Any],
    *,
    ncol: int,
    nk: int,
    dt: float,
    nsound: int,
    rdz: float,
    cp: float = CM1_SOUND3_DEFAULT_CP_F32,
    alph: float = CM1_SOUND3_DEFAULT_ALPH_F32,
    kdiv: float = CM1_SOUND3_DEFAULT_KDIV_F32,
    stream: Any | None = None,
    compiled: CompiledSound3ColumnKernels | None = None,
) -> CudaSound3ColumnLaunch:
    """Enqueue all three native sound stages without host synchronization."""

    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ncol, nk)
    if compiled is None:
        compiled = compile_cuda_sound3_column()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    stride = nk + 2
    grid_columns = (ncol + CUDA_SOUND3_COLUMN_THREADS - 1) // CUDA_SOUND3_COLUMN_THREADS
    words = ncol * stride
    grid_words = (words + CUDA_SOUND3_COLUMN_THREADS - 1) // CUDA_SOUND3_COLUMN_THREADS
    u32 = cupy.uint32
    u64 = cupy.uint64
    f32 = cupy.float32
    alph32 = f32(alph)
    beta32 = f32(f32(1.0) - alph32)
    asq32 = f32(alph32 * alph32)
    cadence = []
    retained_events = []
    for nrk in (1, 2, 3):
        nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
        compiled.kernels[0](
            (grid_columns, 1, 1),
            (CUDA_SOUND3_COLUMN_THREADS, 1, 1),
            tuple(
                a[name]
                for name in INPUT_ARRAY_NAMES
                + STATE_ARRAY_NAMES
                + SCRATCH_ARRAY_NAMES[:13]
            )
            + (
                u32(ncol),
                u32(nk),
                u32(stride),
                u32(nrk),
                u32(nloop),
                f32(dts),
                f32(rdz),
                f32(cp),
                alph32,
                beta32,
                asq32,
                f32(kdiv),
                f32(CM1_SOUND3_SMEPS_F32),
            ),
            shared_mem=0,
            stream=stream,
        )
        compiled.kernels[1](
            (grid_words, 1, 1),
            (CUDA_SOUND3_COLUMN_THREADS, 1, 1),
            (a["ppx"], a[f"ppx_stage{nrk}"], u64(words)),
            shared_mem=0,
            stream=stream,
        )
        stage_event = cupy.cuda.Event(disable_timing=True)
        stage_event.record(stream)
        retained_events.append(stage_event)
        cadence.append(
            {
                "nrk": nrk,
                "dttmp_f32": rk_stage_dt_f32(dt, nrk),
                "nloop": nloop,
                "dts_f32": float(f32(dts)),
                "executed_small_steps": nloop,
                "ppx_written": nrk == 3,
            }
        )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_SOUND3_COLUMN_SCHEMA,
        "claim_limit": CUDA_SOUND3_COLUMN_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"independent_columns": ncol, "nk": nk},
        "cadence": cadence,
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
        "kernel": {
            "abi": CUDA_SOUND3_COLUMN_ABI,
            "names": list(CUDA_SOUND3_COLUMN_KERNEL_NAMES),
            "source_sha256": CUDA_SOUND3_COLUMN_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": CUDA_SOUND3_COLUMN_THREADS,
            "grid_columns_x": grid_columns,
            "grid_words_x": grid_words,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaSound3ColumnLaunch(
        ready_event=event,
        compiled=compiled,
        retained_events=tuple(retained_events),
        receipt=receipt,
    )


__all__ = [
    "ALL_ARRAY_NAMES",
    "CM1_SOUND3_DEFAULT_ALPH_F32",
    "CM1_SOUND3_DEFAULT_CP_F32",
    "CM1_SOUND3_DEFAULT_KDIV_F32",
    "CUDA_SOUND3_COLUMN_ABI",
    "CUDA_SOUND3_COLUMN_CLAIM_LIMIT",
    "CUDA_SOUND3_COLUMN_KERNEL_NAMES",
    "CUDA_SOUND3_COLUMN_SCHEMA",
    "CUDA_SOUND3_COLUMN_SOURCE",
    "CUDA_SOUND3_COLUMN_SOURCE_SHA256",
    "CompiledSound3ColumnKernels",
    "CpuSound3StageDiagnostics",
    "CpuSound3TrajectoryResult",
    "CudaSound3ColumnLaunch",
    "INPUT_ARRAY_NAMES",
    "SCRATCH_ARRAY_NAMES",
    "SOURCE_ANCHORS",
    "STATE_ARRAY_NAMES",
    "column_shape",
    "compile_cuda_sound3_column",
    "cpu_sound3_stage_inplace",
    "cpu_sound3_three_stage_inplace",
    "launch_cuda_sound3_three_stage",
    "rk_stage_dt_f32",
    "sound3_stage_schedule_f32",
]
