"""Every boundary-family kernel, raw-word against its own CPU oracle.

A CM1-vs-GM1 differential on a storm case cannot settle these operators: the
open-boundary companions move the solution by far less than the two codes'
own rounding disagreement on the same case, so the trajectory comparison is
saturated before it can see the option.  What does settle them is the pattern
this repository already uses for the microphysics layout: run the packaged
CUDA text and the NumPy transcription of the same ``src/bc.F`` lines on the
same inputs, and compare the raw words.

These tests need a CUDA device and skip without one.
"""

from __future__ import annotations

import numpy as np
import pytest

from cm1core.cuda_boundary_families import (
    KIND_SCALAR,
    KIND_U,
    KIND_V,
    KIND_W,
    LateralBoundarySelection,
    compile_cuda_boundary_families,
    cpu_family_sn_inplace,
    cpu_family_we_inplace,
    cpu_hrdamp_tauf_inplace,
    cpu_irdamp2_operands,
    cpu_nudgeobc_sn_inplace,
    cpu_nudgeobc_we_inplace,
    cpu_pressure_sn_inplace,
    cpu_pressure_we_inplace,
    cpu_radbcew_inplace,
    cpu_radbcns_inplace,
    cpu_restrict_openbc_sn_inplace,
    cpu_restrict_openbc_we_inplace,
    family_array_shape,
)


cupy = pytest.importorskip("cupy")

NI, NJ, NK, NG = 11, 9, 7, 3
THREADS = (128, 1, 1)


@pytest.fixture(scope="module")
def compiled():
    try:
        cupy.cuda.Device().compute_capability
    except Exception as error:  # pragma: no cover - no device in this environment
        pytest.skip(f"no CUDA device: {error}")
    return compile_cuda_boundary_families()


def _noise(shape, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.standard_normal(shape) * scale).astype(np.float32))


def _grid(words):
    return ((words + THREADS[0] - 1) // THREADS[0], 1, 1)


def _same(host, device, label):
    got = cupy.asnumpy(device)
    assert got.tobytes() == host.tobytes(), f"{label}: CUDA and CPU words differ"


@pytest.mark.parametrize("kind", [KIND_SCALAR, KIND_U, KIND_V, KIND_W])
@pytest.mark.parametrize("wbc, ebc", [(1, 1), (2, 2), (3, 3), (4, 4), (1, 2), (2, 3)])
def test_family_we_is_raw_word_identical_to_the_bc_f_transcription(compiled, kind, wbc, ebc):
    shape = family_array_shape(kind, ni=NI, nj=NJ, nk=NK, ng=NG)
    host = _noise(shape, 1000 + kind * 10 + wbc)
    device = cupy.asarray(host)
    cpu_family_we_inplace(host, ni=NI, nj=NJ, nk=NK, ng=NG, kind=kind, wbc=wbc, ebc=ebc)
    compiled.kernels[0](
        _grid(int(np.prod(shape))), THREADS,
        (device, *map(cupy.uint32, (NI, NJ, NK, NG, kind, wbc, ebc))),
    )
    _same(host, device, f"family_we kind={kind} ({wbc},{ebc})")


@pytest.mark.parametrize("kind", [KIND_SCALAR, KIND_U, KIND_V, KIND_W])
@pytest.mark.parametrize("sbc, nbc", [(1, 1), (2, 2), (3, 3), (4, 4), (2, 1), (4, 2)])
def test_family_sn_is_raw_word_identical_to_the_bc_f_transcription(compiled, kind, sbc, nbc):
    shape = family_array_shape(kind, ni=NI, nj=NJ, nk=NK, ng=NG)
    host = _noise(shape, 2000 + kind * 10 + sbc)
    device = cupy.asarray(host)
    cpu_family_sn_inplace(host, ni=NI, nj=NJ, nk=NK, ng=NG, kind=kind, sbc=sbc, nbc=nbc)
    compiled.kernels[1](
        _grid(int(np.prod(shape))), THREADS,
        (device, *map(cupy.uint32, (NI, NJ, NK, NG, kind, sbc, nbc))),
    )
    _same(host, device, f"family_sn kind={kind} ({sbc},{nbc})")


@pytest.mark.parametrize("first, second", [(1, 1), (2, 2), (3, 4), (1, 2)])
def test_pressure_fills_are_raw_word_identical(compiled, first, second):
    shape = (NK + 2, NJ + 2, NI + 2)
    host = _noise(shape, 3000 + first * 7 + second)
    device = cupy.asarray(host)
    cpu_pressure_we_inplace(host, ni=NI, nj=NJ, nk=NK, wbc=first, ebc=second)
    cpu_pressure_sn_inplace(host, ni=NI, nj=NJ, nk=NK, sbc=first, nbc=second)
    args = tuple(map(cupy.uint32, (NI, NJ, NK)))
    compiled.kernels[2](_grid(NK * (NJ + 2)), THREADS, (device, *args, cupy.uint32(first), cupy.uint32(second)))
    compiled.kernels[3](_grid(NK * (NI + 2)), THREADS, (device, *args, cupy.uint32(first), cupy.uint32(second)))
    _same(host, device, f"bcp ({first},{second})")


def test_radbc_histories_are_raw_word_identical(compiled):
    ua = _noise(family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG), 4001, 40.0)
    va = _noise(family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG), 4002, 40.0)
    radbcw = np.zeros((NK + 2, NJ + 2 * NG), dtype=np.float32)
    radbce = np.zeros_like(radbcw)
    radbcs = np.zeros((NK + 2, NI + 2 * NG), dtype=np.float32)
    radbcn = np.zeros_like(radbcs)
    d = [cupy.asarray(v) for v in (ua, va, radbcw, radbce, radbcs, radbcn)]
    cstar = 30.0
    cpu_radbcew_inplace(ua, radbcw, radbce, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2, cstar=cstar)
    cpu_radbcns_inplace(va, radbcs, radbcn, ni=NI, nj=NJ, nk=NK, ng=NG, sbc=2, nbc=2, cstar=cstar)
    common = tuple(map(cupy.uint32, (NI, NJ, NK, NG)))
    compiled.kernels[4](
        _grid(NK * max(NI, NJ)), THREADS,
        (d[0], d[2], d[3], *common, cupy.uint32(2), cupy.uint32(2), cupy.float32(cstar)),
    )
    compiled.kernels[5](
        _grid(NK * max(NI, NJ)), THREADS,
        (d[1], d[4], d[5], *common, cupy.uint32(2), cupy.uint32(2), cupy.float32(cstar)),
    )
    _same(radbcw, d[2], "radbcw")
    _same(radbce, d[3], "radbce")
    _same(radbcs, d[4], "radbcs")
    _same(radbcn, d[5], "radbcn")


def _roflux_inputs(seed):
    rng = np.random.default_rng(seed)
    rho0 = np.ascontiguousarray((0.5 + rng.random((NK, NJ, NI))).astype(np.float32))
    rmh = np.ones((NK, NJ, NI), dtype=np.float32)
    ruh = np.ones((NI + 2 * NG,), dtype=np.float32)
    rvh = np.ones((NJ + 2 * NG,), dtype=np.float32)
    return rho0, rmh, ruh, rvh


def test_restrict_openbc_we_matches_the_double_precision_cpu_oracle(compiled):
    rho0, rmh, _, rvh = _roflux_inputs(5001)
    u3d = _noise(family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG), 5002, 6.0)
    device_u = cupy.asarray(u3d)
    d_rho0, d_rmh, d_rvh = (cupy.asarray(v) for v in (rho0, rmh, rvh))
    temout = cupy.zeros(NK, dtype=cupy.float64)
    temin = cupy.zeros(NK, dtype=cupy.float64)
    t3 = cupy.zeros(1, dtype=cupy.float64)
    expected = cpu_restrict_openbc_we_inplace(
        rho0, u3d, rvh, rmh, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2
    )
    common = tuple(map(cupy.uint32, (NI, NJ, NK, NG)))
    compiled.kernels[6](
        _grid(NK), THREADS,
        (d_rho0, device_u, d_rvh, d_rmh, temout, temin, *common, cupy.uint32(2), cupy.uint32(2)),
    )
    compiled.kernels[8]((1, 1, 1), (1, 1, 1), (temout, temin, t3, cupy.uint32(NK)))
    compiled.kernels[9](
        _grid(NK * NJ), THREADS,
        (d_rho0, device_u, t3, *common, cupy.uint32(2), cupy.uint32(2)),
    )
    assert float(cupy.asnumpy(t3)[0]) == expected
    _same(u3d, device_u, "restrict_openbc_we")


def test_restrict_openbc_sn_matches_the_double_precision_cpu_oracle(compiled):
    rho0, rmh, ruh, _ = _roflux_inputs(6001)
    v3d = _noise(family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG), 6002, 6.0)
    device_v = cupy.asarray(v3d)
    d_rho0, d_rmh, d_ruh = (cupy.asarray(v) for v in (rho0, rmh, ruh))
    temout = cupy.zeros(NK, dtype=cupy.float64)
    temin = cupy.zeros(NK, dtype=cupy.float64)
    t3 = cupy.zeros(1, dtype=cupy.float64)
    expected = cpu_restrict_openbc_sn_inplace(
        rho0, v3d, ruh, rmh, ni=NI, nj=NJ, nk=NK, ng=NG, sbc=2, nbc=2
    )
    common = tuple(map(cupy.uint32, (NI, NJ, NK, NG)))
    compiled.kernels[7](
        _grid(NK), THREADS,
        (d_rho0, device_v, d_ruh, d_rmh, temout, temin, *common, cupy.uint32(2), cupy.uint32(2)),
    )
    compiled.kernels[8]((1, 1, 1), (1, 1, 1), (temout, temin, t3, cupy.uint32(NK)))
    compiled.kernels[10](
        _grid(NK * NI), THREADS,
        (d_rho0, device_v, t3, *common, cupy.uint32(2), cupy.uint32(2)),
    )
    assert float(cupy.asnumpy(t3)[0]) == expected
    _same(v3d, device_v, "restrict_openbc_sn")


def test_nudgeobc_tendency_is_raw_word_identical(compiled):
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    vshape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    u3d, v3d = _noise(ushape, 7001, 8.0), _noise(vshape, 7002, 8.0)
    uten, vten = _noise(ushape, 7003, 0.1), _noise(vshape, 7004, 0.1)
    u0 = _noise((NK,), 7005, 5.0)
    v0 = _noise((NK,), 7006, 5.0)
    device = [cupy.asarray(v) for v in (u3d, v3d, uten, vten, u0, v0)]
    alphobc = 60.0
    cpu_nudgeobc_we_inplace(u3d, u0, uten, ni=NI, nj=NJ, nk=NK, ng=NG, wbc=2, ebc=2, alphobc=alphobc)
    cpu_nudgeobc_sn_inplace(v3d, v0, vten, ni=NI, nj=NJ, nk=NK, ng=NG, sbc=2, nbc=2, alphobc=alphobc)
    rtem = cupy.float32(np.divide(np.float32(1.0), np.float32(alphobc), dtype=np.float32))
    common = tuple(map(cupy.uint32, (NI, NJ, NK, NG)))
    compiled.kernels[11](
        _grid(NK * max(NI, NJ)), THREADS,
        (device[0], device[4], device[2], *common, cupy.uint32(2), cupy.uint32(2), rtem),
    )
    compiled.kernels[12](
        _grid(NK * max(NI, NJ)), THREADS,
        (device[1], device[5], device[3], *common, cupy.uint32(2), cupy.uint32(2), rtem),
    )
    _same(uten, device[2], "nudgeobc uten")
    _same(vten, device[3], "nudgeobc vten")


def test_hrdamp_taper_agrees_with_the_param_f_transcription(compiled):
    tauf = _noise((NK + 1, NJ + 2, NI + 2), 8001, 0.0) + np.float32(0.0)
    tauf = np.ascontiguousarray(np.abs(_noise((NK + 1, NJ + 2, NI + 2), 8002)) * np.float32(0.1))
    device = cupy.asarray(tauf)
    dx = dy = 1000.0
    xhd = 3000.0
    cpu_hrdamp_tauf_inplace(
        tauf, ni=NI, nj=NJ, nk=NK, i0=0, j0=0, nx=NI, ny=NJ, dx=dx, dy=dy, xhd=xhd
    )
    compiled.kernels[16](
        _grid((NK + 1) * (NJ + 2) * (NI + 2)), THREADS,
        (device, *map(cupy.uint32, (NI, NJ, NK, 0, 0, NI, NJ)),
         cupy.float32(dx), cupy.float32(dy), cupy.float32(xhd)),
    )
    got = cupy.asnumpy(device)
    # cosf on the device and the host libm cos CM1 links are both correctly
    # rounded to within an ulp but not to the same word, so this coefficient
    # is held to rounding rather than to the byte, and the receipt says so.
    np.testing.assert_allclose(got, tauf, rtol=2e-6, atol=2e-7)
    # The sponge must actually be a sponge: unity-ish at the wall, zero inside.
    assert float(got[0, NJ // 2 + 1, 1]) > 0.5
    assert float(got[0, NJ // 2 + 1, NI // 2 + 1]) < 0.2


def test_irdamp2_operands_are_raw_word_identical(compiled):
    uavg, vavg, thavg = (_noise((NK,), 9000 + n, 5.0) for n in range(3))
    th0 = _noise((NK, NJ, NI), 9100, 300.0)
    tha = _noise((NK, NJ, NI), 9200, 1.0)
    u0e, v0e, thae = cpu_irdamp2_operands(uavg, vavg, thavg, th0, tha, ni=NI, nj=NJ, nk=NK)
    d = [cupy.asarray(v) for v in (uavg, vavg, thavg, th0, tha)]
    out = [
        cupy.zeros((NK, NJ, NI + 1), dtype=cupy.float32),
        cupy.zeros((NK, NJ + 1, NI), dtype=cupy.float32),
        cupy.zeros((NK, NJ, NI), dtype=cupy.float32),
    ]
    compiled.kernels[17](
        _grid(NK * (NJ + 1) * (NI + 1)), THREADS,
        (*d, *out, *map(cupy.uint32, (NI, NJ, NK))),
    )
    _same(u0e, out[0], "irdamp2 u0")
    _same(v0e, out[1], "irdamp2 v0")
    _same(thae, out[2], "irdamp2 tha")


def test_non_open_faces_are_held_across_the_sealed_ssopen_operator(compiled):
    ushape = family_array_shape(KIND_U, ni=NI, nj=NJ, nk=NK, ng=NG)
    vshape = family_array_shape(KIND_V, ni=NI, nj=NJ, nk=NK, ng=NG)
    u3d, v3d = _noise(ushape, 10001, 5.0), _noise(vshape, 10002, 5.0)
    du, dv = cupy.asarray(u3d), cupy.asarray(v3d)
    face_u = cupy.zeros((2, NK, NJ), dtype=cupy.float32)
    face_v = cupy.zeros((2, NK, NI), dtype=cupy.float32)
    # west periodic, east open, south rigid, north open.
    sides = (1, 2, 3, 2)
    args = (*map(cupy.uint32, (NI, NJ, NK, NG)), *map(cupy.uint32, sides))
    compiled.kernels[14](_grid(NK * max(NI, NJ)), THREADS, (du, dv, face_u, face_v, *args))
    # The sealed operator would have written every face; simulate that.
    du[1:NK + 1, NG:NG + NJ, NG] = cupy.float32(-999.0)
    du[1:NK + 1, NG:NG + NJ, NI + NG] = cupy.float32(-999.0)
    dv[1:NK + 1, NG, NG:NG + NI] = cupy.float32(-999.0)
    dv[1:NK + 1, NJ + NG, NG:NG + NI] = cupy.float32(-999.0)
    compiled.kernels[15](_grid(NK * max(NI, NJ)), THREADS, (du, dv, face_u, face_v, *args))
    got_u, got_v = cupy.asnumpy(du), cupy.asnumpy(dv)
    # wbc=1 and sbc=3 are restored; ebc=2 and nbc=2 keep the operator's word.
    np.testing.assert_array_equal(got_u[1:NK + 1, NG:NG + NJ, NG], u3d[1:NK + 1, NG:NG + NJ, NG])
    assert np.all(got_u[1:NK + 1, NG:NG + NJ, NI + NG] == np.float32(-999.0))
    np.testing.assert_array_equal(got_v[1:NK + 1, NG, NG:NG + NI], v3d[1:NK + 1, NG, NG:NG + NI])
    assert np.all(got_v[1:NK + 1, NJ + NG, NG:NG + NI] == np.float32(-999.0))


def test_the_selection_the_runtime_binds_is_the_one_the_kernels_receive(compiled):
    # The bound wrapper appends the two side values to the sealed call
    # signature; a fill launched through it must equal a direct launch.
    selection = LateralBoundarySelection(west=1, east=3, south=2, north=4)
    from cm1core.cuda_boundary_families import bind_lateral_families
    from cm1core.cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary

    bound = bind_lateral_families(compile_cuda_sound3_open_boundary(), compiled, selection)
    shape = family_array_shape(KIND_SCALAR, ni=NI, nj=NJ, nk=NK, ng=NG)
    host = _noise(shape, 11001)
    through_binding = cupy.asarray(host)
    direct = cupy.asarray(host)
    words = _grid(int(np.prod(shape)))
    bound.kernels[5](words, THREADS, (through_binding, *map(cupy.uint32, (NI, NJ, NK, NG, KIND_SCALAR))))
    compiled.kernels[0](
        words, THREADS,
        (direct, *map(cupy.uint32, (NI, NJ, NK, NG, KIND_SCALAR, selection.west, selection.east))),
    )
    _same(cupy.asnumpy(direct), through_binding, "bound family_we")
