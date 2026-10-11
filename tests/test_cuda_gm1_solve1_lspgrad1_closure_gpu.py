"""GPU-gated: the lspgrad=1 closure kernel on a real CUDA device, raw word for raw word.

Without a device, the same packaged kernel TEXT is compiled by g++ through
``tools/oracle/cuda_kernel_host_emulator.py`` (a development tool) and checked
word for word against the source-exact CPU operator, and the real
``CartesianClosure`` is driven on the host with it
(``tests/test_cartesian_lspgrad.py``).  This file proves the device path: NVRTC
compilation with the closure's options, the launch geometry, per-tile
application on the smallest tiled plan the closure builds, and that every
published word equals the host emulation of the same text.

Run on a GPU host with the GPU mutex held::

    CM1_CUDA_TESTS=1 CM1_R21_1_PRISTINE=/path/to/cm1-21.1/pristine \\
        python3 -m pytest -q -p no:cacheprovider tests/test_cuda_gm1_solve1_lspgrad1_closure_gpu.py
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from cm1core import cuda_cartesian_closure as closure
from cm1core.cuda_gm1_solve1_lspgrad1_closure import (
    CUDA_GM1_LSPGRAD1_KERNEL_NAME,
    CUDA_GM1_LSPGRAD1_SOURCE,
    build_cartesian_lspgrad1_profiles,
    compile_cuda_gm1_lspgrad1,
    cpu_lspgrad1_apply,
    launch_cuda_gm1_lspgrad1,
)
from cm1core.grid_geometry import CartesianGridGeometry


pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires coordinated GPU mutex",
)
F32 = np.float32


def _profiles(nk: int):
    rng = np.random.default_rng(nk + 101)
    u0 = (6.0 * rng.standard_normal(nk)).astype(F32)
    v0 = (6.0 * rng.standard_normal(nk)).astype(F32)
    return build_cartesian_lspgrad1_profiles(u0, v0, fcor_s_inv=1.0e-4, u_move_m_s=8.5, v_move_m_s=6.0)


def test_one_device_launch_is_the_operator_word_for_word():
    cupy = pytest.importorskip("cupy")
    ni, nj, nk = 33, 21, 115
    profiles = _profiles(nk)
    rng = np.random.default_rng(5)
    host_u = rng.standard_normal((nk, nj, ni + 1)).astype(F32)
    host_v = rng.standard_normal((nk, nj + 1, ni)).astype(F32)
    expected_u, expected_v = host_u.copy(), host_v.copy()
    cpu_lspgrad1_apply(expected_u, expected_v, profiles.ul, profiles.vl)
    stream = cupy.cuda.Stream()
    compiled = compile_cuda_gm1_lspgrad1()
    with stream:
        uten1, vten1 = cupy.asarray(host_u), cupy.asarray(host_v)
        ul, vl = cupy.asarray(profiles.ul), cupy.asarray(profiles.vl)
        launch_cuda_gm1_lspgrad1(uten1, vten1, ul, vl, ni=ni, nj=nj, nk=nk, stream=stream, compiled=compiled)
    stream.synchronize()
    assert np.array_equal(cupy.asnumpy(uten1).view(np.uint32), expected_u.view(np.uint32))
    assert np.array_equal(cupy.asnumpy(vten1).view(np.uint32), expected_v.view(np.uint32))
    assert np.array_equal(cupy.asnumpy(ul).view(np.uint32), profiles.ul.view(np.uint32))


def test_the_tiled_device_application_equals_the_host_emulation_of_the_same_text():
    cupy = pytest.importorskip("cupy")
    if shutil.which("g++") is None:
        pytest.skip("the host emulation needs g++")
    from tools.oracle.cuda_kernel_host_emulator import HostKernelLibrary

    nx, ny, nk = 40, 30, 12
    grid = CartesianGridGeometry(nx, ny, nk, 100.0, 100.0, 100.0)
    plan = closure._tile_plan(grid, 24, 24)
    assert plan.ntiles > 1
    profiles = _profiles(nk)
    rng = np.random.default_rng(11)
    whole_u = rng.standard_normal(grid.u_shape).astype(F32)
    whole_v = rng.standard_normal(grid.v_shape).astype(F32)

    emulated = HostKernelLibrary.compile(CUDA_GM1_LSPGRAD1_SOURCE).kernel(CUDA_GM1_LSPGRAD1_KERNEL_NAME)
    host_u, host_v = whole_u.copy(), whole_v.copy()
    count = max(nk * ny * (nx + 1), nk * (ny + 1) * nx)
    emulated(((count + 255) // 256, 1, 1), (256, 1, 1),
             (profiles.ul, profiles.vl, host_u, host_v, np.uint32(nx), np.uint32(ny), np.uint32(nk)))

    stream = cupy.cuda.Stream()
    compiled = compile_cuda_gm1_lspgrad1()
    spec0 = plan.specs[0]
    with stream:
        source_u, source_v = cupy.asarray(whole_u), cupy.asarray(whole_v)
        published_u, published_v = cupy.zeros_like(source_u), cupy.zeros_like(source_v)
        ul, vl = cupy.asarray(profiles.ul), cupy.asarray(profiles.vl)
        tile_u = cupy.empty((nk, spec0.cny, spec0.cnx + 1), dtype=cupy.float32)
        tile_v = cupy.empty((nk, spec0.cny + 1, spec0.cnx), dtype=cupy.float32)
        for spec in plan.specs:
            closure._copy_tile(cupy, tile_u, source_u, spec)
            closure._copy_tile(cupy, tile_v, source_v, spec)
            launch_cuda_gm1_lspgrad1(tile_u, tile_v, ul, vl, ni=spec.cnx, nj=spec.cny, nk=nk,
                                     stream=stream, compiled=compiled)
            closure._scatter(cupy, published_u, tile_u, spec, "u")
            closure._scatter(cupy, published_v, tile_v, spec, "v")
    stream.synchronize()
    assert np.array_equal(cupy.asnumpy(published_u).view(np.uint32), host_u.view(np.uint32))
    assert np.array_equal(cupy.asnumpy(published_v).view(np.uint32), host_v.view(np.uint32))


def test_the_launcher_refuses_a_carrier_of_the_wrong_shape_before_any_launch():
    cupy = pytest.importorskip("cupy")
    nk = 6
    profiles = _profiles(nk)
    compiled = compile_cuda_gm1_lspgrad1()
    stream = cupy.cuda.Stream()
    with stream:
        uten1 = cupy.zeros((nk, 4, 4), dtype=cupy.float32)  # ni+1 would be 6
        vten1 = cupy.zeros((nk, 5, 5), dtype=cupy.float32)
        ul, vl = cupy.asarray(profiles.ul), cupy.asarray(profiles.vl)
        with pytest.raises(ValueError, match="uten1 must be"):
            launch_cuda_gm1_lspgrad1(uten1, vten1, ul, vl, ni=5, nj=4, nk=nk, stream=stream, compiled=compiled)
    assert not cupy.asnumpy(uten1).any()
