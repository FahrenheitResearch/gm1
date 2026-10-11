"""GPU-gated: the composite reflectivity kernels on a real CUDA device.

Every test here launches a kernel, so every test here is gated behind the
GPU mutex flag the device tests use.  The arithmetic they check is
also checked without a device, by compiling the same packaged source text with
``g++`` and comparing raw words against the host transcription
(``tests/test_composite_reflectivity_kernel_oracle.py``).  That harness proves
the text; this one proves the device path: NVRTC compilation, the entry-point
and cubin gates, the launch configuration, and the arrays the launcher writes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cm1core.cuda_composite_reflectivity import (
    CM1_CREF_SEED_DBZ,
    CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES,
    compile_cuda_composite_reflectivity,
    cpu_composite_reflectivity,
    cpu_reflectivity_level,
    launch_cuda_composite_reflectivity,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires coordinated GPU mutex",
)


def _volume(nk: int, nj: int, ni: int) -> np.ndarray:
    rng = np.random.default_rng(20260914)
    volume = (rng.standard_normal((nk, nj, ni)).astype(np.float32) * np.float32(18.0))
    volume[:, 0, 0] = np.float32(-35.0)
    volume[:, 1, 1] = np.float32(47.5)
    volume[-1, -1, -1] = np.float32(72.25)
    return np.ascontiguousarray(volume)


def test_the_device_composite_is_raw_word_identical_to_the_transcription():
    cupy = pytest.importorskip("cupy")
    ni, nj, nk, level = 7, 5, 9, 4
    host = _volume(nk, nj, ni)
    dbz = cupy.asarray(host)
    cref = cupy.empty((nj, ni), dtype=cupy.float32)
    plane = cupy.empty((nj, ni), dtype=cupy.float32)
    launch = launch_cuda_composite_reflectivity(
        {"dbz": dbz, "cref": cref, "plane": plane},
        ni=ni, nj=nj, nk=nk, level=level,
    )
    launch.ready_event.synchronize()
    assert np.array_equal(
        cupy.asnumpy(cref).view(np.uint32),
        cpu_composite_reflectivity(host).view(np.uint32),
    )
    assert np.array_equal(
        cupy.asnumpy(plane).view(np.uint32),
        cpu_reflectivity_level(host, level).view(np.uint32),
    )
    assert launch.receipt["composite_seed_dbz"] == CM1_CREF_SEED_DBZ
    assert launch.receipt["level"] == level
    assert launch.receipt["same_stream_no_host_sync"] is True


def test_the_compiled_image_carries_both_entry_points():
    pytest.importorskip("cupy")
    compiled = compile_cuda_composite_reflectivity()
    assert len(compiled.kernels) == len(CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES)
    assert compiled.cubin[:7] == b"\x7fELF\x02\x01\x01"
    for name in CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES:
        assert name in compiled.ptx.decode("utf-8")


def test_a_level_outside_the_volume_refuses_before_any_launch():
    cupy = pytest.importorskip("cupy")
    ni, nj, nk = 4, 3, 5
    arrays = {
        "dbz": cupy.zeros((nk, nj, ni), dtype=cupy.float32),
        "cref": cupy.empty((nj, ni), dtype=cupy.float32),
        "plane": cupy.empty((nj, ni), dtype=cupy.float32),
    }
    with pytest.raises(ValueError, match="another level's words"):
        launch_cuda_composite_reflectivity(arrays, ni=ni, nj=nj, nk=nk, level=nk)
