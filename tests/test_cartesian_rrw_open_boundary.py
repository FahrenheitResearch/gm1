"""The pre-acoustic ``bcw(rrw,0)`` fill, checked against the CM1 authority.

``src/solve2.F:574-582`` fills the lateral vertical-mass-flux boundaries after
the vertical mass flux is built and before divergence and advection consume it.
``src/bc.F:942-1143`` is the fill itself.  Two properties of the ``flag=0``
call are easy to get wrong and are what these tests pin:

* every lateral loop is ``do k=2,nk``, so the ``k=1`` and ``k=nk+1`` planes are
  not lateral-filled, and
* ``flag=0`` skips the ``w(i,j,1)=w(i,j,nk+1)=0`` rigid-face block entirely.

The host reference below is transcribed from ``bcw`` for the open lateral
configuration (``wbc=ebc=sbc=nbc=2``) that this runtime uses.
"""

from __future__ import annotations

import numpy as np
import pytest


cupy = pytest.importorskip("cupy")

from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes  # noqa: E402
from cm1core.cuda_sound3_open_boundary import (  # noqa: E402
    CUDA_SOUND3_OPEN_THREADS,
    compile_cuda_sound3_open_boundary,
)


NI, NJ, NK, NG = 9, 7, 11, 3


def _device_available() -> bool:
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - exercised only without a GPU
        return False
    return cupy.cuda.runtime.getDeviceCount() > 0


pytestmark = pytest.mark.skipif(
    not _device_available(), reason="the open-boundary kernels require a CUDA device"
)


def _seeded_rrw() -> np.ndarray:
    shape = terrain_open_sound_shapes(NI, NJ, NK)["rrw"]
    rng = np.random.default_rng(20260913)
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(3.0)


def _host_bcw_flag0(array: np.ndarray) -> np.ndarray:
    """``bcw(w,0)`` for open lateral boundaries, west/east then south/north.

    Storage index ``s`` relates to the CM1 index ``c`` by ``s = c + NG - 1``
    horizontally, so CM1 ``i=1`` is storage ``NG`` and CM1 ``i=0`` is ``NG-1``.
    Vertically CM1 ``k`` is storage ``k``.
    """

    out = array.copy()
    # CM1 j = 0 .. nj+1  ->  storage NG-1 .. NG+NJ
    j_lo, j_hi = NG - 1, NG + NJ + 1
    i_lo, i_hi = NG - 1, NG + NI + 1
    # do k=2,nk
    for k in range(2, NK + 1):
        # west: w(i,j,k) = w(1,j,k) for i = 1-ngxy .. 0
        for layer in range(NG):
            out[k, j_lo:j_hi, layer] = out[k, j_lo:j_hi, NG]
        # east: w(i,j,k) = w(ni,j,k) for i = ni+1 .. ni+ngxy
        for layer in range(NG):
            out[k, j_lo:j_hi, NI + NG + layer] = out[k, j_lo:j_hi, NI + NG - 1]
    for k in range(2, NK + 1):
        # south: w(i,j,k) = w(i,1,k) for j = 1-ngxy .. 0
        for layer in range(NG):
            out[k, layer, i_lo:i_hi] = out[k, NG, i_lo:i_hi]
        # north: w(i,j,k) = w(i,nj,k) for j = nj+1 .. nj+ngxy
        for layer in range(NG):
            out[k, NJ + NG + layer, i_lo:i_hi] = out[k, NJ + NG - 1, i_lo:i_hi]
    return out


def _run_hook(device: "cupy.ndarray", compiled) -> None:
    """Exactly the launch ``CartesianDynamics._rrw_open_boundary`` performs."""

    grid = ((int(device.size) + CUDA_SOUND3_OPEN_THREADS - 1) // CUDA_SOUND3_OPEN_THREADS, 1, 1)
    block = (CUDA_SOUND3_OPEN_THREADS, 1, 1)
    arguments = (
        device,
        cupy.uint32(NI),
        cupy.uint32(NJ),
        cupy.uint32(NK),
        cupy.uint32(NG),
        cupy.uint32(3),
    )
    compiled.kernels[5](grid, block, arguments)
    compiled.kernels[6](grid, block, arguments)


def test_flag_zero_fill_matches_the_bcw_authority_raw_word_exactly() -> None:
    compiled = compile_cuda_sound3_open_boundary()
    seed = _seeded_rrw()
    device = cupy.asarray(seed)
    _run_hook(device, compiled)
    cupy.cuda.Stream.null.synchronize()
    actual = cupy.asnumpy(device)
    expected = _host_bcw_flag0(seed)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_flag_zero_leaves_the_rigid_faces_and_k1_plane_untouched() -> None:
    """``do k=2,nk`` and ``flag=0`` together protect three whole planes."""

    compiled = compile_cuda_sound3_open_boundary()
    seed = _seeded_rrw()
    device = cupy.asarray(seed)
    _run_hook(device, compiled)
    cupy.cuda.Stream.null.synchronize()
    actual = cupy.asnumpy(device)
    for plane in (0, 1, NK + 1):
        assert np.array_equal(
            actual[plane].view(np.uint32), seed[plane].view(np.uint32)
        ), f"flag-zero bcw must not write the k={plane} storage plane"
    assert np.count_nonzero(actual[NK + 1]) > 0, "the top face must not be zeroed"


def test_missing_fill_is_detected_by_the_lateral_words() -> None:
    """Negative control: without the hook the ghost words differ."""

    compiled = compile_cuda_sound3_open_boundary()
    seed = _seeded_rrw()
    filled = cupy.asarray(seed)
    _run_hook(filled, compiled)
    cupy.cuda.Stream.null.synchronize()
    unfilled = cupy.asnumpy(cupy.asarray(seed))
    assert not np.array_equal(
        cupy.asnumpy(filled).view(np.uint32), unfilled.view(np.uint32)
    ), "the fill must change the lateral halo, otherwise the test proves nothing"


def test_west_then_east_then_south_then_north_order_reaches_the_corners() -> None:
    """Corner ghosts must see the freshly published lateral words.

    Running south/north before west/east leaves different corner values, so
    this pins the source order rather than merely the per-face result.
    """

    compiled = compile_cuda_sound3_open_boundary()
    seed = _seeded_rrw()

    forward = cupy.asarray(seed)
    _run_hook(forward, compiled)

    reversed_order = cupy.asarray(seed)
    grid = ((int(reversed_order.size) + CUDA_SOUND3_OPEN_THREADS - 1) // CUDA_SOUND3_OPEN_THREADS, 1, 1)
    block = (CUDA_SOUND3_OPEN_THREADS, 1, 1)
    arguments = (
        reversed_order,
        cupy.uint32(NI),
        cupy.uint32(NJ),
        cupy.uint32(NK),
        cupy.uint32(NG),
        cupy.uint32(3),
    )
    compiled.kernels[6](grid, block, arguments)
    compiled.kernels[5](grid, block, arguments)
    cupy.cuda.Stream.null.synchronize()

    assert not np.array_equal(
        cupy.asnumpy(forward).view(np.uint32),
        cupy.asnumpy(reversed_order).view(np.uint32),
    ), "west/east before south/north must be observable in the corner ghosts"
    assert np.array_equal(
        cupy.asnumpy(forward).view(np.uint32), _host_bcw_flag0(seed).view(np.uint32)
    )
