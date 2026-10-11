from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_terrain_deformation import (
    CUDA_TERRAIN_DEFORMATION_SOURCE,
    CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
    TerrainDeformationSelection,
    cpu_terrain_deformation_inplace,
    launch_cuda_terrain_deformation,
    terrain_deformation_shapes,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture(ni: int = 4, nj: int = 3, nk: int = 5) -> dict[str, np.ndarray]:
    shapes = terrain_deformation_shapes(ni, nj, nk)
    a = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    dx, dy, dz = np.float32(100.0), np.float32(100.0), np.float32(200.0)
    zt = np.float32(nk) * dz
    zs = np.empty((nj, ni), dtype=np.float32)
    for j in range(nj):
        for i in range(ni):
            zs[j, i] = np.float32(120.0 + 7.0 * i + 5.0 * j + 0.5 * i * j)
    a["gz"][:] = zt / (zt - zs)
    a["rgz"][:] = (zt - zs) / zt
    for j in range(nj):
        for i in range(ni + 1):
            left, right = max(i - 1, 0), min(i, ni - 1)
            terrain = np.float32(0.5) * np.float32(zs[j, left] + zs[j, right])
            a["gzu"][j, i] = zt / (zt - terrain)
            a["rgzu"][j, i] = (zt - terrain) / zt
    for j in range(nj + 1):
        bottom, top = max(j - 1, 0), min(j, nj - 1)
        for i in range(ni):
            terrain = np.float32(0.5) * np.float32(zs[bottom, i] + zs[top, i])
            a["gzv"][j, i] = zt / (zt - terrain)
            a["rgzv"][j, i] = (zt - terrain) / zt
    for kf in range(nk + 1):
        sigmaf = np.float32(kf) * dz
        for j in range(nj):
            for i in range(ni):
                a["gx"][kf, j, i] = (
                    (zt - sigmaf)
                    * a["gz"][j, i]
                    * (a["rgzu"][j, i + 1] - a["rgzu"][j, i])
                    / dx
                )
                a["gy"][kf, j, i] = (
                    (zt - sigmaf)
                    * a["gz"][j, i]
                    * (a["rgzv"][j + 1, i] - a["rgzv"][j, i])
                    / dy
                )
    for k in range(nk):
        for j in range(nj):
            for i in range(ni + 1):
                a["u3d"][k, j, i] = np.float32(
                    4.0 + 0.21 * i - 0.13 * j + 0.07 * k + 0.009 * i * k
                )
        for j in range(nj + 1):
            for i in range(ni):
                a["v3d"][k, j, i] = np.float32(
                    8.0 - 0.17 * i + 0.19 * j + 0.04 * k - 0.006 * j * k
                )
        for j in range(nj):
            for i in range(ni):
                a["rho"][k, j, i] = np.float32(1.12 - 0.035 * k + 0.002 * i - 0.001 * j)
    for kf in range(nk + 1):
        for j in range(nj):
            for i in range(ni):
                if kf in (0, nk):
                    a["w3d"][kf, j, i] = np.float32(0.0)
                else:
                    a["w3d"][kf, j, i] = np.float32(
                        0.02 * kf + 0.004 * i - 0.003 * j + 0.001 * i * j
                    )
                if kf == 0:
                    a["rf"][kf, j, i] = np.float32(
                        1.5 * a["rho"][0, j, i] - 0.5 * a["rho"][1, j, i]
                    )
                elif kf == nk:
                    a["rf"][kf, j, i] = np.float32(
                        1.5 * a["rho"][nk - 1, j, i] - 0.5 * a["rho"][nk - 2, j, i]
                    )
                else:
                    a["rf"][kf, j, i] = np.float32(
                        0.5 * np.float32(a["rho"][kf - 1, j, i] + a["rho"][kf, j, i])
                    )
                a["c1"][kf, j, i] = np.float32(0.5)
                a["c2"][kf, j, i] = np.float32(0.5)
    return a


def test_source_pin_shapes_and_cuda_cold_import() -> None:
    assert (
        hashlib.sha256(CUDA_TERRAIN_DEFORMATION_SOURCE.encode()).hexdigest()
        == CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256
    )
    assert terrain_deformation_shapes(4, 3, 5)["defv"] == (6, 3, 4)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; import cm1core.cuda_terrain_deformation; "
        "assert 'cupy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_cpu_selected_terrain_calcdef_is_deterministic_and_live() -> None:
    first = _fixture()
    second = _fixture()
    one = cpu_terrain_deformation_inplace(
        first, ni=4, nj=3, nk=5, dx=100.0, dy=100.0, dz=200.0
    )
    two = cpu_terrain_deformation_inplace(
        second, ni=4, nj=3, nk=5, dx=100.0, dy=100.0, dz=200.0
    )
    assert one == two
    assert one["live_words"] == 48
    assert one["boundary_face_zero_words"] == 24
    assert np.array_equal(first["defv"].view(np.uint32), second["defv"].view(np.uint32))
    assert np.array_equal(first["defh"].view(np.uint32), second["defh"].view(np.uint32))
    assert np.count_nonzero(first["defv"][1:5]) == 48
    assert np.count_nonzero(first["defh"][1:5]) == 48
    assert not np.count_nonzero(first["defv"][[0, 5]])
    assert not np.count_nonzero(first["defh"][[0, 5]])


def test_selection_and_layout_refusals() -> None:
    arrays = _fixture()
    with pytest.raises(Exception, match="CM1-CALCDEF-UNSUPPORTED"):
        cpu_terrain_deformation_inplace(
            arrays,
            ni=4,
            nj=3,
            nk=5,
            dx=100.0,
            dy=100.0,
            dz=200.0,
            selection=TerrainDeformationSelection(axisymmetric=1),
        )
    arrays["rho"] = arrays["rho"].astype(np.float64)
    with pytest.raises(TypeError, match="rho must have dtype float32"):
        cpu_terrain_deformation_inplace(
            arrays, ni=4, nj=3, nk=5, dx=100.0, dy=100.0, dz=200.0
        )


def test_cuda_is_raw_bit_exact_against_cpu_fixture() -> None:
    if os.environ.get("CM1CORE_RUN_CUDA_TERRAIN_DEFORMATION_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_TERRAIN_DEFORMATION_TEST=1 only while "
            "holding the GPU mutex"
        )
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as error:  # pragma: no cover - environment-specific
        pytest.skip(f"CUDA unavailable: {error}")
    expected = _fixture()
    cpu_terrain_deformation_inplace(
        expected, ni=4, nj=3, nk=5, dx=100.0, dy=100.0, dz=200.0
    )
    device = {name: cupy.asarray(value) for name, value in _fixture().items()}
    launch = launch_cuda_terrain_deformation(
        device, ni=4, nj=3, nk=5, dx=100.0, dy=100.0, dz=200.0
    )
    launch.ready_event.synchronize()
    for name in ("defv", "defh"):
        observed = cupy.asnumpy(device[name])
        assert np.array_equal(observed.view(np.uint32), expected[name].view(np.uint32))
    assert launch.receipt["domain"]["live_words"] == 48
    assert (
        launch.receipt["kernel"]["source_sha256"]
        == CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256
    )
