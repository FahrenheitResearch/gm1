from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.cuda_storm_diagnostics import (
    CUDA_STORM_DIAGNOSTICS_SCHEMA,
    CUDA_STORM_DIAGNOSTICS_SOURCE_BYTES,
    CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256,
    cpu_storm_diagnostics,
    launch_cuda_storm_diagnostics,
    storm_diagnostic_shapes,
)


def _fixture(
    ni: int = 5,
    nj: int = 5,
    nk: int = 4,
    *,
    terrain_shear: bool = False,
) -> dict[str, np.ndarray]:
    shapes = storm_diagnostic_shapes(ni, nj, nk)
    ng = CM1_OPEN_NGXY
    arrays = {
        name: np.zeros(shapes[name], dtype=np.float32)
        for name in (
            "u3d",
            "v3d",
            "w3d",
            "terrain_msl",
            "rgzu",
            "rgzv",
            "reflectivity_dbz",
        )
    }
    _, uy, _ = np.indices(shapes["u3d"], dtype=np.float32)
    uz, _, _ = np.indices(shapes["u3d"], dtype=np.float32)
    _, _, vx = np.indices(shapes["v3d"], dtype=np.float32)
    vz, _, _ = np.indices(shapes["v3d"], dtype=np.float32)
    arrays["u3d"][:] = np.float32(-0.5) * (uy - np.float32(ng))
    arrays["v3d"][:] = np.float32(0.5) * (vx - np.float32(ng))
    arrays["w3d"].fill(np.float32(7.5))
    arrays["rgzu"].fill(np.float32(1.0))
    arrays["rgzv"].fill(np.float32(1.0))
    z, y, x = np.indices((nk, nj, ni), dtype=np.float32)
    arrays["reflectivity_dbz"][:] = (
        np.float32(-12.0)
        + np.float32(0.5) * x
        + np.float32(0.25) * y
        + np.float32(0.125) * z
    )
    if terrain_shear:
        ty, tx = np.indices((nj, ni), dtype=np.float32)
        arrays["terrain_msl"][:] = np.float32(3.0) * tx + np.float32(2.0) * ty
        ruy, _ = np.indices(shapes["rgzu"], dtype=np.float32)
        _, rvx = np.indices(shapes["rgzv"], dtype=np.float32)
        arrays["rgzu"][:] = np.float32(1.0) + np.float32(0.002) * ruy
        arrays["rgzv"][:] = np.float32(1.0) + np.float32(0.001) * rvx
        arrays["u3d"] += np.float32(0.03125) * uz
        arrays["v3d"] -= np.float32(0.015625) * vz
    return arrays


def test_module_import_is_cold_and_source_is_pinned() -> None:
    assert (
        hashlib.sha256(CUDA_STORM_DIAGNOSTICS_SOURCE_BYTES).hexdigest()
        == CUDA_STORM_DIAGNOSTICS_SOURCE_SHA256
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cm1core.cuda_storm_diagnostics; "
            "assert 'cupy' not in sys.modules",
        ],
        check=False,
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cpu_solid_rotation_and_mass_centered_w_are_exact() -> None:
    arrays = _fixture()
    result = cpu_storm_diagnostics(
        arrays,
        ni=5,
        nj=5,
        nk=4,
        dx=128.0,
        dy=128.0,
        dz=64.0,
    )
    assert np.array_equal(result.reflectivity_dbz, arrays["reflectivity_dbz"])
    assert np.array_equal(result.updraft_ms, np.full((4, 5, 5), 7.5, np.float32))
    assert np.array_equal(
        result.vertical_vorticity_s,
        np.full((4, 5, 5), np.float32(1.0 / 128.0), np.float32),
    )
    assert result.exact_interior_words == 4 * 3 * 3
    assert result.display_closed_boundary_words == 4 * 5 * 5 - 4 * 3 * 3
    repeated = cpu_storm_diagnostics(
        arrays,
        ni=5,
        nj=5,
        nk=4,
        dx=128.0,
        dy=128.0,
        dz=64.0,
    )
    assert np.array_equal(
        repeated.vertical_vorticity_s.view(np.uint32),
        result.vertical_vorticity_s.view(np.uint32),
    )


def test_cpu_terrain_vertical_shear_is_a_real_negative_control() -> None:
    flat = cpu_storm_diagnostics(
        _fixture(), ni=5, nj=5, nk=4, dx=128.0, dy=128.0, dz=64.0
    )
    terrain = cpu_storm_diagnostics(
        _fixture(terrain_shear=True),
        ni=5,
        nj=5,
        nk=4,
        dx=128.0,
        dy=128.0,
        dz=64.0,
    )
    assert np.isfinite(terrain.vertical_vorticity_s).all()
    assert not np.array_equal(
        terrain.vertical_vorticity_s.view(np.uint32),
        flat.vertical_vorticity_s.view(np.uint32),
    )


def test_validation_refuses_missing_wrong_shape_and_bad_spacing() -> None:
    arrays = _fixture()
    missing = dict(arrays)
    del missing["reflectivity_dbz"]
    with pytest.raises(ValueError, match="missing storm diagnostic arrays"):
        cpu_storm_diagnostics(
            missing, ni=5, nj=5, nk=4, dx=128.0, dy=128.0, dz=64.0
        )
    wrong = dict(arrays)
    wrong["terrain_msl"] = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="terrain_msl"):
        cpu_storm_diagnostics(
            wrong, ni=5, nj=5, nk=4, dx=128.0, dy=128.0, dz=64.0
        )
    with pytest.raises(ValueError, match="dx must be finite and > 0"):
        cpu_storm_diagnostics(
            arrays, ni=5, nj=5, nk=4, dx=0.0, dy=128.0, dz=64.0
        )


def test_cuda_matches_cpu_raw_bits_only_with_mutex() -> None:
    if os.environ.get("CM1CORE_RUN_CUDA_STORM_DIAGNOSTICS_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_STORM_DIAGNOSTICS_TEST=1 only while holding "
            "the GPU mutex"
        )
    cupy = pytest.importorskip("cupy")
    arrays = _fixture(terrain_shear=True)
    expected = cpu_storm_diagnostics(
        arrays, ni=5, nj=5, nk=4, dx=128.0, dy=128.0, dz=64.0
    )
    launch = launch_cuda_storm_diagnostics(
        {name: cupy.asarray(value) for name, value in arrays.items()},
        ni=5,
        nj=5,
        nk=4,
        dx=128.0,
        dy=128.0,
        dz=64.0,
    )
    launch.ready_event.synchronize()
    assert launch.receipt["schema"] == CUDA_STORM_DIAGNOSTICS_SCHEMA
    for name, reference in (
        ("reflectivity_dbz", expected.reflectivity_dbz),
        ("updraft_ms", expected.updraft_ms),
        ("vertical_vorticity_s", expected.vertical_vorticity_s),
    ):
        candidate = cupy.asnumpy(launch.fields[name])
        assert np.array_equal(candidate.view(np.uint32), reference.view(np.uint32))
