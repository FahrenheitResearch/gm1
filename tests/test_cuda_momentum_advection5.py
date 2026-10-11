from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from authority_trees import requires_native_reference
from cm1core.cuda_momentum_advection5 import (
    CM1_ADV_F_SHA256,
    CM1_ADV_ROUTINES_F_SHA256,
    CM1_SOLVE2_F_SHA256,
    CUDA_MOMENTUM_ADVECTION5_CLAIM_LIMIT,
    CUDA_MOMENTUM_ADVECTION5_SOURCE_BYTES,
    CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256,
    MomentumAdvection5Selection,
    cpu_terrain_open_momentum_advection5_inplace,
    launch_cuda_terrain_open_momentum_advection5,
    validate_momentum_advection5_selection,
)
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes


ROOT = Path(__file__).resolve().parents[1]
NI = 8
NJ = 8
NK = 8
NG = 3
PARAMETERS = {
    "ni": NI,
    "nj": NJ,
    "nk": NK,
    "nrk": 2,
    "dx": 100.0,
    "dy": 100.0,
    "dz": 200.0,
}
MUTATED = ("uten", "vten", "wten", "rru", "rrv", "rrw")


def _fixture(*, constant: bool = False) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    shapes = terrain_open_sound_shapes(NI, NJ, NK)
    arrays = {
        name: np.zeros(shapes[name], dtype=np.float32)
        for name in (
            "u3d",
            "uten",
            "rru",
            "v3d",
            "vten",
            "rrv",
            "w3d",
            "wten",
            "rrw",
            "uh",
            "vh",
        )
    }
    arrays["uh"].fill(np.float32(1.0))
    arrays["vh"].fill(np.float32(1.0))
    arrays["u3d"].fill(np.float32(4.0))
    arrays["v3d"].fill(np.float32(-2.0))
    if not constant:
        for number, name in enumerate(("u3d", "v3d", "w3d"), start=1):
            z, y, x = np.indices(arrays[name].shape, dtype=np.float32)
            arrays[name][:] = (
                np.float32(0.25 * number)
                + np.float32(0.013) * x
                - np.float32(0.009) * y
                + np.float32(0.017) * z
                + np.float32(0.0007) * x * y
            )
        for number, name in enumerate(("uten", "vten", "wten"), start=1):
            arrays[name].fill(np.float32(0.03125 * number))

    metrics = {
        name: np.ones(shape, dtype=np.float32)
        for name, shape in terrain_static_shapes(ni=NI, nj=NJ, nk=NK).items()
    }
    metrics["dzdx"].fill(np.float32(0.0))
    metrics["dzdy"].fill(np.float32(0.0))
    for k in range(NK):
        density = np.float32(1.12 - 0.018 * k)
        metrics["rho0_3d"][k].fill(density)
        metrics["rr0"][k].fill(np.float32(1.0) / density)
    for k in range(NK + 1):
        face_density = np.float32(1.13 - 0.017 * k)
        metrics["rf0"][k].fill(face_density)
        metrics["rrf0"][k].fill(np.float32(1.0) / face_density)
        metrics["c2"][k].fill(np.float32(0.5))
        metrics["c1"][k].fill(np.float32(0.5))
    if not constant:
        y, x = np.indices((NJ, NI), dtype=np.float32)
        metrics["dzdx"][:] = np.float32(0.0012) * (x - np.float32(3.5))
        metrics["dzdy"][:] = np.float32(-0.0009) * (y - np.float32(3.5))
        metrics["gz"][:] = np.float32(1.01)
        metrics["gzu"][:] = np.float32(1.01)
        metrics["gzv"][:] = np.float32(1.01)
        metrics["rgzu"][:] = np.float32(1.0) / np.float32(1.01)
        metrics["rgzv"][:] = np.float32(1.0) / np.float32(1.01)
    return arrays, metrics, np.zeros((NK, NJ, NI), dtype=np.float32)


def _copy(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy(order="C") for name, value in values.items()}


def test_import_is_cuda_cold_and_source_is_pinned() -> None:
    command = (
        "import sys; import cm1core.cuda_momentum_advection5 as m; "
        "assert m.CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256; "
        "assert not any(x == 'cupy' or x.startswith('cupy.') for x in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        hashlib.sha256(CUDA_MOMENTUM_ADVECTION5_SOURCE_BYTES).hexdigest()
        == CUDA_MOMENTUM_ADVECTION5_SOURCE_SHA256
    )
    assert "coupled forecast" in CUDA_MOMENTUM_ADVECTION5_CLAIM_LIMIT


@requires_native_reference
def test_authority_sources_are_exact_cm1_r21_1_bytes() -> None:
    expected = {
        "adv.F": CM1_ADV_F_SHA256,
        "adv_routines.F": CM1_ADV_ROUTINES_F_SHA256,
        "solve2.F": CM1_SOLVE2_F_SHA256,
    }
    authority = ROOT / "work/native-reference/cm1r21.1/src"
    for name, sha256 in expected.items():
        assert hashlib.sha256((authority / name).read_bytes()).hexdigest() == sha256


def test_selection_refuses_every_unimplemented_branch() -> None:
    cases = (
        MomentumAdvection5Selection(hadvordrv=6),
        MomentumAdvection5Selection(vadvordrv=3),
        MomentumAdvection5Selection(advwenov=2),
        MomentumAdvection5Selection(axisymmetric=1),
        MomentumAdvection5Selection(terrain=False),
        MomentumAdvection5Selection(idiff=1),
        MomentumAdvection5Selection(immersed_boundary=True),
        MomentumAdvection5Selection(uniform_horizontal_grid=False),
        MomentumAdvection5Selection(uniform_vertical_sigma=False),
        MomentumAdvection5Selection(lateral_boundaries=(1, 2, 2, 2)),
    )
    for selection in cases:
        with pytest.raises(ValueError, match="unsupported momentum-advection"):
            validate_momentum_advection5_selection(selection)


def test_constant_flat_flow_has_zero_divergence_and_zero_increment() -> None:
    arrays, metrics, divx = _fixture(constant=True)
    cpu_terrain_open_momentum_advection5_inplace(
        arrays, metrics, divx, **PARAMETERS
    )
    assert not np.count_nonzero(divx)
    assert not np.count_nonzero(arrays["uten"])
    assert not np.count_nonzero(arrays["vten"])
    assert not np.count_nonzero(arrays["wten"])
    assert np.all(np.isfinite(arrays["rru"]))
    assert np.all(np.isfinite(arrays["rrv"]))
    assert not np.count_nonzero(arrays["rrw"])


def test_cpu_slice_is_deterministic_finite_and_preserves_normal_open_faces() -> None:
    first_arrays, first_metrics, first_divx = _fixture()
    second_arrays, second_metrics, second_divx = _fixture()
    u_seed = first_arrays["uten"].copy()
    v_seed = first_arrays["vten"].copy()
    cpu_terrain_open_momentum_advection5_inplace(
        first_arrays, first_metrics, first_divx, **PARAMETERS
    )
    cpu_terrain_open_momentum_advection5_inplace(
        second_arrays, second_metrics, second_divx, **PARAMETERS
    )
    for name in MUTATED:
        assert np.array_equal(
            first_arrays[name].view(np.uint32),
            second_arrays[name].view(np.uint32),
        ), name
        assert np.all(np.isfinite(first_arrays[name])), name
    assert np.array_equal(first_divx.view(np.uint32), second_divx.view(np.uint32))
    assert np.count_nonzero(first_divx)
    assert np.count_nonzero(first_arrays["uten"].view(np.uint32) != u_seed.view(np.uint32))
    assert np.count_nonzero(first_arrays["vten"].view(np.uint32) != v_seed.view(np.uint32))
    assert np.array_equal(
        first_arrays["uten"][:, :, NG].view(np.uint32),
        u_seed[:, :, NG].view(np.uint32),
    )
    assert np.array_equal(
        first_arrays["uten"][:, :, NG + NI].view(np.uint32),
        u_seed[:, :, NG + NI].view(np.uint32),
    )
    assert np.array_equal(
        first_arrays["vten"][:, NG, :].view(np.uint32),
        v_seed[:, NG, :].view(np.uint32),
    )
    assert np.array_equal(
        first_arrays["vten"][:, NG + NJ, :].view(np.uint32),
        v_seed[:, NG + NJ, :].view(np.uint32),
    )


@pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="bounded GPU gate is opt-in",
)
def test_cuda_bounded_fixture_is_raw_bit_equal_to_cpu() -> None:
    cupy = pytest.importorskip("cupy")
    expected_arrays, expected_metrics, expected_divx = _fixture()
    actual_arrays, actual_metrics, _ = _fixture()
    cpu_terrain_open_momentum_advection5_inplace(
        expected_arrays, expected_metrics, expected_divx, **PARAMETERS
    )
    device_arrays = {
        name: cupy.asarray(value) for name, value in actual_arrays.items()
    }
    device_metrics = {
        name: cupy.asarray(value) for name, value in actual_metrics.items()
    }
    launch = launch_cuda_terrain_open_momentum_advection5(
        device_arrays, device_metrics, **PARAMETERS
    )
    launch.ready_event.synchronize()
    assert launch.compiled.target_arch == "86"
    observed_divx = cupy.asnumpy(launch.divx)
    assert np.array_equal(observed_divx.view(np.uint32), expected_divx.view(np.uint32))
    for name in MUTATED:
        observed = cupy.asnumpy(device_arrays[name])
        assert np.array_equal(
            observed.view(np.uint32), expected_arrays[name].view(np.uint32)
        ), name
