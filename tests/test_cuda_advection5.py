from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_advection5 import (
    Advection5Selection,
    CUDA_ADVECTION5_CLAIM_LIMIT,
    CUDA_ADVECTION5_SOURCE,
    CUDA_ADVECTION5_SOURCE_SHA256,
    advection5_full_domain_interior_bounds,
    cpu_scalar_advection5_inplace,
    cpu_terrain_mass_flux5_inplace,
    reconstruct_face5_f32,
    scalar_advection_uses_weno,
    validate_advection5_selection,
)
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
NI = 8
NJ = 8
NK = 8
NG = 3


def _fixture() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray
]:
    shapes = terrain_open_sound_shapes(NI, NJ, NK)
    arrays = {
        name: np.zeros(shapes[name], dtype=np.float32)
        for name in ("u3d", "v3d", "w3d", "rru", "rrv", "rrw")
    }
    for name in ("u3d", "v3d", "w3d"):
        z, y, x = np.indices(arrays[name].shape)
        arrays[name][:] = (
            np.float32(2.0)
            + np.float32(0.01) * x.astype(np.float32)
            - np.float32(0.02) * y.astype(np.float32)
            + np.float32(0.03) * z.astype(np.float32)
        )
    metrics = {
        name: np.ones(shape, dtype=np.float32)
        for name, shape in terrain_static_shapes(ni=NI, nj=NJ, nk=NK).items()
    }
    for k in range(NK):
        metrics["rho0_3d"][k] = np.float32(1.1 - 0.01 * k)
        metrics["rr0"][k] = np.float32(1.0) / metrics["rho0_3d"][k]
    for k in range(NK + 1):
        metrics["rf0"][k] = np.float32(1.05 - 0.009 * k)
        metrics["c1"][k] = np.float32(0.5)
        metrics["c2"][k] = np.float32(0.5)
    y, x = np.indices((NJ, NI))
    metrics["dzdx"][:] = np.float32(0.001) * (x - 3).astype(np.float32)
    metrics["dzdy"][:] = np.float32(-0.001) * (y - 3).astype(np.float32)
    metrics["gz"][:] = np.float32(1.01)
    scalar = np.zeros(shapes["scalar"], dtype=np.float32)
    z, y, x = np.indices(scalar.shape)
    scalar[:] = (
        np.where(x < NG + 4, np.float32(0.1), np.float32(1.0))
        + np.float32(0.001) * z.astype(np.float32)
        + np.float32(0.002) * y.astype(np.float32)
    )
    return arrays, metrics, scalar, np.zeros((NK, NJ, NI), dtype=np.float32)


def _fluxes(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x": np.zeros_like(arrays["rru"]),
        "y": np.zeros_like(arrays["rrv"]),
        "z": np.zeros_like(arrays["rrw"]),
    }


def _tendency(
    scalar: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    divx: np.ndarray,
    *,
    nrk: int,
    wflag: bool = True,
) -> np.ndarray:
    result = np.zeros((NK, NJ, NI), dtype=np.float32)
    cpu_scalar_advection5_inplace(
        scalar,
        arrays,
        metrics,
        divx,
        result,
        _fluxes(arrays),
        ni=NI,
        nj=NJ,
        nk=NK,
        nrk=nrk,
        dx=100.0,
        dy=100.0,
        dz=200.0,
        weps=1.0e-17,
        wflag=wflag,
    )
    return result


def test_import_is_cuda_cold_and_cuda_source_is_pinned() -> None:
    command = (
        "import sys; import cm1core.cuda_advection5 as m; "
        "assert m.CUDA_ADVECTION5_SOURCE_SHA256; "
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
        hashlib.sha256(CUDA_ADVECTION5_SOURCE.encode()).hexdigest()
        == CUDA_ADVECTION5_SOURCE_SHA256
    )
    assert "forecast" in CUDA_ADVECTION5_CLAIM_LIMIT


def test_authority_sources_are_exact_cm1_r21_1_bytes() -> None:
    expected = {
        "adv.F": "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f",
        "adv_routines.F": "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    }
    authority = ROOT / "work/native-reference/cm1r21.1/src"
    for name, sha256 in expected.items():
        assert hashlib.sha256((authority / name).read_bytes()).hexdigest() == sha256


def test_final_stage_weno_selector_and_strict_refusals() -> None:
    assert scalar_advection_uses_weno(1) is False
    assert scalar_advection_uses_weno(2) is False
    assert scalar_advection_uses_weno(3) is True
    assert scalar_advection_uses_weno(3, wflag=False) is False
    with pytest.raises(TypeError, match="nrk must be an integer"):
        scalar_advection_uses_weno(True)
    with pytest.raises(ValueError, match="nrkmax=3"):
        scalar_advection_uses_weno(3, nrkmax=4)
    for selection, option in (
        (Advection5Selection(scalar_horizontal_order=6), "dynamics.scalar_horizontal_order"),
        (Advection5Selection(velocity_weno=2), "dynamics.velocity_weno"),
        (Advection5Selection(positive_definite=True), "advection.positive_definite"),
    ):
        with pytest.raises(CM1Refusal) as error:
            validate_advection5_selection(selection)
        assert error.value.detail.code == "CM1-ADVECTION5-UNSUPPORTED"
        assert error.value.detail.gate == "dry-dynamics"
        assert error.value.detail.option == option


def test_cpu_terrain_flux_and_stage_controls_detect_weno_activation() -> None:
    arrays, metrics, scalar, divx = _fixture()
    cpu_terrain_mass_flux5_inplace(
        arrays,
        metrics,
        divx,
        ni=NI,
        nj=NJ,
        nk=NK,
        dx=100.0,
        dy=100.0,
        dz=200.0,
    )
    stage1 = _tendency(scalar, arrays, metrics, divx, nrk=1)
    stage2 = _tendency(scalar, arrays, metrics, divx, nrk=2)
    weno = _tendency(scalar, arrays, metrics, divx, nrk=3)
    wflag_off = _tendency(scalar, arrays, metrics, divx, nrk=3, wflag=False)
    assert np.array_equal(stage1.view(np.uint32), stage2.view(np.uint32))
    assert np.array_equal(stage1.view(np.uint32), wflag_off.view(np.uint32))
    assert np.count_nonzero(stage1)
    assert np.count_nonzero(weno.view(np.uint32) != stage2.view(np.uint32)) == 32
    mask = np.ones((NJ, NI), dtype=bool)
    mask[3 : NJ - 3, 3 : NI - 3] = False
    assert not np.count_nonzero(stage1[:, mask])
    assert advection5_full_domain_interior_bounds(300, 300) == {
        "i": (4, 297),
        "j": (4, 297),
    }


def test_face_reconstruction_is_bit_deterministic_and_refuses_wrong_layout() -> None:
    stencil = np.array([0.1, 0.2, 1.3, -0.4, 0.5], dtype=np.float32)
    first = reconstruct_face5_f32(
        stencil, 1.25, use_weno=True, weps=1.0e-17
    )
    second = reconstruct_face5_f32(
        stencil, 1.25, use_weno=True, weps=1.0e-17
    )
    assert int(first.view(np.uint32)) == int(second.view(np.uint32))
    with pytest.raises(TypeError, match="float32"):
        reconstruct_face5_f32(
            stencil.astype(np.float64), 1.25, use_weno=True, weps=1.0e-17
        )
