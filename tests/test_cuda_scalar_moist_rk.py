from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_scalar_moist_rk import (
    CM1_SCALAR_MOIST_SPECIES,
    CUDA_SCALAR_MOIST_RK_CLAIM_LIMIT,
    CUDA_SCALAR_MOIST_RK_KERNEL_NAMES,
    CUDA_SCALAR_MOIST_RK_SOURCE,
    CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
    cpu_compose_scalar_moist_rk_stage_inplace,
    cpu_pdef_direction_inplace,
    cpu_theta_transport_scalar_inplace,
    moisture_weps_f64,
    scalar_moist_rk_cadence,
    scalar_moist_rk_interior_bounds,
    scalar_moist_rk_stage_timestep_f32,
    theta_weps_f64,
)


ROOT = Path(__file__).resolve().parents[1]


def test_import_is_cuda_cold_and_kernel_source_is_pinned() -> None:
    command = (
        "import sys; import cm1core.cuda_scalar_moist_rk as m; "
        "assert m.CUDA_SCALAR_MOIST_RK_SOURCE_SHA256; "
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
        hashlib.sha256(CUDA_SCALAR_MOIST_RK_SOURCE.encode()).hexdigest()
        == CUDA_SCALAR_MOIST_RK_SOURCE_SHA256
    )
    assert all(
        name in CUDA_SCALAR_MOIST_RK_SOURCE
        for name in CUDA_SCALAR_MOIST_RK_KERNEL_NAMES
    )
    assert "forecast" in CUDA_SCALAR_MOIST_RK_CLAIM_LIMIT


def test_authority_sources_are_exact_cm1_r21_1_bytes() -> None:
    expected = {
        "solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
        "adv.F": "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f",
        "adv_routines.F": "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "pdef.F": "815cb7729c441b07c8d171596543b15ec630ab0c87eda5ab59a95d4eab2ae670",
        "constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    }
    authority = ROOT / "work/native-reference/cm1r21.1/src"
    for name, sha256 in expected.items():
        assert hashlib.sha256((authority / name).read_bytes()).hexdigest() == sha256


def test_exact_low_storage_schedule_and_final_stage_controls() -> None:
    dt = np.float32(0.5)
    expected = (
        np.float32(dt / np.float32(3.0)),
        np.float32(dt / np.float32(2.0)),
        dt,
    )
    actual = tuple(
        scalar_moist_rk_stage_timestep_f32(float(dt), nrk) for nrk in (1, 2, 3)
    )
    assert [int(value.view(np.uint32)) for value in actual] == [
        int(value.view(np.uint32)) for value in expected
    ]
    for nrk in (1, 2):
        cadence = scalar_moist_rk_cadence(nrk)
        assert cadence["uses_weno"] is False
        assert cadence["moisture_positive_definite"] is False
        assert cadence["refresh_scalar_halos_for_next_stage"] is True
    final = scalar_moist_rk_cadence(3)
    assert final["uses_weno"] is True
    assert final["moisture_positive_definite"] is True
    assert final["refresh_scalar_halos_for_next_stage"] is False
    with pytest.raises(ValueError, match="1..3"):
        scalar_moist_rk_stage_timestep_f32(0.5, 4)


def test_theta_transport_is_relative_to_300_k_and_preserves_statement_order() -> None:
    ni = nj = nk = 10
    th0 = np.linspace(298.0, 307.0, nk, dtype=np.float32)
    th3d = np.empty((nk, nj, ni), dtype=np.float32)
    z, y, x = np.indices(th3d.shape)
    th3d[:] = (
        np.float32(0.125)
        + np.float32(0.01) * x.astype(np.float32)
        - np.float32(0.02) * y.astype(np.float32)
        + np.float32(0.03) * z.astype(np.float32)
    )
    padded = np.zeros((nk + 2, nj + 6, ni + 6), dtype=np.float32)
    cpu_theta_transport_scalar_inplace(th0, th3d, padded, ni=ni, nj=nj, nk=nk)
    expected = np.float32(np.float32(th0[:, None, None] - np.float32(300.0)) + th3d)
    assert np.array_equal(
        padded[1 : nk + 1, 3 : nj + 3, 3 : ni + 3].view(np.uint32),
        expected.view(np.uint32),
    )
    assert not np.count_nonzero(padded[:, :3])


def test_cpu_stage_is_a_anchored_not_chained_and_clamps_only_after_update() -> None:
    shape = (2, 2, 3)
    qshape = (6,) + shape
    theta_a = np.full(shape, np.float32(1.0), dtype=np.float32)
    moisture_a = np.full(qshape, np.float32(0.01), dtype=np.float32)
    theta_held = np.full(shape, np.float32(0.25), dtype=np.float32)
    moisture_held = np.full(qshape, np.float32(-0.001), dtype=np.float32)
    divergence = np.full(shape, np.float32(0.05), dtype=np.float32)
    theta_adv = np.full(shape, np.float32(0.4), dtype=np.float32)
    moisture_adv = np.full(qshape, np.float32(0.003), dtype=np.float32)
    theta_current = np.full(shape, np.float32(99.0), dtype=np.float32)
    moisture_current = np.full(qshape, np.float32(99.0), dtype=np.float32)
    cpu_compose_scalar_moist_rk_stage_inplace(
        theta_a,
        moisture_a,
        theta_held,
        moisture_held,
        divergence,
        theta_adv,
        moisture_adv,
        theta_current,
        moisture_current,
        nrk=2,
        dt=0.5,
    )
    dttmp = np.float32(np.float32(0.5) / np.float32(2.0))
    theta_expected = np.float32(
        theta_a
        + np.float32(
            dttmp * np.float32(np.float32(theta_held - divergence) + theta_adv)
        )
    )
    moisture_expected = np.float32(
        moisture_a + np.float32(dttmp * np.float32(moisture_held + moisture_adv))
    )
    assert np.array_equal(theta_current.view(np.uint32), theta_expected.view(np.uint32))
    assert np.array_equal(
        moisture_current.view(np.uint32), moisture_expected.view(np.uint32)
    )
    assert not np.any(theta_current == np.float32(99.0))

    theta_a.fill(np.float32(0.0))
    moisture_a.fill(np.float32(0.0))
    theta_held.fill(np.float32(5.0e-31))
    moisture_held.fill(np.float32(-5.0e-31))
    divergence.fill(np.float32(0.0))
    theta_adv.fill(np.float32(0.0))
    moisture_adv.fill(np.float32(0.0))
    cpu_compose_scalar_moist_rk_stage_inplace(
        theta_a,
        moisture_a,
        theta_held,
        moisture_held,
        divergence,
        theta_adv,
        moisture_adv,
        theta_current,
        moisture_current,
        nrk=3,
        dt=0.5,
    )
    assert np.all(theta_current.view(np.uint32) == np.uint32(0))
    assert np.all(moisture_current.view(np.uint32) == np.uint32(0))


def test_species_weps_and_source_independent_bounds_are_fixed() -> None:
    assert CM1_SCALAR_MOIST_SPECIES == ("qv", "qc", "qr", "qi", "qs", "qg")
    expected_q = float(np.float32(np.float32(1.0e-2) * np.float32(1.0e-18)))
    assert all(moisture_weps_f64(index) == expected_q for index in range(6))
    assert theta_weps_f64() == float(np.float32(np.float32(10.0) * np.float32(1.0e-18)))
    assert scalar_moist_rk_interior_bounds(300, 300, positive_definite=False) == {
        "i": (4, 297),
        "j": (4, 297),
    }
    assert scalar_moist_rk_interior_bounds(300, 300, positive_definite=True) == {
        "i": (5, 296),
        "j": (5, 296),
    }


@pytest.mark.parametrize("axis,spacing", [("x", 100.0), ("y", 100.0), ("z", 200.0)])
def test_cpu_pdef_direction_repairs_activated_negative_candidate(
    axis: str, spacing: float
) -> None:
    ni = nj = nk = 12
    ng = 3
    shape = (nk, nj, ni)
    scalar_a = np.full(shape, np.float32(0.002), dtype=np.float32)
    padded = np.zeros((nk + 2, nj + 2 * ng, ni + 2 * ng), dtype=np.float32)
    padded[1 : nk + 1, ng : ng + nj, ng : ng + ni] = scalar_a
    index = np.indices(shape)[{"z": 0, "y": 1, "x": 2}[axis]]
    directional = np.where(
        index % 3 == 1, np.float32(-0.012), np.float32(0.008)
    ).astype(np.float32)
    flux_shapes = {
        "x": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "y": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "z": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    candidate = np.zeros(shape, dtype=np.float32)
    cpu_pdef_direction_inplace(
        scalar_a,
        padded,
        np.ones(shape, dtype=np.float32),
        np.ones((nj, ni), dtype=np.float32),
        np.ones((nj, ni), dtype=np.float32),
        np.zeros(flux_shapes[axis], dtype=np.float32),
        directional,
        candidate,
        axis=axis,
        dt=0.5,
        spacing=spacing,
    )
    safe = (slice(2, -2), slice(2, -2), slice(2, -2))
    assert np.count_nonzero(candidate[safe] < np.float32(0.0)) == 128
    corrected = np.float32(
        scalar_a[safe] + np.float32(np.float32(0.5) * directional[safe])
    )
    assert np.count_nonzero(corrected < np.float32(0.0)) == 0
