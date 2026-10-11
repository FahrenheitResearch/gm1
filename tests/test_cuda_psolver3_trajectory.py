from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from cm1core.cuda_psolver3_trajectory import (
    ALL_ARRAY_NAMES,
    CUDA_PSOLVER3_TRAJECTORY_SOURCE_BYTES,
    CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256,
    STATE_OUTPUT_NAMES,
    TrajectoryAblations,
    compact_carrier_contract,
    compare_psolver3_trajectory_raw,
    cpu_psolver3_trajectory_inplace,
    trajectory_array_shapes,
    validate_trajectory_contract,
)
from cm1core.cuda_sound3_periodic import (
    default_periodic_face_weights,
    fill_periodic_scalar_inplace,
    fill_periodic_uv_inplace,
)


NI = 7
NJ = 6
NK = 7
PARAMETERS = {
    "ni": NI,
    "nj": NJ,
    "nk": NK,
    "dt": 0.4,
    "nsound": 6,
    "dx": 1000.0,
    "dy": 1000.0,
    "dz": 500.0,
}


def make_fixture() -> dict[str, np.ndarray]:
    shapes = trajectory_array_shapes(NI, NJ, NK)
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    arrays["dtu0"], arrays["dtv0"] = default_periodic_face_weights(NI, NJ)
    for name in ("uf", "vf", "uh", "vh", "mh", "mf", "rho0", "rr0", "rf0", "rrf0"):
        arrays[name].fill(np.float32(1.0))
    arrays["c1"].fill(np.float32(0.4375))
    arrays["c2"].fill(np.float32(0.5625))
    for k in range(NK + 2):
        value = np.float32(300.0) + np.float32(k - 1) * np.float32(2.0**-3)
        arrays["th0"][k].fill(value)
        arrays["rth0"][k].fill(np.float32(1.0) / value)
    arrays["pi0"].fill(np.float32(1.0))

    for k in range(1, NK + 1):
        z = np.float32(k - 1) / np.float32(NK - 1)
        for j0 in range(NJ):
            y = np.float32(2.0 * np.pi * j0 / NJ)
            j = j0 + 1
            for i0 in range(NI):
                x = np.float32(2.0 * np.pi * i0 / NI)
                i = i0 + 1
                bubble = np.float32(0.2) * np.sin(x) * np.cos(y) * np.sin(
                    np.float32(np.pi) * z
                )
                arrays["th3d"][k, j, i] = np.float32(bubble)
                arrays["pp3d"][k, j, i] = np.float32(
                    np.float32(2.0e-4) * np.cos(x) * np.cos(y)
                    * np.sin(np.float32(np.pi) * z)
                )
        for j0 in range(NJ):
            y = np.float32(2.0 * np.pi * j0 / NJ)
            j = j0 + 1
            for f in range(NI + 1):
                x = np.float32(2.0 * np.pi * f / NI)
                arrays["u3d"][k, j, f] = np.float32(
                    np.float32(2.0) + np.float32(0.18) * np.sin(x) * np.cos(y)
                )
        for g in range(NJ + 1):
            y = np.float32(2.0 * np.pi * g / NJ)
            for i0 in range(NI):
                x = np.float32(2.0 * np.pi * i0 / NI)
                arrays["v3d"][k, g, i0 + 1] = np.float32(
                    np.float32(-0.7) - np.float32(0.14) * np.cos(x) * np.sin(y)
                )
    for q in range(2, NK + 1):
        z = np.float32(q - 1) / np.float32(NK)
        for j0 in range(NJ):
            y = np.float32(2.0 * np.pi * j0 / NJ)
            for i0 in range(NI):
                x = np.float32(2.0 * np.pi * i0 / NI)
                arrays["w3d"][q, j0, i0] = np.float32(
                    np.float32(0.03)
                    * np.sin(x)
                    * np.sin(y)
                    * np.sin(np.float32(np.pi) * z)
                )

    for name in ("pi0", "rho0", "rr0", "rf0", "rrf0", "th0", "rth0", "pp3d", "th3d"):
        fill_periodic_scalar_inplace(arrays[name], ni=NI, nj=NJ, nk=NK)
    fill_periodic_uv_inplace(arrays["u3d"], arrays["v3d"], ni=NI, nj=NJ, nk=NK)
    np.copyto(arrays["ua"], arrays["u3d"])
    np.copyto(arrays["va"], arrays["v3d"])
    np.copyto(arrays["wa"], arrays["w3d"])
    np.copyto(arrays["ppi"], arrays["pp3d"])
    np.copyto(arrays["ppx"], arrays["pp3d"])
    np.copyto(arrays["tha"], arrays["th3d"])
    fill_periodic_scalar_inplace(arrays["ppi"], ni=NI, nj=NJ, nk=NK)
    fill_periodic_scalar_inplace(arrays["ppx"], ni=NI, nj=NJ, nk=NK)
    fill_periodic_scalar_inplace(arrays["tha"], ni=NI, nj=NJ, nk=NK)
    validate_trajectory_contract(arrays, ni=NI, nj=NJ, nk=NK)
    return arrays


def test_public_contract_and_validation() -> None:
    arrays = make_fixture()
    assert tuple(arrays) == ALL_ARRAY_NAMES
    contract = compact_carrier_contract(NI, NJ, NK)
    assert contract["state"]["u3d"] == [NK, NJ, NI + 1]
    assert contract["state"]["w3d"] == [NK + 1, NJ, NI]
    auxiliary = contract["required_flat_auxiliary"]
    assert auxiliary["defaults_allowed"] is False
    assert set(auxiliary["source_profile_shapes"]) == {"c1", "c2", "rf0", "mh", "mf"}
    broken = copy.deepcopy(arrays)
    broken["w3d"][1, 0, 0] = np.float32(1.0)
    broken["wa"][1, 0, 0] = np.float32(1.0)
    with pytest.raises(ValueError, match="rigid"):
        validate_trajectory_contract(broken, ni=NI, nj=NJ, nk=NK)

    broken = make_fixture()
    broken["c1"][3].fill(np.float32(0.25))
    with pytest.raises(ValueError, match="sum exactly"):
        validate_trajectory_contract(broken, ni=NI, nj=NJ, nk=NK)

    broken = make_fixture()
    broken["ppx"].view(np.uint32)[3, 2, 2] ^= np.uint32(1)
    with pytest.raises(ValueError, match="fresh entry ppx"):
        validate_trajectory_contract(broken, ni=NI, nj=NJ, nk=NK)


def test_cuda_source_pin_and_raw_comparator_no_null_semantics() -> None:
    assert hashlib.sha256(CUDA_PSOLVER3_TRAJECTORY_SOURCE_BYTES).hexdigest() == (
        CUDA_PSOLVER3_TRAJECTORY_SOURCE_SHA256
    )
    expected = make_fixture()
    candidate = copy.deepcopy(expected)
    comparison = compare_psolver3_trajectory_raw(expected, candidate)
    assert comparison["passed"]
    assert comparison["mode"] == "raw-bit"
    assert comparison["max_ulp"] == 0
    assert comparison["null_or_missing_is_pass"] is False

    missing = dict(candidate)
    missing.pop(STATE_OUTPUT_NAMES[0])
    with pytest.raises(ValueError, match="missing trajectory comparison outputs"):
        compare_psolver3_trajectory_raw(expected, missing)

    nonfinite = copy.deepcopy(candidate)
    nonfinite["pp3d"][3, 2, 2] = np.float32(np.nan)
    with pytest.raises(ValueError, match="must be finite"):
        compare_psolver3_trajectory_raw(expected, nonfinite)


def test_cpu_one_step_cadence_carriers_and_state_transition() -> None:
    arrays = make_fixture()
    before = {name: arrays[name].copy() for name in ("u3d", "v3d", "w3d", "pp3d", "th3d")}
    result = cpu_psolver3_trajectory_inplace(arrays, steps=1, **PARAMETERS)
    assert [(stage.nrk, stage.nloop) for stage in result.cadence] == [(1, 2), (2, 3), (3, 6)]
    assert all(stage.rigid_w_boundary_exact for stage in result.cadence)
    assert all(stage.max_tridiagonal_residual < 2.0e-6 for stage in result.cadence)
    assert all(stage.horizontal_mass_flux_sum == 0.0 for stage in result.cadence)
    assert all(abs(stage.total_mass_flux_divergence_sum) < 1.0e-8 for stage in result.cadence)
    assert result.ledgers[0]["horizontal_mass_flux_telescoping_max_abs_f64"] == 0.0
    assert result.ledgers[0]["total_mass_flux_divergence_sum_max_abs_f64"] < 1.0e-8
    for name in before:
        assert not np.array_equal(before[name].view(np.uint32), arrays[name].view(np.uint32))
    for current, carrier in (("u3d", "ua"), ("v3d", "va"), ("w3d", "wa"), ("pp3d", "ppi"), ("th3d", "tha")):
        assert np.array_equal(arrays[current].view(np.uint32), arrays[carrier].view(np.uint32))
    assert not np.any(arrays["w3d"][1].view(np.uint32))
    assert not np.any(arrays["w3d"][NK + 1].view(np.uint32))


def test_cpu_multi_step_is_repeat_exact() -> None:
    first = make_fixture()
    second = copy.deepcopy(first)
    one = cpu_psolver3_trajectory_inplace(first, steps=3, **PARAMETERS)
    two = cpu_psolver3_trajectory_inplace(second, steps=3, **PARAMETERS)
    assert one.cadence == two.cadence
    assert one.ledgers == two.ledgers
    for name in ("u3d", "v3d", "w3d", "pp3d", "ppx", "th3d"):
        assert np.array_equal(first[name].view(np.uint32), second[name].view(np.uint32))


@pytest.mark.parametrize(
    "ablation",
    [
        TrajectoryAblations(momentum_advection=False),
        TrajectoryAblations(scalar_advection=False),
        TrajectoryAblations(base_theta_advection=False),
        TrajectoryAblations(buoyancy=False),
        TrajectoryAblations(post_step_ppx=False),
        TrajectoryAblations(carrier_promotion=False),
    ],
)
def test_negative_ablations_are_detected(ablation: TrajectoryAblations) -> None:
    reference = make_fixture()
    candidate = copy.deepcopy(reference)
    cpu_psolver3_trajectory_inplace(reference, steps=1, **PARAMETERS)
    cpu_psolver3_trajectory_inplace(candidate, steps=1, ablations=ablation, **PARAMETERS)
    assert any(
        not np.array_equal(reference[name].view(np.uint32), candidate[name].view(np.uint32))
        for name in ("ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "ppx", "tha", "th3d")
    )
