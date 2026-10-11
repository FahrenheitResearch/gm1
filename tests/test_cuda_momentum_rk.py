from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from authority_trees import requires_native_reference
from cm1core.cuda_momentum_rk import (
    CM1_BC_F_SHA256,
    CM1_SOLVE1_F_SHA256,
    CM1_SOLVE2_F_SHA256,
    CM1_SOLVE3_F_SHA256,
    CM1_SOUND_F_SHA256,
    CUDA_MOMENTUM_RK_CLAIM_LIMIT,
    CUDA_MOMENTUM_RK_SCHEMA,
    CUDA_MOMENTUM_RK_SOURCE_BYTES,
    CUDA_MOMENTUM_RK_SOURCE_SHA256,
    begin_cuda_momentum_rk_step,
    compile_cuda_momentum_rk,
    cpu_bc2d_open_w_surface_inplace,
    cpu_momentum_rk_stage_inplace,
    cpu_seed_momentum_rk_tendencies_inplace,
    create_cuda_momentum_rk_state,
    launch_cuda_momentum_rk_stage,
    momentum_rk_boundary_cadence,
    momentum_rk_live_regions,
    momentum_rk_stage_timestep_f32,
)
from cm1core.cuda_psolver3_terrain_open import (
    COMPARE_OUTPUT_NAMES,
    TerrainOpenSoundAblations,
    compare_psolver3_terrain_open_raw,
    terrain_open_sound_shapes,
)
from cm1core.cuda_sound3_open_boundary import (
    cpu_bcp_open_inplace,
    cpu_bcs_open_inplace,
    cpu_bcu_open_inplace,
    cpu_bcv_open_inplace,
    cpu_bcw_open_inplace,
)
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes
from host_portability import recorded_name


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/cuda-momentum-rk-v1"
NI = 8
NJ = 8
NK = 8
NG = 3
PARAMETERS = {
    "ni": NI,
    "nj": NJ,
    "nk": NK,
    "dt": 0.6,
    "nsound": 6,
    "dx": 100.0,
    "dy": 100.0,
    "dz": 200.0,
}


def _metrics() -> dict[str, np.ndarray]:
    values = {
        name: np.full(shape, np.float32(1.0 + 0.001 * index), dtype=np.float32)
        for index, (name, shape) in enumerate(
            terrain_static_shapes(ni=NI, nj=NJ, nk=NK).items(), start=1
        )
    }
    y, x = np.indices((NJ, NI), dtype=np.float32)
    terrain = np.float32(120.0) + np.float32(2.5) * x - np.float32(1.25) * y
    values["terrain_msl"][:] = terrain
    zt = np.float32(NK * PARAMETERS["dz"])
    values["gz"][:] = np.divide(
        zt, np.subtract(zt, terrain, dtype=np.float32), dtype=np.float32
    )
    values["rgz"][:] = np.divide(np.float32(1.0), values["gz"], dtype=np.float32)
    values["gzu"][:] = np.float32(1.03)
    values["rgzu"][:] = np.float32(1.0) / values["gzu"]
    values["gzv"][:] = np.float32(1.031)
    values["rgzv"][:] = np.float32(1.0) / values["gzv"]
    values["dzdx"][:] = np.float32(0.002) * (x - np.float32(3.5))
    values["dzdy"][:] = np.float32(-0.0015) * (y - np.float32(3.5))
    values["c1"][:] = np.float32(0.5)
    values["c2"][:] = np.float32(0.5)
    for k in range(NK):
        values["zh"][k] = terrain + np.float32((k + 0.5) * PARAMETERS["dz"])
        density = np.float32(1.15 - 0.025 * k)
        values["rho0_3d"][k] = density
        values["rr0"][k] = np.float32(1.0) / density
    for k in range(NK + 1):
        values["zf"][k] = terrain + np.float32(k * PARAMETERS["dz"])
        density = np.float32(1.16 - 0.024 * k)
        values["rf0"][k] = density
        values["rrf0"][k] = np.float32(1.0) / density
        values["gx"][k] = np.float32(0.0004 * (k + 1))
        values["gxu"][k] = np.float32(0.00045 * (k + 1))
        values["gy"][k] = np.float32(-0.0003 * (k + 1))
        values["gyv"][k] = np.float32(-0.00035 * (k + 1))
    values["mh"][:] = values["gz"][None]
    values["rmh"][:] = np.float32(1.0) / values["mh"]
    values["mf"][:] = values["gz"][None]
    values["rmf"][:] = np.float32(1.0) / values["mf"]
    return values


def _arrays() -> dict[str, np.ndarray]:
    shapes = terrain_open_sound_shapes(NI, NJ, NK)
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    for name in ("uh", "vh", "ruf", "rvf"):
        arrays[name].fill(np.float32(1.0))
    for k in range(1, NK + 1):
        for j in range(1, NJ + 1):
            y = j + NG - 1
            for face in range(1, NI + 2):
                arrays["u3d"][k, y, face + NG - 1] = np.float32(
                    1.1 + 0.025 * face - 0.012 * j + 0.018 * k
                )
            for i in range(1, NI + 1):
                x = i + NG - 1
                arrays["w3d"][k, y, x] = np.float32(
                    0.015 * np.sin(0.21 * i + 0.17 * j + 0.13 * k)
                )
        for face in range(1, NJ + 2):
            y = face + NG - 1
            for i in range(1, NI + 1):
                arrays["v3d"][k, y, i + NG - 1] = np.float32(
                    -0.55 + 0.016 * i + 0.021 * face - 0.009 * k
                )
    cpu_bcu_open_inplace(arrays["u3d"], ni=NI, nj=NJ, nk=NK)
    cpu_bcv_open_inplace(arrays["v3d"], ni=NI, nj=NJ, nk=NK)
    cpu_bcw_open_inplace(arrays["w3d"], ni=NI, nj=NJ, nk=NK, flag=1)
    np.copyto(arrays["ua"], arrays["u3d"])
    np.copyto(arrays["va"], arrays["v3d"])
    np.copyto(arrays["wa"], arrays["w3d"])
    for k in range(1, NK + 1):
        for j in range(1, NJ + 1):
            for i in range(1, NI + 1):
                perturbation = np.float32(0.00008 * np.sin(0.3 * i + 0.2 * j + 0.1 * k))
                arrays["ppi"][k, j, i] = perturbation
                arrays["pp3d"][k, j, i] = perturbation
                arrays["ppx"][k, j, i] = perturbation
                arrays["piadv"][k, j, i] = np.float32(0.9) + perturbation
                arrays["ppterm"][k, j, i] = np.float32(
                    arrays["piadv"][k, j, i] * np.float32(287.04 / (1005.7 - 287.04))
                )
                arrays["thv"][k, j, i] = np.float32(
                    300.0 + 0.22 * k + 0.015 * i - 0.008 * j
                )
                arrays["ppten"][k, j, i] = np.float32(8.0e-7 * (i - j))
    for name in ("ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv"):
        cpu_bcp_open_inplace(arrays[name], ni=NI, nj=NJ, nk=NK)
    arrays["scalar"][1 : NK + 1, NG : NG + NJ, NG : NG + NI] = np.float32(0.25)
    cpu_bcs_open_inplace(arrays["scalar"], ni=NI, nj=NJ, nk=NK)
    return arrays


def _held() -> dict[str, np.ndarray]:
    shapes = terrain_open_sound_shapes(NI, NJ, NK)
    values = {
        "uten1": np.zeros(shapes["uten"], dtype=np.float32),
        "vten1": np.zeros(shapes["vten"], dtype=np.float32),
        "wten1": np.zeros(shapes["wten"], dtype=np.float32),
    }
    for number, value in enumerate(values.values(), start=1):
        z, y, x = np.indices(value.shape, dtype=np.float32)
        value[:] = (
            np.float32(number * 0.0002)
            + np.float32(1.0e-7) * x
            - np.float32(0.75e-7) * y
            + np.float32(0.5e-7) * z
        )
    return values


def _clone(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy(order="C") for name, value in values.items()}


def test_import_is_cuda_cold_and_kernel_source_is_pinned() -> None:
    command = (
        "import sys; import cm1core.cuda_momentum_rk as m; "
        "assert m.CUDA_MOMENTUM_RK_SOURCE_SHA256; "
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
        hashlib.sha256(CUDA_MOMENTUM_RK_SOURCE_BYTES).hexdigest()
        == CUDA_MOMENTUM_RK_SOURCE_SHA256
    )
    assert "complete RK stage" in CUDA_MOMENTUM_RK_CLAIM_LIMIT


@requires_native_reference
def test_authority_sources_are_exact_cm1_r21_1_bytes() -> None:
    expected = {
        "solve1.F": CM1_SOLVE1_F_SHA256,
        "solve2.F": CM1_SOLVE2_F_SHA256,
        "solve3.F": CM1_SOLVE3_F_SHA256,
        "sound.F": CM1_SOUND_F_SHA256,
        "bc.F": CM1_BC_F_SHA256,
    }
    authority = ROOT / "work/native-reference/cm1r21.1/src"
    for name, digest in expected.items():
        assert hashlib.sha256((authority / name).read_bytes()).hexdigest() == digest


def test_stage_timestep_regions_and_boundary_cadence_are_source_exact() -> None:
    assert [
        momentum_rk_stage_timestep_f32(0.6, nrk).view(np.uint32).item()
        for nrk in (1, 2, 3)
    ] == [
        np.float32(0.2).view(np.uint32).item(),
        np.float32(0.3).view(np.uint32).item(),
        np.float32(0.6).view(np.uint32).item(),
    ]
    cadence = [momentum_rk_boundary_cadence(nrk, dt=0.6, nsound=6) for nrk in (1, 2, 3)]
    assert [item["acoustic"]["nloop"] for item in cadence] == [2, 3, 6]
    assert [item["stage_entry_terrain_bcwsfc"] for item in cadence] == [0, 1, 1]
    assert [item["stage_entry_open_bc2d_bottom"] for item in cadence] == [0, 1, 1]
    assert [item["radiation4_history_productions"] for item in cadence] == [1, 1, 1]
    assert [item["stage_scalar_pressure_family_fills"] for item in cadence] == [1, 1, 0]
    regions = momentum_rk_live_regions(7, 9, 11)
    assert regions["source_seed_common"] == {"i": [1, 8], "j": [1, 10], "k": [1, 11]}
    assert regions["u_physical"]["i"] == [1, 8]
    assert regions["v_physical"]["j"] == [1, 10]
    assert regions["w_physical"]["k"] == [1, 12]
    assert regions["u_pressure_advection_interior"]["i"] == [2, 7]
    assert regions["v_pressure_advection_interior"]["j"] == [2, 9]
    assert regions["w_advection_implicit_interior"]["k"] == [2, 11]


def test_cpu_held_seed_is_raw_copy_on_exact_common_source_rectangle() -> None:
    arrays = _arrays()
    held = _held()
    patterns = np.array(
        [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x3F800001],
        dtype=np.uint32,
    )
    for value in held.values():
        words = value.view(np.uint32).reshape(-1)
        words[:] = np.resize(patterns, words.size)
    before = {
        name: arrays[name].view(np.uint32).copy() for name in ("uten", "vten", "wten")
    }
    cpu_seed_momentum_rk_tendencies_inplace(held, arrays, ni=NI, nj=NJ, nk=NK)
    region = (slice(1, NK + 1), slice(NG, NG + NJ + 1), slice(NG, NG + NI + 1))
    for held_name, tendency_name in zip(
        ("uten1", "vten1", "wten1"), ("uten", "vten", "wten"), strict=True
    ):
        assert np.array_equal(
            arrays[tendency_name][region].view(np.uint32),
            held[held_name][region].view(np.uint32),
        )
        mask = np.zeros(arrays[tendency_name].shape, dtype=bool)
        mask[region] = True
        assert np.array_equal(
            arrays[tendency_name].view(np.uint32)[~mask], before[tendency_name][~mask]
        )


def test_cpu_bc2d_bottom_is_raw_copy_with_source_corner_order() -> None:
    w = _arrays()["w3d"]
    plane = w[1].view(np.uint32)
    y, x = np.indices(plane.shape, dtype=np.uint32)
    plane[:] = np.uint32(0x3F000000) + y * np.uint32(101) + x
    before = plane.copy()
    cpu_bc2d_open_w_surface_inplace(w, ni=NI, nj=NJ, nk=NK)
    expected = before.copy()
    expected[:, :NG] = expected[:, NG : NG + 1]
    expected[:, NG + NI : NG + NI + NG] = expected[:, NG + NI - 1 : NG + NI]
    expected[:NG, :] = expected[NG : NG + 1, :]
    expected[NG + NJ : NG + NJ + NG, :] = expected[NG + NJ - 1 : NG + NJ, :]
    assert np.array_equal(plane, expected)
    assert plane[0, 0] == before[NG, NG]
    assert plane[-1, -1] == before[NG + NJ - 1, NG + NI - 1]
    assert np.array_equal(
        w[0].view(np.uint32), np.zeros_like(w[0], dtype=np.float32).view(np.uint32)
    )


def test_cpu_bounded_stage_is_deterministic_and_reports_exact_cadence() -> None:
    initial = _arrays()
    metrics = _metrics()
    held = _held()
    first, second = _clone(initial), _clone(initial)
    first_divx = np.empty((NK, NJ, NI), dtype=np.float32)
    second_divx = np.empty_like(first_divx)
    result1 = cpu_momentum_rk_stage_inplace(
        first, metrics, held, first_divx, nrk=1, **PARAMETERS
    )
    result2 = cpu_momentum_rk_stage_inplace(
        second, metrics, held, second_divx, nrk=1, **PARAMETERS
    )
    assert result1 == result2
    assert result1.acoustic.nloop == 2
    assert result1.stage_entry_terrain_refreshes == 0
    assert np.array_equal(first_divx.view(np.uint32), second_divx.view(np.uint32))
    for name in (*COMPARE_OUTPUT_NAMES, "uten", "vten", "wten"):
        assert np.array_equal(
            first[name].view(np.uint32), second[name].view(np.uint32)
        ), name
        assert np.isfinite(first[name]).all(), name
    assert compare_psolver3_terrain_open_raw(first, second)["passed"] is True
    assert np.count_nonzero(first_divx)


def test_coordinator_refuses_acoustic_ablations_before_mutating_tendencies() -> None:
    arrays = _arrays()
    before = {
        name: arrays[name].view(np.uint32).copy() for name in ("uten", "vten", "wten")
    }
    with pytest.raises(ValueError, match="every admitted terrain/open"):
        cpu_momentum_rk_stage_inplace(
            arrays,
            _metrics(),
            _held(),
            np.empty((NK, NJ, NI), dtype=np.float32),
            nrk=1,
            acoustic_ablations=TerrainOpenSoundAblations(open_consumption=False),
            **PARAMETERS,
        )
    for name in before:
        assert np.array_equal(arrays[name].view(np.uint32), before[name])


def test_cuda_cadence_guard_refuses_out_of_order_before_compilation() -> None:
    with pytest.raises(TypeError, match="CudaMomentumRKState"):
        launch_cuda_momentum_rk_stage(
            object(), nrk=1, dt=0.6, nsound=6, dx=100.0, dy=100.0, dz=200.0
        )


def test_corrected_real_three_stage_evidence_is_complete_and_hash_bound() -> None:
    receipt_path = EVIDENCE / "corrected-real-crop-sm86.json"
    figure_path = EVIDENCE / "corrected-real-crop-stage3.png"
    inventory = json.loads((EVIDENCE / "artifact-inventory.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "gpuwm-cm1.cuda-momentum-rk-evidence/v1"
    assert receipt["result"] == "passed-corrected-real-crop-three-stage-raw-bit-exact"
    assert receipt["data_classification"].endswith("no synthetic field generation")
    assert receipt["domain"] == {
        "crop_origin_zero_based_yx": [142, 142],
        "ni": 16,
        "nj": 16,
        "nk": 12,
    }
    assert [item["acoustic"]["nloop"] for item in receipt["cpu_cadence"]] == [2, 3, 6]
    assert [item["acoustic"]["nloop"] for item in receipt["gpu_cadence"]] == [2, 3, 6]
    assert len(receipt["comparison"]) == 19
    assert all(item["passed"] is True for item in receipt["comparison"].values())
    assert all(item["max_ulp"] == 0 for item in receipt["comparison"].values())
    assert receipt["compiled_images"]["momentum_rk"]["target_arch"] == "86"
    assert receipt["gpu_mutex"]["clean_release"] is True
    assert receipt["gpu_mutex"]["existing_owner_evicted"] is False
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    by_name = {recorded_name(item["path"]): item for item in inventory["artifacts"]}
    for path in (receipt_path, figure_path):
        item = by_name[path.name]
        payload = path.read_bytes()
        assert item["nbytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="bounded GPU gate is opt-in and requires the canonical mutex",
)
def test_cuda_full_bounded_stage_is_raw_bit_equal_to_cpu() -> None:
    cupy = pytest.importorskip("cupy")
    initial = _arrays()
    metrics = _metrics()
    held = _held()
    expected = _clone(initial)
    expected_divx = np.empty((NK, NJ, NI), dtype=np.float32)
    cpu_momentum_rk_stage_inplace(
        expected, metrics, held, expected_divx, nrk=1, **PARAMETERS
    )
    device_arrays = {name: cupy.asarray(value) for name, value in initial.items()}
    device_metrics = {name: cupy.asarray(value) for name, value in metrics.items()}
    device_held = {name: cupy.asarray(value) for name, value in held.items()}
    state = create_cuda_momentum_rk_state(
        device_arrays, device_metrics, device_held, ni=NI, nj=NJ, nk=NK
    )
    launch = launch_cuda_momentum_rk_stage(state, nrk=1, **PARAMETERS)
    launch.ready_event.synchronize()
    assert launch.receipt["schema"] == CUDA_MOMENTUM_RK_SCHEMA
    assert launch.compiled.target_arch == "86"
    assert state.next_nrk == 2 and state.completed is False
    assert np.array_equal(
        cupy.asnumpy(state.divx).view(np.uint32), expected_divx.view(np.uint32)
    )
    observed = {
        name: cupy.asnumpy(device_arrays[name]) for name in COMPARE_OUTPUT_NAMES
    }
    assert compare_psolver3_terrain_open_raw(expected, observed)["passed"] is True
    for name in ("uten", "vten", "wten"):
        assert np.array_equal(
            cupy.asnumpy(device_arrays[name]).view(np.uint32),
            expected[name].view(np.uint32),
        ), name
    with pytest.raises(ValueError, match="before stage 3"):
        begin_cuda_momentum_rk_step(state)


def test_compile_entry_points_only_with_gpu_opt_in() -> None:
    if os.environ.get("CM1_RUN_GPU_TESTS") != "1":
        pytest.skip("bounded GPU gate is opt-in")
    compiled = compile_cuda_momentum_rk()
    assert len(compiled.kernels) == 2
    assert compiled.target_arch == "86"
