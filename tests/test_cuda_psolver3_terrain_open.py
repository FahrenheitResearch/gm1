from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_psolver3_terrain_open import (
    COMPARE_OUTPUT_NAMES,
    CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
    CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_SCHEMA,
    TerrainOpenSoundAblations,
    compare_psolver3_terrain_open_raw,
    cpu_psolver3_terrain_open_three_stage_inplace,
    launch_cuda_psolver3_terrain_open_stage,
    launch_cuda_psolver3_terrain_open_three_stage,
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


ROOT = Path(__file__).resolve().parents[1]
NG = 3


def _metrics(ni: int, nj: int, nk: int) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for index, (name, shape) in enumerate(
        terrain_static_shapes(ni=ni, nj=nj, nk=nk).items(), start=1
    ):
        values[name] = np.full(shape, np.float32(1.0 + 0.001 * index), dtype=np.float32)
    terrain = np.empty((nj, ni), dtype=np.float32)
    for j in range(nj):
        for i in range(ni):
            terrain[j, i] = np.float32(350.0 + 3.0 * i - 1.5 * j)
    values["terrain_msl"][:] = terrain
    zt = np.float32(nk * 200.0)
    values["gz"][:] = np.divide(zt, np.subtract(zt, terrain, dtype=np.float32), dtype=np.float32)
    values["rgz"][:] = np.divide(np.float32(1.0), values["gz"], dtype=np.float32)
    values["gzu"][:] = np.float32(1.03)
    values["rgzu"][:] = np.divide(np.float32(1.0), values["gzu"], dtype=np.float32)
    values["gzv"][:] = np.float32(1.031)
    values["rgzv"][:] = np.divide(np.float32(1.0), values["gzv"], dtype=np.float32)
    values["dzdx"][:] = np.float32(0.01)
    values["dzdy"][:] = np.float32(-0.0075)
    values["c1"][:] = np.float32(0.5)
    values["c2"][:] = np.float32(0.5)
    for k in range(nk):
        values["zh"][k] = terrain + np.float32((k + 0.5) * 200.0)
        density = np.float32(1.1 - 0.08 * k)
        values["rho0_3d"][k] = density
        values["rr0"][k] = np.float32(1.0) / density
    for k in range(nk + 1):
        values["zf"][k] = terrain + np.float32(k * 200.0)
        density = np.float32(1.14 - 0.08 * k)
        values["rf0"][k] = density
        values["rrf0"][k] = np.float32(1.0) / density
        values["gx"][k] = np.float32(0.001 * (k + 1))
        values["gxu"][k] = np.float32(0.0012 * (k + 1))
        values["gy"][k] = np.float32(-0.0008 * (k + 1))
        values["gyv"][k] = np.float32(-0.0011 * (k + 1))
    values["mh"][:] = values["gz"][None]
    values["rmh"][:] = np.divide(np.float32(1.0), values["mh"], dtype=np.float32)
    values["mf"][:] = values["gz"][None]
    values["rmf"][:] = np.divide(np.float32(1.0), values["mf"], dtype=np.float32)
    return values


def _arrays(ni: int, nj: int, nk: int) -> dict[str, np.ndarray]:
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    a = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    for name in ("uh", "vh", "ruf", "rvf"):
        a[name][:] = np.float32(1.0)
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            y = j + NG - 1
            for face in range(1, ni + 2):
                a["u3d"][k, y, face + NG - 1] = np.float32(
                    1.5 + 0.03 * face - 0.01 * j + 0.02 * k
                )
            for i in range(1, ni + 1):
                x = i + NG - 1
                a["w3d"][k, y, x] = np.float32(
                    0.02 * np.sin(0.3 * i + 0.2 * j + 0.1 * k)
                )
        for face in range(1, nj + 2):
            y = face + NG - 1
            for i in range(1, ni + 1):
                a["v3d"][k, y, i + NG - 1] = np.float32(
                    -0.7 + 0.02 * i + 0.025 * face - 0.01 * k
                )
    cpu_bcu_open_inplace(a["u3d"], ni=ni, nj=nj, nk=nk)
    cpu_bcv_open_inplace(a["v3d"], ni=ni, nj=nj, nk=nk)
    cpu_bcw_open_inplace(a["w3d"], ni=ni, nj=nj, nk=nk, flag=1)
    np.copyto(a["ua"], a["u3d"])
    np.copyto(a["va"], a["v3d"])
    np.copyto(a["wa"], a["w3d"])
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                perturbation = np.float32(0.0001 * np.sin(0.3 * i + 0.2 * j + 0.1 * k))
                a["ppi"][k, j, i] = perturbation
                a["pp3d"][k, j, i] = perturbation
                a["ppx"][k, j, i] = perturbation
                a["piadv"][k, j, i] = np.float32(0.9 + perturbation)
                a["ppterm"][k, j, i] = np.float32(a["piadv"][k, j, i] * np.float32(287.04 / (1005.7 - 287.04)))
                a["thv"][k, j, i] = np.float32(300.0 + 0.3 * k + 0.02 * i - 0.01 * j)
                a["ppten"][k, j, i] = np.float32(1.0e-6 * (i - j))
    for name in ("ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv"):
        cpu_bcp_open_inplace(a[name], ni=ni, nj=nj, nk=nk)
    a["scalar"][1 : nk + 1, NG : NG + nj, NG : NG + ni] = np.float32(0.25)
    cpu_bcs_open_inplace(a["scalar"], ni=ni, nj=nj, nk=nk)
    a["radbcw"][:] = np.float32(0.0)
    a["radbce"][:] = np.float32(0.0)
    a["radbcs"][:] = np.float32(0.0)
    a["radbcn"][:] = np.float32(0.0)
    return a


def _clone(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy(order="C") for name, value in values.items()}


def test_import_is_cuda_cold_and_shapes_preserve_native_families() -> None:
    command = (
        "import sys; import cm1core.cuda_psolver3_terrain_open as m; "
        "assert not any(x == 'cupy' or x.startswith('cupy.') for x in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, env=environment, check=True)
    shapes = terrain_open_sound_shapes(5, 4, 5)
    assert shapes["pp3d"] == (7, 6, 7)
    assert shapes["u3d"] == (7, 10, 12)
    assert shapes["v3d"] == (7, 11, 11)
    assert shapes["w3d"] == (8, 10, 11)
    assert "full trajectory" in CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT


def test_cpu_three_stage_transition_is_deterministic_and_has_native_cadence() -> None:
    ni, nj, nk = 5, 4, 5
    metrics = _metrics(ni, nj, nk)
    initial = _arrays(ni, nj, nk)
    first, second = _clone(initial), _clone(initial)
    result1 = cpu_psolver3_terrain_open_three_stage_inplace(
        first, metrics, ni=ni, nj=nj, nk=nk, dt=0.6, nsound=6, dx=100.0, dy=100.0, dz=200.0
    )
    result2 = cpu_psolver3_terrain_open_three_stage_inplace(
        second, metrics, ni=ni, nj=nj, nk=nk, dt=0.6, nsound=6, dx=100.0, dy=100.0, dz=200.0
    )
    assert [item.nloop for item in result1.cadence] == [2, 3, 6]
    assert [item.open_consumptions for item in result1.cadence] == [2, 3, 6]
    assert [item.pressure_fills for item in result1.cadence] == [1, 2, 5]
    assert [item.ppx_written for item in result1.cadence] == [False, False, True]
    assert result1 == result2
    comparison = compare_psolver3_terrain_open_raw(first, second)
    assert comparison["passed"] is True
    assert all(np.isfinite(first[name]).all() for name in COMPARE_OUTPUT_NAMES)
    assert any(
        not np.array_equal(first[name].view(np.uint32), initial[name].view(np.uint32))
        for name in ("u3d", "v3d", "w3d", "pp3d")
    )
    assert max(item.max_tridiagonal_residual for item in result1.cadence) < 1.0e-5


def test_acoustic_ablation_controls_are_detected() -> None:
    ni, nj, nk = 5, 4, 5
    metrics = _metrics(ni, nj, nk)
    initial = _arrays(ni, nj, nk)
    baseline = _clone(initial)
    cpu_psolver3_terrain_open_three_stage_inplace(
        baseline, metrics, ni=ni, nj=nj, nk=nk, dt=0.6, nsound=6, dx=100.0, dy=100.0, dz=200.0
    )
    for ablations in (
        TerrainOpenSoundAblations(open_consumption=False),
        TerrainOpenSoundAblations(terrain_pressure_gradient=False),
        TerrainOpenSoundAblations(transformed_divergence=False),
        TerrainOpenSoundAblations(pressure_advection=False),
        TerrainOpenSoundAblations(implicit_vertical_solve=False),
        TerrainOpenSoundAblations(between_step_pressure_fill=False),
        TerrainOpenSoundAblations(radiation_history_production=False),
        TerrainOpenSoundAblations(stage_family_fills=False),
    ):
        candidate = _clone(initial)
        cpu_psolver3_terrain_open_three_stage_inplace(
            candidate,
            metrics,
            ni=ni,
            nj=nj,
            nk=nk,
            dt=0.6,
            nsound=6,
            dx=100.0,
            dy=100.0,
            dz=200.0,
            ablations=ablations,
        )
        assert compare_psolver3_terrain_open_raw(baseline, candidate)["passed"] is False


def test_comparator_refuses_missing_output() -> None:
    values = _arrays(5, 4, 5)
    candidate = copy.copy(values)
    candidate.pop("tri_residual")
    with pytest.raises(ValueError, match="missing terrain/open comparison output tri_residual"):
        compare_psolver3_terrain_open_raw(values, candidate)


def test_cuda_stage_selector_refuses_before_importing_cuda() -> None:
    with pytest.raises(ValueError, match="stage 1, 2, or 3"):
        launch_cuda_psolver3_terrain_open_stage(
            {},
            {},
            ni=5,
            nj=4,
            nk=5,
            nrk=0,
            dt=0.6,
            nsound=6,
            dx=100.0,
            dy=100.0,
            dz=200.0,
        )


def _raw_mismatch_summary(
    expected: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in COMPARE_OUTPUT_NAMES:
        left = expected[name].view(np.uint32)
        right = candidate[name].view(np.uint32)
        indices = np.argwhere(left != right)
        if not indices.size:
            continue
        first = tuple(int(value) for value in indices[0])
        summary[name] = {
            "mismatch_count": int(indices.shape[0]),
            "first_index": first,
            "expected_word": f"{int(left[first]):08x}",
            "candidate_word": f"{int(right[first]):08x}",
        }
    return summary


def test_cuda_three_stage_transition_matches_cpu_only_with_mutex() -> None:
    if os.environ.get("CM1CORE_RUN_CUDA_PSOLVER3_TERRAIN_OPEN_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_PSOLVER3_TERRAIN_OPEN_TEST=1 only while "
            "holding the GPU mutex"
        )
    cupy = pytest.importorskip("cupy")
    ni, nj, nk = 5, 4, 5
    cases = {
        "core": TerrainOpenSoundAblations(False, False, False, False, False, False, False, False),
        "open": TerrainOpenSoundAblations(True, False, False, False, False, False, False, False),
        "pgrad": TerrainOpenSoundAblations(False, True, False, False, False, False, False, False),
        "divergence": TerrainOpenSoundAblations(False, False, True, False, False, False, False, False),
        "advection": TerrainOpenSoundAblations(False, False, False, True, False, False, False, False),
        "implicit": TerrainOpenSoundAblations(False, False, False, False, True, False, False, False),
        "pressure_fill": TerrainOpenSoundAblations(False, False, False, False, False, True, False, False),
        "radiation": TerrainOpenSoundAblations(False, False, False, False, False, False, True, False),
        "family_fills": TerrainOpenSoundAblations(False, False, False, False, False, False, False, True),
        "all": TerrainOpenSoundAblations(),
    }
    metrics = _metrics(ni, nj, nk)
    initial = _arrays(ni, nj, nk)
    compiled: dict[str, object] = {}
    failures: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    started = time.perf_counter()
    for label, ablations in cases.items():
        expected = _clone(initial)
        cpu_psolver3_terrain_open_three_stage_inplace(
            expected,
            metrics,
            ni=ni,
            nj=nj,
            nk=nk,
            dt=0.6,
            nsound=6,
            dx=100.0,
            dy=100.0,
            dz=200.0,
            ablations=ablations,
        )
        device = {name: cupy.asarray(value) for name, value in initial.items()}
        device_metrics = {name: cupy.asarray(value) for name, value in metrics.items()}
        launch = launch_cuda_psolver3_terrain_open_three_stage(
            device,
            device_metrics,
            ni=ni,
            nj=nj,
            nk=nk,
            dt=0.6,
            nsound=6,
            dx=100.0,
            dy=100.0,
            dz=200.0,
            ablations=ablations,
            **compiled,
        )
        launch.ready_event.synchronize()
        if not compiled:
            compiled = {
                "compiled": launch.compiled,
                "terrain_compiled": launch.terrain_compiled,
                "open_compiled": launch.open_compiled,
                "lateral_compiled": launch.lateral_compiled,
            }
        candidate = {name: cupy.asnumpy(device[name]) for name in COMPARE_OUTPUT_NAMES}
        comparison = compare_psolver3_terrain_open_raw(expected, candidate)
        comparisons[label] = comparison
        if comparison["passed"] is not True:
            failures[label] = _raw_mismatch_summary(expected, candidate)
    assert failures == {}
    evidence_path = os.environ.get("CM1CORE_TERRAIN_OPEN_SYNTHETIC_EVIDENCE_PATH")
    if evidence_path:
        payload = {
            "cases": comparisons,
            "claim_limit": CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT,
            "comparison_contract": {
                "max_ulp": 0,
                "mode": "raw-bit",
                "null_or_missing_is_pass": False,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "kernel": {
                "source_sha256": CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
                "target_arch": launch.compiled.target_arch,
            },
            "schema": "gpuwm-cm1.cuda-psolver3-terrain-open-synthetic/v1",
            "status": "passed-10-of-10-raw-bit-exact",
        }
        Path(evidence_path).write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )


def test_cuda_explicit_stage_sequence_matches_three_stage_only_with_mutex() -> None:
    if os.environ.get("CM1CORE_RUN_CUDA_PSOLVER3_STAGE_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_PSOLVER3_STAGE_TEST=1 only while holding the GPU mutex"
        )
    cupy = pytest.importorskip("cupy")
    ni, nj, nk = 5, 4, 5
    metrics = _metrics(ni, nj, nk)
    initial = _arrays(ni, nj, nk)
    device_metrics = {name: cupy.asarray(value) for name, value in metrics.items()}

    composed = {name: cupy.asarray(value) for name, value in initial.items()}
    whole = launch_cuda_psolver3_terrain_open_three_stage(
        composed,
        device_metrics,
        ni=ni,
        nj=nj,
        nk=nk,
        dt=0.6,
        nsound=6,
        dx=100.0,
        dy=100.0,
        dz=200.0,
    )
    whole.ready_event.synchronize()

    staged = {name: cupy.asarray(value) for name, value in initial.items()}
    compiled = {
        "compiled": whole.compiled,
        "terrain_compiled": whole.terrain_compiled,
        "open_compiled": whole.open_compiled,
        "lateral_compiled": whole.lateral_compiled,
    }
    launches = []
    for nrk in (1, 2, 3):
        launch = launch_cuda_psolver3_terrain_open_stage(
            staged,
            device_metrics,
            ni=ni,
            nj=nj,
            nk=nk,
            nrk=nrk,
            dt=0.6,
            nsound=6,
            dx=100.0,
            dy=100.0,
            dz=200.0,
            **compiled,
        )
        assert launch.receipt["schema"] == CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_SCHEMA
        assert launch.receipt["nrk"] == nrk
        launches.append(launch)
    launches[-1].ready_event.synchronize()
    expected = {name: cupy.asnumpy(composed[name]) for name in COMPARE_OUTPUT_NAMES}
    candidate = {name: cupy.asnumpy(staged[name]) for name in COMPARE_OUTPUT_NAMES}
    assert compare_psolver3_terrain_open_raw(expected, candidate)["passed"] is True
