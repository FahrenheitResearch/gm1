from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_initialized_product_terrain_open import (
    AUTHORITY_SOURCE_SHA256,
    CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT,
    CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE,
    CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256,
    PREPARATION_COMPARE_NAMES,
    compare_initialized_terrain_open_preparation_raw,
    cpu_pack_initialized_product_terrain_open,
    prepare_cuda_initialized_product_terrain_open,
)
from cm1core.cuda_psolver3_terrain_open import (
    COMPARE_OUTPUT_NAMES,
    compare_psolver3_terrain_open_raw,
    cpu_psolver3_terrain_open_three_stage_inplace,
    launch_cuda_psolver3_terrain_open_three_stage,
)
from cm1core.cuda_terrain_open_boundary import TERRAIN_STATIC_REQUIRED, terrain_static_shapes


ROOT = Path(__file__).resolve().parents[1]
REAL_CROP = ROOT / "work/initialized-product-v3/crop-j142-i142-16x16"
MANIFEST_SHA256 = "2c09879f73afff9f8004134896eedb4b28e96c91194e871967c18f9359fb57ad"
PAYLOAD_SHA256 = "2237bc92d839cfd96b6e4cc82c9c39dc55c8f308353fc9b0b8e7db1db1bf6f82"
SOURCE_ARTIFACT_SHA256 = "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f"
SOURCE_CONTRACT_SHA256 = "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12"
NG = 3


def _carriers(ni: int = 5, nj: int = 4, nk: int = 5) -> dict[str, np.ndarray]:
    mass = (nk, nj, ni)
    result: dict[str, np.ndarray] = {
        "pi0": np.linspace(0.91, 0.82, nk, dtype=np.float32),
        "th0": np.linspace(299.0, 307.0, nk, dtype=np.float32),
        "ua": np.zeros((nk, nj, ni + 1), dtype=np.float32),
        "u3d": np.zeros((nk, nj, ni + 1), dtype=np.float32),
        "va": np.zeros((nk, nj + 1, ni), dtype=np.float32),
        "v3d": np.zeros((nk, nj + 1, ni), dtype=np.float32),
        "wa": np.zeros((nk + 1, nj, ni), dtype=np.float32),
        "w3d": np.zeros((nk + 1, nj, ni), dtype=np.float32),
        "ppi": np.zeros(mass, dtype=np.float32),
        "pp3d": np.zeros(mass, dtype=np.float32),
        "ppx": np.zeros(mass, dtype=np.float32),
        "th3d": np.zeros(mass, dtype=np.float32),
        "q3d": np.zeros((6, nk, nj, ni), dtype=np.float32),
    }
    for index, (name, shape) in enumerate(
        terrain_static_shapes(ni=ni, nj=nj, nk=nk).items(), start=1
    ):
        result[name] = np.full(shape, np.float32(1.0 + index * 0.001), dtype=np.float32)
    terrain = np.empty((nj, ni), dtype=np.float32)
    for j in range(nj):
        for i in range(ni):
            terrain[j, i] = np.float32(320.0 + 2.5 * i - 1.25 * j)
    result["terrain_msl"][:] = terrain
    zt = np.float32(200.0 * nk)
    result["gz"][:] = np.divide(
        zt, np.subtract(zt, terrain, dtype=np.float32), dtype=np.float32
    )
    result["rgz"][:] = np.divide(np.float32(1.0), result["gz"], dtype=np.float32)
    result["gzu"][:] = np.float32(1.02)
    result["rgzu"][:] = np.divide(np.float32(1.0), result["gzu"], dtype=np.float32)
    result["gzv"][:] = np.float32(1.025)
    result["rgzv"][:] = np.divide(np.float32(1.0), result["gzv"], dtype=np.float32)
    result["dzdx"][:] = np.float32(0.01)
    result["dzdy"][:] = np.float32(-0.0075)
    result["c1"][:] = np.float32(0.5)
    result["c2"][:] = np.float32(0.5)
    for k in range(nk):
        result["zh"][k] = terrain + np.float32((k + 0.5) * 200.0)
        density = np.float32(1.12 - 0.07 * k)
        result["rho0_3d"][k] = density
        result["rr0"][k] = np.float32(1.0) / density
        for j in range(nj):
            for i in range(ni + 1):
                result["ua"][k, j, i] = np.float32(
                    4.0 + 0.1 * i - 0.03 * j + 0.02 * k
                )
                result["u3d"][k, j, i] = result["ua"][k, j, i]
        for j in range(nj + 1):
            for i in range(ni):
                result["va"][k, j, i] = np.float32(
                    -2.0 + 0.04 * i + 0.08 * j - 0.01 * k
                )
                result["v3d"][k, j, i] = result["va"][k, j, i]
    for k in range(nk + 1):
        result["zf"][k] = terrain + np.float32(k * 200.0)
        density = np.float32(1.15 - 0.07 * k)
        result["rf0"][k] = density
        result["rrf0"][k] = np.float32(1.0) / density
        result["gx"][k] = np.float32(0.001 * (k + 1))
        result["gxu"][k] = np.float32(0.0012 * (k + 1))
        result["gy"][k] = np.float32(-0.0008 * (k + 1))
        result["gyv"][k] = np.float32(-0.0011 * (k + 1))
        for j in range(nj):
            for i in range(ni):
                result["wa"][k, j, i] = np.float32(
                    0.01 * np.sin(0.2 * i + 0.3 * j + 0.1 * k)
                )
                result["w3d"][k, j, i] = result["wa"][k, j, i]
    result["mh"][:] = result["gz"][None]
    result["rmh"][:] = np.divide(np.float32(1.0), result["mh"], dtype=np.float32)
    result["mf"][:] = result["gz"][None]
    result["rmf"][:] = np.divide(np.float32(1.0), result["mf"], dtype=np.float32)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                perturbation = np.float32(
                    2.0e-4 * np.sin(0.2 * i + 0.3 * j + 0.1 * k)
                )
                result["ppi"][k, j, i] = perturbation
                result["pp3d"][k, j, i] = perturbation
                result["ppx"][k, j, i] = perturbation
                result["th3d"][k, j, i] = np.float32(
                    0.5 * np.cos(0.1 * i + 0.2 * j + 0.05 * k)
                )
                result["q3d"][0, k, j, i] = np.float32(0.012 - 0.001 * k)
                result["q3d"][1, k, j, i] = np.float32(2.0e-4 + 1.0e-5 * i)
                result["q3d"][2, k, j, i] = np.float32(4.0e-5 + 2.0e-6 * j)
                result["q3d"][3, k, j, i] = np.float32(3.0e-5 + 1.0e-6 * k)
                result["q3d"][4, k, j, i] = np.float32(2.0e-5)
                result["q3d"][5, k, j, i] = np.float32(1.0e-5)
    assert all(value.flags.c_contiguous for value in result.values())
    assert len({int(value.ctypes.data) for value in result.values()}) == len(result)
    return result


def _clone(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy(order="C") for name, value in values.items()}


def test_import_is_cuda_cold_and_adapter_source_is_pinned() -> None:
    command = (
        "import sys; import cm1core.cuda_initialized_product_terrain_open as m; "
        "assert m.CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256; "
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
        hashlib.sha256(CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE.encode()).hexdigest()
        == CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256
    )
    assert "forecast" in CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT
    for relative, expected in AUTHORITY_SOURCE_SHA256.items():
        assert hashlib.sha256(
            (ROOT / "upstream/cm1-21.1/pristine" / relative).read_bytes()
        ).hexdigest() == expected


def test_cpu_product_pack_is_deterministic_source_ordered_and_zero_history() -> None:
    carriers = _carriers()
    first = cpu_pack_initialized_product_terrain_open(carriers, ni=5, nj=4, nk=5)
    second = cpu_pack_initialized_product_terrain_open(carriers, ni=5, nj=4, nk=5)
    comparison = compare_initialized_terrain_open_preparation_raw(
        first.arrays, second.arrays
    )
    assert comparison["passed"] is True
    assert comparison["max_ulp"] == 0
    assert comparison["null_or_missing_is_pass"] is False
    assert all(
        np.count_nonzero(first.arrays[name]) == 0
        for name in ("radbcw", "radbce", "radbcs", "radbcn")
    )
    assert all(np.isfinite(first.arrays[name]).all() for name in PREPARATION_COMPARE_NAMES)
    assert tuple(first.metrics) == TERRAIN_STATIC_REQUIRED
    assert np.count_nonzero(first.arrays["w3d"][1, NG : NG + 4, NG : NG + 5])


def test_eqtset2_preparation_controls_detect_input_species_location_and_output() -> None:
    carriers = _carriers()
    baseline = cpu_pack_initialized_product_terrain_open(carriers, ni=5, nj=4, nk=5)

    input_bit = _clone(carriers)
    word = input_bit["pi0"].view(np.uint32)
    word[2] ^= np.uint32(1)
    changed = cpu_pack_initialized_product_terrain_open(input_bit, ni=5, nj=4, nk=5)
    assert compare_initialized_terrain_open_preparation_raw(
        baseline.arrays, changed.arrays
    )["passed"] is False

    species = _clone(carriers)
    species["q3d"][[1, 3]] = species["q3d"][[3, 1]]
    changed = cpu_pack_initialized_product_terrain_open(species, ni=5, nj=4, nk=5)
    assert not np.array_equal(
        baseline.arrays["ppterm"].view(np.uint32),
        changed.arrays["ppterm"].view(np.uint32),
    )

    location = _clone(carriers)
    location["pp3d"][:, :, [1, 2]] = location["pp3d"][:, :, [2, 1]]
    changed = cpu_pack_initialized_product_terrain_open(location, ni=5, nj=4, nk=5)
    assert compare_initialized_terrain_open_preparation_raw(
        baseline.arrays, changed.arrays
    )["passed"] is False

    missing = dict(carriers)
    missing.pop("gx")
    with pytest.raises(ValueError, match="missing initialized-product carriers: gx"):
        cpu_pack_initialized_product_terrain_open(missing, ni=5, nj=4, nk=5)
    with pytest.raises(ValueError, match="missing initialized terrain/open field thv"):
        candidate = dict(baseline.arrays)
        candidate.pop("thv")
        compare_initialized_terrain_open_preparation_raw(baseline.arrays, candidate)


def test_cpu_product_pack_refuses_alias_and_wrong_species_shape() -> None:
    carriers = _carriers()
    carriers["u3d"] = carriers["ua"]
    with pytest.raises(ValueError, match="must not alias"):
        cpu_pack_initialized_product_terrain_open(carriers, ni=5, nj=4, nk=5)
    carriers = _carriers()
    carriers["q3d"] = carriers["q3d"][:5].copy(order="C")
    with pytest.raises(ValueError, match="carrier q3d must have"):
        cpu_pack_initialized_product_terrain_open(carriers, ni=5, nj=4, nk=5)


def test_real_crop_preparation_and_acoustic_match_cpu_only_with_mutex() -> None:
    if os.environ.get("CM1CORE_RUN_CUDA_INITIALIZED_TERRAIN_OPEN_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_INITIALIZED_TERRAIN_OPEN_TEST=1 only while "
            "holding the GPU mutex"
        )
    if not REAL_CROP.is_dir():
        pytest.skip("ignored corrected-v3 initialized-product crop is unavailable")
    cupy = pytest.importorskip("cupy")
    from cm1core.cuda_initialized_product import (
        initialize_cuda_initialized_product,
        load_initialized_product_export,
    )

    started = time.perf_counter()
    loaded = load_initialized_product_export(
        REAL_CROP,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_payload_sha256=PAYLOAD_SHA256,
        expected_source_artifact_sha256=SOURCE_ARTIFACT_SHA256,
        expected_source_contract_sha256=SOURCE_CONTRACT_SHA256,
    )
    product = initialize_cuda_initialized_product(loaded, allow_cropped_validation=True)
    state = prepare_cuda_initialized_product_terrain_open(
        product, allow_cropped_validation=True
    )
    state.ready_event.synchronize()
    host_carriers = {
        name: cupy.asnumpy(product.field(name))
        for name in ("pi0", "th0", "ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "ppx", "th3d", "q3d", *TERRAIN_STATIC_REQUIRED)
    }
    expected = cpu_pack_initialized_product_terrain_open(
        host_carriers, ni=16, nj=16, nk=75
    )
    prepared = {
        name: cupy.asnumpy(state.arrays[name]) for name in PREPARATION_COMPARE_NAMES
    }
    initial_radbc_zero = all(
        np.count_nonzero(prepared[name]) == 0
        for name in ("radbcw", "radbce", "radbcs", "radbcn")
    )
    assert initial_radbc_zero
    preparation_comparison = compare_initialized_terrain_open_preparation_raw(
        expected.arrays, prepared
    )
    assert preparation_comparison["passed"] is True

    cpu_arrays = _clone(expected.arrays)
    cpu_metrics = _clone(expected.metrics)
    cpu_result = cpu_psolver3_terrain_open_three_stage_inplace(
        cpu_arrays,
        cpu_metrics,
        ni=16,
        nj=16,
        nk=75,
        dt=0.6,
        nsound=6,
        dx=100.0,
        dy=100.0,
        dz=200.0,
    )
    launch = launch_cuda_psolver3_terrain_open_three_stage(
        state.arrays,
        state.metrics,
        ni=16,
        nj=16,
        nk=75,
        dt=0.6,
        nsound=6,
        dx=100.0,
        dy=100.0,
        dz=200.0,
        stream=product.stream,
        terrain_compiled=state.terrain_compiled,
        open_compiled=state.open_compiled,
    )
    launch.ready_event.synchronize()
    candidate = {
        name: cupy.asnumpy(state.arrays[name])
        for name in COMPARE_OUTPUT_NAMES
    }
    final_radbc_nonzero = {
        name: int(np.count_nonzero(candidate[name]))
        for name in ("radbcw", "radbce", "radbcs", "radbcn")
    }
    assert sum(final_radbc_nonzero.values()) > 0
    acoustic_comparison = compare_psolver3_terrain_open_raw(cpu_arrays, candidate)
    assert acoustic_comparison["passed"] is True
    evidence_path = os.environ.get("CM1CORE_TERRAIN_OPEN_REAL_EVIDENCE_PATH")
    if evidence_path:
        payload = {
            "acoustic_comparison": acoustic_comparison,
            "cadence": launch.cadence,
            "claim_limit": CM1_INITIALIZED_TERRAIN_OPEN_CLAIM_LIMIT,
            "comparison_contract": {
                "max_ulp": 0,
                "mode": "raw-bit",
                "null_or_missing_is_pass": False,
            },
            "domain": {"ni": 16, "nj": 16, "nk": 75, "ngxy": 3},
            "elapsed_seconds": time.perf_counter() - started,
            "invariants": {
                "all_outputs_finite": all(
                    np.isfinite(candidate[name]).all() for name in COMPARE_OUTPUT_NAMES
                ),
                "initial_radiation_histories_exact_zero": initial_radbc_zero,
                "final_radiation_nonzero_words": final_radbc_nonzero,
                "maximum_cpu_tridiagonal_residual": max(
                    item.max_tridiagonal_residual for item in cpu_result.cadence
                ),
            },
            "kernel": {
                "adapter_cubin_sha256": hashlib.sha256(state.compiled.cubin).hexdigest(),
                "adapter_ptx_sha256": hashlib.sha256(state.compiled.ptx).hexdigest(),
                "adapter_source_sha256": CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256,
                "acoustic_cubin_sha256": hashlib.sha256(launch.compiled.cubin).hexdigest(),
                "acoustic_ptx_sha256": hashlib.sha256(launch.compiled.ptx).hexdigest(),
                "acoustic_source_sha256": launch.receipt["kernel"]["source_sha256"],
                "target_arch": launch.compiled.target_arch,
            },
            "preparation_comparison": preparation_comparison,
            "schema": "gpuwm-cm1.cuda-psolver3-terrain-open-real-crop/v1",
            "source": {
                "artifact_sha256": SOURCE_ARTIFACT_SHA256,
                "contract_sha256": SOURCE_CONTRACT_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "payload_sha256": PAYLOAD_SHA256,
            },
            "source_anchors": state.receipt["source_anchors"],
            "status": "passed-preparation-and-acoustic-raw-bit-exact",
        }
        Path(evidence_path).write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
