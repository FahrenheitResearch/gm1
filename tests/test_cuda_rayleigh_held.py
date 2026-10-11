from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from cm1core.errors import CM1Refusal
from cm1core.cuda_rayleigh_held import (
    ALL_ARRAY_NAMES,
    AUTHORITY_SELECTION,
    CUDA_RAYLEIGH_HELD_SOURCE_BYTES,
    CUDA_RAYLEIGH_HELD_SOURCE_SHA256,
    DEFAULT_RAYLEIGH_HELD_SELECTION,
    OUTPUT_NAMES,
    RayleighHeldSelection,
    cpu_build_rayleigh_coefficients,
    cpu_rayleigh_held_inplace,
    launch_cuda_rayleigh_held,
    rayleigh_held_shapes,
    validate_rotation_large_scale_disabled,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "evidence/oracles/rayleigh-held-real-v1"
NI, NJ, NK = 4, 4, 8


def _take(
    raw: np.ndarray, offset: int, shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape))
    return raw[offset : offset + count].reshape(shape).copy(order="C"), offset + count


def _oracle_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw_input = np.fromfile(ORACLE / "corrected-real-input.f32le", dtype="<f4")
    offset = 0
    inputs: dict[str, np.ndarray] = {}
    for name, shape in (
        ("zh_halo", (NK, NJ + 2, NI + 2)),
        ("zf_halo", (NK + 1, NJ + 2, NI + 2)),
        ("u0", (NK,)),
        ("v0", (NK,)),
        ("ua", (NK, NJ, NI + 1)),
        ("va", (NK, NJ + 1, NI)),
        ("wa", (NK, NJ, NI)),
        ("tha", (NK, NJ, NI)),
    ):
        inputs[name], offset = _take(raw_input, offset, shape)
    assert offset == raw_input.size

    raw_output = np.fromfile(ORACLE / "native-output.f32le", dtype="<f4")
    offset = 0
    outputs: dict[str, np.ndarray] = {}
    for name, shape in (
        ("tauh", (NK, NJ + 2, NI + 2)),
        ("taus", (NK, NJ + 2, NI + 2)),
        ("tauf", (NK + 1, NJ + 2, NI + 2)),
        ("uten1", (NK, NJ, NI + 1)),
        ("vten1", (NK, NJ + 1, NI)),
        ("wten1", (NK, NJ, NI)),
        ("thten1", (NK, NJ, NI)),
    ):
        outputs[name], offset = _take(raw_output, offset, shape)
    assert offset == raw_output.size
    return inputs, outputs


def _operator_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    inputs, native = _oracle_arrays()
    shapes = rayleigh_held_shapes(NI, NJ, NK)
    arrays = {
        name: inputs[name].copy(order="C")
        for name in ("u0", "v0", "ua", "va", "wa", "tha")
    }
    arrays.update(
        {name: native[name].copy(order="C") for name in ("tauh", "taus", "tauf")}
    )
    arrays.update(
        {name: np.empty(shapes[name], dtype=np.float32) for name in OUTPUT_NAMES}
    )
    return arrays, native


def _ordered(bits: np.ndarray) -> np.ndarray:
    return np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000))


def test_cold_import_does_not_load_numpy_or_cupy() -> None:
    code = (
        "import sys; import cm1core.cuda_rayleigh_held; "
        "assert 'numpy' not in sys.modules; assert 'cupy' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_cuda_source_pin_is_exact() -> None:
    assert hashlib.sha256(CUDA_RAYLEIGH_HELD_SOURCE_BYTES).hexdigest() == (
        CUDA_RAYLEIGH_HELD_SOURCE_SHA256
    )


def test_shapes_are_exact_staggered_physical_domains() -> None:
    shapes = rayleigh_held_shapes(300, 300, 75)
    assert shapes["tauh"] == (75, 302, 302)
    assert shapes["tauf"] == (76, 302, 302)
    assert shapes["ua"] == shapes["uten1"] == (75, 300, 301)
    assert shapes["va"] == shapes["vten1"] == (75, 301, 300)
    assert shapes["wa"] == shapes["wten1"] == (75, 300, 300)
    assert shapes["tha"] == shapes["thten1"] == (75, 300, 300)


def test_authority_rotation_and_large_scale_selection_is_disabled() -> None:
    validate_rotation_large_scale_disabled(dict(AUTHORITY_SELECTION))
    for name in AUTHORITY_SELECTION:
        changed = dict(AUTHORITY_SELECTION)
        value = changed[name]
        changed[name] = not value if isinstance(value, bool) else value + 1
        with pytest.raises(CM1Refusal, match="CM1-ROTATION-LARGE-SCALE-SELECTION"):
            validate_rotation_large_scale_disabled(changed)


def test_rayleigh_selection_refuses_drift() -> None:
    assert DEFAULT_RAYLEIGH_HELD_SELECTION.damping_start_m == 12000.0
    arrays, _ = _operator_arrays()
    with pytest.raises(CM1Refusal, match="CM1-RAYLEIGH-HELD-SELECTION"):
        cpu_rayleigh_held_inplace(
            arrays,
            ni=NI,
            nj=NJ,
            nk=NK,
            selection=RayleighHeldSelection(damping_rate_s_inv=0.004),
        )


def test_cpu_source_formula_coefficients_are_bounded_to_two_ulp() -> None:
    inputs, native = _oracle_arrays()
    result = cpu_build_rayleigh_coefficients(inputs["zh_halo"], inputs["zf_halo"])
    for name in ("tauh", "taus", "tauf"):
        actual = getattr(result, name)
        expected = native[name]
        distance = np.abs(
            _ordered(actual.view(np.uint32)).astype(np.int64)
            - _ordered(expected.view(np.uint32)).astype(np.int64)
        )
        assert int(distance.max(initial=0)) <= 2
    assert "cross-libm raw-bit identity is not claimed" in result.receipt["libm_claim"]


def test_cpu_held_assignment_matches_corrected_real_native_oracle_raw_bits() -> None:
    arrays, native = _operator_arrays()
    cpu_rayleigh_held_inplace(arrays, ni=NI, nj=NJ, nk=NK)
    for name in OUTPUT_NAMES:
        assert np.array_equal(
            arrays[name].view(np.uint32), native[name].view(np.uint32)
        )
        assert np.count_nonzero(arrays[name]) > 0


def test_cpu_requires_complete_nonaliased_f32_contract() -> None:
    arrays, _ = _operator_arrays()
    del arrays["tauh"]
    with pytest.raises(ValueError, match="missing Rayleigh-held arrays"):
        cpu_rayleigh_held_inplace(arrays, ni=NI, nj=NJ, nk=NK)


def test_committed_full_domain_evidence_is_complete_and_hash_bound() -> None:
    directory = ROOT / "evidence/validation/real-rayleigh-held-v1"
    receipt = json.loads((directory / "full-domain-sm86.json").read_bytes())
    audit = json.loads((directory / "configuration-audit.json").read_bytes())
    assert receipt["result"] == "pass"
    assert receipt["domain"] == {
        "nx": 300,
        "ny": 300,
        "nz": 75,
        "spacing_m": [100.0, 100.0, 200.0],
    }
    assert all(
        receipt["comparison"][name]["mismatch_words"] == 0 for name in OUTPUT_NAMES
    )
    assert all(
        receipt["statistics"][name]["nonzero_words"] > 0 for name in OUTPUT_NAMES
    )
    assert audit["authority"]["selected_values"]["icor"] == 0
    assert audit["authority"]["selected_values"]["lspgrad"] == 0
    assert audit["decision"]["coriolis_kernel"].startswith("not implemented")
    for figure in receipt["figures"]:
        path = directory / figure["path"]
        assert path.stat().st_size == figure["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == figure["sha256"]
    arrays, _ = _operator_arrays()
    arrays["uten1"] = arrays["ua"]
    with pytest.raises(ValueError, match="must not alias"):
        cpu_rayleigh_held_inplace(arrays, ni=NI, nj=NJ, nk=NK)


@pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="set CM1_RUN_GPU_TESTS=1 while holding the canonical GPU mutex",
)
def test_cuda_matches_corrected_real_native_oracle_raw_bits() -> None:
    cupy = importlib.import_module("cupy")
    arrays, native = _operator_arrays()
    device = {name: cupy.asarray(value) for name, value in arrays.items()}
    launch = launch_cuda_rayleigh_held(device, ni=NI, nj=NJ, nk=NK)
    launch.ready_event.synchronize()
    for name in OUTPUT_NAMES:
        actual = cupy.asnumpy(device[name])
        assert np.array_equal(actual.view(np.uint32), native[name].view(np.uint32))
    assert launch.receipt["comparison_contract"]["max_ulp"] == 0
    assert launch.receipt["selection"]["icor"] == 0
    assert launch.receipt["selection"]["lspgrad"] == 0
    assert set(device) == set(ALL_ARRAY_NAMES)
