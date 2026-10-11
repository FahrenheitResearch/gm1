from __future__ import annotations

import copy
import hashlib
import os

import numpy as np
import pytest

from cm1core.cuda_sound3_column import (
    ALL_ARRAY_NAMES as COLUMN_ARRAY_NAMES,
    cpu_sound3_three_stage_inplace,
)
from cm1core.cuda_sound3_periodic import (
    ALL_ARRAY_NAMES,
    CUDA_SOUND3_PERIODIC_SOURCE_BYTES,
    CUDA_SOUND3_PERIODIC_SOURCE_SHA256,
    array_shapes,
    cpu_sound3_periodic_three_stage_inplace,
    default_periodic_face_weights,
    fill_periodic_scalar_inplace,
    fill_periodic_uv_inplace,
    launch_cuda_sound3_periodic_three_stage,
    validate_periodic_contract,
)


NI = 5
NJ = 4
NK = 7
PARAMETERS = {
    "ni": NI,
    "nj": NJ,
    "nk": NK,
    "dt": 2.0,
    "nsound": 6,
    "rdx": 1.0 / 1000.0,
    "rdy": 1.0 / 1000.0,
    "rdz": 1.0 / 500.0,
}
OUTPUT_NAMES = (
    "u3d",
    "v3d",
    "w3d",
    "pp3d",
    "ppx",
    "ppd",
    "rru",
    "rrv",
    "rrw",
    "fpk",
    "divh",
    "hadvp",
    "tri_pivot",
    "tri_residual",
)


def make_fixture(*, horizontally_varying: bool = True) -> dict[str, np.ndarray]:
    shapes = array_shapes(NI, NJ, NK)
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    arrays["dtu0"], arrays["dtv0"] = default_periodic_face_weights(NI, NJ)
    for name in ("uf", "vf", "uh", "vh"):
        arrays[name].fill(np.float32(1.0))

    x_phase = np.arange(NI, dtype=np.float32) * np.float32(2.0 * np.pi / NI)
    y_phase = np.arange(NJ, dtype=np.float32) * np.float32(2.0 * np.pi / NJ)
    horizontal = np.zeros((NJ, NI), dtype=np.float32)
    if horizontally_varying:
        horizontal = np.float32(0.7) * np.sin(x_phase)[None, :]
        horizontal = np.float32(horizontal + np.float32(0.4) * np.cos(y_phase)[:, None])

    for k in range(NK + 2):
        z = np.float32(k) / np.float32(NK + 1)
        for j0 in range(NJ):
            j = j0 + 1
            for i0 in range(NI):
                i = i0 + 1
                h = horizontal[j0, i0]
                arrays["mh"][k, j, i] = np.float32(1.0)
                arrays["rr0"][k, j, i] = np.float32(1.0 + np.float32(0.01) * z)
                arrays["rf0"][k, j, i] = np.float32(0.99 + np.float32(0.012) * z)
                arrays["c1"][k, j, i] = np.float32(0.45)
                arrays["c2"][k, j, i] = np.float32(0.55)
                arrays["mf"][k, j, i] = np.float32(1.0)
                arrays["piadv"][k, j, i] = np.float32(
                    1.0 + np.float32(0.0003) * z + np.float32(0.00001) * h
                )
                arrays["ppterm"][k, j, i] = np.float32(0.014 + np.float32(0.0002) * z)
                arrays["thv"][k, j, i] = np.float32(
                    299.0 + np.float32(1.8) * z + np.float32(0.03) * h
                )
                arrays["ppten"][k, j, i] = np.float32(
                    np.float32(2.0e-6) * h + np.float32(1.0e-6) * z
                )
                pressure = np.float32(np.float32(2.0e-4) * h + np.float32(4.0e-5) * z)
                arrays["ppi"][k, j, i] = pressure
                arrays["ppx"][k, j, i] = np.float32(
                    np.float32(0.2) * pressure + np.float32(1.0e-6) * z
                )
    for name in (
        "mh",
        "rr0",
        "rf0",
        "c1",
        "c2",
        "mf",
        "piadv",
        "ppterm",
        "thv",
        "ppten",
        "ppi",
        "ppx",
    ):
        fill_periodic_scalar_inplace(arrays[name], ni=NI, nj=NJ, nk=NK)

    for k in range(1, NK + 1):
        z = np.float32(k) / np.float32(NK)
        for j0 in range(NJ):
            for i0 in range(NI):
                h = horizontal[j0, i0]
                arrays["wa"][k, j0, i0] = np.float32(
                    np.float32(0.003) * h * np.sin(np.float32(np.pi) * z)
                )
                arrays["wten"][k, j0, i0] = np.float32(np.float32(2.0e-5) * h)
        for j in range(1, NJ + 1):
            y = np.float32(2.0 * np.pi * (j - 1) / NJ)
            for face in range(NI + 1):
                x = np.float32(2.0 * np.pi * face / NI)
                value = np.float32(4.0 + np.float32(0.15) * np.sin(x) * np.cos(y))
                arrays["ua"][k, j, face] = value
                arrays["uten"][k, j, face] = np.float32(
                    np.float32(1.0e-4) * np.cos(x) * np.cos(y)
                )
        for face in range(NJ + 1):
            y = np.float32(2.0 * np.pi * face / NJ)
            for i in range(1, NI + 1):
                x = np.float32(2.0 * np.pi * (i - 1) / NI)
                value = np.float32(-1.5 + np.float32(0.12) * np.cos(x) * np.sin(y))
                arrays["va"][k, face, i] = value
                arrays["vten"][k, face, i] = np.float32(
                    -np.float32(8.0e-5) * np.sin(x) * np.cos(y)
                )
    if not horizontally_varying:
        arrays["ua"][1 : NK + 1, 1 : NJ + 1, :] = np.float32(4.0)
        arrays["uten"].fill(np.float32(0.0))
        arrays["va"][1 : NK + 1, :, 1 : NI + 1] = np.float32(-1.5)
        arrays["vten"].fill(np.float32(0.0))
    fill_periodic_uv_inplace(arrays["ua"], arrays["va"], ni=NI, nj=NJ, nk=NK)
    np.copyto(arrays["u3d"], arrays["ua"])
    np.copyto(arrays["v3d"], arrays["va"])
    np.copyto(arrays["w3d"], arrays["wa"])
    np.copyto(arrays["pp3d"], arrays["ppi"])
    np.copyto(arrays["ppd"], arrays["ppx"])
    validate_periodic_contract(arrays, ni=NI, nj=NJ, nk=NK)
    return arrays


def digest_outputs(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in OUTPUT_NAMES:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(arrays[name].tobytes(order="C"))
    return digest.hexdigest()


def digest_trajectory(arrays: dict[str, np.ndarray], result: object) -> str:
    digest = hashlib.sha256()
    digest.update(digest_outputs(arrays).encode("ascii"))
    for stage in result.cadence:
        digest.update(stage.boundary_trace_sha256.encode("ascii"))
    return digest.hexdigest()


def to_column_fixture(periodic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ncol = NI * NJ
    column = {
        name: np.zeros((ncol, NK + 2), dtype=np.float32) for name in COLUMN_ARRAY_NAMES
    }
    scalar_inputs = (
        "mh",
        "rr0",
        "rf0",
        "c1",
        "c2",
        "mf",
        "piadv",
        "ppterm",
        "thv",
        "ppten",
        "ppi",
        "pp3d",
        "ppx",
        "ppd",
    )
    w_inputs = ("wten", "wa", "w3d", "rrw")
    for j0 in range(NJ):
        for i0 in range(NI):
            col = j0 * NI + i0
            for name in scalar_inputs:
                column[name][col] = periodic[name][:, j0 + 1, i0 + 1]
            for name in w_inputs:
                column[name][col] = periodic[name][:, j0, i0]
    return column


def test_source_pin_and_periodic_contract() -> None:
    assert hashlib.sha256(CUDA_SOUND3_PERIODIC_SOURCE_BYTES).hexdigest() == (
        CUDA_SOUND3_PERIODIC_SOURCE_SHA256
    )
    arrays = make_fixture()
    validate_periodic_contract(arrays, ni=NI, nj=NJ, nk=NK)
    assert np.all(arrays["dtu0"][1 : NJ + 1, :] == np.float32(1.0))
    assert not np.any(arrays["dtu0"][[0, NJ + 1], :])
    assert np.all(arrays["dtv0"][:, 1 : NI + 1] == np.float32(1.0))
    assert not np.any(arrays["dtv0"][:, [0, NI + 1]])
    broken = {name: value.copy() for name, value in arrays.items()}
    broken["ppx"][2, 0, 2] = np.nextafter(
        broken["ppx"][2, 0, 2], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ValueError, match="south periodic halo"):
        validate_periodic_contract(broken, ni=NI, nj=NJ, nk=NK)


def test_cpu_horizontal_transition_cadence_and_invariants() -> None:
    arrays = make_fixture()
    initial_u = arrays["u3d"].copy()
    initial_v = arrays["v3d"].copy()
    result = cpu_sound3_periodic_three_stage_inplace(arrays, **PARAMETERS)
    assert [(item.nrk, item.nloop) for item in result.cadence] == [
        (1, 2),
        (2, 3),
        (3, 6),
    ]
    assert [item.ppd_boundary_fills for item in result.cadence] == [1, 2, 5]
    assert [item.stage_pressure_boundary_fills for item in result.cadence] == [1, 1, 0]
    assert not np.array_equal(arrays["u3d"].view(np.uint32), initial_u.view(np.uint32))
    assert not np.array_equal(arrays["v3d"].view(np.uint32), initial_v.view(np.uint32))
    assert np.any(arrays["divh"][1 : NK + 1, 1 : NJ + 1, 1 : NI + 1] != 0)
    assert np.any(arrays["hadvp"][1 : NK + 1, 1 : NJ + 1, 1 : NI + 1] != 0)
    residual = arrays["tri_residual"][2 : NK + 1, 1 : NJ + 1, 1 : NI + 1]
    assert float(np.max(np.abs(residual))) <= 2.0e-7
    pivots = arrays["tri_pivot"][2 : NK + 1, 1 : NJ + 1, 1 : NI + 1]
    assert float(np.min(np.abs(pivots))) > 0.9
    assert np.array_equal(
        arrays["u3d"][1 : NK + 1, 1 : NJ + 1, 0].view(np.uint32),
        arrays["u3d"][1 : NK + 1, 1 : NJ + 1, NI].view(np.uint32),
    )
    assert np.array_equal(
        arrays["v3d"][1 : NK + 1, 0, 1 : NI + 1].view(np.uint32),
        arrays["v3d"][1 : NK + 1, NJ, 1 : NI + 1].view(np.uint32),
    )
    divergence_sum = np.sum(
        arrays["divh"][1 : NK + 1, 1 : NJ + 1, 1 : NI + 1],
        axis=(1, 2),
        dtype=np.float64,
    )
    assert float(np.max(np.abs(divergence_sum))) < 2.0e-8


def test_horizontally_uniform_limit_is_column_oracle() -> None:
    periodic = make_fixture(horizontally_varying=False)
    column = to_column_fixture(periodic)
    cpu_sound3_periodic_three_stage_inplace(periodic, **PARAMETERS)
    cpu_sound3_three_stage_inplace(
        column,
        ncol=NI * NJ,
        nk=NK,
        dt=PARAMETERS["dt"],
        nsound=PARAMETERS["nsound"],
        rdz=PARAMETERS["rdz"],
    )
    for j0 in range(NJ):
        for i0 in range(NI):
            col = j0 * NI + i0
            for name in ("w3d", "rrw"):
                assert np.array_equal(
                    periodic[name][:, j0, i0].view(np.uint32),
                    column[name][col].view(np.uint32),
                ), name
            for name in (
                "pp3d",
                "ppx",
                "ppd",
                "qk",
                "pk1",
                "pk2",
                "mm",
                "fpk",
                "elim_c",
                "elim_d",
                "tri_aa",
                "tri_bb",
                "tri_cc",
                "tri_dd",
                "tri_pivot",
                "tri_residual",
            ):
                assert np.array_equal(
                    periodic[name][:, j0 + 1, i0 + 1].view(np.uint32),
                    column[name][col].view(np.uint32),
                ), name
    assert not np.any(periodic["divh"][1 : NK + 1, 1 : NJ + 1, 1 : NI + 1])
    assert not np.any(periodic["hadvp"][1 : NK + 1, 1 : NJ + 1, 1 : NI + 1])


def test_four_defect_classes_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = make_fixture()
    baseline_result = cpu_sound3_periodic_three_stage_inplace(baseline, **PARAMETERS)
    expected = digest_trajectory(baseline, baseline_result)

    input_bit = make_fixture()
    word = input_bit["ppx"].view(np.uint32)
    word[3, 2, 2] ^= np.uint32(1 << 16)
    fill_periodic_scalar_inplace(input_bit["ppx"], ni=NI, nj=NJ, nk=NK)
    input_bit_result = cpu_sound3_periodic_three_stage_inplace(input_bit, **PARAMETERS)

    location = make_fixture()
    live = location["ppx"][:, 1 : NJ + 1, 1 : NI + 1].copy()
    location["ppx"][:, 1 : NJ + 1, 1 : NI + 1] = np.roll(live, 1, axis=2)
    fill_periodic_scalar_inplace(location["ppx"], ni=NI, nj=NJ, nk=NK)
    location_result = cpu_sound3_periodic_three_stage_inplace(location, **PARAMETERS)

    import cm1core.cuda_sound3_periodic as implementation

    boundary_cadence = make_fixture()
    original_fill = implementation.fill_periodic_scalar_inplace
    calls = 0

    def skip_first_interstep_fill(value: np.ndarray, **kwargs: int) -> None:
        nonlocal calls
        calls += 1
        if calls != 1:
            original_fill(value, **kwargs)

    monkeypatch.setattr(
        implementation, "fill_periodic_scalar_inplace", skip_first_interstep_fill
    )
    boundary_result = cpu_sound3_periodic_three_stage_inplace(
        boundary_cadence, **PARAMETERS
    )
    monkeypatch.setattr(implementation, "fill_periodic_scalar_inplace", original_fill)

    cadence = make_fixture()
    altered = dict(PARAMETERS)
    altered["nsound"] = 8
    cadence_result = cpu_sound3_periodic_three_stage_inplace(cadence, **altered)

    output = copy.deepcopy(baseline)
    output["pp3d"].view(np.uint32)[3, 2, 2] ^= np.uint32(1)

    controls = {
        "input-bit": digest_trajectory(input_bit, input_bit_result),
        "location": digest_trajectory(location, location_result),
        "boundary-cadence": digest_trajectory(boundary_cadence, boundary_result),
        "acoustic-cadence": digest_trajectory(cadence, cadence_result),
        "output": digest_trajectory(output, baseline_result),
    }
    assert all(value != expected for value in controls.values())


@pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1", reason="bounded GPU gate is opt-in"
)
def test_sm86_raw_bit_parity() -> None:
    cupy = pytest.importorskip("cupy")
    expected = make_fixture()
    actual = {name: cupy.asarray(value) for name, value in make_fixture().items()}
    cpu_sound3_periodic_three_stage_inplace(expected, **PARAMETERS)
    launch = launch_cuda_sound3_periodic_three_stage(actual, **PARAMETERS)
    launch.ready_event.synchronize()
    assert launch.compiled.target_arch == "86"
    for name in OUTPUT_NAMES:
        observed = cupy.asnumpy(actual[name])
        assert np.array_equal(
            observed.view(np.uint32), expected[name].view(np.uint32)
        ), name


def test_fixture_contains_every_operator_array() -> None:
    assert set(make_fixture()) == set(ALL_ARRAY_NAMES)
