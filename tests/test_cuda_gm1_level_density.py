"""The face density at the ground and the model top, with the column's own weights.

CM1 r21.1 extrapolates the density to the lowest and the highest w level with
``cgs1..3`` and ``cgt1..3`` (``src/solve3.F:678-679``), which
``src/param.F:8985-9060`` computes from the levels.  The sealed kernel carries
the equal-spacing numbers 1.875, -1.25 and 0.375.  These tests hold the level
variant to three things: it is the sealed text with the six weights as launch
arguments and nothing else; handed the equal-spacing numbers it is the sealed
operator word for word; and a column whose ends are equally spaced never
reaches it.
"""

from __future__ import annotations

import difflib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_gm1_forecast as sealed
from cm1core import cuda_gm1_level_density as level
from cm1core.cm1_vertical_grid import (
    PINNED_FORNBERG_F32,
    build_vertical_levels,
    uniform_vertical_levels,
)

ROOT = Path(__file__).resolve().parents[1]
F32 = np.float32
requires_host_compiler = pytest.mark.skipif(shutil.which("g++") is None, reason="requires g++")


def _column(**words):
    return build_vertical_levels(stretch_z=1, dz_m=words.pop("dz_m", 500.0), **words)


def _hurricane_column():
    """CM1's stock hurricane column: 59 levels that stretch from the ground."""

    return _column(nk=59, ztop_m=25000.0, str_bot_m=0.0, str_top_m=5500.0, dz_bot_m=50.0, dz_top_m=500.0)


def test_the_level_kernel_is_the_sealed_text_with_the_weights_as_arguments() -> None:
    assert level.source_pin_is_valid()
    diff = [
        line for line in difflib.unified_diff(
            sealed.GM1_FORECAST_CUDA_SOURCE.splitlines(), level.GM1_LEVEL_DENSITY_CUDA_SOURCE.splitlines(),
            lineterm="", n=0,
        )
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert len(diff) == 9
    removed = [line[1:] for line in diff if line.startswith("-")]
    added = [line[1:] for line in diff if line.startswith("+")]
    assert removed[0] == "// gm1.cuda-real-forecast-kernels/v1" and added[0] == f"// {level.GM1_LEVEL_DENSITY_ABI}"
    assert added[2] == " float cgs1,float cgs2,float cgs3,float cgt1,float cgt2,float cgt3){"
    # The two extrapolations, with each literal replaced by its argument and nothing else moved.
    for old, new, names in ((removed[2], added[3], ("cgs1", "cgs2", "cgs3")),
                            (removed[3], added[4], ("cgt1", "cgt2", "cgt3"))):
        assert old.replace("1.875f", names[0]).replace("-1.25f", names[1]).replace("0.375f", names[2]) == new
    assert "1.875f" not in "".join(added)
    # Every other kernel of the image is the sealed text.
    assert level.GM1_LEVEL_DENSITY_CUDA_SOURCE.count('extern "C" __global__') == len(sealed.GM1_FORECAST_KERNEL_NAMES)


def test_a_column_with_equally_deep_end_layers_never_reaches_the_level_kernel(monkeypatch) -> None:
    sentinel = SimpleNamespace(kernels=("k0", "k1", "k2", "k3"), module_owner=object())
    monkeypatch.setattr(level, "compile_cuda_gm1_forecast", lambda: sentinel)
    monkeypatch.setattr(level, "_compile_level_density_kernel", lambda: pytest.fail("compiled the level kernel"))
    columns = [
        None,
        uniform_vertical_levels(40, 500.0),
        # dz with no exact binary32 spelling: CM1's own recomputed cgt differs in the last places,
        # and the grid still runs the sealed numbers, as it always has.
        uniform_vertical_levels(40, 333.3),
        # the stretched column of the moist comparison (three 100 m layers at the ground)
        _column(nk=53, ztop_m=19300.0, str_bot_m=300.0, str_top_m=9300.0, dz_bot_m=100.0, dz_top_m=500.0),
        # the dry oracle's stretched column and the tornado-grade one
        _column(nk=40, dz_m=200.0, ztop_m=8000.0, str_bot_m=1000.0, str_top_m=5000.0, dz_bot_m=100.0, dz_top_m=300.0),
        _column(nk=75, dz_m=100.0, ztop_m=15000.0, str_bot_m=200.0, str_top_m=7000.0, dz_bot_m=20.0, dz_top_m=320.0),
    ]
    for levels in columns:
        assert level.boundary_density_weights(levels) is None
        assert level.compile_cuda_gm1_forecast_for_levels(levels) is sentinel
    # A caller hands over its own compile function, and gets what that returns.
    own = SimpleNamespace(kernels=(), module_owner=None)
    assert level.compile_cuda_gm1_forecast_for_levels(columns[1], compile_sealed=lambda: own) is own


def test_each_end_takes_its_own_weights_only_when_its_layers_are_not_equally_deep() -> None:
    ground_only = _hurricane_column()
    ground, top = level.boundary_density_weights(ground_only)
    assert ground == ground_only.cgs and ground != PINNED_FORNBERG_F32
    assert top == PINNED_FORNBERG_F32
    assert ground_only.spacings_m[:3] == pytest.approx((50.0, 73.684, 97.368), abs=1e-3)
    # Fornberg's value weights interpolate: they sum to one.
    assert abs(sum(ground) - 1.0) < 1e-6

    both = _column(nk=40, dz_m=200.0, ztop_m=8000.0, str_bot_m=0.0, str_top_m=8000.0, dz_bot_m=100.0, dz_top_m=300.0)
    ground, top = level.boundary_density_weights(both)
    assert ground == both.cgs != PINNED_FORNBERG_F32
    assert top == both.cgt != PINNED_FORNBERG_F32

    top_only = _column(nk=51, dz_m=100.0, ztop_m=7320.0, str_bot_m=200.0, str_top_m=7000.0, dz_bot_m=20.0, dz_top_m=320.0)
    ground, top = level.boundary_density_weights(top_only)
    assert ground == PINNED_FORNBERG_F32 and top == top_only.cgt != PINNED_FORNBERG_F32


def test_the_level_image_replaces_the_density_kernel_and_keeps_the_others(monkeypatch) -> None:
    sealed_image = sealed.CompiledGm1ForecastKernels("owner", ("k0", "k1", "k2", "k3"), b"", b"", "89", ())
    calls = []
    monkeypatch.setattr(level, "compile_cuda_gm1_forecast", lambda: sealed_image)
    monkeypatch.setattr(level, "_compile_level_density_kernel",
                        lambda: ("module", lambda grid, block, args, **kw: calls.append((grid, block, args, kw))))
    levels = _hurricane_column()
    image = level.compile_cuda_gm1_forecast_for_levels(levels)
    assert image is not sealed_image
    assert image.kernels[0] == "k0" and image.kernels[2:] == ("k2", "k3")
    assert image.module_owner == ("owner", "module")
    kernel = image.kernels[level.DENSITY_KERNEL_INDEX]
    assert isinstance(kernel, level.LevelDensityKernel)
    # Launched with the sealed kernel's own eight arguments; the six weights follow, as binary32.
    kernel((1, 1, 1), (128, 1, 1), tuple(range(8)), stream="s")
    grid, block, args, keywords = calls[0]
    assert args[:8] == tuple(range(8)) and keywords == {"stream": "s"}
    assert all(type(value) is np.float32 for value in args[8:])
    assert tuple(float(v) for v in args[8:]) == (*levels.cgs, *PINNED_FORNBERG_F32)
    record = kernel.record()
    assert record["source_sha256"] == level.GM1_LEVEL_DENSITY_CUDA_SOURCE_SHA256
    assert record["ground_weights"] == list(levels.cgs)
    json.dumps(record, allow_nan=False)


def _fixture(nk: int = 9, nj: int = 5, ni: int = 6):
    rng = np.random.default_rng(20261012)
    rho = (1.2 * np.exp(-np.arange(nk)[:, None, None] / 7.0)
           * (1.0 + 0.01 * rng.standard_normal((nk, nj, ni)))).astype(F32)
    c1 = rng.uniform(0.3, 0.7, (nk + 1, nj, ni)).astype(F32)
    c2 = (F32(1.0) - c1).astype(F32)
    return rho, c1, c2


def _launch(library, rho, c1, c2, extra=()):
    from tools.oracle.cuda_kernel_host_emulator import launch_grid

    nk, nj, ni = rho.shape
    rr = np.zeros((nk, nj, ni), F32)
    rf = np.zeros((nk + 1, nj, ni), F32)
    grid, block = launch_grid(ni * nj * (nk + 1), 128)
    library.kernel("gm1_solve3_density_v1")(
        grid, block, (rho, c1, c2, rr, rf, np.uint32(ni), np.uint32(nj), np.uint32(nk)) + extra)
    return rr, rf


@requires_host_compiler
def test_handed_the_equal_spacing_numbers_it_is_the_sealed_operator_word_for_word() -> None:
    from tools.oracle.cuda_kernel_host_emulator import HostKernelLibrary

    rho, c1, c2 = _fixture()
    left = _launch(HostKernelLibrary.compile(sealed.GM1_FORECAST_CUDA_SOURCE), rho, c1, c2)
    right = _launch(HostKernelLibrary.compile(level.GM1_LEVEL_DENSITY_CUDA_SOURCE), rho, c1, c2,
                    tuple(F32(v) for v in PINNED_FORNBERG_F32 * 2))
    for a, b in zip(left, right):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@requires_host_compiler
def test_with_the_columns_weights_it_is_cm1s_extrapolation_at_the_two_faces_only() -> None:
    from tools.oracle.cuda_kernel_host_emulator import HostKernelLibrary

    rho, c1, c2 = _fixture()
    nk = rho.shape[0]
    levels = _column(nk=40, dz_m=200.0, ztop_m=8000.0, str_bot_m=0.0, str_top_m=8000.0, dz_bot_m=100.0, dz_top_m=300.0)
    ground, top = level.boundary_density_weights(levels)
    sealed_rr, sealed_rf = _launch(HostKernelLibrary.compile(sealed.GM1_FORECAST_CUDA_SOURCE), rho, c1, c2)
    rr, rf = _launch(HostKernelLibrary.compile(level.GM1_LEVEL_DENSITY_CUDA_SOURCE), rho, c1, c2,
                     tuple(F32(v) for v in (*ground, *top)))
    want_rr, want_rf = level.cpu_level_density(rho, c1, c2, ground, top)
    assert np.array_equal(rr.view(np.uint32), want_rr.view(np.uint32))
    assert np.array_equal(rf.view(np.uint32), want_rf.view(np.uint32))
    # solve3.F:678-679 in plain binary32, written out for the ground face.
    g = [F32(v) for v in ground]
    by_hand = ((g[0] * rho[0]).astype(F32) + (g[1] * rho[1]).astype(F32)).astype(F32) + (g[2] * rho[2]).astype(F32)
    assert np.array_equal(rf[0].view(np.uint32), by_hand.astype(F32).view(np.uint32))
    # Only the two end faces move; the interior faces and rr are the sealed kernel's words.
    assert np.array_equal(rr.view(np.uint32), sealed_rr.view(np.uint32))
    assert np.array_equal(rf[1:nk].view(np.uint32), sealed_rf[1:nk].view(np.uint32))
    assert not np.array_equal(rf[0], sealed_rf[0]) and not np.array_equal(rf[nk], sealed_rf[nk])


def test_the_weights_are_the_ones_cm1_printed_for_the_same_column() -> None:
    """CM1 r21.1 prints cgs and cgt in its log; the evidence keeps the lines of its runs."""

    record = ROOT / "evidence" / "oracle" / "stretched-ground-supercell" / "cm1-weights.json"
    if not record.is_file():
        pytest.skip("the development evidence is not distributed")
    for row in json.loads(record.read_text())["columns"]:
        levels = _column(**row["column"])
        for name in ("cgs", "cgt"):
            printed = row["cm1_log"][name]
            ours = getattr(levels, name)
            # CM1 prints nine significant digits, which names one binary32 word.
            assert [float(F32(v)) for v in printed] == [float(F32(v)) for v in ours], (row["name"], name)


def _device():
    """CuPy with a CUDA device, or skip: the CPU suite has none."""

    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - exercised only without a driver
        available = False
    if not available:
        pytest.skip("no CUDA device")
    return cupy


def test_on_a_device_the_level_image_is_the_sealed_one_at_the_equal_spacing_numbers() -> None:
    cupy = _device()
    rho, c1, c2 = _fixture(nk=12, nj=7, ni=9)
    nk, nj, ni = rho.shape
    words = ni * nj * (nk + 1)
    grid, block = ((words + 127) // 128, 1, 1), (128, 1, 1)

    def launch(kernel):
        rr = cupy.zeros((nk, nj, ni), cupy.float32)
        rf = cupy.zeros((nk + 1, nj, ni), cupy.float32)
        kernel(grid, block, (cupy.asarray(rho), cupy.asarray(c1), cupy.asarray(c2), rr, rf,
                             cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)))
        return cupy.asnumpy(rr), cupy.asnumpy(rf)

    sealed_image = sealed.compile_cuda_gm1_forecast()
    sealed_rr, sealed_rf = launch(sealed_image.kernels[level.DENSITY_KERNEL_INDEX])
    _module, raw = level._compile_level_density_kernel()
    same_rr, same_rf = launch(level.LevelDensityKernel(raw, PINNED_FORNBERG_F32, PINNED_FORNBERG_F32))
    assert np.array_equal(sealed_rr.view(np.uint32), same_rr.view(np.uint32))
    assert np.array_equal(sealed_rf.view(np.uint32), same_rf.view(np.uint32))

    levels = _column(nk=40, dz_m=200.0, ztop_m=8000.0, str_bot_m=0.0, str_top_m=8000.0, dz_bot_m=100.0, dz_top_m=300.0)
    image = level.compile_cuda_gm1_forecast_for_levels(levels)
    assert isinstance(image.kernels[level.DENSITY_KERNEL_INDEX], level.LevelDensityKernel)
    rr, rf = launch(image.kernels[level.DENSITY_KERNEL_INDEX])
    want_rr, want_rf = level.cpu_level_density(rho, c1, c2, levels.cgs, levels.cgt)
    assert np.array_equal(rr.view(np.uint32), want_rr.view(np.uint32))
    assert np.array_equal(rf.view(np.uint32), want_rf.view(np.uint32))
    assert np.array_equal(rf[1:nk].view(np.uint32), sealed_rf[1:nk].view(np.uint32))
    # An equally spaced column gets the sealed image: its density kernel is the sealed one, unwrapped.
    even = level.compile_cuda_gm1_forecast_for_levels(uniform_vertical_levels(40, 500.0))
    assert not isinstance(even.kernels[level.DENSITY_KERNEL_INDEX], level.LevelDensityKernel)
    even_rr, even_rf = launch(even.kernels[level.DENSITY_KERNEL_INDEX])
    assert np.array_equal(even_rf.view(np.uint32), sealed_rf.view(np.uint32))
    assert np.array_equal(even_rr.view(np.uint32), sealed_rr.view(np.uint32))


# ---------------------------------------------------------------- the measured evidence, where it is present

EVIDENCE = ROOT / "evidence" / "oracle"
NINE = (
    "adaptive-timestep-32", "beta-wide-512-beta0", "beta-wide-512", "coriolis-adaptive-32",
    "coriolis-fplane-32-icor0-alph050", "coriolis-fplane-32-icor0", "coriolis-fplane-32",
    "coriolis-fplane-moving-32-icor0", "coriolis-fplane-moving-32",
)


def _evidence(relative: str) -> dict:
    record = EVIDENCE / relative
    if not record.is_file():
        pytest.skip("the development evidence is not distributed")
    return json.loads(record.read_text(encoding="utf-8"))


def test_existing_grids_published_the_same_words_on_the_device() -> None:
    """The nine decks of the dynamics receipt and four more, release 0.1.2 against this change."""

    identity = _evidence("dynamics-cm1-words/stretched-levels-word-identity-rtx4090.json")
    rows = {row["case"]: row for row in identity["runs"]}
    assert set(NINE) <= set(rows)
    assert identity["result"]["decks_identical"] is True
    for case in (*NINE, "acoustic-warm-bubble-32"):
        row = rows[case]
        assert row["identical"] and row["base"]["frame_arena_sha256"] == row["head"]["frame_arena_sha256"], case
    assert sum(len(rows[case]["base"]["frame_arena_sha256"]) for case in NINE) == 102
    # On the stretched decks one carrier moves, the precipitable water that was wrong there; nothing else does.
    assert identity["carriers_expected_to_differ"] == ["cwp", "lwp", "pwat"]
    assert identity["result"]["no_other_carrier_differs"] is True
    for case in ("stretched-column-32", "explicit-w-levels-32", "explicit-scalar-levels-32"):
        assert rows[case]["carriers_that_differ"] == ["pwat"], case
    storms = _evidence("stretched-ground-supercell/existing-storms-word-identity.json")
    assert storms["result"] == {"frames_compared": 34, "head_equals_base_every_frame": True}


@pytest.mark.parametrize("case", ["stretched-ground-32", "stretched-both-ends-32", "stretched-ground-drag-32"])
def test_the_dry_comparisons_on_unequal_end_layers_agree_with_cm1(case: str) -> None:
    receipt = _evidence(f"{case}/receipt.json")
    assert receipt["cm1"]["terminated_normally"] is True
    assert receipt["gm1"]["status"] == "executed-cartesian-integration-to-requested-duration"
    measured = receipt["measured_max_abs_over_peak"]
    # The magnitudes the column with equal end layers reaches (evidence/oracle/stretched-column-32).
    for field, limit in (("prs", 2.0e-6), ("rho", 2.0e-6), ("u3d", 1.0e-5), ("v3d", 3.0e-5), ("th3d", 1.0e-3),
                         ("q3d:qv", 2.0e-6), ("pwat", 1.0e-6)):
        assert measured[field]["t20"] <= limit and measured[field]["t100"] <= limit, (case, field)
    # CM1's printed end weights for this column are the generator's words.
    column = {"stretched-both-ends-32": dict(nk=40, str_top_m=8000.0)}.get(case, dict(nk=35, str_top_m=5000.0))
    levels = _column(dz_m=200.0, ztop_m=8000.0, str_bot_m=0.0, dz_bot_m=100.0, dz_top_m=300.0, **column)
    assert [float(F32(v)) for v in receipt["cm1"]["cgs"]] == [float(F32(v)) for v in levels.cgs]
    assert [float(F32(v)) for v in receipt["cm1"]["cgt"]] == [float(F32(v)) for v in levels.cgt]


def test_the_supercells_on_cm1s_hurricane_column_say_what_was_measured() -> None:
    """With drag every statistic is inside its band; with free slip one is not, and the record says which."""

    drag = _evidence("stretched-ground-supercell/drag.json")["verdicts"]["gm1"]
    assert drag["all_inside_band"] is True
    free = _evidence("stretched-ground-supercell/free-slip.json")
    verdict = free["verdicts"]["gm1"]
    outside = [key for key, row in verdict["rows"].items() if not row["inside_band"]]
    assert outside == ["zvort_max"] and verdict["all_inside_band"] is False
    assert all(abs(row["ratio"] - 1.0) <= 0.025 for key, row in verdict["rows"].items() if key != "zvort_max")
    assert verdict["max_w_column"]["inside"] and verdict["onset"]["inside"]
    # Under free slip the ground weights reach no statistic: the control with the equal-spacing numbers is the same run.
    control = free["verdicts"]["gm1_equal_spacing_weights"]
    assert {k: row["ratio"] for k, row in control["rows"].items()} == {k: row["ratio"] for k, row in verdict["rows"].items()}
    # The same excess on a column with three equal layers at the ground, which ran before this change.
    equal = _evidence("stretched-ground-supercell/three-equal-layers.json")["verdicts"]["gm1"]
    assert [key for key, row in equal["rows"].items() if not row["inside_band"]] == ["zvort_max"]
