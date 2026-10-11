"""The pre-WSM6 positive-definite repair, spread across the device.

CM1's ``pdefq`` with ``pdscheme=1`` (src/misclibs.F:1037-1117) is a per-column
repair with two ordered carries wrapped around it.  The transcription this port
shipped first gave one thread to each physical row ``j`` and let that thread
walk ``i`` and ``k`` itself, so its parallelism was ``nj`` alone: a 448-row
domain resolved the whole repair -- 1.5 GB of moving state -- to four thread
blocks, whatever ``ni`` and ``nk`` were.

These tests pin the restructure and, more importantly, pin why it is allowed.
The only things ``pdefq`` orders are misclibs.F:1065-1081's per-column ``k``
accumulations, misclibs.F:1082's left-associated west-to-east ``budj`` carry
and misclibs.F:1086-1088's south-to-north ``asq`` sum.  The ``i`` loop carries
nothing but ``budj``, so the columns themselves are independent and may be
evaluated in any order.  That claim is checked here on the host, raw word for
raw word, over column orders the authority never uses -- no device required.
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_door import build_runtime_config
from cm1core.config import load_config, parse_mapping
from cm1core.cuda_wsm6_pdefq import (
    CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES,
    CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256,
    CUDA_WSM6_PDEFQ_KERNEL_NAMES,
    CUDA_WSM6_PDEFQ_SOURCE_SHA256,
    DEFAULT_WSM6_PDEFQ_LAYOUT,
    WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT,
    WSM6_PDEFQ_LAYOUTS,
    WSM6_PDEFQ_ROW_SERIAL_LAYOUT,
    cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace,
    cpu_wsm6_pdefq_pdscheme1_inplace,
    require_wsm6_pdefq_layout,
    wsm6_pdefq_column_order,
    wsm6_pdefq_launch_plan,
    wsm6_pdefq_workspace_bytes,
)
from cm1core.errors import CM1Refusal
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"
GEOMETRY = {"ni": 7, "nj": 5, "nk": 4, "dx": 50.0, "dy": 50.0, "dz": 20.0}


def _fixture() -> dict[str, np.ndarray]:
    """The module's own repair fixture, including its problem words."""

    from test_cuda_wsm6_pdefq import _fixture as build

    return build()


def _raw(value: np.ndarray) -> np.ndarray:
    return value.view(np.uint32 if value.dtype == np.float32 else np.uint64)


# --------------------------------------------------------------------------
# What the launch actually asks the device for
# --------------------------------------------------------------------------


def test_the_pinned_layout_leaves_the_device_idle_and_the_new_one_does_not() -> None:
    """The defect, stated as arithmetic rather than as a benchmark.

    ``row-serial``'s block count is a function of ``nj`` alone, so widening or
    deepening the domain buys no parallelism at all: the same four blocks carry
    28x more work at 448x448x160 than at 448x448x160 with nk=1.  That is the
    fingerprint the phase map measured -- a cost that scales as ``ni*nk`` and
    is independent of ``nj``.
    """

    tall = wsm6_pdefq_launch_plan(ni=448, nj=448, nk=160, layout=WSM6_PDEFQ_ROW_SERIAL_LAYOUT)
    shallow = wsm6_pdefq_launch_plan(ni=448, nj=448, nk=1, layout=WSM6_PDEFQ_ROW_SERIAL_LAYOUT)
    wide = wsm6_pdefq_launch_plan(ni=4096, nj=448, nk=160, layout=WSM6_PDEFQ_ROW_SERIAL_LAYOUT)
    assert tall["blocks_per_species"] == shallow["blocks_per_species"] == wide["blocks_per_species"] == 4
    assert tall["threads_per_species"] == 512

    spread = wsm6_pdefq_launch_plan(ni=448, nj=448, nk=160)
    assert spread["layout"] == WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT == DEFAULT_WSM6_PDEFQ_LAYOUT
    assert spread["blocks_per_species"] == 4 * 448 == 1792
    assert spread["independent_columns"] == 448 * 448
    # One thread per independent column, rounded up to the block width.
    assert spread["threads_per_species"] >= spread["independent_columns"]


def test_each_layout_names_the_kernels_it_launches_and_shares_the_finish() -> None:
    row = wsm6_pdefq_launch_plan(ni=64, nj=48, nk=32, layout=WSM6_PDEFQ_ROW_SERIAL_LAYOUT)
    column = wsm6_pdefq_launch_plan(ni=64, nj=48, nk=32, layout=WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT)
    assert [stage["kernel"] for stage in row["stages"]] == [
        CUDA_WSM6_PDEFQ_KERNEL_NAMES[0],
        CUDA_WSM6_PDEFQ_KERNEL_NAMES[1],
    ]
    assert [stage["kernel"] for stage in column["stages"]] == [
        CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES[0],
        CUDA_WSM6_PDEFQ_COLUMN_KERNEL_NAMES[1],
        CUDA_WSM6_PDEFQ_KERNEL_NAMES[1],
    ]
    # misclibs.F:1086-1088 is one serial sum in both layouts, and it is
    # literally the same compiled kernel.
    assert row["stages"][-1] == column["stages"][-1]
    assert row["stages"][-1]["grid"] == [1, 1, 1]
    # The scan that replaces misclibs.F:1082's in-thread carry stays serial
    # over i: one thread per row, exactly as the authority walks it.
    assert column["stages"][1]["grid"] == [(48 + 127) // 128, 1, 1]


def test_the_pinned_transcription_is_still_byte_identical() -> None:
    """Every recorded receipt names this source by hash; it did not move."""

    assert CUDA_WSM6_PDEFQ_SOURCE_SHA256 == (
        "38fb779397c513a4ceb5288e1519188b11315fea636208430c4189bb4dde7522"
    )
    assert CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256 != CUDA_WSM6_PDEFQ_SOURCE_SHA256


def test_the_column_workspace_is_priced_without_a_device() -> None:
    row = wsm6_pdefq_workspace_bytes(ni=448, nj=448, nk=160, layout=WSM6_PDEFQ_ROW_SERIAL_LAYOUT)
    column = wsm6_pdefq_workspace_bytes(ni=448, nj=448, nk=160)
    assert row == 6 * 448 * 8
    assert column == row + 2 * 448 * 448 * 8
    # The state the repair itself moves at this size, for scale.
    assert column < 6 * 160 * 448 * 448 * 4 // 200


# --------------------------------------------------------------------------
# Why the restructure is allowed: the reordering is raw-word identical
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 6, 8, 11, 13, 17, 34])
def test_evaluating_the_columns_in_any_order_is_raw_word_identical(stride: int) -> None:
    """misclibs.F:1065-1081 carries nothing from one column to the next.

    This is the whole bit-exactness argument, executed rather than asserted.
    The source-order oracle and a run that visits the same columns in an order
    the authority never uses must agree on every repaired ``q3d`` word, every
    ``asq`` ledger word, every ``budj`` row and every intermediate ``(a1,a2)``
    pair -- raw word for raw word, not to a tolerance.
    """

    ordered = _fixture()
    reference = cpu_wsm6_pdefq_pdscheme1_inplace(ordered, **GEOMETRY)

    shuffled = _fixture()
    order = wsm6_pdefq_column_order(ni=GEOMETRY["ni"], nj=GEOMETRY["nj"], stride=stride)
    actual = cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace(
        shuffled, column_order=order, **GEOMETRY
    )

    assert actual["column_order_is_source_order"] == (stride == 1)
    assert np.array_equal(_raw(shuffled["q3d"]), _raw(ordered["q3d"]))
    assert np.array_equal(_raw(shuffled["asq"]), _raw(ordered["asq"]))
    assert np.array_equal(
        _raw(actual["budget_columns"]), _raw(reference["budget_columns"])
    )
    assert np.array_equal(_raw(actual["column_pairs"]), _raw(reference["column_pairs"]))
    assert actual["corrected_input_words"] == reference["corrected_input_words"] == 10


def test_the_default_column_order_is_not_the_authority_order() -> None:
    """A test that only ever ran the source order would prove nothing."""

    shuffled = _fixture()
    actual = cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace(shuffled, **GEOMETRY)
    assert actual["column_order_is_source_order"] is False
    ordered = _fixture()
    cpu_wsm6_pdefq_pdscheme1_inplace(ordered, **GEOMETRY)
    assert np.array_equal(_raw(shuffled["q3d"]), _raw(ordered["q3d"]))
    assert np.array_equal(_raw(shuffled["asq"]), _raw(ordered["asq"]))


def test_the_west_to_east_carry_is_a_recurrence_not_a_sum() -> None:
    """misclibs.F:1082 is ``(budj+a2)-a1``, and that is not the same function.

    Fortran parses ``budj(j)=budj(j)+a2-a1`` left-associated, so the carry is a
    two-step recurrence over ``i`` rather than a sum of per-column differences.
    The two formulas are different float64 computations -- demonstrated here on
    explicit words -- which is why the restructure keeps the carry serial
    instead of reducing it.  The second half checks that the implementation
    really does compute the authority's one.
    """

    a1 = (0.0, 1.0e300)
    a2 = (1.0, 1.0e300)
    carried = 0.0
    for one, two in zip(a1, a2, strict=True):
        carried = (carried + two) - one
    regrouped = 0.0
    for one, two in zip(a1, a2, strict=True):
        regrouped = regrouped + (two - one)
    assert carried == 0.0
    assert regrouped == 1.0

    ordered = _fixture()
    reference = cpu_wsm6_pdefq_pdscheme1_inplace(ordered, **GEOMETRY)
    pairs = reference["column_pairs"]
    rebuilt = np.empty_like(reference["budget_columns"])
    for species in range(6):
        for j in range(GEOMETRY["nj"]):
            budj = 0.0
            for i in range(GEOMETRY["ni"]):
                budj = (budj + pairs[species, 1, i, j]) - pairs[species, 0, i, j]
            rebuilt[species, j] = budj
    assert np.array_equal(_raw(rebuilt), _raw(reference["budget_columns"]))


def test_column_order_must_be_a_permutation() -> None:
    with pytest.raises(ValueError, match=r"every \(i,j\) column"):
        cpu_wsm6_pdefq_pdscheme1_column_parallel_inplace(
            _fixture(), column_order=[(0, 0)] * (GEOMETRY["ni"] * GEOMETRY["nj"]), **GEOMETRY
        )
    with pytest.raises(ValueError, match="coprime"):
        wsm6_pdefq_column_order(ni=4, nj=4, stride=2)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_unadmitted_layout_refuses_by_key_and_value_before_cuda_import() -> None:
    before = {name for name in sys.modules if name == "cupy" or name.startswith("cupy.")}
    with pytest.raises(CM1Refusal) as caught:
        require_wsm6_pdefq_layout("one-thread")
    detail = caught.value.detail
    assert detail.code == "ARWEN-WSM6-PDEFQ-LAYOUT"
    assert detail.option == "pdefq_launch"
    assert detail.value == "one-thread"
    assert "misclibs.F:1082" in detail.reason
    assert "trailing bits" in detail.reason
    assert {name for name in sys.modules if name == "cupy" or name.startswith("cupy.")} == before


def test_a_domain_taller_than_the_grid_limit_refuses_and_names_the_wrong_result() -> None:
    with pytest.raises(CM1Refusal) as caught:
        wsm6_pdefq_launch_plan(ni=8, nj=65536, nk=8)
    detail = caught.value.detail
    assert detail.code == "ARWEN-WSM6-PDEFQ-ROWS"
    assert detail.option == "nj"
    assert detail.value == 65536
    assert "unrepaired" in detail.reason
    assert WSM6_PDEFQ_ROW_SERIAL_LAYOUT in detail.remedy
    # The refusal must not fire on anything this port can actually run.
    assert wsm6_pdefq_launch_plan(ni=8, nj=65535, nk=8)["blocks_per_species"] > 0


def test_every_admitted_layout_plans_without_a_device() -> None:
    for layout in WSM6_PDEFQ_LAYOUTS:
        plan = wsm6_pdefq_launch_plan(ni=9, nj=11, nk=13, layout=layout)
        assert plan["layout"] == layout
        assert plan["authority"] == "src/misclibs.F:1037-1117"


# --------------------------------------------------------------------------
# Configuration text to changed behaviour
# --------------------------------------------------------------------------


def _cartesian_mapping() -> dict[str, object]:
    payload = load_config(EXAMPLE).canonical_dict()
    return {
        section: copy.deepcopy(payload[section])
        for section in (
            "run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime"
        )
    }


def test_declaring_the_layout_in_the_document_changes_the_launched_grid() -> None:
    """From TOML text to the grid the device is actually asked for."""

    plans = {}
    for declared in (None, WSM6_PDEFQ_ROW_SERIAL_LAYOUT, WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT):
        raw = _cartesian_mapping()
        if declared is not None:
            raw["runtime"]["pdefq_launch"] = declared  # type: ignore[index]
        runtime_config = build_runtime_config(parse_mapping(raw))
        record = runtime_config.record()
        plans[declared] = (runtime_config.pdefq_launch, record["pdefq_launch_plan"])
        assert record["pdefq_launch"] == runtime_config.pdefq_launch

    assert plans[None][0] == DEFAULT_WSM6_PDEFQ_LAYOUT
    assert plans[None] == plans[WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT]
    row_blocks = plans[WSM6_PDEFQ_ROW_SERIAL_LAYOUT][1]["blocks_per_species"]
    column_blocks = plans[WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT][1]["blocks_per_species"]
    assert column_blocks > row_blocks
    grid = load_config(EXAMPLE).grid
    assert row_blocks == (grid.ny + 127) // 128
    assert column_blocks == ((grid.nx + 127) // 128) * grid.ny


def test_an_undeclared_layout_leaves_the_sealed_document_identity_alone() -> None:
    """The existing default configuration is bit-identical, by hash."""

    config = load_config(EXAMPLE)
    assert config.runtime is not None
    assert config.runtime.pdefq_launch is None
    assert "pdefq_launch" not in config.canonical_dict()["runtime"]  # type: ignore[operator]
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )
    declared = _cartesian_mapping()
    declared["runtime"]["pdefq_launch"] = DEFAULT_WSM6_PDEFQ_LAYOUT  # type: ignore[index]
    # Declaring it is a different document even when it selects the default:
    # the seal describes the bytes, not the behaviour.
    assert parse_mapping(declared).sha256 != config.sha256
    assert build_runtime_config(parse_mapping(declared)).pdefq_launch == (
        build_runtime_config(config).pdefq_launch
    )


def test_the_document_refuses_an_unadmitted_layout_by_key_and_value() -> None:
    raw = _cartesian_mapping()
    raw["runtime"]["pdefq_launch"] = "whatever-is-fastest"  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert caught.value.detail.option == "runtime.pdefq_launch"
    assert caught.value.detail.value == "whatever-is-fastest"
    assert "misclibs.F:1082" in caught.value.detail.reason

    raw = _cartesian_mapping()
    raw["runtime"]["pdefq_launch"] = 4  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-STRING"


def test_the_runtime_contract_refuses_an_unadmitted_layout_on_its_own() -> None:
    grid = CartesianGridGeometry(16, 16, 16, 1000.0, 1000.0, 500.0)
    assert RuntimeConfig(grid).pdefq_launch == DEFAULT_WSM6_PDEFQ_LAYOUT
    with pytest.raises(CM1Refusal) as caught:
        RuntimeConfig(grid, pdefq_launch="fastest")
    assert caught.value.detail.code == "ARWEN-WSM6-PDEFQ-LAYOUT"


# --------------------------------------------------------------------------
# The call site
# --------------------------------------------------------------------------


def test_the_cartesian_step_hands_the_configured_layout_to_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last link: what the document says reaches the kernel launch.

    A capability the configuration can name but no lane can select is this
    port's dominant failure mode, so this drives ``CartesianFinish.pdefq``
    itself rather than re-reading the configuration.
    """

    from cm1core import cuda_cartesian_finish as finish_module

    raw = _cartesian_mapping()
    raw["runtime"]["pdefq_launch"] = WSM6_PDEFQ_ROW_SERIAL_LAYOUT  # type: ignore[index]
    runtime_config = build_runtime_config(parse_mapping(raw))

    seen: dict[str, object] = {}

    def fake_launch(arrays, **kwargs):
        seen.update(kwargs)
        seen["arrays"] = arrays
        return SimpleNamespace(
            receipt={"schema": "test", "launch": {"layout": kwargs["layout"]}},
            ready_event=SimpleNamespace(synchronize=lambda: None),
        )

    monkeypatch.setattr(finish_module, "launch_cuda_wsm6_pdefq_pdscheme1", fake_launch)

    grid = runtime_config.grid
    step = object.__new__(finish_module.CartesianFinish)
    step.config = runtime_config
    step.grid = grid
    step.stream = object()
    step.pdefq_compiled = object()
    step.pdefq_workspace = object()
    step.step_receipts = {}
    step.asq = object()
    # 4b3f004: pdefq repairs only the profile's contiguous mass prefix.
    step.repaired_scalars = 6
    step.eos_invalid = SimpleNamespace(get=lambda stream: [0])
    step.dynamics = SimpleNamespace(
        # sliced to the repaired prefix since 4b3f004, so it must be indexable
        fields={"q3d": np.zeros((6, 1, 1, 1), np.float32), "rho": object()},
        arrays={"uh": np.zeros(grid.nx + 6), "vh": np.zeros(grid.ny + 6)},
        metrics={"rmh": object()},
    )

    step.pdefq()

    assert seen["layout"] == WSM6_PDEFQ_ROW_SERIAL_LAYOUT
    assert seen["workspace"] is step.pdefq_workspace
    assert seen["compiled"] is step.pdefq_compiled
    assert (seen["ni"], seen["nj"], seen["nk"]) == (grid.nx, grid.ny, grid.nz)
    assert step.step_receipts["pdefq"]["launch"]["layout"] == WSM6_PDEFQ_ROW_SERIAL_LAYOUT
    assert step.eos_invalid is None


# --------------------------------------------------------------------------
# The packaged kernel text, executed on the host
# --------------------------------------------------------------------------


def test_the_packaged_kernel_text_is_raw_word_exact_on_the_host() -> None:
    """Compile and run the shipped CUDA source itself, without a device.

    The ``_rn`` intrinsics the kernels use are exact IEEE operations, so with a
    shim that maps each one onto the identical host operation and with FP
    contraction turned off, the packaged kernel text runs under a C++ compiler.
    That checks the text that is actually shipped -- both layouts, in forward
    and reverse thread order, at three block widths -- against the Python
    source-order oracle, raw word for raw word.  NVRTC code generation and real
    device scheduling are out of its reach and stay with the mutexed test below.
    """

    if shutil.which("g++") is None:
        pytest.skip("no host C++ compiler is installed")
    sys.path.insert(0, str(ROOT / "tools"))
    from oracle.wsm6_pdefq_layout_oracle import verify

    receipt = verify()
    assert receipt["status"] == "passed-host-raw-word-exact-both-layouts"
    assert receipt["corrected_input_words"] == 10
    assert receipt["kernel_sources"][WSM6_PDEFQ_ROW_SERIAL_LAYOUT] == (
        CUDA_WSM6_PDEFQ_SOURCE_SHA256
    )
    assert receipt["kernel_sources"][WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT] == (
        CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256
    )
    assert "-ffp-contract=off" in receipt["compile_flags"]
    comparisons = receipt["comparisons"]
    assert len(comparisons) == 12
    assert {item["layout"] for item in comparisons} == set(WSM6_PDEFQ_LAYOUTS)
    assert {item["thread_order"] for item in comparisons} == {"forward", "reverse"}
    for item in comparisons:
        assert all(item["raw_word_exact"].values()), item
    # The intermediate the restructure introduces is checked too, and only the
    # layout that has one reports it.
    assert all(
        ("column_pairs" in item["raw_word_exact"])
        == (item["layout"] == WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT)
        for item in comparisons
    )


# --------------------------------------------------------------------------
# The device itself
# --------------------------------------------------------------------------


def test_both_cuda_layouts_are_raw_word_exact_only_with_mutex() -> None:
    """Raw-word equality on the device, against the CPU oracle and each other."""

    if os.environ.get("CM1CORE_RUN_CUDA_WSM6_PDEFQ_TEST") != "1":
        pytest.skip("set CM1CORE_RUN_CUDA_WSM6_PDEFQ_TEST=1 only while holding GPU mutex")
    import cupy

    from cm1core.cuda_wsm6_pdefq import (
        compile_cuda_wsm6_pdefq,
        create_cuda_wsm6_pdefq_workspace,
        launch_cuda_wsm6_pdefq_pdscheme1,
    )

    expected = _fixture()
    oracle = cpu_wsm6_pdefq_pdscheme1_inplace(expected, **GEOMETRY)
    compiled = compile_cuda_wsm6_pdefq()

    results: dict[str, dict[str, object]] = {}
    for layout in WSM6_PDEFQ_LAYOUTS:
        host = _fixture()
        device = {name: cupy.asarray(value) for name, value in host.items()}
        workspace = create_cuda_wsm6_pdefq_workspace(
            ni=GEOMETRY["ni"], nj=GEOMETRY["nj"], nk=GEOMETRY["nk"], layout=layout
        )
        launch = launch_cuda_wsm6_pdefq_pdscheme1(
            device, layout=layout, compiled=compiled, workspace=workspace, **GEOMETRY
        )
        launch.ready_event.synchronize()
        results[layout] = {
            "q3d": cupy.asnumpy(device["q3d"]),
            "asq": cupy.asnumpy(device["asq"]),
            "budgets": cupy.asnumpy(workspace.budget_columns),
            "pairs": (
                None
                if workspace.column_pairs is None
                else cupy.asnumpy(workspace.column_pairs)
            ),
            "receipt": launch.receipt,
        }

    for layout, actual in results.items():
        assert np.array_equal(_raw(actual["q3d"]), _raw(expected["q3d"])), layout
        assert np.array_equal(_raw(actual["asq"]), _raw(expected["asq"])), layout
        assert np.array_equal(
            _raw(actual["budgets"]), _raw(oracle["budget_columns"])
        ), layout

    row = results[WSM6_PDEFQ_ROW_SERIAL_LAYOUT]
    column = results[WSM6_PDEFQ_COLUMN_PARALLEL_LAYOUT]
    for name in ("q3d", "asq", "budgets"):
        assert np.array_equal(_raw(column[name]), _raw(row[name])), name

    # The intermediate the new layout adds is the authority's own (a1,a2); the
    # workspace is reused across species, so it holds the last one.
    assert row["pairs"] is None
    assert np.array_equal(
        _raw(column["pairs"]), _raw(oracle["column_pairs"][len(oracle["source_order"]) - 1])
    )

    assert row["receipt"]["launch"]["blocks_per_species"] == 1
    assert column["receipt"]["launch"]["blocks_per_species"] == GEOMETRY["nj"]
    assert column["receipt"]["layout_kernel"]["source_sha256"] == (
        CUDA_WSM6_PDEFQ_COLUMN_SOURCE_SHA256
    )
    assert row["receipt"]["kernel"]["source_sha256"] == CUDA_WSM6_PDEFQ_SOURCE_SHA256
