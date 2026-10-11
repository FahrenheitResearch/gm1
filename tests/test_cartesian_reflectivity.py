"""Simulated radar reflectivity, from the TOML key to the changed behaviour.

CM1's own word for this is ``output_dbz`` (``README.namelist:1366-1369``): it
writes the three-dimensional ``dbz`` field (``writeout.F:4089-4107``) and the
composite -- column-maximum -- reflectivity ``cref`` beside it
(``writeout.F:1081-1091`` from ``misclibs.F:2259-2292``), on the steps
``cm1.F:3214-3237`` raises ``getdbz`` for, and it is refused by ``ptype`` in
``param.F:1583-1603`` because the formula belongs to the scheme.

Every test here runs on the host.  No CUDA context is opened: the device
launches are replaced by host stand-ins that record what they were asked for,
so what is proven is the path -- the key, the parse, the runtime contract, the
schedule, the component argument, the scatter, the reduction, the carrier and
the receipt -- and not the arithmetic of a kernel, which has its own oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_cartesian_finish as finish_module
from cm1core.cartesian_door import build_runtime_config
from cm1core.cartesian_history import (
    cartesian_history_profile,
    reflectivity_envelope_cost,
)
from cm1core.config import load_config
from cm1core.cuda_cartesian_finish import CartesianFinish, cartesian_wsm6_scratch_reserve
from cm1core.cuda_composite_reflectivity import (
    CM1_CREF_SEED_DBZ,
    cpu_composite_reflectivity,
    cpu_reflectivity_level,
)
from cm1core.errors import CM1Refusal
from cm1core.gm1_runtime import RuntimeConfig, reflectivity_level_index
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BASE_DOCUMENT = (ROOT / "examples/cartesian-uniform-grid.toml").read_text(encoding="utf-8")


def _document(tmp_path: Path, name: str, extra: str = "") -> Path:
    path = tmp_path / name
    path.write_text(BASE_DOCUMENT + extra, encoding="utf-8")
    return path


# --------------------------------------------------------------- host stand-ins


class _Stream:
    def __init__(self) -> None:
        self.syncs = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def synchronize(self) -> None:
        self.syncs += 1


class _Registry:
    """Only the allocate/release contract the physics loop actually uses."""

    def __init__(self) -> None:
        self.buffers: dict[str, np.ndarray] = {}
        self.registered_allocation_bytes = 0

    def allocate(self, name, shape, *, zero=False, dtype=None):
        assert name not in self.buffers, name
        value = (np.zeros if zero else np.empty)(
            shape, dtype=np.float32 if dtype is None else dtype
        )
        self.buffers[name] = value
        self.registered_allocation_bytes += value.nbytes
        return value

    def release(self, names) -> None:
        for name in names:
            self.buffers.pop(name)


_CP = SimpleNamespace(
    copyto=np.copyto, multiply=np.multiply, float32=np.float32,
    float64=np.float64, uint32=np.uint32, amax=np.amax,
    asnumpy=np.asarray, count_nonzero=np.count_nonzero,
)


def _finish_shell(runtime_config: RuntimeConfig) -> CartesianFinish:
    """A CartesianFinish with host storage and no compiled kernel anywhere."""

    g = runtime_config.grid
    instance = CartesianFinish.__new__(CartesianFinish)
    instance.config = runtime_config
    instance.grid = g
    instance.cp = _CP
    instance.stream = _Stream()
    instance.registry = _Registry()
    instance.source_identity_sha256 = "a" * 64
    instance.capacity_profile = None
    instance.post_compiled = None
    instance.step_receipts = {"pdefq": {"token": "b" * 64}}
    instance.completed_physics_step = 0
    instance.completed_solve3_step = 0
    instance.persistence = {}
    instance.output_dbz = bool(runtime_config.output_dbz)
    instance.reflectivity_level = runtime_config.reflectivity_level_index
    instance.reflectivity = {}
    instance.reflectivity_compiled = None
    instance.completed_reflectivity_step = 0
    # CartesianFinish.__init__ now reads the component spec and the q3d
    # profile from the configuration (4b3f004, 46eb419), and the runtime
    # always binds a reflectivity sink before stepping
    # (cuda_cartesian_runtime.py, 59b359b); the shell mirrors both.
    from cm1core.column_components import column_component_spec

    instance.spec = column_component_spec(runtime_config.microphysics)
    instance.profile = runtime_config.profile
    instance.numq = runtime_config.profile.numq
    instance.repaired_scalars = runtime_config.profile.repaired_scalars
    instance.reflectivity_sink = lambda **_kwargs: None

    for name in instance.spec.component_persistence:
        shape = (
            g.surface_shape if name in instance.spec.surface_persistence
            else g.mass_shape
        )
        instance.persistence[name] = np.zeros(shape, dtype=np.float32)
    if instance.output_dbz:
        instance._allocate_reflectivity()
    fields = {
        name: np.zeros(g.mass_shape, dtype=np.float32)
        for name in ("pi0", "th0", "pp3d", "th3d", "rho", "prs")
    }
    fields["q3d"] = np.zeros((6, *g.mass_shape), dtype=np.float32)
    fields["qa"] = np.zeros((6, *g.mass_shape), dtype=np.float32)
    instance.dynamics = SimpleNamespace(
        fields=fields,
        metrics={"zf": np.zeros(g.w_shape, dtype=np.float32)},
        release_phase_workspace=lambda _name: None,
    )
    return instance


def _patch_device_calls(monkeypatch, *, dbz_source=None):
    """Replace the three device launches with recording host stand-ins."""

    seen: dict[str, object] = {"due": [], "composite": []}

    def _snapshot(*_args, **_kwargs):
        return SimpleNamespace(receipt={"operation_token_sha256": "c" * 64})

    # The component-generic launcher takes the column spec first (11f3ac3).
    def _wsm6(_spec, values, *, nx, ny, nk, refl_10cm_due=False, **_kwargs):
        seen["due"].append(refl_10cm_due)
        reflectivity = None
        if refl_10cm_due:
            reflectivity = np.zeros((nk, ny, nx), dtype=np.float32)
            if dbz_source is not None:
                reflectivity[:] = dbz_source(nk, ny, nx)
        return SimpleNamespace(
            receipt={"operation_token_sha256": "d" * 64},
            reflectivity_dbz=reflectivity,
        )

    def _post(*_args, **_kwargs):
        return SimpleNamespace(receipt={"operation_token_sha256": "e" * 64})

    def _composite(arrays, *, ni, nj, nk, level=None, **_kwargs):
        # The host transcription of misclibs.F:2259-2292 and the volume slice,
        # so the wiring is proven against the same arithmetic the kernel text
        # is measured against in the oracle test.
        np.copyto(arrays["cref"], cpu_composite_reflectivity(arrays["dbz"]))
        if level is not None:
            np.copyto(arrays["plane"], cpu_reflectivity_level(arrays["dbz"], level))
        seen["composite"].append({"ni": ni, "nj": nj, "nk": nk, "level": level})
        return SimpleNamespace(receipt={"schema": "test", "level": level})

    monkeypatch.setattr(
        finish_module, "prepare_cuda_gm1_tiled_post_wsm6_snapshot", _snapshot
    )
    monkeypatch.setattr(finish_module, "launch_cuda_gm1_tiled_column_block", _wsm6)
    monkeypatch.setattr(
        finish_module, "launch_cuda_gm1_tiled_post_wsm6_eos_qa", _post
    )
    monkeypatch.setattr(
        finish_module, "launch_cuda_composite_reflectivity", _composite
    )
    # The constant-pressure-to-eqtset=2 heating step (cuda_cartesian_moist_heating).
    monkeypatch.setattr(finish_module, "enqueue_rcvm_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        finish_module, "enqueue_moist_heating",
        lambda *_a, **_k: {"operation_token_sha256": "f" * 64},
    )
    monkeypatch.setattr(finish_module, "compile_cuda_cartesian_moist_heating", lambda: object())
    # The kernel compile imports CuPy; these tests never open a device.
    monkeypatch.setattr(
        finish_module, "compile_cuda_composite_reflectivity", lambda: object()
    )
    return seen


# ----------------------------------------------------- config text to call site


def test_a_toml_that_asks_for_reflectivity_reaches_the_component_and_the_carrier(
    tmp_path, monkeypatch
):
    """The whole path, from the two keys a user writes to the published words.

    CM1's ``output_dbz`` selects a diagnostic the microphysics scheme
    produces; this is the same decision travelling from the document to
    ``refl_10cm_due`` on the public component call, to the scattered volume,
    to ``calccref``'s composite, to the chosen level, to the frame profile.
    """

    document = _document(
        tmp_path, "on.toml", "output_dbz = true\ndbz_level_m = 1000.0\n"
    )
    config = load_config(document)
    assert config.physics.output_dbz is True
    assert config.physics.dbz_level_m == 1000.0
    # The document's canonical bytes carry the decision, so a sealed run
    # cannot claim reflectivity it did not ask for or the other way round.
    assert config.canonical_dict()["physics"]["output_dbz"] is True
    assert config.canonical_dict()["physics"]["dbz_level_m"] == 1000.0

    runtime_config = build_runtime_config(config)
    assert runtime_config.output_dbz is True
    assert runtime_config.dbz_level_m == 1000.0
    level = runtime_config.reflectivity_level_index
    # misclibs.F:2454-2466 keeps the lowest level at or above the request.
    column = runtime_config.grid.levels.zh
    assert float(column[level + 1]) >= 1000.0
    assert float(column[level]) < 1000.0

    # A grid this small keeps the host stand-in honest without a device.
    small = RuntimeConfig(
        CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0),
        tile_nx=7, tile_ny=7, output_dbz=True, dbz_level_m=1000.0,
    )
    finish = _finish_shell(small)
    assert set(finish.reflectivity) == {"dbz", "cref", "dbz_level"}

    def _pattern(nk, ny, nx):
        # A distinct word per level so a mis-scattered block is visible.
        return np.arange(nk, dtype=np.float32)[:, None, None] * np.float32(3.0)

    seen = _patch_device_calls(monkeypatch, dbz_source=_pattern)
    finish.microphysics(step=1, reflectivity_due=True)

    # cm1.F:3214-3237: the flag reached every column block of the step.
    assert seen["due"] and all(seen["due"])
    assert finish.completed_reflectivity_step == 1
    expected_volume = np.broadcast_to(
        np.arange(small.grid.nz, dtype=np.float32)[:, None, None] * np.float32(3.0),
        small.grid.mass_shape,
    )
    np.testing.assert_array_equal(finish.reflectivity["dbz"], expected_volume)
    # misclibs.F:2259-2292: the composite is the column maximum, everywhere.
    np.testing.assert_array_equal(
        finish.reflectivity["cref"],
        np.full(small.grid.surface_shape, (small.grid.nz - 1) * 3.0, np.float32),
    )
    # The chosen level is the level the configuration named, not the top.
    small_level = small.reflectivity_level_index
    np.testing.assert_array_equal(
        finish.reflectivity["dbz_level"],
        np.full(small.grid.surface_shape, small_level * 3.0, np.float32),
    )
    assert seen["composite"] == [
        {"ni": 9, "nj": 8, "nk": 12, "level": small_level}
    ]

    # The published frame carries them under CM1's own output names.
    profile = cartesian_history_profile(small)
    names = [carrier.name for carrier in profile.carriers]
    # 7183ce6: &param9 rows append after the fixed rows, so the reflectivity
    # rows follow the twelve prognostic carriers rather than ending the frame.
    assert names[12:15] == ["dbz", "cref", "dbz_level"]
    reflectivity = [c for c in profile.carriers if c.group == "reflectivity"]
    assert [c.units for c in reflectivity] == ["dBZ", "dBZ", "dBZ"]
    assert [c.shape for c in reflectivity] == [
        small.grid.mass_shape, small.grid.surface_shape, small.grid.surface_shape
    ]


def test_a_run_that_did_not_ask_for_reflectivity_never_asks_the_component(
    tmp_path, monkeypatch
):
    """The same call site, the same step, with the key absent."""

    config = load_config(_document(tmp_path, "off.toml"))
    assert config.physics.output_dbz is False
    runtime_config = build_runtime_config(config)
    assert runtime_config.output_dbz is False
    assert runtime_config.dbz_level_m is None

    small = RuntimeConfig(
        CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0), tile_nx=7, tile_ny=7
    )
    finish = _finish_shell(small)
    assert finish.reflectivity == {}
    seen = _patch_device_calls(monkeypatch)
    finish.microphysics(step=1)
    assert seen["due"] and not any(seen["due"])
    assert seen["composite"] == []
    assert finish.completed_reflectivity_step == 0
    assert "reflectivity" not in finish.step_receipts


def test_scheduling_reflectivity_without_a_carrier_refuses_rather_than_dropping_it():
    small = RuntimeConfig(
        CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0), tile_nx=7, tile_ny=7
    )
    finish = _finish_shell(small)
    with pytest.raises(RuntimeError, match="computed and dropped"):
        finish.microphysics(step=1, reflectivity_due=True)
    with pytest.raises(RuntimeError, match="column maximum of an unwritten carrier"):
        finish.composite_reflectivity(step=1)


# ---------------------------------------------------------------- the schedule


class _ScheduleProbe:
    """The runtime's getdbz decision, with nothing else of the runtime."""

    def __init__(self, output_dbz: bool) -> None:
        from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

        self.runtime = CudaCartesianRuntime.__new__(CudaCartesianRuntime)
        self.runtime.reflectivity_frame_steps = frozenset()
        self.runtime.finish = SimpleNamespace(output_dbz=output_dbz)


def test_getdbz_is_raised_only_on_the_steps_a_frame_is_published_at():
    """``cm1.F:3214-3237``: the flag is a schedule, not a permanent setting."""

    probe = _ScheduleProbe(output_dbz=True)
    probe.runtime.schedule_reflectivity((4, 8, 12))
    assert [probe.runtime.reflectivity_due(step) for step in range(1, 13)] == [
        False, False, False, True,
        False, False, False, True,
        False, False, False, True,
    ]
    # A run that never declared the diagnostic never raises the flag, even
    # when a caller schedules one.
    off = _ScheduleProbe(output_dbz=False)
    off.runtime.schedule_reflectivity((4,))
    assert off.runtime.reflectivity_due(4) is False


def test_a_frame_refuses_a_reflectivity_field_from_another_model_time():
    """A stale dBZ field beside this step's winds is refused, not published."""

    from cm1core.cartesian_history import require_cartesian_publication_boundary

    runtime = SimpleNamespace(
        clock=SimpleNamespace(failed=False, completed_steps=8),
        dynamics=SimpleNamespace(_expected_phase=("held", 1)),
        finish=SimpleNamespace(
            completed_physics_step=8, completed_solve3_step=8,
            output_dbz=True, completed_reflectivity_step=4,
        ),
    )
    with pytest.raises(RuntimeError, match="beside another model time's winds"):
        require_cartesian_publication_boundary(runtime)
    runtime.finish.completed_reflectivity_step = 8
    assert require_cartesian_publication_boundary(runtime) == 8
    # A run without the diagnostic is unaffected: the check is not reached.
    runtime.finish.output_dbz = False
    runtime.finish.completed_reflectivity_step = 0
    assert require_cartesian_publication_boundary(runtime) == 8


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "extra, code, fragment",
    (
        (
            "output_reflectivity = true\n",
            "CM1-CFG-UNKNOWN",
            "unconsumed configuration value",
        ),
        (
            "output_dbz = 1\n",
            "CM1-CFG-BOOLEAN",
            "truth values",
        ),
        (
            "dbz_level_m = 1000.0\n",
            "CM1-CFG-DBZ-LEVEL-INACTIVE",
            "never appear in a published frame",
        ),
        (
            "output_dbz = true\ndbz_level_m = 40000.0\n",
            "CM1-CFG-DBZ-LEVEL-ABOVE-TOP",
            "under a label naming a height it is not",
        ),
        (
            "output_dbz = true\ndbz_level_m = -5.0\n",
            "CM1-CFG-NUMERIC-RANGE",
            "",
        ),
    ),
)
def test_reflectivity_is_refused_by_key_and_by_value(tmp_path, extra, code, fragment):
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "bad.toml", extra))
    assert error.value.detail.code == code
    assert fragment in str(error.value)


def test_a_component_with_no_admitted_formula_refuses_output_dbz(tmp_path):
    """``param.F:1583-1603``: CM1 refuses output_dbz by ptype for this reason.

    A reflectivity formula is a moment of the scheme's own assumed size
    distributions.  The dry lane runs no microphysics at all and the native
    context keeps CM1's own component identity, so neither has one here.
    """

    for example in ("dry-psolver2-bootstrap.toml", "official-supercell-native-context.toml"):
        path = tmp_path / example
        path.write_text(
            (ROOT / "examples" / example).read_text(encoding="utf-8")
            + "\noutput_dbz = true\n",
            encoding="utf-8",
        )
        with pytest.raises(CM1Refusal) as error:
            load_config(path)
        assert error.value.detail.code == "CM1-CFG-DBZ-COMPONENT"
        message = str(error.value)
        assert "size distributions" in message
        assert "never scattered" in message
        assert "param.F:1583-1603" in message


def test_the_runtime_contract_refuses_the_same_things_without_the_door():
    """A programmatic caller cannot widen what the document grammar refuses."""

    grid = CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0)
    with pytest.raises(ValueError, match="never computed"):
        RuntimeConfig(grid, dbz_level_m=1000.0)
    with pytest.raises(ValueError, match="above this grid's highest scalar level"):
        RuntimeConfig(grid, output_dbz=True, dbz_level_m=40000.0)
    with pytest.raises(TypeError, match="strict boolean"):
        RuntimeConfig(grid, output_dbz=1)
    # The admitted component is accepted; anything else names the concrete
    # wrong result rather than falling back to a generic formula.
    assert RuntimeConfig(grid, output_dbz=True).output_dbz is True
    from dataclasses import replace

    from cm1core.gm1_runtime import REFLECTIVITY_COMPONENTS

    assert REFLECTIVITY_COMPONENTS == frozenset({"wsm6-mp6"})
    with pytest.raises(ValueError, match="never scattered"):
        replace(RuntimeConfig(grid, output_dbz=True), microphysics="kessler")
    # Without the diagnostic the same component is refused for the more
    # fundamental reason: there is no binding for it at all.  4b3f004 moved
    # that refusal into moisture_species, which names it by code.
    with pytest.raises(CM1Refusal, match="GM1-MOISTURE-COMPONENT-UNSUPPORTED"):
        replace(RuntimeConfig(grid), microphysics="kessler")


def test_reflectivity_level_index_is_cm1s_own_descending_scan():
    """``misclibs.F:2454-2466`` keeps the lowest level at or above a height."""

    grid = CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0)
    column = [float(value) for value in grid.levels.zh]
    # zh is CM1-indexed: index 0 is the ghost below ground.
    assert column[1:3] == [250.0, 750.0]
    assert reflectivity_level_index(grid, 250.0) == 0
    assert reflectivity_level_index(grid, 251.0) == 1
    assert reflectivity_level_index(grid, 750.0) == 1
    assert reflectivity_level_index(grid, 1000.0) == 2
    assert reflectivity_level_index(grid, column[grid.nz]) == grid.nz - 1
    assert reflectivity_level_index(grid, column[grid.nz] + 1.0) is None


# -------------------------------------------------------------- envelope budget


def test_the_envelope_cost_is_stated_and_the_carriers_are_optional_per_run():
    grid = CartesianGridGeometry(64, 64, 48, 1000.0, 1000.0, 500.0)
    off = RuntimeConfig(grid)
    on = RuntimeConfig(grid, output_dbz=True, dbz_level_m=1000.0)
    cost = reflectivity_envelope_cost(on)
    # 386071a: the &param9 default selection grew the frame from 12 carriers
    # to 32 without reflectivity (35 with dbz, cref and dbz_level), and from
    # about sixteen to about 17.39 float32 words per gridpoint.
    assert cost["carriers_without_reflectivity"] == 32
    assert cost["carriers_with_reflectivity"] == 35
    assert round(cost["words_per_gridpoint_without_reflectivity"], 2) == 17.39
    # dbz is exactly one more word per gridpoint; cref and the level plane are
    # one word per column each, which is 2/nz of a word per gridpoint.
    assert cost["added_words_per_gridpoint"] == pytest.approx(1.0 + 2.0 / 48.0)
    assert cost["added_payload_nbytes"] == 4 * (64 * 64 * 48 + 2 * 64 * 64)
    assert cost["optional_per_run"] is True
    assert reflectivity_envelope_cost(off)["added_payload_nbytes"] == 0
    # A run without the level plane pays for two carriers, not three.
    volume_only = RuntimeConfig(grid, output_dbz=True)
    assert reflectivity_envelope_cost(volume_only)["carriers_with_reflectivity"] == 34


def test_the_component_scratch_reserve_prices_the_reflectivity_volume():
    grid = CartesianGridGeometry(64, 64, 48, 1000.0, 1000.0, 500.0)
    off = cartesian_wsm6_scratch_reserve(RuntimeConfig(grid))
    on = cartesian_wsm6_scratch_reserve(RuntimeConfig(grid, output_dbz=True))
    assert "component_reflectivity_scratch_bytes" not in off
    block = 4 * 48 * 64 * 64
    assert on["component_reflectivity_scratch_bytes"] == block
    assert on["additional_reserve_bytes"] - off["additional_reserve_bytes"] == block


# --------------------------------------------------------------- the composite


def test_the_composite_is_cm1s_own_seeded_column_fold():
    """``misclibs.F:2267-2291``: seed at -1000.0, then fold max up the column."""

    rng = np.random.default_rng(20260914)
    dbz = (rng.standard_normal((7, 5, 3)).astype(np.float32) * np.float32(20.0))
    composite = cpu_composite_reflectivity(dbz)
    np.testing.assert_array_equal(composite, dbz.max(axis=0))
    # A zero-level column reports the seed, which no meteorological floor can
    # produce, rather than a plausible 0 dBZ.
    empty = cpu_composite_reflectivity(np.zeros((0, 5, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        empty, np.full((5, 3), CM1_CREF_SEED_DBZ, np.float32)
    )
    for level in range(dbz.shape[0]):
        np.testing.assert_array_equal(cpu_reflectivity_level(dbz, level), dbz[level])
    with pytest.raises(ValueError, match="another column's words"):
        cpu_reflectivity_level(dbz, dbz.shape[0])


# ------------------------------------------------------------------ the record


def test_the_support_matrix_and_the_frame_receipt_name_the_cm1_authority():
    from importlib import resources
    import csv

    rows = list(
        csv.DictReader(
            resources.files("cm1core.records")
            .joinpath("CM1-SUPPORT-MATRIX.csv")
            .open("r", encoding="utf-8", newline="")
        )
    )
    by_option = {row["option"]: row for row in rows}
    assert "README.namelist:1366-1369" in by_option["physics.output_dbz"]["source_anchor"]
    assert "src/param.F:1583-1603" in by_option["physics.output_dbz"]["source_anchor"]
    assert "src/misclibs.F:2454-2466" in by_option["physics.dbz_level_m"]["source_anchor"]

    grid = CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0)
    record = RuntimeConfig(grid, output_dbz=True, dbz_level_m=1000.0).record()
    assert record["output_dbz"] is True
    assert record["dbz_level_m"] == 1000.0
    assert record["dbz_level_index"] == 2
    assert "output_dbz" in record["reflectivity_authority"]
    assert "misclibs.F:2259-2292" in record["reflectivity_authority"]
    # The record must remain JSON-serializable: it is hashed into every frame.
    json.dumps(record, allow_nan=False, sort_keys=True)


# --------------------------------------------------------- the shipped example


def test_the_shipped_example_turns_reflectivity_on_end_to_end():
    """A deck a user can run, and the frame it would publish.

    The distribution ships the reflectivity deck beside the one it was made
    from, so "turn this on" is a document a reader can copy rather than a
    sentence in a document.
    """

    base = load_config(EXAMPLES / "cartesian-supercell-translating-frame.toml")
    example = load_config(EXAMPLES / "cartesian-supercell-reflectivity.toml")
    assert example.physics.output_dbz is True
    assert example.physics.dbz_level_m == 1000.0
    # Nothing but the two reflectivity words differs from the deck it is made
    # from: same sounding, shear, grid, clock, frame and closures.
    left, right = base.canonical_dict(), example.canonical_dict()
    assert right["physics"].pop("output_dbz") is True
    assert right["physics"].pop("dbz_level_m") == 1000.0
    assert left == right

    runtime_config = build_runtime_config(example)
    profile = cartesian_history_profile(runtime_config)
    # 7183ce6/386071a: the &param9 carriers append AFTER the fixed rows, so the
    # reflectivity rows follow the twelve prognostic carriers, not the tail.
    assert [carrier.name for carrier in profile.carriers][12:15] == [
        "dbz", "cref", "dbz_level"
    ]
    cost = reflectivity_envelope_cost(runtime_config)
    # The deck's own header states this number; keep the two together.
    assert round(cost["added_words_per_gridpoint"], 2) == 1.05
    # 386071a: the base deck's frame is now 25 carriers, about 17.29 words
    # per gridpoint (the deck header's 16.16 predates &param9).
    assert round(cost["words_per_gridpoint_without_reflectivity"], 2) == 17.29
    # The only thing that can still stop this deck is the installed ArWen
    # estate, which is a property of the machine and not of the document.
    from cm1core.engine_pin import inspect
    from cm1core.preflight import blockers

    drift = [] if inspect().admitted else ["CM1-ARWEN-ESTATE-DRIFT"]
    assert [item.code for item in blockers(example)] == drift


# ----------------------------------------------------------- frame diagnostics


def _diagnostics_shell(runtime_config: RuntimeConfig, finish: CartesianFinish):
    from cm1core.storm_diagnostics import FrameDiagnostics

    instance = FrameDiagnostics.__new__(FrameDiagnostics)
    instance.cp = _CP
    instance.stream = _Stream()
    instance.config = runtime_config
    instance.grid = runtime_config.grid
    instance.finish = finish
    instance.zh_column_m = [
        float(value) for value in runtime_config.grid.levels.zh[1:runtime_config.grid.nz + 1]
    ]
    return instance


def test_the_frame_row_carries_the_plan_view_radar_products(monkeypatch):
    """Column maximum, chosen level, and how much ground the core covers."""

    config = RuntimeConfig(
        CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0),
        tile_nx=7, tile_ny=7, output_dbz=True, dbz_level_m=1000.0,
    )
    finish = _finish_shell(config)
    _patch_device_calls(monkeypatch, dbz_source=lambda nk, ny, nx: np.float32(15.0))
    finish.microphysics(step=1, reflectivity_due=True)
    # One column of real core, so the coverage counts are not all-or-nothing.
    finish.reflectivity["dbz"][:, 2, 3] = np.float32(52.0)
    finish.reflectivity["cref"][:] = np.asarray(
        cpu_composite_reflectivity(finish.reflectivity["dbz"])
    )
    finish.reflectivity["dbz_level"][:] = np.asarray(
        cpu_reflectivity_level(finish.reflectivity["dbz"], finish.reflectivity_level)
    )

    diagnostics = _diagnostics_shell(config, finish)
    row = diagnostics._reflectivity(1)
    assert row["reflectivity_available"] is True
    assert row["reflectivity_max_dbz"] == 52.0
    assert row["composite_reflectivity_max_dbz"] == 52.0
    assert row["reflectivity_level_index"] == finish.reflectivity_level
    assert row["reflectivity_level_m"] == 1250.0
    assert row["reflectivity_at_level_max_dbz"] == 52.0
    coverage = row["composite_reflectivity_coverage"]
    cell_km2 = 0.5 * 0.5
    assert coverage["columns_ge_20_dbz"] == 1
    assert coverage["area_km2_ge_20_dbz"] == pytest.approx(cell_km2)
    assert coverage["columns_ge_40_dbz"] == 1
    assert coverage["columns_ge_50_dbz"] == 1
    assert "misclibs.F:2259-2292" in row["reflectivity_authority"]

    # A row taken at a step the diagnostic was not raised on says so rather
    # than reporting the previous output step's echo as this step's.
    stale = diagnostics._reflectivity(2)
    assert stale == {
        "reflectivity_available": False, "reflectivity_computed_at_step": 1
    }
    # A run without the diagnostic carries no reflectivity words at all.
    off = _diagnostics_shell(
        RuntimeConfig(CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0)),
        _finish_shell(RuntimeConfig(CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0))),
    )
    assert off._reflectivity(1) == {}


def test_the_analysis_tool_reads_the_radar_products_and_says_when_there_are_none():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_analyze_storm_run", ROOT / "tools/analyze_storm_run.py"
    )
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)

    frames = [
        {
            "step": step, "model_seconds": step * 0.5,
            "reflectivity_available": True,
            "composite_reflectivity_max_dbz": 20.0 + step,
            "reflectivity_max_dbz": 25.0 + step,
            "reflectivity_at_level_max_dbz": 10.0 + step,
            "reflectivity_level_m": 1250.0,
            "composite_reflectivity_coverage": {
                "area_km2_ge_40_dbz": float(step), "columns_ge_40_dbz": step,
            },
        }
        for step in (10, 20, 30)
    ]
    section = analyze._reflectivity(frames, 40.0)
    assert section["frames_with_reflectivity"] == 3
    assert section["peak_composite_dbz"]["value"] == 50.0
    assert section["peak_composite_dbz"]["step"] == 30
    assert section["first_reaching_threshold"]["step"] == 20
    assert section["peak_level_dbz"]["value"] == 40.0
    assert section["level_m"] == 1250.0
    assert section["peak_core_area"] == {
        "threshold_dbz": 40.0, "area_km2": 30.0, "step": 30, "final_area_km2": 30.0
    }
    # A run that never asked for reflectivity is not a storm with no echo.
    silent = analyze._reflectivity([{"step": 1}], 40.0)
    assert silent["frames_with_reflectivity"] == 0
    assert "declared physics.output_dbz off" in silent["note"]


def test_the_device_memory_plan_prices_reflectivity_before_a_device_opens():
    """A run can be priced, and refused, without a GPU."""

    from cm1core.cuda_cartesian_finish import cartesian_reflectivity_allocation_plan
    from cm1core.storm_diagnostics import frame_diagnostic_scratch_bytes

    grid = CartesianGridGeometry(64, 64, 48, 1000.0, 1000.0, 500.0)
    off = RuntimeConfig(grid)
    volume_only = RuntimeConfig(grid, output_dbz=True)
    with_level = RuntimeConfig(grid, output_dbz=True, dbz_level_m=1000.0)

    assert cartesian_reflectivity_allocation_plan(off)["resident_total_bytes"] == 0
    assert cartesian_reflectivity_allocation_plan(off)["words_per_gridpoint"] == 0.0
    plan = cartesian_reflectivity_allocation_plan(with_level)
    assert plan["resident_volume_bytes"] == 4 * 48 * 64 * 64
    assert plan["resident_plane_bytes"] == 2 * 4 * 64 * 64
    assert plan["words_per_gridpoint"] == pytest.approx(1.0 + 2.0 / 48.0)
    assert cartesian_reflectivity_allocation_plan(volume_only)[
        "words_per_gridpoint"
    ] == pytest.approx(1.0 + 1.0 / 48.0)

    # The coverage counts form one boolean surface at a time, and it is priced.
    base = frame_diagnostic_scratch_bytes(off)
    priced = frame_diagnostic_scratch_bytes(with_level)
    assert "reflectivity_coverage_mask_bytes" not in base
    assert priced["reflectivity_coverage_mask_bytes"] == 64 * 64
    assert (
        priced["frame_scratch_peak_bytes"] - base["frame_scratch_peak_bytes"] == 64 * 64
    )


def test_the_run_receipt_says_whether_reflectivity_was_on_and_where_it_came_from():
    """A receipt from a run without it is exactly the receipt it always was."""

    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

    def _probe(runtime_config: RuntimeConfig, steps=()):
        runtime = CudaCartesianRuntime.__new__(CudaCartesianRuntime)
        runtime.config = runtime_config
        runtime.reflectivity_frame_steps = frozenset(steps)
        runtime.finish = SimpleNamespace(
            output_dbz=runtime_config.output_dbz, completed_reflectivity_step=max(steps, default=0)
        )
        return runtime._reflectivity_record()

    grid = CartesianGridGeometry(64, 64, 48, 1000.0, 1000.0, 500.0)
    assert _probe(RuntimeConfig(grid)) == {}
    record = _probe(
        RuntimeConfig(grid, output_dbz=True, dbz_level_m=1000.0), steps=(120, 240)
    )["reflectivity"]
    assert record["output_dbz"] is True
    assert record["output_steps"] == [120, 240]
    assert record["computed_at_step"] == 240
    assert record["dbz_level_index"] == 2
    assert record["device_plan"]["resident_volume_bytes"] == 4 * 48 * 64 * 64
    # 386071a: the &param9 default selection grew the frame (12 -> 32 + 3).
    assert record["envelope_cost"]["carriers_with_reflectivity"] == 35
    assert "README.namelist:1366-1369" in record["authority"]
    json.dumps(record, allow_nan=False, sort_keys=True)


def test_the_reflectivity_producer_is_inside_the_byte_pinned_arwen_surface():
    """The formula is the pinned scheme's, and the record says so.

    Nothing in this port derives reflectivity.  The producer is ArWen's own
    WSM6 routine behind the public seam, and both the kernel translation unit
    and its host driver are inside the byte-admitted surface, so a changed
    formula is an engine-drift refusal rather than a quietly different field.
    """

    from cm1core import engine_pin
    from cm1core.arwen_wsm6_component import (
        ARWEN_PUBLIC_REFLECTIVITY_CALL,
        component_metadata,
    )
    from cm1core.cuda_gm1_tilestream_wsm6 import WSM6_REFLECTIVITY_AUTHORITY

    surface = engine_pin.audited_surface()
    assert "gpuwm/core/kernels/refl.cu" in surface
    assert "gpuwm/core/refl.py" in surface
    assert all(len(value) == 64 for value in surface.values())

    reflectivity = component_metadata()["execution"]["reflectivity"]
    assert reflectivity["consumer"] == ARWEN_PUBLIC_REFLECTIVITY_CALL
    assert reflectivity["units"] == "dBZ"
    assert reflectivity["species_mapping"] == ["qv", "qr", "qs", "qg"]
    # 4b3f004 renamed the key when the metadata became component-generic.
    assert reflectivity["absent_species"] == ["qc", "qi"]
    assert "not restart state" in reflectivity["lifetime"]

    assert "mp_wsm6.F90:2275-2444" in WSM6_REFLECTIVITY_AUTHORITY
    seam = (
        Path(__file__).resolve().parents[1]
        / "src/cm1core/records/CM1-ARWEN-SEAM.md"
    ).read_text(encoding="utf-8")
    assert "output_dbz" in seam
    assert "misclibs.F:2259-2292" in seam
    assert "param.F:1583-1603" in seam


def test_the_published_carrier_names_are_the_ones_both_renderers_look_for():
    """A rename on one side of the seam leaves the field unreachable.

    ``cm1-render`` selects a carrier by the exact name in the envelope's field
    table, and ``gm1-view`` admits the reflectivity volume out of the history
    by the same name, so the three have to agree by construction rather than
    by memory.
    """

    config = RuntimeConfig(
        CartesianGridGeometry(9, 8, 12, 500.0, 500.0, 500.0),
        output_dbz=True, dbz_level_m=1000.0,
    )
    published = {
        carrier.name: carrier
        for carrier in cartesian_history_profile(config).carriers
        if carrier.group == "reflectivity"
    }
    assert set(published) == {"dbz", "cref", "dbz_level"}
    # CM1's own output names, so a reader who knows CM1 needs no translation.
    assert published["dbz"].units == "dBZ"
    assert published["cref"].meaning.startswith("composite")
    # The level plane is labelled with the level it is actually taken at, not
    # only with the height the document asked for.
    assert "zh=1250.0 m" in published["dbz_level"].meaning
    assert "requested 1000.0 m" in published["dbz_level"].meaning

    viewer = (ROOT / "rust/gm1-view/src/input.rs").read_text(encoding="utf-8")
    assert 'for name in ["th3d", "ppi", "dbz"]' in viewer
    # The viewer picks the radar scale off the declared units, so the carrier
    # has to declare them exactly.
    slice_source = (ROOT / "rust/gm1-view/src/slice.rs").read_text(encoding="utf-8")
    assert 'units.eq_ignore_ascii_case("dbz")' in slice_source

    # One NWS table, read by both renderers, so they cannot drift apart.
    scale = (ROOT / "rust/cm1-state/src/nws_reflectivity.rs").read_text(encoding="utf-8")
    assert "NWS_DBZ_CLASSES" in scale
    for consumer in ("rust/cm1-render/src/main.rs", "rust/gm1-view/src/science_layers.rs",
                     "rust/gm1-view/src/slice.rs"):
        text = (ROOT / consumer).read_text(encoding="utf-8")
        assert "cm1_state::nws_reflectivity" in text, consumer
