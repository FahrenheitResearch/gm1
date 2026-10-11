"""Tiling is optional: one whole-domain closure tile, and the proof it computes the same words.

A document that leaves ``runtime.tile_nx`` and ``runtime.tile_ny`` out runs the
closure as one tile covering the domain.  These tests pin four things:

* the grammar -- the pair is all-or-none, no other spelling means "untiled",
  and a document that declares the pair keeps its exact canonical bytes;
* the door -- an untiled closure is admitted only when its priced device peak
  fits the declared budget, and a refusal names the explicit tile that fits;
* the words -- the REAL ``CartesianClosure`` run through
  :mod:`tools.oracle.host_cupy` (packaged CUDA text compiled by g++) publishes
  raw-word identical ``held``/``tke`` carriers tiled and untiled, across the
  lower-boundary, damping, radiation and two-part SGS selections;
* the one exception -- CM1's cecd=2/3 wind-dependent water roughness, whose
  tiled path starts a halo column's iteration from a roughness length a
  previous tile already advanced, so the tiled words differ from the untiled
  (and CM1-ordered) ones at tile seams only.
"""

from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np
import pytest

from cm1core.cartesian_door import build_runtime_config, closure_decomposition_peak_bytes
from cm1core.config import load_config, parse_mapping
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"
needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="the host kernel emulator needs g++")


def _mapping(**runtime: object) -> dict[str, object]:
    payload = load_config(EXAMPLE).canonical_dict()
    raw = {
        section: copy.deepcopy(payload[section])
        for section in ("run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime")
    }
    for key, value in runtime.items():
        if value is None:
            raw["runtime"].pop(key, None)  # type: ignore[union-attr]
        else:
            raw["runtime"][key] = value  # type: ignore[index]
    return raw


def test_a_declared_tile_keeps_the_sealed_document_identity() -> None:
    config = load_config(EXAMPLE)
    assert (config.runtime.tile_nx, config.runtime.tile_ny) == (64, 64)
    assert config.canonical_dict()["runtime"]["tile_nx"] == 64
    # The example's own identity, as it was before a0cee5e and is again since
    # 43b4792.  This pin was first taken while canonical_dict leaked a
    # top-level "tornado_diagnostics": null into every document (c01e2acf...);
    # the tile keys never moved it.
    assert config.sha256 == "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    runtime_config = build_runtime_config(config)
    assert runtime_config.tile_selection is None
    assert "tile_selection" not in runtime_config.record()


def test_leaving_both_keys_out_projects_neither_and_changes_nothing_else() -> None:
    tiled = parse_mapping(_mapping())
    untiled = parse_mapping(_mapping(tile_nx=None, tile_ny=None))
    assert (untiled.runtime.tile_nx, untiled.runtime.tile_ny) == (None, None)
    left, right = tiled.canonical_dict(), untiled.canonical_dict()
    del left["runtime"]["tile_nx"], left["runtime"]["tile_ny"]
    assert left == right
    assert tiled.sha256 != untiled.sha256


@pytest.mark.parametrize("present", ("tile_nx", "tile_ny"))
def test_half_a_tile_is_refused_by_name(present: str) -> None:
    missing = "tile_ny" if present == "tile_nx" else "tile_nx"
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_mapping(**{missing: None}))
    assert caught.value.detail.code == "CM1-CFG-CARTESIAN-TILE-PAIR"
    assert caught.value.detail.option == f"runtime.{missing}"


@pytest.mark.parametrize("value", (0, "auto", 256, 96, -64, 64.0, True))
def test_no_other_spelling_means_untiled(value: object) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_mapping(tile_nx=value))
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert caught.value.detail.option == "runtime.tile_nx"
    assert "leave both" in caught.value.detail.remedy


def test_the_door_binds_one_whole_domain_tile_that_fits_the_budget() -> None:
    config = build_runtime_config(parse_mapping(_mapping(tile_nx=None, tile_ny=None)))
    grid = config.grid
    assert (config.tile_nx, config.tile_ny) == (grid.nx, grid.ny)
    # The undeclared WSM6 block follows the closure tile, so it is one block.
    assert (config.physics_block_nx, config.physics_block_ny) == (grid.nx, grid.ny)
    record = config.record()
    assert record["tile_shape_yx"] == [grid.ny, grid.nx]
    assert record["physics_block_count"] == 1
    assert record["tile_selection"]["declared"] is False
    assert record["tile_selection"]["priced_peak_bytes"] <= config.memory_budget_bytes
    from cm1core.cuda_cartesian_closure import _tile_plan

    assert _tile_plan(grid, config.tile_nx, config.tile_ny).ntiles == 1


def test_an_untiled_closure_that_does_not_fit_is_refused_naming_the_tile_that_does() -> None:
    raw = _mapping(tile_nx=None, tile_ny=None, memory_budget_mib=700)
    raw["grid"]["nx"] = raw["grid"]["ny"] = 160  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        build_runtime_config(parse_mapping(raw))
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-UNTILED-BUDGET"
    assert detail.option == "runtime.memory_budget_mib"
    assert "runtime.tile_nx = runtime.tile_ny = 128" in detail.remedy
    raw["runtime"]["memory_budget_mib"] = 1024  # type: ignore[index]
    admitted = build_runtime_config(parse_mapping(raw))
    assert (admitted.tile_nx, admitted.tile_ny) == (160, 160)


def test_the_priced_peak_grows_with_the_tile_and_not_the_dynamics() -> None:
    from cm1core.gm1_runtime import RuntimeConfig
    from cm1core.grid_geometry import CartesianGridGeometry

    grid = CartesianGridGeometry(512, 512, 75, 100.0, 100.0, 100.0)
    tiled = closure_decomposition_peak_bytes(RuntimeConfig(grid, tile_nx=64, tile_ny=64))
    whole = closure_decomposition_peak_bytes(RuntimeConfig(grid, tile_nx=512, tile_ny=512))
    assert tiled["dynamics_phase_peak_bytes"] == whole["dynamics_phase_peak_bytes"]
    assert whole["held_phase_bytes"] > 50 * tiled["held_phase_bytes"]
    assert whole["priced_peak_bytes"] > tiled["priced_peak_bytes"]


# ---------------------------------------------------------------- the words


def _identity(overrides: dict[str, object], *, nx: int = 40, ny: int = 34, tile: int = 16):
    from tools.oracle import cartesian_tiling_identity as harness

    tiled = harness.run(nx=nx, ny=ny, nz=8, tile_nx=tile, tile_ny=tile, steps=2, overrides=overrides)
    whole = harness.run(nx=nx, ny=ny, nz=8, tile_nx=nx, tile_ny=ny, steps=2, overrides=overrides)
    assert tiled["ntiles"] > 1 and whole["ntiles"] == 1
    return tiled, whole


@needs_gxx
@pytest.mark.parametrize(
    "overrides",
    (
        {},
        {"bbc": 2},
        {"bbc": 3, "sfcmodel": 1, "isfcflx": 1, "cecd": 1},
        {"bbc": 3, "sfcmodel": 1, "set_znt": 1, "cecd": 3},
        {"irdamp": 2},
        {"hrdamp": 1, "xhd_m": 1500.0},
        {"rterm": 1},
        {"icor": 1, "fcor_s_inv": 1.0e-4, "lspgrad": 1, "u_move_m_s": 8.5, "v_move_m_s": 6.0},
        {"sgsmodel": 3},
        {"sgsmodel": 4, "bbc": 3, "sfcmodel": 1},
        {"microphysics": "morrison-mp10"},
    ),
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "supercell-default",
)
def test_the_tiled_and_untiled_closure_publish_the_same_raw_words(overrides) -> None:
    tiled, whole = _identity(overrides)
    assert tiled["words"].keys() == whole["words"].keys()
    differing = [name for name, value in tiled["words"].items()
                 if value.tobytes() != whole["words"][name].tobytes()]
    assert differing == []
    # The comparison is not vacuous: the closure produced nonzero, finite words.
    held = whole["words"]["step2.held.held.uten"]
    assert np.isfinite(held).all() and np.count_nonzero(held) > held.size // 2


@needs_gxx
def test_a_stretch_z1_column_is_also_identical() -> None:
    """The lane's storm decks run stretch_z=1; the vertical metric changes no seam."""

    from tools.oracle import cartesian_tiling_identity as harness

    vertical = dict(stretch_z=1, ztop_m=1220.0, str_bot_m=240.0, str_top_m=900.0,
                    dz_bot_m=60.0, dz_top_m=160.0)
    tiled = harness.run(nx=40, ny=34, nz=12, tile_nx=16, tile_ny=16, steps=2, vertical=vertical)
    whole = harness.run(nx=40, ny=34, nz=12, tile_nx=40, tile_ny=34, steps=2, vertical=vertical)
    assert tiled["ntiles"] == 9 and whole["ntiles"] == 1
    assert harness.compare(tiled, whole)["raw_word_identical"] is True


@needs_gxx
def test_a_domain_barely_wider_than_one_tile_is_also_identical() -> None:
    tiled, whole = _identity({}, nx=23, ny=22, tile=16)
    assert all(value.tobytes() == whole["words"][name].tobytes() for name, value in tiled["words"].items())


@needs_gxx
@pytest.mark.parametrize("cecd", (2, 3))
def test_the_water_roughness_iteration_is_the_one_seam_dependent_path(cecd: int) -> None:
    """src/sfcphys.F:138-194 iterates znt from the previous step's value.

    The tiled closure scatters each tile's advanced ``znt`` back into the
    resident carrier before the next tile gathers its halo, so a halo column
    near a seam starts the iteration from an already-advanced roughness and
    its ``ust`` -- which the seam face's stress averages -- differs.  The
    untiled closure starts every column from the previous step, as CM1's own
    column loop does.  The difference is confined to the lowest two levels at
    tile seams: the stress, and the scalar tendencies, whose lowest eddy
    diffusivities come from tkebc's ust-dependent surface TKE once a fresh
    start runs tkekm from step 1 (src/init3d.F:2688-2760).  Nothing else moves.
    """

    tiled, whole = _identity({"bbc": 3, "sfcmodel": 1, "cecd": cecd})
    seams_x, seams_y = {16, 32}, {16, 32}
    moved = False
    for name, value in tiled["words"].items():
        other = whole["words"][name]
        if value.tobytes() == other.tobytes():
            continue
        moved = True
        assert name.split(".")[-1] in ("uten", "vten", "wten", "thten1", "qten"), name
        where = np.argwhere(value.view(np.uint32) != other.view(np.uint32))
        assert set(where[:, -3].tolist()) <= {0, 1}, name
        on_seam = [(j in seams_y or i in seams_x) for _, j, i in where[:, -3:]]
        assert all(on_seam), name
    assert moved


# ------------------------------------------------------- the GPU benchmark tool


def test_the_benchmark_variants_differ_only_in_the_decomposition_and_the_clock(tmp_path) -> None:
    from tools.benchmark_cartesian_untiled import variants

    texts = variants(EXAMPLE, tile=64, steps=4, warmup=1, memory_budget_mib=None)
    paths = {}
    for name, text in texts.items():
        paths[name] = tmp_path / f"{name}.toml"
        paths[name].write_text(text)
    tiled, untiled = (load_config(paths[name]) for name in ("tiled", "untiled"))
    assert (tiled.runtime.tile_nx, untiled.runtime.tile_nx) == (64, None)
    left, right = tiled.canonical_dict(), untiled.canonical_dict()
    del left["runtime"]["tile_nx"], left["runtime"]["tile_ny"]
    assert left == right
    assert tiled.run.duration_s == 4 * tiled.run.dt_s
    assert build_runtime_config(untiled).tile_nx == untiled.grid.nx


def test_the_benchmark_compares_every_carrier_word_for_word(tmp_path) -> None:
    from tools.benchmark_cartesian_untiled import compare_frames

    words = np.arange(24, dtype="<f4")
    for name in ("a", "b"):
        frame = tmp_path / name / "frame"
        frame.mkdir(parents=True)
        (frame / "history-carriers.f32").write_bytes(words.tobytes())
        (frame / "history-field-table.tsv").write_text(
            "u\tm\tu\tmass\t2,3,2\t-\tpath\t0\nw\tm\tu\tmass\t12\t-\tpath\t48\n"
        )
        (frame / "history-scheduler-scalars.tsv").write_text("x\t1\n")
    assert compare_frames(tmp_path / "a", tmp_path / "b")["raw_word_identical"] is True
    changed = words.copy()
    changed[20] = np.nextafter(changed[20], np.float32(1e9))
    (tmp_path / "b" / "frame" / "history-carriers.f32").write_bytes(changed.tobytes())
    result = compare_frames(tmp_path / "a", tmp_path / "b")
    assert result["raw_word_identical"] is False
    assert result["frames"][0]["differing"]["w"]["differing_words"] == 1
    assert "u" not in result["frames"][0]["differing"]
