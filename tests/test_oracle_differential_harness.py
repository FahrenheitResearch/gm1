"""The differential harness: the key map, the translator and the readers.

Nothing here needs a GPU or the CM1 binary except the cases marked with
``requires_cm1``, which skip when no built oracle is on the search path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1core.oracle import compare as _compare
from cm1core.oracle import cm1_run, grads, horizon, keymap, namelist
from cm1core.oracle.spec import CaseError, DifferentialCase, load_case

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "examples" / "oracle" / "acoustic-warm-bubble-32.differential.toml"


def _authority_run() -> Path | None:
    try:
        return cm1_run.authority_run_dir()
    except cm1_run.CM1RunError:
        return None


requires_cm1 = pytest.mark.skipif(
    _authority_run() is None, reason="no built CM1 r21.1 oracle on the authority search path"
)


# -- the key map ---------------------------------------------------------


def test_keymap_record_is_self_consistent() -> None:
    table = keymap.load()
    assert table.schema == "gm1.cm1-r21.1-to-gpuwm-cm1-keymap/v1"
    assert len(table.mapped) + len(table.unmapped) == len(table.entries)
    assert table.counts["cm1_fields_mapped"] == len(table.mapped)
    assert table.counts["cm1_fields_unmapped"] == len(table.unmapped)
    assert (
        table.counts["cm1_fields_physical"]
        + table.counts["cm1_fields_mapped_from_non_physical_groups"]
    ) == len(table.entries)


def test_keymap_names_only_gpuwm_cm1_schema_keys() -> None:
    """Every gpuwm-cm1 key the map names must exist in ``config.py``'s schema.

    A map that named a key the parser does not read would emit a document the
    door refuses, and the refusal would look like a configuration error rather
    than a stale map.
    """

    from cm1core.config import _SCHEMA  # noqa: PLC0415  (test-only private read)

    schema = {f"{section}.{key}" for section, keys in _SCHEMA.items() for key in keys}
    named = {e.gm1_key for e in keymap.load().mapped}
    assert named <= schema


def test_every_gpuwm_cm1_key_is_mapped_or_declared_gm1_only() -> None:
    from cm1core.config import _SCHEMA  # noqa: PLC0415

    schema = {f"{section}.{key}" for section, keys in _SCHEMA.items() for key in keys}
    table = keymap.load()
    covered = {e.gm1_key for e in table.mapped} | {r["gm1_key"] for r in table.gm1_only}
    assert schema - covered == set()


def test_zd_is_a_height_and_not_a_depth() -> None:
    """The one transform whose identity form would silently misplace a damper."""

    entry = keymap.load().by_cm1("zd")
    assert entry is not None
    assert entry.gm1_key == "closures.upper_damping_depth_m"
    assert entry.transform == "cm1-zd-to-depth"


# -- the namelist reader -------------------------------------------------


def test_namelist_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "namelist.input"
    source.write_text(
        " &param0\n nx = 20,\n terrain_flag = .false.,  ! a comment\n /\n"
        " &param1\n dx = 1000.0,\n dtl = 1.5d0,\n /\n",
        encoding="utf-8",
    )
    deck = namelist.read(source)
    assert deck["param0"]["nx"] == 20
    assert deck["param0"]["terrain_flag"] is False
    assert deck["param1"]["dx"] == 1000.0
    assert deck["param1"]["dtl"] == 1.5
    target = tmp_path / "out.input"
    namelist.write(deck, target)
    assert namelist.read(target) == deck


def test_namelist_apply_reports_keys_the_deck_has_no_group_for(tmp_path: Path) -> None:
    source = tmp_path / "namelist.input"
    source.write_text(" &param0\n nx = 20,\n /\n", encoding="utf-8")
    deck = namelist.read(source)
    assert namelist.apply(deck, {"nx": 32, "ny": 32}) == ["ny"]
    assert deck["param0"]["nx"] == 32


# -- the case document ---------------------------------------------------


def test_case_refuses_a_cm1_key_that_is_not_a_physical_selector(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        '[case]\nname = "x"\nterm = "acoustic"\n[cm1]\ndoazimavg = 1\n'
        '[gm1]\n"physics.component" = "off"\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="output, statistics, parcel and restart selectors"):
        load_case(path)


def test_case_refuses_an_output_selector_the_harness_owns(tmp_path: Path) -> None:
    """&param9 is mapped key-for-key now, and still not a case document's to set.

    The harness forces the same selection on both sides so that every field
    the comparator pairs was written by both models.  A case that moved one
    would silently compare a field CM1 wrote against one gpuwm-cm1 did not.
    """

    path = tmp_path / "case.toml"
    path.write_text(
        '[case]\nname = "x"\nterm = "acoustic"\n[cm1]\noutput_th = 1\n'
        '[gm1]\n"physics.component" = "off"\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="&param9 output selectors"):
        load_case(path)


def test_case_refuses_a_gm1_key_cm1_has_a_word_for(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        '[case]\nname = "x"\nterm = "acoustic"\n[cm1]\nnx = 20\n'
        '[gm1]\n"grid.nx" = 20\n"physics.component" = "off"\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseError, match="gm1_only"):
        load_case(path)


@requires_cm1
def test_worked_example_translates_with_nothing_unrepresentable() -> None:
    case = load_case(CASE)
    base = cm1_run.base_deck("supercell")
    translation = case.translate(base)
    assert translation.unrepresentable == ()
    assert translation.receipt()["unvalidated"] is False
    # depth = ztop - zd = 40*500 - 15000
    assert translation.gm1["closures"]["upper_damping_depth_m"] == 5000.0
    # an unstretched grid states the param6 words by omission
    assert "ztop_m" not in translation.gm1["grid"]
    # config.py:1228 requires the two spellings of bbc to agree
    assert translation.gm1["boundaries"]["bottom"] == translation.gm1["closures"]["bbc"]


@requires_cm1
def test_a_cm1_selector_with_no_gm1_key_is_named_in_the_receipt(tmp_path: Path) -> None:
    """The gap detector must fire on physics, and only on physics."""

    case = load_case(CASE)
    base = cm1_run.base_deck("supercell")
    # icor/fcor used to be the example here: they were CM1 selectors with no
    # GM1 key at all.  They are now dynamics.coriolis/dynamics.fcor_s_inv, so
    # the detector must stay silent about them and the translated document
    # must carry them.
    with_coriolis = DifferentialCase(
        name=case.name, term=case.term, notes=case.notes,
        cm1={**case.cm1, "icor": 1, "fcor": 1.0e-4},
        gm1_only=case.gm1_only, source=case.source,
    )
    translated = with_coriolis.translate(base)
    assert translated.unrepresentable == ()
    assert translated.gm1["dynamics"]["coriolis"] == 1
    assert translated.gm1["dynamics"]["fcor_s_inv"] == 1.0e-4
    # horizturb is CM1's 2-D Smagorinsky closure and iptra its passive-tracer
    # switch; neither exists anywhere in this port, so both must be named.
    with_missing = DifferentialCase(
        name=case.name, term=case.term, notes=case.notes,
        cm1={**case.cm1, "horizturb": 1, "iptra": 1},
        gm1_only=case.gm1_only, source=case.source,
    )
    receipt = with_missing.translate(base).receipt()
    assert receipt["unvalidated"] is True
    named = {row["cm1_key"] for row in receipt["unrepresentable_overrides"]}
    assert {"horizturb", "iptra"} <= named
    # statfrq changes a write cadence, not state; it must never be reported
    # as physics the two codes disagree about.
    with_cadence = DifferentialCase(
        name=case.name, term=case.term, notes=case.notes,
        cm1={**case.cm1, "statfrq": 7.0}, gm1_only=case.gm1_only, source=case.source,
    )
    assert with_cadence.translate(base).unrepresentable == ()


@requires_cm1
def test_emitted_gm1_document_is_admitted_by_the_configuration_door() -> None:
    from cm1core.config import load_config  # noqa: PLC0415

    case = load_case(CASE)
    text = case.gm1_toml(cm1_run.base_deck("supercell"))
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        load_config(path)
    finally:
        path.unlink()


# -- the horizon calculator ----------------------------------------------


def test_horizon_acoustic_is_one_hundred_cell_crossings() -> None:
    grid = horizon.Grid(nx=32, ny=32, nz=40, dx_m=1000.0, dy_m=1000.0, dz_m=500.0, dt_s=1.0)
    result = horizon.compute("acoustic", grid)
    assert result.timescale_s == pytest.approx(1000.0 / 360.0)
    assert result.large_steps == 278
    assert result.acoustic_steps == 278 * 6
    assert result.differential_unsuitable is False


def test_horizon_refuses_terms_whose_mechanism_it_does_not_know() -> None:
    grid = horizon.Grid(nx=8, ny=8, nz=8, dx_m=100.0, dy_m=100.0, dz_m=100.0, dt_s=1.0)
    with pytest.raises(KeyError, match="horizon is a guess wearing a formula"):
        horizon.compute("volcano", grid)


@pytest.mark.parametrize("term", ["coriolis", "radiation"])
def test_slow_terms_are_marked_unsuitable_and_name_the_alternative(term: str) -> None:
    grid = horizon.Grid(nx=32, ny=32, nz=40, dx_m=1000.0, dy_m=1000.0, dz_m=500.0, dt_s=1.0)
    result = horizon.compute(term, grid)
    assert result.differential_unsuitable is True
    assert result.instead


# -- the comparator ------------------------------------------------------


def test_th3d_is_compared_against_thpert_not_th() -> None:
    """GM1's th3d carrier holds the perturbation despite its meaning string."""

    assert _compare.CM1_NAME["th3d"] == "thpert"


def test_difference_statistics(tmp_path: Path) -> None:
    import numpy as np

    a = np.zeros((2, 3, 4), dtype="<f4")
    b = np.zeros((2, 3, 4), dtype="<f4")
    b[1, 2, 3] = 10.0
    a[1, 2, 3] = 11.0
    row = _compare._difference("f", "f", 5.0, a, b)
    assert row.max_abs == pytest.approx(1.0)
    assert row.max_abs_at == (1, 2, 3)
    assert row.cm1_peak == pytest.approx(10.0)
    assert row.max_abs_over_peak == pytest.approx(0.1)
    assert row.model_time_s == 5.0


# -- the GrADS reader ----------------------------------------------------


def test_grads_reader_reads_raw_binary32(tmp_path: Path) -> None:
    import array

    nx, ny, nz = 3, 2, 4
    ctl = tmp_path / "cm1out_s.ctl"
    ctl.write_text(
        "dset ^cm1out_00%y4_s.dat\n"
        "options template\n"
        "undef  -999999.9\n"
        f"xdef {nx} linear 0.0 1.0\n"
        f"ydef {ny} linear 0.0 1.0\n"
        f"zdef {nz} linear 0.0 1.0\n"
        "tdef 1 linear 00:00Z01JAN0001 1YR\n"
        "vars 2\n"
        "surf              0 99 a surface field                                  (1)\n"
        f"th               {nz} 99 potential temperature                          (K)\n"
        "endvars\n",
        encoding="utf-8",
    )
    words = array.array("f", [float(i) for i in range(nx * ny * (1 + nz))])
    (tmp_path / "cm1out_000001_s.dat").write_bytes(words.tobytes())
    dataset = grads.read_ctl(ctl)
    assert [v.name for v in dataset.variables] == ["surf", "th"]
    assert dataset.offset_words("th") == nx * ny
    values, shape = dataset.read("th", 1)
    assert shape == (nz, ny, nx)
    assert list(values) == [float(i) for i in range(nx * ny, nx * ny * (1 + nz))]


@requires_cm1
def test_authority_tree_is_never_written(tmp_path: Path) -> None:
    """``stage`` copies out of the authority tree and never into it."""

    run = cm1_run.authority_run_dir()
    before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
    cm1_run.stage(tmp_path / "staged")
    after = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
    assert before == after
    assert (tmp_path / "staged" / "cm1.exe").is_file()


# ---------------------------------------------------------------------------
# A stretched vertical coordinate on both sides
# ---------------------------------------------------------------------------


def test_ctl_reader_carries_an_explicit_level_list() -> None:
    """CM1 writes ``zdef N levels`` whenever the column is not uniform.

    The heights follow on their own lines.  A reader that only understood
    ``zdef N linear`` raised ``KeyError('z')`` on every stretched run, which is
    a stretched history that cannot be compared at all.
    """

    from cm1core.oracle.grads import read_ctl

    ctl = REPO / "evidence" / "oracle" / "stretched-column-32" / "cm1out_s.ctl"
    dataset = read_ctl(ctl)
    assert dataset.nz == 40
    assert len(dataset.levels) == 40
    # km in the descriptor; 100 m at the ground and 300 m aloft.
    assert round((dataset.levels[1] - dataset.levels[0]) * 1000.0, 3) == 100.0
    assert round((dataset.levels[-1] - dataset.levels[-2]) * 1000.0, 3) == 300.0
    # z0 is still the first height, so a caller that only wants one number has
    # one; dz is the first spacing and is explicitly not the column's spacing.
    assert dataset.z0 == dataset.levels[0]

    faces = read_ctl(ctl.with_name("cm1out_w.ctl"))
    assert faces.nz == 41 and faces.levels[0] == 0.0


def test_uniform_ctl_still_reads_as_linear() -> None:
    from cm1core.oracle.grads import read_ctl

    text = (
        "dset ^x_%y4.dat\noptions template\nundef -999999.9\n"
        "xdef 4 linear 0.5 1.0\nydef 4 linear 0.5 1.0\nzdef 3 linear 0.25 0.5\n"
        "tdef 2 linear 00:00Z01JAN0001 1YR\nvars 1\nth 3 99 theta\nendvars\n"
    )
    path = REPO / "evidence" / "oracle" / "stretched-column-32" / "cm1out_s.ctl"
    dataset = read_ctl(path)  # smoke: the stretched one still parses
    assert dataset.levels
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        uniform = Path(tmp) / "u.ctl"
        uniform.write_text(text)
        got = read_ctl(uniform)
    assert (got.nz, got.z0, got.dz) == (3, 0.25, 0.5)
    assert got.levels == ()


def test_the_model_top_comes_from_the_generator_on_a_stretched_column() -> None:
    """param.F:8193 maxz = sigmaf(nk+1), and zd is a height above it."""

    from cm1core.oracle.spec import CaseError, _model_top_m

    stretched = {
        "stretch_z": 1, "nz": 40, "dz": 200.0, "ztop": 8000.0,
        "str_bot": 1000.0, "str_top": 5000.0, "dz_bot": 100.0, "dz_top": 300.0,
    }
    assert _model_top_m(stretched) == 8000.0
    assert _model_top_m({"stretch_z": 0, "nz": 40, "dz": 500.0}) == 20000.0
    # A column param.F:7828-7877 would stop on does not silently become a top.
    with pytest.raises(CaseError, match="does not build"):
        _model_top_m({**stretched, "str_top": 20000.0})
    # param.F:8005-8064 reads its levels from a file, so the namelist does not
    # determine the top and the translation refuses rather than guess.
    for branch in (3, 4):
        with pytest.raises(CaseError, match="reads its levels from a file"):
            _model_top_m({**stretched, "stretch_z": branch})


def test_the_stretched_case_translates_and_gpuwm_cm1_admits_it() -> None:
    """Config text to admitted document, on the coordinate CM1 itself ran."""

    import tomllib

    from cm1core.cartesian_door import (
        build_runtime_config,
        cartesian_vertical_operator_record,
    )
    from cm1core.config import parse_mapping
    from cm1core.oracle.spec import load_case

    case = load_case(REPO / "examples" / "oracle" / "stretched-column-32.differential.toml")
    base = REPO / "evidence" / "oracle" / "stretched-column-32" / "namelist.input"
    document = tomllib.loads(case.gm1_toml(base))
    assert document["grid"]["stretch_z"] == 1
    # zd = 6000 m above ground on an 8 km top is a 2 km damping depth.
    assert document["closures"]["upper_damping_depth_m"] == 2000.0
    grid = build_runtime_config(parse_mapping(document)).grid
    assert grid.stretch_z == 1 and grid.ztop_m == 8000.0
    assert cartesian_vertical_operator_record(grid)["family"] == "profile-bound"


def test_an_explicit_level_case_writes_cm1s_own_input_grid_z(tmp_path) -> None:
    """param.F:8083 opens a file; the level list is declared once, on one side.

    CM1 has no namelist word for the column, so a case that declared it twice
    would be two columns waiting to drift apart.  The key map holds
    ``grid.levels_m`` as GM1-only and ``cm1_run`` writes CM1's ``input_grid_z``
    from it.
    """

    from cm1core.oracle import cm1_run
    from cm1core.oracle.namelist import read as read_namelist
    from cm1core.oracle.spec import load_case

    case = load_case(REPO / "examples" / "oracle" / "explicit-w-levels-32.differential.toml")
    base = REPO / "evidence" / "oracle" / "stretched-column-32" / "namelist.input"
    deck = case.namelist_deck(base)
    cm1_run._write_declared_levels(case, deck, tmp_path)
    written = [float(line) for line in (tmp_path / "input_grid_z").read_text().split()]
    # param.F:8086 reads nk+1 w levels, beginning at the ground.
    assert len(written) == 41 and written[0] == 0.0 and written[-1] == 8000.0

    # A list on a branch that reads no file is refused rather than written.
    unstretched = dict(deck)
    unstretched["param6"] = {**deck["param6"], "stretch_z": 0}
    with pytest.raises(cm1_run.CM1RunError, match="reads no level file"):
        cm1_run._write_declared_levels(case, unstretched, tmp_path)

    # And a branch that needs one but was given none.
    analytic = load_case(
        REPO / "examples" / "oracle" / "stretched-column-32.differential.toml"
    )
    deck4 = case.namelist_deck(base)
    with pytest.raises(cm1_run.CM1RunError, match="reads its levels from input_grid_z"):
        cm1_run._write_declared_levels(analytic, deck4, tmp_path)


def test_the_two_stretched_cases_describe_one_column() -> None:
    """The explicit case lists exactly what the analytic case generates.

    Anything the explicit differential then finds is the reading of the list,
    not the shape of the column, which is what makes it a test of
    param.F:8072-8132 rather than a second test of param.F:7819-7953.
    """

    import tomllib

    from cm1core.cartesian_door import require_grid_extent
    from cm1core.config import parse_mapping
    from cm1core.oracle.spec import load_case

    base = REPO / "evidence" / "oracle" / "stretched-column-32" / "namelist.input"
    grids = {}
    for name in ("stretched-column-32", "explicit-w-levels-32"):
        case = load_case(REPO / "examples" / "oracle" / f"{name}.differential.toml")
        grids[name] = require_grid_extent(
            parse_mapping(tomllib.loads(case.gm1_toml(base)))
        )
    analytic, explicit = grids["stretched-column-32"], grids["explicit-w-levels-32"]
    assert (analytic.stretch_z, explicit.stretch_z) == (1, 4)
    assert analytic.face_heights_m == explicit.face_heights_m
    assert analytic.mass_heights_m == explicit.mass_heights_m
    assert analytic.ztop_m == explicit.ztop_m == 8000.0


@requires_cm1
@pytest.mark.parametrize("name", ["balance-column-lspgrad1-cd005", "balance-column-lspgrad1-freeslip"])
def test_the_morrison_balance_columns_translate_to_a_document_the_door_admits(name: str) -> None:
    """A Morrison case gets no WSM6-only output from the harness, and keeps CM1's rain and hail swaths.

    HARNESS_OUTPUT asks CM1 for dbz and the rain/graupel swaths; GM1 publishes
    reflectivity only for a component whose formula it carries
    (CM1-CFG-DBZ-COMPONENT), so the emitted document switches dbz off rather
    than being refused.  The two swaths read CM1's qr and qg rows, which
    Morrison carries at CM1's nqr = 3 and nqg = 6 as WSM6 does
    (solve3.F:993-1047, param.F:3414-3473), so they stay on.  The WSM6
    worked example keeps all three.
    """

    import tomllib

    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import parse_mapping

    base = cm1_run.base_deck("supercell")
    document = tomllib.loads(load_case(REPO / "examples" / "oracle" / f"{name}.differential.toml").gm1_toml(base))
    assert document["physics"]["component"] == "morrison-mp10"
    assert document["physics"]["output_dbz"] is False
    assert document["diagnostics"]["output_srs"] == 1 and document["diagnostics"]["output_sgs"] == 1
    assert document["dynamics"]["coriolis"] == 1 and document["dynamics"]["lspgrad"] == 1
    assert all(document["boundaries"][side] == 1 for side in ("west", "east", "south", "north"))
    build_runtime_config(parse_mapping(document))

    # The worked example runs Morrison now (CM1 has no eqtset=2 WSM6,
    # src/param.F:936-948); the same case with WSM6 keeps all three.
    worked = load_case(CASE)
    wsm6_case = DifferentialCase(
        name=worked.name, term=worked.term, notes=worked.notes, cm1=worked.cm1,
        gm1_only={**worked.gm1_only, "physics.component": "wsm6-mp6"}, source=worked.source,
    )
    wsm6 = tomllib.loads(wsm6_case.gm1_toml(base))
    assert wsm6["physics"]["output_dbz"] is True
    # lspgrad=0 (CM1's shipped value) is left out, so every sealed case
    # document keeps the sha256 its receipts name.
    assert "lspgrad" not in wsm6["dynamics"]
