"""``cm1import.sounding``: input_sounding read as CM1 r21.1's base.F reads it.

CM1 reads a surface record (hPa, K, g/kg) and then one record per level;
it stops on a short record, on fewer than two levels, and on a top below the
model's highest scalar level (base.F:678-683), and it brackets model levels
between rising sounding heights.  Each is named here, before anything runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

EFFECTIVE = {"isnd": 7, "iwnd": 0, "nz": 40, "dz": 500.0, "stretch_z": 0, "imove": 0}


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "input_sounding"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _good(top: float = 20000.0, step: float = 250.0) -> list[str]:
    lines = ["1000.0 300.0 14.0"]
    z = step
    while z <= top + 1e-6:
        lines.append(f"{z:.1f} {300.0 + z / 250.0:.2f} {max(14.0 - z / 1000.0, 0.0):.3f} 5.0 0.0")
        z += step
    return lines


def _codes(check) -> list[str]:
    return [problem.code for problem in check.problems]


def test_isnd_5_reads_no_file() -> None:
    from cm1core.cm1import.sounding import check_sounding

    check = check_sounding(None, {**EFFECTIVE, "isnd": 5})
    assert not check.reads_file and check.problems == ()


def test_a_file_sounding_that_is_missing_is_named() -> None:
    from cm1core.cm1import.sounding import check_sounding

    assert _codes(check_sounding(None, EFFECTIVE)) == ["CM1-IMPORT-SOUNDING-MISSING"]


def test_a_good_sounding_passes_and_reports_its_extent(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    check = check_sounding(_write(tmp_path, _good()), EFFECTIVE)
    assert check.problems == () and check.warnings == ()
    assert check.surface == {"pressure_hpa": 1000.0, "theta_k": 300.0, "qv_g_per_kg": 14.0}
    assert check.heights_m[0] == 250.0 and check.heights_m[-1] == 20000.0
    # zh(nk) of a uniform 40 x 500 m column is 19750 m (param.F:7789-7811).
    assert check.highest_mass_level_m == pytest.approx(19750.0)


def test_a_top_below_the_model_is_refused_as_cm1_stops(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    check = check_sounding(_write(tmp_path, _good(top=15000.0)), EFFECTIVE)
    assert _codes(check) == ["CM1-IMPORT-SOUNDING-TOP"]
    assert "base.F:678-683" in check.problems[0].text and "19750" in check.problems[0].fix


def test_heights_that_do_not_rise_are_refused(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    lines = _good()
    lines[5], lines[6] = lines[6], lines[5]
    assert "CM1-IMPORT-SOUNDING-HEIGHTS" in _codes(check_sounding(_write(tmp_path, lines), EFFECTIVE))


def test_a_short_record_is_refused_with_its_line(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    lines = _good()
    lines[3] = "750.0 303.0 11.25"  # isnd = 7 reads five values
    check = check_sounding(_write(tmp_path, lines), EFFECTIVE)
    assert _codes(check)[0] == "CM1-IMPORT-SOUNDING-FORMAT" and check.problems[0].line == 4
    # isnd = 17 reads three per level: the same file is fine there.
    lines = ["1000.0 300.0 14.0"] + [" ".join(line.split()[:3]) for line in _good()[1:]]
    assert check_sounding(_write(tmp_path, lines), {**EFFECTIVE, "isnd": 17}).problems == ()


def test_one_level_is_too_few(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    check = check_sounding(_write(tmp_path, ["1000.0 300.0 14.0", "25000.0 400.0 0.0 5.0 0.0"]), EFFECTIVE)
    assert "CM1-IMPORT-SOUNDING-LEVELS" in _codes(check)


def test_unit_slips_are_warned_not_refused(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    lines = _good()
    lines[0] = "100000.0 300.0 0.014"
    lines[1:] = [" ".join([parts[0], parts[1], f"{float(parts[2]) / 1000:.6f}", parts[3], parts[4]])
                 for parts in (line.split() for line in lines[1:])]
    check = check_sounding(_write(tmp_path, lines), EFFECTIVE)
    assert check.problems == ()
    assert any("looks like Pa" in warning for warning in check.warnings)
    assert any("kg/kg" in warning for warning in check.warnings)


def test_fortran_d_exponents_and_commas_read(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding

    lines = ["1.0d3, 300.0, 14.0"] + [line.replace(" ", ", ") for line in _good()[1:]]
    assert check_sounding(_write(tmp_path, lines), EFFECTIVE).problems == ()


def test_the_environment_block_needs_the_initializer_library(tmp_path: Path) -> None:
    from cm1core.cm1import.sounding import check_sounding, environment

    check = environment(check_sounding(None, {**EFFECTIVE, "isnd": 5}), {**EFFECTIVE, "isnd": 5, "iwnd": 2},
                        library=tmp_path / "missing.so")
    assert check.environment_status == "not-computed" and check.environment_reason
    import os

    library = os.environ.get("GM1_TEST_INITIALIZER_LIBRARY")
    if not library:
        pytest.skip("set GM1_TEST_INITIALIZER_LIBRARY to a built libcm1_state.so to compute the block")
    check = environment(check_sounding(None, {**EFFECTIVE, "isnd": 5}), {**EFFECTIVE, "isnd": 5, "iwnd": 2},
                        library=library)
    assert check.environment_status == "computed"
    assert check.environment["schema"] == "gm1.sounding-parameters/v1"
