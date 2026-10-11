from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cm1core.cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_BYTES,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_BYTES,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_BYTES,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION,
    Gm1ChildPsolver3OpenSelection,
    _validate_selection,
    profile_operator_probe_f32,
)
from cm1core.cuda_stretch_z1_terrain import cpu_build_stretch_z1_terrain
from cm1core.stretch_z1_child_15km import build_stretch_z1_child_15km_grid


def _profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = build_stretch_z1_child_15km_grid()
    terrain = cpu_build_stretch_z1_terrain(
        np.zeros((3, 3), dtype=np.float32), allow_compact_validation=True
    )
    return (
        grid.sigma,
        grid.sigmaf,
        terrain.arrays["rds"],
        terrain.arrays["rdsf"],
    )


def test_generated_sources_are_pinned_and_have_no_scalar_vertical_fallback() -> None:
    assert CUDA_GM1_CHILD_PSOLVER3_OPEN_ABI in CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE
    assert (
        hashlib.sha256(CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_BYTES).hexdigest()
        == CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256
    )
    assert (
        hashlib.sha256(CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_BYTES).hexdigest()
        == CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
    )
    for source in (
        CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE,
        CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE,
    ):
        assert "float dz" not in source
        assert "1.0f/dz" not in source
        assert "rds(dz)" not in source


def test_probe_matches_sealed_stretched_profiles_raw_bits() -> None:
    sigma, sigmaf, rds, rdsf = _profiles()
    probe = profile_operator_probe_f32(
        sigma=sigma, sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=37
    )
    words = np.asarray(list(probe.values()), dtype=np.float32).view(np.uint32)
    assert words.tolist() == [
        0x3F024712,
        0x3EFB71DD,
        0xBF490B9B,
        0x3B979883,
        0x3B94F214,
    ]


def test_stretched_probe_refuses_uniform_dz_substitution() -> None:
    sigma, sigmaf, rds, rdsf = _profiles()
    probe = profile_operator_probe_f32(
        sigma=sigma, sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=37
    )
    assert np.float32(probe["r2"]).view(np.uint32) != np.float32(0.5).view(np.uint32)
    assert np.float32(probe["rds"]).view(np.uint32) != np.float32(0.01).view(np.uint32)


def test_probe_refuses_wrong_profile_shape_and_top() -> None:
    sigma, sigmaf, rds, rdsf = _profiles()
    with pytest.raises(ValueError, match="sigma must be"):
        profile_operator_probe_f32(
            sigma=sigma[1:], sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=37
        )
    with pytest.raises(ValueError, match="15000"):
        profile_operator_probe_f32(
            sigma=sigma,
            sigmaf=sigmaf,
            rds=rds,
            rdsf=rdsf,
            k=37,
            ztop=17650.0,
        )


def test_selection_fails_closed() -> None:
    _validate_selection(DEFAULT_GM1_CHILD_PSOLVER3_OPEN_SELECTION)
    with pytest.raises(ValueError, match="sealed"):
        _validate_selection(Gm1ChildPsolver3OpenSelection(mpi=True))
    with pytest.raises(TypeError, match="selection"):
        _validate_selection(object())  # type: ignore[arg-type]


def test_real_full_and_native_evidence_is_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    validation = json.loads(
        (
            root
            / "evidence/validation/gm1-child-psolver3-open-stretch-z1-v1/validation-receipt.json"
        ).read_text(encoding="utf-8")
    )
    native = json.loads(
        (
            root
            / "evidence/oracles/gm1-child-psolver3-open-stretch-z1-v1/native-receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS_REAL_FULL_AND_COMPACT"
    assert validation["gpu_mutex"]["clean_release"] is True
    assert validation["profile_probe"]["passed"] is True
    assert validation["compact_stage"]["passed"] is True
    assert validation["full_real_stage"]["status"] == "PASS_REAL_300X300X75_RK1"
    assert all(validation["full_real_stage"]["all_finite"].values())
    assert all(validation["full_real_stage"]["active_outputs"].values())
    assert (
        validation["kernels"]["core_source_sha256"]
        == CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256
    )
    assert (
        validation["kernels"]["boundary_source_sha256"]
        == CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
    )
    assert native["status"] == "PASS_O0_O2_RAW_BIT"
    assert native["outputs"]["O0"] == native["expected_words_hex"]
    assert native["outputs"]["O2"] == native["expected_words_hex"]


def test_the_probe_is_column_generic_and_binds_each_columns_own_top() -> None:
    """The operator is not the 15 km child; it is whatever column it is given."""

    from cm1core.cm1_vertical_grid import uniform_vertical_levels

    levels = uniform_vertical_levels(40, 500.0)
    sigma = np.ascontiguousarray(np.asarray(levels.sigma, dtype=np.float32))
    sigmaf = np.ascontiguousarray(np.asarray(levels.sigmaf, dtype=np.float32))
    rds = np.ascontiguousarray(np.asarray(levels.rds, dtype=np.float32))
    rdsf = np.ascontiguousarray(np.asarray(levels.rdsf, dtype=np.float32))
    probe = profile_operator_probe_f32(
        sigma=sigma, sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=20, ztop=20000.0, nk=40,
    )
    # param.F:8162-8209 on a uniform column: the face sits exactly halfway
    # between the two scalar levels it separates.
    assert probe["r1"] == pytest.approx(0.5)
    assert probe["r2"] == pytest.approx(0.5)
    assert probe["rdsf"] == pytest.approx(1.0 / 500.0)
    # param.F:8193: the top the operator binds is this column's own top face.
    with pytest.raises(ValueError, match="sigmaf"):
        profile_operator_probe_f32(
            sigma=sigma, sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=20, ztop=15000.0, nk=40,
        )
    with pytest.raises(ValueError, match="k must be 2..40"):
        profile_operator_probe_f32(
            sigma=sigma, sigmaf=sigmaf, rds=rds, rdsf=rdsf, k=41, ztop=20000.0, nk=40,
        )


def test_every_generated_branch_reaches_the_operator_and_no_other_word_widens() -> None:
    from cm1core.cm1_vertical_grid import ADMITTED_STRETCH_Z

    for branch in ADMITTED_STRETCH_Z:
        _validate_selection(Gm1ChildPsolver3OpenSelection(stretch_z=branch))
    with pytest.raises(ValueError, match="stretch_z must be one of"):
        _validate_selection(Gm1ChildPsolver3OpenSelection(stretch_z=2))
    for change in (
        {"psolver": 2},
        {"boundaries": (1, 2, 2, 2)},
        {"terrain": False},
        {"axisymmetric": 1},
        {"mpi": True},
        {"product_admission": False},
    ):
        with pytest.raises(ValueError, match="sealed"):
            _validate_selection(Gm1ChildPsolver3OpenSelection(**change))


def test_the_domain_is_admitted_by_its_stencils_not_by_a_fixture() -> None:
    """nk=75, 300x300, dx=dy=50 m and dt=0.5 s were a fixture, not a bound."""

    from cm1core.cuda_gm1_child_psolver3_open import (
        launch_cuda_gm1_child_psolver3_open_stage,
    )

    def launch(**overrides: object) -> None:
        arguments: dict[str, object] = {
            "arrays": {}, "metrics": {}, "ni": 64, "nj": 64, "nk": 40,
            "nrk": 3, "dt": 1.0, "nsound": 6, "dx": 250.0, "dy": 250.0,
            "vertical": object(),
        }
        arguments.update(overrides)
        launch_cuda_gm1_child_psolver3_open_stage(**arguments)  # type: ignore[arg-type]

    # param.F:8985-9040 fits four scalar levels at each end of the column.
    # These structural refusals all precede the CuPy import.
    with pytest.raises(ValueError, match="nk >= 4"):
        launch(nk=3)
    with pytest.raises(ValueError, match=r"ni,nj >= 3"):
        launch(ni=2)
    for name in ("dt", "dx", "dy"):
        with pytest.raises(ValueError):
            launch(**{name: 0.0})
    with pytest.raises(ValueError, match="nrk"):
        launch(nrk=4)
    # A 64x64x40 domain at 250 m and dt=1 s is past every structural gate; the
    # operator now fails on the empty carrier map, which is the next real
    # check.  Before this change it failed on the domain itself.  The carrier
    # map is checked only after the operator imports CuPy (the empty-map
    # refusal itself touches no device), so this last check needs CuPy.
    pytest.importorskip("cupy")
    with pytest.raises(ValueError, match="missing CUDA terrain/open sound arrays"):
        launch()


# ---------------------------------------------------------------------------
# The compact core: the same solve without sound.F:659-730's column report
# ---------------------------------------------------------------------------

def _entry_points(source: str) -> dict[str, str]:
    import re

    found = re.findall(
        r'extern "C" __global__ void ([A-Za-z0-9_]+)\((.*?)\)\{', source, re.S
    )
    assert found
    return {name: " ".join(parameters.split()) for name, parameters in found}


def test_the_compact_core_is_the_same_kernels_without_the_column_stores() -> None:
    """``tri_aa``..``tri_residual`` are a report, not part of the solve.

    The Cartesian lane owns them at one word each because on its production
    domain they are 762 MB that nothing reads, so this operator has to be
    compilable without them -- and that variant is only legitimate if it is the
    sealed source's own words with the diagnostic stores removed.  Same entry
    points, same parameter lists, same live elimination storage; strictly
    fewer bytes.  The only ``ec`` write it lacks is the update's dead
    ``ec[p]=tend`` (tests/test_acoustic_compact_dead_stores.py).
    """

    assert (
        hashlib.sha256(CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_BYTES).hexdigest()
        == CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256
    )
    assert (
        CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE_SHA256
        != CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256
    )
    assert _entry_points(CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE) == _entry_points(
        CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE
    )
    assert len(CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE) < len(
        CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE
    )
    # The stores that go, and the elimination storage that stays.
    assert "aa0[p]=aa;bb0[p]=bb;" in CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE
    assert "aa0[p]=aa;bb0[p]=bb;" not in CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE
    assert "res[p]=" not in CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE
    for live in ("ec[p]=m32(-cc,inv);", "ed[p]=m32("):
        assert live in CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE
    for dead in ("divh[", "hadvp[", "dum3[", "ec[p]=tend;"):
        assert dead not in CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE
    # Both variants remain profile-bound; neither may rebuild a level from dz.
    for token in ("float dz", "1.0f/dz", "rds(dz)", "sigf("):
        assert token not in CUDA_GM1_CHILD_PSOLVER3_OPEN_COMPACT_SOURCE


def test_the_retention_choice_is_a_boolean_and_nothing_else() -> None:
    """A truthy string would compile the wrong image and be admitted silently."""

    from cm1core.cuda_gm1_child_psolver3_open import (
        compile_cuda_gm1_child_psolver3_open,
    )

    for value in ("false", 0, 1, None):
        with pytest.raises(TypeError, match="retain_column_diagnostics"):
            compile_cuda_gm1_child_psolver3_open(retain_column_diagnostics=value)
