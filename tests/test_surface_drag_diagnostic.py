"""CM1's wind-dependent ocean drag (``cecd = 2`` and ``3``) in the ground-layer diagnostic.

A run with ``bbc = 3``, ``sfcmodel = 1`` and ``cecd = 3`` (CM1's hurricane case, and CM1's supercell with drag as
its namelist has it) stopped before its first step: the swath and vorticity diagnostics rebuilt the surface layer
for the constant drag only and refused any other.  The diagnostic now takes the roughness length the run's own
``getcecd`` left in its state, and the import still refuses the setting until a GPU run has shown it.

CPU only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cm1core.cm1import import coverage
from cm1core.cm1run import presets as P
from cm1core.cm1run.fit import set_namelist_values
from cm1import_support import namelist_tool

HURRICANE = "hurricane-8km"


@pytest.fixture(autouse=True)
def _parser(monkeypatch):
    namelist_tool()
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def _guided(namelist_text: str, tmp_path: Path, name: str, sounding: Path | None = None) -> dict:
    """The import report's guided block for a namelist, as ``presets.examine`` reads it."""

    path = tmp_path / f"{name}.namelist.input"
    path.write_text(namelist_text, encoding="ascii")
    report = coverage.build_report(coverage.import_for_report(path), environment=False, sounding=sounding)
    return {**report.guided(), "exit": report.exit_code}


def _column(nz: int, dz: float):
    from cm1core.tornado_diagnostics import VerticalColumn

    zh = tuple((k + 0.5) * dz for k in range(nz))
    zf = tuple(k * dz for k in range(nz + 1))
    one = tuple(1.0 for _ in range(nz))
    return VerticalColumn(zh, zf, one, one + (1.0,), 1.0 / dz, (-2.0, 3.0, -1.0), (2.0, -3.0, 1.0))


def test_the_ground_layer_diagnostic_takes_the_runs_own_roughness_for_a_wind_dependent_drag():
    """``cecd = 3`` stopped a run before its first step: the swath and vorticity diagnostics rebuilt the surface
    layer for the constant drag only.  Given the roughness length getcecd left in the run's own state, the
    diagnostic's friction velocity is getcecd's own (sfcphys.F:190), checked here against the CPU transcription
    of getcecd on the same lowest-level wind."""

    from cm1core.surface_exchange_coefficients import (
        SurfaceExchangeSelection, cpu_surface_exchange_coefficients_inplace,
    )
    from cm1core.tornado_diagnostics import (
        CM1_KARMAN, LowerBoundaryState, surface_layer_state, vorticity_components,
    )

    rng = np.random.default_rng(7)
    nz, ny, nx, dz = 6, 9, 11, 50.0
    column = _column(nz, dz)
    za = column.zh_m[0]
    # Winds from calm to hurricane force, so the drag is on both plateaus of sfcphys.F's formula and its ramp.
    u = (rng.standard_normal((nz, ny, nx + 1)) * 22.0).astype(np.float32)
    v = (rng.standard_normal((nz, ny + 1, nx)) * 22.0).astype(np.float32)
    w = rng.standard_normal((nz + 1, ny, nx)).astype(np.float32)
    constant = LowerBoundaryState(bbc=3, tbc=1, cecd=1, cnstcd=0.001)
    wind = surface_layer_state(np, u=u, v=v, lower=constant, za_m=za)
    assert float(wind["s1"].min()) < 5.0 and float(wind["s1"].max()) > 25.0
    for cecd in (2, 3):
        arrays = {name: np.zeros((ny, nx), dtype=np.float32)
                  for name in ("u10", "v10", "s10", "ust", "cd", "ch", "cq")}
        arrays.update({"u1": wind["u1"].copy(), "v1": wind["v1"].copy(), "s1": wind["s1"].copy(),
                       "za": np.full((ny, nx), za, dtype=np.float32),
                       "xland": np.full((ny, nx), 2.0, dtype=np.float32),      # water everywhere
                       "znt": np.full((ny, nx), 1.0e-4, dtype=np.float32)})
        cpu_surface_exchange_coefficients_inplace(arrays, ni=nx, nj=ny, selection=SurfaceExchangeSelection(
            set_znt=0, set_ust=0, cecd=cecd, isfcflx=1, use_avg_sfc=False, cnst_ust_m_s=0.25, cnstcd=0.001,
            cnstce=0.001, avgsfcu_m_s=0.0, avgsfcv_m_s=0.0, avgsfcs_m_s=0.0))
        lower = LowerBoundaryState(bbc=3, tbc=1, cecd=cecd, cnstcd=0.001)
        state = surface_layer_state(np, u=u, v=v, lower=lower, za_m=za, znt=arrays["znt"])
        assert np.array_equal(state["s1"], arrays["s1"])
        np.testing.assert_allclose(state["ust"], arrays["ust"], rtol=4e-6)
        np.testing.assert_allclose(state["cd"], arrays["cd"], rtol=2e-4)
        if cecd == 3:
            # Fairall and Donelan's limits (sfcphys.F:5-9): 1.0e-3 below 5 m/s, 2.4e-3 above 25 m/s.
            assert float(state["cd"].min()) == pytest.approx(1.0e-3, rel=1e-3)
            assert float(state["cd"].max()) == pytest.approx(2.4e-3, rel=1e-3)
        out = vorticity_components(np, u=u, v=v, w=w, column=column, rdx=np.float32(1 / 8000.0),
                                   rdy=np.float32(1 / 8000.0), lower=lower, surface_znt=arrays["znt"])
        scale = state["ust"] / ((state["znt"] + np.float32(za)) * np.float32(CM1_KARMAN))
        speed = np.maximum(state["s1"], np.float32(0.01))
        assert np.array_equal(out["xvort"][0], np.float32(-1.0) * scale * (state["v1"] / speed))
        assert np.array_equal(out["yvort"][0], scale * (state["u1"] / speed))
        assert np.all(np.isfinite(out["xvort"])) and np.all(np.isfinite(out["yvort"]))
        # Without the run's roughness it is refused as before: none is invented here.
        with pytest.raises(ValueError, match="wind-dependent"):
            surface_layer_state(np, u=u, v=v, lower=lower, za_m=za)
        with pytest.raises(ValueError, match="wind-dependent"):
            vorticity_components(np, u=u, v=v, w=w, column=column, rdx=np.float32(1 / 8000.0),
                                 rdy=np.float32(1 / 8000.0), lower=lower)
        with pytest.raises(ValueError, match="whole-domain plane"):
            surface_layer_state(np, u=u, v=v, lower=lower, za_m=za, znt=arrays["znt"][:, :-1])
        with pytest.raises(ValueError, match="positive everywhere"):
            surface_layer_state(np, u=u, v=v, lower=lower, za_m=za, znt=np.zeros((ny, nx), dtype=np.float32))
    # The constant drag is what it was, and reads no roughness plane.
    again = surface_layer_state(np, u=u, v=v, lower=constant, za_m=za, znt=None)
    assert isinstance(again["znt"], float) and np.array_equal(again["ust"], wind["ust"])


def test_the_import_still_refuses_the_wind_dependent_drag_until_a_gpu_run_shows_it(tmp_path):
    hurricane = P.load(HURRICANE)
    for cecd in (2, 3):
        guided = _guided(set_namelist_values(hurricane.namelist_text, {"cecd": cecd}), tmp_path, f"cecd{cecd}",
                         hurricane.sounding_path)
        assert guided["exit"] != 0 and guided["offered"] is False
