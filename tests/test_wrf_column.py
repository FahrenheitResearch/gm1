"""One column of a WRF or ArWen run as CM1's input_sounding (cm1core.cm1import.wrfcolumn), host only.

A hydrostatic synthetic column is reproduced (theta, qv, z, the surface line and
the winds rotated by a known alpha); destaggering is exact on linear fields; the written file passes
cm1import.sounding and CM1's own hydrostatic reconstruction matches P+PB within 0.5 hPa.  Also: USE_THETA_M with
THM, missing surface fields, rotation rules per projection, the point on the map, several times and a d02 file,
the model-top check, and the 0-6 km shear map.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("netCDF4")

from cm1core import wrf_history as wh  # noqa: E402
from cm1core.cm1import import wrfcolumn as wc  # noqa: E402
from cm1core.cm1import.sounding import check_sounding  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402

import wrf_support as ws  # noqa: E402


def column(path, record=0, **kw):
    with wh.open_history(path) as history:
        return wc.extract_column(history, record, **kw)


def refused(code, fn, *a, **kw):
    with pytest.raises(CM1Refusal) as caught:
        fn(*a, **kw)
    assert caught.value.detail.code == code, caught.value.detail
    return caught.value.detail


@pytest.mark.parametrize("fmt", ["CDF-2", "netCDF-4"])
@pytest.mark.parametrize("alpha_deg", [0.0, 25.0, -40.0])
def test_a_hydrostatic_synthetic_column_is_reproduced(tmp_path, fmt, alpha_deg):
    path = tmp_path / "wrfout_d01"
    facts = ws.write_wrfout(path, fmt=fmt, alpha_deg=alpha_deg, wind_slope=0.0)
    col = column(path, ij=(4, 3))
    j, i = 2, 3
    zm = facts["zm"][:, j, i]
    assert np.allclose(col.z_m, zm, atol=2e-3)
    assert np.allclose(col.theta_k, ws.theta(zm), atol=2e-4)
    assert np.allclose(col.qv_kgkg, ws.qv(zm), rtol=1e-6, atol=1e-9)
    ue, ve = ws.earth_winds_at_mass(facts, 0, j, i)
    assert np.allclose(col.u_ms, ue, atol=1e-4) and np.allclose(col.v_ms, ve, atol=1e-4)
    assert col.rotation["alpha_deg"] == pytest.approx(alpha_deg, abs=1e-4)
    s = col.surface
    assert s["pressure_hpa"] == pytest.approx(facts["psfc"][j, i] / 100.0, abs=1e-3)
    assert s["theta_k"] == pytest.approx(float(ws.theta(0.0)), abs=1e-4)
    assert s["qv_gkg"] == pytest.approx(float(ws.qv(0.0)) * 1000.0, rel=1e-6)
    assert s["sources"] == {"pressure": "PSFC", "theta": "TH2", "qv": "Q2"}
    ue10, ve10 = ws.wind(10.0)
    assert s["u10_ms"] == pytest.approx(float(ue10), abs=1e-4) and s["v10_ms"] == pytest.approx(float(ve10), abs=1e-4)
    assert col.point == {"requested": {"i": 4, "j": 3}, "i": 4, "j": 3, "distance_km": 0.0,
                         "lat": pytest.approx(float(facts["lat"][j, i])), "lon": pytest.approx(float(facts["lon"][j, i])),
                         "terrain_m": 0.0}


def test_destaggering_is_exact_on_linear_fields(tmp_path):
    path = tmp_path / "wrfout_d01"
    facts = ws.write_wrfout(path, alpha_deg=0.0, wind_slope=0.75)
    with wh.open_history(path) as history:
        for (i, j) in ((1, 1), (5, 4), (8, 7)):
            col = wc.extract_column(history, 0, ij=(i, j))
            ue, ve = ws.earth_winds_at_mass(facts, 0, j - 1, i - 1)
            # U = u(z) + 0.75 * i_stag: the mean of faces i-1 and i (0-based i-1, i) is u(z) + 0.75 (i - 0.5)
            assert np.array_equal(col.u_ms.astype(np.float32), ue.astype(np.float32)) or \
                np.max(np.abs(col.u_ms - ue)) < 5e-6
            assert np.max(np.abs(col.v_ms - ve)) < 5e-6


def test_the_written_file_passes_cm1s_reader_and_its_hydrostatics(tmp_path):
    path = tmp_path / "wrfout_d01"
    ws.write_wrfout(path, terrain_m=300.0, alpha_deg=15.0, nz=40)
    for ij in ((1, 1), (8, 7)):
        col = column(path, ij=ij)
        out = tmp_path / f"input_sounding_{ij[0]}"
        out.write_bytes(col.sounding_bytes())
        check = check_sounding(out, {"isnd": 7, "nz": 40, "dz": 500.0})
        assert not check.problems, check.problems
        assert not check.warnings, check.warnings
        assert check.surface["pressure_hpa"] == pytest.approx(col.surface["pressure_hpa"], abs=1e-4)
        assert len(check.heights_m) == 40 and check.heights_m[-1] > 19750.0
        z, p = ws.cm1_reconstructed_pressure(out.read_text())
        diff = np.abs(p[1:] - col.p_pa) / 100.0
        assert diff.max() < 0.5, diff.max()
        hydro = col.hydrostatic()
        assert hydro["max_abs_hpa"] == pytest.approx(float(diff.max()), abs=1e-6)
        assert hydro["max_abs_hpa"] < 0.5
        record = col.record()
        assert record["schema"] == wc.SCHEMA and record["sounding"]["bytes"] == len(col.sounding_bytes())
        assert record["point"]["terrain_m"] == pytest.approx(0.0 if ij[0] == 1 else 300.0, abs=1e-3)
        assert "#" not in out.read_text()          # CM1 stops on a comment line


def test_theta_m(tmp_path):
    good = tmp_path / "good"
    ws.write_wrfout(good, use_theta_m=1, thm="dry")
    col = column(good, ij=(2, 2))
    assert col.theta_m["USE_THETA_M"] == 1 and col.theta_m["dry_T_residual_K"] < 1e-2
    moist = tmp_path / "moist"
    ws.write_wrfout(moist, use_theta_m=1, thm="moist")
    detail = refused("GM1-WRF-THETA-M", column, moist, ij=(2, 2))
    assert "dry potential temperature" in detail.reason
    none = tmp_path / "none"
    ws.write_wrfout(none, use_theta_m=1, thm="none")
    refused("GM1-WRF-THETA-M", column, none, ij=(2, 2))
    zero = tmp_path / "zero"
    ws.write_wrfout(zero, use_theta_m=0, thm="none")
    assert column(zero, ij=(2, 2)).theta_m == {"USE_THETA_M": 0}


def test_missing_surface_fields_are_filled_by_named_rules(tmp_path):
    path = tmp_path / "nosfc"
    facts = ws.write_wrfout(path, drop=("PSFC", "TH2", "T2", "Q2", "U10", "V10"))
    col = column(path, ij=(3, 3))
    s = col.surface
    assert s["sources"]["pressure"].startswith("derived")
    assert s["sources"]["theta"].startswith("derived") and s["sources"]["qv"].startswith("derived")
    # one Exner step from the lowest level (about 26 m) lands within 0.05 hPa of the true surface pressure
    assert s["pressure_hpa"] == pytest.approx(facts["psfc"][2, 2] / 100.0, abs=0.05)
    assert s["theta_k"] == pytest.approx(float(col.theta_k[0]))
    assert "u10_ms" not in s
    t2only = tmp_path / "t2only"
    ws.write_wrfout(t2only, with_th2=False)
    col = column(t2only, ij=(3, 3))
    assert col.surface["sources"]["theta"].startswith("T2")
    assert col.surface["theta_k"] == pytest.approx(float(ws.theta(0.0)), abs=1e-3)
    nothing = tmp_path / "nothing"
    ws.write_wrfout(nothing, drop=("PSFC", "P", "PB"))
    refused("GM1-WRF-SURFACE", column, nothing, ij=(3, 3))
    for name in ("U", "PH", "QVAPOR"):
        missing = tmp_path / f"no{name}"
        ws.write_wrfout(missing, drop=(name,))
        refused("GM1-WRF-VARIABLE", column, missing, ij=(3, 3))


def test_rotation_rules_per_projection(tmp_path):
    lam = tmp_path / "lambert"
    ws.write_wrfout(lam, map_proj=1, alpha_deg=10.0, drop=("SINALPHA", "COSALPHA"))
    refused("GM1-WRF-ROTATION", column, lam, ij=(2, 2))
    for proj in (0, 3):
        path = tmp_path / f"p{proj}"
        facts = ws.write_wrfout(path, map_proj=proj)
        col = column(path, ij=(2, 2))
        assert col.rotation["alpha_deg"] == 0.0 and "earth-aligned" in col.rotation["source"]
        ue, ve = ws.earth_winds_at_mass(facts, 0, 1, 1)
        assert np.max(np.abs(col.u_ms - ue)) < 1e-4
    polar = tmp_path / "polar"
    facts = ws.write_wrfout(polar, map_proj=2, alpha_deg=-70.0)
    col = column(polar, ij=(2, 2))
    ue, ve = ws.earth_winds_at_mass(facts, 0, 1, 1)
    assert np.max(np.abs(col.u_ms - ue)) < 1e-4 and np.max(np.abs(col.v_ms - ve)) < 1e-4


def test_the_point_on_the_map(tmp_path):
    import netCDF4

    path = tmp_path / "wrfout_d01"
    facts = ws.write_wrfout(path, nx=10, ny=9)
    lat, lon = facts["lat"], facts["lon"]
    col = column(path, at=(float(lat[5, 7]) + 0.004, float(lon[5, 7]) - 0.004))
    assert (col.point["i"], col.point["j"]) == (8, 6)
    assert col.point["distance_km"] < 1.0
    refused("GM1-WRF-POINT", column, path, at=(float(lat.max()) + 1.0, float(lon[0, 0])))
    refused("GM1-WRF-POINT", column, path, ij=(11, 1))
    refused("GM1-WRF-POINT", column, path, ij=(0, 1))
    ideal = tmp_path / "ideal"
    ws.write_wrfout(ideal, map_proj=0)
    ds = netCDF4.Dataset(str(ideal), "a")
    ds["XLAT"][:] = 0.0
    ds["XLONG"][:] = 0.0
    ds.close()
    detail = refused("GM1-WRF-POINT", column, ideal, at=(0.0, 0.0))
    assert "grid point" in detail.remedy
    assert column(ideal, ij=(2, 2)).point["i"] == 2


def test_several_times_and_a_d02_file(tmp_path):
    path = tmp_path / "wrfout_d02_2026-05-01_18:00:00"
    facts = ws.write_wrfout(path, grid_id=2, records=3, interval_s=900.0)
    with wh.open_history(path) as history:
        assert history.domain_id == 2
        cols = [wc.extract_column(history, r, ij=(3, 3)) for r in range(3)]
    for r, col in enumerate(cols):
        ue, _ = ws.earth_winds_at_mass(facts, r, 2, 2)
        assert np.max(np.abs(col.u_ms - ue)) < 1e-4
        assert col.source["record"] == r and col.source["times"] == facts["times"][r]
        assert col.source["domain_id"] == 2
    assert cols[2].u_ms[0] - cols[0].u_ms[0] == pytest.approx(2.0, abs=1e-4)


def test_the_top_must_reach_the_models_highest_level(tmp_path):
    path = tmp_path / "wrfout_d01"
    ws.write_wrfout(path)
    col = column(path, ij=(2, 2), need_top_m=19750.0)
    assert col.top_m > 19750.0
    detail = refused("GM1-WRF-TOP", column, path, ij=(2, 2), need_top_m=25000.0)
    assert "zmax of sounding" in detail.reason


def test_negative_vapour_is_written_as_zero_and_counted(tmp_path):
    import netCDF4

    path = tmp_path / "wrfout_d01"
    ws.write_wrfout(path)
    ds = netCDF4.Dataset(str(path), "a")
    q = ds["QVAPOR"][0]
    q[-3:, 1, 1] = -1e-7
    ds["QVAPOR"][0] = q
    ds.close()
    col = column(path, ij=(2, 2))
    assert col.negative_qv_levels == 3
    assert "-" not in "".join(line.split()[2] for line in col.sounding_text().splitlines()[1:])


def test_the_shear_map(tmp_path):
    path = tmp_path / "wrfout_d01"
    facts = ws.write_wrfout(path, alpha_deg=30.0, terrain_m=500.0)
    with wh.open_history(path) as history:
        shear = wc.shear_map(history, 1)
    zm = facts["zm"]
    for (j, i) in ((0, 0), (3, 5), (6, 7)):
        u6, v6 = ws.wind(6000.0, 1)
        u1, v1 = ws.wind(zm[0, j, i], 1)
        # the map interpolates linearly between levels; the analytic wind is smooth, so within 0.2 m/s
        assert shear[j, i] == pytest.approx(math.hypot(u6 - u1, v6 - v1), abs=0.2)


def test_environment_numbers_when_the_library_is_installed(tmp_path):
    lib = Path(wc.__file__).resolve().parents[1] / "lib" / "libcm1_state.so"
    if not lib.is_file():
        pytest.skip("no built cm1-state library on this machine")
    path = tmp_path / "wrfout_d01"
    ws.write_wrfout(path, nz=40)
    col = column(path, ij=(2, 2))
    report, why = wc.environment_report(col, lib)
    assert why is None, why
    numbers = wc.summary_numbers(report)
    assert numbers["cape_sb"] > 500.0          # 14 g/kg at the ground under 4 K/km: a moist, unstable column
    ue1, ve1 = ws.wind(1000.0)
    ue0, ve0 = ws.wind(float(col.z_m[0]))
    assert numbers["shear_0_1km"] > 0.0


def test_follow_the_storm_moves_the_frame_the_storm_track_check_asks_for(tmp_path):
    from cm1core.cm1import.door import follow_the_storm

    path = tmp_path / "wrfout_d01"
    ws.write_wrfout(path, nz=40)
    col = column(path, ij=(2, 2))
    effective = {"nx": 120, "ny": 120, "dx": 1000.0, "dy": 1000.0, "timax": 7200.0, "imove": 1, "umove": -10.0,
                 "vmove": -10.0, "wbc": 2, "ebc": 2, "sbc": 2, "nbc": 2, "iorigin": 2}
    plan = follow_the_storm(effective, col.sounding_bytes())
    assert plan["action"] == "follow" and plan["stays_inside"] is True
    assert plan["umove"] * 8 == int(plan["umove"] * 8) and plan["vmove"] * 8 == int(plan["vmove"] * 8)
    kept = follow_the_storm({**effective, "umove": plan["umove"], "vmove": plan["vmove"]}, col.sounding_bytes())
    assert kept["action"] in ("keep", "follow")
    periodic = follow_the_storm({**effective, "wbc": 1, "ebc": 1, "sbc": 1, "nbc": 1}, col.sounding_bytes())
    assert periodic["action"] == "keep"
