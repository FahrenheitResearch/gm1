"""The ArWen chain on real-WRF-shaped histories, host only.

Histories are written by ``tests/wrf_support.py`` (a physically consistent synthetic atmosphere) in the
containers WRF and ArWen use, with the species each scheme carries added, and read through the one wrfout
reader, :mod:`cm1core.wrf_history`.  What is covered:

* a WRF-like netCDF-4 history -- three records in one file, ``USE_THETA_M = 1`` with THM, domain ``d02`` --
  is admitted for WSM6 (6), Thompson (8) and Morrison (10), and prepared;
* an idealized (``MAP_PROJ = 0``) history is admitted as GM1's own geometry, with the Coriolis
  parameter from its ``F`` when it has one and none when it has neither ``F`` nor ``XLAT``;
* other schemes are refused with the scheme named (WSM3, P3, NSSL);
* WRF's and ISO time spellings name the same time; frames are selected by time across records;
* the surface line falls back to WRF's own surface-pressure extrapolation and the lowest level;
* the admission report goes on past a refusal and says what it could not check;
* GM1's top must be under the history's last level (the Exner extension the first ArWen device run met);
* a record read straight from the file and the same record extracted to a frame directory prepare the same
  arenas word for word.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
netCDF4 = pytest.importorskip("netCDF4")

from cm1core import arwen_chain as chain  # noqa: E402
from cm1core import chain_door as door  # noqa: E402
from cm1core.cm1_vertical_grid import uniform_vertical_levels  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.moisture_species import moisture_species_profile  # noqa: E402

import wrf_support as ws  # noqa: E402

START = "2026-05-01_18:00:00"
BOX = "5,5,5,5"


def history(tmp_path: Path, *, mp: int = 6, fmt: str = "netCDF-4", producer: str = "WRF", map_proj: int = 1,
            name: str = "wrfout_d02_2026-05-01_18:00:00", species_for: str | None = None, records: int = 3,
            f_value: float | None = None, drop: tuple[str, ...] = (), start: str = START, **kwargs) -> Path:
    """A 16 x 16 x 30 history at 3 km, three records 300 s apart, with every species of the scheme."""

    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    ws.write_wrfout(path, fmt=fmt, producer=producer, nx=16, ny=16, nz=30, records=records, interval_s=300.0,
                    start=start, map_proj=map_proj, mp_physics=mp, grid_id=2, drop=drop, **kwargs)
    component = species_for or chain.CHAIN_SCHEMES.get(mp, "wsm6-mp6")
    with netCDF4.Dataset(path, "a") as ds:
        for species in moisture_species_profile(component).names:
            wrf = chain.WRF_SPECIES[species]
            if wrf not in ds.variables and wrf not in drop:
                v = ds.createVariable(wrf, "f4", ("Time", "bottom_top", "south_north", "west_east"))
                for r in range(records):
                    v[r] = np.zeros((30, 16, 16), np.float32)
        if f_value is not None:
            v = ds.createVariable("F", "f4", ("Time", "south_north", "west_east"))
            for r in range(records):
                v[r] = np.full((16, 16), f_value, np.float32)
    return path


def admitted(path: Path, **choice):
    frames = door.open_frames([path])
    made, chosen = door.make_choice(frames, window=BOX, ratio=3, **{"dt_s": None, **choice})
    return door.check(chosen, made, price_it=False), made, chosen


def row(payload, name):
    return next(r for r in payload["checks"] if r["id"] == name)


@pytest.mark.parametrize("mp", [6, 8, 10])
def test_a_wrf_netcdf4_history_with_theta_m_three_records_d02_is_admitted(tmp_path, mp):
    path = history(tmp_path, mp=mp, use_theta_m=1, thm="dry")
    payload, made, chosen = admitted(path)
    assert payload["admitted"], [r for r in payload["checks"] if r["status"] != "pass"]
    assert made.component == chain.CHAIN_SCHEMES[mp]
    assert len(chosen) == 3 and chosen[0].times == START
    facts = payload["facts"]
    assert facts["theta"]["USE_THETA_M"] == 1 and facts["theta"]["dry_T_residual_K"] < 1e-2
    assert made.coriolis == 1 and made.fcor_s_inv == pytest.approx(
        2 * chain.WRF_EOMEG_S_INV * math.sin(math.radians(float(chosen[0].array("XLAT")[7, 7]))), rel=1e-6)
    # Prepared from the file's records (no extraction), with each frame's source naming file and record.
    receipt = chain.prepare_chain(chosen, door.request_of(made), tmp_path / "prepared")
    assert receipt["admission"]["producer"] == "WRF"
    assert [f["source"]["record"] for f in receipt["admission"]["frames"]] == [0, 1, 2]
    assert all(f["source"]["format"] == "netCDF-4" for f in receipt["admission"]["frames"])
    assert (tmp_path / "prepared" / "input_sounding").is_file()


def test_theta_m_that_is_not_dry_is_still_refused(tmp_path):
    path = history(tmp_path, use_theta_m=1, thm="moist")
    payload, _, _ = admitted(path)
    assert row(payload, "theta_m")["status"] == "refused" and row(payload, "theta_m")["code"] == "GM1-CHAIN-THETA-M"


@pytest.mark.parametrize("fmt", ["CDF-1", "CDF-2", "CDF-5"])
def test_every_classic_container_is_admitted(tmp_path, fmt):
    payload, _, _ = admitted(history(tmp_path, fmt=fmt, producer="ArWen"))
    assert payload["admitted"]
    assert payload["facts"]["microphysics"]["scheme"] == "WSM6 (6)"


def test_an_idealized_history_is_gm1s_own_geometry(tmp_path):
    path = history(tmp_path, map_proj=0, drop=("XLAT", "XLONG", "MAPFAC_M"))
    payload, made, _ = admitted(path)
    assert payload["admitted"], [r for r in payload["checks"] if r["status"] != "pass"]
    assert made.coriolis == 0 and made.fcor_s_inv == 0.0
    facts = payload["facts"]
    assert facts["projection"]["name"] == "idealized (no map)"
    assert facts["map_factor_departure"] == 0.0 and facts["map_factor_source"].startswith("1 (MAP_PROJ = 0")
    assert facts["coriolis"]["source"].startswith("none")
    # With the history's own F, the rotation is F's.
    path = history(tmp_path / "f", map_proj=0, drop=("XLAT", "XLONG", "MAPFAC_M"), f_value=1.0e-4)
    payload, made, _ = admitted(path)
    assert payload["admitted"] and made.coriolis == 1 and made.fcor_s_inv == pytest.approx(1.0e-4, rel=1e-6)
    assert payload["facts"]["coriolis"]["source"] == "F"


def test_a_mapped_history_without_xlat_or_map_factor_is_refused(tmp_path):
    payload, _, _ = admitted(history(tmp_path, drop=("MAPFAC_M",)))
    assert row(payload, "map_factor")["code"] == "GM1-CHAIN-VARIABLE"


@pytest.mark.parametrize("mp, name", [(3, "WSM3 (3)"), (50, "P3 (50)"), (17, "NSSL (17)"), (1, "Kessler (1)")])
def test_other_schemes_are_refused_with_the_scheme_named(tmp_path, mp, name):
    with pytest.raises(CM1Refusal, match=r"GM1-CHAIN-MICROPHYSICS") as caught:
        chain.component_for_mp_physics(mp)
    assert name in caught.value.detail.reason
    payload, _, _ = admitted(history(tmp_path, mp=mp, species_for="wsm6-mp6"))
    micro = row(payload, "microphysics")
    assert micro["status"] == "refused" and name in micro["text"] and name in micro["detail"]
    # A scheme GM1 does not have leaves the species question unasked, not falsely refused.
    assert row(payload, "species")["status"] == "not-checked"


def test_wrf_and_iso_time_spellings_name_the_same_time():
    t = chain.parse_wrf_time("2026-05-01_18:00:00")
    assert chain.parse_wrf_time("2026-05-01 18:00:00") == t == chain.parse_wrf_time("2026-05-01T18:00:00")
    assert chain.wrf_time(t) == "2026-05-01_18:00:00"
    assert chain.wrf_time(chain.parse_wrf_time("0001-01-01_01:05:00")) == "0001-01-01_01:05:00"


def test_frames_are_selected_by_time_across_records_and_files(tmp_path):
    a = history(tmp_path, records=3)
    b = tmp_path / "later"
    b.mkdir()
    # A second file continuing the first: its records are 900, 1200 and 1500 s.
    ws.write_wrfout(b / "wrfout_d02_2026-05-01_18:15:00", fmt="CDF-2", producer="WRF", nx=16, ny=16, nz=30,
                    records=3, interval_s=300.0, start=ws.stamp(START, 900.0), mp_physics=6, grid_id=2)
    frames = door.open_frames([a, b / "wrfout_d02_2026-05-01_18:15:00"])
    assert [f.times for f in frames] == [ws.stamp(START, s) for s in (0, 300, 600, 900, 1200, 1500)]
    picked = door.select_frames(frames, start=ws.stamp(START, 300), end=ws.stamp(START, 1500), every_s=600.0)
    assert [f.times for f in picked] == [ws.stamp(START, s) for s in (300, 900, 1500)]
    with pytest.raises(door.ChainDoorError, match="not a saved time"):
        door.select_frames(frames, start=ws.stamp(START, 100), end=None, every_s=None)
    # Two domains' files for one time are refused by name.
    ws.write_wrfout(tmp_path / "wrfout_d01_2026-05-01_18:00:00", fmt="CDF-2", nx=16, ny=16, nz=30, records=1,
                    start=START, grid_id=1)
    with pytest.raises(door.ChainDoorError, match="d01 or d02"):
        door.open_frames([a, tmp_path / "wrfout_d01_2026-05-01_18:00:00"])


def test_the_surface_line_falls_back_to_wrf_phy_prep_and_the_lowest_level(tmp_path):
    path = history(tmp_path, drop=("PSFC", "T2", "TH2", "Q2"))
    frames = door.open_frames([path])
    sources = chain.surface_sources(frames[0])
    assert sources["pressure"].startswith("P+PB extrapolated")
    assert sources["theta"].startswith("lowest-level") and sources["qv"].startswith("lowest-level")
    made, chosen = door.make_choice(frames, window=BOX, ratio=3, dt_s=None)
    receipt = chain.prepare_chain(chosen, door.request_of(made), tmp_path / "prepared")
    assert receipt["admission"]["surface_record"] == sources
    header = (tmp_path / "prepared" / "input_sounding").read_text().splitlines()[0].split()
    # WRF phy_prep's surface pressure, by hand, over the box's history cells.
    f = frames[0]
    sl = (slice(5, 10), slice(5, 10))
    z = (f.array("PH")[:3].astype(float) + f.array("PHB")[:3].astype(float)) / chain.WRF_GRAVITY_M_S2
    zm = 0.5 * (z[:-1] + z[1:])
    p = f.array("P")[:2].astype(float) + f.array("PB")[:2].astype(float)
    w1 = (z[0] - zm[1]) / (zm[0] - zm[1])
    psfc = (w1 * p[0] + (1 - w1) * p[1])[sl].mean()
    assert float(header[0]) == pytest.approx(psfc / 100.0, rel=1e-7)
    assert float(header[1]) == pytest.approx(float((f.array("T")[0][sl].astype(float) + 300.0).mean()), rel=1e-7)


def test_the_report_goes_past_a_refusal_and_names_what_it_could_not_check(tmp_path):
    rough = history(tmp_path, terrain_m=900.0)
    payload, _, _ = admitted(rough)
    assert row(payload, "terrain")["status"] == "refused"
    # Everything that does not depend on flat ground is still answered.
    assert row(payload, "microphysics")["status"] == "pass" and row(payload, "crossing")["status"] == "pass"
    frames = door.open_frames([history(tmp_path / "edge")])
    made, chosen = door.make_choice(frames, window="1,5,5,5", ratio=3, dt_s=None)
    payload = door.check(chosen, made, price_it=False)
    assert row(payload, "placement")["status"] == "refused"
    for name in ("spacing", "terrain", "map_factor", "coriolis", "theta_m", "model_top", "crossing"):
        assert row(payload, name)["status"] == "not-checked", name
        assert "the box" in row(payload, name)["text"]
    assert row(payload, "microphysics")["status"] == "pass"


def test_gm1s_top_must_be_under_the_historys_last_level(tmp_path):
    frames = door.open_frames([history(tmp_path)])
    made, chosen = door.make_choice(frames, window=BOX, ratio=3, dt_s=None)
    last_mass, top_face = door._history_tops(chosen[0], (5, 5, 5, 5))
    assert made.nz * made.dz_m <= top_face and made.nz * made.dz_m - made.dz_m / 2 <= last_mass
    # One level more puts GM1's top scalar level above the history's last mass level (inside CM1's ghost,
    # which the transfer would fill by a constant Exner function): refused at admission.
    over = door.ChainChoice(**{**made.__dict__, "nz": made.nz + 1})
    payload = door.check(chosen, over, price_it=False)
    assert row(payload, "model_top")["status"] == "refused" and row(payload, "model_top")["code"] == "GM1-CHAIN-MODEL-TOP"


def test_a_record_read_from_the_file_prepares_what_its_extracted_frame_prepares(tmp_path):
    from cm1core.wrf_history import open_history

    path = history(tmp_path, fmt="CDF-2", producer="ArWen")
    frames = door.open_frames([path])
    made, chosen = door.make_choice(frames, window=BOX, ratio=3, dt_s=None)
    direct = chain.prepare_chain(chosen, door.request_of(made), tmp_path / "direct")
    h = open_history(path)
    extracted = [h.write_frame_dir(r, tmp_path / f"frame-{r}") for r in range(h.records)]
    viaframes = chain.prepare_chain(extracted, door.request_of(made), tmp_path / "extracted")
    for key in ("initial_state", "boundary_frames"):
        assert direct[key]["arena_sha256"] == viaframes[key]["arena_sha256"], key
    assert direct["sounding"]["sha256"] == viaframes["sounding"]["sha256"]


def test_the_prepare_tool_reads_wrf_histories(tmp_path):
    """tools/arwen_chain_prepare.py on a document's [chain] table.

    A multi-record netCDF-4 WRF history is read by the default reader (cm1core.wrf_history) and prepared; on
    one-record classic files, the default reader and the Rust extractor (``--reader rust``, the oracle) prepare the
    same arenas word for word.
    """

    import json
    import os
    import subprocess
    import sys

    import test_arwen_chain as synth
    import test_arwen_chain_document as docs

    root = Path(__file__).resolve().parents[1]
    tool = root / "tools" / "arwen_chain_prepare.py"
    extractor = root / "rust/cm1-state/target/release/gm1-arwen-history-extract"

    def prepare(folder: Path, reader: str, name: str) -> Path:
        deck = tmp_path / f"{name}.toml"
        prepared = tmp_path / name
        deck.write_text(docs.DOCUMENT.format(fcor=synth.FCOR) + docs.CHAIN.format(
            history=folder, start=START, end=ws.stamp(START, 600), prepared=prepared))
        done = subprocess.run([sys.executable, str(tool), str(deck), "--reader", reader], capture_output=True,
                              text=True, timeout=300, env={**os.environ, "PYTHONPATH": str(root / "src")})
        assert done.returncode == 0, done.stderr[-3000:]
        out = json.loads(done.stdout)
        assert out["frames"] == 3 and out["interval_s"] == 300.0
        receipt = json.loads((prepared / chain.RECEIPT).read_text())
        assert [f["times"] for f in receipt["admission"]["frames"]] == [ws.stamp(START, s) for s in (0, 300, 600)]
        return prepared

    multi = tmp_path / "multi"
    history(multi, fmt="netCDF-4", records=3, name=f"wrfout_d02_{START}")
    prepare(multi, "python", "prepared-netcdf4")
    if not extractor.exists():
        pytest.skip("build rust/cm1-state gm1-arwen-history-extract first")
    single = tmp_path / "single"
    for seconds in (0.0, 300.0, 600.0):
        history(single, fmt="CDF-2", records=1, name=f"wrfout_d02_{ws.stamp(START, seconds)}",
                start=ws.stamp(START, seconds))
    a = prepare(single, "python", "prepared-python")
    b = prepare(single, "rust", "prepared-rust")
    for arena in ("chain-initial-state.f32le", "chain-boundary-frames.f32le", "input_sounding"):
        assert (a / arena).read_bytes() == (b / arena).read_bytes(), arena


def test_a_file_whose_times_a_post_processor_rewrote_in_the_iso_spelling_reads_the_same(tmp_path):
    """Verifier: ISO-spelled Times inside the file, not only typed ones, name WRF's times."""

    plain = history(tmp_path / "wrf")
    iso = history(tmp_path / "iso")
    with netCDF4.Dataset(iso, "a") as ds:
        for r, sep in enumerate((" ", "T", " ")):
            text = ds["Times"][r].tobytes().decode()
            ds["Times"][r] = np.array(list(text[:10] + sep + text[11:]), dtype="S1")
    a, b = door.open_frames([plain]), door.open_frames([iso])
    assert [f.times for f in b] == [f.times for f in a] == [ws.stamp(START, s) for s in (0, 300, 600)]
    assert [f.valid_s for f in b] == [f.valid_s for f in a]
    payload, _, _ = admitted(iso, start="2026-05-01T18:05:00")
    assert payload["admitted"], [r for r in payload["checks"] if r["status"] != "pass"]
    from cm1core.wrf_history import open_history

    with open_history(iso) as h:
        assert h.record_of("2026-05-01 18:10") == 2


def test_plain_words_name_what_is_missing_and_count_the_saved_times(tmp_path):
    """Verifier: a refusal line says which field is missing, and one saved time is counted as one."""

    payload, _, _ = admitted(history(tmp_path / "one", records=1))
    assert row(payload, "frame_count")["text"].startswith("Only 1 saved time")
    assert row(payload, "frame_grid")["text"] == "Not checked until the two saved times check passes."
    payload, _, _ = admitted(history(tmp_path / "sp", mp=10, drop=("QNRAIN", "QNSNOW")))
    species = row(payload, "species")
    assert species["status"] == "refused" and "QNSNOW" in species["text"] and "QNRAIN" in species["text"]
    payload, _, _ = admitted(history(tmp_path / "mf", drop=("MAPFAC_M",)))
    assert "MAPFAC_M" in row(payload, "map_factor")["text"] and "nan" not in row(payload, "map_factor")["text"]
    payload, _, _ = admitted(history(tmp_path / "thm", use_theta_m=1, thm="none"))
    assert row(payload, "theta_m")["status"] == "refused" and "THM" in row(payload, "theta_m")["text"]
