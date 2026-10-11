"""The one wrfout reader (cm1core.wrf_history) on synthetic WRF and ArWen histories (host only).

What is covered:

* every container netCDF4 writes -- CDF-1, CDF-2, CDF-5, netCDF-4 -- read raw-word, ArWen-like and WRF-like;
* the producer (gpuwm's seal or title, else WRF), the domain (GRID_ID, or d0N in the name), multi-record Times;
* frames with ArwenFrame's interface; USE_THETA_M 0 and 1 with THM; MAP_PROJ 0, 1, 2 and 3;
* write_frame_dir equals the Rust extractor word for word on CDF-1 and CDF-2 (when the extractor is built);
* the reader's variable list is the extractor's; refusals name the record, time and type.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("netCDF4")

from cm1core import wrf_history as wh  # noqa: E402
from cm1core.arwen_chain import ArwenFrame  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402

from wrf_support import write_wrfout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXTRACTOR = REPO / "rust/cm1-state/target/release/gm1-arwen-history-extract"


def bits(a):
    return np.ascontiguousarray(a, dtype="<f4").view("<u4")


@pytest.mark.parametrize("fmt", ["CDF-1", "CDF-2", "CDF-5", "netCDF-4"])
@pytest.mark.parametrize("producer", ["ArWen", "WRF"])
def test_every_container_is_read_raw_word(tmp_path, fmt, producer):
    path = tmp_path / "wrfout_d01_2026-05-01_18:00:00"
    facts = write_wrfout(path, fmt=fmt, producer=producer, alpha_deg=12.0)
    with wh.open_history(path) as history:
        assert history.format == fmt
        assert history.producer == producer
        assert history.records == 3
        assert list(history.times) == facts["times"]
        assert history.domain_id == 1
        import netCDF4

        ds = netCDF4.Dataset(str(path))
        ds.set_auto_maskandscale(False)
        for record in range(3):
            frame = history.frame(record)
            assert frame.times == facts["times"][record]
            assert frame.valid_s == wh.parse_wrf_time(facts["times"][record])
            for name in ("U", "V", "T", "PH", "PHB", "P", "PB", "QVAPOR", "HGT", "XLAT", "PSFC"):
                raw = np.asarray(ds[name][record], dtype=np.float32)
                assert np.array_equal(bits(frame.array(name)), bits(raw)), (fmt, name, record)
        ds.close()
        assert history.attributes["DX"] == 3000
        assert history.attributes["MAP_PROJ"] == 1


def test_producer_domain_and_times(tmp_path):
    a = tmp_path / "wrfout_d02_2026-05-01_18:00:00"
    write_wrfout(a, producer="WRF", grid_id=None, records=3, interval_s=600.0)
    with wh.open_history(a) as history:
        assert history.producer == "WRF"
        assert history.domain_id == 2           # from the file name when GRID_ID is absent
        assert history.times[1] == "2026-05-01_18:10:00"
        assert history.record_of("2026-05-01 18:20") == 2
        assert history.record_of("2026-05-01T18:10:00Z") == 1
        with pytest.raises(CM1Refusal) as refused:
            history.record_of("2026-05-01_19:00:00")
        assert refused.value.detail.code == "GM1-WRF-TIME"
        with pytest.raises(CM1Refusal) as refused:
            history.frame(3)
        assert refused.value.detail.code == "GM1-WRF-RECORD"
    b = tmp_path / "hist.nc"
    write_wrfout(b, producer="WRF", title="gpuwm direct export", grid_id=3)
    with wh.open_history(b) as history:
        assert history.producer == "ArWen"      # gpuwm in the title, no GPUWM_VERSION
        assert history.domain_id == 3


@pytest.mark.parametrize("map_proj", [0, 1, 2, 3])
@pytest.mark.parametrize("use_theta_m", [0, 1])
def test_map_projections_and_theta_m_frames(tmp_path, map_proj, use_theta_m):
    path = tmp_path / "wrfout_d01"
    write_wrfout(path, producer="WRF", fmt="netCDF-4", map_proj=map_proj, use_theta_m=use_theta_m)
    with wh.open_history(path) as history:
        frame = history.frame(1)
        assert frame.attributes["MAP_PROJ"] == map_proj
        assert frame.attributes["USE_THETA_M"] == use_theta_m
        assert frame.has("THM")
        assert frame.has("SINALPHA") == (map_proj in (1, 2))
        entry = frame.entry("T")
        assert entry["dims"] == ["bottom_top", "south_north", "west_east"]
        assert entry["shape"] == [24, 7, 8]
        t = frame.array("T").astype(np.float64) + 300.0
        qv = frame.array("QVAPOR").astype(np.float64)
        thm = frame.array("THM").astype(np.float64) + 300.0
        assert np.max(np.abs(t * (1 + 461.6 / 287.0 * qv) - thm)) < 1e-2
        facts = history.facts()
        assert facts["nx"] == 8 and facts["ny"] == 7 and facts["nz"] == 24
        assert facts["map_proj_name"] == wh.MAP_PROJECTIONS[map_proj]


def test_frame_interface_matches_arwen_frame(tmp_path):
    path = tmp_path / "wrfout_d01"
    write_wrfout(path)
    with wh.open_history(path) as history:
        frame = history.frame(0, wh.CHAIN_VARIABLES)
        import dataclasses

        arwen_names = {f.name for f in dataclasses.fields(ArwenFrame)} | set(dir(ArwenFrame))
        for name in ("times", "valid_s", "attributes", "dimensions", "has", "entry", "array", "directory", "header"):
            assert hasattr(frame, name) and name in arwen_names
        assert frame.dimensions["Time"] == 3
        header = frame.header
        assert header["schema"] == wh.FRAME_SCHEMA
        assert header["source"]["numrecs"] == 3 and header["source"]["format"] == "CDF-2"
        assert "SINALPHA" not in [row["name"] for row in header["fields"]]   # not an extractor variable
        assert "QRAIN" in header["absent"]
        with pytest.raises(CM1Refusal) as refused:
            frame.entry("QRAIN")
        assert refused.value.detail.code == "GM1-CHAIN-VARIABLE"      # the chain's own refusal, as ArwenFrame
        # the same frame written to disk opens as an ArwenFrame with identical words
        written = history.write_frame_dir(0, tmp_path / "frame")
        assert isinstance(written, ArwenFrame)
        for name in ("U", "T", "PH", "XLAT"):
            assert np.array_equal(bits(written.array(name)), bits(frame.array(name)))
            assert written.entry(name)["sha256"] == frame.entry(name)["sha256"]
        with pytest.raises(CM1Refusal) as refused:
            history.write_frame_dir(0, tmp_path / "frame")
        assert refused.value.detail.code == "GM1-WRF-OUTPUT"


def test_a_double_field_is_refused_not_narrowed(tmp_path):
    import netCDF4

    path = tmp_path / "wrfout_d01"
    write_wrfout(path, fmt="netCDF-4", drop=("HGT",))
    ds = netCDF4.Dataset(str(path), "a")
    v = ds.createVariable("HGT", "f8", ("Time", "south_north", "west_east"))
    v[0] = np.zeros((7, 8))
    ds.close()
    with wh.open_history(path) as history:
        with pytest.raises(CM1Refusal) as refused:
            history.array("HGT", 0)
        assert refused.value.detail.code == "GM1-WRF-TYPE"


def test_not_a_history_is_refused_in_plain_words(tmp_path):
    text = tmp_path / "namelist.input"
    text.write_text("&param1\n/\n")
    with pytest.raises(CM1Refusal) as refused:
        wh.open_history(text)
    assert refused.value.detail.code == "GM1-WRF-FILE"
    with pytest.raises(CM1Refusal):
        wh.open_history(tmp_path / "missing")
    import netCDF4

    nc = tmp_path / "other.nc"
    ds = netCDF4.Dataset(str(nc), "w")
    ds.createDimension("x", 2)
    ds.createVariable("a", "f4", ("x",))
    ds.close()
    with pytest.raises(CM1Refusal) as refused:
        wh.open_history(nc)
    assert "Times" in refused.value.detail.reason


def test_chain_variables_are_the_extractors():
    source = (REPO / "rust/cm1-state/src/arwen_history.rs").read_text(encoding="utf-8")
    block = source[source.index("pub const CHAIN_VARIABLES"):]
    block = block[:block.index("];")]
    assert tuple(re.findall(r'"([A-Z0-9_]+)"', block)) == wh.CHAIN_VARIABLES


def test_open_frames_orders_records_across_files(tmp_path):
    a = tmp_path / "wrfout_d01_a"
    b = tmp_path / "wrfout_d01_b"
    write_wrfout(a, records=2, start="2026-05-01_19:00:00", interval_s=1800.0)
    write_wrfout(b, records=2, start="2026-05-01_18:00:00", interval_s=1800.0)
    frames = wh.open_frames([a, b])
    assert [f.times for f in frames] == ["2026-05-01_18:00:00", "2026-05-01_18:30:00", "2026-05-01_19:00:00",
                                         "2026-05-01_19:30:00"]


@pytest.mark.skipif(not EXTRACTOR.exists(), reason="build rust/cm1-state gm1-arwen-history-extract first")
@pytest.mark.parametrize("fmt", ["CDF-1", "CDF-2"])
@pytest.mark.parametrize("producer", ["ArWen", "WRF"])
def test_write_frame_dir_equals_the_rust_extractor_word_for_word(tmp_path, fmt, producer):
    path = tmp_path / "wrfout_d01_2026-05-01_18:00:00"
    write_wrfout(path, fmt=fmt, producer=producer, use_theta_m=1, alpha_deg=-8.0, terrain_m=300.0)
    for record in (0, 2):
        rust = tmp_path / f"rust-{record}"
        subprocess.run([os.fspath(EXTRACTOR), os.fspath(path), str(record), os.fspath(rust)], check=True,
                       capture_output=True, text=True)
        with wh.open_history(path) as history:
            mine = tmp_path / f"python-{record}"
            history.write_frame_dir(record, mine)
        assert (rust / "fields.f32le").read_bytes() == (mine / "fields.f32le").read_bytes()
        a = json.loads((rust / "frame.json").read_text())
        b = json.loads((mine / "frame.json").read_text())
        assert a == b


def real_history_paths() -> list[Path]:
    """Real histories, read-only, where they exist on this machine (never copied into the repository)."""

    candidates = [os.environ.get("GM1_WRF_REAL_FILES", "")]
    return [Path(p) for p in ":".join(candidates).split(":") if p and Path(p).is_file()]


@pytest.mark.skipif(not EXTRACTOR.exists() or not real_history_paths(),
                    reason="set GM1_WRF_REAL_FILES to real classic wrfout paths and build the extractor")
def test_real_classic_files_equal_the_rust_extractor(tmp_path):
    for n, path in enumerate(real_history_paths()):
        with wh.open_history(path) as history:
            if history.format not in ("CDF-1", "CDF-2"):
                continue
            record = history.records - 1
            mine = tmp_path / f"python-{n}"
            history.write_frame_dir(record, mine)
        rust = tmp_path / f"rust-{n}"
        subprocess.run([os.fspath(EXTRACTOR), os.fspath(path), str(record), os.fspath(rust)], check=True,
                       capture_output=True, text=True)
        assert (rust / "fields.f32le").read_bytes() == (mine / "fields.f32le").read_bytes()
        assert json.loads((rust / "frame.json").read_text()) == json.loads((mine / "frame.json").read_text())
