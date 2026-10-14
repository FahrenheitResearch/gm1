"""``gpuwm-cm1 quicklook`` on CM1 netCDF history files (:mod:`cm1core.quicklook.ncread`).

Three kinds of file: CM1 r21.1's own (``tests/data/cm1-ref-24x24/cm1out_000001.nc``, unmodified), GM1's
``cm1out`` files written from frames of CM1's own storm state (``quicklook_support``), and small hand-made files
for what the other two do not have (stretched levels, a missing field, files that cannot be drawn).  What is
checked: a look drawn from a GM1 run's netCDF matches the same look drawn from its frames (the same numbers, the same
pixels outside the words); CM1's own output says it is CM1's; the file's own heights are used; a missing field is
named with the CM1 switch that writes it; and what cannot be drawn says why, in plain words.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from cm1core.quicklook import ncread as N
from cm1core.quicklook import render as RD
from cm1core.quicklook import slices as SL
from cm1core.quicklook import style as S
from cm1core.quicklook import facts as FA
from cm1core.quicklook import products as P
from cm1core.quicklook.frame import QuickFrame, lineage

from cm1import_support import namelist_tool  # noqa: E402
from quicklook_support import NAMELIST_24, run_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CM1_FILE = ROOT / "tests" / "data" / "cm1-ref-24x24" / "cm1out_000001.nc"
CM1_NAMELIST = ROOT / "tests" / "data" / "cm1-ref-24x24" / "namelist.input"
NEEDS = pytest.mark.skipif(importlib.util.find_spec("netCDF4") is None or importlib.util.find_spec("matplotlib") is None,
                           reason="netCDF4 and matplotlib come with gpuwm")
pytestmark = NEEDS
#: The looks drawn from a GM1 run's frames and from its cm1out files that must match picture for picture.
MATCHING = ("reflectivity", "spin-near-ground", "updraft-5km", "surface", "rain", "clouds-from-above",
            "updraft-slice")


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


@pytest.fixture()
def pictures(monkeypatch):
    seen: list[tuple[str, S.Overlay]] = []
    original = P.save

    def spy(ov, base, path):
        seen.append((Path(path).parent.name, ov))
        return original(ov, base, path)

    monkeypatch.setattr(P, "save", spy)
    return seen


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _cm1out(run: Path, *extra: str) -> Path:
    done = subprocess.run([sys.executable, "-m", "cm1core", "cm1out", str(run), "--format", "nc", "--no-stats",
                           *extra], capture_output=True, text=True, env=_env())
    assert done.returncode == 0, done.stderr
    return run / "cm1out"


def _pixels(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def cm1_file(path: Path, *, nx: int = 12, ny: int = 10, zf_m=None, times=(0.0, 600.0), units: str = "km",
             xf_m=None, leave_out: tuple[str, ...] = (), umove: float = 0.0, attrs: dict | None = None) -> Path:
    """A small history file laid out as CM1's writeout_nc.F writes it, with smooth made-up fields."""

    import netCDF4

    zf = np.asarray(zf_m if zf_m is not None else np.arange(11) * 500.0, dtype=np.float64)
    nz = len(zf) - 1
    dx = 1000.0
    xf = np.asarray(xf_m if xf_m is not None else (np.arange(nx + 1) - nx / 2) * dx, dtype=np.float64)
    yf = (np.arange(ny + 1) - ny / 2) * dx
    conv = 0.001 if units == "km" else 1.0
    ds = netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")
    for name, size in (("xh", nx), ("yh", ny), ("zh", nz), ("xf", nx + 1), ("yf", ny + 1), ("zf", nz + 1),
                       ("time", None), ("one", 1)):
        ds.createDimension(name, size)
    var = ds.createVariable("time", "f4", ("time",))
    var.units = "seconds"
    var[:] = np.asarray(times, np.float32)
    for name, values in (("xh", 0.5 * (xf[1:] + xf[:-1])), ("xf", xf), ("yh", 0.5 * (yf[1:] + yf[:-1])), ("yf", yf),
                         ("zh", 0.5 * (zf[1:] + zf[:-1])), ("zf", zf)):
        var = ds.createVariable(name, "f4", (name,))
        var.units = units
        var[:] = (values * conv).astype(np.float32)
    if umove:
        for name, value in (("umove", umove), ("vmove", 0.0)):
            var = ds.createVariable(name, "f4", ("time",))
            var.units = "m/s"
            var[:] = np.full(len(times), value, np.float32)
    zh = 0.5 * (zf[1:] + zf[:-1])
    j, i = np.mgrid[0:ny, 0:nx]
    bump = np.exp(-(((i - nx / 2) / 3.0) ** 2 + ((j - ny / 2) / 3.0) ** 2))
    fields = {
        "th": ("K", "potential temperature", ("zh", "yh", "xh"),
               (300.0 + 0.004 * zh)[:, None, None] - 2.0 * bump[None] * (zh < 1500)[:, None, None]),
        "prs": ("Pa", "pressure", ("zh", "yh", "xh"), np.broadcast_to((1.0e5 * np.exp(-zh / 8000.0))[:, None, None],
                                                                      (nz, ny, nx))),
        "qv": ("kg/kg", "water vapor mixing ratio", ("zh", "yh", "xh"),
               np.broadcast_to((0.014 * np.exp(-zh / 3000.0))[:, None, None], (nz, ny, nx))),
        "u": ("m/s", "E-W (x) velocity (grid-relative)", ("zh", "yh", "xf"),
              np.broadcast_to((5.0 + 0.002 * zh)[:, None, None], (nz, ny, nx + 1))),
        "v": ("m/s", "N-S (y) velocity (grid-relative)", ("zh", "yf", "xh"),
              np.broadcast_to(np.linspace(-3, 3, nx)[None, None, :], (nz, ny + 1, nx))),
        "w": ("m/s", "vertical velocity", ("zf", "yh", "xh"), 12.0 * bump[None] * np.sin(np.pi * zf / zf[-1])[:, None, None]),
        "rain": ("cm", "accumulated surface rainfall", ("yh", "xh"), 0.5 * bump),
    }
    for name, (unit, long_name, dims, values) in fields.items():
        if name in leave_out:
            continue
        var = ds.createVariable(name, "f4", ("time", *dims))
        var.units, var.long_name = unit, long_name
        for n, _t in enumerate(times):
            var[n] = np.array(values, np.float32) * np.float32(1.0 + 0.1 * n if name in ("w", "rain") else 1.0)
    ds.setncattr("CM1 version", "cm1r21.1")
    ds.setncattr("ptype", np.int32(5))
    for key, value in (attrs or {}).items():
        ds.setncattr(key, value)
    ds.close()
    return path


# ---------------------------------------------------------------- a GM1 run: its frames against its netCDF

def test_a_gm1_runs_looks_from_its_cm1out_files_match_its_frames(tmp_path):
    run = run_directory(tmp_path)                       # three frames of CM1's own storm, on a moving domain
    files = _cm1out(run, "--filetype", "2")
    frames = RD.render(run, out=tmp_path / "from-frames")
    netcdf = RD.render(files, out=tmp_path / "from-netcdf")
    assert frames.problems == [] and netcdf.problems == [] and netcdf.frames_drawn == 3
    a = json.loads((tmp_path / "from-frames" / "index.json").read_text())
    b = json.loads((tmp_path / "from-netcdf" / "index.json").read_text())
    assert [f["time_s"] for f in b["frames"]] == [900.0, 1200.0, 1500.0]
    looks_a = {p["id"]: p for p in a["products"]}
    looks_b = {p["id"]: p for p in b["products"]}
    for look in MATCHING:
        fa, fb = looks_a[look]["frames"][-1], looks_b[look]["frames"][-1]
        same = {k: v for k, v in fa["facts"].items() if k != "operator"}
        assert {k: fb["facts"][k] for k in same} == pytest.approx(same, rel=1e-5, abs=1e-6), look
        # the reflectivity's words are the frames' own (the name already says "from netCDF")
        assert fb["facts"].get("operator") == fa["facts"].get("operator"), look
        pa, pb = _pixels(tmp_path / "from-frames" / fa["file"]), _pixels(tmp_path / "from-netcdf" / fb["file"])
        maps = np.abs(pa - pb)[200:950].max(axis=2)
        if look == "surface":
            maps = maps[:, 520:]          # beside the map the netCDF picture says how its θ′ was formed
        # the same picture: at most a few pixels where a value on a colour step differs in its last bits
        assert int((maps > 2).sum()) <= 20 and int((maps > 0).sum()) <= 200, look
    # the swath of the files' translated swaths, placed under the domain at each saved time: the same strongest wind
    sa, sb = looks_a["swath"]["frames"][-1]["facts"], looks_b["swath"]["frames"][-1]["facts"]
    assert sb["source"] == "model" and sb["wind_max_ground_relative"] == pytest.approx(sa["wind_max_ground_relative"])
    assert sb["uh_max"] == pytest.approx(sa["uh_max"])
    # the series: the same numbers, labelled with where they were read
    assert b["run"]["name"] == "storm, from netCDF" and b["run"]["gm1_version"] == a["run"]["gm1_version"]
    assert b["run"]["microphysics"] == a["run"]["microphysics"] == "Morrison with hail"


def test_the_netcdf_pictures_carry_the_runs_own_footer_and_their_source(tmp_path, pictures):
    run = run_directory(tmp_path, times=(1500.0,))
    files = _cm1out(run, "--filetype", "1")
    assert (files / "cm1out.nc").is_file()
    RD.render(files)
    assert (files / "quicklook" / "index.json").is_file()       # pictures go beside the files
    for name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == [], name
        assert ov.mark_word == S.SIMULATED                        # GM1 simulated this storm
        assert any(t.startswith("storm, from netCDF") for t in ov.texts), name
        assert any(t.startswith("simulated  ·  GM1 ") for t in ov.texts), name


# ---------------------------------------------------------------- CM1's own output

def test_cm1s_own_file_is_drawn_and_says_it_is_cm1s(tmp_path, pictures):
    folder = tmp_path / "cm1run"
    folder.mkdir()
    shutil.copy(CM1_FILE, folder / "cm1out_000001.nc")
    shutil.copy(CM1_NAMELIST, folder / "namelist.input")
    layout = N.discover(folder)
    assert layout.kind == "netcdf" and not layout.gm1 and [t.time_s for t in layout.times] == [0.0]
    assert (layout.grid.nx, layout.grid.ny, layout.grid.nz, layout.grid.dx) == (24, 24, 40, 1000.0)
    assert layout.grid.iorigin == 2 and layout.grid.uniform and layout.grid.zh[0] == 250.0
    summary = RD.render(folder)
    assert summary.frames_drawn == 1
    index = json.loads((folder / "quicklook" / "index.json").read_text())
    assert index["run"]["made_by"] == "CM1 r21.1" and index["run"]["model"] == "CM1"
    assert index["run"]["name"] == "cm1run, CM1 output" and index["run"]["microphysics"] == "Morrison with hail"
    drawn = {p["id"] for p in index["products"] if p["frames"]}
    assert {"reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain"} <= drawn
    for name, ov in pictures:
        assert ov.check() == [] and ov.footer_cut == [], name
        footer = "\n".join(ov.texts)
        assert "CM1 r21.1" in footer and "GM1 0." not in footer and "of CM1 on" not in footer, name
        assert ov.mark_word == FA.CM1_MARK_WORD, name           # GM1 made the picture; CM1 simulated the storm
    reflectivity = next(p for p in index["products"] if p["id"] == "reflectivity")["frames"][0]["facts"]
    assert reflectivity["operator"] == "CM1's own reflectivity (dbz)"       # the scheme is in the footer
    series = json.loads((folder / "quicklook" / "series.json").read_text())
    assert series["gm1_label"] == "cm1run, CM1 output" and all("cm1" not in chart for chart in series["charts"])


def test_slices_and_columns_of_cm1s_file_show_its_own_fields(tmp_path):
    folder = tmp_path / "cm1run"
    folder.mkdir()
    shutil.copy(CM1_FILE, folder / "cm1out_000001.nc")
    env = _env()
    base = [sys.executable, "-m", "cm1core", "quicklook", str(folder / "cm1out_000001.nc")]
    listed = subprocess.run(base + ["--fields", "--json"], capture_output=True, text=True, env=env)
    assert listed.returncode == 0, listed.stderr
    record = json.loads(listed.stdout)
    names = {row["name"] for row in record["fields"]}
    assert {"u", "v", "w", "th", "prs", "qv", "dbz"} <= names
    w = next(row for row in record["fields"] if row["name"] == "w")
    assert w["grid"] == "w" and w["levels_m"][:3] == [0.0, 500.0, 1000.0]
    # no namelist beside the file: said once, and the pictures are drawn as a thunderstorm run's
    assert any("No CM1 namelist" in note for note in record["notes"])
    section = subprocess.run(base + ["--section", "th", "--from", "-10", "0", "--to", "10", "0", "--json"],
                             capture_output=True, text=True, env=env)
    assert section.returncode == 0, section.stderr
    assert Path(json.loads(section.stdout)["path"]).parent == folder / "cm1out_000001-quicklook" / "slices"
    column = subprocess.run(base + ["--column", "--at", "0", "0", "--json"], capture_output=True, text=True, env=env)
    assert column.returncode == 0, column.stderr
    slice_ = subprocess.run(base + ["--slice", "dbz", "--height", "1000", "--json"], capture_output=True, text=True,
                            env=env)
    assert slice_.returncode == 0 and json.loads(slice_.stdout)["height_m"] == 750.0


# ---------------------------------------------------------------- heights, times and the moving domain

def test_stretched_levels_are_the_files_own_heights(tmp_path):
    zf = np.concatenate([np.arange(0.0, 1000.0, 100.0), np.arange(1000.0, 10001.0, 500.0)])
    path = cm1_file(tmp_path / "cm1out_000001.nc", zf_m=zf, times=(0.0,))
    layout = N.discover(path)
    geometry = layout.geometry
    assert not geometry.uniform and geometry.nz == len(zf) - 1
    assert np.allclose(geometry.zf, zf) and np.allclose(geometry.zh, 0.5 * (zf[1:] + zf[:-1]))
    rows = layout.lineage()
    qf = QuickFrame.open(rows[0][2], key=rows[0][1])
    try:
        facts = FA.run_facts(layout, qf.config, qf.dx, qf.dy)
        record = SL.horizontal(qf, facts, None, "th", tmp_path / "out", height_m=2100.0, tag="t0")
        assert record["height_m"] == 2250.0 and record["level"] == 13     # the file's own level nearest 2.1 km
        assert qf.stretched and qf.level_near(5000.0, faces=True) == 18
    finally:
        qf.close()


def test_one_file_with_every_time_and_the_moving_domain(tmp_path):
    path = cm1_file(tmp_path / "cm1out.nc", times=(0.0, 600.0, 1200.0), umove=10.0)
    layout = N.discover(tmp_path)
    assert [r[1] for r in layout.lineage()] == ["cm1out.nc[1]", "cm1out.nc[2]", "cm1out.nc[3]"]
    locx, _locy, umove, _vmove = layout.domain_position(layout.times[2])
    assert umove == 10.0 and locx == pytest.approx(12000.0)
    summary = RD.render(path)
    assert summary.frames_drawn == 3
    assert (tmp_path / "cm1out-quicklook" / "index.json").is_file()


# ---------------------------------------------------------------- what is not there, and what cannot be drawn

def test_a_missing_field_is_named_with_the_cm1_switch_that_writes_it(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    cm1_file(folder / "cm1out_000001.nc", leave_out=("w",), times=(0.0,))
    summary = RD.render(folder)
    index = json.loads((folder / "quicklook" / "index.json").read_text())
    updraft = next(p for p in index["products"] if p["id"] == "updraft-5km")
    assert updraft["not_drawn"][0]["reason"] == ("cm1out_000001.nc has no vertical velocity (w) "
                                                 "(CM1 writes it with output_w = 1).")
    # a look the files cannot give is not a fault: nothing in the problems reads like a program error
    assert all("Error" not in problem and "Traceback" not in problem for problem in summary.problems)
    env = _env()
    slice_ = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(folder), "--column"],
                            capture_output=True, text=True, env=env)
    assert slice_.returncode == 2 and "has no vertical velocity (w)" in slice_.stderr
    assert "Traceback" not in slice_.stderr
    # CM1's parcel file is not read: the parcels look says so, and does not ask for iprcl
    parcels = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(folder), "--look", "parcels"],
                             capture_output=True, text=True, env=env)
    assert parcels.returncode == 2 and "cm1out_pdata.nc" in parcels.stderr and "iprcl" not in parcels.stderr


def test_files_that_cannot_be_drawn_say_why(tmp_path):
    env = _env()

    def refused(target: Path, *extra: str) -> str:
        done = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(target), *extra],
                              capture_output=True, text=True, env=env)
        assert done.returncode == 2 and "Traceback" not in done.stderr, done.stderr
        return done.stderr

    # CM1's stretched horizontal grid
    xf = np.concatenate([np.arange(-6000.0, 0.0, 1000.0), np.arange(0.0, 13000.0, 2000.0)])
    (tmp_path / "x").mkdir()
    stretched = cm1_file(tmp_path / "x" / "cm1out_000001.nc", xf_m=xf, nx=len(xf) - 1)
    assert "the x spacing changes across the domain" in refused(stretched)
    # one file per process (output_filetype = 3): two pieces of the same time
    pieces = tmp_path / "pieces"
    pieces.mkdir()
    cm1_file(pieces / "cm1out_000000_000001.nc", times=(0.0,))
    cm1_file(pieces / "cm1out_000001_000001.nc", times=(0.0,))
    assert "output_filetype = 3" in refused(pieces)
    # only CM1's statistics file
    stats = tmp_path / "stats"
    stats.mkdir()
    import netCDF4

    with netCDF4.Dataset(stats / "cm1out_stats.nc", "w") as ds:
        ds.createDimension("time", None)
        ds.createVariable("time", "f4", ("time",))[:] = [0.0]
    assert "holds no CM1 netCDF history file" in refused(stats)
    # a file that is not netCDF
    text = tmp_path / "notes.nc"
    text.write_text("not netCDF")
    assert "not a netCDF file" in refused(text)
    # CM1's GrADS output
    grads = tmp_path / "grads"
    grads.mkdir()
    (grads / "cm1out_s.ctl").write_text("dset ^cm1out_s.dat")
    assert "CM1's GrADS output (output_format = 1)" in refused(grads)
    # the files are all there: --follow is for a GM1 run that is going
    ok = tmp_path / "ok"
    ok.mkdir()
    cm1_file(ok / "cm1out_000001.nc", times=(0.0,))
    assert "--follow is for a GM1 run" in refused(ok, "--follow")


def test_a_gm1_run_folder_is_still_drawn_from_its_frames(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))
    _cm1out(run, "--filetype", "2")
    layout, rows = lineage(run)
    assert layout.kind == "output" and Path(rows[0][2]).name == "frame-000000"
    assert N.discover(run / "cm1out").kind == "netcdf"
    assert N.discover(run / "cm1out").records_root == run


def _without(source: Path, target: Path, leave_out: tuple[str, ...]) -> Path:
    """A copy of a netCDF file without some of its variables."""

    import netCDF4

    with netCDF4.Dataset(source) as src, netCDF4.Dataset(target, "w", format="NETCDF4_CLASSIC") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
        for name, var in src.variables.items():
            if name in leave_out:
                continue
            out = dst.createVariable(name, var.datatype, var.dimensions)
            out.setncatts({key: var.getncattr(key) for key in var.ncattrs()})
            out[:] = var[:]
    return target


def test_without_dbz_a_morrison_files_reflectivity_is_cm1s_operator_on_the_cpu(tmp_path):
    run = run_directory(tmp_path, times=(1500.0,))         # its namelist writes rho and says ihail = 1
    files = _cm1out(run, "--filetype", "2")
    written = sorted(files.glob("cm1out_0*.nc"))[-1]
    bare = tmp_path / "bare"
    bare.mkdir()
    _without(written, bare / written.name, ("dbz", "cref"))
    shutil.copy(run / "namelist.input", bare / "namelist.input")
    frames = RD.render(run, out=tmp_path / "from-frames", only=("reflectivity",))
    netcdf = RD.render(bare, only=("reflectivity",))
    assert frames.problems == [] and netcdf.problems == []
    a = json.loads((tmp_path / "from-frames" / "index.json").read_text())["products"][0]["frames"][-1]["facts"]
    b = json.loads((bare / "quicklook" / "index.json").read_text())["products"][0]["frames"][-1]["facts"]
    assert b["operator"] == a["operator"] == "CM1's calc_refl10cm (Morrison, 10 cm) on the CPU"
    assert b["dbz_max_column"] == pytest.approx(a["dbz_max_column"], abs=1e-3)
    assert b["dbz_max_1km"] == pytest.approx(a["dbz_max_1km"], abs=1e-3)


# ---------------------------------------------------------------- the review's findings

def test_a_long_data_line_is_set_smaller_and_never_runs_into_the_panel_titles(tmp_path, pictures):
    ov = S.Overlay((1920, 1080))
    ov.title("Reflectivity at 1 km", S.data_line("storm", "1 km grid", "whole 120 × 120 km domain"))
    assert ov._head_rows == 1 and ov.layout_problems == []
    long = S.data_line("supercell-dx1000-moving-20261013-run2-with-a-much-longer-name, from netCDF", "1 km grid",
                       "whole 120 × 120 km domain", "CM1's calc_refl10cm (Morrison, 10 cm) on the CPU")
    ov = S.Overlay((1920, 1080))
    ov.title("Reflectivity at 1 km", long)
    assert ov._head_rows == 1 and long in ov.texts             # one row, in smaller type; every word is said
    ov = S.Overlay((1920, 1080))
    ov.title("Reflectivity at 1 km", S.data_line("x" * 300, "1 km grid", "whole 120 × 120 km domain"))
    assert ov._head_rows == 1                                     # the name gives way to an ellipsis
    # the check: a data line on two rows that reaches a panel title is a fault the picture is refused for
    ov = S.Overlay((1920, 1080))
    ov.title("Reflectivity at 1 km", "a" * 12 + " " + " ".join(["word"] * 400))
    assert ov._head_rows > 1
    ov.clear_of_head(150, "the panel titles")
    assert ov.layout_problems == ["the data line under the title runs into the panel titles"]
    assert "the data line under the title runs into the panel titles" in ov.check()
    # drawn: CM1 output with a long name keeps its head on one row
    folder = tmp_path / "cm1run"
    folder.mkdir()
    shutil.copy(CM1_FILE, folder / "cm1out_000001.nc")
    shutil.copy(CM1_NAMELIST, folder / "namelist.input")
    RD.render(folder, label="supercell-dx1000-moving-20261013-run2", only=("reflectivity", "swath", "hail-track"))
    assert pictures
    for name, ov in pictures:
        assert ov.check() == [] and ov._head_rows == 1, name


def _retimed(source: Path, target: Path, time_s: float) -> Path:
    """A copy of a netCDF file whose one saved time is ``time_s`` (output after a restart, or a later file)."""

    shutil.copy(source, target)
    import netCDF4

    with netCDF4.Dataset(target, "a") as ds:
        ds.variables["time"][:] = [time_s]
    return target


def test_the_start_box_is_inside_the_swath_when_the_files_begin_late(tmp_path, pictures):
    path = cm1_file(tmp_path / "cm1out.nc", times=(3600.0, 4200.0), umove=10.0)
    # the domain at T+0 (cells 0 to nx) and at the last saved time (42 km on: cells 42 to 54) are both on the canvas
    assert N.discover(path).ground_cells() == (0, 0, 42 + 12, 10)
    late = tmp_path / "late"
    late.mkdir()
    _retimed(CM1_FILE, late / "cm1out_000013.nc", 3600.0)         # CM1's file, moving at 12.5 and 3 m/s
    layout = N.discover(late)
    x0, y0, x1, y1 = layout.ground_cells()
    assert (x0, y0) == (0, 0) and (x1, y1) == (45 + 24, 11 + 24)
    RD.render(late, only=("swath", "hail-track"))
    index = json.loads((late / "quicklook" / "index.json").read_text())
    assert all(len(p["frames"]) == 1 for p in index["products"] if p["id"] in ("swath", "hail-track"))
    # what the files' translated swaths leave out is said beside the pictures
    from cm1core.quicklook.swath import WINDOW_NOTE

    assert WINDOW_NOTE.startswith("Each file holds the swath only under the domain at its time")
    assert {name for name, _ov in pictures} == {"swath", "hail-track"}
    for name, ov in pictures:
        assert ov.check() == [] and "Each file holds the swath only" in " ".join(ov.texts), name


def test_a_2d_or_axisymmetric_cm1_run_is_refused_in_plain_words(tmp_path):
    flat = tmp_path / "xz"
    flat.mkdir()
    cm1_file(flat / "cm1out_000001.nc", ny=1, times=(0.0,))
    done = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(flat)], capture_output=True, text=True,
                          env=_env())
    assert done.returncode == 2 and "is a 2-D CM1 run (12 × 1 columns)" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr and "nan" not in done.stderr.lower()
    assert not (flat / "quicklook").exists()
    round_ = tmp_path / "axisymm"
    round_.mkdir()
    cm1_file(round_ / "cm1out_000001.nc", ny=1, times=(0.0,), attrs={"axisymm": np.int32(1)})
    with pytest.raises(N.NcError, match="axisymmetric CM1 run"):
        N.discover(round_)


def test_an_even_grid_as_cm1_writes_it_is_not_taken_for_a_stretched_one():
    """CM1 works the faces out in double precision, keeps them as REAL and writes 0.001 times them in km: a fine
    grid over a big domain comes back a few millimetres uneven, which is not CM1's stretch_x."""

    f32 = np.float32

    def written(dx: float, nx: int) -> np.ndarray:
        xfdp = np.arange(nx + 1, dtype=np.float64) * dx - 0.5 * nx * dx
        return N._metres((f32(0.001) * xfdp.astype(f32)).astype(f32), 1000.0)

    uneven = written(30.0, 2700)
    assert float(np.ptp(np.diff(uneven))) > 1e-4 * 30.0 + 1e-3    # beyond what was allowed before
    for dx, nx in ((30.0, 2700), (50.0, 4000), (67.0, 3000), (67.0, 736), (82.9, 800), (8000.0, 512)):
        assert N._spacing(written(dx, nx), "x", Path("cm1out.nc"), 1000.0) == pytest.approx(dx), (dx, nx)
    # a real stretch, even of one per cent across the domain, is still refused
    steps = np.linspace(100.0, 101.0, 400)
    faces = N._metres((f32(0.001) * np.concatenate([[0.0], np.cumsum(steps)]).astype(f32)).astype(f32), 1000.0)
    with pytest.raises(N.NcError, match="stretch_x"):
        N._spacing(faces, "x", Path("cm1out.nc"), 1000.0)


def test_every_reason_a_look_is_not_drawn_names_the_field_and_the_switch(tmp_path):
    folder = tmp_path / "th-only"
    folder.mkdir()
    # CM1's file with only th and prs written: no wind, w, rain, moisture or dbz, and no namelist beside it
    _without(CM1_FILE, folder / "cm1out_000001.nc",
             ("u", "v", "w", "dbz", "cref", "rain", "rain2", "prate", "qv", "qc", "qr", "qi", "qs", "qg", "ncr", "nci",
              "ncs", "ncg", "sws", "sws2", "svs", "svs2", "sgs", "sgs2", "shs", "shs2", "sps", "sps2", "srs", "srs2",
              "sus", "sus2", "uh", "zvort", "uinterp", "vinterp", "winterp", "xvort", "yvort"))
    summary = RD.render(folder)
    assert summary.problems == []
    index = json.loads((folder / "quicklook" / "index.json").read_text())
    reasons = {p["id"]: p["not_drawn"][0]["reason"] for p in index["products"] if p.get("not_drawn")}
    assert set(reasons) >= {"reflectivity", "spin-near-ground", "updraft-5km", "swath", "surface", "rain",
                            "clouds-from-above", "updraft-slice", "hail-track", "spin-track", "environment"}
    from cm1core.quicklook import runcard as RC

    for look, reason in reasons.items():
        if look in P.NEAR_GROUND:            # not a field the file lacks: a grid these looks are not drawn for
            assert reason.startswith("This look is drawn for grids of 250 m and finer"), (look, reason)
        elif look == "run-card":             # no netCDF file carries a GM1 run's receipt
            assert reason == RC.NETCDF_WHY
        else:
            assert reason.startswith("cm1out_000001.nc has no "), (look, reason)
            assert "output_" in reason and "Error" not in reason and "frame" not in reason, (look, reason)
    # the looks added beside these name the field too
    assert {"infrared", "spectra", "side-view", "water-paths", "updraft-time-height", "spin-time-height",
            "radar-velocity", "rotation-axis"} <= set(reasons)
    assert "(CM1 writes it with output_dbz = 1)" in reasons["reflectivity"]
    assert "namelist.input" in reasons["reflectivity"] and "ihail" in reasons["reflectivity"]
    assert reasons["rain"] == "cm1out_000001.nc has no rain at the ground (rain) (CM1 writes it with output_rain = 1)."


def test_without_its_namelist_cm1s_dense_ice_is_graupel_or_hail(tmp_path, pictures):
    folder = tmp_path / "cm1run"
    folder.mkdir()
    shutil.copy(CM1_FILE, folder / "cm1out_000001.nc")           # Morrison (ptype 5); ihail is in the namelist
    RD.render(folder, only=("hail-track", "reflectivity"))
    said = {name: " ".join(ov.texts) for name, ov in pictures}
    assert "GRAUPEL OR HAIL NEAR THE GROUND" in said["hail-track"] and "most graupel or hail" in said["hail-track"]
    assert "rain, snow and graupel or hail" in said["reflectivity"]


def test_files_on_two_grids_are_refused_before_anything_is_drawn(tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    cm1_file(folder / "cm1out_000001.nc", times=(0.0,))
    cm1_file(folder / "cm1out_000002.nc", times=(600.0,), nx=10)
    with pytest.raises(N.NcError, match="is on another grid than cm1out_000001.nc"):
        N.discover(folder)
    done = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(folder)], capture_output=True, text=True,
                          env=_env())
    assert done.returncode == 2 and not (folder / "quicklook").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    done = subprocess.run([sys.executable, "-m", "cm1core", "quicklook", str(empty)], capture_output=True, text=True,
                          env=_env())
    assert done.returncode == 2 and "CM1 netCDF history files" in done.stderr


def test_cm1s_reflectivity_chart_draws_no_echo_at_0_dbz_and_the_index_holds_no_nan():
    from cm1core.quicklook import series as SE

    rows = [(0.0, {"dbz_max": 0.0}), (300.0, {"dbz_max": -35.2}), (600.0, {"dbz_max": 42.0})]
    cm1 = SE.series_document(rows, z0=250.0, name="x, CM1 output", reference=None, t_end=None, model="CM1")
    gm1 = SE.series_document(rows, z0=250.0, name="x", reference=None, t_end=None)
    assert cm1["charts"][0]["gm1"] == [[0.0, 0.0], [300.0, 0.0], [600.0, 42.0]]
    assert cm1["charts"][0]["label"] == "highest reflectivity, no echo at 0"
    assert gm1["charts"][0]["gm1"][1] == [300.0, -35.2]          # GM1's own chart is as it was
    assert RD._finite({"a": [1.0, float("nan")], "b": {"c": float("inf")}}) == {"a": [1.0, None], "b": {"c": None}}
