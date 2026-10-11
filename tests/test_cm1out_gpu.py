"""A GM1 run of CM1's 24 x 24 x 40 reference namelist, converted and held to CM1's own files.

Opt-in (``CM1_RUN_GPU_TESTS=1`` and ``GM1_NATIVE_RELEASE_DIR`` naming the
directory with ``libcm1_state.so`` and ``cm1_forecast_publish``): it imports
``tests/data/cm1-ref-24x24/namelist.input`` with the importer, runs it for
its 120 s with the GUI run flags (events, control directory, no envelope,
the t = 0 frame), converts the frames with ``gpuwm-cm1 cm1out`` and checks:

* ``cm1out_000002.nc`` has CM1's dimensions, CM1's coordinate words, CM1's 71
  global attributes and, for every variable GM1 writes, CM1's dimensions,
  units and long_name; Morrison's number concentrations are among them;
* ``cm1out_stats.nc`` carries CM1's names and units, and at t = 0, where
  both models hold the same initial state, every statistic is CM1's value.

About 15 s on an RTX 3080.  ``GM1_PYTHON`` names the interpreter whose CuPy
runs the model when the test's own has none.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1" or not os.environ.get("GM1_NATIVE_RELEASE_DIR"),
    reason="GPU run: set CM1_RUN_GPU_TESTS=1 and GM1_NATIVE_RELEASE_DIR",
)

netCDF4 = pytest.importorskip("netCDF4")

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import REF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: dbz and cref are written (Morrison: CM1's calc_refl10cm on the CPU).
NOT_PRODUCED_STOCK = {"srs", "sgs", "rain2", "srs2", "sgs2", "kmh", "kmv", "khh", "khv"}


def _words(array) -> list[str]:
    return [f"{int(w):08x}" for w in np.ascontiguousarray(np.asarray(array, dtype=np.float32)).view(np.uint32).reshape(-1)]


def _run_gm1(work: Path, namelist: Path, *extra: str, restart_s: float = 120.0) -> Path:
    """Import ``namelist``, run it with the GUI run flags, and return the run directory."""

    os.environ["GM1_CM1_R21_NAMELIST"] = os.fspath(namelist_tool())
    from cm1core.cm1import import import_namelist

    run = work / "run"
    run.mkdir()
    shutil.copy(namelist, run / "namelist.input")
    budget = int(os.environ.get("GM1_MEMORY_BUDGET_MIB", "5000"))
    imported = import_namelist(run / "namelist.input", gm1_only={"runtime.memory_budget_mib": budget})
    deck = imported.toml()
    # A GUI run restarts at its end when the namelist asks for no restart.
    deck = "\n".join(f"restart_interval_s = {restart_s}" if line.startswith("restart_interval_s") else line
                     for line in deck.splitlines()) + "\n"
    (run / "deck.toml").write_text(deck)
    native = Path(os.environ["GM1_NATIVE_RELEASE_DIR"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.fspath(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    command = [
        os.environ.get("GM1_PYTHON") or sys.executable, "-m", "cm1core", "cartesian", os.fspath(run / "deck.toml"),
        "--integrate", "--output", os.fspath(run / "receipt.json"),
        "--history-dir", os.fspath(run / "frames"), "--restart-dir", os.fspath(run / "restarts"),
        "--library", os.fspath(native / "libcm1_state.so"),
        "--history-publisher", os.fspath(native / "cm1_forecast_publish"),
        "--events", os.fspath(run / "events.jsonl"), "--control-dir", os.fspath(run / "control"),
        "--history-envelope", "off", "--history-initial", *extra,
    ]
    with (run / "run.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
    return run


@pytest.fixture(scope="module")
def gm1_run(tmp_path_factory):
    from cm1core.cm1out import convert

    run = _run_gm1(tmp_path_factory.mktemp("cm1out-gpu"), REF / "namelist.input")
    summary = convert(run)
    return run, summary


def test_the_gm1_run_converts_to_cm1s_layout(gm1_run):
    run, summary = gm1_run
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
    assert set(summary.not_produced) == NOT_PRODUCED_STOCK
    header = json.loads((REF / "cm1out_000002.header.json").read_text())
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_000002.nc")
    try:
        assert [[k, len(d), d.isunlimited()] for k, d in ds.dimensions.items()] == header["dimensions"]
        assert ds.ncattrs()[:71] == [name for name, _value in header["global_attributes"]]
        for name, value in header["global_attributes"]:
            if name != "CM1 version":
                assert np.asarray(ds.getncattr(name)).tolist() == value["value"], name
        for row in header["variables"]:
            if row["name"] not in ds.variables:
                assert row["name"] in NOT_PRODUCED_STOCK
                continue
            var = ds[row["name"]]
            assert list(var.dimensions) == row["dimensions"], row["name"]
            assert {a: var.getncattr(a) for a in var.ncattrs()} == row["attributes"], row["name"]
            if "float32_words" in row:
                assert _words(var[:]) == row["float32_words"], row["name"]
        for name in ("nci", "ncs", "ncr", "ncg"):
            assert ds[name].units == "#/kg" and np.isfinite(ds[name][:]).all()
        # Morrison's reflectivity, formed after the frame (the dims and attributes are held above).
        assert json.loads(ds.getncattr("gm1_formed"))["dbz"] == "derived"
        assert np.isfinite(ds["dbz"][:]).all() and float(ds["cref"][:].min()) >= float(ds["dbz"][:].min())
    finally:
        ds.close()
    with netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc") as ds:
        # t = 0: CM1's zero start, as CM1's own cm1out_000001.nc holds.
        assert not np.asarray(ds["dbz"][:]).any() and not np.asarray(ds["cref"][:]).any()


def test_the_statistics_carry_cm1s_names_and_start_on_cm1s_words(gm1_run):
    run, _summary = gm1_run
    header = json.loads((REF / "cm1out_stats.header.json").read_text())
    cm1 = {row["name"]: row for row in header["variables"]}
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_stats.nc")
    try:
        ours = [v for v in ds.variables if v not in ("xh", "yh", "zh", "time")]
        assert len(ours) == 100
        order = [row["name"] for row in header["variables"] if row["name"] in ds.variables]
        assert list(ds.variables) == order
        for name in ours:
            assert {a: ds[name].getncattr(a) for a in ds[name].ncattrs()} == cm1[name]["attributes"], name
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0]
        # t = 0: the same initial state gives CM1's own statistics.  Compared as
        # IEEE values: GM1's initial w holds -0.0 where CM1's holds +0.0, and
        # maxmin.F keeps the first zero it meets, so wmax/wmin are -0.0 here.
        first = lambda words: np.uint32(int(words[0], 16)).view(np.float32)  # noqa: E731
        differ = [n for n in ours if np.float32(ds[n][0]) != first(cm1[n]["float32_words"])]
        assert differ == []
        signed_zero = [n for n in ours if _words(ds[n][0:1]) != cm1[n]["float32_words"][:1]]
        assert set(signed_zero) <= {"wmax", "wmin"}
    finally:
        ds.close()


def test_the_user_script_runs_on_the_gm1_file(gm1_run):
    run, _summary = gm1_run
    python = os.environ.get("GM1_XARRAY_PYTHON") or sys.executable
    if subprocess.run([python, "-c", "import xarray, pandas"], capture_output=True).returncode != 0:
        pytest.skip(f"{python} has no xarray/pandas; set GM1_XARRAY_PYTHON")
    script = ROOT / "tools" / "release" / "cm1_user_script.py"
    done = subprocess.run([python, "-W", "ignore", os.fspath(script), os.fspath(run / "cm1out" / "cm1out_000002.nc")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "wmax =" in done.stdout


def test_a_stretched_gm1_run_writes_cm1s_vertical_coordinates(tmp_path_factory):
    """The stretch_z = 1 variant GM1 runs: zh, zf, ztop and the Fornberg attributes are CM1's words.

    Also the reduced carrier profile (prs and zf dropped from the frame): prs
    comes back from pi0 + pipert, and zf from the frame's profile column.
    """

    import re

    from cm1core.cm1out import convert

    variants = json.loads((REF / "variants.json").read_text())
    text = (REF / "namelist.input").read_text()
    for key, value in variants["edits"]["stretchg"].items():
        text, count = re.subn(rf"^(\s*{re.escape(key)}\s*=\s*)([^,\n]*)(,?)",
                              lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1, flags=re.M)
        assert count == 1, key
    work = tmp_path_factory.mktemp("cm1out-gpu-stretch")
    namelist = work / "stretch.namelist.input"
    namelist.write_text(text)
    run = _run_gm1(work, namelist, "--history-carriers", "reconstructible-omitted", restart_s=60.0)
    summary = convert(run)
    record = variants["variants"]["stretchg"]
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc"]
    # The importer turns the rotation products off on a stretched column; the reason says so.
    assert "stretched" in summary.not_produced["zvort"] and "stretched" in summary.not_produced["shs2"]
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_000002.nc")
    try:
        for name in ("zh", "zf", "ztop"):
            assert _words(ds[name][:]) == record["float32_words"][name], name
        for name, words in record["global_attributes"].items():
            assert _words([ds.getncattr(name)]) == words, name
        cm1 = [row[0] for row in record["variables"]]
        assert [v for v in cm1 if v in ds.variables] == [v for v in ds.variables if ds[v].ndim >= 2]
        assert json.loads(ds.getncattr("gm1_formed"))["prs"] == "derived"
        assert np.isfinite(ds["prs"][:]).all() and float(ds["prs"][0].min()) > 5000.0
    finally:
        ds.close()
