"""``gpuwm-cm1 cm1out`` GrADS output (``output_format = 1``) held to CM1 r21.1's own files.

``tests/data/cm1-ref-24x24/grads/`` holds what CM1 r21.1 itself wrote for the
24 x 24 x 40 conformance namelist with ``output_format = 1`` (README there):
its descriptors, its metadata and statistics records, and the sha256 of every
t = 0 history record.  A frame carrying CM1's own t = 0 words must convert to
CM1's own records, and each descriptor must be CM1's line for line, apart
from what the module says differs (the title's version, tdef, and the
variables GM1 does not produce).  The oracle's CM1 GrADS reader
(``cm1core.oracle.grads``), written against CM1's own output, must read ours.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from cm1core.cm1out import convert  # noqa: E402
from cm1core.cm1out import grads as G  # noqa: E402
from cm1core.cm1out import settings as S  # noqa: E402
from cm1core.oracle.grads import open_history, read_ctl  # noqa: E402

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import REF, frame_from_cm1, publish  # noqa: E402

GREF = REF / "grads"
REFERENCE = json.loads((GREF / "reference.json").read_text(encoding="utf-8"))
#: What GM1 does not write for the conformance namelist (tests/test_cm1out.py NOT_PRODUCED_STOCK).
NOT_PRODUCED = {"srs", "sgs", "rain2", "srs2", "sgs2", "kmh", "kmv", "khh", "khv"}


@pytest.fixture()
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def _frames_at(times: list[float], *, zf: np.ndarray | None = None) -> list[tuple[dict, dict]]:
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    if zf is not None:
        carriers = dict(carriers, zf=np.broadcast_to(zf.reshape(-1, 1, 1), (41, 24, 24)).copy())
    return [(carriers, dict(frame, time_s=t, step=int(round(t / 6.0)), domainloc=(12.5 * t, 3.0 * t)))
            for t in times]


def _run(tmp_path: Path, variant: str, frames: list[tuple[dict, dict]], name: str = "run") -> Path:
    run = tmp_path / name
    root = run / "segments" / "001" / "frames"
    root.mkdir(parents=True)
    shutil.copy(GREF / variant / "namelist.input", run / "namelist.input")
    for index, (carriers, frame) in enumerate(frames):
        publish(root, index, carriers, **frame)
    return run


def _lines(text: str) -> list[str]:
    return text.split("\n")


def _normalised_cm1(path: Path, *, drop: set[str], tdef: int | None = None) -> list[str]:
    """CM1's descriptor as GM1 writes it: GM1's title version, its tdef, the produced variables only."""

    lines = _lines(path.read_text(encoding="ascii"))
    out = []
    count = None
    for line in lines:
        if line.startswith("title "):
            line = line.replace("using version cm1r21.1", f"using version {G.GRADS_VERSION}")
        elif line.startswith("tdef") and tdef is not None:
            line = f"tdef {tdef:10d}" + line[15:]
        elif line.startswith("vars "):
            count = len(out)
        elif count is not None and line.split() and line.split()[0] in drop:
            continue
        out.append(line)
    kept = [line for line in out[count + 1:] if line.split() and line.split()[0] != "endvars"]
    out[count] = f"vars {len(kept):6d}"
    return out


# ---------------------------------------------------------------- descriptors, line for line

@pytest.mark.parametrize("variant,filetype", [("g1", 1), ("g2", 2), ("gn", 2), ("gu", 2)])
def test_the_descriptors_are_cm1s_line_for_line(tmp_path, parser_tool, variant, filetype):
    times = [0.0, 60.0, 120.0] if variant == "g1" else [0.0, 60.0]
    run = _run(tmp_path, variant, _frames_at(times))
    summary = convert(run)
    assert summary.format == "grads" and summary.filetype == filetype
    out = run / "cm1out"
    for ctl in sorted((GREF / variant).glob("cm1out_[suvw].ctl")) + [GREF / variant / "cm1out_metadata.ctl"]:
        mine = _lines((out / ctl.name).read_text(encoding="ascii"))
        assert mine == _normalised_cm1(ctl, drop=NOT_PRODUCED), ctl.name
    # The metadata records are CM1's words: mtime, nstep, dt, nwrite (and the domain's motion).
    assert (out / "cm1out_metadata.dat").read_bytes() == (GREF / variant / "cm1out_metadata.dat").read_bytes()


def test_output_filetype_3_names_the_files_gm1s_one_process_wrote(tmp_path, parser_tool):
    run = _run(tmp_path, "g3", _frames_at([0.0, 60.0]))
    summary = convert(run)
    out = run / "cm1out"
    assert {"cm1out_000000_000001_s.dat", "cm1out_000000_000002_w.dat"} <= set(summary.files)
    for grid in ("s", "u", "v", "w"):
        mine = _lines((out / f"cm1out_{grid}.ctl").read_text(encoding="ascii"))
        cm1 = _normalised_cm1(GREF / "g3" / f"cm1out_{grid}.ctl", drop=NOT_PRODUCED)
        # CM1's dset names the joined files its per-process files are combined into.
        assert cm1[0].startswith(f"dset ^cm1out_00%y4_{grid}.dat")
        assert mine[0] == f"dset ^{f'cm1out_000000_00%y4_{grid}.dat':<120}"
        assert mine[1:] == cm1[1:]
    history = open_history(out)
    assert history.datasets["s"].path_for(2).name == "cm1out_000000_000002_s.dat"
    assert history.read("th", 2)[0].tobytes() == history.read("th", 1)[0].tobytes()


def test_a_stretched_column_lists_its_actual_levels(tmp_path, parser_tool):
    """stretch_z's descriptor lists zh (and zf for w), as CM1's own does, word for word."""

    from cm1core.cm1_vertical_grid import build_vertical_levels

    settings = S.from_namelist(GREF / "gs" / "namelist.input")
    levels = build_vertical_levels(
        stretch_z=1, nk=40, dz_m=settings.float("dz"), ztop_m=settings.float("ztop"),
        str_bot_m=settings.float("str_bot"), str_top_m=settings.float("str_top"),
        dz_bot_m=settings.float("dz_bot"), dz_top_m=settings.float("dz_top"))
    zf = np.asarray(levels.zf, dtype=np.float32).reshape(-1)
    zf = zf if zf.size == 41 else zf[1:42]
    run = _run(tmp_path, "gs", _frames_at([0.0, 60.0], zf=zf))
    convert(run, stats=False)
    out = run / "cm1out"
    for grid in ("s", "u", "v", "w"):
        mine = _lines((out / f"cm1out_{grid}.ctl").read_text(encoding="ascii"))
        assert mine == _normalised_cm1(GREF / "gs" / f"cm1out_{grid}.ctl", drop=NOT_PRODUCED), grid
    s = read_ctl(out / "cm1out_s.ctl")
    w = read_ctl(out / "cm1out_w.ctl")
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    assert np.allclose(s.levels, zh / 1000.0, rtol=0, atol=5e-7 * zh.max() / 1000.0)
    assert len(w.levels) == 41 and w.levels[0] == 0.0
    # a stretched column that the settings do not call stretched is still listed, never "linear"
    uniform_words = S.from_namelist(GREF / "g2" / "namelist.input")
    geometry = S.Geometry(24, 24, 40, 1000.0, 1000.0, 500.0, zf, zh,
                          (np.float32(500.0) / (zf[1:] - zf[:-1])).astype(np.float32),
                          np.ones(40, np.float32), 6.0)
    text = G.history_ctl("s", [G.GradsVariable("th", "potential temperature", "K", "s")], uniform_words,
                         geometry, filetype=2, times=1)
    assert "zdef     40 levels " in text and " linear 2.500000E-01" not in text


# ---------------------------------------------------------------- records

def test_cm1s_t0_words_convert_to_cm1s_own_records(tmp_path, parser_tool):
    """Each t = 0 record GM1 writes is CM1's own record (sha256 recorded from CM1's cm1out_s.dat ...)."""

    run = _run(tmp_path, "g1", _frames_at([0.0]))
    convert(run, stats=False)
    out = run / "cm1out"
    recorded = REFERENCE["g1_t0_record_sha256"]
    compared = 0
    for grid in ("s", "u", "v", "w"):
        ctl = read_ctl(out / f"cm1out_{grid}.ctl")
        raw = (out / f"cm1out_{grid}.dat").read_bytes()
        plane = ctl.nx * ctl.ny
        for variable in ctl.variables:
            start = ctl.offset_words(variable.name) * 4
            block = raw[start:start + variable.nz * plane * 4]
            assert hashlib.sha256(block).hexdigest() == recorded[f"{grid}/{variable.name}"], variable.name
            compared += 1
    assert compared == len(recorded) - len(NOT_PRODUCED)
    # dbz and cref among them: CM1 writes its zero start before the first microphysics step.
    assert "s/dbz" in recorded and "s/cref" in recorded
    notes = json.loads((out / "cm1out_gm1.json").read_text(encoding="utf-8"))
    assert set(notes["not_produced"]) == NOT_PRODUCED and all(notes["not_produced"].values())
    assert notes["formed"]["dbz"] == "derived"


def test_the_oracles_cm1_reader_reads_every_time_back(tmp_path, parser_tool):
    """The reader written against CM1's own GrADS output reads GM1's: times, steps and every word."""

    frames = _frames_at([0.0, 60.0, 120.0])
    for index, (carriers, _frame) in enumerate(frames):
        carriers["w3d"] = np.full_like(carriers["w3d"], np.float32(index + 1))
        carriers["th"] = (carriers["th"] + np.float32(index)).astype(np.float32)
    for filetype in (1, 2):
        run = _run(tmp_path, "g1", frames, name=f"run{filetype}")
        convert(run, filetype=filetype)
        history = open_history(run / "cm1out")
        assert history.model_time_s == (0.0, 60.0, 120.0) and history.step == (0, 10, 20)
        assert history.dt_s == (6.0, 6.0, 6.0)
        for index, (carriers, _frame) in enumerate(frames, 1):
            words, shape = history.read("w", index)
            assert shape == (41, 24, 24)
            assert np.array_equal(np.frombuffer(words, np.float32), carriers["w3d"].ravel())
            th, _shape = history.read("th", index)
            assert np.array_equal(np.frombuffer(th, np.float32), carriers["th"].ravel())


def test_statistics_are_cm1s_grads_record_and_descriptor(tmp_path, parser_tool):
    run = _run(tmp_path, "g1", _frames_at([0.0, 60.0]))
    summary = convert(run)
    out = run / "cm1out"
    assert summary.stats_file == "cm1out_stats.ctl"
    mine = _lines((out / "cm1out_stats.ctl").read_text(encoding="ascii"))
    cm1 = _lines((GREF / "g1" / "cm1out_stats.ctl").read_text(encoding="ascii"))
    assert mine[0] == cm1[0]
    assert mine[2:6] == cm1[2:6]
    assert mine[6] == "tdef          2 linear 00:00Z01JAN0001 1YR"
    lines = {line.split()[0]: line for line in cm1[8:-2]}
    ours = mine[8:-2]
    assert ours and all(line == lines[line.split()[0]] for line in ours), "CM1's name, description and unit"
    order = [line.split()[0] for line in cm1[8:-2]]
    names = [line.split()[0] for line in ours]
    assert names == [name for name in order if name in names], "CM1's order"
    table = np.fromfile(out / "cm1out_stats.dat", dtype="<f4").reshape(2, len(names))
    assert list(table[:, names.index("mtime")]) == [0.0, 60.0]
    # CM1's own t = 0 record, where both hold the same initial state: equal for most statistics.
    cm1_table = np.fromfile(GREF / "g1" / "cm1out_stats.dat", dtype="<f4").reshape(-1, len(order))
    same = sum(table[0, i] == cm1_table[0, order.index(n)] for i, n in enumerate(names))
    assert same >= 0.8 * len(names)


def test_a_variable_a_later_frame_cannot_supply_is_undef_there(tmp_path, parser_tool):
    frames = _frames_at([0.0, 60.0])
    frames[1] = (dict(frames[1][0]), frames[1][1])
    del frames[1][0]["uh"]
    run = _run(tmp_path, "g2", frames)
    summary = convert(run, stats=False)
    history = open_history(run / "cm1out")
    first = np.frombuffer(history.read("uh", 1)[0], np.float32)
    second = np.frombuffer(history.read("uh", 2)[0], np.float32)
    assert np.array_equal(first, frames[0][0]["uh"].ravel())
    assert np.all(second == np.float32(G.UNDEF))
    assert "uh" in summary.notes


def test_a_resumed_segment_rewrites_the_times_it_repeats(tmp_path, parser_tool):
    frames = _frames_at([0.0, 60.0, 120.0])
    for filetype in (1, 2):
        run = _run(tmp_path, "g1", frames, name=f"run{filetype}")
        convert(run, filetype=filetype, stats=False)
        root2 = run / "segments" / "002" / "frames"
        root2.mkdir(parents=True)
        for index, (carriers, frame) in enumerate(_frames_at([60.0, 120.0, 180.0])):
            publish(root2, index, dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(7.0))), **frame)
        convert(run, filetype=filetype, stats=False)
        history = open_history(run / "cm1out")
        assert history.model_time_s == (0.0, 60.0, 120.0, 180.0)
        assert read_ctl(run / "cm1out" / "cm1out_w.ctl").times == 4
        peaks = [float(np.frombuffer(history.read("w", i)[0], np.float32).max()) for i in range(1, 5)]
        assert peaks[1:] == [7.0, 7.0, 7.0] and peaks[0] != 7.0


def test_grads_and_netcdf_share_a_folder_and_settings_can_ask_for_both(tmp_path, parser_tool):
    run = _run(tmp_path, "g2", _frames_at([0.0, 60.0]))
    (run / "settings.toml").write_text("[output]\nformat = 1\nfiletype = 2\nalso_netcdf = true\n", encoding="utf-8")
    summary = convert(run)
    out = run / "cm1out"
    assert summary.format == "grads" and summary.also is not None
    assert summary.also.files == ["cm1out_000001.nc", "cm1out_000002.nc"]
    assert (out / "cm1out_stats.nc").is_file() and (out / "cm1out_stats.dat").is_file()
    record = summary.record()
    assert record["also"]["format"] == "nc"
    # Starting one format again leaves the other alone.
    convert(run, fmt="nc", force=True, also_netcdf=False)
    assert (out / "cm1out_000001_s.dat").is_file() and (out / "cm1out_s.ctl").is_file()
    convert(run, force=True, also_netcdf=False)
    assert (out / "cm1out_000001.nc").is_file()


def test_the_cli_writes_grads_and_says_so(tmp_path, parser_tool, capsys):
    from cm1core.cli import main

    run = _run(tmp_path, "g2", _frames_at([0.0]))
    assert main(["cm1out", str(run)]) == 0
    printed = capsys.readouterr().out
    assert "Wrote grads (output_filetype = 2)" in printed
    assert f"open with grads: open {run / 'cm1out'}/cm1out_s.ctl" in printed
    assert main(["cm1out", str(run), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["format"] == "grads" and "cm1out_s.ctl" in record["files"]


# ---------------------------------------------------------------- outside readers

def _cdo_imports_a_template(folder: Path) -> bool:
    """Whether the installed CDO's import_binary reads any templated descriptor.

    CM1's descriptors (and so GM1's) name one file per time with
    ``options template``.  Some CDO builds abort on every such descriptor
    (the CDO 2.5.4-1 package of Ubuntu 26.04 ends with "buffer overflow
    detected" on this two-time, one-variable example), which says nothing
    about the descriptor under test.
    """

    folder.mkdir()
    for n in (1, 2):
        np.zeros((2, 2), dtype=">f4").tofile(folder / f"probe_00000{n}.dat")
    (folder / "probe.ctl").write_text(
        "dset ^probe_00%y4.dat\noptions template\nundef -999.\ntitle probe\n"
        "xdef 2 linear 0 1\nydef 2 linear 0 1\nzdef 1 levels 0\n"
        "tdef 2 linear 00:00Z01JAN0001 1YR\nvars 1\nv 0 99 probe\nendvars\n",
        encoding="ascii",
    )
    probe = subprocess.run(["cdo", "-s", "import_binary", "probe.ctl", "probe.nc"],
                           cwd=folder, capture_output=True, text=True)
    return probe.returncode == 0


@pytest.mark.skipif(shutil.which("cdo") is None, reason="CDO is not installed here")
def test_cdo_imports_the_binary(tmp_path, parser_tool):
    """Where CDO is installed, cdo import_binary reads the stretched run's descriptor."""

    if not _cdo_imports_a_template(tmp_path / "cdo-probe"):
        pytest.skip("the installed CDO's import_binary fails on a minimal templated GrADS descriptor too")

    from cm1core.cm1_vertical_grid import build_vertical_levels

    settings = S.from_namelist(GREF / "gs" / "namelist.input")
    levels = build_vertical_levels(
        stretch_z=1, nk=40, dz_m=settings.float("dz"), ztop_m=settings.float("ztop"),
        str_bot_m=settings.float("str_bot"), str_top_m=settings.float("str_top"),
        dz_bot_m=settings.float("dz_bot"), dz_top_m=settings.float("dz_top"))
    zf = np.asarray(levels.zf, dtype=np.float32).reshape(-1)
    run = _run(tmp_path, "gs", _frames_at([0.0, 60.0], zf=zf if zf.size == 41 else zf[1:42]))
    convert(run, stats=False)
    out = run / "cm1out"
    completed = subprocess.run(["cdo", "-s", "import_binary", "cm1out_s.ctl", str(tmp_path / "s.nc")],
                               cwd=out, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_a_left_branch_leaves_both_formats_and_only_its_own_times(tmp_path, parser_tool):
    """A resume from the 60 s restart: 180 s belongs to the branch the run left, in GrADS and in netCDF."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, RunDirectory, write_json_atomic

    run = _run(tmp_path, "g2", _frames_at([0.0, 60.0, 120.0, 180.0]))
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": "run", "segments": [
        {"number": 1, "directory": "segments/001", "resumed_from": None}]})
    convert(run, also_netcdf=True)
    out = run / "cm1out"
    assert (out / "cm1out_000004_s.dat").is_file() and (out / "cm1out_000004.nc").is_file()
    root2 = run / "segments" / "002" / "frames"
    root2.mkdir(parents=True)
    # a resumed segment's frames go on numbering from the restart's (frame-000002 at 120 s)
    for carriers, frame in _frames_at([120.0]):
        publish(root2, 2, dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(5.0))), **frame)
    RunDirectory(run).update(lambda record: record["segments"].append(
        {"number": 2, "directory": "segments/002", "resumed_from": {"segment": 1, "model_time_s": 60.0,
                                                                      "restart": 1}}))
    summary = convert(run, also_netcdf=True)
    assert not (out / "cm1out_000004_s.dat").exists() and not (out / "cm1out_000004.nc").exists()
    assert read_ctl(out / "cm1out_s.ctl").times == 3
    history = open_history(out)
    assert history.model_time_s == (0.0, 60.0, 120.0)
    assert float(np.frombuffer(history.read("w", 3)[0], np.float32).max()) == 5.0
    assert np.fromfile(out / "cm1out_stats.dat", dtype="<f4").size == 3 * len(summary.statistics)
    import netCDF4

    with netCDF4.Dataset(out / "cm1out_stats.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0]
    assert summary.also.files == ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
