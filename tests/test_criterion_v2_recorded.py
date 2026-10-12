"""The published criterion v2 record can be checked by a reader, and the verdicts page's commands run.

1. ``tools/criterion_v2/check_recorded.py`` decides every recorded verdict again from its published tables
   with the frozen rule and gets the recorded result; a changed table row is noticed.
2. ``tools/criterion_v2/window_excerpt.py`` cuts a stream down to what the scorer reads, and
   ``tools/criterion_v2/netcdf_to_window.py`` builds the same from CM1-style netCDF window records: the
   scorer's per-sample measurements on either equal those on the full stream.
3. ``docs/GM1-TORNADO-VERDICTS-V2.md``: every command block is marked as the guides mark theirs, the blocks
   a reader types without other data run from the top of the tree and exit 0, the printed lines shown are
   the ones the commands print, and the numbers the page quotes for the headline are the recorded ones.

Synthetic streams only; no GM1 run is read.  CPU only.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "criterion_v2"
RESULTS = TOOLS / "results"
PAGE = ROOT / "docs" / "GM1-TORNADO-VERDICTS-V2.md"
HEADLINE = "E100R2-PF-4m-window-pre3312"
NG = 3


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load("synthetic_v2", TOOLS / "synthetic_v2.py")
CHECK = _load("criterion_v2_check_recorded", TOOLS / "check_recorded.py")
EXCERPT = _load("window_excerpt", TOOLS / "window_excerpt.py")
NETCDF = _load("criterion_v2_netcdf_to_window", TOOLS / "netcdf_to_window.py")
SCORE = _load("criterion_v2_score", TOOLS / "score.py")


# ---------------------------------------------------------------------------
# 1. The recorded verdicts.
# ---------------------------------------------------------------------------
def test_every_recorded_verdict_is_decided_again_from_its_tables(capsys) -> None:
    assert CHECK.main([]) == 0
    printed = capsys.readouterr().out
    assert "17 of 17 recorded sources decided again with the same result" in printed
    assert "DIFFERENT" not in printed
    # The two sources that reach past 3312 s say so under their peak wind; the headline source does not.
    lines = printed.splitlines()
    noted = [lines[index - 2].split()[0] for index, line in enumerate(lines) if "do not quote this wind" in line]
    assert noted == ["E100R2-PF-4m-window", "E100R2-PF-4m-window-3312on"], noted


def test_the_headline_numbers_are_the_recorded_ones() -> None:
    folder = RESULTS / "scoring-v2-pf4m" / HEADLINE
    record = CHECK.check_source(folder, CHECK.velocities(RESULTS)[HEADLINE])
    assert record["same"], record["differs"]
    assert (record["recorded"], record["again"], record["samples"]) == ("TORNADO", "TORNADO", 12287)
    assert record["genesis_model_time_s"] == pytest.approx(2856.9375, abs=1e-3)
    assert record["longest_tracked_qualifying_s"] == pytest.approx(454.96875, abs=1e-3)
    assert record["episode_peak_ground_wind_m_s"] == pytest.approx(129.0447, abs=1e-3)
    verdict = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["matches_frozen"] and verdict["protocol_conforming"]
    assert verdict["record_end_s"] < 3312.0  # the headline span ends before the Courant number passed 1
    # The page says the same.
    page = PAGE.read_text(encoding="utf-8")
    for quoted in ("Genesis T+47:36.9 (2856.9 s)", "455.0 s, 4851 records", "129.0 m/s at T+55:11.9",
                   "127.6 m/s at T+54:28.1 before 3300 s",
                   "about 128 m/s (129.0 m/s in the last record before the time step limit was passed)"):
        assert quoted in page, quoted


def test_a_changed_table_is_noticed(tmp_path: Path) -> None:
    """Take the strongest minutes out of the headline table: the check must not call that the same."""

    results = tmp_path / "results"
    source = RESULTS / "scoring-v2-pf4m"
    (results / "scoring-v2-pf4m" / HEADLINE).mkdir(parents=True)
    shutil.copy(source / "tornado-verdicts-v2-pf4m.json", results / "scoring-v2-pf4m")
    shutil.copy(source / HEADLINE / "verdict.json", results / "scoring-v2-pf4m" / HEADLINE)
    for name in ("samples.tsv", "candidates.tsv"):
        lines = gzip.decompress((source / HEADLINE / (name + ".gz")).read_bytes()).decode().splitlines()
        column = lines[0].split("\t").index("class")
        time = lines[0].split("\t").index("model_time_s")
        changed = [lines[0]]
        for line in lines[1:]:
            cells = line.split("\t")
            if 3000.0 <= float(cells[time]) < 3010.0 and cells[column] == "Q":
                cells[column] = "N"       # a ten-second hole in the qualifying run
            changed.append("\t".join(cells))
        (results / "scoring-v2-pf4m" / HEADLINE / name).write_text("\n".join(changed) + "\n", encoding="utf-8")
    assert CHECK.main(["--results", os.fspath(results)]) == 1


# ---------------------------------------------------------------------------
# 2. Excerpts and netCDF records.
# ---------------------------------------------------------------------------
DX, NX, NY, NZ = 4.0, 225, 225, 6
ZH_FULL = [-2.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0]
TIMES = [3000.0 + 7.5 * k for k in range(18)]       # 127.5 s, after any spin-up
NEST = ["--nest-start-s", "2160", "--parent-dx", "100"]


def _planes() -> list[tuple[float, np.ndarray, np.ndarray]]:
    """(time, u, v) on NZ levels: a two-celled vortex that stays put in the moving frame, weaker aloft."""

    factor = np.array([0.8, 0.9, 1.0, 1.05, 1.1, 1.15])
    out = []
    for q, t in enumerate(TIMES):
        comp = S.VortexComponent("sullivan", 60.0, 100.0, 450.0 + 1.0 * q, 450.0, drift=False)
        plane = S.make_plane([comp], [], NX * DX, DX, 0.0)
        out.append((t, (factor[:, None, None] * plane.u[None]).astype("<f4"),
                    (factor[:, None, None] * plane.v[None]).astype("<f4")))
    return out


def _full_stream(root: Path) -> None:
    root.mkdir(parents=True)
    shapes = {"uadv": (NZ + 2, NY + 2 * NG, NX + 1 + 2 * NG), "vadv": (NZ + 2, NY + 1 + 2 * NG, NX + 2 * NG),
              "other": (NZ + 1, NY + 2, NX + 2)}
    fields, offset = [], 0
    for name, shape in shapes.items():
        fields.append({"name": name, "shape_zyx": list(shape), "offset_words": offset})
        offset += int(np.prod(shape))
    rows = ["\t".join(["index", "model_time_s", "offset_bytes", "bytes", "i0", "j0", "nx", "ny", "nz",
                       "origin_x_m", "origin_y_m", "sha256"])]
    with open(root / "window-stream.f32", "wb") as arena:
        for q, (t, u, v) in enumerate(_planes()):
            ua, va = np.zeros(shapes["uadv"], "<f4"), np.zeros(shapes["vadv"], "<f4")
            ua[1:NZ + 1, NG:NG + NY, NG:NG + NX + 1] = u
            va[1:NZ + 1, NG:NG + NY + 1, NG:NG + NX] = v
            blob = ua.tobytes() + va.tobytes() + np.full(shapes["other"], 7.0, "<f4").tobytes()
            arena.write(blob)
            rows.append("\t".join(str(x) for x in [100 + q, repr(t), q * len(blob), len(blob), 137, 137, NX, NY,
                                                   NZ, 1000.0, 2000.0, "0" * 64]))
    (root / "window-records.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    receipt = {"schema": "gm1.nest-child-window-stream/v3",
               "child": {"dx_m": DX, "dy_m": DX, "nx": 500, "ny": 500, "u_move_m_s": 16.5, "v_move_m_s": 16.5,
                         "spec_zone": 1, "relax_zone": 4},
               "window": {"nx": NX, "ny": NY, "nz": NZ}, "zh_cm1_indexed": ZH_FULL, "fields": fields}
    (root / "window-stream.json").write_text(json.dumps(receipt), encoding="utf-8")


def _score(stream: Path, out: Path) -> dict:
    assert SCORE.main([os.fspath(stream), "--out", os.fspath(out), *NEST]) == 0
    return json.loads((out / "verdict.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("criterion-v2-recorded")
    _full_stream(base / "stream")
    verdict = _score(base / "stream", base / "scored")
    assert verdict["outcome"] == "TORNADO", verdict
    return {"base": base, "stream": base / "stream", "scored": base / "scored", "verdict": verdict}


def test_an_excerpt_scores_as_the_full_stream_does(full: dict, tmp_path: Path) -> None:
    assert EXCERPT.main([os.fspath(full["stream"]), "--out", os.fspath(tmp_path / "all")]) == 0
    receipt = json.loads((tmp_path / "all" / "window-stream.json").read_text(encoding="utf-8"))
    assert receipt["window"]["nz"] == 3 and receipt["excerpt"]["scalar_levels_kept"] == 3   # 2, 6 and 10 m
    assert [field["name"] for field in receipt["fields"]] == ["uadv", "vadv"]
    small, big = (tmp_path / "all" / "window-stream.f32").stat().st_size, (full["stream"] / "window-stream.f32").stat().st_size
    assert small == 18 * receipt["record_bytes"] and small < 0.5 * big
    verdict = _score(tmp_path / "all", tmp_path / "all-scored")
    # every record kept: the two tables are the full stream's, byte for byte, and so is the verdict
    for name in ("samples.tsv", "candidates.tsv"):
        assert (tmp_path / "all-scored" / name).read_bytes() == (full["scored"] / name).read_bytes()
    for key in ("outcome", "genesis_model_time_s", "episodes", "class_counts", "longest_qualifying_run_s"):
        assert verdict[key] == full["verdict"][key], key


def test_a_thinned_excerpt_keeps_each_records_measurements(full: dict, tmp_path: Path) -> None:
    last = TIMES[-1]
    assert EXCERPT.main([os.fspath(full["stream"]), "--out", os.fspath(tmp_path / "thin"), "--every", "2",
                         "--last-at", repr(last), "--start", "3020"]) == 0
    rows = (tmp_path / "thin" / "window-records.tsv").read_text(encoding="utf-8").splitlines()[1:]
    times = [float(row.split("\t")[1]) for row in rows]
    assert times[-1] == last and times[0] >= 3020.0 and all(b - a == 15.0 for a, b in zip(times, times[1:]))
    _score(tmp_path / "thin", tmp_path / "thin-scored")
    record = CHECK.compare_rows(tmp_path / "thin-scored", full["scored"])
    assert record["same"] and record["samples"] == len(times), record
    # and a different wind is noticed
    arena = tmp_path / "thin" / "window-stream.f32"
    data = np.fromfile(arena, dtype="<f4")
    data *= np.float32(1.01)
    data.tofile(arena)
    _score(tmp_path / "thin", tmp_path / "thin-scored-2")
    assert not CHECK.compare_rows(tmp_path / "thin-scored-2", full["scored"])["same"]


def test_netcdf_window_records_score_as_the_stream_does(full: dict, tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    folder = tmp_path / "nc"
    folder.mkdir()
    for q, (t, u, v) in enumerate(_planes()):
        with netCDF4.Dataset(folder / f"cm1out_{q:06d}.nc", "w") as data:
            for name, size in (("xh", NX), ("yh", NY), ("zh", NZ), ("xf", NX + 1), ("yf", NY + 1), ("time", 1),
                               ("one", 1)):
                data.createDimension(name, size)
            data.createVariable("zh", "f4", ("zh",))[:] = np.array(ZH_FULL[1:NZ + 1]) / 1000.0     # km, as CM1
            data.createVariable("xf", "f4", ("xf",))[:] = (1000.0 + DX * np.arange(NX + 1)) / 1000.0
            data.createVariable("yf", "f4", ("yf",))[:] = (2000.0 + DX * np.arange(NY + 1)) / 1000.0
            data.createVariable("time", "f4", ("time",))[:] = t
            data.createVariable("umove", "f4", ("one",))[:] = 16.5
            data.createVariable("vmove", "f4", ("one",))[:] = 16.5
            data.createVariable("u", "f4", ("time", "zh", "yh", "xf"))[0] = u
            data.createVariable("v", "f4", ("time", "zh", "yf", "xh"))[0] = v
            data.setncatts({"gm1_record_index": 100 + q, "gm1_window_i0": 137, "gm1_window_j0": 137,
                            "gm1_window_origin_x_m": 1000.0, "gm1_window_origin_y_m": 2000.0,
                            "gm1_record_sha256": "0" * 64, "gm1_run": "TEST: a made-up vortex"})
    assert NETCDF.main([os.fspath(folder), "--out", os.fspath(tmp_path / "stream")]) == 0
    verdict = _score(tmp_path / "stream", tmp_path / "scored")
    assert verdict["outcome"] == "TORNADO"
    for name in ("samples.tsv", "candidates.tsv"):
        assert (tmp_path / "scored" / name).read_bytes() == (full["scored"] / name).read_bytes()
    # the part before a time, as the page does for 3312 s
    assert NETCDF.main([os.fspath(folder), "--out", os.fspath(tmp_path / "early"), "--before", "3060"]) == 0
    kept = (tmp_path / "early" / "window-records.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assert len(kept) == 8 and float(kept[-1].split("\t")[1]) == 3052.5


# ---------------------------------------------------------------------------
# 3. The page.
# ---------------------------------------------------------------------------
DOC_COMMANDS = runpy.run_path(str(ROOT / "tools" / "release" / "doc_commands.py"))


def test_the_pages_command_blocks_are_marked_and_the_typed_ones_run(tmp_path: Path) -> None:
    text = PAGE.read_text(encoding="utf-8")
    section = text[text.index("## How to check this yourself, and what cannot be checked yet"):
                   text.index("## What changed from v1")]
    kinds = [kind for kind, _body, _line in DOC_COMMANDS["blocks"](section)]
    assert set(kinds) <= {"sh", "sh example", "text"} and kinds.count("sh") >= 2 and kinds.count("sh example") >= 4
    env = {**os.environ, "HOME": os.fspath(tmp_path), "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "python3"                           # the page says python3: this Python
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}"
    printed = ""
    ran = 0
    for kind, body, line_no in DOC_COMMANDS["blocks"](section):
        for command in DOC_COMMANDS["lines"](body):
            if kind == "sh example":
                import shlex
                shlex.split(command, comments=True)      # checked for form: it needs data the tree does not bring
                continue
            if kind != "sh":
                continue
            done = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, capture_output=True, text=True,
                                  timeout=900)
            assert done.returncode == 0, f"line {line_no}: `{command}` exited {done.returncode}: {done.stderr[-600:]}"
            assert "Traceback" not in done.stderr
            printed += done.stdout
            ran += 1
    assert ran >= 3
    # every line the page shows under check 1 is a line the commands printed
    shown = [body for kind, body, _line in DOC_COMMANDS["blocks"](section) if kind == "text"][:2]
    for body in shown:
        for line in body.splitlines():
            assert line in printed.splitlines(), line
    # the table listing ends with the headline's last record: 3311.9062 s, Q, 129.0 m/s
    assert printed.rstrip().splitlines()[-1].split("\t") == ["3311.9062", "Q", "138.0000", "129.0447"]


def test_the_page_names_no_file_the_tree_does_not_have() -> None:
    """Every `tools/...`, `tests/...` and `docs/...` path the two criterion pages name exists here."""

    import re

    for page in (PAGE, ROOT / "docs" / "GM1-TORNADO-CRITERION-V2.md"):
        text = page.read_text(encoding="utf-8")
        for path in sorted(set(re.findall(r"`((?:tools|tests|docs)/[A-Za-z0-9_./-]+)`", text))):
            assert (ROOT / path.rstrip("/")).exists(), f"{page.name}: {path}"
