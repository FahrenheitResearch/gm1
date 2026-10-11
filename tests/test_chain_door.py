"""``gpuwm-cm1 chain`` on the command line, host only.

* ``chain inspect`` says what a history is, as JSON and as plain lines, and draws the domain map with the pixel
  box a page turns a drag into cells with.
* ``chain check`` exits 0 when every check passes and 3 when one does not, printing every check in plain words.
* ``chain new`` writes an ordinary run folder (the namelist imported as any namelist is, the prepared products in
  ``RUNDIR/chain``, the prepared t0 column as ``input_sounding``, ``[chain]`` in the deck); the line it echoes is
  the line that was typed; a second ``chain new`` into the same folder is refused.
* ``chain frames-from-gm1`` turns a GM1 run's frames into an idealized history the chain admits and prepares.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("netCDF4")

from cm1core import arwen_chain as chain  # noqa: E402
from cm1core import chain_door as door  # noqa: E402
from cm1import_support import namelist_tool  # noqa: E402
from test_arwen_chain_wrf import START, history  # noqa: E402
import wrf_support as ws  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gm1(*words: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.fspath(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "",
           "GM1_CM1_R21_NAMELIST": os.fspath(namelist_tool()), "OMP_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "-m", "cm1core", "chain", *words], env=env, capture_output=True,
                          text=True, timeout=300)
    if check:
        assert done.returncode == 0, (words, done.stdout[-3000:], done.stderr[-3000:])
    return done


def test_inspect_says_what_the_history_is_and_draws_the_map(tmp_path):
    path = history(tmp_path, mp=8, use_theta_m=1, thm="dry")
    png = tmp_path / "map.png"
    data = json.loads(gm1("inspect", os.fspath(path), "--map-png", os.fspath(png), "--map-time",
                          ws.stamp(START, 300), "--json", check=True).stdout)
    assert data["producer"] == "WRF" and data["count"] == 3 and data["intervals_s"] == [300.0]
    assert data["grid"]["west_east"] == 16 and data["grid"]["dx_m"] == 3000.0 and data["grid"]["map_proj"] == 1
    assert data["microphysics"] == {"mp_physics": 8, "scheme": "Thompson (8)", "component": "thompson-mp8",
                                    "offered": True}
    assert data["verified"] == door.VERIFIED and data["command"].endswith("--json")
    m = data["map"]
    assert png.is_file() and m["time"] == ws.stamp(START, 300) and m["cells"] == {"west_east": 16, "south_north": 16}
    box = m["domain_px"]
    assert 0 <= box["left"] and box["left"] + box["width"] <= m["width"] + 1e-6
    assert 0 <= box["top"] and box["top"] + box["height"] <= m["height"] + 1e-6
    assert abs(box["width"] / box["height"] - 1.0) < 1e-9          # square cells stay square
    plain = gm1("inspect", os.fspath(path), check=True).stdout
    assert plain.splitlines()[0] == f"WRF history, 3 saved time(s): {START} .. {ws.stamp(START, 600)}"
    assert "microphysics Thompson (8) -> GM1 thompson-mp8" in plain


def test_check_exits_0_when_admitted_and_3_when_not_with_every_check_in_plain_words(tmp_path):
    good = history(tmp_path / "good", mp=6)
    done = gm1("check", "--history", os.fspath(good), "--window", "5,5,5,5", "--ratio", "3")
    assert done.returncode == 0, done.stdout
    lines = done.stdout.splitlines()
    assert sum(line.startswith("  ok ") for line in lines) == len(chain.ADMISSION_CHECKS)
    assert "Every check passes. GM1 can run this box." in lines and door.VERIFIED in lines
    bad = history(tmp_path / "bad", mp=6, terrain_m=400.0)
    done = gm1("check", "--history", os.fspath(bad), "--window", "5,5,5,5", "--ratio", "3", "--json")
    assert done.returncode == 3
    data = json.loads(done.stdout)
    rows = {row["id"]: row for row in data["checks"]}
    assert data["admitted"] is False and rows["terrain"]["status"] == "refused"
    assert rows["terrain"]["code"] == "GM1-CHAIN-TERRAIN" and "over the 50 m limit" in rows["terrain"]["text"]
    assert all(row["status"] == "pass" for key, row in rows.items() if key != "terrain")
    # A ratio GM1 cannot nest exactly, and a box on the edge, are named too.
    done = gm1("check", "--history", os.fspath(good), "--window", "1,1,5,5", "--ratio", "4", "--json")
    rows = {row["id"]: row for row in json.loads(done.stdout)["checks"]}
    assert done.returncode == 3 and rows["placement"]["status"] == "refused"


def test_new_writes_an_ordinary_run_folder_and_echoes_the_line_typed(tmp_path):
    path = history(tmp_path, mp=6, use_theta_m=1, thm="dry")
    out = tmp_path / "runs" / "box"
    words = ["new", "--history", os.fspath(path), "--window", "5,5,5,5", "--ratio", "3", "--end",
             ws.stamp(START, 600), "--out", os.fspath(out), "--name", "my box", "--json"]
    data = json.loads(gm1(*words, check=True).stdout)
    import shlex

    assert data["command"] == shlex.join(["gpuwm-cm1", "chain", "new", "--history", os.fspath(path), "--window",
                                          "5,5,5,5", "--ratio", "3", "--end", ws.stamp(START, 600), "--out",
                                          os.fspath(out), "--name", "my box", "--json"])
    assert data["run"] == os.fspath(out) and data["start"] == f"gpuwm-cm1 run {out}"
    run = json.loads((out / "run.json").read_text())
    assert run["schema"] == "gm1.cm1-run/v1" and run["name"] == "my box"
    assert run["chain"]["table"]["prepared_directory"] == os.fspath(out / "chain")
    assert run["chain"]["verified"] == door.VERIFIED
    deck = (out / "deck.toml").read_text()
    assert "[chain]" in deck and f'prepared_directory = "{out / "chain"}"' in deck
    assert (out / "input_sounding").read_bytes() == (out / "chain" / chain.SOUNDING).read_bytes()
    # The import report names the run folder's own namelist, not the staging copy it was read from.
    report = (out / "import-report.txt").read_text()
    assert f"CM1 namelist: {out / 'namelist.input'}" in report and "staging" not in report
    import re

    said = {m.group(1): m.group(2) for m in re.finditer(r"^\s*(\w+)\s*=\s*([^,\n]+),", (out / "namelist.input").read_text(),
                                                          re.M)}
    assert (int(said["nx"]), int(said["ny"])) == (15, 15) and float(said["dx"]) == 1000.0
    assert int(said["iinit"]) == 0 and int(said["isnd"]) == 7 and int(said["irandp"]) == 0
    again = gm1(*words)
    assert again.returncode == 2 and "already exists" in again.stdout


def test_frames_from_gm1_write_an_idealized_history_the_chain_admits(tmp_path):
    from cm1out_support import morrison_frame, morrison_state
    from cm1run_support import make_run

    run = make_run(tmp_path, duration=600.0, history=300.0)
    (run / "deck.toml").write_text((ROOT / "examples/cartesian-supercell-morrison-hail.toml").read_text())
    frames_dir = run / "segments" / "001" / "frames"
    frames_dir.mkdir(parents=True)
    nz, ny, nx = 30, 16, 16
    state = morrison_state(nz=nz, ny=ny, nx=nx)
    z = (np.arange(nz) + 0.5) * 500.0
    theta = ws.theta(z).astype(np.float32)
    pi = ws.exner(z, 97000.0)
    state["th0"] = theta
    state["pi0"] = pi.astype(np.float32)
    state["th3d"] = np.zeros((nz, ny, nx), np.float32)
    extra = {
        "th": np.broadcast_to(theta[:, None, None], (nz, ny, nx)).astype(np.float32),
        "prs": np.broadcast_to((chain.CM1_P00_PA * pi ** (chain.CM1_CP / chain.CM1_RD))[:, None, None],
                               (nz, ny, nx)).astype(np.float32),
        "terrain_msl": np.zeros((ny, nx), np.float32),
    }
    for index, t in enumerate((0.0, 300.0, 600.0)):
        published = morrison_frame(frames_dir, index, time_s=t, step=int(t / 6.0), state=state)
        # The runtime's frames carry th, prs and the ground as well; append them to the arena.
        table = published / "history-field-table.tsv"
        rows = table.read_text().splitlines()
        with (published / "history-carriers.f32").open("ab") as arena:
            offset = arena.tell()
            for name, array in extra.items():
                data = np.ascontiguousarray(array, "<f4")
                arena.write(data.tobytes())
                shape = ",".join(map(str, data.shape))
                rows.append("\t".join((name, f"{name} (test)", "1", "surface" if data.ndim == 2 else "mass", shape,
                                       ",".join(f"1,{n}" for n in data.shape), "history-carriers.f32", str(offset))))
                offset += data.nbytes
        table.write_text("\n".join(rows) + "\n")
    out = tmp_path / "frames"
    made = door.frames_from_gm1(run, out)
    assert made["component"] == "morrison-mp10" and made["mp_physics"] == 10 and len(made["frames"]) == 3
    assert [f["times"] for f in made["frames"]] == ["0001-01-01_00:00:00", "0001-01-01_00:05:00",
                                                    "0001-01-01_00:10:00"]
    frames = door.open_frames([out])
    assert chain.producer_of(frames[0].attributes) == "GM1" and int(frames[0].attributes["MAP_PROJ"]) == 0
    choice, chosen = door.make_choice(frames, window="4,4,6,6", ratio=5, dt_s=None)
    payload = door.check(chosen, choice, price_it=False)
    assert payload["admitted"], [r for r in payload["checks"] if r["status"] != "pass"]
    receipt = chain.prepare_chain(chosen, door.request_of(choice), tmp_path / "prepared")
    assert receipt["admission"]["producer"] == "GM1"
    # A second write into the same folder is refused rather than mixed with the first.
    with pytest.raises(FileExistsError):
        door.frames_from_gm1(run, out)


def test_the_downscaling_guide_runs_in_order_and_passes_the_copy_lint(tmp_path, monkeypatch):
    """docs/WRF-DOWNSCALING.md, doc-tested as the guides are (tools/release/doc_commands.py, fake GPU).

    The guide's reader has a WRF run in ``~/wrf-run``: a WRF-like netCDF-4 history of three saved times, 5 min
    apart, is put there first.
    """

    import runpy
    import site

    from cm1core.gui.copy_lint import lint_markdown

    guide = ROOT / "docs" / "WRF-DOWNSCALING.md"
    assert lint_markdown(guide) == []
    work = tmp_path / "guide"
    history(work / "home" / "wrf-run", mp=6, use_theta_m=1, thm="dry", name="wrfout_d02_2026-05-01_18:00:00")
    # The reader's own Python keeps its packages when HOME moves (numpy and netCDF4 may be user-installed).
    monkeypatch.setenv("PYTHONUSERBASE", site.getuserbase())
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    doc = runpy.run_path(str(ROOT / "tools" / "release" / "doc_commands.py"))
    record = doc["run_guide"](guide, work, "fake", timeout_s=600)
    assert record["passed"], "\n".join(record["problems"])
    ran = [row for row in record["commands"] if "command" in row]
    assert [row["command"].split()[1:3] for row in ran] == [["chain", "inspect"], ["chain", "check"],
                                                            ["chain", "new"], ["run", "~/gm1-runs/downscaled"]]
    assert (work / "home" / "gm1-runs" / "downscaled" / "chain" / chain.RECEIPT).is_file()
    assert (work / "home" / "wrf-run" / "map.png").is_file()
    # The text blocks are what the commands print (the first check's lines, word for word).
    printed = gm1("check", "--history", os.fspath(work / "home" / "wrf-run"), "--window", "5,5,5,5", "--ratio", "3",
                  check=True).stdout.splitlines()
    shown = guide.read_text()
    for line in printed[:3] + printed[-5:-3]:
        assert line in shown, line
