"""A GM1 history frame, exported as CM1 ``output_format=1`` and read back.

The reader is the differential harness's own GrADS reader
(``cm1core.oracle.grads``), which was written to read NATIVE CM1 output.  If it
reads a GM1 export and returns the same words the arena holds, then so does
every other tool that reads CM1's format.
"""

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cm1core.grads_export import GRADS_EXPORT_ABI
from cm1core.oracle.grads import read_ctl

_ROOT = Path(__file__).resolve().parents[1]


def _exporter() -> Path:
    """The exporter: a crate build, the wheel build's target (build/rust-native-target), or an installed wheel's."""

    for candidate in (
        _ROOT / "rust" / "cm1-state" / "target" / "release" / "gm1-history-grads-export",
        _ROOT / "build" / "rust-native-target" / "release" / "gm1-history-grads-export",
    ):
        if candidate.is_file():
            return candidate
    try:
        from cm1core import rust_tools

        installed = rust_tools.verify_native_tools().get("gm1-history-grads-export")
    except Exception:  # noqa: BLE001 - a checkout has no sealed tools
        installed = None
    return Path(installed) if installed else _ROOT / "rust" / "cm1-state" / "target" / "release" / "gm1-history-grads-export"


_BINARY = _exporter()

pytestmark = pytest.mark.skipif(
    not _BINARY.is_file(),
    reason="build with: cd rust/cm1-state && cargo build --release --bin gm1-history-grads-export",
)

NX, NY, NZ = 7, 5, 6
DX = DY = 1000.0
DZ = 500.0


def _write_frame(directory: Path) -> dict[str, np.ndarray]:
    """One frame in the publisher's own arena + tab-separated field table."""

    directory.mkdir(parents=True, exist_ok=True)
    arena = directory / "history-carriers.f32"
    rng = np.random.default_rng(31337)
    carriers = [
        ("closure.tkea", "subgrid tke", "m2 s-2", "w", (NZ + 1, NY, NX)),
        ("pp3d", "perturbation exner", "nondimensional", "mass", (NZ, NY, NX)),
        ("prs", "pressure", "Pa", "mass", (NZ, NY, NX)),
        ("q3d", "six species", "kg kg-1", "mass-species", (6, NZ, NY, NX)),
        ("rho", "density", "kg m-3", "mass", (NZ, NY, NX)),
        ("terrain_msl", "terrain", "m", "surface", (NY, NX)),
        ("th3d", "theta perturbation", "K", "mass", (NZ, NY, NX)),
        ("u3d", "u", "m s-1", "u", (NZ, NY, NX + 1)),
        ("v3d", "v", "m s-1", "v", (NZ, NY + 1, NX)),
        ("w3d", "w", "m s-1", "w", (NZ + 1, NY, NX)),
        ("uh", "updraft helicity", "m2 s-2", "surface", (NY, NX)),
        ("sws", "max surface wind", "m s-1", "surface", (NY, NX)),
        ("dbz", "reflectivity", "dBZ", "mass", (NZ, NY, NX)),
        ("th0", "base theta", "K", "profile", (NZ,)),
    ]
    values: dict[str, np.ndarray] = {}
    rows = []
    offset = 0
    with arena.open("wb") as handle:
        for name, meaning, units, stagger, shape in carriers:
            array = rng.uniform(-40.0, 60.0, size=shape).astype(np.float32)
            values[name] = array
            handle.write(array.tobytes())
            bounds = ",".join(
                str(value) for extent in shape for value in (1, extent)
            )
            rows.append("\t".join((
                name, meaning, units, stagger,
                ",".join(str(extent) for extent in shape),
                bounds, os.fspath(arena.resolve()), str(offset),
            )))
            offset += array.nbytes
    (directory / "history-field-table.tsv").write_text("\n".join(rows) + "\n")
    return values


def _export(frame: Path, out: Path) -> dict:
    import json

    completed = subprocess.run(
        [os.fspath(_BINARY), "--frame", os.fspath(frame), "--out", os.fspath(out),
         "--dx", str(DX), "--dy", str(DY), "--dz", str(DZ)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_exporter_reports_the_abi_the_door_admits():
    completed = subprocess.run([os.fspath(_BINARY), "--abi"], capture_output=True, text=True)
    assert completed.stdout.strip() == GRADS_EXPORT_ABI


def test_a_frame_exports_to_cm1s_four_grid_files_and_reads_back_word_for_word(tmp_path):
    values = _write_frame(tmp_path / "frame")
    receipt = _export(tmp_path / "frame", tmp_path / "grads")
    assert receipt["nx"] == NX and receipt["ny"] == NY and receipt["nz"] == NZ
    grids = {entry["grid"] for entry in receipt["files"]}
    assert grids == {"s", "u", "v", "w"}

    scalar = read_ctl(tmp_path / "grads" / "cm1out_s.ctl")
    names = [variable.name for variable in scalar.variables]
    # CM1's own names: th3d is thpert, pp3d is pipert, the q3d block is six
    # named species, and terrain_msl is zs.
    for expected in ("thpert", "pipert", "prs", "rho", "qv", "qg", "zs", "uh",
                     "sws", "dbz", "th0"):
        assert expected in names, names
    assert scalar.nx == NX and scalar.ny == NY and scalar.nz == NZ

    def read(dataset, name):
        words, _ = dataset.read(name, 1)
        return np.asarray(words, dtype=np.float32)

    assert np.array_equal(read(scalar, "prs").reshape(NZ, NY, NX), values["prs"])
    assert np.array_equal(read(scalar, "thpert").reshape(NZ, NY, NX), values["th3d"])
    assert np.array_equal(read(scalar, "qv").reshape(NZ, NY, NX), values["q3d"][0])
    assert np.array_equal(read(scalar, "qg").reshape(NZ, NY, NX), values["q3d"][5])
    assert np.array_equal(read(scalar, "dbz").reshape(NZ, NY, NX), values["dbz"])
    assert np.array_equal(read(scalar, "uh").reshape(NY, NX), values["uh"])

    wind = read_ctl(tmp_path / "grads" / "cm1out_u.ctl")
    assert wind.nx == NX + 1
    assert np.array_equal(
        read(wind, "u").reshape(NZ, NY, NX + 1), values["u3d"]
    )
    face = read_ctl(tmp_path / "grads" / "cm1out_w.ctl")
    assert face.nz == NZ + 1
    assert np.array_equal(read(face, "w").reshape(NZ + 1, NY, NX), values["w3d"])
    assert np.array_equal(read(face, "tke").reshape(NZ + 1, NY, NX), values["closure.tkea"])


def test_a_base_state_profile_is_broadcast_to_cm1s_three_dimensional_record(tmp_path):
    values = _write_frame(tmp_path / "frame")
    _export(tmp_path / "frame", tmp_path / "grads")
    scalar = read_ctl(tmp_path / "grads" / "cm1out_s.ctl")
    words, _ = scalar.read("th0", 1)
    th0 = np.asarray(words, dtype=np.float32).reshape(NZ, NY, NX)
    for level in range(NZ):
        assert np.all(th0[level] == values["th0"][level])


def test_a_carrier_on_no_admitted_grid_is_refused_rather_than_misplaced(tmp_path):
    frame = tmp_path / "frame"
    _write_frame(frame)
    table = frame / "history-field-table.tsv"
    rows = table.read_text().splitlines()
    columns = rows[0].split("\t")
    columns[0] = "impostor"
    columns[4] = f"{NZ},{NY + 3},{NX + 3}"
    rows.append("\t".join(columns))
    table.write_text("\n".join(rows) + "\n")
    completed = subprocess.run(
        [os.fspath(_BINARY), "--frame", os.fspath(frame), "--out",
         os.fspath(tmp_path / "grads"), "--dx", str(DX), "--dy", str(DY),
         "--dz", str(DZ)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "impostor" in completed.stderr
    assert "C-grid families" in completed.stderr


def test_a_guessed_grid_spacing_is_refused_rather_than_written(tmp_path):
    _write_frame(tmp_path / "frame")
    completed = subprocess.run(
        [os.fspath(_BINARY), "--frame", os.fspath(tmp_path / "frame"),
         "--out", os.fspath(tmp_path / "grads")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "--dx, --dy and --dz are required" in completed.stderr


# ------------------------------- frames as --integrate publishes them


def _integrated_frames(root: Path) -> Path:
    """Frames published by the shipped integrate(), staged and renamed."""

    from test_cartesian_restart import CpuCartesianRuntime, _config

    runtime = CpuCartesianRuntime(_config(), seed=5)
    runtime.integrate(
        duration_s=2.0, history_interval_s=1.0, history_root=root / "frames",
        history_publisher=None,
    )
    assert not (root / "frames" / ".incomplete").exists()
    return root / "frames"


def _export_frame(frame: Path, out: Path, spacing: float = 500.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [os.fspath(_BINARY), "--frame", os.fspath(frame), "--out", os.fspath(out),
         "--dx", str(spacing), "--dy", str(spacing), "--dz", str(spacing)],
        capture_output=True, text=True, check=False,
    )


def _carriers(frame: Path) -> dict[str, tuple[tuple[int, ...], int]]:
    rows = {}
    for line in (frame / "history-field-table.tsv").read_text().splitlines():
        columns = line.split("\t")
        rows[columns[0]] = (tuple(int(v) for v in columns[4].split(",")), int(columns[7]))
    return rows


def test_a_frame_published_by_integrate_exports_after_its_staging_rename(tmp_path):
    """The regression: the table named the arena at its .incomplete/ path."""

    import json

    frames = _integrated_frames(tmp_path)
    frame = frames / "frame-000002"
    assert {line.split("\t")[6] for line in
            (frame / "history-field-table.tsv").read_text().splitlines()} == {
        "history-carriers.f32"
    }
    completed = _export_frame(frame, tmp_path / "grads")
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["arena"] == os.fspath(frame / "history-carriers.f32")
    assert receipt["not_exported"] == []
    arena = np.fromfile(frame / "history-carriers.f32", dtype="<f4")
    shape, offset = _carriers(frame)["th3d"]
    expected = arena[offset // 4: offset // 4 + int(np.prod(shape))].reshape(shape)
    scalar = read_ctl(tmp_path / "grads" / "cm1out_s.ctl")
    words, _ = scalar.read("thpert", 1)
    assert np.array_equal(np.asarray(words, dtype=np.float32).reshape(shape), expected)
    # A copy of the frame somewhere else reads the same, from its own arena.
    moved = tmp_path / "elsewhere" / "frame"
    shutil.copytree(frame, moved)
    completed = _export_frame(moved, tmp_path / "grads-moved")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["arena"] == os.fspath(moved / "history-carriers.f32")


def test_a_frame_published_before_the_fix_reads_its_own_arena(tmp_path):
    """Older frames name their vanished staging path; the frame's file is that arena."""

    import json

    frame = _integrated_frames(tmp_path) / "frame-000001"
    table = frame / "history-field-table.tsv"
    stale = os.fspath(tmp_path / "frames" / ".incomplete" / "frame-000001" / "history-carriers.f32")
    rows = [line.split("\t") for line in table.read_text().splitlines()]
    for row in rows:
        row[6] = stale
    table.write_text("\n".join("\t".join(row) for row in rows) + "\n")
    assert not Path(stale).exists()
    completed = _export_frame(frame, tmp_path / "grads")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["arena"] == os.fspath(frame / "history-carriers.f32")


def test_the_ground_canvas_swaths_are_named_not_exported(tmp_path):
    import json

    frame = tmp_path / "frame"
    _write_frame(frame)
    table = frame / "history-field-table.tsv"
    rows = table.read_text().splitlines()
    canvas = (NY + 4, NX + 6)
    with (frame / "history-carriers.f32").open("ab") as handle:
        offset = handle.tell()
        handle.write(np.zeros(canvas, dtype="<f4").tobytes())
    rows.append("\t".join((
        "sws2", "run-long max ground-relative wind speed, ground-fixed canvas", "m s-1",
        "ground-canvas", f"{canvas[0]},{canvas[1]}", f"1,{canvas[0]},1,{canvas[1]}",
        "history-carriers.f32", str(offset),
    )))
    table.write_text("\n".join(rows) + "\n")
    completed = subprocess.run(
        [os.fspath(_BINARY), "--frame", os.fspath(frame), "--out", os.fspath(tmp_path / "grads"),
         "--dx", str(DX), "--dy", str(DY), "--dz", str(DZ)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert [row["name"] for row in receipt["not_exported"]] == ["sws2"]
    assert receipt["not_exported"][0]["shape"] == list(canvas)
    assert "canvas origin" in receipt["not_exported"][0]["reason"]
    names = [variable.name for variable in read_ctl(tmp_path / "grads" / "cm1out_s.ctl").variables]
    assert "sws" in names and "sws2" not in names
