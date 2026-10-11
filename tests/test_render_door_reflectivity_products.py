"""The two radar products a caller can actually select through the door.

The renderer has drawn reflectivity for a long time.  What decides whether
anyone can ask for it is the door's admitted ``--product`` set and the command
line it builds, so this walks the whole path: the CLI word, the door's
admission, and the exact argument vector handed to the native renderer.

``storm-fusion`` is checked here too.  It existed in the renderer and in its
own usage line, and the door refused it by omission, so no caller could reach
it; naming it in the door is the same kind of fix as the reflectivity level.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cm1core import doors, rust_tools
from cm1core.cli import main


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _renderer(tmp_path: Path, monkeypatch) -> dict[str, list[str]]:
    native = tmp_path / "cm1-render"
    native.write_bytes(b"verified-by-native-tools-manifest")
    observed: dict[str, list[str]] = {}
    monkeypatch.setattr(
        rust_tools, "verify_native_tools", lambda: {"cm1-render": native}
    )

    def fake_run(command, **kwargs):
        arguments = [str(value) for value in command]
        observed["command"] = arguments
        Path(arguments[arguments.index("--output") + 1]).write_bytes(PNG)
        return subprocess.CompletedProcess(arguments, 0, "rendered\n", "")

    monkeypatch.setattr(doors.subprocess, "run", fake_run)
    return observed


def write_envelope(path: Path, contract: str = "1" * 64, specs=None, magic=b"CM1FHS1\0") -> None:
    """A CM1FHS1 prefix and header; the payload is Rust's to verify, not the door's."""

    if specs is None:
        specs = {name: [40, 60, 60] for name in ("dbz", "updraft_ms", "vertical_vorticity_s")}
    header = json.dumps(
        {
            "contract": {"array_specs": [
                {"name": name, "shape": shape} for name, shape in specs.items()
            ]},
            "contract_sha256": contract,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(magic + len(header).to_bytes(8, "little") + header + b"payload-owned-by-rust")


def _render(tmp_path: Path, *extra: str) -> list[str]:
    history = tmp_path / "real.cm1fhs"
    if not history.exists():
        write_envelope(history)
    return [
        "render", str(history),
        "--output", str(tmp_path / "render.png"),
        "--contract-sha256", "1" * 64,
        "--nx", "60", "--ny", "60", "--nz", "40",
        "--valid-seconds", "0", "--json", *extra,
    ]


def test_the_composite_reflectivity_product_reaches_the_native_renderer(
    tmp_path, capsys, monkeypatch
):
    observed = _renderer(tmp_path, monkeypatch)
    code = main(_render(tmp_path, "--product", "reflectivity", "--field", "cref"))
    assert code == 0
    command = observed["command"]
    assert command[command.index("--product") + 1] == "reflectivity"
    assert command[command.index("--field") + 1] == "cref"
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["product"] == "reflectivity"
    assert receipt["status"] == "rendered"


def test_the_chosen_level_reflectivity_product_carries_its_level(
    tmp_path, capsys, monkeypatch
):
    observed = _renderer(tmp_path, monkeypatch)
    code = main(
        _render(tmp_path, "--product", "reflectivity-level", "--field", "dbz",
                "--level", "3")
    )
    assert code == 0
    command = observed["command"]
    assert command[command.index("--product") + 1] == "reflectivity-level"
    assert command[command.index("--level") + 1] == "3"
    assert json.loads(capsys.readouterr().out)["product"] == "reflectivity-level"


def test_a_level_outside_the_declared_volume_is_refused_before_the_renderer_runs(
    tmp_path, capsys, monkeypatch
):
    observed = _renderer(tmp_path, monkeypatch)
    code = main(
        _render(tmp_path, "--product", "reflectivity-level", "--field", "dbz",
                "--level", "40")
    )
    assert code == 2
    assert "command" not in observed
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-REQUEST"
    assert "another column's words" in refusal["reason"]
    assert refusal["option"] == "level"


def test_storm_fusion_is_selectable_now_that_the_door_names_it(
    tmp_path, capsys, monkeypatch
):
    observed = _renderer(tmp_path, monkeypatch)
    code = main(
        _render(
            tmp_path, "--product", "storm-fusion",
            "--field", "dbz", "--field", "updraft_ms",
            "--field", "vertical_vorticity_s",
        )
    )
    assert code == 0
    command = observed["command"]
    assert command[command.index("--product") + 1] == "storm-fusion"
    assert command.count("--field") == 3
    assert json.loads(capsys.readouterr().out)["product"] == "storm-fusion"


def test_an_unknown_render_product_still_refuses_by_value(tmp_path, capsys, monkeypatch):
    _renderer(tmp_path, monkeypatch)
    code = main(_render(tmp_path, "--product", "radar", "--field", "dbz"))
    assert code == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-REQUEST"
    assert "reflectivity-level" in refusal["reason"]


def test_the_door_and_the_renderer_admit_the_same_product_names():
    """A product the door names and the renderer does not is unreachable."""

    door = Path("src/cm1core/doors.py").read_text(encoding="utf-8")
    renderer = Path("rust/cm1-render/src/main.rs").read_text(encoding="utf-8")
    start = door.index("    products = {")
    block = door[start:door.index("}", start)]
    names = {
        line.strip().strip(',').strip('"')
        for line in block.splitlines()
        if line.strip().startswith('"')
    }
    assert "reflectivity-level" in names
    assert "storm-fusion" in names
    accepted = renderer[renderer.index('"reflectivity" | "reflectivity-level"'):]
    accepted = accepted[: accepted.index("=>")]
    for name in sorted(names):
        assert f'"{name}"' in accepted, f"the renderer refuses the door's {name!r}"


def test_the_grid_and_contract_default_from_the_envelope_header(tmp_path, capsys, monkeypatch):
    observed = _renderer(tmp_path, monkeypatch)
    history = tmp_path / "frame.cm1fhs"
    write_envelope(history, contract="a" * 64, specs={"w3d": [41, 24, 32], "cape": [24, 32]})
    code = main([
        "render", str(history), "--output", str(tmp_path / "w.png"),
        "--product", "updraft", "--field", "w3d", "--json",
    ])
    assert code == 0
    command = observed["command"]
    assert command[command.index("--contract-sha256") + 1] == "a" * 64
    assert [command[command.index(f"--{axis}") + 1] for axis in ("nx", "ny", "nz")] == ["32", "24", "41"]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["contract_sha256_source"] == "frame-header-self-described"
    assert receipt["grid"] == {"nx": 32, "ny": 24, "nz": 41}
    # A surface plane is one value per column: nz defaults to 1.
    code = main([
        "render", str(history), "--output", str(tmp_path / "cape.png"),
        "--product", "cape", "--field", "cape", "--json",
    ])
    assert code == 0
    command = observed["command"]
    assert command[command.index("--nz") + 1] == "1"
    capsys.readouterr()


def test_the_publication_receipt_anchors_the_contract_when_present(tmp_path, capsys, monkeypatch):
    observed = _renderer(tmp_path, monkeypatch)
    history = tmp_path / "history.cm1fhs"
    write_envelope(history, contract="b" * 64)
    (tmp_path / "history-publication.json").write_text(json.dumps({
        "forecast_envelope": {
            "path": "/elsewhere/history.cm1fhs",
            "rust_output": "abi=x\ncontract_sha256=" + "b" * 64 + "\n{}",
        },
    }), encoding="utf-8")
    code = main([
        "render", str(history), "--output", str(tmp_path / "r.png"),
        "--product", "reflectivity", "--field", "dbz", "--json",
    ])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["contract_sha256_source"] == "publication-receipt"
    # A receipt that names another contract is a different frame: refused.
    write_envelope(history.with_name("x.cm1fhs"), contract="c" * 64)
    history.unlink()
    history.with_name("x.cm1fhs").rename(history)
    code = main([
        "render", str(history), "--output", str(tmp_path / "s.png"),
        "--product", "reflectivity", "--field", "dbz", "--json",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-RENDER-CONTRACT-MISMATCH"
    assert observed["command"][observed["command"].index("--output") + 1] != str(tmp_path / "s.png")


def test_explicit_values_are_checked_against_the_header(tmp_path, capsys, monkeypatch):
    observed = _renderer(tmp_path, monkeypatch)
    history = tmp_path / "frame.cm1fhs"
    write_envelope(history, contract="d" * 64)
    code = main([
        "render", str(history), "--output", str(tmp_path / "a.png"),
        "--product", "reflectivity", "--field", "dbz", "--nx", "61", "--json",
    ])
    assert code == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-REQUEST" and refusal["option"] == "nx"
    code = main([
        "render", str(history), "--output", str(tmp_path / "b.png"),
        "--product", "reflectivity", "--field", "dbz",
        "--contract-sha256", "e" * 64, "--json",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-RENDER-CONTRACT-MISMATCH"
    assert "command" not in observed


def test_a_file_without_an_envelope_prefix_is_refused_by_name(tmp_path, capsys, monkeypatch):
    _renderer(tmp_path, monkeypatch)
    history = tmp_path / "frame.cm1fhs"
    history.write_bytes(b"not an envelope")
    code = main([
        "render", str(history), "--output", str(tmp_path / "a.png"),
        "--product", "reflectivity", "--field", "dbz", "--json",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-HISTORY-ENVELOPE"
