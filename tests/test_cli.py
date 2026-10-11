from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cm1core.doors as doors
from cm1core.cli import main
import cm1core.rust_tools as rust_tools


ROOT = Path(__file__).resolve().parents[1]


def test_check_accepts_official_product(capsys) -> None:
    assert (
        main(["check", str(ROOT / "examples" / "official-supercell-product.toml")]) == 0
    )
    assert '"executable": false' in capsys.readouterr().out


def test_check_accepts_fixed_real_100m_product(capsys) -> None:
    assert (
        main(["check", str(ROOT / "examples" / "real-atmosphere-100m-product.toml")])
        == 0
    )
    output = capsys.readouterr().out
    assert '"lane": "real-product"' in output
    assert '"dx_m": 100.0' in output


def test_check_accepts_the_cartesian_example(capsys) -> None:
    assert main(["check", str(ROOT / "examples" / "cartesian-uniform-grid.toml")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["run"]["lane"] == "cartesian"
    assert payload["config"]["runtime"]["acoustic_substeps"] == 6
    identity = payload["config"]["profile_identity"]
    assert identity["lane"] == "cartesian"
    assert identity["profile"] == "gm1-cartesian-configurable-grid/v1"
    assert payload["executable"] is False


def test_cartesian_preflight_drops_the_repaired_september_blockers(capsys) -> None:
    status = main(["preflight", str(ROOT / "examples" / "cartesian-uniform-grid.toml")])
    payload = json.loads(capsys.readouterr().out)
    assert status == (0 if payload["runnable"] else 1)
    assert payload["lane"] == "cartesian"
    codes = {blocker["code"] for blocker in payload["blockers"]}
    assert "CM1-CUDA-TRAJECTORY-INCOMPLETE" not in codes
    assert "CM1-INSTRUMENTS-MISSING" not in codes
    assert codes <= {"CM1-ARWEN-ESTATE-DRIFT"}


def test_cartesian_door_refuses_another_lane_before_creating_output(
    tmp_path: Path, capsys
) -> None:
    receipt = tmp_path / "must-not-exist.json"
    status = main(
        [
            "cartesian",
            str(ROOT / "examples" / "official-supercell-product.toml"),
            "--output",
            str(receipt),
            "--json",
        ]
    )
    assert status == 2
    assert not receipt.exists()
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-CARTESIAN-LANE"


def test_admitted_forecast_lane_refuses_instead_of_asserting(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """A retired blocker list must not reach a bare AssertionError traceback."""

    monkeypatch.setattr(doors, "require_runnable", lambda config, **kwargs: None)
    output = tmp_path / "must-not-exist"
    status = main(
        [
            "forecast",
            str(ROOT / "examples" / "official-supercell-product.toml"),
            "--output",
            str(output),
            "--json",
        ]
    )
    assert status == 2
    assert not output.exists()
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "CM1-RUN-BACKEND-MISSING"
    assert payload["option"] == "run.lane"
    assert "cartesian" in payload["remedy"]


def test_preflight_is_nonzero_and_machine_readable(capsys) -> None:
    assert (
        main(["preflight", str(ROOT / "examples" / "official-supercell-product.toml")])
        == 1
    )
    output = capsys.readouterr().out
    assert '"runnable": false' in output
    assert "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING" in output
    assert "CM1-ARWEN-SEAM-MISSING" not in output


def test_forecast_refuses_before_creating_output(tmp_path: Path, capsys) -> None:
    output = tmp_path / "must-not-exist"
    code = main(
        [
            "forecast",
            str(ROOT / "examples" / "official-supercell-product.toml"),
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert not output.exists()
    assert (
        "CM1-CUDA-TRAJECTORY-INCOMPLETE" in capsys.readouterr().err
        or not output.exists()
    )


def test_json_refusal_safely_represents_toml_datetime(tmp_path: Path, capsys) -> None:
    source = (ROOT / "examples" / "official-supercell-product.toml").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "[run]\n", "[run]\nunknown_clock = 2026-08-28T12:00:00Z\n", 1
    )
    config = tmp_path / "unknown-datetime.toml"
    config.write_text(source, encoding="utf-8")
    code = main(["check", str(config), "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["option"] == "run.unknown_clock"
    assert payload["value"] == {
        "type": "datetime",
        "value": "2026-08-28T12:00:00+00:00",
    }


def test_missing_paths_keep_json_envelope(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing"
    assert main(["check", str(missing), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-CFG-PATH"
    assert (
        main(
            [
                "restart",
                str(missing),
                "--output",
                str(tmp_path / "out"),
                "--contract-sha256",
                "1" * 64,
                "--artifact-sha256",
                "2" * 64,
                "--operator-manifest-sha256",
                "3" * 64,
                "--json",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-RESTART-PATH"
    assert (
        main(["render", str(missing), "--output", str(tmp_path / "out"), "--json"]) == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-HISTORY-PATH"


def test_render_door_orchestrates_verified_native_renderer(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    history = tmp_path / "real.cm1fhs"
    from test_render_door_reflectivity_products import write_envelope

    write_envelope(
        history,
        specs={name: [300, 300] for name in ("reflectivity_dbz", "updraft_ms", "vertical_vorticity_s")},
    )
    native = tmp_path / "cm1-render"
    native.write_bytes(b"verified-by-native-tools-manifest")
    output = tmp_path / "render.png"
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    observed: dict[str, list[str]] = {}

    monkeypatch.setattr(
        rust_tools,
        "verify_native_tools",
        lambda: {"cm1-render": native},
    )

    def fake_run(command, **kwargs):
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        arguments = [str(value) for value in command]
        observed["command"] = arguments
        temporary = Path(arguments[arguments.index("--output") + 1])
        assert temporary.parent == tmp_path
        assert temporary != output
        temporary.write_bytes(png)
        return subprocess.CompletedProcess(arguments, 0, "rendered\n", "")

    monkeypatch.setattr(doors.subprocess, "run", fake_run)
    code = main(
        [
            "render",
            str(history),
            "--output",
            str(output),
            "--contract-sha256",
            "1" * 64,
            "--product",
            "vortex-composite",
            "--field",
            "reflectivity_dbz",
            "--field",
            "updraft_ms",
            "--field",
            "vertical_vorticity_s",
            "--nx",
            "300",
            "--ny",
            "300",
            "--nz",
            "1",
            "--valid-seconds",
            "0",
            "--json",
        ]
    )
    assert code == 0
    assert output.read_bytes() == png
    assert observed["command"][0] == str(native)
    assert observed["command"][observed["command"].index("--format") + 1] == "cm1fhs"
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"] == "gpuwm-cm1.render/v1"
    assert receipt["model"] == "GM1"
    assert receipt["product"] == "vortex-composite"
    assert receipt["math_authority"] == "CM1 r21.1-derived mathematics"
    assert receipt["state_classification"] == "initialization-not-tornado-not-forecast"
    assert receipt["status"] == "rendered"
    assert receipt["sha256"]


def test_restart_door_orchestrates_verified_rust_restore(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    checkpoint = tmp_path / "safe.cm1fcp"
    checkpoint.write_bytes(b"strict-envelope-owned-by-rust")
    native = tmp_path / "gm1-forecast-checkpoint"
    native.write_bytes(b"verified-by-native-tools-manifest")
    output = tmp_path / "restore"
    contract = "1" * 64
    artifact = "2" * 64
    operator = "3" * 64
    observed: dict[str, list[str]] = {}

    monkeypatch.setattr(
        rust_tools,
        "verify_native_tools",
        lambda: {"gm1-forecast-checkpoint": native},
    )

    def fake_run(command, **kwargs):
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        arguments = [str(value) for value in command]
        observed["command"] = arguments
        destination = Path(arguments[3])
        destination.mkdir()
        (destination / "manifest.json").write_text("{}", encoding="utf-8")
        (destination / "receipt.json").write_text("{}", encoding="utf-8")
        (destination / "gm1-carriers.f32le").write_bytes(b"raw")
        receipt = {
            "carrier_abi": "gm1.forecast-checkpoint-carriers/v1",
            "carrier_count": 37,
            "checkpoint_artifact_sha256": artifact,
            "checkpoint_contract_sha256": contract,
            "model": "GM1",
            "schema": "gm1.forecast-restore-receipt/v1",
        }
        return subprocess.CompletedProcess(arguments, 0, json.dumps(receipt), "")

    monkeypatch.setattr(doors.subprocess, "run", fake_run)
    code = main(
        [
            "restart",
            str(checkpoint),
            "--output",
            str(output),
            "--contract-sha256",
            contract,
            "--artifact-sha256",
            artifact,
            "--operator-manifest-sha256",
            operator,
            "--json",
        ]
    )
    assert code == 0
    assert observed["command"] == [
        str(native),
        "restore",
        str(checkpoint),
        str(output),
        contract,
        artifact,
        operator,
    ]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"] == "gpuwm-cm1.restart/v1"
    assert receipt["model"] == "GM1"
    assert receipt["status"] == "restored-before-next-large-step"
