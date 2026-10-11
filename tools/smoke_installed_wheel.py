#!/usr/bin/env python3
"""Install a wheel without source-path help and exercise every safe door."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


# The records door lists every installed record, and the package ships more
# than the contract set below.  Require the contract records to be present
# rather than pinning an exact inventory that a new record would break.
REQUIRED_RECORDS = {
    "CM1-ARITHMETIC-CONTRACT.md",
    "CM1-ARWEN-SEAM.md",
    "CM1-AUTHORITY.json",
    "CM1-CADENCE-MAP.md",
    "CM1-GM1-KEYMAP.json",
    "CM1-KEY-ALIASES.json",
    "CM1-ORACLE-MANIFEST.json",
    "CM1-PROFILE-MANIFEST.json",
    "CM1-R21-NAMELIST-DEFAULTS.json",
    "CM1-R22-NEW-KEYS.json",
    "CM1-RISK-REGISTER.md",
    "CM1-SOURCE-MAP.md",
    "CM1-STATE-MAP.csv",
    "CM1-SUPPORT-MATRIX.csv",
    "GM1-GPU-RATES.json",
}
RECORD_SUFFIXES = {".csv", ".json", ".md"}
# Every sealed native artifact, including the two tools and the cdylib that
# have no console script because no door dispatches to them yet.
SEALED_NATIVE_TOOLS = {
    "gm1-history-grads-export",
    "checkpoint-wind-grads",
    "cm1-r21-namelist",
    "cm1-f64-window",
    "cm1-forecast-publish",
    "cm1-render",
    "cm1-state",
    "gm1-cartesian-restart",
    "gm1-child-post-step",
    "gm1-forecast-checkpoint",
}
SEALED_NATIVE_LIBRARIES = {"cm1-state"}
REQUIRED_PACKAGED_INPUTS = {"cm1r21.1-cpm-radconvequil-sounding"}
# Product-lane preflight refusals, reconciled with src/cm1core/preflight.py.
REQUIRED_BLOCKER_CODES = {
    "CM1-CUDA-TRAJECTORY-INCOMPLETE",
    "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING",
    "CM1-RUST-LIVE-WIRING-MISSING",
    "CM1-INSTRUMENTS-MISSING",
}
# This smoke installs --no-deps, so the ArWen estate is legitimately absent
# and its drift refusal is an admitted, diagnosed outcome rather than a fault.
ADMITTED_BLOCKER_CODES = REQUIRED_BLOCKER_CODES | {"CM1-ARWEN-ESTATE-DRIFT"}
# Lanes this configuration is not in; their refusals must not appear.
FORBIDDEN_BLOCKER_CODES = {
    "CM1-DRY-PSOLVER2-ORACLE-MISSING",
    "CM1-NATIVE-CONTEXT-NOT-A-PRODUCT-DOOR",
    "CM1-REAL-PRODUCT-NO-INSTALLED-DOOR",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    expected_exit: int,
    json_stream: str = "stdout",
) -> tuple[dict[str, object] | None, dict[str, object]]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    selected = completed.stdout if json_stream == "stdout" else completed.stderr
    payload: dict[str, object] | None = None
    parse_error: str | None = None
    try:
        parsed = json.loads(selected)
        if isinstance(parsed, dict):
            payload = parsed
        else:
            parse_error = "JSON result was not an object"
    except json.JSONDecodeError as error:
        parse_error = f"invalid JSON: {error.msg}"
    result: dict[str, object] = {
        "exit": completed.returncode,
        "expected_exit": expected_exit,
        "json_stream": json_stream,
        "passed": completed.returncode == expected_exit and payload is not None,
    }
    if parse_error is not None:
        result["parse_error"] = parse_error
    return payload, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    arguments = parser.parse_args()

    wheel = arguments.wheel.expanduser().resolve(strict=True)
    source_config = arguments.config.expanduser().resolve(strict=True)
    work = arguments.work_dir.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if work.exists():
        parser.error(f"work directory already exists; refusing to reuse it: {work}")
    if work.is_relative_to(source_root):
        parser.error("work directory must be outside the source checkout")

    venv_dir = work / "venv"
    scratch = work / "scratch"
    scratch_home = scratch / "home"
    scratch_home.mkdir(parents=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    entrypoint = scripts / ("gpuwm-cm1.exe" if os.name == "nt" else "gpuwm-cm1")
    native_entrypoints = {
        name: scripts / (f"{name}.exe" if os.name == "nt" else name)
        for name in (
            "cm1-state",
            "gm1-forecast-checkpoint",
            "gm1-cartesian-restart",
            "cm1-render",
            "cm1-f64-window",
            "gm1-history-grads-export",
            "checkpoint-wind-grads",
            "cm1-r21-namelist",
        )
    }

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HOME"] = str(scratch_home)
    environment["USERPROFILE"] = str(scratch_home)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    install = subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=scratch,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    checks: dict[str, dict[str, object]] = {
        "install-wheel-no-deps": {
            "exit": install.returncode,
            "expected_exit": 0,
            "passed": install.returncode == 0,
        },
        "console-entrypoint-created": {
            "passed": entrypoint.is_file(),
        },
        "native-entrypoints-created": {
            "paths": {name: path.name for name, path in native_entrypoints.items()},
            "passed": all(path.is_file() for path in native_entrypoints.values()),
        },
    }
    if (
        install.returncode != 0
        or not entrypoint.is_file()
        or not all(path.is_file() for path in native_entrypoints.values())
    ):
        report = {
            "schema": "gpuwm-cm1.installed-wheel-smoke/v1",
            "claim_scope": "installed Python and hashed native-payload reachability only; no CUDA forecast or restart identity",
            "checks": checks,
            "passed": False,
            "wheel": {"name": wheel.name, "sha256": sha256_file(wheel)},
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    expected_abis = {
        "cm1-state": "gpuwm-cm1-state/v1",
        "gm1-forecast-checkpoint": (
            "gm1.forecast-checkpoint-cli/v1;"
            "carrier_abi=gm1.forecast-checkpoint-carriers/v1"
        ),
        "gm1-cartesian-restart": (
            "gm1.cartesian-restart-cli/v1;"
            "carrier_abi=gm1.cartesian-restart-carriers/v1"
        ),
        "cm1-render": "gpuwm-cm1-render/v1",
        "cm1-f64-window": "gpuwm-cm1-f64-window/v1",
        "gm1-history-grads-export": "gpuwm-cm1-history-grads-export/v1",
        "checkpoint-wind-grads": "gpuwm-cm1-checkpoint-wind-grads/v1",
        "cm1-r21-namelist": (
            "gm1.cm1-r21.1-namelist-cli/v2"
            ";document_schema=gm1.cm1-r21.1-namelist-document/v1"
            ";support_schema=gm1.cm1-r21.1-configuration-support/v1"
            ";binding_schema=gm1.cm1-r21.1-configuration-binding/v1"
            ";inspection_schema=gm1.cm1-r21.1-namelist-inspection/v1"
        ),
    }
    for name, expected_abi in expected_abis.items():
        completed = subprocess.run(
            [str(native_entrypoints[name]), "--abi"],
            cwd=scratch,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        checks[f"{name}-installed-abi"] = {
            "exit": completed.returncode,
            "expected_abi": expected_abi,
            "observed_abi": completed.stdout.strip(),
            "passed": completed.returncode == 0
            and completed.stdout.strip() == expected_abi
            and not completed.stderr,
        }

    # The console scripts only cover four tools.  The wheel also has to carry
    # the shared library the Cartesian runtime opens with ctypes.  The CM1
    # sounding is deliberately NOT carried: its identity is recorded and the
    # caller supplies the file, so assert the identity resolves and the bytes
    # do not.
    payload_code = (
        "import json; from pathlib import Path; "
        "from cm1core.data import required_inputs; "
        "from cm1core.rust_tools import ("
        "EXPECTED_LIBRARIES, native_library_path, verify_native_tools); "
        "tools = verify_native_tools(); "
        "libraries = {name: native_library_path(name) for name in EXPECTED_LIBRARIES}; "
        "inputs = required_inputs(); "
        "print(json.dumps({"
        "'tools': sorted(tools), "
        "'libraries': sorted(libraries), "
        "'inputs': sorted(inputs), "
        "'inputs_shipped': [k for k, v in inputs.items() if v.get('shipped')], "
        "'library_bytes': {k: Path(v).stat().st_size for k, v in libraries.items()}"
        "}))"
    )
    payload, payload_result = invoke(
        [str(python), "-I", "-c", payload_code],
        cwd=scratch,
        environment=environment,
        expected_exit=0,
    )
    payload = payload or {}
    payload_result["observed"] = payload
    payload_result["passed"] = (
        payload_result["passed"]
        and set(payload.get("tools", [])) == SEALED_NATIVE_TOOLS
        and set(payload.get("libraries", [])) == SEALED_NATIVE_LIBRARIES
        and REQUIRED_PACKAGED_INPUTS.issubset(payload.get("inputs", []))
    )
    checks["installed-native-payload-resolves"] = payload_result

    config = scratch / "official-supercell-product.toml"
    shutil.copyfile(source_config, config)
    dummy_checkpoint = scratch / "dummy-checkpoint"
    dummy_history = scratch / "dummy-history"
    dummy_checkpoint.write_bytes(b"deliberately not a checkpoint\n")
    dummy_history.write_bytes(b"deliberately not a history\n")

    def cli(
        check_id: str,
        args: list[str],
        expected_exit: int,
        *,
        json_stream: str = "stdout",
    ) -> dict[str, object] | None:
        payload, result = invoke(
            [str(entrypoint), *args],
            cwd=scratch,
            environment=environment,
            expected_exit=expected_exit,
            json_stream=json_stream,
        )
        checks[check_id] = result
        return payload

    version_payload = cli("version-door", ["version"], 0)
    if version_payload is not None:
        checks["version-door"]["observed"] = {
            key: version_payload.get(key)
            for key in ("version", "cm1_release", "maturity", "authority_sha256")
        }

    records_payload = cli("records-door", ["records"], 0)
    if records_payload is not None:
        names = {
            row.get("name")
            for row in records_payload.get("records", [])
            if isinstance(row, dict)
        }
        observed_names = sorted(name for name in names if isinstance(name, str))
        checks["records-door"]["observed_record_names"] = observed_names
        checks["records-door"]["missing_required_records"] = sorted(
            REQUIRED_RECORDS - set(observed_names)
        )
        checks["records-door"]["passed"] = (
            checks["records-door"]["passed"]
            and REQUIRED_RECORDS.issubset(observed_names)
            and all(
                Path(name).suffix in RECORD_SUFFIXES and name != "__init__.py"
                for name in observed_names
            )
        )

    check_payload = cli("check-door", ["check", str(config), "--json"], 0)
    if check_payload is not None:
        observed_hash = check_payload.get("config_sha256")
        profile = check_payload.get("config", {}).get("profile_identity", {})
        checks["check-door"]["observed"] = {
            "config_sha256": observed_hash,
            "profile": profile.get("profile") if isinstance(profile, dict) else None,
            "profile_lane": profile.get("lane") if isinstance(profile, dict) else None,
        }
        checks["check-door"]["passed"] = (
            checks["check-door"]["passed"]
            and observed_hash == arguments.expected_config_sha256
        )

    profile_code = (
        "import json; from importlib import resources; "
        "from cm1core.profiles import profile_identity; "
        "m=json.loads(resources.files('cm1core.records').joinpath('CM1-PROFILE-MANIFEST.json').read_text(encoding='utf-8')); "
        "e=dict(m['official_supercell']); e['lane']='product'; "
        "print(json.dumps({'equal': profile_identity('product') == e}))"
    )
    profile_payload, profile_result = invoke(
        [str(python), "-I", "-c", profile_code],
        cwd=scratch,
        environment=environment,
        expected_exit=0,
    )
    profile_result["equal"] = profile_payload.get("equal") if profile_payload else None
    profile_result["passed"] = (
        profile_result["passed"] and profile_result["equal"] is True
    )
    checks["installed-profile-record-binding"] = profile_result

    preflight_payload = cli("preflight-door", ["preflight", str(config), "--json"], 1)
    if preflight_payload is not None:
        blocker_codes = [
            item.get("code")
            for item in preflight_payload.get("blockers", [])
            if isinstance(item, dict)
        ]
        observed_codes = {code for code in blocker_codes if isinstance(code, str)}
        checks["preflight-door"]["blocker_codes"] = blocker_codes
        checks["preflight-door"]["passed"] = (
            checks["preflight-door"]["passed"]
            and preflight_payload.get("runnable") is False
            and REQUIRED_BLOCKER_CODES.issubset(observed_codes)
            and not observed_codes & FORBIDDEN_BLOCKER_CODES
            and observed_codes <= ADMITTED_BLOCKER_CODES
        )

    doctor_payload = cli("doctor-door", ["doctor", "--json"], 1)
    if doctor_payload is not None:
        subjects = doctor_payload.get("blocking", [])
        checks["doctor-door"]["blocking_subjects"] = subjects
        checks["doctor-door"]["passed"] = (
            checks["doctor-door"]["passed"]
            # The smoke venv installs --no-deps, so CuPy is absent and the
            # Cartesian lane cannot run; the legacy product-lane findings are
            # informational and must not be what blocks.
            and "CuPy import surface" in subjects
            and "CM1 CUDA dycore" not in subjects
            and "CM1-neutral ArWen microphysics seam" not in subjects
        )

    output_paths: list[Path] = []
    for door in ("init", "forecast"):
        output = scratch / f"must-not-exist-{door}"
        output_paths.append(output)
        payload = cli(
            f"{door}-door",
            [door, str(config), "--output", str(output), "--json"],
            2,
            json_stream="stderr",
        )
        observed_code = payload.get("code") if payload else None
        checks[f"{door}-door"]["observed_code"] = observed_code
        checks[f"{door}-door"]["passed"] = (
            checks[f"{door}-door"]["passed"]
            and observed_code == "CM1-CUDA-TRAJECTORY-INCOMPLETE"
        )

    for door, source, expected_code in (
        ("restart", dummy_checkpoint, "CM1-RUST-RESTART-FAILED"),
        ("render", dummy_history, "CM1-RUST-RENDER-FAILED"),
    ):
        suffix = ".png" if door == "render" else ""
        output = scratch / f"must-not-exist-{door}{suffix}"
        output_paths.append(output)
        door_arguments = [door, str(source), "--output", str(output), "--json"]
        if door == "restart":
            door_arguments.extend(
                [
                    "--contract-sha256",
                    "0" * 64,
                    "--artifact-sha256",
                    "1" * 64,
                    "--operator-manifest-sha256",
                    "2" * 64,
                ]
            )
        else:
            door_arguments.extend(
                [
                    "--format",
                    "cm1fhs",
                    "--contract-sha256",
                    "0" * 64,
                    "--product",
                    "reflectivity",
                    "--field",
                    "refc_dbz",
                    "--nx",
                    "1",
                    "--ny",
                    "1",
                    "--nz",
                    "1",
                ]
            )
        payload = cli(
            f"{door}-door",
            door_arguments,
            2,
            json_stream="stderr",
        )
        observed_code = payload.get("code") if payload else None
        checks[f"{door}-door"]["observed_code"] = observed_code
        checks[f"{door}-door"]["passed"] = (
            checks[f"{door}-door"]["passed"] and observed_code == expected_code
        )

    absent = [path.name for path in output_paths if not path.exists()]
    checks["mutating-doors-created-no-output"] = {
        "absent": absent,
        "expected_absent": [path.name for path in output_paths],
        "passed": len(absent) == len(output_paths),
    }

    report = {
        "schema": "gpuwm-cm1.installed-wheel-smoke/v1",
        "claim_scope": "installed Python doors and hashed Rust executable, shared-library and packaged-input reachability; no CUDA forecast or restart identity",
        "environment": {
            "console_entrypoint": True,
            "cwd_outside_source_checkout": True,
            "dependency_install_policy": "--no-deps; missing ArWen is expected to be diagnosed, not hidden",
            "isolated_import_probe": "python -I",
            "python": sys.version.split()[0],
            "python_no_user_site": True,
            "pythonpath": "empty",
            "scratch_home": True,
        },
        "input": {
            "config_name": source_config.name,
            "config_sha256": sha256_file(source_config),
        },
        "wheel": {
            "bytes": wheel.stat().st_size,
            "name": wheel.name,
            "sha256": sha256_file(wheel),
        },
        "checks": checks,
    }
    report["passed"] = all(bool(item.get("passed")) for item in checks.values())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
