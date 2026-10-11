#!/usr/bin/env python3
"""Install a wheel offline and prove its CUDA modules remain import-cold.

This smoke test deliberately does not compile or launch a CUDA kernel.  Its
scope is the installed artifact: wheel/module byte identity, importability
without CuPy or another Python CUDA runtime, embedded-source pin integrity,
and installed console/native entrypoint, shared-library and packaged-input
reachability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import venv
import zipfile


CUDA_RUNTIME_PREFIXES = ("cuda", "cupy", "cupy_backends", "numba.cuda")
# Only these tools have a console script; the remaining sealed artifacts
# are reached through the verified inventory instead.
EXPECTED_NATIVE_ABIS = {
    "cm1-state": "gpuwm-cm1-state/v1",
    "gm1-forecast-checkpoint": (
        "gm1.forecast-checkpoint-cli/v1;carrier_abi=gm1.forecast-checkpoint-carriers/v1"
    ),
    "gm1-cartesian-restart": (
        "gm1.cartesian-restart-cli/v1;carrier_abi=gm1.cartesian-restart-carriers/v1"
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_cuda_inventory(path: Path) -> list[dict[str, object]]:
    """Return the sealed Python bytes for every top-level CUDA module."""

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            parts = member.split("/")
            if (
                len(parts) != 2
                or parts[0] != "cm1core"
                or not parts[1].startswith("cuda_")
                or not parts[1].endswith(".py")
            ):
                continue
            payload = archive.read(member)
            rows.append(
                {
                    "bytes": len(payload),
                    "member": member,
                    "module": f"cm1core.{parts[1][:-3]}",
                    "sha256": sha256_bytes(payload),
                }
            )
    return rows


def source_pin_rows(module: object) -> list[dict[str, object]]:
    """Recompute conventional CUDA ``*_SOURCE_SHA256`` module pins."""

    namespace = vars(module)
    source_names = sorted(
        name
        for name, value in namespace.items()
        if name.startswith("CUDA_")
        and name.endswith("_SOURCE")
        and isinstance(value, (bytes, str))
    )
    rows: list[dict[str, object]] = []
    for source_name in source_names:
        value = namespace[source_name]
        payload = value.encode("utf-8") if isinstance(value, str) else value
        bytes_name = f"{source_name}_BYTES"
        sealed_bytes = namespace.get(bytes_name, payload)
        byte_alias_matches = isinstance(sealed_bytes, bytes) and sealed_bytes == payload
        pin_name = f"{source_name}_SHA256"
        expected = namespace.get(pin_name)
        observed = sha256_bytes(payload)
        pin_matches = (
            isinstance(expected, str) and len(expected) == 64 and expected == observed
        )
        rows.append(
            {
                "bytes": len(payload),
                "byte_alias_matches": byte_alias_matches,
                "expected_sha256": expected if isinstance(expected, str) else None,
                "observed_sha256": observed,
                "passed": byte_alias_matches and pin_matches,
                "pin": pin_name,
                "source": source_name,
            }
        )
    return rows


def loaded_cuda_runtime_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in CUDA_RUNTIME_PREFIXES
        )
    )


def installed_probe() -> dict[str, object]:
    """Run inside the artifact-only venv and return a path-sanitized report."""

    import cm1core

    source_root = Path(__file__).resolve().parents[1]
    package_dir = Path(cm1core.__file__).resolve().parent
    short_names = sorted(
        item.name
        for item in pkgutil.iter_modules(cm1core.__path__)
        if item.name.startswith("cuda_")
    )
    modules: list[dict[str, object]] = []
    for short_name in short_names:
        qualified = f"cm1core.{short_name}"
        row: dict[str, object] = {
            "member": f"cm1core/{short_name}.py",
            "module": qualified,
        }
        try:
            module = importlib.import_module(qualified)
            module_path = Path(inspect.getfile(module)).resolve()
            row.update(
                {
                    "file_sha256": sha256_file(module_path),
                    "installed_under_package": module_path.parent == package_dir,
                    "source_pins": source_pin_rows(module),
                }
            )
        except BaseException as error:  # preserve a row for every broken import
            row.update(
                {
                    "error_type": type(error).__name__,
                    "installed_under_package": False,
                    "source_pins": [],
                }
            )
        modules.append(row)

    native: dict[str, object] = {}
    try:
        from cm1core.data import packaged_input_path, packaged_inputs
        from cm1core.rust_tools import (
            EXPECTED_LIBRARIES,
            native_library_path,
            verify_native_tools,
        )

        native["tools"] = sorted(verify_native_tools())
        native["libraries"] = sorted(
            name for name in EXPECTED_LIBRARIES if native_library_path(name).is_file()
        )
        native["inputs"] = sorted(
            name for name in packaged_inputs() if packaged_input_path(name).is_file()
        )
    except BaseException as error:  # a broken payload is a reported row, not a crash
        native = {"error_type": type(error).__name__}

    return {
        "cuda_runtime_modules_loaded": loaded_cuda_runtime_modules(),
        "module_count": len(modules),
        "modules": modules,
        "native_payload": native,
        "package_from_source_checkout": package_dir.is_relative_to(source_root),
    }


def _command_result(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    required_stdout: tuple[str, ...] = (),
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "error_type": type(error).__name__,
            "passed": False,
        }
    missing = [token for token in required_stdout if token not in completed.stdout]
    return {
        "exit": completed.returncode,
        "expected_exit": 0,
        "missing_stdout_tokens": missing,
        "passed": completed.returncode == 0 and not missing,
        "stderr_bytes": len(completed.stderr.encode("utf-8")),
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
    }


def _emit(report: dict[str, object]) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    arguments = parser.parse_args()

    wheel = arguments.wheel.expanduser().resolve(strict=True)
    work = arguments.work_dir.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if work.exists():
        parser.error(f"work directory already exists; refusing to reuse it: {work}")
    if work.is_relative_to(source_root):
        parser.error("work directory must be outside the source checkout")

    wheel_modules = wheel_cuda_inventory(wheel)
    wheel_by_name = {str(row["module"]): row for row in wheel_modules}
    scratch = work / "scratch"
    scratch_home = scratch / "home"
    venv_dir = work / "venv"
    scratch_home.mkdir(parents=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    console = scripts / ("gpuwm-cm1.exe" if os.name == "nt" else "gpuwm-cm1")

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["HOME"] = str(scratch_home)
    environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = ""
    environment["USERPROFILE"] = str(scratch_home)

    install = subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        cwd=scratch,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    checks: dict[str, dict[str, object]] = {
        "install-wheel-offline-no-deps": {
            "exit": install.returncode,
            "expected_exit": 0,
            "passed": install.returncode == 0,
        },
        "wheel-cuda-module-inventory": {
            "module_count": len(wheel_modules),
            "passed": bool(wheel_modules) and len(wheel_by_name) == len(wheel_modules),
        },
    }

    if install.returncode != 0:
        return _emit(
            {
                "checks": checks,
                "claim_scope": "installed-artifact verification only; no CUDA compile, launch, forecast, or numerical claim",
                "passed": False,
                "schema": "gpuwm-cm1.installed-cuda-wheel-smoke/v1",
                "wheel": {
                    "bytes": wheel.stat().st_size,
                    "name": wheel.name,
                    "sha256": sha256_file(wheel),
                },
            }
        )

    probe = subprocess.run(
        [str(python), "-I", str(Path(__file__).resolve()), "--installed-probe"],
        cwd=scratch,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    probe_payload: dict[str, object] | None = None
    try:
        parsed = json.loads(probe.stdout)
        if isinstance(parsed, dict):
            probe_payload = parsed
    except json.JSONDecodeError:
        pass
    checks["installed-probe-process"] = {
        "exit": probe.returncode,
        "expected_exit": 0,
        "json_object": probe_payload is not None,
        "passed": probe.returncode == 0 and probe_payload is not None,
        "stderr_bytes": len(probe.stderr.encode("utf-8")),
    }

    installed_modules: list[dict[str, object]] = []
    if probe_payload is not None:
        candidate = probe_payload.get("modules")
        if isinstance(candidate, list):
            installed_modules = [row for row in candidate if isinstance(row, dict)]
    installed_by_name = {
        str(row.get("module")): row
        for row in installed_modules
        if isinstance(row.get("module"), str)
    }
    missing = sorted(set(wheel_by_name) - set(installed_by_name))
    unexpected = sorted(set(installed_by_name) - set(wheel_by_name))
    checks["installed-cuda-module-inventory"] = {
        "installed_count": len(installed_by_name),
        "missing": missing,
        "passed": not missing and not unexpected and bool(installed_by_name),
        "unexpected": unexpected,
        "wheel_count": len(wheel_by_name),
    }

    byte_mismatches: list[str] = []
    import_failures: list[str] = []
    source_pin_failures: list[str] = []
    source_pin_count = 0
    for name, installed in installed_by_name.items():
        wheel_row = wheel_by_name.get(name)
        if wheel_row is None:
            continue
        if installed.get("file_sha256") != wheel_row["sha256"]:
            byte_mismatches.append(name)
        if not installed.get("installed_under_package") or "error_type" in installed:
            import_failures.append(name)
        pins = installed.get("source_pins")
        if isinstance(pins, list):
            for pin in pins:
                if not isinstance(pin, dict):
                    source_pin_failures.append(name)
                    continue
                source_pin_count += 1
                if pin.get("passed") is not True:
                    source_pin_failures.append(f"{name}:{pin.get('pin')}")

    checks["installed-module-bytes-match-wheel"] = {
        "mismatches": sorted(byte_mismatches),
        "passed": not byte_mismatches and bool(installed_by_name),
    }
    checks["cuda-module-imports"] = {
        "failures": sorted(import_failures),
        "passed": not import_failures and bool(installed_by_name),
    }
    runtime_modules = (
        probe_payload.get("cuda_runtime_modules_loaded", [])
        if probe_payload is not None
        else []
    )
    from_checkout = (
        probe_payload.get("package_from_source_checkout")
        if probe_payload is not None
        else None
    )
    checks["cuda-cold-imports"] = {
        "loaded_python_cuda_runtime_modules": runtime_modules,
        "package_from_source_checkout": from_checkout,
        "passed": runtime_modules == [] and from_checkout is False,
    }
    checks["embedded-cuda-source-pins"] = {
        "failures": sorted(source_pin_failures),
        "passed": source_pin_count > 0 and not source_pin_failures,
        "pin_count": source_pin_count,
    }

    help_tokens = ("usage: gpuwm-cm1", "forecast", "restart", "render")
    checks["console-entrypoint-help"] = _command_result(
        [str(console), "--help"],
        cwd=scratch,
        environment=environment,
        required_stdout=help_tokens,
    )
    checks["python-module-help"] = _command_result(
        [str(python), "-I", "-m", "cm1core", "--help"],
        cwd=scratch,
        environment=environment,
        required_stdout=help_tokens,
    )

    native_rows: dict[str, dict[str, object]] = {}
    for name, expected in EXPECTED_NATIVE_ABIS.items():
        entrypoint = scripts / (f"{name}.exe" if os.name == "nt" else name)
        if not entrypoint.is_file():
            native_rows[name] = {
                "expected": expected,
                "observed": None,
                "passed": False,
            }
            continue
        completed = subprocess.run(
            [str(entrypoint), "--abi"],
            cwd=scratch,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        observed = completed.stdout.strip()
        native_rows[name] = {
            "exit": completed.returncode,
            "expected": expected,
            "observed": observed,
            "passed": completed.returncode == 0
            and observed == expected
            and not completed.stderr,
        }
    checks["native-entrypoint-abis"] = {
        "passed": all(row["passed"] is True for row in native_rows.values()),
        "tools": native_rows,
    }

    native_payload = (
        probe_payload.get("native_payload") if probe_payload is not None else None
    )
    if not isinstance(native_payload, dict):
        native_payload = {}
    checks["installed-native-payload"] = {
        "observed": native_payload,
        "passed": set(native_payload.get("tools", [])) == SEALED_NATIVE_TOOLS
        and set(native_payload.get("libraries", [])) == SEALED_NATIVE_LIBRARIES
        and REQUIRED_PACKAGED_INPUTS.issubset(native_payload.get("inputs", [])),
    }

    report: dict[str, object] = {
        "checks": checks,
        "claim_scope": "installed-artifact import, bytes, embedded-source pins, help, native ABI and sealed native-payload reachability only; no CUDA compile, launch, forecast, restart, or numerical claim",
        "environment": {
            "cuda_visible_devices": "empty",
            "cwd_outside_source_checkout": True,
            "dependency_install_policy": "--no-index --no-deps",
            "isolated_import_probe": "python -I",
            "nvidia_visible_devices": "none",
            "python": sys.version.split()[0],
            "python_no_user_site": True,
            "pythonpath": "empty",
            "scratch_home": True,
        },
        "installed_cuda_modules": installed_modules,
        "passed": all(item.get("passed") is True for item in checks.values()),
        "schema": "gpuwm-cm1.installed-cuda-wheel-smoke/v1",
        "wheel": {
            "bytes": wheel.stat().st_size,
            "cuda_modules": wheel_modules,
            "name": wheel.name,
            "sha256": sha256_file(wheel),
        },
    }
    return _emit(report)


if __name__ == "__main__":
    if sys.argv[1:] == ["--installed-probe"]:
        print(json.dumps(installed_probe(), indent=2, sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(main())
