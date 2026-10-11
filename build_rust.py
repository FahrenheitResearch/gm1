"""Setuptools commands for building and sealing the Rust command-line tools."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Distribution
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py


NATIVE_MANIFEST_SCHEMA = "gpuwm-cm1.native-tools/v1"
# Example decks kept in the source tree but not installed (see RustBuildPy).
UNSHIPPED_EXAMPLES = frozenset({"cartesian-supercell-thompson-aerosol.toml"})
# (installed name, crate manifest, required --abi, accepted Cargo file stems).
# Cargo names an auto-discovered binary after its source file, so a crate
# target without an explicit [[bin]] stanza lands under the underscored
# spelling.  Accepting both keeps the seal correct before and after
# rust/cm1-state/Cargo.toml declares the hyphenated target; trim the second
# stem once that stanza exists.
NATIVE_TOOLS = (
    ("cm1-state", "rust/cm1-state/Cargo.toml", "gpuwm-cm1-state/v1", ("cm1-state",)),
    (
        "gm1-forecast-checkpoint",
        "rust/cm1-state/Cargo.toml",
        "gm1.forecast-checkpoint-cli/v1;carrier_abi=gm1.forecast-checkpoint-carriers/v1",
        ("gm1-forecast-checkpoint",),
    ),
    (
        "gm1-child-post-step",
        "rust/cm1-state/Cargo.toml",
        "gm1.real-50m-child-post-step-cli/v1"
        ";checkpoint_carrier_abi=gm1.forecast-checkpoint-carriers/v1"
        ";history_carrier_abi=gm1.real-50m-child-render-carriers/v1",
        ("gm1-child-post-step",),
    ),
    (
        "checkpoint-wind-grads",
        "rust/cm1-render/Cargo.toml",
        "gpuwm-cm1-checkpoint-wind-grads/v1",
        ("checkpoint-wind-grads",),
    ),
    (
        "gm1-cartesian-restart",
        "rust/cm1-state/Cargo.toml",
        "gm1.cartesian-restart-cli/v1;carrier_abi=gm1.cartesian-restart-carriers/v1",
        ("gm1-cartesian-restart", "gm1_cartesian_restart"),
    ),
    (
        "cm1-forecast-publish",
        "rust/cm1-state/Cargo.toml",
        "gpuwm-cm1-forecast-publish/v3",
        ("cm1-forecast-publish", "cm1_forecast_publish"),
    ),
    (
        # The exporter that makes a GM1 frame readable by CM1's own ecosystem:
        # GrADS, CDO, ncview via the .ctl, and every CM1 post-processing
        # script.  writeout.F:8761-8995 and :9265-9528.
        "gm1-history-grads-export",
        "rust/cm1-state/Cargo.toml",
        "gpuwm-cm1-history-grads-export/v1",
        ("gm1-history-grads-export",),
    ),
    (
        # The CM1 namelist parser the importer runs: types every assignment
        # against the 583-field r21.1 inventory and, with --unknown report,
        # returns the keys r21.1 does not declare instead of refusing.
        "cm1-r21-namelist",
        "rust/cm1-state/Cargo.toml",
        "gm1.cm1-r21.1-namelist-cli/v2"
        ";document_schema=gm1.cm1-r21.1-namelist-document/v1"
        ";support_schema=gm1.cm1-r21.1-configuration-support/v1"
        ";binding_schema=gm1.cm1-r21.1-configuration-binding/v1"
        ";inspection_schema=gm1.cm1-r21.1-namelist-inspection/v1",
        ("cm1-r21-namelist",),
    ),
    ("cm1-render", "rust/cm1-render/Cargo.toml", "gpuwm-cm1-render/v1", ("cm1-render",)),
    (
        "cm1-f64-window",
        "rust/cm1-render/Cargo.toml",
        "gpuwm-cm1-f64-window/v1",
        ("cm1-f64-window",),
    ),
)
# (installed name, crate manifest, Cargo [lib] name).  The Cartesian runtime
# opens these with ctypes.CDLL, so the cdylib is an installed artifact, not a
# build by-product of the executables that share the cargo invocation.
NATIVE_LIBRARIES = (("cm1-state", "rust/cm1-state/Cargo.toml", "cm1_state"),)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _library_name(lib_name: str) -> str:
    if os.name == "nt":
        return f"{lib_name}.dll"
    if sys.platform == "darwin":
        return f"lib{lib_name}.dylib"
    return f"lib{lib_name}.so"


def platform_tag() -> str:
    """Return the wheel platform tag this interpreter's payload is built for.

    ``sysconfig.get_platform()`` is the raw string; the wheel tag is the same
    string with ``-``, ``.`` and spaces replaced by ``_``.  The tag, not the
    raw string, is the identity a wheel is selected by, so it is what the
    installed manifest is allowed to gate on.  ``PlatformWheel.get_tag``
    refuses any build whose real wheel tag disagrees with this derivation.
    """

    raw = sysconfig.get_platform()
    return raw.replace("-", "_").replace(".", "_").replace(" ", "_")


def _read_abi(path: Path) -> str:
    completed = subprocess.run(
        [str(path), "--abi"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native tool ABI probe failed for {path.name}: "
            f"exit={completed.returncode}, stderr={completed.stderr.strip()!r}"
        )
    abi = completed.stdout.strip()
    if not abi or "\n" in abi or "\r" in abi:
        raise RuntimeError(f"native tool {path.name} emitted an invalid ABI {abi!r}")
    return abi


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific even though Python has no extension ABI."""

    def has_ext_modules(self) -> bool:
        return True

    def is_pure(self) -> bool:
        return False


class PlatformWheel(_bdist_wheel):
    """Use a Python-independent tag for the platform-specific Rust payload."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, wheel_platform = super().get_tag()
        build = self.get_finalized_command("build")
        manifest_path = Path(build.build_lib) / "cm1core" / "native-tools.json"
        if manifest_path.is_file():
            # A wheel whose tag disagrees with the sealed platform installs a
            # payload that rust_tools then refuses on every machine, so fail
            # the build instead of publishing an unusable artifact.
            sealed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if sealed.get("platform_tag") != wheel_platform:
                raise RuntimeError(
                    f"sealed native payload targets platform tag "
                    f"{sealed.get('platform_tag')!r}; this wheel is tagged "
                    f"{wheel_platform!r}"
                )
        return "py3", "none", wheel_platform


class RustBuildPy(_build_py):
    """Build locked Rust crates and copy only the admitted executables."""

    def run(self) -> None:
        root = Path(__file__).resolve().parent
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError(
                "building gpuwm-cm1 wheels requires Cargo because cm1-state and "
                "cm1-render are mandatory installed runtime artifacts"
            )
        target = root / "build" / "rust-native-target"
        environment = os.environ.copy()
        environment["CARGO_INCREMENTAL"] = "0"
        inherited_rustflags = environment.get("RUSTFLAGS", "").strip()
        remap_flag = f"--remap-path-prefix={root}=."
        environment["RUSTFLAGS"] = " ".join(
            value for value in (inherited_rustflags, remap_flag) if value
        )
        built_manifests: set[str] = set()
        crate_manifests = [relative for _, relative, _, _ in NATIVE_TOOLS]
        crate_manifests.extend(relative for _, relative, _ in NATIVE_LIBRARIES)
        for relative_manifest in crate_manifests:
            if relative_manifest in built_manifests:
                continue
            manifest = root / relative_manifest
            if not manifest.is_file():
                raise RuntimeError(
                    f"required Rust manifest is missing: {relative_manifest}"
                )
            self.announce(f"building locked Rust crate {relative_manifest}", level=2)
            subprocess.run(
                [
                    cargo,
                    "build",
                    "--release",
                    "--locked",
                    "--manifest-path",
                    str(manifest),
                    "--target-dir",
                    str(target),
                ],
                cwd=root,
                env=environment,
                check=True,
            )
            built_manifests.add(relative_manifest)

        super().run()
        package_root = Path(self.build_lib) / "cm1core"
        # The example decks are the only runnable documents a user has, and a
        # wheel install has no checkout to find them in.  Ship the top-level
        # decks (not examples/oracle, which are differential-harness inputs)
        # so `gpuwm-cm1 examples` can list and copy them.
        # thompson-aerosol-mp28 is not offered by this release (the door
        # refuses it; cartesian_door.UNLISTED_COMPONENTS), so its deck stays in
        # the source tree and out of the wheel.
        examples = sorted(
            deck
            for deck in (root / "examples").glob("*.toml")
            if deck.name not in UNSHIPPED_EXAMPLES
        )
        if not examples:
            raise RuntimeError(
                "examples/*.toml is missing; the wheel would ship no runnable deck"
            )
        example_destination = package_root / "examples"
        example_destination.mkdir(parents=True, exist_ok=True)
        for deck in examples:
            shutil.copyfile(deck, example_destination / deck.name)
        destination = package_root / "bin"
        destination.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for name, _, expected_abi, stems in NATIVE_TOOLS:
            filename = _executable_name(name)
            emitted = [
                target / "release" / _executable_name(stem)
                for stem in stems
                if (target / "release" / _executable_name(stem)).is_file()
            ]
            if not emitted:
                raise RuntimeError(
                    f"Cargo did not emit required native tool {filename}"
                )
            source = emitted[0]
            observed_abi = _read_abi(source)
            if observed_abi != expected_abi:
                raise RuntimeError(
                    f"native tool {filename} ABI is {observed_abi!r}; "
                    f"required {expected_abi!r}"
                )
            output = destination / filename
            shutil.copyfile(source, output)
            # copyfile drops the mode bits.  Without the executable bit the
            # wheel installs native tools that cannot be executed at all on
            # POSIX, so restore it explicitly and keep it in the wheel.
            output.chmod(0o755)
            records.append(
                {
                    "abi": observed_abi,
                    "bytes": output.stat().st_size,
                    "filename": filename,
                    "sha256": _sha256(output),
                }
            )

        library_destination = package_root / "lib"
        library_destination.mkdir(parents=True, exist_ok=True)
        libraries: list[dict[str, object]] = []
        for name, _, lib_name in NATIVE_LIBRARIES:
            filename = _library_name(lib_name)
            source = target / "release" / filename
            if not source.is_file():
                raise RuntimeError(
                    f"Cargo did not emit required native library {filename}"
                )
            output = library_destination / filename
            shutil.copyfile(source, output)
            # A cdylib is dlopen'd, not exec'd, but the same copyfile mode loss
            # leaves it 0600 in the wheel; keep the conventional 0755 so a
            # shared installation stays readable to every user of the venv.
            output.chmod(0o755)
            libraries.append(
                {
                    "bytes": output.stat().st_size,
                    "filename": filename,
                    "name": name,
                    "sha256": _sha256(output),
                }
            )

        manifest = {
            "libraries": libraries,
            "platform": sysconfig.get_platform(),
            "platform_tag": platform_tag(),
            "python_minimum": "3.12",
            "schema": NATIVE_MANIFEST_SCHEMA,
            "tools": records,
        }
        manifest_path = package_root / "native-tools.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.announce(
            f"sealed {len(records)} Rust executables and {len(libraries)} Rust "
            f"libraries for {platform_tag()} with Python "
            f"{sys.version_info.major}.{sys.version_info.minor}",
            level=2,
        )
