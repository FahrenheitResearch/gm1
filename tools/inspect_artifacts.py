#!/usr/bin/env python3
"""Mechanically inspect gpuwm-cm1 wheel/sdist contents before admission."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import sys
import sysconfig
import tarfile
import zipfile


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

FORBIDDEN_SEGMENTS = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    "upstream",
    "work",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ctl",
    ".dat",
    ".dll",
    ".exe",
    ".gz",
    ".mod",
    ".nc",
    ".o",
    ".p12",
    ".pem",
    ".so",
    ".tar",
    ".tgz",
}
PRIVATE_CONTENT_PATTERNS = {
    # Accept one-or-more separators so JSON-escaped Windows paths cannot evade
    # the same privacy gate that catches their plain-text form.
    "windows-user-path": re.compile(
        rb"(?i)[a-z]:(?:[\\/])+users(?:[\\/])+[^\\/\s\"']+"
    ),
    "wsl-user-path": re.compile(rb"(?i)/mnt/[a-z]/users/[^/\s\"']+"),
    "macos-user-path": re.compile(rb"(?i)/" + rb"users/[a-z0-9._-]+(?:/|$)"),
    # Split the literal so the inspector does not detect its own source.
    "posix-home-path": re.compile(rb"(?i)/" + rb"home/[^/\s\"']+"),
    "unc-user-path": re.compile(rb"(?i)\\\\[^\\\s]+\\" + rb"users\\[^\\\s\"']+"),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    # A private LAN address says where something was run, which is not
    # distribution content (split like the home path).  Machine names are the
    # release denylist gate's job (see release_denylist_problems).
    "private-lan-address": re.compile(rb"\b192\.168" + rb"\.\d{1,3}\.\d{1,3}\b"),
}
NATIVE_TOOL_ABIS = {
    "cm1-state": "gpuwm-cm1-state/v1",
    "gm1-forecast-checkpoint": (
        "gm1.forecast-checkpoint-cli/v1;carrier_abi=gm1.forecast-checkpoint-carriers/v1"
    ),
    "gm1-child-post-step": (
        "gm1.real-50m-child-post-step-cli/v1"
        ";checkpoint_carrier_abi=gm1.forecast-checkpoint-carriers/v1"
        ";history_carrier_abi=gm1.real-50m-child-render-carriers/v1"
    ),
    "cm1-forecast-publish": "gpuwm-cm1-forecast-publish/v3",
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
# The Cartesian runtime opens this cdylib with ctypes, so it is an installed
# artifact whose absence is a broken wheel, not a build by-product.
NATIVE_LIBRARY_NAMES = {"cm1-state": "cm1_state"}
# CM1 runtime assets whose identity this package records and whose bytes it
# must NOT redistribute.  CM1-AUTHORITY.json blocks the tar-only runtime
# assets and its package policy is to ship identities and maps.
REQUIRED_PACKAGED_INPUTS = {"cm1r21.1-cpm_RadConvEquil-input_sounding"}
REQUIRED_INPUT_INDEX = "packaged-inputs.json"
# rust/cm1-state embeds these schemas with include_bytes!; an sdist missing
# one cannot build its own wheel.
REQUIRED_STATE_SCHEMAS = {"cm1-r21.1-namelist-support.json"}


#: Set by --require-manylinux: a release wheel must carry a manylinux_2_28 (or older) tag.
REQUIRE_MANYLINUX = False
#: Set by --require-denylist: the release denylist gate must be present and run.
REQUIRE_DENYLIST = False
#: The release denylist gate.  It is development tooling (it spells the words it
#: denies), so it is in the development repository and not in the sdist or the
#: public source tree; there the denylist checks report that they did not run.
RELEASE_SCANNER = Path(__file__).resolve().parent / "release" / "scan_release.py"


def native_filename(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def native_library_filename(lib_name: str) -> str:
    if os.name == "nt":
        return f"{lib_name}.dll"
    if sys.platform == "darwin":
        return f"lib{lib_name}.dylib"
    return f"lib{lib_name}.so"


def platform_tag() -> str:
    """Mirror the wheel platform tag rust_tools gates the installed payload on."""

    raw = sysconfig.get_platform()
    return raw.replace("-", "_").replace(".", "_").replace(" ", "_")


MANYLINUX_TAG = re.compile(r"^manylinux_(\d+)_(\d+)_(?P<arch>[a-z0-9_]+)$")


def wheel_tag_carries(sealed: object, wheel_platform: str) -> bool:
    """Whether a wheel's platform tag selects the machines the sealed payload was built for.

    The build seals the interpreter's own tag (``linux_x86_64``).  ``auditwheel
    repair`` then renames the wheel to the manylinux tags it proved (for example
    ``manylinux_2_28_x86_64``) without touching the payload, so a manylinux tag of
    the same architecture carries the sealed ``linux_<arch>`` payload.
    """

    if not isinstance(sealed, str):
        return False
    tags = wheel_platform.split(".")
    for tag in tags:
        if tag == sealed:
            continue
        match = MANYLINUX_TAG.match(tag)
        if match is None or sealed != f"linux_{match.group('arch')}":
            return False
    return True


def manylinux_floor(wheel_platform: str) -> tuple[int, int] | None:
    """The lowest glibc a manylinux wheel tag promises, or None for a bare tag."""

    floors = []
    for tag in wheel_platform.split("."):
        match = MANYLINUX_TAG.match(tag)
        if match is not None:
            floors.append((int(match.group(1)), int(match.group(2))))
    return min(floors) if floors else None


# What the GUI release's wheel must carry for `gpuwm-cm1 gui`, the guided
# presets and the quick-looks to work from an install.
REQUIRED_GUI_MEMBERS = {
    "cm1core/gui/static/index.html",
    "cm1core/gui/static/css/app.css",
    "cm1core/gui/static/js/app.js",
    "cm1core/gui/static/js/router.js",
    "cm1core/gui/static/guided/index.js",
    "cm1core/gui/static/guided/guided.css",
    "cm1core/gui/copy/guided.json",
    "cm1core/gui/copy/looks.json",
    "cm1core/data/cm1r21.1-supercell-namelist.input",
    "cm1core/quicklook/fonts/LICENSE-DejaVu.txt",
}
REQUIRED_PRESETS = ("supercell-1km", "supercell-250m", "supercell-100m", "squall-line-1km", "squall-line-250m")
REQUIRED_GUI_MEMBERS |= {
    f"cm1core/presets/{preset}/{name}" for preset in REQUIRED_PRESETS for name in ("namelist.input", "preset.json")
}
# Hidden in this release (cartesian_door.require_listed_component): no member may name it.
UNLISTED_COMPONENT_TOKENS = ("thompson-aerosol",)


EXPECTED_NATIVE_MEMBERS = {
    f"cm1core/bin/{native_filename(name)}" for name in NATIVE_TOOL_ABIS
}
EXPECTED_LIBRARY_MEMBERS = {
    f"cm1core/lib/{native_library_filename(lib_name)}"
    for lib_name in NATIVE_LIBRARY_NAMES.values()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_problems(
    names: list[str], *, allowed_native: set[str] | None = None
) -> list[dict[str, str]]:
    allowed_native = set() if allowed_native is None else allowed_native
    problems: list[dict[str, str]] = []
    for raw_name in names:
        name = raw_name.replace("\\", "/")
        pure = PurePosixPath(name)
        lowered = [part.lower() for part in pure.parts]
        reason: str | None = None
        if raw_name != name:
            reason = "archive member uses a non-POSIX separator"
        elif pure.is_absolute() or ".." in pure.parts:
            reason = "archive member is not a safe relative path"
        elif any(part in FORBIDDEN_SEGMENTS for part in lowered):
            reason = "checkout/build-only directory is forbidden"
        elif pure.suffix.lower() in FORBIDDEN_SUFFIXES and name not in allowed_native:
            reason = "unadmitted binary, dataset, archive, or build product"
        elif any(
            part in {".env", "id_rsa", "id_ed25519"}
            or "credential" in part
            or part.endswith(".key")
            for part in lowered
        ):
            reason = "credential-like member name"
        if reason is not None:
            problems.append({"member": raw_name, "reason": reason})
    return problems


def content_problems(contents: dict[str, bytes]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    machine_tokens = {
        value.encode("utf-8").lower()
        for value in (os.environ.get("COMPUTERNAME"), socket.gethostname())
        if value
        and len(value) >= 4
        and value.lower() not in {"localhost", "localhost.localdomain"}
    }
    for name, payload in contents.items():
        for label, pattern in PRIVATE_CONTENT_PATTERNS.items():
            if pattern.search(payload):
                problems.append({"member": name, "reason": label})
        lowered = payload.lower()
        if any(token in lowered for token in machine_tokens):
            problems.append({"member": name, "reason": "build-hostname"})
    return problems


#: The wheel's base requirements, exactly: ArWen and its data (one patch
#: window each), and the CUDA 13 stack a run needs, so that
#: `pip install gpuwm-cm1` alone is a working install. The ranges hold the
#: validated versions (constraints/gpu-cu13-validated.txt).
EXPECTED_BASE_REQUIREMENTS = (
    ("gpuwm", frozenset({">=2.8.9", "<2.8.10"})),
    ("gpuwm-data", frozenset({">=2.8.9", "<2.8.10"})),
    ("cupy-cuda13x[ctk]", frozenset({">=14.2", "<15"})),
    ("nvidia-cuda-nvrtc", frozenset({">=13.3", "<13.4"})),
    ("nvidia-nvjitlink", frozenset({">=13.3", "<13.4"})),
)


def requirement_key(requirement: str) -> tuple[str, frozenset[str]]:
    """("name[extras]", {specifiers}) of a Requires-Dist value, in any specifier order."""

    text = requirement.split(";", 1)[0].replace(" ", "")
    match = re.match(r"^([A-Za-z0-9_.-]+(?:\[[^\]]*\])?)(.*)$", text)
    if match is None:
        return text.lower(), frozenset()
    name = match.group(1).lower().replace("_", "-")
    return name, frozenset(part for part in match.group(2).strip("()").split(",") if part)

def add_check(
    checks: list[dict[str, object]], check_id: str, passed: bool, detail: object
) -> None:
    checks.append({"id": check_id, "passed": passed, "detail": detail})


def _release_scanner():
    """tools/release/scan_release.py, the release denylist gate, loaded by path."""

    import importlib.util

    path = RELEASE_SCANNER
    spec = importlib.util.spec_from_file_location("gpuwm_cm1_scan_release", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the release denylist gate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


#: Hit rows reported per artifact in the inspection (the gate's own CLI prints all).
DENYLIST_DETAIL_ROWS = 50


def release_denylist_problems(path: Path) -> dict[str, object]:
    """The release denylist gate over every member of one artifact, nested archives opened.

    Without the gate (a tree that is not the development repository) the result
    says so; ``denylist_passed`` fails it only under --require-denylist.
    """

    if not RELEASE_SCANNER.is_file():
        return {"ran": False, "hits": 0, "reason": "the release denylist gate is development tooling, absent here"}
    result = _release_scanner().scan_target(path)
    return {
        "ran": True,
        "hits": result["hits"],
        "hits_by_rule": result["hits_by_rule"],
        "allowlisted": result["allowlisted"],
        "rows": [
            {key: row[key] for key in ("member", "rule", "line", "text")}
            for row in result["hit_rows"][:DENYLIST_DETAIL_ROWS]
        ],
    }


def denylist_passed(denylist: dict[str, object]) -> bool:
    if not denylist["ran"]:
        return not REQUIRE_DENYLIST
    return denylist["hits"] == 0


def inspect_wheel(path: Path, checks: list[dict[str, object]]) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        contents = {
            name: archive.read(name) for name in names if not name.endswith("/")
        }

    add_check(
        checks,
        "wheel-member-policy",
        not (
            issues := member_problems(
                names,
                allowed_native=EXPECTED_NATIVE_MEMBERS | EXPECTED_LIBRARY_MEMBERS,
            )
        ),
        issues,
    )
    add_check(
        checks,
        "wheel-private-content",
        not (leaks := content_problems(contents)),
        leaks,
    )
    denylist = release_denylist_problems(path)
    add_check(checks, "wheel-release-denylist", denylist_passed(denylist), denylist)

    expected_records = {f"cm1core/records/{name}" for name in REQUIRED_RECORDS}
    missing_records = sorted(expected_records - set(names))
    add_check(checks, "wheel-required-records", not missing_records, missing_records)

    # CM1 runtime assets are named, not shipped: CM1-AUTHORITY.json's
    # redistribution record blocks the tar-only runtime assets and its package
    # policy is to ship identities and maps.  Assert the identity index is
    # present and that none of the assets it names is in the wheel.
    redistributed = sorted(
        name
        for name in names
        if name.startswith("cm1core/data/") and Path(name).name in REQUIRED_PACKAGED_INPUTS
    )
    index_present = f"cm1core/data/{REQUIRED_INPUT_INDEX}" in names
    add_check(
        checks,
        "wheel-required-input-identities",
        index_present and not redistributed,
        ([] if index_present else [f"cm1core/data/{REQUIRED_INPUT_INDEX}"]) + redistributed,
    )

    native_manifest_name = "cm1core/native-tools.json"
    missing_native = sorted(
        (EXPECTED_NATIVE_MEMBERS | EXPECTED_LIBRARY_MEMBERS | {native_manifest_name})
        - set(names)
    )
    add_check(checks, "wheel-required-native-tools", not missing_native, missing_native)
    native_issues: list[str] = []
    if not missing_native:
        try:
            native_manifest = json.loads(contents[native_manifest_name])
        except (UnicodeError, json.JSONDecodeError) as error:
            native_issues.append(f"invalid native-tools.json: {error}")
        else:
            if set(native_manifest) != {
                "libraries",
                "platform",
                "platform_tag",
                "python_minimum",
                "schema",
                "tools",
            }:
                native_issues.append("native manifest keys differ from v1")
            if native_manifest.get("schema") != "gpuwm-cm1.native-tools/v1":
                native_issues.append("native manifest schema differs from v1")
            if native_manifest.get("platform") != sysconfig.get_platform():
                native_issues.append(
                    "native manifest platform differs from inspection host"
                )
            if native_manifest.get("platform_tag") != platform_tag():
                native_issues.append(
                    "native manifest platform tag differs from inspection host"
                )
            # The wheel filename carries the tag pip selects on; a payload
            # sealed for another tag installs and then refuses everywhere.
            wheel_tag_platform = path.name[: -len(".whl")].rsplit("-", 1)[-1]
            if not wheel_tag_carries(native_manifest.get("platform_tag"), wheel_tag_platform):
                native_issues.append(
                    f"native manifest platform tag "
                    f"{native_manifest.get('platform_tag')!r} differs from wheel "
                    f"tag {wheel_tag_platform!r}"
                )
            if native_manifest.get("python_minimum") != "3.12":
                native_issues.append("native manifest Python floor differs from 3.12")
            rows = native_manifest.get("tools")
            if not isinstance(rows, list) or len(rows) != len(NATIVE_TOOL_ABIS):
                native_issues.append("native manifest tool inventory is incomplete")
            else:
                observed: set[str] = set()
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {
                        "abi",
                        "bytes",
                        "filename",
                        "sha256",
                    }:
                        native_issues.append("native manifest row keys differ from v1")
                        continue
                    filename = row.get("filename")
                    member = f"cm1core/bin/{filename}"
                    logical = next(
                        (
                            name
                            for name in NATIVE_TOOL_ABIS
                            if native_filename(name) == filename
                        ),
                        None,
                    )
                    if logical is None or member not in EXPECTED_NATIVE_MEMBERS:
                        native_issues.append(f"unadmitted native filename {filename!r}")
                        continue
                    if logical in observed:
                        native_issues.append(f"duplicate native row {logical}")
                        continue
                    observed.add(logical)
                    payload = contents[member]
                    if row.get("abi") != NATIVE_TOOL_ABIS[logical]:
                        native_issues.append(f"ABI mismatch for {logical}")
                    if row.get("bytes") != len(payload):
                        native_issues.append(f"byte-count mismatch for {logical}")
                    if row.get("sha256") != hashlib.sha256(payload).hexdigest():
                        native_issues.append(f"SHA-256 mismatch for {logical}")
                    if os.name == "nt" and not payload.startswith(b"MZ"):
                        native_issues.append(f"{logical} is not a PE executable")
                    if os.name != "nt" and not payload.startswith(b"\x7fELF"):
                        native_issues.append(f"{logical} is not an ELF executable")
                if observed != set(NATIVE_TOOL_ABIS):
                    native_issues.append("native logical tool inventory is incomplete")
            library_rows = native_manifest.get("libraries")
            if not isinstance(library_rows, list) or len(library_rows) != len(
                NATIVE_LIBRARY_NAMES
            ):
                native_issues.append("native manifest library inventory is incomplete")
            else:
                seen: set[str] = set()
                for row in library_rows:
                    if not isinstance(row, dict) or set(row) != {
                        "bytes",
                        "filename",
                        "name",
                        "sha256",
                    }:
                        native_issues.append(
                            "native manifest library row keys differ from v1"
                        )
                        continue
                    logical = row.get("name")
                    member = f"cm1core/lib/{row.get('filename')}"
                    if (
                        logical not in NATIVE_LIBRARY_NAMES
                        or member not in EXPECTED_LIBRARY_MEMBERS
                    ):
                        native_issues.append(
                            f"unadmitted native library {row.get('filename')!r}"
                        )
                        continue
                    if logical in seen:
                        native_issues.append(f"duplicate native library row {logical}")
                        continue
                    seen.add(logical)
                    payload = contents[member]
                    if row.get("bytes") != len(payload):
                        native_issues.append(f"byte-count mismatch for {logical} library")
                    if row.get("sha256") != hashlib.sha256(payload).hexdigest():
                        native_issues.append(f"SHA-256 mismatch for {logical} library")
                    if os.name == "nt" and not payload.startswith(b"MZ"):
                        native_issues.append(f"{logical} library is not a PE image")
                    if os.name != "nt" and not payload.startswith(b"\x7fELF"):
                        native_issues.append(f"{logical} library is not an ELF image")
                if seen != set(NATIVE_LIBRARY_NAMES):
                    native_issues.append(
                        "native logical library inventory is incomplete"
                    )
    add_check(checks, "wheel-native-tool-manifest", not native_issues, native_issues)

    missing_gui = sorted(REQUIRED_GUI_MEMBERS - set(names))
    add_check(checks, "wheel-gui-presets-and-copy", not missing_gui, missing_gui)
    mock = sorted(name for name in names if name.startswith("cm1core/gui/static/mock/"))
    add_check(checks, "wheel-no-taste-check-mock", not mock, mock[:10])
    unlisted = sorted(
        name for name in names if any(token in name.lower() for token in UNLISTED_COMPONENT_TOKENS)
    )
    add_check(checks, "wheel-unlisted-component-decks-absent", not unlisted, unlisted)
    wheel_platform = path.name[: -len(".whl")].rsplit("-", 1)[-1]
    floor = manylinux_floor(wheel_platform)
    add_check(
        checks,
        "wheel-manylinux-2-28",
        (not REQUIRE_MANYLINUX) or (floor is not None and floor <= (2, 28)),
        {"platform_tag": wheel_platform, "glibc_floor": None if floor is None else f"{floor[0]}.{floor[1]}",
         "required": REQUIRE_MANYLINUX},
    )

    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    add_check(checks, "wheel-one-metadata", len(metadata_names) == 1, metadata_names)
    if len(metadata_names) == 1:
        metadata = BytesParser(policy=policy.default).parsebytes(
            contents[metadata_names[0]]
        )
        base_requirements = [
            value.replace(" ", "")
            for value in metadata.get_all("Requires-Dist", [])
            if ";extra==" not in value.replace(" ", "").lower()
        ]
        # The engine and its data distribution, one bounded patch window each,
        # in either specifier order the build backend may write.
        allowed_requirements = (
            {"gpuwm<2.8.10,>=2.8.9", "gpuwm>=2.8.9,<2.8.10"},
            {"gpuwm-data<2.8.10,>=2.8.9", "gpuwm-data>=2.8.9,<2.8.10"},
        )
        arwen_requirements = [
            requirement for requirement in base_requirements
            if requirement_key(requirement)[0] in ("gpuwm", "gpuwm-data")
        ]
        add_check(
            checks,
            "wheel-bounded-arwen-dependency",
            len(arwen_requirements) == len(allowed_requirements)
            and all(
                any(requirement in allowed for requirement in arwen_requirements)
                for allowed in allowed_requirements
            ),
            arwen_requirements,
        )
        # `pip install gpuwm-cm1` alone installs the CUDA 13 stack: nothing
        # else is required, and every requirement is bounded.
        found = {requirement_key(requirement) for requirement in base_requirements}
        add_check(
            checks,
            "wheel-gpu-stack-in-base-dependencies",
            found == set(EXPECTED_BASE_REQUIREMENTS),
            {"found": sorted(base_requirements),
             "expected": sorted(f"{name}{','.join(sorted(specs))}" for name, specs in EXPECTED_BASE_REQUIREMENTS)},
        )
        # The old install lines ("gpuwm-cm1[gpu-cu13]", "gpuwm-cm1[gpu]") keep working.
        extras = {value.strip().lower() for value in metadata.get_all("Provides-Extra", [])}
        extra_requirements = sorted(
            value for value in metadata.get_all("Requires-Dist", [])
            if re.search(r"extra\s*==\s*['\"](?:gpu|gpu-cu13)['\"]", value)
        )
        add_check(
            checks,
            "wheel-gpu-extras-are-accepted-aliases",
            {"gpu", "gpu-cu13"} <= extras and not extra_requirements,
            {"provides_extra": sorted(extras), "gpu_extra_requirements": extra_requirements},
        )
        requires_python = metadata.get("Requires-Python")
        add_check(
            checks, "wheel-requires-python-3-12", requires_python == ">=3.12", requires_python
        )
        project_urls = metadata.get_all("Project-URL", [])
        add_check(
            checks, "wheel-no-invented-project-url", not project_urls, project_urls
        )

    license_members = [
        name
        for name in names
        if ".dist-info/licenses/" in name
        and name.rsplit("/", 1)[-1] in {"LICENSE", "NOTICE"}
    ]
    add_check(
        checks,
        "wheel-package-license-notice",
        len(license_members) == 2,
        license_members,
    )
    return {
        "bytes": path.stat().st_size,
        "member_count": len(names),
        "path": path.name,
        "sha256": sha256_file(path),
    }


def inspect_sdist(path: Path, checks: list[dict[str, object]]) -> dict[str, object]:
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        unsafe_types = [
            member.name for member in members if not (member.isfile() or member.isdir())
        ]
        contents: dict[str, bytes] = {}
        for member in members:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is not None:
                contents[member.name] = stream.read()

    add_check(checks, "sdist-regular-members", not unsafe_types, unsafe_types)
    add_check(
        checks, "sdist-member-policy", not (issues := member_problems(names)), issues
    )
    add_check(
        checks,
        "sdist-private-content",
        not (leaks := content_problems(contents)),
        leaks,
    )
    denylist = release_denylist_problems(path)
    add_check(checks, "sdist-release-denylist", denylist_passed(denylist), denylist)

    roots = {
        PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts
    }
    add_check(checks, "sdist-single-root", len(roots) == 1, sorted(roots))
    root = next(iter(roots)) if len(roots) == 1 else "<invalid-root>"
    required = {
        f"{root}/LICENSE",
        f"{root}/NOTICE",
        f"{root}/README.md",
        f"{root}/build_rust.py",
        f"{root}/docs/RELEASE-NOTES.md",
        f"{root}/examples/official-supercell-product.toml",
        f"{root}/examples/real-atmosphere-100m-product.toml",
        f"{root}/tools/authority/fetch_cm1_21_1.py",
        f"{root}/tools/inspect_artifacts.py",
        f"{root}/tools/verify_authority.py",
        f"{root}/tools/authority/cm1r21.1-content-sha256.tsv",
        f"{root}/rust/cm1-state/Cargo.lock",
        f"{root}/rust/cm1-state/Cargo.toml",
        f"{root}/rust/cm1-state/src/lib.rs",
        f"{root}/rust/cm1-state/src/main.rs",
        f"{root}/rust/cm1-state/src/bin/cm1_forecast_publish.rs",
        f"{root}/rust/cm1-state/src/bin/gm1_child_post_step.rs",
        # analytic_soundings.rs's crate test reads it with include_str!.
        f"{root}/rust/cm1-state/evidence/cm1-base-columns.txt",
        f"{root}/rust/cm1-render/Cargo.lock",
        f"{root}/rust/cm1-render/Cargo.toml",
        f"{root}/rust/cm1-render/src/main.rs",
        f"{root}/setup.py",
        f"{root}/tools/smoke_installed_cuda_wheel.py",
        f"{root}/tools/smoke_installed_wheel.py",
        f"{root}/tools/release/build_manylinux.sh",
        f"{root}/tools/release/smoke_cm1_user.sh",
        f"{root}/tools/release/cm1_user_script.py",
        f"{root}/tools/release/doc_commands.py",
        f"{root}/docs/QUICKSTART.md",
        f"{root}/docs/CM1-USERS.md",
        f"{root}/docs/COMMANDS.md",
        f"{root}/docs/GM1-DECKS.md",
        f"{root}/docs/INSTALL.md",
        f"{root}/docs/TROUBLESHOOTING.md",
        f"{root}/constraints/gpu-cu13-validated.txt",
    }
    required |= {f"{root}/src/cm1core/records/{name}" for name in REQUIRED_RECORDS}
    required.add(f"{root}/src/cm1core/data/{REQUIRED_INPUT_INDEX}")
    # cm1_namelist.rs embeds these with include_bytes!, so an sdist without
    # them cannot build its own wheel at all.
    required |= {
        f"{root}/rust/cm1-state/schema/{name}" for name in REQUIRED_STATE_SCHEMAS
    }
    missing = sorted(required - set(names))
    add_check(checks, "sdist-required-source-evidence", not missing, missing)
    return {
        "bytes": path.stat().st_size,
        "member_count": len(names),
        "path": path.name,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument(
        "--require-manylinux",
        action="store_true",
        help="fail unless the wheel carries a manylinux tag of glibc 2.28 or older (a release wheel)",
    )
    parser.add_argument(
        "--require-denylist",
        action="store_true",
        help="fail unless the release denylist gate (development tooling) is present and finds nothing",
    )
    arguments = parser.parse_args()
    global REQUIRE_MANYLINUX, REQUIRE_DENYLIST
    REQUIRE_MANYLINUX = arguments.require_manylinux
    REQUIRE_DENYLIST = arguments.require_denylist
    wheel = arguments.wheel.expanduser().resolve(strict=True)
    sdist = arguments.sdist.expanduser().resolve(strict=True)
    checks: list[dict[str, object]] = []
    report = {
        "schema": "gpuwm-cm1.artifact-inspection/v1",
        "artifacts": {
            "wheel": inspect_wheel(wheel, checks),
            "sdist": inspect_sdist(sdist, checks),
        },
        "checks": checks,
    }
    report["passed"] = all(bool(check["passed"]) for check in checks)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
