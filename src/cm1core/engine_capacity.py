"""Reproducible, explicitly selected ArWen WSM6 column-capacity builds.

The default 2.8.9 estate remains separately byte-admitted.  This module
changes two capacity declarations in a new directory, never an installed
dependency or an imported module.  WSM6 equations, coefficients, public
dispatch and CUDA source bytes are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from . import engine_pin


#: The ``.dist-info`` directory the overlay copies alongside ``gpuwm/`` so the
#: built tree answers ``importlib.metadata.version("gpuwm")`` with the audited
#: version.  Derived from the pinned version rather than typed, so a future
#: version bump cannot leave a stale directory name behind.
AUDITED_DIST_INFO_DIRNAME = f"gpuwm-{engine_pin.GPUWM_MEASURED_VERSION}.dist-info"

WSM6_256_CAPACITY_PROFILE = "gm1.arwen-wsm6-column-capacity-256/v1"
WSM6_CAPACITY_TIERS = (64, 80, 128, 256)
WSM6_CAPACITY_LEVEL_BOUNDS = (2, 256)
CAPACITY_RECEIPT_NAME = "gm1-wsm6-capacity-overlay.json"

#: The two-file delta applied to the 2.8.9 base.  ``wsm6_constants.py`` is
#: byte-identical in 2.5.8, 2.7.4, 2.8.0 and 2.8.9, so its patched digest is
#: unchanged from the 2.5.8 profile; ``physics_vertical_contract.py`` gained
#: rows for Milbrandt-Yau, WDM6, P3 and New Tiedtke between 2.5.8 and 2.7.4,
#: was byte-identical between 2.7.4 and 2.8.0, and gained four more rows
#: (FAST_SBM, UW PBL, TEMPO and the original Tiedtke) in 2.8.9, so both its
#: base and its patched digest moved at the 2.7.4 and 2.8.9 pins (the 2.8.0
#: patched digest was
#: 3fcf990ed68c09a3b594e2ee35393f2992a2b73ec97c0f6fafbdfa9f7aa72a8c).  WSM6's
#: own row in that file is still ``(2, 80)`` in stock 2.8.9 -- the ceiling
#: this profile lifts is not fixed upstream, so the profile is not retired.
#: The kernel the deeper tiers compile is 2.8.9's wsm6.cu, whose constant
#: divisions changed spelling (see :mod:`cm1core.engine_pin`); the overlay
#: itself still changes no CUDA source byte.
CAPACITY_PATCHED_SURFACE: Mapping[str, str] = MappingProxyType({
    "gpuwm/core/wsm6_constants.py":
        "a1bf764f2a7553bdb148524463ada69e26c855482c7db76cbc2f6374ac08aa2a",
    "gpuwm/physics_vertical_contract.py":
        "ec10320c5a7053994cd3ec6733f2b8c78ff1d4c62af7122ed0c90d911d8fe58c",
})

_PATCHES: Mapping[str, tuple[tuple[bytes, bytes], ...]] = MappingProxyType({
    "gpuwm/core/wsm6_constants.py": (
        (b"at most two WSM6 modules.", b"at most four WSM6 modules."),
        (b"WSM6_DEEP_KMAX = 80\n", b"WSM6_DEEP_KMAX = 256\n"),
        (
            b"WSM6_KERNEL_LEVEL_TIERS = (WSM6_SHALLOW_KMAX, WSM6_DEEP_KMAX)\n",
            b"WSM6_KERNEL_LEVEL_TIERS = (WSM6_SHALLOW_KMAX, 80, 128, WSM6_DEEP_KMAX)\n",
        ),
    ),
    "gpuwm/physics_vertical_contract.py": (
        (
            b"WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)\n",
            b"WSM6_VERTICAL_LEVEL_BOUNDS = (2, 256)\n",
        ),
    ),
})


def _canonical(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def require_capacity_profile(profile: str) -> str:
    if profile != WSM6_256_CAPACITY_PROFILE:
        raise ValueError(
            f"unknown WSM6 capacity_profile={profile!r}; an unpinned capacity "
            "could overrun per-thread column arrays or execute changed physics"
        )
    return profile


def capacity_surface(profile: str) -> dict[str, str]:
    """Return the exact source surface (the WSM6 surface) for the selected build."""
    require_capacity_profile(profile)
    return {**engine_pin.AUDITED_WSM6_SURFACE, **CAPACITY_PATCHED_SURFACE}


def capacity_level_bounds(profile: str | None = None) -> tuple[int, int]:
    if profile is None:
        return (2, 80)
    require_capacity_profile(profile)
    return WSM6_CAPACITY_LEVEL_BOUNDS


def capacity_level_tier(nz: int, *, profile: str = WSM6_256_CAPACITY_PROFILE) -> int:
    require_capacity_profile(profile)
    if isinstance(nz, bool) or not isinstance(nz, int) or not 2 <= nz <= 256:
        raise ValueError("WSM6 capacity profile requires integer 2 <= nz <= 256")
    return next(tier for tier in WSM6_CAPACITY_TIERS if nz <= tier)


def capacity_profile_metadata(profile: str = WSM6_256_CAPACITY_PROFILE) -> dict[str, object]:
    require_capacity_profile(profile)
    return {
        "schema": "gm1.arwen-wsm6-capacity-profile/v1",
        "profile": profile,
        "base_source_identity": engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY,
        "base_version": engine_pin.GPUWM_MEASURED_VERSION,
        "vertical_level_bounds": list(WSM6_CAPACITY_LEVEL_BOUNDS),
        "kernel_level_tiers": list(WSM6_CAPACITY_TIERS),
        "kernel_integer_define": "WSM6_KMAX",
        "public_entry_point": "gpuwm.core.microphysics.apply",
        "patched_surface": {
            name: {"before": engine_pin.AUDITED_WSM6_SURFACE[name], "after": digest}
            for name, digest in sorted(CAPACITY_PATCHED_SURFACE.items())
        },
        "unchanged_surface": {
            name: digest for name, digest in sorted(engine_pin.AUDITED_WSM6_SURFACE.items())
            if name not in CAPACITY_PATCHED_SURFACE
        },
        "authority_decision": "docs/ARWEN-WSM6-CAPACITY-256-AUTHORITY.md",
        "claim_limit": (
            "resource-capacity build and exact dependency-source admission only; "
            "the extended column tiers require independent GPU numerical and "
            "resource measurements before any validation or performance claim"
        ),
    }


def capacity_profile_sha256(profile: str = WSM6_256_CAPACITY_PROFILE) -> str:
    return hashlib.sha256(_canonical(capacity_profile_metadata(profile))).hexdigest()


def patch_capacity_source(relative: str, payload: bytes) -> bytes:
    """Apply the closed capacity-only delta after verifying the complete input."""
    if relative not in _PATCHES:
        raise ValueError(f"no WSM6 capacity patch is admitted for {relative!r}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != engine_pin.AUDITED_WSM6_SURFACE[relative]:
        raise ValueError(f"capacity source {relative} differs from the audited base SHA-256")
    for before, after in _PATCHES[relative]:
        if payload.count(before) != 1:
            raise ValueError(f"capacity patch anchor inventory changed in {relative}")
        payload = payload.replace(before, after)
    if hashlib.sha256(payload).hexdigest() != CAPACITY_PATCHED_SURFACE[relative]:
        raise ValueError(f"capacity patch result SHA-256 changed for {relative}")
    return payload


def _verify_surface(root: Path, surface: Mapping[str, str]) -> None:
    for relative, expected in surface.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"capacity source requires regular file {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"capacity source {relative} differs from its audited SHA-256")


def _source_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for name in ("gpuwm", AUDITED_DIST_INFO_DIRNAME):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"capacity source requires regular directory {name}")
        for path in sorted(directory.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"capacity source refuses symlink {path.relative_to(root)}")
            if path.is_file():
                files.append(path)
    return tuple(files)


def build_wsm6_capacity_overlay(
    source_root: str | Path,
    destination: str | Path,
    *,
    profile: str = WSM6_256_CAPACITY_PROFILE,
) -> dict[str, object]:
    """Atomically create a fresh capacity overlay without importing ArWen/CUDA.

    ``source_root`` is the separately preserved audited 2.8.9 archive/copy.
    The complete package and its version metadata are copied, excluding stale
    Python bytecode.  Existing destinations, missing files and source drift
    refuse; the source tree is never written.
    """
    require_capacity_profile(profile)
    source = Path(source_root)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("capacity source_root must be a regular directory")
    source = source.resolve(strict=True)
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"capacity overlay destination already exists: {target}")
    target = target.resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError("capacity overlay destination must be outside the audited source tree")
    _verify_surface(source, engine_pin.AUDITED_WSM6_SURFACE)
    files = _source_files(source)
    metadata = (source / f"{AUDITED_DIST_INFO_DIRNAME}/METADATA").read_bytes()
    expected_version = f"Version: {engine_pin.GPUWM_MEASURED_VERSION}\n".encode()
    if b"Name: gpuwm\n" not in metadata or expected_version not in metadata:
        raise ValueError(
            "capacity overlay requires the audited gpuwm "
            f"{engine_pin.GPUWM_MEASURED_VERSION} version metadata"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        source_manifest: dict[str, str] = {}
        output_manifest: dict[str, str] = {}
        for path in files:
            relative = path.relative_to(source).as_posix()
            payload = path.read_bytes()
            source_manifest[relative] = hashlib.sha256(payload).hexdigest()
            if relative in _PATCHES:
                payload = patch_capacity_source(relative, payload)
            output_manifest[relative] = hashlib.sha256(payload).hexdigest()
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
        _verify_surface(source, engine_pin.AUDITED_WSM6_SURFACE)
        _verify_surface(staging, capacity_surface(profile))
        changed = sorted(name for name in source_manifest
                         if source_manifest[name] != output_manifest[name])
        if changed != sorted(CAPACITY_PATCHED_SURFACE):
            raise ValueError("capacity overlay changed files outside its exact two-file delta")
        receipt: dict[str, object] = {
            "schema": "gm1.arwen-wsm6-capacity-overlay/v1",
            "status": "BUILT_CAPACITY_OVERLAY_GPU_VALIDATION_PENDING",
            "capacity_profile": capacity_profile_metadata(profile),
            "capacity_profile_sha256": capacity_profile_sha256(profile),
            "audited_surface": capacity_surface(profile),
            "changed_files": changed,
            "copied_file_count": len(output_manifest),
            "source_file_manifest_sha256": hashlib.sha256(_canonical(source_manifest)).hexdigest(),
            "output_file_manifest_sha256": hashlib.sha256(_canonical(output_manifest)).hexdigest(),
            "source_tree_modified": False,
            "runtime_monkeypatch": False,
            "gpu_used": False,
        }
        (staging / CAPACITY_RECEIPT_NAME).write_bytes(_canonical(receipt))
        staging.rename(target)
        return receipt
    except BaseException:
        # Only the directory created by this invocation is eligible for cleanup.
        if staging.exists():
            if staging.resolve().parent != target.parent.resolve():
                raise RuntimeError("capacity overlay staging escaped its owning directory")
            shutil.rmtree(staging)
        raise


__all__ = [
    "AUDITED_DIST_INFO_DIRNAME", "CAPACITY_PATCHED_SURFACE", "CAPACITY_RECEIPT_NAME", "WSM6_256_CAPACITY_PROFILE",
    "WSM6_CAPACITY_LEVEL_BOUNDS", "WSM6_CAPACITY_TIERS", "build_wsm6_capacity_overlay",
    "capacity_level_bounds", "capacity_level_tier", "capacity_profile_metadata",
    "capacity_profile_sha256", "capacity_surface", "patch_capacity_source",
]
