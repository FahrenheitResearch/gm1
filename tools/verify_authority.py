#!/usr/bin/env python3
"""Verify a locally supplied CM1 21.1 authority artifact/extraction.

    python tools/verify_authority.py cm1r21.1.tar.gz [--extracted DIR]

``--extracted`` checks every file of an extraction against the content
manifest, by default tools/authority/cm1r21.1-content-sha256.tsv beside this
tool (sha256, bytes and path of each file of the release tarball).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AUTHORITY_SHA256 = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
AUTHORITY_BYTES = 345_874_148
#: The release tarball's per-file content manifest, shipped beside this tool.
CONTENT_MANIFEST = Path(__file__).resolve().parent / "authority" / "cm1r21.1-content-sha256.tsv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected sha256<TAB>bytes<TAB>path")
        digest, size, relative = fields
        rows.append((digest, int(size), relative))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--extracted", type=Path)
    parser.add_argument(
        "--content-manifest",
        type=Path,
        default=CONTENT_MANIFEST,
        help="sha256<TAB>bytes<TAB>path per file (default: %(default)s)",
    )
    arguments = parser.parse_args()
    artifact = arguments.artifact.expanduser().resolve(strict=True)
    artifact_result = {
        "bytes": artifact.stat().st_size,
        "expected_bytes": AUTHORITY_BYTES,
        "expected_sha256": AUTHORITY_SHA256,
        "path": str(artifact),
        "sha256": sha256_file(artifact),
    }
    artifact_result["passed"] = (
        artifact_result["bytes"] == AUTHORITY_BYTES
        and artifact_result["sha256"] == AUTHORITY_SHA256
    )
    report: dict[str, object] = {"artifact": artifact_result}
    passed = bool(artifact_result["passed"])
    if arguments.extracted is not None:
        root = arguments.extracted.expanduser().resolve(strict=True)
        if not arguments.content_manifest.expanduser().is_file():
            parser.error(f"no content manifest at {arguments.content_manifest}; pass --content-manifest")
        manifest = arguments.content_manifest.expanduser().resolve(strict=True)
        moved: list[str] = []
        absent: list[str] = []
        checked = 0
        expected_paths: set[str] = set()
        for expected_hash, expected_size, relative in load_manifest(manifest):
            expected_paths.add(relative.replace("\\", "/"))
            target = root.joinpath(*Path(relative).parts)
            if not target.is_file():
                absent.append(relative)
                continue
            checked += 1
            if target.stat().st_size != expected_size or sha256_file(target) != expected_hash:
                moved.append(relative)
        extra = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in expected_paths
        )
        extraction = {
            "absent": sorted(absent),
            "checked": checked,
            "extra": extra,
            "manifest": str(manifest),
            "moved": sorted(moved),
            "root": str(root),
        }
        extraction["passed"] = not absent and not moved and not extra
        report["extraction"] = extraction
        passed = passed and bool(extraction["passed"])
    report["passed"] = passed
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
