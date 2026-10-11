#!/usr/bin/env python3
"""Fetch the pinned CM1 21.1 release artifact and verify it before admission."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


AUTHORITY_URL = "https://www2.mmm.ucar.edu/people/bryan/cm1/cm1r21.1.tar.gz"
AUTHORITY_SHA256 = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
AUTHORITY_BYTES = 345_874_148


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_authority(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == AUTHORITY_BYTES
        and sha256_file(path) == AUTHORITY_SHA256
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the exact pinned CM1 21.1 tarball; never extract or patch it."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("upstream/cm1-21.1/cm1r21.1.tar.gz"),
    )
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if is_authority(output):
            print(f"verified existing authority: {output}")
            return 0
        parser.error(f"refusing to overwrite non-authority file: {output}")

    request = Request(AUTHORITY_URL, headers={"User-Agent": "gpuwm-cm1-authority-fetch/1"})
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".cm1r21.1-", suffix=".partial", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path = Path(temporary_name)
        if not is_authority(temporary_path):
            raise RuntimeError(
                "downloaded artifact failed the pinned byte-count/SHA-256 admission gate"
            )
        temporary_path.replace(output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    print(f"downloaded and verified authority: {output}")
    print(f"sha256: {AUTHORITY_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
