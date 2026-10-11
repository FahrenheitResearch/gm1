"""``gpuwm-cm1 reproduce RUNDIR``: one folder that says exactly what ran (``RUNDIR/reproduce/``, the Results screen's bundle).

The Results screen's "Reproduce bundle" is this command.  It copies the
small files that define and record a run into ``RUNDIR/reproduce/``, writes
what produced them into ``version.json``, the commands that run it again
into ``README.txt``, a ``SHA256SUMS``, and packs the same files into
``RUNDIR/reproduce/NAME-reproduce.tar.gz`` to send to someone:

- the namelist as run and the user's original, the sounding, the coverage
  report, ``settings.toml`` (the values CM1 keeps in Fortran), the generated
  ``deck.toml``, the estimate, a guided run's ``fit.json``;
- ``run.json`` and ``commands.log`` (every command the GUI ran);
- each segment's deck, receipt and ``segment.json``;
- ``version.json``: gpuwm-cm1 (the wheel's RECORD hash, or the checkout's
  commit), gpuwm (ArWen), Python, the CM1 authority, and the native tools
  each segment used.

Frames, restarts and cm1out files are never copied: the bundle is the recipe
and the record, not the output.  Running the command again replaces the
bundle.  Standard library only; it never opens a device.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
from typing import Any, Mapping

from .rundir import RunDirectory, installation, read_json_or_none

REPRODUCE_SCHEMA = "gpuwm-cm1.reproduce-bundle/v1"
BUNDLE_DIR = "reproduce"
#: The run folder's own files the bundle copies when they exist.
RUN_FILES = (
    "namelist.input", "namelist.input.original", "input_sounding", "settings.toml", "deck.toml",
    "import-report.json", "import-report.txt", "estimate.json", "geometry.json", "fit.json", "namelist.diff",
    "run.json", "commands.log",
)
SEGMENT_FILES = ("deck.toml", "receipt.json", "segment.json")
#: A file larger than this is not a recipe (a stray frame someone put in the folder); it is left out and listed.
FILE_LIMIT = 64 * 1024 * 1024


class ReproduceError(ValueError):
    """The folder is not a run directory, or the bundle could not be written; the message says which."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_word(text: str) -> str:
    """``text`` as one safe file-name word: the tarball's name and its top folder never leave the bundle."""

    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip(".-")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpuwm_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("gpuwm")
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _setting_word(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def rerun_commands(run: RunDirectory, record: Mapping[str, Any]) -> list[str]:
    """The commands that make and run the same case again from the bundle's files."""

    from ..cm1import.usersettings import read_file

    new = f"{record.get('name') or run.path.name}-again"
    words = ["gpuwm-cm1", "import", "namelist.input"]
    if run.sounding.is_file():
        words += ["--sounding", "input_sounding"]
    words += ["--out", new]
    acknowledged = {str(item.get("id") if isinstance(item, Mapping) else item)
                    for item in record.get("acknowledgments") or []}
    if "output-keys" in acknowledged:
        words.append("--acknowledge-output-keys")
    lines = [" ".join(shlex.quote(word) for word in words)]
    settings = read_file(run.path / "settings.toml") if (run.path / "settings.toml").is_file() else {}
    edits: list[str] = []
    for key, value in sorted(settings.items()):
        edits += ["--setting", f"{key}={_setting_word(value)}"]
    if "thompson" in acknowledged:
        edits.append("--acknowledge-thompson")
    if edits:
        lines.append(" ".join(shlex.quote(word) for word in ["gpuwm-cm1", "edit", new, *edits]))
    lines.append(f"gpuwm-cm1 run {shlex.quote(new)}")
    return lines


def version_record(run: RunDirectory, record: Mapping[str, Any]) -> dict[str, Any]:
    from .. import CM1_AUTHORITY_SHA256

    segments = []
    for segment in run.segments():
        row = run.segment_record(segment.number) or {}
        receipt = read_json_or_none(segment.receipt) or {}
        segments.append({
            "segment": segment.name,
            "kind": row.get("kind"),
            "gm1": row.get("gm1"),
            "host": row.get("host"),
            "gpu": (row.get("gpu") or {}).get("name") if isinstance(row.get("gpu"), Mapping) else row.get("gpu"),
            "library": row.get("library"),
            "history_publisher": row.get("history_publisher"),
            "receipt_version": receipt.get("version"),
            "config_sha256": receipt.get("config_sha256"),
            "initializer_library_sha256": receipt.get("initializer_library_sha256"),
            "status": receipt.get("status"),
        })
    return {
        "schema": REPRODUCE_SCHEMA,
        "written_utc": _utc(),
        "run": record.get("name") or run.path.name,
        "gpuwm_cm1": installation(),
        "gpuwm": _gpuwm_version(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "cm1_authority_sha256": CM1_AUTHORITY_SHA256,
        "cm1_release": "21.1",
        "created_by": record.get("gm1"),
        "segments": segments,
    }


def _readme(run: RunDirectory, record: Mapping[str, Any], version: Mapping[str, Any], files: list[str],
            skipped: list[str]) -> str:
    install = version["gpuwm_cm1"]
    which = (f"gpuwm-cm1 {install.get('version')} (wheel, RECORD sha256 {install.get('record_sha256')})"
             if install.get("install") == "wheel" else
             f"gpuwm-cm1 {install.get('version')} (source checkout, commit {install.get('git_commit', 'unknown')})")
    lines = [
        f"Reproduce bundle for the run '{version['run']}'",
        f"written {version['written_utc']} from {run.path}",
        "",
        "What ran:",
        f"  {which}",
        f"  gpuwm (ArWen) {version.get('gpuwm') or 'not installed here'}, Python {version['python']}",
        f"  CM1 r21.1 authority {version['cm1_authority_sha256']}",
    ]
    for segment in version["segments"]:
        lines.append(f"  segment {segment['segment']}: {segment.get('kind') or 'run'} on "
                     f"{segment.get('host') or '?'}, {segment.get('gpu') or '?'}; receipt "
                     f"{segment.get('status') or 'not written'}")
    lines += [
        "",
        "Run the same case again, from this folder (the same gpuwm-cm1 as above gives the same words):",
        *[f"  {line}" for line in rerun_commands(run, record)],
        "",
        "The files:",
        *[f"  {name}" for name in files],
    ]
    if skipped:
        lines += ["", "Left out (larger than a recipe file):", *[f"  {name}" for name in skipped]]
    lines += ["", "Frames, restarts and cm1out files are not in the bundle: it is the recipe and the record, "
              "not the output.", ""]
    return "\n".join(lines)


def build_bundle(rundir: Path | str) -> dict[str, Any]:
    """Write ``RUNDIR/reproduce/`` and its tarball; return what was written."""

    run = RunDirectory(rundir)
    if not run.is_run():
        raise ReproduceError(f"{run.path} is not a run directory (it has no run.json)")
    record = run.load()
    name = _file_word(str(record.get("name") or run.path.name)) or _file_word(run.path.name) or "run"
    target = run.path / BUNDLE_DIR
    staging = run.path / f".{BUNDLE_DIR}.partial-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    copied: list[str] = []
    skipped: list[str] = []
    try:
        def copy(source: Path, relative: str) -> None:
            if not source.is_file():
                return
            if source.stat().st_size > FILE_LIMIT:
                skipped.append(relative)
                return
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(relative)

        for filename in RUN_FILES:
            copy(run.path / filename, filename)
        for segment in run.segments():
            for filename in SEGMENT_FILES:
                copy(segment.path / filename, f"segments/{segment.name}/{filename}")
        version = version_record(run, record)
        (staging / "version.json").write_text(json.dumps(version, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        copied.append("version.json")
        (staging / "README.txt").write_text(_readme(run, record, version, ["README.txt", *copied], skipped),
                                            encoding="utf-8")
        copied.insert(0, "README.txt")
        sums = [f"{_sha256(staging / relative)}  {relative}" for relative in sorted(copied)]
        (staging / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        copied.append("SHA256SUMS")
        tarball_name = f"{name}-reproduce.tar.gz"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(copied):
                archive.add(staging / relative, arcname=f"{name}-reproduce/{relative}", recursive=False)
        (staging / tarball_name).write_bytes(buffer.getvalue())
        old = run.path / f".{BUNDLE_DIR}.old-{os.getpid()}"
        if target.exists():
            os.replace(target, old)
        os.replace(staging, target)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    files = [{"name": relative, "bytes": (target / relative).stat().st_size} for relative in sorted(copied)]
    return {
        "schema": REPRODUCE_SCHEMA,
        "ok": True,
        "run": name,
        "folder": os.fspath(target),
        "tarball": os.fspath(target / tarball_name),
        "tarball_name": tarball_name,
        "tarball_bytes": (target / tarball_name).stat().st_size,
        "files": files,
        "skipped": skipped,
        "rerun": rerun_commands(run, record),
        "command": f"gpuwm-cm1 reproduce {shlex.quote(os.fspath(run.path))}",
    }


def add_command(commands: Any) -> None:
    command = commands.add_parser(
        "reproduce",
        help="write RUNDIR/reproduce/: the namelist, sounding, report, settings, deck, receipts and versions of a "
             "run, with the commands that run it again and a .tar.gz of them",
    )
    command.add_argument("rundir", type=Path, help="a run directory (gpuwm-cm1 import, or a GUI run)")
    command.add_argument("--json", action="store_true", help="print what was written as JSON")
    command.set_defaults(handler=reproduce_door)


def reproduce_door(arguments: argparse.Namespace) -> int:
    try:
        summary = build_bundle(arguments.rundir)
    except (ReproduceError, OSError) as error:
        if arguments.json:
            print(json.dumps({"ok": False, "code": "GM1-REPRODUCE", "reason": str(error),
                              "remedy": "name a run directory: a folder with run.json"}, indent=2), file=sys.stderr)
        else:
            print(f"gpuwm-cm1 reproduce: {error}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote {len(summary['files'])} files to {summary['folder']}")
        print(f"  to send: {summary['tarball']} ({summary['tarball_bytes'] / 1024:.0f} KB)")
        print("  run it again:")
        for line in summary["rerun"]:
            print(f"    {line}")
    return 0


__all__ = ["BUNDLE_DIR", "REPRODUCE_SCHEMA", "ReproduceError", "add_command", "build_bundle", "rerun_commands",
           "reproduce_door", "version_record"]
