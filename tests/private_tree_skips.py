"""Skips for tests that read files the public source snapshot leaves out.

The development repository keeps its measurement receipts (``evidence/``),
the CM1 differential oracle (``tools/oracle/``), most of ``tools/`` and the
working documents under ``docs/`` private.  The public snapshot is the same
source without them, so a test that reads one of those files there fails with
``FileNotFoundError`` or a missing-module ``ImportError``, neither of which is
a defect of the package.

What the snapshot left out is not guessed.  ``tools/release/stage_snapshot.py``
writes ``tests/data/snapshot-omitted.json`` into the snapshot: a salted digest
of every repository path it left out, and of every folder it left out whole.
The manifest carries digests, not names, so the public tree does not list the
private paths.  In the public snapshot (no ``evidence/`` directory at the
repository root) a failure is reported as a skip only when the path or module
its exception names is one of those:

* ``FileNotFoundError`` (or an ``OSError`` with ``ENOENT``): its ``filename``
  lies under the repository, is absent, and it or a folder above it is in the
  manifest;
* ``ModuleNotFoundError`` and ``from tools import x``: the module's file, as a
  top-level module or under ``tests/``, ``tools/`` or ``tools/oracle/`` (the
  folders tests put on ``sys.path``), is in the manifest itself.

Everything else fails as it would anywhere: a missing third-party dependency,
a misspelt module, a shipped file the snapshot lost by mistake, and every
``AssertionError``.  Nothing is decided from the report's text, and without
the manifest nothing is skipped at all.  In the development repository, where
``evidence/`` is present, nothing here changes any outcome.

Collection errors are the one place pytest keeps only text: it re-raises an
import failure as its own error carrying the formatted traceback.  There the
final exception line pytest writes (``E   ModuleNotFoundError: ...`` or
``E   FileNotFoundError: ...``) is parsed, and only that line.

Register from ``tests/conftest.py`` with::

    from private_tree_skips import (  # noqa: F401,E402
        pytest_make_collect_report, pytest_runtest_makereport, pytest_runtest_setup,
    )
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
import errno
import hashlib
import json
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The public snapshot is recognised by the absence of evidence/.
PUBLIC_SNAPSHOT = not (ROOT / "evidence").is_dir()

#: Written by tools/release/stage_snapshot.py into the public snapshot only.
OMITTED_MANIFEST = ROOT / "tests" / "data" / "snapshot-omitted.json"
OMITTED_SCHEMA = "gpuwm-cm1.snapshot-omitted/v1"
_SALT = b"gpuwm-cm1 snapshot-omitted v1\0"

#: The folders a test puts on ``sys.path`` to import a repository-local helper
#: ("" is the repository root, for ``tools.*``).
LOCAL_MODULE_FOLDERS = ("", "tests", "tools", "tools/oracle")


def path_digest(relative: str) -> str:
    """The manifest's digest of one repository path (POSIX, relative)."""

    return hashlib.sha256(_SALT + relative.encode("utf-8")).hexdigest()[:16]


def omitted_digests(paths: Iterable[str]) -> list[str]:
    """The manifest rows for ``paths`` (the stage's side of ``path_digest``)."""

    return sorted({path_digest(path) for path in paths})


def load_omitted(manifest: Path = OMITTED_MANIFEST) -> frozenset[str]:
    """The digests the stage recorded, or none when the manifest is absent or not v1."""

    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    if not isinstance(record, dict) or record.get("schema") != OMITTED_SCHEMA:
        return frozenset()
    rows = record.get("omitted")
    if not isinstance(rows, list):
        return frozenset()
    return frozenset(row for row in rows if isinstance(row, str))


OMITTED: frozenset[str] = load_omitted() if PUBLIC_SNAPSHOT else frozenset()


def _table(table: Collection[str] | None) -> Collection[str]:
    return OMITTED if table is None else table


def was_omitted(relative: str, table: Collection[str] | None = None) -> bool:
    """True when ``relative``, or a folder above it, was left out of the snapshot."""

    digests = _table(table)
    parts = relative.split("/")
    return any(path_digest("/".join(parts[:end])) in digests for end in range(1, len(parts) + 1))


def private_path(filename: object, table: Collection[str] | None = None) -> str | None:
    """``filename`` relative to the repository if the snapshot left it out and it is absent."""

    if filename is None:
        return None
    try:
        path = Path(os.fsdecode(filename))
    except TypeError:
        return None
    candidates = [path] if path.is_absolute() else [ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        try:
            relative = Path(os.path.normpath(candidate)).relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if relative == "." or (ROOT / relative).exists():
            continue
        if was_omitted(relative, table):
            return relative
    return None


def local_module(name: str | None, table: Collection[str] | None = None) -> str | None:
    """The left-out file whose absence makes ``import name`` fail, if there is one."""

    if not name:
        return None
    digests = _table(table)
    dotted = "/".join(name.split("."))
    for folder in LOCAL_MODULE_FOLDERS:
        base = f"{folder}/{dotted}" if folder else dotted
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if path_digest(candidate) in digests and not (ROOT / candidate).exists():
                return candidate
    return None


_CANNOT_IMPORT = re.compile(r"^cannot import name '(\w+)' from 'tools'")


def missing_private(exc: BaseException | None, table: Collection[str] | None = None) -> str | None:
    """The left-out path or module whose absence raised ``exc``, or None.

    Only ``FileNotFoundError`` (or an ``OSError`` with ``ENOENT``) and
    ``ImportError`` count; every other exception, an ``AssertionError`` above
    all, is a real failure.
    """

    if isinstance(exc, ModuleNotFoundError):
        return local_module(exc.name, table)
    if isinstance(exc, ImportError):
        # ``from tools import x`` with tools/x.py left out.
        match = _CANNOT_IMPORT.match(str(exc))
        if match and exc.name == "tools":
            return local_module(f"tools.{match.group(1)}", table)
        return None
    if isinstance(exc, OSError) and (isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT):
        return private_path(exc.filename, table) or private_path(exc.filename2, table)
    return None


#: An exception line as pytest writes it (``E   SomeError: message``).
_EXCEPTION_LINE = re.compile(r"^E\s+([\w.]+): (.*)$", re.M)
_NO_MODULE = re.compile(r"^No module named '([\w.]+)'$")
_NO_FILE = re.compile(r"^\[Errno 2\] No such file or directory: '([^']+)'$")


def missing_private_in_collection(longrepr: str, table: Collection[str] | None = None) -> str | None:
    """``missing_private`` for a collection error, from pytest's final exception line only."""

    lines = [(kind.rsplit(".", 1)[-1], message) for kind, message in _EXCEPTION_LINE.findall(longrepr)
             if kind.rsplit(".", 1)[-1].endswith(("Error", "Exception"))]
    if not lines:
        return None
    # The exception that ended collection is the last one written.
    kind, message = lines[-1]
    message = message.strip()
    if kind not in {"ModuleNotFoundError", "ImportError", "FileNotFoundError"}:
        return None
    if kind == "ModuleNotFoundError":
        match = _NO_MODULE.match(message)
        return local_module(match.group(1), table) if match else None
    if kind == "ImportError":
        match = _CANNOT_IMPORT.match(message)
        return local_module(f"tools.{match.group(1)}", table) if match else None
    match = _NO_FILE.match(message)
    return private_path(match.group(1), table) if match else None


def _reason(missing: str) -> str:
    return f"needs {missing}, which stays in the development repository (absent from the public source snapshot)"


#: Tests whose failure without a left-out file does not raise one of the
#: exceptions above (a glob that finds nothing, a subprocess whose output is
#: captured, a search that comes back empty), skipped up front in the public
#: snapshot only, and only when the manifest says the file was left out.
KNOWN_PRIVATE_READERS = {
    "tests/test_gui_mock_static.py::test_the_ten_screens_exist": "src/cm1core/gui/static/mock/",
    "tests/test_gm1_historical_closure_impact.py::test_negative_control_is_preserved_not_relabelled": "evidence/",
    "tests/test_run_gm1_child_initialization.py::test_runner_import_is_cuda_cold": (
        "tools/run_gm1_child_initialization.py"
    ),
    # Subprocesses of a development tool: the failure is the child's exit status.
    "tests/test_arwen_chain_wrf.py::test_the_prepare_tool_reads_wrf_histories": "tools/arwen_chain_prepare.py",
    "tests/test_background_publication.py::test_the_publication_benchmark_times_a_restart_alone_and_on_a_frame_step": (
        "tools/benchmark_publication_io.py"
    ),
    "tests/test_presets.py::test_the_preset_files_are_what_the_generator_writes_from_cm1s_own_namelists": (
        "tools/presets/make_presets.py"
    ),
    "tests/test_presets_gallery.py::test_the_generator_matches_the_preset_files_and_every_change_has_its_reason": (
        "tools/presets/make_presets.py"
    ),
    "tests/test_presets_gallery.py::test_every_shipped_sounding_is_rederived_from_its_stored_response": (
        "tools/release/uwyo_to_input_sounding.py"
    ),
    "tests/test_quicklook.py::test_the_stock_series_record_is_current": "tools/generate_stock_supercell_series.py",
    "tests/test_initiation_forcing.py::test_the_host_forms_are_cm1s_statements_word_for_word": (
        "tools/oracle/initiation_forcing_oracle.py"
    ),
    "tests/test_runtime_profiler_failsafe.py::test_door_launch_counts_lose_nothing_to_disabled_call_sites": (
        "tools/oracle/profiler_call_site_identity.py"
    ),
    # The step-speed probe is a development tool loaded into the child process.
    "tests/test_step_speed_identity.py::test_a_probe_that_never_reached_cupy_says_so_and_the_report_fails_it": (
        "tools/perf_probe/"
    ),
    "tests/test_step_speed_identity.py::test_the_probe_announces_itself_and_times_only_the_dynamics_frees": (
        "tools/perf_probe/"
    ),
    "tests/test_step_speed_identity.py::test_the_probe_keeps_or_poisons_as_asked_and_refuses_an_unknown_mode": (
        "tools/perf_probe/"
    ),
    "tests/test_step_speed_identity.py::test_the_probe_still_patches_cupy_after_the_doors_find_spec_lookup": (
        "tools/perf_probe/"
    ),
    # The speed measurement reads a sealed estimator receipt in a child process.
    "tests/test_cm1run_estimate.py::test_measure_speed_gives_the_harness_the_budget_its_own_peak_needs": "evidence/",
    # is_file() assertions over evidence paths.
    "tests/test_evidence_integrity.py::test_authority_evidence_hashes_are_live": (
        "evidence/authority/authority-audit.md"
    ),
    "tests/test_physics_support.py::test_the_admitted_kessler_slice_cites_evidence_that_exists": "evidence/",
}


def pytest_runtest_setup(item) -> None:  # noqa: ANN001
    if not PUBLIC_SNAPSHOT:
        return
    missing = KNOWN_PRIVATE_READERS.get(f"tests/{item.path.name}::{item.originalname}")
    if missing is not None and not (ROOT / missing).exists() and was_omitted(missing.rstrip("/")):
        pytest.skip(_reason(missing))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ANN001, ANN201
    outcome = yield
    if not PUBLIC_SNAPSHOT:
        return
    report = outcome.get_result()
    if not report.failed or report.when not in {"setup", "call"} or call.excinfo is None:
        return
    missing = missing_private(call.excinfo.value)
    if missing is None:
        return
    path, line, _ = item.location
    report.outcome = "skipped"
    report.longrepr = (str(ROOT / path), (line or 0) + 1, "Skipped: " + _reason(missing))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):  # noqa: ANN001, ANN201
    outcome = yield
    if not PUBLIC_SNAPSHOT:
        return
    report = outcome.get_result()
    if not report.failed:
        return
    missing = missing_private_in_collection(str(report.longrepr))
    if missing is None:
        return
    report.outcome = "skipped"
    report.longrepr = (str(collector.path), 0, "Skipped: " + _reason(missing))
    report.result = []
