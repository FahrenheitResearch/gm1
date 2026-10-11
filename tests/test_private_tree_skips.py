"""tests/private_tree_skips.py: what counts as a left-out-file failure, and when."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

import private_tree_skips as skips

ROOT = Path(__file__).resolve().parents[1]

#: What a stage might have left out: a private tree, a private tool, a test
#: helper the denylist dropped, a tools/oracle module.
LEFT_OUT = (
    "evidence",
    "evidence/zz-absent/receipt.json",
    "docs/zz-absent",
    "docs/zz-absent/plan.md",
    "tools/zz_absent.py",
    "tests/zz_absent_helper.py",
    "tools/oracle",
    "tools/oracle/zz_oracle.py",
)
TABLE = frozenset(skips.omitted_digests(LEFT_OUT))


def _not_found(path: str) -> FileNotFoundError:
    return FileNotFoundError(2, "No such file or directory", path)


def _no_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named '{name}'", name=name)


def test_the_public_snapshot_is_recognised_by_the_absence_of_evidence() -> None:
    assert skips.PUBLIC_SNAPSHOT is (not (ROOT / "evidence").is_dir())


def test_the_manifest_carries_digests_and_is_ignored_unless_it_is_v1(tmp_path: Path) -> None:
    manifest = tmp_path / "snapshot-omitted.json"
    assert skips.load_omitted(manifest) == frozenset()
    manifest.write_text(json.dumps({"schema": "other/v1", "omitted": list(TABLE)}), encoding="utf-8")
    assert skips.load_omitted(manifest) == frozenset()
    manifest.write_text(json.dumps({"schema": skips.OMITTED_SCHEMA, "omitted": sorted(TABLE)}), encoding="utf-8")
    assert skips.load_omitted(manifest) == TABLE
    assert all(len(row) == 16 and "/" not in row for row in TABLE)


def test_a_left_out_path_is_named() -> None:
    absent = f"{ROOT.as_posix()}/evidence/zz-absent/receipt.json"
    assert skips.missing_private(_not_found(absent), TABLE) == "evidence/zz-absent/receipt.json"
    # A folder left out whole covers files the stage never saw.
    assert skips.missing_private(_not_found("evidence/zz-new/receipt.json"), TABLE) == "evidence/zz-new/receipt.json"
    assert skips.missing_private(_not_found("docs/zz-absent/plan.md"), TABLE) == "docs/zz-absent/plan.md"


def test_a_shipped_or_unknown_path_is_a_failure() -> None:
    # A shipped document or tool file the snapshot lost by mistake is not in the
    # manifest, so its absence fails, whatever tree it is in.
    for relative in ("docs/RELEASE-NOTES-zz.md", "tools/authority/zz-content-sha256.tsv",
                     "src/cm1core/zz_absent.py", "tests/data/zz-absent.json"):
        assert skips.missing_private(_not_found(f"{ROOT.as_posix()}/{relative}"), TABLE) is None, relative
    assert skips.missing_private(_not_found(f"{ROOT.as_posix()}/tools/__init__.py"), TABLE) is None
    assert skips.missing_private(_not_found("/tmp/zz-absent/evidence/receipt.json"), TABLE) is None
    # Without a manifest nothing is a left-out file.
    assert skips.missing_private(_not_found("evidence/zz-absent/receipt.json"), frozenset()) is None


def test_only_left_out_modules_are_skips() -> None:
    assert skips.missing_private(_no_module("tools.zz_absent"), TABLE) == "tools/zz_absent.py"
    named = ImportError("cannot import name 'zz_absent' from 'tools' (/x/tools/__init__.py)", name="tools")
    assert skips.missing_private(named, TABLE) == "tools/zz_absent.py"
    assert skips.missing_private(_no_module("zz_absent_helper"), TABLE) == "tests/zz_absent_helper.py"
    assert skips.missing_private(_no_module("zz_oracle"), TABLE) == "tools/oracle/zz_oracle.py"
    # A missing dependency, a misspelt module and a module under a left-out
    # folder that was never a file there are failures.
    for name in ("numpy", "gpuwm", "cupy", "cm1core", "cm1core.zz_absent", "zarr", "cm1core_typo",
                 "tools.zz_other", "zz_unknown_helper"):
        assert skips.missing_private(_no_module(name), TABLE) is None, name
    other = ImportError("cannot import name 'zz_other' from 'tools' (/x/tools/__init__.py)", name="tools")
    assert skips.missing_private(other, TABLE) is None


def test_a_failing_assertion_is_never_a_skip_whatever_it_mentions() -> None:
    # The report text of such a failure names private paths and modules (the
    # test's source is in it); only the exception counts.
    absent = f"{ROOT.as_posix()}/evidence/zz-absent/receipt.json"
    assert skips.missing_private(AssertionError(f"open({absent!r}) differs: 1.0 != 1.5"), TABLE) is None
    assert skips.missing_private(AssertionError("No module named 'tools.zz_absent'"), TABLE) is None
    assert skips.missing_private(ValueError(f"cannot read {absent}"), TABLE) is None
    assert skips.missing_private(None, TABLE) is None


def test_collection_errors_are_read_from_the_final_exception_line_only() -> None:
    source_mention = (
        "    data = json.loads((ROOT / 'evidence/zz-absent/receipt.json').read_text())\n"
        "E   AssertionError: assert 1.0 == 1.5\n"
    )
    assert skips.missing_private_in_collection(source_mention, TABLE) is None
    module = "ImportError while importing test module\nE   ModuleNotFoundError: No module named 'tools.zz_absent'\n"
    assert skips.missing_private_in_collection(module, TABLE) == "tools/zz_absent.py"
    dependency = "ImportError while importing test module\nE   ModuleNotFoundError: No module named 'zarr'\n"
    assert skips.missing_private_in_collection(dependency, TABLE) is None
    absent = f"{ROOT.as_posix()}/evidence/zz-absent/receipt.json"
    chained = (
        f"E   FileNotFoundError: [Errno 2] No such file or directory: '{absent}'\n"
        "During handling of the above exception, another exception occurred:\n"
        "E   RuntimeError: the receipt is not readable\n"
    )
    assert skips.missing_private_in_collection(chained, TABLE) is None
    direct = f"E   FileNotFoundError: [Errno 2] No such file or directory: '{absent}'\n"
    assert skips.missing_private_in_collection(direct, TABLE) == "evidence/zz-absent/receipt.json"
    shipped = f"E   FileNotFoundError: [Errno 2] No such file or directory: '{ROOT.as_posix()}/docs/zz.md'\n"
    assert skips.missing_private_in_collection(shipped, TABLE) is None


def test_every_known_private_reader_names_a_test_that_exists() -> None:
    for node, _missing in skips.KNOWN_PRIVATE_READERS.items():
        path, name = node.split("::")
        assert f"def {name}(" in (ROOT / path).read_text(encoding="utf-8"), node


def test_every_known_private_reader_reads_a_file_the_stage_leaves_out() -> None:
    stage_tool = ROOT / "tools" / "release" / "stage_snapshot.py"
    if not stage_tool.is_file():
        pytest.skip("the stage tool stays in the development repository")
    stage = runpy.run_path(str(stage_tool))
    for node, missing in skips.KNOWN_PRIVATE_READERS.items():
        relative = missing.rstrip("/")
        assert not stage["allowed"](relative) or stage["excluded"](relative) or stage["excluded"](relative + "/x"), node


def test_the_stage_writes_the_digests_this_module_reads() -> None:
    stage_tool = ROOT / "tools" / "release" / "stage_snapshot.py"
    if not stage_tool.is_file():
        pytest.skip("the stage tool stays in the development repository")
    stage = runpy.run_path(str(stage_tool))
    for relative in LEFT_OUT + ("tests/host_portability.py",):
        assert stage["path_digest"](relative) == skips.path_digest(relative), relative
    assert stage["OMITTED_MANIFEST"] == skips.OMITTED_MANIFEST.relative_to(ROOT).as_posix()
    kept = {"tests/test_a.py", "docs/RELEASE-NOTES.md", "tools/__init__.py"}
    every = sorted(kept | {"docs/private/plan.md", "evidence/a/b.json", "tools/private.py", "tests/helper.py"})
    omitted = stage["omitted_paths"](every, kept)
    assert {"docs/private/plan.md", "docs/private", "evidence", "evidence/a", "evidence/a/b.json",
            "tools/private.py", "tests/helper.py"} <= omitted
    assert not omitted & (kept | {"docs", "tools", "tests"})


def test_the_stage_leaves_out_a_test_that_needs_a_left_out_module_when_loaded() -> None:
    stage_tool = ROOT / "tools" / "release" / "stage_snapshot.py"
    if not stage_tool.is_file():
        pytest.skip("the stage tool stays in the development repository")
    stage = runpy.run_path(str(stage_tool))
    source = (
        b"import os\n"
        b"from host_portability import child_environment\n"
        b"import two_part_terms as terms\n"
        b"from tools.private_tool import run\n"
        b"try:\n    import optional_oracle\nexcept ImportError:\n    optional_oracle = None\n"
        b"def test_x():\n    import late_oracle\n"
    )
    every = {"tests/host_portability.py", "tools/oracle/two_part_terms.py", "tools/private_tool.py",
             "tools/oracle/optional_oracle.py", "tools/oracle/late_oracle.py", "tools/__init__.py"}
    chosen = {"tools/__init__.py"}
    assert stage["imported_left_out"](source, every, chosen) == [
        "tests/host_portability.py", "tools/oracle/two_part_terms.py", "tools/private_tool.py"]
    chosen |= {"tests/host_portability.py", "tools/oracle/two_part_terms.py", "tools/private_tool.py"}
    assert stage["imported_left_out"](source, every, chosen) == []
