from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def test_authority_fetch_constants_match_installed_record() -> None:
    fetch = runpy.run_path(str(ROOT / "tools" / "authority" / "fetch_cm1_21_1.py"))
    authority = json.loads(
        (ROOT / "src" / "cm1core" / "records" / "CM1-AUTHORITY.json").read_text(
            encoding="utf-8"
        )
    )["mathematical_authority"]
    assert fetch["AUTHORITY_URL"] == authority["url"]
    assert fetch["AUTHORITY_BYTES"] == authority["bytes"]
    assert fetch["AUTHORITY_SHA256"] == authority["sha256"]


def test_artifact_inspector_rejects_checkout_and_private_material() -> None:
    inspector = runpy.run_path(str(ROOT / "tools" / "inspect_artifacts.py"))
    member_problems = inspector["member_problems"]
    content_problems = inspector["content_problems"]
    assert member_problems(["gpuwm_cm1-0/upstream/cm1.exe"])
    assert member_problems(["gpuwm_cm1-0/evidence/model.nc"])
    assert not member_problems(
        ["cm1core/bin/cm1-state.exe"],
        allowed_native={"cm1core/bin/cm1-state.exe"},
    )
    assert member_problems(
        ["cm1core/bin/unadmitted.exe"],
        allowed_native={"cm1core/bin/cm1-state.exe"},
    )
    # A made-up Windows profile path, built here so this file carries none.
    profile = "\\".join(("C:", "Users", "someone", "checkout"))
    assert content_problems({"receipt.txt": f"path={profile}".encode()})
    assert not member_problems(["gpuwm_cm1-0/evidence/native-reference/build.patch"])
    assert not content_problems({"record.md": b"no local paths or credentials"})
    inspector_source = (ROOT / "tools" / "inspect_artifacts.py").read_bytes()
    assert not content_problems({"tools/inspect_artifacts.py": inspector_source})


def test_a_manylinux_tag_carries_the_sealed_linux_payload() -> None:
    inspector = runpy.run_path(str(ROOT / "tools" / "inspect_artifacts.py"))
    carries = inspector["wheel_tag_carries"]
    floor = inspector["manylinux_floor"]
    assert carries("linux_x86_64", "linux_x86_64")
    assert carries("linux_x86_64", "manylinux_2_28_x86_64")
    assert carries("linux_x86_64", "manylinux_2_27_x86_64.manylinux_2_28_x86_64")
    assert not carries("linux_x86_64", "manylinux_2_28_aarch64")
    assert not carries("linux_x86_64", "win_amd64")
    assert not carries(None, "manylinux_2_28_x86_64")
    assert floor("manylinux_2_28_x86_64") == (2, 28)
    assert floor("manylinux_2_27_x86_64.manylinux_2_28_x86_64") == (2, 27)
    assert floor("linux_x86_64") is None


def test_the_release_scripts_are_in_the_sdist() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    for name in ("build_manylinux.sh", "smoke_cm1_user.sh", "cm1_user_script.py"):
        assert f"include tools/release/{name}" in manifest


def test_verify_authority_defaults_to_the_manifest_shipped_beside_it() -> None:
    verify = runpy.run_path(str(ROOT / "tools" / "verify_authority.py"))
    manifest = verify["CONTENT_MANIFEST"]
    assert manifest == ROOT / "tools" / "authority" / "cm1r21.1-content-sha256.tsv"
    rows = verify["load_manifest"](manifest)
    paths = {relative for _, _, relative in rows}
    assert len(rows) == len(paths) > 100
    assert {"README", "run/LANDUSE.TBL", "src/solve2.F"} <= paths
    # The development repository's authority evidence holds the same manifest.
    evidence = ROOT / "evidence" / "authority" / "cm1r21.1-content-sha256.tsv"
    if evidence.is_file():
        assert evidence.read_bytes() == manifest.read_bytes()
    sdist = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert "include tools/authority/cm1r21.1-content-sha256.tsv" in sdist
