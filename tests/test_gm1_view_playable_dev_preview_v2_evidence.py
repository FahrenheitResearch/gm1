from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/validation/gm1-view-dev-preview-v2-playable"
FILE_HASHES = {
    "README.md": "91852f91f8662a9c2c03524b89d2a56230151946ae4430e4bca9c9475c932e7c",
    "build-receipt.json": (
        "fecca0aee518191a214f57497c352cebc7798f36d04ab7956ec70c307b832d21"
    ),
    "demo.json": "4ffdaa67358e15e75b61c6ecb6ec004628b04ed85ebec41975e281ff22550874",
    "launch-playable.cmd": (
        "e9f49dfa45b3542799d155e231199a7e0c5f5a46caf286ffa6fa6b16f2620ac0"
    ),
    "package-launch-verification.log": (
        "ed0c4d9ed0d550d7b705559897947bd9d1b38690a2299c5691d396b504b81ccc"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_playable_preview_v2_receipts_and_sources_are_exact() -> None:
    for relative, expected in FILE_HASHES.items():
        assert _sha256(BASE / relative) == expected

    receipt = json.loads((BASE / "build-receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == (
        "PASS_RUNNABLE_PLAYABLE_DEV_PREVIEW_CINEMATIC_SOURCE_MORPHOLOGY_REFUSED"
    )
    assert receipt["gameplay_source_commit"] == "7d62537"
    assert receipt["executable_sha256"] == (
        "63f67e9786c0af0d77daed46a321e0b96977912000d7d9590f23845a46eecf6e"
    )
    assert receipt["validation"]["cargo_test_all_targets"] == "PASS 52/52"
    assert receipt["validation"]["package_runtime_dx12"] == (
        "PASS_FROM_PACKAGED_DIRECTORY_ON_NVIDIA_RTX3080_DX12"
    )
    assert receipt["validation"]["package_runtime_exact_steps"] == [43, 600, 1200]
    reviewed = {
        "app.rs": "1e2fb75 Stabilize GM1 View shell frame loop",
        "input.rs": "5669952 Simplify GM1 View scientific shell",
        "watch.rs": "b2576c2 Admit typed idealized viewer source pairs",
    }
    for relative, expected in receipt["source_sha256"].items():
        assert_recaptured_or_owed(
            ROOT / "rust/gm1-view/src" / relative,
            expected,
            reviewed_by=reviewed.get(relative, "a reviewed viewer commit"),
        )


def test_playable_preview_v2_is_exact_discrete_science_and_honest_preview() -> None:
    demo = json.loads((BASE / "demo.json").read_text(encoding="utf-8"))
    assert demo["status"] == (
        "PLAYABLE_DEV_PREVIEW_SCIENCE_EXACT_CINEMATIC_SOURCE_MORPHOLOGY_REFUSED"
    )
    assert demo["field"] == "cloud_condensate = qc + qi"
    assert demo["field_abi"] == "little-endian binary32 C-order [75,300,300]"
    assert demo["scientific_interpolation"] is False
    assert demo["no_synthetic_meteorology"] is True
    snapshots = {frame["step"]: frame for frame in demo["snapshots"]}
    assert set(snapshots) == {43, 600, 1200}
    assert all(frame["source_gate"].startswith("REFUSED_") for frame in snapshots.values())
    assert snapshots[43]["payload_sha256"] == (
        "68743a9458742ad9a0345fd32a989526e5231ab00e82457857110dfa43c8dcf4"
    )
    assert snapshots[600]["payload_sha256"] == (
        "a7eeb6d2a480ae04b0b951cd87d4833b8f53e7a51dbab882b0c6e9d8a6251e47"
    )
    assert snapshots[1200]["payload_sha256"] == (
        "62e072798e6c5f7c988a46af5617611588caca413cbf360797daaa1fecf3eacd"
    )
    claims = "\n".join(
        (BASE / name).read_text(encoding="utf-8")
        for name in ("README.md", "demo.json", "build-receipt.json")
    ).casefold()
    assert "tornado" not in claims
    assert "developer preview" in claims
    assert "display-only" in claims
