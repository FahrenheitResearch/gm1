from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/validation/gm1-view-source-morphology-selection-v1"
RECEIPT_SHA256 = "b5581fda9ba4484865549bb4936e8a3977632b22fdec7fa468dacf54c673657d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt() -> dict[str, object]:
    path = BASE / "receipt.json"
    assert _sha256(path) == RECEIPT_SHA256
    return json.loads(path.read_text(encoding="utf-8"))


def test_three_exact_packs_remain_source_morphology_refusals() -> None:
    receipt = _receipt()
    assert receipt["status"] == "REFUSED_AVAILABLE_EXACT_PACK_SOURCE_MORPHOLOGY"
    assert receipt["field"] == {
        "id": "cloud_condensate",
        "formula": "q3d[qc]+q3d[qi]",
        "shape_zyx": [75, 300, 300],
        "storage": "C-order little-endian binary32",
        "values_unchanged": True,
    }
    packs = {pack["step"]: pack for pack in receipt["packs"]}
    assert set(packs) == {43, 600, 1200}
    assert packs[43]["payload_sha256"] == (
        "68743a9458742ad9a0345fd32a989526e5231ab00e82457857110dfa43c8dcf4"
    )
    assert packs[600]["payload_sha256"] == (
        "a7eeb6d2a480ae04b0b951cd87d4833b8f53e7a51dbab882b0c6e9d8a6251e47"
    )
    assert packs[1200]["payload_sha256"] == (
        "62e072798e6c5f7c988a46af5617611588caca413cbf360797daaa1fecf3eacd"
    )
    assert packs[43]["dominant_component"]["mass_fraction"] > 0.99998
    assert packs[600]["dominant_component"]["mass_fraction"] < 0.35
    assert packs[1200]["dominant_component"]["vertical_to_horizontal_ratio"] < 0.25
    assert all(str(pack["status"]).startswith("REFUSED_") for pack in packs.values())


def test_mass_center_correction_and_current_gpu_gate_are_sealed() -> None:
    receipt = _receipt()
    correction = receipt["mass_center_correction"]
    assert correction["status"] == "PASS_CORRECTED_EXACT_CELL_CENTER_MAPPING"
    assert "REFUSED_MASS_CENTER_HALF_CELL_MAPPING" in correction[
        "old_render_disposition"
    ]
    assert correction["roundtrip"]["vertical_exact_sigma_mass_levels_tested"] == 75
    assert correction["roundtrip"]["wgsl_parse_gate"] == "PASS"
    assert correction["qualitative_reversal"] is False

    gate = receipt["corrected_gpu_gate"]
    assert gate["status"] == (
        "PASS_MAPPING_AND_PAIRED_MASKS_REFUSED_CINEMATIC_MORPHOLOGY"
    )
    assert gate["backend"] == "Dx12"
    assert gate["dense_q99_mask_pixels"] == 19739
    assert gate["onset_mask_pixels"] == 517778
    assert gate["terrain_mask_pixels"] == 470020
    for descriptor in (gate["png"], gate["log"]):
        assert _sha256(BASE / descriptor["file"]) == descriptor["sha256"]
    png = (BASE / gate["png"]["file"]).read_bytes()
    assert png[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert struct.unpack(">II", png[16:24]) == (1920, 1080)


def test_renderer_and_historical_refusal_hashes_are_exact() -> None:
    receipt = _receipt()
    source_paths = {
        "app_rs": "rust/gm1-view/src/app.rs",
        "volume_rs": "rust/gm1-view/src/volume.rs",
        "gm1_volume_wgsl": "rust/gm1-view/src/gm1_volume.wgsl",
        "physical_atmosphere_rs": "rust/gm1-view/src/physical_atmosphere.rs",
        "physical_atmosphere_wgsl": "rust/gm1-view/src/physical_atmosphere.wgsl",
        "she_admitted_terrain_rs": "rust/gm1-view/src/she_admitted_terrain.rs",
    }
    reviewed = {
        "app_rs": "1e2fb75 Stabilize GM1 View shell frame loop",
        "volume_rs": "c77acce Add strict production renderer residency",
    }
    for key, relative in source_paths.items():
        assert_recaptured_or_owed(
            ROOT / relative,
            receipt["renderer_sources"][key],
            reviewed_by=reviewed.get(key, "a reviewed renderer commit"),
        )
    for descriptor in receipt["historical_pre_correction_refusals"]:
        assert _sha256(BASE / descriptor["file"]) == descriptor["sha256"]
    claims = json.dumps(receipt, sort_keys=True).casefold()
    assert "tornado" not in claims
    assert "synthesized" in claims
