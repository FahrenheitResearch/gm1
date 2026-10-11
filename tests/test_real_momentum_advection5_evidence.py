from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from host_portability import recorded_name


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/real-momentum-advection5-v1"
RECEIPT = EVIDENCE / "full-domain-sm86.json"
EXPECTED_OUTPUTS = {
    "divx_sha256": "ac1002d10854b44ef5a23f918a85e0dd455991fcb2d5ce8ef87e009071d566e7",
    "rru_sha256": "ad66f15db590eda0bac40e3f4071a099d5b02a838610308449622558258b06b3",
    "rrv_sha256": "57c6e430bdb1603a19eb62540aa0c81954b79a8c9e016b3bce2448e888abdd3c",
    "rrw_sha256": "04d4860710c649ff4e8eb1bbbff18cc52da1037a703ac95bc56cf16d4b6a258e",
    "uten_sha256": "c22749fa1884c8215b25cde790d57d71784bf74d27eb97ae1cde4f3450089bef",
    "vten_sha256": "ca093f95258221abdcb17e4422fd73fe107815f0982a696d0ace08877d04e0d3",
    "wten_sha256": "ec449054c6efc121726444c5f3e9cd3bc628c65998793384895938179e17c3db",
}
EXPECTED_FIGURES = {
    "real-momentum-advection-sections-sm86.png": (
        "661b2020a4b2e8118b824b7522ef3789309170edd1ec3fb67a62da1084a4ac8b",
        (2790, 2160),
    ),
    "real-momentum-advection-xz-sm86.png": (
        "965961f97ece2b1ff2da5cfd6c79cfe8b15ed2878b97b574a632394ddbb35daa",
        (2790, 972),
    ),
}


def _payload() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    return struct.unpack(">II", payload[16:24])


def test_real_momentum_receipt_is_strict_and_bounded() -> None:
    payload = _payload()
    assert payload["schema"] == "gpuwm-cm1.real-momentum-advection5-evidence/v1"
    assert payload["result"] == "passed-real-product-full-domain-momentum-admission"
    assert payload["domain"] == {"ni": 300, "nj": 300, "nk": 75}
    assert payload["spacing_m"] == {"dx": 100.0, "dy": 100.0, "dz": 200.0}
    assert payload["source_export"] == {
        "artifact_sha256": "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f",
        "contract_sha256": "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12",
        "manifest_sha256": "7fd218a005b1c9f01d37b57452b281c2df6c797973268e8fe86b996eb6b61e2e",
        "payload_nbytes": 1_169_829_300,
        "payload_sha256": "8f589cee64b044afcfef34dca6b638c3829946a999537885b1db3576d4ec7981",
    }
    assert payload["output"] == EXPECTED_OUTPUTS
    assert payload["gpu_mutex"]["clean_release"] is True  # type: ignore[index]
    assert payload["runtime"]["device_name"] == "NVIDIA GeForce RTX 3080"  # type: ignore[index]
    claim = str(payload["claim_limit"])
    assert "uniform-horizontal and uniform-sigma" in claim
    assert "no" in claim and "coupled forecast" in claim and "tornado" in claim


def test_real_momentum_statistics_cover_every_live_operator_output() -> None:
    statistics = _payload()["statistics"]
    assert isinstance(statistics, dict)
    expected_words = {
        "divx": 6_750_000,
        "rru": 6_772_500,
        "rrv": 6_772_500,
        "rrw": 6_660_000,
        "uten_open_interior_faces": 6_727_500,
        "vten_open_interior_faces": 6_727_500,
        "wten_internal_faces": 6_660_000,
    }
    for name, words in expected_words.items():
        values = statistics[name]
        assert values["nonfinite_words"] == 0
        assert values["nonzero_words"] == words
        assert values["minimum_f32"] < values["maximum_f32"]
        assert values["rms_f64"] > 0.0


def test_real_momentum_pngs_are_exact_labeled_raster_outputs() -> None:
    payload = _payload()
    inventory = {recorded_name(item["path"]): item for item in payload["figures"]}
    assert set(inventory) == set(EXPECTED_FIGURES)
    for name, (sha256, dimensions) in EXPECTED_FIGURES.items():
        path = EVIDENCE / name
        raster = path.read_bytes()
        assert hashlib.sha256(raster).hexdigest() == sha256
        assert _png_dimensions(raster) == dimensions
        assert inventory[name]["sha256"] == sha256
        assert inventory[name]["nbytes"] == len(raster)
