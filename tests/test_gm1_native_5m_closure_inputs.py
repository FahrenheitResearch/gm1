from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_stretched_closure import INPUT_NAMES
from cm1core import gm1_native_5m_closure_inputs as native
from authority_trees import requires_work


ROOT = Path(__file__).resolve().parents[1]
AUXILIARY = Path(r"F:\gm1-artifacts\work\gm1-real-5m-child-auxiliary-v2")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_and_validation_pins_match_the_admitted_native_estate() -> None:
    roots = {
        "terrain": ROOT / "work/gm1-real-5m-child-terrain-v2",
        "atmosphere": ROOT / "work/gm1-real-5m-child-atmosphere-v2",
        "static_heights": ROOT / "work/gm1-real-5m-child-static-heights-v2",
        "auxiliary": AUXILIARY,
    }
    evidence = {
        "terrain": ROOT
        / "evidence/validation/gm1-real-5m-child-terrain-v2/validation-receipt.json",
        "atmosphere": ROOT
        / "evidence/validation/gm1-real-5m-child-atmosphere-v2/validation-receipt.json",
        "static_heights": ROOT
        / "evidence/validation/gm1-real-5m-child-static-heights-v2/validation-receipt.json",
        "auxiliary": ROOT
        / "evidence/validation/gm1-real-5m-child-auxiliary-stream-v2/validation-receipt.json",
    }
    for name, root in roots.items():
        if not root.exists():
            pytest.skip(f"native artifact is external on this host: {root}")
        pins = native.ARTIFACT_PINS[name]
        assert _sha(root / "manifest.json") == pins["manifest_sha256"]
        assert _sha(root / "receipt.json") == pins["receipt_sha256"]
        assert _sha(evidence[name]) == pins["validation_sha256"]


def test_consumed_carrier_layout_is_exact_and_byte_ranges_fit_source_files() -> None:
    assert tuple(native.CONSUMED_CARRIERS) == (
        "terrain_msl",
        "pp3d",
        "q3d",
        "th3d",
        "u3d",
        "v3d",
        "w3d",
        "pi0",
        "th0",
        "prs",
        "rf",
        "rho",
    )
    assert sum(value.nbytes for value in native.CONSUMED_CARRIERS.values()) == (
        43_475_998_176
    )
    for carrier in native.CONSUMED_CARRIERS.values():
        assert carrier.nbytes == 4 * __import__("math").prod(carrier.shape)
        assert len(carrier.sha256) == 64
        pin = native.PAYLOAD_PINS[carrier.group]
        assert carrier.offset_nbytes + carrier.nbytes <= pin["nbytes"]
    assert native.CONSUMED_CARRIERS["q3d"].shape == (6, 75, 3000, 3000)
    assert native.CONSUMED_CARRIERS["u3d"].shape == (75, 3000, 3001)
    assert native.CONSUMED_CARRIERS["v3d"].shape == (75, 3001, 3000)
    assert native.CONSUMED_CARRIERS["pi0"].shape == (77, 3006, 3006)


def test_manifest_descriptors_match_exact_consumed_carriers_without_payload_scan() -> None:
    if not AUXILIARY.exists():
        pytest.skip("native F: auxiliary is not attached")
    loaded = native.load_native_5m_closure_inputs(
        repo=ROOT,
        terrain=ROOT / "work/gm1-real-5m-child-terrain-v2",
        atmosphere=ROOT / "work/gm1-real-5m-child-atmosphere-v2",
        static_heights=ROOT / "work/gm1-real-5m-child-static-heights-v2",
        auxiliary=AUXILIARY,
        verify_consumed_bytes=False,
    )
    assert tuple(loaded.arrays) == tuple(native.CONSUMED_CARRIERS)
    assert loaded.identity["consumed_nbytes"] == 43_475_998_176
    assert loaded.identity["full_domain_duplicate_bytes"] == 0
    assert loaded.identity["independent_consumed_byte_verification"][
        "performed"
    ] is False
    assert loaded.identity["synthetic_detail"] is False
    assert loaded.identity["synthetic_translation"] is False


def test_base_slice_and_staggered_tile_shapes_are_source_exact() -> None:
    plan = build_gm1_open_tile_plan(
        nx=3000, ny=3000, tile_nx=192, tile_ny=192
    )
    first, last = plan.specs[0], plan.specs[-1]
    assert first.record()["compute_mass_yx"] == [0, 198, 0, 198]
    assert last.record()["compute_mass_yx"] == [2802, 3000, 2802, 3000]
    for spec in (first, last):
        base_y = (3 + spec.cj0, 3 + spec.cj0 + spec.cny)
        base_x = (3 + spec.ci0, 3 + spec.ci0 + spec.cnx)
        assert 3 <= base_y[0] < base_y[1] <= 3003
        assert 3 <= base_x[0] < base_x[1] <= 3003
    assert INPUT_NAMES == (
        "terrain_msl",
        "u3d",
        "v3d",
        "w3d",
        "rho",
        "rf",
        "pi0",
        "th0",
        "prs",
        "pp3d",
        "th3d",
        "q3d",
    )


@requires_work(
    "gm1-real-5m-child-terrain-v2/manifest.json",
    "gm1-real-5m-child-atmosphere-v2/manifest.json",
)
def test_top_level_source_manifests_remain_non_synthetic() -> None:
    for relative in (
        "work/gm1-real-5m-child-terrain-v2/manifest.json",
        "work/gm1-real-5m-child-atmosphere-v2/manifest.json",
    ):
        value = json.loads((ROOT / relative).read_bytes())
        assert value["synthetic_detail"] is False
        assert value["synthetic_translation"] is False
