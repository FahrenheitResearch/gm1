from __future__ import annotations

from pathlib import Path

import pytest

from cm1core.gm1_historical_closure_impact import (
    STATUS,
    build_historical_closure_impact_map,
)
from authority_trees import requires_work


ROOT = Path(__file__).resolve().parents[1]

# The impact map walks work/ and evidence/validation/ (the module's
# build_historical_closure_impact_map), so the lineage it classifies is only
# complete where the captured child products are staged.
requires_child_lineage = requires_work(
    "gm1-t300-child-full-state-v1/manifest.json",
    "gm1-real-50m-child-live-v2/step-00000001/manifest.json",
    "gm1-real-50m-child-live-v2/step-00000600/manifest.json",
    "gm1-real-50m-child-view-packs-v1/step-00001200/manifest.json",
)
# This receipt was never committed; it exists only in a locally captured
# evidence/ tree beside work/.
ONE_STEP_V2_RECEIPT = (
    ROOT / "evidence/validation/gm1-real-50m-child-one-step-v2/receipt.json"
)
requires_one_step_v2_receipt = pytest.mark.skipif(
    not ONE_STEP_V2_RECEIPT.is_file(),
    reason=(
        "requires the locally captured, uncommitted "
        "evidence/validation/gm1-real-50m-child-one-step-v2/receipt.json"
    ),
)


@requires_child_lineage
@requires_one_step_v2_receipt
def test_impact_map_is_exhaustive_for_known_complete_child_lineage() -> None:
    result = build_historical_closure_impact_map(ROOT)
    paths = {record["path"]: record for record in result["records"]}
    required = {
        "work/gm1-t300-child-full-state-v1/manifest.json",
        "evidence/validation/gm1-real-50m-child-one-step-v2/receipt.json",
        "work/gm1-real-50m-child-live-v2/step-00000001/manifest.json",
        "work/gm1-real-50m-child-live-v2/step-00000600/manifest.json",
        "evidence/validation/gm1-real-50m-child-continuation-step3-v1/receipt.json",
        "work/gm1-real-50m-child-view-packs-v1/step-00001200/manifest.json",
        "evidence/validation/gm1-real-50m-child-step600-diagnostics-v1/receipt.json",
    }
    assert required.issubset(paths)
    assert all(paths[path]["status"] == STATUS for path in required)
    assert all(
        paths[path]["evolution_affected"]
        for path in required
        if "full-state" not in path
    )


def test_negative_control_is_preserved_not_relabelled() -> None:
    result = build_historical_closure_impact_map(ROOT)
    record = next(
        item
        for item in result["records"]
        if item["path"].endswith(
            "gm1-child-stretched-live-nm-def-real-v1/receipt.json"
        )
    )
    assert record["status"] == "PRESERVED_NEGATIVE_CONTROL"
    assert record["evolution_affected"] is False


@requires_child_lineage
def test_all_non_control_products_are_refused() -> None:
    result = build_historical_closure_impact_map(ROOT)
    assert result["summary"]["record_count"] >= 30
    assert result["summary"]["evolution_affected_record_count"] >= 8
    assert result["summary"]["all_non_control_products_refused_revalidation"] is True
