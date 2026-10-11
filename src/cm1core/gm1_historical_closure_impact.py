"""Machine-readable impact audit for the refused uniform-z child closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "gm1.historical-uniform-closure-downstream-impact/v1"
STATUS = "REFUSED_REVALIDATION_REQUIRED"
HISTORICAL_CLOSURE_PINS = {
    "manifest": "7aba693cd3a5c984c5a0d969bffa0793f80dfea138aa661da34a86f69982d9d9",
    "payload": "dc277233d7bead2009749e09f402fdff6c072323a57568fef270dbe6904dca6c",
    "receipt": "9972f6e981dc99e5635a844b70906328339535450bc57cdb01e61469b8080985",
    "defv": "d7f044de8f9dea821f3f71616d64c5cdc770617ec92c189d8f46936d3dc05a78",
    "defh": "0ba40b06156908886c5eb6560e9030c19bc36a37913c3f50f1b84735c948d24d",
}
CORRECTED_STRETCHED_PINS = {
    "defv": "35c0ce245ef7b647331cf6796b492d81bd1d6da6d63bfe19f21c3e1bce5c4167",
    "defh": "930f8c270bdc59e8ba2d6d37301bbfaab79de9d81f3cbe490fc74b14c22f0184",
}
LINEAGE_PINS = {
    "full_state_manifest": "02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d",
    "full_state_first_step": "132f2f7f877abe9bbfac44900406ceae17c6de93d0bcfc9dacacefe8d993edeb",
    "full_state_checkpoint_admission": "23c0f09129084690c16cae85461d23f79f5645b430a64988242a6bee0119ba1a",
    "full_state_receipt": "792dc1ade597590e985b44c434a05c0c47751390bd5949520abb309527fdebb6",
    "step1_pair_manifest": "06a1df66b7c1132d742337a9329314e62b7ff655934e916270e59ade50e7d05e",
    "step600_pair_manifest": "c37e5bec8101b797fe516fe21f3474785f3bd2884337d646f2ba0f7fbe8961e5",
    "step600_history": "94108b08e926a9d4ebc09029b66afcc3db4948bb1f769296a9e100d9e2f7bb4e",
    "step1200_pair_manifest": "4455ff8af3d63959c783f13cc368e01cd72f96f3e1d63f317c27337cdf54b290",
    "step1200_history": "fab96b6a63a89bbad8892f2944ddf82321fcf79b92b538d78678573fb8e3f9ec",
}


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _walk(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{pointer}/{index}")
    elif isinstance(value, str):
        yield pointer, value.lower()


def _classify(path: str) -> dict[str, Any]:
    lower = path.lower()
    if "gm1-t300-child-closure-v2" in lower:
        return {
            "kind": "historical_closure_artifact",
            "carrier_role": "initialization producer output",
            "evolution_affected": False,
            "status": "REFUSED_STRETCHED_Z_ORACLE",
        }
    if "stretched-live-nm-def" in lower:
        return {
            "kind": "negative_control_evidence",
            "carrier_role": "comparison-only uniform-z negative control",
            "evolution_affected": False,
            "status": "PRESERVED_NEGATIVE_CONTROL",
        }
    if "t300-child-full-state" in lower:
        return {
            "kind": "complete_child_envelope",
            "carrier_role": "initialization closure and derived coefficient state",
            "evolution_affected": False,
            "status": STATUS,
        }
    if "tilestream" in lower:
        return {
            "kind": "isolated_tilestream_operator_evidence",
            "carrier_role": (
                "accepted-real fixture lineage; isolated downstream stage did not "
                "recompute or directly validate corrected initial deformation"
            ),
            "evolution_affected": False,
            "operator_local_raw_bit_comparison_preserved": True,
            "status": STATUS,
        }
    if any(
        token in lower
        for token in (
            "one-step",
            "continuation-step",
            "child-live",
            "child-restore",
            "resume-trajectory",
            "step600",
            "step0000",
            "view-cloud-pack",
            "view-packs",
            "post-step",
            "q3d-layout",
        )
    ):
        return {
            "kind": "trajectory_checkpoint_history_or_derived_product",
            "carrier_role": (
                "initial closure coefficients/TKE fed the first evolved solve1; "
                "later state and derived products inherit that trajectory"
            ),
            "evolution_affected": True,
            "status": STATUS,
        }
    if any(token in lower for token in ("turbsmag", "sgs-tke", "nm-def-tke")):
        return {
            "kind": "closure_operator_or_replay_evidence",
            "carrier_role": "direct historical deformation/derived-closure input or output",
            "evolution_affected": False,
            "status": STATUS,
        }
    return {
        "kind": "closure_lineage_reference",
        "carrier_role": "historical closure identity or transitive complete-child lineage",
        "evolution_affected": False,
        "status": STATUS,
    }


def build_historical_closure_impact_map(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    targets = {
        **{f"historical_{key}": value for key, value in HISTORICAL_CLOSURE_PINS.items()},
        **{f"lineage_{key}": value for key, value in LINEAGE_PINS.items()},
    }
    records: list[dict[str, Any]] = []
    for relative_root in (Path("work"), Path("evidence/validation")):
        search_root = root / relative_root
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*.json")):
            if any(part.startswith("target") for part in path.parts):
                continue
            try:
                document = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            hits = [
                {"identity": name, "json_pointer": pointer}
                for pointer, value in _walk(document)
                for name, sha256 in targets.items()
                if value == sha256
            ]
            if not hits:
                continue
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "path": relative,
                    "file_sha256": _sha_path(path),
                    "references": hits,
                    **_classify(relative),
                }
            )
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    affected = [record for record in records if record["evolution_affected"]]
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "historical_closure": {
            **HISTORICAL_CLOSURE_PINS,
            "disposition": "immutable historical bytes; never relabel or overwrite",
            "defv_defh_identity": "sealed uniform-z negative control",
        },
        "corrected_stretched_deformation": CORRECTED_STRETCHED_PINS,
        "lineage_pins": LINEAGE_PINS,
        "records": records,
        "summary": {
            "record_count": len(records),
            "evolution_affected_record_count": len(affected),
            "kind_counts": kinds,
            "all_non_control_products_refused_revalidation": all(
                record["status"]
                in {"REFUSED_REVALIDATION_REQUIRED", "REFUSED_STRETCHED_Z_ORACLE"}
                for record in records
                if record["kind"] != "negative_control_evidence"
            ),
        },
        "interpretation": (
            "The historical defv/defh were not merely labels: fresh km/kh/TKE "
            "were derived from them and the first child solve consumed that closure. "
            "Complete-child and evolved descendants therefore require rerun from "
            "the corrected stretched closure. Isolated operator-local raw-bit "
            "comparisons may remain useful but are not a revalidated product lineage."
        ),
    }


__all__ = (
    "CORRECTED_STRETCHED_PINS",
    "HISTORICAL_CLOSURE_PINS",
    "LINEAGE_PINS",
    "SCHEMA",
    "STATUS",
    "build_historical_closure_impact_map",
)
