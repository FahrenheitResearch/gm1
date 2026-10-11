"""Fail-closed native 5 m complete-child envelope requalification.

The historical 300/50 full-state envelope is dimension pinned and names the
now-refused uniform-z closure-v2 artifact.  It is immutable evidence, not a
template whose closure hash may be rotated.  This module instead constructs a
small reference-only envelope over the independently admitted native 5 m
terrain, static-height, atmosphere, auxiliary, and stretched-closure stores.

No numeric carrier is copied.  The envelope projects the exact shape-parametric
37-carrier checkpoint ABI onto the step-zero sources and deliberately refuses
first-step, checkpoint, and runnable-product admission until the remaining
tiled acoustic/RK/lifecycle gates produce live tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from cm1core.cuda_gm1_tilestream import (
    CHECKPOINT_NAMES,
    checkpoint_carrier_shape,
    checkpoint_store_bytes,
)


SCHEMA = "gm1.native-5m-complete-child-envelope-requalification/v1"
MANIFEST_SCHEMA = f"{SCHEMA}/manifest"
FIRST_STEP_SCHEMA = f"{SCHEMA}/first-step-refusal"
CHECKPOINT_SCHEMA = f"{SCHEMA}/checkpoint-admission"
RECEIPT_SCHEMA = f"{SCHEMA}/receipt"
STATUS = "REFUSED_RK_REVALIDATION_REQUIRED"
CLAIM_LIMIT = (
    "exact reference-only binding of the independently admitted native "
    "3000x3000x75/5 m terrain, static-height, atmosphere, auxiliary, and "
    "corrected stretched-closure artifacts to the canonical 37-carrier "
    "initialization projection; no first large step, completed-step "
    "checkpoint, trajectory, forecast, storm, tornado, synthetic detail, or "
    "runnable 5 m product is admitted"
)
NX = 3000
NY = 3000
NZ = 75
DX_M = 5.0
DY_M = 5.0
CHECKPOINT_NBYTES = 97_603_303_392
IMPACT_MAP_SHA256 = (
    "4523b1f12f56ab5639c5bde82694a335b2db2327478ff376706085d153981307"
)
CLOSURE_VALIDATION_SHA256 = (
    "b46a3ca1d245dc6cf87c863f148c7776c0d350516458cde457c159f3fd39d31c"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")

ARTIFACT_PINS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "terrain": MappingProxyType(
            {
                "manifest_sha256": "7a6b5139168af3384c58998bff8df1dd61f102c2f611ac163251425e607ca45e",
                "receipt_sha256": "553b27baa2f0fda542703a09ba924284f78e9c251e14cd56309958d39c151152",
                "validation_sha256": "d99f1ae105773cc808a2fd11e330ddeed2ca5cf6a7f59f6db96d09f13ad239cd",
                "inventory": (
                    "manifest.json",
                    "receipt.json",
                    "terrain-msl.f32le",
                ),
            }
        ),
        "atmosphere": MappingProxyType(
            {
                "manifest_sha256": "ba1391a3d9f4914f4b9a23f0d5021994d7587786829f4b85cb6eee0db5047447",
                "receipt_sha256": "24790ddca9403c5066c69517e51442ff24d19d7f6dc38f999cabe96116983483",
                "validation_sha256": "c51a34ffc1625bb60222a96c60d13b3fe60a04bb82cab396d6ad7acb95b69b48",
                "inventory": ("arena.f32le", "manifest.json", "receipt.json"),
            }
        ),
        "static_heights": MappingProxyType(
            {
                "manifest_sha256": "e991bf9b85ca71d835b5d34c16e4c877a97af3e36dd12e69a44a930bc96e8104",
                "receipt_sha256": "1501521cbf0dd4a025d842916bf2cd0a5939f7c7150484751b52dd64ce920155",
                "validation_sha256": "cd3032b2a515cf40200532b84780e8b1a1d6a1b92b645d67c8df3333176e3766",
                "inventory": (
                    "manifest.json",
                    "receipt.json",
                    "zf.f32le",
                    "zh.f32le",
                ),
            }
        ),
        "auxiliary": MappingProxyType(
            {
                "manifest_sha256": "282b1ddbc09b3fe70b45c38ba60ce9ec96dc30c5cee2420a645247515a723002",
                "receipt_sha256": "f0762b758a997f4443ad311c66ade1e5b43f3d51353b1a45d9fa4553e24009e9",
                "validation_sha256": "6c7852db9d55bb96e844b577312b269f2e84cdbb206669051349afb43f7442d7",
                "inventory": (
                    "base-wind.f32le",
                    "live-density.f32le",
                    "manifest.json",
                    "receipt.json",
                    "terrain-open.f32le",
                    "total-geopotential-zf.f32le",
                ),
            }
        ),
        "closure": MappingProxyType(
            {
                "manifest_sha256": "f08bd8dbd58708f6c6e673b15cdad213670ff0d895de853d275e712fd6680612",
                "receipt_sha256": "ab49061667936aa38a8ff410b3e7502eca78bde540da302ab109357abf056100",
                "validation_sha256": CLOSURE_VALIDATION_SHA256,
                "inventory": ("closure.f32le", "manifest.json", "receipt.json"),
            }
        ),
    }
)

HISTORICAL_FULL_STATE_PINS = MappingProxyType(
    {
        "manifest.json": "02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d",
        "first-step.json": "132f2f7f877abe9bbfac44900406ceae17c6de93d0bcfc9dacacefe8d993edeb",
        "checkpoint-admission.json": "23c0f09129084690c16cae85461d23f79f5645b430a64988242a6bee0119ba1a",
        "receipt.json": "792dc1ade597590e985b44c434a05c0c47751390bd5949520abb309527fdebb6",
        "base-wind.f32le": "adf38848863b1cb36c7918e5617d9f38b81443ddf965bc6358ef7dd60ffaf8dd",
    }
)

MISSING_LIFECYCLE_GATES = (
    "live same-stage nrk=1 post-acoustic mass-flux and getdivx token",
    "source-ordered RK1 current updates, physical boundary publication, and TKE RK",
    "source-ordered RK2 and RK3 acoustic/momentum/scalar/moist/TKE stages",
    "global psolver3 pressure workspace/decomposition and admitted open-boundary cadence",
    "final solve3 current-to-A/TKE/density and completed-step closure refresh tokens",
    "one complete coupled native 5 m step with quiescent 37-carrier publication",
    "native 5 m raw-bit checkpoint restore and uninterrupted/resumed identity",
)


@dataclass(frozen=True, slots=True)
class Native5mEnvelopePaths:
    repo: Path
    terrain: Path
    atmosphere: Path
    static_heights: Path
    auxiliary: Path
    closure: Path
    impact_map: Path
    historical_full_state: Path
    validation: Mapping[str, Path]


def default_paths(repo: Path) -> Native5mEnvelopePaths:
    root = Path(repo).resolve()
    evidence = root / "evidence/validation"
    return Native5mEnvelopePaths(
        repo=root,
        terrain=root / "work/gm1-real-5m-child-terrain-v2",
        atmosphere=root / "work/gm1-real-5m-child-atmosphere-v2",
        static_heights=root / "work/gm1-real-5m-child-static-heights-v2",
        auxiliary=Path(r"F:\gm1-artifacts\work\gm1-real-5m-child-auxiliary-v2"),
        closure=Path(
            r"F:\gm1-artifacts\work\gm1-real-5m-child-stretched-closure-v3"
        ),
        impact_map=evidence
        / "gm1-t300-child-stretched-closure-v3/downstream-impact-map.json",
        historical_full_state=root / "work/gm1-t300-child-full-state-v1",
        validation=MappingProxyType(
            {
                "terrain": evidence
                / "gm1-real-5m-child-terrain-v2/validation-receipt.json",
                "atmosphere": evidence
                / "gm1-real-5m-child-atmosphere-v2/validation-receipt.json",
                "static_heights": evidence
                / "gm1-real-5m-child-static-heights-v2/validation-receipt.json",
                "auxiliary": evidence
                / "gm1-real-5m-child-auxiliary-stream-v2/validation-receipt.json",
                "closure": evidence
                / "gm1-real-5m-child-stretched-closure-v3/receipt.json",
            }
        ),
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_pinned_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    observed = sha_path(path)
    if observed != expected_sha256:
        raise ValueError(f"{label} identity changed")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_inventory(directory: Path, expected: tuple[str, ...], label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing {label} directory {directory}")
    observed = tuple(sorted(path.name for path in directory.iterdir()))
    if observed != tuple(sorted(expected)):
        raise ValueError(f"{label} exact inventory changed: {observed}")


def _payload_specs(name: str, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if name == "terrain":
        raw = [manifest["payload"]]
    elif name == "atmosphere":
        raw = [manifest["arena"]]
    elif name == "static_heights":
        raw = list(manifest["payloads"].values())
    elif name == "auxiliary":
        raw = list(manifest["outputs"])
    elif name == "closure":
        raw = [manifest["payload"]]
    else:  # pragma: no cover - internal exhaustive guard
        raise KeyError(name)
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        file = item["file"]
        nbytes = item["nbytes"]
        sha256 = item["sha256"]
        if (
            not isinstance(file, str)
            or isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes < 1
        ):
            raise ValueError(f"{name} payload descriptor is invalid")
        _require_sha(sha256, f"{name} {file} sha256")
        if file in result:
            raise ValueError(f"{name} payload file repeated")
        result[file] = {"nbytes": nbytes, "sha256": sha256}
    return result


def _artifact_record(
    name: str, directory: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    pins = ARTIFACT_PINS[name]
    _require_inventory(directory, tuple(pins["inventory"]), name)
    manifest = _read_pinned_json(
        directory / "manifest.json", str(pins["manifest_sha256"]), f"{name} manifest"
    )
    receipt = _read_pinned_json(
        directory / "receipt.json", str(pins["receipt_sha256"]), f"{name} receipt"
    )
    validation = _read_pinned_json(
        validation_path, str(pins["validation_sha256"]), f"{name} validation"
    )
    payloads = _payload_specs(name, manifest)
    for file, spec in payloads.items():
        path = directory / file
        if not path.is_file() or path.stat().st_size != spec["nbytes"]:
            raise ValueError(f"{name} {file} size changed")
    record = {
        "directory": str(directory.resolve()),
        "manifest_sha256": pins["manifest_sha256"],
        "receipt_sha256": pins["receipt_sha256"],
        "validation_sha256": pins["validation_sha256"],
        "payloads": payloads,
        "payload_hash_disposition": (
            "manifest-bound and independently every-byte inspected by the exact "
            "validation receipt; this requalification performs metadata/stat checks "
            "and does not reread or duplicate the immutable large payload"
        ),
    }
    return record, {"manifest": manifest, "receipt": receipt, "validation": validation}


def _carrier_index(manifest: Mapping[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if kind == "atmosphere":
        file = manifest["arena"]["file"]
        rows = [dict(item, file=file) for item in manifest["carriers"]]
        for row in rows:
            row["offset_nbytes"] = row.pop("offset_bytes")
    elif kind == "auxiliary":
        for output in manifest["outputs"]:
            rows.extend(dict(item, file=output["file"]) for item in output["carriers"])
    elif kind == "closure":
        file = manifest["payload"]["file"]
        rows = [dict(item, file=file) for item in manifest["carriers"]]
    else:  # pragma: no cover
        raise KeyError(kind)
    result = {row["name"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{kind} carrier names are not unique")
    return result


def _slice_shape(source_slices: tuple[tuple[int, int, int], ...]) -> tuple[int, ...]:
    return tuple(len(range(start, stop, step)) for start, stop, step in source_slices)


def _binding_plan() -> dict[str, tuple[str, str, tuple[tuple[int, int, int], ...] | None]]:
    result: dict[
        str, tuple[str, str, tuple[tuple[int, int, int], ...] | None]
    ] = {}
    for name in CHECKPOINT_NAMES:
        if name == "closure.tkea":
            result[name] = ("closure", "tkea", None)
        elif name.startswith("component.") or name.startswith("held.") or name in {
            "qa",
            "q3d",
        }:
            result[name] = ("atmosphere", name, None)
        elif name in {"prs", "rho", "rr", "rf"}:
            result[name] = ("auxiliary", name, None)
        elif name in {"radbcw", "radbce", "radbcs", "radbcn"}:
            result[name] = ("auxiliary", name, None)
        elif name in {"ppi", "pp3d", "ppx"}:
            result[name] = (
                "auxiliary",
                name,
                ((1, NZ + 1, 1), (1, NY + 1, 1), (1, NX + 1, 1)),
            )
        elif name in {"tha", "th3d"}:
            result[name] = (
                "auxiliary",
                name,
                ((1, NZ + 1, 1), (3, NY + 3, 1), (3, NX + 3, 1)),
            )
        elif name in {"ua", "u3d"}:
            result[name] = (
                "auxiliary",
                name,
                ((1, NZ + 1, 1), (3, NY + 3, 1), (3, NX + 4, 1)),
            )
        elif name in {"va", "v3d"}:
            result[name] = (
                "auxiliary",
                name,
                ((1, NZ + 1, 1), (3, NY + 4, 1), (3, NX + 3, 1)),
            )
        elif name in {"wa", "w3d"}:
            result[name] = (
                "auxiliary",
                name,
                ((1, NZ + 2, 1), (3, NY + 3, 1), (3, NX + 3, 1)),
            )
        else:  # pragma: no cover - CHECKPOINT_NAMES is exact
            raise KeyError(name)
    return result


def build_persistent_sources(
    atmosphere_manifest: Mapping[str, Any],
    auxiliary_manifest: Mapping[str, Any],
    closure_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    indices = {
        "atmosphere": _carrier_index(atmosphere_manifest, "atmosphere"),
        "auxiliary": _carrier_index(auxiliary_manifest, "auxiliary"),
        "closure": _carrier_index(closure_manifest, "closure"),
    }
    result: list[dict[str, Any]] = []
    for name in CHECKPOINT_NAMES:
        artifact, source_name, slices = _binding_plan()[name]
        source = indices[artifact].get(source_name)
        if source is None:
            raise ValueError(f"missing {artifact} carrier {source_name}")
        expected_shape = checkpoint_carrier_shape(name, nx=NX, ny=NY, nz=NZ)
        source_shape = tuple(source["shape"])
        if slices is not None and (
            len(slices) != len(source_shape)
            or any(
                start < 0 or stop > extent or step < 1 or start >= stop
                for (start, stop, step), extent in zip(slices, source_shape)
            )
        ):
            raise ValueError(f"{name} source slices exceed the sealed carrier shape")
        view_shape = source_shape if slices is None else _slice_shape(slices)
        if view_shape != expected_shape:
            raise ValueError(
                f"{name} source view shape {view_shape} != checkpoint {expected_shape}"
            )
        _require_sha(source["sha256"], f"{artifact} {source_name} sha256")
        expected_nbytes = math.prod(expected_shape) * 4
        record: dict[str, Any] = {
            "name": name,
            "shape": list(expected_shape),
            "nbytes": expected_nbytes,
            "dtype": "little-endian-f32",
            "order": "C",
            "source": {
                "artifact": artifact,
                "file": source["file"],
                "carrier": source_name,
                "carrier_shape": list(source_shape),
                "carrier_offset_nbytes": source["offset_nbytes"],
                "carrier_nbytes": source["nbytes"],
                "carrier_sha256": source["sha256"],
            },
            "lifecycle_disposition": (
                "step-zero initialization view only; after evolution this canonical "
                "carrier must come from a quiescent completed-step publication"
            ),
        }
        if slices is not None:
            record["source"]["source_slices"] = [list(item) for item in slices]
        result.append(record)
    if tuple(row["name"] for row in result) != CHECKPOINT_NAMES:
        raise AssertionError("persistent source order changed")
    if sum(row["nbytes"] for row in result) != CHECKPOINT_NBYTES:
        raise AssertionError("native 5 m checkpoint projection byte count changed")
    return result


def audit_native_5m_complete_envelope(
    paths: Native5mEnvelopePaths,
) -> dict[str, Any]:
    if checkpoint_store_bytes(nx=NX, ny=NY, nz=NZ) != CHECKPOINT_NBYTES:
        raise AssertionError("shape-parametric 37-carrier checkpoint ABI changed")
    artifact_paths = {
        "terrain": paths.terrain,
        "atmosphere": paths.atmosphere,
        "static_heights": paths.static_heights,
        "auxiliary": paths.auxiliary,
        "closure": paths.closure,
    }
    artifacts: dict[str, Any] = {}
    documents: dict[str, Any] = {}
    for name, directory in artifact_paths.items():
        artifacts[name], documents[name] = _artifact_record(
            name, directory, paths.validation[name]
        )
    closure_validation = documents["closure"]["validation"]
    independent = closure_validation.get("independent_every_output_byte_inspection")
    if not isinstance(independent, dict) or independent.get("action") != "inspected":
        raise ValueError("closure independent every-byte inspection is absent")
    for key, expected in (
        ("manifest_sha256", ARTIFACT_PINS["closure"]["manifest_sha256"]),
        ("payload_sha256", artifacts["closure"]["payloads"]["closure.f32le"]["sha256"]),
        ("receipt_sha256", ARTIFACT_PINS["closure"]["receipt_sha256"]),
    ):
        if independent.get(key) != expected:
            raise ValueError(f"closure independent inspection {key} changed")
    impact = _read_pinned_json(paths.impact_map, IMPACT_MAP_SHA256, "impact map")
    summary = impact.get("summary")
    if (
        impact.get("status") != "REFUSED_REVALIDATION_REQUIRED"
        or not isinstance(summary, dict)
        or summary.get("record_count") != 59
        or summary.get("all_non_control_products_refused_revalidation") is not True
    ):
        raise ValueError("historical impact map disposition changed")
    historical_files: dict[str, dict[str, object]] = {}
    for file, expected in HISTORICAL_FULL_STATE_PINS.items():
        path = paths.historical_full_state / file
        observed = sha_path(path)
        if observed != expected:
            raise ValueError(f"historical full-state {file} identity changed")
        historical_files[file] = {
            "sha256": observed,
            "nbytes": path.stat().st_size,
        }
    persistent_sources = build_persistent_sources(
        documents["atmosphere"]["manifest"],
        documents["auxiliary"]["manifest"],
        documents["closure"]["manifest"],
    )
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "claim_limit": CLAIM_LIMIT,
        "model": "GM1",
        "grid": {
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "dx_m": DX_M,
            "dy_m": DY_M,
            "ztop_m": 15000.0,
            "vertical_grid_sha256": "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56",
        },
        "artifacts": artifacts,
        "canonical_checkpoint_projection": {
            "abi": "gm1.forecast-checkpoint-carriers/v1",
            "carrier_count": len(persistent_sources),
            "nbytes": CHECKPOINT_NBYTES,
            "persistent_sources": persistent_sources,
            "initialization_boundary_admitted_as_checkpoint": False,
            "status": "REFERENCE_MAP_ONLY_NOT_A_CHECKPOINT",
        },
        "corrected_closure": {
            "validation_receipt_sha256": CLOSURE_VALIDATION_SHA256,
            "independent_every_output_byte_inspection": independent,
            "historical_impact_map_sha256": IMPACT_MAP_SHA256,
            "historical_impact_record_count": 59,
        },
        "historical_300x300_complete_child": {
            "directory": str(paths.historical_full_state.resolve()),
            "files": historical_files,
            "status": "REFUSED_REVALIDATION_REQUIRED",
            "disposition": "immutable historical bytes; never relabel or overwrite",
            "reasons": [
                "hard-pinned 300x300x75/50 m grid",
                "binds refused uniform-z closure-v2",
                "evolved descendants remain individually refused by the 59-record impact map",
            ],
        },
        "initialization_component_set_admitted": True,
        "first_large_step_ready": False,
        "completed_large_step": False,
        "checkpoint_available": False,
        "runnable_5m_product": False,
        "missing_lifecycle_gates": list(MISSING_LIFECYCLE_GATES),
    }


def build_envelope_documents(
    audit: Mapping[str, Any], *, producer: Mapping[str, Any]
) -> dict[str, bytes]:
    if audit.get("status") != STATUS or audit.get("runnable_5m_product") is not False:
        raise ValueError("audit is not the fail-closed native 5 m requalification")
    producer_record = {key: producer[key] for key in sorted(producer)}
    for key, value in producer_record.items():
        if key.endswith("sha256"):
            _require_sha(value, f"producer {key}")
    manifest = dict(audit)
    manifest["schema"] = MANIFEST_SCHEMA
    manifest["producer"] = producer_record
    manifest_bytes = canonical_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    first_step = {
        "schema": FIRST_STEP_SCHEMA,
        "status": "REFUSED_NATIVE_5M_FIRST_STEP_RK_GATES_INCOMPLETE",
        "manifest_sha256": manifest_sha256,
        "model": "GM1",
        "step_index": 0,
        "model_time_seconds_f64_bits": "0000000000000000",
        "initialization_component_set_admitted": True,
        "first_large_step_ready": False,
        "next_required_gate": MISSING_LIFECYCLE_GATES[0],
        "missing_lifecycle_gates": list(MISSING_LIFECYCLE_GATES),
        "claim_limit": CLAIM_LIMIT,
    }
    first_step_bytes = canonical_bytes(first_step)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "REFUSED_NO_COMPLETED_NATIVE_5M_LARGE_STEP",
        "manifest_sha256": manifest_sha256,
        "carrier_abi": "gm1.forecast-checkpoint-carriers/v1",
        "carrier_count": 37,
        "projected_nbytes": CHECKPOINT_NBYTES,
        "initialization_boundary_admitted_as_checkpoint": False,
        "required_boundary": "quiescent after one admitted complete native 5 m large step",
        "runnable_5m_product": False,
        "claim_limit": CLAIM_LIMIT,
    }
    checkpoint_bytes = canonical_bytes(checkpoint)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": STATUS,
        "manifest_sha256": manifest_sha256,
        "first_step_refusal_sha256": hashlib.sha256(first_step_bytes).hexdigest(),
        "checkpoint_admission_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "closure_validation_receipt_sha256": CLOSURE_VALIDATION_SHA256,
        "historical_impact_map_sha256": IMPACT_MAP_SHA256,
        "carrier_count": 37,
        "projected_checkpoint_nbytes": CHECKPOINT_NBYTES,
        "initialization_component_set_admitted": True,
        "first_large_step_ready": False,
        "checkpoint_available": False,
        "runnable_5m_product": False,
        "claim_limit": CLAIM_LIMIT,
    }
    return {
        "checkpoint-admission.json": checkpoint_bytes,
        "first-step-refusal.json": first_step_bytes,
        "manifest.json": manifest_bytes,
        "receipt.json": canonical_bytes(receipt),
    }


def publish_envelope(
    destination: Path, audit: Mapping[str, Any], *, producer: Mapping[str, Any]
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    documents = build_envelope_documents(audit, producer=producer)
    try:
        for name, payload in documents.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, destination)
    except Exception:
        for path in sorted(staging.glob("*")):
            if path.is_file():
                path.unlink()
        if staging.exists():
            staging.rmdir()
        raise
    return inspect_envelope(destination)


def inspect_envelope(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    expected = tuple(
        sorted(
            (
                "checkpoint-admission.json",
                "first-step-refusal.json",
                "manifest.json",
                "receipt.json",
            )
        )
    )
    _require_inventory(destination, expected, "native 5 m requalification envelope")
    files = {
        name: {
            "nbytes": (destination / name).stat().st_size,
            "sha256": sha_path(destination / name),
        }
        for name in expected
    }
    manifest = json.loads((destination / "manifest.json").read_bytes())
    first_step = json.loads((destination / "first-step-refusal.json").read_bytes())
    checkpoint = json.loads((destination / "checkpoint-admission.json").read_bytes())
    receipt = json.loads((destination / "receipt.json").read_bytes())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != STATUS
        or manifest.get("runnable_5m_product") is not False
        or manifest.get("canonical_checkpoint_projection", {}).get("carrier_count") != 37
        or first_step.get("first_large_step_ready") is not False
        or checkpoint.get("initialization_boundary_admitted_as_checkpoint") is not False
        or receipt.get("status") != STATUS
        or receipt.get("manifest_sha256") != files["manifest.json"]["sha256"]
        or receipt.get("first_step_refusal_sha256")
        != files["first-step-refusal.json"]["sha256"]
        or receipt.get("checkpoint_admission_sha256")
        != files["checkpoint-admission.json"]["sha256"]
    ):
        raise ValueError("native 5 m requalification envelope failed inspection")
    return {
        "schema": f"{SCHEMA}/inspection",
        "status": "PASS_REFERENCE_ENVELOPE_INSPECTION_RK_STILL_REFUSED",
        "directory": str(destination),
        "files": files,
        "runnable_5m_product": False,
    }


__all__ = (
    "ARTIFACT_PINS",
    "CHECKPOINT_NBYTES",
    "CHECKPOINT_SCHEMA",
    "CLAIM_LIMIT",
    "CLOSURE_VALIDATION_SHA256",
    "FIRST_STEP_SCHEMA",
    "HISTORICAL_FULL_STATE_PINS",
    "IMPACT_MAP_SHA256",
    "MANIFEST_SCHEMA",
    "MISSING_LIFECYCLE_GATES",
    "Native5mEnvelopePaths",
    "RECEIPT_SCHEMA",
    "SCHEMA",
    "STATUS",
    "audit_native_5m_complete_envelope",
    "build_envelope_documents",
    "build_persistent_sources",
    "canonical_bytes",
    "default_paths",
    "inspect_envelope",
    "publish_envelope",
    "sha_path",
)
