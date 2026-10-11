from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

import cm1core.cuda_real_initialization as implementation
from cm1core.cuda_real_initialization import (
    CARRIER_NAMES,
    CUDA_REAL_TEST_INITIALIZATION_SCHEMA,
    initialize_cuda_real_carriers,
    load_cuda_live_export,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARTIFACT_SHA = "aa" * 32
SOURCE_CONTRACT_SHA = "bb" * 32


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def shapes(nz: int, ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    return {
        **{name: (nz,) for name in CARRIER_NAMES[:7]},
        "ua": (nz, ny, nx + 1),
        "u3d": (nz, ny, nx + 1),
        "va": (nz, ny + 1, nx),
        "v3d": (nz, ny + 1, nx),
        "wa": (nz + 1, ny, nx),
        "w3d": (nz + 1, ny, nx),
        "ppi": (nz, ny, nx),
        "pp3d": (nz, ny, nx),
        "ppx": (nz, ny, nx),
        "tha": (nz, ny, nx),
        "th3d": (nz, ny, nx),
        "qa": (6, nz, ny, nx),
        "q3d": (6, nz, ny, nx),
        "terrain_msl": (ny, nx),
    }


def write_export(
    root: Path,
    *,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
) -> tuple[str, str]:
    root.mkdir()
    nz, ny, nx = (2, 2, 3)
    field_shapes = shapes(nz, ny, nx)
    source_seed: dict[str, int] = {}
    blocks: dict[str, bytes] = {}
    for index, name in enumerate(CARRIER_NAMES):
        duplicate = implementation._DUPLICATES.get(name)
        if duplicate is not None:
            seed = source_seed[duplicate]
        else:
            seed = index + 1
        source_seed[name] = seed
        words = 1
        for extent in field_shapes[name]:
            words *= extent
        blocks[name] = struct.pack("<f", seed / 32.0) * words

    offset = 0
    carriers = []
    payload = bytearray()
    for name in CARRIER_NAMES:
        block = blocks[name]
        word_bits = struct.unpack("<I", block[:4])[0]
        words = len(block) // 4
        carriers.append(
            {
                "duplicate_of": implementation._DUPLICATES.get(name),
                "dtype": "<f4",
                "itemsize": 4,
                "name": name,
                "nbytes": len(block),
                "offset_nbytes": offset,
                "order": "C",
                "role": f"test-{name}",
                "shape": list(field_shapes[name]),
                "sha256": hashlib.sha256(block).hexdigest(),
                "source_arrays": list(implementation._SOURCE_ARRAYS[name]),
                "species_order": (
                    list(implementation.SPECIES_ORDER)
                    if name in {"qa", "q3d"}
                    else []
                ),
                "statistics": {
                    "finite_count": words,
                    "maximum_f32_bits": f"{word_bits:08x}",
                    "minimum_f32_bits": f"{word_bits:08x}",
                    "nonzero_count": words,
                },
            }
        )
        payload.extend(block)
        offset += len(block)
    payload_bytes = bytes(payload)
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    manifest: dict[str, object] = {
        "admission": "real-product-300x300x75",
        "carrier_abi": implementation.CUDA_REAL_CARRIER_ABI,
        "carriers": carriers,
        "claim_limit": implementation.CUDA_REAL_EXPORT_CLAIM_LIMIT,
        "crop": {
            "count_zyx": [nz, ny, nx],
            "origin_zyx": [0, 10, 20],
            "stagger_rule": "mass crop plus one terminal face on each staggered axis",
        },
        "dtype": "<f4",
        "itemsize": 4,
        "order": "C",
        "payload": {
            "file": implementation.PAYLOAD_NAME,
            "nbytes": len(payload_bytes),
            "sha256": payload_sha,
        },
        "schema": implementation.CUDA_REAL_EXPORT_SCHEMA,
        "source": {
            "artifact_nbytes": 1234,
            "artifact_sha256": SOURCE_ARTIFACT_SHA,
            "contract_sha256": SOURCE_CONTRACT_SHA,
            "dt_f64_bits": "3fe0000000000000",
            "model_time_f64_bits": "0000000000000000",
            "profile_id": "hrrr-cm1-real-state-v1",
            "step": 0,
        },
        "source_grid": {
            "shape_zyx": [75, 300, 300],
            "target_dx_m_f64_bits": f"{struct.unpack('<Q', struct.pack('<d', 100.0))[0]:016x}",
            "target_dy_m_f64_bits": f"{struct.unpack('<Q', struct.pack('<d', 100.0))[0]:016x}",
            "target_dz_m_f64_bits": f"{struct.unpack('<Q', struct.pack('<d', 200.0))[0]:016x}",
        },
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_bytes = canonical(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    receipt = {
        "admission": "real-product-300x300x75",
        "carrier_abi": implementation.CUDA_REAL_CARRIER_ABI,
        "carrier_count": len(carriers),
        "claim_limit": implementation.CUDA_REAL_EXPORT_CLAIM_LIMIT,
        "manifest_nbytes": len(manifest_bytes),
        "manifest_sha256": manifest_sha,
        "payload_nbytes": len(payload_bytes),
        "payload_sha256": payload_sha,
        "publication_contract": "test atomic publication",
        "schema": implementation.CUDA_REAL_EXPORT_RECEIPT_SCHEMA,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA,
    }
    (root / implementation.PAYLOAD_NAME).write_bytes(payload_bytes)
    (root / implementation.MANIFEST_NAME).write_bytes(manifest_bytes)
    (root / implementation.RECEIPT_NAME).write_bytes(canonical(receipt))
    return manifest_sha, payload_sha


def load(root: Path, manifest_sha: str, payload_sha: str):
    return load_cuda_live_export(
        root,
        expected_manifest_sha256=manifest_sha,
        expected_payload_sha256=payload_sha,
        expected_source_artifact_sha256=SOURCE_ARTIFACT_SHA,
        expected_source_contract_sha256=SOURCE_CONTRACT_SHA,
    )


def test_import_is_cuda_and_numpy_cold() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = (
        "import sys; import cm1core.cuda_real_initialization; "
        "assert 'cupy' not in sys.modules; assert 'numpy' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_strict_loader_and_fake_driver_preserve_opaque_carrier_contract(
    tmp_path: Path,
) -> None:
    manifest_sha, payload_sha = write_export(tmp_path / "export")
    export = load(tmp_path / "export", manifest_sha, payload_sha)
    assert export.crop_count_zyx == (2, 2, 3)
    assert not export.is_full_domain
    assert tuple(carrier.name for carrier in export.carriers) == CARRIER_NAMES
    for name, source in implementation._DUPLICATES.items():
        assert export.carrier(name).sha256 == export.carrier(source).sha256
        assert export.carrier(name).offset_nbytes != export.carrier(source).offset_nbytes

    class FakeDriver:
        def allocate_and_copy(self, loaded):
            assert loaded.payload_path.read_bytes()
            return implementation._DriverAllocation(
                arena=object(),
                fields=tuple((name, object()) for name in CARRIER_NAMES),
                ready_event=object(),
                stream=object(),
                pinned_host=b"retained",
                device_id=7,
                stream_pointer=17,
                event_pointer=23,
                runtime_identity=(("driver", "fake"),),
            )

    sys.modules.pop("cupy", None)
    state = initialize_cuda_real_carriers(
        export,
        allow_cropped_validation=True,
        driver_factory=FakeDriver,
    )
    assert "cupy" not in sys.modules
    assert state.schema == CUDA_REAL_TEST_INITIALIZATION_SCHEMA
    assert state.audited_default_driver is False
    assert state.field("ppx") is not state.field("ppi")
    receipt = state.receipt_dict()
    assert receipt["crop"]["full_domain"] is False
    assert receipt["arena"]["payload_sha256"] == payload_sha
    json.dumps(receipt, allow_nan=False, sort_keys=True)


def test_cropped_export_requires_explicit_validation_admission(tmp_path: Path) -> None:
    manifest_sha, payload_sha = write_export(tmp_path / "export")
    export = load(tmp_path / "export", manifest_sha, payload_sha)
    with pytest.raises(ValueError, match="validation-only"):
        initialize_cuda_real_carriers(export, driver_factory=lambda: object())
    assert "cupy" not in sys.modules


def test_manifest_payload_inventory_and_canonical_tamper_controls(tmp_path: Path) -> None:
    root = tmp_path / "export"
    manifest_sha, payload_sha = write_export(root)

    original = (root / implementation.MANIFEST_NAME).read_bytes()
    (root / implementation.MANIFEST_NAME).write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        load(root, manifest_sha, payload_sha)
    (root / implementation.MANIFEST_NAME).write_bytes(original)

    payload = bytearray((root / implementation.PAYLOAD_NAME).read_bytes())
    payload[-1] ^= 1
    (root / implementation.PAYLOAD_NAME).write_bytes(payload)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch|aggregate SHA-256"):
        load(root, manifest_sha, payload_sha)
    payload[-1] ^= 1
    (root / implementation.PAYLOAD_NAME).write_bytes(payload)

    (root / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        load(root, manifest_sha, payload_sha)


def test_rehashed_semantic_offset_and_species_drift_still_refuse(tmp_path: Path) -> None:
    def alias_current(manifest: dict[str, object]) -> None:
        carriers = manifest["carriers"]
        carriers[8]["offset_nbytes"] = carriers[7]["offset_nbytes"]

    alias_root = tmp_path / "alias"
    manifest_sha, payload_sha = write_export(alias_root, mutate_manifest=alias_current)
    with pytest.raises(ValueError, match="offset is not contiguous|aliases"):
        load(alias_root, manifest_sha, payload_sha)

    def reorder_species(manifest: dict[str, object]) -> None:
        carriers = manifest["carriers"]
        carriers[18]["species_order"] = ["qc", "qv", "qr", "qi", "qs", "qg"]

    species_root = tmp_path / "species"
    manifest_sha, payload_sha = write_export(species_root, mutate_manifest=reorder_species)
    with pytest.raises(ValueError, match="wrong semantic species order"):
        load(species_root, manifest_sha, payload_sha)


def test_independent_hashes_are_mandatory_and_checked_before_cuda(tmp_path: Path) -> None:
    manifest_sha, payload_sha = write_export(tmp_path / "export")
    with pytest.raises(ValueError, match="independently expected"):
        load(tmp_path / "export", "cc" * 32, payload_sha)
    with pytest.raises(ValueError, match="independent SHA-256"):
        load(tmp_path / "export", manifest_sha, "dd" * 32)
    assert "cupy" not in sys.modules


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda allocation: replace(
                allocation, fields=tuple(reversed(allocation.fields))
            ),
            "wrong carrier view inventory/order",
        ),
        (
            lambda allocation: replace(allocation, arena=None),
            "null arena/event/stream handles",
        ),
        (
            lambda allocation: replace(allocation, event_pointer=0),
            "invalid device/stream/event metadata",
        ),
        (
            lambda allocation: replace(allocation, runtime_identity=()),
            "no runtime identity",
        ),
    ],
)
def test_fake_driver_cannot_bypass_production_handle_contract(
    tmp_path: Path,
    change: Callable[
        [implementation._DriverAllocation], implementation._DriverAllocation
    ],
    message: str,
) -> None:
    manifest_sha, payload_sha = write_export(tmp_path / "export")
    export = load(tmp_path / "export", manifest_sha, payload_sha)
    baseline = implementation._DriverAllocation(
        arena=object(),
        fields=tuple((name, object()) for name in CARRIER_NAMES),
        ready_event=object(),
        stream=object(),
        pinned_host=b"retained",
        device_id=0,
        stream_pointer=0,
        event_pointer=1,
        runtime_identity=(("driver", "fake"),),
    )

    class FakeDriver:
        def allocate_and_copy(self, _loaded):
            return change(baseline)

    with pytest.raises(ValueError, match=message):
        initialize_cuda_real_carriers(
            export,
            allow_cropped_validation=True,
            driver_factory=FakeDriver,
        )
    assert "cupy" not in sys.modules


def test_guarded_sm86_evidence_is_byte_exact_and_claim_bounded() -> None:
    path = (
        ROOT
        / "evidence"
        / "validation"
        / "cuda-real-initialization-v1"
        / "cuda-real-initialization-sm86.json"
    )
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "1a355c2c3c8413cc280915173d958a0c1738ff2d62e374960a4cf365defa9970"
    )
    evidence = json.loads(raw)
    assert evidence["result"] == "passed"
    assert evidence["validation"] == {
        "all_21_named_carrier_hashes_equal_rust_manifest": True,
        "arena_sha256_equal_rust_payload": True,
        "dtype_and_order": "Rust-described <f4 C-order views over one uint8 arena",
        "production_state_schema": True,
        "python_scientific_transforms": 0,
    }
    source = evidence["source_export"]
    assert source["carrier_count"] == 21
    assert source["payload_nbytes"] == 622_442_100
    assert source["payload_sha256"] == (
        "1bba92c10ba82cf2817ea2e8caf1052a5e5c389e8eaf2e3edd01fbf4b5e8cfb0"
    )
    cuda = evidence["cuda_initialization"]
    assert cuda["runtime_identity"]["device_name"] == "NVIDIA GeForce RTX 3080"
    assert cuda["runtime_identity"]["compute_capability"] == "86"
    assert cuda["arena"]["nbytes"] == source["payload_nbytes"]
    assert cuda["arena"]["payload_sha256"] == source["payload_sha256"]
    manifest_carriers = {carrier["name"]: carrier for carrier in cuda["carriers"]}
    assert set(evidence["carrier_readback"]) == set(manifest_carriers)
    for name, readback in evidence["carrier_readback"].items():
        assert readback["sha256"] == manifest_carriers[name]["sha256"]
        assert readback["nbytes"] == manifest_carriers[name]["nbytes"]
        assert readback["shape"] == manifest_carriers[name]["shape"]
    assert evidence["lock_protocol"]["clean_release"] is True
    assert evidence["lock_protocol"]["ownership_rechecked_before_release"] is True
    assert "dynamics, physics, trajectory, forecast, tornado, or model validation" in (
        evidence["claim_boundary"]["excluded"]
    )


def test_a_parent_published_at_another_domain_is_transported_not_refused(
    tmp_path: Path,
) -> None:
    """The transport reads the parent domain out of the export's own manifest.

    A larger parent (the extent a translating-frame run actually needs) used to
    be refused by a fixed 75x300x300 comparison even though every carrier,
    offset and hash in the export was self-consistent.
    """

    root = tmp_path / "live"

    def widen(manifest: dict) -> None:
        manifest["source_grid"]["shape_zyx"] = [75, 600, 600]

    manifest_sha, payload_sha = write_export(root, mutate_manifest=widen)
    export = load(root, manifest_sha, payload_sha)
    assert export.source_shape_zyx == (75, 600, 600)
    assert export.source_spacing_m == (100.0, 100.0, 200.0)
    # The crop is a strict subset of the wider parent, so it is not full-domain.
    assert export.is_full_domain is False


def test_a_crop_outside_the_recorded_parent_domain_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "live"

    def shrink(manifest: dict) -> None:
        manifest["source_grid"]["shape_zyx"] = [75, 12, 12]

    manifest_sha, payload_sha = write_export(root, mutate_manifest=shrink)
    with pytest.raises(ValueError) as error:
        load(root, manifest_sha, payload_sha)
    assert "the parent domain does not contain" in str(error.value)


def test_the_declared_source_grid_reaches_the_published_receipt(tmp_path: Path) -> None:
    """The spacing is decoded, carried, and consumed - not merely stored.

    `tools/run_cuda_real_initialization.py` stamps exactly this record into its
    evidence receipt, so an export at a different vertical spacing publishes
    that spacing rather than a constant the transport used to assume.
    """

    root = tmp_path / "live"

    def fifty_metre_dz(manifest: dict) -> None:
        manifest["source_grid"]["target_dz_m_f64_bits"] = (
            f"{struct.unpack('<Q', struct.pack('<d', 50.0))[0]:016x}"
        )

    manifest_sha, payload_sha = write_export(root, mutate_manifest=fifty_metre_dz)
    export = load(root, manifest_sha, payload_sha)
    assert export.source_spacing_m == (100.0, 100.0, 50.0)
    assert export.source_grid_record() == {
        "shape_zyx": [75, 300, 300],
        "dx_m": 100.0,
        "dy_m": 100.0,
        "dz_m": 50.0,
    }
    runner = ROOT / "tools" / "run_cuda_real_initialization.py"
    assert '"source_grid": export.source_grid_record()' in runner.read_text()


def test_a_nonsense_grid_spacing_is_refused_by_value(tmp_path: Path) -> None:
    root = tmp_path / "live"

    def zero_dz(manifest: dict) -> None:
        manifest["source_grid"]["target_dz_m_f64_bits"] = "0000000000000000"

    manifest_sha, payload_sha = write_export(root, mutate_manifest=zero_dz)
    with pytest.raises(ValueError) as error:
        load(root, manifest_sha, payload_sha)
    assert "non-positive grid spacing" in str(error.value)
