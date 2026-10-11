from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import cm1core.gm1_child_closure_v2 as closure_v2
from cm1core.gm1_child_closure_v2 import (
    GM1_CHILD_CLOSURE_ABI,
    GM1_CHILD_CLOSURE_CARRIERS,
    GM1_CHILD_CLOSURE_CARRIER_ORDER,
    GM1_CHILD_CLOSURE_MANIFEST_SHA256,
    GM1_CHILD_CLOSURE_PAYLOAD_NBYTES,
    GM1_CHILD_CLOSURE_PAYLOAD_SHA256,
    GM1_CHILD_CLOSURE_RECEIPT_SHA256,
    GM1_CHILD_CLOSURE_SCHEMA,
    GM1_CHILD_CLOSURE_UPLOAD_SCHEMA,
    LoadedGm1ChildClosureV2,
    load_gm1_child_closure_v2,
    upload_gm1_child_closure_v2,
)
from authority_trees import requires_work


ROOT = Path(__file__).resolve().parents[1]
PRODUCT = ROOT / "work/gm1-t300-child-closure-v2"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def loaded() -> LoadedGm1ChildClosureV2:
    if not PRODUCT.is_dir():
        pytest.skip("sealed real closure-v2 publication is unavailable")
    return load_gm1_child_closure_v2(PRODUCT)


class _DeviceArray(np.ndarray):
    pass


class _FakeStream:
    def __init__(self, pointer: int = 51_270) -> None:
        self.ptr = pointer
        self.enter_count = 0
        self.exit_count = 0
        self.synchronize_count = 0

    def __enter__(self) -> _FakeStream:
        self.enter_count += 1
        return self

    def __exit__(self, *_: object) -> None:
        self.exit_count += 1

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeEvent:
    def __init__(self, *, disable_timing: bool) -> None:
        self.disable_timing = disable_timing
        self.recorded_stream: object | None = None

    def record(self, stream: object) -> None:
        self.recorded_stream = stream


class _FakeCupy:
    ndarray = _DeviceArray
    float32 = np.float32

    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.upload_count = 0
        self.cuda = SimpleNamespace(
            get_current_stream=lambda: self.stream,
            Event=_FakeEvent,
        )

    @staticmethod
    def dtype(value: object) -> np.dtype[object]:
        return np.dtype(value)

    def asarray(
        self, value: object, *, dtype: object, order: str
    ) -> _DeviceArray:
        assert order == "C"
        self.upload_count += 1
        return np.array(value, dtype=dtype, order=order, copy=True).view(_DeviceArray)


def test_import_is_cuda_cold_and_publication_constants_are_exact() -> None:
    import subprocess
    import sys

    command = (
        "import sys; sys.path.insert(0, 'src'); "
        "import cm1core.gm1_child_closure_v2 as c; "
        "assert 'cupy' not in sys.modules; print(c.GM1_CHILD_CLOSURE_PAYLOAD_NBYTES)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(GM1_CHILD_CLOSURE_PAYLOAD_NBYTES)
    assert GM1_CHILD_CLOSURE_SCHEMA == "gm1.real-50m-child-fresh-closure/v2"
    assert GM1_CHILD_CLOSURE_ABI == "gm1.real-50m-child-fresh-closure-f32/v2"
    assert tuple(carrier.name for carrier in GM1_CHILD_CLOSURE_CARRIERS) == (
        GM1_CHILD_CLOSURE_CARRIER_ORDER
    )


def test_real_closure_v2_load_is_hash_layout_and_word_exact(
    loaded: LoadedGm1ChildClosureV2,
) -> None:
    assert loaded.manifest_sha256 == GM1_CHILD_CLOSURE_MANIFEST_SHA256
    assert loaded.payload_sha256 == GM1_CHILD_CLOSURE_PAYLOAD_SHA256
    assert loaded.receipt_sha256 == GM1_CHILD_CLOSURE_RECEIPT_SHA256
    assert loaded.payload_path.stat().st_size == GM1_CHILD_CLOSURE_PAYLOAD_NBYTES
    assert _sha(loaded.payload_path) == GM1_CHILD_CLOSURE_PAYLOAD_SHA256
    assert tuple(loaded.arrays) == GM1_CHILD_CLOSURE_CARRIER_ORDER
    for carrier in GM1_CHILD_CLOSURE_CARRIERS:
        value = loaded.field(carrier.name)
        assert isinstance(value, np.memmap)
        assert value.dtype == np.dtype("float32")
        assert value.shape == carrier.shape
        assert value.flags.c_contiguous
        assert value.flags.writeable is False
        assert value.nbytes == carrier.nbytes
    assert np.array_equal(
        loaded.field("tkea").view(np.uint32),
        loaded.field("tke3d").view(np.uint32),
    )
    assert np.all(loaded.field("tketen").view(np.uint32) == np.uint32(0))
    with pytest.raises(TypeError):
        loaded.arrays["tkea"] = loaded.field("tke3d")  # type: ignore[index]
    with pytest.raises(KeyError, match="unknown GM1 child closure carrier"):
        loaded.field("missing")


@requires_work("gm1-t300-child-closure-v2")
def test_layout_validator_refuses_stale_offset_and_receipt_hash_table() -> None:
    manifest = json.loads((PRODUCT / "manifest.json").read_bytes())
    receipt = json.loads((PRODUCT / "receipt.json").read_bytes())
    manifest["carriers"][4]["offset_nbytes"] += 4
    with pytest.raises(ValueError, match="layout changed"):
        closure_v2._require_carrier_layout(manifest, receipt)
    manifest = json.loads((PRODUCT / "manifest.json").read_bytes())
    receipt["carrier_sha256"]["tkea"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 table changed"):
        closure_v2._require_carrier_layout(manifest, receipt)


@requires_work("gm1-t300-child-closure-v2")
def test_loader_refuses_noncanonical_metadata_before_payload(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "stale-metadata"
    directory.mkdir()
    shutil.copyfile(PRODUCT / "manifest.json", directory / "manifest.json")
    receipt_raw = (PRODUCT / "receipt.json").read_bytes()
    (directory / "receipt.json").write_bytes(receipt_raw + b"\n")
    with (directory / "closure.f32le").open("wb") as stream:
        stream.truncate(GM1_CHILD_CLOSURE_PAYLOAD_NBYTES)
    with pytest.raises(ValueError, match="not canonical compact JSON"):
        load_gm1_child_closure_v2(directory)


@requires_work("gm1-t300-child-closure-v2")
def test_loader_refuses_corrupt_full_length_payload(tmp_path: Path) -> None:
    directory = tmp_path / "corrupt-payload"
    directory.mkdir()
    shutil.copyfile(PRODUCT / "manifest.json", directory / "manifest.json")
    shutil.copyfile(PRODUCT / "receipt.json", directory / "receipt.json")
    with (directory / "closure.f32le").open("wb") as stream:
        stream.truncate(GM1_CHILD_CLOSURE_PAYLOAD_NBYTES)
    with pytest.raises(ValueError, match="payload SHA-256 differs"):
        load_gm1_child_closure_v2(directory)


def test_loader_refuses_extra_inventory_without_reading_files(tmp_path: Path) -> None:
    directory = tmp_path / "extra-inventory"
    directory.mkdir()
    for name in ("manifest.json", "receipt.json", "closure.f32le", "extra"):
        (directory / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="directory inventory mismatch"):
        load_gm1_child_closure_v2(directory)


def test_upload_refuses_stale_loaded_identity_before_import(
    loaded: LoadedGm1ChildClosureV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = replace(loaded, payload_sha256="0" * 64)

    def fail_import(_: str) -> object:
        raise AssertionError("CUDA import must stay cold for stale closure identity")

    monkeypatch.setattr(closure_v2.importlib, "import_module", fail_import)
    with pytest.raises(ValueError, match="loaded identity"):
        upload_gm1_child_closure_v2(stale)


def test_upload_enqueues_all_carriers_on_supplied_stream_without_sync(
    loaded: LoadedGm1ChildClosureV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _FakeStream()
    cupy = _FakeCupy(stream)
    original_import = importlib.import_module

    def import_module(name: str) -> object:
        if name == "cupy":
            return cupy
        return original_import(name)

    monkeypatch.setattr(closure_v2.importlib, "import_module", import_module)
    result = upload_gm1_child_closure_v2(loaded, stream=stream)
    assert tuple(result.arrays) == GM1_CHILD_CLOSURE_CARRIER_ORDER
    assert cupy.upload_count == 13
    assert stream.enter_count == stream.exit_count == 1
    assert stream.synchronize_count == 0
    assert result.ready_event.disable_timing is True
    assert result.ready_event.recorded_stream is stream
    assert result.receipt["schema"] == GM1_CHILD_CLOSURE_UPLOAD_SCHEMA
    assert result.receipt["same_stream_no_host_sync"] is True
    assert result.receipt["mapping_sealed_device_storage_writable"] is True
    assert result.receipt["stream_pointer"] == stream.ptr
    for carrier in GM1_CHILD_CLOSURE_CARRIERS:
        actual = result.arrays[carrier.name]
        expected = loaded.arrays[carrier.name]
        assert isinstance(actual, _DeviceArray)
        assert actual.shape == carrier.shape
        assert actual.flags.c_contiguous
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    with pytest.raises(TypeError):
        result.arrays["tkea"] = result.arrays["tke3d"]  # type: ignore[index]
