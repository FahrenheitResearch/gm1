"""Strict mmap and same-stream CUDA upload for the sealed GM1 closure-v2."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


GM1_CHILD_CLOSURE_SCHEMA = "gm1.real-50m-child-fresh-closure/v2"
GM1_CHILD_CLOSURE_MANIFEST_SCHEMA = (
    "gm1.real-50m-child-fresh-closure-manifest/v2"
)
GM1_CHILD_CLOSURE_RECEIPT_SCHEMA = (
    "gm1.real-50m-child-fresh-closure-receipt/v2"
)
GM1_CHILD_CLOSURE_ABI = "gm1.real-50m-child-fresh-closure-f32/v2"
GM1_CHILD_CLOSURE_UPLOAD_SCHEMA = "gm1.cuda-real-50m-child-fresh-closure/v2"
GM1_CHILD_CLOSURE_STATUS = "PASS_REAL_CHILD_GENERIC_FRESH_TKE_RAW_BIT_EXACT"
GM1_CHILD_CLOSURE_MANIFEST_SHA256 = (
    "7aba693cd3a5c984c5a0d969bffa0793f80dfea138aa661da34a86f69982d9d9"
)
GM1_CHILD_CLOSURE_PAYLOAD_SHA256 = (
    "dc277233d7bead2009749e09f402fdff6c072323a57568fef270dbe6904dca6c"
)
GM1_CHILD_CLOSURE_RECEIPT_SHA256 = (
    "9972f6e981dc99e5635a844b70906328339535450bc57cdb01e61469b8080985"
)
GM1_CHILD_CLOSURE_PAYLOAD_NBYTES = 355_320_000
GM1_CHILD_CLOSURE_CLAIM_LIMIT = (
    "strict hash-bound mmap and same-stream CUDA upload of the one sealed real "
    "300x300x75 fresh-child closure-v2 publication; the mapping is immutable "
    "but uploaded carrier storage is writable for the admitted time integrator; "
    "no recomputation, stale artifact, alternate grid, time evolution, forecast, "
    "storm, tornado, or performance claim"
)
GM1_CHILD_CLOSURE_CARRIER_ORDER = (
    "nm",
    "defv",
    "defh",
    "grdscl",
    "lenscl",
    "dissten",
    "kmh",
    "kmv",
    "khh",
    "khv",
    "tkea",
    "tke3d",
    "tketen",
)

_FILES = frozenset({"closure.f32le", "manifest.json", "receipt.json"})
_MANIFEST_KEYS = {
    "abi",
    "artifact_schema",
    "authority",
    "carriers",
    "claim_limit",
    "disposition",
    "fixture_class",
    "grid",
    "inputs",
    "model",
    "operator_receipts",
    "payload",
    "producer",
    "schema",
    "status",
}
_RECEIPT_KEYS = {
    "abi",
    "artifact_schema",
    "authority",
    "carrier_count",
    "carrier_order",
    "carrier_sha256",
    "claim_limit",
    "inputs",
    "manifest_sha256",
    "model",
    "payload_nbytes",
    "payload_sha256",
    "producer",
    "schema",
    "status",
}
_CARRIER_KEYS = {
    "dtype",
    "finite_count",
    "maximum_f32_bits",
    "minimum_f32_bits",
    "name",
    "nbytes",
    "nonzero_count",
    "offset_nbytes",
    "order",
    "positive_zero_count",
    "sha256",
    "shape",
}


@dataclass(frozen=True, slots=True)
class Gm1ChildClosureCarrier:
    name: str
    shape: tuple[int, int, int]
    offset_nbytes: int
    nbytes: int
    sha256: str
    finite_count: int
    nonzero_count: int
    positive_zero_count: int
    minimum_f32_bits: str
    maximum_f32_bits: str


def _carrier(
    name: str,
    shape: tuple[int, int, int],
    offset_nbytes: int,
    sha256: str,
    nonzero_count: int,
    positive_zero_count: int,
    minimum_f32_bits: str,
    maximum_f32_bits: str,
) -> Gm1ChildClosureCarrier:
    count = shape[0] * shape[1] * shape[2]
    return Gm1ChildClosureCarrier(
        name=name,
        shape=shape,
        offset_nbytes=offset_nbytes,
        nbytes=count * 4,
        sha256=sha256,
        finite_count=count,
        nonzero_count=nonzero_count,
        positive_zero_count=positive_zero_count,
        minimum_f32_bits=minimum_f32_bits,
        maximum_f32_bits=maximum_f32_bits,
    )


_FACE = (76, 300, 300)
GM1_CHILD_CLOSURE_CARRIERS = (
    _carrier(
        "nm",
        _FACE,
        0,
        "1331c1feb09033a98ee5ce14cada19c3870dd9a8ded083ecb4c5d0ec3fe20802",
        6_659_862,
        180_138,
        "b9fb2bc6",
        "3b3fa3d0",
    ),
    _carrier(
        "defv",
        _FACE,
        27_360_000,
        "d7f044de8f9dea821f3f71616d64c5cdc770617ec92c189d8f46936d3dc05a78",
        6_660_000,
        180_000,
        "00000000",
        "3e2bdad1",
    ),
    _carrier(
        "defh",
        _FACE,
        54_720_000,
        "0ba40b06156908886c5eb6560e9030c19bc36a37913c3f50f1b84735c948d24d",
        6_660_000,
        180_000,
        "00000000",
        "3db53c27",
    ),
    _carrier(
        "grdscl",
        (75, 300, 300),
        82_080_000,
        "efac27072598bb0d437f27e9170972139c5ed5aa28d0aa87bcc9a62fa32bfd65",
        6_660_000,
        90_000,
        "00000000",
        "42b8646f",
    ),
    _carrier(
        "lenscl",
        _FACE,
        109_080_000,
        "b78ec35f2896d2410a5fad43903aef2d2367d0596f804e7f79b57aade201518f",
        6_660_000,
        180_000,
        "00000000",
        "42b8646f",
    ),
    _carrier(
        "dissten",
        _FACE,
        136_440_000,
        "180b5d7ccfeb7e17e5d9f623f1304edae862058198ad7b91d7c30d6e425a6303",
        6_660_000,
        180_000,
        "00000000",
        "4197ef71",
    ),
    _carrier(
        "kmh",
        _FACE,
        163_800_000,
        "1697ce2a2067bec75e4ae849333c98f96857b9111a27f214f3c5fc779107d85c",
        6_660_000,
        180_000,
        "00000000",
        "42df3789",
    ),
    _carrier(
        "kmv",
        _FACE,
        191_160_000,
        "1697ce2a2067bec75e4ae849333c98f96857b9111a27f214f3c5fc779107d85c",
        6_660_000,
        180_000,
        "00000000",
        "42df3789",
    ),
    _carrier(
        "khh",
        _FACE,
        218_520_000,
        "df37293bb9ade9b54c697befbc466cb248aaf7faf8cace761c5e2ef18c8ca820",
        6_660_000,
        180_000,
        "00000000",
        "43a769a7",
    ),
    _carrier(
        "khv",
        _FACE,
        245_880_000,
        "df37293bb9ade9b54c697befbc466cb248aaf7faf8cace761c5e2ef18c8ca820",
        6_660_000,
        180_000,
        "00000000",
        "43a769a7",
    ),
    _carrier(
        "tkea",
        _FACE,
        273_240_000,
        "6dfa15585027291ab71fa59a73303ac2d8e754cd8f624171fe140d6db0b0ea2a",
        6_750_000,
        90_000,
        "00000000",
        "43128c90",
    ),
    _carrier(
        "tke3d",
        _FACE,
        300_600_000,
        "6dfa15585027291ab71fa59a73303ac2d8e754cd8f624171fe140d6db0b0ea2a",
        6_750_000,
        90_000,
        "00000000",
        "43128c90",
    ),
    _carrier(
        "tketen",
        _FACE,
        327_960_000,
        "9b6fc14170dd44e635d8834923372483516e2612bf8ecd79fbc6e87c205ace05",
        0,
        6_840_000,
        "00000000",
        "00000000",
    ),
)
_CARRIER_BY_NAME = MappingProxyType(
    {carrier.name: carrier for carrier in GM1_CHILD_CLOSURE_CARRIERS}
)


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildClosureV2:
    directory: Path
    manifest_path: Path
    payload_path: Path
    receipt_path: Path
    manifest_sha256: str
    payload_sha256: str
    receipt_sha256: str
    arrays: Mapping[str, Any]
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def field(self, name: str) -> Any:
        try:
            return self.arrays[name]
        except KeyError as error:
            raise KeyError(f"unknown GM1 child closure carrier: {name}") from error


@dataclass(frozen=True, slots=True)
class CudaGm1ChildClosureV2:
    ready_event: Any
    source: LoadedGm1ChildClosureV2
    arrays: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_range(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("closure payload ended inside a carrier")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _read_canonical(path: Path, label: str) -> tuple[bytes, Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if not raw or b"\r" in raw:
        raise ValueError(f"{label} bytes are empty or non-canonical")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    canonical = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    if raw != canonical:
        raise ValueError(f"{label} is not canonical compact JSON")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return raw, value


def _exact_keys(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"{label} keys changed: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _require_carrier_layout(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    carriers = manifest.get("carriers")
    if not isinstance(carriers, list) or len(carriers) != 13:
        raise ValueError("closure manifest must contain exactly 13 carriers")
    expected_end = 0
    for index, (record, expected) in enumerate(
        zip(carriers, GM1_CHILD_CLOSURE_CARRIERS, strict=True)
    ):
        record = _exact_keys(record, _CARRIER_KEYS, f"closure carrier {index}")
        expected_record = {
            "dtype": "little-endian-f32",
            "finite_count": expected.finite_count,
            "maximum_f32_bits": expected.maximum_f32_bits,
            "minimum_f32_bits": expected.minimum_f32_bits,
            "name": expected.name,
            "nbytes": expected.nbytes,
            "nonzero_count": expected.nonzero_count,
            "offset_nbytes": expected.offset_nbytes,
            "order": "C",
            "positive_zero_count": expected.positive_zero_count,
            "sha256": expected.sha256,
            "shape": list(expected.shape),
        }
        if record != expected_record or expected.offset_nbytes != expected_end:
            raise ValueError(f"closure carrier layout changed at index {index}")
        expected_end += expected.nbytes
    if expected_end != GM1_CHILD_CLOSURE_PAYLOAD_NBYTES:
        raise RuntimeError("internal closure carrier layout length changed")
    if receipt.get("carrier_count") != 13 or receipt.get("carrier_order") != list(
        GM1_CHILD_CLOSURE_CARRIER_ORDER
    ):
        raise ValueError("closure receipt carrier count or order changed")
    expected_hashes = {
        carrier.name: carrier.sha256 for carrier in GM1_CHILD_CLOSURE_CARRIERS
    }
    if receipt.get("carrier_sha256") != expected_hashes:
        raise ValueError("closure receipt carrier SHA-256 table changed")


def load_gm1_child_closure_v2(
    directory: str | os.PathLike[str],
) -> LoadedGm1ChildClosureV2:
    """Open the one sealed closure-v2 publication as read-only mmap views."""

    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("closure-v2 directory must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("closure-v2 publication must be a real directory")
    inventory = {entry.name for entry in os.scandir(root)}
    if inventory != _FILES:
        raise ValueError(
            "closure-v2 directory inventory mismatch; "
            f"missing={sorted(_FILES - inventory)}, extra={sorted(inventory - _FILES)}"
        )
    manifest_path = root / "manifest.json"
    payload_path = root / "closure.f32le"
    receipt_path = root / "receipt.json"
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("closure-v2 payload must be a regular non-symlink file")

    manifest_raw, manifest = _read_canonical(manifest_path, "closure-v2 manifest")
    receipt_raw, receipt = _read_canonical(receipt_path, "closure-v2 receipt")
    if hashlib.sha256(manifest_raw).hexdigest() != GM1_CHILD_CLOSURE_MANIFEST_SHA256:
        raise ValueError("closure-v2 manifest SHA-256 differs from the sealed identity")
    if hashlib.sha256(receipt_raw).hexdigest() != GM1_CHILD_CLOSURE_RECEIPT_SHA256:
        raise ValueError("closure-v2 receipt SHA-256 differs from the sealed identity")
    if payload_path.stat().st_size != GM1_CHILD_CLOSURE_PAYLOAD_NBYTES:
        raise ValueError("closure-v2 payload byte length changed")
    if _sha_path(payload_path) != GM1_CHILD_CLOSURE_PAYLOAD_SHA256:
        raise ValueError("closure-v2 payload SHA-256 differs from the sealed identity")

    manifest = _exact_keys(manifest, _MANIFEST_KEYS, "closure-v2 manifest")
    receipt = _exact_keys(receipt, _RECEIPT_KEYS, "closure-v2 receipt")
    payload = _exact_keys(
        manifest.get("payload"),
        {"carrier_count", "dtype", "file", "nbytes", "order", "sha256"},
        "closure-v2 payload descriptor",
    )
    grid = _exact_keys(
        manifest.get("grid"), {"dx_m", "dy_m", "dz_m", "shape_zyx"}, "closure-v2 grid"
    )
    if (
        manifest.get("schema") != GM1_CHILD_CLOSURE_MANIFEST_SCHEMA
        or manifest.get("artifact_schema") != GM1_CHILD_CLOSURE_SCHEMA
        or manifest.get("abi") != GM1_CHILD_CLOSURE_ABI
        or manifest.get("status") != GM1_CHILD_CLOSURE_STATUS
        or manifest.get("model") != "GM1"
        or manifest.get("fixture_class")
        != "real accepted pdefq-clean T+300 f02 child state; no synthetic data"
        or grid
        != {"dx_m": 50.0, "dy_m": 50.0, "dz_m": 100.0, "shape_zyx": [75, 300, 300]}
        or payload
        != {
            "carrier_count": 13,
            "dtype": "little-endian-f32",
            "file": "closure.f32le",
            "nbytes": GM1_CHILD_CLOSURE_PAYLOAD_NBYTES,
            "order": "C; carrier-major in manifest order",
            "sha256": GM1_CHILD_CLOSURE_PAYLOAD_SHA256,
        }
        or receipt.get("schema") != GM1_CHILD_CLOSURE_RECEIPT_SCHEMA
        or receipt.get("artifact_schema") != GM1_CHILD_CLOSURE_SCHEMA
        or receipt.get("abi") != GM1_CHILD_CLOSURE_ABI
        or receipt.get("status") != GM1_CHILD_CLOSURE_STATUS
        or receipt.get("model") != "GM1"
        or receipt.get("manifest_sha256") != GM1_CHILD_CLOSURE_MANIFEST_SHA256
        or receipt.get("payload_sha256") != GM1_CHILD_CLOSURE_PAYLOAD_SHA256
        or receipt.get("payload_nbytes") != GM1_CHILD_CLOSURE_PAYLOAD_NBYTES
        or receipt.get("authority") != manifest.get("authority")
        or receipt.get("inputs") != manifest.get("inputs")
        or receipt.get("producer") != manifest.get("producer")
        or receipt.get("claim_limit") != manifest.get("claim_limit")
    ):
        raise ValueError("closure-v2 schema, grid, source binding, or identity changed")
    _require_carrier_layout(manifest, receipt)

    np = importlib.import_module("numpy")
    arrays: dict[str, Any] = {}
    for carrier in GM1_CHILD_CLOSURE_CARRIERS:
        if _sha_range(payload_path, carrier.offset_nbytes, carrier.nbytes) != carrier.sha256:
            raise ValueError(f"closure-v2 carrier {carrier.name} SHA-256 changed")
        array = np.memmap(
            payload_path,
            dtype="<f4",
            mode="r",
            offset=carrier.offset_nbytes,
            shape=carrier.shape,
            order="C",
        )
        bits = array.view(np.uint32)
        if (
            array.dtype != np.dtype("float32")
            or array.shape != carrier.shape
            or not array.flags.c_contiguous
            or array.flags.writeable
            or not bool(np.all(np.isfinite(array)))
            or int(np.count_nonzero(array)) != carrier.nonzero_count
            or int(np.count_nonzero(bits == np.uint32(0))) != carrier.positive_zero_count
            or f"{int(np.min(array).view(np.uint32)):08x}" != carrier.minimum_f32_bits
            or f"{int(np.max(array).view(np.uint32)):08x}" != carrier.maximum_f32_bits
        ):
            raise ValueError(f"closure-v2 carrier {carrier.name} words or layout changed")
        arrays[carrier.name] = array
    if not np.array_equal(
        arrays["tkea"].view(np.uint32), arrays["tke3d"].view(np.uint32)
    ) or not bool(np.all(arrays["tketen"].view(np.uint32) == np.uint32(0))):
        raise ValueError("closure-v2 TKE copy/reset disposition changed")

    return LoadedGm1ChildClosureV2(
        directory=root,
        manifest_path=manifest_path,
        payload_path=payload_path,
        receipt_path=receipt_path,
        manifest_sha256=GM1_CHILD_CLOSURE_MANIFEST_SHA256,
        payload_sha256=GM1_CHILD_CLOSURE_PAYLOAD_SHA256,
        receipt_sha256=GM1_CHILD_CLOSURE_RECEIPT_SHA256,
        arrays=MappingProxyType(arrays),
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(receipt)),
    )


def upload_gm1_child_closure_v2(
    closure: LoadedGm1ChildClosureV2, *, stream: Any | None = None
) -> CudaGm1ChildClosureV2:
    """Enqueue all 13 sealed host carriers on one supplied CUDA stream."""

    if not isinstance(closure, LoadedGm1ChildClosureV2):
        raise TypeError("closure must be LoadedGm1ChildClosureV2")
    if (
        closure.manifest_sha256 != GM1_CHILD_CLOSURE_MANIFEST_SHA256
        or closure.payload_sha256 != GM1_CHILD_CLOSURE_PAYLOAD_SHA256
        or closure.receipt_sha256 != GM1_CHILD_CLOSURE_RECEIPT_SHA256
        or tuple(closure.arrays) != GM1_CHILD_CLOSURE_CARRIER_ORDER
    ):
        raise ValueError("closure-v2 loaded identity or carrier inventory changed")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    device_arrays: dict[str, Any] = {}
    with stream:
        for carrier in GM1_CHILD_CLOSURE_CARRIERS:
            host = closure.arrays[carrier.name]
            value = cupy.asarray(host, dtype=cupy.float32, order="C")
            if (
                not isinstance(value, cupy.ndarray)
                or value.dtype != cupy.dtype(cupy.float32)
                or value.shape != carrier.shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"CUDA closure carrier {carrier.name} layout changed")
            device_arrays[carrier.name] = value
        ready = cupy.cuda.Event(disable_timing=True)
        ready.record(stream)
    receipt = MappingProxyType(
        {
            "schema": GM1_CHILD_CLOSURE_UPLOAD_SCHEMA,
            "status": "ENQUEUED_EXACT_REAL_CHILD_CLOSURE_V2",
            "source": {
                "schema": GM1_CHILD_CLOSURE_SCHEMA,
                "abi": GM1_CHILD_CLOSURE_ABI,
                "manifest_sha256": closure.manifest_sha256,
                "payload_sha256": closure.payload_sha256,
                "receipt_sha256": closure.receipt_sha256,
                "payload_nbytes": GM1_CHILD_CLOSURE_PAYLOAD_NBYTES,
            },
            "carrier_order": list(GM1_CHILD_CLOSURE_CARRIER_ORDER),
            "carrier_shapes": {
                carrier.name: list(carrier.shape)
                for carrier in GM1_CHILD_CLOSURE_CARRIERS
            },
            "stream_pointer": int(stream.ptr),
            "same_stream_no_host_sync": True,
            "mapping_sealed_device_storage_writable": True,
            "claim_limit": GM1_CHILD_CLOSURE_CLAIM_LIMIT,
        }
    )
    return CudaGm1ChildClosureV2(
        ready_event=ready,
        source=closure,
        arrays=MappingProxyType(device_arrays),
        receipt=receipt,
    )


__all__ = (
    "CudaGm1ChildClosureV2",
    "GM1_CHILD_CLOSURE_ABI",
    "GM1_CHILD_CLOSURE_CARRIERS",
    "GM1_CHILD_CLOSURE_CARRIER_ORDER",
    "GM1_CHILD_CLOSURE_CLAIM_LIMIT",
    "GM1_CHILD_CLOSURE_MANIFEST_SCHEMA",
    "GM1_CHILD_CLOSURE_MANIFEST_SHA256",
    "GM1_CHILD_CLOSURE_PAYLOAD_NBYTES",
    "GM1_CHILD_CLOSURE_PAYLOAD_SHA256",
    "GM1_CHILD_CLOSURE_RECEIPT_SCHEMA",
    "GM1_CHILD_CLOSURE_RECEIPT_SHA256",
    "GM1_CHILD_CLOSURE_SCHEMA",
    "GM1_CHILD_CLOSURE_STATUS",
    "GM1_CHILD_CLOSURE_UPLOAD_SCHEMA",
    "Gm1ChildClosureCarrier",
    "LoadedGm1ChildClosureV2",
    "load_gm1_child_closure_v2",
    "upload_gm1_child_closure_v2",
)
