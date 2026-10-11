"""Strict completed-step checkpoint binding for the real 50 m child.

Rust owns the durable CM1FCP1/CM1FHS1 pair and its raw-bit restore.  This
module owns only the live CUDA seam needed before the next complete large
step: upload all 37 restored carriers, rebuild explicitly recomputable padded
open-boundary and closure state, restore source cadence, and prove the live
carrier hashes before returning the composer state.

The pdefq ``asq`` ledger is diagnostic/non-dynamical and is not in the bounded
checkpoint.  It is reset to +0 on resume and budget-restart identity remains
explicitly unclaimed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    create_arwen_wsm6_component,
)
from cm1core.cuda_gm1_child_post_final_rk import (
    create_cuda_gm1_child_post_final_rk_state,
)
from cm1core.cuda_gm1_child_step_composer import (
    CudaGm1ChildCompletedClosureRefresh,
    CudaGm1ChildStepState,
    CudaGm1ChildWsm6Transition,
    _canonical_receipt_token,
    _gm1_child_live_nm_def_arrays,
    _load_gm1_child_solve1_operator_functions,
)
from cm1core.cuda_sgs_tke import launch_cuda_sgs_tke
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY
from cm1core.eqtset2_wsm6_bridge import (
    EQTSET2_RESTART_FIELDS,
    Eqtset2Wsm6Bridge,
    Eqtset2Wsm6Carriers,
)


GM1_CHILD_POST_STEP_LIVE_RESUME_SCHEMA = "gm1.cuda-child-post-step-live-resume/v1"
GM1_CHILD_RESUMED_CLOSURE_SCHEMA = "gm1.cuda-child-resumed-completed-closure/v1"
GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_SCHEMA = (
    "gm1.real-50m-child-tilestream-post-rk3-publication-admission/v1"
)
GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_STATUS = (
    "PASS_EXACT_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION"
)
GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_CLAIM_LIMIT = (
    "successful exact completed-RK3 same-residency carrier admission for one "
    "existing-format 37-carrier checkpoint and 13-carrier history pair only; "
    "this receipt changes no CM1FCP1/CM1FHS1 bytes and claims no continuation, "
    "forecast skill, tornado, or five-metre product"
)
GM1_CHILD_POST_STEP_LIVE_RESUME_CLAIM_LIMIT = (
    "exact live-CUDA reconstruction from one explicitly pinned completed-step "
    "real 50 m child post-step pair before its next complete large step; all 37 "
    "persistent "
    "carriers are raw-bit checked, padded/open and closure scratch are rebuilt "
    "only by admitted operators, and pdefq asq budget trajectory identity is "
    "not claimed"
)

STEP1_PAIR_PINS = MappingProxyType(
    {
        "checkpoint.cm1fcp": "910996c635b775e961e6b5d15d157e8febb793e514bc42ba88ea06ffc71e5d7c",
        "history.cm1fhs": "9e62e7947ac5d5a4ca344f035c09acdf1c0b0d11d149b2663ba2dc9fbcb77c60",
        "manifest.json": "06a1df66b7c1132d742337a9329314e62b7ff655934e916270e59ade50e7d05e",
        "receipt.json": "093334d547fea2be9d2287a53949eb14019f8beeb1deb73dfdd8d57f2042b221",
    }
)
STEP1_RESTORE_PINS = MappingProxyType(
    {
        "checkpoint-arena.f32le": "a31b2d8a004d4275195733cd46e1ad78990ef47932af3db4470d91367c958c4a",
        "history-arena.f32le": "cb68dd85892a4ca8b70149ff9d9b7b9b8d9a15779b4b4b8b752f063790aec30f",
        "manifest.json": "49a23b30571043e4c23a25cbc56cec7d4bea2ad962c393043f42b322c0b9b872",
        "receipt.json": "c3d7b8418bde25231f35b5d1f70d3efb3d3512ee86eee863ad57671aa0ad297a",
    }
)
STEP1_EXECUTION_RECEIPT_SHA256 = (
    "c2bae6ceade0e4eb552906916517bdcfcac3d2729b932c08e9e084b8cd5daa71"
)
STEP1_OPERATOR_MANIFEST_SHA256 = (
    "7577c3435a14c22ca1133f1667abe1bdca6428ba41db75ee089730969975a076"
)
STEP1_CHECKPOINT_CONTRACT_SHA256 = (
    "1ba302ea7450a21ed356201a58b3c2e9a863721ae2e8ad74915e1aed8bb7ec01"
)
STEP1_HISTORY_CONTRACT_SHA256 = (
    "7c73d1f683a170865ce50498ae9aafe150490980f5048adf8bb4a61730ab2bd8"
)
STEP1_MASS1_F64_BITS = "415bdb8f1a3286e0"
STEP1_MODEL_TIME_F64_BITS = "3fe0000000000000"

STEP2_PAIR_PINS = MappingProxyType(
    {
        "checkpoint.cm1fcp": "1b612a047da9d3227fc866dff93c0cf8244267147c13401b4fbdcaf143503a39",
        "history.cm1fhs": "1049142058611400abffc595561a5be3b4674cb88434604d0d6f070b767c22eb",
        "manifest.json": "05c33488d4976beb54640c002f43276e9bc8c98fb6b52711a66e25cd32f8ec0b",
        "receipt.json": "2b08ae910d389b460e50701733db9d43de42cfa3e871ad842ffb009b69121830",
    }
)
STEP2_RESTORE_PINS = MappingProxyType(
    {
        "checkpoint-arena.f32le": "ca3f6e471d2aeebc4941b8ff6f25629dc26e2880b3083d1e998b05f4e71c326a",
        "history-arena.f32le": "dbcafab05217dd398ad77d6a8c7f67af210b3bb29157537776307ada66ec398e",
        "manifest.json": "4fe27b4598402b9e6d44d1ea5e74fa83161c624036a99fb4958c143192fcff4d",
        "receipt.json": "55b16e57d33a8af54c081569d4ec016154833b93b694b8833ae161edf58e9010",
    }
)
STEP2_OPERATOR_MANIFEST_SHA256 = (
    "8821cc12cfa4b982f43cfbf1edebf4b0de0cb3bf452d2c0bb23553912ac5c4a7"
)
STEP2_CHECKPOINT_CONTRACT_SHA256 = (
    "323ca9100b868de07b5f9eb378e4aabed2422ecad6e147aaffcc2712a13f46d0"
)
STEP2_HISTORY_CONTRACT_SHA256 = (
    "544dadf8a7f2b39faf224ccf508f72bb6032749664ce93f1cd52a4c5ec358969"
)
STEP2_MODEL_TIME_F64_BITS = "3ff0000000000000"
STEP2_RECOVERY_COMMIT = "f0754c7b6e5bcf644c6d7d7942ea68149537fb6b"

CHECKPOINT_NAMES = (
    "closure.tkea",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.h_diabatic",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_sr",
    "held.qcten",
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "pp3d",
    "ppi",
    "ppx",
    "prs",
    "q3d",
    "qa",
    "radbce",
    "radbcn",
    "radbcs",
    "radbcw",
    "rf",
    "rho",
    "rr",
    "th3d",
    "tha",
    "u3d",
    "ua",
    "v3d",
    "va",
    "w3d",
    "wa",
)


def _shape(name: str, *, nx: int, ny: int, nz: int) -> tuple[int, ...]:
    mass = (nz, ny, nx)
    if name in {"q3d", "qa"}:
        return (6, *mass)
    if name in {"u3d", "ua"}:
        return (nz, ny, nx + 1)
    if name in {"v3d", "va"}:
        return (nz, ny + 1, nx)
    if name in {"w3d", "wa", "rf", "closure.tkea"}:
        return (nz + 1, ny, nx)
    if name in {"radbcw", "radbce"}:
        return (nz + 2, ny + 2 * CM1_OPEN_NGXY)
    if name in {"radbcs", "radbcn"}:
        return (nz + 2, nx + 2 * CM1_OPEN_NGXY)
    if name.startswith("component.mp_"):
        return (ny, nx)
    return mass


CHECKPOINT_SHAPES = MappingProxyType(
    {name: _shape(name, nx=300, ny=300, nz=75) for name in CHECKPOINT_NAMES}
)
CHECKPOINT_NBYTES = sum(
    math.prod(CHECKPOINT_SHAPES[name]) * 4 for name in CHECKPOINT_NAMES
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be one nonempty regular file")
    return resolved


def _pinned_inventory(directory: Path, pins: Mapping[str, str], label: str) -> Path:
    root = directory.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be one real directory")
    observed = {entry.name for entry in root.iterdir()}
    if observed != set(pins):
        raise ValueError(f"{label} inventory differs: {sorted(observed)}")
    for name, expected in pins.items():
        path = _regular(root / name, f"{label} {name}")
        if _sha256_file(path) != expected:
            raise ValueError(f"{label} {name} differs from its SHA-256 pin")
    return root


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class Gm1ChildPostStepSourcePins:
    """Complete immutable identity for one durable completed-step restart."""

    step: int
    pair_pins: Mapping[str, str]
    restore_pins: Mapping[str, str]
    operator_manifest_sha256: str
    checkpoint_contract_sha256: str
    history_contract_sha256: str
    mass1_f64_bits: str
    model_time_f64_bits: str
    execution_receipt_sha256: str | None
    recovery_commit: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("completed-step source step must be a positive integer")
        if set(self.pair_pins) != {
            "checkpoint.cm1fcp",
            "history.cm1fhs",
            "manifest.json",
            "receipt.json",
        }:
            raise ValueError("completed-step pair pins have the wrong inventory")
        if set(self.restore_pins) != {
            "checkpoint-arena.f32le",
            "history-arena.f32le",
            "manifest.json",
            "receipt.json",
        }:
            raise ValueError("completed-step restore pins have the wrong inventory")
        hex64 = (
            *self.pair_pins.values(),
            *self.restore_pins.values(),
            self.operator_manifest_sha256,
            self.checkpoint_contract_sha256,
            self.history_contract_sha256,
        )
        if self.execution_receipt_sha256 is not None:
            hex64 = (*hex64, self.execution_receipt_sha256)
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hex64
        ):
            raise ValueError("completed-step source contains a malformed SHA-256 pin")
        for label, bits in (
            ("mass1", self.mass1_f64_bits),
            ("model time", self.model_time_f64_bits),
        ):
            if (
                not isinstance(bits, str)
                or len(bits) != 16
                or any(char not in "0123456789abcdef" for char in bits)
            ):
                raise ValueError(f"completed-step source {label} bits are malformed")


STEP1_SOURCE_PINS = Gm1ChildPostStepSourcePins(
    step=1,
    pair_pins=STEP1_PAIR_PINS,
    restore_pins=STEP1_RESTORE_PINS,
    operator_manifest_sha256=STEP1_OPERATOR_MANIFEST_SHA256,
    checkpoint_contract_sha256=STEP1_CHECKPOINT_CONTRACT_SHA256,
    history_contract_sha256=STEP1_HISTORY_CONTRACT_SHA256,
    mass1_f64_bits=STEP1_MASS1_F64_BITS,
    model_time_f64_bits=STEP1_MODEL_TIME_F64_BITS,
    execution_receipt_sha256=STEP1_EXECUTION_RECEIPT_SHA256,
)
STEP2_SOURCE_PINS = Gm1ChildPostStepSourcePins(
    step=2,
    pair_pins=STEP2_PAIR_PINS,
    restore_pins=STEP2_RESTORE_PINS,
    operator_manifest_sha256=STEP2_OPERATOR_MANIFEST_SHA256,
    checkpoint_contract_sha256=STEP2_CHECKPOINT_CONTRACT_SHA256,
    history_contract_sha256=STEP2_HISTORY_CONTRACT_SHA256,
    mass1_f64_bits=STEP1_MASS1_F64_BITS,
    model_time_f64_bits=STEP2_MODEL_TIME_F64_BITS,
    execution_receipt_sha256=None,
    recovery_commit=STEP2_RECOVERY_COMMIT,
)


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildPostStepResume:
    pair_directory: Path
    restore_directory: Path
    execution_receipt_path: Path | None
    checkpoint_arena_path: Path
    carrier_specs: Mapping[str, Mapping[str, Any]]
    carrier_shapes: Mapping[str, tuple[int, ...]]
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dt_s: float
    ztop_m: float
    mass1_f64: float
    source_step: int
    source_model_time_s: float
    tke_commit_receipt: Mapping[str, Any]
    tke_commit_receipt_sha256: str
    source_pins: Gm1ChildPostStepSourcePins

    def carrier(self, name: str) -> Any:
        try:
            spec = self.carrier_specs[name]
        except KeyError as error:
            raise KeyError(f"unknown restored child carrier {name!r}") from error
        np = importlib.import_module("numpy")
        return np.memmap(
            self.checkpoint_arena_path,
            mode="r",
            dtype="<f4",
            offset=int(spec["offset_nbytes"]),
            shape=self.carrier_shapes[name],
            order="C",
        )


@dataclass(frozen=True, slots=True)
class CudaGm1ChildPostStepResume:
    state: CudaGm1ChildStepState
    ready_event: Any
    live_carrier_sha256: Mapping[str, str]
    live_arena_sha256: str
    receipt: Mapping[str, Any]


def load_gm1_child_post_step_resume(
    pair_directory: str | Path,
    restore_directory: str | Path,
    execution_receipt_path: str | Path | None = None,
    *,
    source_pins: Gm1ChildPostStepSourcePins = STEP1_SOURCE_PINS,
) -> LoadedGm1ChildPostStepResume:
    """Admit one exact completed-step pair and raw-bit restore."""

    if not isinstance(source_pins, Gm1ChildPostStepSourcePins):
        raise TypeError("source_pins must be Gm1ChildPostStepSourcePins")
    pair = _pinned_inventory(Path(pair_directory), source_pins.pair_pins, "child pair")
    restore = _pinned_inventory(
        Path(restore_directory), source_pins.restore_pins, "child raw-bit restore"
    )
    execution_path: Path | None = None
    if source_pins.execution_receipt_sha256 is not None:
        if execution_receipt_path is None:
            raise ValueError("completed-step source requires its exact execution receipt")
        execution_path = _regular(
            Path(execution_receipt_path), "child execution receipt"
        )
        if _sha256_file(execution_path) != source_pins.execution_receipt_sha256:
            raise ValueError("child execution receipt differs from its exact pin")
    elif execution_receipt_path is not None:
        raise ValueError(
            "durable-recovery source refuses an unrelated execution receipt"
        )

    pair_manifest = _json(pair / "manifest.json", "child pair manifest")
    pair_receipt = _json(pair / "receipt.json", "child pair receipt")
    restore_manifest = _json(restore / "manifest.json", "child restore manifest")
    restore_receipt = _json(restore / "receipt.json", "child restore receipt")
    execution = (
        _json(execution_path, "child execution receipt")
        if execution_path is not None
        else None
    )
    if (
        pair_manifest.get("schema") != "gm1.real-50m-child-post-step-pair-manifest/v1"
        or pair_manifest.get("status") != "PASS_RUST_PAIRED_POST_STEP_ENVELOPE"
        or pair_manifest.get("step") != source_pins.step
        or pair_manifest.get("operator_manifest_sha256")
        != source_pins.operator_manifest_sha256
        or pair_manifest.get("mass1_f64_bits") != source_pins.mass1_f64_bits
        or pair_manifest.get("checkpoint", {}).get("contract_sha256")
        != source_pins.checkpoint_contract_sha256
        or pair_manifest.get("history", {}).get("contract_sha256")
        != source_pins.history_contract_sha256
    ):
        raise ValueError("child pair manifest semantic identity changed")
    if (
        pair_receipt.get("schema") != "gm1.real-50m-child-post-step-pair-receipt/v1"
        or pair_receipt.get("step") != source_pins.step
        or pair_receipt.get("manifest_sha256")
        != source_pins.pair_pins["manifest.json"]
        or pair_receipt.get("operator_manifest_sha256")
        != source_pins.operator_manifest_sha256
    ):
        raise ValueError("child pair receipt semantic identity changed")
    if (
        restore_manifest.get("schema") != "gm1.real-50m-child-post-step-restore/v1"
        or restore_manifest.get("step") != source_pins.step
        or restore_manifest.get("source_pair_manifest_sha256")
        != source_pins.pair_pins["manifest.json"]
        or restore_receipt.get("schema")
        != "gm1.real-50m-child-post-step-restore-receipt/v1"
        or restore_receipt.get("status") != "PASS_RAW_BIT_RESTORE"
        or restore_receipt.get("step") != source_pins.step
    ):
        raise ValueError("child restore semantic identity changed")

    configuration = pair_manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("child pair manifest omits its domain configuration")
    domain_shape = configuration.get("domain_shape_zyx")
    if (
        not isinstance(domain_shape, list)
        or len(domain_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in domain_shape
        )
    ):
        raise ValueError("child pair domain shape is not three positive integers")
    nz, ny, nx = domain_shape

    def f64_from_bits(key: str) -> float:
        bits = configuration.get(key)
        if not isinstance(bits, str) or len(bits) != 16:
            raise ValueError(f"child pair configuration {key} is malformed")
        try:
            value = struct.unpack(">d", bytes.fromhex(bits))[0]
        except ValueError as error:
            raise ValueError(f"child pair configuration {key} is malformed") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"child pair configuration {key} is not positive finite")
        return value

    dx_m = f64_from_bits("dx_m_f64_bits")
    dy_m = f64_from_bits("dy_m_f64_bits")
    dt_s = f64_from_bits("dt_f64_bits")
    ztop_m = f64_from_bits("ztop_m_f64_bits")
    shapes = MappingProxyType(
        {name: _shape(name, nx=nx, ny=ny, nz=nz) for name in CHECKPOINT_NAMES}
    )
    checkpoint_nbytes = sum(math.prod(shapes[name]) * 4 for name in CHECKPOINT_NAMES)

    checkpoint = restore_manifest.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("file") != "checkpoint-arena.f32le"
        or checkpoint.get("dtype") != "little-endian-f32"
        or checkpoint.get("order") != "C"
        or checkpoint.get("nbytes") != checkpoint_nbytes
        or checkpoint.get("sha256")
        != source_pins.restore_pins["checkpoint-arena.f32le"]
    ):
        raise ValueError("child restored checkpoint arena identity changed")
    rows = checkpoint.get("carriers")
    if not isinstance(rows, list) or len(rows) != len(CHECKPOINT_NAMES):
        raise ValueError("child restored checkpoint carrier inventory changed")
    specs: dict[str, Mapping[str, Any]] = {}
    offset = 0
    for name, row in zip(CHECKPOINT_NAMES, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "name",
            "nbytes",
            "offset_nbytes",
            "sha256",
        }:
            raise ValueError("child restored checkpoint carrier descriptor changed")
        nbytes = math.prod(shapes[name]) * 4
        if (
            row["name"] != name
            or row["nbytes"] != nbytes
            or row["offset_nbytes"] != offset
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
        ):
            raise ValueError(f"restored checkpoint layout changed for {name}")
        specs[name] = MappingProxyType(dict(row))
        offset += nbytes
    if offset != checkpoint_nbytes:
        raise ValueError("restored checkpoint arena is not exactly exhausted")

    if (
        execution is not None
        and execution.get("schema")
        == GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_SCHEMA
    ):
        expected_keys = {
            "checkpoint_carrier_sha256",
            "claim_limit",
            "completed_rk3_operation_token_sha256",
            "final_step_operation_token_sha256",
            "gates",
            "history_carrier_sha256",
            "model",
            "model_time_f64_bits",
            "operator_manifest_sha256",
            "publication",
            "producer_git_commit",
            "runner_sha256",
            "schema",
            "stage_identity_sha256",
            "status",
            "step",
            "stream_pointer",
        }
        gates = execution.get("gates")
        publication = execution.get("publication")
        checkpoint_hashes = execution.get("checkpoint_carrier_sha256")
        history_hashes = execution.get("history_carrier_sha256")
        if (
            set(execution) != expected_keys
            or execution.get("status")
            != GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_STATUS
            or execution.get("claim_limit")
            != GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_CLAIM_LIMIT
            or execution.get("model") != "GM1"
            or execution.get("step") != source_pins.step
            or execution.get("model_time_f64_bits")
            != source_pins.model_time_f64_bits
            or execution.get("operator_manifest_sha256")
            != source_pins.operator_manifest_sha256
            or not isinstance(execution.get("producer_git_commit"), str)
            or len(execution["producer_git_commit"]) != 40
            or any(
                char not in "0123456789abcdef"
                for char in execution["producer_git_commit"]
            )
            or not isinstance(execution.get("runner_sha256"), str)
            or len(execution["runner_sha256"]) != 64
            or any(
                char not in "0123456789abcdef"
                for char in execution["runner_sha256"]
            )
            or not isinstance(gates, dict)
            or any(
                gates.get(name) is not True
                for name in (
                    "all_finite",
                    "moisture_nonnegative",
                    "qa_q3d_raw_bit_equal",
                    "current_a_raw_bit_equal",
                    "checkpoint_history_overlap_same_object",
                )
            )
            or not isinstance(publication, dict)
            or set(publication)
            != {
                "checkpoint_artifact_sha256",
                "checkpoint_contract_sha256",
                "checkpoint_raw_sha256",
                "history_artifact_sha256",
                "history_contract_sha256",
                "history_raw_sha256",
                "manifest_sha256",
                "receipt_sha256",
            }
            or publication.get("manifest_sha256")
            != source_pins.pair_pins["manifest.json"]
            or publication.get("receipt_sha256")
            != source_pins.pair_pins["receipt.json"]
            or publication.get("checkpoint_artifact_sha256")
            != source_pins.pair_pins["checkpoint.cm1fcp"]
            or publication.get("history_artifact_sha256")
            != source_pins.pair_pins["history.cm1fhs"]
            or publication.get("checkpoint_contract_sha256")
            != source_pins.checkpoint_contract_sha256
            or publication.get("history_contract_sha256")
            != source_pins.history_contract_sha256
            or publication.get("checkpoint_raw_sha256")
            != source_pins.restore_pins["checkpoint-arena.f32le"]
            or publication.get("history_raw_sha256")
            != source_pins.restore_pins["history-arena.f32le"]
            or not isinstance(checkpoint_hashes, dict)
            or set(checkpoint_hashes) != set(CHECKPOINT_NAMES)
            or not isinstance(history_hashes, dict)
            or set(history_hashes)
            != {
                "closure.tkea",
                "pp3d",
                "prs",
                "q3d",
                "rf",
                "rho",
                "terrain_msl",
                "th3d",
                "total_geopotential_zf",
                "u3d",
                "v3d",
                "w3d",
                "zf",
            }
        ):
            raise ValueError(
                "tilestream post-RK3 publication admission identity changed"
            )
        for name in CHECKPOINT_NAMES:
            if checkpoint_hashes[name] != specs[name]["sha256"]:
                raise ValueError(
                    f"tilestream checkpoint carrier hash changed for {name}"
                )
        for name, digest in (*checkpoint_hashes.items(), *history_hashes.items()):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(
                    f"tilestream publication carrier hash is malformed for {name}"
                )
        for name in (
            "closure.tkea",
            "pp3d",
            "prs",
            "q3d",
            "rf",
            "rho",
            "th3d",
            "u3d",
            "v3d",
            "w3d",
        ):
            if checkpoint_hashes[name] != history_hashes[name]:
                raise ValueError(
                    f"tilestream checkpoint/history hash changed for {name}"
                )
        for key in (
            "completed_rk3_operation_token_sha256",
            "final_step_operation_token_sha256",
            "stage_identity_sha256",
        ):
            value = execution.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"tilestream publication {key} is malformed")
        stream_pointer = execution.get("stream_pointer")
        if (
            isinstance(stream_pointer, bool)
            or not isinstance(stream_pointer, int)
            or stream_pointer <= 0
        ):
            raise ValueError("tilestream publication stream pointer is invalid")
        tke_commit = {
            "checkpoint_carrier_sha256": checkpoint_hashes["closure.tkea"],
            "checkpoint_raw_sha256": source_pins.restore_pins[
                "checkpoint-arena.f32le"
            ],
            "completed_rk3_operation_token_sha256": execution[
                "completed_rk3_operation_token_sha256"
            ],
            "disposition": (
                "successful exact completed-RK3 closure.tkea persisted through "
                "the canonical 37-carrier checkpoint"
            ),
            "execution_receipt_sha256": source_pins.execution_receipt_sha256,
            "final_step_operation_token_sha256": execution[
                "final_step_operation_token_sha256"
            ],
            "schema": "gm1.cuda-child-persisted-tke-restart-admission/v1",
            "source_pair_manifest_sha256": source_pins.pair_pins[
                "manifest.json"
            ],
            "source_step": source_pins.step,
        }
        tke_commit_sha = _canonical_receipt_token(tke_commit)
    elif execution is not None:
        gates = execution.get("gates", {})
        publication = execution.get("publication", {}).get("publish_receipt", {})
        post_final = (
            execution.get("operators", {}).get("finish", {}).get("post_final", {})
        )
        solve3 = post_final.get("solve3", {})
        tke_commit = solve3.get("tke_commit")
        closure_refresh = post_final.get("closure_refresh", {})
        if (
            execution.get("status")
            != "PASS_REAL_300X300X75_CHILD_ONE_COUPLED_LARGE_STEP"
            or any(
                gates.get(name) is not True
                for name in (
                    "all_finite",
                    "moisture_nonnegative",
                    "qa_q3d_raw_bit_equal",
                    "current_a_raw_bit_equal",
                )
            )
            or publication.get("manifest_sha256")
            != source_pins.pair_pins["manifest.json"]
            or publication.get("receipt_sha256")
            != source_pins.pair_pins["receipt.json"]
            or not isinstance(tke_commit, dict)
        ):
            raise ValueError("child execution receipt is not the accepted source run")
        tke_commit_sha = _canonical_receipt_token(tke_commit)
        if closure_refresh.get("tke_commit_receipt_sha256") != tke_commit_sha:
            raise ValueError("source closure refresh does not bind its TKE commit")
    else:
        tkea_spec = specs["closure.tkea"]
        tke_commit = {
            "checkpoint_carrier_sha256": tkea_spec["sha256"],
            "checkpoint_raw_sha256": source_pins.restore_pins[
                "checkpoint-arena.f32le"
            ],
            "disposition": (
                "prior in-memory TKE commit receipt unavailable after accepted "
                "post-publication validator refusal; persisted closure.tkea is the "
                "canonical numerical restart carrier"
            ),
            "recovery_commit": source_pins.recovery_commit,
            "schema": "gm1.cuda-child-persisted-tke-restart-admission/v1",
            "source_pair_manifest_sha256": source_pins.pair_pins["manifest.json"],
            "source_step": source_pins.step,
        }
        tke_commit_sha = _canonical_receipt_token(tke_commit)

    mass1 = struct.unpack(">d", bytes.fromhex(source_pins.mass1_f64_bits))[0]
    model_time = struct.unpack(">d", bytes.fromhex(source_pins.model_time_f64_bits))[0]
    return LoadedGm1ChildPostStepResume(
        pair_directory=pair,
        restore_directory=restore,
        execution_receipt_path=execution_path,
        checkpoint_arena_path=restore / "checkpoint-arena.f32le",
        carrier_specs=MappingProxyType(specs),
        carrier_shapes=shapes,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=dx_m,
        dy_m=dy_m,
        dt_s=dt_s,
        ztop_m=ztop_m,
        mass1_f64=mass1,
        source_step=source_pins.step,
        source_model_time_s=model_time,
        tke_commit_receipt=MappingProxyType(dict(tke_commit)),
        tke_commit_receipt_sha256=tke_commit_sha,
        source_pins=source_pins,
    )


def _destinations(state: CudaGm1ChildStepState) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in CHECKPOINT_NAMES:
        if name == "closure.tkea":
            value = state.closure.sgs["tkea"]
        elif name.startswith("component."):
            value = state.product.field(name)
        elif name.startswith("held."):
            value = state.product.field(name.removeprefix("held."))
        elif name.startswith("radbc"):
            value = state.prepared.arrays[name]
        else:
            value = state.product.field(name)
        result[name] = value
    return result


def _create_resumed_transition(
    state: CudaGm1ChildStepState,
    *,
    completed_calls: int,
    cupy: Any,
) -> CudaGm1ChildWsm6Transition:
    """Bind one fresh WSM6 bridge directly to restored persistence arrays."""

    if (
        isinstance(completed_calls, bool)
        or not isinstance(completed_calls, int)
        or completed_calls < 0
    ):
        raise ValueError("completed_calls must be a strict nonnegative integer")
    s, product = state.selection, state.product
    component = create_arwen_wsm6_component(
        component_id=ARWEN_WSM6_COMPONENT_ID,
        nx=s.nx,
        ny=s.ny,
        nz=s.nz,
        dx_m=s.dx_m,
        dy_m=s.dy_m,
        dz_m=s.dz_m,
        # A stretched selection declares its own top; a uniform one leaves it
        # to param.F:2608's dz*nk, which is the same number there.
        ztop_m=s.ztop_m,
        dt_s=s.dt_s,
    )
    bridge = Eqtset2Wsm6Bridge(
        component=component,
        nx=s.nx,
        ny=s.ny,
        nz=s.nz,
        dt_s=s.dt_s,
        idiss=False,
        efall=False,
        product_admission=False,
    )
    restart = {name: product.field(name) for name in EQTSET2_RESTART_FIELDS}
    bridge.restore_restart_arrays(restart, completed_calls=completed_calls)
    q3d = product.field("q3d")
    with state.stream:
        php = cupy.zeros_like(product.field("total_geopotential_zf"))
    carriers = Eqtset2Wsm6Carriers(
        pi0=product.field("pi0"),
        th0=product.field("th0"),
        rho=product.field("rho"),
        pp3d=product.field("pp3d"),
        th3d=product.field("th3d"),
        p=product.field("prs"),
        alt=product.field("rr"),
        phb=product.field("total_geopotential_zf"),
        php=php,
        qv=q3d[0],
        qc=q3d[1],
        qr=q3d[2],
        qi=q3d[3],
        qs=q3d[4],
        qg=q3d[5],
        qpten=product.field("qpten"),
        qtten=product.field("qtten"),
        qvten=product.field("qvten"),
        qcten=product.field("qcten"),
    )
    return CudaGm1ChildWsm6Transition(
        bridge=bridge,
        carriers=carriers,
        completed_transitions=completed_calls,
    )


def _rebuild_padded_open_state(
    state: CudaGm1ChildStepState, cupy: Any
) -> Mapping[str, Any]:
    s, a, p = state.selection, state.prepared.arrays, state.product
    ng = CM1_OPEN_NGXY
    copies = {
        "u3d": (
            p.field("u3d"),
            a["u3d"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny), slice(ng, ng + s.nx + 1)),
        ),
        "ua": (
            p.field("ua"),
            a["ua"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny), slice(ng, ng + s.nx + 1)),
        ),
        "v3d": (
            p.field("v3d"),
            a["v3d"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny + 1), slice(ng, ng + s.nx)),
        ),
        "va": (
            p.field("va"),
            a["va"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny + 1), slice(ng, ng + s.nx)),
        ),
        "w3d": (
            p.field("w3d"),
            a["w3d"],
            (slice(1, s.nz + 2), slice(ng, ng + s.ny), slice(ng, ng + s.nx)),
        ),
        "wa": (
            p.field("wa"),
            a["wa"],
            (slice(1, s.nz + 2), slice(ng, ng + s.ny), slice(ng, ng + s.nx)),
        ),
        **{
            name: (
                p.field(name),
                a[name],
                (slice(1, s.nz + 1), slice(1, s.ny + 1), slice(1, s.nx + 1)),
            )
            for name in ("pp3d", "ppi", "ppx", "rho")
        },
    }
    max_words = max(value.size for value in a.values())
    threads = (128, 1, 1)
    grid = ((max_words + 127) // 128, 1, 1)
    u32 = cupy.uint32
    _, _, _, bcp_we, bcp_sn, family_we, family_sn, rigid_w, _, _ = (
        state.open_compiled.kernels
    )
    with state.stream:
        for source, destination, slices in copies.values():
            view = destination[slices]
            if tuple(view.shape) != tuple(source.shape):
                raise ValueError("resume physical-to-padded slice shape changed")
            cupy.copyto(view, source)
        for name in ("u3d", "ua"):
            family_we(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(1)),
                stream=state.stream,
            )
            family_sn(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(1)),
                stream=state.stream,
            )
        for name in ("v3d", "va"):
            family_sn(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(2)),
                stream=state.stream,
            )
            family_we(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(2)),
                stream=state.stream,
            )
        for name in ("w3d", "wa"):
            family_we(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(3)),
                stream=state.stream,
            )
            family_sn(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng), u32(3)),
                stream=state.stream,
            )
            rigid_w(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz), u32(ng)),
                stream=state.stream,
            )
        for name in ("pp3d", "ppi", "ppx", "rho"):
            bcp_we(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz)),
                stream=state.stream,
            )
            bcp_sn(
                grid,
                threads,
                (a[name], u32(s.nx), u32(s.ny), u32(s.nz)),
                stream=state.stream,
            )
    return MappingProxyType(
        {
            "source_order": [
                "raw-bit physical current/A upload",
                "bcu west/east then south/north",
                "bcv south/north then west/east",
                "bcw west/east then south/north then rigid top/bottom",
                "bcp west/east then south/north for pressure and density",
            ],
            "physical_copy_names": list(copies),
            "open_boundary_kernel_source": "src/cm1core/cuda_sound3_open_boundary.py",
        }
    )


def _rebuild_resumed_closure(
    state: CudaGm1ChildStepState,
    loaded: LoadedGm1ChildPostStepResume,
    cupy: Any,
) -> CudaGm1ChildCompletedClosureRefresh:
    operators = _load_gm1_child_solve1_operator_functions()
    s, c = state.selection, state.closure
    if c.refresh_index != 1 or c.completed_stage != 0:
        raise ValueError("fresh child closure cadence differs before resume rebuild")
    source_step = loaded.source_step
    with state.stream:
        nm_def = operators.launch_nm_def(
            _gm1_child_live_nm_def_arrays(state),
            metrics=state.static_metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            ztop=state.vertical.ztop_f32,
            refresh_index=source_step,
            stream=state.stream,
            compiled_calcdef=c.nm_def_compiled_calcdef,
            compiled_calcnm=c.nm_def_compiled_calcnm,
        )
        c.sgs.update(nm_def.arrays)
        c.stress.update(
            {
                name: nm_def.workspace[name]
                for name in ("s11", "s22", "s33", "s12", "s13", "s23")
            }
        )
        c.sgs["dissten"].fill(cupy.float32(0.0))
        c.sgs["tketen"].fill(cupy.float32(0.0))
        coefficients = launch_cuda_sgs_tke(
            c.sgs,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            stream=state.stream,
            compiled=c.sgs_compiled,
        )
        cupy.copyto(c.tke["tke3d"], c.sgs["tkea"])
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    persistent = MappingProxyType(
        {
            "nm": c.sgs["nm"],
            "defv": c.sgs["defv"],
            "defh": c.sgs["defh"],
            "grdscl": c.sgs["grdscl"][1:],
            "lenscl": c.sgs["lenscl"],
            "dissten": c.sgs["dissten"],
            "kmh": c.sgs["kmh"],
            "kmv": c.sgs["kmv"],
            "khh": c.sgs["khh"],
            "khv": c.sgs["khv"],
            "tkea": c.sgs["tkea"],
            "tke3d": c.tke["tke3d"],
            "tketen": c.sgs["tketen"],
        }
    )
    descriptor: dict[str, Any] = {
        "completed_step_index": source_step,
        "nm_def_receipt": dict(nm_def.receipt),
        "persistent_ready": [
            {"name": name, "shape": list(value.shape)}
            for name, value in persistent.items()
        ],
        "refresh_index": source_step,
        "same_stream_no_host_sync": True,
        "schema": GM1_CHILD_RESUMED_CLOSURE_SCHEMA,
        "sgs_receipt": dict(coefficients.receipt),
        "source_checkpoint_raw_sha256": loaded.source_pins.restore_pins[
            "checkpoint-arena.f32le"
        ],
        "source_order": [
            "restore closure.tkea raw bits",
            "cm1.F:3718-3788 live calcnm/calcdef",
            "turb.F:1326-1413 persistent-tkea tkekm coefficients",
            "next solve1",
        ],
        "step_index": source_step - 1,
        "stream_pointer": int(state.stream.ptr),
        "tke_commit_receipt_sha256": loaded.tke_commit_receipt_sha256,
        "token_scope": (
            "same-stream resume reconstruction and exact device bindings; "
            "not a device-content hash"
        ),
    }
    token = _canonical_receipt_token(descriptor)
    receipt = MappingProxyType({**descriptor, "operation_token_sha256": token})
    result = CudaGm1ChildCompletedClosureRefresh(
        event,
        source_step - 1,
        source_step,
        source_step,
        int(state.stream.ptr),
        loaded.tke_commit_receipt_sha256,
        token,
        persistent,
        receipt,
    )
    c.last_nm_def = nm_def
    c.nm_def_compiled_calcdef = nm_def.compiled_calcdef
    c.nm_def_compiled_calcnm = nm_def.compiled_calcnm
    c.refresh_index = source_step + 1
    c.ready_event = event
    c.retained.extend((nm_def, coefficients, result))
    state.last_completed_closure_refresh = result
    state.ready_event = event
    state.retained.extend((nm_def, coefficients, result))
    return result


def restore_cuda_gm1_child_post_step_state(
    state: CudaGm1ChildStepState,
    loaded: LoadedGm1ChildPostStepResume,
) -> CudaGm1ChildPostStepResume:
    """Bind exact completed-step words and admit the next complete step."""

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be a fresh CudaGm1ChildStepState")
    if not isinstance(loaded, LoadedGm1ChildPostStepResume):
        raise TypeError("loaded must be LoadedGm1ChildPostStepResume")
    if (
        state.step_index != 0
        or state.time_seconds != 0.0
        or state.next_stage != 1
        or state.solve1_ready
        or state.transition.completed_transitions != 0
        or state.post_final.next_step_index != 0
    ):
        raise ValueError(
            "child resume requires one untouched fresh initialization state"
        )
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.get_current_stream().ptr) != int(state.stream.ptr):
        raise ValueError("child resume requires the source-owning CUDA stream")
    selection = state.selection
    if (
        (selection.nx, selection.ny, selection.nz) != (loaded.nx, loaded.ny, loaded.nz)
        or selection.dx_m != loaded.dx_m
        or selection.dy_m != loaded.dy_m
        or selection.dt_s != loaded.dt_s
        or float(state.vertical.ztop_f32) != loaded.ztop_m
    ):
        raise ValueError("live child geometry differs from the restored checkpoint")

    destinations = _destinations(state)
    host_views: dict[str, Any] = {}
    with state.stream:
        for name in CHECKPOINT_NAMES:
            host = loaded.carrier(name)
            destination = destinations[name]
            if (
                tuple(host.shape) != loaded.carrier_shapes[name]
                or tuple(destination.shape) != loaded.carrier_shapes[name]
                or destination.dtype != cupy.float32
                or not destination.flags.c_contiguous
            ):
                raise ValueError(f"resume carrier ABI changed for {name}")
            destination.set(host, stream=state.stream)
            host_views[name] = host
        state.pdefq_asq.fill(cupy.float64(0.0))
        padded_receipt = _rebuild_padded_open_state(state, cupy)

    state.transition = _create_resumed_transition(
        state,
        completed_calls=loaded.source_step,
        cupy=cupy,
    )
    state.post_final = create_cuda_gm1_child_post_final_rk_state(
        product=state.product,
        prepared=state.prepared,
        transition=state.transition,
        pdefq_asq=state.pdefq_asq,
        stream=state.stream,
        forecast_compiled=state.compiled,
        pdefq_compiled=state.pdefq_compiled,
        pdefq_workspace=state.pdefq_workspace,
        ni=state.selection.nx,
        nj=state.selection.ny,
        nk=state.selection.nz,
        dx=state.selection.dx_m,
        dy=state.selection.dy_m,
        dz=state.selection.dz_m,
        dt=state.selection.dt_s,
        next_step_index=loaded.source_step,
    )
    closure = _rebuild_resumed_closure(state, loaded, cupy)
    state.mass1_f64 = loaded.mass1_f64
    state.step_index = loaded.source_step
    state.time_seconds = loaded.source_model_time_s
    state.next_stage = 1
    state.solve1_ready = False
    state.ready_event = closure.ready_event
    state.ready_event.synchronize()

    rebound = _destinations(state)
    aggregate = hashlib.sha256()
    hashes: dict[str, str] = {}
    for name in CHECKPOINT_NAMES:
        raw = cupy.asnumpy(rebound[name]).tobytes(order="C")
        observed = hashlib.sha256(raw).hexdigest()
        expected = loaded.carrier_specs[name]["sha256"]
        if observed != expected:
            raise ValueError(f"live resumed carrier changed raw words for {name}")
        hashes[name] = observed
        aggregate.update(raw)
    live_arena_sha = aggregate.hexdigest()
    if live_arena_sha != loaded.source_pins.restore_pins["checkpoint-arena.f32le"]:
        raise ValueError("live resumed aggregate differs from the Rust raw arena")
    if (
        state.transition.completed_transitions != loaded.source_step
        or state.post_final.completed_transitions != loaded.source_step
        or state.post_final.next_step_index != loaded.source_step
        or state.closure.refresh_index != loaded.source_step + 1
        or state.last_completed_closure_refresh is not closure
    ):
        raise RuntimeError("resumed child cadence differs before its next step")

    receipt = MappingProxyType(
        {
            "claim_limit": GM1_CHILD_POST_STEP_LIVE_RESUME_CLAIM_LIMIT,
            "closure_reconstruction": dict(closure.receipt),
            "live_arena_sha256": live_arena_sha,
            "live_carrier_sha256": hashes,
            "model": "GM1",
            "padded_open_reconstruction": dict(padded_receipt),
            "pdefq_asq_disposition": (
                "diagnostic non-dynamical ledger reset to exact +0; budget-restart "
                "trajectory identity unclaimed"
            ),
            "schema": GM1_CHILD_POST_STEP_LIVE_RESUME_SCHEMA,
            "source_checkpoint_artifact_sha256": loaded.source_pins.pair_pins[
                "checkpoint.cm1fcp"
            ],
            "source_checkpoint_contract_sha256": (
                loaded.source_pins.checkpoint_contract_sha256
            ),
            "source_checkpoint_raw_sha256": loaded.source_pins.restore_pins[
                "checkpoint-arena.f32le"
            ],
            "source_execution_receipt_sha256": (
                loaded.source_pins.execution_receipt_sha256
            ),
            "source_pair_manifest_sha256": loaded.source_pins.pair_pins[
                "manifest.json"
            ],
            "source_recovery_commit": loaded.source_pins.recovery_commit,
            "source_step": loaded.source_step,
            "source_time_f64_bits": loaded.source_pins.model_time_f64_bits,
            "status": (
                f"READY_BEFORE_REAL_CHILD_LARGE_STEP_{loaded.source_step + 1}"
            ),
            "tke_restart_admission": dict(loaded.tke_commit_receipt),
            "wsm6_completed_calls": loaded.source_step,
        }
    )
    json.dumps(dict(receipt), allow_nan=False, sort_keys=True)
    del host_views
    return CudaGm1ChildPostStepResume(
        state,
        closure.ready_event,
        MappingProxyType(hashes),
        live_arena_sha,
        receipt,
    )


__all__ = (
    "CHECKPOINT_NAMES",
    "CHECKPOINT_NBYTES",
    "CHECKPOINT_SHAPES",
    "CudaGm1ChildPostStepResume",
    "GM1_CHILD_POST_STEP_LIVE_RESUME_CLAIM_LIMIT",
    "GM1_CHILD_POST_STEP_LIVE_RESUME_SCHEMA",
    "GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_CLAIM_LIMIT",
    "GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_SCHEMA",
    "GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_STATUS",
    "Gm1ChildPostStepSourcePins",
    "LoadedGm1ChildPostStepResume",
    "STEP1_EXECUTION_RECEIPT_SHA256",
    "STEP1_PAIR_PINS",
    "STEP1_RESTORE_PINS",
    "STEP1_SOURCE_PINS",
    "STEP2_PAIR_PINS",
    "STEP2_RESTORE_PINS",
    "STEP2_SOURCE_PINS",
    "load_gm1_child_post_step_resume",
    "restore_cuda_gm1_child_post_step_state",
)
