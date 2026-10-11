"""Strict first-large-step adapters for the real 50 m GM1 child.

This module begins the child-specific composition at the two places where the
legacy 100 m coordinator is scientifically inapplicable:

* the Rust full-state publication is a hash-bound reference envelope with a
  fresh-child (not completed-step) lifecycle; and
* fixed terrain makes CM1's base wind three-dimensional on the staggered
  ``u`` and ``v`` grids.  Broadcasting the former one-dimensional Rayleigh
  profiles would apply the wrong held momentum tendency.

The terrain Rayleigh operator is the exact selected ``solve1`` assignment.
The source-backed construction of ``u0``/``v0`` remains Rust-owned and is
sealed by ``base-wind.f32le``.  This file never reconstructs, translates, or
smooths those atmospheric data.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cm1core.cuda_rayleigh_held import (
    DEFAULT_RAYLEIGH_HELD_SELECTION,
    RayleighHeldSelection,
)
from cm1core.cuda_scalar_moist_rk import (
    CompiledScalarMoistRKKernels,
    CudaScalarMoistRKStageLaunch,
    CudaScalarMoistRKState,
    compile_cuda_scalar_moist_rk,
    create_cuda_scalar_moist_rk_state,
    launch_cuda_scalar_moist_rk_stage,
)
from cm1core.errors import refuse


GM1_CHILD_FULL_STATE_SCHEMA = "gm1.real-50m-child-full-state/v1"
GM1_CHILD_FULL_STATE_MANIFEST_SCHEMA = (
    "gm1.real-50m-child-full-state-manifest/v1"
)
GM1_CHILD_FIRST_STEP_SCHEMA = "gm1.real-50m-child-first-step/v1"
GM1_CHILD_CHECKPOINT_ADMISSION_SCHEMA = (
    "gm1.real-50m-child-checkpoint-admission/v1"
)
GM1_CHILD_FULL_STATE_RECEIPT_SCHEMA = (
    "gm1.real-50m-child-full-state-receipt/v1"
)
GM1_CHILD_BASE_WIND_ABI = "gm1.real-50m-child-terrain-base-wind-f32/v1"
GM1_CHILD_FULL_STATE_STATUS = "PASS_REAL_CHILD_READY_FOR_FIRST_LARGE_STEP"

GM1_CHILD_NI = 300
GM1_CHILD_NJ = 300
GM1_CHILD_NK = 75
GM1_CHILD_DT_F64_BITS = "3fe0000000000000"
GM1_CHILD_ZERO_F64_BITS = "0000000000000000"

GM1_CHILD_FULL_STATE_FILES = (
    "base-wind.f32le",
    "checkpoint-admission.json",
    "first-step.json",
    "manifest.json",
    "receipt.json",
)

GM1_CHILD_RAYLEIGH_SCHEMA = "gm1.cuda-child-terrain-rayleigh-held/v1"
GM1_CHILD_RAYLEIGH_ABI = "gm1.cuda-child-terrain-rayleigh-held-f32/v1"
GM1_CHILD_RAYLEIGH_KERNEL_NAME = "gm1_child_terrain_rayleigh_held_f32_v1"
GM1_CHILD_RAYLEIGH_THREADS = 256
GM1_CHILD_RAYLEIGH_CLAIM_LIMIT = (
    "CM1 r21.1 selected irdamp=1/hrdamp=0 solve1 held assignment using "
    "source-backed terrain-varying staggered u0/v0; no base-wind producer, "
    "other solve1 tendencies, RK integration, complete child large step, "
    "trajectory, forecast, tornado, or model-validation claim"
)

SOURCE_ANCHORS = {
    "terrain_base_wind": "src/base.F:674-723",
    "held_assignment": "src/solve1.F:412-454",
    "u_departure": "src/solve1.F:420",
    "v_departure": "src/solve1.F:421",
    "authority_tarball_sha256": (
        "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
    ),
}


@dataclass(frozen=True, slots=True)
class Gm1ChildFullStatePins:
    """Independent hashes that admit one exact historical envelope.

    These are for replaying a committed publication byte for byte.  Production
    consumers pass ``pins=None`` and the envelope is admitted by the identity
    its own receipt records, so a child composed from a parent at another
    completed step is loadable instead of refused for not being the first one.
    """

    manifest_sha256: str
    first_step_sha256: str
    checkpoint_admission_sha256: str
    receipt_sha256: str
    base_wind_sha256: str
    producer_source_sha256: str
    producer_executable_sha256: str

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be exactly 64 lower-case hexadecimal characters")


REAL_GM1_CHILD_FULL_STATE_PINS = Gm1ChildFullStatePins(
    manifest_sha256="02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d",
    first_step_sha256="132f2f7f877abe9bbfac44900406ceae17c6de93d0bcfc9dacacefe8d993edeb",
    checkpoint_admission_sha256=(
        "23c0f09129084690c16cae85461d23f79f5645b430a64988242a6bee0119ba1a"
    ),
    receipt_sha256="792dc1ade597590e985b44c434a05c0c47751390bd5949520abb309527fdebb6",
    base_wind_sha256="adf38848863b1cb36c7918e5617d9f38b81443ddf965bc6358ef7dd60ffaf8dd",
    producer_source_sha256=(
        "402fb91e6670200677ad8c2e55a022cda8ade695bb4a234c80a313c711c910f5"
    ),
    producer_executable_sha256=(
        "5272cd84a7d0861c4a752c0045eda9e8daa7896327e859142f9c8443a9dd6bfa"
    ),
)


@dataclass(frozen=True, slots=True)
class Gm1ChildFullStateArtifactRef:
    """One sibling artifact the envelope was composed from.

    The envelope records, for every artifact it consumed, the sibling directory
    name and the exact byte count and digest of each file in it.  A consumer
    that opens a sibling must open the one recorded here and check these
    digests: otherwise the receipt it publishes would name the artifacts the
    envelope was composed from while the run consumed different ones, and the
    substitution would be invisible in the receipt.
    """

    role: str
    directory: str
    schema: str
    abi: str
    # ``None`` for a publication made before the envelope recorded it.
    status: str | None
    files: Mapping[str, tuple[int, str]]

    def digest(self, name: str) -> str:
        try:
            return self.files[name][1]
        except KeyError as error:
            raise KeyError(
                f"the composed child's {self.role} artifact records no file {name!r}"
            ) from error

    def nbytes(self, name: str) -> int:
        try:
            return self.files[name][0]
        except KeyError as error:
            raise KeyError(
                f"the composed child's {self.role} artifact records no file {name!r}"
            ) from error


_ARTIFACT_ROLES = ("atmosphere", "auxiliary", "closure", "static_heights", "terrain")


# The envelope's artifact reference gained `status` after the T+300 child was
# published, so exactly two key sets are admitted: the current one, and the same
# one without `status`.  A publication that predates the field records no status
# and is read as recording none, rather than being handed an invented one -- the
# artifact's own receipt is the authority on how it was published, and the
# composer opens and digest-checks that receipt anyway.  Any other key set is
# still refused: this is one named historical shape, not a relaxation.
_ARTIFACT_REFERENCE_KEYS = frozenset(
    {"abi", "directory", "files", "reference_rule", "role", "schema", "status"}
)
_ARTIFACT_REFERENCE_KEYS_BEFORE_STATUS = _ARTIFACT_REFERENCE_KEYS - {"status"}


def _parse_artifact_reference(role: str, value: Any) -> Gm1ChildFullStateArtifactRef:
    if not isinstance(value, dict) or set(value) not in (
        _ARTIFACT_REFERENCE_KEYS,
        _ARTIFACT_REFERENCE_KEYS_BEFORE_STATUS,
    ):
        raise ValueError(f"full-state {role} artifact reference contract changed")
    directory = value["directory"]
    if (
        not isinstance(directory, str)
        or not directory
        or directory in {".", ".."}
        or "/" in directory
        or "\\" in directory
        or directory != directory.strip()
    ):
        raise ValueError(
            f"full-state {role} artifact directory {directory!r} is not one plain "
            "sibling directory name; a consumer could be steered outside the "
            "publication to an artifact the envelope never admitted"
        )
    _require_equal(value["role"], role, f"{role} artifact role")
    for key in ("abi", "schema"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"full-state {role} artifact {key} is not a non-empty string")
    if "status" in value and (not isinstance(value["status"], str) or not value["status"]):
        raise ValueError(f"full-state {role} artifact status is not a non-empty string")
    records = value["files"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"full-state {role} artifact lists no files")
    files: dict[str, tuple[int, str]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "nbytes", "sha256"}:
            raise ValueError(f"full-state {role} artifact file record contract changed")
        name = record["name"]
        nbytes = record["nbytes"]
        digest = record["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError(f"full-state {role} artifact file name {name!r} is not plain")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"full-state {role} artifact file {name} has no positive byte count")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"full-state {role} artifact file {name} digest is not a SHA-256")
        if name in files:
            raise ValueError(f"full-state {role} artifact names {name} twice")
        files[name] = (nbytes, digest)
    return Gm1ChildFullStateArtifactRef(
        role=role,
        directory=directory,
        schema=value["schema"],
        abi=value["abi"],
        status=value.get("status"),
        files=MappingProxyType(files),
    )


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildFullStateEnvelope:
    """Verified small envelope and its read-only terrain base-wind views.

    The five large sibling artifacts are intentionally not repacked here.
    Their existing strict loaders must rehash them before CUDA upload.
    """

    directory: Path
    manifest: dict[str, Any]
    first_step: dict[str, Any]
    checkpoint_admission: dict[str, Any]
    receipt: dict[str, Any]
    u0: Any
    v0: Any
    pins: Gm1ChildFullStatePins
    domain_shape_zyx: tuple[int, int, int]
    artifacts: Mapping[str, Gm1ChildFullStateArtifactRef]

    def artifact(self, role: str) -> Gm1ChildFullStateArtifactRef:
        try:
            return self.artifacts[role]
        except KeyError as error:
            raise KeyError(
                f"the composed child envelope names no {role!r} artifact"
            ) from error


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if canonical != data:
        raise ValueError(f"{path.name} is not canonical JSON")
    return value, data


def _require_regular_exact_inventory(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("child full-state path must be a regular non-symlink directory")
    actual: list[str] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("child full-state directory contains a non-regular member")
        actual.append(path.name)
    if tuple(sorted(actual)) != GM1_CHILD_FULL_STATE_FILES:
        raise ValueError(
            "child full-state inventory differs from the exact five-file contract"
        )


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:  # noqa: E721
        raise ValueError(f"{label} differs from the sealed child full-state contract")


def _pins_from_receipt(receipt: dict[str, Any], receipt_sha256: str) -> Gm1ChildFullStatePins:
    """Take the envelope's expected digests from the receipt it published.

    The receipt is the document that anchors the publication: it names the
    manifest, first-step and checkpoint-admission documents, the base-wind
    arena and the producer that wrote them.  Admitting against those turns
    every check below into a self-consistency check over one publication,
    which is what a consumer of a child composed from any admissible parent
    needs - instead of a comparison against the first child ever composed,
    which refuses every later one for nothing that is wrong with it.
    """

    values: dict[str, Any] = {}
    for field, key in (
        ("manifest_sha256", "manifest_sha256"),
        ("first_step_sha256", "first_step_sha256"),
        ("checkpoint_admission_sha256", "checkpoint_admission_sha256"),
        ("base_wind_sha256", "base_wind_sha256"),
        ("producer_source_sha256", "producer_source_sha256"),
        ("producer_executable_sha256", "producer_executable_sha256"),
    ):
        value = receipt.get(key)
        if not isinstance(value, str):
            raise ValueError(
                f"child full-state receipt records no {key}; the publication cannot "
                "anchor its own documents and nothing binds the files beside it"
            )
        values[field] = value
    pins = Gm1ChildFullStatePins(receipt_sha256=receipt_sha256, **values)
    pins.validate()
    return pins


def load_gm1_child_full_state_envelope(
    directory: str | Path,
    *,
    pins: Gm1ChildFullStatePins | None = None,
) -> LoadedGm1ChildFullStateEnvelope:
    """Rehash and structurally admit the exact Rust child publication.

    This is deliberately fail-closed at the fresh step-zero boundary.  It
    cannot be used to label initialization as a restart checkpoint.
    """

    if pins is not None and not isinstance(pins, Gm1ChildFullStatePins):
        raise TypeError("pins must be a Gm1ChildFullStatePins or None")
    root = Path(directory).resolve(strict=True)
    _require_regular_exact_inventory(root)
    paths = {name: root / name for name in GM1_CHILD_FULL_STATE_FILES}
    if pins is None:
        receipt_value, receipt_raw = _read_canonical_json(paths["receipt.json"])
        pins = _pins_from_receipt(
            receipt_value, hashlib.sha256(receipt_raw).hexdigest()
        )
    else:
        pins.validate()

    expected_hashes = {
        "manifest.json": pins.manifest_sha256,
        "first-step.json": pins.first_step_sha256,
        "checkpoint-admission.json": pins.checkpoint_admission_sha256,
        "receipt.json": pins.receipt_sha256,
        "base-wind.f32le": pins.base_wind_sha256,
    }
    for name, expected in expected_hashes.items():
        if _hash_file(paths[name]) != expected:
            refuse(
                code="GM1-CHILD-FULL-STATE-DRIFT",
                gate="product-profile",
                option=name,
                value="sha256-mismatch",
                reason="the real child reference envelope differs from its independent pin",
                remedy="regenerate and independently reseal the complete child publication",
            )

    manifest, _ = _read_canonical_json(paths["manifest.json"])
    first_step, _ = _read_canonical_json(paths["first-step.json"])
    checkpoint, _ = _read_canonical_json(paths["checkpoint-admission.json"])
    receipt, _ = _read_canonical_json(paths["receipt.json"])

    _require_equal(manifest.get("schema"), GM1_CHILD_FULL_STATE_MANIFEST_SCHEMA, "manifest schema")
    _require_equal(manifest.get("state_schema"), GM1_CHILD_FULL_STATE_SCHEMA, "state schema")
    _require_equal(manifest.get("model"), "GM1", "manifest model")
    _require_equal(manifest.get("status"), GM1_CHILD_FULL_STATE_STATUS, "manifest status")
    _require_equal(
        manifest.get("authority", {}).get("tarball_sha256"),
        SOURCE_ANCHORS["authority_tarball_sha256"],
        "mathematical authority",
    )
    config = manifest.get("configuration", {})
    # The child geometry is read from the envelope, not compared to one child.
    # What every consumer needs is that the shape the envelope declares is the
    # shape its own base-wind carriers and byte counts describe, which is
    # checked below; which child grid a given CUDA lane is compiled for is that
    # lane's own refusal to make, by value, where the arena is allocated.
    shape = config.get("domain_shape_zyx")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
    ):
        raise ValueError(
            "child full-state configuration does not declare a positive z/y/x child shape; "
            "no carrier extent in this envelope could be checked against the arenas it points at"
        )
    domain_shape_zyx = (int(shape[0]), int(shape[1]), int(shape[2]))
    _require_equal(config.get("target_dt_seconds_f64_bits"), GM1_CHILD_DT_F64_BITS, "child dt")
    _require_equal(config.get("terrain"), True, "terrain selector")
    _require_equal(config.get("imove"), 0, "moving-domain selector")
    for key in ("dx_m_f64_bits", "dy_m_f64_bits"):
        bits = config.get(key)
        if not isinstance(bits, str) or re.fullmatch(r"[0-9a-f]{16}", bits) is None:
            raise ValueError(f"child full-state configuration {key} is not f64 bits")
    parent = manifest.get("parent_provenance", {})
    _require_equal(parent.get("synthetic_atmospheric_data"), False, "synthetic atmospheric data flag")
    _require_equal(parent.get("synthetic_translation"), False, "synthetic translation flag")
    producer = manifest.get("producer", {})
    _require_equal(producer.get("source_sha256"), pins.producer_source_sha256, "producer source")
    _require_equal(
        producer.get("executable_sha256"),
        pins.producer_executable_sha256,
        "producer executable",
    )
    artifacts_value = manifest.get("artifacts")
    if not isinstance(artifacts_value, dict) or set(artifacts_value) != set(_ARTIFACT_ROLES):
        raise ValueError("full-state manifest must reference exactly five sibling artifacts")
    artifacts = MappingProxyType(
        {
            role: _parse_artifact_reference(role, artifacts_value[role])
            for role in _ARTIFACT_ROLES
        }
    )

    nz, ny, nx = domain_shape_zyx
    u0_nbytes = nz * ny * (nx + 1) * 4
    v0_nbytes = nz * (ny + 1) * nx * 4
    base_wind_nbytes = u0_nbytes + v0_nbytes
    if paths["base-wind.f32le"].stat().st_size != base_wind_nbytes:
        raise ValueError(
            "base-wind.f32le is not the byte length the child shape this envelope "
            "declares requires; the staggered u0/v0 views would run past the arena"
        )
    base = manifest.get("base_wind", {})
    _require_equal(base.get("abi"), GM1_CHILD_BASE_WIND_ABI, "base-wind ABI")
    _require_equal(base.get("file"), "base-wind.f32le", "base-wind file")
    _require_equal(base.get("nbytes"), base_wind_nbytes, "base-wind length")
    _require_equal(base.get("sha256"), pins.base_wind_sha256, "base-wind hash")
    carriers = base.get("carriers")
    if not isinstance(carriers, list) or len(carriers) != 2:
        raise ValueError("base-wind layout must contain exactly u0 and v0")
    expected_carriers = (
        ("u0", [nz, ny, nx + 1], 0, u0_nbytes),
        ("v0", [nz, ny + 1, nx], u0_nbytes, v0_nbytes),
    )
    for record, (name, shape, offset, nbytes) in zip(carriers, expected_carriers, strict=True):
        _require_equal(record.get("name"), name, f"{name} carrier name")
        _require_equal(record.get("shape"), shape, f"{name} shape")
        _require_equal(record.get("offset_nbytes"), offset, f"{name} offset")
        _require_equal(record.get("nbytes"), nbytes, f"{name} length")
        _require_equal(record.get("dtype"), "little-endian-f32", f"{name} dtype")
        _require_equal(record.get("order"), "C", f"{name} order")

    _require_equal(first_step.get("schema"), GM1_CHILD_FIRST_STEP_SCHEMA, "first-step schema")
    _require_equal(first_step.get("manifest_sha256"), pins.manifest_sha256, "first-step manifest pin")
    _require_equal(first_step.get("status"), "READY_FOR_FIRST_COMPLETE_LARGE_STEP", "first-step status")
    lifecycle = first_step.get("lifecycle", {})
    for key, expected in {
        "completed_large_step": False,
        "completed_large_steps": 0,
        "first_large_step_ready": True,
        "model_time_seconds_f64_bits": GM1_CHILD_ZERO_F64_BITS,
        "next_stage": 1,
        "open_boundary_histories_initialized": True,
        "step_index": 0,
        "target_dt_seconds_f64_bits": GM1_CHILD_DT_F64_BITS,
    }.items():
        _require_equal(lifecycle.get(key), expected, f"first-step lifecycle {key}")
    rayleigh = first_step.get("terrain_rayleigh_base_wind", {})
    _require_equal(rayleigh.get("abi"), GM1_CHILD_BASE_WIND_ABI, "first-step base-wind ABI")
    _require_equal(rayleigh.get("file"), "base-wind.f32le", "first-step base-wind file")
    _require_equal(rayleigh.get("u0_shape_zyx"), [nz, ny, nx + 1], "first-step u0 shape")
    _require_equal(rayleigh.get("v0_shape_zyx"), [nz, ny + 1, nx], "first-step v0 shape")
    dispositions = first_step.get("carrier_dispositions", {})
    _require_equal(
        dispositions.get("a_to_current_raw_bit_pairs"),
        [
            ["ua", "u3d"],
            ["va", "v3d"],
            ["wa", "w3d"],
            ["ppi", "pp3d"],
            ["tha", "th3d"],
            ["qa", "q3d"],
        ],
        "fresh current/A disposition",
    )
    _require_equal(
        dispositions.get("held_positive_zero"),
        ["held.qpten", "held.qtten", "held.qvten", "held.qcten"],
        "fresh held disposition",
    )
    component_persistence = dispositions.get("component_persistence")
    if not isinstance(component_persistence, list) or len(component_persistence) != 11:
        raise ValueError("fresh child must expose exactly 11 WSM6 component-persistence arrays")
    _require_equal(
        first_step.get("physical_base_views"),
        [
            {
                "artifact": "auxiliary",
                "carrier": name,
                "file": "terrain-open.f32le",
                "physical_slice_exclusive_zyx": [1, nz + 1, 3, ny + 3, 3, nx + 3],
            }
            for name in ("pi0", "th0", "qv0")
        ],
        "physical base-state view map",
    )

    _require_equal(
        checkpoint.get("schema"),
        GM1_CHILD_CHECKPOINT_ADMISSION_SCHEMA,
        "checkpoint-admission schema",
    )
    _require_equal(checkpoint.get("manifest_sha256"), pins.manifest_sha256, "checkpoint manifest pin")
    _require_equal(checkpoint.get("carrier_count"), 37, "checkpoint carrier count")
    _require_equal(
        checkpoint.get("initialization_boundary_admitted_as_checkpoint"),
        False,
        "initialization checkpoint refusal",
    )
    names = [entry.get("name") for entry in checkpoint.get("persistent_sources", [])]
    if len(names) != 37 or len(set(names)) != 37:
        raise ValueError("checkpoint admission must contain exactly 37 unique carriers")

    _require_equal(receipt.get("schema"), GM1_CHILD_FULL_STATE_RECEIPT_SCHEMA, "receipt schema")
    _require_equal(receipt.get("status"), GM1_CHILD_FULL_STATE_STATUS, "receipt status")
    _require_equal(receipt.get("manifest_sha256"), pins.manifest_sha256, "receipt manifest pin")
    _require_equal(receipt.get("first_step_sha256"), pins.first_step_sha256, "receipt first-step pin")
    _require_equal(
        receipt.get("checkpoint_admission_sha256"),
        pins.checkpoint_admission_sha256,
        "receipt checkpoint pin",
    )
    _require_equal(receipt.get("base_wind_sha256"), pins.base_wind_sha256, "receipt base-wind pin")
    _require_equal(receipt.get("producer_source_sha256"), pins.producer_source_sha256, "receipt source pin")
    _require_equal(
        receipt.get("producer_executable_sha256"),
        pins.producer_executable_sha256,
        "receipt executable pin",
    )

    np = importlib.import_module("numpy")
    base_path = paths["base-wind.f32le"]
    u0 = np.memmap(
        base_path, mode="r", dtype="<f4", offset=0, shape=(nz, ny, nx + 1), order="C"
    )
    v0 = np.memmap(
        base_path,
        mode="r",
        dtype="<f4",
        offset=u0_nbytes,
        shape=(nz, ny + 1, nx),
        order="C",
    )
    if _hash_float32_view(u0) != carriers[0]["sha256"]:
        raise ValueError("u0 carrier bytes differ from the manifest carrier pin")
    if _hash_float32_view(v0) != carriers[1]["sha256"]:
        raise ValueError("v0 carrier bytes differ from the manifest carrier pin")
    if not bool(np.all(np.isfinite(u0))) or not bool(np.all(np.isfinite(v0))):
        raise ValueError("terrain base wind contains non-finite values")
    return LoadedGm1ChildFullStateEnvelope(
        directory=root,
        manifest=manifest,
        first_step=first_step,
        checkpoint_admission=checkpoint,
        receipt=receipt,
        u0=u0,
        v0=v0,
        pins=pins,
        domain_shape_zyx=domain_shape_zyx,
        artifacts=artifacts,
    )


def _hash_float32_view(array: Any) -> str:
    view = memoryview(array).cast("B")
    return hashlib.sha256(view).hexdigest()


def child_terrain_rayleigh_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    mass = (nk, nj, ni)
    return {
        "tauh": (nk, nj + 2, ni + 2),
        "taus": (nk, nj + 2, ni + 2),
        "tauf": (nk + 1, nj + 2, ni + 2),
        "u0": (nk, nj, ni + 1),
        "v0": (nk, nj + 1, ni),
        "ua": (nk, nj, ni + 1),
        "va": (nk, nj + 1, ni),
        "wa": mass,
        "tha": mass,
        "uten1": (nk, nj, ni + 1),
        "vten1": (nk, nj + 1, ni),
        "wten1": mass,
        "thten1": mass,
    }


CHILD_TERRAIN_RAYLEIGH_INPUTS = (
    "tauh", "taus", "tauf", "u0", "v0", "ua", "va", "wa", "tha"
)
CHILD_TERRAIN_RAYLEIGH_OUTPUTS = ("uten1", "vten1", "wten1", "thten1")
CHILD_TERRAIN_RAYLEIGH_ARRAYS = (
    CHILD_TERRAIN_RAYLEIGH_INPUTS + CHILD_TERRAIN_RAYLEIGH_OUTPUTS
)


def _validate_rayleigh_selection(selection: RayleighHeldSelection) -> None:
    if not isinstance(selection, RayleighHeldSelection):
        raise TypeError("selection must be a RayleighHeldSelection")
    if selection != DEFAULT_RAYLEIGH_HELD_SELECTION:
        refuse(
            code="GM1-CHILD-RAYLEIGH-SELECTION",
            gate="product-profile",
            option="selection",
            value=repr(selection),
            reason="the child bridge is sealed to irdamp=1/hrdamp=0 and the 12-15 km damping layer",
            remedy="use the sealed selection or build a separately evidenced child profile",
        )


def _require_host_rayleigh_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = child_terrain_rayleigh_shapes(ni, nj, nk)
    missing = [name for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"missing child terrain-Rayleigh arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be a C-order float32 array with shape {shapes[name]}"
            )
        if name in CHILD_TERRAIN_RAYLEIGH_INPUTS and not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        if name in CHILD_TERRAIN_RAYLEIGH_OUTPUTS and not value.flags.writeable:
            raise ValueError(f"{name} must be writeable")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("child terrain-Rayleigh arrays must not alias")
    return result


def cpu_child_terrain_rayleigh_held_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RayleighHeldSelection = DEFAULT_RAYLEIGH_HELD_SELECTION,
) -> None:
    """Apply CM1's source-ordered held assignment with three-dimensional base wind."""

    _validate_rayleigh_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_rayleigh_arrays(arrays, ni=ni, nj=nj, nk=nk)
    rate = np.float32(selection.damping_rate_s_inv)
    negative_rate = np.float32(-rate)
    negative_rate_half = np.multiply(negative_rate, np.float32(0.5), dtype=np.float32)
    u_pair = np.add(
        a["tauh"][:, 1 : nj + 1, 0 : ni + 1],
        a["tauh"][:, 1 : nj + 1, 1 : ni + 2],
        dtype=np.float32,
    )
    u_coefficient = np.multiply(negative_rate_half, u_pair, dtype=np.float32)
    u_departure = np.subtract(a["ua"], a["u0"], dtype=np.float32)
    np.copyto(a["uten1"], np.multiply(u_coefficient, u_departure, dtype=np.float32))
    v_pair = np.add(
        a["tauh"][:, 0 : nj + 1, 1 : ni + 1],
        a["tauh"][:, 1 : nj + 2, 1 : ni + 1],
        dtype=np.float32,
    )
    v_coefficient = np.multiply(negative_rate_half, v_pair, dtype=np.float32)
    v_departure = np.subtract(a["va"], a["v0"], dtype=np.float32)
    np.copyto(a["vten1"], np.multiply(v_coefficient, v_departure, dtype=np.float32))
    w_coefficient = np.multiply(
        negative_rate, a["tauf"][:nk, 1 : nj + 1, 1 : ni + 1], dtype=np.float32
    )
    np.copyto(a["wten1"], np.multiply(w_coefficient, a["wa"], dtype=np.float32))
    theta_coefficient = np.multiply(
        negative_rate, a["taus"][:, 1 : nj + 1, 1 : ni + 1], dtype=np.float32
    )
    np.copyto(
        a["thten1"], np.multiply(theta_coefficient, a["tha"], dtype=np.float32)
    )


GM1_CHILD_RAYLEIGH_SOURCE = r"""
__device__ __forceinline__ float cm1_add(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float cm1_sub(const float a,const float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float cm1_mul(const float a,const float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}

extern "C" __global__ void gm1_child_terrain_rayleigh_held_f32_v1(
 const float* __restrict__ tauh,const float* __restrict__ taus,const float* __restrict__ tauf,
 const float* __restrict__ u0,const float* __restrict__ v0,
 const float* __restrict__ ua,const float* __restrict__ va,
 const float* __restrict__ wa,const float* __restrict__ tha,
 float* __restrict__ uten1,float* __restrict__ vten1,
 float* __restrict__ wten1,float* __restrict__ thten1,
 unsigned int ni,unsigned int nj,unsigned int nk,float rdalpha)
{
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned long long nu=(unsigned long long)nk*nj*(ni+1u);
 const unsigned long long nv=(unsigned long long)nk*(nj+1u)*ni;
 const unsigned long long ns=(unsigned long long)nk*nj*ni;
 const float neg=-rdalpha;
 const float negh=cm1_mul(neg,0.5f);
 for(unsigned long long z=first;z<nu;z+=step){
  unsigned int f=(unsigned int)(z%(ni+1u));unsigned long long q=z/(ni+1u);
  unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);
  float pair=cm1_add(tauh[i3(k,j+1u,f,nj+2u,ni+2u)],tauh[i3(k,j+1u,f+1u,nj+2u,ni+2u)]);
  uten1[z]=cm1_mul(cm1_mul(negh,pair),cm1_sub(ua[z],u0[z]));
 }
 for(unsigned long long z=first;z<nv;z+=step){
  unsigned int i=(unsigned int)(z%ni);unsigned long long q=z/ni;
  unsigned int f=(unsigned int)(q%(nj+1u)),k=(unsigned int)(q/(nj+1u));
  float pair=cm1_add(tauh[i3(k,f,i+1u,nj+2u,ni+2u)],tauh[i3(k,f+1u,i+1u,nj+2u,ni+2u)]);
  vten1[z]=cm1_mul(cm1_mul(negh,pair),cm1_sub(va[z],v0[z]));
 }
 for(unsigned long long z=first;z<ns;z+=step){
  unsigned int i=(unsigned int)(z%ni);unsigned long long q=z/ni;
  unsigned int j=(unsigned int)(q%nj),k=(unsigned int)(q/nj);
  unsigned long long p=i3(k,j+1u,i+1u,nj+2u,ni+2u);
  wten1[z]=cm1_mul(cm1_mul(neg,tauf[p]),wa[z]);
  thten1[z]=cm1_mul(cm1_mul(neg,taus[p]),tha[z]);
 }
}
"""

GM1_CHILD_RAYLEIGH_SOURCE_SHA256 = (
    "6509e757e6f2a1f87b6ee50cb20d540ce9f618c262e4c2da40be04bda4bb138d"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledGm1ChildTerrainRayleigh:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_child_terrain_rayleigh() -> CompiledGm1ChildTerrainRayleigh:
    source_hash = hashlib.sha256(GM1_CHILD_RAYLEIGH_SOURCE.encode("utf-8")).hexdigest()
    if source_hash != GM1_CHILD_RAYLEIGH_SOURCE_SHA256:
        raise RuntimeError("packaged child terrain-Rayleigh CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if arch != compiler_arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx_program = compiler._NVRTCProgram(
        GM1_CHILD_RAYLEIGH_SOURCE, "gm1_child_terrain_rayleigh_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        GM1_CHILD_RAYLEIGH_SOURCE, "gm1_child_terrain_rayleigh_cubin.cu", method="cubin"
    )
    cubin, _ = cubin_program.compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if re.search(
        rf"\.entry\s+{GM1_CHILD_RAYLEIGH_KERNEL_NAME}(?:\s|\()",
        ptx.decode("utf-8"),
    ) is None:
        raise ValueError("compiled PTX lacks the child terrain-Rayleigh entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child terrain-Rayleigh image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function(GM1_CHILD_RAYLEIGH_KERNEL_NAME)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ChildTerrainRayleigh(
        module_owner=module,
        kernel=kernel,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_rayleigh_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = child_terrain_rayleigh_shapes(ni, nj, nk)
    missing = [name for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA child terrain-Rayleigh arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in CHILD_TERRAIN_RAYLEIGH_ARRAYS:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be a C-order CUDA float32 array with shape {shapes[name]}"
            )
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("CUDA child terrain-Rayleigh arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaGm1ChildTerrainRayleighLaunch:
    ready_event: Any
    compiled: CompiledGm1ChildTerrainRayleigh
    receipt: dict[str, Any]


def launch_cuda_gm1_child_terrain_rayleigh(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RayleighHeldSelection = DEFAULT_RAYLEIGH_HELD_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGm1ChildTerrainRayleigh | None = None,
) -> CudaGm1ChildTerrainRayleighLaunch:
    """Enqueue terrain-aware held tendencies without host synchronization."""

    _validate_rayleigh_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_rayleigh_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    if compiled is None:
        compiled = compile_cuda_gm1_child_terrain_rayleigh()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    count = max(nk * nj * (ni + 1), nk * (nj + 1) * ni, nk * nj * ni)
    grid = (count + GM1_CHILD_RAYLEIGH_THREADS - 1) // GM1_CHILD_RAYLEIGH_THREADS
    compiled.kernel(
        (grid, 1, 1),
        (GM1_CHILD_RAYLEIGH_THREADS, 1, 1),
        (
            a["tauh"], a["taus"], a["tauf"], a["u0"], a["v0"],
            a["ua"], a["va"], a["wa"], a["tha"], a["uten1"], a["vten1"],
            a["wten1"], a["thten1"], cupy.uint32(ni), cupy.uint32(nj),
            cupy.uint32(nk), cupy.float32(selection.damping_rate_s_inv),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": GM1_CHILD_RAYLEIGH_SCHEMA,
        "abi": GM1_CHILD_RAYLEIGH_ABI,
        "claim_limit": GM1_CHILD_RAYLEIGH_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "production_grid": [GM1_CHILD_NI, GM1_CHILD_NJ, GM1_CHILD_NK],
        "production_shape": [ni, nj, nk] == [GM1_CHILD_NI, GM1_CHILD_NJ, GM1_CHILD_NK],
        "base_wind": {
            "u0_shape": list(a["u0"].shape),
            "v0_shape": list(a["v0"].shape),
            "terrain_varying": True,
            "legacy_1d_broadcast_allowed": False,
        },
        "cadence": "once per large step in solve1, before all three solve2 RK stages",
        "kernel": {
            "name": GM1_CHILD_RAYLEIGH_KERNEL_NAME,
            "source_sha256": GM1_CHILD_RAYLEIGH_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "block_x": GM1_CHILD_RAYLEIGH_THREADS,
            "grid_x": grid,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaGm1ChildTerrainRayleighLaunch(event, compiled, receipt)


GM1_CHILD_THETA_PACK_SCHEMA = "gm1.cuda-child-base3d-theta-pack/v1"
GM1_CHILD_THETA_PACK_ABI = "gm1.cuda-child-base3d-theta-pack-f32/v1"
GM1_CHILD_THETA_PACK_KERNEL_NAME = "gm1_child_rk_pack_theta_base3d_f32_v1"
GM1_CHILD_THETA_PACK_SOURCE = r"""
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_child_rk_pack_theta_base3d_f32_v1(
 const float* __restrict__ th0,const float* __restrict__ th,float* __restrict__ p,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned long long n=(unsigned long long)nk*nj*ni;
 const unsigned int nx=ni+2u*ng,ny=nj+2u*ng;
 for(;q<n;q+=step){
  unsigned int i=(unsigned int)(q%ni),j=(unsigned int)((q/ni)%nj),k=(unsigned int)(q/((unsigned long long)ni*nj));
  p[i3(k+1u,j+ng,i+ng,ny,nx)]=a32(s32(th0[q],300.0f),th[q]);
 }}
"""
GM1_CHILD_THETA_PACK_SOURCE_SHA256 = (
    "f6e1b2d054ea538f772965843b7c7d01629ff9a3998da48bd3fd97d6841154a5"
)


@dataclass(frozen=True, slots=True)
class CompiledGm1ChildThetaPack:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_child_theta_pack() -> CompiledGm1ChildThetaPack:
    source_hash = hashlib.sha256(GM1_CHILD_THETA_PACK_SOURCE.encode("utf-8")).hexdigest()
    if source_hash != GM1_CHILD_THETA_PACK_SOURCE_SHA256:
        raise RuntimeError("packaged child base3d theta-pack source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        GM1_CHILD_THETA_PACK_SOURCE, "gm1_child_theta_pack_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        GM1_CHILD_THETA_PACK_SOURCE, "gm1_child_theta_pack_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if re.search(
        rf"\.entry\s+{GM1_CHILD_THETA_PACK_KERNEL_NAME}(?:\s|\()", ptx.decode("utf-8")
    ) is None:
        raise ValueError("compiled PTX lacks the child base3d theta-pack entry")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child theta-pack image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function(GM1_CHILD_THETA_PACK_KERNEL_NAME)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ChildThetaPack(
        module, kernel, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


@dataclass(frozen=True, slots=True)
class Gm1ChildScalarMoistRKBinding:
    state: CudaScalarMoistRKState
    compiled: CompiledScalarMoistRKKernels
    child_pack: CompiledGm1ChildThetaPack
    legacy_compiled: CompiledScalarMoistRKKernels


def create_cuda_gm1_child_scalar_moist_rk_state(
    carriers: Mapping[str, Any],
    held: Mapping[str, Any],
    th0_3d: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any | None = None,
    legacy_compiled: CompiledScalarMoistRKKernels | None = None,
    child_pack: CompiledGm1ChildThetaPack | None = None,
) -> Gm1ChildScalarMoistRKBinding:
    """Bind the existing source-ordered RK family with a truthful base3d pack."""

    cupy = importlib.import_module("cupy")
    if (
        not isinstance(th0_3d, cupy.ndarray)
        or th0_3d.dtype != cupy.dtype("float32")
        or th0_3d.shape != (nk, nj, ni)
        or not th0_3d.flags.c_contiguous
    ):
        raise ValueError(f"th0_3d must be a C-order CUDA float32 array with shape {(nk, nj, ni)}")
    if int(cupy.cuda.Device().id) != int(th0_3d.device.id):
        raise ValueError("th0_3d is resident on another CUDA device")
    stream = stream or cupy.cuda.get_current_stream()
    legacy_compiled = legacy_compiled or compile_cuda_scalar_moist_rk()
    child_pack = child_pack or compile_cuda_gm1_child_theta_pack()
    if legacy_compiled.target_arch != child_pack.target_arch:
        raise ValueError("child theta pack and scalar/moist RK target different architectures")
    # The legacy constructor owns scratch/cadence validation.  Its temporary
    # profile is never launched; the state is rebound to the exact 3-D base
    # before it is returned to the caller.
    temporary_profile = cupy.empty((nk,), dtype=cupy.float32)
    state = create_cuda_scalar_moist_rk_state(
        carriers,
        held,
        temporary_profile,
        ni=ni,
        nj=nj,
        nk=nk,
        stream=stream,
    )
    state.th0 = th0_3d
    hybrid = CompiledScalarMoistRKKernels(
        (legacy_compiled.module_owner, child_pack.module_owner),
        (child_pack.kernel, *legacy_compiled.kernels[1:]),
        legacy_compiled.ptx,
        legacy_compiled.cubin,
        legacy_compiled.target_arch,
        legacy_compiled.ptx_options,
        legacy_compiled.cubin_options,
        legacy_compiled.runtime_identity,
    )
    return Gm1ChildScalarMoistRKBinding(state, hybrid, child_pack, legacy_compiled)


@dataclass(frozen=True, slots=True)
class CudaGm1ChildScalarMoistRKStageLaunch:
    underlying: CudaScalarMoistRKStageLaunch
    ready_event: Any
    receipt: dict[str, Any]


def launch_cuda_gm1_child_scalar_moist_rk_stage(
    binding: Gm1ChildScalarMoistRKBinding,
    mass_flux: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    theta_divergence_term: Any,
    *,
    nrk: int,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    stream: Any | None = None,
    advection_compiled: Any | None = None,
    open_compiled: Any | None = None,
) -> CudaGm1ChildScalarMoistRKStageLaunch:
    """Launch the legacy RK family with only its 1-D theta pack replaced."""

    if not isinstance(binding, Gm1ChildScalarMoistRKBinding):
        raise TypeError("binding must be a Gm1ChildScalarMoistRKBinding")
    launch = launch_cuda_scalar_moist_rk_stage(
        binding.state,
        mass_flux,
        metrics,
        divx,
        theta_divergence_term,
        nrk=nrk,
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        stream=stream,
        compiled=binding.compiled,
        advection_compiled=advection_compiled,
        open_compiled=open_compiled,
    )
    receipt = dict(launch.receipt)
    # v2: the wrapped gpuwm-cm1.cuda-scalar-moist-rk/v2 keys (numq, one species
    # name and weps per bound row) are carried at this receipt's top level.
    # v3: so is the wrapped v3 ``moisture_advection``.
    receipt["schema"] = "gm1.cuda-child-scalar-moist-rk-stage/v3"
    receipt["claim_limit"] = (
        "CM1 r21.1 scalar/moist RK family with the theta total-state pack changed "
        "only from th0[k] to the admitted terrain child th0[k,j,i]; no complete "
        "child stage, large step, trajectory, forecast, or tornado claim"
    )
    receipt["base_state"] = {
        "theta_base_shape": list(binding.state.th0.shape),
        "terrain_varying": True,
        "legacy_1d_broadcast_allowed": False,
    }
    receipt["legacy_kernel_family"] = {
        "source_sha256": launch.receipt["kernel"]["source_sha256"],
        "ptx_sha256": launch.receipt["kernel"]["ptx_sha256"],
        "cubin_sha256": launch.receipt["kernel"]["cubin_sha256"],
        "legacy_theta_pack_entry_used": False,
        "used_kernel_indices": list(range(1, len(binding.legacy_compiled.kernels))),
    }
    receipt["child_theta_pack"] = {
        "schema": GM1_CHILD_THETA_PACK_SCHEMA,
        "abi": GM1_CHILD_THETA_PACK_ABI,
        "kernel_name": GM1_CHILD_THETA_PACK_KERNEL_NAME,
        "source_sha256": GM1_CHILD_THETA_PACK_SOURCE_SHA256,
        "ptx_sha256": hashlib.sha256(binding.child_pack.ptx).hexdigest(),
        "cubin_sha256": hashlib.sha256(binding.child_pack.cubin).hexdigest(),
        "source_anchor": "src/adv.F scalar total-theta packing at the selected RK stage",
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return CudaGm1ChildScalarMoistRKStageLaunch(launch, launch.ready_event, receipt)


__all__ = [
    "CHILD_TERRAIN_RAYLEIGH_ARRAYS",
    "CHILD_TERRAIN_RAYLEIGH_INPUTS",
    "CHILD_TERRAIN_RAYLEIGH_OUTPUTS",
    "CompiledGm1ChildTerrainRayleigh",
    "CompiledGm1ChildThetaPack",
    "CudaGm1ChildTerrainRayleighLaunch",
    "CudaGm1ChildScalarMoistRKStageLaunch",
    "GM1_CHILD_BASE_WIND_ABI",
    "GM1_CHILD_FULL_STATE_STATUS",
    "GM1_CHILD_RAYLEIGH_ABI",
    "GM1_CHILD_RAYLEIGH_SCHEMA",
    "GM1_CHILD_RAYLEIGH_SOURCE",
    "GM1_CHILD_RAYLEIGH_SOURCE_SHA256",
    "GM1_CHILD_THETA_PACK_ABI",
    "GM1_CHILD_THETA_PACK_SCHEMA",
    "GM1_CHILD_THETA_PACK_SOURCE",
    "GM1_CHILD_THETA_PACK_SOURCE_SHA256",
    "Gm1ChildScalarMoistRKBinding",
    "Gm1ChildFullStatePins",
    "LoadedGm1ChildFullStateEnvelope",
    "REAL_GM1_CHILD_FULL_STATE_PINS",
    "SOURCE_ANCHORS",
    "child_terrain_rayleigh_shapes",
    "compile_cuda_gm1_child_terrain_rayleigh",
    "compile_cuda_gm1_child_theta_pack",
    "cpu_child_terrain_rayleigh_held_inplace",
    "launch_cuda_gm1_child_terrain_rayleigh",
    "create_cuda_gm1_child_scalar_moist_rk_state",
    "launch_cuda_gm1_child_scalar_moist_rk_stage",
    "load_gm1_child_full_state_envelope",
]
