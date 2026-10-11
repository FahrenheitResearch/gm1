"""Thin device/host byte transport for the Rust-owned GM1 checkpoint ABI.

Rust defines and validates the carrier order, shapes, hashes, cadence, source
identity, and durable envelope.  This module only synchronizes a safe live
state, proves canonical views and WSM aliases, moves exact float32 words to or
from one raw arena, and invokes the Rust CLI without a shell.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GM1_CHECKPOINT_TRANSPORT_SCHEMA = "gm1.python-checkpoint-transport/v1"
GM1_CHECKPOINT_CARRIER_ABI = "gm1.forecast-checkpoint-carriers/v1"
GM1_CHECKPOINT_CLI_ABI = "gm1.forecast-checkpoint-cli/v1"
GM1_CHECKPOINT_EXECUTABLE = "gm1-forecast-checkpoint"
GM1_WSM6_ALIASES = {
    "state.pp3d": "pp3d",
    "state.prs": "prs",
    "state.q3d": "q3d",
    "state.rho": "rho",
    "state.rr": "rr",
    "state.th3d": "th3d",
}
GM1_PHYSICAL_CHECKPOINT_FIELDS = (
    "ua",
    "u3d",
    "va",
    "v3d",
    "wa",
    "w3d",
    "ppi",
    "pp3d",
    "ppx",
    "tha",
    "th3d",
    "qa",
    "q3d",
    "prs",
    "rho",
    "rr",
    "rf",
)
GM1_RADIATION_HISTORY_FIELDS = ("radbcw", "radbce", "radbcs", "radbcn")
GM1_WSM6_UNIQUE_FIELDS = (
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
)
# CM1 ptype=5.  The WSM6 list plus `component.effr` (Morrison diagnoses a rain
# effective radius; WSM6 does not) and `component.nc`, the constant cloud
# droplet number CM1 reassigns from ndcnst because src/morrison.F:325 fixes
# INUM=1 and it is therefore not on the transported axis.  The order is the
# Rust `MORRISON_UNIQUE_NAMES` order, which is the artifact's order.
GM1_MORRISON_UNIQUE_FIELDS = (
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effr",
    "component.effi",
    "component.effs",
    "component.nc",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
)
# CM1 has no aerosol-aware Thompson ptype.  The component retains WSM6's
# carriers plus WRF's two surface aerosol emission rates, which are
# two-dimensional fields the scheme reads and CM1 has no equivalent for.
GM1_THOMPSON_AEROSOL_UNIQUE_FIELDS = (
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.nwfa2d",
    "component.nifa2d",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
)
GM1_CHECKPOINT_FIELDS = (
    *GM1_PHYSICAL_CHECKPOINT_FIELDS,
    *GM1_RADIATION_HISTORY_FIELDS,
    *GM1_WSM6_UNIQUE_FIELDS,
    "closure.tkea",
)
GM1_MORRISON_CHECKPOINT_FIELDS = (
    *GM1_PHYSICAL_CHECKPOINT_FIELDS,
    *GM1_RADIATION_HISTORY_FIELDS,
    *GM1_MORRISON_UNIQUE_FIELDS,
    "closure.tkea",
)
GM1_THOMPSON_AEROSOL_CHECKPOINT_FIELDS = (
    *GM1_PHYSICAL_CHECKPOINT_FIELDS,
    *GM1_RADIATION_HISTORY_FIELDS,
    *GM1_THOMPSON_AEROSOL_UNIQUE_FIELDS,
    "closure.tkea",
)
# The transport's inventory per column component.  The default answer, for a
# caller that names none, is the WSM6 one this module has always used.
GM1_COMPONENT_CHECKPOINT_FIELDS = {
    "wsm6-mp6": GM1_CHECKPOINT_FIELDS,
    "morrison-mp10": GM1_MORRISON_CHECKPOINT_FIELDS,
    # Classic Thompson (CM1 ptype=3) and WDM6 retain exactly WSM6's carriers:
    # one latent heating rate, three effective radii and the seven surface
    # accumulators.  Neither has a component-owned prognostic like Morrison's
    # nc, so the unique list is the same object.  Their checkpoints are still
    # distinct artifact contracts -- the packed q3d carrier beside these names
    # is eight and nine rows rather than six -- and Rust's
    # Gm1CheckpointComponent carries that as a separate profile id.
    "thompson-mp8": GM1_CHECKPOINT_FIELDS,
    "thompson-aerosol-mp28": GM1_THOMPSON_AEROSOL_CHECKPOINT_FIELDS,
    # WDM6 is not selectable -- ArWen's WDM6 refuses without WRF's XLAND mask
    # (cm1core.column_component_blockers) -- but its artifact contract is
    # declared, and Rust declares the matching profile id, so a future binding
    # inherits a checkpoint rather than inventing one.
    "wdm6-mp16": GM1_CHECKPOINT_FIELDS,
}


def gm1_checkpoint_fields(component_id: str | None = None) -> tuple[str, ...]:
    """The exact carrier inventory one component's checkpoint declares.

    Rust owns the artifact contract; this is the transport's matching view of
    it.  An unknown component is refused rather than silently transported as
    six species, because the arena has no per-carrier header and the first
    wrong name renames every word that follows it.
    """

    if component_id is None:
        return GM1_CHECKPOINT_FIELDS
    try:
        return GM1_COMPONENT_CHECKPOINT_FIELDS[component_id]
    except KeyError:
        raise ValueError(
            f"component {component_id!r} declares no GM1 checkpoint carrier "
            "inventory; transporting it would move an arena whose packed scalar "
            "axis has an undeclared length"
        ) from None
GM1_CHECKPOINT_CLAIM_LIMIT = (
    "thin exact-byte capture/restore transport at the completed-large-step "
    "boundary only; Rust owns transformations and validation; no mid-stage "
    "restart, uninterrupted/resumed trajectory identity, forecast skill, or "
    "tornado claim"
)


@dataclass(frozen=True, slots=True)
class Gm1CheckpointTransportReceipt:
    schema: str
    checkpoint: Mapping[str, Any]
    raw_arena_sha256: str
    raw_arena_nbytes: int
    carrier_count: int
    model_step: int
    model_time_f64_bits: str
    mass1_f64_bits: str
    claim_limit: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "carrier_count": self.carrier_count,
            "checkpoint": dict(self.checkpoint),
            "claim_limit": self.claim_limit,
            "mass1_f64_bits": self.mass1_f64_bits,
            "model_step": self.model_step,
            "model_time_f64_bits": self.model_time_f64_bits,
            "raw_arena_nbytes": self.raw_arena_nbytes,
            "raw_arena_sha256": self.raw_arena_sha256,
            "schema": self.schema,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()


def _regular_file(path: Path | str, label: str) -> Path:
    result = Path(path)
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return result


def _executable_path(executable: Path | str | None) -> str:
    if executable is None:
        discovered = shutil.which(GM1_CHECKPOINT_EXECUTABLE)
        if discovered is None:
            raise FileNotFoundError(
                f"{GM1_CHECKPOINT_EXECUTABLE} is not installed or on PATH"
            )
        return discovered
    path = _regular_file(executable, "GM1 checkpoint executable")
    return str(path)


def _run_cli(executable: str, arguments: list[str]) -> dict[str, Any] | str:
    result = subprocess.run(
        [executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown refusal"
        raise ValueError(f"Rust GM1 checkpoint refused: {detail}")
    output = result.stdout.strip()
    if arguments and arguments[0] == "--abi":
        return output
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("Rust GM1 checkpoint returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("Rust GM1 checkpoint result must be a JSON object")
    return parsed


def _layout(executable: str, operator_manifest: Path) -> dict[str, Any]:
    result = _run_cli(executable, ["layout", str(operator_manifest)])
    if not isinstance(result, dict):
        raise RuntimeError("layout command returned non-object output")
    if (
        result.get("schema") != "gm1.forecast-checkpoint-layout/v1"
        or result.get("carrier_abi") != GM1_CHECKPOINT_CARRIER_ABI
        or result.get("model") != "GM1"
        or result.get("capture_phase") != "post-large-step"
        or result.get("dtype") != "<f4"
        or result.get("order") != "C"
    ):
        raise ValueError("Rust GM1 checkpoint layout identity changed")
    # The layout names the component it was built for; the transport takes its
    # inventory from that rather than assuming one.
    expected = gm1_checkpoint_fields(
        (result.get("configuration") or {}).get("selection", {}).get("microphysics")
        if isinstance(result.get("configuration"), dict)
        else None
    )
    carriers = result.get("carriers")
    if not isinstance(carriers, list) or len(carriers) != len(expected):
        raise ValueError("Rust GM1 checkpoint carrier inventory changed")
    names = tuple(item.get("name") for item in carriers if isinstance(item, dict))
    if set(names) != set(expected) or len(names) != len(set(names)):
        raise ValueError(
            f"Rust GM1 checkpoint layout lacks the canonical {len(expected)} carriers"
        )
    return result


def _array_metadata(value: object, name: str) -> tuple[tuple[int, ...], int, int, str]:
    np = importlib.import_module("numpy")
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    flags = getattr(value, "flags", None)
    if dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if not isinstance(shape, tuple) or not all(type(item) is int for item in shape):
        raise TypeError(f"{name} must expose an exact tuple shape")
    if flags is None or not bool(flags.c_contiguous):
        raise ValueError(f"{name} must be C contiguous")
    nbytes = int(getattr(value, "nbytes", -1))
    if nbytes != math.prod(shape) * 4:
        raise ValueError(f"{name} byte span differs from its float32 shape")
    if hasattr(value, "__cuda_array_interface__"):
        interface = value.__cuda_array_interface__
        pointer = int(interface["data"][0])
        space = "cuda"
    elif hasattr(value, "__array_interface__"):
        interface = value.__array_interface__
        pointer = int(interface["data"][0])
        space = "host"
    else:
        raise TypeError(f"{name} must expose a host or CUDA array interface")
    if pointer <= 0 or nbytes <= 0:
        raise ValueError(f"{name} has an invalid live memory span")
    return shape, pointer, nbytes, space


def _validate_views(
    carriers: Mapping[str, object],
    aliases: Mapping[str, object],
    layout: Mapping[str, Any],
) -> list[tuple[str, object, int]]:
    expected = set(spec["name"] for spec in layout["carriers"])
    if set(carriers) != expected:
        missing = sorted(expected - set(carriers))
        extra = sorted(set(carriers) - expected)
        raise ValueError(
            f"GM1 canonical carrier inventory drifted; missing={missing}, extra={extra}"
        )
    if set(aliases) != set(GM1_WSM6_ALIASES):
        missing = sorted(set(GM1_WSM6_ALIASES) - set(aliases))
        extra = sorted(set(aliases) - set(GM1_WSM6_ALIASES))
        raise ValueError(
            f"GM1 WSM alias inventory drifted; missing={missing}, extra={extra}"
        )
    ordered: list[tuple[str, object, int]] = []
    spans: list[tuple[str, int, int, str]] = []
    for spec in layout["carriers"]:
        name = spec["name"]
        value = carriers[name]
        shape, pointer, nbytes, space = _array_metadata(value, name)
        expected_shape = tuple(spec["shape"])
        if shape != expected_shape or nbytes != spec["nbytes"]:
            raise ValueError(
                f"{name} has shape/bytes {shape}/{nbytes}; Rust requires "
                f"{expected_shape}/{spec['nbytes']}"
            )
        for prior_name, prior_pointer, prior_nbytes, prior_space in spans:
            if space == prior_space and max(pointer, prior_pointer) < min(
                pointer + nbytes, prior_pointer + prior_nbytes
            ):
                raise ValueError(
                    f"canonical GM1 carriers {prior_name} and {name} alias"
                )
        spans.append((name, pointer, nbytes, space))
        ordered.append((name, value, nbytes))
    for alias, canonical in GM1_WSM6_ALIASES.items():
        alias_meta = _array_metadata(aliases[alias], alias)
        canonical_meta = _array_metadata(carriers[canonical], canonical)
        if alias_meta != canonical_meta:
            raise ValueError(
                f"WSM alias {alias} is not the exact {canonical} live allocation/span"
            )
    return ordered


def _host_bytes(value: object, name: str) -> bytes:
    np = importlib.import_module("numpy")
    if hasattr(value, "__cuda_array_interface__"):
        cupy = importlib.import_module("cupy")
        host = cupy.asnumpy(value, order="C")
    else:
        host = np.asarray(value, dtype=np.float32, order="C")
    if host.dtype != np.dtype("float32") or not host.flags.c_contiguous:
        raise RuntimeError(f"{name} host transfer changed dtype/order")
    return host.tobytes(order="C")


def _copy_from_host(destination: object, source: object, name: str) -> None:
    np = importlib.import_module("numpy")
    if hasattr(destination, "__cuda_array_interface__"):
        cupy = importlib.import_module("cupy")
        cupy.copyto(destination, cupy.asarray(source))
    else:
        np.copyto(destination, source, casting="no")
    if getattr(destination, "dtype", None) != np.dtype("float32"):
        raise RuntimeError(f"{name} restore destination dtype changed")


def _raw_words_equal(left: object, right: object) -> bool:
    np = importlib.import_module("numpy")
    if hasattr(left, "__cuda_array_interface__") or hasattr(
        right, "__cuda_array_interface__"
    ):
        if not (
            hasattr(left, "__cuda_array_interface__")
            and hasattr(right, "__cuda_array_interface__")
        ):
            return False
        cupy = importlib.import_module("cupy")
        return bool(
            cupy.array_equal(left.view(cupy.uint32), right.view(cupy.uint32)).item()
        )
    return bool(
        np.array_equal(
            np.asarray(left).view(np.uint32), np.asarray(right).view(np.uint32)
        )
    )


def compose_live_gm1_checkpoint_views(
    state: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Synchronize and expose the exact live completed-step carrier inventory."""

    ready_event = getattr(state, "ready_event", None)
    if ready_event is None or not callable(getattr(ready_event, "synchronize", None)):
        raise TypeError("GM1 forecast state lacks a synchronizable ready_event")
    ready_event.synchronize()
    scalar = state.scalar_rk
    moist = state.moist_bridge
    closure = state.closure
    transition = state.transition
    step = getattr(state, "step_index", None)
    completed = getattr(transition, "completed_transitions", None)
    if (
        type(step) is not int
        or step <= 0
        or completed != step
        or state.next_stage != 1
        or state.solve1_ready is not False
        or scalar.next_nrk != 1
        or scalar.completed is not False
        or moist.terms_ready is not False
        or moist.buoyancy_applied is not False
        or moist.post_acoustic_ready is not False
        or closure.completed_stage != 0
    ):
        raise ValueError(
            "GM1 forecast state is not at the safe post-large-step boundary"
        )
    carriers = {
        name: state.product.field(name) for name in GM1_PHYSICAL_CHECKPOINT_FIELDS
    }
    carriers.update(
        {name: state.prepared.arrays[name] for name in GM1_RADIATION_HISTORY_FIELDS}
    )
    transition_restart = transition.capture_restart()
    carriers.update(
        {name: transition_restart.arrays[name] for name in GM1_WSM6_UNIQUE_FIELDS}
    )
    closure_arrays = closure.checkpoint_arrays()
    if set(closure_arrays) != {"tkea"}:
        raise ValueError("GM1 closure checkpoint inventory changed")
    carriers["closure.tkea"] = closure_arrays["tkea"]
    if set(closure.tke) < {"tke3d"} or not _raw_words_equal(
        closure.tke["tke3d"], closure_arrays["tkea"]
    ):
        raise ValueError(
            "GM1 safe boundary requires recomputable closure tke3d to be raw-word "
            "identical to persistent tkea"
        )
    aliases = {alias: transition_restart.arrays[alias] for alias in GM1_WSM6_ALIASES}
    cadence = {
        "completed_transitions": completed,
        "mass1_f64": state.mass1_f64,
        "model_step": step,
        "model_time_s": state.time_seconds,
        "scheduler_entries": {
            "closure.completed_stage": closure.completed_stage,
            "closure.refresh_index": closure.refresh_index,
            "closure.tke3d_reconstructible_from_tkea": True,
        },
    }
    return carriers, aliases, cadence


def capture_live_gm1_checkpoint(
    state: object,
    output: Path | str,
    operator_manifest: Path | str,
    *,
    executable: Path | str | None = None,
    scheduler_entries: Mapping[str, object] | None = None,
    source_identities: Path | str | None = None,
) -> Gm1CheckpointTransportReceipt:
    """Capture one synchronized completed-large-step state through Rust."""

    executable_path = _executable_path(executable)
    abi = _run_cli(executable_path, ["--abi"])
    if not isinstance(abi, str) or GM1_CHECKPOINT_CLI_ABI not in abi:
        raise ValueError("GM1 checkpoint CLI ABI changed")
    operator_path = _regular_file(operator_manifest, "GM1 operator manifest")
    source_identity_path = (
        None
        if source_identities is None
        else _regular_file(source_identities, "GM1 source identities")
    )
    layout = _layout(executable_path, operator_path)
    carriers, aliases, cadence = compose_live_gm1_checkpoint_views(state)
    ordered = _validate_views(carriers, aliases, layout)
    entries = dict(cadence["scheduler_entries"])
    if scheduler_entries is not None:
        collisions = set(entries) & set(scheduler_entries)
        if collisions:
            raise ValueError(f"scheduler extension keys collide: {sorted(collisions)}")
        entries.update(scheduler_entries)
    scheduler = _canonical_json(
        {"entries": entries, "schema": "gm1.scheduler-extension/v1"}
    )
    output = Path(output)
    parent = output.parent if output.parent != Path("") else Path(".")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace GM1 checkpoint {output}")
    if not parent.is_dir():
        raise FileNotFoundError(f"GM1 checkpoint parent does not exist: {parent}")
    digest = hashlib.sha256()
    total = 0
    with tempfile.TemporaryDirectory(
        prefix=".gm1-checkpoint-", dir=parent
    ) as temporary:
        temporary = Path(temporary)
        raw_path = temporary / "carriers.f32le"
        with raw_path.open("xb") as stream:
            for name, value, nbytes in ordered:
                payload = _host_bytes(value, name)
                if len(payload) != nbytes:
                    raise RuntimeError(f"{name} host transfer byte count changed")
                stream.write(payload)
                digest.update(payload)
                total += nbytes
            stream.flush()
            os.fsync(stream.fileno())
        scheduler_path = temporary / "scheduler.json"
        scheduler_path.write_bytes(scheduler)
        raw_sha256 = digest.hexdigest()
        step = int(cadence["model_step"])
        model_time = float(cadence["model_time_s"])
        mass1 = float(cadence["mass1_f64"])
        build_arguments = [
            "build",
            str(raw_path),
            str(output),
            raw_sha256,
            str(operator_path),
            str(step),
            struct.pack(">d", model_time).hex(),
            struct.pack(">d", mass1).hex(),
            str(cadence["completed_transitions"]),
            str(scheduler_path),
        ]
        if source_identity_path is not None:
            build_arguments.append(str(source_identity_path))
        checkpoint = _run_cli(executable_path, build_arguments)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("GM1 checkpoint build returned non-object output")
    return Gm1CheckpointTransportReceipt(
        schema=GM1_CHECKPOINT_TRANSPORT_SCHEMA,
        checkpoint=checkpoint,
        raw_arena_sha256=raw_sha256,
        raw_arena_nbytes=total,
        carrier_count=len(ordered),
        model_step=step,
        model_time_f64_bits=struct.pack(">d", model_time).hex(),
        mass1_f64_bits=struct.pack(">d", mass1).hex(),
        claim_limit=GM1_CHECKPOINT_CLAIM_LIMIT,
    )


def restore_gm1_checkpoint_to_views(
    checkpoint: Path | str,
    destinations: Mapping[str, object],
    aliases: Mapping[str, object],
    operator_manifest: Path | str,
    output_directory: Path | str,
    *,
    contract_sha256: str,
    artifact_sha256: str,
    executable: Path | str | None = None,
    source_identities: Path | str | None = None,
) -> dict[str, Any]:
    """Use Rust to verify/extract a checkpoint, then copy exact words to views."""

    executable_path = _executable_path(executable)
    operator_path = _regular_file(operator_manifest, "GM1 operator manifest")
    source_identity_path = (
        None
        if source_identities is None
        else _regular_file(source_identities, "GM1 source identities")
    )
    operator_sha256 = hashlib.sha256(operator_path.read_bytes()).hexdigest()
    layout = _layout(executable_path, operator_path)
    ordered = _validate_views(destinations, aliases, layout)
    output_directory = Path(output_directory)
    restore_arguments = [
        "restore",
        str(_regular_file(checkpoint, "GM1 checkpoint")),
        str(output_directory),
        contract_sha256,
        artifact_sha256,
        operator_sha256,
    ]
    if source_identity_path is not None:
        restore_arguments.append(str(source_identity_path))
    receipt = _run_cli(executable_path, restore_arguments)
    if not isinstance(receipt, dict):
        raise RuntimeError("GM1 checkpoint restore returned non-object output")
    arena = _regular_file(output_directory / "gm1-carriers.f32le", "restored arena")
    np = importlib.import_module("numpy")
    offset = 0
    mapped = np.memmap(arena, dtype=np.dtype("<f4"), mode="r", order="C")
    try:
        for spec, (name, destination, nbytes) in zip(
            layout["carriers"], ordered, strict=True
        ):
            if spec["name"] != name or spec["offset_nbytes"] != offset:
                raise ValueError("GM1 restore layout order/offset changed")
            words = nbytes // 4
            source = mapped[offset // 4 : offset // 4 + words].reshape(
                tuple(spec["shape"]), order="C"
            )
            _copy_from_host(destination, source, name)
            offset += nbytes
    finally:
        del mapped
    if offset != layout["payload_nbytes"]:
        raise ValueError("GM1 restored payload byte count changed")
    return receipt


__all__ = [
    "GM1_CHECKPOINT_CARRIER_ABI",
    "GM1_CHECKPOINT_FIELDS",
    "GM1_COMPONENT_CHECKPOINT_FIELDS",
    "GM1_MORRISON_CHECKPOINT_FIELDS",
    "GM1_MORRISON_UNIQUE_FIELDS",
    "GM1_THOMPSON_AEROSOL_CHECKPOINT_FIELDS",
    "GM1_THOMPSON_AEROSOL_UNIQUE_FIELDS",
    "gm1_checkpoint_fields",
    "GM1_CHECKPOINT_TRANSPORT_SCHEMA",
    "GM1_WSM6_ALIASES",
    "Gm1CheckpointTransportReceipt",
    "capture_live_gm1_checkpoint",
    "compose_live_gm1_checkpoint_views",
    "restore_gm1_checkpoint_to_views",
]
