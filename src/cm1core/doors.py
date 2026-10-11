"""Installed front-door handlers.

Every mutating door performs its complete scientific preflight before it
creates an output directory or imports a numerical runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from importlib import resources
from pathlib import Path
import re
import subprocess
import tempfile

from . import CM1_AUTHORITY_SHA256, CM1_RELEASE, MATURITY, __version__
from .config import load_config
from .doctor import blocking as doctor_blocking
from .doctor import collect as doctor_collect
from .doctor import render as doctor_render
from .doctor import unvalidated as doctor_unvalidated
from .preflight import blockers, require_runnable
from .profiles import CARTESIAN_LANE_NAME, cartesian_live_admission


def version_door(arguments: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "authority_sha256": CM1_AUTHORITY_SHA256,
                "cm1_release": CM1_RELEASE,
                "distribution": "gpuwm-cm1",
                "import": "cm1core",
                "maturity": MATURITY,
                "version": __version__,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def doctor_door(arguments: argparse.Namespace) -> int:
    if getattr(arguments, "plain", False) or getattr(arguments, "report", False):
        from .cm1run.system import doctor_plain

        return doctor_plain(arguments)
    findings = doctor_collect()
    if arguments.json:
        print(
            json.dumps(
                {
                    "blocking": [item.subject for item in doctor_blocking(findings)],
                    "distribution": "gpuwm-cm1",
                    "findings": [item.as_dict() for item in findings],
                    # Exposed and running, with no CM1 differential behind it.
                    # A caller that reads only this key still cannot miss one.
                    "unvalidated": [
                        item.as_dict() for item in doctor_unvalidated(findings)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(doctor_render(findings, explain=arguments.explain))
    return 1 if doctor_blocking(findings) else 0


def check_door(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    payload = {
        "config": config.canonical_dict(),
        "config_sha256": config.sha256,
        "executable": False,
        "schema": config.schema,
        "status": "accepted-grammar-and-first-slice",
    }
    if config.run.lane.value == CARTESIAN_LANE_NAME:
        # config.profile_identity is the sealed identity config_sha256 is taken
        # over, so its text cannot move; its native_ptype_5 line predates the
        # lane's other components.  The live differences sit beside it.
        payload["product_differences"] = cartesian_live_admission()["product_differences"]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def preflight_door(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    gaps = blockers(config)
    payload = {
        "blockers": [gap.as_dict() for gap in gaps],
        "config_sha256": config.sha256,
        "lane": config.run.lane.value,
        "runnable": not gaps,
        "status": "passed" if not gaps else "refused",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not gaps else 1


def run_door(arguments: argparse.Namespace) -> int:
    from .errors import refuse

    config = load_config(arguments.config)
    # Do not resolve/create/touch arguments.output above this line.  The first
    # runnable implementation will receive a sealed path only after admission.
    require_runnable(config)
    # A lane whose blockers are all retired still needs a bound backend.  A
    # bare assertion here would print a traceback and leave an empty output
    # directory that reads as a completed CM1 forecast.
    refuse(
        code="CM1-RUN-BACKEND-MISSING",
        gate="product-door",
        reason=f"the {config.run.lane.value} lane passed admission but this distribution binds no device-resident backend to the init/forecast doors",
        remedy="run the admitted `gpuwm-cm1 cartesian` lane, or wait for the product forecast backend",
        option="run.lane",
        value=config.run.lane.value,
    )


def restart_door(arguments: argparse.Namespace) -> int:
    from .errors import refuse
    from .rust_tools import NativeToolError, verify_native_tools

    requested_checkpoint = Path(arguments.checkpoint).expanduser()
    if requested_checkpoint.is_symlink():
        refuse(
            code="CM1-RESTART-PATH",
            gate="restart-identity",
            reason="the checkpoint path is a symlink",
            remedy="supply the existing regular checkpoint artifact directly",
            option="checkpoint",
            value=str(arguments.checkpoint),
        )
    try:
        checkpoint = requested_checkpoint.resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-RESTART-PATH",
            gate="restart-identity",
            reason=f"the checkpoint path cannot be resolved exactly: {error}",
            remedy="supply an existing regular gpuwm-cm1 restart artifact",
            option="checkpoint",
            value=str(arguments.checkpoint),
        )
    if checkpoint.is_symlink() or not checkpoint.is_file():
        refuse(
            code="CM1-RESTART-PATH",
            gate="restart-identity",
            reason="the checkpoint path is not a regular file",
            remedy="supply an existing regular gpuwm-cm1 restart artifact",
            option="checkpoint",
            value=str(checkpoint),
        )

    # This door drives the 37-carrier gm1-forecast-checkpoint profile.  The
    # Cartesian lane writes a different profile into the same CM1FCP1 envelope
    # and ships its own tool, so an artifact that carries a Cartesian sidecar
    # is named here rather than handed to a tool that will refuse it with a
    # contract mismatch and leave a reader thinking the restart is corrupt.
    sidecar = checkpoint.with_suffix(".json")
    if sidecar.is_file() and not sidecar.is_symlink():
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            document = None
        if (
            isinstance(document, dict)
            and document.get("schema") == "gm1.cartesian-restart-sidecar/v1"
        ):
            refuse(
                code="CM1-RESTART-PROFILE",
                gate="restart-identity",
                reason=(
                    f"{checkpoint.name} is a Cartesian-lane restart: "
                    f"{sidecar.name} beside it is a "
                    "gm1.cartesian-restart-sidecar/v1 document naming "
                    f"{document.get('carrier_count')} carriers at step "
                    f"{document.get('step')}. This door drives the 37-carrier "
                    "gm1-forecast-checkpoint profile, which would refuse it on "
                    "a contract mismatch and read as a damaged artifact"
                ),
                remedy=(
                    "resume it with `gpuwm-cm1 cartesian <config> --integrate "
                    "--restart-dir <its directory>` and [restart] irst = 1, "
                    "rstnum = <its number>; to inspect or extract it without "
                    "running, use `gm1-cartesian-restart inspect` or "
                    "`gm1-cartesian-restart restore` with the contract, "
                    "artifact and inventory hashes the sidecar records"
                ),
                option="checkpoint",
                value=str(checkpoint),
            )
    hashes = {
        "contract_sha256": arguments.contract_sha256,
        "artifact_sha256": arguments.artifact_sha256,
        "operator_manifest_sha256": arguments.operator_manifest_sha256,
    }
    for name, digest in hashes.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            refuse(
                code="CM1-RESTART-IDENTITY",
                gate="restart-identity",
                reason=f"{name} is not a canonical lowercase SHA-256",
                remedy="pass the independently recorded checkpoint/operator hash",
                option=name,
                value=digest,
            )
    requested = Path(arguments.output).expanduser()
    if requested.exists() or requested.is_symlink():
        refuse(
            code="CM1-RESTART-OUTPUT",
            gate="restart-identity",
            reason="the restart output already exists or is a symlink",
            remedy="select a new output directory under an existing real parent",
            option="output",
            value=str(requested),
        )
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-RESTART-OUTPUT",
            gate="restart-identity",
            reason=f"the restart output parent cannot be resolved exactly: {error}",
            remedy="select a new output directory under an existing real parent",
            option="output",
            value=str(requested),
        )
    if not parent.is_dir() or not requested.name or requested.name in {".", ".."}:
        refuse(
            code="CM1-RESTART-OUTPUT",
            gate="restart-identity",
            reason="the restart output parent/name is not a safe new directory target",
            remedy="select a simple new child name under an existing directory",
            option="output",
            value=str(requested),
        )
    output = parent / requested.name
    try:
        executable = verify_native_tools()["gm1-forecast-checkpoint"]
    except (OSError, NativeToolError, KeyError) as error:
        refuse(
            code="CM1-RUST-RESTART-TOOL",
            gate="restart-identity",
            reason=f"the installed verified GM1 restart tool is unavailable: {error}",
            remedy="reinstall a platform wheel containing the sealed Rust tools",
        )
    result = subprocess.run(
        [
            str(executable),
            "restore",
            str(checkpoint),
            str(output),
            hashes["contract_sha256"],
            hashes["artifact_sha256"],
            hashes["operator_manifest_sha256"],
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if output.exists() or output.is_symlink():
            refuse(
                code="CM1-RUST-RESTART-PARTIAL",
                gate="restart-identity",
                reason="the Rust restart refusal unexpectedly left an output target",
                remedy="preserve the target for forensic inspection and choose a new path",
                option="output",
                value=str(output),
            )
        refuse(
            code="CM1-RUST-RESTART-FAILED",
            gate="restart-identity",
            reason=result.stderr.strip()
            or result.stdout.strip()
            or "Rust refused without detail",
            remedy="supply the exact artifact, contract, operator hash, and a new output path",
        )
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        refuse(
            code="CM1-RUST-RESTART-RECEIPT",
            gate="restart-identity",
            reason=f"the verified Rust tool emitted invalid JSON: {error}",
            remedy="reinstall the sealed platform wheel",
        )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "gm1.forecast-restore-receipt/v1"
        or receipt.get("model") != "GM1"
        or receipt.get("checkpoint_contract_sha256") != hashes["contract_sha256"]
        or receipt.get("checkpoint_artifact_sha256") != hashes["artifact_sha256"]
        or receipt.get("carrier_abi") != "gm1.forecast-checkpoint-carriers/v1"
        or receipt.get("carrier_count") != 37
        or not output.is_dir()
        or not (output / "manifest.json").is_file()
        or not (output / "receipt.json").is_file()
        or not (output / "gm1-carriers.f32le").is_file()
    ):
        refuse(
            code="CM1-RUST-RESTART-RECEIPT",
            gate="restart-identity",
            reason="the verified Rust restore output differs from the GM1 v1 contract",
            remedy="preserve the output for forensic inspection and reinstall the sealed wheel",
            option="output",
            value=str(output),
        )
    payload = {
        "model": "GM1",
        "output": str(output),
        "rust_restore": receipt,
        "schema": "gpuwm-cm1.restart/v1",
        "status": "restored-before-next-large-step",
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


_ENVELOPE_MAGIC = {"cm1fhs": b"CM1FHS1\0", "cm1fcp": b"CM1FCP1\0"}
_MAX_ENVELOPE_HEADER_BYTES = 1024 * 1024
#: Carriers a reduced-profile frame omits and cm1-render rebuilds, with the
#: carrier whose extent the rebuilt one takes (rust/cm1-state/src/reduced_history.rs).
_RECONSTRUCTED_EXTENT = {"prs": "pp3d", "zf": "w3d"}


def _forecast_envelope_header(history: Path, history_format: str) -> dict:
    """Read the CM1FHS1/CM1FCP1 prefix and JSON header, nothing else.

    Rust still verifies the whole envelope (canonical header, contract object,
    every payload hash); this only reads what the door needs to default and
    cross-check its arguments, so a malformed prefix is refused here by name.
    """

    from .errors import refuse

    def malformed(reason: str) -> None:
        refuse(
            code="CM1-HISTORY-ENVELOPE",
            gate="rust-io-render",
            reason=f"{history.name} is not a readable {history_format} envelope: {reason}",
            remedy="pass a frame written by the GM1 history publisher or checkpoint writer",
            option="history",
            value=str(history),
        )

    with history.open("rb") as stream:
        prefix = stream.read(16)
        if len(prefix) < 16 or prefix[:8] != _ENVELOPE_MAGIC[history_format]:
            malformed("the 8-byte magic does not identify the format")
        length = int.from_bytes(prefix[8:16], "little")
        if length > _MAX_ENVELOPE_HEADER_BYTES:
            malformed(f"the declared header length {length} exceeds 1 MiB")
        raw = stream.read(length)
    if len(raw) != length:
        malformed("the header extends past the end of the file")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        malformed(f"the header is not UTF-8 JSON ({error})")
    if not isinstance(header, dict):
        malformed("the header is not a JSON object")
    return header


def _publication_receipt_contract(history: Path) -> tuple[str | None, str | None]:
    """The contract hash the publisher's sibling receipt recorded, if any."""

    receipt = history.parent / "history-publication.json"
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        envelope = payload["forecast_envelope"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    if not isinstance(envelope, dict) or Path(str(envelope.get("path", ""))).name != history.name:
        return None, None
    for line in str(envelope.get("rust_output", "")).splitlines():
        if line.startswith("contract_sha256="):
            return line.split("=", 1)[1].strip(), "publication-receipt"
    return None, None


def _envelope_dimensions(header: dict, fields: list[str]) -> dict[str, int | None]:
    """nx/ny/nz implied by the requested fields' shapes in the header contract."""

    specs = {}
    contract = header.get("contract")
    if isinstance(contract, dict) and isinstance(contract.get("array_specs"), list):
        for spec in contract["array_specs"]:
            if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                specs[spec["name"]] = spec.get("shape")
    implied: dict[str, int | None] = {"nx": None, "ny": None, "nz": None}
    for field in fields:
        shape = specs.get(field)
        if shape is None and field in _RECONSTRUCTED_EXTENT:
            shape = specs.get(_RECONSTRUCTED_EXTENT[field])
        if not isinstance(shape, list) or not all(
            isinstance(extent, int) and not isinstance(extent, bool) for extent in shape
        ):
            continue
        if len(shape) == 2:
            candidate = {"nz": 1, "ny": shape[0], "nx": shape[1]}
        elif len(shape) == 3:
            candidate = {"nz": shape[0], "ny": shape[1], "nx": shape[2]}
        else:
            continue
        if implied["nx"] is None:
            implied = dict(candidate)
        elif candidate != implied:
            # The renderer takes one grid for every field; fields of two
            # extents cannot be defaulted, so leave the axes that disagree to
            # the caller and let the renderer's own shape check speak.
            implied = {
                axis: implied[axis] if implied[axis] == candidate[axis] else None
                for axis in implied
            }
    return implied


def render_door(arguments: argparse.Namespace) -> int:
    from .errors import refuse
    from .rust_tools import NativeToolError, verify_native_tools

    try:
        history = Path(arguments.history).expanduser().resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-HISTORY-PATH",
            gate="rust-io-render",
            reason=f"the history path cannot be resolved exactly: {error}",
            remedy="supply an existing regular GM1 state-history artifact",
            option="history",
            value=str(arguments.history),
        )
    if not history.is_file():
        refuse(
            code="CM1-HISTORY-PATH",
            gate="rust-io-render",
            reason="the history path is not a regular file",
            remedy="supply an existing regular GM1 state-history artifact",
            option="history",
            value=str(history),
        )

    history_format = arguments.history_format
    if history_format is None:
        history_format = {
            ".cm1fhs": "cm1fhs",
            ".cm1fcp": "cm1fcp",
        }.get(history.suffix.lower())
    if history_format not in {"cm1fhs", "cm1fcp"}:
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason="the strict render door accepts only CM1FHS1 history or CM1FCP1 checkpoint envelopes",
            remedy="use a .cm1fhs/.cm1fcp input or pass --format cm1fhs|cm1fcp",
            option="format",
            value=history_format,
        )
    header = _forecast_envelope_header(history, history_format)
    header_digest = header.get("contract_sha256")
    digest = arguments.contract_sha256
    if digest is None:
        # The publisher writes history-publication.json beside every frame,
        # carrying the contract hash Rust reported when it sealed the file.
        # That receipt is a separate artifact from the envelope, so it still
        # anchors admission independently; only when it is absent does the
        # door fall back to the envelope's own header, and the render receipt
        # then says the contract was self-described rather than anchored.
        digest, contract_source = _publication_receipt_contract(history)
        if digest is None:
            digest, contract_source = header_digest, "frame-header-self-described"
    else:
        contract_source = "argument"
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason="the forecast-contract hash must be 64 lowercase hexadecimal characters",
            remedy="pass --contract-sha256 from the run receipt, or omit it to read the frame's publication receipt or header",
            option="contract_sha256",
            value=digest,
        )
    if digest != header_digest:
        refuse(
            code="CM1-RENDER-CONTRACT-MISMATCH",
            gate="rust-io-render",
            reason=(
                f"the {contract_source} contract hash {digest} is not the "
                f"contract {header_digest} this envelope's header declares; the "
                "frame is not the artifact the anchoring receipt describes"
            ),
            remedy="render the frame the receipt was written for, or pass the receipt that belongs to this frame",
            option="contract_sha256",
            value=digest,
        )
    products = {
        # CM1 writes both reflectivity products: writeout.F:1081-1091 the
        # composite (column maximum, misclibs.F:2259-2292) and
        # writeout.F:4089-4107 the volume, which a user slices at one level.
        # Both are painted on the standard NWS dBZ scale by the renderer.
        "reflectivity",
        "reflectivity-level",
        "updraft",
        "vorticity",
        "cold-pool",
        "scalar",
        "supercell",
        "vortex-composite",
        # The renderer has always accepted this three-channel composite; the
        # door did not name it, so no caller could select it.
        "storm-fusion",
        # CM1 r21.1 writeout.F:2937-3028.  The four environment planes are
        # surface fields, one value per column, so they are rendered with
        # --nz 1 --level 0 against the carrier of the same name in the frame.
        "cape",
        "cin",
        "lcl",
        "lfc",
    }
    # Products that take a single horizontal level out of a volume.  The rest
    # reduce the whole column, so a --level for them would be declared and
    # silently unread.
    level_products = {"reflectivity-level", "vorticity", "cold-pool", "scalar"}
    product = arguments.product
    if product not in products:
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason=f"unknown render product; admitted values are {sorted(products)}",
            remedy="select one admitted --product",
            option="product",
            value=product,
        )
    fields = list(arguments.field)
    required_fields = (
        3 if product in {"supercell", "vortex-composite", "storm-fusion"} else 1
    )
    if len(fields) != required_fields or any(not value for value in fields):
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason=f"product {product!r} requires exactly {required_fields} named field(s)",
            remedy="pass one --field, or three ordered reflectivity/updraft/vorticity fields for supercell or vortex-composite",
            option="field",
            value=fields,
        )
    implied = _envelope_dimensions(header, fields)
    for axis in ("nx", "ny", "nz"):
        declared = getattr(arguments, axis)
        if declared is None:
            setattr(arguments, axis, implied[axis])
        elif implied[axis] is not None and declared != implied[axis]:
            refuse(
                code="CM1-RENDER-REQUEST",
                gate="rust-io-render",
                reason=(
                    f"--{axis} {declared} contradicts the envelope header, whose "
                    f"contract gives {', '.join(fields)} the grid "
                    f"nx={implied['nx']}, ny={implied['ny']}, nz={implied['nz']}; "
                    "the renderer would refuse the shape after decoding the whole frame"
                ),
                remedy=f"omit --{axis} to take it from the header, or pass {implied[axis]}",
                option=axis,
                value=declared,
            )
    dimensions = {"nx": arguments.nx, "ny": arguments.ny, "nz": arguments.nz}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions.values()
    ):
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason="nx, ny, and nz must each be positive integers matching the sealed contract",
            remedy="pass --nx, --ny, and --nz from the independent run receipt",
            option="dimensions",
            value=dimensions,
        )
    if arguments.level < 0 or (arguments.record is not None and arguments.record < 0):
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason="record and level indices cannot be negative",
            remedy="pass zero-based non-negative --record/--level indices",
            option="indices",
            value={"record": arguments.record, "level": arguments.level},
        )
    if product in level_products and arguments.level >= arguments.nz:
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason=(
                f"product {product!r} draws the horizontal plane at --level "
                f"{arguments.level}, but the declared volume has only "
                f"{arguments.nz} level(s); the renderer would read past the "
                "field and paint another column's words"
            ),
            remedy=f"pass --level between 0 and {arguments.nz - 1}",
            option="level",
            value=arguments.level,
        )
    if arguments.valid_seconds is not None and (
        not math.isfinite(arguments.valid_seconds) or arguments.valid_seconds < 0.0
    ):
        refuse(
            code="CM1-RENDER-REQUEST",
            gate="rust-io-render",
            reason="valid time must be finite and non-negative",
            remedy="omit --valid-seconds or pass a finite number of seconds",
            option="valid_seconds",
            value=arguments.valid_seconds,
        )

    requested_output = Path(arguments.output).expanduser()
    try:
        parent = requested_output.parent.resolve(strict=True)
    except OSError as error:
        refuse(
            code="CM1-RENDER-OUTPUT",
            gate="rust-io-render",
            reason=f"the output parent cannot be resolved exactly: {error}",
            remedy="create the intended output directory and pass a new .png path inside it",
            option="output",
            value=str(arguments.output),
        )
    output = parent / requested_output.name
    if not parent.is_dir() or output.suffix.lower() != ".png" or output.exists():
        refuse(
            code="CM1-RENDER-OUTPUT",
            gate="rust-io-render",
            reason="the output must be a new .png file in an existing directory",
            remedy="choose a non-existent .png path; existing outputs are never overwritten",
            option="output",
            value=str(output),
        )
    try:
        executable = verify_native_tools()["cm1-render"]
    except (OSError, NativeToolError) as error:
        refuse(
            code="CM1-NATIVE-TOOLS-INVALID",
            gate="rust-io-render",
            reason=f"the installed native renderer failed manifest verification: {error}",
            remedy="reinstall a platform wheel whose native-tools.json and executable hashes pass",
        )

    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".owned-render", dir=parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    command = [
        str(executable),
        "--input",
        str(history),
        "--format",
        history_format,
        "--contract-sha256",
        digest,
        "--output",
        str(temporary_path),
        "--product",
        product,
    ]
    for field in fields:
        command.extend(("--field", field))
    command.extend(
        (
            "--nx",
            str(arguments.nx),
            "--ny",
            str(arguments.ny),
            "--nz",
            str(arguments.nz),
            "--level",
            str(arguments.level),
            "--label",
            arguments.label,
        )
    )
    if arguments.record is not None:
        command.extend(("--record", str(arguments.record)))
    if arguments.valid_seconds is not None:
        command.extend(("--valid-seconds", repr(arguments.valid_seconds)))
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            reason = (
                completed.stderr.strip()
                or f"native renderer exited {completed.returncode}"
            )
            refuse(
                code="CM1-RUST-RENDER-FAILED",
                gate="rust-io-render",
                reason=reason,
                remedy="verify the envelope, independent contract hash, field names, dimensions, and level",
            )
        payload = temporary_path.read_bytes()
        if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            refuse(
                code="CM1-RUST-RENDER-FAILED",
                gate="rust-io-render",
                reason="the verified native renderer did not emit a valid PNG envelope",
                remedy="reinstall the admitted wheel and repeat the render gate",
            )
        try:
            os.link(temporary_path, output)
        except OSError as error:
            refuse(
                code="CM1-RENDER-OUTPUT",
                gate="rust-io-render",
                reason=f"the completed PNG could not be published without overwriting: {error}",
                remedy="choose a new output path on a filesystem that supports same-directory hard links",
                option="output",
                value=str(output),
            )
    finally:
        temporary_path.unlink(missing_ok=True)

    receipt = {
        "bytes": len(payload),
        "contract_sha256": digest,
        "contract_sha256_source": contract_source,
        "grid": dimensions,
        "fields": fields,
        "format": history_format,
        "input": str(history),
        "math_authority": "CM1 r21.1-derived mathematics",
        "model": "GM1",
        "native_abi": "gpuwm-cm1-render/v1",
        "output": str(output),
        "product": product,
        "schema": "gpuwm-cm1.render/v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "state_classification": (
            "initialization-not-tornado-not-forecast"
            if arguments.valid_seconds == 0.0
            else "diagnostic-fields-no-tornado-detection"
        ),
        "status": "rendered",
        "visualization": "bilinear cell-center reconstruction with 2x2 supersampling; visualization-only; source arrays unchanged",
    }
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"rendered {len(payload)} bytes to {output}")
    return 0


def records_door(arguments: argparse.Namespace) -> int:
    root = resources.files("cm1core.records")
    rows: list[dict[str, object]] = []
    for item in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not item.is_file() or item.name == "__init__.py":
            continue
        payload = item.read_bytes()
        rows.append(
            {
                "bytes": len(payload),
                "name": item.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    print(json.dumps({"records": rows}, indent=2, sort_keys=True))
    return 0


def _example_summary(deck: Path) -> dict[str, object]:
    """Describe one shipped deck without opening a device or a sounding."""

    text = deck.read_text(encoding="utf-8")
    summary = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        body = stripped.lstrip("#").strip()
        if body:
            summary = body
            break
    row: dict[str, object] = {"name": deck.stem, "file": deck.name, "summary": summary}
    try:
        config = load_config(deck)
    except Exception as error:  # a deck from another lane still lists
        row["loads"] = False
        row["load_error"] = str(error)
        return row
    row["loads"] = True
    row["lane"] = str(config.run.lane.value)
    if row["lane"] == "cartesian":
        from .cartesian_door import require_initializer_selection

        grid = config.grid
        if grid is not None:
            row["grid"] = f"{grid.nx}x{grid.ny}x{grid.nz}"
        try:
            selection, _ = require_initializer_selection(config)
        except Exception as error:
            row["needs_sounding"] = None
            row["load_error"] = str(error)
        else:
            row["needs_sounding"] = bool(selection.reads_input_sounding)
            if row["needs_sounding"]:
                row["sounding"] = _deck_sounding(text)
    row["quickstart_ready"] = bool(
        row.get("lane") == "cartesian"
        and row.get("needs_sounding") is False
        or (row.get("sounding") or {}).get("shipped") is not None
    )
    return row


def _shipped_soundings() -> dict[str, dict[str, object]]:
    """The observed input_sounding files this wheel carries, by SHA-256."""

    root = resources.files("cm1core.data") / "soundings"
    try:
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    shipped: dict[str, dict[str, object]] = {}
    for row in index.get("soundings", []):
        item = root / str(row["filename"])
        try:
            payload = item.read_bytes()
        except OSError:
            continue
        # A file whose bytes no longer match its recorded identity is not
        # offered: pairing a deck with it would run another atmosphere.
        if hashlib.sha256(payload).hexdigest() == row["sha256"]:
            shipped[str(row["sha256"])] = {**row, "resource": item}
    return shipped


def _deck_sounding(text: str) -> dict[str, object]:
    """Where a deck that reads input_sounding gets it.

    A deck built for one observed profile names its SHA-256 in its header
    comment (``#   sha256  <hex>``); when this wheel ships that file the deck
    is runnable as copied.  Every other sounding deck reads CM1's own
    run/config_files profile, which CM1-AUTHORITY.json's redistribution record
    does not let this package ship.
    """

    from .cartesian_door import CARTESIAN_SOUNDING_RESOURCE
    from .data import PackagedInputError, required_input_identity

    named = re.search(r"^#\s+sha256\s+([0-9a-f]{64})\s*$", text, flags=re.MULTILINE)
    if named is not None:
        digest = named.group(1)
        row = _shipped_soundings().get(digest)
        if row is not None:
            return {
                "shipped": str(row["filename"]),
                "sha256": digest,
                "observation": row["observation"],
                "how": "copied beside the deck by `gpuwm-cm1 examples NAME --copy DIR`; pass it with --sounding",
            }
        return {
            "shipped": None,
            "sha256": digest,
            "how": "needs sounding: the input_sounding file with this SHA-256 named in the deck header; pass it with --sounding",
        }
    try:
        identity = required_input_identity(CARTESIAN_SOUNDING_RESOURCE)
    except (PackagedInputError, OSError, ValueError):
        return {"shipped": None, "how": "needs sounding: a CM1 r21.1 input_sounding; pass it with --sounding"}
    return {
        "shipped": None,
        "sha256": identity["sha256"],
        "how": (
            f"needs sounding: CM1 r21.1 {identity['source_path']} "
            f"({identity['bytes']} bytes, sha256 {identity['sha256']}), not "
            "redistributed; download cm1r21.1.tar.gz from "
            "https://www2.mmm.ucar.edu/people/bryan/cm1/, extract it and pass "
            "that file with --sounding"
        ),
    }


def _cm1_namelists() -> dict[str, dict[str, object]]:
    """CM1's own namelists this wheel ships unchanged, by case name (run/config_files/<case>)."""

    from .data import packaged_input_path, packaged_inputs

    cases: dict[str, dict[str, object]] = {}
    for name, row in packaged_inputs().items():
        source = str(row.get("source_path") or "")
        match = re.fullmatch(r"run/config_files/([A-Za-z0-9_]+)/namelist\.input", source)
        if row.get("shipped") is not True or match is None:
            continue
        cases[match.group(1)] = {
            "case": match.group(1),
            "source": f"{row.get('source_release')} {source}, unchanged",
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "path": packaged_input_path(name),
        }
    return dict(sorted(cases.items()))


def _cm1_examples_door(arguments: argparse.Namespace) -> int:
    """List CM1's shipped namelists, or copy them as DIR/<case>/namelist.input."""

    cases = _cm1_namelists()
    wanted = list(arguments.names or [])
    unknown = [name for name in wanted if name not in cases]
    if unknown:
        raise ValueError(
            f"unknown CM1 example(s) {unknown}; `gpuwm-cm1 examples --cm1` lists {sorted(cases)}"
        )
    chosen = [cases[name] for name in (wanted or cases)]
    if arguments.copy is None:
        print(json.dumps({"cm1_namelists": [
            {**{key: value for key, value in row.items() if key != "path"},
             "import": f"gpuwm-cm1 examples --cm1 {row['case']} --copy DIR && "
                       f"gpuwm-cm1 import DIR/{row['case']}/namelist.input --out RUNDIR"}
            for row in chosen
        ]}, indent=2, sort_keys=True))
        return 0
    destination = Path(arguments.copy)
    clashes = [f"{row['case']}/namelist.input" for row in chosen
               if (destination / str(row["case"]) / "namelist.input").exists()]
    if clashes:
        raise FileExistsError(f"{destination} already holds {clashes}; examples never overwrite a file")
    written = []
    for row in chosen:
        target = destination / str(row["case"]) / "namelist.input"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row["path"]).read_bytes())
        written.append(str(target))
    print(json.dumps({"copied": written, "directory": str(destination),
                      "notice": "CM1's namelists, unchanged, under the CM1 release grant quoted in NOTICE"},
                     indent=2, sort_keys=True))
    return 0


def examples_door(arguments: argparse.Namespace) -> int:
    """List the shipped example decks, or copy them into a new directory."""

    if getattr(arguments, "cm1", False):
        return _cm1_examples_door(arguments)
    root = resources.files("cm1core") / "examples"
    decks = sorted(
        (Path(os.fspath(item)) for item in root.iterdir() if item.name.endswith(".toml")),
        key=lambda path: path.name,
    ) if root.is_dir() else []
    if not decks:
        raise OSError(
            "no example decks are installed with this distribution; install the "
            "wheel built from the sdist, or read examples/ in the source tree"
        )
    wanted = list(arguments.names or [])
    if wanted:
        by_name = {deck.stem: deck for deck in decks}
        unknown = [name for name in wanted if name not in by_name]
        if unknown:
            raise ValueError(
                f"unknown example(s) {unknown}; run `gpuwm-cm1 examples` for the list"
            )
        decks = [by_name[name] for name in wanted]
    if arguments.copy is None:
        print(json.dumps({"examples": [_example_summary(deck) for deck in decks]},
                         indent=2, sort_keys=True))
        return 0
    destination = Path(arguments.copy)
    destination.mkdir(parents=True, exist_ok=True)
    clashes = [deck.name for deck in decks if (destination / deck.name).exists()]
    if clashes:
        raise FileExistsError(
            f"{destination} already holds {clashes}; examples never overwrite a file"
        )
    soundings: dict[str, object] = {}
    for deck in decks:
        sounding = _example_summary(deck).get("sounding") or {}
        if sounding.get("shipped"):
            target = destination / f"{deck.stem}.input_sounding"
            if target.exists():
                clashes.append(target.name)
            else:
                soundings[deck.name] = sounding
    if clashes:
        raise FileExistsError(
            f"{destination} already holds {clashes}; examples never overwrite a file"
        )
    shipped = _shipped_soundings()
    copied_soundings: dict[str, str] = {}
    for deck in decks:
        (destination / deck.name).write_bytes(deck.read_bytes())
        sounding = soundings.get(deck.name)
        if sounding is not None:
            target = destination / f"{deck.stem}.input_sounding"
            target.write_bytes(shipped[str(sounding["sha256"])]["resource"].read_bytes())
            copied_soundings[deck.name] = target.name
    print(json.dumps({"copied": [deck.name for deck in decks],
                      "soundings": copied_soundings,
                      "directory": str(destination)}, indent=2, sort_keys=True))
    return 0


__all__ = [
    "check_door",
    "doctor_door",
    "examples_door",
    "preflight_door",
    "records_door",
    "render_door",
    "restart_door",
    "run_door",
    "version_door",
]
