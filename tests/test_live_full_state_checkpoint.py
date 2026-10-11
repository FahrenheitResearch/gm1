from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from cm1core import live_full_state_checkpoint as live


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _small_layout() -> live.LiveFullStateLayout:
    carriers = tuple(
        live.LiveFullStateCarrier(name, (1,), index * 4, 4)
        for index, name in enumerate(live.LIVE_FULL_STATE_ARRAY_NAMES)
    )
    return live.LiveFullStateLayout(carriers, len(carriers) * 4, "bb" * 32)


def _views() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(live.LIVE_FULL_STATE_ARRAY_NAMES):
        # All words are finite; include signed zero and a subnormal to prove the
        # path is bytes, not a numeric round trip.
        bits = 0x3F000000 + index
        if name == "ppten":
            bits = 0x80000000
        elif name == "thten":
            bits = 0x00000001
        result[name] = np.array([bits], dtype=np.uint32).view(np.float32)
    return result


def test_import_is_cuda_and_numpy_cold() -> None:
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import cm1core.live_full_state_checkpoint; "
        "assert 'numpy' not in sys.modules; "
        "assert not any(n == 'cupy' or n.startswith('cupy.') for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=True,
    )


def test_capture_transports_every_raw_word_in_rust_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _small_layout()
    views = _views()
    binary = tmp_path / "cm1-live-full-state.exe"
    binary.write_bytes(b"stub")
    output = tmp_path / "state.cm1fcp"
    observed: dict[str, bytes] = {}

    monkeypatch.setattr(live, "load_live_full_state_layout", lambda _: layout)

    def fake_run(_: Path, arguments: list[str]) -> str:
        assert arguments[0] == "build"
        raw = Path(arguments[1]).read_bytes()
        extension = Path(arguments[-1]).read_bytes()
        observed["raw"] = raw
        observed["extension"] = extension
        assert hashlib.sha256(raw).hexdigest() == arguments[3]
        output.write_bytes(b"CM1FCP1-stub")
        receipt = {
            "artifact_nbytes": output.stat().st_size,
            "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "carrier_abi": live.LIVE_FULL_STATE_CARRIER_ABI,
            "carrier_count": len(live.LIVE_FULL_STATE_ARRAY_NAMES),
            "capture_phase": live.LIVE_FULL_STATE_CAPTURE_PHASE,
            "claim_limit": "test",
            "completed_wsm6_transitions": 4,
            "contract_sha256": "bb" * 32,
            "payload_sha256": "cc" * 32,
            "publication_contract": "test",
            "raw_arena_nbytes": layout.payload_nbytes,
            "raw_arena_sha256": hashlib.sha256(raw).hexdigest(),
            "schema": live.LIVE_FULL_STATE_CHECKPOINT_RECEIPT_SCHEMA,
            "source": {},
            "step": 4,
        }
        return _canonical(receipt).decode("ascii")

    monkeypatch.setattr(live, "_run_rust", fake_run)
    checkpoint = live.capture_live_full_state_checkpoint_new(
        views,
        output,
        rust_binary=binary,
        initialized_product_restart_sha256="11" * 32,
        initialized_product_manifest_sha256="22" * 32,
        initialized_product_payload_sha256="33" * 32,
        terrain_open_restart_sha256="44" * 32,
        wsm6_transition_restart_sha256="55" * 32,
        step=4,
        model_time_s=2.0,
        completed_wsm6_transitions=4,
        scheduler_entries={"coupled.scheduler.epoch": 4},
        dt_s=live.LIVE_FULL_STATE_REAL_DT_S,
    )
    expected = b"".join(
        memoryview(views[name]).cast("B") for name in live.LIVE_FULL_STATE_ARRAY_NAMES
    )
    assert observed["raw"] == expected
    assert observed["extension"] == _canonical(
        {
            "entries": {"coupled.scheduler.epoch": 4},
            "schema": live.LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA,
        }
    )
    assert checkpoint.raw_arena_sha256 == hashlib.sha256(expected).hexdigest()
    assert not any(
        path.name.startswith(".cm1-live-full-state-") for path in tmp_path.iterdir()
    )


def test_view_and_scheduler_validation_refuse_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _small_layout()
    views = _views()
    views.pop("pp3d")
    with pytest.raises(ValueError, match="inventory"):
        live._validate_views(views, layout)
    with pytest.raises(ValueError, match="collides"):
        live._scheduler_extension_bytes({"nrk": 2})
    with pytest.raises(ValueError, match="lower-case"):
        live._scheduler_extension_bytes({"Bad Key": 2})


def _write_restore_directory(
    root: Path, *, dt_s: float = live.LIVE_FULL_STATE_REAL_DT_S
) -> tuple[str, str, str, str, bytes]:
    root.mkdir()
    layout = _small_layout()
    views = _views()
    payload = b"".join(
        memoryview(views[name]).cast("B") for name in live.LIVE_FULL_STATE_ARRAY_NAMES
    )
    payload_sha = hashlib.sha256(payload).hexdigest()
    (root / live.LIVE_FULL_STATE_ARENA_FILE).write_bytes(payload)
    carriers = []
    for carrier in layout.carriers:
        raw = payload[carrier.offset_nbytes : carrier.offset_nbytes + carrier.nbytes]
        carriers.append(
            {
                "dtype": "<f4",
                "itemsize": 4,
                "name": carrier.name,
                "nbytes": carrier.nbytes,
                "offset_nbytes": carrier.offset_nbytes,
                "order": "C",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(carrier.shape),
            }
        )
    artifact_sha = "aa" * 32
    contract_sha = "bb" * 32
    manifest = {
        "capture_phase": live.LIVE_FULL_STATE_CAPTURE_PHASE,
        "carrier_abi": live.LIVE_FULL_STATE_CARRIER_ABI,
        "carriers": carriers,
        "claim_limit": "test",
        "dtype": "<f4",
        "itemsize": 4,
        "order": "C",
        "payload": {
            "file": live.LIVE_FULL_STATE_ARENA_FILE,
            "nbytes": len(payload),
            "sha256": payload_sha,
        },
        "restore_phase": live.LIVE_FULL_STATE_RESTORE_PHASE,
        "scheduler_extension": {
            "entries": {"coupled.scheduler.epoch": 4},
            "schema": live.LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA,
        },
        "scheduler_scalars": {
            "acoustic_substep": 0,
            "completed_large_steps": 4,
            "completed_wsm6_transitions": 4,
            "large_step_boundary": True,
            "nrk": 0,
            "radiation_history_initialized": True,
            "scalar_stage": 0,
            "wsm6_transition_ready": True,
        },
        "schema": live.LIVE_FULL_STATE_RESTORE_MANIFEST_SCHEMA,
        "source": {
            "initialized_product_manifest_sha256": "22" * 32,
            "initialized_product_payload_sha256": "33" * 32,
            "initialized_product_restart_sha256": "11" * 32,
            "terrain_open_restart_sha256": "44" * 32,
            "wsm6_transition_restart_sha256": "55" * 32,
        },
        "source_checkpoint": {
            "artifact_sha256": artifact_sha,
            "contract_sha256": contract_sha,
            "dt_f64_bits": struct.pack(">d", float(dt_s)).hex(),
            "model_time_f64_bits": "4000000000000000",
            "step": 4,
        },
    }
    manifest_raw = _canonical(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    (root / live.LIVE_FULL_STATE_MANIFEST_FILE).write_bytes(manifest_raw)
    receipt = {
        "carrier_abi": live.LIVE_FULL_STATE_CARRIER_ABI,
        "carrier_count": len(layout.carriers),
        "checkpoint_artifact_sha256": artifact_sha,
        "checkpoint_contract_sha256": contract_sha,
        "claim_limit": "test",
        "manifest_nbytes": len(manifest_raw),
        "manifest_sha256": manifest_sha,
        "payload_nbytes": len(payload),
        "payload_sha256": payload_sha,
        "publication_contract": "test",
        "schema": live.LIVE_FULL_STATE_RESTORE_RECEIPT_SCHEMA,
        "step": 4,
    }
    (root / live.LIVE_FULL_STATE_RECEIPT_FILE).write_bytes(_canonical(receipt))
    return manifest_sha, payload_sha, artifact_sha, contract_sha, payload


def test_host_restore_views_and_copy_preserve_raw_bits(tmp_path: Path) -> None:
    restore_dir = tmp_path / "restore"
    manifest_sha, payload_sha, artifact_sha, contract_sha, payload = (
        _write_restore_directory(restore_dir)
    )
    restored = live.load_restored_live_full_state(
        restore_dir,
        expected_manifest_sha256=manifest_sha,
        expected_payload_sha256=payload_sha,
        expected_checkpoint_artifact_sha256=artifact_sha,
        expected_checkpoint_contract_sha256=contract_sha,
        resume_dt_s=live.LIVE_FULL_STATE_REAL_DT_S,
    )
    assert bytes(restored.arena) == payload
    assert restored.array("ppten").view(np.uint32)[0] == 0x80000000
    assert restored.array("thten").view(np.uint32)[0] == 0x00000001
    destinations = {
        carrier.name: np.empty(carrier.shape, dtype=np.float32)
        for carrier in restored.carriers
    }
    assert live.copy_restored_live_full_state_into(restored, destinations) is None
    copied = b"".join(
        memoryview(destinations[name]).cast("B")
        for name in live.LIVE_FULL_STATE_ARRAY_NAMES
    )
    assert copied == payload


def test_compose_accepts_transition_restart_without_copying() -> None:
    views = _views()

    class Restart:
        arrays = MappingProxyType(
            {name: views[name] for name in live.WSM6_TRANSITION_ARRAY_NAMES}
        )

    class Transition:
        def capture_restart(self) -> Restart:
            return Restart()

    terrain = {name: views[name] for name in live.TERRAIN_OPEN_CARRIER_NAMES}
    scalar = {name: views[name] for name in live.SCALAR_RESTART_ARRAY_NAMES}
    closure = {name: views[name] for name in live.CLOSURE_RESTART_ARRAY_NAMES}
    composed = live.compose_live_full_state_views(
        terrain, scalar, Transition(), closure
    )
    assert tuple(sorted(composed)) == live.LIVE_FULL_STATE_ARRAY_NAMES
    assert composed["state.q3d"] is views["state.q3d"]


def test_a_checkpoint_captured_at_a_fine_step_restores_at_that_step(
    tmp_path: Path,
) -> None:
    """The restore compares the artifact's own step, not the half second.

    ``capture_live_full_state_checkpoint_new`` admits any step that is a
    positive normal binary32 word, so a tens-of-metres run captured at 0.05 s
    is a legitimate artifact.  Neither ``resume_dt_s`` nor the capture's
    ``dt_s`` has a default, so no caller can omit it and silently be compared
    against ``LIVE_FULL_STATE_REAL_DT_S``: the fine step restores, and only a
    genuine mismatch is refused.
    """

    fine = 0.05
    restore_dir = tmp_path / "restore"
    manifest_sha, payload_sha, artifact_sha, contract_sha, payload = (
        _write_restore_directory(restore_dir, dt_s=fine)
    )
    common = dict(
        expected_manifest_sha256=manifest_sha,
        expected_payload_sha256=payload_sha,
        expected_checkpoint_artifact_sha256=artifact_sha,
        expected_checkpoint_contract_sha256=contract_sha,
    )
    restored = live.load_restored_live_full_state(
        restore_dir, resume_dt_s=fine, **common
    )
    assert bytes(restored.arena) == payload

    with pytest.raises(ValueError) as caught:
        live.load_restored_live_full_state(
            restore_dir, resume_dt_s=live.LIVE_FULL_STATE_REAL_DT_S, **common
        )
    assert "written at dt=0.05" in str(caught.value)


def test_neither_checkpoint_step_may_be_defaulted() -> None:
    """An omitted step is a TypeError, never a silent half second."""

    import inspect

    for function, name in (
        (live.capture_live_full_state_checkpoint_new, "dt_s"),
        (live.restore_live_full_state_checkpoint_new, "resume_dt_s"),
        (live.load_restored_live_full_state, "resume_dt_s"),
    ):
        parameter = inspect.signature(function).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function
        assert parameter.default is inspect.Parameter.empty, function
