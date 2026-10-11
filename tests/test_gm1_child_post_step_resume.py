from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from cm1core import gm1_child_post_step_resume as resume


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metadata_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pair = tmp_path / "pair"
    restore = tmp_path / "restore"
    execution = tmp_path / "execution.json"
    pair.mkdir()
    restore.mkdir()

    _write_json(
        pair / "manifest.json",
        {
            "schema": "gm1.real-50m-child-post-step-pair-manifest/v1",
            "status": "PASS_RUST_PAIRED_POST_STEP_ENVELOPE",
            "step": 1,
            "operator_manifest_sha256": resume.STEP1_OPERATOR_MANIFEST_SHA256,
            "mass1_f64_bits": resume.STEP1_MASS1_F64_BITS,
            "configuration": {
                "domain_shape_zyx": [75, 300, 300],
                "dt_f64_bits": "3fe0000000000000",
                "dx_m_f64_bits": "4049000000000000",
                "dy_m_f64_bits": "4049000000000000",
                "ztop_m_f64_bits": "40cd4c0000000000",
            },
            "checkpoint": {"contract_sha256": resume.STEP1_CHECKPOINT_CONTRACT_SHA256},
            "history": {"contract_sha256": resume.STEP1_HISTORY_CONTRACT_SHA256},
        },
    )
    _write_json(
        pair / "receipt.json",
        {
            "schema": "gm1.real-50m-child-post-step-pair-receipt/v1",
            "step": 1,
            "manifest_sha256": resume.STEP1_PAIR_PINS["manifest.json"],
            "operator_manifest_sha256": resume.STEP1_OPERATOR_MANIFEST_SHA256,
        },
    )
    rows = []
    offset = 0
    for name in resume.CHECKPOINT_NAMES:
        nbytes = math.prod(resume.CHECKPOINT_SHAPES[name]) * 4
        rows.append(
            {
                "name": name,
                "nbytes": nbytes,
                "offset_nbytes": offset,
                "sha256": "a" * 64,
            }
        )
        offset += nbytes
    _write_json(
        restore / "manifest.json",
        {
            "schema": "gm1.real-50m-child-post-step-restore/v1",
            "step": 1,
            "source_pair_manifest_sha256": resume.STEP1_PAIR_PINS["manifest.json"],
            "checkpoint": {
                "file": "checkpoint-arena.f32le",
                "dtype": "little-endian-f32",
                "order": "C",
                "nbytes": resume.CHECKPOINT_NBYTES,
                "sha256": resume.STEP1_RESTORE_PINS["checkpoint-arena.f32le"],
                "carriers": rows,
            },
        },
    )
    _write_json(
        restore / "receipt.json",
        {
            "schema": "gm1.real-50m-child-post-step-restore-receipt/v1",
            "status": "PASS_RAW_BIT_RESTORE",
            "step": 1,
        },
    )
    tke_commit = {"schema": "gm1.test-tke-commit/v1", "completed_nrk": 3}
    _write_json(
        execution,
        {
            "status": "PASS_REAL_300X300X75_CHILD_ONE_COUPLED_LARGE_STEP",
            "gates": {
                "all_finite": True,
                "moisture_nonnegative": True,
                "qa_q3d_raw_bit_equal": True,
                "current_a_raw_bit_equal": True,
            },
            "publication": {
                "publish_receipt": {
                    "manifest_sha256": resume.STEP1_PAIR_PINS["manifest.json"],
                    "receipt_sha256": resume.STEP1_PAIR_PINS["receipt.json"],
                }
            },
            "operators": {
                "finish": {
                    "post_final": {
                        "solve3": {"tke_commit": tke_commit},
                        "closure_refresh": {
                            "tke_commit_receipt_sha256": (
                                resume._canonical_receipt_token(tke_commit)
                            )
                        },
                    }
                }
            },
        },
    )
    (restore / "checkpoint-arena.f32le").write_bytes(b"x")
    return pair, restore, execution


def test_checkpoint_layout_matches_frozen_rust_abi() -> None:
    assert len(resume.CHECKPOINT_NAMES) == 37
    assert resume.CHECKPOINT_NBYTES == 976_696_992
    assert resume.CHECKPOINT_SHAPES["q3d"] == (6, 75, 300, 300)
    assert resume.CHECKPOINT_SHAPES["closure.tkea"] == (76, 300, 300)
    assert resume.CHECKPOINT_SHAPES["radbcw"] == (77, 306)


def test_restore_destinations_do_not_require_one_completed_bridge_call() -> None:
    objects = {name: object() for name in resume.CHECKPOINT_NAMES}

    class Product:
        def field(self, name: str) -> object:
            return objects[f"held.{name}" if f"held.{name}" in objects else name]

    state = SimpleNamespace(
        closure=SimpleNamespace(sgs={"tkea": objects["closure.tkea"]}),
        prepared=SimpleNamespace(
            arrays={
                name: objects[name]
                for name in resume.CHECKPOINT_NAMES
                if name.startswith("radbc")
            }
        ),
        product=Product(),
        transition=SimpleNamespace(
            bridge=SimpleNamespace(
                restart_arrays=lambda: (_ for _ in ()).throw(
                    AssertionError("fresh bridge restart arrays must not be queried")
                )
            )
        ),
    )

    destinations = resume._destinations(state)

    assert tuple(destinations) == resume.CHECKPOINT_NAMES
    assert all(destinations[name] is objects[name] for name in resume.CHECKPOINT_NAMES)


@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_transition_completed_call_counter_refuses_noncanonical_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="strict nonnegative integer"):
        resume._create_resumed_transition(
            object(),
            completed_calls=value,
            cupy=object(),  # type: ignore[arg-type]
        )


def test_metadata_loader_binds_step_one_cadence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair, restore, execution = _metadata_fixture(tmp_path)
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )
    monkeypatch.setattr(
        resume,
        "_sha256_file",
        lambda path: (
            resume.STEP1_EXECUTION_RECEIPT_SHA256
            if Path(path) == execution
            else "0" * 64
        ),
    )

    loaded = resume.load_gm1_child_post_step_resume(pair, restore, execution)

    assert loaded.source_step == 1
    assert loaded.source_model_time_s == 0.5
    assert loaded.mass1_f64 == pytest.approx(7_302_716.4093339145)
    assert (loaded.nx, loaded.ny, loaded.nz) == (300, 300, 75)
    assert (loaded.dx_m, loaded.dy_m, loaded.dt_s, loaded.ztop_m) == (
        50.0,
        50.0,
        0.5,
        15_000.0,
    )
    assert tuple(loaded.carrier_specs) == resume.CHECKPOINT_NAMES


def test_metadata_loader_refuses_wrong_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair, restore, execution = _metadata_fixture(tmp_path)
    manifest_path = pair / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["step"] = 2
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )
    monkeypatch.setattr(
        resume,
        "_sha256_file",
        lambda path: (
            resume.STEP1_EXECUTION_RECEIPT_SHA256
            if Path(path) == execution
            else "0" * 64
        ),
    )

    with pytest.raises(ValueError, match="pair manifest semantic identity"):
        resume.load_gm1_child_post_step_resume(pair, restore, execution)


def test_step_two_loader_binds_persisted_tke_without_stale_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair, restore, _execution = _metadata_fixture(tmp_path)
    pair_manifest = json.loads((pair / "manifest.json").read_text(encoding="utf-8"))
    pair_manifest.update(
        {
            "step": 2,
            "operator_manifest_sha256": resume.STEP2_OPERATOR_MANIFEST_SHA256,
            "mass1_f64_bits": resume.STEP2_SOURCE_PINS.mass1_f64_bits,
        }
    )
    pair_manifest["checkpoint"]["contract_sha256"] = (
        resume.STEP2_CHECKPOINT_CONTRACT_SHA256
    )
    pair_manifest["history"]["contract_sha256"] = (
        resume.STEP2_HISTORY_CONTRACT_SHA256
    )
    _write_json(pair / "manifest.json", pair_manifest)
    _write_json(
        pair / "receipt.json",
        {
            "schema": "gm1.real-50m-child-post-step-pair-receipt/v1",
            "step": 2,
            "manifest_sha256": resume.STEP2_PAIR_PINS["manifest.json"],
            "operator_manifest_sha256": resume.STEP2_OPERATOR_MANIFEST_SHA256,
        },
    )
    restore_manifest = json.loads(
        (restore / "manifest.json").read_text(encoding="utf-8")
    )
    restore_manifest.update(
        {
            "step": 2,
            "source_pair_manifest_sha256": resume.STEP2_PAIR_PINS["manifest.json"],
        }
    )
    restore_manifest["checkpoint"]["sha256"] = resume.STEP2_RESTORE_PINS[
        "checkpoint-arena.f32le"
    ]
    _write_json(restore / "manifest.json", restore_manifest)
    _write_json(
        restore / "receipt.json",
        {
            "schema": "gm1.real-50m-child-post-step-restore-receipt/v1",
            "status": "PASS_RAW_BIT_RESTORE",
            "step": 2,
        },
    )
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )

    loaded = resume.load_gm1_child_post_step_resume(
        pair,
        restore,
        source_pins=resume.STEP2_SOURCE_PINS,
    )

    assert loaded.source_step == 2
    assert loaded.source_model_time_s == 1.0
    assert loaded.execution_receipt_path is None
    assert (
        loaded.tke_commit_receipt["schema"]
        == "gm1.cuda-child-persisted-tke-restart-admission/v1"
    )
    assert loaded.tke_commit_receipt["checkpoint_carrier_sha256"] == "a" * 64
    assert loaded.tke_commit_receipt["recovery_commit"] == resume.STEP2_RECOVERY_COMMIT


def test_step_two_loader_refuses_unrelated_execution_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )
    with pytest.raises(ValueError, match="refuses an unrelated execution receipt"):
        resume.load_gm1_child_post_step_resume(
            tmp_path,
            tmp_path,
            tmp_path / "execution.json",
            source_pins=resume.STEP2_SOURCE_PINS,
        )


def _tilestream_publication_admission() -> dict[str, object]:
    history_names = (
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
    )
    return {
        "checkpoint_carrier_sha256": {
            name: "a" * 64 for name in resume.CHECKPOINT_NAMES
        },
        "claim_limit": (
            resume.GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_CLAIM_LIMIT
        ),
        "completed_rk3_operation_token_sha256": "b" * 64,
        "final_step_operation_token_sha256": "c" * 64,
        "gates": {
            "all_finite": True,
            "checkpoint_history_overlap_same_object": True,
            "current_a_raw_bit_equal": True,
            "moisture_nonnegative": True,
            "qa_q3d_raw_bit_equal": True,
        },
        "history_carrier_sha256": {
            name: ("a" * 64 if name in resume.CHECKPOINT_NAMES else "d" * 64)
            for name in history_names
        },
        "model": "GM1",
        "model_time_f64_bits": resume.STEP1_MODEL_TIME_F64_BITS,
        "operator_manifest_sha256": resume.STEP1_OPERATOR_MANIFEST_SHA256,
        "producer_git_commit": "e" * 40,
        "publication": {
            "checkpoint_artifact_sha256": resume.STEP1_PAIR_PINS[
                "checkpoint.cm1fcp"
            ],
            "checkpoint_contract_sha256": (
                resume.STEP1_CHECKPOINT_CONTRACT_SHA256
            ),
            "checkpoint_raw_sha256": resume.STEP1_RESTORE_PINS[
                "checkpoint-arena.f32le"
            ],
            "history_artifact_sha256": resume.STEP1_PAIR_PINS[
                "history.cm1fhs"
            ],
            "history_contract_sha256": resume.STEP1_HISTORY_CONTRACT_SHA256,
            "history_raw_sha256": resume.STEP1_RESTORE_PINS[
                "history-arena.f32le"
            ],
            "manifest_sha256": resume.STEP1_PAIR_PINS["manifest.json"],
            "receipt_sha256": resume.STEP1_PAIR_PINS["receipt.json"],
        },
        "runner_sha256": "f" * 64,
        "schema": resume.GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_SCHEMA,
        "stage_identity_sha256": "1" * 64,
        "status": resume.GM1_TILESTREAM_POST_RK3_PUBLICATION_ADMISSION_STATUS,
        "step": 1,
        "stream_pointer": 73,
    }


def test_loader_accepts_exact_successful_tilestream_publication_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair, restore, execution = _metadata_fixture(tmp_path)
    execution.write_text(
        json.dumps(_tilestream_publication_admission(), sort_keys=True),
        encoding="utf-8",
    )
    execution_sha = "2" * 64
    pins = resume.Gm1ChildPostStepSourcePins(
        step=1,
        pair_pins=resume.STEP1_PAIR_PINS,
        restore_pins=resume.STEP1_RESTORE_PINS,
        operator_manifest_sha256=resume.STEP1_OPERATOR_MANIFEST_SHA256,
        checkpoint_contract_sha256=resume.STEP1_CHECKPOINT_CONTRACT_SHA256,
        history_contract_sha256=resume.STEP1_HISTORY_CONTRACT_SHA256,
        mass1_f64_bits=resume.STEP1_MASS1_F64_BITS,
        model_time_f64_bits=resume.STEP1_MODEL_TIME_F64_BITS,
        execution_receipt_sha256=execution_sha,
    )
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )
    monkeypatch.setattr(
        resume,
        "_sha256_file",
        lambda path: execution_sha if Path(path) == execution else "0" * 64,
    )

    loaded = resume.load_gm1_child_post_step_resume(
        pair,
        restore,
        execution,
        source_pins=pins,
    )

    assert loaded.execution_receipt_path == execution
    assert loaded.tke_commit_receipt["checkpoint_carrier_sha256"] == "a" * 64
    assert (
        loaded.tke_commit_receipt["completed_rk3_operation_token_sha256"]
        == "b" * 64
    )
    assert loaded.tke_commit_receipt["execution_receipt_sha256"] == execution_sha
    assert "successful exact completed-RK3" in loaded.tke_commit_receipt[
        "disposition"
    ]


def test_loader_refuses_wrong_tilestream_publication_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair, restore, execution = _metadata_fixture(tmp_path)
    admission = _tilestream_publication_admission()
    admission["checkpoint_carrier_sha256"]["closure.tkea"] = "3" * 64
    execution.write_text(json.dumps(admission, sort_keys=True), encoding="utf-8")
    execution_sha = "2" * 64
    pins = resume.Gm1ChildPostStepSourcePins(
        step=1,
        pair_pins=resume.STEP1_PAIR_PINS,
        restore_pins=resume.STEP1_RESTORE_PINS,
        operator_manifest_sha256=resume.STEP1_OPERATOR_MANIFEST_SHA256,
        checkpoint_contract_sha256=resume.STEP1_CHECKPOINT_CONTRACT_SHA256,
        history_contract_sha256=resume.STEP1_HISTORY_CONTRACT_SHA256,
        mass1_f64_bits=resume.STEP1_MASS1_F64_BITS,
        model_time_f64_bits=resume.STEP1_MODEL_TIME_F64_BITS,
        execution_receipt_sha256=execution_sha,
    )
    monkeypatch.setattr(
        resume,
        "_pinned_inventory",
        lambda directory, _pins, _label: Path(directory),
    )
    monkeypatch.setattr(
        resume,
        "_sha256_file",
        lambda path: execution_sha if Path(path) == execution else "0" * 64,
    )

    with pytest.raises(ValueError, match="carrier hash changed for closure.tkea"):
        resume.load_gm1_child_post_step_resume(
            pair,
            restore,
            execution,
            source_pins=pins,
        )
