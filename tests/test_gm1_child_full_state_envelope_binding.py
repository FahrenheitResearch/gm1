"""The composed child envelope admits any parent and binds what it names.

Two properties are proved here, and both are the difference between a nest
that is selectable and a nest that is sealed to one historical run:

  * an envelope composed from a parent other than the first one this project
    admitted is loadable, because it is admitted by the identity its own
    receipt records rather than by a frozen digest set; and
  * whatever the envelope was composed from is what the run must open.  The
    historical envelope was composed from the uniform-z closure this project's
    own evidence marks REFUSED_STRETCHED_Z_ORACLE; pairing it with the
    corrected stretched-z arena the child now uploads is refused by name.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import cm1core.cuda_gm1_child_large_step as envelope_module
import cm1core.cuda_gm1_child_step_composer as composer
from cm1core.cuda_gm1_child_large_step import (
    Gm1ChildFullStatePins,
    load_gm1_child_full_state_envelope,
)
from cm1core.gm1_child_stretched_closure_v3 import refuse_uniform_z_closure

# The closure the historical full-state envelope was composed from, exactly as
# evidence/validation/gm1-t300-child-full-state-v1/validation-receipt.json
# records it, and the corrected stretched-z closure the child now uploads,
# exactly as evidence/validation/gm1-t300-child-stretched-closure-v3/receipt.json
# records it.
REFUSED_V2 = {
    "manifest.json": "7aba693cd3a5c984c5a0d969bffa0793f80dfea138aa661da34a86f69982d9d9",
    "closure.f32le": "dc277233d7bead2009749e09f402fdff6c072323a57568fef270dbe6904dca6c",
    "receipt.json": "9972f6e981dc99e5635a844b70906328339535450bc57cdb01e61469b8080985",
}
CORRECTED_V3 = {
    "manifest.json": "6ff7d5a61d8e7597c46b33887311ada5e1c763f18e2a822f5d5329392eb23ee5",
    "closure.f32le": "cf8a3731006959a23445ea9af94405d1812235de18144329d0b830dd4c5a8281",
    "receipt.json": "8b471481978dd3a8cecbe7f324a0f7ad55c93ff6c982e4e5dd3694ccbd6945ed",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _f64_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def _artifact(role: str, directory: str, files: dict[str, tuple[int, str]]) -> dict:
    return {
        "abi": f"fixture.{role}-abi/v1",
        "directory": directory,
        "files": [
            {"name": name, "nbytes": nbytes, "sha256": sha}
            for name, (nbytes, sha) in sorted(files.items())
        ],
        "reference_rule": "sibling directory; rehash every file before mmap/upload",
        "role": role,
        "schema": f"fixture.{role}-schema/v1",
        "status": "PASS_FIXTURE",
    }


def _fixture_files(seed: str) -> dict[str, tuple[int, str]]:
    return {
        "manifest.json": (11, hashlib.sha256(f"{seed}-manifest".encode()).hexdigest()),
        "receipt.json": (13, hashlib.sha256(f"{seed}-receipt".encode()).hexdigest()),
    }


def _write_envelope(
    root: Path,
    *,
    nz: int = 2,
    ny: int = 3,
    nx: int = 4,
    closure_files: dict[str, str] | None = None,
) -> None:
    """Publish one small, self-consistent child full-state envelope.

    The grid is deliberately not the accepted 50 m child: the envelope contract
    is grid-generic, and the CUDA lane's own refusal is what names the geometry
    it is compiled for.
    """

    root.mkdir(parents=True)
    rng = np.random.default_rng(7)
    u0 = rng.standard_normal((nz, ny, nx + 1)).astype("<f4")
    v0 = rng.standard_normal((nz, ny + 1, nx)).astype("<f4")
    base_bytes = u0.tobytes(order="C") + v0.tobytes(order="C")
    (root / "base-wind.f32le").write_bytes(base_bytes)
    base_wind_sha256 = hashlib.sha256(base_bytes).hexdigest()

    closure = closure_files or {
        name: hashlib.sha256(f"closure-{name}".encode()).hexdigest()
        for name in ("manifest.json", "closure.f32le", "receipt.json")
    }
    manifest = {
        "artifacts": {
            "atmosphere": _artifact(
                "atmosphere",
                "fixture-child-atmosphere",
                {
                    **_fixture_files("atmosphere"),
                    "arena.f32le": (
                        17,
                        hashlib.sha256(b"atmosphere-arena").hexdigest(),
                    ),
                },
            ),
            "auxiliary": _artifact(
                "auxiliary", "fixture-child-auxiliary", _fixture_files("auxiliary")
            ),
            "closure": _artifact(
                "closure",
                "fixture-child-closure",
                {name: (19, sha) for name, sha in closure.items()},
            ),
            "static_heights": _artifact(
                "static_heights",
                "fixture-child-static",
                {
                    **_fixture_files("static"),
                    "zh.f32le": (29, hashlib.sha256(b"static-zh").hexdigest()),
                    "zf.f32le": (31, hashlib.sha256(b"static-zf").hexdigest()),
                },
            ),
            "terrain": _artifact(
                "terrain",
                "fixture-child-terrain",
                {
                    **_fixture_files("terrain"),
                    "terrain-msl.f32le": (
                        37,
                        hashlib.sha256(b"terrain-msl").hexdigest(),
                    ),
                },
            ),
        },
        "authority": {
            "cm1_release": "CM1 r21.1",
            "tarball_sha256": composer.SOURCE_ANCHORS["authority_tarball_sha256"],
        },
        "base_wind": {
            "abi": envelope_module.GM1_CHILD_BASE_WIND_ABI,
            "carriers": [
                {
                    "dtype": "little-endian-f32",
                    "name": "u0",
                    "nbytes": u0.nbytes,
                    "offset_nbytes": 0,
                    "order": "C",
                    "sha256": hashlib.sha256(u0.tobytes(order="C")).hexdigest(),
                    "shape": [nz, ny, nx + 1],
                },
                {
                    "dtype": "little-endian-f32",
                    "name": "v0",
                    "nbytes": v0.nbytes,
                    "offset_nbytes": u0.nbytes,
                    "order": "C",
                    "sha256": hashlib.sha256(v0.tobytes(order="C")).hexdigest(),
                    "shape": [nz, ny + 1, nx],
                },
            ],
            "file": "base-wind.f32le",
            "nbytes": len(base_bytes),
            "sha256": base_wind_sha256,
        },
        "configuration": {
            "domain_shape_zyx": [nz, ny, nx],
            "dx_m_f64_bits": _f64_bits(50.0),
            "dy_m_f64_bits": _f64_bits(50.0),
            "imove": 0,
            "target_dt_seconds_f64_bits": envelope_module.GM1_CHILD_DT_F64_BITS,
            "terrain": True,
        },
        "model": "GM1",
        "parent_provenance": {
            "synthetic_atmospheric_data": False,
            "synthetic_translation": False,
        },
        "producer": {
            "executable_sha256": hashlib.sha256(b"producer-exe").hexdigest(),
            "source_sha256": hashlib.sha256(b"producer-src").hexdigest(),
        },
        "schema": envelope_module.GM1_CHILD_FULL_STATE_MANIFEST_SCHEMA,
        "state_schema": envelope_module.GM1_CHILD_FULL_STATE_SCHEMA,
        "status": envelope_module.GM1_CHILD_FULL_STATE_STATUS,
    }
    manifest_sha256 = _digest(manifest)
    (root / "manifest.json").write_bytes(_canonical(manifest))

    first_step = {
        "carrier_dispositions": {
            "a_to_current_raw_bit_pairs": [
                ["ua", "u3d"],
                ["va", "v3d"],
                ["wa", "w3d"],
                ["ppi", "pp3d"],
                ["tha", "th3d"],
                ["qa", "q3d"],
            ],
            "component_persistence": [f"component.{index}" for index in range(11)],
            "held_positive_zero": [
                "held.qpten",
                "held.qtten",
                "held.qvten",
                "held.qcten",
            ],
        },
        "lifecycle": {
            "completed_large_step": False,
            "completed_large_steps": 0,
            "first_large_step_ready": True,
            "model_time_seconds_f64_bits": envelope_module.GM1_CHILD_ZERO_F64_BITS,
            "next_stage": 1,
            "open_boundary_histories_initialized": True,
            "step_index": 0,
            "target_dt_seconds_f64_bits": envelope_module.GM1_CHILD_DT_F64_BITS,
        },
        "manifest_sha256": manifest_sha256,
        "physical_base_views": [
            {
                "artifact": "auxiliary",
                "carrier": name,
                "file": "terrain-open.f32le",
                "physical_slice_exclusive_zyx": [1, nz + 1, 3, ny + 3, 3, nx + 3],
            }
            for name in ("pi0", "th0", "qv0")
        ],
        "schema": envelope_module.GM1_CHILD_FIRST_STEP_SCHEMA,
        "status": "READY_FOR_FIRST_COMPLETE_LARGE_STEP",
        "terrain_rayleigh_base_wind": {
            "abi": envelope_module.GM1_CHILD_BASE_WIND_ABI,
            "file": "base-wind.f32le",
            "u0_shape_zyx": [nz, ny, nx + 1],
            "v0_shape_zyx": [nz, ny + 1, nx],
        },
    }
    first_step_sha256 = _digest(first_step)
    (root / "first-step.json").write_bytes(_canonical(first_step))

    checkpoint = {
        "carrier_count": 37,
        "initialization_boundary_admitted_as_checkpoint": False,
        "manifest_sha256": manifest_sha256,
        "persistent_sources": [{"name": f"carrier.{index}"} for index in range(37)],
        "schema": envelope_module.GM1_CHILD_CHECKPOINT_ADMISSION_SCHEMA,
    }
    checkpoint_sha256 = _digest(checkpoint)
    (root / "checkpoint-admission.json").write_bytes(_canonical(checkpoint))

    receipt = {
        "base_wind_sha256": base_wind_sha256,
        "checkpoint_admission_sha256": checkpoint_sha256,
        "first_step_sha256": first_step_sha256,
        "manifest_sha256": manifest_sha256,
        "producer_executable_sha256": manifest["producer"]["executable_sha256"],
        "producer_source_sha256": manifest["producer"]["source_sha256"],
        "schema": envelope_module.GM1_CHILD_FULL_STATE_RECEIPT_SCHEMA,
        "status": envelope_module.GM1_CHILD_FULL_STATE_STATUS,
    }
    (root / "receipt.json").write_bytes(_canonical(receipt))


def test_an_envelope_from_another_parent_is_admitted_by_its_own_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    loaded = load_gm1_child_full_state_envelope(root)
    assert loaded.domain_shape_zyx == (2, 3, 4)
    assert loaded.u0.shape == (2, 3, 5)
    assert loaded.v0.shape == (2, 4, 4)
    # Nothing here is the first child this project composed.
    assert (
        loaded.pins.manifest_sha256
        != "02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d"
    )
    assert set(loaded.artifacts) == {
        "atmosphere",
        "auxiliary",
        "closure",
        "static_heights",
        "terrain",
    }
    assert loaded.artifact("closure").directory == "fixture-child-closure"


def test_the_historical_pins_still_replay_one_exact_envelope(tmp_path: Path) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    loaded = load_gm1_child_full_state_envelope(root)
    replayed = load_gm1_child_full_state_envelope(root, pins=loaded.pins)
    assert replayed.pins == loaded.pins

    wrong = Gm1ChildFullStatePins(
        manifest_sha256="0" * 64,
        first_step_sha256=loaded.pins.first_step_sha256,
        checkpoint_admission_sha256=loaded.pins.checkpoint_admission_sha256,
        receipt_sha256=loaded.pins.receipt_sha256,
        base_wind_sha256=loaded.pins.base_wind_sha256,
        producer_source_sha256=loaded.pins.producer_source_sha256,
        producer_executable_sha256=loaded.pins.producer_executable_sha256,
    )
    with pytest.raises(Exception) as error:
        load_gm1_child_full_state_envelope(root, pins=wrong)
    assert "GM1-CHILD-FULL-STATE-DRIFT" in str(error.value)


def test_the_cuda_lane_refuses_a_child_grid_it_is_not_compiled_for(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    loaded = load_gm1_child_full_state_envelope(root)
    with pytest.raises(ValueError) as error:
        composer._require_cuda_lane_shape(loaded)
    assert "the CUDA child lane is compiled for [75, 300, 300]" in str(error.value)


def test_the_envelopes_closure_must_be_the_closure_the_run_opens(
    tmp_path: Path,
) -> None:
    """Defect: a v2-composed envelope beside a v3 closure upload.

    The historical envelope textually records the discredited uniform-z
    closure.  The child now uploads the corrected stretched-z arena.  Pairing
    them would publish a receipt stating the child was initialized from the
    closure this project refuses, with the substitution invisible.
    """

    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root, closure_files=REFUSED_V2)
    loaded = load_gm1_child_full_state_envelope(root)
    reference = loaded.artifact("closure")
    assert reference.digest("closure.f32le") == REFUSED_V2["closure.f32le"]

    with pytest.raises(ValueError) as error:
        composer._require_artifact_files(dict(CORRECTED_V3), reference)
    message = str(error.value)
    assert "the substitution would be invisible in the receipt" in message
    assert REFUSED_V2["manifest.json"] in message

    # And the envelope's own recorded closure is refused by name, so a v2
    # envelope cannot be composed by simply also supplying the v2 arena.
    with pytest.raises(ValueError) as refusal:
        refuse_uniform_z_closure(
            manifest_sha256=reference.digest("manifest.json"),
            payload_sha256=reference.digest("closure.f32le"),
            receipt_sha256=reference.digest("receipt.json"),
            carrier_sha256={},
        )
    assert "superseded uniform-z child closure is refused" in str(refusal.value)


def test_a_corrected_envelope_and_a_corrected_closure_agree(tmp_path: Path) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root, closure_files=CORRECTED_V3)
    reference = load_gm1_child_full_state_envelope(root).artifact("closure")
    composer._require_artifact_files(dict(CORRECTED_V3), reference)
    refuse_uniform_z_closure(
        manifest_sha256=reference.digest("manifest.json"),
        payload_sha256=reference.digest("closure.f32le"),
        receipt_sha256=reference.digest("receipt.json"),
        carrier_sha256={},
    )


def test_a_sibling_outside_the_publication_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    loaded = load_gm1_child_full_state_envelope(root)
    with pytest.raises(ValueError) as error:
        composer._sibling(root.parent, loaded.artifact("terrain"))
    assert "not a regular sibling directory" in str(error.value)


def test_an_artifact_directory_that_escapes_the_publication_is_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["artifacts"]["terrain"]["directory"] = "../elsewhere"
    (root / "manifest.json").write_bytes(_canonical(manifest))
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["manifest_sha256"] = _digest(manifest)
    (root / "receipt.json").write_bytes(_canonical(receipt))
    first_step = json.loads((root / "first-step.json").read_bytes())
    first_step["manifest_sha256"] = receipt["manifest_sha256"]
    (root / "first-step.json").write_bytes(_canonical(first_step))
    receipt["first_step_sha256"] = _digest(first_step)
    checkpoint = json.loads((root / "checkpoint-admission.json").read_bytes())
    checkpoint["manifest_sha256"] = receipt["manifest_sha256"]
    (root / "checkpoint-admission.json").write_bytes(_canonical(checkpoint))
    receipt["checkpoint_admission_sha256"] = _digest(checkpoint)
    (root / "receipt.json").write_bytes(_canonical(receipt))

    with pytest.raises(ValueError) as error:
        load_gm1_child_full_state_envelope(root)
    assert "not one plain sibling directory name" in str(error.value)


class _Auxiliary:
    """Only the part of a loaded estate the input binding reads."""

    def __init__(self, static_inputs: dict[str, str]) -> None:
        self.manifest = {"static_inputs": static_inputs}


def test_the_estate_must_be_built_over_the_terrain_and_heights_admitted(
    tmp_path: Path,
) -> None:
    """The binding the retired combined_23_sha256 key stood in for.

    The auxiliary estate records the terrain it stands on and the static
    heights it was reconstructed over.  Composing an estate built over one
    surface with the vertical metrics of another gives the child a base state
    that never balanced on the terrain it is standing on.
    """

    root = tmp_path / "work" / "child-full-state"
    _write_envelope(root)
    loaded = load_gm1_child_full_state_envelope(root)
    terrain = loaded.artifact("terrain")
    static = loaded.artifact("static_heights")

    agreeing = _Auxiliary(
        {
            "zh_sha256": static.digest("zh.f32le"),
            "zf_sha256": static.digest("zf.f32le"),
            "terrain_manifest_sha256": terrain.digest("manifest.json"),
            "terrain_payload_sha256": terrain.digest("terrain-msl.f32le"),
            "terrain_receipt_sha256": terrain.digest("receipt.json"),
        }
    )
    composer._require_auxiliary_inputs(agreeing, terrain, static)

    for key, what in (
        ("zh_sha256", "mass heights"),
        ("zf_sha256", "face heights"),
        ("terrain_payload_sha256", "terrain surface"),
    ):
        wrong = _Auxiliary(dict(agreeing.manifest["static_inputs"]))
        wrong.manifest["static_inputs"][key] = "5" * 64
        with pytest.raises(ValueError) as error:
            composer._require_auxiliary_inputs(wrong, terrain, static)
        message = str(error.value)
        assert what in message and "never balanced against" in message
