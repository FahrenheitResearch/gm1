from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core.gm1_child_atmosphere_admission import (
    GM1_CHILD_ATMOSPHERE_ARENA_NBYTES,
    GM1_CHILD_ATMOSPHERE_ARENA_SHA256,
    GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256,
    GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256,
    load_gm1_child_atmosphere,
)
from authority_trees import requires_work


ROOT = Path(__file__).resolve().parents[1]
ATMOSPHERE = ROOT / "work/gm1-t300-child-atmosphere-v1"


@requires_work("gm1-t300-child-atmosphere-v1")
def test_import_is_cuda_cold_and_real_artifact_is_exact() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cm1core.gm1_child_atmosphere_admission; "
                "assert not any(n == 'cupy' or n.startswith('cupy.') "
                "for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = load_gm1_child_atmosphere(ATMOSPHERE)
    assert loaded.manifest_sha256 == GM1_CHILD_ATMOSPHERE_MANIFEST_SHA256
    assert loaded.arena_sha256 == GM1_CHILD_ATMOSPHERE_ARENA_SHA256
    assert loaded.receipt_sha256 == GM1_CHILD_ATMOSPHERE_RECEIPT_SHA256
    assert loaded.arena_path.stat().st_size == GM1_CHILD_ATMOSPHERE_ARENA_NBYTES
    assert len(loaded.carriers) == 28
    assert tuple(loaded.carriers) == tuple(sorted(loaded.carriers))
    assert not loaded.closure_tke_ready
    assert not loaded.hydrostatic_base_ready
    assert not loaded.open_boundary_ready


@requires_work("gm1-t300-child-atmosphere-v1")
def test_views_are_read_only_exact_shapes_and_a_current_pairs_are_sealed() -> None:
    loaded = load_gm1_child_atmosphere(ATMOSPHERE)
    assert loaded.field("qa").shape == (6, 75, 300, 300)
    assert loaded.field("ua").shape == (75, 300, 301)
    assert loaded.field("va").shape == (75, 301, 300)
    assert loaded.field("wa").shape == (76, 300, 300)
    assert loaded.field("component.mp_rainnc").shape == (300, 300)
    assert loaded.field("qa").dtype == np.dtype("<f4")
    assert not loaded.field("qa").flags.writeable
    for current, prognostic in (
        ("pp3d", "ppi"),
        ("q3d", "qa"),
        ("th3d", "tha"),
        ("u3d", "ua"),
        ("v3d", "va"),
        ("w3d", "wa"),
    ):
        assert loaded.carriers[current].sha256 == loaded.carriers[prognostic].sha256
    assert float(np.min(loaded.field("qa"))) >= 0.0
    with pytest.raises(KeyError, match="unknown child-atmosphere carrier"):
        loaded.field("closure.tkea")


@requires_work("gm1-t300-child-atmosphere-v1")
def test_semantically_resealed_synthetic_translation_is_refused(tmp_path: Path) -> None:
    manifest = json.loads((ATMOSPHERE / "manifest.json").read_bytes())
    manifest["synthetic_translation"] = True
    raw = json.dumps(
        manifest, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    target = tmp_path / "atmosphere"
    target.mkdir()
    (target / "manifest.json").write_bytes(raw)
    (target / "receipt.json").write_bytes((ATMOSPHERE / "receipt.json").read_bytes())
    os.link(ATMOSPHERE / "arena.f32le", target / "arena.f32le")
    with pytest.raises(ValueError, match="model/schema/status changed"):
        load_gm1_child_atmosphere(
            target,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )

