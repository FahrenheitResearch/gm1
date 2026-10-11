from __future__ import annotations

from pathlib import Path

import pytest

from cm1core.config import load_config
from cm1core.errors import CM1Refusal
from cm1core.preflight import blockers, require_runnable


ROOT = Path(__file__).resolve().parents[1]


def test_product_names_all_known_scientific_blockers() -> None:
    config = load_config(ROOT / "examples" / "official-supercell-product.toml")
    codes = {item.code for item in blockers(config)}
    assert "CM1-CUDA-TRAJECTORY-INCOMPLETE" in codes
    assert "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING" in codes
    assert "CM1-ARWEN-SEAM-MISSING" not in codes
    assert "CM1-RUST-LIVE-WIRING-MISSING" in codes


def test_first_runtime_refusal_occurs_before_any_backend() -> None:
    config = load_config(ROOT / "examples" / "official-supercell-product.toml")
    with pytest.raises(CM1Refusal) as caught:
        require_runnable(config)
    assert caught.value.detail.code == "CM1-CUDA-TRAJECTORY-INCOMPLETE"


def test_dry_bootstrap_names_its_missing_psolver2_oracle() -> None:
    config = load_config(ROOT / "examples" / "dry-psolver2-bootstrap.toml")
    codes = {item.code for item in blockers(config)}
    assert "CM1-DRY-PSOLVER2-ORACLE-MISSING" in codes


def test_real_product_names_the_only_gap_that_is_still_true() -> None:
    """The four numerical gaps are closed; only the missing door remains.

    The retired refusal named the Rust HRRR-to-CM1 mapper, geometric-w
    conversion, CM1 carrier publication and time-varying lateral forcing as
    absent.  All four are implemented and hash-admitted, so the refusal must no
    longer claim they are missing - it must name the one thing that is still
    true, which is that no installed door executes this lane.
    """

    config = load_config(ROOT / "examples" / "real-atmosphere-100m-product.toml")
    gaps = {item.code: item for item in blockers(config)}
    assert "CM1-REAL-100M-LIVE-WIRING-MISSING" not in gaps
    assert "CM1-REAL-PRODUCT-NO-INSTALLED-DOOR" in gaps
    reason = gaps["CM1-REAL-PRODUCT-NO-INSTALLED-DOOR"].reason
    assert "implemented and hash-admitted" in reason
    assert "installs no door" in reason
    assert "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING" in gaps
    assert "CM1-RUST-LIVE-WIRING-MISSING" in gaps
