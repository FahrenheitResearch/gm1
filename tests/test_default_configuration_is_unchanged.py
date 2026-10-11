"""Every distributed default is pinned, and every move of one is named.

This is not an assurance that nothing changes; it is the place a change has to
be written down.  Every example document must parse to exactly the canonical
record pinned here and be reported with exactly the same runnable decision,
and the module comment beside each seal says why it last moved.

Two moves are recorded.  The real-product lane's blocker lost four numerical
clauses because the things they named as missing exist; the refusal now names
only the gap that is still true (no installed door drives that lane) and a
remedy that works, exercised below.  And the acoustic off-centering weight
``alph`` moved from this port's 0.5 to CM1 r21.1's own documented 0.60, which
changes every psolver=3 result and therefore every sealed identity; the
comment on ``SEALED_CONFIG_SHA256`` states exactly which canonical keys that
added, and a test below pins that list so the next move is visible too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cm1core import engine_pin, preflight
from cm1core.config import Lane, load_config
from cm1core.errors import CM1Refusal
from cm1core.preflight import blockers, require_runnable

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# `CM1Config.sha256` of every distributed example.
#
# These moved once, deliberately, when CM1's own ``param2``/``param3``
# dynamical words became selectable.  Two things changed in the canonical
# record, and nothing else:
#
#   * ``dynamics.alph`` and ``dynamics.kdiv`` are now projected by value.
#     ``alph`` is README.namelist:756-759's documented default 0.60 and the
#     value all 23 of CM1's shipped decks set; this port ran 0.5, exactly
#     centred in time, so the vertically implicit acoustic step damped
#     nothing.  Moving to CM1's answer changes every psolver=3 result, so the
#     identity had to move with it: a hash that stayed still while the numbers
#     changed would name two different runs.
#   * ``dynamics.coriolis`` and ``dynamics.idiff`` are projected as the
#     selectors they are.  Both are 0 in every shipped example, which is what
#     this port always integrated and what CM1's own supercell deck sets.
#
# ``beta_plane``/``fcor_s_inv`` are projected only under icor=1 and the four
# diffusion reals only under idiff>=1, so a word no operator reads under a
# document's own selectors still cannot move that document's identity.
# ``test_only_the_named_dynamics_words_moved_the_seal`` below pins that list.
#
# They moved a second time when canonical_dict stopped projecting a top-level
# ``"tornado_diagnostics": null``.  a0cee5e added that dataclass field without
# removing it from the asdict payload, so every document -- including ones that
# declare no diagnostics at all -- hashed a key it never wrote, and d9509b9
# sealed the leaked digests here.  The rotation/swath words are still projected,
# inside ``diagnostics``, by the documents that declare them; nothing else in
# the canonical record changed.
SEALED_CONFIG_SHA256 = {
    "cartesian-supercell-attempt.toml": (
        "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b"
    ),
    "cartesian-supercell-translating-frame.toml": (
        "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca"
    ),
    "cartesian-uniform-grid.toml": (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    ),
    "cartesian-wk-analytic-sounding-shear.toml": (
        "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7"
    ),
    "dry-psolver2-bootstrap.toml": (
        "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340"
    ),
    "official-supercell-native-context.toml": (
        "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c"
    ),
    "official-supercell-product.toml": (
        "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038"
    ),
    "real-atmosphere-100m-product.toml": (
        "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00"
    ),
}

# The canonical digest of the complete blocker list each example is reported
# with, from the same pre-change tree.
#
# The five documents whose preflight reaches the ArWen estate re-digested when
# the seam was re-pinned from 2.5.8 to 2.7.4, because `CM1-ARWEN-ESTATE-DRIFT`
# names the version it admits inside its own reason text -- which is the point
# of that message: an operator reading it has to be told which estate is
# missing.  The DECISION did not move, and `SEALED_BLOCKER_CODES` below pins
# that separately, per document, so the digests here cannot be refreshed to
# paper over a blocker that actually appeared or disappeared.  The four
# Cartesian documents previously hashed to
# bf8579b357446b936c58b54cbeaee2260c08d9fe06d4b5aa504b2dbfe84ea025 and
# official-supercell-product.toml to
# 056bff5d05839abbdc21517e8d2f58c47a4ea39b0b9d78237c5f62067851a790.
# The same five re-digested again at the 2.7.4 -> 2.8.0 re-pin, for the same
# reason and with the same codes; under 2.7.4 the four Cartesian documents
# hashed to febf95e5336e561729d491442188d9721d57c9e3802b354cefa6fbca1a81304c
# and official-supercell-product.toml to
# 7c6ffb57881d442bce64f0b6570379b62f206ddfe5d8623c484298ca828138f9.
#: The digests below cover the decisions as GM1 spells them now (gate composed-trajectory, code
#: CM1-INSTRUMENTS-MISSING, "every time-dependent operator deck"); every other byte of every decision is the
#: one the earlier digests sealed, checked by applying the same rename to the old decisions.
SEALED_BLOCKER_SHA256 = {
    "cartesian-supercell-attempt.toml": (
        "ca6857bf6d6bf302c51b88f87f0f84fb8e7aaaaf7a12363d4e1aa0d4797ead08"
    ),
    "cartesian-supercell-translating-frame.toml": (
        "ca6857bf6d6bf302c51b88f87f0f84fb8e7aaaaf7a12363d4e1aa0d4797ead08"
    ),
    "cartesian-uniform-grid.toml": (
        "ca6857bf6d6bf302c51b88f87f0f84fb8e7aaaaf7a12363d4e1aa0d4797ead08"
    ),
    "cartesian-wk-analytic-sounding-shear.toml": (
        "ca6857bf6d6bf302c51b88f87f0f84fb8e7aaaaf7a12363d4e1aa0d4797ead08"
    ),
    "dry-psolver2-bootstrap.toml": (
        "fcccab04639cef332305f01d8d8d12d7e7528f9dde1619b2f893ae7b1a7e6a75"
    ),
    "official-supercell-native-context.toml": (
        "3518414d160292e39dfe6e7afc0b02613d60b1f3e7397b54b42cccd916ceb653"
    ),
    "official-supercell-product.toml": (
        "a73f526431f2d4ec0cdbb372451edbebb6f5bffb305a5687c0cb175389bfe09c"
    ),
}



# The blocker CODES each example is reported with, which is the decision
# itself: a digest above may move because a refusal's prose was corrected, but
# a code appearing, disappearing or changing order is a behaviour change and
# must be argued for, not re-sealed.
SEALED_BLOCKER_CODES = {
    "cartesian-supercell-attempt.toml": ["CM1-ARWEN-ESTATE-DRIFT"],
    "cartesian-supercell-translating-frame.toml": ["CM1-ARWEN-ESTATE-DRIFT"],
    "cartesian-uniform-grid.toml": ["CM1-ARWEN-ESTATE-DRIFT"],
    "cartesian-wk-analytic-sounding-shear.toml": ["CM1-ARWEN-ESTATE-DRIFT"],
    "dry-psolver2-bootstrap.toml": [
        "CM1-CUDA-TRAJECTORY-INCOMPLETE",
        "CM1-INSTRUMENTS-MISSING",
        "CM1-DRY-PSOLVER2-ORACLE-MISSING",
        "CM1-RUST-LIVE-WIRING-MISSING",
    ],
    "official-supercell-native-context.toml": [
        "CM1-CUDA-TRAJECTORY-INCOMPLETE",
        "CM1-INSTRUMENTS-MISSING",
        "CM1-NATIVE-CONTEXT-NOT-A-PRODUCT-DOOR",
        "CM1-RUST-LIVE-WIRING-MISSING",
    ],
    "official-supercell-product.toml": [
        "CM1-CUDA-TRAJECTORY-INCOMPLETE",
        "CM1-INSTRUMENTS-MISSING",
        "CM1-ARWEN-ESTATE-DRIFT",
        "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING",
        "CM1-RUST-LIVE-WIRING-MISSING",
    ],
}

@pytest.fixture
def absent_estate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report blockers against a machine with no ArWen estate installed.

    ``CM1-ARWEN-ESTATE-DRIFT`` is an installation condition of the machine
    (preflight.py asks ``engine_pin.inspect``), and its reason text names what
    was found.  The seals below were taken with no gpuwm installed, so every
    decision test pins that estate explicitly: otherwise the same tree reports
    one blocker list where gpuwm 2.8.0 is installed and another where it is
    not, and the seal would describe the machine rather than the document.
    """

    monkeypatch.setattr(engine_pin, "installed_version", lambda: None)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: None)
    engine_pin._INSPECTION_CACHE.clear()
    yield
    engine_pin._INSPECTION_CACHE.clear()


def _blocker_digest(path: Path) -> str:
    items = [item.as_dict() for item in blockers(load_config(path))]
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_every_distributed_example_is_still_present() -> None:
    """Every sealed example still ships, and every new one is a real deck.

    This asked for set equality, which made the distribution unable to grow: a
    branch that ships a worked example alongside a new CM1 branch -- the fifty
    metre fine step, the stretched tornado-grade grid, the surface-drag
    supercell -- failed a test whose subject is that *existing* defaults did not
    move. Shipping a new document changes no existing default.

    What the equality was really buying is that nothing leaves or is renamed
    without the seal following, and that an unreviewed file cannot be dropped
    into examples/ and inherit the distribution's credibility. Both are checked
    directly: every sealed name must still be there, and every example that is
    not sealed must parse into a configuration and be reportable, so it is a
    deck this port actually admits rather than a stray file.
    """

    present = {path.name for path in EXAMPLES.glob("*.toml")}
    missing = sorted(set(SEALED_CONFIG_SHA256) - present)
    assert not missing, (
        f"sealed example(s) {missing} are no longer in the distribution; a "
        "removal or rename must move the seal with them"
    )
    for name in sorted(present - set(SEALED_CONFIG_SHA256)):
        config = load_config(EXAMPLES / name)
        assert len(config.sha256) == 64
        # Reportable: `blockers` runs the whole preflight over it.
        assert isinstance(blockers(config), tuple | list)


@pytest.mark.parametrize("name", sorted(SEALED_CONFIG_SHA256))
def test_the_parsed_configuration_record_is_bit_identical(name: str) -> None:
    assert load_config(EXAMPLES / name).sha256 == SEALED_CONFIG_SHA256[name]


# The canonical [dynamics] keys of a document that declares none of CM1's
# param2/param3 words, exactly as the seals above were computed from.
_PROJECTED_DYNAMICS_KEYS = {
    "adaptive_timestep",
    "alph",
    "coriolis",
    "dry_mass_correction",
    "idiff",
    "kdiv",
    "psolver",
    "scalar_horizontal_order",
    "scalar_vertical_order",
    "scalar_weno",
    "velocity_horizontal_order",
    "velocity_vertical_order",
    "velocity_weno",
    "weno_order",
}


@pytest.mark.parametrize("name", sorted(SEALED_CONFIG_SHA256))
def test_only_the_named_dynamics_words_moved_the_seal(name: str) -> None:
    """A word no operator reads must not appear in any identity.

    The seals above moved once because ``alph`` and ``kdiv`` are read by every
    psolver=3 small step and are now projected by value.  ``beta_plane``,
    ``fcor_s_inv`` and the four diffusion reals are read only under selectors
    no shipped example sets, so they must stay out of the record entirely --
    otherwise the next word added to [dynamics] moves eight identities for a
    number nothing reads.
    """

    dynamics = load_config(EXAMPLES / name).canonical_dict()["dynamics"]
    assert set(dynamics) == _PROJECTED_DYNAMICS_KEYS
    # README.namelist:756-759 and run/namelist.input:95-96: CM1's own values.
    assert dynamics["alph"] == 0.60
    assert dynamics["kdiv"] == 0.10


@pytest.mark.parametrize("name", sorted(SEALED_BLOCKER_SHA256))
def test_the_reported_runnable_decision_is_bit_identical(
    name: str, absent_estate: None
) -> None:
    assert _blocker_digest(EXAMPLES / name) == SEALED_BLOCKER_SHA256[name]


@pytest.mark.parametrize("name", sorted(SEALED_BLOCKER_CODES))
def test_the_reported_blocker_codes_are_unchanged(
    name: str, absent_estate: None
) -> None:
    """The decision, separately from the words it is reported in."""
    reported = [item.code for item in blockers(load_config(EXAMPLES / name))]
    assert reported == SEALED_BLOCKER_CODES[name]


@pytest.mark.parametrize("name", sorted(SEALED_BLOCKER_CODES))
def test_an_admitted_estate_removes_exactly_the_drift_blocker(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other installation state: the estate is byte-admitted.

    Nothing else in the decision may depend on what is installed, so the
    reported blockers are the sealed ones with only the estate refusal gone.
    """

    class _Admitted:
        admitted = True

    monkeypatch.setattr(preflight, "inspect", lambda **_: _Admitted())
    reported = [item.code for item in blockers(load_config(EXAMPLES / name))]
    assert reported == [
        code for code in SEALED_BLOCKER_CODES[name]
        if code != "CM1-ARWEN-ESTATE-DRIFT"
    ]


def test_the_one_deliberate_change_is_the_real_product_refusal(
    absent_estate: None,
) -> None:
    name = "real-atmosphere-100m-product.toml"
    reported = {
        item.code: item for item in blockers(load_config(EXAMPLES / name))
    }
    # Same lane, same document, same set of blockers except the one clause that
    # was no longer true.
    assert "CM1-REAL-100M-LIVE-WIRING-MISSING" not in reported
    assert "CM1-REAL-PRODUCT-NO-INSTALLED-DOOR" in reported
    assert set(reported) - {"CM1-REAL-PRODUCT-NO-INSTALLED-DOOR"} == {
        "CM1-CUDA-TRAJECTORY-INCOMPLETE",
        "CM1-INSTRUMENTS-MISSING",
        "CM1-ARWEN-ESTATE-DRIFT",
        "CM1-EQTSET2-COUPLED-TRAJECTORY-MISSING",
        "CM1-RUST-LIVE-WIRING-MISSING",
    }


def test_the_real_product_remedy_actually_works() -> None:
    """The remedy must reach a door, not another lane refusal.

    A refusal that names a remedy which is itself refused leaves the operator
    with nothing to do.  The real-product blocker tells the reader to select
    the Cartesian lane, so that lane must carry no blocker of its own: the only
    thing that can still stop it is the engine pin, which is an installation
    condition of this machine and not a property of the configuration.
    """

    blocker = next(
        item
        for item in blockers(load_config(EXAMPLES / "real-atmosphere-100m-product.toml"))
        if item.code == "CM1-REAL-PRODUCT-NO-INSTALLED-DOOR"
    )
    assert "run.lane = 'cartesian'" in blocker.remedy

    cartesian = load_config(EXAMPLES / "cartesian-uniform-grid.toml")
    assert cartesian.run.lane is Lane.CARTESIAN
    remaining = {item.code for item in blockers(cartesian)}
    assert remaining <= {"CM1-ARWEN-ESTATE-DRIFT"}, remaining
    if remaining:
        # The substituted-physics engine is not installed on this machine, so
        # `require_runnable` stops here for a reason that has nothing to do
        # with the lane.  Prove that is the only thing in the way.
        with pytest.raises(CM1Refusal) as refusal:
            require_runnable(cartesian)
        assert refusal.value.detail.code == "CM1-ARWEN-ESTATE-DRIFT"
    else:
        require_runnable(cartesian)
