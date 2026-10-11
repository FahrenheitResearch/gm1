"""Thompson (CM1 ptype=3) and WDM6 (no CM1 ptype) as bound column components.

Every check here runs without a CUDA device and without ``gpuwm``: the point of
the CM1-owned adapter is that a wrong scalar axis, a wrong moment map or a
wrong dense-ice selector is refused on the host, before any device work is
priced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1core.arwen_scheme_components import (
    ARWEN_THOMPSON_AEROSOL_COMPONENT_ID,
    ARWEN_THOMPSON_COMPONENT_ID,
    THOMPSON_AEROSOL_SPEC,
    ARWEN_WDM6_COMPONENT_ID,
    THOMPSON_CARRIER_NAMES,
    THOMPSON_SPEC,
    WDM6_CARRIER_NAMES,
    WDM6_SPEC,
    thompson_component_metadata,
    wdm6_component_metadata,
)
from cm1core.column_components import COLUMN_COMPONENT_SPECS, column_component_spec
from cm1core.component_validation import UNVALIDATED_MARKER, component_validation
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_bindings_touches_no_runtime() -> None:
    """A misconfigured scheme must refuse before any device work is priced.

    Checked in a fresh interpreter rather than against this one's
    ``sys.modules``: a full-suite run has other tests in it, and what matters
    is that importing this module alone pulls in neither runtime.
    """

    import subprocess
    import sys

    source = (ROOT / "src" / "cm1core" / "arwen_scheme_components.py").read_text(
        encoding="utf-8"
    )
    assert "import cupy" not in source
    assert "import gpuwm" not in source
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import cm1core.arwen_scheme_components as m; "
            "print(sorted(n for n in sys.modules if n.split('.')[0] "
            "in ('cupy', 'gpuwm')))",
        ],
        cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "[]", completed.stdout


def test_the_thompson_axis_is_cm1_ptype_three() -> None:
    """src/param.F:3322-3398, and ArWen's own required tuple under its names."""

    assert THOMPSON_SPEC.cm1_ptype == 3
    assert THOMPSON_SPEC.mp_physics == 8
    assert THOMPSON_SPEC.profile.numq == 8
    assert THOMPSON_SPEC.species == (
        "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr"
    )
    # ArWen's classic Thompson launcher requires qc,qi,ni,qs,qg,qr,nr; CM1's
    # seventh scalar is cloud-ice number and its eighth is rain number.  Every
    # consumer goes through this map, so the two are never bound positionally.
    assert THOMPSON_CARRIER_NAMES["nci"] == "ni"
    assert THOMPSON_CARRIER_NAMES["ncr"] == "nr"
    assert [THOMPSON_SPEC.arwen_carrier(name) for name in THOMPSON_SPEC.species] == [
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr"
    ]
    # src/param.F:936-950 does not list ptype=3, so CM1 keeps eqtset=2 for it,
    # which is the source order this lane's large step executes.
    assert THOMPSON_SPEC.profile.cm1_eqtset == 2


def test_the_wdm6_axis_is_arwens_and_transcribed_but_not_selectable() -> None:
    """The axis is real work, kept; the scheme is refused by a measured blocker.

    Dispatching a WDM6 column through this adapter refuses inside ArWen at
    gpuwm/core/wdm6.py:156, which needs WRF's XLAND land/sea mask.  That was
    measured on a device, not inferred, and it is recorded in
    cm1core.column_component_blockers rather than left as a surprise.
    """

    from cm1core.column_component_blockers import COLUMN_COMPONENT_BLOCKERS

    assert ARWEN_WDM6_COMPONENT_ID not in COLUMN_COMPONENT_SPECS
    blocker = COLUMN_COMPONENT_BLOCKERS[ARWEN_WDM6_COMPONENT_ID]
    assert blocker.blocker == "arwen-wdm6-requires-an-xland-land-sea-mask"
    assert "measured, not inferred" in blocker.detail

    assert WDM6_SPEC.cm1_ptype is None
    assert WDM6_SPEC.mp_physics == 16
    assert WDM6_SPEC.profile.numq == 9
    assert WDM6_SPEC.species == (
        "qv", "qc", "qr", "qi", "qs", "qg", "ncn", "ncc", "ncr"
    )
    # gpuwm/core/wdm6.py:151 requires nn (CCN), nc and nr beside the six.
    assert [WDM6_SPEC.arwen_carrier(name) for name in WDM6_SPEC.species] == [
        "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr"
    ]
    assert WDM6_CARRIER_NAMES["ncn"] == "nn"
    assert WDM6_SPEC.extra_execution["cm1_scheme_present"] is False
    assert "names no WDM6 scheme" in WDM6_SPEC.extra_execution["cm1_absence"]
    assert "CM1 r21.1 has no WDM6 ptype" in WDM6_SPEC.profile.axis_authority


def test_both_axes_keep_cm1s_six_mass_rows_first() -> None:
    """The sealed post-physics EOS/qa kernel publishes exactly six mass rows.

    A component whose mass prefix is not CM1's own six has no publication path
    at all, which is why the blocked register exists.
    """

    for spec in (THOMPSON_SPEC, WDM6_SPEC, THOMPSON_AEROSOL_SPEC):
        assert spec.profile.mass_indices == tuple(range(6)), spec.component_id
        assert spec.species[:6] == ("qv", "qc", "qr", "qi", "qs", "qg")
        assert spec.profile.repaired_scalars == 6


def test_the_weno_epsilon_scales_with_the_row_it_guards() -> None:
    """src/solve2.F:1472 forms weps as qmag(n)*epsilon, per scalar.

    A number concentration given a mass row's qmag would be guarded by an
    epsilon twenty-two orders of magnitude below the field.
    """

    assert THOMPSON_SPEC.profile.qmag == (1.0e-2,) * 6 + (1.0e6, 1.0e6)
    assert WDM6_SPEC.profile.qmag == (1.0e-2,) * 6 + (1.0e6,) * 3


def test_thompson_has_no_dense_ice_selector_and_wdm6_has_wsm6s() -> None:
    from cm1core.dense_ice import dense_ice_selection

    thompson = dense_ice_selection(ARWEN_THOMPSON_COMPONENT_ID, None)
    assert thompson.arwen_option is None and thompson.ihail is None
    with pytest.raises(CM1Refusal) as captured:
        dense_ice_selection(ARWEN_THOMPSON_COMPONENT_ID, 1)
    assert captured.value.detail.code == "GM1-DENSE-ICE-SELECTOR-ABSENT"

    wdm6 = dense_ice_selection(ARWEN_WDM6_COMPONENT_ID, 1)
    assert wdm6.arwen_option == "wdm6_hail_opt"
    assert (wdm6.constants.n0g, wdm6.constants.deng) == (4.0e4, 700.0)


def test_metadata_is_complete_and_carries_its_own_identity() -> None:
    for metadata, spec in (
        (thompson_component_metadata(), THOMPSON_SPEC),
        (wdm6_component_metadata(), WDM6_SPEC),
    ):
        assert metadata["component_id"] == spec.component_id
        assert metadata["component_abi"] == spec.component_abi
        assert metadata["execution"]["mp_physics"] == spec.mp_physics
        assert metadata["carriers"]["species"] == list(spec.species)
        assert metadata["moisture_species"]["numq"] == spec.profile.numq
        assert metadata["persistence"]["component_owned"] == list(
            spec.component_persistence
        )
    # Two different schemes must never hash to one document.
    from cm1core.arwen_scheme_components import (
        thompson_component_metadata_sha256,
        wdm6_component_metadata_sha256,
    )

    assert thompson_component_metadata_sha256() != wdm6_component_metadata_sha256()


def test_the_capacity_overlay_is_refused_for_a_scheme_it_never_rebuilt() -> None:
    from cm1core.arwen_scheme_components import create_arwen_thompson_component

    with pytest.raises(CM1Refusal) as captured:
        create_arwen_thompson_component(
            component_id=ARWEN_THOMPSON_COMPONENT_ID, nx=2, ny=2, nz=8,
            dx_m=1000.0, dy_m=1000.0, dz_m=250.0, dt_s=0.5,
            capacity_profile="gm1.arwen-wsm6-column-capacity-256/v1",
        )
    assert captured.value.detail.code == "CM1-THOM-CAPACITY-PROFILE"


def test_wdm6_refuses_a_column_deeper_than_its_compiled_tier() -> None:
    """gpuwm/core/wdm6_constants.py WDM6_DEEP_KMAX is 80, not 256."""

    from cm1core.arwen_scheme_components import create_arwen_wdm6_component

    with pytest.raises(CM1Refusal) as captured:
        create_arwen_wdm6_component(
            component_id=ARWEN_WDM6_COMPONENT_ID, nx=2, ny=2, nz=81,
            dx_m=1000.0, dy_m=1000.0, dz_m=250.0, dt_s=0.5,
        )
    assert captured.value.detail.code == "CM1-WDM6-VERTICAL-BOUNDS"


def test_every_bound_component_declares_a_validation_state() -> None:
    """A bound component without one would publish receipts that look measured."""

    for component_id in COLUMN_COMPONENT_SPECS:
        state = component_validation(component_id)
        assert state.checked, component_id
        if not state.validated:
            assert state.not_checked, component_id
            assert UNVALIDATED_MARKER in state.sentence()
    assert component_validation("wsm6-mp6").validated is True
    assert component_validation(ARWEN_THOMPSON_COMPONENT_ID).validated is False
    # WDM6 is transcribed and not selectable, so it declares no validation
    # state at all: a state here would be a claim about a component no lane
    # can run.
    assert ARWEN_WDM6_COMPONENT_ID not in COLUMN_COMPONENT_SPECS
    with pytest.raises(KeyError):
        component_validation(ARWEN_WDM6_COMPONENT_ID)


def test_the_shipped_document_selects_the_new_component() -> None:
    """Config text to a bound axis, with no device."""

    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import load_config

    thompson = build_runtime_config(
        load_config(ROOT / "examples" / "cartesian-supercell-thompson.toml")
    )
    assert thompson.microphysics == ARWEN_THOMPSON_COMPONENT_ID
    assert thompson.numq == 8 and thompson.ihail is None
    assert column_component_spec(thompson.microphysics) is THOMPSON_SPEC


def test_doctor_says_out_loud_which_capabilities_are_unmeasured() -> None:
    """The unvalidated-capability marker, rendered.

    Every selectable component that nothing has measured prints as UNVALID and
    carries the grep token plus the list of what was not checked; the schemes
    that are not selectable at all print their one blocker.  Neither is a
    blocking finding: an unmeasured capability is exposed, not refused.
    """

    from cm1core import doctor

    findings = doctor._column_component_findings()
    rows = {
        item.subject: item
        for item in findings
        if item.subject.startswith("column component ")
    }
    assert set(rows) == {
        f"column component {name}" for name in COLUMN_COMPONENT_SPECS
    }
    assert rows["column component wsm6-mp6"].status == doctor.OK
    for name in ("morrison-mp10", "thompson-mp8", "thompson-aerosol-mp28"):
        item = rows[f"column component {name}"]
        assert item.status == doctor.UNVALIDATED, name
        assert UNVALIDATED_MARKER in item.detail, name
        assert item.required is False, name
    assert doctor.blocking(findings) == []
    assert len(doctor.unvalidated(findings)) == 3

    rendered = doctor.render(findings)
    assert "UNVALID" in rendered
    assert "unvalidated exposed capabilities: 3" in rendered

    blocked = [
        item
        for item in findings
        if item.subject == "column components not yet selectable"
    ]
    assert len(blocked) == 1
    from cm1core.column_component_blockers import COLUMN_COMPONENT_BLOCKERS

    for component_id in COLUMN_COMPONENT_BLOCKERS:
        assert component_id in blocked[0].detail


def test_a_receipt_from_an_unmeasured_component_says_so() -> None:
    """The same words reach the run receipt, not just the console."""

    from cm1core.component_validation import component_validation

    record = component_validation("thompson-mp8").record()
    assert record["validated"] is False
    assert record["marker"] == UNVALIDATED_MARKER
    assert any("a differential against native CM1" in line for line in record["not_checked"])
    measured = component_validation("wsm6-mp6").record()
    assert measured["validated"] is True
    assert "marker" not in measured


def test_aerosol_aware_thompson_is_bound_with_its_surface_aerosol_carriers() -> None:
    """Eleven transported scalars, two surface-owned ones, and w.

    ``gpuwm/core/microphysics_aerosol.py:220`` requires ``nwfa2d`` and
    ``nifa2d`` by name, and :246,463 read ``state.w[:-1]``.  The two aerosol
    emission rates are WRF's two-dimensional surface fields, so they are
    component-owned SURFACE persistence, not rows on the transported axis: a
    scheme that carried them as column scalars would advect an emission rate.
    """

    spec = THOMPSON_AEROSOL_SPEC
    assert spec.mp_physics == 28 and spec.cm1_ptype is None
    assert spec.profile.numq == 11
    assert spec.extra_interface_carriers == ("w",)
    assert tuple(spec.component_owned_surface_carriers) == ("nwfa2d", "nifa2d")
    assert spec.surface_persistence[:2] == ("nwfa2d", "nifa2d")
    assert set(spec.component_owned_surface_carriers).isdisjoint(spec.species)
    # The receipt has to say the aerosol state is ArWen's fallback, because it
    # is: this seam never calls gpuwm.core.microphysics.microphysics_init.
    assert "fallback" in spec.extra_execution["aerosol_initial_state"]
    validation = component_validation(spec.component_id)
    assert validation.validated is False
    assert any("fallback" in line for line in validation.not_checked)
