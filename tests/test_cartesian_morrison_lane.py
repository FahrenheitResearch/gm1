"""From a TOML document to a different device state: the Morrison lane.

These are CPU storage/inventory checks.  No kernel is launched and no CUDA
context is opened; what they prove is that ``physics.component`` and
``physics.ihail`` reach the carriers, the allocation plan, the published
frame, the diagnostics and the ArWen selector, and that the default WSM6
configuration is bit-for-bit the one this port already ran.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from cm1core.arwen_morrison_component import MORRISON_SPEC
from cm1core.arwen_wsm6_component import WSM6_SPEC
from cm1core.cartesian_door import build_runtime_config, require_level_capacity
from cm1core.cartesian_history import cartesian_history_profile
from cm1core.column_components import column_component_spec
from cm1core.config import load_config, parse_mapping
from cm1core.cuda_cartesian_closure import cartesian_closure_allocation_plan
from cm1core.cuda_cartesian_dynamics import (
    CartesianDynamics,
    cartesian_dynamics_allocation_plan,
)
from cm1core.cuda_cartesian_finish import cartesian_wsm6_scratch_reserve
from cm1core.cuda_gm1_tilestream_column import tiled_column_device_buffer_shapes
from cm1core.cuda_scalar_moist_rk import moisture_weps_f64
from cm1core.errors import CM1Refusal
from cm1core.gm1_idealized_residency import IdealizedSource
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry

from test_cuda_cartesian_dynamics import CpuStorage, MetadataOwner, Stream


ROOT = Path(__file__).resolve().parents[1]
MORRISON_EXAMPLE = ROOT / "examples" / "cartesian-supercell-morrison-hail.toml"
WSM6_EXAMPLE = ROOT / "examples" / "cartesian-supercell-translating-frame.toml"


def _raw(path: Path) -> dict[str, object]:
    """The document exactly as TOML gives it, for by-value refusal cases."""

    return tomllib.loads(path.read_text(encoding="utf-8"))


class _CpuStorage(CpuStorage):
    """The dynamics harness plus the one array operation this lane adds."""

    @staticmethod
    def copyto(destination, source) -> None:
        np.copyto(destination, source)


def _grid(nz: int = 8) -> CartesianGridGeometry:
    return CartesianGridGeometry(9, 11, nz, 75.0, 75.0, 75.0)


def _runtime(**changes: object) -> RuntimeConfig:
    values: dict[str, object] = {
        "grid": _grid(),
        "memory_budget_bytes": 400_000_000,
    }
    values.update(changes)
    return RuntimeConfig(**values)  # type: ignore[arg-type]


def _dynamics(monkeypatch, config: RuntimeConfig) -> CartesianDynamics:
    stream = Stream()
    storage = _CpuStorage(stream)
    monkeypatch.setattr(
        CartesianDynamics,
        "_compile_operators",
        lambda self: (
            setattr(self, "moist_compiled", object()),
            setattr(self, "open_compiled", object()),
        ),
    )
    monkeypatch.setattr(CartesianDynamics, "_initial_dry_mass", lambda self: 1.0)
    monkeypatch.setattr(
        CartesianDynamics,
        "_publish_startup_thermodynamics",
        lambda self: {"schema": "stub", "performed": False},
    )
    with IdealizedSource(
        "unused", b"metadata", config.grid, owner_factory=MetadataOwner
    ) as source:
        return CartesianDynamics(source, config, cupy=storage, stream=stream)


# ---------------------------------------------------------------- selection


def test_the_document_selects_the_component_and_its_dense_ice_branch() -> None:
    config = load_config(MORRISON_EXAMPLE)
    assert config.physics.component == "morrison-mp10"
    assert config.physics.ihail == 1
    assert config.physics.ndcnst == 250.0
    runtime = build_runtime_config(config)
    assert runtime.microphysics == "morrison-mp10"
    assert runtime.numq == 10
    assert runtime.ihail == 1
    assert runtime.dense_ice.category == "hail"
    # The ArWen option this integer is handed to is the component's own.
    assert runtime.dense_ice.arwen_option == "morr_rimed_ice"
    assert runtime.record()["dense_ice"]["selected_constants"]["rhog_kg_m3"] == 900.0
    assert column_component_spec(runtime.microphysics) is MORRISON_SPEC


def test_the_shipped_wsm6_document_now_binds_cm1s_own_hail_branch() -> None:
    """The shipped document declares no ihail and its meaning has moved.

    run/namelist.input:56 and run/config_files/supercell/namelist.input:55 ship
    ihail=1, and src/param.F:3697-3706 carries that integer into WSM6.  GM1's
    undeclared case bound 0 until the default moved to CM1's answer, so this
    document's dense ice is now hail-like: n0g 4e4 rather than 4e6.  A run that
    wants the previously recorded words declares physics.ihail = 0.
    """

    config = load_config(WSM6_EXAMPLE)
    assert config.physics.ihail is None
    assert "ihail" not in config.canonical_dict()["physics"]
    runtime = build_runtime_config(config)
    assert (runtime.microphysics, runtime.numq, runtime.ihail) == ("wsm6-mp6", 6, 1)
    assert runtime.dense_ice.arwen_option == "wsm6_hail_opt"
    assert runtime.dense_ice.constants.n0g == 4.0e4
    assert runtime.dense_ice.constants.deng == 700.0
    # The branch the document used to get, still one integer away.
    raw = _raw(WSM6_EXAMPLE)
    raw["physics"]["ihail"] = 0
    previous = build_runtime_config(parse_mapping(raw))
    assert previous.dense_ice.constants.n0g == 4.0e6
    assert column_component_spec(runtime.microphysics) is WSM6_SPEC


def test_an_undeclared_ihail_binds_cm1s_own_deck_value_for_each_component() -> None:
    """A document that omits ihail still names one dense-ice branch per scheme.

    CM1 supplies no compiled ihail default -- ``src/param.F:1567-1580`` stops
    the model outside 0..1 -- so an undeclared selector has to bind something,
    and every component with a CM1 ptype binds what CM1 itself ships:
    ``run/namelist.input:55-56`` and
    ``run/config_files/supercell/namelist.input:54-55`` both say ``ihail = 1``.
    ``src/param.F:3407`` carries it into Morrison and ``:3697-3706`` into WSM6.
    This is the whole point of the selector: the omitted case must not silently
    run the reference supercell's dense ice as graupel where CM1 runs it as
    hail.
    """

    raw = _raw(MORRISON_EXAMPLE)
    del raw["physics"]["ihail"]
    runtime = build_runtime_config(parse_mapping(raw))
    assert runtime.ihail == 1
    assert runtime.dense_ice.category == "hail"
    # src/morrison.F:407-412,441-444 -- the constants the branch actually is.
    assert (
        runtime.dense_ice.constants.ag,
        runtime.dense_ice.constants.bg,
        runtime.dense_ice.constants.rhog,
    ) == (114.5, 0.5, 900.0)

    wsm6 = build_runtime_config(load_config(WSM6_EXAMPLE))
    assert (wsm6.ihail, wsm6.dense_ice.category) == (1, "hail")
    # src/mp_wsm6.F:88-93 -- the constants the WSM6 branch actually is.
    assert (
        wsm6.dense_ice.constants.n0g,
        wsm6.dense_ice.constants.deng,
        wsm6.dense_ice.constants.avtg,
        wsm6.dense_ice.constants.lamdagmax,
    ) == (4.0e4, 700.0, 285.0, 2.0e4)


def test_declaring_ihail_changes_the_transcribed_constants_not_just_the_record() -> None:
    """The same document with the flag flipped binds a different fall speed.

    This is the config-text-to-behaviour path: one integer in ``[physics]``
    reaches the constants ``V = A*D**B`` is evaluated with, and hail falls
    enough faster than graupel that precipitation is carried a different
    distance from the updraft.
    """

    raw = _raw(MORRISON_EXAMPLE)
    raw["physics"]["ihail"] = 0
    graupel = build_runtime_config(parse_mapping(raw))
    raw["physics"]["ihail"] = 1
    hail = build_runtime_config(parse_mapping(raw))

    assert graupel.dense_ice.constants.record() != hail.dense_ice.constants.record()
    assert (graupel.dense_ice.constants.ag, graupel.dense_ice.constants.bg) == (19.3, 0.37)
    assert (hail.dense_ice.constants.ag, hail.dense_ice.constants.bg) == (114.5, 0.5)
    # One centimetre is severe-criterion hail; both branches are evaluated on
    # the same diameter so only the selected constants differ.
    assert hail.dense_ice.constants.fall_speed_m_s(0.01) > (
        2.0 * graupel.dense_ice.constants.fall_speed_m_s(0.01)
    )
    # Selecting the branch never changes what the column transports.
    assert graupel.numq == hail.numq == 10
    assert graupel.profile.names == hail.profile.names


def test_ihail_is_refused_by_value_from_the_document() -> None:
    raw = _raw(MORRISON_EXAMPLE)
    raw["physics"]["ihail"] = 2
    with pytest.raises(CM1Refusal) as captured:
        parse_mapping(raw)
    assert captured.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert captured.value.detail.option == "physics.ihail"


def test_ndcnst_is_refused_outside_the_component_that_reads_it() -> None:
    raw = _raw(WSM6_EXAMPLE)
    raw["physics"]["ndcnst"] = 250.0
    with pytest.raises(CM1Refusal) as captured:
        parse_mapping(raw)
    assert captured.value.detail.code == "CM1-CFG-CARTESIAN-NDCNST-SCOPE"
    raw = _raw(MORRISON_EXAMPLE)
    raw["physics"]["ndcnst"] = 100.0
    with pytest.raises(CM1Refusal) as captured:
        parse_mapping(raw)
    assert captured.value.detail.code == "GM1-MORRISON-NDCNST"


def test_an_untranscribed_component_is_refused_by_the_lane() -> None:
    raw = _raw(WSM6_EXAMPLE)
    raw["physics"]["component"] = "goddard-lfo"
    with pytest.raises(CM1Refusal) as captured:
        parse_mapping(raw)
    assert captured.value.detail.option == "physics.component"


def test_every_bound_component_is_selectable_from_a_document() -> None:
    """Config text to a bound scalar axis, for all four, with no device."""

    from cm1core.column_components import COLUMN_COMPONENT_SPECS

    expected = {
        "wsm6-mp6": (6, 1), "morrison-mp10": (10, 1), "thompson-mp8": (8, None),
        "thompson-aerosol-mp28": (11, None),
    }
    assert set(expected) == set(COLUMN_COMPONENT_SPECS)
    for component, (numq, ihail) in expected.items():
        raw = _raw(WSM6_EXAMPLE)
        raw["physics"]["component"] = component
        runtime = build_runtime_config(parse_mapping(raw))
        assert (runtime.numq, runtime.ihail) == (numq, ihail), component
        assert runtime.profile.mass_indices == tuple(range(6)), component


def test_the_morrison_column_capacity_is_its_own_kernels_depth() -> None:
    assert require_level_capacity(load_config(MORRISON_EXAMPLE)) == (8, 256)
    assert require_level_capacity(load_config(WSM6_EXAMPLE)) == (8, 80)


# ------------------------------------------------------- device inventories


def test_every_scalar_carrier_is_sized_from_cm1s_numq(monkeypatch) -> None:
    for component, numq in (
        ("wsm6-mp6", 6), ("morrison-mp10", 10), ("thompson-mp8", 8),
        ("thompson-aerosol-mp28", 11),
    ):
        config = _runtime(microphysics=component)
        plan = cartesian_dynamics_allocation_plan(config)
        fields = plan["groups_payload_bytes"]
        assert fields  # priced without touching CUDA
        state = _dynamics(monkeypatch, config)
        mass = config.grid.mass_shape
        for name in ("qa", "q3d"):
            assert state.fields[name].shape == (numq, *mass)
        assert state.held["qten"].shape == (numq, *mass)
        assert state.scalar_rk.qmag == config.profile.qmag
        # The initializer publishes six mass rows; a double-moment axis starts
        # its number concentrations at exact zero, as CM1's own does.
        assert np.array_equal(
            state.fields["q3d"][6:], np.zeros((numq - 6, *mass), dtype=np.float32)
        )


def test_the_number_rows_carry_cm1s_own_weno_epsilon() -> None:
    # src/solve2.F:1472: weps = qmag(n)*epsilon.  Mass rows and number rows do
    # not share a qmag, so they must not share a weps.
    qmag = RuntimeConfig(grid=_grid(), microphysics="morrison-mp10").profile.qmag
    mass_weps = moisture_weps_f64(0, qmag)
    number_weps = moisture_weps_f64(6, qmag)
    assert number_weps == pytest.approx(mass_weps * 1.0e8, rel=1e-6)
    # And the WSM6 answer is exactly what it was before qmag was a parameter.
    assert moisture_weps_f64(0) == mass_weps


def test_the_tile_inventory_adds_morrisons_own_retained_carriers() -> None:
    wsm6 = tiled_column_device_buffer_shapes(WSM6_SPEC, tile_nx=4, tile_ny=3, nk=8)
    morrison = tiled_column_device_buffer_shapes(
        MORRISON_SPEC, tile_nx=4, tile_ny=3, nk=8
    )
    assert wsm6["q3d"][0] == 6 and morrison["q3d"][0] == 10
    assert set(morrison) - set(wsm6) == {"effr", "nc"}
    assert set(wsm6) - set(morrison) == set()
    reserve = cartesian_wsm6_scratch_reserve(
        _runtime(microphysics="morrison-mp10", tile_nx=16, tile_ny=16)
    )
    assert reserve["component_id"] == "morrison-mp10"
    assert reserve["component_mass_scratch_volumes"] == 6


def test_the_closure_diffuses_every_scalar_the_component_transports() -> None:
    for component, numq in (
        ("wsm6-mp6", 6), ("morrison-mp10", 10), ("thompson-mp8", 8),
        ("thompson-aerosol-mp28", 11),
    ):
        config = _runtime(microphysics=component, tile_nx=16, tile_ny=16)
        plan = cartesian_closure_allocation_plan(config)
        tile = plan["held_groups"]["tile"]
        assert tile["qa"][0] == numq
        assert tile["qten"][0] == numq


# ------------------------------------------------------------- publication


def test_the_published_frame_names_every_scalar_with_its_own_unit() -> None:
    wsm6 = cartesian_history_profile(_runtime())
    morrison = cartesian_history_profile(_runtime(microphysics="morrison-mp10"))
    wsm6_names = [carrier.name for carrier in wsm6.carriers]
    morrison_names = [carrier.name for carrier in morrison.carriers]
    assert "nc3d" not in wsm6_names
    assert morrison_names == wsm6_names[:4] + ["nc3d"] + wsm6_names[4:]
    by_name = {carrier.name: carrier for carrier in morrison.carriers}
    assert by_name["q3d"].shape[0] == 6 and by_name["q3d"].units == "kg kg-1"
    assert by_name["nc3d"].shape[0] == 4 and by_name["nc3d"].units == "kg-1"
    assert "nci,ncs,ncr,ncg" in by_name["nc3d"].meaning
    # Nothing the component transports is dropped from the frame.
    published = by_name["q3d"].shape[0] + by_name["nc3d"].shape[0]
    assert published == MORRISON_SPEC.profile.numq
    assert morrison.payload_nbytes > wsm6.payload_nbytes


def test_a_frame_that_cannot_carry_every_scalar_refuses_and_names_the_omission() -> None:
    # 9d54616 (merged a492d33) lifted the one-payload cap from 2 GiB to
    # MAX_FORECAST_PAYLOAD_BYTES = 16 GiB, under which the 1000x1000x60
    # Morrison frame (~5.4 GB) now fits; 2000x2000x60 (~21 GB) still cannot,
    # while no single carrier exceeds the per-carrier cap.
    config = _runtime(
        grid=CartesianGridGeometry(2000, 2000, 60, 100.0, 100.0, 100.0),
        microphysics="morrison-mp10",
    )
    with pytest.raises(ValueError) as captured:
        cartesian_history_profile(config)
    message = str(captured.value)
    assert "morrison-mp10" in message
    assert "would require omitting" in message
    assert "nc3d=" in message
    assert "will not publish a frame that silently drops a carrier" in message


# ------------------------------------------------ the default stays the same


def _wsm6_default_signature(monkeypatch) -> dict[str, object]:
    """Everything about the default lane a component change could move."""

    config = _runtime()
    state = _dynamics(monkeypatch, config)
    plan = cartesian_dynamics_allocation_plan(config)
    history = cartesian_history_profile(config)
    spec = column_component_spec(config.microphysics)
    return {
        "numq": config.numq,
        "ihail": config.ihail,
        "arwen_option": config.dense_ice.arwen_option,
        "mp_physics": spec.mp_physics,
        "species": list(spec.species),
        "carrier_map": dict(spec.carrier_names),
        "persistence": list(spec.component_persistence),
        "effective_radius": list(spec.effective_radius_initial_microns),
        "qmag": list(config.profile.qmag),
        "weps": [moisture_weps_f64(index) for index in range(6)],
        "repaired_scalars": config.profile.repaired_scalars,
        "payload_bytes": plan["groups_payload_bytes"],
        "peak_bytes": plan["maximum_listed_phase_payload_bytes"],
        "tile_shapes": {
            name: list(shape)
            for name, shape in tiled_column_device_buffer_shapes(
                spec, tile_nx=4, tile_ny=3, nk=8
            ).items()
        },
        "history_carriers": [carrier.record() for carrier in history.carriers],
        "history_payload_bytes": history.payload_nbytes,
        "field_shapes": {
            name: list(value.shape) for name, value in sorted(state.fields.items())
        },
        "held_shapes": {
            name: list(value.shape) for name, value in sorted(state.held.items())
        },
        "audited_surface": engine_surface(),
    }


def engine_surface() -> dict[str, str]:
    from cm1core import engine_pin

    return dict(engine_pin.audited_surface(None, "wsm6-mp6"))


def test_the_default_wsm6_configuration_moved_only_its_dense_ice_branch(
    monkeypatch,
) -> None:
    """Everything measurable about the WSM6 lane except ihail is unchanged.

    The default ihail moved from 0 to CM1's own 1 (run/namelist.input:56),
    which changes the five constants of src/mp_wsm6.F:88-99 and therefore the
    words of every WSM6 run.  Nothing else about the lane moved with it: the
    scalar axis, carrier map, persistence inventory, effective radii, WENO
    epsilons, device inventory and audited ArWen surface are the same objects
    they were, and this test is what says so.
    """

    signature = _wsm6_default_signature(monkeypatch)
    assert signature["numq"] == 6
    assert signature["ihail"] == 1
    assert signature["arwen_option"] == "wsm6_hail_opt"
    assert signature["mp_physics"] == 6
    assert signature["species"] == ["qv", "qc", "qr", "qi", "qs", "qg"]
    assert signature["carrier_map"] == {
        name: name for name in ("qv", "qc", "qr", "qi", "qs", "qg")
    }
    assert signature["persistence"] == [
        "h_diabatic", "effc", "effi", "effs",
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_sr",
    ]
    assert signature["effective_radius"] == [
        ("effc", 2.49), ("effi", 4.99), ("effs", 9.99)
    ]
    assert signature["qmag"] == [1.0e-2] * 6
    assert signature["repaired_scalars"] == 6
    # The WENO epsilon of every WSM6 species, as CM1 forms it in binary32.
    assert signature["weps"] == [
        float(np.float32(np.float32(1.0e-2) * np.float32(1.0e-18)))
    ] * 6
    # The exact device inventory the WSM6 lane has always priced.
    assert signature["field_shapes"]["q3d"] == [6, 8, 11, 9]
    assert signature["field_shapes"]["qa"] == [6, 8, 11, 9]
    assert signature["held_shapes"]["qten"] == [6, 8, 11, 9]
    assert signature["tile_shapes"]["q3d"] == [6, 8, 3, 4]
    assert "nc" not in signature["tile_shapes"]
    assert "effr" not in signature["tile_shapes"]
    # The twelve fixed carriers, unchanged and first; 7183ce6 (int/output)
    # appended CM1's default &param9 products after them so their arena
    # offsets do not move.
    names = [carrier["name"] for carrier in signature["history_carriers"]]
    assert names[:12] == [
        "closure.tkea", "pp3d", "prs", "q3d", "rf", "rho",
        "terrain_msl", "th3d", "u3d", "v3d", "w3d", "zf",
    ]
    assert names[12:] == [
        "rain", "prate", "snownc", "graupelnc", "sr", "sws", "svs", "sps",
        "srs", "sgs", "sus", "shs", "dbz", "cref", "th", "zhval",
        "pi0", "th0", "prs0", "qv0", "u0", "v0",
    ]
    q3d = next(
        carrier for carrier in signature["history_carriers"] if carrier["name"] == "q3d"
    )
    assert q3d["meaning"] == "current qv,qc,qr,qi,qs,qg mixing ratios"
    assert q3d["units"] == "kg kg-1"
    assert q3d["shape"] == [6, 8, 11, 9]
    # Adding the Morrison surface never widens the WSM6 one.
    from cm1core import engine_pin

    assert signature["audited_surface"] == dict(engine_pin.AUDITED_WSM6_SURFACE)


def test_selecting_morrison_changes_that_signature_everywhere_it_should(
    monkeypatch,
) -> None:
    default = _wsm6_default_signature(monkeypatch)
    config = _runtime(microphysics="morrison-mp10", ihail=1)
    state = _dynamics(monkeypatch, config)
    spec = column_component_spec(config.microphysics)
    assert config.numq == 10 and default["numq"] == 6
    assert spec.mp_physics == 10 and default["mp_physics"] == 6
    assert config.dense_ice.arwen_option == "morr_rimed_ice"
    assert list(state.fields["q3d"].shape) == [10, 8, 11, 9]
    assert list(state.held["qten"].shape) == [10, 8, 11, 9]
    plan = cartesian_dynamics_allocation_plan(config)
    assert (
        plan["maximum_listed_phase_payload_bytes"] > default["peak_bytes"]
    ), "a ten-scalar axis must cost more device memory than a six-scalar one"


def test_the_tiled_launcher_binds_each_component_to_its_own_adapter() -> None:
    from cm1core.arwen_morrison_component import ArwenMorrisonComponent
    from cm1core.arwen_wsm6_component import ArwenWsm6Component
    from cm1core.cuda_gm1_tilestream_column import (
        TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256,
        TILESTREAM_RECORD_IDENTITY,
        _component_class,
    )

    assert _component_class("wsm6-mp6") is ArwenWsm6Component
    assert _component_class("morrison-mp10") is ArwenMorrisonComponent
    from cm1core.arwen_scheme_components import ArwenThompsonComponent

    assert _component_class("thompson-mp8") is ArwenThompsonComponent
    for unbound in ("p3-mp50", "wdm6-mp16"):
        with pytest.raises(ValueError, match="no CM1-owned tiled binding"):
            _component_class(unbound)
    # Each component's tile record has its own artifact schema, so a reader
    # cannot mistake a ten-scalar frame for a six-scalar one.
    assert (
        TILESTREAM_RECORD_IDENTITY["wsm6-mp6"]["schema"]
        != TILESTREAM_RECORD_IDENTITY["morrison-mp10"]["schema"]
    )
    # No Morrison component oracle has been measured, and the record says so
    # rather than borrowing WSM6's receipt.
    assert TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256["morrison-mp10"] is None
    assert TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256["wsm6-mp6"] is not None
    # Every bound component has its own record identity and its own oracle
    # answer; none borrows another scheme's.
    schemas = {
        identity["schema"] for identity in TILESTREAM_RECORD_IDENTITY.values()
    }
    assert len(schemas) == len(TILESTREAM_RECORD_IDENTITY)
    assert set(TILESTREAM_NATIVE_ORACLE_RECEIPT_SHA256) == set(
        TILESTREAM_RECORD_IDENTITY
    )


def test_the_component_registry_and_the_species_table_are_one_inventory() -> None:
    from cm1core.column_components import COLUMN_COMPONENT_SPECS
    from cm1core.moisture_species import MOISTURE_COMPONENT_IDS

    assert tuple(COLUMN_COMPONENT_SPECS) == MOISTURE_COMPONENT_IDS
    with pytest.raises(CM1Refusal) as captured:
        column_component_spec("goddard-lfo")
    assert captured.value.detail.code == "GM1-COLUMN-COMPONENT-UNSUPPORTED"
    with pytest.raises(CM1Refusal) as captured:
        column_component_spec("kessler-mp1")
    assert captured.value.detail.code == "GM1-COLUMN-COMPONENT-BLOCKED"


def test_the_checkpoint_transport_declares_a_matching_inventory() -> None:
    from cm1core.gm1_checkpoint import (
        GM1_CHECKPOINT_FIELDS,
        GM1_MORRISON_CHECKPOINT_FIELDS,
        gm1_checkpoint_fields,
    )

    assert gm1_checkpoint_fields() == GM1_CHECKPOINT_FIELDS
    assert gm1_checkpoint_fields("wsm6-mp6") == GM1_CHECKPOINT_FIELDS
    morrison = gm1_checkpoint_fields("morrison-mp10")
    assert morrison == GM1_MORRISON_CHECKPOINT_FIELDS
    assert set(morrison) - set(GM1_CHECKPOINT_FIELDS) == {
        "component.effr", "component.nc"
    }
    assert set(GM1_CHECKPOINT_FIELDS) - set(morrison) == set()
    # The transport's per-component inventory is the component's persistence
    # inventory, prefixed exactly as the artifact names it.
    persistence = {
        f"component.{name}"
        for name in MORRISON_SPEC.component_persistence
        if name != "h_diabatic"
    } | {"component.h_diabatic"}
    assert persistence <= set(morrison)
    # Thompson and WDM6 retain exactly WSM6's carriers; only the packed axis
    # beside them differs, and Rust's profile id carries that.
    assert gm1_checkpoint_fields("thompson-mp8") == GM1_CHECKPOINT_FIELDS
    assert gm1_checkpoint_fields("wdm6-mp16") == GM1_CHECKPOINT_FIELDS
    with pytest.raises(ValueError, match="declares no GM1 checkpoint carrier"):
        gm1_checkpoint_fields("p3-mp50")


@pytest.mark.parametrize(
    "document",
    [
        "official-supercell-product.toml",
        "official-supercell-native-context.toml",
        "dry-psolver2-bootstrap.toml",
        "real-atmosphere-100m-product.toml",
    ],
)
@pytest.mark.parametrize(("key", "value"), [("ihail", 1), ("ndcnst", 250.0)])
def test_cm1_microphysics_options_are_refused_outside_the_configurable_lane(
    document: str, key: str, value: object
) -> None:
    raw = _raw(ROOT / "examples" / document)
    raw["physics"][key] = value
    with pytest.raises(CM1Refusal) as captured:
        parse_mapping(raw)
    detail = captured.value.detail
    assert detail.code == "CM1-CFG-PHYSICS-OPTION-SCOPE"
    assert detail.option == f"physics.{key}"
    # A sealed lane must say what the declared value would have done, not just
    # that it is unknown here.
    assert "sealed profile" in detail.reason
