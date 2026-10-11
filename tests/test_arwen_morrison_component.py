"""CM1 ptype=5 bound to ArWen's public Morrison dispatch, without CUDA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cm1core import engine_pin
from cm1core.arwen_morrison_component import (
    ARWEN_MORRISON_COMPONENT_ID,
    MORRISON_CARRIER_NAMES,
    MORRISON_COMPONENT_PERSISTENCE,
    MORRISON_NDCNST_CM3,
    MORRISON_SPEC,
    ArwenMorrisonCarriers,
    component_metadata,
    component_metadata_bytes,
    component_metadata_sha256,
    create_arwen_morrison_component,
    require_ndcnst,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


class _FakeDtype:
    name = "float32"
    kind = "f"
    itemsize = 4

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDtype)

    def __str__(self) -> str:
        return "float32"


class _FakeFlags:
    c_contiguous = True


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.dtype = _FakeDtype()
        self.flags = _FakeFlags()


def _carriers(nx: int = 4, ny: int = 3, nz: int = 8) -> ArwenMorrisonCarriers:
    volume = (nz, ny, nx)
    return ArwenMorrisonCarriers(
        thp=_FakeArray(volume),
        thb=_FakeArray((nz,)),
        p=_FakeArray(volume),
        alt=_FakeArray(volume),
        phb=_FakeArray((nz + 1,)),
        php=_FakeArray((nz + 1, ny, nx)),
        **{name: _FakeArray(volume) for name in MORRISON_SPEC.species},
    )


def _component(**changes: object):
    values: dict[str, object] = {
        "component_id": ARWEN_MORRISON_COMPONENT_ID,
        "nx": 4,
        "ny": 3,
        "nz": 8,
        "dx_m": 1000.0,
        "dy_m": 1000.0,
        "dz_m": 500.0,
        "dt_s": 0.5,
    }
    values.update(changes)
    return create_arwen_morrison_component(**values)  # type: ignore[arg-type]


def test_import_is_cuda_and_arwen_cold() -> None:
    source = (ROOT / "src" / "cm1core" / "arwen_morrison_component.py").read_text(
        encoding="utf-8"
    )
    assert "import cupy" not in source
    assert "import gpuwm" not in source
    assert "gpuwm.core.morrison" not in source


def test_metadata_declares_the_ten_scalar_axis_and_the_name_map() -> None:
    metadata = component_metadata()
    assert metadata["component_id"] == ARWEN_MORRISON_COMPONENT_ID
    assert metadata["execution"]["mp_physics"] == 10
    assert metadata["execution"]["inum"] == 1
    assert metadata["execution"]["ndcnst_cm3"] == MORRISON_NDCNST_CM3
    assert metadata["carriers"]["species"] == [
        "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"
    ]
    # src/mp_driver.F:546-558 against src/morrison.F:687: CM1's seventh scalar
    # is cloud-ice number and ArWen calls it ni; its ninth is rain number.
    assert metadata["carriers"]["cm1_to_arwen_carrier"] == {
        "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
        "nci": "ni", "ncs": "ns", "ncr": "nr", "ncg": "ng",
    }
    assert metadata["moisture_species"]["numq"] == 10
    assert metadata["persistence"]["component_owned"] == list(
        MORRISON_COMPONENT_PERSISTENCE
    )
    assert "nc" in MORRISON_COMPONENT_PERSISTENCE
    assert "effr" in MORRISON_COMPONENT_PERSISTENCE
    assert component_metadata_bytes() == (
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert component_metadata_sha256() == __import__("hashlib").sha256(
        component_metadata_bytes()
    ).hexdigest()


def test_ihail_reaches_the_morrison_run_config_option() -> None:
    assert component_metadata(ihail=0)["execution"]["morr_rimed_ice"] == 0
    assert component_metadata(ihail=1)["execution"]["morr_rimed_ice"] == 1
    assert component_metadata(ihail=1)["dense_ice"]["selected_constants"] == {
        "ag": 114.5, "bg": 0.5, "rhog_kg_m3": 900.0,
        "cg": 900.0 * 3.141592653589793 / 6.0,
    }
    component = _component(ihail=1)
    assert component.ihail == 1
    assert component.dense_ice.arwen_option == "morr_rimed_ice"


def test_the_morrison_column_capacity_is_its_own_and_the_wsm6_overlay_is_refused() -> None:
    from cm1core.engine_capacity import WSM6_256_CAPACITY_PROFILE

    assert _component(nz=200).nz == 200
    with pytest.raises(CM1Refusal) as captured:
        _component(nz=257)
    assert captured.value.detail.code == "CM1-MORR-VERTICAL-BOUNDS"
    with pytest.raises(CM1Refusal) as captured:
        _component(capacity_profile=WSM6_256_CAPACITY_PROFILE)
    assert captured.value.detail.code == "CM1-MORR-CAPACITY-PROFILE"


def test_ndcnst_is_refused_by_value_naming_the_literal_it_protects() -> None:
    assert require_ndcnst(250.0) == 250.0
    with pytest.raises(CM1Refusal) as captured:
        require_ndcnst(100.0)
    detail = captured.value.detail
    assert detail.code == "GM1-MORRISON-NDCNST"
    assert "morrison.cu:124,:1310" in detail.reason


def test_the_audited_estate_names_the_two_morrison_files() -> None:
    extra = engine_pin.component_extra_surface("morrison-mp10")
    assert set(extra) == {"gpuwm/core/morrison.py", "gpuwm/core/kernels/morrison.cu"}
    # The WSM6 surface is untouched: the default answer is the historical one.
    assert engine_pin.component_extra_surface("wsm6-mp6") == {}
    assert engine_pin.component_extra_surface(None) == {}
    assert engine_pin.audited_surface() == engine_pin.AUDITED_WSM6_SURFACE
    morrison = engine_pin.audited_surface(None, "morrison-mp10")
    assert len(morrison) == len(engine_pin.AUDITED_WSM6_SURFACE) + 2
    assert all(
        morrison[name] == digest
        for name, digest in engine_pin.AUDITED_WSM6_SURFACE.items()
    )
    # Every bound component names its own extra files and no other's.
    thompson = engine_pin.component_extra_surface("thompson-mp8")
    wdm6 = engine_pin.component_extra_surface("wdm6-mp16")
    assert set(thompson).isdisjoint(extra)
    assert set(wdm6).isdisjoint(extra)
    assert set(thompson).isdisjoint(engine_pin.AUDITED_WSM6_SURFACE)
    assert set(wdm6).isdisjoint(engine_pin.AUDITED_WSM6_SURFACE)
    # A scheme with no admitted public surface is still refused by name.
    with pytest.raises(ValueError, match="never audited"):
        engine_pin.component_extra_surface("p3-mp50")


def test_carrier_contract_refuses_a_wsm6_carrier_set() -> None:
    from cm1core.arwen_wsm6_component import ArwenWsm6Carriers

    component = _component()
    volume = (8, 3, 4)
    wsm6 = ArwenWsm6Carriers(
        thp=_FakeArray(volume), thb=_FakeArray((8,)), p=_FakeArray(volume),
        alt=_FakeArray(volume), phb=_FakeArray((9,)),
        php=_FakeArray((9, 3, 4)),
        **{name: _FakeArray(volume) for name in ("qv", "qc", "qr", "qi", "qs", "qg")},
    )
    with pytest.raises(CM1Refusal) as captured:
        component.apply(wsm6)
    assert captured.value.detail.code == "CM1-MORR-CARRIER-SET"


def test_engine_drift_refuses_before_cuda_or_arwen_import(monkeypatch) -> None:
    import cm1core.arwen_column_component as adapter

    inspection = engine_pin.EngineInspection(
        "2.5.8", Path("drifted"), (), ("gpuwm/core/morrison.py",), (),
    )
    monkeypatch.setattr(adapter.engine_pin, "inspect", lambda **_: inspection)
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"imported {name}")),
    )
    with pytest.raises(CM1Refusal) as captured:
        _component().apply(_carriers())
    assert captured.value.detail.code == "CM1-ARWEN-ENGINE-DRIFT"


def test_cloud_droplet_number_is_refreshed_from_cm1s_prognostic_density() -> None:
    """src/morrison.F:1830-1833 with :1673, before the size distributions."""

    recorded: dict[str, object] = {}

    class _Cupy:
        @staticmethod
        def float32(value: float) -> float:
            return float(value)

        @staticmethod
        def multiply(left: object, right: object, *, out: object) -> None:
            recorded["left"] = left
            recorded["right"] = right
            recorded["out"] = out

    component = _component()
    carriers = _carriers()
    state = SimpleNamespace(nc=object())
    component.prepare_component_owned_carriers(_Cupy, carriers, state)
    assert recorded["left"] is carriers.alt
    assert recorded["out"] is state.nc
    # ndcnst is cm-3; CM1 converts with *1.e6 and divides by rho, and alt is
    # CM1's own 1/rho, so the multiply is the same number.
    assert recorded["right"] == MORRISON_NDCNST_CM3 * 1.0e6


def test_the_carrier_name_map_is_a_bijection_onto_arwens_keywords() -> None:
    assert sorted(MORRISON_CARRIER_NAMES) == sorted(MORRISON_SPEC.species)
    targets = list(MORRISON_CARRIER_NAMES.values())
    assert len(set(targets)) == len(targets)
    assert set(targets) == {"qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr", "ng"}
