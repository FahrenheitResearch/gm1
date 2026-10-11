from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cm1core import engine_pin
from cm1core.arwen_wsm6_component import (
    ARWEN_KERNEL_MANIFEST_MODULE,
    ARWEN_PUBLIC_MICROPHYSICS_CALL,
    ARWEN_PUBLIC_MICROPHYSICS_MODULE,
    ARWEN_PUBLIC_REFLECTIVITY_CALL,
    ARWEN_PUBLIC_REFLECTIVITY_MODULE,
    ARWEN_RUN_CONFIG_MODULE,
    ARWEN_WSM6_COMPONENT_ABI,
    ARWEN_WSM6_COMPONENT_ID,
    ARWEN_WSM6_METADATA_SCHEMA,
    ARWEN_WSM6_RECEIPT_SCHEMA,
    ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI,
    ArwenWsm6Carriers,
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SPECIES,
    WSM6_SURFACE_SCRATCH,
    component_metadata,
    component_metadata_bytes,
    component_metadata_sha256,
    create_arwen_wsm6_component,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


class _FakeDtype:
    def __init__(self, name: str = "float32") -> None:
        self.name = name
        self.kind = "f" if name.startswith("float") else "i"
        self.itemsize = 4 if name in {"float32", "int32"} else 8

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeDtype) and (
            self.name,
            self.kind,
            self.itemsize,
        ) == (other.name, other.kind, other.itemsize)

    def __str__(self) -> str:
        return self.name


class _FakeArray:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: str = "float32",
        contiguous: bool = True,
        device_id: int = 0,
        fill_value: float = 0.0,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = _FakeDtype(dtype)
        self.flags = SimpleNamespace(c_contiguous=contiguous)
        self.device = SimpleNamespace(id=device_id)
        self.fill_value = fill_value
        self.touched = False


def _array(shape: tuple[int, ...], **changes: object) -> _FakeArray:
    return _FakeArray(shape, **changes)


def _carriers(*, nx: int = 4, ny: int = 3, nz: int = 8) -> ArwenWsm6Carriers:
    volume = (nz, ny, nx)
    interface = (nz + 1, ny, nx)
    return ArwenWsm6Carriers(
        thp=_array(volume),
        thb=_array((nz,)),
        p=_array(volume),
        alt=_array(volume),
        phb=_array((nz + 1,)),
        php=_array(interface),
        qv=_array(volume),
        qc=_array(volume),
        qr=_array(volume),
        qi=_array(volume),
        qs=_array(volume),
        qg=_array(volume),
    )


def _component(*, nx: int = 4, ny: int = 3, nz: int = 8, capacity_profile: str | None = None):
    return create_arwen_wsm6_component(
        component_id=ARWEN_WSM6_COMPONENT_ID,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=1000.0,
        dy_m=1000.0,
        dz_m=500.0,
        dt_s=6.0,
        capacity_profile=capacity_profile,
    )


def _persistence(*, nx: int = 4, ny: int = 3, nz: int = 8):
    volume = (nz, ny, nx)
    surface = (ny, nx)
    return {
        name: _array(surface if name in WSM6_SURFACE_SCRATCH else volume)
        for name in WSM6_COMPONENT_PERSISTENCE
    }


def test_import_and_metadata_are_cuda_and_arwen_free() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = (
        "import hashlib,json,sys; "
        "import cm1core.arwen_wsm6_component as c; "
        "assert 'gpuwm' not in sys.modules; "
        "assert 'cupy' not in sys.modules; "
        "b=c.component_metadata_bytes(); "
        "assert b.endswith(b'\\n') and b'\\r' not in b; "
        "assert c.component_metadata_sha256()==hashlib.sha256(b).hexdigest(); "
        "print(c.ARWEN_WSM6_COMPONENT_ID)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ARWEN_WSM6_COMPONENT_ID


def test_metadata_is_canonical_complete_and_stable() -> None:
    metadata = component_metadata()
    assert metadata["schema"] == ARWEN_WSM6_METADATA_SCHEMA
    assert metadata["component_id"] == ARWEN_WSM6_COMPONENT_ID
    assert metadata["component_abi"] == ARWEN_WSM6_COMPONENT_ABI
    assert metadata["execution"]["entry_point"] == ARWEN_PUBLIC_MICROPHYSICS_CALL
    assert metadata["execution"]["reflectivity"] == {
        "producer": "public ArWen WSM6 output-due diagnostic",
        "consumer": ARWEN_PUBLIC_REFLECTIVITY_CALL,
        "consumer_abi": ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI,
        "shape": ["nz", "ny", "nx"],
        "dtype": "float32",
        "units": "dBZ",
        "lifetime": "consumed exactly once before apply returns; not restart state",
        "species_mapping": ["qv", "qr", "qs", "qg"],
        "absent_species": ["qc", "qi"],
    }
    # CM1 param.F:3697-3706 carries ihail into WSM6 as hail_opt, and CM1 ships
    # ihail=1 (run/namelist.input:56).  An undeclared ihail now binds that,
    # not the graupel branch this lane executed before the default moved.
    assert metadata["execution"]["wsm6_hail_opt"] == 1
    assert metadata["dense_ice"]["ihail"] == 1
    assert metadata["dense_ice"]["category"] == "hail"
    assert metadata["dense_ice"]["arwen_run_config_option"] == "wsm6_hail_opt"
    assert component_metadata(ihail=0)["execution"]["wsm6_hail_opt"] == 0
    assert component_metadata(ihail=0)["dense_ice"]["category"] == "graupel"
    assert metadata["moisture_species"]["numq"] == 6
    assert metadata["carriers"]["cm1_to_arwen_carrier"] == {
        name: name for name in WSM6_SPECIES
    }
    assert metadata["carriers"]["species"] == list(WSM6_SPECIES)
    assert metadata["persistence"]["component_owned"] == list(
        WSM6_COMPONENT_PERSISTENCE
    )
    assert component_metadata_bytes() == (
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert (
        component_metadata_sha256()
        == hashlib.sha256(component_metadata_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"component_id": "5"}, "CM1-COMPONENT-ID"),
        ({"nx": True}, "CM1-WSM6-DIMENSION"),
        ({"nz": 1}, "CM1-WSM6-DIMENSION"),
        ({"nz": 81}, "CM1-WSM6-VERTICAL-BOUNDS"),
        ({"dt_s": 0.0}, "CM1-WSM6-SCALAR"),
        ({"dz_m": float("nan")}, "CM1-WSM6-SCALAR"),
    ],
)
def test_factory_refuses_before_engine_or_cuda_import(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    code: str,
) -> None:
    import cm1core.arwen_column_component as adapter

    monkeypatch.setattr(
        adapter.engine_pin,
        "inspect",
        lambda **_: (_ for _ in ()).throw(AssertionError("engine inspected")),
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"imported {name}")),
    )
    values: dict[str, object] = {
        "component_id": ARWEN_WSM6_COMPONENT_ID,
        "nx": 4,
        "ny": 3,
        "nz": 8,
        "dx_m": 1000.0,
        "dy_m": 1000.0,
        "dz_m": 500.0,
        "dt_s": 6.0,
    }
    values.update(changes)
    with pytest.raises(CM1Refusal) as captured:
        create_arwen_wsm6_component(**values)
    assert captured.value.detail.code == code


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda c: replace(c, p=_array((8, 3, 5))),
            "CM1-WSM6-CARRIER-SHAPE",
        ),
        (
            lambda c: replace(c, qi=_array((8, 3, 4), dtype="float64")),
            "CM1-WSM6-CARRIER-DTYPE",
        ),
        (
            lambda c: replace(c, qg=_array((8, 3, 4), contiguous=False)),
            "CM1-WSM6-CARRIER-STRIDES",
        ),
        (
            lambda c: replace(c, qg=c.qv),
            "CM1-WSM6-CARRIER-ALIAS",
        ),
    ],
)
def test_carrier_contract_refuses_before_engine_or_cuda_import(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    code: str,
) -> None:
    import cm1core.arwen_column_component as adapter

    component = _component()
    monkeypatch.setattr(
        adapter.engine_pin,
        "inspect",
        lambda **_: (_ for _ in ()).throw(AssertionError("engine inspected")),
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"imported {name}")),
    )
    with pytest.raises(CM1Refusal) as captured:
        component.apply(mutate(_carriers()))
    assert captured.value.detail.code == code


def test_non_boolean_reflectivity_refuses_before_engine_or_cuda_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    component = _component()
    monkeypatch.setattr(
        adapter.engine_pin,
        "inspect",
        lambda **_: (_ for _ in ()).throw(AssertionError("engine inspected")),
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"imported {name}")),
    )
    with pytest.raises(CM1Refusal) as captured:
        component.apply(_carriers(), refl_10cm_due=1)
    assert captured.value.detail.code == "CM1-WSM6-REFLECTIVITY-FLAG"


def test_engine_drift_refuses_before_cuda_or_arwen_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    # The admitted version, so the ONLY reason this inspection is
    # inadmissible is the moved file.  A stale literal here would also fail
    # the version check and stop proving that byte drift alone refuses.
    inspection = engine_pin.EngineInspection(
        engine_pin.GPUWM_MEASURED_VERSION,
        Path("/drifted"),
        (),
        ("gpuwm/core/microphysics.py",),
        (),
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


class _FakeEvent:
    def __init__(self, *, disable_timing: bool) -> None:
        self.disable_timing = disable_timing
        self.recorded_stream = None
        self.synchronized = False

    def record(self, stream: object) -> None:
        self.recorded_stream = stream


class _FakeRunConfig:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


def _fake_runtime_modules():
    cupy = ModuleType("cupy")
    cupy.ndarray = _FakeArray
    cupy.float32 = _FakeDtype("float32")
    cupy.dtype = lambda value: value if isinstance(value, _FakeDtype) else _FakeDtype()
    cupy.zeros = lambda shape, dtype: _FakeArray(tuple(shape))
    cupy.full = lambda shape, fill, dtype: _FakeArray(
        tuple(shape), fill_value=float(fill)
    )
    stream = SimpleNamespace(ptr=0xA11CE)
    cupy.cuda = SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0),
        get_current_stream=lambda: stream,
        Event=_FakeEvent,
    )

    config = ModuleType(ARWEN_RUN_CONFIG_MODULE)
    config.RunConfig = _FakeRunConfig

    calls: list[tuple[object, object, float, bool]] = []
    microphysics = ModuleType(ARWEN_PUBLIC_MICROPHYSICS_MODULE)

    def public_apply(state, cfg, dt, *, refl_10cm_due=False):
        calls.append((state, cfg, dt, refl_10cm_due))
        state.qv.touched = True
        return SimpleNamespace(
            rainnc=state.existing_scratch("mp_rainnc"),
            rainncv=state.existing_scratch("mp_rainncv"),
            snownc=state.existing_scratch("mp_snownc"),
            snowncv=state.existing_scratch("mp_snowncv"),
            graupelnc=state.existing_scratch("mp_graupelnc"),
            graupelncv=state.existing_scratch("mp_graupelncv"),
            sr=state.existing_scratch("mp_sr"),
        )

    public_apply.__module__ = ARWEN_PUBLIC_MICROPHYSICS_MODULE
    microphysics.apply = public_apply

    manifest = ModuleType(ARWEN_KERNEL_MANIFEST_MODULE)
    manifest_snapshots = [
        {},
        {
            "wsm6": {
                "source_sha256": "a" * 64,
                "options": ["--std=c++14"],
                "compiled_image": {"status": "unavailable", "reason": "fake"},
            }
        },
    ]

    def kernel_manifest():
        return manifest_snapshots.pop(0)

    manifest.kernel_manifest = kernel_manifest
    reflectivity = ModuleType(ARWEN_PUBLIC_REFLECTIVITY_MODULE)

    def consume_refl_10cm(state):
        value = state.physics.refl_10cm
        if value is None:
            raise RuntimeError("missing fake reflectivity stash")
        state.physics.refl_10cm = None
        return value

    consume_refl_10cm.__module__ = ARWEN_PUBLIC_REFLECTIVITY_MODULE
    reflectivity.consume_refl_10cm = consume_refl_10cm
    original_apply = microphysics.apply

    def apply_with_reflectivity(state, cfg, dt, *, refl_10cm_due=False):
        result = original_apply(
            state, cfg, dt, refl_10cm_due=refl_10cm_due
        )
        if refl_10cm_due:
            value = state.scratch(state.p.shape, "refl_10cm")
            state.physics.refl_10cm = value
        return result

    apply_with_reflectivity.__module__ = ARWEN_PUBLIC_MICROPHYSICS_MODULE
    microphysics.apply = apply_with_reflectivity
    return cupy, config, microphysics, manifest, reflectivity, calls, stream


@pytest.mark.parametrize("nz,capacity_profile", [
    (8, None), (237, "gm1.arwen-wsm6-column-capacity-256/v1"),
])
def test_public_apply_binds_zero_copy_state_and_persistence_without_sync(
    monkeypatch: pytest.MonkeyPatch,
    nz: int,
    capacity_profile: str | None,
) -> None:
    import cm1core.arwen_column_component as adapter

    cupy, config, microphysics, manifest, reflectivity, calls, stream = _fake_runtime_modules()
    modules = {
        "cupy": cupy,
        ARWEN_RUN_CONFIG_MODULE: config,
        ARWEN_PUBLIC_MICROPHYSICS_MODULE: microphysics,
        ARWEN_KERNEL_MANIFEST_MODULE: manifest,
        ARWEN_PUBLIC_REFLECTIVITY_MODULE: reflectivity,
    }
    inspection = engine_pin.EngineInspection(
        engine_pin.GPUWM_MEASURED_VERSION,
        Path("C:/audited"),
        tuple(sorted(engine_pin.AUDITED_WSM6_SURFACE)),
        (),
        (),
    )
    def inspect(**kwargs):
        # The adapter always names the component it is about to execute, so
        # a Morrison-only file can never be admitted for a WSM6 run.
        assert kwargs == {
            "capacity_profile": capacity_profile,
            "component_id": ARWEN_WSM6_COMPONENT_ID,
        }
        return inspection

    monkeypatch.setattr(adapter.engine_pin, "inspect", inspect)
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)

    carriers = _carriers(nz=nz)
    component = _component(nz=nz, capacity_profile=capacity_profile)
    result = component.apply(carriers)

    assert imported == [
        "cupy",
        ARWEN_RUN_CONFIG_MODULE,
        ARWEN_PUBLIC_MICROPHYSICS_MODULE,
        ARWEN_KERNEL_MANIFEST_MODULE,
    ]
    assert len(calls) == 1
    state, cfg, dt, refl_due = calls[0]
    assert component.state is state
    assert dt == 6.0
    assert refl_due is False
    assert cfg.mp_physics == 6
    # CM1 param.F:3697-3706 with run/namelist.input:56: the undeclared value
    # this component is constructed with is CM1's own ihail=1.
    assert cfg.wsm6_hail_opt == 1
    assert cfg.moist is True
    for name in ("thp", "thb", "p", "alt", "phb", "php") + WSM6_SPECIES:
        assert getattr(state, name) is getattr(carriers, name)
    assert carriers.qv.touched is True

    persistence = component.persistent_arrays()
    assert tuple(persistence) == WSM6_COMPONENT_PERSISTENCE
    assert persistence["effc"].fill_value == 2.49
    assert persistence["effi"].fill_value == 4.99
    assert persistence["effs"].fill_value == 9.99
    assert all(
        component.existing_scratch(slot) is persistence[slot]
        for slot in WSM6_SURFACE_SCRATCH
    )

    assert result.call_index == 1
    assert result.reflectivity_10cm is None
    assert result.stream_pointer == stream.ptr
    assert result.ready_event.recorded_stream is stream
    assert result.ready_event.synchronized is False
    assert result.h_diabatic is persistence["h_diabatic"]
    receipt = component.receipt_dict()
    assert receipt["schema"] == ARWEN_WSM6_RECEIPT_SCHEMA
    assert receipt["binding"]["zero_copy_object_identity"] is True
    assert receipt["kernel_manifest_changed"] == receipt["kernel_manifest_after"]
    assert receipt["synchronization"] == {
        "explicit_synchronize": False,
        "host_copy": False,
        "ready_event_recorded": True,
        "stream_pointer": stream.ptr,
    }
    assert component.receipt_bytes() == (
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert (
        component.receipt_sha256()
        == hashlib.sha256(component.receipt_bytes()).hexdigest()
    )


def test_public_reflectivity_is_consumed_once_without_entering_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    cupy, config, microphysics, manifest, reflectivity, calls, _stream = (
        _fake_runtime_modules()
    )
    modules = {
        "cupy": cupy,
        ARWEN_RUN_CONFIG_MODULE: config,
        ARWEN_PUBLIC_MICROPHYSICS_MODULE: microphysics,
        ARWEN_KERNEL_MANIFEST_MODULE: manifest,
        ARWEN_PUBLIC_REFLECTIVITY_MODULE: reflectivity,
    }
    inspection = engine_pin.EngineInspection(
        engine_pin.GPUWM_MEASURED_VERSION,
        Path("C:/audited"),
        tuple(sorted(engine_pin.AUDITED_WSM6_SURFACE)),
        (),
        (),
    )
    monkeypatch.setattr(adapter.engine_pin, "inspect", lambda **_: inspection)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: modules[name])

    component = _component()
    result = component.apply(_carriers(), refl_10cm_due=True)

    assert calls[0][3] is True
    assert result.reflectivity_10cm is component.existing_scratch("refl_10cm")
    assert component.state.physics.refl_10cm is None
    assert "refl_10cm" not in component.persistent_arrays()
    receipt = component.receipt_dict()
    assert receipt["reflectivity"] == {
        "consumer_abi": ARWEN_WSM6_REFLECTIVITY_CONSUMER_ABI,
        "consumer": ARWEN_PUBLIC_REFLECTIVITY_CALL,
        "due": True,
        "produced": True,
        "consumed_once": True,
        "restart_state": False,
        "dtype": "float32",
        "shape": [8, 3, 4],
        "units": "dBZ",
        "species_order": list(WSM6_SPECIES),
        "consumer_species": ["qv", "qr", "qs", "qg"],
        "excluded_species_by_scheme_authority": ["qc", "qi"],
    }


def test_restore_persistence_binds_exact_arrays_without_physics_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    cupy, config, microphysics, manifest, reflectivity, calls, stream = _fake_runtime_modules()
    modules = {
        "cupy": cupy,
        ARWEN_RUN_CONFIG_MODULE: config,
        ARWEN_PUBLIC_MICROPHYSICS_MODULE: microphysics,
        ARWEN_KERNEL_MANIFEST_MODULE: manifest,
        ARWEN_PUBLIC_REFLECTIVITY_MODULE: reflectivity,
    }
    inspection = engine_pin.EngineInspection(
        engine_pin.GPUWM_MEASURED_VERSION,
        Path("C:/audited"),
        tuple(sorted(engine_pin.AUDITED_WSM6_SURFACE)),
        (),
        (),
    )
    monkeypatch.setattr(adapter.engine_pin, "inspect", lambda **_: inspection)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: modules[name])

    carriers = _carriers()
    persistence = _persistence()
    component = _component()
    ready = component.restore_persistent_arrays(
        carriers, persistence, completed_calls=7
    )

    assert calls == []
    assert ready.recorded_stream is stream
    assert tuple(component.persistent_arrays()) == WSM6_COMPONENT_PERSISTENCE
    assert all(
        component.persistent_arrays()[name] is persistence[name]
        for name in WSM6_COMPONENT_PERSISTENCE
    )

    result = component.apply(carriers)
    assert len(calls) == 1
    assert result.call_index == 8
    assert all(
        component.persistent_arrays()[name] is persistence[name]
        for name in WSM6_COMPONENT_PERSISTENCE
    )
    receipt = component.receipt_dict()
    assert receipt["persistence"]["restored_before_apply"] is True
    assert receipt["persistence"]["restored_completed_calls"] == 7
    assert receipt["persistence"]["restore_zero_copy_object_identity"] is True


def test_restore_persistence_inventory_refuses_before_engine_or_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    component = _component()
    persistence = _persistence()
    persistence.pop("mp_sr")
    monkeypatch.setattr(
        adapter.engine_pin,
        "inspect",
        lambda **_: (_ for _ in ()).throw(AssertionError("engine inspected")),
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"imported {name}")),
    )
    with pytest.raises(CM1Refusal) as captured:
        component.restore_persistent_arrays(
            _carriers(), persistence, completed_calls=0
        )
    assert captured.value.detail.code == "CM1-WSM6-RESTART-INVENTORY"


def test_restore_persistence_refuses_alias_with_read_only_live_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cm1core.arwen_column_component as adapter

    cupy, config, microphysics, manifest, reflectivity, _calls, _stream = _fake_runtime_modules()
    modules = {
        "cupy": cupy,
        ARWEN_RUN_CONFIG_MODULE: config,
        ARWEN_PUBLIC_MICROPHYSICS_MODULE: microphysics,
        ARWEN_KERNEL_MANIFEST_MODULE: manifest,
        ARWEN_PUBLIC_REFLECTIVITY_MODULE: reflectivity,
    }
    inspection = engine_pin.EngineInspection(
        engine_pin.GPUWM_MEASURED_VERSION,
        Path("C:/audited"),
        tuple(sorted(engine_pin.AUDITED_WSM6_SURFACE)),
        (),
        (),
    )
    monkeypatch.setattr(adapter.engine_pin, "inspect", lambda **_: inspection)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: modules[name])
    carriers = _carriers()
    persistence = _persistence()
    persistence["h_diabatic"] = carriers.p

    with pytest.raises(CM1Refusal) as captured:
        _component().restore_persistent_arrays(
            carriers, persistence, completed_calls=0
        )

    assert captured.value.detail.code == "CM1-WSM6-RESTART-CARRIER-ALIAS"


def test_source_calls_only_public_arwen_microphysics_surface() -> None:
    source = (ROOT / "src" / "cm1core" / "arwen_wsm6_component.py").read_text(
        encoding="utf-8"
    )
    assert ARWEN_PUBLIC_MICROPHYSICS_MODULE in source
    assert "gpuwm.core.wsm6" not in source
    assert "launch_wsm6" not in source
    assert 'extern "C"' not in source
    assert "__global__" not in source
