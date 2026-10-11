from __future__ import annotations

import os
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core.arwen_wsm6_component import (
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SURFACE_SCRATCH,
)
from cm1core.cuda_initialized_product import (
    CARRIER_NAMES,
    CUDA_INITIALIZED_PRODUCT_SCHEMA,
    CudaInitializedProduct,
    InitializedCarrierSpec,
    LoadedInitializedProduct,
)
from cm1core.errors import CM1Refusal
from cm1core.initialized_product_wsm6_transition import (
    INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
    INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
    PRODUCT_WSM6_RESTART_FIELDS,
    ProductWsm6Restart,
    _make_geopotential,
    initialize_product_wsm6_transition,
    load_durable_product_wsm6_restart,
    load_wsm6_geopotential_auxiliary,
)


NY = 2
NX = 3
NZ = 75
MANIFEST_SHA = "1" * 64
PAYLOAD_SHA = "2" * 64
ROOT = Path(__file__).resolve().parents[1]


class _DeviceArray(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


def _array(shape, fill=0.0):
    return np.full(shape, fill, dtype=np.float32).view(_DeviceArray)


def _field_shapes():
    volume = (NZ, NY, NX)
    result = {name: (1,) for name in CARRIER_NAMES}
    result.update(
        {
            "pi0": (NZ,),
            "th0": (NZ,),
            "pp3d": volume,
            "th3d": volume,
            "q3d": (6,) + volume,
            "prs": volume,
            "rho": volume,
            "rr": volume,
        }
    )
    return result


def _product(*, crop=(NY, NX), schema=CUDA_INITIALIZED_PRODUCT_SCHEMA):
    shapes = _field_shapes()
    fields = tuple((name, _array(shapes[name])) for name in CARRIER_NAMES)
    contracts = {
        "pi0": ("static-base-profile", ("pi0",)),
        "th0": ("static-base-profile", ("th0",)),
        "pp3d": (
            "stage-current-balanced-pressure-perturbation",
            ("pi0", "th0", "th3d", "qv", "qc", "qr", "qi", "qs", "qg", "zh"),
        ),
        "th3d": ("stage-current-theta-perturbation", ("th3d",)),
        "q3d": (
            "stage-current-packed-moisture",
            ("qv", "qc", "qr", "qi", "qs", "qg"),
        ),
        "prs": ("diagnostic-pressure-pa", ("pi0", "ppi")),
        "rho": (
            "diagnostic-moist-density",
            ("prs", "th0", "th3d", "pi0", "ppi", "qv"),
        ),
        "rr": ("reciprocal-diagnostic-moist-density", ("rho",)),
    }
    specs = tuple(
        InitializedCarrierSpec(
            name=name,
            shape=shapes[name],
            offset_nbytes=index * 4,
            nbytes=int(np.prod(shapes[name])) * 4,
            sha256=f"{index:064x}",
            role=contracts.get(name, ("test", (name,)))[0],
            source_arrays=contracts.get(name, ("test", (name,)))[1],
            minimum_f32_bits="00000000",
            maximum_f32_bits="00000000",
            finite_count=int(np.prod(shapes[name])),
            nonzero_count=0,
        )
        for index, name in enumerate(CARRIER_NAMES)
    )
    export = LoadedInitializedProduct(
        directory=None,
        payload_path=None,
        manifest_sha256=MANIFEST_SHA,
        manifest_nbytes=1,
        payload_sha256=PAYLOAD_SHA,
        payload_nbytes=1,
        source_artifact_sha256="3" * 64,
        source_contract_sha256="4" * 64,
        crop_origin_yx=(10, 10),
        crop_count_yx=crop,
        carriers=specs,
    )
    return CudaInitializedProduct(
        export=export,
        arena=object(),
        fields=fields,
        ready_event=object(),
        stream=SimpleNamespace(ptr=37),
        device_id=0,
        stream_pointer=37,
        event_pointer=1,
        runtime_identity=(("compute_capability", "86"),),
        audited_default_driver=True,
        schema=schema,
        claim_limit="test",
        _pinned_host=object(),
    )


def _geopotential(product, *, manifest_sha=MANIFEST_SHA, shape=(NZ + 1, NY, NX)):
    return _make_geopotential(
        value=_array(shape),
        source_product_manifest_sha256=manifest_sha,
        source_product_payload_sha256=product.export.payload_sha256,
        payload_sha256="5" * 64,
        manifest_sha256="6" * 64,
        receipt_sha256="7" * 64,
        device_id=0,
        stream_pointer=37,
        ready_event=object(),
        schema="rust-geopotential-test/v1",
        carrier_name="total_geopotential_zf",
        consumer_gravity_f32_bits="411cf5c3",
        zf_raw_identity=False,
        recovery_word_count=(NZ + 1) * NY * NX,
        recovery_mismatch_count=1,
        recovery_recovered_count=(NZ + 1) * NY * NX - 1,
        recovery_maximum_ulp_distance=1,
        recovery_maximum_absolute_error_m_f64_bits=struct.pack(">d", 1.0e-4).hex(),
        recovery_maximum_relative_error_f64_bits=struct.pack(">d", 1.0e-7).hex(),
        recovery_metric="test bounded recovery",
        recovery_comparison_statement_order="test source order",
    )


class _FakeComponent:
    component_id = "wsm6-mp6"

    def __init__(self, **values):
        self.__dict__.update(values)
        self._calls = 0
        self._persistence = {}

    def restore_persistent_arrays(self, carriers, arrays, *, completed_calls):
        self._calls = completed_calls
        self._persistence = dict(arrays)
        self.restored = (carriers, arrays)
        return object()

    def persistent_arrays(self):
        return self._persistence


class _FakeBridge:
    def __init__(self, *, component, **values):
        self.component = component
        self.__dict__.update(values)
        self._receipt = {"schema": "fake-bridge/v1"}
        self._calls = 0

    def restore_restart_arrays(self, arrays, *, completed_calls):
        self._calls = completed_calls
        self.component._calls = completed_calls
        self.component._persistence = {
            name: arrays[f"component.{name}"] for name in WSM6_COMPONENT_PERSISTENCE
        }
        self.restored_arrays = arrays

    def apply(self, carriers, *, final_rk, refl_10cm_due):
        assert final_rk is True
        assert type(refl_10cm_due) is bool
        self.carriers = carriers
        if not self.component._persistence:
            volume = carriers.pp3d.shape
            surface = volume[1:]
            self.component._persistence = {
                name: _array(
                    surface if name in WSM6_SURFACE_SCRATCH else volume,
                    fill=index + 1,
                )
                for index, name in enumerate(WSM6_COMPONENT_PERSISTENCE)
            }
        self.component._calls += 1
        self._calls += 1
        return SimpleNamespace(
            component_result=SimpleNamespace(
                call_index=self.component._calls,
                reflectivity_10cm=(
                    _array(carriers.pp3d.shape, fill=42.0)
                    if refl_10cm_due
                    else None
                ),
            ),
            call_index=self._calls,
            ready_event=object(),
        )

    def receipt_sha256(self):
        return "8" * 64

    def receipt_dict(self):
        return self._receipt


class _FakeCupy:
    float32 = np.float32
    copyto = staticmethod(np.copyto)
    zeros_like = staticmethod(lambda value: _array(value.shape))
    array = staticmethod(
        lambda value, dtype, order, copy: np.array(
            value, dtype=dtype, order=order, copy=copy
        ).view(_DeviceArray)
    )

    def __init__(self):
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: 0),
            get_current_stream=lambda: SimpleNamespace(ptr=37),
        )


def _patch_runtime(monkeypatch):
    import cm1core.initialized_product_wsm6_transition as transition

    monkeypatch.setattr(transition.importlib, "import_module", lambda name: _FakeCupy())
    monkeypatch.setattr(
        transition,
        "create_arwen_wsm6_component",
        lambda **values: _FakeComponent(**values),
    )
    monkeypatch.setattr(transition, "Eqtset2Wsm6Bridge", _FakeBridge)


def _compact_json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _auxiliary_fixture(tmp_path, *, raw_identity=False):
    import cm1core.initialized_product_wsm6_transition as transition

    source = _product().export
    product_directory = tmp_path / "product"
    product_directory.mkdir()
    product_receipt = b"independently hashed controlled product receipt\n"
    (product_directory / "receipt.json").write_bytes(product_receipt)
    product_receipt_sha = hashlib.sha256(product_receipt).hexdigest()
    zf_shape = (NZ + 1, NY, NX)
    zf_words = int(np.prod(zf_shape))
    zf_nbytes = zf_words * 4
    zf_index = CARRIER_NAMES.index("zf")
    specs = list(source.carriers)
    specs[zf_index] = replace(
        specs[zf_index],
        shape=zf_shape,
        offset_nbytes=4096,
        nbytes=zf_nbytes,
        sha256="a" * 64,
        finite_count=zf_words,
    )
    product = replace(
        source,
        directory=product_directory,
        payload_path=product_directory / "initialized-carriers.f32le",
        carriers=tuple(specs),
    )
    payload = bytes(zf_nbytes)
    payload_sha = hashlib.sha256(payload).hexdigest()
    recovery = {
        "comparison_statement_order": (
            transition.WSM6_GEOPOTENTIAL_RECOVERY_STATEMENT_ORDER
        ),
        "maximum_absolute_error_m_f64_bits": "0000000000000000",
        "maximum_relative_error_f64_bits": "0000000000000000",
        "maximum_ulp_distance": 0,
        "metric": transition.WSM6_GEOPOTENTIAL_RECOVERY_METRIC,
        "mismatch_count": 0,
        "recovered_count": zf_words,
        "word_count": zf_words,
    }
    binding = dict(transition.WSM6_GEOPOTENTIAL_CONSUMER_BINDING)
    binding["zf_raw_identity"] = raw_identity
    manifest = {
        "authority": transition.WSM6_GEOPOTENTIAL_AUTHORITY,
        "carrier": {
            "dtype": "<f4",
            "itemsize": 4,
            "name": transition.WSM6_GEOPOTENTIAL_CARRIER_NAME,
            "nbytes": zf_nbytes,
            "offset_nbytes": 0,
            "order": "C",
            "recovery": recovery,
            "role": "ArWen-WSM6-base-total-geopotential",
            "sha256": payload_sha,
            "shape": list(zf_shape),
            "source_carrier": {
                "name": "zf",
                "nbytes": zf_nbytes,
                "offset_nbytes": 4096,
                "sha256": "a" * 64,
            },
            "statistics": {
                "finite_count": zf_words,
                "maximum_f32_bits": "00000000",
                "minimum_f32_bits": "00000000",
                "nonzero_count": 0,
            },
            "units": "m2 s-2",
        },
        "carrier_abi": transition.WSM6_GEOPOTENTIAL_CARRIER_ABI,
        "claim_limit": transition.WSM6_GEOPOTENTIAL_CLAIM_LIMIT,
        "consumer_binding": binding,
        "dtype": "<f4",
        "itemsize": 4,
        "order": "C",
        "payload": {
            "file": transition.WSM6_GEOPOTENTIAL_PAYLOAD_NAME,
            "nbytes": zf_nbytes,
            "sha256": payload_sha,
        },
        "schema": transition.WSM6_GEOPOTENTIAL_SCHEMA,
        "source_product": {
            "carrier_abi": transition.INITIALIZED_PRODUCT_CARRIER_ABI,
            "manifest_sha256": product.manifest_sha256,
            "payload_sha256": product.payload_sha256,
            "receipt_sha256": product_receipt_sha,
            "source_artifact_sha256": product.source_artifact_sha256,
            "source_contract_sha256": product.source_contract_sha256,
        },
        "source_restart": {
            "artifact_nbytes": 16,
            "artifact_sha256": "b" * 64,
            "schema": "gpuwm-cm1.initialized-product-restart/v1",
        },
        "transform": transition.WSM6_GEOPOTENTIAL_TRANSFORM,
    }
    manifest_raw = _compact_json(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    receipt = {
        "carrier_abi": transition.WSM6_GEOPOTENTIAL_CARRIER_ABI,
        "carrier_count": 1,
        "claim_limit": transition.WSM6_GEOPOTENTIAL_CLAIM_LIMIT,
        "manifest_nbytes": len(manifest_raw),
        "manifest_sha256": manifest_sha,
        "payload_nbytes": zf_nbytes,
        "payload_sha256": payload_sha,
        "publication_contract": transition.WSM6_GEOPOTENTIAL_PUBLICATION_CONTRACT,
        "recovery": recovery,
        "schema": transition.WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA,
        "source_product_manifest_sha256": product.manifest_sha256,
        "source_product_payload_sha256": product.payload_sha256,
        "source_restart_artifact_sha256": "b" * 64,
    }
    receipt_raw = _compact_json(receipt)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    auxiliary = tmp_path / "auxiliary"
    auxiliary.mkdir()
    (auxiliary / transition.WSM6_GEOPOTENTIAL_PAYLOAD_NAME).write_bytes(payload)
    (auxiliary / transition.WSM6_GEOPOTENTIAL_MANIFEST_NAME).write_bytes(manifest_raw)
    (auxiliary / transition.WSM6_GEOPOTENTIAL_RECEIPT_NAME).write_bytes(receipt_raw)
    return product, auxiliary, manifest_sha, payload_sha, receipt_sha


def _durable_fixture(tmp_path):
    import cm1core.initialized_product_wsm6_transition as transition

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    product, auxiliary, manifest_sha, payload_sha, receipt_sha = _auxiliary_fixture(
        inputs
    )
    geopotential = load_wsm6_geopotential_auxiliary(
        auxiliary,
        product,
        expected_manifest_sha256=manifest_sha,
        expected_payload_sha256=payload_sha,
        expected_receipt_sha256=receipt_sha,
        expected_source_restart_artifact_sha256="b" * 64,
    )
    shapes = transition._restart_shapes(NY, NX)
    restored = tmp_path / "restored"
    restored.mkdir()
    arrays = []
    aggregate = hashlib.sha256()
    for name, role in zip(
        PRODUCT_WSM6_RESTART_FIELDS,
        transition.WSM6_DURABLE_ROLES,
        strict=True,
    ):
        shape = shapes[name]
        words = int(np.prod(shape))
        raw = bytes(words * 4)
        aggregate.update(raw)
        raw_sha = hashlib.sha256(raw).hexdigest()
        filename = f"{name}.f32le"
        (restored / filename).write_bytes(raw)
        arrays.append(
            {
                "dtype": "<f4",
                "file": filename,
                "itemsize": 4,
                "name": name,
                "nbytes": len(raw),
                "order": "C",
                "role": role,
                "shape": list(shape),
                "sha256": raw_sha,
                "species_order": ["qv", "qc", "qr", "qi", "qs", "qg"]
                if name == "state.q3d"
                else [],
                "statistics": {
                    "finite_count": words,
                    "maximum_f32_bits": "00000000",
                    "minimum_f32_bits": "00000000",
                    "nonzero_count": 0,
                },
            }
        )
    checkpoint_sha = "c" * 64
    durable_manifest = {
        "arrays": arrays,
        "capture_phase": transition.WSM6_DURABLE_CAPTURE_PHASE,
        "carrier_abi": transition.WSM6_DURABLE_CARRIER_ABI,
        "claim_limit": transition.WSM6_DURABLE_CLAIM_LIMIT,
        "completed_transitions": 1,
        "restore_phase": transition.WSM6_DURABLE_RESTORE_PHASE,
        "schema": transition.WSM6_DURABLE_RESTORE_MANIFEST_SCHEMA,
        "source": {
            "geopotential_manifest_sha256": geopotential.manifest_sha256,
            "geopotential_payload_sha256": geopotential.payload_sha256,
            "product_manifest_sha256": product.manifest_sha256,
            "product_payload_sha256": product.payload_sha256,
            "source_artifact_sha256": product.source_artifact_sha256,
            "source_contract_sha256": product.source_contract_sha256,
        },
        "source_checkpoint_artifact_sha256": checkpoint_sha,
    }
    manifest_raw = _compact_json(durable_manifest)
    (restored / "manifest.json").write_bytes(manifest_raw)
    return (
        product,
        geopotential,
        restored,
        hashlib.sha256(manifest_raw).hexdigest(),
        checkpoint_sha,
        aggregate.hexdigest(),
    )


def test_import_is_cupy_arwen_and_cuda_free():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    script = (
        "import sys; "
        "import cm1core.initialized_product_wsm6_transition as t; "
        "assert 'cupy' not in sys.modules; "
        "assert 'gpuwm' not in sys.modules; "
        "print(t.INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI


def test_strict_geopotential_loader_is_cuda_free_and_preserves_measured_bound(tmp_path):
    product, auxiliary, manifest_sha, payload_sha, receipt_sha = _auxiliary_fixture(
        tmp_path
    )
    cupy_before = "cupy" in sys.modules

    loaded = load_wsm6_geopotential_auxiliary(
        auxiliary,
        product,
        expected_manifest_sha256=manifest_sha,
        expected_payload_sha256=payload_sha,
        expected_receipt_sha256=receipt_sha,
        expected_source_restart_artifact_sha256="b" * 64,
    )

    assert ("cupy" in sys.modules) is cupy_before
    assert loaded.shape == (NZ + 1, NY, NX)
    assert loaded.recovery_word_count == (NZ + 1) * NY * NX
    assert loaded.recovery_mismatch_count == 0
    assert loaded.recovery_maximum_ulp_distance == 0


def test_strict_geopotential_loader_refuses_raw_identity_claim(tmp_path):
    product, auxiliary, manifest_sha, payload_sha, receipt_sha = _auxiliary_fixture(
        tmp_path, raw_identity=True
    )

    with pytest.raises(ValueError, match="consumer binding changed"):
        load_wsm6_geopotential_auxiliary(
            auxiliary,
            product,
            expected_manifest_sha256=manifest_sha,
            expected_payload_sha256=payload_sha,
            expected_receipt_sha256=receipt_sha,
            expected_source_restart_artifact_sha256="b" * 64,
        )


def test_strict_durable_restart_loader_is_cuda_and_numpy_free(tmp_path):
    product, geopotential, restored, manifest_sha, checkpoint_sha, payload_sha = (
        _durable_fixture(tmp_path)
    )
    cupy_before = "cupy" in sys.modules

    loaded = load_durable_product_wsm6_restart(
        restored,
        product,
        geopotential,
        expected_manifest_sha256=manifest_sha,
        expected_checkpoint_artifact_sha256=checkpoint_sha,
        expected_payload_sha256=payload_sha,
        expected_completed_transitions=1,
    )

    assert ("cupy" in sys.modules) is cupy_before
    assert tuple(carrier.name for carrier in loaded.carriers) == (
        PRODUCT_WSM6_RESTART_FIELDS
    )
    assert loaded.payload_sha256 == payload_sha
    assert loaded.completed_transitions == 1


def test_strict_durable_restart_loader_refuses_extra_inventory(tmp_path):
    product, geopotential, restored, manifest_sha, checkpoint_sha, payload_sha = (
        _durable_fixture(tmp_path)
    )
    (restored / "unexpected.bin").write_bytes(b"")

    with pytest.raises(ValueError, match="inventory mismatch"):
        load_durable_product_wsm6_restart(
            restored,
            product,
            geopotential,
            expected_manifest_sha256=manifest_sha,
            expected_checkpoint_artifact_sha256=checkpoint_sha,
            expected_payload_sha256=payload_sha,
            expected_completed_transitions=1,
        )


def test_cold_bind_uses_exact_49_carrier_views_and_complete_restart(monkeypatch):
    product = _product()
    _patch_runtime(monkeypatch)

    transition = initialize_product_wsm6_transition(
        product, _geopotential(product), allow_cropped_validation=True
    )
    result = transition.apply()

    assert result.completed_transitions == 1
    assert tuple(result.restart.arrays) == PRODUCT_WSM6_RESTART_FIELDS
    assert len(result.restart.arrays) == 21
    assert transition.carriers.pi0 is product.field("pi0")
    assert transition.carriers.th0 is product.field("th0")
    assert transition.carriers.p is product.field("prs")
    assert transition.carriers.rho is product.field("rho")
    assert transition.carriers.alt is product.field("rr")
    assert all(
        np.shares_memory(getattr(transition.carriers, name), product.field("q3d"))
        for name in ("qv", "qc", "qr", "qi", "qs", "qg")
    )
    assert result.receipt["binding"]["python_science_transforms"] is False
    assert result.receipt["restart"]["restore_before_physics_without_replay"] is True


def test_output_due_reflectivity_is_transient_and_six_species_bound(monkeypatch):
    product = _product()
    _patch_runtime(monkeypatch)
    transition = initialize_product_wsm6_transition(
        product, _geopotential(product), allow_cropped_validation=True
    )

    result = transition.apply(refl_10cm_due=True)

    assert result.reflectivity_10cm.shape == (NZ, NY, NX)
    assert "reflectivity_10cm" not in result.restart.arrays
    assert result.receipt["configuration"]["refl_10cm_due"] is True
    assert result.receipt["binding"]["reflectivity"] == {
        "consumer_species": ["qv", "qr", "qs", "qg"],
        "excluded_species_by_wsm6_authority": ["qc", "qi"],
        "six_species_order": ["qv", "qc", "qr", "qi", "qs", "qg"],
        "transient_not_restart_state": True,
    }
    assert result.receipt["red_gates"]["reflectivity_consumer"] is False


def test_restore_binds_held_and_component_state_and_continues_cadence(monkeypatch):
    source = _product()
    _patch_runtime(monkeypatch)
    first = initialize_product_wsm6_transition(
        source, _geopotential(source), allow_cropped_validation=True
    )
    first.apply()
    checkpoint = first.clone_restart()

    resumed_product = _product()
    resumed = initialize_product_wsm6_transition(
        resumed_product,
        _geopotential(resumed_product),
        allow_cropped_validation=True,
        restart=checkpoint,
    )
    staged = resumed.capture_restart()
    assert staged.completed_transitions == 1
    assert tuple(staged.arrays) == PRODUCT_WSM6_RESTART_FIELDS
    assert all(
        staged.arrays[f"component.{name}"] is checkpoint.arrays[f"component.{name}"]
        for name in WSM6_COMPONENT_PERSISTENCE
    )
    result = resumed.apply()

    assert result.completed_transitions == 2
    assert all(
        getattr(resumed.carriers, name) is checkpoint.arrays[f"held.{name}"]
        for name in ("qpten", "qtten", "qvten", "qcten")
    )
    assert all(
        resumed.bridge.component.persistent_arrays()[name]
        is checkpoint.arrays[f"component.{name}"]
        for name in WSM6_COMPONENT_PERSISTENCE
    )


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("crop", "CM1-PRODUCT-WSM6-CROP-VALIDATION-ONLY"),
        ("geopotential-source", "CM1-PRODUCT-WSM6-GEOPOTENTIAL-SOURCE"),
        ("geopotential-shape", "CM1-PRODUCT-WSM6-GEOPOTENTIAL-SHAPE"),
        ("nz", "CM1-PRODUCT-WSM6-GEOMETRY"),
    ],
)
def test_negative_controls_refuse_before_cupy(monkeypatch, case, code):
    import cm1core.initialized_product_wsm6_transition as transition_module

    product = _product()
    geopotential = _geopotential(product)
    allow_crop = True
    if case == "crop":
        allow_crop = False
    elif case == "geopotential-source":
        geopotential = _geopotential(product, manifest_sha="9" * 64)
    elif case == "geopotential-shape":
        geopotential = _geopotential(product, shape=(NZ, NY, NX))
    elif case == "nz":
        fields = list(product.fields)
        fields[CARRIER_NAMES.index("pi0")] = ("pi0", _array((74,)))
        product = replace(product, fields=tuple(fields))
    monkeypatch.setattr(
        transition_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("CuPy imported")),
    )

    with pytest.raises(CM1Refusal) as captured:
        initialize_product_wsm6_transition(
            product, geopotential, allow_cropped_validation=allow_crop
        )
    assert captured.value.detail.code == code


def test_restart_inventory_control_refuses_before_cupy(monkeypatch):
    import cm1core.initialized_product_wsm6_transition as transition_module

    product = _product()
    arrays = {name: _array((NZ, NY, NX)) for name in PRODUCT_WSM6_RESTART_FIELDS[:-1]}
    restart = ProductWsm6Restart(
        schema=INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
        transition_abi=INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
        source_product_manifest_sha256=MANIFEST_SHA,
        source_product_payload_sha256=PAYLOAD_SHA,
        completed_transitions=1,
        arrays=MappingProxyType(arrays),
    )
    monkeypatch.setattr(
        transition_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("CuPy imported")),
    )

    with pytest.raises(CM1Refusal) as captured:
        initialize_product_wsm6_transition(
            product,
            _geopotential(product),
            allow_cropped_validation=True,
            restart=restart,
        )
    assert captured.value.detail.code == "CM1-PRODUCT-WSM6-RESTART-INVENTORY"
