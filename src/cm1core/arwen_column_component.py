"""CM1-owned adapter for ArWen's public column-microphysics component surface.

This module is intentionally import-light.  Importing it does not import
``gpuwm`` or CuPy, inspect a CUDA device, or create a CUDA context.  The
factory and the first half of :meth:`ArwenColumnComponent.apply` validate the
semantic component, dimensions, timestep, carrier shapes, binary32 dtypes,
and C contiguity before either numerical runtime is imported.

The sole physics execution edge is ``gpuwm.core.microphysics.apply``.  CM1
owns the adapter and carrier binding; ArWen owns the microphysics
preparation, scheme dispatch, latent-heating finish, and compiled kernel
lifecycle behind that public function.  The adapter never imports a
scheme-private module and never copies an ArWen kernel.

One :class:`ColumnComponentSpec` per admitted CM1 ``ptype`` is the whole of
what differs between components: the transported scalar axis, the extra
carriers a double-moment scheme adds, the retained persistence inventory, and
the single ``RunConfig`` keyword CM1's ``ihail`` travels in.  The bindings
live in :mod:`cm1core.arwen_wsm6_component` (CM1 ``ptype=7``) and
:mod:`cm1core.arwen_morrison_component` (CM1 ``ptype=5``).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, NoReturn

from . import engine_pin
from .component_validation import component_validation
from .dense_ice import DEFAULT_IHAIL, dense_ice_selection
from .engine_capacity import capacity_level_bounds, capacity_profile_metadata, capacity_profile_sha256
from .errors import refuse
from .moisture_species import MoistureSpeciesProfile


ARWEN_PUBLIC_MICROPHYSICS_MODULE = "gpuwm.core.microphysics"
ARWEN_PUBLIC_MICROPHYSICS_CALL = "gpuwm.core.microphysics.apply"
ARWEN_PUBLIC_REFLECTIVITY_MODULE = "gpuwm.core.refl"
ARWEN_PUBLIC_REFLECTIVITY_CALL = "gpuwm.core.refl.consume_refl_10cm"
ARWEN_KERNEL_MANIFEST_MODULE = "gpuwm.certify.kernel_manifest"
ARWEN_RUN_CONFIG_MODULE = "gpuwm.config"

#: Carriers every column component binds, whatever its scalar axis.
COLUMN_THERMODYNAMIC_CARRIERS: tuple[str, ...] = (
    "thp", "thb", "p", "alt", "phb", "php",
)


@dataclass(frozen=True, slots=True)
class ColumnComponentSpec:
    """Everything one admitted CM1 ``ptype`` changes about this adapter."""

    component_id: str
    #: The scheme's own name, as CM1 and WRF spell it in prose.
    scheme_name: str
    component_abi: str
    metadata_schema: str
    receipt_schema: str
    reflectivity_consumer_abi: str
    #: ArWen's ``cfg.mp_physics`` for this scheme.
    mp_physics: int
    #: CM1's ``ptype`` for this scheme, which the receipt records as provenance.
    cm1_ptype: int
    #: The transcribed CM1 ``q3d`` axis (``cm1core.moisture_species``).
    profile: MoistureSpeciesProfile
    #: CM1 scalar name -> the ArWen carrier keyword it is bound to.  Mass
    #: species are identity-mapped; CM1's number-concentration names are not
    #: ArWen's, so nothing here may be bound positionally.
    carrier_names: Mapping[str, str]
    #: Carriers this component holds beyond its transported scalars, with the
    #: reason each is not a CM1 prognostic.  These are volumes.
    component_owned_carriers: Mapping[str, str]
    effective_radius_initial_microns: tuple[tuple[str, float], ...]
    surface_scratch: tuple[str, ...]
    minimum_levels: int
    maximum_levels: int
    #: The species ArWen's measured reflectivity diagnostic consumes, and the
    #: transported ones it does not.
    reflectivity_species: tuple[str, ...]
    reflectivity_absent_species: tuple[str, ...]
    #: Whether the explicit ArWen column-capacity overlay applies here.  The
    #: overlay patches ``gpuwm/core/wsm6_constants.py`` and
    #: ``gpuwm/physics_vertical_contract.py``; a scheme whose launcher reads
    #: neither gains nothing from it and must refuse it rather than claim a
    #: capacity nothing on its path was rebuilt for.
    capacity_profile_supported: bool
    #: Scheme-specific execution facts the receipt and metadata must carry,
    #: each already cited in the binding module that declares it.
    extra_execution: Mapping[str, object]
    claim_limit: str
    #: The component token inside this adapter's refusal codes, so a refusal
    #: names which scheme's contract was violated.
    code_token: str
    #: The gate every refusal from this binding is attributed to.
    gate: str
    #: The caller-owned carrier dataclass this binding accepts, declared by
    #: the binding module so the generic validator can refuse a duck-typed
    #: state without knowing any scheme's field list.
    carriers_type: type
    #: Further read-only carriers on the w (interface) grid that this scheme's
    #: ArWen launcher requires beyond the six thermodynamic ones.  Empty for
    #: every scheme that needs none, which keeps those components' carrier
    #: sets, tile inventories and their declared order exactly what they were.
    #: The only member so far is ``w``: WRF hands its microphysics driver
    #: ``grid%w_2`` indexed ``kts:kte`` -- the LOWER full-level slice, not a
    #: mass-level average -- and ArWen's Thompson, Milbrandt-Yau and
    #: aerosol-aware Thompson adapters take ``state.w[:nz]`` or ``state.w[:-1]``
    #: in exactly that reading, so GM1's own ``w3d`` binds unchanged.
    extra_interface_carriers: tuple[str, ...] = ()
    #: Component-owned carriers on the SURFACE, with the reason each is not a
    #: CM1 prognostic.  Empty for every scheme that has none, which keeps those
    #: components' persistence inventories and their declared order unchanged.
    #: The only members so far are aerosol-aware Thompson's ``nwfa2d`` and
    #: ``nifa2d``, WRF's two-dimensional surface aerosol emission rates.
    component_owned_surface_carriers: Mapping[str, str] = MappingProxyType({})

    @property
    def species(self) -> tuple[str, ...]:
        """The CM1 scalar axis in ``q3d`` order."""

        return self.profile.names

    @property
    def effective_radius_fields(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.effective_radius_initial_microns)

    @property
    def component_persistence(self) -> tuple[str, ...]:
        return (
            ("h_diabatic",)
            + self.effective_radius_fields
            + tuple(self.component_owned_carriers)
            + tuple(self.component_owned_surface_carriers)
            + self.surface_scratch
        )

    @property
    def surface_persistence(self) -> tuple[str, ...]:
        """Persistence entries shaped (ny, nx) rather than (nz, ny, nx)."""

        return tuple(self.component_owned_surface_carriers) + self.surface_scratch

    @property
    def mutable_carriers(self) -> tuple[str, ...]:
        return ("thp",) + self.species

    @property
    def volume_carriers(self) -> tuple[str, ...]:
        return ("thp", "p", "alt") + self.species

    @property
    def all_carriers(self) -> tuple[str, ...]:
        return (
            COLUMN_THERMODYNAMIC_CARRIERS
            + self.extra_interface_carriers
            + self.species
        )

    @property
    def read_only_carriers(self) -> tuple[str, ...]:
        return ("thb", "p", "alt", "phb", "php") + self.extra_interface_carriers

    def arwen_carrier(self, cm1_name: str) -> str:
        return self.carrier_names[cm1_name]


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _audited_surface_sha256(
    capacity_profile: str | None = None, component_id: str | None = None
) -> str:
    payload = json.dumps(
        engine_pin.audited_surface(capacity_profile, component_id),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def component_metadata(
    spec: ColumnComponentSpec,
    *,
    capacity_profile: str | None = None,
    ihail: int = DEFAULT_IHAIL,
) -> dict[str, object]:
    """Return the stable, CUDA-free semantic contract for this adapter."""

    bounds = (
        (spec.minimum_levels, spec.maximum_levels)
        if capacity_profile is None
        else capacity_level_bounds(capacity_profile)
    )
    dense_ice = dense_ice_selection(spec.component_id, ihail)
    metadata: dict[str, object] = {
        "schema": spec.metadata_schema,
        "component_id": spec.component_id,
        "component_abi": spec.component_abi,
        "claim_limit": spec.claim_limit,
        "execution": {
            "cadence": "once after the completed CM1 RK stages per large step",
            "entry_point": ARWEN_PUBLIC_MICROPHYSICS_CALL,
            "mp_physics": spec.mp_physics,
            # A scheme with no dense-ice branch contributes no RunConfig
            # keyword, so the document names none; "dense_ice" below records
            # the absence and why.
            **(
                {}
                if dense_ice.arwen_option is None
                else {dense_ice.arwen_option: dense_ice.ihail}
            ),
            "refl_10cm_due": "strict caller cadence boolean",
            **dict(spec.extra_execution),
            "synchronization": (
                "enqueues on the caller's current CUDA stream; the adapter records a "
                "ready event on that stream and performs no synchronization or host copy"
            ),
            "reflectivity": {
                "producer": f"public ArWen {spec.scheme_name} output-due diagnostic",
                "consumer": ARWEN_PUBLIC_REFLECTIVITY_CALL,
                "consumer_abi": spec.reflectivity_consumer_abi,
                "shape": ["nz", "ny", "nx"],
                "dtype": "float32",
                "units": "dBZ",
                "lifetime": "consumed exactly once before apply returns; not restart state",
                "species_mapping": list(spec.reflectivity_species),
                "absent_species": list(spec.reflectivity_absent_species),
            },
        },
        "dense_ice": dense_ice.record(),
        "moisture_species": spec.profile.record(),
        "engine": {
            "requirement": engine_pin.GPUWM_REQUIREMENT,
            "measured_version": engine_pin.GPUWM_MEASURED_VERSION,
            "audited_source_identity": engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY,
            "audited_surface_sha256": _audited_surface_sha256(
                capacity_profile, spec.component_id
            ),
            "audited_surface": dict(sorted(
                engine_pin.audited_surface(capacity_profile, spec.component_id).items()
            )),
        },
        "carriers": {
            "dimension_order": "zyx",
            "dtype": "float32",
            "memory": "C-contiguous CuPy arrays, zero-copy bound by object identity",
            "mutable_in_place": list(spec.mutable_carriers),
            "read_only": list(spec.read_only_carriers),
            "species": list(spec.species),
            "cm1_to_arwen_carrier": dict(spec.carrier_names),
            "component_owned_carriers": dict(spec.component_owned_carriers),
            "volume_shape": ["nz", "ny", "nx"],
            "thb_shapes": [["nz"], ["nz", "ny", "nx"]],
            "phb_shapes": [["nz+1"], ["nz+1", "ny", "nx"]],
            "php_shape": ["nz+1", "ny", "nx"],
            "vertical_level_bounds": list(bounds),
        },
        "persistence": {
            "component_owned": list(spec.component_persistence),
            "restart_required": list(spec.component_persistence),
            "caller_owned_prognostic": list(spec.mutable_carriers),
            "restore": (
                "exact zero-copy device binding before the first public apply; "
                "does not execute physics"
            ),
            "effective_radius_units": "micrometres",
            "h_diabatic_units": "K s-1",
            "surface_accumulator_units": "mm (kg m-2)",
        },
    }
    if capacity_profile is not None:
        metadata["capacity_profile"] = capacity_profile_metadata(capacity_profile)
        metadata["capacity_profile_sha256"] = capacity_profile_sha256(capacity_profile)
    return metadata


def component_metadata_bytes(
    spec: ColumnComponentSpec,
    *,
    capacity_profile: str | None = None,
    ihail: int = DEFAULT_IHAIL,
) -> bytes:
    """Canonical LF-terminated JSON bytes for :func:`component_metadata`."""

    return _canonical_json_bytes(
        component_metadata(spec, capacity_profile=capacity_profile, ihail=ihail)
    )


def component_metadata_sha256(
    spec: ColumnComponentSpec,
    *,
    capacity_profile: str | None = None,
    ihail: int = DEFAULT_IHAIL,
) -> str:
    """SHA-256 of the canonical component metadata bytes."""

    return hashlib.sha256(
        component_metadata_bytes(spec, capacity_profile=capacity_profile, ihail=ihail)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ArwenColumnStepResult:
    """Asynchronous result of one public ArWen microphysics call."""

    diagnostics: object
    reflectivity_10cm: object | None
    h_diabatic: object
    effective_radius: Mapping[str, object]
    ready_event: object
    call_index: int
    stream_pointer: int
    receipt: Mapping[str, object]


def _refuse_option(
    *, code: str, reason: str, remedy: str, option: str, value: object, gate: str
) -> NoReturn:
    refuse(
        code=code,
        gate=gate,
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )


def _require_strict_int(
    name: str, value: object, minimum: int, *,
    code_token: str, gate: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _refuse_option(
            code=f"CM1-{code_token}-DIMENSION",
            gate=gate,
            reason=(
                f"{name} must be a strict integer >= {minimum}; accepting this value "
                "would mis-size the zyx carrier contract"
            ),
            remedy=f"pass {name} as an integer >= {minimum}",
            option=name,
            value=value,
        )
    return value


def _require_positive_finite(
    name: str, value: object, *,
    code_token: str, gate: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse_option(
            code=f"CM1-{code_token}-SCALAR",
            gate=gate,
            reason=(
                f"{name} must be a finite positive real scalar; accepting this value "
                "would create a different or undefined component interval/geometry"
            ),
            remedy=f"pass {name} as a finite number > 0",
            option=name,
            value=value,
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse_option(
            code=f"CM1-{code_token}-SCALAR",
            gate=gate,
            reason=(
                f"{name} must be finite and positive; accepting this value would "
                "create a different or undefined component interval/geometry"
            ),
            remedy=f"pass {name} as a finite number > 0",
            option=name,
            value=value,
        )
    return result


def _shape_of(
    name: str, value: object, *,
    code_token: str, gate: str,
) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        _refuse_option(
            code=f"CM1-{code_token}-CARRIER-TYPE",
            gate=gate,
            reason=f"carrier {name} has no shape and cannot satisfy the zyx ABI",
            remedy="pass a C-contiguous CuPy float32 array",
            option=f"carriers.{name}",
            value=type(value).__name__,
        )
    try:
        normalized = tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError):
        _refuse_option(
            code=f"CM1-{code_token}-CARRIER-SHAPE",
            gate=gate,
            reason=f"carrier {name} has a non-integral shape and cannot index zyx",
            remedy="pass an array with the exact declared carrier shape",
            option=f"carriers.{name}.shape",
            value=repr(shape),
        )
    return normalized


def _dtype_is_float32(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    return (
        dtype is not None
        and getattr(dtype, "name", None) == "float32"
        and getattr(dtype, "kind", None) == "f"
        and getattr(dtype, "itemsize", None) == 4
    )


def _is_c_contiguous(value: object) -> bool:
    flags = getattr(value, "flags", None)
    if flags is None:
        return False
    result = getattr(flags, "c_contiguous", None)
    if result is None:
        try:
            result = flags["C_CONTIGUOUS"]
        except (KeyError, TypeError, IndexError):
            return False
    return bool(result)


def _validate_carriers_without_cuda(
    carriers: object,
    *,
    spec: ColumnComponentSpec,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    code_token, gate = spec.code_token, spec.gate
    if not isinstance(carriers, spec.carriers_type):
        _refuse_option(
            code=f"CM1-{code_token}-CARRIER-SET",
            gate=gate,
            reason=(
                "the public component requires one explicit "
                f"{spec.carriers_type.__name__} set; duck-typing a partial state "
                "could omit or reorder a species"
            ),
            remedy=(
                f"construct {spec.carriers_type.__name__} with all "
                f"{len(spec.all_carriers)} named fields"
            ),
            option="carriers",
            value=type(carriers).__name__,
        )

    volume_shape = (nz, ny, nx)
    interface_shape = (nz + 1, ny, nx)
    expected_shapes: dict[str, tuple[tuple[int, ...], ...]] = {
        name: (volume_shape,) for name in spec.volume_carriers
    }
    expected_shapes["thb"] = ((nz,), volume_shape)
    expected_shapes["phb"] = ((nz + 1,), interface_shape)
    expected_shapes["php"] = (interface_shape,)
    for name in spec.extra_interface_carriers:
        expected_shapes[name] = (interface_shape,)

    for name in spec.all_carriers:
        value = getattr(carriers, name)
        shape = _shape_of(name, value, code_token=code_token, gate=gate)
        if shape not in expected_shapes[name]:
            _refuse_option(
                code=f"CM1-{code_token}-CARRIER-SHAPE",
                gate=gate,
                reason=(
                    f"carrier {name} has shape {shape}, not one of "
                    f"{expected_shapes[name]}; accepting it would transpose, broadcast, "
                    "or truncate a CM1 column"
                ),
                remedy="bind the exact zyx CM1 scalar/interfacial view",
                option=f"carriers.{name}.shape",
                value=shape,
            )
        if not _dtype_is_float32(value):
            dtype = getattr(value, "dtype", None)
            _refuse_option(
                code=f"CM1-{code_token}-CARRIER-DTYPE",
                gate=gate,
                reason=(
                    f"carrier {name} is {dtype!s}, not IEEE-754 float32; accepting it "
                    "would change ArWen kernel argument width or arithmetic"
                ),
                remedy="bind a float32 carrier without casting inside the component",
                option=f"carriers.{name}.dtype",
                value=str(dtype),
            )
        if not _is_c_contiguous(value):
            _refuse_option(
                code=f"CM1-{code_token}-CARRIER-STRIDES",
                gate=gate,
                reason=(
                    f"carrier {name} is not C-contiguous zyx; accepting it would make "
                    "the column kernel read a different layout"
                ),
                remedy="bind a C-contiguous zyx device view prepared by CM1",
                option=f"carriers.{name}.flags.c_contiguous",
                value=False,
            )

    mutable = [getattr(carriers, name) for name in spec.mutable_carriers]
    if len({id(value) for value in mutable}) != len(mutable):
        _refuse_option(
            code=f"CM1-{code_token}-CARRIER-ALIAS",
            gate=gate,
            reason=(
                "two in-place carriers are the same object; one species/theta "
                "update would overwrite another"
            ),
            remedy="bind distinct " + "/".join(spec.mutable_carriers) + " device arrays",
            option="carriers.mutable_identity",
            value="aliased",
        )


def _validate_cupy_residency(
    cupy: object,
    carriers: object,
    *,
    spec: ColumnComponentSpec,
) -> int:
    code_token, gate = spec.code_token, spec.gate
    ndarray_type = getattr(cupy, "ndarray", None)
    if not isinstance(ndarray_type, type):
        raise RuntimeError("installed CuPy module exposes no ndarray type")
    for name in spec.all_carriers:
        value = getattr(carriers, name)
        if not isinstance(value, ndarray_type):
            _refuse_option(
                code=f"CM1-{code_token}-NONCUDA-CARRIER",
                gate=gate,
                reason=(
                    f"carrier {name} is not a CuPy device array; accepting a host or "
                    "foreign array would introduce an implicit state round trip"
                ),
                remedy="bind the resident CuPy view owned by the CM1 CUDA state",
                option=f"carriers.{name}",
                value=type(value).__name__,
            )

    runtime = getattr(getattr(cupy, "cuda"), "runtime")
    current_device = int(runtime.getDevice())
    devices = {
        int(getattr(getattr(carriers, name), "device").id) for name in spec.all_carriers
    }
    if devices != {current_device}:
        _refuse_option(
            code=f"CM1-{code_token}-DEVICE-OWNERSHIP",
            gate=gate,
            reason=(
                f"carrier devices {sorted(devices)} do not all equal current CUDA "
                f"device {current_device}; a cross-device call could copy or fault"
            ),
            remedy="make the carrier-owning CM1 device current before component apply",
            option="carriers.device_ids",
            value=sorted(devices),
        )
    return current_device


def _validate_persistence_without_cuda(
    arrays: Mapping[str, object], *, spec: ColumnComponentSpec,
    nx: int, ny: int, nz: int,
) -> None:
    """Validate one complete restart inventory without importing CuPy.

    Persistence is intentionally bound by object identity.  Restoring through
    an adapter-side numerical conversion would make a resumed trajectory
    depend on Python rather than on the serialized binary32 words.
    """

    code_token, gate = spec.code_token, spec.gate
    if not isinstance(arrays, Mapping):
        _refuse_option(
            code=f"CM1-{code_token}-RESTART-MAPPING",
            gate=gate,
            reason="component persistence restore requires one named mapping",
            remedy="provide the exact ordered component restart inventory",
            option="persistence",
            value=type(arrays).__name__,
        )
    names = tuple(arrays)
    if names != spec.component_persistence:
        _refuse_option(
            code=f"CM1-{code_token}-RESTART-INVENTORY",
            gate=gate,
            reason=(
                "component persistence names or order differ from the versioned "
                "component restart inventory"
            ),
            remedy="restore every declared component field in ABI order",
            option="persistence.names",
            value=list(names),
        )
    volume = (nz, ny, nx)
    surface = (ny, nx)
    for name in spec.component_persistence:
        value = arrays[name]
        expected = surface if name in spec.surface_persistence else volume
        actual = _shape_of(
            f"persistence.{name}", value, code_token=code_token, gate=gate
        )
        if actual != expected:
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-SHAPE",
                gate=gate,
                reason=(
                    f"persistence field {name} has shape {actual}, not {expected}; "
                    "binding it would shift retained column or surface state"
                ),
                remedy="restore the exact product-domain persistence shape",
                option=f"persistence.{name}.shape",
                value=actual,
            )
        if not _dtype_is_float32(value):
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-DTYPE",
                gate=gate,
                reason=f"persistence field {name} is not IEEE-754 binary32",
                remedy="restore the serialized float32 words without conversion",
                option=f"persistence.{name}.dtype",
                value=str(getattr(value, "dtype", None)),
            )
        if not _is_c_contiguous(value):
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-STRIDES",
                gate=gate,
                reason=f"persistence field {name} is not C-contiguous",
                remedy="restore a contiguous product-domain array",
                option=f"persistence.{name}.flags.c_contiguous",
                value=False,
            )
    if len({id(arrays[name]) for name in names}) != len(names):
        _refuse_option(
            code=f"CM1-{code_token}-RESTART-ALIAS",
            gate=gate,
            reason="two component persistence fields share one array object",
            remedy="restore every persistence field into distinct storage",
            option="persistence.identity",
            value="aliased",
        )


class _Cm1ArwenColumnState:
    """Minimal public-attribute state accepted by ArWen microphysics.apply.

    Attribute names are ArWen's, not CM1's: ``spec.carrier_names`` maps each
    CM1 ``q3d`` scalar onto the keyword ArWen's launcher reads it under.  CM1
    orders its number concentrations ``nci,ncs,ncr,ncg``
    (``src/param.F:3446-3449``) and ArWen's launcher takes ``nc,nr,ni,ns,ng``,
    so nothing here may be bound positionally.
    """

    def __init__(
        self,
        cupy: object,
        spec: ColumnComponentSpec,
        carriers: object,
        persistence: Mapping[str, object] | None = None,
    ):
        self._cupy = cupy
        self._spec = spec
        self._scratch: dict[str, object] = {}
        self.physics = _Cm1ReflectivityHandoff()
        self.rebind(carriers)
        shape = tuple(self.p.shape)
        surface = shape[1:]
        if persistence is None:
            self.h_diabatic = cupy.zeros(shape, dtype=cupy.float32)
            for name, initial in spec.effective_radius_initial_microns:
                setattr(
                    self,
                    name,
                    cupy.full(shape, initial, dtype=cupy.float32),
                )
            for name in spec.component_owned_carriers:
                setattr(self, name, cupy.zeros(shape, dtype=cupy.float32))
            for name in spec.component_owned_surface_carriers:
                setattr(self, name, cupy.zeros(surface, dtype=cupy.float32))
            for slot in spec.surface_scratch:
                self.scratch(surface, slot)
        else:
            self.h_diabatic = persistence["h_diabatic"]
            for name in spec.effective_radius_fields:
                setattr(self, name, persistence[name])
            for name in spec.component_owned_carriers:
                setattr(self, name, persistence[name])
            for name in spec.component_owned_surface_carriers:
                setattr(self, name, persistence[name])
            self._scratch.update(
                {slot: persistence[slot] for slot in spec.surface_scratch}
            )

    def rebind(self, carriers: object) -> None:
        for name in COLUMN_THERMODYNAMIC_CARRIERS + self._spec.extra_interface_carriers:
            setattr(self, name, getattr(carriers, name))
        for name in self._spec.species:
            setattr(self, self._spec.arwen_carrier(name), getattr(carriers, name))

    def scratch(self, shape: object, slot: str, dtype: object = None) -> object:
        normalized_shape = (
            tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        )
        requested_dtype = self._cupy.dtype(
            self._cupy.float32 if dtype is None else dtype
        )
        value = self._scratch.get(slot)
        if value is None:
            value = self._cupy.zeros(normalized_shape, dtype=requested_dtype)
            self._scratch[slot] = value
        elif tuple(value.shape) != normalized_shape:
            raise ValueError(
                f"scratch slot {slot!r} has shape {tuple(value.shape)}, requested "
                f"{normalized_shape}"
            )
        elif value.dtype != requested_dtype:
            raise ValueError(
                f"scratch slot {slot!r} has dtype {value.dtype}, requested "
                f"{requested_dtype}"
            )
        return value

    def existing_scratch(self, slot: str) -> object | None:
        return self._scratch.get(slot)

    def total_theta(self) -> object:
        thb = self.thb if self.thb.ndim == 3 else self.thb[:, None, None]
        return thb + self.thp

    def effective_radius_arrays(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self._spec.effective_radius_fields
        }

    def persistent_arrays(self) -> dict[str, object]:
        result: dict[str, object] = {"h_diabatic": self.h_diabatic}
        result.update(self.effective_radius_arrays())
        result.update({
            name: getattr(self, name)
            for name in self._spec.component_owned_carriers
        })
        result.update({
            name: getattr(self, name)
            for name in self._spec.component_owned_surface_carriers
        })
        result.update({slot: self._scratch[slot] for slot in self._spec.surface_scratch})
        if tuple(result) != self._spec.component_persistence:
            raise RuntimeError(
                "component persistence order drifted from the declared inventory"
            )
        return result


class _Cm1ReflectivityHandoff:
    """Minimal public ArWen one-frame diagnostic handoff."""

    def __init__(self) -> None:
        self.refl_10cm: object | None = None


class ArwenColumnComponent:
    """A persistent CM1-owned binding to one ArWen public column dispatch."""

    #: Bound by each binding module; never None on a concrete subclass.
    SPEC: ColumnComponentSpec

    def __init__(
        self,
        *,
        component_id: str,
        nx: int,
        ny: int,
        nz: int,
        dx_m: float,
        dy_m: float,
        dz_m: float,
        dt_s: float,
        ztop_m: float | None = None,
        capacity_profile: str | None = None,
        ihail: int = DEFAULT_IHAIL,
    ) -> None:
        spec = self.SPEC
        self.spec = spec
        code_token, gate = spec.code_token, spec.gate
        if component_id != spec.component_id:
            _refuse_option(
                code="CM1-COMPONENT-ID",
                gate=gate,
                reason=(
                    f"component {component_id!r} is not the measured semantic "
                    f"{spec.component_id} component; translating a CM1 ptype or "
                    "unknown token could run the wrong physics"
                ),
                remedy=f"select the exact semantic id {spec.component_id!r}",
                option="component_id",
                value=component_id,
            )
        self.component_id = component_id
        self.nx = _require_strict_int("nx", nx, 1, code_token=code_token, gate=gate)
        self.ny = _require_strict_int("ny", ny, 1, code_token=code_token, gate=gate)
        self.nz = _require_strict_int(
            "nz", nz, spec.minimum_levels, code_token=code_token, gate=gate
        )
        # CM1's ihail is admitted here, on the host, before any geometry is
        # priced: it changes which constants the scheme is initialized with.
        self.dense_ice = dense_ice_selection(spec.component_id, ihail)
        self.ihail = self.dense_ice.ihail
        if capacity_profile is not None and not spec.capacity_profile_supported:
            _refuse_option(
                code=f"CM1-{code_token}-CAPACITY-PROFILE",
                gate=gate,
                reason=(
                    f"the explicit column-capacity overlay rebuilds WSM6's own "
                    f"constants and vertical contract, which the {spec.component_id} "
                    "launcher never reads; selecting it here would record a deeper "
                    "admitted column than anything on this scheme's path was rebuilt "
                    "for, and the run would then be admitted at an nz its kernel "
                    "indexes past"
                ),
                remedy=(
                    "leave runtime.capacity_profile at \"default\" for this "
                    f"component; its own kernel already admits "
                    f"{spec.minimum_levels} <= nz <= {spec.maximum_levels}"
                ),
                option="capacity_profile",
                value=capacity_profile,
            )
        max_levels = (
            spec.maximum_levels
            if capacity_profile is None
            else capacity_level_bounds(capacity_profile)[1]
        )
        self.capacity_profile = capacity_profile
        if self.nz > max_levels:
            _refuse_option(
                code=f"CM1-{code_token}-VERTICAL-BOUNDS",
                gate=gate,
                reason=(
                    f"nz={self.nz} exceeds the audited {spec.component_id} kernel "
                    f"tier maximum {max_levels}; execution would overrun per-column "
                    "storage"
                ),
                remedy=f"use {spec.minimum_levels} <= nz <= {max_levels}",
                option="nz",
                value=self.nz,
            )
        self.dx_m = _require_positive_finite("dx_m", dx_m, code_token=code_token, gate=gate)
        self.dy_m = _require_positive_finite("dy_m", dy_m, code_token=code_token, gate=gate)
        self.dz_m = _require_positive_finite("dz_m", dz_m, code_token=code_token, gate=gate)
        self.dt_s = _require_positive_finite("dt_s", dt_s, code_token=code_token, gate=gate)
        # param.F:2608 makes ztop = dz*nk only on the uniform column.  On any
        # param.F:7814-8139 branch dz is the metric reference of param.F:8170
        # and the model top is sigmaf(nk+1), so a caller on a stretched grid
        # declares it: dz_m*nz would hand the component a column shorter than
        # the one whose geopotential faces it is about to receive, and every
        # height-dependent fall speed and sedimentation step would be taken on
        # a domain that does not exist.
        if ztop_m is not None:
            ztop = _require_positive_finite("ztop_m", ztop_m, code_token=code_token, gate=gate)
            if ztop < self.dz_m:
                _refuse_option(
                    code=f"CM1-{code_token}-GEOMETRY-OVERFLOW",
                    gate=gate,
                    reason=(
                        f"the declared model top {ztop} m is below one metric "
                        f"reference dz of {self.dz_m} m, so no layer of this "
                        "column fits under it"
                    ),
                    remedy="declare grid.ztop_m as this coordinate's own sigmaf(nk+1)",
                    option="ztop_m",
                    value=ztop,
                )
        else:
            ztop = self.dz_m * self.nz
            if not math.isfinite(ztop):
                _refuse_option(
                    code=f"CM1-{code_token}-GEOMETRY-OVERFLOW",
                    gate=gate,
                    reason="dz_m*nz is not finite, so ArWen could receive invalid geometry",
                    remedy="use finite grid dimensions whose product is representable",
                    option="dz_m*nz",
                    value=ztop,
                )
        self.ztop_m = ztop
        self._state: _Cm1ArwenColumnState | None = None
        self._device_id: int | None = None
        self._stream_pointer: int | None = None
        self._call_count = 0
        self._last_receipt: dict[str, object] | None = None
        self._restored_completed_calls: int | None = None

    def _inspect_engine(self) -> engine_pin.EngineInspection:
        return engine_pin.inspect(
            capacity_profile=self.capacity_profile,
            component_id=self.spec.component_id,
        )

    @property
    def state(self) -> object | None:
        """The public-attribute state bound on the most recent call."""

        return self._state

    @property
    def call_count(self) -> int:
        return self._call_count

    def existing_scratch(self, slot: str) -> object | None:
        if self._state is None:
            return None
        return self._state.existing_scratch(slot)

    def persistent_arrays(self) -> Mapping[str, object]:
        if self._state is None:
            return MappingProxyType({})
        return MappingProxyType(self._state.persistent_arrays())

    def restore_persistent_arrays(
        self,
        carriers: object,
        arrays: Mapping[str, object],
        *,
        completed_calls: int,
    ) -> object:
        spec = self.spec
        code_token, gate = spec.code_token, spec.gate
        """Bind exact restart arrays without executing microphysics.

        The caller must invoke this once on the device and stream that will
        execute subsequent calls.  No array is copied, initialized, or
        numerically transformed; the supplied binary32 objects become the
        component-owned persistence by identity.  A ready event records only
        the binding dependency.
        """

        if (
            isinstance(completed_calls, bool)
            or not isinstance(completed_calls, int)
            or completed_calls < 0
        ):
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-CADENCE",
                gate=gate,
                reason="completed_calls must be a strict nonnegative integer",
                remedy="restore the exact completed public-call count",
                option="completed_calls",
                value=completed_calls,
            )
        if self._state is not None or self._call_count != 0:
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-LIFETIME",
                gate=gate,
                reason="component persistence can only be restored before its first apply",
                remedy="construct a fresh component for checkpoint resume",
                option="component.call_count",
                value=self._call_count,
            )
        _validate_carriers_without_cuda(
            carriers, spec=spec, nx=self.nx, ny=self.ny, nz=self.nz
        )
        _validate_persistence_without_cuda(
            arrays, spec=spec, nx=self.nx, ny=self.ny, nz=self.nz
        )
        inspection = self._inspect_engine()
        if not inspection.admitted:
            refuse(
                code="CM1-RESTART-ARWEN-ENGINE-DRIFT",
                gate=f"ArWen-{code_token}-restart",
                reason=(
                    f"the installed ArWen public {spec.component_id} estate "
                    "differs from the estate that owns this persistence inventory"
                ),
                remedy="install the measured ArWen estate or migrate the restart ABI",
                option="engine_inspection",
                value=inspection.as_dict(),
            )
        cupy = importlib.import_module("cupy")
        _validate_carriers_without_cuda(
            carriers, spec=spec, nx=self.nx, ny=self.ny, nz=self.nz
        )
        _validate_persistence_without_cuda(
            arrays, spec=spec, nx=self.nx, ny=self.ny, nz=self.nz
        )
        device_id = _validate_cupy_residency(cupy, carriers, spec=spec)
        ndarray = getattr(cupy, "ndarray", None)
        current_device = int(cupy.cuda.runtime.getDevice())
        for name in spec.component_persistence:
            value = arrays[name]
            if not isinstance(ndarray, type) or not isinstance(value, ndarray):
                _refuse_option(
                    code=f"CM1-{code_token}-RESTART-NONCUDA",
                    gate=gate,
                    reason=f"persistence field {name} is not a CuPy device array",
                    remedy="load restart persistence into resident CuPy storage",
                    option=f"persistence.{name}",
                    value=type(value).__name__,
                )
            owner = getattr(getattr(value, "device", None), "id", None)
            if owner is None or int(owner) != current_device:
                _refuse_option(
                    code=f"CM1-{code_token}-RESTART-DEVICE",
                    gate=gate,
                    reason=f"persistence field {name} is not on device {current_device}",
                    remedy="restore all component persistence on the carrier device",
                    option=f"persistence.{name}.device",
                    value=owner,
                )
        carrier_objects = {
            id(getattr(carriers, name)) for name in spec.all_carriers
        }
        if any(
            id(arrays[name]) in carrier_objects for name in spec.component_persistence
        ):
            _refuse_option(
                code=f"CM1-{code_token}-RESTART-CARRIER-ALIAS",
                gate=gate,
                reason=(
                    "component persistence aliases a prognostic or read-only "
                    "carrier and could overwrite live transition input"
                ),
                remedy="bind distinct restart storage for component persistence",
                option="persistence.carrier_identity",
                value="aliased",
            )
        stream = cupy.cuda.get_current_stream()
        stream_pointer = int(stream.ptr)
        self._state = _Cm1ArwenColumnState(cupy, spec, carriers, persistence=arrays)
        restored = self._state.persistent_arrays()
        if any(restored[name] is not arrays[name] for name in spec.component_persistence):
            raise RuntimeError("component persistence restore lost object identity")
        self._device_id = device_id
        self._stream_pointer = stream_pointer
        self._call_count = completed_calls
        self._restored_completed_calls = completed_calls
        ready = cupy.cuda.Event(disable_timing=True)
        ready.record(stream)
        return ready

    def receipt_dict(self) -> dict[str, object]:
        if self._last_receipt is None:
            raise RuntimeError(
                f"{self.spec.component_id} component has not completed a public apply call"
            )
        # Round-tripping through canonical JSON returns a detached JSON-safe copy.
        return json.loads(_canonical_json_bytes(self._last_receipt))

    def receipt_bytes(self) -> bytes:
        return _canonical_json_bytes(self.receipt_dict())

    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_bytes()).hexdigest()

    def apply(
        self,
        carriers: object,
        *,
        refl_10cm_due: bool = False,
    ) -> ArwenColumnStepResult:
        """Enqueue one scheme call on the caller's current CUDA stream.

        Every backend-neutral refusal precedes engine inspection and both
        runtime imports.  A successful call mutates ``thp`` and every
        transported scalar in place, retains component persistence, records a
        ready event without synchronizing, and snapshots ArWen's kernel
        manifest.
        """

        spec = self.spec
        code_token, gate = spec.code_token, spec.gate
        if type(refl_10cm_due) is not bool:
            _refuse_option(
                code=f"CM1-{code_token}-REFLECTIVITY-FLAG",
                gate=gate,
                reason="refl_10cm_due must be a strict boolean schedule decision",
                remedy="pass False until the CM1 reflectivity consumer is measured",
                option="refl_10cm_due",
                value=refl_10cm_due,
            )
        _validate_carriers_without_cuda(
            carriers,
            spec=spec,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
        )

        inspection = self._inspect_engine()
        if not inspection.admitted:
            refuse(
                code="CM1-ARWEN-ENGINE-DRIFT",
                gate=gate,
                reason=(
                    "the installed ArWen version/source bytes do not match the audited "
                    f"{spec.component_id} public-dispatch estate; running could "
                    "execute changed physics"
                ),
                remedy=(
                    f"install {engine_pin.GPUWM_REQUIREMENT} at the measured source "
                    "identity, or audit and version a new component ABI"
                ),
                option="engine_inspection",
                value=inspection.as_dict(),
            )

        cupy = importlib.import_module("cupy")
        # Recheck after import so a concurrent caller cannot swap a carrier
        # between the backend-neutral gate and the runtime binding.
        _validate_carriers_without_cuda(
            carriers,
            spec=spec,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
        )
        device_id = _validate_cupy_residency(cupy, carriers, spec=spec)
        stream = cupy.cuda.get_current_stream()
        stream_pointer = int(stream.ptr)
        if self._device_id is not None and self._device_id != device_id:
            _refuse_option(
                code=f"CM1-{code_token}-PERSISTENCE-DEVICE",
                gate=gate,
                reason=(
                    f"component persistence lives on CUDA device {self._device_id}, "
                    f"but this call binds device {device_id}; moving it would break "
                    "restart and stream ownership"
                ),
                remedy="keep one component instance on one device, or construct a new instance",
                option="current_device",
                value=device_id,
            )
        if self._stream_pointer is not None and self._stream_pointer != stream_pointer:
            _refuse_option(
                code=f"CM1-{code_token}-PERSISTENCE-STREAM",
                gate=gate,
                reason=(
                    f"component persistence is ordered on CUDA stream "
                    f"{self._stream_pointer}, but this call selected stream "
                    f"{stream_pointer}; switching without an explicit dependency "
                    "could race the prior microphysics call"
                ),
                remedy=(
                    "keep one component instance on one stream, or construct a new "
                    "instance after the scheduler establishes a separate dependency"
                ),
                option="current_stream_pointer",
                value=stream_pointer,
            )

        config_module = importlib.import_module(ARWEN_RUN_CONFIG_MODULE)
        microphysics = importlib.import_module(ARWEN_PUBLIC_MICROPHYSICS_MODULE)
        manifest_module = importlib.import_module(ARWEN_KERNEL_MANIFEST_MODULE)
        reflectivity_consume = None
        if refl_10cm_due:
            reflectivity_module = importlib.import_module(
                ARWEN_PUBLIC_REFLECTIVITY_MODULE
            )
            reflectivity_consume = getattr(
                reflectivity_module, "consume_refl_10cm", None
            )
            if (
                not callable(reflectivity_consume)
                or getattr(reflectivity_consume, "__module__", None)
                != ARWEN_PUBLIC_REFLECTIVITY_MODULE
            ):
                raise RuntimeError(
                    f"installed {ARWEN_PUBLIC_REFLECTIVITY_CALL} is not the "
                    "audited public callable"
                )
        if getattr(microphysics, "__name__", None) != ARWEN_PUBLIC_MICROPHYSICS_MODULE:
            raise RuntimeError(
                "loaded ArWen microphysics module has the wrong identity"
            )
        public_apply = getattr(microphysics, "apply", None)
        if (
            not callable(public_apply)
            or getattr(public_apply, "__module__", None)
            != ARWEN_PUBLIC_MICROPHYSICS_MODULE
        ):
            raise RuntimeError(
                f"installed {ARWEN_PUBLIC_MICROPHYSICS_CALL} is not the audited public callable"
            )
        run_config = getattr(config_module, "RunConfig")
        # CM1's ihail reaches the scheme here and nowhere else.  The keyword
        # is the component's own (``cm1core.dense_ice.ARWEN_IHAIL_OPTION``),
        # so a component is never handed the other scheme's selector and an
        # unset option can never leave the library default in force.
        cfg = run_config(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx_m,
            dy=self.dy_m,
            ztop=self.ztop_m,
            dt=self.dt_s,
            run_seconds=0.0,
            clock_dt=self.dt_s,
            moist=True,
            mp_physics=spec.mp_physics,
            open_x=True,
            open_y=True,
            # A scheme with no dense-ice branch hands ArWen no selector at
            # all, rather than another scheme's keyword or a None key.
            **(
                {}
                if self.dense_ice.arwen_option is None
                else {self.dense_ice.arwen_option: self.ihail}
            ),
        )

        if self._state is None:
            self._state = _Cm1ArwenColumnState(cupy, spec, carriers)
            self._device_id = device_id
            self._stream_pointer = stream_pointer
        else:
            self._state.rebind(carriers)

        self.prepare_component_owned_carriers(cupy, carriers, self._state)
        before_manifest = manifest_module.kernel_manifest()
        diagnostics = public_apply(
            self._state,
            cfg,
            self.dt_s,
            refl_10cm_due=refl_10cm_due,
        )
        reflectivity_10cm = None
        if refl_10cm_due:
            if reflectivity_consume is None:
                raise RuntimeError("reflectivity consumer binding was lost")
            reflectivity_10cm = reflectivity_consume(self._state)
            expected_reflectivity = self._state.existing_scratch("refl_10cm")
            if reflectivity_10cm is not expected_reflectivity:
                raise RuntimeError("public reflectivity consumer lost scratch identity")
            if (
                tuple(getattr(reflectivity_10cm, "shape", ()))
                != (self.nz, self.ny, self.nx)
                or not _dtype_is_float32(reflectivity_10cm)
                or not _is_c_contiguous(reflectivity_10cm)
            ):
                raise RuntimeError("public reflectivity result violates the zyx ABI")
            if self._state.physics.refl_10cm is not None:
                raise RuntimeError("public reflectivity stash was not consumed exactly once")
        elif self._state.physics.refl_10cm is not None:
            raise RuntimeError("ArWen produced an unscheduled reflectivity stash")
        after_manifest = manifest_module.kernel_manifest()
        changed_manifest = {
            key: value
            for key, value in after_manifest.items()
            if before_manifest.get(key) != value
        }

        ready_event = cupy.cuda.Event(disable_timing=True)
        ready_event.record(stream)
        self._call_count += 1

        adapter_source = Path(__file__).read_bytes()
        persistence = self._state.persistent_arrays()
        zero_copy = all(
            getattr(self._state, name) is getattr(carriers, name)
            for name in COLUMN_THERMODYNAMIC_CARRIERS
        ) and all(
            getattr(self._state, spec.arwen_carrier(name)) is getattr(carriers, name)
            for name in spec.species
        )
        receipt: dict[str, object] = {
            "schema": spec.receipt_schema,
            "component_id": self.component_id,
            "component_abi": spec.component_abi,
            "component_metadata_sha256": component_metadata_sha256(
                spec, capacity_profile=self.capacity_profile, ihail=self.ihail
            ),
            "claim_limit": spec.claim_limit,
            "call_index": self._call_count,
            "configuration": {
                "nx": self.nx,
                "ny": self.ny,
                "nz": self.nz,
                "dx_m": self.dx_m,
                "dy_m": self.dy_m,
                "dz_m": self.dz_m,
                "dt_s": self.dt_s,
                "mp_physics": spec.mp_physics,
                "cm1_ptype": spec.cm1_ptype,
                "ihail": self.ihail,
                # A component whose scheme has no dense-ice branch contributes
                # no RunConfig keyword at all, so the receipt names none
                # either; it records the absence in "dense_ice" instead.
                **(
                    {}
                    if self.dense_ice.arwen_option is None
                    else {self.dense_ice.arwen_option: self.ihail}
                ),
                "refl_10cm_due": refl_10cm_due,
            },
            "dense_ice": self.dense_ice.record(),
            "moisture_species": spec.profile.record(),
            # Every capability is selectable; one that nothing has measured
            # says so in its own receipt, in a token a
            # consumer can grep for, naming what was not checked.
            "validation": component_validation(spec.component_id).record(),
            "cm1_adapter": {
                "module": __name__,
                "source_bytes": len(adapter_source),
                "source_sha256": hashlib.sha256(adapter_source).hexdigest(),
                "source_identity_status": "runtime-read-installed-module",
            },
            "arwen": {
                "entry_point": ARWEN_PUBLIC_MICROPHYSICS_CALL,
                "public_callable_verified": True,
                "requirement": engine_pin.GPUWM_REQUIREMENT,
                "measured_version": engine_pin.GPUWM_MEASURED_VERSION,
                "installed_version": inspection.version,
                "audited_surface_sha256": _audited_surface_sha256(
                    self.capacity_profile, spec.component_id
                ),
                "matched_surface": list(inspection.matched),
                "moved_surface": list(inspection.moved),
                "absent_surface": list(inspection.absent),
            },
            "binding": {
                "dimension_order": "zyx",
                "dtype": "float32",
                "zero_copy_object_identity": zero_copy,
                "carrier_shapes": {
                    name: list(getattr(carriers, name).shape)
                    for name in spec.all_carriers
                },
                "cm1_to_arwen_carrier": dict(spec.carrier_names),
                "device_id": device_id,
                "stream_pointer": stream_pointer,
            },
            "persistence": {
                "inventory": list(spec.component_persistence),
                "restart_required": list(spec.component_persistence),
                "shapes": {
                    name: list(value.shape) for name, value in persistence.items()
                },
                "restored_before_apply": self._restored_completed_calls is not None,
                "restored_completed_calls": self._restored_completed_calls,
                "restore_zero_copy_object_identity": (
                    self._restored_completed_calls is not None
                ),
            },
            "reflectivity": {
                "consumer_abi": spec.reflectivity_consumer_abi,
                "consumer": ARWEN_PUBLIC_REFLECTIVITY_CALL,
                "due": refl_10cm_due,
                "produced": reflectivity_10cm is not None,
                "consumed_once": (
                    refl_10cm_due and self._state.physics.refl_10cm is None
                ),
                "restart_state": False,
                "dtype": (
                    str(getattr(reflectivity_10cm, "dtype", None))
                    if reflectivity_10cm is not None
                    else None
                ),
                "shape": (
                    list(getattr(reflectivity_10cm, "shape"))
                    if reflectivity_10cm is not None
                    else None
                ),
                "units": "dBZ",
                "species_order": list(spec.species),
                "consumer_species": list(spec.reflectivity_species),
                "excluded_species_by_scheme_authority": list(
                    spec.reflectivity_absent_species
                ),
            },
            "kernel_manifest_before": before_manifest,
            "kernel_manifest_after": after_manifest,
            "kernel_manifest_changed": changed_manifest,
            "synchronization": {
                "ready_event_recorded": True,
                "stream_pointer": stream_pointer,
                "host_copy": False,
                "explicit_synchronize": False,
            },
        }
        if self.capacity_profile is not None:
            receipt["capacity_profile"] = capacity_profile_metadata(self.capacity_profile)
            receipt["capacity_profile_sha256"] = capacity_profile_sha256(self.capacity_profile)
        # Prove receipt serializability before exposing a successful result.
        _canonical_json_bytes(receipt)
        self._last_receipt = receipt
        detached_receipt = self.receipt_dict()
        return ArwenColumnStepResult(
            diagnostics=diagnostics,
            reflectivity_10cm=reflectivity_10cm,
            h_diabatic=self._state.h_diabatic,
            effective_radius=MappingProxyType(
                self._state.effective_radius_arrays()
            ),
            ready_event=ready_event,
            call_index=self._call_count,
            stream_pointer=stream_pointer,
            receipt=MappingProxyType(detached_receipt),
        )

    def prepare_component_owned_carriers(
        self, cupy: object, carriers: object, state: _Cm1ArwenColumnState
    ) -> None:
        """Refresh carriers this component owns, immediately before dispatch.

        A single-moment scheme owns none and this is a no-op.  A binding that
        does own one overrides this and must state the CM1 line it reproduces.
        """

        return None


__all__ = [
    "ARWEN_KERNEL_MANIFEST_MODULE",
    "ARWEN_PUBLIC_MICROPHYSICS_CALL",
    "ARWEN_PUBLIC_MICROPHYSICS_MODULE",
    "ARWEN_PUBLIC_REFLECTIVITY_CALL",
    "ARWEN_PUBLIC_REFLECTIVITY_MODULE",
    "ARWEN_RUN_CONFIG_MODULE",
    "COLUMN_THERMODYNAMIC_CARRIERS",
    "ArwenColumnComponent",
    "ArwenColumnStepResult",
    "ColumnComponentSpec",
    "component_metadata",
    "component_metadata_bytes",
    "component_metadata_sha256",
]
