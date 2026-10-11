"""CM1's ``iconly``: build the initial condition, write it, and stop.

``README.namelist:233-236`` and ``cm1.F:4341-4346``: with ``iconly=1`` CM1
completes ``base.F`` and ``init3d.F``, writes the initial output time, and
terminates without integrating.  It is how a CM1 user checks that a sounding,
a hodograph, a bubble placement and a vertical grid are what they intended
before spending machine time on them.

Here that check costs nothing at all: the Rust initializer owns every word of
the initial state on the host, so this door opens no CUDA context, takes no
GPU, and writes exactly the frame layout the device runtime publishes -- the
raw ``history-carriers.f32`` arena, the ``history-field-table.tsv`` semantic
table and the scheduler-scalar table -- so the same readers, the same
differential harness and the same renderer consume it unchanged.

Nothing here integrates, and the receipt says so by status.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .cartesian_history import (
    ARENA_FILENAME,
    FIELD_TABLE_FILENAME,
    RECEIPT_FILENAME,
    SCALAR_TABLE_FILENAME,
    CartesianHistoryProfile,
    cartesian_history_profile,
)
from .forecast_arena import write_forecast_arena_new
from .gm1_runtime import (
    DOMAIN_LOCATION_AUTHORITY,
    DOMAIN_LOCATION_UNITS,
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    MOVE_SPEED_X_NAME,
    MOVE_SPEED_Y_NAME,
    RuntimeConfig,
)

INITIAL_CONDITION_SCHEMA = "gm1.cartesian-initial-condition/v1"
INITIAL_CONDITION_STATUS = "built-initial-condition-and-stopped"
INITIAL_CONDITION_CLAIM_LIMIT = (
    "the CM1 base.F / init3d.F initial condition this configuration names, "
    "written in the runtime's own frame layout at model time zero on the host; "
    "no device was opened, no large step was taken, and nothing here is a "
    "forecast, a trajectory, or a validated simulation"
)

#: Where each published carrier comes from in the Rust initializer's groups.
#: ``None`` means the carrier has no source word at model time zero and is
#: published as exact positive zero; the reason is stated beside it and lands
#: in the receipt, so a reader never has to guess which is which.
_CARRIER_SOURCES: Mapping[str, tuple[str | None, str, str]] = {
    "closure.tkea": (
        None,
        "zero",
        "init3d.F:2196-2210 initializes the prognostic TKE to zero on a cold "
        "start; the idealized initializer holds no tke carrier at all",
    ),
    "pp3d": ("terrain_open.pp3d", "physical", "init3d.F:1735-1830's initial ppi, copied to pp3d"),
    "prs": ("terrain_open.prs", "physical", "init3d.F:2393's prs from pi0+pp3d; the Rust terrain-open group carries the initialized pressure under CM1's prs0 name, exactly as the device runtime reads it"),
    "q3d": ("moist.q3d", "physical", "init3d.F's qa/q3d after the iinit arm"),
    "nc3d": (
        None,
        "zero",
        "a double-moment scheme's number concentrations (param.F:3334-3335, "
        "3427-3428: qa rows nnc1..nnc2); cm1.F allocates qa zero and init3d.F "
        "writes only nqv, nqc and nqi before the nstep=0 frame, so they are "
        "exactly zero at model time zero",
    ),
    "rf": ("terrain_open.rf", "physical", "init3d.F's w-face density of the initial state"),
    "rho": ("terrain_open.rho", "physical", "init3d.F:2394's rho from the equation of state"),
    "terrain_msl": ("terrain_static.terrain_msl", "native", "init_terrain.F's zs; itern=0 is flat"),
    "th3d": ("terrain_open.th3d", "physical", "init3d.F's tha, the theta PERTURBATION"),
    "u3d": ("terrain_open.u3d", "physical", "init3d.F:210-219's ua from base.F's u0"),
    "v3d": ("terrain_open.v3d", "physical", "init3d.F:210-219's va from base.F's v0"),
    "w3d": ("terrain_open.w3d", "physical", "w is exactly zero at model time zero"),
    "zf": ("terrain_static.zf", "native", "param.F:7789-8225's w-face heights"),
}


def _host_zero(source: Any, carrier: Any) -> Any:
    import numpy as np

    return np.zeros(carrier.shape, dtype="<f4")


def _host_filled(value: float) -> Any:
    def fill(source: Any, carrier: Any) -> Any:
        import numpy as np

        return np.full(carrier.shape, np.float32(value), dtype="<f4")

    return fill


def _host_profile(qualified: str) -> Any:
    def read(source: Any, carrier: Any) -> Any:
        import numpy as np

        return np.ascontiguousarray(
            np.asarray(source.field(qualified, layout="profile"), dtype="<f4")
        )

    return read


def _host_th(source: Any, carrier: Any) -> Any:
    import numpy as np

    th0 = np.asarray(source.field("terrain_open.th0", layout="profile"), dtype="<f4")
    th3d = np.asarray(source.field("terrain_open.th3d", layout="physical"), dtype="<f4")
    return np.ascontiguousarray(np.add(th0[:, None, None], th3d, dtype="<f4"))


def _host_prs0(source: Any, carrier: Any) -> Any:
    import numpy as np

    from .cartesian_output_products import CM1_CP, CM1_P00, CM1_RD

    pi0 = np.asarray(source.field("terrain_open.pi0", layout="profile"), dtype="<f4")
    # The expression CartesianOutputProducts._build_profiles evaluates on the
    # device, so a host frame and a device frame carry the same prs0.
    return np.ascontiguousarray(
        (np.float32(CM1_P00) * np.power(pi0, np.float32(CM1_CP / CM1_RD))).astype("<f4")
    )


def _output_product_sources() -> Mapping[str, tuple[Any, str]]:
    """CM1 ``&param9`` products at ``nstep=0``, as CM1's startup writeout has them.

    ``cm1.F:3085-3088``: on a startup CM1 enters the time loop with
    ``dosolve=.false.`` and ``dowriteout=.true.``, so the frame ``iconly``
    stops after (``cm1.F:4341-4351``) is written before any solve, any
    microphysics call and any swath fold.  Every surface accumulator and
    run-long extremum therefore still holds the value it was initialized
    with, and those values -- not invented planes -- are what is published.
    """

    from .cuda_cm1_output_diagnostics import CM1_SWATH_INITIAL

    no_step = (
        "cm1.F:3085-3088 skips solve on a startup, so no microphysics call "
        "precedes the nstep=0 writeout"
    )
    sources: dict[str, tuple[Any, str]] = {
        "rain": (_host_zero, f"init3d.F:187 sets rain=0 and {no_step}; a true zero accumulation"),
        "prate": (_host_zero, f"cm1.F:873-874 allocates prate=0 and {no_step}"),
        "snownc": (_host_zero, f"the WSM6 SNOWNC accumulator starts at zero and {no_step}"),
        "graupelnc": (_host_zero, f"the WSM6 GRAUPELNC accumulator starts at zero and {no_step}"),
        "sr": (_host_zero, f"WSM6 SR is written only by a microphysics call and {no_step}; the device runtime's allocation holds the same zero"),
        "th": (_host_th, "writeout.F 'th' = th0 + tha (solve1.F:893), from terrain_open.th0 (profile) and terrain_open.th3d"),
        "zhval": (_host_profile("terrain_static.zh"), "writeout.F:3179 zh, the terrain_static.zh column the device metrics also read"),
        "pi0": (_host_profile("terrain_open.pi0"), "base.F pi0 (writeout.F:4430-4520)"),
        "th0": (_host_profile("terrain_open.th0"), "base.F th0 (writeout.F:4430-4520)"),
        "qv0": (_host_profile("terrain_open.qv0"), "base.F qv0 (writeout.F:4430-4520)"),
        "u0": (_host_profile("base_wind.u0"), "base.F u0 (writeout.F:4430-4520)"),
        "v0": (_host_profile("base_wind.v0"), "base.F v0 (writeout.F:4430-4520)"),
        "prs0": (_host_prs0, "p00*pi0**(cp/rd), the device runtime's own prs0 derivation"),
    }
    for name, value in CM1_SWATH_INITIAL.items():
        sources[name] = (
            _host_filled(value),
            f"init3d.F:183-199 initializes {name}={value!r}; solve3.F folds it "
            f"only inside solve, and {no_step} or swath fold",
        )
    return sources


#: Products this door cannot source on the host, and the wrong frame each
#: would otherwise be.
_NO_HOST_REFLECTIVITY = (
    "cm1.F:3213-3244 raises getdbz for the nstep=0 writeout, so CM1's frame "
    "carries a real reflectivity field; the host runs no microphysics and no "
    "reflectivity operator, and a zero volume would read as a uniform 0 dBZ "
    "echo. Drop physics.output_dbz for the initial condition"
)
_HOST_UNAVAILABLE_PRODUCTS: Mapping[str, str] = {
    name: _NO_HOST_REFLECTIVITY for name in ("dbz", "cref", "dbz_level")
}


class _HostArrayModule:
    """The two names :mod:`forecast_arena` asks a CUDA module for.

    The arena writer is shape-, dtype- and contiguity-checked and copies in
    bounded chunks; on the host the copy is the identity.  Reusing it, rather
    than writing a second arena writer, is what makes an ``iconly`` frame and
    a device frame the same bytes in the same order.
    """

    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self.ndarray = np.ndarray

    def dtype(self, name: str) -> Any:
        return self._np.dtype(name)

    def asnumpy(self, value: Any) -> Any:
        return self._np.asarray(value)


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb", buffering=0) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _is_budget_carrier(name: str) -> bool:
    """A CM1 ``output_*budget`` term or a GM1 vorticity-budget carrier."""

    prefix = name.split("_", 1)[0]
    return prefix in ("ub", "vb", "wb", "ptb", "qvb", "xvortb", "yvortb", "zvortb")


def initial_condition_carriers(
    source: Any, profile: CartesianHistoryProfile
) -> tuple[dict[str, Any], dict[str, str]]:
    """The profile's carriers, in profile order, from the host initializer."""

    import numpy as np

    from .cm1_output_selection import OUTPUT_GROUP

    products = _output_product_sources()
    carriers: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for carrier in profile.carriers:
        if carrier.name in _HOST_UNAVAILABLE_PRODUCTS:
            raise ValueError(
                f"the initial-condition door cannot source history carrier "
                f"{carrier.name!r}: {_HOST_UNAVAILABLE_PRODUCTS[carrier.name]}"
            )
        if carrier.name in _CARRIER_SOURCES:
            qualified, layout, note = _CARRIER_SOURCES[carrier.name]
            if qualified is None:
                value = np.zeros(carrier.shape, dtype="<f4")
            else:
                value = np.ascontiguousarray(
                    np.asarray(source.field(qualified, layout=layout), dtype="<f4")
                )
                note = f"{qualified} ({note})"
        elif carrier.group == OUTPUT_GROUP and _is_budget_carrier(carrier.name):
            # cm1.F allocates every udiag/vdiag/wdiag/tdiag/qdiag slot as zero
            # and cm1.F:3085-3088 writes the nstep=0 frame before any solve, so
            # CM1's own initial frame carries zero budget terms.
            value = np.zeros(carrier.shape, dtype="<f4")
            note = (
                "cm1.F allocates the budget diag arrays zero and cm1.F:3085-3088 "
                "writes nstep=0 before any solve"
            )
        elif carrier.name in products:
            build, note = products[carrier.name]
            value = build(source, carrier)
            if value.dtype != np.dtype("<f4"):
                raise ValueError(
                    f"initial-condition product {carrier.name} is {value.dtype}, not "
                    "binary32; the arena would reinterpret its words"
                )
        else:
            raise ValueError(
                f"the initial-condition door has no source for history carrier "
                f"{carrier.name!r}; publishing an invented field under a CM1 name "
                "is worse than refusing the frame"
                + (
                    ". It is a &param9 product with no host operator: pass "
                    "--restart-dir to build the initial condition on the device, "
                    "which computes it, or turn its output_* switch off"
                    if carrier.group == OUTPUT_GROUP else ""
                )
            )
        if tuple(value.shape) != tuple(carrier.shape):
            raise ValueError(
                f"initial-condition carrier {carrier.name} has extent "
                f"{tuple(value.shape)} but the published profile declares "
                f"{tuple(carrier.shape)}; every following carrier's byte offset "
                "in the arena would shift"
            )
        carriers[carrier.name] = value
        provenance[carrier.name] = note
    return carriers, provenance


def _scalar_table_bytes(moving_frame: Any) -> bytes:
    """``writeout.F:9614-9648``'s four moving-frame scalars at model time zero.

    ``cm1.F:3512-3513`` accumulates ``domainlocx``/``domainlocy`` from the
    first large step onwards, so both are exactly zero in the initial
    condition; the two speeds are the declared frame.
    """

    rows = [
        (DOMAIN_LOCATION_X_NAME, "0.0"),
        (DOMAIN_LOCATION_Y_NAME, "0.0"),
        (MOVE_SPEED_X_NAME, repr(float(moving_frame.u_move_m_s))),
        (MOVE_SPEED_Y_NAME, repr(float(moving_frame.v_move_m_s))),
    ]
    return ("\n".join(f"{name}\t{value}" for name, value in rows) + "\n").encode("ascii")


def publish_initial_condition(
    source: Any,
    runtime_config: RuntimeConfig,
    directory: Path | str,
    *,
    moving_frame: Any,
) -> dict[str, Any]:
    """Write one model-time-zero frame from the host initializer and stop."""

    from .cartesian_history import _field_table_bytes, require_new_history_directory

    profile = cartesian_history_profile(runtime_config)
    directory = require_new_history_directory(directory)
    carriers, provenance = initial_condition_carriers(source, profile)
    directory.mkdir(parents=True)
    arena = write_forecast_arena_new(
        directory / ARENA_FILENAME,
        carriers,
        shapes=profile.shapes(),
        cupy=_HostArrayModule(),
    )
    table = _field_table_bytes(profile, arena)
    _write_new_fsynced(directory / FIELD_TABLE_FILENAME, table)
    scalars = _scalar_table_bytes(moving_frame)
    _write_new_fsynced(directory / SCALAR_TABLE_FILENAME, scalars)
    receipt = {
        "schema": INITIAL_CONDITION_SCHEMA,
        "status": INITIAL_CONDITION_STATUS,
        "step": 0,
        "model_time_s": 0.0,
        "dt_s": float(runtime_config.dt_s),
        "directory": os.fspath(directory),
        "arena": arena.record(),
        "field_table_path": os.fspath(directory / FIELD_TABLE_FILENAME),
        "field_table_sha256": hashlib.sha256(table).hexdigest(),
        "scalar_table_path": os.fspath(directory / SCALAR_TABLE_FILENAME),
        "scalar_table_sha256": hashlib.sha256(scalars).hexdigest(),
        "carrier_provenance": provenance,
        "domain_location_authority": DOMAIN_LOCATION_AUTHORITY,
        "domain_location_units": DOMAIN_LOCATION_UNITS,
        "device_opened": False,
        "claim_limit": INITIAL_CONDITION_CLAIM_LIMIT,
    }
    _write_new_fsynced(
        directory / RECEIPT_FILENAME,
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"),
    )
    return receipt
