"""The post-physics EOS/qa finish for a column component other than WSM6.

``cuda_gm1_tilestream_post_wsm6`` is sealed: its module bytes are pinned by the
sealed device evidence receipt (``tests/test_gm1_tilestream_post_wsm6_evidence.py``), so
it is not edited.  This module reuses that module's compiled image, its device
inventory and snapshot checks, and its kernels unchanged, and replaces exactly
one check: which column launch may precede the finish.

The arithmetic is CM1's own: ``src/mp_driver.F:603-617`` rebuilds
prs/pp3d/th3d from rho, the new temperature and ``q3d(nqv)``.  It is *not*
component-independent: holding rho fixed is CM1's pairing only for a
temperature heated at constant volume (``src/morrison.F:1640-1650``,
``:6402-6504``), and every ArWen scheme heats at constant pressure.  The
Cartesian lane therefore runs :mod:`cm1core.cuda_cartesian_moist_heating`
between the component and this finish.  The finish kernel reads q3d rows 0..5
at fixed offsets and is handed CM1's six mass rows, so a number concentration
never enters the EOS.
The kernel's six-row copy into the block's ``qa`` scratch is the sealed
image's; the Cartesian lane does not publish it, because ``src/solve3.F:
624-643`` -- which runs after ``mp_driver`` -- is what equates qa to q3d, for
every row, and ``CartesianFinish.solve3`` does that.

A WSM6 run never reaches this module: ``cuda_cartesian_finish`` calls the
sealed launcher for ``wsm6-mp6``, so its receipts are unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from . import cuda_gm1_tilestream_post_wsm6 as sealed
from .cuda_gm1_tilestream_column import (
    TILESTREAM_RECORD_IDENTITY,
    CudaGm1TiledColumnLaunch,
)
from .moisture_species import CM1_MASS_PREFIX_NAMES, moisture_species_profile


def column_launch_identity(receipt: Mapping[str, Any]) -> str | None:
    """The non-WSM6 component a launch receipt names exactly, or ``None``.

    The receipt must carry one admitted component's schema, ABI and status
    triple exactly, so a launch record of one scheme relabelled as another is
    refused; and that component's axis must lead with CM1's six mass rows,
    because those are the offsets the finish kernel reads.
    """

    for component_id, identity in TILESTREAM_RECORD_IDENTITY.items():
        if component_id == "wsm6-mp6":
            continue
        if (
            receipt.get("schema") == identity["schema"]
            and receipt.get("abi") == identity["abi"]
            and receipt.get("status") == identity["status"]
        ):
            profile = moisture_species_profile(component_id)
            if profile.names[: len(CM1_MASS_PREFIX_NAMES)] != CM1_MASS_PREFIX_NAMES:
                return None
            return component_id
    return None


def _require_column_launch(
    launch: CudaGm1TiledColumnLaunch,
    values: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    active_nx: int,
    active_ny: int,
    block_index: int,
    source_identity_sha256: str,
    stream_pointer: int,
) -> tuple[str, str]:
    if not isinstance(launch, CudaGm1TiledColumnLaunch):
        raise TypeError("column_launch must be CudaGm1TiledColumnLaunch")
    receipt = launch.receipt
    component_id = column_launch_identity(receipt)
    if (
        component_id is None
        or receipt.get("fixed_shape_zyx") != [nk, ny, nx]
        or receipt.get("active_shape_yx") != [active_ny, active_nx]
        or receipt.get("block_index") != block_index
        or receipt.get("source_identity_sha256") != source_identity_sha256
        or receipt.get("stream_pointer") != stream_pointer
        or receipt.get("same_stream_no_host_sync") is not True
    ):
        raise ValueError("column launch identity/order differs")
    if any(
        int(launch.arrays[name].data.ptr) != int(values[name].data.ptr)
        for name in values
    ):
        raise ValueError("column launch did not retain the exact device arrays")
    if launch.ready_event is None:
        raise ValueError("column launch has no CUDA event")
    return component_id, sealed._require_token(receipt, label="column launch")


def launch_cuda_gm1_tiled_post_column_eos_qa(
    arrays: Mapping[str, Any],
    aux: Mapping[str, Any],
    *,
    snapshot: sealed.CudaGm1TiledPostWsm6Snapshot,
    column_launch: CudaGm1TiledColumnLaunch,
    nx: int,
    ny: int,
    nk: int,
    source_identity_sha256: str,
    column_receipt_sha256: str,
    block_index: int,
    active_nx: int,
    active_ny: int,
    stream: Any,
    compiled: sealed.CompiledGm1TiledPostWsm6 | None = None,
) -> sealed.CudaGm1TiledPostWsm6Launch:
    """Enqueue the sealed conditional EOS and qa copy after a non-WSM6 column."""

    nx = sealed._positive_int("nx", nx)
    ny = sealed._positive_int("ny", ny)
    nk = sealed._positive_int("nk", nk)
    active_nx = sealed._positive_int("active_nx", active_nx)
    active_ny = sealed._positive_int("active_ny", active_ny)
    if active_nx > nx or active_ny > ny:
        raise ValueError("active post-physics rectangle exceeds the fixed block")
    block_index = sealed._nonnegative_int("block_index", block_index)
    source_identity_sha256 = sealed._require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    column_receipt_sha256 = sealed._require_sha256(
        "column_receipt_sha256", column_receipt_sha256
    )
    cupy = importlib.import_module("cupy")
    values, scratch = sealed._require_device_mapping(
        cupy, arrays, aux, nx=nx, ny=ny, nk=nk
    )
    compiled = snapshot.compiled if compiled is None else compiled
    if compiled.target_arch != snapshot.compiled.target_arch:
        raise ValueError("snapshot and finish target architectures differ")
    stream_pointer = int(stream.ptr)
    snapshot_token = sealed._require_snapshot(
        snapshot, values, scratch, nx=nx, ny=ny, nk=nk,
        active_nx=active_nx, active_ny=active_ny, block_index=block_index,
        source_identity_sha256=source_identity_sha256,
        wsm6_receipt_sha256=column_receipt_sha256,
        stream_pointer=stream_pointer,
    )
    component_id, column_token = _require_column_launch(
        column_launch, values, nx=nx, ny=ny, nk=nk,
        active_nx=active_nx, active_ny=active_ny, block_index=block_index,
        source_identity_sha256=source_identity_sha256,
        stream_pointer=stream_pointer,
    )
    n = nx * ny * nk
    threads = 256
    with stream:
        # The sealed finish kernel, with the sealed argument order.
        compiled.finish_kernel(
            ((n + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                cupy.uint64(n), values["rho"], values["pi0"], values["th0"],
                values["p"], values["pp3d"], values["th3d"], values["q3d"],
                values["thp"], scratch["qa"], scratch["temperature_before"],
                scratch["qv_before"], scratch["pp3d_before"],
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record: dict[str, Any] = {
        "schema": "gm1.cuda-tilestream-post-column/v1",
        "abi": sealed.CUDA_GM1_TILESTREAM_POST_WSM6_ABI,
        "status": "ENQUEUED_POST_COLUMN_EOS_AND_PHYSICAL_QA",
        "column_component_id": component_id,
        "block_index": block_index,
        "fixed_shape_zyx": [nk, ny, nx],
        "active_shape_yx": [active_ny, active_nx],
        "source_identity_sha256": source_identity_sha256,
        "column_receipt_sha256": column_receipt_sha256,
        "snapshot_operation_token_sha256": snapshot_token,
        "column_operation_token_sha256": column_token,
        "stream_pointer": stream_pointer,
        "same_stream_no_host_sync": True,
        "thresholds_f32": {"tsmall": sealed.CM1_TSMALL_F32, "qsmall": sealed.CM1_QSMALL_F32},
        "source_order": [
            "old temperature/qv/pp3d snapshot before the column component",
            f"accepted tile-local public {component_id} column component on the same stream",
            "conditional ptype-5 eqtset-2 prs/pp3d/th3d rebuild",
            "raw q3d[0:6] to physical qa[0:6] publication",
        ],
        "mass_prefix_only": (
            "q3d[0:6] is CM1's qv,qc,qr,qi,qs,qg; the component's number "
            "concentrations never enter the EOS.  qa[0:6] here is block "
            "scratch: the Cartesian lane publishes every row to the domain qa "
            "in solve3 (src/solve3.F:624-643), not at this boundary"
        ),
        "mutated": ["p", "pp3d", "th3d", "qa[0:6]"],
        "rho_fixed_across_component_and_eos": True,
        "qa_raw_copy_of_q3d": True,
        "bindings": {
            "column": sealed._pointer_inventory(values),
            "auxiliary": sealed._pointer_inventory(scratch),
        },
        "kernel": {
            "source_sha256": sealed.CUDA_GM1_TILESTREAM_POST_WSM6_SOURCE_SHA256,
            "glibc_powf_device_source_sha256": sealed.CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "nvrtc_language_standard": "c++17",
        },
        "authority": {
            "tarball_sha256": sealed.AUTHORITY_TARBALL_SHA256,
            "mp_driver_source_sha256": sealed.AUTHORITY_MP_DRIVER_SHA256,
            "solve3_source_sha256": sealed.AUTHORITY_SOLVE3_SHA256,
            "anchors": [
                "src/mp_driver.F:603-617",
                "src/solve3.F:624-643 physical copy only",
                "src/cm1.F:312-314 tsmall/qsmall selected-real constants",
            ],
        },
        "claim_limit": sealed.GM1_TILESTREAM_POST_WSM6_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(sealed._canonical(record)).hexdigest()
    return sealed.CudaGm1TiledPostWsm6Launch(
        event, compiled, MappingProxyType(values), MappingProxyType(scratch),
        MappingProxyType(record),
    )


__all__ = [
    "column_launch_identity",
    "launch_cuda_gm1_tiled_post_column_eos_qa",
]
