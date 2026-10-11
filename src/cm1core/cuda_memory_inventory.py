"""Inspect CUDA array ownership without importing CUDA or reading array data.

The inventory distinguishes the bytes exposed by views from the allocation
that those views keep alive.  Callers supply the live objects to inspect; this
module does not inspect frames, global variables, the memory pool, or the GPU.
It consequently reports a reachable-allocation lower bound, not device usage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import prod
from types import FunctionType, MethodType, ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class ArrayAllocation:
    """Metadata for one view and its owning allocation, in bytes."""

    device_id: int
    allocation_pointer: int
    allocation_bytes: int
    view_pointer: int
    view_bytes: int
    shape: tuple[int, ...]
    dtype: str


def cuda_array_allocation(value: object) -> ArrayAllocation | None:
    """Read the CuPy ownership protocol; never copy, synchronize, or allocate.

    Objects exposing CUDA array metadata without ``data.mem`` are refused:
    their view length does not establish how much storage they retain.
    """

    if not hasattr(type(value), "__cuda_array_interface__") and not hasattr(
        value, "__cuda_array_interface__"
    ):
        return None
    try:
        pointer = value.data  # type: ignore[attr-defined]
        memory = pointer.mem
        shape = tuple(int(extent) for extent in value.shape)  # type: ignore[attr-defined]
        result = ArrayAllocation(
            device_id=int(value.device.id),  # type: ignore[attr-defined]
            allocation_pointer=int(memory.ptr),
            allocation_bytes=int(memory.size),
            view_pointer=int(pointer.ptr),
            view_bytes=int(value.nbytes),  # type: ignore[attr-defined]
            shape=shape,
            dtype=str(value.dtype),  # type: ignore[attr-defined]
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "CUDA array ownership is unavailable; data.mem.ptr/size and device.id "
            "are required to avoid counting a view as an entire allocation"
        ) from error
    if (
        result.device_id < 0
        or result.allocation_bytes < 0
        or result.view_bytes < 0
        or any(extent < 0 for extent in shape)
        or (result.allocation_bytes and result.allocation_pointer <= 0)
    ):
        raise ValueError("CUDA allocation metadata contains invalid sizes or pointers")
    if result.view_bytes and not (
        result.allocation_pointer
        <= result.view_pointer
        < result.allocation_pointer + result.allocation_bytes
    ):
        raise ValueError("CUDA view pointer lies outside its reported owner allocation")
    return result


def _children(value: object) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        return [(f"[{str(key)!r}]", child) for key, child in value.items()]
    if isinstance(value, (tuple, list)):
        return [(f"[{index}]", child) for index, child in enumerate(value)]
    if is_dataclass(value) and not isinstance(value, type):
        return [(f".{field.name}", getattr(value, field.name)) for field in fields(value)]
    if isinstance(value, (ModuleType, FunctionType, MethodType, type)):
        return []
    # Object state only: do not enumerate or execute arbitrary properties.
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        return [(f".{key}", child) for key, child in namespace.items()]
    return []


def snapshot_cuda_allocations(
    roots: Mapping[str, object], *, label: str, maximum_objects: int = 200_000
) -> dict[str, Any]:
    """Count allocation identities across a caller-supplied live object graph.

    Each root is traversed independently, so allocations shared by two roots
    are attributed to both and never called exclusively releasable.  Cycles
    and repeated containers within a root are visited once.  No references to
    the supplied objects are retained in the returned JSON-compatible result.
    """

    if not isinstance(label, str) or not label:
        raise ValueError("label must be a nonempty string")
    if type(maximum_objects) is not int or maximum_objects < 1:
        raise ValueError("maximum_objects must be a positive integer")
    allocations: dict[tuple[int, int], dict[str, Any]] = {}
    unique_views: dict[int, int] = {}
    object_visits = 0
    for root, initial in roots.items():
        if not isinstance(root, str) or not root:
            raise ValueError("root names must be nonempty strings")
        seen: set[int] = set()
        stack = [(root, initial)]
        while stack:
            path, value = stack.pop()
            object_visits += 1
            if object_visits > maximum_objects:
                raise ValueError("CUDA ownership traversal exceeded maximum_objects")
            allocation = cuda_array_allocation(value)
            if allocation is not None:
                key = (allocation.device_id, allocation.allocation_pointer)
                record = allocations.setdefault(
                    key,
                    {
                        "device_id": allocation.device_id,
                        "allocation_pointer": allocation.allocation_pointer,
                        "allocation_bytes": allocation.allocation_bytes,
                        "roots": set(),
                        "views": {},
                    },
                )
                if record["allocation_bytes"] != allocation.allocation_bytes:
                    raise ValueError("one CUDA allocation pointer has conflicting owner sizes")
                record["roots"].add(root)
                view = record["views"].setdefault(
                    id(value),
                    {
                        "view_pointer": allocation.view_pointer,
                        "view_bytes": allocation.view_bytes,
                        "shape": list(allocation.shape),
                        "dtype": allocation.dtype,
                        "paths": [],
                    },
                )
                view["paths"].append(path)
                unique_views[id(value)] = allocation.view_bytes
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            stack.extend((path + suffix, child) for suffix, child in _children(value))
    rows = []
    per_root = {name: {"reachable_bytes": 0, "exclusive_bytes": 0} for name in roots}
    for key in sorted(allocations):
        record = allocations[key]
        root_names = sorted(record["roots"])
        for root in root_names:
            per_root[root]["reachable_bytes"] += record["allocation_bytes"]
            if len(root_names) == 1:
                per_root[root]["exclusive_bytes"] += record["allocation_bytes"]
        rows.append(
            {
                **record,
                "roots": root_names,
                "views": sorted(record["views"].values(), key=lambda view: view["paths"][0]),
            }
        )
    return {
        "schema": "gm1.cuda-reachable-allocation-inventory/v1",
        "label": label,
        "allocation_count": len(rows),
        "allocation_bytes": sum(row["allocation_bytes"] for row in rows),
        "distinct_view_count": len(unique_views),
        "distinct_view_bytes": sum(unique_views.values()),
        "roots": per_root,
        "allocations": rows,
        "claim_limit": (
            "caller-supplied object graphs only; excludes unreferenced pool cache, "
            "driver/context allocations, unexposed library workspaces, and omitted "
            "stack locals; exclusive bytes assume no owners outside supplied roots"
        ),
    }


def shape_bytes(shape: tuple[int, ...], *, itemsize: int = 4) -> int:
    """Count a declared array payload without allocating it."""

    if type(itemsize) is not int or itemsize < 1:
        raise ValueError("itemsize must be a positive integer")
    if any(type(extent) is not int or extent < 0 for extent in shape):
        raise ValueError("shape extents must be nonnegative integers")
    return prod(shape) * itemsize
