"""Fail-closed access to GM1's source-derived physics support ledger."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from types import MappingProxyType
from typing import Any, Mapping


PHYSICS_SUPPORT_SCHEMA = "gm1.physics-support/v1"
_ADMITTED = frozenset({"admitted", "identity"})
_KNOWN_STATES = frozenset({"admitted", "identity", "operator-only", "partial", "unported"})


@dataclass(frozen=True, slots=True)
class PhysicsSupportRefusal(ValueError):
    """A stable, inspectable refusal for an unknown or unported branch."""

    code: str
    selector: str
    value: object
    reason: str

    def __str__(self) -> str:
        return f"{self.code}: {self.selector}={self.value!r}: {self.reason}"


def _record_bytes() -> bytes:
    return (
        resources.files("cm1core.records")
        .joinpath("GM1-PHYSICS-SUPPORT.json")
        .read_bytes()
    )


def load_physics_support_matrix() -> Mapping[str, Any]:
    """Load and minimally validate the packaged support record.

    The returned top-level mapping is immutable. Nested containers come from
    JSON and must be treated as read-only by callers.
    """

    payload = json.loads(_record_bytes())
    if payload.get("schema") != PHYSICS_SUPPORT_SCHEMA:
        raise RuntimeError("installed GM1 physics support record has an unknown schema")
    selectors = payload.get("selectors")
    if not isinstance(selectors, dict) or not selectors:
        raise RuntimeError("installed GM1 physics support record has no selectors")
    for name, selector in selectors.items():
        values = selector.get("values") if isinstance(selector, dict) else None
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"installed GM1 physics selector {name!r} has no values")
        for value, entry in values.items():
            state = entry.get("admission") if isinstance(entry, dict) else None
            if state not in _KNOWN_STATES:
                raise RuntimeError(
                    f"installed GM1 physics selector {name}={value} has invalid admission {state!r}"
                )
    return MappingProxyType(payload)


def _value_key(value: object) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        return value
    raise PhysicsSupportRefusal(
        "GM1-PHYSICS-VALUE-TYPE",
        "<unknown>",
        value,
        "branch values must be bool, int, or str",
    )


def branch_support(selector: str, value: object) -> Mapping[str, Any]:
    """Return one branch entry, refusing unknown selectors and values."""

    matrix = load_physics_support_matrix()
    selectors = matrix["selectors"]
    if selector not in selectors:
        raise PhysicsSupportRefusal(
            "GM1-PHYSICS-UNKNOWN-SELECTOR",
            selector,
            value,
            "selector is outside the source-audited physics inventory",
        )
    key = _value_key(value)
    values = selectors[selector]["values"]
    if key not in values:
        raise PhysicsSupportRefusal(
            "GM1-PHYSICS-UNKNOWN-VALUE",
            selector,
            value,
            "value is not a source-defined selectable branch",
        )
    return MappingProxyType(values[key])


def require_supported_physics(selection: Mapping[str, object]) -> None:
    """Fail closed unless every explicitly requested branch is admitted.

    This is intentionally a branch gate, not a complete namelist validator.
    Native cross-selector constraints remain recorded in the matrix and must
    also be enforced by any future arbitrary-configuration composer.
    """

    if not selection:
        raise PhysicsSupportRefusal(
            "GM1-PHYSICS-EMPTY-SELECTION",
            "<selection>",
            None,
            "an explicit source-selector mapping is required",
        )
    for selector, value in selection.items():
        entry = branch_support(selector, value)
        if entry["admission"] not in _ADMITTED:
            raise PhysicsSupportRefusal(
                "GM1-PHYSICS-UNPORTED-BRANCH",
                selector,
                value,
                f"admission is {entry['admission']!r}; a complete coupled family is required",
            )


__all__ = [
    "PHYSICS_SUPPORT_SCHEMA",
    "PhysicsSupportRefusal",
    "branch_support",
    "load_physics_support_matrix",
    "require_supported_physics",
]
