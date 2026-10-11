"""Versioned, CM1-owned engine lifecycle types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


ENGINE_CONTRACT = "gpuwm-cm1.engine/v1"
HISTORY_SCHEMA = "gpuwm-cm1.history/v1"
RESTART_SCHEMA = "gpuwm-cm1.restart/v1"
RECEIPT_SCHEMA = "gpuwm-cm1.run-receipt/v1"


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    cm1_release: str
    cm1_authority_sha256: str
    arwen_requirement: str
    arwen_seam_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SealedRun:
    config_sha256: str
    source: SourceIdentity
    output_root: Path
    contract: str = ENGINE_CONTRACT


@runtime_checkable
class CM1Engine(Protocol):
    """Capability contract; it deliberately assumes no WRF/MPAS geometry."""

    def preflight(self, run: SealedRun) -> dict[str, object]: ...

    def initialize(self, run: SealedRun) -> object: ...

    def step(self, state: object) -> None: ...

    def write_history(self, state: object, destination: Path) -> dict[str, object]: ...

    def checkpoint(self, state: object, destination: Path) -> dict[str, object]: ...

    def restore(self, checkpoint: Path) -> object: ...

    def finalize(self, state: object) -> dict[str, object]: ...


__all__ = [
    "CM1Engine",
    "ENGINE_CONTRACT",
    "HISTORY_SCHEMA",
    "RECEIPT_SCHEMA",
    "RESTART_SCHEMA",
    "SealedRun",
    "SourceIdentity",
]

