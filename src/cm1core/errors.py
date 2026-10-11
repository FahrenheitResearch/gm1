"""Typed refusal surface for every user-facing door.

:func:`refuse` stops at the first wrong setting.  A document reader that wants
every wrong setting in one pass (the namelist importer's coverage report) runs
the same checks inside :func:`collect_refusals`:

* a refusal a *value check* raises -- one setting judged on its own value
  (``refuse(..., independent=True)``: ``config.py``'s ``_choice``,
  ``_integer`` and the other single-value helpers) -- is recorded, and the
  check returns the value it was given, so the checks after it still run;
* any other refusal judges settings against each other, so what follows it may
  read a value it just refused.  It is recorded and ends the pass;
* at the end of the block every recorded refusal is raised at once as one
  :class:`CM1Refusals`, in the order the checks met them.  Its ``detail`` is
  the first one, which is exactly the refusal the same checks raise outside
  the block, so a caller that only knows :class:`CM1Refusal` sees no change.

A refusal recorded after an earlier one in the same pass carries
``after_earlier=True`` unless it is a value check: it was met on a document
that still held the earlier refusal's value, so it may change once that is
fixed.  :func:`fail_fast` suspends collection for a block that catches a
refusal to translate it or to fall back.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if isinstance(value, (datetime, date, time)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return {"type": type(value).__name__, "value": repr(value)}


@dataclass(frozen=True, slots=True)
class RefusalDetail:
    code: str
    gate: str
    reason: str
    remedy: str
    option: str | None = None
    value: object | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "refused",
            "code": self.code,
            "gate": self.gate,
            "reason": self.reason,
            "remedy": self.remedy,
        }
        if self.option is not None:
            payload["option"] = self.option
            payload["value"] = _json_safe(self.value)
        return payload


class CM1Refusal(RuntimeError):
    """An intentional stop that names the concrete invalid execution path."""

    def __init__(self, detail: RefusalDetail):
        self.detail = detail
        message = f"{detail.code} [{detail.gate}]: {detail.reason}"
        if detail.option is not None:
            message += f" (option {detail.option}={detail.value!r})"
        message += f" Remedy: {detail.remedy}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CollectedRefusal:
    """One refusal met inside :func:`collect_refusals`.

    ``independent`` says it came from a value check (one setting judged on its
    own).  ``after_earlier`` says it is a check across settings met after an
    earlier refusal of the same pass.  ``ended_pass`` says the checks stopped
    at it.
    """

    detail: RefusalDetail
    independent: bool
    after_earlier: bool
    ended_pass: bool = False

    def as_dict(self) -> dict[str, object]:
        payload = self.detail.as_dict()
        payload["independent"] = self.independent
        payload["after_earlier"] = self.after_earlier
        payload["ended_pass"] = self.ended_pass
        return payload


class CM1Refusals(CM1Refusal):
    """Every refusal one pass of checks met, first one first.

    ``detail`` is the first refusal, so ``except CM1Refusal`` callers read the
    same stop the checks raise outside :func:`collect_refusals`.
    ``interrupted`` holds the exception that ended the pass when it was not a
    refusal: a check that cannot run on a value refused above it.
    """

    def __init__(
        self,
        refusals: Sequence[CollectedRefusal],
        *,
        interrupted: BaseException | None = None,
    ):
        if not refusals:
            raise ValueError("CM1Refusals needs at least one refusal")
        self.refusals: tuple[CollectedRefusal, ...] = tuple(refusals)
        self.interrupted = interrupted
        super().__init__(self.refusals[0].detail)
        if len(self.refusals) > 1:
            self.args = (
                f"{len(self.refusals)} refusals; first: {self.args[0]}",
            )

    @property
    def details(self) -> tuple[RefusalDetail, ...]:
        return tuple(row.detail for row in self.refusals)


@dataclass(slots=True)
class RefusalCollector:
    """The refusals recorded so far by one :func:`collect_refusals` block."""

    refusals: list[CollectedRefusal] = field(default_factory=list)
    interrupted: BaseException | None = None
    suspended: int = 0

    def record(self, detail: RefusalDetail, *, independent: bool, ended_pass: bool = False) -> None:
        for index, row in enumerate(self.refusals):
            if row.detail == detail:
                if ended_pass and not row.ended_pass:
                    self.refusals[index] = CollectedRefusal(
                        detail=row.detail, independent=row.independent,
                        after_earlier=row.after_earlier, ended_pass=True,
                    )
                return
        self.refusals.append(CollectedRefusal(
            detail=detail,
            independent=independent,
            after_earlier=bool(self.refusals) and not independent,
            ended_pass=ended_pass,
        ))


_STATE = threading.local()


def _collector() -> RefusalCollector | None:
    stack = getattr(_STATE, "stack", None)
    if not stack:
        return None
    return stack[-1]


@contextmanager
def collect_refusals() -> Iterator[RefusalCollector]:
    """Run checks and raise every refusal they meet as one :class:`CM1Refusals`.

    Inside the block a value check records its refusal and returns; any other
    refusal is recorded and ends the block.  An exception that is not a
    refusal also ends it: after at least one refusal it is kept as
    ``CM1Refusals.interrupted`` (a later check could not run on a refused
    value), and with none it propagates unchanged.  A block that meets no
    refusal behaves exactly as the checks do outside it.
    """

    collector = RefusalCollector()
    stack = getattr(_STATE, "stack", None)
    if stack is None:
        stack = _STATE.stack = []
    stack.append(collector)
    try:
        yield collector
    except CM1Refusals as aggregate:
        for row in aggregate.refusals:
            collector.record(row.detail, independent=row.independent, ended_pass=row.ended_pass)
        if aggregate.interrupted is not None and collector.interrupted is None:
            collector.interrupted = aggregate.interrupted
    except CM1Refusal as refusal:
        collector.record(refusal.detail, independent=False, ended_pass=True)
    except Exception as error:  # noqa: BLE001 - kept, see the docstring
        if not collector.refusals:
            raise
        collector.interrupted = error
    finally:
        stack.pop()
    if collector.refusals:
        raise CM1Refusals(collector.refusals, interrupted=collector.interrupted)


@contextmanager
def fail_fast() -> Iterator[None]:
    """Suspend :func:`collect_refusals` for a block that catches refusals.

    A ``try`` that turns one refusal into another, or falls back when a
    refusal is raised, needs the refusal raised; inside this block every
    refusal is, whatever the enclosing collection does.
    """

    collector = _collector()
    if collector is None:
        yield
        return
    collector.suspended += 1
    try:
        yield
    finally:
        collector.suspended -= 1


def collecting() -> bool:
    """Whether a value check here records its refusal instead of raising it."""

    collector = _collector()
    return collector is not None and collector.suspended == 0


def refuse(
    *,
    code: str,
    gate: str,
    reason: str,
    remedy: str,
    option: str | None = None,
    value: object | None = None,
    independent: bool = False,
) -> "NoReturn":
    """Stop with the named wrong result.

    ``independent=True`` marks a value check: one setting judged on its own
    value.  Inside :func:`collect_refusals` such a refusal is recorded and
    ``refuse`` returns, so the caller must return the value it was given (the
    config helpers do).  Everywhere else it raises exactly as before.
    """

    from typing import NoReturn  # noqa: F401

    detail = RefusalDetail(
        code=code,
        gate=gate,
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )
    if independent:
        collector = _collector()
        if collector is not None and collector.suspended == 0:
            collector.record(detail, independent=True)
            return  # type: ignore[return-value]
    raise CM1Refusal(detail)


__all__ = [
    "CM1Refusal",
    "CM1Refusals",
    "CollectedRefusal",
    "RefusalCollector",
    "RefusalDetail",
    "collect_refusals",
    "collecting",
    "fail_fast",
    "refuse",
]
