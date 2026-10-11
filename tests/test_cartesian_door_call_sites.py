"""Every internal keyword call site agrees with the signature it calls.

``run_cartesian`` passed ``cadence=`` to ``_execute`` while ``_execute``
accepted only ``schedule=``.  Nothing caught it, because the call is reached
only after the ArWen estate is admitted, a CUDA device is opened and the
initializer has run -- so every unit test stopped short of it and every
integration died with ``TypeError: _execute() got an unexpected keyword
argument 'cadence'`` at the one moment that costs a GPU reservation.

These checks are pure ``ast``: no import, no device, no estate.  They compare
each keyword-only signature in the door against every call site of that name
in the same module, which is exactly the mistake that was made and exactly
the shape of mistake a merge between two branches that both edited a long
keyword-only parameter list will make again.
"""

from __future__ import annotations

import ast
from pathlib import Path

DOOR = Path(__file__).resolve().parents[1] / "src" / "cm1core" / "cartesian_door.py"


def _module() -> ast.Module:
    return ast.parse(DOOR.read_text(encoding="utf-8"))


def _keyword_only_signatures(tree: ast.Module) -> dict[str, tuple[set[str], set[str]]]:
    """Map each function name to (accepted keyword names, required ones)."""

    signatures: dict[str, tuple[set[str], set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.args.kwarg is not None:
            continue  # **kwargs accepts anything; there is nothing to check
        accepted = {a.arg for a in node.args.kwonlyargs}
        accepted |= {a.arg for a in node.args.args}
        accepted |= {a.arg for a in node.args.posonlyargs}
        required = {
            a.arg
            for a, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if default is None
        }
        signatures[node.name] = (accepted, required)
    return signatures


def _named_calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    return [
        (node.func.id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_no_call_site_passes_a_keyword_its_target_does_not_accept() -> None:
    tree = _module()
    signatures = _keyword_only_signatures(tree)
    offences: list[str] = []
    for name, call in _named_calls(tree):
        if name not in signatures:
            continue
        accepted, _ = signatures[name]
        passed = {kw.arg for kw in call.keywords if kw.arg is not None}
        extra = passed - accepted
        if extra:
            offences.append(f"{name}() at line {call.lineno} passes {sorted(extra)}")
    assert not offences, (
        "a keyword-only call site in cartesian_door.py names a parameter its "
        "target does not accept; this raises TypeError only when the call is "
        "actually reached, which for _execute is after a CUDA device is open: "
        + "; ".join(offences)
    )


def test_no_call_site_omits_a_keyword_its_target_requires() -> None:
    """The mirror image: a required keyword-only parameter left unpassed.

    A call that splats (``**something``) may supply it, so those are skipped
    rather than guessed at.
    """

    tree = _module()
    signatures = _keyword_only_signatures(tree)
    offences: list[str] = []
    for name, call in _named_calls(tree):
        if name not in signatures:
            continue
        if any(kw.arg is None for kw in call.keywords):
            continue
        if call.args or any(isinstance(a, ast.Starred) for a in call.args):
            continue
        _, required = signatures[name]
        passed = {kw.arg for kw in call.keywords if kw.arg is not None}
        missing = required - passed
        if missing:
            offences.append(f"{name}() at line {call.lineno} omits {sorted(missing)}")
    assert not offences, (
        "a call site omits a keyword-only parameter that has no default: "
        + "; ".join(offences)
    )


def test_execute_is_the_case_this_file_exists_for() -> None:
    """Name the specific regression, so a rename cannot quietly drop it."""

    tree = _module()
    signatures = _keyword_only_signatures(tree)
    assert "_execute" in signatures, "cartesian_door._execute has been renamed"
    accepted, _ = signatures["_execute"]
    assert "schedule" in accepted
    # RunSchedule carries the cadence; there is no second cadence parameter.
    assert "cadence" not in accepted, (
        "_execute accepts a 'cadence' again; RunSchedule.cadence is the one "
        "carrier of the history cadence inside the runtime entry point"
    )
    calls = [call for name, call in _named_calls(tree) if name == "_execute"]
    assert calls, "nothing calls _execute"
    for call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "cadence" not in passed, f"line {call.lineno} passes cadence= again"
        assert "schedule" in passed, f"line {call.lineno} does not pass the schedule"
