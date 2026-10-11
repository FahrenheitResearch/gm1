"""The value rules ``config.py`` applies to one setting at a time, as a record.

``records/CM1-VALUE-RULES.json`` lists every ``_choice``, ``_integer_choice``
and ``_optional_integer_choice`` site in ``config.py``: the gpuwm-cm1 option
it judges, the values it admits, the reason it gives, and when it applies.
:func:`extract_value_rules` reads them from ``config.py``'s own syntax tree
(``tools/generate_cm1_value_rules.py`` writes the record, and
``tests/test_value_rules_parity.py`` holds the record against the code), so
the importer can say which values of a setting run before anything is run,
and the GUI can grey out a value with the reason ``config.py`` would give.

``applies`` is one of:

``cartesian``
    every Cartesian-lane document is judged by it (the site is not inside a
    condition in a function every such document reaches);
``when-declared``
    as ``cartesian``, but only when the document declares the key
    (``_optional_integer_choice``, or a ``"key" in table`` guard);
``conditional``
    the site sits under another condition (a lane, another setting, an early
    return), so only the full check can say whether it applies;
``dynamic``
    the option or the admitted values are computed at run time.

Only ``cartesian`` and ``when-declared`` rules are judged at key level
(:func:`judge`); the others are met by ``check`` itself.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RECORD_NAME = "CM1-VALUE-RULES.json"
SCHEMA = "gm1.cm1-value-rules/v1"

#: The helpers whose calls are value rules, with the argument positions of
#: (option, value, allowed, reason).  ``_optional_integer_choice`` takes the
#: table instead of the value and applies only when the key is declared.
CHOICE_HELPERS: Mapping[str, tuple[int, int, int, int]] = {
    "_choice": (0, 1, 2, 3),
    "_integer_choice": (0, 1, 2, 3),
    "_optional_integer_choice": (0, 1, 2, 3),
}
INTEGER_HELPERS = frozenset({"_integer_choice", "_optional_integer_choice"})

#: Functions every Cartesian-lane document reaches: ``parse_mapping`` and its
#: table parsers, the validators ``_validate_first_slice`` calls without a
#: condition, and the Cartesian lane's own requirements.  A rule outside them
#: is ``conditional``.
CARTESIAN_FUNCTIONS = frozenset({
    "parse_mapping",
    "_parse_runtime",
    "_parse_initialization",
    "_parse_diagnostics",
    "_parse_output",
    "_parse_lower_boundary",
    "_parse_surface",
    "_parse_surface_layer",
    "_parse_radiation",
    "_parse_lagrangian",
    "_parse_turbulence_scale",
    "_parse_sgs_startup_ramp",
    "_parse_adaptive_move",
    "_parse_horizontal_damping",
    "_parse_open_companions",
    "_parse_tornado_diagnostics",
    "_parse_lspgrad",
    "_validate_output_against_lane",
    "_validate_physics_families",
    "_validate_first_slice",
    "_validate_reflectivity",
    "_validate_lower_boundary",
    "_validate_dynamics_words",
    "_validate_surface",
    "_validate_diagnostics",
    "_validate_boundary_family",
    "_validate_horizontal_damping",
    "_require_cartesian_physics",
    "_require_cartesian_envelope",
})

#: ``if "key" not in table: return`` and ``if not (_KEYS & set(table)): return``:
#: the rules after it apply whenever the key they judge is declared.
_PRESENCE = re.compile(r'^(?:"\w+" not in \w+|not \(\w+ & set\(\w+\)\))$')

_SAFE_BUILTINS = {
    "set": set, "frozenset": frozenset, "range": range, "tuple": tuple,
    "list": list, "sorted": sorted, "str": str, "int": int,
}


@dataclass(frozen=True)
class ValueRule:
    """One value rule: ``option`` must hold one of ``allowed``."""

    option: str
    allowed: tuple[Any, ...]
    integer: bool
    reason: str
    function: str
    applies: str
    guard: str | None = None

    @property
    def key_level(self) -> bool:
        return self.applies in ("cartesian", "when-declared")

    def admits(self, value: Any) -> bool:
        if self.integer and (isinstance(value, bool) or not isinstance(value, int)):
            return False
        try:
            return value in set(self.allowed)
        except TypeError:
            return False

    def record(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "option": self.option,
            "allowed": list(self.allowed),
            "integer": self.integer,
            "reason": self.reason,
            "function": self.function,
            "applies": self.applies,
        }
        if self.guard is not None:
            row["guard"] = self.guard
        return row


@dataclass(frozen=True)
class RuleBreach:
    """A document value a key-level rule refuses."""

    rule: ValueRule
    value: Any
    allowed: tuple[Any, ...]

    @property
    def option(self) -> str:
        return self.rule.option

    @property
    def code(self) -> str:
        # config._integer_choice first asks config._integer for an integer >= 0.
        value = self.value
        if self.rule.integer and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            return "CM1-CFG-INTEGER"
        return "CM1-CFG-UNSUPPORTED-VALUE"

    @property
    def remedy(self) -> str:
        return "select one of " + ", ".join(map(repr, self.allowed))


# ---------------------------------------------------------------------------
# Extraction from config.py
# ---------------------------------------------------------------------------


def _config_source() -> tuple[str, str]:
    path = Path(__file__).resolve().parents[1] / "config.py"
    return str(path), path.read_text(encoding="utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"{value!r} is not a JSON scalar")


def _sorted_values(values: Iterable[Any]) -> tuple[Any, ...]:
    items = [_json_value(value) for value in values]

    def order(item: Any) -> tuple[int, Any, str]:
        if isinstance(item, bool):
            return (0, int(item), "")
        if isinstance(item, (int, float)):
            return (1, item, "")
        return (2, 0, str(item))

    return tuple(sorted(items, key=order))


class _Extractor:
    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        from .. import config as config_module

        self.namespace = dict(vars(config_module))
        self.namespace["__builtins__"] = _SAFE_BUILTINS
        self.rules: list[ValueRule] = []

    def text(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ast.unparse(node)

    def evaluate(self, node: ast.AST, bindings: Mapping[str, ast.AST]) -> Any:
        """Evaluate ``node`` over config.py's module names and literal loop bindings."""

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, name: ast.Name) -> ast.AST:  # noqa: N802
                if isinstance(name.ctx, ast.Load) and name.id in bindings:
                    return bindings[name.id]
                return name

        expression = ast.Expression(body=Substitute().visit(ast.fix_missing_locations(
            ast.parse(ast.unparse(node), mode="eval").body)))
        ast.fix_missing_locations(expression)
        return eval(compile(expression, "<config.py value rule>", "eval"), dict(self.namespace))  # noqa: S307

    def run(self) -> list[ValueRule]:
        for function in self.tree.body:
            if isinstance(function, ast.FunctionDef):
                if function.name in CHOICE_HELPERS or function.name == "_integer_choice":
                    continue
                self._walk(function, function.body, [], {}, early_return=None)
        return self.rules

    def _walk(
        self,
        function: ast.FunctionDef,
        body: Sequence[ast.stmt],
        guards: list[tuple[str, str]],
        bindings: dict[str, ast.AST],
        *,
        early_return: str | None,
    ) -> None:
        returned = early_return
        for statement in body:
            self._statement(function, statement, guards, bindings, early_return=returned)
            kind = self._returns_under_condition(statement)
            if kind == "other" or (kind == "presence" and returned is None):
                returned = kind
        return None

    def _returns_under_condition(self, statement: ast.stmt) -> str | None:
        """``presence`` for ``if <table or key absent>: return``, ``other`` for any
        other return under a condition, else None."""

        if not isinstance(statement, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            return None
        if not any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            return None
        if isinstance(statement, ast.If) and not statement.orelse and _PRESENCE.match(self.text(statement.test)):
            return "presence"
        return "other"

    def _statement(
        self,
        function: ast.FunctionDef,
        statement: ast.stmt,
        guards: list[tuple[str, str]],
        bindings: dict[str, ast.AST],
        *,
        early_return: str | None,
    ) -> None:
        if isinstance(statement, ast.If):
            self._expressions(function, statement.test, guards, bindings, early_return)
            guard = ("if", self.text(statement.test))
            self._walk(function, statement.body, guards + [guard], bindings, early_return=early_return)
            negated = ("else", "not (" + self.text(statement.test) + ")")
            self._walk(function, statement.orelse, guards + [negated], bindings, early_return=early_return)
            return
        if isinstance(statement, ast.For):
            expansions = self._literal_loop(statement)
            if expansions is None:
                guard = ("for", self.text(statement.iter))
                self._walk(function, statement.body, guards + [guard], bindings, early_return=early_return)
            else:
                for extra in expansions:
                    merged = dict(bindings)
                    merged.update(extra)
                    self._walk(function, statement.body, guards, merged, early_return=early_return)
            return
        if isinstance(statement, (ast.While, ast.With, ast.Try)):
            label = type(statement).__name__.lower()
            inner: list[ast.stmt] = []
            for name in ("body", "orelse", "finalbody"):
                inner.extend(getattr(statement, name, []) or [])
            for handler in getattr(statement, "handlers", []) or []:
                inner.extend(handler.body)
            if isinstance(statement, ast.With):
                self._walk(function, statement.body, guards, bindings, early_return=early_return)
            else:
                self._walk(function, inner, guards + [(label, label)], bindings, early_return=early_return)
            return
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        self._expressions(function, statement, guards, bindings, early_return)

    def _literal_loop(self, loop: ast.For) -> list[dict[str, ast.AST]] | None:
        """``for a, b in ((x, y), ...)`` over literal tuples: one binding per row."""

        if not isinstance(loop.iter, (ast.Tuple, ast.List)):
            return None
        targets: list[str]
        if isinstance(loop.target, ast.Name):
            targets = [loop.target.id]
        elif isinstance(loop.target, ast.Tuple) and all(isinstance(t, ast.Name) for t in loop.target.elts):
            targets = [t.id for t in loop.target.elts]  # type: ignore[union-attr]
        else:
            return None
        rows: list[dict[str, ast.AST]] = []
        for element in loop.iter.elts:
            if len(targets) == 1:
                rows.append({targets[0]: element})
                continue
            if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) != len(targets):
                return None
            rows.append(dict(zip(targets, element.elts)))
        return rows

    def _expressions(
        self,
        function: ast.FunctionDef,
        node: ast.AST,
        guards: list[tuple[str, str]],
        bindings: Mapping[str, ast.AST],
        early_return: str | None,
    ) -> None:
        # Walk the expression tree, noting IfExp conditions on the way down.
        stack: list[tuple[ast.AST, list[tuple[str, str]]]] = [(node, guards)]
        while stack:
            current, current_guards = stack.pop()
            if isinstance(current, ast.IfExp):
                stack.append((current.test, current_guards))
                stack.append((current.body, current_guards + [("ifexp", self.text(current.test))]))
                stack.append((current.orelse, current_guards + [("ifexp-else", self.text(current.test))]))
                continue
            if isinstance(current, (ast.Lambda, ast.FunctionDef)):
                continue
            if (
                isinstance(current, ast.Call)
                and isinstance(current.func, ast.Name)
                and current.func.id in CHOICE_HELPERS
            ):
                self._rule(function, current, current_guards, bindings, early_return)
            for child in ast.iter_child_nodes(current):
                stack.append((child, current_guards))

    def _rule(
        self,
        function: ast.FunctionDef,
        call: ast.Call,
        guards: list[tuple[str, str]],
        bindings: Mapping[str, ast.AST],
        early_return: str | None,
    ) -> None:
        helper = call.func.id  # type: ignore[union-attr]
        option_at, value_at, allowed_at, reason_at = CHOICE_HELPERS[helper]
        args = call.args
        if len(args) <= reason_at:
            return
        try:
            option = self.evaluate(args[option_at], bindings)
        except Exception:  # noqa: BLE001 - an option computed at run time
            option = None
        try:
            allowed = _sorted_values(self.evaluate(args[allowed_at], bindings))
        except Exception:  # noqa: BLE001 - admitted values computed at run time
            allowed = None
        try:
            reason = str(self.evaluate(args[reason_at], bindings))
        except Exception:  # noqa: BLE001
            reason = self.text(args[reason_at])
        if not isinstance(option, str) or allowed is None:
            applies, guard = "dynamic", None
            option = option if isinstance(option, str) else self.text(args[option_at])
            allowed = allowed if allowed is not None else ()
        else:
            applies, guard = self._applies(function, option, args[value_at], guards, bindings, early_return, helper)
        self.rules.append(ValueRule(
            option=option,
            allowed=allowed,
            integer=helper in INTEGER_HELPERS,
            reason=reason,
            function=function.name,
            applies=applies,
            guard=guard,
        ))

    def _applies(
        self,
        function: ast.FunctionDef,
        option: str,
        value: ast.AST,
        guards: list[tuple[str, str]],
        bindings: Mapping[str, ast.AST],
        early_return: str | None,
        helper: str,
    ) -> tuple[str, str | None]:
        key = option.split(".", 1)[-1]
        presence = f'"{key}" in '
        remaining = []
        declared = helper == "_optional_integer_choice" or early_return == "presence"
        for kind, text in guards:
            if kind in ("ifexp", "if") and text.startswith(presence):
                declared = True  # declared-key guard
                continue
            if kind == "if" and self._same_value_guard(text, option, bindings):
                continue  # ``if flag: _choice(name, True, {False})`` judges the flag itself
            remaining.append((kind, text))
        if function.name not in CARTESIAN_FUNCTIONS:
            return "conditional", f"in {function.name}"
        if early_return == "other":
            return "conditional", f"after an early return in {function.name}"
        if remaining:
            return "conditional", " and ".join(text for _, text in remaining)
        return ("when-declared" if declared else "cartesian"), None

    def _same_value_guard(self, text: str, option: str, bindings: Mapping[str, ast.AST]) -> bool:
        guard = ast.parse(text, mode="eval").body
        if isinstance(guard, ast.Name) and guard.id in bindings:
            guard = bindings[guard.id]
        return ast.unparse(guard) == f"config.{option}"


def extract_value_rules(source: str | None = None) -> list[ValueRule]:
    """Every value rule in ``config.py``, in source order."""

    if source is None:
        _, source = _config_source()
    return _Extractor(source).run()


def record_payload(rules: Sequence[ValueRule]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for rule in rules:
        counts[rule.applies] = counts.get(rule.applies, 0) + 1
    return {
        "schema": SCHEMA,
        "source": "src/cm1core/config.py (_choice, _integer_choice, _optional_integer_choice sites)",
        "generator": "tools/generate_cm1_value_rules.py",
        "counts": dict(sorted(counts.items())),
        "rules": [rule.record() for rule in rules],
    }


# ---------------------------------------------------------------------------
# The record, read side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueRules:
    rules: tuple[ValueRule, ...]

    def for_option(self, option: str) -> tuple[ValueRule, ...]:
        return tuple(rule for rule in self.rules if rule.option == option)

    @property
    def key_level(self) -> tuple[ValueRule, ...]:
        return tuple(rule for rule in self.rules if rule.key_level)

    def admitted(self, option: str) -> tuple[Any, ...] | None:
        """The values every key-level rule on ``option`` admits (their intersection)."""

        rules = [rule for rule in self.for_option(option) if rule.key_level]
        if not rules:
            return None
        allowed = set(rules[0].allowed)
        for rule in rules[1:]:
            allowed &= set(rule.allowed)
        return _sorted_values(allowed)


def admitted_cm1(cm1_key: str, rules: "ValueRules | None" = None) -> tuple[Any, ...] | None:
    """The values of one CM1 setting every key-level rule admits, as the namelist writes them.

    For the GUI's editors: a setting whose values GM1 limits (psolver: 3;
    iinit: 0, 1, 2, 3, 5, 8, 11, 13) greys out the others with the reason
    ``config.py`` gives.  None when no key-level rule limits the setting.
    """

    from . import keymap as _keymap

    entry = _keymap.load().by_cm1(cm1_key)
    if entry is None or not entry.gm1_key:
        return None
    admitted = (rules or load()).admitted(entry.gm1_key)
    if admitted is None:
        return None
    if entry.transform == "int-to-bool":
        return tuple(int(value) if isinstance(value, bool) else value for value in admitted)
    return admitted


def _parse(payload: Mapping[str, Any]) -> ValueRules:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{RECORD_NAME} has schema {payload.get('schema')!r}, not {SCHEMA!r}")
    return ValueRules(rules=tuple(
        ValueRule(
            option=row["option"],
            allowed=tuple(row["allowed"]),
            integer=bool(row["integer"]),
            reason=row["reason"],
            function=row["function"],
            applies=row["applies"],
            guard=row.get("guard"),
        )
        for row in payload["rules"]
    ))


@lru_cache(maxsize=1)
def load() -> ValueRules:
    text = resources.files("cm1core.records").joinpath(RECORD_NAME).read_text(encoding="utf-8")
    return _parse(json.loads(text))


def judge(document: Mapping[str, Mapping[str, Any]], rules: ValueRules | None = None) -> list[RuleBreach]:
    """The key-level rules a gpuwm-cm1 document breaks, one per option, in rule order.

    ``document`` is ``{table: {key: value}}``.  A key the document does not
    declare is not judged: ``config.py`` then takes its own default, which it
    admits.
    """

    rules = rules or load()
    breaches: list[RuleBreach] = []
    seen: set[str] = set()
    for rule in rules.key_level:
        if rule.option in seen:
            continue
        section, _, name = rule.option.partition(".")
        table = document.get(section)
        if not isinstance(table, Mapping) or name not in table:
            continue
        value = table[name]
        if rule.admits(value):
            continue
        seen.add(rule.option)
        breaches.append(RuleBreach(rule=rule, value=value, allowed=rules.admitted(rule.option) or rule.allowed))
    return breaches


__all__ = [
    "RECORD_NAME",
    "admitted_cm1",
    "RuleBreach",
    "SCHEMA",
    "ValueRule",
    "ValueRules",
    "extract_value_rules",
    "judge",
    "load",
    "record_payload",
]
