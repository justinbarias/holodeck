"""FEEL subset validation and evaluation for the deterministic spine (036, T3).

Wraps the embedded evaluator (bkflow-feel) with two responsibilities:

1. **Static subset enforcement** — a load-time allowlist walker over the lark
   parse tree that rejects any construct outside the FR-010 subset with a
   precise locator. The grammar accepts more FEEL than the subset allows (e.g.
   quantifiers, unknown functions that fail silently), so subset rejection
   cannot be delegated to the parser — see research.md caveats 2-3.
2. **Evaluation** — thin helpers for full expressions (input expressions) and
   DMN unary-test rule cells, translating the embedded evaluator's parse/type
   errors into locator-bearing workflow errors (FR-012 loud failures).

Behavior is pinned by the T1 conformance suite
(``tests/unit/workflow/test_feel_conformance.py``) and research.md. Notably,
``date(..)`` accepts only a quoted literal — date math on input *values* uses
bare variable subtraction (``a - b`` over date objects), which yields a
``timedelta`` (see ``research.md`` caveats 1 and 6).
"""

from __future__ import annotations

from typing import Any, cast

import lark.exceptions
from bkflow_feel import parser
from bkflow_feel.api import parse_expression
from bkflow_feel.exceptions import ValidationError as _FeelLibValidationError

from holodeck.lib.errors import FeelEvaluationError, FeelValidationError

# Lark parse-tree node names permitted by the FR-010 FEEL subset, confirmed
# against real parse trees during T3. Posture is allowlist, not blocklist:
# anything not listed here is rejected at load (research.md static-rejection
# list). Inlined grammar rules (``?expr`` etc.) never appear as nodes.
_ALLOWED_NODES: frozenset[str] = frozenset(
    {
        # boolean combinators
        "and_",
        "or_",
        "not_func",
        # numeric / string / date comparisons
        "gt",
        "lt",
        "gte",
        "lte",
        "eq",
        "ne",
        # arithmetic (surplus ratios, date differences)
        "add",
        "sub",
        "mul",
        "div",
        # literals & atoms
        "number",
        "string",
        "true",
        "false",
        "null",
        "list_",
        # variable + dot-path context access
        "variable",
        "context_item",
        # ranges & membership
        "in_",
        "range_atom",
        "close_range_group",
        "open_range_group",
        "left_open_range_group",
        "right_open_range_group",
        # date literals (comparison/difference handled by gt/lt/sub above)
        "date_func",
        "date",
    }
)

# The internal variable a DMN unary test's implicit input binds to. Chosen to
# never collide with an authored input name.
_CELL_INPUT = "__cell_input__"

# Unary-test comparison operators, longest-first so ``>=`` wins over ``>``.
_COMPARISON_OPS = (">=", "<=", "!=", ">", "<", "=")


def _parse_tree(text: str, *, locator: str) -> lark.Tree:
    """Parse FEEL text, raising :class:`FeelValidationError` on syntax errors."""
    try:
        return cast(lark.Tree, parser.parse(text))
    except lark.exceptions.UnexpectedInput as exc:
        raise FeelValidationError(
            locator, f"malformed FEEL expression: {text!r}"
        ) from exc


def validate_expression(text: str, *, locator: str) -> None:
    """Statically validate a full FEEL expression against the subset.

    Args:
        text: FEEL expression source (an ``inputs[].expression`` value).
        locator: Human-readable location for error messages.

    Raises:
        FeelValidationError: If the expression is malformed or uses any
            construct outside the FR-010 subset.
    """
    tree = _parse_tree(text, locator=locator)
    for subtree in tree.iter_subtrees():
        name = str(subtree.data)
        if name not in _ALLOWED_NODES:
            raise FeelValidationError(
                locator,
                f"FEEL construct '{name}' is outside the supported subset "
                f"in {text!r}",
            )


def compile_unary_test(text: str) -> str | None:
    """Translate a DMN unary-test cell into a full boolean FEEL expression.

    A DMN unary test is an implicit predicate on its column's input value:
    ``>= 0.25`` means ``input >= 0.25``; ``[0.10..0.25)`` means
    ``input in [0.10..0.25)``; a bare literal ``"verified"`` means
    ``input = "verified"``; and ``-`` is the irrelevant cell (always matches).
    The cell text is preserved verbatim — only the implicit-input wiring is
    added — so "FEEL syntax is the contract" still holds.

    Args:
        text: The unary-test cell source.

    Returns:
        A full boolean FEEL expression, or ``None`` for the irrelevant cell.
    """
    stripped = text.strip()
    if stripped == "-":
        return None
    if stripped.startswith(("[", "(")):
        return f"{_CELL_INPUT} in {stripped}"
    for op in _COMPARISON_OPS:
        if stripped.startswith(op):
            return f"{_CELL_INPUT} {stripped}"
    return f"{_CELL_INPUT} = {stripped}"


def validate_unary_test(text: str, *, locator: str) -> None:
    """Statically validate a DMN unary-test cell against the subset.

    Args:
        text: The unary-test cell source.
        locator: Human-readable location for error messages.

    Raises:
        FeelValidationError: If the compiled test is malformed or out-of-subset.
    """
    expression = compile_unary_test(text)
    if expression is None:
        return
    validate_expression(expression, locator=locator)


def evaluate_expression(
    text: str, context: dict[str, Any] | None, *, locator: str
) -> Any:
    """Evaluate a full FEEL expression against a variable context.

    Args:
        text: FEEL expression source.
        context: Variable bindings (named inputs / dot-path roots).
        locator: Human-readable location for error messages.

    Returns:
        The evaluated value (number, bool, str, ``datetime.date``,
        ``datetime.timedelta``, ...).

    Raises:
        FeelValidationError: On a syntax error (should not occur post-load).
        FeelEvaluationError: On a type/value error during evaluation.
    """
    try:
        return parse_expression(text, context=context or {})
    except lark.exceptions.UnexpectedInput as exc:
        raise FeelValidationError(
            locator, f"malformed FEEL expression: {text!r}"
        ) from exc
    except _FeelLibValidationError as exc:
        raise FeelEvaluationError(
            locator, f"could not evaluate {text!r}: {exc}"
        ) from exc


def evaluate_unary_test(text: str, value: Any, *, locator: str) -> bool:
    """Evaluate a DMN unary-test cell against a single input value.

    Args:
        text: The unary-test cell source.
        value: The column input value the test is applied to.
        locator: Human-readable location for error messages.

    Returns:
        ``True`` if the value satisfies the test (or the cell is irrelevant).

    Raises:
        FeelEvaluationError: If the test does not evaluate to a boolean, or the
            evaluator raises a type/value error.
    """
    expression = compile_unary_test(text)
    if expression is None:
        return True
    result = evaluate_expression(expression, {_CELL_INPUT: value}, locator=locator)
    if not isinstance(result, bool):
        raise FeelEvaluationError(
            locator,
            f"unary test {text!r} did not evaluate to a boolean "
            f"(got {type(result).__name__})",
        )
    return result
