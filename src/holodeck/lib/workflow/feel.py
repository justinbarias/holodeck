"""FEEL subset validation and evaluation for the deterministic spine (036, T3).

Wraps the embedded evaluator (bkflow-feel) with two responsibilities:

1. **Static subset enforcement** — a load-time allowlist walker over the lark
   parse tree that rejects any construct outside the FR-010 subset with a
   precise locator. The grammar accepts more FEEL than the subset allows (e.g.
   quantifiers, unknown functions that fail silently), so subset rejection
   cannot be delegated to the parser — see research.md caveats 2-3.
2. **Evaluation** — thin helpers for full expressions (input expressions) and
   DMN unary-test rule cells, translating *every* error the embedded evaluator
   raises into locator-bearing workflow errors (FR-012 loud failures) and
   rejecting unresolvable context reads before evaluation, which the embedded
   evaluator would otherwise resolve to a silent ``None`` (caveat 4).

Three things the embedded evaluator gets silently wrong are corrected here:

* **Unresolvable reads.** An unbound root, a missing attribute under a bound
  root, and an attribute read through a non-mapping all evaluate to ``None``.
  ``!=`` has no type validator behind it (it is a bare Python ``!=``), so such
  a ``None`` makes a ``!=`` cell match unconditionally — a silently wrong
  verdict. Every dot-path is therefore resolved against the bindings first.
* **Numeric strictness.** The evaluator compares operands with
  ``isinstance(left, type(right))``, so ``1999.50 <= 2000`` raises. Unary tests
  are evaluated with both sides widened to ``float`` (see
  :data:`_UNARY_TEST_TRANSFORMER`).
* **Native exceptions.** ``ZeroDivisionError`` and the ``TypeError`` raised by
  range cells (which bypass the evaluator's validator) escape its own error
  type entirely; both are re-raised with a locator (caveat 5).

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
from bkflow_feel.parsers import Number
from bkflow_feel.transformer import FEELTransformer
from lark import v_args

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

# Unary-test comparison operator prefixes. The rest of the cell is appended
# verbatim, so only membership in this tuple matters — not its order.
_COMPARISON_OPS = (">=", "<=", "!=", ">", "<", "=")

# Names FEEL parses as literals, never as variables. A fact bound under one of
# them is invisible to every expression *and* to the unbound-read check below,
# so the binding is rejected outright rather than silently shadowed.
_LITERAL_NAMES: frozenset[str] = frozenset({"true", "false", "null"})


@v_args(inline=True)
class _FloatLiteralTransformer(FEELTransformer):
    """Builds the evaluator's AST with every numeric literal as a ``float``.

    The embedded evaluator validates binary operands with
    ``isinstance(left, type(right))`` — exact and asymmetric — so an ``int``
    literal refuses a ``float`` value and vice versa. Widening both the literal
    (here) and the column value (:func:`_widen_number`) to ``float`` makes
    numeric unary tests behave the same whichever side is written with a
    decimal point.
    """

    def number(self, token: lark.Token) -> Number:
        """Build a numeric literal node, always as a ``float``.

        Args:
            token: The matched number token.

        Returns:
            The evaluator's literal node holding a ``float``.
        """
        return Number(float(token.value))


# Full expressions keep the stock literal typing: they compute over facts, and
# widening every literal there would break `int`-typed fact arithmetic for no
# gain. Only unary tests — where the strictness trap actually bites — widen.
_EXPRESSION_TRANSFORMER = FEELTransformer()
_UNARY_TEST_TRANSFORMER = _FloatLiteralTransformer()


def _parse_tree(text: str, *, locator: str) -> lark.Tree:
    """Parse FEEL text, raising :class:`FeelValidationError` on syntax errors."""
    try:
        return cast(lark.Tree, parser.parse(text))
    except lark.exceptions.UnexpectedInput as exc:
        raise FeelValidationError(
            locator, f"malformed FEEL expression: {text!r}"
        ) from exc


def _referenced_paths(tree: lark.Tree) -> frozenset[tuple[str, ...]]:
    """Collect the context reads a parsed FEEL expression performs.

    A bare name is a ``variable`` node; a dot-path parses as
    ``context_item(variable ROOT, attr, ...)``. Each read is returned as its
    full path, so ``income.statement_date`` yields
    ``("income", "statement_date")`` — the whole path is statically known and
    therefore statically checkable.

    Args:
        tree: A parsed FEEL expression.

    Returns:
        Every context read performed anywhere in the tree.
    """
    paths: set[tuple[str, ...]] = set()

    def walk(node: lark.Tree) -> None:
        name = str(node.data)
        if name == "variable":
            paths.add((str(node.children[0]),))
            return
        if name == "context_item":
            head = node.children[0]
            if isinstance(head, lark.Tree) and str(head.data) == "variable":
                paths.add(
                    (str(head.children[0]), *(str(key) for key in node.children[1:]))
                )
                return
        for child in node.children:
            if isinstance(child, lark.Tree):
                walk(child)

    walk(tree)
    return frozenset(paths)


def _root_names(tree: lark.Tree) -> frozenset[str]:
    """Collect the root variable names a parsed FEEL expression references."""
    return frozenset(path[0] for path in _referenced_paths(tree))


def referenced_roots(text: str, *, locator: str) -> frozenset[str]:
    """Return the root variable names a FEEL expression reads.

    Exposed for callers that know something this module cannot: which names are
    actually on offer. A decision table's FEEL is validated standalone, so only
    the workflow node that mounts it can say whether ``evidence.net_income``
    names anything at all — and it can only say so from the roots.

    Args:
        text: FEEL expression source.
        locator: Human-readable location for error messages.

    Returns:
        Every root name referenced, bare or as the head of a dot-path.

    Raises:
        FeelValidationError: If the expression is malformed.
    """
    return _root_names(_parse_tree(text, locator=locator))


def _unresolvable(path: tuple[str, ...], bindings: dict[str, Any]) -> str | None:
    """Describe why a dot-path cannot resolve, or ``None`` when it resolves.

    The root is assumed bound (checked separately, so a typo'd root keeps its
    own message). Both remaining ways a read silently yields ``None`` are
    errors: an attribute absent from a bound mapping, and an attribute read
    through a value that is not a mapping at all. Presence is what is checked,
    so a *leaf* explicitly bound to ``None`` is a bound fact and resolves; a
    ``None`` *intermediate* does not, because no attribute of it can exist.

    Args:
        path: A dot-path with at least two segments.
        bindings: The evaluation context.

    Returns:
        A human-readable reason, or ``None`` if every segment resolves.
    """
    value = bindings[path[0]]
    seen = path[0]
    for key in path[1:]:
        # The evaluator resolves attributes with `dict.get` and gives up on
        # anything that is not a `dict`, so `dict` is the exact test.
        if not isinstance(value, dict):
            return (
                f"'{seen}' is of type {type(value).__name__}, not a mapping, "
                f"so '{'.'.join(path)}' cannot resolve"
            )
        if key not in value:
            return f"missing attribute '{key}' on '{seen}'"
        value = value[key]
        seen = f"{seen}.{key}"
    return None


def validate_expression(
    text: str, *, locator: str, reserved_roots: frozenset[str] = frozenset()
) -> None:
    """Statically validate a full FEEL expression against the subset.

    Args:
        text: FEEL expression source (an ``inputs[].expression`` value).
        locator: Human-readable location for error messages.
        reserved_roots: Variable names that are non-executable metadata and
            must not be referenced, bare or as a dot-path root (FR-032).
            Empty by default, so nothing is reserved unless a caller asks.

    Raises:
        FeelValidationError: If the expression is malformed, uses any construct
            outside the FR-010 subset, or references a reserved root name.
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
    reserved = sorted(_root_names(tree) & reserved_roots)
    if reserved:
        raise FeelValidationError(
            locator,
            f"'{reserved[0]}' is non-executable metadata and cannot be "
            f"referenced from FEEL in {text!r}",
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


def validate_unary_test(
    text: str, *, locator: str, reserved_roots: frozenset[str] = frozenset()
) -> None:
    """Statically validate a DMN unary-test cell against the subset.

    A compiled unary test can only ever bind its column's value, so a cell
    naming any other variable — a forgotten pair of quotes around a string
    (``verified``), or a reference to a threshold that does not exist
    (``>= threshold``) — is unconditionally broken. Under ``FIRST`` such a rule
    may never be reached, so the failure is data-dependent at runtime: reject
    it at load instead (FR-010 static rejection).

    Args:
        text: The unary-test cell source.
        locator: Human-readable location for error messages.
        reserved_roots: Variable names the cell must not reference; see
            :func:`validate_expression`.

    Raises:
        FeelValidationError: If the compiled test is malformed, out-of-subset,
            references a reserved root name, or names any variable other than
            its own column value.
    """
    expression = compile_unary_test(text)
    if expression is None:
        return
    validate_expression(expression, locator=locator, reserved_roots=reserved_roots)
    free = sorted(_root_names(_parse_tree(expression, locator=locator)) - {_CELL_INPUT})
    if free:
        names = ", ".join(repr(name) for name in free)
        raise FeelValidationError(
            locator,
            f"unary test {text!r} references {names}, but a rule cell can only "
            f"test its own column value (quote a string cell as '\"...\"')",
        )


def _evaluate(
    text: str, bindings: dict[str, Any], *, locator: str, transformer: Any
) -> Any:
    """Resolve every context read, then evaluate ``text`` with ``transformer``.

    Args:
        text: FEEL expression source.
        bindings: Variable bindings.
        locator: Human-readable location for error messages.
        transformer: The evaluator AST builder to use.

    Returns:
        The evaluated value.

    Raises:
        FeelValidationError: On a syntax error (should not occur post-load).
        FeelEvaluationError: If a binding shadows a FEEL literal, if a read
            cannot resolve, or on any error the evaluator raises.
    """
    shadowed = sorted(bindings.keys() & _LITERAL_NAMES)
    if shadowed:
        raise FeelEvaluationError(
            locator,
            f"'{shadowed[0]}' is a FEEL literal, not a usable fact name: the "
            f"binding would be invisible to every expression",
        )
    # Parsed again here (the evaluator parses internally): the tree is the only
    # place the referenced paths are visible ahead of evaluation.
    paths = _referenced_paths(_parse_tree(text, locator=locator))
    missing = sorted({path[0] for path in paths} - bindings.keys())
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise FeelEvaluationError(
            locator,
            f"unbound variable{'s' if len(missing) > 1 else ''} {names} "
            f"referenced by {text!r}",
        )
    unresolvable = [
        problem
        for path in sorted(paths)
        if len(path) > 1 and (problem := _unresolvable(path, bindings)) is not None
    ]
    if unresolvable:
        raise FeelEvaluationError(
            locator, f"{'; '.join(unresolvable)}, referenced by {text!r}"
        )
    try:
        return parse_expression(text, context=bindings, transformer=transformer)
    except _FeelLibValidationError as exc:
        raise FeelEvaluationError(
            locator, f"could not evaluate {text!r}: {exc}"
        ) from exc
    except Exception as exc:
        # The embedded evaluator re-raises native Python errors unchanged —
        # ZeroDivisionError from `/`, and TypeError from range cells, which
        # bypass its operand validator. Left alone they escape the error
        # taxonomy with no locator at all (research.md caveat 5).
        raise FeelEvaluationError(
            locator, f"could not evaluate {text!r}: {type(exc).__name__}: {exc}"
        ) from exc


def evaluate_expression(
    text: str, context: dict[str, Any] | None, *, locator: str
) -> Any:
    """Evaluate a full FEEL expression against a variable context.

    Every context read the expression performs must resolve *before*
    evaluation: the embedded evaluator resolves an unbound name, a missing
    attribute, and an attribute of a non-mapping all to ``None`` silently
    (research.md caveat 4), so a typo in an input expression would otherwise
    turn into a wrong match rather than a failure. Resolution is by key
    presence, so a name (or leaf attribute) explicitly bound to ``None`` is a
    bound fact and still evaluates — an absent one is not.

    Args:
        text: FEEL expression source.
        context: Variable bindings (named inputs / dot-path roots).
        locator: Human-readable location for error messages.

    Returns:
        The evaluated value (number, bool, str, ``datetime.date``,
        ``datetime.timedelta``, ...).

    Raises:
        FeelValidationError: On a syntax error (should not occur post-load).
        FeelEvaluationError: If a referenced path does not resolve, if a
            binding shadows a FEEL literal, or on any evaluation error.
    """
    return _evaluate(
        text, context or {}, locator=locator, transformer=_EXPRESSION_TRANSFORMER
    )


def _widen_number(value: Any) -> Any:
    """Widen an ``int`` column value to ``float``; pass anything else through.

    Pairs with :class:`_FloatLiteralTransformer` so both sides of a numeric
    unary test are ``float`` and the evaluator's exact-type operand check
    always agrees. ``bool`` is an ``int`` subclass but is deliberately *not*
    widened: a boolean column has to keep comparing against ``true``/``false``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def evaluate_unary_test(text: str, value: Any, *, locator: str) -> bool:
    """Evaluate a DMN unary-test cell against a single input value.

    Numeric values and numeric literals are both widened to ``float`` first, so
    a cell written ``<= 2000`` accepts ``1999.50`` and a cell written
    ``<= 2000.0`` accepts ``1999`` — the embedded evaluator's operand check is
    exact-typed and would reject both.

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
    result = _evaluate(
        expression,
        {_CELL_INPUT: _widen_number(value)},
        locator=locator,
        transformer=_UNARY_TEST_TRANSFORMER,
    )
    if not isinstance(result, bool):
        raise FeelEvaluationError(
            locator,
            f"unary test {text!r} did not evaluate to a boolean "
            f"(got {type(result).__name__})",
        )
    return result
