"""FEEL evaluator conformance suite for the deterministic spine (036, T1).

Characterization tests pinning bkflow-feel's observed behavior across the
FR-010 FEEL subset: numeric comparisons, ranges, booleans, string equality,
list membership, and date literals/comparison/difference. Each test encodes
behavior verified during the Phase 0 spike — if an upgrade of bkflow-feel
breaks one of these, the spine's table-evaluation contract is at risk.

Library decision and full verdict table: specs/036-deterministic-spine/research.md
"""

import datetime
from typing import Any

import lark.exceptions
import pytest
from bkflow_feel.api import parse_expression
from bkflow_feel.exceptions import ValidationError


def evaluate(expression: str, context: dict[str, Any] | None = None) -> Any:
    """Evaluate a FEEL expression against an optional variable context.

    Args:
        expression: FEEL expression source text.
        context: Variable bindings visible to the expression.

    Returns:
        The evaluated result, as produced by bkflow-feel.
    """
    return parse_expression(expression, context=context or {})


@pytest.mark.unit
class TestNumericComparisons:
    """FR-010: numeric comparison operators."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("5 > 3", True),
            ("5 < 3", False),
            ("5 >= 5", True),
            ("5 <= 4", False),
            ("5 = 5", True),
            ("5 != 4", True),
        ],
    )
    def test_literal_comparisons(self, expression: str, expected: bool) -> None:
        assert evaluate(expression) is expected

    def test_arithmetic_inside_comparison(self) -> None:
        # The affordability surplus-ratio shape: (income - expenses) / income.
        context = {"a": 4000, "b": 2800}
        assert evaluate("(a - b) / a >= 0.25", context) is True
        assert evaluate("(a - b) / a >= 0.35", context) is False


@pytest.mark.unit
class TestRanges:
    """FR-010: range membership with all four bracket combinations."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("5 in [1..10]", True),
            ("1 in [1..10]", True),  # closed start includes endpoint
            ("10 in [1..10]", True),  # closed end includes endpoint
            ("0.15 in [0.10..0.25)", True),
            ("0.25 in [0.10..0.25)", False),  # open end excludes endpoint
            ("0.10 in (0.10..0.25]", False),  # open start excludes endpoint
            ("0.25 in (0.10..0.25]", True),
            ("0.10 in (0.10..0.25)", False),
            ("0.17 in (0.10..0.25)", True),
        ],
    )
    def test_bracket_semantics(self, expression: str, expected: bool) -> None:
        assert evaluate(expression) is expected

    def test_variable_in_range(self) -> None:
        assert evaluate("ratio in [0.10..0.25)", {"ratio": 0.15}) is True
        assert evaluate("ratio in [0.10..0.25)", {"ratio": 0.30}) is False


@pytest.mark.unit
class TestBooleanLogic:
    """FR-010: and / or / not combinators."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 = 1 and 2 = 2", True),
            ("1 = 1 and 2 = 3", False),
            ("1 = 2 or 2 = 2", True),
            ("1 = 2 or 2 = 3", False),
            ("not(1 = 2)", True),
            ("not(1 = 1)", False),
        ],
    )
    def test_combinators(self, expression: str, expected: bool) -> None:
        assert evaluate(expression) is expected

    def test_combined_predicates_over_context(self) -> None:
        context = {"ratio": 0.3, "status": "verified"}
        assert evaluate('ratio >= 0.25 and status = "verified"', context) is True
        assert evaluate('ratio < 0.25 or status = "expired"', context) is False


@pytest.mark.unit
class TestStringEquality:
    """FR-010: string equality tests."""

    def test_literal_equality(self) -> None:
        assert evaluate('"verified" = "verified"') is True
        assert evaluate('"verified" = "expired"') is False

    def test_variable_equality(self) -> None:
        assert evaluate('status = "verified"', {"status": "verified"}) is True
        assert evaluate('status = "verified"', {"status": "expired"}) is False


@pytest.mark.unit
class TestListMembership:
    """FR-010: `in` over list literals."""

    def test_string_membership(self) -> None:
        context = {"tier": "medium"}
        assert evaluate('tier in ["low", "medium"]', context) is True
        assert evaluate('tier in ["low", "medium"]', {"tier": "high"}) is False

    def test_number_membership(self) -> None:
        assert evaluate("3 in [1, 2, 3]") is True
        assert evaluate("4 in [1, 2, 3]") is False


@pytest.mark.unit
class TestDates:
    """FR-010: date literals, comparison, and difference."""

    def test_date_literal_comparison(self) -> None:
        assert evaluate('date("2026-06-01") > date("2026-01-01")') is True
        assert evaluate('date("2026-01-01") > date("2026-06-01")') is False

    def test_date_in_date_range(self) -> None:
        expression = 'date("2026-03-15") in [date("2026-01-01")..date("2026-06-30")]'
        assert evaluate(expression) is True

    def test_date_difference_returns_timedelta(self) -> None:
        # The core spike risk (retired): date subtraction works and yields a
        # timedelta — NOT a FEEL days-and-time duration object.
        result = evaluate('date("2026-01-31") - date("2026-01-01")')
        assert result == datetime.timedelta(days=30)

    def test_date_difference_is_not_number_comparable(self) -> None:
        # Consequence for T3: a timedelta cannot be compared to a number, so
        # table input expressions doing date math must be converted to days by
        # the wrapper before rule cells compare them.
        with pytest.raises(ValidationError):
            evaluate('(date("2026-04-01") - date("2026-01-01")) > 30')


@pytest.mark.unit
class TestContextAccess:
    """Dot-path access into nested context — how composed node outputs flow in."""

    def test_dot_path_into_nested_dict(self) -> None:
        context = {"income": {"net_monthly_income": 4000, "monthly_expenses": 2800}}
        assert evaluate("income.net_monthly_income > 3000", context) is True

    def test_arithmetic_over_dot_paths(self) -> None:
        context = {"income": {"net_monthly_income": 4000, "monthly_expenses": 2800}}
        expression = (
            "(income.net_monthly_income - income.monthly_expenses)"
            " / income.net_monthly_income >= 0.25"
        )
        assert evaluate(expression, context) is True


@pytest.mark.unit
class TestErrorModes:
    """How the evaluator fails — informs T3's loud-failure wrapping (FR-012)."""

    def test_unknown_variable_raises_validation_error(self) -> None:
        # Missing variables resolve to None and fail at the comparison, not
        # with a dedicated unknown-variable error. T3's wrapper must check
        # required inputs are present BEFORE evaluation to report a useful
        # table/rule/cell locator.
        with pytest.raises(ValidationError):
            evaluate("missing_var > 1", {})

    def test_type_mismatch_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            evaluate('"abc" > 1')

    def test_malformed_expression_raises_lark_error(self) -> None:
        # Syntax errors surface as lark parse errors, not ValidationError —
        # T3's static validation must catch lark.exceptions.UnexpectedInput.
        with pytest.raises(lark.exceptions.UnexpectedInput):
            evaluate("1 +")


@pytest.mark.unit
class TestOutOfSubsetBehavior:
    """Constructs OUTSIDE the FR-010 subset — pins why T3 needs its own allowlist.

    bkflow-feel's grammar accepts more FEEL than FR-010 allows, so static
    subset rejection (T3) cannot delegate to the parser: it must enforce its
    own allowlist over the parse tree (or token stream).
    """

    def test_duration_literals_unsupported_and_fail_silently(self) -> None:
        # No duration() builtin — and worse, an unknown function call returns
        # None silently (the ValueError is caught and logged inside
        # bkflow-feel). T3's static validation must reject non-allowlisted
        # function names itself; runtime gives no exception to catch.
        assert evaluate('duration("P30D")') is None
        # Used inside a comparison, the None then fails as a type mismatch.
        with pytest.raises(ValidationError):
            evaluate('duration("P30D") > 30')

    def test_quantifier_accepted_by_grammar(self) -> None:
        # `some .. satisfies` is OUT of the FR-010 subset but the grammar
        # happily evaluates it. The parser will NOT reject this for us.
        assert evaluate("some x in [1, 2] satisfies x > 1") is True

    def test_for_expression_rejected_by_grammar(self) -> None:
        # `for .. return` IS rejected at parse time — but only some
        # out-of-subset constructs are, hence the allowlist requirement.
        with pytest.raises(lark.exceptions.UnexpectedInput):
            evaluate("for x in [1, 2] return x * 2")
