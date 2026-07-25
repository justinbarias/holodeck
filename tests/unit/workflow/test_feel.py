"""Tests for the FEEL subset wrapper (036, T3).

Covers the wrapper's two jobs on top of the embedded evaluator: static subset
enforcement with a locator (the allowlist walker the grammar can't do for us —
see test_feel_conformance.py) and unary-test compilation/evaluation, including
translating the evaluator's parse/type errors into locator-bearing workflow
errors (FR-010/FR-012).
"""

import datetime

import pytest

from holodeck.lib.errors import FeelEvaluationError, FeelValidationError
from holodeck.lib.workflow import feel


@pytest.mark.unit
class TestValidateExpression:
    """Static subset enforcement over full FEEL expressions."""

    @pytest.mark.parametrize(
        "expression",
        [
            "(income.net - income.expenses) / income.net >= 0.25",
            'status = "verified"',
            "ratio in [0.10..0.25)",
            'tier in ["low", "medium"]',
            "a >= 0.25 and b <= 90",
            'date("2026-06-01") > date("2026-01-01")',
            "application_date - statement_date",
        ],
    )
    def test_in_subset_expressions_pass(self, expression: str) -> None:
        # Should not raise.
        feel.validate_expression(expression, locator="loc")

    @pytest.mark.parametrize(
        "expression",
        [
            "some x in [1, 2] satisfies x > 1",  # quantifier (allowlist rejects)
            'duration("P30D")',  # unknown function (allowlist rejects)
            'if a > 1 then "y" else "n"',  # conditional (parser rejects)
        ],
    )
    def test_out_of_subset_expressions_rejected(self, expression: str) -> None:
        # Rejected either by the allowlist walker ("subset") or, for constructs
        # the grammar itself refuses, at parse time ("malformed") — both are
        # FeelValidationError carrying the locator.
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_expression(expression, locator="table 'x' input 'y'")
        assert exc.value.locator == "table 'x' input 'y'"

    def test_malformed_expression_rejected(self) -> None:
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_expression("1 +", locator="loc")
        assert "malformed" in str(exc.value)


@pytest.mark.unit
class TestCompileUnaryTest:
    """A DMN unary-test cell compiles to a full boolean FEEL expression."""

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("-", None),
            (">= 0.25", "__cell_input__ >= 0.25"),
            ("<= 90", "__cell_input__ <= 90"),
            ("> 90", "__cell_input__ > 90"),
            ("< 0.10", "__cell_input__ < 0.10"),
            ("!= 0", "__cell_input__ != 0"),
            ("[0.10..0.25)", "__cell_input__ in [0.10..0.25)"),
            ("(0.10..0.25]", "__cell_input__ in (0.10..0.25]"),
            ('"verified"', '__cell_input__ = "verified"'),
            ("5", "__cell_input__ = 5"),
        ],
    )
    def test_compilation(self, cell: str, expected: str | None) -> None:
        assert feel.compile_unary_test(cell) == expected

    def test_whitespace_is_stripped(self) -> None:
        assert feel.compile_unary_test("  >= 0.25  ") == "__cell_input__ >= 0.25"


@pytest.mark.unit
class TestValidateUnaryTest:
    """Static subset enforcement over compiled unary-test cells."""

    def test_irrelevant_cell_passes(self) -> None:
        feel.validate_unary_test("-", locator="loc")

    @pytest.mark.parametrize("cell", [">= 0.25", "[0.10..0.25)", '"verified"'])
    def test_in_subset_cells_pass(self, cell: str) -> None:
        feel.validate_unary_test(cell, locator="loc")

    def test_malformed_cell_rejected(self) -> None:
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_unary_test(">=", locator="table 'x' rule 1 cell 'c'")
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"


@pytest.mark.unit
class TestEvaluateExpression:
    """Full-expression evaluation with locator-bearing error wrapping."""

    def test_returns_value(self) -> None:
        result = feel.evaluate_expression(
            "(a - b) / a", {"a": 4000, "b": 2800}, locator="loc"
        )
        assert result == pytest.approx(0.3)

    def test_date_difference_returns_timedelta(self) -> None:
        result = feel.evaluate_expression(
            "a - b",
            {
                "a": datetime.date(2026, 1, 31),
                "b": datetime.date(2026, 1, 1),
            },
            locator="loc",
        )
        assert result == datetime.timedelta(days=30)

    def test_type_error_wrapped(self) -> None:
        # A None (missing var) poisons the comparison -> eval-time type error.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression("missing > 1", {}, locator="table 'x' input 'y'")
        assert exc.value.locator == "table 'x' input 'y'"

    def test_malformed_wrapped_as_validation_error(self) -> None:
        with pytest.raises(FeelValidationError):
            feel.evaluate_expression("1 +", {}, locator="loc")


@pytest.mark.unit
class TestEvaluateUnaryTest:
    """Unary-test evaluation against a single input value."""

    def test_irrelevant_cell_always_true(self) -> None:
        assert feel.evaluate_unary_test("-", 999, locator="loc") is True

    @pytest.mark.parametrize(
        ("cell", "value", "expected"),
        [
            (">= 0.25", 0.30, True),
            (">= 0.25", 0.10, False),
            ("[0.10..0.25)", 0.15, True),
            ("[0.10..0.25)", 0.25, False),
            ('"verified"', "verified", True),
            ('"verified"', "expired", False),
            ("<= 90", 30, True),
            ("<= 90", 120, False),
        ],
    )
    def test_evaluation(self, cell: str, value: object, expected: bool) -> None:
        assert feel.evaluate_unary_test(cell, value, locator="loc") is expected

    def test_type_mismatch_wrapped(self) -> None:
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_unary_test(">= 5", "abc", locator="table 'x' rule 1 cell 'c'")
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"
