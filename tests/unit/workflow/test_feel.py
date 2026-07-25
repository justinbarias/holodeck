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
class TestReservedRoots:
    """Opt-in rejection of non-executable metadata roots (FR-032)."""

    @pytest.mark.parametrize(
        "expression",
        ["meta", "meta.reviewed_by", "meta.a.b > 1", "x = 1 and meta.flag"],
    )
    def test_reserved_root_rejected(self, expression: str) -> None:
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_expression(
                expression, locator="loc", reserved_roots=frozenset({"meta"})
            )
        assert "non-executable metadata" in str(exc.value)

    def test_reserved_root_rejected_in_unary_test(self) -> None:
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_unary_test(
                "= meta.flag", locator="loc", reserved_roots=frozenset({"meta"})
            )
        assert exc.value.locator == "loc"

    def test_nothing_reserved_by_default(self) -> None:
        feel.validate_expression("meta.reviewed_by", locator="loc")

    def test_a_cell_is_rejected_for_any_free_variable_reserved_or_not(self) -> None:
        # A rule cell can only test its own column value, so the free-variable
        # rule subsumes reserved roots there — see TestFreeVariablesInCells.
        with pytest.raises(FeelValidationError):
            feel.validate_unary_test("= meta.flag", locator="loc")

    @pytest.mark.parametrize("expression", ["metadata.x", "x.meta", "other"])
    def test_only_exact_root_is_reserved(self, expression: str) -> None:
        feel.validate_expression(
            expression, locator="loc", reserved_roots=frozenset({"meta"})
        )


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
class TestUnboundVariables:
    """Unbound roots fail before evaluation, not as a silent None (caveat 4)."""

    def test_unbound_root_names_the_variable(self) -> None:
        # Arrange / Act
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression(
                "surplus >= 0.25", {"income": 1}, locator="table 'x' input 'y'"
            )

        # Assert
        assert exc.value.locator == "table 'x' input 'y'"
        assert "unbound variable 'surplus'" in str(exc.value)
        assert "surplus >= 0.25" in str(exc.value)

    def test_only_the_dot_path_root_must_be_bound(self) -> None:
        # Arrange / Act — `incom` is the typo; `net` is an attribute, not a root.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression("incom.net", {"income": {"net": 1}}, locator="loc")

        # Assert
        assert "'incom'" in str(exc.value)
        assert "'net'" not in str(exc.value)

    def test_bound_root_with_any_attribute_path_evaluates(self) -> None:
        # Arrange / Act
        result = feel.evaluate_expression(
            "income.net", {"income": {"net": 1}}, locator="loc"
        )

        # Assert
        assert result == 1

    def test_every_unbound_name_is_reported(self) -> None:
        # Arrange / Act
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression("a - b", {}, locator="loc")

        # Assert
        assert "unbound variables 'a', 'b'" in str(exc.value)

    def test_bound_to_none_is_not_unbound(self) -> None:
        # Arrange / Act — presence is what is checked: an explicit null is a
        # bound fact and still evaluates.
        result = feel.evaluate_expression("x", {"x": None}, locator="loc")

        # Assert
        assert result is None

    def test_unbound_variable_in_a_unary_test_cell_fails(self) -> None:
        # Arrange / Act — a cell may only reference its implicit input.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_unary_test(
                ">= threshold", 0.4, locator="table 'x' rule 1 cell 'c'"
            )

        # Assert
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"
        assert "'threshold'" in str(exc.value)

    @pytest.mark.parametrize(
        ("cell", "value"),
        [
            ("-", 0.4),
            (">= 0.25", 0.4),
            ("[0.10..0.25)", 0.4),
            ('"verified"', "expired"),
        ],
    )
    def test_literal_cells_are_unaffected(self, cell: str, value: object) -> None:
        # Arrange / Act / Assert — the implicit input is a bound root.
        assert isinstance(feel.evaluate_unary_test(cell, value, locator="loc"), bool)


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

    def test_non_boolean_unary_test_is_rejected(self) -> None:
        # Arrange / Act — `> 1 or 5` short-circuits to the truthy 5, not a bool.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_unary_test("> 1 or 5", 0, locator="table 'x' rule 1 cell 'c'")

        # Assert
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"
        assert "did not evaluate to a boolean" in str(exc.value)
        assert "got float" in str(exc.value)


@pytest.mark.unit
class TestMissingAttributes:
    """A missing attribute under a bound root is loud, never a silent None.

    The embedded evaluator resolves it to ``None``; ``!=`` has no operand
    validator, so a ``!=`` cell would then match unconditionally and the table
    would emit a real-looking verdict (research.md caveat 4).
    """

    def test_missing_attribute_names_the_attribute_and_its_owner(self) -> None:
        # Arrange / Act
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression(
                "income.net_monthly", {"income": {"gross": 1}}, locator="table 'x'"
            )

        # Assert
        assert exc.value.locator == "table 'x'"
        assert "missing attribute 'net_monthly' on 'income'" in str(exc.value)

    def test_missing_nested_attribute_names_the_full_owner_path(self) -> None:
        # Arrange / Act
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression(
                "a.b.c", {"a": {"b": {"d": 1}}}, locator="table 'x'"
            )

        # Assert
        assert "missing attribute 'c' on 'a.b'" in str(exc.value)

    def test_attribute_read_through_a_non_mapping_fails(self) -> None:
        # Arrange / Act — `a.b` is an int, so `a.b.c` can never resolve.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression("a.b.c", {"a": {"b": 1}}, locator="table 'x'")

        # Assert
        assert "not a mapping" in str(exc.value)
        assert "'a.b.c'" in str(exc.value)

    def test_a_not_equal_cell_no_longer_matches_a_missing_attribute(self) -> None:
        # Arrange / Act / Assert — the silently-wrong-verdict path (F1).
        with pytest.raises(FeelEvaluationError):
            feel.evaluate_expression(
                "applicant.flag != true", {"applicant": {}}, locator="loc"
            )

    def test_leaf_attribute_bound_to_none_is_a_bound_fact(self) -> None:
        # Arrange / Act — presence, not truthiness, is what is checked.
        result = feel.evaluate_expression(
            "income.net", {"income": {"net": None}}, locator="loc"
        )

        # Assert
        assert result is None

    def test_every_unresolvable_path_is_reported(self) -> None:
        # Arrange / Act
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression("a.x - a.y", {"a": {"z": 1}}, locator="loc")

        # Assert
        assert "missing attribute 'x' on 'a'" in str(exc.value)
        assert "missing attribute 'y' on 'a'" in str(exc.value)


@pytest.mark.unit
class TestNativeExceptionsAreWrapped:
    """Native Python errors must not escape the taxonomy (caveat 5)."""

    def test_division_by_zero_carries_the_locator(self) -> None:
        # Arrange / Act — a zero-income applicant is schema-valid.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression(
                "(income.net - income.expenses) / income.net",
                {"income": {"net": 0, "expenses": 100}},
                locator="table 'affordability' input 'surplus_ratio'",
            )

        # Assert
        assert exc.value.locator == "table 'affordability' input 'surplus_ratio'"
        assert "ZeroDivisionError" in str(exc.value)

    @pytest.mark.parametrize("value", [None, "not-a-number"])
    def test_range_cell_type_error_carries_the_locator(self, value: object) -> None:
        # Arrange / Act — range cells bypass the evaluator's operand validator
        # and compare with raw Python, raising a bare TypeError.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_unary_test(
                "[0..90]", value, locator="table 'x' rule 1 cell 'c'"
            )

        # Assert
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"
        assert "TypeError" in str(exc.value)


@pytest.mark.unit
class TestNumericWidening:
    """Numeric unary tests behave the same whichever side has a decimal point."""

    @pytest.mark.parametrize(
        ("cell", "value", "expected"),
        [
            # float value against an int literal — the monetary-cents case.
            ("<= 2000", 1999.50, True),
            ("<= 2000", 2000.50, False),
            # int value against a float literal.
            ("<= 2000.0", 1999, True),
            ("<= 2000.0", 2001, False),
            ("= 2000", 2000.0, True),
            ("!= 2000", 2000.0, False),
            ("> 0", -0.5, False),
            ("< 0", -0.5, True),
        ],
    )
    def test_int_and_float_compare_alike(
        self, cell: str, value: object, expected: bool
    ) -> None:
        assert feel.evaluate_unary_test(cell, value, locator="loc") is expected

    @pytest.mark.parametrize("value", [45, 45.5])
    def test_range_and_comparison_cells_agree(self, value: object) -> None:
        # Arrange / Act / Assert — two cells an author considers equivalent.
        assert feel.evaluate_unary_test("[0..90]", value, locator="loc") is True
        assert feel.evaluate_unary_test("<= 90", value, locator="loc") is True

    def test_boolean_is_not_silently_compared_as_a_number(self) -> None:
        # Arrange / Act — `bool` is an `int` subclass, so `True <= 90` would
        # otherwise quietly compare as `1 <= 90`.
        with pytest.raises(FeelEvaluationError):
            feel.evaluate_unary_test("<= 90", True, locator="loc")

    @pytest.mark.parametrize(("cell", "value"), [("= true", True), ("= false", False)])
    def test_boolean_cells_still_compare_against_booleans(
        self, cell: str, value: bool
    ) -> None:
        assert feel.evaluate_unary_test(cell, value, locator="loc") is True


@pytest.mark.unit
class TestFreeVariablesInCells:
    """A rule cell may only test its own column value (FR-010, static)."""

    @pytest.mark.parametrize("cell", ["verified", ">= threshold", "= other.attr"])
    def test_free_variable_rejected_at_load(self, cell: str) -> None:
        # Arrange / Act
        with pytest.raises(FeelValidationError) as exc:
            feel.validate_unary_test(cell, locator="table 'x' rule 1 cell 'c'")

        # Assert
        assert exc.value.locator == "table 'x' rule 1 cell 'c'"
        assert "can only test its own column value" in str(exc.value)

    @pytest.mark.parametrize(
        "cell", ["-", '"verified"', ">= 0.25", "[0.10..0.25)", '["a", "b"]', "= true"]
    )
    def test_variable_free_cells_still_pass(self, cell: str) -> None:
        feel.validate_unary_test(cell, locator="loc")


@pytest.mark.unit
class TestLiteralFactNames:
    """`true`/`false`/`null` parse as literals, so they cannot name a fact."""

    @pytest.mark.parametrize("name", ["true", "false", "null"])
    def test_binding_named_after_a_literal_is_rejected(self, name: str) -> None:
        # Arrange / Act — the binding is invisible to FEEL *and* to the
        # unbound-read check, the one route a fact name defeats that guard.
        with pytest.raises(FeelEvaluationError) as exc:
            feel.evaluate_expression(name, {name: 5}, locator="table 'x'")

        # Assert
        assert exc.value.locator == "table 'x'"
        assert f"'{name}' is a FEEL literal" in str(exc.value)

    def test_a_normal_fact_name_is_unaffected(self) -> None:
        assert feel.evaluate_expression("truthy", {"truthy": 5}, locator="loc") == 5
