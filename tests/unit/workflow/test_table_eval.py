"""SC-004 conformance suite for decision-table evaluation (036, T4).

Covers the three hit policies (UNIQUE / FIRST / PRIORITY) with standard DMN
semantics, the irrelevant cell, ``days`` column coercion, and the two FR-012
loud failures: no-match without a declared default, and a UNIQUE table with
multiple matches. Evaluation is pure — no LLM is involved anywhere.
"""

import datetime
from typing import Any

import pytest

from holodeck.lib.errors import (
    FeelEvaluationError,
    FeelValidationError,
    TableEvalError,
)
from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.models.decision_table import DecisionTable

SURPLUS_INPUT = {
    "name": "surplus_ratio",
    "expression": "(income.net - income.expenses) / income.net",
    "type": "number",
}
RESIDENCY_INPUT = {
    "name": "residency_status",
    "expression": "residency.status",
    "type": "string",
}
AFFORDABILITY_OUTPUT = {
    "name": "affordability",
    "type": "string",
    "values": ["affordable", "marginal", "unaffordable"],
}

# surplus_ratio = 0.4, residency_status = "verified"
NAMED_INPUTS: dict[str, Any] = {
    "income": {"net": 5000, "expenses": 3000},
    "residency": {"status": "verified"},
}


def _table(**overrides: Any) -> DecisionTable:
    """Build a valid affordability table, overriding any top-level field."""
    table: dict[str, Any] = {
        "id": "affordability",
        "version": "2026-06-01.1",
        "hit_policy": "UNIQUE",
        "inputs": [SURPLUS_INPUT, RESIDENCY_INPUT],
        "outputs": [AFFORDABILITY_OUTPUT],
        "rules": [
            {
                "when": {"surplus_ratio": ">= 0.25", "residency_status": '"verified"'},
                "then": {"affordability": "affordable"},
                "annotation": "Comfortable surplus",
            },
            {
                "when": {"surplus_ratio": "[0.10..0.25)"},
                "then": {"affordability": "marginal"},
            },
            {
                "when": {"surplus_ratio": "< 0.10"},
                "then": {"affordability": "unaffordable"},
            },
        ],
    }
    table.update(overrides)
    return DecisionTable.model_validate(table)


@pytest.mark.unit
def test_unique_single_match_returns_verdict_with_provenance() -> None:
    """A UNIQUE table with one matching rule emits its outputs and identity."""
    # Arrange
    table = _table()

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert isinstance(verdict, Verdict)
    assert verdict.outputs == {"affordability": "affordable"}
    assert verdict.matched_rule_index == 1
    assert verdict.matched_rule_annotation == "Comfortable surplus"
    assert verdict.table_id == "affordability"
    assert verdict.table_version == "2026-06-01.1"
    assert verdict.is_default is False
    assert verdict.rule_identity == "rule 1"


@pytest.mark.unit
def test_verdict_outputs_deep_copy_isolates_the_source_table() -> None:
    """Mutating a returned ``Verdict.outputs`` must not corrupt the table.

    ``outputs`` is built from ``rule.then`` (and ``table.default``), both
    ``dict[str, Any]`` on the loaded ``DecisionTable`` — sharing the same
    hazard :class:`~holodeck.lib.workflow.edge.GatedOutput` deep-copies
    against. Without the copy, a caller mutating a nested value on the
    verdict mutates the table itself, corrupting it for the rest of the run.
    """
    # Arrange
    table = _table(
        outputs=[AFFORDABILITY_OUTPUT, {"name": "reasons", "type": "string"}],
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25", "residency_status": '"verified"'},
                "then": {
                    "affordability": "affordable",
                    "reasons": ["comfortable surplus"],
                },
            },
        ],
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)
    verdict.outputs["reasons"].append("forged evidence")

    # Assert — the source rule's `then` is untouched.
    assert table.rules[0].then["reasons"] == ["comfortable surplus"]


@pytest.mark.unit
def test_unique_multi_match_raises_naming_all_matched_rules() -> None:
    """FR-012: a UNIQUE table matching several rules fails loudly."""
    # Arrange — rules 1 and 3 both match surplus_ratio = 0.4.
    table = _table(
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25"},
                "then": {"affordability": "affordable"},
            },
            {
                "when": {"surplus_ratio": "< 0.10"},
                "then": {"affordability": "unaffordable"},
            },
            {
                "when": {"residency_status": '"verified"'},
                "then": {"affordability": "marginal"},
            },
        ]
    )

    # Act / Assert
    with pytest.raises(TableEvalError) as exc_info:
        evaluate(table, NAMED_INPUTS)
    message = str(exc_info.value)
    assert "affordability" in message
    assert "UNIQUE" in message
    assert "[1, 3]" in message


@pytest.mark.unit
def test_first_takes_the_first_matching_rule_in_document_order() -> None:
    """FIRST resolves several matches by document order."""
    # Arrange — rules 1 and 2 both match.
    table = _table(
        hit_policy="FIRST",
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25"},
                "then": {"affordability": "affordable"},
            },
            {
                "when": {"surplus_ratio": "> 0.10"},
                "then": {"affordability": "marginal"},
            },
        ],
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert verdict.outputs == {"affordability": "affordable"}
    assert verdict.matched_rule_index == 1


@pytest.mark.unit
def test_priority_beats_document_order() -> None:
    """PRIORITY ranks by declared `values` order, not by rule position."""
    # Arrange — rule 1 (document-first) emits the lowest-priority value, so a
    # FIRST implementation would pick it and PRIORITY must not.
    table = _table(
        hit_policy="PRIORITY",
        outputs=[
            {
                "name": "affordability",
                "type": "string",
                "values": ["unaffordable", "marginal", "affordable"],
            }
        ],
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25"},
                "then": {"affordability": "affordable"},
            },
            {
                "when": {"surplus_ratio": "> 0.10"},
                "then": {"affordability": "marginal"},
            },
            {
                "when": {"residency_status": '"verified"'},
                "then": {"affordability": "unaffordable"},
            },
        ],
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert — "unaffordable" is index 0 in `values`, so rule 3 wins.
    assert verdict.outputs == {"affordability": "unaffordable"}
    assert verdict.matched_rule_index == 3


@pytest.mark.unit
def test_priority_ranks_multiple_outputs_lexicographically() -> None:
    """With several outputs the first declared output dominates the ranking."""
    # Arrange — ranks are (1, 0) for rule 1 and (0, 1) for rule 2, so the two
    # rules disagree on which output favours them. Only an ordering that reads
    # the declared outputs left to right picks rule 2; reversing the tuple
    # picks rule 1.
    table = _table(
        hit_policy="PRIORITY",
        outputs=[
            {"name": "affordability", "type": "string", "values": ["deny", "allow"]},
            {"name": "review", "type": "string", "values": ["manual", "auto"]},
        ],
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25"},
                "then": {"affordability": "allow", "review": "manual"},
            },
            {
                "when": {"surplus_ratio": "> 0.10"},
                "then": {"affordability": "deny", "review": "auto"},
            },
        ],
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert — rule 2 wins on the dominant first output despite ranking last
    # on the second.
    assert verdict.outputs == {"affordability": "deny", "review": "auto"}
    assert verdict.matched_rule_index == 2


@pytest.mark.unit
def test_a_rule_needs_every_one_of_its_cells_to_match() -> None:
    """A multi-cell rule is a conjunction: one false cell and it must not hit."""
    # Arrange — rule 1's first cell is true (surplus 0.4 >= 0.25) and its
    # second is false (residency is "verified", not "expired").
    table = _table(
        hit_policy="FIRST",
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25", "residency_status": '"expired"'},
                "then": {"affordability": "affordable"},
            },
            {
                "when": {"residency_status": '"verified"'},
                "then": {"affordability": "marginal"},
            },
        ],
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert verdict.matched_rule_index == 2
    assert verdict.outputs == {"affordability": "marginal"}


@pytest.mark.unit
def test_a_rule_whose_first_cell_is_false_does_not_match() -> None:
    """The conjunction holds whichever cell of the rule is the false one."""
    # Arrange — rule 1's first cell is false, its second true.
    table = _table(
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.9", "residency_status": '"verified"'},
                "then": {"affordability": "affordable"},
            },
        ],
        default={"affordability": "unaffordable"},
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert verdict.is_default is True
    assert verdict.matched_rule_index is None


@pytest.mark.unit
def test_irrelevant_cells_always_match() -> None:
    """An omitted cell and an explicit `-` cell both match any value."""
    # Arrange
    table = _table(
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.9"},
                "then": {"affordability": "affordable"},
            },
            {
                # surplus_ratio omitted entirely; residency_status explicit "-".
                "when": {"residency_status": "-"},
                "then": {"affordability": "marginal"},
            },
        ]
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert verdict.matched_rule_index == 2
    assert verdict.outputs == {"affordability": "marginal"}


@pytest.mark.unit
def test_days_column_coerces_date_difference_to_a_number() -> None:
    """A `days` column compares as a number of days (research.md caveat 1)."""
    # Arrange — 2026-06-01 minus 2026-03-01 is 92 days.
    table = _table(
        inputs=[
            {
                "name": "statement_age",
                "expression": "application_date - income.statement_date",
                "type": "days",
            }
        ],
        outputs=[
            {"name": "statements", "type": "string", "values": ["fresh", "stale"]}
        ],
        rules=[
            {"when": {"statement_age": "<= 90"}, "then": {"statements": "fresh"}},
            {"when": {"statement_age": "> 90"}, "then": {"statements": "stale"}},
        ],
    )
    named_inputs = {
        "application_date": datetime.date(2026, 6, 1),
        "income": {"statement_date": datetime.date(2026, 3, 1)},
    }

    # Act
    verdict = evaluate(table, named_inputs)

    # Assert
    assert verdict.outputs == {"statements": "stale"}
    assert verdict.matched_rule_index == 2


@pytest.mark.unit
def test_days_column_accepts_an_already_numeric_expression() -> None:
    """A `days` expression that already yields a number passes through."""
    # Arrange
    table = _table(
        inputs=[
            {
                "name": "statement_age",
                "expression": "income.days_outstanding",
                "type": "days",
            }
        ],
        outputs=[
            {"name": "statements", "type": "string", "values": ["fresh", "stale"]}
        ],
        rules=[
            {"when": {"statement_age": "<= 90"}, "then": {"statements": "fresh"}},
            {"when": {"statement_age": "> 90"}, "then": {"statements": "stale"}},
        ],
    )

    # Act
    verdict = evaluate(table, {"income": {"days_outstanding": 12}})

    # Assert
    assert verdict.outputs == {"statements": "fresh"}


def _days_table(**overrides: Any) -> DecisionTable:
    """Build a `days` table over `application_date - income.statement_date`."""
    fields: dict[str, Any] = {
        "inputs": [
            {
                "name": "statement_age",
                "expression": "application_date - income.statement_date",
                "type": "days",
            }
        ],
        "outputs": [
            {"name": "statements", "type": "string", "values": ["fresh", "stale"]}
        ],
        "rules": [
            {"when": {"statement_age": "<= 1"}, "then": {"statements": "fresh"}},
            {"when": {"statement_age": "> 1"}, "then": {"statements": "stale"}},
        ],
    }
    fields.update(overrides)
    return _table(**fields)


@pytest.mark.unit
def test_days_column_keeps_sub_day_precision_against_a_whole_day_cell() -> None:
    """A 36-hour duration is 1.5 days, so `<= 1` must not match it."""
    # Arrange
    table = _days_table()
    named_inputs = {
        "application_date": datetime.datetime(2026, 6, 2, 12),
        "income": {"statement_date": datetime.datetime(2026, 6, 1)},
    }

    # Act
    verdict = evaluate(table, named_inputs)

    # Assert
    assert verdict.outputs == {"statements": "stale"}


@pytest.mark.unit
def test_days_column_matches_a_fractional_duration_below_the_threshold() -> None:
    """A 12-hour duration is 0.5 days and must match a whole-day `<= 1` cell."""
    # Arrange
    table = _days_table()
    named_inputs = {
        "application_date": datetime.datetime(2026, 6, 1, 12),
        "income": {"statement_date": datetime.datetime(2026, 6, 1)},
    }

    # Act
    verdict = evaluate(table, named_inputs)

    # Assert
    assert verdict.outputs == {"statements": "fresh"}


@pytest.mark.unit
def test_days_column_handles_a_negative_duration() -> None:
    """A statement dated after the application yields negative days."""
    # Arrange
    table = _days_table(
        rules=[
            {"when": {"statement_age": "< 0"}, "then": {"statements": "fresh"}},
            {"when": {"statement_age": ">= 0"}, "then": {"statements": "stale"}},
        ]
    )
    named_inputs = {
        "application_date": datetime.date(2026, 6, 1),
        "income": {"statement_date": datetime.date(2026, 6, 6)},
    }

    # Act
    verdict = evaluate(table, named_inputs)

    # Assert
    assert verdict.outputs == {"statements": "fresh"}


@pytest.mark.unit
def test_days_column_rejects_a_boolean() -> None:
    """`bool` is an `int` subclass and must not pass as a number of days."""
    # Arrange
    table = _table(
        inputs=[
            {
                "name": "statement_age",
                "expression": "income.is_stale",
                "type": "days",
            }
        ],
        outputs=[
            {"name": "statements", "type": "string", "values": ["fresh", "stale"]}
        ],
        rules=[{"when": {"statement_age": "<= 90"}, "then": {"statements": "fresh"}}],
    )

    # Act / Assert
    with pytest.raises(TableEvalError, match="'days' column"):
        evaluate(table, {"income": {"is_stale": True}})


@pytest.mark.unit
def test_days_column_rejects_a_non_duration_value() -> None:
    """A `days` column that evaluates to a string fails loudly."""
    # Arrange
    table = _table(
        inputs=[
            {
                "name": "statement_age",
                "expression": "residency.status",
                "type": "days",
            }
        ],
        outputs=[
            {"name": "statements", "type": "string", "values": ["fresh", "stale"]}
        ],
        rules=[
            {"when": {"statement_age": "<= 90"}, "then": {"statements": "fresh"}},
        ],
    )

    # Act / Assert
    with pytest.raises(TableEvalError, match="'days' column"):
        evaluate(table, NAMED_INPUTS)


@pytest.mark.unit
def test_no_match_emits_the_declared_default() -> None:
    """A no-match with a declared default yields it, marked as the default."""
    # Arrange
    table = _table(
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.9"},
                "then": {"affordability": "affordable"},
            }
        ],
        default={"affordability": "unaffordable"},
    )

    # Act
    verdict = evaluate(table, NAMED_INPUTS)

    # Assert
    assert verdict.outputs == {"affordability": "unaffordable"}
    assert verdict.matched_rule_index is None
    assert verdict.is_default is True
    assert verdict.rule_identity == "default, no rule matched"
    assert verdict.table_version == "2026-06-01.1"


@pytest.mark.unit
@pytest.mark.parametrize("hit_policy", ["UNIQUE", "FIRST", "PRIORITY"])
def test_no_match_without_a_default_raises(hit_policy: str) -> None:
    """FR-012: a no-match without a default stops the run, for every policy."""
    # Arrange
    table = _table(
        hit_policy=hit_policy,
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.9"},
                "then": {"affordability": "affordable"},
            }
        ],
    )

    # Act / Assert
    with pytest.raises(TableEvalError, match="no rule matched"):
        evaluate(table, NAMED_INPUTS)


@pytest.mark.unit
def test_unbound_input_expression_fails_instead_of_falling_through_to_default() -> None:
    """A typo in an input expression is loud, not a silent slide to `default`.

    research.md caveat 4: the embedded evaluator resolves an unbound name to
    `None`, so every unary test would fail and the table would emit its
    declared default — a silently wrong determination (FR-012, SC-004).
    """
    # Arrange — `incom` is a typo for the bound `income` root.
    table = _table(
        inputs=[
            {
                "name": "surplus_ratio",
                "expression": "(incom.net - incom.expenses) / incom.net",
                "type": "number",
            }
        ],
        rules=[
            {
                "when": {"surplus_ratio": ">= 0.25"},
                "then": {"affordability": "affordable"},
            }
        ],
        default={"affordability": "unaffordable"},
    )

    # Act / Assert
    with pytest.raises(FeelEvaluationError) as exc_info:
        evaluate(table, NAMED_INPUTS)
    assert exc_info.value.locator == "table 'affordability' input 'surplus_ratio'"
    assert "'incom'" in str(exc_info.value)


@pytest.mark.unit
def test_free_variable_in_a_rule_cell_is_rejected_at_load() -> None:
    """A rule cell referencing a name outside its implicit input never loads.

    Previously this only failed at evaluation, and under FIRST a rule that is
    never reached would never fail at all — so the table could pass every test
    and break on the first input that fell through to it (FR-010).
    """
    # Arrange / Act / Assert
    with pytest.raises(FeelValidationError) as exc_info:
        _table(
            rules=[
                {
                    "when": {"surplus_ratio": ">= threshold"},
                    "then": {"affordability": "affordable"},
                }
            ],
            default={"affordability": "unaffordable"},
        )
    assert exc_info.value.locator == "table 'affordability' rule 1 cell 'surplus_ratio'"
    assert "'threshold'" in str(exc_info.value)


@pytest.mark.unit
def test_missing_attribute_does_not_produce_a_verdict_via_a_not_equal_cell() -> None:
    """F1: a `!=` cell over a missing attribute must not decide anything.

    `!=` has no operand validator in the embedded evaluator, so an absent
    attribute (which resolves to `None`) made the cell match unconditionally —
    a verdict indistinguishable, in the run record, from a real determination.
    """
    # Arrange — `flag` is absent from the bound `applicant` object.
    table = _table(
        hit_policy="FIRST",
        inputs=[{"name": "flag", "expression": "applicant.flag", "type": "boolean"}],
        outputs=[
            {"name": "affordability", "type": "string", "values": ["allow", "deny"]}
        ],
        rules=[
            {"when": {"flag": "!= true"}, "then": {"affordability": "allow"}},
            {"when": {"flag": "= true"}, "then": {"affordability": "deny"}},
        ],
    )

    # Act / Assert
    with pytest.raises(FeelEvaluationError) as exc_info:
        evaluate(table, {"applicant": {}})
    assert exc_info.value.locator == "table 'affordability' input 'flag'"
    assert "missing attribute 'flag' on 'applicant'" in str(exc_info.value)


@pytest.mark.unit
def test_missing_attribute_does_not_slide_to_the_default_via_a_list_cell() -> None:
    """F4: `None in [..]` is False with no error, which reached `default`."""
    # Arrange — `tier` is absent from the bound `applicant` object.
    table = _table(
        inputs=[{"name": "tier", "expression": "applicant.tier", "type": "string"}],
        rules=[
            {
                "when": {"tier": '["gold", "silver"]'},
                "then": {"affordability": "affordable"},
            }
        ],
        default={"affordability": "unaffordable"},
    )

    # Act / Assert
    with pytest.raises(FeelEvaluationError) as exc_info:
        evaluate(table, {"applicant": {}})
    assert exc_info.value.locator == "table 'affordability' input 'tier'"
    assert "missing attribute 'tier' on 'applicant'" in str(exc_info.value)


@pytest.mark.unit
def test_zero_denominator_fails_with_the_column_locator() -> None:
    """F2: a zero-income applicant is schema-valid and must fail loudly."""
    # Arrange
    table = _table(default={"affordability": "unaffordable"})

    # Act / Assert
    with pytest.raises(FeelEvaluationError) as exc_info:
        evaluate(
            table,
            {"income": {"net": 0, "expenses": 0}, "residency": {"status": "verified"}},
        )
    assert exc_info.value.locator == "table 'affordability' input 'surplus_ratio'"
    assert "ZeroDivisionError" in str(exc_info.value)


@pytest.mark.unit
def test_number_column_compares_a_fractional_value_against_a_whole_literal() -> None:
    """F3: cents against a whole-number threshold is the common author case."""
    # Arrange
    table = _table(
        inputs=[{"name": "amount", "expression": "loan.amount", "type": "number"}],
        rules=[
            {"when": {"amount": "<= 2000"}, "then": {"affordability": "affordable"}},
            {"when": {"amount": "> 2000"}, "then": {"affordability": "unaffordable"}},
        ],
    )

    # Act
    verdict = evaluate(table, {"loan": {"amount": 1999.50}})

    # Assert
    assert verdict.outputs == {"affordability": "affordable"}


@pytest.mark.unit
def test_number_column_rejects_a_boolean() -> None:
    """F3: `bool` is an `int` subclass and would silently compare as 1/0."""
    # Arrange
    table = _table(
        inputs=[{"name": "amount", "expression": "loan.flag", "type": "number"}],
        rules=[
            {"when": {"amount": "<= 2000"}, "then": {"affordability": "affordable"}}
        ],
    )

    # Act / Assert
    with pytest.raises(TableEvalError, match="'number' column"):
        evaluate(table, {"loan": {"flag": True}})


@pytest.mark.unit
class TestVerdictRoundTrip:
    """A Verdict must survive dump -> validate.

    That dict -> JSON -> back trip is what a Temporal data converter performs
    on anything held in workflow state; a computed field plus extra="forbid"
    silently breaks it (this pins the fix that made rule_identity a plain
    property).
    """

    def test_matched_rule_round_trips(self) -> None:
        # Arrange
        verdict = Verdict(
            table_id="points",
            table_version="1.0",
            outputs={"points": 20, "nested": {"unit": "per_week"}},
            matched_rule_index=3,
            matched_rule_annotation="rule note",
        )

        # Act
        revalidated = Verdict.model_validate(verdict.model_dump())

        # Assert
        assert revalidated == verdict
        assert revalidated.rule_identity == "rule 3"

    def test_default_verdict_round_trips(self) -> None:
        # Arrange
        verdict = Verdict(
            table_id="points",
            table_version="1.0",
            outputs={"points": 0},
            matched_rule_index=None,
            is_default=True,
        )

        # Act / Assert
        assert Verdict.model_validate(verdict.model_dump()) == verdict
        assert verdict.rule_identity == "default, no rule matched"
