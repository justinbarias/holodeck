"""Tests for the decision-table model and loader (036, T3).

Covers the three failure channels the model is built around: structural /
cross-field problems raise DecisionTableError, FEEL problems raise
FeelValidationError with a table/rule/cell locator, and field-shape problems
raise pydantic ValidationError. Also covers the YAML loader's read/parse guards.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from holodeck.lib.errors import DecisionTableError, FeelValidationError
from holodeck.models.decision_table import (
    DecisionTable,
    FeelType,
    HitPolicy,
    load_decision_table,
)


def _table(**overrides: object) -> dict:
    """A minimal valid single-output decision-table dict."""
    table = {
        "id": "affordability",
        "name": "Hardship affordability assessment",
        "version": "2026-06-01.1",
        "hit_policy": "UNIQUE",
        "inputs": [
            {
                "name": "surplus_ratio",
                "expression": "(income.net - income.expenses) / income.net",
                "type": "number",
            },
            {
                "name": "residency_status",
                "expression": "residency.status",
                "type": "string",
            },
        ],
        "outputs": [
            {
                "name": "affordability",
                "type": "string",
                "values": ["affordable", "marginal", "unaffordable"],
            },
        ],
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
    return table


@pytest.mark.unit
class TestValidTable:
    """A well-formed table parses with typed enums."""

    def test_parses(self) -> None:
        table = DecisionTable.model_validate(_table())
        assert table.id == "affordability"
        assert table.hit_policy is HitPolicy.UNIQUE
        assert table.inputs[0].type is FeelType.NUMBER
        assert table.outputs[0].values == ["affordable", "marginal", "unaffordable"]
        assert len(table.rules) == 3

    @pytest.mark.parametrize("policy", ["UNIQUE", "FIRST", "PRIORITY"])
    def test_all_hit_policies_accepted(self, policy: str) -> None:
        table = DecisionTable.model_validate(_table(hit_policy=policy))
        assert table.hit_policy.value == policy

    def test_irrelevant_cell_allowed(self) -> None:
        # A rule that omits a column (and one with an explicit "-") is valid.
        rules = [
            {
                "when": {"surplus_ratio": "< 0.10"},
                "then": {"affordability": "unaffordable"},
            },
            {"when": {"residency_status": "-"}, "then": {"affordability": "marginal"}},
        ]
        table = DecisionTable.model_validate(_table(rules=rules))
        assert len(table.rules) == 2

    def test_default_entry_allowed(self) -> None:
        table = DecisionTable.model_validate(
            _table(default={"affordability": "unaffordable"})
        )
        assert table.default == {"affordability": "unaffordable"}


@pytest.mark.unit
class TestStructuralErrors:
    """Cross-field problems raise DecisionTableError (not pydantic)."""

    def test_duplicate_input_name(self) -> None:
        inputs = [
            {"name": "dup", "expression": "income.net", "type": "number"},
            {"name": "dup", "expression": "residency.status", "type": "string"},
        ]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(inputs=inputs))
        assert "duplicate input" in str(exc.value)

    def test_duplicate_output_name(self) -> None:
        outputs = [
            {"name": "dup", "type": "string"},
            {"name": "dup", "type": "string"},
        ]
        rules = [{"when": {}, "then": {"dup": "x"}}]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(outputs=outputs, rules=rules))
        assert "duplicate output" in str(exc.value)

    def test_when_references_unknown_input(self) -> None:
        rules = [{"when": {"ghost": ">= 1"}, "then": {"affordability": "affordable"}}]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(rules=rules))
        assert "unknown input 'ghost'" in str(exc.value)
        assert "rule 1" in str(exc.value)

    def test_then_references_unknown_output(self) -> None:
        rules = [{"when": {}, "then": {"ghost": "x", "affordability": "affordable"}}]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(rules=rules))
        assert "unknown output" in str(exc.value)

    def test_then_missing_declared_output(self) -> None:
        # Two outputs; a rule that produces only one is incomplete.
        outputs = [
            {"name": "affordability", "type": "string"},
            {"name": "tier", "type": "string"},
        ]
        rules = [{"when": {}, "then": {"affordability": "affordable"}}]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(outputs=outputs, rules=rules))
        assert "missing output" in str(exc.value)
        assert "tier" in str(exc.value)

    def test_then_value_not_in_declared_values(self) -> None:
        rules = [{"when": {}, "then": {"affordability": "nope"}}]
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(rules=rules))
        assert "not one of" in str(exc.value)

    def test_default_value_not_in_declared_values(self) -> None:
        with pytest.raises(DecisionTableError) as exc:
            DecisionTable.model_validate(_table(default={"affordability": "nope"}))
        assert "default" in str(exc.value)
        assert "not one of" in str(exc.value)


@pytest.mark.unit
class TestFeelErrors:
    """FEEL problems raise FeelValidationError with a precise locator."""

    def test_out_of_subset_input_expression(self) -> None:
        inputs = [
            {
                "name": "surplus_ratio",
                "expression": "some x in [1, 2] satisfies x > 1",
                "type": "number",
            },
            {
                "name": "residency_status",
                "expression": "residency.status",
                "type": "string",
            },
        ]
        with pytest.raises(FeelValidationError) as exc:
            DecisionTable.model_validate(_table(inputs=inputs))
        assert exc.value.locator == "table 'affordability' input 'surplus_ratio'"

    def test_malformed_rule_cell(self) -> None:
        rules = [
            {"when": {"surplus_ratio": ">="}, "then": {"affordability": "affordable"}}
        ]
        with pytest.raises(FeelValidationError) as exc:
            DecisionTable.model_validate(_table(rules=rules))
        assert exc.value.locator == (
            "table 'affordability' rule 1 cell 'surplus_ratio'"
        )


@pytest.mark.unit
class TestFieldShapeErrors:
    """Plain field-shape problems surface as pydantic ValidationError."""

    def test_missing_version(self) -> None:
        table = _table()
        del table["version"]
        with pytest.raises(ValidationError):
            DecisionTable.model_validate(table)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DecisionTable.model_validate(_table(sneaky="x"))

    def test_invalid_hit_policy(self) -> None:
        with pytest.raises(ValidationError):
            DecisionTable.model_validate(_table(hit_policy="SOMETIMES"))

    def test_empty_inputs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecisionTable.model_validate(_table(inputs=[]))


@pytest.mark.unit
class TestLoader:
    """load_decision_table read/parse guards and round-trip."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "affordability.dmn.yaml"
        path.write_text(yaml.safe_dump(_table()), encoding="utf-8")
        table = load_decision_table(path)
        assert table.id == "affordability"
        assert table.hit_policy is HitPolicy.UNIQUE

    def test_missing_file_raises_decision_table_error(self, tmp_path: Path) -> None:
        with pytest.raises(DecisionTableError) as exc:
            load_decision_table(tmp_path / "nope.dmn.yaml")
        assert "could not read" in str(exc.value)

    def test_non_mapping_yaml_raises_decision_table_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.dmn.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(DecisionTableError) as exc:
            load_decision_table(path)
        assert "must be a YAML mapping" in str(exc.value)

    def test_structural_error_propagates_from_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.dmn.yaml"
        rules = [{"when": {"ghost": ">= 1"}, "then": {"affordability": "affordable"}}]
        path.write_text(yaml.safe_dump(_table(rules=rules)), encoding="utf-8")
        with pytest.raises(DecisionTableError):
            load_decision_table(path)


class TestPriorityRequiresValues:
    """PRIORITY ranks by each output's ``values`` ordering (FR-011).

    Without ``values`` there is nothing to rank by, so the table would silently
    degrade to first-match instead of failing. These pin the load-time error.
    """

    def test_priority_without_values_raises(self) -> None:
        """PRIORITY + an output missing 'values' fails at load with a locator."""
        table = _table(hit_policy="PRIORITY")
        table["outputs"] = [{"name": "affordability", "type": "string"}]

        with pytest.raises(DecisionTableError) as exc_info:
            DecisionTable.model_validate(table)

        message = str(exc_info.value)
        assert "PRIORITY" in message
        assert "affordability" in message

    def test_priority_with_values_is_accepted(self) -> None:
        """The same table validates once 'values' declares the ordering."""
        assert DecisionTable.model_validate(_table(hit_policy="PRIORITY"))

    @pytest.mark.parametrize("policy", ["UNIQUE", "FIRST"])
    def test_other_policies_do_not_require_values(self, policy: str) -> None:
        """Only PRIORITY needs an ordering; UNIQUE/FIRST are unaffected."""
        table = _table(hit_policy=policy)
        table["outputs"] = [{"name": "affordability", "type": "string"}]

        assert DecisionTable.model_validate(table)
