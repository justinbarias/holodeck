"""Tests for the decision-table model and loader (036, T3).

Covers the three failure channels the model is built around: structural /
cross-field problems raise DecisionTableError, FEEL problems raise
FeelValidationError with a table/rule/cell locator, and field-shape problems
raise pydantic ValidationError. Also covers the YAML loader's read/parse guards.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from holodeck.lib.errors import DecisionTableError, FeelValidationError
from holodeck.models.decision_table import (
    DecisionTable,
    FeelType,
    HitPolicy,
    Provenance,
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


_PROVENANCE = {
    "generated_by": "claude-opus-5",
    "source": "Social Security Guide 3.11.13.50",
    "source_doc": "corpus/wa-guidelines-part-b-v1.24.pdf",
    "source_sha256": "35cfdef9",
    "reviewed_by": "J. Smith, Delegate",
    "reviewed_at": "2026-07-20T10:00:00Z",
}


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

    def test_malformed_yaml_raises_decision_table_error(self, tmp_path: Path) -> None:
        # A scanner error must leave through the module's declared channel,
        # not as a bare yaml.YAMLError a WorkflowError catcher would miss.
        path = tmp_path / "broken.dmn.yaml"
        path.write_text("id: [unclosed\n  bracket: {", encoding="utf-8")
        with pytest.raises(DecisionTableError) as exc:
            load_decision_table(path)
        assert "is not valid YAML" in str(exc.value)

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


@pytest.mark.unit
class TestProvenance:
    """The optional non-executable ``provenance`` block (FR-029)."""

    def test_full_block_loads(self) -> None:
        """Every provenance field parses, with reviewed_at as a datetime."""
        table = DecisionTable.model_validate(_table(provenance=_PROVENANCE))

        assert table.provenance is not None
        assert table.provenance.generated_by == "claude-opus-5"
        assert table.provenance.source == "Social Security Guide 3.11.13.50"
        assert table.provenance.source_doc == "corpus/wa-guidelines-part-b-v1.24.pdf"
        assert table.provenance.source_sha256 == "35cfdef9"
        assert table.provenance.reviewed_by == "J. Smith, Delegate"
        assert table.provenance.reviewed_at == datetime(
            2026, 7, 20, 10, 0, tzinfo=timezone.utc
        )

    def test_hand_authored_table_has_no_provenance(self) -> None:
        """A table omitting the block loads with provenance None."""
        assert DecisionTable.model_validate(_table()).provenance is None

    def test_partial_block_loads(self) -> None:
        """Every field is optional — a generated-but-unreviewed table loads."""
        table = DecisionTable.model_validate(
            _table(provenance={"generated_by": "claude-opus-5"})
        )

        assert table.provenance is not None
        assert table.provenance.reviewed_by is None

    def test_unknown_key_rejected(self) -> None:
        """extra='forbid' keeps the block from silently absorbing typos."""
        with pytest.raises(ValidationError):
            DecisionTable.model_validate(
                _table(provenance={**_PROVENANCE, "approved_by": "nobody"})
            )

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """The block survives a round-trip through the YAML loader."""
        path = tmp_path / "affordability.dmn.yaml"
        path.write_text(
            yaml.safe_dump(_table(provenance=_PROVENANCE)), encoding="utf-8"
        )

        table = load_decision_table(path)

        assert table.provenance is not None
        assert table.provenance.generated_by == "claude-opus-5"


@pytest.mark.unit
class TestAwaitingReview:
    """The FR-030 review-state predicate (enforcement itself lands in T6)."""

    @pytest.mark.parametrize(
        ("generated_by", "reviewed_by", "expected"),
        [
            ("claude-opus-5", None, True),
            ("claude-opus-5", "J. Smith", False),
            (None, None, False),
            (None, "J. Smith", False),
        ],
    )
    def test_predicate(
        self, generated_by: str | None, reviewed_by: str | None, expected: bool
    ) -> None:
        provenance = Provenance(generated_by=generated_by, reviewed_by=reviewed_by)

        assert provenance.awaiting_review is expected


@pytest.mark.unit
class TestProvenanceNotFeelReferenceable:
    """FR-032: provenance is metadata; FEEL must not be able to read it."""

    @pytest.mark.parametrize(
        "expression",
        ["provenance.reviewed_by", "provenance", "provenance.source_sha256 != null"],
    )
    def test_input_expression_referencing_provenance_rejected(
        self, expression: str
    ) -> None:
        inputs = [
            {"name": "surplus_ratio", "expression": expression, "type": "string"},
        ]
        rules = [{"when": {}, "then": {"affordability": "affordable"}}]

        with pytest.raises(FeelValidationError) as exc:
            DecisionTable.model_validate(
                _table(inputs=inputs, rules=rules, provenance=_PROVENANCE)
            )

        assert exc.value.locator == "table 'affordability' input 'surplus_ratio'"
        assert "non-executable metadata" in str(exc.value)

    @pytest.mark.parametrize("cell", ["provenance.reviewed_by", "= provenance"])
    def test_rule_cell_referencing_provenance_rejected(self, cell: str) -> None:
        rules = [
            {
                "when": {"residency_status": cell},
                "then": {"affordability": "affordable"},
            }
        ]

        with pytest.raises(FeelValidationError) as exc:
            DecisionTable.model_validate(_table(rules=rules))

        assert exc.value.locator == (
            "table 'affordability' rule 1 cell 'residency_status'"
        )
        assert "non-executable metadata" in str(exc.value)

    def test_similarly_named_variable_still_allowed(self) -> None:
        """Only the exact root name is reserved, not names containing it."""
        inputs = [
            {
                "name": "surplus_ratio",
                "expression": "provenance_score.value",
                "type": "number",
            },
        ]
        rules = [{"when": {}, "then": {"affordability": "affordable"}}]

        assert DecisionTable.model_validate(_table(inputs=inputs, rules=rules))
