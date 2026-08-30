"""The committed hardship fixtures are real, loadable artifacts (spec 040, T9).

Fixtures that only ever run inside a live integration test rot silently: the
integration suite is skipped by default, so a fixture can be broken for months
without a red mark. These are fast unit checks over the same files, asserting
each is a valid artifact of the model that consumes it — agents through
``Agent``, gates through ``load_gate_schema``/``check_gate``, ``worker.yaml``
through ``load_worker_config``, and the workflow module through the T5 sandbox
harness.

Two of the assertions are contracts rather than smoke tests:

* the decision table is *the same model* as the one the 036 conformance suite
  evaluates — asserted against that suite's own builder, not a copy of its
  values, so drift on either side fails here (AC-3 comparability), and
* the evidence gate's shape is the table's named-input shape, asserted by
  running a gated object straight through ``evaluate`` with no mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import GateValidationError
from holodeck.lib.workflow.edge import check_gate, load_gate_schema, resolve_agent_path
from holodeck.lib.workflow.table_eval import evaluate
from holodeck.models.agent import Agent
from holodeck.models.decision_table import load_decision_table
from holodeck.temporal.worker_config import WorkerConfig, load_worker_config
from tests.unit.temporal.test_sandbox_safety import _prepare
from tests.unit.workflow.test_table_eval import NAMED_INPUTS
from tests.unit.workflow.test_table_eval import _table as canonical_036_table

pytestmark = pytest.mark.unit

FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "integration"
    / "temporal"
    / "fixtures"
    / "hardship"
)

# A gate-conforming evidence object. Deliberately the same numbers the 036
# suite's NAMED_INPUTS carry, so the two verdicts are comparable.
GATED_EVIDENCE: dict[str, Any] = {
    "income": {"net": 5000, "expenses": 3000},
    "residency": {"status": "verified"},
}


@pytest.fixture(scope="module")
def worker_config() -> WorkerConfig:
    """The fixture ``worker.yaml``, loaded through the T7 loader."""
    return load_worker_config(FIXTURES / "worker.yaml")


def _node(config: WorkerConfig, node_id: str) -> Any:
    """Return the configured node with the given id.

    Args:
        config: The loaded worker configuration.
        node_id: The node id to find.

    Returns:
        The matching :class:`~holodeck.models.workflow.EdgeNode`.
    """
    return next(node for node in config.nodes if node.id == node_id)


class TestWorkerConfig:
    """``worker.yaml`` is a valid T7 document and its paths are confined."""

    def test_loads_with_both_nodes(self, worker_config: WorkerConfig) -> None:
        # Assert
        assert worker_config.temporal.task_queue == "hardship"
        assert [node.id for node in worker_config.nodes] == ["evidence", "letter"]
        assert worker_config.base_dir == FIXTURES

    def test_node_paths_resolve_inside_the_fixture_directory(
        self, worker_config: WorkerConfig
    ) -> None:
        # Act — resolve_agent_path raises ConfigError on an escape.
        paths = [
            resolve_agent_path(node, worker_config.base_dir)
            for node in worker_config.nodes
        ]

        # Assert
        for path in paths:
            assert path.is_relative_to(FIXTURES)
            assert path.exists()


class TestAgents:
    """Both agents load through the existing ``Agent`` model."""

    @pytest.mark.parametrize(
        ("node_id", "agent_name"),
        [
            ("evidence", "hardship-evidence-extractor"),
            ("letter", "hardship-letter-writer"),
        ],
    )
    def test_agent_loads_and_declares_structured_output(
        self, worker_config: WorkerConfig, node_id: str, agent_name: str
    ) -> None:
        # Arrange
        path = resolve_agent_path(_node(worker_config, node_id), FIXTURES)

        # Act
        agent = ConfigLoader().load_agent_yaml(str(path))

        # Assert — response_format is what makes a gate reachable at all.
        assert isinstance(agent, Agent)
        assert agent.name == agent_name
        assert agent.response_format is not None
        assert agent.response_format["type"] == "object"

    @pytest.mark.parametrize("node_id", ["evidence", "letter"])
    def test_response_format_matches_its_gate(
        self, worker_config: WorkerConfig, node_id: str
    ) -> None:
        """The agent is asked for exactly the shape the gate then enforces."""
        # Arrange
        node = _node(worker_config, node_id)
        agent = ConfigLoader().load_agent_yaml(str(resolve_agent_path(node, FIXTURES)))
        gate = load_gate_schema(node, FIXTURES)
        assert agent.response_format is not None

        # Assert — descriptions may differ; the enforced shape may not.
        assert sorted(agent.response_format["properties"]) == sorted(gate["properties"])
        assert sorted(agent.response_format["required"]) == sorted(gate["required"])


class TestGates:
    """The gate schemas load and actually gate."""

    @pytest.mark.parametrize("node_id", ["evidence", "letter"])
    def test_gate_schema_loads_closed(
        self, worker_config: WorkerConfig, node_id: str
    ) -> None:
        # Act
        gate = load_gate_schema(_node(worker_config, node_id), FIXTURES)

        # Assert — a closed gate is what keeps unasked-for keys out.
        assert gate["additionalProperties"] is False

    def test_evidence_gate_accepts_a_conforming_object(
        self, worker_config: WorkerConfig
    ) -> None:
        # Arrange
        gate = load_gate_schema(_node(worker_config, "evidence"), FIXTURES)

        # Act
        gated = check_gate(GATED_EVIDENCE, gate, node_id="evidence")

        # Assert
        assert gated == GATED_EVIDENCE

    def test_evidence_gate_rejects_an_unasked_for_key(
        self, worker_config: WorkerConfig
    ) -> None:
        # Arrange
        gate = load_gate_schema(_node(worker_config, "evidence"), FIXTURES)
        smuggled = {**GATED_EVIDENCE, "recommendation": "approve"}

        # Act / Assert
        with pytest.raises(GateValidationError):
            check_gate(smuggled, gate, node_id="evidence")

    def test_letter_gate_rejects_an_out_of_enum_tone(
        self, worker_config: WorkerConfig
    ) -> None:
        # Arrange
        gate = load_gate_schema(_node(worker_config, "letter"), FIXTURES)

        # Act / Assert
        with pytest.raises(GateValidationError):
            check_gate(
                {"letter": "Dear applicant", "tone": "jubilant"},
                gate,
                node_id="letter",
            )


class TestDecisionTable:
    """The fixture table is the 036 suite's table, not a lookalike."""

    def test_table_is_the_same_model_as_the_036_suite_evaluates(self) -> None:
        """Structural identity against the source of truth, not a re-assertion.

        ``tests/unit/workflow/test_table_eval.py::_table`` is where the 036
        conformance suite's table is defined. Comparing the loaded fixture to
        the object that builder returns means drift on *either* side fails
        here, which a hand-written copy of the expected values would not
        catch.
        """
        # Arrange
        expected = canonical_036_table()

        # Act
        loaded = load_decision_table(FIXTURES / "tables" / "hardship.dmn.yaml")

        # Assert
        assert loaded == expected

    def test_table_produces_the_036_suites_verdict_for_its_inputs(self) -> None:
        # Arrange
        loaded = load_decision_table(FIXTURES / "tables" / "hardship.dmn.yaml")

        # Act
        from_fixture = evaluate(loaded, NAMED_INPUTS)
        from_036 = evaluate(canonical_036_table(), NAMED_INPUTS)

        # Assert
        assert from_fixture.outputs == from_036.outputs
        assert from_fixture.rule_identity == from_036.rule_identity
        assert from_fixture.table_version == from_036.table_version

    def test_gated_evidence_is_the_tables_input_shape(
        self, worker_config: WorkerConfig
    ) -> None:
        """No mapping step: the gate output goes straight into ``evaluate``."""
        # Arrange
        gate = load_gate_schema(_node(worker_config, "evidence"), FIXTURES)
        gated = check_gate(GATED_EVIDENCE, gate, node_id="evidence")
        table = load_decision_table(FIXTURES / "tables" / "hardship.dmn.yaml")

        # Act
        verdict = evaluate(table, gated)

        # Assert
        assert verdict.outputs == {"affordability": "affordable"}


class TestSampleWorkflow:
    """The sample workflow module passes the T5 sandbox harness."""

    def test_workflow_module_validates_in_the_sandbox(self) -> None:
        # Arrange — the same harness, imported from the T5 suite so the
        # passthrough set cannot drift between the two.
        from tests.integration.temporal.fixtures.hardship.workflow import (
            HardshipWorkflow,
        )

        # Act / Assert — no exception is the assertion.
        _prepare(HardshipWorkflow)

    def test_table_is_loaded_once_at_import_time(self) -> None:
        """Decision 7: the table is module state, not per-run I/O."""
        # Arrange
        from tests.integration.temporal.fixtures.hardship import policy, workflow

        # Assert — the workflow holds the very object the policy module read.
        assert workflow.TABLE is policy.TABLE

    def test_activity_parameters_carry_caller_side_timeouts_and_retries(
        self,
    ) -> None:
        """Decision 10: every scheduling knob is on the workflow's call."""
        # Arrange
        from tests.integration.temporal.fixtures.hardship import workflow

        # Act
        evidence = workflow.EVIDENCE_PARAMETERS.to_activity_kwargs()
        letter = workflow.LETTER_PARAMETERS.to_activity_kwargs()

        # Assert — extraction retries (a gate rejection may pass next time),
        # letter writing does not (the decision is already made).
        assert evidence["start_to_close_timeout"].total_seconds() == 180
        assert evidence["retry_policy"].maximum_attempts == 3
        assert letter["retry_policy"].maximum_attempts == 1


class TestFixturesAreCommitted:
    """The fixture tree must actually be in the repository."""

    def test_every_fixture_file_exists(self) -> None:
        # Assert
        for relative in (
            "worker.yaml",
            "workflow.py",
            "policy.py",
            "agents/evidence.yaml",
            "agents/letter.yaml",
            "gates/evidence.schema.json",
            "gates/letter.schema.json",
            "tables/hardship.dmn.yaml",
        ):
            assert (FIXTURES / relative).is_file(), relative

    def test_gate_schemas_are_valid_json(self) -> None:
        # Assert
        for name in ("evidence.schema.json", "letter.schema.json"):
            parsed = json.loads((FIXTURES / "gates" / name).read_text(encoding="utf-8"))
            assert isinstance(parsed, dict)
