"""Tests for the workflow runner (036, T6).

Everything runs against a fake backend injected at the ``BackendSelector``
boundary, plus an autouse guard that fails the run if any concrete backend is
ever constructed — so "zero LLM calls" is enforced rather than assumed. The
load-time tests additionally assert the injected selector was never even asked
for a backend, which is the FR-003 claim: a cycle, an unresolved reference, a
missing table, an unusable gate, or an invalid fact stops the run before any
agent is invoked.

Covers US1 scenarios 1-3, the FR-030 review gate across all four
generated/reviewed combinations (SC-009), and the T8 human-node refusal.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError as PydanticValidationError

from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.errors import (
    DecisionTableError,
    GateSchemaError,
    GateValidationError,
    InputDataError,
    PolicyReviewError,
    WorkflowError,
)
from holodeck.lib.workflow import edge, runner
from holodeck.models.observability import ObservabilityConfig

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "workflow" / "single_gate"

# surplus_ratio = (5000 - 3000) / 5000 = 0.4 -> rule 1 -> "affordable".
VALID_EVIDENCE: dict[str, Any] = {"net_income": 5000, "expenses": 3000}
VALID_PAYLOAD: dict[str, Any] = {"applicant": {"residency_status": "verified"}}


class _FakeBackend:
    """Stands in for an ``AgentBackend`` — records calls, never talks to a model."""

    def __init__(self, result: ExecutionResult) -> None:
        self.result = result
        self.messages: list[str] = []
        self.torn_down = False

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        self.messages.append(message)
        return self.result

    async def teardown(self) -> None:
        self.torn_down = True


class _RecordingSelector:
    """Stands in for ``BackendSelector``; records every selection request."""

    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.calls: list[Any] = []

    async def select(
        self,
        agent: Any,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> _FakeBackend:
        self.calls.append(agent)
        return self.backend


@pytest.fixture(autouse=True)
def forbid_real_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blow up if any concrete backend is constructed (proves zero LLM calls)."""

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a real backend was constructed — this test must never reach an LLM"
        )

    monkeypatch.setattr(ClaudeBackend, "__init__", _boom)
    monkeypatch.setattr(OpenAIAgentsBackend, "__init__", _boom)


@pytest.fixture
def workflow_path(tmp_path: Path) -> Path:
    """A writable copy of the single-gate fixture; returns its workflow.yaml."""
    shutil.copytree(FIXTURE_DIR, tmp_path / "wf")
    return tmp_path / "wf" / "workflow.yaml"


def _install_backend(
    monkeypatch: pytest.MonkeyPatch, result: ExecutionResult
) -> _RecordingSelector:
    """Inject a fake backend at the ``BackendSelector`` boundary."""
    selector = _RecordingSelector(_FakeBackend(result))
    monkeypatch.setattr(edge, "BackendSelector", selector)
    return selector


def _gated(output: dict[str, Any] | None) -> ExecutionResult:
    """An agent result carrying ``output`` as its structured output."""
    return ExecutionResult(response="extraction complete", structured_output=output)


def _rewrite_table(workflow_path: Path, **fields: Any) -> None:
    """Merge ``fields`` into the fixture's decision table on disk."""
    table_path = workflow_path.parent / "tables" / "affordability.dmn.yaml"
    table = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    table.update(fields)
    table_path.write_text(yaml.safe_dump(table), encoding="utf-8")


# ---------------------------------------------------------------------------
# US1 scenario 1 — a valid edge output crosses the gate and the table decides.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_output_crosses_gate_and_policy_node_emits_verdict(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """A schema-valid extraction reaches the table, which emits the verdict."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    result = await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert
    assert result.order == ("evidence", "affordability")
    assert result.gated_outputs["evidence"].value == VALID_EVIDENCE
    verdict = result.verdicts["affordability"]
    assert verdict.outputs == {"affordability": "affordable"}
    assert verdict.table_id == "affordability"
    assert verdict.table_version == "2026-06-01.1"
    assert verdict.matched_rule_index == 1
    # Exactly one agent invocation, through the injected fake.
    assert len(selector.calls) == 1
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_prompt_carries_the_validated_facts_of_record(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """The edge agent is prompted with the validated facts of record as JSON."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert — the facts, and only the facts, are what the agent is handed.
    (message,) = selector.backend.messages
    assert "Facts of record" in message
    assert json.loads(message.split("\n", 1)[1]) == VALID_PAYLOAD


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undeclared_payload_keys_never_reach_the_edge_agent(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """Only declared facts of record are prompted; extra payload keys are dropped."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    payload = {**VALID_PAYLOAD, "undeclared": "should never be seen"}

    # Act
    await runner.run_workflow(workflow_path, payload)

    # Assert
    (message,) = selector.backend.messages
    assert "undeclared" not in message


# ---------------------------------------------------------------------------
# US1 scenario 2 — ungated output fails the run and produces no verdict.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("structured_output", "fragment"),
    [
        (None, "free text"),
        ({"net_income": 5000}, "violates the gate schema"),
        ({"net_income": "lots", "expenses": 3000}, "violates the gate schema"),
    ],
    ids=["free-text", "missing-field", "wrong-type"],
)
async def test_gate_rejection_fails_the_run_with_no_verdict(
    monkeypatch: pytest.MonkeyPatch,
    workflow_path: Path,
    structured_output: dict[str, Any] | None,
    fragment: str,
) -> None:
    """Free text or schema-invalid output is rejected; no verdict is produced."""
    # Arrange
    _install_backend(monkeypatch, _gated(structured_output))

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    assert excinfo.value.node_id == "evidence"
    assert fragment in str(excinfo.value)


# ---------------------------------------------------------------------------
# US1 scenario 3 — one OTel span per node, gated on the observability config.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spans_emitted_for_edge_and_policy_nodes(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """With observability enabled, each executed node emits one span."""
    # Arrange
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(runner, "get_tracer", provider.get_tracer)
    _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    await runner.run_workflow(
        workflow_path, VALID_PAYLOAD, ObservabilityConfig(enabled=True)
    )

    # Assert
    spans = exporter.get_finished_spans()
    kinds = {
        span.attributes["holodeck.workflow.node_id"]: span.attributes[  # type: ignore[index]
            "holodeck.workflow.node_kind"
        ]
        for span in spans
    }
    assert kinds == {"evidence": "edge", "affordability": "policy"}
    assert {span.name for span in spans} == {
        "holodeck.workflow.node.evidence",
        "holodeck.workflow.node.affordability",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_tracer_acquired_when_observability_is_disabled(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """Observability off means no spans and no tracer lookup at all."""

    # Arrange
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a tracer was acquired with observability disabled")

    monkeypatch.setattr(runner, "get_tracer", _boom)
    _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    result = await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert
    assert result.verdicts["affordability"].outputs == {"affordability": "affordable"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disabled_observability_config_emits_no_spans(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """An explicitly disabled ObservabilityConfig produces zero spans."""
    # Arrange
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(runner, "get_tracer", provider.get_tracer)
    _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    await runner.run_workflow(
        workflow_path, VALID_PAYLOAD, ObservabilityConfig(enabled=False)
    )

    # Assert
    assert exporter.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# FR-003 — every load-time failure happens before any agent is invoked.
# ---------------------------------------------------------------------------


def _make_cyclic(workflow_path: Path) -> None:
    """Rewrite the workflow as two policy nodes that reference each other."""
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "name": "cyclic",
                "version": "1.0.0",
                "nodes": [
                    {
                        "id": "a",
                        "decision": "tables/affordability.dmn.yaml",
                        "inputs": ["b"],
                    },
                    {
                        "id": "b",
                        "decision": "tables/affordability.dmn.yaml",
                        "inputs": ["a"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _make_unresolved(workflow_path: Path) -> None:
    """Point the policy node at an input nothing produces."""
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    data["nodes"][1]["inputs"] = ["evidence", "nowhere"]
    workflow_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _delete_table(workflow_path: Path) -> None:
    """Remove the referenced decision table."""
    (workflow_path.parent / "tables" / "affordability.dmn.yaml").unlink()


def _break_gate_schema(workflow_path: Path) -> None:
    """Make the edge node's gate schema unparseable."""
    (workflow_path.parent / "gates" / "evidence.schema.json").write_text(
        "not json at all", encoding="utf-8"
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "payload", "expected"),
    [
        (_make_cyclic, VALID_PAYLOAD, PydanticValidationError),
        (_make_unresolved, VALID_PAYLOAD, PydanticValidationError),
        (_delete_table, VALID_PAYLOAD, DecisionTableError),
        (_break_gate_schema, VALID_PAYLOAD, GateSchemaError),
        (lambda _: None, {}, InputDataError),
        (lambda _: None, {"applicant": {"residency_status": "maybe"}}, InputDataError),
    ],
    ids=[
        "cycle",
        "unresolved-reference",
        "missing-table",
        "unusable-gate-schema",
        "missing-fact",
        "schema-invalid-fact",
    ],
)
async def test_load_time_failures_never_invoke_an_agent(
    monkeypatch: pytest.MonkeyPatch,
    workflow_path: Path,
    mutate: Callable[[Path], None],
    payload: dict[str, Any],
    expected: type[Exception],
) -> None:
    """Validation failures stop the run before a backend is ever selected."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    mutate(workflow_path)

    # Act / Assert
    with pytest.raises(expected):
        await runner.run_workflow(workflow_path, payload)

    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_fact_fails_before_any_backend_is_constructed(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """FR-025: facts are validated before the first node, not alongside it.

    Guards the ``validate_input_data`` call itself: dropping it would let a
    schema-invalid fact flow into the run, and the edge agent would be invoked
    before anything noticed.
    """
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act / Assert
    with pytest.raises(InputDataError) as excinfo:
        await runner.run_workflow(
            workflow_path, {"applicant": {"residency_status": "maybe"}}
        )

    assert excinfo.value.name == "applicant"
    assert selector.calls == []
    assert selector.backend.messages == []


# ---------------------------------------------------------------------------
# input_data names are facts, not nodes — they are never scheduled.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fact_names_are_absent_from_the_executed_order(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """The executed order holds node ids only — never an input_data name."""
    # Arrange
    _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    result = await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert — 'applicant' is a fact in hand, not work to schedule.
    assert result.order == ("evidence", "affordability")
    assert "applicant" not in result.order


@pytest.mark.unit
@pytest.mark.asyncio
async def test_node_fed_only_by_facts_is_scheduled_as_a_graph_root(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """A node whose inputs are all facts still runs, and no fact is executed."""
    # Arrange — a workflow with no edge node at all: one policy node whose only
    # input is a fact of record. If facts were scheduled as graph nodes, the
    # runner would try to execute 'applicant' and fail to find it.
    (workflow_path.parent / "tables" / "residency.dmn.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "residency",
                "version": "2026-06-01.1",
                "hit_policy": "UNIQUE",
                "inputs": [
                    {
                        "name": "status",
                        "expression": "applicant.residency_status",
                        "type": "string",
                    }
                ],
                "outputs": [
                    {
                        "name": "eligible",
                        "type": "boolean",
                    }
                ],
                "rules": [
                    {"when": {"status": '"verified"'}, "then": {"eligible": True}},
                    {"when": {"status": '"unverified"'}, "then": {"eligible": False}},
                ],
            }
        ),
        encoding="utf-8",
    )
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "name": "facts-only",
                "version": "1.0.0",
                "input_data": {
                    "applicant": {"schema": "schemas/applicant.schema.json"}
                },
                "nodes": [
                    {
                        "id": "residency",
                        "decision": "tables/residency.dmn.yaml",
                        "inputs": ["applicant"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))

    # Act
    result = await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert
    assert result.order == ("residency",)
    assert result.verdicts["residency"].outputs == {"eligible": True}
    # No edge node exists, so nothing may have been invoked.
    assert selector.calls == []


# ---------------------------------------------------------------------------
# FR-030 / SC-009 — the review gate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generated_and_unreviewed_table_is_refused(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """A generated table with no reviewer is refused, before any agent runs."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    _rewrite_table(
        workflow_path,
        provenance={
            "generated_by": "claude-sonnet-4-20250514",
            "source": "Hardship Policy v4.2 §72",
        },
    )

    # Act / Assert
    with pytest.raises(PolicyReviewError) as excinfo:
        await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    assert excinfo.value.table_id == "affordability"
    assert "reviewed_by" in str(excinfo.value)
    assert selector.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provenance", "case"),
    [
        (None, "hand-authored, no provenance block"),
        ({"source": "Hardship Policy v4.2 §72"}, "neither field"),
        ({"reviewed_by": "policy.owner@example.gov"}, "reviewed but not generated"),
        (
            {
                "generated_by": "claude-sonnet-4-20250514",
                "reviewed_by": "policy.owner@example.gov",
            },
            "generated and reviewed",
        ),
    ],
    ids=["no-provenance", "neither", "reviewed-only", "generated-and-reviewed"],
)
async def test_review_gate_lets_every_other_combination_through(
    monkeypatch: pytest.MonkeyPatch,
    workflow_path: Path,
    provenance: dict[str, str] | None,
    case: str,
) -> None:
    """Only generated-without-reviewer is refused; the other three run."""
    # Arrange
    _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    if provenance is not None:
        _rewrite_table(workflow_path, provenance=provenance)

    # Act
    result = await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    # Assert
    assert result.verdicts["affordability"].outputs == {
        "affordability": "affordable"
    }, case


# ---------------------------------------------------------------------------
# Human nodes belong to T8.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_human_node_is_refused_pointing_at_t8(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """A requires_human node is refused at load, before any agent is invoked."""
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    data["nodes"][1]["requires_human"] = True
    workflow_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    # Act / Assert
    with pytest.raises(WorkflowError) as excinfo:
        await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    assert "affordability" in str(excinfo.value)
    assert "T8" in str(excinfo.value)
    assert selector.calls == []


# ---------------------------------------------------------------------------
# Workflow file handling.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_non_mapping_workflow_file_is_rejected(tmp_path: Path) -> None:
    """A workflow file that is not a YAML mapping fails with a config error."""
    # Arrange
    path = tmp_path / "workflow.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(Exception, match="must be a YAML mapping"):
        runner.prepare_workflow(path)
