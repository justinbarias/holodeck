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
    ConfigError,
    DecisionTableError,
    GateSchemaError,
    GateValidationError,
    InputDataError,
    PolicyReviewError,
    WorkflowError,
)
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
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


def _break_gate_schema_structure(workflow_path: Path) -> None:
    """Leave the gate parseable JSON, but not a valid JSON Schema.

    ``required`` must be an array. Nothing about the file is wrong until the
    schema is judged as a schema, which is the whole point: the defect is
    invisible to a read-and-parse check and would otherwise surface only once
    the agent's output was handed to the validator.
    """
    (workflow_path.parent / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": "net_income",
            }
        ),
        encoding="utf-8",
    )


def _write_gate_schema(workflow_path: Path, schema: dict[str, Any]) -> None:
    """Replace the edge node's gate schema with ``schema``."""
    (workflow_path.parent / "gates" / "evidence.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )


def _gate_ref_to_a_remote_document(workflow_path: Path) -> None:
    """Point the gate at a schema only a network fetch could supply."""
    _write_gate_schema(
        workflow_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"net_income": {"$ref": "https://example.com/nope.json"}},
        },
    )


def _gate_ref_to_a_sibling_file(workflow_path: Path) -> None:
    """Point the gate at a sibling file, which retrieval refuses to load."""
    _write_gate_schema(
        workflow_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"net_income": {"$ref": "money.json"}},
        },
    )


def _gate_ref_typos_a_local_pointer(workflow_path: Path) -> None:
    """Typo a pointer into the gate's own ``$defs`` — no network involved.

    The likeliest ``$ref`` defect an author actually writes: ``$defs`` defines
    ``Money`` and the reference asks for ``Mony``. Everything needed to settle
    it is inside the file.
    """
    _write_gate_schema(
        workflow_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "$defs": {"Money": {"type": "number"}},
            "properties": {"net_income": {"$ref": "#/$defs/Mony"}},
            "required": ["net_income"],
        },
    )


def _gate_ref_hidden_under_a_property_named_default(workflow_path: Path) -> None:
    """Hide a remote ``$ref`` under a property literally named ``default``.

    The reference walk once skipped ``const``/``enum``/``default``/``examples``
    wherever the *key* appeared, so this entirely ordinary field name hid its
    subtree — and the remote reference in it — from load-time checking. The
    run then reached the edge agent and only discovered the defect after
    paying for the call, which is exactly what FR-003 forbids.
    """
    _write_gate_schema(
        workflow_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"default": {"$ref": "https://example.com/nope.json"}},
        },
    )


def _misname_table_input_root(workflow_path: Path) -> None:
    """Typo the root of a table input expression so nothing feeds it.

    The node declares ``inputs: [evidence, applicant]``; the table now reads
    ``evidnce``. The table alone cannot tell — it never sees the node — so only
    the runner can catch it, and it must do so before the edge agent runs.
    """
    _rewrite_table(
        workflow_path,
        inputs=[
            {
                "name": "surplus_ratio",
                "expression": (
                    "(evidnce.net_income - evidnce.expenses) / evidnce.net_income"
                ),
                "type": "number",
            },
            {
                "name": "residency_status",
                "expression": "applicant.residency_status",
                "type": "string",
            },
        ],
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
        (_break_gate_schema_structure, VALID_PAYLOAD, GateSchemaError),
        (_gate_ref_to_a_remote_document, VALID_PAYLOAD, GateSchemaError),
        (_gate_ref_to_a_sibling_file, VALID_PAYLOAD, GateSchemaError),
        (_gate_ref_typos_a_local_pointer, VALID_PAYLOAD, GateSchemaError),
        (
            _gate_ref_hidden_under_a_property_named_default,
            VALID_PAYLOAD,
            GateSchemaError,
        ),
        (_misname_table_input_root, VALID_PAYLOAD, WorkflowError),
        (lambda _: None, {}, InputDataError),
        (lambda _: None, {"applicant": {"residency_status": "maybe"}}, InputDataError),
    ],
    ids=[
        "cycle",
        "unresolved-reference",
        "missing-table",
        "unusable-gate-schema",
        "structurally-invalid-gate-schema",
        "gate-ref-to-a-remote-document",
        "gate-ref-to-a-sibling-file",
        "gate-ref-typos-a-local-pointer",
        "gate-ref-hidden-under-a-property-named-default",
        "table-input-names-an-undeclared-root",
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
    """Validation failures stop the run before a backend is ever selected.

    The four ``gate-ref-*`` cases are all 2020-12 gates, and that is not
    incidental: ``load_gate_schema`` walks a gate's ``$ref``\\ s only for the
    dialects in ``edge._WALKED_SPECIFICATIONS`` (2020-12 and 2019-09, plus a
    gate with no ``$schema``). An identical defect written in draft-07,
    draft-06, draft-04 or draft-03 is *not* caught here — it costs exactly one
    agent call and is then refused by the backstop in ``_apply_gate``, which
    ``test_a_skipped_dialect_gate_is_refused_after_one_agent_call`` pins.
    """
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
@pytest.mark.parametrize(
    "gate_schema",
    [
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "dependencies": {
                "net_income": ["expenses"],
                "expenses": {"$ref": "https://example.com/nope.json"},
            },
        },
        {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "type": "object",
            "extends": {"$ref": "https://example.com/nope.json"},
        },
    ],
    ids=["draft-07-dependencies-non-schema-first", "draft-03-extends-as-object"],
)
async def test_a_skipped_dialect_gate_is_refused_after_one_agent_call(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path, gate_schema: dict[str, Any]
) -> None:
    """The honest limit of FR-003, and the guarantee that replaces it.

    Both gates hold a remote ``$ref`` that no instance can satisfy, in the two
    shapes ``referencing``'s pre-2019 subresource tables mishandle: a
    ``dependencies`` whose first value is an array (the table then yields
    nothing at all) and a draft-03 ``extends`` written as an object (the table
    yields its *keys*). ``load_gate_schema`` does not walk these dialects at
    all, precisely so that nothing rests on those tables — so unlike the
    2020-12 equivalents in
    ``test_load_time_failures_never_invoke_an_agent``, each costs one agent
    call.

    What must still hold is the channel: exactly one invocation, then a
    ``GateSchemaError``. Not two calls, not a ``GateValidationError`` blaming
    the model for an authoring defect, and — the reason this test exists — not
    the bare ``AttributeError`` ``referencing`` raises when its crawl meets the
    ``str``/``list`` those tables hand it.
    """
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    _write_gate_schema(workflow_path, gate_schema)

    # Act / Assert
    with pytest.raises(GateSchemaError):
        await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    assert len(selector.calls) == 1
    assert len(selector.backend.messages) == 1


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
# FR-003 — the edge agent config is part of "everything resolves first".
# ---------------------------------------------------------------------------

SECOND_AGENT_YAML = """\
name: hardship-evidence-b
description: A second edge agent.
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "Extract the applicant's monthly expenses."
"""


def _add_second_edge_node(workflow_path: Path, agent_yaml: str | None) -> None:
    """Append a second edge node whose ``agent.yaml`` is ``agent_yaml``.

    Two edge nodes is the minimum that can show the defect: with one, the only
    agent is also the first, so a broken config costs nothing whichever half of
    the runner discovers it. ``None`` leaves the file absent entirely.
    """
    if agent_yaml is not None:
        (workflow_path.parent / "agents" / "evidence_b.yaml").write_text(
            agent_yaml, encoding="utf-8"
        )
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    data["nodes"].insert(
        1,
        {
            "id": "evidence_b",
            "edge": {"agent": "agents/evidence_b.yaml"},
            "gate": {"schema": "gates/evidence.schema.json"},
        },
    )
    workflow_path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_yaml", "expected"),
    [
        (None, HoloDeckFileNotFoundError),
        ("name: broken\nmodel:\n  provider: nope\n  name: x\n", ConfigError),
    ],
    ids=["missing-agent-yaml", "invalid-agent-yaml"],
)
async def test_a_broken_edge_agent_config_never_bills_an_earlier_node(
    monkeypatch: pytest.MonkeyPatch,
    workflow_path: Path,
    agent_yaml: str | None,
    expected: type[Exception],
) -> None:
    """An unloadable ``agent.yaml`` is an authoring defect, so it costs nothing.

    The workflow has two edge nodes; only the second one's config is broken. If
    the config were read at execution time, the first node's agent would have
    been invoked — and billed — before anything noticed.
    """
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    _add_second_edge_node(workflow_path, agent_yaml)

    # Act / Assert
    with pytest.raises(expected):
        await runner.run_workflow(workflow_path, VALID_PAYLOAD)

    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_swapped_after_preparation_is_not_the_one_invoked(
    monkeypatch: pytest.MonkeyPatch, workflow_path: Path
) -> None:
    """What runs is what was validated: the agent file is read exactly once.

    Rewriting ``agents/evidence.yaml`` between preparation and execution — with
    a different provider, no less — must not change which agent the backend
    selector is handed.
    """
    # Arrange
    selector = _install_backend(monkeypatch, _gated(dict(VALID_EVIDENCE)))
    prepared = runner.prepare_workflow(workflow_path, VALID_PAYLOAD)
    (workflow_path.parent / "agents" / "evidence.yaml").write_text(
        "name: swapped-in\n"
        "description: Not the agent that was validated.\n"
        "model:\n"
        "  provider: openai\n"
        "  name: gpt-4o\n"
        "instructions:\n"
        '  inline: "Do something else entirely."\n',
        encoding="utf-8",
    )

    # Act
    await runner.execute_workflow(prepared)

    # Assert — the prepared agent, not whatever the file says now.
    (invoked,) = selector.calls
    assert invoked.name == "hardship-evidence"
    assert invoked.model.provider.value == "anthropic"


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
