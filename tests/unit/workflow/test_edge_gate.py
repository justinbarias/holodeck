"""Tests for the edge-node executor and schema gate (036, T5).

Every test runs against a fake backend injected at the ``BackendSelector``
boundary; an autouse guard fails the run if any concrete backend is ever
constructed, so "zero live LLM calls" is enforced rather than assumed.

Covers: a valid structured output crossing as the canonical value, free text
rejected, schema-invalid output rejected with the node id and the failure
named, an unreadable gate schema failing before any invocation, and the
channel discipline between a broken invocation (``ExecutionError``), an
authoring defect (``GateSchemaError``) and a rejected model output
(``GateValidationError``).
"""

from __future__ import annotations

import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.errors import (
    ExecutionError,
    GateSchemaError,
    GateValidationError,
)
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.lib.workflow import edge
from holodeck.models.workflow import EdgeNode

AGENT_YAML = """\
name: hardship-evidence
description: Edge agent under test
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "Extract the applicant's income evidence."
"""

GATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "net_income": {"type": "number"},
        "residency_status": {"type": "string", "enum": ["verified", "unverified"]},
    },
    "required": ["net_income", "residency_status"],
    "additionalProperties": False,
}

VALID_OUTPUT: dict[str, Any] = {"net_income": 4200.0, "residency_status": "verified"}


class _FakeBackend:
    """Stands in for an ``AgentBackend`` — records calls, never talks to a model."""

    def __init__(
        self,
        result: ExecutionResult,
        invoke_error: Exception | None = None,
        teardown_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.invoke_error = invoke_error
        self.teardown_error = teardown_error
        self.messages: list[str] = []
        self.torn_down = False

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        self.messages.append(message)
        if self.invoke_error is not None:
            raise self.invoke_error
        return self.result

    async def teardown(self) -> None:
        self.torn_down = True
        if self.teardown_error is not None:
            raise self.teardown_error


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
def workflow_dir(tmp_path: Path) -> Path:
    """A workflow directory holding an edge agent.yaml and a gate schema."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "evidence.yaml").write_text(AGENT_YAML, encoding="utf-8")
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "evidence.schema.json").write_text(
        json.dumps(GATE_SCHEMA), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def node() -> EdgeNode:
    """The edge node under test, with paths relative to workflow.yaml."""
    return EdgeNode(
        id="evidence",
        edge={"agent": "agents/evidence.yaml"},  # type: ignore[arg-type]
        gate={"schema": "gates/evidence.schema.json"},  # type: ignore[arg-type]
    )


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    result: ExecutionResult,
    invoke_error: Exception | None = None,
    teardown_error: Exception | None = None,
) -> _RecordingSelector:
    """Inject a fake backend at the ``BackendSelector`` boundary."""
    selector = _RecordingSelector(
        _FakeBackend(result, invoke_error=invoke_error, teardown_error=teardown_error)
    )
    monkeypatch.setattr(edge, "BackendSelector", selector)
    return selector


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_structured_output_crosses_the_gate(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A schema-valid structured output crosses as the canonical value."""
    # Arrange
    result = ExecutionResult(
        response="The applicant nets about 4.2k and residency looks fine.",
        structured_output=dict(VALID_OUTPUT),
    )
    selector = _install_backend(monkeypatch, result)

    # Act
    gated = await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    # Assert — the gated object, not the prose, is what crosses (FR-008).
    assert gated.node_id == "evidence"
    assert gated.value == VALID_OUTPUT
    # The prose is nowhere in what crosses — there is no field carrying it.
    assert result.response not in json.dumps(gated.model_dump())
    assert set(gated.model_dump()) == {"node_id", "value", "gate_schema"}
    # The schema is snapshotted by content, not by path (T10 replay).
    assert gated.gate_schema == GATE_SCHEMA
    # And no real backend was ever constructed — only the injected fake ran.
    assert len(selector.calls) == 1
    assert isinstance(selector.backend, _FakeBackend)
    assert selector.backend.messages == ["assess the evidence"]
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_free_text_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """An agent that returns prose only never crosses the gate."""
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(response="Looks affordable to me.", structured_output=None),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert "free text" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_schema_invalid_output_is_rejected_naming_the_failure(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A structured output that violates the gate schema is rejected loudly."""
    # Arrange — residency_status is outside the declared enum.
    _install_backend(
        monkeypatch,
        ExecutionResult(
            response="done",
            structured_output={"net_income": 4200.0, "residency_status": "probably"},
        ),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    message = str(excinfo.value)
    assert excinfo.value.node_id == "evidence"
    assert "evidence" in message
    assert "residency_status" in message
    assert "probably" in message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_required_field_is_rejected(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A structured output missing a required field names that field."""
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output={"net_income": 4200.0}),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert "residency_status" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_gate_schema_fails_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """An unreadable gate schema stops the node without spending an agent call."""
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").unlink()
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert "could not be read" in str(excinfo.value)
    # An authoring defect is not a rejection of model output; conflating the two
    # would inflate the SC-003 gate-rejection count with non-evidence.
    assert not isinstance(excinfo.value, GateValidationError)
    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_gate_schema_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A gate schema that is not parseable JSON stops the node."""
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        "{not json", encoding="utf-8"
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert "not valid JSON" in str(excinfo.value)
    assert not isinstance(excinfo.value, GateValidationError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_error_without_output_is_an_invocation_failure(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """An error that produced nothing to judge raises ExecutionError."""
    # Arrange — nothing was produced, so there is no evidence about the model.
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(
            response="",
            structured_output=None,
            is_error=True,
            error_reason="upstream 529 overloaded",
        ),
    )

    # Act / Assert
    with pytest.raises(ExecutionError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert not isinstance(excinfo.value, GateValidationError)
    assert "evidence" in str(excinfo.value)
    assert "upstream 529 overloaded" in str(excinfo.value)
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_error_with_output_is_judged_by_the_gate(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``is_error`` with output present is a gate rejection, not a broken call.

    This is the commonest real SC-003 event: ``ClaudeBackend.invoke_once``
    flags ``is_error`` when the model violates the agent's own
    ``response_format`` but still returns the offending object. That object is
    evidence about the model and must be judged on its merits.
    """
    # Arrange — residency_status is outside the gate's declared enum.
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(
            response="",
            structured_output={"net_income": 4200.0, "residency_status": "probably"},
            is_error=True,
            error_reason="Structured output schema validation failed",
        ),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    # The gate actually judged the value — it names the offending field/value.
    message = str(excinfo.value)
    assert excinfo.value.node_id == "evidence"
    assert "residency_status" in message
    assert "probably" in message
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_error_with_gate_valid_output_still_crosses(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """Output that satisfies *this node's* gate crosses, whatever the flag says.

    The gate, not the backend's own ``response_format``, is what the spine
    trusts; an object that satisfies the gate is a valid edge value.
    """
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(
            response="",
            structured_output=dict(VALID_OUTPUT),
            is_error=True,
            error_reason="Structured output schema validation failed",
        ),
    )

    # Act
    gated = await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    # Assert
    assert gated.value == VALID_OUTPUT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_ref_is_a_schema_error_and_never_hits_the_network(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A remote ``$ref`` fails loudly as an authoring defect, fetching nothing.

    jsonschema resolves ``$ref`` lazily at validate time and, left at its
    defaults, *retrieves* remote references over the network — an SSRF surface,
    a blocking ``urlopen`` on the event loop, and a gate whose effective content
    is not the snapshotted ``gate_schema`` (breaking T10 replay).
    """
    # Arrange — spy on the exact call jsonschema's default retriever makes.
    fetches: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        fetches.append(args[0] if args else kwargs)
        raise AssertionError("gate validation must never open a network connection")

    monkeypatch.setattr(urllib.request, "urlopen", _spy)
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "net_income": {"$ref": "http://127.0.0.1:1/evil.schema.json"}
                },
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )

    # Act / Assert — an unusable gate is the author's defect, not the model's.
    with pytest.raises(GateSchemaError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert not isinstance(excinfo.value, GateValidationError)
    assert fetches == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_teardown_failure_does_not_mask_a_gate_rejection(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A raising teardown must not replace the real failure."""
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(
            response="done",
            structured_output={"net_income": 4200.0, "residency_status": "probably"},
        ),
        teardown_error=RuntimeError("sdk subprocess would not stop"),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert "residency_status" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_teardown_failure_does_not_discard_a_valid_result(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A raising teardown must not destroy output that already crossed."""
    # Arrange
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
        teardown_error=RuntimeError("sdk subprocess would not stop"),
    )

    # Act
    gated = await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    # Assert
    assert gated.value == VALID_OUTPUT
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invocation_exception_is_an_execution_error(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A provider exception is typed as ExecutionError, preserving its cause."""
    # Arrange
    cause = ConnectionResetError("peer reset the connection")
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="", structured_output=None),
        invoke_error=cause,
    )

    # Act / Assert
    with pytest.raises(ExecutionError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert not isinstance(excinfo.value, GateValidationError)
    assert "evidence" in str(excinfo.value)
    assert excinfo.value.__cause__ is cause
    assert selector.backend.torn_down is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_object_gate_schema_fails_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A gate that declares a non-object type is caught before an agent call.

    The spine addresses an edge value by node id and dot-paths its fields, so
    ``{"type": "array"}`` is unaddressable. Catching it at load keeps an
    authoring defect out of the SC-003 gate-rejection count *and* out of the
    billing record.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps({"type": "array"}), encoding="utf-8"
    )
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=[1, 2, 3]),
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert not isinstance(excinfo.value, GateValidationError)
    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_constraints_are_enforced(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A declared ``format`` is enforced, not treated as an annotation."""
    # Arrange — 036 turns gate-schema `format: date` fields into date objects at
    # the workflow boundary, so an unparseable date must never cross the gate.
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"statement_date": {"type": "string", "format": "date"}},
                "required": ["statement_date"],
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(
            response="done", structured_output={"statement_date": "not-a-date"}
        ),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert "statement_date" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_agent_yaml_stops_the_node(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A missing ``agent.yaml`` fails as a HoloDeck FileNotFoundError."""
    # Arrange
    (workflow_dir / "agents" / "evidence.yaml").unlink()
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )

    # Act / Assert
    with pytest.raises(HoloDeckFileNotFoundError):
        await edge.execute_edge_node(node, workflow_dir, "assess the evidence")

    assert selector.calls == []


@pytest.mark.unit
def test_gated_output_is_immutable_through_nesting() -> None:
    """``frozen=True`` must be a real guarantee, not a top-level one.

    The gated value crosses a security boundary and is persisted for replay;
    a caller mutating the source object must not be able to change it.
    """
    # Arrange
    source_value = {"evidence": {"documents": ["payslip"]}}
    source_schema = {"type": "object", "properties": {"evidence": {"type": "object"}}}
    gated = edge.GatedOutput(
        node_id="evidence", value=source_value, gate_schema=source_schema
    )
    before = copy.deepcopy(gated.model_dump())

    # Act — mutate the caller's objects after construction.
    source_value["evidence"]["documents"].append("forged-bank-statement")
    source_schema["properties"]["evidence"] = {"type": "array"}

    # Assert
    assert gated.model_dump() == before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_object_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permissive gate cannot let a value the spine cannot name fields on through."""
    # Arrange — a schema that admits anything, and an array output.
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "evidence.yaml").write_text(AGENT_YAML, encoding="utf-8")
    (tmp_path / "open.schema.json").write_text("{}", encoding="utf-8")
    open_node = EdgeNode(
        id="evidence",
        edge={"agent": "agents/evidence.yaml"},  # type: ignore[arg-type]
        gate={"schema": "open.schema.json"},  # type: ignore[arg-type]
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=[1, 2, 3]),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await edge.execute_edge_node(open_node, tmp_path, "assess the evidence")

    assert "JSON object" in str(excinfo.value)
