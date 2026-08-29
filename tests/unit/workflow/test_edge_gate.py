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
from typing import Any, NamedTuple

import pytest

from holodeck.config.context import agent_base_dir
from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.errors import (
    ConfigError,
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
response_format:
  type: object
  properties:
    net_income:
      type: number
    residency_status:
      type: string
"""

# The same agent with no response_format: it can never produce structured
# output, so execute_edge_node must refuse it before any backend is built.
AGENT_YAML_NO_RESPONSE_FORMAT = """\
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
    # execute_edge_node imports BackendSelector lazily (so the pure gate half
    # of edge.py stays SDK-free), so the patch lands on the selector module.
    monkeypatch.setattr("holodeck.lib.backends.selector.BackendSelector", selector)
    return selector


async def _execute(
    node: EdgeNode, workflow_dir: Path, message: str
) -> edge.GatedOutput:
    """Resolve the node's gate and agent, then run it, exactly as the runner does.

    ``execute_edge_node`` reads neither file — the runner resolves every gate
    *and* every ``agent.yaml`` at preparation and hands the results down, so
    what runs is what was validated. Sequencing the three steps here keeps
    "an unusable gate or agent costs no agent call" a property these tests
    still observe rather than one they assume.
    """
    gate_schema = edge.load_gate_schema(node, workflow_dir)
    agent_path = edge.resolve_agent_path(node, workflow_dir)
    agent = ConfigLoader().load_agent_yaml(str(agent_path))
    return await edge.execute_edge_node(node, agent, agent_path, message, gate_schema)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_without_response_format_is_refused_before_any_call(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """No response_format means no structured output is ever possible.

    Without the guard every run would spend a model call to land at the
    gate's "free text" rejection — an SC-003 rejection charged to a model
    that was never asked for structure. The autouse forbid_real_backends
    fixture proves nothing was built; no fake backend is installed at all.
    """
    # Arrange
    (workflow_dir / "agents" / "evidence.yaml").write_text(
        AGENT_YAML_NO_RESPONSE_FORMAT, encoding="utf-8"
    )

    # Act / Assert
    with pytest.raises(ConfigError) as exc:
        await _execute(node, workflow_dir, "Extract the evidence.")
    assert "response_format" in str(exc.value)


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
    gated = await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
    gated = await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert not isinstance(excinfo.value, GateValidationError)
    assert fetches == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ref", "reason"),
    [
        ("https://example.com/nope.json", "remote"),
        ("money.json", "sibling file"),
        ("#/$defs/Mony", "typo'd pointer into the gate's own $defs"),
    ],
    ids=["remote", "sibling-file", "typo-local-pointer"],
)
def test_unresolvable_ref_is_settled_at_load(
    workflow_dir: Path, node: EdgeNode, ref: str, reason: str
) -> None:
    """A ``$ref`` the gate cannot resolve is an authoring defect, found at load.

    jsonschema resolves ``$ref`` lazily by default, but that is a default, not
    a constraint: the registry that refuses retrieval can settle every one of
    these without a network round trip, so none of them needs to cost an agent
    call. The local typo is the case that matters most — it is the ``$ref``
    defect an author actually writes, and nothing about it is remote.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "$defs": {"Money": {"type": "number"}},
                "properties": {"net_income": {"$ref": ref}},
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        edge.load_gate_schema(node, workflow_dir)

    assert excinfo.value.node_id == "evidence"
    assert ref in str(excinfo.value), reason


@pytest.mark.unit
def test_a_bundled_metaschema_ref_is_accepted_at_load(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """The load-time walk must be no stricter than the validator itself.

    ``jsonschema`` combines a caller's registry with the metaschemas it ships
    before resolving anything, so a reference to a metaschema resolves at
    validate time with no retrieval. Resolving the walk against the bare
    no-retrieval registry would refuse a gate the gate can actually use.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "nested": {"$ref": "https://json-schema.org/draft/2020-12/schema"}
                },
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    assert edge.load_gate_schema(node, workflow_dir)["type"] == "object"


@pytest.mark.unit
def test_a_resolvable_local_ref_is_accepted_at_load(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """The load-time walk must not over-reject a gate that is self-contained."""
    # Arrange — refs under $defs, properties, items and anyOf, plus a `$ref`
    # key that is a property *name* and a literal `const` that merely looks
    # like a reference.
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {
            "Money": {"type": "number"},
            "Line": {
                "type": "object",
                "properties": {"amt": {"$ref": "#/$defs/Money"}},
            },
        },
        "properties": {
            "net_income": {"$ref": "#/$defs/Money"},
            "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
            "either": {"anyOf": [{"$ref": "#/$defs/Money"}, {"type": "null"}]},
            "$ref": {"type": "string"},
            "tag": {"const": {"$ref": "#/$defs/NotAReference"}},
        },
    }
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    # Act / Assert
    assert edge.load_gate_schema(node, workflow_dir) == schema


D2020 = "https://json-schema.org/draft/2020-12/schema"

#: A reference nothing in the registry can satisfy and retrieval will not fetch.
HIDDEN_REF = "https://remote.invalid/hidden.json"


def _schema_hiding_a_ref_under(container: str, name: str) -> dict[str, Any]:
    """Build a gate whose only broken ``$ref`` sits under an author-chosen name.

    ``name`` is one of the keywords whose *value* is literal instance data
    (``const``/``enum``/``default``/``examples``). Here it is not in keyword
    position at all — it names a property, a pattern, a dependent schema or a
    definition — so the subschema underneath it is a subschema, and the broken
    reference in it is a reference.

    Args:
        container: The keyword whose keys are author-chosen names.
        name: The author-chosen name, deliberately spelled like a keyword.

    Returns:
        A structurally valid gate schema carrying exactly one unresolvable ref.
    """
    broken = {"$ref": HIDDEN_REF}
    if container == "$defs":
        # Referenced, so validate time would reach it too — this case is about
        # position-awareness, not about the unreferenced-$defs policy.
        return {
            "$schema": D2020,
            "type": "object",
            "properties": {"a": {"$ref": f"#/$defs/{name}"}},
            "$defs": {name: broken},
        }
    return {"$schema": D2020, "type": "object", container: {name: broken}}


@pytest.mark.unit
@pytest.mark.parametrize("name", ["const", "enum", "default", "examples"])
@pytest.mark.parametrize(
    "container", ["properties", "patternProperties", "dependentSchemas", "$defs"]
)
def test_a_property_named_like_a_literal_keyword_does_not_hide_a_ref(
    workflow_dir: Path, node: EdgeNode, container: str, name: str
) -> None:
    """The keys under ``properties`` and friends are names, not keywords.

    Skipping ``const``/``enum``/``default``/``examples`` wherever the *key*
    appeared meant a property called ``default`` hid its whole subtree from the
    walk, remote ``$ref``\\ s included — an FR-003 hole reachable by naming a
    field something entirely ordinary. Descent is now decided by position, so
    an author-chosen name never suppresses anything.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(_schema_hiding_a_ref_under(container, name)), encoding="utf-8"
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        edge.load_gate_schema(node, workflow_dir)

    assert excinfo.value.node_id == "evidence"
    assert HIDDEN_REF in str(excinfo.value)


@pytest.mark.unit
def test_a_container_named_like_a_container_is_still_walked(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """Nesting a keyword name under an author-chosen name must not confuse it.

    A property called ``properties``, holding a property called ``const``,
    holding a ``$defs`` entry called ``$defs``: every one of those names is
    data, and the broken reference at the bottom is still a reference.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": D2020,
                "type": "object",
                "properties": {
                    "properties": {
                        "type": "object",
                        "properties": {
                            "const": {
                                "$defs": {"$defs": {"$ref": HIDDEN_REF}},
                                "$ref": "#/properties/properties/properties/const"
                                "/$defs/$defs",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        edge.load_gate_schema(node, workflow_dir)

    assert HIDDEN_REF in str(excinfo.value)


@pytest.mark.unit
def test_a_ref_shaped_literal_is_data_not_a_reference(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """In keyword position, ``const``/``enum``/``default``/``examples`` are data.

    The mirror of the test above: the same unresolvable string, this time
    inside values the validator never reads as schemas, must not be walked —
    otherwise a gate that pins a literal ``{"$ref": ...}`` object becomes
    unloadable while ``validate()`` accepts it happily.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": D2020,
                "type": "object",
                "properties": {
                    "pinned": {"const": {"$ref": HIDDEN_REF}},
                    "chosen": {"enum": [{"$ref": HIDDEN_REF}, None]},
                    "seeded": {"type": "object", "default": {"$ref": HIDDEN_REF}},
                    "shown": {"type": "object", "examples": [{"$ref": HIDDEN_REF}]},
                },
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    assert edge.load_gate_schema(node, workflow_dir)["type"] == "object"


@pytest.mark.unit
def test_a_nested_id_rebases_a_relative_ref_at_load(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """A legal, self-contained gate with a nested ``$id`` must load.

    ``#/$defs/B`` inside ``inner.json`` is resolved against *that* resource,
    not against the document root. Resolving every reference against the root
    judged this one against the wrong base and refused a workflow the validator
    runs without complaint — a false rejection, which under FR-003 is far worse
    than a missed early catch: it makes a correct workflow permanently
    unrunnable and blames the author for it.
    """
    # Arrange
    schema = {
        "$schema": D2020,
        "$id": "https://example.com/root.json",
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Inner"}},
        "$defs": {
            "Inner": {
                "$id": "https://example.com/inner.json",
                "type": "object",
                "$defs": {"B": {"type": "string"}},
                "properties": {"b": {"$ref": "#/$defs/B"}},
            }
        },
    }
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    # Act / Assert
    assert edge.load_gate_schema(node, workflow_dir) == schema


@pytest.mark.unit
def test_an_unresolvable_ref_in_an_unreferenced_defs_is_refused(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """Pins the one place load is deliberately stricter than ``validate()``.

    Nothing points at ``Unused``, so no instance ever makes the validator
    resolve it and ``validate()`` accepts the gate for every input. Load
    refuses it anyway: a reference that cannot resolve *at all* is an authoring
    defect whether or not this run's output reaches it, and the same reasoning
    covers the losing branch of an ``if``/``then``. The cost of refusing is a
    message naming the ref; the cost of accepting is a gate that silently stops
    constraining anything the first time an instance does reach that branch.

    This disagreement is intentional and narrow. It is recorded here so that
    changing it is a decision rather than an accident.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": D2020,
                "type": "object",
                "$defs": {"Unused": {"$ref": HIDDEN_REF}},
                "properties": {"net_income": {"type": "number"}},
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        edge.load_gate_schema(node, workflow_dir)

    assert HIDDEN_REF in str(excinfo.value)


# ---------------------------------------------------------------------------
# The differential: load-time acceptance must track validate-time acceptance.
# ---------------------------------------------------------------------------

_DRAFT_LOCAL_REF: dict[str, Any] = {
    "type": "object",
    "definitions": {"Money": {"type": "number"}},
    "properties": {"net_income": {"$ref": "#/definitions/Money"}},
}

#: ``(schema, instance, load_and_validate_agree_on_acceptance)``. Every entry
#: asserts the *same* verdict from both ends; the last two are the reject side,
#: without which the differential could pass by accepting everything.
DIFFERENTIAL_GATES: list[tuple[dict[str, Any], dict[str, Any], bool]] = [
    (
        {
            "$schema": D2020,
            "$id": "https://example.com/root.json",
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Inner"}},
            "$defs": {
                "Inner": {
                    "$id": "https://example.com/inner.json",
                    "type": "object",
                    "$defs": {"B": {"type": "string"}},
                    "properties": {"b": {"$ref": "#/$defs/B"}},
                }
            },
        },
        {"a": {"b": "x"}},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"$anchor": "money", "type": "number"}},
            "properties": {"net_income": {"$ref": "#money"}},
        },
        {"net_income": 4200.0},
        True,
    ),
    (
        {"$schema": D2020, "type": "object", "properties": {"child": {"$ref": "#"}}},
        {"child": {"child": {}}},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "properties": {"$ref": {"type": "string"}},
        },
        {"$ref": "a property name, not a reference"},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "properties": {"tag": {"const": {"$ref": HIDDEN_REF}}},
        },
        {"tag": {"$ref": HIDDEN_REF}},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"type": "number"}},
            "properties": {
                name: {"$ref": "#/$defs/Money"}
                for name in ("const", "enum", "default", "examples")
            },
        },
        {"const": 1.0, "enum": 2.0, "default": 3.0, "examples": 4.0},
        True,
    ),
    (
        {"$schema": "http://json-schema.org/draft-04/schema#", **_DRAFT_LOCAL_REF},
        {"net_income": 4200.0},
        True,
    ),
    (
        {"$schema": "http://json-schema.org/draft-06/schema#", **_DRAFT_LOCAL_REF},
        {"net_income": 4200.0},
        True,
    ),
    (
        {"$schema": "http://json-schema.org/draft-07/schema#", **_DRAFT_LOCAL_REF},
        {"net_income": 4200.0},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"type": "number"}},
            "patternProperties": {"^amount_": {"$ref": "#/$defs/Money"}},
        },
        {"amount_net": 4200.0},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"type": "number"}, "Text": {"type": "string"}},
            "if": {"properties": {"kind": {"const": "cash"}}, "required": ["kind"]},
            "then": {"properties": {"v": {"$ref": "#/$defs/Money"}}},
            "else": {"properties": {"v": {"$ref": "#/$defs/Text"}}},
        },
        {"kind": "cash", "v": 4200.0},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Empty": {"type": "object", "maxProperties": 0}},
            "not": {"$ref": "#/$defs/Empty"},
        },
        {"net_income": 4200.0},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"type": "number"}},
            "properties": {
                "pair": {
                    "type": "array",
                    "prefixItems": [{"$ref": "#/$defs/Money"}, {"type": "string"}],
                }
            },
        },
        {"pair": [4200.0, "aud"]},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"NeedsTotal": {"required": ["total"]}},
            "dependentSchemas": {"lines": {"$ref": "#/$defs/NeedsTotal"}},
        },
        {"lines": [], "total": 4200.0},
        True,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "properties": {"const": {"$ref": HIDDEN_REF}},
        },
        {"const": 4200.0},
        False,
    ),
    (
        {
            "$schema": D2020,
            "type": "object",
            "$defs": {"Money": {"type": "number"}},
            "properties": {"net_income": {"$ref": "#/$defs/Mony"}},
        },
        {"net_income": 4200.0},
        False,
    ),
]

DIFFERENTIAL_IDS = [
    "nested-id-rebases-a-relative-pointer",
    "anchor",
    "self-ref-to-the-document-root",
    "a-property-named-ref",
    "a-ref-shaped-const-literal",
    "properties-named-like-literal-keywords",
    "draft-04",
    "draft-06",
    "draft-07",
    "pattern-properties",
    "if-then-else",
    "not",
    "prefix-items",
    "dependent-schemas",
    "remote-ref-under-a-property-named-const",
    "typo-local-pointer",
]


def _runtime_gate_is_usable(schema: dict[str, Any], instance: dict[str, Any]) -> bool:
    """Report whether the runtime gate could enforce ``schema`` on ``instance``.

    Runs the real runtime path, ``_apply_gate``, rather than a re-derived
    validator, so the differential compares load against what actually happens
    after an agent call.

    Args:
        schema: The gate schema.
        instance: The structured output to present to the gate.

    Returns:
        ``True`` if the schema was usable — including when it was usable and
        simply rejected the instance; ``False`` only if the *gate* was refused.
    """
    result = ExecutionResult(response="done", structured_output=instance)
    try:
        edge._apply_gate("evidence", result, schema)
    except GateSchemaError:
        return False
    except GateValidationError:
        return True
    return True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("schema", "instance", "accepted"), DIFFERENTIAL_GATES, ids=DIFFERENTIAL_IDS
)
def test_load_time_acceptance_agrees_with_validate_time(
    workflow_dir: Path,
    node: EdgeNode,
    schema: dict[str, Any],
    instance: dict[str, Any],
    accepted: bool,
) -> None:
    """The governing rule for :func:`edge.load_gate_schema`, pinned per shape.

    Load-time rejection exists to save an agent call, so the two verdicts must
    not diverge. A missed early catch costs one call and is caught by the
    backstop in ``_apply_gate``; a false rejection makes a legal workflow
    permanently unrunnable. Every shape here — nested ``$id``, ``$anchor``,
    self-reference, author-chosen names that collide with keywords, the older
    drafts, and the applicator keywords whose subschemas live in different
    shapes — must get the same answer from both ends.

    The one deliberate exception is covered by
    ``test_an_unresolvable_ref_in_an_unreferenced_defs_is_refused``.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    # Act
    try:
        edge.load_gate_schema(node, workflow_dir)
        load_accepted = True
    except GateSchemaError:
        load_accepted = False
    runtime_accepted = _runtime_gate_is_usable(schema, instance)

    # Assert
    assert load_accepted is accepted
    assert runtime_accepted is accepted


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
        await _execute(node, workflow_dir, "assess the evidence")

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
    gated = await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

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
        await _execute(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert not isinstance(excinfo.value, GateValidationError)
    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structurally_invalid_gate_schema_fails_before_any_invocation(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A gate that is not a valid JSON Schema is caught before an agent call.

    ``required`` must be an array. The file reads and parses, so only the
    metaschema can tell — and judging it against the model's output instead
    would bill an agent call for a typo the author could have been shown.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": "net_income",
            }
        ),
        encoding="utf-8",
    )
    selector = _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        await _execute(node, workflow_dir, "assess the evidence")

    assert excinfo.value.node_id == "evidence"
    assert "not a valid JSON Schema" in str(excinfo.value)
    assert not isinstance(excinfo.value, GateValidationError)
    assert selector.calls == []
    assert selector.backend.messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_format_constraints_are_enforced(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """A ``format`` with a registered checker is enforced, not annotated.

    The gate installs jsonschema's ``FORMAT_CHECKER``, so every format that
    checker implements is a constraint. A format name it does not implement
    stays an annotation — see
    ``test_a_format_with_no_registered_checker_is_accepted_and_ignored``.
    """
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
        await _execute(node, workflow_dir, "assess the evidence")

    assert "statement_date" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_date_time_format_constraint_is_enforced(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``format: date-time`` is enforced by a declared dependency, not by luck.

    Before pinning ``jsonschema[format-nongpl]`` this format only happened to
    be checked because ``rfc3339-validator`` was pulled in transitively by an
    unrelated dependency (openapi-core); an unrelated dependency bump could
    silently have dropped the check with no test failure. This pins the
    guarantee explicitly.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "recorded_at": {"type": "string", "format": "date-time"}
                },
                "required": ["recorded_at"],
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(
            response="done", structured_output={"recorded_at": "not-a-date-time"}
        ),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await _execute(node, workflow_dir, "assess the evidence")

    assert "recorded_at" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uri_format_constraint_is_enforced(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``format: uri`` is enforced now that ``format-nongpl`` is declared.

    Without the ``jsonschema[format-nongpl]`` extra this format silently
    no-ops (no checker package is installed for it), so a declared ``format:
    uri`` constraint would let anything through with no failure at all.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"source_url": {"type": "string", "format": "uri"}},
                "required": ["source_url"],
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output={"source_url": "not a uri"}),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await _execute(node, workflow_dir, "assess the evidence")

    assert "source_url" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("format_name", ["color", "emial"])
async def test_a_format_with_no_registered_checker_is_accepted_and_ignored(
    monkeypatch: pytest.MonkeyPatch,
    workflow_dir: Path,
    node: EdgeNode,
    format_name: str,
) -> None:
    """Pins the limit of format enforcement rather than overstating it.

    ``format-nongpl`` supplies checkers for the formats jsonschema knows about;
    a name it has no checker for is an annotation and nothing more.

    Enforcement is *per draft*, because ``load_gate_schema`` dispatches through
    ``validator_for(schema)``. ``color`` is a draft-03 format, and ``webcolors``
    does register a checker for it on ``Draft3Validator.FORMAT_CHECKER`` — so a
    gate declaring ``$schema: http://json-schema.org/draft-03/schema#`` really
    would enforce it. This gate declares 2020-12, whose checker set does not
    carry ``color``, so here it is annotation only. ``emial`` is the
    misspelling an author makes, and no draft has a checker for it.

    Both are accepted silently under this gate — the gate cannot enforce a
    constraint nobody implements *for the draft it declares*, and saying it can
    would be the overclaim this test exists to prevent.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"tint": {"type": "string", "format": format_name}},
                "required": ["tint"],
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output={"tint": "not a colour"}),
    )

    # Act
    gated = await _execute(node, workflow_dir, "assess the evidence")

    # Assert
    assert gated.value == {"tint": "not a colour"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_same_format_is_enforced_under_the_draft_that_registers_it(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``format`` enforcement follows the draft the gate declares, not the file.

    The sibling test above must not be read as "``color`` is never checked".
    ``load_gate_schema`` dispatches through ``validator_for(schema)``, and
    ``webcolors`` registers a ``color`` checker on
    ``Draft3Validator.FORMAT_CHECKER`` — so the *same* format name that is a
    silent annotation under 2020-12 is a hard constraint under draft-03. Pinned
    here so the pair states the rule instead of implying a false absolute.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "http://json-schema.org/draft-03/schema#",
                "type": "object",
                "properties": {"tint": {"type": "string", "format": "color"}},
            }
        ),
        encoding="utf-8",
    )
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output={"tint": "not a colour"}),
    )

    # Act / Assert
    with pytest.raises(GateValidationError) as excinfo:
        await _execute(node, workflow_dir, "assess the evidence")

    assert "tint" in str(excinfo.value)


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
        await _execute(node, workflow_dir, "assess the evidence")

    assert selector.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_base_dir_is_restored_after_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``agent_base_dir`` must not leak past a successful run.

    It is a ``ContextVar`` with no scope of its own; a long-lived caller (a
    future server/embedded run) would otherwise see this node's directory
    bleed into whatever runs after it.
    """
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(response="done", structured_output=dict(VALID_OUTPUT)),
    )
    token = agent_base_dir.set("caller-set-this")
    try:
        # Act
        await _execute(node, workflow_dir, "assess the evidence")

        # Assert — restored to the caller's prior value, not merely non-empty.
        assert agent_base_dir.get() == "caller-set-this"
    finally:
        agent_base_dir.reset(token)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_base_dir_is_restored_after_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, workflow_dir: Path, node: EdgeNode
) -> None:
    """``agent_base_dir`` must be restored even when the node raises."""
    # Arrange
    _install_backend(
        monkeypatch,
        ExecutionResult(response="", structured_output=None),
        invoke_error=ConnectionResetError("peer reset the connection"),
    )
    token = agent_base_dir.set("caller-set-this")
    try:
        # Act / Assert
        with pytest.raises(ExecutionError):
            await _execute(node, workflow_dir, "assess the evidence")

        assert agent_base_dir.get() == "caller-set-this"
    finally:
        agent_base_dir.reset(token)


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
        await _execute(open_node, tmp_path, "assess the evidence")

    assert "JSON object" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The dialect x shape matrix.
#
# `load_gate_schema` walks the gate for `$ref`s only under the dialects whose
# `referencing` subresource table is trusted (2020-12 and 2019-09, plus the
# gates with no `$schema` at all, which `jsonschema` itself resolves to
# 2020-12). Every other dialect is skipped at load and left to the runtime
# backstop in `_apply_gate`. That split is a correctness claim about two
# separate code paths, so it is pinned per dialect and per shape here rather
# than argued for in a docstring.
# ---------------------------------------------------------------------------

D2019 = "https://json-schema.org/draft/2019-09/schema"
D07 = "http://json-schema.org/draft-07/schema#"
D06 = "http://json-schema.org/draft-06/schema#"
D04 = "http://json-schema.org/draft-04/schema#"
D03 = "http://json-schema.org/draft-03/schema#"

#: A reference nothing can settle: the host does not exist and retrieval is
#: refused anyway, so it is unresolvable at load and at validate time alike.
BAD_REF = "https://remote.invalid/absent.json#/nope"


class _DialectProfile(NamedTuple):
    """How one dialect spells the constructs the matrix exercises.

    ``walked`` records whether :func:`edge.load_gate_schema` performs its
    load-time ``$ref`` walk for this dialect at all.
    """

    name: str
    schema_uri: str | None
    walked: bool
    id_kw: str
    defs_kw: str
    dependent_schema_kw: str
    booleans_are_schemas: bool
    has_const: bool
    has_extends: bool


_DIALECTS: tuple[_DialectProfile, ...] = (
    _DialectProfile(
        "2020-12", D2020, True, "$id", "$defs", "dependentSchemas", True, True, False
    ),
    _DialectProfile(
        "2019-09", D2019, True, "$id", "$defs", "dependentSchemas", True, True, False
    ),
    # No `$schema` at all: `jsonschema.validators.validator_for` falls back to
    # the latest draft, so this *is* 2020-12 at validate time and must be
    # walked. Skipping it would give up the load-time check for the commonest
    # gate an author writes.
    _DialectProfile(
        "no-schema-keyword",
        None,
        True,
        "$id",
        "$defs",
        "dependentSchemas",
        True,
        True,
        False,
    ),
    _DialectProfile(
        "draft-07", D07, False, "$id", "definitions", "dependencies", True, True, False
    ),
    _DialectProfile(
        "draft-06", D06, False, "$id", "definitions", "dependencies", True, True, False
    ),
    _DialectProfile(
        "draft-04", D04, False, "id", "definitions", "dependencies", False, False, False
    ),
    _DialectProfile(
        "draft-03", D03, False, "id", "definitions", "dependencies", False, False, True
    ),
)


class _MatrixCase(NamedTuple):
    """One cell: a gate, an output to present to it, and both verdicts."""

    case_id: str
    dialect_walked: bool
    schema: dict[str, Any]
    instance: dict[str, Any]
    load_refuses: bool
    runtime_refuses: bool


def _matrix_cases(dialect: _DialectProfile) -> list[_MatrixCase]:
    """Build every shape in the matrix for one dialect.

    ``carries_a_live_bad_ref`` says whether the gate holds an unresolvable
    ``$ref`` that the *validator* actually reaches for the paired instance.
    From it both verdicts follow mechanically, and that derivation is the rule
    under test: the runtime always refuses such a gate, and load refuses it
    exactly when the dialect is walked.

    Args:
        dialect: The dialect profile to instantiate the shapes for.

    Returns:
        The cells for this dialect, in shape order.
    """
    bad: dict[str, Any] = {"$ref": BAD_REF}
    base: dict[str, Any] = (
        {} if dialect.schema_uri is None else {"$schema": dialect.schema_uri}
    )
    cases: list[_MatrixCase] = []

    def add(
        name: str,
        body: dict[str, Any],
        instance: dict[str, Any],
        *,
        carries_a_live_bad_ref: bool,
    ) -> None:
        cases.append(
            _MatrixCase(
                case_id=f"{dialect.name}/{name}",
                dialect_walked=dialect.walked,
                schema={**base, **body},
                instance=instance,
                load_refuses=carries_a_live_bad_ref and dialect.walked,
                runtime_refuses=carries_a_live_bad_ref,
            )
        )

    # The dialect's own dependent-schema keyword, in schema form.
    add(
        "dependent-schema-form",
        {"type": "object", dialect.dependent_schema_kw: {"a": bad}},
        {"a": 1},
        carries_a_live_bad_ref=True,
    )
    # `dependencies` mixed, non-schema entry FIRST. This is the exact shape
    # that hides the later reference from `referencing`'s pre-2019 table. Under
    # 2020-12/2019-09 `dependencies` is deprecated *and inert* — neither
    # validator evaluates it — so the reference is never reached at either end.
    add(
        "dependencies-mixed-non-schema-first",
        {"type": "object", "dependencies": {"a": ["b"], "c": bad}},
        {"a": 1, "b": 1, "c": 1},
        carries_a_live_bad_ref=not dialect.walked,
    )
    # `dependencies` mixed the other way: schema first, array second.
    # `referencing` yields the *array* as if it were a subschema, and crawling
    # it calls `.get` on a list.
    add(
        "dependencies-mixed-schema-first",
        {"type": "object", "dependencies": {"a": bad, "b": ["c"]}},
        {"a": 1, "b": 1, "c": 1},
        carries_a_live_bad_ref=not dialect.walked,
    )
    if dialect.has_extends:
        # draft-03 `extends` takes a schema or an array of them. Asked for the
        # subresources of the object form, `referencing` yields its *keys*.
        add(
            "extends-as-object",
            {"type": "object", "extends": bad},
            {"a": 1},
            carries_a_live_bad_ref=True,
        )
        add(
            "extends-as-array",
            {"type": "object", "extends": [bad]},
            {"a": 1},
            carries_a_live_bad_ref=True,
        )
    # A nested `$id`/`id` rebases a relative reference that does resolve.
    add(
        "nested-id-rebases-a-relative-ref",
        {
            dialect.id_kw: f"https://ex.test/{dialect.name}/root.json",
            "type": "object",
            dialect.defs_kw: {
                "Helper": {
                    dialect.id_kw: f"https://ex.test/{dialect.name}/helper.json",
                    "type": "integer",
                }
            },
            "properties": {"a": {"$ref": "helper.json"}},
        },
        {"a": 1},
        carries_a_live_bad_ref=False,
    )
    # A `$ref`-shaped literal in a data position is data.
    add(
        "ref-shaped-literal-under-default",
        {"type": "object", "properties": {"a": {"default": bad}}},
        {"a": 1},
        carries_a_live_bad_ref=False,
    )
    add(
        "ref-shaped-literal-under-enum",
        {"type": "object", "properties": {"a": {"enum": [bad]}}},
        {"a": dict(bad)},
        carries_a_live_bad_ref=False,
    )
    if dialect.has_const:
        add(
            "ref-shaped-literal-under-const",
            {"type": "object", "properties": {"a": {"const": bad}}},
            {"a": dict(bad)},
            carries_a_live_bad_ref=False,
        )
    # A property *named* like a literal keyword is still an ordinary subschema.
    add(
        "property-named-const",
        {"type": "object", "properties": {"const": bad}},
        {"const": 1},
        carries_a_live_bad_ref=True,
    )
    add(
        "property-named-default",
        {"type": "object", "properties": {"default": bad}},
        {"default": 1},
        carries_a_live_bad_ref=True,
    )
    # A boolean in schema position carries neither a reference nor an id, and
    # asking a legacy dialect for its id raises. draft-04 and draft-03 have no
    # boolean schemas, so `additionalProperties: true` stands in for them.
    add(
        "boolean-subschema-beside-an-unresolvable-ref",
        (
            {"type": "object", "properties": {"ok": True, "a": bad}}
            if dialect.booleans_are_schemas
            else {
                "type": "object",
                "additionalProperties": True,
                "properties": {"a": bad},
            }
        ),
        {"ok": 1, "a": 1},
        carries_a_live_bad_ref=True,
    )
    add(
        "plain-unresolvable-ref",
        {"type": "object", "properties": {"a": bad}},
        {"a": 1},
        carries_a_live_bad_ref=True,
    )
    return cases


_MATRIX: list[_MatrixCase] = [
    case for dialect in _DIALECTS for case in _matrix_cases(dialect)
]


@pytest.mark.unit
@pytest.mark.parametrize("case", _MATRIX, ids=[case.case_id for case in _MATRIX])
def test_dialect_matrix_pins_where_an_unresolvable_ref_is_caught(
    workflow_dir: Path, node: EdgeNode, case: _MatrixCase
) -> None:
    """Where each dialect catches a bad ``$ref`` — and that one of them does.

    Two claims, one per column:

    * **Walked dialect** (2020-12, 2019-09, and a gate with no ``$schema``):
      load and validate time agree. An unresolvable reference is refused before
      an agent call, and — the direction that actually costs something — a gate
      the runtime can use is never refused at load.
    * **Skipped dialect** (draft-07/06/04/03): load performs no reference check
      at all, so it accepts; the backstop in ``_apply_gate`` is then the *only*
      protection and has to raise ``GateSchemaError`` rather than let anything
      escape. ``referencing``'s pre-2019 subresource table mishandles both
      ``dependencies`` and draft-03 ``extends``, which is why the walk does not
      run for these dialects — but the shapes are exercised here anyway,
      because the backstop is what now stands behind them.

    The one cell where the two disagree by design is covered separately by
    ``test_an_unresolvable_ref_in_an_unreferenced_defs_is_refused``: nothing in
    this matrix hides a bad reference behind an instance that never reaches it.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(case.schema), encoding="utf-8"
    )

    # Act
    try:
        edge.load_gate_schema(node, workflow_dir)
        load_refuses = False
    except GateSchemaError:
        load_refuses = True
    runtime_refuses = not _runtime_gate_is_usable(case.schema, case.instance)

    # Assert
    assert load_refuses is case.load_refuses
    assert runtime_refuses is case.runtime_refuses
    # The governing rule: load may only refuse what the runtime also refuses.
    assert not (load_refuses and not runtime_refuses)
    if case.dialect_walked:
        assert load_refuses is runtime_refuses
    else:
        assert load_refuses is False


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_an_unknown_dialect_is_still_walked(workflow_dir: Path, node: EdgeNode) -> None:
    """A ``$schema`` nothing recognises is 2020-12, so the walk still runs.

    ``validator_for`` falls back to the latest draft for an unrecognised
    ``$schema``, and ``_apply_gate`` derives its validator the same way. The
    walk's dialect is therefore genuinely the validator's, and skipping the
    walk here would hand an agent call to a typo'd ``$schema`` URI.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.invalid/draft/2027-01/schema",
                "type": "object",
                "properties": {"a": {"$ref": BAD_REF}},
            }
        ),
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(GateSchemaError) as excinfo:
        edge.load_gate_schema(node, workflow_dir)

    assert BAD_REF in str(excinfo.value)


# ---------------------------------------------------------------------------
# Embedded resources that switch dialect: the walk's remaining blind spot, and
# the one place a *walked* gate can still crash `referencing`'s crawl.
# ---------------------------------------------------------------------------


def _embedding(inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``inner`` as an embedded resource inside a 2020-12 gate.

    Args:
        inner: The embedded subschema; it supplies its own ``$id``/``$schema``.

    Returns:
        A 2020-12 gate whose ``properties.b`` is that embedded resource.
    """
    return {
        "$schema": D2020,
        "$id": "https://ex.test/root.json",
        "type": "object",
        "properties": {"a": {"$ref": BAD_REF}, "b": inner},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "inner",
    [
        {
            "$id": "https://ex.test/e1.json",
            "$schema": D03,
            "extends": {"type": "object"},
        },
        {
            "$id": "https://ex.test/e2.json",
            "$schema": D07,
            "dependencies": {"p": {"type": "object"}, "q": ["r"]},
        },
    ],
    ids=["draft-03-extends-as-object", "draft-07-dependencies-schema-then-array"],
)
def test_an_embedded_legacy_resource_cannot_break_load_with_a_bare_error(
    workflow_dir: Path, node: EdgeNode, inner: dict[str, Any]
) -> None:
    """A crawl that trips over an embedded legacy resource is still a GateSchemaError.

    Narrowing the walk to 2020-12/2019-09 does *not* put these out of reach.
    Resolving any reference makes ``referencing`` crawl the whole document, and
    the crawl re-dialects an embedded resource from its own ``$schema`` — so a
    2020-12 gate that embeds one of the two shapes ``referencing`` mishandles
    reaches ``_legacy_anchor_in_id`` with a ``str``/``list`` and raises
    ``AttributeError``. That is a workflow-authoring defect and has to leave
    ``load_gate_schema`` through the channel its docstring declares, not as a
    traceback out of ``prepare_workflow``.
    """
    # Arrange
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(_embedding(inner)), encoding="utf-8"
    )

    # Act / Assert
    with pytest.raises(GateSchemaError):
        edge.load_gate_schema(node, workflow_dir)


@pytest.mark.unit
def test_a_ref_inside_an_embedded_legacy_resource_is_not_walked(
    workflow_dir: Path, node: EdgeNode
) -> None:
    """The disclosed gap: the walk never re-dialects an embedded resource.

    ``_collect_refs`` descends with the *root* dialect's subresource table
    throughout, so a reference that only an embedded dialect's table would
    reach — here draft-07 ``dependencies`` — is invisible to it even though the
    validator honours the embedded ``$schema`` and does reach it. Load accepts;
    the backstop refuses after one agent call. This is the safe direction of
    the trade, and it is pinned so that changing it is a decision.
    """
    # Arrange
    schema = {
        "$schema": D2020,
        "$id": "https://ex.test/outer.json",
        "type": "object",
        "properties": {
            "b": {
                "$id": "https://ex.test/inner.json",
                "$schema": D07,
                "type": "object",
                "dependencies": {"p": ["q"], "r": {"$ref": BAD_REF}},
            }
        },
    }
    (workflow_dir / "gates" / "evidence.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    # Act
    loaded = edge.load_gate_schema(node, workflow_dir)
    usable = _runtime_gate_is_usable(schema, {"b": {"p": 1, "q": 1, "r": 1}})

    # Assert
    assert loaded == schema
    assert usable is False


@pytest.mark.unit
class TestGatePathConfinement:
    """A gate schema path must resolve inside the workflow directory.

    The module refuses remote ``$ref`` retrieval because a workflow file is
    attacker-influenceable; a ``gate.schema`` of ``/etc/passwd`` or ``../../x``
    is the same hole through the front door and gets the same refusal.
    """

    @pytest.mark.parametrize(
        "schema_path",
        [
            "../outside.schema.json",
            "gates/../../outside.schema.json",
        ],
    )
    def test_traversal_is_rejected(self, workflow_dir: Path, schema_path: str) -> None:
        # Arrange — the file exists, so only the confinement check can reject.
        (workflow_dir.parent / "outside.schema.json").write_text("{}", encoding="utf-8")
        node = EdgeNode(
            id="evidence",
            edge={"agent": "agents/evidence.yaml"},  # type: ignore[arg-type]
            gate={"schema": schema_path},  # type: ignore[arg-type]
        )

        # Act / Assert
        with pytest.raises(GateSchemaError) as exc:
            edge.load_gate_schema(node, workflow_dir)
        assert "escapes the workflow directory" in str(exc.value)

    def test_absolute_path_is_rejected(self, workflow_dir: Path) -> None:
        # Arrange — an absolute path replaces workflow_dir entirely under `/`.
        outside = workflow_dir.parent / "outside.schema.json"
        outside.write_text("{}", encoding="utf-8")
        node = EdgeNode(
            id="evidence",
            edge={"agent": "agents/evidence.yaml"},  # type: ignore[arg-type]
            gate={"schema": str(outside)},  # type: ignore[arg-type]
        )

        # Act / Assert
        with pytest.raises(GateSchemaError) as exc:
            edge.load_gate_schema(node, workflow_dir)
        assert "escapes the workflow directory" in str(exc.value)

    def test_nested_relative_path_inside_stays_legal(
        self, workflow_dir: Path, node: EdgeNode
    ) -> None:
        # The confinement check must not break ordinary nested layouts.
        assert edge.load_gate_schema(node, workflow_dir)


@pytest.mark.unit
class TestAgentPathConfinement:
    """``edge.agent`` gets the same confinement control as ``gate.schema``."""

    @pytest.mark.parametrize(
        "agent_path",
        [
            "../outside/agent.yaml",
            "agents/../../outside/agent.yaml",
            "/etc/agent.yaml",
        ],
    )
    def test_escaping_agent_path_is_rejected(
        self, workflow_dir: Path, agent_path: str
    ) -> None:
        node = EdgeNode(
            id="evidence",
            edge={"agent": agent_path},  # type: ignore[arg-type]
            gate={"schema": "gates/evidence.schema.json"},  # type: ignore[arg-type]
        )

        with pytest.raises(ConfigError) as exc:
            edge.resolve_agent_path(node, workflow_dir)
        assert "escapes the workflow directory" in str(exc.value)

    def test_inside_path_resolves(self, workflow_dir: Path, node: EdgeNode) -> None:
        resolved = edge.resolve_agent_path(node, workflow_dir)

        assert resolved == (workflow_dir / "agents" / "evidence.yaml").resolve()


@pytest.mark.unit
class TestGateSchemaLoadGuards:
    """Load-time guards over the gate document itself."""

    def test_null_permitting_gate_is_rejected(
        self, workflow_dir: Path, node: EdgeNode
    ) -> None:
        # Arrange — a gate promising null can never accept it: _apply_gate
        # reads structured_output None as "free text, nothing to validate".
        (workflow_dir / "gates" / "evidence.schema.json").write_text(
            json.dumps({"type": ["object", "null"]}), encoding="utf-8"
        )

        # Act / Assert
        with pytest.raises(GateSchemaError) as exc:
            edge.load_gate_schema(node, workflow_dir)
        assert "null" in str(exc.value)

    def test_overdeep_gate_is_rejected_as_schema_error(
        self, workflow_dir: Path, node: EdgeNode
    ) -> None:
        # Arrange — 150 nested levels: past the 100-level bound, far short of
        # the interpreter's recursion limit.
        schema: dict = {"type": "object"}
        for _ in range(150):
            schema = {"type": "object", "properties": {"a": schema}}
        (workflow_dir / "gates" / "evidence.schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

        # Act / Assert — GateSchemaError, never a bare RecursionError.
        with pytest.raises(GateSchemaError) as exc:
            edge.load_gate_schema(node, workflow_dir)
        assert "deeper than 100 levels" in str(exc.value)

    def test_deep_but_legal_gate_still_loads(
        self, workflow_dir: Path, node: EdgeNode
    ) -> None:
        # Arrange — 48 schema-nesting levels. Each level costs two container
        # levels ({"properties": {"a": ...}}), so this sits just under the
        # 100-container bound. The property worth pinning: the bound exists to
        # keep check_schema and the $ref walk inside the interpreter's
        # recursion limit, so a legal deep gate must load cleanly — neither
        # rejected by the guard nor felled by a RecursionError.
        schema: dict = {"type": "object", "$ref": "#/$defs/leaf"}
        for _ in range(48):
            schema = {"type": "object", "properties": {"a": schema}}
        schema["$defs"] = {"leaf": {"type": "object"}}
        (workflow_dir / "gates" / "evidence.schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

        # Act
        loaded = edge.load_gate_schema(node, workflow_dir)

        # Assert
        assert loaded == schema

    def test_oversized_gate_is_rejected(
        self, workflow_dir: Path, node: EdgeNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — shrink the cap instead of writing a real 5 MB file: the
        # assurance wanted is that the stat() guard is wired ahead of the
        # read, not that 5 MB of x's is big.
        monkeypatch.setattr(edge, "_MAX_GATE_BYTES", 16)
        (workflow_dir / "gates" / "evidence.schema.json").write_text(
            '{"type": "object", "description": "over the tiny cap"}',
            encoding="utf-8",
        )

        # Act / Assert — the message derives from the (patched) constant, so
        # the guard and its report cannot drift apart.
        with pytest.raises(GateSchemaError) as exc:
            edge.load_gate_schema(node, workflow_dir)
        assert "exceeds 16 bytes" in str(exc.value)


@pytest.mark.unit
class TestGatedOutputRoundTrip:
    """A GatedOutput must survive dump -> validate (Temporal data-converter shape)."""

    def test_round_trips(self) -> None:
        gated = edge.GatedOutput(
            node_id="evidence",
            value={"net_income": 1200.0},
            gate_schema={"type": "object", "properties": {"net_income": {}}},
        )

        assert edge.GatedOutput.model_validate(gated.model_dump()) == gated


@pytest.mark.unit
class TestFormatCheckerRegistration:
    """The format checkers the gate relies on must actually be registered.

    _apply_gate passes format_checker=validator_cls.FORMAT_CHECKER; without
    the jsonschema[format-nongpl] extra those checkers are silently absent and
    a gate declaring `format: email` accepts everything. This fails loudly on
    that dependency drift instead.
    """

    @pytest.mark.parametrize(
        "format_name", ["date", "date-time", "email", "uri", "uuid"]
    )
    def test_checker_is_registered(self, format_name: str) -> None:
        import jsonschema

        checkers = jsonschema.Draft202012Validator.FORMAT_CHECKER.checkers

        assert format_name in checkers, (
            f"format {format_name!r} has no registered checker — is the "
            "jsonschema[format-nongpl] extra installed?"
        )
