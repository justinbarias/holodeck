"""Activity factory for an edge node (spec 040, T3).

Every test runs against a fake backend injected at the ``BackendSelector``
boundary, with an autouse guard that fails the run if a concrete backend is
ever constructed — "zero live LLM calls" is enforced, not assumed.

Covers: the returned callable is a real Temporal activity definition named
after the node (decision 11), the gate-validated object is what lands in the
envelope while the raw response text does not (FR-008), the factory settles
authoring faults at registration rather than per call, and the failure
channels stay separate ahead of T4's classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from temporalio import activity

from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.backends.claude_backend import ClaudeBackend
from holodeck.lib.backends.openai_agents_backend import OpenAIAgentsBackend
from holodeck.lib.errors import (
    ConfigError,
    ExecutionError,
    GateSchemaError,
    GateValidationError,
)
from holodeck.models.token_usage import TokenUsage
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.models import AgentActivityInput, AgentActivityResult

pytestmark = pytest.mark.unit

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

RAW_TEXT = "Sure! The applicant's net income is about $4,200."


class _FakeBackend:
    """Stands in for an ``AgentBackend`` — records calls, never talks to a model."""

    def __init__(
        self, result: ExecutionResult, invoke_error: Exception | None = None
    ) -> None:
        self.result = result
        self.invoke_error = invoke_error
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
def base_dir(tmp_path: Path) -> Path:
    """A worker base directory holding an edge agent.yaml and a gate schema."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "evidence.yaml").write_text(AGENT_YAML, encoding="utf-8")
    (tmp_path / "gates").mkdir()
    (tmp_path / "gates" / "evidence.schema.json").write_text(
        json.dumps(GATE_SCHEMA), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def node() -> EdgeNode:
    """The edge node under test, with paths relative to the base directory."""
    return EdgeNode(
        id="evidence",
        edge={"agent": "agents/evidence.yaml"},  # type: ignore[arg-type]
        gate={"schema": "gates/evidence.schema.json"},  # type: ignore[arg-type]
    )


def _result(
    structured_output: Any = None,
    response: str = RAW_TEXT,
    is_error: bool = False,
    error_reason: str | None = None,
) -> ExecutionResult:
    """Build an ``ExecutionResult`` the fake backend will return."""
    return ExecutionResult(
        response=response,
        structured_output=structured_output,
        token_usage=TokenUsage(
            prompt_tokens=120, completion_tokens=40, total_tokens=160
        ),
        num_turns=2,
        is_error=is_error,
        error_reason=error_reason,
    )


def _install_backend(
    monkeypatch: pytest.MonkeyPatch,
    result: ExecutionResult,
    invoke_error: Exception | None = None,
) -> _RecordingSelector:
    """Inject a fake backend at the ``BackendSelector`` boundary."""
    selector = _RecordingSelector(_FakeBackend(result, invoke_error=invoke_error))
    # The activity imports BackendSelector lazily, so the patch lands on the
    # selector module rather than on a name already bound in activity.py.
    monkeypatch.setattr("holodeck.lib.backends.selector.BackendSelector", selector)
    return selector


class TestActivityDefinition:
    """Temporal introspection of the returned callable."""

    def test_definition_name_is_the_node_id(self, base_dir: Path, node: EdgeNode):
        """Activity name is the node id — replay-load-bearing (decision 11)."""
        # Act
        fn = agent_activity(node, base_dir)
        definition = activity._Definition.must_from_callable(fn)

        # Assert
        assert definition is not None
        assert definition.name == "evidence"

    def test_definition_is_async_with_typed_payloads(
        self, base_dir: Path, node: EdgeNode
    ):
        """The activity takes the input model and returns the result envelope."""
        # Act
        definition = activity._Definition.must_from_callable(
            agent_activity(node, base_dir)
        )

        # Assert
        assert definition.is_async is True
        assert definition.arg_types == [AgentActivityInput]
        assert definition.ret_type is AgentActivityResult

    def test_factory_takes_no_timeout_or_retry_kwargs(self):
        """Scheduling options are caller-side (decision 10)."""
        # Arrange
        import inspect

        # Act
        parameters = inspect.signature(agent_activity).parameters

        # Assert
        assert list(parameters) == ["node", "base_dir"]


class TestFactoryTimeValidation:
    """Authoring faults are settled at registration, before any model call."""

    def test_agent_path_escaping_the_base_dir_is_refused(
        self, base_dir: Path, node: EdgeNode
    ):
        """Path confinement stays with ``resolve_agent_path`` (decision 1)."""
        # Arrange
        escaping = node.model_copy(
            update={"edge": node.edge.model_copy(update={"agent": "../outside.yaml"})}
        )

        # Act / Assert
        with pytest.raises(ConfigError):
            agent_activity(escaping, base_dir)

    def test_unreadable_gate_schema_is_refused(self, base_dir: Path, node: EdgeNode):
        """An unusable gate fails at registration, not per execution."""
        # Arrange
        (base_dir / "gates" / "evidence.schema.json").write_text(
            "{not json", encoding="utf-8"
        )

        # Act / Assert
        with pytest.raises(GateSchemaError):
            agent_activity(node, base_dir)

    def test_agent_without_response_format_is_refused(
        self, base_dir: Path, node: EdgeNode
    ):
        """An agent that can never produce structured output is an authoring fault."""
        # Arrange
        (base_dir / "agents" / "evidence.yaml").write_text(
            AGENT_YAML_NO_RESPONSE_FORMAT, encoding="utf-8"
        )

        # Act / Assert
        with pytest.raises(ConfigError) as excinfo:
            agent_activity(node, base_dir)
        assert "response_format" in str(excinfo.value)

    def test_agent_and_gate_are_read_once_at_factory_time(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """Worker-side binding: deleting the files does not break execution."""
        # Arrange
        fn = agent_activity(node, base_dir)
        _install_backend(monkeypatch, _result(structured_output=VALID_OUTPUT))
        (base_dir / "agents" / "evidence.yaml").unlink()
        (base_dir / "gates" / "evidence.schema.json").unlink()

        # Act
        import asyncio

        result = asyncio.run(fn(AgentActivityInput(message="Extract the evidence.")))

        # Assert
        assert result.output == VALID_OUTPUT


class TestGatedExecution:
    """One activity call is one gated ``invoke_once`` (decision 5)."""

    @pytest.mark.asyncio
    async def test_validated_object_lands_in_the_envelope(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """The gate-validated dict is the canonical value (FR-008)."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=VALID_OUTPUT))
        fn = agent_activity(node, base_dir)

        # Act
        result = await fn(AgentActivityInput(message="Extract the evidence."))

        # Assert
        assert isinstance(result, AgentActivityResult)
        assert result.output == VALID_OUTPUT
        assert type(result.output) is dict
        assert result.agent_id == "evidence"
        assert result.num_turns == 2
        assert result.token_usage.total_tokens == 160

    @pytest.mark.asyncio
    async def test_raw_response_text_never_crosses(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """No field of the envelope carries the model's prose (FR-008)."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=VALID_OUTPUT))
        fn = agent_activity(node, base_dir)

        # Act
        result = await fn(AgentActivityInput(message="Extract the evidence."))

        # Assert
        assert RAW_TEXT not in result.model_dump_json()

    @pytest.mark.asyncio
    async def test_one_call_is_one_invoke_once_and_a_teardown(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """Stateless: the backend is built, used, and torn down per call."""
        # Arrange
        selector = _install_backend(
            monkeypatch, _result(structured_output=VALID_OUTPUT)
        )
        fn = agent_activity(node, base_dir)

        # Act
        await fn(AgentActivityInput(message="one"))
        await fn(AgentActivityInput(message="two"))

        # Assert
        assert len(selector.calls) == 2
        assert selector.backend.messages == ["one", "two"]
        assert selector.backend.torn_down is True

    @pytest.mark.asyncio
    async def test_context_is_appended_to_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """A context object reaches the agent as a deterministic JSON block."""
        # Arrange
        selector = _install_backend(
            monkeypatch, _result(structured_output=VALID_OUTPUT)
        )
        fn = agent_activity(node, base_dir)

        # Act
        await fn(
            AgentActivityInput(message="Extract.", context={"b": 2, "a": "applicant"})
        )

        # Assert
        sent = selector.backend.messages[0]
        assert sent.startswith("Extract.")
        assert '{"a": "applicant", "b": 2}' in sent

    @pytest.mark.asyncio
    async def test_no_context_leaves_the_message_untouched(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """The None sentinel adds nothing to the prompt."""
        # Arrange
        selector = _install_backend(
            monkeypatch, _result(structured_output=VALID_OUTPUT)
        )
        fn = agent_activity(node, base_dir)

        # Act
        await fn(AgentActivityInput(message="Extract."))

        # Assert
        assert selector.backend.messages == ["Extract."]


class TestFailureChannels:
    """Channels stay distinct ahead of T4's retry classification."""

    @pytest.mark.asyncio
    async def test_free_text_is_a_gate_rejection(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """No structured output means nothing was presented to the gate."""
        # Arrange
        _install_backend(monkeypatch, _result(structured_output=None))
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(GateValidationError):
            await fn(AgentActivityInput(message="Extract."))

    @pytest.mark.asyncio
    async def test_schema_invalid_output_is_a_gate_rejection(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """Output the gate rejects is evidence about the model, not a crash."""
        # Arrange
        _install_backend(
            monkeypatch,
            _result(structured_output={"net_income": "a lot", "residency_status": "x"}),
        )
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(GateValidationError) as excinfo:
            await fn(AgentActivityInput(message="Extract."))
        assert "evidence" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_raising_invocation_is_an_execution_error(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """A broken invocation produced nothing to judge."""
        # Arrange
        selector = _install_backend(
            monkeypatch, _result(), invoke_error=RuntimeError("transport down")
        )
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(ExecutionError) as excinfo:
            await fn(AgentActivityInput(message="Extract."))
        assert "transport down" in str(excinfo.value)
        assert selector.backend.torn_down is True

    @pytest.mark.asyncio
    async def test_error_result_with_output_still_goes_to_the_gate(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """``is_error`` with an object is evidence about the model (SC-003)."""
        # Arrange
        _install_backend(
            monkeypatch,
            _result(
                structured_output=VALID_OUTPUT,
                is_error=True,
                error_reason="response_format violated",
            ),
        )
        fn = agent_activity(node, base_dir)

        # Act
        result = await fn(AgentActivityInput(message="Extract."))

        # Assert
        assert result.output == VALID_OUTPUT

    @pytest.mark.asyncio
    async def test_error_result_without_output_is_an_execution_error(
        self, monkeypatch: pytest.MonkeyPatch, base_dir: Path, node: EdgeNode
    ):
        """A failure with nothing to judge never reaches the gate."""
        # Arrange
        _install_backend(
            monkeypatch,
            _result(
                structured_output=None, is_error=True, error_reason="429 throttled"
            ),
        )
        fn = agent_activity(node, base_dir)

        # Act / Assert
        with pytest.raises(ExecutionError) as excinfo:
            await fn(AgentActivityInput(message="Extract."))
        assert "429 throttled" in str(excinfo.value)
