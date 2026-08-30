"""OpenTelemetry acceptance coverage for Temporal agent activities (AC-6)."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.types import AttributeValue
from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends import selector as selector_module
from holodeck.lib.backends.base import ExecutionResult
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner import executor as executor_module
from holodeck.lib.test_runner.executor import TestExecutor as HoloDeckTestExecutor
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_case import TestCaseModel as HoloDeckTestCase
from holodeck.models.token_usage import TokenUsage
from holodeck.temporal.activity import agent_activity
from holodeck.temporal.models import AgentActivityInput, AgentActivityResult
from holodeck.temporal.worker_config import WorkerConfig, load_worker_config

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hardship"
EVIDENCE_AGENT_YAML = FIXTURE_DIR / "agents" / "evidence.yaml"
WORKER_YAML = FIXTURE_DIR / "worker.yaml"

EVIDENCE_AGENT = "hardship-evidence-extractor"
EVIDENCE_MODEL = "claude-sonnet-4-6"
EVIDENCE_ACTIVITY = "evidence"
STATEMENT = (
    "I take home $5,000 a month and my outgoings are $3,500. My residency was "
    "verified by the case officer in March."
)
EVIDENCE_OUTPUT: dict[str, object] = {
    "income": {"net": 5000, "expenses": 3500},
    "residency": {"status": "verified"},
}
TOKEN_USAGE = TokenUsage(
    prompt_tokens=37,
    completion_tokens=11,
    total_tokens=48,
)

GENAI_SPAN_NAME = f"invoke_agent {EVIDENCE_AGENT}"
GENAI_ATTRIBUTES: dict[str, AttributeValue] = {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.system": "anthropic",
    "gen_ai.agent.name": EVIDENCE_AGENT,
    "gen_ai.request.model": EVIDENCE_MODEL,
    "gen_ai.usage.input_tokens": TOKEN_USAGE.prompt_tokens,
    "gen_ai.usage.output_tokens": TOKEN_USAGE.completion_tokens,
}


class _ScriptedBackend:
    """Return fixed evidence while emitting deterministic GenAI telemetry."""

    def __init__(self, selector: _ScriptedSelector) -> None:
        self._selector = selector

    async def invoke_once(
        self,
        message: str,
        context: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Record one invocation and emit the scripted agent span.

        Args:
            message: The prompt supplied by the execution path.
            context: Unused prior-turn context, present for the backend protocol.

        Returns:
            A fixed result carrying gated evidence and token usage.
        """
        del context
        self._selector.messages.append(message)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            GENAI_SPAN_NAME,
            attributes=GENAI_ATTRIBUTES,
        ):
            return ExecutionResult(
                response=json.dumps(EVIDENCE_OUTPUT, sort_keys=True),
                structured_output=EVIDENCE_OUTPUT,
                token_usage=TOKEN_USAGE,
            )

    async def teardown(self) -> None:
        """Record that the activity released the scripted backend."""
        self._selector.teardowns += 1


class _ScriptedSelector:
    """Credential-free ``BackendSelector`` replacement for the evidence agent."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.teardowns = 0

    async def select(
        self,
        agent: Agent,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> _ScriptedBackend:
        """Return a fresh backend for the configured evidence agent.

        Args:
            agent: The loaded evidence-agent configuration.
            tool_instances: Unused initialized tools, present for the selector seam.
            mode: Execution mode, which must remain the activity's test mode.

        Returns:
            A backend that emits the fixed evidence result and GenAI span.
        """
        del tool_instances
        assert agent.name == EVIDENCE_AGENT
        assert mode == "test"
        return _ScriptedBackend(self)


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Load the committed hardship worker configuration."""
    return load_worker_config(WORKER_YAML)


@pytest.fixture
def in_memory_tracing() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """Install a fresh active provider with an in-memory span processor."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # OTel providers are process-global and set-once through the public setter.
    # Existing observability integration tests isolate them by restoring this
    # private slot; use the same idiom so xdist workers do not retain this test's
    # provider after fixture teardown.
    original = trace._TRACER_PROVIDER
    trace._TRACER_PROVIDER = provider
    try:
        yield provider, exporter
    finally:
        trace._TRACER_PROVIDER = original
        provider.shutdown()


def _install_scripted_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> _ScriptedSelector:
    """Patch both deferred and imported selector module bindings.

    Args:
        monkeypatch: The pytest patcher.

    Returns:
        The shared selector, for invocation and teardown assertions.
    """
    selector = _ScriptedSelector()
    monkeypatch.setattr(selector_module, "BackendSelector", selector)
    monkeypatch.setattr(executor_module, "BackendSelector", selector)
    return selector


def _evidence_activity(
    worker_config: WorkerConfig,
) -> Callable[[AgentActivityInput], Awaitable[AgentActivityResult]]:
    """Build the committed evidence node's decorated activity callable."""
    node = next(node for node in worker_config.nodes if node.id == EVIDENCE_ACTIVITY)
    return cast(
        "Callable[[AgentActivityInput], Awaitable[AgentActivityResult]]",
        agent_activity(node, worker_config.base_dir),
    )


def _genai_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """Return finished spans carrying at least one GenAI semantic attribute."""
    return [
        span
        for span in exporter.get_finished_spans()
        if any(key.startswith("gen_ai.") for key in (span.attributes or {}))
    ]


def _genai_attributes(span: ReadableSpan) -> dict[str, AttributeValue]:
    """Extract only GenAI semantic-convention attributes from one span."""
    return {
        key: value
        for key, value in (span.attributes or {}).items()
        if key.startswith("gen_ai.")
    }


def _assert_genai_parity(
    test_spans: list[ReadableSpan], activity_spans: list[ReadableSpan]
) -> None:
    """Assert GenAI span-name multisets and per-name attributes are identical."""
    assert Counter(span.name for span in test_spans) == Counter(
        span.name for span in activity_spans
    )

    test_attributes = {
        name: [_genai_attributes(span) for span in test_spans if span.name == name]
        for name in {span.name for span in test_spans}
    }
    activity_attributes = {
        name: [_genai_attributes(span) for span in activity_spans if span.name == name]
        for name in {span.name for span in activity_spans}
    }
    assert test_attributes == activity_attributes


@pytest.mark.asyncio
async def test_span_parity_between_holodeck_test_and_activity(
    tmp_path: Path,
    worker_config: WorkerConfig,
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracing: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """AC-6: test-runner and activity paths emit identical GenAI spans."""
    # Arrange
    provider, exporter = in_memory_tracing
    selector = _install_scripted_selector(monkeypatch)
    agent = ConfigLoader().load_agent_yaml(str(EVIDENCE_AGENT_YAML))
    agent = agent.model_copy(
        update={
            "test_cases": [
                HoloDeckTestCase(name="otel-span-parity", input=STATEMENT),
            ]
        }
    )
    executor = HoloDeckTestExecutor(
        str(EVIDENCE_AGENT_YAML),
        agent_config=agent,
        resolved_execution_config=ExecutionConfig(parallel_test_cases=1),
        file_processor=FileProcessor(cache_dir=str(tmp_path / "cache")),
    )

    # Act — run the same evidence agent once through each public execution path.
    report = await executor.execute_tests()
    provider.force_flush()
    test_spans = _genai_spans(exporter)

    exporter.clear()
    activity = _evidence_activity(worker_config)
    activity_result = await ActivityEnvironment().run(
        activity,
        AgentActivityInput(message=STATEMENT),
    )
    provider.force_flush()
    activity_spans = _genai_spans(exporter)

    # Assert
    assert report.summary.total_tests == 1
    assert report.summary.passed == 1
    assert report.results[0].token_usage == TOKEN_USAGE
    assert activity_result.output == EVIDENCE_OUTPUT
    assert activity_result.token_usage == TOKEN_USAGE
    assert selector.messages == [STATEMENT, STATEMENT]
    assert selector.teardowns == 1

    assert [span.name for span in test_spans] == [GENAI_SPAN_NAME]
    assert [_genai_attributes(span) for span in test_spans] == [GENAI_ATTRIBUTES]
    _assert_genai_parity(test_spans, activity_spans)


@pytest.mark.asyncio
async def test_genai_spans_nest_under_temporal_activity_span(
    worker_config: WorkerConfig,
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracing: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    """GenAI spans descend from ``RunActivity`` with tracing propagation active."""
    # Arrange
    provider, exporter = in_memory_tracing
    selector = _install_scripted_selector(monkeypatch)
    activity = _evidence_activity(worker_config)
    task_queue = f"otel-{uuid.uuid4()}"
    activity_id = f"otel-{uuid.uuid4()}"

    # Act — a real client/worker pair is required for interceptor header
    # propagation; ActivityEnvironment intentionally has no interceptor chain.
    environment = await WorkflowEnvironment.start_local(
        data_converter=pydantic_data_converter
    )
    try:
        interceptor = TracingInterceptor(
            tracer=provider.get_tracer("holodeck.temporal.test")
        )
        client = await Client.connect(
            environment.client.service_client.config.target_host,
            namespace=environment.client.namespace,
            data_converter=pydantic_data_converter,
            interceptors=[interceptor],
        )
        async with Worker(
            client,
            task_queue=task_queue,
            activities=[activity],
        ):
            result: AgentActivityResult = await client.execute_activity(
                activity,
                AgentActivityInput(message=STATEMENT),
                id=activity_id,
                task_queue=task_queue,
                start_to_close_timeout=timedelta(seconds=30),
            )
    finally:
        await environment.shutdown()
    provider.force_flush()

    # Assert
    assert result.output == EVIDENCE_OUTPUT
    assert selector.messages == [STATEMENT]
    spans = list(exporter.get_finished_spans())
    genai_spans = _genai_spans(exporter)
    activity_spans = [
        span for span in spans if span.name == f"RunActivity:{EVIDENCE_ACTIVITY}"
    ]
    assert len(activity_spans) == 1
    assert genai_spans

    activity_span = activity_spans[0]
    spans_by_id = {span.context.span_id: span for span in spans}
    for genai_span in genai_spans:
        ancestor_ids: set[int] = set()
        parent = genai_span.parent
        while parent is not None:
            ancestor_ids.add(parent.span_id)
            parent_span = spans_by_id.get(parent.span_id)
            parent = parent_span.parent if parent_span is not None else None

        assert activity_span.context.span_id in ancestor_ids
        assert genai_span.context.trace_id == activity_span.context.trace_id
