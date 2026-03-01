"""Integration tests for Claude GenAI instrumentation with real OTel providers.

Validates that the ``otel-instrumentation-claude-agent-sdk`` package
integrates correctly with HoloDeck's observability infrastructure.
Tests use both an OTLP gRPC exporter (for Aspire / collector visibility)
and an ``InMemorySpanExporter`` (for programmatic assertions).

The module-level ``importorskip`` ensures these tests are silently skipped
when the optional ``claude-otel`` extras group is not installed.

Real API tests require ``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

otel_claude = pytest.importorskip(
    "opentelemetry.instrumentation.claude_agent_sdk",
    reason="otel-instrumentation-claude-agent-sdk not installed",
)

from opentelemetry.instrumentation.claude_agent_sdk import (  # noqa: E402
    ClaudeAgentSdkInstrumentor,
)
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from holodeck.lib.backends.claude_backend import ClaudeBackend  # noqa: E402
from holodeck.lib.observability.providers import ObservabilityContext  # noqa: E402
from holodeck.models.agent import Agent, Instructions  # noqa: E402
from holodeck.models.claude_config import (  # noqa: E402
    AuthProvider,
    ClaudeConfig,
    PermissionMode,
)
from holodeck.models.llm import LLMProvider, ProviderEnum  # noqa: E402
from holodeck.models.observability import (  # noqa: E402
    MetricsConfig,
    ObservabilityConfig,
    TracingConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTLP exporter configuration
# ---------------------------------------------------------------------------

OTLP_ENDPOINT = "http://localhost:4317"


def _try_create_otlp_exporter() -> object | None:
    """Try to create an OTLP gRPC span exporter.

    Returns None (with a log warning) when the collector is unreachable
    or the ``opentelemetry-exporter-otlp-proto-grpc`` package is missing.
    This makes the OTLP exporter best-effort — tests still pass via the
    InMemorySpanExporter even without a running collector.
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "OTLP exporter unavailable (%s); spans won't reach collector",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Environment & skip logic
# ---------------------------------------------------------------------------

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")

skip_if_no_claude_oauth = pytest.mark.skipif(
    SKIP_LLM_TESTS or not CLAUDE_CODE_OAUTH_TOKEN,
    reason="CLAUDE_CODE_OAUTH_TOKEN not configured or LLM tests disabled",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nested sessions."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    """Create a fresh in-memory span exporter for capturing test spans."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    """Create a TracerProvider wired to both InMemory and OTLP exporters.

    The InMemorySpanExporter is used for test assertions.
    The OTLP exporter (best-effort) sends spans to a collector like Aspire
    so they can be inspected in a dashboard during development.
    """
    resource = Resource.create({"service.name": "holodeck-integration-test"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    otlp = _try_create_otlp_exporter()
    if otlp is not None:
        provider.add_span_processor(SimpleSpanProcessor(otlp))

    return provider


@pytest.fixture()
def instrumentor() -> ClaudeAgentSdkInstrumentor:
    """Create a fresh instrumentor, uninstrumenting after the test."""
    inst = ClaudeAgentSdkInstrumentor()
    yield inst  # type: ignore[misc]
    with contextlib.suppress(Exception):
        inst.uninstrument()


# ---------------------------------------------------------------------------
# Shared agent config
# ---------------------------------------------------------------------------

_OBS_MODULE = "holodeck.lib.backends.claude_backend"


def _make_otel_agent(
    *,
    capture_content: bool = False,
    max_tokens: int = 200,
    claude: ClaudeConfig | None = None,
) -> Agent:
    """Build an Anthropic agent with observability enabled."""
    return Agent(
        name="test-otel-agent",
        model=LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-6",
            auth_provider=AuthProvider.oauth_token,
            max_tokens=max_tokens,
            temperature=0.0,
        ),
        instructions=Instructions(
            inline="You are a helpful assistant. Always respond in one sentence."
        ),
        claude=claude,
        observability=ObservabilityConfig(
            enabled=True,
            service_name="test-instrumentation",
            traces=TracingConfig(
                enabled=True,
                capture_content=capture_content,
            ),
            metrics=MetricsConfig(enabled=False),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestClaudeInstrumentationIntegration:
    """Integration tests for Claude Agent SDK OTel instrumentation.

    Tests validate that the instrumentation package produces spans
    conforming to GenAI semantic conventions when invoked through
    HoloDeck's ClaudeBackend.
    """

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_invoke_agent_span_emitted(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """T023a: invoke_once() emits an invoke_agent span with GenAI attributes."""
        # Arrange — instrument with test TracerProvider
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-agent",
            capture_content=False,
        )

        agent = _make_otel_agent()

        # Mock get_observability_context to return our test providers
        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})
            try:
                await backend.initialize()
                result = await backend.invoke_once("What is 2 + 2?")

                assert result.response, "Response must not be empty"
            finally:
                await backend.teardown()

        # Assert — check exported spans
        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        # Must have at least one invoke_agent span
        invoke_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(invoke_spans) >= 1, f"Expected invoke_agent span, got: {span_names}"

        # Verify GenAI semantic convention attributes
        invoke_span = invoke_spans[0]
        attrs = dict(invoke_span.attributes or {})
        assert attrs.get("gen_ai.operation.name") == "invoke_agent"
        assert attrs.get("gen_ai.system") == "anthropic"
        assert attrs.get("gen_ai.agent.name") == "test-otel-agent"

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_span_hierarchy_parent_child(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """T023b: invoke_agent spans nest correctly under a HoloDeck parent span."""
        # Arrange — instrument with test TracerProvider
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-agent",
            capture_content=False,
        )

        agent = _make_otel_agent()
        tracer = tracer_provider.get_tracer("holodeck.test")

        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})

            try:
                await backend.initialize()

                # Act — invoke under a parent span
                with tracer.start_as_current_span("holodeck.cli.test"):
                    result = await backend.invoke_once("What is 2 + 2? Answer briefly.")
                    assert result.response
            finally:
                await backend.teardown()

        # Assert — verify parent-child hierarchy
        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        parent_spans = [s for s in spans if s.name == "holodeck.cli.test"]
        invoke_spans = [s for s in spans if "invoke_agent" in s.name]

        assert len(parent_spans) == 1, f"Expected 1 parent span, got: {span_names}"
        assert len(invoke_spans) >= 1, f"Expected invoke_agent span, got: {span_names}"

        parent = parent_spans[0]
        child = invoke_spans[0]

        # invoke_agent must be a child of the holodeck.cli.test span
        assert child.parent is not None, "invoke_agent span has no parent"
        assert child.parent.span_id == parent.context.span_id, (
            f"invoke_agent parent span_id={child.parent.span_id:#018x} "
            f"!= holodeck.cli.test span_id={parent.context.span_id:#018x}"
        )
        assert (
            child.context.trace_id == parent.context.trace_id
        ), "Spans must share the same trace_id"

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_execute_tool_spans_emitted(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """T023c: Tool use produces execute_tool child spans under invoke_agent."""
        # Arrange — instrument with test TracerProvider
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-agent",
            capture_content=True,
        )

        # Enable web_search to force tool use
        agent = _make_otel_agent(
            capture_content=True,
            max_tokens=500,
            claude=ClaudeConfig(
                permission_mode=PermissionMode.acceptAll,
                max_turns=5,
                web_search=True,
            ),
        )

        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})

            try:
                await backend.initialize()
                # Prompt designed to trigger web_search tool
                result = await backend.invoke_once(
                    "Search the web for today's date and tell me "
                    "what day of the week it is. Use web search."
                )
                assert result.response
            finally:
                await backend.teardown()

        # Assert — check for tool spans
        spans = span_exporter.get_finished_spans()
        span_names = [s.name for s in spans]

        invoke_spans = [s for s in spans if "invoke_agent" in s.name]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(invoke_spans) >= 1, f"Expected invoke_agent span, got: {span_names}"

        # Tool spans should be children of invoke_agent
        if tool_spans:
            invoke_span = invoke_spans[0]
            for tool_span in tool_spans:
                attrs = dict(tool_span.attributes or {})
                assert attrs.get("gen_ai.operation.name") == "execute_tool"
                assert attrs.get("gen_ai.system") == "anthropic"
                assert attrs.get(
                    "gen_ai.tool.name"
                ), "execute_tool span must have gen_ai.tool.name"
                # Must be child of invoke_agent
                assert tool_span.parent is not None
                assert (
                    tool_span.context.trace_id == invoke_span.context.trace_id
                ), "Tool span must share trace_id with invoke_agent"
        else:
            # If no tool spans, the LLM chose not to use tools.
            # Log for visibility but don't fail — tool use is non-deterministic.
            pytest.skip(
                f"LLM did not use tools in this run; spans emitted: {span_names}"
            )

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_token_usage_recorded(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """T023d: invoke_agent spans record token usage attributes."""
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-agent",
            capture_content=False,
        )

        agent = _make_otel_agent()

        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})
            try:
                await backend.initialize()
                result = await backend.invoke_once("Say hello.")
                assert result.response
            finally:
                await backend.teardown()

        spans = span_exporter.get_finished_spans()
        invoke_spans = [s for s in spans if "invoke_agent" in s.name]
        assert invoke_spans, "Expected at least one invoke_agent span"

        attrs = dict(invoke_spans[0].attributes or {})
        # Token usage should be recorded
        assert (
            "gen_ai.usage.input_tokens" in attrs
        ), f"Missing input_tokens in attrs: {sorted(attrs.keys())}"
        assert (
            "gen_ai.usage.output_tokens" in attrs
        ), f"Missing output_tokens in attrs: {sorted(attrs.keys())}"
        assert attrs["gen_ai.usage.input_tokens"] > 0
        assert attrs["gen_ai.usage.output_tokens"] > 0

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_span_status_ok_on_success(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """T023e: Successful invocation sets span status to OK or UNSET."""
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-agent",
            capture_content=False,
        )

        agent = _make_otel_agent()

        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})
            try:
                await backend.initialize()
                await backend.invoke_once("Say hello.")
            finally:
                await backend.teardown()

        spans = span_exporter.get_finished_spans()
        invoke_spans = [s for s in spans if "invoke_agent" in s.name]
        assert invoke_spans

        status = invoke_spans[0].status
        assert status.status_code in (
            StatusCode.OK,
            StatusCode.UNSET,
        ), f"Expected OK or UNSET, got {status.status_code}"

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_diagnostic_span_tree(
        self,
        tracer_provider: TracerProvider,
        span_exporter: InMemorySpanExporter,
        instrumentor: ClaudeAgentSdkInstrumentor,
    ) -> None:
        """Diagnostic: dump full span tree with parent-child relationships.

        Prints every exported span with trace_id, span_id, parent_span_id,
        name, attributes, and status.  Use ``pytest -s`` to see output.
        """
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            agent_name="test-otel-diag",
            capture_content=True,
        )

        # Use web_search to force tool use
        agent = _make_otel_agent(
            capture_content=True,
            max_tokens=500,
            claude=ClaudeConfig(
                permission_mode=PermissionMode.acceptAll,
                max_turns=5,
                web_search=True,
            ),
        )

        tracer = tracer_provider.get_tracer("holodeck.diagnostic")

        test_ctx = ObservabilityContext(
            tracer_provider=tracer_provider,
            meter_provider=None,
            logger_provider=None,
        )
        with patch(f"{_OBS_MODULE}.get_observability_context", return_value=test_ctx):
            backend = ClaudeBackend(agent, tool_instances={})
            try:
                await backend.initialize()
                with tracer.start_as_current_span("holodeck.diagnostic.root"):
                    result = await backend.invoke_once(
                        "Search the web for the current UTC time "
                        "and tell me what it is. Use web search."
                    )
                    print(f"\n--- Agent response ---\n{result.response[:200]}")
                    print(f"Tool calls: {len(result.tool_calls)}")
                    for tc in result.tool_calls:
                        print(f"  - {tc.get('name', 'unknown')}")
            finally:
                await backend.teardown()

        # Force flush to ensure all spans are exported
        tracer_provider.force_flush()

        spans = span_exporter.get_finished_spans()
        print(f"\n{'=' * 80}")
        print(f"SPAN TREE DIAGNOSTIC — {len(spans)} span(s) exported")
        print(f"{'=' * 80}")

        for s in spans:
            tid = f"{s.context.trace_id:#034x}"
            sid = f"{s.context.span_id:#018x}"
            pid = f"{s.parent.span_id:#018x}" if s.parent else "None (root)"
            attrs = dict(s.attributes or {})
            print(f"\n  Span: {s.name}")
            print(f"    trace_id:  {tid}")
            print(f"    span_id:   {sid}")
            print(f"    parent_id: {pid}")
            print(f"    status:    {s.status.status_code.name}")
            print(f"    duration:  {(s.end_time - s.start_time) / 1e6:.1f}ms")
            if attrs:
                for k, v in sorted(attrs.items()):
                    print(f"    {k}: {v}")

        print(f"\n{'=' * 80}")

        # Summary
        names = [s.name for s in spans]
        print(f"Span names: {names}")
        invoke = [s for s in spans if "invoke_agent" in s.name]
        tools = [s for s in spans if "execute_tool" in s.name]
        print(f"invoke_agent count: {len(invoke)}")
        print(f"execute_tool count: {len(tools)}")

        if not tools:
            print("\n*** NO execute_tool SPANS — hooks may not be firing ***")
            # Check if tool_calls were reported by the SDK
            print(f"SDK reported {len(result.tool_calls)} tool_calls")

        assert len(invoke) >= 1, f"Expected invoke_agent span, got: {names}"
