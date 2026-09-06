"""Exporter-level tests for LiteLLM GenAI spans (035 T2, LiteLLM plan Task 6).

These drive the real ``litellm`` entry points with ``mock_response`` (no
network) through the callback registered by ``enable_litellm_telemetry`` and
read the finished spans from an in-memory exporter behind
``RedactingSpanProcessor``. They pin the span shape HoloDeck documents for the
installed LiteLLM version: span names, operation names, token attributes,
content capture on/off, redaction, and the ``OTEL_SEMCONV_STABILITY_OPT_IN``
opt-in shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor
from holodeck.lib.litellm_support import LiteLLMEmbeddingService, LiteLLMModelSpec
from holodeck.lib.llm_context_generator import LLMContextGenerator
from holodeck.lib.observability.instrumentation import enable_litellm_telemetry
from holodeck.models.observability import ObservabilityConfig, TracingConfig

SECRET = "ghp_" + "q" * 36
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

_CALLBACK_LISTS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "input_callback",
)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Isolated SDK provider (redaction + in-memory export) seen as global.

    ``enable_litellm_telemetry`` reuses whatever ``trace.get_tracer_provider``
    returns, so the lookup is patched rather than mutating the process-global
    provider. LiteLLM's global callback lists are snapshotted and restored.
    """
    import litellm

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: provider)
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)

    import litellm.utils as litellm_utils

    saved = {name: list(getattr(litellm, name)) for name in _CALLBACK_LISTS}
    for name in _CALLBACK_LISTS:
        getattr(litellm, name).clear()
    # litellm.utils.function_setup initialises callbacks only while this
    # module-level list is empty; restore it so later tests still initialise.
    saved_callback_list = litellm_utils.callback_list
    litellm_utils.callback_list = []
    yield exporter
    for name in _CALLBACK_LISTS:
        target = getattr(litellm, name)
        target.clear()
        target.extend(saved[name])
    litellm_utils.callback_list = saved_callback_list


def _config(capture_content: bool) -> ObservabilityConfig:
    return ObservabilityConfig(
        enabled=True, traces=TracingConfig(capture_content=capture_content)
    )


async def _wait_for_spans(
    exporter: InMemorySpanExporter, minimum: int = 1, timeout: float = 5.0
) -> list[ReadableSpan]:
    """LiteLLM runs success callbacks on a background task; wait for export."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        spans = list(exporter.get_finished_spans())
        if len(spans) >= minimum:
            return spans
        await asyncio.sleep(0.02)
    return list(exporter.get_finished_spans())


def _request_spans(spans: list[ReadableSpan]) -> list[ReadableSpan]:
    return [s for s in spans if s.name != "raw_gen_ai_request"]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


async def _embed(texts: list[str]) -> list[list[float]]:
    service = LiteLLMEmbeddingService(
        LiteLLMModelSpec(model=EMBED_MODEL, api_key="sk-test")
    )
    import litellm

    original = litellm.aembedding

    async def with_mock(**kwargs: Any) -> Any:
        # litellm's mock_embedding returns exactly one vector per call.
        return await original(mock_response=[0.1, 0.2, 0.3], **kwargs)

    litellm.aembedding = with_mock  # type: ignore[assignment]
    try:
        return await service.generate_embeddings(texts)
    finally:
        litellm.aembedding = original  # type: ignore[assignment]


async def _generate_context(reply: str) -> str:
    """Run LLMContextGenerator's real call path with a mocked provider reply."""
    import litellm

    original = litellm.acompletion

    async def with_mock(**kwargs: Any) -> Any:
        return await original(mock_response=reply, **kwargs)

    generator = LLMContextGenerator(
        LiteLLMModelSpec(model=CHAT_MODEL, api_key="sk-test"),
        max_context_tokens=64,
    )
    litellm.acompletion = with_mock  # type: ignore[assignment]
    try:
        return await generator.generate_context(
            chunk_text=f"token {SECRET}", document_text="doc"
        )
    finally:
        litellm.acompletion = original  # type: ignore[assignment]


class TestEmbeddingSpans:
    @pytest.mark.asyncio
    async def test_embedding_span_default_shape_with_tokens(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_litellm_telemetry(_config(capture_content=False))

        vectors = await _embed(["a"])
        spans = _request_spans(await _wait_for_spans(exporter))

        assert vectors == [[0.1, 0.2, 0.3]]
        assert len(spans) == 1
        span = spans[0]
        attrs = _attrs(span)
        assert span.name == "litellm_request"
        # Legacy shape: the operation always rides on llm.request.type;
        # gen_ai.operation.name is only written alongside captured content.
        assert attrs["llm.request.type"] == "aembedding"
        assert "gen_ai.operation.name" not in attrs
        assert attrs["gen_ai.request.model"] == EMBED_MODEL
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 0
        assert attrs["gen_ai.usage.total_tokens"] == 0
        assert "gen_ai.input.messages" not in attrs

    @pytest.mark.asyncio
    async def test_capture_disabled_emits_no_content_and_no_raw_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_litellm_telemetry(_config(capture_content=False))

        await _embed([f"secret {SECRET}"])
        spans = await _wait_for_spans(exporter)

        assert [s.name for s in spans] == ["litellm_request"]
        serialized = repr([_attrs(s) for s in spans])
        assert SECRET not in serialized
        assert "gen_ai.input.messages" not in serialized

    @pytest.mark.asyncio
    async def test_capture_enabled_content_is_redacted(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_litellm_telemetry(_config(capture_content=True))

        await _embed([f"plain then secret {SECRET}"])
        spans = _request_spans(await _wait_for_spans(exporter))

        attrs = _attrs(spans[0])
        assert attrs["gen_ai.operation.name"] == "aembedding"
        messages = attrs["gen_ai.input.messages"]
        assert "plain then secret" in messages
        assert SECRET not in messages
        assert "[REDACTED" in messages
        assert SECRET not in repr([_attrs(s) for s in spans])

    @pytest.mark.asyncio
    async def test_semconv_opt_in_uses_current_genai_shape(
        self, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        enable_litellm_telemetry(_config(capture_content=False))

        await _embed(["a"])
        spans = _request_spans(await _wait_for_spans(exporter))

        span = spans[0]
        attrs = _attrs(span)
        assert span.name == f"embeddings {EMBED_MODEL}"
        # Operation name is still capture-gated; the span name carries it.
        assert "gen_ai.operation.name" not in attrs
        assert attrs["gen_ai.request.model"] == EMBED_MODEL
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert "gen_ai.provider.name" in attrs
        assert "gen_ai.input.messages" not in attrs

    @pytest.mark.asyncio
    async def test_semconv_opt_in_with_capture_has_no_raw_span(
        self, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under the opt-in LiteLLM skips raw_gen_ai_request; content is redacted."""
        monkeypatch.setenv(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        enable_litellm_telemetry(_config(capture_content=True))

        await _embed([f"secret {SECRET}"])
        spans = await _wait_for_spans(exporter)

        assert [s.name for s in spans] == [f"embeddings {EMBED_MODEL}"]
        attrs = _attrs(spans[0])
        assert attrs["gen_ai.operation.name"] == "embeddings"
        assert "[REDACTED" in attrs["gen_ai.input.messages"]
        assert SECRET not in repr(attrs)


class TestFailureSpans:
    @pytest.mark.asyncio
    async def test_provider_failure_span_is_redacted(
        self, exporter: InMemorySpanExporter
    ) -> None:
        """error.* attributes and the exception event never carry a credential."""
        from holodeck.lib.litellm_support import EmbeddingServiceError

        enable_litellm_telemetry(_config(capture_content=False))
        import litellm

        original = litellm.aembedding

        async def failing(**kwargs: Any) -> Any:
            return await original(mock_response="error", **kwargs)

        service = LiteLLMEmbeddingService(
            LiteLLMModelSpec(model=EMBED_MODEL, api_key="sk-test")
        )
        litellm.aembedding = failing  # type: ignore[assignment]
        try:
            with pytest.raises(EmbeddingServiceError):
                await service.generate_embeddings([f"secret {SECRET}"])
        finally:
            litellm.aembedding = original  # type: ignore[assignment]

        spans = await _wait_for_spans(exporter)
        assert spans, "failure span was not exported"
        serialized = repr(
            [
                (_attrs(s), [(e.name, dict(e.attributes or {})) for e in s.events])
                for s in spans
            ]
        )
        assert SECRET not in serialized


class TestChatSpans:
    @pytest.mark.asyncio
    async def test_context_generator_chat_span_with_tokens(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_litellm_telemetry(_config(capture_content=False))

        context = await _generate_context("This chunk covers auth.")
        spans = _request_spans(await _wait_for_spans(exporter))

        assert context == "This chunk covers auth."
        assert len(spans) == 1
        span = spans[0]
        attrs = _attrs(span)
        assert span.name == "litellm_request"
        assert attrs["llm.request.type"] == "acompletion"
        assert "gen_ai.operation.name" not in attrs
        assert attrs["gen_ai.request.model"] == CHAT_MODEL
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 20
        assert attrs["gen_ai.usage.total_tokens"] == 30
        assert "gen_ai.input.messages" not in attrs
        assert "gen_ai.output.messages" not in attrs
        assert SECRET not in repr(attrs)

    @pytest.mark.asyncio
    async def test_chat_capture_enabled_redacts_prompt_completion_and_raw(
        self, exporter: InMemorySpanExporter
    ) -> None:
        enable_litellm_telemetry(_config(capture_content=True))

        await _generate_context(f"reply with {SECRET}")
        spans = await _wait_for_spans(exporter, minimum=2)

        by_name = {s.name: _attrs(s) for s in spans}
        request = by_name["litellm_request"]
        assert request["gen_ai.operation.name"] == "acompletion"
        assert request["gen_ai.response.finish_reasons"] == '["stop"]'
        assert "gen_ai.input.messages" in request
        assert "gen_ai.output.messages" in request
        assert "[REDACTED" in request["gen_ai.input.messages"]
        assert "[REDACTED" in request["gen_ai.output.messages"]
        raw = by_name["raw_gen_ai_request"]
        assert "llm.openai.stringified_raw_response" in raw
        assert SECRET not in repr(by_name)

    @pytest.mark.asyncio
    async def test_semconv_opt_in_chat_shape(
        self, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
        )
        enable_litellm_telemetry(_config(capture_content=False))

        await _generate_context("ok")
        spans = _request_spans(await _wait_for_spans(exporter))

        span = spans[0]
        attrs = _attrs(span)
        assert span.name == f"chat {CHAT_MODEL}"
        assert "gen_ai.operation.name" not in attrs
        assert attrs["gen_ai.provider.name"] == "openai"
        assert attrs["gen_ai.usage.total_tokens"] == 30
