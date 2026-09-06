"""Runner-level acceptance for ``openai.fallback_model`` (FR-033, T1 / D14).

Unlike ``test_openai_agents_fallback.py`` (which calls the wrapper directly),
these tests drive the **real** SDK ``Runner`` over real ``OpenAIResponsesModel``
instances whose ``AsyncOpenAI`` clients talk to an in-process ``httpx``
``MockTransport``. No network, no credentials. They pin the bounded order
documented in ``openai_agents_fallback.py``:

* the primary's provider-managed client retries exhaust first (observable as
  N+1 HTTP requests to the primary host), then exactly one fallback attempt;
* after a fallback failure the Runner surfaces the error — no third attempt;
* a streamed run falls back only before its first event and never duplicates
  streamed output; after the first event a failure propagates untouched;
* both attempts are visible in the trace: two ``response`` spans under one
  trace reach the OTel mirror through the HoloDeck trace router.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, RateLimitError
from openai.types.responses import Response, ResponseCreatedEvent
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from holodeck.lib.backends import openai_agents_tracing as tracing_module
from holodeck.lib.backends.openai_agents_fallback import build_fallback_model
from holodeck.lib.backends.openai_agents_tracing import (
    TRACE_POLICY_METADATA_KEY,
    TracingPolicy,
    build_tracing_mirror,
    register_tracing_policy,
)

PRIMARY_HOST = "primary.test"
FALLBACK_HOST = "fallback.test"
POLICY_ID = "runner-fallback-test"

# ---------------------------------------------------------------------------
# Fake Responses API upstream
# ---------------------------------------------------------------------------


def _response_payload(text: str, model: str) -> dict[str, Any]:
    """A minimal, schema-valid Responses API ``Response`` body."""
    raw: dict[str, Any] = {
        "id": "resp_1",
        "created_at": 1.0,
        "model": model,
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return Response.model_validate(raw).model_dump(mode="json")


def _sse(events: list[dict[str, Any]]) -> bytes:
    """Encode Responses stream events as a text/event-stream body."""
    body = ""
    for event in events:
        body += f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
    return body.encode()


def _stream_events(text: str, model: str, deltas: list[str]) -> list[dict[str, Any]]:
    payload = _response_payload(text, model)
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": payload, "sequence_number": 0}
    ]
    for index, delta in enumerate(deltas, start=1):
        events.append(
            {
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": delta,
                "item_id": "msg_1",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": index,
            }
        )
    events.append(
        {
            "type": "response.completed",
            "response": payload,
            "sequence_number": len(deltas) + 1,
        }
    )
    return events


def _status(status_code: int, code: str | None = None) -> httpx.Response:
    """An error response; ``retry-after-ms: 1`` keeps client backoff at 1 ms."""
    error: dict[str, Any] = {"message": "upstream says no", "type": "error"}
    if code is not None:
        error["code"] = code
    return httpx.Response(
        status_code, headers={"retry-after-ms": "1"}, json={"error": error}
    )


def _ok(request: httpx.Request, text: str, model: str) -> httpx.Response:
    """A successful response, streamed (SSE) or not per the request body."""
    if json.loads(request.content or b"{}").get("stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_stream_events(text, model, ["hel", "lo"])),
        )
    return httpx.Response(200, json=_response_payload(text, model))


class FakeUpstream:
    """Per-host scripted Responses API behind one ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.status: dict[str, int | None] = {PRIMARY_HOST: None, FALLBACK_HOST: None}
        self.error_code: dict[str, str | None] = {}
        self.order: list[str] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.calls[host] += 1
        self.order.append(host)
        status = self.status.get(host)
        if status is not None:
            return _status(status, self.error_code.get(host))
        return _ok(request, text=f"hello from {host}", model=f"model@{host}")

    def client(self, host: str, *, max_retries: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="sk-test",
            base_url=f"http://{host}/v1",
            max_retries=max_retries,
            http_client=httpx.AsyncClient(transport=self.transport),
        )


def _wrapped_agent(upstream: FakeUpstream, *, primary_retries: int = 0) -> Any:
    """SDK ``Agent`` whose model is the fallback wrapper over two real models."""
    from agents import Agent, OpenAIResponsesModel

    primary = OpenAIResponsesModel(
        model="primary-model",
        openai_client=upstream.client(PRIMARY_HOST, max_retries=primary_retries),
    )
    fallback = OpenAIResponsesModel(
        model="fallback-model",
        openai_client=upstream.client(FALLBACK_HOST, max_retries=0),
    )
    return Agent(
        name="fallback-agent",
        instructions="Reply briefly.",
        model=build_fallback_model(primary, fallback),
    )


def _run_config() -> Any:
    from agents import RunConfig

    return RunConfig(
        workflow_name="fallback-agent",
        trace_metadata={TRACE_POLICY_METADATA_KEY: POLICY_ID},
        trace_include_sensitive_data=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def otel_mirror(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Route this test's SDK traces through the HoloDeck router into OTel.

    Installs the real router (``upload=False`` so nothing leaves the process)
    with an OTel mirror backed by an in-memory exporter, and restores the SDK's
    processor list afterwards.
    """
    import agents
    from agents.tracing.setup import get_trace_provider

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "holodeck.lib.backends.openai_agents_tracing.get_tracer",
        lambda name: provider.get_tracer(name),
    )
    monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)
    agents.set_tracing_disabled(False)

    # Test-only: snapshot the SDK's current processor tuple so it can be restored.
    provider_impl: Any = get_trace_provider()
    previous = list(provider_impl._multi_processor._processors)
    tracing_module._reset_tracing_router()
    register_tracing_policy(
        POLICY_ID, TracingPolicy(upload=False, mirror=build_tracing_mirror("t"))
    )
    yield exporter
    tracing_module._reset_tracing_router()
    agents.set_trace_processors(previous)


def _response_spans(exporter: InMemorySpanExporter) -> list[Any]:
    return [
        s for s in exporter.get_finished_spans() if s.name == "openai_agents.response"
    ]


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_primary_client_retries_exhaust_then_one_fallback(
    upstream: FakeUpstream, otel_mirror: InMemorySpanExporter
) -> None:
    """429 on the primary: N+1 primary requests, then exactly one fallback."""
    from agents import Runner

    upstream.status[PRIMARY_HOST] = 429
    result = await Runner.run(
        _wrapped_agent(upstream, primary_retries=2), "hi", run_config=_run_config()
    )

    assert result.final_output == f"hello from {FALLBACK_HOST}"
    assert upstream.calls[PRIMARY_HOST] == 3  # 1 + max_retries
    assert upstream.calls[FALLBACK_HOST] == 1

    spans = _response_spans(otel_mirror)
    assert len(spans) == 2, "both attempts must reach the trace"
    assert spans[0].status.status_code is StatusCode.ERROR
    assert spans[1].status.status_code is StatusCode.UNSET
    assert (
        spans[0].attributes["gen_ai.openai.trace_id"]
        == spans[1].attributes["gen_ai.openai.trace_id"]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runner_surfaces_fallback_failure_without_third_attempt(
    upstream: FakeUpstream, otel_mirror: InMemorySpanExporter
) -> None:
    """Primary 503 then fallback 503: the error propagates, no extra attempt."""
    from agents import Runner

    upstream.status[PRIMARY_HOST] = 503
    upstream.status[FALLBACK_HOST] = 503
    with pytest.raises(Exception) as excinfo:
        await Runner.run(_wrapped_agent(upstream), "hi", run_config=_run_config())

    assert getattr(excinfo.value, "status_code", None) == 503
    assert upstream.calls[PRIMARY_HOST] == 1
    assert upstream.calls[FALLBACK_HOST] == 1
    spans = _response_spans(otel_mirror)
    assert [s.status.status_code for s in spans] == [StatusCode.ERROR, StatusCode.ERROR]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_retryable_primary_error_never_reaches_fallback(
    upstream: FakeUpstream, otel_mirror: InMemorySpanExporter
) -> None:
    """401 on the primary propagates; the fallback host sees no request."""
    from agents import Runner

    upstream.status[PRIMARY_HOST] = 401
    with pytest.raises(AuthenticationError):
        await Runner.run(_wrapped_agent(upstream), "hi", run_config=_run_config())

    assert upstream.calls[PRIMARY_HOST] == 1
    assert upstream.calls[FALLBACK_HOST] == 0
    assert len(_response_spans(otel_mirror)) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_locked_compat_retry_is_bounded_to_four_pairs(
    upstream: FakeUpstream,
    otel_mirror: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK ``conversation_locked`` compatibility re-runs the pair at most 3x (D14)."""
    from agents import Runner
    from agents.run_internal import model_retry

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(model_retry, "_sleep_for_retry", _no_sleep)
    upstream.status[PRIMARY_HOST] = 503
    upstream.status[FALLBACK_HOST] = 400
    upstream.error_code[FALLBACK_HOST] = "conversation_locked"

    with pytest.raises(BadRequestError):
        await Runner.run(_wrapped_agent(upstream), "hi", run_config=_run_config())

    assert upstream.order == [PRIMARY_HOST, FALLBACK_HOST] * 4
    assert len(_response_spans(otel_mirror)) == 8


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def _collect_deltas(result: Any) -> tuple[list[str], list[str]]:
    """Drain a streamed run; return (text deltas, raw response event types)."""
    from openai.types.responses import ResponseTextDeltaEvent

    deltas: list[str] = []
    kinds: list[str] = []
    async for event in result.stream_events():
        if event.type != "raw_response_event":
            continue
        kinds.append(str(getattr(event.data, "type", "")))
        if isinstance(event.data, ResponseTextDeltaEvent):
            deltas.append(event.data.delta)
    return deltas, kinds


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streamed_run_falls_back_before_first_event_without_duplicates(
    upstream: FakeUpstream, otel_mirror: InMemorySpanExporter
) -> None:
    """A 429 before any stream event falls back; deltas arrive exactly once."""
    from agents import Runner

    upstream.status[PRIMARY_HOST] = 429
    result = Runner.run_streamed(
        _wrapped_agent(upstream, primary_retries=1), "hi", run_config=_run_config()
    )
    deltas, kinds = await _collect_deltas(result)

    assert deltas == ["hel", "lo"]
    assert kinds.count("response.created") == 1
    assert kinds.count("response.completed") == 1
    assert result.final_output == f"hello from {FALLBACK_HOST}"
    assert upstream.calls[PRIMARY_HOST] == 2
    assert upstream.calls[FALLBACK_HOST] == 1
    spans = _response_spans(otel_mirror)
    assert [s.status.status_code for s in spans] == [StatusCode.ERROR, StatusCode.UNSET]


def _failing_after_first_event_model() -> Any:
    """A ``Model`` whose stream emits ``response.created`` then raises 429."""
    from agents.models.interface import Model

    class _FailsAfterFirstEvent(Model):
        async def get_response(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("non-streaming path not exercised")

        async def stream_response(self, *_a: Any, **_k: Any) -> AsyncIterator[Any]:
            payload = _response_payload("partial", "primary-model")
            yield ResponseCreatedEvent(
                type="response.created",
                response=Response.model_validate(payload),
                sequence_number=0,
            )
            raise RateLimitError(
                "rate limited mid-stream",
                response=httpx.Response(
                    429, request=httpx.Request("POST", f"http://{PRIMARY_HOST}/v1")
                ),
                body=None,
            )

    return _FailsAfterFirstEvent()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streamed_run_does_not_restart_after_first_event(
    upstream: FakeUpstream, otel_mirror: InMemorySpanExporter
) -> None:
    """A retryable failure after ``response.created`` propagates; no fallback."""
    from agents import Agent, OpenAIResponsesModel, Runner

    fallback = OpenAIResponsesModel(
        model="fallback-model",
        openai_client=upstream.client(FALLBACK_HOST, max_retries=0),
    )
    agent = Agent(
        name="fallback-agent",
        instructions="Reply briefly.",
        model=build_fallback_model(_failing_after_first_event_model(), fallback),
    )

    result = Runner.run_streamed(agent, "hi", run_config=_run_config())
    kinds: list[str] = []
    with pytest.raises(RateLimitError):
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                kinds.append(str(getattr(event.data, "type", "")))

    assert kinds.count("response.created") == 1
    assert upstream.calls[FALLBACK_HOST] == 0
