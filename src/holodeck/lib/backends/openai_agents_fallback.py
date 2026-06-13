"""Fallback model wrapping for the OpenAI Agents backend (``fallback_model``, FR-033).

Wraps a primary SDK ``Model`` so that, when a request fails with a *retryable*
upstream error, it is re-issued once against a fallback ``Model``. The retryable
set is fixed in v1 (not user-tunable): HTTP 429 (rate limit) and 5xx
(server-side) responses from the OpenAI client — i.e. the openai
``APIStatusError`` family with ``status_code == 429`` or ``500 <= status_code <
600``. Every other error (400, 401, 403, 404, 422, connection/timeout errors,
and any non-OpenAI exception) is non-retryable and propagates unchanged; the
fallback is never consulted.

Retry / fallback ordering
-------------------------
HoloDeck does not set ``ModelSettings.retry`` by default, so the wrapper sees
the first failure directly and engages its single fallback attempt immediately.
When a user *does* enable the SDK's runner-managed retry
(``ModelSettings.retry`` / ``ModelRetrySettings``), the contract is:

    primary retries exhaust first  ->  one fallback attempt.

The runner drives ``ModelSettings.retry`` *around* a single ``get_response`` /
``stream_response`` call, retrying the primary model until its budget is spent;
only the final failure reaches this wrapper, which then makes exactly one
fallback attempt. There is no double-fallback and no fallback mid-retry: the
wrapper itself never retries the fallback.

Streaming semantics
-------------------
On the streaming path the wrapper only falls back when the primary stream fails
*before yielding its first event*. Once any event has been emitted from the
primary stream, a later failure is surfaced unchanged — silently restarting on
the fallback would replay already-delivered deltas and corrupt the stream.

Tracing
-------
Each underlying ``Model`` opens its own ``response_span`` per call
(``OpenAIResponsesModel.get_response`` / ``stream_response`` wrap their body in
``response_span``), so the primary attempt and the fallback attempt each get
their own generation span automatically — both attempts are visible in the
trace with no extra instrumentation here (FR-033).

Every ``import agents`` is performed lazily inside the factory so importing this
module never pulls the SDK (SC-005). The wrapping ``Model`` subclass is defined
inside :func:`build_fallback_model`, which imports the SDK ``Model`` base class
at runtime, mirroring the pattern used in ``openai_agents_cost.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
    from agents.model_settings import ModelSettings
    from agents.models.interface import Model, ModelTracing
    from agents.tool import Tool
    from openai.types.responses.response_prompt_param import ResponsePromptParam

logger = logging.getLogger(__name__)


def is_retryable_error(exc: BaseException) -> bool:
    """Return whether *exc* is a retryable upstream error (429 or 5xx).

    The retryable set is fixed in v1: an openai ``APIStatusError`` whose
    ``status_code`` is 429 (rate limit) or in the 5xx range (server error).
    Every other exception — 400/401/403/404/422, connection/timeout errors, and
    any non-OpenAI exception — is non-retryable and returns ``False``.

    The openai client is imported lazily so this module stays SDK-free at import
    time (SC-005); if the client is somehow unavailable the error is treated as
    non-retryable.

    Args:
        exc: The exception raised by the primary model call.

    Returns:
        ``True`` when the request should be re-issued against the fallback
        model, ``False`` otherwise.
    """
    try:
        from openai import APIStatusError
    except ImportError:  # pragma: no cover - openai is a hard dependency here
        return False

    if not isinstance(exc, APIStatusError):
        return False
    status = int(exc.status_code)
    return status == 429 or 500 <= status < 600


def build_fallback_model(primary: Model, fallback: Model) -> Model:
    """Wrap *primary* in a ``Model`` that falls back to *fallback* on 429/5xx.

    Both arguments are concrete SDK ``Model`` instances (the caller resolves any
    model-name string to a ``Model`` and builds the fallback with the *same*
    credentials/client as the primary, differing only in the model name — for
    Azure the fallback name is a deployment on the same endpoint). The returned
    ``Model`` implements both ``get_response`` and ``stream_response``:

    * ``get_response`` calls the primary; on a retryable error (:func:`
      is_retryable_error`) it makes exactly one fallback attempt and returns its
      result. Non-retryable errors propagate unchanged.
    * ``stream_response`` calls the primary; it only falls back when the primary
      stream fails *before yielding its first event*. After any event has been
      emitted, a later failure propagates unchanged (no silent replay).

    Args:
        primary: The primary SDK ``Model``.
        fallback: The SDK ``Model`` to use on a retryable primary failure.

    Returns:
        An SDK ``Model`` wrapping *primary* with single-attempt fallback to
        *fallback*.
    """
    from agents.models.interface import Model as SDKModel

    class _FallbackModel(SDKModel):
        """``Model`` that re-issues a retryable primary failure to a fallback."""

        def __init__(self, primary: Model, fallback: Model) -> None:
            self._primary = primary
            self._fallback = fallback

        async def get_response(
            self,
            system_instructions: str | None,
            input: str | list[TResponseInputItem],
            model_settings: ModelSettings,
            tools: list[Tool],
            output_schema: AgentOutputSchemaBase | None,
            handoffs: list[Handoff],
            tracing: ModelTracing,
            *,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt: ResponsePromptParam | None,
        ) -> ModelResponse:
            """Call the primary model; fall back once on a retryable error.

            Args:
                system_instructions: System instructions for the model.
                input: The input items in OpenAI Responses format.
                model_settings: The model settings to use.
                tools: The tools available to the model.
                output_schema: The output schema to use, if any.
                handoffs: The handoffs available to the model.
                tracing: Tracing configuration.
                previous_response_id: Previous response id (Responses API).
                conversation_id: Stored conversation id, if any.
                prompt: The prompt config to use, if any.

            Returns:
                The primary response, or the fallback response when the primary
                raised a retryable error.

            Raises:
                Exception: The primary error, unchanged, when it is not
                    retryable.
            """
            try:
                return await self._primary.get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised when non-retryable
                if not is_retryable_error(exc):
                    raise
                logger.warning(
                    "fallback_model: primary model failed with retryable error "
                    "(%s); re-issuing against the fallback model.",
                    type(exc).__name__,
                )
                return await self._fallback.get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )

        async def stream_response(
            self,
            system_instructions: str | None,
            input: str | list[TResponseInputItem],
            model_settings: ModelSettings,
            tools: list[Tool],
            output_schema: AgentOutputSchemaBase | None,
            handoffs: list[Handoff],
            tracing: ModelTracing,
            *,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt: ResponsePromptParam | None,
        ) -> AsyncIterator[TResponseStreamEvent]:
            """Stream the primary model; fall back only before the first event.

            The primary stream is consumed; if it raises *before* yielding any
            event, the request is re-issued against the fallback and its stream
            is forwarded. Once any event has been yielded, a later failure
            propagates unchanged — restarting on the fallback would replay the
            already-delivered events.

            Args:
                system_instructions: System instructions for the model.
                input: The input items in OpenAI Responses format.
                model_settings: The model settings to use.
                tools: The tools available to the model.
                output_schema: The output schema to use, if any.
                handoffs: The handoffs available to the model.
                tracing: Tracing configuration.
                previous_response_id: Previous response id (Responses API).
                conversation_id: Stored conversation id, if any.
                prompt: The prompt config to use, if any.

            Yields:
                Response stream events in OpenAI Responses format.

            Raises:
                Exception: The primary error, unchanged, when it is not
                    retryable or when at least one event was already yielded.
            """
            yielded = False
            try:
                async for event in self._primary.stream_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                ):
                    yielded = True
                    yield event
                return
            except Exception as exc:  # noqa: BLE001 - re-raised when non-retryable
                if yielded or not is_retryable_error(exc):
                    raise
                logger.warning(
                    "fallback_model: primary stream failed with retryable error "
                    "(%s) before any event; re-issuing against the fallback model.",
                    type(exc).__name__,
                )

            async for event in self._fallback.stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                yield event

    return _FallbackModel(primary, fallback)
