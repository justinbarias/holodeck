"""Unit tests for holodeck.lib.backends.openai_agents_fallback.

The fallback ``Model`` wrapper is exercised against mocked SDK ``Model``
instances — no network calls and no real credentials. The `openai-agents` and
`openai` packages are installed (dev extra) so the lazy imports inside
``build_fallback_model`` and ``is_retryable_error`` resolve.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from holodeck.lib.backends.openai_agents_fallback import (
    build_fallback_model,
    is_retryable_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(status_code: int) -> httpx.Response:
    """Build an httpx.Response carrying *status_code* for an APIStatusError."""
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


def _status_error(exc_cls: type, status_code: int) -> Any:
    """Construct an openai APIStatusError subclass with a response."""
    return exc_cls("boom", response=_response(status_code), body=None)


def _get_response_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return positional + keyword args for a get_response / stream_response call."""
    positional = (
        "system instructions",
        "user input",
        MagicMock(name="model_settings"),
        [],  # tools
        None,  # output_schema
        [],  # handoffs
        MagicMock(name="tracing"),
    )
    keyword = {
        "previous_response_id": None,
        "conversation_id": None,
        "prompt": None,
    }
    return positional, keyword


async def _drain(stream: Any) -> list[Any]:
    """Collect every event from an async iterator into a list."""
    return [event async for event in stream]


def _streaming_model(
    events: list[Any] | None = None, raises: BaseException | None = None
):
    """Build a MagicMock Model whose stream yields *events* then maybe raises."""

    async def _gen(*_args: Any, **_kwargs: Any):
        for event in events or []:
            yield event
        if raises is not None:
            raise raises

    model = MagicMock()
    model.stream_response = MagicMock(side_effect=lambda *a, **k: _gen(*a, **k))
    return model


# ---------------------------------------------------------------------------
# is_retryable_error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_retryable_error_true_for_429() -> None:
    """A RateLimitError (429) is retryable."""
    assert is_retryable_error(_status_error(RateLimitError, 429)) is True


@pytest.mark.unit
def test_is_retryable_error_true_for_500() -> None:
    """An InternalServerError (500) is retryable."""
    assert is_retryable_error(_status_error(InternalServerError, 500)) is True


@pytest.mark.unit
def test_is_retryable_error_true_for_503() -> None:
    """A 5xx APIStatusError (503) is retryable."""
    assert is_retryable_error(_status_error(InternalServerError, 503)) is True


@pytest.mark.unit
def test_is_retryable_error_false_for_400() -> None:
    """A BadRequestError (400) is NOT retryable."""
    assert is_retryable_error(_status_error(BadRequestError, 400)) is False


@pytest.mark.unit
def test_is_retryable_error_false_for_401() -> None:
    """An AuthenticationError (401) is NOT retryable."""
    assert is_retryable_error(_status_error(AuthenticationError, 401)) is False


@pytest.mark.unit
def test_is_retryable_error_false_for_404() -> None:
    """A NotFoundError (404) is NOT retryable."""
    assert is_retryable_error(_status_error(NotFoundError, 404)) is False


@pytest.mark.unit
def test_is_retryable_error_false_for_connection_error() -> None:
    """A connection error (no status code) is NOT retryable."""
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    assert is_retryable_error(APIConnectionError(request=req)) is False
    assert is_retryable_error(APITimeoutError(req)) is False


@pytest.mark.unit
def test_is_retryable_error_false_for_plain_exception() -> None:
    """A non-OpenAI exception is NOT retryable."""
    assert is_retryable_error(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# get_response — non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_uses_primary_on_success() -> None:
    """When the primary succeeds, the fallback is never consulted."""
    primary = MagicMock()
    primary.get_response = AsyncMock(return_value="primary-response")
    fallback = MagicMock()
    fallback.get_response = AsyncMock(return_value="fallback-response")

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    result = await model.get_response(*positional, **keyword)

    assert result == "primary-response"
    fallback.get_response.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_falls_back_on_429() -> None:
    """A retryable (429) primary error routes the request to the fallback once."""
    primary = MagicMock()
    primary.get_response = AsyncMock(side_effect=_status_error(RateLimitError, 429))
    fallback = MagicMock()
    fallback.get_response = AsyncMock(return_value="fallback-response")

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    result = await model.get_response(*positional, **keyword)

    assert result == "fallback-response"
    primary.get_response.assert_awaited_once()
    fallback.get_response.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_falls_back_on_500() -> None:
    """A retryable (5xx) primary error routes the request to the fallback once."""
    primary = MagicMock()
    primary.get_response = AsyncMock(
        side_effect=_status_error(InternalServerError, 500)
    )
    fallback = MagicMock()
    fallback.get_response = AsyncMock(return_value="fallback-response")

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    result = await model.get_response(*positional, **keyword)

    assert result == "fallback-response"
    fallback.get_response.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_propagates_non_retryable_error() -> None:
    """A non-retryable (401) primary error propagates; fallback NOT consulted."""
    err = _status_error(AuthenticationError, 401)
    primary = MagicMock()
    primary.get_response = AsyncMock(side_effect=err)
    fallback = MagicMock()
    fallback.get_response = AsyncMock(return_value="fallback-response")

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    with pytest.raises(AuthenticationError):
        await model.get_response(*positional, **keyword)

    fallback.get_response.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_forwards_all_arguments_to_primary() -> None:
    """Every get_response argument is forwarded to the primary unchanged."""
    primary = MagicMock()
    primary.get_response = AsyncMock(return_value="ok")
    fallback = MagicMock()
    fallback.get_response = AsyncMock(return_value="fallback")

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    await model.get_response(*positional, **keyword)

    primary.get_response.assert_awaited_once_with(*positional, **keyword)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_response_does_not_double_fallback() -> None:
    """The fallback is attempted exactly once even if it also raises."""
    primary = MagicMock()
    primary.get_response = AsyncMock(side_effect=_status_error(RateLimitError, 429))
    fallback = MagicMock()
    fallback.get_response = AsyncMock(side_effect=_status_error(RateLimitError, 429))

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    with pytest.raises(RateLimitError):
        await model.get_response(*positional, **keyword)

    fallback.get_response.assert_awaited_once()


# ---------------------------------------------------------------------------
# stream_response — streaming path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_response_uses_primary_on_success() -> None:
    """A clean primary stream forwards events; fallback is never consulted."""
    primary = _streaming_model(events=["a", "b", "c"])
    fallback = _streaming_model(events=["x"])

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    events = await _drain(model.stream_response(*positional, **keyword))

    assert events == ["a", "b", "c"]
    fallback.stream_response.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_response_falls_back_before_first_event() -> None:
    """A primary stream that fails (429) before any event falls back."""
    primary = _streaming_model(events=[], raises=_status_error(RateLimitError, 429))
    fallback = _streaming_model(events=["x", "y"])

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()
    events = await _drain(model.stream_response(*positional, **keyword))

    assert events == ["x", "y"]
    fallback.stream_response.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_response_does_not_restart_after_first_event() -> None:
    """A mid-stream primary failure (after events) propagates; no fallback."""
    primary = _streaming_model(
        events=["a", "b"], raises=_status_error(RateLimitError, 429)
    )
    fallback = _streaming_model(events=["x"])

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()

    collected: list[Any] = []
    with pytest.raises(RateLimitError):
        async for event in model.stream_response(*positional, **keyword):
            collected.append(event)

    assert collected == ["a", "b"]
    fallback.stream_response.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_response_propagates_non_retryable_before_first_event() -> None:
    """A non-retryable (400) primary stream failure propagates; no fallback."""
    primary = _streaming_model(events=[], raises=_status_error(BadRequestError, 400))
    fallback = _streaming_model(events=["x"])

    model = build_fallback_model(primary, fallback)
    positional, keyword = _get_response_args()

    with pytest.raises(BadRequestError):
        await _drain(model.stream_response(*positional, **keyword))

    fallback.stream_response.assert_not_called()
