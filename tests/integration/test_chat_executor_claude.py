"""Integration tests: AgentExecutor → BackendSelector → ClaudeBackend → Claude API.

These tests exercise the real end-to-end chat path with no mocks.
Requires a valid ``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum

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


# ---------------------------------------------------------------------------
# Shared agent config
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    """Build a minimal Anthropic agent config for integration testing."""
    return Agent(
        name="test-claude-chat-executor",
        model=LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-6",
            auth_provider=AuthProvider.oauth_token,
            max_tokens=100,
            temperature=0.0,
        ),
        instructions=Instructions(
            inline="You are a helpful assistant. Always respond in one sentence."
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestChatExecutorClaudeIntegration:
    """End-to-end integration tests for AgentExecutor with ClaudeBackend."""

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_execute_turn_returns_response(self) -> None:
        """Non-streaming turn returns a well-formed AgentResponse."""
        executor = AgentExecutor(_make_agent())
        try:
            response = await executor.execute_turn("What is 2 + 2? Answer in one word.")

            assert isinstance(response, AgentResponse)
            assert response.content, "Response content must not be empty"
            assert "4" in response.content or "four" in response.content.lower()
            assert response.execution_time > 0
            assert response.tool_executions == []
            assert response.tokens_used is not None
            assert response.tokens_used.total_tokens > 0
        finally:
            await executor.shutdown()

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_execute_turn_streaming_yields_chunks(self) -> None:
        """Streaming turn yields at least one string chunk."""
        executor = AgentExecutor(_make_agent())
        try:
            chunks: list[str] = []
            async for chunk in executor.execute_turn_streaming("Say hello."):
                chunks.append(chunk)

            assert len(chunks) >= 1, "Expected at least one streamed chunk"
            assert all(isinstance(c, str) for c in chunks)
            joined = "".join(chunks)
            assert joined, "Joined streaming content must not be empty"
        finally:
            await executor.shutdown()

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_history_tracked_after_turn(self) -> None:
        """History records the user message and assistant reply."""
        executor = AgentExecutor(_make_agent())
        try:
            await executor.execute_turn("What color is the sky?")
            history = executor.get_history()

            assert len(history) == 2
            assert history[0]["role"] == "user"
            assert "sky" in history[0]["content"].lower()
            assert history[1]["role"] == "assistant"
            assert history[1]["content"], "Assistant content must not be empty"
        finally:
            await executor.shutdown()

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Two turns maintain conversational memory."""
        executor = AgentExecutor(_make_agent())
        try:
            await executor.execute_turn("My name is Alice.")
            response = await executor.execute_turn("What is my name?")

            history = executor.get_history()
            assert len(history) == 4

            assert "Alice" in response.content
        finally:
            await executor.shutdown()

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self) -> None:
        """Shutdown nullifies session and backend; second call is idempotent."""
        executor = AgentExecutor(_make_agent())
        try:
            await executor.execute_turn("Hello.")

            assert executor._session is not None
            assert executor._backend is not None

            await executor.shutdown()

            assert executor._session is None
            assert executor._backend is None

            # Second shutdown must not raise
            await executor.shutdown()
        except Exception:
            # Ensure cleanup even if assertions fail
            await executor.shutdown()
            raise
