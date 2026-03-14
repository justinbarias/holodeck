"""Shared fixtures for agent and chat unit tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock

import pytest

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    ExecutionResult,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.chat import ChatConfig
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.token_usage import TokenUsage


@pytest.fixture
def make_agent():
    """Factory fixture that creates minimal Agent instances.

    Returns:
        Callable that creates an Agent with sensible defaults.
    """

    def _make_agent(
        name: str = "test-agent",
        description: str = "Test agent",
        provider: ProviderEnum = ProviderEnum.OPENAI,
        model_name: str = "gpt-4o-mini",
        api_key: str = "test-key",
        instructions: str = "Be helpful.",
    ) -> Agent:
        return Agent(
            name=name,
            description=description,
            model=LLMProvider(
                provider=provider,
                name=model_name,
                api_key=api_key,
            ),
            instructions=Instructions(inline=instructions),
        )

    return _make_agent


@pytest.fixture
def make_config():
    """Factory fixture that creates minimal ChatConfig instances.

    Returns:
        Callable that creates a ChatConfig with sensible defaults.
    """

    def _make_config(
        agent_config_path: Path | None = None,
        verbose: bool = False,
        enable_observability: bool = False,
        max_messages: int = 50,
    ) -> ChatConfig:
        if agent_config_path is None:
            with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                agent_config_path = Path(f.name)

        return ChatConfig(
            agent_config_path=agent_config_path,
            verbose=verbose,
            enable_observability=enable_observability,
            max_messages=max_messages,
        )

    return _make_config


@pytest.fixture
def make_mock_backend():
    """Factory fixture that creates mock AgentBackend/AgentSession pairs.

    Returns:
        Callable that creates a (mock_backend, mock_session) tuple.
    """

    def _make_mock_backend(
        response_text: str = "Hello!",
        tool_calls: list | None = None,
        token_usage: TokenUsage | None = None,
    ) -> tuple[AsyncMock, AsyncMock]:
        if token_usage is None:
            token_usage = TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

        mock_session = AsyncMock(spec=AgentSession)
        mock_session.send.return_value = ExecutionResult(
            response=response_text,
            tool_calls=tool_calls or [],
            token_usage=token_usage,
        )

        async def _stream_chunks(message: str):
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        mock_session.send_streaming = _stream_chunks

        mock_backend = AsyncMock(spec=AgentBackend)
        mock_backend.create_session.return_value = mock_session
        return mock_backend, mock_session

    return _make_mock_backend
