"""Default hooks are merged into ClaudeAgentOptions by default."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from holodeck.lib.backends.claude_backend import build_options
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import ClaudeConfig
from holodeck.models.llm import LLMProvider, ProviderEnum

_SDK_MODULE = "holodeck.lib.backends.claude_backend"


def _minimal_agent(claude: ClaudeConfig | None = None) -> Agent:
    return Agent(
        name="test",
        description="test",
        model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-6"),
        instructions=Instructions(inline="Be helpful."),
        claude=claude or ClaudeConfig(),
    )


@pytest.mark.unit
@patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
@patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
def test_build_options_includes_default_hooks_by_default(
    mock_resolve: object, mock_opts_cls: object
) -> None:
    """Credential-redaction PostToolUse hook is merged in by default."""
    agent = _minimal_agent()
    build_options(
        agent=agent,
        tool_server=None,
        tool_names=[],
        mcp_configs={},
        auth_env={},
        otel_env={},
        mode="test",
    )

    import unittest.mock as um

    assert isinstance(mock_opts_cls, um.MagicMock)
    kwargs = mock_opts_cls.call_args[1]
    hooks = kwargs.get("hooks")
    assert hooks is not None
    assert "PostToolUse" in hooks
    # Bash deny hook was removed 2026-05-23 — see plan Task 2 note.
    assert "PreToolUse" not in hooks
    post = hooks["PostToolUse"]
    assert len(post) >= 1


@pytest.mark.unit
@patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
@patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
def test_build_options_omits_default_hooks_when_disabled(
    mock_resolve: object, mock_opts_cls: object, caplog: pytest.LogCaptureFixture
) -> None:
    """disable_default_hooks=True drops them and logs a warning."""
    agent = _minimal_agent(ClaudeConfig(disable_default_hooks=True))
    with caplog.at_level(logging.WARNING):
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
        )

    import unittest.mock as um

    assert isinstance(mock_opts_cls, um.MagicMock)
    kwargs = mock_opts_cls.call_args[1]
    hooks = kwargs.get("hooks")
    assert hooks is None or "PostToolUse" not in (hooks or {})
    assert any("disable_default_hooks" in record.message for record in caplog.records)


@pytest.mark.unit
@patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
@patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
def test_build_options_sets_subprocess_env_scrub_by_default(
    mock_resolve: object, mock_opts_cls: object
) -> None:
    """Subprocess env scrub flags are injected into env by default."""
    agent = _minimal_agent()
    build_options(
        agent=agent,
        tool_server=None,
        tool_names=[],
        mcp_configs={},
        auth_env={},
        otel_env={},
        mode="test",
    )

    import unittest.mock as um

    assert isinstance(mock_opts_cls, um.MagicMock)
    kwargs = mock_opts_cls.call_args[1]
    env = kwargs.get("env") or {}
    assert env.get("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB") == "1"
    assert env.get("CLAUDE_CODE_MCP_ALLOWLIST_ENV") == "1"


@pytest.mark.unit
@patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
@patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
def test_build_options_omits_subprocess_env_scrub_when_disabled(
    mock_resolve: object, mock_opts_cls: object, caplog: pytest.LogCaptureFixture
) -> None:
    """disable_subprocess_env_scrub=True drops both flags and logs a warning."""
    agent = _minimal_agent(ClaudeConfig(disable_subprocess_env_scrub=True))
    with caplog.at_level(logging.WARNING):
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
        )

    import unittest.mock as um

    assert isinstance(mock_opts_cls, um.MagicMock)
    kwargs = mock_opts_cls.call_args[1]
    env = kwargs.get("env") or {}
    assert "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB" not in env
    assert "CLAUDE_CODE_MCP_ALLOWLIST_ENV" not in env
    assert any(
        "disable_subprocess_env_scrub" in record.message for record in caplog.records
    )
