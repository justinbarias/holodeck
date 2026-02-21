"""Startup validators for Claude backend initialization.

These validators are called by ClaudeBackend.initialize() before spawning
the Claude subprocess to surface configuration errors at startup, not at
runtime.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from holodeck.config.env_loader import get_env_var
from holodeck.lib.errors import ConfigError
from holodeck.models.agent import Agent
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import HierarchicalDocumentToolConfig, VectorstoreTool

logger = logging.getLogger(__name__)


def validate_nodejs() -> None:
    """Validate that Node.js is available on PATH.

    Claude Agent SDK requires Node.js to spawn its subprocess.

    Raises:
        ConfigError: If node is not found on PATH.
    """
    if shutil.which("node") is None:
        raise ConfigError(
            "nodejs",
            "Node.js is required to run Claude Agent SDK but was not found on "
            "PATH. Install Node.js from https://nodejs.org/ and ensure it is "
            "on your PATH.",
        )


def validate_credentials(model: LLMProvider) -> dict[str, str]:
    """Validate authentication credentials for the LLM provider.

    Checks that the required environment variables are present for the
    configured auth_provider. Returns a dict of extra environment variables
    to inject into the Claude subprocess for provider-specific auth.

    Args:
        model: LLM provider configuration.

    Returns:
        Dict of environment variables to set for the subprocess.

    Raises:
        ConfigError: If required credentials are absent.
    """
    auth = model.auth_provider or AuthProvider.api_key

    if auth == AuthProvider.api_key:
        key = get_env_var("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it with: export ANTHROPIC_API_KEY=sk-ant-...",
            )
        return {}

    if auth == AuthProvider.oauth_token:
        token = get_env_var("CLAUDE_CODE_OAUTH_TOKEN")
        if not token:
            raise ConfigError(
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN environment variable is not set. "
                "Run `claude setup-token` to authenticate with OAuth.",
            )
        return {}

    if auth == AuthProvider.bedrock:
        return {"CLAUDE_CODE_USE_BEDROCK": "1"}

    if auth == AuthProvider.vertex:
        return {"CLAUDE_CODE_USE_VERTEX": "1"}

    # AuthProvider.foundry (and any future values)
    return {"CLAUDE_CODE_USE_FOUNDRY": "1"}


def validate_embedding_provider(agent: Agent) -> None:
    """Validate embedding provider configuration for vectorstore tools.

    Anthropic does not support generating embeddings, so an external
    embedding_provider must be specified when using vectorstore tools
    with the Anthropic LLM provider.

    Args:
        agent: Agent configuration to validate.

    Raises:
        ConfigError: If embedding configuration is invalid for the provider.
    """
    if not agent.tools:
        return

    has_vectorstore = any(
        isinstance(tool, VectorstoreTool | HierarchicalDocumentToolConfig)
        for tool in agent.tools
    )
    if not has_vectorstore:
        return

    # Anthropic cannot be used as an embedding provider
    if agent.embedding_provider is not None:
        if agent.embedding_provider.provider == ProviderEnum.ANTHROPIC:
            raise ConfigError(
                "embedding_provider",
                "Anthropic cannot generate embeddings. Use a different provider "
                "such as openai or azure_openai for embedding_provider when "
                "using vectorstore tools.",
            )
        return

    # Anthropic LLM + vectorstore tool + no embedding_provider
    if agent.model.provider == ProviderEnum.ANTHROPIC:
        raise ConfigError(
            "embedding_provider",
            "embedding_provider is required when using vectorstore tools with "
            "provider: anthropic. Add an embedding_provider using openai or "
            "azure_openai.",
        )


def validate_tool_filtering(agent: Agent) -> None:
    """Warn if tool_filtering is configured for Anthropic provider.

    Claude Agent SDK manages tool selection natively; tool_filtering is a
    Semantic Kernel feature that is not supported by the Claude backend.

    This validator never raises — it only emits a warning. The
    tool_filtering field is not mutated.

    Args:
        agent: Agent configuration to validate.
    """
    if agent.tool_filtering is None:
        return

    if agent.model.provider == ProviderEnum.ANTHROPIC:
        logger.warning(
            "tool_filtering is configured but will be ignored when using "
            "provider: anthropic with the Claude Agent SDK backend. "
            "Claude manages tool selection natively.",
        )


def validate_working_directory(path: str | None) -> None:
    """Warn if CLAUDE.md in working directory may conflict with agent instructions.

    Detects a CLAUDE.md file that contains a '# CLAUDE.md' header, which
    is the standard format used by Claude Code project instructions. Such
    a file may override or conflict with the agent's configured instructions.

    Args:
        path: Working directory path, or None to skip validation.
    """
    if path is None:
        return

    claude_md = Path(path) / "CLAUDE.md"
    if not claude_md.exists():
        return

    content = claude_md.read_text()
    if "# CLAUDE.md" in content:
        logger.warning(
            "A CLAUDE.md file with a '# CLAUDE.md' header was found in the "
            "working directory '%s'. This may conflict with the agent's system "
            "instructions. Review the file to avoid unexpected behavior.",
            path,
        )


def validate_response_format(response_format: dict[str, Any] | str | None) -> None:
    """Validate response format schema is serializable and accessible.

    Args:
        response_format: Inline schema dict, file path string, or None.

    Raises:
        ConfigError: If the schema is not JSON-serializable or file not found.
    """
    if response_format is None:
        return

    if isinstance(response_format, str):
        schema_path = Path(response_format)
        if not schema_path.exists():
            raise ConfigError(
                "response_format",
                f"response_format file not found: {response_format}",
            )
        try:
            json.loads(schema_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise ConfigError(
                "response_format",
                f"response_format file is not valid JSON: {e}",
            ) from e
        return

    # Must be a dict — verify it is JSON-serializable
    try:
        json.dumps(response_format)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            "response_format",
            f"response_format contains non-JSON-serializable values: {e}",
        ) from e
